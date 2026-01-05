from tinygrad import Tensor, TinyJit, dtypes
from tinygrad.nn import Conv2d, BatchNorm, Linear, optim
from tinygrad.nn.state import safe_save, safe_load, get_state_dict, load_state_dict, get_parameters
from tqdm import tqdm
import numpy as np
import random
import os
from game.static import *
from agents.memory import AgentMemory
from math import sin
# Neural Network blocks to compose with

def conv_block(inp, outp, ks:int|tuple=3, s:int=1, p:int|tuple=0, act=Tensor.relu, p_ks=(2,2), p_s:int=None, p_pad:int|tuple=0):
    return [
        Conv2d(inp, outp, ks, stride=s, padding=p), act,
        lambda x: x.dropout(DROPOUT_RATE), BatchNorm(outp), 
        lambda x: x.max_pool2d(p_ks, p_s, padding=p_pad),
    ]

def lin_block(inp, outp, act=Tensor.relu): return [Linear(inp, outp), act, lambda x: x.dropout(DROPOUT_RATE)]

# class for convenience of inference and saving / loading
class Model:
    def __init__(self):
        self.layers = [
            *conv_block(3, 16, (3,3), p=1, p_ks=(2,2)),          # 280 x 120 x 3 -> 140 x 60 x 16
            *conv_block(16, 64, (3,3), p=(0,1), p_ks=(3,3)),     # 140 x 60 x 16 -> 46 x 20 x 64
            *conv_block(64, 128, (3,3), p=(0,1), p_ks=(2,2)),    # 46 x 20 x 64 -> 22 x 10 x 128
            *conv_block(128, 256, (3,3), p=(0,1), p_ks=(4,2)),   # 22 x 10 x 128 -> 5 x 5 x 256
            *conv_block(256, 512, (3,3), p_ks=(3,3)),            # 5 x 5 x 256 -> 1 x 1 x 512
            lambda x: x.flatten(1),
            Linear(512, 5), 
        ]
        self.opt = optim.AdamW(get_parameters(self), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    def __call__(self, x: Tensor) -> Tensor: 
        return x.sequential(self.layers)
    
    def save(self, filename='models/model.tiny'):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        state_dict = get_state_dict(self)
        safe_save(state_dict, filename)

    def load(self, filename='models/model.tiny'):
        state_dict = safe_load(filename)
        load_state_dict(self, state_dict)
        return self

    def clone(self):
        new = Model()
        temp_file = f'models/tmp_{random.randint(0, 999999)}.tiny'
        self.save(temp_file)
        new.load(temp_file)
        if os.path.exists(temp_file):
            os.remove(temp_file)
        return new

def huber_loss(x, y, delta=1.0) -> Tensor:
    """Huber loss implementation for tinygrad."""
    diff = x - y
    abs_diff = diff.abs()
    quadratic_mask = abs_diff <= delta
    quadratic_loss = 0.5 * diff * diff
    linear_loss = delta * (abs_diff - 0.5 * delta)
    loss = quadratic_mask.where(quadratic_loss, linear_loss)
    return loss.mean()

@TinyJit
def train_step(x: Tensor, y: Tensor, model: Model) -> Tensor:
    with Tensor.train():
        model.opt.zero_grad()
        output = model(x)
        loss = huber_loss(output, y)
        loss.backward()
        model.opt.step()
        return loss.realize()

class PongAgent:
    def __init__(self, train, max_games):
        self.train = train
        self.memory = AgentMemory()
        self.epsilon = BASE_EPSILON if train else 0.0
        if train:
            self.decay = MIN_EPSILON ** (1.0 / max_games)

        self.model = Model()
        self.target_model = self.model.clone()
        self.stats = {
            'train_loss': [],
            'win_rate': []
        }
        
        if os.path.exists(LOAD_PATH):
            self.load(LOAD_PATH)
        self.step = 0    

    def __call__(self, state):
        self.step_itter()
        act = [0, 0, 0, 0, 0]
        
        if random.random() < self.epsilon:
            act[random.randint(0, 4)] = 1
        else:
            # Input should be (1, 3, 280, 120) for tinygrad
            inp = Tensor(state, dtype=dtypes.half).reshape(1, 3, 280, 120)
            action_idx = self._optimized_predict(inp)
            act[action_idx] = 1
        
        return act

    @TinyJit
    def _optimized_predict(self, state):
        output = self.model(state)
        return output.realize().argmax().numpy()

    def _process_batch(self, batch, batch_size):
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors - ensure proper shape (N, 3, 280, 120)
        train_states = Tensor(np.array(states), requires_grad=False, dtype=dtypes.half).reshape(-1, 3, 280, 120)
        states = Tensor(np.array(states), requires_grad=True, dtype=dtypes.half).reshape(-1, 3, 280, 120)
        next_states = Tensor(np.array(next_states), requires_grad=False, dtype=dtypes.half).reshape(-1, 3, 280, 120)
        rewards = Tensor(np.array(rewards), dtype=dtypes.half).reshape(-1, 1)
        dones = Tensor(np.array(dones), dtype=dtypes.half).reshape(-1, 1)
        actions = Tensor(np.array(actions), dtype=dtypes.int32)
        
        # Get Q-values for current states from the main model
        current_q = self.model(train_states)  # (N, 5)
        
        # Get Q-values for next states using target network (no gradients needed)
        next_q = self.target_model(next_states).detach()  # (N, 5)
        
        # Calculate target Q-values using Bellman equation
        # target = reward + gamma * max(Q(s', a')) * (1 - done)
        max_next_q = next_q.max(axis=1, keepdim=True)  # (N, 1)
        target_q = rewards + GAMMA * max_next_q * (1 - dones)  # (N, 1)
        
        # Get the action indices that were taken
        action_indices = actions.argmax(axis=1)  # (N,)
        
        # Create the updated Q-values
        # Start with current Q-values and update only the taken actions
        updated_q = current_q.detach()
        
        # Create one-hot mask for the actions taken
        action_mask = action_indices.one_hot(5)  # (N, 5)
        
        # Update Q-values using Q-learning update rule:
        # Q(s,a) = (1-α) * Q(s,a) + α * target
        # Where α is Q_VAL_RATIO
        current_q_selected = (current_q * action_mask).sum(axis=1, keepdim=True)  # (N, 1)
        updated_q_selected = (1 - Q_VAL_RATIO) * current_q_selected + Q_VAL_RATIO * target_q  # (N, 1)
        
        # Broadcast the updated values back to the full Q-value tensor
        updated_q = current_q * (1 - action_mask) + updated_q_selected * action_mask
        
        # Train on the batch
        loss = train_step(states, updated_q, self.model)
        return loss.numpy()

    def replay(self, mems=2048):
        if len(self.memory) < mems:
            print('Not enough memories to train yet')
            return
        else:
            print(f'Training with {mems} samples')
        
        num_batches = mems // BATCH_SIZE
        minibatch = self.memory(mems)
        
        total_loss = 0.0
        for i in tqdm(range(num_batches), desc="Training batches"):
            batch_start = i * BATCH_SIZE
            batch_end = (i + 1) * BATCH_SIZE
            batch = minibatch[batch_start:batch_end]
            loss = self._process_batch(batch, BATCH_SIZE)
            total_loss += loss
        
        avg_loss = total_loss / num_batches
        self.stats['train_loss'].append(avg_loss)
        print(f"Average loss: {avg_loss:.4f}")

    def remember(self, state, action, reward, next_state, done):
        self.memory.remember((state, action, reward, next_state, done))

    def apply_decay(self):
        # if self.train and self.epsilon > MIN_EPSILON:
        #     self.epsilon *= self.decay
        self.epsilon = 1 + sin(0.1*len(self.stats['win_rate'])) * 0.25

    def step_itter(self):
        self.step = (self.step + 1) % MODEL_SAMPLE_RATE
        return self.step

    def reset(self):
        self.step = 0

    def load(self, name):
        self.model.load(name)
        self.target_model = self.model.clone()

    def save(self, name):
        self.model.save(name)

    def update_target(self):
        """Update target network with current model weights"""
        self.target_model = self.model.clone()