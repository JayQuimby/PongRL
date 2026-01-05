from tinygrad import Tensor, TinyJit, dtypes
from tinygrad.nn import Conv2d, BatchNorm, Linear, optim
from tinygrad.nn.state import safe_save, safe_load, get_state_dict, load_state_dict, get_parameters
from tqdm import tqdm
import numpy as np
import random
import os
from game.static import *
# from concurrent.futures import ThreadPoolExecutor
from agents.memory import AgentMemory

# Neural Network blocks to compose with

def conv_block(inp, outp, ks:int|tuple=3, s:int=1, p:int|tuple=0, act=Tensor.relu, p_ks=(2,2), p_s:int=None, p_pad:int|tuple=0):
    return [
        Conv2d(inp, outp, ks, stride=s, padding=p), act,
        lambda x: x.dropout(DROPOUT_RATE), BatchNorm(outp), 
        lambda x: x.max_pool2d(p_ks, p_s, padding=p_pad),
    ]

def lin_block(inp, outp, act=Tensor.relu): return [Linear(inp, outp), act, lambda x: x.dropout(DROPOUT_RATE)]

# input dims: (N, 280, 120, 3)
# idea: merge 2 or 3 together | (N, 280, 360, 9)


# class for convienience of inverence and saving / loading
class Model:
    def __init__(self):
        self.layers = [
            *conv_block(3, 16, (3,3), p=1, p_ks=(2,2)),          # 280 x 120 x 3 -> 140 x 60 x n
            *conv_block(16, 64, (3,3), p=(0,1), p_ks=(3,3)),     # 70 x 60 x n -> 46 x 20 x 2n
            *conv_block(64, 128, (3,3), p=(0,1), p_ks=(2,2)),    # 46 x 20 x 2n -> 22 x 10 x 4n
            *conv_block(128, 256, (3,3), p=(0,1), p_ks=(4,2)),   # 22 x 10 x 4n -> 5 x 5 x 8n
            *conv_block(256, 512, (3,3), p_ks=(3,3)),            # 5 x 5 x 8n -> 1 x 1 x 16n
            lambda x: x.flatten(1),
            # *lin_block(512, 64),
            Linear(512, 5), 
            # Tensor.dropout,
            # Tensor.softmax,
        ]
        self.opt = optim.AdamW(get_parameters(self), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    def __call__(self, x: Tensor) -> Tensor: 
        return x.sequential(self.layers)
    
    def save(self, filename='models/model.tiny'):
        state_dict = get_state_dict(self)
        safe_save(state_dict, filename)

    def load(self, filename='models/model.tiny'):
        state_dict = safe_load(filename)
        load_state_dict(self, state_dict)
        return self

    def clone(self):
        new = Model()
        self.save('models/tmp.tnsr')
        new.load('models/tmp.tnsr')
        import os
        os.remove('models/tmp.tnsr')
        return new

# def sparse_categorical_crossentropy(self, Y, ignore_index=-1) -> Tensor:
#     loss_mask = Y != ignore_index
#     y_counter = Tensor.arange(self.shape[-1], dtype=dtypes.int32, requires_grad=False, device=self.device).unsqueeze(0).expand(Y.numel(), self.shape[-1])
#     y = ((y_counter == Y.flatten().reshape(-1, 1)).where(-1.0, 0) * loss_mask.reshape(-1, 1)).reshape(*Y.shape, self.shape[-1])
#     return self.log_softmax().mul(y).sum() / loss_mask.sum()

def huber_loss(X, Y, delta=1.0) -> Tensor:
    """Huber loss implementation for tinygrad."""
    diff = X - Y
    abs_diff = diff.abs()
    quadratic_mask = abs_diff <= delta
    quadratic_loss = 0.5 * diff * diff
    linear_loss = delta * (abs_diff - 0.5 * delta)
    loss = quadratic_mask.where(quadratic_loss, linear_loss)
    return loss.mean()

@TinyJit
@Tensor.train()
def train_step(x: Tensor, y: Tensor, model: Model) -> Tensor:
    model.opt.zero_grad()
    loss = huber_loss(model(x), y).backward()
    return loss.realize(*model.opt.schedule_step())

@TinyJit
@Tensor.train()
def train_step_batch(x: Tensor, y: Tensor, model: Model, **cfg) -> Tensor:
    # model.opt.zero_grad()
    total_loss = Tensor(0.0)
    
    for i in tqdm(range(cfg['num_batch']), desc="Batch", leave=False):
        mask = Tensor.arange(cfg['batch_size']*i, cfg['batch_size']*(i+1))
        loss = train_step(x[mask], y[mask], model)
        # loss = huber_loss(model(x[mask]), y[mask]).backward()
        # loss.realize(*model.opt.schedule_step())
        total_loss += loss

    total_loss = total_loss / cfg['num_batch']
    # total_loss.backward()
    return total_loss#.realize(*model.opt.schedule_step())


class PongAgent:
    def __init__(self, train, max_games):
        self.train = train
        self.memory = AgentMemory()
        self.epsilon = BASE_EPSILON if train else 0.0
        if train:
            self.decay = MIN_EPSILON**(self.epsilon/max_games)

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
        act = [0,0,0,0,0]
        if random.random() < self.epsilon:
            act[random.randint(0, 4)] = 1
        else:
            inp = Tensor(state, dtype=dtypes.half).reshape(-1, 3, 280, 120)
            act[self._optimized_predict(inp)] = 1
        return act

    @TinyJit
    def _optimized_predict(self, state):
        return self.model(state).realize().argmax().numpy()

    def _process_batch(self, batch, **cfg):
        states, actions, rewards, next_states, dones = zip(*batch)
        states = Tensor(np.vstack(states), dtype=dtypes.half).reshape(-1, 3, 280, 120)
        next_states = Tensor(np.vstack(next_states), dtype=dtypes.half).reshape(-1, 3, 280, 120)
        rewards = Tensor(np.vstack(rewards), dtype=dtypes.half)
        dones = Tensor(np.vstack(dones), dtype=dtypes.int16)
        actions = Tensor(np.vstack(actions), dtype=dtypes.int16)

        minibatch_size = actions.shape[0]

        # Get Q-values for current states and next_states
        target = self.model(states) # Nx280x120x3 -> Nx5
        q_val = self.target_model(next_states) # Nx280x120x3 -> Nx5
        
        # This is the Q-Value update step where we are approximating the current reward and future rewards
        # Rewards is given by the state of the board at next_state
        # The discount factor increases as time goes on so the values learned over time get more important
        # The Q-Value gets compounded more into the future as epsilon decreases
        # GG - if the next_state was a game over there is no q_val for the future
        #       |  R  | + |Dis| * |       Q Val       | * |         GG        |
        # q_val = rewards + GAMMA * np.max(q_val, axis=1) * (1 - np.array(dones))
        q_val = rewards + GAMMA * q_val.max(axis=1).reshape(-1, 1) * (1 - dones)
        assert q_val.shape == (minibatch_size, 1), f"{q_val.shape} should be {(minibatch_size, 1)}"

        # vectorized lazy q-value update method
        mask = actions.argmax(1).one_hot(5)
        assert mask.shape == (states.shape[0], 5), f"Mask shape was improper for mask: {mask.shape}"

        pre_dim = target.shape
        target += mask * (1-Q_VAL_RATIO) * target + mask * Q_VAL_RATIO * q_val
        # target = (1-Q_VAL_RATIO) * mask * target + Q_VAL_RATIO * q_val
        assert pre_dim == target.shape, f"target shape mismatch - expected: {pre_dim}, got: {target.shape}"

        # train on minibatch
        assert target.shape[0] == states.shape[0], 'batch dims dont match'
        assert target.shape == (minibatch_size, 5), f'target shape improper {target.shape}'
        assert states.shape == (minibatch_size, 3, 280, 120), f'states shape improper {states.shape}'
        return train_step_batch(states, target, self.model, **cfg)

    def replay(self, mems=2048):
        if len(self.memory) < mems:
            print('not enough memories to train yet')
            return
        else:
            print(f'Trianing with {mems} samples')
        params = {
            "num_batch": mems // BATCH_SIZE,
            "batch_size": BATCH_SIZE
        }
        # num_batches =   # Number of batches
        # num_sub_batches = 4  # Adjust based on your needs
        # sub_batch_size = BATCH_SIZE // num_sub_batches
        minibatch = self.memory(mems)
        loss = self._process_batch(minibatch, **params)
        print(loss.numpy())
        #print(f'Training on {num_memories} memories.')
        # losses = []
        # for _ in range(num_batches):
        #     minibatch = self.memory(percent) 
        #     total_loss = 0
        #     for i in range(0, BATCH_SIZE, sub_batch_size):
        #         sub_batch = minibatch[i:i+sub_batch_size]
        #         total_loss += self._process_batch(sub_batch)
        #     tl = total_loss.numpy()
        #     print(tl)
        #     losses.append(tl/(BATCH_SIZE/sub_batch_size))
        # self.stats['train_loss'].append(sum(losses)/len(losses))
        self.stats['train_loss'].append(loss.numpy().mean())
        # print(self.stats['train_loss'])

    def remember(self, state, action, reward, next_state, done):
        self.memory.remember((state, action, reward, next_state, done))

    def apply_decay(self):
        if self.train and self.epsilon > MIN_EPSILON:
            self.epsilon *= self.decay

    def step_itter(self):
        self.step = (self.step + 1) % MODEL_SAMPLE_RATE
        return self.step

    def reset(self):
        #self.memory.clear()
        self.step = 0

    def load(self, name):
        self.model.load(name)

    def save(self, name):
        self.model.save(name)

    def update_target(self):
        # self.model = self.target_model.clone()
        self.target_model = self.model.clone()
        # self.target_model.set_weights(self.model.get_weights())

