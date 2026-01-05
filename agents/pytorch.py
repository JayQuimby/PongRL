import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import random
import os
from game.static import *
from agents.memory import AgentMemory
from math import cos

# Neural Network blocks to compose with
def conv_block(inp, outp, ks=3, s=1, p=0, p_ks=(2,2), p_s=None, p_pad=0, dropout=0.5):
    return nn.Sequential(
        nn.Conv2d(inp, outp, ks, stride=s, padding=p),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.BatchNorm2d(outp),
        nn.MaxPool2d(p_ks, stride=p_s, padding=p_pad),
    )

class Model(nn.Module):
    def __init__(self, dropout_rate=DROPOUT_RATE, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY):
        super().__init__()
        self.net = nn.Sequential(
            conv_block(3, 16, 3, p=1, p_ks=(2,2), dropout=dropout_rate),
            conv_block(16, 64, 3, p=(0,1), p_ks=(3,3), dropout=dropout_rate),
            conv_block(64, 128, 3, p=(0,1), p_ks=(2,2), dropout=dropout_rate),
            conv_block(128, 256, 3, p=(0,1), p_ks=(4,2), dropout=dropout_rate),
            conv_block(256, 512, 3, p_ks=(3,3), dropout=dropout_rate),
            nn.Flatten(1),
            nn.Linear(512, 5),
        )
        self.opt = optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, x):
        return self.net(x)
    
    def save(self, filename='models/model.pth'):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.opt.state_dict(),
        }, filename)

    def load(self, filename='models/model.pth'):
        checkpoint = torch.load(filename, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.opt.load_state_dict(checkpoint['optimizer_state_dict'])
        return self

    def clone(self):
        new = Model()
        new.load_state_dict(self.state_dict())
        new.to(self.device)
        return new

def huber_loss(x, y, delta=1.0):
    """Huber loss implementation for PyTorch."""
    diff = x - y
    abs_diff = torch.abs(diff)
    quadratic_mask = abs_diff <= delta
    quadratic_loss = 0.5 * diff * diff
    linear_loss = delta * (abs_diff - 0.5 * delta)
    loss = torch.where(quadratic_mask, quadratic_loss, linear_loss)
    return loss.mean()

def train_step(x, y, model):
    model.opt.zero_grad()
    output = model(x)
    loss = huber_loss(output, y)
    loss.backward()
    model.opt.step()
    return loss.item()

class PongAgent:
    def __init__(self, train, max_games):
        self.train = train
        self.memory = AgentMemory()
        self.epsilon = BASE_EPSILON if train else 0.0
        if train:
            self.decay = MIN_EPSILON ** (1.0 / max_games)

        self.model = Model()
        self.target_model = self.model.clone()
        self.target_model.eval()  # Target model always in eval mode
        
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
            with torch.no_grad():
                # Input shape should be (1, 3, 280, 120) - channels first for PyTorch
                # print(state.shape)
                inp = torch.from_numpy(state.copy()).permute(0,3,1,2).float().to(self.model.device)
                # inp = inp  # Convert HWC to CHW if needed
                # inp = inp.unsqueeze(0)  # Add batch dimension
                
                output = self.model(inp)
                action_idx = output.argmax().item()
                act[action_idx] = 1
        
        return act

    # def _process_batch(self, batch):
    #     states, actions, rewards, next_states, dones = zip(*batch)
        
    #     states = torch.from_numpy(np.vstack(states)).float().to(self.model.device)
    #     next_states = torch.from_numpy(np.vstack(next_states)).float().to(self.model.device)

    #     if len(states.shape) == 5:
    #         states = states[0]
    #         next_states = next_states[0]
    #     # Ensure states are in correct shape: (N, 3, 280, 120)
    #     if states.shape[1] != 3:  # If channels are last
    #         states = states.permute(0, 3, 1, 2)
    #         next_states = next_states.permute(0, 3, 1, 2)

    #     # Get Q-values for current states
    #     with torch.no_grad():
    #         self.model.eval()
    #         target = self.model(states)  # (N, 5)

    #         self.target_model.eval()
    #         q_val = self.target_model(next_states)  # (N, 5)

    #     target = target.detach().numpy()
    #     q_val = q_val.detach().numpy()

    #     q_val = np.array(rewards) + GAMMA * q_val.max(axis=1) * (1- np.array(dones))
        
    #     actions = np.argmax(actions, axis=1)

    #     target[:,actions] *= (1-Q_VAL_RATIO)
    #     target[:,actions] += (Q_VAL_RATIO * q_val.reshape(-1, 1))
        
    #     target = torch.from_numpy(target).float().to(self.model.device)
    #     # Train on the batch
    #     loss = train_step(states, target, self.model)
    #     return loss

    # def replay(self, mems=2048):
    #     if len(self.memory) < mems:
    #         print('Not enough memories to train yet')
    #         return
    #     else:
    #         print(f'Training with {mems} samples')
        
    #     num_batches = mems // BATCH_SIZE
    #     minibatch = self.memory(mems)
        
    #     total_loss = 0.0
    #     for i in tqdm(range(num_batches), desc="Training batches"):
    #         batch_start = i * BATCH_SIZE
    #         batch_end = (i + 1) * BATCH_SIZE
    #         batch = minibatch[batch_start:batch_end]
    #         loss = self._process_batch(batch)
    #         total_loss += loss
        
    #     avg_loss = total_loss / num_batches
    #     self.stats['train_loss'].append(avg_loss)
    #     print(f"Average loss: {avg_loss:.4f}")

    def replay(self, mems=2048):
        if len(self.memory) < mems:
            print('Not enough memories to train yet')
            return
        else:
            print(f'Training with {mems} samples')
        
        # Get all samples at once
        minibatch = self.memory(mems)
        
        # Preprocess ALL data at once
        states, actions, rewards, next_states, dones = zip(*minibatch)
        
        states = torch.from_numpy(np.vstack(states)).float().to(self.model.device)
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(self.model.device)
        actions = torch.from_numpy(np.vstack(actions)).float().to(self.model.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.model.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.model.device)
        
        # Handle shape corrections
        if len(states.shape) == 5:
            states = states[0]
            next_states = next_states[0]
        
        if states.shape[1] != 3:
            states = states.permute(0, 3, 1, 2)
            next_states = next_states.permute(0, 3, 1, 2)
        
        # Precompute ALL targets at once
        with torch.no_grad():
            self.model.eval()
            all_targets = self.model(states)  # (mems, 5)
            
            self.target_model.eval()
            q_next = self.target_model(next_states)  # (mems, 5)
            
            # Bellman update for all samples
            q_val = rewards + GAMMA * q_next.max(dim=1)[0] * (1 - dones)
            
            # Get action indices
            action_indices = torch.argmax(actions, dim=1)
            
            # Update all target values
            all_targets = all_targets.clone()
            all_targets[:, action_indices] *= (1 - Q_VAL_RATIO)
            all_targets[:, action_indices] += (Q_VAL_RATIO * q_val)
        
        # Now train on batches with precomputed targets
        num_batches = mems // BATCH_SIZE
        total_loss = 0.0
        
        self.model.train()
        for i in tqdm(range(num_batches), desc="Training batches"):
            batch_start = i * BATCH_SIZE
            batch_end = (i + 1) * BATCH_SIZE
            
            # Slice the precomputed tensors
            batch_states = states[batch_start:batch_end]
            batch_targets = all_targets[batch_start:batch_end]
            
            loss = train_step(batch_states, batch_targets, self.model)
            total_loss += loss
        
        avg_loss = total_loss / num_batches
        self.stats['train_loss'].append(avg_loss)
        print(f"Average loss: {avg_loss:.4f}")

    def remember(self, state, action, reward, next_state, done):
        self.memory.remember((state, action, reward, next_state, done))

    def apply_decay(self):
        # if self.train and self.epsilon > MIN_EPSILON:
        #     self.epsilon *= self.decay
        self.epsilon = 0.85 + cos(0.05 * len(self.stats['train_loss'])) * 0.1
        # print(self.epsilon)


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
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()