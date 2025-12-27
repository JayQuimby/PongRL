import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
from static import *
from concurrent.futures import ThreadPoolExecutor
from memory import AgentMemory
import copy

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Enable optimizations
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    # Enable TF32 for better performance on Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


class DQNModel(nn.Module):
    def __init__(self):
        super(DQNModel, self).__init__()
        kernels = 32
        # Activation
        self.relu = nn.ReLU()
        # First 3D Conv block
        # Input shape: (batch, channels, depth, height, width) = (batch, 1, 3, 120, 280)
        # For 'same' padding with stride s and kernel k: pad = (k - 1) // 2 when stride > 1
        # But for strided conv, we need: pad = ((out_size - 1) * stride + k - in_size) / 2
        # For 'same' effect: pad = (k - 1) // 2 works when we want output = ceil(input / stride)
        self.conv1 = nn.Conv3d(1, kernels, kernel_size=(3, 9, 17), stride=(1, 3, 7), padding=(1, 4, 8))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.dropout1 = nn.Dropout3d(DROPOUT_RATE)
        
        # Second conv block
        self.conv2 = nn.Conv3d(kernels, kernels, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3))
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.dropout2 = nn.Dropout3d(DROPOUT_RATE)
        
        # Third conv block
        self.conv3 = nn.Conv3d(kernels, kernels, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.dropout3 = nn.Dropout3d(DROPOUT_RATE)
        
        # Calculate flattened size
        self._to_linear = None
        # self._get_conv_output_size()
        
        # Dense layers
        self.fc1 = nn.Linear(self._to_linear, 256)
        self.dropout_fc1 = nn.Dropout(DROPOUT_RATE)
        
        self.fc2 = nn.Linear(256, 128)
        self.dropout_fc2 = nn.Dropout(DROPOUT_RATE)
        
        self.fc3 = nn.Linear(128, 64)
        self.dropout_fc3 = nn.Dropout(DROPOUT_RATE)
        
        # Output layer
        self.output = nn.Linear(64, ACTION_SIZE)
            
    def _forward_conv(self, x):
        """Forward pass through conv layers only"""
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        
        return x
    
    def forward(self, x):
        # Conv blocks
        x = self._forward_conv(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Dense layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout_fc1(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout_fc2(x)
        
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout_fc3(x)
        
        # Output (linear activation)
        x = self.output(x)
        
        return x


class PongAgent:
    def __init__(self, train, max_games):
        self.train = train
        self.memory = AgentMemory()
        self.epsilon = BASE_EPSILON if train else 0.0
        self.device = device
        
        if train:
            self.decay = MIN_EPSILON**(self.epsilon/max_games)
            self.thread_pool = ThreadPoolExecutor(max_workers=os.cpu_count())

        self.model = self._build_model()
        self.target_model = copy.deepcopy(self.model)
        self.target_model.eval()
        
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=LEARNING_RATE, 
            weight_decay=WEIGHT_DECAY
        )
        
        # Huber loss (smooth L1 loss in PyTorch)
        self.criterion = nn.SmoothL1Loss()
        
        self.stats = {
            'train_loss': [],
            'win_rate': []
        }
        
        if os.path.exists(LOAD_PATH):
            self.load(LOAD_PATH)
        self.step = 0

    def _build_model(self) -> DQNModel:
        model = DQNModel().to(self.device)
        return model

    def __call__(self, state):
        self.step_itter()
        act = [0, 0, 0, 0, 0]
        
        if random.random() < self.epsilon:
            act[random.randint(0, 4)] = 1
        else:
            with torch.no_grad():
                # Convert state to PyTorch tensor
                state_copy = np.array(state, copy=True)
                state_tensor = torch.FloatTensor(state_copy).to(self.device)
                print(f"State shape before permute: {state_tensor.shape}")
                
                # State shape: (width, height, depth, channels) = (280, 120, 3, 1)
                # Need: (batch, channels, depth, height, width) = (1, 1, 3, 120, 280)
                
                if state_tensor.dim() == 4:
                    # Add batch dimension
                    state_tensor = state_tensor.unsqueeze(0)  # (1, 280, 120, 3, 1)
                    # Permute: (batch, width, height, depth, channels) -> (batch, channels, depth, height, width)
                    # indices:    0      1      2      3       4     ->     0       4        3      2      1
                    # state_tensor = state_tensor.permute(0, 4, 3, 2, 1)  # (1, 1, 3, 120, 280)
                    state_tensor = state_tensor.permute(0, 1, 4, 2, 3)  # (1, 1, 3, 120, 280)
                elif state_tensor.dim() == 5:
                    state_tensor = state_tensor.permute(0, 1, 4, 2, 3)
                    # state_tensor = state_tensor.permute(0, 4, 3, 2, 1)
                
                print(f"State shape after permute: {state_tensor.shape}")

                q_values = self.model(state_tensor)
                action_idx = torch.argmax(q_values).item()
                act[action_idx] = 1
        
        return act

    def _process_batch(self, batch):
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Stack states
        states = np.vstack(states)
        next_states = np.vstack(next_states)
        
        # Convert to tensors and reorder dimensions
        # From (batch, width, height, depth, 1) to (batch, 1, depth, height, width)
        states_tensor = torch.FloatTensor(states).to(self.device).permute(0, 4, 3, 2, 1)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device).permute(0, 4, 3, 2, 1)
        
        with torch.no_grad():
            # Get Q-values
            current_q = self.model(states_tensor).cpu().numpy()
            next_q = self.target_model(next_states_tensor).cpu().numpy()
        
        # Calculate target Q-values
        rewards_array = np.array(rewards)
        dones_array = np.array(dones)
        max_next_q = np.max(next_q, axis=1)
        target_q = rewards_array + GAMMA * max_next_q * (1 - dones_array)
        
        # Update Q-values for actions taken
        target = current_q.copy()
        for i, (t, q, a) in enumerate(zip(target, target_q, actions)):
            action_idx = np.argmax(a)
            t[action_idx] = t[action_idx] * (1 - Q_VAL_RATIO) + Q_VAL_RATIO * q
        
        return states_tensor, torch.FloatTensor(target).to(self.device)

    def replay(self, percent):
        num_memories = int(len(self.memory) * percent)
        if num_memories < BATCH_SIZE:
            return

        num_batches = num_memories // BATCH_SIZE
        num_sub_batches = 4
        sub_batch_size = BATCH_SIZE // num_sub_batches

        self.model.train()
        losses = []
        
        for _ in range(num_batches):
            minibatch = self.memory(percent)

            futures = []
            for i in range(0, BATCH_SIZE, sub_batch_size):
                sub_batch = minibatch[i:i+sub_batch_size]
                futures.append(self.thread_pool.submit(self._process_batch, sub_batch))

            all_states = []
            all_targets = []
            for future in futures:
                states, targets = future.result()
                all_states.append(states)
                all_targets.append(targets)

            all_states = torch.cat(all_states, dim=0)
            all_targets = torch.cat(all_targets, dim=0)

            # Training step
            self.optimizer.zero_grad()
            predictions = self.model(all_states)
            loss = self.criterion(predictions, all_targets)
            loss.backward()
            self.optimizer.step()
            
            losses.append(loss.item())
        
        self.model.eval()
        self.stats['train_loss'].append(sum(losses)/len(losses))

    def remember(self, state, action, reward, next_state, done):
        self.memory.remember((state, action, reward, next_state, done))

    def apply_decay(self):
        if self.train and self.epsilon > MIN_EPSILON:
            self.epsilon *= self.decay

    def step_itter(self):
        self.step = (self.step + 1) % MODEL_SAMPLE_RATE
        return self.step

    def reset(self):
        self.step = 0

    def load(self, name):
        """Load model weights"""
        if os.path.exists(name):
            checkpoint = torch.load(name, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                if 'optimizer_state_dict' in checkpoint and self.train:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"Model loaded from {name}")

    def save(self, name):
        """Save model weights and optimizer state"""
        os.makedirs(os.path.dirname(name) if os.path.dirname(name) else '.', exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, name)
        print(f"Model saved to {name}")

    def update_target(self):
        """Update target network"""
        self.target_model = copy.deepcopy(self.model)
        self.target_model.eval()