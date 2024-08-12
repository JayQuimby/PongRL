import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2
from keras.optimizers import Adam
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
from keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
import random
import os
from global_vars import *
from concurrent.futures import ThreadPoolExecutor

class AgentMemory:
    def __init__(self) -> None:
        self.max_mem = MEMORY_SIZE
        self.memories = []
        self.reward_ind = 2

    def __call__(self, x):
        if x < 1:
            return self.memories
        return random.sample(self.memories, x) 

    def __len__(self):
        return len(self.memories)

    def remember(self, action):
        if len(self.memories) < self.max_mem:
            self.memories.append(action)
        else:
            self.purge_memory()
            self.memories.append(action)

    def purge_memory(self):
        self.memories.sort(key=lambda x: x[self.reward_ind], reverse=True)
        keep_top = int(0.1 * self.max_mem)
        new_memories = self.memories[:keep_top]
        new_lim = self.max_mem // 2

        for memory in self.memories[keep_top:]:
            prob = 1 / (1 + math.exp(-memory[self.reward_ind]))
            if random.random() < prob:
                new_memories.append(memory)
            
            if len(new_memories) >= new_lim:
                break
        
        self.memories = new_memories

    def clear(self):
        self.memories = []

class PongAgent:
    def __init__(self, train, games):
        self.train = train
        self.num_itters = games
        self.memory = AgentMemory()#deque(maxlen=MEMORY_SIZE)
        self.epsilon = 1.0 if train else 0.0
        self.decay = 0.2**(1/games)
        self.model = self._build_model()
        self.stats = {
            'train_loss': [],
            'win_rate': []
        }
        if os.path.exists(MODEL_PATH):
            self.load(MODEL_PATH)
        self.step = 0
        self.thread_pool = ThreadPoolExecutor(max_workers=os.cpu_count())
        self.last_act = [0,0,0,0,1]

    def create_dense_block(units, dropout_rate=DROPOUT_RATE, l2_lambda=L2_LAMBDA):
        return [
            Dense(units, activation='relu', kernel_regularizer=l2(l2_lambda)),
            Dropout(dropout_rate)
        ]

    def _build_model(self, hidden_layers=[128, 256, 128, 64, 64, 32]) -> Sequential:
        model = Sequential()
        
        # Input layer
        model.add(Dense(hidden_layers[0], input_dim=INPUT_SIZE, activation='relu', kernel_regularizer=l2(L2_LAMBDA)))
        model.add(Dropout(DROPOUT_RATE))
        
        # Hidden layers
        for units in hidden_layers[1:]:
            model.add(Dense(units, activation='relu', kernel_regularizer=l2(L2_LAMBDA)))
            model.add(Dropout(DROPOUT_RATE))
        
        # Output layer
        model.add(Dense(ACTION_SIZE, activation='tanh'))
        
        model.compile(
            loss='mse', 
            optimizer=Adam(learning_rate=LEARNING_RATE),
            metrics=['mae']
        )
        return model

    def remember(self, state, action, reward, next_state, done):
        if abs(reward) > 0.5:
            self.memory.remember((state, action, reward, next_state, done))#.append()

    def __call__(self, state):
        self.step_itter()
        act = [0,0,0,0,0]
        if random.random() < self.epsilon:
            act[random.randint(0, 4)] = 1
        else:
            if self.step == 0:
                act[np.argmax(self._optimized_predict(state))] = 1
            else:
                act = self.last_act
        return act

    @tf.function
    def _optimized_predict(self, state):
        return self.model(state, training=False)[0]

    def _process_batch(self, batch):
        states, actions, rewards, next_states, dones = zip(*batch)
        states = np.vstack(states)
        next_states = np.vstack(next_states)

        # Get Q-values for current states and next_states
        target_f = self.model.predict(states, verbose=0) # Nx5
        next_q_values = self.model.predict(next_states, verbose=0) # Nx5
        
        # Calculate targets
        max_next_q_values = np.max(next_q_values, axis=1) # Nx1
        targets = rewards + GAMMA * max_next_q_values * (1 - np.array(dones))

        # Update Q-values for the actions taken
        for i, action in enumerate(actions):
            target_f[i][action] = targets[i]

        return states, target_f

    def replay(self, divisor):
        num_memories = len(self.memory) // divisor
        if num_memories < BATCH_SIZE:
            return

        num_batches = num_memories // BATCH_SIZE  # Number of batches
        num_sub_batches = 4  # Adjust based on your needs
        sub_batch_size = BATCH_SIZE // num_sub_batches

        print(f'Training on {num_memories} memories.')
        losses = []
        for _ in range(num_batches):
            minibatch = self.memory(num_memories) 

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

            all_states = np.vstack(all_states)
            all_targets = np.vstack(all_targets)

            loss = self.model.fit(all_states, all_targets, epochs=1, verbose=0, batch_size=BATCH_SIZE)
            losses.extend(loss.history['mae'])
        self.stats['train_loss'].append(sum(losses)/len(losses))
        if self.train and self.epsilon > MIN_EPSILON:
            self.epsilon *= self.decay

    def step_itter(self):
        self.step = (self.step + 1) % MODEL_SAMPLE_RATE
        return self.step

    def reset_mem(self):
        self.memory.clear()
        self.epsilon = 1.0
        self.step = 0
        self.last_act = [0,0,0,0,0]

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


