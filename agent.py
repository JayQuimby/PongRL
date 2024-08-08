import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
import random
import os
from global_vars import *
from concurrent.futures import ThreadPoolExecutor

class PongAgent:
    def __init__(self):
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = 1.0
        self.model = self._build_model()
        if os.path.exists(MODEL_PATH):
            self.load(MODEL_PATH)
        self.step = 0
        self.thread_pool = ThreadPoolExecutor(max_workers=4)  # Adjust based on your CPU

    def _build_model(self):
        model = Sequential([
            Dense(24, input_dim=INPUT_SIZE, activation='relu'),
            Dense(ACTION_SIZE, activation='sigmoid')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE))
        return model

    def remember(self, state, action, reward, next_state, done):
        if abs(reward) > 0.2:
            self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        self.step_itter()
        if self.step == 0:
            act_values = self.model.predict(state, verbose=0)[0]
            return np.where(act_values > 0.5, 1, 0)
        elif random.random() < self.epsilon:
            act = []
            for _ in range(4):
                act.append(0 if random.random() > 0.5 else 1)
            act.append(sum(act) == 0)
            return act
        else:
            return NULL_ACT

    def _process_batch(self, batch):
        states, actions, rewards, next_states, dones = zip(*batch)
        states = np.vstack(states)
        next_states = np.vstack(next_states)

        # Predict Q-values for next_states
        next_q_values = self.model.predict(next_states, verbose=0)
        
        # Calculate targets
        max_next_q_values = np.max(next_q_values, axis=1)
        targets = rewards + GAMMA * max_next_q_values * (1 - np.array(dones))
        
        # Get Q-values for current states
        target_f = self.model.predict(states, verbose=0)
        
        # Update Q-values for the actions taken
        for i, action in enumerate(actions):
            target_f[i][action] = targets[i]

        return states, target_f

    def replay(self):
        print(f'{len(self.memory)} memories to sample.')
        if len(self.memory) < BATCH_SIZE:
            return

        minibatch = random.sample(self.memory, BATCH_SIZE)
        num_sub_batches = 4  # Adjust based on your needs
        sub_batch_size = BATCH_SIZE // num_sub_batches

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

        self.model.fit(all_states, all_targets, epochs=1, verbose=1, batch_size=32)

        if self.epsilon > MIN_EPSILON:
            self.epsilon *= EPSILON_DECAY

    def step_itter(self):
        self.step = (self.step + 1) % MODEL_SAMPLE_RATE
        return self.step

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


