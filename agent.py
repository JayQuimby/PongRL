import numpy as np
from collections import deque
import random

# Hyperparameters
ACTION_SIZE = 5
GAMMA = 0.95
LEARNING_RATE = 0.001
MEMORY_SIZE = 2000
BATCH_SIZE = 32
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01

class PongAgent:
    def __init__(self, input_size):
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = 1.0
        self.model = self._build_model(input_size)

    def _build_model(self, inp_dim):
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense

        model = Sequential()
        model.add(Dense(24, input_dim=inp_dim, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(ACTION_SIZE, activation='sigmoid'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return -1 #random.randrange(ACTION_SIZE)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return

        minibatch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + GAMMA * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > MIN_EPSILON:
            self.epsilon *= EPSILON_DECAY

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


