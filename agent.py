import numpy as np
from collections import deque
import random
import os
from global_vars import *

class PongAgent:
    def __init__(self):
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = 1.0
        self.model = self._build_model()

    def _build_model(self):
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if os.path.exists(MODEL_PATH):
            return self.load(MODEL_PATH)
        else:
            model = Sequential()
            model.add(Dense(24, input_dim=INPUT_SIZE, activation='relu'))
            model.add(Dense(24, activation='relu'))
            model.add(Dense(ACTION_SIZE, activation='sigmoid'))
            model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE))
            return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        act_values = self.model.predict(state)[0]
        return np.where(act_values > 0.5, 1, 0)

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


