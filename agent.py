import tensorflow as tf
import numpy as np
import random
from collections import deque

class PongAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

def preprocess_state(state):
    return np.reshape(list(state.values()), [1, 8])

def train_agent(episodes, batch_size=32):
    from game import run_game, get_game_state  # Import the game functions

    state_size = 8  # Number of input parameters
    action_size = 3  # Up, Down, Stay
    agent = PongAgent(state_size, action_size)

    for e in range(episodes):
        state = preprocess_state(get_game_state())
        for time in range(500):  # Limit the number of steps per episode
            action = agent.act(state)
            # Run one frame of the game and get the new state
            next_state, reward, done = run_game(lambda x: action)
            next_state = preprocess_state(next_state)
            
            # Reward shaping: You might want to adjust this based on the game outcome
            reward = reward if not done else -10

            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                agent.update_target_model()
                print(f"episode: {e}/{episodes}, score: {time}, e: {agent.epsilon:.2}")
                break
            
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        if e % 10 == 0:
            agent.save(f"pong_agent_{e}.h5")

    return agent

if __name__ == "__main__":
    trained_agent = train_agent(episodes=1000)
    trained_agent.save("final_pong_agent.h5")