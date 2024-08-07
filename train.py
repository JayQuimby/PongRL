import tensorflow as tf
from tensorflow import keras
import numpy as np
from game import Game
from global_vars import SCREEN_WIDTH, SCREEN_HEIGHT, MIDGROUND_SPLIT

def create_model(input_size):
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(input_size,)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(4, activation='softmax')  # Up, Down, Left, Right
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_network():
    game = Game(players=0, nn=True)
    
    input_size = 4 + 4 + (SCREEN_WIDTH // MIDGROUND_SPLIT) * (SCREEN_HEIGHT // MIDGROUND_SPLIT)
    model = create_model(input_size)
    
    num_episodes = 1000
    epsilon = 1.0
    epsilon_decay = 0.995
    min_epsilon = 0.01
    
    for episode in range(num_episodes):
        game.ball.reset()
        game.left_paddle.reset()
        game.right_paddle.reset()
        for obstacle in game.obstacles:
            obstacle.reset()
        
        score = {'p1': 0, 'p2': 0}
        total_reward = 0
        done = False
        
        while not done:
            state = game.get_ai_inputs(game.left_paddle)
            
            if np.random.random() < epsilon:
                action = np.random.randint(0, 4)
            else:
                q_values = model.predict(np.array([state]))[0]
                action = np.argmax(q_values)
            
            # Apply action
            if action == 0:  # Up
                game.left_paddle.vy -= game.config['paddle_vel']
            elif action == 1:  # Down
                game.left_paddle.vy += game.config['paddle_vel']
            elif action == 2:  # Left
                game.left_paddle.vx -= game.config['paddle_vel']
            elif action == 3:  # Right
                game.left_paddle.vx += game.config['paddle_vel']
            
            score, rewards = game.step(score)
            next_state = game.get_ai_inputs(game.left_paddle)
            reward = rewards[0]
            total_reward += reward
            
            # Store experience
            experience = (state, action, reward, next_state)
            
            # Train on this experience
            target = reward + 0.99 * np.max(model.predict(np.array([next_state]))[0])
            target_vec = model.predict(np.array([state]))[0]
            target_vec[action] = target
            model.fit(np.array([state]), np.array([target_vec]), epochs=1, verbose=0)
            
            if score['p1'] >= 10 or score['p2'] >= 10:
                done = True
        
        print(f"Episode: {episode+1}, Score: {score['p1']}-{score['p2']}, Total Reward: {total_reward}")
        
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
    
    return model

if __name__ == "__main__":
    trained_model = train_network()
    trained_model.save("pong_ai_model.h5")