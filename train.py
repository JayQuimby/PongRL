import numpy as np
from game import Game
from agent import PongAgent

if __name__ == "__main__":
    train_conf = {
        'nn': 2,
        'players':0
    }
    agent = PongAgent()
    game = Game()
    # Add your game loop here to train the agent
    # The loop should involve initializing the game state, getting the state, choosing an action, and then applying the DQN agent logic

    for e in range(1000):  # Number of episodes
        # Reset the game to start a new episode
        raw_state = game.get_state(game.left_paddle)
        state = np.reshape(raw_state, [1, len(raw_state)])  # initial_state() should return the initial state of the game

        for time in range(500):  # Max time steps per episode
            action = agent.act(state)
            next_state, reward, done = game.step(action)  # step() should return next_state, reward, done after taking the action
            next_state = np.reshape(next_state, [1, len(next_state)])
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                print(f"Episode {e+1}/{1000} finished after {time+1} timesteps")
                break

        agent.replay()
        if e % 50 == 0:
            agent.save(f"pong-dqn-{e}.h5")