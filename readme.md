# Enhanced Pong Game with Self-play AI Agent

## Game Implementation (game.py)

**Objective:**
Develop an enhanced version of the classic Pong game that includes additional obstacles and more complex mechanics.

**Features:**

1. **Obstacles:**
   - Place obstacles in the middle of the field.
   - The obstacles should move up and down and change speed and location throughout the game.
   - Obstacles should have a distinct color to differentiate them from the paddles and ball.

2. **Paddles:**
   - Each player controls a paddle, and the two paddles should be different colors.
   - The paddles get reset at the start of every point back to the original middle height.

3. **Ball Mechanics:**
   - The ball should be a circle.
   - The reflection angle of the ball off the paddle should depend on the distance from the paddle's center.
   - Implement a mechanic that accelerates the ball after a paddle hit to increase the game's difficulty.

4. **Collision Detection:**
   - Implement border logic carefully to ensure accurate collision detection.
   - Handle collisions between the ball and paddles, obstacles, and walls properly.

## AI Agent Implementation (agent.py)

**Objective:**
Implement an AI agent using TensorFlow that can learn to play the enhanced Pong game through self-play and Deep Q-Learning.

**Features:**

1. **Agent Architecture:**
   - Use Deep Q Networks (DQN) to implement the agent.
   - Utilize the self-play principle for training the agent.
   - The agent should be both players in the game and learn to play against itself.

2. **Input to the Agent:**
   - Top and bottom positions of the paddle the agent controls.
   - Current x and y coordinates of the ball.
   - Absolute velocity components (x and y) of the ball.

3. **Training Process:**
   - The agent should learn the optimal strategy to play the game through continuous self-play and reinforcement learning.
  
### Additional Notes:
- Ensure all components (paddles, obstacles, ball) are properly color-coded for clarity.
- Pay close attention to the game's physics, especially in how collisions and reflections are handled.
- Implement the acceleration mechanic for the ball to add an increasing challenge as the game progresses.
