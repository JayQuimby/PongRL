import pygame
from global_vars import *
from objects import Ball, Paddle, Obstacle
from agent import PongAgent
from utils import load_conf, distance, collide
p_conf = load_conf('paddle')
b_conf = load_conf('ball')
o_conf = load_conf('obstacle')
rr = lambda x: round(x, 1)

class GameDisplay:
    def __init__(self, game):
        self.game = game
        
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT + 100))
        pygame.display.set_caption('Self play simulation')
        self.clock = pygame.time.Clock()
        self.fps = 60

        pygame.font.init()
        self.font = pygame.font.Font(None, 28)
        self.big_font = pygame.font.Font(None, 48)
        self.small_font = pygame.font.Font(None, 18)

    def render_text(self, text, font, color=WHITE):
        return font.render(text, True, color)

    def blit_text(self, text, position):
        self.screen.blit(text, position)

    def display_variables(self, score, rewards):
        variables = [
            (f'Ball Position: ({self.game.ball.x // 10 * 10}, {self.game.ball.y // 10 * 10})', self.font, (MID_WIDTH + 50, SCREEN_HEIGHT + 10)),
            (f'Ball Velocity: ({rr(self.game.ball.vx)}, {rr(self.game.ball.vy)})', self.font, (MID_WIDTH - 210, SCREEN_HEIGHT + 10)),
            (f'Left Paddle Y: {rr(self.game.left_paddle.y)}', self.font, (10, SCREEN_HEIGHT + 10)),
            (f'Right Paddle Y: {rr(self.game.right_paddle.y)}', self.font, (SCREEN_WIDTH - 210, SCREEN_HEIGHT + 10)),
            (f'P1: {score["p1"]} | P2: {score["p2"]}', self.big_font, (MID_WIDTH - 70, SCREEN_HEIGHT + 55)),
            (f'fps: {self.fps}', self.small_font, (0, 0), (255, 255, 0)),
            (f'P1 Reward: {rr(rewards[0])}', self.font, (20, SCREEN_HEIGHT + 55)),
            (f'P2 Reward: {rr(rewards[1])}', self.font, (SCREEN_WIDTH - 160, SCREEN_HEIGHT + 55)),
        ]

        for text, font, position, *color in variables:
            color = color[0] if color else WHITE
            rendered_text = self.render_text(text, font, color)
            self.blit_text(rendered_text, position)

    def redraw_screen(self):
        # Drawing
        self.screen.fill(BLACK)
        left_mid = MID_WIDTH - OBSTACLE_SPREAD
        right_mid = MID_WIDTH + OBSTACLE_SPREAD
        mid_range = right_mid - left_mid
        for i in range(left_mid, right_mid + 1, MIDGROUND_SPLIT):
            pygame.draw.rect(self.screen, (255,255,255), (i, 0, 1, SCREEN_HEIGHT))

        for i in range(0, SCREEN_HEIGHT, MIDGROUND_SPLIT):
            pygame.draw.rect(self.screen, (255,255,255), (left_mid, i, mid_range, 1))

        self.game.ball.draw(self.screen)
        self.game.left_paddle.draw(self.screen)
        self.game.right_paddle.draw(self.screen)
        for obstacle in self.game.obstacles:
            obstacle.draw(self.screen)

        pygame.draw.rect(self.screen, (255,255,255), (0, SCREEN_HEIGHT, SCREEN_WIDTH, 1))
        pygame.draw.rect(self.screen, (255,255,255), (0, SCREEN_HEIGHT + 40, SCREEN_WIDTH, 1))

    def update_display(self, s, r):
        self.redraw_screen()
        self.display_variables(s, r)
        pygame.display.flip()
        self.clock.tick(self.fps)

    def inc_fps(self, neg=False):
        self.fps -= 1 if neg else -1

class Game:
    def __init__(self, **config):
        pygame.init()
        self.config = config
        self.read_out = GameDisplay(self)

        self.nn_model = None
        if config.get('nn', False):
            self.nn_model = PongAgent()

        self.ball = Ball(b_conf)
        
        p_conf['x'] = PADDLE_RADIUS * 2
        p_conf['color'] = BLUE
        self.left_paddle = Paddle(p_conf)

        p_conf['x'] = SCREEN_WIDTH - PADDLE_RADIUS * 2
        p_conf['color'] = GREEN
        self.right_paddle = Paddle(p_conf)
        
        self.obstacles = []
        for _ in range(NUM_OBSTACLES):
            new_o = Obstacle(o_conf)
            self.obstacles.append(new_o)

        self.running = True

    def get_player_moves(self, keys, np):
        # Paddle movement
        if np > 0:
            self.left_paddle.vy += PADDLE_VEL * (-keys[pygame.K_w] + keys[pygame.K_s])
            self.left_paddle.vx += PADDLE_VEL * (-keys[pygame.K_a] + keys[pygame.K_d])

        if np > 1:
            self.right_paddle.vy += PADDLE_VEL * (-keys[pygame.K_UP] + keys[pygame.K_DOWN])
            self.right_paddle.vy += PADDLE_VEL * (-keys[pygame.K_LEFT] + keys[pygame.K_RIGHT])

    def get_state(self, paddle):
        inputs = []
        
        # Ball position and velocity
        inputs.extend([self.ball.x / SCREEN_WIDTH, self.ball.y / SCREEN_HEIGHT])
        inputs.extend([self.ball.vx / BALL_MAX_SPEED, self.ball.vy / BALL_MAX_SPEED])
        
        # Paddle position and velocity
        inputs.extend([paddle.x / SCREEN_WIDTH, paddle.y / SCREEN_HEIGHT])
        inputs.extend([paddle.vx / PADDLE_VEL, paddle.vy / PADDLE_VEL])
        
        # Encode obstacles positions
        grid_width = SCREEN_WIDTH // MIDGROUND_SPLIT
        grid_height = SCREEN_HEIGHT // MIDGROUND_SPLIT
        obstacle_grid = [0] * (grid_width * grid_height)
        
        shift = MID_WIDTH - OBSTACLE_SPREAD

        for obstacle in self.obstacles:
            grid_x = int((obstacle.x - shift) // MIDGROUND_SPLIT)
            grid_y = int(obstacle.y // MIDGROUND_SPLIT)
            index = grid_y * grid_width + grid_x
            obstacle_grid[index] = 1
        
        inputs.extend(obstacle_grid)
    
        return inputs

    def get_ai_moves(self, players):
        bh = self.ball.y
        vl = self.ball.vy
        fbh = bh + vl**2
        buffer = PADDLE_RADIUS // 8
        nn_bots = self.config.get('nn', 0)
        follow = lambda x: PADDLE_VEL * (-(x + buffer > fbh) + (x -  buffer < fbh))
        center = lambda x: PADDLE_VEL * min(1, max(-1, (MID_HEIGHT - x)//10))

        def update_paddle(paddle, direction, x_condition, follow_action, center_action):
            if x_condition:
                paddle.vy += follow_action(paddle.y)
                if distance(paddle, self.ball) < paddle.r * 3:
                    paddle.vx += PADDLE_VEL // 2 * direction
            else:
                paddle.vx -= PADDLE_VEL * direction
                paddle.vy += center_action(paddle.y)

        def nn_play(paddle):
            model_input = self.get_state(paddle)
            action = self.nn_model.predict(model_input)

            if action == 0:  # Up
                paddle.vy -= PADDLE_VEL
            elif action == 1:  # Down
                paddle.vy += PADDLE_VEL
            elif action == 2:  # Left
                paddle.vx -= PADDLE_VEL
            elif action == 3:  # Right
                paddle.vx += PADDLE_VEL

        if players == 0:
            if nn_bots == 0: 
                update_paddle(self.left_paddle, 1, self.ball.vx < 0 and self.ball.x < MID_WIDTH, follow, center)
                update_paddle(self.right_paddle, -1, self.ball.vx > 0 and self.ball.x > MID_WIDTH, follow, center)
            elif nn_bots == 1:
                nn_play(self.left_paddle)
                update_paddle(self.right_paddle, -1, self.ball.vx > 0 and self.ball.x > MID_WIDTH, follow, center)
            else:
                nn_play(self.left_paddle)
                nn_play(self.right_paddle)

        elif players == 1:
            if nn_bots > 0:
                nn_play(self.right_paddle)
            else:
                update_paddle(self.right_paddle, -1, self.ball.vx > 0 and self.ball.x > MID_WIDTH, follow, center)
    
    def resistance(self):
        self.left_paddle.vy *= PADDLE_MOVE_DECAY
        self.left_paddle.vx *= PADDLE_MOVE_DECAY
        self.right_paddle.vy *= PADDLE_MOVE_DECAY
        self.right_paddle.vx *= PADDLE_MOVE_DECAY

    def toggle_keys(self, keys):
        if keys[pygame.K_COMMA]: self.read_out.inc_fps(1)
        if keys[pygame.K_PERIOD]: self.read_out.inc_fps()

    def move_objects(self):
        keys = pygame.key.get_pressed()
        self.resistance()
        self.toggle_keys(keys)

        human_players = self.config['players']
        if human_players > 0:
            self.get_player_moves(keys, human_players)

        if human_players < 2: 
            self.get_ai_moves(human_players)

        self.left_paddle.move()
        self.right_paddle.move()
        self.ball.move()

        for obs in self.obstacles:
            obs.move()

    def check_for_score(self, score, dr):
        p1_score = self.ball.x > SCREEN_WIDTH
        p2_score = self.ball.x < 0
        
        if p1_score or p2_score:
            if p1_score:
                score['p1'] += 1
            else:
                score['p2'] += 1

            self.ball.reset()
            self.left_paddle.reset()
            self.right_paddle.reset()
            for ob in self.obstacles:
                ob.reset()
        p1_reward = (self.ball.x - MID_WIDTH) / SCREEN_WIDTH + SCORE_REWARD_MULT * (p1_score - p2_score)
        return score, (dr[0] + p1_reward, dr[1] - p1_reward)
    
    def obstacle_collision(self):
        for i, obstacle in enumerate(self.obstacles):
            if obstacle.collides(self.ball):
                collide(obstacle, self.ball)
            if i < len(self.obstacles):
                for other in self.obstacles[i+1:]:
                    if obstacle.collides(other):
                        collide(obstacle, other)

    def paddle_collision(self, paddles, rewards):
        for paddle, rew in zip(paddles, rewards):
            if paddle.collides(self.ball):
                rew += 2
                paddle.hit = True
                collide(paddle, self.ball)
            else:
                paddle.hit = False
        return rewards

    def detect_collision(self, score):
        rewards = [0,0]
        paddles = [self.left_paddle, self.right_paddle]
        
        self.obstacle_collision()
        rewards = self.paddle_collision(paddles, rewards)
        return self.check_for_score(score, rewards)

    def step(self, score):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
        self.move_objects()
        score, rewards = self.detect_collision(score)
        self.read_out.update_display(score, rewards)
        return score, rewards

if __name__ == "__main__":
    conf = {'players': 0}
    score = {'p1': 0, 'p2': 0}
    g = Game(**conf)
    while g.running:
        score, rewards = g.step(score)
    pygame.quit()

