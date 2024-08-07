import pygame
import math
from global_vars import *
from objects import Ball, Paddle, Obstacle
from utils import load_conf, distance, collide
p_conf = load_conf('paddle')
b_conf = load_conf('ball')
o_conf = load_conf('obstacle')

pygame.init()
pygame.font.init()

font = pygame.font.Font(None, 28)
big_font = pygame.font.Font(None, 48)
small_font = pygame.font.Font(None, 18)
rr = lambda x: round(x, 1)


class Game:

    def __init__(self, **config):
        self.config = config
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT + 100))
        pygame.display.set_caption('Self play simulation')
        self.clock = pygame.time.Clock()

        self.ball = Ball(b_conf)
        
        p_conf['x'] = PADDLE_RADIUS * 2
        p_conf['color'] = BLUE
        self.left_paddle = Paddle(p_conf)

        p_conf['x'] = SCREEN_WIDTH - PADDLE_RADIUS * 2
        p_conf['color'] = GREEN
        self.right_paddle = Paddle(p_conf)
        self.fps = 60

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

    # TODO: This needs to be changed!!!
    def get_ai_inputs(self, paddle):
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
        
        for obstacle in self.obstacles:
            grid_x = int(obstacle.x // MIDGROUND_SPLIT)
            grid_y = int(obstacle.y // MIDGROUND_SPLIT)
            index = grid_y * grid_width + grid_x
            obstacle_grid[index] = 1
        
        inputs.extend(obstacle_grid)
    
        return inputs

    def get_ai_moves(self, players):
        bh = self.ball.y
        vl = self.ball.vy
        fbh = bh + vl**2
        lph = self.left_paddle.y
        rph = self.right_paddle.y
        buffer = PADDLE_RADIUS // 8

        follow = lambda x: PADDLE_VEL * (-(x + buffer > fbh) + (x -  buffer < fbh))
        center = lambda x: PADDLE_VEL * min(1, max(-1, (SCREEN_HEIGHT//2 - x)//10))
        
        if players < 1:
            if self.ball.vx < 0 and self.ball.x < SCREEN_WIDTH//2:
                self.left_paddle.vy += follow(lph)
                if distance(self.left_paddle, self.ball) < self.left_paddle.r * 3:
                    self.left_paddle.vx += PADDLE_VEL // 2
            else:
                self.left_paddle.vx -= PADDLE_VEL
                self.left_paddle.vy += center(self.left_paddle.y)

        if players < 2:
            if self.ball.vx > 0 and self.ball.x > SCREEN_WIDTH//2:
                self.right_paddle.vy += follow(rph)
                if distance(self.right_paddle, self.ball) < self.right_paddle.r * 3:
                    self.right_paddle.vx -= PADDLE_VEL // 2
            else:
                self.right_paddle.vx += PADDLE_VEL
                self.right_paddle.vy += center(self.right_paddle.y)

    def move_objects(self):
        self.left_paddle.vy *= PADDLE_MOVE_DECAY
        self.left_paddle.vx *= PADDLE_MOVE_DECAY
        self.right_paddle.vy *= PADDLE_MOVE_DECAY
        self.right_paddle.vx *= PADDLE_MOVE_DECAY

        keys = pygame.key.get_pressed()
        if keys[pygame.K_COMMA]:
            self.fps -= 1
        if keys[pygame.K_PERIOD]:
            self.fps += 1

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
        p1_reward = (self.ball.x - SCREEN_WIDTH//2) / SCREEN_WIDTH + SCORE_REWARD_MULT * (p1_score - p2_score)
        return score, (dr[0] + p1_reward, dr[1] - p1_reward)
    
    def detect_collision(self):
        # Collision with left paddle
        paddles = [self.left_paddle, self.right_paddle]
        rewards = [0,0]
        for paddle, rew in zip(paddles, rewards):
            if paddle.collides(self.ball):
                rew += 2
                paddle.hit = True
                collide(paddle, self.ball)
            else:
                paddle.hit = False

        for i, obstacle in enumerate(self.obstacles):
            if obstacle.collides(self.ball):
                collide(obstacle, self.ball)
            
            for ind, other in enumerate(self.obstacles):
                if ind == i:
                    continue
                if obstacle.collides(other):
                    collide(obstacle, other)

        # Ball out of bounds       
        if self.ball.y < 0 and self.ball.vy < 0 or self.ball.y > SCREEN_HEIGHT and self.ball.vy > 0:
            self.ball.vy *= -1
        
        return rewards

    def redraw_screen(self):
        # Drawing
        self.screen.fill(BLACK)
        left_mid = SCREEN_WIDTH // 2 - OBSTACLE_SPREAD
        right_mid = SCREEN_WIDTH // 2 + OBSTACLE_SPREAD
        mid_range = right_mid - left_mid
        for i in range(left_mid, right_mid + 1, MIDGROUND_SPLIT):
            pygame.draw.rect(self.screen, (255,255,255), (i, 0, 1, SCREEN_HEIGHT))

        for i in range(0, SCREEN_HEIGHT, MIDGROUND_SPLIT):
            pygame.draw.rect(self.screen, (255,255,255), (left_mid, i, mid_range, 1))

        self.ball.draw(self.screen)
        self.left_paddle.draw(self.screen)
        self.right_paddle.draw(self.screen)
        for obstacle in self.obstacles:
            obstacle.draw(self.screen)

        pygame.draw.rect(self.screen, (255,255,255), (0, SCREEN_HEIGHT, SCREEN_WIDTH, 1))
        pygame.draw.rect(self.screen, (255,255,255), (0, SCREEN_HEIGHT + 40, SCREEN_WIDTH, 1))

    def display_variables(self, score, rewards):
        ball_position_text = font.render(f'Ball Position: ({self.ball.x//10*10}, {self.ball.y//10*10})', True, WHITE)
        ball_velocity_text = font.render(f'Ball Velocity: ({rr(self.ball.vx)}, {rr(self.ball.vy)})', True, WHITE)
        left_paddle_text = font.render(f'Left Paddle Y: {rr(self.left_paddle.y)}', True, WHITE)
        right_paddle_text = font.render(f'Right Paddle Y: {rr(self.right_paddle.y)}', True, WHITE)
        score_text = big_font.render(f'P1: {score["p1"]} | P2: {score["p2"]}', True, WHITE)
        fps_count = small_font.render(f'fps: {self.fps}', True, (255,255,0))
        p1_reward = font.render(f'P1 Reward: {rr(rewards[0])}', True, WHITE)
        p2_reward = font.render(f'P2 Reward: {rr(rewards[1])}', True, WHITE)

        self.screen.blit(fps_count, (0, 0))
        self.screen.blit(ball_position_text, (SCREEN_WIDTH//2 + 50, SCREEN_HEIGHT + 10))
        self.screen.blit(ball_velocity_text, (SCREEN_WIDTH//2 - 210, SCREEN_HEIGHT + 10))
        self.screen.blit(score_text, (SCREEN_WIDTH//2 - 70, SCREEN_HEIGHT + 55))
        self.screen.blit(p1_reward, (20, SCREEN_HEIGHT + 55))
        self.screen.blit(p2_reward, (SCREEN_WIDTH - 160, SCREEN_HEIGHT + 55))
        self.screen.blit(left_paddle_text, (10, SCREEN_HEIGHT + 10))
        self.screen.blit(right_paddle_text, (SCREEN_WIDTH - 210, SCREEN_HEIGHT + 10))

    def step(self, score):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
        self.move_objects()
        def_rewards = self.detect_collision()
        self.redraw_screen()
        score, rewards = self.check_for_score(score, def_rewards)
        self.display_variables(score, rewards)
        pygame.display.flip()
        self.clock.tick(self.fps)
        return score, rewards
