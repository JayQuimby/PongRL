import pygame
import math
from global_vars import *

from utils import load_conf
p_conf = load_conf('paddle')
b_conf = load_conf('ball')
o_conf = load_conf('obstacle')

from ball import Ball
from paddle import Paddle
from obstacle import Obstacle
pygame.init()
pygame.font.init()
font = pygame.font.Font(None, 28)
big_font = pygame.font.Font(None, 48)

rr = lambda x: round(x, 1)

def collide(obj1, obj2, momentum_factor=1.5, size_speed_factor=1.5, min_speed=2.5, max_speed=15):
    # Calculate relative velocity
    rvx = obj2.vx - obj1.vx
    rvy = obj2.vy - obj1.vy

    # Calculate relative position
    dx = obj2.x - obj1.x
    dy = obj2.y - obj1.y

    # Calculate the distance between the objects
    distance = math.sqrt(dx ** 2 + dy ** 2)

    # Normalize the distance vector
    nx = dx / distance
    ny = dy / distance

    # Relative velocity in terms of the normalized direction
    vel_along_normal = rvx * nx + rvy * ny

    # Do not resolve if velocities are separating
    if vel_along_normal > 0:
        return

    # Calculate size-based speed multiplier
    size_multiplier1 = max(1, 1 / (obj1.h2 * obj1.w2 * size_speed_factor))
    size_multiplier2 = max(1, 1 / (obj2.h2 * obj2.w2 * size_speed_factor))

    # Calculate impulse scalar with increased momentum
    j = -(1 + momentum_factor) * vel_along_normal
    j /= 1 / obj1.mass + 1 / obj2.mass

    # Apply impulse with size-based speed adjustments
    impulse_x = j * nx
    impulse_y = j * ny
    obj1.vx -= impulse_x / obj1.mass * size_multiplier1
    obj1.vy -= impulse_y / obj1.mass * size_multiplier1
    obj2.vx += impulse_x / obj2.mass * size_multiplier2
    obj2.vy += impulse_y / obj2.mass * size_multiplier2

    # Clamp velocities
    obj1.vx = max(min(obj1.vx, max_speed), -max_speed)
    obj1.vy = max(min(obj1.vy, max_speed), -max_speed)
    obj2.vx = max(min(obj2.vx, max_speed), -max_speed)
    obj2.vy = max(min(obj2.vy, max_speed), -max_speed)

    # Ensure minimum speed
    for obj in (obj1, obj2):
        speed = math.sqrt(obj.vx**2 + obj.vy**2)
        if speed < min_speed:
            scale = min_speed / speed
            obj.vx *= scale
            obj.vy *= scale

class Game:
    def __init__(self, **config):
        self.config = config
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption('Self play simulation')

        self.clock = pygame.time.Clock()
        self.ball = Ball(b_conf)
        p_conf['x'] = PADDLE_WIDTH * 1.5
        p_conf['color'] = BLUE
        self.left_paddle = Paddle(p_conf)
        p_conf['x'] = SCREEN_WIDTH - PADDLE_WIDTH * 1.5
        p_conf['color'] = GREEN
        self.right_paddle = Paddle(p_conf)
        
        self.obstacles = []
        for _ in range(NUM_OBSTACLES):
            new_o = Obstacle(o_conf)
            self.obstacles.append(new_o)
        
        self.running = True
        self.play_game()

    def get_player_moves(self):
        # Paddle movement
        keys = pygame.key.get_pressed()
        self.left_paddle.vy += PADDLE_VEL * (-keys[pygame.K_w] + keys[pygame.K_s])
        self.right_paddle.vy += PADDLE_VEL * (-keys[pygame.K_UP] + keys[pygame.K_DOWN])

    def get_ai_moves(self):
        bh = self.ball.y
        fbh = self.ball.vy * 8 + bh
        lph = self.left_paddle.y
        rph = self.right_paddle.y
        buffer = 40
        self.left_paddle.vy += PADDLE_VEL * (-(lph + buffer > fbh) + (lph -buffer < fbh))
        self.right_paddle.vy += PADDLE_VEL * (-(rph + buffer > bh) + (rph -buffer < bh))

    def move_objects(self):
        self.left_paddle.vy *= .9
        self.right_paddle.vy *= .9
        if self.config['players']:
            self.get_player_moves()
        else:
            self.get_ai_moves()
        self.left_paddle.move()
        self.right_paddle.move()
        self.ball.move()
        for obs in self.obstacles:
            obs.move()

    def check_for_score(self, score):
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
        return score
    
    def detect_collision(self):
        # Collision with left paddle
        if self.left_paddle.collides(self.ball):
            collide(self.left_paddle, self.ball)
            self.ball.vx = max(2, abs(self.ball.vx))

        # Collision with right paddle
        if self.right_paddle.collides(self.ball):
            collide(self.right_paddle, self.ball)
            self.ball.vx = -max(2, abs(self.ball.vx))

        for i, obstacle in enumerate(self.obstacles):
            if obstacle.collides(self.ball):
                collide(obstacle, self.ball)
            
            for ind, other in enumerate(self.obstacles):
                if ind == i:
                    continue
                if obstacle.collides(other):
                    collide(obstacle, other)

        # Ball out of bounds       
        if self.ball.y < 0:
            self.ball.vy *= -1

        if self.ball.y > SCREEN_HEIGHT:
            self.ball.vy *= -1

    def redraw_screen(self):
        # Drawing
        self.screen.fill(BLACK)
        self.ball.draw(self.screen)
        self.left_paddle.draw(self.screen)
        self.right_paddle.draw(self.screen)
        for obstacle in self.obstacles:
            obstacle.draw(self.screen)

        pygame.draw.rect(self.screen, (255,255,255), (SCREEN_WIDTH // 2 - OBSTACLE_SPREAD, 0, 1, SCREEN_HEIGHT))
        pygame.draw.rect(self.screen, (255,255,255), (SCREEN_WIDTH // 2 + OBSTACLE_SPREAD, 0, 1, SCREEN_HEIGHT))

    def display_variables(self, score):
        ball_position_text = font.render(f'Ball Position: ({self.ball.x//10*10}, {self.ball.y//10*10})', True, WHITE)
        ball_velocity_text = font.render(f'Ball Velocity: ({rr(self.ball.vx)}, {rr(self.ball.vy)})', True, WHITE)
        left_paddle_text = font.render(f'Left Paddle Y: {rr(self.left_paddle.y)}', True, WHITE)
        right_paddle_text = font.render(f'Right Paddle Y: {rr(self.right_paddle.y)}', True, WHITE)
        score_text = big_font.render(f'P1: {score["p1"]} | P2: {score["p2"]}', True, WHITE)

        self.screen.blit(ball_position_text, (SCREEN_WIDTH//2 - 100, SCREEN_HEIGHT - 40))
        self.screen.blit(ball_velocity_text, (SCREEN_WIDTH//2 - 100, SCREEN_HEIGHT - 80))
        self.screen.blit(score_text, (SCREEN_WIDTH//2 - 50, 10))
        self.screen.blit(left_paddle_text, (10, 10))
        self.screen.blit(right_paddle_text, (SCREEN_WIDTH - 300, 10))

    def step(self, score):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
        self.move_objects()
        self.detect_collision()
        self.redraw_screen()
        self.display_variables(score)
        return self.check_for_score(score)

    def play_game(self):
        score = {
            'p1': 0,  
            'p2': 0
        }
        while self.running:
            score = self.step(score)     
            pygame.display.flip()
            self.clock.tick(60)
        pygame.quit()

if __name__ == "__main__":
    g = Game(**{'players': False})
