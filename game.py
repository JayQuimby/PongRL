import pygame
import math
from global_vars import *

from ball import Ball
from paddle import Paddle
from obstacle import Obstacle
pygame.init()
pygame.font.init()
font = pygame.font.Font(None, 28)

rr = lambda x: round(x, 1)

class Game:
    def __init__(self, **config):
        
        self.config = config
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption('Self play simulation')

        self.clock = pygame.time.Clock()
        self.ball = Ball()
        
        self.left_paddle = Paddle(PADDLE_WIDTH // 2, BLUE)
        self.right_paddle = Paddle(SCREEN_WIDTH - PADDLE_WIDTH * 1.5, GREEN)
        
        self.obstacles = []
        for _ in range(3):
            new_o = Obstacle()
            self.obstacles.append(new_o)
        
        self.running = True
        self.main()

    def get_player_moves(self):
        # Paddle movement
        keys = pygame.key.get_pressed()
        self.left_paddle.vy += PADDLE_VEL * (-keys[pygame.K_w] + keys[pygame.K_s])
        self.right_paddle.vy += PADDLE_VEL * (-keys[pygame.K_UP] + keys[pygame.K_DOWN])

    def get_ai_moves(self):
        bh = self.ball.y
        lph = self.left_paddle.y
        rph = self.right_paddle.y
        buffer = 30
        self.left_paddle.vy += PADDLE_VEL * (-(lph + buffer > bh) + (lph -buffer < bh))
        self.right_paddle.vy += PADDLE_VEL * (-(rph + buffer > bh) + (rph -buffer < bh))

    def move_objects(self):
        self.left_paddle.vy /= 2
        self.right_paddle.vy /= 2
        if self.config['players']:
            self.get_player_moves()
        else:
            self.get_ai_moves()
        self.left_paddle.move()
        self.right_paddle.move()
        self.ball.move()
        for obs in self.obstacles:
            obs.move()

    def detect_collision(self):
        # Ball collision with paddles
        pw2 = PADDLE_WIDTH // 2
        ph2 = PADDLE_HEIGHT // 2
        max_bounce_angle = math.radians(60)  # Maximum bounce angle in radians

        def calculate_new_velocity(ball_y, paddle_y, ball_speed):
            relative_intersect_y = (ball_y - paddle_y) / ph2
            bounce_angle = relative_intersect_y * max_bounce_angle
            new_vx = ball_speed * math.cos(bounce_angle)
            new_vy = ball_speed * math.sin(bounce_angle)
            return new_vx, new_vy

        ball_speed = math.hypot(self.ball.vx, self.ball.vy) * BALL_SPEED_INCREMENT

        # Collision with left paddle
        if (self.left_paddle.x + pw2 >= self.ball.x - BALL_RADIUS and
            self.left_paddle.y - ph2 <= self.ball.y <= self.left_paddle.y + ph2):
            self.ball.vx, self.ball.vy = calculate_new_velocity(self.ball.y, self.left_paddle.y, ball_speed)
            self.ball.vx = abs(self.ball.vx)  # Ensure the ball moves right

        # Collision with right paddle
        if (self.right_paddle.x - pw2 <= self.ball.x + BALL_RADIUS and
            self.right_paddle.y - ph2 <= self.ball.y <= self.right_paddle.y + ph2):
            self.ball.vx, self.ball.vy = calculate_new_velocity(self.ball.y, self.right_paddle.y, ball_speed)
            self.ball.vx = -abs(self.ball.vx)  # Ensure the ball moves left

        ball_speed = math.hypot(self.ball.vx, self.ball.vy) * OBSTACLE_FRICTION
        # Ball collision with obstacles

        for obstacle in self.obstacles:
            if obstacle.collides(self.ball):
                self.ball.vx, self.ball.vy = calculate_new_velocity(self.ball.y, obstacle.y, ball_speed)

        # Ball out of bounds
        if self.ball.x < 0 or self.ball.x > SCREEN_WIDTH:
            self.ball.reset()
            self.left_paddle.y = SCREEN_HEIGHT // 2
            self.right_paddle.y = SCREEN_HEIGHT // 2
        
        if self.ball.y < BALL_RADIUS:
            self.ball.y = BALL_RADIUS + 2
        if self.ball.y > SCREEN_HEIGHT - BALL_RADIUS:
            self.ball.y = SCREEN_HEIGHT - BALL_RADIUS - 2

    def redraw_screen(self):
        # Drawing
        self.screen.fill(BLACK)
        self.ball.draw(self.screen)
        self.left_paddle.draw(self.screen)
        self.right_paddle.draw(self.screen)
        for obstacle in self.obstacles:
            obstacle.draw(self.screen)

    def display_variables(self):
        ball_position_text = font.render(f'Ball Position: ({rr(self.ball.x)}, {rr(self.ball.y)})', True, WHITE)
        ball_velocity_text = font.render(f'Ball Velocity: ({rr(self.ball.vx)}, {rr(self.ball.vy)})', True, WHITE)
        left_paddle_text = font.render(f'Left Paddle Y: {rr(self.left_paddle.y)}', True, WHITE)
        right_paddle_text = font.render(f'Right Paddle Y: {rr(self.right_paddle.y)}', True, WHITE)

        self.screen.blit(ball_position_text, (SCREEN_WIDTH//2 - 100, 10))
        self.screen.blit(ball_velocity_text, (SCREEN_WIDTH//2 - 100, 30))
        self.screen.blit(left_paddle_text, (10, 10))
        self.screen.blit(right_paddle_text, (SCREEN_WIDTH - 300, 10))

    def step(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
        self.move_objects()
        self.detect_collision()
        self.redraw_screen()
        self.display_variables()

    def main(self):
        while self.running:
            self.step()        
            pygame.display.flip()
            self.clock.tick(60)
        pygame.quit()

if __name__ == "__main__":
    g = Game(**{'players': False})
