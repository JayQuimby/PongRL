import pygame
import random
from global_vars import *

from ball import Ball
from paddle import Paddle
from obstacle import Obstacle

# Initialize Pygame
pygame.init()

# Main game function
def main():
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption('Self play simulation')

    clock = pygame.time.Clock()
    ball = Ball()
    left_paddle = Paddle(PADDLE_WIDTH // 2, BLUE)
    right_paddle = Paddle(SCREEN_WIDTH - PADDLE_WIDTH // 2, GREEN)
    obstacle = Obstacle()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Paddle movement
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            left_paddle.vy = -5
        elif keys[pygame.K_s]:
            left_paddle.vy = 5
        else:
            left_paddle.vy = 0

        if keys[pygame.K_UP]:
            right_paddle.vy = -5
        elif keys[pygame.K_DOWN]:
            right_paddle.vy = 5
        else:
            right_paddle.vy = 0

        left_paddle.move()
        right_paddle.move()
        ball.move()
        obstacle.move()

        # Ball collision with paddles
        if (left_paddle.x + PADDLE_WIDTH // 2 >= ball.x - BALL_RADIUS and
            left_paddle.y - PADDLE_HEIGHT // 2 <= ball.y <= left_paddle.y + PADDLE_HEIGHT // 2):
            ball.vx *= -BALL_SPEED_INCREMENT
            ball.vy = (ball.y - left_paddle.y) / (PADDLE_HEIGHT // 2) * ball.vy * BALL_SPEED_INCREMENT

        if (right_paddle.x - PADDLE_WIDTH // 2 <= ball.x + BALL_RADIUS and
            right_paddle.y - PADDLE_HEIGHT // 2 <= ball.y <= right_paddle.y + PADDLE_HEIGHT // 2):
            ball.vx *= -BALL_SPEED_INCREMENT
            ball.vy = (ball.y - right_paddle.y) / (PADDLE_HEIGHT // 2) * ball.vy * BALL_SPEED_INCREMENT

        # Ball collision with obstacle
        if (obstacle.x - OBSTACLE_WIDTH // 2 <= ball.x + BALL_RADIUS <= obstacle.x + OBSTACLE_WIDTH // 2 and
            obstacle.y - OBSTACLE_HEIGHT // 2 <= ball.y <= obstacle.y + OBSTACLE_HEIGHT // 2):
            ball.vx *= -1
            ball.vy *= -1

        # Ball out of bounds
        if ball.x < 0 or ball.x > SCREEN_WIDTH:
            ball.reset()
            left_paddle.y = SCREEN_HEIGHT // 2
            right_paddle.y = SCREEN_HEIGHT // 2

        # Drawing
        screen.fill(BLACK)
        ball.draw(screen)
        left_paddle.draw(screen)
        right_paddle.draw(screen)
        obstacle.draw(screen)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
