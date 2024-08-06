import pygame
import random

from global_vars import (
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    BALL_RADIUS,
    BALL_MAX_SPEED,
    WHITE
)

# Game objects
class Ball:
    def __init__(self):
        self.reset()
        self.w2 = BALL_RADIUS
        self.h2 = BALL_RADIUS

    def reset(self):
        self.x = SCREEN_WIDTH // 2
        self.y = SCREEN_HEIGHT // 2
        self.vx = random.choice([-4, 4])
        self.vy = random.choice([-2, 2])

    def move(self):
        if abs(self.vx) < BALL_MAX_SPEED:
            self.x += self.vx

        if abs(self.vy) < BALL_MAX_SPEED:
            self.y += self.vy

        if self.y - BALL_RADIUS <= 0 or self.y + BALL_RADIUS >= SCREEN_HEIGHT:
            self.vy *= -1

    def draw(self, screen):
        pygame.draw.circle(screen, WHITE, (self.x, self.y), BALL_RADIUS)