import pygame
import math

from global_vars import (
    PADDLE_HEIGHT,
    SCREEN_HEIGHT,
    PADDLE_WIDTH,
    BALL_RADIUS,
)

max_bounce_angle = math.radians(60)

class Paddle:
    def __init__(self, config):
        self.x = 0
        self.vx = 0
        self.vy = 0
        self.mass = 0
        self.bounciness = 0
        self.slipperiness = 0
        self.color = (0,0,0)
        
        for key, value in config.items():
            setattr(self, key, value)
        self.h2 = PADDLE_HEIGHT // 2
        self.w2 = PADDLE_WIDTH // 2
        self.reset()

    def reset(self):
        self.y = SCREEN_HEIGHT // 2

    def collides(self, other):
        left = self.x - self.w2 - BALL_RADIUS
        right = self.x + self.w2 + BALL_RADIUS
        top = self.y + self.h2 + BALL_RADIUS
        bot = self.y - self.h2 - BALL_RADIUS
        return (left < other.x < right) & (bot < other.y < top)

    def move(self):
        self.y += self.vy
        self.y = max(self.h2, min(SCREEN_HEIGHT - self.h2, self.y))

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, (self.x - self.w2, self.y - self.h2, PADDLE_WIDTH, PADDLE_HEIGHT))
        pygame.draw.circle(screen, (0,0,0), (self.x, self.y), 5)