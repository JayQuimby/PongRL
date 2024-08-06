import pygame
import random
import math

from global_vars import (
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    BALL_RADIUS,
    BALL_MAX_SPEED,
    WHITE
)

# Game objects
class Ball:
    def __init__(self, config):
        self.x = 0
        self.y = 0
        self.vx = 0
        self.vy = 0
        self.mass = 0
        self.bounciness = 0
        self.slipperiness = 0

        for key, value in config.items():
            setattr(self, key, value)
        self.w2 = BALL_RADIUS
        self.h2 = BALL_RADIUS
        self.reset()

    def reset(self):
        self.x = SCREEN_WIDTH // 2
        self.y = SCREEN_HEIGHT // 2
        self.vx = random.choice([-4, 4])
        self.vy = random.choice([-1, 1])

    def move(self):
        if abs(self.vx) < BALL_MAX_SPEED:
            self.x += self.vx
        else:
            self.vx /= 2

        if abs(self.vy) < BALL_MAX_SPEED:
            self.y += self.vy
        else:
            self.vy /= 2

        if self.y - BALL_RADIUS <= 0:
            if self.vy < 0:
                self.vy *= -1

        if self.y + BALL_RADIUS >= SCREEN_HEIGHT:
            if self.vy > 0:
                self.vy *= -1

        if self.velocity() < 2:
            self.vx /= abs(self.vx)
            self.vy /= abs(self.vy)
        
        if self.vx < 4:
            self.vx /= (abs(self.vx)/4)

    def velocity(self):
        return math.hypot(self.vx, self.vy)

    def draw(self, screen):
        pygame.draw.circle(screen, WHITE, (self.x, self.y), BALL_RADIUS)
        pygame.draw.circle(screen, (215,35,245), (self.x, self.y), 5)
        