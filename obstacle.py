import pygame
import random
from global_vars import (
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    OBSTACLE_HEIGHT,
    OBSTACLE_WIDTH,
    OBSTACLE_SPREAD,
    RED,
)

class Obstacle:
    def __init__(self):
        self.reset()
        self.h2 = OBSTACLE_HEIGHT // 2
        self.w2 = OBSTACLE_WIDTH // 2

    def reset(self):
        self.x = SCREEN_WIDTH // 2 + (2 * random.random() - 1) * random.randint(OBSTACLE_SPREAD//2, OBSTACLE_SPREAD)
        self.y = random.randint(self.h2, SCREEN_HEIGHT - self.h2)
        self.vy = random.choice([-3, -2, 2, 3])
        self.vx = random.choice([-3, -2, 2, 3])

    def collides(self, ball):
        left = self.x - self.w2
        right = self.x + self.w2
        top = self.y + self.h2
        bot = self.y - self.h2
        return (left < ball.x < right) & (bot < ball.y < top)

    def move(self):

        if random.random() < 0.01:
            if random.random() > 0.5:
                self.vx *= 0.98
                self.vy *= 0.98
            else:
                self.vx *= 1.02
                self.vy *= 1.02


        self.y += self.vy

        if self.y - self.h2 <= 0 or self.y + self.h2 >= SCREEN_HEIGHT:
            self.vy *= -1
            self.vx *= (1 + random.random()) * 2 / 3

        self.x += self.vx

        if self.x < SCREEN_WIDTH // 2 - OBSTACLE_SPREAD or self.x > SCREEN_WIDTH // 2 + OBSTACLE_SPREAD:
            self.vx *= -1
            self.vy *= (1 + random.random()) * 2 / 3

    def draw(self, screen):
        pygame.draw.rect(screen, RED, (self.x - self.w2, self.y - self.h2, OBSTACLE_WIDTH, OBSTACLE_HEIGHT))