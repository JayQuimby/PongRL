import pygame
import random
from global_vars import (
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    OBSTACLE_HEIGHT,
    OBSTACLE_WIDTH,
    RED
)

class Obstacle:
    def __init__(self):
        self.reset()

    def reset(self):
        self.x = SCREEN_WIDTH // 2
        self.y = random.randint(OBSTACLE_HEIGHT // 2, SCREEN_HEIGHT - OBSTACLE_HEIGHT // 2)
        self.vy = random.choice([-3, 3])

    def move(self):
        self.y += self.vy

        if self.y - OBSTACLE_HEIGHT // 2 <= 0 or self.y + OBSTACLE_HEIGHT // 2 >= SCREEN_HEIGHT:
            self.vy *= -1

    def draw(self, screen):
        pygame.draw.rect(screen, RED, (self.x - OBSTACLE_WIDTH // 2, self.y - OBSTACLE_HEIGHT // 2, OBSTACLE_WIDTH, OBSTACLE_HEIGHT))