import pygame

from global_vars import (
    PADDLE_HEIGHT,
    SCREEN_HEIGHT,
    PADDLE_WIDTH
)

class Paddle:
    def __init__(self, x, color):
        self.x = x
        self.y = SCREEN_HEIGHT // 2
        self.vy = 0
        self.color = color

    def move(self):
        self.y += self.vy
        self.y = max(PADDLE_HEIGHT // 2, min(SCREEN_HEIGHT - PADDLE_HEIGHT // 2, self.y))

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, (self.x, self.y - PADDLE_HEIGHT // 2, PADDLE_WIDTH, PADDLE_HEIGHT))