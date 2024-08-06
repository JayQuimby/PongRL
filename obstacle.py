import pygame
import random
import math
from global_vars import (
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    OBSTACLE_RADIUS,
    OBSTACLE_SPREAD,
    RED
)

class Obstacle:
    def __init__(self, config):
        self.x = 0
        self.y = 0
        self.vx = 0
        self.vy = 0
        self.mass = 0

        for key, value in config.items():
            setattr(self, key, value)
        self.h2 = OBSTACLE_RADIUS
        self.w2 = OBSTACLE_RADIUS
        self.reset()
        
    def reset(self):
        self.x = SCREEN_WIDTH // 2 + (2 * random.random() - 1) * random.randint(OBSTACLE_SPREAD//2, OBSTACLE_SPREAD)
        self.y = random.randint(self.h2, SCREEN_HEIGHT - self.h2)
        self.vy = random.choice([-3, -2, 2, 3])
        self.vx = random.choice([-3, -2, 2, 3])

    def collides(self, other):
        left = self.x - self.w2
        right = self.x + self.w2
        top = self.y + self.h2
        bot = self.y - self.h2
        horiz = (left < other.x + other.w2 < right) | (left < other.x - other.w2 < right)
        vert = (bot < other.y + other.h2 < top) | (bot < other.y - other.h2 < top)
        return  horiz & vert

    def move(self):
        self.y += self.vy
        self.x += self.vx

        #check too low
        if self.y - self.h2 <= 0:
            self.x += 5
            if self.vy < 0:
                self.vy *= -1 
            self.vx *= (1 + random.random()) * 2 / 4

        # check too high
        if self.y + self.h2 >= SCREEN_HEIGHT:
            self.x -= 5 
            if self.vy > 0:
                self.vy *= -1 
            self.vx *= (1 + random.random()) * 2 / 4

        #check too far left
        if self.x - self.w2 < SCREEN_WIDTH // 2 - OBSTACLE_SPREAD:
            #self.x += 5
            if self.vx < 0:
                self.vx *= -1
            self.vy *= (1 + random.random()) * 2 / 4

        #check too far right
        if self.x + self.w2 > SCREEN_WIDTH // 2 + OBSTACLE_SPREAD:
            #self.x -= 5
            if self.vx > 0:
                self.vx *= -1
            self.vy *= (1 + random.random()) * 2 / 4

        if self.vx < 2:
            self.vx /= (abs(self.vx)/2)

    def velocity(self):
        return math.hypot(self.vx, self.vy)

    def draw(self, screen):
        pygame.draw.circle(screen, RED, (self.x, self.y), OBSTACLE_RADIUS)
        pygame.draw.circle(screen, (0,0,0), (self.x, self.y), 10)
