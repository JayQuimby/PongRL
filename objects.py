import pygame
import random
import math
from utils import distance, check_bounds
from static import *

class Ball:
    def __init__(self, config):
        self.x = 0
        self.y = 0
        self.vx = 0
        self.vy = 0
        self.mass = 0
        for key, value in config.items():
            setattr(self, key, value)
        self.r = BALL_RADIUS
        self.reset()
        self.last = (self.x, self.y)

    def reset(self):
        self.x = MID_WIDTH
        self.y = MID_HEIGHT
        self.vx = random.choice([-4, 4])
        self.vy = random.choice([-1, 1])

    def move(self):
        def correct_speed(pos, velo):
            if abs(velo) < BALL_MAX_SPEED:
                pos += velo
            else:
                velo /= 2
            return pos, velo

        self.x, self.vx = correct_speed(self.x, self.vx)
        self.y, self.vy = correct_speed(self.y, self.vy)
        
        self.vy *= check_bounds(self.y - BALL_RADIUS, 0, self.vy, False)
        self.vy *= check_bounds(self.y + BALL_RADIUS, SCREEN_HEIGHT, self.vy)

        if self.velocity() < 2:
            self.vx /= abs(self.vx)
            self.vy /= abs(self.vy)
        
        if abs(self.vx) < 4:
            self.vx /= abs(self.vx) / 4

        self.x = min(SCREEN_WIDTH, max(0, self.x))

    def velocity(self):
        return math.hypot(self.vx, self.vy)

    def draw(self, screen):
        pygame.draw.circle(screen, BLACK, (self.last[0], self.last[1]), self.r)
        pygame.draw.circle(screen, BALL_COLOR, (self.x, self.y), self.r)
        pygame.draw.circle(screen, WHITE, (self.x, self.y), 5)
        self.last = (self.x, self.y)

class Obstacle:
    def __init__(self, config):
        self.x = 0
        self.y = 0
        self.vx = 0
        self.vy = 0
        self.mass = 0

        for key, value in config.items():
            setattr(self, key, value)
        self.r = OBSTACLE_RADIUS
        self.reset()
        self.last = (self.x, self.y)
        
    def reset(self):
        self.x = MID_WIDTH + (2 * random.random() - 1) * random.randint(OBSTACLE_SPREAD//2, OBSTACLE_SPREAD)
        self.y = random.randint(self.r, SCREEN_HEIGHT - self.r)
        self.vy = random.choice([-3, -2, 2, 3])
        self.vx = random.choice([-3, -2, 2, 3])

    def collides(self, other):
        return distance(self, other) <= (self.r + other.r)

    def move(self):
        self.y += self.vy
        self.x += self.vx

        #if abs(self.vx) < 2:
            #self.vx = 2/(self.vx+1e-5)

        if self.vx > BALL_MAX_SPEED:
            self.vx = BALL_MAX_SPEED
        if self.vy > BALL_MAX_SPEED:
            self.vy = BALL_MAX_SPEED

        self.vy *= check_bounds(self.y - self.r, 0, self.vy, False) #check too low
        self.vy *= check_bounds(self.y + self.r, SCREEN_HEIGHT-1, self.vy) # check too high
        self.vx *= check_bounds(self.x - self.r, MID_WIDTH - OBSTACLE_SPREAD, self.vx, False) #check too far left
        self.vx *= check_bounds(self.x + self.r, MID_WIDTH + OBSTACLE_SPREAD, self.vx) #check too far right

    def velocity(self):
        return math.hypot(self.vx, self.vy)

    def draw(self, screen):
        pygame.draw.circle(screen, BLACK, (self.last[0], self.last[1]), self.r)
        pygame.draw.circle(screen, RED, (self.x, self.y), OBSTACLE_RADIUS)
        pygame.draw.circle(screen, WHITE, (self.x, self.y), 10)
        self.last = (self.x, self.y)

class Paddle:
    def __init__(self, config):
        self.x = 0
        self.y = 0
        self.vx = 0
        self.vy = 0
        self.mass = 0
        self.color = (0,0,0)
        self.hit = False
        for key, value in config.items():
            setattr(self, key, value)
        self.r = PADDLE_RADIUS
        self.base_x = self.x
        self.reset()
        self.last = (self.x, self.y)

    def reset(self):
        self.y = MID_HEIGHT
        self.x = self.base_x

    def collides(self, other):
        return distance(self, other) <= (self.r + other.r)

    def move(self):
        if abs(self.vx) > PADDLE_VEL:
            self.vx /= 2
        if abs(self.vy) > PADDLE_VEL:
            self.vy /= 2
        self.y += self.vy
        self.x += self.vx
        self.y = max(self.r, min(SCREEN_HEIGHT - self.r, self.y))
        self.x = max(min(self.x, self.base_x + self.r*2), self.base_x - self.r*2)

    def draw(self, screen):
        pygame.draw.circle(screen, BLACK, (self.last[0], self.last[1]), self.r)
        pygame.draw.circle(screen, self.color if not self.hit else RED, (self.x, self.y), PADDLE_RADIUS)
        pygame.draw.circle(screen, WHITE, (self.x, self.y), 5)
        self.last = (self.x, self.y)
        if self.hit:
            self.hit = False
