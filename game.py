import pygame
import random
import math

# Initialize Pygame
pygame.init()

# Set up the game window
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pong with Obstacles")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)

# Game objects
PADDLE_WIDTH, PADDLE_HEIGHT = 15, 90
BALL_RADIUS = 10
OBSTACLE_WIDTH, OBSTACLE_HEIGHT = 20, 100

# Paddle class
class Paddle:
    def __init__(self, x, y, color):
        self.rect = pygame.Rect(x, y, PADDLE_WIDTH, PADDLE_HEIGHT)
        self.color = color
        self.speed = 5

    def move(self, up=True):
        if up:
            self.rect.y -= self.speed
        else:
            self.rect.y += self.speed
        self.rect.clamp_ip(screen.get_rect())

    def draw(self):
        pygame.draw.rect(screen, self.color, self.rect)

# Ball class
class Ball:
    def __init__(self):
        self.reset()
        self.speed_increase = 1.001

    def reset(self):
        self.x = WIDTH // 2
        self.y = HEIGHT // 2
        self.speed_x = random.choice([-4, 4])
        self.speed_y = random.choice([-4, 4])

    def move(self):
        self.x += self.speed_x
        self.y += self.speed_y
        self.speed_x *= self.speed_increase
        self.speed_y *= self.speed_increase

    def draw(self):
        pygame.draw.circle(screen, WHITE, (int(self.x), int(self.y)), BALL_RADIUS)

# Obstacle class
class Obstacle:
    def __init__(self, x, y):
        self.rect = pygame.Rect(x, y, OBSTACLE_WIDTH, OBSTACLE_HEIGHT)

    def draw(self):
        pygame.draw.rect(screen, GREEN, self.rect)

# Create game objects
player_paddle = Paddle(50, HEIGHT // 2 - PADDLE_HEIGHT // 2, RED)
ai_paddle = Paddle(WIDTH - 50 - PADDLE_WIDTH, HEIGHT // 2 - PADDLE_HEIGHT // 2, BLUE)
ball = Ball()

# Create obstacles
obstacles = [
    Obstacle(WIDTH // 2 - OBSTACLE_WIDTH // 2, HEIGHT // 4 - OBSTACLE_HEIGHT // 2),
    Obstacle(WIDTH // 2 - OBSTACLE_WIDTH // 2, 3 * HEIGHT // 4 - OBSTACLE_HEIGHT // 2)
]

def handle_collision(paddle):
    offset = (ball.y - paddle.rect.centery) / (PADDLE_HEIGHT / 2)
    ball.speed_x = -ball.speed_x
    ball.speed_y = 7 * offset

def check_collisions():
    # Ball collision with top and bottom
    if ball.y <= BALL_RADIUS or ball.y >= HEIGHT - BALL_RADIUS:
        ball.speed_y = -ball.speed_y

    # Ball collision with paddles
    if ball.x <= player_paddle.rect.right + BALL_RADIUS and player_paddle.rect.top - BALL_RADIUS <= ball.y <= player_paddle.rect.bottom + BALL_RADIUS:
        handle_collision(player_paddle)
    elif ball.x >= ai_paddle.rect.left - BALL_RADIUS and ai_paddle.rect.top - BALL_RADIUS <= ball.y <= ai_paddle.rect.bottom + BALL_RADIUS:
        handle_collision(ai_paddle)

    # Ball collision with obstacles
    for obstacle in obstacles:
        if obstacle.rect.collidepoint(ball.x, ball.y):
            if abs(ball.x - obstacle.rect.centerx) > abs(ball.y - obstacle.rect.centery):
                ball.speed_x = -ball.speed_x
            else:
                ball.speed_y = -ball.speed_y

def draw_objects():
    screen.fill(BLACK)
    player_paddle.draw()
    ai_paddle.draw()
    ball.draw()
    for obstacle in obstacles:
        obstacle.draw()
    pygame.display.flip()

def get_game_state():
    return {
        'player_top': player_paddle.rect.top,
        'player_bottom': player_paddle.rect.bottom,
        'ai_top': ai_paddle.rect.top,
        'ai_bottom': ai_paddle.rect.bottom,
        'ball_x': ball.x,
        'ball_y': ball.y,
        'ball_speed_x': abs(ball.speed_x),
        'ball_speed_y': abs(ball.speed_y)
    }

def run_game(get_ai_action):
    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            player_paddle.move(up=True)
        if keys[pygame.K_s]:
            player_paddle.move(up=False)

        # AI paddle movement
        ai_action = get_ai_action(get_game_state())
        if ai_action == 1:
            ai_paddle.move(up=True)
        elif ai_action == 2:
            ai_paddle.move(up=False)

        ball.move()
        check_collisions()

        # Check for scoring
        if ball.x <= 0 or ball.x >= WIDTH:
            ball.reset()

        draw_objects()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    # For testing without AI
    def dummy_ai_action(state):
        return random.choice([0, 1, 2])

    run_game(dummy_ai_action)