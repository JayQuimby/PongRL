import pygame
import numpy as np
from global_vars import *
from objects import Ball, Paddle, Obstacle
from agent import PongAgent
from utils import load_conf, distance, collide
p_conf = load_conf('paddle')
b_conf = load_conf('ball')
o_conf = load_conf('obstacle')
rr = lambda x: round(x, 1)
verify = lambda x: x['p1'] < MAX_SCORE and x['p2'] < MAX_SCORE

class GameDisplay:
    def __init__(self, game):
        self.game = game
        
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT + UI_SIZE))
        pygame.display.set_caption('Self play simulation')
        self.clock = pygame.time.Clock()
        self.fps = 60

        pygame.font.init()
        self.font = pygame.font.Font(None, 28)
        self.big_font = pygame.font.Font(None, 48)
        self.small_font = pygame.font.Font(None, 18)

        self.step = 0

    def step_itter(self):
        self.step = (self.step + 1) % UI_SAMPLE_RATE

    def render_text(self, text, font, color=WHITE):
        return font.render(text, True, color)

    def blit_text(self, text, position):
        self.screen.blit(text, position)

    def display_variables(self, score, rewards):
        self.screen.fill(BLACK, pygame.Rect(0,SCREEN_HEIGHT, SCREEN_WIDTH, UI_SIZE))
        variables = [
            (f'Ball Position: ({self.game.ball.x // 10 * 10}, {self.game.ball.y // 10 * 10})', self.font, (MID_WIDTH + 50, SCREEN_HEIGHT + 10)),
            (f'Ball Velocity: ({rr(self.game.ball.vx)}, {rr(self.game.ball.vy)})', self.font, (MID_WIDTH - 210, SCREEN_HEIGHT + 10)),
            (f'Left Paddle Y: {rr(self.game.paddles[0].y)}', self.font, (10, SCREEN_HEIGHT + 10)),
            (f'Right Paddle Y: {rr(self.game.paddles[1].y)}', self.font, (SCREEN_WIDTH - 210, SCREEN_HEIGHT + 10)),
            (f'P1: {score["p1"]} | P2: {score["p2"]}', self.big_font, (MID_WIDTH - 70, SCREEN_HEIGHT + 55)),
            (f'fps: {self.fps}', self.small_font, (0, SCREEN_HEIGHT), (255, 255, 0)),
            (f'P1 Reward: {rr(rewards[0])}', self.font, (20, SCREEN_HEIGHT + 55)),
            (f'P2 Reward: {rr(rewards[1])}', self.font, (SCREEN_WIDTH - 160, SCREEN_HEIGHT + 55)),
        ]

        for text, font, position, *color in variables:
            color = color[0] if color else WHITE
            rendered_text = self.render_text(text, font, color)
            self.blit_text(rendered_text, position)
        pygame.draw.rect(self.screen, WHITE, (0, SCREEN_HEIGHT, SCREEN_WIDTH, 1))
        pygame.draw.rect(self.screen, WHITE, (0, SCREEN_HEIGHT + 40, SCREEN_WIDTH, 1))

    def redraw_screen(self):
        # Drawing
        self.screen.fill(BLACK, pygame.Rect(0,0, SCREEN_WIDTH, SCREEN_HEIGHT))

        pygame.draw.rect(self.screen, (255,255,255), (MID_WIDTH - OBSTACLE_SPREAD, 0, 1, SCREEN_HEIGHT))
        pygame.draw.rect(self.screen, (255,255,255), (MID_WIDTH + OBSTACLE_SPREAD, 0, 1, SCREEN_HEIGHT))

        self.game.ball.draw(self.screen)
        self.game.paddles[0].draw(self.screen)
        self.game.paddles[1].draw(self.screen)
        for obstacle in self.game.obstacles:
            obstacle.draw(self.screen)

    def update_display(self, s, r):
        if self.step % SCREEN_SAMPLE_RATE == 0:
            self.redraw_screen()
        if self.step == 0:
            self.display_variables(s, r)
        pygame.display.flip()
        self.clock.tick(self.fps)
        self.step_itter()

    def inc_fps(self, neg=False):
        self.fps -= 1 if neg else -1

class Game:
    def __init__(self, **config):
        pygame.init()
        self.players = config.get('players', 0)
        self.nn = config.get('nn', 0)
        self.train = config.get('training', False)
        self.games = config.get('num_games', 1)
        self.save = config.get('save_prog', False)
        self.read_out = GameDisplay(self)

        self.ball = Ball(b_conf)
        
        self.paddles = []
        p_conf['x'] = PADDLE_RADIUS * 2
        p_conf['color'] = BLUE
        self.paddles.append(Paddle(p_conf))

        p_conf['x'] = SCREEN_WIDTH - PADDLE_RADIUS * 2
        p_conf['color'] = GREEN
        self.paddles.append(Paddle(p_conf))
        
        self.obstacles = []
        for _ in range(NUM_OBSTACLES):
            new_o = Obstacle(o_conf)
            self.obstacles.append(new_o)

        self.cur_state = [self.get_state(0), self.get_state(1)]
        self.nn_model = None
        if self.nn:
            self.nn_model = PongAgent()
        self.running = True
        self.run()
        pygame.quit()

    def get_player_moves(self, keys):
        # Paddle movement
        actions = []
        
        if self.players > 0:
            act = [0, 0, 0, 0, 0]
            up_down = (-keys[pygame.K_w] + keys[pygame.K_s])
            left_right = (-keys[pygame.K_a] + keys[pygame.K_d])
            
            if up_down or left_right:
                act[0] = up_down > 0  # Move up
                act[1] = up_down < 0  # Move dow
                act[2] = left_right > 0  # Move left
                act[3] = left_right < 0  # Move right
                self.paddles[0].vy += PADDLE_VEL * up_down
                self.paddles[0].vx += PADDLE_VEL * left_right
            if not any(act[:4]):
                act[4] = 1  # None
            actions.append(act)

        if self.players > 1:
            act = [0, 0, 0, 0, 0]
            up_down = (-keys[pygame.K_UP] + keys[pygame.K_DOWN])
            left_right = (-keys[pygame.K_LEFT] + keys[pygame.K_RIGHT])
            
            if up_down or left_right:
                act[0] = up_down > 0  # Move up
                act[1] = up_down < 0  # Move dow
                act[2] = left_right > 0  # Move left
                act[3] = left_right < 0  # Move right
                self.paddles[0].vy += PADDLE_VEL * up_down
                self.paddles[0].vx += PADDLE_VEL * left_right
            if not any(act[:4]):
                act[4] = 1  # None
            actions.append(act)

        return actions

    def get_state(self, rev=False):
        inputs = []
        paddle, op_paddle = self.paddles[::-1] if rev else self.paddles
        # Ball position and velocity
        inputs.extend([self.ball.x / SCREEN_WIDTH, self.ball.y / SCREEN_HEIGHT])
        inputs.extend([self.ball.vx / BALL_MAX_SPEED, self.ball.vy / BALL_MAX_SPEED])
        
        # Paddle position and velocity
        inputs.extend([paddle.x / SCREEN_WIDTH, paddle.y / SCREEN_HEIGHT])
        inputs.extend([paddle.vx / PADDLE_VEL, paddle.vy / PADDLE_VEL])

        inputs.extend([op_paddle.x / SCREEN_WIDTH, op_paddle.y / SCREEN_HEIGHT])
        inputs.extend([op_paddle.vx / PADDLE_VEL, op_paddle.vy / PADDLE_VEL])
        # Encode obstacles positions
        grid_width = SCREEN_WIDTH // MIDGROUND_SPLIT
        grid_height = SCREEN_HEIGHT // MIDGROUND_SPLIT
        obstacle_grid = [0] * (grid_width * grid_height)
        
        shift = MID_WIDTH - OBSTACLE_SPREAD

        for obstacle in self.obstacles:
            grid_x = int((obstacle.x - shift) // MIDGROUND_SPLIT)
            grid_y = int(obstacle.y // MIDGROUND_SPLIT)
            index = grid_y * grid_width + grid_x
            obstacle_grid[index] = 1
        
        inputs.extend(obstacle_grid)
        
        return np.reshape(inputs, [1, len(inputs)])

    def get_ai_moves(self):
        bh = self.ball.y
        vl = self.ball.vy
        fbh = bh + vl**2
        buffer = PADDLE_RADIUS // 8
        follow = lambda x: PADDLE_VEL * (-(x + buffer > fbh) + (x -  buffer < fbh))
        center = lambda x: PADDLE_VEL * min(1, max(-1, (MID_HEIGHT - x)//10))

        def update_paddle(rev, direction, x_condition, follow_action, center_action):
            act = [0 for _ in range(5)]
            paddle = self.paddles[rev]
            if x_condition:
                fdy = follow_action(paddle.y)
                act[0] = 1 if fdy > 0 else 0
                act[1] = 1 if fdy < 0 else 0
                paddle.vy += fdy
                if distance(paddle, self.ball) < paddle.r * 3:
                    adx = PADDLE_VEL // 2 * direction
                    act[2] = 1 if adx < 0 else 0
                    act[3] = 1 if adx > 0 else 0
                    paddle.vx += adx
            else:
                ddy = center_action(paddle.y)
                ddx = PADDLE_VEL * direction
                act[0] = 1 if ddy > 0 else 0
                act[1] = 1 if ddy < 0 else 0
                act[2] = 1 if ddx < 0 else 0
                act[3] = 1 if ddx > 0 else 0
                paddle.vx -= ddx
                paddle.vy += ddy
            act[-1] = sum(act) == 0
            return act

        def nn_play(rev=False):
            model_input = self.get_state(rev)
            action = self.nn_model.act(model_input)
            paddle = self.paddles[rev]
            if action[0]:  # Up
                paddle.vy -= PADDLE_VEL
            elif action[1]:  # Down
                paddle.vy += PADDLE_VEL
            elif action[2]:  # Left
                paddle.vx -= PADDLE_VEL
            elif action[3]:  # Right
                paddle.vx += PADDLE_VEL
            return action

        bot_actions = []
        if self.players == 0:
            if self.nn == 0: 
                bot_actions.append(update_paddle(0, 1, self.ball.vx < 0 and self.ball.x < MID_WIDTH, follow, center))
                bot_actions.append(update_paddle(1, -1, self.ball.vx > 0 and self.ball.x > MID_WIDTH, follow, center))
            elif self.nn == 1:
                bot_actions.append(nn_play(0))
                bot_actions.append(update_paddle(1, -1, self.ball.vx > 0 and self.ball.x > MID_WIDTH, follow, center))
            else:
                if self.ball.x < MID_WIDTH:
                    bot_actions.append(nn_play(0))
                    bot_actions.append(NULL_ACT)
                else:
                    bot_actions.append(NULL_ACT)
                    bot_actions.append(nn_play(1))

        elif self.players == 1:
            if self.nn > 0:
                bot_actions.append(nn_play(1))
            else:
                bot_actions.append(update_paddle(1, -1, self.ball.vx > 0 and self.ball.x > MID_WIDTH, follow, center))
        
        return bot_actions
    
    def resistance(self):
        ai_p1 = AI_MOVE_EDGE * (self.players == 0 and self.nn >= 1)
        ai_p2 = AI_MOVE_EDGE * (self.players == 0 and self.nn == 2 or self.players == 1 and self.nn == 1)
        self.paddles[0].vy *= PADDLE_MOVE_DECAY + ai_p1
        self.paddles[0].vx *= PADDLE_MOVE_DECAY + ai_p1
        self.paddles[1].vy *= PADDLE_MOVE_DECAY + ai_p2
        self.paddles[1].vx *= PADDLE_MOVE_DECAY + ai_p2

    def toggle_keys(self, keys):
        if keys[pygame.K_COMMA]: self.read_out.inc_fps(1)
        if keys[pygame.K_PERIOD]: self.read_out.inc_fps()

    def move_objects(self):
        keys = pygame.key.get_pressed()
        self.resistance()
        self.toggle_keys(keys)

        if self.players > 0:
            player_moves = self.get_player_moves(keys)

        if self.players < 2: 
            bot_moves = self.get_ai_moves()
            print(len(bot_moves))

        self.paddles[0].move()
        self.paddles[1].move()
        self.ball.move()

        for obs in self.obstacles:
            obs.move()
        return player_moves + bot_moves

    def check_for_score(self, score, dr):
        p1_score = self.ball.x > SCREEN_WIDTH
        p2_score = self.ball.x < 0
        p1_win = False
        p2_win = False
        if p1_score or p2_score:
            if p1_score:
                score['p1'] += 1
                self.running = verify(score)
                p1_win = not self.running
            else:
                score['p2'] += 1
                self.running = verify(score)
                p2_win = not self.running

            self.ball.reset()
            self.paddles[0].reset()
            self.paddles[1].reset()
            for ob in self.obstacles:
                ob.reset()

        p1_reward = (self.ball.x - MID_WIDTH) / SCREEN_WIDTH + SCORE_REWARD_MULT * (p1_score - p2_score) + WIN_REWARD * (p1_win - p2_win)
        return score, (dr[0] + p1_reward, dr[1] - p1_reward)
    
    def obstacle_collision(self):
        for i, obstacle in enumerate(self.obstacles):
            if obstacle.collides(self.ball):
                collide(obstacle, self.ball)
            if i < len(self.obstacles):
                for other in self.obstacles[i+1:]:
                    if obstacle.collides(other):
                        collide(obstacle, other)

    def paddle_collision(self, rewards):
        for paddle, rew in zip(self.paddles, rewards):
            if paddle.collides(self.ball):
                rew += 2
                paddle.hit = True
                collide(paddle, self.ball)
            else:
                paddle.hit = False
        return rewards

    def detect_collision(self, score):
        self.obstacle_collision()
        rewards = self.paddle_collision([0,0])
        return self.check_for_score(score, rewards)

    def reset(self):
        self.ball.reset()
        for p in self.paddles:
            p.reset()
        for o in self.obstacles:
            o.reset()
        self.running = True

    def step(self, score):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
        moves = self.move_objects()
        score, rewards = self.detect_collision(score)
        self.read_out.update_display(score, rewards)
        return score, rewards, moves

    def run(self):
        for game_num in range(self.games):
            score = {'p1': 0, 'p2': 0}

            while self.running:
                score, rewards, actions = self.step(score)
                
                if self.train:
                    next_state = [self.get_state(0), self.get_state(1)]
                    for x in [0, 1]:
                        self.nn_model.remember(self.cur_state[x], actions[x], rewards[x], next_state[x], self.running)
                    self.cur_state = next_state
            self.reset()
            if self.train:
                self.nn_model.replay()
        if self.train and self.save:
            self.nn_model.save(MODEL_PATH)

if __name__ == "__main__":
    conf = {
        'players': 1,
        'nn': 1,
        'training': True,
        'save_prog': True,
        'num_games': 20
    }
    g = Game(**conf)
    
    

