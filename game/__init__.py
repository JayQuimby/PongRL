import pygame as pg
import numpy as np
import random
from game.static import *
from game.utils import get_obj_state_repr
from game.objects import Ball, Paddle, Obstacle
from agents.pytorch import PongAgent
# from agents.tiny import PongAgent
# from agents.tiny_2 import PongAgent
# from agents.flow import PongAgent
from game.utils import load_conf, distance, collide, timer, verify
from game.display import GameDisplay

from math import log10

p_conf = load_conf('paddle')
b_conf = load_conf('ball')
o_conf = load_conf('obstacle')

# TODO: fix the directions that the netork sees with P1 vs P2, not sure if this is an issue or not

class Game:
    def __init__(self, **config):
        pg.init()
        self.start_check: bool = True
        self.steps_taken: int = 0
        self.players: int = config.get('players', 0)
        self.nn: int = config.get('nn', 0)
        self.num_obs: int = config.get('obstacles', 0)
        self.slow_mo: bool = config.get('slow', False)
        self.games: int = config.get('num_games', 8)
        self.cur_game: int = 1
        
        if self.nn:
            # game settings
            self.cur_match: int = 0
            self.cur_set: int = 0
            self.num_matches: int = config.get('matches', 5)
            self.num_sets: int = config.get('sets', 10)
            self.total_games: int = self.games * self.num_matches * self.num_sets
            # weight settings
            self.train: bool = config.get('training', False)
            self.save: bool = False if not self.train else config.get('save_prog', True)
        else:
            self.train: bool = False
        
        self.read_out: GameDisplay = GameDisplay(self)
        self.ball: Ball = Ball(b_conf)
        
        self.paddles: list[Paddle] = []
        p_conf['x'] = PADDLE_RADIUS * 2
        p_conf['color'] = BLUE
        self.paddles.append(Paddle(p_conf))

        p_conf['x'] = SCREEN_WIDTH - PADDLE_RADIUS * 2
        p_conf['color'] = GREEN
        self.paddles.append(Paddle(p_conf))
        
        self.obstacles: list[Obstacle] = []
        for _ in range(self.num_obs):
            self.add_obstacle()

        self.cur_state = self.get_whole_state()
        
        if self.nn:
            self.nn_model = PongAgent(self.train, self.total_games)
        else:
            self.nn_model = None

        self.running = True

        self.win_loss = {
            'p1_wins': 0,
            'p2_wins': 0,
            'w_l': []
        }
        if not config.get('jupyter', False):
            if self.train:
                self.train_model()
            else:
                self.run()

    def reset(self):
        self.ball.reset()
        for p in self.paddles:
            p.reset()
        for o in self.obstacles:
            o.reset()
    
    def new_game(self):
        self.running = True
        self.reset()

    def end(self):
        self.nn_model.save(CHECKPOINT)
        pg.quit()

    def add_obstacle(self):
        new_o = Obstacle(o_conf)
        self.obstacles.append(new_o)

    def apply_action(self, rev, action):
        if action[0]:  # Up
            self.paddles[rev].vy -= PADDLE_VEL * (-1 if rev else 1)
        elif action[1]:  # Down
            self.paddles[rev].vy += PADDLE_VEL * (-1 if rev else 1)
        elif action[2]:  # Left
            self.paddles[rev].vx -= PADDLE_VEL * (-1 if rev else 1)
        elif action[3]:  # Right
            self.paddles[rev].vx += PADDLE_VEL * (-1 if rev else 1)

    def get_player_moves(self, keys):
        # Paddle movement
        actions = []
        
        if self.players > 0:
            act = [0, 0, 0, 0, 0]
            up_down = (-keys[pg.K_w] + keys[pg.K_s])
            left_right = (-keys[pg.K_a] + keys[pg.K_d])
            
            if up_down or left_right:
                act[1] = up_down > 0  # Move up
                act[0] = up_down < 0  # Move dow
                act[3] = left_right > 0  # Move left
                act[2] = left_right < 0  # Move right
            if not any(act[:4]):
                act[4] = 1  # None
            actions.append(act)

        if self.players > 1:
            act = [0, 0, 0, 0, 0]
            up_down = (-keys[pg.K_UP] + keys[pg.K_DOWN])
            left_right = (-keys[pg.K_LEFT] + keys[pg.K_RIGHT])
            
            if up_down or left_right:
                act[1] = up_down > 0  # Move up
                act[0] = up_down < 0  # Move dow
                act[3] = left_right > 0  # Move left
                act[2] = left_right < 0  # Move right
            if not any(act[:4]):
                act[4] = 1  # None
            actions.append(act)

        return actions

    def get_state(self, rev=False):

        def add_box(img, locate, rad, val, delim=STATE_SPLIT):
            offset = int(rad // delim)
            off_2 = offset // 2
            # Calculate the square boundaries, ensuring they stay within array limits
            x_start = locate[1] - offset
            x_end = locate[1] + offset
            y_start = locate[2] - offset
            y_end = locate[2] + offset
            # left and right split for OBOB
            img[:, x_start: locate[1], y_start: y_end, :] = val
            img[:, locate[1]: x_end, y_start: y_end, :] = val

            # right box
            img[:, x_end: x_end + off_2, y_start + off_2: y_end - off_2, :] = val
            # left box
            img[:, x_start - off_2: max(0,x_start), y_start + off_2: y_end - off_2, :] = val
            # left top/bot box
            img[:, x_start + off_2: locate[1], y_start - off_2: y_end + off_2, :] = val
            # right top/bot box
            img[:, locate[1]:x_end - off_2 , y_start - off_2: y_end + off_2, :] = val
                        
            return img

        state_space = np.zeros((1, SCREEN_WIDTH//STATE_SPLIT, SCREEN_HEIGHT//STATE_SPLIT, 3))
        normalizer = 1.4
        loc, val = get_obj_state_repr(self.ball, 4, BALL_MAX_SPEED, rev)
        state_space = add_box(state_space, loc, self.ball.r/normalizer, val)

        pads = self.paddles[::-1] if rev else self.paddles
        for i, p in enumerate(pads):
            ot = 1 if i else 3
            loc, val = get_obj_state_repr(p, ot, PADDLE_VEL, rev)
            state_space = add_box(state_space, loc, p.r/normalizer, val)
        
        for ob in self.obstacles:
            loc, val = get_obj_state_repr(ob, 2, BALL_MAX_SPEED, rev)
            state_space = add_box(state_space, loc, ob.r/normalizer, val)
        if rev:
            state_space = state_space[:, ::-1, ::-1, :]
        return state_space

    def get_whole_state(self):
        return [self.get_state(0), self.get_state(1)]

    def get_ai_moves(self):
        bh = self.ball.y
        vl = self.ball.vy
        fbh = bh + vl*10
        buffer = PADDLE_RADIUS // 4
        follow = lambda x: (-(x - buffer > fbh) + (x + buffer < fbh)) > 0
        center = lambda x: min(1, max(-1, (MID_HEIGHT - x)//20)) > 0

        def update_paddle(rev, direction, x_condition, follow_action=follow, center_action=center):
            act = [0 for _ in range(5)]
            paddle = self.paddles[rev]
            if x_condition:
                if self.train and random.random() < self.nn_model.epsilon:
                    ind = random.randint(0, 4)
                    act[ind] = 1
                    return act
                fdy = follow_action(paddle.y)
                act[1] = fdy
                act[0] = not fdy

                if distance(paddle, self.ball) < paddle.r * 2:
                    adx = direction < 0
                    act[2] = adx
                    act[3] = not adx

            else:
                ddy = center_action(paddle.y)
                ddx = direction < 0
                act[1] = ddy
                act[0] = not ddy
                act[3] = ddx 
                act[2] = not ddx 
            
            if rev:
                tmp = act[0]
                act[0] = act[1]
                act[1] = tmp
                tmp = act[2]
                act[2] = act[3]
                act[3] = tmp

            act[-1] = sum(act) == 0
            return act

        def nn_play(rev=False):
            return self.nn_model(self.cur_state[rev])     

        bot_actions = []
        if self.players == 0:
            if self.nn == 0: 
                bot_actions.append(update_paddle(0, 1, self.ball.vx <= 0))
                bot_actions.append(update_paddle(1, -1, self.ball.vx >= 0))
            elif self.nn == 1:
                bot_actions.append(update_paddle(0, 1, self.ball.vx <= 0))
                bot_actions.append(nn_play(1))
            else:
                bot_actions.append(nn_play(0))
                bot_actions.append(nn_play(1))

        elif self.players == 1:
            if self.nn > 0:
                bot_actions.append(nn_play(1))
            else:
                bot_actions.append(update_paddle(1, -1, self.ball.vx >= 0))

        return bot_actions
    
    def resistance(self):
        for p in self.paddles:
            p.vy *= PADDLE_MOVE_DECAY 
            p.vx *= PADDLE_MOVE_DECAY 

    def toggle_keys(self, keys):
        if keys[pg.K_COMMA]: self.read_out.inc_fps(1)
        if keys[pg.K_PERIOD]: self.read_out.inc_fps()

    def move_objects(self):
        keys = pg.key.get_pressed()
        self.resistance()
        self.toggle_keys(keys)

        player_moves = []
        if self.players > 0:
            player_moves = self.get_player_moves(keys)
    
        bot_moves = []
        if self.players < 2: 
            bot_moves = self.get_ai_moves()

        all_moves = player_moves + bot_moves
        for i, mv in enumerate(all_moves):
            self.apply_action(i, mv)

        for p in self.paddles:
            p.move()
        
        self.ball.move()

        for obs in self.obstacles:
            obs.move()
        
        return all_moves
    
    def reward_func(self, p1s, p1w, p2s, p2w):
        def calculate_paddle_reward(paddle, ball, ticks):
            reward = 0
            delta = ball.vy * ticks
            future = ball.y + delta
            top_lim = SCREEN_HEIGHT - ball.r
            while not 0 < future < SCREEN_HEIGHT:
                if future < ball.r:
                    future = abs(future) + ball.r
                elif future > top_lim:
                    future = top_lim - (future - top_lim)
            
            #pg.draw.circle(self.read_out.screen, RED, (self.ball.x + self.ball.vx*ticks, future), 5)
            norm_dist = (future - paddle.y) / SCREEN_HEIGHT
            reward += -abs(norm_dist)
            #reward += sigmoid(paddle.vy * norm_dist) - 0.5
            reward += MISS_PENALTY * ((ball.x - paddle.x) if paddle.x < MID_WIDTH else (paddle.x - ball.x)) / SCREEN_WIDTH
            return reward

        def calculate_hit_reward(paddle, ball):
            if not paddle.hit:
                return 0
            hit_direction = 2 * (not ((paddle.x < MID_WIDTH) ^ (ball.vx > 0))) - 1
            return HIT_REWARD * hit_direction #* (abs(ball.velocity())/BALL_MAX_SPEED)

        score_factor = SCORE_REWARD_MULT * (p1s - p2s)
        win_factor = WIN_REWARD * (p1w - p2w)
        base_reward = score_factor + win_factor

        rewards = []
        for i, paddle in enumerate(self.paddles):
            reward = -base_reward if i else base_reward
            # Calculate distance-based reward
            dist = distance(paddle, self.ball) 
            norm_dist = dist / SCREEN_WIDTH
            
            if i ^ (self.ball.vx < 0):      # ball is coming toward this paddle
                reward += 0.5 - min(1, norm_dist)
                time_delta = abs(self.ball.x - (paddle.x))/abs(self.ball.vx)
                reward += calculate_paddle_reward(paddle, self.ball, time_delta-(60/abs(self.ball.vx)))
                reward += calculate_hit_reward(paddle, self.ball)
                # make the reward proportional to the distance from the ball
                # reward = (1/((distance(paddle, self.ball, axis=1) / SCREEN_HEIGHT)+0.5))-1.4
                # reward *= (5 - log10(distance(paddle, self.ball, axis=0)))**2
            else:       # ball is going away from paddle
               reward += abs(paddle.x - self.ball.x) / SCREEN_WIDTH
               reward -= abs(paddle.y - MID_HEIGHT) / MID_HEIGHT
                # reward = log10(distance(paddle, self.ball, axis=0)) * (abs(self.ball.vx) + abs(self.ball.vx)) * 0.01
            rewards.append(reward)

        return rewards

    # def reward_func(self, p1s, p1w, p2s, p2w):
    #     """
    #     Improved reward function focusing on:
    #     1. Intercepting the ball when it's coming toward paddle
    #     2. Punishing being out of position
    #     3. Rewarding successful hits with good angles
    #     4. Encouraging center positioning when ball is away
    #     """
        
    #     rewards = []
    #     ball_speed = self.ball.velocity()
        
    #     for i, paddle in enumerate(self.paddles):
    #         is_left_paddle = paddle.x < MID_WIDTH
    #         ball_moving_toward = (is_left_paddle and self.ball.vx < 0) or (not is_left_paddle and self.ball.vx > 0)
            
    #         if ball_moving_toward:
    #             # === BALL APPROACHING: Focus on interception ===
                
    #             # Predict where ball will be at paddle's x-position
    #             if abs(self.ball.vx) > 0.1:
    #                 dx = abs(paddle.x - self.ball.x)
    #                 time_to_intercept = dx / abs(self.ball.vx)
    #                 predicted_y = self.ball.y + self.ball.vy * time_to_intercept
                    
    #                 # Handle wall bounces in prediction
    #                 bounces = 0
    #                 while predicted_y < BALL_RADIUS or predicted_y > SCREEN_HEIGHT - BALL_RADIUS:
    #                     bounces += 1
    #                     if bounces > 5:  # Safety limit
    #                         predicted_y = self.ball.y
    #                         break
    #                     if predicted_y < BALL_RADIUS:
    #                         predicted_y = BALL_RADIUS + (BALL_RADIUS - predicted_y)
    #                     else:
    #                         predicted_y = (SCREEN_HEIGHT - BALL_RADIUS) - (predicted_y - (SCREEN_HEIGHT - BALL_RADIUS))
                    
    #                 # Distance from predicted interception point
    #                 y_error = abs(paddle.y - predicted_y) / SCREEN_HEIGHT
                    
    #                 # Strong reward for being close to interception point
    #                 interception_reward = 1.0 / (1.0 + 5.0 * y_error)
                    
    #                 # Urgency factor: reward increases as ball gets closer
    #                 urgency = 1.0 - (dx / SCREEN_WIDTH)
    #                 urgency_bonus = urgency * 0.5
                    
    #                 reward = interception_reward + urgency_bonus
                    
    #                 # Bonus for moving toward the correct position
    #                 if abs(self.ball.vy) > 0.1:
    #                     correct_direction = (paddle.vy > 0 and predicted_y > paddle.y) or \
    #                                     (paddle.vy < 0 and predicted_y < paddle.y)
    #                     if correct_direction:
    #                         reward += 0.2
                    
    #             else:
    #                 # Ball moving mostly horizontal
    #                 y_error = abs(paddle.y - self.ball.y) / SCREEN_HEIGHT
    #                 reward = 1.0 / (1.0 + 3.0 * y_error)
                
    #             # Hit reward: big bonus for successful interception
    #             if paddle.hit:
    #                 # Reward hitting ball back toward opponent
    #                 good_hit = (is_left_paddle and self.ball.vx > 0) or (not is_left_paddle and self.ball.vx < 0)
    #                 reward += 3.0 if good_hit else -1.0
                    
    #                 # Bonus for hitting with high speed
    #                 reward += ball_speed / (BALL_MAX_SPEED * 2)
            
    #         else:
    #             # === BALL MOVING AWAY: Prepare for return ===
                
    #             # Encourage returning to center position
    #             center_error = abs(paddle.y - MID_HEIGHT) / MID_HEIGHT
    #             center_reward = 0.3 * (1.0 - center_error)
                
    #             # Small reward for tracking ball vertically (stay aware)
    #             y_tracking = 1.0 - (abs(paddle.y - self.ball.y) / SCREEN_HEIGHT)
    #             tracking_reward = 0.1 * y_tracking
                
    #             # Penalty for being too far from base x position
    #             x_error = abs(paddle.x - paddle.base_x) / (paddle.r * 2)
    #             x_penalty = -0.2 * min(x_error, 1.0)
                
    #             reward = center_reward + tracking_reward + x_penalty
            
    #         rewards.append(reward)
        
    #     return rewards

    def check_for_score(self, score):
        p1_score = self.ball.x == SCREEN_WIDTH
        p2_score = self.ball.x == 0
        p1_win = False
        p2_win = False
        ns = self.get_whole_state()
        if p1_score or p2_score:
            if p1_score:
                score['p1'] += 1
                self.running = verify(score)
                p1_win = not self.running
            else:
                score['p2'] += 1
                self.running = verify(score)
                p2_win = not self.running

            self.reset()

        rewards = self.reward_func(p1_score, p1_win, p2_score, p2_win)
        return score, rewards, ns
    
    def obstacle_collision(self):
        for i, obstacle in enumerate(self.obstacles):
            if obstacle.collides(self.ball):
                collide(obstacle, self.ball)
            if i < len(self.obstacles):
                for other in self.obstacles[i+1:]:
                    if obstacle.collides(other):
                        collide(obstacle, other)

    def paddle_collision(self):
        for i, paddle in enumerate(self.paddles):
            if paddle.collides(self.ball):
                paddle.hit = True
                collide(paddle, self.ball)
                if i:
                    self.ball.vx *= -1 if self.ball.vx > 0 and self.ball.x < paddle.x else 1
                else:
                    self.ball.vx *= -1 if self.ball.vx < 0 and self.ball.x > paddle.x else 1

    def detect_collision(self, score):
        self.obstacle_collision()
        self.paddle_collision()
        return self.check_for_score(score)

    #@timer
    def step(self, score):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.running = False
        moves = self.move_objects()
        score, rewards, next_state = self.detect_collision(score)
        self.read_out.update_display(score, rewards)
        return score, rewards, moves, next_state

    def run(self):
        for game_num in range(1, self.games + 1):
            self.cur_game = game_num
            score = {'p1': 0, 'p2': 0}

            while self.running:
                score, rewards, actions, next_state = self.step(score)
                # self.steps_taken += 1
                # if self.steps_taken > 20 and self.start_check:
                #     from matplotlib import pyplot as plt
                #     plt.figure(figsize=(20, 14))
                #     plt.imshow(self.cur_state[0][0])
                #     plt.axis('off')          # hide axis ticks
                #     plt.show()
                for x in list(range(2)):
                    if self.train:
                        self.nn_model.remember(self.cur_state[x], actions[x], rewards[x], next_state[x], self.running)
                    self.cur_state[x] = (next_state[x] + self.cur_state[x] * 0.95).clip(0,1)

            if score['p1'] == MAX_SCORE:
                self.win_loss['p1_wins'] += 1
            else:
                self.win_loss['p2_wins'] += 1
            self.win_loss['w_l'].append(self.win_loss['p1_wins']/(self.win_loss['p2_wins']+self.win_loss['p1_wins']))

            self.new_game()
            if self.train:
                self.nn_model.replay()
                self.nn_model.apply_decay()

        if self.train:
            self.nn_model.reset()
            if self.save:
                self.nn_model.save(SAVE_PATH)
        else:
            self.end()
    
    def train_model(self):
        for match in range(1, self.num_matches+1):
            self.cur_match = match
            for match_set in range(1, self.num_sets+1):
                self.cur_set = match_set
                self.run()
                #self.nn = match_set % 2 + 1
                self.nn_model.update_target()
            #if match == self.num_matches:
            #    self.add_obstacle()
        self.end()

# MODE = -1
# if __name__ == "__main__":
#     # train
#     if MODE == 0: 
#         conf = {'nn': 2, 'training': True}

#     # test
#     elif MODE == 1: 
#         conf = {'nn': 2,'num_games': 5, 'slow':True}

#     # watch
#     elif MODE == 2: 
#         conf = {'nn': 2,'num_games': 5}

#     # play
#     else: 
#         conf = {'players': 1, 'nn': 1}

#     print(conf)
#     g = None
#     try:
#         g = Game(**conf)
#     except Exception as e:
#         if g: g.end()
