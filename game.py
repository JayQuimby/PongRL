import pygame as pg
import numpy as np
import random
from global_vars import *
from utils import (
    bresenham_line, 
    interpolate_color, 
    sigmoid, 
    get_obj_state_repr
)
from objects import Ball, Paddle, Obstacle
from agent import PongAgent
from utils import load_conf, distance, collide
p_conf = load_conf('paddle')
b_conf = load_conf('ball')
o_conf = load_conf('obstacle')
rr = lambda x: round(x, 1)
verify = lambda x: x['p1'] < MAX_SCORE and x['p2'] < MAX_SCORE

def timer(func):
    def wrapper(*args, **kwargs):
        # Start the timer
        start_time = pg.time.get_ticks()

        # Execute the wrapped function
        result = func(*args, **kwargs)

        # Stop the timer
        end_time = pg.time.get_ticks()
        runtime = (end_time - start_time) / 1000.0 + 0.0001
        
        # Use the function's name as the message
        message = f"{func.__name__}"
        print(f"{message} took \033[0;32m{runtime:.4f}\033[0m seconds\t\033[0;35m{(1.0 / runtime):.2f}\033[0mfps")

        return result
    return wrapper

class GameDisplay:
    def __init__(self, game):
        self.game = game
        self.game_rate = SCREEN_SAMPLE_RATE * (5 if game.train else 1)
        self.ui_rate = UI_SAMPLE_RATE * (3 if game.train else 1)
        self.fps = 100 if not game.train else 1000

        if game.slow_mo:
            self.game_rate = 1
            self.ui_rate = 1
            self.fps = 10
        
        self.screen = pg.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT + UI_SIZE))
        pg.display.set_caption('Self play simulation')
        self.clock = pg.time.Clock()

        pg.font.init()
        self.font = pg.font.Font(None, 28)
        self.big_font = pg.font.Font(None, 48)
        self.small_font = pg.font.Font(None, 18)
        self.start = pg.time.get_ticks()
        self.step = 0

    def step_itter(self):
        self.step = (self.step + 1) % self.ui_rate

    def render_text(self, text, font, color=WHITE):
        return font.render(text, True, color)

    def blit_text(self, text, position):
        self.screen.blit(text, position)

    def display_variables(self, score, rewards):
        end = pg.time.get_ticks()
        self.screen.fill(BLACK, pg.Rect(0,SCREEN_HEIGHT+1, SCREEN_WIDTH, UI_SIZE))
        runtime = (end - self.start) / 1000.0 + 0.0001
        variables = [
            (f'FPS | {self.fps} : {int(self.ui_rate/runtime)} |', self.small_font, (5, SCREEN_HEIGHT+40), YELLOW),
            (f'Ball Position: ({self.game.ball.x // 10 * 10}, {self.game.ball.y // 10 * 10})', self.font, (MID_WIDTH + 110, SCREEN_HEIGHT + STAT_BAR_OFFSET)),
            (f'Ball Velocity: ({rr(self.game.ball.vx)}, {rr(self.game.ball.vy)})', self.font, (MID_WIDTH - 310, SCREEN_HEIGHT + STAT_BAR_OFFSET)),
            (f'Left Paddle Y: {rr(self.game.paddles[0].y)}', self.font, (10, SCREEN_HEIGHT + STAT_BAR_OFFSET)),
            (f'Right Paddle Y: {rr(self.game.paddles[1].y)}', self.font, (SCREEN_WIDTH - 210, SCREEN_HEIGHT + STAT_BAR_OFFSET)),
            (f'P1 Reward: {rr(rewards[0])}', self.font, (20, SCREEN_HEIGHT + REWARD_OFFSET), BLUE),
            (f'P2 Reward: {rr(rewards[1])}', self.font, (SCREEN_WIDTH - 160, SCREEN_HEIGHT + REWARD_OFFSET), GREEN),
            (f'Game: {self.game.cur_game}', self.font, (MID_WIDTH-25, SCREEN_HEIGHT + 40), RED),
            (f'{score["p1"]} - {score["p2"]}', self.big_font, (MID_WIDTH-20, SCREEN_HEIGHT + 70), RED),
        ]
        if self.game.nn_model:
            variables.extend([
                    (f'NN Control: {round((1-self.game.nn_model.epsilon)*100, 2)}%', self.small_font, (SCREEN_WIDTH-120, SCREEN_HEIGHT + 40), YELLOW),
                    (f'Match: {self.game.cur_match}', self.font, (MID_WIDTH-130, SCREEN_HEIGHT+40), RED),
                    (f'Set: {self.game.cur_set}', self.font, (MID_WIDTH-115, SCREEN_HEIGHT+80), RED),
                ])
            
            loss_vals = self.game.nn_model.stats['train_loss']
            self.blit_text(self.create_dense_plot(loss_vals, colors=(RED, GREEN)), (250, SCREEN_HEIGHT + 40))

            if len(loss_vals) >  0:
                variables.extend([
                    (f'{round(max(loss_vals), 4)} -', self.small_font, (190, SCREEN_HEIGHT + 40), WHITE),
                    (f'{METRIC[:min(len(METRIC), 15)]}', self.small_font, (MID_WIDTH - OBSTACLE_SPREAD, SCREEN_HEIGHT + 109), YELLOW),
                    (f'{round(min(loss_vals),4)} -', self.small_font, (190, SCREEN_HEIGHT + 30 + PLOT_HEIGHT), WHITE),
                ])
            
            plt_offset = SCREEN_WIDTH - PLOT_WIDTH - 210
            w_l = self.game.win_loss['w_l']
            self.blit_text(self.create_dense_plot(w_l, colors=(BLUE, GREEN)), (plt_offset, SCREEN_HEIGHT + 40))

            if len(w_l) >  0:
                variables.extend([
                    (f'{round(max(w_l)*100,2)}% -', self.small_font, (plt_offset-50, SCREEN_HEIGHT + 40), WHITE),
                    (f'Win/Loss', self.small_font, (MID_WIDTH + OBSTACLE_SPREAD - 70, SCREEN_HEIGHT+109), YELLOW),
                    (f'{round(min(w_l)*100,2)}% -', self.small_font, (plt_offset-50, SCREEN_HEIGHT + 30 + PLOT_HEIGHT), WHITE),
                ])

        for text, font, position, *color in variables:
            color = color[0] if color else WHITE
            rendered_text = self.render_text(text, font, color)
            self.blit_text(rendered_text, position)
        
        pg.draw.rect(self.screen, WHITE, (0, SCREEN_HEIGHT + 35, SCREEN_WIDTH, 1))
        self.start = pg.time.get_ticks()

    def redraw_screen(self):
        self.game.ball.draw(self.screen)
        self.game.paddles[0].draw(self.screen)
        self.game.paddles[1].draw(self.screen)
        for obstacle in self.game.obstacles:
            obstacle.draw(self.screen)
        pg.draw.rect(self.screen, WHITE, (MID_WIDTH - OBSTACLE_SPREAD, 0, 1, SCREEN_HEIGHT))
        pg.draw.rect(self.screen, WHITE, (MID_WIDTH + OBSTACLE_SPREAD, 0, 1, SCREEN_HEIGHT))
        pg.draw.rect(self.screen, WHITE, (0, SCREEN_HEIGHT, SCREEN_WIDTH, 1))

    def update_display(self, s, r):
        if self.step % self.game_rate == 0:
            self.redraw_screen()
        if self.step == 0:
            self.display_variables(s, r)
        pg.display.flip()
        self.clock.tick(self.fps)
        self.step_itter()

    def create_dense_plot(self, values, width=PLOT_WIDTH, height=PLOT_HEIGHT, point_size=1, colors=(GREEN, GRAY), bg_color=BLACK):
        # Create a 3D numpy array for the pixel data (height, width, RGB)
        pixel_array = np.full((width, height, 3), bg_color, dtype=np.uint8)
        
        if len(values) == 0:
            return pg.surfarray.make_surface(pixel_array)

        min_val, max_val = min(values), max(values)
        if min_val == max_val:
            normalized_values = [height // 2] * len(values)
        else:
            normalized_values = [int((v - min_val) / (max_val - min_val) * (height - point_size)) for v in values]

        min_norm = min(normalized_values)
        max_norm = max(normalized_values)
        # Determine the step size for x-axis
        step = max(1, len(values) // width)

        # Plot the points
        previous_point = None
        for i in range(0, len(values), step):
            x = int(i / len(values) * (width - point_size))
            y = max(0, min(height - point_size, height - 1 - normalized_values[i]))  

            if previous_point is not None:
                for px, py in bresenham_line(previous_point[0], previous_point[1], x, y):
                    pixel_array[px, py] = interpolate_color(py, min_norm, max_norm, colors[1], colors[0])
            
            previous_point = (x, y)

        return pg.surfarray.make_surface(pixel_array)

    def inc_fps(self, neg=False):
        self.fps -= (1 if neg else -1)
        self.fps = min(max(self.fps, 0), 1000)

class Game:
    def __init__(self, **config):
        pg.init()
        self.players = config.get('players', 0)
        self.nn = config.get('nn', 0)
        self.num_obs = config.get('obstacles', 0)
        self.slow_mo = config.get('slow', False)
        self.cur_game = 1
        self.games = config.get('num_games', 1)
        if self.nn:
            self.train = config.get('training', False)
            self.save = False if not self.nn else config.get('save_prog', False)
            self.cur_match = 0
            self.cur_set = 0
            self.num_matches = config.get('matches', 5)
            self.num_sets = config.get('sets', 2)
            self.total_games = self.games * self.num_matches * self.num_sets
        else:
            self.train = False
        
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
        
    def add_obstacle(self):
        new_o = Obstacle(o_conf)
        self.obstacles.append(new_o)

    def apply_action(self, rev, action):
        if action[0]:  # Up
            self.paddles[rev].vy -= PADDLE_VEL
        if action[1]:  # Down
            self.paddles[rev].vy += PADDLE_VEL
        if action[2]:  # Left
            self.paddles[rev].vx -= PADDLE_VEL #* -1 if rev else 1
        if action[3]:  # Right
            self.paddles[rev].vx += PADDLE_VEL #* -1 if rev else 1

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
        loc, val = get_obj_state_repr(self.ball, 4, BALL_MAX_SPEED)
        state_space = add_box(state_space, loc, self.ball.r/normalizer, val)

        pads = self.paddles[::-1] if rev else self.paddles
        for i, p in enumerate(pads):
            ot = 3 if i else 1
            loc, val = get_obj_state_repr(p, ot, PADDLE_VEL)
            state_space = add_box(state_space, loc, p.r/normalizer, val)
        
        for ob in self.obstacles:
            loc, val = get_obj_state_repr(ob, 2, BALL_MAX_SPEED)
            state_space = add_box(state_space, loc, ob.r/normalizer, val)
        
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

            act[-1] = sum(act) == 0
            return act

        def nn_play(rev=False):
            return self.nn_model(self.get_state(rev))          

        bot_actions = []
        if self.players == 0:
            if self.nn == 0: 
                bot_actions.append(update_paddle(0, 1, self.ball.vx <= 0))
                bot_actions.append(update_paddle(1, -1, self.ball.vx >= 0))
            elif self.nn == 1:
                bot_actions.append(update_paddle(0, 1, self.ball.vx <= 0))
                bot_actions.append(nn_play(1))
            else:
                bot_actions.extend([nn_play(0), nn_play(1)])

        elif self.players == 1:
            if self.nn > 0:
                bot_actions.append(nn_play(1))
            else:
                bot_actions.extend(update_paddle(1, -1, self.ball.vx >= 0))

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
            r = 0
            if ticks < 200:
                delta = ball.vy * ticks
                future = ball.y + delta
                while not 0 < future < SCREEN_HEIGHT:
                    if future < ball.r:
                        future = abs(future)
                    elif future > SCREEN_HEIGHT - ball.r:
                        future = SCREEN_HEIGHT - (future - SCREEN_HEIGHT)
            else:
                future = ball.y

            norm_dist = (future - paddle.y) / SCREEN_HEIGHT
            if norm_dist < paddle.r * 2:
                r += sigmoid(MAX_ANTICIPATION_TIME - ticks)
            r += -abs(norm_dist)
            r += sigmoid(paddle.vy * norm_dist) - 0.5
            r += MISS_PENALTY * ((ball.x - paddle.x) if paddle.x < MID_WIDTH else (paddle.x - ball.x)) / SCREEN_WIDTH
            return r

        def calculate_hit_reward(paddle, ball):
            if not paddle.hit:
                return 0
            hit_direction = 2 * (not ((paddle.x < MID_WIDTH) ^ (ball.vx > 0))) - 1
            hit_position = (abs(paddle.y - ball.y) / paddle.r) + 0.5
            return HIT_REWARD * hit_direction * hit_position

        score_factor = SCORE_REWARD_MULT * (p1s - p2s)
        win_factor = WIN_REWARD * (p1w - p2w)
        base_reward = score_factor + win_factor

        rewards = []
        for i, paddle in enumerate(self.paddles):
            reward = -base_reward if i else base_reward
            # Calculate distance-based reward
            dist = distance(paddle, self.ball)
            reward += 0.5 - min(1, dist / SCREEN_WIDTH)
            time_to_intercept = min(200, abs((dist-paddle.r) / (abs(self.ball.vx) + 1e-6)))
            reward += calculate_paddle_reward(paddle, self.ball, time_to_intercept)
            reward += calculate_hit_reward(paddle, self.ball)
            
            rewards.append(reward)

        return rewards

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

    def reset(self):
        self.ball.reset()
        for p in self.paddles:
            p.reset()
        for o in self.obstacles:
            o.reset()
    
    def new_game(self):
        self.running = True
        self.reset()

    #@timer
    def step(self, score):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.running = False
        moves = self.move_objects()
        score, rewards, next_state = self.detect_collision(score)
        self.read_out.update_display(score, rewards)
        return score, rewards, moves, next_state

    def end(self):
        pg.quit()

    def run(self):
        for game_num in range(1, self.games + 1):
            self.cur_game = game_num
            score = {'p1': 0, 'p2': 0}

            while self.running:
                score, rewards, actions, next_state = self.step(score)
                
                if self.train:
                    for x in list(range(2)):
                        self.nn_model.remember(self.cur_state[x], actions[x], rewards[x], next_state[x], self.running)
                    self.cur_state = next_state
            
            if score['p1'] == MAX_SCORE:
                self.win_loss['p1_wins'] += 1
            else:
                self.win_loss['p2_wins'] += 1
            self.win_loss['w_l'].append(self.win_loss['p1_wins']/(self.win_loss['p2_wins']+self.win_loss['p1_wins']))

            self.new_game()
            if self.train:
                self.nn_model.replay(0.2)
                self.nn_model.apply_decay()

        if self.train:
            self.nn_model.reset()
            if self.save:
                self.nn_model.save(MODEL_PATH)
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
            #if self.num_obs > 0:
            #self.add_obstacle()
        self.end()


if __name__ == "__main__":
    conf = {
        'nn': 2,
        'training': True,
        'save_prog': True,
        'num_games': 5,
        'matches': 10,
        'sets': 10,
        'obstacles': 2
    }
    #conf = {'players':  2, 'nn':0, 'num_games': 5, 'obstacles':0, 'slow':True}
    Game(**conf)


