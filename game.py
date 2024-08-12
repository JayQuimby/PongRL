import pygame as pg
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
            self.ui_rate = 5
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
            (f'FPS | {self.fps} : {int(self.ui_rate/runtime)} |', self.small_font, (0, SCREEN_HEIGHT+40), YELLOW),
            (f'Ball Position: ({self.game.ball.x // 10 * 10}, {self.game.ball.y // 10 * 10})', self.font, (MID_WIDTH + 110, SCREEN_HEIGHT + STAT_BAR_OFFSET)),
            (f'Ball Velocity: ({rr(self.game.ball.vx)}, {rr(self.game.ball.vy)})', self.font, (MID_WIDTH - 310, SCREEN_HEIGHT + STAT_BAR_OFFSET)),
            (f'Left Paddle Y: {rr(self.game.paddles[0].y)}', self.font, (10, SCREEN_HEIGHT + STAT_BAR_OFFSET)),
            (f'Right Paddle Y: {rr(self.game.paddles[1].y)}', self.font, (SCREEN_WIDTH - 210, SCREEN_HEIGHT + STAT_BAR_OFFSET)),
            (f'P1 Reward: {rr(rewards[0])}', self.font, (20, SCREEN_HEIGHT + REWARD_OFFSET), BLUE),
            (f'P2 Reward: {rr(rewards[1])}', self.font, (SCREEN_WIDTH - 160, SCREEN_HEIGHT + REWARD_OFFSET), GREEN),
            (f'Game: {self.game.cur_game}', self.font, (MID_WIDTH - 39, SCREEN_HEIGHT + 40), RED),
            (f'{score["p1"]} - {score["p2"]}', self.big_font, (MID_WIDTH - 32, SCREEN_HEIGHT + 65), RED),
        ]
        if self.game.nn_model:
            variables.extend([
                    (f'NN Control: {round((1-self.game.nn_model.epsilon)*100, 2)}%', self.small_font, (110, SCREEN_HEIGHT + 40), YELLOW),
                    (f'Match: {self.game.cur_match}', self.font, (MID_WIDTH-150, SCREEN_HEIGHT+40), RED),
                    (f'Set: {self.game.cur_set}', self.font, (MID_WIDTH-150, SCREEN_HEIGHT+70), RED),
                ])
            
            loss_vals = self.game.nn_model.stats['train_loss']
            loss_plot = self.create_dense_plot(loss_vals, PLOT_WIDTH, PLOT_HEIGHT)
            self.blit_text(loss_plot, (300, SCREEN_HEIGHT + 35))

            if len(loss_vals) >  0:
                variables.extend([
                    (f'{round(max(loss_vals), 2)} -', self.small_font, (260, SCREEN_HEIGHT + 37), GREEN),
                    (f'{round(min(loss_vals),2)} -', self.small_font, (260, SCREEN_HEIGHT + 27 + PLOT_HEIGHT), GREEN),
                ])
            plt_offset = SCREEN_WIDTH - PLOT_WIDTH - 300
            win_loss = self.game.win_loss['w_l']
            wl_plot = self.create_dense_plot(win_loss, PLOT_WIDTH, PLOT_HEIGHT)
            self.blit_text(wl_plot, (plt_offset, SCREEN_HEIGHT + 35))

            if len(win_loss) >  0:
                variables.extend([
                    (f'{round(max(win_loss),2)} -', self.small_font, (plt_offset-40, SCREEN_HEIGHT + 37), GREEN),
                    (f'{round(min(win_loss),2)} -', self.small_font, (plt_offset-40, SCREEN_HEIGHT + 27 + PLOT_HEIGHT), GREEN),
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

    def create_dense_plot(self, values, width, height, point_size=10, color=(0, 255, 0), bg_color=(0, 0, 0)):
        # Create a 3D numpy array for the pixel data (height, width, RGB)
        pixel_array = np.full((height, width, 3), bg_color, dtype=np.uint8)
        
        if len(values) == 0:
            return pg.surfarray.make_surface(pixel_array)

        # Normalize the values to fit within the height
        min_val, max_val = min(values), max(values)
        if min_val == max_val:
            normalized_values = [height // 2] * len(values)
        else:
            normalized_values = [int((v - min_val) / (max_val - min_val) * (height - point_size)) for v in values]

        # Determine the step size for x-axis
        step = max(1, len(values) // width)

        # Plot the points
        for i in range(0, len(values), step):
            x = int(i / len(values) * (width - point_size))
            y = max(0, min(height - point_size, height - 1 - normalized_values[i]))  # Ensure y is within bounds
            pixel_array[y, x] = color

        return pg.transform.rotate(pg.transform.flip(pg.surfarray.make_surface(pixel_array),1,0),90)

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
        if self.nn:
            self.train = config.get('training', False)
            self.save = False if not self.nn else config.get('save_prog', False)
            self.cur_match = 0
            self.cur_set = 0
        else:
            self.train = False
        self.games = config.get('num_games', 1)
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

        self.cur_state = [self.get_state(0), self.get_state(1)]
        self.nn_model = None
        if self.nn:
            self.nn_model = PongAgent(self.train, self.games)
        self.running = True

        self.win_loss = {
            'p1_wins': 0,
            'p2_wins': 0,
            'w_l': []
        }

        if self.train:
            self.train_model()
        else:
            self.run()
        pg.quit()

    def add_obstacle(self):
        new_o = Obstacle(o_conf)
        self.obstacles.append(new_o)

    def get_player_moves(self, keys):
        # Paddle movement
        actions = []
        
        if self.players > 0:
            act = [0, 0, 0, 0, 0]
            up_down = (-keys[pg.K_w] + keys[pg.K_s])
            left_right = (-keys[pg.K_a] + keys[pg.K_d])
            
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
            up_down = (-keys[pg.K_UP] + keys[pg.K_DOWN])
            left_right = (-keys[pg.K_LEFT] + keys[pg.K_RIGHT])
            
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

    def get_whole_state(self):
        return [self.get_state(0), self.get_state(1)]

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
            action = self.nn_model(model_input)
            paddle = self.paddles[rev]
            if action[0]:  # Up
                paddle.vy -= PADDLE_VEL
            if action[1]:  # Down
                paddle.vy += PADDLE_VEL
            if action[2]:  # Left
                paddle.vx -= PADDLE_VEL * -1 if rev else 1
            if action[3]:  # Right
                paddle.vx += PADDLE_VEL * -1 if rev else 1
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
                bot_actions.extend([nn_play(x) for x in range(2)])
                '''right = MID_WIDTH + OBSTACLE_SPREAD
                left = MID_WIDTH - OBSTACLE_SPREAD
                if self.ball.x < right:
                    bot_actions.append(nn_play(0))
                if self.ball.x > left:
                    bot_actions.append(nn_play(1))
                if len(bot_actions) < 2:
                    if self.ball.x > right:
                        bot_actions.insert(0, NULL_ACT)
                    else:
                        bot_actions.append(NULL_ACT)'''

        elif self.players == 1:
            if self.nn > 0:
                bot_actions.append(nn_play(1))
            else:
                bot_actions.append(update_paddle(1, -1, self.ball.vx > 0 and self.ball.x > MID_WIDTH, follow, center))
        
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

        self.paddles[0].move()
        self.paddles[1].move()
        self.ball.move()

        for obs in self.obstacles:
            obs.move()
        return player_moves + bot_moves
    
    def reward_func(self, p1s, p1w, p2s, p2w):
        score_factor = SCORE_REWARD_MULT * (p1s - p2s)
        win_factor = WIN_REWARD * (p1w - p2w)
        base_reward = score_factor + win_factor
        rewards = [[base_reward], [-base_reward]]
        cur_hit_reward = HIT_REWARD * abs(self.ball.vx)

        for i, pad in enumerate(self.paddles):
            if i == 0: # left paddle
                rewards[i].append(MISS_PENALTY if self.ball.x < pad.x else 0.3)
                rewards[i].append(0 if not pad.hit or not self.ball.vx > 0 else cur_hit_reward)
            else: # right paddle
                rewards[i].append(MISS_PENALTY if self.ball.x > pad.x else 0.3)
                rewards[i].append(0 if not pad.hit or not self.ball.vx < 0 else cur_hit_reward)
                
            rewards[i].append(-abs(pad.y - self.ball.y)/SCREEN_HEIGHT)
        
        return [sum(r) for r in rewards]

    def check_for_score(self, score):
        p1_score = self.ball.x > SCREEN_WIDTH
        p2_score = self.ball.x < 0
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
            else:
                paddle.hit = False

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

    def run(self):
        for game_num in range(self.games):
            self.cur_game = game_num
            score = {'p1': 0, 'p2': 0}

            while self.running:
                score, rewards, actions, next_state = self.step(score)
                
                if self.train:
                    for x in [0, 1]:
                        self.nn_model.remember(self.cur_state[x], actions[x], rewards[x], next_state[x], self.running)
                    self.cur_state = next_state

            if score['p1'] == 11:
                self.win_loss['p1_wins'] += 1
            else:
                self.win_loss['p2_wins'] += 1
            self.win_loss['w_l'].append(self.win_loss['p1_wins']/(self.win_loss['p2_wins']+self.win_loss['p1_wins']))

            self.new_game()
            if self.train:
                self.nn_model.replay(5)
        if self.train and self.save:
            self.nn_model.replay(2)
            self.nn_model.reset_mem()
            self.nn_model.save(MODEL_PATH)
    
    def train_model(self):
        for match in range(10):
            self.cur_match = match
            for match_set in range(2):
                self.cur_set = match_set
                #self.nn = match_set + 1
                self.run()
            if self.num_obs > 0:
                self.add_obstacle()

if __name__ == "__main__":
    conf = {
        'nn': 2,
        'training': True,
        'save_prog': True,
        'num_games': 100
    }
    #conf = {'players': 1,'nn': 1, 'slow':True}
    Game(**conf)


