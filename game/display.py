import numpy as np
import pygame as pg
from game.static import *
from game.utils import (
    bresenham_line, interpolate_color, rr
)

class GameDisplay:
    def __init__(self, game):
        self.game = game
        render_skipping = game.train
        self.game_rate = SCREEN_SAMPLE_RATE * (5 if render_skipping else 1)
        self.ui_rate = UI_SAMPLE_RATE * (3 if render_skipping else 1)
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
            normalized_values = [int((v - min_val) / max(0.01,max_val - min_val) * max(0.01,height - point_size)) for v in values]

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
                    # Clip px and py to valid array bounds
                    px = max(0, min(width - 1, px))
                    py = max(0, min(height - 1, py))
                    pixel_array[px, py] = interpolate_color(py, min_norm, max_norm, colors[1], colors[0])
            
            previous_point = (x, y)

        return pg.surfarray.make_surface(pixel_array)

    def inc_fps(self, neg=False):
        self.fps -= (1 if neg else -1)
        self.fps = min(max(self.fps, 0), 1000)