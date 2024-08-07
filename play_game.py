from game import Game
import pygame

conf = {
    'players': 1,
}

def run_game():
    score = {
        'p1': 0,  
        'p2': 0
    }
    g = Game(**conf)
    while g.running:
        score, rewards = g.step(score)
    pygame.quit()

run_game()