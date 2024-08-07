from game import Game
import pygame

if __name__ == "__main__":
    conf = {'players': 0}
    score = {'p1': 0, 'p2': 0}
    g = Game(**conf)
    while g.running:
        score, rewards = g.step(score)
    pygame.quit()
