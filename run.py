import argparse
from game import Game


def main():
    parser = argparse.ArgumentParser(description="Game configuration script")
    parser.add_argument("--mode", type=int, default=-1, choices=[-1, 0, 1, 2],
                        help="Game mode: 0 (train), 1 (test), 2 (watch), -1 (play, default)")
    parser.add_argument("--nn", type=int, default=2, choices=[0, 1, 2], help="Number of Neural network players")
    parser.add_argument("--num_games", type=int, default=5, help="Number of games to play")
    parser.add_argument("--slow", action="store_true", help="Slow mode for testing")
    parser.add_argument("--players", type=int, default=1, help="Number of players for play mode")

    args = parser.parse_args()

    if args.mode == 0:  # train
        conf = {'nn': args.nn, 'training': True}
    elif args.mode == 1:  # test
        conf = {'nn': args.nn, 'num_games': args.num_games, 'slow': args.slow}
    elif args.mode == 2:  # watch
        conf = {'nn': args.nn, 'num_games': args.num_games}
    else:  # play (default)
        conf = {'players': args.players, 'nn': args.nn}

    print(conf)
    Game(**conf)

if __name__ == "__main__":
    main()