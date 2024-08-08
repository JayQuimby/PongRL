import cProfile
import pstats
import io
from game import Game

conf = {
        'players': 0,
        'nn': 1,
        'training': False,
        'num_games': 1
    }

# Profile the function
pr = cProfile.Profile()
pr.enable()
Game(**conf)
pr.disable()

# Create a stream to hold profiling results
s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.TIME, pstats.SortKey.CUMULATIVE).print_stats(50)

# Print the profiling results
print(s.getvalue())