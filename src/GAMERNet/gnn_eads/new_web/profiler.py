import pstats

stats = pstats.Stats('output.prof')
stats.sort_stats('time').print_stats(20)  # Print stats of the top 10 functions
