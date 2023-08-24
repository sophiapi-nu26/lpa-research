from lpafunctions import *

N_values = [3200, 6400, 12800]
community_values = [(i+1) for i in range(4)]
size_values = [33]
trial_values = [4]

N_values = [100]
community_values = [3]

for N in N_values:
    for comm in community_values:
        for size in size_values:
            for trials in trial_values:
                generatePQHeatmap(N, comm, size, trials)