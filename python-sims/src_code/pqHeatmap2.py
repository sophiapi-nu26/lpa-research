from lpafunctions import *

N_values = [3200, 6400, 12800]
community_values = [(i+1) for i in range(4)]
size_values = [33]
trial_values = [4]
type_values = ['minRand', 'minMin']

N_values = [1000, 2500, 5000]
community_values = [2]

for N in N_values:
    for comm in community_values:
        for size in size_values:
            for trials in trial_values:
                for type in type_values:
                    print("starting...")
                    generatePQHeatmap(N, comm, size, trials, type=type)