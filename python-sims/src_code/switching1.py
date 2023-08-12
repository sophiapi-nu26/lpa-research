# generate histogram

from lpafunctions import *

N_arr = [100, 1000, 5000]
numCommunities_arr = [1, 2, 3, 4]

a_arr = [0, 0.5, 0.7, 0.75, 0.9, 1]
b_arr = [0, 0.1, 0.25, 0.3, 0.5, 0]

for N in N_arr:
    for numCommunities in numCommunities_arr:
        for a in a_arr:
            for b in b_arr:
                p = N**(-a)
                q = N**(-b)
                graphSwitching(N, numCommunities, p, q, numRounds=10)