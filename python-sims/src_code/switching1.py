# generate histogram

from lpafunctions import *

N_arr = [100]
numCommunities_arr = [1, 2, 3, 4]

# a_arr = np.linspace(-1, -0.7, 4)
# b_arr = np.linspace(-0.3, 0, 4)

# a_arr = np.linspace(0, -1, 16)
# b_arr = np.linspace(0, -1, 16)

a_arr = [-1]
b_arr = [0]
numCommunities_arr = [2]

for N in N_arr:
    for numCommunities in numCommunities_arr:
        for a in a_arr:
            for b in b_arr:
                labelHistory(N, numCommunities, a, b, numRounds=8)