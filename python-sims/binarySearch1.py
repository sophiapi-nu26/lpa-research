from lpafunctions import *
import numpy as np

# test = [[0, 3]
#         [1, 2]
#         [2, 9]
#         [3, 10]]

# print(test)

# test2 = test[]



N_values = np.zeros(11)
for i in range(np.size(N_values)):
    N_values[i] = (10000 * int(np.power(2, i)))
N_values = N_values.astype(np.int64)

N_values = [1000]
print("hello")
estThresholdDegrees, estThresholdVs = ER_BinSearchThreshold_v(32, N_values)
print(estThresholdDegrees)
print(estThresholdVs)