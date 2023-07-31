import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy import stats as st
from collections import Counter



# ER(N, p)
"""
Alias for nx.erdos_renyi_graph
Inputs:
    N = number of nodes
    p = probability that two nodes are adjacent
Outputs:
    Erdos-Renyi/binomial random graph G(N, p)
"""
def ER(N, p):
    G = nx.erdos_renyi_graph(N, p)
    return G



# SBM(sizes, probs)
"""
Alias for nx.stochastic_block_model
Inputs:
    sizes = array where sizes[i] = # of vertices in community i
    probs = 2D array where probs(i, j) = probability that vertex i and j are adjacent
Outputs:
    stochastic block model w/ given parameters
"""
def SBM(sizes, probs):
    return nx.stochastic_block_model(sizes, probs)    



# SBM_default(N, numCommunities, p, q)
"""
Inputs:
    N = number of nodes
    numCommunities = number of planted communities
    p = probability that two vertices within same community are adjacent
    q = probability that two vertices in different communities are adjacent
Outputs:
    Stochastic block model with numCommunities symmetric communities
"""
def SBM_default(N, numCommunities, p, q):
    remainder = N % numCommunities
    sizes = np.ones(numCommunities) * (N / numCommunities)
    for i in range(remainder):
        ind = np.random.randint(numCommunities)
        sizes[ind] = sizes[ind] + 1
    probs = np.ones((numCommunities, numCommunities)) * q
    np.fill_diagonal(probs, np.ones(numCommunities) * p)
    return nx.stochastic_block_model(sizes, probs)



# MinRandLPA(G, cap)
"""
Inputs:
    G = graph
    cap = (optional) max number of iterations allowed before termination of algorithm, 100 by default

Outputs:
    history = N by (iteration + 1) matrix with the state history of the simulation
    iteration = Last completed iteration
"""
def MinRandLPA(G, cap=100):
    N = nx.number_of_nodes(G)
    # initialize history; 0th column is initial labels 1 through N
    history = np.reshape(list(nx.nodes(G)), (-1, 1)) # reshape [1:N] to column shape
    # initialize iteration
    iteration = 1
    # variable to hold the next label for each vertex; default to its own label
    nextLabels = history[:, 0]
    # round 1 iteration; break ties to min
    for i in range(N):
        # get list of neighbors
        neighbors = [n for n in G.neighbors(i)]
        # only overwrite nextLabels if vertex has neighbors
        if np.size(neighbors) != 0:
            nextLabels[i] = np.min([np.min(neighbors), i])
    nextLabels = np.reshape(nextLabels, (-1, 1))
    # print("history:")
    # print(history)
    # print("nextLabels")
    # print(nextLabels)
    history = np.hstack((history, nextLabels))
    # iterate until convergence or cap; break ties uniformly at random
    while iteration <= cap:
        # print(iteration)
        # print(history)
        for i in range(N):
            # get list of neighbors
            neighbors = [n for n in G.neighbors(i)]
            # only overwrite nextLabels if vertex has neighbors
            if np.size(neighbors) != 0:
                neighborLabels = history[neighbors, iteration]
                nextLabels[i] = __mode(neighborLabels)
        history = np.hstack((history, nextLabels))
        # if the labels are the same, break
        if np.array_equal(history[:, -1], history[:, -2]):
            break
        # update iteration
        iteration += 1
    return history, iteration



# SurvivingLabels(finalLabels)
"""
Returns the surviving labels
Inputs:
    finalLabels = final labels (final column of history)
Outputs:
    array of surviving labels 
"""
def SurvivingLabels(finalLabels):
    return np.unique(finalLabels)



# LabelDist(finalLabels)
"""
Returns the distribution of labels (frequency)
Inputs:
    finalLabels = final labels (final column of history)
Outputs:
    array ld where ld[i] = # of vertices with label i at algo termination 
"""
def LabelDist(finalLabels):
    ld = np.zeros(np.size(finalLabels))
    surviving = SurvivingLabels(finalLabels)
    for sl in surviving:
        ld[sl] = np.count_nonzero(finalLabels == sl)
    return ld



# ER_BinSearchThreshold_v(numTrials, testNValues)
"""
Conducts binary search for threshold value of LPA convergence (following procedure in Pfeister thesis) on v wher p = N^v
Inputs:
    numTrials = number of independent trials run each time we calculate the proportion of trials that reached consensus
        defaults to 32, as in Pfeister thesis
    testNValues = array of values of N for which a threshold value will be estimated
        defaults to an array indexed 0 through 10 (length 11), where testNValues[i] = 1e5*(2^i)
    cap = (optional) max number of iterations allowed (per trial) (i.e. if a trial does not reach consensus after [cap] iterations,
        we consider it not in consensus, even if it would have reached consensus after more iterations), 100 by default
Outputs:
    array estThresholdDegrees where estThresholds[i] is the estimated threshold degree value (note this is Np, not p)
    array estThresholdVs where estThresholdVs is the estimated threshold v value (p = N^v)
"""
def ER_BinSearchThreshold_v(numTrials, testNValues, cap=100):
    # array to hold the estimated values of np for each value of N
    estThresholdDegrees = np.zeros(np.size(testNValues))
    # array to hold the estimated values of v for each value of N
    estThresholdVs = np.zeros(np.size(testNValues))
    # initialize value of v to be -1
    curr_v = -1.0
    # initialize current step size to 0.5
    curr_step = 0.5
    # counter to hold the number of trials that reached consensus
    numConsensus = 0
    # loop through all values of N
    for i in range(np.size(testNValues)):
        N = testNValues[i]
        print("N = ", N)
        # run binary search until half of the trials reach consensus
        while (np.abs(numConsensus - numTrials/2) >= 0.5):
            print("curr_v = ", curr_v)
            # if less than half of the trials reached consensus, increase the value of v
            if numConsensus < numTrials/2:
                curr_v += curr_step
            # if more than half of the trials reached consensus, decrease the value of v
            else:
                curr_v -= curr_step
            # halve step size
            curr_step /= 2
            print("curr_v = ", curr_v)
            # reset the number of trials that reached consensus to 0
            numConsensus = 0
            # run trials
            for trial in range(numTrials):
                G = ER(N, np.power(N, curr_v))
                history, iteration = MinRandLPA(G, cap)
                # if reached consensus (i.e. only one surviving label), increment numConsensus
                if np.size(SurvivingLabels(history[:,-1])) == 1:
                    numConsensus += 1
            print("numConsensus = ", numConsensus)
        print("final curr_v = ", curr_v)
        estThresholdDegrees[i] = N * np.power(N, curr_v)
        estThresholdVs[i] = curr_v
    return estThresholdDegrees, estThresholdVs



# other functions
"""
- number of connected components (nx.connected_components(G))
"""
            


# __mode(data)
"""
Helper method to randomly select mode of an array, which is a weirdly involved process
Inputs:
    data = array of numbers
Outputs:
    mode, breaking ties uniformly at random
"""
def __mode(data):
    # Compute the frequency of each value in the array
    value_counts = Counter(data)

    # Find the maximum frequency (mode)
    max_frequency = max(value_counts.values())

    # Get all values with the maximum frequency (modes)
    modes = [value for value, count in value_counts.items() if count == max_frequency]

    # Randomly select one of the modes
    return np.random.choice(modes)






###############     LANDFILL     #############


# # SBM(N, probs, sizes, communities)
# """
# Inputs:
#     N = number of nodes
#     probs = (symmetric) matrix of probabilities where probs(i, j) = P(vertex from community i is adjacent to vertex from community j).
#         Number of rows/cols must equal length of sizes or number of unique values in communities, depending on which is passed in
#     sizes = optional argument that lists the number of nodes in each community. Sum of elements must equal N
#     communities = optional argument that lists which community each vertex belongs to. Length must equal N
# Outputs:
#     Stochastic block model with parameters 
#     adjacency matrix of graph
# Note: if both sizes and communities are provided, defaults to using sizes (ignores communities)
# """
# def SBM(N, probs, sizes=None, communities=None):
#     if sizes == None and communities == None:
#         TypeError("SBM: both sizes and communities are None")
#     if sizes != None:
#         if np.sum(sizes) != N:
#             ValueError("SBM: sum of values in sizes does not equal N")
#         return nx.stochastic_block_model(sizes, probs)
#     if np.size(communities) != N:
#         ValueError("SBM: length of communities is not equal to N")