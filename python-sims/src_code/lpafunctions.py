import numpy as np
from numpy import array
import networkx as nx
import matplotlib.pyplot as plt
from scipy import stats as st
from collections import Counter
import matplotlib.colors as mcolors



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
    sizes = np.ones(numCommunities) * (N // numCommunities)
    for i in range(remainder):
        ind = np.random.randint(numCommunities)
        sizes[ind] = sizes[ind] + 1
    sizes = array(sizes)
    sizes = sizes.astype('i')
    probs = np.ones((numCommunities, numCommunities)) * q
    np.fill_diagonal(probs, np.ones(numCommunities) * p)
    return nx.stochastic_block_model(sizes, probs)



def generate_randomized_stochastic_block_model(N, numCommunities, p, q):
    """
    Generates a graph in the stochastic block model with randomized communities.

    Parameters:
    N (int): Number of nodes in the graph.
    p (float): Probability of intra-community edges.
    q (float): Probability of inter-community edges.

    Returns:
    nx.Graph: A randomized graph in the stochastic block model.
    """
    # Generate community assignments for each node
    communities = np.random.randint(0, numCommunities, N)

    # Create an empty graph
    G = nx.Graph()

    # Add nodes to the graph with the assigned community
    G.add_nodes_from((node, {"community": communities[node]}) for node in range(N))

    # Add edges based on community assignments
    for u in range(N):
        for v in range(u + 1, N):
            prob = p if communities[u] == communities[v] else q
            if np.random.random() < prob:
                G.add_edge(u, v)

    return G



def generate_randomized_stochastic_block_model_with_comm(N, numCommunities, p, q, communities):
    """
    Generates a graph in the stochastic block model with randomized communities.

    Parameters:
    N (int): Number of nodes in the graph.
    p (float): Probability of intra-community edges.
    q (float): Probability of inter-community edges.

    Returns:
    nx.Graph: A randomized graph in the stochastic block model.
    """
    # Create an empty graph
    G = nx.Graph()

    # Add nodes to the graph with the assigned community
    G.add_nodes_from((node, {"community": communities[node]}) for node in range(N))

    # Add edges based on community assignments
    for u in range(N):
        for v in range(u + 1, N):
            prob = p if communities[u] == communities[v] else q
            if np.random.random() < prob:
                G.add_edge(u, v)

    return G



# MinRandLPA(G, cap)
"""
Inputs:
    G = graph
    cap = (optional) max number of iterations allowed before termination of algorithm, 20 by default

Outputs:
    history = N by (iteration + 1) matrix with the state history of the simulation
    iteration = Last completed iteration (as in history[iteration] = last UNIQUE round)
"""
def MinRandLPA(G, cap=20):
    N = nx.number_of_nodes(G)
    # initialize history; 0th column is initial labels 1 through N
    history = np.array(list(nx.nodes(G)))
    history = history[:, np.newaxis]
    #history = np.reshape(list(nx.nodes(G)), (-1, 1)) # reshape [1:N] to column shape
    # initialize iteration
    iteration = 1
    # variable to hold the next label for each vertex; default to its own label
    nextLabels = np.copy(history[:, 0])
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
    history_copy = np.copy(history)
    history = np.hstack((history_copy, nextLabels))
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



# MinMinLPA(G, cap)
"""
Runs LPA where ties are broken towards the minimum label
Inputs:
    G = graph
    cap = (optional) max number of iterations allowed before termination of algorithm, 20 by default

Outputs:
    history = N by (iteration + 1) matrix with the state history of the simulation
    iteration = Last completed iteration
"""
def MinMinLPA(G, cap=20):
    N = nx.number_of_nodes(G)
    # initialize history; 0th column is initial labels 1 through N
    history = np.array(list(nx.nodes(G)))
    history = history[:, np.newaxis]
    # initialize iteration
    iteration = 1
    # variable to hold the next label for each vertex; default to its own label
    nextLabels = np.copy(history[:, 0])
    # round 1 iteration; break ties to min
    for i in range(N):
        # get list of neighbors
        neighbors = [n for n in G.neighbors(i)]
        # only overwrite nextLabels if vertex has neighbors
        if np.size(neighbors) != 0:
            nextLabels[i] = np.min([np.min(neighbors), i])
    nextLabels = np.reshape(nextLabels, (-1, 1))
    history_copy = np.copy(history)
    history = np.hstack((history_copy, nextLabels))
    # iterate until convergence or cap; break ties uniformly at random
    while iteration <= cap:
        for i in range(N):
            # get list of neighbors
            neighbors = [n for n in G.neighbors(i)]
            # only overwrite nextLabels if vertex has neighbors
            if np.size(neighbors) != 0:
                neighborLabels = history[neighbors, iteration]
                nextLabels[i] = __min_mode(neighborLabels)
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
Conducts binary search for threshold value of LPA convergence (following procedure in Pfeister thesis) on v wher p = N^v. 
To avoid excessively long output, the algorithm terminates once v is precise to +/- 1/N
Inputs:
    numTrials = number of independent trials run each time we calculate the proportion of trials that reached consensus
        defaults to 32, as in Pfeister thesis
    testNValues = array of values of N for which a threshold value will be estimated
        defaults to an array indexed 0 through 10 (length 11), where testNValues[i] = 1e5*(2^i)
    cap = (optional) max number of iterations allowed (per trial) (i.e. if a trial does not reach consensus after [cap] iterations,
        we consider it not in consensus, even if it would have reached consensus after more iterations), 20 by default
    
Outputs:
    array estThresholdDegrees where estThresholds[i] is the estimated threshold degree value (note this is Np, not p)
    array estThresholdVs where estThresholdVs is the estimated threshold v value (p = N^v)
"""
def ER_BinSearchThreshold_v(numTrials, testNValues, cap=20):
    # array to hold the estimated values of np for each value of N
    estThresholdDegrees = np.zeros(np.size(testNValues))
    # array to hold the estimated values of v for each value of N
    estThresholdVs = np.zeros(np.size(testNValues))
    # loop through all values of N
    for i in range(np.size(testNValues)):
        N = testNValues[i]
        # initialize value of v to be -1
        curr_v = -1.0
        # initialize current step size to 0.5
        curr_step = 0.5
        # counter to hold the number of trials that reached consensus
        numConsensus = 0
        # keep track of proportion of trials reaching consensus
        proportionCons = 0
        # create file object to write to
        file_object = open('N%d_ERBinSearch_output.txt' % N, 'w')
        file_object.write("N = %d\n" % N)
        # keep track of all the values of v
        v_arr = [curr_v]
        # run binary search until half of the trials reach consensus or curr_step is within 1/N
        while (np.abs(numConsensus - numTrials/2) >= 0.5 and curr_step > 1/N):
            file_object.write("curr_v = %f\n" % curr_v)
            # if less than half of the trials reached consensus, increase the value of v
            if proportionCons < 0.5:
                curr_v += curr_step
                v_arr.append(curr_v)
            # if more than half of the trials reached consensus, decrease the value of v
            else:
                curr_v -= curr_step
                v_arr.append(curr_v)
            # halve step size
            curr_step /= 2
            file_object.write("curr_v %f\n" % curr_v)
            # reset the number of trials that reached consensus to 0
            numConsensus = 0
            # reset the proportion of trials reaching consensus
            proportionCons = 0
            # run trials
            for trial in range(numTrials):
                G = ER(N, np.power(N, curr_v))
                history, iteration = MinRandLPA(G, cap)
                # if reached consensus (i.e. only one surviving label), increment numConsensus
                if np.size(SurvivingLabels(history[:,-1])) == 1:
                    numConsensus += 1
                proportionCons = numConsensus / (trial + 1)
                #file_object.write("trial = %d\n" % trial)
                #file_object.write("proportionCons = %f\n" % proportionCons)
                # if at any point after the first quarter of the trials the proportion of consensus-reaching trials is <= 0.2 or => 0.8, terimnate
                if trial >= numTrials/4:
                    if proportionCons <= 0.2 or proportionCons >= 0.8:
                        #file_object.write("numConsensus/trials = ", numConsensus/(trial+1))
                        file_object.write("trials run = %d\n" % (trial+1))
                        break
            file_object.write("numConsensus = %d\n" % numConsensus)
            file_object.write("proportionCons = %f\n" % proportionCons)
        #file_object.write("final curr_v = %f" % curr_v)
        estThresholdDegrees[i] = N * np.power(N, curr_v)
        file_object.write("estThresholdDegree = %f\n" % estThresholdDegrees[i])
        estThresholdVs[i] = curr_v
        file_object.write("estThresholdV = %f\n" % estThresholdVs[i])
        file_object.write("v_arr:")
        file_object.write(str(v_arr))
        file_object.write("\n")
        file_object.write("length of v_arr = %d\n" % np.size(v_arr))
        # plot v_arr
        plotV(N, v_arr)
        file_object.close()
    return estThresholdDegrees, estThresholdVs



# plotV(v_arr)
"""
Plots values of v from binary search
Inputs:
    v_arr = array containing tested values of v
Outputs:
    None (generates and saves plot)
"""
def plotV(N, v_arr):
    x = [i for i in range(np.size(v_arr))]
    y = np.abs(v_arr)
    plt.plot(x, y)
    plt.yscale('log')
    plt.title('Tested values of v for N = %d' % N)
    plt.xlabel('Test')
    plt.ylabel('Absolute Value of v')
    plt.tight_layout()
    #plt.show()
    plt.savefig('vPlot_N%d' % N)
    plt.clf()



# ER_BinSearchThreshold_p(numTrials, testNValues)
"""
Conducts binary search for threshold value of LPA convergence (following procedure in Pfeister thesis) on p 
To avoid excessively long output, the algorithm terminates once v is precise to +/- 1/N
Inputs:
    numTrials = number of independent trials run each time we calculate the proportion of trials that reached consensus
        defaults to 32, as in Pfeister thesis
    testNValues = array of values of N for which a threshold value will be estimated
        defaults to an array indexed 0 through 10 (length 11), where testNValues[i] = 1e5*(2^i)
    cap = (optional) max number of iterations allowed (per trial) (i.e. if a trial does not reach consensus after [cap] iterations,
        we consider it not in consensus, even if it would have reached consensus after more iterations), 20 by default
    
Outputs:
    array estThresholdDegrees where estThresholds[i] is the estimated threshold degree value (note this is Np, not p)
    array estThresholdPs where estThresholdVs is the estimated threshold p value 
"""
def ER_BinSearchThreshold_p(numTrials, testNValues, cap=20):
    # array to hold the estimated values of np for each value of N
    estThresholdDegrees = np.zeros(np.size(testNValues))
    # array to hold the estimated values of p for each value of N
    estThresholdPs = np.zeros(np.size(testNValues))
    # loop through all values of N
    for i in range(np.size(testNValues)):
        N = testNValues[i]
        # initialize value of p to be 0
        curr_p = 0
        # initialize current step size to 0.5
        curr_step = 0.5
        # counter to hold the number of trials that reached consensus
        numConsensus = 0
        # keep track of proportion of trials reaching consensus
        proportionCons = 0
        # create file object to write to
        file_object = open('N%d_ERBinSearch_onP_output.txt' % N, 'w')
        file_object.write("N = %d\n" % N)
        # keep track of all the values of p
        p_arr = [curr_p]
        # run binary search until half of the trials reach consensus or curr_step is within 1/(log(N)*N)
        while (np.abs(numConsensus - numTrials/2) >= 0.5 and curr_step > 1/(np.log2(N)*N)):
            file_object.write("curr_p = %f\n" % curr_p)
            # if less than half of the trials reached consensus, increase the value of p
            if proportionCons < 0.5:
                curr_p += curr_step
                p_arr.append(curr_p)
            # if more than half of the trials reached consensus, decrease the value of p
            else:
                curr_p -= curr_step
                p_arr.append(curr_p)
            # halve step size
            curr_step /= 2
            file_object.write("curr_p %f\n" % curr_p)
            # reset the number of trials that reached consensus to 0
            numConsensus = 0
            # reset the proportion of trials reaching consensus
            proportionCons = 0
            # run trials
            for trial in range(numTrials):
                G = ER(N, curr_p)
                history, iteration = MinRandLPA(G, cap)
                # if reached consensus (i.e. only one surviving label), increment numConsensus
                if np.size(SurvivingLabels(history[:,-1])) == 1:
                    numConsensus += 1
                proportionCons = numConsensus / (trial + 1)
                #file_object.write("trial = %d\n" % trial)
                #file_object.write("proportionCons = %f\n" % proportionCons)
                # if at any point after the first quarter of the trials the proportion of consensus-reaching trials is <= 0.2 or => 0.8, terimnate
                if trial >= numTrials/4:
                    if proportionCons <= 0.2 or proportionCons >= 0.8:
                        #file_object.write("numConsensus/trials = ", numConsensus/(trial+1))
                        file_object.write("trials run = %d\n" % (trial+1))
                        break
            file_object.write("numConsensus = %d\n" % numConsensus)
            file_object.write("proportionCons = %f\n" % proportionCons)
        #file_object.write("final curr_p = %f" % curr_p)
        estThresholdDegrees[i] = N * curr_p
        file_object.write("estThresholdDegree = %f\n" % estThresholdDegrees[i])
        estThresholdPs[i] = curr_p
        file_object.write("estThresholdp = %f\n" % estThresholdPs[i])
        file_object.write("p_arr:")
        file_object.write(str(p_arr))
        file_object.write("\n")
        file_object.write("length of p_arr = %d\n" % np.size(p_arr))
        # plot p_arr
        plotP(N, p_arr)
        file_object.close()
    return estThresholdDegrees, estThresholdPs



# MinMin_ER_BinSearchThreshold_p(numTrials, testNValues)
"""
Conducts binary search for threshold value of LPA convergence (following procedure in Pfeister thesis) on p 
To avoid excessively long output, the algorithm terminates once v is precise to +/- 1/N
Inputs:
    numTrials = number of independent trials run each time we calculate the proportion of trials that reached consensus
        defaults to 32, as in Pfeister thesis
    testNValues = array of values of N for which a threshold value will be estimated
        defaults to an array indexed 0 through 10 (length 11), where testNValues[i] = 1e5*(2^i)
    cap = (optional) max number of iterations allowed (per trial) (i.e. if a trial does not reach consensus after [cap] iterations,
        we consider it not in consensus, even if it would have reached consensus after more iterations), 20 by default
    
Outputs:
    array estThresholdDegrees where estThresholds[i] is the estimated threshold degree value (note this is Np, not p)
    array estThresholdPs where estThresholdVs is the estimated threshold p value 
"""
def MinMin_ER_BinSearchThreshold_p(numTrials, testNValues, cap=20):
    # array to hold the estimated values of np for each value of N
    estThresholdDegrees = np.zeros(np.size(testNValues))
    # array to hold the estimated values of p for each value of N
    estThresholdPs = np.zeros(np.size(testNValues))
    # loop through all values of N
    for i in range(np.size(testNValues)):
        N = testNValues[i]
        print('N = %d' % N)
        # initialize value of p to be 0
        curr_p = 0
        # initialize current step size to 0.5
        curr_step = 0.5
        # counter to hold the number of trials that reached consensus
        numConsensus = 0
        # keep track of proportion of trials reaching consensus
        proportionCons = 0
        # create file object to write to
        file_object = open('N%d_MinMin_ERBinSearch_onP_output.txt' % N, 'w')
        file_object.write("N = %d\n" % N)
        # keep track of all the values of p
        p_arr = [curr_p]
        # run binary search until half of the trials reach consensus or curr_step is within 1/(log(N)*N)
        while (np.abs(numConsensus - numTrials/2) >= 0.5 and curr_step > 1/(np.log2(N)*N)):
            file_object.write("curr_p = %f\n" % curr_p)
            print("curr_p = %f\n" % curr_p)
            # if less than half of the trials reached consensus, increase the value of p
            if proportionCons < 0.5:
                curr_p += curr_step
                p_arr.append(curr_p)
            # if more than half of the trials reached consensus, decrease the value of p
            else:
                curr_p -= curr_step
                p_arr.append(curr_p)
            # halve step size
            curr_step /= 2
            file_object.write("curr_p %f\n" % curr_p)
            print("curr_p = %f\n" % curr_p)
            # reset the number of trials that reached consensus to 0
            numConsensus = 0
            # reset the proportion of trials reaching consensus
            proportionCons = 0
            # run trials
            for trial in range(numTrials):
                G = ER(N, curr_p)
                history, iteration = MinMinLPA(G, cap)
                # if reached consensus (i.e. only one surviving label), increment numConsensus
                if np.size(SurvivingLabels(history[:,-1])) == 1:
                    numConsensus += 1
                proportionCons = numConsensus / (trial + 1)
                # if at any point after the first quarter of the trials the proportion of consensus-reaching trials is <= 0.2 or => 0.8, terimnate
                if trial >= numTrials/4:
                    if proportionCons <= 0.2 or proportionCons >= 0.8:
                        #file_object.write("numConsensus/trials = ", numConsensus/(trial+1))
                        file_object.write("trials run = %d\n" % (trial+1))
                        print("trials run = %d\n" % (trial+1))
                        break
            file_object.write("numConsensus = %d\n" % numConsensus)
            print("numConsensus = %d\n" % numConsensus)
            file_object.write("proportionCons = %f\n" % proportionCons)
            print("proportionCons = %f\n" % proportionCons)
        estThresholdDegrees[i] = N * curr_p
        file_object.write("estThresholdDegree = %f\n" % estThresholdDegrees[i])
        print("estThresholdDegree = %f\n" % estThresholdDegrees[i])
        estThresholdPs[i] = curr_p
        file_object.write("estThresholdp = %f\n" % estThresholdPs[i])
        print("estThresholdp = %f\n" % estThresholdPs[i])
        file_object.write("p_arr:")
        print("p_arr:")
        file_object.write(str(p_arr))
        print(str(p_arr))
        file_object.write("\n")
        file_object.write("length of p_arr = %d\n" % np.size(p_arr))
        print("length of p_arr = %d\n" % np.size(p_arr))
        # plot p_arr
        plotP(N, p_arr)
        file_object.close()
    return estThresholdDegrees, estThresholdPs



# plotP(p_arr)
"""
Plots values of p from binary search
Inputs:
    p_arr = array containing tested values of p
Outputs:
    None (generates and saves plot)
"""
def plotP(N, p_arr):
    x = [i for i in range(np.size(p_arr))]
    y = np.abs(p_arr)
    plt.plot(x, y)
    plt.yscale('log')
    plt.title('Tested values of p for N = %d' % N)
    plt.xlabel('Test')
    plt.ylabel('Absolute value of p')
    plt.tight_layout()
    #plt.show()
    plt.savefig('pPlot_N%d' % N)
    plt.clf()



# generatePQHeatmap(N, size, numTrials)
"""
Generates heatmap of number of surviving labels for the stochastic block model with parameters p = N^a, q = N^b, where a, b between -1 and 0
Regenerates random graph (and random communities) each time
Inputs:
    N = number of nodes to build graphs with
    size = number of p, q values each
    numTrials = (optional) number of trials for each (p, q) pair, defaults to 8
Outputs:
    2D array numLabels_data where heatmapData[j][i] is the average number of surviving labels for p = N^(-i/(size-1)), q = N^(-j/(size-1))
    2D array largestNonzeroLabel_data where heatmapData[j][i] is the average largest surviving label for p = N^(-i/(size-1)), q = N^(-j/(size-1))
"""
def generatePQHeatmap(N, numCommunities, size, numTrials, cap=20, type='minRand'):
    # initialize 2D array
    numLabels_data = np.zeros((size, size))
    maxNonzeroLabel_data = np.zeros((size, size))
    numIterations_data = np.zeros((size, size))
    switching_data = np.zeros((size, size))
    # --- SPLASH ZONE< THIS IS MESSY ---
    intraCommMin_data = []
    for _ in range(numCommunities):
        intraCommMin_data.append(np.zeros((cap + 1, size, size)))
    # --- END SPLASH ZONE ---
    correctness_data = np.zeros((size, size))
    b_arr = np.linspace(0, -1, num = size)
    a_arr = np.linspace(0, -1, num = size)
    for j in range(size):
        for i in range(size):
            numLabels_sum = 0
            maxNonzero_sum = 0
            numIterations_sum = 0
            switching_sum = 0
            correctness_sum = 0
            for trial in range(numTrials):
                p = 1/np.power(N, np.abs(a_arr[i]))
                q = 1/np.power(N, np.abs(b_arr[j]))
                #G = SBM_default(N, numCommunities, p, q)
                # G = generate_randomized_stochastic_block_model(N, numCommunities, p, q)

                # generate communities
                # --- this was to get *exactly* half of the vertices in one community
                # communities = np.zeros(N)
                # inds = np.random.choice(np.arange(N), int(N/numCommunities), replace=False)
                # communities[inds] = 1
                # ------
                communities = np.random.randint(numCommunities, size=N)
                # generate graph
                G = generate_randomized_stochastic_block_model_with_comm(N, numCommunities, p, q, communities)

                if type == 'minRand':
                    history, iteration = MinRandLPA(G, cap)
                elif type == 'minMin':
                    history, iteration = MinMinLPA(G, cap)
                else :
                    TypeError('type argument must be either minRand or minMin')
                surviving = SurvivingLabels(history[:,-1])
                numLabels_sum += np.size(surviving)
                maxNonzero_sum += np.max(surviving)
                numIterations_sum += iteration
                switching_sum += isSwitching(history)
                correctness_sum += isCorrect(history, communities)
                # update intraCommMin_data
                intraCommMin_data = commMinConcentrations(N, numTrials, history, iteration, intraCommMin_data, communities, j, i, cap)
            numLabels_data[j][i] = numLabels_sum / numTrials
            maxNonzeroLabel_data[j][i] = maxNonzero_sum / numTrials
            numIterations_data[j][i] = numIterations_sum / numTrials
            switching_data[j][i] = switching_sum / numTrials
            correctness_data[j][i] = correctness_sum / numTrials
    plotNumLabelsHeatmap(N, numCommunities, size, numTrials, numLabels_data, type)
    plotMaxNonzeroHeatmap(N, numCommunities, size, numTrials, maxNonzeroLabel_data, type)
    plotNumIterationsHeatmap(N, numCommunities, size, numTrials, numIterations_data, cap, type)
    plotSwitchingHeatmap(N, numCommunities, size, numTrials, switching_data, cap, type)
    plotCommMinHeatmap(N, numCommunities, size, numTrials, intraCommMin_data, cap, type)
    plotCorrectnessHeatmap(N, numCommunities, size, numTrials, correctness_data, cap, type)
    # filter out values greater than numCommunities; replace with -1
    filtered_numLabels_data = numLabels_data
    filtered_numLabels_data = np.multiply(filtered_numLabels_data, (filtered_numLabels_data <= numCommunities))
    plotFiltNumLabelsHeatmap(N, numCommunities, size, numTrials, filtered_numLabels_data, type)
    return numLabels_data, maxNonzeroLabel_data



# switched(history)



# plotNumLabelsHeatmap(N, numCommmunities, numTrials, numLabels_data)
"""
Plots and saves heatmap of number avg # of surviving labels
"""
def plotNumLabelsHeatmap(N, numCommunities, size, numTrials, numLabels_data, type):
    print('plotting numLabels_heatmap_N%d_%dcomm_%dtrials_%s' % (N, numCommunities, numTrials, type))
    b_arr = np.linspace(0, -1, num = size)
    a_arr = np.linspace(0, -1, num = size)

    fig, ax = plt.subplots()
    im = ax.imshow(numLabels_data)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(a_arr)))#, labels=a_arr)
    ax.set_yticks(np.arange(len(b_arr)))#, labels=b_arr)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")
    
    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)

    # Loop over data dimensions and create text annotations.
    for i in range(len(b_arr)):
        for j in range(len(a_arr)):
            text = ax.text(i, j, numLabels_data[j, i],
                        fontsize = 4, ha="center", va="center", color="w")

    ax.set_title("(%sLPA) Average # of Surviving Labels on SBM where \nN = %d on %d Communities Over %d Trials" % (type, N, numCommunities, numTrials))
    ax.set_xlabel('a where p = N^(-a/32)')
    ax.set_ylabel('b where q = N^(-b/32)')
    fig.tight_layout()
    #plt.show()
    plt.savefig('numLabels_heatmap_N%d_%dcomm_%dtrials_%s' % (N, numCommunities, numTrials, type))
    plt.close()



# plotFiltNumLabelsHeatmap(N, numCommmunities, numTrials, numLabels_data)
"""
Plots and saves heatmap of number avg # of surviving labels, filtered outfor the values that are 
"""
def plotFiltNumLabelsHeatmap(N, numCommunities, size, numTrials, numLabels_data, type):
    print('plotting numLabelsFilt_heatmap_N%d_%dcomm_%dtrials' %(N, numCommunities, numTrials))
    b_arr = np.linspace(0, -1, num = size)
    a_arr = np.linspace(0, -1, num = size)

    fig, ax = plt.subplots()
    im = ax.imshow(numLabels_data)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(a_arr)))#, fontsize = 5)#, labels=a_arr)
    ax.set_yticks(np.arange(len(b_arr)))#, fontsize = 5)#, labels=b_arr)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")
    
    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)

    # Loop over data dimensions and create text annotations.
    for i in range(len(b_arr)):
        for j in range(len(a_arr)):
            text = ax.text(i, j, numLabels_data[j, i],
                        fontsize = 4, ha="center", va="center", color="w")

    ax.set_title("(%sLPA) Average # of Surviving Labels (Filtered) on SBM where \nN = %d on %d Communities Over %d Trials" % (type, N, numCommunities, numTrials))
    ax.set_xlabel('a where p = N^(-a/32)')
    ax.set_ylabel('b where q = N^(-b/32)')
    fig.tight_layout()
    #plt.show()
    plt.savefig('numLabelsFilt_heatmap_N%d_%dcomm_%dtrials_%s' %(N, numCommunities, numTrials, type))
    plt.close()



# plotMaxNonzeroHeatmap(N, numCommmunities, numTrials, maxNonzero_data)
"""
Plots and saves heatmap of average maximum surviving label
"""
def plotMaxNonzeroHeatmap(N, numCommunities, size, numTrials, maxNonzero_data, type):
    print('plotting maxSurviving_heatmap_N%d_%dcomm_%dtrials' %(N, numCommunities, numTrials))
    b_arr = np.linspace(0, -1, num = size)
    a_arr = np.linspace(0, -1, num = size)

    fig, ax = plt.subplots()
    im = ax.imshow(maxNonzero_data)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(a_arr)))#, labels=a_arr)
    ax.set_yticks(np.arange(len(b_arr)))#, labels=b_arr)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")
    
    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)

    # Loop over data dimensions and create text annotations.
    for i in range(len(b_arr)):
        for j in range(len(a_arr)):
            text = ax.text(i, j, maxNonzero_data[j, i],
                        fontsize = 4, ha="center", va="center", color="w")

    ax.set_title("(%sLPA) Average Max Surviving Label on SBM where \nN = %d on %d Communities with %d Trials" % (type, N, numCommunities, numTrials))
    ax.set_xlabel('-32*a where p = N^(-a/32)')
    ax.set_ylabel('b where q = N^(-b/32)')
    fig.tight_layout()
    #plt.show()
    plt.savefig('maxSurviving_heatmap_N%d_%dcomm_%dtrials_%s' %(N, numCommunities, numTrials, type))
    plt.close()



# plotNumIterationsHeatmap(N, numCommmunities, numTrials, maxNonzero_data)
"""
Plots and saves heatmap of average total number of iterations
"""
def plotNumIterationsHeatmap(N, numCommunities, size, numTrials, numIterations_data, cap, type):
    print('plotting numIterations_heatmap_N%d_%dcomm_%dtrials' %(N, numCommunities, numTrials))
    b_arr = np.linspace(0, -1, num = size)
    a_arr = np.linspace(0, -1, num = size)

    fig, ax = plt.subplots()
    im = ax.imshow(numIterations_data)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(a_arr)))#, labels=a_arr)
    ax.set_yticks(np.arange(len(b_arr)))#, labels=b_arr)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")
    
    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)

    # Loop over data dimensions and create text annotations.
    for i in range(len(b_arr)):
        for j in range(len(a_arr)):
            text = ax.text(i, j, numIterations_data[j, i],
                        fontsize = 4, ha="center", va="center", color="w")

    ax.set_title("(%sLPA) Average Total Number of Iterations (cap %d) on SBM where \nN = %d on %d Communities with %d Trials" % (type, cap, N, numCommunities, numTrials))
    ax.set_xlabel('-32*a where p = N^(-a/32)')
    ax.set_ylabel('b where q = N^(-b/32)')
    fig.tight_layout()
    #plt.show()
    plt.savefig('numIterations_heatmap_N%d_%dcomm_%dtrials_%s' %(N, numCommunities, numTrials, type))
    plt.close()



# plotSwitchingHeatmap(N, numCommmunities, numTrials, switching_data)
"""
Plots and saves heatmap of average switching
"""
def plotSwitchingHeatmap(N, numCommunities, size, numTrials, switching_data, cap, type):
    print('plotting switching_heatmap_N%d_%dcomm_%dtrials' %(N, numCommunities, numTrials))
    b_arr = np.linspace(0, -1, num = size)
    a_arr = np.linspace(0, -1, num = size)

    fig, ax = plt.subplots()
    im = ax.imshow(switching_data)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(a_arr)))#, labels=a_arr)
    ax.set_yticks(np.arange(len(b_arr)))#, labels=b_arr)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")
    
    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)

    # Loop over data dimensions and create text annotations.
    for i in range(len(b_arr)):
        for j in range(len(a_arr)):
            text = ax.text(i, j, switching_data[j, i],
                        fontsize = 4, ha="center", va="center", color="w")

    ax.set_title("(%sLPA) Average Switching (cap %d) on SBM where \nN = %d on %d Communities with %d Trials\n(0: no switching, 1: (precise) switching)" % (type, cap, N, numCommunities, numTrials))
    ax.set_xlabel('a where p = N^(-a/32)')
    ax.set_ylabel('b where q = N^(-b/32)')
    fig.tight_layout()
    #plt.show()
    plt.savefig('switching_heatmap_N%d_%dcomm_%dtrials_%s' % (N, numCommunities, numTrials, type))
    plt.close()



# plotCorrectnessHeatmap(N, numCommmunities, numTrials, correctness_data)
"""
Plots and saves heatmap of average correctness
"""
def plotCorrectnessHeatmap(N, numCommunities, size, numTrials, correctness_data, cap, type):
    print('plotting switching_heatmap_N%d_%dcomm_%dtrials' %(N, numCommunities, numTrials))
    b_arr = np.linspace(0, -1, num = size)
    a_arr = np.linspace(0, -1, num = size)

    fig, ax = plt.subplots()
    im = ax.imshow(correctness_data)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(a_arr)))#, labels=a_arr)
    ax.set_yticks(np.arange(len(b_arr)))#, labels=b_arr)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")
    
    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)

    # Loop over data dimensions and create text annotations.
    for i in range(len(b_arr)):
        for j in range(len(a_arr)):
            text = ax.text(i, j, correctness_data[j, i],
                        fontsize = 4, ha="center", va="center", color="w")

    ax.set_title("(%sLPA) Average Correctness (cap %d) on SBM where \nN = %d on %d Communities with %d Trials\n(0: incorrect communities, 1: correct communities" % (type, cap, N, numCommunities, numTrials))
    ax.set_xlabel('a where p = N^(-a/32)')
    ax.set_ylabel('b where q = N^(-b/32)')
    fig.tight_layout()
    #plt.show()
    plt.savefig('correctness_heatmap_N%d_%dcomm_%dtrials_%s' % (N, numCommunities, numTrials, type))
    plt.close()



# plotCommMinHeatmap(N, numCommunities, size, numTrials, intraCommMin_data, cap, type)
"""
Plots and saves heatmap of the proportion of vertices that are labeled with an intra-community minimum label.
Note that this will generate cap images, each containing numCommunities heatmaps.
In each image, the kth heatmap from the left will display the average proportion of vertices that are labeled with the kth smallest label out of the set of all intra-community minimum labels.
"""
def plotCommMinHeatmap(N, numCommunities, size, numTrials, intraCommMin_data, cap, type):
    # Create a custom colormap that interpolates between yellow and purple
    cmap_custom = mcolors.LinearSegmentedColormap.from_list("Custom", [(1, 1, 0), (0.5, 0, 0.5)])
    for round in range(cap + 1):
        fig, axes = plt.subplots(1, numCommunities, figsize=(numCommunities * 10, 10))
        # Add a centered title above both subplots
        fig.suptitle('Proportion of Vertices with k-Smallest Intra-Community Label, Round %d of %d\n(%s, N = %d on %d Communities with %d Trials)' % (round, cap, type, N, numCommunities, numTrials), fontsize=16)
        # plot heatmap for rank-th smallest intra-community minimum label
        for rank in range(numCommunities):
            heatmap_data = intraCommMin_data[rank][round]
            im = axes[rank].imshow(heatmap_data, cmap=cmap_custom, interpolation='nearest', vmin=0, vmax=1)
            axes[rank].set_title('%d-Smallest Intra-Comm Label' % rank)
            axes[rank].set_xlabel('a where p = N^(-a/32)')
            axes[rank].set_ylabel('b where q = N^(-b/32)')
            plt.colorbar(im, ax=axes[rank], fraction=0.046, pad=0.04)

            for i in range(size):
                for j in range(size):
                    t = '%0.2f' % heatmap_data[j, i]
                    axes[rank].text(i, j, t,
                        fontsize = 4, ha="center", va="center", color="w")
                    
            axes[rank].set_xticks(np.arange(size))
            axes[rank].set_yticks(np.arange(size))
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        plt.savefig('commMin_round%dof%d_heatmap_N%d_%dcomm_%dtrials_%s' % (round, cap, N, numCommunities, numTrials, type))
        plt.close()




# labelHistory(N, numCommunities, p, q, numRounds=5)
"""
Generates bar graph/histogram for labels over numRounds rounds of the LPA
Inputs:
    N = number of nodes
    numCommunities = number of communities
    p = inter-community adjacency probability
    q = intra-community adjacency probability
    numRounds = (optional) input that specifies the number of rounds to generate
Outputs:
    None (saves plot to file)
"""
def labelHistory(N, numCommunities, a, b, numRounds=5):
    # communities = np.random.randint(numCommunities, size=N)
    communities = np.zeros(N)
    inds = np.random.choice(np.arange(N), int(N/numCommunities), replace=False)
    communities[inds] = 1

    print('community == 0:')
    print(len(communities[communities == 0]))
    print('community == 1:')
    print(len(communities[communities == 1]))

    p = N**(a)
    q = N**(b)

    # p = a * np.log(N)/N
    # q = b * np.log(N)/N
    # print('p = %f' % p)
    # print('q = %f' % q)

    G = generate_randomized_stochastic_block_model_with_comm(N, numCommunities, p, q, communities)
    history, iteration = MinRandLPA(G, cap=numRounds)

    # print('history:')
    # print(history)

    # if the round is larger than the last iteration before convergence
    totalRounds = np.min((numRounds, iteration)) + 1 # round 0
    bar_data = np.zeros((totalRounds, numCommunities, numCommunities))

    # find smallest label in each community
    smallestLabels = np.zeros(numCommunities)
    for community in range(numCommunities):
        member_indices = np.where(communities == community)[0]
        smallestLabels[community] = member_indices[0]
    
    for round in range(totalRounds):
        round_hist = history[:, round]
        for community in range(numCommunities):
            # separate out by community
            member_indices = np.where(communities == community)[0]
            filtered_hist = round_hist[member_indices]
            for labelInd in range(numCommunities):
                bar_data[round, community, labelInd] = np.sum(filtered_hist == smallestLabels[labelInd])
    
    # print('bar_data:')
    # print(bar_data)

    generate_stacked_bar_graphs(N, numRounds, totalRounds, numCommunities, a, b, smallestLabels, bar_data, N)



def generate_stacked_bar_graphs(N, maxRounds, numRounds, numCommunities, a, b, smallestLabels, data_list, y_limit):
    # Create a figure with subplots
    fig, axes = plt.subplots(nrows=1, ncols=numRounds)

    # Set y-axis limits
    for ax in axes:
        ax.set_ylim(0, y_limit)
    
    # Loop through each round and create a stacked bar graph
    for round_idx, ax in enumerate(axes):
        data = data_list[round_idx]
        dataCopy = np.copy(data).T
        # num_vars = len(data)
        bar_width = 0.8
        
        bottom = np.zeros(numCommunities)
        for var_idx, var_data in enumerate(dataCopy):
            ax.bar(np.arange(numCommunities), var_data, bar_width, label=f'Label {int(smallestLabels[var_idx])}', bottom=bottom)
            bottom += var_data
        
        ax.set_title(f'Round {round_idx}')
        #ax.set_yscale('log')
        ax.set_xticks(np.arange(numCommunities))
        ax.set_xticklabels([f'C{i + 1}' for i in range(numCommunities)])
        ax.legend()
    
    # set title
    #ax.text(0.5, 0.5, 'Label History (N=%d, numComms=%d, \np=%.3f, q=%.3f, maxRounds=%d)' % (N, numCommunities, p, q, maxRounds), 
    #         horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes)

    #plt.title('a=%.3f, b=%.3f' % (a, b))

    text = fig.text(0.50, 0.02, 'a=%.3f, b=%.3f' % (a, b), horizontalalignment='center', wrap=True ) 

    # Adjust layout
    plt.tight_layout()

    plt.show()
    
    # save
    #plt.savefig('labelHistoryBars_N%d_%dcomm_a%.3f_b%.3f_%dmaxRounds.png' % (N, numCommunities, a, b, maxRounds))

    plt.clf()
    plt.close()



# isSwitching(N, numCommunities, p, q, cap, type, history)
"""
Returns a value of 0 (false) or 1 or 2 (true) depending on whether this particular run of the algorithm exhibited switching
Inputs:
    [most are the same as always]
    history = N by k array (k = number of iterations before convergence)
Outputs: 0 if not switching, 1 if yes switching (precise label)
"""
def isSwitching(history):
    if (history.shape[1] < 3):
        return 0
    lastRound = history[:, -1]
    thirdToLast = history[:, -3]
    if (np.array_equal(lastRound, thirdToLast)):
        return 1
    return 0



# isCorrect(history, communities)
"""
Returns proportion of vertices where last round of history matches the original communities
"""
def isCorrect(history, communities):
    # if np.array_equal(history[:,-1], communities):
    #     return 1
    # return 0
    N = len(communities)
    count1 = 0
    count2 = 0
    for i in range(N):
        if history[i, -1] == communities[i]: count1 += 1
        if history[i, -2] == communities[i]: count2 += 1
    return np.max([count1, count2]) / N



# commMinConcentrations(history, intraCommMin_sum)
"""
Processes history to count the proportion of vertices with the smallest label (ascending order), and updates intraCommMin_sum
"""
def commMinConcentrations(N, numTrials, history, iteration, intraCommMin_data, communities, j, i, cap):
    # first find the smallest labels in each community
    numCommunities = len(np.unique(communities))
    smallestLabels = np.zeros(numCommunities)
    for comm in np.unique(communities):
        member_indices = np.where(communities == comm)[0]
        smallestLabels[int(comm)] = member_indices[0]
    # sort the labels so that they are in ascending order
    smallestLabels = np.sort(smallestLabels)
    # for each label that is the smallest in its community, starting from the global minimum...
    for rank in range(len(smallestLabels)):
        # copy over block; should be of size cap x size x size
        block = intraCommMin_data[rank]
        # ... for each round of the algorithm...
        for round in range(cap+1):
            # sometimes history will have fewer iterations than cap allows because it will converge before; if round > last iteration, just keep choosing the last iteration
            histIndex = min(iteration, round)
            round_hist = history[:, histIndex]
            label = smallestLabels[rank]
            # ...add the proportion of vertices with the label of that rank after that round (averaged over the number of trials)
            block[round, j, i] += (np.count_nonzero(round_hist==label) / N) / numTrials
        # copy back block to intraComm data
        intraCommMin_data[rank] = block
    # return updated data
    return intraCommMin_data



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



# __min_mode(data)
"""
Helper method to select minimum mode of an array
Inputs:
    data = array of numbers
Outputs:
    mode, breaking ties uniformly at random
"""
def __min_mode(data):
    # Compute the frequency of each value in the array
    value_counts = Counter(data)

    # Find the maximum frequency (mode)
    max_frequency = max(value_counts.values())

    # Get all values with the maximum frequency (modes)
    modes = [value for value, count in value_counts.items() if count == max_frequency]

    # Randomly select one of the modes
    return np.min(modes)






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