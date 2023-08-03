import numpy as np
from collections import Counter
from lpafunctions import *

# Assuming you have an array of numbers
data = np.array([1, 2, 3, 2, 4, 3, 3, 4, 4])

# Compute the frequency of each value in the array
value_counts = Counter(data)

# Find the maximum frequency (mode)
max_frequency = max(value_counts.values())

# Get all values with the maximum frequency (modes)
modes = [value for value, count in value_counts.items() if count == max_frequency]

# Randomly select one of the modes
selected_mode = np.random.choice(modes)

print("Modes:", modes)
print("Selected Mode:", selected_mode)


filter_data = (data < 3)
print(filter_data)
print(np.multiply(data, filter_data))

N = 10
numComm = 2
p = 1
q = 0
test_ER = ER(N, p)
A = nx.adjacency_matrix(test_ER).todense()
print(A)
print("")
test_SBM = SBM_default(N, numComm, p, q)
B = nx.adjacency_matrix(test_SBM).todense()
print(B)
# relabel nodes
# Create the dictionary where the keys are range(N) and values are the random permutation
newLabels_dict = {k: v for k, v in zip(range(N), np.random.permutation(N))}
print("")
print(newLabels_dict)
test_SBM = nx.relabel_nodes(test_SBM, newLabels_dict, copy=True)
print("")
B = nx.adjacency_matrix(test_SBM).todense()
print(B)

fig, ax = plt.subplots()
im = ax.imshow(B)
plt.show()










G = test_SBM

# Get the adjacency matrix
adj_matrix = nx.adjacency_matrix(G).toarray()

# Get the number of nodes in the graph
num_nodes = G.number_of_nodes()

# Create a random permutation of the node labels
random_permutation = np.random.permutation(num_nodes)

# Create a mapping of old node labels to new shuffled labels
mapping = {old_label: new_label for old_label, new_label in enumerate(random_permutation)}

# Relabel the nodes in the graph with the shuffled labels
shuffled_G = nx.relabel_nodes(G, mapping)

# Get the adjacency matrix of the shuffled graph
shuffled_adj_matrix = nx.adjacency_matrix(shuffled_G).toarray()

print("Original adjacency matrix:")
print(adj_matrix)

print("Shuffled adjacency matrix:")
print(shuffled_adj_matrix)









r = 0.5 # split between sizes of communities (defaults to half and half)
communities = np.random.rand(N)
communities = (communities >= r)
sizes = np.ones(N) # pretend every vertex is its own community 
# construct probs matrix
probs = np.ones((N, N)) * p 
np.fill_diagonal(np.zeros(N)) # no vertex is adjacent to itself
probs[:, np.nonzero(communities)]

