import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

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

# Parameters for the stochastic block model
N = 20  # Number of nodes
p = 1  # Probability of intra-community edges
q = 0  # Probability of inter-community edges

# Generate the graph with randomized communities
graph = generate_randomized_stochastic_block_model(N, 2, p, q)

# Print the communities of each node
print(nx.get_node_attributes(graph, "community"))

print("")
B = nx.adjacency_matrix(graph).todense()
print(B)

print("")
print(np.sum(B, axis = 1))

fig, ax = plt.subplots()
im = ax.imshow(B)
plt.show()