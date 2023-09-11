import numpy as np
import matplotlib.pyplot as plt
# from collections import Counter
from lpafunctions import *
import networkx as nx

# # Assuming you have an array of numbers
# data = np.array([1, 2, 3, 2, 4, 3, 3, 4, 4])

# # Compute the frequency of each value in the array
# value_counts = Counter(data)

# # Find the maximum frequency (mode)
# max_frequency = max(value_counts.values())

# # Get all values with the maximum frequency (modes)
# modes = [value for value, count in value_counts.items() if count == max_frequency]

# # Randomly select one of the modes
# selected_mode = np.random.choice(modes)

# print("Modes:", modes)
# print("Selected Mode:", selected_mode)


# filter_data = (data < 3)
# print(filter_data)
# print(np.multiply(data, filter_data))

# N = 10
# numComm = 2
# p = 1
# q = 0
# test_ER = ER(N, p)
# A = nx.adjacency_matrix(test_ER).todense()
# print(A)
# print("")
# test_SBM = SBM_default(N, numComm, p, q)
# B = nx.adjacency_matrix(test_SBM).todense()
# print(B)
# # relabel nodes
# # Create the dictionary where the keys are range(N) and values are the random permutation
# newLabels_dict = {k: v for k, v in zip(range(N), np.random.permutation(N))}
# print("")
# print(newLabels_dict)
# test_SBM = nx.relabel_nodes(test_SBM, newLabels_dict, copy=True)
# print("")
# B = nx.adjacency_matrix(test_SBM).todense()
# print(B)

# fig, ax = plt.subplots()
# im = ax.imshow(B)
# plt.show()





# G = test_SBM

# # Get the adjacency matrix
# adj_matrix = nx.adjacency_matrix(G).toarray()

# # Get the number of nodes in the graph
# num_nodes = G.number_of_nodes()

# # Create a random permutation of the node labels
# random_permutation = np.random.permutation(num_nodes)

# # Create a mapping of old node labels to new shuffled labels
# mapping = {old_label: new_label for old_label, new_label in enumerate(random_permutation)}

# # Relabel the nodes in the graph with the shuffled labels
# shuffled_G = nx.relabel_nodes(G, mapping)

# # Get the adjacency matrix of the shuffled graph
# shuffled_adj_matrix = nx.adjacency_matrix(shuffled_G).toarray()

# print("Original adjacency matrix:")
# print(adj_matrix)

# print("Shuffled adjacency matrix:")
# print(shuffled_adj_matrix)









# r = 0.5 # split between sizes of communities (defaults to half and half)
# communities = np.random.rand(N)
# communities = (communities >= r)
# sizes = np.ones(N) # pretend every vertex is its own community 
# # construct probs matrix
# probs = np.ones((N, N)) * p 
# np.fill_diagonal(np.zeros(N)) # no vertex is adjacent to itself
# probs[:, np.nonzero(communities)]



# features = np.arange(4) + 1
# features = np.reshape(features, (1, 4))
# print(features)
# print(np.shape(features))
# trans = np.atleast_2d(features).T
# print(trans)
# print(np.shape(trans))
# mat = np.matmul(trans, features)
# print(mat)
# print(np.linalg.inv(mat))


# # generate x and y
# x = np.linspace(0, 1, 11)
# print(x)
# y = 1 + x + x * np.random.random(len(x))
# print(y)

# # assemble matrix A
# A = np.vstack([x, np.ones(len(x))]).T
# print(A)

# # turn y into a column vector
# y = y[:, np.newaxis]
# print(y)

# def generate_bar_graphs(numRounds, numCommunities):
#     # Generate random data for each bar graph
#     data = np.random.rand(numRounds, numCommunities)
    
#     # Create a figure with subplots
#     fig, axes = plt.subplots(nrows=1, ncols=numRounds)
    
#     # Loop through each round and create a bar graph
#     for round_idx, ax in enumerate(axes):
#         ax.bar(np.arange(numCommunities), data[round_idx])
#         ax.set_title(f'Round {round_idx + 1}')
#         ax.set_xticks(np.arange(numCommunities))
#         ax.set_xticklabels([f'C{i + 1}' for i in range(numCommunities)])
    
#     # Adjust layout
#     plt.tight_layout()
    
#     # Show the figure
#     plt.show()



# import matplotlib.pyplot as plt
# import numpy as np

# def generate_multicolored_bar_graphs(numRounds, numCommunities, data_list, y_limit):
#     # Create a figure with subplots
#     fig, axes = plt.subplots(nrows=1, ncols=numRounds)
    
#     # Set y-axis limits
#     for ax in axes:
#         ax.set_ylim(0, y_limit)
    
#     # Loop through each round and create a multicolored bar graph
#     for round_idx, ax in enumerate(axes):
#         data = data_list[round_idx]
#         num_vars = len(data)
#         bar_width = 0.8 / num_vars  # Adjust the width of each bar
        
#         for var_idx, var_data in enumerate(data):
#             x_pos = np.arange(numCommunities) + (var_idx - num_vars / 2) * bar_width
#             ax.bar(x_pos, var_data, bar_width, label=f'Variable {var_idx + 1}')
        
#         ax.set_title(f'Round {round_idx + 1}')
#         ax.set_xticks(np.arange(numCommunities))
#         ax.set_xticklabels([f'C{i + 1}' for i in range(numCommunities)])
#         ax.legend()
    
#     # Adjust layout
#     plt.tight_layout()
    
#     # Show the figure
#     plt.show()

# # Example data (replace this with your actual data)
# numRounds = 5
# numCommunities = 8
# numVariables = 3
# data_list = [np.random.rand(numVariables, numCommunities) for _ in range(numRounds)]
# y_limit = 1.5  # Set the y-axis limit

# generate_multicolored_bar_graphs(numRounds, numCommunities, data_list, y_limit)



# import matplotlib.pyplot as plt
# import numpy as np

# def generate_stacked_bar_graphs(numRounds, numCommunities, data_list, y_limit):
#     # Create a figure with subplots
#     fig, axes = plt.subplots(nrows=1, ncols=numRounds)

#     # Set y-axis limits
#     for ax in axes:
#         ax.set_ylim(0, y_limit)
    
#     # Loop through each round and create a stacked bar graph
#     for round_idx, ax in enumerate(axes):
#         data = data_list[round_idx]
#         num_vars = len(data)
#         bar_width = 0.8
        
#         bottom = np.zeros(numCommunities)
#         for var_idx, var_data in enumerate(data):
#             ax.bar(np.arange(numCommunities), var_data, bar_width, label=f'Label {var_idx + 1}', bottom=bottom)
#             bottom += var_data
        
#         ax.set_title(f'Round {round_idx + 1}')
#         ax.set_xticks(np.arange(numCommunities))
#         ax.set_xticklabels([f'C{i + 1}' for i in range(numCommunities)])
#         ax.legend()
    
#     # Adjust layout
#     plt.tight_layout()
    
#     # Show the figure
#     plt.show()

# # Example data (replace this with your actual data)
# numRounds = 5
# numCommunities = 8
# numVariables = 3
# data_list = [np.random.rand(numVariables, numCommunities) for _ in range(numRounds)] 
# # shape = (numRounds, numCommunities, numVariables)
# print(np.shape(data_list))
# y_limit = 5

# generate_stacked_bar_graphs(numRounds, numCommunities, data_list, y_limit)

# a = np.array([1, 2, 3, 4])
# print(a)
# b = a[:, np.newaxis]
# print(b)

# N = 10
# numCommunities = 1
# p = 0.15
# q = 0
# communities = np.zeros(10)
# cap = 5

# G = generate_randomized_stochastic_block_model_with_comm(N, numCommunities, p, q, communities)
# history, iteration = MinRandLPA(G, cap)

# print(history)
# print(iteration)


# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# import numpy as np

# # Define the number of images
# number_of_images = 3

# # Generate sample heatmap data (replace with your own data)
# heatmap_data = [np.random.rand(10, 10) for _ in range(number_of_images * 2)]

# # Create a custom colormap that interpolates between yellow and purple
# cmap_custom = mcolors.LinearSegmentedColormap.from_list("Custom", [(1, 1, 0), (0.5, 0, 0.5)])

# for image_number in range(number_of_images):
#     # Create a new figure and axes
#     fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
#     # Create the first subplot (left)
#     im1 = axes[0].imshow(heatmap_data[image_number * 2], cmap=cmap_custom, interpolation='nearest', vmin=0, vmax=1)
#     axes[0].set_title("Heatmap 1")
#     plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)  # Add colorbar
    
#     # Customize tick marks for the first subplot
#     axes[0].set_xticks(np.arange(0, 10, 1))  # Example: custom x-axis tick marks
#     axes[0].set_yticks(np.arange(0, 10, 2))  # Example: custom y-axis tick marks
    
#     # Create the second subplot (right)
#     im2 = axes[1].imshow(heatmap_data[image_number * 2 + 1], cmap=cmap_custom, interpolation='nearest', vmin=0, vmax=1)
#     axes[1].set_title("Heatmap 2")
#     plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)  # Add colorbar
    
#     # Customize tick marks for the second subplot
#     axes[1].set_xticks(np.arange(0, 10, 2))  # Example: custom x-axis tick marks
#     axes[1].set_yticks(np.arange(0, 10, 2))  # Example: custom y-axis tick marks
    
#     # Adjust layout for better spacing
#     plt.tight_layout()
    
#     # Save the figure with a unique filename
#     plt.savefig(f"heatmap_image_{image_number}.png")
    
#     # Close the figure to free up memory
#     plt.close()

# print(str(round(1.32403927, 3)))

# count = 0
# comms = 1
# N = 1000
# c = 1.5
# p = c * np.log(N)/N
# q = 0.6/N
# for _ in range(50):
#     G = generate_randomized_stochastic_block_model(N, comms, p, q)
#     if nx.is_connected(G): count += 1
# print(count)




#print('plotting numLabels_heatmap_N%d_%dcomm_%dtrials_%s' % (N, numCommunities, numTrials, type))
N = 1000000
comms = 2
c = 0.5
size = 33
b_arr = np.linspace(0, -1, num = size)
a_arr = np.linspace(0, -1, num = size)

data = np.zeros((size, size))
for i in range(size):
    for j in range(size):
        p = N**(a_arr[i])
        q = N**(b_arr[j])
        if (N/comms)*(p**2) > 8*N*q: data[j, i] += 1
        #if (N/comms)*(p**4) > 1800*c*np.log(N): data[j, i] += 1

fig, ax = plt.subplots()
im = ax.imshow(data)

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
        text = ax.text(i, j, data[j, i],
                    fontsize = 4, ha="center", va="center", color="w")

# ax.set_title("(%sLPA) Average # of Surviving Labels on SBM where \nN = %d on %d Communities Over %d Trials" % (type, N, numCommunities, numTrials))
ax.set_xlabel('a where p = N^(-a/32)')
ax.set_ylabel('b where q = N^(-b/32)')
fig.tight_layout()
plt.show()
plt.close()