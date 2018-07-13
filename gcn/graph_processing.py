import numpy as np
from numpy import linalg as LA
import time
"""
Helper class for graph computations/algorithms
"""

# Constant running time


# paths_to_known_list return has the following format :
# node 0 [[num known neighbors, num second hop known neighbors, ....],
# node 1  [num known neighbors, num second hop known neighbors, ....],
#...
# node N  [num known neighbors, num second hop known neighbors, ....]]
def get_num_paths_to_known(adj: np.array, list_known_node: list, MAX_ITER=3) -> np.array:
    paths_to_known_list = np.zeros((adj.shape[0], MAX_ITER))
    all_nodes_are_reached = False  # flag for every node as at least one path to a known node
    num_hop = 1
    adj_power = np.identity(adj.shape[0])
    while (not (all_nodes_are_reached or MAX_ITER < num_hop)):
        # raise the adjacency matrix to have the number of path of lenght num_hop
        adj_power = np.matmul(adj_power, adj)
        paths_to_known_list[:, num_hop - 1] = np.sum(adj_power[list_known_node], axis=0)  # count the path to known node
        all_nodes_are_reached = np.all(np.sum(paths_to_known_list,
                                              axis=1))  # check if all nodes have a path that reaches info
        num_hop += 1
    return paths_to_known_list[:, 0:num_hop - 1]  # cut stopped hop number
