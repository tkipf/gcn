import numpy as np
import networkx as nx
from itertools import combinations
from random import randrange
from numpy.linalg import inv
import scipy.sparse as sp


def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).toarray()


def get_sparse_eigen_decomposition(graph, K):
    adj = nx.adjacency_matrix(graph).toarray()
    normalized_adj = normalize_adj(adj)
    eigenval, eigenvectors = np.linalg.eig(normalized_adj)

    eigenval_Ksparse = np.argsort(eigenval)[-K:]
    V_ksparse = np.zeros(adj.shape)
    V_ksparse[:, eigenval_Ksparse] = eigenvectors[:, eigenval_Ksparse]

    V_ksparse = np.matrix(V_ksparse)
    V_ksparse_H = V_ksparse.getH()

    def get_v(index):
        v_index = V_ksparse_H[:, index]
        v_index_H = V_ksparse[index, :]
        return v_index, v_index_H

    return V_ksparse, V_ksparse_H, get_v


def plot_graph(graph):
    nx.draw_shell(
        graph,
        with_labels=True,
    )


# Erdos Renyi graph: Add an edge with prob = 0.2
def generate_Erdos_Renyi_graph(n):
    Erdos_Renyi_graph = nx.Graph()
    for node_pair in combinations(range(n), 2):  # Generate each possible egde
        if randrange(5) == 1:  # p = 0.2
            Erdos_Renyi_graph.add_edge(node_pair[0], node_pair[1])

    if (len(Erdos_Renyi_graph.nodes()) < n):  # Recursivly generate a new graph until all nodes are connected
        del Erdos_Renyi_graph  #  Delete the graph to avoid using memeroy unnecessarily
        return generate_Erdos_Renyi_graph(n)

    return Erdos_Renyi_graph
