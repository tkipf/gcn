import numpy as np
import networkx as nx
from itertools import combinations
from random import randrange
from numpy.linalg import inv
import scipy.sparse as sp

Erdos_Renyi_Prob = 0.2


def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    #d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    #return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).toarray()
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


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
    Erdos_Renyi_graph = nx.erdos_renyi_graph(n, Erdos_Renyi_Prob)
    return Erdos_Renyi_graph


#preferential attachment model, in which nodes are added
# one at a time and connected to a node already in the graph with
# probability proportional to its degree
def generate_pref_attachment_graph(n):
    m0 = 1
    Pref_Attach_graph = nx.barabasi_albert_graph(n, m0)
    return Pref_Attach_graph


def generate_random_graph(n):
    Random_graph = nx.erdos_renyi_graph(n, 0.5)
    return Random_graph
