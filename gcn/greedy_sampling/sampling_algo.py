import numpy as np
from itertools import combinations
from sampling_algo_util import *
import numpy.linalg as LA


def greedy_algo(get_v, cov_x, cov_w, W, number_node_sampled, num_nodes):
    # Variable initialisation
    G_subset = []
    remaining_node = list(range(0, num_nodes))
    K = cov_x

    for j in range(number_node_sampled):
        u = argmax(K, W, cov_w, remaining_node, get_v)
        K = update_K(K, W, cov_w, u, get_v)

        G_subset.append(u)  # Iterativly add a new node to the set
        remaining_node.remove(u)
       
    return G_subset


def leverage_algo(V_ksparse, number_node_sampled):
    norms = LA.norm(V_ksparse, axis=1)
    leverage_subset = np.argsort(norms)[-number_node_sampled:]
    return list(leverage_subset)


def random_leverage_algo(V_ksparse, number_node_sampled):
    norms = LA.norm(V_ksparse, axis=1)
    normalized_norms = norms / sum(norms)
    choices = list(range(0, V_ksparse.shape[0]))
    list_random_leverage = np.random.choice(choices, number_node_sampled, replace=False, p=normalized_norms)
    return list(list_random_leverage)


def uniform_random_algo(number_node_sampled, num_nodes):
    choices = list(range(0, num_nodes))
    return [int(i) for i in np.random.choice(choices, number_node_sampled, replace=False)]


def brute_force_algo(V_ksparse, V_ksparse_H, get_v, H, H_h, cov_x, cov_w, W, number_node_sampled, num_nodes):
    subset_scores = {}
    optimal_subset = []
    optimal_K_T = get_upper_bound_trace_K(W, cov_x)
    c = False
    all_possible_set_combination = combinations(range(num_nodes), number_node_sampled)
    for possible_set in all_possible_set_combination:  # Try every subset
        score = get_MSE_score(V_ksparse, V_ksparse_H, get_v, H, H_h, cov_x, cov_w, possible_set)
        subset_scores[str(list(possible_set))] = score
        if score <= optimal_K_T:
            c = True
            optimal_K_T = score
            optimal_subset = possible_set
    if not c:
        print("whut")
    # Add the empty set score to compute performance metric
    empty_set_score = get_MSE_score(V_ksparse, V_ksparse_H, get_v, H, H_h, cov_x, cov_w, [])
    subset_scores[str([])] = empty_set_score

    return optimal_subset, subset_scores


def get_relative_suboptimality(optimal_K_T, f_K_T, empty_K_T):
    return (f_K_T - optimal_K_T) / (empty_K_T - optimal_K_T)
