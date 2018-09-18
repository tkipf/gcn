import numpy as np
from numpy.linalg import inv
from numpy.linalg import multi_dot
"""
Helper functions for sampling algorithms. 
"""


# W = V_k_H * H_H * H * V_K
def get_W(V_ksparse_H, H_h, H, V_ksparse):
    W = multi_dot([V_ksparse_H, H_h, H, V_ksparse])
    return W


# Returns the index of the best node to add to the sampling set
def argmax(K, W, cov_w, remaining_nodes, get_v):
    u = (-float("inf"), -1)  # (score, index) to keep track of the best node so far
    for candidate in remaining_nodes:
        v_u, v_u_H = get_v(candidate)
        numerator = multi_dot([v_u_H, K, W, K, v_u])
        lamda_inv = 1.0 / float(cov_w[candidate][candidate])
        denumerator = lamda_inv + multi_dot([v_u_H, K, v_u])
        score = numerator / denumerator
        if score > u[0]:
            u = (score, candidate)
    return u[1]


# Update the K*j matrix
def update_K(K, W, cov_w, u, get_v):
    v_u, v_u_H = get_v(u)
    numerator = multi_dot([W, K, v_u, v_u_H, K])
    lamda_inv = 1.0 / float(cov_w[u][u])
    denumerator = lamda_inv + multi_dot([v_u_H, K, v_u])
    matrix = numerator / denumerator
    return K - matrix

# Only used to get he initial best sore
def get_upper_bound_trace_K(W, cov_x):
    upper_bound_matrix = np.matrix(W * cov_x)
    return float(upper_bound_matrix.trace())


def get_MSE_score(V_ksparse, V_ksparse_H, get_v, H, H_h, cov_x, cov_w, possible_set):
    inv_cov_x = inv(cov_x)
    for i in possible_set:
        v_i, v_i_H = get_v(i)
        lamda_inv = 1.0 / float(cov_w[i][i])
        inv_cov_x = inv_cov_x + lamda_inv * multi_dot([v_i, v_i_H])
    K = multi_dot([get_W(V_ksparse_H, H_h, H, V_ksparse),inv(inv_cov_x)])
    return float(K.trace())
