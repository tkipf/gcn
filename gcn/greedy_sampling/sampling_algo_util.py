import numpy as np
from numpy.linalg import inv


# W = V_k_H * H_H * H * K_k
def get_W(VH, H_h, H, V):
    a = np.matmul(VH, H_h)
    b = np.matmul(a, H)
    W = np.matmul(b, V)
    return W


# Returns the index of the best node to add to the sampling set
def argmax(K, W, cov_w, remaining_node, get_v):
    u = (-1, -1)  # score, index of the best node
    for candidate in remaining_node:
        v_u, v_u_H = get_v(candidate)
        a = (v_u_H * K)
        numerator = (((a * W) * K) * v_u)
        if numerator <=0:
            print(type(numerator))
            print(W)
            print(K)
        lamda_inv = 1.0 / float(cov_w[candidate][candidate])  # get lam^(-1)_w,u should always be the same
        denumerator = lamda_inv + (a * v_u)
        score = numerator / denumerator
        if score > u[0]:
            u = (score, candidate)
    return u[1]


def update_K(K, W, cov_w, u, get_v):  #Should be O(K^2)
    v_u, v_u_H = get_v(u)
    numerator = (((K * v_u) * v_u_H) * K)
    lamda_inv = 1.0 / float(cov_w[u][u])  # get lam^(-1)_w,u should always be the same
    denumerator = lamda_inv + ((v_u_H * K) * v_u)
    matrix = numerator / denumerator
    x = (W * matrix)
    return K - x


def get_upper_bound_trace_K(W, cov_x):
    upper_bound_matrix = np.matrix(W * cov_x)
    return float(upper_bound_matrix.trace())


def get_MSE_score(V_ksparse, V_ksparse_H, get_v, H, H_h, cov_x, cov_w, possible_set):
    inv_cov_x = inv(cov_x)
    for i in possible_set:
        v_i, v_i_H = get_v(i)
        lamda_inv = 1.0 / float(cov_w[i][i])
        inv_cov_x = inv_cov_x + lamda_inv * (v_i * v_i_H)
    K = np.matrix((((H * V_ksparse) * inv(inv_cov_x)) * V_ksparse_H) * H_h)
    return float(K.trace())
