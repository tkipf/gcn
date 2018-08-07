import numpy as np


def get_identity_H(num_nodes):
    H = np.matrix(np.identity(num_nodes))
    H_h = H.getH()
    return H, H_h