import numpy as np


# Builds a signal with circular variance and zero mean
def get_random_signal_zero_mean_circular(var: int, num_nodes: int) -> (np.array, np.array):
    cov_matrix = var * np.identity(num_nodes)
    mean = np.zeros((cov_matrix.shape[0],))
    return np.random.multivariate_normal(mean, cov_matrix), cov_matrix
