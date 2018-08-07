import numpy as np
import matplotlib.pyplot as plt
from graph_generator import generate_Erdos_Renyi_graph, get_sparse_eigen_decomposition
from linear_H import get_identity_H
from signal_generator import get_random_signal_zero_mean_circular
from sampling_algo import greedy_algo, brute_force_algo, get_relative_suboptimality
from sampling_algo_util import get_W
np.set_printoptions(precision=2)

# SIMULATION PARAMS

NUM_NODES = 20  # Size of graph generated
NOISE_CONSTANT = 10e2
K_sparse = 5  # Set sparsity of the signal frequence
number_node_sampled = 5
NUM_SIMULATIONS = 10

# RESULT STORING ARRAYS

relative_sub_Erdos_greedy = []
relative_sub_Erdos_randomized = []

#TODO //
for simul_iter in range(NUM_SIMULATIONS):
    # Generate the graphs. Ensures that every node has at least one neighboors.
    Erdos_graph = generate_Erdos_Renyi_graph(NUM_NODES)

    # Compute spectral properties of graphs.
    V_ksparse, V_ksparse_H, get_v = get_sparse_eigen_decomposition(Erdos_graph, K_sparse)

    # Linear transformation of the signal
    H, H_h = get_identity_H(NUM_NODES)

    # Random signal and noise vectors
    x, cov_x = get_random_signal_zero_mean_circular(1, NUM_NODES)
    w, cov_w = get_random_signal_zero_mean_circular(NOISE_CONSTANT, NUM_NODES)

    # Noisy observation
    y = x + w

    # Pre computation
    W = get_W(V_ksparse_H, H_h, H, V_ksparse)

    # Get sampling selected by greedy algorithm
    greedy_subset = greedy_algo(V_ksparse, V_ksparse_H, get_v, H, H_h, cov_x, cov_w, W, number_node_sampled, NUM_NODES)

    optimal_subset, subset_scores = brute_force_algo(V_ksparse, V_ksparse_H, get_v, H, H_h, cov_x, cov_w, W,
                                                     number_node_sampled, NUM_NODES)
    empty_set_K_T = subset_scores[str([])]
    optimal_K_T = subset_scores[str(list(sorted(optimal_subset)))]
    greedy_K_T = subset_scores[str(list(sorted(greedy_subset)))]

    relative_sub_Erdos_greedy.append(get_relative_suboptimality(optimal_K_T, greedy_K_T, empty_set_K_T))

n_bins = 30
fig, axs = plt.subplots(1, 3, sharey=True)
axs[0].set_ylabel('Count')
axs[0].hist(relative_sub_Erdos_greedy, bins=n_bins)
axs[0].set_title("Erdos-Renyi")

axs[1].set_ylabel('Count')
axs[1].hist(relative_sub_Erdos_greedy, bins=n_bins)
axs[1].set_title("Pref. attachment")

axs[2].set_ylabel('Count')
axs[2].hist(relative_sub_Erdos_greedy, bins=n_bins)
axs[2].set_title("Random")
fig.tight_layout()

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=3, hspace=0.5, wspace=0.3)

plt.savefig("test")