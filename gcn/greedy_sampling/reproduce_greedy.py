import numpy as np
import matplotlib.pyplot as plt
from graph_generator import generate_Erdos_Renyi_graph, generate_pref_attachment_graph, get_sparse_eigen_decomposition
from linear_H import get_identity_H
from signal_generator import get_random_signal_zero_mean_circular
from sampling_algo import greedy_algo, brute_force_algo, get_relative_suboptimality, leverage_algo, uniform_random_algo, random_leverage_algo
from sampling_algo_util import get_W
import pickle as pk

np.set_printoptions(precision=2)

# SIMULATION PARAMS

NUM_NODES = 20  # Size of graph generated
NOISE_CONSTANT = 10e-2
K_sparse = 5  # Set sparsity of the signal frequence
number_node_sampled = 5
NUM_SIMULATIONS = 10

# RESULT STORING ARRAYS

relative_sub_Erdos = {}
relative_sub_Pref = {}

#TODO // TELL JONAS IS A POTATO
for graph_gen, result_dict in [(generate_Erdos_Renyi_graph, relative_sub_Erdos), (generate_pref_attachment_graph,
                                                                                  relative_sub_Pref)]:
    result_dict['greedy'] = []
    result_dict['deterministic'] = []
    result_dict['random_leverage'] = []
    result_dict['uniform_random'] = []
    for simul_iter in range(NUM_SIMULATIONS):
        # Generate the graphs.
        graph = graph_gen(NUM_NODES)
        # Compute spectral properties of graphs.
        V_ksparse, V_ksparse_H, get_v = get_sparse_eigen_decomposition(graph, K_sparse)

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
        greedy_subset = greedy_algo(V_ksparse, V_ksparse_H, get_v, H, H_h, cov_x, cov_w, W, number_node_sampled,
                                    NUM_NODES)
        leverage_subset = leverage_algo(V_ksparse, number_node_sampled)
        random_leverage_subset = random_leverage_algo(V_ksparse, number_node_sampled)
        uniform_random_subset = uniform_random_algo(number_node_sampled, NUM_NODES)
        optimal_subset, subset_scores = brute_force_algo(V_ksparse, V_ksparse_H, get_v, H, H_h, cov_x, cov_w, W,
                                                         number_node_sampled, NUM_NODES)
        empty_set_K_T = subset_scores[str([])]
        optimal_K_T = subset_scores[str(list(sorted(optimal_subset)))]

        greedy_K_T = subset_scores[str(list(sorted(greedy_subset)))]
        leverage_K_T = subset_scores[str(list(sorted(leverage_subset)))]
        random_leverage_K_T = subset_scores[str(list(sorted(random_leverage_subset)))]
        uniform_random_K_T = subset_scores[str(list(sorted(uniform_random_subset)))]

        score_greedy = get_relative_suboptimality(optimal_K_T, greedy_K_T, empty_set_K_T)
        score_leverage = get_relative_suboptimality(optimal_K_T, leverage_K_T, empty_set_K_T)
        score_random_leverage = get_relative_suboptimality(optimal_K_T, random_leverage_K_T, empty_set_K_T)
        score_uniform_random = get_relative_suboptimality(optimal_K_T, uniform_random_K_T, empty_set_K_T)

        print("Greedy : " + str(score_greedy) + " Deterministic : " + str(score_leverage) + " random_leverage : " +
              str(score_random_leverage) + " uniform_random : " + str(score_uniform_random))
        result_dict['greedy'].append(score_greedy)
        result_dict['deterministic'].append(score_leverage)
        result_dict['random_leverage'].append(score_random_leverage)
        result_dict['uniform_random'].append(score_uniform_random)

pk.dump({"results": [relative_sub_Erdos, relative_sub_Pref]}, open('results.p'))

# n_bins = 30
# fig, axs = plt.subplots(1, 3, sharey=True)
# for resuls_dit in [relative_sub_Erdos, relative_sub_Pref, relative_sub_Pref]:
#     axs[0].set_ylabel('Count')
#     axs[0].hist(resuls_dit['greedy'], bins=n_bins, alpha=0.3, color='r')
#     axs[0].hist(resuls_dit['deterministic'], bins=n_bins, color='k')
#     axs[0].hist(resuls_dit['random_leverage'], bins=n_bins, color='g')
#     axs[0].hist(resuls_dit['uniform_random'], bins=n_bins, color='b')

# fig.tight_layout()

# plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=3, hspace=0.5, wspace=0.3)

# plt.savefig("test")
