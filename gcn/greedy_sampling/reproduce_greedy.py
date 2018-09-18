import numpy as np
from graph_generator import get_sparse_eigen_decomposition, plot_graph
from linear_H import get_identity_H
from signal_generator import get_random_signal_zero_mean_circular
from sampling_algo import greedy_algo, brute_force_algo, get_relative_suboptimality, leverage_algo, uniform_random_algo, random_leverage_algo
from sampling_algo_util import get_W
import time

SEED = 13


def simulate(graph_gen, num_iter, config):
    np.random.seed(seed=SEED)
    simul_result_dict = {'greedy': [], 'deterministic': [], 'random_leverage': [], 'uniform_random': []}
    test_time = time.time()
    for i in range(num_iter):
        print(str(i) + "/" + str(num_iter))
        # Generate the graphs.
        graph = graph_gen(config['NUM_NODES'])

        # Compute spectral properties of graphs.
        V_ksparse, V_ksparse_H, get_v = get_sparse_eigen_decomposition(graph, config['K_sparse'])

        # Linear transformation of the signal
        H, H_h = get_identity_H(config['NUM_NODES'])
        # Random signal and noise vectors
        x, cov_x = get_random_signal_zero_mean_circular(1.0, config['K_sparse'])
        w, cov_w = get_random_signal_zero_mean_circular(config['NOISE_CONSTANT'], config['NUM_NODES'])

        # Noisy observation. (Not used for now)
        #y = x + w

        # Pre computation
        W = get_W(V_ksparse_H, H_h, H, V_ksparse)

        # Get sampling set selected by the diff. algorithms
        greedy_subset = greedy_algo(get_v, cov_x, cov_w, W, config['number_node_sampled'], config['NUM_NODES'])

        leverage_subset = leverage_algo(V_ksparse, config['number_node_sampled'])

        random_leverage_subset = random_leverage_algo(V_ksparse, config['number_node_sampled'])

        uniform_random_subset = uniform_random_algo(config['number_node_sampled'], config['NUM_NODES'])

        # Get the optimal sampling set and the MSE of every possible set in a dict.
        optimal_subset, subset_scores = brute_force_algo(V_ksparse, V_ksparse_H, get_v, H, H_h, cov_x, cov_w, W,
                                                         config['number_node_sampled'], config['NUM_NODES'])
        # Find the trace of K (MSE) of the subsets.
        empty_set_K_T = subset_scores[str([])]
        optimal_K_T = subset_scores[str(list(sorted(optimal_subset)))]

        greedy_K_T = subset_scores[str(list(sorted(greedy_subset)))]
        leverage_K_T = subset_scores[str(list(sorted(leverage_subset)))]
        random_leverage_K_T = subset_scores[str(list(sorted(random_leverage_subset)))]
        uniform_random_K_T = subset_scores[str(list(sorted(uniform_random_subset)))]

        # Compute the relative sub. of each subsets
        score_greedy = get_relative_suboptimality(optimal_K_T, greedy_K_T, empty_set_K_T)
        score_leverage = get_relative_suboptimality(optimal_K_T, leverage_K_T, empty_set_K_T)
        score_random_leverage = get_relative_suboptimality(optimal_K_T, random_leverage_K_T, empty_set_K_T)
        score_uniform_random = get_relative_suboptimality(optimal_K_T, uniform_random_K_T, empty_set_K_T)
        if score_greedy > 0.15:
            print("not suppose to happen often.")
        print(greedy_subset)
        print(optimal_subset)
        print("Greedy : " + str(score_greedy) + " Deterministic : " + str(score_leverage) + " random_leverage : " +
              str(score_random_leverage) + " uniform_random : " + str(score_uniform_random))
        # print("OPTIMAL : " + str(optimal_K_T) + " EMPTY : " + str(empty_set_K_T))
        simul_result_dict['greedy'].append(score_greedy)
        simul_result_dict['deterministic'].append(score_leverage)
        simul_result_dict['random_leverage'].append(score_random_leverage)
        simul_result_dict['uniform_random'].append(score_uniform_random)
    simul_result_dict['time'] = time.time() - test_time
    return simul_result_dict
