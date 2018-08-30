import numpy as np
from graph_generator import generate_Erdos_Renyi_graph, generate_pref_attachment_graph, generate_random_graph, get_sparse_eigen_decomposition, plot_graph
from linear_H import get_identity_H
from signal_generator import get_random_signal_zero_mean_circular
from sampling_algo import greedy_algo, brute_force_algo, get_relative_suboptimality, leverage_algo, uniform_random_algo, random_leverage_algo
from sampling_algo_util import get_W
import time
import pickle as pk
import multiprocessing as mp
import sys
import networkx as nx
np.set_printoptions(precision=8)

# SIMULATION PARAMS
NUM_NODES = 20  # Size of graph generated
NOISE_CONSTANT = 0.01
K_sparse = 5  # Set sparsity of the signal frequence
number_node_sampled = 5
NUM_SIMULATIONS = 100
CORES = 6
want_multiprocessing = True
SEED = 13

# INFO ABOUT SIMUL TO STORE
simul_info = {
    "num_nodes": NUM_NODES,
    "noise": NOISE_CONSTANT,
    "K": K_sparse,
    "simul_num": NUM_SIMULATIONS,
    'time':
        time.time()  # will be updated at the end
}

# RESULT STORING ARRAYS
relative_sub_Erdos = {'greedy': [], 'deterministic': [], 'random_leverage': [], 'uniform_random': []}
relative_sub_Pref = {'greedy': [], 'deterministic': [], 'random_leverage': [], 'uniform_random': []}
reltive_sub_Random = {'greedy': [], 'deterministic': [], 'random_leverage': [], 'uniform_random': []}


def simulate(graph_gen, num_iter):
    np.random.seed(seed=SEED)

    simul_result_dict = {'greedy': [], 'deterministic': [], 'random_leverage': [], 'uniform_random': []}
    test_time = time.time()
    for i in range(num_iter):

        # Generate the graphs.
        graph = graph_gen(NUM_NODES)

        # Compute spectral properties of graphs.
        V_ksparse, V_ksparse_H, get_v = get_sparse_eigen_decomposition(graph, K_sparse)

        # Linear transformation of the signal
        H, H_h = get_identity_H(NUM_NODES)
        # Random signal and noise vectors
        x, cov_x = get_random_signal_zero_mean_circular(1.0, NUM_NODES)
        w, cov_w = get_random_signal_zero_mean_circular(NOISE_CONSTANT, NUM_NODES)

        # Noisy observation. (Not used for now)
        y = x + w

        # Pre computation
        W = get_W(V_ksparse_H, H_h, H, V_ksparse)

        # Get sampling set selected by the diff. algorithms
        greedy_subset, K = greedy_algo(get_v, cov_x, cov_w, W, number_node_sampled, NUM_NODES)

        leverage_subset = leverage_algo(V_ksparse, number_node_sampled)

        random_leverage_subset = random_leverage_algo(V_ksparse, number_node_sampled)

        uniform_random_subset = uniform_random_algo(number_node_sampled, NUM_NODES)

        # Get the optimal sampling set and the MSE of every possible set in a dict.
        optimal_subset, subset_scores = brute_force_algo(V_ksparse, V_ksparse_H, get_v, H, H_h, cov_x, cov_w, W,
                                                            number_node_sampled, NUM_NODES)
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
        # print("Greedy : " + str(score_greedy) + " Deterministic : " + str(score_leverage) + " random_leverage : " +
        #       str(score_random_leverage) + " uniform_random : " + str(score_uniform_random))
        # print("OPTIMAL : " + str(optimal_K_T) + " EMPTY : " + str(empty_set_K_T))
        simul_result_dict['greedy'].append(score_greedy)
        simul_result_dict['deterministic'].append(score_leverage)
        simul_result_dict['random_leverage'].append(score_random_leverage)
        simul_result_dict['uniform_random'].append(score_uniform_random)
    simul_result_dict['time'] = time.time() - test_time
    return simul_result_dict
    


for graph_gen, result_dict in [(generate_random_graph, reltive_sub_Random),
                               (generate_Erdos_Renyi_graph, relative_sub_Erdos), (generate_pref_attachment_graph,
                                                                                  relative_sub_Pref)]:
    if want_multiprocessing:
        num_iter = int(NUM_SIMULATIONS / CORES)
        pool = mp.Pool(processes=CORES)
        pool_results = [pool.apply_async(simulate, (graph_gen, num_iter)) for indices in range(CORES)]
        pool.close()
        pool.join()
        for pr in pool_results:
            dict_simul = pr.get()
            print(dict_simul['time'])
            result_dict['greedy'] += (dict_simul['greedy'])
            result_dict['deterministic'] += (dict_simul['deterministic'])
            result_dict['random_leverage'] += (dict_simul['random_leverage'])
            result_dict['uniform_random'] += (dict_simul['uniform_random'])

    else:
        dicts = simulate(graph_gen, NUM_SIMULATIONS)
        print(dicts)
        result_dict['greedy'].append(dicts['greedy'])
        result_dict['deterministic'].append(dicts['deterministic'])
        result_dict['random_leverage'].append(dicts['random_leverage'])
        result_dict['uniform_random'].append(dicts['uniform_random'])

done = time.time()
elapsed = done - simul_info['time']
simul_info['time'] = elapsed
print("elapsed =" + str(elapsed))
# Store the results of the simul in a pickle file.
filename = 'results_' + str(NOISE_CONSTANT) + '.p'
pk.dump({
    "results": [relative_sub_Erdos, relative_sub_Pref, reltive_sub_Random],
    'params_simul': simul_info
}, open((filename), 'wb'))
