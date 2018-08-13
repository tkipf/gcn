import numpy as np
from graph_generator import generate_Erdos_Renyi_graph, generate_pref_attachment_graph, generate_random_graph, get_sparse_eigen_decomposition
from linear_H import get_identity_H
from signal_generator import get_random_signal_zero_mean_circular
from sampling_algo import greedy_algo, brute_force_algo, get_relative_suboptimality, leverage_algo, uniform_random_algo, random_leverage_algo
from sampling_algo_util import get_W
import time
import pickle as pk
from multiprocessing import Process, Queue

np.set_printoptions(precision=2)

# SIMULATION PARAMS

NUM_NODES = 20  # Size of graph generated
NOISE_CONSTANT = 10e-2
K_sparse = 5  # Set sparsity of the signal frequence
number_node_sampled = 5
NUM_SIMULATIONS = 8
CORES = 4

simul_info = {
    "num_nodes": NUM_NODES,
    "noise": NOISE_CONSTANT,
    "K": K_sparse,
    "simul_num": NUM_SIMULATIONS,
    'time': time.time()
}
# RESULT STORING ARRAYS
relative_sub_Erdos = {}
relative_sub_Pref = {}
reltive_sub_Random = {}


def simulate(graph_gen, result_dict):
    for i in range(NUM_SIMULATIONS):
        # Generate the graphs.
        graph = graph_gen(NUM_NODES)
        # Compute spectral properties of graphs.
        V_ksparse, V_ksparse_H, get_v = get_sparse_eigen_decomposition(graph, K_sparse)

        # Linear transformation of the signal
        H, H_h = get_identity_H(NUM_NODES)

        # Random signal and noise vectors
        x, cov_x = get_random_signal_zero_mean_circular(1.0, NUM_NODES)
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
        result_dict['greedy'].put(score_greedy)
        result_dict['deterministic'].put(score_leverage)
        result_dict['random_leverage'].put(score_random_leverage)
        result_dict['uniform_random'].put(score_uniform_random)

multiprocessing_want = False

for graph_gen, result_dict in [(generate_Erdos_Renyi_graph, relative_sub_Erdos),
                               (generate_pref_attachment_graph, relative_sub_Pref), (generate_random_graph,
                                                                                     reltive_sub_Random)]:
    result_dict['greedy'] = []
    result_dict['deterministic'] = []
    result_dict['random_leverage'] = []
    result_dict['uniform_random'] = []
    if multiprocessing_want : 
        dicts = [{
            'greedy': Queue(),
            'deterministic': Queue(),
            'random_leverage': Queue(),
            'uniform_random': Queue()
        } for i in range(CORES)]
        args = [(graph_gen, dicts[i]) for i in range(CORES)]
        jobs = [Process(target=simulate, args=(a)) for a in args]
        for j in jobs:
            j.start()
        for d in dicts:
            result_dict['greedy'].append(d['greedy'].get())
            result_dict['deterministic'].append(d['deterministic'].get())
            result_dict['random_leverage'].append(d['random_leverage'].get())
            result_dict['uniform_random'].append(d['uniform_random'].get())
        for j in jobs:
            j.join()
    else:
        dicts = {
            'greedy': Queue(),
            'deterministic': Queue(),
            'random_leverage': Queue(),
            'uniform_random': Queue()
        }
        simulate(graph_gen,dicts)
        result_dict['greedy'].append(dicts['greedy'].get())
        result_dict['deterministic'].append(dicts['deterministic'].get())
        result_dict['random_leverage'].append(dicts['random_leverage'].get())
        result_dict['uniform_random'].append(dicts['uniform_random'].get())

    print(result_dict)

done = time.time()
elapsed = done - simul_info['time']
simul_info['time'] = elapsed
pk.dump({
    "results": [relative_sub_Erdos, relative_sub_Pref, reltive_sub_Random],
    'params_simul': simul_info
}, open(('results.p'), 'wb'))
