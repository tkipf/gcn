import argparse
import numpy as np
from graph_generator import generate_Erdos_Renyi_graph, generate_pref_attachment_graph, generate_random_graph, get_sparse_eigen_decomposition, plot_graph
import time
import pickle as pk
import multiprocessing as mp
import sys
from reproduce_greedy import simulate
np.set_printoptions(precision=2)
print(">>>>>>> Starting greedy Smapling")

# extract parsed information
parser = argparse.ArgumentParser(description='Simulations parameters')
parser.add_argument('-n', '--NOISE_CONSTANT', type=int, help='Noise on signal')
parser.add_argument('-N', '--NUM_NODES', type=int, default=20, help='Size of graph generated')
parser.add_argument('-S', '--NUM_SIMULATIONS', type=int, default=10, help='Number of simulations')
parser.add_argument('-K', '--K_sparse', type=int, default=5, help='K sparsity')
parser.add_argument('-l', '--number_node_sampled', type=int, default=5, help=' Subset size')
parser.add_argument('-C', '--CORES', type=int, default=4, help=' Num of process in parrallel')
parser.add_argument('-mp', '--want_multiprocessing', action="store_true", help='multiprocessing mode')
parser.add_argument('--debug', action="store_true", help='debug mode')

print(">>>>>>> Reading parameters")

args = parser.parse_args()
config = vars(args)
print(config)
config['time'] = time.time()  # will be updated at the end
# RESULT STORING ARRAYS
relative_sub_Erdos = {'greedy': [], 'deterministic': [], 'random_leverage': [], 'uniform_random': []}
relative_sub_Pref = {'greedy': [], 'deterministic': [], 'random_leverage': [], 'uniform_random': []}
reltive_sub_Random = {'greedy': [], 'deterministic': [], 'random_leverage': [], 'uniform_random': []}

for graph_gen, result_dict in [(generate_random_graph, reltive_sub_Random),
                               (generate_Erdos_Renyi_graph, relative_sub_Erdos), (generate_pref_attachment_graph,
                                                                                  relative_sub_Pref)]:
    if config['want_multiprocessing']:
        num_iter = int(config['NUM_SIMULATIONS'] / config['CORES'])
        pool = mp.Pool(processes=config['CORES'])
        pool_results = [pool.apply_async(simulate, (graph_gen, num_iter, config)) for indices in range(CORES)]
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
        dicts = simulate(graph_gen, config['NUM_SIMULATIONS'], config)
        print(dicts)
        result_dict['greedy'].append(dicts['greedy'])
        result_dict['deterministic'].append(dicts['deterministic'])
        result_dict['random_leverage'].append(dicts['random_leverage'])
        result_dict['uniform_random'].append(dicts['uniform_random'])

done = time.time()
elapsed = done - config['time']
config['time'] = elapsed
print("elapsed =" + str(elapsed))
# Store the results of the simul in a pickle file.
filename = 'results_' + str(config['NOISE_CONSTANT']) + '.p'
pk.dump({
    "results": [relative_sub_Erdos, relative_sub_Pref, reltive_sub_Random],
    'params_simul': config
}, open((filename), 'wb'))
