import numpy as np
from graph_processing import get_num_paths_to_known


def get_classification_stats(adj: np.array, list_node_correctly_classified: list, list_test_node: list,
                             list_node_trained: list):
    paths_to_known_list = get_num_paths_to_known(adj, list_node_trained, MAX_ITER=3)
    print("accurcay " + str(len(list_node_correctly_classified) / len(list_test_node)))

    print("Corretly classified")
    correct_paths = paths_to_known_list[list_node_correctly_classified, :]
    max_paths = int(np.amax(correct_paths))
    max_display = 10
    title = '\t'.join(str(x) for x in range(0, max_display + 1))
    print('\t' + title)
    for hop in range(paths_to_known_list.shape[1]):
        te = correct_paths[:, hop]
        unique, counts = np.unique(te, return_counts=True)
        ra = dict(zip(unique, counts))
        string_hop = '\t'.join(
            str(ra[x]) if (x < 11 and x in ra) else " 0" if x < 11 else '' for x in range(0, max_paths + 1))
        print(str(hop + 1) + " hop : " + string_hop)

    print("Incorretly classified")
    incorrect_paths = paths_to_known_list[[x for x in list_test_node if x not in list_node_correctly_classified], :]
    max_paths = int(np.amax(incorrect_paths))
    max_display = 10
    title = '\t'.join(str(x) for x in range(0, max_display + 1))
    print('\t' + title)
    for hop in range(paths_to_known_list.shape[1]):
        te = incorrect_paths[:, hop]
        unique, counts = np.unique(te, return_counts=True)
        ra = dict(zip(unique, counts))
        string_hop = '\t'.join(
            str(ra[x]) if (x < 11 and x in ra) else " 0" if x < 11 else '' for x in range(0, max_paths + 1))
        print(str(hop + 1) + " hop : " + string_hop)
