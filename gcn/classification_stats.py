import numpy as np
from graph_processing import get_num_paths_to_known

MAX_NUM_PATHS_DISPLAY = 3


def get_classification_stats(list_node_correctly_classified: list, list_test_node: list, paths_to_known_list):

    paths_to_known = only_keep_closest_hop(paths_to_known_list)
    correct_paths_to_known = paths_to_known[list_node_correctly_classified, :]
    incorrect_paths_to_known = paths_to_known[[x for x in list_test_node if x not in list_node_correctly_classified], :]
    return correct_paths_to_known, incorrect_paths_to_known


def print_classification_stats(correct_paths_to_known, incorrect_paths_to_known):
    print()
    print("Corretly classified")
    print_paths_to_known_stats(correct_paths_to_known)
    print()
    print("Incorretly classified")
    print_paths_to_known_stats(incorrect_paths_to_known)


def print_paths_to_known_stats(paths_to_known):
    num_node_remaining = paths_to_known.shape[0]
    max_paths = int(np.amax(paths_to_known))
    if MAX_NUM_PATHS_DISPLAY < max_paths:  # If there is too much connection
        max_display = MAX_NUM_PATHS_DISPLAY
        title = '\t' + '\t'.join(str(x) for x in range(0, max_display))
        title += '\t' + str(max_display) + '>'
    else:
        max_display = max_paths
        title = '\t\t' + '\t'.join(str(x) for x in range(0, max_display + 1))
   
    print(title)
    for num_hop in range(paths_to_known.shape[1]):
        unique, counts = np.unique(
            paths_to_known[:, num_hop],
            return_counts=True)  # group nodes by number of paths to known node of lenght (num_hop)
        dict_number_node_grouped = dict(zip(unique, counts))
        # aggregate number above max display to the same MAX_NUM_PATHS_DISPLAY bin
        if MAX_NUM_PATHS_DISPLAY < max_paths:
            list_keys_to_concat_in_one_entry = [
                num_paths for num_paths in dict_number_node_grouped.keys() if (num_paths > MAX_NUM_PATHS_DISPLAY)
            ]
            number_node_above_display_threshold = sum(
                [dict_number_node_grouped[key_concat] for key_concat in list_keys_to_concat_in_one_entry])
            if MAX_NUM_PATHS_DISPLAY in dict_number_node_grouped:
                dict_number_node_grouped[MAX_NUM_PATHS_DISPLAY] += number_node_above_display_threshold
            else:
                dict_number_node_grouped[MAX_NUM_PATHS_DISPLAY] = number_node_above_display_threshold
            [dict_number_node_grouped.pop(k, None) for k in list_keys_to_concat_in_one_entry]  # remove

        string_hop = '\t'.join("{0:.2f}".format(dict_number_node_grouped[x] / num_node_remaining) if (
            x in dict_number_node_grouped) else " 0" for x in range(0, max_display + 1))
        num_node_remaining = dict_number_node_grouped[0]
        print(str(num_hop + 1) + " hop \t " + string_hop)


#remove the stats for a node after closest hop,
# 0,0, x, (every thing set to zero after that to not be accounted for)
def only_keep_closest_hop(paths_to_known):
    closest_paths_to_known = np.full(paths_to_known.shape, -1)
    remaining_nodes = list(range(paths_to_known.shape[0]))
    for i in range(paths_to_known.shape[1]):
        closest_paths_to_known[remaining_nodes, i] = paths_to_known[remaining_nodes, i]
        node_to_remove = np.argwhere(paths_to_known[remaining_nodes, i])
        remaining_nodes = [remaining_nodes[i] for i in range(len(remaining_nodes)) if i not in node_to_remove]
    return closest_paths_to_known
