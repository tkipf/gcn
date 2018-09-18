from __future__ import division
from __future__ import print_function

import time
from datetime import datetime
import tensorflow as tf
import pickle as pk
import os
from subsample import *
from utils import *
from train import train_model
from output_stats import *
from build_support import get_model_and_support
from settings import set_tf_flags, graph_settings
from classification_stats import get_classification_stats
from graph_processing import *
"""
Class used to plot graph for multiple labels % . 
"""


def train_and_save_results(adj,
                           features,
                           y_train,
                           y_val,
                           y_test,
                           initial_train_mask,
                           val_mask,
                           test_mask,
                           maintain_label_balance_list,
                           with_test_features_list,
                           models_list,
                           labels_percent_list,
                           SHOW_TEST_VAL_DATASET_STATS,
                           VERBOSE_TRAINING,
                           settings={},
                           fileinfo="",
                           stats_adj_helper=None):
    result = []
    if not os.path.exists('results'):
        os.makedirs('results')
    for MAINTAIN_LABEL_BALANCE in maintain_label_balance_list:
        for WITH_TEST in with_test_features_list:
            for model_gcn in models_list:
                for label_percent in labels_percent_list:
                    train_mask = get_train_mask(label_percent, y_train, initial_train_mask, MAINTAIN_LABEL_BALANCE)
                    paths_to_known_list = get_num_paths_to_known(get_list_from_mask(train_mask), stats_adj_helper)
                    print_partition_index(initial_train_mask, "Train", y_train)
                    print_partition_index(val_mask, "Val", y_val)
                    print_partition_index(test_mask, "Test", y_test)
                    known_percent = show_data_stats(
                        adj,
                        features,
                        y_train,
                        y_val,
                        y_test,
                        train_mask,
                        val_mask,
                        test_mask,
                    )
                    model_func, support, sub_sampled_support, num_supports = get_model_and_support(
                        model_gcn, adj, initial_train_mask, train_mask, val_mask, test_mask, WITH_TEST)

                    test_acc, list_node_correctly_classified = train_model(
                        model_func,
                        num_supports,
                        support,
                        features,
                        y_train,
                        y_val,
                        y_test,
                        train_mask,
                        val_mask,
                        test_mask,
                        sub_sampled_support,
                        VERBOSE_TRAINING,
                        seed=settings['seed'],
                        list_adj=stats_adj_helper)
                    print(test_acc)
                    result.append((model_gcn, known_percent, test_acc))
                    correct_paths_to_known, incorrect_paths_to_known = get_classification_stats(
                        list_node_correctly_classified, get_list_from_mask(test_mask), paths_to_known_list)
            info = {
                'MAINTAIN_LABEL_BALANCE': MAINTAIN_LABEL_BALANCE,
                'WITH_TEST': WITH_TEST,
                'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'params': settings
            }

            dict_output = {
                'results': result,
                'info': info,
                'stats': {
                    'correct_paths_to_known': correct_paths_to_known,
                    'incorrect_paths_to_known': incorrect_paths_to_known
                }
            }
            pk.dump(dict_output,
                    open(
                        os.path.join('results', fileinfo + 'w_test_features=' + str(WITH_TEST) + '_label_balance=' +
                                     str(MAINTAIN_LABEL_BALANCE) + '_results.p'), 'wb'))
            result.clear()


if __name__ == "__main__":
    # Settings
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    settings = graph_settings()['default']
    set_tf_flags(settings['params'], flags)
    # Verbose settings
    SHOW_TEST_VAL_DATASET_STATS = False
    VERBOSE_TRAINING = False

    # Random seed
    seed = settings['seed']
    np.random.seed(seed)

    # Load data
    adj, features, y_train, y_val, y_test, initial_train_mask, val_mask, test_mask = load_data(FLAGS.dataset)
    # Some preprocessing
    features = preprocess_features(features)

    labels_percent_list = [0.1, 5, 10, 20, 40, 50, 75, 100]
    list_adj = get_adj_powers(adj.toarray())
    maintain_label_balance_list = [False, True]
    with_test_features_list = [False, True]
    models_list = ['gcn_subsampled', 'gcn','dense','k-nn']
    
    # RUN
    train_and_save_results(
        adj,
        features,
        y_train,
        y_val,
        y_test,
        initial_train_mask,
        val_mask,
        test_mask,
        maintain_label_balance_list,
        with_test_features_list,
        models_list,
        labels_percent_list,
        SHOW_TEST_VAL_DATASET_STATS,
        VERBOSE_TRAINING,
        settings=settings,
        stats_adj_helper=list_adj)
