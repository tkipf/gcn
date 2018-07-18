from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
from utils import *
from subsample import *
from train import train_model
from output_stats import *
from build_support import get_model_and_support
from settings import *
from classification_stats import get_classification_stats, print_classification_stats
from graph_processing import get_adj_powers, get_num_paths_to_known
# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
settings = graph_settings()['default']
set_tf_flags(settings['params'], flags)

WITH_TEST = False
MAINTAIN_LABEL_BALANCE = False
MAX_LABEL_PERCENT = 20
#75

# Verbose settings
SHOW_TEST_VAL_DATASET_STATS = False
VERBOSE_TRAINING = False

# Random seed
seed = settings['seed']
np.random.seed(seed)

# Load data
adj, features, y_train, y_val, y_test, initial_train_mask, val_mask, test_mask = load_data(FLAGS.dataset)
train_mask = get_train_mask(MAX_LABEL_PERCENT, y_train, initial_train_mask, MAINTAIN_LABEL_BALANCE)
list_adj = get_adj_powers(adj.toarray())
paths_to_known_list = get_num_paths_to_known(get_list_from_mask(train_mask), list_adj)
# Partitioning check, ensures that no mask overlaps and that there is a label for every input in the maskS.
print_partition_index(train_mask, "Train", y_train)
print_partition_index(val_mask, "Val", y_val)
print_partition_index(test_mask, "Test", y_test)

# Some preprocessing
features = preprocess_features(features)
known_percent = show_data_stats(adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask,
                                SHOW_TEST_VAL_DATASET_STATS)

model_func, support, sub_sampled_support, num_supports = get_model_and_support(
    'gcn_subsampled', adj, initial_train_mask, train_mask, val_mask, test_mask, WITH_TEST)
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
    seed=seed,
    return_classified_node=True)

correct_paths_to_known, incorrect_paths_to_known = get_classification_stats(list_node_correctly_classified,
                                                                            get_list_from_mask(test_mask),
                                                                            paths_to_known_list)
print_classification_stats(correct_paths_to_known, incorrect_paths_to_known)