from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
from utils import *
from subsample import *
from train import train_model
from output_stats import *
from build_support import get_model_and_support
# Set random seed
seed = 13
WITH_TEST = False
MAINTAIN_LABEL_BALANCE = True
SHOW_TEST_VAL_DATASET_STATS = True
VERBOSE_TRAINING = True
np.random.seed(seed)
tf.set_random_seed(seed)
MAX_LABEL_PERCENT = 100  # min % to end up with 20 labels per class, as described in the paper
# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense', 'subsampled'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 4, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

# Load data
adj, features, y_train, y_val, y_test, initial_train_mask, val_mask, test_mask = load_data(FLAGS.dataset)
train_mask = get_train_mask(MAX_LABEL_PERCENT, y_train, initial_train_mask, MAINTAIN_LABEL_BALANCE)

print_partition_index(train_mask, "Train", y_train)
print_partition_index(val_mask, "Val", y_val)
print_partition_index(test_mask, "Test", y_test)

# Some preprocessing
features = preprocess_features(features)
known_percent = show_data_stats(adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask,
                                SHOW_TEST_VAL_DATASET_STATS)

hyper_params = {'epochs': FLAGS.epochs, 'dropout': FLAGS.dropout}
model_func, support, sub_sampled_support, num_supports = get_model_and_support(FLAGS.model, adj, initial_train_mask,
                                                                           train_mask, val_mask, test_mask, WITH_TEST)
train_model(model_func, hyper_params, num_supports, support, features, y_train, y_val, y_test, train_mask, val_mask,
            test_mask, sub_sampled_support, VERBOSE_TRAINING)
