from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
from subsample import *
from utils import *
from models import GCN, MLP
from train import train_model
from output_stats import *
from build_support import get_model_and_support
import scipy.sparse as sp

VERBOSE_TRAINING = False
# Set random seed
seed = 13
np.random.seed(seed)
# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_integer('epochs', 300, 'Number of epochs to train.')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_integer('early_stopping', 30, 'Tolerance for early stopping (# of epochs).')
# Load data
adj, _, y_train, y_val, y_test, initial_train_mask, val_mask, test_mask = load_data(FLAGS.dataset)
#BUILD FEATURE AS IDENTITY MATRIX
print(type(_))
num_node = _.shape[0]

features = sp.csr_matrix(np.identity(num_node))
print(type(features))
# Some preprocessing
features = preprocess_features(features)# no need to preprocess
hyper_params = {'epochs': FLAGS.epochs, 'dropout': FLAGS.dropout}
labels_percent_list = [0.1,2,5,7,10,20,30,40,50,60,75,100]
result = []
for MAINTAIN_LABEL_BALANCE in [False,True]:
    for WITH_TEST in [False]:
        for model_gcn in ['gcn_subsampled', 'gcn']:
            for label_percent in labels_percent_list:
                train_mask = get_train_mask(label_percent, y_train, initial_train_mask, MAINTAIN_LABEL_BALANCE)
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

                test_acc = train_model(model_func, hyper_params, num_supports, support, features, y_train, y_val, y_test,
                                    train_mask, val_mask, test_mask, sub_sampled_support, VERBOSE_TRAINING)
                result.append((model_gcn, known_percent, test_acc))

        if WITH_TEST and MAINTAIN_LABEL_BALANCE:
            print(result)
            import pickle as pk
            pk.dump(result, open('I_results.p', 'wb'))
        elif MAINTAIN_LABEL_BALANCE:
            print(result)
            import pickle as pk
            pk.dump(result, open('I_results_without_test.p', 'wb'))
        elif WITH_TEST:
            print(result)
            import pickle as pk
            pk.dump(result, open('I_results_random.p', 'wb'))
        else:
            print(result)
            import pickle as pk
            pk.dump(result, open('I_results_without_test_random.p', 'wb'))

        result.clear()
