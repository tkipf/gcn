from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
from subsample import *
from utils import *
from models import GCN, MLP

# Set random seed
seed = 13
np.random.seed(seed)
tf.set_random_seed(seed)
VERBOSE_TRAINING = False
WITH_TEST = False
# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense', 'gcn_subsampled'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 300, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 30, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
# Load data
adj, features, y_train, y_val, y_test, initial_train_mask, val_mask, test_mask = load_data(FLAGS.dataset)
# Some preprocessing
features = preprocess_features(features)
labels_percent_list = [5,10,20,30,50,75,100]
result = []
for model_gcn in ['gcn_subsampled','gcn']:
    for label_percent in labels_percent_list:
        train_mask = get_train_mask(label_percent,y_train)
        FLAGS.model = model_gcn
        known_percent = show_data_stats(adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, False)
        num_inputs = features[2][0]
        if FLAGS.model == 'gcn':
            support = [preprocess_adj(adj)]
            if WITH_TEST:
                sub_sampled_support = support
            else: # cut the test and validation features
                initial_sample_list = get_list_from_mask(initial_train_mask)
                sub_sampled_support = [get_sub_sampled_support_fast(complete_support = support[0], node_to_keep = initial_sample_list)]
            num_supports = 1
            model_func = GCN
        elif FLAGS.model == 'gcn_cheby':
            support = chebyshev_polynomials(adj, FLAGS.max_degree)
            sub_sampled_support = support
            num_supports = 1 + FLAGS.max_degree
            model_func = GCN
        elif FLAGS.model == 'dense':
            support = [preprocess_adj(adj)]  # Not used
            sub_sampled_support = support # Not used
            num_supports = 1
            model_func = MLP
        elif FLAGS.model == 'gcn_subsampled':
            support = [preprocess_adj(adj)]
            num_supports = 1
            model_func = GCN
        #  initial_sample_list = get_random_percent(num_nodes = num_inputs, percent = 50) # obtain index of initial nodes in the sample set
            if WITH_TEST:
                initial_sample_list = get_list_from_mask(train_mask + val_mask+test_mask)
            else:
                initial_sample_list = get_list_from_mask(train_mask)
                sub_sampled_support = [get_sub_sampled_support_fast(complete_support = support[0], node_to_keep = initial_sample_list)]
            note_doesnt_exist = [x for x in range(features[2][0]) if x not in initial_sample_list]
            train_mask[note_doesnt_exist] = False
        else:
            raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

        # Define placeholders
        placeholders = {
            'sub_sampled_support' : [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
            'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
            'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
            'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
            'labels_mask': tf.placeholder(tf.int32),
            'dropout': tf.placeholder_with_default(0., shape=()),
            'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
        }

        # Create model
        model = model_func(placeholders, input_dim=features[2][1], logging=True)

        # Initialize session
        sess = tf.Session()


        # Define model evaluation function
        def evaluate(features, support, labels, mask, sub_sampled_support,placeholders):
            t_test = time.time()
            feed_dict_val = construct_feed_dict(features, support, labels, mask, sub_sampled_support,placeholders)
            outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
            return outs_val[0], outs_val[1], (time.time() - t_test)


        # Init variables
        sess.run(tf.global_variables_initializer())

        cost_val = []

        # Train model
        for epoch in range(FLAGS.epochs):

            t = time.time()
            # Construct feed dictionary
            feed_dict = construct_feed_dict(features, support, y_train, train_mask, sub_sampled_support,placeholders)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})
            # Training step
            outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
            # Validation
            cost, acc, duration = evaluate(features, support, y_val, val_mask, sub_sampled_support,placeholders)
            cost_val.append(cost)
            if VERBOSE_TRAINING:
                # Print results
                print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
                    "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
                    "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

            if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
                print("Early stopping...")
                break

        print("Optimization Finished!")

        # Testing
        test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, sub_sampled_support,placeholders)
        print("Test set results:", "cost=", "{:.5f}".format(test_cost),
            "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
        
        tf.reset_default_graph()

        result.append((model_gcn,known_percent,test_acc))

if WITH_TEST:
    print(result)
    import pickle as pk
    pk.dump(result,open('results.p','wb'))
else:
    print(result)
    import pickle as pk
    pk.dump(result,open('results_without_test.p','wb'))
    
