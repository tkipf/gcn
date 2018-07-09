import time
import tensorflow as tf
import numpy as np
from utils import construct_feed_dict

flags = tf.app.flags
FLAGS = flags.FLAGS


def train_model(model_func,
                num_supports,
                support,
                features,
                y_train,
                y_val,
                y_test,
                train_mask,
                val_mask,
                test_mask,
                sub_sampled_support=None,
                VERBOSE_TRAINING=False,
                seed=123):
    tf.set_random_seed(seed)
    # Define placeholders
    placeholders = {
        'sub_sampled_support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero':
            tf.placeholder(tf.int32)  # helper variable for sparse dropout
    }

    # Create model
    model = model_func(placeholders, input_dim=features[2][1], logging=True)

    # Initialize session
    sess = tf.Session()

    # Define model evaluation function
    def evaluate(features, support, labels, mask, sub_sampled_support, placeholders):
        t_test = time.time()
        feed_dict_val = construct_feed_dict(features, support, labels, mask, sub_sampled_support, placeholders)
        outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
        return outs_val[0], outs_val[1], (time.time() - t_test)

    # Init variables
    sess.run(tf.global_variables_initializer())

    cost_val = []

    # Train model
    for epoch in range(FLAGS.epochs):

        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(features, support, y_train, train_mask, sub_sampled_support, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        # Training step
        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
        # Validation
        cost, acc, duration = evaluate(features, support, y_val, val_mask, sub_sampled_support, placeholders)
        cost_val.append(cost)
        if VERBOSE_TRAINING:
            # Print results
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]), "train_acc=",
                  "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost), "val_acc=", "{:.5f}".format(acc),
                  "time=", "{:.5f}".format(time.time() - t))

        if FLAGS.early_stopping is not None and epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(
                cost_val[-(FLAGS.early_stopping + 1):-1]):
            print("Early stopping...")
            break

    print("Optimization Finished!")

    # Testing
    test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, sub_sampled_support,
                                                  placeholders)
    print("Test set results:", "cost=", "{:.5f}".format(test_cost), "accuracy=", "{:.5f}".format(test_acc), "time=",
          "{:.5f}".format(test_duration))
    tf.reset_default_graph()
    return test_acc
