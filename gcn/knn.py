from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from subsample import get_list_from_mask
from random import randint


def knn_eucledian(features,
                  y_train,
                  y_val,
                  y_test,
                  train_mask,
                  val_mask,
                  test_mask,
                  VERBOSE_TRAINING,
                  seed,
                  n_neighbors=5):
    indices = features[0]
    values = features[1]
    shape = features[2]

    features_nparray = np.zeros(shape)
    for i in range(indices.shape[0]):
        pair_index = indices[i]
        features_nparray[pair_index[0], pair_index[1]] = values[i]

    X = features_nparray[np.argwhere(train_mask + val_mask + test_mask).flatten(), :]
    y = (y_train + y_val + y_test)[np.argwhere(train_mask + val_mask + test_mask).flatten(), :]
    neigh = KNeighborsClassifier(n_neighbors, n_jobs=-1)
    neigh.fit(X, y)
    X_test = features_nparray[np.argwhere(test_mask).flatten(), :]
    y_test_only = y_test[np.argwhere(test_mask).flatten(), :]

    predicted_labels = neigh.predict(features_nparray)

    labels_equal = (np.equal(np.argmax(predicted_labels, axis=1), np.argmax(y_test, axis=1)))
    list_node_correctly_classified = np.argwhere(labels_equal).reshape(-1)
    list_node_correctly_classified_test = list(filter(lambda x: test_mask[x], list(list_node_correctly_classified)))
    # Not using the score unction from scikit learn because of misleading metrics https://github.com/scikit-learn/scikit-learn/issues/7332
    test_acc = len(list_node_correctly_classified_test) / X_test.shape[0]

    return test_acc, list_node_correctly_classified_test


# Different vesrion of K-nn. Instead of using the features to determine proximity, we use the connectivity in the graph.
# A node is in the neighborhood if it's within 2 hop. The label is then averaged over all the neighboors.
# Only he labels of the training set can be used
def knn_hop(y_train, y_val, y_test, train_mask, val_mask, test_mask, list_adj):
    predicted_labels = np.zeros(y_test.shape)
    for test_index in get_list_from_mask(test_mask):
        concated_adj = list_adj[1][test_index] + list_adj[2][test_index] # HARDCODED 2 hops
        neighboors_index = np.argwhere(concated_adj).flatten()
        neighboors_index_train = list(filter(lambda x: train_mask[x], list(neighboors_index)))
        if len(neighboors_index_train) > 0:
            neighboors_labels_train = y_train[neighboors_index_train, :]
            max_label = np.argmax(np.sum(neighboors_labels_train, axis=0))
        else:
            max_label = (randint(0, y_train.shape[1])) - 1  # random label
        predicted_labels[test_index, max_label] = 1

    labels_equal = (np.equal(np.argmax(predicted_labels, axis=1), np.argmax(y_test, axis=1)))
    list_node_correctly_classified = np.argwhere(labels_equal).reshape(-1)
    list_node_correctly_classified_test = list(filter(lambda x: test_mask[x], list(list_node_correctly_classified)))
    # Not using the score unction from scikit learn because of misleading metrics https://github.com/scikit-learn/scikit-learn/issues/7332
    print(np.sum(test_mask))
    test_acc = len(list_node_correctly_classified_test) / np.sum(test_mask)
    return test_acc, list_node_correctly_classified_test
