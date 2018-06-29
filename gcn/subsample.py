import random
import time
import numpy as np
from itertools import compress
random.seed(123)


#remove a column from adj matrix.
#TODO needs additional scaling?
def get_sub_sampled_support_fast(complete_support, node_to_keep):
    start = time.time()
    index_array = complete_support[0]
    values = np.zeros(complete_support[1].shape)
    index_array_sorted = index_array[:, 1].argsort()
    j = 0
    node_to_keep.sort()
    for index_to_keep in node_to_keep:
        while (j < len(index_array_sorted)
               and index_to_keep >= index_array[index_array_sorted[j]][1]):
            if (index_to_keep == index_array[index_array_sorted[j]][1]):
                values[index_array_sorted[j]] = complete_support[1][
                    index_array_sorted[j]]
            j += 1
    sub_sampled_support = (index_array, values, complete_support[2])
    end = time.time()
    #print("Time faster :" + str(end - start))
    return sub_sampled_support


# Keep smallest number of labels per class in training set
def get_train_mask(label_percent, y_train):
    train_mask = np.zeros((y_train.shape[0], ), dtype=bool)
    ones_index = []
    for i in range(y_train.shape[1]):
        ones_index.append(np.argwhere(y_train[:, i] > 0).reshape(-1))
    if label_percent < 100:
        smaller_num = min(
            int(len(l) * (label_percent / 100)) for l in ones_index)
        print(smaller_num)
        for ones in ones_index:
            train_mask[ones[0:smaller_num]] = True
    else:
        for ones in ones_index:
            train_mask[ones[0:int(len(ones) * (label_percent / 100))]] = True
    return train_mask


def get_sub_sampled_support(complete_support, node_to_keep):
    start = time.time()
    index_array = complete_support[0]
    values = np.zeros(complete_support[1].shape)
    for i in range(index_array.shape[0]):
        if index_array[i][1] in node_to_keep:
            values[i] = complete_support[1][i]
    sub_sampled_support = (index_array, values, complete_support[2])
    end = time.time()
    #print("Time slower :" + str(end - start))
    return sub_sampled_support


#returns a random list of indexes of the node to be kept at random.
def get_random_percent(num_nodes, percent):
    if percent > 100:
        print("This is not how percentage works.")
        exit()
    random_sampling_set_size = int((percent * num_nodes) / 100)
    return random.sample(range(num_nodes), random_sampling_set_size)


#returns a random list of indexes for the mask
def get_list_from_mask(mask):
    return list(compress(range(len(mask)), mask))


# Set features of node that shouldn't be in the set to crazy things to make sure they are not in the gcnn
def modify_features_that_shouldnt_change_anything(features, note_to_keep):
    note_doesnt_exist = [
        x for x in range(features[2][0]) if x not in note_to_keep
    ]
    a = np.where(np.isin(features[0][:, 0], note_doesnt_exist))
    features[1][a[0]] = 10000000


def show_data_stats(adj,
                    features,
                    y_train,
                    y_val,
                    y_test,
                    train_mask,
                    val_mask,
                    test_mask,
                    print_val_test_stats=False):
    print("Dataset size = " + str(features[2][0]))
    print("Number of features = " + str(features[2][1]))

    def label_stats(y, mask, complete_mask=[], incomplete=False):
        labels = y[mask, :]
        num_label = sum(list(mask))
        label_percent = num_label / mask.shape[0]
        print("Number of known labels = " + str(num_label))
        if incomplete:
            complete_labels = y[mask, :]
            num_complete_label = sum(list(complete_mask))
            known_percent = int((num_label / num_complete_label) * 100)
            print("\tNumber of total label = " + str(num_complete_label))
            print("\tKnown percentage = " +
                  str(known_percent) + "%")
            for i in range(labels.shape[1]):
                sum_class = sum(labels[:, i])
                print(
                    str(sum_class) + " of class" + str(i) + " ( " +
                    str(int((sum_class / num_label) * 100)) + "%)")
            return known_percent
        else:
            for i in range(labels.shape[1]):
                sum_class = sum(labels[:, i])
                print(
                    str(sum_class) + " of class" + str(i) + " ( " +
                    str(int((sum_class / num_label) * 100)) + "%)")

    print()
    print("TRAINING SET ---------------")
    known_percent = label_stats(y_train, train_mask,
                np.logical_not(np.logical_or(val_mask, test_mask)), True)
    print()
    if print_val_test_stats:
        print("VALIDATION SET ---------------")
        label_stats(y_val, val_mask)
        print()
        print("TESTING SET ---------------")
        label_stats(y_test, test_mask)
        print()
    return known_percent