import numpy as np


def show_data_stats(adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, print_val_test_stats=False):
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
            print("\tKnown percentage = " + str(known_percent) + "%")
            for i in range(labels.shape[1]):
                sum_class = sum(labels[:, i])
                try:
                    print(str(sum_class) + " of class" + str(i) + " ( " + str(int((sum_class / num_label) * 100)) + "%)")
                except:
                    print(str(sum_class))   
                    print(str(num_label))           
            return known_percent
        else:
            for i in range(labels.shape[1]):
                sum_class = sum(labels[:, i])
                print(str(sum_class) + " of class" + str(i) + " ( " + str(int((sum_class / num_label) * 100)) + "%)")

    print()
    print("TRAINING SET ---------------")
    known_percent = label_stats(y_train, train_mask, np.logical_not(np.logical_or(val_mask, test_mask)), True)
    print()
    if print_val_test_stats:
        print("VALIDATION SET ---------------")
        label_stats(y_val, val_mask)
        print()
        print("TESTING SET ---------------")
        label_stats(y_test, test_mask)
        print()
    return known_percent