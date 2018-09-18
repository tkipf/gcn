from utils import preprocess_adj
from subsample import *
from Models.models import GCN, MLP


def get_model_and_support(model_string, adj, initial_train, train_mask, val_mask, test_mask, with_test):
    if model_string == 'gcn':
        support = [preprocess_adj(adj)]
        num_supports = 1
        model_func = GCN
        if with_test:
            sub_sampled_support = support
        else:  # cut the test and validation features
            initial_sample_list = get_list_from_mask(initial_train)
            sub_sampled_support = [
                get_sub_sampled_support(complete_support=support[0], node_to_keep=initial_sample_list)
            ]
            A = adj.toarray()[initial_sample_list]
            one_hop_sample_list = np.argwhere(np.sum(A, axis=0))
            sub_sampled_support_second = [
                get_sub_sampled_support(complete_support=support[0], node_to_keep=one_hop_sample_list)
            ]

            return model_func, sub_sampled_support_second, sub_sampled_support, num_supports

    elif model_string == 'gcn_cheby':
        support = chebyshev_polynomials(adj, FLAGS.max_degree)
        sub_sampled_support = support
        num_supports = 1 + FLAGS.max_degree
        model_func = GCN
    elif model_string == 'dense':
        support = [preprocess_adj(adj)]  # Not used
        sub_sampled_support = support  # Not used
        num_supports = 1
        model_func = MLP
    elif model_string == 'k-nn':
        support = [preprocess_adj(adj)]  # Not used
        sub_sampled_support = support  # Not used
        num_supports = 1
        model_func = None
    elif model_string == 'gcn_subsampled':  # FLOFLO's making
        support = [preprocess_adj(adj)]
        num_supports = 1
        model_func = GCN
        if with_test:
            initial_sample_list = get_list_from_mask(train_mask + val_mask + test_mask)
        else:
            initial_sample_list = get_list_from_mask(train_mask)
        sub_sampled_support = [get_sub_sampled_support(complete_support=support[0], node_to_keep=initial_sample_list)]
        A = adj.toarray()[initial_sample_list]
        one_hop_sample_list = np.argwhere(np.sum(A, axis=0))
        sub_sampled_support_second = [
            get_sub_sampled_support(complete_support=support[0], node_to_keep=one_hop_sample_list)
        ]

        return model_func, sub_sampled_support_second, sub_sampled_support, num_supports

    else:
        raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

    return model_func, support, sub_sampled_support, num_supports
