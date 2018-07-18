import numpy as np
import os
import copy
import tensorflow as tf

current_dir = os.getcwd()
project_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))


def graph_settings():
    settings = {}

    settings['presets'] = ['default', 'Kipf', 'quick']

    for p in settings['presets']:
        settings[p] = {}
        settings[p]['params'] = {}

    ###########################################################
    # DEFAULT PARAMETERS                              #
    ###########################################################

    settings['default']['params']['dataset'] = 'cora'  # 'cora', 'citeseer', 'pubmed'
    settings['default']['params']['epochs'] = 300
    settings['default']['params']['learning_rate'] = 0.01
    settings['default']['params']['hidden1'] = 16
    settings['default']['params']['weight_decay'] = 5e-4
    settings['default']['params']['dropout'] = 0.5
    settings['default']['params']['early_stopping'] = 50
    settings['default']['seed'] = 13

    ##########################################################
    # PARAMETERS TO REPRODUCE KIPF RESULTS ON CORA           #
    ##########################################################
    settings['Kipf'] = copy.deepcopy(settings['default'])
    settings['Kipf']['params']['dataset'] = 'cora'
    settings['Kipf']['params']['epochs'] = 300
    settings['Kipf']['params']['learning_rate'] = 0.01
    settings['Kipf']['params']['hidden1'] = 16
    settings['Kipf']['params']['weight_decay'] = 5e-4
    settings['Kipf']['params']['dropout'] = 0.5
    settings['Kipf']['params']['early_stopping'] = 30
    settings['Kipf']['classifier'] = 'dense'
    settings['Kipf']['with_test'] = True
    settings['Kipf']['maintain_label_balance'] = True
    settings['Kipf']['max_label_percent'] = 27

    ##########################################################
    # PARAMETERS FOR A QUICK RUN             #
    ##########################################################
    settings['quick'] = copy.deepcopy(settings['default'])
    settings['quick']['params']['epochs'] = 3
    settings['quick']['params']['dropout'] = 0

    return settings


def set_tf_flags(params, flags):

    flags.DEFINE_string('dataset', params['dataset'], 'Dataset string.')
    flags.DEFINE_integer('epochs', params['epochs'], 'Number of epochs to train.')
    flags.DEFINE_float('learning_rate', params['learning_rate'], 'Initial learning rate.')
    flags.DEFINE_integer('hidden1', params['hidden1'], 'Number of units in hidden layer 1.')
    flags.DEFINE_float('weight_decay', params['weight_decay'], 'Weight for L2 loss on embedding matrix.')
    flags.DEFINE_float('dropout', params['dropout'], 'Dropout rate (1 - keep probability).')
    flags.DEFINE_integer('early_stopping', params['early_stopping'], 'Tolerance for early stopping (# of epochs).')
