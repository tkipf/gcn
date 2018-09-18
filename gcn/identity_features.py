import tensorflow as tf
from subsample import *
from utils import *
import scipy.sparse as sp
from plot_train import train_and_save_results
from settings import set_tf_flags, graph_settings
from graph_processing import *
# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
settings = graph_settings()['default']
set_tf_flags(settings['params'], flags)
# Verbose settings
SHOW_TEST_VAL_DATASET_STATS = False
VERBOSE_TRAINING = False

# Random seed
seed = settings['seed']
np.random.seed(seed)

# Load data
adj, _, y_train, y_val, y_test, initial_train_mask, val_mask, test_mask = load_data(FLAGS.dataset)

# Replaace features by Identity matrix
num_node = _.shape[0]
features = sp.csr_matrix(np.identity(num_node))
features = preprocess_features(features)  # no need to preprocess

labels_percent_list = [0.1, 2,5, 7,10,15, 20, 30,40, 50, 75, 100]
list_adj = get_adj_powers(adj.toarray())
maintain_label_balance_list = [False]
with_test_features_list = [False]
models_list = ['gcn_subsampled', 'gcn']

# RUN
train_and_save_results(
    adj,
    features,
    y_train,
    y_val,
    y_test,
    initial_train_mask,
    val_mask,
    test_mask,
    maintain_label_balance_list,
    with_test_features_list,
    models_list,
    labels_percent_list,
    SHOW_TEST_VAL_DATASET_STATS,
    VERBOSE_TRAINING,
    settings=settings,
    fileinfo="Features_Identity_",
    stats_adj_helper=list_adj)
