import tensorflow as tf
import time
import numpy as np
import logging
from memory_profiler import LogFile
import sys
import os

# ------- CONSTANTS ------- #

# ------ DIRECTORIES ------ #

LOG_DIR = os.path.join('logs', time.strftime("%Y-%m-%dT%H%M%S"))
FIG_DIR = 'figures'
DATA_DIR = 'data'
MODEL_SAVE_DIR = 'models'
PREV_PLOTS_DIR = os.path.join(FIG_DIR, 'restore')
tf.gfile.MakeDirs(PREV_PLOTS_DIR)

# ------ SAVED GRAPHS ------ #

OPT_GRAPH = 'optimized-graph.pb'
FROZEN_GRAPH = 'frozen-graph.pb'

# ------- SETTINGS ------- #

dt = 1
N = 100
TOL = 1e-5
NUM_SAMPLES = 1
BATCH_SIZE = 64
HESSIAN_BATCH_SIZE = 256
EVAL_EVERY_N = 10
MAX_BATCHES_TO_SAMPLE_ACC_COST = 450
MAX_BATCHES_TO_SAMPLE_HESS = 50
ALLOW_FULL_SAMPLING = False  # overrides the previous settings
# scale down the number of parameters from the desired number
p_s = [0.2, 0.2, 0.083]
DTYPE = tf.float32
NP_DTYPE = np.float16
INT_DTYPE = tf.int32
NP_INT_DTYPE = np.int32

GPU_OPTIONS = tf.GPUOptions(
    allow_growth=False, per_process_gpu_memory_fraction=0.95)
SESSION_CONFIG = tf.ConfigProto(
    gpu_options=GPU_OPTIONS,
    allow_soft_placement=False,
    log_device_placement=False)

# ------- MEMORY LOGGING ------- #

mem_logger_ = logging.getLogger('Memory_Profile')
mem_logger_.setLevel(logging.DEBUG)

mem_fh = logging.FileHandler('memory_profile.log')
mem_fh.setLevel(logging.DEBUG)
mem_fh.setFormatter(logging.Formatter('%(asctime)s: %(message)s'))
mem_logger_.addHandler(mem_fh)
mem_logger = LogFile('Memory_Profile', reportIncrementFlag=True)

DEFAULT_PARAMS = {
    'M':
    1200,
    'Layers':
    5,
    'LayerTypes': [
        tf.layers.conv2d, tf.layers.max_pooling2d, tf.layers.conv2d,
        tf.layers.max_pooling2d, tf.layers.dense
    ],
    'K': [
        np.floor(32 * p_s[0]), 2,
        np.floor(64 * p_s[1]), 2,
        np.floor(1024 * p_s[2])
    ],
    'ActFun':
    tf.nn.relu,
    'Optimizer':
    tf.train.AdamOptimizer
}

PARAMS_TO_EVALUATE = {
    'LeNet_0_attack': {
        'noise_mix': 0.0
    },
    'LeNet_10_attack': {
        'noise_mix': 0.1
    },
    'LeNet_20_attack': {
        'noise_mix': 0.2
    },
    'LeNet_25_attack': {
        'noise_mix': 0.25
    },
    'LeNet_30_attack': {
        'noise_mix': 0.3
    }
}
