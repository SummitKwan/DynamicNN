"""
use CNN to classify cifar10
this script is modified from tutorials of CNN on cifar10 dataset at:
https://www.tensorflow.org/tutorials/deep_cnn
with code repository at:
https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10
"""

# import
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile
import numpy as np

from six.moves import urllib
import tensorflow as tf

import cifar10_input

# get current directory
cur_dir = os.path.dirname( __file__) if '__file__' in locals() else '.'

batch_size = 128
data_dir = './data/cifar10_data'
use_fp16 = False

""" get dimension of data """
# get dimension of data
IMAGE_SIZE = 24

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


def test_scope():
    print(data_dir)

test_scope()


""" functions to read data """

def distorted_inputs():
    """Construct distorted input for CIFAR training using the Reader ops.

    Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

    Raises:
    ValueError: If no data_dir
    """
    data_dir_full = os.path.join(data_dir, 'cifar-10-batches-bin')
    images, labels = cifar10_input.distorted_inputs(data_dir=data_dir_full,
                                                  batch_size=batch_size)
    if use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
    return images, labels


def inputs(eval_data):
    """Construct input for CIFAR evaluation using the Reader ops.

    Args:
    eval_data: bool, indicating if one should use the train or eval data set.

    Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

    Raises:
    ValueError: If no data_dir
    """
    data_dir_full = os.path.join(data_dir, 'cifar-10-batches-bin')
    images, labels = cifar10_input.inputs(eval_data=eval_data,
                                        data_dir=data_dir_full,
                                        batch_size=batch_size)
    if use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
    return images, labels

