""" utitily functions """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import gzip
import six.moves.cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt

def load_data(dataset = 'mnist.pkl.gz'):
    ''' Loads the dataset, based on lisa-lab/DeepLearningTutorials
    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    print('... loading data')

    # Load the dataset
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input

    return train_set, valid_set, test_set



def data_unravel(X, r=None):
    """ reshape data to images: form N*M to N*r*c """
    dim_X = 1
    if len(X.shape) == 1:
        N = -1
        M = X.shape[0]
        dim_X =1
    elif len(X.shape) == 2:
        N, M = X.shape
        dim_X = 2
    else:
        raise Exception('input X must be of shape [M], or [N, M]')

    # determine row and columns (r, c)
    if r is None:
        r = int(np.sqrt(M))
    c = M//r

    if r*c != M:
        raise Exception('input dimension not right, please specify r (num of rows)')

    if dim_X == 1:
        X_img = np.reshape(X, [r, c])
    elif dim_X == 2:
        X_img = np.reshape(X, [N, r, c])
    else:
        X_img = None

    return X_img


def data_ravel(X_img):
    """ reshape data to images: form  N*r*c to N*M """
    dim_X = 2
    if len(X_img.shape) == 2:
        N = -1
        r, c = X_img.shape
        dim_X = 2
    elif len(X_img.shape) == 3:
        N, r, c = X_img.shape
        dim_X = 3
    else:
        raise Exception('input X must be of shape [r, c], or [N, r, c]')

    M = r*c

    if dim_X == 2:
        X = np.reshape(X_img, [M])
    elif dim_X == 3:
        X = np.reshape(X_img, [N, M])
    else:
        X = None

    return X



def data_plot(X, Y=None, i=None):
    """
    plot a sample of data

    :param X: images, raveled, [N, M]
    :param Y: lables, N
    :param i: index to plot, if None, randomly select one
    :return: imshow handle
    """

    N = len(X)
    dimX = len(X.shape)

    if i is None:
        i = np.random.randint(0, N, 1)[0]

    if dimX == 2:
        im = data_unravel(X[i])
    elif dimX == 3:
        im = X[i]
    elif dimX == 4:
        if X.shape[2] in (3, 4):
            im = X[i]
        else:
            im = X[i, :, :, 0]
    else:
        raise Exception('input X dimension not allowed')
    h = plt.imshow(im, cmap='gray')
    if Y is not None:
        plt.text(0.5, 1, Y[i], ha='center', va='bottom', fontsize='small', transform=plt.gca().transAxes)
    plt.axis('off')
    return h

def auto_row_col(N, nrows = None):
    """ automatically determine num rows and num_columns """
    if nrows is None:
        nrows = int(np.floor(np.sqrt(N)))
    ncols, rem = divmod(N, nrows)
    ncols = ncols + (rem>0)
    return nrows, ncols