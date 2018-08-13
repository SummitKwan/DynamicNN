""" utitily functions """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import gzip
import six.moves.cPickle as pickle
import numpy as np
import matplotlib as mpl
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


def data_plot(X, Y=None, i=None, n=None, yn_random=True):
    """
    plot a sample of data

    :param X: images, raveled, [N, M]
    :param Y: lables, N
    :param i: index to plot, if None, randomly select one, if a list, plot all of them
    :param n: defalt to None, if not None, overwirtes i, plot N randomly selected examples
    :return: imshow handle
    """

    N = len(X)
    dimX = len(X.shape)

    if n is not None:
        if yn_random:
            i = np.random.randint(0, N, size=n)
        else:
            i = np.arange(n)

    if i is None:
        i = np.random.randint(0, N, size=1)[0]
    elif np.size(i) > 1:
        list_i = i
        n = np.size(i)
        r, c = auto_row_col(n)
        h_fig, h_axes = plt.subplots(r, c)
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        h_axes = np.ravel(h_axes)
        for i_plot, i in enumerate(list_i):
            plt.axes(h_axes[i_plot])
            data_plot(X, Y, i)
        return h_fig

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


def auto_row_col(N, nrows=None):
    """ automatically determine num rows and num_columns """
    if nrows is None:
        nrows = int(np.floor(np.sqrt(N)))
    ncols, rem = divmod(N, nrows)
    ncols = ncols + (rem>0)
    return nrows, ncols


def subplots_autorc(n=None, nrows=None, ncols=None,  **kwargs):
    """
    subplots either by specify nrows, ncols, or specify total number of panels
    :param n:      total number of panels
    :param nrows:  nrows, useful when n is not given
    :param ncols:  nrows, useful when n is not given
    :param kwargs:
    :return:
    """
    if n is not None:
        nrows, ncols = auto_row_col(n, nrows=nrows)
    h_fig, h_axes = plt.subplots(nrows, ncols, **kwargs)
    h_axes = np.ravel(h_axes)
    return h_fig, h_axes


def data_binarize(X, threshold=0.5, states='0,1'):
    """ return binary form of data, either '0,1' or '-1,1' """

    if states == '0,1':
        return (X>threshold)*1
    elif states == '-1,1':
        return (X>threshold)*2-1


def imshow_fast_subplot(data_list, layout=None, gap = 0.05,
                    yn_nmlz = False, cmap='inferno'):
    """
    stick image data into a big image for fast subplot
    :param data_list: 1D list of all images to plot
    :param layout:    number of panels, or (nrow, ncol)
    :param gap:       size of gap relative to penal size
    :param yn_nmlz:   True/False to normalize data in every panel
    :param cmap:      colormap to use
    :return:
    """

    N_data = len(data_list)


    if layout is None:             # if not give, set layout to be the number of data
        layout = N_data
    if np.array(layout).size==1:   # if is a single number, turn to [num_row, num_col]
        ncol = int(np.ceil(np.sqrt(layout)))
        nrow = int(np.ceil(1.0*layout/ncol))
        layout = (nrow, ncol)
    if np.array(layout).size == 2:
        nrow, ncol = layout

    plt.axes([0.00, 0.00, 1.00, 1.00])

    """ get the sizes """
    ny_mesh, nx_mesh = data_list[0].shape    # size of every panel
    nx_gap = int(np.ceil(nx_mesh * gap))     # size of gap between panels
    ny_gap = int(np.ceil(ny_mesh * gap))
    nx_cell = nx_mesh+nx_gap                 # a cell is panel with gap
    ny_cell = ny_mesh+ny_gap
    nx_shift = nx_gap//2                     # shift the starting point of panel to be half the gap
    ny_shift = ny_gap // 2
    nx_canvas = (nx_mesh + nx_gap) * ncol    # size of canvas
    ny_canvas = (ny_mesh + ny_gap) * nrow

    # initialize the data for canvas
    mesh_canvas = np.zeros([ny_canvas, nx_canvas])*np.nan
    # contains the data for mask (create frames between panels)
    mask_canvas = np.zeros([ny_canvas, nx_canvas])

    """ normalize individual mesh plot to range [0,1] before putting them together """
    if yn_nmlz:
        for i in range(N_data):
            cur_min = np.nanmin(data_list[i])
            cur_max = np.nanmax(data_list[i])
            data_list[i] = (data_list[i] - cur_min) / (cur_max - cur_min)

    """ put mesh plot together into the big canvas matrix """
    def indx_in_canvas(indx, rowcol='row', startend='start'):
        # function to compute the index on cavas
        if rowcol == 'row':
            n_cell = ny_cell
            n_shift  = ny_shift
            n_mesh = ny_mesh
        else:
            n_cell = nx_cell
            n_shift = nx_shift
            n_mesh = nx_mesh
        return indx * n_cell + n_shift + n_mesh * (startend=='end')


    for i in range(N_data):
        row = i // ncol    # row index of panel
        col = i %  ncol    # col index of panel
        # fill the mesh data of the panel in to the right location of canvas matrix for mesh plot
        mesh_canvas[indx_in_canvas(row, 'row','start') : indx_in_canvas(row, 'row','end'),
                    indx_in_canvas(col, 'col', 'start') : indx_in_canvas(col, 'col','end')] = data_list[i]
        # fill value 1.0 to the pixels that contains mesh data in the canvas matrix for mask
        mask_canvas[indx_in_canvas(row, 'row','start') : indx_in_canvas(row, 'row','end'),
                    indx_in_canvas(col, 'col', 'start') : indx_in_canvas(col, 'col','end')] = 1

    # create colormap for mask (transparent if 0.0, opaque if 1.0)
    cmap_mask = mpl.colors.LinearSegmentedColormap.from_list('cmap_mask', [(0.9, 0.9, 0.9, 1.0), (0.9, 0.9, 0.9, 0.1)], N=2)

    """ plot big matrix containing all mesh plots """
    # # mesh data
    plt.imshow(mesh_canvas, vmin=np.nanmin(mesh_canvas), vmax=np.nanmax(mesh_canvas), cmap=cmap, aspect='auto')
    # # maks that forms the frames that seperates data panels
    plt.imshow(mask_canvas, vmin=0, vmax=1, cmap=cmap_mask, aspect='auto')

    """ make plot look better """
    # set y axis direction in the imshwow format
    ylim =np.array(plt.gca().get_ylim())
    plt.gca().set_ylim( ylim.max(), ylim.min() )
    plt.axis('off')


def gen_distinct_colors(n, luminance=0.9, alpha=0.8, style='discrete', cm='rainbow'):
    """
    tool funciton to generate n distinct colors for plotting

    :param n:          num of colors
    :param luminance:  num between [0,1]
    :param alhpa:      num between [0,1]
    :param style:      sting, 'discrete', or 'continuous'
    :param cm:      sting, 'discrete', or 'continuous'
    :return:           n*4 rgba color matrix
    """

    assert style in ('discrete', 'continuous')
    colormap = getattr(plt.cm, cm)
    if style == 'discrete':
        magic_number = 0.618  # from the golden ratio, to make colors evely distributed
        initial_number = 0.25
        colors_ini = colormap((initial_number + np.arange(n)) * magic_number % 1)
    else:    # style == 'continuous':
        colors_ini = colormap( 1.0*np.arange(n)/(n-0.5) )

    return colors_ini* np.array([[luminance, luminance, luminance, alpha]])
