"""
Restricted Boltzmann machines,  implemented in numpy
Shaobo Guan,
2018-0813
"""

import os
import time
import re
import numpy as np
import h5py
import pickle


modelname = 'RBM'

def sigmoid_fun(z):
    """ sigmoid activation function, does not change the shape is input z """
    return 1.0 / (1.0 + np.exp(-z))


def bernoulli_sample(p):
    """ bernoulli sampling process, does not change the shape of input p(x_i=1) """
    return (np.random.rand(*p.shape) <= p) * 1


def remove_diagonal(W):
    return W * (1 - np.identity(W.shape[0]))


class RestrictedBoltzmannMachine:

    def __init__(self, m1, m2):
        """
        initialize model
        :param m:  number of nodes in network
        """
        self.parameters = ('m1', 'm2', 'W', 'b1', 'b2')
        self.m1 = m1
        self.m2 = m2
        self.b1 = np.random.randn(*[1, m1]) * 0.001
        self.b2 = np.random.randn(*[1, m2]) * 0.001
        self.W = np.random.randn(*[m1, m2]) * 0.001


    def cal_energy(self, X, Y=None, batchsize=128):
        """
        compute energy
        :param X:  data, shape=[num_data, num_neurons]
        :return:   energy of hopfield model, shape=[num_data]
        """

        if Y is None:
            Y = self.compute_p_y_given_x(X)
        energy = -np.sum(X*self.b1,  axis=1) - np.sum(Y*self.b2, axis=1) \
                 - np.sum(X[:, :] * np.sum(Y[:, None, :] * self.W[None, :, :], axis=2), axis=1)
        return energy


    def compute_p_y_given_x(self, X):
        """
        inference of Y based on X

        :param X: data, shape=[num_data, m1]
        :return:  updated Y, shape=[num_data, m2]
        """
        return sigmoid_fun(np.sum(X[:, :, None] * self.W[None, :, :], axis=1) + self.b2 )


    def compute_p_x_given_y(self, Y):
        """
        inference of Y based on X

        :param Y: data, shape=[num_data, m2]
        :return:  updated X, shape=[num_data, m1]
        """
        return sigmoid_fun(np.sum(Y[:, None, :] * self.W[None, :, :], axis=2) + self.b1)


    def inference(self, X=None, Y=None, mask_update=None, num_steps=1, direction='auto', proportion_original=None):
        """
        evolve to a steady state using MCMC
        :param X:  data, shape=[num_data, num_neurons]
        :param mask_update:  binary masks of whether to allow updating every pixel, shape=[num_data, num_neurons]
        :param num_steps: number of steps to run the sampling process over the network
        :param direction: one of ['XY', 'YX', 'auto'].  X->Y or Y->X, default to 'auto', determined by whether X or Y is given
        :return:
        """

        assert (X is not None) or (Y is not None)
        assert direction in ['XY', 'YX', 'auto']
        if direction == 'auto':
            if (X is not None) and (Y is None):
                direction = 'XY'
            elif (X is None) and (Y is not None):
                direction = 'YX'
            else:
                direction = 'XY'
        X_orignal = X
        pX = None
        pY = None

        for iter in range(num_steps):
            if direction == 'XY':
                pY = self.compute_p_y_given_x(X)
                Y = bernoulli_sample(pY)
                direction = 'YX'
            elif direction == 'YX':
                pX = self.compute_p_x_given_y(Y)
                X = bernoulli_sample(pX)
                if mask_update is not None:
                    X = np.where(mask_update, X, X_orignal)
                if proportion_original is not None:
                    X = (1-proportion_original) * X + proportion_original * X_orignal
                direction = 'XY'

        return (X, Y), (pX, pY)


    def train(self, X, lr=0.001, steps_negstats=1, batchsize=64, yn_verbose=False, batches_per_print=100):
        """ training using contrastive divergence algorithm, wrapper of self.train_batch() """
        n = X.shape[0]
        num_batch = n//batchsize
        tic = time.time()
        # self.train_cost = {t: []; cost: []}
        for i_batch in range(num_batch):
            X_batch = X[i_batch * batchsize: i_batch * batchsize + batchsize]
            self.train_batch(X_batch, lr=lr, steps_negstats=steps_negstats)
            if yn_verbose and i_batch%batches_per_print == 0:
                toc = time.time()
                energy_batch = np.mean(self.cal_energy(X_batch))
                print('batch {}/{}, time total={:.0f} sec, energy_tr={}'.format(i_batch, num_batch, toc-tic, energy_batch))


    def train_batch(self, X, lr=0.001, steps_negstats=1):
        """ training using contrastive divergence algorithm """

        X_pos = X
        (_, Y_pos), (_, pY_pos) = self.inference(X=X_pos, num_steps=1)
        (X_neg, Y_neg), (pX_neg, pY_neg) = self.inference(Y=Y_pos, num_steps=steps_negstats*2)


        stats_X_pos = np.mean( X_pos, axis=0, keepdims=True)
        stats_Y_pos = np.mean(pY_pos, axis=0, keepdims=True)
        stats_XY_pos = np.mean(X_pos[:, :, None] * pY_pos[:, None, :], axis=0)

        stats_X_neg = np.mean(pX_neg, axis=0, keepdims=True)
        stats_Y_neg = np.mean(pY_neg, axis=0, keepdims=True)
        stats_XY_neg = np.mean(X_neg[:, :, None] * pY_neg[:, None, :], axis=0)

        self.b1 += lr * (stats_X_pos - stats_X_neg)
        self.b2 += lr * (stats_Y_pos - stats_Y_neg)
        self.W += lr * (stats_XY_pos - stats_XY_neg)



    def save_parameters(self, filepathname=None,
                   filedir='../model_save', filename=modelname, fileext='.h5', auto_timecode=True):
        """
        save model parameters to a hdf5 (h5) or pickle (pkl) file
        :param filepathname:  the complete file path and name of the file to store,
            if not given, generate using filedir, filename, fileext, autotimecode
        :param filedir:       directory of file
        :param filename:      name of file
        :param fileext:       extension of file, has to be either '.h5' or '.pkl'
        :param auto_timecode: whether automatically add datetime code to the end filename, default to True
        :return:
        """

        if filepathname is not None:
            filedir, basename = os.path.split(filepathname)
            filename, fileext = os.path.splitext(basename)
        else:
            assert os.path.isdir(filedir),   'directory {} does not exist'.format(filedir)
            assert fileext in ('.h5', '.pkl'), 'fileext must be one of (".h5", ".pkl")'
            if auto_timecode:
                time_str = '_' + time.strftime("%Y%m%d_%H%M%S")
            else:
                time_str = ''

            filepathname = os.path.join(filedir, filename + time_str + fileext)

        print('save model parameters to file {}'.format(filepathname))

        if fileext == '.h5':
            with h5py.File(filepathname, 'w') as f:
                for key in self.parameters:
                    f.create_dataset(key, data=getattr(self, key))
        elif fileext == '.pkl':
            with open(filepathname, 'wb') as f:
                data_dict = {key: getattr(self, key) for key in self.parameters}
                pickle.dump(data_dict, f)
        else:
            raise Exception('fileext must be one of (".h5", ".pkl")')


    def load_parameters(self, filepathname=None,
                   filedir='../model_save', filename=modelname, fileext='.h5', find_last=True):
        """
        save model parameters to a hdf5 (h5) or pickle (pkl) file
        :param filepathname: the complete file path and name of the file to store,
            if not given, generate using filedir, filename, fileext, find_last
        :param filedir:      directory of file
        :param filename:     name of file
        :param fileext:      extension of file, has to be either '.h5' or '.pkl'
        :param find_last:    whether to find the most recent model of not based on timecode
        :return:
        """

        if filepathname is not None:
            filedir, basename = os.path.split(filepathname)
            filename, fileext = os.path.splitext(basename)
        else:
            if find_last:
                files = os.listdir(filedir)
                file_keep = []
                date_keep = []
                for file in files:
                    if re.search(r'{}'.format(filename), file) and re.search(r'{}$'.format(fileext), file) \
                            and re.search('\d{8}_\d{6}', file):
                        timecode = re.search('\d{8}_\d{6}', file).group(0)
                        file_keep.append(file)
                        date_keep.append(timecode)
                filenameext = sorted(zip(date_keep, file_keep))[-1][1]
                filepathname = os.path.join(filedir, filenameext)
            else:
                filepathname = os.path.join(filedir, filename + fileext)

        print('loading model parameters from file {}'.format(filepathname))

        if fileext == '.h5':
            with h5py.File(filepathname, 'r') as f:
                for key in self.parameters:
                    setattr(self, key, f.get(key).value)
        elif fileext == '.pkl':
            with open(filepathname, 'rb') as f:
                data_dict = pickle.load(f)
                for key in self.parameters:
                    setattr(self, key, data_dict[key])
        else:
            raise Exception('fileext must be one of (".h5", ".pkl")')
