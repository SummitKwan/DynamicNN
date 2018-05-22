"""
Boltzmann machines, fully observed, no hidden units, implemented in numpy
Shaobo Guan,
2018-0517
"""

import time
import numpy as np



def sigmoid_fun(z):
    """ sigmoid activation function, does not change the shape is input z """
    return 1.0 / (1.0 + np.exp(-z))


def bernoulli_sample(p):
    """ bernoulli sampling process, does not change the shape of input p(x_i=1) """
    return (np.random.rand(*p.shape) <= p) * 1


def remove_diagonal(W):
    return W * (1 - np.identity(W.shape[0]))


class BoltzmannNetNoHidden:

    def __init__(self, m):
        """
        initialize model
        :param m:  number of nodes in network
        """
        self.m = m
        self.b = np.random.randn(*[1, m]) * 0.001
        self.W = np.random.randn(*[1, m]) * 0.001
        self.W = remove_diagonal(self.W + self.W.transpose())

    def cal_energy(self, X):
        """
        compute energy
        :param X:  data, shape=[num_data, num_neurons]
        :return:   energy of hopfield model, shape=[num_data]
        """
        energy = -np.sum(X*self.b + np.matmul(X, self.W) * X / 2, axis=1)
        return energy

    def inference(self, X, mask_update=None, num_steps=1):
        """
        evolve to a steady state using MCMC
        :param X:  data, shape=[num_data, num_neurons]
        :param mask_update:  binary masks of whether to allow updating every pixel, shape=[num_data, num_neurons]
        :param num_steps: number of steps to run the sampling process over the network
        :return:   updated sample of X
        """
        n, m = X.shape
        X = X + 0
        if mask_update is None:
            mask_update = np.ones(X.shape).astype('bool')
        for iter in range(num_steps):
            for mm in np.random.permutation(m):
                Xmm = bernoulli_sample(sigmoid_fun(np.matmul(X, self.W[:, mm]) + self.b[:, mm]))
                X[:, mm] = Xmm * mask_update[:, mm] + X[:, mm] * (1 - mask_update[:, mm])

        return X

    def train(self, X, lr=0.001, steps_negstats=1, batchsize=50, yn_verbose=False, batches_per_print=100):
        """ training using exponential family's gradient formula """
        n, m = X.shape
        num_batch = n//batchsize
        tic = time.time()
        for i_batch in range(num_batch):
            self.train_batch(X[i_batch*batchsize : i_batch*batchsize+batchsize], lr=lr, steps_negstats=steps_negstats)
            if yn_verbose and i_batch%batches_per_print==0:
                toc = time.time()
                print('batch {}/{}, time total={:.0f} sec'.format(i_batch, num_batch, toc-tic))

    def train_batch(self, X, lr=0.001, steps_negstats=1):
        """ training using exponential family's gradient formula """
        n, m = X.shape
        X_sample = self.inference(X, num_steps=steps_negstats)

        stats_pos_uni = np.mean(X, axis=0, keepdims=True)
        stats_pos_bin = np.matmul(X.transpose(), X) * 1.0 / n

        stats_neg_uni = np.mean(X_sample, axis=0, keepdims=True)
        stats_neg_bin = np.matmul(X_sample.transpose(), X_sample) * 1.0 / n

        self.b += lr * (stats_pos_uni - stats_neg_uni)
        self.W += lr * (stats_pos_bin - stats_neg_bin)
        self.W = remove_diagonal(self.W)




