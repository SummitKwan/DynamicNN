"""
hopfield model, implemented in numpy
Shaobo Guan,
2018-0517
"""

import numpy as np

class HopfieldNet():

    def __init__(self, m):
        """
        initialize model
        :param m:  number of nodes in network
        """
        self.m = m
        self.W = np.zeros([m, m])
        self.b = np.zeros([1, m])

    def train_Hebbian(self, X):
        """
        train network using Hebbian rule, update W
        :param X:  data, shape=[num_data, num_neurons]
        :return: None
        """

        n, m = X.shape
        assert m == self.m
        self.b = np.mean(X, axis=0, keepdims=True)
        self.W = (np.matmul(X.transpose(), X) * (1.0-np.eye(m))) / n

    def cal_energy(self, X):
        """
        compute energy
        :param X:  data, shape=[num_data, num_neurons]
        :return:   energy of hopfield model, shape=[num_data]
        """
        energy = -np.sum(X*self.b + np.matmul(X, self.W) * X, axis=1)
        return energy

    def inference(self, X, num_steps=1):
        """
        evolve to a steady state
        :param X:
        :return:
        """
        n, m = X.shape
        X_new = X + 0
        for iter in range(num_steps):
            for mm in np.random.permutation(m):
                X_new[:, mm] = np.sign(np.matmul(X, self.W[:, mm:mm+1]) + self.b[0, mm])
        return X_new

    def train(self, X):
        self.train_Hebbian(X)

