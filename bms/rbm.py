""" module of RBM (restricted Boltzmann machine implemented using TensorFlow ) """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import tensorflow as tf

from . import utils

class TfModel:
    """ container of all objects in a tf graph, e.g., placeholders, variables and operations """

    def __init__(self, list_of_names=[]):
        """
        add variables as the attributes of a TfModel object

        :param list_of_names: a single str of a var name, or a list of str for multiple variables
        """

        if isinstance(list_of_names, list):
            self.add_vars(list_of_names)
        elif isinstance(list_of_names, str):
            self.add_var(list_of_names)

    def add_var(self, name=''):
        if name != '':
            print(name)
            print(eval(name))
            setattr(self, name, eval(name))

    def add_vars(self, list_of_names=[]):
        for name in list_of_names:
            self.add_var(name=name)


class RBM:
    """ RBM class """

    def __init__(self,
                 M0,             # num of visible layer units
                 M1,             # num of hidden  layer units
                 batchsize = 32,         # batchsize
                 lr = 0.01,      # learning rate
                 momentum=0.95,  # momentum for CD learning
                 wd_l2 = 0.0001  # weight decay, L2 loss
                 ):

        # set the attributes of RBM
        self.M = [M0, M1]
        self.batchsize = batchsize
        self.lr = lr
        self.momentum = momentum
        self.wd_l2 = wd_l2
        # self.theta = [np.zeros(M0), np.zeros(M1), np.zeros([M0, M1])]   # initial parameters

        self.graph = tf.Graph()  # initialize the tf computational graph
        self.tensors = utils.ObjDict()   # dict of tf tensors
        self.create_tf_graph()


    def create_tf_graph(self):
        """ create the tf graph """

        M0, M1 = self.M   # network size
        batchsize = self.batchsize        # batch szie
        lr = self.lr      # learning rate
        momentum = self.momentum  # momentum for CD learning
        wd_l2 = self.wd_l2
        graph = self.graph        # tf graph

        with graph.as_default():
            X0_in = tf.placeholder(dtype=tf.float32, shape=[batchsize, M0], name='X0_in')

            # variables
            b0 = tf.Variable(tf.truncated_normal(shape=[M0], stddev=0.1), name='b0')
            b1 = tf.Variable(tf.truncated_normal(shape=[M1], stddev=0.1), name='b1')
            W1 = tf.Variable(tf.truncated_normal(shape=[M0, M1], stddev=0.1), name='W1')


            """ network operations """

            def cal_p(X, W, b):
                return tf.nn.sigmoid(tf.matmul(X, W) + b)


            def sample_X(p):
                sample_shape = p.shape
                return tf.where(p > tf.random_uniform(sample_shape),
                                tf.ones(sample_shape), tf.zeros(sample_shape))

            def sample_Gibbs(X0, counter=1, W1=W1, b0=b0, b1=b1):
                W1t = tf.transpose(W1)

                def one_cycle_Gibbs(X0, p0, counter):
                    X1 = sample_X(cal_p(X0, W1, b1))
                    p0 = cal_p(X1, W1t, b0)
                    X0 = sample_X(p0)
                    counter -= 1
                    return X0, p0, counter

                X0_gibbs, p0_gibbs, _ = tf.while_loop(lambda _0, _1, c: c > 0,
                                                      one_cycle_Gibbs, [X0, X0, counter])
                return p0_gibbs

            """ contrastive_divergence_CD """
            # with tf.control_dependencies(assign_all):

            # up, down and sample
            W1t = tf.transpose(W1)
            p1 = cal_p(X0_in, W1, b1)
            X1 = sample_X(p1)
            p0 = cal_p(X1, W1t, b0)
            X0 = sample_X(p0)
            p1_ = cal_p(X0, W1, b1)

            # gradient of log(p(x0, x1)) with weight decay (L2 loss)
            grad_b0 = tf.reduce_mean(X0_in - p0, axis=0) - wd_l2 * b0
            grad_b1 = tf.reduce_mean(p1 - p1_, axis=0) - wd_l2 * b1
            grad_W1 = tf.reduce_mean(X0_in[:, :, None] * p1[:, None, :] - p0[:, :, None] * X1[:, None, :],
                                     axis=0) - wd_l2*W1

            # update: gradient ascend
            assign_b0 = tf.assign_add(b0, lr * grad_b0)
            assign_b1 = tf.assign_add(b1, lr * grad_b1)
            assign_W1 = tf.assign_add(W1, lr * grad_W1)
            assign_all = [assign_b0, assign_b1, assign_W1]

            # evaluation measure: free energy
            energy = - tf.reduce_mean(X0_in * b0[None, :], axis=[0, 1]) - tf.reduce_mean(X1 * b1[None, :], axis=[0, 1]) \
                     - tf.reduce_mean(tf.matmul(tf.matmul(X0_in, W1), tf.transpose(X1)), axis=[0, 1])
            # saver
            saver = tf.train.Saver()

        # store tensor names to a dictionary for other functions to access them
        list_var_names = ['X0_in',
                          'b0', 'b1', 'W1',
                          'assign_all',
                          'energy',
                          'saver']
        dict_var = utils.ObjDict()
        for name in list_var_names:
            dict_var[name] = eval(name)

        self.tensors = dict_var
        # dict_var = {k: eval(k) for k in list_var_names}
        # self.dict_var = dict_var


    def contrastive_divergence(self, dataX=None, num_steps=1, steps_per_check=100):
        """ do contrastive divergence training in a tf session """

        num_data = len(dataX)
        batchsize = self.batchsize
        X0_in = self.tensors.X0_in
        assign_all = self.tensors.assign_all

        time_tic = time.time()
        with tf.Session(graph=self.graph) as session:
            tf.global_variables_initializer().run()
            for step in range(num_steps):
                offset = (step * batchsize) % (num_data - batchsize)
                feed_dict = {X0_in: dataX[offset:offset + batchsize]}
                session.run(assign_all, feed_dict=feed_dict)
                if step % steps_per_check == 0:
                    energy_eval = session.run(self.tensors.energy, feed_dict=feed_dict)
                    time_per_batch = (time.time() - time_tic) / steps_per_check
                    time_tic = time.time()
                    print('step={:>6}, energy={:>6f}, sec/step={:>6f}'.format(step, energy_eval,
                                                                              time_per_batch))

            self.tensors.saver.save(session, './checkpoints/mnist_RBM.ckpt')




        pass


    def get_theta(self, var_names = None):
        """ evaluate tf variables and returns  """

        with tf.Session(graph=self.graph) as session:
            tf.global_variables_initializer().run()
            temp = self.tensors.W1.eval()

        return temp