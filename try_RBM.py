"""
try Boltzmann machine using tensorflow. start from MNIST dataset
based on http://deeplearning.net/tutorial/rbm.html
"""

""" load packages """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import importlib
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
import sklearn.preprocessing as preprocess
import tensorflow as tf
import utils

matplotlib.style.use('ggplot')

""" load and plot mnist data """

# load nmist data
(X_dtr, y_dtr), (X_dvl, y_dvl), (X_dts, y_dts) = utils.load_data()

# unravel data for doing convolution:
# X_dtr = utils.data_unravel(X_dtr)[:, :, :, None]
# X_dvl = utils.data_unravel(X_dvl)[:, :, :, None]
# X_dts = utils.data_unravel(X_dts)[:, :, :, None]



# one hot encoding of labels
ohe = preprocess.OneHotEncoder(sparse=False)
ohe.fit(y_dtr[:, None])
Y_dtr = ohe.transform(y_dtr[:, None])
Y_dvl = ohe.transform(y_dvl[:, None])
Y_dts = ohe.transform(y_dts[:, None])

# plot example data
importlib.reload(utils)
h_fig, h_ax = plt.subplots(nrows=4, ncols=5)
for ax in h_ax.ravel():
    plt.axes(ax)
    utils.data_plot(X_dtr, y_dtr)


M0 = X_dtr.shape[1]
M1 = 512

batch_size = 32
total_steps = 1000000
learning_rate = 0.03
wd_l2 = 0.001         # weight decay
""" define a tf graph for computation """
graph = tf.Graph()
with graph.as_default():
    """ constant, variable and placeholder """
    # place holder
    X0_in = tf.placeholder(dtype=tf.float32, shape=[batch_size, M0])

    # variable
    b0 = tf.Variable(tf.truncated_normal(shape = [M0], stddev=0.1))
    b1 = tf.Variable(tf.truncated_normal(shape = [M1], stddev=0.1))
    W1 = tf.Variable(tf.truncated_normal(shape=[M0, M1], stddev=0.1))

    """ network operations """
    def cal_p(X, W, b):
        return tf.nn.sigmoid(tf.matmul(X, W) + b)

    def sample_X(p):
        sample_shape = p.shape
        return tf.where(p>tf.random_uniform(sample_shape), tf.ones(sample_shape), tf.zeros(sample_shape))

    def sample_Gibbs(X0, counter=1, W1=W1, b0=b0, b1=b1):
        W1t = tf.transpose(W1)
        def one_cycle_Gibbs(X0, p0, counter):
            X1 = sample_X(cal_p(X0, W1, b1))
            p0 = cal_p(X1, W1t, b0)
            X0 = sample_X(p0)
            counter -= 1
            return X0, p0, counter
        X0_gibbs, p0_gibbs, _ = tf.while_loop(lambda _0, _1, c: c>0, one_cycle_Gibbs, [X0, X0, counter])
        return p0_gibbs

    """ contrastive_divergence_CD """
    # with tf.control_dependencies(assign_all):

    # up, down and sample
    W1t= tf.transpose(W1)
    p1 = cal_p(X0_in, W1, b1)
    X1 = sample_X(p1)
    p0 = cal_p(X1, W1t, b0)
    X0 = sample_X(p0)
    p1_= cal_p(X0, W1, b1)

    # gradient of log(p(x0, x1)) with weight decay (L2 loss)
    grad_b0 = tf.reduce_mean(X0_in - p0, axis=0) - wd_l2 * b0
    grad_b1 = tf.reduce_mean(p1 - p1_, axis=0) - wd_l2 * b1
    grad_W1 = tf.reduce_mean(X0_in[:, :, None] * p1[:, None, :] - p0[:, :, None] * X1[:, None, :], axis=0) - wd_l2 * W1

    # update: gradient ascend
    assign_b0 = tf.assign(b0, b0 + learning_rate * grad_b0)
    assign_b1 = tf.assign(b1, b1 + learning_rate * grad_b1)
    assign_W1 = tf.assign(W1, W1 + learning_rate * grad_W1)
    assign_all = [assign_b0, assign_b1, assign_W1]


    # evaluation measure: free energy
    energy = - tf.reduce_mean(X0_in * b0[None, :], axis=[0,1]) - tf.reduce_mean(X1 * b1[None, :], axis=[0,1]) \
             - tf.reduce_mean(tf.matmul(tf.matmul(X0_in, W1), tf.transpose(X1)), axis=[0, 1])
    # saver
    saver = tf.train.Saver()


""" running a session """
time_tic = time.time()
steps_check = 200
with tf.Session(graph=graph) as session:
    # tf.global_variables_initializer().run()
    saver.restore(session, './checkpoints/mnist_RBM.ckpt')
    for step in range(total_steps):
        offset = (step * batch_size) % (len(X_dtr) - batch_size)
        feed_dict = {X0_in: X_dtr[offset:offset + batch_size]}
        session.run(assign_all, feed_dict=feed_dict)
        if step % steps_check == 0:
            energy_eval = session.run(energy, feed_dict=feed_dict)
            time_per_batch = (time.time()-time_tic)/steps_check
            time_tic = time.time()
            print('step={:>6}, energy={:>6f}, sec/step={:>6f}'.format(step, energy_eval, time_per_batch))

    temp = W1.eval()
    saver.save(session, './checkpoints/mnist_RBM.ckpt')

""" GPU: sec/step=0.010806 """
""" CPU: sec/step=0.094567 """

# plot example filters
importlib.reload(utils)
h_fig, h_ax = plt.subplots(nrows=4, ncols=5)
for ax in h_ax.ravel():
    plt.axes(ax)
    utils.data_plot(temp.transpose())


""" analyze the network: the effect of familiarity """
X_dvl_rot = utils.data_ravel((np.rot90(utils.data_unravel(X_dvl), k=1, axes=[1,2])))
X_dvl_noi = (np.random.rand(*X_dvl.shape) + X_dvl)*0.5
with tf.Session(graph=graph) as session:
    X_vl_org = tf.constant(X_dvl, dtype='float32')
    X_vl_rot = tf.constant(X_dvl_rot, dtype='float32')
    X_vl_noi = tf.constant(X_dvl_noi, dtype='float32')
    saver.restore(session, './checkpoints/mnist_RBM.ckpt')
    X1_vl_org = cal_p(X_vl_org, W1, b1).eval()
    X1_vl_rot = cal_p(X_vl_rot, W1, b1).eval()
    X1_vl_noi = cal_p(X_vl_noi, W1, b1).eval()

def cal_tuning(H):
    tuning_H = np.vstack([np.sort(H[:, i])[::-1] for i in range(H.shape[1])])
    mean_tuning_H = np.mean(tuning_H, axis=0)
    q25_tuning_H = np.percentile(tuning_H, 25, axis=0)
    q75_tuning_H = np.percentile(tuning_H, 75, axis=0)
    return tuning_H, mean_tuning_H, q25_tuning_H, q75_tuning_H

_, tuning_org, _, _ = cal_tuning(X1_vl_org)
_, tuning_rot, _, _ = cal_tuning(X1_vl_rot)
_, tuning_noi, _, _ = cal_tuning(X1_vl_noi)
plt.plot(tuning_org)
plt.plot(tuning_rot)
plt.plot(tuning_noi)


""" analyze the network: samples """
h_fig, h_ax = plt.subplots(nrows=4, ncols=5)
h_ax = h_ax.ravel()
with tf.Session(graph=graph) as session:
    saver.restore(session, './checkpoints/mnist_RBM.ckpt')
    # X_gibbs = tf.random_uniform(shape=[20, M0], dtype='float32')
    X_gibbs = ( 3* tf.constant(X_dvl[:20,:], dtype='float32') + tf.random_uniform(shape=[20, M0], dtype='float32') )/4

    for i in range(20):
        X_gibbs = session.run(sample_Gibbs(sample_X(X_gibbs), 2, W1, b0, b1))
        for j, ax in enumerate(h_ax):
            plt.axes(ax)
            utils.data_plot(X_gibbs[j:j+1,:])
        plt.pause(0.5)



h_fig, h_ax = plt.subplots(nrows=4, ncols=5)
for ax in h_ax.ravel():
    plt.axes(ax)
    utils.data_plot(temp)



""" do not use TensorFlow, comare speed """

X0_in = np.zeros(shape=[batch_size, M0])

b0 = np.random.randn(M0)*0.1
b1 = np.random.randn(M1)*0.1
W1 = np.random.randn(M0, M1)*0.1

def cal_p(X, W, b, direction='up'):
    def sigmoid(x):
        return 1/(1+np.exp(-x))

    return sigmoid(np.matmul(X, W) + b)


def sample_X(p):
    sample_shape = p.shape
    return np.floor(p + np.random.rand(*sample_shape))


steps_check = 200
time_tic = time.time()
for step in range(total_steps):
    offset = (step * batch_size) % (len(X_dtr) - batch_size)
    X0_in = X_dtr[offset:offset + batch_size]

    """ contrastive_divergence_CD """
    # up, down and sample
    W1t = np.transpose(W1)
    p1 = cal_p(X0_in, W1, b1)
    X1 = sample_X(p1)
    p0 = cal_p(X1, W1t, b0)
    X0 = sample_X(p0)
    p1_ = cal_p(X0, W1, b1)

    # gradient
    grad_b0 = np.mean(X0_in - p0, axis=0)
    grad_b1 = np.mean(p1 - p1_, axis=0)
    grad_W1 = np.mean(X0_in[:, :, None] * p1[:, None, :] - p0[:, :, None] * X1[:, None, :], axis=0)

    # update: gradient ascend
    b0 += learning_rate * grad_b0
    b1 += learning_rate * grad_b1
    W1 += learning_rate * grad_W1


    if step % steps_check == 0:
        energy = - np.mean(X0_in * b0[None, :]) - np.mean(X1 * b1[None, :]) \
                 - np.mean(np.matmul(np.matmul(X0_in, W1), np.transpose(X1)))
        time_per_batch = (time.time()-time_tic)/steps_check
        time_tic = time.time()
        print('step={:>6}, energy={:>6f}, sec/step={:>6f}'.format(step, energy_eval, time_per_batch))

""" numpy: sec/step=0.113794, ten times slower than tensorflow on gpu """
""" TF GPU: sec/step=0.010806 """
""" TF CPU: sec/step=0.094567 """





""" below as tests for tf functions """


# test of tf.where, used for sample from Bernoulli distribution
temp = tf.where(0.1>tf.random_uniform([10]), tf.zeros([10]), tf.ones([10]))
with tf.Session() as session:
    print(temp.eval())

# test for loops
graph = tf.Graph()
with graph.as_default():
    i = tf.constant(0)
    temp = [i, i]
    c = lambda a, _: a<10
    b = lambda a, _: [a+1, _]
    r = tf.while_loop(c, b, temp, back_prop=False)
with tf.Session(graph=graph) as session:
    print(session.run(r))


