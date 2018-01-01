""" load packages """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import importlib
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as preprocess
import tensorflow as tf
import utils
from utils import load_data, data_unravel

""" load and plot mnist data """

# load nmist data
(X_dtr, y_dtr), (X_dvl, y_dvl), (X_dts, y_dts) = load_data()

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
    utils.data_plot(X_train, y_train)

""" use tensorflow to classify, based on Udacity DeepLearning course assignment 3 """
def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

# write the computational graph from scratch
N, M = X_dtr.shape
size_batch = 256
M_X = M
M_H0 = 512
M_H1 = 128
M_Y = Y_dtr.shape[1]
lambda_l2 = 0.001

# first, we need a grpah
graph = tf.Graph()
with graph.as_default():
    # global step for rate decay
    global_step = tf.Variable(0)  # count the number of steps taken.
    learning_rate = tf.train.exponential_decay(0.5, global_step, decay_steps=10000, decay_rate=0.1)

    # input data
    X_tr = tf.placeholder(dtype=tf.float32, shape=[size_batch, M_X])  # alarm: use placeholder, but not variable
    Y_tr = tf.placeholder(dtype=tf.float32, shape=[size_batch, M_Y])
    X_vl = tf.constant(X_dvl)
    X_ts = tf.constant(X_dts)

    # variables to train
    W0 = tf.Variable(tf.truncated_normal([M_X, M_H0], stddev=0.05))  # alarm: initilize them, make stddev small
    b0 = tf.Variable(tf.zeros([M_H0]))
    W1 = tf.Variable(tf.truncated_normal([M_H0, M_H1], stddev=0.05))
    b1 = tf.Variable(tf.zeros([M_H1]))
    W2 = tf.Variable(tf.truncated_normal([M_H1, M_Y], stddev=0.05))
    b2 = tf.Variable(tf.zeros([M_Y]))

    # loss
    H0 = tf.nn.dropout(tf.nn.relu(tf.matmul(X_tr, W0) + b0), keep_prob=0.7)  # alarm: use matrix multiply
    H1 = tf.nn.dropout(tf.nn.relu(tf.matmul(H0, W1) + b1), keep_prob=0.7)  # alarm: use matrix multiply
    logits = tf.matmul(H1, W2) + b2
    # loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_tr) ) # original
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_tr)) \
           + lambda_l2 * (tf.nn.l2_loss(W0) + tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2))

    # optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # model preditction
    Y_tr_hat = tf.nn.softmax(logits)

    def XtoH0(X):
        return tf.nn.relu(tf.matmul(X, W0) + b0)
    def H0toH1(H0):
        return tf.nn.relu(tf.matmul(H0, W1) + b1)
    def H1toY(H1):
        return tf.nn.softmax(tf.matmul(H1, W2) + b2)
    H0_vl = XtoH0(X_vl)
    H1_vl = H0toH1(H0_vl)
    Y_vl_hat = H1toY(H1_vl)
    H0_ts = XtoH0(X_ts)
    H1_ts = H0toH1(H0_ts)
    Y_ts_hat = H1toY(H1_ts)

    saver = tf.train.Saver()

# next, we need a session
num_steps = 6001
step_plot = []
acc_tr = []
acc_vl = []
loss_tr = []

with tf.Session(graph=graph) as session:
    # initilize variables
    tf.global_variables_initializer().run()

    # loop
    for step in range(num_steps):
        # get batch data
        offset = (step * size_batch) % (N - size_batch)
        X_batch = X_dtr[offset:offset + size_batch, :]
        Y_batch = Y_dtr[offset:offset + size_batch, :]
        feed_dict = {X_tr: X_batch, Y_tr: Y_batch}
        [_, l, predictions] = session.run([optimizer, loss, Y_tr_hat], feed_dict=feed_dict)
        if (step % 500 == 0):
            step_plot.append(step)
            loss_tr.append(l)
            acc_tr.append(accuracy(predictions, Y_batch))
            acc_vl.append(accuracy(Y_vl_hat.eval(), Y_dvl))
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, Y_batch))
            print("Validation accuracy: %.1f%%" % accuracy(Y_vl_hat.eval(), Y_dvl))
    saver.save(session, './checkpoints/mnist_4L.ckpt')
    print("Test accuracy: %.1f%%" % accuracy(Y_ts_hat.eval(),Y_dts))

# plot
plt.subplot(2, 1, 1)
plt.plot(step_plot, loss_tr)
plt.title('training loss')
plt.subplot(2, 1, 2)
plt.plot(step_plot, acc_tr)
plt.plot(step_plot, acc_vl)
plt.legend(['train', 'validate'])
plt.title('accuracty')


""" analyze the trained model """
# create a rotated dataset as novel images
X_dvl_rot = utils.data_ravel((np.rot90(utils.data_unravel(X_dvl), k=1, axes=[1,2])))

with tf.Session(graph=graph) as session:
    saver.restore(session, './checkpoints/mnist_4L.ckpt')
    aH0_vl = H0_vl.eval()
    aH1_vl = H1_vl.eval()
    X_vl_rot = tf.constant(X_dvl_rot)
    H0_vl_rot = XtoH0(X_vl_rot)
    H1_vl_rot = H0toH1(H0_vl_rot)
    aH0_vl_rot = H0_vl_rot.eval()
    aH1_vl_rot = H1_vl_rot.eval()


# tuning curve
def cal_tuning(H):
    tuning_H = np.vstack([np.sort(H[:, i])[::-1] for i in range(H.shape[1])])
    mean_tuning_H = np.mean(tuning_H, axis=0)
    q25_tuning_H = np.percentile(tuning_H, 25, axis=0)
    q75_tuning_H = np.percentile(tuning_H, 75, axis=0)
    return tuning_H, mean_tuning_H, q25_tuning_H, q75_tuning_H


plt.figure()
tuning_H0, mean_tuning_H0, q25_tuning_H0, q75_tuning_H0 = cal_tuning(aH0_vl)
tuning_H1, mean_tuning_H1, q25_tuning_H1, q75_tuning_H1 = cal_tuning(aH1_vl)
plt.fill_between(np.arange(tuning_H0.shape[1]), q25_tuning_H0, q75_tuning_H0, alpha=0.2)
plt.fill_between(np.arange(tuning_H1.shape[1]), q25_tuning_H1, q75_tuning_H1, alpha=0.2)
h_H0, = plt.plot(mean_tuning_H0)
h_H1, = plt.plot(mean_tuning_H1)
# plt.legend([h_H0, h_H1], ['H0', 'H1'])
# plt.title('tuning cure')

tuning_H0_rot, mean_tuning_H0_rot, q25_tuning_H0_rot, q75_tuning_H0_rot = cal_tuning(aH0_vl_rot)
tuning_H1_rot, mean_tuning_H1_rot, q25_tuning_H1_rot, q75_tuning_H1_rot = cal_tuning(aH1_vl_rot)
plt.fill_between(np.arange(tuning_H0_rot.shape[1]), q25_tuning_H0_rot, q75_tuning_H0_rot, alpha=0.2)
plt.fill_between(np.arange(tuning_H1_rot.shape[1]), q25_tuning_H1_rot, q75_tuning_H1_rot, alpha=0.2)
h_H0_rot, = plt.plot(mean_tuning_H0_rot)
h_H1_rot, = plt.plot(mean_tuning_H1_rot)
plt.legend([h_H0, h_H1, h_H0_rot, h_H1_rot], ['H0_fam', 'H1_fam', 'H0_nov', 'H1_nov'])
plt.title('MLP: ave tuning curve of hidden layer activity')
plt.savefig('./figures/MLP_ave_tuning_curve.pdf')
plt.savefig('./figures/MLP_ave_tuning_curve.png')
