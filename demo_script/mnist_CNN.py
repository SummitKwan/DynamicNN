""" load packages """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import importlib
import numpy as np
import datetime
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
X_dtr = utils.data_unravel(X_dtr)[:, :, :, None]
X_dvl = utils.data_unravel(X_dvl)[:, :, :, None]
X_dts = utils.data_unravel(X_dts)[:, :, :, None]



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

# get augmented data: horizontal flip
X_dtr_flip = X_dtr[:, :, ::-1]

h_fig, h_ax = plt.subplots(nrows=4, ncols=5)
for ax in h_ax.ravel():
    plt.axes(ax)
    utils.data_plot(X_dtr_flip, y_dtr)

X_dtr_all = np.concatenate((X_dtr, X_dtr_flip), axis=0)
Y_dtr_all = np.concatenate((Y_dtr, Y_dtr), axis=0)

indx_reorder = np.random.permutation(X_dtr_all.shape[0])
X_dtr_all = X_dtr_all[indx_reorder]
Y_dtr_all = Y_dtr_all[indx_reorder]


""" use tensorflow to train and NN model (LeNet), based on Udacity DeepLearning course assignment 4 """
# tf_device = '/device:CPU:0'
tf_device = '/device:GPU:0'

image_size = X_dtr.shape[1]
num_channels = 1
num_labels = Y_dtr.shape[1]

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

# model parameters

batch_size = 64
M_conv_p = 5  # patch size
M_conv1_d = 6  # depth
M_conv2_d = 16  # depth
M_pool_s = 2  # pooling stride
M_pool_k = 2  # pooling kernel size
M_full3 = 120
M_full4 = 84
lambda_l2 = 0.002    # coefficient for L2 penalty of connection weights



def size_after(size_in, stride=1, kernel=1, padding='SAME'):
    """ tool function to calculate the size of next layer after convolution or pooling """
    devident, remainder = divmod(size_in - (padding == 'VALID') * (kernel - 1), stride)
    return devident + (remainder > 0)


size_M1_conv = size_after(image_size, kernel=M_conv_p, padding='SAME')
size_M1_pool = size_after(size_M1_conv, stride=M_pool_s, kernel=M_pool_k, padding='VALID')
size_M2_conv = size_after(size_M1_pool, kernel=M_conv_p, padding='VALID')
size_M2_pool = size_after(size_M2_conv, stride=M_pool_s, kernel=M_pool_k, padding='VALID')

M2_size = size_M2_pool * size_M2_pool * M_conv2_d

graph = tf.Graph()

with graph.as_default():
    with tf.device(tf_device):
        # global step for rate decay
        global_step = tf.Variable(0)  # count the number of steps taken.
        learning_rate = tf.train.exponential_decay(0.2, global_step, decay_steps=10000, decay_rate=0.1)

        """ declare placeholder, constant and variable """
        # placeholder
        X_tr = tf.placeholder(dtype=tf.float32, shape=[batch_size, image_size, image_size, num_channels])
        Y_tr = tf.placeholder(dtype=tf.float32, shape=[batch_size, num_labels])

        # constant
        X_vl = tf.constant(X_dvl, dtype=tf.float32)
        X_ts = tf.constant(X_dts, dtype=tf.float32)

        # variable
        W1_conv = tf.Variable(tf.truncated_normal(shape=[M_conv_p, M_conv_p, num_channels, M_conv1_d], stddev=0.05))
        b1_conv = tf.Variable(tf.zeros(M_conv1_d))
        W2_conv = tf.Variable(tf.truncated_normal(shape=[M_conv_p, M_conv_p, M_conv1_d, M_conv2_d], stddev=0.05))
        b2_conv = tf.Variable(tf.zeros(M_conv2_d))
        W3_full = tf.Variable(tf.truncated_normal(shape=[M2_size, M_full3], stddev=0.05))
        b3_full = tf.Variable(tf.zeros(M_full3))
        W4_full = tf.Variable(tf.truncated_normal(shape=[M_full3, M_full4], stddev=0.05))
        b4_full = tf.Variable(tf.zeros(M_full4))
        W5_full = tf.Variable(tf.truncated_normal(shape=[M_full4, num_labels], stddev=0.05))
        b5_full = tf.Variable(tf.zeros(num_labels))

        """ define the neural network model """


        def model(X_in, dropout_keep_prob=1.0):
            X1_conv = tf.nn.relu(b1_conv + \
                                 tf.nn.conv2d(X_in, filter=W1_conv, strides=[1, 1, 1, 1], padding='SAME'))
            X1_pool = tf.nn.max_pool(X1_conv, ksize=[1, M_pool_k, M_pool_k, 1], strides=[1, M_pool_s, M_pool_s, 1],
                                     padding='SAME')
            X1_drop = tf.nn.dropout(X1_pool, keep_prob=dropout_keep_prob)
            X2_conv = tf.nn.relu(b2_conv + \
                                 tf.nn.conv2d(X1_drop, filter=W2_conv, strides=[1, 1, 1, 1], padding='VALID'))
            X2_pool = tf.nn.max_pool(X2_conv, ksize=[1, M_pool_k, M_pool_k, 1], strides=[1, M_pool_s, M_pool_s, 1],
                                     padding='SAME')
            X2_drop = tf.nn.dropout(X2_pool, keep_prob=dropout_keep_prob)
            X2_flat = tf.reshape(X2_drop, [-1, M2_size])
            X3_full = tf.nn.relu(b3_full + tf.matmul(X2_flat, W3_full))
            X3_drop = tf.nn.dropout(X3_full, keep_prob=dropout_keep_prob)
            X4_full = tf.nn.relu(b4_full + tf.matmul(X3_drop, W4_full))
            X4_drop = tf.nn.dropout(X4_full, keep_prob=dropout_keep_prob)
            X_out = b5_full + tf.matmul(X4_drop, W5_full)

            X_all_layers = {'X1': X1_pool, 'X2': X2_pool, 'X3': X3_full, 'X4': X4_full, 'X_out': X_out}

            return X_out, X_all_layers

        """ define training loss and optimizer """
        logits, X_all_layers_batch = model(X_tr, dropout_keep_prob=0.80)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_tr, logits=logits)) + \
               lambda_l2 * tf.nn.l2_loss(W1_conv) + lambda_l2 * tf.nn.l2_loss(W2_conv) + \
               lambda_l2 * tf.nn.l2_loss(W3_full) + lambda_l2 * tf.nn.l2_loss(W4_full)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

        """ get train/valid/test predictions """
        Y_hat_tr, X_all_layers_tr = model(X_tr, dropout_keep_prob=1.0)
        y_hat_tr = tf.nn.softmax(Y_hat_tr)
        Y_hat_vl, X_all_layers_vl = model(X_vl, dropout_keep_prob=1.0)
        y_hat_vl = tf.nn.softmax(Y_hat_vl)
        Y_hat_ts, X_all_layers_ts = model(X_ts, dropout_keep_prob=1.0)
        y_hat_ts = tf.nn.softmax(Y_hat_ts)

        saver = tf.train.Saver()

# make a session to run the model
num_steps = 3001

time_start = datetime.datetime.now()
with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True)) as session:
    tf.global_variables_initializer().run()  # remember to initialize variables
    print('model variable initialized')
    print('training with original and flipped data')
    for step in range(num_steps):
        offset = (step * batch_size) % (len(X_dtr_all) - batch_size)
        if offset < batch_size:  # shuffle the dataset after going over it once
            indx_shuffle = np.random.permutation(len(X_dtr_all))
            X_dtr_all = X_dtr_all[indx_shuffle]
            Y_dtr_all = Y_dtr_all[indx_shuffle]
        feed_dict = {X_tr: X_dtr_all[offset:offset + batch_size], Y_tr: Y_dtr_all[offset:offset + batch_size]}
        [_, loss_cur, y_hat_tr_cur] = session.run([optimizer, loss, y_hat_tr], feed_dict=feed_dict)
        if step % 100 == 0:
            acc_tr = accuracy(y_hat_tr_cur, Y_dtr_all[offset:offset + batch_size])
            acc_vl = accuracy(y_hat_vl.eval(), Y_dvl)
            print('step={:>6}, loss={:>6f},  accuracy: tr={:6.2f}, vl={:6.2f}'.format(step, loss_cur, acc_tr, acc_vl))
    acc_ts = accuracy(y_hat_ts.eval(), Y_dts)
    print('finished, loss={:>6f},  accuracy: vl={:6.2f}, test={:6.2f}'.format(loss_cur, acc_vl, acc_ts))

    print('training with only original data')
    for step in range(num_steps):
        offset = (step * batch_size) % (len(X_dtr) - batch_size)
        if offset < batch_size:  # shuffle the dataset after going over it once
            indx_shuffle = np.random.permutation(len(X_dtr))
            X_dtr = X_dtr[indx_shuffle]
            Y_dtr = Y_dtr[indx_shuffle]
        feed_dict = {X_tr: X_dtr[offset:offset + batch_size], Y_tr: Y_dtr[offset:offset + batch_size]}
        [_, loss_cur, y_hat_tr_cur] = session.run([optimizer, loss, y_hat_tr], feed_dict=feed_dict)
        if step % 100 == 0:
            acc_tr = accuracy(y_hat_tr_cur, Y_dtr[offset:offset + batch_size])
            acc_vl = accuracy(y_hat_vl.eval(), Y_dvl)
            print('step={:>6}, loss={:>6f},  accuracy: tr={:6.2f}, vl={:6.2f}'.format(step, loss_cur, acc_tr, acc_vl))
    acc_ts = accuracy(y_hat_ts.eval(), Y_dts)
    print('finished, loss={:>6f},  accuracy: vl={:6.2f}, test={:6.2f}'.format(loss_cur, acc_vl, acc_ts))

    saver.save(session, './checkpoints/mnist_CNN_LeNet_with_flip.ckpt')
print('training takes {}'.format(datetime.datetime.now()-time_start))



""" analyze the trained model """

# create a rotated dataset as novel images
X_dvl_flp = X_dvl[:, :, ::-1]
X_dvl_rot = (np.rot90(X_dvl, k=1, axes=[1,2]))
X_dvl_noi = ((np.random.rand(*X_dvl.shape) + X_dvl)/2).astype('float32')

# plot example data to compare
n_example = 4
indx_example = np.random.permutation(len(X_dvl))[:n_example]
h_fig, h_axes = plt.subplots(n_example, 4)
for i in range(n_example):
    plt.axes(h_axes[i, 0])
    utils.data_plot(X_dvl, y_dvl, i=indx_example[i])
    plt.axes(h_axes[i, 1])
    utils.data_plot(X_dvl_flp, y_dvl, i=indx_example[i])
    plt.axes(h_axes[i, 2])
    utils.data_plot(X_dvl_rot, y_dvl, i=indx_example[i])
    plt.axes(h_axes[i, 3])
    utils.data_plot(X_dvl_noi, y_dvl, i=indx_example[i])
plt.suptitle('example data: [fam (ori), nov (flip), non (rot), noi (noisy)]')
plt.savefig('./figures/example_fam_nov_non_noi.pdf')
plt.savefig('./figures/example_fam_nov_non_noi.png')

with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True)) as session:
    saver.restore(session, './checkpoints/mnist_CNN_LeNet_with_flip.ckpt')
    X_all_layers_vl_eval = {key: value.eval() for key, value in X_all_layers_vl.items()}

    Y_hat_vl, X_all_layers_vl = model(X_vl, dropout_keep_prob=1.0)
    y_hat_vl = tf.nn.softmax(Y_hat_vl)

    X_vl_flp = tf.constant(X_dvl_flp)
    Y_hat_vl_flp, X_all_layers_vl_flp = model(X_vl_flp, dropout_keep_prob=1.0)
    X_all_layers_vl_flp_eval = {key: value.eval() for key, value in X_all_layers_vl_flp.items()}

    X_vl_rot = tf.constant(X_dvl_rot)
    Y_hat_vl_rot, X_all_layers_vl_rot = model(X_vl_rot, dropout_keep_prob=1.0)
    X_all_layers_vl_rot_eval = {key: value.eval() for key, value in X_all_layers_vl_rot.items()}

    X_vl_flp = tf.constant(X_dvl_noi)
    Y_hat_vl_noi, X_all_layers_vl_noi = model(X_vl_flp, dropout_keep_prob=1.0)
    X_all_layers_vl_noi_eval = {key: value.eval() for key, value in X_all_layers_vl_noi.items()}

# tuning curve
def cal_tuning(H):
    if len(H.shape) > 2:
        H = H.reshape([len(H), -1])
    tuning_H = np.vstack([np.sort(H[:, i])[::-1] for i in range(H.shape[1])])
    mean_tuning_H = np.mean(tuning_H, axis=0)
    q25_tuning_H = np.percentile(tuning_H, 25, axis=0)
    q75_tuning_H = np.percentile(tuning_H, 75, axis=0)
    return tuning_H, mean_tuning_H, q25_tuning_H, q75_tuning_H

num_plot = len(X_all_layers_vl_eval)
nrows, ncols = utils.auto_row_col(num_plot, 3)
h_fig, h_axes = plt.subplots(nrows, ncols, sharex='all', figsize=[8,6])
h_axes = h_axes.ravel()
layer_names = sorted(X_all_layers_vl_eval.keys())
for i in range(num_plot):
    layer_name = layer_names[i]
    plt.axes(h_axes[i])

    layer_activity = X_all_layers_vl_eval[layer_name]
    tuning, mean_tuning, q25_tuning, q75_tuning = cal_tuning(layer_activity)
    plt.fill_between(np.arange(tuning.shape[1]), q25_tuning, q75_tuning, alpha=0.2)
    h_ori, = plt.plot(mean_tuning)

    layer_activity = X_all_layers_vl_flp_eval[layer_name]
    tuning, mean_tuning, q25_tuning, q75_tuning = cal_tuning(layer_activity)
    plt.fill_between(np.arange(tuning.shape[1]), q25_tuning, q75_tuning, alpha=0.2)
    h_flp, = plt.plot(mean_tuning)

    layer_activity = X_all_layers_vl_rot_eval[layer_name]
    tuning, mean_tuning, q25_tuning, q75_tuning = cal_tuning(layer_activity)
    plt.fill_between(np.arange(tuning.shape[1]), q25_tuning, q75_tuning, alpha=0.2)
    h_rot, = plt.plot(mean_tuning)

    layer_activity = X_all_layers_vl_noi_eval[layer_name]
    tuning, mean_tuning, q25_tuning, q75_tuning = cal_tuning(layer_activity)
    plt.fill_between(np.arange(tuning.shape[1]), q25_tuning, q75_tuning, alpha=0.2)
    h_noi, = plt.plot(mean_tuning)

    if i == 0:
        plt.legend([h_ori, h_flp, h_rot, h_noi], ['fam', 'nov', 'non', 'noi'])

    plt.title(layer_name)

h_axes[0].set_xticklabels([])
plt.suptitle('CNN: ave tuning curve of hidden layer activity')
plt.savefig('./figures/CNN_ave_tuning_curve.pdf')
plt.savefig('./figures/CNN_ave_tuning_curve.png')
# plt.savefig('./figures/CNN_ave_tuning_curve_with_inhibition.pdf')
# plt.savefig('./figures/CNN_ave_tuning_curve_with_inhibition.png')


