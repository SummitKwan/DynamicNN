""" test script for sub-package EBMs_tensorflow """

##
import importlib
import time
import numpy as np
import tensorflow as tf

import utils
import EBMs_tensorflow


importlib.reload(EBMs_tensorflow)

# load nmist data
(X_dtr, y_dtr), (X_dvl, y_dvl), (X_dts, y_dts) = utils.load_data()

##

m0, m1 = 784, 512
batchsize = 32

n_total = X_dtr.shape[0]
x_in = X_dtr[:32]


##
importlib.reload(EBMs_tensorflow.ebm)
importlib.reload(EBMs_tensorflow.rbm)
importlib.reload(EBMs_tensorflow)
model = EBMs_tensorflow.rbm.RestrictedBoltzmannMachine(m0, m1, batchsize=batchsize, path_log_dir='./model_log')

model.init_graph()
model.create_training_graph()

##
""" train model """

model.run_training(x0_data=X_dtr, num_epochs=20)


##
""" test model """



##
""" below is legacy code """

""" design computational graph """

with model.graph.as_default():

    # define computational graph
    x0_in = model.tensors['x0_in']

    op_cd = model.contrastive_divergence(x0=x0_in)

    op_energy = model.cal_energy(x0=x0_in)

    p1 = model.cal_p1_given_x0(x0_in)

    gibbs_outcome = model.gibbs_sample(x0_in, num_steps=10)

    with tf.name_scope('summary'):
        tf.summary.histogram('b0', model.tensors['b0'])
        tf.summary.histogram('b1', model.tensors['b1'])
        tf.summary.histogram('w', model.tensors['w'])
        tf.summary.histogram('p1', p1)
        merged_summary = tf.summary.merge_all()



""" train model """

num_epochs = 1
steps_check = 100
toc = time.time()
index_data_shuffle = np.arange(n_total)
yn_load_file = True
yn_save_file = False
yn_refresh_dict_params = False
model.lr = 0.003

with tf.Session(graph=model.graph) as session:

    tf.global_variables_initializer().run()

    if yn_load_file:
        model.load_parameters(filedir='./model_save', filename='RBM_tf')

    if yn_refresh_dict_params:
        model.init_dict_params()

    model.params_dict_to_tensor()
    x0_in = model.tensors['x0_in']
    op_energy = model.tensors['energy']
    op_cd = model.tensors['cd']
    merged_summary = model.tensors['merged_summary']


    print(np.std(model.dict_params['w']))

    for i_loop in range(num_epochs):
        for i_batch in range(n_total//model.batchsize-1):
            x_batch = X_dtr[index_data_shuffle[i_batch*model.batchsize: (i_batch+1)*model.batchsize]]

            if i_loop == 0:
                index_data_shuffle = np.random.permutation(n_total)

            if i_batch % steps_check == 0:
                cur_energy, summary_out = session.run([op_energy, merged_summary], feed_dict={x0_in: x_batch})
                tic, toc = toc, time.time()
                time_per_batch = (toc - tic) / steps_check
                with tf.summary.FileWriter('./model_log') as writer:
                    writer.add_summary(summary_out)
                print('step={:>4}_{:>5}, energy={:>+.5}, sec/batch={:>.4}, ms/sample={:.4}'.format(
                    i_loop, i_batch, np.mean(cur_energy), time_per_batch, time_per_batch/model.batchsize*1000))

            session.run(op_cd, feed_dict={x0_in: x_batch})

    model.params_tensor_to_dict()

    if yn_save_file:
        model.write_parameters(model.dict_params, filedir='./model_save', filename='RBM_tf')

    with tf.summary.FileWriter('./model_log') as writer:
        writer.add_graph(model.graph)


##
""" test reconstruction """

yn_load_file = False
i_batch = 0
x_batch = X_dtr[index_data_shuffle[i_batch*model.batchsize: (i_batch+1)*model.batchsize]]

with tf.Session(graph=model.graph) as session:

    with tf.summary.FileWriter('./model_log') as writer:
        writer.add_graph(session.graph)

    tf.global_variables_initializer().run()

    if yn_load_file:
        model.load_parameters(filedir='./model_save', filename='RBM_tf')

    model.params_dict_to_tensor()

    gibss_result = session.run(gibbs_outcome, feed_dict={x0_in: x_batch})

##


utils.data_plot(gibss_result[1], n=10)

utils.data_plot(model.dict_params['w'][:, :10].transpose(), n=10)

##

with tf.Session(graph=model.graph) as session:
    tf.global_variables_initializer().run()
    temp0 = session.run(model.cal_energy(x0=x_in))
    print(temp0)
    session.run(model.load_parameters(filedir='./model_save', filename='RBM_tf'))
    temp1 = session.run(model.cal_energy(x0=x_in))
    print(temp1)

##


with tf.Session(graph=model.graph) as session:
    temp1 = session.run(model.cal_energy(x0=x_in))
    print(temp1)

##
import tensorflow as tf

a = tf.constant(np.arange(4), shape=[1,4])
b = tf.constant(np.arange(2), shape=[2,1])
a[:, None, :]*b[:, None, :]


##
model = EBMs_tensorflow.ebm.EnergyBasedModel()
dict_params = model.read_parameters(filedir='model_save', filename='RBM')
dict_params_new = dict()
dict_params_new['b0'] = dict_params['b1'].ravel()
dict_params_new['b1'] = dict_params['b2'].ravel()
dict_params_new['w'] = dict_params['W']
dict_params = model.write_parameters(dict_params_new, filedir='model_save', filename='RBM_tf')