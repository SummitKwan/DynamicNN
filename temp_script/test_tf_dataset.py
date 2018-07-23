import importlib
import time
import numpy as np
import tensorflow as tf

import utils
import EBMs_tensorflow


importlib.reload(EBMs_tensorflow)

# load nmist data
(X_dtr, y_dtr), (X_dvl, y_dvl), (X_dts, y_dts) = utils.load_data()

m0, m1 = 784, 512
batchsize = 32


##

numpy_dataset = (np.arange(0,20)[:, None] * np.ones([1, 10])).astype('float32')

ph_dataset = tf.placeholder(dtype=tf.float32, shape=numpy_dataset.shape)
ds = tf.data.Dataset.from_tensor_slices(ph_dataset)
ds = ds.shuffle(buffer_size=10)
ds = ds.repeat()
ds = ds.map(lambda x: x + tf.random_normal([1]))
ds = ds.batch(5)


iterator = ds.make_initializable_iterator()

next_element = iterator.get_next()

with tf.Session() as sess:
    sess.run(iterator.initializer, feed_dict={ph_dataset: numpy_dataset})
    for i in range(10):
        temp = sess.run(next_element, feed_dict={ph_dataset: numpy_dataset})
        print(temp)
        print(temp.dtype)




