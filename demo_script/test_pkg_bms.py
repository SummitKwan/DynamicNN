""" script to test the bms (Boltzmann machine) pakage """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib


import bms
import utils

importlib.reload(bms.rbm)
importlib.reload(bms)

(dataX_tr, dataY_tr), (dataX_vl, dataY_vl), (dataX_ts, dataY_ts) = utils.load_data()

test_rbm = bms.rbm.RBM(784,128)
test_rbm.tensors    # store tensors in an dictionary
test_rbm.get_theta()
test_rbm.contrastive_divergence(dataX=dataX_tr, num_steps=100000)




# test TfModel object
import bms.rbm
from bms.rbm import TfModel
a = 0
b = 1
c = [2,3]


