import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from DBN import DBN


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
X_train = mnist.train.images
Y_train = mnist.train.labels
X_test = mnist.test.images
Y_test = mnist.test.labels


dbn = DBN(        
        n_units = [784,500,300,100,10],
        learning_rate_rbm = [0.001,0.001,0.001],
        batch_size_rbm = [100,100,100],
        n_epoch_rbm = [10,10,10],
        visible_unit_type_rbm = ['bin','bin','bin'],
        weight_cost_rbm = 0.0001,
	    momentum_rbm = 0.5,
        learning_rate_dbn = 0.001,
        batch_size_dbn = 100,
        n_epoch_dbn = 100,
        dropout_dbn = [1,1,1])

dbn.pretrain(X_train,X_test)
dbn.fit(X_train,Y_train,X_test,Y_test)
 
