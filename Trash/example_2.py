import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np
from navieBayes import navieBayes


mnist = input_data.read_data_sets("../data/MNIST_data/", one_hot=True)
X_train = mnist.train.images
Y_train = mnist.train.labels
X_test = mnist.test.images
Y_test = mnist.test.labels


clf = navieBayes(        
        lambda_ls = 1)

clf.fit(train_x,train_y)
acc = clf.test(X_test,Y_test)
print(acc)
 
