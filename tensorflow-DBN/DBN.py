import numpy as np
import tensorflow as tf
import tensorlayer as tl
from ReconLayer import ReconLayer

class DBN(object):

    def __init__(
        self,
        n_units = [],
        learning_rate_rbm = [],
        batch_size_rbm = [],
        n_epoch_rbm = [],
        visible_unit_type_rbm = [],
        weight_cost_rbm = 0.0001,
	    momentum_rbm = 0.5,
        learning_rate_dbn = 0.0001,
        batch_size_dbn = 100,
        n_epoch_dbn = 100,
        dropout_dbn = []
        ):
        assert len(n_units)-2==len(learning_rate_rbm)==len(batch_size_rbm)==len(dropout_dbn)==len(visible_unit_type_rbm), \
        "learning_rate_rbm,batch_size_rbm,dropout_dbn and visible_unit_type_rbm should have the same length"
        
        
        self.n_units = n_units
        self.learning_rate_rbm = learning_rate_rbm
        self.batch_size_rbm = batch_size_rbm
        self.n_epoch_rbm = n_epoch_rbm
        self.visible_unit_type_rbm = visible_unit_type_rbm
        self.weight_cost_rbm = weight_cost_rbm
        self.momentum_rbm = momentum_rbm
        self.learning_rate_dbn = learning_rate_dbn
        self.batch_size_dbn = batch_size_dbn
        self.n_epoch_dbn = n_epoch_dbn
        self.dropout_dbn = dropout_dbn
        
        self._init_graph()
      
    def _init_graph(self):
    
        self.graph = tf.Graph()
        with self.graph.as_default():
        
            #input
            self.x = tf.placeholder(tf.float32, shape=[None, self.n_units[0]], name='x')
            self.network = tl.layers.InputLayer(self.x, name='input_layer')
            
            #rbm
            self.recon_layers = []
            for i in range(1,len(self.n_units)-1):
                self.network = tl.layers.DropoutLayer(self.network, keep=self.dropout_dbn[i-1], name='drop'+str(i))
                self.network = tl.layers.DenseLayer(layer=self.network, n_units=self.n_units[i],act=tf.nn.sigmoid, name='sigmoid_'+str(i))
                self.recon_layers.append(ReconLayer(self.network, x_recon= self.x if i==1 else self.recon_layers[-1].outputs, n_units=self.n_units[i-1],
                visible_unit_type = self.visible_unit_type_rbm[i-1], weight_cost = self.weight_cost_rbm,momentum = self.momentum_rbm,
                learning_rate = self.learning_rate_rbm[i-1],name='recon_layer_'+str(i)))
            
            #dbn
            h_outputs = self.network.outputs
            W = tf.Variable(tf.truncated_normal([self.n_units[-2], self.n_units[-1]], stddev=0.1))
            b = tf.Variable(tf.constant(0.1, shape=[self.n_units[-1]]))
            self.y = tf.add(tf.matmul(h_outputs, W), b)
            
            #train_op...
            self.y_ = tf.placeholder(tf.float32, shape=[None,self.n_units[-1]], name='y_')
            self.cost = tf.losses.softmax_cross_entropy(self.y_, self.y)
            self.correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
            self.acc = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
            self.train_op = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate_dbn, use_locking=False).minimize(self.cost)
        
            # init
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)

    def pretrain(self,X_train,X_val):
        for i in range(len(self.learning_rate_rbm)):
            self.recon_layers[i].pretrain(self.sess, x=self.x, X_train=X_train, X_val=X_val,
                       denoise_name=None, n_epoch=self.n_epoch_rbm[i], batch_size=self.batch_size_rbm[i],
                        print_freq=1, save=False, save_name='w1pre_')
    
    def fit(self,X_train,Y_train,X_val,Y_val):
        tl.utils.fit(self.sess, self.network, self.train_op, self.cost, X_train, Y_train, self.x, self.y_,
        acc=self.acc, batch_size=self.batch_size_dbn, n_epoch=self.n_epoch_dbn, print_freq=1,
        X_val=X_val, y_val=Y_val, eval_train=True,tensorboard=False)
        
    def test(self,X_test,Y_test):  
        tl.utils.test(self.sess, self.network, self.acc, X_test, Y_test, self.x, self.y_,batch_size=None, cost=self.cost)
        
    def predict(self,X_test):
        feed_dict = {self.x:X_test}
        predict = self.sess.run(self.y, feed_dict=feed_dict)
        return predict
    
