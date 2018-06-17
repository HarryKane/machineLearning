import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm


class logisticRegression(object):

    def __init__(
        self,
        learning_rate = 0.1,
        epoch = 100,
        bacth_size = 100,
        feature_size = 10
        ):
        
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.batch_size = bacth_size
        self.feature_size = feature_size
        
        self._init_graph()
        
        
    def _init_graph(self):
    
        self.graph = tf.Graph()
        with self.graph.as_default():
            
            self.x = tf.placeholder(tf.float32, shape=[None, self.feature_size+1], name='x')
            self.weights = self._initialize_weights()
            self.y = self.sigmoid(tf.matmul(self.x,self.weights['weights']))
            self.y_clf = tf.round(self.y)
            self.y_ = tf.placeholder(tf.float32, shape=[None, 1], name='y_')
            
            self.update_weights = self.weights['weights'].assign_add(self.learning_rate * 1/self.batch_size * tf.matmul(tf.transpose(self.x),self.y_ - self.y))
            
            self.loss = self.y_ * tf.matmul(self.x,self.weights['weights']) - tf.log(1 + tf.exp(tf.matmul(self.x,self.weights['weights'])))
            self.loss = tf.reduce_mean(self.loss)
            self.acc = tf.equal(self.y_clf,self.y_)
            self.acc = tf.reduce_mean(tf.cast(self.acc, "float"))
            
            # init
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)
            
            
            
            
    def _initialize_weights(self):
        weights = dict()
        weights["weights"] = tf.Variable(tf.random_normal([self.feature_size+1, 1], 0.0, 0.01),
            name="weights")
            
        return weights

    def sigmoid(self,X):
        return 1.0/(1+tf.exp(-X))
        
    def gen_batches(self,data, batch_size):
        data = np.array(data)

        for i in range(0, data.shape[0], batch_size):
            yield data[i:i + batch_size]
        
    def fit(self,X_train,Y_train,X_val,Y_val):
    
        X_train = pd.DataFrame(X_train)
        X_train['b_'] = 1
        X_train = np.array(X_train)
        X_val = pd.DataFrame(X_val)
        X_val['b_'] = 1
        X_val = np.array(X_val)
        Y_train = np.array(Y_train)
        Y_val = np.array(Y_val)
        
        shuff = list(zip(X_train, Y_train))
        
        pbar = tqdm(range(self.epoch))
        for i in pbar:           
            np.random.shuffle(shuff)
            batches = [_ for _ in self.gen_batches(
                shuff, self.batch_size)]
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                feed_dict = {self.x:X_train,self.y_:Y_train}
                self.sess.run(self.update_weights, feed_dict=feed_dict)
                
            acc,loss = self.test(X_train,Y_train)
            acc_test,loss_test = self.test(X_val,Y_val)
            pbar.set_description("Train Accuracy: %f,Loss: %f,Test Accuracy: %f,Loss: %f" % (acc,loss,acc_test,loss_test))              
            
        
        
    def test(self,X_test,Y_test):  
        feed_dict = {self.x:X_test,self.y_:Y_test}
        acc = self.sess.run(self.acc, feed_dict=feed_dict)
        loss = self.sess.run(self.loss, feed_dict=feed_dict)
        return acc,loss
        
        
    def predict(self,X_test):
        feed_dict = {self.x:X_test}
        predict = self.sess.run(self.y_clf, feed_dict=feed_dict)
        return predict
       
    def predict_proba(self,X_test):
        feed_dict = {self.x:X_test}
        predict_proba = self.sess.run(self.y, feed_dict=feed_dict)
        return predict_proba



