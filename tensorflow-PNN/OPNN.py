import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm

class OPNN(object):
    def __init__(self, D1,feature_size, field_size,
                 embedding_size=8, 
                 deep_layers=[32, 32], dropout_deep=[0.5, 0.5, 0.5],
                 deep_layers_activation=tf.nn.relu,
                 epoch=10, batch_size=256,
                 learning_rate=0.001,):

        self.D1 = D1 #first hidden layer units

        self.feature_size = feature_size        # denote as M, size of the feature dictionary
        self.field_size = field_size            # denote as F, size of the feature fields
        self.embedding_size = embedding_size    # denote as K, size of the feature embedding
        self.deep_layers = deep_layers
        self.dropout_deep = dropout_deep
        self.deep_layers_activation = deep_layers_activation
        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self._init_graph()


    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():

            self.feat_index = tf.placeholder(tf.int32, shape=[None, None],
                                                 name="feat_index")  # None * F
            self.feat_value = tf.placeholder(tf.float32, shape=[None, None],
                                                 name="feat_value")  # None * F
            self.y_ = tf.placeholder(tf.float32, shape=[None], name="y_")  # None * 1
            self.dropout_keep_deep = tf.placeholder(tf.float32, shape=[None], name="dropout_keep_deep")

            self.weights = self._initialize_weights()

            # model
            self.embeddings = tf.nn.embedding_lookup(self.weights["feature_embeddings"],
                                                             self.feat_index)  # None * F * K
            feat_value = tf.reshape(self.feat_value, shape=[-1, self.field_size, 1])
            self.embeddings = tf.multiply(self.embeddings, feat_value)

            # ---------- z ---------- embeddings层与z各节点的全连接
            self.z = tf.reshape(self.embeddings, shape=[-1, self.field_size * self.embedding_size])  # None * (F*K)
            self.z = tf.matmul(self.z , self.weights["WZ"]) # None * D1
            self.z = tf.nn.dropout(self.z, self.dropout_keep_deep[0])

            # ---------- p --------------- embeddings层各fi向量之和形成F,F(T).F再与权重矩阵做multiply
            self.p = None
            for i,split_wp in enumerate(tf.split(self.weights["WP"], num_or_size_splits=self.D1, axis=0)):
                if i== 0:
                    self.F = tf.reduce_sum(self.embeddings,1,keep_dims=True)
                    self.p = tf.reshape(tf.reduce_sum(tf.multiply(tf.matmul(tf.transpose(self.F,perm=[0, 2, 1]),self.F ),split_wp),[1,2],keep_dims=True),
                                        shape=[-1,1])
                else:
                    self.p = tf.concat([self.p ,tf.reshape(tf.reduce_sum(tf.multiply(tf.matmul(tf.transpose(self.F,perm=[0, 2, 1] ),self.F ),split_wp),[1,2],keep_dims=True),
                                        shape=[-1,1])],axis=1)


            # None * D1
            self.p = tf.nn.dropout(self.p, self.dropout_keep_deep[0])

            # ---------- Deep component ----------
            self.y_deep = self.deep_layers_activation(self.z + self.p + self.weights['b1'])
            self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[0])

            for i in range(0, len(self.deep_layers)):
                self.y_deep = tf.add(tf.matmul(self.y_deep, self.weights["layer_%d" %i]), self.weights["bias_%d"%i]) # None * layer[i] * 1
                self.y_deep = self.deep_layers_activation(self.y_deep)
                self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[1+i]) # dropout at each Deep layer

            # ---------- output ----------
            self.y = tf.add(tf.matmul(self.y_deep, self.weights["concat_projection"]), self.weights["concat_bias"])
            self.y = tf.nn.sigmoid(self.y)
            self.y = tf.reshape(self.y,shape=[-1])
            
            #train_op...
            self.cost = tf.losses.log_loss(self.y_, self.y)
            self.train_op = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate, use_locking=False).minimize(self.cost)
        
            # init
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)
            



    def _initialize_weights(self):
        weights = dict()

        # embeddings
        weights["feature_embeddings"] = tf.Variable(
            tf.random_normal([self.feature_size, self.embedding_size], 0.0, 0.01),
            name="feature_embeddings")  # feature_size * K
        weights["feature_bias"] = tf.Variable(
            tf.random_uniform([self.feature_size, 1], 0.0, 1.0), name="feature_bias")  # feature_size * 1

        #product layer
        weights["WZ"] = tf.Variable(
            tf.random_normal([self.field_size*self.embedding_size,self.D1], 0.0, 0.01),name="WZ")
        weights["WP"] = tf.Variable(
            tf.random_normal([self.D1,self.embedding_size,self.embedding_size], 0.0, 0.01),name="WP")

        #first hidden layer
        glorot = np.sqrt(2.0 / (2*self.D1 + self.D1))
        weights["b1"] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.D1)),
                                                        dtype=np.float32)

        # deep layers
        num_layer = len(self.deep_layers)
        input_size = self.D1
        glorot = np.sqrt(2.0 / (input_size + self.deep_layers[0]))
        weights["layer_0"] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(input_size, self.deep_layers[0])), dtype=np.float32)
        weights["bias_0"] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[0])),
                                                        dtype=np.float32)  # 1 * layers[0]
        for i in range(1, num_layer):
            glorot = np.sqrt(2.0 / (self.deep_layers[i-1] + self.deep_layers[i]))
            weights["layer_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[i-1], self.deep_layers[i])),
                dtype=np.float32)  # layers[i-1] * layers[i]
            weights["bias_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[i])),
                dtype=np.float32)  # 1 * layer[i]

        # final concat projection layer

        input_size = self.deep_layers[-1]
        glorot = np.sqrt(2.0 / (input_size + 1))
        weights["concat_projection"] = tf.Variable(
                        np.random.normal(loc=0, scale=glorot, size=(input_size, 1)),
                        dtype=np.float32)  # layers[i-1]*layers[i]
        weights["concat_bias"] = tf.Variable(tf.constant(0.01), dtype=np.float32)

        return weights

    def gen_batches(self,data, batch_size):
        data = np.array(data)

        for i in range(0, data.shape[0], batch_size):
            yield data[i:i + batch_size]

    def fit(self,Xi_train,Xv_train,Y_train,Xi_val,Xv_val,Y_val):
    

        
        shuff = list(zip(Xi_train,Xv_train, Y_train))
        
        pbar = tqdm(range(self.epoch))
        for i in pbar:           
            np.random.shuffle(shuff)
            batches = [_ for _ in self.gen_batches(
                shuff, self.batch_size)]
            for batch in batches:
                xi_batch,xv_batch, y_batch = zip(*batch)
                feed_dict = {self.feat_index:xi_batch,
                             self.feat_value:xv_batch,
                             self.y_:y_batch,
                             self.dropout_keep_deep: self.dropout_deep}
                self.sess.run(self.train_op, feed_dict=feed_dict)
                
            loss = self.test(Xi_train,Xv_train,Y_train)
            loss_test = self.test(Xi_val,Xv_val,Y_val)
            pbar.set_description("Train Loss: %f,Test Loss: %f" % (loss,loss_test))


    def test(self,Xi_test,Xv_test,Y_test):  
        feed_dict = {self.feat_index:Xi_test,
                     self.feat_value:Xv_test,
                     self.y_:Y_test,
                     self.dropout_keep_deep: [1.0] * len(self.dropout_deep)}
        loss = self.sess.run(self.loss, feed_dict=feed_dict)
        return loss


    def predict_proba(self,Xi_test,Xv_test):
        feed_dict = {self.feat_index:Xi_test,
                     self.feat_value:Xv_test,
                     self.dropout_keep_deep: [1.0] * len(self.dropout_deep)}
        predict = self.sess.run(self.y, feed_dict=feed_dict)
        return predict






