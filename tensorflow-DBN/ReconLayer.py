import numpy as np
import time
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import DenseLayer


class ReconLayer(DenseLayer):
    

    def __init__(
        self,
        layer=None,
        x_recon=None,
        name='recon_layer',
        n_units=100,
        visible_unit_type = 'bin',
        weight_cost = 0.0001,
        momentum = 0.5,
        learning_rate = 0.001,
        gibbs_sampling_steps = 1,
        batch_size = 100,
        act=tf.nn.sigmoid,
        b_init=tf.constant_initializer(value=0.1)
			
    ):
        DenseLayer.__init__(self, layer=layer, n_units=n_units, act=act, name=name,b_init=b_init)
        print("     [TL] %s is a ReconLayer" % self.name)


        self.train_params = self.all_params[-4:]
        self.layer = layer
        self.x = x_recon
        self.name = name
        self.n_units = n_units
        self.visible_unit_type = visible_unit_type
        self.weight_cost = weight_cost
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.gibbs_sampling_steps = gibbs_sampling_steps
        self.batch_size = batch_size
        
        self.w_update = tf.Variable(tf.constant(0.0, shape=[n_units, layer.n_units]),name='weights_update')
        self.bh_update = tf.Variable(tf.constant(0.0, shape=[layer.n_units]),name='bh_update')
        self.bv_update = tf.Variable(tf.constant(0.0, shape=[n_units]),name='bv_update')     
        self.hrand = tf.placeholder(tf.float32, [None, layer.n_units], name='hrand')
        self.vrand = tf.placeholder(tf.float32, [None, n_units], name='vrand')       
        self.outputs = self.sample_hidden_from_visible(self.x)[0]
        
        self._init_graph()
        
    def _init_graph(self):
 

        '''k-step contrastive divergence'''
        hprob0, hstate0, vprob, hprob1, hstate1 = self.gibbs_sampling_step(
            self.x, self.n_units)
        positive = self.compute_positive_association(self.x,
                                                     hprob0, hstate0)

        nn_input = vprob

        for step in range(self.gibbs_sampling_steps - 1):
            hprob, hstate, vprob, hprob1, hstate1 = self.gibbs_sampling_step(
                nn_input, n_units)
            nn_input = vprob

        negative = tf.matmul(tf.transpose(vprob), hprob1) 


        
        W_update_new = self.momentum * self.w_update + self.learning_rate * (positive - negative) / self.batch_size - self.learning_rate * self.weight_cost * self.train_params[0]
        w_2 = self.train_params[0].assign_add(W_update_new)                               
        w_3 = self.train_params[2].assign(tf.transpose(w_2))              
                                                                                 
        w_update_2 = self.w_update.assign(W_update_new)

        bh_update_new = self.momentum * self.bh_update +tf.multiply(self.learning_rate, tf.reduce_mean(
            tf.subtract(hprob0, hprob1), 0))
        bh_2 = self.train_params[1].assign_add(bh_update_new)
        bh_update_2 = self.bh_update.assign(bh_update_new)

        bv_update_new = self.momentum * self.bv_update +tf.multiply(self.learning_rate, tf.reduce_mean(
            tf.subtract(self.x, vprob), 0))
        bv_2 = self.train_params[3].assign_add(bv_update_new)
        bv_update_2 = self.bv_update.assign(bv_update_new)

        updates = [w_3, bh_2, bv_2]
        updates_delta = [w_update_2,bh_update_2,bv_update_2]
        self.train_op = updates + updates_delta

        # Mean-square-error 
        self.cost = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(vprob, self.x))))
        

        
       

    def gibbs_sampling_step(self, visible, n_features):

        hprobs, hstates = self.sample_hidden_from_visible(visible)
        vprobs = self.sample_visible_from_hidden(hprobs, n_features)
        hprobs1, hstates1 = self.sample_hidden_from_visible(vprobs)

        return hprobs, hstates, vprobs, hprobs1, hstates1

    def sample_hidden_from_visible(self, visible):

        hprobs = tf.nn.sigmoid(tf.add(tf.matmul(visible, self.train_params[0]), self.train_params[1]))
        hstates = self.sample_prob(hprobs,self.hrand)

        return hprobs, hstates

    def sample_visible_from_hidden(self, hidden, n_features):

        visible_activation = tf.add(
            tf.matmul(hidden, tf.transpose(self.train_params[0])),
            self.train_params[3]
        )

        if self.visible_unit_type == 'bin':
            vprobs = tf.nn.sigmoid(visible_activation)

        elif self.visible_unit_type == 'gauss':
            vprobs = tf.truncated_normal(
                (1, n_features), mean=visible_activation, stddev=1) 


        else:
            vprobs = None

        return vprobs

    def compute_positive_association(self, visible,
                                     hidden_probs, hidden_states):

        if self.visible_unit_type == 'bin':
            #positive = tf.matmul(tf.transpose(visible), hidden_states)
            positive = tf.matmul(tf.transpose(visible), hidden_probs)

        elif self.visible_unit_type == 'gauss':
            positive = tf.matmul(tf.transpose(visible), hidden_probs)

        else:
            positive = None

        return positive

    def sample_prob(self,probs,rand):

        return tf.nn.relu(tf.sign(probs - rand))

    def pretrain(self, sess, x, X_train, X_val, denoise_name=None, n_epoch=100, batch_size=128, print_freq=10, save=True, save_name='w1pre_'):

        print("     [*] %s start pretrain" % self.name)
        print("     batch_size: %d" % batch_size)
        if denoise_name:
            print("     denoising layer keep: %f" % self.all_drop[set_keep[denoise_name]])
            dp_denoise = self.all_drop[set_keep[denoise_name]]
        else:
            print("     no denoising layer")

        self.batch_size = batch_size
        feed_dict = {x: X_train}
        #sess.run(self.normalization, feed_dict=feed_dict)
        for epoch in range(n_epoch):
            start_time = time.time()
            if epoch >= 5:
                self.momentum = 0.9
            for X_train_a, _ in tl.iterate.minibatches(X_train, X_train, self.batch_size, shuffle=True):
                dp_dict = tl.utils.dict_to_one(self.all_drop)
                rand_dict = {self.hrand: np.random.rand(X_train_a.shape[0], self.layer.n_units),
                self.vrand: np.random.rand(X_train_a.shape[0], self.n_units)}
                if denoise_name:
                    dp_dict[set_keep[denoise_name]] = dp_denoise
                feed_dict = {x: X_train_a}
                feed_dict.update(dp_dict)
                feed_dict.update(rand_dict)
                sess.run(self.train_op, feed_dict=feed_dict)

            if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
                print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
                train_loss, n_batch = 0, 0
                for X_train_a, _ in tl.iterate.minibatches(X_train, X_train, self.batch_size, shuffle=True):
                    dp_dict = tl.utils.dict_to_one(self.all_drop)
                    feed_dict = {x: X_train_a}
                    feed_dict.update(dp_dict)
                    err = sess.run(self.cost, feed_dict=feed_dict)
                    train_loss += err
                    n_batch += 1
                print("   train loss: %f" % (train_loss / n_batch))
                val_loss, n_batch = 0, 0
                for X_val_a, _ in tl.iterate.minibatches(X_val, X_val, self.batch_size, shuffle=True):
                    dp_dict = tl.utils.dict_to_one(self.all_drop)
                    feed_dict = {x: X_val_a}
                    feed_dict.update(dp_dict)
                    err = sess.run(self.cost, feed_dict=feed_dict)
                    val_loss += err
                    n_batch += 1
                print("   val loss: %f" % (val_loss / n_batch))
                if save:
                    try:
                        tl.visualize.W(self.train_params[0].eval(), second=10, saveable=True, shape=[28, 28], name=save_name + str(epoch + 1), fig_idx=2012)
                        tl.files.save_npz([self.all_params[0]], name=save_name + str(epoch + 1) + '.npz')
                    except:
                        raise Exception(
                            "You should change the tl.visualize.W() in ReconLayer.pretrain(), if you want to save the feature images for different dataset")

