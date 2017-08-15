#!/usr/bin/env python
#
# Copyright (c) 2014, Georgia Tech Research Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the Georgia Tech Research Corporation nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY GEORGIA TECH RESEARCH CORPORATION ''AS IS'' AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL GEORGIA TECH BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

#  \author Daehyung Park (Healthcare Robotics Lab, Georgia Tech.)

# system & utils
import os, sys, copy, random
import numpy
import numpy as np
import scipy

# Keras
import h5py 
from keras.models import Sequential, Model
from keras.layers import Merge, Input, TimeDistributed, Layer
from keras.layers import Activation, Dropout, Flatten, Dense, merge, Lambda, RepeatVector, LSTM
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD, Adagrad, Adadelta, RMSprop, Adam
from keras import backend as K
from keras import objectives

from hrl_anomaly_detection.vae import keras_util as ku
from hrl_anomaly_detection.vae import util as vutil

import gc



def lstm_vae(trainData, testData, weights_file=None, batch_size=32, nb_epoch=500, \
             patience=20, fine_tuning=False, save_weights_file=None, \
             noise_mag=0.0, timesteps=4, sam_epoch=1, \
             x_std_div=1.0, x_std_offset=0, z_std=0.5,\
             renew=False, plot=True, trainable=None, **kwargs):
    """
    Variational Autoencoder with two LSTMs and one fully-connected layer
    x_train is (sample x length x dim)
    x_test is (sample x length x dim)
    """

    x_train = trainData[0]
    x_test = testData[0]

    length = len(x_train[0])
    h1_dim = input_dim = len(x_train[0][0])
    z_dim  = 2

    def slicing(x): return x[:,:,:input_dim]
           
    inputs = Input(batch_shape=(batch_size, timesteps, input_dim+1))
    encoded = Lambda(slicing)(inputs)     
    encoded = LSTM(h1_dim, return_sequences=False, activation='tanh', stateful=True)(encoded)
    z_mean  = Dense(z_dim)(encoded) 
    z_log_var = Dense(z_dim)(encoded) 
    
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=K.shape(z_mean), mean=0., stddev=1.0)
        return z_mean + K.exp(z_log_var/2.0) * epsilon    
        
    # we initiate these layers to reuse later.
    decoded_h1 = Dense(h1_dim)
    decoded_h2 = RepeatVector(timesteps)
    decoded_L1 = LSTM(input_dim, return_sequences=True, activation='sigmoid', stateful=True)

    # Encoder --------------------------------------------------
    vae_encoder_mean = Model(inputs, z_mean)
    vae_encoder_var  = Model(inputs, z_log_var)

    # Decoder (generator) --------------------------------------
    ## decoder_input = Input(batch_shape=(batch_size,z_dim))
    ## _decoded = decoded_h1(decoder_input)
    ## _decoded = decoded_h2(_decoded)
    ## _decoded = decoded_L1(_decoded)
    ## generator = Model(decoder_input, _decoded)
    generator = None

    z = Lambda(sampling)([z_mean, z_log_var])    
    decoded = decoded_h1(z)
    decoded = decoded_h2(decoded)
    outputs = decoded_L1(decoded)

    vae_autoencoder = Model(inputs, outputs)
    print(vae_autoencoder.summary())

    def vae_loss(x_tr, x_pred):

        x_true = x_tr[:,:,:input_dim]
        p      = x_tr[:,0,input_dim-1:]

        x_mean = K.mean(K.reshape(x_pred, shape=(-1,input_dim)), axis=0) #dim
        x_std  = K.std(K.reshape(x_pred, shape=(-1,input_dim)), axis=0)/x_std_div+x_std_offset

        # sample x length 
        log_p_x_z = -0.5 * ( K.sum(K.square((x_true-x_mean)/x_std), axis=-1) \
                             + float(input_dim) * K.log(2.0*np.pi) 
                             + K.sum(K.log(K.square(x_std)), axis=-1) )
        # sample
        xent_loss = K.mean(-log_p_x_z, axis=-1)

        # sample
        sig2 = z_std
        kl_loss = - 0.5 * K.sum(1 + z_log_var -K.log(sig2*sig2) - K.square(z_mean-p)
                                - K.exp(z_log_var)/(sig2*sig2), axis=-1)
        ## ## kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    # VAE --------------------------------------
    vae_mean = Model(inputs, outputs)

    if weights_file is not None and os.path.isfile(weights_file) and fine_tuning is False and\
      renew is False:
        vae_autoencoder.load_weights(weights_file)
    else:
        if fine_tuning:
            vae_autoencoder.load_weights(weights_file)
            lr = 0.01            
            optimizer = Adam(lr=lr, clipvalue=4.)# 5)                
            #vae_autoencoder.compile(optimizer=optimizer, loss=vae_loss)
            vae_autoencoder.compile(optimizer='adam', loss=vae_loss)
        else:
            lr = 0.01
            #optimizer = RMSprop(lr=lr, rho=0.9, epsilon=1e-08, decay=0.0001, clipvalue=10)
            optimizer = Adam(lr=lr, clipvalue=10) #, decay=1e-5)                
            #vae_autoencoder.compile(optimizer=optimizer, loss=None)
            vae_autoencoder.compile(optimizer='adam', loss=vae_loss)
            #vae_autoencoder.compile(optimizer='rmsprop', loss=None)

        # ---------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------
        nDim         = len(x_train[0][0])
        wait         = 0
        plateau_wait = 0
        min_loss = 1e+15
        for epoch in xrange(nb_epoch):
            print 

            # ---------------------------------------------------------------------
            mean_tr_loss = []
            for sample in xrange(sam_epoch):

                # shuffle
                idx_list = range(len(x_train))
                np.random.shuffle(idx_list)
                x_train = x_train[idx_list]
                
                ## for i in xrange(0,len(x_train),batch_size): # per batch
                for i in xrange(len(x_train)): # per a sample
                    seq_tr_loss = []
                    np.random.seed(3334 + i)                        
                    x = x_train[i] + np.random.normal(0, noise_mag, (timesteps, nDim))

                    # Make batch size input
                    x = [x.tolist() for j in range(batch_size)]
                    x = np.array(x)

                    for j in xrange(len(x[0])-timesteps+1): # per window
                        p = float(j)/float(length-timesteps+1) *2.0-1.0
                        tr_loss = vae_autoencoder.train_on_batch(
                            np.concatenate((x[:,j:j+timesteps],
                                            p*np.ones((len(x), timesteps, 1))), axis=-1),
                            x[:,j:j+timesteps] )

                        seq_tr_loss.append(tr_loss)
                    mean_tr_loss.append( np.mean(seq_tr_loss) )
                    vae_autoencoder.reset_states()

                sys.stdout.write('Epoch {} / {} : loss training = {} , loss validating = {}\r'.format(epoch, nb_epoch, np.mean(mean_tr_loss), 0))
                sys.stdout.flush()

                
            # ---------------------------------------------------------------------
            mean_te_loss = []
            for i in xrange(0,len(x_test),batch_size):
                
                seq_te_loss = []

                # batch augmentation
                if i+batch_size > len(x_test):
                    x = x_test[i:]
                    r = i+batch_size-len(x_test)

                    for k in xrange(r/len(x_test)):
                        x = np.vstack([x, x_test])
                    
                    if (r%len(x_test)>0):
                        idx_list = range(len(x_test))
                        random.shuffle(idx_list)
                        x = np.vstack([x,
                                       x_test[idx_list[:r%len(x_test)]]])
                else:
                    x = x_test[i:i+batch_size]

                for j in xrange(len(x[0])-timesteps+1):
                    p = float(j)/float(length-timesteps+1) *2.0-1.0

                    te_loss = vae_autoencoder.test_on_batch(
                        np.concatenate((x[:,j:j+timesteps],
                                        p*np.ones((len(x), timesteps,1))), axis=-1),
                        x[:,j:j+timesteps] )
                    seq_te_loss.append(te_loss)
                mean_te_loss.append( np.mean(seq_te_loss) )
                vae_autoencoder.reset_states()

            val_loss = np.mean(mean_te_loss)
            sys.stdout.write('Epoch {} / {} : loss training = {} , loss validating = {}\r'.format(epoch, nb_epoch, np.mean(mean_tr_loss), val_loss))
            sys.stdout.flush()   


            # ---------------------------------------------------------------------
            # Early Stopping
            if val_loss <= min_loss:
                min_loss = val_loss
                wait         = 0
                plateau_wait = 0

                if save_weights_file is not None:
                    vae_autoencoder.save_weights(save_weights_file)
                else:
                    vae_autoencoder.save_weights(weights_file)
                
            else:
                if wait > patience:
                    print "Over patience!"
                    break
                else:
                    wait += 1
                    plateau_wait += 1

            # ---------------------------------------------------------------------
            #ReduceLROnPlateau
            if plateau_wait > 2:
                old_lr = float(K.get_value(vae_autoencoder.optimizer.lr))
                new_lr = old_lr * 0.2
                K.set_value(vae_autoencoder.optimizer.lr, new_lr)
                plateau_wait = 0
                print 'Reduced learning rate {} to {}'.format(old_lr, new_lr)

        gc.collect()

    # ---------------------------------------------------------------------------------
    # visualize outputs
    if False:
        print "latent variable visualization"
        

    if plot:
        print "variance visualization"
        nDim = len(x_test[0,0]) 
        
        for i in xrange(len(x_test)): # per sample
            if i!=6: continue #for data viz lstm_vae_custom -4 

            # Make batch size input
            x = [x_test[i].tolist() for j in range(batch_size)]
            x = np.array(x)

            vae_mean.reset_states()
            x_pred_mean = []
            x_pred_std  = []
            for j in xrange(len(x[0])-timesteps+1):
                
                x_pred = vae_mean.predict(np.concatenate((x[:,j:j+timesteps],
                                                          np.zeros((len(x), timesteps,1))
                                                          ), axis=-1),
                                                          batch_size=batch_size)
                #print np.shape(x_pred)
                x_pred_mean.append( np.mean(x_pred[:,-1,:nDim], axis=0) )
                x_pred_std.append( np.std(x_pred[:,-1,:nDim], axis=0) )
                #x_pred_std.append(x_pred[0,-1,nDim:]/x_std_div*1.5+x_std_offset)

            vutil.graph_variations(x_test[i], x_pred_mean, x_pred_std)#, scaler_dict=kwargs['scaler_dict'])
        


    return vae_autoencoder, vae_mean, None, vae_encoder_mean, vae_encoder_var, generator


## class ResetStatesCallback(Callback):
##     def __init__(self, max_len):
##         self.counter = 0
##         self.max_len = max_len
        
##     def on_batch_begin(self, batch, logs={}):
##         if self.counter % self.max_len == 0:
##             self.model.reset_states()
##         self.counter += 1



















    ## def marginal_loglikelihood(self, x, x_d_mean, x_d_std, p):
    ##     '''
    ##     p : phase variable
    ##     '''
    ##     log_p_x_z = -0.5 * ( K.sum(K.square((x-x_d_mean)/x_d_std), axis=-1) \
    ##                          + float(input_dim) * K.log(2.0*np.pi) + K.sum(K.log(K.square(x_d_std)),
    ##                                                                        axis=-1) )
    ##     xent_loss = K.mean(-log_p_x_z, axis=-1)

    ##     sig2 = z_std
    ##     kl_loss = - 0.5 * K.sum(1 + z_log_var -K.log(sig2*sig2) - K.square(z_mean-p)
    ##                             - K.exp(z_log_var)/(sig2*sig2), axis=-1)
    ##     ## kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    ##     return K.mean(xent_loss + kl_loss) 


    ## def vae_loss(y_true, y_pred):

    ##     z_mean_s    = K.repeat(z_mean, L)
    ##     z_log_var_s = K.repeat(z_log_var, L)

    ##     # Case 1: following sampling function
    ##     epsilon  = K.random_normal(shape=K.shape(z_mean_s), mean=0., stddev=1.0)
    ##     z_sample = z_mean_s + K.exp((z_log_var_s)/2.0) * epsilon

    ##     # Case 2: using raw
    ##     #?
        
    ##     x_sample = K.map_fn(generator, z_sample)

    ##     x_mean = K.mean(x_sample, axis=1) # length x dim
    ##     x_var  = K.var(x_sample, axis=1)

    ##     log_p_x_z = loglikelihood(y_true, loc=x_mean, scale=x_var)
    ##     xent_loss = K.mean(-log_p_x_z, axis=-1)
    ##     #xent_loss = K.sum(-log_p_x_z, axis=-1)
        
    ##     kl_loss   = -0.5 * K.mean(1.0 + z_log_var - K.square(z_mean) \
    ##                               - K.exp(z_log_var), axis=-1)

                                  
    ##     #loss = xent_loss + kl_loss
    ##     loss = K.mean(xent_loss + kl_loss)
    ##     ## K.print_tensor(xent_loss)
    ##     return loss












    ## # Custom loss layer
    ## class CustomVariationalLayer(Layer):
    ##     def __init__(self, **kwargs):
    ##         self.is_placeholder = True
    ##         super(CustomVariationalLayer, self).__init__(**kwargs)

    ##     def vae_loss(self, x_true, p):
    ##         '''
    ##         p : phase variable
    ##         x_true: sample x length x dim
    ##         z_mean: sample x z_dim
    ##         '''
    ##         L = batch_size
    ##         z_mean_s    = K.repeat(z_mean, L) #=> sample x L x z_dim
    ##         z_log_var_s = K.repeat(z_log_var, L)

    ##         eps  = K.random_normal(shape=K.shape(z_mean_s), mean=0., stddev=1.0)
    ##         z_sample = z_mean_s + K.exp((z_log_var_s)/2.0) * eps #=> sample x L x z_dim
    ##         x_sample = K.map_fn(generator, z_sample) # batch x batch x ? x dim
    ##         #x_sample = K.map_fn(generator, K.reshape(z_sample, shape=(-1,z_dim)))
    ##         #=> sample x L x (length x x_dim)?
    ##         #x_sample = K.reshape(x_sample, shape=(batch_size, L, length, input_dim))

    ##         x_mean = K.mean(x_sample, axis=1)
    ##         x_std  = K.std(x_sample, axis=1)
            
    ##         log_p_x_z = -0.5 * ( K.sum(K.square((x_true-x_mean)/x_std), axis=-1) \
    ##                              + float(input_dim) * K.log(2.0*np.pi) + K.sum(K.log(K.square(x_std)),
    ##                                                                            axis=-1) )
    ##         #xent_loss += -log_p_x_z/float(batch_size)
    ##         xent_loss = K.mean(-log_p_x_z, axis=-1)

    ##         sig2 = z_std
    ##         kl_loss = - 0.5 * K.sum(1 + z_log_var -K.log(sig2*sig2) - K.square(z_mean-p)
    ##                                 - K.exp(z_log_var)/(sig2*sig2), axis=-1)
    ##         ## kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    ##         return K.mean(xent_loss + kl_loss) 

    ##     def call(self, args):
            
    ##         x_true = args[0][:,:,:input_dim]
    ##         p      = args[0][:,:,input_dim:]
    ##         x_pred = args[1][:,:,:input_dim]

    ##         loss = self.vae_loss(x_true, p)                
    ##         self.add_loss( loss, inputs=args )
    ##         return x_pred

