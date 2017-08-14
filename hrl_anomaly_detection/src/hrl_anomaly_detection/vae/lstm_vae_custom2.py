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
             x_std_div=1, x_std_offset=0.001, z_std=0.5,\
             renew=False, plot=True, trainable=None, **kwargs):
    """
    Variational Autoencoder with two LSTMs and one fully-connected layer
    x_train is (sample x length x dim)
    x_test is (sample x length x dim)
    """

    x_train = trainData[0]
    y_train = trainData[1]
    x_test = testData[0]
    y_test = testData[1]

    input_dim = len(x_train[0][0])
    length = len(x_train[0])

    h1_dim = input_dim
    z_dim  = 2


    def slicing(x):
        return x[:,:,:input_dim]
           
    inputs = Input(shape=(timesteps, input_dim+1))
    encoded = Lambda(slicing)(inputs)     
    encoded = LSTM(h1_dim, return_sequences=False, activation='tanh', 
                   trainable=True if trainable==0 or trainable is None else False)(encoded)
    z_mean  = Dense(z_dim, trainable=True if trainable==1 or trainable is None else False)(encoded) 
    z_log_var = Dense(z_dim, trainable=True if trainable==1 or trainable is None else False)(encoded) 
    
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=K.shape(z_mean), mean=0., stddev=1.)
        return z_mean + K.exp(z_log_var/2.0) * epsilon    
        
    # we initiate these layers to reuse later.
    decoded_h1 = Dense(h1_dim, trainable=True if trainable==2 or trainable is None else False) 
    decoded_h2 = RepeatVector(timesteps, name='h_2')
    decoded_L21 = LSTM(input_dim*2, return_sequences=True, activation='sigmoid', 
                       trainable=True if trainable==3 or trainable is None else False)

    # Custom loss layer
    class CustomVariationalLayer(Layer):
        def __init__(self, **kwargs):
            self.is_placeholder = True
            super(CustomVariationalLayer, self).__init__(**kwargs)

        def vae_loss(self, x, x_d_mean, x_d_std, p):
            '''
            p : phase variable
            '''
            log_p_x_z = -0.5 * ( K.sum(K.square((x-x_d_mean)/x_d_std), axis=-1) \
                                 + float(input_dim) * K.log(2.0*np.pi) + K.sum(K.log(K.square(x_d_std)),
                                                                               axis=-1) )
            xent_loss = K.mean(-log_p_x_z, axis=-1)

            sig2 = z_std
            kl_loss = - 0.5 * K.sum(1 + z_log_var -K.log(sig2*sig2) - K.square(z_mean-p)
                                    - K.exp(z_log_var)/(sig2*sig2), axis=-1)
            ## kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return K.mean(xent_loss + kl_loss) 

        def call(self, args):
            x = args[0][:,:,:input_dim]
            p = args[0][:,:,input_dim:]
            x_d_mean = args[1][:,:,:input_dim]
            x_d_std  = args[1][:,:,input_dim:]/x_std_div + x_std_offset
            
            loss = self.vae_loss(x, x_d_mean, x_d_std, p)
            self.add_loss(loss, inputs=args)
            # We won't actually use the output.
            return x_d_mean


    z = Lambda(sampling)([z_mean, z_log_var])    
    decoded = decoded_h1(z)
    decoded = decoded_h2(decoded)
    decoded = decoded_L21(decoded)
    outputs = CustomVariationalLayer()([inputs, decoded])

    vae_autoencoder = Model(inputs, outputs)
    print(vae_autoencoder.summary())

    # Encoder --------------------------------------------------
    vae_encoder_mean = Model(inputs, z_mean)
    vae_encoder_var  = Model(inputs, z_log_var)
    generator = None

    # VAE --------------------------------------
    vae_mean_std = Model(inputs, decoded)

    if weights_file is not None and os.path.isfile(weights_file) and fine_tuning is False and\
        renew is False:
        vae_autoencoder.load_weights(weights_file)
    else:
        if fine_tuning:
            vae_autoencoder.load_weights(weights_file)
            lr = 0.01            
            optimizer = Adam(lr=lr, clipvalue=4.)# 5)                
            vae_autoencoder.compile(optimizer=optimizer, loss=None)
            #vae_autoencoder.compile(optimizer='rmsprop', loss=None)
        else:
            lr = 0.01
            #optimizer = RMSprop(lr=lr, rho=0.9, epsilon=1e-08, decay=0.0001, clipvalue=10)
            optimizer = Adam(lr=lr, clipvalue=10) #, decay=1e-5)                
            #vae_autoencoder.compile(optimizer=optimizer, loss=None)
            vae_autoencoder.compile(optimizer='adam', loss=None)

        # ---------------------------------------------------------------------------------
        nDim         = len(x_train[0][0])

        from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
        callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=patience,
                                   verbose=0, mode='auto'),
                    ModelCheckpoint(weights_file,
                                    save_best_only=True,
                                    save_weights_only=True,
                                    monitor='val_loss'),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                      patience=2, min_lr=0.0)]


        # add phase
        p = np.arange(0.,float(length),1.)*2.0 - 1.0
        p = [p.reshape((-1,1)).tolist() for k in range(len(x_train))]
        x_tr = np.concatenate((x_train, p), axis=-1)
        
        p = np.arange(0.,float(length),1.)*2.0 - 1.0
        p = [p.reshape((-1,1)).tolist() for k in range(len(x_test))] 
        x_te = np.concatenate((x_test, p), axis=-1)
        
        # data conversion
        # => sample x length x window x dim
        x_tr, _ = create_dataset(x_tr, timesteps)
        x_te, _ = create_dataset(x_te, timesteps)
        ## sys.exit()
        x_tr = x_tr.reshape((-1,timesteps,nDim+1))
        x_te = x_te.reshape((-1,timesteps,nDim+1))

        train_datagen = sigGenerator(augmentation=True, noise_mag=noise_mag, with_phase=True)
        train_generator = train_datagen.flow(x_tr, x_tr, batch_size=batch_size, seed=3334,
                                             shuffle=False)

        test_datagen = sigGenerator(augmentation=False, with_phase=True)
        test_generator = train_datagen.flow(x_te, x_te, batch_size=batch_size, seed=3334,
                                             shuffle=False)
                        
        hist = vae_autoencoder.fit_generator(train_generator,
                                             steps_per_epoch=sam_epoch,
                                             epochs=nb_epoch,
                                             validation_data=test_generator,
                                             validation_steps=1,
                                             callbacks=callbacks)

        gc.collect()

    # ---------------------------------------------------------------------------------
    # visualize outputs
    if False:
        print "latent variable visualization"
        

    if plot:
        print "variance visualization"
        nDim = len(x_test[0,0]) 
        
        for i in xrange(len(x_test)):
            if i!=6: continue #for data viz lstm_vae_custom -4 

            x = x_test[i:i+1]
            #for j in xrange(batch_size-1):
            #    x = np.vstack([x,x_test[i:i+1]])

            vae_autoencoder.reset_states()
            vae_mean_std.reset_states()
            
            x_pred_mean = []
            x_pred_std  = []
            for j in xrange(len(x[0])-timesteps+1):
                x_pred = vae_mean_std.predict(np.concatenate((x[:,j:j+timesteps],
                                                              np.zeros((len(x), timesteps,1))
                                                              ), axis=-1))
                                   
                x_pred_mean.append(x_pred[0,-1,:nDim])
                x_pred_std.append(x_pred[0,-1,nDim:]/x_std_div*1.5+x_std_offset)

            vutil.graph_variations(x_test[i], x_pred_mean, x_pred_std, scaler_dict=kwargs['scaler_dict'])
        


    return vae_autoencoder, vae_mean_std, vae_mean_std, vae_encoder_mean, vae_encoder_var, generator


def create_dataset(dataset, window_size=1, step=0):
    '''
    dataset: sample x timesteps x dim
    '''
    
    dataX, dataY = [], []
    for i in xrange(len(dataset)):
        x = []
        y = []
        for j in range(len(dataset[i])-step-window_size):
            x.append(dataset[i,j:(j+window_size), :].tolist())
            y.append(dataset[i,j+step:(j+step+window_size), :].tolist())
        dataX.append(x)
        dataY.append(y)
    return numpy.array(dataX), numpy.array(dataY)



class sigGenerator():
    ''' signal data augmentation
    Signals should be normalized before.
    '''
    
    def __init__(self, augmentation=False, noise_mag=0.03, with_phase=True):

        self.augmentation  = augmentation
        self.noise_mag     = noise_mag
        self.with_phase     = with_phase
        self.total_batches_seen = 0
        self.batch_index = 0

    def reset(self):
        self.batch_index = 0

    def flow(self, x, y, batch_size=32, shuffle=True, seed=None):

        assert len(x) == len(y), "data should have the same length"
        
        if type(x) is not np.ndarray: x = np.array(x)
        if type(y) is not np.ndarray: y = np.array(y)

        # Ensure self.batch_index is 0.
        self.reset()
        x_new = x[:]

        # TODO: Need to add random selection
        if len(x_new) % batch_size > 0:
            n_add = batch_size - len(x_new) % batch_size
            x_new = np.vstack([x_new, x_new[:n_add] ])

        n = len(x_new)
        idx_list = range(n)
        n_dim  = len(x_new[0,0])
        timesteps = len(x_new[0])

        while 1:

            if seed is not None:
                np.random.seed(seed + self.total_batches_seen)

            if self.batch_index == 0:
                idx_list = range(n)
                if shuffle:
                    random.shuffle(idx_list)
                x_new = x_new[idx_list]

            current_index = (self.batch_index * batch_size) % n
            if n > current_index + batch_size:
                current_batch_size = batch_size
                self.batch_index += 1
            else:
                current_batch_size = n - current_index
                self.batch_index = 0

            ## print " aa ", batch_size, " aa ", self.total_batches_seen, self.batch_index, current_index,"/",n
            
            self.total_batches_seen += 1
            if self.augmentation:
                # noise
                if self.with_phase:
                    noise = np.random.normal(0.0, self.noise_mag, \
                                             np.shape(x_new[current_index:current_index+current_batch_size,
                                                            :,:n_dim-1]))
                    noise = np.concatenate((noise, np.zeros((current_batch_size,timesteps,1))),axis=-1)
                else:
                    noise = np.random.normal(0.0, self.noise_mag, \
                                             np.shape(x_new[current_index:current_index+current_batch_size])) 

                yield x_new[current_index:current_index+current_batch_size]+noise,\
                  x_new[current_index:current_index+current_batch_size]+noise
                                          
            else:
                yield x_new[current_index:current_index+current_batch_size],\
                  x_new[current_index:current_index+current_batch_size]


                
