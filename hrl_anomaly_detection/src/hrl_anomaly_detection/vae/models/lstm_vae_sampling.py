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

import gc

def lstm_vae(trainData, testData, weights_file=None, batch_size=1024, nb_epoch=500, \
             patience=20, fine_tuning=False, save_weights_file=None):
    """
    Variational Autoencoder with two LSTMs and one fully-connected layer
    x_train is (sample x length x dim)
    x_test is (sample x length x dim)
    """
    
    x_train = trainData[0]
    y_train = trainData[1]
    x_test = testData[0]
    y_test = testData[1]

    timesteps = len(x_train[0])
    input_dim = len(x_train[0][0])

    h1_dim = input_dim
    h2_dim = 2 #input_dim
    z_dim  = 2
    L      = 50

    inputs = Input(shape=(timesteps, input_dim))
    encoded = LSTM(h1_dim, return_sequences=True, activation='tanh')(inputs)
    encoded = LSTM(h2_dim, return_sequences=False, activation='tanh')(encoded)
    z_mean  = Dense(z_dim)(encoded) #, activation='tanh'
    z_log_var = Dense(z_dim)(encoded) #, activation='sigmoid')
    
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(z_dim,), mean=0., stddev=1.0)
        return z_mean + K.exp(z_log_var/2.0) * epsilon    
        
    # we initiate these layers to reuse later.
    decoded_h1 = Dense(h2_dim, name='h_1') #, activation='tanh'
    decoded_h2 = RepeatVector(timesteps, name='h_2')
    decoded_L1 = LSTM(h1_dim, return_sequences=True, activation='tanh', name='L_1')
    decoded_L2 = LSTM(input_dim, return_sequences=True, activation='sigmoid', name='L_2')

    z = Lambda(sampling)([z_mean, z_log_var])    
    decoded = decoded_h1(z)
    decoded = decoded_h2(decoded)
    decoded = decoded_L1(decoded)
    decoded_mean = decoded_L2(decoded)

    vae_autoencoder = Model(inputs, decoded_mean)
    print(vae_autoencoder.summary())

    # Encoder --------------------------------------------------
    vae_encoder_mean = Model(inputs, z_mean)
    vae_encoder_var  = Model(inputs, z_log_var)

    # Decoder (generator) --------------------------------------
    decoder_input = Input(shape=(z_dim,))
    _decoded_H1 = decoded_h1(decoder_input)
    _decoded_H2 = decoded_h2(_decoded_H1)
    _decoded_L1 = decoded_L1(_decoded_H2)
    _decoded_L2 = decoded_L2(_decoded_L1)
    generator = Model(decoder_input, _decoded_L2)


    def loglikelihood(x, loc=0, scale=1):
        '''
        It uses diagonal co-variance elements only.
        '''
        return -0.5 * ( K.sum(K.square((x-loc))/(scale+1e-10), axis=-1) \
          + float(input_dim) * K.log(2.0*np.pi) + K.sum(K.log(scale+1e-10), axis=-1) )

    ## def vae_loss(inputs, x_decoded_mean):
    ##     xent_loss = K.mean(objectives.binary_crossentropy(inputs, x_decoded_mean), axis=-1)
    ##     kl_loss   = -0.5 * K.mean(1.0 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1) 
    ##     ## return xent_loss + kl_loss
    ##     return K.mean(xent_loss + kl_loss)

    def vae_loss(y_true, y_pred):

        z_mean_s    = K.repeat(z_mean, L)
        z_log_var_s = K.repeat(z_log_var, L)

        # Case 1: following sampling function
        epsilon  = K.random_normal(shape=K.shape(z_mean_s), mean=0., stddev=1.0)
        z_sample = z_mean_s + K.exp((z_log_var_s)/2.0) * epsilon

        # Case 2: using raw
        #?
        
        x_sample = K.map_fn(generator, z_sample)

        x_mean = K.mean(x_sample, axis=1) # length x dim
        x_var  = K.var(x_sample, axis=1)

        log_p_x_z = loglikelihood(y_true, loc=x_mean, scale=x_var)
        xent_loss = K.mean(-log_p_x_z, axis=-1)
        #xent_loss = K.sum(-log_p_x_z, axis=-1)
        
        kl_loss   = -0.5 * K.mean(1.0 + z_log_var - K.square(z_mean) \
                                  - K.exp(z_log_var), axis=-1)

                                  
        #loss = xent_loss + kl_loss
        loss = K.mean(xent_loss + kl_loss)
        ## K.print_tensor(xent_loss)
        return loss




    if weights_file is not None and os.path.isfile(weights_file) and fine_tuning is False:
        vae_autoencoder.load_weights(weights_file)
        return vae_autoencoder, vae_encoder_mean, vae_encoder_var, generator
    else:
        ## vae_autoencoder.load_weights(weights_file)
        if fine_tuning:
            vae_autoencoder.load_weights(weights_file)
            lr = 0.001
        else:
            lr = 0.01
        optimizer = RMSprop(lr=lr, rho=0.9, epsilon=1e-08, decay=0.0001)
        #optimizer = Adam(lr=lr)                
        vae_autoencoder.compile(optimizer=optimizer, loss=vae_loss)

        ## vae_autoencoder.load_weights(weights_file)
        from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
        callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=patience,
                                   verbose=0, mode='auto'),
                    ModelCheckpoint(weights_file,
                                    save_best_only=True,
                                    save_weights_only=True,
                                    monitor='val_loss'),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                      patience=3, min_lr=0.0001)]

        train_datagen = ku.sigGenerator(augmentation=True, noise_mag=0.03)
        train_generator = train_datagen.flow(x_train, x_train, batch_size=batch_size, seed=3334)
        ## test_datagen = ku.sigGenerator(augmentation=False)
        ## test_generator = test_datagen.flow(x_test, x_test, batch_size=len(x_test),
        ##                                    shuffle=False)

        hist = vae_autoencoder.fit_generator(train_generator,
                                             steps_per_epoch=512,
                                             epochs=nb_epoch,
                                             validation_data=(x_test, x_test),
                                             ## validation_data=test_generator,
                                             ## validation_steps=1,
                                             callbacks=callbacks)
        if save_weights_file is not None:
            vae_autoencoder.save_weights(save_weights_file)
        else:
            vae_autoencoder.save_weights(weights_file)

        gc.collect()
        
        return vae_autoencoder, vae_encoder_mean, vae_encoder_var, generator
