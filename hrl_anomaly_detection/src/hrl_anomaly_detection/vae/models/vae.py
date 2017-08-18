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


def variational_autoencoder(x_train, x_test, weights_file=None, batch_size=140):
    # This code is based on the Keras example.
    
    # train the VAE on MNIST digits
    ## (x_train, y_train), (x_test, y_test) = mnist.load_data()

    original_dim = len(x_train[0])
    latent_dim = 2
    intermediate_dim = len(x_train[0])/2
    nb_epoch = 50
    epsilon_std = 1.0

    print "Dimension info: ", len(x_train[0]), len(x_train[0])/2, 2
    print "Size: ", len(x_train), " batch_size: ", batch_size

    x = Input(batch_shape=(batch_size, original_dim))
    h = Dense(intermediate_dim, activation='relu')(x)
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)


    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                                  stddev=epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    # we instantiate these layers separately so as to reuse them later
    decoder_h    = Dense(intermediate_dim, activation='relu')
    decoder_mean = Dense(original_dim, activation='sigmoid')
    h_decoded    = decoder_h(z)
    x_decoded_mean = decoder_mean(h_decoded)


    def vae_loss(x, x_decoded_mean):
        xent_loss = original_dim * objectives.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return xent_loss + kl_loss

    vae = Model(x, x_decoded_mean)
    optimizer = SGD(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.001)                        
    vae.compile(optimizer=optimizer, loss=vae_loss)
    #vae.compile(optimizer='rmsprop', loss=vae_loss)


    x_train = x_train.astype('float32') 
    x_test = x_test.astype('float32') 

    if weights_file is not None and os.path.isfile(weights_file):
        vae.load_weights(weights_file)
    else:
        vae.fit(x_train, x_train,
                shuffle=True,
                epochs=nb_epoch,
                batch_size=batch_size,
                validation_data=(x_test, x_test))
        vae.save_weights(weights_file)

    # build a model to project inputs on the latent space
    ## encoder = Model(x, z_mean)

    return vae
    
