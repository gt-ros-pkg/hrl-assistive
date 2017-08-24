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

def lstm_ae(trainData, testData, weights_file=None, batch_size=1024, nb_epoch=500, patience=20,
             fine_tuning=False, save_weights_file=None):
    """
    Variational Autoencoder with two LSTMs and one fully-connected layer
    x_train is (sample x length x dim)
    x_test is (sample x length x dim)

    Note: it uses offline data.
    This code based on "https://gist.github.com/tushuhei/e684c2a5382c324880532aded9faf4e6"
    """
    x_train = trainData[0]
    y_train = trainData[1]
    x_test = testData[0]
    y_test = testData[1]

    timesteps = len(x_train[0])
    input_dim = len(x_train[0][0])

    h1_dim = input_dim
    z_dim  = 2
    
    inputs = Input(shape=(timesteps, input_dim))
    encoded = LSTM(h1_dim, return_sequences=True, activation='tanh')(inputs)
    encoded = LSTM(z_dim, return_sequences=False, activation='tanh')(encoded)
        
    # we initiate these layers to reuse later.
    decoded_H2 = RepeatVector(timesteps, name='H_2')
    decoded_L1 = LSTM(h1_dim, return_sequences=True, activation='tanh', name='L_1')
    decoded_L2 = LSTM(input_dim, return_sequences=True, activation='tanh', name='L_2')

    decoded = decoded_H2(encoded)
    decoded = decoded_L1(decoded)
    decoded = decoded_L2(decoded)
    
    ae = Model(inputs, decoded)
    print(ae.summary())

    if weights_file is not None and os.path.isfile(weights_file) and fine_tuning is False:
        ae.load_weights(weights_file)
        #generator.load_weights(weights_file, by_name=True)
        return ae
    else:
        if fine_tuning:
            ae.load_weights(weights_file)
            lr = 0.001
        else:
            lr = 0.1
        ## optimizer = RMSprop(lr=lr, rho=0.9, epsilon=1e-08, decay=0.0001)
        ## #optimizer = Adam(lr=lr)                
        ## ae.compile(optimizer=optimizer, loss='mse')
        ae.compile(loss='mean_squared_error', optimizer='adam')
        #ae.compile(loss=vae_loss, optimizer='adam')
            
        ## vae_autoencoder.load_weights(weights_file)
        from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
        callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=patience,
                                   verbose=0, mode='auto'),
                    ModelCheckpoint(weights_file,
                                    save_best_only=True,
                                    save_weights_only=True,
                                    monitor='val_loss'),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                      patience=5, min_lr=0.0001)]

        train_datagen = ku.sigGenerator(augmentation=True, noise_mag=0.05 )
        #test_datagen = ku.sigGenerator(augmentation=False)
        train_generator = train_datagen.flow(x_train, x_train, batch_size=batch_size)
        #test_generator = test_datagen.flow(x_test, x_test, batch_size=batch_size, shuffle=False)

        hist = ae.fit_generator(train_generator,
                                samples_per_epoch=512,
                                epochs=nb_epoch,
                                validation_data=(x_test, x_test),
                                #validation_data=test_generator,
                                #nb_val_samples=len(x_test),
                                callbacks=callbacks) #, class_weight=class_weight)

        ## ae.fit(x_train, x_train,
        ##        shuffle=True,
        ##        epochs=nb_epoch,
        ##        batch_size=batch_size,
        ##        callbacks=callbacks,
        ##        validation_data=(x_test, x_test))
        ## if save_weights_file is not None:
        ##     ae.save_weights(save_weights_file)
        ## else:
        ##     ae.save_weights(weights_file)

        gc.collect()
        
        return ae


if __name__ == '__main__':

    print "N/A"
    


    
