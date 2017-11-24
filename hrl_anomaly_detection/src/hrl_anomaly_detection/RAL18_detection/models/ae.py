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

from hrl_anomaly_detection.RAL18_detection import util as vutil
from hrl_anomaly_detection.RAL18_detection import keras_util as ku



def autoencoder(trainData, testData, weights_file=None, batch_size=32, nb_epoch=500, \
                patience=20, fine_tuning=False, save_weights_file=None, \
                noise_mag=0.0, timesteps=4, sam_epoch=1, \
                renew=False, plot=True, trainable=None, **kwargs):

    x_train = trainData[0]
    x_test = testData[0]

    nSample = len(x_train)
    input_dim = len(x_train[0][0])
    length = len(x_train[0])

    x_train, y_train = create_dataset(x_train, timesteps, 0)
    x_test, y_test   = create_dataset(x_test, timesteps, 0)

    x_train = x_train.reshape((-1, input_dim*timesteps))
    x_test  = x_test.reshape((-1, input_dim*timesteps))
   
    h1_dim = kwargs.get('h1_dim', input_dim)
    z_dim  = kwargs.get('z_dim', 2)
    
    x = Input(shape=(input_dim*timesteps,))
    h = Dense(h1_dim, activation='tanh')(x)
    z = Dense(z_dim, activation='tanh')(h)
    h = Dense(h1_dim, activation='tanh')(z)
    x_decoded = Dense(input_dim*timesteps, activation='sigmoid')(h)

    ae = Model(x, x_decoded)

    # Encoder and Decoder -------------------------------------------
    ae_encoder = Model(x, z)
    ae_decoder = None #Model(z, x_decoded)

    #optimizer = SGD(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.001)                        
    #ae.compile(optimizer=optimizer, loss=vae_loss)
    ae.compile(optimizer='rmsprop', loss='mse')

    x_train = x_train.astype('float32') 
    x_test = x_test.astype('float32') 

    print np.shape(x_train), np.shape(x_test)

    if weights_file is not None and os.path.isfile(weights_file):
        ae.load_weights(weights_file)
    else:
        ae.fit(x_train, x_train,
               shuffle=True,
               epochs=nb_epoch,
               batch_size=batch_size,
               validation_data=(x_test, x_test))
        ae.save_weights(weights_file)

    # ---------------------------------------------------------------------------------
    # visualize outputs
    if plot:
        x_test = testData[0]
        x_test, y_test   = create_dataset(x_test, timesteps, 0)
        #x_test = x_test.reshape((nSample, -1, timesteps, input_dim))
        
        print "variance visualization"
        nDim = input_dim
        
        for i in xrange(len(x_test)):
            #if i!=6: continue #for data viz lstm_vae_custom -4 

            x = x_test[i]

            x_pred_mean = []
            for j in xrange(len(x)):
                x_pred = ae.predict(np.expand_dims(x[j].flatten(), axis=0) )
                x_pred_mean.append(x_pred.reshape((timesteps, input_dim))[-1])
            vutil.graph_variations(x_test[i,:,-1], x_pred_mean) #, x_pred_std) #, scaler_dict=kwargs['scaler_dict'])
        

    return ae, ae_encoder, ae_decoder
    

def create_dataset(dataset, window_size=5, step=5):
    '''
    Input: dataset= sample x timesteps x dim
    Output: dataX = sample x timesteps? x window x dim
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


