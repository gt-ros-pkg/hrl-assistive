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
from keras.layers.wrappers import Bidirectional
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD, Adagrad, Adadelta, RMSprop, Adam
from keras import backend as K
from keras import objectives

from hrl_anomaly_detection.RAL18_detection import keras_util as ku
from hrl_anomaly_detection.RAL18_detection import util as vutil

import gc

def lstm_ae(trainData, testData, weights_file=None, batch_size=1024, nb_epoch=500,
            patience=20, fine_tuning=False, save_weights_file=None,
            noise_mag=0.0, timesteps=4, sam_epoch=1,
            renew=False, plot=True, **kwargs):
    """
    x_train is (sample x length x dim)
    x_test is (sample x length x dim)
    """
    x_train = trainData[0]
    y_train = trainData[1]
    x_test = testData[0]
    y_test = testData[1]

    input_dim = len(x_train[0][0])
    length    = len(x_train[0])
    x_train, y_train = create_dataset(x_train, timesteps, 0)
    x_test, y_test   = create_dataset(x_test, timesteps, 0)
    
    x_train = x_train.reshape((-1, timesteps, input_dim))
    x_test  = x_test.reshape((-1, timesteps, input_dim))

    ## inputs  = Input(shape=(timesteps, input_dim))
    ## outputs = Bidirectional( LSTM(input_dim, activation='sigmoid', return_sequences=True) )(inputs)    
    ## encoded = LSTM(z_dim, return_sequences=False, activation='tanh')(inputs)
    ## #decoded_H2 = RepeatVector(timesteps)    
    ## decoded = LSTM(input_dim, return_sequences=True, go_backwards=True, activation='sigmoid',
    ##                stateful=False)
    ## ae = Model(inputs, outputs)

    import seq2seq
    from seq2seq.models import SimpleSeq2Seq
    ae = SimpleSeq2Seq(input_dim=input_dim, hidden_dim=2, output_length=timesteps, output_dim=input_dim)
    print(ae.summary())

    # Encoder --------------------------------------------------
    ae_encoder = None #Model(inputs, encoded)

    ## def mse_loss(y_true, y_pred):
    ##     #y = y[:,:,:input_dim]
    ##     #print np.shape(y_true), np.shape(y_pred)
    ##     #y_pred = y_pred[:,:,::-1]
    ##     return K.mean(K.mean(K.square(y_true-y_pred),axis=-1), axis=-1)

    # AE --------------------------------------
    if weights_file is not None and os.path.isfile(weights_file) and fine_tuning is False and\
        renew is False:
        ae.load_weights(weights_file)
    else:
        if fine_tuning:
            ae.load_weights(weights_file)
            lr = 0.0001
            optimizer = Adam(lr=lr, clipvalue=10)                
            ae.compile(optimizer=optimizer, loss='mse')
            #ae.compile(optimizer='adam', loss='mse')
        else:
            ae.compile(optimizer='adam', loss='mse') #mse_loss)

        # ---------------------------------------------------------------------------------
        from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
        callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=patience,
                                   verbose=0, mode='auto'),
                    ModelCheckpoint(weights_file,
                                    save_best_only=True,
                                    save_weights_only=True,
                                    monitor='val_loss'),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                      patience=5, min_lr=0.0001)]

        train_datagen = ku.sigGenerator(augmentation=True, noise_mag=noise_mag )
        train_generator = train_datagen.flow(x_train, x_train, batch_size=batch_size)

        hist = ae.fit_generator(train_generator,
                                samples_per_epoch=sam_epoch,
                                epochs=nb_epoch,
                                validation_data=(x_test, x_test),
                                callbacks=callbacks) 

        gc.collect()

    # ---------------------------------------------------------------------------------
    # visualize outputs
    if plot:
        print "variance visualization"
        x_test = testData[0]
        x_test, y_test   = create_dataset(x_test, timesteps, 0)        
        nDim = input_dim
        
        for i in xrange(len(x_test)): #per sample

            x_pred_mean = []
            for j in xrange(len(x_test[i])):
                x_pred = ae.predict( x_test[i,j:j+1] )
                x_pred_mean.append(x_pred[0,0])

            vutil.graph_variations(x_test[i,:,-1], x_pred_mean)

    return ae


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


