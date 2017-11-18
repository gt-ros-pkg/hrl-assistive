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
from keras.layers import Input, TimeDistributed, Layer
from keras.layers import Activation, Dropout, Flatten, Dense, merge, Lambda, GaussianNoise
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD, Adagrad, Adadelta, RMSprop
from keras import backend as K
from keras import objectives

from hrl_execution_monitor.keras_util import keras_util as ku
## from hrl_anomaly_detection.vae import util as vutil

import gc



def sig_net(trainData, testData, batch_size=512, nb_epoch=500, \
            patience=20, fine_tuning=False, noise_mag=0.0,\
            weights_file=None, save_weights_file='./sig_weights.h5', renew=False, **kwargs):
    """
    Variational Autoencoder with two LSTMs and one fully-connected layer
    x_train is (sample x dim)
    x_test is (sample x dim)

    y_train should contain all labels
    """

    x_train = trainData[0]
    y_train = trainData[1]
    x_test = testData[0]
    y_test = testData[1]

    n_labels = len(np.unique(y_train))
    print "Labels: ", np.unique(y_train)
    print "#Labels: ", n_labels    

    # Model construction
    model = Sequential()
    model.add(Dense(128, init='uniform', input_shape=np.shape(x_train)[1:],
                    W_regularizer=L1L2Regularizer(0,0),\
                    name='fc_sig_1'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(n_labels, activation='softmax',
                    W_regularizer=L1L2Regularizer(0,0),
                    name='fc_sig_out'))    
    print(model.summary())


    if weights_file is not None and os.path.isfile(weights_file) and fine_tuning is False and\
      renew is False:
        model.load_weights(weights_file)
    else:

        callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=patience,
                                   verbose=0, mode='auto'),
                     ModelCheckpoint(save_weights_file,
                                     save_best_only=True,
                                     save_weights_only=True,
                                     monitor='val_loss'),
                     ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                       patience=patience/2, min_lr=0.00001)]

        if fine_tuning:
            model.load_weights(weights_file)
            optimizer = SGD(lr=0.0001, decay=1e-7, momentum=0.9, nesterov=True)
            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        else:
            if os.path.isfile(weights_file): model.load_weights(weights_file)
            model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


        train_datagen = ku.sigGenerator(augmentation=True, noise_mag=noise_mag )
        test_datagen = ku.sigGenerator(augmentation=False)
        train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)
        test_generator = test_datagen.flow(x_test, y_test, batch_size=batch_size)

        hist = model.fit_generator(train_generator,
                                   samples_per_epoch=len(y_train),
                                   nb_epoch=nb_epoch,
                                   validation_data=test_generator,
                                   nb_val_samples=len(y_test),
                                   callbacks=callbacks)

        scores.append( hist.history['val_acc'][-1] )
        gc.collect()


        print "score : ", scores
        print 
        print np.mean(scores), np.std(scores)

    return model


## class ResetStatesCallback(Callback):
##     def __init__(self, max_len):
##         self.counter = 0
##         self.max_len = max_len
        
##     def on_batch_begin(self, batch, logs={}):
##         if self.counter % self.max_len == 0:
##             self.model.reset_states()
##         self.counter += 1

