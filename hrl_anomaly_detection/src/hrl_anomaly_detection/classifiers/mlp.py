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

# system
import os, sys, copy, time

# util
import numpy as np
import scipy

from hrl_anomaly_detection.classifiers.clf_base import clf_base
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import SGD, Adagrad, Adadelta, RMSprop

from keras.models import Model
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD, Adagrad, Adadelta, RMSprop
import h5py 


class mlp(clf_base):
    def __init__(self, ths=0, encoding_dim=4, patience=10, nb_epoch=1500, verbose=False, **kwargs):
        ''' '''        
        clf_base.__init__(self)
        self.ths          = ths
        self.encoding_dim = encoding_dim
        self.nb_epoch     = nb_epoch

        self.ml = None
        self.callbacks  = [EarlyStopping(monitor='loss', min_delta=0.0001, patience=patience,
                                         verbose=0, mode='auto'),
                           ReduceLROnPlateau(monitor='loss', factor=0.2,
                                             patience=10, min_lr=0.00001)]
                                                                                                            

    def fit(self, X, y=None, ll_idx=None, **kwargs):
        '''
        X: trainig set (sample x feature)
        Xv: validation set (sample x feature)
        '''
        self.ml = mlp_net(np.shape(X)[1:], encoding_dim=self.encoding_dim)
        #optimizer = SGD(lr=0.2, decay=1e-9, momentum=0.9, nesterov=True)                
        optimizer = RMSprop(lr=0.1, rho=0.9, epsilon=1e-08, decay=0.001)                        
        self.ml.compile(optimizer=optimizer, loss='mse') #, metrics=['accuracy'])
        ## self.ml.compile(optimizer='adadelta', loss='mse') #, metrics=['accuracy'])

        if kwargs['Xv'] is not None:
            hist = self.ml.fit(X, X, nb_epoch=self.nb_epoch, batch_size=len(X), shuffle=True,\
                               validation_data=(kwargs['Xv'], kwargs['Xv']), callbacks=self.callbacks)
        else:
            hist = self.ml.fit(X, X, nb_epoch=self.nb_epoch, batch_size=len(X), shuffle=True,\
                               callbacks=self.callbacks)
        
        # temp
        # Need to estimate reconstruction errors on training data to set up ths range
        ## errs = self.predict(X)+self.ths
        ## print np.mean(errs), np.amax(errs)
        ## sys.exit()
        
        return

    def partial_fit(self, X, y=None, **kwargs):

        assert self.ml is not None        
        
        hist = self.ml.fit(X, X, nb_epoch=self.nb_epoch, batch_size=len(X), shuffle=True,
                           callbacks=self.callbacks)       
        return

    def predict(self, X, y=None):
        ''' '''
        decoded_data = self.ml.predict(X)
        errs = np.mean(np.array(X)-np.array(decoded_data)**2, axis=1)

        return errs - self.ths


    def save_model(self, fileName):
        self.ml.save_weights(fileName)
        return

    def load_model(self, fileName):        
        self.ml.load_weights(fileName)
        return

           
def mlp_net(input_shape, encoding_dim):

    ## if activ_type == 'PReLU': activ_type = PReLU(init='zero', weights=None)

    inputs = Input(shape=input_shape)
    encoded  = Dense(encoding_dim*2, activation='sigmoid')(inputs)
    encoded1 = Dense(encoding_dim, activation='sigmoid')(encoded)
    
    decoded1 = Dense(encoding_dim*2, activation='sigmoid')(encoded1)
    decoded  = Dense(input_shape[0], activation='sigmoid')(decoded1)

    
    ## encoded  = Dense(encoding_dim*2, activation='sigmoid')(inputs)
    ## encoded1 = Dense(encoding_dim, activation='sigmoid')(encoded)    
    ## decoded1 = Dense(encoding_dim*2, activation='sigmoid')(encoded1)
    ## decoded  = Dense(input_shape[0], activation='sigmoid')(decoded1)


    ## encoded  = Dense(encoding_dim, activation='sigmoid')(inputs)
    ## decoded  = Dense(input_shape[0], activation='sigmoid')(encoded)

    
    # this model maps an input to its reconstruction
    autoencoder = Model(input=inputs, output=decoded)
    return autoencoder
