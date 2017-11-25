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
np.random.seed(1337)

# Keras
import h5py
import keras
from keras.models import Sequential, Model
from keras.layers import Input, TimeDistributed, Layer
from keras.layers import Activation, Dropout, Flatten, Dense, Lambda, RepeatVector, LSTM, GaussianNoise
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD, Adagrad, Adadelta, RMSprop, Adam
from keras import backend as K
from keras import objectives

from hrl_anomaly_detection.RAL18_detection import keras_util as ku
from distutils.version import LooseVersion, StrictVersion
import gc


def lstm_ae(trainData, testData, weights_file=None, batch_size=1024, nb_epoch=500, sam_epoch=40,\
            patience=20, timesteps=4,\
            fine_tuning=False, save_weights_file=None, noise_mag=0.0, renew=False, **kwargs):
    """
    Autoencoder with one LSTM and one fully-connected layer
    x_train is (sample x length x dim)
    x_test is (sample x length x dim)
    """
    x_train = trainData[0]
    y_train = trainData[1]
    x_test = testData[0]
    y_test = testData[1]

    length = len(x_train[0])
    input_dim = len(x_train[0][0])
    z_dim  = kwargs.get('z_dim', 2) 
    
    inputs = Input(batch_shape=(batch_size, timesteps, input_dim))
    encoded = GaussianNoise(noise_mag)(inputs)
    encoded = LSTM(z_dim, return_sequences=False, activation='tanh', stateful=True)(encoded)
        
    # we initiate these layers to reuse later.
    decoded_H1 = RepeatVector(timesteps)
    decoded_L1 = LSTM(input_dim, return_sequences=True, activation='tanh', stateful=True)
    decoded_mu = TimeDistributed(Dense(input_dim, activation='linear'))
    decoded = decoded_H1(encoded)
    decoded = decoded_L1(decoded)
    decoded = decoded_mu(decoded)
    
    ae = Model(inputs, decoded)
    print(ae.summary())

    # Encoder --------------------------------------------------
    encoder_mean = Model(inputs, encoded)



    if weights_file is not None and os.path.isfile(weights_file) and fine_tuning is False and \
      renew is False:
        ae.load_weights(weights_file)
    else:
        if fine_tuning: ae.load_weights(weights_file)
        ae.compile(loss='mean_squared_error', optimizer='adam')

        # ---------------------------------------------------------------------------------
        nDim         = len(x_train[0][0])
        wait         = 0
        plateau_wait = 0
        min_loss = 1e+15
        np.random.seed(3334)
        for epoch in xrange(nb_epoch):
            print 

            mean_tr_loss = []
            for sample in xrange(sam_epoch):

                # shuffle
                idx_list = range(len(x_train))
                np.random.shuffle(idx_list)
                x_train = x_train[idx_list]
                
                for i in xrange(0,len(x_train),batch_size):
                    seq_tr_loss = []

                    if i+batch_size > len(x_train):
                        r = (i+batch_size-len(x_train))%len(x_train)
                        idx_list = range(len(x_train))
                        random.shuffle(idx_list)
                        x = np.vstack([x_train[i:],
                                       x_train[idx_list[:r]]])
                        while True:
                            if len(x)<batch_size: x = np.vstack([x, x_train])
                            else:                 break
                    else:
                        x = x_train[i:i+batch_size]


                    for j in xrange(len(x[0])-timesteps+1): # per window
                        tr_loss = ae.train_on_batch(x[:,j:j+timesteps], x[:,j:j+timesteps] )
                        seq_tr_loss.append(tr_loss)
                    mean_tr_loss.append( np.mean(seq_tr_loss) )
                    ae.reset_states()

                sys.stdout.write('Epoch {} / {} : loss training = {} , loss validating = {}\r'.format(epoch, nb_epoch, np.mean(mean_tr_loss), 0))
                sys.stdout.flush()   

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
                        x = np.vstack([x, x_test[idx_list[:r%len(x_test)]]])
                else:
                    x = x_test[i:i+batch_size]
                
                for j in xrange(len(x[0])-timesteps+1):
                    te_loss = ae.test_on_batch(x[:,j:j+timesteps], x[:,j:j+timesteps] )
                    seq_te_loss.append(te_loss)
                mean_te_loss.append( np.mean(seq_te_loss) )
                ae.reset_states()

            val_loss = np.mean(mean_te_loss)
            sys.stdout.write('Epoch {} / {} : loss training = {} , loss validating = {}\r'.format(epoch, nb_epoch, np.mean(mean_tr_loss), val_loss))
            sys.stdout.flush()   


            # Early Stopping
            if val_loss <= min_loss:
                min_loss = val_loss
                wait         = 0
                plateau_wait = 0

                if save_weights_file is not None:
                    ae.save_weights(save_weights_file)
                else:
                    ae.save_weights(weights_file)
                
            else:
                if wait > patience:
                    print "Over patience!"
                    break
                else:
                    wait += 1
                    plateau_wait += 1

            #ReduceLROnPlateau
            if plateau_wait > 2:
                old_lr = float(K.get_value(ae.optimizer.lr))
                new_lr = old_lr * 0.2
                K.set_value(ae.optimizer.lr, new_lr)
                plateau_wait = 0
                print 'Reduced learning rate {} to {}'.format(old_lr, new_lr)

        gc.collect()
        
        return ae, encoder_mean


def predict(x_test, ae, nDim, batch_size, timesteps=1 ):
    '''
    x_test: 1 x timestep x dim
    '''

    ae.reset_states()

    x = x_test
    for j in xrange(batch_size-1):
        x = np.vstack([x,x_test])
    
    x_pred_mean = []
    for j in xrange(len(x[0])-timesteps+1):
        x_pred = ae.predict(x[:,j:j+timesteps], batch_size=batch_size)
        x_pred_mean.append(x_pred[0,-1])

    return x_pred_mean


    
