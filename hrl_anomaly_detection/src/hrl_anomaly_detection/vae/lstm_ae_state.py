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
            fine_tuning=False, save_weights_file=None,
            noise_mag=0.0, sam_epoch=1, re_load=False, plot=True):
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

    input_dim = len(x_train[0][0])

    h1_dim = input_dim
    z_dim  = 2
    timesteps = 1
    
    inputs = Input(batch_shape=(1, timesteps, input_dim))
    encoded = LSTM(h1_dim, return_sequences=True, activation='tanh', stateful=True)(inputs)
    encoded = LSTM(z_dim, return_sequences=False, activation='tanh', stateful=True)(encoded)
        
    # we initiate these layers to reuse later.
    decoded_H2 = RepeatVector(timesteps, name='H_2')
    decoded_L1 = LSTM(h1_dim, return_sequences=True, activation='tanh', stateful=True, name='L_1')
    decoded_L2 = LSTM(input_dim, return_sequences=True, activation='sigmoid', stateful=True, name='L_2')

    decoded = decoded_H2(encoded)
    decoded = decoded_L1(decoded)
    decoded = decoded_L2(decoded)
    
    ae = Model(inputs, decoded)
    print(ae.summary())

    if weights_file is not None and os.path.isfile(weights_file) and fine_tuning is False:
        ae.load_weights(weights_file)
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

        # ---------------------------------------------------------------------------------
        nDim         = len(x_train[0][0])
        wait         = 0
        plateau_wait = 0
        min_loss     = 1e+15
        for epoch in xrange(nb_epoch):
            print('Epoch', epoch, '/', nb_epoch),

            mean_tr_loss = []
            for sample in xrange(sam_epoch):
                for i in xrange(len(x_train)):
                    seq_tr_loss = []
                    for j in xrange(len(x_train[i])):
                        np.random.seed(3334 + i*len(x_train[i]) + j)
                        noise = np.random.normal(0, noise_mag, (nDim,))

                        tr_loss = ae.train_on_batch(
                            np.expand_dims(np.expand_dims(x_train[i,j]+noise, axis=0), axis=0),
                            np.expand_dims(np.expand_dims(x_train[i,j]+noise, axis=0), axis=0))
                        seq_tr_loss.append(tr_loss)
                    mean_tr_loss.append( np.mean(seq_tr_loss) )
                    ae.reset_states()

            mean_te_loss = []
            for i in xrange(len(x_test)):
                seq_te_loss = []
                for j in xrange(len(x_test[i])):
                    te_loss = ae.test_on_batch(
                        np.expand_dims(np.expand_dims(x_test[i,j], axis=0), axis=0),
                        np.expand_dims(np.expand_dims(x_test[i,j], axis=0), axis=0))
                    seq_te_loss.append(te_loss)
                mean_te_loss.append( np.mean(seq_te_loss) )
                ae.reset_states()

            val_loss = np.mean(mean_te_loss)
            print('loss training = {} , loss validating = {}'.format(np.mean(mean_tr_loss), val_loss))
            print('___________________________________')


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

    # ---------------------------------------------------------------------------------
    # visualize outputs
    if False:
        print "latent variable visualization"

    if plot:
        print "variance visualization"
        nDim = len(x_test[0,0])
        
        for i in xrange(len(x_test)):

            x_pred_mean = []
            for j in xrange(len(x_test[i])):
                x_pred = ae.predict(x_test[i:i+1,j:j+1])
                x_pred_mean.append(x_pred[0,0,:nDim])

            vutil.graph_variations(x_test[i], x_pred_mean)
        
    return ae


if __name__ == '__main__':

    print "N/A"
    


    
