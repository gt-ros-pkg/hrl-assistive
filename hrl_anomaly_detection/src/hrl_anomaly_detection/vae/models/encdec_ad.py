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

def lstm_ae(trainData, testData, weights_file=None, batch_size=1024, nb_epoch=500,
            patience=20, fine_tuning=False, save_weights_file=None,
            noise_mag=0.0, timesteps=4, sam_epoch=1,
            re_load=False, renew=False, plot=True):
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
    
    inputs = Input(batch_shape=(batch_size, timesteps, input_dim*2))
    def slicing(x): return x[:,:,:input_dim]
    encoded = Lambda(slicing)(inputs)         
    encoded = LSTM(z_dim, return_sequences=False, activation='tanh', stateful=True)(encoded)
        
    # we initiate these layers to reuse later.
    #decoded_H2 = RepeatVector(timesteps)
    decoded_L1 = LSTM(input_dim, return_sequences=True, go_backwards=True, activation='sigmoid',
                      stateful=True)
    
    def slicing_last(x): return x[:,:,input_dim:]
    last_outputs = Lambda(slicing_last)(inputs)             
        
    decoded = merge([last_outputs, encoded], mode='concat')
    decoded = decoded_L1(decoded)    
    ae = Model(inputs, decoded)
    print(ae.summary())

    # Encoder --------------------------------------------------
    ae_encoder = Model(inputs, encoded)


    def mse_loss(y_true, y_pred):
        y = y[:,:,:input_dim]
        return K.mean(K.mean(K.square(y-y_pred),axis=-1), axis=-1)

    # AE --------------------------------------
    if weights_file is not None and os.path.isfile(weights_file) and fine_tuning is False and\
        re_load is False and renew is False:
        ae.load_weights(weights_file)
    else:
        if fine_tuning:
            ae.load_weights(weights_file)
            lr = 0.0001
            optimizer = Adam(lr=lr, clipvalue=10)                
            ae.compile(optimizer=optimizer, loss=mse_loss)
        else:
            if re_load and os.path.isfile(weights_file):
                ae.load_weights(weights_file)
            ae.compile(optimizer='adam', loss=mse_loss)

        # ---------------------------------------------------------------------------------
        nDim         = len(x_train[0][0])
        wait         = 0
        plateau_wait = 0
        min_loss = 1e+15
        for epoch in xrange(nb_epoch):
            print 

            mean_tr_loss = []
            for sample in xrange(sam_epoch):
                for i in xrange(0,len(x_train),batch_size):
                    seq_tr_loss = []
                    
                    if i+batch_size > len(x_train):
                        r = (i+batch_size-len(x_train))%len(x_train)
                        idx_list = range(len(x_train))
                        random.shuffle(idx_list)
                        x = np.vstack([x_train[i:],
                                       x_train[idx_list[:r]]])
                        while True:
                            if len(x)<batch_size:
                                x = np.vstack([x, x_train])
                            else:
                                break
                        
                    else:
                        x = x_train[i:i+batch_size]
                    
                    last_outputs = x[:,0:timesteps]                    
                    for j in xrange(len(x[0])-timesteps+1): # per window
                        np.random.seed(3334 + i*len(x[0]) + j)                        
                        noise = np.random.normal(0, noise_mag, (batch_size, timesteps, nDim))

                        tr_loss = ae.train_on_batch(np.concatenate((x[:,j:j+timesteps]+noise,
                                                                    last_outputs), axis=-1),
                                                                    x[:,j:j+timesteps]+noise )
                        last_outputs = ae.predict(np.concatenate((x[:,j:j+timesteps]+noise,
                                                                 last_outputs), axis=-1),
                                                  batch_size=batch_size)

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
                        x = np.vstack([x,
                                       x_test[idx_list[:r%len(x_test)]]])
                else:
                    x = x_test[i:i+batch_size]
                
                for j in xrange(len(x[0])-timesteps+1):
                    te_loss = ae.test_on_batch(
                        x[:,j:j+timesteps],
                        x[:,j:j+timesteps] )
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

    # ---------------------------------------------------------------------------------
    # visualize outputs
    if False:
        print "latent variable visualization"

    if plot:
        print "variance visualization"
        nDim = len(x_test[0,0])
        
        for i in xrange(len(x_test)):

            x = x_test[i:i+1]
            for j in xrange(batch_size-1):
                x = np.vstack([x,x_test[i:i+1]])

            ae.reset_states()
            
            x_pred_mean = []
            x_pred_std  = []
            for j in xrange(len(x[0])-timesteps+1):
                x_pred = ae.predict(x[:,j:j+timesteps], batch_size=batch_size)
                x_pred_mean.append(x_pred[0,-1,:])

            vutil.graph_variations(x_test[i], x_pred_mean)

    return ae, ae, None, ae_encoder


if __name__ == '__main__':

    print "N/A"
    


    
