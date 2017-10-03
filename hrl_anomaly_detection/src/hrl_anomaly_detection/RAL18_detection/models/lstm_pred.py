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



def lstm_pred(trainData, testData, weights_file=None, batch_size=32, nb_epoch=500, \
              patience=20, fine_tuning=False, save_weights_file=None, \
              noise_mag=0.0, timesteps=4, sam_epoch=1, \
              re_load=False, renew=False, plot=True, trainable=None):
    """
    Variational Autoencoder with two LSTMs and one fully-connected layer
    x_train is (sample x length x dim)
    x_test is (sample x length x dim)
    """

    x_train = trainData[0]
    x_test = testData[0]

    input_dim = len(x_train[0][0])
    length = len(x_train[0])

    x_train, y_train = create_dataset(x_train, timesteps, 5)
    x_test, y_test   = create_dataset(x_test, timesteps, 5)


    h_dim = input_dim
    o_dim  = timesteps
    np.random.seed(3334)

    inputs  = Input(batch_shape=(batch_size, timesteps, input_dim))
    encoded = LSTM(h_dim, return_sequences=True, activation='tanh', stateful=True,
                   trainable=True if trainable==0 or trainable is None else False)(inputs)
    outputs = LSTM(input_dim, return_sequences=True, activation='sigmoid', stateful=True,
                   trainable=True if trainable==0 or trainable is None else False)(encoded)
    ## outputs  = Dense(o_dim, trainable=True if trainable==1 or trainable is None else False)(encoded) 
    net = Model(inputs, outputs)
    print(net.summary())

    if weights_file is not None and os.path.isfile(weights_file) and fine_tuning is False and\
        re_load is False and renew is False:
        net.load_weights(weights_file)
    else:
        if fine_tuning:
            net.load_weights(weights_file)
            lr = 0.001
            optimizer = Adam(lr=lr, clipvalue=10)                
            #net.compile(optimizer=optimizer, loss=None)
            net.compile(optimizer='rmsprop', loss=None)
        else:
            if re_load and os.path.isfile(weights_file):
                net.load_weights(weights_file)
            lr = 0.01
            #optimizer = RMSprop(lr=lr, rho=0.9, epsilon=1e-08, decay=0.0001, clipvalue=10)
            optimizer = Adam(lr=lr, clipvalue=10) #, decay=1e-5)                
            #net.compile(optimizer=optimizer, loss=None)
            net.compile(optimizer='adam', loss='mse')

        # ---------------------------------------------------------------------------------
        nDim         = len(x_train[0][0][0])
        wait         = 0
        plateau_wait = 0
        min_loss = 1e+15
        for epoch in xrange(nb_epoch):
            print 

            mean_tr_loss = []
            for sample in xrange(sam_epoch):

                # shuffle
                idx_list = range(len(x_train))
                np.random.shuffle(idx_list)
                x_train = x_train[idx_list]
                y_train = y_train[idx_list]
                
                for i in xrange(0,len(x_train),batch_size):
                    seq_tr_loss = []

                    #shift_offset = int(np.random.normal(0,2,size=batch_size))
                    ## if shift_offset>0:
                    ##     x = np.pad(x_train[i],((0,shift_offset),(0,0)), 'edge')
                    ## else:
                    ##     x = np.pad(x_train[i],((abs(shift_offset),0),(0,0)), 'edge')
                    ##     shift_offset = 0
                    
                    shift_offset = 0
                    if i+batch_size > len(x_train):
                        r = (i+batch_size-len(x_train))%len(x_train)
                        idx_list = range(len(x_train))
                        random.shuffle(idx_list)
                        x = np.vstack([x_train[i:],
                                       x_train[idx_list[:r]]])
                        y = np.vstack([y_train[i:],
                                       y_train[idx_list[:r]]])
                        
                        while True:
                            if len(x)<batch_size:
                                x = np.vstack([x, x_train])
                                y = np.vstack([y, y_train])
                            else:
                                break
                    else:
                        x = x_train[i:i+batch_size]
                        y = y_train[i:i+batch_size]

                    for j in xrange(len(x[0])): # per window
                        np.random.seed(3334 + i*len(x[0]) + j)                        
                        noise = np.random.normal(0, noise_mag, (batch_size, timesteps, nDim))

                        tr_loss = net.train_on_batch(
                            x[:,j]+noise,
                            y[:,j]+noise)
                        seq_tr_loss.append(tr_loss)
                    mean_tr_loss.append( np.mean(seq_tr_loss) )
                    net.reset_states()

                sys.stdout.write('Epoch {} / {} : loss training = {} , loss validating = {}\r'.format(epoch, nb_epoch, np.mean(mean_tr_loss), 0))
                sys.stdout.flush()   

            mean_te_loss = []
            for i in xrange(0,len(x_test),batch_size):
                seq_te_loss = []

                # batch augmentation
                if i+batch_size > len(x_test):
                    x = x_test[i:]
                    y = y_test[i:]
                    r = i+batch_size-len(x_test)

                    for k in xrange(r/len(x_test)):
                        x = np.vstack([x, x_test])
                        y = np.vstack([y, y_test])
                    
                    if (r%len(x_test)>0):
                        idx_list = range(len(x_test))
                        random.shuffle(idx_list)
                        x = np.vstack([x,
                                       x_test[idx_list[:r%len(x_test)]]])
                        y = np.vstack([y,
                                       y_test[idx_list[:r%len(y_test)]]])
                else:
                    x = x_test[i:i+batch_size]
                    y = y_test[i:i+batch_size]
                
                for j in xrange(len(x[0])):
                    te_loss = net.test_on_batch(x[:,j], y[:,j])
                    seq_te_loss.append(te_loss)
                mean_te_loss.append( np.mean(seq_te_loss) )
                net.reset_states()

            val_loss = np.mean(mean_te_loss)
            sys.stdout.write('Epoch {} / {} : loss training = {} , loss validating = {}\r'.format(epoch, nb_epoch, np.mean(mean_tr_loss), val_loss))
            sys.stdout.flush()   


            # Early Stopping
            if val_loss <= min_loss:
                min_loss = val_loss
                wait         = 0
                plateau_wait = 0

                if save_weights_file is not None:
                    net.save_weights(save_weights_file)
                else:
                    net.save_weights(weights_file)
                
            else:
                if wait > patience:
                    print "Over patience!"
                    break
                else:
                    wait += 1
                    plateau_wait += 1

            #ReduceLROnPlateau
            if plateau_wait > 2:
                old_lr = float(K.get_value(net.optimizer.lr))
                new_lr = old_lr * 0.2
                K.set_value(net.optimizer.lr, new_lr)
                plateau_wait = 0
                print 'Reduced learning rate {} to {}'.format(old_lr, new_lr)

        gc.collect()

    # ---------------------------------------------------------------------------------
    # visualize outputs
    if False:
        print "latent variable visualization"
        

    if plot:
        print "variance visualization"
        nDim = input_dim
        
        for i in xrange(len(x_test)): #per sample

            x = x_test[i:i+1]
            for j in xrange(batch_size-1):
                x = np.vstack([x,x_test[i:i+1]])

            net.reset_states()
            
            x_pred_mean = []
            x_true = []
            ## x_pred_std  = []
            for j in xrange(len(x[0])):
                x_pred = net.predict(x[:,j], batch_size=batch_size)
                x_pred_mean.append(x_pred[0,-1,:nDim])
                x_true.append(x_test[i,j,-1])



            vutil.graph_variations(x_true, x_pred_mean)
        


    return net #, vae_mean_std, vae_mean_std, vae_encoder_mean, vae_encoder_var, generator


def create_dataset(dataset, window_size=5, step=5):
    '''
    dataset: sample x timesteps x dim
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
