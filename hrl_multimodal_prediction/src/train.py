#!/usr/bin/python
#
# Copyright (c) 2017, Georgia Tech Research Corporation
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
#  \author Michael Park (Healthcare Robotics Lab, Georgia Tech.)

import rosbag
import rospy
import numpy as np
from matplotlib import pyplot 
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from numpy import array
import math
from matplotlib.pylab import *
from mpl_toolkits.axes_grid1 import host_subplot
import matplotlib.animation as animation
from sklearn.model_selection import train_test_split
import gc
import librosa
import os, copy, sys
import tensorflow as tf
from tqdm import tqdm
import numpy.matlib
import scipy.io.wavfile as wav
import scipy as sp
import scipy.interpolate

from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout, RepeatVector, TimeDistributed, Input
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from keras import optimizers
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from keras import backend as K

#Use these config only in main
IMAGE_DIM = 3 #image dimension
MFCC_DIM = 3 #audio dimension
INPUT_DIM = MFCC_DIM + IMAGE_DIM #total dimension in LSTM
TIMESTEP_IN = 1
TIMESTEP_OUT = 10
N_NEURONS = TIMESTEP_OUT

BATCH_SIZE = 256
NUM_BATCH = 200 #Total #samples = Num_batch x Batch_size
NB_EPOCH = 500
PRED_BATCH_SIZE = 1

WEIGHT_FILE = './weights/real_data.h5'
PLOT = True
DENSE = True #True if TimeDistributedDense layer is used

def define_network(batch_size, time_in, time_out, input_dim, n_neurons, load_weight=False):
    model = Sequential()
    model.add(LSTM(input_dim*time_out, batch_input_shape=(batch_size, time_in, input_dim),
                    stateful=True, return_sequences=True, activation='tanh'))
    model.add(LSTM(input_dim*time_out, stateful=True, return_sequences=True, activation='tanh'))
    model.add(TimeDistributed(Dense(input_dim*time_out, activation='linear')))
    model.compile(loss='mse', optimizer='RMSprop')
    print model.summary()
    print "Inputs: {}".format(model.input_shape)
    print "Outputs: {}".format(model.output_shape)
    return model

def fit_lstm(model, x_train, x_test, y_train, y_test):
    wait         = 0
    plateau_wait = 0
    min_loss = 1e+15
    patience = 5
    plot_tr_loss = []
    plot_te_loss = []
    for epoch in range(NB_EPOCH):
        #train
        mean_tr_loss = []
        for i in range(0,x_train.shape[0], BATCH_SIZE):
            #per window
            seq_tr_loss = []
            x = x_train[i:i+BATCH_SIZE]
            y = y_train[i:i+BATCH_SIZE]
            x = np.swapaxes(x, 0, 1)
            y = np.swapaxes(y, 0, 1)
            for j in range(x.shape[0]):
                tr_loss = model.train_on_batch(x[j], y[j])
                seq_tr_loss.append(tr_loss)
            mean_tr_loss.append( np.mean(seq_tr_loss) )
            model.reset_states()
        tr_loss = np.mean(mean_tr_loss)
        sys.stdout.write('Epoch {} / {} : loss training = {} , loss validating = {}\n'.format(epoch, NB_EPOCH, tr_loss, 0))
        sys.stdout.flush()
        plot_tr_loss.append(tr_loss)

        #test  
        mean_te_loss = []
        for i in xrange(0, x_test.shape[0], BATCH_SIZE):
            seq_te_loss = []
            x = x_test[i:i+BATCH_SIZE]
            y = y_test[i:i+BATCH_SIZE]
            x = np.swapaxes(x, 0, 1)
            y = np.swapaxes(y, 0, 1)
            for j in xrange(x.shape[0]):
                te_loss = model.test_on_batch(x[j], y[j])
                seq_te_loss.append(te_loss)
            mean_te_loss.append( np.mean(seq_te_loss) )
            model.reset_states()
        val_loss = np.mean(mean_te_loss)
        sys.stdout.write('Epoch {} / {} : loss training = {} , loss validating = {}\n'.format(epoch, NB_EPOCH, tr_loss, val_loss))
        sys.stdout.flush()   
        plot_te_loss.append(val_loss)

        # Early Stopping
        if val_loss <= min_loss:
            min_loss = val_loss
            wait         = 0
            plateau_wait = 0
            print 'saving model'
            model.save_weights(WEIGHT_FILE) 
        else:
            if wait > patience:
                print "Over patience!"
                break
            else:
                wait += 1
                plateau_wait += 1

        #ReduceLROnPlateau
        if plateau_wait > 2:
            old_lr = float(K.get_value(model.optimizer.lr)) #K is a backend
            new_lr = old_lr * 0.2
            K.set_value(model.optimizer.lr, new_lr)
            plateau_wait = 0
            print 'Reduced learning rate {} to {}'.format(old_lr, new_lr)

        gc.collect()    

    # ---------------------------------------------------------------------------------
    # visualize outputs
    print "Training history"
    fig = pyplot.figure(figsize=(10,4))
    ax1 = fig.add_subplot(1, 2, 1)
    pyplot.plot(plot_tr_loss)
    ax1.set_title('loss')
    ax2 = fig.add_subplot(1, 2, 2)
    pyplot.plot(plot_te_loss)
    ax2.set_title('validation loss')
    pyplot.show()

    return model


def main():
    '''
    dataset.shape:: (num_window, batch x N, window_size, dim)
    '''

    #####-------Loading Preprocessed Data--------##########
    inputFile = './processed_data/combined'
    #Load up the training data
    print ('Loading training data')
    X_train = np.load(inputFile + '_x.npy')
    y_train = np.load(inputFile + '_y.npy')
    print ('Finished loading training data')
    print(X_train.shape)
    print(y_train.shape)   

    X, y = data_generator() #generates entire dataset to be used shape is listed above

    X = X.reshape(X.shape[0],X.shape[1],X.shape[2],INPUT_DIM)
    y = y.reshape(y.shape[0],y.shape[1],y.shape[2],INPUT_DIM) #comment input_dim for 1D
    X = np.swapaxes(X, 0, 1)
    y = np.swapaxes(y, 0, 1)
    print 'in main'
    print X.shape, y.shape

    X = add_noise(X)
    ########################################################################

    # flatten data to shape into lstm
    if DENSE:
        y = y.reshape(y.shape[0], y.shape[1], 1, y.shape[2]*y.shape[3])
        print y.shape
    else:
        y = y.reshape(y.shape[0], y.shape[1], y.shape[2]*y.shape[3])
        print y.shape

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print X_train.shape, X_test.shape, y_train.shape, y_test.shape

    # for i in range(X_test.shape[0]):
    #   pyplot.plot(X_test[i,:,:,0])
    # pyplot.show()
    # for i in range(y_test.shape[0]):
 #        pyplot.plot(y_test[i,:,:])
    # pyplot.show()

    np.random.seed(3334)
    print('creating model...')
    #train phase
    lstm_model = define_network(BATCH_SIZE, TIMESTEP_IN, TIMESTEP_OUT, INPUT_DIM, N_NEURONS, False)
    print('training model...')
    lstm_model = fit_lstm(lstm_model, X_train, X_test, y_train, y_test)



if __name__ == '__main__':
    main()
