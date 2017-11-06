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
from attrdict import AttrDict
import numpy as np
import matplotlib.pyplot as plt
import seaborn
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
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint

#configurations
I_DIM = 3 #image dimension
N_MFCC = 3 #audio dimension
NUM_FEATURE = N_MFCC + I_DIM #total dimension in LSTM
WINDOW_SIZE_IN = 5
WINDOW_SIZE_OUT = 1 
LOAD_WEIGHT = False

def create_model():
    model = Sequential()
    model.add(LSTM(output_dim=NUM_FEATURE, input_shape=(WINDOW_SIZE_IN, NUM_FEATURE)))
    model.add(Activation('linear'))  

    if LOAD_WEIGHT:
        model.load_weights('./models/combined.hdf5')

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])  
    return model

def train_model(model, dataX, dataY):
    csv_logger = CSVLogger('training_audio.log')
    escb = EarlyStopping(monitor='val_loss', patience=2, verbose=1)
    checkpoint = ModelCheckpoint("models/combined-{epoch:02d}-{val_loss:.2f}.hdf5", 
        monitor='val_loss', save_best_only=True, verbose=1) #, period=2)

    model.fit(dataX, dataY, shuffle=True, batch_size=256, verbose=1, #initial_epoch=50,
              validation_split=0.3, nb_epoch=500, callbacks=[csv_logger, escb, checkpoint])
    
def main():
    os.environ["KERAS_BACKEND"] = "tensorflow"

    inputFile = './processed_data/combined'
    #Load up the training data
    print ('Loading training data')
    X_train = np.load(inputFile + '_x.npy')
    y_train = np.load(inputFile + '_y.npy')
    print ('Finished loading training data')
    print(X_train.shape)
    print(y_train.shape)
    
    print('creating model...')
    model = create_model()
    print('training model...')
    train_model(model, X_train, y_train)


if __name__ == '__main__':
    main()
