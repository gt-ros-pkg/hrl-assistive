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

#configuration
I_DIM = 3
N_MFCC = 3

NUM_SAMPLES = 3 # N aka number of experiments
NUM_FEATURE = N_MFCC + I_DIM #Dimension in LSTM
NUM_TIME_SAMPLE = 344 #Number of total time samples
WINDOW_SIZE_IN = 1
WINDOW_SIZE_OUT = 1 
LOAD_WEIGHT = True
NUM_STEP_SHOW = NUM_TIME_SAMPLE - WINDOW_SIZE_IN #86
UNPACK = False

# SAMPLEBAG_PATH = './samplebag/'
# bagfile  = SAMPLEBAG_PATH + 'data50.bag'
# wavfile  = SAMPLEBAG_PATH + 'data50.wav'
wavfileMFCC = './predicted/testdata_MFCC.wav'
txtfile = './predicted/testdata_XYZ.txt'

def normalize(y):
    # normalize - for feeding into LSTM
    min_y = np.min(y)
    max_y = np.max(y)
    y = (y - min_y) / (max_y - min_y)
    #print y.dtype, min_y, max_y
    return y, min_y, max_y

def scale_back(seq, min_y, max_y):
    # scale back 
    seq = seq * (max_y - min_y) + min_y
    return seq

def create_model():
    model = Sequential()
    model.add(LSTM(output_dim=NUM_FEATURE, input_shape=(WINDOW_SIZE_IN, NUM_FEATURE)))
    model.add(Activation('linear'))  

    if LOAD_WEIGHT:
        model.load_weights('./models/zcombined-13-0.00.hdf5')
    
    optimizer = RMSprop(lr=0.01) 
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])  
    return model

def test_prediction():
    os.environ["KERAS_BACKEND"] = "tensorflow"
 
    print('creating model...')
    model = create_model()
    
    inputFile = './processed_data/combined_test'
    #Load up the training data
    print ('Loading test data')
    X_test = np.load(inputFile + '_x.npy')
    y_test = np.load(inputFile + '_y.npy')
    print ('Finished loading test data')
    print(X_test.shape)
    print(y_test.shape)

    print ('Loading minmax data')
    AudioMinMax = np.load('./processed_data/combined_test_AudioMinMax_x.npy')
    ImageMinMax = np.load('./processed_data/combined_test_ImageMinMax_x.npy')
    a_min = AudioMinMax[0]
    a_max = AudioMinMax[1]
    i_min = ImageMinMax[0]
    i_max = ImageMinMax[1]
    print a_min, a_max
    print i_min, i_max

    datain = X_test
    y_shape = 512*(NUM_STEP_SHOW-1)

    ########### Regular ################ - Works! Confirmed
    # seq = []
    # for i in range(datain.shape[0]):
    #     tmp = datain[i]
    #     tmp = np.expand_dims(tmp, axis=0)
    #     print tmp.shape
    #     predicted = model.predict(tmp)
    #     print predicted.shape
    #     seq.append(predicted[0])
    # seq = np.array(seq)
    # print seq
    # print seq.shape
    # combined_predict = seq
    #################################

    ############# Prediction Base ###### -- keep image but predict audio based on next audio
    seq = []
    tmp = datain[0]
    tmp = np.expand_dims(tmp, axis=0)
    for i in range(datain.shape[0]-1):
        predicted = model.predict(tmp) #(1,6)
        seq.append(predicted[0]) #(6,)
        #create next input
        predicted = np.expand_dims(predicted, axis=0) #(1,1,6)

        audio_p = tmp[:, 1:WINDOW_SIZE_IN, 0:3] #(1,4,3)
        a_predicted_ext = predicted[:,:,0:3] #(1,1,3)
        tmp = np.concatenate((audio_p, a_predicted_ext), axis=1)

        image_p = datain[i+1, :, 3:6]
        image_p = np.expand_dims(image_p, axis=0) #(1,1,6)

        tmp = np.concatenate((tmp, image_p), axis=2) #(1,5,6)
    seq = np.array(seq)
    combined_predict = seq
    #########################################################3

    combined_predict = np.rollaxis(combined_predict, 1, 0) # (array, axis, start=0), mfccformat = (feature, time)
    print 'combined predicted axis rolled'
    print combined_predict.shape
    #print combined_predict
    #Need to extract mfccs only from the combined
    audio_predict = combined_predict[0:3,:] #mfcc=[0:3,:], xyz=[3:6,:]
    image_predict = combined_predict[3:6,:]
    print 'extract audio'
    print audio_predict.shape
    print 'extract image'
    print image_predict.shape
    audio_predict = scale_back(audio_predict, a_min, a_max)
    image_predict = scale_back(image_predict, i_min, i_max)
    # print audio_predict
    reconstruct_mfcc(audio_predict, 44100, y_shape)
    reconstruct_image(image_predict)
    # print image_predict.shape
    # print image_predict

def reconstruct_image(image):
    np.savetxt(txtfile, image)

def reconstruct_mfcc(mfccs, sr, y_shape):
    #build reconstruction mappings
    n_mfcc = mfccs.shape[0]
    n_mel = 128
    dctm = librosa.filters.dct(n_mfcc, n_mel)
    n_fft = 2048
    mel_basis = librosa.filters.mel(44100, n_fft)

    #Empirical scaling of channels to get ~flat amplitude mapping.
    bin_scaling = 1.0/np.maximum(0.0005, np.sum(np.dot(mel_basis.T, mel_basis), axis=0))
    #Reconstruct the approximate STFT squared-magnitude from the MFCCs.
    recon_stft = bin_scaling[:, np.newaxis] * np.dot(mel_basis.T, invlogamplitude(np.dot(dctm.T, mfccs)))
    #Impose reconstructed magnitude on white noise STFT.
    excitation = np.random.randn(y_shape)
    E = librosa.stft(excitation)
    recon = librosa.istft(E/np.abs(E)*np.sqrt(recon_stft))
    #print recon
    #print recon.shape
    wav.write(wavfileMFCC, 44100, recon)

def invlogamplitude(S):
#"""librosa.logamplitude is actually 10_log10, so invert that."""
    return 10.0**(S/10.0)

def main():
    test_prediction()
    return 1

if __name__ == "__main__":
    sys.exit(main())

