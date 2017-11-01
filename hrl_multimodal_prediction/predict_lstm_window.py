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

#Training and Testing phase done separately (code to be cleaned up)
#comment code in testing phase when training
#comment code in training phase except create model when testing. See below for detail

#For Prediction experiments, consider changing below (For now cuz code is crappy)
# 1) N_MFCC=?
# 2) weight file(hdf), LOAD_WEIGHT=1
# 3) inputfile when creating prediction data
# 4) in audio_to_tensor, uncomment audio_dataX = audio_dataX[0] and comment two lines below
# 5) comment code for training phase except for create_model

#For Prediction experiments, consider changing below 
# 1) LOAD_WEIGHT=0
# 2) save best trained model and delete reset from the model file
# 3) in audio_to_tensor, comment audio_dataX = audio_dataX[0] and uncomment two lines below
# 4) comment code for prediction/testing phase 
# 5) Try training with different N_MFCC cuz predicted sound quality degrades significantly

#*Note that parameters and operations related to the window size are hardcoded for now, so don't change them

#Configurations
SOUND_FOLDER = './sounds/cropped/'
CSV_FOLDER = './csv/'
EPOCHS = 10
# N_PRE = 10
# N_POST = 1
AUDIO_DATA = 1
IMAGE_DATA = 1
DATA_COMBINE = 1

#Image Configuration
IMAGE_FILENAME = ['data1.txt', 'data2.txt', 'data5.txt']
I_DIM = 3

#Audio Configuration
AUDIO_FILENAME = ['data1crop4.wav', 'data2crop4.wav', 'data5crop4.wav']
N_MFCC = 3

COMBINED_FILENAME = {'data1.txt':'data1crop4.wav', 'data2.txt':'data2crop4.wav', 'data5.txt':'data5crop4.wav'}

#LSTM Configuration
NUM_SAMPLES = 3 # N aka number of experiments
NUM_FEATURE = N_MFCC + I_DIM #Dimension in LSTM
NUM_TIME_SAMPLE = 91 #Number of total time samples
WINDOW_SIZE_IN = 5
WINDOW_SIZE_OUT = 1 
LOAD_WEIGHT = 1
NUM_STEP_SHOW = NUM_TIME_SAMPLE - WINDOW_SIZE_IN #86

def create_model():
    # For multiple LSTMS
    # hidden_neurons = 300
    # if LAYERS == 1:
    #     hidden_neurons = feature_count
    model = Sequential()
    model.add(LSTM(output_dim=NUM_FEATURE, input_shape=(WINDOW_SIZE_IN, NUM_FEATURE)))
    #model.add(LSTM(output_dim=hidden_neurons, return_sequences=True))
    #model.add(TimeDistributed(Dense(feature_count)))
    model.add(Activation('linear'))  

    if LOAD_WEIGHT == 1:
        model.load_weights('./models/combined.hdf5')

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])  
    return model

def train_model(model, dataX, dataY):
    #history = model.fit(dataX, dataY, batch_size=3, nb_epoch=epoch_count, validation_split=0.05)

    csv_logger = CSVLogger('training_audio.log')
    escb = EarlyStopping(monitor='val_loss', patience=2, verbose=1)
    checkpoint = ModelCheckpoint("models/combined-{epoch:02d}-{val_loss:.2f}.hdf5", 
        monitor='val_loss', save_best_only=True, verbose=1) #, period=2)

    model.fit(dataX, dataY, shuffle=True, batch_size=256, verbose=1, #initial_epoch=50,
              validation_split=0.3, nb_epoch=500, callbacks=[csv_logger, escb, checkpoint])

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

# def image_to_tensor(csv_filename):
    
# def audio_to_tensor(audio_filename):
#     #these data have to be in same length
#     # audio_filename = ['data1crop4.wav', 'data2crop4.wav', 'data5crop4.wav']
#     audio_dataX = []
#     audio_dataY = []

#     for audio_file in audio_filename:
#         y, sr = librosa.load(SOUND_FOLDER + audio_file, mono=True)
#         mfccs = librosa.feature.mfcc(y, n_mfcc=N_MFCC) #default hop_length=512
#         mfccs = np.rollaxis(mfccs, 1, 0)
#         dX, dY = [], []
#         for i in range(mfccs.shape[0] - WINDOW_SIZE_IN):
#                 dX.append(mfccs[i:i+WINDOW_SIZE_IN])
#                 dY.append(mfccs[i+WINDOW_SIZE_IN:i+WINDOW_SIZE_IN+WINDOW_SIZE_OUT][0])
#         audio_dataX.append(dX)
#         audio_dataY.append(dY)
#     audio_dataX = np.array(audio_dataX)
#     audio_dataY = np.array(audio_dataY)

#     #For Prediction
#     audio_dataX = audio_dataX[0]
#     #Comment out for Predction -- really crappy code for now
#     # audio_dataX = np.concatenate((audio_dataX[0], audio_dataX[1], audio_dataX[2]), axis=0)
#     # audio_dataY = np.concatenate((audio_dataY[0], audio_dataY[1], audio_dataY[2]), axis=0)

#     print 'audiodata shape'
#     print audio_dataX.shape, audio_dataY.shape

#     #normalization should be done feature by feature
#     audio_dataX, min_audio_dataX, max_audio_dataX = normalize(audio_dataX) 
#     audio_dataY, min_audio_dataY, max_audio_dataY = normalize(audio_dataY) 

#     audio_dataX = np.array(audio_dataX, dtype=float)
#     audio_dataY = np.array(audio_dataY, dtype=float) 
#     print audio_dataX.shape, audio_dataY.shape    

#     return sr, audio_dataX, min_audio_dataX, max_audio_dataX, audio_dataY, min_audio_dataY, max_audio_dataY

def create_data(n_pre, n_post, combined_filename):
    if AUDIO_DATA and IMAGE_DATA:
        print 'using audio and image mode...'
        audio_dataX = []
        audio_dataY = []
        image_dataX = []
        image_dataY = []
        combined_dataX = []
        combined_dataY = []

        #first read both data from wav and csv
        for image_filename in combined_filename:
            audio_filename = combined_filename[image_filename]
            image = np.loadtxt(CSV_FOLDER + image_filename)
            audio, sr = librosa.load(SOUND_FOLDER + audio_filename, mono=True)
            mfccs = librosa.feature.mfcc(audio, n_mfcc=N_MFCC) #default hop_length=512
            mfccs = np.rollaxis(mfccs, 1, 0)
            # print image.shape
            # print mfccs.shape

            #Interpolate image data by averaging
            #Note that these csv and wav file don't exactly correspond to each other
            # -> Must Crop the Rosbag first then create the data
            new_length = mfccs.shape[0] #91
            #make this a 1d interpolation where x is a time, y is the values
            # x=np.linspace(0,2*np.pi,45)
            # y=np.zeros((2,45))
            # y[0,:]=sp.sin(x)
            # y[1,:]=sp.sin(2*x)
            # f=sp.interpolate.interp1d(x,y)
            # y2=f(np.linspace(0,2*np.pi,100))
            im_time_len = image.shape[0] #47
            im_feature_len = image.shape[1] #3 xyz

            l = np.arange(im_time_len)
            image = np.rollaxis(image, 1, 0)
            f1=sp.interpolate.interp1d(l, image[0]) #x
            f2=sp.interpolate.interp1d(l, image[1]) #y
            f3=sp.interpolate.interp1d(l, image[2]) #z

            image_intp = np.zeros((im_feature_len, new_length))
            image_intp[0] = f1(np.linspace(0,im_time_len-1, new_length))
            image_intp[1] = f2(np.linspace(0,im_time_len-1, new_length))
            image_intp[2] = f3(np.linspace(0,im_time_len-1, new_length))
            image_intp = np.rollaxis(image_intp, 1, 0)        
            
            print image_intp.shape
            print mfccs.shape

            # Create a windowed set dX_audio, dY_audio, concatenate, normalize(91,mfcc)
            dX_audio, dY_audio = [], []
            for i in range(mfccs.shape[0] - WINDOW_SIZE_IN):
                dX_audio.append(mfccs[i:i+WINDOW_SIZE_IN])
                dY_audio.append(mfccs[i+WINDOW_SIZE_IN:i+WINDOW_SIZE_IN+WINDOW_SIZE_OUT][0])
            audio_dataX.append(dX_audio)
            audio_dataY.append(dY_audio)
            # Create a windowed set dX_image, dY_image, concatenate, normalize(91,xyz)
            dX_image, dY_image = [], []
            for i in range(image_intp.shape[0] - WINDOW_SIZE_IN):
                dX_image.append(image_intp[i:i+WINDOW_SIZE_IN])
                dY_image.append(image_intp[i+WINDOW_SIZE_IN:i+WINDOW_SIZE_IN+WINDOW_SIZE_OUT][0])
            image_dataX.append(dX_image)
            image_dataY.append(dY_image)
        
        #audio normalize over entire samples and mfccs
        audio_dataX = np.array(audio_dataX)
        audio_dataY = np.array(audio_dataY)
        print audio_dataX.shape
        print audio_dataY.shape
        #For Prediction
        print 'bug fixed at this point'
        audio_dataX = audio_dataX[0]
        audio_dataY = audio_dataY[0]
        #Comment out 2 lines below for Predction -- really crappy code for now
        # audio_dataX = np.concatenate((audio_dataX[0], audio_dataX[1], audio_dataX[2]), axis=0)
        # audio_dataY = np.concatenate((audio_dataY[0], audio_dataY[1], audio_dataY[2]), axis=0)
        audio_dataX, min_audio_dataX, max_audio_dataX = normalize(audio_dataX) 
        audio_dataY, min_audio_dataY, max_audio_dataY = normalize(audio_dataY)
        audio_dataX = np.array(audio_dataX, dtype=float)
        audio_dataY = np.array(audio_dataY, dtype=float) 
        print audio_dataX.shape
        print audio_dataY.shape

        #image normalize over entire samples and xyz
        image_dataX = np.array(image_dataX)
        image_dataY = np.array(image_dataY)
        #For Prediction
        print 'bug fixed at this point'
        image_dataX = image_dataX[0]
        image_dataY = image_dataY[0]
        #Comment out 2 lines below for Predction -- really crappy code for now
        # image_dataX = np.concatenate((image_dataX[0], image_dataX[1], image_dataX[2]), axis=0)
        # image_dataY = np.concatenate((image_dataY[0], image_dataY[1], image_dataY[2]), axis=0)
        image_dataX, min_image_dataX, max_image_dataX = normalize(image_dataX) 
        image_dataY, min_image_dataY, max_image_dataY = normalize(image_dataY)
        image_dataX = np.array(image_dataX, dtype=float)
        image_dataY = np.array(image_dataY, dtype=float) 

        print 'audio image separate'
        print audio_dataX.shape
        print audio_dataY.shape
        print image_dataX.shape
        print image_dataY.shape

        #combine normalized audio and image data
        combined_dataX = np.concatenate((audio_dataX, image_dataX), axis=2)
        combined_dataY = np.concatenate((audio_dataY, image_dataY), axis=1)
        print 'audio image combined'
        print combined_dataX.shape
        print combined_dataY.shape

        return (sr, min_audio_dataX, max_audio_dataX, min_audio_dataY, max_audio_dataY, 
                min_image_dataX, max_image_dataX, min_image_dataY, max_image_dataY, 
                combined_dataX, combined_dataY)
    
    else:
        print 'using single modality'
        if AUDIO_DATA == 1:
            print 'using audio mode only'
            audio_dataX = []
            audio_dataY = []

            for audio_file in audio_filename:
                y, sr = librosa.load(SOUND_FOLDER + audio_file, mono=True)
                mfccs = librosa.feature.mfcc(y, n_mfcc=N_MFCC) #default hop_length=512
                mfccs = np.rollaxis(mfccs, 1, 0)
                dX, dY = [], []
                for i in range(mfccs.shape[0] - WINDOW_SIZE_IN):
                    dX.append(mfccs[i:i+WINDOW_SIZE_IN])
                    dY.append(mfccs[i+WINDOW_SIZE_IN:i+WINDOW_SIZE_IN+WINDOW_SIZE_OUT][0])
                audio_dataX.append(dX)
                audio_dataY.append(dY)
            audio_dataX = np.array(audio_dataX)
            audio_dataY = np.array(audio_dataY)

            #For Prediction
            audio_dataX = audio_dataX[0]
            #Comment out for Predction -- really crappy code for now
            # audio_dataX = np.concatenate((audio_dataX[0], audio_dataX[1], audio_dataX[2]), axis=0)
            # audio_dataY = np.concatenate((audio_dataY[0], audio_dataY[1], audio_dataY[2]), axis=0)

            print 'audiodata shape'
            print audio_dataX.shape, audio_dataY.shape

            #normalization should be done feature by feature
            audio_dataX, min_audio_dataX, max_audio_dataX = normalize(audio_dataX) 
            audio_dataY, min_audio_dataY, max_audio_dataY = normalize(audio_dataY) 

            audio_dataX = np.array(audio_dataX, dtype=float)
            audio_dataY = np.array(audio_dataY, dtype=float) 
            print audio_dataX.shape, audio_dataY.shape    
            
            return (sr, audio_dataX, audio_dataY, 
                    min_audio_dataX, max_audio_dataX, min_audio_dataY, max_audio_dataY)

        elif IMAGE_DATA == 1:
            print 'using image mode only'
            for image_file in csv_filename:
                image = np.loadtxt(CSV_FOLDER + image_file)      
            return image 
            #Windowing and creating training set missing. But no need for now

    #final data should have (batch_size=num_samples, time_step, num_features) (eg)(3,10,3)
    # (eg) [dataX: (3,10,3), dataY:(3,1,3)] x 90(total time series)

def invlogamplitude(S):
#"""librosa.logamplitude is actually 10_log10, so invert that."""
    return 10.0**(S/10.0)

def reconstruct_audio(mfccs, sr, y_shape):
    #build reconstruction mappings
    n_mfcc = mfccs.shape[0]
    n_mel = 128
    dctm = librosa.filters.dct(n_mfcc, n_mel)
    n_fft = 2048
    mel_basis = librosa.filters.mel(sr, n_fft)

    #Empirical scaling of channels to get ~flat amplitude mapping.
    bin_scaling = 1.0/np.maximum(0.0005, np.sum(np.dot(mel_basis.T, mel_basis), axis=0))
    #Reconstruct the approximate STFT squared-magnitude from the MFCCs.
    recon_stft = bin_scaling[:, np.newaxis] * np.dot(mel_basis.T, invlogamplitude(np.dot(dctm.T, mfccs)))
    #Impose reconstructed magnitude on white noise STFT.
    #tot_timeseq = 91
    #y = np.zeros((N_MFCC,tot_timeseq))
    excitation = np.random.randn(y_shape)
    E = librosa.stft(excitation)
    recon = librosa.istft(E/np.abs(E)*np.sqrt(recon_stft))
    #print recon
    #print recon.shape

    wav.write('./sounds/predicted/' + 'combined_predict_testdata20' +'FromMFCC3.wav', sr, recon)

def reconstruct_image(image):
    np.savetxt('./csv/predicted/' + 'combined_predict_testdata20' + '.txt', image)

def test_prediction():
    os.environ["KERAS_BACKEND"] = "tensorflow"
    # have Three options
    # 1) one to one (eg) n_pre=1, n_post=1
    # 2) sequence to one (eg) n_pre=10, n_post=1
    # 3) sequence to sequence (eg) n_pre=10, n_post=10
    # n_pre = N_PRE
    # n_post = N_POST

    # ******************************************************** #
    # Training Phase
    # ******************************************************** #
    # print('creating dataset...')
    # (sr, min_audio_dataX, max_audio_dataX, min_audio_dataY, max_audio_dataY, 
    #     min_image_dataX, max_image_dataX, min_image_dataY, max_image_dataY, 
    #     combined_dataX, combined_dataY) = create_data(WINDOW_SIZE_IN, WINDOW_SIZE_OUT, COMBINED_FILENAME)
    print('creating model...')
    model = create_model()
    # print combined_dataX.shape, combined_dataY.shape
    # print('training model...')
    # train_model(model, combined_dataX, combined_dataY)
    
    # ******************************************************** #
    # Testing Phase 
    # ******************************************************** #
    # Prepare Testing Data - ORIGINAL(AUDIO ONLY)
    # sr, audio_dataX, min_audio_dataX, max_audio_dataX, audio_dataY, min_audio_dataY, max_audio_dataY = create_data(WINDOW_SIZE_IN, WINDOW_SIZE_OUT, ['data13crop4.wav'])
    # datain = audio_dataX
    # print 'shape?'
    # print datain.shape
    # y_shape = 512*(NUM_STEP_SHOW-1)

    # audio_predict = model.predict(datain)
    # audio_predict = np.rollaxis(audio_predict, 1, 0) # (array, axis, start=0)
    # print 'predicted test data'
    # print audio_predict.shape
    # print audio_predict
    # audio_predict = scale_back(audio_predict, min_audio_dataX, max_audio_dataX)
    # print audio_predict
    # reconstruct_audio(audio_predict, sr, y_shape)

    # Prepare Testing Data - MODIFIED(COMBINED DATA)
    (sr, min_audio_dataX, max_audio_dataX, min_audio_dataY, max_audio_dataY, 
        min_image_dataX, max_image_dataX, min_image_dataY, max_image_dataY, 
        combined_dataX, combined_dataY) = create_data(WINDOW_SIZE_IN, WINDOW_SIZE_OUT, {'data20.txt':'data20crop4.wav'})
    datain = combined_dataX
    print 'shape?'
    print datain.shape
    print combined_dataY.shape
    y_shape = 512*(NUM_STEP_SHOW-1)

    #################################
    seq = []
    for i in range(datain.shape[0]):
        tmp = datain[i]
        tmp = np.expand_dims(tmp, axis=0)
        print 'preidction loop'
        print tmp.shape
        predicted = model.predict(tmp)
        print predicted.shape
        seq.append(predicted[0])
    seq = np.array(seq)
    print seq
    print seq.shape
    combined_predict = seq
    #################################

    combined_predict = np.rollaxis(combined_predict, 1, 0) # (array, axis, start=0), mfccformat = (feature, time)
    print 'predicted test data'
    print combined_predict.shape
    #print combined_predict
    #Need to extract mfccs only from the combined
    audio_predict = combined_predict[0:3,:] #mfcc=[0:3,:], xyz=[3:6,:]
    image_predict = combined_predict[3:6,:]
    # print audio_predict.shape
    audio_predict = scale_back(audio_predict, min_audio_dataX, max_audio_dataX)
    image_predict = scale_back(image_predict, min_image_dataX, max_image_dataX)
    # print audio_predict
    reconstruct_audio(audio_predict, sr, y_shape)
    reconstruct_image(image_predict)
    print image_predict.shape
    print image_predict

def main():
    test_prediction()
    return 1

if __name__ == "__main__":
    sys.exit(main())

