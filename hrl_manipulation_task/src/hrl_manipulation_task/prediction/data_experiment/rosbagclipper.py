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

# Start up ROS pieces.
PKG = 'MFCC_Reconstruct'
# import roslib; roslib.load_manifest(PKG)
import rosbag
import rospy
import os, copy, sys
## from hrl_msgs.msg import FloatArray
## from std_msgs.msg import Float64
from hrl_anomaly_detection.msg import audio

# util
import numpy as np
import math
import pyaudio
import struct
import array
# try:
#     from features import mfcc
# except:
#     from python_speech_features import mfcc
#     from python_speech_features import logfbank
from python_speech_features import mfcc
from scipy import signal, fftpack, conj, stats

import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from matplotlib import cm

import librosa
import numpy as np
from IPython.lib.display import Audio

#GOAL: parse fusion.bag and clip data +-0.1s door close


def invlogamplitude(S):
    #"""librosa.logamplitude is actually 10_log10, so invert that."""
    return 10.0*(S/10.0)

#(1) Audio -> MFCC -> Reconstruction #
def audio_creator(): 
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    RECORD_SECONDS = 5
    WAVE_OUTPUT_FILENAME = "data1.wav"   
    BAG_INPUT_FILENAME = 'data1.bag'
    WINLEN = float(RATE)/float(CHUNK) #WINLEN=43/second


    data_store = []
    time_store = []
    for topic, msg, t in rosbag.Bag(BAG_INPUT_FILENAME).read_messages():
        if msg._type == 'hrl_anomaly_detection/audio':
            data_store.append(np.array(msg.audio_data, dtype=np.int16))
            time_store.append(t)

    # print data_store 
    # print "\n"
    # print time_store 
    # print "\n"

    #copy the frame and insert to lengthen
    data_store_long = []
    baglen = len(data_store)
    num_frames = RATE/CHUNK * RECORD_SECONDS
    recovered_len = num_frames/baglen

    mfcc_store = []
    for frame in data_store:
        for i in range(0, recovered_len): 
            data_store_long.append(frame)
            # audio_mfcc = mfcc(frame, samplerate=RATE, nfft=CHUNK, winlen=WINLEN).tolist()[0]
            # mfcc_store.append(audio_mfcc)
            # print audio_mfcc
            # print "\n"

    # mfccdata = np.hstack(mfcc_store)
    # mfccdata = np.reshape(mfccdata, (len(mfccdata)/CHANNELS, CHANNELS))    
    # mfccdata = mfcc(numpydata, samplerate=RATE, nfft=CHUNK, winlen=WINLEN).tolist()[0]
    # wav.write('data1mfcc_speachfe.wav', RATE, mfccdata)

    numpydata = np.hstack(data_store_long)

    # mfccdata = mfcc(numpydata, samplerate=RATE, nfft=CHUNK, winlen=WINLEN).tolist()[0]
    # wav.write('data1mfcc_speachfe.wav', RATE, mfccdata)

    #audio_mfcc = mfcc(numpydata, samplerate=RATE, nfft=CHUNK, winlen=WINLEN).tolist()[0]
    #print "audio mfcc"
    #print audio_mfcc
    #print "\n"

    #read WAV - Using numpydata works!(bad quality) but data read from WAV doesn't work.
    #Invalid value encountered -- 0. ??? 
    
    filename = "data1.wav"
    y, sr = librosa.load(filename) # notsure why sr = RATE/2
    #y = numpydata
    #sr = RATE

    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #^^^^^^^^^Converting to MFCC and reconstructing^^^^^^^^^^^^^^^^^^^^^
    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #calculate mfccs
    Y = librosa.stft(y)#, n_fft=CHUNK, win_length=WINLEN)
    mfccs = librosa.feature.mfcc(y)#, sr=sr)
    
    print griffinlim(mfccs)

    # print "mfccs"
    # print mfccs

    #build reconstruction mappings
    n_mfcc = mfccs.shape[0]
    n_mel = 128
    dctm = librosa.filters.dct(n_mfcc, n_mel)
    n_fft = 2048
    mel_basis = librosa.filters.mel(sr, n_fft)

    #Empirical scaling of channels to get ~flat amplitude mapping.
    bin_scaling = 1.0/np.maximum(0.0005, np.sum(np.dot(mel_basis.T, mel_basis),axis=0))

    #Reconstruct the approximate STFT squared-magnitude from the MFCCs.
    recon_stft = bin_scaling[:, np.newaxis] * np.dot(mel_basis.T, invlogamplitude(np.dot(dctm.T, mfccs)))

    #Impose reconstructed magnitude on white noise STFT.
    excitation = np.random.randn(y.shape[0])
    E = librosa.stft(excitation)
    recon = librosa.istft(E/np.abs(E)*np.sqrt(recon_stft))

    #Listen to the reconstruction.
    wav.write('data1mfcc.wav', sr, recon)
    Audio(recon, rate=sr)
    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    numpydata = np.reshape(numpydata, (len(numpydata)/CHANNELS, CHANNELS))    
    #wav.write(WAVE_OUTPUT_FILENAME, RATE, numpydata)

    
    # Audio(recon, rate=sr)

    # print numpydata
    # print len(numpydata)
    # audio_mfcc = mfcc(numpydata, samplerate=RATE, nfft=CHUNK, winlen=WINLEN).tolist()[0]
    # print audio_mfcc
    # audio_mfcc = np.reshape(audio_mfcc, (13, len(audio_mfcc)/13))

    # fig, ax = plt.subplots()
    # mfcc_data= np.swapaxes(audio_mfcc, 0 ,1)
    # cax = ax.imshow(mfcc_data, interpolation='nearest', cmap=cm.coolwarm, origin='lower')
    # ax.set_title('MFCC')
    # plt.show()

    # mfccdata = np.hstack(mfcc_store)
    # mfccdata = np.reshape(mfcc_store, (len(numpydata)/CHANNELS, CHANNELS))    
    # wav.write('data1mfcc.wav', RATE, numpydata)

    #NEED MFCC on each Chunk
    #numpydada is a sound samples with 2 columns (2 channels)
    # audio_mfcc = mfcc(numpydata, samplerate=RATE, nfft=CHUNK, winlen=WINLEN).tolist()[0]
 
    # print audio_mfcc
    # print "\n"

    # plt.plot(mfcc_store)
    # plt.show()
    # print audio_mfcc_array
    # print "\n"

def griffinlim(spectrogram, n_iter = 100, window = 'hann', n_fft = 2048, hop_length = -1, verbose = False):
    if hop_length == -1:
        hop_length = n_fft // 4

    angles = np.exp(2j * np.pi * np.random.rand(*spectrogram.shape))

    t = tqdm(range(n_iter), ncols=100, mininterval=2.0, disable=not verbose)
    for i in t:
        full = np.abs(spectrogram).astype(np.complex) * angles
        inverse = librosa.istft(full, hop_length = hop_length, window = window)
        rebuilt = librosa.stft(inverse, n_fft = n_fft, hop_length = hop_length, window = window)
        angles = np.exp(1j * np.angle(rebuilt))

        if verbose:
            diff = np.abs(spectrogram) - np.abs(rebuilt)
            t.set_postfix(loss=np.linalg.norm(diff, 'fro'))

    full = np.abs(spectrogram).astype(np.complex) * angles
    inverse = librosa.istft(full, hop_length = hop_length, window = window)

    return inverse

def main():
    rospy.init_node(PKG)
    audio_creator()

if __name__ == '__main__':
    main()


