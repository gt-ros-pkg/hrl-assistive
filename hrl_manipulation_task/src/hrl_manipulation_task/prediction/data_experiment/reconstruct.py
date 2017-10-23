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

# To Use
# Run pubPyAudio.py to read and publish data
# Or Run, rosbag record -O file.bag /hrl_manipulation_task/wrist_audio
# Run this Script to reconstruct the Audio Wav from data collected in rosbag


# Start up ROS pieces.
PKG = 'my_package'
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
try:
    from features import mfcc
except:
    from python_speech_features import mfcc
from scipy import signal, fftpack, conj, stats

import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
# from matplotlib import cm

import librosa
import librosa.display
# import IPython.display
# from IPython.lib.display import Audio

def audio_creator(): 
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    RECORD_SECONDS = 3
    WAVE_OUTPUT_FILENAME = "data1.wav"   
    BAG_INPUT_FILENAME = 'data1.bag'

    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
    #^^^^^^^^^^^^^^^^^ Original Audio Reconstruction ^^^^^^^^^^^^^^^^^^^^^^^^^#
    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
    data_store = []
    time_store = []
    for topic, msg, t in rosbag.Bag(BAG_INPUT_FILENAME).read_messages():
        if msg._type == 'hrl_anomaly_detection/audio':
            data_store.append(np.array(msg.audio_data, dtype=np.int16))
            time_store.append(t)

    #copy the frame and insert to lengthen
    data_store_long = []
    baglen = len(data_store)
    num_frames = RATE/CHUNK * RECORD_SECONDS
    recovered_len = num_frames/baglen

    for frame in data_store:
        for i in range(0, recovered_len): ##This happens to work cuz recovered len is 2 and num of channels is 2 ???
            data_store_long.append(frame)

    numpydata = np.hstack(data_store_long)
    numpydata = np.reshape(numpydata, (len(numpydata)/CHANNELS, CHANNELS))    
    wav.write(WAVE_OUTPUT_FILENAME, RATE, numpydata)

    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
    #^^^^^^^^^^^^^^^^^ Original Audio Reconstruction ^^^^^^^^^^^^^^^^^^^^^^^^^#
    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#


    # Generate mfccs from a time series
    #y, sr = librosa.load("data1.wav")
    
    y = numpydata
    sr = RATE/2
    librosa.feature.mfcc(y=y, sr=sr)

    # array([[ -5.229e+02,  -4.944e+02, ...,  -5.229e+02,  -5.229e+02],
    # [  7.105e-15,   3.787e+01, ...,  -7.105e-15,  -7.105e-15],
    # ...,
    # [  1.066e-14,  -7.500e+00, ...,   1.421e-14,   1.421e-14],
    # [  3.109e-14,  -5.058e+00, ...,   2.931e-14,   2.931e-14]])

    # Use a pre-computed log-power Mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    librosa.feature.mfcc(S=librosa.power_to_db(S))
    # array([[ -5.207e+02,  -4.898e+02, ...,  -5.207e+02,  -5.207e+02],
    # [ -2.576e-14,   4.054e+01, ...,  -3.997e-14,  -3.997e-14],
    # ...,
    # [  7.105e-15,  -3.534e+00, ...,   0.000e+00,   0.000e+00],
    # [  3.020e-14,  -2.613e+00, ...,   3.553e-14,   3.553e-14]])

    # Get more components
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=3)
    mfcc_delta = librosa.feature.delta(mfccs, axis=0, order=1)
    
    print len(mfccs[0])
    print len(mfcc_delta)

    # Visualize the MFCC series
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.show()

    wav.write("data1mfcc.wav", sr, mfccs)

def main():
    rospy.init_node(PKG)
    audio_creator()

if __name__ == '__main__':
    main()


