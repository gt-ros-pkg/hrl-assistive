#!/usr/bin/env python
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

# system
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

#Globals (static)
count = 0
data_store = []

#Globals (constants)
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 20
WAVE_OUTPUT_FILENAME = "out.wav"

def audio_callback(data):
    #collect data and generate a wave file --see if it works
    #one frame per one callback 
    #if it doesn't then do baremetal publish and try -- should work
    global count
    global data_store

    #print data
    #tmplen = 44100/1024 #data.audio_data.getnframes()
    if count < (RATE / CHUNK * RECORD_SECONDS):
        # (1) collect and join frames1
        print("(2) Frame collect Started")
        #data_store.append(data.audio_data)
        data_store.append(np.array(data.audio_data, dtype=np.int16))
        count = count + 1    

    elif count == (RATE / CHUNK * RECORD_SECONDS):
        print("(2) Frame join finished")
        print data_store
        numpydata = np.hstack(data_store)
        numpydata = np.reshape(numpydata, (len(numpydata)/CHANNELS, CHANNELS))    

        wav.write(WAVE_OUTPUT_FILENAME, RATE, numpydata)
        count = count + 1

        print numpydata


## 50 seconds -> packed into 20 seconds because of communication & data processing latency

def main():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber('/hrl_manipulation_task/wrist_audio', audio, audio_callback)
    rospy.spin()

if __name__ == '__main__':
    main()
