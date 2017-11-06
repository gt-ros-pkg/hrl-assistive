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

def audio_creator(): 
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    RECORD_SECONDS = 3
    WAVE_OUTPUT_FILENAME = "data1.wav"   
    BAG_INPUT_FILENAME = 'data1.bag'

    data_store = []
    time_store = []
    for topic, msg, t in rosbag.Bag(BAG_INPUT_FILENAME).read_messages():
        if msg._type == 'hrl_anomaly_detection/audio':
            data_store.append(np.array(msg.audio_data, dtype=np.int16))
            time_store.append(t)

    print data_store
    print time_store
    print len(data_store)
    print len(time_store)

    #copy the frame and insert to lengthen
    data_store_long = []
    baglen = len(data_store)
    num_frames = RATE/CHUNK * RECORD_SECONDS
    recovered_len = num_frames/baglen

    for frame in data_store:
        for i in range(0, recovered_len): ##This happens to work cuz recovered len is 2 and num of channels is 2 ???
            data_store_long.append(frame)
    # newframe = None
    # for frame in data_store:
    #     newframe = np.repeat(frame, recovered_len)
    #     newframe = np.split(newframe, recovered_len)
    #     for new in newframe:
    #         data_store_long.append(new)

    print data_store_long

    numpydata = np.hstack(data_store_long)
    numpydata = np.reshape(numpydata, (len(numpydata)/CHANNELS, CHANNELS))    

    wav.write(WAVE_OUTPUT_FILENAME, RATE, numpydata)

    print numpydata

def main():
    rospy.init_node(PKG)
    audio_creator()

if __name__ == '__main__':
    main()


