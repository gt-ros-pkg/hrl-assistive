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
PKG = 'my_package'
# import roslib; roslib.load_manifest(PKG)
import rosbag
import rospy
import os, copy, sys
import librosa
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

def dataset_creator(): 
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    RECORD_SECONDS = 5 #needs to be dynamic, length of audio data should be equal

    # bag_files = ['data1', 'data2', 'data3', 'data4', 'data5']
    # file_length = [5, 3, 3, 3, 3]

    # bag_files = {'data1':5, 'data2':3, 'data3':3, 'data4':3, 'data5':3, 'data13':3}

    bagfile = 'data1'
    wavfile = 'data1'
    # for bagfile in bag_files:
    static_ar = []
    dynamic_ar = []
    audio_store = []
    audio_samples = []
    time_store = []
    #Read in all the messages into buffers
    for topic, msg, t in rosbag.Bag('./bagfiles/'+bagfile+'.bag').read_messages():
        #print msg
        if msg._type == 'hrl_anomaly_detection/audio':
            audio_store.append(np.array(msg.audio_data, dtype=np.int16))
        elif msg._type == 'visualization_msgs/Marker':
            if msg.id == 0: #id 0 = static
                static_ar.append([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
            elif msg.id == 9: #id 9 = dynamic
                dynamic_ar.append([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        time_store.append(t)
    static_ar = np.array(static_ar, dtype=np.float64)
    dynamic_ar = np.array(dynamic_ar, dtype=np.float64)

    print len(time_store)
    print len(static_ar)
    print len(dynamic_ar)
    print len(audio_store[0])

    #********** AR TAG processing *********#
    # if len(static_ar)<len(dynamic_ar):
    #     ar_length = len(static_ar) 
    # else:
    #     ar_length = len(dynamic_ar)

    # static_ar = static_ar[0:ar_length]
    # dynamic_ar = dynamic_ar[0:ar_length]

    # relative_position = []
    # for i in range(ar_length):
    #     relative_position.append(static_ar[i] - dynamic_ar[i])
    # relative_position = np.array(relative_position)
    # print relative_position.shape
    # np.savetxt('./csv/' + bagfile + '.txt', relative_position)
    # # a = np.loadtxt('./csv/' + bagfile + '.txt')
    # # print a

    # ##**** Audio Processing ****##
    # #copy the frame and insert to lengthen
    # data_store_long = []
    # baglen = len(audio_store)
    # RECORD_SECONDS = bag_files[bagfile]
    # num_frames = RATE/CHUNK * RECORD_SECONDS
    # recovered_len = num_frames/baglen

    # for frame in audio_store:
    #     for i in range(0, recovered_len): 
    #         data_store_long.append(frame)

    # #print data_store_long
    # numpydata = np.hstack(data_store_long)
    # numpydata = np.reshape(numpydata, (len(numpydata)/CHANNELS, CHANNELS))    

    # #***** Note that this is not enough!!! *******#
    # # 1) Must clean up code
    # # 2) Must combine audio_cropper functionality for generating training & prediction data
    # # 3) save CSV

    # #print numpydata
    # wav.write('./sounds/original_downsampled/'+bagfile+'.wav', RATE, numpydata)

    # y,sr = librosa.load('./sounds/original_downsampled/'+bagfile+'.wav', mono=True)
    #print y
    #print np.max(y)

def main():
    rospy.init_node(PKG)
    dataset_creator()

if __name__ == '__main__':
    main()


