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

# system
import rospy, roslib
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
from features import mfcc

QUEUE_SIZE = 10

class wrist_audio_collector:
    FRAME_SIZE = 4096 # frame per buffer
    RATE       = 44100 # sampling rate
    CHANNEL    = 2 # number of channels
    FORMAT     = pyaudio.paInt16
    MAX_INT    = 32768.0
    WINLEN     = float(RATE)/float(FRAME_SIZE)

    def __init__(self, verbose=False):
        self.verbose = verbose
        
        ## self.initParams()
        self.initComms()

        if self.verbose: print "Wrist Audio>> initialization complete"

    ## def initParams(self):
    ##     self.audio_data = array.array('h') 
        
    def initComms(self):
        '''
        Initialize pusblishers and subscribers
        '''            
        self.p = pyaudio.PyAudio()
        deviceIndex = self.find_input_device()
        devInfo = self.p.get_device_info_by_index(deviceIndex)
        print 'Audio device:', deviceIndex
        print 'Sample rate:', devInfo['defaultSampleRate']
        print 'Max input channels:',  devInfo['maxInputChannels']
        
        self.stream = self.p.open(format=self.FORMAT, channels=self.CHANNEL, rate=self.RATE, input=True, frames_per_buffer=self.FRAME_SIZE, input_device_index=deviceIndex)
        ## self.stream.start_stream()

        self.audio_pub  = rospy.Publisher("hrl_manipulation_task/wrist_audio", audio, \
                                          queue_size=QUEUE_SIZE, latch=True)
    
    def find_input_device(self):
        device_index = None
        for i in range(self.p.get_device_count()):
            devinfo = self.p.get_device_info_by_index(i)
            print('Device %d: %s'%(i, devinfo['name']))

            for keyword in ['mic', 'input', 'icicle', 'creative']:
                if keyword in devinfo['name'].lower():
                    print('Found an input: device %d - %s'%(i, devinfo['name']))
                    device_index = i
                    return device_index

        if device_index is None:
            print('No preferred input found; using default input device.')

        return device_index

        
    def get_data(self):
        audio_rms = audio_mfcc = 0
        
        try:
            data       = self.stream.read(self.FRAME_SIZE)
            audio_rms  = self.get_rms(data)
            audio_data = np.fromstring(data, np.int16)
            audio_mfcc = mfcc(audio_data, samplerate=self.RATE, nfft=self.FRAME_SIZE, winlen=self.WINLEN).tolist()[0]
        except:
            print "Audio read failure due to input over flow. Please, adjust frame_size(chunk size)"
            print "If you are running record_data.py, please ignore this message since it is just one time warning by delay"
            data       = self.stream.read(self.FRAME_SIZE)
            audio_rms  = self.get_rms(data)
            audio_data = np.fromstring(data, np.int16)
            audio_mfcc = mfcc(audio_data, samplerate=self.RATE, nfft=self.FRAME_SIZE, winlen=self.WINLEN).tolist()[0]
            ## self.stream.stop_stream()
            ## self.stream.close()
            ## sys.exit()
            
        audio_time = rospy.Time.now() #rospy.get_rostime().to_sec()        
        return audio_time, audio_data, audio_rms, audio_mfcc

    def get_data2(self):
        try:
            data       = self.stream.read(self.FRAME_SIZE)
        except:
            print "Audio read failure due to input over flow. Please, adjust frame_size(chunk size)"
            data       = self.stream.read(self.FRAME_SIZE)
        
        decoded = np.fromstring(data, 'Int16')
        return decoded
        
    def get_rms(self, block):
        # Copy from http://stackoverflow.com/questions/4160175/detect-tap-with-pyaudio-from-live-mic

        # RMS amplitude is defined as the square root of the 
        # mean over time of the square of the amplitude.
        # so we need to convert this string of bytes into 
        # a string of 16-bit samples...

        # we will get one short out for each 
        # two chars in the string.
        count = len(block)/2
        format = "%dh"%(count)
        shorts = struct.unpack( format, block )

        # iterate over the block.
        sum_squares = 0.0
        for sample in shorts:
        # sample is a signed short in +/- 32768. 
        # normalize it to 1.0
            n = sample / self.MAX_INT
            sum_squares += n*n

        return math.sqrt( sum_squares / count )        


    def run(self):
        
        ## import hrl_lib.circular_buffer as cb
        ## self.rms_buf  = cb.CircularBuffer(100, (1,))
        ## import matplotlib.pyplot as plt
        
        ## fig = plt.figure()
        ## ax = fig.add_subplot(111)
        ## plt.ion()
        ## plt.show()
        
        msg = audio()
        
        ## rate = rospy.Rate(25) # 25Hz, nominally.    
        while not rospy.is_shutdown():
            ## print "running test: ", len(self.centers)
            audio_time, audio_data, audio_rms, audio_mfcc = self.get_data()
            ## data = self.get_data2()

            msg.header.stamp = audio_time #rospy.Time.now()
            msg.audio_rms  = audio_rms
            msg.audio_mfcc = audio_mfcc
            msg.audio_data = audio_data            
            self.audio_pub.publish(msg)



if __name__ == '__main__':
    rospy.init_node('wrist_audio_publisher')

    kv = wrist_audio_collector()
    kv.run()



        
