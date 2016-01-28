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
import os, threading, copy, sys

# util
import numpy as np
import math
import pyaudio
import struct

class wrist_audio(threading.Thread):
    FRAME_SIZE = 8192 # frame per buffer
    RATE       = 48000 # sampling rate
    CHANNEL    = 2 # number of channels
    FORMAT     = pyaudio.paInt16
    MAX_INT    = 32768.0

    def __init__(self, verbose=False):
        super(wrist_audio, self).__init__()        
        self.daemon = True
        self.cancelled = False
        self.isReset = False
        self.verbose = verbose
        
        self.enable_log = False
        self.init_time = 0.0

        # instant data
        self.time  = None
        self.power = None
        
        # Declare containers
        self.time_data = []
        self.audio_power   = []
        
        self.lock = threading.RLock()

        self.initParams()
        self.initComms()

        if self.verbose: print "Wrist Audio>> initialization complete"
        
    def initComms(self):
        '''
        Initialize pusblishers and subscribers
        '''
        if self.verbose: print "Wrist Audio>> Initialized pusblishers and subscribers"
            
        self.p = pyaudio.PyAudio()
        deviceIndex = self.find_input_device()
        if self.verbose:
            devInfo = self.p.get_device_info_by_index(deviceIndex)
            print 'Audio device:', deviceIndex
            print 'Sample rate:', devInfo['defaultSampleRate']
            print 'Max input channels:',  devInfo['maxInputChannels']

        
        self.stream = self.p.open(format=self.FORMAT, channels=self.CHANNEL, rate=self.RATE, input=True, frames_per_buffer=self.FRAME_SIZE, input_device_index=deviceIndex)
        self.stream.start_stream()
        

    def initParams(self):
        '''
        Get parameters
        '''
        return

    
    def reset(self, init_time):
        self.init_time = init_time

        # Reset containers
        self.time_data = []
        self.audio_power   = []        
        
        self.isReset = True

        
    def isReady(self):

        if self.power is not None:
          return True
        else:
          return False


    def find_input_device(self):
        device_index = None
        for i in range(self.p.get_device_count()):
            devinfo = self.p.get_device_info_by_index(i)
            print('Device %d: %s'%(i, devinfo['name']))

            for keyword in ['mic', 'input', 'icicle']:
                if keyword in devinfo['name'].lower():
                    print('Found an input: device %d - %s'%(i, devinfo['name']))
                    device_index = i
                    return device_index

        if device_index is None:
            print('No preferred input found; using default input device.')

        return device_index


    def get_data(self):
        try:
            data       = self.stream.read(self.FRAME_SIZE)
            ## audio_FFT  = np.fft.fft(audio_data / float(self.MAX_INT))  #normalization & FFT
            ## audio_FFT  = []
            ## audio_rms  = self.get_rms(data)
        except:
            print "Audio read failure due to input over flow"
            self.stream.stop_stream()
            self.stream.close()
            sys.exit()

        return data #audio_rms, audio_FFT


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


    def test(self):
        
        import hrl_lib.circular_buffer as cb
        self.rms_buf  = cb.CircularBuffer(100, ())
        import matplotlib.pyplot as plt
        
        ## fig = plt.figure()
        ## ax = fig.add_subplot(111)
        ## plt.ion()
        ## plt.show()        
        
        ## rate = rospy.Rate(10) # 25Hz, nominally.    
        while not rospy.is_shutdown():
            ## print "running test: ", len(self.centers)
            with self.lock:
                ## rms, _ = self.get_data()
                ## print rms
                data = self.get_data()
                rms  = self.get_rms(data)

                ## audio_data = np.fromstring(data, self.FORMAT)
                ## audio_FFT  = np.fft.fft(float(audio_data) / float(self.MAX_INT))  #normalization & FFT
                audio_data = np.fromstring(data, np.int16)
                audio_FFT  = np.fft.fft(audio_data / float(self.MAX_INT))  #normalization & FFT
                
                print len(data), rms, sys.getsizeof(data), sys.getsizeof(rms), sys.getsizeof(audio_FFT), np.shape(audio_FFT[-1])
                ## self.rms_buf.append(rms)
                ## print "==> ", rms_buf.get_array()
                
                ## del ax.collections[:] 
                ## ax.scatter( self.rms_buf.get_array() )
                ## plt.draw()
                
            ## rate.sleep()



if __name__ == '__main__':
    rospy.init_node('wrist_audio')

    kv = wrist_audio()
    kv.test()



        
