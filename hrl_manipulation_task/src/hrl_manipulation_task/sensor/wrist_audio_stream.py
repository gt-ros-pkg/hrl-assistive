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
from features import mfcc

FRAME_SIZE = 4096 # frame per buffer
RATE       = 44100 # sampling rate
CHANNEL    = 2 # number of channels
FORMAT     = pyaudio.paInt16
MAX_INT    = 32768.0
WINLEN     = float(RATE)/float(FRAME_SIZE)

class wrist_audio(threading.Thread):

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
        
        # Declare containers
        self.time_data = []
        self.audio_data = []
        
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

        
        self.stream = self.p.open(format=FORMAT, channels=CHANNEL, rate=RATE, input=True, frames_per_buffer=FRAME_SIZE, input_device_index=deviceIndex)
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
        self.audio_data = []
        
        self.isReset = True
        self.cancelled = False

        
    def isReady(self):

        try:
            data = self.stream.read(FRAME_SIZE)
            return True
        except:
            return False
        ## if self.power is not None:
        ##   return True
        ## else:
        ##   return False


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


    def log_start(self):
        self.logger = threading.Thread(target=self.run)
        self.logger.setDaemon(True)
        self.logger.start()
        
    def run(self):

        while not rospy.is_shutdown():
            try:
                data       = self.stream.read(FRAME_SIZE)
            except:
                print "Audio read failure due to input over flow. Please, adjust frame_size(chunk size)"
                ## data       = self.stream.read(self.FRAME_SIZE)
                continue

            self.time_data.append(rospy.get_rostime().to_sec())
            self.audio_data.append(data)

            if self.cancelled: break            
                
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
                audio_time, rms, mfcc = self.get_data()
                ## rms  = self.get_rms(data)

                ## audio_data = np.fromstring(data, self.FORMAT)
                ## audio_FFT  = np.fft.fft(float(audio_data) / float(self.MAX_INT))  #normalization & FFT
                ## audio_data = np.fromstring(data, np.int16)
                ## audio_FFT  = np.fft.fft(audio_data / float(self.MAX_INT))  #normalization & FFT

                ## mfcc_feat = mfcc(audio_data, samplerate=48000, nfft=self.FRAME_SIZE, winlen=48000./8192.0)

                print audio_time, sys.getsizeof(rms), sys.getsizeof(mfcc), np.shape(mfcc)
                ## print mfcc_feat
                ## print len(data), rms, sys.getsizeof(data), sys.getsizeof(rms), sys.getsizeof(audio_FFT), np.shape(audio_FFT[-1])
                ## self.rms_buf.append(rms)
                ## print "==> ", rms_buf.get_array()
                
                ## del ax.collections[:] 
                ## ax.scatter( self.rms_buf.get_array() )
                ## plt.draw()
                
            ## rate.sleep()

    def get_data(self):
        
        try:
            audio_data = self.stream.read(FRAME_SIZE)
        except:
            print "Audio read failure due to input over flow. Please, adjust frame_size(chunk size)"
            print "If you are running record_data.py, please ignore this message since it is just one time warning by delay"
            audio_data = []
            ## audio_data = self.stream.read(FRAME_SIZE)
            ## audio_data = np.fromstring(data, np.int16)
            
        audio_time = rospy.get_rostime().to_sec()

        return audio_time, audio_data

            
    def get_feature(self, data):
        audio_rms = self.get_rms(data)
        num_data = np.fromstring(data, np.int16)
        audio_mfcc = mfcc(num_data, samplerate=RATE, nfft=FRAME_SIZE, winlen=WINLEN).tolist()[0]        
        return audio_rms, audio_mfcc
    

    def get_features(self, audio_data):
        audio_rms = []
        audio_mfcc = []
        for data in audio_data:
            audio_rms.append( self.get_rms(data) )
            num_data = np.fromstring(data, np.int16)
            audio_mfcc.append(mfcc(num_data, samplerate=RATE, nfft=FRAME_SIZE, \
                                   winlen=WINLEN).tolist()[0])
        ## num_data = np.fromstring(audio_data, np.int16)
        ## audio_mfcc = mfcc(num_data, samplerate=RATE, nfft=FRAME_SIZE, winlen=WINLEN)
        ## print np.shape(audio_mfcc)

        return audio_rms, audio_mfcc

            
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
            n = sample / MAX_INT
            sum_squares += n*n

        return math.sqrt( sum_squares / count )       
            


if __name__ == '__main__':
    rospy.init_node('wrist_audio')

    kv = wrist_audio()
    kv.test()



        



        
    ## def get_stream_data(self):
    ##     try:
    ##         data       = self.stream.read(self.FRAME_SIZE)
    ##     except:
    ##         ## print "Audio read failure due to input over flow. Please, adjust frame_size(chunk size)"
    ##         data       = self.stream.read(self.FRAME_SIZE)
        
    ##     decoded = np.fromstring(data, 'Int16')
    ##     return decoded
        

