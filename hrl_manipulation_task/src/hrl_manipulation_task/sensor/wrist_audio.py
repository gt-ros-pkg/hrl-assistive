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

from hrl_anomaly_detection.msg import audio

# util
import numpy as np
import math
import pyaudio
import struct
try:
    from features import mfcc
except:
    from python_speech_features import mfcc
    

class wrist_audio():
    ## ## FRAME_SIZE = 4096 #8192 # frame per buffer
    ## FRAME_SIZE = 4096 # frame per buffer
    ## RATE       = 44100 # sampling rate
    ## CHANNEL    = 2 # number of channels
    ## FORMAT     = pyaudio.paInt16
    ## MAX_INT    = 32768.0
    ## WINLEN     = float(RATE)/float(FRAME_SIZE)

    def __init__(self, verbose=False):
        ## super(wrist_audio, self).__init__()        
        ## self.daemon = True
        ## self.cancelled = False
        self.isReset = False
        self.verbose = verbose
        
        self.enable_log = False
        self.init_time = 0.0

        # instant data
        self.time  = None
        self.audio_rms  = None
        self.audio_mfcc = None
        self.audio_data = None
        
        # Declare containers
        
        self.lock = threading.RLock()

        self.initParams()
        self.initComms()

        if self.verbose: print "Wrist Audio>> initialization complete"
        
    def initComms(self):
        '''
        Initialize pusblishers and subscribers
        '''
        if self.verbose: print "Wrist Audio>> Initialized pusblishers and subscribers"
            
        rospy.Subscriber("/hrl_manipulation_task/wrist_audio", audio, self.audioCallback)
            
            
    def initParams(self):
        '''
        Get parameters
        '''
        return


    def audioCallback(self, msg):
        
        time_stamp = msg.header.stamp
        self.time  = time_stamp.to_sec()
        self.audio_rms  = msg.audio_rms
        self.audio_mfcc = msg.audio_mfcc
        self.audio_data = msg.audio_data
        
    
    def reset(self, init_time):
        self.init_time = init_time
        self.isReset = True

        
    def isReady(self):
        if self.audio_rms is not None:
          return True
        else:
          return False




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



if __name__ == '__main__':
    rospy.init_node('wrist_audio')

    kv = wrist_audio()
    kv.test()



        
