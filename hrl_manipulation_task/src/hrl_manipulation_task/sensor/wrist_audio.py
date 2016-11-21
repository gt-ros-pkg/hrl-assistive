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
    def __init__(self, verbose=False):
        self.isReset = False
        self.verbose = verbose
        
        self.enable_log = False
        self.init_time = 0.0

        # instant data
        self.time  = None
        self.audio_rms     = None
        self.audio_azimuth = None
        self.audio_mfcc = None
        self.audio_data = None
        
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
        self.audio_rms     = msg.audio_rms
        self.audio_azimuth = msg.audio_azimuth
        self.audio_mfcc    = msg.audio_mfcc
        self.audio_data    = msg.audio_data
        
    
    def reset(self, init_time):
        self.init_time = init_time
        self.isReset = True

        
    def isReady(self):
        if self.audio_rms is not None or self.audio_data is not None:
          return True
        else:
          return False


