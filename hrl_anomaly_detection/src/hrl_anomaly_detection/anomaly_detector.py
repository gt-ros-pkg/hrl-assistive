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
import rospy
import roslib
roslib.load_manifest('hrl_anomaly_detection')
import os, sys, copy
import random

# util
import numpy as np
import hrl_lib.util as ut
from hrl_anomaly_detection.util import *

# learning
from sklearn.decomposition import PCA
from hrl_multimodal_anomaly_detection.hmm import learning_hmm_multi_n as hmm

# msg
from hrl_srvs.srv import Bool_None, Bool_NoneResponse

class anomaly_detector:

    def __init__(self, verbose=False):
        rospy.loginfo('Initializing anomaly detector')

        self.start_detector = False
        
        self.initParams()
        self.initComms()

    def initParams(self):
        '''
        Load feature list
        '''
        
        return
    
    def initComms(self):
        '''
        Subscribe raw data
        '''
        # Publisher
        self.action_interruption_pub = rospy.Publisher('InterruptAction', String)
        
        # Subscriber
        
        # Service
        self.detection_service = rospy.Service('anomaly_detector_enable', Bool_None, self.enablerCallback)
        
        pass

    def enablerCallback(self, msg):

        if msg.data is True: 
            # Start detection thread
            self.run()
        else:
            # Reset detector
            self.reset()
            
        
        return Bool_NoneResponse()

    def reset(self):
        '''
        Reset parameters
        '''
        self.initParams()

    def run(self):
        '''
        Run detector
        '''
        self.action_interruption_pub.publish('anomaly?')
