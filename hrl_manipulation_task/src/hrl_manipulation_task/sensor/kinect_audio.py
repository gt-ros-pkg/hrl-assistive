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
import os, threading, copy

# util
import numpy as np
import PyKDL

# ROS message
import tf
from std_msgs.msg import Bool, Empty, Int32, Int64, Float32, Float64, String
from hark_msgs.msg import HarkSource, HarkSrcFFT, HarkSrcFeature
from pr2_controllers_msgs.msg import JointTrajectoryControllerState

class kinect_audio(threading.Thread):
    CHUNK   = 512 # frame per buffer
    RATE    = 16000 # sampling rate
    CHANNEL = 4 # number of channels
    
    def __init__(self, verbose=False):
        super(kinect_audio, self).__init__()        
        self.daemon = True
        self.cancelled = False
        self.isReset = False
        self.verbose = verbose

        
        self.enable_log = False
        self.init_time = 0.0

        # instant data
        self.time  = None
        self.power = None
        self.azimuth = None
        self.base_azimuth = None
        self.feature = None
        self.recog_cmd = None
        self.head_joints = None
        
        # Declare containers
        self.time_data = []
        self.audio_feature = None
        self.audio_power   = []
        self.audio_azimuth = []
        self.audio_cmd     = []
        self.audio_head_joints = None
        
        self.src_feature_lock = threading.RLock()
        self.recog_cmd_lock = threading.RLock()
        self.head_state_lock = threading.RLock()

        self.initParams()
        self.initComms()

        if self.verbose: print "Kinect Audio>> initialization complete"
        
    def initComms(self):
        '''
        Initialize pusblishers and subscribers
        '''
        if self.verbose: print "Kinect Audio>> Initialized pusblishers and subscribers"
        ## rospy.Subscriber('HarkSrcFeature/all', HarkSrcFeature, \
        ##                  self.harkSrcFeatureCallback)
        rospy.Subscriber('HarkSource/all', HarkSource, self.harkSourceCallback)
        rospy.Subscriber('julius_recog_cmd', String, self.harkCmdCallback)
        rospy.Subscriber('/head_traj_controller/state', JointTrajectoryControllerState, self.headStateCallback)
        

    def initParams(self):
        '''
        Get parameters
        '''
        self.torso_frame = 'torso_lift_link'
        

    def harkSrcFeatureCallback(self, msg):
        '''
        Get MFCC features from hark. 
        '''
        with self.src_feature_lock:
            self.count_feature     = msg.count
            self.exist_feature_num = msg.exist_src_num
            self.src_feature       = msg.src
            time_stamp             = msg.header.stamp

            if self.exist_feature_num > 1:
                if self.verbose: print "Too many number of sound sources!!"
            
            if self.exist_feature_num == 1:
                # select only first one since there is only single source
                i = 0
                
                # save data
                src_id = self.src_feature[i].id

                # Force to use single source id
                src_id = 0

                ## if len(self.showFeatureData.keys()) > 0:
                ##     self.src_feature_cen[i]
                if len(self.src_feature[i].featuredata) < 1: return

                self.time    = time_stamp.to_sec() #- self.init_time
                self.power   = self.src_feature[i].power #float32
                self.azimuth = self.src_feature[i].azimuth + self.base_azimuth #float32
                self.length  = self.src_feature[i].length
                self.feature = np.array([self.src_feature[i].featuredata]).T #float32 list

                if self.enable_log == True:
                    self.time_data.append(self.time)

                    if self.audio_feature is None: self.audio_feature = self.feature
                    else: self.audio_feature = np.hstack([ self.audio_feature, self.feature ])
                    self.audio_power.append(self.power)
                    self.audio_azimuth.append(self.azimuth)
                    self.audio_head_joints = self.head_joints
                    self.audio_cmd.append(self.recog_cmd) # lock??
                

    def harkSourceCallback(self, msg):
        '''
        Get MFCC features from hark. 
        '''
        with self.src_feature_lock:
            self.count_feature     = msg.count
            self.exist_feature_num = msg.exist_src_num
            self.src_feature       = msg.src
            time_stamp             = msg.header.stamp

            if self.exist_feature_num > 1:
                if self.verbose: print "Too many number of sound sources!!"
            
            if self.exist_feature_num == 1:
                # select only first one since there is only single source
                i = 0
                
                # save data
                src_id = self.src_feature[i].id

                # Force to use single source id
                src_id = 0

                self.time    = time_stamp.to_sec() - self.init_time
                self.power   = self.src_feature[i].power if self.src_feature[i].power<50 else self.power
                self.azimuth = self.src_feature[i].azimuth+self.base_azimuth #float32
                ## self.length  = self.src_feature[i].length
                ## self.feature = np.array([self.src_feature[i].featuredata]).T #float32 list

                if self.enable_log == True:
                    self.time_data.append(self.time)

                    ## if self.audio_feature is None: self.audio_feature = self.feature
                    ## else: self.audio_feature = np.hstack([ self.audio_feature, self.feature ])
                    self.audio_power.append(self.power)
                    self.audio_azimuth.append(self.azimuth)
                    self.audio_head_joints = self.head_joints
                    self.audio_cmd.append(self.recog_cmd) # lock??
        

    def harkCmdCallback(self, msg):
        '''
        Get recognized cmd
        '''
        with self.recog_cmd_lock:
            self.recog_cmd = msg.data


    def headStateCallback(self, msg):
        with self.head_state_lock:
            self.head_joints  = msg.actual.positions
            self.base_azimuth = self.head_joints[0] * 180.0/np.pi
            
    ## def run(self):
    ##     """Overloaded Thread.run, runs the update
    ##     method once per every xx milliseconds."""
        
    ##     while not self.cancelled:
    ##         if self.isReset:
    ##             self.time_data.append(rospy.get_rostime().to_sec() - self.init_time)

    ##             with self.src_feature_lock:                
    ##                 if self.audio_feature is None: self.audio_feature = self.feature
    ##                 else: self.audio_feature = np.hstack([ self.audio_feature, self.feature ])
    ##                 self.audio_power.append(self.power)
    ##                 self.audio_azimuth.append(self.azimuth+self.base_azimuth)
    ##                 self.audio_head_joints = self.head_joints

    ##             with self.recog_cmd_lock:                    
    ##                 self.audio_cmd.append(self.recog_cmd)

        
    ## def cancel(self):
    ##     """End this timer thread"""
    ##     self.cancelled = True
    ##     self.isReset = False


    def reset(self, init_time):
        self.init_time = init_time

        # Reset containers
        self.time_data = []
        self.audio_feature = None
        self.audio_power   = []        
        self.audio_azimuth = []
        self.audio_cmd     = []
        self.audio_head_joints = None
        
        self.isReset = True

        
    def isReady(self):
        if self.azimuth is not None and self.power is not None and \
          self.head_joints is not None:
          return True
        else:
          return False






        
