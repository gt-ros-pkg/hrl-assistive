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
roslib.load_manifest('hrl_manipulation_task')
import os, threading, copy

# util
import numpy as np
import PyKDL

# ROS message
import tf
from std_msgs.msg import Bool, Empty, Int32, Int64, Float32, Float64, String
from hark_msgs.msg import HarkSource, HarkSrcFFT, HarkSrcFeature

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

        self.init_time = 0.0        
        self.power = None
        self.azimuth = None
        self.feature = None
        self.recog_cmd = None
        
        # Declare containers
        self.time_data = []
        self.audio_feature = None
        self.audio_power   = []
        self.audio_azimuth = []
        self.audio_cmd     = []
        
        self.src_feature_lock = threading.RLock()
        self.recog_cmd_lock = threading.RLock()
        
        self.initParams()
        self.initComms()
        self.getHeadFrame()

        if self.verbose: print "Kinect Audio>> initialization complete"
        
    def initComms(self):
        '''
        Initialize pusblishers and subscribers
        '''
        if self.verbose: print "Kinect Audio>> Initialized pusblishers and subscribers"
        rospy.Subscriber('HarkSrcFeature/all', HarkSrcFeature, \
                         self.harkSrcFeatureCallback)
        rospy.Subscriber('julius_recog_cmd', String, self.harkCmdCallback)

        # tf
        try:
            self.tf_lstnr = tf.TransformListener()
        except rospy.ServiceException, e:
            rospy.loginfo("ServiceException caught while instantiating a TF listener. Seems to be normal")
            pass
              

    def initParams(self):
        '''
        Get parameters
        '''
        self.torso_frame = 'torso_lift_link'
        self.head_frame = rospy.get_param('/hrl_manipulation_task/head_audio_frame')
        

    def harkSrcFeatureCallback(self, msg):
        '''
        Get MFCC features from hark. 
        '''
        with self.src_feature_lock:
            self.count_feature     = msg.count
            self.exist_feature_num = msg.exist_src_num
            self.src_feature       = msg.src
            
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

                self.power   = self.src_feature[i].power #float32
                self.azimuth = self.src_feature[i].azimuth #float32
                self.length  = self.src_feature[i].length
                self.feature = np.array([self.src_feature[i].featuredata]).T #float32 list

            if self.exist_feature_num > 1:
                if self.verbose: print "Too many number of sound sources!!"
        

    def harkCmdCallback(self, msg):
        '''
        Get recognized cmd
        '''
        with self.recog_cmd_lock:
            self.recog_cmd = msg.data


    def getHeadFrame(self):

        try:
            self.tf_lstnr.waitForTransform(self.torso_frame, self.head_frame, rospy.Time(0), \
                                           rospy.Duration(5.0))
        except:
            self.tf_lstnr.waitForTransform(self.torso_frame, self.head_frame, rospy.Time(0), \
                                           rospy.Duration(5.0))
                                           
        [self.head_pos, self.head_orient_quat] = \
          self.tf_lstnr.lookupTransform(self.torso_frame, self.head_frame, rospy.Time(0))  


        rot = PyKDL.Rotation.Quaternion(self.head_orient_quat[0], 
                                        self.head_orient_quat[1], 
                                        self.head_orient_quat[2], 
                                        self.head_orient_quat[3])        

        cur_x   = rot.UnitX()
        x = PyKDL.Vector(1.0, 0.0, 0.0)
        y = PyKDL.Vector(0.0, 1.0, 0.0)
        
        head_dir = PyKDL.Vector(PyKDL.dot(cur_x,x), PyKDL.dot(cur_x,y), 0.0)
        head_dir.Normalize()

        if (head_dir * x).z() > 0.0: sign = -1.0
        else: sign = 1.0
        
        self.base_azimuth = np.arccos(PyKDL.dot(head_dir, x)) * sign * 180.0/np.pi
        if self.verbose: print "Computed head azimuth: ", self.base_azimuth

            
    def run(self):
        """Overloaded Thread.run, runs the update
        method once per every xx milliseconds."""
        while not self.cancelled:
            if self.isReset:
                self.time_data.append(rospy.get_time() - self.init_time)

                with self.src_feature_lock:                
                    if self.audio_feature is None: self.audio_feature = self.feature
                    else: self.audio_feature = np.hstack([ self.audio_feature, self.feature ])
                    self.audio_power.append(self.power)
                    self.audio_azimuth.append(self.azimuth+self.base_azimuth)

                with self.recog_cmd_lock:                    
                    self.audio_cmd.append(self.recog_cmd)

        
    def cancel(self):
        """End this timer thread"""
        self.cancelled = True
        self.isReset = False


    def reset(self, init_time):
        self.getHeadFrame()
        self.init_time = init_time

        # Reset containers
        self.time_data = []
        self.audio_feature = None
        self.audio_power   = []        
        self.audio_azimuth = []
        self.audio_cmd     = []
        
        self.isReset = True

        
    def isReady(self):
        if self.azimuth is not None and self.power is not None and \
          self.feature is not None:
          return True
        else:
          return False
