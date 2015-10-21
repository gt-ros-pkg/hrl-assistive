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

# msg
from pr2_msgs.msg import PressureState

class pps_skin(threading.Thread):

    def __init__(self, verbose=False, viz=False):
        super(pps_skin, self).__init__()        
        self.daemon = True
        self.cancelled = False
        self.isReset = False
        self.verbose = verbose
        self.viz     = viz

        self.init_time = 0.0        
        self.counter = 0
        self.counter_prev = 0


        # instant data
        self.l_fingertip = None
        self.r_fingertip = None
        
        # Declare containers        
        self.time_data    = []
        self.pps_skin_left  = None
        self.pps_skin_right = None

        self.lock = threading.Lock()        

        self.initParams()
        self.initComms()
        if self.verbose: print "pps_skin>> initialization complete"

    def initComms(self):
        '''
        Initialize pusblishers and subscribers
        '''
        if self.verbose: print "pps_skin>> Initialize pusblishers and subscribers"

        input_topic = '/pressure/'+self.pps_arm+'_gripper_motor'
        rospy.Subscriber(input_topic, PressureState, self.ppsCallback)

    def initParams(self):
        '''
        Get parameters
        '''
        self.pps_arm = rospy.get_param('hrl_manipulation_task/arm')

            
    def ppsCallback(self, msg):
        with self.lock:
            self.l_fingertip = copy.copy(msg.l_finger_tip)
            self.r_fingertip = copy.copy(msg.r_finger_tip)
            self.counter += 1

    def run(self):
        """Overloaded Thread.run, runs the update
        method once per every xx milliseconds."""
        while not self.cancelled:
            if self.isReset:

                if self.counter > self.counter_prev:
                    self.counter_prev = self.counter

                    self.lock.acquire()                           
                    l = self.l_fingertip
                    r = self.r_fingertip 

                    #front, bottom, top is order of taxels
                    data_left = np.array([[l[3]+l[4], l[5]+l[6], l[1]+l[2]]]).T
                    data_right = np.array([[r[3]+r[4], r[1]+r[2], r[5]+r[6]]]).T

                    self.time_data.append(rospy.get_time() - self.init_time)
                    if self.pps_skin_left is None:
                        self.pps_skin_left  = data_left
                        self.pps_skin_right = data_right
                    else:
                        self.pps_skin_left  = np.hstack([self.pps_skin_left, data_left])
                        self.pps_skin_right = np.hstack([self.pps_skin_right, data_right])
                                        
                    self.lock.release()

    def cancel(self):
        """End this timer thread"""
        self.cancelled = True
        self.isReset = False

    def reset(self, init_time):
        self.init_time = init_time
        self.isReset = True

        # Reset containers
        self.time_data    = []
        self.pps_skin_left  = None
        self.pps_skin_right = None

        self.counter = 0
        self.counter_prev = 0
        

    def isReady(self):
        if self.l_fingertip is not None and self.r_fingertip is not None:
          return True
        else:
          return False
        
        
            
