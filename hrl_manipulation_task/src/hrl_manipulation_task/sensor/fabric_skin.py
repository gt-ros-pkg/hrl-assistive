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

# msg
import hrl_haptic_manipulation_in_clutter_msgs.msg as haptic_msgs
from std_msgs.msg import Empty

class fabric_skin():

    def __init__(self, verbose=False, viz=False):
        self.isReset = False
        self.verbose = verbose
        self.viz     = viz

        self.init_time = 0.0        
        self.counter = 0
        self.counter_prev = 0

        # instant data
        self.time       = None
        self.link_names = None
        self.centers_x  = None
        self.centers_y  = None
        self.centers_z  = None
        self.normals_x  = None
        self.normals_y  = None
        self.normals_z  = None
        self.values_x   = None
        self.values_y   = None
        self.values_z   = None
        
        # Declare containers        
        self.time_data = []

        self.lock = threading.Lock()        

        self.initParams()
        self.initComms()
        if self.verbose: print "fabric_skin>> initialization complete"

    def initComms(self):
        '''
        Initialize pusblishers and subscribers
        '''
        if self.verbose: print "fabric_skin>> Initialize pusblishers and subscribers"

        self.zero_forearm_pub = rospy.Publisher('/pr2_fabric_l_forearm_sensor/zero_sensor', Empty)
        self.zero_upperarm_pub = rospy.Publisher('/pr2_fabric_l_upperarm_sensor/zero_sensor', Empty)

        input_topic = '/pressure/'+self.fabric_arm+'_gripper_motor'
        rospy.Subscriber('haptic_mpc/robot_state', haptic_msgs.RobotHapticState, \
                         self.robotStateCallback)

    def initParams(self):
        '''
        Get parameters
        '''
        self.fabric_arm = rospy.get_param('hrl_manipulation_task/arm')

            
    def robotStateCallback(self, msg):
        time_stamp = msg.header.stamp
        
        with self.lock:
            self.counter += 1

            self.time = time_stamp.to_sec() #- self.init_time
            self.centers_x = []
            self.centers_y = []
            self.centers_z = []
            self.normals_x = []
            self.normals_y = []
            self.normals_z = []
            self.values_x  = []
            self.values_y  = []
            self.values_z  = []
            
            for ta_msg in msg.skins:
                self.centers_x += ta_msg.centers_x
                self.centers_y += ta_msg.centers_y
                self.centers_z += ta_msg.centers_z
                self.normals_x += ta_msg.normals_x
                self.normals_y += ta_msg.normals_y
                self.normals_z += ta_msg.normals_z
                self.values_x  += ta_msg.values_x
                self.values_y  += ta_msg.values_y
                self.values_z  += ta_msg.values_z
                    

    ## def run(self):
    ##     """Overloaded Thread.run, runs the update
    ##     method once per every xx milliseconds."""
    ##     while not self.cancelled:
    ##         if self.isReset:

    ##             if self.counter > self.counter_prev:
    ##                 self.counter_prev = self.counter

    ##                 self.lock.acquire()                           
    ##                 l = self.l_fingertip
    ##                 r = self.r_fingertip 

    ##                 #front, bottom, top is order of taxels
    ##                 data_left = np.array([[l[3]+l[4], l[5]+l[6], l[1]+l[2]]]).T
    ##                 data_right = np.array([[r[3]+r[4], r[1]+r[2], r[5]+r[6]]]).T

    ##                 self.time_data.append(rospy.get_time() - self.init_time)
    ##                 if self.fabric_skin_left is None:
    ##                     self.fabric_skin_left  = data_left
    ##                     self.fabric_skin_right = data_right
    ##                 else:
    ##                     self.fabric_skin_left  = np.hstack([self.fabric_skin_left, data_left])
    ##                     self.fabric_skin_right = np.hstack([self.fabric_skin_right, data_right])
                                        
    ##                 self.lock.release()

    ## def cancel(self):
    ##     """End this timer thread"""
    ##     self.cancelled = True
    ##     self.isReset = False

    def reset(self, init_time):
        
        ## self.zeroSkinHandler()
        ## self.zeroSkinHandler()
        
        self.init_time = init_time
        self.isReset = True

        # Reset containers
        self.time_data    = []

        self.counter = 0
        self.counter_prev = 0
        

    def isReady(self):
        if self.centers_x is not None:
          return True
        else:
          return False
        
        
    ## Zeroes the PR2 skin (eg, to correct for calibration errors).
    def zeroSkinHandler(self):
        ## self.zero_gripper_pub.publish(Empty())
        ## self.zero_gripper_right_link_pub.publish(Empty())
        ## self.zero_gripper_left_link_pub.publish(Empty())
        ## self.zero_gripper_palm_pub.publish(Empty())
        self.zero_forearm_pub.publish(Empty())
        self.zero_upperarm_pub.publish(Empty())
        ## self.zero_pps_left_pub.publish(Empty())
        ## self.zero_pps_right_pub.publish(Empty())

