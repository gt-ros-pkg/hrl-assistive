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
from geometry_msgs.msg import WrenchStamped

class tool_ft(threading.Thread):

    def __init__(self, verbose=False):
        super(tool_ft, self).__init__()        
        self.daemon = True
        self.cancelled = False
        self.isReset = False
        self.verbose = verbose

        self.init_time = 0.0        
        self.counter = 0
        self.counter_prev = 0

        self.time       = None
        self.force_raw  = None
        self.torque_raw = None

        # Declare containers        
        self.time_data    = []
        self.force_array  = None
        self.torque_array = None

        self.lock = threading.Lock()        

        self.initParams()
        self.initComms()
        if self.verbose: print "tool_ft>> initialization complete"
        

    def initComms(self):
        '''
        Initialize pusblishers and subscribers
        '''
        if self.verbose: print "tool_ft>> Initialize pusblishers and subscribers"
        self.force_raw_sub = rospy.Subscriber('/netft_data', WrenchStamped, self.force_raw_cb)


    def initParams(self):
        '''
        Get parameters
        '''
        return

    def force_raw_cb(self, msg):
        time_stamp = msg.header.stamp

        self.lock.acquire()
        self.time       = time_stamp.to_sec() - self.init_time
        self.force_raw  = np.array([[msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z]]).T
        self.torque_raw = np.array([[msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z]]).T
        self.counter += 1
        self.lock.release()

        
    ## def run(self):
    ##     """Overloaded Thread.run, runs the update
    ##     method once per every xx milliseconds."""
    ##     rate = rospy.Rate(20)
    ##     while not self.cancelled:
    ##         if self.isReset:

    ##             if self.counter > self.counter_prev:
    ##                 self.counter_prev = self.counter

    ##                 self.lock.acquire()                            
                    
    ##                 self.time_data.append(rospy.get_time() - self.init_time)
    ##                 if self.force_array is None:
    ##                     self.force_array  = self.force_raw
    ##                     self.torque_array = self.torque_raw
    ##                 else:
    ##                     self.force_array = np.hstack([self.force_array, self.force_raw])
    ##                     self.torque_array = np.hstack([self.torque_array, self.torque_raw])
                                        
    ##                 self.lock.release()
    ##         rate.sleep()

    ## def cancel(self):
    ##     """End this timer thread"""
    ##     self.cancelled = True
    ##     self.isReset = False

    def reset(self, init_time):
        self.init_time = init_time
        self.isReset = True

        # Reset containers
        self.time_data    = []
        self.force_array  = None
        self.torque_array = None

        self.counter = 0
        self.counter_prev = 0
        

    def isReady(self):
        if self.force_raw is not None and self.torque_raw is not None:
          return True
        else:
          return False
        
        
