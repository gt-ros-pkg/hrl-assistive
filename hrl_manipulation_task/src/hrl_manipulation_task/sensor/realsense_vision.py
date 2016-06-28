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
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

# msg
from geometry_msgs.msg import PoseStamped

class realsense_vision():
    def __init__(self, verbose=False):

        self.isReset = False
        self.verbose = verbose
       
        # instant data
        self.time     = None
        
        # Declare containers        
        self.landmark_pos  = None
        self.landmark_quat = None
        
        self.lock = threading.RLock()
        
        self.initParams()
        self.initComms()
        if self.verbose: print "Realsense Vision>> initialization complete"
        
    def initComms(self):
        '''
        Initialize pusblishers and subscribers
        '''
        if self.verbose: print "Realsense Vision>> Initialized pusblishers and subscribers"
        rospy.Subscriber('/hrl_manipulation_task/mouth_pose', PoseStamped, self.mouthPoseCallback)        

    def initParams(self):
        '''
        Get parameters
        '''
        return
    
    def mouthPoseCallback(self, msg):
        time_stamp = msg.header.stamp

        with self.lock:
            self.time = time_stamp.to_sec()             
            self.landmark_pos = np.array([msg.pose.pose.position.x,
                                          msg.pose.pose.position.y,
                                          msg.pose.pose.position.z]).T
            self.landmark_quat = np.array([msg.pose.pose.orientation.x,
                                           msg.pose.pose.orientation.y,
                                           msg.pose.pose.orientation.z,
                                           msg.pose.pose.orientation.w]).T

            if self.verbose: print np.shape(self.landmark_pos)
            
    def test(self, save_pdf=False):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.ion()
        plt.show()        
        
        rate = rospy.Rate(10) # 25Hz, nominally.    
        while not rospy.is_shutdown():
            ## print "running test: ", len(self.centers)
            with self.lock:
                del ax.collections[:] 
                ax.scatter(self.landmark_pos[0], self.landmark_pos[1],\
                           self.landmark_pos[2] )
                ax.set_xlim([0.3, 1.4])
                ax.set_ylim([-0.2, 1.0])
                ax.set_zlim([-0.5, 0.5])
                plt.draw()
                
            rate.sleep()
            
    def reset(self, init_time):
        self.init_time = init_time
        self.isReset = True

        
    def isReady(self):
        if self.landmark_pos is not None:
          return True
        else:
          return False




if __name__ == '__main__':
    rospy.init_node('realsense_vision')

    kv = realsense_vision()
    kv.test(True)


        
