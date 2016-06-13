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

#  \author Hokeun Kim  (Healthcare Robotics Lab, Georgia Tech.)
#  \author Daehyung Park (Healthcare Robotics Lab, Georgia Tech.)

import rospy, roslib
import numpy as np
import os, threading, copy

import PyKDL

from tf_conversions import posemath
from hrl_lib import quaternion as qt
import hrl_lib.util as ut
import hrl_lib.circular_buffer as cb

from ar_track_alvar_msgs.msg import AlvarMarkers
import geometry_msgs
from geometry_msgs.msg import PoseStamped, PointStamped, PoseArray

QUEUE_SIZE = 10

class LandmarkMouthDetector:

    def __init__(self):

        print "Start LandmarkMouthConversion"

        #frame for class to read
        self.mouth_frame  = None

        #circular buffer for moving average
        self.hist_size = 10
        self.mouth_pos_buf  = cb.CircularBuffer(self.hist_size, (3,))
        self.mouth_quat_buf = cb.CircularBuffer(self.hist_size, (4,))               

        #Subscribers
        rospy.Subscriber("/kinect_pose/mouth", PoseStamped, self.KinectMouthCallback)

        #for locking info during update
        self.frame_lock = threading.RLock()


    def KinectMouthCallback(self, msg):
        with self.frame_lock:
            mouth_frame = posemath.fromMsg(msg.pose)

            if mouth_frame.p.Norm() > 2.0: 
                print "Detected mouth is located at too far location."
                return

            cur_p = np.array([mouth_frame.p[0], mouth_frame.p[1], mouth_frame.p[2]])
            cur_q = np.array([mouth_frame.M.GetQuaternion()[0], 
                              mouth_frame.M.GetQuaternion()[1], 
                              mouth_frame.M.GetQuaternion()[2],
                              mouth_frame.M.GetQuaternion()[3]])

            if len(self.mouth_quat_buf) < 1:
                self.mouth_pos_buf.append( cur_p )
                self.mouth_quat_buf.append( cur_q )
            else:
                first_p = self.mouth_pos_buf[0]
                first_q = self.mouth_quat_buf[0]

                # check close quaternion and inverse
                if np.dot(cur_q, first_q) < 0.0:
                    cur_q *= -1.0
                    
                self.mouth_pos_buf.append( cur_p )
                self.mouth_quat_buf.append( cur_q )
                            
                        
            positions  = self.mouth_pos_buf.get_array()
            quaternions  = self.mouth_quat_buf.get_array() 

            p = None
            q = None
            if False:
                # Moving average
                p = np.sum(positions, axis=0)                    
                p /= float(len(positions))
                    
                q = np.sum(quaternions, axis=0)
                q /= float(len(quaternions))
            else:
                # median
                positions = np.sort(positions, axis=0)
                p = positions[len(positions)/2]
                
                quaternions = np.sort(quaternions, axis=0)
                q = quaternions[len(quaternions)/2]
                        
            mouth_frame.p[0] = p[0]
            mouth_frame.p[1] = p[1]
            mouth_frame.p[2] = p[2]                    
            mouth_frame.M = PyKDL.Rotation.Quaternion(q[0], q[1], q[2], q[3])
            
            self.mouth_frame = mouth_frame
            print self.mouth_frame


        
if __name__ == '__main__':
    rospy.init_node('kinect_mouth_estimation')

    import optparse
    p = optparse.OptionParser()
    p.add_option('--renew', action='store_true', dest='bRenew',
                 default=False, help='Renew frame pickle files.')
    p.add_option('--virtual', '--v', action='store_true', dest='bVirtual',
                 default=False, help='Send a vitual frame.')
    opt, args = p.parse_args()
    
    lmd = LandmarkMouthDetector()
    
    rate = rospy.Rate(10) # 25Hz, nominally.    
    while not rospy.is_shutdown():
        rate.sleep()


        
        
