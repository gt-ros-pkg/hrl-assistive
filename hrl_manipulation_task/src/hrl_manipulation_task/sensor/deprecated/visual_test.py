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

#  \author Zackory Erickson (Healthcare Robotics Lab, Georgia Tech.)

# system library
import time
import random
import numpy as np
import multiprocessing

# ROS library
import rospy, roslib
from roslib import message
import PyKDL

# HRL library
from hrl_srvs.srv import String_String, String_StringRequest

import tf
#from tf import TransformListener
from sensor_msgs.msg import PointCloud2, CameraInfo
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, PoseStamped, TransformStamped, PointStamped
from std_msgs.msg import Empty
import image_geometry

try :
    import sensor_msgs.point_cloud2 as pc2
except:
    import point_cloud2 as pc2

class ArmReacherClient:
    def __init__(self, verbose=True):
        rospy.init_node('visual_scooping')
        self.tf = tf.TransformListener()
	self.verbose = verbose

        # Connect to point cloud from Kinect
        self.cloudSub = rospy.Subscriber('/head_mount_kinect/qhd/points', PointCloud2, self.cloudCallback)
        if self.verbose: print 'Connected to Kinect depth'

    def cancel(self):
        self.cloudSub.unregister()
        self.cameraSub.unregister()
        self.bowlSub.unregister()

    def cloudCallback(self, data):
	print 'cloud callback'


if __name__ == '__main__':
    client = ArmReacherClient(verbose=True)
    # client.initialize()

    rate = rospy.Rate(10) # 25Hz, nominally.
    while not rospy.is_shutdown():
        rate.sleep()

    client.cancel()

