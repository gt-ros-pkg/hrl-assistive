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

# system library
import time, sys

# ROS library
import rospy, roslib
from roslib import message

# HRL library
from hrl_srvs.srv import String_String, String_StringRequest

from tf import TransformListener
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import multiprocessing
import numpy as np

try :
    import sensor_msgs.point_cloud2 as pc2
except:
    import point_cloud2 as pc2

class ArmReacherClient:
    def __init__(self):
        self.tf = TransformListener()

        self.points3D = None
        self.highestBowlPoint = None
        self.initialized = False

        # ROS publisher for data points
        self.publisher = rospy.Publisher('visualization_marker', Marker)

        # Connect to point cloud from Kinect
        self.cloudSub = rospy.Subscriber('/head_mount_kinect/depth_registered/points', PointCloud2, self.cloudCallback)
        print 'Connected to Kinect depth'

        # Connect to both PR2 arms
        rospy.init_node('feed_client')
        print 'waiting for /arm_reach_enable'
        rospy.wait_for_service("/arm_reach_enable")
        self.armReachActionLeft  = rospy.ServiceProxy('/arm_reach_enable', String_String)
        print 'waiting for /right/arm_reach_enable'
        self.armReachActionRight = rospy.ServiceProxy('/right/arm_reach_enable', String_String)
        print 'Connected to both services'

    def cancel(self):
        self.cloudSub.unregister()

    def initScooping(self):
        print 'Initializing arm joints for scooping'
        leftProc = multiprocessing.Process(target=self.armReachLeft, args=('initScooping1',))
        rightProc = multiprocessing.Process(target=self.armReachRight, args=('initScooping1',))
        print 'Beginning - left arm init #1'
        t1 = time.time()
        leftProc.start()
        print 'Beginning - right arm init #1'
        t2 = time.time()
        rightProc.start()
        leftProc.join()
        print 'Completed - left arm init #1, time:', time.time() - t1
        rightProc.join()
        print 'Completed - right arm init #1, time:', time.time() - t2

        print 'Beginning - right arm (bowl holding arm) init #2'
        t1 = time.time()
        self.armReachActionRight('initScooping2')
        print 'Completed - right arm (bowl holding arm) init #2, time:', time.time() - t1
        print 'Beginning - getBowPos'
        t1 = time.time()
        self.armReachActionLeft('getBowlPos')
        print 'Completed - getBowPos, time:', time.time() - t1
        print 'Beginning - lookAtBowl'
        t1 = time.time()
        self.armReachActionLeft('lookAtBowl')
        print 'Completed - lookAtBowl, time:', time.time() - t1

        self.initialized = True

    def run(self):
        # Don't run unless highest point in bowl has been obtained, use: "while not run(): pass"
        if self.highestBowlPoint is None:
            return False

        time.sleep(5)

        print 'Beginning to scoop!'
        print self.armReachActionLeft('initScooping2')
        print self.armReachActionLeft('runScooping')

        return True

    def armReachLeft(self, action):
        self.armReachActionLeft(action)

    def armReachRight(self, action):
        self.armReachActionRight(action)

    def getHeadPos(self):
        print self.armReachActionLeft('lookAtMouth')
        print self.armReachActionLeft('getHeadPos')

    def cloudCallback(self, data):
        # print 'Time between cloud calls:', time.time() - self.cloudTime
        # startTime = time.time()

        # Wait to obtain cloud data until after arms have been initialized
        if not self.initialized:
            return

        pointCloud = data

        bowlPos = self.armReachActionLeft('returnBowlPos')
        print 'Received Bowl Position:', bowlPos
        bowlXY = np.array([bowlPos.p.x(), bowlPos.p.y()])
        bowlZ = bowlPos.p.z()

        pointCloud = self.tf.transformPointCloud(self, 'r_gripper_tool_frame', pointCloud)

        # points2D = [[x, y] for y in xrange(lowY, highY) for x in xrange(lowX, highX)]
        try:
            points3D = pc2.read_points(pointCloud, field_names=('x', 'y', 'z'), skip_nans=True) # uvs=points2D
        except:
            print 'Unable to unpack from PointCloud2!'
            return

        # X, Y Positions must be within a radius of 5 cm from the bowl center, and Z positions must be within 4 cm of center
        # TODO: This could be sped up by restricting to a 2D window in the point cloud (uvs=points2D)
        self.points3D = np.array([point for point in points3D if np.linalg.norm(bowlXY - np.array(point[:2])) < 0.05 and abs(bowlZ - point[2]) < 0.04])

        # Find the highest point (based on Z axis value) that is within the bowl
        maxIndex = self.points3D[:, 2].argmax()
        self.highestBowlPoint = self.points3D[maxIndex]

        # Publish highest bowl point and all 3D points in bowl
        self.publishPoints('highestPoint', [self.highestBowlPoint], g=1.0)
        self.publishPoints('allPoints', self.points3D, g=0.6, b=1.0)

    def publishPoints(self, name, points, size=0.01, r=0.0, g=0.0, b=0.0, a=1.0):
        marker = Marker()
        marker.header.frame_id = 'r_gripper_tool_frame'
        marker.ns = name
        marker.type = marker.POINTS
        marker.action = marker.ADD
        marker.scale.x = size
        marker.scale.y = size
        marker.color.a = a
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        for point in points:
            p = Point()
            p.x, p.y, p.z = point
            marker.points.append(p)
        self.publisher.publish(marker)


if __name__ == '__main__':
    client = ArmReacherClient()
    client.initScooping()
    print 'Beginning - left arm init #2'
    t1 = time.time()
    client.armReachActionLeft('initScooping2')
    print 'Completed - left arm init #2, time:', time.time() - t1
    print 'Beginning - scooping'
    t1 = time.time()
    client.armReachActionLeft('runScoopingCenter')
    # client.armReachActionLeft('runScoopingRight')
    print 'Completed - scooping, time:', time.time() - t1

    # while not client.run():
    #     pass
    client.cancel()

    # if True:
    #     print 'Detecting AR tag on head'
    #     print armReachActionLeft('lookAtMouth')
    #     print armReachActionLeft("getHeadPos")
    #     sys.exit()
    #
    #
    # ## Scooping -----------------------------------
    # if True:
    #     # print armReachActionLeft("initScooping1")
    #     # print armReachActionRight("initScooping1")
    #
    #     pass
    #     # sys.exit()
    #
    # ## Feeding -----------------------------------
    # if True:
    #     leftProc = multiprocessing.Process(target=armReachLeft, args=('initFeeding',))
    #     rightProc = multiprocessing.Process(target=armReachRight, args=('initFeeding',))
    #     # headProc = multiprocessing.Process(target=getHeadPos) # TODO: Make a new service this!
    #     leftProc.start()
    #     rightProc.start()
    #     # headProc.start()
    #     leftProc.join()
    #     rightProc.join()
    #     # headProc.join()
    #
    #     # print 'Initializing both arms for feeding and detecting the user\'s head'
    #     # print armReachActionLeft("initFeeding")
    #     # print armReachActionRight("initFeeding")
    #
    #     # print "Detect ar tag on the head"
    #     print armReachActionLeft('lookAtMouth')
    #     print armReachActionLeft("getHeadPos")
    #     # ut.get_keystroke('Hit a key to proceed next')
    #
    #     print "Running feeding!"
    #     print armReachActionLeft("runFeeding1")
    #     print armReachActionLeft("runFeeding2")
    #
    # cloudSub.unregister()

