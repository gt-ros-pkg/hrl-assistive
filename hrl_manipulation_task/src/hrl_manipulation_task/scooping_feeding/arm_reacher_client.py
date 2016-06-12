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

#  \author Daehyung Park and Zackory Erickson (Healthcare Robotics Lab, Georgia Tech.)

# system library
import time
import random

# ROS library
import rospy, roslib
from roslib import message
import PyKDL

# HRL library
from hrl_srvs.srv import String_String, String_StringRequest

from tf import TransformListener
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, PoseStamped, TransformStamped, PointStamped
import multiprocessing
import numpy as np
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud

try :
    import sensor_msgs.point_cloud2 as pc2
except:
    import point_cloud2 as pc2

class ArmReacherClient:
    def __init__(self, isScooping=True):
        rospy.init_node('feed_client')
        self.tf = TransformListener()

        self.isScooping = isScooping
        self.points3D = None
        self.highestBowlPoint = None
        self.initialized = False
        self.bowlPos = None
        # self.torso_frame = rospy.get_param('haptic_mpc/pr2/torso_frame')

        # ROS publisher for data points
        self.publisher = rospy.Publisher('visualization_marker', Marker)

        # Connect to point cloud from Kinect
        self.cloudSub = rospy.Subscriber('/head_mount_kinect/qhd/points', PointCloud2, self.cloudCallback)
        print 'Connected to Kinect depth'

        # Connect to bowl center location
        self.bowlSub = rospy.Subscriber('/hrl_manipulation_task/bowl_cen_pose', PoseStamped, self.bowlCallback)
        print 'Connected to bowl center location'

        # Connect to both PR2 arms
        if isScooping:
            print 'waiting for /arm_reach_enable'
            rospy.wait_for_service("/arm_reach_enable")
            self.armReachActionLeft  = rospy.ServiceProxy('/arm_reach_enable', String_String)
            print 'waiting for /right/arm_reach_enable'
            self.armReachActionRight = rospy.ServiceProxy('/right/arm_reach_enable', String_String)
            print 'Connected to both services'

        print 'Waiting for tf to gripper'
        self.tf.waitForTransform('head_mount_kinect_rgb_optical_frame', 'r_gripper_tool_frame', rospy.Time(0), rospy.Duration(60.0))
        print 'Received tf to gripper'
        [self.trans, self.rot] = self.tf.lookupTransform('head_mount_kinect_rgb_optical_frame', 'r_gripper_tool_frame', rospy.Time(0))

    def cancel(self):
        self.cloudSub.unregister()
        self.bowlSub.unregister()

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

    def bowlCallback(self, data):
        # print 'bowl::', data
        self.bowlPos = data.pose.position
        print 'bowlCallback with bowl position:', self.bowlPos

    def cloudCallback(self, data):
        # print 'Time between cloud calls:', time.time() - self.cloudTime
        # startTime = time.time()

        # Wait to obtain cloud data until after arms have been initialized
        if self.isScooping and not self.initialized:
            return

        pointCloud = data

        if self.bowlPos is not None:
            print 'Received Bowl Position from self.bowlPos:', self.bowlPos
            point = PointStamped()
            point.header.frame_id = 'torso_lift_link'
            point.point.x = self.bowlPos.x
            point.point.y = self.bowlPos.y
            point.point.z = self.bowlPos.z
            self.publishPoints('bowlCenterPre', [[self.bowlPos.x, self.bowlPos.y, self.bowlPos.z]], g=1.0, frame='torso_lift_link')
            print 'bowl point after transform:', point
            point = self.tf.transformPoint('head_mount_kinect_rgb_optical_frame', point)
            print 'bowl point after transform:', point
            bowlXY = np.array([point.point.x, point.point.y])
            bowlZ = point.point.z

            self.publishPoints('bowlCenterPost', [[bowlXY[0], bowlXY[1], bowlZ]], r=1.0, frame='head_mount_kinect_rgb_optical_frame')
        else:
            # bowlPos = self.armReachActionLeft('returnBowlPos')
            bowlPos = self.getBowlFrame()
            print 'Received Bowl Position from self.getBowlFrame():', bowlPos
            bowlXY = np.array([bowlPos.p.x(), bowlPos.p.y()])
            bowlZ = bowlPos.p.z()


        # pointCloud = self.tf.transformPointCloud('r_gripper_tool_frame', pointCloud)

        # transform = TransformStamped()
        # transform.header.frame_id = 'head_mount_kinect_rgb_optical_frame'
        # transform.child_frame_id = 'r_gripper_tool_frame'
        # transform.transform.translation.x = self.trans[0]
        # transform.transform.translation.y = self.trans[1]
        # transform.transform.translation.z = self.trans[2]
        # transform.transform.rotation.x = self.rot[0]
        # transform.transform.rotation.y = self.rot[1]
        # transform.transform.rotation.z = self.rot[2]
        # transform.transform.rotation.w = self.rot[3]
        # transform.header.stamp = rospy.Time.now()
        # pointCloud = do_transform_cloud(pointCloud, transform)

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

    def getBowlFrame(self):
        # Get frame info from right arm and upate bowl_pos

        # 1. right arm ('r_gripper_tool_frame') from tf
        self.tf.waitForTransform(self.torso_frame, 'r_gripper_tool_frame', rospy.Time(0), rospy.Duration(5.0))
        [pos, quat] = self.tf.lookupTransform(self.torso_frame, 'r_gripper_tool_frame', rospy.Time(0))
        p = PyKDL.Vector(pos[0], pos[1], pos[2])
        M = PyKDL.Rotation.Quaternion(quat[0], quat[1], quat[2], quat[3])

        # 2. add offset to called TF value. Make sure Orientation is up right.
        ## Off set : 11 cm x direction, - 5 cm z direction.
        pos_offset = rospy.get_param('hrl_manipulation_task/sub_ee_pos_offset')
        orient_offset = rospy.get_param('hrl_manipulation_task/sub_ee_orient_offset')

        p = p + M * PyKDL.Vector(pos_offset['x'], pos_offset['y'], pos_offset['z'])
        M.DoRotX(orient_offset['rx'])
        M.DoRotY(orient_offset['ry'])
        M.DoRotZ(orient_offset['rz'])

        print 'Bowl frame:', p

        return PyKDL.Frame(M, p)

    def publishPoints(self, name, points, size=0.01, r=0.0, g=0.0, b=0.0, a=1.0, frame='r_gripper_tool_frame'):
        marker = Marker()
        marker.header.frame_id = frame
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
    scooping = False
    client = ArmReacherClient(scooping)

    if scooping:
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

    time.sleep(60)

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

