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
        self.initialized = False
	self.highestPointPublished = False
        self.bowlRawPos = None
        self.bowlCenter = None
        self.pinholeCamera = None
        self.cameraWidth = None
        self.cameraHeight = None

        # ROS publisher for data points
        if self.verbose: self.pointPublisher = rospy.Publisher('visualization_marker', Marker, queue_size=100)
        self.highestBowlPointPublisher = rospy.Publisher('/hrl_manipulation_task/bowl_highest_point', Point, queue_size=10)

        # Connect to point cloud from Kinect
        self.cloudSub = rospy.Subscriber('/head_mount_kinect/qhd/points', PointCloud2, self.cloudCallback)
        if self.verbose: print 'Connected to Kinect depth'
        self.cameraSub = rospy.Subscriber('/head_mount_kinect/qhd/camera_info', CameraInfo, self.cameraRGBInfoCallback)
        if self.verbose: print 'Connected to Kinect camera info'

        # Connect to arm reacher
        self.initSub = rospy.Subscriber('/hrl_manipulation_task/arm_reacher/init_bowl_height', Empty, self.initCallback)

        # Connect to bowl center location
        self.bowlSub = rospy.Subscriber('/hrl_manipulation_task/arm_reacher/bowl_cen_pose', PoseStamped, self.bowlCallback)
        if self.verbose: print 'Connected to bowl center location'

    def cancel(self):
        self.cloudSub.unregister()
        self.cameraSub.unregister()
        self.bowlSub.unregister()

    # Call this right after 'lookAtBowl' and right before 'initScooping2'
    def initialize(self):
        self.reset()
        self.initialized = True

    def reset(self):
        self.highestPointPublished = False
        self.bowlCenter = None

    def bowlCallback(self, data):
        bowlPosePos = data.pose.position
        # Account for the fact  that the bowl center position is not directly in the center
        self.bowlRawPos = [bowlPosePos.x + 0.015, bowlPosePos.y + 0.02, bowlPosePos.z]
        if self.verbose: print 'Bowl position:', self.bowlRawPos

    def cameraRGBInfoCallback(self, data):
        if self.pinholeCamera is None:
            self.cameraWidth = data.width
            self.cameraHeight = data.height
            self.pinholeCamera = image_geometry.PinholeCameraModel()
            self.pinholeCamera.fromCameraInfo(data)

    def initCallback(self, req):
	self.initialize()
	print 'Initialize Callback'

    def cloudCallback(self, data):
        # print 'Time between cloud calls:', time.time() - self.cloudTime
        # startTime = time.time()

        # Wait to obtain cloud data until after arms have been initialized
        if not self.initialized or self.highestPointPublished:
            return

        pointCloud = data

        # Transform the raw bowl center to the Kinect frame
        if self.bowlCenter is None:
            if self.bowlRawPos is not None:
                if self.verbose: print 'Using self.bowlRawPos'
                point = PointStamped()
                point.header.frame_id = 'torso_lift_link'
                point.point.x = self.bowlRawPos[0]
                point.point.y = self.bowlRawPos[1]
                point.point.z = self.bowlRawPos[2]
                # self.publishPoints('bowlCenterPre', [], g=1.0, frame='torso_lift_link')
                point = self.tf.transformPoint('head_mount_kinect_ir_optical_frame', point)
                self.bowlCenter = np.array([point.point.x, point.point.y, point.point.z])
            else:
                print 'No bowl center location has been published by the arm reacher server yet.'
                return
            if self.verbose: self.publishPoints('bowlCenterPost', [self.bowlCenter], r=1.0)

        # Project bowl position to 2D pixel location to narrow search for bowl points (use point projected to kinect frame)
        bowlProjX, bowlProjY = [int(x) for x in self.pinholeCamera.project3dToPixel(self.bowlCenter)]
        # print '3D bowl point:', self.bowlCenter, 'Projected X, Y:', bowlProjX, bowlProjY, 'Camera width, height:', self.cameraWidth, self.cameraHeight
        points2D = [[x, y] for y in xrange(bowlProjY-75, bowlProjY+75) for x in xrange(bowlProjX-75, bowlProjX+75)]
        try:
            points3D = pc2.read_points(pointCloud, field_names=('x', 'y', 'z'), skip_nans=True, uvs=points2D)
        except:
            print 'Unable to unpack from PointCloud2!'
            return

	points3D = [x for x in points3D]
	if self.verbose: self.publishPoints('allDepthPoints', points3D, r=0.5, g=0.5)

        t = time.time()

        # Define an ellipsoid with dimensional lengths a, b, c, and verify if points are within the ellipsoid when (x/a)^2 + (y/b)^2 + (z/c)^2 < 1
        # https://www.wikiwand.com/en/Ellipsoid
        a = 0.065 # width
        b = 0.04 # depth
        c = 0.05 # height
        # Subtract bowl center location since ellipsoid in not at the origin.
        cx, cy, cz = self.bowlCenter
        points3D = np.array([(x, y, z) for (x, y, z) in points3D if ((x - cx)/a)**2 + ((y - cy)/b)**2 + ((z - cz)/c)**2 < 1])

        # X, Y Positions must be within a radius of 5 cm from the bowl center, and Z positions must be within 4 cm of center
        # points3D = np.array([point for point in points3D if np.linalg.norm(self.bowlCenter[:2] - np.array(point[:2])) < 0.045 and abs(self.bowlCenter[2] - point[2]) < 0.03])

        print 'Time to determine points near bowl center:', time.time() - t

        if len(points3D) == 0:
            print 'No highest point detected within the bowl.'

        # Find the highest point (based on Z axis value) that is within the bowl. Positive Z is facing towards the floor, so find the min Z value
        if self.verbose: print 'points3D:', np.shape(points3D)

        # Begin transforming each point back into torso_lift_link. This way we can find the highest point in the bowl.
        # This ir_optical_frame is not fixed and the Z axis is not guaranteed to be pointing upwards
        t = time.time()
        self.tf.waitForTransform('torso_lift_link', 'head_mount_kinect_ir_optical_frame', rospy.Time(), rospy.Duration(5))
        try :
            translation, rotation = self.tf.lookupTransform('torso_lift_link', 'head_mount_kinect_ir_optical_frame', rospy.Time())
            transformationMatrix = np.dot(tf.transformations.translation_matrix(translation), tf.transformations.quaternion_matrix(rotation))
        except tf.ExtrapolationException:
            print 'Transpose of bowl points failed!'
            return

        newPoints = np.empty_like(points3D)
        # Append a column of 1's to the points matrix (i.e. at a 1 to each point)
        points3D = np.concatenate((points3D, np.ones((len(points3D), 1))), axis=1)
        for i, point in enumerate(points3D):
            newPoints[i] = np.dot(transformationMatrix, point)[:3]
        points3D = newPoints

        print 'Transform time:', time.time() - t

        # Use five multivariate (trivariate) Gaussian distributions in the bowl to determine the best scooping location (https://www.wikiwand.com/en/Multivariate_normal_distribution)
        # Shift bowl center location down to bottom of bowl
	# Note: This is using torso_lift_link frame
        bowlBottom = self.bowlRawPos + np.array([0, 0, 0.0])
        # mu (center) locations for Gaussian distributions are as follows: center, left, right, forwards, backwards)
        muLocations = [bowlBottom, bowlBottom + np.array([0, 0.03, 0]), bowlBottom + np.array([0, -0.03, 0]), bowlBottom + np.array([0.025, 0, 0]), bowlBottom + np.array([-0.025, 0, 0])]
        if self.verbose: self.publishPoints('muLocations', muLocations, size=0.004, r=1.0, g=1.0, frame='torso_lift_link')
        n, m = np.shape(points3D)
        highestBowlPoint = None
        highestPdfValue = 0
        for mu in muLocations:
            sigma = np.zeros((m, m))
            # Compute sample covariance matrix (https://en.wikipedia.org/wiki/Sample_mean_and_covariance)
            for j in xrange(m):
                for k in xrange(m):
                    sigma[j, k] = 1.0 / (n-1) * np.dot((points3D[:, j] - mu[j]).T, points3D[:, k] - mu[k])
            # Compute constant portion of distribution
            constant = 1.0 / np.sqrt((2*np.pi)**m * np.linalg.det(sigma))
            sigmaInv = np.linalg.inv(sigma)
            # Evaluate the Probability Density Function for each point and determine a cumulative pdf value
            pdfValue = 0
            for point in newPoints:
                pointMu = point - mu
                pdfValue += constant * np.exp(-1.0/2.0 * np.dot(np.dot(pointMu.T, sigmaInv), pointMu))
            # Check if this is the largest pdf value thus far
            if pdfValue > highestPdfValue:
                highestBowlPoint = mu
                highestPdfValue = pdfValue

        if self.verbose: print 'Highest bowl point:', highestBowlPoint, '| Highest pdf value:', highestPdfValue

        # A crude solution is to merely use the tallest point within the bowl as the 
        # Z axis for torso_lift_link frame is towards the sky, thus find max Z value for highest point
        # maxIndex = points3D[:, 2].argmax()
        # highestBowlPoint = np.array(points3D[maxIndex]) #+ np.array([0, 0.25, 0])

	self.publishHighestBowlPoint(highestBowlPoint)
	# We only want to publish the highest point once.
	self.highestPointPublished = True

        # Publish highest bowl point and all 3D points in bowl
        if self.verbose:
            self.publishPoints('highestPoint', [highestBowlPoint], size=0.008, r=.5, b=.5, frame='torso_lift_link')
            self.publishPoints('allPoints', points3D, g=0.6, b=1.0, frame='torso_lift_link')

    def publishHighestBowlPoint(self, highestBowlPoint):
        p = Point()
        p.x, p.y, p.z = highestBowlPoint
        self.highestBowlPointPublisher.publish(p)

    def publishPoints(self, name, points, size=0.002, r=0.0, g=0.0, b=0.0, a=1.0, frame='head_mount_kinect_ir_optical_frame'):
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
        self.pointPublisher.publish(marker)


if __name__ == '__main__':
    client = ArmReacherClient(verbose=True)
    # client.initialize()

    rate = rospy.Rate(10) # 25Hz, nominally.
    while not rospy.is_shutdown():
        rate.sleep()

    client.cancel()

