#!/usr/bin/env python

__author__ = 'zerickson'

import time
import rospy
import operator
import numpy as np

try :
    import sensor_msgs.point_cloud2 as pc2
except:
    import point_cloud2 as pc2
from visualization_msgs.msg import Marker
from sensor_msgs.msg import PointCloud2, CameraInfo
from geometry_msgs.msg import Point
from roslib import message

# Clustering
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

import roslib
roslib.load_manifest('hrl_multimodal_anomaly_detection')
import tf
import image_geometry
from hrl_multimodal_anomaly_detection.msg import Circle, Rectangle, ImageFeatures

class wideStereoDepth:
    def __init__(self, targetFrame=None, visual=False, tfListener=None):
        # ROS publisher for data points
        self.publisher = rospy.Publisher('visualization_marker', Marker)
        # List of features we are tracking
        self.cloudPoints = None

        # self.dbscan = DBSCAN(eps=0.12, min_samples=10)
        self.cloudTime = time.time()
        self.pointCloud = None
        self.visual = visual
        self.targetFrame = targetFrame
        self.updateNumber = 0

        if tfListener is None:
            self.transformer = tf.TransformListener()
        else:
            self.transformer = tfListener

        # RGB Camera
        self.rgbCameraFrame = None
        self.cameraWidth = None
        self.cameraHeight = None
        self.pinholeCamera = None

        # Gripper
        self.lGripperPosition = None
        self.lGripperRotation = None
        self.lGripperTransposeMatrix = None
        self.lGripX = None
        self.lGripY = None
        self.gripperPoint = None
        # Spoon
        self.spoonX = None
        self.spoonY = None

        rospy.Subscriber('/wide_stereo/points2', PointCloud2, self.cloudCallback)
        print 'Connected to Kinect depth'
        rospy.Subscriber('/wide_stereo/right/camera_info', CameraInfo, self.cameraRGBInfoCallback)
        print 'Connected to Kinect camera info'

    def getAllRecentPoints(self):
        self.transformer.waitForTransform(self.targetFrame, self.rgbCameraFrame, rospy.Time(0), rospy.Duration(5))
        try :
            targetTrans, targetRot = self.transformer.lookupTransform(self.targetFrame, self.rgbCameraFrame, rospy.Time(0))
            transMatrix = np.dot(tf.transformations.translation_matrix(targetTrans), tf.transformations.quaternion_matrix(targetRot))
        except tf.ExtrapolationException:
            return None
        return [np.dot(transMatrix, np.array([p[0], p[1], p[2], 1.0]))[:3].tolist() for p in self.cloudPoints], \
                    np.dot(transMatrix, np.array([self.gripperPoint[0], self.gripperPoint[1], self.gripperPoint[2], 1.0]))[:3].tolist()

    def cloudCallback(self, data):
        print 'Time between cloud calls:', time.time() - self.cloudTime
        startTime = time.time()

        self.pointCloud = data

        self.transposeGripperToCamera()

        # Determine location of spoon
        spoon3D = [0.22, -0.050, 0]
        spoon = np.dot(self.lGripperTransposeMatrix, np.array([spoon3D[0], spoon3D[1], spoon3D[2], 1.0]))[:3]
        self.spoonX, self.spoonY = self.pinholeCamera.project3dToPixel(spoon)

        lowX, highX, lowY, highY = self.boundingBox()

        points2D = [[x, y] for y in xrange(lowY, highY) for x in xrange(lowX, highX)]
        try:
            points3D = pc2.read_points(self.pointCloud, field_names=('x', 'y', 'z'), skip_nans=True, uvs=points2D)
        except:
            print 'Cloud reading error'
            return
        try:
            self.gripperPoint = pc2.read_points(self.pointCloud, field_names=('x', 'y', 'z'), skip_nans=True, uvs=[[self.lGripX, self.lGripY]]).next()
        except:
            print 'Gripper reading error'
            return

        self.cloudPoints = np.array([point for point in points3D])

        # print 'Cloud gathering time:', time.time() - startTime
        # stepTime = time.time()
        # self.publishPoints('points', self.cloudPoints, g=1.0)
        # print 'Cloud publishing time:', time.time() - stepTime

        self.updateNumber += 1
        print 'Cloud computation time:', time.time() - startTime
        self.cloudTime = time.time()

    def publishPoints(self, name, points, size=0.01, r=0.0, g=0.0, b=0.0, a=1.0):
        marker = Marker()
        marker.header.frame_id = self.rgbCameraFrame
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
            if len(point) <= 0:
                continue
            if len(point) > 3:
                continue
            p = Point()
            p.x, p.y, p.z = point
            marker.points.append(p)
        self.publisher.publish(marker)

    def transposeGripperToCamera(self):
        # Transpose gripper position to camera frame
        self.transformer.waitForTransform(self.rgbCameraFrame, '/l_gripper_tool_frame', rospy.Time(0), rospy.Duration(5))
        try :
            self.lGripperPosition, self.lGripperRotation = self.transformer.lookupTransform(self.rgbCameraFrame, '/l_gripper_tool_frame', rospy.Time(0))
            self.lGripperTransposeMatrix = np.dot(tf.transformations.translation_matrix(self.lGripperPosition), tf.transformations.quaternion_matrix(self.lGripperRotation))
        except tf.ExtrapolationException:
            pass
        gripX, gripY = self.pinholeCamera.project3dToPixel(self.lGripperPosition)
        self.lGripX, self.lGripY = int(gripX), int(gripY)

    # Returns coordinates (lowX, highX, lowY, highY)
    def boundingBox(self):
        size = 150
        left = self.lGripX
        right = left + size
        bottom = self.lGripY + 60
        top = bottom - size

        # Check if box extrudes past image bounds
        if left < 0:
            left = 0
            right = left + size
        if right > self.cameraWidth - 1:
            right = self.cameraWidth - 1
            left = right - size
        if top < 0:
            top = 0
            bottom = top + size
        if bottom > self.cameraHeight - 1:
            bottom = self.cameraHeight - 1
            top = bottom - size

        return int(left), int(right), int(top), int(bottom)

    def cameraRGBInfoCallback(self, data):
        if self.cameraWidth is None:
            self.cameraWidth = data.width
            self.cameraHeight = data.height
            self.pinholeCamera = image_geometry.PinholeCameraModel()
            self.pinholeCamera.fromCameraInfo(data)
            self.rgbCameraFrame = data.header.frame_id
