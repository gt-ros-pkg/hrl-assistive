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

class depthPerception:
    def __init__(self, targetFrame=None, visual=False, tfListener=None):
        # ROS publisher for data points
        self.publisher2D = rospy.Publisher('image_features', ImageFeatures)

        self.rgbTime = time.time()
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
        # Spoon
        self.spoonX = None
        self.spoonY = None

        rospy.Subscriber('/head_mount_kinect/depth_registered/points', PointCloud2, self.cloudCallback)
        print 'Connected to Kinect depth'
        rospy.Subscriber('/head_mount_kinect/rgb_lowres/camera_info', CameraInfo, self.cameraRGBInfoCallback)
        print 'Connected to Kinect camera info'

    def getAllRecentPoints(self):
        self.transformer.waitForTransform(self.targetFrame, self.rgbCameraFrame, rospy.Time(0), rospy.Duration(5))
        try :
            targetTrans, targetRot = self.transformer.lookupTransform(self.targetFrame, self.rgbCameraFrame, rospy.Time(0))
            transMatrix = np.dot(tf.transformations.translation_matrix(targetTrans), tf.transformations.quaternion_matrix(targetRot))
        except tf.ExtrapolationException:
            return None
        return [np.dot(transMatrix, np.array([p[0], p[1], p[2], 1.0]))[:3].tolist() for p in self.clusterPoints]

    def imageCallback(self, data):
        startTime = time.time()
        print 'Time between rgb calls:', time.time() - self.rgbTime
        if self.rgbCameraFrame is None:
            self.rgbCameraFrame = data.header.frame_id

        self.transposeGripperToCamera()

        # Determine location of spoon
        spoon3D = [0.22, -0.050, 0]
        spoon = np.dot(self.lGripperTransposeMatrix, np.array([spoon3D[0], spoon3D[1], spoon3D[2], 1.0]))[:3]
        self.spoonX, self.spoonY = self.pinholeCamera.project3dToPixel(spoon)

        # Find half way point between spoon end and gripper
        self.centerX = (self.lGripX + self.spoonX) / 2.0
        self.centerY = (self.lGripY + self.spoonY) / 2.0
        self.diffX = np.abs(self.centerX - self.lGripX)
        self.diffY = np.abs(self.centerY - self.lGripY)

        self.updateNumber += 1
        print 'RGB computation time:', time.time() - startTime
        self.rgbTime = time.time()

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

    # Finds a bounding box given defined features
    # Returns coordinates (lowX, highX, lowY, highY)
    def boundingBox(self, leftRight, up, down, margin, widthDiff, heightDiff):
        left = self.centerX - 100
        right = self.centerX + 100
        top = self.centerY - 100
        bottom = self.centerY + 100

        # Make sure box encompasses the spoon

        # Check if box extrudes past image bounds
        if left < 0:
            left = 0
        if right > self.cameraWidth - 1:
            right = self.cameraWidth - 1
        if top < 0:
            top = 0
        if bottom > self.cameraHeight - 1:
            bottom = self.cameraHeight - 1



        # Left is on -z axis
        left3D =  [0, 0, -leftRight]
        right3D = [0, 0, leftRight]
        # Up is on +x axis
        up3D = [up, 0, 0]
        down3D = [down, 0, 0]

        # Transpose box onto orientation of gripper
        left = np.dot(self.lGripperTransposeMatrix, np.array([left3D[0], left3D[1], left3D[2], 1.0]))[:3]
        right = np.dot(self.lGripperTransposeMatrix, np.array([right3D[0], right3D[1], right3D[2], 1.0]))[:3]
        top = np.dot(self.lGripperTransposeMatrix, np.array([up3D[0], up3D[1], up3D[2], 1.0]))[:3]
        bottom = np.dot(self.lGripperTransposeMatrix, np.array([down3D[0], down3D[1], down3D[2], 1.0]))[:3]

        # Project 3D box locations to 2D for the camera
        left, _ = self.pinholeCamera.project3dToPixel(left)
        right, _ = self.pinholeCamera.project3dToPixel(right)
        _, top = self.pinholeCamera.project3dToPixel(top)
        _, bottom = self.pinholeCamera.project3dToPixel(bottom)

        # Adjust incase hand is upside down
        if left > right:
            left, right = right, left
        if top > bottom:
            top, bottom = bottom, top

        # Make sure box encompases the spoon
        if left > self.spoonX - margin:
            left = self.spoonX - margin
        if right < self.spoonX + margin:
            right = self.spoonX + margin
        if top > self.spoonY - margin:
            top = self.spoonY - margin
        if bottom < self.spoonY + margin:
            bottom = self.spoonY + margin

        # Verify that the box bounds are not too small
        diff = widthDiff - np.abs(right - left)
        if np.abs(right - left) < 100:
            if left < diff/2.0:
                right += diff
            elif right > self.cameraWidth - diff/2.0 - 1:
                left -= diff
            else:
                left -= diff/2.0
                right += diff/2.0
        diff = heightDiff - np.abs(bottom - top)
        if np.abs(bottom - top) < 50:
            if top < diff/2.0:
                bottom += diff
            elif bottom > self.cameraHeight - diff/2.0 - 1:
                top -= diff
            else:
                top -= diff/2.0
                bottom += diff/2.0

        return int(left), int(right), int(top), int(bottom)

    def cameraRGBInfoCallback(self, data):
        if self.cameraWidth is None:
            self.cameraWidth = data.width
            self.cameraHeight = data.height
            self.pinholeCamera = image_geometry.PinholeCameraModel()
            self.pinholeCamera.fromCameraInfo(data)
