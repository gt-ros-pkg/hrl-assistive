#!/usr/bin/env python

__author__ = 'zerickson'

import time
import rospy
import numpy as np

try :
    import sensor_msgs.point_cloud2 as pc2
except:
    import point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
from roslib import message

import roslib
roslib.load_manifest('hrl_multimodal_anomaly_detection')
import tf
import image_geometry
from cv_bridge import CvBridge, CvBridgeError

class pr2WideDepth:
    def __init__(self, targetFrame=None, visual=False, tfListener=None):
        self.cloudTime = time.time()
        self.readTime = time.time()
        self.pointCloud = None
        self.visual = visual
        self.targetFrame = targetFrame
        self.updateNumber = 0

        self.points3D = None

        if tfListener is None:
            self.transformer = tf.TransformListener()
        else:
            self.transformer = tfListener

        self.bridge = CvBridge()
        self.imageData = None
        self.spoonImageData = None

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
        self.micLocation = None
        self.grips = []
        # Spoon
        self.spoonX = None
        self.spoonY = None
        self.spoon = None

        self.targetTrans = None
        self.targetRot = None
        self.gripperTrans = None
        self.gripperRot = None

        self.cloudSub = rospy.Subscriber('/wide_stereo/points2', PointCloud2, self.cloudCallback)
        print 'Connected to depth'
        self.imageSub = rospy.Subscriber('/wide_stereo/right/image_color', Image, self.imageCallback)
        print 'Connected to image'
        self.cameraSub = rospy.Subscriber('/wide_stereo/right/camera_info', CameraInfo, self.cameraRGBInfoCallback)
        print 'Connected to camera info'

    def getAllRecentPoints(self):
        print 'Time between read calls:', time.time() - self.readTime
        startTime = time.time()

        self.transformer.waitForTransform(self.targetFrame, self.rgbCameraFrame, rospy.Time(0), rospy.Duration(5))
        try:
            self.targetTrans, self.targetRot = self.transformer.lookupTransform(self.targetFrame, self.rgbCameraFrame, rospy.Time(0))
        except tf.ExtrapolationException:
            print 'TF Target Error!'
            pass

        self.transformer.waitForTransform('/l_gripper_tool_frame', self.targetFrame, rospy.Time(0), rospy.Duration(5))
        try:
            self.gripperTrans, self.gripperRot = self.transformer.lookupTransform('/l_gripper_tool_frame', self.targetFrame, rospy.Time(0))
        except tf.ExtrapolationException:
            print 'TF Gripper Error!'
            pass

        print 'Read computation time:', time.time() - startTime
        self.readTime = time.time()
        return self.points3D, self.imageData, self.micLocation, self.spoon, [self.targetTrans, self.targetRot], [self.gripperTrans, self.gripperRot], \
               [self.lGripX, self.lGripY], [self.spoonX, self.spoonY]


    def cancel(self):
        self.cloudSub.unregister()
        self.cameraSub.unregister()
        self.imageSub.unregister()

    def imageCallback(self, data):
        if self.lGripX is None:
            return

        try:
            image = self.bridge.imgmsg_to_cv(data)
            self.imageData = np.asarray(image[:,:])
        except CvBridgeError, e:
            print e
            return

        # Crop imageGray to bounding box size
        # lowX, highX, lowY, highY = self.boundingBox()
        # self.imageData = image[lowY:highY, lowX:highX, :]

    def cloudCallback(self, data):
        print 'Time between cloud calls:', time.time() - self.cloudTime
        startTime = time.time()

        self.pointCloud = data

        self.transposeGripperToCamera()

        # Determine location of spoon
        spoon3D = [0.22, -0.050, 0]
        self.spoon = np.dot(self.lGripperTransposeMatrix, np.array([spoon3D[0], spoon3D[1], spoon3D[2], 1.0]))[:3]
        self.spoonX, self.spoonY = self.pinholeCamera.project3dToPixel(self.spoon)

        lowX, highX, lowY, highY = self.boundingBox()
        self.spoonImageData = self.imageData[lowY:highY, lowX:highX, :]

        points2D = [[x, y] for y in xrange(lowY, highY) for x in xrange(lowX, highX)]
        try:
            points3D = pc2.read_points(self.pointCloud, field_names=('x', 'y', 'z'), skip_nans=True, uvs=points2D)
        except:
            print 'Unable to unpack from PointCloud2!', self.cameraWidth, self.cameraHeight, self.pointCloud.width, self.pointCloud.height
            return

        self.points3D = np.array([point for point in points3D])

        self.updateNumber += 1
        print 'Cloud computation time:', time.time() - startTime
        self.cloudTime = time.time()

    def transposeGripperToCamera(self):
        # Transpose gripper position to camera frame
        self.transformer.waitForTransform(self.rgbCameraFrame, '/l_gripper_tool_frame', rospy.Time(0), rospy.Duration(5))
        try :
            self.lGripperPosition, self.lGripperRotation = self.transformer.lookupTransform(self.rgbCameraFrame, '/l_gripper_tool_frame', rospy.Time(0))
            transMatrix = np.dot(tf.transformations.translation_matrix(self.lGripperPosition), tf.transformations.quaternion_matrix(self.lGripperRotation))
        except tf.ExtrapolationException:
            print 'Transpose of gripper failed!'
            return

        mic = [0.12, -0.02, 0]

        # self.micLocation = np.dot(transMatrix, np.array([mic[0], mic[1], mic[2], 1.0]))[:3]
        # gripX, gripY = self.pinholeCamera.project3dToPixel(self.micLocation)
        # self.lGripX, self.lGripY, self.lGripperTransposeMatrix = int(gripX), int(gripY), transMatrix

        if len(self.grips) >= 2:
            self.lGripX, self.lGripY, self.lGripperTransposeMatrix = self.grips[-2]
            self.micLocation = np.dot(self.lGripperTransposeMatrix, np.array([mic[0], mic[1], mic[2], 1.0]))[:3]
            gripX, gripY = self.pinholeCamera.project3dToPixel(self.micLocation)
        else:
            self.micLocation = np.dot(transMatrix, np.array([mic[0], mic[1], mic[2], 1.0]))[:3]
            gripX, gripY = self.pinholeCamera.project3dToPixel(self.micLocation)
            self.lGripX, self.lGripY, self.lGripperTransposeMatrix = int(gripX), int(gripY), transMatrix
        self.grips.append((int(gripX), int(gripY), transMatrix))

    # Returns coordinates (lowX, highX, lowY, highY)
    def boundingBox(self):
        size = 150
        left = self.lGripX - 50
        right = left + size
        bottom = self.lGripY + 75
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
