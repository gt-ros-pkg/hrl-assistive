#!/usr/bin/env python

__author__ = 'zerickson'

import time
import rospy
import numpy as np

try :
    import sensor_msgs.point_cloud2 as pc2
except:
    import point_cloud2 as pc2
from visualization_msgs.msg import Marker
from sensor_msgs.msg import CameraInfo
from stereo_msgs.msg import DisparityImage
from geometry_msgs.msg import Point
from roslib import message

# Clustering
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

import roslib
roslib.load_manifest('hrl_multimodal_anomaly_detection')
import tf
import image_geometry
from cv_bridge import CvBridge, CvBridgeError
from hrl_multimodal_anomaly_detection.msg import Circle, Rectangle, ImageFeatures

class wideStereoRGB:
    def __init__(self, targetFrame=None, visual=False, tfListener=None):
        # ROS publisher for data points
        self.publisher = rospy.Publisher('visualization_marker', Marker)

        # List of features we are tracking
        self.points3D = None

        self.leftInfo = None
        self.rightInfo = None
        self.camera = None
        self.cameraWidth = None
        self.cameraHeight = None
        self.bridge = CvBridge()

        # self.dbscan = DBSCAN(eps=0.12, min_samples=10)
        self.imageTime = time.time()
        self.visual = visual
        self.targetFrame = targetFrame
        self.updateNumber = 0

        if tfListener is None:
            self.transformer = tf.TransformListener()
        else:
            self.transformer = tfListener

        # RGB Camera
        self.rgbCameraFrame = None

        # Gripper
        self.lGripperPosition = None
        self.lGripperRotation = None
        self.lGripperTransposeMatrix = None
        self.lGripX = None
        self.lGripY = None
        self.gripperPoint = None
        self.grips = []
        # Spoon
        self.spoonX = None
        self.spoonY = None

        rospy.Subscriber('/wide_stereo/disparity', DisparityImage, self.imageCallback)
        print 'Connected to Kinect depth'
        rospy.Subscriber('/wide_stereo/left/camera_info', CameraInfo, self.cameraInfoLeftCallback)
        rospy.Subscriber('/wide_stereo/right/camera_info', CameraInfo, self.cameraInfoRightCallback)
        print 'Connected to Kinect camera info'

    def getAllRecentPoints(self):
        print 'Time between recent point calls:', time.time() - self.imageTime
        startTime = time.time()
        self.transformer.waitForTransform(self.targetFrame, self.rgbCameraFrame, rospy.Time(0), rospy.Duration(5))
        try:
            targetTrans, targetRot = self.transformer.lookupTransform(self.targetFrame, self.rgbCameraFrame, rospy.Time(0))
            transMatrix = np.dot(tf.transformations.translation_matrix(targetTrans), tf.transformations.quaternion_matrix(targetRot))
            print transMatrix
        except tf.ExtrapolationException:
            return None
        print 'Transform computation time:', time.time() - startTime
        tempTime = time.time()
        points = np.c_[self.points3D, np.ones(self.points3D.shape[0])]
        values = np.dot(transMatrix, points.T).T[:, :3]
        print 'Dot computation time:', time.time() - tempTime
        # values = [np.dot(transMatrix, np.array([p[0], p[1], p[2], 1.0]))[:3].tolist() for p in self.points3D]
        print 'Recent points computation time:', time.time() - startTime
        self.imageTime = time.time()
        return values, np.dot(transMatrix, np.array([self.gripperPoint[0], self.gripperPoint[1], self.gripperPoint[2], 1.0]))[:3].tolist()

    def imageCallback(self, data):
        if self.camera is None and self.leftInfo is not None and self.rightInfo is not None:
            self.camera = image_geometry.StereoCameraModel()
            self.camera.fromCameraInfo(self.leftInfo, self.rightInfo)
        elif self.camera is None:
            return

        # print 'Time between image calls:', time.time() - self.imageTime
        # startTime = time.time()

        try:
            image = self.bridge.imgmsg_to_cv(data.image)
            image = np.asarray(image[:,:])
        except CvBridgeError, e:
            print e
            return

        self.transposeGripperToCamera()

        # Determine location of spoon
        spoon3D = [0.22, -0.050, 0]
        spoon = np.dot(self.lGripperTransposeMatrix, np.array([spoon3D[0], spoon3D[1], spoon3D[2], 1.0]))[:3]
        self.spoonX, self.spoonY = self.camera.project3dToPixel(spoon)

        lowX, highX, lowY, highY = self.boundingBox()

        self.points3D = [self.camera.projectPixelTo3d((x, y), image[y, x]) for y in xrange(lowY, highY) for x in xrange(lowX, highX) if x % 2 == 0]
        self.gripperPoint = self.camera.projectPixelTo3d((self.lGripX, self.lGripY), image[self.lGripY, self.lGripX])

        # print 'Cloud gathering time:', time.time() - startTime
        # stepTime = time.time()
        # self.publishPoints('points', self.points3D, g=1.0)
        # print 'Cloud publishing time:', time.time() - stepTime

        self.updateNumber += 1
        # print 'Image computation time:', time.time() - startTime
        # self.imageTime = time.time()

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
        try:
            self.lGripperPosition, self.lGripperRotation = self.transformer.lookupTransform(self.rgbCameraFrame, '/l_gripper_tool_frame', rospy.Time(0))
            self.lGripperTransposeMatrix = np.dot(tf.transformations.translation_matrix(self.lGripperPosition), tf.transformations.quaternion_matrix(self.lGripperRotation))
        except tf.ExtrapolationException:
            pass
        # gripX, gripY = self.camera.project3dToPixel(self.lGripperPosition)[0]

        mic = [0.10, 0, 0]
        micLoc = np.dot(self.lGripperTransposeMatrix, np.array([mic[0], mic[1], mic[2], 1.0]))[:3]
        gripX, gripY = self.camera.project3dToPixel(micLoc)[0]
        if len(self.grips) >= 3:
            self.lGripX, self.lGripY = self.grips[-3]
        else:
            self.lGripX, self.lGripY = int(gripX), int(gripY)
        self.grips.append((int(gripX), int(gripY)))
        self.lGripX, self.lGripY = int(gripX), int(gripY)

    # Returns coordinates (lowX, highX, lowY, highY)
    def boundingBox(self):
        size = 200
        left = self.lGripX - 75
        right = left + size
        bottom = self.lGripY + 100
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

    def cameraInfoLeftCallback(self, data):
        if self.leftInfo is None:
            self.leftInfo = data
            self.cameraWidth = data.width
            self.cameraHeight = data.height
            self.rgbCameraFrame = data.header.frame_id

    def cameraInfoRightCallback(self, data):
        if self.rightInfo is None:
            self.rightInfo = data
