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
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
from geometry_msgs.msg import Point, PointStamped
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

class kinectDepth:
    def __init__(self, targetFrame=None, visual=False, tfListener=None):
        # ROS publisher for data points
        self.publisher = rospy.Publisher('visualization_marker', Marker)
        # List of features we are tracking
        self.clusterPoints = None
        self.nonClusterPoints = None

        self.dbscan = DBSCAN(eps=0.12, min_samples=10)
        self.cloudTime = time.time()
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

        self.gripperPoints = None

        self.cloudSub = rospy.Subscriber('/head_mount_kinect/depth_registered/points', PointCloud2, self.cloudCallback)
        print 'Connected to Kinect depth'
        # self.imageSub = rospy.Subscriber('/head_mount_kinect/rgb_lowres/image', Image, self.imageCallback)
        # print 'Connected to Kinect image'
        self.cameraSub = rospy.Subscriber('/head_mount_kinect/depth_lowres/camera_info', CameraInfo, self.cameraRGBInfoCallback)
        print 'Connected to Kinect camera info'

    def getAllRecentPoints(self):
        # print 'Time between read calls:', time.time() - self.cloudTime
        # startTime = time.time()

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

        # print 'Read computation time:', time.time() - startTime
        # self.cloudTime = time.time()
        return self.points3D, self.gripperPoints, self.imageData, self.micLocation, self.spoon, [self.targetTrans, self.targetRot], [self.gripperTrans, self.gripperRot]


        # self.transformer.waitForTransform(self.targetFrame, self.rgbCameraFrame, rospy.Time(0), rospy.Duration(5))
        # try:
        #     targetTrans, targetRot = self.transformer.lookupTransform(self.targetFrame, self.rgbCameraFrame, rospy.Time(0))
        #     self.transMatrix = np.dot(tf.transformations.translation_matrix(targetTrans), tf.transformations.quaternion_matrix(targetRot))
        #     # print transMatrix
        # except tf.ExtrapolationException:
        #     print 'TF Error!'
        #     pass
        # points = np.c_[self.points3D, np.ones(len(self.points3D))]
        # values = np.dot(self.transMatrix, points.T).T[:, :3]
        # return values, np.dot(self.transMatrix, np.array([self.micLocation[0], self.micLocation[1], self.micLocation[2], 1.0]))[:3].tolist(), \
        #        np.dot(self.transMatrix, np.array([self.spoon[0], self.spoon[1], self.spoon[2], 1.0]))[:3].tolist()

    def cancel(self):
        self.publisher.unregister()
        self.cloudSub.unregister()
        self.cameraSub.unregister()
        # self.imageSub.unregister()

    def imageCallback(self, data):
        if self.lGripX is None:
            return

        try:
            image = self.bridge.imgmsg_to_cv(data)
            image = np.asarray(image[:,:])
        except CvBridgeError, e:
            print e
            return

        # Crop imageGray to bounding box size
        lowX, highX, lowY, highY = self.boundingBox()
        self.imageData = image[lowY:highY, lowX:highX, :]

    def cloudCallback(self, data):
        # print 'Time between cloud calls:', time.time() - self.cloudTime
        # startTime = time.time()

        self.pointCloud = data

        self.transposeGripperToCamera()

        # Determine location of spoon
        spoon3D = [0.22, -0.050, 0]
        self.spoon = np.dot(self.lGripperTransposeMatrix, np.array([spoon3D[0], spoon3D[1], spoon3D[2], 1.0]))[:3]
        # self.spoonX, self.spoonY = self.pinholeCamera.project3dToPixel(spoon)

        lowX, highX, lowY, highY = self.boundingBox()

        points2D = [[x, y] for y in xrange(lowY, highY) for x in xrange(lowX, highX)]
        try:
            points3D = pc2.read_points(self.pointCloud, field_names=('x', 'y', 'z'), skip_nans=True, uvs=points2D)
        except:
            print 'Unable to unpack from PointCloud2!', self.cameraWidth, self.cameraHeight, self.pointCloud.width, self.pointCloud.height
            return

        self.points3D = np.array([point for point in points3D])

        self.gripperPoints = []
        # Transform all points into gripper frame
        for point in self.points3D:
            pointStamped = PointStamped()
            pointStamped.header.frame_id = self.rgbCameraFrame
            pointStamped.point.x = point[0]
            pointStamped.point.y = point[1]
            pointStamped.point.z = point[2]
            self.gripperPoints.append(self.transformer.transformPoint('/l_gripper_tool_frame', pointStamped))

        self.updateNumber += 1
        # print 'Cloud computation time:', time.time() - startTime
        # self.cloudTime = time.time()

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
                print point
            p = Point()
            p.x, p.y, p.z = point
            marker.points.append(p)
        self.publisher.publish(marker)

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
