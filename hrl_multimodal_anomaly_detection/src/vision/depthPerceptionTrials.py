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
from sensor_msgs.msg import Image, CameraInfo
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

class depthPerceptionTrials:
    def __init__(self, targetFrame=None, visual=False, tfListener=None):
        # ROS publisher for data points
        self.publisher = rospy.Publisher('visualization_marker', Marker)
        # List of features we are tracking
        self.clusterPoints = None
        self.nonClusterPoints = None
        self.bridge = CvBridge()

        self.dbscan = DBSCAN(eps=0.12, min_samples=10)
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
        # Spoon
        self.spoonX = None
        self.spoonY = None

        self.wooh1 = None
        self.wooh2 = None

        rospy.Subscriber('/head_mount_kinect/depth_lowres/image', Image, self.cloudCallback)
        print 'Connected to Kinect depth'
        rospy.Subscriber('/head_mount_kinect/depth_lowres/camera_info', CameraInfo, self.cameraRGBInfoCallback)
        print 'Connected to Kinect camera info'

    def getAllRecentPoints(self):
        self.transformer.waitForTransform(self.targetFrame, self.rgbCameraFrame, rospy.Time(0), rospy.Duration(5))
        try :
            targetTrans, targetRot = self.transformer.lookupTransform(self.targetFrame, self.rgbCameraFrame, rospy.Time(0))
            transMatrix = np.dot(tf.transformations.translation_matrix(targetTrans), tf.transformations.quaternion_matrix(targetRot))
        except tf.ExtrapolationException:
            return None
        return [np.dot(transMatrix, np.array([p[0], p[1], p[2], 1.0]))[:3].tolist() for p in self.wooh1], np.dot(transMatrix, np.array([self.wooh2[0], self.wooh2[1], self.wooh2[2], 1.0]))[:3].tolist()
        # return [np.dot(transMatrix, np.array([p[0], p[1], p[2], 1.0]))[:3].tolist() for p in self.clusterPoints], \
        #             [np.dot(transMatrix, np.array([p[0], p[1], p[2], 1.0]))[:3].tolist() for p in self.nonClusterPoints]

    def cloudCallback(self, data):
        # print 'Time between cloud calls:', time.time() - self.cloudTime
        # startTime = time.time()

        self.pointCloud = data

        self.transposeGripperToCamera()

        # Determine location of spoon
        spoon3D = [0.22, -0.050, 0]
        spoon = np.dot(self.lGripperTransposeMatrix, np.array([spoon3D[0], spoon3D[1], spoon3D[2], 1.0]))[:3]
        self.spoonX, self.spoonY = self.pinholeCamera.project3dToPixel(spoon)

        # lowX, highX, lowY, highY = self.boundingBox(0.05, 0.30, 0.05, 20, 100, 50)
        lowX, highX, lowY, highY = self.boundingBox()

        # Grab image from Kinect sensor
        try:
            image = self.bridge.imgmsg_to_cv(data)
            image = np.asarray(image[:,:], dtype=np.float32) / 1000.0
        except CvBridgeError, e:
            print e
            return

        # print 'Time for first call:', time.time() - startTime
        # ticker = time.time()

        # self.transformer.waitForTransform(self.rgbCameraFrame, '/head_mount_kinect_depth_optical_frame', rospy.Time(0), rospy.Duration(5))
        # try :
        #     pos ,rot = self.transformer.lookupTransform(self.rgbCameraFrame, '/head_mount_kinect_depth_optical_frame', rospy.Time(0))
        #     matrix = np.dot(tf.transformations.translation_matrix(pos), tf.transformations.quaternion_matrix(rot))
        # except tf.ExtrapolationException:
        #     return

        # points3D = []
        # for y in xrange(lowY, highY):
        #     for x in xrange(lowX, highX):
        #         pixel = self.pinholeCamera.projectPixelTo3dRay((x, y))
        #         points3D.append(np.dot(matrix, np.array([pixel[0], pixel[1], pixel[2], 1.0]))[:3]*image[y, x])
        # points3D = np.array(points3D)

        # points3D = np.array([np.array(self.pinholeCamera.projectPixelTo3dRay(np.dot(matrix, np.array([x, y, 0, 1.0]))[:3]))*image[y, x] for y in xrange(lowY, highY) for x in xrange(lowX, highX)])

        self.wooh1 = np.array([np.array(self.pinholeCamera.projectPixelTo3dRay((x, y)))*image[y, x] for y in xrange(lowY, highY) for x in xrange(lowX, highX)])
        self.wooh2 = np.array(self.pinholeCamera.projectPixelTo3dRay((self.lGripX, self.lGripY)))*image[self.lGripX, self.lGripY]

        # # try:
        # #     points3D = pc2.read_points(self.pointCloud, field_names=('x', 'y', 'z'), skip_nans=True, uvs=points2D)
        # #     gripperPoint = pc2.read_points(self.pointCloud, field_names=('x', 'y', 'z'), skip_nans=True, uvs=[[self.lGripX, self.lGripY]]).next()
        # # except:
        # #     # print 'Unable to unpack from PointCloud2.', self.cameraWidth, self.cameraHeight, self.pointCloud.width, self.pointCloud.height
        # #     return
        #
        # # points3D = np.array([point for point in points3D])
        # #
        # # self.clusterPoints = points3D
        #
        # # print 'Time for second call:', time.time() - ticker
        # # ticker = time.time()
        #
        # # Perform dbscan clustering
        # X = StandardScaler().fit_transform(points3D)
        # labels = self.dbscan.fit_predict(X)
        # # unique_labels = set(labels)
        #
        # # print 'Time for third call:', time.time() - ticker
        # # ticker = time.time()
        #
        # # Find the point closest to our gripper and it's corresponding label
        # index, closePoint = min(enumerate(np.linalg.norm(points3D - gripperPoint, axis=1)), key=operator.itemgetter(1))
        # closeLabel = labels[index]
        # while closeLabel == -1 and points3D.size > 0:
        #     np.delete(points3D, [index])
        #     np.delete(labels, [index])
        #     index, closePoint = min(enumerate(np.linalg.norm(points3D - gripperPoint, axis=1)), key=operator.itemgetter(1))
        #     closeLabel = labels[index]
        # if points3D.size <= 0:
        #     return
        # # print 'Label:', closeLabel
        #
        # # print 'Time for fourth call:', time.time() - ticker
        # # ticker = time.time()
        #
        # # Find the cluster closest to our gripper
        # self.clusterPoints = points3D[labels==closeLabel]
        # self.nonClusterPoints = points3D[labels!=closeLabel]
        #
        # # if self.visual:
        # #     # Publish depth features for spoon features
        # #     self.publishPoints('spoonPoints', self.clusterPoints, g=1.0)
        # #
        # #     # Publish depth features for non spoon features
        # #     self.publishPoints('nonSpoonPoints', self.nonClusterPoints, r=1.0)
        #
        # # print 'Time for fifth call:', time.time() - ticker

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
            self.lGripperTransposeMatrix = np.dot(tf.transformations.translation_matrix(self.lGripperPosition), tf.transformations.quaternion_matrix(self.lGripperRotation))
        except tf.ExtrapolationException:
            pass
        gripX, gripY = self.pinholeCamera.project3dToPixel(self.lGripperPosition)
        self.lGripX, self.lGripY = int(gripX), int(gripY)

    # Returns coordinates (lowX, highX, lowY, highY)
    def boundingBox(self):
        size = 150
        left = self.lGripX - 20
        right = left + size
        bottom = self.lGripY + 20
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

    # # Finds a bounding box given defined features
    # # Returns coordinates (lowX, highX, lowY, highY)
    # def boundingBox(self, leftRight, up, down, margin, widthDiff, heightDiff):
    #     # Left is on -z axis
    #     left3D =  [0, 0, -leftRight]
    #     right3D = [0, 0, leftRight]
    #     # Up is on +x axis
    #     up3D = [up, 0, 0]
    #     down3D = [down, 0, 0]
    #
    #     # Transpose box onto orientation of gripper
    #     left = np.dot(self.lGripperTransposeMatrix, np.array([left3D[0], left3D[1], left3D[2], 1.0]))[:3]
    #     right = np.dot(self.lGripperTransposeMatrix, np.array([right3D[0], right3D[1], right3D[2], 1.0]))[:3]
    #     top = np.dot(self.lGripperTransposeMatrix, np.array([up3D[0], up3D[1], up3D[2], 1.0]))[:3]
    #     bottom = np.dot(self.lGripperTransposeMatrix, np.array([down3D[0], down3D[1], down3D[2], 1.0]))[:3]
    #
    #     # Project 3D box locations to 2D for the camera
    #     left, _ = self.pinholeCamera.project3dToPixel(left)
    #     right, _ = self.pinholeCamera.project3dToPixel(right)
    #     _, top = self.pinholeCamera.project3dToPixel(top)
    #     _, bottom = self.pinholeCamera.project3dToPixel(bottom)
    #
    #     # Adjust incase hand is upside down
    #     if left > right:
    #         left, right = right, left
    #     if top > bottom:
    #         top, bottom = bottom, top
    #
    #     # Make sure box encompases the spoon
    #     if left > self.spoonX - margin:
    #         left = self.spoonX - margin
    #     if right < self.spoonX + margin:
    #         right = self.spoonX + margin
    #     if top > self.spoonY - margin:
    #         top = self.spoonY - margin
    #     if bottom < self.spoonY + margin:
    #         bottom = self.spoonY + margin
    #
    #     # Check if box extrudes past image bounds
    #     if left < 0:
    #         left = 0
    #     if right > self.cameraWidth - 1:
    #         right = self.cameraWidth - 1
    #     if top < 0:
    #         top = 0
    #     if bottom > self.cameraHeight - 1:
    #         bottom = self.cameraHeight - 1
    #
    #     # Verify that the box bounds are not too small
    #     diff = widthDiff - np.abs(right - left)
    #     if np.abs(right - left) < 100:
    #         if left < diff/2.0:
    #             right += diff
    #         elif right > self.cameraWidth - diff/2.0 - 1:
    #             left -= diff
    #         else:
    #             left -= diff/2.0
    #             right += diff/2.0
    #     diff = heightDiff - np.abs(bottom - top)
    #     if np.abs(bottom - top) < 50:
    #         if top < diff/2.0:
    #             bottom += diff
    #         elif bottom > self.cameraHeight - diff/2.0 - 1:
    #             top -= diff
    #         else:
    #             top -= diff/2.0
    #             bottom += diff/2.0
    #
    #     return int(left), int(right), int(top), int(bottom)

    def cameraRGBInfoCallback(self, data):
        if self.cameraWidth is None:
            self.cameraWidth = data.width
            self.cameraHeight = data.height
            self.pinholeCamera = image_geometry.PinholeCameraModel()
            self.pinholeCamera.fromCameraInfo(data)
            self.rgbCameraFrame = data.header.frame_id
