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
from geometry_msgs.msg import PoseStamped
from roslib import message

import roslib
roslib.load_manifest('hrl_multimodal_anomaly_detection')
import tf
import image_geometry
from cv_bridge import CvBridge, CvBridgeError
from hrl_multimodal_anomaly_detection.msg import Circle, Rectangle, ImageFeatures

class kinectDepthWithBowl:
    def __init__(self, targetFrame=None, visual=False, tfListener=None):
        self.publisher2D = rospy.Publisher('image_features', ImageFeatures)
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

        self.bowlPosition = None
        self.bowlPositionKinect = None
        self.bowlToKinectMat = None
        self.bowlX = None
        self.bowlY = None

        self.targetTrans = None
        self.targetRot = None

        self.cloudSub = rospy.Subscriber('/head_mount_kinect/depth_registered/points', PointCloud2, self.cloudCallback)
        print 'Connected to Kinect depth'
        # self.imageSub = rospy.Subscriber('/head_mount_kinect/rgb_lowres/image', Image, self.imageCallback)
        # print 'Connected to Kinect image'
        self.cameraSub = rospy.Subscriber('/head_mount_kinect/depth_lowres/camera_info', CameraInfo, self.cameraRGBInfoCallback)
        print 'Connected to Kinect camera info'
        rospy.Subscriber('hrl_feeding_task/manual_bowl_location', PoseStamped, self.bowlPoseManualCallback)

    def getAllRecentPoints(self):
        # print 'Time between read calls:', time.time() - self.cloudTime
        # startTime = time.time()

        if self.bowlPosition is None:
            return None

        self.transformer.waitForTransform(self.targetFrame, self.rgbCameraFrame, rospy.Time(0), rospy.Duration(5))
        try:
            self.targetTrans, self.targetRot = self.transformer.lookupTransform(self.targetFrame, self.rgbCameraFrame, rospy.Time(0))
        except tf.ExtrapolationException:
            print 'TF Target Error!'
            pass

        # print 'Read computation time:', time.time() - startTime
        # self.cloudTime = time.time()
        return self.points3D, self.bowlPosition, self.bowlPositionKinect, [self.bowlX, self.bowlY], self.bowlToKinectMat, [self.targetTrans, self.targetRot]


    def cancel(self):
        self.cloudSub.unregister()
        self.cameraSub.unregister()
        # self.imageSub.unregister()

    def cloudCallback(self, data):
        # print 'Time between cloud calls:', time.time() - self.cloudTime
        # startTime = time.time()

        self.pointCloud = data

        self.transposeBowlToCamera()

        lowX, highX, lowY, highY = self.boundingBox()

        points2D = [[x, y] for y in xrange(lowY, highY) for x in xrange(lowX, highX)]
        try:
            points3D = pc2.read_points(self.pointCloud, field_names=('x', 'y', 'z'), skip_nans=True, uvs=points2D)
        except:
            print 'Unable to unpack from PointCloud2!', self.cameraWidth, self.cameraHeight, self.pointCloud.width, self.pointCloud.height
            return

        self.points3D = np.array([point for point in points3D])

        self.publishImageFeatures()

        self.updateNumber += 1
        # print 'Cloud computation time:', time.time() - startTime
        # self.cloudTime = time.time()

    def bowlPoseManualCallback(self, data):
        self.bowlPosition = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z])

    def transposeBowlToCamera(self):
        self.transformer.waitForTransform(self.rgbCameraFrame, '/torso_lift_link', rospy.Time(0), rospy.Duration(5))
        try :
            position, rotation = self.transformer.lookupTransform(self.rgbCameraFrame, '/torso_lift_link', rospy.Time(0))
            self.bowlToKinectMat = np.dot(tf.transformations.translation_matrix(position), tf.transformations.quaternion_matrix(rotation))
        except tf.ExtrapolationException:
            print 'Transpose of bowl failed!'
            return
        self.bowlPositionKinect = np.dot(self.bowlToKinectMat, self.bowlPosition)[:3]
        self.bowlX, self.bowlY = self.pinholeCamera.project3dToPixel(self.bowlPositionKinect)

    # Returns coordinates (lowX, highX, lowY, highY)
    def boundingBox(self):
        size = 150
        left = self.bowlX - size/2
        right = left + size
        top = self.bowlY - size/2
        bottom = top + size

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

    def publishImageFeatures(self):
        imageFeatures = ImageFeatures()

        if self.bowlX is not None:
            circle = Circle()
            circle.x, circle.y = int(self.bowlX), int(self.bowlY)
            circle.radius = 10
            circle.r = 255
            circle.g = 128
            imageFeatures.circles.append(circle)

        rect = Rectangle()
        rect.lowX, rect.highX, rect.lowY, rect.highY = self.boundingBox()
        rect.r = 75
        rect.g = 150
        rect.thickness = 5
        imageFeatures.rectangles.append(rect)

        self.publisher2D.publish(imageFeatures)


