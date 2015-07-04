#!/usr/bin/env python

__author__ = 'zerickson'

import cv2
import time
import rospy
import random
import numpy as np

try :
    import sensor_msgs.point_cloud2 as pc2
except:
    import point_cloud2 as pc2
from sensor_msgs.msg import Image, CameraInfo
from roslib import message

import roslib
roslib.load_manifest('hrl_multimodal_anomaly_detection')
import tf
import image_geometry
from cv_bridge import CvBridge, CvBridgeError
from hrl_multimodal_anomaly_detection.msg import Circle, Rectangle, ImageFeatures

class rgbPerception:
    def __init__(self, targetFrame=None, visual=False, tfListener=None):
        # ROS publisher for data points
        self.publisher2D = rospy.Publisher('image_features', ImageFeatures)
        self.bridge = CvBridge()

        self.rgbTime = time.time()
        self.visual = visual
        self.targetFrame = targetFrame
        self.updateNumber = 0

        if tfListener is None:
            self.transformer = tf.TransformListener()
        else:
            self.transformer = tfListener

        self.imageData = None
        self.activeFeatures = []
        self.currentIndex = 0

        # RGB Camera
        self.rgbCameraFrame = None
        self.cameraWidth = None
        self.cameraHeight = None
        self.pinholeCamera = None
        self.prevGray = None

        # Gripper
        self.lGripperPosition = None
        self.lGripperRotation = None
        self.lGripperTransposeMatrix = None
        self.lGripX = None
        self.lGripY = None
        # Spoon
        self.spoonX = None
        self.spoonY = None

        # Bounding Box
        self.box = None
        self.spoonBox = None
        self.centerX = None
        self.centerY = None
        self.diffX = None
        self.diffY = None

        # Parameters for ShiTomasi corner detection
        self.feature_params = dict(maxCorners=100, qualityLevel=0.1, minDistance=7, blockSize=7)
        # Parameters for Lucas Kanade optical flow
        self.lk_params = dict(winSize=(15,15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        self.N = 50

        rospy.Subscriber('/head_mount_kinect/rgb_lowres/image', Image, self.imageCallback)
        print 'Connected to Kinect images'
        rospy.Subscriber('/head_mount_kinect/rgb_lowres/camera_info', CameraInfo, self.cameraRGBInfoCallback)
        print 'Connected to Kinect camera info'

    def getAllRecentPoints(self):
        return self.imageData, self.getNovelAndClusteredFeatures()

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

        self.box = self.boundingBox()

        # Grab image from Kinect sensor
        try:
            image = self.bridge.imgmsg_to_cv(data)
            image = np.asarray(image[:,:])
        except CvBridgeError, e:
            print e
            return
        print image.shape

        lowX, highX, lowY, highY = self.box

        # Crop imageGray to bounding box size
        self.imageData = image[lowY:highY, lowX:highX, :]

        print 'Time for first step:', time.time() - startTime
        timeStamp = time.time()
        # TODO This is all optional

        # Convert to grayscale
        imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        self.spoonBox = self.boundingBoxSpoon()

        if not self.activeFeatures:
            # Determine initial features
            self.determineGoodFeatures(imageGray)
            self.prevGray = imageGray
            self.rgbTime = time.time()
            return

        # Add new features to our feature tracker
        self.determineGoodFeatures(imageGray)

        print 'Time for second step:', time.time() - timeStamp
        timeStamp = time.time()
        if self.activeFeatures:
            self.opticalFlow(imageGray)
        print 'Time for third step:', time.time() - timeStamp
        timeStamp = time.time()
        if self.visual:
            self.publishImageFeatures()
        print 'Time for fourth step:', time.time() - timeStamp

        self.prevGray = imageGray

        self.updateNumber += 1
        print 'RGB computation time:', time.time() - startTime
        self.rgbTime = time.time()

    def determineGoodFeatures(self, imageGray):
        if len(self.activeFeatures) >= self.N:
            return
        lowX, highX, lowY, highY = self.spoonBox
        boxX, _, boxY, _ = self.box

        # Crop imageGray to bounding box size
        imageGray = imageGray[lowY:highY, lowX:highX]

        # Take a frame and find corners in it
        feats = cv2.goodFeaturesToTrack(imageGray, mask=None, **self.feature_params)

        # Reposition features back into original bounding box image size
        feats[:, 0, 0] += lowX - boxX
        feats[:, 0, 1] += lowY - boxY
        feats = feats.tolist()

        while len(self.activeFeatures) < self.N and len(feats) > 0:
            feat = random.choice(feats)
            feats.remove(feat)

            # Add feature to tracking list
            newFeat = feature(self.currentIndex, feat[0], boxX, boxY)
            self.activeFeatures.append(newFeat)
            self.currentIndex += 1

    def opticalFlow(self, imageGray):
        feats = []
        for feat in self.activeFeatures:
            feats.append([feat.position])
        feats = np.array(feats, dtype=np.float32)

        lowX, highX, lowY, highY = self.box

        # Crop imageGray to bounding box size
        imageGray = imageGray[lowY:highY, lowX:highX]

        print self.prevGray.shape, imageGray.shape

        newFeats, status, error = cv2.calcOpticalFlowPyrLK(self.prevGray, imageGray, feats, None, **self.lk_params)
        statusRemovals = [i for i, s in enumerate(status) if s == 0]

        # Update all features
        for i, feat in enumerate(self.activeFeatures):
            feat.update(newFeats[i][0], lowX, lowY)

        # Remove all features that are no longer being tracked (ie. status == 0)
        self.activeFeatures = np.delete(self.activeFeatures, statusRemovals, axis=0).tolist()

        # Remove all features outside the bounding box
        self.activeFeatures = [feat for feat in self.activeFeatures if self.pointInBoundingBox(feat.position, self.box)]

    def getNovelAndClusteredFeatures(self):
        feats = [feat for feat in self.activeFeatures if feat.isNovel]
        if not feats:
            # No novel features
            return None
        return {feat.index: feat.position.tolist() for i, feat in enumerate(feats) if self.pointInBoundingBox(feat.position, self.box)}

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
    def boundingBox(self):
        size = 150
        left = self.lGripX - 10
        right = left + size
        bottom = self.lGripY + 10
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

    # Finds a tighter bounding box around the spoon
    # Returns coordinates (lowX, highX, lowY, highY)
    def boundingBoxSpoon(self):
        left = self.lGripX + 10
        right = self.spoonX + 10
        bottom = self.lGripY - 10
        top = self.spoonY - 10

        # Check if box extrudes past image bounds
        if left < 0:
            left = 0
        if right > self.cameraWidth - 1:
            right = self.cameraWidth - 1
        if top < 0:
            top = 0
        if bottom > self.cameraHeight - 1:
            bottom = self.cameraHeight - 1

        # Check if spoon box extrudes past large bounding box
        lowX, highX, lowY, highY = self.box
        if left < lowX:
            print 'Spoon left is less than bounding box!'
        if right > highX:
            print 'Spoon right is greater than bounding box!'
        if top < lowY:
            print 'Spoon top is less than bounding box!'
        if bottom > highY:
            print 'Spoon bottom is greater than bounding box!'

        return int(left), int(right), int(top), int(bottom)

    @staticmethod
    def pointInBoundingBox(point, boxPoints):
        px, py = point
        lowX, highX, lowY, highY = boxPoints
        return lowX <= px <= highX and lowY <= py <= highY

    def cameraRGBInfoCallback(self, data):
        if self.cameraWidth is None:
            self.cameraWidth = data.width
            self.cameraHeight = data.height
            self.pinholeCamera = image_geometry.PinholeCameraModel()
            self.pinholeCamera.fromCameraInfo(data)

    def publishImageFeatures(self):
        imageFeatures = ImageFeatures()

        # Draw an orange point on image for gripper
        circle = Circle()
        circle.x, circle.y = int(self.lGripX), int(self.lGripY)
        circle.radius = 10
        circle.r = 255
        circle.g = 128
        imageFeatures.circles.append(circle)

        # Draw an blue point on image for spoon tip
        circle = Circle()
        circle.x, circle.y = int(self.spoonX), int(self.spoonY)
        circle.radius = 10
        circle.r = 50
        circle.g = 255
        circle.b = 255
        imageFeatures.circles.append(circle)

        # Draw all features (as red)
        for feat in self.activeFeatures:
            circle = Circle()
            circle.x, circle.y = feat.position
            circle.radius = 5
            circle.r = 255
            imageFeatures.circles.append(circle)

        # Draw a bounding box around spoon (or left gripper)
        rect = Rectangle()
        rect.lowX, rect.highX, rect.lowY, rect.highY = self.box
        rect.r = 75
        rect.g = 150
        rect.thickness = 5
        imageFeatures.rectangles.append(rect)

        # Draw a bounding box around spoon (or left gripper)
        rect = Rectangle()
        rect.lowX, rect.highX, rect.lowY, rect.highY = self.spoonBox
        rect.r = 128
        rect.b = 128
        rect.thickness = 3
        imageFeatures.rectangles.append(rect)

        features = self.getNovelAndClusteredFeatures()
        if features is not None:
            # Draw all novel and bounded box features
            for feat in features.values():
                circle = Circle()
                circle.x, circle.y = feat
                circle.radius = 5
                circle.b = 255
                circle.g = 128
                imageFeatures.circles.append(circle)

        self.publisher2D.publish(imageFeatures)

class feature:
    def __init__(self, index, position, lowX, lowY):
        # position = np.array(position)
        self.index = index
        self.position = position
        self.globalStart = position + [lowX, lowY, 0]
        self.globalNow = position + [lowX, lowY, 0]
        self.isNovel = False
        self.velocity = None
        self.lastTime = None

    def update(self, newPosition, lowX, lowY):
        newPosition = np.array(newPosition)
        # Update velocity of feature
        if self.lastTime is not None:
            distChange = newPosition - self.position
            timeChange = time.time() - self.lastTime
            self.velocity = distChange / timeChange
        self.lastTime = time.time()
        self.position = newPosition
        self.globalNow = newPosition + [lowX, lowY, 0]
        distance = np.linalg.norm(self.globalNow - self.globalStart)
        if distance >= 15:
            self.isNovel = True
