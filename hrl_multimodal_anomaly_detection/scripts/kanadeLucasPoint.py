#!/usr/bin/env python

__author__ = 'zerickson'

import cv2
import time
import math
import rospy
import random
import numpy as np
try :
    import sensor_msgs.point_cloud2 as pc2
except:
    import point_cloud2 as pc2
from visualization_msgs.msg import Marker
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import Point, PointStamped
from roslib import message

# Clustering
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

import roslib
roslib.load_manifest('hrl_multimodal_anomaly_detection')
import tf
import image_geometry
from cv_bridge import CvBridge, CvBridgeError
from hrl_multimodal_anomaly_detection.msg import Circle, Rectangle, ImageFeatures

class kanadeLucasPoint:
    def __init__(self, caller, targetFrame=None, publish=False, visual=False, tfListener=None):
        self.caller = caller
        self.bridge = CvBridge()
        # ROS publisher for data points
        self.publisher = rospy.Publisher('visualization_marker', Marker)
        self.publisher2D = rospy.Publisher('image_features', ImageFeatures)
        # params for ShiTomasi corner detection
        self.feature_params = dict(maxCorners=100, qualityLevel=0.1, minDistance=7, blockSize=7)
        # Parameters for lucas kanade optical flow
        self.lk_params = dict(winSize=(15,15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        # Variables to store past iteration output
        self.prevGray = None
        self.currentIndex = 0
        # List of features we are tracking
        self.activeFeatures = []
        self.allFeatures = []
        # self.features = []
        # PointCloud2 data used for determining 3D location from 2D pixel location
        self.pointCloud = None
        # Transformations
        self.frameId = None
        self.cameraWidth = None
        self.cameraHeight = None
        if tfListener is None:
            self.transformer = tf.TransformListener()
        else:
            self.transformer = tfListener
        self.targetFrame = targetFrame
        self.transMatrix = None
        # Whether to publish data to a topic
        self.publish = publish
        # Whether to display visual plots or not
        self.visual = visual
        self.updateNumber = 0
        # Gripper data
        self.lGripperTranslation = None
        self.lGripperRotation = None
        self.lGripperTransposeMatrix = None
        self.lGripX = None
        self.lGripY = None
        self.spoonX = None
        self.spoonY = None
        self.pinholeCamera = None
        self.rgbCameraFrame = None
        self.box = None
        self.lastTime = time.time()

        self.dbscan = DBSCAN(eps=3, min_samples=6)
        # self.dbscan2D = DBSCAN(eps=0.6, min_samples=6)

        self.N = 30

        # XBox 360 Kinect
        # rospy.Subscriber('/camera/rgb/image_color', Image, self.imageCallback)
        # rospy.Subscriber('/camera/depth_registered/points', PointCloud2, self.cloudCallback)
        # rospy.Subscriber('/camera/rgb/camera_info', CameraInfo, self.cameraRGBInfoCallback)
        # Kinect 2
        rospy.Subscriber('/head_mount_kinect/rgb_lowres/image', Image, self.imageCallback)
        print 'Connected to Kinect images'
        rospy.Subscriber('/head_mount_kinect/depth_registered/points', PointCloud2, self.cloudCallback)
        print 'Connected to Kinect depth'
        rospy.Subscriber('/head_mount_kinect/rgb_lowres/camera_info', CameraInfo, self.cameraRGBInfoCallback)
        print 'Connected to Kinect camera info'
        # PR2 Simulated
        # rospy.Subscriber('/head_mount_kinect/rgb/image_color', Image, self.imageCallback)
        # rospy.Subscriber('/head_mount_kinect/depth_registered/points', PointCloud2, self.cloudCallback)
        # rospy.Subscriber('/head_mount_kinect/rgb/camera_info', CameraInfo, self.cameraRGBInfoCallback)
        # print 'Connected to all topics'

        # spin() simply keeps python from exiting until this node is stopped
        # rospy.spin()

    def getRecentPoint(self, index):
        if index >= self.markerRecentCount():
            return None
        return self.activeFeatures[index].recent3DPosition

    # Returns a dictionary, with keys as point indices and values as a 3D point
    def getAllRecentPoints(self):
        if self.markerRecentCount() <= 0:
            print 'No novel features'
            return None
        return self.getNovelAndClusteredFeatures(returnFeatures=False)

    def getAllMarkersWithHistory(self):
        if self.markerRecentCount() <= 0:
            return None
        return self.getNovelAndClusteredFeatures(returnFeatures=True)

    def markerRecentCount(self):
        if self.activeFeatures is None:
            return 0
        return len([feat for feat in self.activeFeatures if feat.isNovel])

    def getNovelAndClusteredFeatures(self, returnFeatures=False):
        # Cluster feature points
        points = []
        feats = []
        for feat in self.activeFeatures:
            if feat.isNovel:
                points.append(feat.recent3DPosition)
                feats.append(feat)
        if not points:
            # No novel features
            return None
        points = np.array(points)

        # Perform dbscan clustering
        X = StandardScaler().fit_transform(points)
        labels = self.dbscan.fit_predict(X)

        # # Find the cluster closest to our gripper (To be continued possibly)
        # unique_labels = set(labels)
        # clusterPoints = points[labels==k]

        if self.lGripperTranslation is None:
            return None

        if returnFeatures:
            # Return a list of features
            return [feat for i, feat in enumerate(feats) if labels[i] != -1 and self.pointInBoundingBox(feat.recent2DPosition, self.box)]
        else:
            # Return a dictionary of indices and 3D points
            return {feat.index: feat.recent3DPosition for i, feat in enumerate(feats) if labels[i] != -1 and self.pointInBoundingBox(feat.recent2DPosition, self.box)}

    def determineGoodFeatures(self, imageGray):
        if len(self.activeFeatures) >= self.N or self.lGripperTranslation is None:
            return

        # Determine a bounding box around spoon (or left gripper) to narrow search area
        lowX, highX, lowY, highY = self.box
        print lowX, highX, lowY, highY, imageGray.shape

        # Crop imageGray to bounding box size
        imageGray = imageGray[lowY:highY, lowX:highX]
        # print imageGray.shape

        # Take a frame and find corners in it
        feats = cv2.goodFeaturesToTrack(imageGray, mask=None, **self.feature_params)

        # Reposition features back into original image size
        # print feats.shape
        feats[:, 0, 0] += lowX
        feats[:, 0, 1] += lowY
        feats = feats.tolist()

        while len(self.activeFeatures) < self.N and len(feats) > 0:
            feat = random.choice(feats)
            feats.remove(feat)
            # Check to make feature is near gripper when transformed into 3D
            # feat3D = self.get3DPointFromCloud(feat[0])
            # if feat3D is None:
            #     continue
            # distFromGripper = np.linalg.norm(self.lGripperTranslation - feat3D)
            # if distFromGripper > 0.4:
            #     continue
            # Add feature to tracking list
            newFeat = feature(self.currentIndex, feat[0], self)
            self.activeFeatures.append(newFeat)
            self.allFeatures.append(newFeat)
            self.currentIndex += 1

    def opticalFlow(self, imageGray):
        feats = []
        for feat in self.activeFeatures:
            feats.append([feat.recent2DPosition])
        feats = np.array(feats, dtype=np.float32)

        newFeats, status, error = cv2.calcOpticalFlowPyrLK(self.prevGray, imageGray, feats, None, **self.lk_params)
        statusRemovals = [i for i, s in enumerate(status) if s == 0]

        # Update all features
        for i, feat in enumerate(self.activeFeatures):
            feat.update(newFeats[i][0])

        # Remove all features that are no longer being tracked (ie. status == 0)
        self.activeFeatures = np.delete(self.activeFeatures, statusRemovals, axis=0).tolist()

        # Remove all features outside the bounding box
        self.activeFeatures = [feat for feat in self.activeFeatures if self.pointInBoundingBox(feat.recent2DPosition, self.box)]

        # Define features as novel if they meet a given criteria
        for feat in self.activeFeatures:
            # Consider features that have traveled 5 cm
            if feat.distance >= 0.15:
                feat.isNovel = True

    def drawOnImage(self, image):
        # Draw all features
        for feat in self.activeFeatures:
            a, b = feat.recent2DPosition
            cv2.circle(image, (a, b), 5, [0, 0, 255], -1)

        # Draw an orange point on image for gripper
        cv2.circle(image, (int(self.lGripX), int(self.lGripY)), 10, [0, 125, 255], -1)

        # Draw a bounding box around spoon (or left gripper)
        lowX, highX, lowY, highY = self.box
        cv2.rectangle(image, (lowX, lowY), (highX, highY), color=[0, 150, 75], thickness=5)

        features = self.getNovelAndClusteredFeatures(returnFeatures=True)
        if features is None:
            # print 'no novel features to draw'
            return image

        # Draw all novel and bounded box features
        for feat in features:
            a, b = feat.recent2DPosition
            cv2.circle(image, (a, b), 5, [255, 125, 0], -1)

        return image

    def publishImageFeatures(self):
        imageFeatures = ImageFeatures()
        # Draw all features (as red)
        for feat in self.activeFeatures:
            circle = Circle()
            circle.x, circle.y = feat.recent2DPosition
            circle.radius = 5
            circle.r = 255
            imageFeatures.circles.append(circle)

        # Draw an orange point on image for gripper
        circle = Circle()
        circle.x, circle.y = int(self.lGripX), int(self.lGripY)
        circle.radius = 10
        circle.r = 255
        circle.g = 125
        imageFeatures.circles.append(circle)

        # Draw an blue point on image for spoon tip
        circle = Circle()
        circle.x, circle.y = int(self.spoonX), int(self.spoonY)
        circle.radius = 10
        circle.r = 50
        circle.g = 255
        circle.b = 255
        imageFeatures.circles.append(circle)

        # Draw a bounding box around spoon (or left gripper)
        rect = Rectangle()
        rect.lowX, rect.highX, rect.lowY, rect.highY = self.box
        rect.r = 75
        rect.g = 150
        rect.thickness = 5
        imageFeatures.rectangles.append(rect)

        features = self.getNovelAndClusteredFeatures(returnFeatures=True)
        if features is not None:
            # Draw all novel and bounded box features
            for feat in features:
                circle = Circle()
                circle.x, circle.y = feat.recent2DPosition
                circle.radius = 5
                circle.b = 255
                circle.g = 125
                imageFeatures.circles.append(circle)

        self.publisher2D.publish(imageFeatures)

    # Finds a bounding box around a given point
    # Returns coordinates (lowX, highX, lowY, highY)
    def boundingBox(self, point):
        # Define a box with respect to the l_gripper_tool_frame
        # Left is on +y axis
        left3D =  [0, 0.1, 0]
        right3D = [0, -0.1, 0]
        # Up is on +x axis
        up3D = [0.3, 0, 0]
        down3D = [0.05, 0, 0]
        spoon3D = [0.22, 0, 0]

        # Transpose box onto orientation of gripper
        left = np.dot(self.lGripperTransposeMatrix, np.array([left3D[0], left3D[1], left3D[2], 1.0]))[:3]
        right = np.dot(self.lGripperTransposeMatrix, np.array([right3D[0], right3D[1], right3D[2], 1.0]))[:3]
        top = np.dot(self.lGripperTransposeMatrix, np.array([up3D[0], up3D[1], up3D[2], 1.0]))[:3]
        bottom = np.dot(self.lGripperTransposeMatrix, np.array([down3D[0], down3D[1], down3D[2], 1.0]))[:3]
        spoon = np.dot(self.lGripperTransposeMatrix, np.array([spoon3D[0], spoon3D[1], spoon3D[2], 1.0]))[:3]

        # Project 3D box locations to 2D for the camera
        left, _ = self.pinholeCamera.project3dToPixel(left)
        right, _ = self.pinholeCamera.project3dToPixel(right)
        _, top = self.pinholeCamera.project3dToPixel(top)
        _, bottom = self.pinholeCamera.project3dToPixel(bottom)
        self.spoonX, self.spoonY = self.pinholeCamera.project3dToPixel(spoon)

        # Adjust incase hand is upside down
        if left > right:
            left, right = right, left
        if top > bottom:
            top, bottom = bottom, top

        # Make sure box encompases the spoon
        if left > self.spoonX - 15:
            left = self.spoonX - 15
        if right < self.spoonX + 15:
            right = self.spoonX + 15
        if top > self.spoonY - 15:
            top = self.spoonY - 15
        if bottom < self.spoonY + 15:
            bottom = self.spoonY + 15

        # Check if box extrudes past image bounds
        if left < 0:
            left = 0
        if right > self.cameraWidth - 1:
            right = self.cameraWidth - 1
        if top < 0:
            top = 0
        if bottom > self.cameraHeight - 1:
            bottom = self.cameraHeight - 1

        # Verify that the box bounds are not too small (50 pixels is smallest)
        diff = 100 - np.abs(right - left)
        if np.abs(right - left) < 100:
            if left < diff/2.0:
                right += diff
            elif right > self.cameraWidth - diff/2.0 - 1:
                left -= diff
            else:
                left -= diff/2.0
                right += diff/2.0
        diff = 50 - np.abs(bottom - top)
        if np.abs(bottom - top) < 50:
            if top < diff/2.0:
                bottom += diff
            elif bottom > self.cameraHeight - diff/2.0 - 1:
                top -= diff
            else:
                top -= diff/2.0
                bottom += diff/2.0

        return left, right, top, bottom

    def pointInBoundingBox(self, point, boxPoints):
        px, py = point
        lowX, highX, lowY, highY = boxPoints
        return lowX <= px <= highX and lowY <= py <= highY

    def publishFeatures(self):
        if self.cameraWidth is None:
            return
        # Display all novel (object) features that we are tracking.
        marker = Marker()
        marker.header.frame_id = self.targetFrame
        marker.ns = 'points'
        marker.type = marker.POINTS
        marker.action = marker.ADD
        marker.scale.x = 0.01
        marker.scale.y = 0.01
        marker.color.a = 1.0
        marker.color.b = 1.0
        for feature in self.activeFeatures:
            if not feature.isNovel:
                continue
            x, y, depth = feature.recent3DPosition
            # Create 3d point
            p = Point()
            p.x = x
            p.y = y
            p.z = depth
            marker.points.append(p)

        self.publisher.publish(marker)

    def get3DPointFromCloud(self, feature2DPosition):
        if self.cameraWidth is None:
            return None
        # Grab x, y values from feature and verify they are within the image bounds
        x, y = feature2DPosition
        x, y = int(x), int(y)
        if x >= self.cameraWidth or y >= self.cameraHeight:
            # print 'x, y greater than camera size! Feature', x, y, self.cameraWidth, self.cameraHeight
            return None
        # Retrieve 3d location of feature from PointCloud2
        if self.pointCloud is None:
            # print 'AHH! The PointCloud2 data is not available!'
            return None
        try:
            points = pc2.read_points(self.pointCloud, field_names=('x', 'y', 'z'), skip_nans=False, uvs=[[x, y]])
            # Grab the first 3D point received from PointCloud2
            px, py, depth = points.next()
        except:
            # print 'Unable to unpack from PointCloud2.', self.cameraWidth, self.cameraHeight, self.pointCloud.width, self.pointCloud.height
            return None
        if any([math.isnan(v) for v in [px, py, depth]]):
            # print 'NaN! Feature', px, py, depth
            return None

        xyz = None
        # Transpose point to targetFrame
        if self.targetFrame is not None:
            xyz = np.dot(self.transMatrix, np.array([px, py, depth, 1.0]))[:3]

        return xyz

    def imageCallback(self, data):
        # start = time.time()
        # print 'Time between image calls:', start - self.lastTime
        # Grab image from Kinect sensor
        try:
            image = self.bridge.imgmsg_to_cv(data)
            image = np.asarray(image[:,:])
        except CvBridgeError, e:
            print e
            return
        # Grab image from video input
        # video = cv2.VideoCapture('../images/cabinet.mov')
        # ret, image = video.read()

        # Convert to grayscale
        imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self.lGripperTranslation is None:
            return

        # Used to verify that each point is within our defined box
        self.box = [int(x) for x in self.boundingBox((self.lGripX, self.lGripY))]

        # Find frameId for transformations and determine a good set of starting features
        if self.frameId is None or not self.activeFeatures:
            # Grab frame id for later transformations
            self.frameId = data.header.frame_id
            if self.targetFrame is not None:
                t = rospy.Time(0)
                self.transformer.waitForTransform(self.targetFrame, self.frameId, t, rospy.Duration(5.0))
                trans, rot = self.transformer.lookupTransform(self.targetFrame, self.frameId, t)
                self.transMatrix = np.dot(tf.transformations.translation_matrix(trans), tf.transformations.quaternion_matrix(rot))
            # Determine initial set of features
            self.determineGoodFeatures(imageGray)
            self.prevGray = imageGray
            self.lastTime = time.time()
            return

        # Add new features to our feature tracker
        self.determineGoodFeatures(imageGray)

        if self.activeFeatures:
            self.opticalFlow(imageGray)
            if self.publish:
                self.publishFeatures()
        if self.visual:
            self.publishImageFeatures()
            # This takes a long time!! (Publish the information instead)
            # image = self.drawOnImage(image)
            # cv2.imshow('Image window', image)
            # cv2.waitKey(30)

        self.updateNumber += 1

        self.prevGray = imageGray

        # print 'Image calculation time:', time.time() - start
        # self.lastTime = time.time()

        # Call our caller now that new data has been processed
        if self.caller is not None:
            self.caller()

    def cloudCallback(self, data):
        # Store PointCloud2 data for use when determining 3D locations
        self.pointCloud = data

    def cameraRGBInfoCallback(self, data):
        if self.cameraWidth is None:
            self.cameraWidth = data.width
            self.cameraHeight = data.height
        if self.pinholeCamera is None:
            self.pinholeCamera = image_geometry.PinholeCameraModel()
            self.pinholeCamera.fromCameraInfo(data)
            self.rgbCameraFrame = data.header.frame_id
        # Transpose gripper position to camera frame
        self.transformer.waitForTransform(self.rgbCameraFrame, '/l_gripper_tool_frame', rospy.Time(0), rospy.Duration(1.0))
        try :
            self.lGripperTranslation, self.lGripperRotation = self.transformer.lookupTransform(self.rgbCameraFrame, '/l_gripper_tool_frame', rospy.Time(0))
            # print self.lGripperTranslation, tf.transformations.euler_from_quaternion(self.lGripperRotation)
            self.lGripperTransposeMatrix = np.dot(tf.transformations.translation_matrix(self.lGripperTranslation), tf.transformations.quaternion_matrix(self.lGripperRotation))
            # Find 2D location of gripper
            self.lGripX, self.lGripY = self.pinholeCamera.project3dToPixel(self.lGripperTranslation)
        except tf.ExtrapolationException:
            pass

minDist = 0.015
maxDist = 0.03
# minDist = 0.05
# maxDist = 0.1
class feature:
    def __init__(self, index, position, kanadeLucas):
        self.index = index
        self.kanadeLucas = kanadeLucas
        self.startPosition = None
        self.recent2DPosition = position
        self.recent3DPosition = None
        self.frame = 0
        self.distance = 0.0
        self.isNovel = False
        self.history = []
        self.lastHistoryPosition = None
        self.lastHistoryCount = 0

        self.setStartPosition()

    def setStartPosition(self):
        self.startPosition = self.kanadeLucas.get3DPointFromCloud(self.recent2DPosition)
        self.history = [self.startPosition] if self.startPosition is not None else []
        self.lastHistoryPosition = self.startPosition

    def update(self, newPosition):
        newPosition = np.array(newPosition)
        self.recent2DPosition = newPosition
        self.frame += 1
        # Check if start position has been successfully set yet
        if self.startPosition is None:
            self.setStartPosition()
            return
        # Grab 3D location for this feature
        position3D = self.kanadeLucas.get3DPointFromCloud(self.recent2DPosition)
        if position3D is None:
            return
        self.recent3DPosition = position3D
        # Update total distance traveled
        self.distance = np.linalg.norm(self.recent3DPosition - self.startPosition)
        # Check if the point has traveled far enough to add a new history point
        dist = np.linalg.norm(self.recent3DPosition - self.lastHistoryPosition)
        if minDist <= dist <= maxDist:
            self.history.append(self.recent3DPosition)
            self.lastHistoryPosition = self.recent3DPosition

    def isAvailableForNewPath(self):
        if len(self.history) - self.lastHistoryCount >= 10:
            self.lastHistoryCount = len(self.history)
            return True
        return False


''' sensor_msgs/Image data
std_msgs/Header header
  uint32 seq
  time stamp
  string frame_id
uint32 height
uint32 width
string encoding
uint8 is_bigendian
uint32 step
uint8[] data
'''
