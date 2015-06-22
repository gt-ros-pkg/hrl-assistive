#!/usr/bin/env python

__author__ = 'zerickson'

import cv2
import math
import rospy
import numpy as np
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
roslib.load_manifest('cv_bridge')
import tf
from cv_bridge import CvBridge, CvBridgeError

class kanadeLucasPoint:
    def __init__(self, caller, targetFrame=None, visual=False):
        self.caller = caller
        self.bridge = CvBridge()
        # ROS publisher for data points
        self.publisher = rospy.Publisher('visualization_marker', Marker)
        # params for ShiTomasi corner detection
        self.feature_params = dict(maxCorners=100, qualityLevel=0.1, minDistance=7, blockSize=7)
        # Parameters for lucas kanade optical flow
        self.lk_params = dict(winSize=(15,15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        # Variables to store past iteration output
        self.prevGray = None
        self.prevFeats = None
        # List of features we are tracking
        self.features = None
        # PointCloud2 data used for determining 3D location from 2D pixel location
        self.pointCloud = None
        # Transformations
        self.frameId = None
        self.cameraWidth = None
        self.cameraHeight = None
        self.transformer = tf.TransformListener()
        self.targetFrame = targetFrame
        self.transMatrix = None
        # Whether to display visual plots or not
        self.visual = visual
        self.updateNumber = 0

        self.dbscan = DBSCAN(eps=0.6, min_samples=10)
        self.N = 30

        # XBox 360 Kinect
        # rospy.Subscriber('/camera/rgb/image_color', Image, self.imageCallback)
        # rospy.Subscriber('/camera/depth_registered/points', PointCloud2, self.cloudCallback)
        # rospy.Subscriber('/camera/rgb/camera_info', CameraInfo, self.cameraRGBInfoCallback)
        # Kinect 2
        rospy.Subscriber('/head_mount_kinect/rgb/image', Image, self.imageCallback)
        rospy.Subscriber('/head_mount_kinect/depth_registered/points', PointCloud2, self.cloudCallback)
        rospy.Subscriber('/head_mount_kinect/rgb/camera_info', CameraInfo, self.cameraRGBInfoCallback)

        # spin() simply keeps python from exiting until this node is stopped
        # rospy.spin()

    def getRecentPoint(self, index):
        if index >= self.markerRecentCount():
            return None
        return self.features[index].recent3DPosition

    def getAllRecentPoints(self):
        if self.markerRecentCount() <= 0:
            return None
        return [feat.recent3DPosition for feat in self.features if feat.isNovel]

    def getAllMarkersWithHistory(self):
        if self.markerRecentCount() <= 0:
            return None
        featureSet = []
        for feature in self.features:
            if feature.isNovel:
                featureSet.append(feature)
        return featureSet

    def markerRecentCount(self):
        if self.features is None:
            return 0
        return len([feat for feat in self.features if feat.isNovel])

    def determineGoodFeatures(self, imageGray):
        # Take a frame and find corners in it
        self.prevFeats = cv2.goodFeaturesToTrack(imageGray, mask=None, **self.feature_params)
        self.features = np.array([feature(self.prevFeats[i][0], self) for i in xrange(len(self.prevFeats))])

    def opticalFlow(self, imageGray):
        newFeats, status, error = cv2.calcOpticalFlowPyrLK(self.prevGray, imageGray, self.prevFeats, None, **self.lk_params)
        statusRemovals = [i for i, s in enumerate(status) if s == 0]

        # Update all features
        for i, feat in enumerate(self.features):
            feat.update(newFeats[i][0])

        # Remove all features that are no longer being tracked (ie. status == 0)
        self.features = np.delete(self.features, statusRemovals, axis=0)
        newFeats = np.delete(newFeats, statusRemovals, axis=0)

        # Define features as novel if they meet a given criteria
        for feat in self.features:
            # Consider features that have traveled 15 cm
            if feat.distance >= 0.15:
                feat.isNovel = True

        self.prevFeats = newFeats

    def drawOnImage(self, image):
        # Cluster feature points
        points = []
        for feat in self.features:
            if feat.isNovel:
                points.append(feat.recent2DPosition)
        points = np.array(points)

        # Perform dbscan clustering
        X = StandardScaler().fit_transform(points)
        labels = self.dbscan.fit_predict(X)

        unique_labels = set(labels)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))*255
        # Drop alpha channel
        colors = np.delete(colors, -1, 1)
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0]

            # Draw feature points
            clusterPoints = points[labels==k]
            for point in clusterPoints:
                a, b = point
                if a >= self.cameraWidth or b >= self.cameraHeight:
                    # Red used for features not within the camera's dimensions
                    col = [1, 0, 0]
                cv2.circle(image, (a, b), 5, col, -1)

        return image

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
        for feature in self.features:
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
        points = pc2.read_points(self.pointCloud, field_names=('x', 'y', 'z'), skip_nans=False, uvs=[[x, y]])
        for point in points:
            px, py, depth = point
            break
        if any([math.isnan(v) for v in [px, py, depth]]):
            # print 'NaN! Feature', ps, py, depth
            return None

        xyz = None
        # Transpose point to targetFrame
        if self.targetFrame is not None:
            xyz = np.dot(self.transMatrix, np.array([px, py, depth, 1.0]))[:3]

        return xyz

    def imageCallback(self, data):
        # Grab image from Kinect sensor
        try:
            image = self.bridge.imgmsg_to_cv(data)
            image = np.asarray(image[:,:])
            # print data.header.frame_id
        except CvBridgeError, e:
            print e
            return
        # Grab image from video input
        # video = cv2.VideoCapture('../images/cabinet.mov')
        # ret, image = video.read()

        # Convert to grayscale
        imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find frameId for transformations and determine a good set of starting features
        if self.frameId is None:
            # Grab frame id for later transformations
            self.frameId = data.header.frame_id
            if self.targetFrame is not None:
                self.transformer.waitForTransform(self.targetFrame, self.frameId, rospy.Time(0), rospy.Duration(5.0))
                trans, rot = self.transformer.lookupTransform(self.targetFrame, self.frameId, rospy.Time(0))
                self.transMatrix = np.dot(tf.transformations.translation_matrix(trans), tf.transformations.quaternion_matrix(rot))
            # Determine initial set of features
            self.determineGoodFeatures(imageGray)
            self.prevGray = imageGray
            return

        if len(self.features) > 0:
            self.opticalFlow(imageGray)
            if self.visual:
                self.publishFeatures()
                image = self.drawOnImage(image)
                cv2.imshow('Image window', image)
                cv2.waitKey(30)

        self.updateNumber += 1

        self.prevGray = imageGray

        # Call our caller now that new data has been processed
        if self.caller is not None:
            self.caller()

    def cloudCallback(self, data):
        # Store PointCloud2 data for use when determining 3D locations
        self.pointCloud = data

    def cameraRGBInfoCallback(self, data):
        self.cameraWidth = data.width
        self.cameraHeight = data.height

minDist = 0.015
maxDist = 0.03
class feature:
    def __init__(self, position, kanadeLucas):
        self.kanadeLucas = kanadeLucas
        self.startPosition = None
        self.recent2DPosition = np.array(position)
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
