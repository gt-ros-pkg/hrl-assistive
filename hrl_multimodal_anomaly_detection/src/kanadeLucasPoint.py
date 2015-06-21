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

import roslib
roslib.load_manifest('cv_bridge')
import tf
from cv_bridge import CvBridge, CvBridgeError

class kanadeLucasPoint:
    def __init__(self, caller, visual=False):
        self.markers = None
        self.caller = caller
        self.bridge = CvBridge()
        # ROS publisher for data points
        self.publisher = rospy.Publisher('visualization_marker', Marker)
        # params for ShiTomasi corner detection
        self.feature_params = dict(maxCorners=100, qualityLevel=0.1, minDistance=7, blockSize=7)
        # Parameters for lucas kanade optical flow
        self.lk_params = dict(winSize=(15,15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        # Create some random colors
        self.color = np.random.randint(0, 255, (100, 3))
        # Variables to store past iteration output
        self.prevGray = None
        self.prevFeats = None
        # List of features we are checking to see if they are novel
        self.novelQueue = None
        # Novel features part of the object that we want to track
        self.novelFeats = None
        # PointCloud2 data used for determining 3D location from 2D pixel location
        self.pointCloud = None
        # Frame id used for transformations
        self.imageFrameId = None
        self.cameraWidth = None
        self.cameraHeight = None
        self.transformer = tf.TransformListener()
        # Whether to display visual plots or not
        self.visual = visual

        rospy.Subscriber('/camera/rgb/image_color', Image, self.imageCallback)
        rospy.Subscriber('/camera/depth_registered/points', PointCloud2, self.cloudCallback)
        rospy.Subscriber('/camera/rgb/camera_info', CameraInfo, self.cameraRGBInfoCallback)

        # spin() simply keeps python from exiting until this node is stopped
        # rospy.spin()

    def getPoint(self, index, targetFrame):
        if index >= self.markerCount() or self.markerCount() < 15:
            return None
        feature = self.novelFeats[10]
        return self.get3DPointFromCloud(feature, targetFrame)

    def markerCount(self):
        if self.novelFeats is None:
            return 0
        return len(self.novelFeats)

    def determineGoodFeatures(self, imageGray):
        # Take a frame and find corners in it
        self.prevFeats = cv2.goodFeaturesToTrack(imageGray, mask=None, **self.feature_params)
        self.novelQueue = np.array([feature(self.prevFeats[i]) for i in xrange(len(self.prevFeats))])

    def opticalFlowQueue(self, imageGray):
        newFeats, status, error = cv2.calcOpticalFlowPyrLK(self.prevGray, imageGray, self.prevFeats, None, **self.lk_params)
        statusRemovals = [i for i, s in enumerate(status) if s == 0]

        # Update all features
        for i in xrange(len(newFeats)):
            self.novelQueue[i].frame += 1
            self.novelQueue[i].distance = np.linalg.norm(self.novelQueue[i].position - newFeats[i])

        # Remove all features that are no longer being tracked (ie. status == 0)
        self.novelQueue = np.delete(self.novelQueue, statusRemovals, axis=0)
        newFeats = np.delete(newFeats, statusRemovals, axis=0)

        # Remove specific features that match a given criteria
        removeList = []
        for i in xrange(len(self.novelQueue)):
            # Check if certain features have now become 'novel' by traveling a far enough distance
            if self.novelQueue[i].distance >= 50:
                if self.novelFeats is None:
                    self.novelFeats = np.array(newFeats[i])
                else:
                    self.novelFeats = np.vstack((self.novelFeats, newFeats[i]))
                removeList.append(i)
            # Else if certain features hit the frame limit, remove them
            # elif novelQueue[i].frame >= 40:
                # removeList.append(i)

        # Remove specified features from tracking queue process
        self.novelQueue = np.delete(self.novelQueue, removeList, axis=0)
        # oldFeats = np.delete(oldFeats, removeList, axis=0)
        newFeats = np.delete(newFeats, removeList, axis=0)

        self.prevFeats = newFeats

    def opticalFlowNovel(self, imageGray):
        novel, status, error = cv2.calcOpticalFlowPyrLK(self.prevGray, imageGray, self.novelFeats, None, **self.lk_params)
        if novel is None or status is None:
            return
        statusRemovals = [i for i, s in enumerate(status) if s == 0]
        # Remove all features that are no longer being tracked (ie. status == 0)
        novel = np.delete(novel, statusRemovals, axis=0)
        self.novelFeats = novel

    def drawOnImage(self, image, features):
        # Draw feature points
        for i, feature in enumerate(features):
            a, b = feature.ravel()
            cv2.circle(image, (a, b), 5, self.color[i].tolist(), -1)
        return image

    def publishFeatures(self, features, pointCloud):
        if self.cameraWidth is None:
            return
        # Display all novel (object) features that we are tracking.
        marker = Marker()
        marker.header.frame_id = pointCloud.header.frame_id
        marker.ns = 'points'
        marker.type = marker.POINTS
        marker.action = marker.ADD
        marker.scale.x = 0.01
        marker.scale.y = 0.01
        marker.color.a = 1.0
        marker.color.b = 1.0
        for feature in features:
            point = self.get3DPointFromCloud(feature, None)
            if point is None:
                continue
            x, y, depth = point
            # Create 3d point
            p = Point()
            p.x = x
            p.y = y
            p.z = depth
            marker.points.append(p)

        self.publisher.publish(marker)

    def get3DPointFromCloud(self, feature, targetFrame):
        if self.cameraWidth is None:
            return None
        # Grab x, y values from feature and verify they are within the image bounds
        x, y = feature
        x, y = int(x), int(y)
        if x >= self.cameraWidth or y >= self.cameraHeight:
            print 'x, y greater than camera size! Feature', x, y, self.cameraWidth, self.cameraHeight
            return None
        # Retrieve 3d location of feature from PointCloud2
        points = pc2.read_points(self.pointCloud, field_names=('x', 'y', 'z'), skip_nans=False, uvs=[[x, y]])
        for point in points:
            px, py, depth = point
            break
        if any([math.isnan(v) for v in [px, py, depth]]):
            # print 'NaN! Feature', ps, py, depth
            return None

        # Transpose point to targetFrame
        if targetFrame is not None:
            trans, rot = self.transformer.lookupTransform(targetFrame, self.pointCloud.header.frame_id, rospy.Time(0))
            mat = np.dot(tf.transformations.translation_matrix(trans), tf.transformations.quaternion_matrix(rot))
            xyz = tuple(np.dot(mat, np.array([px, py, depth, 1.0])))[:3]
            px, py, depth = xyz

        return np.array([px, py, depth])

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

        if self.prevGray is None:
            # Grab frame id for later transformations
            self.imageFrameId = data.header.frame_id
            # Determine initial set of features
            self.determineGoodFeatures(imageGray)
            self.prevGray = imageGray
            return

        if len(self.prevFeats) > 0:
            self.opticalFlowQueue(imageGray)

        if self.novelFeats is not None:
            self.opticalFlowNovel(imageGray)
            if self.visual:
                self.publishFeatures(self.novelFeats, self.pointCloud)
                image = self.drawOnImage(image, self.novelFeats)
                cv2.imshow('Image window', image)
                cv2.waitKey(30)

        self.prevGray = imageGray

        # Call our caller now that new data has been processed
        self.caller(self)

    def cloudCallback(self, data):
        # Store PointCloud2 data for use when determining 3D locations
        try:
            self.pointCloud = data
        except CvBridgeError, e:
            print e

    def cameraRGBInfoCallback(self, data):
        self.cameraWidth = data.width
        self.cameraHeight = data.height

class feature:
    def __init__(self, position):
        self.position = position
        self.frame = 0
        self.distance = 0.0

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
