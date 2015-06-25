#!/usr/bin/env python

__author__ = 'zerickson'

import cv2
import math
import rospy
import numpy as np
try :
    import sensor_msgs.point_cloud2 as pc2
except:
    import point_cloud2 as pc2
from visualization_msgs.msg import Marker
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import Point, PointStamped
from nav_msgs.msg import Odometry
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

class kanadeLucasPoint:
    def __init__(self, caller, targetFrame=None, publish=False, visual=False, tfListener=None):
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
        self.pinholeCamera = None
        self.rgbCameraFrame = None

        self.dbscan = DBSCAN(eps=5, min_samples=10)
        self.dbscan2D = DBSCAN(eps=0.6, min_samples=10)
        self.N = 30

        # XBox 360 Kinect
        # rospy.Subscriber('/camera/rgb/image_color', Image, self.imageCallback)
        # rospy.Subscriber('/camera/depth_registered/points', PointCloud2, self.cloudCallback)
        # rospy.Subscriber('/camera/rgb/camera_info', CameraInfo, self.cameraRGBInfoCallback)
        # Kinect 2
        # rospy.Subscriber('/head_mount_kinect/rgb_lowres/image', Image, self.imageCallback)
        # rospy.Subscriber('/head_mount_kinect/depth_registered/points', PointCloud2, self.cloudCallback)
        # rospy.Subscriber('/head_mount_kinect/rgb_lowres/camera_info', CameraInfo, self.cameraRGBInfoCallback)
        # rospy.Subscriber('/l_gripper_palm_pose_ground_truth', Odometry, self.leftGripperCallback)
        # PR2 Simulated
        rospy.Subscriber('/head_mount_kinect/rgb/image_color', Image, self.imageCallback)
        rospy.Subscriber('/head_mount_kinect/depth_registered/points', PointCloud2, self.cloudCallback)
        rospy.Subscriber('/head_mount_kinect/rgb/camera_info', CameraInfo, self.cameraRGBInfoCallback)

        # spin() simply keeps python from exiting until this node is stopped
        # rospy.spin()

    def getRecentPoint(self, index):
        if index >= self.markerRecentCount():
            return None
        return self.features[index].recent3DPosition

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
        if self.features is None:
            return 0
        return len([feat for feat in self.features if feat.isNovel])

    def getNovelAndClusteredFeatures(self, returnFeatures=False):
        # Cluster feature points
        points = []
        feats = []
        for feat in self.features:
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

        if returnFeatures:
            # Return a list of features
            return [feat for i, feat in enumerate(feats) if labels[i] != -1]
        else:
            # Return a dictionary of indices and 3D points
            return {feat.index: feat.recent3DPosition for i, feat in enumerate(feats) if labels[i] != -1}

    def determineGoodFeatures(self, imageGray):
        # Take a frame and find corners in it
        self.prevFeats = cv2.goodFeaturesToTrack(imageGray, mask=None, **self.feature_params)
        self.features = np.array([feature(i, self.prevFeats[i][0], self) for i in xrange(len(self.prevFeats))])

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
        points = [feat.recent2DPosition for feat in self.features]# if feat.isNovel]
        if not points:
            # No novel features
            return image
        points = np.array(points)

        # Perform dbscan clustering
        X = StandardScaler().fit_transform(points)
        labels = self.dbscan2D.fit_predict(X)

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
                    col = [0, 0, 255]
                cv2.circle(image, (a, b), 5, col, -1)

        # Add position of left gripper to image
        if self.lGripperTranslation is None:
            return image
        # Project 3D position to 2D
        px, py = self.pinholeCamera.project3dToPixel(self.lGripperTranslation)
        # Draw an orange point on image for gripper
        cv2.circle(image, (int(px), int(py)), 10, [0, 125, 255], -1)

        # Draw a bounding box around spoon (or left gripper)
        lowX, highX, lowY, highY = self.boundingBox((px, py))
        cv2.rectangle(image, (int(lowX), int(lowY)), (int(highX), int(highY)), color=[0, 150, 75], thickness=5)

        return image

    # Finds a bounding box around a given point
    # Returns coordinates (lowX, highX, lowY, highY)
    def boundingBox(self, point):
        boxHalfWidth = self.cameraWidth/8.0
        boxHalfHeight = self.cameraHeight/6.0
        px, py = point

        # Determine X coordinates of bounding box
        if px <= boxHalfWidth:
            lowX = 0
        elif px >= self.cameraWidth - boxHalfWidth - 1:
            lowX = self.cameraWidth - 2*boxHalfWidth - 1
        else:
            lowX = px - boxHalfWidth
        highX = lowX + 2*boxHalfWidth

        # Determine Y coordinates of bounding box
        if py <= boxHalfHeight:
            lowY = 0
        elif py >= self.cameraHeight - boxHalfHeight - 1:
            lowY = self.cameraHeight - 2*boxHalfHeight - 1
        else:
            lowY = py - boxHalfHeight
        highY = lowY + 2*boxHalfHeight

        return lowX, highX, lowY, highY

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

        # Determine a bounding box around spoon (or left gripper) to narrow search area
        if self.lGripperTranslation is None:
            return
        # Find 2D location of gripper
        px, py = self.pinholeCamera.project3dToPixel(self.lGripperTranslation)

        # Find a bounding box
        lowX, highX, lowY, highY = [int(x) for x in self.boundingBox((px, py))]
        print lowX, highX, lowY, highY, imageGray.shape

        # Crop imageGray to bounding box size
        imageGray = imageGray[lowY:highY, lowX:highX]
        print imageGray.shape

        # Find frameId for transformations and determine a good set of starting features
        if self.frameId is None or self.features is None:
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
            return

        if len(self.features) > 0:
            self.opticalFlow(imageGray)
            if self.publish:
                self.publishFeatures()
        if self.visual:
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
        except tf.ExtrapolationException:
            pass

minDist = 0.015
maxDist = 0.03
class feature:
    def __init__(self, index, position, kanadeLucas):
        self.index = index
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
