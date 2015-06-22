#!/usr/bin/env python

__author__ = 'zerickson'

import numpy as np
import rospy
import roslib
roslib.load_manifest('ar_track_alvar')
import tf
from ar_track_alvar.msg import AlvarMarkers

class arTagPoint:
    def __init__(self, caller, targetFrame=None, tfListener=None):
        # Recent Markers in last frame
        self.recentMarkers = None
        # All markers with past history
        self.markers = dict()
        self.caller = caller
        # Transformations
        self.frameId = None
        if tfListener is None:
            self.transformer = tf.TransformListener()
        else:
            self.transformer = tfListener
        self.targetFrame = targetFrame
        self.transMatrix = None
        self.updateNumber = 0

        # 'rostopic info /ar_pose_marker' -> 'rosmsg show ar_track_alvar/AlvarMarkers'
        rospy.Subscriber('ar_pose_marker', AlvarMarkers, self.callback)

        # spin() simply keeps python from exiting until this node is stopped
        # rospy.spin()

    def callback(self, data):
        self.recentMarkers = data.markers
        if self.markerRecentCount() <= 0:
            return

        # Find frameId for transformations
        if self.frameId is None:
            self.frameId = self.recentMarkers[0].header.frame_id
            if self.targetFrame is not None:
                self.transformer.waitForTransform(self.targetFrame, self.frameId, rospy.Time(0), rospy.Duration(5.0))
                trans, rot = self.transformer.lookupTransform(self.targetFrame, self.frameId, rospy.Time(0))
                self.transMatrix = np.dot(tf.transformations.translation_matrix(trans), tf.transformations.quaternion_matrix(rot))

        # Update markers
        for m in self.recentMarkers:
            index = m.id
            if index not in [0, 7]:
                continue
            position = m.pose.pose.position
            point = self.transposePoint([position.x, position.y, position.z])
            marker = self.markers.get(index)
            if marker is None:
                print 'New marker! Index:', index
                self.markers[index] = feature(point)
            else:
                marker.update(point)

        self.updateNumber += 1

        # Call our caller now that new data has been collected
        if self.caller is not None:
            self.caller()

    def getRecentPoint(self, index):
        if index >= self.markerRecentCount():
            return None
        marker = self.recentMarkers[index].pose.pose.position
        point = [marker.x, marker.y, marker.z]
        return self.transposePoint(point)

    def getAllRecentPoints(self):
        if self.markerRecentCount() == 0:
            return None
        points = []
        for i in xrange(self.markerRecentCount()):
            points.append(self.getRecentPoint(i))
        return points

    def getAllMarkersWithHistory(self):
        if len(self.markers) <= 0:
            return None
        markerSet = []
        for marker in self.markers.values():
            markerSet.append(marker)
        return markerSet

    def markerRecentCount(self):
        if self.recentMarkers is None:
            return 0
        return len(self.recentMarkers)

    def transposePoint(self, point):
        if self.transMatrix is None:
            return point
        x, y, z = point
        xyz = np.dot(self.transMatrix, np.array([x, y, z, 1.0]))[:3]
        return xyz

minDist = 0.005
maxDist = 0.02
class feature:
    def __init__(self, position):
        # position = np.array(position)
        self.recentPosition = position
        self.history = [position]
        self.lastHistoryPosition = position
        self.lastHistoryCount = 0

    def update(self, newPosition):
        # newPosition = np.array(newPosition)
        self.recentPosition = newPosition

        # Check if the point has traveled far enough to add a new history point
        dist = np.linalg.norm(self.recentPosition - self.lastHistoryPosition)
        if minDist <= dist <= maxDist:
            self.history.append(self.recentPosition)
            self.lastHistoryPosition = self.recentPosition

    def isAvailableForNewPath(self):
        if len(self.history) - self.lastHistoryCount >= 5:
            self.lastHistoryCount = len(self.history)
            return True
        return False


''' ar_track_alvar/AlvarMarkers data
std_msgs/Header header
  uint32 seq
  time stamp
  string frame_id
ar_track_alvar/AlvarMarker[] markers
  std_msgs/Header header
    uint32 seq
    time stamp
    string frame_id
  uint32 id
  uint32 confidence
  geometry_msgs/PoseStamped pose
    std_msgs/Header header
      uint32 seq
      time stamp
      string frame_id
    geometry_msgs/Pose pose
      geometry_msgs/Point position
        float64 x
        float64 y
        float64 z
      geometry_msgs/Quaternion orientation
        float64 x
        float64 y
        float64 z
        float64 w
'''
