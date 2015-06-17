#!/usr/bin/env python

__author__ = 'zerickson'

import numpy as np
import rospy
import roslib
roslib.load_manifest('ar_track_alvar')
import tf
from ar_track_alvar.msg import AlvarMarkers

class arTagPoint:
    def __init__(self, caller):
        self.markers = None
        self.frameId = None
        self.caller = caller
        self.transformer = tf.TransformListener()

        # 'rostopic info /ar_pose_marker' -> 'rosmsg show ar_track_alvar/AlvarMarkers'
        rospy.Subscriber('ar_pose_marker', AlvarMarkers, self.callback)

        # spin() simply keeps python from exiting until this node is stopped
        # rospy.spin()

    def callback(self, data):
        self.markers = data.markers
        if self.markerCount() > 0:
            # Find frameId for transformations
            self.frameId = self.markers[0].header.frame_id
        # Call our caller now that new data has been collected
        self.caller()

    def getPoint(self, index, targetFrame):
        if index >= self.markerCount():
            return None
        marker = self.markers[index]
        point = marker.pose.pose.position
        x, y, z = np.array([point.x, point.y, point.z])

        # Transpose point to targetFrame
        if targetFrame is not None:
            trans, rot = self.transformer.lookupTransform(targetFrame, self.frameId, rospy.Time(0))
            mat = np.dot(tf.transformations.translation_matrix(trans), tf.transformations.quaternion_matrix(rot))
            xyz = tuple(np.dot(mat, np.array([x, y, z, 1.0])))[:3]
            x, y, z = xyz

        return np.array([x, y, z])

    def markerCount(self):
        if self.markers is None:
            return 0
        return len(self.markers)

''' ar_track_alvar/AlvarMarker data
header:
  seq: 0
  stamp:
    secs: 1433342469
    nsecs: 717930841
  frame_id: /camera_link
id: 0
confidence: 0
pose:
  header:
    seq: 0
    stamp:
      secs: 0
      nsecs: 0
    frame_id: ''
  pose:
    position:
      x: 0.534185739926
      y: 0.0230105213608
      z: -0.0610259223197
    orientation:
      x: 0.397652120585
      y: 0.00343235958882
      z: -0.506289901878
      w: 0.765200330083
'''
