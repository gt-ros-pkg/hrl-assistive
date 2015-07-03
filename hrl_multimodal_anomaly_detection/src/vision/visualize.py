#!/usr/bin/env python

__author__ = 'zerickson'

import time
import rospy
import cPickle as pickle

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

import roslib
roslib.load_manifest('hrl_multimodal_anomaly_detection')
from hrl_multimodal_anomaly_detection.msg import Circle, Rectangle, ImageFeatures



# ROS publisher for data points
rospy.init_node('visualization')
publisher = rospy.Publisher('visualization_marker', Marker)
publisher2D = rospy.Publisher('image_features', ImageFeatures)

def publishPoints(name, points, size=0.01, r=0.0, g=0.0, b=0.0, a=1.0):
    marker = Marker()
    marker.header.frame_id = '/torso_lift_link'
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
        p = Point()
        # print point
        p.x, p.y, p.z = point
        marker.points.append(p)
    publisher.publish(marker)

fileName = '/home/zerickson/Downloads/full_sixth_run_SHORTER_PAUSES_scooping_07-03-2015_15-51-33/iteration_1_success.pkl'
with open(fileName, 'rb') as f:
    data = pickle.load(f)
    visual = data['visual_points']
    # print visual
    for pointSet in visual:
        print 'Number of points:', len(pointSet)
        publishPoints('points', pointSet, g=1.0)
        time.sleep(1)
