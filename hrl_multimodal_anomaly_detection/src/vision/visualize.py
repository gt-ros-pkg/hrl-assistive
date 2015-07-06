#!/usr/bin/env python

__author__ = 'zerickson'

import cv2
import time
import rospy
import operator
import numpy as np
import cPickle as pickle

# Clustering
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

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
        print point
        p.x, p.y, p.z = point
        marker.points.append(p)
    publisher.publish(marker)

dbscan = DBSCAN(eps=0.12, min_samples=10)
fileName = '/home/zerickson/Downloads/rgbfun_scooping_07-06-2015_11-07-36/iteration_0_success.pkl'

def readDepth():
    with open(fileName, 'rb') as f:
        data = pickle.load(f)
        visual = data['visual_points']
        times = data['visual_time']
        # print visual
        for (pointSet, gripper, spoon), t in zip(visual, times):
            # print 'Number of points:', len(pointSet)
            print 'Time:', t

            # Check for invalid points
            pointSet = pointSet[np.linalg.norm(pointSet, axis=1) < 5]

            # Perform dbscan clustering
            X = StandardScaler().fit_transform(pointSet)
            labels = dbscan.fit_predict(X)

            index, closePoint = min(enumerate(np.linalg.norm(pointSet - np.array(gripper), axis=1)), key=operator.itemgetter(1))
            closeLabel = labels[index]
            while closeLabel == -1 and pointSet.size > 0:
                np.delete(pointSet, [index])
                np.delete(labels, [index])
                index, closePoint = min(enumerate(np.linalg.norm(pointSet - np.array(gripper), axis=1)), key=operator.itemgetter(1))
                closeLabel = labels[index]
            if pointSet.size <= 0:
                return
            clusterPoints = pointSet[labels==closeLabel]
            nonClusterPoints = pointSet[labels!=closeLabel]

            publishPoints('points', clusterPoints, g=1.0)
            publishPoints('nonpoints', nonClusterPoints, r=1.0)

            publishPoints('gripper', [gripper], size=0.05, g=1.0, b=1.0)
            publishPoints('spoon', [spoon], size=0.05, b=1.0)
            time.sleep(0.25)

def readVisual():
    with open(fileName, 'rb') as f:
        data = pickle.load(f)
        visual = data['visual_points']
        for image, points, gripper, spoon in visual:
            rgb = [255, 128, 0]
            cv2.circle(image, (int(gripper[0]), int(gripper[1])), 5, rgb, -1)
            rgb = [0, 128, 255]
            cv2.circle(image, (int(spoon[0]), int(spoon[1])), 5, rgb, -1)
            if points is None:
                continue
            for point in points.values():
                # Get the non global point
                p = point[1]
                rgb = [0, 255, 0]
                cv2.circle(image, (int(p[0]), int(p[1])), 5, rgb, -1)
            cv2.imshow('Image window', image)
            cv2.waitKey(100)

readDepth()
# readVisual()
