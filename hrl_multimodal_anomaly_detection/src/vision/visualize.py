#!/usr/bin/env python

__author__ = 'zerickson'

import cv2
import time
import rospy
import operator
import numpy as np
import cPickle as pickle
from scipy import ndimage
import matplotlib.pyplot as plt

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
        # print point
        p.x, p.y, p.z = point
        marker.points.append(p)
    publisher.publish(marker)

dbscan = DBSCAN(eps=0.12, min_samples=10)
fileName = '/home/zerickson/Recordings/fancyFruits_scooping_07-07-2015_16-42-15/iteration_0_success.pkl'

def readDepth():
    with open(fileName, 'rb') as f:
        data = pickle.load(f)
        visual = data['visual_points']
        times = data['visual_time']
        # print visual
        for (pointSet, gripper, spoon), t in zip(visual, times):
            # print 'Number of points:', len(pointSet)
            print 'Time:', t
            gripper = np.array(gripper)
            spoon = np.array(spoon)

            # Check for invalid points
            pointSet = pointSet[np.linalg.norm(pointSet, axis=1) < 5]

            # Determine a line between the gripper and spoon
            directionVector = spoon - gripper
            linePoints = gripper + [t*directionVector for t in np.linspace(0, 1, 5)]

            # Find points within a sphere of radius 6 cm around each point on the line
            nearbyPoints = None
            for linePoint in linePoints:
                pointsNear = np.linalg.norm(pointSet - linePoint, axis=1) < 0.06
                nearbyPoints = nearbyPoints + pointsNear if nearbyPoints is not None else pointsNear

            # Points near spoon
            clusterPoints = pointSet[nearbyPoints]
            # Points outside of spoon radius
            nonClusterPoints = pointSet[nearbyPoints == False]

            publishPoints('points', clusterPoints, g=1.0)
            publishPoints('nonpoints', nonClusterPoints, r=1.0)

            publishPoints('gripper', [gripper], size=0.05, g=1.0, b=1.0)
            publishPoints('spoon', [spoon], size=0.05, b=1.0)
            time.sleep(0.25)

def readVisual():
    # fgbg = cv2.BackgroundSubtractorMOG()
    with open(fileName, 'rb') as f:
        data = pickle.load(f)
        visual = data['visual_points']
        print len(visual)
        kernel = np.ones((3, 3), np.uint8)
        kernel[[0, 0, 2, 2], [0, 2, 0, 2]] = 0
        print kernel
        for image, points, gripper, spoon in visual:
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            #
            # image = cv2.Canny(image, 200, 200)
            # image = 255 - image
            #
            # # image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 0)
            # # Opening
            # # image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=1)
            #
            # # Erosion
            # image = cv2.erode(image, kernel, iterations=1)
            # # Dilation
            # # image = cv2.dilate(image, kernel, iterations=1)
            # # Closing
            # image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=1)
            #
            # # image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            #
            # # image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
            #
            # # global thresholding
            # # ret1, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
            #
            # # Otsu's thresholding
            # # ret2, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            #
            # # # Otsu's thresholding after Gaussian filtering
            # # image = cv2.GaussianBlur(image, (5, 5), 0)
            # # ret3, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            #
            # contourImage = image.copy()
            # contours, hierarchy = cv2.findContours(contourImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #
            # grip = np.array(gripper)
            # minIndex = -1
            # minDist = 20000
            # for index, contour in enumerate(contours):
            #     # Make sure contour isn't too small or too large
            #     if len(contour) < 20 or len(contour) > 100:
            #         continue
            #     contour = np.reshape(contour, (len(contour), 2))
            #     # Make sure contour center is well above our gripper or too far to the left
            #     center = np.mean(contour, axis=0)
            #     if center[0] < grip[0] or center[1] < grip[1] - 10:
            #         continue
            #     # Find the point in this contour that is closest to our gripper
            #     distances = np.linalg.norm(grip - contour, axis=1)
            #     if np.max(distances) > 50:
            #         continue
            #     dist = np.min(distances)
            #     if dist < minDist:
            #         minDist = dist
            #         minIndex = index
            #
            # image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            #
            # if minIndex >= 0:
            #     print contours[minIndex].shape
            #     cv2.drawContours(image, contours, minIndex, (255, 0, 0), thickness=-1)
            #
            # rgb = [255, 128, 0]
            # cv2.circle(image, (int(gripper[0]), int(gripper[1])), 5, rgb, -1)
            # rgb = [0, 128, 255]
            # cv2.circle(image, (int(spoon[0]), int(spoon[1])), 5, rgb, -1)
            if points is None:
                continue
            for point in points.values():
                # Get the non global point
                p = point[1]
                rgb = [0, 255, 0]
                cv2.circle(image, (int(p[0]), int(p[1])), 5, rgb, -1)
            cv2.imshow('Image window', image)
            cv2.waitKey(250)

readDepth()
# readVisual()
