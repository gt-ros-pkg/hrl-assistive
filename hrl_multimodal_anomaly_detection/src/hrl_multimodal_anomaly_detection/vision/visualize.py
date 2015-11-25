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

from skimage.segmentation import slic, felzenszwalb, quickshift
from skimage.segmentation import mark_boundaries

# Clustering
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

import roslib
roslib.load_manifest('hrl_multimodal_anomaly_detection')
import tf
from hrl_multimodal_anomaly_detection.msg import Circle, Rectangle, ImageFeatures



# ROS publisher for data points
rospy.init_node('visualization')
publisher = rospy.Publisher('visualization_marker', Marker)
publisher2D = rospy.Publisher('image_features', ImageFeatures)

def publishPoints(name, points, size=0.0025, r=0.0, g=0.0, b=0.0, a=1.0):
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

kernel = np.ones((3, 3), np.uint8)
# kernel[[0, 0, 2, 2], [0, 2, 0, 2]] = 0
def displayImage(image, gripper, spoon):
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    imageGray = cv2.Canny(imageGray, 200, 200)
    imageGray = 255 - imageGray

    # image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 0)
    # Opening
    # image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=1)

    # Erosion
    imageGray = cv2.erode(imageGray, kernel, iterations=1)
    # Dilation
    # image = cv2.dilate(image, kernel, iterations=1)
    # Closing
    imageGray = cv2.morphologyEx(imageGray, cv2.MORPH_CLOSE, kernel, iterations=1)

    # image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    # global thresholding
    # ret1, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Otsu's thresholding
    # ret2, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # # Otsu's thresholding after Gaussian filtering
    # image = cv2.GaussianBlur(image, (5, 5), 0)
    # ret3, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # contourImage = image.copy()
    contours, hierarchy = cv2.findContours(imageGray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    grip = np.array(gripper)
    minIndex = -1
    minDist = 20000
    for index, contour in enumerate(contours):
        # Make sure contour isn't too small or too large
        if len(contour) < 20 or len(contour) > 100:
            continue
        contour = np.reshape(contour, (len(contour), 2))
        # Make sure contour center is well above our gripper or too far to the left
        center = np.mean(contour, axis=0)
        if center[0] < grip[0] or center[1] < grip[1] - 10:
            continue
        # Find the point in this contour that is closest to our gripper
        distances = np.linalg.norm(grip - contour, axis=1)
        if np.max(distances) > 50:
            continue
        dist = np.min(distances)
        if dist < minDist:
            minDist = dist
            minIndex = index

    # image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if minIndex >= 0:
        print contours[minIndex].shape
        cv2.drawContours(image, contours, minIndex, (255, 0, 0), thickness=-1)

    rgb = [255, 128, 0]
    cv2.circle(image, (int(gripper[0]), int(gripper[1])), 5, rgb, -1)
    # rgb = [0, 128, 255]
    # cv2.circle(image, (int(spoon[0]), int(spoon[1])), 5, rgb, -1)
    # if points is None:
    #     continue
    # for point in points.values():
    #     # Get the non global point
    #     p = point[1]
    #     rgb = [0, 255, 0]
    #     cv2.circle(image, (int(p[0]), int(p[1])), 5, rgb, -1)
    cv2.imshow('Image window', image)
    cv2.waitKey(200)


dbscan = DBSCAN(eps=0.12, min_samples=10)
# fileName = '/home/zerickson/Recordings/pinkSpoon_scooping_fvk_07-10-2015_18-30-38/iteration_0_success.pkl'
fileName = '/home/zerickson/Recordings/bowl2Stage1Train_scooping_fvk_07-24-2015_08-24-58/iteration_4_success.pkl'

def readDepth():
    with open(fileName, 'rb') as f:
        data = pickle.load(f)
        visual = data['visual_points']
        times = data['visual_time']
        time.sleep(3)
        for (pointSet, mic, spoon, bowlPosition, bowlPositionKinect, (bowlX, bowlY), bowlToKinectMat, (targetTrans, targetRot)), timeStamp in zip(visual, times):
            print 'Time:', timeStamp

            # displayImage(image, gripperTF[0], spoon)
            # continue

            # segments = slic(image, sigma=5)
            # segments_fz = felzenszwalb(image, scale=50, sigma=1, min_size=50)
            # segments_slic = slic(image, n_segments=250, compactness=10, sigma=1)
            # segments_quick = quickshift(image, kernel_size=3, max_dist=6, ratio=1)
            # image = mark_boundaries(image, segments_quick)


            # cv2.imshow('Image window', image)
            # cv2.waitKey(200)
            # continue

            # Transform mic and spoon into torso_lift_link
            targetMatrix = np.dot(tf.transformations.translation_matrix(targetTrans), tf.transformations.quaternion_matrix(targetRot))
            mic = np.dot(targetMatrix, np.array([mic[0], mic[1], mic[2], 1.0]))[:3]
            spoon = np.dot(targetMatrix, np.array([spoon[0], spoon[1], spoon[2], 1.0]))[:3]

            pointSet = np.c_[pointSet, np.ones(len(pointSet))]
            pointSet = np.dot(targetMatrix, pointSet.T).T[:, :3]

            # Check for invalid points
            pointSet = pointSet[np.linalg.norm(pointSet, axis=1) < 5]

            # Find points within a sphere of radius 6 cm around the center of bowl
            nearbyPoints = np.linalg.norm(pointSet - bowlPosition, axis=1) < 0.11

            # # Determine a line between the gripper and spoon
            # directionVector = spoon - mic
            # linePoints = mic + [t*directionVector for t in np.linspace(0, 1, 5)]
            #
            # # Find points within a sphere of radius 6 cm around each point on the line
            # nearbyPoints = None
            # for linePoint in linePoints:
            #     pointsNear = np.linalg.norm(pointSet - linePoint, axis=1) < 0.06
            #     nearbyPoints = nearbyPoints + pointsNear if nearbyPoints is not None else pointsNear

            # Points near spoon
            clusterPoints = pointSet[nearbyPoints]
            # Points outside of spoon radius
            nonClusterPoints = pointSet[nearbyPoints == False]

            if len(clusterPoints) <= 0:
                print 'Ahh no cluster points!'

            publishPoints('points', clusterPoints, g=1.0)
            publishPoints('nonpoints', nonClusterPoints, b=1.0)

            # publishPoints('bowl', [bowlPosition], size=0.05, r=1.0, g=1.0, b=1.0)
            # publishPoints('gripper', [mic], size=0.05, g=1.0, b=1.0)
            # publishPoints('spoon', [spoon], size=0.05, b=1.0)

            time.sleep(5)
            # time.sleep(0.15) if timeStamp < 19 else time.sleep(0.4)

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
