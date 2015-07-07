#!/usr/bin/env python

__author__ = 'zerickson'

import time
import numpy as np

import rospy
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from arTagPoint import arTagPoint
from kanadeLucasPoint import kanadeLucasPoint
from depthPerception import depthPerception
from depthPerceptionTrials import depthPerceptionTrials
from rgbPerception import rgbPerception
from wideStereoDepth import wideStereoDepth

from kinectDepth import kinectDepth

from cloudTrial import cloudTrial

from wideStereoRGB import wideStereoRGB

import kinectCircularPath as circularPath
import kinectLinearPath as linearPath


class visionTracker:
    def __init__(self, useARTags=True, targetFrame='/camera_link', shouldSpin=False, publish=False, visual=False, tfListener=None):
        self.publisher = rospy.Publisher('visualization_marker', Marker)
        self.lastPosition = None
        self.lastPosition2 = None
        self.points = None
        self.normals = None
        self.targetFrame = targetFrame
        self.lastUpdateNumber = 0
        self.lastTime = time.time()

        if shouldSpin:
            rospy.init_node('listener_kinect')
        # if useARTags:
        #     self.tracker = arTagPoint(self.spinner if publish else None, targetFrame=targetFrame, tfListener=tfListener)
        # else:
        #     self.tracker = kanadeLucasPoint(self.multispinner if publish else None, targetFrame=targetFrame, publish=publish, visual=visual, tfListener=tfListener)

        # self.tracker = depthPerception(targetFrame=targetFrame, visual=visual, tfListener=tfListener)
        # self.tracker = rgbPerception(targetFrame=targetFrame, visual=visual, tfListener=tfListener)
        # self.tracker = depthPerceptionTrials(targetFrame=targetFrame, visual=visual, tfListener=tfListener)

        # self.tracker = wideStereoDepth(targetFrame=targetFrame, visual=visual, tfListener=tfListener)
        # self.tracker = cloudTrial(False)

        # self.tracker = wideStereoRGB(targetFrame=targetFrame, visual=visual, tfListener=tfListener)

        self.tracker = kinectDepth(targetFrame=targetFrame, visual=visual, tfListener=tfListener)

        if shouldSpin:
            rospy.spin()

    def getLogData(self):
        if self.tracker.updateNumber <= self.lastUpdateNumber:
            return None
        self.lastUpdateNumber = self.tracker.updateNumber
        return self.tracker.getAllRecentPoints()

    # def spinner(self):
    #     markers = self.tracker.getAllMarkersWithHistory()
    #     if markers is None:
    #         return
    #
    #     for index, marker in enumerate(markers):
    #         # Verify there are enough new points before regenerating paths
    #         if marker.isAvailableForNewPath():
    #             # Find a linear fit of the points
    #             endPoints, linearError = linearPath.calcLinearPath(marker.history, verbose=False, plot=False)
    #
    #             radius, centerPoint, normal, circularError, synthetic = circularPath.calcCircularPath(marker.history, normal=None, maxRadius=10, verbose=False, plot=False)
    #
    #             # Compare whether the linear or circular path provides a better fit
    #             if linearError < circularError:
    #                 self.publishLinearPath(endPoints, index)
    #             else:
    #                 self.publishCircularPath(centerPoint, normal, synthetic, index)
    #
    #             self.publishDataPoints(marker.history, index)
    #
    # def multispinner(self):
    #     markers = self.tracker.getAllMarkersWithHistory()
    #     if markers is None:
    #         return
    #
    #     if time.time() - self.lastTime >= 5:
    #         self.lastTime = time.time()
    #         lines = 0
    #         ends = []
    #         circles = 0
    #         centers = []
    #         normals = []
    #         for index, marker in enumerate(markers):
    #             # Verify there are enough new points before regenerating paths
    #             # if not marker.isAvailableForNewPath():
    #             if len(marker.history) <= 5:
    #                 continue
    #             # Find a linear fit of the points
    #             endPoints, linearError = linearPath.calcLinearPath(marker.history, verbose=False, plot=False)
    #
    #             radius, centerPoint, normal, circularError, synthetic = circularPath.calcCircularPath(marker.history, normal=None, maxRadius=10, verbose=False, plot=False)
    #             if radius > 0.75:
    #                 lines += 1
    #                 ends.append(endPoints)
    #                 self.publishLinearPath(endPoints, index)
    #                 continue
    #
    #             # Compare whether the linear or circular path provides a better fit
    #             if linearError < circularError:
    #                 lines += 1
    #                 ends.append(endPoints)
    #                 self.publishLinearPath(endPoints, index)
    #             else:
    #                 circles += 1
    #                 centers.append(centerPoint)
    #                 normals.append(normal)
    #                 self.publishCircularPath(centerPoint, normal, synthetic, index)
    #
    #             self.publishDataPoints(marker.history, index)
    #
    #         print len(markers), circles, lines
    #         if circles > 0 and circles >= lines:
    #             self.publishNormal(np.mean(centers, axis=0), np.mean(normals, axis=0))
    #         elif lines > 0:
    #             ends = np.array(ends)
    #             leftEnd = [np.mean(ends[:, 0, 0]), np.mean(ends[:, 0, 1]), np.mean(ends[:, 0, 2])]
    #             rightEnd = [np.mean(ends[:, 1, 0]), np.mean(ends[:, 1, 1]), np.mean(ends[:, 1, 2])]
    #             self.publishLinearPath([leftEnd, rightEnd], 0)
    #             pass
    #
    # def publishLinearPath(self, endPoints, index):
    #     startPoint, endPoint = endPoints
    #     # Display best fit line through points (linear axis of translation)
    #     marker = Marker()
    #     marker.header.frame_id = self.targetFrame
    #     marker.ns = 'axis_vector_%d' % index
    #     marker.type = marker.LINE_LIST
    #     marker.action = marker.ADD
    #     marker.scale.x = 0.005
    #     # Green color
    #     marker.color.a = 1.0
    #     marker.color.g = 1.0
    #     # Define start and end points for our normal vector
    #     start = Point()
    #     start.x = startPoint[0]
    #     start.y = startPoint[1]
    #     start.z = startPoint[2]
    #     marker.points.append(start)
    #     end = Point()
    #     end.x = endPoint[0]
    #     end.y = endPoint[1]
    #     end.z = endPoint[2]
    #     marker.points.append(end)
    #     self.publisher.publish(marker)
    #
    # def publishCircularPath(self, centerPoint, normal, synthetic, index):
    #     # Display normal vector (axis of rotation)
    #     marker = Marker()
    #     marker.header.frame_id = self.targetFrame
    #     marker.ns = 'axis_vector_%d' % index
    #     marker.type = marker.LINE_LIST
    #     marker.action = marker.ADD
    #     # print normal
    #     marker.scale.x = 0.005
    #     # Green color
    #     marker.color.a = 1.0
    #     marker.color.g = 1.0
    #     # Define start and end points for our normal vector
    #     start = Point()
    #     start.x = centerPoint[0]
    #     start.y = centerPoint[1]
    #     start.z = centerPoint[2]
    #     marker.points.append(start)
    #     end = Point()
    #     end.x = centerPoint[0] + normal[0]
    #     end.y = centerPoint[1] + normal[1]
    #     end.z = centerPoint[2] + normal[2]
    #     marker.points.append(end)
    #     self.publisher.publish(marker)
    #
    #     # Display all points on the estimated circular path
    #     pointsMarker = Marker()
    #     pointsMarker.header.frame_id = self.targetFrame
    #     pointsMarker.ns = 'circularPoints_%d' % index
    #     pointsMarker.type = pointsMarker.LINE_STRIP
    #     pointsMarker.action = pointsMarker.ADD
    #     pointsMarker.scale.x = 0.005
    #     pointsMarker.scale.y = 0.005
    #     pointsMarker.color.a = 1.0
    #     pointsMarker.color.r = 1.0
    #     pointsMarker.color.g = 0.5
    #     for point in synthetic:
    #         p = Point()
    #         p.x = point[0]
    #         p.y = point[1]
    #         p.z = point[2]
    #         pointsMarker.points.append(p)
    #
    #     self.publisher.publish(pointsMarker)
    #
    # def publishDataPoints(self, points, index):
    #     # Display all points that were used to generate the estimated path
    #     pointsMarker = Marker()
    #     pointsMarker.header.frame_id = self.targetFrame
    #     pointsMarker.ns = 'path_points_%d' % index
    #     pointsMarker.type = pointsMarker.POINTS
    #     pointsMarker.action = pointsMarker.ADD
    #     pointsMarker.scale.x = 0.01
    #     pointsMarker.scale.y = 0.01
    #     pointsMarker.color.a = 1.0
    #     pointsMarker.color.b = 0.5
    #     pointsMarker.color.r = 0.5
    #     for point in points:
    #         p = Point()
    #         p.x = point[0]
    #         p.y = point[1]
    #         p.z = point[2]
    #         pointsMarker.points.append(p)
    #
    #     self.publisher.publish(pointsMarker)
    #
    # def publishNormal(self, centerPoint, normal):
    #     # Display normal vector (axis of rotation)
    #     marker = Marker()
    #     marker.header.frame_id = self.targetFrame
    #     marker.ns = 'axis_vector'
    #     marker.type = marker.LINE_LIST
    #     marker.action = marker.ADD
    #     # print normal
    #     marker.scale.x = 0.01
    #     # Green color
    #     marker.color.a = 1.0
    #     marker.color.g = 1.0
    #     # Define start and end points for our normal vector
    #     start = Point()
    #     start.x = centerPoint[0]
    #     start.y = centerPoint[1]
    #     start.z = centerPoint[2]
    #     marker.points.append(start)
    #     end = Point()
    #     end.x = centerPoint[0] + normal[0]
    #     end.y = centerPoint[1] + normal[1]
    #     end.z = centerPoint[2] + normal[2]
    #     marker.points.append(end)
    #     self.publisher.publish(marker)
