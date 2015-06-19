#!/usr/bin/env python

__author__ = 'zerickson'

import rospy
import numpy as np
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from arTagPoint import arTagPoint
# from kanadeLucasPoint import kanadeLucasPoint

import kinectCircularPath as circularPath
import kinectLinearPath as linearPath

class visionTracker:
    def __init__(self, useARTags=True, shouldSpin=False, visual=False):
        self.publisher = rospy.Publisher('visualization_marker', Marker)
        self.lastPosition = None
        self.lastPosition2 = None
        self.points = None
        self.normals = None
        self.lastUpdate = 0
        self.visual = visual
        self.newDataAvailable = False

        if shouldSpin:
            rospy.init_node('listener_kinect')
        if useARTags:
            self.tracker = arTagPoint(self.spinner, '/camera_link')
        # else:
            # self.tracker = kanadeLucasPoint(self.spinner, True)
            # self.tracker = kanadeLucasPoint(self.multiFeature, True)
        if shouldSpin:
            rospy.spin()

    def getLogData(self):
        if not self.newDataAvailable:
            return None
        self.newDataAvailable = False
        return self.tracker.getRecentPoint(0)

    def spinner(self):
        markers = self.tracker.getAllMarkersWithHistory()
        if markers is None:
            return

        for index, marker in enumerate(markers):
            # Verify there are enough new points before regenerating paths
            if len(marker.history) - marker.lastHistoryCount >= 5:
                marker.lastHistoryCount = len(marker.history)
                # Find a linear fit of the points
                endPoints, linearError = linearPath.calcLinearPath(marker.history, verbose=False, plot=False)

                radius, centerPoint, normal, circularError, synthetic = circularPath.calcCircularPath(marker.history, normal=None, maxRadius=10, verbose=False, plot=False)

                # Compare whether the linear or circular path provides a better fit
                if linearError < circularError:
                    self.publishLinearPath(endPoints, index)
                else:
                    self.publishCircularPath(centerPoint, normal, synthetic, index)

                self.publishDataPoints(marker.history, index)

    def multiFeature(self):
        features, status = self.tracker.getAllPoints('/camera_link')
        if features is None:
            return

        # Append feature list
        if self.points is None:
            self.points = [features]
        else:
            self.points.append(features)

        # After several new points, update and publish a visual path in ROS (rviz)
        if self.visual and len(self.points) - self.lastUpdate >= 30:
            self.lastUpdate = len(self.points)

            lineCount = 0
            circularCount = 0
            centers = []
            normals = []
            # Loop through all features which are still active (status is True)
            for i in [i for i, x in enumerate(status) if x]:
                feats = []
                for featSet in self.points:
                    if i < len(featSet) and featSet[i] is not None:
                        feats.append(featSet[i])
                if len(feats) < 5:
                    continue
                feats = np.array(feats)

                # Find a linear fit of the features
                endPoints, linearError = linearPath.calcLinearPath(feats, verbose=True, plot=False)

                radius, centerPoint, normal, circularError, synthetic = circularPath.calcCircularPath(feats, normal=None, maxRadius=10, verbose=True, plot=False)

                # Compare whether the linear or circular path provides a better fit
                if linearError < circularError:
                    lineCount += 1
                else:
                    circularCount += 1
                    centers.append(centerPoint)
                    normals.append(normal)

            if circularCount > 0:
                normal = np.mean(normals, axis=0)
                center = np.mean(centers, axis=0)
                self.publishCircularPathKanade(center, normal)

    def appendDataPoint(self, point, point2):
        if self.points is None:
            self.points = np.array(point)
        else:
            self.points = np.vstack((self.points, point))
        self.lastPosition = point

        if point2 is not None and (self.lastPosition2 is None or self.minDist/2.0 <= np.linalg.norm(point2 - self.lastPosition2) <= self.maxDist*2.0):
            # We have 2 AR markers, thus we can find the normal of a circular path
            normal = point - point2
            # Normalize the normal vector. Yeah I know, normalizing the normal :)
            normal = normal / np.linalg.norm(normal)
            if self.normals is None:
                self.normals = np.array(normal)
            else:
                self.normals = np.vstack((self.normals, normal))
            self.lastPosition2 = point2

        self.newDataAvailable = True
        # rospy.loginfo(rospy.get_caller_id() + '\tI heard: \n%s', str(point))

    def publishLinearPath(self, endPoints, index):
        startPoint, endPoint = endPoints
        # Display best fit line through points (linear axis of translation)
        marker = Marker()
        marker.header.frame_id = '/camera_link'
        marker.ns = 'axis_vector_%d' % index
        marker.type = marker.LINE_LIST
        marker.action = marker.ADD
        marker.scale.x = 0.02
        # Green color
        marker.color.a = 1.0
        marker.color.g = 1.0
        # Define start and end points for our normal vector
        start = Point()
        start.x = startPoint[0]
        start.y = startPoint[1]
        start.z = startPoint[2]
        marker.points.append(start)
        end = Point()
        end.x = endPoint[0]
        end.y = endPoint[1]
        end.z = endPoint[2]
        marker.points.append(end)
        self.publisher.publish(marker)

    def publishCircularPath(self, centerPoint, normal, synthetic, index):
        # Display normal vector (axis of rotation)
        marker = Marker()
        marker.header.frame_id = '/camera_link'
        marker.ns = 'axis_vector_%d' % index
        marker.type = marker.LINE_LIST
        marker.action = marker.ADD
        # print normal
        marker.scale.x = 0.02
        # Green color
        marker.color.a = 1.0
        marker.color.g = 1.0
        # Define start and end points for our normal vector
        start = Point()
        start.x = centerPoint[0]
        start.y = centerPoint[1]
        start.z = centerPoint[2]
        marker.points.append(start)
        end = Point()
        end.x = centerPoint[0] + normal[0]
        end.y = centerPoint[1] + normal[1]
        end.z = centerPoint[2] + normal[2]
        marker.points.append(end)
        self.publisher.publish(marker)

        # Display all points on the estimated circular path
        pointsMarker = Marker()
        pointsMarker.header.frame_id = '/camera_link'
        pointsMarker.ns = 'circularPoints_%d' % index
        pointsMarker.type = pointsMarker.LINE_STRIP
        pointsMarker.action = pointsMarker.ADD
        pointsMarker.scale.x = 0.005
        pointsMarker.scale.y = 0.005
        pointsMarker.color.a = 1.0
        pointsMarker.color.r = 1.0
        pointsMarker.color.g = 0.5
        for point in synthetic:
            p = Point()
            p.x = point[0]
            p.y = point[1]
            p.z = point[2]
            pointsMarker.points.append(p)

        self.publisher.publish(pointsMarker)

    def publishDataPoints(self, points, index):
        # Display all points that were used to generate the estimated path
        pointsMarker = Marker()
        pointsMarker.header.frame_id = '/camera_link'
        pointsMarker.ns = 'path_points_%d' % index
        pointsMarker.type = pointsMarker.POINTS
        pointsMarker.action = pointsMarker.ADD
        pointsMarker.scale.x = 0.01
        pointsMarker.scale.y = 0.01
        pointsMarker.color.a = 1.0
        pointsMarker.color.b = 0.5
        pointsMarker.color.r = 0.5
        for point in points:
            p = Point()
            p.x = point[0]
            p.y = point[1]
            p.z = point[2]
            pointsMarker.points.append(p)

        self.publisher.publish(pointsMarker)

    def publishCircularPathKanade(self, centerPoint, normal):
        # Display normal vector (axis of rotation)
        marker = Marker()
        marker.header.frame_id = '/camera_link'
        marker.ns = 'axis_vector'
        marker.type = marker.LINE_LIST
        marker.action = marker.ADD
        # print normal
        marker.scale.x = 0.02
        # Green color
        marker.color.a = 1.0
        marker.color.g = 1.0
        # Define start and end points for our normal vector
        start = Point()
        start.x = centerPoint[0]
        start.y = centerPoint[1]
        start.z = centerPoint[2]
        marker.points.append(start)
        end = Point()
        end.x = centerPoint[0] + normal[0]
        end.y = centerPoint[1] + normal[1]
        end.z = centerPoint[2] + normal[2]
        marker.points.append(end)
        self.publisher.publish(marker)
