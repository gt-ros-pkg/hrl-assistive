#!/usr/bin/env python

__author__ = 'zerickson'

import time
import rospy
import numpy as np
from threading import Thread
import onlineHMMLauncher as onlineHMM

try :
    import sensor_msgs.point_cloud2 as pc2
except:
    import vision.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, CameraInfo
from geometry_msgs.msg import PoseStamped, WrenchStamped
from roslib import message

import roslib
roslib.load_manifest('hrl_multimodal_anomaly_detection')
import tf
import image_geometry
from cv_bridge import CvBridge, CvBridgeError
from sound_play.msg import SoundRequest
from sound_play.libsoundplay import SoundClient
from hrl_multimodal_anomaly_detection.msg import Circle, Rectangle, ImageFeatures

class onlineAnomalyDetection(Thread):
    def __init__(self, targetFrame=None, tfListener=None):
        super(onlineAnomalyDetection, self).__init__()
        self.daemon = True
        self.cancelled = False

        self.publisher2D = rospy.Publisher('image_features', ImageFeatures)
        self.cloudTime = time.time()
        self.pointCloud = None
        self.targetFrame = targetFrame

        # Data logging
        self.updateNumber = 0
        self.lastUpdateNumber = 0
        self.init_time = rospy.get_time()

        self.points3D = None

        if tfListener is None:
            self.transformer = tf.TransformListener()
        else:
            self.transformer = tfListener

        self.bridge = CvBridge()
        self.imageData = None
        self.targetMatrix = None

        # RGB Camera
        self.rgbCameraFrame = None
        self.cameraWidth = None
        self.cameraHeight = None
        self.pinholeCamera = None

        self.bowlPosition = None
        self.bowlPositionKinect = None
        self.bowlToKinectMat = None
        self.bowlX = None
        self.bowlY = None

        # Gripper
        self.lGripperPosition = None
        self.lGripperRotation = None
        self.lGripperTransposeMatrix = None
        self.mic = None
        self.grips = []
        # Spoon
        self.spoon = None

        # FT sensor
        self.force = None
        self.torque = None

        self.soundHandle = SoundClient()

        self.hmm, self.minVals, self.maxVals, self.forces, self.distances, self.angles, self.pdfs, self.times = onlineHMM.setupMultiHMM()
        self.anomalyOccured = False

        self.cloudSub = rospy.Subscriber('/head_mount_kinect/depth_registered/points', PointCloud2, self.cloudCallback)
        print 'Connected to Kinect depth'
        self.cameraSub = rospy.Subscriber('/head_mount_kinect/depth_lowres/camera_info', CameraInfo, self.cameraRGBInfoCallback)
        print 'Connected to Kinect camera info'
        self.bowlSub = rospy.Subscriber('hrl_feeding_task/manual_bowl_location', PoseStamped, self.bowlPoseManualCallback)
        self.forceSub = rospy.Subscriber('/netft_data', WrenchStamped, self.forceCallback)
        print 'Connected to FT sensor'

    def reset(self):
        pass

    def run(self):
        """Overloaded Thread.run, runs the update
        method once per every xx milliseconds."""
        rate = rospy.Rate(1000) # 25Hz, nominally.
        while not self.cancelled:
            if self.updateNumber > self.lastUpdateNumber:
                self.lastUpdateNumber = self.updateNumber
                self.processData()
                if not self.anomalyOccured:
                    (anomaly, error) = self.hmm.anomaly_check(self.forces, self.distances, self.angles, self.pdfs, -5)
                    if anomaly > 0:
                        self.anomalyOccured = True
                        self.soundHandle.play(2)
                        print 'AHH!! There is an anomaly at time stamp', rospy.get_time() - self.init_time, (anomaly, error)
            rate.sleep()

    def cancel(self):
        """End this timer thread"""
        self.cancelled = True
        self.cloudSub.unregister()
        self.cameraSub.unregister()
        self.bowlSub.unregister()
        self.forceSub.unregister()
        rospy.sleep(1.0)

    def scaling(self, x, minVal, maxVal, scale=1.0):
        return (x - minVal) / (maxVal - minVal) * scale

    def processData(self):
        if self.bowlPosition is None:
            return None

        # Find nearest time stamp from training data
        timeStamp = rospy.get_time() - self.init_time
        index = np.abs(self.times - timeStamp).argmin()

        # Use magnitude of forces
        force = np.linalg.norm(self.force).flatten()
        force = self.scaling(force, self.minVals[0], self.maxVals[0])

        # Determine distance between mic and bowl
        distance = np.linalg.norm(self.mic - self.bowlPosition)
        distance = self.scaling(distance, self.minVals[1], self.maxVals[1])
        # Find angle between gripper-bowl vector and gripper-spoon vector
        micSpoonVector = self.spoon - self.mic
        micBowlVector = self.bowlPosition - self.mic
        angle = np.arccos(np.dot(micSpoonVector, micBowlVector) / (np.linalg.norm(micSpoonVector) * np.linalg.norm(micBowlVector)))
        angle = self.scaling(angle, self.minVals[2], self.maxVals[2])

        self.transformer.waitForTransform(self.targetFrame, self.rgbCameraFrame, rospy.Time(0), rospy.Duration(5))
        try:
            targetTrans, targetRot = self.transformer.lookupTransform(self.targetFrame, self.rgbCameraFrame, rospy.Time(0))
            self.targetMatrix = np.dot(tf.transformations.translation_matrix(targetTrans), tf.transformations.quaternion_matrix(targetRot))
        except tf.ExtrapolationException:
            print 'TF Target Error!'
            pass

        pointSet = np.c_[self.points3D, np.ones(len(self.points3D))]
        pointSet = np.dot(self.targetMatrix, pointSet.T).T[:, :3]

        # Check for invalid points
        pointSet = pointSet[np.linalg.norm(pointSet, axis=1) < 5]

        # Find points within a sphere of radius 8 cm around the center of bowl
        nearbyPoints = np.linalg.norm(pointSet - self.bowlPosition, axis=1) < 0.08

        # Points near bowl
        points = pointSet[nearbyPoints]

        if len(points) <= 0:
            print 'ARGH, no points within 8 cm of bowl location found'

        pdfValue = 0
        # If no points found, try opening up to 10 cm
        if len(points) <= 0:
            # Find points within a sphere of radius 10 cm around the center of bowl
            nearbyPoints = np.linalg.norm(pointSet - self.bowlPosition, axis=1) < 0.10
            # Points near bowl
            points = pointSet[nearbyPoints]
            if len(points) <= 0:
                print 'No points within 10 cm of bowl location found'

        if len(points) > 0:
            # Try an exponential dropoff instead of Trivariate Gaussian Distribution
            pdfValue = np.sum(np.exp(np.linalg.norm(points - self.bowlPosition, axis=1) * -10.0))
            pdfValue = self.scaling(pdfValue, self.minVals[3], self.maxVals[3])

            # # Scale all points to prevent division by small numbers and singular matrices
            # newPoints = points * 20
            # # Define a receptive field within the bowl
            # mu = self.bowlPosition * 20
            #
            # # Trivariate Gaussian Distribution
            # n, m = newPoints.shape
            # sigma = np.zeros((m, m))
            # # Compute covariances
            # for h in xrange(m):
            #     for j in xrange(m):
            #         sigma[h, j] = 1.0/n * np.dot((newPoints[:, h] - mu[h]).T, newPoints[:, j] - mu[j])
            # constant = 1.0 / np.sqrt((2*np.pi)**m * np.linalg.det(sigma))
            # sigmaInv = np.linalg.inv(sigma)
            # # Evaluate the Probability Density Function for each point
            # for point in newPoints:
            #     pointMu = point - mu
            #     # scalar = np.exp(np.abs(np.linalg.norm(point - newBowlPosition))*-2.0)
            #     pdfValue += constant * np.exp(-1.0/2.0 * np.dot(np.dot(pointMu.T, sigmaInv), pointMu))

        if index >= len(self.forces):
            self.forces.append(force)
            self.distances.append(distance)
            self.angles.append(angle)
            self.pdfs.append(pdfValue)
        else:
            self.forces[index] = force
            self.distances[index] = force
            self.angles[index] = angle
            self.pdfs[index] = pdfValue

    def cloudCallback(self, data):
        # print 'Time between cloud calls:', time.time() - self.cloudTime
        # startTime = time.time()

        self.pointCloud = data

        self.transposeBowlToCamera()
        self.transposeGripperToCamera()

        lowX, highX, lowY, highY = self.boundingBox()

        points2D = [[x, y] for y in xrange(lowY, highY) for x in xrange(lowX, highX)]
        try:
            points3D = pc2.read_points(self.pointCloud, field_names=('x', 'y', 'z'), skip_nans=True, uvs=points2D)
        except:
            print 'Unable to unpack from PointCloud2!', self.cameraWidth, self.cameraHeight, self.pointCloud.width, self.pointCloud.height
            return

        self.points3D = np.array([point for point in points3D])

        # self.publishImageFeatures()

        # self.updateNumber += 1
        # print 'Cloud computation time:', time.time() - startTime
        # self.cloudTime = time.time()

    def forceCallback(self, msg):
        self.force = np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z])
        self.torque = np.array([msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z])
        self.updateNumber += 1

    def bowlPoseManualCallback(self, data):
        self.bowlPosition = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z])

    def transposeBowlToCamera(self):
        self.transformer.waitForTransform(self.rgbCameraFrame, '/torso_lift_link', rospy.Time(0), rospy.Duration(5))
        try :
            position, rotation = self.transformer.lookupTransform(self.rgbCameraFrame, '/torso_lift_link', rospy.Time(0))
            self.bowlToKinectMat = np.dot(tf.transformations.translation_matrix(position), tf.transformations.quaternion_matrix(rotation))
        except tf.ExtrapolationException:
            print 'Transpose of bowl failed!'
            return
        pos = self.bowlPosition
        self.bowlPositionKinect = np.dot(self.bowlToKinectMat, np.array([pos[0], pos[1], pos[2], 1.0]))[:3]
        self.bowlX, self.bowlY = self.pinholeCamera.project3dToPixel(self.bowlPositionKinect)

    def transposeGripperToCamera(self):
        # Transpose gripper position to camera frame
        self.transformer.waitForTransform(self.targetFrame, '/l_gripper_tool_frame', rospy.Time(0), rospy.Duration(5))
        try :
            self.lGripperPosition, self.lGripperRotation = self.transformer.lookupTransform(self.rgbCameraFrame, '/l_gripper_tool_frame', rospy.Time(0))
            transMatrix = np.dot(tf.transformations.translation_matrix(self.lGripperPosition), tf.transformations.quaternion_matrix(self.lGripperRotation))
        except tf.ExtrapolationException:
            print 'Transpose of gripper failed!'
            return

        if len(self.grips) >= 2:
            self.lGripperTransposeMatrix = self.grips[-2]
        else:
            self.lGripperTransposeMatrix = transMatrix
        self.grips.append(transMatrix)

        # Determine location of mic
        mic = [0.12, -0.02, 0]
        self.mic = np.dot(self.lGripperTransposeMatrix, np.array([mic[0], mic[1], mic[2], 1.0]))[:3]
        # Determine location of spoon
        spoon3D = [0.22, -0.050, 0]
        self.spoon = np.dot(self.lGripperTransposeMatrix, np.array([spoon3D[0], spoon3D[1], spoon3D[2], 1.0]))[:3]


    # Returns coordinates (lowX, highX, lowY, highY)
    def boundingBox(self):
        size = 150
        left = self.bowlX - size/2
        right = left + size
        top = self.bowlY - size/2
        bottom = top + size

        # Check if box extrudes past image bounds
        if left < 0:
            left = 0
            right = left + size
        if right > self.cameraWidth - 1:
            right = self.cameraWidth - 1
            left = right - size
        if top < 0:
            top = 0
            bottom = top + size
        if bottom > self.cameraHeight - 1:
            bottom = self.cameraHeight - 1
            top = bottom - size

        return int(left), int(right), int(top), int(bottom)

    def cameraRGBInfoCallback(self, data):
        if self.cameraWidth is None:
            self.cameraWidth = data.width
            self.cameraHeight = data.height
            self.pinholeCamera = image_geometry.PinholeCameraModel()
            self.pinholeCamera.fromCameraInfo(data)
            self.rgbCameraFrame = data.header.frame_id

    def publishImageFeatures(self):
        imageFeatures = ImageFeatures()

        if self.bowlX is not None:
            circle = Circle()
            circle.x, circle.y = int(self.bowlX), int(self.bowlY)
            circle.radius = 10
            circle.r = 255
            circle.g = 128
            imageFeatures.circles.append(circle)

        rect = Rectangle()
        rect.lowX, rect.highX, rect.lowY, rect.highY = self.boundingBox()
        rect.r = 75
        rect.g = 150
        rect.thickness = 5
        imageFeatures.rectangles.append(rect)

        self.publisher2D.publish(imageFeatures)


