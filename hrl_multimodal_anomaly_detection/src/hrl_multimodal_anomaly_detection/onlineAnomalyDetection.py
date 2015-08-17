#!/usr/bin/env python

__author__ = 'zerickson'

import math
import time
import rospy
import struct
import pyaudio
import numpy as np
from threading import Thread
import matplotlib.pyplot as plt
import onlineHMMLauncher as onlineHMM

try :
    import sensor_msgs.point_cloud2 as pc2
except:
    import vision.point_cloud2 as pc2
from geometry_msgs.msg import PoseStamped, WrenchStamped, Point
from std_msgs.msg import String
from visualization_msgs.msg import Marker
from roslib import message

import roslib
roslib.load_manifest('hrl_multimodal_anomaly_detection')
import tf
import image_geometry
from cv_bridge import CvBridge, CvBridgeError
from hrl_multimodal_anomaly_detection.msg import Circle, Rectangle, ImageFeatures

class onlineAnomalyDetection(Thread):
    MAX_INT = 32768.0
    CHUNK   = 1024 # frame per buffer
    RATE    = 44100 # sampling rate
    UNIT_SAMPLE_TIME = 1.0 / float(RATE)
    CHANNEL = 2 # number of channels
    FORMAT  = pyaudio.paInt16

    def __init__(self, targetFrame=None, tfListener=None, isScooping=True, useAudio=False):
        super(onlineAnomalyDetection, self).__init__()
        self.daemon = True
        self.cancelled = False

        self.isScooping = isScooping

        print 'is scooping:', self.isScooping

        self.publisher = rospy.Publisher('visualization_marker', Marker)
        self.interruptPublisher = rospy.Publisher('InterruptAction', String)
        self.targetFrame = targetFrame

        # Data logging
        self.updateNumber = 0
        self.lastUpdateNumber = 0
        self.init_time = rospy.get_time()

        if tfListener is None:
            self.transformer = tf.TransformListener()
        else:
            self.transformer = tfListener

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

        ## self.soundHandle = SoundClient()

        # Setup HMM to perform online anomaly detection
        self.hmm, self.minVals, self.maxVals, self.forces, self.distances, self.angles, self.audios, self.times, self.forcesList, self.distancesList, self.anglesList, self.audioList, self.timesList = onlineHMM.setupMultiHMM(isScooping=self.isScooping)
        # if not self.isScooping:
        #     self.forces, self.distances, self.angles, self.audios = self.forcesList[1], self.distancesList[1], self.anglesList[1], self.audioList[1]
        self.times = np.array(self.times)
        self.anomalyOccured = False

        self.p = pyaudio.PyAudio()
        deviceIndex = self.find_input_device()
        print 'Audio device:', deviceIndex
        self.stream = self.p.open(format=self.FORMAT, channels=self.CHANNEL, rate=self.RATE, input=True, frames_per_buffer=self.CHUNK, input_device_index=deviceIndex)

        self.forceSub = rospy.Subscriber('/netft_data', WrenchStamped, self.forceCallback)
        print 'Connected to FT sensor'

        self.objectCenter = None
        self.objectCenterSub = rospy.Subscriber('/ar_track_alvar/bowl_cen_pose' if isScooping else '/ar_track_alvar/mouth_pose', PoseStamped, self.objectCenterCallback)
        print 'Connected to center of object publisher'

    def reset(self):
        pass

    def run(self):
        """Overloaded Thread.run, runs the update
        method once per every xx milliseconds."""
        # rate = rospy.Rate(1000) # 25Hz, nominally.
        while not self.cancelled:
            if self.updateNumber > self.lastUpdateNumber and self.objectCenter is not None:
                self.lastUpdateNumber = self.updateNumber
                self.processData()
                if not self.anomalyOccured:
                    # Best c value gains determined from cross validation set
                    c = -1 if self.isScooping else -9
                    # Perform anomaly detection
                    (anomaly, error) = self.hmm.anomaly_check(self.forces, self.distances, self.angles, self.audios, c)
                    print 'Anomaly error:', error
                    if anomaly > 0:
                        if self.isScooping:
                            self.interruptPublisher.publish('Interrupt')
                        else:
                            self.interruptPublisher.publish('InterruptHead')
                        self.anomalyOccured = True
                        print 'AHH!! There is an anomaly at time stamp', rospy.get_time() - self.init_time, (anomaly, error)
                        # for modality in [[self.forces] + self.forcesList[:5], [self.distances] + self.distancesList[:5], [self.angles] + self.anglesList[:5], [self.pdfs] + self.pdfList[:5]]:
                        #     for index, (modal, times) in enumerate(zip(modality, [self.times] + self.timesList[:5])):
                        #         plt.plot(times, modal, label='%d' % index)
                        #     plt.legend()
                        #     plt.show()
            # rate.sleep()

    def cancel(self):
        """End this timer thread"""
        self.cancelled = True
        self.forceSub.unregister()
        self.objectCenterSub.unregister()
        self.publisher.unregister()
        rospy.sleep(1.0)

    @staticmethod
    def scaling(x, minVal, maxVal, scale=1.0):
        return (x - minVal) / (maxVal - minVal) * scale

    def processData(self):
        # Find nearest time stamp from training data
        timeStamp = rospy.get_time() - self.init_time
        index = np.abs(self.times - timeStamp).argmin()

        self.transposeGripper()

        # Use magnitude of forces
        force = np.linalg.norm(self.force)
        force = self.scaling(force, self.minVals[0], self.maxVals[0])

        # Determine distance between mic and center of object
        distance = np.linalg.norm(self.mic - self.objectCenter)
        distance = self.scaling(distance, self.minVals[1], self.maxVals[1])
        # Find angle between gripper-object vector and gripper-spoon vector
        micSpoonVector = self.spoon - self.mic
        micObjectVector = self.objectCenter - self.mic
        angle = np.arccos(np.dot(micSpoonVector, micObjectVector) / (np.linalg.norm(micSpoonVector) * np.linalg.norm(micObjectVector)))
        angle = self.scaling(angle, self.minVals[2], self.maxVals[2])

        # Process either visual or audio data depending on which we're using
        audio = self.processAudio()

        # Magnify relative errors for online detection
        # Since we are overwriting a sample from the training set, we need to magnify errors
        # for them to be recognized as an anomaly
        if index < len(self.forces):
            scalar = 1.0 if self.isScooping else 3.0
            forceDiff = np.abs(self.forces[index] - force)
            if forceDiff > 0.5:
                if force < self.forces[index]:
                    force -= scalar*forceDiff
                else:
                    force += scalar*forceDiff
            distanceDiff = np.abs(self.distances[index] - distance)
            if distanceDiff > 0.5:
                if distance < self.distances[index]:
                    distance -= scalar*distanceDiff
                else:
                    distance += scalar*distanceDiff
            angleDiff = np.abs(self.angles[index] - angle)
            if angleDiff > 0.5:
                if angle < self.angles[index]:
                    angle -= scalar*angleDiff
                else:
                    angle += scalar*angleDiff
            audioDiff = np.abs(self.audios[index] - audio)
            if audioDiff > 0.5:
                if audio < self.audios[index]:
                    audio -= scalar*audioDiff
                else:
                    audio += scalar*audioDiff

        if index >= len(self.forces):
            self.forces.append(force)
            self.distances.append(distance)
            self.angles.append(angle)
            self.audios.append(audio)
        else:
            print 'Current force:', self.forces[index], 'New force:', force
            self.forces[index] = force
            print 'Current distance:', self.distances[index], 'New distance:', distance
            self.distances[index] = distance
            print 'Current angle:', self.angles[index], 'New angle:', angle
            self.angles[index] = angle
            print 'Current audio:', self.audios[index], 'New audio:', audio
            self.audios[index] = audio

    def processAudio(self):
        data = self.stream.read(self.CHUNK)
        audio = self.get_rms(data)
        audio = self.scaling(audio, self.minVals[3], self.maxVals[3])
        return audio

    def forceCallback(self, msg):
        self.force = np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z])
        self.torque = np.array([msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z])
        self.updateNumber += 1

    def objectCenterCallback(self, msg):
        self.objectCenter = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])

    def transposeGripper(self):
        # Transpose gripper position to camera frame
        self.transformer.waitForTransform(self.targetFrame, '/l_gripper_tool_frame', rospy.Time(0), rospy.Duration(5))
        try :
            self.lGripperPosition, self.lGripperRotation = self.transformer.lookupTransform(self.targetFrame, '/l_gripper_tool_frame', rospy.Time(0))
            transMatrix = np.dot(tf.transformations.translation_matrix(self.lGripperPosition), tf.transformations.quaternion_matrix(self.lGripperRotation))
        except tf.ExtrapolationException:
            print 'Transpose of gripper failed!'
            return

        # Use a buffer of gripper positions
        if len(self.grips) >= 2:
            self.lGripperTransposeMatrix = self.grips[-2]
        else:
            self.lGripperTransposeMatrix = transMatrix
        self.grips.append(transMatrix)

        # Determine location of mic
        mic = [0.12, -0.02, 0]
        # print 'Mic before', mic
        self.mic = np.dot(self.lGripperTransposeMatrix, np.array([mic[0], mic[1], mic[2], 1.0]))[:3]
        # print 'Mic after', self.mic
        # Determine location of spoon
        spoon3D = [0.22, -0.050, 0]
        self.spoon = np.dot(self.lGripperTransposeMatrix, np.array([spoon3D[0], spoon3D[1], spoon3D[2], 1.0]))[:3]

    def get_rms(self, block):
        # RMS amplitude is defined as the square root of the
        # mean over time of the square of the amplitude.
        # so we need to convert this string of bytes into
        # a string of 16-bit samples...

        # we will get one short out for each
        # two chars in the string.
        count = len(block)/2
        structFormat = '%dh' % count
        shorts = struct.unpack(structFormat, block)

        # iterate over the block.
        sum_squares = 0.0
        for sample in shorts:
            # sample is a signed short in +/- 32768.
            # normalize it to 1.0
            n = sample / self.MAX_INT
            sum_squares += n*n

        return math.sqrt(sum_squares / count)

    def publishPoints(self, name, points, size=0.01, r=0.0, g=0.0, b=0.0, a=1.0):
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
        self.publisher.publish(marker)


