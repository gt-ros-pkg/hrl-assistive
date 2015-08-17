#!/usr/bin/env python

import rospy
import numpy as np
from threading import Thread
from geometry_msgs.msg import PoseStamped

import roslib
roslib.load_manifest('hrl_multimodal_anomaly_detection')
import tf

class tool_kinematics(Thread):
    def __init__(self, tfListener, targetFrame='/torso_lift_link', isScooping=False):
        super(tool_kinematics, self).__init__()
        self.daemon = True
        self.cancelled = False

        self.init_time = rospy.get_time()
        self.transformer = tfListener
        self.targetFrame = targetFrame
        self.mic = None
        self.spoon = None
        self.grips = []

        # self.updated = False
        self.time_data = []
        self.kinematics_data = []

        self.objectCenter = None
        self.objectCenterSub = rospy.Subscriber('/ar_track_alvar/bowl_cen_pose' if isScooping else '/ar_track_alvar/mouth_pose', PoseStamped, self.objectCenterCallback)

    def objectCenterCallback(self, msg):
        self.objectCenter = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        # self.updated = True

    def reset(self):
        pass

    def run(self):
        """Overloaded Thread.run, runs the update
        method once per every xx milliseconds."""

        rate = rospy.Rate(1000) # 25Hz, nominally.
        while not self.cancelled:
            self.log()
            rate.sleep()

    def log(self):
        # if not self.updated:
        #     return
        self.time_data.append(rospy.get_time() - self.init_time)
        self.transposeGripper()
        self.kinematics_data.append([self.mic, self.spoon, self.objectCenter])

    def transposeGripper(self):
        # Transpose gripper position to camera frame
        self.transformer.waitForTransform(self.targetFrame, '/l_gripper_tool_frame', rospy.Time(0), rospy.Duration(5))
        try :
            lGripperPosition, lGripperRotation = self.transformer.lookupTransform(self.targetFrame, '/l_gripper_tool_frame', rospy.Time(0))
            transMatrix = np.dot(tf.transformations.translation_matrix(lGripperPosition), tf.transformations.quaternion_matrix(lGripperRotation))
        except tf.ExtrapolationException:
            print 'Transpose of gripper failed!'
            return

        # Use a buffer of gripper positions
        if len(self.grips) >= 2:
            lGripperTransposeMatrix = self.grips[-2]
        else:
            lGripperTransposeMatrix = transMatrix
        self.grips.append(transMatrix)

        # Determine location of mic
        mic = [0.12, -0.02, 0]
        # print 'Mic before', mic
        self.mic = np.dot(lGripperTransposeMatrix, np.array([mic[0], mic[1], mic[2], 1.0]))[:3]
        # print 'Mic after', self.mic
        # Determine location of spoon
        spoon3D = [0.22, -0.050, 0]
        self.spoon = np.dot(lGripperTransposeMatrix, np.array([spoon3D[0], spoon3D[1], spoon3D[2], 1.0]))[:3]

    def cancel(self):
        """End this timer thread"""
        self.cancelled = True
        self.objectCenterSub.unregister()
        rospy.sleep(1.0)
