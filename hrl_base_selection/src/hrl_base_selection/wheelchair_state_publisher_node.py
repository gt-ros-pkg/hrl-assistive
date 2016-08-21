#!/usr/bin/env python
import numpy as np
import roslib; roslib.load_manifest('hrl_msgs'); roslib.load_manifest('tf')
import rospy
import tf
from hrl_msgs.msg import FloatArrayBare
from sensor_msgs.msg import JointState
from math import *
import math as m
import operator
import threading
from scipy.signal import remez
from scipy.signal import lfilter
from hrl_srvs.srv import None_Bool, None_BoolResponse
from std_msgs.msg import Bool

from visualization_msgs.msg import Marker
from geometry_msgs.msg import TransformStamped, Point, Pose, PoseStamped 
from std_msgs.msg import ColorRGBA
from autobed_occupied_client import autobed_occupied_status_client
from tf.transformations import quaternion_from_euler


MAT_WIDTH = 0.762 #metres
MAT_HEIGHT = 1.854 #metres
MAT_HALF_WIDTH = MAT_WIDTH/2 
NUMOFTAXELS_X = 64#73 #taxels
NUMOFTAXELS_Y = 27#30 
LOW_TAXEL_THRESH_X = 0
LOW_TAXEL_THRESH_Y = 0
HIGH_TAXEL_THRESH_X = (NUMOFTAXELS_X - 1) 
HIGH_TAXEL_THRESH_Y = (NUMOFTAXELS_Y - 1) 


class AutobedStatePublisherNode(object):
    def __init__(self):
        self.joint_pub = rospy.Publisher('wheelchair/joint_states', JointState, queue_size=100)

        # self.marker_pub=rospy.Publisher('visualization_marker', Marker)
        self.frame_lock = threading.RLock()
        print 'Autobed robot state publisher is ready and running!'


    def run(self):
        # rate = rospy.Rate(30) #30 Hz

        rate = rospy.Rate(20.0)
        while not rospy.is_shutdown():
            with self.frame_lock:
                self.set_wheelchair_user_configuration()
                rate.sleep()
        return

    def set_wheelchair_user_configuration(self):
        with self.frame_lock:
            human_joint_state = JointState()
            human_joint_state.header.stamp = rospy.Time.now()

            human_joint_state.name = [None]*(17)
            human_joint_state.position = [None]*(17)
            human_joint_state.name[0] = "wheelchair/neck_body_joint"
            human_joint_state.name[1] = "wheelchair/upper_mid_body_joint"
            human_joint_state.name[2] = "wheelchair/mid_lower_body_joint"
            human_joint_state.name[3] = "wheelchair/body_quad_left_joint"
            human_joint_state.name[4] = "wheelchair/body_quad_right_joint"
            human_joint_state.name[5] = "wheelchair/quad_calf_left_joint"
            human_joint_state.name[6] = "wheelchair/quad_calf_right_joint"
            human_joint_state.name[7] = "wheelchair/calf_foot_left_joint"
            human_joint_state.name[8] = "wheelchair/calf_foot_right_joint"
            human_joint_state.name[9] = "wheelchair/body_arm_left_joint"
            human_joint_state.name[10] = "wheelchair/body_arm_right_joint"
            human_joint_state.name[11] = "wheelchair/arm_forearm_left_joint"
            human_joint_state.name[12] = "wheelchair/arm_forearm_right_joint"
            human_joint_state.name[13] = "wheelchair/forearm_hand_left_joint"
            human_joint_state.name[14] = "wheelchair/forearm_hand_right_joint"
            human_joint_state.name[15] = "wheelchair/head_neck_joint1"
            human_joint_state.name[16] = "wheelchair/head_neck_joint2"

            human_joint_state.position[0] = 0.0
            human_joint_state.position[1] = -0.15
            human_joint_state.position[2] = -0.72
            human_joint_state.position[3] = 0.72
            human_joint_state.position[4] = 0.72
            human_joint_state.position[5] = 1.1
            human_joint_state.position[6] = 1.1
            human_joint_state.position[7] = 0.5
            human_joint_state.position[8] = 0.5
            human_joint_state.position[9] = 0.8
            human_joint_state.position[10] = 0.8
            human_joint_state.position[11] = 0.9
            human_joint_state.position[12] = 0.9
            human_joint_state.position[13] = -0.2
            human_joint_state.position[14] = -0.2
            human_joint_state.position[15] = 0.17
            human_joint_state.position[16] = 0.0


            self.joint_pub.publish(human_joint_state)


if __name__ == "__main__":
    rospy.init_node('autobed_state_publisher_node', anonymous = False)
    a = AutobedStatePublisherNode()
    a.run()
