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



class AutobedStatePublisherNode(object):
    def __init__(self):

        self.joint_pub = rospy.Publisher('human_model/joint_states', JointState, queue_size=100)
        self.frame_lock = threading.RLock()
        self.broadcaster = tf.TransformBroadcaster()
        rospy.sleep(1)
        print 'Human model state publisher node is ready and running!'

    def run(self):
        # rate = rospy.Rate(30) #30 Hz

        joint_state = JointState()
        joint_state.name = [None]*(42)
        joint_state.position = [None]*(42)

        rospy.sleep(2.)

        rate = rospy.Rate(10.0)
        while not rospy.is_shutdown():
            with self.frame_lock:
                joint_state.header.stamp = rospy.Time.now()
                joint_state.name[0] = "human_model/head_neck_x_joint"
                joint_state.name[1] = "human_model/head_neck_y_joint"
                joint_state.name[2] = "human_model/head_neck_z_joint"
                joint_state.name[3] = "human_model/neck_upper_body_top_joint"
                joint_state.name[4] = "human_model/upper_body_mid_body_x_joint"
                joint_state.name[5] = "human_model/upper_body_mid_body_y_joint"
                joint_state.name[6] = "human_model/upper_body_mid_body_z_joint"
                joint_state.name[7] = "human_model/mid_body_pelvis_x_joint"
                joint_state.name[8] = "human_model/mid_body_pelvis_y_joint"
                joint_state.name[9] = "human_model/mid_body_pelvis_z_joint"
                joint_state.name[10] = "human_model/hip_thigh_left_x_joint"
                joint_state.name[11] = "human_model/hip_thigh_left_y_joint"
                joint_state.name[12] = "human_model/hip_thigh_left_z_joint"
                joint_state.name[13] = "human_model/knee_calf_left_joint"
                joint_state.name[14] = "human_model/ankle_foot_left_x_joint"
                joint_state.name[15] = "human_model/ankle_foot_left_y_joint"
                joint_state.name[16] = "human_model/ankle_foot_left_z_joint"
                joint_state.name[17] = "human_model/hip_thigh_right_x_joint"
                joint_state.name[18] = "human_model/hip_thigh_right_y_joint"
                joint_state.name[19] = "human_model/hip_thigh_right_z_joint"
                joint_state.name[20] = "human_model/knee_calf_right_joint"
                joint_state.name[21] = "human_model/ankle_foot_right_x_joint"
                joint_state.name[22] = "human_model/ankle_foot_right_y_joint"
                joint_state.name[23] = "human_model/ankle_foot_right_z_joint"
                joint_state.name[24] = "human_model/upper_body_scapula_left_x_joint"
                joint_state.name[25] = "human_model/upper_body_scapula_left_z_joint"
                joint_state.name[26] = "human_model/shoulder_bicep_left_x_joint"
                joint_state.name[27] = "human_model/shoulder_bicep_left_y_joint"
                joint_state.name[28] = "human_model/shoulder_bicep_left_z_joint"
                joint_state.name[29] = "human_model/elbow_forearm_left_joint"
                joint_state.name[30] = "human_model/wrist_hand_left_x_joint"
                joint_state.name[31] = "human_model/wrist_hand_left_y_joint"
                joint_state.name[32] = "human_model/wrist_hand_left_z_joint"
                joint_state.name[33] = "human_model/upper_body_scapula_right_x_joint"
                joint_state.name[34] = "human_model/upper_body_scapula_right_z_joint"
                joint_state.name[35] = "human_model/shoulder_bicep_right_x_joint"
                joint_state.name[36] = "human_model/shoulder_bicep_right_y_joint"
                joint_state.name[37] = "human_model/shoulder_bicep_right_z_joint"
                joint_state.name[38] = "human_model/elbow_forearm_right_joint"
                joint_state.name[39] = "human_model/wrist_hand_right_x_joint"
                joint_state.name[40] = "human_model/wrist_hand_right_y_joint"
                joint_state.name[41] = "human_model/wrist_hand_right_z_joint"
                joint_state.position[0] = 0.
                joint_state.position[1] = 0.
                joint_state.position[2] = 0.
                joint_state.position[3] = 0.
                joint_state.position[4] = 0.
                joint_state.position[5] = 0.
                joint_state.position[6] = 0.
                joint_state.position[7] = 0.
                joint_state.position[8] = 0.
                joint_state.position[9] = 0.
                joint_state.position[10] = 0.
                joint_state.position[11] = 0.
                joint_state.position[12] = 0.
                joint_state.position[13] = 0.
                joint_state.position[14] = 0.
                joint_state.position[15] = 0.
                joint_state.position[16] = 0.
                joint_state.position[17] = -0.3
                joint_state.position[18] = -0.5
                joint_state.position[19] = -0.2
                joint_state.position[20] = 1.0
                joint_state.position[21] = 0.
                joint_state.position[22] = 0.8
                joint_state.position[23] = 0.
                joint_state.position[24] = 0.
                joint_state.position[25] = 0.
                joint_state.position[26] = 0.5
                joint_state.position[27] = 0.
                joint_state.position[28] = 0.
                joint_state.position[29] = -0.5
                joint_state.position[30] = 0.
                joint_state.position[31] = 0.
                joint_state.position[32] = 0.
                joint_state.position[33] = 0.
                joint_state.position[34] = 0.
                joint_state.position[35] = -0.5
                joint_state.position[36] = 0.
                joint_state.position[37] = 0.
                joint_state.position[38] = -1.
                joint_state.position[39] = 0.
                joint_state.position[40] = 0.
                joint_state.position[41] = 0.

                # self.broadcaster.sendTransform([0,0,0], [0,0,0,1],
                #                                rospy.Time.now(),
                #                                'base_link',
                #                                'human_model/pelvis_link')

                self.joint_pub.publish(joint_state)
                rate.sleep()
        return


if __name__ == "__main__":
    rospy.init_node('human_model_state_publisher_node', anonymous = False)
    a = AutobedStatePublisherNode()
    a.run()
