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

from tf.transformations import quaternion_from_euler


class WheelchairStatePublisherNode(object):
    def __init__(self):
        self.listener = tf.TransformListener()
        rospy.sleep(2)
        self.joint_pub = rospy.Publisher('wheelchair/joint_states', JointState, queue_size=100)

        # self.marker_pub=rospy.Publisher('visualization_marker', Marker)
        self.frame_lock = threading.RLock()

        self.run()
        print 'Wheelchair robot state publisher is ready and running!'


    def run(self):
        # rate = rospy.Rate(30) #30 Hz

        rate = rospy.Rate(5.0)
        while not rospy.is_shutdown():
            with self.frame_lock:
                self.set_wheelchair_user_configuration()
                rate.sleep()
        return

    def set_wheelchair_user_configuration(self):
        with self.frame_lock:
            human_joint_state = JointState()
            human_joint_state.header.stamp = rospy.Time.now()

            human_joint_state.name = [None]*(20)
            human_joint_state.position = [None]*(20)
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
            human_joint_state.name[15] = "wheelchair/neck_tilt_joint"
            human_joint_state.name[16] = "wheelchair/neck_head_roty_joint"
            human_joint_state.name[17] = "wheelchair/neck_head_rotx_joint"
            human_joint_state.name[18] = "wheelchair/neck_twist_joint"
            human_joint_state.name[19] = "wheelchair/neck_head_rotz_joint"

            human_joint_state.position[0] = -0.15
            human_joint_state.position[1] = 0.4
            human_joint_state.position[2] = 0.4
            human_joint_state.position[3] = 0.5
            human_joint_state.position[4] = 0,5
            human_joint_state.position[5] = 1.3
            human_joint_state.position[6] = 1.3
            human_joint_state.position[7] = 0.2
            human_joint_state.position[8] = 0.2
            human_joint_state.position[9] = 0.6
            human_joint_state.position[10] = 0.6
            human_joint_state.position[11] = 0.8
            human_joint_state.position[12] = 0.8
            human_joint_state.position[13] = 0.
            human_joint_state.position[14] = 0.
            human_joint_state.position[15] = 0.75
            human_joint_state.position[16] = -0.45
            human_joint_state.position[17] = 0.
            if self.listener.canTransform('/wheelchair/base_link', '/base_link', rospy.Time(0)):
                (trans, rot) = self.listener.lookupTransform('/wheelchair/base_link', '/base_link', rospy.Time(0))
                human_joint_state.position[18] = m.copysign(m.radians(60.), trans[1])
                human_joint_state.position[19] = 0.
            else:
                human_joint_state.position[18] = 0.
                human_joint_state.position[19] = 0.

            self.joint_pub.publish(human_joint_state)


if __name__ == "__main__":
    rospy.init_node('wheelchair_state_publisher_node', anonymous = False)
    a = WheelchairStatePublisherNode()
    #a.run()
