#!/usr/bin/env python

import roslib
roslib.load_manifest('hrl_base_selection')
import rospy
import tf
import numpy as np
import math as m
from sensor_msgs.msg import JointState
from helper_functions import createBMatrix, Bmat_to_pos_quat
from geometry_msgs.msg import PoseStamped


class GlobalTFBroadcaster(object):
    def __init__(self):
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.tf_listener = tf.TransformListener()
        rospy.sleep(2)
        rate = rospy.Rate(50.0)
        self.map_B_bed = None
        self.out_trans = None
        self.out_rot = None
        B_mat = np.eye(4)
        B_mat[3, 0] = 2.0
        self.out_trans, self.out_rot = Bmat_to_pos_quat(B_mat)
        joint_pub = rospy.Publisher('autobed/joint_states', JointState, queue_size=1)
        # self.ar_tag_autobed_sub = rospy.Subscriber('/ar_tag_tracking/autobed_pose', PoseStamped, self.ar_tag_autobed_cb)
        while not rospy.is_shutdown():
            self.tf_broadcaster.sendTransform(self.out_trans, self.out_rot,
                                              rospy.Time.now(),
                                              'autobed/base_link',
                                              'base_link')
            # print 'published transform'
            # print rospy.Time.now()

            autobed_joint_state = JointState()
            autobed_joint_state.header.stamp = rospy.Time.now()

            autobed_joint_state.name = [None]*(23)
            autobed_joint_state.position = [None]*(23)
            autobed_joint_state.name[0] = "autobed/head_bed_updown_joint"
            autobed_joint_state.name[1] = "autobed/head_bed_leftright_joint"
            autobed_joint_state.name[2] = "autobed/head_rest_hinge"
            autobed_joint_state.name[3] = "autobed/neck_body_joint"
            autobed_joint_state.name[4] = "autobed/upper_mid_body_joint"
            autobed_joint_state.name[5] = "autobed/mid_lower_body_joint"
            autobed_joint_state.name[6] = "autobed/body_quad_left_joint"
            autobed_joint_state.name[7] = "autobed/body_quad_right_joint"
            autobed_joint_state.name[8] = "autobed/quad_calf_left_joint"
            autobed_joint_state.name[9] = "autobed/quad_calf_right_joint"
            autobed_joint_state.name[10] = "autobed/calf_foot_left_joint"
            autobed_joint_state.name[11] = "autobed/calf_foot_right_joint"
            autobed_joint_state.name[12] = "autobed/body_arm_left_joint"
            autobed_joint_state.name[13] = "autobed/body_arm_right_joint"
            autobed_joint_state.name[14] = "autobed/arm_forearm_left_joint"
            autobed_joint_state.name[15] = "autobed/arm_forearm_right_joint"
            autobed_joint_state.name[16] = "autobed/forearm_hand_left_joint"
            autobed_joint_state.name[17] = "autobed/forearm_hand_right_joint"
            autobed_joint_state.name[18] = "autobed/tele_legs_joint"
            autobed_joint_state.name[19] = "autobed/head_neck_joint1"
            autobed_joint_state.name[20] = "autobed/head_neck_joint2"
            autobed_joint_state.name[21] = "autobed/leg_rest_upper_joint"
            autobed_joint_state.name[22] = "autobed/leg_rest_upper_lower_joint"


            autobed_joint_state.position[0] = 15
            autobed_joint_state.position[1] = 0

            # bth = m.degrees(30)
            bth = 30
            # 0 degrees, 0 height
            if (bth >= 0) and (bth <= 40):  # between 0 and 40 degrees
                autobed_joint_state.position[2] = (bth/40)*(0.6981317 - 0)+0
                autobed_joint_state.position[3] = (bth/40)*(-.2-(-.1))+(-.1)
                autobed_joint_state.position[4] = (bth/40)*(-.17-.4)+.4
                autobed_joint_state.position[5] = (bth/40)*(-.76-(-.72))+(-.72)
                autobed_joint_state.position[6] = -0.4
                autobed_joint_state.position[7] = -0.4
                autobed_joint_state.position[8] = 0.1
                autobed_joint_state.position[9] = 0.1
                autobed_joint_state.position[10] = (bth/40)*(-.05-.02)+.02
                autobed_joint_state.position[11] = (bth/40)*(-.05-.02)+.02
                autobed_joint_state.position[12] = (bth/40)*(-.06-(-.12))+(-.12)
                autobed_joint_state.position[13] = (bth/40)*(-.06-(-.12))+(-.12)
                autobed_joint_state.position[14] = (bth/40)*(.58-0.05)+.05
                autobed_joint_state.position[15] = (bth/40)*(.58-0.05)+.05
                autobed_joint_state.position[16] = -0.1
                autobed_joint_state.position[17] = -0.1
                autobed_joint_state.position[18] = 0.2
                autobed_joint_state.position[19] = 0.
                autobed_joint_state.position[20] = 0.
                autobed_joint_state.position[21] = 0.
                autobed_joint_state.position[22] = 0.
            joint_pub.publish(autobed_joint_state)

            rate.sleep()



    # def ar_tag_autobed_cb(self, msg):
    #     trans = [msg.pose.position.x,
    #              msg.pose.position.y,
    #              msg.pose.position.z]
    #     rot = [msg.pose.orientation.x,
    #            msg.pose.orientation.y,
    #            msg.pose.orientation.z,
    #            msg.pose.orientation.w]
    #     now = rospy.Time.now()
    #     pr2_B_bed = createBMatrix(trans, rot)
    #     self.listener.waitForTransform('map', 'torso_lift_link', now, rospy.Duration(1))
    #     (trans, rot) = self.tf_listener.lookupTransform('map', 'torso_lift_link', now)
    #     map_B_pr2 = createBMatrix(trans, rot)
    #     self.map_B_bed = map_B_pr2*pr2_B_bed
    #     (self.out_trans, self.out_rot) = Bmat_to_pos_quat(self.map_B_bed)


if __name__ == '__main__':
    rospy.init_node('global_tf_broadcaster')
    global_tf_broadcaster = GlobalTFBroadcaster()
    # rospy.spin()








