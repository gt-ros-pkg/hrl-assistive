#!/usr/bin/env python

from threading import Lock
import copy
import numpy as np

import roslib
roslib.load_manifest('hrl_face_adls')
import rospy
from hrl_msgs.msg import FloatArrayBare
from std_msgs.msg import String, Int32, Int8, Bool
from geometry_msgs.msg import PoseStamped, Point, Quaternion, TransformStamped
import tf
from tf import TransformListener
from tf import transformations as tft
roslib.load_manifest('hrl_base_selection')
from helper_functions import createBMatrix, Bmat_to_pos_quat
roslib.load_manifest('hrl_pr2_ar_servo')
from ar_pose.msg import ARMarker


class TF_Goal(object):
    # Object for spoofing tf frames and ar tags to allow use of servoing without AR tags. For manual base positioning

    def __init__(self, goal_pose, tf_listener):
        # self.tf_listener = tf_listener
        self.goal_pose = goal_pose
        pos, ori = Bmat_to_pos_quat(goal_pose)
        self.tf_broadcaster = tf.TransformBroadcaster()
        print 'The goal_pose tf publisher is working, as far as I can tell!'
        while not rospy.is_shutdown():
            self.tf_broadcaster.sendTransform((pos[0], pos[1], 0.), (ori[0], ori[1], ori[2], ori[3]),
                                              rospy.Time.now(), '/base_goal_pose', '/optitrak')
            rospy.sleep(.2)

if __name__ == '__main__':
    rospy.init_node('tf_goal')
    # myrobot = '/base_location'
    # mytarget = '/goal_location'
    tf_goal = TF_Goal()
    rospy.spin()

