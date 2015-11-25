#!/usr/bin/env python  

#Chris Birmingham, HRL, 7/17/14
#This file is for reading looking up tf transforms to determine goal poses


import roslib
roslib.load_manifest('hrl_feeding_task')
roslib.load_manifest('hrl_haptic_mpc')
import rospy
import math
import time
import numpy as np
import tf
import std_msgs.msg
import sys
import os
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from hrl_haptic_manipulation_in_clutter_srvs.srv import *
import geometry_msgs.msg
import tf_conversions.posemath as pm


#class tfListener(self)


if __name__ == '__main__':
    rospy.init_node('tf_listener')
    broadcaster = tf.TransformBroadcaster()
    listener = tf.TransformListener()
    while not rospy.is_shutdown():
        check = raw_input("Press enter to look up transform again")
        #broadcaster.sendTransform((0.1, -0.03, 0),(0, 0, 0, 1),
        #                    rospy.Time.now(),"/l_gripper_spoon_frame", "/l_gripper_shaver0_frame")
        try:
            (trans, rot) = listener.lookupTransform('/torso_lift_link', '/l_gripper_spoon_frame', rospy.Time(0))
            print trans, rot #Finds and prints the transform between the torso and the spoon
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print "tf lookup failed"
            continue
    rospy.spin()




       
