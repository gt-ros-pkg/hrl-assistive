#!/usr/bin/env python
import numpy as np
import roslib; roslib.load_manifest('hrl_haptic_mpc')
import rospy

import hrl_lib.transforms as tr
import openravepy as op

from geometry_msgs.msg import PoseStamped, Twist


def base_goal_publisher(goal):
    pub1 = rospy.Publisher('/base_controller/command', Twist)
    error_pos = [Bubase[3,0] - goal[0], Bubase[3,1] - goal[1],0]
    normalized_pos = error_pos / np.linalg.norm(error_pos)
    while (np.linalg.norm(error_pos)>0.1):
        tw = Twist()
        tw.linear.x=normalized[0]
        tw.linear.y=normalized[1]
        tw.linear.z=0
        tw.angular.x=0
        tw.angular.y=0
        tw.angular.z=0
        while not rospy.is_shutdown() and (np.linalg.norm(error_pos)>0.1):
            pub1.publish(tw)
            rospy.sleep(.5)
    error_ori = math.acos(Bubase[1,1])-goal[2]
    normalized_ori = error_ori / np.linalg.norm(error_ori)
    while (np.linalg.norm(error_ori)>0.1):
        tw = Twist()
        tw.linear.x=0
        tw.linear.y=0
        tw.linear.z=0
        tw.angular.x=0
        tw.angular.y=0
        tw.angular.z=normalized_ori
        while not rospy.is_shutdown() and (np.linalg.norm(error_ori)>0.1):
            pub1.publish(tw)
            rospy.sleep(.5)




def createBMatrix(pos, ori):
    goalB = np.zeros([4,4])
    goalB[3, 3] = 1

    goalB[0:3, 0:3] = np.array(tr.quaternion_to_matrix(ori))
    for i in xrange(0,3):
        goalB[i, 3] = pos[i]
	
    return np.matrix(goalB)
