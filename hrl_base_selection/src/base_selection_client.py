#!/usr/bin/env python
import sys, optparse

import rospy
import openravepy as op
import numpy as np
import math as m
from hrl_base_selection.srv import *
import roslib; roslib.load_manifest('hrl_haptic_mpc')
import hrl_lib.transforms as tr
import tf
import rospy
from helper_functions import createBMatrix

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist

def select_base_client():
    angle = m.pi
    pr2_B_head  =  np.matrix([[    m.cos(angle),     -m.sin(angle),                0.,              3.],
                              [    m.sin(angle),      m.cos(angle),                0.,              .3],
                              [              0.,                0.,                1.,             1.4],
                              [              0.,                0.,                0.,              1.]])
    pos_goal = [pr2_B_head[0,3],pr2_B_head[1,3],pr2_B_head[2,3]]
    ori_goal = tr.matrix_to_quaternion(pr2_B_head[0:3,0:3])
    psm_head = PoseStamped()
    psm_head.header.frame_id = '/base_link'
    psm_head.pose.position.x=pos_goal[0]
    psm_head.pose.position.y=pos_goal[1]
    psm_head.pose.position.z=pos_goal[2]
    psm_head.pose.orientation.x=ori_goal[0]
    psm_head.pose.orientation.y=ori_goal[1]
    psm_head.pose.orientation.z=ori_goal[2]
    psm_head.pose.orientation.w=ori_goal[3]

    goal_angle = 0#m.pi/2
    pr2_B_goal  =  np.matrix([[    m.cos(goal_angle),     -m.sin(goal_angle),                0.,              2.7],
                              [    m.sin(goal_angle),      m.cos(goal_angle),                0.,              .3],
                              [                   0.,                     0.,                1.,             1.],
                              [                   0.,                     0.,                0.,              1.]])
    pos_goal = [pr2_B_goal[0,3],pr2_B_goal[1,3],pr2_B_goal[2,3]]
    ori_goal = tr.matrix_to_quaternion(pr2_B_goal[0:3,0:3])
    psm_goal = PoseStamped()
    psm_goal.header.frame_id = '/base_link'
    psm_goal.pose.position.x=pos_goal[0]
    psm_goal.pose.position.y=pos_goal[1]
    psm_goal.pose.position.z=pos_goal[2]
    psm_goal.pose.orientation.x=ori_goal[0]
    psm_goal.pose.orientation.y=ori_goal[1]
    psm_goal.pose.orientation.z=ori_goal[2]
    psm_goal.pose.orientation.w=ori_goal[3]
    head_pub = rospy.Publisher("/haptic_mpc/head_pose", PoseStamped, latch=True)
    head_pub.publish(psm_head)
    goal_pub = rospy.Publisher("/haptic_mpc/goal_pose", PoseStamped, latch=True)
    goal_pub.publish(psm_goal)

    rospy.wait_for_service('select_base_position')
    try:
        select_base_position = rospy.ServiceProxy('select_base_position', BaseMove)
        response = select_base_position(psm_goal, psm_head)
        print 'response is: \n', response
        return response.base_goal#, response.ik_solution
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e

def usage():
    return "%s [current_loc goal head]"%sys.argv[0]



if __name__ == "__main__":
    rospy.init_node('client_node')
    #if len(sys.argv) == 3:
    #    current_loc = PoseStamped(sys.argv[0])
    #    goal = PoseStamped(sys.argv[1])
    #    head = PoseStamped(sys.argv[2])
    #else:
    #    print usage()
    #    sys.exit(1)
    print "Requesting Base Goal Position"
    goal = select_base_client()
    rospy.spin()
    print "Base Goal Position is:\n", goal
    #print "ik solution is: \n", ik
