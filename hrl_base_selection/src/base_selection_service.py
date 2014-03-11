#!/usr/bin/env python
import sys, optparse

import rospy
import openravepy as op
import numpy as np
import math as m
import roslib
roslib.load_manifest('hrl_base_selection')
roslib.load_manifest('hrl_haptic_mpc')
import hrl_lib.transforms as tr
from hrl_base_selection.srv import *
import openravepy as op
from helper_functions import createBMatrix
from geometry_msgs.msg import PoseStamped





def handle_select_base(req):
    pos_temp = [req.head.pose.position.x,req.head.pose.position.y,req.head.pose.position.z]
    ori_temp = [req.head.pose.orientation.x,req.head.pose.orientation.y,req.head.pose.orientation.z,req.head.pose.orientation.w]
    head = createBMatrix(pos_temp,ori_temp)
    
    pos_temp = [req.goal.pose.position.x,req.goal.pose.position.y,req.goal.pose.position.z]
    ori_temp = [req.goal.pose.orientation.x,req.goal.pose.orientation.y,req.goal.pose.orientation.z,req.goal.pose.orientation.w]
    goal = createBMatrix(pos_temp,ori_temp)

    print 'I will move to be able to reach the mouth.'
    env = op.Environment()
    env.Load('robots/pr2-beta-static.zae')
    robot = env.GetRobots()[0]
    v = robot.GetActiveDOFValues()
    v[robot.GetJoint('l_shoulder_pan_joint').GetDOFIndex()]= 3.14/2
    v[robot.GetJoint('r_shoulder_pan_joint').GetDOFIndex()] = -3.14/2
    v[robot.GetJoint('l_gripper_l_finger_joint').GetDOFIndex()] = .54
    v[robot.GetJoint('torso_lift_joint').GetDOFIndex()] = .3
    robot.SetActiveDOFValues(v)
    env.Load('../models/ADA_Wheelchair.dae')
    wheelchair = env.GetBodies()[1]
    wc_angle =  m.pi
    pr2_B_wc =   np.matrix([[    head[0,0],     head[0,1],               0,    head[0,3]],
                            [    head[1,0],     head[1,1],               0,    head[1,3]],
                            [            0,             0,                1,           0],
                            [            0,             0,                0,           1]])
    
    corner_B_head = np.matrix([[m.cos(0),-m.sin(0),0,.3],[m.sin(0),m.cos(0),0,.385],[0,0,1,0],[0,0,0,1]])
    wheelchair_location = pr2_B_wc * corner_B_head.I
    wheelchair.SetTransform(array(wheelchair_location))
    for i in [0,.1,.2,-.1,-.2]:
        for j in [0,.1,.2,-.1,-.2]:
            for k in [0,m.pi/8,m.pi/4,-m.pi/8,-m.pi/4]:
                #goal_pose = req.goal
                angle = m.pi
                head_B_goal =  np.matrix([[    m.cos(angle),     -m.sin(angle),                0,              .1],
                                          [    m.sin(angle),      m.cos(angle),                0,               0],
                                          [               0,                 0,                1,               0],
                                          [               0,                 0,                0,               1]])
                pr2_B_goal = head * head_B_goal
                angle_base = m.pi
                wc_B_goalpr2  =   np.matrix([[    m.cos(angle_base),     -m.sin(angle_base),                0,             .5+i],
                                             [    m.sin(angle_base),      m.cos(angle_base),                0,             .5+j],
                                             [                    0,                      0,                1,                0],
                                             [                    0,                      0,                0,                1]])

                base_position = pr2_B_wc * wc_B_goalpr2
                robot.SetTransform(base_position)

                manip = robot.SetActiveManipulator('leftarm_torso')
                ikmodel = op.databases.inversekinematics.InverseKinematicsModel(self.robot,iktype=op.IkParameterization.Type.Transform6D)

                if not ikmodel.load():
                    ikmodel.autogenerate()
                with env:
                    sol = manip.FindIKSolution(pr2_B_goal, op.IkFilterOptions.CheckEnvCollisions)
                    if sol != None:
                        (trans,rot) = listener.lookupTransform('/odom_combined', '/base_link', rospy.Time(0))
                        odom_goal = createBMatrix(trans,rot)*base_position
	                pos_goal = [odom_goal[0,3],odom_goal[1,3],odom_goal[2,3]]
	                ori_goal = tr.matrix_to_quaternion(odom_goal[0:3,0:3])
	                psm = PoseStamped()
                        
	                psm.header.frame_id = '/odom_combined'
	                psm.pose.position.x=pos_goal[0]
	                psm.pose.position.y=pos_goal[1]
	                psm.pose.position.z=pos_goal[2]
	                psm.pose.orientation.x=ori_goal[0]
	                psm.pose.orientation.y=ori_goal[1]
	                psm.pose.orientation.z=ori_goal[2]
	                psm.pose.orientation.w=ori_goal[3]
                        print 'I found a goal location! It is at B transform: \n',base_position
                        print 'The quaternion to the goal location is: \n',psm
                        return psm
                    else:
                        print 'I found a bad goal location. Trying again!'
      
    print 'I found nothing! My given inputs were: \n', req.goal, req.head
    return None

def select_base_server():
    listener = tf.TransformListener()
    rospy.init_node('select_base_server')
    s = rospy.Service('select_base_position', BaseMove, handle_select_base)
    print "Ready to select base."
    rospy.spin()

if __name__ == "__main__":
    select_base_server()
