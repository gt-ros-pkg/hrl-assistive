#!/usr/bin/env python

import openravepy as op
import numpy

import os

#from openravepy.misc import InitOpenRAVELogging 
#InitOpenRAVELogging() 

# from openravepy import *
import openravepy as op
import numpy, time
import rospkg
import math as m
import numpy as np
import rospy

rospy.init_node('test_node')

env = op.Environment()  # create openrave environment
env.SetViewer('qtcoin')  # attach viewer (optional)

env.Load('robots/pr2-beta-static.zae')
robot = env.GetRobots()[0]

rospack = rospkg.RosPack()
pkg_path = rospack.get_path('hrl_base_selection')

human_urdf_path = pkg_path + '/urdf/green_kevin/robots/green_kevin.urdf'
human_srdf_path = pkg_path + '/urdf/green_kevin/robots/green_kevin.srdf'
module = op.RaveCreateModule(env, 'urdf')
name = module.SendCommand('LoadURI '+human_urdf_path+' '+human_srdf_path)
human_model = env.GetRobots()[1]

rotx_correction = np.matrix([[1., 0., 0., 0.],
                             [0., 0., -1., 0.],
                             [0., 1., 0., 0.],
                             [0., 0., 0., 1.]])

roty_correction = np.matrix([[0., 0., 1., 0.],
                             [0., 1., 0., 0.],
                             [-1., 0., 0., 0.],
                             [0., 0., 0., 1.]])

rotz_correction = np.matrix([[1., 0., 0., 0.],
                             [0., 1., 0., 0.],
                             [0., 0., 1., 0.],
                             [0., 0., 0., 1.]])

human_rot_correction = roty_correction*rotx_correction




human_trans_start = np.matrix([[1., 0., 0., 0.6],
                               [0., 1., 0., 0.],
                               [0., 0., 1., 1.5],
                               [0., 0., 0., 1.]])

human_model.SetTransform(np.array(human_trans_start*human_rot_correction))


model = 'green_kevin'

def set_human_model(dof, human_arm):
    if not len(dof) == 7:
        print 'There should be exactly 7 values used for arm configuration. ' \
              'But instead ' + str(len(dof)) + 'was sent. This is a problem!'

    v = human_model.GetActiveDOFValues()
    if human_arm == 'leftarm' and model == 'green_kevin':
        v[human_model.GetJoint('green_kevin/body_arm_left_rotx_joint').GetDOFIndex()] = dof[0]
        v[human_model.GetJoint('green_kevin/body_arm_left_rotz_joint').GetDOFIndex()] = dof[1]
        v[human_model.GetJoint('green_kevin/body_arm_left_roty_joint').GetDOFIndex()] = dof[2]
        v[human_model.GetJoint('green_kevin/arm_forearm_left_joint').GetDOFIndex()] = dof[3]
        v[human_model.GetJoint('green_kevin/forearm_hand_left_rotx_joint').GetDOFIndex()] = dof[4]
        v[human_model.GetJoint('green_kevin/forearm_hand_left_roty_joint').GetDOFIndex()] = dof[5]
        v[human_model.GetJoint('green_kevin/forearm_hand_left_rotz_joint').GetDOFIndex()] = dof[6]
    elif human_arm == 'rightarm' and model == 'green_kevin':
        v[human_model.GetJoint('green_kevin/body_arm_right_rotx_joint').GetDOFIndex()] = dof[0]
        v[human_model.GetJoint('green_kevin/body_arm_right_rotz_joint').GetDOFIndex()] = dof[1]
        v[human_model.GetJoint('green_kevin/body_arm_right_roty_joint').GetDOFIndex()] = dof[2]
        v[human_model.GetJoint('green_kevin/arm_forearm_right_joint').GetDOFIndex()] = dof[3]
        v[human_model.GetJoint('green_kevin/forearm_hand_right_rotx_joint').GetDOFIndex()] = dof[4]
        v[human_model.GetJoint('green_kevin/forearm_hand_right_roty_joint').GetDOFIndex()] = dof[5]
        v[human_model.GetJoint('green_kevin/forearm_hand_right_rotz_joint').GetDOFIndex()] = dof[6]
    else:
        print 'Either Im not sure what arm or what model to use to set the arm dof for!'
    human_model.SetActiveDOFValues(v)
    env.UpdatePublishedBodies()
    # print z, bth
    rospy.sleep(1)

# for h in np.arange(0., 0.3, 0.01):
#     for th in np.arange(0.,80.,1.):
arm_dof = [0., 1.57, 0., 0.0, 0.0, 0., -1.3]


set_human_model(arm_dof, 'leftarm')

set_human_model(arm_dof, 'rightarm')


rospy.sleep(100)
# set_autobed(h, th, 0, 0)
# time.sleep(30)

# with env:
#
#     v = autobed.GetActiveDOFValues()
    

