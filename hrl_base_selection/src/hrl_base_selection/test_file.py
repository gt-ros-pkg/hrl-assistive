#!/usr/bin/env python

import openravepy as op
import numpy

import os

#from openravepy.misc import InitOpenRAVELogging 
#InitOpenRAVELogging() 

from openravepy import *
import numpy, time
import rospkg
import math as m
import numpy as np
import rospy

rospy.init_node('test_node')

env = Environment()  # create openrave environment
env.SetViewer('qtcoin')  # attach viewer (optional)

rospack = rospkg.RosPack()
pkg_path = rospack.get_path('hrl_base_selection')
env.Load(''.join([pkg_path, '/collada/bed_and_environment_henry_rounded.dae']))
# env.Load(''.join([pkg_path, '/collada/wheelchair_henry_rounded.dae']))
# env.Load(''.join([pkg_path, '/collada/human.dae']))
autobed = env.GetRobots()[0]
# wheelchair = env.GetRobots()[0]

def set_autobed(z, headrest_th, head_x, head_y):
    bz = z
    # bth = m.degrees(headrest_th)
    bth = headrest_th
    v = autobed.GetActiveDOFValues()
    v[autobed.GetJoint('autobed/tele_legs_joint').GetDOFIndex()] = bz
    v[autobed.GetJoint('autobed/bed_neck_base_updown_bedframe_joint').GetDOFIndex()] = head_x
    v[autobed.GetJoint('autobed/bed_neck_base_leftright_joint').GetDOFIndex()] = head_y
    if bth >= 80. and bth < 85.:
        bth = 80.
    if bth >= -1. and bth <= 0.:
        bth = 0.
        # 0 degrees, 0 height
    if (bth >= 0.) and (bth <= 40.):  # between 0 and 40 degrees
        v[autobed.GetJoint('autobed/bed_neck_worldframe_updown_joint').GetDOFIndex()] = (bth/40)*(0.03 - 0)+0
        v[autobed.GetJoint('autobed/bed_neck_base_updown_bedframe_joint').GetDOFIndex()] = (bth/40)*(-0.13 - 0)+0
        v[autobed.GetJoint('autobed/head_rest_hinge').GetDOFIndex()] = m.radians(bth)
        v[autobed.GetJoint('autobed/headrest_bed_to_worldframe_joint').GetDOFIndex()] = -m.radians(bth)
        v[autobed.GetJoint('autobed/bed_neck_to_bedframe_joint').GetDOFIndex()] = m.radians(bth)
        v[autobed.GetJoint('autobed/neck_twist_joint').GetDOFIndex()] = -((bth/40)*(0 - 0)+0)
        v[autobed.GetJoint('autobed/neck_tilt_joint').GetDOFIndex()] = ((bth/40)*(.7 - 0)+0)
        v[autobed.GetJoint('autobed/neck_body_joint').GetDOFIndex()] = (bth/40)*(.02-(0))+(0)
        v[autobed.GetJoint('autobed/neck_head_rotz_joint').GetDOFIndex()] = -((bth/40)*(0 - 0)+0)
        v[autobed.GetJoint('autobed/neck_head_roty_joint').GetDOFIndex()] = -((bth/40)*(-0.2 - 0)+0)
        v[autobed.GetJoint('autobed/neck_head_rotx_joint').GetDOFIndex()] = -((bth/40)*(0 - 0)+0)
        v[autobed.GetJoint('autobed/upper_mid_body_joint').GetDOFIndex()] = (bth/40)*(0.5-0)+0
        v[autobed.GetJoint('autobed/mid_lower_body_joint').GetDOFIndex()] = (bth/40)*(0.26-0)+(0)
        v[autobed.GetJoint('autobed/body_quad_left_joint').GetDOFIndex()] = -0.05
        v[autobed.GetJoint('autobed/body_quad_right_joint').GetDOFIndex()] = -0.05
        v[autobed.GetJoint('autobed/quad_calf_left_joint').GetDOFIndex()] = .05
        v[autobed.GetJoint('autobed/quad_calf_right_joint').GetDOFIndex()] = .05
        v[autobed.GetJoint('autobed/calf_foot_left_joint').GetDOFIndex()] = (bth/40)*(.0-0)+0
        v[autobed.GetJoint('autobed/calf_foot_right_joint').GetDOFIndex()] = (bth/40)*(.0-0)+0
        v[autobed.GetJoint('autobed/body_arm_left_joint').GetDOFIndex()] = (bth/40)*(-0.15-(-0.15))+(-0.15)
        v[autobed.GetJoint('autobed/body_arm_right_joint').GetDOFIndex()] = (bth/40)*(-0.15-(-0.15))+(-0.15)
        v[autobed.GetJoint('autobed/arm_forearm_left_joint').GetDOFIndex()] = (bth/40)*(.86-0.1)+0.1
        v[autobed.GetJoint('autobed/arm_forearm_right_joint').GetDOFIndex()] = (bth/40)*(.86-0.1)+0.1
        v[autobed.GetJoint('autobed/forearm_hand_left_joint').GetDOFIndex()] = 0
        v[autobed.GetJoint('autobed/forearm_hand_right_joint').GetDOFIndex()] = 0
    elif (bth > 40.) and (bth <= 80.):  # between 0 and 40 degrees
        v[autobed.GetJoint('autobed/bed_neck_worldframe_updown_joint').GetDOFIndex()] = ((bth-40)/40)*(0.03 - (0.03))+(0.03)
        v[autobed.GetJoint('autobed/bed_neck_base_updown_bedframe_joint').GetDOFIndex()] = (bth/40)*(-0.18 - (-0.13))+(-0.13)
        v[autobed.GetJoint('autobed/head_rest_hinge').GetDOFIndex()] = m.radians(bth)
        v[autobed.GetJoint('autobed/headrest_bed_to_worldframe_joint').GetDOFIndex()] = -m.radians(bth)
        v[autobed.GetJoint('autobed/bed_neck_to_bedframe_joint').GetDOFIndex()] = m.radians(bth)
        v[autobed.GetJoint('autobed/neck_twist_joint').GetDOFIndex()] = -(((bth-40)/40)*(0 - 0)+0)
        v[autobed.GetJoint('autobed/neck_tilt_joint').GetDOFIndex()] = (((bth-40)/40)*(0.7 - 0.7)+0.7)
        v[autobed.GetJoint('autobed/neck_body_joint').GetDOFIndex()] = ((bth-40)/40)*(-0.1-(.02))+(.02)
        v[autobed.GetJoint('autobed/neck_head_rotz_joint').GetDOFIndex()] = -((bth/40)*(0 - 0)+0)
        v[autobed.GetJoint('autobed/neck_head_roty_joint').GetDOFIndex()] = -((bth/40)*(.02 - (-0.2))+(-0.2))
        v[autobed.GetJoint('autobed/neck_head_rotx_joint').GetDOFIndex()] = -((bth/40)*(0 - 0)+0)
        v[autobed.GetJoint('autobed/upper_mid_body_joint').GetDOFIndex()] = ((bth-40)/40)*(.7-(.5))+(.5)
        v[autobed.GetJoint('autobed/mid_lower_body_joint').GetDOFIndex()] = ((bth-40)/40)*(.63-(.26))+(.26)
        v[autobed.GetJoint('autobed/body_quad_left_joint').GetDOFIndex()] = -0.05
        v[autobed.GetJoint('autobed/body_quad_right_joint').GetDOFIndex()] = -0.05
        v[autobed.GetJoint('autobed/quad_calf_left_joint').GetDOFIndex()] = 0.05
        v[autobed.GetJoint('autobed/quad_calf_right_joint').GetDOFIndex()] = 0.05
        v[autobed.GetJoint('autobed/calf_foot_left_joint').GetDOFIndex()] = ((bth-40)/40)*(0-0)+(0)
        v[autobed.GetJoint('autobed/calf_foot_right_joint').GetDOFIndex()] = ((bth-40)/40)*(0-0)+(0)
        v[autobed.GetJoint('autobed/body_arm_left_joint').GetDOFIndex()] = ((bth-40)/40)*(-0.1-(-0.15))+(-0.15)
        v[autobed.GetJoint('autobed/body_arm_right_joint').GetDOFIndex()] = ((bth-40)/40)*(-0.1-(-0.15))+(-0.15)
        v[autobed.GetJoint('autobed/arm_forearm_left_joint').GetDOFIndex()] = ((bth-40)/40)*(1.02-0.86)+.86
        v[autobed.GetJoint('autobed/arm_forearm_right_joint').GetDOFIndex()] = ((bth-40)/40)*(1.02-0.86)+.86
        v[autobed.GetJoint('autobed/forearm_hand_left_joint').GetDOFIndex()] = ((bth-40)/40)*(.35-0)+0
        v[autobed.GetJoint('autobed/forearm_hand_right_joint').GetDOFIndex()] = ((bth-40)/40)*(.35-0)+0
    else:
        print 'Error: Bed angle out of range (should be 0 - 80 degrees)'
    autobed.SetActiveDOFValues(v)
    env.UpdatePublishedBodies()
    # print z, bth
    rospy.sleep(30)

def set_wheelchair():
    v = wheelchair.GetActiveDOFValues()
    v[wheelchair.GetJoint('wheelchair/neck_twist_joint').GetDOFIndex()] = 0#m.radians(60)
    v[wheelchair.GetJoint('wheelchair/neck_tilt_joint').GetDOFIndex()] = 0.75
    v[wheelchair.GetJoint('wheelchair/neck_head_rotz_joint').GetDOFIndex()] = 0#-m.radians(30)
    v[wheelchair.GetJoint('wheelchair/neck_head_roty_joint').GetDOFIndex()] = -0.45
    v[wheelchair.GetJoint('wheelchair/neck_head_rotx_joint').GetDOFIndex()] = 0
    v[wheelchair.GetJoint('wheelchair/neck_body_joint').GetDOFIndex()] = -0.15
    v[wheelchair.GetJoint('wheelchair/upper_mid_body_joint').GetDOFIndex()] = 0.4
    v[wheelchair.GetJoint('wheelchair/mid_lower_body_joint').GetDOFIndex()] = 0.4
    v[wheelchair.GetJoint('wheelchair/body_quad_left_joint').GetDOFIndex()] = 0.5
    v[wheelchair.GetJoint('wheelchair/body_quad_right_joint').GetDOFIndex()] = 0.5
    v[wheelchair.GetJoint('wheelchair/quad_calf_left_joint').GetDOFIndex()] = 1.3
    v[wheelchair.GetJoint('wheelchair/quad_calf_right_joint').GetDOFIndex()] = 1.3
    v[wheelchair.GetJoint('wheelchair/calf_foot_left_joint').GetDOFIndex()] = 0.2
    v[wheelchair.GetJoint('wheelchair/calf_foot_right_joint').GetDOFIndex()] = 0.2
    v[wheelchair.GetJoint('wheelchair/body_arm_left_joint').GetDOFIndex()] = 0.6
    v[wheelchair.GetJoint('wheelchair/body_arm_right_joint').GetDOFIndex()] = 0.6
    v[wheelchair.GetJoint('wheelchair/arm_forearm_left_joint').GetDOFIndex()] = .8
    v[wheelchair.GetJoint('wheelchair/arm_forearm_right_joint').GetDOFIndex()] = .8
    v[wheelchair.GetJoint('wheelchair/forearm_hand_left_joint').GetDOFIndex()] = 0.
    v[wheelchair.GetJoint('wheelchair/forearm_hand_right_joint').GetDOFIndex()] = 0.

    wheelchair.SetActiveDOFValues(v)
    env.UpdatePublishedBodies()
    # print z, bth
    rospy.sleep(30)

# for h in np.arange(0., 0.3, 0.01):
#     for th in np.arange(0.,80.,1.):
h = 0
th = 20.
head_x = 0.
head_y = 0.
set_autobed(h, th, head_x, head_y)
# set_wheelchair()
time.sleep(30)
# set_autobed(h, th, 0, 0)
# time.sleep(30)

# with env:
#
#     v = autobed.GetActiveDOFValues()
    



    
    # print v

    #0 everything
    # v[autobed.GetJoint('autobed/tele_legs_joint').GetDOFIndex()] = 0.
    # v[autobed.GetJoint('autobed/leg_rest_upper_hinge').GetDOFIndex()] = 0.
    # v[autobed.GetJoint('autobed/leg_rest_lower_hinge').GetDOFIndex()] = 0.
    # v[autobed.GetJoint('autobed/head_rest_hinge').GetDOFIndex()] = 0.0
    # v[autobed.GetJoint('autobed/head_bed_to_worldframe_joint').GetDOFIndex()] = 0.
    # v[autobed.GetJoint('autobed/head_bed_updown_joint').GetDOFIndex()] = 0.
    # v[autobed.GetJoint('autobed/head_bed_to_bedframe_joint').GetDOFIndex()] = 0.
    # v[autobed.GetJoint('autobed/head_bed_leftright_joint').GetDOFIndex()] = 0.
    # v[autobed.GetJoint('autobed/head_contact_to_head_center').GetDOFIndex()] = 0.
    #
    # v[autobed.GetJoint('autobed/head_neck_x_joint').GetDOFIndex()] = 0.
    # v[autobed.GetJoint('autobed/head_neck_y_joint').GetDOFIndex()] = 0.
    # v[autobed.GetJoint('autobed/head_neck_z_joint').GetDOFIndex()] = 0.
    # v[autobed.GetJoint('autobed/neck_upper_body_top_joint').GetDOFIndex()] = 0.
    # v[autobed.GetJoint('autobed/upper_body_mid_body_x_joint').GetDOFIndex()] = 0.
    #
    # v[autobed.GetJoint('autobed/upper_body_mid_body_y_joint').GetDOFIndex()] = 0.
    # v[autobed.GetJoint('autobed/upper_body_mid_body_z_joint').GetDOFIndex()] = 0.
    # v[autobed.GetJoint('autobed/mid_body_pelvis_x_joint').GetDOFIndex()] = 0.
    # v[autobed.GetJoint('autobed/mid_body_pelvis_y_joint').GetDOFIndex()] = 0.
    # v[autobed.GetJoint('autobed/mid_body_pelvis_z_joint').GetDOFIndex()] = 0.
    # v[autobed.GetJoint('autobed/hip_thigh_left_x_joint').GetDOFIndex()] = 0.
    # v[autobed.GetJoint('autobed/hip_thigh_left_y_joint').GetDOFIndex()] = 0.
    # v[autobed.GetJoint('autobed/hip_thigh_left_z_joint').GetDOFIndex()] = 0.
    # v[autobed.GetJoint('autobed/knee_calf_left_joint').GetDOFIndex()] = 0.
    # v[autobed.GetJoint('autobed/ankle_foot_left_x_joint').GetDOFIndex()] = 0.
    # v[autobed.GetJoint('autobed/ankle_foot_left_y_joint').GetDOFIndex()] = 0.
    # v[autobed.GetJoint('autobed/ankle_foot_left_z_joint').GetDOFIndex()] = 0.
    #
    # v[autobed.GetJoint('autobed/hip_thigh_right_x_joint').GetDOFIndex()] = 0.
    # v[autobed.GetJoint('autobed/hip_thigh_right_y_joint').GetDOFIndex()] = 0.
    # v[autobed.GetJoint('autobed/hip_thigh_right_z_joint').GetDOFIndex()] = 0.
    # v[autobed.GetJoint('autobed/knee_calf_right_joint').GetDOFIndex()] = 0.
    # v[autobed.GetJoint('autobed/ankle_foot_right_x_joint').GetDOFIndex()] = 0.
    # v[autobed.GetJoint('autobed/ankle_foot_right_y_joint').GetDOFIndex()] = 0.
    # v[autobed.GetJoint('autobed/ankle_foot_right_z_joint').GetDOFIndex()] = 0.
    #
    # v[autobed.GetJoint('autobed/upper_body_scapula_left_x_joint').GetDOFIndex()] = 0.
    # v[autobed.GetJoint('autobed/upper_body_scapula_left_z_joint').GetDOFIndex()] = 0.
    # v[autobed.GetJoint('autobed/shoulder_bicep_left_x_joint').GetDOFIndex()] = 0.
    # v[autobed.GetJoint('autobed/shoulder_bicep_left_y_joint').GetDOFIndex()] = 0.
    # v[autobed.GetJoint('autobed/shoulder_bicep_left_z_joint').GetDOFIndex()] = 0.
    # v[autobed.GetJoint('autobed/elbow_forearm_left_joint').GetDOFIndex()] = 0.
    # v[autobed.GetJoint('autobed/wrist_hand_left_x_joint').GetDOFIndex()] = 0.
    # v[autobed.GetJoint('autobed/wrist_hand_left_y_joint').GetDOFIndex()] = 0.
    # v[autobed.GetJoint('autobed/wrist_hand_left_z_joint').GetDOFIndex()] = 0.
    #
    #
    # v[autobed.GetJoint('autobed/upper_body_scapula_right_x_joint').GetDOFIndex()] = 0.
    # v[autobed.GetJoint('autobed/upper_body_scapula_right_z_joint').GetDOFIndex()] = 0.
    # v[autobed.GetJoint('autobed/shoulder_bicep_right_x_joint').GetDOFIndex()] = 0.
    # v[autobed.GetJoint('autobed/shoulder_bicep_right_y_joint').GetDOFIndex()] = 0.
    # v[autobed.GetJoint('autobed/shoulder_bicep_right_z_joint').GetDOFIndex()] = 0.
    # v[autobed.GetJoint('autobed/elbow_forearm_right_joint').GetDOFIndex()] = 0.
    # v[autobed.GetJoint('autobed/wrist_hand_right_x_joint').GetDOFIndex()] = 0.
    # v[autobed.GetJoint('autobed/wrist_hand_right_y_joint').GetDOFIndex()] = 0.
    # v[autobed.GetJoint('autobed/wrist_hand_right_z_joint').GetDOFIndex()] = 0.
    #
    # v[autobed.GetJoint('autobed/head_contact_to_head_center').GetDOFIndex()] = 0.
    # v[autobed.GetJoint('autobed/head_contact_to_head_center').GetDOFIndex()] = 0.
    # v[autobed.GetJoint('autobed/head_contact_to_head_center').GetDOFIndex()] = 0.
    # v[autobed.GetJoint('autobed/head_contact_to_head_center').GetDOFIndex()] = 0.
    # v[autobed.GetJoint('autobed/head_contact_to_head_center').GetDOFIndex()] = 0.
    # v[autobed.GetJoint('autobed/head_contact_to_head_center').GetDOFIndex()] = 0.
    # v[autobed.GetJoint('autobed/head_contact_to_head_center').GetDOFIndex()] = 0.
    # v[autobed.GetJoint('autobed/head_contact_to_head_center').GetDOFIndex()] = 0.
    # v[autobed.GetJoint('autobed/head_contact_to_head_center').GetDOFIndex()] = 0.
    # v[autobed.GetJoint('autobed/head_contact_to_head_center').GetDOFIndex()] = 0.
    # v[autobed.GetJoint('autobed/head_contact_to_head_center').GetDOFIndex()] = 0.
    # v[autobed.GetJoint('autobed/head_contact_to_head_center').GetDOFIndex()] = 0.
    # v[autobed.GetJoint('autobed/head_contact_to_head_center').GetDOFIndex()] = 0.
    # v[autobed.GetJoint('autobed/head_contact_to_head_center').GetDOFIndex()] = 0.
    # v[autobed.GetJoint('autobed/head_contact_to_head_center').GetDOFIndex()] = 0.
    # v[autobed.GetJoint('autobed/head_contact_to_head_center').GetDOFIndex()] = 0.
    # v[autobed.GetJoint('autobed/head_contact_to_head_center').GetDOFIndex()] = 0.
    # v[autobed.GetJoint('autobed/head_contact_to_head_center').GetDOFIndex()] = 0.

    #0 degrees, 0 height
    # v[autobed.GetJoint('autobed/tele_legs_joint').GetDOFIndex()] = 0.5
    # v[autobed.GetJoint('autobed/head_rest_hinge').GetDOFIndex()] = 0.5
    # v[autobed.GetJoint('autobed/head_bed_to_worldframe_joint').GetDOFIndex()] = -0.5
    # v[autobed.GetJoint('autobed/head_bed_to_bedframe_joint').GetDOFIndex()] = 0.5



    # autobed.SetActiveDOFValues(v)
    # env.UpdatePublishedBodies()


# time.sleep(30) # sleep 4 seconds
