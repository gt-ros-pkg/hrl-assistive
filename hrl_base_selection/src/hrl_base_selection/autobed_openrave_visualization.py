#!/usr/bin/env python

from openravepy import *
import numpy, time
env = Environment() # create openrave environment
env.SetViewer('qtcoin') # attach viewer (optional)
#env.Load('robots/pr2-beta-static.zae')
env.Load('/home/ari/git/gt-ros-pkg.hrl-assistive/hrl_base_selection/collada/bed_and_body_v3_rounded.dae')
robot = env.GetRobots()[0] # get the first robot
time.sleep(1)
with env: # lock the environment since robot will be used
    raveLogInfo("Robot "+robot.GetName()+" has "+repr(robot.GetDOF())+" joints with values:\n"+repr(robot.GetDOFValues()))
    v = robot.GetActiveDOFValues()

    # 0 degrees, 0 height
    v[robot.GetJoint('head_rest_hinge').GetDOFIndex()]= 0.0
    v[robot.GetJoint('tele_legs_joint').GetDOFIndex()]= 0.
    v[robot.GetJoint('head_bed_updown_joint').GetDOFIndex()]= .0
    v[robot.GetJoint('head_bed_leftright_joint').GetDOFIndex()]= .0
    #v[robot.GetJoint('head_neck_joint1').GetDOFIndex()]= -.5
    #v[robot.GetJoint('head_neck_joint2').GetDOFIndex()]= .5
    v[robot.GetJoint('neck_body_joint').GetDOFIndex()]= -.1
    v[robot.GetJoint('upper_mid_body_joint').GetDOFIndex()]= .4
    v[robot.GetJoint('mid_lower_body_joint').GetDOFIndex()]= -.72
    v[robot.GetJoint('body_quad_left_joint').GetDOFIndex()]= -0.4
    v[robot.GetJoint('body_quad_right_joint').GetDOFIndex()]= -0.4
    v[robot.GetJoint('quad_calf_left_joint').GetDOFIndex()]= 0.1
    v[robot.GetJoint('quad_calf_right_joint').GetDOFIndex()]= 0.1
    v[robot.GetJoint('calf_foot_left_joint').GetDOFIndex()]= .02
    v[robot.GetJoint('calf_foot_right_joint').GetDOFIndex()]= .02
    v[robot.GetJoint('body_arm_left_joint').GetDOFIndex()]=  -.12
    v[robot.GetJoint('body_arm_right_joint').GetDOFIndex()]= -.12
    v[robot.GetJoint('arm_forearm_left_joint').GetDOFIndex()]= 0.05
    v[robot.GetJoint('arm_forearm_right_joint').GetDOFIndex()]= 0.05
    v[robot.GetJoint('forearm_hand_left_joint').GetDOFIndex()]= -0.1
    v[robot.GetJoint('forearm_hand_right_joint').GetDOFIndex()]= -0.1
    v[robot.GetJoint('leg_rest_upper_joint').GetDOFIndex()]= -0.0
    v[robot.GetJoint('leg_rest_upper_lower_joint').GetDOFIndex()]= -0.0
    robot.SetActiveDOFValues(v)
    env.UpdatePublishedBodies()
    time.sleep(10)

    # 0 degrees, 20 cm height
    v[robot.GetJoint('head_rest_hinge').GetDOFIndex()]= 0.0
    v[robot.GetJoint('tele_legs_joint').GetDOFIndex()]= 0.2
    v[robot.GetJoint('neck_body_joint').GetDOFIndex()]= -.1
    v[robot.GetJoint('upper_mid_body_joint').GetDOFIndex()]= .4
    v[robot.GetJoint('mid_lower_body_joint').GetDOFIndex()]= -.72
    v[robot.GetJoint('body_quad_left_joint').GetDOFIndex()]= -0.4
    v[robot.GetJoint('body_quad_right_joint').GetDOFIndex()]= -0.4
    v[robot.GetJoint('quad_calf_left_joint').GetDOFIndex()]= 0.1
    v[robot.GetJoint('quad_calf_right_joint').GetDOFIndex()]= 0.1
    v[robot.GetJoint('calf_foot_left_joint').GetDOFIndex()]= .02
    v[robot.GetJoint('calf_foot_right_joint').GetDOFIndex()]= .02
    v[robot.GetJoint('body_arm_left_joint').GetDOFIndex()]=  -.12
    v[robot.GetJoint('body_arm_right_joint').GetDOFIndex()]= -.12
    v[robot.GetJoint('arm_forearm_left_joint').GetDOFIndex()]= 0.05
    v[robot.GetJoint('arm_forearm_right_joint').GetDOFIndex()]= 0.05
    v[robot.GetJoint('forearm_hand_left_joint').GetDOFIndex()]= -0.1
    v[robot.GetJoint('forearm_hand_right_joint').GetDOFIndex()]= -0.1
    robot.SetActiveDOFValues(v)
    env.UpdatePublishedBodies()
    time.sleep(10)

    # 80 degrees, 0 height
    v[robot.GetJoint('head_rest_hinge').GetDOFIndex()]= 1.3962634
    v[robot.GetJoint('tele_legs_joint').GetDOFIndex()]= 0.
    v[robot.GetJoint('neck_body_joint').GetDOFIndex()]= -.55
    v[robot.GetJoint('upper_mid_body_joint').GetDOFIndex()]= -.51
    v[robot.GetJoint('mid_lower_body_joint').GetDOFIndex()]= -.78
    v[robot.GetJoint('body_quad_left_joint').GetDOFIndex()]= -0.4
    v[robot.GetJoint('body_quad_right_joint').GetDOFIndex()]= -0.4
    v[robot.GetJoint('quad_calf_left_joint').GetDOFIndex()]= 0.1
    v[robot.GetJoint('quad_calf_right_joint').GetDOFIndex()]= 0.1
    v[robot.GetJoint('calf_foot_left_joint').GetDOFIndex()]= -0.1
    v[robot.GetJoint('calf_foot_right_joint').GetDOFIndex()]= -0.1
    v[robot.GetJoint('body_arm_left_joint').GetDOFIndex()]=  -0.01
    v[robot.GetJoint('body_arm_right_joint').GetDOFIndex()]= -0.01
    v[robot.GetJoint('arm_forearm_left_joint').GetDOFIndex()]=  .88
    v[robot.GetJoint('arm_forearm_right_joint').GetDOFIndex()]= .88
    v[robot.GetJoint('forearm_hand_left_joint').GetDOFIndex()]= -0.1
    v[robot.GetJoint('forearm_hand_right_joint').GetDOFIndex()]= -0.1
    robot.SetActiveDOFValues(v)
    env.UpdatePublishedBodies()
    time.sleep(10)

    # 40 degrees, 0 height
    v[robot.GetJoint('head_rest_hinge').GetDOFIndex()]= 1.3962634/2
    v[robot.GetJoint('tele_legs_joint').GetDOFIndex()]= 0.
    v[robot.GetJoint('neck_body_joint').GetDOFIndex()]= -.2
    v[robot.GetJoint('upper_mid_body_joint').GetDOFIndex()]= -.17
    v[robot.GetJoint('mid_lower_body_joint').GetDOFIndex()]= -.76
    v[robot.GetJoint('body_quad_left_joint').GetDOFIndex()]= -0.4
    v[robot.GetJoint('body_quad_right_joint').GetDOFIndex()]= -0.4
    v[robot.GetJoint('quad_calf_left_joint').GetDOFIndex()]= 0.1
    v[robot.GetJoint('quad_calf_right_joint').GetDOFIndex()]= 0.1
    v[robot.GetJoint('calf_foot_left_joint').GetDOFIndex()]= -.05
    v[robot.GetJoint('calf_foot_right_joint').GetDOFIndex()]= -.05
    v[robot.GetJoint('body_arm_left_joint').GetDOFIndex()]=  -.06
    v[robot.GetJoint('body_arm_right_joint').GetDOFIndex()]= -.06
    v[robot.GetJoint('arm_forearm_left_joint').GetDOFIndex()]=  .58
    v[robot.GetJoint('arm_forearm_right_joint').GetDOFIndex()]= .58
    v[robot.GetJoint('forearm_hand_left_joint').GetDOFIndex()]= -0.1
    v[robot.GetJoint('forearm_hand_right_joint').GetDOFIndex()]= -0.1
    robot.SetActiveDOFValues(v)
    env.UpdatePublishedBodies()
    time.sleep(10)

    #robot.SetDOFValues([-0.15],[0]) # set joint 0 to value 0.5
    #robot.SetDOFValues([0.5],[1]) # set joint 0 to value 0.5
#    robot.SetDOFValues([1.5],[2]) # set joint 0 to value 0.5
#    robot.SetDOFValues([1.5],[3]) # set joint 0 to value 0.5
#    robot.SetDOFValues([1.5],[4]) # set joint 0 to value 0.5

    T = robot.GetLinks()[1].GetTransform() # get the transform of link 1
    time.sleep(20)
    raveLogInfo("The transformation of link 1 is:\n"+repr(T))
