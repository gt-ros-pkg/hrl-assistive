#!/usr/bin/env python

import random, time
import numpy as np
import matplotlib.pyplot as plt
import math as m
import copy, threading
import os, sys, time

from hrl_base_selection.helper_functions import createBMatrix, Bmat_to_pos_quat
from matplotlib.cbook import flatten

import roslib, rospkg, rospy
roslib.load_manifest('hrl_lib')
from std_msgs.msg import Float64MultiArray, MultiArrayLayout, MultiArrayDimension
from hrl_msgs.msg import FloatArrayBare
roslib.load_manifest('hrl_base_selection')
from hrl_base_selection.srv import IKService

import openravepy as op
from openravepy.misc import InitOpenRAVELogging


#
# Service to get a a list of IK solutions from Openrave for a PR2 in free space.
# The list is all IK solutions for the upper arm roll joint discretized to 0.1 radians.
# Can either visualize Openrave IK solutions or not.
# Checking collisions makes it run in ~0.12 seconds.
# Not checking collisions makes it run in ~0.02 seconds.
#
class IKFastService(object):
    def __init__(self, visualize=False, check_self_collision=False):
        self.visualize = visualize
        self.check_self_collision = check_self_collision
        self.frame_lock = threading.RLock()

        # Setup Openrave ENV
        op.RaveSetDebugLevel(op.DebugLevel.Error)
        InitOpenRAVELogging()
        self.env = op.Environment()

        # Lets you visualize openrave. Uncomment to see visualization. Does not work through ssh.
        if self.visualize:
            self.env.SetViewer('qtcoin')

        self.env.Load('robots/pr2-beta-static.zae')
        self.robot = self.env.GetRobots()[0]
        self.robot.CheckLimitsAction = 2

        robot_start = np.matrix([[m.cos(0.), -m.sin(0.), 0., 0.],
                                 [m.sin(0.), m.cos(0.), 0., 0.],
                                 [0., 0., 1., 0.],
                                 [0., 0., 0., 1.]])
        self.robot.SetTransform(np.array(robot_start))

        # Gripper coordinate system has z in direction of the gripper, x is the axis of the gripper opening and closing.
        # This transform corrects that to make x in the direction of the gripper, z the axis of the gripper open.
        # Centered at the very tip of the gripper.
        self.goal_B_gripper = np.matrix([[ 0., 0., 1., 0.0],
                                         [ 0., 1., 0., 0.0],
                                         [-1., 0., 0., 0.0],
                                         [ 0., 0., 0., 1.0]])

        self.gripper_B_tool = np.matrix([[0., -1., 0.,  0.03],
                                         [1.,  0., 0.,   0.0],
                                         [0.,  0., 1., -0.05],
                                         [0.,  0., 0.,   1.0]])

        self.origin_B_grasp = None

        self.arm = None
        self.set_arm('leftarm')
        self.set_arm('rightarm')

        self.simulator_input_service = rospy.Service('ikfast_service', IKService, self.ik_request_handler)
        print 'IKFast service is now ready!'

    def set_arm(self, arm):
        # Set robot manipulators, ik, planner
        if self.arm == arm:
            return False
        elif 'left' in arm or 'right' in arm:
            if 'left' in arm:
                arm = 'leftarm'
            elif 'right' in arm:
                arm = 'rightarm'
            # print 'Switching to ', arm
            self.arm = arm
            self.robot.SetActiveManipulator(arm)
            self.manip = self.robot.GetActiveManipulator()
            ikmodel = op.databases.inversekinematics.InverseKinematicsModel(self.robot, iktype=op.IkParameterization.Type.Transform6D)
            if not ikmodel.load():
                print 'IK model not found for this arm. Generating the ikmodel for the ', arm
                print 'This will take a while'
                ikmodel.autogenerate()
            self.manipprob = op.interfaces.BaseManipulation(self.robot)
            return True
        else:
            print 'I do not know what arm to use. Unrecognized arm choice. Must be leftarm or rightarm!'
            return False

    def ik_request_handler(self, req):
        # self.start_time = rospy.Time.now()
        # print 'received request'
        with self.frame_lock:
            jacobians = []
            self.set_arm(str(req.arm))
            self.position = [float(i) for i in req.position]
            self.orientation = [float(i) for i in req.orientation]
            self.spine_z = float(req.spine_height)

            with self.env:
                v = self.robot.GetActiveDOFValues()
                v[self.robot.GetJoint('l_shoulder_pan_joint').GetDOFIndex()] = 3.14 / 2
                v[self.robot.GetJoint('l_shoulder_lift_joint').GetDOFIndex()] = -0.52
                v[self.robot.GetJoint('l_upper_arm_roll_joint').GetDOFIndex()] = 0.
                v[self.robot.GetJoint('l_elbow_flex_joint').GetDOFIndex()] = -3.14 * 2 / 3
                v[self.robot.GetJoint('l_forearm_roll_joint').GetDOFIndex()] = 0.
                v[self.robot.GetJoint('l_wrist_flex_joint').GetDOFIndex()] = 0.
                v[self.robot.GetJoint('l_wrist_roll_joint').GetDOFIndex()] = 0.
                v[self.robot.GetJoint('r_shoulder_pan_joint').GetDOFIndex()] = -3.14 / 2
                v[self.robot.GetJoint('r_shoulder_lift_joint').GetDOFIndex()] = -0.52
                v[self.robot.GetJoint('r_upper_arm_roll_joint').GetDOFIndex()] = 0.
                v[self.robot.GetJoint('r_elbow_flex_joint').GetDOFIndex()] = -3.14 * 2 / 3
                v[self.robot.GetJoint('r_forearm_roll_joint').GetDOFIndex()] = 0.
                v[self.robot.GetJoint('r_wrist_flex_joint').GetDOFIndex()] = 0.
                v[self.robot.GetJoint('r_wrist_roll_joint').GetDOFIndex()] = 0.
                v[self.robot.GetJoint('torso_lift_joint').GetDOFIndex()] = self.spine_z
                self.robot.SetActiveDOFValues(v, checklimits=2)
                self.env.UpdatePublishedBodies()

                base_footprint_B_tool_goal = createBMatrix(self.position, self.orientation)

                self.origin_B_grasp = np.array(base_footprint_B_tool_goal*self.goal_B_gripper)  # * self.gripper_B_tool.I * self.goal_B_gripper)
                # print 'here'
                # print self.origin_B_grasp
                # sols = self.manip.FindIKSolutions(self.origin_B_grasp,4)
                # init_time = rospy.Time.now()
                if self.check_self_collision:
                    sols = self.manip.FindIKSolutions(self.origin_B_grasp, filteroptions=op.IkFilterOptions.CheckEnvCollisions)
                else:
                    sols = self.manip.FindIKSolutions(self.origin_B_grasp, filteroptions=op.IkFilterOptions.IgnoreSelfCollisions)
                # after_time = rospy.Time.now()-init_time
                # print 'down here'
                # print 'ik time: ', after_time.to_sec()
                if list(sols):
                    with self.robot:
                        for sol in sols:
                            # self.robot.SetDOFValues(sol, self.manip.GetArmIndices(), checklimits=2)
                            self.robot.SetDOFValues(sol, self.manip.GetArmIndices())
                            self.env.UpdatePublishedBodies()
                            jacobians.append(np.vstack([self.manip.CalculateJacobian(), self.manip.CalculateAngularVelocityJacobian()]))
                            if self.visualize:
                                rospy.sleep(1.5)
                    # print jacobians[0]
                    return self.handle_output(sols, jacobians)
                else:
                    return Float64MultiArray(), Float64MultiArray()

    def handle_output(self, sols, jacobians):
        solsmsg = Float64MultiArray()
        solsdata = list(flatten(sols))
        solsdata = [float(i) for i in solsdata]
        solsmsg.data = solsdata
        solslayout = MultiArrayLayout()
        solsdimension = MultiArrayDimension()
        solsdimension.label = 'solutions'
        solsdimension.size = len(sols)
        solsdimension.stride = len(sols)*len(sols[0]) * 1
        solslayout.dim.append(copy.copy(solsdimension))
        solsdimension = MultiArrayDimension()
        solsdimension.label = 'joint_values_per_solution'
        solsdimension.size = len(sols[0])
        solsdimension.stride = len(sols[0]) * 1
        solslayout.dim.append(copy.copy(solsdimension))
        solsmsg.layout = solslayout

        jacmsg = Float64MultiArray()
        jacdata = list(flatten(jacobians))
        jacdata = [float(i) for i in jacdata]
        jacmsg.data = jacdata
        jaclayout = MultiArrayLayout()
        jacdimension = MultiArrayDimension()
        jacdimension.label = 'jacobians'
        jacdimension.size = len(jacobians)
        jacdimension.stride = len(jacobians) * len(jacobians[0]) * len(jacobians[0][0])
        jaclayout.dim.append(copy.copy(jacdimension))
        jacdimension = MultiArrayDimension()
        jacdimension.label = 'jacobians_per_solution 7x6 flattened'
        jacdimension.size = len(jacobians[0])
        jacdimension.stride = len(jacobians[0]) * len(jacobians[0][0])
        jaclayout.dim.append(copy.copy(jacdimension))
        jacmsg.layout = jaclayout
        # elapsed_time = rospy.Time.now() - self.start_time
        # print 'elapsed_time = ', elapsed_time.to_sec()
        return solsmsg, jacmsg


if __name__ == "__main__":
    rospy.init_node('ikfast_service_node')
    dressingtest = IKFastService(visualize=False, check_self_collision=True)
    rospy.spin()
