#!/usr/bin/env python

import numpy as np
import math as m
import openravepy as op
import copy

import time
import roslib
roslib.load_manifest('hrl_base_selection')
roslib.load_manifest('hrl_haptic_mpc')
import rospy, rospkg
import tf
from geometry_msgs.msg import PoseStamped
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.cbook import flatten
from itertools import combinations as comb
from operator import itemgetter

from sensor_msgs.msg import JointState
from std_msgs.msg import String
# import hrl_lib.transforms as tr
from hrl_base_selection.srv import BaseMove, BaseMove_multi
from visualization_msgs.msg import Marker, MarkerArray
from helper_functions import createBMatrix, Bmat_to_pos_quat
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

import pickle as pkl
roslib.load_manifest('hrl_lib')
from hrl_lib.util import save_pickle
from random import gauss
# import hrl_haptic_mpc.haptic_mpc_util
# from hrl_haptic_mpc.robot_haptic_state_node import RobotHapticStateServer
import hrl_lib.util as ut

from joblib import Parallel, delayed


class ConfigVisualize(object):

    def __init__(self, config, goals, visualize=True, targets='all_goals', reference_names=['head'], model='chair',
                 tf_listener=None):
        if tf_listener is None:
            self.tf_listener = tf.TransformListener()
        else:
            self.tf_listener = tf_listener
        self.visualize = visualize
        self.model = model
        self.goals = goals
        self.pr2_B_reference = []
        self.reference_names = reference_names

        self.reachable = {}
        self.manipulable = {}
        self.scores = {}
        self.score_length = {}
        self.sorted_scores = {}
        self.setup_openrave()
        if self.model == 'chair':
            headmodel = self.wheelchair.GetLink('head_center')
        elif self.model == 'autobed':
            headmodel = self.autobed.GetLink('head_link')
            ual = self.autobed.GetLink('arm_left_link')
            uar = self.autobed.GetLink('arm_right_link')
            fal = self.autobed.GetLink('forearm_left_link')
            far = self.autobed.GetLink('forearm_right_link')
            thl = self.autobed.GetLink('quad_left_link')
            thr = self.autobed.GetLink('quad_right_link')
            ch = self.autobed.GetLink('upper_body_link')
            pr2_B_ual = np.matrix(ual.GetTransform())
            pr2_B_uar = np.matrix(uar.GetTransform())
            pr2_B_fal = np.matrix(fal.GetTransform())
            pr2_B_far = np.matrix(far.GetTransform())
            pr2_B_thl = np.matrix(thl.GetTransform())
            pr2_B_thr = np.matrix(thr.GetTransform())
            pr2_B_ch = np.matrix(ch.GetTransform())
        else:
            print 'I GOT A BAD MODEL. NOT SURE WHAT TO DO NOW!'
        pr2_B_head = np.matrix(headmodel.GetTransform())
        for y in self.reference_names:
            if y == 'head':
                self.pr2_B_reference.append(pr2_B_head)
            elif y == 'base_link':
                self.pr2_B_reference.append(pr2_B_base_link)
            elif y == 'upper_arm_left':
                self.pr2_B_reference.append(pr2_B_ual)
            elif y == 'upper_arm_right':
                self.pr2_B_reference.append(pr2_B_uar)
            elif y == 'forearm_left':
                self.pr2_B_reference.append(pr2_B_fal)
            elif y == 'forearm_right':
                self.pr2_B_reference.append(pr2_B_far)
            elif y == 'thigh_left':
                self.pr2_B_reference.append(pr2_B_thl)
            elif y == 'thigh_right':
                self.pr2_B_reference.append(pr2_B_thr)
            elif y == 'chest':
                self.pr2_B_reference.append(pr2_B_ch)
        # Sets the wheelchair location based on the location of the head using a few homogeneous transforms.
        self.pr2_B_headfloor = np.matrix([[       1.,        0.,   0.,         0.],
                                          [       0.,        1.,   0.,         0.],
                                          [       0.,        0.,   1.,         0.],
                                          [       0.,        0.,   0.,         1.]])

        self.goal_B_gripper = np.matrix([[0.,  0.,   1.,   0.0],
                                         [0.,  1.,   0.,   0.0],
                                         [-1.,  0.,   0.,  0.0],
                                         [0.,  0.,   0.,   1.0]])

        self.selection_mat = []
        self.reference_mat = []
        self.Tgrasps = []
        self.weights = []
        self.goal_list = []
        self.number_goals = len(self.Tgrasps)
        print 'Score generator received a list of desired goal locations. It contains ', len(goals), ' goal ' \
                                                                                                         'locations.'
        self.selection_mat = np.zeros(len(self.goals))
        self.goal_list = np.zeros([len(self.goals), 4, 4])
        self.reference_mat = np.zeros(len(self.goals))
        for it in xrange(len(self.goals)):
            #self.goal_list.append(pr2_B_head*np.matrix(target[0])*goal_B_gripper)
            self.reference_mat[it] = int(self.goals[it, 2])
            self.goal_list[it] = copy.copy(self.pr2_B_reference[int(self.reference_mat[it])]*np.matrix(self.goals[it, 0]))
            self.selection_mat[it] = int(self.goals[it, 1])

        self.set_goals()
        self.visualize_config(config)

    def receive_new_goals(self, goals):
        self.goals = goals
        # print 'Score generator received a list of desired goal locations. It contains ', len(goals), ' goal ' \
        #                                                                                                  'locations.'
        self.selection_mat = np.zeros(len(self.goals))
        self.goal_list = np.zeros([len(self.goals), 4, 4])
        self.reference_mat = np.zeros(len(self.goals))
        for w in xrange(len(self.goals)):
            #self.goal_list.append(pr2_B_head*np.matrix(target[0])*goal_B_gripper)
            self.reference_mat[w] = int(self.goals[w, 2])
            self.goal_list[w] = copy.copy(self.pr2_B_reference[int(self.reference_mat[w])] *
                                          np.matrix(self.goals[w, 0]))
            self.selection_mat[w] = self.goals[w, 1]

        self.set_goals()

    def set_goals(self):
        self.Tgrasps = []
        self.weights = []
        #total = 0

        for num, selection in enumerate(self.selection_mat):
            #print selection
            if selection != 0.:
                #self.Tgrasps.append(np.array(self.goal_list[num]))
                self.Tgrasps.append(np.array(np.matrix(self.goal_list[num])*self.goal_B_gripper))
                self.weights.append(selection)
                #total += selection
        #print 'Total weights (should be 1) is: ',total

    def choose_task(self, task):
        if task == 'all_goals':
            self.selection_mat = np.ones(len(self.goal_list))
        elif task == 'wipe_face':
            self.selection_mat = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        elif task == 'shoulder':
            self.selection_mat = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        elif task == 'knee':
            self.selection_mat = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0])
        elif task == 'arm':
            self.selection_mat = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1])
        else:
            print 'Somehow I got a bogus task!? \n'
            return None
        self.set_goals()
        print 'The task was just set. The set of goals selected was: ',task
        return self.selection_mat

    def setup_openrave(self):
        # Setup Openrave ENV
        self.env = op.Environment()
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('hrl_base_selection')

        # Lets you visualize openrave. Uncomment to see visualization. Does not work through ssh.
        if self.visualize:
            self.env.SetViewer('qtcoin')
        if self.model == 'chair':
            # This is the new wheelchair model
            self.env.Load(''.join([pkg_path, '/collada/wheelchair_and_body_assembly.dae']))
            self.wheelchair = self.env.GetRobots()[0]
            headmodel = self.wheelchair.GetLink('head_center')
            head_T = np.matrix(headmodel.GetTransform())
            self.originsubject_B_headfloor = np.matrix([[1., 0.,  0., head_T[0, 3]],  # .442603 #.45 #.438
                                                        [0., 1.,  0., head_T[1, 3]],  # 0.34 #.42
                                                        [0., 0.,  1.,           0.],
                                                        [0., 0.,  0.,           1.]])
            self.originsubject_B_originworld = np.matrix(np.eye(4))
        elif self.model == 'autobed':
            self.env.Load(''.join([pkg_path, '/collada/bed_and_body_v3_rounded.dae']))
            self.autobed = self.env.GetRobots()[0]
            # v = self.autobed.GetActiveDOFValues()

            #0 degrees, 0 height
            self.set_autobed(0., 0., 0., 0.)
            headmodel = self.autobed.GetLink('head_link')
            head_T = np.matrix(headmodel.GetTransform())
            self.originsubject_B_headfloor = np.matrix([[1.,  0., 0.,  head_T[0, 3]],  #.45 #.438
                                                        [0.,  1., 0.,  head_T[1, 3]],  # 0.34 #.42
                                                        [0.,  0., 1.,           0.],
                                                        [0.,  0., 0.,           1.]])
            self.originsubject_B_originworld = np.matrix(np.eye(4))

        else:
            print 'I got a bad model. What is going on???'
            return None
        self.subject = self.env.GetBodies()[0]
        self.subject.SetTransform(np.array(self.originsubject_B_originworld))

    def visualize_config(self, config):
        self.robot = {}
        self.manip = {}
        self.manipprob = {}
        print 'config is: ', config
        print config[0][0]
        print len(config[0][0])
        for config_num in xrange(len(config[0][0])):
            ## Load OpenRave PR2 Model
            self.env.Load('robots/pr2-beta-static.zae')
            self.robot[config_num] = self.env.GetRobots()[config_num+1]
            v = self.robot[config_num].GetActiveDOFValues()
            v[self.robot[config_num].GetJoint('l_shoulder_pan_joint').GetDOFIndex()] = 3.14/2
            v[self.robot[config_num].GetJoint('r_shoulder_pan_joint').GetDOFIndex()] = -3.14/2
            v[self.robot[config_num].GetJoint('r_shoulder_lift_joint').GetDOFIndex()] = -0.52
            v[self.robot[config_num].GetJoint('r_elbow_flex_joint').GetDOFIndex()] = -3.14*2/3
            v[self.robot[config_num].GetJoint('r_forearm_roll_joint').GetDOFIndex()] = 0
            v[self.robot[config_num].GetJoint('r_wrist_flex_joint').GetDOFIndex()] = 0
            v[self.robot[config_num].GetJoint('r_wrist_roll_joint').GetDOFIndex()] = 0
            v[self.robot[config_num].GetJoint('l_gripper_l_finger_joint').GetDOFIndex()] = .54
            v[self.robot[config_num].GetJoint('torso_lift_joint').GetDOFIndex()] = .3
            self.robot[config_num].SetActiveDOFValues(v)
            robot_start = np.matrix([[m.cos(0.), -m.sin(0.), 0., 0.],
                                     [m.sin(0.),  m.cos(0.), 0., 0.],
                                     [0.       ,         0., 1., 0.],
                                     [0.       ,         0., 0., 1.]])
            self.robot[config_num].SetTransform(np.array(robot_start))

            ## Set robot manipulators, ik, planner
            self.robot[config_num].SetActiveManipulator('leftarm')
            self.manip[config_num] = self.robot[config_num].GetActiveManipulator()
            ikmodel = op.databases.inversekinematics.InverseKinematicsModel(self.robot[config_num], iktype=op.IkParameterization.Type.Transform6D)
            if not ikmodel.load():
                ikmodel.autogenerate()
            # create the interface for basic manipulation programs
            self.manipprob[config_num] = op.interfaces.BaseManipulation(self.robot[config_num])

            delete_index = []
            x = config[0][0][config_num]
            y = config[0][1][config_num]
            th = config[0][2][config_num]
            z = config[0][3][config_num]
            bz = config[0][4][config_num]
            bth = config[0][5][config_num]
            origin_B_pr2 = np.matrix([[ m.cos(th), -m.sin(th),     0.,         x],
                                      [ m.sin(th),  m.cos(th),     0.,         y],
                                      [        0.,         0.,     1.,        0.],
                                      [        0.,         0.,     0.,        1.]])
            self.robot[config_num].SetTransform(np.array(origin_B_pr2))
            v = self.robot[config_num].GetActiveDOFValues()
            v[self.robot[config_num].GetJoint('torso_lift_joint').GetDOFIndex()] = z
            self.robot[config_num].SetActiveDOFValues(v)
            self.env.UpdatePublishedBodies()

            if self.model == 'chair':
                headmodel = self.wheelchair.GetLink('head_center')
                origin_B_head = np.matrix(headmodel.GetTransform())
                self.selection_mat = np.zeros(len(self.goals))
                self.goal_list = np.zeros([len(self.goals), 4, 4])
                for thing in xrange(len(self.reference_names)):
                    if self.reference_names[thing] == 'head':
                        self.pr2_B_reference[thing] = origin_B_pr2.I*origin_B_head
                    elif self.reference_names[thing] == 'base_link':
                        self.pr2_B_reference[thing] = np.matrix(np.eye(4))
                        # self.pr2_B_reference[j] = np.matrix(self.robot.GetTransform())

                for thing in xrange(len(self.goals)):
                    self.goal_list[thing] = copy.copy(origin_B_pr2*self.pr2_B_reference[int(self.reference_mat[thing])]*np.matrix(self.goals[thing, 0]))
                    self.selection_mat[thing] = copy.copy(self.goals[thing, 1])
    #            for target in self.goals:
    #                self.goal_list.append(pr2_B_head*np.matrix(target[0]))
    #                self.selection_mat.append(target[1])
                self.set_goals()
            elif self.model == 'autobed':
                self.set_autobed(bz, bth, 0., 0.)
                self.selection_mat = np.zeros(len(self.goals))
                self.goal_list = np.zeros([len(self.goals), 4, 4])
                headmodel = self.autobed.GetLink('head_link')
                ual = self.autobed.GetLink('arm_left_link')
                uar = self.autobed.GetLink('arm_right_link')
                fal = self.autobed.GetLink('forearm_left_link')
                far = self.autobed.GetLink('forearm_right_link')
                thl = self.autobed.GetLink('quad_left_link')
                thr = self.autobed.GetLink('quad_right_link')
                ch = self.autobed.GetLink('upper_body_link')
                origin_B_head = np.matrix(headmodel.GetTransform())
                origin_B_ual = np.matrix(ual.GetTransform())
                origin_B_uar = np.matrix(uar.GetTransform())
                origin_B_fal = np.matrix(fal.GetTransform())
                origin_B_far = np.matrix(far.GetTransform())
                origin_B_thl = np.matrix(thl.GetTransform())
                origin_B_thr = np.matrix(thr.GetTransform())
                origin_B_ch = np.matrix(ch.GetTransform())
                for thing in xrange(len(self.reference_names)):
                    if self.reference_names[thing] == 'head':
                        self.pr2_B_reference[thing] = origin_B_pr2.I*origin_B_head
                        # self.pr2_B_reference[thing] = np.matrix(headmodel.GetTransform())
                    elif self.reference_names[thing] == 'base_link':
                        self.pr2_B_reference[thing] = np.matrix(np.eye(4))
                        # self.pr2_B_reference[i] = np.matrix(self.robot.GetTransform())
                    elif self.reference_names[thing] == 'upper_arm_left':
                        self.pr2_B_reference[thing] = origin_B_pr2.I*origin_B_ual
                    elif self.reference_names[thing] == 'upper_arm_right':
                        self.pr2_B_reference[thing] = origin_B_pr2.I*origin_B_uar
                    elif self.reference_names[thing] == 'forearm_left':
                        self.pr2_B_reference[thing] = origin_B_pr2.I*origin_B_fal
                    elif self.reference_names[thing] == 'forearm_right':
                        self.pr2_B_reference[thing] = origin_B_pr2.I*origin_B_far
                    elif self.reference_names[thing] == 'thigh_left':
                        self.pr2_B_reference[thing] = origin_B_pr2.I*origin_B_thl
                    elif self.reference_names[thing] == 'thigh_right':
                        self.pr2_B_reference[thing] = origin_B_pr2.I*origin_B_thr
                    elif self.reference_names[thing] == 'chest':
                        self.pr2_B_reference[thing] = origin_B_pr2.I*origin_B_ch

                for thing in xrange(len(self.goals)):
                    self.goal_list[thing] = copy.copy(origin_B_pr2*self.pr2_B_reference[int(self.reference_mat[thing])]*np.matrix(self.goals[thing, 0]))
                    self.selection_mat[thing] = copy.copy(self.goals[thing, 1])
                # for target in self.goals:
                #     self.goal_list.append(pr2_B_head*np.matrix(target[0]))
                #     self.selection_mat.append(target[1])
                self.set_goals()
            # print 'self.goals length: ', len(self.goals)
            # print 'self.Tgrasps length: ', len(self.Tgrasps)
            reached = 0
            delete_index = []
            # with True: self.robot[config_num]:
            if True:
                if True:
                # if not self.manip.CheckIndependentCollision(op.CollisionReport()):
                    #print 'not colliding with environment'
                    for num, Tgrasp in enumerate(self.Tgrasps):
                        sol = None
                        sol = self.manip[config_num].FindIKSolution(Tgrasp, filteroptions=op.IkFilterOptions.CheckEnvCollisions)
                        # sol = self.manip.FindIKSolution(Tgrasp,filteroptions=op.IkFilterOptions.IgnoreSelfCollisions)
                        if sol is not None and reached == 0:
                            reached = 1
                            delete_index.append(num)
                            if self.visualize:
                                self.robot[config_num].SetDOFValues(sol, self.manip[config_num].GetArmIndices())
                                self.env.UpdatePublishedBodies()
                                rospy.sleep(2)
            if len(self.goals) > 0:
                    self.goals = np.delete(self.goals, delete_index, 0)
        rospy.spin()




    def eval_init_config(self, init_config, goal_data):
        start_time = time.time()
        reached = 0.
        mod_x_err_min = -.025
        mod_x_err_max = .025+.02
        mod_x_err_int = .025
        mod_y_err_min = -.025
        mod_y_err_max = .025+.02
        mod_y_err_int = .025
        mod_th_err_min = -m.pi/36.
        mod_th_err_max = m.pi/36.+.02
        mod_th_err_int = m.pi/36.
        x_err_min = -.05
        x_err_max = .05+.02
        x_err_int = .05
        y_err_min = -.05
        y_err_max = .05+.02
        y_err_int = .05
        th_err_min = -m.pi/36.
        th_err_max = m.pi/36.+.02
        th_err_int = m.pi/36.
        h_err_min = -m.pi/9.
        h_err_max = m.pi/9.+.02
        h_err_int = m.pi/9.
        if self.model == 'chair':
            modeling_error = np.array([err for err in ([x_e, y_e, th_e, h_e, m_x_e, m_y_e, m_th_e]
                                                       for x_e in np.arange(x_err_min, x_err_max, x_err_int)
                                                       for y_e in np.arange(y_err_min, y_err_max, y_err_int)
                                                       for th_e in np.arange(th_err_min, th_err_max, th_err_int)
                                                       for h_e in np.arange(h_err_min, h_err_max, h_err_int)
                                                       for m_x_e in np.arange(mod_x_err_min, mod_x_err_max, mod_x_err_int)
                                                       for m_y_e in np.arange(mod_y_err_min, mod_y_err_max, mod_y_err_int)
                                                       for m_th_e in np.arange(mod_th_err_min, mod_th_err_max, mod_th_err_int)
                                                       )
                                       ])
            # modeling_error = np.array([err for err in ([x_e, y_e, th_e, h_e, 0, 0, 0]
            #                                            for x_e in np.arange(x_err_min, x_err_max, x_err_int)
            #                                            for y_e in np.arange(y_err_min, y_err_max, y_err_int)
            #                                            for th_e in np.arange(th_err_min, th_err_max, th_err_int)
            #                                            for h_e in np.arange(h_err_min, h_err_max, h_err_int)
            #                                            # for m_x_e in np.arange(mod_x_err_min, mod_x_err_max, mod_x_err_int)
            #                                            # for m_y_e in np.arange(mod_y_err_min, mod_y_err_max, mod_y_err_int)
            #                                            # for m_th_e in np.arange(mod_th_err_min, mod_th_err_max, mod_th_err_int)
            #                                            )
            #                            ])
        elif self.model == 'autobed':
            modeling_error = np.array([err for err in ([x_e, y_e]
                                                       for x_e in np.arange(x_err_min, x_err_max, x_err_int)
                                                       for y_e in np.arange(y_err_min, y_err_max, y_err_int)
                                                       )
                                       ])
        # print len(modeling_error)
        # for error in modeling_error:
        #     print error

        total_length = copy.copy(len(self.goals)*len(modeling_error))
        for error in modeling_error:
            self.receive_new_goals(goal_data)
            # origin_B_wheelchair = np.matrix([[m.cos(error[2]), -m.sin(error[2]),     0.,  error[0]],
            #                                  [m.sin(error[2]),  m.cos(error[2]),     0.,  error[1]],
            #                                  [             0.,               0.,     1.,        0.],
            #                                  [             0.,               0.,     0.,        1.]])
            # self.wheelchair.SetTransform(np.array(origin_B_wheelchair))
            if self.model == 'chair':
                origin_B_wheelchair = np.matrix([[m.cos(error[6]), -m.sin(error[6]),     0.,  error[4]],
                                                 [m.sin(error[6]),  m.cos(error[6]),     0.,  error[5]],
                                                 [             0.,               0.,     1.,        0.],
                                                 [             0.,               0.,     0.,        1.]])
                self.wheelchair.SetTransform(np.array(origin_B_wheelchair))
                v = self.wheelchair.GetActiveDOFValues()
                v[self.wheelchair.GetJoint('wheelchair_body_x_joint').GetDOFIndex()] = error[0]
                v[self.wheelchair.GetJoint('wheelchair_body_y_joint').GetDOFIndex()] = error[1]
                v[self.wheelchair.GetJoint('wheelchair_body_rotation_joint').GetDOFIndex()] = error[2]
                v[self.wheelchair.GetJoint('head_neck_joint').GetDOFIndex()] = error[3]
                self.wheelchair.SetActiveDOFValues(v)
                self.env.UpdatePublishedBodies()

            for ic in xrange(len(init_config[0][0])):
                delete_index = []
                x = init_config[0][0][ic]
                y = init_config[0][1][ic]
                th = init_config[0][2][ic]
                z = init_config[0][3][ic]
                bz = init_config[0][4][ic]
                bth = init_config[0][5][ic]
                # print 'bth: ', bth
                origin_B_pr2 = np.matrix([[ m.cos(th), -m.sin(th),     0.,         x],
                                          [ m.sin(th),  m.cos(th),     0.,         y],
                                          [        0.,         0.,     1.,        0.],
                                          [        0.,         0.,     0.,        1.]])
                self.robot.SetTransform(np.array(origin_B_pr2))
                v = self.robot.GetActiveDOFValues()
                v[self.robot.GetJoint('torso_lift_joint').GetDOFIndex()] = z
                self.robot.SetActiveDOFValues(v)
                self.env.UpdatePublishedBodies()
                if self.model == 'chair':
                    headmodel = self.wheelchair.GetLink('head_center')
                    origin_B_head = np.matrix(headmodel.GetTransform())
                    self.selection_mat = np.zeros(len(self.goals))
                    self.goal_list = np.zeros([len(self.goals), 4, 4])
                    for thing in xrange(len(self.reference_names)):
                        if self.reference_names[thing] == 'head':
                            self.pr2_B_reference[thing] = origin_B_pr2.I*origin_B_head
                        elif self.reference_names[thing] == 'base_link':
                            self.pr2_B_reference[thing] = np.matrix(np.eye(4))
                            # self.pr2_B_reference[j] = np.matrix(self.robot.GetTransform())

                    for thing in xrange(len(self.goals)):
                        self.goal_list[thing] = copy.copy(origin_B_pr2*self.pr2_B_reference[int(self.reference_mat[thing])]*np.matrix(self.goals[thing, 0]))
                        self.selection_mat[thing] = copy.copy(self.goals[thing, 1])
        #            for target in self.goals:
        #                self.goal_list.append(pr2_B_head*np.matrix(target[0]))
        #                self.selection_mat.append(target[1])
                    self.set_goals()
                elif self.model == 'autobed':
                    self.set_autobed(bz, bth, error[0], error[1])
                    self.selection_mat = np.zeros(len(self.goals))
                    self.goal_list = np.zeros([len(self.goals), 4, 4])
                    headmodel = self.autobed.GetLink('head_link')
                    ual = self.autobed.GetLink('arm_left_link')
                    uar = self.autobed.GetLink('arm_right_link')
                    fal = self.autobed.GetLink('forearm_left_link')
                    far = self.autobed.GetLink('forearm_right_link')
                    thl = self.autobed.GetLink('quad_left_link')
                    thr = self.autobed.GetLink('quad_right_link')
                    ch = self.autobed.GetLink('upper_body_link')
                    origin_B_head = np.matrix(headmodel.GetTransform())
                    origin_B_ual = np.matrix(ual.GetTransform())
                    origin_B_uar = np.matrix(uar.GetTransform())
                    origin_B_fal = np.matrix(fal.GetTransform())
                    origin_B_far = np.matrix(far.GetTransform())
                    origin_B_thl = np.matrix(thl.GetTransform())
                    origin_B_thr = np.matrix(thr.GetTransform())
                    origin_B_ch = np.matrix(ch.GetTransform())
                    for thing in xrange(len(self.reference_names)):
                        if self.reference_names[thing] == 'head':
                            self.pr2_B_reference[thing] = origin_B_pr2.I*origin_B_head
                            # self.pr2_B_reference[thing] = np.matrix(headmodel.GetTransform())
                        elif self.reference_names[thing] == 'base_link':
                            self.pr2_B_reference[thing] = np.matrix(np.eye(4))
                            # self.pr2_B_reference[i] = np.matrix(self.robot.GetTransform())
                        elif self.reference_names[thing] == 'upper_arm_left':
                            self.pr2_B_reference[thing] = origin_B_pr2.I*origin_B_ual
                        elif self.reference_names[thing] == 'upper_arm_right':
                            self.pr2_B_reference[thing] = origin_B_pr2.I*origin_B_uar
                        elif self.reference_names[thing] == 'forearm_left':
                            self.pr2_B_reference[thing] = origin_B_pr2.I*origin_B_fal
                        elif self.reference_names[thing] == 'forearm_right':
                            self.pr2_B_reference[thing] = origin_B_pr2.I*origin_B_far
                        elif self.reference_names[thing] == 'thigh_left':
                            self.pr2_B_reference[thing] = origin_B_pr2.I*origin_B_thl
                        elif self.reference_names[thing] == 'thigh_right':
                            self.pr2_B_reference[thing] = origin_B_pr2.I*origin_B_thr
                        elif self.reference_names[thing] == 'chest':
                            self.pr2_B_reference[thing] = origin_B_pr2.I*origin_B_ch

                    for thing in xrange(len(self.goals)):
                        self.goal_list[thing] = copy.copy(origin_B_pr2*self.pr2_B_reference[int(self.reference_mat[thing])]*np.matrix(self.goals[thing, 0]))
                        self.selection_mat[thing] = copy.copy(self.goals[thing, 1])
                    # for target in self.goals:
                    #     self.goal_list.append(pr2_B_head*np.matrix(target[0]))
                    #     self.selection_mat.append(target[1])
                    self.set_goals()
                # print 'self.goals length: ', len(self.goals)
                # print 'self.Tgrasps length: ', len(self.Tgrasps)
                with self.robot:
                    if True:
                    # if not self.manip.CheckIndependentCollision(op.CollisionReport()):
                        #print 'not colliding with environment'
                        for num, Tgrasp in enumerate(self.Tgrasps):
                            sol = None
                            sol = self.manip.FindIKSolution(Tgrasp, filteroptions=op.IkFilterOptions.CheckEnvCollisions)
                            # sol = self.manip.FindIKSolution(Tgrasp,filteroptions=op.IkFilterOptions.IgnoreSelfCollisions)
                            if sol is not None:
                                reached += 1.
                                delete_index.append(num)
                                if self.visualize:
                                    self.robot.SetDOFValues(sol, self.manip.GetArmIndices())
                                    self.env.UpdatePublishedBodies()
                                    rospy.sleep(2)

                # print 'goal list: ', self.goals
                # print 'delete list: ', delete_index
                if len(self.goals) > 0:
                    self.goals = np.delete(self.goals, delete_index, 0)
        score = reached/total_length
        print 'Score is (% of reached goals): ', score
        print 'Time to score this initial configuration: %fs' % (time.time()-start_time)
        return score

    def set_autobed(self, z, headrest_th, head_x, head_y):
        bz = z
        bth = m.degrees(headrest_th)
        v = self.autobed.GetActiveDOFValues()
        v[self.autobed.GetJoint('tele_legs_joint').GetDOFIndex()] = bz
        v[self.autobed.GetJoint('head_bed_updown_joint').GetDOFIndex()] = head_x
        v[self.autobed.GetJoint('head_bed_leftright_joint').GetDOFIndex()] = head_y

            # 0 degrees, 0 height
        if (bth >= 0) and (bth <= 40):  # between 0 and 40 degrees
            v[self.autobed.GetJoint('head_rest_hinge').GetDOFIndex()] = (bth/40)*(0.6981317 - 0)+0
            v[self.autobed.GetJoint('neck_body_joint').GetDOFIndex()] = (bth/40)*(-.2-(-.1))+(-.1)
            v[self.autobed.GetJoint('upper_mid_body_joint').GetDOFIndex()] = (bth/40)*(-.17-.4)+.4
            v[self.autobed.GetJoint('mid_lower_body_joint').GetDOFIndex()] = (bth/40)*(-.76-(-.72))+(-.72)
            v[self.autobed.GetJoint('body_quad_left_joint').GetDOFIndex()] = -0.4
            v[self.autobed.GetJoint('body_quad_right_joint').GetDOFIndex()] = -0.4
            v[self.autobed.GetJoint('quad_calf_left_joint').GetDOFIndex()] = 0.1
            v[self.autobed.GetJoint('quad_calf_right_joint').GetDOFIndex()] = 0.1
            v[self.autobed.GetJoint('calf_foot_left_joint').GetDOFIndex()] = (bth/40)*(-.05-.02)+.02
            v[self.autobed.GetJoint('calf_foot_right_joint').GetDOFIndex()] = (bth/40)*(-.05-.02)+.02
            v[self.autobed.GetJoint('body_arm_left_joint').GetDOFIndex()] = (bth/40)*(-.06-(-.12))+(-.12)
            v[self.autobed.GetJoint('body_arm_right_joint').GetDOFIndex()] = (bth/40)*(-.06-(-.12))+(-.12)
            v[self.autobed.GetJoint('arm_forearm_left_joint').GetDOFIndex()] = (bth/40)*(.58-0.05)+.05
            v[self.autobed.GetJoint('arm_forearm_right_joint').GetDOFIndex()] = (bth/40)*(.58-0.05)+.05
            v[self.autobed.GetJoint('forearm_hand_left_joint').GetDOFIndex()] = -0.1
            v[self.autobed.GetJoint('forearm_hand_right_joint').GetDOFIndex()] = -0.1
        elif (bth > 40) and (bth <= 80):  # between 0 and 40 degrees
            v[self.autobed.GetJoint('head_rest_hinge').GetDOFIndex()] = ((bth-40)/40)*(1.3962634 - 0.6981317)+0.6981317
            v[self.autobed.GetJoint('neck_body_joint').GetDOFIndex()] = ((bth-40)/40)*(-.55-(-.2))+(-.2)
            v[self.autobed.GetJoint('upper_mid_body_joint').GetDOFIndex()] = ((bth-40)/40)*(-.51-(-.17))+(-.17)
            v[self.autobed.GetJoint('mid_lower_body_joint').GetDOFIndex()] = ((bth-40)/40)*(-.78-(-.76))+(-.76)
            v[self.autobed.GetJoint('body_quad_left_joint').GetDOFIndex()] = -0.4
            v[self.autobed.GetJoint('body_quad_right_joint').GetDOFIndex()] = -0.4
            v[self.autobed.GetJoint('quad_calf_left_joint').GetDOFIndex()] = 0.1
            v[self.autobed.GetJoint('quad_calf_right_joint').GetDOFIndex()] = 0.1
            v[self.autobed.GetJoint('calf_foot_left_joint').GetDOFIndex()] = ((bth-40)/40)*(-0.1-(-.05))+(-.05)
            v[self.autobed.GetJoint('calf_foot_right_joint').GetDOFIndex()] = ((bth-40)/40)*(-0.1-(-.05))+(-.05)
            v[self.autobed.GetJoint('body_arm_left_joint').GetDOFIndex()] = ((bth-40)/40)*(-.01-(-.06))+(-.06)
            v[self.autobed.GetJoint('body_arm_right_joint').GetDOFIndex()] = ((bth-40)/40)*(-.01-(-.06))+(-.06)
            v[self.autobed.GetJoint('arm_forearm_left_joint').GetDOFIndex()] = ((bth-40)/40)*(.88-0.58)+.58
            v[self.autobed.GetJoint('arm_forearm_right_joint').GetDOFIndex()] = ((bth-40)/40)*(.88-0.58)+.58
            v[self.autobed.GetJoint('forearm_hand_left_joint').GetDOFIndex()] = -0.1
            v[self.autobed.GetJoint('forearm_hand_right_joint').GetDOFIndex()] = -0.1
        else:
            print 'Error: Bed angle out of range (should be 0 - 80 degrees)'

        self.autobed.SetActiveDOFValues(v)
        self.env.UpdatePublishedBodies()


if __name__ == "__main__":
    rospy.init_node('config_visualize')
    mytask = 'shoulder'
    mymodel = 'chair'
    #mytask = 'all_goals'
    start_time = time.time()
    # selector = ScoreGenerator(visualize=False,task=mytask,goals = None,model=mymodel)
    # selector.choose_task(mytask)
    # score_sheet = selector.handle_score()
    print 'I can\'t be run directly at the moment. Try using me correctly!'



