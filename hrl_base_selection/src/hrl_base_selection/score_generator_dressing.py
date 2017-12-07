#!/usr/bin/env python

import numpy as np
import math as m
import openravepy as op
from openravepy.misc import InitOpenRAVELogging
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
from hrl_base_selection.srv import BaseMove#, BaseMove_multi
from visualization_msgs.msg import Marker, MarkerArray
from helper_functions import createBMatrix, Bmat_to_pos_quat
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import random

import pickle as pkl
roslib.load_manifest('hrl_lib')
from hrl_lib.util import save_pickle
from random import gauss
# import hrl_haptic_mpc.haptic_mpc_util
# from hrl_haptic_mpc.robot_haptic_state_node import RobotHapticStateServer
import hrl_lib.util as ut

import sensor_msgs.point_cloud2 as pc2

import cma

from joblib import Parallel, delayed


class ScoreGeneratorDressing(object):

    def __init__(self, visualize=False, targets='all_goals', reference_names=['head'],
                 goals=None, model='green_kevin',
                 task='hospital_gown_full'):

        self.visualize = visualize
        self.model = model

        self.arm = 'leftarm'
        self.opposite_arm = 'rightarm'

        self.human_rot_correction = None

        self.human_arm = None
        self.human_manip = None
        self.human_model = None

        self.a_model_is_loaded = False
        self.goals = goals
        self.pr2_B_reference = []
        self.task = task
        self.task_dict = None

        self.reference_names = reference_names

        self.head_angles = []

        self.reachable = {}
        self.manipulable = {}
        self.scores = {}

        self.distance = 0.
        self.score_length = {}
        self.sorted_scores = {}

        self.gripper_B_tool = np.matrix([[0., -1., 0., 0.03],
                                         [1., 0., 0., 0.0],
                                         [0., 0., 1., -0.05],
                                         [0., 0., 0., 1.]])

        self.optimal_z_offset = 0.05

        self.setup_openrave()
        # The reference frame for the pr2 base link
        origin_B_pr2 = np.matrix([[       1.,        0.,   0.,         0.0],
                                  [       0.,        1.,   0.,         0.0],
                                  [       0.,        0.,   1.,         0.0],
                                  [       0.,        0.,   0.,         1.0]])
        pr2_B_head = []
        # Sets the wheelchair location based on the location of the head using a few homogeneous transforms.
        # This is only used to visualize in rviz, because the visualization is done before initializing openrave
        self.origin_B_references = []
        if self.model == 'green_kevin':
            headmodel = self.human_model.GetLink('green_kevin/head_link')
            ual = self.human_model.GetLink('green_kevin/arm_left_link')
            uar = self.human_model.GetLink('green_kevin/arm_right_link')
            fal = self.human_model.GetLink('green_kevin/forearm_left_link')
            far = self.human_model.GetLink('green_kevin/forearm_right_link')
            hl = self.human_model.GetLink('green_kevin/hand_left_link')
            hr = self.human_model.GetLink('green_kevin/hand_right_link')
            thl = self.human_model.GetLink('green_kevin/quad_left_link')
            thr = self.human_model.GetLink('green_kevin/quad_right_link')
            calfl = self.human_model.GetLink('green_kevin/calf_left_link')
            calfr = self.human_model.GetLink('green_kevin/calf_right_link')
            ub = self.human_model.GetLink('green_kevin/upper_body_link')
            origin_B_head = np.matrix(headmodel.GetTransform())
            origin_B_ual = np.matrix(ual.GetTransform())
            origin_B_uar = np.matrix(uar.GetTransform())
            origin_B_fal = np.matrix(fal.GetTransform())
            origin_B_far = np.matrix(far.GetTransform())
            origin_B_hl = np.matrix(hl.GetTransform())
            origin_B_hr = np.matrix(hr.GetTransform())
            origin_B_thl = np.matrix(thl.GetTransform())
            origin_B_thr = np.matrix(thr.GetTransform())
            origin_B_calfl = np.matrix(calfl.GetTransform())
            origin_B_calfr = np.matrix(calfr.GetTransform())
            origin_B_ub = np.matrix(ub.GetTransform())
        else:
            print 'I GOT A BAD MODEL. NOT SURE WHAT TO DO NOW!'
        # origin_B_head = np.matrix(headmodel.GetTransform())

        for y in self.reference_names:
            if y == 'head':
                self.origin_B_references.append(origin_B_head)
            elif y == 'base_link':
                self.origin_B_references.append(origin_B_pr2)
            elif y == 'upper_arm_left':
                self.origin_B_references.append(origin_B_ual)
            elif y == 'upper_arm_right':
                self.origin_B_references.append(origin_B_uar)
            elif y == 'forearm_left':
                self.origin_B_references.append(origin_B_fal)
            elif y == 'forearm_right':
                self.origin_B_references.append(origin_B_far)
            elif y == 'hand_left':
                self.origin_B_references.append(origin_B_hl)
            elif y == 'hand_right':
                self.origin_B_references.append(origin_B_hr)
            elif y == 'thigh_left':
                self.origin_B_references.append(origin_B_thl)
            elif y == 'thigh_right':
                self.origin_B_references.append(origin_B_thr)
            elif y == 'knee_left':
                self.origin_B_references.append(origin_B_calfl)
            elif y == 'knee_right':
                self.origin_B_references.append(origin_B_calfr)
            elif y == 'upper_body':
                self.origin_B_references.append(origin_B_ub)
            elif y is None:
                self.origin_B_references.append(np.matrix(np.eye(4)))
        # Sets the wheelchair location based on the location of the head using a few homogeneous transforms.
        self.pr2_B_headfloor = np.matrix([[       1.,        0.,   0.,         0.],
                                          [       0.,        1.,   0.,         0.],
                                          [       0.,        0.,   1.,         0.],
                                          [       0.,        0.,   0.,         1.]])

        # Gripper coordinate system has z in direction of the gripper, x is the axis of the gripper opening and closing.
        # This transform corrects that to make x in the direction of the gripper, z the axis of the gripper open.
        # Centered at the very tip of the gripper.
        self.goal_B_gripper = np.matrix([[0.,  0.,   1.,   0.0],
                                         [0.,  1.,   0.,   0.0],
                                         [-1.,  0.,   0.,  0.0],
                                         [0.,  0.,   0.,   1.0]])

        self.selection_mat = []
        self.reference_mat = []
        self.origin_B_grasps = []
        self.weights = []
        self.goal_list = []
        if self.goals is not None:
            self.number_goals = len(self.goals)
            print 'Score generator received a list of desired goal locations on initialization. ' \
                  'It contains ', len(goals), ' goal locations.'
            self.selection_mat = np.zeros(len(self.goals))
            self.goal_list = np.zeros([len(self.goals), 4, 4])
            self.reference_mat = np.zeros(len(self.goals))
            for it in xrange(len(self.goals)):
                #self.goal_list.append(pr2_B_head*np.matrix(target[0])*goal_B_gripper)
                self.reference_mat[it] = int(self.goals[it, 2])

                # goal_list is origin_B_goal
                self.goal_list[it] = copy.copy(self.origin_B_references[int(self.reference_mat[it])]*np.matrix(self.goals[it, 0]))
                self.selection_mat[it] = self.goals[it, 1]
            self.set_goals()

    # def set_all_goals(self, ):
    #
    #
    #         for goal in self.goals:
    #
    #
    #         self.goals.append()




    def receive_new_goals(self, goals, task, human_arm, pr2_arm, reference_options=None):
        if pr2_arm == 'rightarm':
            pass
            # self.gripper_B_tool = np.matrix([[0., -1., 0., 0.03],
            #                                  [1., 0., 0., 0.0],
            #                                  [0., 0., 1., -0.08],
            #                                  [0., 0., 0., 1.]])
        else:
            print 'I do not know the transform for this pr2 arm tool'

        if reference_options:
            self.reference_names = reference_options
            self.origin_B_references = []
            if self.model == 'green_kevin':
                if human_arm == 'rightarm':
                    if task == 'hospital_gown':
                        headmodel = self.human_model.GetLink('green_kevin/head_link')
                        ual = self.human_model.GetLink('green_kevin/arm_left_link')
                        uar = self.human_model.GetLink('green_kevin/arm_right_link')
                        fal = self.human_model.GetLink('green_kevin/forearm_left_link')
                        far = self.human_model.GetLink('green_kevin/forearm_right_link')
                        hl = self.human_model.GetLink('green_kevin/hand_left_link')
                        hr = self.human_model.GetLink('green_kevin/hand_right_link')
                        thl = self.human_model.GetLink('green_kevin/quad_left_link')
                        thr = self.human_model.GetLink('green_kevin/quad_right_link')
                        calfl = self.human_model.GetLink('green_kevin/calf_left_link')
                        calfr = self.human_model.GetLink('green_kevin/calf_right_link')
                        ub = self.human_model.GetLink('green_kevin/upper_body_link')
                        origin_B_head = np.matrix(headmodel.GetTransform())
                        origin_B_ual = np.matrix(ual.GetTransform())
                        origin_B_uar = np.matrix(uar.GetTransform())
                        origin_B_fal = np.matrix(fal.GetTransform())
                        origin_B_far = np.matrix(far.GetTransform())
                        origin_B_hl = np.matrix(hl.GetTransform())
                        origin_B_hr = np.matrix(hr.GetTransform())
                        origin_B_thl = np.matrix(thl.GetTransform())
                        origin_B_thr = np.matrix(thr.GetTransform())
                        origin_B_calfl = np.matrix(calfl.GetTransform())
                        origin_B_calfr = np.matrix(calfr.GetTransform())
                        origin_B_ub = np.matrix(ub.GetTransform())
                        origin_B_head = np.matrix(headmodel.GetTransform())
            else:
                print 'I GOT A BAD MODEL. NOT SURE WHAT TO DO NOW!'

            for y in self.reference_names:
                if y == 'head':
                    self.origin_B_references.append(origin_B_head)
                elif y == 'base_link':
                    self.origin_B_references.append(np.matrix(np.eye(4)))
                elif y == 'upper_arm_left':
                    self.origin_B_references.append(origin_B_ual)
                elif y == 'upper_arm_right':
                    self.origin_B_references.append(origin_B_uar)
                elif y == 'forearm_left':
                    self.origin_B_references.append(origin_B_fal)
                elif y == 'forearm_right':
                    self.origin_B_references.append(origin_B_far)
                elif y == 'hand_left':
                    self.origin_B_references.append(origin_B_hl)
                elif y == 'hand_right':
                    self.origin_B_references.append(origin_B_hr)
                elif y == 'thigh_left':
                    self.origin_B_references.append(origin_B_thl)
                elif y == 'thigh_right':
                    self.origin_B_references.append(origin_B_thr)
                elif y == 'knee_left':
                    self.origin_B_references.append(origin_B_calfl)
                elif y == 'knee_right':
                    self.origin_B_references.append(origin_B_calfr)
                elif y == 'upper_body':
                    self.origin_B_references.append(origin_B_ub)
                else:
                    print 'The refence options is bogus! I dont know what to do!'
                    return
            self.goals = goals
        # print 'Score generator received a new list of desired goal locations. It contains ', len(goals), ' goal ' \
        #                                                                                                  'locations.'
        self.selection_mat = np.zeros(len(self.goals))
        self.goal_list = np.zeros([len(self.goals), 4, 4])
        self.reference_mat = np.zeros(len(self.goals))
        for w in xrange(len(self.goals)):
            temp_B = self.origin_B_references[int(self.reference_mat[w])]*\
                          np.matrix(self.goals[w, 0])
            temp_B[2, 3] += self.optimal_z_offset

            #self.goal_list.append(pr2_B_head*np.matrix(target[0])*goal_B_gripper)
            self.reference_mat[w] = int(self.goals[w, 2])
            self.goal_list[w] = np.matrix(copy.copy(temp_B))*self.gripper_B_tool
            self.selection_mat[w] = self.goals[w, 1]

        self.set_goals()

    def set_goals(self, single_goal=False):
        if not single_goal:
            self.origin_B_grasps = []
            self.weights = []
            for num in xrange(len(self.selection_mat)):
                if self.selection_mat[num] != 0:
                    #self.origin_B_grasps.append(np.array(self.goal_list[num]))
                    self.origin_B_grasps.append(np.array(np.matrix(self.goal_list[num])*self.goal_B_gripper))
                    self.weights.append(self.selection_mat[num])
        else:
            self.origin_B_grasps = []
            self.weights = []
            if self.selection_mat[0] != 0:
                self.origin_B_grasps.append(np.array(np.matrix(self.goal_list[0])*self.goal_B_gripper))
                self.weights.append(self.selection_mat[0])

    def set_pr2_arm(self, arm):
        ## Set robot manipulators, ik, planner
        print 'Setting the arm being used by base selection to ', arm
        self.arm = arm
        if arm == 'leftarm':
            self.opposite_arm = 'rightarm'
        elif arm == 'rightarm':
            self.opposite_arm = 'leftarm'
        else:
            print 'ERROR'
            print 'I do not know what arm to be using'
            return None
        self.robot.SetActiveManipulator(arm)
        self.manip = self.robot.GetActiveManipulator()
        ikmodel = op.databases.inversekinematics.InverseKinematicsModel(self.robot, iktype=op.IkParameterization.Type.Transform6D)
        if not ikmodel.load():
            print 'IK model not found. Will now generate an IK model. This will take a while!'
            ikmodel.autogenerate()
        self.manipprob = op.interfaces.BaseManipulation(self.robot)

    def set_human_arm(self, arm):
        ## Set robot manipulators, ik, planner
        print 'Setting the human arm being used by base selection to ', arm
        self.human_arm = arm
        # self.human_model.SetActiveManipulator(arm)
        # self.human_manip = self.human_model.GetActiveManipulator()

    def handle_score_generation(self, plot=False):
        scoring_start_time = rospy.Time.now()
        print 'Starting to generate the score. This is going to take a while.'
        # Results are stored in the following format:
        # optimization_results[<model>, <number_of_configs>]
        # Negative head read angle means head rest angle is a free DoF.

        if self.model == 'green_kevin':
            score_parameters = ([t for t in ((tuple([self.model, num_configs, human_arm, pr2_arm]))
                                             for num_configs in [1]
                                             for human_arm in ['rightarm']
                                             for pr2_arm in ['rightarm']
                                             )
                                 ])
        else:
            print 'ERROR'
            print 'I do not know what model to use!'
            return None

        start_time = rospy.Time.now()

        optimization_results = dict.fromkeys(score_parameters)
        score_stuff = dict.fromkeys(score_parameters)
        # optimization_results[<model>, <max_number_of_configs>]

        for parameters in score_parameters:
            parameter_start_time = time.time()
            print 'Generating score for the following parameters: '
            print '[<model>, <max_number_of_configs>]'
            print parameters
            num_config = parameters[1]

            self.set_pr2_arm(parameters[3])
            self.set_human_arm(parameters[2])

            if self.model == 'green_kevin' and num_config == 1:
                maxiter = 20
                popsize = m.pow(5, 2)*100
                # cma parameters: [pr2_base_x, pr2_base_y, pr2_base_theta, pr2_base_height,
                # human_arm_dof_1, human_arm_dof_2, human_arm_dof_3, human_arm_dof_4, human_arm_dof_5,
                # human_arm_dof_6, human_arm_dof_7]
                parameters_min = np.array([-2., -2., -m.pi-.001, 0.,
                                           -0.1, -0.1, -0.35, 0.0, 0.0, -0.1, -1.3])
                                           # -0.05, 1.05, 0.025, 0.03, 0.0, -0.05, -0.05])
                parameters_max = np.array([2., 2., m.pi+.001, 0.3,
                                           0.1, 2., 2., 2., 2.5, 0.1, 0.3])
                                           # 0.1, 1.07, 0.03, 0.07, 0.05, 0.01, 0.03])
                parameters_scaling = (parameters_max-parameters_min)/4.
                parameters_initialization = (parameters_max+parameters_min)/2.
                opts1 = {'seed': 1234, 'ftarget': -1.,'popsize': popsize, 'maxiter': maxiter, 'maxfevals': 1e8,
                         'CMA_cmean':0.25,
                         'scaling_of_variables': list(parameters_scaling),
                         'bounds': [list(parameters_min), list(parameters_max)]}

                # optimization_results[<model>, <number_of_configs>, <head_rest_angle>, <headx>, <heady>, <allow_bed_movement>]
                optimization_results[parameters] = cma.fmin(self.objective_function_one_config,
                                                            list(parameters_initialization),
                                                            1.,
                                                            options=opts1)
                config = optimization_results[parameters][0]
                score = optimization_results[parameters][1]
                print 'Config: ', config
                print 'Score: ', score
                # optimization_results[<model>, <max_number_of_configs>]
                score_stuff[parameters] = [config, score]

            elif self.model == 'green_kevin' and num_config == 2:
                maxiter = 10
                popsize = m.pow(4, 2)*100
                parameters_min = np.array([-2., -2., -m.pi-.001, 0., -2., -2., -m.pi-.001, 0., -1., -1., -1., -1., -1., -1., -1.])
                parameters_max = np.array([2., 2., m.pi+.001, 0.3, 2., 2., m.pi+.001, 0.3, 1., 1., 1., 1., 1., 1., 1.])
                parameters_scaling = (parameters_max-parameters_min)/4.
                parameters_initialization = (parameters_max+parameters_min)/2.
                parameters_initialization[1] = 1.0
                parameters_initialization[5] = -1.0
                opts2 = {'seed': 1234, 'ftarget': -1., 'popsize': popsize, 'maxiter': maxiter, 'maxfevals': 1e8, 'CMA_cmean': 0.25,
                         'scaling_of_variables': list(parameters_scaling),
                         'bounds': [list(parameters_min),
                                    list(parameters_max)]}
                # print 'Working on heady location:', self.heady
                # optimization_results[<model>, <number_of_configs>]
                optimization_results[parameters] = cma.fmin(self.objective_function_two_configs,
                                                            list(parameters_initialization),
                                                            1.,
                                                            options=opts2)

                config = optimization_results[parameters][0]
                score = optimization_results[parameters][1]
                optimization_results[parameters] = [config, score]

                score_stuff[parameters] = self.check_which_num_base_is_better(optimization_results[parameters])

            else:
                print 'Im not sure what parameters to use or model or something is wrong.'
                return None

            # print 'Time to find scores for this set of parameters: %fs' % (time.time()-parameter_start_time)
            # print 'Time elapsed so far for parameters: %fs' % (time.time()-scoring_start_time)

        print 'SCORE RESULTS:'
        for item in score_stuff:
            print '(<model>, <number_of_configs>):', item
            print '[[[x], [y], [th], [z], human_7_dof], score]'
            print 'Or, if there are two configurations:'
            print '[[[x1, x2], [y1, y2], [th1, th2], [z1, z2], human_7_dof], score]'
            print score_stuff[item]

        # print 'Time to generate all scores for individual base locations: %fs' % (time.time()-start_time)
        # print 'Number of configurations that were evaluated: ', len(score_stuff)
        # start_time = time.time()

        return score_stuff

    def check_which_num_base_is_better(self, results):
        result_bases = results[0]
        score = results[1]
        base1 = np.reshape(result_bases[0:6], [6, 1])
        base2 = np.reshape(result_bases[6:12], [6, 1])
        double_base = np.hstack([base1, base2])
        bases = []
        bases.append(base1)
        bases.append(base2)
        bases.append(double_base)
        # scores = []
        # for base in bases:
        # self.visualize = True
        scores = self.score_two_configs(double_base)
        print 'Scores are: ', scores
        ind = scores.argmin()
        if ind == 2:
            print 'Two bases was better'
        else:
            print 'One base was better. It was base (0 or 1) from the two base config solution:', ind
        output = [bases[ind], scores[ind]]
        # if 10.-scores[ind] < 0.95*(10.-score):
        #     print 'Somehow the best score when comparing the single and double base configurations was less than the' \
        #           'score given earlier, even given the discount on two configs'
        return output

    def compare_results_one_vs_two_configs(self, results1, results2):
        if (10. - results1[0][1]) >= (10. - results2[0][1])*0.95 and (10. - results1[0][1]) > 0.:
            print 'One config is as good or better than two configs'
            output = [np.resize(results1[0][0], [1, 6]), 10. - results1[0][1]]
        elif (10. - results2[0][1]) > (10. - results1[0][1]) and (10. - results2[0][1]) > 0.:
            print 'Two configs are better than one config'
            output = [np.resize(results2[0][0], [2, 6]), 10. - results2[0][1]]
        else:
            print 'There is no good result from the optimization with either one or two configs'
            output = [np.zeros(6), 0.]
        return output

    def objective_function_one_config(self, current_parameters):
        # current_parameters = [-0.51079047,  0.29280798, -0.25168599,  0.27497463, -0.01284536,
        #                       0.25523379, -0.26786303,  0.05807374,  0.20383625, -0.09342414,
        #                       -0.45891824]
        x = current_parameters[0]
        y = current_parameters[1]
        th = current_parameters[2]
        z = current_parameters[3]
        human_dof = [current_parameters[4], current_parameters[5], current_parameters[6], current_parameters[7],
                     current_parameters[8], current_parameters[9], current_parameters[10]]

        origin_B_pr2 = np.matrix([[ m.cos(th), -m.sin(th),     0.,         x],
                                  [ m.sin(th),  m.cos(th),     0.,         y],
                                  [        0.,         0.,     1.,        0.],
                                  [        0.,         0.,     0.,        1.]])
        self.robot.SetTransform(np.array(origin_B_pr2))
        v = self.robot.GetActiveDOFValues()
        v[self.robot.GetJoint('torso_lift_joint').GetDOFIndex()] = z
        self.robot.SetActiveDOFValues(v, 2)

        if self.model == 'green_kevin':
            self.set_human_model_dof(human_dof)
            # self.env.UpdatePublishedBodies()

            headmodel = self.human_model.GetLink('green_kevin/head_link')
            ual = self.human_model.GetLink('green_kevin/arm_left_link')
            uar = self.human_model.GetLink('green_kevin/arm_right_link')
            fal = self.human_model.GetLink('green_kevin/forearm_left_link')
            far = self.human_model.GetLink('green_kevin/forearm_right_link')
            hl = self.human_model.GetLink('green_kevin/hand_left_link')
            hr = self.human_model.GetLink('green_kevin/hand_right_link')
            thl = self.human_model.GetLink('green_kevin/quad_left_link')
            thr = self.human_model.GetLink('green_kevin/quad_right_link')
            calfl = self.human_model.GetLink('green_kevin/calf_left_link')
            calfr = self.human_model.GetLink('green_kevin/calf_right_link')
            ub = self.human_model.GetLink('green_kevin/upper_body_link')
            origin_B_head = np.matrix(headmodel.GetTransform())
            origin_B_ual = np.matrix(ual.GetTransform())
            # print 'ual', origin_B_ual
            origin_B_uar = np.matrix(uar.GetTransform())
            # print 'uar', origin_B_uar
            origin_B_fal = np.matrix(fal.GetTransform())
            # print 'fal', origin_B_fal
            origin_B_far = np.matrix(far.GetTransform())
            # print 'far', origin_B_far
            origin_B_hl = np.matrix(hl.GetTransform())
            # print 'hl', origin_B_hl
            origin_B_hr = np.matrix(hr.GetTransform())
            # print 'hr', origin_B_hr
            origin_B_thl = np.matrix(thl.GetTransform())
            origin_B_thr = np.matrix(thr.GetTransform())
            origin_B_calfl = np.matrix(calfl.GetTransform())
            origin_B_calfr = np.matrix(calfr.GetTransform())
            origin_B_ub = np.matrix(ub.GetTransform())
            self.selection_mat = np.zeros(len(self.goals))
            self.goal_list = np.zeros([len(self.goals), 4, 4])
            self.origin_B_references = []
            for thing in xrange(len(self.reference_names)):
                if self.reference_names[thing] == 'head':
                    self.origin_B_references[thing] = origin_B_head
                elif self.reference_names[thing] == 'base_link':
                    self.origin_B_references[thing] = origin_B_pr2
                    # self.origin_B_references[thing] = np.matrix(self.robot.GetTransform())
                elif self.reference_names[thing] == 'upper_arm_left':
                    self.origin_B_references.append(origin_B_ual)
                elif self.reference_names[thing] == 'upper_arm_right':
                    self.origin_B_references.append(origin_B_uar)
                elif self.reference_names[thing] == 'forearm_left':
                    self.origin_B_references.append(origin_B_fal)
                elif self.reference_names[thing] == 'forearm_right':
                    self.origin_B_references.append(origin_B_far)
                elif self.reference_names[thing] == 'hand_left':
                    self.origin_B_references.append(origin_B_hl)
                elif self.reference_names[thing] == 'hand_right':
                    self.origin_B_references.append(origin_B_hr)
                elif self.reference_names[thing] == 'thigh_left':
                    self.origin_B_references.append(origin_B_thl)
                elif self.reference_names[thing] == 'thigh_right':
                    self.origin_B_references.append(origin_B_thr)
                elif self.reference_names[thing] == 'knee_left':
                    self.origin_B_references.append(origin_B_calfl)
                elif self.reference_names[thing] == 'knee_right':
                    self.origin_B_references.append(origin_B_calfr)
                elif self.reference_names[thing] == 'upper_body':
                    self.origin_B_references.append(origin_B_ub)
                else:
                    print 'I dont know the refence name'
            for thing in xrange(len(self.goals)):
                temp = self.origin_B_references[int(self.reference_mat[thing])]

                if 'body' in self.reference_names[int(self.reference_mat[thing])]:
                    z_origin = np.array([0., 0., 1.])
                    x_vector = np.array([temp[0, 1], temp[1, 1], temp[2, 1]])
                    y_orth = np.cross(z_origin, x_vector)
                    y_orth = y_orth/np.linalg.norm(y_orth)
                    z_orth = np.cross(x_vector, y_orth)
                    z_orth = z_orth/np.linalg.norm(z_orth)
                    temp[0:3, 0] = np.reshape(x_vector, [3, 1])
                    temp[0:3, 1] = np.reshape(y_orth, [3, 1])
                    temp[0:3, 2] = np.reshape(z_orth, [3, 1])
                else:
                    z_origin = np.array([0., 0., 1.])
                    x_vector = np.array([temp[0, 0], temp[1, 0], temp[2, 0]])
                    y_orth = np.cross(z_origin, x_vector)
                    y_orth = y_orth/np.linalg.norm(y_orth)
                    z_orth = np.cross(x_vector, y_orth)
                    z_orth = z_orth/np.linalg.norm(z_orth)
                    temp[0:3, 1] = np.reshape(y_orth, [3, 1])
                    temp[0:3, 2] = np.reshape(z_orth, [3, 1])
                    # print self.origin_B_references[int(self.reference_mat[thing])]
                    # temp_B = self.origin_B_references[int(self.reference_mat[thing])]*\
                temp_B = temp * np.matrix(self.goals[thing, 0])
                temp_B[2, 3] += self.optimal_z_offset
                self.goal_list[thing] = np.matrix(copy.copy(temp_B))*self.gripper_B_tool.I
                self.selection_mat[thing] = self.goals[thing, 1]
#            for target in self.goals:
#                self.goal_list.append(pr2_B_head*np.matrix(target[0]))
#                self.selection_mat.append(target[1])
            self.set_goals()
        else:
            print 'I GOT A BAD MODEL. NOT SURE WHAT TO DO NOW!'
        distance = 10000000.
        out_of_reach = True

        for origin_B_grasp in self.origin_B_grasps:
            pr2_B_goal = origin_B_pr2.I*origin_B_grasp
            distance = np.min([np.linalg.norm(pr2_B_goal[:2, 3]), distance])

            if distance <= 1.25:
                out_of_reach = False
                # print 'not out of reach'
                break
        if out_of_reach:
            # print 'location is out of reach'
            return 10. +1.+ 20.*(distance - 1.25)

        #print 'Time to update autobed things: %fs'%(time.time()-starttime)
        reach_score = 0.
        manip_score = 0.
        goal_scores = []
        # std = 1.
        # mean = 0.
        # allmanip = []
        manip = 0.
        reached = 0.

        #allmanip2=[]
        # space_score = (1./(std*(m.pow((2.*m.pi), 0.5))))*m.exp(-(m.pow(np.linalg.norm([x, y])-mean, 2.)) /
        #                                                        (2.*m.pow(std, 2.)))
        #print space_score
        with self.robot:
            sign_flip = 1.
            if 'right' in self.arm:
                sign_flip = -1.
            v = self.robot.GetActiveDOFValues()
            v[self.robot.GetJoint(self.opposite_arm[0]+'_shoulder_pan_joint').GetDOFIndex()] = -sign_flip*3.14/2
            v[self.robot.GetJoint(self.opposite_arm[0]+'_shoulder_lift_joint').GetDOFIndex()] = -0.52
            v[self.robot.GetJoint(self.opposite_arm[0]+'_upper_arm_roll_joint').GetDOFIndex()] = 0.
            v[self.robot.GetJoint(self.opposite_arm[0]+'_elbow_flex_joint').GetDOFIndex()] = -3.14*2/3
            v[self.robot.GetJoint(self.opposite_arm[0]+'_forearm_roll_joint').GetDOFIndex()] = 0.
            v[self.robot.GetJoint(self.opposite_arm[0]+'_wrist_flex_joint').GetDOFIndex()] = 0.
            v[self.robot.GetJoint(self.opposite_arm[0]+'_wrist_roll_joint').GetDOFIndex()] = 0.
            self.robot.SetActiveDOFValues(v, 2)
            self.env.UpdatePublishedBodies()
            not_close_to_collision = True
            if self.manip.CheckIndependentCollision(op.CollisionReport()):
                not_close_to_collision = False
            check_if_base_is_near_collision = False
            if check_if_base_is_near_collision:
                origin_B_pr2 = np.matrix([[ m.cos(th), -m.sin(th),     0., x+.04],
                                          [ m.sin(th),  m.cos(th),     0., y+.04],
                                          [        0.,         0.,     1.,        0.],
                                          [        0.,         0.,     0.,        1.]])
                self.robot.SetTransform(np.array(origin_B_pr2))
                self.env.UpdatePublishedBodies()
                if self.manip.CheckIndependentCollision(op.CollisionReport()):
                    not_close_to_collision = False

                origin_B_pr2 = np.matrix([[ m.cos(th), -m.sin(th),     0., x-.04],
                                          [ m.sin(th),  m.cos(th),     0., y+.04],
                                          [        0.,         0.,     1.,        0.],
                                          [        0.,         0.,     0.,        1.]])
                self.robot.SetTransform(np.array(origin_B_pr2))
                self.env.UpdatePublishedBodies()
                if self.manip.CheckIndependentCollision(op.CollisionReport()):
                    not_close_to_collision = False

                origin_B_pr2 = np.matrix([[ m.cos(th), -m.sin(th),     0., x-.04],
                                          [ m.sin(th),  m.cos(th),     0., y-.04],
                                          [        0.,         0.,     1.,        0.],
                                          [        0.,         0.,     0.,        1.]])
                self.robot.SetTransform(np.array(origin_B_pr2))
                self.env.UpdatePublishedBodies()
                if self.manip.CheckIndependentCollision(op.CollisionReport()):
                    not_close_to_collision = False

                origin_B_pr2 = np.matrix([[ m.cos(th), -m.sin(th),     0., x+.04],
                                          [ m.sin(th),  m.cos(th),     0., y-.04],
                                          [        0.,         0.,     1.,        0.],
                                          [        0.,         0.,     0.,        1.]])
                self.robot.SetTransform(np.array(origin_B_pr2))
                self.env.UpdatePublishedBodies()
                if self.manip.CheckIndependentCollision(op.CollisionReport()):
                    not_close_to_collision = False

                origin_B_pr2 = np.matrix([[ m.cos(th), -m.sin(th),     0., x],
                                          [ m.sin(th),  m.cos(th),     0., y],
                                          [        0.,         0.,     1.,        0.],
                                          [        0.,         0.,     0.,        1.]])
                self.robot.SetTransform(np.array(origin_B_pr2))
                self.env.UpdatePublishedBodies()

            if not_close_to_collision:
                reached = np.zeros(len(self.origin_B_grasps))
                manip = np.zeros(len(self.origin_B_grasps))
                for num, Tgrasp in enumerate(self.origin_B_grasps):
                    sols = []
                    sols = self.manip.FindIKSolutions(Tgrasp, filteroptions=op.IkFilterOptions.CheckEnvCollisions)
                    if not list(sols):
                        sign_flip = 1.
                        if 'right' in self.arm:
                            sign_flip = -1.
                        v = self.robot.GetActiveDOFValues()
                        v[self.robot.GetJoint(self.opposite_arm[0]+'_shoulder_pan_joint').GetDOFIndex()] = -sign_flip*0.023593
                        v[self.robot.GetJoint(self.opposite_arm[0]+'_shoulder_lift_joint').GetDOFIndex()] = 1.1072800
                        v[self.robot.GetJoint(self.opposite_arm[0]+'_upper_arm_roll_joint').GetDOFIndex()] = -sign_flip*1.5566882
                        v[self.robot.GetJoint(self.opposite_arm[0]+'_elbow_flex_joint').GetDOFIndex()] = -2.124408
                        v[self.robot.GetJoint(self.opposite_arm[0]+'_forearm_roll_joint').GetDOFIndex()] = -sign_flip*1.4175
                        v[self.robot.GetJoint(self.opposite_arm[0]+'_wrist_flex_joint').GetDOFIndex()] = -1.8417
                        v[self.robot.GetJoint(self.opposite_arm[0]+'_wrist_roll_joint').GetDOFIndex()] = 0.21436
                        self.robot.SetActiveDOFValues(v, 2)
                        self.env.UpdatePublishedBodies()
                        sols = self.manip.FindIKSolutions(Tgrasp, filteroptions=op.IkFilterOptions.CheckEnvCollisions)

                    if list(sols):  # not None:
                        reached[num] = 1.
                        for solution in sols:
                            self.robot.SetDOFValues(solution, self.manip.GetArmIndices())
                            self.env.UpdatePublishedBodies()
                            if self.visualize:
                                rospy.sleep(0.5)
                            J = np.matrix(np.vstack([self.manip.CalculateJacobian(), self.manip.CalculateAngularVelocityJacobian()]))
                            try:
                                joint_limit_weight = self.gen_joint_limit_weight(solution, self.arm)
                                manip[num] = np.max([copy.copy((m.pow(np.linalg.det(J*joint_limit_weight*J.T), (1./6.)))/(np.trace(J*joint_limit_weight*J.T)/6.)), manip[num]])
                            except ValueError:
                                print 'WARNING!!'
                                print 'Jacobian may be singular or close to singular'
                                print 'Determinant of J*JT is: ', np.linalg.det(J*J.T)
                                manip[num] = np.max([0., manip[num]])
                for num in xrange(len(reached)):
                    manip_score += copy.copy(reached[num] * manip[num]*self.weights[num])
                    reach_score += copy.copy(reached[num] * self.weights[num])
            else:
                # print 'In base collision! single config distance: ', distance
                if distance < 2.0:
                    return 10. + 1. + (1.25 - distance)

        # self.human_model.SetActiveManipulator('leftarm')
        # self.human_manip = self.robot.GetActiveManipulator()
        # human_torques = self.human_manip.ComputeInverseDynamics([])
        # torque_cost = np.linalg.norm(human_torques)/10.

        angle_cost = np.sum(np.abs(human_dof))


        # Set the weights for the different scores.
        beta = 10.  # Weight on number of reachable goals
        gamma = 1.  # Weight on manipulability of arm at each reachable goal
        zeta = 0.05  # Weight on torques
        if reach_score == 0.:
            return 10. + 2*random.random()
        else:
            # print 'Reach score: ', reach_score
            # print 'Manip score: ', manip_score
            return 10.-beta*reach_score-gamma*manip_score + zeta*angle_cost

    def objective_function_two_configs(self, current_parameters):
        if not self.a_model_is_loaded:
            print 'Somehow a model has not been loaded. This is bad!'
            return None
        # print current_parameters
        # print len(current_parameters)
        # print 'head rest angle: ', self.head_rest_angle
        if len(current_parameters) == 12:
            x = [current_parameters[0], current_parameters[6]]
            y = [current_parameters[1], current_parameters[7]]
            th = [current_parameters[2], current_parameters[8]]
            z = [current_parameters[3], current_parameters[9]]
            bz = [current_parameters[4], current_parameters[10]]
            bth = [current_parameters[5], current_parameters[11]]
        elif len(current_parameters) == 10:
            x = [current_parameters[0], current_parameters[5]]
            y = [current_parameters[1], current_parameters[6]]
            th = [current_parameters[2], current_parameters[7]]
            z = [current_parameters[3], current_parameters[8]]
            bz = [current_parameters[4], current_parameters[9]]
            bth = [np.radians(self.head_rest_angle), np.radians(self.head_rest_angle)]
        else:
            x = [current_parameters[0], current_parameters[4]]
            y = [current_parameters[1], current_parameters[5]]
            th = [current_parameters[2], current_parameters[6]]
            z = [current_parameters[3], current_parameters[7]]
            bz = [0., 0.]
            bth = [0., 0.]
        # print bth
        # print bz

        # planar_difference = np.linalg.norm([x[0]-x[1], y[0]-y[1]])
        # if planar_difference < 0.2:
        #     return 10 + 10*(0.2 - planar_difference)

        # Cost on distanced moved.
        # travel = [np.linalg.norm([self.start_x - x[0], self.start_y - y[0]]),
        #           np.linalg.norm([self.start_x - x[1], self.start_y - y[1]])]
        # travel.append(travel[0]+travel[1])

        # distance = 10000000.
        out_of_reach = False

        reach_score = np.array([0., 0., 0.])
        manip_score = np.array([0., 0., 0.])
        reached = np.zeros([len(self.goals), 3])
        manip = np.zeros([len(self.goals), 3])

        # better_config = np.array([-1]*len(self.goals))
        best = None
        for num in xrange(len(self.goals)):
            fully_collided = 0
            # manip = [0., 0., 0.]
            # reached = [0., 0., 0.]
            distance = [100000., 100000.]
            for config_num in xrange(len(x)):
                origin_B_pr2 = np.matrix([[ m.cos(th[config_num]), -m.sin(th[config_num]),     0., x[config_num]],
                                          [ m.sin(th[config_num]),  m.cos(th[config_num]),     0., y[config_num]],
                                          [        0.,         0.,     1.,        0.],
                                          [        0.,         0.,     0.,        1.]])
                self.robot.SetTransform(np.array(origin_B_pr2))
                v = self.robot.GetActiveDOFValues()
                v[self.robot.GetJoint('torso_lift_joint').GetDOFIndex()] = z[config_num]
                self.robot.SetActiveDOFValues(v, 2)
                # self.env.UpdatePublishedBodies()

                for head_angle in self.head_angles:
                    self.rotate_head_only(head_angle[0], head_angle[1])
                    if self.model == 'chair':
                        self.env.UpdatePublishedBodies()
                        headmodel = self.wheelchair.GetLink('wheelchair/head_link')
                        ual = self.wheelchair.GetLink('wheelchair/arm_left_link')
                        uar = self.wheelchair.GetLink('wheelchair/arm_right_link')
                        fal = self.wheelchair.GetLink('wheelchair/forearm_left_link')
                        far = self.wheelchair.GetLink('wheelchair/forearm_right_link')
                        thl = self.wheelchair.GetLink('wheelchair/quad_left_link')
                        thr = self.wheelchair.GetLink('wheelchair/quad_right_link')
                        calfl = self.wheelchair.GetLink('wheelchair/calf_left_link')
                        calfr = self.wheelchair.GetLink('wheelchair/calf_right_link')
                        ub = self.wheelchair.GetLink('wheelchair/upper_body_link')
                        origin_B_head = np.matrix(headmodel.GetTransform())
                        origin_B_ual = np.matrix(ual.GetTransform())
                        origin_B_uar = np.matrix(uar.GetTransform())
                        origin_B_fal = np.matrix(fal.GetTransform())
                        origin_B_far = np.matrix(far.GetTransform())
                        origin_B_thl = np.matrix(thl.GetTransform())
                        origin_B_thr = np.matrix(thr.GetTransform())
                        origin_B_calfl = np.matrix(calfl.GetTransform())
                        origin_B_calfr = np.matrix(calfr.GetTransform())
                        origin_B_ub = np.matrix(ub.GetTransform())
                        self.selection_mat = np.zeros(len(self.goals))
                        self.goal_list = np.zeros([len(self.goals), 4, 4])
                        for thing in xrange(len(self.reference_names)):
                            if self.reference_names[thing] == 'head':
                                self.origin_B_references[thing] = origin_B_head
                            elif self.reference_names[thing] == 'base_link':
                                self.origin_B_references[thing] = origin_B_pr2
                                # self.origin_B_references[thing] = np.matrix(self.robot.GetTransform())
                            elif self.reference_names[thing] == 'upper_arm_left':
                                self.origin_B_references.append(origin_B_ual)
                            elif self.reference_names[thing] == 'upper_arm_right':
                                self.origin_B_references.append(origin_B_uar)
                            elif self.reference_names[thing] == 'forearm_left':
                                self.origin_B_references.append(origin_B_fal)
                            elif self.reference_names[thing] == 'forearm_right':
                                self.origin_B_references.append(origin_B_far)
                            elif self.reference_names[thing] == 'thigh_left':
                                self.origin_B_references.append(origin_B_thl)
                            elif self.reference_names[thing] == 'thigh_right':
                                self.origin_B_references.append(origin_B_thr)
                            elif self.reference_names[thing] == 'knee_left':
                                self.origin_B_references.append(origin_B_calfl)
                            elif self.reference_names[thing] == 'knee_right':
                                self.origin_B_references.append(origin_B_calfr)
                            elif self.reference_names[thing] == 'upper_body':
                                self.origin_B_references.append(origin_B_ub)

                        thing = num
                        self.goal_list[0] = copy.copy(self.origin_B_references[int(self.reference_mat[thing])]*np.matrix(self.goals[thing, 0]))
                        self.selection_mat[0] = copy.copy(self.goals[thing, 1])

                        self.set_goals()
                        headmodel = self.wheelchair.GetLink('wheelchair/head_link')

                    elif self.model == 'autobed':
                        self.selection_mat = np.zeros(1)
                        self.goal_list = np.zeros([1, 4, 4])
                        self.set_autobed(bz[config_num], bth[config_num], self.headx, self.heady)
                        self.env.UpdatePublishedBodies()

                        headmodel = self.autobed.GetLink('autobed/head_link')
                        ual = self.autobed.GetLink('autobed/arm_left_link')
                        uar = self.autobed.GetLink('autobed/arm_right_link')
                        fal = self.autobed.GetLink('autobed/forearm_left_link')
                        far = self.autobed.GetLink('autobed/forearm_right_link')
                        thl = self.autobed.GetLink('autobed/quad_left_link')
                        thr = self.autobed.GetLink('autobed/quad_right_link')
                        calfl = self.autobed.GetLink('autobed/calf_left_link')
                        calfr = self.autobed.GetLink('autobed/calf_right_link')
                        ub = self.autobed.GetLink('autobed/upper_body_link')
                        origin_B_head = np.matrix(headmodel.GetTransform())
                        origin_B_ual = np.matrix(ual.GetTransform())
                        origin_B_uar = np.matrix(uar.GetTransform())
                        origin_B_fal = np.matrix(fal.GetTransform())
                        origin_B_far = np.matrix(far.GetTransform())
                        origin_B_thl = np.matrix(thl.GetTransform())
                        origin_B_thr = np.matrix(thr.GetTransform())
                        origin_B_calfl = np.matrix(calfl.GetTransform())
                        origin_B_calfr = np.matrix(calfr.GetTransform())
                        origin_B_ub = np.matrix(ub.GetTransform())
                        self.origin_B_references = []
                        # for thing in xrange(len(self.reference_names)):
                        thing = int(self.goals[num, 2])
                        if self.reference_names[thing] == 'head':
                            self.origin_B_references.append(origin_B_head)
                        elif self.reference_names[thing] == 'base_link':
                            self.origin_B_references.append(origin_B_pr2)
                        elif self.reference_names[thing] == 'upper_arm_left':
                            self.origin_B_references.append(origin_B_ual)
                        elif self.reference_names[thing] == 'upper_arm_right':
                            self.origin_B_references.append(origin_B_uar)
                        elif self.reference_names[thing] == 'forearm_left':
                            self.origin_B_references.append(origin_B_fal)
                        elif self.reference_names[thing] == 'forearm_right':
                            self.origin_B_references.append(origin_B_far)
                        elif self.reference_names[thing] == 'thigh_left':
                            self.origin_B_references.append(origin_B_thl)
                        elif self.reference_names[thing] == 'thigh_right':
                            self.origin_B_references.append(origin_B_thr)
                        elif self.reference_names[thing] == 'knee_left':
                            self.origin_B_references.append(origin_B_calfl)
                        elif self.reference_names[thing] == 'knee_right':
                            self.origin_B_references.append(origin_B_calfr)
                        elif self.reference_names[thing] == 'upper_body':
                            self.origin_B_references.append(origin_B_ub)
                        else:
                            print 'The refence options is bogus! I dont know what to do!'
                            return

                        # for thing in xrange(len(self.goals)):
                        # thing = num
                        self.goal_list[0] = copy.copy(self.origin_B_references[0]*np.matrix(self.goals[num, 0]))
                        self.selection_mat[0] = copy.copy(self.goals[num, 1])
                        self.set_goals(single_goal=True)
                    else:
                        print 'I GOT A BAD MODEL. NOT SURE WHAT TO DO NOW!'

                    # for origin_B_goal in self.origin_B_grasps:
                    origin_B_grasp = self.origin_B_grasps[0]
                    pr2_B_goal = origin_B_pr2.I*origin_B_grasp
                    this_distance = np.linalg.norm(pr2_B_goal[:2, 3])
                    distance[config_num] = np.min([this_distance, distance[config_num]])
                    if this_distance < 1.25:
                        with self.robot:
                            sign_flip = 1.
                            if 'right' in self.arm:
                                sign_flip = -1.
                            v = self.robot.GetActiveDOFValues()
                            v[self.robot.GetJoint(self.opposite_arm[0]+'_shoulder_pan_joint').GetDOFIndex()] = -sign_flip*3.14/2
                            v[self.robot.GetJoint(self.opposite_arm[0]+'_shoulder_lift_joint').GetDOFIndex()] = -0.52
                            v[self.robot.GetJoint(self.opposite_arm[0]+'_upper_arm_roll_joint').GetDOFIndex()] = 0.
                            v[self.robot.GetJoint(self.opposite_arm[0]+'_elbow_flex_joint').GetDOFIndex()] = -3.14*2/3
                            v[self.robot.GetJoint(self.opposite_arm[0]+'_forearm_roll_joint').GetDOFIndex()] = 0.
                            v[self.robot.GetJoint(self.opposite_arm[0]+'_wrist_flex_joint').GetDOFIndex()] = 0.
                            v[self.robot.GetJoint(self.opposite_arm[0]+'_wrist_roll_joint').GetDOFIndex()] = 0.
                            self.robot.SetActiveDOFValues(v,2)
                            self.env.UpdatePublishedBodies()
                            not_close_to_collision = True
                            if self.manip.CheckIndependentCollision(op.CollisionReport()):
                                not_close_to_collision = False

                            origin_B_pr2 = np.matrix([[ m.cos(th[config_num]), -m.sin(th[config_num]),     0., x[config_num]+.04],
                                              [ m.sin(th[config_num]),  m.cos(th[config_num]),     0., y[config_num]+.04],
                                              [        0.,         0.,     1.,        0.],
                                              [        0.,         0.,     0.,        1.]])
                            self.robot.SetTransform(np.array(origin_B_pr2))
                            self.env.UpdatePublishedBodies()
                            if self.manip.CheckIndependentCollision(op.CollisionReport()):
                                not_close_to_collision = False

                            origin_B_pr2 = np.matrix([[ m.cos(th[config_num]), -m.sin(th[config_num]),     0., x[config_num]-.04],
                                              [ m.sin(th[config_num]),  m.cos(th[config_num]),     0., y[config_num]+.04],
                                              [        0.,         0.,     1.,        0.],
                                              [        0.,         0.,     0.,        1.]])
                            self.robot.SetTransform(np.array(origin_B_pr2))
                            self.env.UpdatePublishedBodies()
                            if self.manip.CheckIndependentCollision(op.CollisionReport()):
                                not_close_to_collision = False

                            origin_B_pr2 = np.matrix([[ m.cos(th[config_num]), -m.sin(th[config_num]),     0., x[config_num]-.04],
                                              [ m.sin(th[config_num]),  m.cos(th[config_num]),     0., y[config_num]-.04],
                                              [        0.,         0.,     1.,        0.],
                                              [        0.,         0.,     0.,        1.]])
                            self.robot.SetTransform(np.array(origin_B_pr2))
                            self.env.UpdatePublishedBodies()
                            if self.manip.CheckIndependentCollision(op.CollisionReport()):
                                not_close_to_collision = False

                            origin_B_pr2 = np.matrix([[ m.cos(th[config_num]), -m.sin(th[config_num]),     0., x[config_num]+.04],
                                              [ m.sin(th[config_num]),  m.cos(th[config_num]),     0., y[config_num]-.04],
                                              [        0.,         0.,     1.,        0.],
                                              [        0.,         0.,     0.,        1.]])
                            self.robot.SetTransform(np.array(origin_B_pr2))
                            self.env.UpdatePublishedBodies()
                            if self.manip.CheckIndependentCollision(op.CollisionReport()):
                                not_close_to_collision = False

                            origin_B_pr2 = np.matrix([[ m.cos(th[config_num]), -m.sin(th[config_num]),     0., x[config_num]],
                                              [ m.sin(th[config_num]),  m.cos(th[config_num]),     0., y[config_num]],
                                              [        0.,         0.,     1.,        0.],
                                              [        0.,         0.,     0.,        1.]])
                            self.robot.SetTransform(np.array(origin_B_pr2))
                            self.env.UpdatePublishedBodies()

                            if not_close_to_collision:
                                Tgrasp = self.origin_B_grasps[0]

                                # print 'no collision!'
                                # for num, Tgrasp in enumerate(self.origin_B_grasps):
                                    # sol = None
                                    # sol = self.manip.FindIKSolution(Tgrasp, filteroptions=op.IkFilterOptions.CheckEnvCollisions)

                                    #sol = self.manip.FindIKSolution(Tgrasp,filteroptions=op.IkFilterOptions.IgnoreSelfCollisions)
                                sols = []
                                sols = self.manip.FindIKSolutions(Tgrasp, filteroptions=op.IkFilterOptions.CheckEnvCollisions)
                                if not list(sols):
                                    sign_flip = 1.
                                    if 'right' in self.arm:
                                        sign_flip = -1.
                                    v[self.robot.GetJoint(self.opposite_arm[0]+'_shoulder_pan_joint').GetDOFIndex()] = -sign_flip*0.023593
                                    v[self.robot.GetJoint(self.opposite_arm[0]+'_shoulder_lift_joint').GetDOFIndex()] = 1.1072800
                                    v[self.robot.GetJoint(self.opposite_arm[0]+'_upper_arm_roll_joint').GetDOFIndex()] = -sign_flip*1.5566882
                                    v[self.robot.GetJoint(self.opposite_arm[0]+'_elbow_flex_joint').GetDOFIndex()] = -2.124408
                                    v[self.robot.GetJoint(self.opposite_arm[0]+'_forearm_roll_joint').GetDOFIndex()] = -sign_flip*1.4175
                                    v[self.robot.GetJoint(self.opposite_arm[0]+'_wrist_flex_joint').GetDOFIndex()] = -1.8417
                                    v[self.robot.GetJoint(self.opposite_arm[0]+'_wrist_roll_joint').GetDOFIndex()] = 0.21436
                                    self.robot.SetActiveDOFValues(v, 2)
                                    self.env.UpdatePublishedBodies()
                                    sols = self.manip.FindIKSolutions(Tgrasp, filteroptions=op.IkFilterOptions.CheckEnvCollisions)

                                if list(sols):  # not None:
                                    # print 'I got a solution!!'
                                    # print 'sol is:', sol
                                    # print 'sols are: \n', sols
                                    #print 'I was able to find a grasp to this goal'
                                    reached[num, config_num] = 1
                                    for solution in sols:
                                        self.robot.SetDOFValues(solution, self.manip.GetArmIndices())
                                        # Tee = self.manip.GetEndEffectorTransform()
                                        self.env.UpdatePublishedBodies()
                                        if self.visualize:
                                            rospy.sleep(0.2)

                                        J = np.matrix(np.vstack([self.manip.CalculateJacobian(), self.manip.CalculateAngularVelocityJacobian()]))
                                        try:
                                            joint_limit_weight = self.gen_joint_limit_weight(solution)
                                            manip[num, config_num] = np.max([copy.copy((m.pow(np.linalg.det(J*joint_limit_weight*J.T), (1./6.)))/(np.trace(J*joint_limit_weight*J.T)/6.)), manip[num, config_num]])
                                        except ValueError:
                                            print 'WARNING!!'
                                            print 'Jacobian may be singular or close to singular'
                                            print 'Determinant of J*JT is: ', np.linalg.det(J*J.T)
                                            manip[num, config_num] = np.max([0., manip[num, config_num]])
                            else:
                                # print 'Too close, robot base in collision with bed'
                                # print 10 + 1.25 - distance
                                fully_collided += 1
                                if this_distance > 3:
                                    print 'This shouldnt be possible. Distance is: ', this_distance
                                # if this_distance < 0.5:
                                #     return 10 + 2. - this_distance
                                # else:
                                #     return 10 + 2*random.random()
                if fully_collided == 2 and np.min(distance) < 2.:
                    return 10. + 1. + (1.25 - np.min(distance))
            reached[num, 2] = np.max(reached[num])
            manip[num, 2] = np.max(manip[num])
        # if np.sum(reached[:, 2]) > np.sum(reached[num, 0]) + 0.00001 and np.sum(reached[:, 2]) > np.sum(reached[num, 1]) + 0.00001:
        #     best = 2
        # elif np.sum(reached[num, 0]) > np.sum(reached[num, 1]) + 0.00001:
        #     best = 0
        # elif np.sum(reached[num, 1]) > np.sum(reached[num, 0]) + 0.00001:
        #     best = 1
        # elif np.sum(manip[:, 2])*0.95 > np.sum(manip[num, 0]) + 0.00001 and np.sum(manip[:, 2])*0.95 > np.sum(manip[num, 1]) + 0.00001:
        #     best = 2
        # elif np.sum(manip[num, 0]) > np.sum(manip[num, 1]) + 0.00001:
        #     best = 0
        # else:  # if np.sum(manip[num, 1]) > np.sum(manip[num, 0]) + 0.00001:
        #     best = 1

            # if manip[0] >= manip[1] and manip[0] > 0.:
            #     better_config[num] = 0
            # elif manip[0] < manip[1] and manip[1] > 0:
            #     better_config[num] = 1
        # print 'Manip score: ', manip_score
        # print 'Reach score: ', reach_score
        # print 'Distance: ', distance

        over_dist = 0.
        for dist in distance:
            if dist >= 1.25:
                over_dist += 2*(dist - 1.25)
        if over_dist > 0.001:
            return 10 + 1 + 10*over_dist

        reach_score[0] = np.sum(reached[:, 0] * self.weights[0])
        reach_score[1] = np.sum(reached[:, 1] * self.weights[0])
        reach_score[2] = np.sum(reached[:, 2] * self.weights[0])

        if np.max(reach_score) == 0:
            return 10 + 2*random.random()

        manip_score[0] = np.sum(reached[:, 0]*manip[:, 0]*self.weights[0])
        manip_score[1] = np.sum(reached[:, 1]*manip[:, 1]*self.weights[0])
        manip_score[2] = np.sum(reached[:, 2]*manip[:, 2]*self.weights[0])*0.95



        # if reach_score == 0:
        #     if np.min(distance) >= 0.8:
        #         # output =
        #         # print output
        #         return 10 + 1 + 2*(np.min(distance) - 0.8)
        # if 0 not in better_config or 1 not in better_config:
        #     return 10. + 2*random.random()
        # else:
        ## Set the weights for the different scores.
        beta = 10.  # Weight on number of reachable goals
        gamma = 1.  # Weight on manipulability of arm at each reachable goal
        zeta = .05  # Weight on distance to move to get to that goal location
        # thisScore =
        # print 'Reach score: ', reach_score
        # print 'Manip score: ', manip_score
        # print 'Calculated score: ', 10.-beta*reach_score-gamma*manip_score
        best = None


        # Travel cost
        # travel_score = np.array([0., 0., 0.])
        # travel_score[0] = np.min([travel[0], 2.0])
        # travel_score[1] = np.min([travel[1], 2.0])
        # travel_score[2] = np.min([travel[2], 2.0])



        # 1. - m.pow(1.0, np.abs(2.0 - travel[0]))
        # print 'reach score', reach_score
        # print 'manip score', manip_score
        # print 'travel score', travel_score

        outputs = 10. - beta*reach_score - gamma*manip_score #+ zeta*travel_score
        # print 'outputs', outputs
        best = np.argmin(outputs)
        if np.min(outputs) < -1.0:
            print 'reach score', reach_score
            print 'manip score', manip_score
            #print 'travel score', travel_score
            print outputs
        if outputs[best] < -1.0:
            print 'reach score', reach_score
            print 'manip score', manip_score
            #print 'travel score', travel_score
            print outputs
        return outputs[best]
        # return 10.-beta*reach_score-gamma*manip_score  # +zeta*self.distance

    def score_two_configs(self, config):
        if self.visualize:
            self.env.SetViewer('qtcoin')
            rospy.sleep(5)
        x = config[0]
        y = config[1]
        th = config[2]
        z = config[3]
        bz = config[4]
        bth = config[5]
        reach_score = []
        manip_score = []

        # for num in xrange(len(x)+1):
        #     reach_score.append(0.)
        #     manip_score.append(0.)

        # Cost on distance traveled
        # travel = [np.linalg.norm([self.start_x - x[0], self.start_y - y[0]]),
        #           np.linalg.norm([self.start_x - x[1], self.start_y - y[1]])]
        # travel.append(travel[0]+travel[1])

        reach_score = np.array([0., 0., 0.])
        manip_score = np.array([0., 0., 0.])
        reached = np.zeros([len(self.goals), 3])
        manip = np.zeros([len(self.goals), 3])

        for num in xrange(len(self.goals)):
            fully_collided = 0
            # manip = [0., 0., 0.]
            # reached = [0., 0., 0.]
            distance = [100000., 100000.]
            for config_num in xrange(len(x)):
                origin_B_pr2 = np.matrix([[ m.cos(th[config_num]), -m.sin(th[config_num]),     0., x[config_num]],
                                          [ m.sin(th[config_num]),  m.cos(th[config_num]),     0., y[config_num]],
                                          [        0.,         0.,     1.,        0.],
                                          [        0.,         0.,     0.,        1.]])
                self.robot.SetTransform(np.array(origin_B_pr2))
                v = self.robot.GetActiveDOFValues()
                v[self.robot.GetJoint('torso_lift_joint').GetDOFIndex()] = z[config_num]
                self.robot.SetActiveDOFValues(v, 2)
                # self.env.UpdatePublishedBodies()

                for head_angle in self.head_angles:

                    self.rotate_head_only(head_angle[0], head_angle[1])

                    if self.model == 'chair':
                        self.env.UpdatePublishedBodies()
                        headmodel = self.wheelchair.GetLink('wheelchair/head_link')
                        ual = self.wheelchair.GetLink('wheelchair/arm_left_link')
                        uar = self.wheelchair.GetLink('wheelchair/arm_right_link')
                        fal = self.wheelchair.GetLink('wheelchair/forearm_left_link')
                        far = self.wheelchair.GetLink('wheelchair/forearm_right_link')
                        thl = self.wheelchair.GetLink('wheelchair/quad_left_link')
                        thr = self.wheelchair.GetLink('wheelchair/quad_right_link')
                        calfl = self.wheelchair.GetLink('wheelchair/calf_left_link')
                        calfr = self.wheelchair.GetLink('wheelchair/calf_right_link')
                        ub = self.wheelchair.GetLink('wheelchair/upper_body_link')
                        origin_B_head = np.matrix(headmodel.GetTransform())
                        origin_B_ual = np.matrix(ual.GetTransform())
                        origin_B_uar = np.matrix(uar.GetTransform())
                        origin_B_fal = np.matrix(fal.GetTransform())
                        origin_B_far = np.matrix(far.GetTransform())
                        origin_B_thl = np.matrix(thl.GetTransform())
                        origin_B_thr = np.matrix(thr.GetTransform())
                        origin_B_calfl = np.matrix(calfl.GetTransform())
                        origin_B_calfr = np.matrix(calfr.GetTransform())
                        origin_B_ub = np.matrix(ub.GetTransform())
                        self.selection_mat = np.zeros(len(self.goals))
                        self.goal_list = np.zeros([len(self.goals), 4, 4])
                        for thing in xrange(len(self.reference_names)):
                            if self.reference_names[thing] == 'head':
                                self.origin_B_references[thing] = origin_B_head
                            elif self.reference_names[thing] == 'base_link':
                                self.origin_B_references[thing] = origin_B_pr2
                                # self.origin_B_references[thing] = np.matrix(self.robot.GetTransform())
                            elif self.reference_names[thing] == 'upper_arm_left':
                                self.origin_B_references.append(origin_B_ual)
                            elif self.reference_names[thing] == 'upper_arm_right':
                                self.origin_B_references.append(origin_B_uar)
                            elif self.reference_names[thing] == 'forearm_left':
                                self.origin_B_references.append(origin_B_fal)
                            elif self.reference_names[thing] == 'forearm_right':
                                self.origin_B_references.append(origin_B_far)
                            elif self.reference_names[thing] == 'thigh_left':
                                self.origin_B_references.append(origin_B_thl)
                            elif self.reference_names[thing] == 'thigh_right':
                                self.origin_B_references.append(origin_B_thr)
                            elif self.reference_names[thing] == 'knee_left':
                                self.origin_B_references.append(origin_B_calfl)
                            elif self.reference_names[thing] == 'knee_right':
                                self.origin_B_references.append(origin_B_calfr)
                            elif self.reference_names[thing] == 'upper_body':
                                self.origin_B_references.append(origin_B_ub)

                        thing = num
                        self.goal_list[0] = copy.copy(self.origin_B_references[int(self.reference_mat[thing])]*np.matrix(self.goals[thing, 0]))
                        self.selection_mat[0] = copy.copy(self.goals[thing, 1])

                        self.set_goals()
                        headmodel = self.wheelchair.GetLink('wheelchair/head_link')

                    elif self.model == 'autobed':
                        self.selection_mat = np.zeros(1)
                        self.goal_list = np.zeros([1, 4, 4])
                        self.set_autobed(bz[config_num], bth[config_num], self.headx, self.heady)
                        self.env.UpdatePublishedBodies()

                        headmodel = self.autobed.GetLink('autobed/head_link')
                        ual = self.autobed.GetLink('autobed/arm_left_link')
                        uar = self.autobed.GetLink('autobed/arm_right_link')
                        fal = self.autobed.GetLink('autobed/forearm_left_link')
                        far = self.autobed.GetLink('autobed/forearm_right_link')
                        thl = self.autobed.GetLink('autobed/quad_left_link')
                        thr = self.autobed.GetLink('autobed/quad_right_link')
                        calfl = self.autobed.GetLink('autobed/calf_left_link')
                        calfr = self.autobed.GetLink('autobed/calf_right_link')
                        ub = self.autobed.GetLink('autobed/upper_body_link')
                        origin_B_head = np.matrix(headmodel.GetTransform())
                        origin_B_ual = np.matrix(ual.GetTransform())
                        origin_B_uar = np.matrix(uar.GetTransform())
                        origin_B_fal = np.matrix(fal.GetTransform())
                        origin_B_far = np.matrix(far.GetTransform())
                        origin_B_thl = np.matrix(thl.GetTransform())
                        origin_B_thr = np.matrix(thr.GetTransform())
                        origin_B_calfl = np.matrix(calfl.GetTransform())
                        origin_B_calfr = np.matrix(calfr.GetTransform())
                        origin_B_ub = np.matrix(ub.GetTransform())
                        self.origin_B_references = []
                        # for thing in xrange(len(self.reference_names)):
                        thing = int(self.goals[num, 2])
                        if self.reference_names[thing] == 'head':
                            self.origin_B_references.append(origin_B_head)
                        elif self.reference_names[thing] == 'base_link':
                            self.origin_B_references.append(origin_B_pr2)
                        elif self.reference_names[thing] == 'upper_arm_left':
                            self.origin_B_references.append(origin_B_ual)
                        elif self.reference_names[thing] == 'upper_arm_right':
                            self.origin_B_references.append(origin_B_uar)
                        elif self.reference_names[thing] == 'forearm_left':
                            self.origin_B_references.append(origin_B_fal)
                        elif self.reference_names[thing] == 'forearm_right':
                            self.origin_B_references.append(origin_B_far)
                        elif self.reference_names[thing] == 'thigh_left':
                            self.origin_B_references.append(origin_B_thl)
                        elif self.reference_names[thing] == 'thigh_right':
                            self.origin_B_references.append(origin_B_thr)
                        elif self.reference_names[thing] == 'knee_left':
                            self.origin_B_references.append(origin_B_calfl)
                        elif self.reference_names[thing] == 'knee_right':
                            self.origin_B_references.append(origin_B_calfr)
                        elif self.reference_names[thing] == 'upper_body':
                            self.origin_B_references.append(origin_B_ub)
                        else:
                            print 'The refence options is bogus! I dont know what to do!'
                            return

                        # for thing in xrange(len(self.goals)):
                        # thing = num
                        self.goal_list[0] = copy.copy(self.origin_B_references[0]*np.matrix(self.goals[num, 0]))
                        self.selection_mat[0] = copy.copy(self.goals[num, 1])
                        self.set_goals(single_goal=True)
                    else:
                        print 'I GOT A BAD MODEL. NOT SURE WHAT TO DO NOW!'

                    # for origin_B_goal in self.origin_B_grasps:
                    origin_B_grasp = self.origin_B_grasps[0]
                    pr2_B_goal = origin_B_pr2.I*origin_B_grasp
                    this_distance = np.linalg.norm(pr2_B_goal[:2, 3])
                    distance[config_num] = np.min([this_distance, distance[config_num]])
                    if this_distance < 1.25:
                        with self.robot:
                            sign_flip = 1.
                            if 'right' in self.arm:
                                sign_flip = -1.
                            v = self.robot.GetActiveDOFValues()
                            v[self.robot.GetJoint(self.opposite_arm[0]+'_shoulder_pan_joint').GetDOFIndex()] = -sign_flip*3.14/2
                            v[self.robot.GetJoint(self.opposite_arm[0]+'_shoulder_lift_joint').GetDOFIndex()] = -0.52
                            v[self.robot.GetJoint(self.opposite_arm[0]+'_upper_arm_roll_joint').GetDOFIndex()] = 0.
                            v[self.robot.GetJoint(self.opposite_arm[0]+'_elbow_flex_joint').GetDOFIndex()] = -3.14*2/3
                            v[self.robot.GetJoint(self.opposite_arm[0]+'_forearm_roll_joint').GetDOFIndex()] = 0.
                            v[self.robot.GetJoint(self.opposite_arm[0]+'_wrist_flex_joint').GetDOFIndex()] = 0.
                            v[self.robot.GetJoint(self.opposite_arm[0]+'_wrist_roll_joint').GetDOFIndex()] = 0.
                            self.robot.SetActiveDOFValues(v, 2)
                            if not self.manip.CheckIndependentCollision(op.CollisionReport()):
                                Tgrasp = self.origin_B_grasps[0]

                                # print 'no collision!'
                                # for num, Tgrasp in enumerate(self.origin_B_grasps):
                                    # sol = None
                                    # sol = self.manip.FindIKSolution(Tgrasp, filteroptions=op.IkFilterOptions.CheckEnvCollisions)

                                    #sol = self.manip.FindIKSolution(Tgrasp,filteroptions=op.IkFilterOptions.IgnoreSelfCollisions)
                                sols = []
                                sols = self.manip.FindIKSolutions(Tgrasp, filteroptions=op.IkFilterOptions.CheckEnvCollisions)
                                if not list(sols):
                                    sign_flip = 1.
                                    if 'right' in self.arm:
                                        sign_flip = -1.
                                    v[self.robot.GetJoint(self.opposite_arm[0]+'_shoulder_pan_joint').GetDOFIndex()] = -sign_flip*0.023593
                                    v[self.robot.GetJoint(self.opposite_arm[0]+'_shoulder_lift_joint').GetDOFIndex()] = 1.1072800
                                    v[self.robot.GetJoint(self.opposite_arm[0]+'_upper_arm_roll_joint').GetDOFIndex()] = -sign_flip*1.5566882
                                    v[self.robot.GetJoint(self.opposite_arm[0]+'_elbow_flex_joint').GetDOFIndex()] = -2.124408
                                    v[self.robot.GetJoint(self.opposite_arm[0]+'_forearm_roll_joint').GetDOFIndex()] = -sign_flip*1.4175
                                    v[self.robot.GetJoint(self.opposite_arm[0]+'_wrist_flex_joint').GetDOFIndex()] = -1.8417
                                    v[self.robot.GetJoint(self.opposite_arm[0]+'_wrist_roll_joint').GetDOFIndex()] = 0.21436
                                    self.robot.SetActiveDOFValues(v, 2)
                                    sols = self.manip.FindIKSolutions(Tgrasp, filteroptions=op.IkFilterOptions.CheckEnvCollisions)

                                if list(sols):  # not None:
                                    # print 'I got a solution!!'
                                    # print 'sol is:', sol
                                    # print 'sols are: \n', sols
                                    #print 'I was able to find a grasp to this goal'
                                    reached[num, config_num] = 1.
                                    for solution in sols:
                                        self.robot.SetDOFValues(solution, self.manip.GetArmIndices())
                                        # Tee = self.manip.GetEndEffectorTransform()
                                        self.env.UpdatePublishedBodies()
                                        if self.visualize:
                                            rospy.sleep(0.2)


                                        J = np.matrix(np.vstack([self.manip.CalculateJacobian(), self.manip.CalculateAngularVelocityJacobian()]))
                                        try:
                                            joint_limit_weight = self.gen_joint_limit_weight(solution, self.arm)
                                            manip[num, config_num] = np.max([copy.copy((m.pow(np.linalg.det(J*joint_limit_weight*J.T), (1./6.)))/(np.trace(J*joint_limit_weight*J.T)/6.)), manip[num, config_num]])
                                        except ValueError:
                                            print 'WARNING!!'
                                            print 'Jacobian may be singular or close to singular'
                                            print 'Determinant of J*JT is: ', np.linalg.det(J*J.T)
                                            manip[num, config_num] = np.max([0., manip[num, config_num]])
                            # else:
                            #     return 0.
                                # rospy.sleep(5)
                                # print np.degrees(solution)

            reached[num, 2] = np.max(reached[num])
            manip[num, 2] = np.max(manip[num])
            # manip_score[0] += reached[0]*manip[0]*self.weights[0]
            # manip_score[1] += reached[1]*manip[1]*self.weights[0]
            # manip_score[2] += reached[2]*0.95*manip[2]*self.weights[0]
            #
            # # np.max(reached)*np.max([manip[0], manip[1], 0.95*manip[2]])*self.weights[0]
            # reach_score[0] += reached[0] * self.weights[0]
            # reach_score[1] += reached[1] * self.weights[0]
            # reach_score[2] += reached[2] * self.weights[0]

        reach_score[0] = np.sum(reached[:, 0] * self.weights[0])
        reach_score[1] = np.sum(reached[:, 1] * self.weights[0])
        reach_score[2] = np.sum(reached[:, 2] * self.weights[0])

        manip_score[0] = np.sum(reached[:, 0]*manip[:, 0]*self.weights[0])
        manip_score[1] = np.sum(reached[:, 1]*manip[:, 1]*self.weights[0])
        manip_score[2] = np.sum(reached[:, 2]*manip[:, 2]*self.weights[0])*0.95

        ## Set the weights for the different scores.
        beta = 10.  # Weight on number of reachable goals
        gamma = 1.  # Weight on manipulability of arm at each reachable goal
        zeta = .05  # Weight on distance to move to get to that goal location
        # thisScore =
        # print 'Reach score: ', reach_score
        # print 'Manip score: ', manip_score
        # print 'Calculated score: ', 10.-beta*reach_score-gamma*manip_score
        best = None

        # Distance travel cost
        # travel_score = np.array([0., 0., 0.])
        # travel_score[0] = np.min([travel[0], 2.0])
        # travel_score[1] = np.min([travel[1], 2.0])
        # travel_score[2] = np.min([travel[2], 2.0])


        # 1. - m.pow(1.0, np.abs(2.0 - travel[0]))

        outputs = 10. - beta*reach_score - gamma*manip_score #- zeta*travel_score
        # best = np.argmin(outputs)
        return outputs

        # print manip_score
        # print reach_score
        # ## Set the weights for the different scores.
        # beta = 5.  # Weight on number of reachable goals
        # gamma = 1.  # Weight on manipulability of arm at each reachable goal
        # zeta = .0007  # Weight on distance to move to get to that goal location
        # # thisScore =
        # # print 'Reach score: ', reach_score
        # # print 'Manip score: ', manip_score
        # # print 'Calculated score: ', 10.-beta*reach_score-gamma*manip_score
        # output = []
        # for sco in xrange(len(manip_score)):
        #     output.append(beta*reach_score[sco]+gamma*manip_score[sco])
        # return output  # +zeta*self.distance

    def setup_openrave(self):
        # Setup Openrave ENV
        InitOpenRAVELogging()
        self.env = op.Environment()

        # Lets you visualize openrave. Uncomment to see visualization. Does not work through ssh.
        if self.visualize:
            self.env.SetViewer('qtcoin')

        ## Set up robot state node to do Jacobians. This works, but is commented out because we can do it with openrave
        #  fine.
        #torso_frame = '/torso_lift_link'
        #inertial_frame = '/base_link'
        #end_effector_frame = '/l_gripper_tool_frame'
        #from pykdl_utils.kdl_kinematics import create_kdl_kin
        #self.kinematics = create_kdl_kin(torso_frame, end_effector_frame)

        ## Load OpenRave PR2 Model
        self.env.Load('robots/pr2-beta-static.zae')
        self.robot = self.env.GetRobots()[0]
        self.robot.CheckLimitsAction=2
        sign_flip = 1.
        if 'right' in self.arm:
            sign_flip = -1.
        v = self.robot.GetActiveDOFValues()
        v[self.robot.GetJoint(self.arm[0]+'_shoulder_pan_joint').GetDOFIndex()] = sign_flip*3.14/2
        v[self.robot.GetJoint(self.opposite_arm[0]+'_shoulder_pan_joint').GetDOFIndex()] = -sign_flip*3.14/2
        v[self.robot.GetJoint(self.opposite_arm[0]+'_shoulder_lift_joint').GetDOFIndex()] = -0.52
        v[self.robot.GetJoint(self.opposite_arm[0]+'_upper_arm_roll_joint').GetDOFIndex()] = 0.
        v[self.robot.GetJoint(self.opposite_arm[0]+'_elbow_flex_joint').GetDOFIndex()] = -3.14*2/3
        v[self.robot.GetJoint(self.opposite_arm[0]+'_forearm_roll_joint').GetDOFIndex()] = 0.
        v[self.robot.GetJoint(self.opposite_arm[0]+'_wrist_flex_joint').GetDOFIndex()] = 0.
        v[self.robot.GetJoint(self.opposite_arm[0]+'_wrist_roll_joint').GetDOFIndex()] = 0.
        v[self.robot.GetJoint(self.arm[0]+'_gripper_l_finger_joint').GetDOFIndex()] = .54
        v[self.robot.GetJoint('torso_lift_joint').GetDOFIndex()] = .3
        self.robot.SetActiveDOFValues(v, 2)
        robot_start = np.matrix([[1., 0., 0., 0.],
                                 [0., 1., 0., 0.],
                                 [0., 0., 1., 0.],
                                 [0., 0., 0., 1.]])
        self.robot.SetTransform(np.array(robot_start))

        ## Set robot manipulators, ik, planner
        self.robot.SetActiveManipulator('leftarm')
        self.manip = self.robot.GetActiveManipulator()
        ikmodel = op.databases.inversekinematics.InverseKinematicsModel(self.robot, iktype=op.IkParameterization.Type.Transform6D)
        if not ikmodel.load():
            print 'IK model not found for leftarm. Will now generate an IK model. This will take a while!'
            ikmodel.autogenerate()

        self.robot.SetActiveManipulator('rightarm')
        self.manip = self.robot.GetActiveManipulator()
        ikmodel = op.databases.inversekinematics.InverseKinematicsModel(self.robot, iktype=op.IkParameterization.Type.Transform6D)
        if not ikmodel.load():
            print 'IK model not found for rightarm. Will now generate an IK model. This will take a while!'
            ikmodel.autogenerate()

        # create the interface for basic manipulation programs
        self.manipprob = op.interfaces.BaseManipulation(self.robot)

        ## Find and load Wheelchair Model
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('hrl_base_selection')

        # Transform from the coordinate frame of the wc model in the back right bottom corner, to the head location on the floor
        if self.model == 'green_kevin':

            '''
            self.env.Load(''.join([pkg_path, '/collada/wheelchair_and_body_assembly.dae']))
            originsubject_B_headfloor = np.matrix([[m.cos(0.), -m.sin(0.),  0.,      0.], #.45 #.438
                                                   [m.sin(0.),  m.cos(0.),  0.,      0.], #0.34 #.42
                                                   [       0.,         0.,  1.,      0.],
                                                   [       0.,         0.,  0.,      1.]])
            '''
            # This is the new wheelchair model
            # self.env.Load(''.join([pkg_path, '/collada/wheelchair_and_body_assembly.dae']))
            # self.env.Load(''.join([pkg_path, '/collada/human_model_rounded.dae']))
            human_urdf_path = pkg_path + '/urdf/green_kevin/robots/green_kevin.urdf'
            human_srdf_path = pkg_path + '/urdf/green_kevin/robots/green_kevin.srdf'
            module = op.RaveCreateModule(self.env, 'urdf')
            name = module.SendCommand('LoadURI '+human_urdf_path+' '+human_srdf_path)
            self.human_model = self.env.GetRobots()[1]
            # self.set_human_arm('rightarm')
            # self.human_model.SetActiveManipulator('leftarm')

            rotx_correction = np.matrix([[1., 0., 0., 0.],
                                         [0., 0., -1., 0.],
                                         [0., 1., 0., 0.],
                                         [0., 0., 0., 1.]])

            roty_correction = np.matrix([[0., 0., 1., 0.],
                                         [0., 1., 0., 0.],
                                         [-1., 0., 0., 0.],
                                         [0., 0., 0., 1.]])

            self.human_rot_correction = roty_correction*rotx_correction

            human_trans_start = np.matrix([[1., 0., 0., 0.],
                                     [0., 1., 0., 0.],
                                     [0., 0., 1., 1.45],
                                     [0., 0., 0., 1.]])
            self.human_model.SetTransform(np.array(human_trans_start*self.human_rot_correction))
            self.originsubject_B_originworld = np.matrix(human_trans_start*self.human_rot_correction).I

            v = self.human_model.GetActiveDOFValues()
            v[self.human_model.GetJoint('green_kevin/neck_twist_joint').GetDOFIndex()] = 0  # m.radians(60)
            v[self.human_model.GetJoint('green_kevin/neck_tilt_joint').GetDOFIndex()] = 0.0
            v[self.human_model.GetJoint('green_kevin/neck_head_rotz_joint').GetDOFIndex()] = 0  # -m.radians(30)
            v[self.human_model.GetJoint('green_kevin/neck_head_roty_joint').GetDOFIndex()] = -0.0
            v[self.human_model.GetJoint('green_kevin/neck_head_rotx_joint').GetDOFIndex()] = 0
            v[self.human_model.GetJoint('green_kevin/neck_body_joint').GetDOFIndex()] = -0.
            v[self.human_model.GetJoint('green_kevin/upper_mid_body_joint').GetDOFIndex()] = 0.
            v[self.human_model.GetJoint('green_kevin/mid_lower_body_joint').GetDOFIndex()] = 0.
            v[self.human_model.GetJoint('green_kevin/body_quad_left_joint').GetDOFIndex()] = 0.
            v[self.human_model.GetJoint('green_kevin/body_quad_right_joint').GetDOFIndex()] = 0.
            v[self.human_model.GetJoint('green_kevin/quad_calf_left_joint').GetDOFIndex()] = 0.
            v[self.human_model.GetJoint('green_kevin/quad_calf_right_joint').GetDOFIndex()] = 0.
            v[self.human_model.GetJoint('green_kevin/calf_foot_left_joint').GetDOFIndex()] = 0.
            v[self.human_model.GetJoint('green_kevin/calf_foot_right_joint').GetDOFIndex()] = 0.
            v[self.human_model.GetJoint('green_kevin/body_arm_right_rotx_joint').GetDOFIndex()] = 0.
            v[self.human_model.GetJoint('green_kevin/body_arm_right_rotz_joint').GetDOFIndex()] = 0.
            v[self.human_model.GetJoint('green_kevin/body_arm_right_roty_joint').GetDOFIndex()] = 0.
            v[self.human_model.GetJoint('green_kevin/arm_forearm_right_joint').GetDOFIndex()] = 0.
            v[self.human_model.GetJoint('green_kevin/forearm_hand_right_rotx_joint').GetDOFIndex()] = 0.
            v[self.human_model.GetJoint('green_kevin/forearm_hand_right_roty_joint').GetDOFIndex()] = 0.
            v[self.human_model.GetJoint('green_kevin/forearm_hand_right_rotz_joint').GetDOFIndex()] = 0.
            v[self.human_model.GetJoint('green_kevin/body_arm_left_rotx_joint').GetDOFIndex()] = 0.
            v[self.human_model.GetJoint('green_kevin/body_arm_left_rotz_joint').GetDOFIndex()] = 0.
            v[self.human_model.GetJoint('green_kevin/body_arm_left_roty_joint').GetDOFIndex()] = 0.
            v[self.human_model.GetJoint('green_kevin/arm_forearm_left_joint').GetDOFIndex()] = 0.
            v[self.human_model.GetJoint('green_kevin/forearm_hand_left_rotx_joint').GetDOFIndex()] = 0.
            v[self.human_model.GetJoint('green_kevin/forearm_hand_left_roty_joint').GetDOFIndex()] = 0.
            v[self.human_model.GetJoint('green_kevin/forearm_hand_left_rotz_joint').GetDOFIndex()] = 0.
            self.human_model.SetActiveDOFValues(v, 2)
            self.env.UpdatePublishedBodies()
            # headmodel = self.human_model.GetLink('green_kevin/head_link')
            # head_T = np.matrix(headmodel.GetTransform())
        else:
            self.a_model_is_loaded = False
            print 'I got a bad model. What is going on???'
            return None
        print 'OpenRave has succesfully been initialized. \n'

    def set_human_model_dof(self, dof):
        # bth = m.degrees(headrest_th)
        if not len(dof) == 7:
            print 'There should be exactly 7 values used for arm configuration. ' \
                  'But instead ' + str(len(dof)) + 'was sent. This is a problem!'

        v = self.human_model.GetActiveDOFValues()
        if self.human_arm == 'leftarm' and self.model == 'green_kevin':
            v[self.human_model.GetJoint('green_kevin/body_arm_left_rotx_joint').GetDOFIndex()] = dof[0]
            v[self.human_model.GetJoint('green_kevin/body_arm_left_roty_joint').GetDOFIndex()] = dof[1]
            v[self.human_model.GetJoint('green_kevin/body_arm_left_rotz_joint').GetDOFIndex()] = dof[2]
            v[self.human_model.GetJoint('green_kevin/arm_forearm_left_joint').GetDOFIndex()] = dof[3]
            v[self.human_model.GetJoint('green_kevin/forearm_hand_left_rotx_joint').GetDOFIndex()] = dof[4]
            v[self.human_model.GetJoint('green_kevin/forearm_hand_left_roty_joint').GetDOFIndex()] = dof[5]
            v[self.human_model.GetJoint('green_kevin/forearm_hand_left_rotz_joint').GetDOFIndex()] = dof[6]
        elif self.human_arm == 'rightarm' and self.model == 'green_kevin':
            v[self.human_model.GetJoint('green_kevin/body_arm_right_rotx_joint').GetDOFIndex()] = dof[0]
            v[self.human_model.GetJoint('green_kevin/body_arm_right_roty_joint').GetDOFIndex()] = dof[1]
            v[self.human_model.GetJoint('green_kevin/body_arm_right_rotz_joint').GetDOFIndex()] = dof[2]
            v[self.human_model.GetJoint('green_kevin/arm_forearm_right_joint').GetDOFIndex()] = dof[3]
            v[self.human_model.GetJoint('green_kevin/forearm_hand_right_rotx_joint').GetDOFIndex()] = dof[4]
            v[self.human_model.GetJoint('green_kevin/forearm_hand_right_roty_joint').GetDOFIndex()] = dof[5]
            v[self.human_model.GetJoint('green_kevin/forearm_hand_right_rotz_joint').GetDOFIndex()] = dof[6]
        else:
            print 'Either Im not sure what arm or what model to use to set the arm dof for!'
        self.human_model.SetActiveDOFValues(v)
        self.env.UpdatePublishedBodies()

    def set_human_configuration(self, head_height):
        # bth = m.degrees(headrest_th)

        human_transform = np.matrix([[1., 0., 0., 0.],
                                     [0., 1., 0., 0.],
                                     [0., 0., 1., head_height],
                                     [0., 0., 0., 1.]])
        self.human_model.SetTransform(np.array(human_transform))
        bth = 0
        v = self.human_model.GetActiveDOFValues()
            # 0 degrees, 0 height
        if (head_height >= 0.) and (head_height <= 40.):  # between 0 and 40 degrees
            # v[self.human_model.GetJoint('green_kevin/neck_twist_joint').GetDOFIndex()] = -((bth/40)*(0 - 0)+0)
            v[self.human_model.GetJoint('green_kevin/neck_tilt_joint').GetDOFIndex()] = ((bth/40)*(.7 - 0)+0)
            v[self.human_model.GetJoint('green_kevin/neck_body_joint').GetDOFIndex()] = (bth/40)*(.02-(0))+(0)
            # v[self.human_model.GetJoint('green_kevin/neck_head_rotz_joint').GetDOFIndex()] = -((bth/40)*(0 - 0)+0)
            v[self.human_model.GetJoint('green_kevin/neck_head_roty_joint').GetDOFIndex()] = -((bth/40)*(-0.2 - 0)+0)
            v[self.human_model.GetJoint('green_kevin/neck_head_rotx_joint').GetDOFIndex()] = -((bth/40)*(0 - 0)+0)
            v[self.human_model.GetJoint('green_kevin/upper_mid_body_joint').GetDOFIndex()] = (bth/40)*(0.5-0)+0
            v[self.human_model.GetJoint('green_kevin/mid_lower_body_joint').GetDOFIndex()] = (bth/40)*(0.26-0)+(0)
            v[self.human_model.GetJoint('green_kevin/body_quad_left_joint').GetDOFIndex()] = -0.05
            v[self.human_model.GetJoint('green_kevin/body_quad_right_joint').GetDOFIndex()] = -0.05
            v[self.human_model.GetJoint('green_kevin/quad_calf_left_joint').GetDOFIndex()] = .05
            v[self.human_model.GetJoint('green_kevin/quad_calf_right_joint').GetDOFIndex()] = .05
            v[self.human_model.GetJoint('green_kevin/calf_foot_left_joint').GetDOFIndex()] = (bth/40)*(.0-0)+0
            v[self.human_model.GetJoint('green_kevin/calf_foot_right_joint').GetDOFIndex()] = (bth/40)*(.0-0)+0
            v[self.human_model.GetJoint('green_kevin/body_arm_left_joint').GetDOFIndex()] = (bth/40)*(-0.15-(-0.15))+(-0.15)
            v[self.human_model.GetJoint('green_kevin/body_arm_right_joint').GetDOFIndex()] = (bth/40)*(-0.15-(-0.15))+(-0.15)
            v[self.human_model.GetJoint('green_kevin/arm_forearm_left_joint').GetDOFIndex()] = (bth/40)*(.86-0.1)+0.1
            v[self.human_model.GetJoint('green_kevin/arm_forearm_right_joint').GetDOFIndex()] = (bth/40)*(.86-0.1)+0.1
            v[self.human_model.GetJoint('green_kevin/forearm_hand_left_joint').GetDOFIndex()] = 0
            v[self.human_model.GetJoint('green_kevin/forearm_hand_right_joint').GetDOFIndex()] = 0
        elif (head_height > 40.) and (head_height <= 80.):  # between 0 and 40 degrees
            v[self.human_model.GetJoint('green_kevin/neck_tilt_joint').GetDOFIndex()] = (((bth-40)/40)*(0.7 - 0.7)+0.7)
            v[self.human_model.GetJoint('green_kevin/neck_body_joint').GetDOFIndex()] = ((bth-40)/40)*(-0.1-(.02))+(.02)
            # v[self.human_model.GetJoint('green_kevin/neck_head_rotz_joint').GetDOFIndex()] = -((bth/40)*(0 - 0)+0)
            v[self.human_model.GetJoint('green_kevin/neck_head_roty_joint').GetDOFIndex()] = -((bth/40)*(.02 - (-0.2))+(-0.2))
            v[self.human_model.GetJoint('green_kevin/neck_head_rotx_joint').GetDOFIndex()] = -((bth/40)*(0 - 0)+0)
            v[self.human_model.GetJoint('green_kevin/upper_mid_body_joint').GetDOFIndex()] = ((bth-40)/40)*(.7-(.5))+(.5)
            v[self.human_model.GetJoint('green_kevin/mid_lower_body_joint').GetDOFIndex()] = ((bth-40)/40)*(.63-(.26))+(.26)
            v[self.human_model.GetJoint('green_kevin/body_quad_left_joint').GetDOFIndex()] = -0.05
            v[self.human_model.GetJoint('green_kevin/body_quad_right_joint').GetDOFIndex()] = -0.05
            v[self.human_model.GetJoint('green_kevin/quad_calf_left_joint').GetDOFIndex()] = 0.05
            v[self.human_model.GetJoint('green_kevin/quad_calf_right_joint').GetDOFIndex()] = 0.05
            v[self.human_model.GetJoint('green_kevin/calf_foot_left_joint').GetDOFIndex()] = ((bth-40)/40)*(0-0)+(0)
            v[self.human_model.GetJoint('green_kevin/calf_foot_right_joint').GetDOFIndex()] = ((bth-40)/40)*(0-0)+(0)
            v[self.human_model.GetJoint('green_kevin/body_arm_left_joint').GetDOFIndex()] = ((bth-40)/40)*(-0.1-(-0.15))+(-0.15)
            v[self.human_model.GetJoint('green_kevin/body_arm_right_joint').GetDOFIndex()] = ((bth-40)/40)*(-0.1-(-0.15))+(-0.15)
            v[self.human_model.GetJoint('green_kevin/arm_forearm_left_joint').GetDOFIndex()] = ((bth-40)/40)*(1.02-0.86)+.86
            v[self.human_model.GetJoint('green_kevin/arm_forearm_right_joint').GetDOFIndex()] = ((bth-40)/40)*(1.02-0.86)+.86
            v[self.human_model.GetJoint('green_kevin/forearm_hand_left_joint').GetDOFIndex()] = ((bth-40)/40)*(.35-0)+0
            v[self.human_model.GetJoint('green_kevin/forearm_hand_right_joint').GetDOFIndex()] = ((bth-40)/40)*(.35-0)+0
        else:
            print 'Error: Head height out of range (should be 0 - 80 degrees)'

        self.human_model.SetActiveDOFValues(v, 2)
        # self.env.UpdatePublishedBodies()

    def show_rviz(self):
        #rospy.init_node(''.join(['base_selection_goal_visualization']))
        sub_pos, sub_ori = Bmat_to_pos_quat(self.originsubject_B_originworld)
        self.publish_sub_marker(sub_pos, sub_ori)

#         if self.model == 'autobed':
#             self.selection_mat = np.zeros(len(self.goals))
#             self.goal_list = np.zeros([len(self.goals),4,4])
#             headmodel = self.autobed.GetLink('head_link')
#             pr2_B_head = np.matrix(headmodel.GetTransform())
#             for i in xrange(len(self.goals)):
#                 self.goal_list[i] = copy.copy(pr2_B_head*np.matrix(self.goals[i,0]))
#                 self.selection_mat[i] = copy.copy(self.goals[i,1])
# #            for target in self.goals:
# #                self.goal_list.append(pr2_B_head*np.matrix(target[0]))
# #                self.selection_mat.append(target[1])
#             self.set_goals()

        self.publish_goal_markers(self.goal_list)
        #for i in xrange(len(self.goal_list)):
        #    g_pos,g_ori = Bmat_to_pos_quat(self.goal_list[i])
        #    self.publish_goal_marker(g_pos, g_ori, ''.join(['goal_',str(i)]))

    # Publishes as a marker array the goal marker locations used by openrave to rviz so we can see how it overlaps with the subject
    def publish_goal_markers(self, goals):
        vis_pub = rospy.Publisher('~goal_markers', MarkerArray, queue_size=1, latch=True)
        goal_markers = MarkerArray()
        for num, goal_marker in enumerate(goals):
            pos, ori = Bmat_to_pos_quat(goal_marker)
            marker = Marker()
            #marker.header.frame_id = "/base_footprint"
            marker.header.frame_id = "/base_link"
            marker.header.stamp = rospy.Time()
            marker.ns = str(num)
            marker.id = 0
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.pose.position.x = pos[0]
            marker.pose.position.y = pos[1]
            marker.pose.position.z = pos[2]
            marker.pose.orientation.x = ori[0]
            marker.pose.orientation.y = ori[1]
            marker.pose.orientation.z = ori[2]
            marker.pose.orientation.w = ori[3]
            marker.scale.x = .05*3
            marker.scale.y = .05*3
            marker.scale.z = .01*3
            marker.color.a = 1.
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            goal_markers.markers.append(marker)
        vis_pub.publish(goal_markers)
        print 'Published a goal marker to rviz'

    # Publishes a goal marker location used by openrave to rviz so we can see how it overlaps with the subject
    def publish_goal_marker(self, pos, ori, name):
        vis_pub = rospy.Publisher(''.join(['~', name]), Marker, queue_size=1, latch=True)
        marker = Marker()
        #marker.header.frame_id = "/base_footprint"
        marker.header.frame_id = "/base_link"
        marker.header.stamp = rospy.Time()
        marker.ns = name
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose.position.x = pos[0]
        marker.pose.position.y = pos[1]
        marker.pose.position.z = pos[2]
        marker.pose.orientation.x = ori[0]
        marker.pose.orientation.y = ori[1]
        marker.pose.orientation.z = ori[2]
        marker.pose.orientation.w = ori[3]
        marker.scale.x = .2
        marker.scale.y = .2
        marker.scale.z = .2
        marker.color.a = 1.
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        vis_pub.publish(marker)
        print 'Published a goal marker to rviz'

    # Publishes the wheelchair model location used by openrave to rviz so we can see how it overlaps with the real wheelchair
    def publish_sub_marker(self, pos, ori):
        marker = Marker()
        #marker.header.frame_id = "/base_footprint"
        marker.header.frame_id = "/base_link"
        marker.header.stamp = rospy.Time()
        marker.id = 0
        marker.type = Marker.MESH_RESOURCE
        marker.action = Marker.ADD
        marker.pose.position.x = pos[0]
        marker.pose.position.y = pos[1]
        marker.pose.position.z = pos[2]
        marker.pose.orientation.x = ori[0]
        marker.pose.orientation.y = ori[1]
        marker.pose.orientation.z = ori[2]
        marker.pose.orientation.w = ori[3]
        marker.color.a = 1.
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        if self.model == 'chair':
            name = 'subject_model'
            marker.mesh_resource = "package://hrl_base_selection/models/wheelchair_and_body_assembly_rviz.STL"
            marker.scale.x = 1.0
            marker.scale.y = 1.0
            marker.scale.z = 1.0
        elif self.model == 'bed':
            name = 'subject_model'
            marker.mesh_resource = "package://hrl_base_selection/models/head_bed.dae"
            marker.scale.x = 1.0
            marker.scale.y = 1.0
            marker.scale.z = 1.0
        elif self.model == 'autobed':
            name = 'subject_model'
            marker.mesh_resource = "package://hrl_base_selection/models/bed_and_body_v3_rviz.dae"
            marker.scale.x = 1.0
            marker.scale.y = 1.0
            marker.scale.z = 1.0
        elif self.model is None:
            print 'Not publishing a marker, no specific model is being used'
        else:
            print 'I got a bad model. What is going on???'
            return None
        vis_pub = rospy.Publisher(''.join(['~',name]), Marker, queue_size=1, latch=True)
        marker.ns = ''.join(['base_service_',name])
        vis_pub.publish(marker)
        print 'Published a model of the subject to rviz'


    def gen_joint_limit_weight(self, q, side):
        # define the total range limit for each joint
        if 'left' in side:
            joint_min = [-40., -30., -44., -133., -400., -130., -400.]
            joint_max = [130., 80., 224., 0., 400., 0., 400.]
        elif 'right' in side:
            # print 'Need to check the joint limits for the right arm'
            joint_min = [-130., -30., -224., -133., -400., -130., -400.]
            joint_max = [40., 80., 44., 0., 400., 0., 400.]
        # l1_max = 40.
        # l2_min = -30.
        # l2_max = 80.
        # l3_min = -44.
        # l3_max = 224.
        # l4_min = 0.
        # l4_max = 133.
        # l5_min = -1000.  # continuous
        # l5_max = 1000.  # continuous
        # l6_min = 0.
        # l6_max = 130.
        # l7_min = -1000.  # continuous
        # l7_max = 1000  # continuous
        # l2 = 120.
        # l3 = 268.
        # l4 = 133.
        # l5 = 10000.  # continuous
        # l6 = 130.
        # l7 = 10000.  # continuous

        weights = np.zeros(7)
        for joint in xrange(len(weights)):
            weights[joint] = 1. - m.pow(0.5, np.abs((joint_max[joint] - joint_min[joint])/2 - m.degrees(q[joint]) - joint_min[joint]))
        weights[4] = 1.
        weights[6] = 1.
        return np.diag(weights)

if __name__ == "__main__":
    rospy.init_node('score_generator')
    mytask = 'shoulder'
    mymodel = 'chair'
    #mytask = 'all_goals'
    start_time = time.time()
    selector = ScoreGeneratorDressing(visualize=False,task=mytask,goals = None,model=mymodel)
    #selector.choose_task(mytask)
    score_sheet = selector.handle_score_generation()

    print 'Time to load find generate all scores: %fs'%(time.time()-start_time)

    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('hrl_base_selection')
    save_pickle(score_sheet, ''.join([pkg_path, '/data/', mymodel, '_', mytask, '.pkl']))
    print 'Time to complete program, saving all data: %fs' % (time.time()-start_time)






