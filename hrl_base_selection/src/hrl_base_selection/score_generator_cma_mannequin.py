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

#from joblib import Parallel, delayed


class ScoreGenerator(object):

    def __init__(self, visualize=False, targets='all_goals', reference_names=['head'], goals=None, model='autobed',
                 tf_listener=None, task='shaving'):
        # if tf_listener is None:
        #     self.tf_listener = tf.TransformListener()
        # else:
        #     self.tf_listener = tf_listener
        self.visualize = visualize
        self.model = model

        self.arm = 'leftarm'
        self.opposite_arm = 'rightarm'

        self.a_model_is_loaded = False
        self.goals = goals
        self.pr2_B_reference = []
        self.task = task

        self.reference_names = reference_names

        self.head_angles = []

        self.reachable = {}
        self.manipulable = {}
        self.scores = {}
        self.headx = 0.
        self.heady = 0.
        self.distance = 0.
        self.score_length = {}
        self.sorted_scores = {}
        self.environment_model = None
        self.vision_cone = None
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
        if self.model == 'chair':
            headmodel = self.wheelchair.GetLink('wheelchair/head_link')
            ual = self.wheelchair.GetLink('wheelchair/arm_left_link')
            uar = self.wheelchair.GetLink('wheelchair/arm_right_link')
            fal = self.wheelchair.GetLink('wheelchair/forearm_left_link')
            far = self.wheelchair.GetLink('wheelchair/forearm_right_link')
            footl = self.wheelchair.GetLink('wheelchair/quad_left_link')
            footr = self.wheelchair.GetLink('wheelchair/quad_right_link')
            kneel = self.wheelchair.GetLink('wheelchair/calf_left_link')
            kneer = self.wheelchair.GetLink('wheelchair/calf_right_link')
            ch = self.wheelchair.GetLink('wheelchair/upper_body_link')
            origin_B_head = np.matrix(headmodel.GetTransform())
            origin_B_ual = np.matrix(ual.GetTransform())
            origin_B_uar = np.matrix(uar.GetTransform())
            origin_B_fal = np.matrix(fal.GetTransform())
            origin_B_far = np.matrix(far.GetTransform())
            origin_B_footl = np.matrix(footl.GetTransform())
            origin_B_footr = np.matrix(footr.GetTransform())
            origin_B_kneel = np.matrix(kneel.GetTransform())
            origin_B_kneer = np.matrix(kneer.GetTransform())
            origin_B_ch = np.matrix(ch.GetTransform())
        elif self.model == 'autobed':
            headmodel = self.autobed.GetLink('autobed/head_link')
            ual = self.autobed.GetLink('autobed/upper_arm_left_link')
            uar = self.autobed.GetLink('autobed/upper_arm_right_link')
            fal = self.autobed.GetLink('autobed/fore_arm_left_link')
            far = self.autobed.GetLink('autobed/fore_arm_right_link')
            hal = self.autobed.GetLink('autobed/hand_left_link')
            har = self.autobed.GetLink('autobed/hand_right_link')
            footl = self.autobed.GetLink('autobed/foot_left_link')
            footr = self.autobed.GetLink('autobed/foot_right_link')
            kneel = self.autobed.GetLink('autobed/knee_left_link')
            kneer = self.autobed.GetLink('autobed/knee_right_link')
            ch = self.autobed.GetLink('autobed/torso_link')
            origin_B_ual = np.matrix(ual.GetTransform())
            origin_B_uar = np.matrix(uar.GetTransform())
            origin_B_fal = np.matrix(fal.GetTransform())
            origin_B_far = np.matrix(far.GetTransform())
            origin_B_hal = np.matrix(hal.GetTransform())
            origin_B_har = np.matrix(har.GetTransform())
            origin_B_footl = np.matrix(footl.GetTransform())
            origin_B_footr = np.matrix(footr.GetTransform())
            origin_B_kneel = np.matrix(kneel.GetTransform())
            origin_B_kneer = np.matrix(kneer.GetTransform())
            origin_B_ch = np.matrix(ch.GetTransform())
            origin_B_head = np.matrix(headmodel.GetTransform())
        elif self.model == None:
            print 'Running score generator in real-time mode!'
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
                self.origin_B_references.append(origin_B_hal)
            elif y == 'hand_right':
                self.origin_B_references.append(origin_B_har)
            elif y == 'foot_left':
                self.origin_B_references.append(origin_B_footl)
            elif y == 'foot_right':
                self.origin_B_references.append(origin_B_footr)
            elif y == 'knee_left':
                self.origin_B_references.append(origin_B_kneel)
            elif y == 'knee_right':
                self.origin_B_references.append(origin_B_kneer)
            elif y == 'chest':
                self.origin_B_references.append(origin_B_ch)
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

    def receive_new_goals(self, goals, reference_options=None):
        origin_B_pr2 = np.matrix([[       1.,        0.,   0.,         0.0],
                                  [       0.,        1.,   0.,         0.0],
                                  [       0.,        0.,   1.,         0.0],
                                  [       0.,        0.,   0.,         1.0]])
        if reference_options:
            self.reference_names = reference_options
            self.origin_B_references = []
            if self.model == 'chair':
                headmodel = self.wheelchair.GetLink('wheelchair/head_link')
                ual = self.wheelchair.GetLink('wheelchair/arm_left_link')
                uar = self.wheelchair.GetLink('wheelchair/arm_right_link')
                fal = self.wheelchair.GetLink('wheelchair/forearm_left_link')
                far = self.wheelchair.GetLink('wheelchair/forearm_right_link')
                footl = self.wheelchair.GetLink('wheelchair/quad_left_link')
                footr = self.wheelchair.GetLink('wheelchair/quad_right_link')
                kneel = self.wheelchair.GetLink('wheelchair/calf_left_link')
                kneer = self.wheelchair.GetLink('wheelchair/calf_right_link')
                ch = self.wheelchair.GetLink('wheelchair/upper_body_link')
                origin_B_head = np.matrix(headmodel.GetTransform())
                origin_B_ual = np.matrix(ual.GetTransform())
                origin_B_uar = np.matrix(uar.GetTransform())
                origin_B_fal = np.matrix(fal.GetTransform())
                origin_B_far = np.matrix(far.GetTransform())
                origin_B_footl = np.matrix(footl.GetTransform())
                origin_B_footr = np.matrix(footr.GetTransform())
                origin_B_kneel = np.matrix(kneel.GetTransform())
                origin_B_kneer = np.matrix(kneer.GetTransform())
                origin_B_ch = np.matrix(ch.GetTransform())
                origin_B_head = np.matrix(headmodel.GetTransform())
            elif self.model == 'autobed':
                headmodel = self.autobed.GetLink('autobed/head_link')
                ual = self.autobed.GetLink('autobed/upper_arm_left_link')
                uar = self.autobed.GetLink('autobed/upper_arm_right_link')
                fal = self.autobed.GetLink('autobed/fore_arm_left_link')
                far = self.autobed.GetLink('autobed/fore_arm_right_link')
                footl = self.autobed.GetLink('autobed/foot_left_link')
                footr = self.autobed.GetLink('autobed/foot_right_link')
                kneel = self.autobed.GetLink('autobed/knee_left_link')
                kneer = self.autobed.GetLink('autobed/knee_right_link')
                ch = self.autobed.GetLink('autobed/torso_link')
                origin_B_ual = np.matrix(ual.GetTransform())
                origin_B_uar = np.matrix(uar.GetTransform())
                origin_B_fal = np.matrix(fal.GetTransform())
                origin_B_far = np.matrix(far.GetTransform())
                origin_B_footl = np.matrix(footl.GetTransform())
                origin_B_footr = np.matrix(footr.GetTransform())
                origin_B_kneel = np.matrix(kneel.GetTransform())
                origin_B_kneer = np.matrix(kneer.GetTransform())
                origin_B_ch = np.matrix(ch.GetTransform())
                origin_B_head = np.matrix(headmodel.GetTransform())
            elif self.model is None:
                origin_B_pr2 = np.matrix(np.eye(4))
            else:
                print 'I GOT A BAD MODEL. NOT SURE WHAT TO DO NOW!'


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
                elif y == 'foot_left':
                    self.origin_B_references.append(origin_B_footl)
                elif y == 'foot_right':
                    self.origin_B_references.append(origin_B_footr)
                elif y == 'knee_left':
                    self.origin_B_references.append(origin_B_kneel)
                elif y == 'knee_right':
                    self.origin_B_references.append(origin_B_kneer)
                elif y == 'chest':
                    self.origin_B_references.append(origin_B_ch)
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
            #self.goal_list.append(pr2_B_head*np.matrix(target[0])*goal_B_gripper)
            self.reference_mat[w] = int(self.goals[w, 2])
            self.goal_list[w] = copy.copy(self.origin_B_references[int(self.reference_mat[w])] *
                                          np.matrix(self.goals[w, 0]))
            self.selection_mat[w] = self.goals[w, 1]

        self.set_goals()

    def set_goals(self, single_goal=False):
        if single_goal is False:
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

    def set_arm(self, arm):
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
        self.robot.SetActiveManipulator(self.arm)
        self.manip = self.robot.GetActiveManipulator()
        self.ikmodel = op.databases.inversekinematics.InverseKinematicsModel(self.robot, iktype=op.IkParameterization.Type.Transform6D)
        # self.ikmodel = op.databases.inversekinematics.InverseKinematicsModel(self.robot, iktype=op.IkParameterization.Type.Translation3D)
        if not self.ikmodel.load():
            print 'IK model not found. Will now generate an IK model. This will take a while!'
            self.ikmodel.autogenerate()
            # self.ikmodel.generate(iktype=op.IkParameterizationType.Translation3D, freejoints=[self.arm[0]+'_shoulder_pan_joint', self.arm[0]+'_shoulder_lift_joint', self.arm[0]+'_upper_arm_roll_joint', self.arm[0]+'_elbow_flex_joint'], freeinc=0.01)
        self.manipprob = op.interfaces.BaseManipulation(self.robot)

    def handle_score_generation(self, plot=False):
        scoring_start_time = time.time()
        if not self.a_model_is_loaded:
            print 'Somehow a model has not been loaded. This is bad!'
            return None
        print 'Starting to generate the score. This is going to take a while.'
        # Results are stored in the following format:
        # optimization_results[<model>, <number_of_configs>, <head_rest_angle>, <headx>, <heady>, <allow_bed_movement>]
        # Negative head read angle means head rest angle is a free DoF.

        head_x_range = [0.]
#        self.head_angles = np.array([[58, 18], [58, 0], [58, -18], [0, 0], [-58, 18], [-58, 0], [-58, -18]])
#         self.head_angles = np.array([[68, 10], [68, 0], [68, -10], [0, 0], [-68, 10], [68, 0], [-68, -10]])
#        if self.task == 'scratching_knee_left':
#        self.head_angles = np.array([[0, 0]])
        head_y_range = (np.arange(11)-5)*.03
        head_rest_range = np.arange(-10, 80.1, 10.)
        head_rest_range = [-10]
        # human_height_range = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
        # head_y_range = [0.]
        # bed_height = 'fixed'
        # score_parameters = []
        # score_parameters.append([self.model, ])
        if self.model == 'autobed':
            # human_height_range = ['1.5', '1.6', '1.7', '1.8', '1.9', '2.0']
            human_height_range = ['1.75']  # Mannequin is 1.75 meters tall
            print 'Task was ', self.task, '. This is for a mannequin which does not move its head.'
            self.head_angles = np.array([[0, 0]])
            score_parameters = ([t for t in ((tuple([self.model, num_configs, head_rest_angle, headx, heady, allow_bed_movement, human_height]))
                                             for num_configs in [2]
                                             for head_rest_angle in head_rest_range
                                             for headx in head_x_range
                                             for heady in head_y_range
                                             for allow_bed_movement in [1, 0]
                                             for human_height in human_height_range
                                             )
                                 ])
        elif self.model == 'chair':
            self.head_angles = np.array([[10, 68], [0, 68], [-10, 68], [0, 0], [10, -68], [0, 68], [-10, -68]])
            score_parameters = ([t for t in ((tuple([self.model, num_configs, 0, 0, 0, 0]))
                                             for num_configs in [1, 2]
                                             )
                                 ])
        else:
            print 'ERROR'
            print 'I do not know what model to use!'
            return

        start_time = time.time()
        # headx_min = 0.
        # headx_max = 0.0+.01
        # headx_int = 0.05
        # heady_min = -0.1
        # heady_min = -0.1
        # heady_max = 0.1+.01
        # heady_int = 0.1
        # # heady_int = 1.05
        # # start_x_min = -1.0
        # start_x_min = 0.0
        # start_x_max = 3.0+.01
        # start_x_int = 10.
        # # start_y_min = -2.0
        # start_y_min = 0.0
        # start_y_max = 2.0+.01
        # start_y_int = 10.
        # #head_y_range = (np.arange(5)-2)*.05  #[0]
        # head_y_range = (np.arange(11)-5)*.03
        # #head_y_range = np.array([0])
        # if self.model == 'chair':
        #     bedz_min = 0.
        #     bedtheta_min = 0.
        #     headx_min = 0.
        #     heady_min = 0.
        #     bedz_int = 100.
        #     bedtheta_int = 100.
        #     headx_int = 100.
        #     heady_int = 100.
        #

        optimization_results = dict.fromkeys(score_parameters)
        score_stuff = dict.fromkeys(score_parameters)
        # optimization_results[<model>, <number_of_configs>, <head_rest_angle>, <headx>, <heady>, <allow_bed_movement>]
        for parameters in score_parameters:
            parameter_start_time = time.time()
            print 'Working on task: ', self.task
            print 'Generating score for the following parameters: '
            print '[<model>, <number_of_configs>, <head_rest_angle>, <headx>, <heady>, <allow_bed_movement>, <human_height>]'
            print parameters
            num_config = parameters[1]
            self.head_rest_angle = parameters[2]
            self.headx = parameters[3]
            self.heady = parameters[4]
            self.allow_bed_movement = parameters[5]
            self.human_height = parameters[6]
            if self.model == 'autobed':
                self.env.Remove(self.autobed)
                self.setup_human_model(height=self.human_height, init=False)
#            self.heady=0.03
            if self.model == 'chair' and num_config == 1:
                maxiter = 10
                # popsize = 1000
                popsize = m.pow(4, 2)*100
                parameters_min = np.array([0.3, -2., -m.pi-.001, 0.])
                parameters_max = np.array([2., 2., m.pi+.001, 0.3])
                parameters_scaling = (parameters_max-parameters_min)/4.
                parameters_initialization = (parameters_max+parameters_min)/2.
                opts1 = {'seed': 1234, 'ftarget': -1., 'popsize': popsize, 'maxiter': maxiter, 'maxfevals': 1e8, 'CMA_cmean': 0.25,
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
                #score_stuff = dict()
                # optimization_results[<model>, <number_of_configs>, <head_rest_angle>, <headx>, <heady>, <allow_bed_movement>]
                score_stuff[parameters] = [config, score]

            elif self.model == 'autobed' and num_config == 1:
                maxiter = 10
                # popsize = 1000
#                popsize = m.pow(6, 2)*100
                popsize = 3000
                parameters_min = np.array([0.5, 0.5, m.radians(-225.)-0.0001, 0., 0., 45.*m.pi/180.])
                parameters_max = np.array([2.5, 2.0, m.radians(-20.)+.0001, 0.3, 0.2, 55.*m.pi/180.])
                parameters_scaling = (parameters_max-parameters_min)/4.

                parameters_scaling[5] = 2.*m.pi/180.
                parameters_initialization = (parameters_max+parameters_min)/2.
                opts1 = {'seed': 1234, 'ftarget': -1., 'popsize': popsize, 'maxiter': maxiter, 'maxfevals': 1e8, 'CMA_cmean': 0.25,
                         'scaling_of_variables': list(parameters_scaling),
                         'bounds': [list(parameters_min),
                                    list(parameters_max)]}
                # optimization_results[<model>, <number_of_configs>, <head_rest_angle>, <headx>, <heady>, <allow_bed_movement>]
                optimization_results[parameters] = cma.fmin(self.objective_function_one_config,
                                                            list(parameters_initialization),
                                                            1.,
                                                            options=opts1)
                config = optimization_results[parameters][0]
                score = optimization_results[parameters][1]
                print 'Config: ', config
                print 'Score: ', score
                #score_stuff = dict()
                # optimization_results[<model>, <number_of_configs>, <head_rest_angle>, <headx>, <heady>, <allow_bed_movement>]
                score_stuff[parameters] = [config, score]

            else:
                # cma.plot()
                # cma.show()
                # rospy.sleep(10)
                maxiter = 10
                popsize = 3000  # m.pow(4, 2)*100
                if self.allow_bed_movement == 0:
                    parameters_min = np.array([0.2, -3., -m.pi-.001, 0., 0.2, -3., -m.pi-.001, 0.])
                    parameters_max = np.array([3., 3., m.pi+.001, 0.3, 3., 3., m.pi+.001, 0.3])
                if self.model == 'chair':
                    parameters_min = np.array([0.3, -2., -m.pi-.001, 0., -1., -2., -m.pi-.001, 0.])
                    parameters_max = np.array([2., 2., m.pi+.001, 0.3, 2., 2., m.pi+.001, 0.3])
                if self.allow_bed_movement == 0 or self.model == 'chair':
                    parameters_scaling = (parameters_max-parameters_min)/4.
                    parameters_initialization = (parameters_max+parameters_min)/2.
                    parameters_initialization[1] = 1.0
                    parameters_initialization[5] = -1.0
                    opts2 = {'seed': 1234, 'ftarget': -1., 'popsize': popsize, 'maxiter': maxiter, 'maxfevals': 1e8, 'CMA_cmean': 0.25,
                             'scaling_of_variables': list(parameters_scaling),
                             'bounds': [list(parameters_min),
                                        list(parameters_max)]}
                    # optimization_results[2, self.heady, self.start_x, self.start_y] = [t for t in ((cma.fmin(self.objective_function_two_configs,
                    print 'Working on heady location:', self.heady
                    # optimization_results[<model>, <number_of_configs>, <head_rest_angle>, <headx>, <heady>, <allow_bed_movement>]
                    optimization_results[parameters] = cma.fmin(self.objective_function_two_configs,
                                                                # [0.75, 0.75, 0., 0.15, 0.75, -0.75, 0., 0.15],
                                                                list(parameters_initialization),
                                                                # [0., 0., 0., 0.15, 0.1, 35*m.pi/180, 0., 0., 0., 0.15, 0.1, 35*m.pi/180],
                                                                1.,
                                                                options=opts2)
                    # print optimization_results[2, self.heady, self.start_x, self.start_y][0]
                    config = optimization_results[parameters][0]
                    score = optimization_results[parameters][1]
                    config = np.insert(config, 4, 0.)
                    config = np.insert(config, 4, 0.)
                    config = np.insert(config, 10, 0.)
                    config = np.insert(config, 10, 0.)
                    optimization_results[parameters] = [config, score]
                    # optimization_results[2, self.heady, self.start_x, self.start_y][0] = np.insert(optimization_results[2, self.heady, self.start_x, self.start_y][0], 4, 0.)
                    # optimization_results[2, self.heady, self.start_x, self.start_y][0] = np.insert(optimization_results[2, self.heady, self.start_x, self.start_y][0], 10, 0.)
                    # optimization_results[2, self.heady, self.start_x, self.start_y][0] = np.insert(optimization_results[2, self.heady, self.start_x, self.start_y][0], 10, 0.)
                elif self.head_rest_angle > -1.:
                    # Deactivated head rest angle
                    # Parameters are: [x, y, th, z, bz, bth]
                    maxiter = 10
                    popsize = 3000  # m.pow(5, 2)*100
                    parameters_min = np.array([0.2, -3., -m.pi-.001, 0., 0., 0.2, -3., -m.pi-.001, 0., 0.])
                    # parameters_max = np.array([3., 3., m.pi+.001, 0.3, 0.2, 3., 3., m.pi+.001, 0.3, 0.2])
                    # At Henry's the bed can only range a few centimeters because of the overbed table
                    parameters_max = np.array([3., 3., m.pi+.001, 0.3, 0.08, 3., 3., m.pi+.001, 0.3, 0.08])
                    parameters_scaling = (parameters_max-parameters_min)/4.
                    parameters_initialization = (parameters_max+parameters_min)/2.
                    parameters_initialization[1] = 1.0
                    parameters_initialization[6] = -1.0
                    opts2 = {'seed': 1234, 'ftarget': -1., 'popsize': popsize, 'maxiter': maxiter, 'maxfevals': 1e8, 'CMA_cmean': 0.25,
                             'scaling_of_variables': list(parameters_scaling),
                             'bounds': [list(parameters_min),
                                        list(parameters_max)]}

                    # optimization_results[2, self.heady, self.start_x, self.start_y] = [t for t in ((cma.fmin(self.objective_function_two_configs,
                    print 'Working on heady location:', self.heady
                    optimization_results[parameters] = cma.fmin(self.objective_function_two_configs,
                                                                list(parameters_initialization),
                                                                # [0.75, 0.75, 0., 0.15, 0., 0.75, -0.75, 0., 0.15, 0.],
                                                                # [0., 0., 0., 0.15, 0.1, 35*m.pi/180, 0., 0., 0., 0.15, 0.1, 35*m.pi/180],
                                                                1.,
                                                                options=opts2)
                    # for self.start_x in np.arange(start_x_min, start_x_max, start_x_int)
                    # for self.start_y in np.arange(start_y_min, start_y_max, start_y_int)
                    # for self.heady in np.arange(heady_min, heady_max, heady_int)
                    config = optimization_results[parameters][0]
                    score = optimization_results[parameters][1]
                    config = np.insert(config, 5, np.radians(self.head_rest_angle))
                    config = np.insert(config, 11, np.radians(self.head_rest_angle))
                    optimization_results[parameters] = [config, score]

                else:
                    maxiter = 10
                    popsize = 3000  # m.pow(6, 2)*100
                    if self.task == 'feeding_trajectory':
                        parameters_min = np.array([0.2, -3., -m.pi - .001, 0., 0., 55.*m.pi/180., 0.2, -3., -m.pi - .001, 0., 0., 55.*m.pi/180.])
                    else:
                        parameters_min = np.array([0.2, -3., -m.pi-.001,  0., 0., 0., 0.2, -3., -m.pi-.001,  0.,  0., 0.])
                     # parameters_max = np.array([ 3.,  3.,  m.pi+.001, 0.3, 0.2, 80.*m.pi/180.,  3.,  3.,  m.pi+.001, 0.3, 0.2, 80.*m.pi/180.])
                    # Henry's bed can only rise a few centimeters because of the overbed table
                    parameters_max = np.array([3.,  3.,  m.pi+.001, 0.3, 0.2, 80.*m.pi/180.,  3.,  3.,  m.pi+.001, 0.3, 0.2, 80.*m.pi/180.])
                    parameters_scaling = (parameters_max-parameters_min)/4.
                    parameters_initialization = (parameters_max+parameters_min)/2.
                    parameters_initialization[1] = 1.0
                    parameters_initialization[7] = -1.0
                    # Parameters are: [x, y, th, z, bz, bth]
                    opts2 = {'seed': 1234, 'ftarget': -1., 'popsize': popsize, 'maxiter': maxiter, 'maxfevals': 1e8, 'CMA_cmean': 0.25,
                             'scaling_of_variables': list(parameters_scaling),
                             'bounds': [list(parameters_min),
                                        list(parameters_max)]}

                    # optimization_results[2, self.heady, self.start_x, self.start_y] = [t for t in ((cma.fmin(self.objective_function_two_configs,
                    print 'Working on heady location:', self.heady
                    optimization_results[parameters] = cma.fmin(self.objective_function_two_configs,
                                                                list(parameters_initialization),
                                                                # [0.5, 0.75, 0., 0.15, 0., 35*m.pi/180, 0.5, -0.75, 0., 0.15, 0., 35*m.pi/180],
                                                                # [0., 0., 0., 0.15, 0.1, 35*m.pi/180, 0., 0., 0., 0.15, 0.1, 35*m.pi/180],
                                                                1.,
                                                                options=opts2)
                    # for self.start_x in np.arange(start_x_min, start_x_max, start_x_int)
                    # for self.start_y in np.arange(start_y_min, start_y_max, start_y_int)
                    # for self.heady in np.arange(heady_min, heady_max, heady_int)
                    config = optimization_results[parameters][0]
                    score = optimization_results[parameters][1]
                    optimization_results[parameters] = [config, score]

                print optimization_results[parameters][0]
                print optimization_results[parameters][1]

                # score_stuff[self.heady, self.distance] = self.compare_results_one_vs_two_configs(optimization_results[1, self.heady, self.distance], optimization_results[2, self.heady, self.distance])
                score_stuff[parameters] = self.check_which_num_base_is_better(optimization_results[parameters])
                print 'Time to find scores for this set of parameters: %fs' % (time.time()-parameter_start_time)
                print 'Time elapsed so far for parameters: %fs' % (time.time()-scoring_start_time)

        # score_stuff = []  # np.zeros([len(optimization_results), 9])
        #
        #     score_stuff[num] = list(flatten([optimization_results[num][0], optimization_results[num][1], optimization_results[num][2][0], optimization_results[num][2][1]]))

        print 'SCORE RESULTS:'
        for item in score_stuff:
            print '(<model>, <number_of_configs>, <head_rest_angle>, <headx>, <heady>, <allow_bed_movement>, <human_height>):', item
            print '[[[x], [y], [th], [z], [bz], [bth]], score]'
            print 'Or, if there are two configurations:'
            print '[[[x1, x2], [y1, y2], [th1, th2], [z1, z2], [bz1, bz2], [bth1, bth2]], score]'
            print score_stuff[item]

        print 'Time to generate all scores for individual base locations: %fs' % (time.time()-start_time)
        print 'Number of configurations that were evaluated: ', len(score_stuff)
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
        # current_parameters = [1.73500286,  0.69365097, -2.17188387,  0.2316305 ,  0.17488576,
        # 0.87341131]
        # current_parameters[0]=2.5
        # self.heady = 0.0
        if not self.a_model_is_loaded:
            print 'Somehow a model has not been loaded. This is bad!'
            return None
        if len(current_parameters) == 6:
            x = current_parameters[0]
            y = current_parameters[1]
            th = current_parameters[2]
            z = current_parameters[3]
            bz = current_parameters[4]
            bth = current_parameters[5]
        else:
            x = current_parameters[0]
            y = current_parameters[1]
            th = current_parameters[2]
            z = current_parameters[3]
            bz = 0.
            bth = 0.

        # freej_min = np.array([-40., -30., -44., -133., -400., -130., -400.])
        # freej_max = np.array([130., 80., 224., 0., 400., 0., 400.])
        # freej_min = np.array([-40., -30., -44., -133.])
        # freej_max = np.array([130., 80., 224., 0.])

        #print 'Calculating new score'
        #starttime = time.time()
        origin_B_pr2 = np.matrix([[ m.cos(th), -m.sin(th),     0.,         x],
                                  [ m.sin(th),  m.cos(th),     0.,         y],
                                  [        0.,         0.,     1.,        0.],
                                  [        0.,         0.,     0.,        1.]])
        self.robot.SetTransform(np.array(origin_B_pr2))
        # rospy.sleep(1)
        v = self.robot.GetActiveDOFValues()
        v[self.robot.GetJoint('torso_lift_joint').GetDOFIndex()] = z
        self.robot.SetActiveDOFValues(v, 2)
        # rospy.sleep(1)

        if self.model == 'chair':
            self.env.UpdatePublishedBodies()
            headmodel = self.wheelchair.GetLink('wheelchair/head_link')
            ual = self.wheelchair.GetLink('wheelchair/arm_left_link')
            uar = self.wheelchair.GetLink('wheelchair/arm_right_link')
            fal = self.wheelchair.GetLink('wheelchair/forearm_left_link')
            far = self.wheelchair.GetLink('wheelchair/forearm_right_link')
            footl = self.wheelchair.GetLink('wheelchair/quad_left_link')
            footr = self.wheelchair.GetLink('wheelchair/quad_right_link')
            kneel = self.wheelchair.GetLink('wheelchair/calf_left_link')
            kneer = self.wheelchair.GetLink('wheelchair/calf_right_link')
            ch = self.wheelchair.GetLink('wheelchair/upper_body_link')
            origin_B_head = np.matrix(headmodel.GetTransform())
            origin_B_ual = np.matrix(ual.GetTransform())
            origin_B_uar = np.matrix(uar.GetTransform())
            origin_B_fal = np.matrix(fal.GetTransform())
            origin_B_far = np.matrix(far.GetTramarnsform())
            origin_B_footl = np.matrix(footl.GetTransform())
            origin_B_footr = np.matrix(footr.GetTransform())
            origin_B_kneel = np.matrix(kneel.GetTransform())
            origin_B_kneer = np.matrix(kneer.GetTransform())
            origin_B_ch = np.matrix(ch.GetTransform())
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
                elif self.reference_names[thing] == 'foot_left':
                    self.origin_B_references.append(origin_B_footl)
                elif self.reference_names[thing] == 'foot_right':
                    self.origin_B_references.append(origin_B_footr)
                elif self.reference_names[thing] == 'knee_left':
                    self.origin_B_references.append(origin_B_kneel)
                elif self.reference_names[thing] == 'knee_right':
                    self.origin_B_references.append(origin_B_kneer)
                elif self.reference_names[thing] == 'chest':
                    self.origin_B_references.append(origin_B_ch)
            for thing in xrange(len(self.goals)):
                self.goal_list[thing] = copy.copy(self.origin_B_references[int(self.reference_mat[thing])]*np.matrix(self.goals[thing, 0]))
                self.selection_mat[thing] = self.goals[thing, 1]
#            for target in self.goals:
#                self.goal_list.append(pr2_B_head*np.matrix(target[0]))
#                self.selection_mat.append(target[1])
            self.set_goals()
            headmodel = self.wheelchair.GetLink('wheelchair/head_link')

        elif self.model == 'autobed':
            self.selection_mat = np.zeros(len(self.goals))
            self.goal_list = np.zeros([len(self.goals), 4, 4])
            self.set_autobed(bz, bth, self.headx, self.heady)
            self.env.UpdatePublishedBodies()

            headmodel = self.autobed.GetLink('autobed/head_link')
            ual = self.autobed.GetLink('autobed/upper_arm_left_link')
            uar = self.autobed.GetLink('autobed/upper_arm_right_link')
            fal = self.autobed.GetLink('autobed/fore_arm_left_link')
            far = self.autobed.GetLink('autobed/fore_arm_right_link')
            hal = self.autobed.GetLink('autobed/fore_arm_left_link')
            har = self.autobed.GetLink('autobed/fore_arm_right_link')
            footl = self.autobed.GetLink('autobed/foot_left_link')
            footr = self.autobed.GetLink('autobed/foot_right_link')
            kneel = self.autobed.GetLink('autobed/knee_left_link')
            kneer = self.autobed.GetLink('autobed/knee_right_link')
            ch = self.autobed.GetLink('autobed/torso_link')
            origin_B_head = np.matrix(headmodel.GetTransform())
            origin_B_ual = np.matrix(ual.GetTransform())
            origin_B_uar = np.matrix(uar.GetTransform())
            origin_B_fal = np.matrix(fal.GetTransform())
            origin_B_far = np.matrix(far.GetTransform())
            origin_B_hal = np.matrix(hal.GetTransform())
            origin_B_har = np.matrix(har.GetTransform())
            origin_B_footl = np.matrix(footl.GetTransform())
            origin_B_footr = np.matrix(footr.GetTransform())
            origin_B_kneel = np.matrix(kneel.GetTransform())
            origin_B_kneer = np.matrix(kneer.GetTransform())
            origin_B_ch = np.matrix(ch.GetTransform())
            self.origin_B_references = []
            for thing in xrange(len(self.reference_names)):
                if self.reference_names[thing] == 'head':
                    self.origin_B_references.append(origin_B_head)
                    # self.origin_B_references.append(np.matrix(headmodel.GetTransform())
                elif self.reference_names[thing] == 'base_link':
                    self.origin_B_references.append(origin_B_pr2)
                    # self.origin_B_references[i] = np.matrix(self.robot.GetTransform())
                elif self.reference_names[thing] == 'upper_arm_left':
                    self.origin_B_references.append(origin_B_ual)
                elif self.reference_names[thing] == 'upper_arm_right':
                    self.origin_B_references.append(origin_B_uar)
                elif self.reference_names[thing] == 'forearm_left':
                    self.origin_B_references.append(origin_B_fal)
                elif self.reference_names[thing] == 'forearm_right':
                    self.origin_B_references.append(origin_B_far)
                elif self.reference_names[thing] == 'hand_left':
                    self.origin_B_references.append(origin_B_hal)
                elif self.reference_names[thing] == 'hand_right':
                    self.origin_B_references.append(origin_B_har)
                elif self.reference_names[thing] == 'foot_left':
                    self.origin_B_references.append(origin_B_footl)
                elif self.reference_names[thing] == 'foot_right':
                    self.origin_B_references.append(origin_B_footr)
                elif self.reference_names[thing] == 'knee_left':
                    self.origin_B_references.append(origin_B_kneel)
                elif self.reference_names[thing] == 'knee_right':
                    self.origin_B_references.append(origin_B_kneer)
                elif self.reference_names[thing] == 'chest':
                    self.origin_B_references.append(origin_B_ch)

            for thing in xrange(len(self.goals)):
                self.goal_list[thing] = copy.copy(self.origin_B_references[int(self.reference_mat[thing])]*np.matrix(self.goals[thing, 0]))
                self.selection_mat[thing] = self.goals[thing, 1]
            # for target in self.goals:
            #     self.goal_list.append(pr2_B_head*np.matrix(target[0]))
            #     self.selection_mat.append(target[1])
            self.set_goals()
        elif self.model is None:
            self.env.UpdatePublishedBodies()
        else:
            print 'I GOT A BAD MODEL. NOT SURE WHAT TO DO NOW!'
        distance = 10000000.
        out_of_reach = True

        for origin_B_grasp in self.origin_B_grasps:
            pr2_B_goal = origin_B_pr2.I*origin_B_grasp
            distance = np.min([np.linalg.norm(pr2_B_goal[:2, 3]), distance])

            if distance <= 1.3:
                out_of_reach = False
                # print 'not out of reach'
                break
        if out_of_reach:
            # print 'location is out of reach'
            return 10. +1.+ 20.*(distance - 1.3)

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
            v = self.robot.GetActiveDOFValues()
            if self.arm[0] == 'l':
                arm_sign = 1
            else:
                arm_sign = -1
            if self.task == 'blanket_feet_knees' or self.task == 'scratching_knee_left':
                v[self.robot.GetJoint(self.arm[0]+'_shoulder_pan_joint').GetDOFIndex()] = arm_sign*3.*3.14159/4.
                v[self.robot.GetJoint(self.arm[0]+'_shoulder_lift_joint').GetDOFIndex()] = -0.6
                v[self.robot.GetJoint(self.arm[0]+'_upper_arm_roll_joint').GetDOFIndex()] = arm_sign*m.radians(20)
                v[self.robot.GetJoint(self.arm[0]+'_elbow_flex_joint').GetDOFIndex()] = m.radians(-150.)
                v[self.robot.GetJoint(self.arm[0]+'_forearm_roll_joint').GetDOFIndex()] = m.radians(150.)
                v[self.robot.GetJoint(self.arm[0]+'_wrist_flex_joint').GetDOFIndex()] = m.radians(-110)
                v[self.robot.GetJoint(self.arm[0]+'_wrist_roll_joint').GetDOFIndex()] = arm_sign*0.0
            elif self.task == 'wiping_mouth' or self.task == 'wiping_forehead':
                # v[self.robot.GetJoint(self.arm[0]+'_shoulder_pan_joint').GetDOFIndex()] = arm_sign*0.8
                # v[self.robot.GetJoint(self.arm[0]+'_shoulder_lift_joint').GetDOFIndex()] = 0.0
                # v[self.robot.GetJoint(self.arm[0]+'_upper_arm_roll_joint').GetDOFIndex()] = arm_sign*1.57
                # v[self.robot.GetJoint(self.arm[0]+'_elbow_flex_joint').GetDOFIndex()] = -2.9
                # v[self.robot.GetJoint(self.arm[0]+'_forearm_roll_joint').GetDOFIndex()] = 3.0
                # v[self.robot.GetJoint(self.arm[0]+'_wrist_flex_joint').GetDOFIndex()] = -1.0
                # v[self.robot.GetJoint(self.arm[0]+'_wrist_roll_joint').GetDOFIndex()] = arm_sign*1.57
                v[self.robot.GetJoint(self.arm[0]+'_shoulder_pan_joint').GetDOFIndex()] = arm_sign*(1.8)
                v[self.robot.GetJoint(self.arm[0]+'_shoulder_lift_joint').GetDOFIndex()] = 0.4
                v[self.robot.GetJoint(self.arm[0]+'_upper_arm_roll_joint').GetDOFIndex()] = arm_sign*(1.9)
                v[self.robot.GetJoint(self.arm[0]+'_elbow_flex_joint').GetDOFIndex()] = -3.0
                v[self.robot.GetJoint(self.arm[0]+'_forearm_roll_joint').GetDOFIndex()] = arm_sign*(-3.5)
                v[self.robot.GetJoint(self.arm[0]+'_wrist_flex_joint').GetDOFIndex()] = -0.5
                v[self.robot.GetJoint(self.arm[0]+'_wrist_roll_joint').GetDOFIndex()] = 0.0
            else:
                print 'The arm initial pose is not defined properly.'
                v[self.robot.GetJoint('I HAVE NO IDEA WHAT TASK Im DOING').GetDOFIndex()] = 0.
            v[self.robot.GetJoint(self.opposite_arm[0]+'_shoulder_pan_joint').GetDOFIndex()] = arm_sign*(-1.8)
            v[self.robot.GetJoint(self.opposite_arm[0]+'_shoulder_lift_joint').GetDOFIndex()] = 2.45
            v[self.robot.GetJoint(self.opposite_arm[0]+'_upper_arm_roll_joint').GetDOFIndex()] = arm_sign*(-1.9)
            v[self.robot.GetJoint(self.opposite_arm[0]+'_elbow_flex_joint').GetDOFIndex()] = -2.0
            v[self.robot.GetJoint(self.opposite_arm[0]+'_forearm_roll_joint').GetDOFIndex()] = arm_sign*3.5
            v[self.robot.GetJoint(self.opposite_arm[0]+'_wrist_flex_joint').GetDOFIndex()] = -1.5
            v[self.robot.GetJoint(self.opposite_arm[0]+'_wrist_roll_joint').GetDOFIndex()] = 0.0
            # v[self.robot.GetJoint(self.opposite_arm[0]+'_shoulder_pan_joint').GetDOFIndex()] = -3.14/2
            # v[self.robot.GetJoint(self.opposite_arm[0]+'_shoulder_lift_joint').GetDOFIndex()] = -0.52
            # v[self.robot.GetJoint(self.opposite_arm[0]+'_upper_arm_roll_joint').GetDOFIndex()] = 0.
            # v[self.robot.GetJoint(self.opposite_arm[0]+'_elbow_flex_joint').GetDOFIndex()] = -3.14*2/3
            # v[self.robot.GetJoint(self.opposite_arm[0]+'_forearm_roll_joint').GetDOFIndex()] = 0.
            # v[self.robot.GetJoint(self.opposite_arm[0]+'_wrist_flex_joint').GetDOFIndex()] = 0.
            # v[self.robot.GetJoint(self.opposite_arm[0]+'_wrist_roll_joint').GetDOFIndex()] = 0.
            self.robot.SetActiveDOFValues(v, 2)
            self.env.UpdatePublishedBodies()
            # rospy.sleep(1)
            not_close_to_collision = True
            if self.env.CheckCollision(self.robot):  # self.manip.CheckIndependentCollision(op.CollisionReport()):
                not_close_to_collision = False
            '''
            origin_B_pr2 = np.matrix([[ m.cos(th), -m.sin(th),     0., x+.02],
                                      [ m.sin(th),  m.cos(th),     0., y+.02],
                                      [        0.,         0.,     1.,        0.],
                                      [        0.,         0.,     0.,        1.]])
            self.robot.SetTransform(np.array(origin_B_pr2))
            self.env.UpdatePublishedBodies()
            if self.manip.CheckIndependentCollision(op.CollisionReport()):
                not_close_to_collision = False

            origin_B_pr2 = np.matrix([[ m.cos(th), -m.sin(th),     0., x-.02],
                                      [ m.sin(th),  m.cos(th),     0., y+.02],
                                      [        0.,         0.,     1.,        0.],
                                      [        0.,         0.,     0.,        1.]])
            self.robot.SetTransform(np.array(origin_B_pr2))
            self.env.UpdatePublishedBodies()
            if self.manip.CheckIndependentCollision(op.CollisionReport()):
                not_close_to_collision = False

            origin_B_pr2 = np.matrix([[ m.cos(th), -m.sin(th),     0., x-.02],
                                      [ m.sin(th),  m.cos(th),     0., y-.02],
                                      [        0.,         0.,     1.,        0.],
                                      [        0.,         0.,     0.,        1.]])
            self.robot.SetTransform(np.array(origin_B_pr2))
            self.env.UpdatePublishedBodies()
            if self.manip.CheckIndependentCollision(op.CollisionReport()):
                not_close_to_collision = False

            origin_B_pr2 = np.matrix([[ m.cos(th), -m.sin(th),     0., x+.02],
                                      [ m.sin(th),  m.cos(th),     0., y-.02],
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
            '''
            if not_close_to_collision:
                # print 'No base collision! single config distance: ', distance
                reached = np.zeros(len(self.origin_B_grasps))
                manip = np.zeros(len(self.origin_B_grasps))
                for head_angle in self.head_angles:
                    self.rotate_head_and_update_goals(head_angle[0], head_angle[1], origin_B_pr2)
                    for num, Tgrasp in enumerate(self.origin_B_grasps):
                        sols = []
                        # print Tgrasp
                        # ikparam = op.IkParameterization([1.5, 0.05, 1.2], self.ikmodel.iktype)
                        # ikparam = op.IkParameterization(Tgrasp[0:3,3], self.ikmodel.iktype)
                        # print 'ikparam done'
                        '''
                        for freej_0 in xrange(int(freej_max[0]-freej_min[0])):
                            for freej_1 in xrange(int(freej_max[1]-freej_min[1])):
                                for freej_2 in xrange(int(freej_max[1]-freej_min[2])):
                                    for freej_3 in xrange(int(freej_max[1]-freej_min[3])):
                                        v = self.robot.GetActiveDOFValues()
                                        v[self.robot.GetJoint(self.arm[0]+'_shoulder_pan_joint').GetDOFIndex()] = freej_0+freej_min[0]
                                        v[self.robot.GetJoint(self.arm[0]+'_shoulder_lift_joint').GetDOFIndex()] = freej_1+freej_min[1]
                                        v[self.robot.GetJoint(self.arm[0]+'_upper_arm_roll_joint').GetDOFIndex()] = freej_2+freej_min[2]
                                        v[self.robot.GetJoint(self.arm[0]+'_elbow_flex_joint').GetDOFIndex()] = freej_3+freej_min[3]
                                        self.robot.SetActiveDOFValues(v, 2)
                                        self.env.UpdatePublishedBodies()
                                        ikparam = op.IkParameterization(np.array([.8, -.5, 1.2]), self.ikmodel.iktype)
                                        # print 'ikparam done'
                                        # print ikparam
                                        curr_sols = self.manip.FindIKSolutions(ikparam, filteroptions=op.IkFilterOptions.CheckEnvCollisions)
                                        # print 'sols done'
                                        if list(curr_sols):
                                            for sol in curr_sols:
                                                sols.append(sol)
                        '''

                        # sols = self.manip.FindIKSolutions(ikparam, filteroptions=op.IkFilterOptions.CheckEnvCollisions)
                        sols = self.manip.FindIKSolutions(Tgrasp, filteroptions=op.IkFilterOptions.CheckEnvCollisions)
                        # print 'sols done'
                        '''
                        if not list(sols):
                            v = self.robot.GetActiveDOFValues()
                            v[self.robot.GetJoint(self.opposite_arm[0]+'_shoulder_pan_joint').GetDOFIndex()] = -0.023593
                            v[self.robot.GetJoint(self.opposite_arm[0]+'_shoulder_lift_joint').GetDOFIndex()] = 1.1072800
                            v[self.robot.GetJoint(self.opposite_arm[0]+'_upper_arm_roll_joint').GetDOFIndex()] = -1.5566882
                            v[self.robot.GetJoint(self.opposite_arm[0]+'_elbow_flex_joint').GetDOFIndex()] = -2.124408
                            v[self.robot.GetJoint(self.opposite_arm[0]+'_forearm_roll_joint').GetDOFIndex()] = -1.4175
                            v[self.robot.GetJoint(self.opposite_arm[0]+'_wrist_flex_joint').GetDOFIndex()] = -1.8417
                            v[self.robot.GetJoint(self.opposite_arm[0]+'_wrist_roll_joint').GetDOFIndex()] = 0.21436
                            self.robot.SetActiveDOFValues(v, 2)
                            self.env.UpdatePublishedBodies()
                            # ikparam = op.IkParameterization(Tgrasp[0:3,3], self.ikmodel.iktype)
                            ikparam = op.IkParameterization(np.array([.8, -.5, 1.2]), self.ikmodel.iktype)
                            sols = self.manip.FindIKSolutions(ikparam, filteroptions=op.IkFilterOptions.CheckEnvCollisions)
                        '''

                        # manip[num] = 0.
                        # reached[num] = 0.
                        if list(sols):  # not None:
                            # print 'Number of sols:', len(sols)

                            for solution in sols:
                                # print solution
                                if False:  # m.degrees(solution[3])<-45:
                                    continue
                                else:
                                    reached[num] = 1.
                                    self.robot.SetDOFValues(solution, self.manip.GetArmIndices())
                                    self.env.UpdatePublishedBodies()

                                    J = np.matrix(np.vstack([self.manip.CalculateJacobian(), self.manip.CalculateAngularVelocityJacobian()]))
                                    try:
                                        joint_limit_weight = self.gen_joint_limit_weight(solution)
                                        manip[num] = np.max([copy.copy((m.pow(np.linalg.det(J*joint_limit_weight*J.T), (1./6.)))/(np.trace(J*joint_limit_weight*J.T)/6.)), manip[num]])
                                    except ValueError:
                                        print 'WARNING!!'
                                        print 'Jacobian may be singular or close to singular'
                                        print 'Determinant of J*JT is: ', np.linalg.det(J*J.T)
                                        manip[num] = np.max([0., manip[num]])
                            if self.visualize:
                                rospy.sleep(1.0)
                for num in xrange(len(reached)):
                    manip_score += copy.copy(reached[num] * manip[num]*self.weights[num])
                    reach_score += copy.copy(reached[num] * self.weights[num])
            else:
                # print 'In base collision! single config distance: ', distance
                if distance < 2.0:
                    return 10. + 1. + (1.3 - distance)

        # Set the weights for the different scores.
        beta = 10.  # Weight on number of reachable goals
        gamma = 1.  # Weight on manipulability of arm at each reachable goal
        zeta = .0007  # Weight on distance to move to get to that goal location
        if reach_score == 0.:
            return 10. + 2*random.random()
        else:
            # print 'Reach score: ', reach_score
            # print 'Manip score: ', manip_score
            return 10.-beta*reach_score-gamma*manip_score  # +zeta*self.distance

    def objective_function_two_configs(self, current_parameters):
        if not self.a_model_is_loaded:
            print 'Somehow a model has not been loaded. This is bad!'
            return None
        parameters = [[ 0.78062408,  0.53540329],
       [ 1.03958875, -0.89983494],
       [-3.11454592, -0.08683541],
       [ 0.2443988 ,  0.11760562],
       [ 0.03440107,  0.07498624],
       [ 0.50149404,  0.23139831]]
        current_parameters = [parameters[0][0], parameters[1][0], parameters[2][0],parameters[3][0],parameters[4][0], parameters[5][0],
                              parameters[0][1],parameters[1][1],parameters[2][1],parameters[3][1],parameters[4][1],parameters[5][1]]
        self.heady = 0.
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

                    if self.model == 'chair':
                        self.env.UpdatePublishedBodies()
                        headmodel = self.wheelchair.GetLink('wheelchair/head_link')
                        ual = self.wheelchair.GetLink('wheelchair/arm_left_link')
                        uar = self.wheelchair.GetLink('wheelchair/arm_right_link')
                        fal = self.wheelchair.GetLink('wheelchair/forearm_left_link')
                        far = self.wheelchair.GetLink('wheelchair/forearm_right_link')
                        footl = self.wheelchair.GetLink('wheelchair/quad_left_link')
                        footr = self.wheelchair.GetLink('wheelchair/quad_right_link')
                        kneel = self.wheelchair.GetLink('wheelchair/calf_left_link')
                        kneer = self.wheelchair.GetLink('wheelchair/calf_right_link')
                        ch = self.wheelchair.GetLink('wheelchair/upper_body_link')
                        origin_B_head = np.matrix(headmodel.GetTransform())
                        origin_B_ual = np.matrix(ual.GetTransform())
                        origin_B_uar = np.matrix(uar.GetTransform())
                        origin_B_fal = np.matrix(fal.GetTransform())
                        origin_B_far = np.matrix(far.GetTransform())
                        origin_B_footl = np.matrix(footl.GetTransform())
                        origin_B_footr = np.matrix(footr.GetTransform())
                        origin_B_kneel = np.matrix(kneel.GetTransform())
                        origin_B_kneer = np.matrix(kneer.GetTransform())
                        origin_B_ch = np.matrix(ch.GetTransform())
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
                            elif self.reference_names[thing] == 'foot_left':
                                self.origin_B_references.append(origin_B_footl)
                            elif self.reference_names[thing] == 'foot_right':
                                self.origin_B_references.append(origin_B_footr)
                            elif self.reference_names[thing] == 'knee_left':
                                self.origin_B_references.append(origin_B_kneel)
                            elif self.reference_names[thing] == 'knee_right':
                                self.origin_B_references.append(origin_B_kneer)
                            elif self.reference_names[thing] == 'chest':
                                self.origin_B_references.append(origin_B_ch)

                        thing = num
                        self.goal_list[0] = copy.copy(self.origin_B_references[int(self.reference_mat[thing])]*np.matrix(self.goals[thing, 0]))
                        self.selection_mat[0] = copy.copy(self.goals[thing, 1])

                        self.set_goals()
                        headmodel = self.wheelchair.GetLink('wheelchair/head_link')

                    elif self.model == 'autobed':
                        self.selection_mat = np.zeros(1)
                        self.goal_list = np.zeros([1, 4, 4])
                        self.set_autobed(bz[config_num], bth[config_num], self.headx, self.heady)
                        self.rotate_head_only(head_angle[0], head_angle[1])
                        self.env.UpdatePublishedBodies()

                        headmodel = self.autobed.GetLink('autobed/head_link')
                        ual = self.autobed.GetLink('autobed/upper_arm_left_link')
                        uar = self.autobed.GetLink('autobed/upper_arm_right_link')
                        fal = self.autobed.GetLink('autobed/fore_arm_left_link')
                        far = self.autobed.GetLink('autobed/fore_arm_right_link')
                        hal = self.autobed.GetLink('autobed/hand_left_link')
                        har = self.autobed.GetLink('autobed/hand_right_link')
                        footl = self.autobed.GetLink('autobed/foot_left_link')
                        footr = self.autobed.GetLink('autobed/foot_right_link')
                        kneel = self.autobed.GetLink('autobed/knee_left_link')
                        kneer = self.autobed.GetLink('autobed/knee_right_link')
                        ch = self.autobed.GetLink('autobed/torso_link')
                        origin_B_head = np.matrix(headmodel.GetTransform())
                        origin_B_ual = np.matrix(ual.GetTransform())
                        origin_B_uar = np.matrix(uar.GetTransform())
                        origin_B_fal = np.matrix(fal.GetTransform())
                        origin_B_far = np.matrix(far.GetTransform())
                        origin_B_hal = np.matrix(hal.GetTransform())
                        origin_B_har = np.matrix(har.GetTransform())
                        origin_B_footl = np.matrix(footl.GetTransform())
                        origin_B_footr = np.matrix(footr.GetTransform())
                        origin_B_kneel = np.matrix(kneel.GetTransform())
                        origin_B_kneer = np.matrix(kneer.GetTransform())
                        origin_B_ch = np.matrix(ch.GetTransform())
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
                        elif self.reference_names[thing] == 'hand_left':
                            self.origin_B_references.append(origin_B_hal)
                        elif self.reference_names[thing] == 'hand_right':
                            self.origin_B_references.append(origin_B_har)
                        elif self.reference_names[thing] == 'foot_left':
                            self.origin_B_references.append(origin_B_footl)
                        elif self.reference_names[thing] == 'foot_right':
                            self.origin_B_references.append(origin_B_footr)
                        elif self.reference_names[thing] == 'knee_left':
                            self.origin_B_references.append(origin_B_kneel)
                        elif self.reference_names[thing] == 'knee_right':
                            self.origin_B_references.append(origin_B_kneer)
                        elif self.reference_names[thing] == 'chest':
                            self.origin_B_references.append(origin_B_ch)
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
                    if this_distance < 1.3:
                        with self.robot:
                            v = self.robot.GetActiveDOFValues()
                            if self.arm[0] == 'l':
                                arm_sign = 1
                            else:
                                arm_sign = -1
                            if self.task == 'blanket_feet_knees' or self.task == 'scratching_knee_left' or True:
                                v[self.robot.GetJoint(
                                    self.arm[0] + '_shoulder_pan_joint').GetDOFIndex()] = arm_sign * 3. * 3.14159 / 4.
                                v[self.robot.GetJoint(self.arm[0] + '_shoulder_lift_joint').GetDOFIndex()] = -0.6
                                v[self.robot.GetJoint(
                                    self.arm[0] + '_upper_arm_roll_joint').GetDOFIndex()] = arm_sign * m.radians(20)
                                v[self.robot.GetJoint(self.arm[0] + '_elbow_flex_joint').GetDOFIndex()] = m.radians(
                                    -150.)
                                v[self.robot.GetJoint(self.arm[0] + '_forearm_roll_joint').GetDOFIndex()] = m.radians(
                                    150.)
                                v[self.robot.GetJoint(self.arm[0] + '_wrist_flex_joint').GetDOFIndex()] = m.radians(
                                    -110)
                                v[self.robot.GetJoint(self.arm[0] + '_wrist_roll_joint').GetDOFIndex()] = arm_sign * 0.0
                            elif self.task == 'wiping_mouth' or self.task == 'wiping_forehead':
                                v[self.robot.GetJoint(self.arm[0] + '_shoulder_pan_joint').GetDOFIndex()] = arm_sign * (
                                    1.8)
                                v[self.robot.GetJoint(self.arm[0] + '_shoulder_lift_joint').GetDOFIndex()] = 0.4
                                v[self.robot.GetJoint(
                                    self.arm[0] + '_upper_arm_roll_joint').GetDOFIndex()] = arm_sign * (1.9)
                                v[self.robot.GetJoint(self.arm[0] + '_elbow_flex_joint').GetDOFIndex()] = -3.0
                                v[self.robot.GetJoint(self.arm[0] + '_forearm_roll_joint').GetDOFIndex()] = arm_sign * (
                                    -3.5)
                                v[self.robot.GetJoint(self.arm[0] + '_wrist_flex_joint').GetDOFIndex()] = -0.5
                                v[self.robot.GetJoint(self.arm[0] + '_wrist_roll_joint').GetDOFIndex()] = 0.0
                            else:
                                print 'The arm initial pose is not defined properly.'
                                v[self.robot.GetJoint('I HAVE NO IDEA WHAT TASK Im DOING').GetDOFIndex()] = 0.
                            v[self.robot.GetJoint(
                                self.opposite_arm[0] + '_shoulder_pan_joint').GetDOFIndex()] = arm_sign * (-1.8)
                            v[self.robot.GetJoint(self.opposite_arm[0] + '_shoulder_lift_joint').GetDOFIndex()] = 2.45
                            v[self.robot.GetJoint(
                                self.opposite_arm[0] + '_upper_arm_roll_joint').GetDOFIndex()] = arm_sign * (-1.9)
                            v[self.robot.GetJoint(self.opposite_arm[0] + '_elbow_flex_joint').GetDOFIndex()] = -2.0
                            v[self.robot.GetJoint(
                                self.opposite_arm[0] + '_forearm_roll_joint').GetDOFIndex()] = arm_sign * 3.5
                            v[self.robot.GetJoint(self.opposite_arm[0] + '_wrist_flex_joint').GetDOFIndex()] = -1.5
                            v[self.robot.GetJoint(self.opposite_arm[0] + '_wrist_roll_joint').GetDOFIndex()] = 0.0
                            self.robot.SetActiveDOFValues(v, 2)
                            self.env.UpdatePublishedBodies()
                            not_close_to_collision = True
                            if self.env.CheckCollision(self.robot):
                                not_close_to_collision = False
                            '''
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
                            '''
                            if not_close_to_collision:
                                Tgrasp = self.origin_B_grasps[0]

                                # print 'no collision!'
                                # for num, Tgrasp in enumerate(self.origin_B_grasps):
                                    # sol = None
                                    # sol = self.manip.FindIKSolution(Tgrasp, filteroptions=op.IkFilterOptions.CheckEnvCollisions)

                                    #sol = self.manip.FindIKSolution(Tgrasp,filteroptions=op.IkFilterOptions.IgnoreSelfCollisions)
                                sols = []
                                sols = self.manip.FindIKSolutions(Tgrasp, filteroptions=op.IkFilterOptions.CheckEnvCollisions)
                                '''
                                if not list(sols):
                                    v[self.robot.GetJoint(self.opposite_arm[0]+'_shoulder_pan_joint').GetDOFIndex()] = -0.023593
                                    v[self.robot.GetJoint(self.opposite_arm[0]+'_shoulder_lift_joint').GetDOFIndex()] = 1.1072800
                                    v[self.robot.GetJoint(self.opposite_arm[0]+'_upper_arm_roll_joint').GetDOFIndex()] = -1.5566882
                                    v[self.robot.GetJoint(self.opposite_arm[0]+'_elbow_flex_joint').GetDOFIndex()] = -2.124408
                                    v[self.robot.GetJoint(self.opposite_arm[0]+'_forearm_roll_joint').GetDOFIndex()] = -1.4175
                                    v[self.robot.GetJoint(self.opposite_arm[0]+'_wrist_flex_joint').GetDOFIndex()] = -1.8417
                                    v[self.robot.GetJoint(self.opposite_arm[0]+'_wrist_roll_joint').GetDOFIndex()] = 0.21436
                                    self.robot.SetActiveDOFValues(v, 2)
                                    self.env.UpdatePublishedBodies()
                                    sols = self.manip.FindIKSolutions(Tgrasp, filteroptions=op.IkFilterOptions.CheckEnvCollisions)
                                '''
                                if list(sols):  # not None:
                                    # print 'I got a solution!!'
                                    # print 'sol is:', sol
                                    # print 'sols are: \n', sols
                                    #print 'I was able to find a grasp to this goal'
                                    reached[num, config_num] = 1
                                    for solution in sols:
                                        # print solution
                                        if False: #m.degrees(solution[3]) < -45.:
                                            print m.degrees(solution[3])
                                            continue
                                        self.robot.SetDOFValues(solution, self.manip.GetArmIndices())
                                        # Tee = self.manip.GetEndEffectorTransform()
                                        self.env.UpdatePublishedBodies()

                                        J = np.matrix(np.vstack([self.manip.CalculateJacobian(), self.manip.CalculateAngularVelocityJacobian()]))
                                        try:
                                            joint_limit_weight = self.gen_joint_limit_weight(solution)
                                            manip[num, config_num] = np.max([copy.copy((m.pow(np.linalg.det(J*joint_limit_weight*J.T), (1./6.)))/(np.trace(J*joint_limit_weight*J.T)/6.)), manip[num, config_num]])
                                        except ValueError:
                                            print 'WARNING!!'
                                            print 'Jacobian may be singular or close to singular'
                                            print 'Determinant of J*JT is: ', np.linalg.det(J*J.T)
                                            manip[num, config_num] = np.max([0., manip[num, config_num]])
                                    if self.visualize:
                                        rospy.sleep(1.)
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
                    return 10. + 1. + (1.3 - np.min(distance))
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
            if dist >= 1.3:
                over_dist += 2*(dist - 1.3)
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
            print 'reach score is too low', reach_score
            print 'manip score is too low', manip_score
            #print 'travel score', travel_score
            print outputs
        if outputs[best] < -1.0:
            print 'reach score is too low', reach_score
            print 'manip score is too low', manip_score
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
                        footl = self.wheelchair.GetLink('wheelchair/quad_left_link')
                        footr = self.wheelchair.GetLink('wheelchair/quad_right_link')
                        kneel = self.wheelchair.GetLink('wheelchair/calf_left_link')
                        kneer = self.wheelchair.GetLink('wheelchair/calf_right_link')
                        ch = self.wheelchair.GetLink('wheelchair/upper_body_link')
                        origin_B_head = np.matrix(headmodel.GetTransform())
                        origin_B_ual = np.matrix(ual.GetTransform())
                        origin_B_uar = np.matrix(uar.GetTransform())
                        origin_B_fal = np.matrix(fal.GetTransform())
                        origin_B_far = np.matrix(far.GetTransform())
                        origin_B_footl = np.matrix(footl.GetTransform())
                        origin_B_footr = np.matrix(footr.GetTransform())
                        origin_B_kneel = np.matrix(kneel.GetTransform())
                        origin_B_kneer = np.matrix(kneer.GetTransform())
                        origin_B_ch = np.matrix(ch.GetTransform())
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
                            elif self.reference_names[thing] == 'foot_left':
                                self.origin_B_references.append(origin_B_footl)
                            elif self.reference_names[thing] == 'foot_right':
                                self.origin_B_references.append(origin_B_footr)
                            elif self.reference_names[thing] == 'knee_left':
                                self.origin_B_references.append(origin_B_kneel)
                            elif self.reference_names[thing] == 'knee_right':
                                self.origin_B_references.append(origin_B_kneer)
                            elif self.reference_names[thing] == 'chest':
                                self.origin_B_references.append(origin_B_ch)

                        thing = num
                        self.goal_list[0] = copy.copy(self.origin_B_references[int(self.reference_mat[thing])]*np.matrix(self.goals[thing, 0]))
                        self.selection_mat[0] = copy.copy(self.goals[thing, 1])

                        self.set_goals()
                        headmodel = self.wheelchair.GetLink('wheelchair/head_link')

                    elif self.model == 'autobed':
                        self.selection_mat = np.zeros(1)
                        self.goal_list = np.zeros([1, 4, 4])
                        self.set_autobed(bz[config_num], bth[config_num], self.headx, self.heady)
                        self.rotate_head_only(head_angle[0], head_angle[1])
                        self.env.UpdatePublishedBodies()

                        headmodel = self.autobed.GetLink('autobed/head_link')
                        ual = self.autobed.GetLink('autobed/upper_arm_left_link')
                        uar = self.autobed.GetLink('autobed/upper_arm_right_link')
                        fal = self.autobed.GetLink('autobed/fore_arm_left_link')
                        far = self.autobed.GetLink('autobed/fore_arm_right_link')
                        hal = self.autobed.GetLink('autobed/hand_left_link')
                        har = self.autobed.GetLink('autobed/hand_right_link')
                        footl = self.autobed.GetLink('autobed/foot_left_link')
                        footr = self.autobed.GetLink('autobed/foot_right_link')
                        kneel = self.autobed.GetLink('autobed/knee_left_link')
                        kneer = self.autobed.GetLink('autobed/knee_right_link')
                        ch = self.autobed.GetLink('autobed/torso_link')
                        origin_B_head = np.matrix(headmodel.GetTransform())
                        origin_B_ual = np.matrix(ual.GetTransform())
                        origin_B_uar = np.matrix(uar.GetTransform())
                        origin_B_fal = np.matrix(fal.GetTransform())
                        origin_B_far = np.matrix(far.GetTransform())
                        origin_B_hal = np.matrix(hal.GetTransform())
                        origin_B_har = np.matrix(har.GetTransform())
                        origin_B_footl = np.matrix(footl.GetTransform())
                        origin_B_footr = np.matrix(footr.GetTransform())
                        origin_B_kneel = np.matrix(kneel.GetTransform())
                        origin_B_kneer = np.matrix(kneer.GetTransform())
                        origin_B_ch = np.matrix(ch.GetTransform())
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
                        elif self.reference_names[thing] == 'hand_left':
                            self.origin_B_references.append(origin_B_hal)
                        elif self.reference_names[thing] == 'hand_right':
                            self.origin_B_references.append(origin_B_har)
                        elif self.reference_names[thing] == 'foot_left':
                            self.origin_B_references.append(origin_B_footl)
                        elif self.reference_names[thing] == 'foot_right':
                            self.origin_B_references.append(origin_B_footr)
                        elif self.reference_names[thing] == 'knee_left':
                            self.origin_B_references.append(origin_B_kneel)
                        elif self.reference_names[thing] == 'knee_right':
                            self.origin_B_references.append(origin_B_kneer)
                        elif self.reference_names[thing] == 'chest':
                            self.origin_B_references.append(origin_B_ch)
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
                    if this_distance < 1.3:
                        with self.robot:
                            v = self.robot.GetActiveDOFValues()
                            if self.arm[0] == 'l':
                                arm_sign = 1
                            else:
                                arm_sign = -1
                            if self.task == 'blanket_feet_knees' or self.task == 'scratching_knee_left' or True:
                                v[self.robot.GetJoint(
                                    self.arm[0] + '_shoulder_pan_joint').GetDOFIndex()] = arm_sign * 3. * 3.14159 / 4.
                                v[self.robot.GetJoint(self.arm[0] + '_shoulder_lift_joint').GetDOFIndex()] = -0.6
                                v[self.robot.GetJoint(
                                    self.arm[0] + '_upper_arm_roll_joint').GetDOFIndex()] = arm_sign * m.radians(20)
                                v[self.robot.GetJoint(self.arm[0] + '_elbow_flex_joint').GetDOFIndex()] = m.radians(
                                    -150.)
                                v[self.robot.GetJoint(self.arm[0] + '_forearm_roll_joint').GetDOFIndex()] = m.radians(
                                    150.)
                                v[self.robot.GetJoint(self.arm[0] + '_wrist_flex_joint').GetDOFIndex()] = m.radians(
                                    -110)
                                v[self.robot.GetJoint(self.arm[0] + '_wrist_roll_joint').GetDOFIndex()] = arm_sign * 0.0
                            elif self.task == 'wiping_mouth' or self.task == 'wiping_forehead':
                                v[self.robot.GetJoint(self.arm[0] + '_shoulder_pan_joint').GetDOFIndex()] = arm_sign * (
                                    1.8)
                                v[self.robot.GetJoint(self.arm[0] + '_shoulder_lift_joint').GetDOFIndex()] = 0.4
                                v[self.robot.GetJoint(
                                    self.arm[0] + '_upper_arm_roll_joint').GetDOFIndex()] = arm_sign * (1.9)
                                v[self.robot.GetJoint(self.arm[0] + '_elbow_flex_joint').GetDOFIndex()] = -3.0
                                v[self.robot.GetJoint(self.arm[0] + '_forearm_roll_joint').GetDOFIndex()] = arm_sign * (
                                    -3.5)
                                v[self.robot.GetJoint(self.arm[0] + '_wrist_flex_joint').GetDOFIndex()] = -0.5
                                v[self.robot.GetJoint(self.arm[0] + '_wrist_roll_joint').GetDOFIndex()] = 0.0
                            else:
                                print 'The arm initial pose is not defined properly.'
                                v[self.robot.GetJoint('I HAVE NO IDEA WHAT TASK Im DOING').GetDOFIndex()] = 0.
                            v[self.robot.GetJoint(
                                self.opposite_arm[0] + '_shoulder_pan_joint').GetDOFIndex()] = arm_sign * (-1.8)
                            v[self.robot.GetJoint(self.opposite_arm[0] + '_shoulder_lift_joint').GetDOFIndex()] = 2.45
                            v[self.robot.GetJoint(
                                self.opposite_arm[0] + '_upper_arm_roll_joint').GetDOFIndex()] = arm_sign * (-1.9)
                            v[self.robot.GetJoint(self.opposite_arm[0] + '_elbow_flex_joint').GetDOFIndex()] = -2.0
                            v[self.robot.GetJoint(
                                self.opposite_arm[0] + '_forearm_roll_joint').GetDOFIndex()] = arm_sign * 3.5
                            v[self.robot.GetJoint(self.opposite_arm[0] + '_wrist_flex_joint').GetDOFIndex()] = -1.5
                            v[self.robot.GetJoint(self.opposite_arm[0] + '_wrist_roll_joint').GetDOFIndex()] = 0.0
                            self.robot.SetActiveDOFValues(v, 2)
                            self.env.UpdatePublishedBodies()
                            not_close_to_collision = True
                            if self.env.CheckCollision(self.robot):
                                not_close_to_collision = False
                            if not_close_to_collision:
                                Tgrasp = self.origin_B_grasps[0]

                                # print 'no collision!'
                                # for num, Tgrasp in enumerate(self.origin_B_grasps):
                                    # sol = None
                                    # sol = self.manip.FindIKSolution(Tgrasp, filteroptions=op.IkFilterOptions.CheckEnvCollisions)

                                    #sol = self.manip.FindIKSolution(Tgrasp,filteroptions=op.IkFilterOptions.IgnoreSelfCollisions)
                                sols = []
                                sols = self.manip.FindIKSolutions(Tgrasp, filteroptions=op.IkFilterOptions.CheckEnvCollisions)
                                '''
                                if not list(sols):
                                    v[self.robot.GetJoint(self.opposite_arm[0]+'_shoulder_pan_joint').GetDOFIndex()] = -0.023593
                                    v[self.robot.GetJoint(self.opposite_arm[0]+'_shoulder_lift_joint').GetDOFIndex()] = 1.1072800
                                    v[self.robot.GetJoint(self.opposite_arm[0]+'_upper_arm_roll_joint').GetDOFIndex()] = -1.5566882
                                    v[self.robot.GetJoint(self.opposite_arm[0]+'_elbow_flex_joint').GetDOFIndex()] = -2.124408
                                    v[self.robot.GetJoint(self.opposite_arm[0]+'_forearm_roll_joint').GetDOFIndex()] = -1.4175
                                    v[self.robot.GetJoint(self.opposite_arm[0]+'_wrist_flex_joint').GetDOFIndex()] = -1.8417
                                    v[self.robot.GetJoint(self.opposite_arm[0]+'_wrist_roll_joint').GetDOFIndex()] = 0.21436
                                    self.robot.SetActiveDOFValues(v, 2)
                                    sols = self.manip.FindIKSolutions(Tgrasp, filteroptions=op.IkFilterOptions.CheckEnvCollisions)
                                '''
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

                                        J = np.matrix(np.vstack([self.manip.CalculateJacobian(), self.manip.CalculateAngularVelocityJacobian()]))
                                        try:
                                            joint_limit_weight = self.gen_joint_limit_weight(solution)
                                            manip[num, config_num] = np.max([copy.copy((m.pow(np.linalg.det(J*joint_limit_weight*J.T), (1./6.)))/(np.trace(J*joint_limit_weight*J.T)/6.)), manip[num, config_num]])
                                        except ValueError:
                                            print 'WARNING!!'
                                            print 'Jacobian may be singular or close to singular'
                                            print 'Determinant of J*JT is: ', np.linalg.det(J*J.T)
                                            manip[num, config_num] = np.max([0., manip[num, config_num]])
                                    if self.visualize:
                                        rospy.sleep(0.2)
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

    def objective_function_real_time(self, current_parameters):
        if not self.a_model_is_loaded:
            print 'Somehow a model has not been loaded. This is bad!'
            return None
        x = current_parameters[0]
        y = current_parameters[1]
        th = current_parameters[2]
        z = current_parameters[3]

        #print 'Calculating new score'
        #starttime = time.time()
        origin_B_pr2 = np.matrix([[ m.cos(th), -m.sin(th),     0.,         x],
                                  [ m.sin(th),  m.cos(th),     0.,         y],
                                  [        0.,         0.,     1.,        0.],
                                  [        0.,         0.,     0.,        1.]])
        self.robot.SetTransform(np.array(origin_B_pr2))
        v = self.robot.GetActiveDOFValues()
        v[self.robot.GetJoint('torso_lift_joint').GetDOFIndex()] = z
        self.robot.SetActiveDOFValues(v, 2)

        self.env.UpdatePublishedBodies()

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
            v = self.robot.GetActiveDOFValues()
            v[self.robot.GetJoint(self.opposite_arm[0]+'_shoulder_pan_joint').GetDOFIndex()] = -3.14/2
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

            if not_close_to_collision:
                # print 'No base collision! single config distance: ', distance
                # reached = np.zeros(len(self.origin_B_grasps))
                # manip = np.zeros(len(self.origin_B_grasps))

                for num, Tgrasp in enumerate(self.origin_B_grasps):
                    sols = []
                    sols = self.manip.FindIKSolutions(Tgrasp, filteroptions=op.IkFilterOptions.CheckEnvCollisions)
                    if not list(sols):
                        v = self.robot.GetActiveDOFValues()
                        v[self.robot.GetJoint(self.opposite_arm[0]+'_shoulder_pan_joint').GetDOFIndex()] = -0.023593
                        v[self.robot.GetJoint(self.opposite_arm[0]+'_shoulder_lift_joint').GetDOFIndex()] = 1.1072800
                        v[self.robot.GetJoint(self.opposite_arm[0]+'_upper_arm_roll_joint').GetDOFIndex()] = -1.5566882
                        v[self.robot.GetJoint(self.opposite_arm[0]+'_elbow_flex_joint').GetDOFIndex()] = -2.124408
                        v[self.robot.GetJoint(self.opposite_arm[0]+'_forearm_roll_joint').GetDOFIndex()] = -1.4175
                        v[self.robot.GetJoint(self.opposite_arm[0]+'_wrist_flex_joint').GetDOFIndex()] = -1.8417
                        v[self.robot.GetJoint(self.opposite_arm[0]+'_wrist_roll_joint').GetDOFIndex()] = 0.21436
                        self.robot.SetActiveDOFValues(v, 2)
                        self.env.UpdatePublishedBodies()
                        sols = self.manip.FindIKSolutions(Tgrasp, filteroptions=op.IkFilterOptions.CheckEnvCollisions)

                    # manip[num] = 0.
                    # reached[num] = 0.
                    if list(sols):  # not None:

                        reached = 1.
                        for solution in sols:
                            self.robot.SetDOFValues(solution, self.manip.GetArmIndices())
                            self.env.UpdatePublishedBodies()
                            if self.visualize:
                                rospy.sleep(0.5)
                            J = np.matrix(np.vstack([self.manip.CalculateJacobian(), self.manip.CalculateAngularVelocityJacobian()]))
                            try:
                                joint_limit_weight = self.gen_joint_limit_weight(solution)
                                manip = np.max([copy.copy((m.pow(np.linalg.det(J*joint_limit_weight*J.T), (1./6.)))/(np.trace(J*joint_limit_weight*J.T)/6.)), manip])
                            except ValueError:
                                print 'WARNING!!'
                                print 'Jacobian may be singular or close to singular'
                                print 'Determinant of J*JT is: ', np.linalg.det(J*J.T)
                                manip = np.max([0., manip])
                manip_score += copy.copy(reached * manip*self.weights[num])
                reach_score += copy.copy(reached * self.weights[num])
            else:
                # print 'In base collision! single config distance: ', distance
                if distance < 2.0:
                    return 10. + 1. + (1.25 - distance)

        # Set the weights for the different scores.
        beta = 10.  # Weight on number of reachable goals
        gamma = 1.  # Weight on manipulability of arm at each reachable goal
        zeta = .0007  # Weight on distance to move to get to that goal location
        if reach_score == 0.:
            return 10. + 2*random.random()
        else:
            # print 'Reach score: ', reach_score
            # print 'Manip score: ', manip_score
            return 10.-beta*reach_score-gamma*manip_score  # +zeta*self.distance

    def rotate_head_only(self, neck_rotation, head_rotation):
        with self.env:
            if self.model == 'chair':
                v = self.wheelchair.GetActiveDOFValues()
                v[self.wheelchair.GetJoint('wheelchair/neck_twist_joint').GetDOFIndex()] = m.radians(neck_rotation)
                v[self.wheelchair.GetJoint('wheelchair/neck_head_rotz_joint').GetDOFIndex()] = m.radians(head_rotation)
                self.wheelchair.SetActiveDOFValues(v)
                self.env.UpdatePublishedBodies()
            elif self.model == 'autobed':
                v = self.autobed.GetActiveDOFValues()
                v[self.autobed.GetJoint('autobed/neck_twist_joint').GetDOFIndex()] = m.radians(neck_rotation)
                v[self.autobed.GetJoint('autobed/neck_head_rotz_joint').GetDOFIndex()] = m.radians(head_rotation)
                self.autobed.SetActiveDOFValues(v)
                self.env.UpdatePublishedBodies()

    def rotate_head_and_update_goals(self, neck_rotation, head_rotation, current_origin_B_pr2):
        with self.env:
            origin_B_pr2 = current_origin_B_pr2
            if self.model == 'chair':
                v = self.wheelchair.GetActiveDOFValues()
                v[self.wheelchair.GetJoint('wheelchair/neck_twist_joint').GetDOFIndex()] = m.radians(neck_rotation)
                v[self.wheelchair.GetJoint('wheelchair/neck_head_rotz_joint').GetDOFIndex()] = m.radians(head_rotation)
                self.wheelchair.SetActiveDOFValues(v)
                self.env.UpdatePublishedBodies()

                self.env.UpdatePublishedBodies()
                headmodel = self.wheelchair.GetLink('wheelchair/head_link')
                ual = self.wheelchair.GetLink('wheelchair/arm_left_link')
                uar = self.wheelchair.GetLink('wheelchair/arm_right_link')
                fal = self.wheelchair.GetLink('wheelchair/forearm_left_link')
                far = self.wheelchair.GetLink('wheelchair/forearm_right_link')
                footl = self.wheelchair.GetLink('wheelchair/quad_left_link')
                footr = self.wheelchair.GetLink('wheelchair/quad_right_link')
                kneel = self.wheelchair.GetLink('wheelchair/calf_left_link')
                kneer = self.wheelchair.GetLink('wheelchair/calf_right_link')
                ch = self.wheelchair.GetLink('wheelchair/upper_body_link')
                origin_B_head = np.matrix(headmodel.GetTransform())
                origin_B_ual = np.matrix(ual.GetTransform())
                origin_B_uar = np.matrix(uar.GetTransform())
                origin_B_fal = np.matrix(fal.GetTransform())
                origin_B_far = np.matrix(far.GetTransform())
                origin_B_footl = np.matrix(footl.GetTransform())
                origin_B_footr = np.matrix(footr.GetTransform())
                origin_B_kneel = np.matrix(kneel.GetTransform())
                origin_B_kneer = np.matrix(kneer.GetTransform())
                origin_B_ch = np.matrix(ch.GetTransform())
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
                    elif self.reference_names[thing] == 'foot_left':
                        self.origin_B_references.append(origin_B_footl)
                    elif self.reference_names[thing] == 'foot_right':
                        self.origin_B_references.append(origin_B_footr)
                    elif self.reference_names[thing] == 'knee_left':
                        self.origin_B_references.append(origin_B_kneel)
                    elif self.reference_names[thing] == 'knee_right':
                        self.origin_B_references.append(origin_B_kneer)
                    elif self.reference_names[thing] == 'chest':
                        self.origin_B_references.append(origin_B_ch)
                for thing in xrange(len(self.goals)):
                    self.goal_list[thing] = copy.copy(self.origin_B_references[int(self.reference_mat[thing])]*np.matrix(self.goals[thing, 0]))
                    self.selection_mat[thing] = self.goals[thing, 1]
    #            for target in self.goals:
    #                self.goal_list.append(pr2_B_head*np.matrix(target[0]))
    #                self.selection_mat.append(target[1])
                self.set_goals()
                headmodel = self.wheelchair.GetLink('wheelchair/head_link')

            elif self.model == 'autobed':
                self.selection_mat = np.zeros(len(self.goals))
                self.goal_list = np.zeros([len(self.goals), 4, 4])
                # self.set_autobed(bz, bth, self.headx, self.heady)
                v = self.autobed.GetActiveDOFValues()
                v[self.autobed.GetJoint('autobed/neck_twist_joint').GetDOFIndex()] = m.radians(neck_rotation)
                v[self.autobed.GetJoint('autobed/neck_head_rotz_joint').GetDOFIndex()] = m.radians(head_rotation)
                self.autobed.SetActiveDOFValues(v)
                self.env.UpdatePublishedBodies()

                headmodel = self.autobed.GetLink('autobed/head_link')
                ual = self.autobed.GetLink('autobed/upper_arm_left_link')
                uar = self.autobed.GetLink('autobed/upper_arm_right_link')
                fal = self.autobed.GetLink('autobed/fore_arm_left_link')
                far = self.autobed.GetLink('autobed/fore_arm_right_link')
                footl = self.autobed.GetLink('autobed/foot_left_link')
                footr = self.autobed.GetLink('autobed/foot_right_link')
                kneel = self.autobed.GetLink('autobed/knee_left_link')
                kneer = self.autobed.GetLink('autobed/knee_right_link')
                ch = self.autobed.GetLink('autobed/torso_link')
                origin_B_head = np.matrix(headmodel.GetTransform())
                origin_B_ual = np.matrix(ual.GetTransform())
                origin_B_uar = np.matrix(uar.GetTransform())
                origin_B_fal = np.matrix(fal.GetTransform())
                origin_B_far = np.matrix(far.GetTransform())
                origin_B_footl = np.matrix(footl.GetTransform())
                origin_B_footr = np.matrix(footr.GetTransform())
                origin_B_kneel = np.matrix(kneel.GetTransform())
                origin_B_kneer = np.matrix(kneer.GetTransform())
                origin_B_ch = np.matrix(ch.GetTransform())
                self.origin_B_references = []
                for thing in xrange(len(self.reference_names)):
                    if self.reference_names[thing] == 'head':
                        self.origin_B_references.append(origin_B_head)
                        # self.origin_B_references.append(np.matrix(headmodel.GetTransform())
                    elif self.reference_names[thing] == 'base_link':
                        self.origin_B_references.append(origin_B_pr2)
                        # self.origin_B_references[i] = np.matrix(self.robot.GetTransform())
                    elif self.reference_names[thing] == 'upper_arm_left':
                        self.origin_B_references.append(origin_B_ual)
                    elif self.reference_names[thing] == 'upper_arm_right':
                        self.origin_B_references.append(origin_B_uar)
                    elif self.reference_names[thing] == 'forearm_left':
                        self.origin_B_references.append(origin_B_fal)
                    elif self.reference_names[thing] == 'forearm_right':
                        self.origin_B_references.append(origin_B_far)
                    elif self.reference_names[thing] == 'foot_left':
                        self.origin_B_references.append(origin_B_footl)
                    elif self.reference_names[thing] == 'foot_right':
                        self.origin_B_references.append(origin_B_footr)
                    elif self.reference_names[thing] == 'knee_left':
                        self.origin_B_references.append(origin_B_kneel)
                    elif self.reference_names[thing] == 'knee_right':
                        self.origin_B_references.append(origin_B_kneer)
                    elif self.reference_names[thing] == 'chest':
                        self.origin_B_references.append(origin_B_ch)

                for thing in xrange(len(self.goals)):
                    self.goal_list[thing] = copy.copy(self.origin_B_references[int(self.reference_mat[thing])]*np.matrix(self.goals[thing, 0]))
                    self.selection_mat[thing] = self.goals[thing, 1]
                # for target in self.goals:
                #     self.goal_list.append(pr2_B_head*np.matrix(target[0]))
                #     self.selection_mat.append(target[1])
                self.set_goals()
            else:
                print 'I GOT A BAD MODEL. NOT SURE WHAT TO DO NOW!'

    def setup_openrave(self):
        # Setup Openrave ENV
        InitOpenRAVELogging()
        self.env = op.Environment()

        # Lets you visualize openrave. Uncomment to see visualization. Does not work footrough ssh.
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
        v = self.robot.GetActiveDOFValues()
        # v[self.robot.GetJoint(self.arm[0]+'_shoulder_pan_joint').GetDOFIndex()] = 3.14/2
        # v[self.robot.GetJoint(self.opposite_arm[0]+'_shoulder_pan_joint').GetDOFIndex()] = -3.14/2
        # v[self.robot.GetJoint(self.opposite_arm[0]+'_shoulder_lift_joint').GetDOFIndex()] = -0.52
        # v[self.robot.GetJoint(self.opposite_arm[0]+'_upper_arm_roll_joint').GetDOFIndex()] = 0.
        # v[self.robot.GetJoint(self.opposite_arm[0]+'_elbow_flex_joint').GetDOFIndex()] = -3.14*2/3
        # v[self.robot.GetJoint(self.opposite_arm[0]+'_forearm_roll_joint').GetDOFIndex()] = 0.
        # v[self.robot.GetJoint(self.opposite_arm[0]+'_wrist_flex_joint').GetDOFIndex()] = 0.
        # v[self.robot.GetJoint(self.opposite_arm[0]+'_wrist_roll_joint').GetDOFIndex()] = 0.
        # v[self.robot.GetJoint(self.arm[0]+'_gripper_l_finger_joint').GetDOFIndex()] = .1

        if self.arm[0] == 'l':
            arm_sign = 1
        else:
            arm_sign = -1
        if self.task == 'blanket_feet_knees' or self.task == 'scratching_knee_left' or True:
            v[self.robot.GetJoint(self.arm[0]+'_shoulder_pan_joint').GetDOFIndex()] = arm_sign*3.*3.14159/4.
            v[self.robot.GetJoint(self.arm[0]+'_shoulder_lift_joint').GetDOFIndex()] = -0.6
            v[self.robot.GetJoint(self.arm[0]+'_upper_arm_roll_joint').GetDOFIndex()] = arm_sign*m.radians(20)
            v[self.robot.GetJoint(self.arm[0]+'_elbow_flex_joint').GetDOFIndex()] = m.radians(-150.)
            v[self.robot.GetJoint(self.arm[0]+'_forearm_roll_joint').GetDOFIndex()] = m.radians(150.)
            v[self.robot.GetJoint(self.arm[0]+'_wrist_flex_joint').GetDOFIndex()] = m.radians(-110)
            v[self.robot.GetJoint(self.arm[0]+'_wrist_roll_joint').GetDOFIndex()] = arm_sign*0.0
        elif self.task == 'wiping_mouth' or self.task == 'wiping_forehead':
            v[self.robot.GetJoint(self.arm[0]+'_shoulder_pan_joint').GetDOFIndex()] = arm_sign*(1.8)
            v[self.robot.GetJoint(self.arm[0]+'_shoulder_lift_joint').GetDOFIndex()] = 0.4
            v[self.robot.GetJoint(self.arm[0]+'_upper_arm_roll_joint').GetDOFIndex()] = arm_sign*(1.9)
            v[self.robot.GetJoint(self.arm[0]+'_elbow_flex_joint').GetDOFIndex()] = -3.0
            v[self.robot.GetJoint(self.arm[0]+'_forearm_roll_joint').GetDOFIndex()] = arm_sign*(-3.5)
            v[self.robot.GetJoint(self.arm[0]+'_wrist_flex_joint').GetDOFIndex()] = -0.5
            v[self.robot.GetJoint(self.arm[0]+'_wrist_roll_joint').GetDOFIndex()] = 0.0
            # v[self.robot.GetJoint(self.arm[0]+'_shoulder_pan_joint').GetDOFIndex()] = arm_sign*0.8
            # v[self.robot.GetJoint(self.arm[0]+'_shoulder_lift_joint').GetDOFIndex()] = 0.0
            # v[self.robot.GetJoint(self.arm[0]+'_upper_arm_roll_joint').GetDOFIndex()] = arm_sign*1.57
            # v[self.robot.GetJoint(self.arm[0]+'_elbow_flex_joint').GetDOFIndex()] = -2.9
            # v[self.robot.GetJoint(self.arm[0]+'_forearm_roll_joint').GetDOFIndex()] = 3.0
            # v[self.robot.GetJoint(self.arm[0]+'_wrist_flex_joint').GetDOFIndex()] = -1.0
            # v[self.robot.GetJoint(self.arm[0]+'_wrist_roll_joint').GetDOFIndex()] = arm_sign*1.57
        else:
            print 'The arm initial pose is not defined properly.'
            v[self.robot.GetJoint('I HAVE NO IDEA WHAT TASK Im DOING').GetDOFIndex()] = 0.
        v[self.robot.GetJoint(self.opposite_arm[0]+'_shoulder_pan_joint').GetDOFIndex()] = arm_sign*(-1.8)
        v[self.robot.GetJoint(self.opposite_arm[0]+'_shoulder_lift_joint').GetDOFIndex()] = 2.45
        v[self.robot.GetJoint(self.opposite_arm[0]+'_upper_arm_roll_joint').GetDOFIndex()] = arm_sign*(-1.9)
        v[self.robot.GetJoint(self.opposite_arm[0]+'_elbow_flex_joint').GetDOFIndex()] = -2.0
        v[self.robot.GetJoint(self.opposite_arm[0]+'_forearm_roll_joint').GetDOFIndex()] = arm_sign*3.5
        v[self.robot.GetJoint(self.opposite_arm[0]+'_wrist_flex_joint').GetDOFIndex()] = -1.5
        v[self.robot.GetJoint(self.opposite_arm[0]+'_wrist_roll_joint').GetDOFIndex()] = 0.0

        # v[self.robot.GetJoint(self.arm[0]+'_gripper_r_finger_joint').GetDOFIndex()] = .54
        v[self.robot.GetJoint(self.arm[0]+'_gripper_l_finger_joint').GetDOFIndex()] = .1
        v[self.robot.GetJoint('torso_lift_joint').GetDOFIndex()] = 0.0
        self.robot.SetActiveDOFValues(v, 2)
        robot_start = np.matrix([[m.cos(0.), -m.sin(0.), 0., 0.],
                                 [m.sin(0.),  m.cos(0.), 0., 0.],
                                 [0.       ,         0., 1., 0.],
                                 [0.       ,         0., 0., 1.]])
        self.robot.SetTransform(np.array(robot_start))

        ## Set robot manipulators, ik, planner
        self.arm = 'leftarm'
        self.robot.SetActiveManipulator(self.arm)

        self.manip = self.robot.GetActiveManipulator()
        self.ikmodel = op.databases.inversekinematics.InverseKinematicsModel(self.robot, iktype=op.IkParameterization.Type.Transform6D)
        # free_joints=[self.arm[0]+'_shoulder_pan_joint', self.arm[0]+'_shoulder_lift_joint', self.arm[0]+'_upper_arm_roll_joint', self.arm[0]+'_elbow_flex_joint']  # , self.arm[0]+'_forearm_roll_joint', self.arm[0]+'_wrist_flex_joint', self.arm[0]+'_wrist_roll_joint']
        # self.ikmodel = op.databases.inversekinematics.InverseKinematicsModel(self.robot, iktype=op.IkParameterization.Type.Translation3D, freejoints=free_joints)
        if not self.ikmodel.load():
            print 'IK model not found for leftarm. Will now generate an IK model. This will take a while!'
            self.ikmodel.autogenerate()
            # free_joints=[self.arm[0]+'_shoulder_pan_joint', self.arm[0]+'_shoulder_lift_joint', self.arm[0]+'_upper_arm_roll_joint', self.arm[0]+'_elbow_flex_joint']  # , self.arm[0]+'_forearm_roll_joint', self.arm[0]+'_wrist_flex_joint', self.arm[0]+'_wrist_roll_joint']
            # print free_joints
            # free_joints=[]
            # self.ikmodel.generate(iktype=op.IkParameterizationType.Translation3D, freejoints=free_joints, freeinc=[0.1,0.1,0.1,0.1])

        if self.model is None:
            ## Set robot manipulators, ik, planner
            self.arm = 'rightarm'
            self.robot.SetActiveManipulator(self.arm)
            self.manip = self.robot.GetActiveManipulator()
            self.ikmodel = op.databases.inversekinematics.InverseKinematicsModel(self.robot, iktype=op.IkParameterization.Type.Transform6D)
            # self.ikmodel = op.databases.inversekinematics.InverseKinematicsModel(self.robot, iktype=op.IkParameterization.Type.Translation3D)
            if not self.ikmodel.load():
                print 'IK model not found for rightarm. Will now generate an IK model. This will take a while!'
                self.ikmodel.autogenerate()
                # self.ikmodel.generate(iktype=op.IkParameterizationType.Translation3D, freejoints=[self.arm[0]+'_shoulder_pan_joint', self.arm[0]+'_shoulder_lift_joint', self.arm[0]+'_upper_arm_roll_joint', self.arm[0]+'_elbow_flex_joint'], freeinc=0.01)
            ## Set robot manipulators, ik, planner
            self.arm = 'leftarm'
            self.robot.SetActiveManipulator(self.arm)
            self.manip = self.robot.GetActiveManipulator()
            self.ikmodel = op.databases.inversekinematics.InverseKinematicsModel(self.robot, iktype=op.IkParameterization.Type.Transform6D)
            # self.ikmodel = op.databases.inversekinematics.InverseKinematicsModel(self.robot, iktype=op.IkParameterization.Type.Translation3D)
            if not self.ikmodel.load():
                print 'IK model not found for leftarm. Will now generate an IK model. This will take a while!'
                self.ikmodel.autogenerate()
                # self.ikmodel.generate(iktype=op.IkParameterizationType.Translation3D, freejoints=[self.arm[0]+'_shoulder_pan_joint', self.arm[0]+'_shoulder_lift_joint', self.arm[0]+'_upper_arm_roll_joint', self.arm[0]+'_elbow_flex_joint'], freeinc=0.01)
        # create the interface for basic manipulation programs
        self.manipprob = op.interfaces.BaseManipulation(self.robot)

        self.setup_human_model(init=True)

    def setup_human_model(self, height=None, init=False):
        # Height must be of the form X.X in meters.

        ## Find and load Wheelchair Model
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('hrl_base_selection')
        # Transform from the coordinate frame of the wc model in the back right bottom corner, to the head location on the floor
        if self.model == 'chair':
            '''
            self.env.Load(''.join([pkg_path, '/collada/wheelchair_and_body_assembly.dae']))
            originsubject_B_headfloor = np.matrix([[m.cos(0.), -m.sin(0.),  0.,      0.], #.45 #.438
                                                   [m.sin(0.),  m.cos(0.),  0.,      0.], #0.34 #.42
                                                   [       0.,         0.,  1.,      0.],
                                                   [       0.,         0.,  0.,      1.]])
            '''
            # This is the new wheelchair model
            # self.env.Load(''.join([pkg_path, '/collada/wheelchair_and_body_assembly.dae']))
            self.env.Load(''.join([pkg_path, '/collada/wheelchair_henry_rounded.dae']))
            self.wheelchair = self.env.GetRobots()[1]

            v = self.wheelchair.GetActiveDOFValues()
            v[self.wheelchair.GetJoint('wheelchair/neck_twist_joint').GetDOFIndex()] = 0  # m.radians(60)
            v[self.wheelchair.GetJoint('wheelchair/neck_tilt_joint').GetDOFIndex()] = 0.75
            v[self.wheelchair.GetJoint('wheelchair/neck_head_rotz_joint').GetDOFIndex()] = 0  # -m.radians(30)
            v[self.wheelchair.GetJoint('wheelchair/neck_head_roty_joint').GetDOFIndex()] = -0.45
            v[self.wheelchair.GetJoint('wheelchair/neck_head_rotx_joint').GetDOFIndex()] = 0
            v[self.wheelchair.GetJoint('wheelchair/neck_body_joint').GetDOFIndex()] = -0.15
            v[self.wheelchair.GetJoint('wheelchair/upper_mid_body_joint').GetDOFIndex()] = 0.4
            v[self.wheelchair.GetJoint('wheelchair/mid_lower_body_joint').GetDOFIndex()] = 0.4
            v[self.wheelchair.GetJoint('wheelchair/body_quad_left_joint').GetDOFIndex()] = 0.5
            v[self.wheelchair.GetJoint('wheelchair/body_quad_right_joint').GetDOFIndex()] = 0.5
            v[self.wheelchair.GetJoint('wheelchair/quad_calf_left_joint').GetDOFIndex()] = 1.3
            v[self.wheelchair.GetJoint('wheelchair/quad_calf_right_joint').GetDOFIndex()] = 1.3
            v[self.wheelchair.GetJoint('wheelchair/calf_foot_left_joint').GetDOFIndex()] = 0.2
            v[self.wheelchair.GetJoint('wheelchair/calf_foot_right_joint').GetDOFIndex()] = 0.2
            v[self.wheelchair.GetJoint('wheelchair/body_arm_left_joint').GetDOFIndex()] = 0.6
            v[self.wheelchair.GetJoint('wheelchair/body_arm_right_joint').GetDOFIndex()] = 0.6
            v[self.wheelchair.GetJoint('wheelchair/arm_forearm_left_joint').GetDOFIndex()] = .8
            v[self.wheelchair.GetJoint('wheelchair/arm_forearm_right_joint').GetDOFIndex()] = .8
            v[self.wheelchair.GetJoint('wheelchair/forearm_hand_left_joint').GetDOFIndex()] = 0.
            v[self.wheelchair.GetJoint('wheelchair/forearm_hand_right_joint').GetDOFIndex()] = 0.
            self.wheelchair.SetActiveDOFValues(v, 2)
            self.env.UpdatePublishedBodies()

            headmodel = self.wheelchair.GetLink('wheelchair/head_link')
            head_T = np.matrix(headmodel.GetTransform())
            self.originsubject_B_headfloor = np.matrix([[1., 0.,  0., head_T[0, 3]],  # .442603 #.45 #.438
                                                        [0., 1.,  0., head_T[1, 3]],  # 0.34 #.42
                                                        [0., 0.,  1.,           0.],
                                                        [0., 0.,  0.,           1.]])
            self.originsubject_B_originworld = np.matrix(np.eye(4))
            self.subject = self.env.GetBodies()[1]
            self.subject.SetTransform(np.array(self.originsubject_B_originworld))
            self.a_model_is_loaded = True
        elif self.model == 'bed':
            self.env.Load(''.join([pkg_path, '/models/head_bed.dae']))
            an = 0#m.pi/2
            self.originsubject_B_headfloor = np.matrix([[ m.cos(an),  0., m.sin(an),  .2954], #.45 #.438
                                                        [        0.,  1.,        0.,     0.], #0.34 #.42
                                                        [-m.sin(an),  0., m.cos(an),     0.],
                                                        [        0.,  0.,        0.,     1.]])
            self.originsubject_B_originworld = copy.copy(self.originsubject_B_headfloor)
            self.subject = self.env.GetBodies()[1]
            self.subject.SetTransform(np.array(self.originsubject_B_originworld))
            self.a_model_is_loaded = True
        elif self.model == 'autobed':
            # self.env.Load(''.join([pkg_path, '/collada/bed_and_body_v3_real_expanded_rounded.dae']))
            # self.env.Load(''.join([pkg_path, '/collada/bed_and_body_expanded_rounded.dae']))
            # self.env.Load(''.join([pkg_path, '/collada/bed_and_environment_henry_tray_rounded.dae']))
            self.env.Load(''.join([pkg_path, '/collada/bed_and_environment_mannequin_openrave_rounded.dae']))
            self.autobed = self.env.GetRobots()[1]
            v = self.autobed.GetActiveDOFValues()
            shift = 0.
#            if self.task == 'scratching_knee_left':
#            shift = 0.02
            #0 degrees, 0 height
            bth = 0.
            if True:  # This is the new parameterized version of the model
                v[self.autobed.GetJoint('autobed/tele_legs_joint').GetDOFIndex()] = 0
                v[self.autobed.GetJoint('autobed/bed_neck_base_leftright_joint').GetDOFIndex()] = 0
                v[self.autobed.GetJoint('autobed/torso_pelvis_joint').GetDOFIndex()] = 0
                v[self.autobed.GetJoint('autobed/bed_neck_worldframe_updown_joint').GetDOFIndex()] = (bth / 40) * (
                    0.11 - 0.1) + 0.1
                v[self.autobed.GetJoint('autobed/bed_neck_base_updown_bedframe_joint').GetDOFIndex()] = (bth / 40) * (
                    -0.16 - (-0.0)) + (-0.0)
                v[self.autobed.GetJoint('autobed/head_rest_hinge').GetDOFIndex()] = m.radians(bth)
                v[self.autobed.GetJoint('autobed/headrest_bed_to_worldframe_joint').GetDOFIndex()] = -m.radians(bth)
                v[self.autobed.GetJoint('autobed/bed_neck_to_bedframe_joint').GetDOFIndex()] = m.radians(bth)
                v[self.autobed.GetJoint('autobed/torso_pelvis_joint').GetDOFIndex()] = m.radians(bth)
                v[self.autobed.GetJoint('autobed/neck_twist_joint').GetDOFIndex()] = -((bth / 40) * (0 - 0) + 0)
                v[self.autobed.GetJoint('autobed/neck_tilt_joint').GetDOFIndex()] = ((bth / 40) * (0. - 0.) + 0.)
                v[self.autobed.GetJoint('autobed/neck_head_rotz_joint').GetDOFIndex()] = -((bth / 40) * (0 - 0) + 0)
                v[self.autobed.GetJoint('autobed/neck_head_roty_joint').GetDOFIndex()] = -((bth / 40) * (0. - 0) + 0)
                v[self.autobed.GetJoint('autobed/neck_head_rotx_joint').GetDOFIndex()] = -((bth / 40) * (0 - 0) + 0)
                v[self.autobed.GetJoint('autobed/torso_upper_arm_right_joint').GetDOFIndex()] = -((bth / 40) * (0.0 - 0) + 0)
                v[self.autobed.GetJoint('autobed/torso_upper_arm_left_joint').GetDOFIndex()] = -((bth / 40) * (0.0 - 0) + 0)
                v[self.autobed.GetJoint('autobed/upper_arm_fore_arm_right_joint').GetDOFIndex()] = -(
                    (bth / 40) * (0.6 - 0) + 0)
                v[self.autobed.GetJoint('autobed/upper_arm_fore_arm_left_joint').GetDOFIndex()] = -(
                    (bth / 40) * (0.6 - 0) + 0)
                v[self.autobed.GetJoint('autobed/fore_arm_hand_right_joint').GetDOFIndex()] = -((bth / 40) * (-0.2 - 0) + 0)
                v[self.autobed.GetJoint('autobed/fore_arm_hand_left_joint').GetDOFIndex()] = -((bth / 40) * (-0.2 - 0) + 0)

            self.autobed.SetActiveDOFValues(v, 2)
            self.env.UpdatePublishedBodies()
            self.set_autobed(0., 0., 0., 0.)
            headmodel = self.autobed.GetLink('autobed/head_link')
            head_T = np.matrix(headmodel.GetTransform())

            self.originsubject_B_headfloor = np.matrix([[1.,  0., 0.,  head_T[0, 3]],  #.45 #.438
                                                        [0.,  1., 0.,  head_T[1, 3]],  # 0.34 #.42
                                                        [0.,  0., 1.,           0.],
                                                        [0.,  0., 0.,           1.]])
            self.originsubject_B_originworld = np.matrix(np.eye(4))
            self.subject = self.env.GetBodies()[1]
            self.subject.SetTransform(np.array(self.originsubject_B_originworld))
            self.a_model_is_loaded = True
        elif self.model is None:
            self.a_model_is_loaded = False
            with self.env:
                self.environment_model = op.RaveCreateKinBody(self.env, '')
                self.environment_model.SetName('environment_model')
                self.environment_model.InitFromBoxes(np.array([[-1.2, -1.2, -1.2, 1.01, 1.01, 1.01]]),True) # set geometry as one box of extents 0.1, 0.2, 0.3
                self.env.AddKinBody(self.environment_model)

            'Using a custom model at in a real-time mode.'
        else:
            self.a_model_is_loaded = False
            print 'I got a bad model. What is going on???'
            return None

        # self.subject_location = originsubject_B_headfloor.I

        print 'OpenRave has succesfully been initialized. \n'

    def set_autobed(self, z, headrest_th, head_x, head_y):
        with self.env:
            bz = z
            # print headrest_th
            bth = m.degrees(headrest_th)
            # print bth
            v = self.autobed.GetActiveDOFValues()
            v[self.autobed.GetJoint('autobed/tele_legs_joint').GetDOFIndex()] = bz
            v[self.autobed.GetJoint('autobed/bed_neck_base_updown_bedframe_joint').GetDOFIndex()] = head_x
            shift = 0.
    #        if self.task == 'scratching_knee_left':
    #        shift = 0.02
            v[self.autobed.GetJoint('autobed/bed_neck_base_leftright_joint').GetDOFIndex()] = head_y

            v[self.autobed.GetJoint('autobed/neck_twist_joint').GetDOFIndex()] = 0
            v[self.autobed.GetJoint('autobed/neck_head_rotz_joint').GetDOFIndex()] = 0

            if bth >= 80. and bth < 85.:
                bth = 80.
            if bth >= -1. and bth <= 0.:
                bth = 0.
                # 0 degrees, 0 height

            if (bth >= 0.) and (bth < 40.):  # between 0 and 40 degrees
                v[self.autobed.GetJoint('autobed/bed_neck_worldframe_updown_joint').GetDOFIndex()] = (bth / 40) * (
                    0.11 - 0.1) + 0.1
                v[self.autobed.GetJoint('autobed/bed_neck_base_updown_bedframe_joint').GetDOFIndex()] = (bth / 40) * (
                    -0.16 - (-0.0)) + (-0.0)
                v[self.autobed.GetJoint('autobed/head_rest_hinge').GetDOFIndex()] = m.radians(bth)
                v[self.autobed.GetJoint('autobed/headrest_bed_to_worldframe_joint').GetDOFIndex()] = -m.radians(bth)
                v[self.autobed.GetJoint('autobed/bed_neck_to_bedframe_joint').GetDOFIndex()] = m.radians(bth)
                v[self.autobed.GetJoint('autobed/torso_pelvis_joint').GetDOFIndex()] = m.radians(bth)
                v[self.autobed.GetJoint('autobed/neck_twist_joint').GetDOFIndex()] = -((bth / 40) * (0 - 0) + 0)
                v[self.autobed.GetJoint('autobed/neck_tilt_joint').GetDOFIndex()] = ((bth / 40) * (0. - 0.) + 0.)
                v[self.autobed.GetJoint('autobed/neck_head_rotz_joint').GetDOFIndex()] = -((bth / 40) * (0 - 0) + 0)
                v[self.autobed.GetJoint('autobed/neck_head_roty_joint').GetDOFIndex()] = -((bth / 40) * (0. - 0) + 0)
                v[self.autobed.GetJoint('autobed/neck_head_rotx_joint').GetDOFIndex()] = -((bth / 40) * (0 - 0) + 0)
                v[self.autobed.GetJoint('autobed/torso_upper_arm_right_joint').GetDOFIndex()] = -((bth / 40) * (0.0 - 0) + 0)
                v[self.autobed.GetJoint('autobed/torso_upper_arm_left_joint').GetDOFIndex()] = -((bth / 40) * (0.0 - 0) + 0)
                v[self.autobed.GetJoint('autobed/upper_arm_fore_arm_right_joint').GetDOFIndex()] = -(
                    (bth / 40) * (0.6 - 0) + 0)
                v[self.autobed.GetJoint('autobed/upper_arm_fore_arm_left_joint').GetDOFIndex()] = -(
                    (bth / 40) * (0.6 - 0) + 0)
                v[self.autobed.GetJoint('autobed/fore_arm_hand_right_joint').GetDOFIndex()] = -((bth / 40) * (-0.2 - 0) + 0)
                v[self.autobed.GetJoint('autobed/fore_arm_hand_left_joint').GetDOFIndex()] = -((bth / 40) * (-0.2 - 0) + 0)
            elif (bth >= 40.) and (bth <= 80.):  # between 0 and 40 degrees
                v[self.autobed.GetJoint('autobed/bed_neck_worldframe_updown_joint').GetDOFIndex()] = ((bth - 40) / 40) * (
                    0.06 - (0.11)) + (0.11)
                v[self.autobed.GetJoint('autobed/bed_neck_base_updown_bedframe_joint').GetDOFIndex()] = ((bth - 40) / 40) * (
                    -0.25 - (-0.16)) + (-0.16)
                v[self.autobed.GetJoint('autobed/head_rest_hinge').GetDOFIndex()] = m.radians(bth)
                v[self.autobed.GetJoint('autobed/headrest_bed_to_worldframe_joint').GetDOFIndex()] = -m.radians(bth)
                v[self.autobed.GetJoint('autobed/bed_neck_to_bedframe_joint').GetDOFIndex()] = m.radians(bth)
                v[self.autobed.GetJoint('autobed/torso_pelvis_joint').GetDOFIndex()] = m.radians(bth)
                v[self.autobed.GetJoint('autobed/neck_twist_joint').GetDOFIndex()] = -(((bth - 40) / 40) * (0 - 0) + 0)
                v[self.autobed.GetJoint('autobed/neck_tilt_joint').GetDOFIndex()] = (((bth - 40) / 40) * (0. - 0.) + 0.)
                v[self.autobed.GetJoint('autobed/neck_head_rotz_joint').GetDOFIndex()] = -(((bth - 40) / 40) * (0 - 0) + 0)
                v[self.autobed.GetJoint('autobed/neck_head_roty_joint').GetDOFIndex()] = -(
                    ((bth - 40) / 40) * (0.0 - (0.)) + (0.))
                v[self.autobed.GetJoint('autobed/neck_head_rotx_joint').GetDOFIndex()] = -(((bth - 40) / 40) * (0 - 0) + 0)
                v[self.autobed.GetJoint('autobed/torso_upper_arm_right_joint').GetDOFIndex()] = -(
                    ((bth - 40) / 40) * (0. - 0) + 0)
                v[self.autobed.GetJoint('autobed/torso_upper_arm_left_joint').GetDOFIndex()] = -(
                    ((bth - 40) / 40) * (0. - 0) + 0)
                v[self.autobed.GetJoint('autobed/upper_arm_fore_arm_right_joint').GetDOFIndex()] = -(
                    ((bth - 40) / 40) * (1.3 - 0.6) + 0.6)
                v[self.autobed.GetJoint('autobed/upper_arm_fore_arm_left_joint').GetDOFIndex()] = -(
                    ((bth - 40) / 40) * (1.3 - 0.6) + 0.6)
                v[self.autobed.GetJoint('autobed/fore_arm_hand_right_joint').GetDOFIndex()] = -(
                    ((bth - 40) / 40) * (-0.1 - (-0.2)) + (-0.2))
                v[self.autobed.GetJoint('autobed/fore_arm_hand_left_joint').GetDOFIndex()] = -(
                    ((bth - 40) / 40) * (-0.1 - (-0.2)) + (-0.2))
            else:
                print 'Error: Bed angle out of range (should be 0 - 80 degrees)'
            self.autobed.SetActiveDOFValues(v, 2)
            self.env.UpdatePublishedBodies()
        # self.create_vision_cone()
        # rospy.sleep(0.02)
        self.env.UpdatePublishedBodies()

    def gen_joint_limit_weight(self, q):
        # define the total range limit for each joint
        l_min = np.array([-40., -30., -44., -133., -400., -130., -400.])
        # l_min = np.array([-40., -30., -44., -45., -400., -130., -400.])
        l_max = np.array([130., 80., 224., 0., 400., 0., 400.])
        l_range = l_max - l_min
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
            weights[joint] = (1. - m.pow(0.5, ((l_range[joint])/2. - np.abs((l_range[joint])/2. - m.degrees(q[joint]) + l_min[joint]))/(l_range[joint]/40.)))
            # weights[joint] = 1. - m.pow(0.5, (l_max[joint]-l_min[joint])/2. - np.abs((l_max[joint] - l_min[joint])/2. - m.degrees(q[joint]) + l_min[joint]))
            if weights[joint] < 0.001:
                weights[joint] = 0.001
#        print 'q', q
#        print 'weights', weights
        weights[4] = 1.
        weights[6] = 1.
        return np.matrix(np.diag(weights))

if __name__ == "__main__":
    rospy.init_node('score_generator')
    mytask = 'shoulder'
    mymodel = 'chair'
    #mytask = 'all_goals'
    start_time = time.time()
    selector = ScoreGenerator(visualize=False,task=mytask,goals = None,model=mymodel)
    #selector.choose_task(mytask)
    score_sheet = selector.handle_score_generation()

    print 'Time to load find generate all scores: %fs'%(time.time()-start_time)

    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('hrl_base_selection')
    save_pickle(score_sheet, ''.join([pkg_path, '/data/', mymodel, '_', mytask, '.pkl']))
    print 'Time to complete program, saving all data: %fs' % (time.time()-start_time)




