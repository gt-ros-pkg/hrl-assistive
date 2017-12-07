#!/usr/bin/env python

import roslib
roslib.load_manifest('hrl_base_selection')
roslib.load_manifest('hrl_haptic_mpc')
roslib.load_manifest('hrl_lib')
import numpy as np
import math as m
import time
import roslib
import rospy
from geometry_msgs.msg import PoseArray, PoseStamped, Pose
from visualization_msgs.msg import Marker, MarkerArray
from helper_functions import createBMatrix, Bmat_to_pos_quat
from data_reader_dressing import DataReader as DataReader_dressing
from data_reader_cma import DataReader as DataReader_cma
from data_reader import DataReader as DataReader_brute
from matplotlib.cbook import flatten
import copy



class DataReader_Task(object):
    def __init__(self, task, model, optimization, subject=None, visualize=False):
        self.optimization = optimization
        self.visualize = visualize
        self.model = model
        self.task = task
        self.subject = subject
        self.num = []
        self.reference_options = []
        self.reference = []
        self.goals = []
        self.reset_goals()

    def reset_goals(self):
        self.goals = []
        if self.task == 'shaving':
            liftlink_B_reference = createBMatrix([1.249848, -0.013344, 0.1121597], [0.044735, -0.010481, 0.998626, -0.025188])
            liftlink_B_goal = [[1.087086, -0.019988, 0.014680, 0.011758, 0.014403, 0.031744, 0.999323],
                               [1.089931, -0.023529, 0.115044, 0.008146, 0.125716, 0.032856, 0.991489],
                               [1.123504, -0.124174, 0.060517, 0.065528, -0.078776, 0.322874, 0.940879],
                               [1.192543, -0.178014, 0.093005, -0.032482, 0.012642, 0.569130, 0.821509],
                               [1.194537, -0.180350, 0.024144, 0.055151, -0.113447, 0.567382, 0.813736],
                               [1.103003, 0.066879, 0.047237, 0.023224, -0.083593, -0.247144, 0.965087],
                               [1.180539, 0.155222, 0.061160, -0.048171, -0.076155, -0.513218, 0.853515],
                               [1.181696, 0.153536, 0.118200, 0.022272, 0.045203, -0.551630, 0.832565]]
            # goal_B_gripper = np.matrix(np.eye(4))
            goal_B_gripper = np.matrix([[1., 0., 0., -0.04],
                                        [0., 1., 0., 0.0],
                                        [0., 0., 1., 0.0],
                                        [0., 0., 0., 1.]])
            for i in xrange(len(liftlink_B_goal)):
                self.goals.append(liftlink_B_reference.I*createBMatrix(liftlink_B_goal[i][0:3], liftlink_B_goal[i][3:])*goal_B_gripper)  # all in reference to head
            self.num = np.ones([len(self.goals), 1])
            self.reference_options = ['head']
            self.reference = np.zeros([len(self.goals), 1])
        elif self.task == 'wiping_mouth':
            # liftlink_B_reference = createBMatrix([1.249848, -0.013344, 0.1121597], [0.044735, -0.010481, 0.998626, -0.025188])
            # liftlink_B_goal = [[1.107086, -0.019988, 0.014680, 0.011758, 0.014403, 0.031744, 0.999323],
            #                    [1.089931, -0.023529, 0.115044, 0.008146, 0.125716, 0.032856, 0.991489],
            #                    [1.123504, -0.124174, 0.060517, 0.065528, -0.078776, 0.322874, 0.940879],
            #                    #[1.192543, -0.178014, 0.093005, -0.032482, 0.012642, 0.569130, 0.821509]]
            #                    #[1.194537, -0.180350, 0.024144, 0.055151, -0.113447, 0.567382, 0.813736],
            #                    [1.103003, 0.066879, 0.047237, 0.023224, -0.083593, -0.247144, 0.965087]]
            #                    #[1.180539, 0.155222, 0.061160, -0.048171, -0.076155, -0.513218, 0.853515]]
            #                    #[1.181696, 0.153536, 0.118200, 0.022272, 0.045203, -0.551630, 0.832565]]
            liftlink_B_reference = createBMatrix([0., 0., 0.], [0., 0., 0., 1.])
            liftlink_B_goal = [[.15, 0.0, -0.05, 0., 1., 1., 0.],
                               [.15, 0.0, -0.08, 0., 1., 1., 0.],
                               [.15, 0.03, -0.065, -0.1, 1., 1., -0.1],
                               #[1.192543, -0.178014, 0.093005, -0.032482, 0.012642, 0.569130, 0.821509]]
                               #[1.194537, -0.180350, 0.024144, 0.055151, -0.113447, 0.567382, 0.813736],
                               [.15, -0.03, -0.065, 0.1, 1., 1., 0.1]]
                               #[1.180539, 0.155222, 0.061160, -0.048171, -0.076155, -0.513218, 0.853515]]
                               #[1.181696, 0.153536, 0.118200, 0.022272, 0.045203, -0.551630, 0.832565]]
            for i in xrange(len(liftlink_B_goal)):
                self.goals.append(liftlink_B_reference.I*createBMatrix(liftlink_B_goal[i][0:3], liftlink_B_goal[i][3:]))  # all in reference to head
            # left_side = self.goals[2]
            # right_side = self.goals[3]
            # #left_side[0:3,0:3] = -1*right_side[0:3,0:3]
            # left_side[0,3] = right_side[0,3]
            # left_side[1,3] = -right_side[1,3]
            # left_side[2,3] = right_side[2,3]
            # self.goals[2] = left_side
            self.num = np.ones([len(self.goals), 1])
            self.reference_options = ['head']
            self.reference = np.zeros([len(self.goals), 1])
        elif self.task == 'scratching_forehead':
            liftlink_B_reference = createBMatrix([0., 0., 0.], [0., 0., 0., 1.])
            liftlink_B_goal = [[.15, 0.0, 0.04, 0., 1., 1., 0.],
                               [.15, 0.03, 0.03, 0., 1., 1., 0.],
                               [.15, -0.03, 0.03, 0., 1., 1., 0.]]
            for i in xrange(len(liftlink_B_goal)):
                self.goals.append(liftlink_B_reference.I*createBMatrix(liftlink_B_goal[i][0:3], liftlink_B_goal[i][3:]))  # all in reference to head
            self.num = np.ones([len(self.goals), 1])
            self.reference_options = ['head']
            self.reference = np.zeros([len(self.goals), 1])
        elif self.task == 'brushing_teeth':
            liftlink_B_reference = createBMatrix([0., 0., 0.], [0., 0., 0., 1.])
            liftlink_B_goal = [[.125, 0.0, -0.065, 0.5,  0.5,  0.5,  0.5],
                               [.115, 0.025, -0.065,  0.40557979,  0.57922797,  0.57922797,  0.40557979],
                               #[1.192543, -0.178014, 0.093005, -0.032482, 0.012642, 0.569130, 0.821509]]
                               #[1.194537, -0.180350, 0.024144, 0.055151, -0.113447, 0.567382, 0.813736],
                               [.115, -0.025, -0.065, -0.40557979,  0.57922797, -0.57922797,  0.40557979]]
                               #[1.180539, 0.155222, 0.061160, -0.048171, -0.076155, -0.513218, 0.853515]]
                               #[1.181696, 0.153536, 0.118200, 0.022272, 0.045203, -0.551630, 0.832565]]
            # goal_B_gripper = np.matrix(np.eye(4))
            goal_B_gripper = np.matrix([[1., 0., 0., -0.185],
                                        [0., 1., 0., 0.0],
                                        [0., 0., 1., 0.02],
                                        [0., 0., 0., 1.]])
            for i in xrange(len(liftlink_B_goal)):
                self.goals.append(liftlink_B_reference.I*createBMatrix(liftlink_B_goal[i][0:3], liftlink_B_goal[i][3:])*goal_B_gripper)  # all in reference to head
            self.num = np.ones([len(self.goals), 1])
            self.reference_options = ['head']
            self.reference = np.zeros([len(self.goals), 1])
        elif self.task == 'full_hospital_gown':

            rotx = np.matrix([[1., 0., 0., 0.],
                              [0., 0., 1., 0.],
                              [0., -1., 0., 0.],
                              [0., 0., 0., 1.]])
            rotz_180 = np.matrix([[-1., 0., 0., 0.],
                                  [0., -1., 0., 0.],
                                  [0., 0., 1., 0.],
                                  [0., 0., 0., 1.]])
            rotz_90 = np.matrix([[0., -1., 0., 0.],
                                 [1., 0., 0., 0.],
                                 [0., 0., 1., 0.],
                                 [0., 0., 0., 1.]])
            rotz_n90 = np.matrix([[0., 1., 0., 0.],
                                  [-1., 0., 0., 0.],
                                  [0., 0., 1., 0.],
                                  [0., 0., 0., 1.]])
            roty_90 = np.matrix([[0., 0., 1., 0.],
                                 [0., 1., 0., 0.],
                                 [-1., 0., 0., 0.],
                                 [0., 0., 0., 1.]])
            roty_n90 = np.matrix([[0., 0., -1., 0.],
                                  [0., 1., 0., 0.],
                                  [1., 0., 0., 0.],
                                  [0., 0., 0., 1.]])
            rot_correction = [rotz_180,
                              rotz_180,
                              rotz_180,
                              rotz_180,
                              rotz_180,
                              rotz_180,
                              rotz_90]
            # rot_correction = np.matrix(np.eye(4))
            trans = np.array([[.075, -0.05, 0.0],
                              [.18, -0.05, 0.0],
                              [.075, 0., 0.],
                              [.18, 0., 0.],
                              [.09, 0., 0.],
                              [.15, 0., 0.],
                              [0., -0.17, -0.1]])
            reference_B_goal = []
            for i in xrange(len(trans)):
                rot_correction[i][0:3, 3] = copy.copy(np.reshape(trans[i], [3, 1]))
                # rot_correction = np.matrix(np.eye(4))
                reference_B_goal.append(copy.copy(rot_correction[i]))
            # Note that I tranform the coordinates system to have x pointing out front of chest, Z up, in Openrave.
            self.reference_options = ['upper_arm_right',
                                      'forearm_right',
                                      'hand_right',
                                      'upper_body']
            self.reference = np.array([[0],
                                       [0],
                                       [1],
                                       [1],
                                       [2],
                                       [2],
                                       [3]])
            self.num = np.ones([len(self.reference), 1])
            self.goals = reference_B_goal
            self.task = dict()
            # for task in ['full_hospital_gown']:
            #     for subtask in ['gown_right_arm', 'gown_left_arm']:
            #         for subsubtask in ['sleeve_upper_arm', 'sleeve_forearm', 'sleeve_hand', 'sleeve_upper_body']:
            self.task['full_hospital_gown'] = {}
            self.task['full_hospital_gown']['gown_right_arm'] = {}

            # Right Arm
            ref_options = ['upper_arm_right']
            ref_choice = np.array([[0], [0]])
            ref_B_goal = [rotz_180, rotz_180]
            trans = np.array([[.075, 0., 0.],
                              [.18, 0., 0.]])
            for i in xrange(len(trans)):
                ref_B_goal[i][0:3, 3] = copy.copy(np.reshape(trans[i], [3, 1]))
            self.task['full_hospital_gown']['gown_right_arm']['sleeve_upper_arm_right'] = copy.copy([ref_B_goal, ref_choice])

            ref_options = ['forearm_arm_right']
            ref_choice = np.array([[1], [1]])
            ref_B_goal = [rotz_180, rotz_180]
            trans = np.array([[.075, 0., 0.],
                              [.18, 0., 0.]])
            for i in xrange(len(trans)):
                ref_B_goal[i][0:3, 3] = copy.copy(np.reshape(trans[i], [3, 1]))
            self.task['full_hospital_gown']['gown_right_arm']['sleeve_forearm_right'] = copy.copy([ref_B_goal, ref_choice])

            ref_options = ['hand_right']
            ref_choice = np.array([[2], [2]])
            ref_B_goal = [rotz_180, rotz_180]
            trans = np.array([[.09, 0., 0.],
                              [.15, 0., 0.]])
            for i in xrange(len(trans)):
                ref_B_goal[i][0:3, 3] = copy.copy(np.reshape(trans[i], [3, 1]))
            self.task['full_hospital_gown']['gown_right_arm']['sleeve_hand_right'] = copy.copy([ref_B_goal, ref_choice])

            ref_options = ['upper_body']
            ref_choice = np.array([[6], [6]])
            ref_B_goal = [rotz_180, rotz_180]
            trans = np.array([[0., -0.17, -0.1]])
            for i in xrange(len(trans)):
                ref_B_goal[i][0:3, 3] = copy.copy(np.reshape(trans[i], [3, 1]))
            self.task['full_hospital_gown']['gown_right_arm']['sleeve_upper_body_right'] = copy.copy([ref_B_goal, ref_choice])

            # Left Arm
            ref_options = ['upper_arm_left']
            ref_choice = np.array([[3], [3]])
            ref_B_goal = [rotz_90]
            trans = np.array([[.075, 0.05, 0.0],
                              [.18, 0.05, 0.0]])
            for i in xrange(len(trans)):
                ref_B_goal[i][0:3, 3] = copy.copy(np.reshape(trans[i], [3, 1]))
            self.task['full_hospital_gown']['gown_left_arm']['sleeve_upper_arm_left'] = copy.copy([ref_B_goal, ref_choice])

            ref_options = ['forearm_arm_left']
            ref_choice = np.array([[4], [4]])
            ref_B_goal = [rotz_180, rotz_180]
            trans = np.array([[.075, -0.05, 0.0],
                              [.18, -0.05, 0.0]])
            for i in xrange(len(trans)):
                ref_B_goal[i][0:3, 3] = copy.copy(np.reshape(trans[i], [3, 1]))
            self.task['full_hospital_gown']['gown_left_arm']['sleeve_forearm_left'] = copy.copy([ref_B_goal, ref_choice])

            ref_options = ['hand_left']
            ref_choice = np.array([[5], [5]])
            ref_B_goal = [rotz_180, rotz_180]
            trans = np.array([[.09, 0., 0.],
                              [.15, 0., 0.]])
            for i in xrange(len(trans)):
                ref_B_goal[i][0:3, 3] = copy.copy(np.reshape(trans[i], [3, 1]))
            self.task['full_hospital_gown']['gown_left_arm']['sleeve_hand_left'] = copy.copy([ref_B_goal, ref_choice])

            ref_options = ['upper_body']
            ref_choice = np.array([[6]])
            ref_B_goal = [rotz_90]
            trans = np.array([[0., 0.17, -0.1]])
            for i in xrange(len(trans)):
                ref_B_goal[i][0:3, 3] = copy.copy(np.reshape(trans[i], [3, 1]))
            self.task['full_hospital_gown']['gown_left_arm']['sleeve_upper_arm_left'] = copy.copy([ref_B_goal, ref_choice])
            self.ref_options = ['upper_arm_right',
                           'forearm_right',
                           'hand_right',
                           'upper_arm_left',
                           'forearm_left',
                           'hand_left',
                           'upper_body']
            return self.task, ref_options

            # self.task['full_hospital_gown']['gown_left_arm'] = \
            #     {'sleeve_upper_arm_left', 'selforearm_left', 'sleeve_hand_left', 'sleeve_upper_body_left'}
        elif self.task == 'hospital_gown':
            gripper_B_goal = np.matrix([[0., -1., 0., 0.03],
                                        [1., 0., 0., 0.0],
                                        [0., 0., 1., -0.08],
                                        [0., 0., 0., 1.]])
            rotx = np.matrix([[1., 0., 0., 0.],
                              [0., 0., 1., 0.],
                              [0., -1., 0., 0.],
                              [0., 0., 0., 1.]])
            rotz_180 = np.matrix([[-1., 0., 0., 0.],
                                  [0., -1., 0., 0.],
                                  [0., 0., 1., 0.],
                                  [0., 0., 0., 1.]])
            rotz_90 = np.matrix([[0., -1., 0., 0.],
                                 [1., 0., 0., 0.],
                                 [0., 0., 1., 0.],
                                 [0., 0., 0., 1.]])
            rotz_n90 = np.matrix([[0., 1., 0., 0.],
                                  [-1., 0., 0., 0.],
                                  [0., 0., 1., 0.],
                                  [0., 0., 0., 1.]])
            roty_90 = np.matrix([[0., 0., 1., 0.],
                                 [0., 1., 0., 0.],
                                 [-1., 0., 0., 0.],
                                 [0., 0., 0., 1.]])
            roty_n90 = np.matrix([[0., 0., -1., 0.],
                                  [0., 1., 0., 0.],
                                  [1., 0., 0., 0.],
                                  [0., 0., 0., 1.]])
            # rot_correction = rotz#*rotx
            rot_correction = [rotz_180,
                              rotz_180,
                              rotz_180,
                              rotz_180,
                              rotz_180,
                              rotz_180,
                              rotz_90]
            # rot_correction = np.matrix(np.eye(4))
            trans = np.array([[.075, -0.05, 0.0],
                              [.18, -0.05, 0.0],
                              [.075, 0., 0.],
                              [.18, 0., 0.],
                              [.09, 0., 0.],
                              [.15, 0., 0.],
                              [0., -0.17, -0.1]])
            reference_B_goal = []
            for i in xrange(len(trans)):
                rot_correction[i][0:3, 3] = copy.copy(np.reshape(trans[i], [3, 1]))
                # rot_correction = np.matrix(np.eye(4))
                reference_B_goal.append(copy.copy(rot_correction[i]))
            # Note that I tranform the coordinates system to have x pointing out front of chest, Z up, in Openrave.
            self.reference_options = ['upper_arm_right',
                                      'forearm_right',
                                      'hand_right',
                                      'upper_arm_left',
                                      'forearm_left',
                                      'hand_left',
                                      'upper_body']
            self.reference = np.array([[0],
                                       [0],
                                       [1],
                                       [1],
                                       [2],
                                       [2],
                                       [3]])
            self.num = np.ones([len(self.reference), 1])
            self.goals = reference_B_goal
            return self.goals, self.num, self.reference, self.reference_options
        elif self.task == 'bathing':
            liftlink_B_reference = []
            liftlink_B_reference.append(createBMatrix([0.521625, 0.175031, 0.596279], [0.707105, 0.006780, -0.006691, 0.707044]))
            liftlink_B_reference.append(createBMatrix([0.521668, -0.174969, 0.596249], [0.706852, 0.020076, -0.019986, 0.706794]))
            liftlink_B_reference.append(createBMatrix([0.803358, 0.225067, 0.579914], [0.707054, -0.010898, 0.010985, 0.706991]))
            liftlink_B_reference.append(createBMatrix([0.803222, -0.224933, 0.580274], [0.707054, -0.010898, 0.010985, 0.706991]))
            liftlink_B_reference.append(createBMatrix([0.977372, 0.086085, 0.624297], [0.702011, 0.084996, -0.084900, 0.701960]))
            liftlink_B_reference.append(createBMatrix([0.977394, -0.085915, 0.624282], [0.702011, 0.084996, -0.084900, 0.701960]))
            liftlink_B_reference.append(createBMatrix([0.424870, 0.000019, 0.589686], [0.706732, -0.023949, 0.024036, 0.706667]))
            liftlink_B_goal = [[0.554345, 0.233102, 0.693507, -0.524865, 0.025934, 0.850781, -0.003895],
                               [0.661972, 0.231151, 0.699075, -0.630902, 0.025209, 0.775420, -0.007229],
                               [0.595982, -0.218764, 0.702598, -0.582908, 0.005202, 0.812505, -0.005172],
                               [0.673797, -0.218444, 0.712133, -0.665426, 0.004626, 0.746428, -0.005692],
                               [0.884083, 0.234311, 0.708599, -0.599114, 0.005096, 0.800630, -0.005276],
                               [1.005796, 0.234676, 0.714125, -0.599114, 0.005096, 0.800630, -0.005276],
                               [0.905275, -0.223041, 0.714310, -0.599114, 0.005096, 0.800630, -0.005276],
                               [1.000633, -0.223224, 0.686273, -0.599114, 0.005096, 0.800630, -0.005276],
                               [1.139935, 0.087965, 0.748606, -0.599114, 0.005096, 0.800630, -0.005276],
                               [1.257200, 0.088435, 0.762122, -0.599114, 0.005096, 0.800630, -0.005276],
                               [1.257598, -0.081394, 0.764582, -0.599114, 0.005096, 0.800630, -0.005276],
                               [1.097844, -0.081661, 0.772003, -0.599114, 0.005096, 0.800630, -0.005276],
                               [0.606949, -0.012159, 0.723371, -0.599114, 0.005096, 0.800630, -0.005276],
                               [0.660066, -0.011944, 0.729623, -0.599114, 0.005096, 0.800630, -0.005276]]
            for i in xrange(len(liftlink_B_goal)):
                self.goals.append(liftlink_B_reference[i/2].I*createBMatrix(liftlink_B_goal[i][0:3], liftlink_B_goal[i][3:]))  # all in reference to head
            self.num = np.ones([len(self.goals), 1])
            self.reference_options = ['upper_arm_left', 'upper_arm_right', 'forearm_left', 'forearm_right', 'thigh_left', 'thigh_right', 'chest']
            self.reference = np.array([[0],
                                       [0],
                                       [1],
                                       [1],
                                       [2],
                                       [2],
                                       [3],
                                       [3],
                                       [4],
                                       [4],
                                       [5],
                                       [5],
                                       [6],
                                       [6]])
        else:
            print 'THE TASK I GOT WAS BOGUS. SOMETHING PROBABLY WRONG WITH INPUTS TO DATA READER TASK'
        self.goals = np.array(self.goals)
        #print self.goals
        return self.goals, self.num, self.reference, self.reference_options

    def generate_score(self):
        # for item in data:
        #     print Bmat_to_pos_quat(item)
        print 'Starting to convert data!'
        if self.optimization == 'dressing' and self.task == 'full_hospital_gown':
            run_data = DataReader_dressing(model=self.model)
            run_data.receive_task(self.task, self.ref_options)
            run_data.generate_score(viz_rviz=True, visualize=self.visualize, plot=False)
        elif self.optimization == 'dressing':
            run_data = DataReader_dressing(model=self.model, task=self.task)
            run_data.receive_input_data(self.goals, self.num, self.reference_options, self.reference)
            run_data.generate_output_goals()
            run_data.generate_score(viz_rviz=True, visualize=self.visualize, plot=False)
        elif self.optimization == 'brute':
            run_data = DataReader_brute(subject=self.subject, model=self.model, task=self.task)
            run_data.receive_input_data(self.goals, self.num, self.reference_options, self.reference)
            run_data.generate_output_goals()
            run_data.generate_score(viz_rviz=True, visualize=self.visualize, plot=False)
        else:
            print 'I got a bad optimization type!!'
            return None


        print 'Starting to generate the data for the task', self.task


    def visualize_only(self):

        marker = Marker()
        #marker.header.frame_id = "/base_footprint"
        marker.header.frame_id = "/base_link"
        marker.header.stamp = rospy.Time()
        marker.id = 0
        marker.type = Marker.MESH_RESOURCE
        marker.action = Marker.ADD
        marker.pose.position.x = 0
        marker.pose.position.y = 0
        marker.pose.position.z = 0
        marker.pose.orientation.x = 0
        marker.pose.orientation.y = 0
        marker.pose.orientation.z = 0
        marker.pose.orientation.w = 1
        marker.color.a = 1.
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        if self.model == 'chair':
            name = 'subject_model'
            # marker.mesh_resource = "package://hrl_base_selection/models/wheelchair_and_body_assembly_rviz.STL"
            marker.mesh_resource = "package://hrl_base_selection/urdf/wheelchair_henry/meshes/head_link.STL"
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
            # marker.mesh_resource = "package://hrl_base_selection/models/bed_and_body_v3_rviz.dae"
            marker.mesh_resource = "package://hrl_base_selection/urdf/bed_and_environment_henry/meshes/head_link.STL"
            marker.scale.x = 1.0
            marker.scale.y = 1.0
            marker.scale.z = 1.0
        elif self.model == 'green_kevin':
            name = 'subject_model'
            # marker.mesh_resource = "package://hrl_base_selection/models/bed_and_body_v3_rviz.dae"
            marker.mesh_resource = "package://hrl_base_selection/urdf/green_kevin/meshes/upper_body_link.STL"
            marker.scale.x = 1.0
            marker.scale.y = 1.0
            marker.scale.z = 1.0
        else:
            print 'I got a bad model. What is going on???'
            return None
        vis_pub = rospy.Publisher(''.join(['~',name]), Marker, queue_size=1, latch=True)
        marker.ns = ''.join(['base_service_',name])
        vis_pub.publish(marker)
        print 'Published a model of the subject to rviz'


        ref_vis_pub = rospy.Publisher('~reference_pose', PoseStamped, queue_size=1, latch=True)
        ref_pose = PoseStamped()
        ref_pose.header.frame_id = "/base_link"
        ref_pose.header.stamp = rospy.Time.now()
        ref_pose.pose.position.x = 0.
        ref_pose.pose.position.y = 0.
        ref_pose.pose.position.z = 0.
        ref_pose.pose.orientation.x = 0.
        ref_pose.pose.orientation.y = 0.
        ref_pose.pose.orientation.z = 0.
        ref_pose.pose.orientation.w = 1.
        ref_vis_pub.publish(ref_pose)
        # goal_vis_pub = rospy.Publisher('~goal_poses', MarkerArray, queue_size=1, latch=True)
        goal_vis_pub = rospy.Publisher('~goal_poses', PoseArray, queue_size=1, latch=True)
        goal_markers = MarkerArray()
        goal_markers = PoseArray()
        goal_markers.header.frame_id = "/base_link"
        goal_markers.header.stamp = rospy.Time.now()

        for num, goal_marker in enumerate(self.goals):
            #print goal_marker
            pos, ori = Bmat_to_pos_quat(goal_marker)
            # marker = Marker()
            marker = Pose()
            #marker.header.frame_id = "/base_footprint"
            # marker.header.frame_id = "/base_link"
            # marker.header.stamp = rospy.Time.now()
            # marker.ns = str(num)
            # marker.id = 0
            # marker.type = Marker.ARROW
            # marker.action = Marker.ADD
            # marker.pose.position.x = pos[0]
            # marker.pose.position.y = pos[1]
            # marker.pose.position.z = pos[2]
            # marker.pose.orientation.x = ori[0]
            # marker.pose.orientation.y = ori[1]
            # marker.pose.orientation.z = ori[2]
            # marker.pose.orientation.w = ori[3]
            marker.position.x = pos[0]
            marker.position.y = pos[1]
            marker.position.z = pos[2]
            marker.orientation.x = ori[0]
            marker.orientation.y = ori[1]
            marker.orientation.z = ori[2]
            marker.orientation.w = ori[3]
            # marker.scale.x = .05*1
            # marker.scale.y = .01*1
            # marker.scale.z = .01*1
            # marker.color.a = 1.
            # marker.color.r = 1.0
            # marker.color.g = 0.0
            # marker.color.b = 0.0
            #print marker
            # goal_markers.markers.append(marker)
            goal_markers.poses.append(marker)
        goal_vis_pub.publish(goal_markers)
        print 'Published a goal marker to rviz'

if __name__ == "__main__":
    visualize_only = False
    model = 'green_kevin'  # options are: 'chair', 'bed', 'autobed', 'wall'
    optimization = 'dressing'  # 'cma' or 'brute'
    task = 'hospital_gown' # scratching_knee_left # options are: wiping_face, bathing, brushing, feeding, shaving, scratching_upperarm/forearm/thigh/chest/knee_left/right
    rospy.init_node(optimization+'_'+model+'_'+task)
    full_start_time = time.time()
    if visualize_only:
        shaving_data_reader = DataReader_Task(task, model, optimization)
        shaving_data_reader.visualize_only()
        print 'Visualizing the goals for the task', task, ' only.'
        rospy.spin()
    else:
        # for task in ['wiping_face', 'scratching_knee_left', 'scratching_knee_left', 'scratching_upper_arm_left', 'scratching_upper_arm_right', 'scratching_forearm_left', 'scratching_forearm_right']: #'wiping_face', 'scratching_knee_left', 'scratching_forearm_left','scratching_upper_arm_left']:#'scratching_knee_left', 'scratching_knee_right', 'scratching_thigh_left', 'scratching_thigh_right']:
        for task in ['hospital_gown']:
            subject = 'any_subject'
            #rospy.init_node(''.join(['data_reader_', subject, '_', model, '_', task]))
            this_start_time = time.time()
            shaving_data_reader = DataReader_Task(task, model, optimization, visualize=True)
            shaving_data_reader.generate_score()
            print 'Done! Time to generate all scores for this task: %fs' % (time.time() - this_start_time)
        print 'Done! Time to generate all scores for all tasks: %fs' % (time.time() - full_start_time)
