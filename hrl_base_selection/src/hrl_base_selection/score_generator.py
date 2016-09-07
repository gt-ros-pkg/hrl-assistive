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
from hrl_base_selection.srv import BaseMove#, BaseMove_multi
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


class ScoreGenerator(object):

    def __init__(self, visualize=False, targets='all_goals', reference_names=['head'], goals=None, model='autobed',
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
        # The reference frame for the pr2 base link
        pr2_B_base_link = np.matrix([[       1.,        0.,   0.,         0.0],
                                     [       0.,        1.,   0.,         0.0],
                                     [       0.,        0.,   1.,         0.0],
                                     [       0.,        0.,   0.,         1.0]])
        pr2_B_head = []
        # Sets the wheelchair location based on the location of the head using a few homogeneous transforms.
        # This is only used to visualize in rviz, because the visualization is done before initializing openrave
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
            calfl = self.autobed.GetLink('calf_left_link')
            calfr = self.autobed.GetLink('calf_right_link')
            ch = self.autobed.GetLink('upper_body_link')
            pr2_B_ual = np.matrix(ual.GetTransform())
            pr2_B_uar = np.matrix(uar.GetTransform())
            pr2_B_fal = np.matrix(fal.GetTransform())
            pr2_B_far = np.matrix(far.GetTransform())
            pr2_B_thl = np.matrix(thl.GetTransform())
            pr2_B_thr = np.matrix(thr.GetTransform())
            pr2_B_calfl = np.matrix(calfl.GetTransform())
            pr2_B_calfr = np.matrix(calfr.GetTransform())
            pr2_B_ch = np.matrix(ch.GetTransform())
        else:
            print 'I GOT A BAD MODEL. NOT SURE WHAT TO DO NOW!'
        pr2_B_head = np.matrix(headmodel.GetTransform())

        #
        # # Sets the wheelchair location based on the location of the head using a few homogeneous transforms.
        # # This is only used to visualize in rviz, because the visualization is done before initializing openrave
        # if self.model == 'chair':
        #     pr2_B_head = np.matrix([[       1.,        0.,   0.,         0.0],
        #                             [       0.,        1.,   0.,         0.0],
        #                             [       0.,        0.,   1.,     1.30626],  # 1.33626
        #                             [       0.,        0.,   0.,         1.0]])
        # if self.model == 'bed':
        #     an = -m.pi/4
        #     pr2_B_head = np.matrix([[ m.cos(an),  0., m.sin(an),     0.], #.45 #.438
        #                             [        0.,  1.,        0.,     0.], # 0.34 #.42
        #                             [-m.sin(an),  0., m.cos(an), 1.1546],
        #                             [        0.,  0.,        0.,     1.]])
        # if self.model == 'autobed':
        #     pr2_B_head = np.matrix([[   2.59156317e-01,   2.12275759e-04,  -9.65835368e-01,              0.], # .45 #.438
        #                             [  -2.12275759e-04,   9.99999964e-01,   1.62826039e-04,              0.], # 0.34 #.42
        #                             [   9.65835368e-01,   1.62826039e-04,   2.59156353e-01,  6.85000000e-01],
        #                             [        0.,  0.,        0.,     1.]])
        #     pr2_B_head = np.matrix([[  2.58031289e-01,  -1.01870021e-04,  -9.66136555e-01,  2.94000000e-01],
        #                             [  1.33691899e-05,   9.99999995e-01,  -1.01870021e-04,  1.45571068e-16],
        #                             [  9.66136561e-01,   1.33691899e-05,   2.58031289e-01,  6.82000000e-01],
        #                             [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,  1.00000000e+00]])

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
            elif y == 'knee_left':
                self.pr2_B_reference.append(pr2_B_calfl)
            elif y == 'knee_right':
                self.pr2_B_reference.append(pr2_B_calfr)
            elif y == 'chest':
                self.pr2_B_reference.append(pr2_B_ch)
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
        self.Tgrasps = []
        self.weights = []
        self.goal_list = []
        self.number_goals = len(self.Tgrasps)
        if self.goals is None:
            TARGETS =  np.array([[[0.252, -0.067, -0.021], [0.102, 0.771, 0.628, -0.002]],    # Face area
                                 [[0.252, -0.097, -0.021], [0.102, 0.771, 0.628, -0.002]],    # Face area
                                 [[0.252, -0.097, -0.061], [0.102, 0.771, 0.628, -0.002]],    # Face area
                                 [[0.252,  0.067, -0.021], [0.102, 0.771, 0.628, -0.002]],    # Face area
                                 [[0.252,  0.097, -0.061], [0.102, 0.771, 0.628, -0.002]],    # Face area
                                 [[0.252,  0.097, -0.021], [0.102, 0.771, 0.628, -0.002]],    # Face area
                                 [[0.108, -0.236, -0.105], [0.346, 0.857, 0.238,  0.299]],    # Shoulder area
                                 [[0.108, -0.256, -0.105], [0.346, 0.857, 0.238,  0.299]],    # Shoulder area
                                 [[0.443, -0.032, -0.716], [0.162, 0.739, 0.625,  0.195]],    # Knee area
                                 [[0.443, -0.032, -0.716], [0.162, 0.739, 0.625,  0.195]],    # Knee area
                                 [[0.337, -0.228, -0.317], [0.282, 0.850, 0.249,  0.370]],    # Arm area
                                 [[0.367, -0.228, -0.317], [0.282, 0.850, 0.249,  0.370]]])   # Arm area
            for target in TARGETS:
                #self.goal_list.append(pr2_B_head*createBMatrix(target[0],target[1])*goal_B_gripper)
                self.goal_list = []
                self.goal_list.append(pr2_B_head*createBMatrix(target[0], target[1]))
                self.goal_list = np.array(self.goal_list)
            self.choose_task(targets)
        else:
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

            #print 'The weight of all goals: \n',self.weights

            #print 'The list of goals from the score generator: \n',
            #for item in self.goal_list:
            #    print item
            #self.goal_list = goals

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



    def handle_score(self, plot=False):
        start_time = time.time()
        x_min = -1.0
        x_max = 2.0+.01
        x_int = 0.05
        y_min = -1.5
        y_max = 1.5 + .01
        y_int = 0.05
        theta_min = 0.
        theta_max = 2*m.pi-.01
        theta_int = m.pi/4
        z_min = 0.
        z_max = 0.30+.01
        z_int = 0.15
        bedz_min = 0.
        bedz_max = 0.2+.01
        bedz_int = 0.1
        bedtheta_min = 0.
        bedtheta_max = 70*m.pi/180+.1
        bedtheta_int = (bedtheta_max-.1)/2
        headx_min = 0.
        headx_max = 0.0+.01
        headx_int = 0.05
        heady_min = 0.
        heady_max = 0.1+.01
        heady_int = 10.05
        if self.model == 'autobed':
            x_int = .1
            y_int = .1
            theta_int = m.pi/2
        if self.model == 'chair':
            x_min = -1.5
            x_max = 1.5+.01
            bedz_min = 0.
            bedtheta_min = 0.
            headx_min = 0.
            heady_min = 0.
            bedz_int = 100.
            bedtheta_int = 100.
            headx_int = 100.
            heady_int = 100.
        print 'Starting to generate the score. This is going to take a while. Estimated 100+ seconds per goal location.'
        score_stuff = np.array([t for t in ((list(flatten([x, y, th, z, bz, bth, headx, heady,
                                                           self.generate_score(x, y, th, z, bz, bth, headx, heady)])))
                                            for x in np.arange(x_min, x_max, x_int)
                                            for y in np.arange(y_min, y_max, y_int)
                                            for th in np.arange(theta_min, theta_max, theta_int)
                                            for z in np.arange(z_min, z_max, z_int)
                                            for bz in np.arange(bedz_min, bedz_max, bedz_int)
                                            for bth in np.arange(bedtheta_min, bedtheta_max, bedtheta_int)
                                            for headx in np.arange(headx_min, headx_max, headx_int)
                                            for heady in np.arange(heady_min, heady_max, heady_int)
                                            )
                                ])

        print 'Time to generate all scores for individual base locations: %fs' % (time.time()-start_time)
        print 'Number of configurations that were evaluated: ', len(score_stuff)
        start_time = time.time()

        # Reduces the degrees of the score. This reduces the length of the score. We ultimately only care about the best
        # scores and we later only update based on x-y position. As a result, we only need one score per x-y location.
        # Need to save the other values for reference, but we don't need repeated x-y, just the best for each x-y.
        # Might need to keep certain DOF (like bed heights), but certainly don't need to keep thetas.
        # Keep DOF if we want the combination to only combine if those DOF match (or don't match). Remove DOF we don't
        # care particularly about for combinations.
        # Also get rid of items in score sheet with zero score.
        #for item in score_stuff:
        #    if item[7]>0:
        #        print item
        # self.reachable = {}
        # self.manipulable = {}
        # self.scores = {}
        # self.score_length = {}

        # quick_fix = True
        # if self.model == 'autobed':
        #     quick_fix = False
        if self.model == 'chair':
            del_index = []
            this_score = []
            this_reachable = []
            this_manipulable = []
            for t in xrange(len(score_stuff)):
                if score_stuff[t, 9] < 0.40:  # Only keeps base configurations with reachability score > .4
                    del_index.append(t)
            score_stuff = np.delete(score_stuff, del_index, 0)
            for aScore in score_stuff:
                reachable_line = []
                manipulable_line = []
                this_score.append(aScore[0:11])
                for number in xrange(int((len(aScore)-11)/2.)):
                    reachable_line.append(aScore[11+2*number])
                    manipulable_line.append(aScore[12+2*number])
                this_reachable.append(reachable_line)
                this_manipulable.append(manipulable_line)

            self.reachable[0., 0.] = np.array(this_reachable)
            self.manipulable[0., 0.] = np.array(this_manipulable)
            #print 'reachable ',self.reachable

            self.score_length[0., 0.] = len(this_score)
            self.scores[0., 0.] = np.array(this_score)
            self.sorted_scores[0., 0.] = np.array(sorted(this_score, key=lambda p: (p[9], p[10]), reverse=True))
            print 'The best score I found with single configuration is: ', self.sorted_scores[0., 0.][0][0:11]

        elif self.model == 'autobed':
            for hx in np.arange(headx_min, headx_max, headx_int):
                for hy in np.arange(heady_min, heady_max, heady_int):
                    self.scores[hx, hy] = []
                    self.reachable[hx, hy] = []
                    self.manipulable[hx, hy] = []
                    this_score = []
                    this_reachable = []
                    this_manipulable = []

                    temp_scores = []
                    del_index = []
                    s_len = copy.copy(len(score_stuff))
                    for t in xrange(s_len):
                        if np.array_equal(np.array([score_stuff[t, 6], score_stuff[t, 7]]), np.array([hx, hy])):
                            del_index.append(t)
                            if score_stuff[t, 9] >= 0.4:
                                temp_scores.append(score_stuff[t])
                    score_stuff = np.delete(score_stuff, del_index, 0)
                    for aScore in temp_scores:
                        reachable_line = []
                        manipulable_line = []
                        this_score.append(aScore[0:11])
                        for number in xrange(int((len(aScore)-11)/2.)):
                            reachable_line.append(aScore[11+2*number])
                            manipulable_line.append(aScore[12+2*number])
                        this_reachable.append(reachable_line)
                        this_manipulable.append(manipulable_line)

                    self.reachable[hx, hy] = np.array(this_reachable)
                    self.manipulable[hx, hy] = np.array(this_manipulable)
                    self.score_length[hx, hy] = len(this_score)
                    self.sorted_scores[hx, hy] = np.array(sorted(this_score, key=lambda p: (p[9], p[10]), reverse=True))
                    self.scores[hx, hy] = np.array(this_score)
                    #print 'The best score I found with single configuration is: ', self.sorted_scores[hx, hy][0][0:11]
                    #print 'at hx and hx: (', hx, ', ', hy, ')'
        else:
            print 'I GOT A BAD MODEL. WHAT MODEL SHOULD I BE USING? I DON\'T KNOW WHAT TO DO!!'
        #
        #
        # elif not quick_fix:
        #     for hx in np.arange(headx_min, headx_max, headx_int):
        #         for hy in np.arange(heady_min, heady_max, heady_int):
        #             self.scores[hx, hy] = []
        #             self.reachable[hx, hy] = []
        #             self.manipulable[hx, hy] = []
        #             this_score = []
        #             this_reachable = []
        #             this_manipulable = []
        #
        #             for x in np.arange(x_min, x_max, x_int):
        #                 for y in np.arange(y_min, y_max, y_int):
        #                     for th in np.arange(theta_min, theta_max, theta_int):
        #                         for bz in np.arange(bedz_min, bedz_max, bedz_int):
        #                             for bth in np.arange(bedtheta_min, bedtheta_max, bedtheta_int):
        #                                 best_score = []
        #                                 #print 'bz: ',bz
        #                                 temp_scores = []
        #                                 del_index = []
        #                                 s_len = copy.copy(len(score_stuff))
        #                                 for t in xrange(s_len):
        #                                 hx, hy    #t = s_len - k - 1
        #                                     #if bz == score_stuff[t,4] and bz>.11:
        #                                     #    if i==score_stuff[t,0] and i>.85:
        #                                     #        if score_stuff[t,7]>0:
        #                                     #            print 'X: ', score_stuff[t,0], np.round(i,4)
        #                                     #            print 'Y: ', score_stuff[t,1], np.round(j,4)
        #                                     #            print 'BedZ: ',score_stuff[t,4], np.round(bz,4)
        #                                     #            print 'BedTheta:', score_stuff[t,5], np.round(bth,4)
        #                                             #print score_stuff[t]
        #                                             #if bz == score_stuff[t,4]:
        #                                                 #print 'they are obviously the same'
        #
        #                                     #print i,' ',j,' ',k,' ',l,' ',score_stuff[t]
        #                                     #if score_stuff[t,7]>0:
        #                                         #if score_stuff[t,0]==.9 and i == .9:
        #                                             #if score_stuff[t,1]==-.6 and j == -.6:
        #                                                         #print 'X: ', score_stuff[t,0], np.round(i,4)
        #                                                         #print 'Y: ', score_stuff[t,1], np.round(j,4)
        #                                                         #print 'BedZ: ',score_stuff[t,4], np.round(bz,4)
        #                                                         #print 'BedTheta:', score_stuff[t,5], np.round(bth,4)
        #
        #                                     # if ((score_stuff[t,0]==np.round(i,4)) and (score_stuff[t,1]==np.round(j,4)) and
        #                                     # (score_stuff[t,4]==np.round(bz,4)) and (score_stuff[t,5]==np.round(bth,4))):
        #                                     if np.array_equal(np.round([score_stuff[t, 0], score_stuff[t, 1],
        #                                                                 score_stuff[t, 2], score_stuff[t, 4],
        #                                                                 score_stuff[t, 5], score_stuff[t, 6],
        #                                                                 score_stuff[t, 7]], 4),
        #                                                       np.round([x, y, th, bz, bth, hx, hy], 4)):
        #                                     # if "{0:.3f}".format(score_stuff[t, 0]) == "{0:.3f}".format(x) and \
        #                                     #    "{0:.3f}".format(score_stuff[t, 1]) == "{0:.3f}".format(y) and \
        #                                     #    "{0:.3f}".format(score_stuff[t, 4]) == "{0:.3f}".format(bz) and \
        #                                     #    "{0:.3f}".format(score_stuff[t, 5]) == "{0:.3f}".format(bth) and \
        #                                     #    "{0:.3f}".format(score_stuff[t, 6]) == "{0:.3f}".format(bth) and \
        #                                     #    "{0:.3f}".format(score_stuff[t, 7]) == "{0:.3f}".format(bth) and \
        #                                     #    "{0:.3f}".format(score_stuff[t, 6]) == "{0:.3f}".format(hx) and \
        #                                     #    "{0:.3f}".format(score_stuff[t, 7]) == "{0:.3f}".format(hy):
        #
        #                                         if score_stuff[t, 9] > 0.80:
        #                                             #print score_stuff[t]
        #                                             #print 'raw things:'
        #                                             #print score_stuff[t]
        #                                             temp_scores.append(score_stuff[t])
        #                                         del_index.append(t)
        #                                         #score_stuff = np.delete(score_stuff,t,0)
        #                                 score_stuff = np.delete(score_stuff, del_index, 0)
        #                                 if temp_scores != []:
        #                                     #print 'temp scores:'
        #                                     #for item in temp_scores:
        #                                     #    print item
        #
        #                                     best_score = copy.copy(np.array(sorted(temp_scores, key=itemgetter(9, 10),
        #                                                                             reverse=True))[0])
        #                                     this_score.append(best_score[0:11])
        #                                     # self.scores[hx, hy].append(temp_scores[0:11])
        #                                     reachable_line = []
        #                                     manipulable_line = []
        #                                     #print 'I was able to find a base location where I can reach at least one goal'
        #                                     for number in xrange(int((len(best_score)-11)/2.)):
        #                                         reachable_line.append(best_score[11+2*number])
        #                                         manipulable_line.append(best_score[12+2*number])
        #                                     this_reachable.append(reachable_line)
        #                                     this_manipulable.append(manipulable_line)
        #                                     # self.reachable[hx, hy].append(reachable_line)
        #                                     # self.manipulable[hx, hy].append(manipulable_line)
        #             self.reachable[hx, hy] = np.array(this_reachable)
        #             self.manipulable[hx, hy] = np.array(this_manipulable)
        #             #print 'reachable ',self.reachable
        #
        #             self.score_length[hx, hy] = len(this_score)
        #             self.scores[hx, hy] = np.array(this_score)

        self.number_goals = len(self.Tgrasps)
        #print 'scores:'
        #for item in self.scores:
        #    if item[7]>0:
        #        print item

        print 'The number of base configurations with default body location with non-zero reach scores is: ', \
            self.score_length[0., 0.]
        print 'The number of goals is: ', self.number_goals
        there_is_a_good_location = False
        for myScore in self.scores:
            if self.score_length[myScore] > 0.:
                there_is_a_good_location = True
        if not there_is_a_good_location:
            print 'There are no base locations with a score greater than 0. There are no good base locations!!'
            return [[[0], [0], [0], [0], [0], [0]], [0, 0, 0]]
        max_base_locations = np.min([3, self.number_goals+1])
        print 'Time to manage data sets and eliminate base configurations with zero reach score: %fs'%(time.time()-start_time)
        start_time = time.time()
        print 'Now starting to look at multiple base location combinations. Checking ', max_base_locations-1, ' max ' \
              'number of bases in combination. This may take a long time as well.'


        mult_base_scores = {}

        for hx in np.arange(headx_min, headx_max, headx_int):
            for hy in np.arange(heady_min, heady_max, heady_int):
                if len(self.sorted_scores[hx, hy])>0:
                    self.best_score = []
                    self.best_score.append([self.sorted_scores[hx, hy][0][9], self.sorted_scores[hx, hy][0][10]])
                    mult_base_scores[hx, hy] = np.array([t for t in ((list([self.get_xyths(comb_nums, hx, hy), self.combination_score(comb_nums,hx,hy)]))
                                                                      for num_base_locations in xrange(1, max_base_locations)
                                                                      for comb_nums in comb(xrange(self.score_length[hx,hy]),num_base_locations)
                                                                     )
                                                         if ((t[1]!=None) and (t[0]!=None))
                                                         ])
                    mult_base_scores[hx, hy] = np.array(sorted(mult_base_scores[hx, hy], key=lambda t: (t[1][1], t[1][2]), reverse=True))
                else:
                    print 'At hx, hy =', hx, ',', hy, 'There were no base configurations that could reach at least 0.4 of the goals.'
                    mult_base_scores[hx, hy] = np.array([[[0], [0], [0], [0], [0], [0]], [0, 0, 0]])
        print 'Time to generate all scores for combinations of base locations: %fs' % (time.time()-start_time)

        #print mult_base_scores

        if plot:
            print 'I am now going to plot the scoresheet for individual base locations for the default body location.'
            self.plot_scores(np.array(self.scores[0., 0.]))
        if self.score_length[0., 0.] == 0:
            default_is_zero = True
        else:
            default_is_zero = False
        return mult_base_scores, default_is_zero

    def combination_score(self, config_selections, hx, hz):

        this_reachable = np.zeros(self.number_goals)
        this_manipulable = np.zeros(self.number_goals)
        this_personal_space = np.max([q for q in ((self.scores[hx, hz][cs][8])
                                                  for cs in config_selections
                                                  )
                                      ])
        for g in xrange(self.number_goals):
            this_reachable[g] = np.max([q for q in ((self.reachable[hx, hz][cs][g])
                                                    for cs in config_selections
                                                    )
                                        ])
            this_manipulable[g] = np.max([q for q in ((self.manipulable[hx, hz][cs][g])
                                                      for cs in config_selections
                                                      )
                                          ])
        comparison = np.max([q for q in ((np.sum(self.manipulable[hx, hz][cs]))
                                         for cs in config_selections
                                         )
                             ])
        if len(config_selections) == 1:
            return [this_personal_space, np.sum(this_reachable), np.sum(this_manipulable)]
        elif np.sum(this_manipulable) >= 1.01*self.best_score[0][1] and np.sum(this_reachable) >= .98*self.best_score[0][0] and np.sum(this_manipulable) >= comparison*1.01:
            return [this_personal_space, np.sum(this_reachable), np.sum(this_manipulable)]
        else:
            return None

    def get_xyths(self, config_selections, hx, hz):
        this_x = []
        this_y = []
        this_theta = []
        this_z = []
        this_bz = []
        this_btheta = []
        too_close = False
        for sel in config_selections:
            this_x.append(round(self.scores[hx, hz][sel][0], 3))
            this_y.append(round(self.scores[hx, hz][sel][1], 3))
            this_theta.append(self.scores[hx, hz][sel][2])
            this_z.append(self.scores[hx, hz][sel][3])
            this_bz.append(self.scores[hx, hz][sel][4])
            this_btheta.append(self.scores[hx, hz][sel][5])
        comparison = np.vstack([this_x, this_y, this_bz, this_btheta, this_theta, this_z])
        for choice in xrange(len(config_selections)-1):
            # diff_xy = np.linalg.norm(comparison[0:2, choice]-comparison[0:2, choice+1])
            diff_xy = np.linalg.norm(comparison[0:2, choice]-comparison[0:2, choice+1])
            # diff_all = np.linalg.norm(comparison[0:4, choice]-comparison[0:4, choice+1])
            if diff_xy < .4:
                too_close = True
        if too_close and len(config_selections) > 1:
            return None
        else:
            return [this_x, this_y, this_theta, this_z, this_bz, this_btheta]

    def generate_score(self, x, y, th, z, bz, bth, headx, heady):
        #print 'Calculating new score'
        #starttime = time.time()
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
                    # self.pr2_B_reference[thing] = np.matrix(self.robot.GetTransform())

            for thing in xrange(len(self.goals)):
                self.goal_list[thing] = copy.copy(origin_B_pr2*self.pr2_B_reference[int(self.reference_mat[thing])]*np.matrix(self.goals[thing, 0]))
                self.selection_mat[thing] = copy.copy(self.goals[thing, 1])
#            for target in self.goals:
#                self.goal_list.append(pr2_B_head*np.matrix(target[0]))
#                self.selection_mat.append(target[1])
            self.set_goals()

        if self.model == 'chair':
            headmodel = self.wheelchair.GetLink('head_center')
        elif self.model == 'autobed':
            headmodel = self.autobed.GetLink('head_link')

        else:
            print 'I GOT A BAD MODEL. NOT SURE WHAT TO DO NOW!'
        pr2_B_head = np.matrix(headmodel.GetTransform())

        if self.model == 'autobed':
            self.selection_mat = np.zeros(len(self.goals))
            self.goal_list = np.zeros([len(self.goals), 4, 4])
            self.set_autobed(bz, bth, headx, heady)
            headmodel = self.autobed.GetLink('head_link')
            ual = self.autobed.GetLink('arm_left_link')
            uar = self.autobed.GetLink('arm_right_link')
            fal = self.autobed.GetLink('forearm_left_link')
            far = self.autobed.GetLink('forearm_right_link')
            thl = self.autobed.GetLink('quad_left_link')
            thr = self.autobed.GetLink('quad_right_link')
            calfl = self.autobed.GetLink('calf_left_link')
            calfr = self.autobed.GetLink('calf_right_link')
            ch = self.autobed.GetLink('upper_body_link')
            origin_B_head = np.matrix(headmodel.GetTransform())
            origin_B_ual = np.matrix(ual.GetTransform())
            origin_B_uar = np.matrix(uar.GetTransform())
            origin_B_fal = np.matrix(fal.GetTransform())
            origin_B_far = np.matrix(far.GetTransform())
            origin_B_thl = np.matrix(thl.GetTransform())
            origin_B_thr = np.matrix(thr.GetTransform())
            origin_B_calfl = np.matrix(calfl.GetTransform())
            origin_B_calfr = np.matrix(calfr.GetTransform())
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
                elif self.reference_names[thing] == 'knee_left':
                    self.pr2_B_reference[thing] = origin_B_pr2.I*origin_B_calfl
                elif self.reference_names[thing] == 'knee_right':
                    self.pr2_B_reference[thing] = origin_B_pr2.I*origin_B_calfr
                elif self.reference_names[thing] == 'chest':
                    self.pr2_B_reference[thing] = origin_B_pr2.I*origin_B_ch

            for thing in xrange(len(self.goals)):
                self.goal_list[thing] = copy.copy(origin_B_pr2*self.pr2_B_reference[int(self.reference_mat[thing])]*np.matrix(self.goals[thing, 0]))
                self.selection_mat[thing] = copy.copy(self.goals[thing, 1])
            # for target in self.goals:
            #     self.goal_list.append(pr2_B_head*np.matrix(target[0]))
            #     self.selection_mat.append(target[1])
            self.set_goals()
        #print 'Time to update autobed things: %fs'%(time.time()-starttime)
        reach_score = 0.
        manip_score = 0.
        goal_scores = []
        std = 1.
        mean = 0.
        # allmanip = []
        manip = 0.
        reached = 0.

        #allmanip2=[]
        space_score = (1./(std*(m.pow((2.*m.pi), 0.5))))*m.exp(-(m.pow(np.linalg.norm([x, y])-mean, 2.)) /
                                                               (2.*m.pow(std, 2.)))
        #print space_score
        with self.robot:
            if not self.manip.CheckIndependentCollision(op.CollisionReport()):
                #print 'not colliding with environment'
                for num, Tgrasp in enumerate(self.Tgrasps):
                    # sol = None
                    # sol = self.manip.FindIKSolution(Tgrasp, filteroptions=op.IkFilterOptions.CheckEnvCollisions)

                    v = self.robot.GetActiveDOFValues()
                    v[self.robot.GetJoint('r_shoulder_pan_joint').GetDOFIndex()] = -3.14/2
                    v[self.robot.GetJoint('r_shoulder_lift_joint').GetDOFIndex()] = -0.52
                    v[self.robot.GetJoint('r_upper_arm_roll_joint').GetDOFIndex()] = 0.
                    v[self.robot.GetJoint('r_elbow_flex_joint').GetDOFIndex()] = -3.14*2/3
                    v[self.robot.GetJoint('r_forearm_roll_joint').GetDOFIndex()] = 0.
                    v[self.robot.GetJoint('r_wrist_flex_joint').GetDOFIndex()] = 0.
                    v[self.robot.GetJoint('r_wrist_roll_joint').GetDOFIndex()] = 0.
                    self.robot.SetActiveDOFValues(v)
                    self.env.UpdatePublishedBodies()

                    #sol = self.manip.FindIKSolution(Tgrasp,filteroptions=op.IkFilterOptions.IgnoreSelfCollisions)
                    sols = []
                    sols = self.manip.FindIKSolutions(Tgrasp, filteroptions=op.IkFilterOptions.CheckEnvCollisions)
                    if sols == []:
                        v = self.robot.GetActiveDOFValues()
                        v[self.robot.GetJoint('r_shoulder_pan_joint').GetDOFIndex()] = -0.023593
                        v[self.robot.GetJoint('r_shoulder_lift_joint').GetDOFIndex()] = 1.1072800
                        v[self.robot.GetJoint('r_upper_arm_roll_joint').GetDOFIndex()] = -1.5566882
                        v[self.robot.GetJoint('r_elbow_flex_joint').GetDOFIndex()] = -2.124408
                        v[self.robot.GetJoint('r_forearm_roll_joint').GetDOFIndex()] = -1.4175
                        v[self.robot.GetJoint('r_wrist_flex_joint').GetDOFIndex()] = -1.8417
                        v[self.robot.GetJoint('r_wrist_roll_joint').GetDOFIndex()] = 0.21436
                        self.robot.SetActiveDOFValues(v)
                        self.env.UpdatePublishedBodies()
                        sols = self.manip.FindIKSolutions(Tgrasp, filteroptions=op.IkFilterOptions.CheckEnvCollisions)


                    manip = 0.
                    reached = 0.
                    if sols != []:  # not None:
                        # print 'sol is:', sol
                        # print 'sols are: \n', sols
                        #print 'I was able to find a grasp to this goal'
                        reached = 1.
                        for solution in sols:
                            self.robot.SetDOFValues(solution, self.manip.GetArmIndices())
                            # Tee = self.manip.GetEndEffectorTransform()
                            self.env.UpdatePublishedBodies()
                            #rospy.sleep(.2)

                            # reach_score += copy.copy(reached * self.weights[num])

                            # joint_angles = copy.copy(sol)
                            #pos, rot = self.robot.kinematics.FK(self.joint_angles)
                            #self.end_effector_position = pos
                            #self.end_effector_orient_cart = rot
                            #J = np.matrix(self.kinematics.jacobian(joint_angles))
                            #J = [self.robot.kinematics.jacobian(self.joint_angles, self.end_effector_position)]
                            J = np.matrix(np.vstack([self.manip.CalculateJacobian(), self.manip.CalculateAngularVelocityJacobian()]))
                            #print 'J0 ',np.linalg.norm(J[3:6,0])
                            #print 'J1 ',np.linalg.norm(J[3:6,1])
                            #print 'J2 ',np.linalg.norm(J[3:6,2])
                            #print 'J3 ',np.linalg.norm(J[3:6,3])
                            #print 'J4 ',np.linalg.norm(J[3:6,4])
                            #print 'J5 ',np.linalg.norm(J[3:6,5])
                            #print 'J6 ',np.linalg.norm(J[3:6,6])

                            #print 'Jacobian is: \n',J
                            #print Jop
                            #if np.array_equal(J,Jop):
                            #    print 'Jacobians are equal!!!'
                            try:
                                manip = np.max([copy.copy((m.pow(np.linalg.det(J*J.T), (1./6.)))/(np.trace(J*J.T)/6.)), manip])
                            except ValueError:
                                print 'Determinant of J*JT is: ', np.linalg.det(J*J.T)
                                manip = np.max([0., manip])
                            #manip2 = (m.pow(np.linalg.det(Jop*Jop.T),(1./6.)))/(np.trace(Jop*Jop.T)/6.)
                            # allmanip.append(manip)
                            #allmanip2.append(manip2)

                    manip_score += copy.copy(reached * manip*self.weights[num])
                    reach_score += copy.copy(reached * self.weights[num])
                    goal_scores.append([copy.copy(reached * self.weights[num]), copy.copy(manip*reached*self.weights[num])])
            else:
                for num in xrange(len(self.Tgrasps)):
                    goal_scores.append([0., 0.])
                    #print 'goal_scores: ', goal_scores
        #manip_score = manip_score/reachable
        #print 'I just tested base position: (', i,', ',j,', ',k,'). Reachable: ',reachable
        #if reachable !=0:
            #print 'The most manipulable reach with J was: ',np.max(allmanip)
            #print 'The most manipulable reach with Jop was: ',np.max(allmanip2)
        #    print 'weight was: ',self.weights
        #if reach_score>0:
            #print 'reach score was ',"{0:.3f}".format(reach_score)
        #print 'finished calculating a score'
        #print 'Time to calculate a score: %fs'%(time.time()-starttime)
        return space_score, reach_score, manip_score, goal_scores

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
                    calfl = self.autobed.GetLink('calf_left_link')
                    calfr = self.autobed.GetLink('calf_right_link')
                    ch = self.autobed.GetLink('upper_body_link')
                    origin_B_head = np.matrix(headmodel.GetTransform())
                    origin_B_ual = np.matrix(ual.GetTransform())
                    origin_B_uar = np.matrix(uar.GetTransform())
                    origin_B_fal = np.matrix(fal.GetTransform())
                    origin_B_far = np.matrix(far.GetTransform())
                    origin_B_thl = np.matrix(thl.GetTransform())
                    origin_B_thr = np.matrix(thr.GetTransform())
                    origin_B_calfl = np.matrix(calfl.GetTransform())
                    origin_B_calfr = np.matrix(calfr.GetTransform())
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
                        elif self.reference_names[thing] == 'knee_left':
                            self.pr2_B_reference[thing] = origin_B_pr2.I*origin_B_calfl
                        elif self.reference_names[thing] == 'knee_right':
                            self.pr2_B_reference[thing] = origin_B_pr2.I*origin_B_calfr
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
                            if sol:
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

    def mc_eval_init_config(self, init_config, goal_data):
        start_time = time.time()
        reached = 0.
        m_x_e = np.random.normal(0., 0.025/2)
        m_y_e = np.random.normal(0., 0.025/2)
        m_th_e = np.random.normal(0., (m.pi/36)/2)
        x_e = np.random.normal(0., 0.05)
        y_e = np.random.normal(0., 0.05)
        th_e = np.random.normal(0., (m.pi/36))
        h_e = np.random.normal(0., (m.pi/18)/2)

        # mod_x_err_min = -.025
        # mod_x_err_max = .025+.02
        # mod_x_err_int = .025
        # mod_y_err_min = -.025
        # mod_y_err_max = .025+.02
        # mod_y_err_int = .025
        # mod_th_err_min = -m.pi/36.
        # mod_th_err_max = m.pi/36.+.02
        # mod_th_err_int = m.pi/36.
        # x_err_min = -.05
        # x_err_max = .05+.02
        # x_err_int = .05
        # y_err_min = -.05
        # y_err_max = .05+.02
        # y_err_int = .05
        # th_err_min = -m.pi/36.
        # th_err_max = m.pi/36.+.02
        # th_err_int = m.pi/36.
        # h_err_min = -m.pi/9.
        # h_err_max = m.pi/9.+.02
        # h_err_int = m.pi/9.
        if self.model == 'chair':
            modeling_error = np.array([[x_e, y_e, th_e, h_e, m_x_e, m_y_e, m_th_e]])
            # print modeling_error[0]
            # modeling_error = np.array([err for err in ([x_e, y_e, th_e, h_e, m_x_e, m_y_e, m_th_e]
            #                                            for x_e in np.arange(x_err_min, x_err_max, x_err_int)
            #                                            for y_e in np.arange(y_err_min, y_err_max, y_err_int)
            #                                            for th_e in np.arange(th_err_min, th_err_max, th_err_int)
            #                                            for h_e in np.arange(h_err_min, h_err_max, h_err_int)
            #                                            for m_x_e in np.arange(mod_x_err_min, mod_x_err_max, mod_x_err_int)
            #                                            for m_y_e in np.arange(mod_y_err_min, mod_y_err_max, mod_y_err_int)
            #                                            for m_th_e in np.arange(mod_th_err_min, mod_th_err_max, mod_th_err_int)
            #                                            )
            #                            ])
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
            modeling_error = np.array([[x_e, y_e]])
            # modeling_error = np.array([err for err in ([x_e, y_e]
            #                                            for x_e in np.arange(x_err_min, x_err_max, x_err_int)
            #                                            for y_e in np.arange(y_err_min, y_err_max, y_err_int)
            #                                            )
            #                            ])
        # print len(modeling_error)
        # for error in modeling_error:
        #     print error

        total_length = len(goal_data)
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
                    calfl = self.autobed.GetLink('calf_left_link')
                    calfr = self.autobed.GetLink('calf_right_link')
                    ch = self.autobed.GetLink('upper_body_link')
                    origin_B_head = np.matrix(headmodel.GetTransform())
                    origin_B_ual = np.matrix(ual.GetTransform())
                    origin_B_uar = np.matrix(uar.GetTransform())
                    origin_B_fal = np.matrix(fal.GetTransform())
                    origin_B_far = np.matrix(far.GetTransform())
                    origin_B_thl = np.matrix(thl.GetTransform())
                    origin_B_thr = np.matrix(thr.GetTransform())
                    origin_B_calfl = np.matrix(calfl.GetTransform())
                    origin_B_calfr = np.matrix(calfr.GetTransform())
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
                        elif self.reference_names[thing] == 'knee_left':
                            self.pr2_B_reference[thing] = origin_B_pr2.I*origin_B_calfl
                        elif self.reference_names[thing] == 'knee_right':
                            self.pr2_B_reference[thing] = origin_B_pr2.I*origin_B_calfr
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
                            if sol:
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
        # if score < 0.5:
        #     print 'Score was less than 0.5. The error added was: ', modeling_error
        # print 'Score is (% of reached goals): ', score
        # print 'Time to score this initial configuration: %fs' % (time.time()-start_time)
        return score

    def setup_openrave(self):
        # Setup Openrave ENV
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
        v = self.robot.GetActiveDOFValues()
        v[self.robot.GetJoint('l_shoulder_pan_joint').GetDOFIndex()] = 3.14/2
        v[self.robot.GetJoint('r_shoulder_pan_joint').GetDOFIndex()] = -3.14/2
        v[self.robot.GetJoint('r_shoulder_lift_joint').GetDOFIndex()] = -0.52
        v[self.robot.GetJoint('r_upper_arm_roll_joint').GetDOFIndex()] = 0.
        v[self.robot.GetJoint('r_elbow_flex_joint').GetDOFIndex()] = -3.14*2/3
        v[self.robot.GetJoint('r_forearm_roll_joint').GetDOFIndex()] = 0.
        v[self.robot.GetJoint('r_wrist_flex_joint').GetDOFIndex()] = 0.
        v[self.robot.GetJoint('r_wrist_roll_joint').GetDOFIndex()] = 0.
        v[self.robot.GetJoint('torso_lift_joint').GetDOFIndex()] = .3
        self.robot.SetActiveDOFValues(v)
        robot_start = np.matrix([[m.cos(0.), -m.sin(0.), 0., 0.],
                                 [m.sin(0.),  m.cos(0.), 0., 0.],
                                 [0.       ,         0., 1., 0.],
                                 [0.       ,         0., 0., 1.]])
        self.robot.SetTransform(np.array(robot_start))

        ## Set robot manipulators, ik, planner
        self.robot.SetActiveManipulator('leftarm')
        self.manip = self.robot.GetActiveManipulator()
        ikmodel = op.databases.inversekinematics.InverseKinematicsModel(self.robot, iktype=op.IkParameterization.Type.Transform6D)
        if not ikmodel.load():
            ikmodel.autogenerate()
        # create the interface for basic manipulation programs
        self.manipprob = op.interfaces.BaseManipulation(self.robot)

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
            self.env.Load(''.join([pkg_path, '/collada/wheelchair_and_body_assembly.dae']))
            self.wheelchair = self.env.GetRobots()[1]
            headmodel = self.wheelchair.GetLink('head_center')
            head_T = np.matrix(headmodel.GetTransform())
            self.originsubject_B_headfloor = np.matrix([[1., 0.,  0., head_T[0, 3]],  # .442603 #.45 #.438
                                                        [0., 1.,  0., head_T[1, 3]],  # 0.34 #.42
                                                        [0., 0.,  1.,           0.],
                                                        [0., 0.,  0.,           1.]])
            self.originsubject_B_originworld = np.matrix(np.eye(4))
        elif self.model == 'bed':
            self.env.Load(''.join([pkg_path, '/models/head_bed.dae']))
            an = 0#m.pi/2
            self.originsubject_B_headfloor = np.matrix([[ m.cos(an),  0., m.sin(an),  .2954], #.45 #.438
                                                        [        0.,  1.,        0.,     0.], #0.34 #.42
                                                        [-m.sin(an),  0., m.cos(an),     0.],
                                                        [        0.,  0.,        0.,     1.]])
            self.originsubject_B_originworld = copy.copy(self.originsubject_B_headfloor)
        elif self.model == 'autobed':
            self.env.Load(''.join([pkg_path, '/collada/bed_and_body_v3_rounded.dae']))
            self.autobed = self.env.GetRobots()[1]
            v = self.autobed.GetActiveDOFValues()

            #0 degrees, 0 height
            v[self.autobed.GetJoint('head_rest_hinge').GetDOFIndex()] = 0.0
            v[self.autobed.GetJoint('tele_legs_joint').GetDOFIndex()] = -0.
            v[self.autobed.GetJoint('head_bed_to_worldframe_joint').GetDOFIndex()] = 0.
            v[self.autobed.GetJoint('head_bed_to_bedframe_joint').GetDOFIndex()] = 0.
            v[self.autobed.GetJoint('neck_body_joint').GetDOFIndex()] = -.1
            v[self.autobed.GetJoint('upper_mid_body_joint').GetDOFIndex()] = .4
            v[self.autobed.GetJoint('mid_lower_body_joint').GetDOFIndex()] = -.72
            v[self.autobed.GetJoint('body_quad_left_joint').GetDOFIndex()] = -0.4
            v[self.autobed.GetJoint('body_quad_right_joint').GetDOFIndex()] = -0.4
            v[self.autobed.GetJoint('quad_calf_left_joint').GetDOFIndex()] = 0.1
            v[self.autobed.GetJoint('quad_calf_right_joint').GetDOFIndex()] = 0.1
            v[self.autobed.GetJoint('calf_foot_left_joint').GetDOFIndex()] = .02
            v[self.autobed.GetJoint('calf_foot_right_joint').GetDOFIndex()] = .02
            v[self.autobed.GetJoint('body_arm_left_joint').GetDOFIndex()] = -.12
            v[self.autobed.GetJoint('body_arm_right_joint').GetDOFIndex()] = -.12
            v[self.autobed.GetJoint('arm_forearm_left_joint').GetDOFIndex()] = 0.05
            v[self.autobed.GetJoint('arm_forearm_right_joint').GetDOFIndex()] = 0.05
            v[self.autobed.GetJoint('forearm_hand_left_joint').GetDOFIndex()] = -0.1
            v[self.autobed.GetJoint('forearm_hand_right_joint').GetDOFIndex()] = -0.1
            #v[self.autobed.GetJoint('leg_rest_upper_joint').GetDOFIndex()]= -0.1
            self.autobed.SetActiveDOFValues(v)
            self.env.UpdatePublishedBodies()
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
        self.subject = self.env.GetBodies()[1]
        # self.subject_location = originsubject_B_headfloor.I
        self.subject.SetTransform(np.array(self.originsubject_B_originworld))

        print 'OpenRave has succesfully been initialized. \n'

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
            v[self.autobed.GetJoint('head_bed_to_worldframe_joint').GetDOFIndex()] = -((bth/40)*(0.6981317 - 0)+0)
            v[self.autobed.GetJoint('head_bed_to_bedframe_joint').GetDOFIndex()] = ((bth/40)*(0.6981317 - 0)+0)
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
            v[self.autobed.GetJoint('head_bed_to_worldframe_joint').GetDOFIndex()] = -(((bth-40)/40)*(1.3962634 - 0.6981317)+0.6981317)
            v[self.autobed.GetJoint('head_bed_to_bedframe_joint').GetDOFIndex()] = (((bth-40)/40)*(1.3962634 - 0.6981317)+0.6981317)
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
        vis_pub = rospy.Publisher('~goal_markers', MarkerArray, latch=True)
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
        vis_pub = rospy.Publisher(''.join(['~', name]), Marker, latch=True)
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
        else:
            print 'I got a bad model. What is going on???'
            return None
        vis_pub = rospy.Publisher(''.join(['~',name]), Marker, latch=True)
        marker.ns = ''.join(['base_service_',name])
        vis_pub.publish(marker)
        print 'Published a model of the subject to rviz'

    # Plot the score as a scatterplot heat map
    def plot_scores(self,scores):
        #print 'score_sheet:',scores
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('hrl_base_selection')
        data=scores
        '''
        score2d_temp = []
        for i in np.arange(-1.5,1.55,.05):
            for j in np.arange(-1.5,1.55,.05):
                temp = []
                for item in data:
                    newline = []
                #print 'i is:',i
                #print 'j is:',j
                    if item[0]==i and item[1]==j:
                        newline.append([i,j,item[3]])
                        newline.append(item[int(4)])
                        newline.append(item[int(5)])
                        #print 'newest line ',list(flatten(newline))
                        temp.append(list(flatten(newline)))
                if temp != []:
                    temp=np.array(temp)
                    temp_max = []
                    temp_max.append(np.max(temp[:,2]))
                    temp_max.append(np.max(temp[:,3]))
                    temp_max.append(np.max(temp[:,4]))
                    #print 'temp_max is ',temp_max
                    score2d_temp.append(list(flatten([i,j,temp_max])))
        #print '2d score:',np.array(score2d_temp)[0]
        seen_items = []
        score2d = []
        for item in score2d_temp:
            if not (any((item == x) for x in seen_items)):
                score2d.append(item)
                seen_items.append(item)
        score2d = np.array(score2d)
        #print 'score2d with no repetitions',score2d
        '''
        if self.model == 'chair':
            verts_subject = [(-.438, -.32885),  # left, bottom
                             (-.438, .32885),  # left, top
                             (.6397, .32885),  # right, top
                             (.6397, -.32885),  # right, bottom
                             (0., 0.),  # ignored
                             ]
        elif self.model == 'bed':
            verts_subject = [(-.2954, -.475),  # left, bottom
                             (-.2954, .475),  # left, top
                             (1.805, .475),  # right, top
                             (1.805, -.475),  # right, bottom
                             (0., 0.),  # ignored
                             ]
        elif self.model == 'autobed':
            verts_subject = [(-.2954, -.475),  # left, bottom
                             (-.2954, .475),  # left, top
                             (1.805, .475),  # right, top
                             (1.805, -.475),  # right, bottom
                             (0., 0.),  # ignored
                             ]

        verts_pr2 = [(-1.5,  -1.5),  # left, bottom
                     (-1.5, -.835),  # left, top
                     (-.835, -.835),  # right, top
                     (-.835,  -1.5),  # right, bottom
                     (0.,    0.),  # ignored
                     ]

        codes = [Path.MOVETO,
                 Path.LINETO,
                 Path.LINETO,
                 Path.LINETO,
                 Path.CLOSEPOLY,
                 ]

        path_subject = Path(verts_subject, codes)
        path_pr2 = Path(verts_pr2, codes)

        patch_subject = patches.PathPatch(path_subject, facecolor='orange', lw=2)
        patch_pr2 = patches.PathPatch(path_pr2, facecolor='orange', lw=2)

        X = data[:, 0]
        Y = data[:, 1]
        c3 = data[:, 4]

        fig3 = plt.figure(1)
        ax3 = fig3.add_subplot(111)
        surf3 = ax3.scatter(X, Y, s=60, c=c3, alpha=1)
        ax3.set_xlabel('X Axis')
        ax3.set_ylabel('Y Axis')
        fig3.colorbar(surf3, shrink=0.65, aspect=5)
        ax3.add_patch(patch_subject)
        ax3.add_patch(patch_pr2)
        ax3.set_xlim(-2, 2)
        ax3.set_ylim(-2, 2)
        fig3.set_size_inches(14, 11, forward=True)
        ax3.set_title(''.join(['Plot of personal space score on ', self.model, ' Time stamp: ', str(int(time.time()))]))
        plt.savefig(''.join([pkg_path, '/images/space_score_on_', self.model, '_ts_', str(int(time.time())), '.png']),
                    bbox_inches='tight')


        c = copy.copy(data[:,5])
        c2 = copy.copy(data[:,6])

        fig = plt.figure(2)
        ax = fig.add_subplot(111)
        surf = ax.scatter(X, Y, s=60, c=c, alpha=1)
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        fig.colorbar(surf, shrink=0.65, aspect=5)
        ax.add_patch(patch_subject)
        ax.add_patch(patch_pr2)
        ax.set_xlim(-2,2)
        ax.set_ylim(-2,2)
        fig.set_size_inches(14,11,forward=True)
        ax.set_title(''.join(['Plot of reach score on ',self.model,' Time stamp: ',str(int(time.time()))]))
        plt.savefig(''.join([pkg_path, '/images/reach_score_on_',self.model,'_ts_',str(int(time.time())),'.png']), bbox_inches='tight')

        fig2 = plt.figure(3)
        ax2 = fig2.add_subplot(111)
        surf2 = ax2.scatter(X, Y, s=60, c=c2, alpha=1)
        ax2.set_xlabel('X Axis')
        ax2.set_ylabel('Y Axis')
        fig2.colorbar(surf2, shrink=0.65, aspect=5)
        ax2.add_patch(patch_subject)
        ax2.add_patch(patch_pr2)
        ax2.set_xlim(-2, 2)
        ax2.set_ylim(-2, 2)
        fig2.set_size_inches(14, 11, forward=True)
        ax2.set_title(''.join(['Plot of manipulability score on ',self.model,' Time stamp: ',str(int(time.time()))]))
        plt.savefig(''.join([pkg_path, '/images/manip_score_on_',self.model,'_ts_',str(int(time.time())),'.png']), bbox_inches='tight')

        plt.ion()
        plt.show()
        ut.get_keystroke('Hit a key to proceed next')


if __name__ == "__main__":
    rospy.init_node('score_generator')
    mytask = 'shoulder'
    mymodel = 'chair'
    #mytask = 'all_goals'
    start_time = time.time()
    selector = ScoreGenerator(visualize=False,task=mytask,goals = None,model=mymodel)
    #selector.choose_task(mytask)
    score_sheet = selector.handle_score()

    print 'Time to load find generate all scores: %fs'%(time.time()-start_time)

    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('hrl_base_selection')
    save_pickle(score_sheet, ''.join([pkg_path, '/data/', mymodel, '_', mytask, '.pkl']))
    print 'Time to complete program, saving all data: %fs' % (time.time()-start_time)


    # Plot the score as a scatterplot heat map
    #print 'score_sheet:',score_sheet
    score2d_temp = []
    #print t
    for i in np.arange(-1.5, 1.55, .05):
        for j in np.arange(-1.5, 1.55, .05):
            temp = []
            for item in score_sheet:
            #print 'i is:',i
            #print 'j is:',j
                if item[0] == i and item[1] == j:
                    temp.append(item[3])
            if temp != []:
                score2d_temp.append([i, j, np.max(temp)])

    #print '2d score:',np.array(score2d_temp)

    seen_items = []
    score2d = []
    for item in score2d_temp:
#any((a == x).all() for x in my_list)
        #print 'seen_items is: ',seen_items
        #print 'item is: ',item
        #print (any((item == x) for x in seen_items))
        if not (any((item == x) for x in seen_items)):
        #if item not in seen_items:
            #print 'Just added the item to score2d'
            score2d.append(item)
            seen_items.append(item)
    score2d = np.array(score2d)
    #print 'score2d with no repetitions',score2d

    fig, ax = plt.subplots()

    X = score2d[:, 0]
    Y = score2d[:, 1]
    #Th = score_sheet[:,2]
    c = score2d[:, 2]
    #surf = ax.scatter(delta1[:-1], delta1[1:], c=close, s=volume, alpha=0.5)
    surf = ax.scatter(X, Y, s=60, c=c, alpha=1)
    #surf = ax.scatter(X, Y,s=40, c=c,alpha=.6)
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    #ax.set_zlabel('Theta Axis')

    fig.colorbar(surf, shrink=0.5, aspect=5)

    if mymodel == 'chair':
        verts_subject = [(-.438, -.32885),  # left, bottom
                         (-.438, .32885),  # left, top
                         (.6397, .32885),  # right, top
                         (.6397, -.32885),  # right, bottom
                         (0., 0.), # ignored
                         ]
    elif mymodel == 'bed':
        verts_subject = [(-.2954, -.475),  # left, bottom
                         (-.2954, .475),  # left, top
                         (1.805, .475),  # right, top
                         (1.805, -.475),  # right, bottom
                         (0., 0.),  # ignored
                         ]
    elif mymodel == 'autobed':
        verts_subject = [(-.2954, -.475),  # left, bottom
                         (-.2954, .475),  # left, top
                         (1.805, .475),  # right, top
                         (1.805, -.475),  # right, bottom
                         (0., 0.),  # ignored
                         ]

    verts_pr2 = [(-1.5,  -1.5),  # left, bottom
                 (-1.5, -.835),  # left, top
                 (-.835, -.835),  # right, top
                 (-.835,  -1.5),  # right, bottom
                 (0.,    0.),  # ignored
                ]

    codes = [Path.MOVETO,
             Path.LINETO,
             Path.LINETO,
             Path.LINETO,
             Path.CLOSEPOLY,
            ]

    path_subject = Path(verts_subject, codes)
    path_pr2 = Path(verts_pr2, codes)

    patch_subject = patches.PathPatch(path_subject, facecolor='orange', lw=2)
    patch_pr2 = patches.PathPatch(path_pr2, facecolor='orange', lw=2)

    ax.add_patch(patch_subject)
    ax.add_patch(patch_pr2)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)


    plt.show()





    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X  = score_sheet[:,0]
    Y  = score_sheet[:,1]
    Th = score_sheet[:,2]
    c  = score_sheet[:,3]
    surf = ax.scatter(X, Y, Th,s=40, c=c,alpha=.6)
    #surf = ax.scatter(X, Y,s=40, c=c,alpha=.6)
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Theta Axis')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
'''



