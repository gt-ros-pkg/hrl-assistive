#!/usr/bin/env python
import sys, optparse

import rospy, rospkg
import openravepy as op
import numpy as np
import math as m
import copy
from hrl_base_selection.srv import BaseMove_multi
import roslib; roslib.load_manifest('hrl_haptic_mpc')
import time
import roslib
roslib.load_manifest('hrl_base_selection')
roslib.load_manifest('hrl_haptic_mpc')
import rospy, rospkg
import tf
roslib.load_manifest('hrl_base_selection')
import hrl_lib.transforms as tr
import tf
import rospy
from visualization_msgs.msg import Marker
import time
from hrl_msgs.msg import FloatArrayBare
from helper_functions import createBMatrix
from data_reader import DataReader
from data_reader_task import DataReader_Task
from score_generator import ScoreGenerator
from pr2_controllers_msgs.msg import SingleJointPositionActionGoal
from config_visualize import ConfigVisualize
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist

import random

import pickle
import sPickle as pkl
roslib.load_manifest('hrl_lib')
from hrl_lib.util import load_pickle
import joblib


class Manipulability_Testing(object):
    def __init__(self, visualize_best=False, train_subj=6, test_subj=6):
        output_raw_scores = False

        # compare = True
        self.visualize_best = visualize_best

        self.tf_listener = tf.TransformListener()
        self.train_subj = train_subj
        self.test_subj = test_subj
        print 'I will use data that was trained on subject ', self.train_subj
        print 'I will test on data from subject ', self.test_subj

        self.task = 'shaving' # options are: bathing, brushing, feeding, shaving, scratching_upper_arm/forearm/thigh/chest/knee
        self.model = 'chair'  # options are: 'chair', 'bed', 'autobed'

        pos_clust = 2
        ori_clust = 2
        self.mc_simulation_number = None

        self.visualize = False
        data_start = 0
        data_finish = 'end '  # 2000  # 4000 #'end'

        rospack = rospkg.RosPack()
        self.pkg_path = rospack.get_path('hrl_base_selection')
        print 'Loading scores.'
        self.loaded_scores = self.load_task(self.task, self.model, self.train_subj)
        if self.loaded_scores is None:
            print 'The scores do not exist. Must generate scores! This may take a long time...'
            self.generate_scores(data_start, data_finish, pos_clust, ori_clust)
            print 'Scores generated. I will now continue.'
            print 'Now loading the scores I just generated'
            self.loaded_scores = self.load_task(self.task, self.model, self.train_subj)
        if self.loaded_scores is None:
            print 'The scores still do not exist. This is bad. Fixes needed in code.'
            return
        headx = 0
        heady = 0
        self.scores = self.loaded_scores[headx, heady]
        if output_raw_scores:
            self.output_scores()
        subject = ''.join(['sub', str(self.test_subj), '_shaver'])
        print 'Reading in raw data from the task.'
        read_task_data = DataReader_Task(self.task, self.model)
        raw_data, raw_num, raw_reference, self.raw_reference_options = read_task_data.reset_goals()
        read_data = DataReader(subject=subject, data_start=data_start, reference_options=self.raw_reference_options,
                               data_finish=data_finish, model=self.model, task=self.task, tf_listener=self.tf_listener)

        # raw_data = read_data.get_raw_data()
        print 'Raw data is ready!'
        self.goal_data = read_data.generate_output_goals(test_goals=raw_data, test_number=raw_num, test_reference=raw_reference)
        # print 'Setting up openrave'
        # self.setup_openrave()
        # print 'I will now pick base locations to evaluate. They will share the same reachability score, but will have' \
        #       ' differing manipulability scores.'
        # print 'before sorting:'
        # for i in xrange(10):
        #     print self.scores[i]
        self.scores = np.array(sorted(self.scores, key=lambda t: (t[1][1], t[1][2]), reverse=True))
        # print 'after sorting:'
        # for i in xrange(10):
        #     print self.scores[i]
        self.best_base = self.scores[0]
        if self.best_base[1][1] == 0:
            print 'There are no base locations with reachable goals. Something went wrong in the scoring or the setup'
        print 'The best base location is: \n', self.best_base

        if visualize_best:
            read_data.pub_rviz()
            self.visualize_base_config(self.best_base, self.goal_data, self.raw_reference_options)

    def get_best_base(self):
        return self.best_base

    def run_comparisons(self):
        print 'The number of base configurations on the score sheet for the default human position with ' \
              'score > 0 is: ', len(self.scores)
        comparison_bases = []
        max_num_of_comparisons = 400
        max_num_of_comparisons = np.min([max_num_of_comparisons, len(self.scores)])
        max_reach_count = 0
        print 'best base manip score is:', best_base[1][2]
        for i in xrange(len(self.scores)):
            if self.scores[i, 1][1] == best_base[1][1] and self.scores[i, 1][1] > 0:
                max_reach_count += 1
                # print self.scores[i,1][2]
        print 'The number of base configurations that can reach all clustered goals is: ', max_reach_count
        count = 0
        # max_num_of_comparisons = np.min([max_reach_count, max_num_of_comparisons])
        # for i in np.arange(int(max_reach_count/max_num_of_comparisons), int(max_reach_count*10),
        #                    int(max_reach_count/max_num_of_comparisons)):
        do_first = False
        # if max_num_of_comparisons < 400:
        #     do_all = True
        do_all = False
        custom = False
        anova_test = False
        mc_simulation_test = True
        if mc_simulation_test:
            print 'I will do a monte carlo simulation.'
            self.mc_simulation_number = 1000
            single_score = []
            for aScore in self.scores:
                if len(aScore[0][0]) == 1:
                    single_score.append(aScore)
            single_score = np.array(sorted(single_score, key=lambda t: (t[1][1], t[1][2]), reverse=True))
            self.best_single_base = single_score[0]
            print 'The best single configuration is: ', self.best_single_base
            single_score_selection = []
            multi_score_selection = []
            for aScore in self.scores:
                if (len(aScore[0][0]) == 1) and (aScore[1][1] >= self.best_single_base[1][1]*.98):
                    single_score_selection.append(aScore)
                elif (len(aScore[0][0]) >= 1) and (aScore[1][1] >= best_base[1][1]*.98):
                    multi_score_selection.append(aScore)
            multi_score_selection = np.array(multi_score_selection)
            single_score_selection = np.array(single_score_selection)
            print 'Number of single configurations with their max reachability: ', len(single_score_selection)
            print 'Number of multi configurations with their max reachability: ', len(multi_score_selection)
            single_score_sample = []
            multi_score_sample = []
            for sim_count in xrange(self.mc_simulation_number):
                single_score_sample.append(random.sample(single_score_selection, 1)[0])
                if len(multi_score_selection) > 0:
                    multi_score_sample.append(random.sample(multi_score_selection, 1)[0])

        if anova_test:
            single_score = []
            for aScore in self.scores:
                if len(aScore[0][0]) == 1:
                    single_score.append(aScore)
            single_score = np.array(sorted(single_score, key=lambda t: (t[1][1], t[1][2]), reverse=True))
            self.best_single_base = single_score[0]
            print 'The best single configuration is: ', self.best_single_base
            # print self.best_single_base[1][1]
            # print best_base[1][1]
            # print self.best_single_base[1][1]*.98
            # print best_base[1][1]*.98
            single_score_selection = []
            multi_score_selection = []
            for aScore in self.scores:
                if (len(aScore[0][0]) == 1) and (aScore[1][1] >= self.best_single_base[1][1]*.98):
                    single_score_selection.append(aScore)
                elif (len(aScore[0][0]) >= 1) and (aScore[1][1] >= best_base[1][1]*.98):
                    multi_score_selection.append(aScore)
            multi_score_selection = np.array(multi_score_selection)
            single_score_selection = np.array(single_score_selection)
            print 'Number of single configurations with their max reachability: ', len(single_score_selection)
            print 'Number of multi configurations with their max reachability: ', len(multi_score_selection)
            single_score_sample = np.array(random.sample(single_score_selection, np.min([100, len(single_score_selection)])))
            multi_score_sample = np.array(random.sample(multi_score_selection, np.min([100, len(multi_score_selection)])))
        if do_all:
            for j in xrange(len(self.scores)):
                i = j
                try:
                    if self.scores[int(i), 1][1] > 0:
                        comparison_bases.append(self.scores[i])
                except IndexError:
                    print 'there was an index error'
                    pass
        elif mc_simulation_test:
            for sample in single_score_sample:
                comparison_bases.append(sample)
            for sample in multi_score_sample:
                comparison_bases.append(sample)
        elif anova_test:
            for multi_score in multi_score_selection[0:1]:
                comparison_bases.append(multi_score)
            for single_score in single_score_selection[0:1]:
                comparison_bases.append(single_score)
            for sample in single_score_sample:
                comparison_bases.append(sample)
            for sample in multi_score_sample:
                comparison_bases.append(sample)
        elif custom:
            for j in xrange(1, max_num_of_comparisons):
                if j <= 10:
                    i = j
                elif j <= float(max_num_of_comparisons)/2.:
                    i = (j-10)*int(2*float(4000)/float(max_num_of_comparisons))+10
                elif j < 3.*float(max_num_of_comparisons)/4.:
                    i = (j-int(float(max_num_of_comparisons)/2.))+(len(self.scores)-2643)
                else:
                    i += 20
                try:
                    if self.scores[int(i), 1][1] > 0:
                        comparison_bases.append(self.scores[i])
                except IndexError:
                    print 'there was an index error'
                    pass
        elif do_first:
            for j in xrange(1, 11):
                i = j
                try:
                    if self.scores[int(i), 1][1] > 0:
                        comparison_bases.append(self.scores[i])
                except IndexError:
                    print 'there was an index error'
                    pass
        elif float(max_reach_count)/float(max_num_of_comparisons) > 0.5:
            for j in xrange(1, max_num_of_comparisons+1):
                if j <= 10:
                    i = j
                elif j <= int(float(max_num_of_comparisons)/2):
                    i = (j-10)*int(2*float(max_reach_count)/float(max_num_of_comparisons))+10
                else:
                    i = max_reach_count + (j-int(float(max_num_of_comparisons)/2))*int(float(len(self.scores) - max_reach_count)*2/(float(max_num_of_comparisons)))
                # print 'i: ', i
                try:
                    if self.scores[int(i), 1][1] > 0:
                        comparison_bases.append(self.scores[i])
                except IndexError:
                    print 'there was an index error'
                    pass

        else:
            for j in xrange(1, max_num_of_comparisons+1):
                if j <= max_reach_count:
                    i = j
                else:
                    i = (j-max_reach_count-1)*2+max_reach_count+1
                # print 'i: ', i
                try:
                    if self.scores[int(i), 1][1] > 0:
                        comparison_bases.append(self.scores[i])
                except IndexError:
                    print 'there was an index error'
                    pass
                # print self.scores[i,1][2]
                # if self.scores[i, 1][2] <= best_base[1][2]*(max_num_of_comparisons-count)/(max_num_of_comparisons+1):
                # if self.scores[i, 1][2] <= best_base[1][2]*m.pow(.9, count):
                    # comparison_bases.append(self.scores[i])
                    # count += 1
                    # if count > 5:
                    #     break
        print 'There are ', len(comparison_bases), 'number of comparison bases being evaluated'
        # print 'number of base configurations that can reach all 20 clustered goal locations: ',count
        # print 'The comparison base locations are:'
        # for item in comparison_bases:
        #     print item
        print 'I may now proceed with comparison evaluation of the base locations with differing manipulability score'
        self.evaluate_base_locations(best_base, self.best_single_base, comparison_bases, self.goal_data, self.raw_reference_options)

    def evaluate_base_locations(self, best_base, best_single_base, comparison_bases, goal_data, reference_options):
        visualize = False
        mytargets = 'all_goals'
        # reference_options = []
        # reference_options.append('head')
        myReferenceNames = reference_options
        myGoals = goal_data
        # best_multi_base[0][0] = [.95]
        # best_multi_base[0][1] = [-0.1]
        # best_multi_base[0][2] = [m.pi/2]
        # best_multi_base[0][3] = [0.3]
        # best_multi_base[0][4] = [0.]
        # best_multi_base[0][5] = [0.]
        selector = ScoreGenerator(visualize=visualize, targets=mytargets, reference_names=myReferenceNames,
                                  goals=myGoals, model=self.model, tf_listener=self.tf_listener)
        rospy.sleep(5)
        start_time = time.time()
        # selector.receive_new_goals(goal_data)
        # selector.show_rviz()
        if self.mc_simulation_number is not None:
            best_base_scores = []
            best_single_base_scores = []
            for sim_counter in xrange(self.mc_simulation_number):
                selector.receive_new_goals(goal_data)
                best_base_scores.append(selector.mc_eval_init_config(best_base, goal_data))
                selector.receive_new_goals(goal_data)
                best_single_base_scores.append(selector.mc_eval_init_config(best_single_base, goal_data))
            print 'There are ', len(best_base_scores), 'simulation scores for the best base configuration.'
            print 'There are ', len(best_single_base_scores), 'simulation scores for the best single base configuration.'
            print 'There are ', len(comparison_bases), 'comparison bases being put into simulation, including single and multi'
            comparison_base_scores = []
            for item in comparison_bases:
                selector.receive_new_goals(goal_data)
                comparison_base_scores.append(selector.mc_eval_init_config(item, goal_data))
        else:
            best_base_score = selector.eval_init_config(best_base, goal_data)
            print 'The score for the best base was: ', best_base_score
            comparison_base_scores = []
            for item in comparison_bases:
                selector.receive_new_goals(goal_data)
                comparison_base_scores.append(selector.eval_init_config(item, goal_data))
            print 'The best base location is: \n', best_base
            print 'The score for the best base was: ', best_base_score
        # for i in xrange(len(comparison_base_scores)):
            # print 'A comparison base score for base: \n', comparison_bases[i]
            # print 'The score was: ', comparison_base_scores[i]
        print 'Time to generate all scores for comparison and best bases: %fs' % (time.time()-start_time)
        if self.mc_simulation_number is not None:
            locations = open(''.join([self.pkg_path, '/data/', self.task, '_', self.model, '_base_configs_mc_sim.log']), 'w')
            if self.model == 'chair':
                for i in xrange(self.mc_simulation_number):
                    if len(best_base[0][0]) == 1:
                        locations.write(''.join([str(i), ' %f %f %f %f %f %f %f \n' % (best_base[0][0][0],
                                                                                       best_base[0][1][0],
                                                                                       best_base[0][2][0],
                                                                                       best_base[0][3][0],
                                                                                       best_base[1][1],
                                                                                       best_base[1][2],
                                                                                       best_base_scores[i])]))
                    elif len(best_base[0][0]) == 2:
                        locations.write(''.join([str(i), ' %f %f %f %f %f %f %f %f %f %f %f\n' % (best_base[0][0][0],
                                                                                                  best_base[0][1][0],
                                                                                                  best_base[0][2][0],
                                                                                                  best_base[0][3][0],
                                                                                                  best_base[1][1],
                                                                                                  best_base[1][2],
                                                                                                  best_base_scores[i],
                                                                                                  best_base[0][0][1],
                                                                                                  best_base[0][1][1],
                                                                                                  best_base[0][2][1],
                                                                                                  best_base[0][3][1]
                                                                                                  )]))
                for i in xrange(self.mc_simulation_number):
                    if len(best_single_base[0][0]) == 1:
                        locations.write(''.join([str(i), ' %f %f %f %f %f %f %f \n' % (best_single_base[0][0][0],
                                                                                       best_single_base[0][1][0],
                                                                                       best_single_base[0][2][0],
                                                                                       best_single_base[0][3][0],
                                                                                       best_single_base[1][1],
                                                                                       best_single_base[1][2],
                                                                                       best_single_base_scores[i])]))
                    elif len(best_base[0][0]) == 2:
                        print 'Something went horrible wrong, shouldn\'t have a multibase score in single base list'
                        locations.write('Something went horrible wrong, shouldn\'t have a multibase score in single base list')
                comparison_bases = np.array(comparison_bases)
                for i in xrange(len(comparison_bases)):
                    if len(comparison_bases[i, 0][0]) == 1:
                        locations.write(''.join([str(i), ' %f %f %f %f %f %f %f \n' % (comparison_bases[i, 0][0][0],
                                                                                       comparison_bases[i, 0][1][0],
                                                                                       comparison_bases[i, 0][2][0],
                                                                                       comparison_bases[i, 0][3][0],
                                                                                       comparison_bases[i, 1][1],
                                                                                       comparison_bases[i, 1][2],
                                                                                       comparison_base_scores[i])]))
                        # print 'printing comparison base'
                    elif len(comparison_bases[i, 0][0]) == 2:
                        locations.write(''.join([str(i), ' %f %f %f %f %f %f %f %f %f %f %f\n' % (comparison_bases[i, 0][0][0],
                                                                                                  comparison_bases[i, 0][1][0],
                                                                                                  comparison_bases[i, 0][2][0],
                                                                                                  comparison_bases[i, 0][3][0],
                                                                                                  comparison_bases[i, 1][1],
                                                                                                  comparison_bases[i, 1][2],
                                                                                                  comparison_base_scores[i],
                                                                                                  comparison_bases[i, 0][0][1],
                                                                                                  comparison_bases[i, 0][1][1],
                                                                                                  comparison_bases[i, 0][2][1],
                                                                                                  comparison_bases[i, 0][3][1]
                                                                                                  )]))
                        # print 'printing comparison base'
            elif self.model == 'autobed':
                for i in xrange(self.mc_simulation_number):
                    if len(best_base[0][0]) == 1:
                        locations.write(''.join([str(i), ' %f %f %f %f %f %f %f %f %f \n' % (best_base[0][0][0],
                                                                                             best_base[0][1][0],
                                                                                             best_base[0][2][0],
                                                                                             best_base[0][3][0],
                                                                                             best_base[0][4][0],
                                                                                             best_base[0][5][0],
                                                                                             best_base[1][1],
                                                                                             best_base[1][2],
                                                                                             best_base_scores[i])]))
                    elif len(best_base[0][0]) == 2:
                        locations.write(''.join([str(i), ' %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n' % (best_base[0][0][0],
                                                                                                              best_base[0][1][0],
                                                                                                              best_base[0][2][0],
                                                                                                              best_base[0][3][0],
                                                                                                              best_base[0][4][0],
                                                                                                              best_base[0][5][0],
                                                                                                              best_base[1][1],
                                                                                                              best_base[1][2],
                                                                                                              best_base_scores[i],
                                                                                                              best_base[0][0][1],
                                                                                                              best_base[0][1][1],
                                                                                                              best_base[0][2][1],
                                                                                                              best_base[0][3][1],
                                                                                                              best_base[0][4][1],
                                                                                                              best_base[0][5][1]
                        )]))
                for i in xrange(self.mc_simulation_number):
                    if len(best_single_base[0][0]) == 1:
                        locations.write(''.join([str(i), ' %f %f %f %f %f %f %f %f %f \n' % (best_single_base[0][0][0],
                                                                                             best_single_base[0][1][0],
                                                                                             best_single_base[0][2][0],
                                                                                             best_single_base[0][3][0],
                                                                                             best_single_base[0][4][0],
                                                                                             best_single_base[0][5][0],
                                                                                             best_single_base[1][1],
                                                                                             best_single_base[1][2],
                                                                                             best_single_base_scores[i])]))
                    elif len(best_base[0][0]) == 2:
                        print 'Something went horrible wrong, shouldn\'t have a multibase score in single base list'
                        locations.write('Something went horrible wrong, shouldn\'t have a multibase score in single base list')
                comparison_bases = np.array(comparison_bases)
                for i in xrange(len(comparison_bases)):
                    if len(comparison_bases[i, 0][0]) == 1:
                        locations.write(''.join([str(i), ' %f %f %f %f %f %f %f %f %f \n' % (comparison_bases[i, 0][0][0],
                                                                                             comparison_bases[i, 0][1][0],
                                                                                             comparison_bases[i, 0][2][0],
                                                                                             comparison_bases[i, 0][3][0],
                                                                                             comparison_bases[i, 0][4][0],
                                                                                             comparison_bases[i, 0][5][0],
                                                                                             comparison_bases[i, 1][1],
                                                                                             comparison_bases[i, 1][2],
                                                                                             comparison_base_scores[i])]))
                    elif len(comparison_bases[i, 0][0]) == 2:
                        locations.write(''.join([str(i), ' %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n' % (comparison_bases[i, 0][0][0],
                                                                                                              comparison_bases[i, 0][1][0],
                                                                                                              comparison_bases[i, 0][2][0],
                                                                                                              comparison_bases[i, 0][3][0],
                                                                                                              comparison_bases[i, 0][3][0],
                                                                                                              comparison_bases[i, 0][3][0],
                                                                                                              comparison_bases[i, 1][1],
                                                                                                              comparison_bases[i, 1][2],
                                                                                                              comparison_base_scores[i],
                                                                                                              comparison_bases[i, 0][0][1],
                                                                                                              comparison_bases[i, 0][1][1],
                                                                                                              comparison_bases[i, 0][2][1],
                                                                                                              comparison_bases[i, 0][3][1],
                                                                                                              comparison_bases[i, 0][4][1],
                                                                                                              comparison_bases[i, 0][5][1]
                        )]))
        else:
            locations = open(''.join([self.pkg_path, '/data/', self.task, '_', self.model, '_base_configs.log']), 'w')
            # manip_scores = open(''.join([self.pkg_path,'/data/manip_scores.log']), 'w')
            # reach_scores = open(''.join([self.pkg_path,'/data/manip_scores.log']), 'w')
            if self.model == 'chair':
                if len(self.scores[0, 0][0]) == 1:
                    locations.write(''.join(['0', ' %f %f %f %f %f %f %f\n' % (self.scores[0, 0][0][0], self.scores[0, 0][1][0],
                                                                               self.scores[0, 0][2][0], self.scores[0, 0][3][0],
                                                                               self.scores[0, 1][1], self.scores[0, 1][2],
                                                                               best_base_score)]))
                elif len(self.scores[0, 0][0]) == 2:
                    locations.write(''.join(['0', ' %f %f %f %f %f %f %f %f %f %f %f\n' % (self.scores[0, 0][0][0],
                                                                                           self.scores[0, 0][1][0],
                                                                                           self.scores[0, 0][2][0],
                                                                                           self.scores[0, 0][3][0],
                                                                                           self.scores[0, 1][1],
                                                                                           self.scores[0, 1][2],
                                                                                           best_base_score,
                                                                                           self.scores[0, 0][0][1],
                                                                                           self.scores[0, 0][1][1],
                                                                                           self.scores[0, 0][2][1],
                                                                                           self.scores[0, 0][3][1])]))

                comparison_bases = np.array(comparison_bases)
                for i in xrange(len(comparison_base_scores)):
                    if len(comparison_bases[i, 0][0]) == 1:
                        locations.write(''.join([str(i), ' %f %f %f %f %f %f %f \n' % (comparison_bases[i, 0][0][0],
                                                                                       comparison_bases[i, 0][1][0],
                                                                                       comparison_bases[i, 0][2][0],
                                                                                       comparison_bases[i, 0][3][0],
                                                                                       comparison_bases[i, 1][1],
                                                                                       comparison_bases[i, 1][2],
                                                                                       comparison_base_scores[i])]))
                    elif len(comparison_bases[i, 0][0]) == 2:
                        locations.write(''.join([str(i), ' %f %f %f %f %f %f %f %f %f %f %f\n' % (comparison_bases[i, 0][0][0],
                                                                                                  comparison_bases[i, 0][1][0],
                                                                                                  comparison_bases[i, 0][2][0],
                                                                                                  comparison_bases[i, 0][3][0],
                                                                                                  comparison_bases[i, 1][1],
                                                                                                  comparison_bases[i, 1][2],
                                                                                                  comparison_base_scores[i],
                                                                                                  comparison_bases[i, 0][0][1],
                                                                                                  comparison_bases[i, 0][1][1],
                                                                                                  comparison_bases[i, 0][2][1],
                                                                                                  comparison_bases[i, 0][3][1]
                                                                                                  )]))
            elif self.model == 'autobed':
                if len(self.scores[0, 0][0]) == 1:
                    locations.write(''.join(['0', ' %f %f %f %f %f %f %f %f %f\n' % (self.scores[0, 0][0][0], self.scores[0, 0][1][0],
                                                                                     self.scores[0, 0][2][0], self.scores[0, 0][3][0],
                                                                                     self.scores[0, 0][4][0], self.scores[0, 0][5][0],
                                                                                     self.scores[0, 1][1], self.scores[0, 1][2],
                                                                                     best_base_score)]))
                elif len(self.scores[0, 0][0]) == 2:
                    locations.write(''.join(['0', ' %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n' % (self.scores[0, 0][0][0],
                                                                                                       self.scores[0, 0][1][0],
                                                                                                       self.scores[0, 0][2][0],
                                                                                                       self.scores[0, 0][3][0],
                                                                                                       self.scores[0, 0][4][0],
                                                                                                       self.scores[0, 0][5][0],
                                                                                                       self.scores[0, 1][1],
                                                                                                       self.scores[0, 1][2],
                                                                                                       best_base_score,
                                                                                                       self.scores[0, 0][0][1],
                                                                                                       self.scores[0, 0][1][1],
                                                                                                       self.scores[0, 0][2][1],
                                                                                                       self.scores[0, 0][3][1],
                                                                                                       self.scores[0, 0][4][1],
                                                                                                       self.scores[0, 0][5][1])]))

                comparison_bases = np.array(comparison_bases)
                for i in xrange(len(comparison_base_scores)):
                    if len(comparison_bases[i, 0][0]) == 1:
                        locations.write(''.join([str(i), ' %f %f %f %f %f %f %f %f %f \n' % (comparison_bases[i, 0][0][0],
                                                                                             comparison_bases[i, 0][1][0],
                                                                                             comparison_bases[i, 0][2][0],
                                                                                             comparison_bases[i, 0][3][0],
                                                                                             comparison_bases[i, 0][4][0],
                                                                                             comparison_bases[i, 0][5][0],
                                                                                             comparison_bases[i, 1][1],
                                                                                             comparison_bases[i, 1][2],
                                                                                             comparison_base_scores[i])]))
                    elif len(comparison_bases[i, 0][0]) == 2:
                        locations.write(''.join([str(i), ' %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n' % (comparison_bases[i, 0][0][0],
                                                                                                              comparison_bases[i, 0][1][0],
                                                                                                              comparison_bases[i, 0][2][0],
                                                                                                              comparison_bases[i, 0][3][0],
                                                                                                              comparison_bases[i, 0][4][0],
                                                                                                              comparison_bases[i, 0][5][0],
                                                                                                              comparison_bases[i, 1][1],
                                                                                                              comparison_bases[i, 1][2],
                                                                                                              comparison_base_scores[i],
                                                                                                              comparison_bases[i, 0][0][1],
                                                                                                              comparison_bases[i, 0][1][1],
                                                                                                              comparison_bases[i, 0][2][1],
                                                                                                              comparison_bases[i, 0][3][1],
                                                                                                              comparison_bases[i, 0][4][1],
                                                                                                              comparison_bases[i, 0][5][1])]))

        locations.close()
        print 'Saved performance evaluation into a file'
        return # best_base_score, comparison_base_scores

    def generate_scores(self, data_start, data_finish, pos_clust=2, ori_clust=2):
        subject = ''.join(['sub', str(self.train_subj), '_shaver'])

        start_time = time.time()
        print 'Starting to convert data!'
        # runData = DataReader(subject=subject, data_start=data_start, data_finish=data_finish, model=model, task=task,
        #                      pos_clust=pos_clust, ori_clust=ori_clust, tf_listener=self.tf_listener)
        # raw_data = runData.get_raw_data()
        #
        # ## To test clustering by using raw data sampled instead of clusters
        # # sampled_raw = runData.sample_raw_data(raw_data, 100)
        # # runData.generate_output_goals(test_goals=sampled_raw)
        #
        # # To run using the clustering system
        # runData.cluster_data()
        # runData.generate_output_goals()

        runData = DataReader_Task(self.task, self.model)

        # print 'Time to convert data into useful matrices: %fs' % (time.time()-start_time)
        print 'Now starting to generate the score. This will take a long time if there were many goal locations.'
        start_time = time.time()
        # runData.pub_rviz()
        # rospy.spin()
        # runData.plot_goals()
        # runData.generate_score(viz_rviz=True, visualize=False, plot=False)
        runData.generate_score()
        print 'Time to generate all scores: %fs' % (time.time()-start_time)
        print 'Done generating the score sheet for this task and subject'

    def output_scores(self):
        score_save_location = open(''.join([self.pkg_path, '/data/score_output.log']), 'w')
        for score in self.scores:
            if self.model == 'chair':
                if len(score[0][0]) == 1:
                    score_save_location.write(''.join(['%f %f %f %f %f %f\n' % (score[0][0][0], score[0][1][0],
                                                                                score[0][2][0], score[0][3][0],
                                                                                score[1][1], score[1][2])]))
                elif len(score[0][0]) == 2:
                    score_save_location.write(''.join(['%f %f %f %f %f %f %f %f %f %f\n' % (score[0][0][0],
                                                                                            score[0][1][0],
                                                                                            score[0][2][0],
                                                                                            score[0][3][0],
                                                                                            score[1][1],
                                                                                            score[1][2],
                                                                                            score[0][0][1],
                                                                                            score[0][1][1],
                                                                                            score[0][2][1],
                                                                                            score[0][3][1])]))
            if self.model == 'autobed':
                if len(score[0][0]) == 1:
                    score_save_location.write(''.join(['%f %f %f %f %f %f %f %f\n' % (score[0][0][0], score[0][1][0],
                                                                                      score[0][2][0], score[0][3][0],
                                                                                      score[0][4][0], score[0][5][0],
                                                                                      score[1][1], score[1][2])]))
                elif len(score[0][0]) == 2:
                    score_save_location.write(''.join(['%f %f %f %f %f %f %f %f %f %f %f %f %f %f\n' % (score[0][0][0],
                                                                                                        score[0][1][0],
                                                                                                        score[0][2][0],
                                                                                                        score[0][3][0],
                                                                                                        score[0][4][0],
                                                                                                        score[0][5][0],
                                                                                                        score[1][1],
                                                                                                        score[1][2],
                                                                                                        score[0][0][1],
                                                                                                        score[0][1][1],
                                                                                                        score[0][2][1],
                                                                                                        score[0][3][1],
                                                                                                        score[0][4][1],
                                                                                                        score[0][5][1])]))
        score_save_location.close()
        print 'Saved scores into a file'

    def load_task(self, task, model, subj):
        # file_name = ''.join([self.pkg_path, '/data/', task, '_', model, '_subj_', str(subj), '_score_data.pkl'])
        file_name = ''.join([self.pkg_path, '/data/', task, '_', model, '_subj_', str(subj), '_score_data'])
        # return self.load_spickle(file_name)
        print 'loading file with name ', file_name
        try:
            return joblib.load(file_name)
        except IOError:
            print 'Load failed, sorry.'
            return None

    ## read a pickle and return the object.
    # @param filename - name of the pkl
    # @return - object that had been pickled.
    def load_spickle(self, filename):
        try:
            p = open(filename, 'rb')
        except IOError:
            print "hrl_lib.util: Pickle file cannot be opened."
            return None
        try:
            picklelicious = pkl.load(p)
        except ValueError:
            print 'load_spickle failed once, trying again'
            p.close()
            p = open(filename, 'rb')
            picklelicious = pkl.load(p)
        p.close()
        return picklelicious

    def visualize_base_config(self, base, goal_data, reference_options):
        visualize = True
        mytargets = 'all_goals'
        # reference_options = []
        # reference_options.append('head')
        myReferenceNames = reference_options
        myGoals = goal_data
        visualizer = ConfigVisualize(base, goal_data, visualize=visualize, targets=mytargets, reference_names=myReferenceNames,
                                     model=self.model, tf_listener=self.tf_listener)
        rospy.sleep(5)
        selector.visualize_config(best_base, goal_data)


if __name__ == "__main__":
    rospy.init_node('manipulability_shaving_chair')
    train_subj = 0
    test_subj = 0
    visualize_best = True
    myTest = Manipulability_Testing(visualize_best=visualize_best, train_subj=train_subj, test_subj=test_subj)
    best_base = myTest.get_best_base()
    # myTest.run_comparisons()
    rospy.spin()
    # myTest.initialize_test_conditions()
    # myTest.evaluate_task()






