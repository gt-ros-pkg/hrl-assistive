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
from data_reader_cma import DataReader
from data_reader_task import DataReader_Task
from score_generator import ScoreGenerator
from pr2_controllers_msgs.msg import SingleJointPositionActionGoal
from config_visualize import ConfigVisualize
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist

import random

import pickle
# import sPickle as pkl
roslib.load_manifest('hrl_lib')
from hrl_lib.util import load_pickle
import joblib


class Manipulability_Testing(object):
    def __init__(self, visualize_best=False, train_subj=6, test_subj=6):
        output_raw_scores = False

        # compare = True
        self.visualize_best = visualize_best

        self.task = 'shaving' # options are: bathing, brushing, feeding, shaving, scratching_upper_arm/forearm/thigh/chest/knee
        self.model = 'autobed'  # options are: 'chair', 'bed', 'autobed'
        self.tf_listener = tf.TransformListener()

        unformat = [[ 0.51690126, -1.05729766, -0.36703181,  0.17778619,  0.06917491,
                  0.52777768],
                [ 0.48857319,  0.7939337 , -2.67601689,  0.25041255,  0.16480721,
                  0.02473747]]
        a = np.reshape(unformat[0],[6,1])
        b = np.reshape(unformat[1],[6,1])
        base_config = np.hstack([a,b])
        best_base = [base_config, [0.057329581427009745, 1.0, 0.36352068257210146]]
        self.scores = []
        self.scores.append(best_base)

    def load_scores(self):

        self.train_subj = train_subj
        self.test_subj = test_subj
        print 'I will use data that was trained on subject ', self.train_subj
        print 'I will test on data from subject ', self.test_subj



        pos_clust = 2
        ori_clust = 2

        self.visualize = False
        data_start = 0
        data_finish = 'end '  # 2000  # 4000 #'end'

        rospack = rospkg.RosPack()
        self.pkg_path = rospack.get_path('hrl_base_selection')
        self.data_path = '/home/ari/svn/robot1_data/usr/ari/data/base_selection'
        print 'Loading scores.'
        self.loaded_scores = self.load_task(self.task, self.model, self.train_subj)
        if self.loaded_scores is None:
            print 'The scores do not exist. Must generate scores! This may take a long time...'
            # self.generate_scores(data_start, data_finish, pos_clust, ori_clust)
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

    def load_goals(self):
        # subject = ''.join(['sub', str(self.test_subj), '_shaver'])
        subject=None
        print 'Reading in raw data from the task.'
        read_task_data = DataReader_Task(self.task, self.model)
        data_start=0
        data_finish='end'
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
        visualize_best = True
        if visualize_best:
            self.visualize_base_config(self.best_base, self.goal_data, self.raw_reference_options)

    def get_best_base(self):
        return self.best_base

    def run_comparisons(self, best_base):
        print 'The number of base configurations on the score sheet for the default human position with ' \
              'score > 0 is: ', len(self.scores)
        comparison_bases = []
        max_num_of_comparisons = 0 # 400
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
        do_first = True
        if anova_test:
            single_score = []
            for aScore in self.scores:
                if len(aScore[0][0]) == 1:
                    single_score.append(aScore)
            single_score = np.array(sorted(single_score, key=lambda t: (t[1][1], t[1][2]), reverse=True))
            best_single = single_score[0]
            print 'The best single configuration is: ', best_single
            # print best_single[1][1]
            # print best_base[1][1]
            # print best_single[1][1]*.98
            # print best_base[1][1]*.98
            single_score_selection = []
            multi_score_selection = []
            for aScore in self.scores:
                if (len(aScore[0][0]) == 1) and (aScore[1][1] >= best_single[1][1]*.98):
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
            for j in xrange(1, 2):
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
        self.evaluate_base_locations(best_base, comparison_bases, self.goal_data, self.raw_reference_options)

    def evaluate_base_locations(self, best_base, comparison_bases, goal_data, reference_options):
        visualize = True
        mytargets = 'all_goals'
        # reference_options = []
        # reference_options.append('head')
        myReferenceNames = reference_options
        myGoals = goal_data
        # best_base[0][0] = [.95]
        # best_base[0][1] = [-0.1]
        # best_base[0][2] = [m.pi/2]
        # best_base[0][3] = [0.3]
        # best_base[0][4] = [0.]
        # best_base[0][5] = [0.]
        self.selector = ScoreGenerator(visualize=visualize, targets=mytargets, reference_names=myReferenceNames,
                                  goals=myGoals, model=self.model, tf_listener=self.tf_listener)
        rospy.sleep(5)
        start_time = time.time()
        # self.selector.receive_new_goals(goal_data)
        # self.selector.show_rviz()
        best_base_score = self.selector.eval_init_config(best_base, goal_data)

        print 'The score for the best base was: ', best_base_score
        comparison_base_scores = []
        for item in comparison_bases:
            self.selector.receive_new_goals(goal_data)
            comparison_base_scores.append(self.selector.eval_init_config(item, goal_data))
        print 'The best base location is: \n', best_base
        print 'The score for the best base was: ', best_base_score
        # for i in xrange(len(comparison_base_scores)):
            # print 'A comparison base score for base: \n', comparison_bases[i]
            # print 'The score was: ', comparison_base_scores[i]
        print 'Time to generate all scores for comparison and best bases: %fs' % (time.time()-start_time)
        locations = open(''.join([self.data_path, '/', self.task, '_', self.model, '_base_configs.log']), 'w')
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
        return best_base_score, comparison_base_scores

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
        score_save_location = open(''.join([self.data_path, '/score_output.log']), 'w')
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
        if 'scratching' in task.split('_'):
            split_task = task.split('_')
            file_name = ''.join([self.data_path, '/', split_task[0], '/', model, '/', split_task[1], '_', split_task[2], '/', task, '_', model, '_subj_', str(subj), '_score_data'])
        else:
            file_name = ''.join([self.data_path, '/', task, '/', model, '/', task, '_', model, '_subj_', str(subj), '_score_data'])
        # return self.load_spickle(file_name)
        print 'loading file with name ', file_name
        try:
            return joblib.load(file_name)
        except IOError:
            print 'Load failed, sorry.'
            return None

    '''
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
    '''

    def visualize_base_config(self, base, goal_data, reference_options):
        print base
        base = [[[0.46289771], [-0.72847723], [0.1949910376461335], [0.29444783399999996],
                 [0.0030885365000000026], [0.06484810062059854]],
                [0.057329581427009745, 1.0, 0.36352068257210146]]
        base = [[[0.623102892598], [-0.762012001042], [-0.0294729879325], [0.0902860705641],
                 [0.00648402460469], [1.23414184526]],
                [0.057329581427009745, 1.0, 0.36352068257210146]]
        base = [[[0.67848577, -0.81211008], [0.70119037, -0.01048174], [3.63956937, -0.14257091], [0.13847594, 0.04083888],
                 [0.01491837, 0.10748585], [0.18010711, 0.139652]],
                [0.057329581427009745, 1.0, 0.36352068257210146]]
        unformat = [[ 0.4580933 ,  1.00806015, -2.66855508,  0.17084372,  0.13866491,
         0.83056839],
        [ 0.82132058, -0.82340737,  0.68550188,  0.19356394,  0.09474543,
         0.15645363]]
        unformat = [[ 0.71440993, -0.7706963 ,  0.17108766,  0.2724865 ,  0.17209647,
                      0.31666851],
                    [ 1.13256279, -0.95259745,  0.23538584,  0.2464303 ,  0.10706364,
                      1.00092141]]
        unformat = [[ 0.51690126, -1.05729766, -0.36703181,  0.17778619,  0.06917491,
                      0.52777768],
                    [ 0.48857319,  0.7939337 , -2.67601689,  0.25041255,  0.16480721,
                      0.02473747]]

        a = np.reshape(unformat[0],[6,1])
        b = np.reshape(unformat[1],[6,1])
        base_config = np.hstack([a,b])
        base_config = [[ 0.96138881,  0.83774071],
       [ 0.63033125, -1.07430128],
       [-2.0599323 , -0.17919976],
       [ 0.0974726 ,  0.23857654],
       # [ 0.09994156,  0.1857015 ],
       # [ 0.69738434,  0.83414354]]
       [ 0.09994156,  0.09994156 ],
       [ 0.69738434,  0.69738434]]
        base_config = [[ 1.23151836,  0.84031986],
       [ 0.78498528, -0.73199084],
       [-2.81981806, -4.4666789 ],
       [ 0.29643016,  0.03537878],
       [ 0.04969906,  0.0708807 ],
       [ 1.38750928,  0.29672911]]
        base = [base_config, [0.057329581427009745, 1.0, 0.36352068257210146]]

        visualize = True
        mytargets = 'all_goals'
        reference_options = []
        reference_options.append('head')
        myReferenceNames = reference_options
        myGoals = goal_data
        visualizer = ConfigVisualize(base, goal_data, visualize=visualize, targets=mytargets, reference_names=myReferenceNames,
                                     model=self.model, tf_listener=self.tf_listener)
        rospy.sleep(5)
        self.selector.visualize_config(best_base, goal_data)


if __name__ == "__main__":
    rospy.init_node('manipulability_test_shaving_anova')
    train_subj = 0
    test_subj = 0
    visualize_best = True
    myTest = Manipulability_Testing(visualize_best=visualize_best, train_subj=train_subj, test_subj=test_subj)
    # best_base = myTest.get_best_base()
    unformat = [[ 0.51690126, -1.05729766, -0.36703181,  0.17778619,  0.06917491,
                  0.52777768],
                [ 0.48857319,  0.7939337 , -2.67601689,  0.25041255,  0.16480721,
                  0.02473747]]
    a = np.reshape(unformat[0],[6,1])
    b = np.reshape(unformat[1],[6,1])
    base_config = np.hstack([a,b])
    best_base = [base_config, [0.057329581427009745, 1.0, 0.36352068257210146]]
    myTest.load_goals()
    # myTest.run_comparisons(best_base)
    rospy.spin()
    # myTest.initialize_test_conditions()
    # myTest.evaluate_task()




