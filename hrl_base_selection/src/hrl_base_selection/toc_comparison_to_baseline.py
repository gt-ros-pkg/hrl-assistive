#!/usr/bin/env python

import numpy as np
import roslib; roslib.load_manifest('hrl_haptic_mpc')
import roslib
roslib.load_manifest('hrl_base_selection')
roslib.load_manifest('hrl_haptic_mpc')
import rospkg

roslib.load_manifest('hrl_base_selection')
import rospy
import time
from data_reader_cma import DataReader as DataReader_cma
from data_reader_comparisons import DataReader as DataReader_comparisons
from data_reader_task import DataReader_Task
# from score_generator import ScoreGenerator
from score_generator_comparisons import ScoreGenerator
from hrl_base_selection.inverse_reachability_setup import InverseReachabilitySetup
# from config_visualize import ConfigVisualize
import scipy.stats

import gc

import random
import os

# import sPickle as pkl
roslib.load_manifest('hrl_lib')
from hrl_lib.util import load_pickle
# import joblib


class Manipulability_Testing(object):
    def __init__(self, visualize=False):
        rospack = rospkg.RosPack()
        self.pkg_path = rospack.get_path('hrl_base_selection')
        self.data_path = '/home/ari/svn/robot1_data/usr/ari/data/base_selection'

        # self.goal_data = None
        # self.best_base = None
        # self.raw_reference_options = None

        self.mc_sim_number = 1000

        self.selector = ScoreGenerator(visualize=visualize)

        output_raw_scores = False

        # compare = True
        # self.visualize_best = visualize_best
        #
        # self.task = 'shaving' # options are: bathing, brushing, feeding, shaving, scratching_upper_arm/forearm/thigh/chest/knee
        # self.model = 'autobed'  # options are: 'chair', 'bed', 'autobed'
        # self.tf_listener = tf.TransformListener()
        #
        # unformat = [[ 0.51690126, -1.05729766, -0.36703181,  0.17778619,  0.06917491,
        #           0.52777768],
        #         [ 0.48857319,  0.7939337 , -2.67601689,  0.25041255,  0.16480721,
        #           0.02473747]]
        # a = np.reshape(unformat[0],[6,1])
        # b = np.reshape(unformat[1],[6,1])
        # base_config = np.hstack([a,b])
        # best_base = [base_config, [0.057329581427009745, 1.0, 0.36352068257210146]]
        # self.scores = []
        # self.scores.append(best_base)

    def load_scores(self, filename='scores_for_comparison.pkl'):
        self.loaded_scores = load_pickle(self.pkg_path+'/data/'+filename)

    def load_goals(self, task, model):
        print 'Reading in raw data from the task.'
        read_task_data = DataReader_Task(task, model)
        raw_data, raw_num, raw_reference, self.raw_reference_options = read_task_data.reset_goals()
        read_data = DataReader_comparisons(subject=subject, data_start=data_start, reference_options=self.raw_reference_options,
                               data_finish=data_finish, model=model, task=task)

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

    def toc_correlation_evaluation(self, task, model, number_samples=5, mc_sim_number=1000,
                                   reset_save_file=False, save_results=False, force_key=None,
                                   seed=None):
        if seed is None:
            seed = int(time.time())
        self.mc_sim_number = mc_sim_number
        save_file_name = 'toc_correlation_results.log'
        save_file_path = self.pkg_path + '/data/'
        if reset_save_file:
            open(save_file_path + save_file_name, 'w').close()
            open(save_file_path + 'raw_' + save_file_name, 'w').close()
        accuracy = np.zeros([number_samples, self.mc_sim_number])
        success = np.zeros([number_samples, self.mc_sim_number])
        self.selector.ireach = InverseReachabilitySetup(visualize=False, redo_ik=False,
                                               redo_reachability=False, redo_ir=False, manip='leftarm')
        for model in models:
            for task in tasks:
                # task = key[0]
                # model = key[3]
                # task = 'wiping_mouth'
                # method = 'toc'
                # sampling = 'cma'
                # model = 'chair'
                # print loaded_scores
                print 'Reading in raw data from the task.'
                read_task_data = DataReader_Task(task, model, 'comparison')
                raw_data, raw_num, raw_reference, raw_reference_options = read_task_data.reset_goals()
                read_data = DataReader_comparisons(reference_options=raw_reference_options,
                                                   model=model, task=task)
                goal_data = read_data.generate_output_goals(test_goals=raw_data, test_number=raw_num,
                                                            test_reference=raw_reference)
                for j in xrange(number_samples):
                    print 'IK sample', j, 'out of ', number_samples
                    self.selector.receive_new_goals(goal_data, reference_options=raw_reference_options, model=model)
                    ik_result_dict = self.selector.handle_score_generation(method='ik', sampling='uniform')
                    for key in ik_result_dict:
                        ik_result = ik_result_dict[key]
                    if task == 'wiping_mouth' and model == 'chair':
                        self.selector.head_angles = np.array([[60., 0.], [0., 0.], [-60., 0.]])
                    else:
                        self.selector.head_angles = np.array([[0., 0.]])
                    self.selector.receive_new_goals(goal_data, reference_options=raw_reference_options,model=model)
                    toc_score =  self.selector.objective_function_one_config_toc_sample(ik_result[0])
                    self.selector.receive_new_goals(goal_data, reference_options=raw_reference_options, model=model)
                    self.selector.ir_and_collision = False
                    ireach_score = self.selector.objective_function_one_config_ireach_sample(ik_result[0])

                    for i in xrange(self.mc_sim_number):

                        accuracy[j, i], success[j, i] = self.evaluate_configuration_mc(model, task, ik_result[0], goal_data,
                                                                                 raw_reference_options, seed=seed)
                        if save_results:
                            with open(save_file_path + 'raw_' + save_file_name, 'a') as myfile:
                                myfile.write(str(task) + ',' + str(model)
                                             + ',' + str("{:.4f}".format(accuracy[j, i]))
                                             + ',' + str("{:.4f}".format(success[j, i]))
                                             + ',' + str("{:.6f}".format(toc_score))
                                             + ',' + str("{:.6f}".format(ireach_score)) + '\n')
                    if save_results:
                        with open(save_file_path + save_file_name, 'a') as myfile:
                            myfile.write(str(task) + ',' + str(model)
                                         + ',' + str("{:.4f}".format(accuracy[j].mean()))
                                         + ',' + str("{:.4f}".format(accuracy[j].std()))
                                         + ',' + str("{:.4f}".format(success[j].mean()))
                                         + ',' + str("{:.4f}".format(success[j].std()))
                                         + ',' + str("{:.6f}".format(toc_score))
                                         + ',' + str("{:.6f}".format(ireach_score))+ '\n')

                    print 'Accuracy was:', accuracy.mean()
                    print 'Success was:', success.mean()
                    print 'TOC score was:', toc_score
                    print 'Reachability score was:', ireach_score
                    gc.collect()
        print 'All done with all comparisons!!'

    def run_comparisons(self, reset_save_file=False, save_results=False, force_key=None,
                        mc_sim_number=1000, seed=None):
        if seed is None:
            seed = int(time.time())
        self.mc_sim_number = mc_sim_number
        save_file_name = 'mc_scores.log'
        save_file_path = self.pkg_path + '/data/'
        if reset_save_file:
            open(save_file_path+save_file_name, 'w').close()
        accuracy = np.zeros(self.mc_sim_number)
        success = np.zeros(self.mc_sim_number)
        for key in self.loaded_scores:
            if force_key is not None:
                new_key = []
                for i in key:
                    new_key.append(i)
                new_key[0]=force_key[0]
                new_key[1]=force_key[1]
                new_key[2] = force_key[2]
                new_key[3] = force_key[3]
                if force_key[3] == 'autobed':
                    new_key[4]=2
                    new_key[5]=-10
                    new_key[6]=0
                    new_key[7]=0
                    new_key[8]=1
                else:
                    new_key[4] = 1
                    new_key[5] = 0
                    new_key[6] = 0
                    new_key[7] = 0
                    new_key[8] = 0
                key = tuple(new_key)

            print 'I will use data with the saved key:', key
            loaded_score = self.loaded_scores[key]
            task = key[0]
            method = key[1]
            sampling = key[2]
            model = key[3]
            # task = 'wiping_mouth'
            # method = 'toc'
            # sampling = 'cma'
            # model = 'chair'
            # print loaded_scores
            print 'Reading in raw data from the task.'
            read_task_data = DataReader_Task(task, model, 'comparison')
            raw_data, raw_num, raw_references, raw_reference_names = read_task_data.reset_goals()
            # print raw_data, raw_num
            # print raw_reference
            # print 'raw_reference_options',raw_reference_options
            # raw_data = read_data.get_raw_data()
            print 'Raw data is ready!'
            if np.size(loaded_score) == 3:
                best_base = loaded_score[0]
                score = loaded_score[1]
                time_to_calc = loaded_score[2]
            elif np.size(loaded_score) == 2:
                best_base = loaded_score[0][0]
                score = loaded_score[0][1]
                time_to_calc = loaded_score[1]
            # print 'best base is: ', best_base
            read_data = DataReader_comparisons(reference_options=raw_reference_names,
                                       model=model, task=task)
            goal_data = read_data.generate_output_goals(test_goals=raw_data, test_number=raw_num, test_reference=raw_references)
            # print 'goal_data', goal_data
            print 'I will now see the percentage of goals reached in', self.mc_sim_number, ' Monte-carlo simulations'

            for i in xrange(self.mc_sim_number):
                # print 'Monte-carlo simulation number', i, 'out of ', self.mc_sim_number
                accuracy[i], success[i] = self.evaluate_configuration_mc(model, task, best_base, goal_data,
                                                                         raw_reference_names, seed=seed)
                seed += 1
            # print accuracy.mean()
            # print success.mean()
            # print str("{:.3f}".format(accuracy.mean()))
            # print str("{:.3f}".format(success.mean()))
            if save_results:
                with open(save_file_path+save_file_name, 'a') as myfile:
                    myfile.write(str(key[0]) + ',' + str(key[1]) + ',' + str(key[2]) + ',' + str(key[3])
                                 + ',' + str(key[4]) + ',' + str(key[5]) + ',' + str(key[6])
                                 + ',' + str(key[7]) + ',' + str(key[8])
                                 + ',' + str("{:.3f}".format(accuracy.mean()))
                                 + ',' + str("{:.3f}".format(success.mean()))
                                 + ',' + str("{:.5f}".format(score))
                                 + ',' + str("{:.5f}".format(time_to_calc))

                                 + '\n')
            print 'Accuracy was:', accuracy.mean()
            print 'Success was:', success.mean()
            gc.collect()
        print 'All done with all comparisons!!'

    def evaluate_configuration_mc(self, model, task, config, goals, reference_names, seed=None):
        if seed is None:
            seed = int(time.time())
        # print 'config', config
        # scorer = ScoreGenerator(goals=goals, model=model, reference_names=reference_options)
        # rospy.sleep(1)
        # self.selector.model = model
        # self.selector.receive_new_goals(goals, reference_options, model=model)
        result = self.selector.mc_eval_init_config(config, goals, reference_names, model=model, task=task, seed=seed)
        # print 'The result of this evaluation is: ', result
        return result

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

    def load_task_scores(self, task, model, type):
        # file_name = ''.join([self.pkg_path, '/data/', task, '_', model, '_subj_', str(subj), '_score_data.pkl'])
        split_task = task.split('_')
        if 'shaving' in split_task:
            file_name = ''.join([self.data_path, '/', task, '/', model, '/', task, '_', model, '_', type, '_score_data'])
        elif 'scratching' in split_task:

            if 'upper' in split_task:
                file_name = ''.join([self.data_path, '/', split_task[0], '/', model, '/', split_task[1], '_', split_task[2], '_', split_task[3], '/', task, '_', model, '_subj_0', '_score_data'])
            elif 'chest' in split_task:
                file_name = ''.join([self.data_path, '/', split_task[0], '/', model, '/', split_task[1], '/', task, '_', model, '_subj_0', '_score_data'])
            else:
                file_name = ''.join([self.data_path, '/', split_task[0], '/', model, '/', split_task[1], '_', split_task[2], '/', task, '_', model, '_subj_0', '_score_data'])
        else:
            file_name = ''.join([self.data_path, '/', task, '/', model, '/', task, '_', model, '_subj_0', '_score_data'])
        # return self.load_spickle(file_name)
        # print 'loading file with name ', file_name
        try:
            if type == 'cma' or 'shaving' in split_task:
                print 'loading file with name ', file_name+'.pkl'
                return load_pickle(file_name+'.pkl')
            else:
                print 'loading file with name ', file_name
                return joblib.load(file_name)
        except IOError:
            print 'Load failed, sorry.'
            return None

    def output_statistics(self, tasks, model, comparison_types):
        print 'Task, model, optimization, mean, std'
        write_stats_name = ''.join([self.data_path, '/statistics_output.csv'])
        write_stats = open(write_stats_name, 'w')
        write_stats.write('Task, model, optimization, mean, std, rank_sum_statistic, rank_sum_p_value\n')
        for task in tasks:
            for comparison_type in comparison_types:
                file_name = ''.join([self.data_path, '/', task, '_', model, '_', comparison_type, '_mc_scores.log'])
                data = np.loadtxt(file_name)
                if comparison_type == 'cma':
                    X = data
                else:
                    Y = data
                print task, ',', model, ',', comparison_type, ',', data.mean(), ',', data.std()
                write_stats.write(''.join([task, ',', model, ',', comparison_type, ',', str(data.mean()), ',', str(data.std()),'\n']))
            ranksum = scipy.stats.ranksums(X,Y)
            print ranksum
            write_stats.write(''.join([',,,,',str(ranksum[0]),',',str(ranksum[1]),'\n']))
        write_stats.close()


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
    comparison_type_options = ['comparison', 'toc_correlation']
    comparison_type = comparison_type_options[0]

    rospy.init_node('manipulability_test_cma'+comparison_type)
    myTest = Manipulability_Testing(visualize=True)

    # visualize_best = True
    # model = 'autobed'  # 'autobed' or 'chair'
    # tasks = ['feeding']#['face_wiping', 'scratching_upper_arm_left', 'scratching_upper_arm_right', 'scratching_forearm_left',
              # 'scratching_forearm_right', 'scratching_thigh_left', 'scratching_thigh_right',
              # 'scratching_chest', 'shaving', 'scratching_knee_left', 'scratching_knee_right', 'bathing']
    # method_options = ['brute', 'cma']

    # best_base = myTest.get_best_base()
    # unformat = [[ 0.51690126, -1.05729766, -0.36703181,  0.17778619,  0.06917491,
    #               0.52777768],
    #             [ 0.48857319,  0.7939337 , -2.67601689,  0.25041255,  0.16480721,
    #               0.02473747]]
    # a = np.reshape(unformat[0],[6,1])
    # b = np.reshape(unformat[1],[6,1])
    # base_config = np.hstack([a,b])
    # best_base = [base_config, [0.057329581427009745, 1.0, 0.36352068257210146]]
    if comparison_type == 'comparison':
        myTest.load_scores()
        seed = 100
        myTest.run_comparisons(reset_save_file=True, save_results=True, mc_sim_number=50, seed=seed)
                               # force_key=['shaving','toc','cma','autobed'])
                          # force_key=['wiping_mouth', 'toc', 'cma', 'chair'])

    elif comparison_type == 'toc_correlation':
        seed = 100
        models = ['autobed']  # 'autobed', 'chair'
            # for task in ['wiping_mouth', 'scratching_knee_left', 'scratching_knee_left', 'scratching_upper_arm_left', 'scratching_upper_arm_right', 'scratching_forearm_left', 'scratching_forearm_right']: #'wiping_face', 'scratching_knee_left', 'scratching_forearm_left','scratching_upper_arm_left']:#'scratching_knee_left', 'scratching_knee_right', 'scratching_thigh_left', 'scratching_thigh_right']:
        tasks = ['wiping_mouth', 'scratching_knee_left', 'scratching_knee_right', 'scratching_thigh_left',
                 'scratching_thigh_right']  # 'wiping_mouth', 'scratching_knee_left','scratching_knee_right','scratching_thigh_left', 'scratching_thigh_right'
        this_start_time = rospy.Time.now()
        myTest.toc_correlation_evaluation(tasks, models, number_samples=100, mc_sim_number=200,
                                          reset_save_file=True, save_results=True, seed=seed)
        print 'Done! Time to generate all scores for this task, method, and sampling:', (rospy.Time.now() - this_start_time).to_sec()
        gc.collect()
        # print 'Done! Time to generate all scores for all tasks: %fs' % (time.time() - full_start_time)


    # myTest.output_statistics(tasks, model, method_options)
    # myTest.load_goals()
    # myTest.run_comparisons(best_base)
    # rospy.spin()
    # myTest.initialize_test_conditions()
    # myTest.evaluate_task()








