#!/usr/bin/env python

import numpy as np
import math as m
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

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams
from matplotlib.cbook import flatten
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

import re

from matplotlib.patches import Rectangle, Ellipse

import gc

import random, copy
import os

# import sPickle as pkl
roslib.load_manifest('hrl_lib')
from hrl_lib.util import load_pickle, save_pickle
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

        self.selector = ScoreGenerator(visualize=visualize, training=False)

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
                    if (task == 'wiping_mouth' or task == 'shaving' or task == 'feeding_trajectory' or task == 'brushing')  and model == 'chair':
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

    def run_comparisons_monty_carlo(self, reset_save_file=False, save_results=False, force_key=None,
                        mc_sim_number=1000, seed=None, use_error=True):
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
                if force_key[3] == 'autobed' and force_key[1] == 'toc':
                    new_key[4]=2
                    new_key[5]=-10
                    new_key[6]=0
                    new_key[7]=0
                    new_key[8]=1
                elif force_key[3] == 'autobed':
                    new_key[4] = 1
                    new_key[5] = -10
                    new_key[6] = 0
                    new_key[7] = 0
                    new_key[8] = 1
                elif force_key[3] == 'chair' and force_key[1] == 'toc':
                    new_key[4] = 2
                    new_key[5] = 0
                    new_key[6] = 0
                    new_key[7] = 0
                    new_key[8] = 0
                elif force_key[3] == 'chair':
                    new_key[4] = 1
                    new_key[5] = 0
                    new_key[6] = 0
                    new_key[7] = 0
                    new_key[8] = 0
                key = tuple(new_key)
            current_seed = copy.copy(seed)
            print 'I will use data with the saved key:', key
            loaded_score = self.loaded_scores[key]
            print loaded_score
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
            # best_base = np.array([[ 0.64335277,  0.79439213],
            #                       [ 0.78823877, -0.8840706 ],
            #                       [-1.38234847, -4.67764725],
            #                       [ 0.21458089,  0.24799169],
            #                       [ 0.64335277,  0.79439213],
            #                       [ 0.78823877, -0.8840706 ]])
            # x= 1.0
            # y = -0.65
            # th = m.radians(90.)
            # z = 0.3
            # use_error = False
            # best_base = np.reshape([x, y, th, z, 0.2, m.radians(45.)],[6,1])

            read_data = DataReader_comparisons(reference_options=raw_reference_names,
                                       model=model, task=task)
            goal_data = read_data.generate_output_goals(test_goals=raw_data, test_number=raw_num, test_reference=raw_references)
            # print 'goal_data', goal_data
            print 'I will now see the percentage of goals reached in', self.mc_sim_number, ' Monte-carlo simulations'

            for i in xrange(self.mc_sim_number):
                # print current_seed
                # print 'Monte-carlo simulation number', i, 'out of ', self.mc_sim_number
                if use_error == False:
                    accuracy[i], success[i] = self.evaluate_configuration_mc(model, task, best_base, goal_data,
                                                                             raw_reference_names, seed=current_seed,
                                                                             error=np.zeros(6))
                else:
                    accuracy[i], success[i] = self.evaluate_configuration_mc(model, task, best_base, goal_data,
                                                                             raw_reference_names, seed=current_seed)
                current_seed += 1
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
                                 + ',' + str("{:.3f}".format(accuracy.std()))
                                 + ',' + str("{:.3f}".format(success.mean()))
                                 + ',' + str("{:.3f}".format(success.std()))
                                 + ',' + str("{:.5f}".format(score))
                                 + ',' + str("{:.5f}".format(time_to_calc))
                                 + '\n')
            print 'Accuracy was:', accuracy.mean()
            print 'Success was:', success.mean()
            gc.collect()
        print 'All done with all comparisons!!'

    def comparisons_monty_carlo_plotting(self):
        print 'Starting to plot the comparison between TOC and baseline algorithms'
        save_file_name = 'mc_scores.log'
        save_file_path = self.pkg_path + '/data/'
        raw_loaded_data = [line.rstrip('\n').split(',') for line in open(save_file_path + save_file_name)]
        # print raw_loaded_data
        loaded_data = dict()
        # ik_results = dict()
        # capability_map_results = dict()
        # capability_map_collision_results = dict()
        # toc1_results = dict()
        # toc2_results = dict()
        for item in raw_loaded_data:
            # data = [float(i) for i in loaded_data[9:]]
            if item[0] not in loaded_data.keys():
                loaded_data[item[0]] = dict()
            if item[3] not in loaded_data[item[0]].keys():
                loaded_data[item[0]][item[3]] = dict()
            if item[1] == 'toc':
                loaded_data[item[0]][item[3]]['toc'+str(item[4])] = [float(i) for i in item[9:]]
            elif int(item[4]) == 1:
                loaded_data[item[0]][item[3]][item[1]] = [float(i) for i in item[9:]]

        N = len(loaded_data.keys())
        fig_num = 0
        for model in ['autobed','chair']:
            ik_success_means = []
            ik_success_std = []
            capability_success_means = []
            capability_success_std = []
            capability_collision_success_means = []
            capability_collision_success_std = []
            toc1_success_means = []
            toc1_success_std = []
            toc2_success_means = []
            toc2_success_std = []
            tick_labels = []

            # Create human-readable versions of the model names
            if model == 'autobed':
                modelname = 'Autobed'
            elif model == 'chair':
                modelname = 'Wheelchair'
            else:
                print 'Not sure what the model is'

            N = 0
            task_list = ['scratching_knee_left','scratching_knee_right',
                         'scratching_thigh_left', 'scratching_thigh_right',
                         'scratching_upper_arm_left', 'scratching_upper_arm_right',
                         'bathe_legs', 'feeding_trajectory',
                         'wiping_mouth', 'shaving', 'arm_cuffs']
            for task in task_list:
                if (model == 'chair' and task not in ['brushing','scratching_thigh_right',
                                                      'scratching_thigh_left', 'bathe_legs']) \
                        or (model == 'autobed' and task not in ['brushing']):
                    N += 1
                    ik_success_means.append(loaded_data[task][model]['ik'][2])
                    ik_success_std.append(loaded_data[task][model]['ik'][3])

                    capability_success_means.append(loaded_data[task][model]['inverse_reachability'][2])
                    capability_success_std.append(loaded_data[task][model]['inverse_reachability'][3])

                    capability_collision_success_means.append(loaded_data[task][model]['inverse_reachability_collision'][2])
                    capability_collision_success_std.append(loaded_data[task][model]['inverse_reachability_collision'][3])

                    toc1_success_means.append(loaded_data[task][model]['toc1'][2])
                    toc1_success_std.append(loaded_data[task][model]['toc1'][3])

                    toc2_success_means.append(loaded_data[task][model]['toc2'][2])
                    toc2_success_std.append(loaded_data[task][model]['toc2'][3])

                    # Create human-readable versions of the task names
                    taskname = task

                    # Create the tick labels
                    tick_labels.append(taskname)

            # print N
            ind = np.arange(N)  # the x locations for the groups
            width = 0.15  # the width of the bars
            # print ik_success_means
            # print len(capability_success_means)
            # print len(capability_collision_success_means)
            # print len(toc1_success_means)
            # print len(toc2_success_means)
            # print len(ik_success_std)
            # print len(capability_success_std)
            # print len(capability_collision_success_std)
            # print len(toc1_success_std)
            # print len(toc2_success_std)
            fig = plt.figure(fig_num, figsize=(24, 14))
            ax = plt.subplot(111)
            fig_num += 1
            co = plt.get_cmap('Accent')
            rects1 = ax.bar(ind, ik_success_means, width, color=co(0.4), yerr=ik_success_std)
            rects2 = ax.bar(ind+width, capability_success_means, width, color=co(0.3), yerr=capability_success_std)
            rects3 = ax.bar(ind+2*width, capability_collision_success_means, width, color=co(0.2), yerr=capability_collision_success_std)
            rects4 = ax.bar(ind+3*width, toc1_success_means, width, color=co(0.1), yerr=toc1_success_std)
            rects5 = ax.bar(ind+4*width, toc2_success_means, width, color=co(0.0), yerr=toc2_success_std)

            # add some text for labels, title and axes ticks
            ax.set_ylabel('% of Successful Trials')
            ax.set_title('Success vs Error for '+modelname, y=1.15)
            ax.set_xticks(ind + width* 2.5)
            ax.set_xticklabels(tick_labels, rotation=65)
            # plt.xlim([0, N])
            # rcParams.update({'figure.autolayout': True})
            # plt.tight_layout(pad=0.2)
            plt.subplots_adjust(bottom=0.3,top=0.8)

            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                             ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(20)

            ax.set_yticks([0, 0.25, 0.50, 0.75, 1.0])
            ax.set_ylim(0., 1.1)
            plt.autoscale(enable=True, axis='x', tight=True)

            ax.grid(True, axis='y', linestyle='dotted')

            ax.legend([rects1[0], rects2[0], rects3[0], rects4[0], rects5[0]],
                      ['IK Solver', 'Capability Map', 'Capability Map w/ Collision',
                       'TOC: 1 config', 'TOC: 2 configs'], fontsize=20,
                      bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                      ncol=5, mode="expand", borderaxespad=0.)

            # def autolabel(rects):
            #     """
            #     Attach a text label above each bar displaying its height
            #     """
            #     for rect in rects:
            #         height = rect.get_height()
                    # ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                    #         '%d' % int(height),
                    #         ha='center', va='bottom')

            # autolabel(rects1)
            # autolabel(rects2)
            # autolabel(rects3)
            # autolabel(rects4)
            # autolabel(rects5)
            print 'Saving figure!'
            plt.savefig(save_file_path + 'comparison_vs_baseline_' + model + '.png', bbox_inches="tight")

        plt.show()

        plt.figure(fig_num)
        rospy.spin()

    def base_brute_evaluation(self, task, model,
                              discretization_size_xy,
                              discretization_size_theta,
                              discretization_size_z,
                              reset_save_file=True, save_results=True):
        save_file_path = self.pkg_path + '/data/'
        save_file_name_score = 'base_brute_evaluation_' + task + '_' + model + '_toc_score.log'
        save_file_name_accuracy = 'base_brute_evaluation_' + task + '_' + model + '_accuracy.log'

        print 'Reading in raw data from the task.'
        read_task_data = DataReader_Task(task, model, 'comparison')
        raw_data, raw_num, raw_references, raw_reference_names = read_task_data.reset_goals()
        print 'Raw data is ready!'

        read_data = DataReader_comparisons(reference_options=raw_reference_names,
                                           model=model, task=task)
        goal_data = read_data.generate_output_goals(test_goals=raw_data, test_number=raw_num,
                                                    test_reference=raw_references)

        x_range = np.arange(0.4, 3.+ discretization_size_xy / 5., discretization_size_xy)
        y_range = np.arange(-1., 1. + discretization_size_xy / 5., discretization_size_xy)
        theta_range = np.arange(0., 2.*m.pi, discretization_size_theta)
        z_range = np.arange(0., 0.3 + discretization_size_z / 5., discretization_size_z)


        if reset_save_file:
            open(save_file_path + save_file_name_score, 'w').close()
            open(save_file_path + 'raw_' + save_file_name_score, 'w').close()
            open(save_file_path + save_file_name_accuracy, 'w').close()
            open(save_file_path + 'raw_' + save_file_name_accuracy, 'w').close()

        for nx, x in enumerate(x_range):
            print 'Starting on X position', x, 'out of', np.max(x_range)
            for ny, y in enumerate(y_range):
                best_toc_score = [0., [[x, y, 0., 0., 0.2, m.radians(45.)]]]  # [x, y, 0., 0., 0.2, m.radians(45.)]
                best_accuracy = [0., [[x, y, 0., 0., 0.2, m.radians(45.)]]]  # [x, y, 0., 0., 0.2, m.radians(45.)]
                for nth, th in enumerate(theta_range):
                    best_toc_score_at_height = [0., [x, y, th, 0., 0.2, m.radians(45.)]]
                    best_accuracy_at_height = [0., [x, y, th, 0., 0.2, m.radians(45.)]]
                    for nz, z in enumerate(z_range):
                        config = [x, y, th, z, 0.2, m.radians(45.)]
                        toc_score, accuracy = self.selector.eval_one_config_toc(config, goal_data,
                                                                                raw_reference_names,
                                                                                model=model,task=task)
                        if best_toc_score_at_height[0] < toc_score:
                            best_toc_score_at_height[0] = toc_score
                            best_toc_score_at_height[1] = config
                        if best_accuracy_at_height[0] < accuracy:
                            best_accuracy_at_height[0] = accuracy
                            best_accuracy_at_height[1] = config
                    if best_toc_score[0] < best_toc_score_at_height[0]:
                        best_toc_score[0] = best_toc_score_at_height[0]
                        best_toc_score[1] = [best_toc_score_at_height[1]]
                    elif best_toc_score[0] == best_toc_score_at_height[0] and best_toc_score_at_height[0] > 0.01:
                        best_toc_score[1].append(best_toc_score_at_height[1])
                    if best_accuracy[0] < best_accuracy_at_height[0]:
                        best_accuracy[0] = best_accuracy_at_height[0]
                        best_accuracy[1] = [best_accuracy_at_height[1]]
                    elif np.abs(best_accuracy[0] - best_accuracy_at_height[0])< 0.05 and best_accuracy_at_height[0] > 0.01:
                        best_accuracy[1].append(best_accuracy_at_height[1])
                if save_results:
                    with open(save_file_path + save_file_name_score, 'a') as myfile:
                        for i in xrange(len(best_toc_score[1])):
                            myfile.write(str("{:.4f}".format(best_toc_score[0]))
                                         + ',' + str("{:.4f}".format(best_toc_score[1][i][0]))
                                         + ',' + str("{:.4f}".format(best_toc_score[1][i][1]))
                                         + ',' + str("{:.4f}".format(best_toc_score[1][i][2]))
                                         + ',' + str("{:.4f}".format(best_toc_score[1][i][3]))
                                         + ',' + str("{:.4f}".format(best_toc_score[1][i][4]))
                                         + ',' + str("{:.4f}".format(best_toc_score[1][i][5]))
                                         + '\n')

                    with open(save_file_path + save_file_name_accuracy, 'a') as myfile:
                        myfile.write(str("{:.4f}".format(best_accuracy[0]))
                                     + ',' + str("{:.4f}".format(best_accuracy[1][i][0]))
                                     + ',' + str("{:.4f}".format(best_accuracy[1][i][1]))
                                     + ',' + str("{:.4f}".format(best_accuracy[1][i][2]))
                                     + ',' + str("{:.4f}".format(best_accuracy[1][i][3]))
                                     + ',' + str("{:.4f}".format(best_accuracy[1][i][4]))
                                     + ',' + str("{:.4f}".format(best_accuracy[1][i][5]))
                                     + '\n')
            gc.collect()

    def plot_base_brute_evaluation(self, task, model, discretization_size_xy):
        save_file_path = self.pkg_path + '/data/'
        fig = plt.figure(0, figsize=(24, 10))
        if task == 'shaving':
            task_name = 'Shaving'
        elif task == 'arm_cuffs':
            task_name = 'Arm Hygiene'
        elif task == 'wiping_mouth':
            task_name = 'Wiping Mouth'
        if model == 'autobed':
            model_name = 'Autobed'
        elif model == 'chair':
            model_name = 'Wheelchair'
        # plt.suptitle('Scores for Robot Configurations for ' + task_name + ' Task in ' + model_name + ' Environment', fontsize='26')
        subplot_num = 120
        for score_type in ['accuracy', 'toc_score']:
            subplot_num += 1
            save_file_name = 'base_brute_evaluation_' + task + '_' + model + '_' + score_type + '.log'

            loaded_data = [line.rstrip('\n').split(',') for line in open(save_file_path + save_file_name)]
            # print loaded_data
            for j in xrange(len(loaded_data)):
                loaded_data[j] = [float(i) for i in loaded_data[j]]
            loaded_data = np.array(loaded_data)

            score_list = loaded_data[:, 0]
            x = loaded_data[:, 1]
            y = loaded_data[:, 2]
            th = loaded_data[:, 3]

            colors_acc = ['red', 'orange', 'yellow', 'green']
            labels_acc = ['<0.05', '0.05 - 0.5', '0.5 - 0.99', '1.0']
            result_cluster = [[-1., 0.05], [0.05, 0.5], [0.5, 0.99], [0.99, 10.]]

            ax = plt.subplot(subplot_num, aspect='equal')
            # ax.set_title('Task Accuracy: Shaving Task')
            ax.set_xlim(0.-discretization_size_xy/2., 2.4 + discretization_size_xy/2.)
            ax.set_ylim(-1.2-discretization_size_xy/2., 1.2+discretization_size_xy/2.)
            # ax.set_xlim(0.6 - discretization_size_xy, 1.2 + discretization_size_xy)
            # ax.set_ylim(-1. - discretization_size_xy, -0.5 + discretization_size_xy)
            ax.set_xlabel('X-Axis Direction (m)')
            ax.set_ylabel('Y-Axis Direction (m)')
            if score_type == 'accuracy':
                descriptor = 'IK'
            else:
                descriptor = 'TOC'
            ax.set_title('Scoring using '+descriptor)
            # ax.grid(True, linestyle='dotted')
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                             ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(20)

            # if score_type == 'accuracy':
            # co = plt.get_cmap('gist_rainbow')
            cmap1 = plt.get_cmap('gist_rainbow')
            colors1 = cmap1(np.linspace(0, 0.4, 20))
            cmap1 = LinearSegmentedColormap.from_list('my_cmap1', colors1)
            cmap2 = plt.get_cmap('gist_rainbow')
            colors2 = cmap2(np.linspace(0.4, 0.7, 20))
            cmap2 = LinearSegmentedColormap.from_list('my_cmap2', colors2)
            # else:
            #     co1 = plt.get_cmap('gist_rainbow')
                # co2 = plt.get_cmap('gist_rainbow')

            # for result in result_cluster:
            score_x = []
            score_y = []
            score_value = []
            # for i in xrange(len(score_list)):
            #     if score_list[i] >= result[0] and score_list[i] < result[1]:
            #         score_x.append(x[i])
            #         score_y.append(y[i])
            #         score_value.append

            for ix, iy, ith, isc in zip(x, y, th, score_list):
                if score_type == 'accuracy':
                    # color = co(0.4*isc/np.max(score_list))
                    color = cmap1(isc)
                else:
                    if isc <= 10.:
                        # color = co(0.4 * isc / 10.)
                        color = cmap1(isc/ 10.)
                    else:
                        color = cmap2( (isc- np.min(filter(lambda t: t > 10., score_list))) /
                                       (np.max(score_list)-np.min(filter(lambda t: t > 10., score_list))))
                ax.add_artist(Rectangle(xy=(ix - discretization_size_xy/2., iy - discretization_size_xy/2.),
                                               color=color,
                                               width=discretization_size_xy, height=discretization_size_xy))  # Gives a square of area h*h
            for ix, iy, ith, isc in zip(x, y, th, score_list):
                if isc > 0.:
                    ax.arrow(ix, iy, m.cos(ith)*discretization_size_xy/2., m.sin(ith)*discretization_size_xy/2.,
                             width=0.002, head_width=discretization_size_xy/4., length_includes_head=True,
                             head_length=discretization_size_xy/4., overhang=0.0)

            if model == 'autobed':
                ax.add_artist(Rectangle(xy=[-0.04, -0.4515], width=2.201, height=0.903,
                                        facecolor='gray', edgecolor='black', linewidth=3, alpha = 0.5))
                ax.add_artist(Ellipse(xy=[0.54302, 0.], width=0.20, height=0.20, angle=0.,
                                            fill=False, linewidth=3))
                plt.text(0.66, -0.03, 'Head Location', fontsize=16)
                plt.text(1.7, -0.43, 'Bed Frame', fontsize=16)
                # ax.add_artist(ell)
            # cbar = fig.colorbar(ax)

            colors_acc = ['red', 'orange', 'yellow', 'green']
            labels_acc = ['<5%', '5-49%', '50-99%', '100%']
            # ax.scatter(10, 10, marker='s', s=200., edgecolors='none', label=labels_acc[0], c=colors_acc[0])
            # ax.scatter(10, 10, marker='s', s=200., edgecolors='none', label=labels_acc[1], c=colors_acc[1])
            # ax.scatter(10, 10, marker='s', s=200., edgecolors='none', label=labels_acc[2], c=colors_acc[2])
            # ax.scatter(10, 10, marker='s', s=200., edgecolors='none', label=labels_acc[3], c=colors_acc[3])
            # ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
            #           ncol=2, mode="expand", borderaxespad=0., fontsize=20,
            #           title='Score', scatterpoints=1)
            # ax.get_legend().get_title().set_fontsize(20)

        # ax = plt.subplot(subplot_num, aspect='equal')
        # cax = plt.axes([0.85, 0.1, 0.075, 0.8])
        ax1 = fig.add_axes([0.47, 0.10, 0.02, 0.4])
        ax2 = fig.add_axes([0.47, 0.5, 0.02, 0.4])
        # ax1 = plt.subplot(133, )
        # ax2 = fig.add_axes([0.05, 0.475, 0.9, 0.15])
        # ax3 = fig.add_axes([0.05, 0.15, 0.9, 0.15])

        cmap1 = plt.get_cmap('gist_rainbow')
        colors1 = cmap1(np.linspace(0, 0.4, 20))
        cmap1 = LinearSegmentedColormap.from_list('my_cmap1', colors1)
        cmap2 = plt.get_cmap('gist_rainbow')
        colors2 = cmap2(np.linspace(0.4, 0.7, 20))
        cmap2 = LinearSegmentedColormap.from_list('my_cmap2', colors2)

        # Set the colormap and norm to correspond to the data for which
        # the colorbar will be used.
        # cmap = mpl.cm.gist_rainbow
        norm = mpl.colors.Normalize(vmin=1, vmax=1.4, clip=True)
        # norm = mpl.colors.Normalize(clip=False)

        # ColorbarBase derives from ScalarMappable and puts a colorbar
        # in a specified axes, so it has everything needed for a
        # standalone colorbar.  There are many more kwargs, but the
        # following gives a basic continuous colorbar with ticks
        # and labels.
        cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap1, drawedges=False,
                                        # norm=norm,
                                        orientation='vertical')
        cb1.set_ticks([0., 1.0])
        cb1.set_ticklabels(['0.0', '1.0'])


        cb2 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap2, drawedges=False,
                                        norm=norm,
                                        orientation='vertical')
        cb2.set_ticks([1.4])
        cb2.set_ticklabels(['1.4'])

        # for item in [cb1.get_yticklabels(), cb2.get_yticklabels()]:
        #     item.set_fontsize(20)
        cb1.ax.tick_params(labelsize=20)
        cb2.ax.tick_params(labelsize=20)
        plt.tight_layout()
        print 'Saving figure!'
        plt.savefig(save_file_path + 'base_brute_evaluation_fig_' + task + '_' + model + '.png', bbox_inches="tight", )
        print 'Done with all plots!'
        plt.show()
        fig2 = plt.figure(2)

    def robustness_calculation(self, task, model, method, discretization_size, search_area, reset_save_file=False, save_results=False):
        save_file_path = self.pkg_path + '/data/'

        key = []

        key.append(task)
        key.append(method)
        key.append('cma')
        key.append(model)
        if method == 'toc':
            key.append(2)
        else:
            key.append(1)
        if model == 'autobed':
            key.append(-10)
            key.append(0.0)
            key.append(0.0)
            key.append(1)
        else:
            key.append(0)
            key.append(0)
            key.append(0)
            key.append(0)
        key = tuple(key)
        print 'I will use data with the saved key:\n', key
        loaded_score = self.loaded_scores[key]
        print 'score:\n', loaded_score

        print 'Reading in raw data from the task.'
        read_task_data = DataReader_Task(task, model, 'comparison')
        raw_data, raw_num, raw_references, raw_reference_names = read_task_data.reset_goals()
        # print loaded_score
        print 'Raw data is ready!'
        if np.size(loaded_score) == 3:
            best_bases = loaded_score[0]
            score = loaded_score[1]
            time_to_calc = loaded_score[2]
        elif np.size(loaded_score) == 2:
            best_bases = loaded_score[0][0]
            score = loaded_score[0][1]
            time_to_calc = loaded_score[1]
        # print best_bases
        if len(np.shape(best_bases)) == 1:
            best_bases = np.reshape(best_bases,[len(best_bases),1])

        read_data = DataReader_comparisons(reference_options=raw_reference_names,
                                           model=model, task=task)
        goal_data = read_data.generate_output_goals(test_goals=raw_data, test_number=raw_num,
                                                    test_reference=raw_references)
        # print 'goal_data', goal_data
        # print 'I will now see the percentage of goals reached in', self.mc_sim_number, ' Monte-carlo simulations'
        eval_range = search_area
        num_calcs = eval_range/discretization_size+1
        if len(best_bases[0]) == 2:
            num_plots = 3
        else:
            num_plots = 1
        # subplot_num = 130
        # fig_num = 0
        # fig = plt.figure(fig_num, figsize=(24, 14))
        for base in xrange(num_plots):
            save_file_name = 'robustness_visualization_results_' + task + '_' + model + '_' + str(base) + '.log'
            if reset_save_file:
                open(save_file_path + save_file_name, 'w').close()
                open(save_file_path + 'raw_' + save_file_name, 'w').close()
            if base == 0 or base == 1:
                best_base = np.reshape(best_bases[:, base],[6,1])
            else:
                best_base = best_bases
            # x = np.zeros(num_calcs*num_calcs)
            # y = np.zeros(num_calcs*num_calcs)
            # success_list = np.zeros(num_calcs*num_calcs)
            # accuracy_list = np.zeros(num_calcs*num_calcs)
            accuracy = np.zeros([num_calcs, num_calcs])
            success = np.zeros([num_calcs, num_calcs])
            # location = np.zeros([num_calcs, num_calcs, 2])
            # subplot_num +=1

            for ny, j in enumerate(np.arange(-eval_range/2.,eval_range/2.+discretization_size/5., discretization_size)):
                for nx, i in enumerate(np.arange(-eval_range/2.,eval_range/2.+discretization_size/5., discretization_size)):
                    # location[nx,ny] = [i, j]
                    error = [i, j, 0, 0, 0, 0]
                    accuracy[nx,ny], success[nx,ny] = self.evaluate_configuration_mc(model, task, best_base, goal_data,
                                                                                 raw_reference_names, error=error)
                    # x[ny*num_calcs+nx] = i
                    # y[ny * num_calcs + nx] = j

                    # success_list[ny * num_calcs + nx] = int(success[nx,ny])
                    # accuracy_list[ny * num_calcs + nx] = accuracy[nx,ny]
                    # print 'nx, ny, success', nx, ny, success[nx, ny]
                    if save_results:
                        with open(save_file_path + save_file_name, 'a') as myfile:
                            myfile.write(str("{:.4f}".format(i))
                                         + ',' + str("{:.4f}".format(j))
                                         + ',' + str("{:.4f}".format(accuracy[nx, ny]))
                                         + ',' + str("{:.4f}".format(success[nx, ny]))
                                         + '\n')
            print 'Saved file:', save_file_name
        print 'Done with all calculation of robustness'

    def robustness_plotting(self, task, model, method, discretization_size, search_area):
        print 'Starting visualization of robustness'
        subplot_num = 130
        base_file_name = 'robustness_visualization_results_' + task + '_' + model + '_'
        save_file_path = self.pkg_path + '/data/'

        plot_ids = []

        file_list = os.listdir(save_file_path)
        for item in file_list:
            if base_file_name in item:
                item = filter(None, re.split('[_.]', item))
                plot_ids.append(int(item[-2]))
        num_plots = np.max(plot_ids)+1

        print 'Preparing plot visualization. Will make', num_plots+1, 'total plots'
        fig_num = 0
        fig = plt.figure(fig_num, figsize=(24, 10))
        colors_acc = ['red', 'orange', 'yellow', 'green']
        labels_acc = ['<0.05', '0.05 - 0.5', '0.5 - 0.99', '1.0']
        if task == 'shaving':
            task_name = 'Shaving'
        elif task == 'arm_cuffs':
            task_name = 'Arm Hygiene'
        elif task == 'wiping_mouth':
            task_name = 'Wiping Mouth'
        if model == 'autobed':
            model_name = 'Autobed'
        elif model == 'chair':
            model_name = 'Wheelchair'
        plt.suptitle('Accuracy vs Error for '+task_name + ' Task in '+model_name, fontsize='26')

        for base in xrange(num_plots):
            print 'Starting plot', base+1
            save_file_name = 'robustness_visualization_results_' + task + '_' + model + '_' + str(base) + '.log'

            loaded_data = [line.rstrip('\n').split(',') for line in open(save_file_path+save_file_name)]
            # print loaded_data
            for j in xrange(len(loaded_data)):
                loaded_data[j] = [float(i) for i in loaded_data[j]]
            loaded_data = np.array(loaded_data)
            # print loaded_data
            x = loaded_data[:, 0]
            y = loaded_data[:, 1]
            accuracy_list = loaded_data[:, 2]
            success_list = loaded_data[:, 3]

            subplot_num += 1
            if num_plots == 1:
                subplot_num = 111

            ax = plt.subplot(subplot_num, aspect='equal')
            # ax.set_title('Task Accuracy: Shaving Task')
            ax.set_xlim(-search_area / 2.0 - discretization_size-0.5, search_area / 2.0 + discretization_size+0.5)
            ax.set_ylim(-search_area / 2.0 - discretization_size-0.5, search_area / 2.0 + discretization_size+0.5)
            ax.set_xlim(-1.3, 1.3)
            ax.set_ylim(-1.3, 1.3)
            ax.set_xlabel('X-Axis Human Pose Error (m)')
            ax.set_ylabel('Y-Axis Human Pose Error (m)')
            ax.grid(True, linestyle='dotted')
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                             ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(20)
            # print 'Preparing plot visualization'
            result_cluster = [[-1., 0.05], [0.05, 0.5], [0.5, 0.99], [0.99, 10.]]
            pr2_x = []
            pr2_y = []
            # print 'best_base:\n',best_base

            ax.set_xlim(-search_area / 2.0 - discretization_size,
                        search_area / 2.0 + discretization_size)
            ax.set_ylim(-search_area / 2.0 - discretization_size,
                        search_area / 2.0 + discretization_size)
            co = plt.get_cmap('gist_rainbow')
            for result in result_cluster:
                acc_x = []
                acc_y = []
                for i in xrange(len(accuracy_list)):
                    if accuracy_list[i] >= result[0] and accuracy_list[i] < result[1]:
                        acc_x.append(x[i])
                        acc_y.append(y[i])

                for ix, iy in zip(acc_x, acc_y):
                    plt.gca().add_artist(Rectangle(xy=(ix - 0.005, iy - 0.005),
                                                   color=colors_acc[result_cluster.index(result)],
                                                   label=labels_acc[result_cluster.index(result)],
                                                   width=0.01, height=0.01))  # Gives a square of area h*h

                    # ax.legend()
            ell = Ellipse(xy=[0., 0.], width=0.2, height=0.2, angle=0., fill=False, linewidth=3)
            # ax.add_artist(ell)

            colors_acc = ['red', 'orange', 'yellow', 'green']
            labels_acc = ['<5%', '5-49%', '50-99%', '100%']
            ax.scatter(10, 10, marker='s', s=200., edgecolors='none', label=labels_acc[0], c=colors_acc[0])
            ax.scatter(10, 10, marker='s', s=200., edgecolors='none', label=labels_acc[1], c=colors_acc[1])
            ax.scatter(10, 10, marker='s', s=200., edgecolors='none', label=labels_acc[2], c=colors_acc[2])
            ax.scatter(10, 10, marker='s', s=200., edgecolors='none', label=labels_acc[3], c=colors_acc[3])
            # ax.add_artist(Rectangle(xy=[10, 10], width=0.05, height=0.05,
            #                         label=labels_acc[0],
            #                         color=colors_acc[0]))
            # ax.add_artist(Rectangle(xy=[10, 10], width=0.05, height=0.05,
            #                         label=labels_acc[1],
            #                         color=colors_acc[1]))
            # ax.add_artist(Rectangle(xy=[10, 10], width=0.05, height=0.05,
            #                         label=labels_acc[2],
            #                         color=colors_acc[2]))
            # ax.add_artist(Rectangle(xy=[10, 10], width=0.05, height=0.05,
            #                         label=labels_acc[3],
            #                         color=colors_acc[3]))
            # ax.scatter(10, 10, marker='s', label=labels_acc[1], c=colors_acc[1])
            # ax.scatter(10, 10, marker='s', label=labels_acc[2], c=colors_acc[2])
            # ax.scatter(10, 10, marker='s', label=labels_acc[3], c=colors_acc[3])
            # ax.legend(bbox_to_anchor=(0.9, 1), loc=2, borderaxespad=0., fontsize=20)

            if base == 0 and num_plots == 1:
                conf = '1 config'
            elif base == 0 and num_plots == 3:
                conf = '1st config'
            elif base == 1:
                conf = '2nd config'
            elif base == 2:
                conf = 'both configs'
            else:
                print 'ERROR: base is not what I expect based on num_plots!'
            ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                      ncol=2, mode="expand", borderaxespad=0., fontsize=20,
                      title='% Goals Reached: '+conf, scatterpoints = 1)
            # ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,loc='upper center'
            #           ncol=2, mode="expand", borderaxespad=0.)
            ax.get_legend().get_title().set_fontsize(20)
            # ax.legend()
            # ax.set_zorder(20)
            print 'Finished plot', base+1
        # plt.figlegend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
        #               ncol=2, mode="expand", borderaxespad=0., fontsize=20,
        #               title='% Goals Reached: ' + conf, scatterpoints=1)
        # red_patch = patches.Patch(color='red', label=labels_acc[0])
        # orange_patch = patches.Patch(color='orange', label=labels_acc[1])
        # yellow_patch = patches.Patch(color='yellow', label=labels_acc[2])
        # green_patch = patches.Patch(color='green', label=labels_acc[3])
        # ax.add_patch(red_patch)
        # ax.add_patch(orange_patch)
        # ax.add_patch(yellow_patch)
        # ax.add_patch(green_patch)

        # ax.legend(handles=[red_patch, orange_patch, yellow_patch, green_patch], fontsize=20,
        #           bbox_to_anchor=(0.9, 1), loc=2, borderaxespad=0.)
        print 'Saving figure!'
        plt.savefig(save_file_path+'robustness_fig_'+task+'_' + model + '.png',bbox_inches="tight",)
        print 'Done with all plots!'
        plt.show()
        fig2 = plt.figure(2)

    def evaluate_configuration_mc(self, model, task, config, goals, reference_names, seed=None, error=None):
        if seed is None:
            seed = int(time.time())
        # print 'config', config
        # scorer = ScoreGenerator(goals=goals, model=model, reference_names=reference_options)
        # rospy.sleep(1)
        # self.selector.model = model
        # self.selector.receive_new_goals(goals, reference_options, model=model)
        result = self.selector.mc_eval_init_config(config, goals, reference_names, model=model, task=task,
                                                   seed=seed, error=error)
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
    comparison_type_options = ['comparison', 'toc_correlation',
                               'robustness_visualization', 'plot_comparison',
                               'base_brute_evaluation','plot_base_brute_evaluation']
    comparison_type = comparison_type_options[4]

    rospy.init_node('manipulability_test_cma'+comparison_type)
    myTest = Manipulability_Testing(visualize=False)

    if comparison_type == 'comparison':
        myTest.load_scores()
        seed = 100
        # myTest.run_comparisons_monty_carlo(reset_save_file=True, save_results=True, mc_sim_number=200, seed=seed)
        myTest.run_comparisons_monty_carlo(reset_save_file=False, save_results=False, mc_sim_number=100, seed=seed,
                                           force_key=['wiping_mouth', 'toc', 'cma', 'autobed'], use_error=False)

    if comparison_type == 'plot_comparison':
        myTest.comparisons_monty_carlo_plotting()
        rospy.spin()

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

    elif comparison_type == 'robustness_visualization':
        seed = 1000
        search_area = 0.7
        discretization_size = 0.01  # centimeters
        model = 'chair'
        task = 'arm_cuffs'  # 'shaving', 'wiping_mouth', 'arm_cuffs'
        method = 'toc' #inverse_reachability_collision
        this_start_time = rospy.Time.now()
        myTest.load_scores()
        # myTest.robustness_calculation(task, model, method, discretization_size, search_area,
        #                               reset_save_file=True, save_results=True)
        myTest.robustness_plotting(task, model, method, discretization_size, search_area)
        print 'Done! Time to generate all scores for this task, method, and sampling:', (
        rospy.Time.now() - this_start_time).to_sec()

        # gc.collect()
        rospy.spin()

    elif comparison_type == 'base_brute_evaluation':
        discretization_size_xy = 0.05  # centimeters
        discretization_size_theta = m.radians(45.)
        discretization_size_z = 0.15
        model = 'autobed'
        task = 'wiping_mouth'  # 'shaving', 'wiping_mouth', 'arm_cuffs'
        method = 'toc'  # inverse_reachability_collision
        this_start_time = rospy.Time.now()
        myTest.base_brute_evaluation(task, model,
                              discretization_size_xy,
                              discretization_size_theta,
                              discretization_size_z,
                              reset_save_file=False, save_results=False)
        myTest.plot_base_brute_evaluation(task, model, discretization_size_xy)
        print 'Done! Time to generate all scores for this task, method, and sampling:', (
            rospy.Time.now() - this_start_time).to_sec()
        rospy.spin()

    elif comparison_type == 'plot_base_brute_evaluation':
        discretization_size_xy = 0.05  # centimeters
        model = 'autobed'
        task = 'wiping_mouth'  # 'shaving', 'wiping_mouth', 'arm_cuffs'
        myTest.plot_base_brute_evaluation(task, model, discretization_size_xy)
        rospy.spin()






