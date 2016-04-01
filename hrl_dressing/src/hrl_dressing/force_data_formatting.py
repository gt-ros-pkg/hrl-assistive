#!/usr/bin/python

import roslib
import rospy, rospkg, rosparam
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Bool
import numpy as np
import os.path
import copy

roslib.load_manifest('hrl_dressing')
roslib.load_manifest('zenither')
import zenither.zenither as zenither

roslib.load_manifest('hrl_lib')
from hrl_lib.util import save_pickle, load_pickle
from hrl_msgs.msg import FloatArray
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.cbook import flatten
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches


class ForceDataFormatting(object):
    def __init__(self):
        # self.test = True
        # rospack = rospkg.RosPack()
        self.data_path = '/home/ari/svn/robot1_data/usr/ari/data/hrl_dressing'

    def autolabel_and_set_position(self, subject, result):
        print 'Now inserting position data and labeling results automatically according to where they stopped.'
        paramlist = rosparam.load_file(''.join([self.data_path, '/', subject, '/params.yaml']))
        for params, ns in paramlist:
            rosparam.upload_params(ns, params)
        arm_length = rosparam.get_param('crook_to_fist')
        fist_length = arm_length - rosparam.get_param('crook_to_wrist')
        print 'Fist length: ', fist_length, ' cm'
        position_of_initial_contact = (44.0 - arm_length)/100.
        position_profile = load_pickle(''.join([self.data_path, '/position_profiles/position_combined_0_1mps.pkl']))
        vel = 0.1
        ft_threshold_was_exceeded = False
        reference_data = np.array([map(float,line.strip().split()) for line in open(''.join([self.data_path, '/', subject, '/', str(vel), 'mps/height3/ft_sleeve_0.log']))])
        del_index = []
        for k in xrange(len(reference_data)):
            if reference_data[k][0] < 5.0:
                del_index.append(k)
        reference_data = np.delete(reference_data, del_index, 0)

        del_index = []
        for num, j in enumerate(reference_data):
            j[2] = -j[2]
            j[3] = -j[3]
            j[4] = -j[4]
            j[0] = j[0] - 2.0
            # if j[0] < 0.5:
            #     j[1] = 0
            if np.max(np.abs(j[2:5])) > 10. and not ft_threshold_was_exceeded:
                time_of_stop = j[0]
                ft_threshold_was_exceeded = True
                for k in xrange(len(position_profile)-1):
                    if position_profile[k, 0] <= j[0] < position_profile[k+1, 0]:
                        success_position_of_stop = position_profile[k, 1] + \
                                           (position_profile[k+1, 1] - position_profile[k, 1]) * \
                                           (j[0]-position_profile[k, 0])/(position_profile[k+1, 0] - position_profile[k, 0])
            elif ft_threshold_was_exceeded:
                del_index.append(num)
                # j[1] = position_of_stop
            else:
                for k in xrange(len(position_profile)-1):
                    if position_profile[k, 0] <= j[0] < position_profile[k+1, 0]:
                        new_position = position_profile[k, 1] + \
                                       (position_profile[k+1, 1] - position_profile[k, 1]) * \
                                       (j[0]-position_profile[k, 0])/(position_profile[k+1, 0] - position_profile[k, 0])
                j[1] = new_position
        print 'The stop position when the sleeve reaches the elbow is: ', success_position_of_stop

        position_profile = None
        for ind_i in xrange(len(result)):
            for ind_j in xrange(len(result[0])):
                if result[ind_i][ind_j] is not None:
                    if ind_i < len(result)/2:
                        load_num = ind_i
                        vel = 0.1
                    else:
                        load_num = ind_i - len(result)/2
                        vel = 0.15

                    if vel == 0.1:
                        # print ''.join([self.data_path, '/position_profiles/position_combined_0_1mps.pkl'])
                        position_profile = load_pickle(''.join([self.data_path, '/position_profiles/position_combined_0_1mps.pkl']))
                        # print 'Position profile loaded!'
                    elif vel == 0.15:
                        position_profile = load_pickle(''.join([self.data_path, '/position_profiles/position_combined_0_15mps.pkl']))
                        # print ''.join([self.data_path, '/position_profiles/position_combined_0_15mps.pkl'])
                        # print 'Position profile loaded!'
                    else:
                        print 'There is no saved position profile for this velocity! Something has gone wrong!'
                        return None



                    ft_threshold_was_exceeded = False
                    current_data = np.array([map(float,line.strip().split()) for line in open(''.join([self.data_path, '/', subject, '/', str(vel), 'mps/height', str(ind_j), '/ft_sleeve_', str(load_num), '.log']))])
                    del_index = []
                    for k in xrange(len(current_data)):
                        if current_data[k][0] < 5.0:
                            del_index.append(k)
                    current_data = np.delete(current_data, del_index, 0)
                    position_of_stop = 0.
                    del_index = []
                    time_of_initial_contact = None
                    for num, j in enumerate(current_data):
                        j[2] = -j[2]
                        j[3] = -j[3]
                        j[4] = -j[4]
                        j[0] = j[0] - 2.0
                        # if j[0] < 0.5:
                        #     j[1] = 0
                        if np.max(np.abs(j[2:5])) > 10. and not ft_threshold_was_exceeded:
                            time_of_stop = j[0]
                            ft_threshold_was_exceeded = True
                            for k in xrange(len(position_profile)-1):
                                if position_profile[k, 0] <= j[0] < position_profile[k+1, 0]:
                                    position_of_stop = position_profile[k, 1] + \
                                                       (position_profile[k+1, 1] - position_profile[k, 1]) * \
                                                       (j[0]-position_profile[k, 0])/(position_profile[k+1, 0] - position_profile[k, 0])
                            j[1] = new_position
                        elif ft_threshold_was_exceeded:
                            del_index.append(num)
                            # j[1] = position_of_stop
                        else:
                            for k in xrange(len(position_profile)-1):
                                if position_profile[k, 0] <= j[0] < position_profile[k+1, 0]:
                                    new_position = position_profile[k, 1] + \
                                                   (position_profile[k+1, 1] - position_profile[k, 1]) * \
                                                   (j[0]-position_profile[k, 0])/(position_profile[k+1, 0] - position_profile[k, 0])
                            j[1] = new_position
                            if abs(new_position - position_of_stop) < 0.0005 and new_position > 0.8:
                                del_index.append(num)
                            elif new_position < position_of_initial_contact:
                                del_index.append(num)
                            else:
                                if time_of_initial_contact is None:
                                    time_of_initial_contact = j[0]
                            position_of_stop = new_position
                            # if num > 8 and 0.05 < new_position < 0.4 and position_of_initial_contact is None:
                            #     prev_f_x_average = np.mean(current_data[num-7:num,2])
                            #     curr_f_x_average = np.mean(current_data[num-6:num+1,2])
                            #     prev_f_z_average = np.mean(current_data[num-7:num,4])
                            #     curr_f_z_average = np.mean(current_data[num-6:num+1,4])
                            #     if curr_f_x_average < -0.08 and curr_f_x_average < prev_f_x_average and curr_f_z_average < prev_f_z_average and curr_f_z_average < -0.03:
                            #         position_of_initial_contact = copy.copy(new_position)

                    # if position_of_initial_contact is None:
                    #     print 'Position of initial contact was not detected for ', subject, ' and trial ', ind_j, 'at height ', ind_i
                    #     position_of_initial_contact = 0.
                    #     # return
                    # else:
                    #     print 'Position of initial contact is: ', position_of_initial_contact
                    # current_data = np.delete(current_data, del_index, 0)
                    # del_index = []
                    # for i in xrange(len(current_data)):
                    #     if current_data[i, 1] < position_of_initial_contact:
                    #         del_index.append(i)
                    current_data = np.delete(current_data, del_index, 0)
                    output_data = []
                    for item in current_data:
                        output_data.append([item[0]-time_of_initial_contact, item[1]-position_of_initial_contact, item[2], item[3], item[4]])
                    output_data = np.array(output_data)
                    print 'This trial failed at: ', position_of_stop
                    this_label = None
                    if not ft_threshold_was_exceeded:
                        this_label = 'missed'
                    elif abs(position_of_stop - success_position_of_stop) < 0.015 or position_of_stop >= success_position_of_stop:
                        this_label = 'good'
                    else:
                        if abs(position_of_stop - success_position_of_stop) + fist_length/100. + 0.05 >= arm_length/100.:
                            this_label = 'caught_fist'
                        else:
                            this_label = 'caught_other'

                    save_number = 0
                    save_file = ''.join([self.data_path, '/', subject, '/auto_labeled/', str(vel), 'mps/', this_label, '/force_profile_', str(save_number), '.pkl'])
                    while os.path.isfile(save_file):
                        save_number += 1
                        save_file = ''.join([self.data_path, '/', subject, '/auto_labeled/', str(vel), 'mps/', this_label, '/force_profile_', str(save_number), '.pkl'])
                    print 'Saving with file name', save_file
                    save_pickle(output_data, save_file)
        print 'Done editing files!'

    def format_data_four_categories(self, subject, result):
        # print subject
        # print result
        print 'Now editing files to insert position data and storing them in labeled folders.'
        paramlist = rosparam.load_file(''.join([self.data_path, '/', subject, '/params.yaml']))
        for params, ns in paramlist:
            rosparam.upload_params(ns, params)
        arm_length = rosparam.get_param('crook_to_fist')
        fist_length = arm_length - rosparam.get_param('crook_to_wrist')
        print 'Fist length: ', fist_length, ' cm'
        position_of_initial_contact = (44.0 - arm_length)/100.
        position_profile = None
        for ind_i in xrange(len(result)):
            for ind_j in xrange(len(result[0])):
                if ind_j == 2:
                    continue
                else:
                    if result[ind_i][ind_j] is not None:
                        if ind_i < len(result)/2:
                            load_num = ind_i
                            vel = 0.1
                        else:
                            load_num = ind_i - len(result)/2
                            vel = 0.15

                        if vel == 0.1:
                            # print ''.join([self.data_path, '/position_profiles/position_combined_0_1mps.pkl'])
                            position_profile = load_pickle(''.join([self.data_path, '/position_profiles/position_combined_0_1mps.pkl']))
                            # print 'Position profile loaded!'
                        elif vel == 0.15:
                            position_profile = load_pickle(''.join([self.data_path, '/position_profiles/position_combined_0_15mps.pkl']))
                            # print ''.join([self.data_path, '/position_profiles/position_combined_0_15mps.pkl'])
                            # print 'Position profile loaded!'
                        else:
                            print 'There is no saved position profile for this velocity! Something has gone wrong!'
                            return None
                        ft_threshold_was_exceeded = False
                        # current_data = np.array([map(float,line.strip().split()) for line in open(''.join([self.data_path, '/', subject, '/', input_classes[class_num], '/ft_sleeve_', str(load_num), '.log']))])

                    # while os.path.isfile(''.join([self.pkg_path, '/data/', subject, '/',str(vel),'mps/', input_classes[class_num], '/ft_sleeve_', str(i), '.pkl'])):
                        ft_threshold_was_exceeded = False
                        # print ''.join([self.pkg_path, '/data/', subject, '/', input_classes[class_num], '/ft_sleeve_', str(i), '.pkl'])
                        current_data = np.array([map(float,line.strip().split()) for line in open(''.join([self.data_path, '/', subject, '/', str(vel), 'mps/height', str(ind_j), '/ft_sleeve_', str(load_num), '.log']))])
                        del_index = []
                        for k in xrange(len(current_data)):
                            if current_data[k][0] < 5.0:
                                del_index.append(k)
                        current_data = np.delete(current_data, del_index, 0)
                        position_of_stop = 0.
                        del_index = []
                        time_of_initial_contact = None

                        # current_data = load_pickle(''.join([self.pkg_path, '/data/', subject, '/', input_classes[class_num], '/ft_sleeve_', str(i), '.pkl']))

                        # if np.max(current_data[:, 2]) >= 10. or np.max(current_data[:, 3]) >= 10. \
                        #         or np.max(current_data[:, 4]) >= 10.:
                        #     ft_threshold_was_exceeded = True
                        position_of_stop = 0.
                        del_index = []
                        for num, j in enumerate(current_data):
                            j[2] = -j[2]
                            j[3] = -j[3]
                            j[4] = -j[4]
                            j[0] = j[0] - 2.0
                            # if j[0] < 0.5:
                            #     j[1] = 0
                            if np.max(np.abs(j[2:5])) > 10. and not ft_threshold_was_exceeded:
                                time_of_stop = j[0]
                                ft_threshold_was_exceeded = True
                                for k in xrange(len(position_profile)-1):
                                    if position_profile[k, 0] <= j[0] < position_profile[k+1, 0]:
                                        position_of_stop = position_profile[k, 1] + \
                                                           (position_profile[k+1, 1] - position_profile[k, 1]) * \
                                                           (j[0]-position_profile[k, 0])/(position_profile[k+1, 0] - position_profile[k, 0])
                                j[1] = new_position
                            elif ft_threshold_was_exceeded:
                                del_index.append(num)
                                # j[1] = position_of_stop
                            else:
                                for k in xrange(len(position_profile)-1):
                                    if position_profile[k, 0] <= j[0] < position_profile[k+1, 0]:
                                        new_position = position_profile[k, 1] + \
                                                       (position_profile[k+1, 1] - position_profile[k, 1]) * \
                                                       (j[0]-position_profile[k, 0])/(position_profile[k+1, 0] - position_profile[k, 0])
                                j[1] = new_position
                                if abs(new_position - position_of_stop) < 0.0005 and new_position > 0.8:
                                    del_index.append(num)
                                elif new_position < position_of_initial_contact:
                                    del_index.append(num)
                                else:
                                    if time_of_initial_contact is None:
                                        time_of_initial_contact = j[0]
                                position_of_stop = new_position
                        # if result[ind_i][ind_j] == 'good'

                        current_data = np.delete(current_data, del_index, 0)
                        output_data = []
                        for item in current_data:
                            output_data.append([item[0]-time_of_initial_contact, item[1]-position_of_initial_contact, item[2], item[3], item[4]])
                        output_data = np.array(output_data)
                        save_number = 0
                        save_file = ''.join([self.data_path, '/', subject, '/formatted/', str(vel), 'mps/', result[ind_i][ind_j], '/force_profile_', str(save_number), '.pkl'])
                        while os.path.isfile(save_file):
                            save_number += 1
                            save_file = ''.join([self.data_path, '/', subject, '/formatted/', str(vel), 'mps/', result[ind_i][ind_j], '/force_profile_', str(save_number), '.pkl'])
                        print 'Saving with file name', save_file
                        save_pickle(output_data, save_file)
        print 'Done editing files!'
        # if self.plot:
        #     self.plot_data(current_data)

    def format_data_three_categories(self, subject, result):
        # print subject
        # print result
        print 'Now editing files to insert position data and storing them in labeled folders.'
        paramlist = rosparam.load_file(''.join([self.data_path, '/', subject, '/params.yaml']))
        for params, ns in paramlist:
            rosparam.upload_params(ns, params)
        arm_length = rosparam.get_param('crook_to_fist')
        fist_length = arm_length - rosparam.get_param('crook_to_wrist')
        print 'Fist length: ', fist_length, ' cm'
        position_of_initial_contact = (44.0 - arm_length)/100.
        position_profile = None
        for ind_i in xrange(len(result)):
            for ind_j in xrange(len(result[0])):
                if ind_j == 2:
                    continue
                else:
                    if result[ind_i][ind_j] is not None:
                        if ind_i < len(result)/2:
                            load_num = ind_i
                            vel = 0.1
                        else:
                            load_num = ind_i - len(result)/2
                            vel = 0.15

                        if vel == 0.1:
                            # print ''.join([self.data_path, '/position_profiles/position_combined_0_1mps.pkl'])
                            position_profile = load_pickle(''.join([self.data_path, '/position_profiles/position_combined_0_1mps.pkl']))
                            # print 'Position profile loaded!'
                        elif vel == 0.15:
                            position_profile = load_pickle(''.join([self.data_path, '/position_profiles/position_combined_0_15mps.pkl']))
                            # print ''.join([self.data_path, '/position_profiles/position_combined_0_15mps.pkl'])
                            # print 'Position profile loaded!'
                        else:
                            print 'There is no saved position profile for this velocity! Something has gone wrong!'
                            return None
                        ft_threshold_was_exceeded = False
                        # current_data = np.array([map(float,line.strip().split()) for line in open(''.join([self.data_path, '/', subject, '/', input_classes[class_num], '/ft_sleeve_', str(load_num), '.log']))])

                    # while os.path.isfile(''.join([self.pkg_path, '/data/', subject, '/',str(vel),'mps/', input_classes[class_num], '/ft_sleeve_', str(i), '.pkl'])):
                        ft_threshold_was_exceeded = False
                        # print ''.join([self.pkg_path, '/data/', subject, '/', input_classes[class_num], '/ft_sleeve_', str(i), '.pkl'])
                        current_data = np.array([map(float,line.strip().split()) for line in open(''.join([self.data_path, '/', subject, '/', str(vel), 'mps/height', str(ind_j), '/ft_sleeve_', str(load_num), '.log']))])
                        del_index = []
                        for k in xrange(len(current_data)):
                            if current_data[k][0] < 5.0:
                                del_index.append(k)
                        current_data = np.delete(current_data, del_index, 0)
                        position_of_stop = 0.
                        del_index = []
                        time_of_initial_contact = None

                        # current_data = load_pickle(''.join([self.pkg_path, '/data/', subject, '/', input_classes[class_num], '/ft_sleeve_', str(i), '.pkl']))

                        # if np.max(current_data[:, 2]) >= 10. or np.max(current_data[:, 3]) >= 10. \
                        #         or np.max(current_data[:, 4]) >= 10.:
                        #     ft_threshold_was_exceeded = True
                        position_of_stop = 0.
                        del_index = []
                        for num, j in enumerate(current_data):
                            j[2] = -j[2]
                            j[3] = -j[3]
                            j[4] = -j[4]
                            j[0] = j[0] - 2.0
                            # if j[0] < 0.5:
                            #     j[1] = 0
                            if np.max(np.abs(j[2:5])) > 10. and not ft_threshold_was_exceeded:
                                time_of_stop = j[0]
                                ft_threshold_was_exceeded = True
                                for k in xrange(len(position_profile)-1):
                                    if position_profile[k, 0] <= j[0] < position_profile[k+1, 0]:
                                        position_of_stop = position_profile[k, 1] + \
                                                           (position_profile[k+1, 1] - position_profile[k, 1]) * \
                                                           (j[0]-position_profile[k, 0])/(position_profile[k+1, 0] - position_profile[k, 0])
                                j[1] = new_position
                            elif ft_threshold_was_exceeded:
                                del_index.append(num)
                                # j[1] = position_of_stop
                            else:
                                for k in xrange(len(position_profile)-1):
                                    if position_profile[k, 0] <= j[0] < position_profile[k+1, 0]:
                                        new_position = position_profile[k, 1] + \
                                                       (position_profile[k+1, 1] - position_profile[k, 1]) * \
                                                       (j[0]-position_profile[k, 0])/(position_profile[k+1, 0] - position_profile[k, 0])
                                j[1] = new_position
                                if abs(new_position - position_of_stop) < 0.0005 and new_position > 0.8:
                                    del_index.append(num)
                                elif new_position < position_of_initial_contact:
                                    del_index.append(num)
                                else:
                                    if time_of_initial_contact is None:
                                        time_of_initial_contact = j[0]
                                position_of_stop = new_position
                        # if result[ind_i][ind_j] == 'good'

                        current_data = np.delete(current_data, del_index, 0)
                        output_data = []
                        for item in current_data:
                            output_data.append([item[0]-time_of_initial_contact, item[1]-position_of_initial_contact, item[2], item[3], item[4]])
                        output_data = np.array(output_data)
                        save_number = 0
                        if result[ind_i][ind_j] == 'caught_fist' or result[ind_i][ind_j] == 'caught_other':
                            save_file = ''.join([self.data_path, '/', subject, '/formatted_three/', str(vel), 'mps/', 'caught', '/force_profile_', str(save_number), '.pkl'])
                        else:
                            save_file = ''.join([self.data_path, '/', subject, '/formatted_three/', str(vel), 'mps/', result[ind_i][ind_j], '/force_profile_', str(save_number), '.pkl'])
                        while os.path.isfile(save_file):
                            save_number += 1
                            if result[ind_i][ind_j] == 'caught_fist' or result[ind_i][ind_j] == 'caught_other':
                                save_file = ''.join([self.data_path, '/', subject, '/formatted_three/', str(vel), 'mps/', 'caught', '/force_profile_', str(save_number), '.pkl'])
                            else:
                                save_file = ''.join([self.data_path, '/', subject, '/formatted_three/', str(vel), 'mps/', result[ind_i][ind_j], '/force_profile_', str(save_number), '.pkl'])
                        print 'Saving with file name', save_file
                        save_pickle(output_data, save_file)
        print 'Done editing files!'
        # if self.plot:
        #     self.plot_data(current_data)

    def format_data_only_miss_and_good_heights(self, subject, result):
        # print subject
        # print result
        print 'Now editing files to insert position data and storing them in labeled folders.'
        paramlist = rosparam.load_file(''.join([self.data_path, '/', subject, '/params.yaml']))
        for params, ns in paramlist:
            rosparam.upload_params(ns, params)
        arm_length = rosparam.get_param('crook_to_fist')
        fist_length = arm_length - rosparam.get_param('crook_to_wrist')
        print 'Fist length: ', fist_length, ' cm'
        position_of_initial_contact = (44.0 - arm_length)/100.
        position_profile = None
        for ind_i in xrange(len(result)):
            for ind_j in xrange(len(result[0])):
                if ind_j == 2 or ind_j == 1:
                    continue
                else:
                    if result[ind_i][ind_j] is not None:
                        if ind_i < len(result)/2:
                            load_num = ind_i
                            vel = 0.1
                        else:
                            load_num = ind_i - len(result)/2
                            vel = 0.15

                        if vel == 0.1:
                            # print ''.join([self.data_path, '/position_profiles/position_combined_0_1mps.pkl'])
                            position_profile = load_pickle(''.join([self.data_path, '/position_profiles/position_combined_0_1mps.pkl']))
                            # print 'Position profile loaded!'
                        elif vel == 0.15:
                            position_profile = load_pickle(''.join([self.data_path, '/position_profiles/position_combined_0_15mps.pkl']))
                            # print ''.join([self.data_path, '/position_profiles/position_combined_0_15mps.pkl'])
                            # print 'Position profile loaded!'
                        else:
                            print 'There is no saved position profile for this velocity! Something has gone wrong!'
                            return None
                        ft_threshold_was_exceeded = False
                        # current_data = np.array([map(float,line.strip().split()) for line in open(''.join([self.data_path, '/', subject, '/', input_classes[class_num], '/ft_sleeve_', str(load_num), '.log']))])

                    # while os.path.isfile(''.join([self.pkg_path, '/data/', subject, '/',str(vel),'mps/', input_classes[class_num], '/ft_sleeve_', str(i), '.pkl'])):
                        ft_threshold_was_exceeded = False
                        # print ''.join([self.pkg_path, '/data/', subject, '/', input_classes[class_num], '/ft_sleeve_', str(i), '.pkl'])
                        current_data = np.array([map(float,line.strip().split()) for line in open(''.join([self.data_path, '/', subject, '/', str(vel), 'mps/height', str(ind_j), '/ft_sleeve_', str(load_num), '.log']))])
                        del_index = []
                        for k in xrange(len(current_data)):
                            if current_data[k][0] < 5.0:
                                del_index.append(k)
                        current_data = np.delete(current_data, del_index, 0)
                        position_of_stop = 0.
                        del_index = []
                        time_of_initial_contact = None

                        # current_data = load_pickle(''.join([self.pkg_path, '/data/', subject, '/', input_classes[class_num], '/ft_sleeve_', str(i), '.pkl']))

                        # if np.max(current_data[:, 2]) >= 10. or np.max(current_data[:, 3]) >= 10. \
                        #         or np.max(current_data[:, 4]) >= 10.:
                        #     ft_threshold_was_exceeded = True
                        position_of_stop = 0.
                        del_index = []
                        for num, j in enumerate(current_data):
                            j[2] = -j[2]
                            j[3] = -j[3]
                            j[4] = -j[4]
                            j[0] = j[0] - 2.0
                            # if j[0] < 0.5:
                            #     j[1] = 0
                            if np.max(np.abs(j[2:5])) > 10. and not ft_threshold_was_exceeded:
                                time_of_stop = j[0]
                                ft_threshold_was_exceeded = True
                                for k in xrange(len(position_profile)-1):
                                    if position_profile[k, 0] <= j[0] < position_profile[k+1, 0]:
                                        position_of_stop = position_profile[k, 1] + \
                                                           (position_profile[k+1, 1] - position_profile[k, 1]) * \
                                                           (j[0]-position_profile[k, 0])/(position_profile[k+1, 0] - position_profile[k, 0])
                                j[1] = new_position
                            elif ft_threshold_was_exceeded:
                                del_index.append(num)
                                # j[1] = position_of_stop
                            else:
                                for k in xrange(len(position_profile)-1):
                                    if position_profile[k, 0] <= j[0] < position_profile[k+1, 0]:
                                        new_position = position_profile[k, 1] + \
                                                       (position_profile[k+1, 1] - position_profile[k, 1]) * \
                                                       (j[0]-position_profile[k, 0])/(position_profile[k+1, 0] - position_profile[k, 0])
                                j[1] = new_position
                                if abs(new_position - position_of_stop) < 0.0005 and new_position > 0.8:
                                    del_index.append(num)
                                elif new_position < position_of_initial_contact:
                                    del_index.append(num)
                                else:
                                    if time_of_initial_contact is None:
                                        time_of_initial_contact = j[0]
                                position_of_stop = new_position
                        # if result[ind_i][ind_j] == 'good'

                        current_data = np.delete(current_data, del_index, 0)
                        output_data = []
                        for item in current_data:
                            output_data.append([item[0]-time_of_initial_contact, item[1]-position_of_initial_contact, item[2], item[3], item[4]])
                        output_data = np.array(output_data)
                        save_number = 0
                        if result[ind_i][ind_j] == 'caught_fist' or result[ind_i][ind_j] == 'caught_other':
                            print 'I got a caught condition, but that should not be possible!'
                            break
                            # save_file = ''.join([self.data_path, '/', subject, '/formatted_three/', str(vel), 'mps/', 'caught', '/force_profile_', str(save_number), '.pkl'])
                        else:
                            save_file = ''.join([self.data_path, '/', subject, '/formatted_only_only_miss_and_good_heights/', str(vel), 'mps/', result[ind_i][ind_j], '/force_profile_', str(save_number), '.pkl'])
                        while os.path.isfile(save_file):
                            save_number += 1
                            if result[ind_i][ind_j] == 'caught_fist' or result[ind_i][ind_j] == 'caught_other':
                                print 'I got a caught condition, but that should not be possible!'
                                break
                                # save_file = ''.join([self.data_path, '/', subject, '/formatted_three/', str(vel), 'mps/', 'caught', '/force_profile_', str(save_number), '.pkl'])
                            else:
                                save_file = ''.join([self.data_path, '/', subject, '/formatted_only_only_miss_and_good_heights/', str(vel), 'mps/', result[ind_i][ind_j], '/force_profile_', str(save_number), '.pkl'])
                        print 'Saving with file name', save_file
                        save_pickle(output_data, save_file)
        print 'Done editing files!'
        # if self.plot:
        #     self.plot_data(current_data)

    def format_fake_arm_two_categories(self, subject, result):
        # print subject
        # print result
        if not subject == 'fake_arm' and not subject == 'with_sleeve_no_arm' and not subject == 'no_sleeve_no_arm':
            print 'Can only do fake arm formatting for fake arm data'
            return
        print 'Now editing files to insert position data and storing them in labeled folders.'
        # paramlist = rosparam.load_file(''.join([self.data_path, '/', subject, '/params.yaml']))
        # for params, ns in paramlist:
        #     rosparam.upload_params(ns, params)
        # arm_length = rosparam.get_param('crook_to_fist')
        # fist_length = arm_length - rosparam.get_param('crook_to_wrist')
        # print 'Fist length: ', fist_length, ' cm'
        # position_of_initial_contact = (44.0 - arm_length)/100.
        position_of_initial_contact = 0.
        position_profile = None
        for ind_i in xrange(len(result)):
            for ind_j in xrange(len(result[0])):
                if ind_i < len(result)/2:
                    load_num = ind_i
                    vel = 0.1
                else:
                    load_num = ind_i - len(result)/2
                    vel = 0.15

                if vel == 0.1:
                    # print ''.join([self.data_path, '/position_profiles/position_combined_0_1mps.pkl'])
                    position_profile = load_pickle(''.join([self.data_path, '/position_profiles/position_combined_0_1mps.pkl']))
                    # print 'Position profile loaded!'
                elif vel == 0.15:
                    position_profile = load_pickle(''.join([self.data_path, '/position_profiles/position_combined_0_15mps.pkl']))
                    # print ''.join([self.data_path, '/position_profiles/position_combined_0_15mps.pkl'])
                    # print 'Position profile loaded!'
                else:
                    print 'There is no saved position profile for this velocity! Something has gone wrong!'
                    return None
                ft_threshold_was_exceeded = False
                # current_data = np.array([map(float,line.strip().split()) for line in open(''.join([self.data_path, '/', subject, '/', input_classes[class_num], '/ft_sleeve_', str(load_num), '.log']))])

            # while os.path.isfile(''.join([self.pkg_path, '/data/', subject, '/',str(vel),'mps/', input_classes[class_num], '/ft_sleeve_', str(i), '.pkl'])):
                ft_threshold_was_exceeded = False
                # print ''.join([self.pkg_path, '/data/', subject, '/', input_classes[class_num], '/ft_sleeve_', str(i), '.pkl'])
                current_data = np.array([map(float,line.strip().split()) for line in open(''.join([self.data_path, '/', subject, '/', str(vel), 'mps/height', str(ind_j), '/ft_sleeve_', str(load_num), '.log']))])
                del_index = []
                for k in xrange(len(current_data)):
                    if current_data[k][0] < 5.0:
                        del_index.append(k)
                current_data = np.delete(current_data, del_index, 0)
                position_of_stop = 0.
                del_index = []
                time_of_initial_contact = None

                # current_data = load_pickle(''.join([self.pkg_path, '/data/', subject, '/', input_classes[class_num], '/ft_sleeve_', str(i), '.pkl']))

                # if np.max(current_data[:, 2]) >= 10. or np.max(current_data[:, 3]) >= 10. \
                #         or np.max(current_data[:, 4]) >= 10.:
                #     ft_threshold_was_exceeded = True
                position_of_stop = 0.
                del_index = []
                for num, j in enumerate(current_data):
                    j[2] = -j[2]
                    j[3] = -j[3]
                    j[4] = -j[4]
                    j[0] = j[0] - 2.0
                    # if j[0] < 0.5:
                    #     j[1] = 0
                    if np.max(np.abs(j[2:5])) > 10. and not ft_threshold_was_exceeded:
                        time_of_stop = j[0]
                        ft_threshold_was_exceeded = True
                        for k in xrange(len(position_profile)-1):
                            if position_profile[k, 0] <= j[0] < position_profile[k+1, 0]:
                                position_of_stop = position_profile[k, 1] + \
                                                   (position_profile[k+1, 1] - position_profile[k, 1]) * \
                                                   (j[0]-position_profile[k, 0])/(position_profile[k+1, 0] - position_profile[k, 0])
                        j[1] = new_position
                    elif ft_threshold_was_exceeded:
                        del_index.append(num)
                        # j[1] = position_of_stop
                    else:
                        for k in xrange(len(position_profile)-1):
                            if position_profile[k, 0] <= j[0] < position_profile[k+1, 0]:
                                new_position = position_profile[k, 1] + \
                                               (position_profile[k+1, 1] - position_profile[k, 1]) * \
                                               (j[0]-position_profile[k, 0])/(position_profile[k+1, 0] - position_profile[k, 0])
                        j[1] = new_position
                        if abs(new_position - position_of_stop) < 0.0005 and new_position > 0.8:
                            del_index.append(num)
                        elif new_position < position_of_initial_contact:
                            del_index.append(num)
                        else:
                            if time_of_initial_contact is None:
                                time_of_initial_contact = j[0]
                        position_of_stop = new_position
                # if result[ind_i][ind_j] == 'good'

                current_data = np.delete(current_data, del_index, 0)
                output_data = []
                for item in current_data:
                    output_data.append([item[0]-time_of_initial_contact, item[1]-position_of_initial_contact, item[2], item[3], item[4]])
                output_data = np.array(output_data)
                save_number = 0
                if result[ind_i][ind_j] == 'caught_fist' or result[ind_i][ind_j] == 'caught_other':
                    save_file = ''.join([self.data_path, '/', subject, '/formatted_three/', str(vel), 'mps/', 'caught', '/force_profile_', str(save_number), '.pkl'])
                else:
                    save_file = ''.join([self.data_path, '/', subject, '/formatted_three/', str(vel), 'mps/', result[ind_i][ind_j], '/force_profile_', str(save_number), '.pkl'])
                while os.path.isfile(save_file):
                    save_number += 1
                    if result[ind_i][ind_j] == 'caught_fist' or result[ind_i][ind_j] == 'caught_other':
                        save_file = ''.join([self.data_path, '/', subject, '/formatted_three/', str(vel), 'mps/', 'caught', '/force_profile_', str(save_number), '.pkl'])
                    else:
                        save_file = ''.join([self.data_path, '/', subject, '/formatted_three/', str(vel), 'mps/', result[ind_i][ind_j], '/force_profile_', str(save_number), '.pkl'])
                print 'Saving with file name', save_file
                save_pickle(output_data, save_file)
        print 'Done editing files!'
        # if self.plot:
        #     self.plot_data(current_data)

    def format_precollision_data(self, subject, result):
        # print subject
        # print result
        print 'Now editing files to insert position data and storing them in labeled folders.'
        paramlist = rosparam.load_file(''.join([self.data_path, '/', subject, '/params.yaml']))
        for params, ns in paramlist:
            rosparam.upload_params(ns, params)
        arm_length = rosparam.get_param('crook_to_fist')
        fist_length = arm_length - rosparam.get_param('crook_to_wrist')
        print 'Fist length: ', fist_length, ' cm'
        position_of_initial_contact = (44.0 - arm_length)/100.
        position_profile = None
        for ind_i in xrange(len(result)):
            for ind_j in xrange(len(result[0])):
                if ind_j == 12:
                    continue
                else:
                    if result[ind_i][ind_j] is not None:
                        if ind_i < len(result)/2:
                            load_num = ind_i
                            vel = 0.1
                        else:
                            load_num = ind_i - len(result)/2
                            vel = 0.15

                        if vel == 0.1:
                            # print ''.join([self.data_path, '/position_profiles/position_combined_0_1mps.pkl'])
                            position_profile = load_pickle(''.join([self.data_path, '/position_profiles/position_combined_0_1mps.pkl']))
                            # print 'Position profile loaded!'
                        elif vel == 0.15:
                            position_profile = load_pickle(''.join([self.data_path, '/position_profiles/position_combined_0_15mps.pkl']))
                            # print ''.join([self.data_path, '/position_profiles/position_combined_0_15mps.pkl'])
                            # print 'Position profile loaded!'
                        else:
                            print 'There is no saved position profile for this velocity! Something has gone wrong!'
                            return None
                        ft_threshold_was_exceeded = False
                        # current_data = np.array([map(float,line.strip().split()) for line in open(''.join([self.data_path, '/', subject, '/', input_classes[class_num], '/ft_sleeve_', str(load_num), '.log']))])

                    # while os.path.isfile(''.join([self.pkg_path, '/data/', subject, '/',str(vel),'mps/', input_classes[class_num], '/ft_sleeve_', str(i), '.pkl'])):
                        ft_threshold_was_exceeded = False
                        # print ''.join([self.pkg_path, '/data/', subject, '/', input_classes[class_num], '/ft_sleeve_', str(i), '.pkl'])
                        current_data = np.array([map(float,line.strip().split()) for line in open(''.join([self.data_path, '/', subject, '/', str(vel), 'mps/height', str(ind_j), '/ft_sleeve_', str(load_num), '.log']))])
                        del_index = []
                        for k in xrange(len(current_data)):
                            if current_data[k][0] < 4.0:
                                del_index.append(k)
                        current_data = np.delete(current_data, del_index, 0)
                        position_of_stop = 0.
                        del_index = []
                        time_of_initial_contact = None

                        # current_data = load_pickle(''.join([self.pkg_path, '/data/', subject, '/', input_classes[class_num], '/ft_sleeve_', str(i), '.pkl']))

                        # if np.max(current_data[:, 2]) >= 10. or np.max(current_data[:, 3]) >= 10. \
                        #         or np.max(current_data[:, 4]) >= 10.:
                        #     ft_threshold_was_exceeded = True
                        position_of_stop = 0.
                        time_of_start = None
                        del_index = []
                        for num, j in enumerate(current_data):
                            j[2] = -j[2]
                            j[3] = -j[3]
                            j[4] = -j[4]
                            j[0] = j[0] - 2.0
                            # if j[0] < 0.5:
                            #     j[1] = 0
                            if time_of_start is None:
                                time_of_start = j[0]
                            for k in xrange(len(position_profile)-1):
                                if position_profile[k, 0] <= j[0] < position_profile[k+1, 0]:
                                    new_position = position_profile[k, 1] + \
                                                   (position_profile[k+1, 1] - position_profile[k, 1]) * \
                                                   (j[0]-position_profile[k, 0])/(position_profile[k+1, 0] - position_profile[k, 0])
                            j[1] = new_position
                            if new_position <= position_of_initial_contact:
                                time_of_initial_contact = j[0]
                            elif new_position > position_of_initial_contact:
                                del_index.append(num)
                        # if result[ind_i][ind_j] == 'good'

                        current_data = np.delete(current_data, del_index, 0)
                        output_data = []
                        for item in current_data:
                            output_data.append([item[0]-time_of_start, item[1], item[2], item[3], item[4]])
                        output_data = np.array(output_data)
                        save_number = 0
                        save_file = ''.join([self.data_path, '/all/precollision_data/force_profile_', str(save_number), '.pkl'])
                        while os.path.isfile(save_file):
                            save_number += 1
                            save_file = ''.join([self.data_path, '/all/precollision_data/force_profile_', str(save_number), '.pkl'])
                        print 'Saving with file name', save_file
                        save_pickle(output_data, save_file)
        print 'Done editing files!'
        # if self.plot:
        #     self.plot_data(current_data)

    def format_precollision_data_no_arm(self, subject, result):
        # print subject
        # print result
        print 'Now editing files to insert position data and storing them in labeled folders.'
        position_profile = None
        for ind_i in xrange(len(result)):
            for ind_j in xrange(len(result[0])):
                if ind_j == 12:
                    continue
                else:
                    if result[ind_i][ind_j] is not None:
                        if ind_i < len(result)/2:
                            load_num = ind_i
                            vel = 0.1
                        else:
                            load_num = ind_i - len(result)/2
                            vel = 0.15

                        if vel == 0.1:
                            # print ''.join([self.data_path, '/position_profiles/position_combined_0_1mps.pkl'])
                            position_profile = load_pickle(''.join([self.data_path, '/position_profiles/position_combined_0_1mps.pkl']))
                            # print 'Position profile loaded!'
                        elif vel == 0.15:
                            position_profile = load_pickle(''.join([self.data_path, '/position_profiles/position_combined_0_15mps.pkl']))
                            # print ''.join([self.data_path, '/position_profiles/position_combined_0_15mps.pkl'])
                            # print 'Position profile loaded!'
                        else:
                            print 'There is no saved position profile for this velocity! Something has gone wrong!'
                            return None
                        ft_threshold_was_exceeded = False
                        # current_data = np.array([map(float,line.strip().split()) for line in open(''.join([self.data_path, '/', subject, '/', input_classes[class_num], '/ft_sleeve_', str(load_num), '.log']))])

                    # while os.path.isfile(''.join([self.pkg_path, '/data/', subject, '/',str(vel),'mps/', input_classes[class_num], '/ft_sleeve_', str(i), '.pkl'])):
                        ft_threshold_was_exceeded = False
                        # print ''.join([self.pkg_path, '/data/', subject, '/', input_classes[class_num], '/ft_sleeve_', str(i), '.pkl'])
                        current_data = np.array([map(float,line.strip().split()) for line in open(''.join([self.data_path, '/', subject, '/', str(vel), 'mps/height', str(ind_j), '/ft_sleeve_', str(load_num), '.log']))])
                        del_index = []
                        for k in xrange(len(current_data)):
                            if current_data[k][0] < 4.0:
                                del_index.append(k)
                        current_data = np.delete(current_data, del_index, 0)
                        position_of_stop = 0.
                        del_index = []
                        time_of_initial_contact = None

                        # current_data = load_pickle(''.join([self.pkg_path, '/data/', subject, '/', input_classes[class_num], '/ft_sleeve_', str(i), '.pkl']))

                        # if np.max(current_data[:, 2]) >= 10. or np.max(current_data[:, 3]) >= 10. \
                        #         or np.max(current_data[:, 4]) >= 10.:
                        #     ft_threshold_was_exceeded = True
                        position_of_stop = 0.
                        time_of_start = None
                        del_index = []
                        for num, j in enumerate(current_data):
                            j[2] = -j[2]
                            j[3] = -j[3]
                            j[4] = -j[4]
                            j[0] = j[0] - 2.0
                            # if j[0] < 0.5:
                            #     j[1] = 0
                            if time_of_start is None:
                                time_of_start = j[0]
                            for k in xrange(len(position_profile)-1):
                                if position_profile[k, 0] <= j[0] < position_profile[k+1, 0]:
                                    new_position = position_profile[k, 1] + \
                                                   (position_profile[k+1, 1] - position_profile[k, 1]) * \
                                                   (j[0]-position_profile[k, 0])/(position_profile[k+1, 0] - position_profile[k, 0])
                            j[1] = new_position
                        # if result[ind_i][ind_j] == 'good'

                        current_data = np.delete(current_data, del_index, 0)
                        output_data = []
                        for item in current_data:
                            output_data.append([item[0]-time_of_start, item[1], item[2], item[3], item[4]])
                        output_data = np.array(output_data)
                        save_number = 0
                        save_file = ''.join([self.data_path, '/all_new/precollision_data/force_profile_', str(save_number), '.pkl'])
                        while os.path.isfile(save_file):
                            save_number += 1
                            save_file = ''.join([self.data_path, '/all_new/precollision_data/force_profile_', str(save_number), '.pkl'])
                        print 'Saving with file name', save_file
                        save_pickle(output_data, save_file)
        print 'Done editing files!'
        # if self.plot:
        #     self.plot_data(current_data)

    def plot_mean_and_std_precollision_data(self, sources):
        # labels = ['all']
        fig2 = plt.figure(2)
        num_bins = 50.
        bins = np.arange(0, 0.25+0.00001, 0.25/num_bins)
        bin_values = np.arange(0, 0.25, 0.25/num_bins)+0.25/(2.*num_bins)
        ax1 = fig2.add_subplot(311)
        ax1.set_xlim(0., .2)
        ax1.set_ylim(-0.25, 0.25)
        ax1.set_xlabel('Position (m)', fontsize=10)
        ax1.set_ylabel('Force_x (N)', fontsize=15)
        ax1.set_title(''.join(['Force in direction of movement vs Position']), fontsize=15)
        ax1.tick_params(axis='x', labelsize=10)
        ax1.tick_params(axis='y', labelsize=15)
        ax2 = fig2.add_subplot(312)
        ax2.set_xlim(0, .2)
        ax2.set_ylim(-0.25, 0.25)
        ax2.set_xlabel('Position (m)', fontsize=10)
        ax2.set_ylabel('Force_z (N)', fontsize=15)
        ax2.tick_params(axis='x', labelsize=10)
        ax2.tick_params(axis='y', labelsize=15)
        ax2.set_title(''.join(['Force in upward direction vs Position']), fontsize=15)
        ax3 = fig2.add_subplot(313)
        ax3.set_xlim(0, .2)
        ax3.set_ylim(-0.25, 0.25)
        ax3.set_xlabel('Position (m)', fontsize=10)
        ax3.set_ylabel('Force_y (N)', fontsize=15)
        ax3.tick_params(axis='x', labelsize=10)
        ax3.tick_params(axis='y', labelsize=15)
        ax3.set_title(''.join(['Force in out-of-plane direction vs Position']), fontsize=15)
        ax1.axhline(0, color='black')
        ax2.axhline(0, color='black')
        ax3.axhline(0, color='black')
        colors = ['blue', 'red', 'green']
        for num, source in enumerate(sources):
            bin_entries_x = []
            bin_entries_z = []
            bin_entries_y = []
            for i in bin_values:
                bin_entries_x.append([])
                bin_entries_z.append([])
                bin_entries_y.append([])

            directory = ''.join([self.data_path, '/', source, '/precollision_data/'])
            force_file_list = os.listdir(directory)
            for file_name in force_file_list:
                # print directory+file_name
                loaded_data = load_pickle(directory+file_name)
                mean_bin_data_x = []
                mean_bin_data_z = []
                placed_in_bin = np.digitize(loaded_data[:, 1], bins)-1
                # print placed_in_bin
                # nonempty_bins = np.array(sorted(placed_in_bin))
                for i in xrange(len(placed_in_bin)):
                    # print loaded_data[i]
                    # print len(placed_in_bin)
                    # print placed_in_bin[i]
                    # print loaded_data[i, 2]
                    if placed_in_bin[i]<num_bins:
                        bin_entries_x[placed_in_bin[i]].append(loaded_data[i, 2])
                        bin_entries_z[placed_in_bin[i]].append(loaded_data[i, 4])
                        bin_entries_y[placed_in_bin[i]].append(loaded_data[i, 3])
            position_values = []
            mean_x = []
            mean_z = []
            mean_y = []
            std_x = []
            std_z = []
            std_y = []
            for i in xrange(len(bin_entries_x)):
                if not bin_entries_x[i] == []:
                    mean_x.append(np.mean(bin_entries_x[i]))
                    mean_z.append(np.mean(bin_entries_z[i]))
                    mean_y.append(np.mean(bin_entries_y[i]))
                    std_x.append(np.std(bin_entries_x[i]))
                    std_z.append(np.std(bin_entries_z[i]))
                    std_y.append(np.std(bin_entries_y[i]))
                    position_values.append(bin_values[i])
            position_values = np.array(position_values)
            mean_x = np.array(mean_x)
            mean_z = np.array(mean_z)
            mean_y = np.array(mean_y)
            std_x = np.array(std_x)
            std_z = np.array(std_z)
            std_y = np.array(std_y)

            # X1 = position_values
            # Y1 = np.mean(data_x, 0)
            # Y2 = Y1 + np.std(data_x, 0)
            # Y3 = Y1 - np.std(data_x, 0)
            # Y4 = np.mean(data_z, 0)
            # Y5 = Y4 + np.std(data_z, 0)
            # Y6 = Y4 - np.std(data_z, 0)
            # print len(X1)
            # print len(Y1)
            surf1 = ax1.plot(position_values, mean_x, color=colors[num], alpha=1, label=source, linewidth=2)
            surf1 = ax1.fill_between(position_values, mean_x + std_x, mean_x - std_x, color=colors[num], alpha=0.3)
            ax1.legend(bbox_to_anchor=(0.9, 1), loc=2, borderaxespad=0., fontsize=15)
            surf2 = ax2.plot(position_values, mean_z, color=colors[num], alpha=1, label=source, linewidth=2)
            surf2 = ax2.fill_between(position_values, mean_z + std_z, mean_z - std_z, color=colors[num], alpha=0.3)
            ax2.legend(bbox_to_anchor=(0.9, 1), loc=2, borderaxespad=0., fontsize=15)
            surf3 = ax3.plot(position_values, mean_y, color=colors[num], alpha=1, label=source, linewidth=2)
            surf3 = ax3.fill_between(position_values, mean_y + std_y, mean_y - std_y, color=colors[num], alpha=0.3)
            ax3.legend(bbox_to_anchor=(0.9, 1), loc=2, borderaxespad=0., fontsize=15)
        plt.show()

    def create_time_warped_data(self, subject):
        result_list = ['missed', 'good', 'caught_fist', 'caught_other']
        velocity_list = ['0.1mps', '0.15mps']
        for vel_folder in velocity_list:
            for result in result_list:
                force_file_list = os.listdir(''.join([self.data_path, '/',subject, '/auto_labeled/', vel_folder, '/', result, '/']))
                for f_ind in xrange(100000):
                    if 'force_profile_'+str(f_ind)+'.pkl' in force_file_list:
                        load_file = ''.join([self.data_path, '/', subject, '/auto_labeled/', vel_folder, '/', result, '/force_profile_', str(f_ind), '.pkl'])
                        # print path + vel_folders + "/" + exp + '/' + 'force_profile_'+str(f_ind)+'.pkl'
                        current_data = load_pickle(load_file)
                        if vel_folder == '0.1mps':
                            current_data = self.warp_slow_to_fast(current_data)
                        save_number = 0
                        save_file = ''.join([self.data_path, '/', subject, '/time_warped_auto/', vel_folder, '/', result, '/force_profile_', str(save_number), '.pkl'])
                        while os.path.isfile(save_file):
                            save_number += 1
                            save_file = ''.join([self.data_path, '/', subject, '/time_warped_auto/', vel_folder, '/', result, '/force_profile_', str(save_number), '.pkl'])
                        save_pickle(current_data, save_file)
                    else:
                        break

    # Converts 0.1mps data to be synced with 0.15mps data
    def warp_slow_to_fast(self, data):
        interpolated_data = []
        for i in xrange(len(data)-1):
            interpolated_data.append(data[i])
            interpolated_data.append(data[i] + 0.5*(data[i+1]-data[i]))
        interpolated_data.append(data[len(data)-1])
        interpolated_data = np.array(interpolated_data)
        warped_data = []
        for i in xrange(len(interpolated_data)):
            if i%3 == 0:
                warped_data.append(list(flatten([interpolated_data[i, 0]/1.5,interpolated_data[i,1:]])))
        return np.array(warped_data)



if __name__ == "__main__":
    rospy.init_node('force_data_formatting')

    # vel_options = [0.1, 0.15]
    # vel = vel_options[0]

    subject_options = ['subject0', 'subject1', 'subject2', 'subject3', 'subject4', 'subject5', 'subject6', 'subject7', 'subject8', 'subject9', 'subject10', 'subject11', 'subject12', 'with_sleeve_no_arm', 'no_sleeve_no_arm', 'fake_arm', 'tapo_test_data', 'wenhao_test_data', 'test_subj']
    this_subject = subject_options[1]
    height_options = ['height0', 'height1', 'height2', 'height3', 'height4', 'height5']
    height = height_options[0]

    # input_classification = ['missed', 'high', 'caught_forearm', 'caught']
    label = ['missed', 'caught_fist', 'caught_other', 'good']

    fdf = ForceDataFormatting()

    for this_subject in subject_options[0:6]+subject_options[8:13]:
        if this_subject == 'subject0':
            this_result = [[label[0], label[1], label[2], label[3]],
                           [label[0], label[1], label[1], label[3]],
                           [label[0], label[1], label[2], label[3]],
                           [label[0], label[1], label[2], label[3]],
                           [label[0], label[1], label[2], label[3]],
                           [label[0], label[1], label[2], label[3]],
                           [label[0], label[1], label[2], label[3]],
                           [label[0], label[1], label[2], label[3]],
                           [label[0], label[1], label[3], label[3]],
                           [label[0], label[0], label[3], label[3]]]
        elif this_subject == 'subject1':
            this_result = [[label[0], label[1], label[3], label[3]],
                           [label[0], label[1], label[3], label[3]],
                           [label[0], label[1], label[3], label[3]],
                           [label[0], label[1], label[3], label[3]],
                           [label[0], label[1], label[3], label[3]],
                           [label[0], label[1], label[3], label[3]],
                           [label[0], label[1], label[3], label[3]],
                           [label[0], label[1], label[3], label[3]],
                           [label[0], label[1], label[3], label[3]],
                           [label[0], label[1], label[3], label[3]]]
        elif this_subject == 'subject2':
            this_result = [[label[0], label[0], label[2], label[3]],
                           [label[0], label[0], label[2], label[3]],
                           [label[0], label[0], label[2], label[3]],
                           [label[0], label[0], label[2], label[3]],
                           [label[0], label[0], label[3], label[3]],
                           [label[0], label[1], label[2], label[3]],
                           [label[0], label[1], label[2], label[3]],
                           [label[0], label[0], label[2], label[3]],
                           [label[0], label[0], label[3], label[3]],
                           [label[0], label[0], label[3], label[3]]]
        elif this_subject == 'subject3':
            this_result = [[label[0], label[1], label[2], label[3]],
                           [label[0], label[0], label[2], label[3]],
                           [label[0], label[1], label[2], label[3]],
                           [label[0], label[1], label[2], label[3]],
                           [label[0], label[1], label[2], label[3]],
                           [label[0], label[1], label[2], label[3]],
                           [label[0], label[1], label[2], label[3]],
                           [label[0], label[0], label[2], label[3]],
                           [label[0], label[0], label[2], label[3]],
                           [label[0], label[0], label[2], label[3]]]
        elif this_subject == 'subject4':
            this_result = [[label[0], label[1], label[2], label[3]],
                           [label[0], label[1], label[2], label[3]],
                           [label[0], label[1], label[3], label[3]],
                           [label[0], label[1], label[3], label[3]],
                           [label[0], label[1], label[3], label[3]],
                           [label[0], label[1], label[3], label[3]],
                           [label[0], label[1], label[3], label[3]],
                           [label[0], label[1], label[3], label[3]],
                           [label[0], label[1], label[3], label[3]],
                           [label[0], label[2], label[3], label[3]]]
        elif this_subject == 'subject5':
            this_result = [[label[0], label[1], label[3], label[3]],
                           [label[0], label[1], label[3], label[3]],
                           [label[0], label[1], label[3], label[3]],
                           [label[0], label[1], label[3], label[3]],
                           [label[0], label[1], label[3], label[3]],
                           [label[0], label[1], label[3], label[3]],
                           [label[0], label[1], label[3], label[3]],
                           [label[0], label[1], label[3], label[3]],
                           [label[0], label[1], label[3], label[3]],
                           [label[0], label[1], label[3], label[3]]]
        elif this_subject == 'subject6':
            this_result = [[label[0], label[2], label[3], label[3]],
                           [label[0], label[2], label[3], label[3]],
                           [label[0], label[0], label[3], label[3]],
                           [label[0], label[1], label[3], label[3]],
                           [label[0], label[1], label[3], label[3]],
                           [label[0], label[1], label[3], label[3]],
                           [label[0], label[2], label[3], label[3]],
                           [label[0], label[2], label[3], label[3]],
                           [label[0], label[2], label[3], label[3]],
                           [label[0], label[2], label[3], label[3]]]
        elif this_subject == 'subject7':
            # this_result = [[label[0], label[0], label[3], label[3]],
            #                [label[0], label[0], label[3], label[3]],
            #                [label[0], label[0], label[3], label[3]],
            #                [label[0], label[0], label[3], label[3]],
            #                [label[0], label[0], label[3], label[3]],
            #                [None,     label[1],     None,     None],
            #                [None,     label[1],     None,     None],
            #                [label[0], label[0], label[3], label[3]],
            #                [label[0], label[0], label[3], label[3]],
            #                [label[0], label[2], label[3], label[3]],
            #                [label[0], label[2], label[3], label[3]],
            #                [label[0], label[2], label[3], label[3]],
            #                [None,     label[0],     None,     None],
            #                [None,     label[1],     None,     None]]
            this_result = [[label[0], label[0], label[3], label[3]],
                           [label[0], label[0], label[3], label[3]],
                           [label[0], label[0], label[3], label[3]],
                           [label[0], label[0], label[3], label[3]],
                           [label[0], label[0], label[3], label[3]],
                           [label[0], label[0], label[3], label[3]],
                           [label[0], label[0], label[3], label[3]],
                           [label[0], label[2], label[3], label[3]],
                           [label[0], label[2], label[3], label[3]],
                           [label[0], label[2], label[3], label[3]]]
        elif this_subject == 'subject8':
            this_result = [[label[0], label[2], label[2], label[3]],
                           [label[0], label[0], label[2], label[3]],
                           [label[0], label[0], label[2], label[3]],
                           [label[0], label[0], label[2], label[3]],
                           [label[0], label[1], label[2], label[3]],
                           [label[0], label[2], label[2], label[3]],
                           [label[0], label[1], label[2], label[3]],
                           [label[0], label[1], label[2], label[3]],
                           [label[0], label[1], label[2], label[3]],
                           [label[0], label[2], label[2], label[3]]]
        elif this_subject == 'subject9':
            this_result = [[label[0], label[1], label[3], label[3]],
                           [label[0], label[2], label[3], label[3]],
                           [label[0], label[1], label[3], label[3]],
                           [label[0], label[1], label[3], label[3]],
                           [label[0], label[2], label[3], label[3]],
                           [label[0], label[2], label[3], label[3]],
                           [label[0], label[1], label[3], label[3]],
                           [label[0], label[1], label[3], label[3]],
                           [label[0], label[1], label[3], label[3]],
                           [label[0], label[1], label[3], label[3]]]
        elif this_subject == 'subject10':
            this_result = [[label[0], label[2], label[3], label[3]],
                           [label[0], label[2], label[3], label[3]],
                           [label[0], label[2], label[3], label[3]],
                           [label[0], label[2], label[3], label[3]],
                           [label[0], label[0], label[2], label[3]],
                           [label[0], label[1], label[2], label[3]],
                           [label[0], label[1], label[3], label[3]],
                           [label[0], label[2], label[3], label[3]],
                           [label[0], label[1], label[3], label[3]],
                           [label[0], label[1], label[3], label[3]]]
        elif this_subject == 'subject11':
            this_result = [[label[0], label[1], label[2], label[3]],
                           [label[0], label[1], label[2], label[3]],
                           [label[0], label[1], label[3], label[3]],
                           [label[0], label[1], label[2], label[3]],
                           [label[0], label[1], label[2], label[3]],
                           [label[0], label[1], label[2], label[3]],
                           [label[0], label[1], label[2], label[3]],
                           [label[0], label[1], label[2], label[3]],
                           [label[0], label[1], label[2], label[3]],
                           [label[0], label[1], label[2], label[3]]]
        elif this_subject == 'subject12':
            this_result = [[label[0], label[0], label[3], label[3]],
                           [label[0], label[0], label[2], label[3]],
                           [label[0], label[0], label[2], label[3]],
                           [label[0], label[0], label[2], label[3]],
                           [label[0], label[1], label[3], label[3]],
                           [label[0], label[1], label[2], label[3]],
                           [label[0], label[0], label[2], label[3]],
                           [label[0], label[1], label[1], label[3]],
                           [label[0], label[1], label[2], label[3]],
                           [label[0], label[1], label[2], label[3]]]
        elif this_subject == 'fake_arm':
            this_result = [[label[0], label[3]],
                           [label[0], label[3]],
                           [label[0], label[3]],
                           [label[0], label[3]],
                           [label[0], label[3]],
                           [label[0], label[3]],
                           [label[0], label[3]],
                           [label[0], label[3]],
                           [label[0], label[3]],
                           [label[0], label[3]],
                           [label[0], label[3]],
                           [label[0], label[3]],
                           [label[0], label[3]],
                           [label[0], label[3]],
                           [label[0], label[3]],
                           [label[0], label[3]],
                           [label[0], label[3]],
                           [label[0], label[3]],
                           [label[0], label[3]],
                           [label[0], label[3]],
                           [label[0], label[3]],
                           [label[0], label[3]],
                           [label[0], label[3]],
                           [label[0], label[3]],
                           [label[0], label[3]],
                           [label[0], label[3]],
                           [label[0], label[3]],
                           [label[0], label[3]],
                           [label[0], label[3]],
                           [label[0], label[3]],
                           [label[0], label[3]],
                           [label[0], label[3]],
                           [label[0], label[3]],
                           [label[0], label[3]],
                           [label[0], label[3]],
                           [label[0], label[3]],
                           [label[0], label[3]],
                           [label[0], label[3]],
                           [label[0], label[3]],
                           [label[0], label[3]]]
        elif this_subject == 'with_sleeve_no_arm':
            this_result = [[label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]]]
        elif this_subject == 'no_sleeve_no_arm':
            this_result = [[label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]],
                           [label[0]]]

        # fdf.format_data_four_categories(this_subject, this_result)
        # fdf.format_data_three_categories(this_subject, this_result)
        # fdf.format_fake_arm_two_categories(this_subject, this_result)
        # fdf.autolabel_and_set_position(this_subject, this_result)
        # fdf.create_time_warped_data(this_subject)
        # fdf.format_precollision_data(this_subject, this_result)
        # fdf.format_precollision_data_no_arm(this_subject, this_result)
        fdf.format_data_only_miss_and_good_heights(this_subject, this_result)
    # fdf.plot_mean_and_std_precollision_data(['all', 'all_new'])


















