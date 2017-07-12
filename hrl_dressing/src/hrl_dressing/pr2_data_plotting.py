#!/usr/bin/python

import roslib
import rospy, rospkg, rosparam
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Bool
import numpy as np
import os.path

roslib.load_manifest('hrl_dressing')
roslib.load_manifest('zenither')
import zenither.zenither as zenither

roslib.load_manifest('hrl_lib')
from hrl_lib.util import save_pickle, load_pickle
from hrl_msgs.msg import FloatArray
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

from os import listdir

class PR2_FT_Data_Plotting(object):
    def __init__(self, plot=False, trial_number=None):
        # print 'Initializing Sleeve Rig'

        # self.data_path = '/home/ari/svn/robot1_data/usr/ari/data/hrl_dressing'
        rospack = rospkg.RosPack()
        self.pkg_path = rospack.get_path('hrl_dressing')
        # file_names = listdir(data_path)

        file_names = ['ft_sleeve_0.pkl', 'ft_sleeve_1.pkl', 'ft_sleeve_2.pkl']
        X = []
        Z = []
        T = []

        for file_name in file_names:
            data = load_pickle(self.pkg_path + '/data/pr2_test_ft_data/' + file_name)
            # print data
            X.append(data[:,1])
            Z.append(data[:,3])
            T.append(data[:,0])
        fig1 = plt.figure(1)
        ax1 = fig1.add_subplot(111)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Force Direction of movement (N)')
        surf11 = ax1.scatter(T[0], -X[0], color="green", s=1, alpha=1)
        surf12 = ax1.scatter(T[1], -X[1], color="orange", s=1, alpha=1)
        surf13 = ax1.scatter(T[2], -X[2], color="red", s=1, alpha=1)
        plt.show()
        fig2 = plt.figure(2)
        ax2 = fig2.add_subplot(111)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Force Direction of Gravity (N)')
        surf21 = ax2.scatter(T[0], -Z[0], color="green", s=1, alpha=1)
        surf22 = ax2.scatter(T[1], -Z[1], color="orange", s=1, alpha=1)
        surf23 = ax2.scatter(T[2], -Z[2], color="red", s=1, alpha=1)
        plt.show()



    def load_and_plot(self, file_path, label):
        loaded_data = load_pickle(file_path)
        self.plot_data(loaded_data, label)
        print 'Plotted!'

    def plot_data(self, my_data, label):
        fig1 = plt.figure(1)
        ax1 = fig1.add_subplot(111)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Position (m)')
        ax1.set_title(''.join(['Position vs Time for: ', label, ' type']))
        ax1.set_xlim(0.0, 14)
        ax1.set_ylim(0, .86)
        X1 = my_data[:, 0]
        Y1 = my_data[:, 1]
        surf = ax1.scatter(X1, Y1, color="red", s=1, alpha=1)
        fig2 = plt.figure(2)
        ax2 = fig2.add_subplot(111)
        ax2.set_xlim(0., 17)
        ax2.set_ylim(-10.0, 1.0)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Force_x (N)')
        ax2.set_title(''.join(['Force in direction of movement vs Time for: ', label, ' type']))
        X2 = my_data[:, 0]
        Y2 = my_data[:, 2]
        surf = ax2.plot(X2, Y2, color="blue", alpha=1)
        fig3 = plt.figure(3)
        ax3 = fig3.add_subplot(111)
        ax3.set_xlim(0., .8)
        ax3.set_ylim(-10.0, 1.0)
        ax3.set_xlabel('Position (m)')
        ax3.set_ylabel('Force_x (N)')
        ax3.set_title(''.join(['Force in direction of movement vs Position for: ', label, ' type']))
        X3 = my_data[:, 1]
        Y3 = my_data[:, 2]
        surf = ax3.plot(X3, Y3, color="blue", alpha=1)
        fig4 = plt.figure(4)
        ax4 = fig4.add_subplot(111)
        ax4.set_xlim(0, .8)
        ax4.set_ylim(-10.0, 1.0)
        ax4.set_xlabel('Position (m)')
        ax4.set_ylabel('Force_z (N)')
        ax4.set_title(''.join(['Force in upward direction vs Position for: ', label, ' type']))
        X4 = my_data[:, 1]
        Y4 = my_data[:, 4]
        surf = ax4.plot(X4, Y4, color="green", alpha=1)
        fig5 = plt.figure(5)
        ax5 = fig5.add_subplot(111)
        surf = ax5.plot(Y3)

    def plot_all_data(self, subjects, labels):
        fig1 = plt.figure(1)
        for num, label in enumerate(labels):
            for subject in subjects:
                # fig1 = plt.figure(2*num+1)
                ax1 = fig1.add_subplot(331+2*num)
                ax1.set_xlim(0., .8)
                ax1.set_ylim(-10.0, 1.0)
                ax1.set_xlabel('Position (m)')
                ax1.set_ylabel('Force_x (N)')
                ax1.set_title(''.join(['Force in direction of movement vs Position for: ', label, ' type']))
                # fig2 = plt.figure(2*num+2)
                ax2 = fig1.add_subplot(332+2*num)
                ax2.set_xlim(0, .8)
                ax2.set_ylim(-10.0, 1.0)
                ax2.set_xlabel('Position (m)')
                ax2.set_ylabel('Force_z (N)')
                ax2.set_title(''.join(['Force in upward direction vs Position for: ', label, ' type']))
                vel = 0.1
                directory = ''.join([data_path, '/', subject, '/formatted_three/', str(vel),'mps/', label, '/'])
                force_file_list = os.listdir(directory)
                for file_name in force_file_list:
                    # print directory+file_name
                    loaded_data = load_pickle(directory+file_name)
                    X1 = loaded_data[:, 1]
                    Y1 = loaded_data[:, 2]
                    surf1 = ax1.plot(X1, Y1, color="blue", alpha=1)
                    X2 = loaded_data[:, 1]
                    Y2 = loaded_data[:, 4]
                    surf2 = ax2.plot(X2, Y2, color="green", alpha=1)
                vel = 0.15
                directory = ''.join([data_path, '/', subject, '/formatted_three/', str(vel),'mps/', label, '/'])
                force_file_list = os.listdir(directory)
                for file_name in force_file_list:
                    # print directory+file_name
                    loaded_data = load_pickle(directory+file_name)
                    X1 = loaded_data[:, 1]
                    Y1 = loaded_data[:, 2]
                    surf1 = ax1.plot(X1, Y1, color="blue", alpha=1)
                    X2 = loaded_data[:, 1]
                    Y2 = loaded_data[:, 4]
                    surf2 = ax2.plot(X2, Y2, color="green", alpha=1)
        # plt.show()

    def plot_mean_and_std(self, subjects, labels):
        fig2 = plt.figure(2)
        num_bins = 150.
        bins = np.arange(0, 0.85+0.00001, 0.85/num_bins)
        bin_values = np.arange(0, 0.85, 0.85/num_bins)+0.85/(2.*num_bins)
        ax1 = fig2.add_subplot(211)
        ax1.set_xlim(0., .85)
        ax1.set_ylim(-10.0, 1.0)
        ax1.set_xlabel('Position (m)', fontsize=20)
        ax1.set_ylabel('Force_x (N)', fontsize=20)
        ax1.set_title(''.join(['Force in direction of movement vs Position']), fontsize=20)
        ax1.tick_params(axis='x', labelsize=20)
        ax1.tick_params(axis='y', labelsize=20)
        # fig2 = plt.figure(2*num+2)
        ax2 = fig2.add_subplot(212)
        ax2.set_xlim(0, .85)
        ax2.set_ylim(-10.0, 1.0)
        ax2.set_xlabel('Position (m)', fontsize=20)
        ax2.set_ylabel('Force_z (N)', fontsize=20)
        ax2.tick_params(axis='x', labelsize=20)
        ax2.tick_params(axis='y', labelsize=20)
        ax2.set_title(''.join(['Force in upward direction vs Position']), fontsize=20)
        colors = ['blue', 'green', 'red']
        for num, label in enumerate(labels):
            bin_entries_x = []
            bin_entries_z = []
            for i in bin_values:
                bin_entries_x.append([])
                bin_entries_z.append([])
            for subject in subjects:
                this_subj_data_x = []
                this_subj_data_z = []
                # fig1 = plt.figure(2*num+1)
                # ax1 = fig2.add_subplot(331+2*num)
                # ax1.set_xlim(0., .85)
                # ax1.set_ylim(-10.0, 1.0)
                # ax1.set_xlabel('Position (m)')
                # ax1.set_ylabel('Force_x (N)')
                # ax1.set_title(''.join(['Force in direction of movement vs Position for: ', label, ' type']))
                # # fig2 = plt.figure(2*num+2)
                # ax2 = fig2.add_subplot(332+2*num)
                # ax2.set_xlim(0, .85)
                # ax2.set_ylim(-10.0, 1.0)
                # ax2.set_xlabel('Position (m)')
                # ax2.set_ylabel('Force_z (N)')
                # ax2.set_title(''.join(['Force in upward direction vs Position for: ', label, ' type']))
                vel = 0.1
                directory = ''.join([data_path, '/', subject, '/formatted_three/', str(vel),'mps/', label, '/'])
                force_file_list = os.listdir(directory)
                for file_name in force_file_list:
                    # print directory+file_name
                    loaded_data = load_pickle(directory+file_name)
                    mean_bin_data_x = []
                    mean_bin_data_z = []
                    placed_in_bin = np.digitize(loaded_data[:, 1], bins)-1
                    # nonempty_bins = np.array(sorted(placed_in_bin))
                    for i in xrange(len(placed_in_bin)):
                        bin_entries_x[placed_in_bin[i]].append(loaded_data[i, 2])
                        bin_entries_z[placed_in_bin[i]].append(loaded_data[i, 4])
                vel = 0.15
                directory = ''.join([data_path, '/', subject, '/formatted_three/', str(vel),'mps/', label, '/'])
                force_file_list = os.listdir(directory)
                for file_name in force_file_list:
                    # print directory+file_name
                    loaded_data = load_pickle(directory+file_name)
                    mean_bin_data_x = []
                    mean_bin_data_z = []
                    placed_in_bin = np.digitize(loaded_data[:, 1], bins)-1
                    # nonempty_bins = np.array(sorted(placed_in_bin))
                    for i in xrange(len(placed_in_bin)):
                        bin_entries_x[placed_in_bin[i]].append(loaded_data[i, 2])
                        bin_entries_z[placed_in_bin[i]].append(loaded_data[i, 4])
            position_values = []
            mean_x = []
            mean_z = []
            std_x = []
            std_z = []
            for i in xrange(len(bin_entries_x)):
                if not bin_entries_x[i] == []:
                    mean_x.append(np.mean(bin_entries_x[i]))
                    mean_z.append(np.mean(bin_entries_z[i]))
                    std_x.append(np.std(bin_entries_x[i]))
                    std_z.append(np.std(bin_entries_z[i]))
                    position_values.append(bin_values[i])
            position_values = np.array(position_values)
            mean_x = np.array(mean_x)
            mean_z = np.array(mean_z)
            std_x = np.array(std_x)
            std_z = np.array(std_z)

            # X1 = position_values
            # Y1 = np.mean(data_x, 0)
            # Y2 = Y1 + np.std(data_x, 0)
            # Y3 = Y1 - np.std(data_x, 0)
            # Y4 = np.mean(data_z, 0)
            # Y5 = Y4 + np.std(data_z, 0)
            # Y6 = Y4 - np.std(data_z, 0)
            # print len(X1)
            # print len(Y1)
            surf1 = ax1.plot(position_values, mean_x, color=colors[num], alpha=1, label=label, linewidth=2)
            surf1 = ax1.fill_between(position_values, mean_x + std_x, mean_x - std_x, color=colors[num], alpha=0.3)
            ax1.legend(bbox_to_anchor=(0.9, 1), loc=2, borderaxespad=0., fontsize=20)
            surf2 = ax2.plot(position_values, mean_z, color=colors[num], alpha=1, label=label, linewidth=2)
            surf2 = ax2.fill_between(position_values, mean_z + std_z, mean_z - std_z, color=colors[num], alpha=0.3)
            ax2.legend(bbox_to_anchor=(0.9, 1), loc=2, borderaxespad=0., fontsize=20)

    def histogram_of_stop_point_fist(self, subjects, labels):
        fig1 = plt.figure(3)
        labels = ['caught_fist']
        stop_locations = []
        for num, label in enumerate(labels):
            for subject in subjects:
                # fig1 = plt.figure(2*num+1)
                ax1 = fig1.add_subplot(111)
                ax1.set_xlim(0.2, .5)
                # ax1.set_ylim(-10.0, 1.0)
                ax1.set_xlabel('Stop Position (m)', fontsize=20)
                ax1.set_ylabel('Number of trials', fontsize=20)
                ax1.tick_params(axis='x', labelsize=20)
                ax1.tick_params(axis='y', labelsize=20)
                ax1.set_title(''.join(['Stop position when caught on fist, when started at tip of fist']), fontsize=20)
                vel = 0.1
                directory = ''.join([data_path, '/', subject, '/formatted/', str(vel),'mps/', label, '/'])
                force_file_list = os.listdir(directory)
                for file_name in force_file_list:
                    # print directory+file_name
                    loaded_data = load_pickle(directory+file_name)
                    stop_locations.append(np.max(loaded_data[:,1]))
                vel = 0.15
                directory = ''.join([data_path, '/', subject, '/formatted/', str(vel),'mps/', label, '/'])
                force_file_list = os.listdir(directory)
                for file_name in force_file_list:
                    loaded_data = load_pickle(directory+file_name)
                    stop_locations.append(np.max(loaded_data[:,1]))
        mu = np.mean(stop_locations)
        sigma = np.std(stop_locations)
        print 'The mean of the stop location is: ', mu
        print 'The standard deviation of the stop location is: ', sigma
        n, bins, patches = ax1.hist(stop_locations, 10, color="green", alpha=0.75)
        points = np.arange(0, 10, 0.001)
        y = mlab.normpdf(points, mu, sigma)
        l = ax1.plot(points, y, 'r--', linewidth=2)

    def histogram_of_stop_point_elbow(self, subjects, labels):
        fig1 = plt.figure(4)
        fig2 = plt.figure(5)
        labels = ['good']
        stop_locations = []
        arm_lengths = []
        for num, label in enumerate(labels):
            for subj_num, subject in enumerate(subjects):
                subject_stop_locations = []
                paramlist = rosparam.load_file(''.join([data_path, '/', subject, '/params.yaml']))
                for params, ns in paramlist:
                    rosparam.upload_params(ns, params)
                arm_length = rosparam.get_param('crook_to_fist')/100.
                # fig1 = plt.figure(2*num+1)
                ax1 = fig1.add_subplot(111)
                ax1.set_xlim(0.2, .4)
                # ax1.set_ylim(-10.0, 1.0)
                ax1.set_xlabel('Stop Position (m)', fontsize=20)
                ax1.set_ylabel('Number of trials', fontsize=20)
                ax1.set_title(''.join(['Difference between arm length and stop position at the elbow']), fontsize=20)
                ax1.tick_params(axis='x', labelsize=20)
                ax1.tick_params(axis='y', labelsize=20)
                ax2 = fig2.add_subplot(431+subj_num)
                ax2.set_xlim(0.2, .4)
                # ax1.set_ylim(-10.0, 1.0)
                ax2.set_xlabel('Position (m)')
                ax2.set_ylabel('Number of trials')
                ax2.set_title(''.join(['Stop position for "Good" outcome']), fontsize=20)
                vel = 0.1
                directory = ''.join([data_path, '/', subject, '/formatted_three/', str(vel),'mps/', label, '/'])
                force_file_list = os.listdir(directory)
                for file_name in force_file_list:
                    # print directory+file_name
                    loaded_data = load_pickle(directory+file_name)
                    stop_locations.append(np.max(loaded_data[:,1])-arm_length)
                    subject_stop_locations.append(np.max(loaded_data[:,1])-arm_length)
                    arm_lengths.append(arm_length)
                vel = 0.15
                directory = ''.join([data_path, '/', subject, '/formatted_three/', str(vel),'mps/', label, '/'])
                force_file_list = os.listdir(directory)
                for file_name in force_file_list:
                    loaded_data = load_pickle(directory+file_name)
                    stop_locations.append(np.max(loaded_data[:,1]-arm_length))
                    subject_stop_locations.append(np.max(loaded_data[:,1])-arm_length)
                ax2.hist(subject_stop_locations)
        mu = np.mean(stop_locations)
        sigma = np.std(stop_locations)
        print 'The minimum arm length is: ', np.min(arm_lengths)
        print 'The max arm length is: ', np.max(arm_lengths)
        print 'The mean arm length is: ', np.mean(arm_lengths)
        print 'The mean of the stop location is: ', mu
        print arm_lengths
        print 'The standard deviation of the stop location is: ', sigma
        n, bins, patches = ax1.hist(stop_locations, 10, color="green", alpha=0.75)
        points = np.arange(0, 10, 0.001)
        y = mlab.normpdf(points, mu, sigma)
        l = ax1.plot(points, y, 'r--', linewidth=2)

    def start_recording_data(self, num):
        self.array_to_save = np.zeros([3000, 4])
        self.array_line = 0
        # self.arm_file = open(''.join([self.pkg_path, '/data/ft_arm_', str(num), '.log']), 'w')
        self.sleeve_file = open(''.join([self.pkg_path, '/data/pr2_test_ft_data/', 'ft_sleeve_', str(num), '.log']), 'w')
        # self.arm_file.write('Time(us) Pos(m) force_x(N) force_y(N) force_z(N) torque_x(Nm) torque_y(Nm) torque_z(Nm) \n')
        # self.sleeve_file.write('Time(us) Pos(m) force_x(N) force_y(N) force_z(N) torque_x(Nm) torque_y(Nm) torque_z(Nm) \n')
        # self.position_file = open(''.join([self.pkg_path, '/data/position_', str(num), '.log']), 'w')
        self.start_record_time = rospy.Time.now()
        self.recording = True

    def stop_recording_data(self, num):
        self.recording = False
        # self.arm_file.close()
        self.sleeve_file.close()
        # self.position_file.close()
        save_pickle(self.array_to_save, ''.join([self.pkg_path, '/data/pr2_test_ft_data/', 'ft_sleeve_', str(num), '.pkl']))
        self.trial_number += 1
        # self.array_to_save = np.zeros([1000, 5])
        # self.array_line = 0

if __name__ == "__main__":
    rospy.init_node('PR2_FT_Data_Plotting')


    mode = 'autorun'
    # mode = None
    plot = True
    # plot = False
    num = 20
    vel = 0.1
    subject_options = ['subject0', 'subject1', 'subject2', 'subject3', 'subject4', 'subject5', 'subject6', 'subject7', 'subject8', 'subject9', 'subject10', 'subject11', 'subject12',
                       'with_sleeve_no_arm', 'no_sleeve_no_arm', 'moved_rig_onto_drawers', 'moved_rig_back', 'testing_level', 'tapo_test_data','wenhao_test_data', 'test_subj']
    subject = subject_options[17]
    rc = PR2_FT_Data_Plotting()
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('hrl_dressing')
    data_path = '/home/ari/svn/robot1_data/usr/ari/data/hrl_dressing'

    save_number = 0

    # input_classification = ['183mm_height_missed_sleeve', '222mm_height_caught',
    #                         '222mm_height_missed_sleeve', '408mm_height_high', '325mm_height_good']
    # input_classification = ['missed', 'high', 'caught_forearm', 'caught']
    # output_classification = ['missed', 'caught', 'missed', 'high', 'good']

    '''
    # output_classification = ['missed', 'good', 'caught_fist', 'caught_other']
    output_classification = ['missed', 'good', 'caught']
    # rc.plot_all_data(subject_options[0:6]+subject_options[7:13], output_classification)
    rc.histogram_of_stop_point_fist(subject_options[0:6]+subject_options[7:13], output_classification)
    rc.histogram_of_stop_point_elbow(subject_options[0:6]+subject_options[7:13], output_classification)
    # rc.plot_mean_and_std(subject_options[0:6]+subject_options[7:13], output_classification)
    plt.show()
    '''


    '''
    label = output_classification[1]
    for subject in subject_options[0:11]:
        vel = 0.1
        directory = ''.join([data_path, '/', subject, '/formatted/', str(vel),'mps/', label, '/'])
        force_file_list = os.listdir(directory)
        for file_name in force_file_list:
            print directory+file_name
            rc.load_and_plot(directory+file_name, label)
        vel = 0.15
        directory = ''.join([data_path, '/', subject, '/formatted/', str(vel),'mps/', label, '/'])
        force_file_list = os.listdir(directory)
        for file_name in force_file_list:
            print directory+file_name
            rc.load_and_plot(directory+file_name, label)
    plt.show()
    '''

    '''
    for i in xrange(4, 5):
        save_number = i
        vel = 0.1
        file_name = ''.join([data_path, '/', subject, '/time_warped_auto/', str(vel),'mps/', label, '/force_profile_', str(save_number), '.pkl'])
        # rc.load_and_plot(file_name, label)
        vel = 0.15
        file_name = ''.join([data_path, '/', subject, '/auto_labeled/', str(vel),'mps/', label, '/force_profile_', str(save_number), '.pkl'])
        rc.load_and_plot(file_name, label)
    plt.show()
    '''



    # file_name = ''.join([data_path, '/', subject, '/', str(vel),'mps/', output_classification[1], '/force_profile_', str(save_number), '.pkl'])

    # rc.set_position_data_from_saved_movement_profile('wenhao_test_data/183mm_height_missed_sleeve')
    # rc.set_position_data_from_saved_movement_profile(subject, input_classification, output_classification)
    # rc.set_position_data_from_saved_movement_profile('wenhao_test_data/222mm_height_missed_sleeve')
    # rc.set_position_data_from_saved_movement_profile('wenhao_test_data/408mm_height_high')
    # rc.set_position_data_from_saved_movement_profile('wenhao_test_data/325mm_height_good')
    # rig_control.repeated_movements()

    # rospy.spin()




