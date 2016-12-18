#!/usr/bin/python

import roslib
import rospy, rospkg, rosparam
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Bool
import numpy as np
import os.path
import ghmm
import copy

roslib.load_manifest('hrl_dressing')
roslib.load_manifest('zenither')
import zenither.zenither as zenither

roslib.load_manifest('hrl_lib')
from hrl_lib.util import save_pickle, load_pickle
from hrl_msgs.msg import FloatArray
from pr2_controllers_msgs.msg import SingleJointPositionActionGoal
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

from hrl_haptic_manipulation_in_clutter_srvs.srv import EnableHapticMPC

from helper_functions import createBMatrix, Bmat_to_pos_quat

from geometry_msgs.msg import PoseArray, Pose, PoseStamped, Point, Quaternion, PoseWithCovarianceStamped, Twist
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class PR2_FT_Data_Acquisition(object):
    def __init__(self, plot=False, trial_number=None, realtime_HMM=False):
        # print 'Initializing Sleeve Rig'
        self.total_start_time = rospy.Time.now()
        rospy.loginfo('Initializing PR2 F/T Data Acquisition')

        self.categories = ['missed/', 'good/', 'caught/']

        if not trial_number:
            self.trial_number = 0
        else:
            self.trial_number = trial_number
        self.plot = plot

        rospack = rospkg.RosPack()
        self.pkg_path = rospack.get_path('hrl_dressing')
        self.base_selection_pkg_path = rospack.get_path('hrl_base_selection')

        self.data_path = '/home/ari/svn/robot1_data/usr/ari/data/hrl_dressing'

        self.realtime_HMM = realtime_HMM
        if self.realtime_HMM:
            self.HMM_last_run_time = rospy.Time.now()
            self.myHmms = []
            self.F = ghmm.Float()
            self.initialize_HMM()

        self.recording = False

        self.sleeve_file = None

        self.start_record_time = 0.

        self.array_to_save = np.zeros([3000, 4])
        self.array_line = 0

        self.continue_collecting = True

        self.ft_sleeve_bias_x = 0.
        self.ft_sleeve_bias_y = 0.
        self.ft_sleeve_bias_z = 0.
        self.ft_sleeve_bias_t_x = 0.
        self.ft_sleeve_bias_t_y = 0.
        self.ft_sleeve_bias_t_z = 0.
        self.ft_arm_bias_x = 0.
        self.ft_arm_bias_y = 0.
        self.ft_arm_bias_z = 0.
        self.ft_arm_bias_t_x = 0.
        self.ft_arm_bias_t_y = 0.
        self.ft_arm_bias_t_z = 0.

        self.ft_sleeve_biased = False
        self.calibrate_now = False
        self.ft_arm_biased = False

        self.initialize_ft_sensor()

        self.traj_data = load_pickle(self.base_selection_pkg_path+'/data/saved_results/large_search_space/pr2_grasps.pkl')
        self.pr2_config = load_pickle(self.base_selection_pkg_path+'/data/saved_results/large_search_space/best_pr2_config.pkl')
        self.pr2_torso_lift_msg = SingleJointPositionActionGoal()
        self.pr2_torso_lift_msg.goal.position = self.pr2_config[0][2]

        self.r_arm_pub = rospy.Publisher('/right_arm/haptic_mpc/joint_trajectory', JointTrajectory, queue_size=1)
        self.r_arm_pose_array_pub = rospy.Publisher('/right_arm/haptic_mpc/goal_pose_array', PoseArray, queue_size=1)
        self.r_arm_pose_pub = rospy.Publisher('/right_arm/haptic_mpc/goal_pose', PoseStamped, queue_size=1)
        self.rviz_trajectory_pub = rospy.Publisher('dressing_trajectory_visualization', PoseArray, queue_size=1, latch=True)
        self.pr2_lift_pub = rospy.Publisher('torso_controller/position_joint_action/goal',
                                            SingleJointPositionActionGoal, queue_size=10)

        rospy.wait_for_service('/right_arm/haptic_mpc/enable_mpc')
        self.mpc_enabled_service = rospy.ServiceProxy("/right_arm/haptic_mpc/enable_mpc", EnableHapticMPC)

        # self.start_pose = None
        self.start_pose = self.define_start_pose()

        self.r_pose_trajectory = self.define_pose_trajectory()

        self.ready_for_movements()

    def define_pose_trajectory(self):
        basefootprint_B_poses = self.traj_data
        poseArray = PoseArray()

        poseArray.header.frame_id = 'base_footprint'
        # traj_vector = np.array(self.traj_data[7:])-np.array(self.traj_data[:3])
        # ori = self.traj_data[3:7]
        # num_waypoints = int(np.max([np.floor(np.linalg.norm(traj_vector)/.1), 1]))
        for pose in basefootprint_B_poses:
            newPose = Pose()
            # pos = np.array(self.traj_data[:3])+(i+1)/num_waypoints*(np.array(self.traj_data[7:])-np.array(self.traj_data[:3]))
            pos, ori = Bmat_to_pos_quat(pose)
            newPose.position.x = pos[0]
            newPose.position.y = pos[1]
            newPose.position.z = pos[2]
            newPose.orientation.x = ori[0]
            newPose.orientation.y = ori[1]
            newPose.orientation.z = ori[2]
            newPose.orientation.w = ori[3]
            poseArray.poses.append(copy.copy(newPose))
        # poseArray.header.stamp = rospy.Time.now()
        poseArray.header.stamp = rospy.Time(0)
        self.rviz_trajectory_pub.publish(poseArray)
        print 'Trajectory is ready'
        return poseArray

    def define_start_pose(self):

        basefootprint_B_start_pose = self.traj_data[0]
        pos_out, ori_out = Bmat_to_pos_quat(basefootprint_B_start_pose)

        startpose = PoseStamped()
        # startpose.header.stamp = rospy.Time.now()
        startpose.header.stamp = rospy.Time(0)
        startpose.header.frame_id = 'base_footprint'

        startpose.pose.position.x = pos_out[0]
        startpose.pose.position.y = pos_out[1]
        startpose.pose.position.z = pos_out[2]
        startpose.pose.orientation.x = ori_out[0]
        startpose.pose.orientation.y = ori_out[1]
        startpose.pose.orientation.z = ori_out[2]
        startpose.pose.orientation.w = ori_out[3]
        return startpose

    def initialize_pr2_config(self):
        resp = self.mpc_enabled_service('enabled')
        self.pr2_lift_pub.publish(self.pr2_torso_lift_msg)
        self.r_arm_pose_pub.publish(self.start_pose)

    def initialize_HMM(self):

        # self.myHmms = self.ghmm.HMMOpen(self.pkg_path+'/data/hmm_rig_subjects.xml')
        self.myHmms = []
        for i in xrange(len(self.categories)):
            self.myHmms.append(ghmm.HMMOpen('/home/ari/svn/robot1_data/usr/ari/data/hrl_dressing/hmm_rig_subjects_'+self.categories[i][:len(self.categories[i])-1]+'.xml'))
        rospy.loginfo('HMMs loaded! Ready to classify!')

    def initialize_ft_sensor(self):

        rospy.loginfo('Initializing FT sensor')

        self.ft_sleeve_sub = rospy.Subscriber('/force_torque_pr2_sleeve', WrenchStamped, self.ft_sleeve_cb)

        # self.ft_arm_sub = rospy.Subscriber('/force_torque_arm', WrenchStamped, self.ft_arm_cb)

        # print 'Zenither ready for action!'
        rospy.loginfo('FT sensor ready for action!')

        # rospy.spin()

    def ft_sleeve_cb(self, msg):
        if not self.ft_sleeve_biased or self.calibrate_now:
            rospy.sleep(0.5)
            self.time_since_last_cb = rospy.Time.now()
            self.ft_sleeve_bias_x = msg.wrench.force.x
            self.ft_sleeve_bias_y = msg.wrench.force.y
            self.ft_sleeve_bias_z = msg.wrench.force.z
            self.ft_sleeve_bias_t_x = msg.wrench.torque.x
            self.ft_sleeve_bias_t_y = msg.wrench.torque.y
            self.ft_sleeve_bias_t_z = msg.wrench.torque.z
            self.ft_sleeve_biased = True
            self.calibrate_now = False
        if rospy.Time.now().to_sec() - self.time_since_last_cb.to_sec() > 0.05:
            print 'The force-torque sensor callback is too slow. That is potentially very bad. Aborting everything!!!'
        self.time_since_last_cb = rospy.Time.now()
        x_force = msg.wrench.force.x-self.ft_sleeve_bias_x
        y_force = msg.wrench.force.y-self.ft_sleeve_bias_y
        z_force = msg.wrench.force.z-self.ft_sleeve_bias_z
        x_torque = msg.wrench.torque.x-self.ft_sleeve_bias_t_x
        y_torque = msg.wrench.torque.y-self.ft_sleeve_bias_t_y
        z_torque = msg.wrench.torque.z-self.ft_sleeve_bias_t_z

        if self.recording:
            t = rospy.Time.now() - self.start_record_time
            self.sleeve_file.write(''.join([str(t.to_sec()), ' %f %f %f %f %f %f \n' %
                                            (x_force,
                                             y_force,
                                             z_force,
                                             x_torque,
                                             y_torque,
                                             z_torque)]))
            self.array_to_save[self.array_line] = [t.to_sec(), -x_force, -y_force, -z_force]
            if np.linalg.norm([x_force, y_force, z_force]) >= 10.:
                print 'Force exceeded 10 Newtons!! Stopping the arm!'
                stopPose = PoseStamped()
                self.r_arm_pose_pub.publish(stopPose)
                resp = self.mpc_enabled_service('disabled')
                self.recording = False
            if self.realtime_HMM:
                HMM_run_timer = rospy.Time.now() - self.HMM_last_run_time
                if HMM_run_timer.to_sec() > 0.5:
                    self.HMM_last_run_time = rospy.Time.now()
                    testing_with_saved_data = False
                    if testing_with_saved_data:
                        # /home/ari/svn/robot1_data/usr/ari/data/hrl_dressing/subject0/formatted_three/0.1mps/caught
                        # saved_data = load_picke(self.data_path + '/subject0/formatted_three/0.1mps/good/force_profile_1.pkl')
                        saved_data = load_pickle(self.pkg_path + '/data/pr2_test_ft_data/ft_sleeve_2.pkl')
                        # self.run_HMM_realtime(np.dstack([saved_data[:, 1], saved_data[:,4]*1, saved_data[:,2]*1])[0])
                        self.run_HMM_realtime(np.dstack([saved_data[:, 0], saved_data[:,4]*-1, saved_data[:,2]*-1])[0])
                    else:
                        self.run_HMM_realtime(np.dstack([self.array_to_save[:self.array_line+1, 0],
                                                         self.array_to_save[:self.array_line+1, 3]*1,
                                                         self.array_to_save[:self.array_line+1, 1]*1])[0])
            if self.array_line < len(self.array_to_save):
                self.array_line += 1
            else:
                print 'The array has reached max length (3000 entries which is around 30 seconds). Start a new trial'

    def ready_for_movements(self):
        # self.position_file = open(''.join([self.pkg_path, '/data/position_combined_0_15mps', '.log']), 'w')
        # self.position_file.write('Time(s) Pos(m) \n')
        start_move_time = rospy.Time.now()

        while self.continue_collecting and not rospy.is_shutdown():
            user_feedback = raw_input('Hit enter intialize pr2 config. Enter n to exit ')
            if user_feedback == 'n':
                self.continue_collecting = False
            else:
                self.initialize_pr2_config()

                user_feedback = raw_input('Hit enter to re-zero ft sensor and continue collecting data. Enter n to exit ')
                if user_feedback == 'n':
                    self.continue_collecting = False
                else:
                    self.calibrate_now = True
                    self.ft_sleeve_biased = False
                    rospy.sleep(0.5)
                    raw_input('Hit enter to start data collection')
                    self.start_recording_data(self.trial_number)
                    self.r_arm_pose_array_pub.publish(self.r_pose_trajectory)
                    raw_input('Hit enter to stop recording data')
                    self.stop_recording_data(self.trial_number)
                    fig = plt.figure(1)
                    ax1 = fig.add_subplot(111)
                    # ax1.set_title('Data for caught trial on PR2: Classified Correctly!', fontsize=20)
                    ax1.set_title('Data from most recent trial', fontsize=20)
                    ax1.set_xlabel('Time (s)', fontsize=20)
                    ax1.set_ylabel('Force (N)', fontsize=20)
                    ax1.set_xlim(0., 8.2)
                    ax1.set_ylim(-10., 1.0)
                    ax1.tick_params(axis='x', labelsize=20)
                    ax1.tick_params(axis='y', labelsize=20)

                    plot1 = ax1.plot(self.plotTime, self.plotX, label='Direction of Movement', linewidth=2)
                    plot2 = ax1.plot(self.plotTime, self.plotY, label='Direction of Gravity', linewidth=2)
                    # ax1.legend([plot1, plot2], ['Direction of Movement', ])
                    ax1.legend(loc='lower left', borderaxespad=0., fontsize=20)
                    # plt.plot(self.plotTime, self.plotX)
                    # plt.plot(self.plotTime, self.plotY)

                    plt.show()
        # final_time = rospy.Time.now().to_sec() - self.total_start_time.to_sec()
        # print 'Total elapsed time is:', final_time

    def run_HMM_realtime(self, test_data):
        # temp = np.dstack([(test_data[:, 1]), (test_data[:, 3])])[0]
        temp = test_data[:, 1:3]
        # print temp
        # orig = temp[0, 0]
        # for p in range(len(temp[:, 0])):
        #     temp[p, 0] -= orig
        self.plotTime = test_data[:, 0]
        self.plotX = temp[:, 1]
        self.plotY = temp[:, 0]

        # plt.figure(1)
        # plt.plot(temp[:,0],temp[:,1])
        # plt.show()
        # values = []
        max_value = -100000000000
        max_category = -1
        for modelid in xrange(len(self.categories)):
            final_ts_obj = ghmm.EmissionSequence(self.F, np.array(temp).flatten().tolist())
            pathobj = self.myHmms[modelid].viterbi(final_ts_obj)
            # pathobj = self.myHMMs.test(self.model_trained[modelid])
            # print 'Log likelihood for ', self.categories[modelid], ' category'
            # print pathobj[1]
            value = pathobj[1]
            if value > max_value:
                max_value = copy.copy(value)
                max_category = copy.copy(modelid)
        if max_category == -1:
            print 'No category matched in any way well. The HMMs could not perform classification!'
        else:
            if self.categories[max_category] == 'good/':
                hmm_prediction = 'good'
            elif self.categories[max_category] == 'missed/':
                hmm_prediction = 'missed'
            elif self.categories[max_category] == 'caught/':
                hmm_prediction = 'caught'
            else:
                print 'The categories do not match expected namings. Look into this!'
            print 'The HMMs are currently estimating/predicting that the task outcome is/will be '
            print hmm_prediction

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
        if not self.realtime_HMM:
            save_pickle(self.array_to_save, ''.join([self.pkg_path, '/data/pr2_test_ft_data/', 'ft_sleeve_', str(num), '.pkl']))
        self.trial_number += 1
        # self.array_to_save = np.zeros([1000, 5])
        # self.array_line = 0

if __name__ == "__main__":
    rospy.init_node('pr2_data_acquisition')

    import optparse
    p = optparse.OptionParser()

    p.add_option('--test_linear', action='store_true', dest='test_lin',
                 help='constant acceln and max vel for zenither.')
    p.add_option('--calib', action='store_true', dest='calib',
                 help='calibrate the zenither')
    p.add_option('--test_sine', action='store_true', dest='test_sine',
                 help='acceln and vel according to a sinusoid.')
    p.add_option('--sleeve', action='store_true', dest='sleeve',
                 help='Move actuator to pull sleeve on arm.')
    p.add_option('--sine_expt', action='store_true', dest='sine_expt',
                 help='run the sinusoid experiment.')
    p.add_option('--cmp_sine_pkl', action='store', dest='cmp_sine_pkl',
                 type='string', default='',
                 help='pkl saved by test_sine.')

    opt, args = p.parse_args()
    mode = 'autorun'
    # mode = None
    plot = True
    # plot = False
    num = 20
    vel = 0.1
    rc = PR2_FT_Data_Acquisition(trial_number=None, realtime_HMM=True)
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




