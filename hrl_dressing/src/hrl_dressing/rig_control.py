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



class RigControl(object):
    def __init__(self, mode='autorun', plot=False, num=None, vel=None, subj=None, height=None):
        # print 'Initializing Sleeve Rig'
        self.total_start_time = rospy.Time.now()
        rospy.loginfo('Initializing Sleeve Rig')

        self.mode = mode
        self.plot = plot

        if vel is None:
            self.test_vel = 0.1
        else:
            self.test_vel = vel
        if num is None:
            self.number_trials = 1
        else:
            self.number_trials = num
        if subj is None:
            self.subject = 'test_subject'
        else:
            self.subject = subj
        if height is None:
            self.height = 'height0'
        else:
            self.height = height
        self.position_profile_generation = False

        rospack = rospkg.RosPack()
        self.pkg_path = rospack.get_path('hrl_dressing')
        self.recording = False
        self.arm_file = None
        self.sleeve_file = None
        self.position_file = None
        self.start_record_time = 0.

        self.array_to_save = np.zeros([3000, 5])
        self.array_line = 0

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

        self.pulling_force_threshold = 10.0
        self.reset_force_threshold = 60.0
        self.pulling = True

        self.output_stuck = 0

        self.ft_sleeve_biased = False
        self.ft_arm_biased = False

        # self.forces_exceeded_threshold = False

        self.z = None

        self.zenither_pose = 0.

        if self.mode == 'autorun' or self.mode == 'calib':
            self.initialize_zenither(self.mode)
        else:
            self.data_path = '/home/ari/svn/robot1_data/usr/ari/data/hrl_dressing'

    def initialize_zenither(self, mode):
        # print 'Initializing Zenither'
        rospy.loginfo('Initializing Zenither')

        self.z = zenither.Zenither(robot='test_rig')

        self.time_since_last_cb = rospy.Time.now()
        self.force_threshold_exceeded_pub = rospy.Publisher('/ft_threshold_exceeded', Bool, queue_size=1)
        self.force_threshold_exceeded_sub = rospy.Subscriber('/ft_threshold_exceeded', Bool, self.ft_threshold_cb)

        self.ft_sleeve_sub = rospy.Subscriber('/force_torque_sleeve', WrenchStamped, self.ft_sleeve_cb)

        # self.ft_arm_sub = rospy.Subscriber('/force_torque_arm', WrenchStamped, self.ft_arm_cb)

        if mode == 'calib':
            self.zenither_calibrate()
            # print 'Calibrating Zenither'
            rospy.loginfo('Calibrating Zenither')

        if not self.z.calibrated:
            self.zenither_calibrate()
            # print 'Calibrating Zenither'
            rospy.loginfo('Calibrating Zenither')

        # print 'Zenither ready for action!'
        rospy.loginfo('Zenither ready for action!')

        self.repeated_movements()
        # rospy.spin()

    def zenither_calibrate(self):
        self.z.nadir()
        self.z.calibrated = True  # this is a hack!0.54293

        pos = self.z.get_position_meters()
        while True:
            rospy.sleep(0.5)
            new_pos = self.z.get_position_meters()
            if np.abs(new_pos-pos)<0.005:
                rospy.sleep(0.5)
                break
            pos = new_pos

        print 'Hit the end stop.'
        print 'Setting the origin.'
        self.z.estop()
        self.z.set_origin()
        print '__________________________'
        print 'Calibration Over'
        pos = self.z.get_position_meters()
        print 'Current position is: ', pos

    def zenither_move(self, pos, vel, acc):
        acceleration = self.z.limit_acceleration(acc)
        velocity = self.z.limit_velocity(vel)
        self.z.set_pos_absolute(pos)
        self.z.set_velocity(velocity)
        self.z.set_acceleration(acceleration)
        self.z.go()

    def zenither_pose_cb(self, msg):
        self.zenither_pose = msg.data[0]
        if self.recording:
            t = rospy.Time.now() - self.start_record_time
            self.position_file.write(''.join([str(t.to_sec()), ' %f \n' %
                                              self.zenither_pose]))

    def ft_sleeve_cb(self, msg):
        if not self.ft_sleeve_biased:
            rospy.sleep(0.5)
            self.time_since_last_cb = rospy.Time.now()
            self.ft_sleeve_bias_x = msg.wrench.force.x
            self.ft_sleeve_bias_y = msg.wrench.force.y
            self.ft_sleeve_bias_z = msg.wrench.force.z
            self.ft_sleeve_bias_t_x = msg.wrench.torque.x
            self.ft_sleeve_bias_t_y = msg.wrench.torque.y
            self.ft_sleeve_bias_t_z = msg.wrench.torque.z
            self.ft_sleeve_biased = True
        if rospy.Time.now().to_sec() - self.time_since_last_cb.to_sec() > 0.05:
            print 'The force-torque sensor callback is too slow. That is potentially very bad. Aborting everything!!!'
            self.z.estop()
            self.z = None
        self.time_since_last_cb = rospy.Time.now()
        x_force = msg.wrench.force.x-self.ft_sleeve_bias_x
        y_force = msg.wrench.force.y-self.ft_sleeve_bias_y
        z_force = msg.wrench.force.z-self.ft_sleeve_bias_z
        x_torque = msg.wrench.torque.x-self.ft_sleeve_bias_t_x
        y_torque = msg.wrench.torque.y-self.ft_sleeve_bias_t_y
        z_torque = msg.wrench.torque.z-self.ft_sleeve_bias_t_z
        # pos = self.z.get_position()
        if self.recording:
            t = rospy.Time.now() - self.start_record_time
            self.sleeve_file.write(''.join([str(t.to_sec()), ' %f %f %f %f %f %f %f \n' %
                                            (self.zenither_pose,
                                             x_force,
                                             y_force,
                                             z_force,
                                             x_torque,
                                             y_torque,
                                             z_torque)]))
            self.array_to_save[self.array_line] = [t.to_sec(), self.zenither_pose, x_force, y_force, z_force]
            self.array_line += 1
        if self.pulling:
            threshold = self.pulling_force_threshold
        else:
            threshold = self.reset_force_threshold

        if (np.abs(x_force) > threshold) or (np.abs(y_force) > threshold) or (np.abs(z_force) > threshold):
                out = Bool()
                out.data = True
                self.force_threshold_exceeded_pub.publish(out)
        else:
            out = Bool()
            out.data = False
            self.output_stuck = 0
            self.force_threshold_exceeded_pub.publish(out)

    # def ft_arm_cb(self, msg):
    #     if not self.ft_arm_biased:
    #         rospy.sleep(0.5)
    #         self.ft_arm_bias_x = msg.wrench.force.x
    #         self.ft_arm_bias_y = msg.wrench.force.y
    #         self.ft_arm_bias_z = msg.wrench.force.z
    #         self.ft_arm_bias_t_x = msg.wrench.torque.x
    #         self.ft_arm_bias_t_y = msg.wrench.torque.y
    #         self.ft_arm_bias_t_z = msg.wrench.torque.z
    #         self.ft_arm_biased = True
    #     x_force = msg.wrench.force.x-self.ft_arm_bias_x
    #     y_force = msg.wrench.force.y-self.ft_arm_bias_y
    #     z_force = msg.wrench.force.z-self.ft_arm_bias_z
    #     x_torque = msg.wrench.torque.x-self.ft_arm_bias_t_x
    #     y_torque = msg.wrench.torque.y-self.ft_arm_bias_t_y
    #     z_torque = msg.wrench.torque.z-self.ft_arm_bias_t_z
    #     if self.recording:
    #         t = rospy.Time.now() - self.start_record_time
    #         self.arm_file.write(''.join([str(t.to_sec()), ' %f %f %f %f %f %f \n' %
    #                                      (x_force,
    #                                       y_force,
    #                                       z_force,
    #                                       x_torque,
    #                                       y_torque,
    #                                       z_torque)]))
    #     if (np.abs(msg.wrench.force.x-self.ft_arm_bias_x) > self.force_threshold) or \
    #             (np.abs(msg.wrench.force.y-self.ft_arm_bias_y) > self.force_threshold) or \
    #             (np.abs(msg.wrench.force.z-self.ft_arm_bias_z) > self.force_threshold):
    #         out = Bool()
    #         out.data = True
    #         self.force_threshold_exceeded_pub.publish(out)

    def ft_threshold_cb(self, msg):
        if msg.data:
            # rospy.loginfo('Force threshold exceeded! Stopping moving!')
            # print 'Force threshold exceeded! Stopping moving!'
            # self.forces_exceeded_threshold = True
            self.z.estop()
            # if self.output_stuck < 3:
            #     self.zenither_pose = self.z.get_position_meters()
            #     print 'Stopped at ', self.zenither_pose
            #     self.output_stuck += 1

    def repeated_movements(self):
        # self.position_file = open(''.join([self.pkg_path, '/data/position_combined_0_15mps', '.log']), 'w')
        # self.position_file.write('Time(s) Pos(m) \n')
        self.pulling = False
        # self.ft_sleeve_biased = False
        rospy.sleep(0.6)
        reset_pos = 0.9
        reset_vel = 0.1
        reset_acc = 0.1
        self.zenither_move(reset_pos, reset_vel, reset_acc)
        print 'Moving to initial position: ', reset_pos
        pos = self.z.get_position_meters()
        print 'Current position is: ', pos
        start_move_time = rospy.Time.now()
        rospy.sleep(1.0)
        new_pos = self.z.get_position_meters()
        # print 'Current position is: ', new_pos
        while np.abs(new_pos-pos) > 0.005 and rospy.Time.now().to_sec()-start_move_time.to_sec() < 20.0:
            pos = self.z.get_position_meters()
            rospy.sleep(0.5)
            new_pos = self.z.get_position_meters()
            # print 'Current position is: ', new_pos
        self.zenither_pose = self.z.get_position_meters()
        print 'Current position is: ', self.zenither_pose
        rospy.sleep(2.0)
        self.test_vel = 0.1
        for i in xrange(self.number_trials):
            if rospy.is_shutdown():
                break
            else:
                test_pos = 0.05
                test_vel = self.test_vel
                test_acc = 1.0
                self.ft_sleeve_biased = False
                rospy.sleep(0.5)
                print 'Moving to goal position: ', test_pos
                self.pulling = True
                self.start_recording_data(i)
                t = rospy.Time.now() - self.start_record_time
                self.zenither_pose = self.z.get_position_meters()
                # t = rospy.Time.now() - self.start_record_time
                if self.position_profile_generation:
                    self.position_file.write(''.join([str(t.to_sec()), ' %f \n' %
                                                      self.zenither_pose]))
                rospy.sleep(4.0)
                self.zenither_move(test_pos, test_vel, test_acc)

                start_move_time = rospy.Time.now()
                # rospy.sleep(15.0)
                if self.position_profile_generation:
                    rospy.sleep(0.05*i)
                    for j in xrange(8):
                        t = rospy.Time.now() - self.start_record_time
                        self.zenither_pose = self.z.get_position_meters()
                        # t = rospy.Time.now() - self.start_record_time
                        self.position_file.write(''.join([str(t.to_sec()), ' %f \n' %
                                                          self.zenither_pose]))
                        rospy.sleep(1.0)
                else:
                    if self.test_vel==0.1:
                        rospy.sleep(9.2)
                    elif self.test_vel==0.15:
                        rospy.sleep(6.2)
                self.zenither_pose = self.z.get_position_meters()
                pos = self.zenither_pose
                rospy.sleep(1.0)
                self.zenither_pose = self.z.get_position_meters()
                new_pos = self.zenither_pose
                while np.abs(new_pos-pos) > 0.005 and rospy.Time.now().to_sec()-start_move_time.to_sec() < 15.0:
                    pos = new_pos
                    rospy.sleep(1.0)
                    self.zenither_pose = self.z.get_position_meters()
                    new_pos = self.zenither_pose
                # rospy.loginfo('Final position is: ', self.z.get_position_meters())
                print 'Final position is: ', self.z.get_position_meters()
                self.z.estop()
                self.stop_recording_data(i)
                self.pulling = False
                rospy.sleep(1.5)
                # rospy.loginfo('Resetting...')
                print 'Finished trial ', i+1, 'at velocity', self.test_vel
                print 'Resetting...'
                self.pulling = False
                # self.ft_sleeve_biased = False
                rospy.sleep(0.6)
                reset_pos = 0.9
                reset_vel = 0.1
                reset_acc = 0.1
                self.zenither_move(reset_pos, reset_vel, reset_acc)
                print 'Moving to initial position: ', reset_pos
                pos = self.z.get_position_meters()
                print 'Current position is: ', pos
                start_move_time = rospy.Time.now()
                rospy.sleep(1.0)
                new_pos = self.z.get_position_meters()
                # print 'Current position is: ', new_pos
                while np.abs(new_pos-pos) > 0.005 and rospy.Time.now().to_sec()-start_move_time.to_sec() < 20.0:
                    pos = self.z.get_position_meters()
                    rospy.sleep(0.5)
                    new_pos = self.z.get_position_meters()
                    # print 'Current position is: ', new_pos
                self.zenither_pose = self.z.get_position_meters()
                print 'Current position is: ', self.zenither_pose
                rospy.sleep(4.0)
        self.test_vel = 0.15
        for i in xrange(self.number_trials):
            if rospy.is_shutdown():
                break
            else:
                test_pos = 0.05
                test_vel = self.test_vel
                test_acc = 1.0
                self.ft_sleeve_biased = False
                rospy.sleep(0.5)
                print 'Moving to goal position: ', test_pos
                self.pulling = True
                self.start_recording_data(i)
                t = rospy.Time.now() - self.start_record_time
                self.zenither_pose = self.z.get_position_meters()
                # t = rospy.Time.now() - self.start_record_time
                if self.position_profile_generation:
                    self.position_file.write(''.join([str(t.to_sec()), ' %f \n' %
                                                      self.zenither_pose]))
                rospy.sleep(4.0)
                self.zenither_move(test_pos, test_vel, test_acc)

                start_move_time = rospy.Time.now()
                # rospy.sleep(15.0)
                if self.position_profile_generation:
                    rospy.sleep(0.05*i)
                    for j in xrange(8):
                        t = rospy.Time.now() - self.start_record_time
                        self.zenither_pose = self.z.get_position_meters()
                        # t = rospy.Time.now() - self.start_record_time
                        self.position_file.write(''.join([str(t.to_sec()), ' %f \n' %
                                                          self.zenither_pose]))
                        rospy.sleep(1.0)
                else:
                    if self.test_vel==0.1:
                        rospy.sleep(9.2)
                    elif self.test_vel==0.15:
                        rospy.sleep(6.2)
                self.zenither_pose = self.z.get_position_meters()
                pos = self.zenither_pose
                rospy.sleep(1.0)
                self.zenither_pose = self.z.get_position_meters()
                new_pos = self.zenither_pose
                while np.abs(new_pos-pos) > 0.005 and rospy.Time.now().to_sec()-start_move_time.to_sec() < 15.0:
                    pos = new_pos
                    rospy.sleep(1.0)
                    self.zenither_pose = self.z.get_position_meters()
                    new_pos = self.zenither_pose
                # rospy.loginfo('Final position is: ', self.z.get_position_meters())
                print 'Final position is: ', self.z.get_position_meters()
                self.z.estop()
                self.stop_recording_data(i)
                self.pulling = False
                rospy.sleep(1.5)
                # rospy.loginfo('Resetting...')
                print 'Finished trial ', i+1, 'at velocity', self.test_vel
                print 'Resetting...'
                self.pulling = False
                # self.ft_sleeve_biased = False
                rospy.sleep(0.6)
                reset_pos = 0.9
                reset_vel = 0.1
                reset_acc = 0.1
                self.zenither_move(reset_pos, reset_vel, reset_acc)
                print 'Moving to initial position: ', reset_pos
                pos = self.z.get_position_meters()
                print 'Current position is: ', pos
                start_move_time = rospy.Time.now()
                rospy.sleep(1.0)
                new_pos = self.z.get_position_meters()
                # print 'Current position is: ', new_pos
                while np.abs(new_pos-pos) > 0.005 and rospy.Time.now().to_sec()-start_move_time.to_sec() < 20.0:
                    pos = self.z.get_position_meters()
                    rospy.sleep(0.5)
                    new_pos = self.z.get_position_meters()
                    # print 'Current position is: ', new_pos
                self.zenither_pose = self.z.get_position_meters()
                print 'Current position is: ', self.zenither_pose
                rospy.sleep(4.0)
        # rospy.loginfo('Movement complete!')
        if self.position_profile_generation:
            self.position_file.close()
        print 'Movement complete!'
        final_time = rospy.Time.now().to_sec() - self.total_start_time.to_sec()
        print 'Total elapsed time is:', final_time

    def set_position_data_from_saved_movement_profile(self, subject, input_classes, output_classes):
        print 'Now editing files to insert position data.'
        position_profile = None
        for vel in [0.1, 0.15]:
            if vel == 0.1:
                print ''.join([self.pkg_path, '/data/position_profiles/position_combined_0_1mps.pkl'])
                position_profile = load_pickle(''.join([self.pkg_path, '/data/position_profiles/position_combined_0_1mps.pkl']))
                print 'Position profile loaded!'
            elif vel == 0.15:
                position_profile = load_pickle(''.join([self.pkg_path, '/data/position_profiles/position_combined_0_15mps.pkl']))
                print ''.join([self.pkg_path, '/data/position_profiles/position_combined_0_15mps.pkl'])
                print 'Position profile loaded!'
            else:
                print 'There is no saved position profile for this velocity! Something has gone wrong!'
                return None
            for class_num in xrange(len(input_classes)):
                i = 0
                while os.path.isfile(''.join([self.pkg_path, '/data/', subject, '/',str(vel),'mps/', input_classes[class_num], '/ft_sleeve_', str(i), '.pkl'])):
                    ft_threshold_was_exceeded = False
                    print ''.join([self.pkg_path, '/data/', subject, '/', input_classes[class_num], '/ft_sleeve_', str(i), '.pkl'])
                    # current_data = np.array([map(float,line.strip().split()) for line in open(''.join([self.pkg_path, '/data/', subject, '/', input_classes[class_num], '/ft_sleeve_', str(i), '.log']))])
                    current_data = load_pickle(''.join([self.pkg_path, '/data/', subject, '/', input_classes[class_num], '/ft_sleeve_', str(i), '.pkl']))

                    # if np.max(current_data[:, 2]) >= 10. or np.max(current_data[:, 3]) >= 10. \
                    #         or np.max(current_data[:, 4]) >= 10.:
                    #     ft_threshold_was_exceeded = True

                    for j in current_data:
                        j[2] = -j[2]
                        j[3] = -j[3]
                        j[4] = -j[4]
                        if j[0] < 0.5:
                            j[1] = 0
                        else:
                            if np.max(np.abs(j[2:])) > 10. and not ft_threshold_was_exceeded:
                                time_of_stop = j[0]
                                ft_threshold_was_exceeded = True
                                for k in xrange(len(position_profile)-1):
                                    if position_profile[k, 0] <= j[0] < position_profile[k+1, 0]:
                                        position_of_stop = position_profile[k, 1] + \
                                                           (position_profile[k+1, 1] - position_profile[k, 1]) * \
                                                           (j[0]-position_profile[k, 0])/(position_profile[k+1, 0] - position_profile[k, 0])
                            if ft_threshold_was_exceeded:
                                j[1] = position_of_stop
                            else:
                                for k in xrange(len(position_profile)-1):
                                    if position_profile[k, 0] <= j[0] < position_profile[k+1, 0]:
                                        new_position = position_profile[k, 1] + \
                                                       (position_profile[k+1, 1] - position_profile[k, 1]) * \
                                                       (j[0]-position_profile[k, 0])/(position_profile[k+1, 0] - position_profile[k, 0])
                                j[1] = new_position
                    save_number = 0
                    while os.path.isfile(''.join([self.pkg_path, '/data/', subject, '/formatted/', str(vel),'mps/', output_classes[class_num], '/force_profile_', str(save_number), '.pkl'])):
                        save_number += 1
                    print 'Saving with number', save_number
                    save_pickle(current_data, ''.join([self.pkg_path, '/data/', subject, '/formatted/', str(vel),'mps/', output_classes[class_num], '/force_profile_', str(save_number), '.pkl']))
                    i += 1
        print 'Done editing files!'
        # if self.plot:
        #     self.plot_data(current_data)

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
        self.array_to_save = np.zeros([3000, 5])
        self.array_line = 0
        # self.arm_file = open(''.join([self.pkg_path, '/data/ft_arm_', str(num), '.log']), 'w')
        self.sleeve_file = open(''.join([self.pkg_path, '/data/',self.subject,'/',str(self.test_vel),'mps/',self.height,'/ft_sleeve_', str(num), '.log']), 'w')
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
        save_pickle(self.array_to_save, ''.join([self.pkg_path, '/data/',self.subject,'/',str(self.test_vel),'mps/',self.height,'/ft_sleeve_', str(num), '.pkl']))
        # self.array_to_save = np.zeros([1000, 5])
        # self.array_line = 0

if __name__ == "__main__":
    rospy.init_node('rig_control')

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
    num = 3
    vel = 0.1
    subject_options = ['subject0', 'subject1', 'subject2', 'subject3', 'subject4', 'subject5', 'subject6', 'subject7', 'subject8', 'subject9', 'subject10', 'subject11', 'subject12',
                       'with_sleeve_no_arm', 'no_sleeve_no_arm', 'moved_rig_onto_drawers', 'moved_rig_back', 'testing_level', 'tapo_test_data','wenhao_test_data', 'test_subj']
    subject = subject_options[17]
    height_options = ['height0', 'height1', 'height2', 'height3', 'height4', 'height5']
    height = height_options[0]
    rc = RigControl(mode=mode, plot=plot, num=num, vel=vel, subj=subject, height=height)
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

    rospy.spin()




