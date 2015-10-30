#!/usr/bin/python

import roslib
import rospy, rospkg
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
        if self.test_vel == 0.1:
            print ''.join([self.pkg_path, '/data/position_profiles/position_combined_0_1mps.pkl'])
            position_profile = load_pickle(''.join([self.pkg_path, '/data/position_profiles/position_combined_0_1mps.pkl']))
            print 'Position profile loaded!'
        elif self.test_vel == 0.15:
            position_profile = load_pickle(''.join([self.pkg_path, '/data/position_profiles/position_combined_0_15mps.pkl']))
            print ''.join([self.pkg_path, '/data/position_profiles/position_combined_0_15mps.pkl'])
            print 'Position profile loaded!'
        else:
            print 'There is no saved position profile for this velocity! Something has gone wrong!'
            return None
        for class_num in xrange(len(input_classes)):
            i = 0
            while os.path.isfile(''.join([self.pkg_path, '/data/', subject, '/',str(self.test_vel),'mps/', input_classes[class_num], '/ft_sleeve_', str(i), '.pkl'])):
                ft_threshold_was_exceeded = False
                print ''.join([self.pkg_path, '/data/', subject, '/', input_classes[class_num], '/ft_sleeve_', str(i), '.pkl'])
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
                while os.path.isfile(''.join([self.pkg_path, '/data/', subject, '/formatted/', str(self.test_vel),'mps/', output_classes[class_num], '/force_profile_', str(save_number), '.pkl'])):
                    save_number += 1
                print 'Saving with number', save_number
                save_pickle(current_data, ''.join([self.pkg_path, '/data/', subject, '/formatted/', str(self.test_vel),'mps/', output_classes[class_num], '/force_profile_', str(save_number), '.pkl']))
                i += 1
        print 'Done editing files!'
        if self.plot:
            self.plot_data(current_data)

    def load_and_plot(self, file_path):
        loaded_data = load_pickle(file_path)
        self.plot_data(loaded_data)
        print 'Plotted!'


    def plot_data(self, my_data):
        fig1 = plt.figure(1)
        ax1 = fig1.add_subplot(111)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Position (m)')
        ax1.set_xlim(0, 17)
        ax1.set_ylim(0, .86)
        X1 = my_data[:, 0]
        Y1 = my_data[:, 1]
        surf = ax1.scatter(X1, Y1, color="red", s=1, alpha=1)
        fig2 = plt.figure(2)
        ax2 = fig2.add_subplot(111)
        ax2.set_xlim(0, 17)
        ax2.set_ylim(-10.0, 3.0)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Force_x (N)')
        X2 = my_data[:, 0]
        Y2 = my_data[:, 2]
        surf = ax2.scatter(X2, Y2, s=1, alpha=1)
        fig3 = plt.figure(3)
        ax3 = fig3.add_subplot(111)
        ax3.set_xlim(0, .86)
        ax3.set_ylim(-10.0, 3.0)
        ax3.set_xlabel('Position (m)')
        ax3.set_ylabel('Force_x (N)')
        X3 = my_data[:, 1]
        Y3 = my_data[:, 2]
        surf = ax3.scatter(X3, Y3, s=1, alpha=1)
        fig4 = plt.figure(4)
        ax4 = fig4.add_subplot(111)
        ax4.set_xlim(0, .86)
        ax4.set_ylim(-10.0, 3.0)
        ax4.set_xlabel('Position (m)')
        ax4.set_ylabel('Force_z (N)')
        X4 = my_data[:, 1]
        Y4 = my_data[:, 4]
        surf = ax4.scatter(X4, Y4, s=1, alpha=1)
        plt.show()

    def start_recording_data(self, num):
        self.array_to_save = np.zeros([3000, 5])
        self.array_line = 0
        # self.arm_file = open(''.join([self.pkg_path, '/data/ft_arm_', str(num), '.log']), 'w')
        self.sleeve_file = open(''.join([self.pkg_path, '/data/',self.subject,'/',str(self.test_vel),'mps/',self.height,'/ft_sleeve_', str(num), '.log']), 'w')
        # self.arm_file.write('Time(us) Pos(m) force_x(N) force_y(N) force_z(N) torque_x(Nm) torque_y(Nm) torque_z(Nm) \n')
        self.sleeve_file.write('Time(us) Pos(m) force_x(N) force_y(N) force_z(N) torque_x(Nm) torque_y(Nm) torque_z(Nm) \n')
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
    # mode = 'autorun'
    mode = None
    plot = True
    # plot = False
    num = 5
    vel = 0.1
    subject_options = ['subject0', 'subject1', 'subject2', 'subject3', 'subject4', 'subject5', 'subject6', 'tapo_test_data','wenhao_test_data', 'test_subj']
    subject = subject_options[5]
    height_options = ['height0', 'height1', 'height2', 'height3', 'height4', 'height5']
    height = height_options[3]
    rc = RigControl(mode=mode, plot=plot, num=num, vel=vel, subj=subject, height=height)
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('hrl_dressing')

    save_number = 0

    # input_classification = ['183mm_height_missed_sleeve', '222mm_height_caught',
    #                         '222mm_height_missed_sleeve', '408mm_height_high', '325mm_height_good']
    # input_classification = ['missed', 'high', 'caught_forearm', 'caught']
    # output_classification = ['missed', 'high', 'caught', 'caught']
    # output_classification = ['missed', 'caught', 'missed', 'high', 'good']

    # file_name = ''.join([pkg_path, '/data/', subject, '/formatted/', str(vel),'mps/', output_classification[2], '/force_profile_', str(save_number), '.pkl'])
    # rc.load_and_plot(file_name)
    # rc.set_position_data_from_saved_movement_profile('wenhao_test_data/183mm_height_missed_sleeve')
    # rc.set_position_data_from_saved_movement_profile(subject, input_classification, output_classification)
    # rc.set_position_data_from_saved_movement_profile('wenhao_test_data/222mm_height_missed_sleeve')
    # rc.set_position_data_from_saved_movement_profile('wenhao_test_data/408mm_height_high')
    # rc.set_position_data_from_saved_movement_profile('wenhao_test_data/325mm_height_good')
    # rig_control.repeated_movements()

    rospy.spin()




