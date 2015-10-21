#!/usr/bin/python

import roslib
import rospy, rospkg
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Bool
import numpy as np

roslib.load_manifest('hrl_dressing')
roslib.load_manifest('zenither')
import zenither.zenither as zenither

roslib.load_manifest('hrl_lib')
from hrl_lib.util import save_pickle, load_pickle
from hrl_msgs.msg import FloatArray



class RigControl(object):
    def __init__(self, mode=None):
        # print 'Initializing Sleeve Rig'
        rospy.loginfo('Initializing Sleeve Rig')

        rospack = rospkg.RosPack()
        self.pkg_path = rospack.get_path('hrl_dressing')
        self.recording = False
        self.arm_file = None
        self.sleeve_file = None
        self.position_file = None
        self.start_record_time = 0.

        #self.array_to_save = np.zeros([1000, 5])
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
        self.puling = True

        self.ft_sleeve_biased = False
        self.ft_arm_biased = False

        self.forces_exceeded_threshold = False

        # print 'Initializing Zenither'
        rospy.loginfo('Initializing Zenither')

        self.zenither_pose = 0.

        self.z = zenither.Zenither(robot='test_rig')
        # self.z.broadcast()
        self.time_since_last_cb = rospy.Time.now()
        self.force_threshold_exceeded_pub = rospy.Publisher('/ft_threshold_exceeded', Bool, queue_size=1)
        self.force_threshold_exceeded_sub = rospy.Subscriber('/ft_threshold_exceeded', Bool, self.ft_threshold_cb)

        self.ft_sleeve_sub = rospy.Subscriber('/force_torque_sleeve', WrenchStamped, self.ft_sleeve_cb)

        # self.zenither_pose_sub = rospy.Subscriber('/zenither_pose', FloatArray, self.zenither_pose_cb)
        # self.ft_arm_sub = rospy.Subscriber('/force_torque_arm', WrenchStamped, self.ft_arm_cb)

        if mode.calib:
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
        rospy.spin()

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

        if (np.abs(msg.wrench.force.x-self.ft_sleeve_bias_x) > threshold) or \
                (np.abs(msg.wrench.force.y-self.ft_sleeve_bias_y) > threshold) or \
                (np.abs(msg.wrench.force.z-self.ft_sleeve_bias_z) > threshold):
                out = Bool()
                out.data = True
                self.force_threshold_exceeded_pub.publish(out)
        else:
            out = Bool()
            out.data = False
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
        if msg.data is True:
            rospy.loginfo('Force threshold exceeded! Stopping moving!')
            # print 'Force threshold exceeded! Stopping moving!'
            self.forces_exceeded_threshold = True
            self.z.estop()
            self.zenither_pose = self.z.get_position_meters()
            print 'Stopped at ', self.zenither_pose

    def repeated_movements(self):
        # self.position_file = open(''.join([self.pkg_path, '/data/position_combined_0_15mps', '.log']), 'w')
        # self.position_file.write('Time(s) Pos(m) \n')
        for i in xrange(3):
            self.pulling = False
            # self.ft_sleeve_biased = False
            rospy.sleep(0.6)
            reset_pos = 0.9
            reset_vel = 0.1
            reset_acc = 0.1
            # if not self.forces_exceeded_threshold or True:
            if True:
                self.zenither_move(reset_pos, reset_vel, reset_acc)
                # rospy.loginfo('Moving to initial position: ', reset_pos)
                print 'Moving to initial position: ', reset_pos
            else:
                continue
                # print 'Exceeded force threshold, will not move.'
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
                # if self.forces_exceeded_threshold:
                #     break
            self.zenither_pose = self.z.get_position_meters()
            print 'Current position is: ', self.zenither_pose
            rospy.sleep(2.0)
            test_pos = 0.05
            test_vel = 0.15
            test_acc = 1.0
            if True:
                self.ft_sleeve_biased = False
                rospy.sleep(0.5)
                print 'Moving to goal position: ', test_pos
                self.pulling = True
                self.start_recording_data(i)
                t = rospy.Time.now() - self.start_record_time
                self.zenither_pose = self.z.get_position_meters()
                # t = rospy.Time.now() - self.start_record_time
                self.position_file.write(''.join([str(t.to_sec()), ' %f \n' %
                                                  self.zenither_pose]))
                rospy.sleep(2.0)
                self.zenither_move(test_pos, test_vel, test_acc)
            else:
                continue
                # print 'Exceeded force threshold, will not move.'
            start_move_time = rospy.Time.now()
            # rospy.sleep(15.0)
            rospy.sleep(0.05*i)
            for j in xrange(8):
                t = rospy.Time.now() - self.start_record_time
                self.zenither_pose = self.z.get_position_meters()
                # t = rospy.Time.now() - self.start_record_time
                self.position_file.write(''.join([str(t.to_sec()), ' %f \n' %
                                                  self.zenither_pose]))
                rospy.sleep(1.0)
            self.zenither_pose = self.z.get_position_meters()
            pos = self.zenither_pose
            rospy.sleep(1.0)
            self.zenither_pose = self.z.get_position_meters()
            new_pos = self.zenither_pose
            while np.abs(new_pos-pos) > 0.005 and rospy.Time.now().to_sec()-start_move_time.to_sec() < 25.0:
                pos = new_pos
                rospy.sleep(1.0)
                self.zenither_pose = self.z.get_position_meters()
                new_pos = self.zenither_pose
            # rospy.loginfo('Final position is: ', self.z.get_position_meters())
            print 'Final position is: ', self.z.get_position_meters()
            self.z.estop()
            self.stop_recording_data(i)
            self.pulling = False
            rospy.sleep(1.0)
            # rospy.loginfo('Resetting...')
            print 'Resetting...'
        # rospy.loginfo('Movement complete!')
        self.position_file.close()
        print 'Movement complete!'

    def start_recording_data(self, num):
        #self.array_to_save = np.zeros([1000, 5])
        self.array_line = 0
        # self.arm_file = open(''.join([self.pkg_path, '/data/ft_arm_', str(num), '.log']), 'w')
        # self.sleeve_file = open(''.join([self.pkg_path, '/data/ft_sleeve_', str(num), '.log']), 'w')
        # self.arm_file.write('Time(us) Pos(m) force_x(N) force_y(N) force_z(N) torque_x(Nm) torque_y(Nm) torque_z(Nm) \n')
        # self.sleeve_file.write('Time(us) Pos(m) force_x(N) force_y(N) force_z(N) torque_x(Nm) torque_y(Nm) torque_z(Nm) \n')
        # self.position_file = open(''.join([self.pkg_path, '/data/position_', str(num), '.log']), 'w')
        self.start_record_time = rospy.Time.now()
        self.recording = True

    def stop_recording_data(self, num):
        self.recording = False
        # self.arm_file.close()
        # self.sleeve_file.close()
        # self.position_file.close()
        #save_pickle(self.array_to_save, ''.join([self.pkg_path, '/data/ft_sleeve_', str(num), '.pkl']))
        # self.array_to_save = np.zeros([1000, 4])
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

    rig_control = RigControl(opt)
    # rig_control.repeated_movements()

    rospy.spin()




