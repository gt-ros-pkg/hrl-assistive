#!/usr/bin/python

import roslib
import rospy, rospkg, rosparam
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Bool
import numpy as np
import math as m
import os.path
import ghmm
import copy

roslib.load_manifest('hrl_dressing')
import tf, argparse
import threading
import hrl_lib.util as utils

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
import tf.transformations as tft

from hrl_haptic_manipulation_in_clutter_srvs.srv import EnableHapticMPC

from helper_functions import createBMatrix, Bmat_to_pos_quat

from geometry_msgs.msg import PoseArray, Pose, PoseStamped, Point, Quaternion, PoseWithCovarianceStamped, Twist
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from hrl_base_selection.dressing_configuration_optimization_multithread import DressingSimulationProcess

import hrl_dressing.controller as controller


# Class that runs the performance of the dressing task on a person using a PR2.
# Run on the PR2 to inform it of what to do. Run on another machine to visualize the person's pose.
class TOORAD_Dressing_PR2(object):
    def __init__(self, machine='desktop', participant=0, trial=0, enable_realtime_HMM=False,
                 visually_estimate_arm_pose=False, adjust_arm_pose_visually=False):
        # machine is whether this is run on the PR2 or desktop for visualization.
        self.machine = machine
        # trial number sets what number to use to save data.
        # self.trial_number = trial
        # realtime_HMM sets whether or not to also use a realtime HMM to estimate the outcome of the dressing task.

        # Make a thread lock so nothing silly happens during multithreading from callbacks
        self.frame_lock = threading.RLock()

        # Load the configurations for dressing.
        rospack = rospkg.RosPack()
        self.pkg_path = rospack.get_path('hrl_base_selection')
        self.save_file_path = self.pkg_path + '/data/'
        self.save_file_name_final_output = 'arm_config_final_output.log'
        configs = self.load_configs(self.save_file_path+self.save_file_name_final_output)

        # If this is on desktop for visualization, run the visualization
        if self.machine == 'desktop':
            # Set up TOORAD process that includes DART simulation environment
            self.toorad = DressingSimulationProcess(visualize=True)
            print 'Starting visualization of the desired configuration for the dressing task. First select the arm ' \
                  'to be dressed'
            arm = raw_input('\nEnter R (r) for right arm (should be done first. Enter L (l) for left arm (should be '
                            'done second). Otherwise ends.\n')
            while not arm.upper() == 'Q' or not arm.upper() == 'N' or not rospy.is_shutdown():
                if len(arm) == 0:
                    return
                elif arm.upper()[0] == 'R':
                    h_config = configs[0][0]
                    r_config = configs[0][1]
                    self.toorad.set_robot_arm('rightarm')
                    self.toorad.set_human_arm('rightarm')
                elif arm.upper()[0] == 'L':
                    h_config = configs[1][0]
                    r_config = configs[1][1]
                    self.toorad.set_robot_arm('rightarm')
                    self.toorad.set_human_arm('leftarm')
                else:
                    return

                # Set up the simulator for the configuration loaded
                self.toorad.set_human_model_dof_dart([0, 0, 0, 0], self.toorad.human_opposite_arm)
                self.toorad.set_human_model_dof_dart(h_config, self.toorad.human_arm)
                self.toorad.set_pr2_model_dof_dart(r_config)

                # Calculate the trajectories based on the configuration in the simulator
                s_arm = self.toorad.human_arm.split('a')[0]
                self.toorad.generate_dressing_trajectory(s_arm, visualize=True)

                arm = raw_input('\nEnter R (r) for right arm (should be done first. Enter L (l) for left arm (should be '
                                'done second). Otherwise ends.\n')

        # If this is on the PR2, ask for arm to dress (left or right). If wrong input received, ends.
        elif self.machine.upper() == 'PR2':
            # Set up TOORAD process that includes DART simulation environment
            self.toorad = DressingSimulationProcess(visualize=False)
            # This flag is set when the system is ready to start. It then waits for the user to put their hand under
            # the robot end effector
            self.ready_to_start = False

            # This flag is used to check when the subtask is completed so it can ask about starting the next task.
            self.subtask_complete = False

            self.moving = False
            self.hz = 20.
            self.control_rate = rospy.Rate(self.hz)

            self.record_data = True

            # Start up the controller for the PR2
            self.control = controller.Controller('base_footprint')

            # Setup the publishers and subscribers necessary for dressing and the HMMs.
            self.initialize_pr2_comms(enable_realtime_HMM)

            arm = ''
            while not arm.upper() == 'Y' or not rospy.is_shutdown():
                print 'Will moving PR2 to initial configuration for the desired dressing task. First select the arm ' \
                      'to be dressed'
                arm = raw_input('\nEnter R (r) for right arm (should be done first). Enter L (l) for left arm (should be '
                                'done second). Otherwise ends.\n')
                if len(arm) == 0:
                    return
                elif arm.upper()[0] == 'R':
                    self.arm = 'rightarm'
                    h_config = configs[0][0]
                    r_config = configs[0][1]
                    self.toorad.set_robot_arm('rightarm')
                    self.toorad.set_human_arm('rightarm')
                elif arm.upper()[0] == 'L':
                    self.arm = 'leftarm'
                    h_config = configs[1][0]
                    r_config = configs[1][1]
                    self.toorad.set_robot_arm('rightarm')
                    self.toorad.set_human_arm('leftarm')
                else:
                    return

                # Set up the simulator for the configuration loaded
                self.toorad.set_human_model_dof_dart([0, 0, 0, 0], self.toorad.human_opposite_arm)
                self.toorad.set_human_model_dof_dart(h_config, self.toorad.human_arm)
                self.toorad.set_pr2_model_dof_dart(r_config)

                # Calculate the trajectories based on the configuration in the simulator
                s_arm = self.toorad.human_arm.split('a')[0]
                origin_B_goals, \
                origin_B_forearm_pointed_down_arm, \
                origin_B_upperarm_pointed_down_shoulder, \
                origin_B_hand, \
                origin_B_wrist, \
                origin_B_traj_start, \
                origin_B_traj_forearm_end, \
                origin_B_traj_upper_end, \
                origin_B_traj_final_end, \
                angle_from_horizontal, \
                forearm_B_upper_arm, traj_path, all_sols = self.toorad.generate_dressing_trajectory(s_arm, visualize=False)

                if visually_estimate_arm_pose:
                    # Will estimate the arm pose prior to beginning dressing using vision and will execute the
                    # trajectory selected by TOORAD on that arm pose. Will still use capacitance and force to modify
                    # trajectory as needed.
                    print '\nPlease adopt the requested arm pose for a moment while the robot estimates your arm pose.\n'
                    inp = raw_input('\nEnter Y (y) to estimate arm pose. Otherwise ends.\n')
                    if len(inp) == 0:
                        return
                    elif inp.upper()[0] == 'Y':
                        # Grab arm pose from pose estimator
                        self.estimate_arm_pose()
                    else:
                        return
                else:
                    # Will assume the arm pose matches the request arm pose and will execute the trajectory selected by
                    # TOORAD on that arm pose. Will still use capacitance and force to modify trajectory as needed.

                    if adjust_arm_pose_visually:
                        print '\nPlease adopt the requested arm pose for a moment while the robot estimates your ' \
                              'arm pose. See the pose estimation for how the pose should be adjusted\n'
                        inp = raw_input('\nEnter Y (y) when complete. Otherwise ends.\n')
                        if len(inp) == 0:
                            return
                        elif inp.upper()[0] == 'Y':
                            # Grab arm pose from pose estimator
                            pass
                        else:
                            return

                    pr2_B_goals = []
                    for goal in origin_B_goals:
                        pr2_B_goals.append(self.toorad.origin_B_pr2.I*goal)

                    # Find first pose from TOORAD trajectory and move robot arms to start configuration.
                    goal_i = int(traj_path[0].split('-')[0])
                    sol_i = int(traj_path[0].split('-')[1])
                    self.control.setJointGuesses(self, rightGuess=all_sols[goal_i][sol_i], leftGuess=None)
                    pr2_B_goal = pr2_B_goals[goal_i]
                    pos, quat = Bmat_to_pos_quat(pr2_B_goal)
                    inp = 'Y'
                    while inp.upper() == 'Y' or inp.upper() == 'N' or inp.upper() == 'Q' or not rospy.is_shutdown():
                        inp = raw_input(
                            '\nEnter Y (y) to move arms to start configuration. Will attempt movement for up to 10 '
                            'seconds. N or Q ends. Otherwise continues without movement.\n')
                        if len(inp) == 0:
                            pass
                        elif inp.upper()[0] == 'Y':
                            self.control.moveGripperTo(pos, quaternion=quat, useInitGuess=True, rightArm=True, timeout=10.)
                        elif inp.upper()[0] == 'N' or inp.upper()[0] == 'Q':
                            return
                        else:
                            pass

                inp = raw_input('\nEnter Y (y) to zero sensors. Otherwise ends.\n')
                if len(inp) == 0:
                    return
                elif inp.upper()[0] == 'Y':
                    # Zero sensors
                    self.zero_sensor_data()
                else:
                    return

                inp = ''
                while not inp.upper() == 'Y' or not rospy.is_shutdown():
                    self.subtask_complete = False
                    self.ready_to_start = False
                    self.moving = False
                    inp = raw_input('\nEnter Y (y) to start the task. Enter R (r) to re-zero sensors. Enter N (n) or Q (q) to quit.'
                                    'Otherwise will stay here.\n')
                    if len(inp) == 0:
                        pass
                    elif inp.upper()[0] == 'Y':
                        # Set triggers such that the task can be started by the person's arm under the capacitive sensor
                        self.ready_to_start = True
                        # Dressing task is now active.
                        print 'Dressing task for', self.arm, 'is now active. Movement will start when participant moves' \
                                                             'hand under the robot end effector'
                        if visually_estimate_arm_pose:
                            self.begin_dressing_trajectory_from_vision(visually_estimate_arm_pose)
                        else:
                            self.begin_dressing_trajectory_no_vision(trial, participant, pr2_B_goals, traj_path, all_sols)
                    elif inp.upper()[0] == 'R':
                        # Re-zero sensors.
                        self.zero_sensor_data()
                    elif inp.upper()[0] == 'N' or inp.upper()[0] == 'Q':
                        return


                while not self.subtask_complete and not rospy.is_shutdown():
                    rospy.sleep(1)
                print 'Task complete for', self.arm
        else:
            print 'I do not recognize the machine type. Try again!'
            return

    def begin_dressing_trajectory_from_vision(self):
        # TODO Execute trajectory following the pose estimated using vision
        pass

    def begin_dressing_trajectory_no_vision(self, trial_number, participant_number, p_B_g, path, sols):
        save_file_name = self.save_file_path + 'dressing_data/participant'+str(participant_number)+'/' + 'p'+str(participant_number)+'_t'+str(trial_number)+'.log'
        open(save_file_name, 'w').close()

        # Waits to actually begin moving until the participant puts their hand under the capacitive sensor.
        while not self.moving and not rospy.is_shutdown():
            rospy.sleep(0.1)

        self.ready_to_start = False
        self.difference = 0
        self.startTime = rospy.get_time()
        self.lastMoveTime = rospy.get_time()
        self.last_capacitive_reading = rospy.get_time()
        self.last_ft_reading = rospy.get_time()

        prev_error = np.zeros(3)
        prev_distance_error = 0.

        # Set gains for the controller
        Kp = 0.05
        Kd = 0.02

        self.plotTime = []
        self.plotX = []
        self.plotY = []

        # Equilibrium point adjustment is used to alter the trajectory based on input from the capacitive sensor and
        # force-torque sensor.
        equilibrium_point_adjustment = np.zeros(3)
        force_adjustment = np.zeros(3)

        current_goal = 0

        # Move!
        while self.moving and not rospy.is_shutdown():
            if current_goal == len(path):
                'Movement has finished!'
                break
            current_time = rospy.get_time()
            if current_time - self.last_capacitive_reading > 0.1 or current_time - self.last_ft_reading > 0.1:
                'Stopping movement because too much time has passed since the last data reading from the capacitive ' \
                'sensor or the force-torque sensor.'
                break
            print 'Time since last control loop execution:', current_time - self.lastMoveTime
            (current_position, current_orientation) = self.control.getGripperPosition(rightArm=True)
            pr2_B_current_gripper = createBMatrix(current_position, current_orientation)
            current_distance_error = (0.05 - self.distance_to_arm) * np.array(pr2_B_current_gripper)[0:3, 2]
            if np.linalg.norm(self.force) < 4.0:
                force_adjustment *= 0.95
            else:
                force_adjustment += self.force/10.*0.05

            distance_adjustment = Kp * current_distance_error + Kd * (current_distance_error - prev_distance_error)

            equilibrium_point_adjustment += force_adjustment + distance_adjustment

            target_position, target_orientation = Bmat_to_pos_quat(p_B_g[current_goal])
            target_position += equilibrium_point_adjustment
            angle_temp, axis_temp, discard_point = tft.rotation_from_matrix(pr2_B_current_gripper.I*p_B_g)
            if np.linalg.norm(target_position - current_position) < 0.02 and np.abs(angle_temp) <= m.radians(5.):
                # Current goal has been reached. Move on to next goal.
                current_goal += 1
                target_position, target_orientation = Bmat_to_pos_quat(p_B_g[current_goal])
                target_position += equilibrium_point_adjustment
            if 3 < current_goal < 8 and self.distance_to_arm > 10.:
                distance_adjustment += np.array(pr2_B_current_gripper)[0:3, 1] * (-0.5) + 0.2*np.array(pr2_B_current_gripper)[0:3, 2]
                current_goal = 8
                target_position, target_orientation = Bmat_to_pos_quat(p_B_g[current_goal])
                target_position += equilibrium_point_adjustment
            if 10 < current_goal < 16 and self.distance_to_arm > 10.:
                distance_adjustment += np.array(pr2_B_current_gripper)[0:3, 1] * (-0.5) + 0.2 * np.array(pr2_B_current_gripper)[0:3, 2]
                current_goal = 16
                target_position, target_orientation = Bmat_to_pos_quat(p_B_g[current_goal])
                target_position += equilibrium_point_adjustment
            if current_goal > 4:
                self.run_HMM_realtime(np.dstack([self.time_series_forces[:self.array_line + 1, 0] * 1,
                                                 self.time_series_forces[:self.array_line + 1, 3] * 1,
                                                 self.time_series_forces[:self.array_line + 1, 1] * 1])[0])
            goal_i = int(path[current_goal].split('-')[0])
            if not goal_i == current_goal:
                print 'Something has gone wrong. The current goal index and goal index from the path differ'
                print 'goal_i:', goal_i
                print 'current_goal:', current_goal
                self.moving = False
                break
            sol_i = int(path[0].split('-')[1])
            target_posture = sols[goal_i][sol_i]
            self.control.setJointGuesses(self, rightGuess=target_posture, leftGuess=None)
            error = target_position - current_position

            # if np.linalg.norm(error) < 0.02:
            #      = error/np.linalg.norm(error)*0.02
            # elif np.linalg.norm(error) > 0.1:
            #     x = error / np.linalg.norm(error) * 0.1
            # else:
            #     x = error
            # dist = self.update_ee_setpoint()

            x = Kp * error + Kd * (error - prev_error)
            prev_error = error
            prev_distance_error = current_distance_error

            self.control.moveGripperTo(current_position + x, quaternion=target_orientation, useInitGuess=True, rightArm=True, timeout=1./self.hz)

            if self.record_data:
                with open(save_file_name, 'a') as myfile:
                    myfile.write(str("{:.5f}".format(current_time))
                                 + ',' + str("{:.5f}".format(current_position[0]))
                                 + ',' + str("{:.5f}".format(current_position[1]))
                                 + ',' + str("{:.5f}".format(current_position[2]))
                                 + ',' + str("{:.5f}".format(self.current_force[0]))
                                 + ',' + str("{:.5f}".format(self.current_force[1]))
                                 + ',' + str("{:.5f}".format(self.current_force[2]))
                                 + ',' + str("{:.5f}".format(self.current_capacitance))
                                 + ',' + str("{:.5f}".format(self.self.distance_to_arm))
                                 + ',' + self.hmm_prediction
                                 + '\n')

            self.control_rate.sleep()

        isSuccess = raw_input('Success? Is the arm and shoulder both in the sleeve of the gown? [y/n] ').upper() == 'Y'
        with open(save_file_name, 'a') as myfile:
            myfile.write('success,'+ str(isSuccess) + '\n')

        if len(self.plotTime) > 0:
            fig = plt.figure(1)
            ax1 = fig.add_subplot(111)
            # ax1.set_title('Data for caught trial on PR2: Classified Correctly!', fontsize=20)
            ax1.set_title('Data from f/t sensor, estimated outcome: '+self.hmm_prediction, fontsize=20)
            ax1.set_xlabel('Time (s)', fontsize=20)
            ax1.set_ylabel('Force (N)', fontsize=20)
            ax1.set_xlim(0., 8.2)
            ax1.set_ylim(-10., 1.0)
            ax1.tick_params(axis='X', labelsize=20)
            ax1.tick_params(axis='Z', labelsize=20)

            plot1 = ax1.plot(self.plotTime, self.plotX, label='Direction of Movement', linewidth=2)
            plot2 = ax1.plot(self.plotTime, self.plotY, label='Direction of Gravity', linewidth=2)
            # ax1.legend([plot1, plot2], ['Direction of Movement', ])
            ax1.legend(loc='lower left', borderaxespad=0., fontsize=20)
            # plt.plot(self.plotTime, self.plotX)
            # plt.plot(self.plotTime, self.plotY)

            plt.show()

        # Movement is complete!
        self.moving = False
        self.subtask_complete = True

    def estimate_arm_pose(self):
        # TODO Estimate arm pose
        pass

    def load_configs(self, file_name):
        saved_configs = [line.rstrip('\n').split(',')
                            for line in open(file_name)]

        for j in xrange(len(saved_configs)):
            saved_configs[j] = [float(i) for i in saved_configs[j]]
        saved_configs = np.array(saved_configs)
        # saved_configs = np.array([x for x in saved_configs if int(x[0]) == subtask_number])
        out_configs = []
        out_configs.append([saved_configs[0][1:5], saved_configs[0][6:10]])
        out_configs.append([saved_configs[1][1:5], saved_configs[1][6:10]])
        return out_configs

    def initialize_pr2_comms(self, setup_HMMs):
        if setup_HMMs:
            self.HMM_last_run_time = rospy.Time.now()
            self.myHmms = []
            self.F = ghmm.Float()
            self.myHmms = []
            self.categories = ['missed', 'good', 'caught']
            self.time_series_forces = np.zeros([20000, 4])
            for i in xrange(len(self.categories)):
                self.myHmms.append(ghmm.HMMOpen(
                    self.data_path+'trained_hmms/clusteredsample_params1/straightarm_clusteredsample_10state_model' +
                    self.categories[i]+'.xml'))
            rospy.loginfo('HMMs loaded! Ready to classify!')
        self.ft_sleeve_sub = rospy.Subscriber('/force_torque_pr2_sleeve', WrenchStamped, self.ft_sleeve_cb)
        rospy.loginfo('FT sensor subscriber initialized')
        rospy.Subscriber('/capDressing/capacitance', Float64MultiArray, self.capacitive_sensor_cb)
        rospy.loginfo('Capacitive sensor subscriber initialized')

        # TODO: Initialize estimated skeleton subscriber
        rospy.loginfo('Estimated human pose (from vision) subscriber initialized')

    def ft_sleeve_cb(self, msg):
        with self.frame_lock:
            # Collect force and torque data
            if self.force_baseline is not None:
                self.force = np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z]) - self.force_baseline
                self.torque = np.array([msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z]) - self.torque_baseline
                if np.linalg.norm(self.force) >= 10.0:
                    self.moving = False
                if self.moving:
                    self.time_series_forces[self.array_line] = [msg.header.stamp.to_sec(), self.force[0],
                                                                self.force[1], self.force[2]]
                    if self.array_line < len(self.array_to_save):
                        self.array_line += 1
                    else:
                        print 'Exceeded number of data lines being used for time series evalatuion by HMMs'

            elif self.zeroing:
                # Calculate the baseline value for the force-torque sensor
                self.force = np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z])
                self.torque = np.array([msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z])
                self.force_baseline_values.append(self.force)
                self.torque_baseline_values.append(self.torque)
                if len(self.forceList) > 50:
                    force_baseline = np.mean(self.forceList, axis=0)
                    torque_baseline = np.mean(self.torqueList, axis=0)

    def capacitive_sensor_cb(self, msg):
        with self.frame_lock:
            raw_capacitance = msg.data
            # TODO Ask Zackory about this. Is capacitance always len=1? If not, later things in this code do not work (max).
            if len(raw_capacitance) == 1:
                # Use just first capacitance value
                raw_capacitance = raw_capacitance[0]

            # Calculate the baseline value for the capacitive sensor
            if self.capacitance_baseline is None and self.zeroing:
                self.capacitanceBaselineValues.append(raw_capacitance)
                if len(self.capacitanceBaselineValues) > 50:
                    self.capacitance_baseline = np.mean(self.capacitanceBaselineValues, axis=0)
            elif self.capacitance_baseline is not None:
                capacitance = max(-(raw_capacitance - self.capacitance_baseline), -3.5)
                # capacitance = max(capacitance, -3.5)
                self.distance_to_arm = self.estimate_distance_to_arm(capacitance)
                if not self.moving and self.ready_to_start and self.distance_to_arm <= 0.045:
                    self.ready_to_start = False
                    self.moving = True

    def estimate_distance_to_arm(self, cap):
        return 0.8438 / (cap + 4.681)

    def zero_sensor_data(self):
        self.capacitance_baseline, self.force_baseline
        self.capacitance_baseline = None
        self.force_baseline = None
        self.torque_baseline = None
        self.capacitance_baseline_values = []
        self.force_baseline_values = []
        self.torque_baseline_values = []
        self.zeroing = True
        self.time_series_forces = np.zeros([20000, 4])
        self.array_line = 0
        print 'Zeroing data'
        while self.capacitance_baseline is None or self.force_baseline is None:
            rospy.sleep(0.1)
        self.zeroing = False
        print 'Capacitance and force readings have been zeroed'
        rospy.sleep(1.0)

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
            self.hmm_prediction = 'None'
        else:
            if self.categories[max_category] == 'good':
                self.hmm_prediction = 'good'
            elif self.categories[max_category] == 'missed':
                self.hmm_prediction = 'missed'
            elif self.categories[max_category] == 'caught':
                self.hmm_prediction = 'caught'
            else:
                print 'The categories do not match expected namings. Look into this!'

if __name__ == '__main__':
    rospy.init_node('dressing_execution')

    import optparse

    p = optparse.OptionParser()

    p.add_option('-p', action='store', dest='participant', required=True, type=int, help='Participant number')
    p.add_option('-t', action='store', dest='trial', required=True, type=int, help='Trial Number')

    opt, args = p.parse_args()


    toorad_dressing = TOORAD_Dressing_PR2(participant=args.participant, trial=args.participant,
                                          enable_realtime_HMM=False,
                                          visually_estimate_arm_pose=False, adjust_arm_pose_visually=False
                                          )


class PR2_FT_Data_Acquisition(object):
    def __init__(self, plot=False, trial_number=None, realtime_HMM=False):
        # print 'Initializing Sleeve Rig'
        self.total_start_time = rospy.Time.now()
        rospy.loginfo('Initializing PR2 F/T Data Acquisition')

        self.zeroed_netft_pub = rospy.Publisher('/netft_gravity_zeroing/wrench_zeroed', WrenchStamped, queue_size=10)

        self.listener = tf.TransformListener()
        self.broadcaster = tf.TransformBroadcaster()

        self.categories = ['missed/', 'good/', 'caught/']

        self.reference_frame = 'base_footprint'

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

        self.array_to_save = np.zeros([20000, 4])
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
        print 'starting thread'
        t = threading.Thread(target=self.run_tf_broadcaster)
        t.start()
        print 'after thread'
        pr2_config = load_pickle(self.base_selection_pkg_path+'/data/dressing_rightarm_best_pr2_config.pkl')[0]
        human_arm_config =  load_pickle(self.base_selection_pkg_path+'/data/dressing_rightarm_best_trajectory_and_arm_config.pkl')[0]

        sc = ScoreGeneratorDressingwithPhysx(robot_arm='rightarm', human_arm='rightarm', visualize=False, standard_mode=False)

        sc.robot_arm = 'rightarm'
        sc.human_arm = 'rightarm'
        sc.setup_dart()
        arm = sc.human_arm.split('a')[0]
        sc.set_human_model_dof_dart(human_arm_config, 'rightarm')

        self.gripper_B_tool = np.matrix([[0., -1., 0., 0.03],
                                         [1., 0., 0., 0.0],
                                         [0., 0., 1., -0.05],
                                         [0., 0., 0., 1.]])

        sc.goals, \
        origin_B_forearm_pointed_down_arm, \
        origin_B_upperarm_pointed_down_shoulder, \
        self.origin_B_traj_start, \
        origin_B_traj_forearm_end, \
        origin_B_traj_upper_end, \
        origin_B_traj_final_end, \
        angle_from_horizontal, \
        self.forearm_B_upper_arm = sc.find_reference_coordinate_frames_and_goals(arm)
        sc.set_goals()

        self.origin_B_pr2 = np.matrix([[m.cos(pr2_config[2]), -m.sin(pr2_config[2]), 0., pr2_config[0]],
                                  [m.sin(pr2_config[2]), m.cos(pr2_config[2]), 0., pr2_config[1]],
                                  [0., 0., 1., 0.],
                                  [0., 0., 0., 1.]])
        print 'origin_B_pr2 = \n', self.origin_B_pr2
        print 'pr2_B_origin = \n', self.origin_B_pr2.I
        print 'pr2_B_shoulder = \n', self.origin_B_pr2.I * origin_B_upperarm_pointed_down_shoulder
        self.traj_data = [[], []]
        path_distance = 1.
        path_waypoints = np.arange(0.15, path_distance + path_distance * 0.01, (path_distance - 0.15) / 2.)
        for goal in path_waypoints:
            traj_start_B_traj_waypoint = np.matrix(np.eye(4))
            traj_start_B_traj_waypoint[0, 3] = goal
            self.traj_data[0].append(self.origin_B_pr2.I * self.origin_B_traj_start * traj_start_B_traj_waypoint*self.gripper_B_tool.I)
        # for i in xrange(len(sc.origin_B_grasps)):
        #     if i < 3:
        #         j = 0
        #     elif i < 6:
        #         j = 1
        #     else:
        #         j = 2
            # origin_B_grasp in sc.origin_B_grasps:
            # self.traj_data[j].append(self.origin_B_pr2.I * np.matrix(sc.origin_B_grasps[i]))

        # self.traj_data = load_pickle(self.base_selection_pkg_path+'/data/saved_results/large_search_space/pr2_grasps.pkl')
        # self.pr2_config = load_pickle(self.base_selection_pkg_path+'/data/saved_results/large_search_space/best_pr2_config.pkl')
        self.pr2_torso_lift_msg = SingleJointPositionActionGoal()
        self.pr2_torso_lift_msg.goal.position = pr2_config[3]

        self.r_arm_pub = rospy.Publisher('/right_arm/haptic_mpc/joint_trajectory', JointTrajectory, queue_size=1)
        self.r_arm_pose_array_pub = rospy.Publisher('/right_arm/haptic_mpc/goal_pose_array', PoseArray, queue_size=1)
        self.r_arm_pose_pub = rospy.Publisher('/right_arm/haptic_mpc/goal_pose', PoseStamped, queue_size=1)
        self.rviz_trajectory_pub = rospy.Publisher('dressing_trajectory_visualization', PoseArray, queue_size=1, latch=True)
        self.pr2_lift_pub = rospy.Publisher('torso_controller/position_joint_action/goal',
                                            SingleJointPositionActionGoal, queue_size=10)
        print 'Publishers are ready! Now checking for mpc service'
        rospy.wait_for_service('/right_arm/haptic_mpc/enable_mpc')
        self.mpc_enabled_service = rospy.ServiceProxy("/right_arm/haptic_mpc/enable_mpc", EnableHapticMPC)
        print 'Found MPC service. Now defining start pose'
        # self.start_pose = None
        self.define_start_posture()
        self.start_pose = self.define_start_pose()
        print 'Now setting trajectory in MPC'
        self.r_pose_trajectory = self.define_pose_trajectory(0)
        print 'Now ready for movement!'
        self.ready_for_movements()

    def define_pose_trajectory(self, task_part):
        print 'Defining trajectory for subtask ', task_part
        if task_part == 0:
            basefootprint_B_poses = self.traj_data[task_part]
        elif task_part == 1:
            self.traj_data[task_part] = []
            path_distance = 0.4
            path_waypoints = np.arange(0, path_distance + path_distance * 0.01, (path_distance) / 2.)
            current_pose_B_traj_upper_start = np.matrix(np.eye(4))
            current_pose_B_traj_upper_start[0, 3] = -0.05
            current_pose_B_traj_upper_start[2, 3] = -0.05
            current_reference_pose = copy.copy(self.origin_B_pr2.I * self.origin_B_traj_start)
            current_position, current_orientation = self.listener.lookupTransform(self.reference_frame,
                                                                            '/r_gripper_tool_frame',
                                                                            rospy.Time(0))
            current_reference_pose[0:3, 3] = np.reshape(current_position, [3, 1])
            pr2_B_forearm_traj_end = current_reference_pose * current_pose_B_traj_upper_start
            for goal in path_waypoints:
                traj_start_B_traj_waypoint = np.matrix(np.eye(4))
                traj_start_B_traj_waypoint[0, 3] = goal
                self.traj_data[task_part].append(pr2_B_forearm_traj_end *
                                                 self.forearm_B_upper_arm *
                                                 traj_start_B_traj_waypoint *
                                                 self.gripper_B_tool.I)
            basefootprint_B_poses = self.traj_data[task_part]

        basefootprint_B_poses = self.traj_data[task_part]
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

    def define_start_posture(self):
        # basefootprint_B_start_pose = self.traj_data[0]
        # pos_out, ori_out = Bmat_to_pos_quat(basefootprint_B_start_pose)
        #
        # startpose = PoseStamped()
        # # startpose.header.stamp = rospy.Time.now()
        # startpose.header.stamp = rospy.Time(0)
        # startpose.header.frame_id = 'base_footprint'
        #
        # startpose.pose.position.x = pos_out[0]
        # startpose.pose.position.y = pos_out[1]
        # startpose.pose.position.z = pos_out[2]
        # startpose.pose.orientation.x = ori_out[0]
        # startpose.pose.orientation.y = ori_out[1]
        # startpose.pose.orientation.z = ori_out[2]
        # startpose.pose.orientation.w = ori_out[3]
        r_reset_traj_point = JointTrajectoryPoint()
        r_reset_traj_point.positions = [-0.88242068, -0.25032293,  0.5, - 0.67920343, -2.31494573, -1.56943708,
                                        2.35989478]

        r_reset_traj_point.velocities = [0.0] * 7
        r_reset_traj_point.accelerations = [0.0] * 7
        r_reset_traj_point.effort = [100.0] * 7
        r_reset_traj_point.time_from_start = rospy.Duration(5)
        self.r_reset_traj = JointTrajectory()
        self.r_reset_traj.joint_names = ['r_shoulder_pan_joint',
                                         'r_shoulder_lift_joint',
                                         'r_upper_arm_roll_joint',
                                         'r_elbow_flex_joint',
                                         'r_forearm_roll_joint',
                                         'r_wrist_flex_joint',
                                         'r_wrist_roll_joint']
        self.r_reset_traj.points.append(r_reset_traj_point)
        # [-0.88242068, - 0.25032293,  0.5 - 0.67920343, - 2.31494573, - 1.56943708,  2.35989478]
        # return startpose

    def define_start_pose(self):
        basefootprint_B_start_pose = self.traj_data[0][0]
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
        # r_reset_traj_point = JointTrajectoryPoint()
        # r_reset_traj_point.positions = [-0.88242068, -0.25032293, 0.5, - 0.67920343, -2.31494573, -1.56943708,
        #                                 2.35989478 - m.pi]
        #
        # r_reset_traj_point.velocities = [0.0] * 7
        # r_reset_traj_point.accelerations = [0.0] * 7
        # r_reset_traj_point.effort = [100.0] * 7
        # r_reset_traj_point.time_from_start = rospy.Duration(5)
        # self.r_reset_traj = JointTrajectory()
        # self.r_reset_traj.joint_names = ['r_shoulder_pan_joint',
        #                                  'r_shoulder_lift_joint',
        #                                  'r_upper_arm_roll_joint',
        #                                  'r_elbow_flex_joint',
        #                                  'r_forearm_roll_joint',
        #                                  'r_wrist_flex_joint',
        #                                  'r_wrist_roll_joint']
        # self.r_reset_traj.points.append(r_reset_traj_point)
        # # [-0.88242068, - 0.25032293,  0.5 - 0.67920343, - 2.31494573, - 1.56943708,  2.35989478]
        return startpose

    def initialize_pr2_config(self):
        resp = self.mpc_enabled_service('enabled')
        self.pr2_lift_pub.publish(self.pr2_torso_lift_msg)
        self.r_arm_pub.publish(self.r_reset_traj)

    def initialize_pr2_pose(self):
        resp = self.mpc_enabled_service('enabled')
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
        if rospy.Time.now().to_sec() - self.time_since_last_cb.to_sec() > 0.1:
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
            # self.array_to_save[self.array_line] = [t.to_sec(), -x_force, -y_force, -z_force]
            # print 'Forces: ', x_force, ', ', y_force, ', ', z_force
            self.wrench_zeroed = WrenchStamped()
            self.wrench_zeroed.wrench.force.x = x_force*0.5
            self.wrench_zeroed.wrench.force.y = y_force*0.8
            self.wrench_zeroed.wrench.force.z = z_force*1.0
            self.wrench_zeroed.wrench.torque.x = 0.
            self.wrench_zeroed.wrench.torque.y = 0.
            self.wrench_zeroed.wrench.torque.z = 0.
            self.wrench_zeroed.header.stamp = msg.header.stamp
            self.wrench_zeroed.header.frame_id = "right_sleeve_ft_frame"
            self.zeroed_netft_pub.publish(self.wrench_zeroed)
            # print 'Force magnitude: ', np.linalg.norm([x_force, y_force, z_force])
            if np.linalg.norm([x_force, y_force, z_force]) >= 1000.:
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
        else:
            self.wrench_zeroed = WrenchStamped()
            self.wrench_zeroed.wrench.force.x = 0.
            self.wrench_zeroed.wrench.force.y = 0.
            self.wrench_zeroed.wrench.force.z = 0.
            self.wrench_zeroed.wrench.torque.x = 0.
            self.wrench_zeroed.wrench.torque.y = 0.
            self.wrench_zeroed.wrench.torque.z = 0.
            self.wrench_zeroed.header.stamp = msg.header.stamp
            self.wrench_zeroed.header.frame_id = "right_sleeve_ft_frame"
            self.zeroed_netft_pub.publish(self.wrench_zeroed)

    def ready_for_movements(self):

        # self.position_file = open(''.join([self.pkg_path, '/data/position_combined_0_15mps', '.log']), 'w')
        # self.position_file.write('Time(s) Pos(m) \n')
        capstart_move_time = rospy.Time.now()
        # print self.traj_data
        print 'len traj data: ', len(self.traj_data)

        while self.continue_collecting and not rospy.is_shutdown():
            user_feedback = raw_input('Hit enter intialize pr2 config. Enter n to exit ')
            if user_feedback == 'n':
                self.continue_collecting = False
            else:
                self.initialize_pr2_config()
                user_feedback = raw_input('Hit enter to switch to pose control. Enter n to exit ')
                if user_feedback == 'n':
                    self.continue_collecting = False
                else:
                    # self.r_reset_traj = JointTrajectory()
                    # self.initialize_pr2_config()
                    # user_feedback = raw_input('Hit enter to switch to pose control')
                    self.initialize_pr2_pose()
                    user_feedback = raw_input('Hit to re-zero ft sensor.')
                    self.calibrate_now = True
                    self.ft_sleeve_biased = False
                    rospy.sleep(0.5)
                    raw_input('Hit enter to start data collection. Will start after 5 seconds.')
                    rospy.sleep(5.0)
                    self.start_recording_data(self.trial_number)
                    subtask = 0
                    full_task_complete = False
                    task_check_rate = rospy.Rate(10.)
                    while not rospy.is_shutdown() and not full_task_complete:
                        self.r_pose_trajectory = self.define_pose_trajectory(subtask)
                        subtask_complete = False
                        # self.r_arm_pose_array_pub.publish(self.r_pose_trajectory)

                        prev_position, prev_orientation = self.listener.lookupTransform(self.reference_frame,
                                                                                              '/r_gripper_tool_frame',
                                                                                              rospy.Time(0))
                        print '\n\nStarting a new subtask!!!! \n \n '
                        while not rospy.is_shutdown() and not subtask_complete:
                            progress_timer = rospy.Time.now()
                            pose_i = 0
                            while not rospy.is_shutdown() and pose_i < len(self.r_pose_trajectory.poses):
                                # print pose_i
                                pose_msg = PoseStamped()
                                pose_msg.pose = self.r_pose_trajectory.poses[pose_i]
                                pose_msg.header.stamp = rospy.Time.now()
                                pose_msg.header.frame_id = self.reference_frame
                                self.r_arm_pose_pub.publish(pose_msg)
                                current_position, current_orientation = self.listener.lookupTransform(self.reference_frame, '/r_gripper_tool_frame', rospy.Time(0))

                                current_subtask_goal_position = np.array([self.r_pose_trajectory.poses[pose_i].position.x,
                                                                          self.r_pose_trajectory.poses[pose_i].position.y,
                                                                          self.r_pose_trajectory.poses[pose_i].position.z])
                                current_subtask_goal_orientation = np.array([self.r_pose_trajectory.poses[pose_i].orientation.x,
                                                                             self.r_pose_trajectory.poses[pose_i].orientation.y,
                                                                             self.r_pose_trajectory.poses[pose_i].orientation.z,
                                                                             self.r_pose_trajectory.poses[pose_i].orientation.w])
                                # current_subtask_goal_position = np.array([self.r_pose_trajectory.poses[-1].position.x,
                                #                                           self.r_pose_trajectory.poses[-1].position.y,
                                #                                           self.r_pose_trajectory.poses[-1].position.z])
                                # current_subtask_goal_orientation = np.array([self.r_pose_trajectory.poses[-1].orientation.x,
                                #                                              self.r_pose_trajectory.poses[-1].orientation.y,
                                #                                              self.r_pose_trajectory.poses[-1].orientation.z,
                                #                                              self.r_pose_trajectory.poses[-1].orientation.w])
                                # print 'distance to goal', np.linalg.norm(np.array(current_position) - current_subtask_goal_position)
                                # print 'angle to goal', utils.quat_angle(current_orientation, current_subtask_goal_orientation)
                                progress_elapsed_time = rospy.Time.now() - progress_timer
                                if np.linalg.norm(np.array(current_position) - current_subtask_goal_position) < 0.02 and\
                                                utils.quat_angle(current_orientation, current_subtask_goal_orientation) < 5.0:
                                    print 'Close enough to this goal ', pose_i
                                    if pose_i < len(self.r_pose_trajectory.poses) - 1:
                                        pose_i += 1
                                        progress_timer = rospy.Time.now()
                                        continue
                                    else:
                                        pose_i += 1
                                        subtask_complete = True
                                        subtask += 1
                                        break
                                elif progress_elapsed_time.to_sec() > 3.0:

                                    if np.linalg.norm(np.array(current_position) - prev_position) < 0.02 and \
                                                    utils.quat_angle(current_orientation, prev_orientation) < 10.0:
                                        print 'Timed out for this goal ', pose_i
                                        if pose_i < len(self.r_pose_trajectory.poses) - 1:
                                            pose_i += 1
                                            progress_timer = rospy.Time.now()
                                            force_adjusted_movement = np.array([self.wrench_zeroed.wrench.force.x,
                                                                                self.wrench_zeroed.wrench.force.y,
                                                                                self.wrench_zeroed.wrench.force.z])
                                            if np.linalg.norm(force_adjusted_movement) > 2.0:
                                                force_adjusted_movement = np.min(np.linalg.norm(force_adjusted_movement)*0.01, 0.05) * \
                                                                          force_adjusted_movement / np.linalg.norm(force_adjusted_movement)
                                                # pose_msg = PoseStamped()
                                                # pose_msg.pose = self.r_pose_trajectory.poses[pose_i]
                                                pose_msg.header.stamp = rospy.Time.now()
                                                # pose_msg.header.frame_id = self.reference_frame
                                                pose_msg.pose.position.x = force_adjusted_movement[0]
                                                pose_msg.pose.position.y = force_adjusted_movement[1]
                                                pose_msg.pose.position.z = force_adjusted_movement[2]
                                                print 'Making a movement to decrease forces before continuing'
                                                self.r_arm_pose_pub.publish(pose_msg)
                                                rospy.sleep(1.5)
                                            continue
                                        else:
                                            pose_i += 1
                                            subtask_complete = True
                                            subtask += 1
                                            force_adjusted_movement = np.array([self.wrench_zeroed.wrench.force.x,
                                                                                self.wrench_zeroed.wrench.force.y,
                                                                                self.wrench_zeroed.wrench.force.z])
                                            if np.linalg.norm(force_adjusted_movement) > 2.0:
                                                force_adjusted_movement = np.min(
                                                    np.linalg.norm(force_adjusted_movement) * 0.01, 0.05) * \
                                                                          force_adjusted_movement / np.linalg.norm(
                                                    force_adjusted_movement)
                                                # pose_msg = PoseStamped()
                                                # pose_msg.pose = self.r_pose_trajectory.poses[pose_i]
                                                pose_msg.header.stamp = rospy.Time.now()
                                                # pose_msg.header.frame_id = self.reference_frame
                                                pose_msg.pose.position.x = force_adjusted_movement[0]
                                                pose_msg.pose.position.y = force_adjusted_movement[1]
                                                pose_msg.pose.position.z = force_adjusted_movement[2]
                                                print 'Making a movement to decrease forces before continuing'
                                                self.r_arm_pose_pub.publish(pose_msg)
                                                rospy.sleep(1.5)
                                            break
                                    else:
                                        progress_timer = rospy.Time.now()
                                        prev_position = current_position
                                        prev_orientation = current_orientation
                                task_check_rate.sleep()
                        if subtask >= len(self.traj_data):
                            full_task_complete = True
                            print 'Task is entirely complete!'
                    stopPose = PoseStamped()
                    self.r_arm_pose_pub.publish(stopPose)
                    resp = self.mpc_enabled_service('disabled')
                    self.recording = False

                    raw_input('Hit enter to stop recording data')
                    self.stop_recording_data(self.trial_number)
                    continue
                    return
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
            # print 'The HMMs are currently estimating/predicting that the task outcome is/will be '
            # print hmm_prediction

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
        self.array_to_save = np.zeros([20000, 4])
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

    def run_tf_broadcaster(self):
        print 'Setting up TF broadcaster for the FT sensor'
        rate = rospy.Rate(10.)
        while not rospy.is_shutdown():
            self.broadcaster.sendTransform((0.05, 0., -0.065),
                                           tf.transformations.quaternion_from_euler(m.radians(180.), 0, m.radians(90.),'rxyz'),
                                           rospy.Time.now(),
                                           "right_sleeve_ft_frame",
                                           "r_gripper_tool_frame")
            rate.sleep()


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




