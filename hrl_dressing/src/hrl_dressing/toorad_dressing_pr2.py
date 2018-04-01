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

from hrl_base_selection.helper_functions import createBMatrix, Bmat_to_pos_quat

from geometry_msgs.msg import PoseArray, Pose, PoseStamped, Point, Quaternion, PoseWithCovarianceStamped, Twist
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint






# Class that runs the performance of the dressing task on a person using a PR2.
# Run on the PR2 to inform it of what to do. Run on another machine to visualize the person's pose.
class TOORAD_Dressing_PR2(object):
    def __init__(self, machine='desktop', visualize=False, participant=0, trial=0, model='50-percentile-wheelchair',
                 enable_realtime_HMM=False,
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
        self.save_file_path = self.pkg_path + '/data/' + 'saved_results/dressing/'+model+'/'
        self.save_file_name_final_output = 'arm_configs_final_output.log'
        self.trajectory_pickle_file_name = 'trajectory_data_a'
        # configs = self.load_configs(self.save_file_path+self.save_file_name_final_output)

        self.enable_realtime_HMM = enable_realtime_HMM

        self.robot_arm = 'rightarm'
        self.robot_opposite_arm = 'leftarm'

        if self.machine == 'generate_trajectory':
            from hrl_base_selection.dressing_configuration_optimization_multithread_dart import DressingSimulationProcess
            # Generates the trajectory and saves all relevant info to a pickle file.
            pydart.init()
            print('pydart initialization OK')
            self.toorad = DressingSimulationProcess(visualize=False)
            print 'Starting to generate the dressing trajectories from the saved configurations. This will then ' \
                  'be used by the PR2 for execution or by another machine for visualization.'
            save_data = []
            for arm in ['right', 'left']:
                if arm.upper()[0] == 'R':
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
                    print 'HOW DID THIS GO WRONG?? '
                    return

                # print 'objective_function_fine =', self.toorad.objective_function_fine(h_config)

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
                forearm_B_upper_arm, traj_path, all_sols = self.toorad.generate_dressing_trajectory(s_arm,
                                                                                                    visualize=False)
                print arm
                print not traj_path
                print traj_path

                # print 'robot positions',self.toorad.robot.positions()
                # print 'robot q', self.toorad.robot.q
                # print 'human q', self.toorad.human.q
                # print 'human pose'
                # print self.toorad.human.q['j_bicep_' + arm + '_x']
                # print self.toorad.human.q['j_bicep_' + arm + '_y']
                # print self.toorad.human.q['j_bicep_' + arm + '_z']
                # print self.toorad.human.q['j_forearm_' + arm + '_1']
                # print 'robot pose'
                # print self.toorad.robot.q['rootJoint_pos_x']
                # print self.toorad.robot.q['rootJoint_pos_y']
                # print self.toorad.robot.q['rootJoint_pos_z']
                # print self.toorad.robot.q['rootJoint_rot_z']
                # print self.toorad.robot.q['torso_lift_joint']

                save_data.append([arm, origin_B_goals,
                                  origin_B_forearm_pointed_down_arm,
                                  origin_B_upperarm_pointed_down_shoulder,
                                  origin_B_hand,
                                  origin_B_wrist,
                                  origin_B_traj_start,
                                  origin_B_traj_forearm_end,
                                  origin_B_traj_upper_end,
                                  origin_B_traj_final_end,
                                  forearm_B_upper_arm, traj_path, all_sols])

            save_pickle(save_data, self.save_file_path+self.trajectory_pickle_file_name)
            print 'File saved successfully!'

        elif self.machine == 'desktop':
            import pydart2 as pydart
            import openravepy as op
            from openravepy.misc import InitOpenRAVELogging

            # If this is on desktop for visualization, run the visualization
            # Set up TOORAD process that includes DART simulation environment
            pydart.init()
            print('pydart initialization OK')

            self.setup_dart(filename='fullbody_50percentile_capsule.skel', visualize=True)

            # rospy.sleep(20)
            arm = raw_input('\nEnter R (r) for right arm (should be done first. Enter L (l) for left arm (should be '
                            'done second). Otherwise ends.\n')
            while (not arm.upper() == 'Q' and not arm.upper() == 'N') and not rospy.is_shutdown():
                if len(arm) == 0:
                    return
                elif arm.upper()[0] == 'R':
                    subtask = 0
                    h_arm = 'rightarm'
                    h_opposite_arm = 'leftarm'

                    # h_config = configs[0][0]
                    # r_config = configs[0][1]
                    # self.toorad.set_robot_arm('rightarm')
                    # self.toorad.set_human_arm('rightarm')
                elif arm.upper()[0] == 'L':
                    subtask=1
                    h_arm = 'leftarm'
                    h_opposite_arm = 'rightarm'
                    # h_config = configs[1][0]
                    # r_config = configs[1][1]
                    # self.toorad.set_robot_arm('rightarm')
                    # self.toorad.set_human_arm('leftarm')
                else:
                    return
                print self.save_file_path + self.trajectory_pickle_file_name+str(subtask)+'.pkl'
                loaded_data = load_pickle(self.save_file_path + self.trajectory_pickle_file_name+str(subtask)+'.pkl')

                params,\
                z,\
                pr2_params,\
                pr2_B_goals,\
                pr2_B_forearm_pointed_down_arm,\
                pr2_B_upperarm_pointed_down_shoulder,\
                pr2_B_hand,\
                pr2_B_wrist,\
                pr2_B_traj_start,\
                pr2_B_traj_forearm_end,\
                pr2_B_traj_upper_end,\
                pr2_B_traj_final_end,\
                traj_path = loaded_data
                print 'Trajectory data loaded succesfully!'

                print 'pr2_params', pr2_params

                # self.toorad = DressingSimulationProcess(visualize=False)
                print 'Starting visualization of the desired configuration for the dressing task. First select the arm ' \
                      'to be dressed'

                x = pr2_params[0]
                y = pr2_params[1]
                th = pr2_params[2]
                z = pr2_params[3]

                v = self.robot.positions()
                v['rootJoint_pos_x'] = x
                v['rootJoint_pos_y'] = y
                v['rootJoint_pos_z'] = 0.
                v['rootJoint_rot_z'] = th
                self.dart_world.displace_gown()
                v['torso_lift_joint'] = pr2_params[3]
                self.robot.set_positions(v)
                rospy.sleep(0.1)

                self.set_human_model_dof_dart(params, h_arm)
                self.set_human_model_dof_dart([0., 0., 0., 0.], h_opposite_arm)

                # goal_i = int(traj_path[0].split('-')[0])
                # sol_i = int(traj_path[0].split('-')[1])
                # prev_sol = np.array(all_sols[goal_i][sol_i])
                # print 'Solution being visualized:'
                cont = raw_input('\nEnter Q (q) or N (n) to stop visualization of this task. '
                                 'Otherwise visualizes again.\n')
                if len(cont) == 0:
                    cont = ' '
                while not cont.upper()[0] == 'Q' and not cont.upper()[0] == 'N' and not rospy.is_shutdown():
                    count = 0
                    while count < 2 and not rospy.is_shutdown():
                        count += 1
                        # print 'robot pose'
                        # print self.robot.q['rootJoint_pos_x']
                        # print self.robot.q['rootJoint_pos_y']
                        # print self.robot.q['rootJoint_pos_z']
                        # print self.robot.q['rootJoint_rot_z']
                        # print self.robot.q['torso_lift_joint']
                        # print traj_path
                        for num, path_step in enumerate(traj_path):

                            # print path_step
                            # print all_sols[goal_i][sol_i][0]
                            # print all_sols[goal_i][sol_i][1]
                            # print all_sols[goal_i][sol_i][2]
                            # print all_sols[goal_i][sol_i][3]
                            # print all_sols[goal_i][sol_i][4]
                            # print all_sols[goal_i][sol_i][5]
                            # print all_sols[goal_i][sol_i][6]

                            v = self.robot.q
                            v[self.robot_arm[0] + '_shoulder_pan_joint'] = path_step[0]
                            v[self.robot_arm[0] + '_shoulder_lift_joint'] = path_step[1]
                            v[self.robot_arm[0] + '_upper_arm_roll_joint'] = path_step[2]
                            v[self.robot_arm[0] + '_elbow_flex_joint'] = path_step[3]
                            v[self.robot_arm[0] + '_forearm_roll_joint'] = path_step[4]
                            v[self.robot_arm[0] + '_wrist_flex_joint'] = path_step[5]
                            v[self.robot_arm[0] + '_wrist_roll_joint'] = path_step[6]

                            # v = self.robot.q
                            if self.robot_arm == 'rightarm':
                                v['l_shoulder_pan_joint'] = 1. * 3.14 / 2
                                v['l_shoulder_lift_joint'] = -0.52
                                v['l_upper_arm_roll_joint'] = 0.
                                v['l_elbow_flex_joint'] = -3.14 * 2 / 3
                                v['l_forearm_roll_joint'] = 0.
                                v['l_wrist_flex_joint'] = 0.
                                v['l_wrist_roll_joint'] = 0.
                                v['l_gripper_l_finger_joint'] = .15
                                v['l_gripper_r_finger_joint'] = .15
                            else:
                                v['r_shoulder_pan_joint'] = -1. * 3.14 / 2
                                v['r_shoulder_lift_joint'] = -0.52
                                v['r_upper_arm_roll_joint'] = 0.
                                v['r_elbow_flex_joint'] = -3.14 * 2 / 3
                                v['r_forearm_roll_joint'] = 0.
                                v['r_wrist_flex_joint'] = 0.
                                v['r_wrist_roll_joint'] = 0.
                                v['r_gripper_l_finger_joint'] = .15
                                v['r_gripper_r_finger_joint'] = .15

                            self.robot.set_positions(v)
                            self.dart_world.displace_gown()

                            # self.robot.set_positions(v)
                            self.dart_world.displace_gown()
                            # self.dart_world.check_collision()
                            self.dart_world.set_gown([self.robot_arm])
                            if num == 0 or num == len(traj_path)-1:
                                rospy.sleep(1.5)
                            else:
                                rospy.sleep(0.2)
                    cont = raw_input('\nEnter Q (q) or N (n) to stop visualization of this task. '
                                     'Otherwise visualizes again.\n')
                    if len(cont) == 0:
                        cont = ' '
                arm = raw_input('\nEnter R (r) for right arm (should be done first. Enter L (l) for left arm (should be '
                                'done second). Otherwise ends.\n')

        # If this is on the PR2, ask for arm to dress (left or right). If wrong input received, ends.
        elif self.machine.upper() == 'PR2':
            import hrl_dressing.controller as controller
            # Set up TOORAD process that includes DART simulation environment
            # self.toorad = DressingSimulationProcess(visualize=False)
            # This flag is set when the system is ready to start. It then waits for the user to put their hand under
            # the robot end effector
            self.ready_to_start = False

            # This flag is used to check when the subtask is completed so it can ask about starting the next task.
            self.subtask_complete = False

            self.capacitance_baseline = None
            self.force_baseline = None
            self.torque_baseline = None
            self.capacitance_baseline_values = []
            self.force_baseline_values = []
            self.torque_baseline_values = []
            self.zeroing = False
            self.time_series_forces = np.zeros([20000, 4])
            self.array_line = 0


            self.moving = False
            self.hz = 20.
            self.control_rate = rospy.Rate(self.hz)

            self.hmm_prediction = ''

            self.record_data = True

            # Start up the controller for the PR2
#            self.control = controller.Controller('base_footprint')
            self.control = controller.Controller('torso_lift_link')

            # Setup the publishers and subscribers necessary for dressing and the HMMs.
            self.initialize_pr2_comms(enable_realtime_HMM)

            arm = ''
            while not arm.upper() == 'Y' and not rospy.is_shutdown():
                print 'Will moving PR2 to initial configuration for the desired dressing task. First select the arm ' \
                      'to be dressed'
                arm = raw_input('\nEnter R (r) for right arm (should be done first). Enter L (l) for left arm (should be '
                                'done second). Otherwise ends.\n')
                if len(arm) == 0:
                    return
                elif arm.upper()[0] == 'R':
                    subtask = 0
                    h_arm = 'rightarm'
                    h_opposite_arm = 'leftarm'
                elif arm.upper()[0] == 'L':
                    subtask = 1
                    h_arm = 'leftarm'
                    h_opposite_arm = 'rightarm'
                else:
                    return

                # Calculate the trajectories based on the configuration in the simulator
                print self.save_file_path + self.trajectory_pickle_file_name + str(subtask) + '.pkl'
                loaded_data = load_pickle(self.save_file_path + self.trajectory_pickle_file_name + str(subtask) + '.pkl')

                params, \
                z, \
                pr2_params, \
                pr2_B_goals, \
                pr2_B_forearm_pointed_down_arm, \
                pr2_B_upperarm_pointed_down_shoulder, \
                pr2_B_hand, \
                pr2_B_wrist, \
                pr2_B_traj_start, \
                pr2_B_traj_forearm_end, \
                pr2_B_traj_upper_end, \
                pr2_B_traj_final_end, \
                traj_path = loaded_data
                print 'Trajectory data loaded succesfully!'

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

                    params, \
                    z, \
                    pr2_params, \
                    pr2_B_goals, \
                    pr2_B_forearm_pointed_down_arm, \
                    pr2_B_upperarm_pointed_down_shoulder, \
                    pr2_B_hand, \
                    pr2_B_wrist, \
                    pr2_B_traj_start, \
                    pr2_B_traj_forearm_end, \
                    pr2_B_traj_upper_end, \
                    pr2_B_traj_final_end, \
                    traj_path = loaded_data
                    print 'length of trajectory:', len(traj_path)

                    # Find first pose from TOORAD trajectory and move robot arms to start configuration.
                    print traj_path[0]
                    self.control.setJointGuesses(rightGuess=traj_path[0], leftGuess=None)
                    pr2_B_goal = pr2_B_goals[0]
                    print pr2_B_goal
                    pos, quat = Bmat_to_pos_quat(pr2_B_goal)
                    inp = 'Y'
                    while (inp.upper()[0] == 'Y' or inp.upper()[0] == 'N' or inp.upper()[0] == 'Q' or inp.upper()[0] == 'R') and not rospy.is_shutdown():
                        inp = raw_input(
                            '\nEnter Y (y) to move arms to start configuration. Will attempt movement for up to 10 '
                            'seconds. N or Q ends. Otherwise continues without movement.\n')
                        if len(inp) == 0:
                            break
                        elif inp.upper()[0] == 'Y':
#                            self.control.moveToJointAngles(traj_path[0], timeout=10.)
#                            self.control.moveGripperTo(pos, quaternion=quat, useInitGuess=True, rightArm=True, timeout=10.)
                             pos[0] += 0.1
                             pos[1] -= 0.1
                             pos[2] += 0.1
#                            self.control.moveGripperTo(pos, quaternion=quat, useInitGuess=True, rightArm=True, timeout=5., this_frame='base_footprint')
                             self.control.moveGripperTo(pos, useInitGuess=False, rightArm=True, timeout=1., this_frame='base_footprint')
                        elif inp.upper()[0] == 'R':
                            (current_position, current_orientation) = self.control.getGripperPosition(rightArm=True)
                            print 'current position', current_position
                            x=np.array([0.1, 0.0, 0.1])
                            print 'goal position', current_position+x
                            self.control.moveGripperTo(np.array(current_position), useInitGuess=False, rightArm=True, timeout=2.)
                        elif inp.upper()[0] == 'N' or inp.upper()[0] == 'Q':
                            return
                        else:
                            break

                inp = raw_input('\nEnter Y (y) to zero sensors. Otherwise ends.\n')
                if len(inp) == 0:
                    return
                elif inp.upper()[0] == 'Y':
                    # Zero sensors
                    self.zero_sensor_data()
                else:
                    return

                inp = ''
                while not inp.upper() == 'Y' and not rospy.is_shutdown():
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
                        print 'Dressing task for', h_arm, 'is now active. Movement will start when participant moves ' \
                                                             'hand under the robot end effector'
                        if visually_estimate_arm_pose:
                            self.begin_dressing_trajectory_from_vision(visually_estimate_arm_pose)
                        else:
                            self.begin_dressing_trajectory_no_vision(trial, participant, pr2_B_goals, traj_path)
                    elif inp.upper()[0] == 'R':
                        # Re-zero sensors.
                        self.zero_sensor_data()
                    elif inp.upper()[0] == 'N' or inp.upper()[0] == 'Q':
                        return


                while not self.subtask_complete and not rospy.is_shutdown():
                    rospy.sleep(1)
                print 'Task complete for', h_arm
        else:
            print 'I do not recognize the machine type. Try again!'
            return

    def begin_dressing_trajectory_from_vision(self):
        # TODO Execute trajectory following the pose estimated using vision
        pass

    def begin_dressing_trajectory_no_vision(self, trial_number, participant_number, p_B_g, traj):
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
        Kp = 0.1
        Kd = 0.02
        Ki = 0.02
        Ki_max = 1.0
        prev_error = 0.
        integral_error = 0.

        self.plotTime = []
        self.plotX = []
        self.plotY = []

        # Equilibrium point adjustment is used to alter the trajectory based on input from the capacitive sensor and
        # force-torque sensor.
        equilibrium_point_adjustment = np.zeros(3)
        force_adjustment = np.zeros(3)

        (current_goal_position, current_goal_orientation) = self.control.getGripperPosition(rightArm=True, this_frame='base_footprint')
        self.control.moveGripperTo(np.array(current_goal_position), useInitGuess=False, rightArm=True, timeout=2.)
        rospy.sleep(2.)
        current_goal = 0
        exit_count = 0
        # Move!
        while self.moving and not rospy.is_shutdown():
            exit_count = 1
            if exit_count >= 15:
                print 'exitting on max count'
                return
                break
            print 'Now on end effector goal', current_goal, 'out of a total', len(traj)
            current_time = rospy.get_time()
            print 'Time since last control loop execution:', current_time - self.lastMoveTime
            self.lastMoveTime = current_time

            if current_goal == len(traj):
                print 'Movement has finished!'
                break
            if current_time - self.last_capacitive_reading > 0.1 or current_time - self.last_ft_reading > 0.1:
                print 'Stopping movement because too much time has passed since the last data reading from the capacitive ' \
                'sensor or the force-torque sensor.'
                break

            (current_position, current_orientation) = self.control.getGripperPosition(rightArm=True, this_frame='base_footprint')
            pr2_B_current_gripper = createBMatrix(current_position, current_orientation)
            print 'distance to arm:', self.distance_to_arm
            dist_error_capped = max(0.05 - self.distance_to_arm, -0.05)
            current_distance_error = dist_error_capped * np.array(pr2_B_current_gripper)[0:3, 2]
            if np.linalg.norm(self.force) < 4.0:
                force_adjustment *= 0.95
            else:
                force_adjustment += self.force/10.*0.05
            force_adjustment = np.array([0., 0., 0.])

            error = np.array([0., 0., 0.05 - self.distance_to_arm])

            #error = np.array(current_goal_position) - np.array(current_position) + np.array([0., 0., 0.05 - self.distance_to_arm])
            new_goal = np.array(current_goal_position) + current_distance_error*Kp + (current_distance_error - prev_error)*Kd
            prev_error = current_distance_error
#            new_goal = np.array(current_goal_position) + np.array([0., 0., dist_error])
             
            #if not np.sign(error) == np.sign(integral_error):
            #    integral_error = 0.
            integral_error += error
            if np.linalg.norm(integral_error) > Ki_max:
                integral_error = integral_error/np.linalg.norm(integral_error)*Ki_max
            u = Kp*error + Kd*(error - prev_error) + Ki*integral_error
#            self.control.moveGripperTo(np.array(current_position) + u, useInitGuess=False, rightArm=True, timeout=5./(self.hz), this_frame='base_footprint')
            self.control.moveGripperTo(np.array(new_goal), useInitGuess=False, rightArm=True, timeout=1./(self.hz), this_frame='base_footprint')
            current_goal_position = new_goal
            self.control_rate.sleep()
            continue    


            distance_adjustment = current_distance_error#Kp * current_distance_error + Kd * (current_distance_error - prev_distance_error)
            print 'distance adjustment', distance_adjustment

            equilibrium_point_adjustment = force_adjustment + distance_adjustment

            target_position, target_orientation = Bmat_to_pos_quat(p_B_g[current_goal])
            target_position, target_orientation = current_position, current_orientation
            target_position += equilibrium_point_adjustment

            angle_temp, axis_temp, discard_point = tft.rotation_from_matrix(pr2_B_current_gripper.I*p_B_g[current_goal])
            current_goal = 0
            if np.linalg.norm(target_position - current_position) < 0.02 and np.abs(angle_temp) <= m.radians(5.) and current_goal < len(p_B_g):
                # Current goal has been reached. Move on to next goal.
                current_goal += 1
                target_position, target_orientation = Bmat_to_pos_quat(p_B_g[current_goal])
                target_position += equilibrium_point_adjustment
            if 3 < current_goal < 13 and self.distance_to_arm > 0.09:
                distance_adjustment += np.array(pr2_B_current_gripper)[0:3, 1] * (-0.05) + 0.05*np.array(pr2_B_current_gripper)[0:3, 2]
                current_goal = 13
                target_position, target_orientation = Bmat_to_pos_quat(p_B_g[current_goal])
                target_position += equilibrium_point_adjustment
            if 16 < current_goal < 22 and self.distance_to_arm > 0.09:
                distance_adjustment += np.array(pr2_B_current_gripper)[0:3, 1] * (-0.05) + 0.05 * np.array(pr2_B_current_gripper)[0:3, 2]
                current_goal = 22
                target_position, target_orientation = Bmat_to_pos_quat(p_B_g[current_goal])
                target_position += equilibrium_point_adjustment
            if current_goal > 4 and self.enable_realtime_HMM:
                self.run_HMM_realtime(np.dstack([self.time_series_forces[:self.array_line + 1, 0] * 1,
                                                 self.time_series_forces[:self.array_line + 1, 3] * 1,
                                                 self.time_series_forces[:self.array_line + 1, 1] * 1])[0])
            else:
                self.hmm_prediction = ''
            # goal_i = int(path[current_goal].split('-')[0])
            # if not goal_i == current_goal:
            #     print 'Something has gone wrong. The current goal index and goal index from the path differ'
                # print 'goal_i:', goal_i
                # print 'current_goal:', current_goal
                # self.moving = False
                # break
            # sol_i = int(path[0].split('-')[1])
            target_posture = traj[current_goal]
            #self.control.setJointGuesses(rightGuess=target_posture, leftGuess=None)
#            print 'target position\n', target_position
#            print 'current position\n', current_position
            
            error = target_position - current_position
#            print 'error\n', error
            # if np.linalg.norm(error) < 0.02:
            #      = error/np.linalg.norm(error)*0.02
            # elif np.linalg.norm(error) > 0.1:
            #     x = error / np.linalg.norm(error) * 0.1
            # else:
            #     x = error
            # dist = self.update_ee_setpoint()

            x = Kp * error# + Kd * (error - prev_error)
            prev_error = error
            prev_distance_error = current_distance_error
#            print 'current_position', current_position
#            print 'x', x
            x[0] = 0.0105
            x[1] = 0.
            x[2] = 0.0105
            print current_position + x
#            self.control.moveGripperTo(current_position + x, quaternion=target_orientation, useInitGuess=False, rightArm=True, timeout=1./self.hz, this_frame='base_footprint')
            (current_position, current_orientation) = self.control.getGripperPosition(rightArm=True)
            print 'current_position', current_position
            print current_position + x
            #if self.control.moveGripperTo(current_position + x, useInitGuess=False, rightArm=True, timeout=1./self.hz) is None:
            #    break
            
            if self.record_data:
                with open(save_file_name, 'a') as myfile:
                    myfile.write(str("{:.5f}".format(current_time))
                                 + ',' + str("{:.5f}".format(current_position[0]))
                                 + ',' + str("{:.5f}".format(current_position[1]))
                                 + ',' + str("{:.5f}".format(current_position[2]))
                                 + ',' + str("{:.5f}".format(self.force[0]))
                                 + ',' + str("{:.5f}".format(self.force[1]))
                                 + ',' + str("{:.5f}".format(self.force[2]))
                                 + ',' + str("{:.5f}".format(self.distance_to_arm))
                                 + ',' + self.hmm_prediction
                                 + '\n')
            
            self.control_rate.sleep()
            rospy.sleep(1.)
        print rospy.is_shutdown()
        print 'still moving?', self.moving
        isSuccess = raw_input('Success? Is the arm and shoulder both in the sleeve of the gown? [y/n] \n').upper() == 'Y'
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
        # saved_configs = np.array([[0,0.49400,0.10217,0.82392,0.40534,0.05529,0.10348,-0.90135,1.47876,0.29157,-0.35238],
        #                          [1,1.47366,0.54646,1.47147,1.55155,0.01104,0.96537,0.43731,-1.53448,0.29994,-0.33048]])

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
                self.last_ft_reading = rospy.get_time()
                self.force = np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z]) - self.force_baseline
                self.torque = np.array([msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z]) - self.torque_baseline
                if np.linalg.norm(self.force) >= 10.0:
                    print 'Forces exceeded 10 Newtons! Stopping movement!'
                    self.moving = False
                if self.moving:
                    self.time_series_forces[self.array_line] = [msg.header.stamp.to_sec(), self.force[0],
                                                                self.force[1], self.force[2]]
                    if self.array_line < len(self.time_series_forces)-1:
                        self.array_line += 1
                    else:
                        print 'Exceeded number of data lines being used for time series evalatuion by HMMs'

            elif self.zeroing:
                # Calculate the baseline value for the force-torque sensor
                self.force = np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z])
                self.torque = np.array([msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z])
                self.force_baseline_values.append(self.force)
                self.torque_baseline_values.append(self.torque)
                if len(self.force_baseline_values) > 50:
                    self.force_baseline = np.mean(self.force_baseline_values, axis=0)
                    self.torque_baseline = np.mean(self.torque_baseline_values, axis=0)

    def capacitive_sensor_cb(self, msg):
        with self.frame_lock:
            raw_capacitance = msg.data
            # TODO Ask Zackory about this. Is capacitance always len=1? If not, later things in this code do not work (max).
            if len(raw_capacitance) == 1:
                # Use just first capacitance value
                raw_capacitance = raw_capacitance[0]

            # Calculate the baseline value for the capacitive sensor
            if self.capacitance_baseline is None and self.zeroing:
                self.capacitance_baseline_values.append(raw_capacitance)
                if len(self.capacitance_baseline_values) > 50:
                    self.capacitance_baseline = np.mean(self.capacitance_baseline_values, axis=0)
            elif self.capacitance_baseline is not None:
                self.last_capacitive_reading = rospy.get_time()
                capacitance = max(-(raw_capacitance - self.capacitance_baseline), -3.5)
                # capacitance = max(capacitance, -3.5)
                self.distance_to_arm = self.estimate_distance_to_arm(capacitance)
                if not self.moving and self.ready_to_start and self.distance_to_arm <= 0.045:
                    self.ready_to_start = False
                    self.moving = True

    def estimate_distance_to_arm(self, cap):
        return 0.8438 / (cap + 4.681)

    def zero_sensor_data(self):
        self.capacitance_baseline = None
        self.force_baseline = None
        self.torque_baseline = None
        self.capacitance_baseline_values = []
        self.force_baseline_values = []
        self.torque_baseline_values = []
        self.time_series_forces = np.zeros([20000, 4])
        self.array_line = 0
        self.zeroing = True
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

    def setup_dart(self, visualize=True, filename='fullbody_alex_capsule.skel'):
        # Setup Dart ENV
        skel_file = self.pkg_path+'/models/'+filename
        from hrl_base_selection.dart_setup import DartDressingWorld
        self.dart_world = DartDressingWorld(skel_file)

        # Lets you visualize dart.
        if visualize:
            t = threading.Thread(target=self.visualize_dart)
            t.start()

        self.robot = self.dart_world.robot
        self.human = self.dart_world.human
        self.gown_leftarm = self.dart_world.gown_box_leftarm
        self.gown_rightarm = self.dart_world.gown_box_rightarm

        sign_flip = 1.
        if 'right' in self.robot_arm:
            sign_flip = -1.
        v = self.robot.q
        v['l_shoulder_pan_joint'] = 1.*3.14/2
        v['l_shoulder_lift_joint'] = -0.52
        v['l_upper_arm_roll_joint'] = 0.
        v['l_elbow_flex_joint'] = -3.14 * 2 / 3
        v['l_forearm_roll_joint'] = 0.
        v['l_wrist_flex_joint'] = 0.
        v['l_wrist_roll_joint'] = 0.
        v['l_gripper_l_finger_joint'] = .15
        v['l_gripper_r_finger_joint'] = .15
        v['r_shoulder_pan_joint'] = -1.*3.14/2
        v['r_shoulder_lift_joint'] = -0.52
        v['r_upper_arm_roll_joint'] = 0.
        v['r_elbow_flex_joint'] = -3.14*2/3
        v['r_forearm_roll_joint'] = 0.
        v['r_wrist_flex_joint'] = 0.
        v['r_wrist_roll_joint'] = 0.
        v['r_gripper_l_finger_joint'] = .15
        v['r_gripper_r_finger_joint'] = .15
        v['torso_lift_joint'] = 0.3
        self.robot.set_positions(v)
        self.dart_world.displace_gown()

        # robot_start = np.matrix([[1., 0., 0., 0.],
        #                          [0., 1., 0., 0.],
        #                          [0., 0., 1., 0.],
        #                          [0., 0., 0., 1.]])
        # positions = self.robot.positions()
        # pos, ori = Bmat_to_pos_quat(robot_start)
        # eulrpy = tft.euler_from_quaternion(ori, 'sxyz')
        # positions['rootJoint_pos_x'] = pos[0]
        # positions['rootJoint_pos_y'] = pos[1]
        # positions['rootJoint_pos_z'] = pos[2]
        # positions['rootJoint_rot_x'] = eulrpy[0]
        # positions['rootJoint_rot_y'] = eulrpy[1]
        # positions['rootJoint_rot_z'] = eulrpy[2]
        # self.robot.set_positions(positions)

        positions = self.robot.positions()
        positions['rootJoint_pos_x'] = 2.
        positions['rootJoint_pos_y'] = 0.
        positions['rootJoint_pos_z'] = 0.
        positions['rootJoint_rot_z'] = 3.14
        self.robot.set_positions(positions)
        # self.dart_world.set_gown()
        print 'Dart is ready!'

    def visualize_dart(self):
        win = pydart.gui.viewer.PydartWindow(self.dart_world)
        win.camera_event(1)
        win.set_capture_rate(10)
        win.run_application()

    def set_human_model_dof_dart(self, dof, human_arm):
        # bth = m.degrees(headrest_th)
        if not len(dof) == 4:
            print 'There should be exactly 4 values used for arm configuration. Three for the shoulder and one for ' \
                  'the elbow. But instead ' + str(len(dof)) + 'was sent. This is a ' \
                                                              'problem!'
            return False

        q = self.human.q
        # print 'human_arm', human_arm
        # j_bicep_left_x,y,z are euler angles applied in xyz order. x is forward, y is opposite direction of
        # upper arm, z is to the right.
        # j_forearm_left_1 is bend in elbow.
        if human_arm == 'leftarm':
            q['j_bicep_left_x'] = dof[0]
            q['j_bicep_left_y'] = -1*dof[1]
            q['j_bicep_left_z'] = dof[2]
            # q['j_bicep_left_roll'] = -1*0.
            q['j_forearm_left_1'] = dof[3]
            q['j_forearm_left_2'] = 0.
        elif human_arm == 'rightarm':
            q['j_bicep_right_x'] = -1*dof[0]
            q['j_bicep_right_y'] = dof[1]
            q['j_bicep_right_z'] = dof[2]
            # q['j_bicep_right_roll'] = 0.
            q['j_forearm_right_1'] = dof[3]
            q['j_forearm_right_2'] = 0.
        else:
            print 'I am not sure what arm to set the dof for.'
            return False
        self.human.set_positions(q)

if __name__ == '__main__':
    rospy.init_node('dressing_execution')

    import optparse

    p = optparse.OptionParser()

    p.add_option('-p', action='store', dest='participant', type=int, help='Participant number')
    p.add_option('-t', action='store', dest='trial',  type=int, help='Trial Number')
    p.add_option('-v', action="store_true", dest='visualize', help='Visualize on if true')
    p.add_option('-m', action='store', dest='machine', type='string', help='Machine this is running on '
                                                                                          '(PR2 or desktop)')

    opt, args = p.parse_args()
    print 'opt', opt
    print 'args', args

    toorad_dressing = TOORAD_Dressing_PR2(participant=opt.participant, trial=opt.participant,
                                          enable_realtime_HMM=False, visualize=opt.visualize,
                                          visually_estimate_arm_pose=False, adjust_arm_pose_visually=False,
                                          machine=opt.machine)

