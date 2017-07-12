#!/usr/bin/env python

import numpy as np
import math as m
import copy

import pydart2 as pydart

import roslib

import rospy, rospkg
import tf
from geometry_msgs.msg import PoseStamped, Pose, PoseArray
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.cbook import flatten

from sensor_msgs.msg import JointState
from std_msgs.msg import String

roslib.load_manifest('hrl_base_selection')
from hrl_base_selection.helper_functions import createBMatrix, Bmat_to_pos_quat, calc_axis_angle
from hrl_base_selection.dart_setup import DartDressingWorld
from hrl_base_selection.msg import PhysxOutcome
from hrl_base_selection.srv import InitPhysxBodyModel, PhysxInput, IKService, PhysxOutput, PhysxInputWaypoints

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from hrl_msgs.msg import FloatArrayBare

import random, threading

import openravepy as op
from openravepy.misc import InitOpenRAVELogging

import tf.transformations as tft

import pickle as pkl
roslib.load_manifest('hrl_lib')
from hrl_lib.util import save_pickle, load_pickle

import cma


class ScoreGeneratorDressingwithPhysx(object):

    def __init__(self, robot_arm='rightarm', human_arm='rightarm', visualize=False):

        self.visualize = visualize
        self.frame_lock = threading.RLock()

        self.robot_arm = None
        self.robot_opposite_arm = None

        self.human_arm = None
        self.human_opposite_arm = None

        # self.setup_openrave()

        self.set_robot_arm(robot_arm)
        self.set_human_arm(human_arm)

        self.traj_start = []
        self.traj_forearm_end = []

        self.axis = []
        self.angle = None

        rospack = rospkg.RosPack()
        self.pkg_path = rospack.get_path('hrl_base_selection')

        self.human_rot_correction = None

        # self.model = None
        self.force_cost = 0.

        self.goals = None
        self.pr2_B_reference = None
        self.task = None
        self.task_dict = None

        self.reference_names = None

        self.best_physx_score = 100.
        self.best_pr2_results = [[], []]

        self.arm_traj_parameters = []
        self.pr2_parameters = []

        # self.reachable = {}
        # self.manipulable = {}
        # self.scores = {}

        self.distance = 0.
        # self.score_length = {}
        # self.sorted_scores = {}

        self.gripper_B_tool = np.matrix([[0., -1., 0., 0.03],
                                         [1., 0., 0., 0.0],
                                         [0., 0., 1., -0.05],
                                         [0., 0., 0., 1.]])

        # Gripper coordinate system has z in direction of the gripper, x is the axis of the gripper opening and closing.
        # This transform corrects that to make x in the direction of the gripper, z the axis of the gripper open.
        # Centered at the very tip of the gripper.
        self.goal_B_gripper = np.matrix([[0.,  0.,   1.,   0.0],
                                         [0.,  1.,   0.,   0.0],
                                         [-1.,  0.,   0.,  0.0],
                                         [0.,  0.,   0.,   1.0]])
        self.origin_B_grasps = []
        self.goals = []

        self.optimal_z_offset = 0.05
        self.physx_output = False
        self.physx_outcome = None
        self.setup_dart()

        # self.setup_ik_service()
        self.setup_physx()
        self.update_physx_from_dart(initialization=True)
        # self.setup_physx_calls()


    def simulator_result_handler(self, msg):
        with self.frame_lock:
            self.physx_outcome = str(msg.outcome)
            self.physx_outcome_method2 = str(msg.outcome_method2)
            forces = [float(i) for i in msg.forces]
            temp_force_cost = 0.
            for force in forces:
                temp_force_cost += np.max([0., force - 1.0])/9.0

            temp_force_cost /= len(forces)
            self.force_cost = copy.copy(temp_force_cost)
            print 'Force cost from physx: ', self.force_cost
            self.physx_output = True
            print 'Physx simulation outcome: ', self.physx_outcome
            print 'Physx simulation outcome (circle path): ', self.physx_outcome_method2
            return True

    def run_interleaving_optimization_outer_level(self, maxiter=30, popsize=100):
        print 'starting running the optimization'
        # maxiter = 30/
        # popsize = m.pow(5, 2)*100
        maxiter = 2
        popsize = 2

        ### Current: Two positions, first with respect to the fist, second with respect to the upper arm, centered at
        # the shoulder and pointing X down the upper arm
        # cma parameters: [end_effector_trajectory_start_position (from fist): x,y,z,
        #                  end_effector_trajectory_end_position (centered at shoulder, pointing down arm): x, y, z,
        #                  human_arm_elbow_angle,
        #                  human_upper_arm_quaternion(euler:xzy): r, y, p ]

        parameters_min = np.array([m.radians(-5.), m.radians(-10.), m.radians(-10.),
                                   0.])
        parameters_max = np.array([m.radians(100.), m.radians(100.), m.radians(100),
                                   m.radians(135.)])
        parameters_scaling = (parameters_max-parameters_min)/2.
        parameters_initialization = (parameters_max+parameters_min)/2.
        parameters_initialization[0] = m.radians(0.)
        parameters_initialization[1] = m.radians(70.)
        parameters_initialization[2] = m.radians(0.)
        parameters_initialization[3] = m.radians(0.)
        # parameters_scaling[6] = m.radians(90)
        # parameters_scaling[7] = m.radians(90)
        # parameters_scaling[8] = m.radians(90)
        # parameters_scaling[9] = m.radians(40)
        opts1 = {'seed': 1234, 'ftarget': -1., 'popsize': popsize, 'maxiter': maxiter, 'maxfevals': 1e8,
                 'CMA_cmean': 0.25,
                 'verb_filenameprefix': 'outcma_arm_and_trajectory',
                 'scaling_of_variables': list(parameters_scaling),
                 'bounds': [list(parameters_min), list(parameters_max)]}

        # optimization_results[<model>, <number_of_configs>, <head_rest_angle>, <headx>, <heady>, <allow_bed_movement>]
        self.optimization_results = cma.fmin(self.objective_function_traj_and_arm_config,
                                        list(parameters_initialization),
                                        1.,
                                        options=opts1)

        print 'Outcome is: '
        print self.optimization_results
        print 'Best trajectory and arm config: \n', self.optimization_results[0]
        print 'Associated score: ', self.optimization_results[1]
        print 'Best PR2 configuration: \n', self.best_pr2_results[0]
        print 'Associated score: ', self.best_pr2_results[1]
        optimized_traj_arm_output = [self.optimization_results[0], self.optimization_results[1]]
        optimized_pr2_output = [self.best_pr2_results[0], self.best_pr2_results[1]]

        save_pickle(optimized_traj_arm_output, self.pkg_path+'/data/best_trajectory_and_arm_config.pkl')
        save_pickle(optimized_pr2_output, self.pkg_path+'/data/best_pr2_config.pkl')

    def objective_function_traj_and_arm_config(self, params):
        # params[7:] = [0., 0., 1.]
        # params = [m.radians(0.), 0.0,  m.radians(0.),
        #           m.radians(70)]
        # params = [0.17374768,  0.0647019,   0.09521618,  0.01036762, -0.05860782,  0.10786874,
        #           0.01046108, -0.92726507, -0.69534598,  0.72062966]

        arm = self.human_arm.split('a')[0]

        # path_distance = np.linalg.norm(np.array(params[0:3])-np.array(params[3:6]))
        print 'params'
        print params
        self.set_human_model_dof_dart([params[0], params[1], params[2], params[3]], self.human_arm)

        skeleton_frame_B_worldframe = np.matrix([[1.,  0., 0., 0.],
                                                 [0.,  0., 1., 0.],
                                                 [0., -1., 0., 0.],
                                                 [0.,  0., 0., 1.]])

        origin_B_pelvis = np.matrix(self.human.bodynode('h_pelvis').world_transform())
        origin_B_upperarmbase = np.matrix(self.human.bodynode('h_bicep_'+arm).world_transform())
        origin_B_upperarmbase[0:3, 0:3] = origin_B_pelvis[0:3, 0:3]
        # origin_B_uabr = np.matrix(self.human.bodynode('h_bicep_'+arm).world_transform())
        # origin_B_uabr[0:3, 0:3] = np.eye(3)
        origin_B_upperarm = np.matrix(self.human.bodynode('h_bicep_'+arm).world_transform())
        # origin_B_uar = np.matrix(self.human.bodynode('h_bicep_'+arm).world_transform())
        origin_B_forearm = np.matrix(self.human.bodynode('h_forearm_'+arm).world_transform())
        # origin_B_far = np.matrix(self.human.bodynode('h_forearm_'+arm).world_transform())
        origin_B_hand = np.matrix(self.human.bodynode('h_hand_'+arm).world_transform())
        # origin_B_hr = np.matrix(self.human.bodynode('h_hand_right').world_transform())

        z_origin = np.array([0., 0., 1.])
        x_vector = (-1*np.array(origin_B_hand)[0:3, 1])
        x_vector /= np.linalg.norm(x_vector)
        y_orth = np.cross(z_origin, x_vector)
        y_orth /= np.linalg.norm(y_orth)
        z_orth = np.cross(x_vector, y_orth)
        z_orth /= np.linalg.norm(z_orth)
        origin_B_hand_rotated = np.eye(4)
        # print 'x_vector'
        # print x_vector
        # print 'origin_B_hr_rotated'
        # print origin_B_hr_rotated
        # print 'np.reshape(x_vector, [3, 1])'
        # print np.reshape(x_vector, [3, 1])
        # print 'origin_B_hr_rotated[0:3, 0]'
        # print origin_B_hr_rotated[0:3, 0]
        origin_B_hand_rotated[0:3, 0] = copy.copy(x_vector)
        origin_B_hand_rotated[0:3, 1] = copy.copy(y_orth)
        origin_B_hand_rotated[0:3, 2] = copy.copy(z_orth)
        origin_B_hand_rotated[0:3, 3] = copy.copy(np.array(origin_B_hand)[0:3, 3])
        origin_B_hand_rotated = np.matrix(origin_B_hand_rotated)

        rev = m.radians(180.)

        traj_y_offset, traj_z_offset = self.get_best_traj_offset()

        hand_rotated_B_traj_start_pos = np.matrix([[m.cos(rev), -m.sin(rev), 0., 0.3],
                                                   [m.sin(rev), m.cos(rev), 0., traj_y_offset],
                                                   [0., 0., 1., traj_z_offset],
                                                   [0., 0., 0., 1.]])

        # hand_rotated_B_traj_start_pos = np.eye(4)
        # hand_rotated_B_traj_start_pos[0:3, 3] = copy.copy(params[0:3])
        # hand_rotated_B_traj_start_pos[0, 3] += 0.07
        # hand_rotated_B_traj_start_pos = np.matrix(hand_rotated_B_traj_start_pos)

        origin_B_traj_start_pos = origin_B_hand_rotated*hand_rotated_B_traj_start_pos

        origin_B_upperarm_world = origin_B_upperarm * skeleton_frame_B_worldframe
        origin_B_forearm_world = origin_B_forearm * skeleton_frame_B_worldframe

        # print 'origin_B_traj_start_pos'
        # print origin_B_traj_start_pos

        # origin_B_world_rotated_shoulder = createBMatrix(np.reshape(np.array(origin_B_uabr[0:3, 3]), [1, 3])[0], list(tft.quaternion_from_euler(params[7], -params[8], params[9], 'rzxy')))

        # Because green kevin has the upper with a bend in it, I shift the shoulder location by that bend offset.
        # shoulder_origin_B_should_origin_shifted_green_kevin = np.matrix(np.eye(4))
        # shoulder_origin_B_should_origin_shifted_green_kevin[1, 3] = -0.04953
        # origin_B_world_rotated_shoulder = origin_B_world_rotated_shoulder*shoulder_origin_B_should_origin_shifted_green_kevin

        # origin_B_uabr[0:3, 3] = origin_B_world_rotated_shoulder[0:3, 3]
        # origin_B_uar[0:3, 3] = origin_B_world_rotated_shoulder[0:3, 3]

        origin_B_forearm_pointed_down_arm = np.eye(4)
        z_origin = np.array([0., 0., 1.])
        x_vector = -1 * np.array(origin_B_forearm_world)[0:3, 2]
        x_vector /= np.linalg.norm(x_vector)
        if np.abs(x_vector[2]) > 0.99:
            x_vector = np.array([0., 0., np.sign(x_vector[2]) * 1.])
            y_orth = np.array([np.sign(x_vector[2]) * -1., 0., 0.])
            z_orth = np.array([0., np.sign(x_vector[2]) * 1., 0.])
            # origin_B_upperarm_pointed_down_shoulder[0:3, 0] = np.array([0., 0., np.sign(x_vector[2])*1.])
            # origin_B_upperarm_pointed_down_shoulder[0:3, 1] = y_orth
            # origin_B_upperarm_pointed_down_shoulder[0:3, 2] = z_orth
            # origin_B_upperarm_pointed_down_shoulder[0:3, 3] = np.array(origin_B_uabr)[0:3, 3]
        else:
            y_orth = np.cross(z_origin, x_vector)
            y_orth = y_orth / np.linalg.norm(y_orth)
            z_orth = np.cross(x_vector, y_orth)
            z_orth = z_orth / np.linalg.norm(z_orth)
        origin_B_forearm_pointed_down_arm[0:3, 0] = x_vector
        origin_B_forearm_pointed_down_arm[0:3, 1] = y_orth
        origin_B_forearm_pointed_down_arm[0:3, 2] = z_orth
        origin_B_forearm_pointed_down_arm[0:3, 3] = np.array(origin_B_forearm_world)[0:3, 3]
        origin_B_forearm_pointed_down_arm = np.matrix(origin_B_forearm_pointed_down_arm)

        forearm_pointed_down_arm_B_traj_end_pos = np.eye(4)
        forearm_pointed_down_arm_B_traj_end_pos[0:3, 3] = [-0.05, traj_y_offset, traj_z_offset]
        forearm_pointed_down_arm_B_traj_end_pos = np.matrix(forearm_pointed_down_arm_B_traj_end_pos)
        rev = m.radians(180.)
        forearm_pointed_down_arm_B_traj_end = np.matrix([[m.cos(rev), -m.sin(rev), 0., 0.02],
                                                         [m.sin(rev), m.cos(rev), 0., traj_y_offset],
                                                         [0., 0., 1., traj_z_offset],
                                                         [0., 0., 0., 1.]])
        origin_B_traj_forearm_end = origin_B_forearm_pointed_down_arm * forearm_pointed_down_arm_B_traj_end

        origin_B_upperarm_pointed_down_shoulder = np.eye(4)
        z_origin = np.array([0., 0., 1.])
        x_vector = -1*np.array(origin_B_upperarm_world)[0:3, 2]
        x_vector /= np.linalg.norm(x_vector)
        if np.abs(x_vector[2]) > 0.99:
            x_vector = np.array([0., 0., np.sign(x_vector[2])*1.])
            y_orth = np.array([np.sign(x_vector[2])*-1., 0., 0.])
            z_orth = np.array([0., np.sign(x_vector[2])*1., 0.])
            # origin_B_upperarm_pointed_down_shoulder[0:3, 0] = np.array([0., 0., np.sign(x_vector[2])*1.])
            # origin_B_upperarm_pointed_down_shoulder[0:3, 1] = y_orth
            # origin_B_upperarm_pointed_down_shoulder[0:3, 2] = z_orth
            # origin_B_upperarm_pointed_down_shoulder[0:3, 3] = np.array(origin_B_uabr)[0:3, 3]
        else:
            y_orth = np.cross(z_origin, x_vector)
            y_orth = y_orth/np.linalg.norm(y_orth)
            z_orth = np.cross(x_vector, y_orth)
            z_orth = z_orth/np.linalg.norm(z_orth)
        origin_B_upperarm_pointed_down_shoulder[0:3, 0] = x_vector
        origin_B_upperarm_pointed_down_shoulder[0:3, 1] = y_orth
        origin_B_upperarm_pointed_down_shoulder[0:3, 2] = z_orth
        origin_B_upperarm_pointed_down_shoulder[0:3, 3] = np.array(origin_B_upperarm_world)[0:3, 3]
        origin_B_rotated_pointed_down_shoulder = np.matrix(origin_B_upperarm_pointed_down_shoulder)
        # origin_B_rotated_pointed_down_shoulder = origin_B_uabr*uabr_B_uabr_corrected*np.matrix(shoulder_origin_B_rotated_pointed_down_shoulder)
        # print 'origin_B_rotated_pointed_down_shoulder'
        # print origin_B_rotated_pointed_down_shoulder

        # origin_B_rotated_pointed_down_shoulder[0:3, 3] = origin_B_world_rotated_shoulder[0:3, 3]

        upperarm_pointed_down_shoulder_B_traj_end_pos = np.eye(4)
        upperarm_pointed_down_shoulder_B_traj_end_pos[0:3, 3] = [-0.05, traj_y_offset, traj_z_offset]
        upperarm_pointed_down_shoulder_B_traj_end_pos = np.matrix(upperarm_pointed_down_shoulder_B_traj_end_pos)
        rev = m.radians(180.)
        upperarm_pointed_down_shoulder_B_traj_upper_end = np.matrix([[m.cos(rev), -m.sin(rev), 0., -0.05],
                                                                     [m.sin(rev), m.cos(rev), 0., traj_y_offset+.05],
                                                                     [0., 0., 1., traj_z_offset+.05],
                                                                     [0., 0., 0., 1.]])

        origin_B_traj_upper_end = origin_B_upperarm_pointed_down_shoulder * upperarm_pointed_down_shoulder_B_traj_upper_end

        rev = m.radians(180.)
        shoulder_position_B_traj_final_end = np.matrix([[m.cos(rev), -m.sin(rev), 0., -0.03],
                                                        [m.sin(rev), m.cos(rev), 0., 0.],
                                                        [0., 0., 1., 0.13],
                                                        [0., 0., 0., 1.]])
        origin_B_shoulder_position = np.eye(4)
        origin_B_shoulder_position[0:3, 3] = np.array(origin_B_upperarm)[0:3, 3]
        origin_B_traj_final_end = np.matrix(origin_B_shoulder_position)*shoulder_position_B_traj_final_end

        # print 'rotated_pointed_down_shoulder_B_traj_end_pos'
        # print rotated_pointed_down_shoulder_B_traj_end_pos
        # print 'origin_B_traj_start_pos'
        # print origin_B_traj_start_pos
        # origin_B_traj_end_pos = origin_B_forearm_pointed_down_arm*forearm_pointed_down_arm_B_traj_end
        # print 'origin_B_traj_end_pos'
        # print origin_B_traj_end_pos

        # print 'origin_B_uabr_corrected'
        # print origin_B_uabr*uabr_B_uabr_corrected

        # th = m.radians(180.)
        #
        # x_vector = np.array(params[0:3])-np.array(params[3:6])
        # x_vector /= np.linalg.norm(x_vector)
        # y_orth = np.cross(z_origin, x_vector)
        # y_orth = y_orth/np.linalg.norm(y_orth)
        # z_orth = np.cross(x_vector, y_orth)
        # z_orth = z_orth/np.linalg.norm(z_orth)
        # origin_B_traj_start = np.eye(4)
        # origin_B_traj_start[0:3, 0] = np.reshape(x_vector, [3, 1])
        # origin_B_traj_start[0:3, 1] = np.reshape(y_orth, [3, 1])
        # origin_B_traj_start[0:3, 2] = np.reshape(z_orth, [3, 1])
        # origin_B_traj_start[0:3, 3] = np.reshape(params[0:3], [3, 1])
        # origin_B_traj_start = np.matrix(origin_B_traj_start)

        # z_origin = np.array([0., 0., 1.])
        # x_vector = np.array(origin_B_traj_end_pos)[0:3, 3] - np.array(origin_B_traj_start_pos)[0:3, 3]
        # x_vector /= np.linalg.norm(x_vector)
        # y_orth = np.cross(z_origin, x_vector)
        # y_orth /= np.linalg.norm(y_orth)
        # z_orth = np.cross(x_vector, y_orth)
        # z_orth /= np.linalg.norm(z_orth)
        # origin_B_traj_start = np.eye(4)
        # origin_B_traj_start[0:3, 0] = x_vector
        # origin_B_traj_start[0:3, 1] = y_orth
        # origin_B_traj_start[0:3, 2] = z_orth
        # origin_B_traj_start[0:3, 3] = np.array(origin_B_traj_start_pos)[0:3, 3]
        # origin_B_traj_start = np.matrix(origin_B_traj_start)

        origin_B_traj_start = origin_B_traj_start_pos

        # print 'origin_B_upperarm_pointed_down_shoulder'
        # print origin_B_upperarm_pointed_down_shoulder

        # print 'origin_B_traj_start'
        # print origin_B_traj_start

        # path_distance = np.linalg.norm(np.array(origin_B_traj_end_pos)[0:3, 3] - np.array(origin_B_traj_start_pos)[0:3, 3])
        # print 'path_distance'
        # print path_distance
        # uabr_corrected_B_traj_start = uabr_B_uabr_corrected.I*origin_B_uabr.I*origin_B_traj_start
        # test_world_shoulder_B_sleeve_start_rotz = np.matrix([[ m.cos(th), -m.sin(th),     0.],
        #                                                      [ m.sin(th),  m.cos(th),     0.],
        #                                                       [        0.,         0.,     1.]])
        # hr_rotated_B_traj_start = createBMatrix([params[0], params[1], params[2]],
        #                                   tft.quaternion_from_euler(params[3], params[4], params[5], 'rzyx'))
        pos_t, quat_t = Bmat_to_pos_quat(origin_B_traj_start)
        # if np.linalg.norm([params[0]-pos_t[0], params[1]-pos_t[1], params[2]-pos_t[2]]) > 0.01:
        #     print 'Somehow I have mysteriously gotten error in my trajectory start position in openrave. Abort!'
        #     return 100.
        # uar_base_B_uar = createBMatrix([0, 0, 0], tft.quaternion_from_euler(params[10], params[11], params[12], 'rzxy'))
        # hr_B_traj_start = origin_B_hr_rotated.I*origin_B_uabr*uabr_B_uabr_corrected*uabr_corrected_B_traj_start
        # traj_start_B_traj_end = np.matrix(np.eye(4))
        # traj_start_B_traj_end[0, 3] = params[6]
        # uabr_B_traj_end = uabr_B_uabr_corrected*uabr_corrected_B_traj_start*traj_start_B_traj_end
        # uar_B_traj_end = origin_B_uar.I*origin_B_uabr*uabr_B_traj_end

        # print 'origin_B_hr'
        # print origin_B_hr
        # print 'origin_B_uabr'
        # print origin_B_uabr
        # print 'uabr_corrected_B_traj_start'
        # print uabr_corrected_B_traj_start
        testing = True
        if not testing:
            if np.linalg.norm(hand_rotated_B_traj_start_pos[0:3, 3]) > 0.3:
                print 'traj start too far away'
                return 10. + np.linalg.norm(hand_rotated_B_traj_start_pos[0:3, 3])
            if np.linalg.norm(upperarm_pointed_down_shoulder_B_traj_end_pos[0:3, 3]) > 0.3:
                print 'traj end too far away'
                return 10. + np.linalg.norm(upperarm_pointed_down_shoulder_B_traj_end_pos[0:3, 3])
            if hand_rotated_B_traj_start_pos[0, 3] > 0.3 or hand_rotated_B_traj_start_pos[0, 3] < 0.:
                print 'traj start relation to hand is too close or too far'
                return 10. + np.abs(hand_rotated_B_traj_start_pos[0, 3])
            if upperarm_pointed_down_shoulder_B_traj_end_pos[0, 3] > 0.05:
                print 'traj ends too far from upper arm'
                return 10. + upperarm_pointed_down_shoulder_B_traj_end_pos[0, 3]
            if upperarm_pointed_down_shoulder_B_traj_end_pos[0, 3] < -0.15:
                print 'traj ends too far behind upper arm'
                return 10. - upperarm_pointed_down_shoulder_B_traj_end_pos[0, 3]
        testing = True
        # path_waypoints = np.arange(0., path_distance+path_distance*0.01, path_distance/5.)
        # print 'path_distance'
        # print path_distance
        # print 'number of path waypoints'
        # print len(path_waypoints)
        # print 'path_waypoints'
        # print path_waypoints
        # self.goals = []
        # for goal in path_waypoints:
        #     traj_start_B_traj_waypoint = np.matrix(np.eye(4))
        #     traj_start_B_traj_waypoint[0, 3] = goal
        #     origin_B_traj_waypoint = copy.copy(np.matrix(origin_B_traj_start)*np.matrix(traj_start_B_traj_waypoint))
        #     # print 'origin_B_traj_start'
        #     # print origin_B_traj_start
        #     # print 'traj_start_B_traj_waypoint'
        #     # print traj_start_B_traj_waypoint
        #     # print 'origin_B_traj_waypoint'
        #     # print origin_B_traj_waypoint
        #     # print 'origin_B_uabr'
        #     # print origin_B_uabr
        #     # print 'uabr_B_uabr_corrected'
        #     # print uabr_B_uabr_corrected
        #     # print 'uabr_corrected_B_traj_start'
        #     # print uabr_corrected_B_traj_start
        #     # print 'traj_start_B_traj_waypoint'
        #     # print traj_start_B_traj_waypoint
        #     # print 'origin_B_traj_waypoint'
        #     # print origin_B_traj_waypoint
        #     self.goals.append(copy.copy(origin_B_traj_waypoint))
        #     upperarm_vector = np.array((origin_B_upperarm.I*origin_B_traj_waypoint))[0:3, 3]
        #     forearm_vector = np.array((origin_B_forearm.I*origin_B_traj_waypoint))[0:3, 3]
        #     hand_vector = np.array((origin_B_hand.I*origin_B_traj_waypoint))[0:3, 3]
        #     min_distance = np.min([np.linalg.norm([upperarm_vector[0], upperarm_vector[2]]),
        #                            np.linalg.norm([forearm_vector[0], forearm_vector[2]]),
        #                            np.linalg.norm([hand_vector[0], hand_vector[2]])])
        #     if not testing:
        #         if min_distance > 0.28:  # goals too far from any arm segment. The gown sleeve opening is 28 cm long
        #             print 'goals too far from arm segments'
        #             return 10 + min_distance  # distance from arm segment

        start_time = rospy.Time.now()
        # self.set_goals()
        # print self.origin_B_grasps
        # maxiter = 2#20
        # popsize = 2#4*20*2
        # cma parameters: [pr2_base_x, pr2_base_y, pr2_base_theta, pr2_base_height,
        # human_arm_dof_1, human_arm_dof_2, human_arm_dof_3, human_arm_dof_4, human_arm_dof_5,
        # human_arm_dof_6, human_arm_dof_7]
        # parameters_min = np.array([-2., -2., -2.5/3.*m.pi-.001, 0.])
        # parameters_max = np.array([2., 2., 2.5/3.*m.pi+.001, 0.3])
        # parameters_min = np.array([-0.1, -1.0, m.pi/2. - .001, 0.2])
        # parameters_max = np.array([0.8, -0.3, 2.5*m.pi/2. + .001, 0.3])
        # parameters_scaling = (parameters_max-parameters_min)/2.
        # parameters_initialization = (parameters_max+parameters_min)/2.
        # opts1 = {'seed': 1234, 'ftarget': -1., 'popsize': popsize, 'maxiter': maxiter, 'maxfevals': 1e8,
        #          'CMA_cmean': 0.25,
        #          'verb_filenameprefix': 'outcma_pr2_base',
        #          'scaling_of_variables': list(parameters_scaling),
        #          'tolfun': 1e-3,
        #          'bounds': [list(parameters_min), list(parameters_max)]}
        # self.kinematics_optimization_results = cma.fmin(self.objective_function_one_config,
        #                                               list(parameters_initialization),
        #                                               1.,
        #                                               options=opts1)

        # self.pr2_parameters.append([self.kinematics_optimization_results[0], self.kinematics_optimization_results[1]])
        # save_pickle(self.pr2_parameters, self.pkg_path+'/data/all_pr2_configs.pkl')
        # elapsed_time = rospy.Time.now()-start_time
        # print 'Done with openrave round. Time elapsed:', elapsed_time.to_sec()
        # print 'Openrave results:'
        # print self.kinematics_optimization_results
        self.force_cost = 0.
        if True:
            self.physx_output = False
            self.physx_outcome = None
            # traj_start = np.array(origin_B_traj_start_pos)[0:3, 3]
            # traj_end = np.array(origin_B_traj_end_pos)[0:3, 3]

            # print 'origin_B_traj_upper_end'
            # print origin_B_traj_upper_end
            # print 'origin_B_traj_final_end'
            # print origin_B_traj_final_end

            self.update_physx_from_dart(origin_B_forearm_pointed_down_arm=origin_B_forearm_pointed_down_arm,
                                        origin_B_upperarm_pointed_down_shoulder=origin_B_upperarm_pointed_down_shoulder,
                                        traj_start=origin_B_traj_start, traj_forearm_end=origin_B_traj_forearm_end,
                                        traj_upper_end=origin_B_traj_upper_end, traj_final_end=origin_B_traj_final_end)

            # pos_a, quat_a = Bmat_to_pos_quat(origin_B_upperarm_world)

            # print 'quat_a'
            # print quat_a
            # quat_a =
            # quat_a = list(tft.quaternion_from_euler(params[7], -params[8], params[9], 'rzxy'))
            # print 'quat_a'
            # print quat_a
            # out_msg = FloatArrayBare()
            # out_msg.data = [float(t) for t in list(flatten([pos_t, quat_t, path_distance, params[6], quat_a]))]
                # float(list(flatten([pos_t, quat_t, params[6], params[7], quat_a])))
            # self.physx_output = None
            # self.traj_to_simulator_pub.publish(out_msg)
            while not rospy.is_shutdown() and not self.physx_output:
                rospy.sleep(0.5)
                # print 'waiting a sec'
            # self.physx_outcome = None
            self.physx_output = False
            if self.physx_outcome == 'good':
                with self.frame_lock:
                    alpha = 1.  # cost on forces
                    beta = 1.  # cost on manipulability
                    physx_score = self.force_cost*alpha + self.kinematics_optimization_results[1]*beta
                    if physx_score < self.best_physx_score:
                        self.best_physx_score = physx_score
                        self.best_pr2_results = self.kinematics_optimization_results
                    self.arm_traj_parameters.append([params, physx_score])

                    save_pickle(self.arm_traj_parameters, self.pkg_path+'/data/all_arm_traj_configs.pkl')
                    print 'Force cost was: ', self.force_cost
                    print 'Physx score was: ', physx_score
                    return physx_score
        self.physx_outcome = None
        self.physx_output = False
        alpha = 1.  # cost on forces
        beta = 1.  # cost on manipulability
        self.arm_traj_parameters.append([params, 10. + self.force_cost*alpha + self.kinematics_optimization_results[1]*beta])
        save_pickle(self.arm_traj_parameters, self.pkg_path+'/data/all_arm_traj_configs.pkl')
        physx_score = 10. + self.force_cost*alpha + self.kinematics_optimization_results[1]*beta
        print 'Force cost was: ', self.force_cost
        print 'Kinematics score was: ', self.kinematics_optimization_results[1]
        print 'Total score was: ', physx_score
        return physx_score

    def set_goals(self, single_goal=False):
        self.origin_B_grasps = []
        for num in xrange(len(self.goals)):
            self.origin_B_grasps.append(np.array(np.matrix(self.goals[num]))*np.matrix(self.gripper_B_tool.I))#*np.matrix(self.goal_B_gripper)))
            # print 'self.goals[', num, ']'
            # print self.goals[num]
            # print 'self.origin_B_grasps[', num, ']'
            # print self.origin_B_grasps[num]

    def set_robot_arm(self, arm):
        if self.robot_arm == arm:
            return False
        elif 'left' in arm or 'right' in arm:
            # Set robot arm for dressing
            print 'Setting the robot arm being used by base selection to ', arm
            if 'left' in arm:
                self.robot_arm = 'leftarm'
                self.robot_opposite_arm = 'rightarm'
            elif 'right' in arm:
                self.robot_arm = 'rightarm'
                self.robot_opposite_arm = 'leftarm'
            # for robot_arm in [self.robot_opposite_arm, self.robot_arm]:
                # self.op_robot.SetActiveManipulator(robot_arm)
                # self.manip = self.op_robot.GetActiveManipulator()
                # ikmodel = op.databases.inversekinematics.InverseKinematicsModel(self.op_robot,
                #                                                                 iktype=op.IkParameterization.Type.Transform6D)
                # if not ikmodel.load():
                #     print 'IK model not found for this arm. Generating the ikmodel for the ', robot_arm
                #     print 'This will take a while'
                #     ikmodel.autogenerate()
                # self.manipprob = op.interfaces.BaseManipulation(self.op_robot)
            return True
        else:
            print 'ERROR'
            print 'I do not know what arm to be using'
            return False

    def set_human_arm(self, arm):
        # Set human arm for dressing
        print 'Setting the human arm being used by base selection to ', arm
        if 'left' in arm:
            self.human_arm = 'leftarm'
            self.human_opposite_arm = 'rightarm'
            return True
        elif 'right' in arm:
            self.human_arm = 'rightarm'
            self.human_opposite_arm = 'leftarm'
            return True
        else:
            print 'ERROR'
            print 'I do not know what arm to be using'
            return False

    def objective_function_one_config(self, current_parameters):
        # start_time = rospy.Time.now()
        # current_parameters = [0.3, -0.9, 2.5*m.pi/3., 0.3]
        # current_parameters = [-0.34303706, -0.7381042,   1.2557902,   0.29065268]
        return np.random.random()
        x = current_parameters[0]
        y = current_parameters[1]
        th = current_parameters[2]
        z = current_parameters[3]

        origin_B_pr2 = np.matrix([[ m.cos(th), -m.sin(th),     0.,         x],
                                  [ m.sin(th),  m.cos(th),     0.,         y],
                                  [        0.,         0.,     1.,        0.],
                                  [        0.,         0.,     0.,        1.]])
        v = self.robot.positions()
        v['rootJoint_pos_x'] = x
        v['rootJoint_pos_y'] = y
        v['rootJoint_pos_z'] = 0.
        v['rootJoint_rot_z'] = th

        # sign_flip = 1.
        # if 'right' in self.robot_arm:
        #     sign_flip = -1.
        v['l_shoulder_pan_joint'] = 3.14 / 2
        v['l_shoulder_lift_joint'] = -0.52
        v['l_upper_arm_roll_joint'] = 0.
        v['l_elbow_flex_joint'] = -3.14 * 2 / 3
        v['l_forearm_roll_joint'] = 0.
        v['l_wrist_flex_joint'] = 0.
        v['l_wrist_roll_joint'] = 0.
        v['l_gripper_l_finger_joint'] = .24
        v['l_gripper_r_finger_joint'] = .24
        v['r_shoulder_pan_joint'] = -1 * 3.14 / 2
        v['r_shoulder_lift_joint'] = -0.52
        v['r_upper_arm_roll_joint'] = 0.
        v['r_elbow_flex_joint'] = -3.14 * 2 / 3
        v['r_forearm_roll_joint'] = 0.
        v['r_wrist_flex_joint'] = 0.
        v['r_wrist_roll_joint'] = 0.
        v['r_gripper_l_finger_joint'] = .24
        v['r_gripper_r_finger_joint'] = .24
        v['torso_lift_joint'] = 0.3

        v['torso_lift_joint'] = z

        self.robot.set_positions(v)
        self.dart_world.set_gown(self.robot_arm)

        # PR2 is too close to the person (who is at the origin). PR2 base is 0.668m x 0.668m
        distance_from_origin = np.linalg.norm(origin_B_pr2[:2, 3])
        if distance_from_origin <= 0.334:
            return 10. + 1. + (0.4 - distance_from_origin)

        # v = self.robot.q
        # v['torso_lift_joint'] = z
        # self.robot.set_positions(v)

        distance = 10000000.
        out_of_reach = True
        for origin_B_grasp in self.origin_B_grasps:
            pr2_B_goal = origin_B_pr2.I*origin_B_grasp
            distance = np.min([np.linalg.norm(pr2_B_goal[:2, 3]), distance])
            if distance <= 1.25:
                out_of_reach = False
                # print 'not out of reach'
                break
        if out_of_reach:
            # print 'location is out of reach'
            return 10. +1.+ 20.*(distance - 1.25)

        reach_score = 0.
        manip_score = 0.
        goal_scores = []
        manip = 0.
        reached = 0.

        # sign_flip = 1.
        # if 'right' in self.robot_arm:
        #     sign_flip = -1.
        # v = self.robot.q
        # v[self.robot_opposite_arm[0] + '_shoulder_pan_joint'] = -sign_flip * 3.14 / 2
        # v[self.robot_opposite_arm[0] + '_shoulder_lift_joint'] = -0.52
        # v[self.robot_opposite_arm[0] + '_upper_arm_roll_joint'] = 0.
        # v[self.robot_opposite_arm[0] + '_elbow_flex_joint'] = -3.14 * 2 / 3
        # v[self.robot_opposite_arm[0] + '_forearm_roll_joint'] = 0.
        # v[self.robot_opposite_arm[0] + '_wrist_flex_joint'] = 0.
        # v[self.robot_opposite_arm[0] + '_wrist_roll_joint'] = 0.
        # self.robot.set_positions(v)

        in_collision = self.is_dart_in_collision()

        close_to_collision = False
        check_if_PR2_is_near_collision = False
        if check_if_PR2_is_near_collision:
            positions = self.robot.positions()
            positions['rootJoint_pos_x'] = x + 0.04
            positions['rootJoint_pos_y'] = y + 0.04
            positions['rootJoint_pos_z'] = 0.
            positions['rootJoint_rot_z'] = th
            self.robot.set_positions(positions)
            self.dart_world.set_gown(self.robot_arm)
            close_to_collision = np.max([self.is_dart_in_collision(), close_to_collision])

            positions['rootJoint_pos_x'] = x - 0.04
            positions['rootJoint_pos_y'] = y + 0.04
            positions['rootJoint_pos_z'] = 0.
            positions['rootJoint_rot_z'] = th
            self.robot.set_positions(positions)
            self.dart_world.set_gown(self.robot_arm)
            close_to_collision = np.max([self.is_dart_in_collision(), close_to_collision])

            positions['rootJoint_pos_x'] = x - 0.04
            positions['rootJoint_pos_y'] = y - 0.04
            positions['rootJoint_pos_z'] = 0.
            positions['rootJoint_rot_z'] = th
            self.robot.set_positions(positions)
            self.dart_world.set_gown(self.robot_arm)
            close_to_collision = np.max([self.is_dart_in_collision(), close_to_collision])

            positions['rootJoint_pos_x'] = x + 0.04
            positions['rootJoint_pos_y'] = y - 0.04
            positions['rootJoint_pos_z'] = 0.
            positions['rootJoint_rot_z'] = th
            self.robot.set_positions(positions)
            self.dart_world.set_gown(self.robot_arm)
            close_to_collision = np.max([self.is_dart_in_collision(), close_to_collision])

            positions['rootJoint_pos_x'] = x
            positions['rootJoint_pos_y'] = y
            positions['rootJoint_pos_z'] = 0.
            positions['rootJoint_rot_z'] = th
            self.robot.set_positions(positions)
            self.dart_world.set_gown(self.robot_arm)

        if not close_to_collision:# and not in_collision:
            reached = np.zeros(len(self.origin_B_grasps))
            manip = np.zeros(len(self.origin_B_grasps))
            for num, origin_B_grasp in enumerate(self.origin_B_grasps):
                pr2_B_grasp = origin_B_pr2.I * origin_B_grasp
                # resp = self.call_ik_service(pr2_B_grasp, z)
                # sols = resp.ik_sols.data
                # print 'sols'
                # print sols
                # jacobians = resp.jacobians.data
                # jacobians = [float(i) for i in jacobians]
                # if len(sols) > 0:
                #     sols = np.reshape(sols, [len(sols)/7, 7])
                #     print 'sols reshape:'
                    # print sols
                    # jacobians = np.reshape(jacobians, [len(jacobians)/42, 6, 7])
                # else:
                #     sols = []
                #     jacobians = []
                # sols = self.manip.FindIKSolutions(pr2_B_grasp, filteroptions=op.IkFilterOptions.CheckEnvCollisions)
                sols, jacobians = self.ik_request(pr2_B_grasp, z)
                if list(sols):  # not None:
                    # print 'final sols'
                    # print sols
                    for i in xrange(len(sols)):
                        v = self.robot.q
                        v[self.robot_arm[0] + '_shoulder_pan_joint'] = sols[i][0]
                        v[self.robot_arm[0] + '_shoulder_lift_joint'] = sols[i][1]
                        v[self.robot_arm[0] + '_upper_arm_roll_joint'] = sols[i][2]
                        v[self.robot_arm[0] + '_elbow_flex_joint'] = sols[i][3]
                        v[self.robot_arm[0] + '_forearm_roll_joint'] = sols[i][4]
                        v[self.robot_arm[0] + '_wrist_flex_joint'] = sols[i][5]
                        v[self.robot_arm[0] + '_wrist_roll_joint'] = sols[i][6]
                        self.robot.set_positions(v)
                        self.dart_world.set_gown(self.robot_arm)
                        if not self.is_dart_in_collision():
                            reached[num] = 1.
                            J = np.matrix(jacobians[i])
                            try:
                                joint_limit_weight = self.gen_joint_limit_weight(sols[i], self.robot_arm)
                                manip[num] = np.max([copy.copy((m.pow(np.linalg.det(J*joint_limit_weight*J.T), (1./6.)))/(np.trace(J*joint_limit_weight*J.T)/6.)), manip[num]])
                            except ValueError:
                                print 'WARNING!!'
                                print 'Jacobian may be singular or close to singular'
                                print 'Determinant of J*JT is: ', np.linalg.det(J*J.T)
                                manip[num] = np.max([0., manip[num]])
                if self.visualize:
                    rospy.sleep(0.5)
            for num in xrange(len(reached)):
                manip_score += copy.copy(reached[num] * manip[num])
                reach_score += copy.copy(reached[num])
        else:
            # print 'In base collision! single config distance: ', distance
            if distance < 2.0:
                return 10. + 1. + (1.25 - distance)

        # self.human_model.SetActiveManipulator('leftarm')
        # self.human_manip = self.robot.GetActiveManipulator()
        # human_torques = self.human_manip.ComputeInverseDynamics([])
        # torque_cost = np.linalg.norm(human_torques)/10.

        # angle_cost = np.sum(np.abs(human_dof))
        # print 'len(self.goals)'
        # print len(self.goals)

        # print 'reached'
        # print reached

        reach_score /= len(self.goals)
        manip_score /= len(self.goals)

        # print 'reach_score'
        # print reach_score
        # print 'manip_score'
        # print manip_score

        # Set the weights for the different scores.
        beta = 10.  # Weight on number of reachable goals
        gamma = 1.  # Weight on manipulability of arm at each reachable goal
        zeta = 0.05  # Weight on torques
        if reach_score == 0.:
            return 10. + 2*random.random()
        else:
            # print 'Reach score: ', reach_score
            # print 'Manip score: ', manip_score
            if reach_score == 1.:
                if self.visualize:
                    rospy.sleep(2.0)
            return 10.-beta*reach_score-gamma*manip_score #+ zeta*angle_cost

    def is_dart_base_in_collision(self):
        self.dart_world.check_collision()
        for contact in self.dart_world.collision_result.contacts:
            if ((self.robot == contact.skel1 or self.robot == contact.skel2) and
                    (self.robot.bodynode('base_link') == contact.bodynode1
                     or self.robot.bodynode('base_link') == contact.bodynode2)):
                return True
        return False

    def is_dart_in_collision(self):
        self.dart_world.check_collision()
        # collided_bodies = self.dart_world.collision_result.contacted_bodies
        for contact in self.dart_world.collision_result.contacts:
            if ((self.robot == contact.skel1 or self.robot == contact.skel2) and
                    (self.human == contact.skel1 or self.human == contact.skel2)) or \
                    ((self.robot == contact.skel1 or self.robot == contact.skel2) and
                         (self.gown_leftarm == contact.skel1 or self.gown_leftarm == contact.skel2)) or \
                    ((self.robot == contact.skel1 or self.robot == contact.skel2) and
                         (self.gown_rightarm == contact.skel1 or self.gown_rightarm == contact.skel2)):
                return True
        return False

    def visualize_dart(self):
        win = pydart.gui.viewer.PydartWindow(self.dart_world)
        win.camera_event(1)
        win.set_capture_rate(10)
        win.run_application()

    def setup_dart(self, filename='fullbody_alex_capsule.skel'):
        # Setup Dart ENV
        pydart.init()
        print('pydart initialization OK')
        skel_file = self.pkg_path+'/models/'+filename
        # print skel_file
        self.dart_world = DartDressingWorld(skel_file)
        # print '1'
        # Lets you visualize dart.
        if self.visualize:
            t = threading.Thread(target=self.visualize_dart)
            t.start()
        # print '2'
        self.robot = self.dart_world.robot
        self.human = self.dart_world.human
        self.gown_leftarm = self.dart_world.gown_box_leftarm
        self.gown_rightarm = self.dart_world.gown_box_rightarm
        # print '3'
        sign_flip = 1.
        if 'right' in self.robot_arm:
            sign_flip = -1.
        v = self.robot.q
        v['l_shoulder_pan_joint'] = -sign_flip*3.14/2
        v['l_shoulder_lift_joint'] = -0.52
        v['l_upper_arm_roll_joint'] = 0.
        v['l_elbow_flex_joint'] = -3.14 * 2 / 3
        v['l_forearm_roll_joint'] = 0.
        v['l_wrist_flex_joint'] = 0.
        v['l_wrist_roll_joint'] = 0.
        v['l_gripper_l_finger_joint'] = .54
        v['r_shoulder_pan_joint'] = sign_flip*3.14/2
        v['r_shoulder_lift_joint'] = -0.52
        v['r_upper_arm_roll_joint'] = 0.
        v['r_elbow_flex_joint'] = -3.14*2/3
        v['r_forearm_roll_joint'] = 0.
        v['r_wrist_flex_joint'] = 0.
        v['r_wrist_roll_joint'] = 0.
        v['r_gripper_l_finger_joint'] = .54
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

    def setup_openrave(self):
        # Setup Openrave ENV
        op.RaveSetDebugLevel(op.DebugLevel.Error)
        InitOpenRAVELogging()
        self.env = op.Environment()

        self.check_self_collision = True

        # if self.visualize:
        #     self.env.SetViewer('qtcoin')

        self.env.Load('robots/pr2-beta-static.zae')
        self.op_robot = self.env.GetRobots()[0]
        self.op_robot.CheckLimitsAction = 2

        robot_start = np.matrix([[m.cos(0.), -m.sin(0.), 0., 0.],
                                 [m.sin(0.), m.cos(0.), 0., 0.],
                                 [0., 0., 1., 0.],
                                 [0., 0., 0., 1.]])
        self.op_robot.SetTransform(np.array(robot_start))

        self.goal_B_gripper = np.matrix([[0., 0., 1., 0.0],
                                         [0., 1., 0., 0.0],
                                         [-1., 0., 0., 0.0],
                                         [0., 0., 0., 1.0]])

        self.gripper_B_tool = np.matrix([[0., -1., 0., 0.03],
                                         [1., 0., 0., 0.0],
                                         [0., 0., 1., -0.05],
                                         [0., 0., 0., 1.0]])

        self.origin_B_grasp = None

        # self.set_openrave_arm(self.robot_opposite_arm)
        # self.set_openrave_arm(self.robot_arm)
        print 'Openrave IK is now ready'

    def setup_ik_service(self):
        print 'Looking for IK service.'
        # rospy.wait_for_service('ikfast_service')
        # print 'Found IK service.'
        # self.ik_service = rospy.ServiceProxy('ikfast_service', IKService, persistent=True)
        # print 'IK service is ready for use!'

    def ik_request(self, pr2_B_grasp, spine_height):
        with self.frame_lock:
            jacobians = []
            with self.env:
                v = self.op_robot.GetActiveDOFValues()
                v[self.op_robot.GetJoint('l_shoulder_pan_joint').GetDOFIndex()] = 3.14 / 2
                v[self.op_robot.GetJoint('l_shoulder_lift_joint').GetDOFIndex()] = -0.52
                v[self.op_robot.GetJoint('l_upper_arm_roll_joint').GetDOFIndex()] = 0.
                v[self.op_robot.GetJoint('l_elbow_flex_joint').GetDOFIndex()] = -3.14 * 2 / 3
                v[self.op_robot.GetJoint('l_forearm_roll_joint').GetDOFIndex()] = 0.
                v[self.op_robot.GetJoint('l_wrist_flex_joint').GetDOFIndex()] = 0.
                v[self.op_robot.GetJoint('l_wrist_roll_joint').GetDOFIndex()] = 0.
                v[self.op_robot.GetJoint('r_shoulder_pan_joint').GetDOFIndex()] = -3.14 / 2
                v[self.op_robot.GetJoint('r_shoulder_lift_joint').GetDOFIndex()] = -0.52
                v[self.op_robot.GetJoint('r_upper_arm_roll_joint').GetDOFIndex()] = 0.
                v[self.op_robot.GetJoint('r_elbow_flex_joint').GetDOFIndex()] = -3.14 * 2 / 3
                v[self.op_robot.GetJoint('r_forearm_roll_joint').GetDOFIndex()] = 0.
                v[self.op_robot.GetJoint('r_wrist_flex_joint').GetDOFIndex()] = 0.
                v[self.op_robot.GetJoint('r_wrist_roll_joint').GetDOFIndex()] = 0.
                v[self.op_robot.GetJoint('torso_lift_joint').GetDOFIndex()] = spine_height
                self.op_robot.SetActiveDOFValues(v, checklimits=2)
                self.env.UpdatePublishedBodies()

                # base_footprint_B_tool_goal = createBMatrix(goal_position, goal_orientation)

                origin_B_grasp = np.array(np.matrix(pr2_B_grasp)*self.goal_B_gripper)  # * self.gripper_B_tool.I * self.goal_B_gripper)
                # print 'here'
                # print self.origin_B_grasp
                # sols = self.manip.FindIKSolutions(self.origin_B_grasp,4)
                # init_time = rospy.Time.now()
                if self.check_self_collision:
                    sols = self.manip.FindIKSolutions(origin_B_grasp, filteroptions=op.IkFilterOptions.CheckEnvCollisions)
                else:
                    sols = self.manip.FindIKSolutions(origin_B_grasp, filteroptions=op.IkFilterOptions.IgnoreSelfCollisions)
                if list(sols):
                    with self.op_robot:
                        for sol in sols:
                            # self.robot.SetDOFValues(sol, self.manip.GetArmIndices(), checklimits=2)
                            self.op_robot.SetDOFValues(sol, self.manip.GetArmIndices())
                            self.env.UpdatePublishedBodies()
                            jacobians.append(np.vstack([self.manip.CalculateJacobian(), self.manip.CalculateAngularVelocityJacobian()]))
                            # if self.visualize:
                            #     rospy.sleep(1.5)
                    # print jacobians[0]
                    return sols, jacobians
                else:
                    return [], []

    def call_ik_service(self, Bmat, pr2_height):
        pos, quat = Bmat_to_pos_quat(Bmat)
        return self.ik_service(pos, quat, pr2_height, self.robot_arm)

    # Setup physx services and initialized the human in Physx
    def setup_physx(self):
        print 'Setting up services and looking for Physx services'

        # print 'Found Physx services'
        # self.physx_output_service = rospy.Service('body_physx_sleeve_output', PhysxOutput, self.simulator_result_handler)
        self.physx_output_service = rospy.Service('physx_output', PhysxOutput,
                                                  self.simulator_result_handler)
        rospy.wait_for_service('init_physx_body_model')
        self.init_physx_service = rospy.ServiceProxy('init_physx_body_model', InitPhysxBodyModel)
        rospy.wait_for_service('body_config_input_to_physx')
        self.update_physx_config_service = rospy.ServiceProxy('body_config_input_to_physx', PhysxInputWaypoints)
        # rospy.wait_for_service('body_init_physx_sleeve_body_model')
        # self.init_physx_service = rospy.ServiceProxy('body_init_physx_sleeve_body_model', InitPhysxBodyModel)
        # rospy.wait_for_service('body_sleeve_input_to_physx')
        # self.update_physx_config_service = rospy.ServiceProxy('body_sleeve_input_to_physx', PhysxInputWaypoints)

        self.traj_to_simulator_pub = rospy.Publisher('physx_simulator_input', FloatArrayBare, queue_size=1)
        # self.simulator_result_sub = rospy.Subscriber('physx_simulator_result', PhysxOutcome, self.simulator_result_cb)
        print 'Physx calls are ready!'

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
            q['j_bicep_left_roll'] = -1*0.
            q['j_forearm_left_1'] = dof[3]
            q['j_forearm_left_2'] = 0.
        elif human_arm == 'rightarm':
            q['j_bicep_right_x'] = -1*dof[0]
            q['j_bicep_right_y'] = dof[1]
            q['j_bicep_right_z'] = dof[2]
            q['j_bicep_right_roll'] = 0.
            q['j_forearm_right_1'] = dof[3]
            q['j_forearm_right_2'] = 0.
        else:
            print 'I am not sure what arm to set the dof for.'
            return False
        self.human.set_positions(q)

    def update_physx_from_dart(self,
                               origin_B_forearm_pointed_down_arm=[],
                               origin_B_upperarm_pointed_down_shoulder=[],
                               traj_start=[], traj_forearm_end=[],
                               traj_upper_end=[], traj_final_end=[],
                               initialization=False):
        links = []
        spheres = []
        for bodypart in self.dart_world.skel_bodies:
            if 'visualization_shape' in bodypart.keys():
                if 'multi_sphere' in bodypart['visualization_shape']['geometry'].keys():
                    multisphere = bodypart['visualization_shape']['geometry']['multi_sphere']['sphere']
                    first_sphere_transform = np.eye(4)
                    first_sphere_transform[0:3, 3] = np.array([float(t) for t in multisphere[0]['position'].split(' ')])
                    transform = np.matrix(self.human.bodynode(bodypart['@name']).world_transform()) * np.matrix(
                        first_sphere_transform)
                    position = np.round(
                        copy.copy(np.array(transform)[0:3, 3]) - self.dart_world.human_reference_center_floor_point, 10)
                    radius = float(np.round(float(multisphere[0]['radius']), 10))
                    sphere_data = copy.copy(list(flatten([position, radius])))
                    if sphere_data not in spheres:
                        sphere_1_index = len(spheres)
                        spheres.append(copy.copy(list(flatten([position, radius]))))
                    else:
                        sphere_1_index = spheres.index(sphere_data)

                    second_sphere_transform = np.eye(4)
                    second_sphere_transform[0:3, 3] = np.array(
                        [float(t) for t in multisphere[1]['position'].split(' ')])
                    transform = np.matrix(self.human.bodynode(bodypart['@name']).world_transform()) * np.matrix(
                        second_sphere_transform)
                    position = np.round(
                        copy.copy(np.array(transform)[0:3, 3]) - self.dart_world.human_reference_center_floor_point, 10)
                    radius = float(np.round(float(multisphere[1]['radius']), 10))
                    sphere_data = copy.copy(list(flatten([position, radius])))
                    if sphere_data not in spheres:
                        sphere_2_index = len(spheres)
                        spheres.append(copy.copy(list(flatten([position, radius]))))
                    else:
                        print 'the sphere was already there!!'
                        sphere_2_index = spheres.index(sphere_data)
                    links.append([sphere_1_index, sphere_2_index])
        spheres = np.array(spheres)
        links = np.array(links)
        spheres_x = [float(i) for i in spheres[:, 0]]
        spheres_y = [float(i) for i in spheres[:, 1]]
        spheres_z = [float(i) for i in spheres[:, 2]]
        spheres_r = [float(i) for i in spheres[:, 3]]
        first_sphere_list = [int(i) for i in links[:, 0]]
        second_sphere_list = [int(i) for i in links[:, 1]]
        if initialization:
            resp = self.init_physx_service(spheres_x, spheres_y, spheres_z, spheres_r,
                                           first_sphere_list, second_sphere_list)
            print 'Is physx initialized with the person model?', resp
        else:
            if traj_start == []:
                traj_start = self.traj_start
            if traj_forearm_end == []:
                traj_forearm_end = self.traj_forearm_end
            if traj_forearm_end == [] or traj_start == [] or origin_B_upperarm_pointed_down_shoulder == []:
                print 'I am missing a start trajectory, an end trajectory, or the transform from origin to pointing ' \
                      'down shoulder'
                return False
            origin_B_forearm_pointed_down_arm = list(flatten(np.array(origin_B_forearm_pointed_down_arm)))
            origin_B_upperarm_pointed_down_shoulder = list(flatten(np.array(origin_B_upperarm_pointed_down_shoulder)))
            traj_start = list(flatten(np.array(traj_start)))
            traj_forearm_end = list(flatten(np.array(traj_forearm_end)))
            traj_upper_end = list(flatten(np.array(traj_upper_end)))
            traj_final_end = list(flatten(np.array(traj_final_end)))
            print 'right here'
            resp = self.update_physx_config_service(spheres_x, spheres_y, spheres_z, spheres_r, first_sphere_list,
                                                    second_sphere_list, traj_start, traj_forearm_end, traj_upper_end,
                                                    traj_final_end, origin_B_forearm_pointed_down_arm,
                                                    origin_B_upperarm_pointed_down_shoulder)
        print 'Physx update was successful? ', resp
        return resp

    def get_best_traj_offset(self):
        return 0.0, 0.1

    def gen_joint_limit_weight(self, q, side):
        # define the total range limit for each joint
        if 'left' in side:
            joint_min = np.array([-40., -30., -44., -133., -400., -130., -400.])
            joint_max = np.array([130., 80., 224., 0., 400., 0., 400.])
        elif 'right' in side:
            # print 'Need to check the joint limits for the right arm'
            joint_min = np.array([-130., -30., -224., -133., -400., -130., -400.])
            joint_max = np.array([40., 80., 44., 0., 400., 0., 400.])
        joint_range = joint_max - joint_min
        # l1_max = 40.
        # l2_min = -30.
        # l2_max = 80.
        # l3_min = -44.
        # l3_max = 224.
        # l4_min = 0.
        # l4_max = 133.
        # l5_min = -1000.  # continuous
        # l5_max = 1000.  # continuous
        # l6_min = 0.
        # l6_max = 130.
        # l7_min = -1000.  # continuous
        # l7_max = 1000  # continuous
        # l2 = 120.
        # l3 = 268.
        # l4 = 133.
        # l5 = 10000.  # continuous
        # l6 = 130.
        # l7 = 10000.  # continuous

        weights = np.zeros(7)
        for joint in xrange(len(weights)):
            weights[joint] = (1. - m.pow(0.5, ((joint_range[joint])/2. - np.abs((joint_range[joint])/2. - q[joint] + joint_min[joint]))/(joint_range[joint]/40.)))
        weights[4] = 1.
        weights[6] = 1.
        return np.diag(weights)

if __name__ == "__main__":
    rospy.init_node('dressing_sleeve_body_test')
    # start_time = time.time()
    outer_start_time = rospy.Time.now()
    selector = ScoreGeneratorDressingwithPhysx(human_arm='leftarm', visualize=False)
    # selector.visualize_many_configurations()
    # selector.output_results_for_use()
    selector.run_interleaving_optimization_outer_level()
    outer_elapsed_time = rospy.Time.now()-outer_start_time
    print 'Everything is complete!'
    print 'Done with optimization. Total time elapsed:', outer_elapsed_time.to_sec()
    # rospy.spin()

    #selector.choose_task(mytask)
    # score_sheet = selector.handle_score_generation()

    # print 'Time to load find generate all scores: %fs'%(time.time()-start_time)

    # rospack = rospkg.RosPack()
    # pkg_path = rospack.get_path('hrl_base_selection')
    # save_pickle(score_sheet, ''.join([pkg_path, '/data/', mymodel, '_', mytask, '.pkl']))
    # print 'Time to complete program, saving all data: %fs' % (time.time()-start_time)






