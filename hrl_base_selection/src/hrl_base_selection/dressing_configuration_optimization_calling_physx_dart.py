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
from hrl_base_selection.graph_search_functions import SimpleGraph, a_star_search, reconstruct_path
from hrl_base_selection.msg import PhysxOutcome
from hrl_base_selection.srv import InitPhysxBodyModel, PhysxInput, IKService, PhysxOutput, PhysxInputWaypoints

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from hrl_msgs.msg import FloatArrayBare

import random, threading

import openravepy as op
from openravepy.misc import InitOpenRAVELogging

from sklearn.neighbors import NearestNeighbors

import tf.transformations as tft

import pickle as pkl
roslib.load_manifest('hrl_lib')
from hrl_lib.util import save_pickle, load_pickle

import gc

import cma


class ScoreGeneratorDressingwithPhysx(object):

    def __init__(self, robot_arm='rightarm', human_arm='rightarm', visualize=False, standard_mode=True):

        self.visualize = visualize
        self.frame_lock = threading.RLock()

        self.robot_arm = None
        self.robot_opposite_arm = None

        self.human_arm = None
        self.human_opposite_arm = None

        self.optimization_results = dict()

        self.start_traj = []
        self.end_traj = []

        self.axis = []
        self.angle = None

        rospack = rospkg.RosPack()
        self.pkg_path = rospack.get_path('hrl_base_selection')

        self.stretch_allowable = []

        self.human_rot_correction = None

        self.fixed_points = []
        self.add_new_fixed_point = False
        self.fixed_points_to_use = []

        # self.model = None
        self.force_cost = 0.

        self.goals = None
        self.pr2_B_reference = None
        self.task = None
        self.task_dict = None

        self.reference_names = None

        self.best_physx_score = dict() #100.
        self.best_pr2_results = dict()

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

        self.standard_mode = standard_mode

        if self.standard_mode:
            self.setup_openrave()

            self.set_robot_arm(robot_arm)
            self.set_human_arm(human_arm)

            self.setup_dart(filename='fullbody_50percentile_capsule.skel')

            self.arm_configs_eval = load_pickle(rospack.get_path('hrl_dressing') +
                                               '/data/forearm_trajectory_evaluation/entire_results_list.pkl')
            self.arm_configs_checked = []
            for line in self.arm_configs_eval:
                self.arm_configs_checked.append(line[0:4])
            self.arm_knn = NearestNeighbors(8, m.radians(15.))
            self.arm_knn.fit(self.arm_configs_checked)

            # self.setup_ik_service()
            self.setup_physx()
            self.update_physx_from_dart(initialization=True)
            # self.setup_physx_calls()

    def output_results_for_use(self):
        traj_and_arm_config_results = load_pickle(self.pkg_path+'/data/saved_results/large_search_space/best_trajectory_and_arm_config.pkl')
        pr2_config_results = load_pickle(self.pkg_path+'/data/saved_results/large_search_space/best_pr2_config.pkl')
        traj_and_arm_config = traj_and_arm_config_results[0]
        params = traj_and_arm_config
        pr2_config = pr2_config_results[0]
        self.set_human_model_dof([traj_and_arm_config[7], traj_and_arm_config[8], -traj_and_arm_config[9], traj_and_arm_config[6], 0, 0, 0], 'rightarm', 'green_kevin')

        uabl = self.human_model.GetLink('green_kevin/arm_left_base_link')
        uabr = self.human_model.GetLink('green_kevin/arm_right_base_link')
        ual = self.human_model.GetLink('green_kevin/arm_left_link')
        uar = self.human_model.GetLink('green_kevin/arm_right_link')
        fal = self.human_model.GetLink('green_kevin/forearm_left_link')
        far = self.human_model.GetLink('green_kevin/forearm_right_link')
        hl = self.human_model.GetLink('green_kevin/hand_left_link')
        hr = self.human_model.GetLink('green_kevin/hand_right_link')
        origin_B_uabl = np.matrix(uabl.GetTransform())
        origin_B_uabr = np.matrix(uabr.GetTransform())
        origin_B_ual = np.matrix(ual.GetTransform())
        origin_B_uar = np.matrix(uar.GetTransform())
        origin_B_fal = np.matrix(fal.GetTransform())
        origin_B_far = np.matrix(far.GetTransform())
        origin_B_hl = np.matrix(hl.GetTransform())
        origin_B_hr = np.matrix(hr.GetTransform())

        uabr_B_uabr_corrected = np.matrix([[ 0.,  0., -1., 0.],
                                           [ 1.,  0.,  0., 0.],
                                           [ 0., -1.,  0., 0.],
                                           [ 0.,  0.,  0., 1.]])

        z_origin = np.array([0., 0., 1.])
        x_vector = np.reshape(np.array(origin_B_hr[0:3, 0]), [1, 3])[0]
        y_orth = np.cross(z_origin, x_vector)
        y_orth = y_orth/np.linalg.norm(y_orth)
        z_orth = np.cross(x_vector, y_orth)
        z_orth = z_orth/np.linalg.norm(z_orth)
        origin_B_hr_rotated = np.matrix(np.eye(4))
        # print 'x_vector'
        # print x_vector
        # print 'origin_B_hr_rotated'
        # print origin_B_hr_rotated
        # print 'np.reshape(x_vector, [3, 1])'
        # print np.reshape(x_vector, [3, 1])
        # print 'origin_B_hr_rotated[0:3, 0]'
        # print origin_B_hr_rotated[0:3, 0]
        origin_B_hr_rotated[0:3, 0] = copy.copy(np.reshape(x_vector, [3, 1]))
        origin_B_hr_rotated[0:3, 1] = copy.copy(np.reshape(y_orth, [3, 1]))
        origin_B_hr_rotated[0:3, 2] = copy.copy(np.reshape(z_orth, [3, 1]))
        origin_B_hr_rotated[0:3, 3] = copy.copy(origin_B_hr[0:3, 3])
        origin_B_hr_rotated = np.matrix(origin_B_hr_rotated)

        hr_rotated_B_traj_start_pos = np.matrix(np.eye(4))
        hr_rotated_B_traj_start_pos[0:3, 3] = copy.copy(np.reshape([params[0:3]], [3, 1]))
        hr_rotated_B_traj_start_pos[0, 3] = hr_rotated_B_traj_start_pos[0, 3] + 0.07

        origin_B_traj_start_pos = origin_B_hr_rotated*hr_rotated_B_traj_start_pos

        # print 'origin_B_traj_start_pos'
        # print origin_B_traj_start_pos

        # origin_B_world_rotated_shoulder = createBMatrix(np.reshape(np.array(origin_B_uabr[0:3, 3]), [1, 3])[0], list(tft.quaternion_from_euler(params[7], -params[8], params[9], 'rzxy')))

        origin_B_world_rotated_shoulder = origin_B_uar*uabr_B_uabr_corrected

        # Because green kevin has the upper with a bend in it, I shift the shoulder location by that bend offset.
        shoulder_origin_B_should_origin_shifted_green_kevin = np.matrix(np.eye(4))
        shoulder_origin_B_should_origin_shifted_green_kevin[1, 3] = -0.04953
        origin_B_world_rotated_shoulder = origin_B_world_rotated_shoulder*shoulder_origin_B_should_origin_shifted_green_kevin

        origin_B_uabr[0:3, 3] = origin_B_world_rotated_shoulder[0:3, 3]
        origin_B_uar[0:3, 3] = origin_B_world_rotated_shoulder[0:3, 3]

        z_origin = np.array([0., 0., 1.])
        x_vector = np.reshape(np.array(-1*origin_B_world_rotated_shoulder[0:3, 2]), [1, 3])[0]
        y_orth = np.cross(z_origin, x_vector)
        y_orth = y_orth/np.linalg.norm(y_orth)
        z_orth = np.cross(x_vector, y_orth)
        z_orth = z_orth/np.linalg.norm(z_orth)
        origin_B_rotated_pointed_down_shoulder = np.matrix(np.eye(4))
        origin_B_rotated_pointed_down_shoulder[0:3, 0] = np.reshape(x_vector, [3, 1])
        origin_B_rotated_pointed_down_shoulder[0:3, 1] = np.reshape(y_orth, [3, 1])
        origin_B_rotated_pointed_down_shoulder[0:3, 2] = np.reshape(z_orth, [3, 1])
        origin_B_rotated_pointed_down_shoulder[0:3, 3] = origin_B_uabr[0:3, 3]
        # origin_B_rotated_pointed_down_shoulder = origin_B_uabr*uabr_B_uabr_corrected*np.matrix(shoulder_origin_B_rotated_pointed_down_shoulder)
        # print 'origin_B_rotated_pointed_down_shoulder'
        # print origin_B_rotated_pointed_down_shoulder

        # origin_B_rotated_pointed_down_shoulder[0:3, 3] = origin_B_world_rotated_shoulder[0:3, 3]

        rotated_pointed_down_shoulder_B_traj_end_pos = np.matrix(np.eye(4))
        rotated_pointed_down_shoulder_B_traj_end_pos[0:3, 3] = copy.copy(np.reshape([params[3:6]], [3, 1]))

        # print 'rotated_pointed_down_shoulder_B_traj_end_pos'
        # print rotated_pointed_down_shoulder_B_traj_end_pos
        # print 'origin_B_traj_start_pos'
        # print origin_B_traj_start_pos
        origin_B_traj_end_pos = origin_B_rotated_pointed_down_shoulder*rotated_pointed_down_shoulder_B_traj_end_pos
        # print 'origin_B_traj_end_pos'
        # print origin_B_traj_end_pos

        # print 'origin_B_uabr_corrected'
        # print origin_B_uabr*uabr_B_uabr_corrected

        th = m.radians(180.)
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

        z_origin = np.array([0., 0., 1.])
        x_vector = np.reshape(np.array(origin_B_traj_end_pos[0:3, 3] - origin_B_traj_start_pos[0:3, 3]), [1, 3])[0]
        x_vector = x_vector/np.linalg.norm(x_vector)
        y_orth = np.cross(z_origin, x_vector)
        y_orth = y_orth/np.linalg.norm(y_orth)
        z_orth = np.cross(x_vector, y_orth)
        z_orth = z_orth/np.linalg.norm(z_orth)
        origin_B_traj_start = np.matrix(np.eye(4))
        origin_B_traj_start[0:3, 0] = np.reshape(x_vector, [3, 1])
        origin_B_traj_start[0:3, 1] = np.reshape(y_orth, [3, 1])
        origin_B_traj_start[0:3, 2] = np.reshape(z_orth, [3, 1])
        origin_B_traj_start[0:3, 3] = copy.copy(origin_B_traj_start_pos[0:3, 3])
        translation, quaternion = Bmat_to_pos_quat(origin_B_traj_start)
        self.axis, self.angle = calc_axis_angle(quaternion)

        # print 'origin_B_traj_start'
        # print origin_B_traj_start

        path_distance = np.linalg.norm(np.reshape(np.array(origin_B_traj_end_pos[0:3, 3] - origin_B_traj_start_pos[0:3, 3]), [1, 3])[0])
        # print 'path_distance'
        # print path_distance
        uabr_corrected_B_traj_start = uabr_B_uabr_corrected.I*origin_B_uabr.I*origin_B_traj_start
        # test_world_shoulder_B_sleeve_start_rotz = np.matrix([[ m.cos(th), -m.sin(th),     0.],
        #                                                      [ m.sin(th),  m.cos(th),     0.],
        #                                                       [        0.,         0.,     1.]])
        # hr_rotated_B_traj_start = createBMatrix([params[0], params[1], params[2]],
        #                                   tft.quaternion_from_euler(params[3], params[4], params[5], 'rzyx'))
        pos_t, quat_t = Bmat_to_pos_quat(uabr_corrected_B_traj_start)

        path_waypoints = np.arange(0., path_distance+path_distance*0.01, path_distance/5.)

        self.goals = []
        for goal in path_waypoints:
            traj_start_B_traj_waypoint = np.matrix(np.eye(4))
            traj_start_B_traj_waypoint[0, 3] = goal
            origin_B_traj_waypoint = origin_B_traj_start*traj_start_B_traj_waypoint
            self.goals.append(copy.copy(origin_B_traj_waypoint))

        self.origin_B_grasps = []
        for num in xrange(len(self.goals)):
            self.origin_B_grasps.append(np.array(np.matrix(self.goals[num])*np.matrix(self.gripper_B_tool.I)))

        x = pr2_config[0]
        y = pr2_config[1]
        th = pr2_config[2]
        z = pr2_config[3]

        origin_B_pr2 = np.matrix([[ m.cos(th), -m.sin(th),     0.,         x],
                                  [ m.sin(th),  m.cos(th),     0.,         y],
                                  [        0.,         0.,     1.,        0.],
                                  [        0.,         0.,     0.,        1.]])

        self.robot.SetTransform(np.array(origin_B_pr2))
        v = self.robot.GetActiveDOFValues()
        v[self.robot.GetJoint('torso_lift_joint').GetDOFIndex()] = z
        self.robot.SetActiveDOFValues(v, 2)
        self.env.UpdatePublishedBodies()

        self.pr2_B_grasps = []
        for num in xrange(len(self.origin_B_grasps)):
            self.pr2_B_grasps.append(origin_B_pr2.I*np.matrix(self.origin_B_grasps[num]))
        # self.pr2_B_grasps.append(np.matrix(np.eye(4)))
        # poseArray = PoseArray()
        # poseArray.header.stamp = rospy.Time.now()
        # poseArray.header.frame_id = 'base_footprint'
        # for pose in self.pr2_B_grasps:
        #     newPose = Pose()
        #     pos_g, ori_g = Bmat_to_pos_quat(pose)
        #     newPose.position.x = pos_g[0]
        #     newPose.position.y = pos_g[1]
        #     newPose.position.z = pos_g[2]
        #     newPose.orientation.x = ori_g[0]
        #     newPose.orientation.y = ori_g[1]
        #     newPose.orientation.z = ori_g[2]
        #     newPose.orientation.w = ori_g[3]
        #     poseArray.poses.append(copy.copy(newPose))

        save_pickle(self.pr2_B_grasps, self.pkg_path+'/data/saved_results/large_search_space/pr2_grasps.pkl')
        print 'Saved!'
        rospy.spin()

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
            print 'Physx simulation outcome (single-contact rays): ', self.physx_outcome
            print 'Physx simulation outcome (point-in-polygon): ', self.physx_outcome_method2
            return True

    def optimize_entire_dressing_task(self):
        self.set_robot_arm('rightarm')
        subtask_list = ['rightarm', 'leftarm']

        self.fixed_points = []

        self.final_results = []
        self.final_results.append(['subtask', 'overall_score', 'arm_config', 'physx_score', 'pr2_config', 'kinematics_score'])
        for subtask_number, subtask in enumerate(subtask_list):
            self.final_results.append([subtask, '', '', '', '', ''])
            if 'right' in subtask or 'left' in subtask:
                self.set_human_arm(subtask)
            # self.best_pr2_results[subtask_number] = [[], []]
            if subtask_number == 0:
                self.fixed_points_to_use = []
                self.stretch_allowable = []
                self.add_new_fixed_point = True
                self.run_interleaving_optimization_outer_level(subtask=subtask, subtask_step=subtask_number,
                                                               maxiter=5, popsize=5)
            else:
                if subtask_number == 1:
                    self.fixed_points_to_use = [0]
                    self.stretch_allowable = [0.5]
                    self.add_new_fixed_point = True
                self.run_interleaving_optimization_outer_level(subtask=subtask, subtask_step=subtask_number,
                                                               maxiter=500, popsize=50)

    def run_interleaving_optimization_outer_level(self, maxiter=1000, popsize=40, subtask='', subtask_step=0):
        self.subtask_step = subtask_step
        # self.best_overall_score = dict()
        self.best_overall_score = 10000.
        # self.best_physx_config = dict()
        self.best_physx_config = None
        self.best_physx_score = 10000.
        # self.best_kinematics_config = dict()
        self.best_kinematics_config = None
        self.best_kinematics_score = 10000.

        # maxiter = 30/
        # popsize = m.pow(5, 2)*100
        # maxiter = 8
        # popsize = 40

        ### Current: Two positions, first with respect to the fist, second with respect to the upper arm, centered at
        # the shoulder and pointing X down the upper arm
        # cma parameters: [human_upper_arm_quaternion(euler:xzy): r, y, p
        #                  human_arm_elbow_angle]

        parameters_min = np.array([m.radians(-5.), m.radians(-10.), m.radians(-10.),
                                   0.])
        parameters_max = np.array([m.radians(100.), m.radians(100.), m.radians(100),
                                   m.radians(135.)])
        parameters_scaling = (parameters_max - parameters_min) / 8.
        # parameters_initialization = (parameters_max + parameters_min) / 2.
        init_start_arm_configs = [[m.radians(0.), m.radians(0.), m.radians(0.), m.radians(0.)],
                                  [m.radians(45.), m.radians(0.), m.radians(0.), m.radians(0.)],
                                  [m.radians(0.), m.radians(45.), m.radians(0.), m.radians(0.)],
                                  [m.radians(0.), m.radians(0.), m.radians(45.), m.radians(0.)],
                                  [m.radians(0.), m.radians(0.), m.radians(0.), m.radians(45.)],
                                  [0.9679925, 0.18266905, 0.87995157, 0.77562143]]
        opts1 = {'seed': 1234, 'ftarget': -1., 'popsize': popsize, 'maxiter': maxiter,
                 'maxfevals': 1e8, 'CMA_cmean': 0.25, 'tolfun': 1e-3,
                 'tolfunhist': 1e-12, 'tolx': 5e-4,
                 'maxstd': 4.0, 'tolstagnation': 100,
                 'verb_filenameprefix': 'outcma_arm_and_trajectory',
                 'scaling_of_variables': list(parameters_scaling),
                 'bounds': [list(parameters_min), list(parameters_max)]}
        for init_start_arm_config in init_start_arm_configs:
            parameters_initialization = init_start_arm_config
            # parameters_initialization[0] = m.radians(0.)
            # parameters_initialization[1] = m.radians(70.)
            # parameters_initialization[2] = m.radians(0.)
            # parameters_initialization[3] = m.radians(0.)

            # optimization_results[<model>, <number_of_configs>, <head_rest_angle>, <headx>, <heady>, <allow_bed_movement>]
            self.optimization_results = cma.fmin(self.objective_function_traj_and_arm_config,
                                                          list(parameters_initialization),
                                                          1.,
                                                          options=opts1)
            print 'raw cma optimization results:\n',self.optimization_results
            # self.optimization_results = [self.best_config, self.best_score]

        # print 'Outcome is: '
        # print self.optimization_results
        # print 'Best arm config for ',subtask, 'subtask: \n', self.optimization_results[self.subtask_step][0]
        # print 'Associated score: ', self.optimization_results[self.subtask_step][1]
        # print 'Best PR2 configuration: \n', self.best_pr2_results[self.subtask_step][0]
        # print 'Associated score: ', self.best_pr2_results[self.subtask_step][1]
        print 'Best overall score for ', subtask, 'subtask: \n', self.best_overall_score
        print 'Best arm config for ', subtask, 'subtask: \n', self.best_physx_config
        print 'Associated score: ', self.best_physx_score
        print 'Best PR2 configuration: \n', self.best_kinematics_config
        print 'Associated score: ', self.best_kinematics_score
        self.final_results[subtask_step+1] = [subtask, self.best_overall_score,
                                              self.best_physx_config,
                                              self.best_physx_score,
                                              self.best_kinematics_config,
                                              self.best_kinematics_score]
        # optimized_traj_arm_output = []
        # for key in self.optimization_results.keys():
        #     optimized_traj_arm_output.append([self.optimization_results[key][0], self.optimization_results[key][1]])
        # optimized_pr2_output = []
        # for key in self.best_pr2_results.keys():
        #     optimized_pr2_output.append([self.best_pr2_results[key][0], self.best_pr2_results[key][1]])

        # save_pickle(self.final_results, self.pkg_path+'/data/best_trajectory_and_arm_config.pkl')
        save_pickle(self.final_results, self.pkg_path+'/data/dressing_results.pkl')

    def find_reference_coordinate_frames_and_goals(self, arm):
        skeleton_frame_B_worldframe = np.matrix([[1., 0., 0., 0.],
                                                 [0., 0., 1., 0.],
                                                 [0., -1., 0., 0.],
                                                 [0., 0., 0., 1.]])

        origin_B_pelvis = np.matrix(self.human.bodynode('h_pelvis').world_transform())
        origin_B_upperarmbase = np.matrix(self.human.bodynode('h_bicep_' + arm).world_transform())
        origin_B_upperarmbase[0:3, 0:3] = origin_B_pelvis[0:3, 0:3]
        origin_B_upperarm = np.matrix(self.human.bodynode('h_bicep_' + arm).world_transform())
        origin_B_forearm = np.matrix(self.human.bodynode('h_forearm_' + arm).world_transform())
        origin_B_wrist = np.matrix(self.human.bodynode('h_hand_' + arm).world_transform())
        origin_B_hand = np.matrix(self.human.bodynode('h_hand_' + arm+'2').world_transform())
        #print 'origin_B_upperarm\n', origin_B_upperarm
        z_origin = np.array([0., 0., 1.])
        x_vector = (-1 * np.array(origin_B_hand)[0:3, 1])
        x_vector /= np.linalg.norm(x_vector)
        y_orth = np.cross(z_origin, x_vector)
        y_orth /= np.linalg.norm(y_orth)
        z_orth = np.cross(x_vector, y_orth)
        z_orth /= np.linalg.norm(z_orth)
        origin_B_hand_rotated = np.eye(4)

        origin_B_hand_rotated[0:3, 0] = copy.copy(x_vector)
        origin_B_hand_rotated[0:3, 1] = copy.copy(y_orth)
        origin_B_hand_rotated[0:3, 2] = copy.copy(z_orth)
        origin_B_hand_rotated[0:3, 3] = copy.copy(np.array(origin_B_hand)[0:3, 3])
        origin_B_hand_rotated = np.matrix(origin_B_hand_rotated)

        rev = m.radians(180.)

        traj_y_offset, traj_z_offset = self.get_best_traj_offset()

        hand_rotated_B_traj_start_pos = np.matrix([[m.cos(rev), -m.sin(rev), 0., 0.042],
                                                   [m.sin(rev), m.cos(rev), 0., traj_y_offset],
                                                   [0., 0., 1., traj_z_offset],
                                                   [0., 0., 0., 1.]])

        origin_B_traj_start_pos = origin_B_hand_rotated * hand_rotated_B_traj_start_pos

        origin_B_upperarm_world = origin_B_upperarm * skeleton_frame_B_worldframe
        origin_B_forearm_world = origin_B_forearm * skeleton_frame_B_worldframe

        origin_B_forearm_pointed_down_arm = np.eye(4)
        z_origin = np.array([0., 0., 1.])
        x_vector = -1 * np.array(origin_B_forearm_world)[0:3, 2]
        x_vector /= np.linalg.norm(x_vector)
        if np.abs(x_vector[2]) > 0.99:
            x_vector = np.array([0., 0., np.sign(x_vector[2]) * 1.])
            y_orth = np.array([np.sign(x_vector[2]) * -1., 0., 0.])
            z_orth = np.array([0., np.sign(x_vector[2]) * 1., 0.])
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

        origin_B_reference_coordinates = np.eye(4)
        x_horizontal = np.cross(y_orth, z_origin)
        x_horizontal /= np.linalg.norm(x_horizontal)
        origin_B_reference_coordinates[0:3, 0] = x_horizontal
        origin_B_reference_coordinates[0:3, 1] = y_orth
        origin_B_reference_coordinates[0:3, 2] = z_origin
        origin_B_reference_coordinates[0:3, 3] = np.array(origin_B_forearm_world)[0:3, 3]
        origin_B_reference_coordinates = np.matrix(origin_B_reference_coordinates)
        horizontal_B_forearm_pointed_down = origin_B_reference_coordinates.I * origin_B_forearm_pointed_down_arm
        angle_from_horizontal = m.degrees(m.acos(horizontal_B_forearm_pointed_down[0, 0]))

        forearm_pointed_down_arm_B_traj_end_pos = np.eye(4)
        forearm_pointed_down_arm_B_traj_end_pos[0:3, 3] = [-0.05, traj_y_offset, traj_z_offset]
        forearm_pointed_down_arm_B_traj_end_pos = np.matrix(forearm_pointed_down_arm_B_traj_end_pos)
        forearm_pointed_down_arm_B_elbow_reference = np.matrix([[m.cos(rev), -m.sin(rev), 0., 0.0],
                                                                [m.sin(rev), m.cos(rev), 0., traj_y_offset],
                                                                [0., 0., 1., traj_z_offset],
                                                                [0., 0., 0., 1.]])
        origin_B_elbow_reference = origin_B_forearm_pointed_down_arm * forearm_pointed_down_arm_B_elbow_reference
        rev = m.radians(180.)
        forearm_pointed_down_arm_B_traj_end = np.matrix([[m.cos(rev), -m.sin(rev), 0., -0.03],
                                                         [m.sin(rev), m.cos(rev), 0., traj_y_offset],
                                                         [0., 0., 1., traj_z_offset],
                                                         [0., 0., 0., 1.]])
        origin_B_traj_forearm_end = origin_B_forearm_pointed_down_arm * forearm_pointed_down_arm_B_traj_end

        origin_B_upperarm_pointed_down_shoulder = np.eye(4)
        z_origin = np.array([0., 0., 1.])
        x_vector = -1 * np.array(origin_B_upperarm_world)[0:3, 2]
        x_vector /= np.linalg.norm(x_vector)
        if np.abs(x_vector[2]) > 0.99:
            x_vector = np.array([0., 0., np.sign(x_vector[2]) * 1.])
            y_orth = np.array([np.sign(x_vector[2]) * -1., 0., 0.])
            z_orth = np.array([0., np.sign(x_vector[2]) * 1., 0.])
        else:
            y_orth = np.cross(z_origin, x_vector)
            y_orth = y_orth / np.linalg.norm(y_orth)
            z_orth = np.cross(x_vector, y_orth)
            z_orth = z_orth / np.linalg.norm(z_orth)
        origin_B_upperarm_pointed_down_shoulder[0:3, 0] = x_vector
        origin_B_upperarm_pointed_down_shoulder[0:3, 1] = y_orth
        origin_B_upperarm_pointed_down_shoulder[0:3, 2] = z_orth
        origin_B_upperarm_pointed_down_shoulder[0:3, 3] = np.array(origin_B_upperarm_world)[0:3, 3]
        origin_B_rotated_pointed_down_shoulder = np.matrix(origin_B_upperarm_pointed_down_shoulder)

        upperarm_pointed_down_shoulder_B_traj_end_pos = np.eye(4)
        upperarm_pointed_down_shoulder_B_traj_end_pos[0:3, 3] = [-0.05, traj_y_offset, traj_z_offset]
        upperarm_pointed_down_shoulder_B_traj_end_pos = np.matrix(upperarm_pointed_down_shoulder_B_traj_end_pos)
        rev = m.radians(180.)
        upperarm_pointed_down_shoulder_B_traj_upper_end = np.matrix([[m.cos(rev), -m.sin(rev), 0., -0.05],
                                                                     [m.sin(rev), m.cos(rev), 0., -0.0],
                                                                     [0., 0., 1., traj_z_offset],
                                                                     [0., 0., 0., 1.]])

        origin_B_traj_upper_end = origin_B_upperarm_pointed_down_shoulder * upperarm_pointed_down_shoulder_B_traj_upper_end

        origin_B_traj_upper_start = copy.copy(origin_B_elbow_reference)
        origin_B_traj_upper_start[0:3, 0:3] = origin_B_traj_upper_end[0:3, 0:3]

        forearm_B_upper_arm = origin_B_elbow_reference.I*origin_B_traj_upper_start

        # Calculation of goal at the top of the shoulder. Parallel to the ground, but pointing opposite direction of
        # the upper arm.
        # print 'origin_B_traj_upper_end\n', origin_B_traj_upper_end
        origin_B_traj_final_end = np.eye(4)
        z_vector = np.array([0., 0., 1.])
        original_x_vector = np.array(origin_B_traj_upper_end)[0:3, 0]
        # x_vector /= np.linalg.norm(x_vector)
        y_orth = np.cross(z_vector, original_x_vector)
        y_orth = y_orth / np.linalg.norm(y_orth)
        x_vector = np.cross(y_orth, z_vector)
        x_vector = x_vector / np.linalg.norm(x_vector)
        origin_B_traj_final_end[0:3, 0] = x_vector
        origin_B_traj_final_end[0:3, 1] = y_orth
        origin_B_traj_final_end[0:3, 2] = z_vector
        origin_B_traj_final_end[0:3, 3] = np.array([0.0, 0.0, traj_z_offset]) + \
                                          np.array(origin_B_upperarm_world)[0:3, 3]
        # print 'origin_B_traj_final_end\n', origin_B_traj_final_end
        # origin_B_rotated_pointed_down_shoulder = np.matrix(origin_B_upperarm_pointed_down_shoulder)

        # rev = m.radians(180.)
        # shoulder_position_B_traj_final_end = np.matrix([[m.cos(rev), -m.sin(rev), 0., 0.0],
        #                                                 [m.sin(rev), m.cos(rev), 0., -0.0],
        #                                                 [0., 0., 1., traj_z_offset],
        #                                                 [0., 0., 0., 1.]])
        origin_B_shoulder_position = np.eye(4)
        origin_B_shoulder_position[0:3, 3] = np.array(origin_B_upperarm)[0:3, 3]
        # origin_B_traj_final_end = np.matrix(origin_B_shoulder_position) * shoulder_position_B_traj_final_end

        origin_B_traj_start = origin_B_traj_start_pos

        # Find the transforms from the origin to the goal poses.
        goals = []
        # Goals along forearm
        path_distance = np.linalg.norm(np.array(origin_B_traj_start)[0:3, 3] -
                                       np.array(origin_B_traj_forearm_end)[0:3, 3])
        path_waypoints = np.arange(0., path_distance + path_distance * 0.01, (path_distance - 0.15) / 2.)
        for goal in path_waypoints:
            traj_start_B_traj_waypoint = np.matrix(np.eye(4))
            traj_start_B_traj_waypoint[0, 3] = goal
            origin_B_traj_waypoint = copy.copy(np.matrix(origin_B_traj_start) *
                                               np.matrix(traj_start_B_traj_waypoint))
            goals.append(copy.copy(origin_B_traj_waypoint))

        # Goals along upper arm
        path_distance = np.linalg.norm(np.array(origin_B_traj_forearm_end)[0:3, 3] -
                                       np.array(origin_B_traj_upper_end)[0:3, 3])
        path_waypoints = np.arange(path_distance, 0.0 - path_distance *0.01, -path_distance / 2.)
        for goal in path_waypoints:
            traj_start_B_traj_waypoint = np.matrix(np.eye(4))
            traj_start_B_traj_waypoint[0, 3] = -goal
            origin_B_traj_waypoint = copy.copy(np.matrix(origin_B_traj_upper_end) *
                                               np.matrix(traj_start_B_traj_waypoint))
            goals.append(copy.copy(origin_B_traj_waypoint))

        # Goals at the top of the shoulder
        origin_B_traj_waypoint[0:3, 0:3] = origin_B_traj_final_end[0:3, 0:3]
        goals.append(copy.copy(origin_B_traj_waypoint))
        goals.append(copy.copy(origin_B_traj_final_end))

        # for goal in goals:
        #     print goal



        # path_distance = np.linalg.norm(np.array(origin_B_traj_upper_end)[0:3, 3] -
        #                                np.array(origin_B_traj_final_end)[0:3, 3])
        # path_waypoints = np.arange(path_distance,  0.0 - path_distance * 0.01, -path_distance / 1.)
        # for goal in path_waypoints:
        #     traj_start_B_traj_waypoint = np.matrix(np.eye(4))
        #     traj_start_B_traj_waypoint[0, 3] = -goal
        #     origin_B_traj_waypoint = copy.copy(np.matrix(origin_B_traj_final_end) *
        #                                        np.matrix(traj_start_B_traj_waypoint))
        #     goals.append(copy.copy(origin_B_traj_waypoint))
        fixed_point_exceeded_amount = 0.
        # print 'stretch allowable:\n', self.stretch_allowable
        if self.add_new_fixed_point:
            self.add_new_fixed_point = False
            self.fixed_points.append(np.array(goals[-1])[0:3, 3])
        for point_i in self.fixed_points_to_use:
            fixed_point = self.fixed_points[point_i]
            # fixed_position = np.array(fixed_point)[0:3, 3]
            # print 'fixed point:\n', fixed_point
            for goal in goals:
                goal_position = np.array(goal)[0:3, 3]
                # print 'goal_position:\n', goal_position
                # print 'stretch allowable:\n', self.stretch_allowable
                # print 'amount stretched:\n', np.linalg.norm(fixed_point - goal_position)
                # print 'amount exceeded by this goal:\n', np.linalg.norm(fixed_point - goal_position) - self.stretch_allowable[point_i]
                fixed_point_exceeded_amount = np.max([fixed_point_exceeded_amount, np.linalg.norm(fixed_point - goal_position) - self.stretch_allowable[point_i]])
            # if fixed_point_exceeded_amount > 0.:
            #     print 'The gown is being stretched too much to try to do the next part of the task.'

        # print 'fixed_point_exceeded_amount:', fixed_point_exceeded_amount
        return goals, np.matrix(origin_B_forearm_pointed_down_arm), np.matrix(origin_B_upperarm_pointed_down_shoulder), \
               np.matrix(origin_B_hand), np.matrix(origin_B_wrist), \
               np.matrix(origin_B_traj_start), np.matrix(origin_B_traj_forearm_end), np.matrix(origin_B_traj_upper_end), \
               np.matrix(origin_B_traj_final_end), angle_from_horizontal, \
               np.matrix(forearm_B_upper_arm), fixed_point_exceeded_amount

    def objective_function_traj_and_arm_config(self, params):
        # params[7:] = [0., 0., 1.]
        # params = [0.1, 0.,  0.1,
        #           -0.1,  0.0, 0.1,
        #           0.,
        # params = [m.radians(90.0),  m.radians(0.), m.radians(45.), m.radians(0.)]
        # print 'doing subtask', self.subtask_step
        # print 'params:\n', params
        if self.subtask_step == 0 or False:
            # params = [1.41876758,  0.13962405,  1.47350044,  0.95524629]  # old solution with joint jump
            # params = [1.73983062, -0.13343737,  0.42208647,  0.26249355]  # solution with arm snaking
            # params = [0.3654207,  0.80081779,  0.44793856,  1.83270078]  # without checking with phsyx
            params = [0.9679925, 0.18266905, 0.87995157, 0.77562143]

        neigh_distances, neighbors = self.arm_knn.kneighbors([params], 8)
        for neigh_dist, neighbor in zip(neigh_distances[0], neighbors[0]):
            if np.max(np.abs(np.array(self.arm_configs_checked[neighbor] - np.array(params)))) < m.radians(15.):
                if not self.arm_configs_eval[neighbor][5] == 'good':
                    # print 'arm evaluation found this configuration to be bad'
                    this_score = 10. + 10. + 2. + random.random()
                    if this_score < self.best_overall_score:
                        self.best_overall_score = this_score
                        self.best_physx_config = params
                        self.best_physx_score = 10. + 2. + random.random()
                        self.best_kinematics_config = np.zeros(4)
                        self.best_kinematics_score = 10. + 2. + random.random()
                    return this_score
        print 'arm config is not bad'
        arm = self.human_arm.split('a')[0]

        testing = False

        # path_distance = np.linalg.norm(np.array(params[0:3])-np.array(params[3:6]))
        # print 'params'
        # print params
        self.set_human_model_dof_dart([0, 0, 0, 0], self.human_opposite_arm)
        # self.set_human_model_dof_dart([params[0], params[1], params[2], params[3]], self.human_opposite_arm)
        self.set_human_model_dof_dart([params[0], params[1], params[2], params[3]], self.human_arm)

        # self.set_human_model_dof_dart([params[0], params[1], params[2], params[3]], 'leftarm')

        if self.is_human_in_self_collision():
            this_score = 10. + 10. + 1. + random.random()
            if this_score < self.best_overall_score:
                self.best_overall_score = this_score
                self.best_physx_config = params
                self.best_physx_score = 10. + 1. + random.random()
                self.best_kinematics_config = np.zeros(4)
                self.best_kinematics_score = 10. + 1. + random.random()
            return this_score

        print 'arm config is not in self collision'

        self.goals, \
        origin_B_forearm_pointed_down_arm, \
        origin_B_upperarm_pointed_down_shoulder, \
        origin_B_hand, \
        origin_B_wrist, \
        origin_B_traj_start, \
        origin_B_traj_forearm_end, \
        origin_B_traj_upper_end, \
        origin_B_traj_final_end, \
        angle_from_horizontal, \
        forearm_B_upper_arm, \
        fixed_points_exceeded_amount = self.find_reference_coordinate_frames_and_goals(arm)
        print 'arm does not break fixed_points requirement'


        ############################################
        # Body mass from https://msis.jsc.nasa.gov/sections/section03.htm for average human male
        # upper arm: 2.500 kg
        # fore arm: 1.450 kg
        # hand: 0.530 kg
        upper_arm_force = np.array([0, 0, 2.5*-9.8])
        forearm_force = np.array([0., 0., 1.45*-9.8])
        hand_force = np.array([0., 0., 0.53*-9.8])
        shoulder_to_upper_arm_midpoint = (np.array(origin_B_forearm_pointed_down_arm)[0:3, 3] -
                                          np.array(origin_B_upperarm_pointed_down_shoulder)[0:3, 3])/2.
        shoulder_to_forearm = (np.array(origin_B_forearm_pointed_down_arm)[0:3, 3] -
                               np.array(origin_B_upperarm_pointed_down_shoulder)[0:3, 3])
        shoulder_to_forearm_midpoint = (np.array(origin_B_forearm_pointed_down_arm)[0:3, 3] -
                                        np.array(origin_B_upperarm_pointed_down_shoulder)[0:3, 3]) + \
                                       (np.array(origin_B_wrist)[0:3, 3] -
                                        np.array(origin_B_forearm_pointed_down_arm)[0:3, 3]) / 2.
        shoulder_to_hand_midpoint = (np.array(origin_B_hand)[0:3, 3] -
                                     np.array(origin_B_upperarm_pointed_down_shoulder)[0:3, 3])
        # elbow_to_forearm_midpoint = (np.array(origin_B_wrist)[0:3, 3] -
        #                              np.array(origin_B_forearm_pointed_down_arm)[0:3, 3]) / 2.
        # elbow_to_hand_midpoint = (np.array(origin_B_hand)[0:3, 3] -
        #                           np.array(origin_B_forearm_pointed_down_arm)[0:3, 3])
        # print 'shoulder_to_upper_arm_midpoint\n', shoulder_to_upper_arm_midpoint
        # print 'shoulder_to_forearm\n', shoulder_to_forearm
        # print 'shoulder_to_forearm_midpoint\n', shoulder_to_forearm_midpoint
        # print 'shoulder_to_hand_midpoint\n', shoulder_to_hand_midpoint
        torque_at_shoulder = np.cross(-1*shoulder_to_upper_arm_midpoint, upper_arm_force) + \
                             np.cross(-1 * shoulder_to_forearm_midpoint, forearm_force) + \
                             np.cross(-1 * shoulder_to_hand_midpoint, hand_force)
        # torque_at_elbow = np.cross(-1 * elbow_to_forearm_midpoint, forearm_force) + \
        #                   np.cross(-1 * elbow_to_hand_midpoint, hand_force)
                             # forearm_mass*np.linalg.norm(shoulder_to_forearm_midpoint[0:2]) + \
                             # hand_mass*np.linalg.norm(shoulder_to_hand_midpoint[0:2])
        torque_magnitude = np.linalg.norm(torque_at_shoulder) #+ np.linalg.norm(torque_at_elbow)
        max_possible_torque = 12.376665  # found manually with arm straight out from arm
        # print 'torque_at_shoulder\n', torque_at_shoulder
        # print 'torque_magnitude\n', torque_magnitude
        torque_cost = torque_magnitude/max_possible_torque

        ############################################

        if fixed_points_exceeded_amount > 0.:
            # print 'The gown is being stretched too much to try to do the next part of the task.'
            # return 10. + 1. + 10. * fixed_points_exceeded_amount
            this_score = 10. + 10. + 1. + 10. * fixed_points_exceeded_amount
            if this_score < self.best_overall_score:
                self.best_overall_score = this_score
                self.best_physx_config = params
                self.best_physx_score = 10. + 1. + 10. * fixed_points_exceeded_amount
                self.best_kinematics_config = np.zeros(4)
                self.best_kinematics_score = 10. + 1. + 10. * fixed_points_exceeded_amount
            return this_score


        print 'angle from horizontal = ', angle_from_horizontal
        if abs(angle_from_horizontal) > 30.:
            print 'Angle of forearm is too high for success'
            this_score = 10. + 10. + 10. * (abs(angle_from_horizontal) - 30.)
            if this_score < self.best_overall_score:
                self.best_overall_score = this_score
                self.best_physx_config = params
                self.best_physx_score = 10. + 10. * (abs(angle_from_horizontal) - 30.)
                self.best_kinematics_config = np.zeros(4)
                self.best_kinematics_score = 10. + 10. * (abs(angle_from_horizontal) - 30.)
            return this_score

        print 'Number of goals: ', len(self.goals)
        start_time = rospy.Time.now()
        self.set_goals()
        # print self.origin_B_grasps
        maxiter = 5
        popsize = 200#4*20
        if self.subtask_step == 0 or False:
            maxiter = 5
            popsize = 5

        # cma parameters: [pr2_base_x, pr2_base_y, pr2_base_theta, pr2_base_height,
        # human_arm_dof_1, human_arm_dof_2, human_arm_dof_3, human_arm_dof_4, human_arm_dof_5,
        # human_arm_dof_6, human_arm_dof_7]
        parameters_min = np.array([-1.5, -1.5, -2.5*m.pi-.001, 0.0])
        parameters_max = np.array([1.5, 1.5, 2.5*m.pi+.001, 0.3])
        # [0.3, -0.9, 1.57 * m.pi / 3., 0.3]
        # parameters_min = np.array([-0.1, -1.0, m.pi/2. - .001, 0.2])
        # parameters_max = np.array([0.8, -0.3, 2.5*m.pi/2. + .001, 0.3])
        parameters_scaling = (parameters_max-parameters_min)/8.

        init_start_pr2_configs = [[0., 0., 0., m.radians(0.)],
                                  [0., 0., 0., m.radians(0.)],
                                  [0., 0., 0., m.radians(0.)]]

        parameters_initialization = (parameters_max+parameters_min)/2.
        opts2 = {'seed': 1234, 'ftarget': -1., 'popsize': popsize, 'maxiter': maxiter,
                 'maxfevals': 1e8, 'CMA_cmean': 0.25, 'tolfun': 1e-3,
                 'tolfunhist': 1e-12, 'tolx': 5e-4,
                 'maxstd': 4.0, 'tolstagnation': 100,
                 'verb_filenameprefix': 'outcma_pr2_base',
                 'scaling_of_variables': list(parameters_scaling),
                 'bounds': [list(parameters_min), list(parameters_max)]}

        self.this_best_pr2_config = None
        self.this_best_pr2_score = 1000.

        for init_start_pr2_config in init_start_pr2_configs:
            parameters_initialization = init_start_pr2_config
            self.kinematics_optimization_results = cma.fmin(self.objective_function_one_config,
                                                          list(parameters_initialization),
                                                          1.,
                                                          options=opts2)
            print 'Best PR2 configuration for this arm config so far: \n', self.this_best_pr2_config[0]
            print 'Associated score: ', self.this_best_pr2_score
        # self.pr2_parameters.append([self.kinematics_optimization_results[0], self.kinematics_optimization_results[1]])
        # save_pickle(self.pr2_parameters, self.pkg_path+'/data/all_pr2_configs.pkl')
        gc.collect()
        elapsed_time = rospy.Time.now()-start_time
        print 'Done with openrave round. Time elapsed:', elapsed_time.to_sec()
        print 'Openrave results:'
        # print self.kinematics_optimization_results
        self.force_cost = 0.
        if self.this_best_pr2_score < 0.:
            self.physx_output = False
            self.physx_outcome = None
            # traj_start = np.array(origin_B_traj_start_pos)[0:3, 3]
            # traj_end = np.array(origin_B_traj_end_pos)[0:3, 3]

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
                    zeta = 0.5  # cost on torque
                    physx_score = self.force_cost*alpha + torque_cost*zeta
                    this_score = physx_score + self.this_best_pr2_score*beta
                    if this_score < self.best_overall_score:
                        self.best_overall_score = this_score
                        self.best_physx_config = params
                        self.best_physx_score = physx_score
                        self.best_kinematics_config = self.this_best_pr2_config
                        self.best_kinematics_score = self.this_best_pr2_score
                    # return this_score

                    # self.arm_traj_parameters.append([params, physx_score])

                    # save_pickle(self.arm_traj_parameters, self.pkg_path+'/data/all_arm_traj_configs.pkl')
                    print 'Force cost was: ', self.force_cost
                    print 'Torque score was: ', torque_cost
                    print 'Physx score was: ', physx_score
                    print 'Best pr2 kinematics score was: ', self.this_best_pr2_score
                    return this_score
        self.physx_outcome = None
        self.physx_output = False
        alpha = 1.  # cost on forces
        beta = 1.  # cost on manipulability
        zeta = 0.5  # cost on torque
        # self.arm_traj_parameters.append([params, 10. + self.force_cost*alpha + self.kinematics_optimization_results[1]*beta + torque_cost*zeta])
        # save_each_config_score = False
        # if save_each_config_score:
        #     save_pickle(self.arm_traj_parameters, self.pkg_path+'/data/all_arm_traj_configs.pkl')
        # physx_score = 10. + self.force_cost*alpha + self.kinematics_optimization_results[1]*beta + torque_cost*zeta
        print 'Force cost was: ', self.force_cost
        print 'Kinematics score was: ', self.this_best_pr2_score
        print 'Torque score was: ', torque_cost
        physx_score = self.force_cost*alpha + torque_cost*zeta
        this_score = 10. + physx_score + self.this_best_pr2_score*beta
        print 'Total score was: ', this_score
        if this_score < self.best_overall_score:
            self.best_overall_score = this_score
            self.best_physx_config = params
            self.best_physx_score = physx_score
            self.best_kinematics_config = self.this_best_pr2_config
            self.best_kinematics_score = self.this_best_pr2_score
        return this_score

    def calculate_scores(self, task_dict, model, ref_options):
        self.model = model
        self.task_dict = task_dict
        self.reference_names = ref_options

        self.setup_human_openrave(model)

        self.handle_score_generation()

    def receive_new_goals(self, goals, task, human_arm, pr2_arm, reference_options=None):
        if pr2_arm == 'rightarm':
            pass
            # self.gripper_B_tool = np.matrix([[0., -1., 0., 0.03],
            #                                  [1., 0., 0., 0.0],
            #                                  [0., 0., 1., -0.08],
            #                                  [0., 0., 0., 1.]])
        else:
            print 'I do not know the transform for this pr2 arm tool'

        if reference_options:
            self.reference_names = reference_options
            self.origin_B_references = []
            if self.model == 'green_kevin':
                if human_arm == 'rightarm':
                    if task == 'hospital_gown':
                        headmodel = self.human_model.GetLink('green_kevin/head_link')
                        ual = self.human_model.GetLink('green_kevin/arm_left_link')
                        uar = self.human_model.GetLink('green_kevin/arm_right_link')
                        fal = self.human_model.GetLink('green_kevin/forearm_left_link')
                        far = self.human_model.GetLink('green_kevin/forearm_right_link')
                        hl = self.human_model.GetLink('green_kevin/hand_left_link')
                        hr = self.human_model.GetLink('green_kevin/hand_right_link')
                        thl = self.human_model.GetLink('green_kevin/quad_left_link')
                        thr = self.human_model.GetLink('green_kevin/quad_right_link')
                        calfl = self.human_model.GetLink('green_kevin/calf_left_link')
                        calfr = self.human_model.GetLink('green_kevin/calf_right_link')
                        ub = self.human_model.GetLink('green_kevin/upper_body_link')
                        origin_B_head = np.matrix(headmodel.GetTransform())
                        origin_B_ual = np.matrix(ual.GetTransform())
                        origin_B_uar = np.matrix(uar.GetTransform())
                        origin_B_fal = np.matrix(fal.GetTransform())
                        origin_B_far = np.matrix(far.GetTransform())
                        origin_B_hl = np.matrix(hl.GetTransform())
                        origin_B_hr = np.matrix(hr.GetTransform())
                        origin_B_thl = np.matrix(thl.GetTransform())
                        origin_B_thr = np.matrix(thr.GetTransform())
                        origin_B_calfl = np.matrix(calfl.GetTransform())
                        origin_B_calfr = np.matrix(calfr.GetTransform())
                        origin_B_ub = np.matrix(ub.GetTransform())
                        origin_B_head = np.matrix(headmodel.GetTransform())
            else:
                print 'I GOT A BAD MODEL. NOT SURE WHAT TO DO NOW!'

            for y in self.reference_names:
                if y == 'head':
                    self.origin_B_references.append(origin_B_head)
                elif y == 'base_link':
                    self.origin_B_references.append(np.matrix(np.eye(4)))
                elif y == 'upper_arm_left':
                    self.origin_B_references.append(origin_B_ual)
                elif y == 'upper_arm_right':
                    self.origin_B_references.append(origin_B_uar)
                elif y == 'forearm_left':
                    self.origin_B_references.append(origin_B_fal)
                elif y == 'forearm_right':
                    self.origin_B_references.append(origin_B_far)
                elif y == 'hand_left':
                    self.origin_B_references.append(origin_B_hl)
                elif y == 'hand_right':
                    self.origin_B_references.append(origin_B_hr)
                elif y == 'thigh_left':
                    self.origin_B_references.append(origin_B_thl)
                elif y == 'thigh_right':
                    self.origin_B_references.append(origin_B_thr)
                elif y == 'knee_left':
                    self.origin_B_references.append(origin_B_calfl)
                elif y == 'knee_right':
                    self.origin_B_references.append(origin_B_calfr)
                elif y == 'upper_body':
                    self.origin_B_references.append(origin_B_ub)
                else:
                    print 'The refence options is bogus! I dont know what to do!'
                    return
            self.goals = goals
        # print 'Score generator received a new list of desired goal locations. It contains ', len(goals), ' goal ' \
        #                                                                                                  'locations.'
        self.selection_mat = np.zeros(len(self.goals))
        self.goal_list = np.zeros([len(self.goals), 4, 4])
        self.reference_mat = np.zeros(len(self.goals))
        for w in xrange(len(self.goals)):
            temp_B = self.origin_B_references[int(self.reference_mat[w])]*\
                          np.matrix(self.goals[w, 0])
            temp_B[2, 3] += self.optimal_z_offset

            #self.goal_list.append(pr2_B_head*np.matrix(target[0])*goal_B_gripper)
            self.reference_mat[w] = int(self.goals[w, 2])
            self.goal_list[w] = np.matrix(copy.copy(temp_B))*self.gripper_B_tool
            self.selection_mat[w] = self.goals[w, 1]

        self.set_goals()

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
            for robot_arm in [self.robot_opposite_arm, self.robot_arm]:
                self.op_robot.SetActiveManipulator(robot_arm)
                self.manip = self.op_robot.GetActiveManipulator()
                ikmodel = op.databases.inversekinematics.InverseKinematicsModel(self.op_robot,
                                                                                iktype=op.IkParameterization.Type.Transform6D)
                if not ikmodel.load():
                    print 'IK model not found for this arm. Generating the ikmodel for the ', robot_arm
                    print 'This will take a while'
                    ikmodel.autogenerate()
                self.manipprob = op.interfaces.BaseManipulation(self.op_robot)
            return True
        else:
            print 'ERROR'
            print 'I do not know what arm to be using'
            return False

    def set_human_arm(self, arm):
        # Set human arm for dressing
        print 'Setting the human arm being used by base selection to ', arm
        if 'left' in arm:
            self.gripper_B_tool = np.matrix([[0., 1., 0., 0.03],
                                             [-1., 0., 0., 0.0],
                                             [0., 0., 1., -0.05],
                                             [0., 0., 0., 1.]])
            self.human_arm = 'leftarm'
            self.human_opposite_arm = 'rightarm'
            return True
        elif 'right' in arm:
            self.gripper_B_tool = np.matrix([[0., -1., 0., 0.03],
                                             [1., 0., 0., 0.0],
                                             [0., 0., 1., -0.05],
                                             [0., 0., 0., 1.]])
            self.human_arm = 'rightarm'
            self.human_opposite_arm = 'leftarm'
            return True
        else:
            print 'ERROR'
            print 'I do not know what arm to be using'
            return False

    def objective_function_one_config(self, current_parameters):
        # start_time = rospy.Time.now()
        # current_parameters = [0.3, -0.9, 1.57*m.pi/3., 0.3]
        if self.subtask_step == 0 or False:
            # current_parameters = [0.2743685, -0.71015745, 0.20439603, 0.29904425]
            # current_parameters = [0.2743685, -0.71015745, 2.2043960252256807, 0.29904425]  # old solution with joint jump
            # current_parameters = [2.5305254, -0.6124738, -2.37421411, 0.02080042]  # solution with arm snaking
            # current_parameters = [0.44534457, -0.85069379, 2.95625035, 0.07931574]  # solution with arm in lap, no physx
            current_parameters = [-0.00917182, -0.8680934,   1.58936071,  0.045496]
        x = current_parameters[0]
        y = current_parameters[1]
        th = current_parameters[2]
        z = current_parameters[3]

        origin_B_pr2 = np.matrix([[ m.cos(th), -m.sin(th),     0.,         x],
                                  [ m.sin(th),  m.cos(th),     0.,         y],
                                  [        0.,         0.,     1.,        0.],
                                  [        0.,         0.,     0.,        1.]])
        # print 'pr2_B_origin\n', origin_B_pr2.I
        v = self.robot.positions()

        # For a solution for a bit to get screenshots, etc. Check colllision removes old collision markers.
        if False:
            self.dart_world.check_collision()
            rospy.sleep(20)

        v['rootJoint_pos_x'] = x
        v['rootJoint_pos_y'] = y
        v['rootJoint_pos_z'] = 0.
        v['rootJoint_rot_z'] = th
        self.dart_world.displace_gown()
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

        # self.dart_world.set_gown()

        # PR2 is too close to the person (who is at the origin). PR2 base is 0.668m x 0.668m
        distance_from_origin = np.linalg.norm(origin_B_pr2[:2, 3])
        if distance_from_origin <= 0.334:
            this_pr2_score = 10. + 1. + (0.4 - distance_from_origin)
            if this_pr2_score < self.this_best_pr2_score:
                self.this_best_pr2_config = current_parameters
                self.this_best_pr2_score = this_pr2_score
            return this_pr2_score

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
            this_pr2_score = 10. +1.+ 20.*(distance - 1.25)
            if this_pr2_score < self.this_best_pr2_score:
                self.this_best_pr2_config = current_parameters
                self.this_best_pr2_score = this_pr2_score
            return this_pr2_score

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

        # in_collision = self.is_dart_in_collision()
        base_in_collision = self.is_dart_base_in_collision()

        close_to_collision = False
        check_if_PR2_is_near_collision = False
        if check_if_PR2_is_near_collision:
            positions = self.robot.positions()
            positions['rootJoint_pos_x'] = x + 0.04
            positions['rootJoint_pos_y'] = y + 0.04
            positions['rootJoint_pos_z'] = 0.
            positions['rootJoint_rot_z'] = th
            self.robot.set_positions(positions)
            self.dart_world.set_gown([self.robot_arm])
            close_to_collision = np.max([self.is_dart_in_collision(), close_to_collision])

            positions['rootJoint_pos_x'] = x - 0.04
            positions['rootJoint_pos_y'] = y + 0.04
            positions['rootJoint_pos_z'] = 0.
            positions['rootJoint_rot_z'] = th
            self.robot.set_positions(positions)
            self.dart_world.set_gown([self.robot_arm])
            close_to_collision = np.max([self.is_dart_in_collision(), close_to_collision])

            positions['rootJoint_pos_x'] = x - 0.04
            positions['rootJoint_pos_y'] = y - 0.04
            positions['rootJoint_pos_z'] = 0.
            positions['rootJoint_rot_z'] = th
            self.robot.set_positions(positions)
            self.dart_world.set_gown([self.robot_arm])
            close_to_collision = np.max([self.is_dart_in_collision(), close_to_collision])

            positions['rootJoint_pos_x'] = x + 0.04
            positions['rootJoint_pos_y'] = y - 0.04
            positions['rootJoint_pos_z'] = 0.
            positions['rootJoint_rot_z'] = th
            self.robot.set_positions(positions)
            self.dart_world.set_gown([self.robot_arm])
            close_to_collision = np.max([self.is_dart_in_collision(), close_to_collision])

            positions['rootJoint_pos_x'] = x
            positions['rootJoint_pos_y'] = y
            positions['rootJoint_pos_z'] = 0.
            positions['rootJoint_rot_z'] = th
            self.robot.set_positions(positions)
            self.dart_world.set_gown([self.robot_arm])
        best_ik = None
        if not close_to_collision and not base_in_collision:
            reached = np.zeros(len(self.origin_B_grasps))
            manip = np.zeros(len(self.origin_B_grasps))
            is_smooth_ik_possible, joint_change_amount = self.check_smooth_ik_feasiblity(self.origin_B_grasps)
            if not is_smooth_ik_possible:
                this_pr2_score = 10. + 1. + joint_change_amount
                if this_pr2_score < self.this_best_pr2_score:
                    self.this_best_pr2_config = current_parameters
                    self.this_best_pr2_score = this_pr2_score
                return this_pr2_score
            all_sols = []
            all_jacobians = []

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
                single_goal_sols, single_goal_jacobians = self.ik_request(pr2_B_grasp, z)
                all_sols.append(list(single_goal_sols))
                all_jacobians.append(list(single_goal_jacobians))

            graph = SimpleGraph()
            graph.edges['start'] = []
            graph.value['start'] = 0
            graph.edges['end'] = []
            graph.value['end'] = 0
            for goal_i in xrange(len(all_sols)):
                for sol_i in xrange(len(all_sols[goal_i])):
                    v = self.robot.q
                    v[self.robot_arm[0] + '_shoulder_pan_joint'] = all_sols[goal_i][sol_i][0]
                    v[self.robot_arm[0] + '_shoulder_lift_joint'] = all_sols[goal_i][sol_i][1]
                    v[self.robot_arm[0] + '_upper_arm_roll_joint'] = all_sols[goal_i][sol_i][2]
                    v[self.robot_arm[0] + '_elbow_flex_joint'] = all_sols[goal_i][sol_i][3]
                    v[self.robot_arm[0] + '_forearm_roll_joint'] = all_sols[goal_i][sol_i][4]
                    v[self.robot_arm[0] + '_wrist_flex_joint'] = all_sols[goal_i][sol_i][5]
                    v[self.robot_arm[0] + '_wrist_roll_joint'] = all_sols[goal_i][sol_i][6]
                    self.robot.set_positions(v)
                    self.dart_world.set_gown([self.robot_arm])
                    if not self.is_dart_in_collision():
                        graph.edges[str(goal_i)+'-'+str(sol_i)] = []
                        J = np.matrix(all_jacobians[goal_i][sol_i])
                        joint_limit_weight = self.gen_joint_limit_weight(all_sols[goal_i][sol_i], self.robot_arm)
                        manip = (m.pow(np.linalg.det(J * joint_limit_weight * J.T), (1. / 6.))) / (np.trace(J * joint_limit_weight * J.T) / 6.)
                        graph.value[str(goal_i)+'-'+str(sol_i)] = manip
            # print sorted(graph.edges)
            for node in graph.edges.keys():
                if not node == 'start' and not node == 'end':
                    goal_i = int(node.split('-')[0])
                    sol_i = int(node.split('-')[1])
                    if goal_i == 0:
                        graph.edges['start'].append(str(goal_i)+'-'+str(sol_i))
                    if goal_i == len(all_sols) - 1:
                        graph.edges[str(goal_i)+'-'+str(sol_i)].append('end')
                    else:
                        possible_next_nodes = [t for t in (a
                                                           for a in graph.edges.keys())
                                               if str(goal_i+1) in t.split('-')[0]
                                               ]
                        for next_node in possible_next_nodes:
                            goal_j = int(next_node.split('-')[0])
                            sol_j = int(next_node.split('-')[1])
                            if np.max(np.abs(np.array(all_sols[goal_j][sol_j])[0:3]-np.array(all_sols[goal_i][sol_i])[0:3])) < m.radians(40.):
                                graph.edges[str(goal_i)+'-'+str(sol_i)].append(str(goal_j)+'-'+str(sol_j))

            came_from, value_so_far = a_star_search(graph, 'start', 'end')
            # print 'came_from\n', came_from
            path = reconstruct_path(came_from, 'start', 'end')
            # print sorted(graph.edges)

            # print 'path\n', path
            if not path:
                if len(value_so_far) == 1:
                    reach_score = 0.
                    manip_score = 0.
                else:
                    value_so_far.pop('start')
                    furthest_reached = np.argmax([t for t in ((int(a[0].split('-')[0]))
                                                                for a in value_so_far.items())
                                                 ])
                    # print value_so_far.keys()
                    # print 'furthest reached', furthest_reached
                    # print value_so_far.items()[furthest_reached]
                    reach_score = 1.*int(value_so_far.items()[furthest_reached][0].split('-')[0])/len(self.origin_B_grasps)
                    manip_score = 1.*value_so_far.items()[furthest_reached][1]/len(self.origin_B_grasps)
            else:
                # print 'I FOUND A SOLUTION'
                # print 'value_so_far[end]:', value_so_far['end']
                path.pop(0)
                path.pop(path.index('end'))
                reach_score = 1.
                manip_score = value_so_far['end']/len(self.origin_B_grasps)

            if self.visualize or (not self.subtask_step == 0 or False):
                if path:
                    prev_sol = np.zeros(7)
                    print 'Solution being visualized:'
                for path_step in path:
                    # if not path_step == 'start' and not path_step == 'end':
                    goal_i = int(path_step.split('-')[0])
                    sol_i = int(path_step.split('-')[1])
                    print 'solution:\n', all_sols[goal_i][sol_i]
                    print 'diff:\n', np.abs(np.array(all_sols[goal_i][sol_i]) - prev_sol)
                    print 'max diff:\n', np.degrees(np.max(np.abs(np.array(all_sols[goal_i][sol_i]) - prev_sol)))
                    prev_sol = np.array(all_sols[goal_i][sol_i])

                    v = self.robot.q
                    v[self.robot_arm[0] + '_shoulder_pan_joint'] = all_sols[goal_i][sol_i][0]
                    v[self.robot_arm[0] + '_shoulder_lift_joint'] = all_sols[goal_i][sol_i][1]
                    v[self.robot_arm[0] + '_upper_arm_roll_joint'] = all_sols[goal_i][sol_i][2]
                    v[self.robot_arm[0] + '_elbow_flex_joint'] = all_sols[goal_i][sol_i][3]
                    v[self.robot_arm[0] + '_forearm_roll_joint'] = all_sols[goal_i][sol_i][4]
                    v[self.robot_arm[0] + '_wrist_flex_joint'] = all_sols[goal_i][sol_i][5]
                    v[self.robot_arm[0] + '_wrist_roll_joint'] = all_sols[goal_i][sol_i][6]
                    self.robot.set_positions(v)
                    self.dart_world.displace_gown()
                    self.dart_world.check_collision()
                    self.dart_world.set_gown([self.robot_arm])
                    rospy.sleep(1.5)
        else:
            # print 'In base collision! single config distance: ', distance
            if distance < 2.0:
                this_pr2_score = 10. + 1. + (1.25 - distance)
                if this_pr2_score < self.this_best_pr2_score:
                    self.this_best_pr2_config = current_parameters
                    self.this_best_pr2_score = this_pr2_score
                return this_pr2_score

        # self.human_model.SetActiveManipulator('leftarm')
        # self.human_manip = self.robot.GetActiveManipulator()
        # human_torques = self.human_manip.ComputeInverseDynamics([])
        # torque_cost = np.linalg.norm(human_torques)/10.

        # angle_cost = np.sum(np.abs(human_dof))
        # print 'len(self.goals)'
        # print len(self.goals)

        # print 'reached'
        # print reached

        # reach_score /= len(self.goals)
        # manip_score /= len(self.goals)

        # print 'reach_score'
        # print reach_score
        # print 'manip_score'
        # print manip_score

        # Set the weights for the different scores.
        beta = 10.  # Weight on number of reachable goals
        gamma = 1.  # Weight on manipulability of arm at each reachable goal
        zeta = 0.05  # Weight on torques
        if reach_score == 0.:
            this_pr2_score = 10. + 1.+ 2*random.random()
            if this_pr2_score < self.this_best_pr2_score:
                self.this_best_pr2_config = current_parameters
                self.this_best_pr2_score = this_pr2_score
            return this_pr2_score
        else:
            # print 'Reach score: ', reach_score
            # print 'Manip score: ', manip_score
            if reach_score == 1.:
                if self.visualize:
                    rospy.sleep(2.0)
            # print 'reach_score:', reach_score
            # print 'manip_score:', manip_score
            this_pr2_score = 10.-beta*reach_score-gamma*manip_score #+ zeta*angle_cost
            if this_pr2_score < self.this_best_pr2_score:
                self.this_best_pr2_config = current_parameters
                self.this_best_pr2_score = this_pr2_score
            return this_pr2_score

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

    def is_human_in_self_collision(self):
        self.dart_world.human.set_self_collision_check(True)
        self.dart_world.check_collision()
        arm = self.human_arm.split('a')[0]
        arm_parts = [self.human.bodynode('h_bicep_'+arm),
                     self.human.bodynode('h_forearm_'+arm),
                     self.human.bodynode('h_hand_'+arm),
                     self.human.bodynode('h_hand_'+arm+'2')]
        for contact in self.dart_world.collision_result.contacts:
            contacts = [contact.bodynode1, contact.bodynode2]
            for arm_part in arm_parts:
                if arm_part in contacts and self.human == contact.skel1 and self.human == contact.skel2:
                    contacts.remove(arm_part)
                    if contacts:
                        if contacts[0] not in arm_parts and not contacts[0] == self.human.bodynode('h_scapula_'+arm):
                            return True
        self.human.set_self_collision_check(False)
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
        self.dart_world = DartDressingWorld(skel_file)

        # Lets you visualize dart.
        if self.visualize:
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
        v['l_gripper_l_finger_joint'] = .54
        v['r_shoulder_pan_joint'] = -1.*3.14/2
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
        rospy.wait_for_service('ikfast_service')
        print 'Found IK service.'
        self.ik_service = rospy.ServiceProxy('ikfast_service', IKService, persistent=True)
        print 'IK service is ready for use!'

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

    def check_smooth_ik_feasiblity(self, goal_poses):

        max_joint_change = 1.0
        feasible_ik = True
        return feasible_ik, max_joint_change

    # Setup physx services and initialized the human in Physx
    def setup_physx(self):
        print 'Setting up services and looking for Physx services'

        # print 'Found Physx services'
        self.physx_output_service = rospy.Service('physx_output', PhysxOutput, self.simulator_result_handler)
        rospy.wait_for_service('init_physx_body_model')
        self.init_physx_service = rospy.ServiceProxy('init_physx_body_model', InitPhysxBodyModel)
        rospy.wait_for_service('body_config_input_to_physx')
        self.update_physx_config_service = rospy.ServiceProxy('body_config_input_to_physx', PhysxInputWaypoints)

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
            # print 'right here'
            resp = self.update_physx_config_service(spheres_x, spheres_y, spheres_z, spheres_r, first_sphere_list,
                                                    second_sphere_list, traj_start, traj_forearm_end, traj_upper_end,
                                                    traj_final_end, origin_B_forearm_pointed_down_arm,
                                                    origin_B_upperarm_pointed_down_shoulder)
        return resp

    def visualize_many_configurations(self):
        arm_traj_configs = load_pickle(self.pkg_path+'/data/saved_results/large_search_space/all_arm_traj_configs.pkl')
        pr2_configs = load_pickle(self.pkg_path+'/data/saved_results/large_search_space/all_pr2_configs.pkl')
        # arm_traj_configs = load_pickle(self.pkg_path+'/data/all_arm_traj_configs.pkl')
        # pr2_configs = load_pickle(self.pkg_path+'/data/all_pr2_configs.pkl')
        print len(arm_traj_configs)
        print len(pr2_configs)

        for i in xrange(len(arm_traj_configs)):
            print arm_traj_configs[i]
            traj_and_arm_config = arm_traj_configs[i][0]
            params = traj_and_arm_config
            pr2_config = pr2_configs[i][0]
            self.set_human_model_dof([traj_and_arm_config[7], traj_and_arm_config[8], -traj_and_arm_config[9], traj_and_arm_config[6], 0, 0, 0], 'rightarm', 'green_kevin')

            uabl = self.human_model.GetLink('green_kevin/arm_left_base_link')
            uabr = self.human_model.GetLink('green_kevin/arm_right_base_link')
            ual = self.human_model.GetLink('green_kevin/arm_left_link')
            uar = self.human_model.GetLink('green_kevin/arm_right_link')
            fal = self.human_model.GetLink('green_kevin/forearm_left_link')
            far = self.human_model.GetLink('green_kevin/forearm_right_link')
            hl = self.human_model.GetLink('green_kevin/hand_left_link')
            hr = self.human_model.GetLink('green_kevin/hand_right_link')
            origin_B_uabl = np.matrix(uabl.GetTransform())
            origin_B_uabr = np.matrix(uabr.GetTransform())
            origin_B_ual = np.matrix(ual.GetTransform())
            origin_B_uar = np.matrix(uar.GetTransform())
            origin_B_fal = np.matrix(fal.GetTransform())
            origin_B_far = np.matrix(far.GetTransform())
            origin_B_hl = np.matrix(hl.GetTransform())
            origin_B_hr = np.matrix(hr.GetTransform())

            uabr_B_uabr_corrected = np.matrix([[ 0.,  0., -1., 0.],
                                               [ 1.,  0.,  0., 0.],
                                               [ 0., -1.,  0., 0.],
                                               [ 0.,  0.,  0., 1.]])

            z_origin = np.array([0., 0., 1.])
            x_vector = np.reshape(np.array(origin_B_hr[0:3, 0]), [1, 3])[0]
            y_orth = np.cross(z_origin, x_vector)
            y_orth = y_orth/np.linalg.norm(y_orth)
            z_orth = np.cross(x_vector, y_orth)
            z_orth = z_orth/np.linalg.norm(z_orth)
            origin_B_hr_rotated = np.matrix(np.eye(4))
            # print 'x_vector'
            # print x_vector
            # print 'origin_B_hr_rotated'
            # print origin_B_hr_rotated
            # print 'np.reshape(x_vector, [3, 1])'
            # print np.reshape(x_vector, [3, 1])
            # print 'origin_B_hr_rotated[0:3, 0]'
            # print origin_B_hr_rotated[0:3, 0]
            origin_B_hr_rotated[0:3, 0] = copy.copy(np.reshape(x_vector, [3, 1]))
            origin_B_hr_rotated[0:3, 1] = copy.copy(np.reshape(y_orth, [3, 1]))
            origin_B_hr_rotated[0:3, 2] = copy.copy(np.reshape(z_orth, [3, 1]))
            origin_B_hr_rotated[0:3, 3] = copy.copy(origin_B_hr[0:3, 3])
            origin_B_hr_rotated = np.matrix(origin_B_hr_rotated)

            hr_rotated_B_traj_start_pos = np.matrix(np.eye(4))
            hr_rotated_B_traj_start_pos[0:3, 3] = copy.copy(np.reshape([params[0:3]], [3, 1]))
            hr_rotated_B_traj_start_pos[0, 3] = hr_rotated_B_traj_start_pos[0, 3] + 0.07

            origin_B_traj_start_pos = origin_B_hr_rotated*hr_rotated_B_traj_start_pos

            # print 'origin_B_traj_start_pos'
            # print origin_B_traj_start_pos

            # origin_B_world_rotated_shoulder = createBMatrix(np.reshape(np.array(origin_B_uabr[0:3, 3]), [1, 3])[0], list(tft.quaternion_from_euler(params[7], -params[8], params[9], 'rzxy')))

            origin_B_world_rotated_shoulder = origin_B_uar*uabr_B_uabr_corrected

            # Because green kevin has the upper with a bend in it, I shift the shoulder location by that bend offset.
            shoulder_origin_B_should_origin_shifted_green_kevin = np.matrix(np.eye(4))
            shoulder_origin_B_should_origin_shifted_green_kevin[1, 3] = -0.04953
            origin_B_world_rotated_shoulder = origin_B_world_rotated_shoulder*shoulder_origin_B_should_origin_shifted_green_kevin

            origin_B_uabr[0:3, 3] = origin_B_world_rotated_shoulder[0:3, 3]
            origin_B_uar[0:3, 3] = origin_B_world_rotated_shoulder[0:3, 3]

            z_origin = np.array([0., 0., 1.])
            x_vector = np.reshape(np.array(-1*origin_B_world_rotated_shoulder[0:3, 2]), [1, 3])[0]
            y_orth = np.cross(z_origin, x_vector)
            y_orth = y_orth/np.linalg.norm(y_orth)
            z_orth = np.cross(x_vector, y_orth)
            z_orth = z_orth/np.linalg.norm(z_orth)
            origin_B_rotated_pointed_down_shoulder = np.matrix(np.eye(4))
            origin_B_rotated_pointed_down_shoulder[0:3, 0] = np.reshape(x_vector, [3, 1])
            origin_B_rotated_pointed_down_shoulder[0:3, 1] = np.reshape(y_orth, [3, 1])
            origin_B_rotated_pointed_down_shoulder[0:3, 2] = np.reshape(z_orth, [3, 1])
            origin_B_rotated_pointed_down_shoulder[0:3, 3] = origin_B_uabr[0:3, 3]
            # origin_B_rotated_pointed_down_shoulder = origin_B_uabr*uabr_B_uabr_corrected*np.matrix(shoulder_origin_B_rotated_pointed_down_shoulder)
            # print 'origin_B_rotated_pointed_down_shoulder'
            # print origin_B_rotated_pointed_down_shoulder

            # origin_B_rotated_pointed_down_shoulder[0:3, 3] = origin_B_world_rotated_shoulder[0:3, 3]

            rotated_pointed_down_shoulder_B_traj_end_pos = np.matrix(np.eye(4))
            rotated_pointed_down_shoulder_B_traj_end_pos[0:3, 3] = copy.copy(np.reshape([params[3:6]], [3, 1]))

            # print 'rotated_pointed_down_shoulder_B_traj_end_pos'
            # print rotated_pointed_down_shoulder_B_traj_end_pos
            # print 'origin_B_traj_start_pos'
            # print origin_B_traj_start_pos
            origin_B_traj_end_pos = origin_B_rotated_pointed_down_shoulder*rotated_pointed_down_shoulder_B_traj_end_pos
            # print 'origin_B_traj_end_pos'
            # print origin_B_traj_end_pos

            # print 'origin_B_uabr_corrected'
            # print origin_B_uabr*uabr_B_uabr_corrected

            th = m.radians(180.)
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

            z_origin = np.array([0., 0., 1.])
            x_vector = np.reshape(np.array(origin_B_traj_end_pos[0:3, 3] - origin_B_traj_start_pos[0:3, 3]), [1, 3])[0]
            x_vector = x_vector/np.linalg.norm(x_vector)
            y_orth = np.cross(z_origin, x_vector)
            y_orth = y_orth/np.linalg.norm(y_orth)
            z_orth = np.cross(x_vector, y_orth)
            z_orth = z_orth/np.linalg.norm(z_orth)
            origin_B_traj_start = np.matrix(np.eye(4))
            origin_B_traj_start[0:3, 0] = np.reshape(x_vector, [3, 1])
            origin_B_traj_start[0:3, 1] = np.reshape(y_orth, [3, 1])
            origin_B_traj_start[0:3, 2] = np.reshape(z_orth, [3, 1])
            origin_B_traj_start[0:3, 3] = copy.copy(origin_B_traj_start_pos[0:3, 3])

            # print 'origin_B_traj_start'
            # print origin_B_traj_start

            path_distance = np.linalg.norm(np.reshape(np.array(origin_B_traj_end_pos[0:3, 3] - origin_B_traj_start_pos[0:3, 3]), [1, 3])[0])
            # print 'path_distance'
            # print path_distance
            uabr_corrected_B_traj_start = uabr_B_uabr_corrected.I*origin_B_uabr.I*origin_B_traj_start
            # test_world_shoulder_B_sleeve_start_rotz = np.matrix([[ m.cos(th), -m.sin(th),     0.],
            #                                                      [ m.sin(th),  m.cos(th),     0.],
            #                                                       [        0.,         0.,     1.]])
            # hr_rotated_B_traj_start = createBMatrix([params[0], params[1], params[2]],
            #                                   tft.quaternion_from_euler(params[3], params[4], params[5], 'rzyx'))
            pos_t, quat_t = Bmat_to_pos_quat(uabr_corrected_B_traj_start)

            path_waypoints = np.arange(0., path_distance+path_distance*0.01, path_distance/5.)

            self.goals = []
            for goal in path_waypoints:
                traj_start_B_traj_waypoint = np.matrix(np.eye(4))
                traj_start_B_traj_waypoint[0, 3] = goal
                origin_B_traj_waypoint = origin_B_traj_start*traj_start_B_traj_waypoint
                self.goals.append(copy.copy(origin_B_traj_waypoint))

            self.set_goals()

            x = pr2_config[0]
            y = pr2_config[1]
            th = pr2_config[2]
            z = pr2_config[3]

            origin_B_pr2 = np.matrix([[ m.cos(th), -m.sin(th),     0.,         x],
                                      [ m.sin(th),  m.cos(th),     0.,         y],
                                      [        0.,         0.,     1.,        0.],
                                      [        0.,         0.,     0.,        1.]])

            self.robot.SetTransform(np.array(origin_B_pr2))
            v = self.robot.GetActiveDOFValues()
            v[self.robot.GetJoint('torso_lift_joint').GetDOFIndex()] = z
            self.robot.SetActiveDOFValues(v, 2)
            self.env.UpdatePublishedBodies()

            rospy.sleep(4.0)

            with self.robot:
                sign_flip = 1.
                if 'right' in self.robot_arm:
                    sign_flip = -1.
                v = self.robot.GetActiveDOFValues()
                v[self.robot.GetJoint(self.robot_opposite_arm[0]+'_shoulder_pan_joint').GetDOFIndex()] = -sign_flip*3.14/2
                v[self.robot.GetJoint(self.robot_opposite_arm[0]+'_shoulder_lift_joint').GetDOFIndex()] = -0.52
                v[self.robot.GetJoint(self.robot_opposite_arm[0]+'_upper_arm_roll_joint').GetDOFIndex()] = 0.
                v[self.robot.GetJoint(self.robot_opposite_arm[0]+'_elbow_flex_joint').GetDOFIndex()] = -3.14*2/3
                v[self.robot.GetJoint(self.robot_opposite_arm[0]+'_forearm_roll_joint').GetDOFIndex()] = 0.
                v[self.robot.GetJoint(self.robot_opposite_arm[0]+'_wrist_flex_joint').GetDOFIndex()] = 0.
                v[self.robot.GetJoint(self.robot_opposite_arm[0]+'_wrist_roll_joint').GetDOFIndex()] = 0.
                self.robot.SetActiveDOFValues(v, 2)
                self.env.UpdatePublishedBodies()
                not_close_to_collision = True
                if self.manip.CheckIndependentCollision(op.CollisionReport()):
                    not_close_to_collision = False

                if not_close_to_collision:
                    for num, Tgrasp in enumerate(self.origin_B_grasps):
                        sols = []
                        sols = self.manip.FindIKSolutions(Tgrasp, filteroptions=op.IkFilterOptions.CheckEnvCollisions)
                        if not list(sols):
                            sign_flip = 1.
                            if 'right' in self.robot_arm:
                                sign_flip = -1.
                            v = self.robot.GetActiveDOFValues()
                            v[self.robot.GetJoint(self.robot_opposite_arm[0]+'_shoulder_pan_joint').GetDOFIndex()] = -sign_flip*0.023593
                            v[self.robot.GetJoint(self.robot_opposite_arm[0]+'_shoulder_lift_joint').GetDOFIndex()] = 1.1072800
                            v[self.robot.GetJoint(self.robot_opposite_arm[0]+'_upper_arm_roll_joint').GetDOFIndex()] = -sign_flip*1.5566882
                            v[self.robot.GetJoint(self.robot_opposite_arm[0]+'_elbow_flex_joint').GetDOFIndex()] = -2.124408
                            v[self.robot.GetJoint(self.robot_opposite_arm[0]+'_forearm_roll_joint').GetDOFIndex()] = -sign_flip*1.4175
                            v[self.robot.GetJoint(self.robot_opposite_arm[0]+'_wrist_flex_joint').GetDOFIndex()] = -1.8417
                            v[self.robot.GetJoint(self.robot_opposite_arm[0]+'_wrist_roll_joint').GetDOFIndex()] = 0.21436
                            self.robot.SetActiveDOFValues(v, 2)
                            self.env.UpdatePublishedBodies()
                            sols = self.manip.FindIKSolutions(Tgrasp, filteroptions=op.IkFilterOptions.CheckEnvCollisions)

                        if list(sols):  # not None:
                            # for solution in sols:
                            #     self.robot.SetDOFValues(solution, self.manip.GetArmIndices())
                            #     self.env.UpdatePublishedBodies()
                            self.robot.SetDOFValues(sols[0], self.manip.GetArmIndices())
                            self.env.UpdatePublishedBodies()
                            rospy.sleep(1.5)

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
            weights[joint] = (1. - m.pow(0.5, ((joint_range[joint])/2. - np.abs((joint_range[joint])/2. - m.degrees(q[joint]) + joint_min[joint]))/(joint_range[joint]/40.)+1.))
        weights[4] = 1.
        weights[6] = 1.
        return np.diag(weights)

if __name__ == "__main__":
    rospy.init_node('score_generator')
    # start_time = time.time()
    outer_start_time = rospy.Time.now()
    selector = ScoreGeneratorDressingwithPhysx(human_arm='rightarm', visualize=False)
    # selector.visualize_many_configurations()
    # selector.output_results_for_use()
    # selector.run_interleaving_optimization_outer_level()
    selector.optimize_entire_dressing_task()
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






