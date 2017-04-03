#!/usr/bin/env python

import numpy as np
import math as m
import openravepy as op
from openravepy.misc import InitOpenRAVELogging
import copy

import time
import roslib
roslib.load_manifest('hrl_base_selection')
roslib.load_manifest('hrl_haptic_mpc')
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
from itertools import combinations as comb
from operator import itemgetter

from sensor_msgs.msg import JointState
from std_msgs.msg import String
# import hrl_lib.transforms as tr
from hrl_base_selection.srv import BaseMove#, BaseMove_multi
from visualization_msgs.msg import Marker, MarkerArray
from helper_functions import createBMatrix, Bmat_to_pos_quat
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from hrl_msgs.msg import FloatArrayBare
from hrl_base_selection.msg import PhysxOutcome
import random, threading

import tf.transformations as tft

import pickle as pkl
roslib.load_manifest('hrl_lib')
from hrl_lib.util import save_pickle, load_pickle
from random import gauss
# import hrl_haptic_mpc.haptic_mpc_util
# from hrl_haptic_mpc.robot_haptic_state_node import RobotHapticStateServer
import hrl_lib.util as ut

import sensor_msgs.point_cloud2 as pc2

import cma

from joblib import Parallel, delayed


class ScoreGeneratorDressingwithPhysx(object):

    def __init__(self, visualize=False):

        self.visualize = visualize
        self.model = 'green_kevin'
        self.frame_lock = threading.RLock()
        self.arm = 'leftarm'
        self.opposite_arm = 'rightarm'

        rospack = rospkg.RosPack()
        self.pkg_path = rospack.get_path('hrl_base_selection')

        self.human_rot_correction = None

        self.human_arm = None
        self.human_manip = None
        self.human_model = None

        # self.model = None
        self.force_cost = 0.

        self.a_model_is_loaded = False
        self.goals = None
        self.pr2_B_reference = None
        self.task = None
        self.task_dict = None

        self.reference_names = None

        self.best_physx_score = 100.
        self.best_openrave_results = []

        self.arm_traj_parameters = []
        self.pr2_parameters = []

        # self.head_angles = []

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
        self.setup_physx_calls()
        self.setup_base_openrave()
        self.setup_human_openrave(self.model)
        self.set_pr2_arm('rightarm')
        self.set_human_arm('rightarm')

    def setup_physx_calls(self):
        self.traj_to_simulator_pub = rospy.Publisher('physx_simulator_input', FloatArrayBare, queue_size=1)
        self.simulator_result_sub = rospy.Subscriber('physx_simulator_result', PhysxOutcome, self.simulator_result_cb)

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

    def simulator_result_cb(self, msg):
        with self.frame_lock:
            self.physx_output = True
            self.physx_outcome = msg.outcome
            forces = msg.forces
            temp_force_cost = 0.
            for force in forces:
                temp_force_cost += np.max([0., force - 1.0])/9.0

            temp_force_cost /= len(forces)
            self.force_cost = copy.copy(temp_force_cost)
            print 'Force cost from physx: ', self.force_cost
            print 'Physx simulation outcome: ', self.physx_outcome

    def run_interleaving_optimization_outer_level(self):

        maxiter = 30
        popsize = m.pow(5, 2)*100
        maxiter = 5
        popsize = 100
        ## OLD: For position, orientation, and path distance
        # cma parameters: [end_effector_trajectory_start_position: x,y,z,
        #                  end_effector_trajectory_start_orientation(euler:zxy): y, r, p,
        #                  end_effector_trajectory_move_distance,
        #                  human_arm_elbow_angle,
        #                  human_upper_arm_quaternion(euler:zyx): y, p, r ]
        # parameters_min = np.array([-2., -2., -2.,
        #                            -m.pi-.001, -m.pi-.001, -m.pi-.001,
        #                            0.,
        #                            0.,
        #                            -0.1, -0.1, -2.0])
        # parameters_max = np.array([2., 2., 2.,
        #                            m.pi+.001, m.pi+.001, m.pi+.001,
        #                            2.,
        #                            m.radians(115.),
        #                            0.1, 2., 0.35])
        # parameters_min = np.array([0.65, -0.02, 0.05,
        #                            m.radians(179), -m.radians(1)-.001, -m.radians(1)-.001,
        #                            0.5,
        #                            0.,
        #                            -0.01, -0.1, m.radians(-100)])
        # parameters_max = np.array([0.8, 0.02, 0.15,
        #                            m.radians(181), m.radians(1)+.001, m.radians(1)+.001,
        #                            1.0,
        #                            m.radians(5.),
        #                            0.01, .1, m.radians(-80)])

        ### Current: Two positions, first with respect to the fist, second with respect to the upper arm, centered at
        # the shoulder and pointing X down the upper arm
        # cma parameters: [end_effector_trajectory_start_position (from fist): x,y,z,
        #                  end_effector_trajectory_end_position (centered at shoulder, pointing down arm): x, y, z,
        #                  human_arm_elbow_angle,
        #                  human_upper_arm_quaternion(euler:zyx): y, p, r ]
        # parameters_min = np.array([-2., -2., -2.,
        #                            -m.pi-.001, -m.pi-.001, -m.pi-.001,
        #                            0.,
        #                            0.,
        #                            -0.1, -0.1, -2.0])
        # parameters_max = np.array([2., 2., 2.,
        #                            m.pi+.001, m.pi+.001, m.pi+.001,
        #                            2.,
        #                            m.radians(115.),
        #                            0.1, 2., 0.35])
        parameters_min = np.array([0.1, -0.07, 0.05,
                                   -0.15, -0.05, 0.05,
                                   0.,
                                   m.radians(-15), -0.1, m.radians(-120)])
        parameters_max = np.array([0.25, 0.07, 0.2,
                                   0.05, 0.05, 0.2,
                                   m.radians(115.),
                                   m.radians(15), m.radians(115), m.radians(10)])
        # parameters_min = np.array([0.1, -0.07, 0.05,
        #                            -0.15, -0.02, 0.05,
        #                            0.,
        #                            -0.01, -0.1, m.radians(-100)])
        # parameters_max = np.array([0.25, 0.07, 0.15,
        #                            0.05, 0.02, 0.15,
        #                            m.radians(5.),
        #                            0.01, .1, m.radians(-80)])
        parameters_scaling = (parameters_max-parameters_min)/4.
        parameters_initialization = (parameters_max+parameters_min)/2.
        parameters_initialization[6] = 0.001
        parameters_initialization[8] = 0.
        parameters_initialization[9] = m.radians(-90)
        parameters_scaling[6] = m.radians(90)
        parameters_scaling[8] = m.radians(110)
        parameters_scaling[9] = m.radians(110)
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
        opt_traj_arm_output = [self.optimization_results[0], self.optimization_results[1]]
        opt_pr2_output = [self.best_openrave_results[0], self.best_openrave_results[1]]

        save_pickle(opt_traj_arm_output, self.pkg_path+'/data/best_trajectory_and_arm_config.pkl')
        save_pickle(opt_pr2_output, self.pkg_path+'/data/best_pr2_config.pkl')

    def objective_function_traj_and_arm_config(self, params):
        # params[7:] = [0., 0., 1.]
        # params = [0., 0.,  0.2,
        #           -0.2,  0.0, 0.2,
        #           0.,
        #           0.0,  m.radians(0), m.radians(-90)]
        # params = [1.71860146e-01,  -9.45158225e-03,   8.73129841e-02,
        #  3.03018418e-02,  -7.06433255e-03,   1.68311316e-01,
        #  1.37321557e-03,  -1.48717942e-01,   4.26887851e-01,
        # -1.95806881e+00]
        testing = False
        if testing:
            th = m.radians(0.)
            test_world_should_B_sleeve_start_trans = np.matrix([[  m.cos(th), 0.,  m.sin(th),     0.38],
                                                                 [         0., 1.,         0.,    -0.32],
                                                                 [ -m.sin(th), 0.,  m.cos(th),     0.1],
                                                                 [         0., 0.,         0.,     1.]])

            th = m.radians(180.)
            test_world_shoulder_B_sleeve_start_rotz = np.matrix([[ m.cos(th), -m.sin(th),     0.,      0.],
                                                                 [ m.sin(th),  m.cos(th),     0.,      0.],
                                                                 [               0.,         0.,     1.,      0.],
                                                                 [        0.,         0.,     0.,        1.]])
            th = m.radians(0.)
            test_world_shoulder_B_sleeve_start_roty = np.matrix([[  m.cos(th), 0.,  m.sin(th),     0.],
                                                                 [         0., 1.,         0.,    0.],
                                                                 [ -m.sin(th), 0.,  m.cos(th),     0.],
                                                                 [         0., 0.,         0.,     1.]])
            th = m.radians(0.)
            test_world_shoulder_B_sleeve_start_rotx = np.matrix([[1.,          0.,          0., 0.],
                                                                 [0.,   m.cos(th),  -m.sin(th), 0.],
                                                                 [0.,   m.sin(th),   m.cos(th), 0.],
                                                                 [0.,          0.,          0., 1.]])
            th = m.radians(-90)
            shoulder_origin_B_rotated_shoulder_rotx = np.matrix([[1.,          0.,          0., 0.],
                                                                 [0.,   m.cos(th),  -m.sin(th), 0.],
                                                                 [0.,   m.sin(th),   m.cos(th), 0.],
                                                                 [0.,          0.,          0., 1.]])
            pos, rot = Bmat_to_pos_quat(test_world_should_B_sleeve_start_trans*test_world_shoulder_B_sleeve_start_rotz*test_world_shoulder_B_sleeve_start_roty)
            pos_r, rot_r = Bmat_to_pos_quat(shoulder_origin_B_rotated_shoulder_rotx)
            # self.set_simulation([0., 0., 0.], [.15, 0., 0.], 0, [0., 0., 0., 0.])
            params = list(flatten([0.7, 0., 0.1, m.radians(180.), 0., 0., 0.85, m.radians(0), m.radians(0.), m.radians(0), m.radians(-90)]))
        pos_ori_distance = False
        if pos_ori_distance:
            path_distance = params[6]
            self.set_human_model_dof([params[8], params[9], -params[10], params[7], 0, 0, 0], 'rightarm', 'green_kevin')
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

            uabr_corrected_B_traj_start = createBMatrix([params[0], params[1], params[2]],
                                              tft.quaternion_from_euler(params[3], params[4], params[5], 'rzyx'))
            pos_t, quat_t = Bmat_to_pos_quat(uabr_corrected_B_traj_start)
            if np.linalg.norm([params[0]-pos_t[0], params[1]-pos_t[1], params[2]-pos_t[2]]) > 0.01:
                print 'Somehow I have mysteriously gotten error in my trajectory start position in openrave. Abort!'
                return 100.
            # uar_base_B_uar = createBMatrix([0, 0, 0], tft.quaternion_from_euler(params[10], params[11], params[12], 'rzxy'))
            hr_B_traj_start = origin_B_hr.I*origin_B_uabr*uabr_B_uabr_corrected*uabr_corrected_B_traj_start
            traj_start_B_traj_end = np.matrix(np.eye(4))
            traj_start_B_traj_end[0, 3] = params[6]
            uabr_B_traj_end = uabr_B_uabr_corrected*uabr_corrected_B_traj_start*traj_start_B_traj_end
            uar_B_traj_end = origin_B_uar.I*origin_B_uabr*uabr_B_traj_end

            # print 'origin_B_hr'
            # print origin_B_hr
            # print 'origin_B_uabr'
            # print origin_B_uabr
            # print 'uabr_corrected_B_traj_start'
            # print uabr_corrected_B_traj_start
            if not testing:
                if np.linalg.norm(hr_B_traj_start[0:3, 3]) > 0.35:
                    print 'traj start too far away'
                    return 10. + np.linalg.norm(hr_B_traj_start[0:3, 3])
                if np.linalg.norm(uabr_B_traj_end[0:3, 3]) > 0.3:
                    print 'traj end too far away'
                    return 10. + np.linalg.norm(uabr_B_traj_end[0:3, 3])
                if hr_B_traj_start[0, 3] > 0.2 or hr_B_traj_start[0, 3] < 0.:
                    print 'traj start relation to hand is too close or too far'
                    return 10. + np.abs(hr_B_traj_start[0, 3])
                if uar_B_traj_end[0, 3] > 0.05:
                    print 'traj ends too far from upper arm'
                    return 10. + uar_B_traj_end[0, 3]
                if uar_B_traj_end[0, 3] < -0.15:
                    print 'traj ends too far behind upper arm'
                    return 10. - uar_B_traj_end[0, 3]
            path_waypoints = np.arange(0., path_distance+path_distance*0.01, path_distance/5.)
            # print 'length of path waypoints'
            # print len(path_waypoints)
            # print path_waypoints
            # print path_waypoints
            self.goals = []
            for goal in path_waypoints:
                traj_start_B_traj_waypoint = np.matrix(np.eye(4))
                traj_start_B_traj_waypoint[0, 3] = goal
                origin_B_traj_waypoint = origin_B_uabr*uabr_B_uabr_corrected*uabr_corrected_B_traj_start*traj_start_B_traj_waypoint
                # print 'origin_B_uabr'
                # print origin_B_uabr
                # print 'uabr_B_uabr_corrected'
                # print uabr_B_uabr_corrected
                # print 'uabr_corrected_B_traj_start'
                # print uabr_corrected_B_traj_start
                # print 'traj_start_B_traj_waypoint'
                # print traj_start_B_traj_waypoint
                # print 'origin_B_traj_waypoint'
                # print origin_B_traj_waypoint
                self.goals.append(copy.copy(origin_B_traj_waypoint))
                min_distance = np.min([np.linalg.norm((origin_B_uar.I*origin_B_traj_waypoint)[1:3, 3]),
                                       np.linalg.norm((origin_B_far.I*origin_B_traj_waypoint)[1:3, 3]),
                                       np.linalg.norm((origin_B_hr.I*origin_B_traj_waypoint)[1:3, 3])])
                if not testing:
                    if min_distance > 0.28:  # goals too far from any arm segment. The gown sleeve opening is 28 cm long
                        print 'goals too far from arm segments'
                        return 10 + min_distance  # distance from arm segment
        else:  # Based on two 3-D positions
            # path_distance = np.linalg.norm(np.array(params[0:3])-np.array(params[3:6]))
            print 'params'
            print params
            self.set_human_model_dof([params[7], params[8], -params[9], params[6], 0, 0, 0], 'rightarm', 'green_kevin')
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

            # print 'origin_B_uabr_corrected'
            # print origin_B_uabr*uabr_B_uabr_corrected

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
                if np.linalg.norm(hr_rotated_B_traj_start_pos[0:3, 3]) > 0.3:
                    print 'traj start too far away'
                    return 10. + np.linalg.norm(hr_rotated_B_traj_start_pos[0:3, 3])
                if np.linalg.norm(rotated_pointed_down_shoulder_B_traj_end_pos[0:3, 3]) > 0.3:
                    print 'traj end too far away'
                    return 10. + np.linalg.norm(rotated_pointed_down_shoulder_B_traj_end_pos[0:3, 3])
                if hr_rotated_B_traj_start_pos[0, 3] > 0.3 or hr_rotated_B_traj_start_pos[0, 3] < 0.:
                    print 'traj start relation to hand is too close or too far'
                    return 10. + np.abs(hr_rotated_B_traj_start_pos[0, 3])
                if rotated_pointed_down_shoulder_B_traj_end_pos[0, 3] > 0.05:
                    print 'traj ends too far from upper arm'
                    return 10. + rotated_pointed_down_shoulder_B_traj_end_pos[0, 3]
                if rotated_pointed_down_shoulder_B_traj_end_pos[0, 3] < -0.15:
                    print 'traj ends too far behind upper arm'
                    return 10. - rotated_pointed_down_shoulder_B_traj_end_pos[0, 3]
            testing = False
            path_waypoints = np.arange(0., path_distance+path_distance*0.01, path_distance/5.)
            # print 'path_distance'
            # print path_distance
            # print 'number of path waypoints'
            # print len(path_waypoints)
            # print 'path_waypoints'
            # print path_waypoints
            self.goals = []
            for goal in path_waypoints:
                traj_start_B_traj_waypoint = np.matrix(np.eye(4))
                traj_start_B_traj_waypoint[0, 3] = goal
                origin_B_traj_waypoint = copy.copy(np.matrix(origin_B_traj_start)*np.matrix(traj_start_B_traj_waypoint))
                # print 'origin_B_traj_start'
                # print origin_B_traj_start
                # print 'traj_start_B_traj_waypoint'
                # print traj_start_B_traj_waypoint
                # print 'origin_B_traj_waypoint'
                # print origin_B_traj_waypoint
                # print 'origin_B_uabr'
                # print origin_B_uabr
                # print 'uabr_B_uabr_corrected'
                # print uabr_B_uabr_corrected
                # print 'uabr_corrected_B_traj_start'
                # print uabr_corrected_B_traj_start
                # print 'traj_start_B_traj_waypoint'
                # print traj_start_B_traj_waypoint
                # print 'origin_B_traj_waypoint'
                # print origin_B_traj_waypoint
                self.goals.append(copy.copy(origin_B_traj_waypoint))
                min_distance = np.min([np.linalg.norm((origin_B_uar.I*origin_B_traj_waypoint)[1:3, 3]),
                                       np.linalg.norm((origin_B_far.I*origin_B_traj_waypoint)[1:3, 3]),
                                       np.linalg.norm((origin_B_hr.I*origin_B_traj_waypoint)[1:3, 3])])
                if not testing:
                    if min_distance > 0.28:  # goals too far from any arm segment. The gown sleeve opening is 28 cm long
                        print 'goals too far from arm segments'
                        return 10 + min_distance  # distance from arm segment

        start_time = rospy.Time.now()
        self.set_goals()
        # print self.origin_B_grasps
        maxiter = 5
        popsize = 4*20
        # cma parameters: [pr2_base_x, pr2_base_y, pr2_base_theta, pr2_base_height,
        # human_arm_dof_1, human_arm_dof_2, human_arm_dof_3, human_arm_dof_4, human_arm_dof_5,
        # human_arm_dof_6, human_arm_dof_7]
        parameters_min = np.array([-2., -2., -m.pi-.001, 0.])
        parameters_max = np.array([2., 2., m.pi+.001, 0.3])
        parameters_scaling = (parameters_max-parameters_min)/4.
        parameters_initialization = (parameters_max+parameters_min)/2.
        opts1 = {'seed': 1234, 'ftarget': -1., 'popsize': popsize, 'maxiter': maxiter, 'maxfevals': 1e8,
                 'CMA_cmean': 0.25,
                 'verb_filenameprefix': 'outcma_pr2_base',
                 'scaling_of_variables': list(parameters_scaling),
                 'tolfun': 1e-3,
                 'bounds': [list(parameters_min), list(parameters_max)]}
        self.openrave_optimization_results = cma.fmin(self.objective_function_one_config,
                                                      list(parameters_initialization),
                                                      1.,
                                                      options=opts1)

        self.pr2_parameters.append([self.openrave_optimization_results[0], self.openrave_optimization_results[1]])
        save_pickle(self.pr2_parameters, self.pkg_path+'/data/all_pr2_configs.pkl')
        elapsed_time = rospy.Time.now()-start_time
        print 'Done with openrave round. Time elapsed:', elapsed_time.to_sec()
        print 'Openrave results:'
        print self.openrave_optimization_results
        if self.openrave_optimization_results[1] < 0.:
            pos_a, quat_a = Bmat_to_pos_quat(origin_B_world_rotated_shoulder)
            # print 'quat_a'
            # print quat_a
            # quat_a =
            # quat_a = list(tft.quaternion_from_euler(params[7], -params[8], params[9], 'rzxy'))
            # print 'quat_a'
            # print quat_a
            out_msg = FloatArrayBare()
            out_msg.data = [float(t) for t in list(flatten([pos_t, quat_t, path_distance, params[6], quat_a]))]
                # float(list(flatten([pos_t, quat_t, params[6], params[7], quat_a])))
            self.physx_output = None
            self.traj_to_simulator_pub.publish(out_msg)
            while not rospy.is_shutdown() and not self.physx_output:
                rospy.sleep(0.5)
                # print 'waiting a sec'
            if self.physx_outcome == 'good':
                with self.frame_lock:
                    self.physx_outcome = None
                    self.physx_output = None
                    alpha = 1.  # cost on forces
                    beta = 1.  # cost on manipulability
                    physx_score = self.force_cost*alpha + self.openrave_optimization_results[1]*beta
                    if physx_score < self.best_physx_score:
                        self.best_physx_score = physx_score
                        self.best_openrave_results = self.openrave_optimization_results
                    self.arm_traj_parameters.append([params, physx_score])

                    save_pickle(self.arm_traj_parameters, self.pkg_path+'/data/all_arm_traj_configs.pkl')
                    print 'Force cost was: ', self.force_cost
                    print 'Physx score was: ', physx_score
                    return physx_score
        with self.frame_lock:
            self.physx_outcome = None
            self.physx_output = None
            alpha = 1.  # cost on forces
            beta = 1.  # cost on manipulability
            self.arm_traj_parameters.append([params, 10. + self.force_cost*alpha + self.openrave_optimization_results[1]*beta])
            save_pickle(self.arm_traj_parameters, self.pkg_path+'/data/all_arm_traj_configs.pkl')
            physx_score = 10. + self.force_cost*alpha + self.openrave_optimization_results[1]*beta
            print 'Force cost was: ', self.force_cost
            print 'Physx score was: ', physx_score
            return physx_score
        # else:
        #     print ''

            # optimization_results[<model>, <number_of_configs>, <head_rest_angle>, <headx>, <heady>, <allow_bed_movement>]

    def calculate_scores(self, task_dict, model, ref_options):
        self.model = model
        self.task_dict = task_dict
        self.reference_names = ref_options

        self.setup_human_openrave(model)

        self.handle_score_generation()

    def old_stuff(self):
        # The reference frame for the pr2 base link
        origin_B_pr2 = np.matrix(np.eye(4))
        pr2_B_head = []
        # Sets the wheelchair location based on the location of the head using a few homogeneous transforms.
        # This is only used to visualize in rviz, because the visualization is done before initializing openrave
        self.origin_B_references = []
        if self.model == 'green_kevin':
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
        else:
            print 'I GOT A BAD MODEL. NOT SURE WHAT TO DO NOW!'
        # origin_B_head = np.matrix(headmodel.GetTransform())

        for y in self.reference_names:
            if y == 'head':
                self.origin_B_references.append(origin_B_head)
            elif y == 'base_link':
                self.origin_B_references.append(origin_B_pr2)
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
            elif y is None:
                self.origin_B_references.append(np.matrix(np.eye(4)))
        # Sets the wheelchair location based on the location of the head using a few homogeneous transforms.
        self.pr2_B_headfloor = np.matrix([[       1.,        0.,   0.,         0.],
                                          [       0.,        1.,   0.,         0.],
                                          [       0.,        0.,   1.,         0.],
                                          [       0.,        0.,   0.,         1.]])

        self.selection_mat = []
        self.reference_mat = []
        self.origin_B_grasps = []
        self.weights = []
        self.goal_list = []
        if self.goals is not None:
            self.number_goals = len(self.goals)
            print 'Score generator received a list of desired goal locations on initialization. ' \
                  # 'It contains ', len(goals), ' goal locations.'
            self.selection_mat = np.zeros(len(self.goals))
            self.goal_list = np.zeros([len(self.goals), 4, 4])
            self.reference_mat = np.zeros(len(self.goals))
            for it in xrange(len(self.goals)):
                #self.goal_list.append(pr2_B_head*np.matrix(target[0])*goal_B_gripper)
                self.reference_mat[it] = int(self.goals[it, 2])

                # goal_list is origin_B_goal
                self.goal_list[it] = copy.copy(self.origin_B_references[int(self.reference_mat[it])]*np.matrix(self.goals[it, 0]))
                self.selection_mat[it] = self.goals[it, 1]
            self.set_goals()

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
            self.origin_B_grasps.append(np.array(np.matrix(self.goals[num])*np.matrix(self.gripper_B_tool.I)*np.matrix(self.goal_B_gripper)))
            # print 'self.goals[', num, ']'
            # print self.goals[num]
            # print 'self.origin_B_grasps[', num, ']'
            # print self.origin_B_grasps[num]

    def set_pr2_arm(self, arm):
        ## Set robot manipulators, ik, planner
        print 'Setting the arm being used by base selection to ', arm
        self.arm = arm
        if arm == 'leftarm':
            self.opposite_arm = 'rightarm'
        elif arm == 'rightarm':
            self.opposite_arm = 'leftarm'
        else:
            print 'ERROR'
            print 'I do not know what arm to be using'
            return None
        self.robot.SetActiveManipulator(arm)
        self.manip = self.robot.GetActiveManipulator()
        ikmodel = op.databases.inversekinematics.InverseKinematicsModel(self.robot, iktype=op.IkParameterization.Type.Transform6D)
        if not ikmodel.load():
            print 'IK model not found. Will now generate an IK model. This will take a while!'
            ikmodel.autogenerate()
        self.manipprob = op.interfaces.BaseManipulation(self.robot)

    def set_human_arm(self, arm):
        ## Set robot manipulators, ik, planner
        print 'Setting the human arm being used by base selection to ', arm
        self.human_arm = arm
        # self.human_model.SetActiveManipulator(arm)
        # self.human_manip = self.human_model.GetActiveManipulator()

    def objective_function_one_config(self, current_parameters):
        # start_time = rospy.Time.now()
        # current_parameters = [0.70538102, -0.36536968, -1.4965914,  0.12652826]
        x = current_parameters[0]
        y = current_parameters[1]
        th = current_parameters[2]
        z = current_parameters[3]

        origin_B_pr2 = np.matrix([[ m.cos(th), -m.sin(th),     0.,         x],
                                  [ m.sin(th),  m.cos(th),     0.,         y],
                                  [        0.,         0.,     1.,        0.],
                                  [        0.,         0.,     0.,        1.]])
        self.robot.SetTransform(np.array(origin_B_pr2))
        v = self.robot.GetActiveDOFValues()
        v[self.robot.GetJoint('torso_lift_joint').GetDOFIndex()] = z
        self.robot.SetActiveDOFValues(v, 2)
        self.env.UpdatePublishedBodies()

        # return -0.5

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

        #print 'Time to update autobed things: %fs'%(time.time()-starttime)
        reach_score = 0.
        manip_score = 0.
        goal_scores = []
        # std = 1.
        # mean = 0.
        # allmanip = []
        manip = 0.
        reached = 0.

        #allmanip2=[]
        # space_score = (1./(std*(m.pow((2.*m.pi), 0.5))))*m.exp(-(m.pow(np.linalg.norm([x, y])-mean, 2.)) /
        #                                                        (2.*m.pow(std, 2.)))
        #print space_score
        with self.robot:
            sign_flip = 1.
            if 'right' in self.arm:
                sign_flip = -1.
            v = self.robot.GetActiveDOFValues()
            v[self.robot.GetJoint(self.opposite_arm[0]+'_shoulder_pan_joint').GetDOFIndex()] = -sign_flip*3.14/2
            v[self.robot.GetJoint(self.opposite_arm[0]+'_shoulder_lift_joint').GetDOFIndex()] = -0.52
            v[self.robot.GetJoint(self.opposite_arm[0]+'_upper_arm_roll_joint').GetDOFIndex()] = 0.
            v[self.robot.GetJoint(self.opposite_arm[0]+'_elbow_flex_joint').GetDOFIndex()] = -3.14*2/3
            v[self.robot.GetJoint(self.opposite_arm[0]+'_forearm_roll_joint').GetDOFIndex()] = 0.
            v[self.robot.GetJoint(self.opposite_arm[0]+'_wrist_flex_joint').GetDOFIndex()] = 0.
            v[self.robot.GetJoint(self.opposite_arm[0]+'_wrist_roll_joint').GetDOFIndex()] = 0.
            self.robot.SetActiveDOFValues(v, 2)
            self.env.UpdatePublishedBodies()
            not_close_to_collision = True
            if self.manip.CheckIndependentCollision(op.CollisionReport()):
                not_close_to_collision = False
            check_if_base_is_near_collision = False
            if check_if_base_is_near_collision:
                origin_B_pr2 = np.matrix([[ m.cos(th), -m.sin(th),     0., x+.04],
                                          [ m.sin(th),  m.cos(th),     0., y+.04],
                                          [        0.,         0.,     1.,        0.],
                                          [        0.,         0.,     0.,        1.]])
                self.robot.SetTransform(np.array(origin_B_pr2))
                self.env.UpdatePublishedBodies()
                if self.manip.CheckIndependentCollision(op.CollisionReport()):
                    not_close_to_collision = False

                origin_B_pr2 = np.matrix([[ m.cos(th), -m.sin(th),     0., x-.04],
                                          [ m.sin(th),  m.cos(th),     0., y+.04],
                                          [        0.,         0.,     1.,        0.],
                                          [        0.,         0.,     0.,        1.]])
                self.robot.SetTransform(np.array(origin_B_pr2))
                self.env.UpdatePublishedBodies()
                if self.manip.CheckIndependentCollision(op.CollisionReport()):
                    not_close_to_collision = False

                origin_B_pr2 = np.matrix([[ m.cos(th), -m.sin(th),     0., x-.04],
                                          [ m.sin(th),  m.cos(th),     0., y-.04],
                                          [        0.,         0.,     1.,        0.],
                                          [        0.,         0.,     0.,        1.]])
                self.robot.SetTransform(np.array(origin_B_pr2))
                self.env.UpdatePublishedBodies()
                if self.manip.CheckIndependentCollision(op.CollisionReport()):
                    not_close_to_collision = False

                origin_B_pr2 = np.matrix([[ m.cos(th), -m.sin(th),     0., x+.04],
                                          [ m.sin(th),  m.cos(th),     0., y-.04],
                                          [        0.,         0.,     1.,        0.],
                                          [        0.,         0.,     0.,        1.]])
                self.robot.SetTransform(np.array(origin_B_pr2))
                self.env.UpdatePublishedBodies()
                if self.manip.CheckIndependentCollision(op.CollisionReport()):
                    not_close_to_collision = False

                origin_B_pr2 = np.matrix([[ m.cos(th), -m.sin(th),     0., x],
                                          [ m.sin(th),  m.cos(th),     0., y],
                                          [        0.,         0.,     1.,        0.],
                                          [        0.,         0.,     0.,        1.]])
                self.robot.SetTransform(np.array(origin_B_pr2))
                self.env.UpdatePublishedBodies()

            if not_close_to_collision:
                reached = np.zeros(len(self.origin_B_grasps))
                manip = np.zeros(len(self.origin_B_grasps))
                for num, Tgrasp in enumerate(self.origin_B_grasps):
                    sols = []
                    sols = self.manip.FindIKSolutions(Tgrasp, filteroptions=op.IkFilterOptions.CheckEnvCollisions)
                    if not list(sols):
                        sign_flip = 1.
                        if 'right' in self.arm:
                            sign_flip = -1.
                        v = self.robot.GetActiveDOFValues()
                        v[self.robot.GetJoint(self.opposite_arm[0]+'_shoulder_pan_joint').GetDOFIndex()] = -sign_flip*0.023593
                        v[self.robot.GetJoint(self.opposite_arm[0]+'_shoulder_lift_joint').GetDOFIndex()] = 1.1072800
                        v[self.robot.GetJoint(self.opposite_arm[0]+'_upper_arm_roll_joint').GetDOFIndex()] = -sign_flip*1.5566882
                        v[self.robot.GetJoint(self.opposite_arm[0]+'_elbow_flex_joint').GetDOFIndex()] = -2.124408
                        v[self.robot.GetJoint(self.opposite_arm[0]+'_forearm_roll_joint').GetDOFIndex()] = -sign_flip*1.4175
                        v[self.robot.GetJoint(self.opposite_arm[0]+'_wrist_flex_joint').GetDOFIndex()] = -1.8417
                        v[self.robot.GetJoint(self.opposite_arm[0]+'_wrist_roll_joint').GetDOFIndex()] = 0.21436
                        self.robot.SetActiveDOFValues(v, 2)
                        self.env.UpdatePublishedBodies()
                        sols = self.manip.FindIKSolutions(Tgrasp, filteroptions=op.IkFilterOptions.CheckEnvCollisions)

                    if list(sols):  # not None:
                        reached[num] = 1.
                        for solution in sols:
                            self.robot.SetDOFValues(solution, self.manip.GetArmIndices())
                            self.env.UpdatePublishedBodies()
                            # if self.visualize:
                            #     rospy.sleep(0.5)
                            J = np.matrix(np.vstack([self.manip.CalculateJacobian(), self.manip.CalculateAngularVelocityJacobian()]))
                            try:
                                joint_limit_weight = self.gen_joint_limit_weight(solution, self.arm)
                                manip[num] = np.max([copy.copy((m.pow(np.linalg.det(J*joint_limit_weight*J.T), (1./6.)))/(np.trace(J*joint_limit_weight*J.T)/6.)), manip[num]])
                            except ValueError:
                                print 'WARNING!!'
                                print 'Jacobian may be singular or close to singular'
                                print 'Determinant of J*JT is: ', np.linalg.det(J*J.T)
                                manip[num] = np.max([0., manip[num]])
                        if self.visualize:
                            rospy.sleep(1.5)
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

    def setup_base_openrave(self):
        # Setup Openrave ENV
        InitOpenRAVELogging()
        self.env = op.Environment()

        # Lets you visualize openrave. Uncomment to see visualization. Does not work through ssh.
        if self.visualize:
            self.env.SetViewer('qtcoin')

        ## Set up robot state node to do Jacobians. This works, but is commented out because we can do it with openrave
        #  fine.
        #torso_frame = '/torso_lift_link'
        #inertial_frame = '/base_link'
        #end_effector_frame = '/l_gripper_tool_frame'
        #from pykdl_utils.kdl_kinematics import create_kdl_kin
        #self.kinematics = create_kdl_kin(torso_frame, end_effector_frame)

        ## Load OpenRave PR2 Model
        self.env.Load('robots/pr2-beta-static.zae')
        self.robot = self.env.GetRobots()[0]
        self.robot.CheckLimitsAction=2
        sign_flip = 1.
        if 'right' in self.arm:
            sign_flip = -1.
        v = self.robot.GetActiveDOFValues()
        v[self.robot.GetJoint(self.arm[0]+'_shoulder_pan_joint').GetDOFIndex()] = sign_flip*3.14/2
        v[self.robot.GetJoint(self.opposite_arm[0]+'_shoulder_pan_joint').GetDOFIndex()] = -sign_flip*3.14/2
        v[self.robot.GetJoint(self.opposite_arm[0]+'_shoulder_lift_joint').GetDOFIndex()] = -0.52
        v[self.robot.GetJoint(self.opposite_arm[0]+'_upper_arm_roll_joint').GetDOFIndex()] = 0.
        v[self.robot.GetJoint(self.opposite_arm[0]+'_elbow_flex_joint').GetDOFIndex()] = -3.14*2/3
        v[self.robot.GetJoint(self.opposite_arm[0]+'_forearm_roll_joint').GetDOFIndex()] = 0.
        v[self.robot.GetJoint(self.opposite_arm[0]+'_wrist_flex_joint').GetDOFIndex()] = 0.
        v[self.robot.GetJoint(self.opposite_arm[0]+'_wrist_roll_joint').GetDOFIndex()] = 0.
        v[self.robot.GetJoint(self.arm[0]+'_gripper_l_finger_joint').GetDOFIndex()] = .54
        v[self.robot.GetJoint('torso_lift_joint').GetDOFIndex()] = .3
        self.robot.SetActiveDOFValues(v, 2)
        robot_start = np.matrix([[1., 0., 0., 0.],
                                 [0., 1., 0., 0.],
                                 [0., 0., 1., 0.],
                                 [0., 0., 0., 1.]])
        self.robot.SetTransform(np.array(robot_start))

        ## Set robot manipulators, ik, planner
        self.robot.SetActiveManipulator('leftarm')
        self.manip = self.robot.GetActiveManipulator()
        ikmodel = op.databases.inversekinematics.InverseKinematicsModel(self.robot, iktype=op.IkParameterization.Type.Transform6D)
        # ikmodel = op.databases.inversekinematics.InverseKinematicsModel(self.robot, iktype=op.IkParameterization.Type.Transform6D)
        if not ikmodel.load():
            print 'IK model not found for leftarm. Will now generate an IK model. This will take a while!'
            ikmodel.autogenerate()

        self.robot.SetActiveManipulator('rightarm')
        self.manip = self.robot.GetActiveManipulator()
        ikmodel = op.databases.inversekinematics.InverseKinematicsModel(self.robot, iktype=op.IkParameterization.Type.Transform6D)
        if not ikmodel.load():
            print 'IK model not found for rightarm. Will now generate an IK model. This will take a while!'
            ikmodel.autogenerate()

        # create the interface for basic manipulation programs
        self.manipprob = op.interfaces.BaseManipulation(self.robot)

    def setup_human_openrave(self, model):

        # Find and load Wheelchair Model
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('hrl_base_selection')

        # Transform from the coordinate frame of the wc model in the back right bottom corner, to the head location on the floor
        if model == 'green_kevin':

            '''
            self.env.Load(''.join([pkg_path, '/collada/wheelchair_and_body_assembly.dae']))
            originsubject_B_headfloor = np.matrix([[m.cos(0.), -m.sin(0.),  0.,      0.], #.45 #.438
                                                   [m.sin(0.),  m.cos(0.),  0.,      0.], #0.34 #.42
                                                   [       0.,         0.,  1.,      0.],
                                                   [       0.,         0.,  0.,      1.]])
            '''
            # This is the new wheelchair model
            # self.env.Load(''.join([pkg_path, '/collada/wheelchair_and_body_assembly.dae']))
            # self.env.Load(''.join([pkg_path, '/collada/human_model_rounded.dae']))
            human_urdf_path = pkg_path + '/urdf/green_kevin/robots/green_kevin.urdf'
            human_srdf_path = pkg_path + '/urdf/green_kevin/robots/green_kevin.srdf'
            module = op.RaveCreateModule(self.env, 'urdf')
            name = module.SendCommand('LoadURI '+human_urdf_path+' '+human_srdf_path)
            self.human_model = self.env.GetRobots()[1]
            # self.set_human_arm('rightarm')
            # self.human_model.SetActiveManipulator('leftarm')

            rotx_correction = np.matrix([[1., 0., 0., 0.],
                                         [0., 0., -1., 0.],
                                         [0., 1., 0., 0.],
                                         [0., 0., 0., 1.]])

            roty_correction = np.matrix([[0., 0., 1., 0.],
                                         [0., 1., 0., 0.],
                                         [-1., 0., 0., 0.],
                                         [0., 0., 0., 1.]])

            self.human_rot_correction = roty_correction*rotx_correction

            # human_trans_start = np.matrix([[1., 0., 0., 0.],
            #                                [0., 1., 0., 0.],
            #                                [0., 0., 1., 1.45],
            #                                [0., 0., 0., 1.]])
            human_trans_start = np.matrix([[1., 0., 0., 0.],
                                           [0., 1., 0., 0.],
                                           [0., 0., 1., 1.05],
                                           [0., 0., 0., 1.]])
            self.human_model.SetTransform(np.array(human_trans_start*self.human_rot_correction))
            self.originsubject_B_originworld = np.matrix(human_trans_start*self.human_rot_correction).I

            v = self.human_model.GetActiveDOFValues()
            v[self.human_model.GetJoint('green_kevin/neck_twist_joint').GetDOFIndex()] = 0  # m.radians(60)
            v[self.human_model.GetJoint('green_kevin/neck_tilt_joint').GetDOFIndex()] = 0.0
            v[self.human_model.GetJoint('green_kevin/neck_head_rotz_joint').GetDOFIndex()] = 0  # -m.radians(30)
            v[self.human_model.GetJoint('green_kevin/neck_head_roty_joint').GetDOFIndex()] = -0.0
            v[self.human_model.GetJoint('green_kevin/neck_head_rotx_joint').GetDOFIndex()] = 0
            v[self.human_model.GetJoint('green_kevin/neck_body_joint').GetDOFIndex()] = -0.
            v[self.human_model.GetJoint('green_kevin/upper_mid_body_joint').GetDOFIndex()] = 0.
            v[self.human_model.GetJoint('green_kevin/mid_lower_body_joint').GetDOFIndex()] = 0.
            v[self.human_model.GetJoint('green_kevin/body_quad_left_joint').GetDOFIndex()] = 0.
            v[self.human_model.GetJoint('green_kevin/body_quad_right_joint').GetDOFIndex()] = 0.
            v[self.human_model.GetJoint('green_kevin/quad_calf_left_joint').GetDOFIndex()] = 0.
            v[self.human_model.GetJoint('green_kevin/quad_calf_right_joint').GetDOFIndex()] = 0.
            v[self.human_model.GetJoint('green_kevin/calf_foot_left_joint').GetDOFIndex()] = 0.
            v[self.human_model.GetJoint('green_kevin/calf_foot_right_joint').GetDOFIndex()] = 0.
            v[self.human_model.GetJoint('green_kevin/body_arm_right_rotx_joint').GetDOFIndex()] = 0.
            v[self.human_model.GetJoint('green_kevin/body_arm_right_rotz_joint').GetDOFIndex()] = 0.
            v[self.human_model.GetJoint('green_kevin/body_arm_right_roty_joint').GetDOFIndex()] = 0.
            v[self.human_model.GetJoint('green_kevin/arm_forearm_right_joint').GetDOFIndex()] = 0.
            v[self.human_model.GetJoint('green_kevin/forearm_hand_right_rotx_joint').GetDOFIndex()] = 0.
            v[self.human_model.GetJoint('green_kevin/forearm_hand_right_roty_joint').GetDOFIndex()] = 0.
            v[self.human_model.GetJoint('green_kevin/forearm_hand_right_rotz_joint').GetDOFIndex()] = 0.
            v[self.human_model.GetJoint('green_kevin/body_arm_left_rotx_joint').GetDOFIndex()] = 0.
            v[self.human_model.GetJoint('green_kevin/body_arm_left_rotz_joint').GetDOFIndex()] = 0.
            v[self.human_model.GetJoint('green_kevin/body_arm_left_roty_joint').GetDOFIndex()] = 0.
            v[self.human_model.GetJoint('green_kevin/arm_forearm_left_joint').GetDOFIndex()] = 0.
            v[self.human_model.GetJoint('green_kevin/forearm_hand_left_rotx_joint').GetDOFIndex()] = 0.
            v[self.human_model.GetJoint('green_kevin/forearm_hand_left_roty_joint').GetDOFIndex()] = 0.
            v[self.human_model.GetJoint('green_kevin/forearm_hand_left_rotz_joint').GetDOFIndex()] = 0.
            self.human_model.SetActiveDOFValues(v, 2)
            self.env.UpdatePublishedBodies()
            # headmodel = self.human_model.GetLink('green_kevin/head_link')
            # head_T = np.matrix(headmodel.GetTransform())
        else:
            self.a_model_is_loaded = False
            print 'I got a bad model. What is going on???'
            return None
        print 'OpenRave has succesfully been initialized. \n'

    def set_human_model_dof(self, dof, human_arm, model):
        # bth = m.degrees(headrest_th)
        if not len(dof) == 7:
            print 'There should be exactly 7 values used for arm configuration. ' \
                  'But instead ' + str(len(dof)) + 'was sent. This is a problem!'

        v = self.human_model.GetActiveDOFValues()
        if human_arm == 'leftarm' and model == 'green_kevin':
            v[self.human_model.GetJoint('green_kevin/body_arm_left_rotx_joint').GetDOFIndex()] = dof[0]
            v[self.human_model.GetJoint('green_kevin/body_arm_left_roty_joint').GetDOFIndex()] = dof[1]
            v[self.human_model.GetJoint('green_kevin/body_arm_left_rotz_joint').GetDOFIndex()] = dof[2]
            v[self.human_model.GetJoint('green_kevin/arm_forearm_left_joint').GetDOFIndex()] = dof[3]
            v[self.human_model.GetJoint('green_kevin/forearm_hand_left_rotx_joint').GetDOFIndex()] = dof[4]
            v[self.human_model.GetJoint('green_kevin/forearm_hand_left_roty_joint').GetDOFIndex()] = dof[5]
            v[self.human_model.GetJoint('green_kevin/forearm_hand_left_rotz_joint').GetDOFIndex()] = dof[6]
        elif human_arm == 'rightarm' and model == 'green_kevin':
            v[self.human_model.GetJoint('green_kevin/body_arm_right_rotx_joint').GetDOFIndex()] = dof[0]
            v[self.human_model.GetJoint('green_kevin/body_arm_right_roty_joint').GetDOFIndex()] = dof[1]
            v[self.human_model.GetJoint('green_kevin/body_arm_right_rotz_joint').GetDOFIndex()] = dof[2]
            v[self.human_model.GetJoint('green_kevin/arm_forearm_right_joint').GetDOFIndex()] = dof[3]
            v[self.human_model.GetJoint('green_kevin/forearm_hand_right_rotx_joint').GetDOFIndex()] = dof[4]
            v[self.human_model.GetJoint('green_kevin/forearm_hand_right_roty_joint').GetDOFIndex()] = dof[5]
            v[self.human_model.GetJoint('green_kevin/forearm_hand_right_rotz_joint').GetDOFIndex()] = dof[6]
        else:
            print 'Either Im not sure what arm or what model to use to set the arm dof for!'
        self.human_model.SetActiveDOFValues(v)
        self.env.UpdatePublishedBodies()

    def set_human_configuration(self, head_height):
        # bth = m.degrees(headrest_th)

        human_transform = np.matrix([[1., 0., 0., 0.],
                                     [0., 1., 0., 0.],
                                     [0., 0., 1., head_height],
                                     [0., 0., 0., 1.]])
        self.human_model.SetTransform(np.array(human_transform))
        bth = 0
        v = self.human_model.GetActiveDOFValues()
            # 0 degrees, 0 height
        if (head_height >= 0.) and (head_height <= 40.):  # between 0 and 40 degrees
            # v[self.human_model.GetJoint('green_kevin/neck_twist_joint').GetDOFIndex()] = -((bth/40)*(0 - 0)+0)
            v[self.human_model.GetJoint('green_kevin/neck_tilt_joint').GetDOFIndex()] = ((bth/40)*(.7 - 0)+0)
            v[self.human_model.GetJoint('green_kevin/neck_body_joint').GetDOFIndex()] = (bth/40)*(.02-(0))+(0)
            # v[self.human_model.GetJoint('green_kevin/neck_head_rotz_joint').GetDOFIndex()] = -((bth/40)*(0 - 0)+0)
            v[self.human_model.GetJoint('green_kevin/neck_head_roty_joint').GetDOFIndex()] = -((bth/40)*(-0.2 - 0)+0)
            v[self.human_model.GetJoint('green_kevin/neck_head_rotx_joint').GetDOFIndex()] = -((bth/40)*(0 - 0)+0)
            v[self.human_model.GetJoint('green_kevin/upper_mid_body_joint').GetDOFIndex()] = (bth/40)*(0.5-0)+0
            v[self.human_model.GetJoint('green_kevin/mid_lower_body_joint').GetDOFIndex()] = (bth/40)*(0.26-0)+(0)
            v[self.human_model.GetJoint('green_kevin/body_quad_left_joint').GetDOFIndex()] = -0.05
            v[self.human_model.GetJoint('green_kevin/body_quad_right_joint').GetDOFIndex()] = -0.05
            v[self.human_model.GetJoint('green_kevin/quad_calf_left_joint').GetDOFIndex()] = .05
            v[self.human_model.GetJoint('green_kevin/quad_calf_right_joint').GetDOFIndex()] = .05
            v[self.human_model.GetJoint('green_kevin/calf_foot_left_joint').GetDOFIndex()] = (bth/40)*(.0-0)+0
            v[self.human_model.GetJoint('green_kevin/calf_foot_right_joint').GetDOFIndex()] = (bth/40)*(.0-0)+0
            v[self.human_model.GetJoint('green_kevin/body_arm_left_joint').GetDOFIndex()] = (bth/40)*(-0.15-(-0.15))+(-0.15)
            v[self.human_model.GetJoint('green_kevin/body_arm_right_joint').GetDOFIndex()] = (bth/40)*(-0.15-(-0.15))+(-0.15)
            v[self.human_model.GetJoint('green_kevin/arm_forearm_left_joint').GetDOFIndex()] = (bth/40)*(.86-0.1)+0.1
            v[self.human_model.GetJoint('green_kevin/arm_forearm_right_joint').GetDOFIndex()] = (bth/40)*(.86-0.1)+0.1
            v[self.human_model.GetJoint('green_kevin/forearm_hand_left_joint').GetDOFIndex()] = 0
            v[self.human_model.GetJoint('green_kevin/forearm_hand_right_joint').GetDOFIndex()] = 0
        elif (head_height > 40.) and (head_height <= 80.):  # between 0 and 40 degrees
            v[self.human_model.GetJoint('green_kevin/neck_tilt_joint').GetDOFIndex()] = (((bth-40)/40)*(0.7 - 0.7)+0.7)
            v[self.human_model.GetJoint('green_kevin/neck_body_joint').GetDOFIndex()] = ((bth-40)/40)*(-0.1-(.02))+(.02)
            # v[self.human_model.GetJoint('green_kevin/neck_head_rotz_joint').GetDOFIndex()] = -((bth/40)*(0 - 0)+0)
            v[self.human_model.GetJoint('green_kevin/neck_head_roty_joint').GetDOFIndex()] = -((bth/40)*(.02 - (-0.2))+(-0.2))
            v[self.human_model.GetJoint('green_kevin/neck_head_rotx_joint').GetDOFIndex()] = -((bth/40)*(0 - 0)+0)
            v[self.human_model.GetJoint('green_kevin/upper_mid_body_joint').GetDOFIndex()] = ((bth-40)/40)*(.7-(.5))+(.5)
            v[self.human_model.GetJoint('green_kevin/mid_lower_body_joint').GetDOFIndex()] = ((bth-40)/40)*(.63-(.26))+(.26)
            v[self.human_model.GetJoint('green_kevin/body_quad_left_joint').GetDOFIndex()] = -0.05
            v[self.human_model.GetJoint('green_kevin/body_quad_right_joint').GetDOFIndex()] = -0.05
            v[self.human_model.GetJoint('green_kevin/quad_calf_left_joint').GetDOFIndex()] = 0.05
            v[self.human_model.GetJoint('green_kevin/quad_calf_right_joint').GetDOFIndex()] = 0.05
            v[self.human_model.GetJoint('green_kevin/calf_foot_left_joint').GetDOFIndex()] = ((bth-40)/40)*(0-0)+(0)
            v[self.human_model.GetJoint('green_kevin/calf_foot_right_joint').GetDOFIndex()] = ((bth-40)/40)*(0-0)+(0)
            v[self.human_model.GetJoint('green_kevin/body_arm_left_joint').GetDOFIndex()] = ((bth-40)/40)*(-0.1-(-0.15))+(-0.15)
            v[self.human_model.GetJoint('green_kevin/body_arm_right_joint').GetDOFIndex()] = ((bth-40)/40)*(-0.1-(-0.15))+(-0.15)
            v[self.human_model.GetJoint('green_kevin/arm_forearm_left_joint').GetDOFIndex()] = ((bth-40)/40)*(1.02-0.86)+.86
            v[self.human_model.GetJoint('green_kevin/arm_forearm_right_joint').GetDOFIndex()] = ((bth-40)/40)*(1.02-0.86)+.86
            v[self.human_model.GetJoint('green_kevin/forearm_hand_left_joint').GetDOFIndex()] = ((bth-40)/40)*(.35-0)+0
            v[self.human_model.GetJoint('green_kevin/forearm_hand_right_joint').GetDOFIndex()] = ((bth-40)/40)*(.35-0)+0
        else:
            print 'Error: Head height out of range (should be 0 - 80 degrees)'

        self.human_model.SetActiveDOFValues(v, 2)
        # self.env.UpdatePublishedBodies()

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
                if 'right' in self.arm:
                    sign_flip = -1.
                v = self.robot.GetActiveDOFValues()
                v[self.robot.GetJoint(self.opposite_arm[0]+'_shoulder_pan_joint').GetDOFIndex()] = -sign_flip*3.14/2
                v[self.robot.GetJoint(self.opposite_arm[0]+'_shoulder_lift_joint').GetDOFIndex()] = -0.52
                v[self.robot.GetJoint(self.opposite_arm[0]+'_upper_arm_roll_joint').GetDOFIndex()] = 0.
                v[self.robot.GetJoint(self.opposite_arm[0]+'_elbow_flex_joint').GetDOFIndex()] = -3.14*2/3
                v[self.robot.GetJoint(self.opposite_arm[0]+'_forearm_roll_joint').GetDOFIndex()] = 0.
                v[self.robot.GetJoint(self.opposite_arm[0]+'_wrist_flex_joint').GetDOFIndex()] = 0.
                v[self.robot.GetJoint(self.opposite_arm[0]+'_wrist_roll_joint').GetDOFIndex()] = 0.
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
                            if 'right' in self.arm:
                                sign_flip = -1.
                            v = self.robot.GetActiveDOFValues()
                            v[self.robot.GetJoint(self.opposite_arm[0]+'_shoulder_pan_joint').GetDOFIndex()] = -sign_flip*0.023593
                            v[self.robot.GetJoint(self.opposite_arm[0]+'_shoulder_lift_joint').GetDOFIndex()] = 1.1072800
                            v[self.robot.GetJoint(self.opposite_arm[0]+'_upper_arm_roll_joint').GetDOFIndex()] = -sign_flip*1.5566882
                            v[self.robot.GetJoint(self.opposite_arm[0]+'_elbow_flex_joint').GetDOFIndex()] = -2.124408
                            v[self.robot.GetJoint(self.opposite_arm[0]+'_forearm_roll_joint').GetDOFIndex()] = -sign_flip*1.4175
                            v[self.robot.GetJoint(self.opposite_arm[0]+'_wrist_flex_joint').GetDOFIndex()] = -1.8417
                            v[self.robot.GetJoint(self.opposite_arm[0]+'_wrist_roll_joint').GetDOFIndex()] = 0.21436
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

    def show_rviz(self):
        #rospy.init_node(''.join(['base_selection_goal_visualization']))
        sub_pos, sub_ori = Bmat_to_pos_quat(self.originsubject_B_originworld)
        self.publish_sub_marker(sub_pos, sub_ori)

#         if self.model == 'autobed':
#             self.selection_mat = np.zeros(len(self.goals))
#             self.goal_list = np.zeros([len(self.goals),4,4])
#             headmodel = self.autobed.GetLink('head_link')
#             pr2_B_head = np.matrix(headmodel.GetTransform())
#             for i in xrange(len(self.goals)):
#                 self.goal_list[i] = copy.copy(pr2_B_head*np.matrix(self.goals[i,0]))
#                 self.selection_mat[i] = copy.copy(self.goals[i,1])
# #            for target in self.goals:
# #                self.goal_list.append(pr2_B_head*np.matrix(target[0]))
# #                self.selection_mat.append(target[1])
#             self.set_goals()

        self.publish_goal_markers(self.goal_list)
        #for i in xrange(len(self.goal_list)):
        #    g_pos,g_ori = Bmat_to_pos_quat(self.goal_list[i])
        #    self.publish_goal_marker(g_pos, g_ori, ''.join(['goal_',str(i)]))

    # Publishes as a marker array the goal marker locations used by openrave to rviz so we can see how it overlaps with the subject
    def publish_goal_markers(self, goals):
        vis_pub = rospy.Publisher('~goal_markers', MarkerArray, queue_size=1, latch=True)
        goal_markers = MarkerArray()
        for num, goal_marker in enumerate(goals):
            pos, ori = Bmat_to_pos_quat(goal_marker)
            marker = Marker()
            #marker.header.frame_id = "/base_footprint"
            marker.header.frame_id = "/base_link"
            marker.header.stamp = rospy.Time()
            marker.ns = str(num)
            marker.id = 0
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.pose.position.x = pos[0]
            marker.pose.position.y = pos[1]
            marker.pose.position.z = pos[2]
            marker.pose.orientation.x = ori[0]
            marker.pose.orientation.y = ori[1]
            marker.pose.orientation.z = ori[2]
            marker.pose.orientation.w = ori[3]
            marker.scale.x = .05*3
            marker.scale.y = .05*3
            marker.scale.z = .01*3
            marker.color.a = 1.
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            goal_markers.markers.append(marker)
        vis_pub.publish(goal_markers)
        print 'Published a goal marker to rviz'

    # Publishes a goal marker location used by openrave to rviz so we can see how it overlaps with the subject
    def publish_goal_marker(self, pos, ori, name):
        vis_pub = rospy.Publisher(''.join(['~', name]), Marker, queue_size=1, latch=True)
        marker = Marker()
        #marker.header.frame_id = "/base_footprint"
        marker.header.frame_id = "/base_link"
        marker.header.stamp = rospy.Time()
        marker.ns = name
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose.position.x = pos[0]
        marker.pose.position.y = pos[1]
        marker.pose.position.z = pos[2]
        marker.pose.orientation.x = ori[0]
        marker.pose.orientation.y = ori[1]
        marker.pose.orientation.z = ori[2]
        marker.pose.orientation.w = ori[3]
        marker.scale.x = .2
        marker.scale.y = .2
        marker.scale.z = .2
        marker.color.a = 1.
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        vis_pub.publish(marker)
        print 'Published a goal marker to rviz'

    # Publishes the wheelchair model location used by openrave to rviz so we can see how it overlaps with the real wheelchair
    def publish_sub_marker(self, pos, ori):
        marker = Marker()
        #marker.header.frame_id = "/base_footprint"
        marker.header.frame_id = "/base_link"
        marker.header.stamp = rospy.Time()
        marker.id = 0
        marker.type = Marker.MESH_RESOURCE
        marker.action = Marker.ADD
        marker.pose.position.x = pos[0]
        marker.pose.position.y = pos[1]
        marker.pose.position.z = pos[2]
        marker.pose.orientation.x = ori[0]
        marker.pose.orientation.y = ori[1]
        marker.pose.orientation.z = ori[2]
        marker.pose.orientation.w = ori[3]
        marker.color.a = 1.
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        if self.model == 'chair':
            name = 'subject_model'
            marker.mesh_resource = "package://hrl_base_selection/models/wheelchair_and_body_assembly_rviz.STL"
            marker.scale.x = 1.0
            marker.scale.y = 1.0
            marker.scale.z = 1.0
        elif self.model == 'bed':
            name = 'subject_model'
            marker.mesh_resource = "package://hrl_base_selection/models/head_bed.dae"
            marker.scale.x = 1.0
            marker.scale.y = 1.0
            marker.scale.z = 1.0
        elif self.model == 'autobed':
            name = 'subject_model'
            marker.mesh_resource = "package://hrl_base_selection/models/bed_and_body_v3_rviz.dae"
            marker.scale.x = 1.0
            marker.scale.y = 1.0
            marker.scale.z = 1.0
        elif self.model is None:
            print 'Not publishing a marker, no specific model is being used'
        else:
            print 'I got a bad model. What is going on???'
            return None
        vis_pub = rospy.Publisher(''.join(['~', name]), Marker, queue_size=1, latch=True)
        marker.ns = ''.join(['base_service_', name])
        vis_pub.publish(marker)
        print 'Published a model of the subject to rviz'

    def gen_joint_limit_weight(self, q, side):
        # define the total range limit for each joint
        if 'left' in side:
            joint_min = [-40., -30., -44., -133., -400., -130., -400.]
            joint_max = [130., 80., 224., 0., 400., 0., 400.]
        elif 'right' in side:
            # print 'Need to check the joint limits for the right arm'
            joint_min = [-130., -30., -224., -133., -400., -130., -400.]
            joint_max = [40., 80., 44., 0., 400., 0., 400.]
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
            weights[joint] = 1. - m.pow(0.5, np.abs((joint_max[joint] - joint_min[joint])/2 - m.degrees(q[joint]) - joint_min[joint]))
        weights[4] = 1.
        weights[6] = 1.
        return np.diag(weights)

if __name__ == "__main__":
    rospy.init_node('score_generator')
    mytask = 'shoulder'
    mymodel = 'chair'
    #mytask = 'all_goals'
    # start_time = time.time()
    outer_start_time = rospy.Time.now()
    selector = ScoreGeneratorDressingwithPhysx(visualize=True)
    # selector.visualize_many_configurations()
    # selector.output_results_for_use()
    selector.run_interleaving_optimization_outer_level()
    outer_elapsed_time = rospy.Time.now()-outer_start_time
    print 'Done with openrave round. Time elapsed:', outer_elapsed_time.to_sec()
    print 'Everything is complete!'
    rospy.spin()



    #selector.choose_task(mytask)
    # score_sheet = selector.handle_score_generation()

    # print 'Time to load find generate all scores: %fs'%(time.time()-start_time)

    # rospack = rospkg.RosPack()
    # pkg_path = rospack.get_path('hrl_base_selection')
    # save_pickle(score_sheet, ''.join([pkg_path, '/data/', mymodel, '_', mytask, '.pkl']))
    # print 'Time to complete program, saving all data: %fs' % (time.time()-start_time)






