#!/usr/bin/env python

import numpy as np
import math as m
import copy

import time
import roslib
roslib.load_manifest('hrl_base_selection')
roslib.load_manifest('hrl_haptic_mpc')
import rospy, rospkg
import tf
from geometry_msgs.msg import PoseStamped, PoseArray, Pose
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from sklearn.neighbors import KNeighborsClassifier

from sensor_msgs.msg import JointState
from std_msgs.msg import String
import hrl_lib.transforms as tr
import sensor_msgs.point_cloud2 as pc2
from hrl_base_selection.srv import BaseMove, SetBaseModel, RealtimeBaseMove
from visualization_msgs.msg import Marker
from helper_functions import createBMatrix, Bmat_to_pos_quat
from data_reader_task import DataReader_Task
from score_generator_cma import ScoreGenerator
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from itertools import combinations as comb
import tf.transformations as tft
from matplotlib.cbook import flatten
from sensor_msgs.msg import JointState
from hrl_msgs.msg import FloatArrayBare

import pickle
roslib.load_manifest('hrl_lib')
from hrl_lib.util import load_pickle
import joblib
from os.path import expanduser


class BaseSelector(object):
    def __init__(self, transform_listener=None, mode='normal', model='chair', load='shaving'):
        if transform_listener is None:
            self.listener = tf.TransformListener()
        else:
            self.listener = transform_listener
        self.mode = mode
        self.model = model
        self.load = load
        self.score = None
        self.vis_pub = rospy.Publisher("~service_subject_model", Marker, queue_size=1, latch=True)
        self.goal_viz_publisher = rospy.Publisher('base_goal_pose_viz', PoseArray, queue_size=1, latch=True)

        self.bed_state_z = 0.
        self.bed_state_head_theta = 0.
        self.bed_state_leg_theta = 0.

        rospack = rospkg.RosPack()
        self.pkg_path = rospack.get_path('hrl_base_selection')

        self.robot_z = 0
        self.joint_state_sub = rospy.Subscriber('/joint_states', JointState, self.joint_state_cb)
        self.pr2_B_ar = None
        self.pr2_B_model = None
        # Publisher to let me test things with arm_reacher
        #self.wc_position = rospy.Publisher("~pr2_B_wc", PoseStamped, latch=True)

        # Just for testing
        if self.mode == 'test':
            angle = -m.pi/2
            pr2_B_head1  =  np.matrix([[   m.cos(angle),  -m.sin(angle),          0.,        0.],
                                       [   m.sin(angle),   m.cos(angle),          0.,       2.5],
                                       [             0.,             0.,          1.,       1.1546],
                                       [             0.,             0.,          0.,        1.]])
            an = -m.pi/4
            pr2_B_head2 = np.matrix([[  m.cos(an),   0.,  m.sin(an),       0.],
                                     [         0.,   1.,         0.,       0.],
                                     [ -m.sin(an),   0.,  m.cos(an),       0.],
                                     [         0.,   0.,         0.,       1.]])
            self.pr2_B_head = pr2_B_head1*pr2_B_head2

            self.pr2_B_ar = np.matrix([[   m.cos(angle),  -m.sin(angle),          0.,       1.],
                                       [   m.sin(angle),   m.cos(angle),          0.,       0.],
                                       [             0.,             0.,          1.,       .5],
                                       [             0.,             0.,          0.,       1.]])

        # When in sim mode, the ar tag is at the robot's base
        if self.mode == 'sim':
            self.pr2_B_ar = np.eye(4)

        # Here is where the data is loaded.
        start_time = time.time()
        # ros_start_time = rospy.Time.now()
        print 'Loading data, please wait.'

        # First initializing the data as None
        self.scores_dict = {}
        self.scores_dict['chair', 'shaving'] = None
        self.scores_dict['autobed', 'shaving'] = None
        self.scores_dict['chair', 'feeding'] = None
        self.scores_dict['autobed', 'feeding'] = None
        self.scores_dict['chair', 'brushing'] = None
        self.scores_dict['autobed', 'bathing'] = None
        self.scores_dict['autobed', 'scratching_chest'] = None
        self.scores_dict['autobed', 'scratching_thigh_left'] = None
        self.scores_dict['autobed', 'scratching_thigh_right'] = None
        self.scores_dict['autobed', 'scratching_knee_left'] = None
        self.scores_dict['autobed', 'wiping_mouth'] = None
        self.scores_dict['autobed', 'scratching_forearm_left'] = None
        self.scores_dict['autobed', 'scratching_forearm_right'] = None
        self.scores_dict['autobed', 'scratching_upper_arm_left'] = None
        self.scores_dict['autobed', 'scratching_upper_arm_right'] = None

        # Now load the desired files
        if load == 'all':
            if model == 'chair':
                self.scores_dict[model, 'shaving'] = self.load_task('shaving', model)
                self.scores_dict[model, 'feeding'] = self.load_task('feeding', model)
                self.scores_dict[model, 'brushing'] = self.load_task('brushing', model)
            elif model == 'autobed':
                self.scores_dict['autobed', 'shaving'] = self.load_task('shaving', model)
                self.scores_dict['autobed', 'feeding'] = self.load_task('feeding', model)
                self.scores_dict['autobed', 'bathing'] = self.load_task('bathing', model)
                self.scores_dict['autobed', 'scratching_chest'] = self.load_task('scratching_chest', model)
                self.scores_dict['autobed', 'scratching_knee_left'] = self.load_task('scratching_knee_left', model)
                self.scores_dict['autobed', 'scratching_knee_right'] = self.load_task('scratching_knee_right', model)
                self.scores_dict['autobed', 'scratching_thigh_left'] = self.load_task('scratching_thigh_left', model)
                self.scores_dict['autobed', 'scratching_thigh_right'] = self.load_task('scratching_thigh_right', model)
                self.scores_dict['autobed', 'scratching_forearm_left'] = self.load_task('scratching_forearm_left', model)
                self.scores_dict['autobed', 'scratching_forearm_right'] = self.load_task('scratching_forearm_right', model)
                self.scores_dict['autobed', 'scratching_upper_arm_left'] = self.load_task('scratching_upper_arm_left', model)
                self.scores_dict['autobed', 'scratching_upper_arm_right'] = self.load_task('scratching_upper_arm_right', model)
        elif load == 'paper':
            if model == 'autobed':
                self.scores_dict['autobed', 'scratching_knee_left'] = self.load_task('scratching_knee_left', model)
                self.scores_dict['autobed', 'wiping_mouth'] = self.load_task('wiping_mouth', model)
            else:
                print 'Paper work is only with Autobed. Error!'
                return
        elif load == 'henry':
            model = 'chair'
            #self.scores_dict[model, 'shaving'] = self.load_task('shaving', model)
            # self.scores_dict[model, 'wiping_mouth'] = self.load_task('wiping_mouth', model)
            # self.scores_dict[model, 'scratching_knee_left'] = self.load_task('scratching_knee_left', model)
            # self.scores_dict[model, 'scratching_forehead'] = self.load_task('scratching_knee_left', model)
            # self.scores_dict[model, 'shaving'] = self.load_task('scratching_knee_left', model)
            # self.scores_dict[model, 'brushing_teeth'] = self.load_task('scratching_knee_left', model)
            #self.scores_dict[model, 'scratching_knee_right'] = self.load_task('scratching_knee_right', model)
            #self.scores_dict[model, 'scratching_upper_arm_left'] = self.load_task('scratching_upper_arm_left', model)
            #self.scores_dict[model, 'scratching_upper_arm_right'] = self.load_task('scratching_upper_arm_right', model)
            #self.scores_dict[model, 'scratching_forearm_left'] = self.load_task('scratching_forearm_left', model)
            #self.scores_dict[model, 'scratching_forearm_right'] = self.load_task('scratching_forearm_right', model)
            # self.scores_dict[model, 'feeding'] = self.load_task('feeding', model)
            # self.scores_dict[model, 'brushing'] = self.load_task('brushing', model)
            model = 'autobed'
            self.scores_dict[model, 'wiping_mouth'] = self.load_task('wiping_mouth', model)
            self.scores_dict[model, 'scratching_knee_left'] = self.load_task('scratching_knee_left', model)
            # self.scores_dict[model, 'scratching_forehead'] = self.load_task('scratching_knee_left', model)
            # self.scores_dict[model, 'brushing_teeth'] = self.load_task('scratching_knee_left', model)
            # self.scores_dict[model, 'shaving'] = self.load_task('shaving', model)
            # self.scores_dict[model, 'feeding'] = self.load_task('feeding', model)
            # self.scores_dict[model, 'bathing'] = self.load_task('bathing', model)
            #self.scores_dict[model, 'wiping_mouth'] = self.load_task('wiping_mouth', model)
            # self.scores_dict[model, 'scratching_chest'] = self.load_task('scratching_chest', model)
            #self.scores_dict[model, 'scratching_knee_left'] = self.load_task('scratching_knee_left', model)
            #self.scores_dict[model, 'scratching_knee_right'] = self.load_task('scratching_knee_right', model)
            # self.scores_dict[model, 'scratching_thigh_left'] = self.load_task('scratching_thigh_left', model)
            # self.scores_dict[model, 'scratching_thigh_right'] = self.load_task('scratching_thigh_right', model)
            #self.scores_dict[model, 'scratching_forearm_left'] = self.load_task('scratching_forearm_left', model)
            #self.scores_dict[model, 'scratching_forearm_right'] = self.load_task('scratching_forearm_right', model)
            #self.scores_dict[model, 'scratching_upper_arm_left'] = self.load_task('scratching_upper_arm_left', model)
            #self.scores_dict[model, 'scratching_upper_arm_right'] = self.load_task('scratching_upper_arm_right', model)
            self.real_time_score_generator = ScoreGenerator(reference_names=[None], model=None, visualize=False)
            self.model_read_service = rospy.Service('set_environment_model', SetBaseModel, self.handle_read_in_environment_model)
            self.real_time_base_selection_service = rospy.Service('realtime_select_base_position', RealtimeBaseMove, self.realtime_base_selection)
            self.real_time_goal_pose_pub = rospy.Publisher('~rtbs_ee_goal_pose', PoseStamped, queue_size=1, latch=True)
        else:
            self.scores_dict[model, load] = self.load_task(load, model)

        # self.chair_scores = self.load_task('yogurt', 'chair')
        # self.autobed_scores = self.load_task('feeding_quick', 'autobed', 0)
        # self.shaving_scores = self.load_task('shaving', 'chair')
        print 'Time to load all requested data: %fs' % (time.time()-start_time)
        # print 'ROS time to load all requested data: ', (rospy.Time.now() - ros_start_time).to_sec()

        # Initialize the services
        self.base_selection_service = rospy.Service('select_base_position', BaseMove, self.handle_select_base)

        # Subscriber to update robot joint state
        #self.joint_state_sub = rospy.Subscriber('/joint_states', JointState, self.joint_state_cb)

        print "Ready to select base."

    # This gets the joint states of the entire robot and saves only the robot's z-axis state.
    def joint_state_cb(self, msg):
        for num, name in enumerate(msg.name):
            if name == 'torso_lift_joint':
                self.robot_z = msg.position[num]

    # Publishes the wheelchair model location used by openrave to rviz so we can see how it overlaps with the real
    # wheelchair. This is for visualization, serves no vital purpose.
    def publish_sub_marker(self, pos, ori):
        marker = Marker()
        #marker.header.frame_id = "/base_footprint"
        marker.header.frame_id = "/base_footprint"
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
        name = 'BSS_subject_model'
        if self.model == 'chair':
            marker.mesh_resource = "package://hrl_base_selection/models/wheelchair_and_body_assembly_rviz.STL"
            marker.scale.x = 1.0
            marker.scale.y = 1.0
            marker.scale.z = 1.0
        elif self.model == 'bed':
            marker.mesh_resource = "package://hrl_base_selection/models/head_bed.dae"
            marker.scale.x = 1.0
            marker.scale.y = 1.0
            marker.scale.z = 1.0
        elif self.model == 'autobed':
            marker.mesh_resource = "package://hrl_base_selection/models/bed_and_body_v3_rviz.dae"
            marker.scale.x = 1.0
            marker.scale.y = 1.0
            marker.scale.z = 1.0
        elif self.model is None:
            print 'Not publishing a marker, no specific model is being used'
        else:
            print 'I got a bad model. What is going on???'
            return None
        vis_pub = rospy.Publisher(''.join(['~', name]), Marker, queue_size=100, latch=True)
        marker.ns = ''.join(['base_service_', name])
        vis_pub.publish(marker)
        print 'Published a model of the subject to rviz'

    # When we are using the autobed, we probably need to know the state of the autobed. This records the current
    # state of the autobed.
    def bed_state_cb(self, data):
        self.bed_state_z = data.data[1]
        self.bed_state_head_theta = data.data[0]
        self.bed_state_leg_theta = data.data[2]

    # Used only in sim mode. This keeps track of the pose of the head when it is just being published, instead of
    # getting it from tf.
    def head_pose_cb(self, data):
        trans = [data.pose.position.x,
                 data.pose.position.y,
                 data.pose.position.z]
        rot = [data.pose.orientation.x,
               data.pose.orientation.y,
               data.pose.orientation.z,
               data.pose.orientation.w]
        self.pr2_B_head = createBMatrix(trans, rot)
        if self.model == 'chair':
            ar_trans = [data.pose.position.x,
                        data.pose.position.y,
                        0.]
            self.pr2_B_ar = createBMatrix(ar_trans, rot)

    # Used only in sim mode. This keeps track of the location of the bed when it is just being published, instead of
    # getting it from tf.
    def bed_pose_cb(self, data):
        trans = [data.pose.position.x,
                 data.pose.position.y,
                 data.pose.position.z]
        rot = [data.pose.orientation.x,
               data.pose.orientation.y,
               data.pose.orientation.z,
               data.pose.orientation.w]
        self.pr2_B_ar = createBMatrix(trans, rot)

    def handle_read_in_environment_model(self, req):
        service_start_time = time.time()
        print 'Reading in environment model!'
        if req.cloud.data == []:
            print 'I got an empty cloud!'
            return False
        success = self.real_time_score_generator.initialize_environment_model(req.cloud)
        print 'Time to complete service call (reading in environment model to openrave): %fs' % (time.time()-service_start_time)
        return success

    def realtime_base_selection(self, req):
        service_initial_time = time.time()
        start_time = time.time()
        print 'The real-time configuration selection service has been called!'

        # This is real-time base selection mode. Used to find a base configuration for simple tasks with respect to a
        # reference AR tag.

        if 'r' in req.ee_frame[0:2]:
            arm = 'rightarm'
        elif 'l' in req.ee_frame[0:2]:
            arm = 'leftarm'
        else:
            'ERROR'
            'I do not know whether to use the left or right arm'
            return None

        self.origin_B_pr2 = np.matrix(np.eye(4))
        self.pr2_B_ar = np.matrix(np.eye(4))
        pos = req.pose_target.pose.position
        rot = req.pose_target.pose.orientation
        target_reference_frame = req.pose_target.header.frame_id
        now = rospy.Time.now()
        self.listener.waitForTransform('/base_footprint', target_reference_frame, now, rospy.Duration(15))
        (trans, ori) = self.listener.lookupTransform('/base_footprint', target_reference_frame, now)

        # if not target_reference_frame == 'base_footprint':
        #     print 'ERROR!!'
        #     print 'Base selection currently requires goals in the base footprint frame when in real-time mode.'
        #     print 'But it was given in following frame: ', target_reference_frame
        #     return None

        reference_B_goal_pose = createBMatrix([pos.x, pos.y, pos.z], [rot.x, rot.y, rot.z, rot.w])
        base_footprint_B_reference = createBMatrix(trans, ori)
        base_footprint_B_goal_pose = base_footprint_B_reference*reference_B_goal_pose
        print 'The goal end effector pose in the base footprint frame is:'
        print base_footprint_B_goal_pose
        pos, ori = Bmat_to_pos_quat(base_footprint_B_goal_pose)
        ref_pose = PoseStamped()
        ref_pose.header.frame_id = "/base_footprint"
        ref_pose.header.stamp = rospy.Time.now()
        ref_pose.pose.position.x = pos[0]
        ref_pose.pose.position.y = pos[1]
        ref_pose.pose.position.z = pos[2]
        ref_pose.pose.orientation.x = ori[0]
        ref_pose.pose.orientation.y = ori[1]
        ref_pose.pose.orientation.z = ori[2]
        ref_pose.pose.orientation.w = ori[3]
        self.real_time_goal_pose_pub.publish(ref_pose)

        base_selection_goal = []
        base_selection_goal.append([base_footprint_B_goal_pose, 1, 0])
        base_selection_goal = np.array(base_selection_goal)
        # self.real_time_score_generator.generate_environment_model()
        self.real_time_score_generator.set_arm(arm)
        self.real_time_score_generator.receive_new_goals(base_selection_goal, reference_options=['base_link'])
        config, score = self.real_time_score_generator.real_time_scoring()
        self.score = [[[config[0]], [config[1]], [config[2]], [config[3]], [0], [0]], score]
        print 'Time for TOC service to run start to finish: %fs' % (time.time()-service_initial_time)
        if score >=10.:
            print 'Could not find a base configuration from which the PR2 could reach the goal!'
            print 'Sorry!'
            return [], [], []
        else:
            return self.handle_returning_base_goals()


    # The service call function that determines a good base location to be able to reach the goal location.
    # Takes as input the service's inputs. Outputs lists with initial configurations.
    def handle_select_base(self, req):
        service_initial_time = time.time()
        start_time = time.time()
        print 'The initial configuration selection service has been called!'
        model = req.model
        task = req.task
        self.task = task

        # Check if we have previously loaded this task/model (assuming the data file exists).
        if self.load == 'henry':
            self.model = model
        else:
            if self.model != model:
                print 'The model in the service request differs from what was given to the service on initialization. As' \
                      'a result, data for a user in that location (autobed/chair) has not been loaded!'
        if self.load != 'all':
            if self.load != task and self.load != 'paper' and self.load != 'henry':
                print 'The task asked of the service request differs from what was given to the service on ' \
                      'initialization. As a result, data for that task has not been loaded!'
                # print 'As a result, only the runtime version of base selection is active.'

        # Subscribe to the autobed state if we are using autobed
        if model == 'autobed':
            self.autobed_sub = rospy.Subscriber('/abdout0', FloatArrayBare, self.bed_state_cb)

        # In normal mode, gets locations of things from tf. Intended to use head registration for head pose, AR tag
        # detection, and servoing.
        if self.task == 'shaving_test':
            self.mode = 'test'
        elif self.mode == 'normal':
            try:
                if model == 'chair':
                    now = rospy.Time.now()
                    self.listener.waitForTransform('/base_footprint', '/ar_marker_13', now, rospy.Duration(15))
                    (trans, rot) = self.listener.lookupTransform('/base_footprint', '/ar_marker_13', now)
                    self.pr2_B_ar = createBMatrix(trans, rot)
                    self.listener.waitForTransform('/base_footprint', '/wheelchair/base_link', now, rospy.Duration(15))
                    (trans, rot) = self.listener.lookupTransform('/base_footprint', '/wheelchair/base_link', now)
                    self.pr2_B_model = createBMatrix(trans, rot)
                elif model == 'autobed':
                    now = rospy.Time.now()
                    self.listener.waitForTransform('/base_footprint', '/user_head_link', now, rospy.Duration(15))
                    (trans, rot) = self.listener.lookupTransform('/base_footprint', '/user_head_link', now)
                    self.pr2_B_head = createBMatrix(trans, rot)
                    now = rospy.Time.now()
                    self.listener.waitForTransform('/base_footprint', '/ar_marker_4', now, rospy.Duration(15))
                    (trans, rot) = self.listener.lookupTransform('/base_footprint', '/ar_marker_4', now)
                    self.pr2_B_ar = createBMatrix(trans, rot)
                    now = rospy.Time.now()
                    self.listener.waitForTransform('/base_footprint', '/autobed/base_link', now, rospy.Duration(15))
                    (trans, rot) = self.listener.lookupTransform('/base_footprint', '/autobed/base_link', now)
                    self.pr2_B_model = createBMatrix(trans, rot)
                    # print 'The transform from PR2 to autobed is:'
                    # print self.pr2_B_model

                    # Here I do some manual conversion to covert between the coordinate frame of the bed, which should
                    # be located in the center of the headboard of the bed on the floor, to the AR tag's coordinate
                    # frame. To make the manual transformation calculation easier, I split it into 3 homogeneous
                    # transforms, one translation, one rotation about Z, one rotation about x. This should be adjusted
                    # depending on the actual location of the AR tag.
                    # ar_trans_B = np.eye(4)
                    # -.445 if right side of body. .445 if left side.
                    # This is the translational transform from bed origin to the ar tag tf.
                    # ar_trans_B[0:3,3] = np.array([0.625, -.445, .275+(self.bed_state_z-9)/100])
                    # ar_trans_B[0:3,3] = np.array([-.04, 0., .74])
                    # ar_rotz_B = np.eye(4)
                    # If left side of body should be np.array([[-1,0],[0,-1]])
                    # If right side of body should be np.array([[1,0],[0,1]])
                    # ar_rotz_B [0:2,0:2] = np.array([[-1, 0],[0, -1]])
                    # ar_rotz_B
                    # ar_rotx_B = np.eye(4)
                    # ar_rotx_B[1:3,1:3] = np.array([[0,1],[-1,0]])
                    # self.model_B_ar = np.matrix(ar_trans_B)*np.matrix(ar_rotz_B)*np.matrix(ar_rotx_B)
                    # now = rospy.Time.now()
                    # self.listener.waitForTransform('/ar_marker', '/bed_frame', now, rospy.Duration(3))
                    # (trans, rot) = self.listener.lookupTransform('/ar_marker', '/bed_frame', now)
                    # self.ar_B_model = createBMatrix(trans, rot)
                # Probably for the best to not try to do things from too far away. Also, if the AR tag is more than 4m
                # away, it is likely that an error is occurring with its detection.
                if np.linalg.norm(trans) > 5.:
                    rospy.loginfo('AR tag is too far away. Use the \'Testing\' button to move PR2 to 1 meter from AR '
                                  'tag. Or just move it closer via other means. Alternatively, the PR2 may have lost '
                                  'sight of the AR tag or it is having silly issues recognizing it. ')
                    return None, None
            except Exception as e:
                rospy.loginfo("TF Exception. Could not get the AR_tag location, bed location, or "
                              "head location:\r\n%s" % e)
                print 'Error!! In base selection'
                return None, None
        # Demo mode is to use motion capture to get locations. In this case, some things simplify out.
        elif self.mode == 'demo':
            try:
                now = rospy.Time.now()
                self.listener.waitForTransform('/base_footprint', '/head_frame', now, rospy.Duration(15))
                (trans, rot) = self.listener.lookupTransform('/base_footprint', '/head_frame', now)
                self.pr2_B_head = createBMatrix(trans, rot)
                if model == 'chair':
                    now = rospy.Time.now()
                    self.listener.waitForTransform('/base_footprint', '/ar_marker', now, rospy.Duration(15))
                    (trans, rot) = self.listener.lookupTransform('/base_footprint', '/ar_marker', now)
                    self.pr2_B_ar = createBMatrix(trans, rot)
                elif model == 'autobed':
                    now = rospy.Time.now()
                    self.listener.waitForTransform('/base_footprint', '/reference', now, rospy.Duration(15))
                    (trans, rot) = self.listener.lookupTransform('/base_footprint', '/reference', now)
                    self.pr2_B_ar = createBMatrix(trans, rot)
                    ar_trans_B_model = np.eye(4)
                    # -.445 if right side of body. .445 if left side.
                    # This is the translational transform from reference markers to the bed origin.
                    # ar_trans_B[0:3,3] = np.array([0.625, -.445, .275+(self.bed_state_z-9)/100])
                    # ar_trans_B_model[0:3,3] = np.array([.06, 0., -.74])
                    # ar_rotz_B = np.eye(4)
                    # If left side of body should be np.array([[-1,0],[0,-1]])
                    # If right side of body should be np.array([[1,0],[0,1]])
                    # ar_rotz_B [0:2,0:2] = np.array([[-1, 0],[0, -1]])
                    # ar_rotz_B
                    # ar_rotx_B = np.eye(4)
                    # ar_rotx_B[1:3,1:3] = np.array([[0,1],[-1,0]])
                    # self.model_B_ar = np.matrix(ar_trans_B_model).I  # *np.matrix(ar_rotz_B)*np.matrix(ar_rotx_B)
                    self.model_B_ar = np.matrix(np.eye(4))
                    # now = rospy.Time.now()
                    # self.listener.waitForTransform('/ar_marker', '/bed_frame', now, rospy.Duration(3))
                    # (trans, rot) = self.listener.lookupTransform('/ar_marker', '/bed_frame', now)
                    # self.ar_B_model = createBMatrix(trans, rot)
                if np.linalg.norm(trans) > 4:
                    rospy.loginfo('AR tag is too far away. Use the \'Testing\' button to move PR2 to 1 meter from AR '
                                  'tag. Or just move it closer via other means. Alternatively, the PR2 may have lost '
                                  'sight of the AR tag or it is having silly issues recognizing it. ')
                    return None, None
            except Exception as e:
                rospy.loginfo("TF Exception. Could not get the AR_tag location, bed location, or "
                              "head location:\r\n%s" % e)
                return None, None
        # In sim mode, we expect that the head and bed pose (if using autobed) are being published.
        elif self.mode == 'sim':
            self.head_pose_sub = rospy.Subscriber('/haptic_mpc/head_pose', PoseStamped, self.head_pose_cb)
            if self.model == 'autobed':
                self.bed_pose_sub = rospy.Subscriber('/haptic_mpc/bed_pose', PoseStamped, self.bed_pose_cb)
            self.model_B_ar = np.eye(4)

        #print 'The homogeneous transform from PR2 base link to head: \n',# self.pr2_B_head
        if self.model == 'autobed':
            # I now project the head pose onto the ground plane to mitigate potential problems with poorly registered head
            # pose.
            z_origin = np.array([0, 0, 1])
            x_head = np.array([self.pr2_B_head[0, 0], self.pr2_B_head[1, 0], self.pr2_B_head[2, 0]])
            y_head_project = np.cross(z_origin, x_head)
            y_head_project = y_head_project/np.linalg.norm(y_head_project)
            x_head_project = np.cross(y_head_project, z_origin)
            x_head_project = x_head_project/np.linalg.norm(x_head_project)
            self.pr2_B_head_project = np.eye(4)
            for i in xrange(3):
                self.pr2_B_head_project[i, 0] = x_head_project[i]
                self.pr2_B_head_project[i, 1] = y_head_project[i]
                self.pr2_B_head_project[i, 3] = self.pr2_B_head[i, 3]
            self.pr2_B_headfloor = copy.copy(np.matrix(self.pr2_B_head_project))
            self.pr2_B_headfloor[2, 3] = 0.
        # print 'The homogeneous transform from PR2 base link to the head location projected onto the ground plane: \n', \
        #     self.pr2_B_headfloor

        headx = 0
        heady = 0

        # Sets the location of the robot with respect to the person based using a few homogeneous transforms.
        if model == 'chair':
            self.origin_B_pr2 = copy.copy(self.pr2_B_model.I)

        # Regular bed is now deprecated. Do not use this unless you fix it first.
        elif model =='bed':
            an = -m.pi/2
            self.headfloor_B_head = np.matrix([[  m.cos(an),   0.,  m.sin(an),       0.], #.45 #.438
                                               [         0.,   1.,         0.,       0.], #0.34 #.42
                                               [ -m.sin(an),   0.,  m.cos(an),   1.1546],
                                               [         0.,   0.,         0.,       1.]])
            an2 = 0
            originsubject_B_headfloor = np.matrix([[ m.cos(an2),  0., m.sin(an2),  .2954], #.45 #.438
                                                   [         0.,  1.,         0.,     0.], #0.34 #.42
                                                   [-m.sin(an2),  0., m.cos(an2),     0.],
                                                   [         0.,  0.,         0.,     1.]])
            self.origin_B_pr2 = self.headfloor_B_head * self.pr2_B_head.I
        # Slightly more complicated for autobed because the person can move around on the bed.
        elif model == 'autobed':

            self.model_B_pr2 = self.pr2_B_model.I
            # self.model_B_pr2 = self.model_B_ar * self.pr2_B_ar.I
            self.origin_B_pr2 = copy.copy(self.model_B_pr2)
            model_B_head = self.model_B_pr2 * self.pr2_B_headfloor

            # Use the heady of the nearest neighbor from the data.
            heady_possibilities = (np.arange(11)-5)*30
            heady_neigh = KNeighborsClassifier(n_neighbors=1)
            heady_neigh.fit(np.reshape(heady_possibilities,[len(heady_possibilities),1]), heady_possibilities)
            heady = heady_neigh.predict(int(model_B_head[1, 3]*1000))[0]*.001



            headx = 0.
            #heady = 0.
            print 'The nearest neighbor to the current head_y position is:', heady

            ## This next bit selects what entry in the dictionary of scores to use based on the location of the head
            # with respect to the bed model. Currently it just selects the dictionary entry with the closest relative
            # head location. Ultimately it should use a gaussian to get scores from the dictionary based on the actual
            # head location.
            #if model_B_head[0, 3] > -.025 and model_B_head[0, 3] < .025:
            #    headx = 0.
            #elif model_B_head[0, 3] >= .025 and model_B_head[0, 3] < .75:
            #    headx = 0.
            #elif model_B_head[0, 3] <= -.025 and model_B_head[0, 3] > -.75:
            #    headx = 0.
            #elif model_B_head[0, 3] >= .075:
            #    headx = 0.
            #elif model_B_head[0, 3] <= -.075:
            #    headx = 0.

            #if model_B_head[1, 3] > -.025 and model_B_head[1, 3] < .025:
            #    heady = 0
            #elif model_B_head[1, 3] >= .025 and model_B_head[1, 3] < .075:
            #    heady = .05
            #elif model_B_head[1, 3] > -.075 and model_B_head[1, 3] <= -.025:
            #    heady = -.05
            #elif model_B_head[1, 3] >= .075:
            #    heady = .1
            #elif model_B_head[1, 3] <= -.075:
            #    heady = -.1

        # subject_location is the transform from the robot to the origin of the model that was used in creating the
        # databases for base selection
        subject_location = self.origin_B_pr2.I

        print 'Now finished getting poses of the head, etc. Proceeding!'
        print 'Time to receive locations and start things off: %fs' % (time.time()-start_time)

        # Now check that we have the data loaded for the desired task. Load the data if it has not yet been loaded.
        start_time = time.time()
        if self.scores_dict[model, task] is None:
            print 'For whatever reason, the data for this task was not previously loaded. I will now try to load it.'
            all_scores = self.load_task(task, model)
            if all_scores is None:
                print 'Failed to load precomputed reachability data. That is a problem. Abort!'
                return None, None
        else:
            all_scores = self.scores_dict[model, task]
        #scores = all_scores[headx, heady]
        max_num_configs = 1

        head_rest_angle = -10
        allow_bed_movement = 1
        if self.model == 'autobed':
            if head_rest_angle > -1:
                head_rest_possibilities = np.arange(-10, 80.1, 10)
                head_rest_neigh = KNeighborsClassifier(n_neighbors=1)
                head_rest_neigh.fit(np.reshape(head_rest_possibilities,[len(head_rest_possibilities),1]), head_rest_possibilities)
                head_rest_angle = head_rest_neigh.predict(np.degrees(self.bed_state_head_theta))[0]

            self.score = all_scores[model, max_num_configs, head_rest_angle, headx, heady, 1]
        else:
            self.score = all_scores[model, max_num_configs, 0,0,0,0]
        if np.shape(self.score)==(2,) and self.model=='autobed':
            self.score = [[[self.score[0][0]], [self.score[0][1]], [self.score[0][2]], [self.score[0][3]], [self.score[0][4]], [self.score[0][5]]], self.score[1]]
        elif np.shape(self.score)==(2,) and self.model=='chair':
            self.score = [[[self.score[0][0]], [self.score[0][1]], [self.score[0][2]], [self.score[0][3]], [0], [0]], self.score[1]]
        # self.score_length = len(self.score_sheet)
        print 'Best score and configuration is: \n', self.score
        # print 'Number of scores in score sheet: ', self.score_length

        print 'I have finished preparing the data for the task!'
        print 'Time to perform optimization: %fs' % (time.time()-start_time)

        # Published the wheelchair location to create a marker in rviz for visualization to compare where the service believes the wheelchair is to
        # where the person is (seen via kinect).
        pos_goal, ori_goal = Bmat_to_pos_quat(subject_location)
        self.publish_sub_marker(pos_goal, ori_goal)

        # Visualize plot is a function to return a 2-D plot showing the best scores for each robot X-Y base location
        # after the updates to the score from above. Currently deprecated, so don't use it.
        visualize_plot = False
        if visualize_plot:
            self.plot_final_score_sheet()

        # This is the end of this function. The return calls a function that creates an output that is appropriate for
        # whatever the system calling base selection wants.
        print 'Time for service to run start to finish: %fs' % (time.time()-service_initial_time)
        return self.handle_returning_base_goals()

    # Function for handling the output of base selection. If given no input, uses data calculated previously. Can also
    # be given a set of data as input. This is usually done just for testing. Outputs the best location for the pr2
    # base and the best "other" configurations in two separate lists.
    # Format of output is:
    # [x (m), y (m), theta (radians)], [pr2_z_axis (cm), autobed_height (cm), autobed_headrest_angle (radians)]
    # The current output for the robot base location is the transformation from the goal position for the robot base
    # to the AR tag.
    # For a task with a solution of multiple configurations, each configuration will be appended to the previous list.
    # E.g. [x1, y1, th1, x2, y2, th2] where the first three entries correspond to the first configuration.
    def handle_returning_base_goals(self, data=None):
        # If input is received, use it. Otherwise, use data from above.
        if data != None:
            score_sheet = copy.copy(data)
        else:
            score_sheet = copy.copy(self.score)

        # I only want to output the configuration with the best score, so first I grab it from the score sheet.
        best_score_cfg = score_sheet[0]
        best_score_score = score_sheet[1]

        pr2_base_output = []
        configuration_output = []
        distance_output = []

        pose_array = PoseArray()
        pose_array.header.stamp = rospy.Time.now()
        pose_array.header.frame_id = 'odom_combined'


        # Outputs the best location for the pr2
        # base and the best "other" configurations in two separate lists.
        # Format of output is:
        # [x (m), y (m), theta (radians)], [pr2_z_axis (cm), autobed_height (cm), autobed_headrest_angle (radians)]
        # The current output for the robot base location is the transformation from the goal position for the robot base
        # to the AR tag.
        # For a task with a solution of multiple configurations, each configuration will be appended to the previous list.
        # E.g. [x1, y1, th1, x2, y2, th2] where the first three entries correspond to the first configuration.
        for i in xrange(len(best_score_cfg[0])):
            origin_B_goal = np.matrix([[m.cos(best_score_cfg[2][i]), -m.sin(best_score_cfg[2][i]), 0., best_score_cfg[0][i]],
                                       [m.sin(best_score_cfg[2][i]),  m.cos(best_score_cfg[2][i]), 0., best_score_cfg[1][i]],
                                       [0.,                      0.,                           1.,           0.],
                                       [0.,                      0.,                           0.,           1.]])
            print 'model origin to goal:'
            print origin_B_goal
            model_B_goal_trans, model_B_goal_rot = Bmat_to_pos_quat(origin_B_goal)
            #model_B_goal_out = list(flatten([model_B_goal_trans, model_B_goal_rot]))
            #model_B_goal_out_list = [float(i) for i in model_B_goal_out]
            model_B_goal_out_list = [float(model_B_goal_trans[0]), float(model_B_goal_trans[1])]
            rospy.set_param('model_B_goal', model_B_goal_out_list)
            pr2_B_goal = self.origin_B_pr2.I * origin_B_goal
            now = rospy.Time.now()
            self.listener.waitForTransform('/odom_combined', '/base_footprint', now, rospy.Duration(15))
            (trans, rot) = self.listener.lookupTransform('/odom_combined', '/base_footprint', now)
            world_B_pr2 = createBMatrix(trans, rot)

            pr2_B_goal_pose = Pose()
            # pr2_B_goal_pose.header.stamp = rospy.Time.now()
            # pr2_B_goal_pose.header.frame_id = 'odom_combined'
            trans_out, rot_out = Bmat_to_pos_quat(world_B_pr2*pr2_B_goal)
            pr2_B_goal_pose.position.x = trans_out[0]
            pr2_B_goal_pose.position.y = trans_out[1]
            pr2_B_goal_pose.position.z = trans_out[2]
            pr2_B_goal_pose.orientation.x = rot_out[0]
            pr2_B_goal_pose.orientation.y = rot_out[1]
            pr2_B_goal_pose.orientation.z = rot_out[2]
            pr2_B_goal_pose.orientation.w = rot_out[3]
            pose_array.poses.append(pr2_B_goal_pose)
            #self.goal_viz_publisher.publish(pr2_B_goal_pose)
            goal_B_ar = pr2_B_goal.I*self.pr2_B_ar
            print 'pr2_B_goal:'
            print pr2_B_goal
            print 'pr2_B_ar:'
            print self.pr2_B_ar
            print 'goal_B_ar:'
            print goal_B_ar
            print 'ar_B_goal:'
            print goal_B_ar.I
            distance_to_goal = np.linalg.norm([pr2_B_goal[0, 3],pr2_B_goal[1, 3]])
            distance_output.append(distance_to_goal)
            # goal_B_ar = pr2_B_goal.I * self.pr2_B_ar
            # pos_goal, ori_goal = Bmat_to_pos_quat(goal_B_ar)
            pos_goal, ori_goal = Bmat_to_pos_quat(goal_B_ar)
            # if pr2_B_goal[0,1] <= 0:
            #     pr2_base_output.append([pr2_B_goal[0,3], pr2_B_goal[1,3], m.acos(pr2_B_goal[0, 0])])
            # else:
            #     pr2_base_output.append([pr2_B_goal[0,3], pr2_B_goal[1,3], -m.acos(pr2_B_goal[0, 0])])
            pr2_base_output.append([pos_goal, ori_goal])
            configuration_output.append([best_score_cfg[3][i], 100*best_score_cfg[4][i], np.degrees(best_score_cfg[5][i])])
        self.goal_viz_publisher.publish(pose_array)
        print 'Base selection service is done and has completed preparing its result.'
        print 'Base selection output:'
        print list(flatten(pr2_base_output))
        print list(flatten(configuration_output))
        print distance_output
        return list(flatten(pr2_base_output)), list(flatten(configuration_output)), distance_output

    # This function is deprecated. Do not use it for now. It is to plot in 2D the score sheet after it gets updated
    # with the current state of the environment
    def plot_final_score_sheet(self):
        visualize_plot = True
        if visualize_plot:
            # Plot the score as a scatterplot heat map
            #print 'score_sheet:',score_sheet
            score2d_temp = []
            #print t
            for i in np.arange(-1.5,1.55,.05):
                for j in np.arange(-1.5,1.55,.05):
                    temp = []
                    for item in score_sheet:
                    #print 'i is:',i
                    #print 'j is:',j
                        if item[0]==i and item[1]==j:
                            temp.append(item[3])
                    if temp != []:
                        score2d_temp.append([i,j,np.max(temp)])

            seen_items = []
            score2d = []
            for item in score2d_temp:

                #print 'seen_items is: ',seen_items
                #print 'item is: ',item
                #print (any((item == x) for x in seen_items))
                if not (any((item == x) for x in seen_items)):
                #if item not in seen_items:
                    #print 'Just added the item to score2d'
                    score2d.append(item)
                    seen_items.append(item)
            score2d = np.array(score2d)
            #print 'score2d with no repetitions',score2d

            fig, ax = plt.subplots()

            X  = score2d[:,0]
            Y  = score2d[:,1]
            #Th = score_sheet[:,2]
            c  = score2d[:,2]
            #surf = ax.scatter(delta1[:-1], delta1[1:], c=close, s=volume, alpha=0.5)
            surf = ax.scatter(X, Y, s=60,c=c,alpha=1)
            #surf = ax.scatter(X, Y,s=40, c=c,alpha=.6)
            ax.set_xlabel('X Axis')
            ax.set_ylabel('Y Axis')
            #ax.set_zlabel('Theta Axis')

            fig.colorbar(surf, shrink=0.5, aspect=5)


            verts_wc = [(-.438, -.32885), # left, bottom
                        (-.438, .32885), # left, top
                        (.6397, .32885), # right, top
                        (.6397, -.32885), # right, bottom
                        (0., 0.), # ignored
                        ]

            verts_pr2 = [(-1.5,  -1.5), # left, bottom
                         ( -1.5, -.835), # left, top
                         (-.835, -.835), # right, top
                         (-.835,  -1.5), # right, bottom
                         (   0.,    0.), # ignored
                         ]

            codes = [Path.MOVETO,
                     Path.LINETO,
                     Path.LINETO,
                     Path.LINETO,
                     Path.CLOSEPOLY,
                     ]

            path_wc = Path(verts_wc, codes)
            path_pr2 = Path(verts_pr2, codes)

            patch_wc = patches.PathPatch(path_wc, facecolor='orange', lw=2)
            patch_pr2 = patches.PathPatch(path_pr2, facecolor='orange', lw=2)

            ax.add_patch(patch_wc)
            ax.add_patch(patch_pr2)
            ax.set_xlim(-2,2)
            ax.set_ylim(-2,2)
            plt.show()


                            #self.robot.SetDOFValues(sol,self.manip.GetArmIndices()) # set the current solution
                            #Tee = self.manip.GetEndEffectorTransform()
                            #self.env.UpdatePublishedBodies() # allow viewer to update new robot
#                            traj = None
#                            try:
#                                #res = self.manipprob.MoveToHandPosition(matrices=[np.array(pr2_B_goal)],seedik=10) # call motion planner with goal joint angles
#                                traj = self.manipprob.MoveManipulator(goal=sol, outputtrajobj=True)
#                                print 'Got a trajectory! \n'#,traj
#                            except:
#                                #print 'traj = \n',traj
#                                traj = None
#                                print 'traj failed \n'
#                                pass
#                            #traj =1 #This gets rid of traj
#                            if traj is not None:
        #else:
            #print 'I found a bad goal location. Trying again!'
                            #rospy.sleep(.1)
        #print 'I found nothing! My given inputs were: \n', req.task, req.head
        return None

    # Function to load task data. Currently uses joblib. Returns the data in the file if successful. Returns None if
    # unsuccessful. Takes as input strings for the task (shaving, feeding, etc) and for the model (chair or autobed).
    # Takes as input subj, which is the subject number. This is typically 0. Could use other numbers to differentiate
    # between users or user preference.
    # Set to load from svn now, where I have put the data files.
    def load_task(self, task, model):
        home = expanduser("~")
        file_name = self.pkg_path + '/data/' + task + '_' + model + '_cma_score_data.pkl'
#        print file_name
        return load_pickle(file_name)
#        if 'wiping' in task:
#            file_name = ''.join([home, '/svn/robot1_data/usr/ari/data/base_selection/', task, '/', model, '/', task, '_', model, '_cma_score_data.pkl'])
#            return load_pickle(file_name)

#        elif 'scratching' not in task:
            # file_name = ''.join([self.pkg_path, '/data/', task, '_', model, '_subj_', str(subj), '_score_data'])
#            file_name = ''.join([home, '/svn/robot1_data/usr/ari/data/base_selection/', task, '/', model, '/', task, '_', model, '_cma_score_data'])
#        else:
#            task_name = 'scratching'
#            task_location = task.replace('scratching_', '')
#            file_name = ''.join([home, '/svn/robot1_data/usr/ari/data/base_selection/', task_name, '/', model, '/', task_location, '/', task, '_', model, '_cma_score_data.pkl'])
#            return load_pickle(file_name)
        # return self.load_spickle(file_name)
#        print 'loading file with name ', file_name
#        try:
#            return joblib.load(file_name)
#        except IOError:
#            print 'Load failed, sorry.'
#            return None

if __name__ == "__main__":
    import optparse
    p = optparse.OptionParser()

    # Option to select what mode to run the service in. Normal is for typical use. Test
    # is for development testing. Sim is to let you publish reference frames (e.g. head, AR tag)
    # manually for the service to use. Demo is to use frames from the motion capture room.
    p.add_option('--mode', action='store', dest='mode', default='normal', type='string',
                 help='Select mode of use (normal, test, sim, or demo)')

    # Option to select if the user is in a wheelchair or bed. Helps select the data files to
    # load. Some tasks do not have data for chair or bed.
    p.add_option('--user', action='store', dest='user', default='chair', type='string',
                 help='Select if the user is in a chair or autobed (chair or autobed)')

    # Option to select what data files to load when the service starts. If you want full functionality
    # you can just do all, but it will take a while to initialize (30 seconds).
    p.add_option('--load', action='store', dest='load', default='shaving', type='string',
                 help='Select tasks to load (all, paper (for the two tasks used in the paper), shaving, brushing, '
                      'feeding, feeding, bathing, scratching_chest, scratching_thigh_left, wiping_mouth'
                      'scratching_thigh_right, scratching_forearm_left, scratching_forearm_right,'
                      'scratching_upper_arm_left, scratching_upper_arm_right)')

    opt, args = p.parse_args()

    rospy.init_node('base_selection_server')
    selector = BaseSelector(mode=opt.mode, model=opt.user, load=opt.load)
    rospy.spin()


