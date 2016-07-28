#!/usr/bin/env python

import roslib
roslib.load_manifest('hrl_base_selection')
from threading import RLock
import copy
import numpy as np
import math as m
import rospy, rosparam, rospkg, roslib
from hrl_msgs.msg import FloatArrayBare
from std_msgs.msg import String, Int32, Int8, Bool
from geometry_msgs.msg import PoseStamped, Point, Quaternion, PoseWithCovarianceStamped, Twist
from sensor_msgs.msg import JointState
from tf import TransformListener, transformations as tft
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from hrl_base_selection.srv import BaseMove_multi  # , BaseMoveRequest
# from hrl_ellipsoidal_control.msg import EllipsoidParams
from pr2_controllers_msgs.msg import SingleJointPositionActionGoal
from hrl_srvs.srv import None_Bool, None_BoolResponse
from hrl_pr2_ar_servo.msg import ARServoGoalData

import hrl_haptic_manipulation_in_clutter_msgs.msg as haptic_msgs

roslib.load_manifest('hrl_base_selection')
from helper_functions import createBMatrix, Bmat_to_pos_quat
from hrl_srvs.srv import String_String
import time
# from autobed_occupied_client import autobed_occupied_status_client
# from tf_goal import TF_Goal
# from navigation_feedback import *

POSES = {'Knee': ([0.443, -0.032, -0.716], [0.162, 0.739, 0.625, 0.195]),
         'Arm': ([0.337, -0.228, -0.317], [0.282, 0.850, 0.249, 0.370]),
         'Shoulder': ([0.108, -0.236, -0.105], [0.346, 0.857, 0.238, 0.299]),
         'Face': ([0.252, -0.067, -0.021], [0.102, 0.771, 0.628, -0.002])}

class BaseSelectionManager(object):
    """ Manager for providing test goals to pr2 ar servoing. """

    def __init__(self, mode='ar_tag'):
        self.task = 'scratching_knee_left'
        self.model = 'autobed'  # options are 'chair' and 'autobed'
        self.mode = mode

        self.ar_acquired = False
        self.ar_tracking = False

        self.frame_lock = RLock()

        self.listener = TransformListener()

        self.send_task_count = 0

        self.l_reset_traj = None
        self.r_reset_traj = None

        self.head_pose = None
        self.goal_pose = None
        self.marker_topic = None
        self.define_reset()
        self.r_arm_pub = rospy.Publisher('/right_arm/haptic_mpc/joint_trajectory', JointTrajectory, queue_size=1)
        self.l_arm_pub = rospy.Publisher('/left_arm/haptic_mpc/joint_trajectory', JointTrajectory, queue_size=1)
        self.l_arm_pose_pub = rospy.Publisher('/left_arm/haptic_mpc/goal_pose', PoseStamped, queue_size=1)
        rospy.sleep(1)



        if self.model == 'autobed':
            self.bed_state_z = 0.
            self.bed_state_head_theta = 0.
            self.bed_state_leg_theta = 0.
            self.autobed_move_status = False
            self.autobed_sub = rospy.Subscriber('/abdout0', FloatArrayBare, self.bed_state_cb)
            self.autobed_pub = rospy.Publisher('/abdin0', FloatArrayBare, queue_size=1, latch=True)
            self.autobed_move_status_sub = rospy.Subscriber('/abdstatus0', Bool, self.autobed_move_status_cb)
            # self.autobed_joint_sub = rospy.Subscriber('autobed/joint_states', JointState, self.bed_state_cb)
            rospy.wait_for_service('autobed_occ_status')
            try:
                self.AutobedOcc = rospy.ServiceProxy('autobed_occ_status', None_Bool)
                self.autobed_occupied_status = self.AutobedOcc().data
            except rospy.ServiceException, e:
                print "Service call failed: %s" % e

            # rospy.wait_for_service("/arm_reach_enable")
            # self.armReachActionLeft = rospy.ServiceProxy("/arm_reach_enable", String_String)
            rospy.wait_for_service("select_base_position")
            self.base_selection_client = rospy.ServiceProxy("select_base_position", BaseMove_multi)

            # self.autobed_joint_pub = rospy.Publisher('autobed/joint_states', JointState, queue_size=1)

            self.world_B_head = None
            self.world_B_ref_model = None
            self.world_B_robot = None

            rospack = rospkg.RosPack()
            # self.pkg_path = rospack.get_path('hrl_base_selection')
            # self.autobed_empty_model_file = ''.join([self.pkg_path,'/urdf/empty_autobed.urdf'])
            # self.autobed_occupied_model_file = ''.join([self.pkg_path,'/urdf/occupied_autobed.urdf'])
            # self.autobed_occupied_status = autobed_occupied_status_client().state

        if self.mode == 'ar_tag':
            # self.ar_tag_autobed_sub = rospy.Subscriber('/autobed_pose', PoseStamped, self.ar_tag_autobed_cb)
            # self.ar_tag_head_sub = rospy.Subscriber('/head_pose', PoseStamped, self.ar_tag_head_cb)

            self.nav_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1)
        elif self.mode == 'servo':
            self.servo_goal_pub = rospy.Publisher("ar_servo_goal_data", ARServoGoalData, queue_size=1)

        if self.mode == 'manual':
            self.base_pose = None
            self.object_pose = None
            self.raw_head_pose = None
            self.raw_base_pose = None
            self.raw_object_pose = None
            # self.raw_head_sub = rospy.Subscriber('/head_frame', PoseStamped, self.head_frame_cb)
            # self.raw_base_sub = rospy.Subscriber('/robot_frame', PoseStamped, self.robot_frame_cb)
            # self.raw_object_sub = rospy.Subscriber('/reference', PoseStamped, self.reference_cb)
            # self.head_pub = rospy.Publisher('/head_frame', PoseStamped, latch=True)
            # self.base_pub = rospy.Publisher('/robot_frame', PoseStamped, latch=True)
            # self.object_pub = rospy.Publisher('/reference', PoseStamped, latch=True)
            self.base_goal_pub = rospy.Publisher('/base_goal', PoseStamped, queue_size=1, latch=True)
            # self.robot_location_pub = rospy.Publisher('/robot_location', PoseStamped, latch=True)
            # self.navigation = NavigationHelper(robot='/robot_location', target='/base_goal')

        # self.goal_data_pub = rospy.Publisher("ar_servo_goal_data", ARServoGoalData)
        # self.servo_goal_pub = rospy.Publisher('servo_goal_pose', PoseStamped, latch=True)

        # self.test_pub = rospy.Publisher("test_goal_pose", PoseStamped, latch=True)
        # self.test_head_pub = rospy.Publisher("test_head_pose", PoseStamped, latch=True)

        # self.reach_goal_pub = rospy.Publisher("arm_reacher/goal_pose", PoseStamped, queue_size=1)

        self.feedback_pub = rospy.Publisher('wt_log_out', String, queue_size=1)
        self.torso_lift_pub = rospy.Publisher('torso_controller/position_joint_action/goal',
                                              SingleJointPositionActionGoal, queue_size=10, latch=True)

        self.start_finding_AR_publisher = rospy.Publisher('find_AR_now', Bool, queue_size=1)
        self.start_tracking_AR_publisher = rospy.Publisher('track_AR_now', Bool, queue_size=1)

        self.ar_tag_confirmation_publisher = rospy.Publisher('/pr2_ar_servo/tag_confirm', Bool, queue_size=1, latch=True)


        # rospy.wait_for_service('autobed_occ_status')
        # self.base_selection_client = rospy.ServiceProxy("select_base_position", BaseMove_multi)
        # rospy.wait_for_service('autobed_occ_status')
        # self.reach_service = rospy.ServiceProxy("/base_selection/arm_reach_enable", None_Bool)
        #
        # rospy.wait_for_service("/arm_reach_enable")
        # armReachActionLeft  = rospy.ServiceProxy("/arm_reach_enable", String_String)

        ## Place motions! PR2 will executes it sequencially ----------------
        # print armReachActionLeft("leftKnee")

        self.start_task_input_sub = rospy.Subscriber("action_location_goal", String, self.start_task_ui_cb)
        self.move_base_input_sub = rospy.Subscriber("move_base_to_goal", String, self.move_base_ui_cb)
        self.move_arm_input_sub = rospy.Subscriber("move_arm_to_goal", String, self.move_arm_ui_cb)
        self.reset_arm_input_sub = rospy.Subscriber("reset_arm_ui", String, self.reset_arm_ui_cb)
        self.reset_arm_input_sub = rospy.Subscriber("track_ar_ui", Bool, self.track_ar_ui_cb)
        # self.servo_fdbk_sub = rospy.Subscriber("/pr2_ar_servo/state_feedback", Int8, self.servo_fdbk_cb)

        self.ar_acquired_sub = rospy.Subscriber('AR_acquired', Bool, self.ar_acquired_cb)


        print 'Task manager is ready!!'

        rospy.loginfo("[%s] Ready" %rospy.get_name())

        # self.base_selection_complete = False

        # self.send_task_count = 0

    # def ar_tag_autobed_cb(self, msg):
    #     self.autobed_pose = msg
    #
    # def ar_tag_head_cb(self, msg):
    #     self.head_pose = msg

    def ar_acquired_cb(self, msg):
        if msg.data:
            log_msg = 'The AR tag has been acquired and can now be tracked! We can now proceed!!'
            print log_msg
            self.feedback_pub.publish(String(log_msg))
        else:
            log_msg = 'The AR tag is not acquired.'
            print log_msg
            self.feedback_pub.publish(String(log_msg))
        self.ar_acquired = msg.data

    def move_base_ui_cb(self, msg):
        print 'Trying to move base. Received input to move base from user!'
        if not self.autobed_move_status:
            log_msg = 'Waiting for autobed to complete its configuration change before moving PR2 base!'
            print log_msg
            self.feedback_pub.publish(String(log_msg))
            # rospy.sleep(2)
        elif not self.ar_tracking:
            log_msg = 'AR tag must be tracked during base movement! Do this now!'
            print log_msg
            self.feedback_pub.publish(String(log_msg))
        else:
            log_msg = 'Moving PR2 base'
            print log_msg
            self.feedback_pub.publish(String(log_msg))
            if self.mode == 'ar_tag':
                self.nav_pub.publish(self.pr2_goal_pose)
            elif self.mode == 'servo':
                goal = ARServoGoalData()
                goal.tag_id = 4
                goal.marker_topic = '/ar_pose_marker'
                goal.tag_goal_pose = self.pr2_goal_pose
                self.servo_goal_pub.publish(goal)
                rospy.sleep(2)
                move = Bool()
                move.data = True
                self.ar_tag_confirmation_publisher.publish(move)
        return

    def reset_arm_ui_cb(self, msg):
        print 'Resetting arm configuration!'
        # split_msg = msg.data.split()
        # self.task = ''.join([split_msg[0], '_', split_msg[2], '_', split_msg[1]])

        self.r_arm_pub.publish(self.r_reset_traj)
        self.l_arm_pub.publish(self.l_reset_traj)

        # print self.armReachActionLeft('reach_initialization')
        return

    def track_ar_ui_cb(self, msg):
        #out = Bool()
        #out.data = msg.data
        if msg.data:
            print 'Starting acquiring and tracking AR tag'
            self.ar_acquired = False
            self.start_finding_AR_publisher.publish(msg)
            while not self.ar_acquired and not rospy.is_shutdown():
                rospy.sleep(.5)
            self.ar_tracking = True
        else:
            print 'Stopping tracking AR tag'
            self.start_finding_AR_publisher.publish(msg)
            self.ar_tracking = False
        self.start_tracking_AR_publisher.publish(msg)
        return

    def move_arm_ui_cb(self, msg):
        split_msg = msg.data.split()
        if 'face' in split_msg:
            self.task = ''.join([split_msg[0], '_', split_msg[2]])
        else:
            self.task = ''.join([split_msg[0], '_', split_msg[2], '_', split_msg[1]])
        print 'Moving arm for task: ', self.task
        print self.reach_arm(self.task)
        return

    def start_task_ui_cb(self, msg):
        # print 'My task is: ', msg.data
        move = Bool()
        move.data = False
        self.ar_tag_confirmation_publisher.publish(move)
        split_msg = msg.data.split()
        if 'face' in split_msg:
            self.task = ''.join([split_msg[0], '_', split_msg[2]])
            # self.task = ''.join([split_msg[2], '_', split_msg[0]])
        else:
            self.task = ''.join([split_msg[0], '_', split_msg[2], '_', split_msg[1]])
        print 'My task is: ', self.task
        # if self.send_task_count > 1 and self.base_selection_complete:
        #     self.base_selection_complete = False
        #     self.send_task_count = 0
        #     rospy.sleep(2)
        #     movement = self.call_arm_reacher()
        # if self.send_task_count > 3:
        #     self.send_task_count = 0
        #     return
        # self.base_selection_complete = False
        # self.head_pose = self.world_B_head
        # self.head_pose =

        if not self.ar_tracking:
            log_msg = 'AR tag must be tracked to start the task and do base movement! Do this now!'
            print log_msg
            self.feedback_pub.publish(String(log_msg))
            return
        if not self.get_head_pose():
            log_msg = "Head not currently found. Please look at the head."
            self.feedback_pub.publish(String(log_msg))
            rospy.loginfo("[%s] %s" % (rospy.get_name(), log_msg))
            return
        if not self.get_bed_pose():
            log_msg = "Bed not currently found. Please look at the AR tag by the bed."
            self.feedback_pub.publish(String(log_msg))
            rospy.loginfo("[%s] %s" % (rospy.get_name(), log_msg))
            return

        if self.model == 'autobed':

            # self.autobed_occupied_status = autobed_occupied_status_client().state
            if not self.AutobedOcc().data:
                log_msg = "Bed is currently unoccupied. Can't do the task with nobody in the bed."
                self.feedback_pub.publish(String(log_msg))
                rospy.loginfo("[%s] %s" % (rospy.get_name(), log_msg))
                return
                # autobed_description_file = self.autobed_occupied_model_file
                # paramlist = rosparam.load_file(autobed_description_file)
                # for params, ns in paramlist:
                #     rosparam.upload_params(ns, params)
            # else:
            #     autobed_description_file = self.autobed_empty_model_file
            #     paramlist = rosparam.load_file(autobed_description_file)
            #     for params, ns in paramlist:
            #         rosparam.upload_params(ns, params)
            #     log_msg = "A user is not on the bed. A user must be on the bed to perform the task for that user."
            #     self.feedback_pub.publish(String(log_msg))
            #     rospy.loginfo("[%s] %s" % (rospy.get_name(), log_msg))
            #     return

            # headrest_theta = self.bed_state_head_theta
            # head_x = 0
            #
            # now = rospy.Time.now()
            # self.listener.waitForTransform('/autobed/base_link', '/head_link', now, rospy.Duration(15))
            # (trans, rot) = self.listener.lookupTransform('/autobed/base_link', '/head_link', now)
            # head_y = trans[1]
            #
            # self.set_autobed_user_configuration(headrest_theta, head_x, head_y)

        # loc = msg.data
        # if loc not in POSES:
        #     log_msg = "Invalid goal location. No Known Pose for %s" % loc
        #     self.feedback_pub.publish(String(log_msg))
        #     rospy.loginfo("[%s]" % rospy.get_name() + log_msg)
        #     return
        # rospy.loginfo("[%s] Received valid goal location: %s" % (rospy.get_name(), loc))


        # pos, quat = POSES[loc]
        # goal_ps_ell = PoseStamped()
        # goal_ps_ell.header.frame_id = 'head_frame'
        # goal_ps_ell.pose.position = Point(*pos)
        # goal_ps_ell.pose.orientation = Quaternion(*quat)

        now = rospy.Time.now()
        # self.tfl.waitForTransform('base_link', goal_ps_ell.header.frame_id, now, rospy.Duration(10))
        # goal_ps_ell.header.stamp = now
        # goal_ps = self.tfl.transformPose('base_link', goal_ps_ell)
        # self.test_pub.publish(goal_ps)
        # with self.lock:
        #     self.action = "touch"
        #     # self.goal_pose = goal_ps
        #     self.marker_topic = "r_pr2_ar_pose_marker"  # based on location

        all_base_goals = []
        all_configuration_goals = []
        all_distances_to_goals = []

        start_time = rospy.Time.now()
        goal_array, config_array, distance_array = self.call_base_selection()
        print goal_array
        print config_array
        print distance_array
        print 'Time to get results back from base_selection: ', (rospy.Time.now() - start_time).to_sec()
        if len(goal_array) <= 7:
            log_msg = 'Base Selection has returned one configurations for this task.'
            print log_msg
            self.feedback_pub.publish(String(log_msg))
            a_goal = []
            a_config = []
            a_distance = []
            for item in goal_array:
                a_goal.append(item)
            for item in config_array:
                a_config.append(item)
            for item in distance_array:
                a_distance.append(item)
            all_base_goals.append(a_goal)
            all_configuration_goals.append(a_config)
            all_distances_to_goals.append(a_distance)

        else:
            log_msg = 'Base Selection has returned two configurations for this task. Will use the goal ' \
                  'configuration with its desired PR2 base position closest to the current PR2 base position.'
            print log_msg
            self.feedback_pub.publish(String(log_msg))
            for i in xrange(2):
                a_goal = []
                a_config = []
                a_distance = []
                for item in goal_array[0+i*7:7+i*7]:
                    a_goal.append(item)
                for item in config_array[0+i*3:3+i*3]:
                    a_config.append(item)
                for item in distance_array[0+i*1:1+i*1]:
                    a_distance.append(item)
                all_base_goals.append(a_goal)
                all_configuration_goals.append(a_config)
                all_distances_to_goals.append(a_distance)
        base_goals = all_base_goals[np.argmin(all_distances_to_goals)]
        configuration_goals = all_configuration_goals[np.argmin(all_distances_to_goals)]

        # [0.9], [-0.8], [0.0], [0.14999999999999999], [0.10000000000000001], [1.2217304763960306]
        # base_goals[0] = .9
        # base_goals[1] = -.8
        # base_goals[2] = 0
        # configuration_goals[0]=0.15
        # configuration_goals[1]=0.1
        # configuration_goals[2]=1.221730476396

        print "Base Goals returned:\r\n", all_base_goals
        print "Configuration Goals returned:\r\n", all_configuration_goals
        print "Distances to Goals returned:\r\n", all_distances_to_goals
        # if base_goals is None:
        #     rospy.loginfo("No base goal found")
        #     return
        # base_goals_list = []
        # configuration_goals_list = []
        # for i in xrange(int(len(base_goals)/7)):
        #     psm = PoseStamped()
        #     psm.header.frame_id = '/base_link'
        #     psm.pose.position.x = base_goals[int(0+7*i)]
        #     psm.pose.position.y = base_goals[int(1+7*i)]
        #     psm.pose.position.z = base_goals[int(2+7*i)]
        #     psm.pose.orientation.x = base_goals[int(3+7*i)]
        #     psm.pose.orientation.y = base_goals[int(4+7*i)]
        #     psm.pose.orientation.z = base_goals[int(5+7*i)]
        #     psm.pose.orientation.w = base_goals[int(6+7*i)]
        #     base_goals_list.append(copy.copy(psm))
        #     configuration_goals_list.append([configuration_goals[0+3*i], configuration_goals[1+3*i],
        #                                      configuration_goals[2+3*i]])


        # Move autobed if we are dealing with autobed. If not autobed, don't move it. Temporarily fixed to True for
        # testing
        if self.model == 'autobed':
            autobed_goal = FloatArrayBare()

            # Hack to have PR2 avoid the poles and things under the autobed that are not included in its collision
            # model.
            # if configuration_goals[1] > 1.:
            #     configuration_goals[1] += 15
            #     configuration_goals[0] += 0.14
            # elif configuration_goals[1] < 1.:
            #     configuration_goals[1] += 2
            #     configuration_goals[0] += 0.02
            autobed_goal.data = [configuration_goals[2], configuration_goals[1], self.bed_state_leg_theta]
            self.autobed_pub.publish(autobed_goal)
            print 'The autobed should be set to a height of: ', configuration_goals[1], ' cm'
            print 'The autobed should be set to a head rest angle of: ', configuration_goals[2], 'degrees'
            print 'The PR2 spine should be set to a height of: ', configuration_goals[0]*100., 'cm'

        # Here should publish configuration_goal items to robot Z axis and to Autobed.
        # msg.tag_goal_pose.header.frame_id
        torso_lift_msg = SingleJointPositionActionGoal()
        torso_lift_msg.goal.position = configuration_goals[0]
        self.torso_lift_pub.publish(torso_lift_msg)

        if self.mode == 'servo':
            self.pr2_goal_pose = PoseStamped()
            self.pr2_goal_pose.header.stamp = rospy.Time.now()
            self.pr2_goal_pose.header.frame_id = 'base_footprint'
            trans_out = base_goals[:3]
            rot_out = base_goals[3:]
            self.pr2_goal_pose.pose.position.x = trans_out[0]
            self.pr2_goal_pose.pose.position.y = trans_out[1]
            self.pr2_goal_pose.pose.position.z = trans_out[2]
            self.pr2_goal_pose.pose.orientation.x = rot_out[0]
            self.pr2_goal_pose.pose.orientation.y = rot_out[1]
            self.pr2_goal_pose.pose.orientation.z = rot_out[2]
            self.pr2_goal_pose.pose.orientation.w = rot_out[3]
            rospy.loginfo('Ready to move! Click to move PR2 base!')
            rospy.loginfo('Remember: The AR tag must be tracked before moving!')
            print 'Ready to move! Click to move PR2 base!'



        if self.mode == 'ar_tag':
            self.pr2_goal_pose = PoseStamped()
            self.pr2_goal_pose.header.stamp = rospy.Time.now()
            self.pr2_goal_pose.header.frame_id = 'map'

            pr2_B_goal = np.matrix([[m.cos(base_goals[2]), -m.sin(base_goals[2]), 0., base_goals[0]],
                                    [m.sin(base_goals[2]),  m.cos(base_goals[2]), 0., base_goals[1]],
                                    [0.,                   0.,                    1.,                0.],
                                    [0.,                   0.,                    0.,                1.]])
            now = rospy.Time.now()
            self.listener.waitForTransform('/map', '/base_footprint', now, rospy.Duration(15))
            (trans, rot) = self.listener.lookupTransform('/map', '/base_footprint', now)
            map_B_pr2 = createBMatrix(trans, rot)
            map_B_goal = map_B_pr2*pr2_B_goal

            trans_out, rot_out = Bmat_to_pos_quat(map_B_goal)

            self.pr2_goal_pose.pose.position.x = trans_out[0]
            self.pr2_goal_pose.pose.position.y = trans_out[1]
            self.pr2_goal_pose.pose.position.z = trans_out[2]
            self.pr2_goal_pose.pose.orientation.x = rot_out[0]
            self.pr2_goal_pose.pose.orientation.y = rot_out[1]
            self.pr2_goal_pose.pose.orientation.z = rot_out[2]
            self.pr2_goal_pose.pose.orientation.w = rot_out[3]

            rospy.loginfo('Ready to move! Click to move PR2 base!')
            rospy.loginfo('Remember: The AR tag must be tracked before moving!')
            print 'Ready to move! Click to move PR2 base!'
            # rospy.sleep(2)
            # self.base_selection_complete = True

    def call_base_selection(self):
        self.feedback_pub.publish("Finding a good base location, please wait.")
        rospy.loginfo("[%s] Calling base selection. Please wait." %rospy.get_name())

        # bm = BaseMoveRequest()
        # bm.model = self.model
        # bm.task = self.task

        try:
            resp = self.base_selection_client(self.task, self.model)
            self.feedback_pub.publish("Base Position Found. Will now adjust bed and robot configurations and positions!")
            # resp = self.base_selection_client.call(bm)
        except rospy.ServiceException as se:
            rospy.logerr(se)
            self.feedback_pub.publish("Failed to find good base position. Please try again.")
            return None
        return resp.base_goal, resp.configuration_goal, resp.distance_to_goal

    def bed_state_cb(self, data):
        with self.frame_lock:
            # h = data
            # self.bed_state_z = data.data[1]
            # self.bed_state_head_theta = data.data[0]
            self.bed_state_leg_theta = data.data[2]

    def autobed_move_status_cb(self, data):
        with self.frame_lock:
            self.autobed_move_status = data.data

    def get_head_pose(self, head_frame="/user_head_link"):
        try:
            now = rospy.Time.now()
            self.listener.waitForTransform("/base_link", head_frame, now, rospy.Duration(5))
            pos, quat = self.listener.lookupTransform("/base_link", head_frame, now)
            return True
        except Exception as e:
            rospy.loginfo("TF Exception:\r\n%s" %e)
            return False

    def get_bed_pose(self, bed_frame="/autobed/base_link"):
        try:
            now = rospy.Time.now()
            self.listener.waitForTransform("/base_link", bed_frame, now, rospy.Duration(5))
            pos, quat = self.listener.lookupTransform("/base_link", bed_frame, now)
            return True
        except Exception as e:
            rospy.loginfo("TF Exception:\r\n%s" %e)
            return False

    def define_reset(self):
        r_reset_traj_point = JointTrajectoryPoint()
        r_reset_traj_point.positions = [-3.14/2, -0.52, 0.00, -3.14*2/3, 0., -1.5, 0.0]

        r_reset_traj_point.velocities = [0.0]*7
        r_reset_traj_point.accelerations = [0.0]*7
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
        l_reset_traj_point = JointTrajectoryPoint()
        # l_reset_traj_point.positions = [0.0, 1.35, 0.00, -1.60, -3.14, -0.3, 0.0]
        l_reset_traj_point.positions = [0.7629304700932569, -0.3365186041095207, 0.5240000202473829,
                                        -2.003310310963515, 0.9459734129025158, -1.7128778450423763, 0.6123854412633384]
        l_reset_traj_point.velocities = [0.0]*7
        l_reset_traj_point.accelerations = [0.0]*7
        l_reset_traj_point.time_from_start = rospy.Duration(5)
        self.l_reset_traj = JointTrajectory()
        self.l_reset_traj.joint_names = ['l_shoulder_pan_joint',
                                         'l_shoulder_lift_joint',
                                         'l_upper_arm_roll_joint',
                                         'l_elbow_flex_joint',
                                         'l_forearm_roll_joint',
                                         'l_wrist_flex_joint',
                                         'l_wrist_roll_joint']
        self.l_reset_traj.points.append(l_reset_traj_point)

    def reach_arm(self, task):
        goal = PoseStamped()
        if task == 'scratching_knee_left':
            goal.pose.position.x = -0.06310556
            goal.pose.position.y = 0.07347758+0.05
            goal.pose.position.z = 0.00485197
            goal.pose.orientation.x = 0.48790861
            goal.pose.orientation.y = -0.50380292
            goal.pose.orientation.z = 0.51703901
            goal.pose.orientation.w = -0.4907122
            goal.header.frame_id = '/autobed/calf_left_link'
            log_msg = 'Reaching to left knee!'
            print log_msg
            self.feedback_pub.publish(String(log_msg))
        elif task == 'wiping_face':
            goal.pose.position.x = 0.17
            goal.pose.position.y = 0.
            goal.pose.position.z = -0.16
            goal.pose.orientation.x = 0.
            goal.pose.orientation.y = 0.
            goal.pose.orientation.z = 1.
            goal.pose.orientation.w = 0.
            goal.header.frame_id = '/autobed/head_link'
            log_msg = 'Reaching to mouth!'
            print log_msg
            self.feedback_pub.publish(String(log_msg))
        else:
            log_msg = 'I dont know where I should be reaching!!'
            self.feedback_pub.publish(String(log_msg))
        self.l_arm_pose_pub.publish(goal)




if __name__ == '__main__':
    rospy.init_node('base_selection_task_manager')
    mode = 'servo'  # options are 'ar_tag' for using ar tags to locate the bed and user, 'mo-cap' for
                     # using motion capture positioning, 'servo' to use servoing
    manager = BaseSelectionManager(mode=mode)
    rospy.spin()
