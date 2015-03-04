#!/usr/bin/env python

from threading import Lock
import copy
import numpy as np
import math as m
import ros

import roslib
roslib.load_manifest('hrl_face_adls')
import rospy
from hrl_msgs.msg import FloatArrayBare
from std_msgs.msg import String, Int32, Int8, Bool
from geometry_msgs.msg import PoseStamped, Point, Quaternion, PoseWithCovarianceStamped, Twist
from tf import TransformListener, transformations as tft

from hrl_pr2_ar_servo.msg import ARServoGoalData
from hrl_base_selection.srv import BaseMove_multi  # , BaseMoveRequest
from hrl_ellipsoidal_control.msg import EllipsoidParams
from pr2_controllers_msgs.msg import SingleJointPositionActionGoal
from hrl_srvs.srv import None_Bool, None_BoolResponse

roslib.load_manifest('hrl_base_selection')
from helper_functions import createBMatrix, is_number, Bmat_to_pos_quat
from tf_goal import TF_Goal
# from navigation_feedback import *

POSES = {'Knee': ([0.443, -0.032, -0.716], [0.162, 0.739, 0.625, 0.195]),
         'Arm': ([0.337, -0.228, -0.317], [0.282, 0.850, 0.249, 0.370]),
         'Shoulder': ([0.108, -0.236, -0.105], [0.346, 0.857, 0.238, 0.299]),
         'Face': ([0.252, -0.067, -0.021], [0.102, 0.771, 0.628, -0.002])}

class ServoingManager(object):
    """ Manager for providing test goals to pr2 ar servoing. """

    def __init__(self, mode=None):
        self.task = 'feeding_quick'
        self.model = 'autobed' # options are 'chair' and 'autobed'
        self.mode = mode

        if self.model == 'autobed':
            self.bed_state_z = 0.
            self.bed_state_head_theta = 0.
            self.bed_state_leg_theta = 0.
            self.autobed_sub = rospy.Subscriber('/abdout0', FloatArrayBare, self.bed_state_cb)
            self.autobed_pub = rospy.Publisher('/abdin0', FloatArrayBare, latch=True)

            self.world_B_head = None
            self.world_B_ref_model = None
            self.world_B_robot = None

        self.tfl = TransformListener()

        if self.mode == 'manual' or True:
            self.base_pose = None
            self.object_pose = None
            self.raw_head_pose = None
            self.raw_base_pose = None
            self.raw_object_pose = None
            self.raw_head_sub = rospy.Subscriber('/head_frame', PoseStamped, self.head_frame_cb)
            self.raw_base_sub = rospy.Subscriber('/robot_frame', PoseStamped, self.robot_frame_cb)
            self.raw_object_sub = rospy.Subscriber('/reference', PoseStamped, self.reference_cb)
            # self.head_pub = rospy.Publisher('/head_frame', PoseStamped, latch=True)
            # self.base_pub = rospy.Publisher('/robot_frame', PoseStamped, latch=True)
            # self.object_pub = rospy.Publisher('/reference', PoseStamped, latch=True)
            self.base_goal_pub = rospy.Publisher('/base_goal', PoseStamped, latch=True)
            # self.robot_location_pub = rospy.Publisher('/robot_location', PoseStamped, latch=True)
            # self.navigation = NavigationHelper(robot='/robot_location', target='/base_goal')
        self.goal_data_pub = rospy.Publisher("ar_servo_goal_data", ARServoGoalData)
        self.servo_goal_pub = rospy.Publisher('servo_goal_pose', PoseStamped, latch=True)
        self.reach_goal_pub = rospy.Publisher("arm_reacher/goal_pose", PoseStamped)
        # self.test_pub = rospy.Publisher("test_goal_pose", PoseStamped, latch=True)
        # self.test_head_pub = rospy.Publisher("test_head_pose", PoseStamped, latch=True)
        self.feedback_pub = rospy.Publisher('wt_log_out', String)
        self.torso_lift_pub = rospy.Publisher('torso_controller/position_joint_action/goal',
                                              SingleJointPositionActionGoal, latch=True)

        self.base_selection_client = rospy.ServiceProxy("select_base_position", BaseMove_multi)

        self.reach_service = rospy.ServiceProxy("/base_selection/arm_reach_enable", None_Bool)

        self.ui_input_sub = rospy.Subscriber("action_location_goal", String, self.ui_cb)
        self.servo_fdbk_sub = rospy.Subscriber("/pr2_ar_servo/state_feedback", Int8, self.servo_fdbk_cb)

        self.lock = Lock()
        self.head_pose = None
        self.goal_pose = None
        self.marker_topic = None
        rospy.loginfo("[%s] Ready" %rospy.get_name())

        self.base_selection_complete = False

        self.send_task_count = 0

    def servo_fdbk_cb(self, msg):
        if not msg.data == 5:
            return
        #self.reach_goal_pub.publish(self.goal_pose)
        #self.feedback_pub.publish("Servoing succeeded. Reaching to location.")
        #rospy.loginfo("Servoing Succeeded. Sending goal to arm reacher.")
        msg = "Servoing Succeeded."
        if self.base_selection_complete and self.send_task_count > 1:
            movement = False
            msg = "Servoing Succeeded. Arms will proceed to move."
            self.base_selection_complete = False
            self.send_task_count = 0
            movement = self.call_arm_reacher()
            
        self.feedback_pub.publish(msg)
        rospy.loginfo(msg)

    def ui_cb(self, msg):
        if self.model == 'chair':
            self.send_task_count = 0
        if self.send_task_count > 4:
            self.send_task_count = 0
        if self.send_task_count > 0:
            self.send_task_count += 1
            return
        if self.model == 'chair':
            self.send_task_count = 3
        self.base_selection_complete = False
        self.head_pose = self.world_B_head
        if self.head_pose is None:
            log_msg = "Please register your head before sending a task."
            self.feedback_pub.publish(String(log_msg))
            rospy.loginfo("[%s] %s" % (rospy.get_name(), log_msg))
            return
        rospy.loginfo("[%s] Found head frame" % rospy.get_name());
        # self.test_head_pub.publish(self.head_pose)

        loc = msg.data
        if loc not in POSES:
            log_msg = "Invalid goal location. No Known Pose for %s" % loc
            self.feedback_pub.publish(String(log_msg))
            rospy.loginfo("[%s]" % rospy.get_name() + log_msg)
            return
        rospy.loginfo("[%s] Received valid goal location: %s" % (rospy.get_name(), loc))
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

        base_goals = []
        configuration_goals = []
        goal_array, config_array = self.call_base_selection()
        for item in goal_array:
            base_goals.append(item)
        for item in config_array:
            configuration_goals.append(item)
        base_goals[0] = 1.1
        base_goals[1] = -1.1

        # print "Base Goals returned:\r\n", base_goals
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
        # Here should publish configuration_goal items to robot Z axis and to Autobed.
        # msg.tag_goal_pose.header.frame_id
        torso_lift_msg = SingleJointPositionActionGoal()
        torso_lift_msg.goal.position = configuration_goals[0]
        self.torso_lift_pub.publish(torso_lift_msg)

        # Move autobed if we are dealing with autobed. If not autobed, don't move it. Temporarily fixed to True for
        # testing
        if self.model == 'autobed':
            # autobed_goal = FloatArrayBare()
            # autobed_goal.data = [configuration_goals[2], configuration_goals[1]+9, self.bed_state_leg_theta]
            # self.autobed_pub.publish(autobed_goal)
            print 'The autobed should be set to a height of: ', configuration_goals[1]+9
            print 'The autobed should be set to a head rest angle of: ', configuration_goals[2]


        if self.mode == 'manual':
            self.navigation.start_navigate()
        elif True:
            goal_B_ref_model= np.matrix([[m.cos(base_goals[2]), -m.sin(base_goals[2]),     0.,  base_goals[0]],
                                         [m.sin(base_goals[2]),  m.cos(base_goals[2]),     0.,  base_goals[1]],
                                         [             0.,               0.,     1.,        0.],
                                         [             0.,               0.,     0.,        1.]])
            world_B_goal = self.world_B_ref_model*goal_B_ref_model.I
            # pub_goal_tf = TF_Goal(world_B_goal, self.tfl)
            if self.servo_to_pose(world_B_goal):
                self.base_selection_complete = True
                print 'At desired location!!'
        else:
            self.servo_goal_pub.publish(base_goals_list[0])
            ar_data = ARServoGoalData()
            # 'base_link' in msg.tag_goal_pose.header.frame_id
            with self.lock:
                ar_data.tag_id = -1
                ar_data.marker_topic = self.marker_topic
                ar_data.tag_goal_pose = base_goals_list[0]
                self.action = None
                self.location = None
            self.feedback_pub.publish("Base Position Found. Please use servoing tool.")
            rospy.loginfo("[%s] Base position found. Sending Servoing goals." % rospy.get_name())
        self.send_task_count += 1
        # self.goal_data_pub.publish(ar_data)

    def servo_to_pose(self, world_B_goal):
        # self.tfl.waitForTransform('/base_link', '/r_forearm_cam_optical_frame', rospy.Time(), rospy.Duration(15.0))
        # (trans, rot) = self.tf_listener.lookupTransform('/base_link', '/r_forearm_cam_optical_frame', rospy.Time())
        #
        # self.tfl.waitForTransform(


        # ref_model_B_goal = np.matrix([[m.cos(goal_base_pose[2]), -m.sin(goal_base_pose[2]),     0.,  goal_base_pose[0]],
        #                               [m.sin(goal_base_pose[2]),  m.cos(goal_base_pose[2]),     0.,  goal_base_pose[1]],
        #                               [             0.,               0.,     1.,        0.],
        #                               [             0.,               0.,     0.,        1.]])
        base_move_pub = rospy.Publisher('/base_controller/command', Twist)
        # error_pos = 1
        done_moving = False
        rate = rospy.Rate(2)
        while not done_moving:
            done = False
            tw = Twist()
            tw.linear.x=0
            tw.linear.y=0
            tw.linear.z=0
            tw.angular.x=0
            tw.angular.y=0
            tw.angular.z=0
            # while not rospy.is_shutdown() and np.abs(world_B_goal[0, 3]-self.world_B_robot[0, 3]) > 0.1:
            while not done:
                error_mat = self.world_B_robot.I*world_B_goal
                if np.abs(error_mat[0, 3]) < 0.1:
                    done = True
                else:
                    tw.linear.x = np.sign(error_mat[0, 3])*0.15
                    base_move_pub.publish(tw)
                    rospy.sleep(.1)
            rospy.loginfo('Finished moving to X pose!')
            print 'Finished moving to X pose!'
            done = False
            tw = Twist()
            tw.linear.x=0
            tw.linear.y=0
            tw.linear.z=0
            tw.angular.x=0
            tw.angular.y=0
            tw.angular.z=0
            while not done:
                error_mat = self.world_B_robot.I*world_B_goal
                if np.abs(error_mat[1, 3]) < 0.1:
                    done = True
                else:
                    tw.linear.y = np.sign(error_mat[1, 3])*0.15
                    base_move_pub.publish(tw)
                    rospy.sleep(.1)
            rospy.loginfo('Finished moving to Y pose!')
            print 'Finished moving to Y pose!'
            done = False
            tw = Twist()
            tw.linear.x=0
            tw.linear.y=0
            tw.linear.z=0
            tw.angular.x=0
            tw.angular.y=0
            tw.angular.z=0
            while not done:
                error_mat = self.world_B_robot.I*world_B_goal
                if np.abs(m.acos(error_mat[0, 0])) < 0.1:
                    done = True
                else:
                    tw.angular.z = np.sign(m.acos(error_mat[0, 0]))*0.1
                    base_move_pub.publish(tw)
                    rospy.sleep(.1)
            rospy.loginfo('Finished moving to goal pose!')
            print 'Finished moving to goal pose!'
            done_moving = True
            # error_mat = self.world_B_robot.I*self.world_B_ref_model*ref_model_B_goal
            # error_pos = [error_mat[0,3], error_mat[1,3]]
            # error_ori = m.acos(error_mat[0,0])
            # # while not (rospy.is_shutdown() and (np.linalg.norm(error_pos)>0.1)) and False:
            # error_mat = self.world_B_robot.I*self.world_B_ref_model*ref_model_B_goal
            # error_pos = [error_mat[0,3], error_mat[1,3]]
            # move = np.array([error_mat[0,3],error_mat[1,3],error_mat[2,3]])
            # normalized_pos = move / (np.linalg.norm(move))
            # tw = Twist()
            # tw.linear.x=normalized_pos[0]
            # tw.linear.y=normalized_pos[1]
            # tw.linear.z=0
            # tw.angular.x=0
            # tw.angular.y=0
            # tw.angular.z=0
            # base_move_pub.publish(tw)
            # rospy.sleep(.1)
            # # rospy.loginfo('Finished moving to X-Y position. Now correcting orientation!')
            # # print 'Finished moving to X-Y position. Now correcting orientation!'
            # # while not rospy.is_shutdown() and (np.linalg.norm(error_ori)>0.1) and False:
            # error_mat = self.world_B_robot.I*self.world_B_ref_model*ref_model_B_goal
            # error_ori = m.acos(error_mat[0,0])
            # move = -error_ori
            # normalized_ori = move / (np.linalg.norm(move))
            # tw = Twist()
            # tw.linear.x=0
            # tw.linear.y=0
            # tw.linear.z=0
            # tw.angular.x=0
            # tw.angular.y=0
            # tw.angular.z=normalized_ori
            # base_move_pub.publish(tw)
            # rospy.sleep(.1)
            # # self.world_B_robot
            # # self.world_B_head
            # # self.world_B_ref_model
            # # world_B_ref = createBMatrix(self)
            # # error =
            # error_mat = self.world_B_robot.I*self.world_B_ref_model*ref_model_B_goal
            # error_pos = [error_mat[0,3], error_mat[1,3]]
            # error_ori = m.acos(error_mat[0,0])
            # if np.linalg.norm(error_pos)<0.05 and np.linalg.norm(error_ori)<0.05:
            #     done_moving = True
            # # rospy.loginfo('Finished moving to goal pose!')
            # print 'Finished moving to goal pose!'
        return True

    def call_arm_reacher(self):
        # Place holder return
        #bg = PoseStamped()
        #bg.header.stamp = rospy.Time.now()
        #bg.header.frame_id = 'ar_marker'
        #bg.pose.position = Point(0., 0., 0.5)
        #q = tft.quaternion_from_euler(0., np.pi/2, 0.)
        #bg.pose.orientation = Quaternion(*q)
        #return bg
        ## End Place Holder
        self.feedback_pub.publish("Reaching arm to goal, please wait.")
        rospy.loginfo("[%s] Calling arm reacher. Please wait." %rospy.get_name())

        # bm = BaseMoveRequest()
        # bm.model = self.model
        # bm.task = self.task
        # try:
        #     resp = self.call_arm_reacher()
        #     # resp = self.base_selection_client.call(bm)
        # except rospy.ServiceException as se:
        #     rospy.logerr(se)
        #     self.feedback_pub.publish("Failed to find good base position. Please try again.")
        #     return None
        return self.reach_service()

    def call_base_selection(self):
        # Place holder return
        #bg = PoseStamped()
        #bg.header.stamp = rospy.Time.now()
        #bg.header.frame_id = 'ar_marker'
        #bg.pose.position = Point(0., 0., 0.5)
        #q = tft.quaternion_from_euler(0., np.pi/2, 0.)
        #bg.pose.orientation = Quaternion(*q)
        #return bg
        ## End Place Holder
        self.feedback_pub.publish("Finding a good base location, please wait.")
        rospy.loginfo("[%s] Calling base selection. Please wait." %rospy.get_name())

        # bm = BaseMoveRequest()
        # bm.model = self.model
        # bm.task = self.task

        try:
            resp = self.base_selection_client(self.task, self.model)
            # resp = self.base_selection_client.call(bm)
        except rospy.ServiceException as se:
            rospy.logerr(se)
            self.feedback_pub.publish("Failed to find good base position. Please try again.")
            return None
        return resp.base_goal, resp.configuration_goal

    def bed_state_cb(self, data):
        self.bed_state_z = data.data[1]
        self.bed_state_head_theta = data.data[0]
        self.bed_state_leg_theta = data.data[2]



    def base_goal_cb(self, data):
        goal_trans = [data.pose.position.x,
                 data.pose.position.y,
                 data.pose.position.z]
        goal_rot = [data.pose.orientation.x,
               data.pose.orientation.y,
               data.pose.orientation.z,
               data.pose.orientation.w]
        pr2_B_goal = createBMatrix(goal_trans, goal_rot)
        pr2_trans = [data.pose.position.x,
                 data.pose.position.y,
                 data.pose.position.z]
        pr2_rot = [data.pose.orientation.x,
               data.pose.orientation.y,
               data.pose.orientation.z,
               data.pose.orientation.w]
        world_B_pr2 = createBMatrix(pr2_trans, pr2_rot)
        world_B_goal = world_B_pr2*pr2_B_goal

        # self.head_pose = data

    def update_relations(self):
        pr2_trans = [self.raw_base_pose.pose.position.x,
                     self.raw_base_pose.pose.position.y,
                     self.raw_base_pose.pose.position.z]
        pr2_rot = [self.raw_base_pose.pose.orientation.x,
                   self.raw_base_pose.pose.orientation.y,
                   self.raw_base_pose.pose.orientation.z,
                   self.raw_base_pose.pose.orientation.w]
        world_B_pr2 = createBMatrix(pr2_trans, pr2_rot)
        head_trans = [self.raw_head_pose.pose.position.x,
                     self.raw_head_pose.pose.position.y,
                     self.raw_head_pose.pose.position.z]
        head_rot = [self.raw_head_pose.pose.orientation.x,
                   self.raw_head_pose.pose.orientation.y,
                   self.raw_head_pose.pose.orientation.z,
                   self.raw_head_pose.pose.orientation.w]
        world_B_head = createBMatrix(head_trans, head_rot)
        pr2_B_head = world_B_pr2.I*world_B_head
        pos, ori = Bmat_to_pos_quat(pr2_B_head)
        psm = PoseStamped()
        psm.header.frame_id = '/base_link'
        psm.pose.position.x = pos[0]
        psm.pose.position.y = pos[1]
        psm.pose.position.z = pos[2]
        psm.pose.orientation.x = ori[0]
        psm.pose.orientation.y = ori[1]
        psm.pose.orientation.z = ori[2]
        psm.pose.orientation.w = ori[3]
        self.head_pub(psm)

    def head_frame_cb(self, data):
        trans = [data.pose.position.x,
                 data.pose.position.y,
                 data.pose.position.z]
        rot = [data.pose.orientation.x,
               data.pose.orientation.y,
               data.pose.orientation.z,
               data.pose.orientation.w]
        self.world_B_head = createBMatrix(trans, rot)
        # self.update_relations()

    def robot_frame_cb(self, data):
        trans = [data.pose.position.x,
                 data.pose.position.y,
                 data.pose.position.z]
        rot = [data.pose.orientation.x,
               data.pose.orientation.y,
               data.pose.orientation.z,
               data.pose.orientation.w]
        self.world_B_robot = createBMatrix(trans, rot)
        # self.update_relations()

    def reference_cb(self, data):
        trans = [data.pose.position.x,
                 data.pose.position.y,
                 data.pose.position.z]
        rot = [data.pose.orientation.x,
               data.pose.orientation.y,
               data.pose.orientation.z,
               data.pose.orientation.w]
        self.world_B_ref_model = createBMatrix(trans, rot)

    def get_head_pose(self, head_frame="/head_frame"):
        if self.mode == 'manual' or True:
            return self.head_pose
        else:
            try:
                now = rospy.Time.now()
                self.tfl.waitForTransform("/base_link", head_frame, now, rospy.Duration(15))
                pos, quat = self.tfl.lookupTransform("/base_link", head_frame, now)
                head_pose = PoseStamped()
                head_pose.header.frame_id = "/base_link"
                head_pose.header.stamp = now
                head_pose.pose.position = Point(*pos)
                head_pose.pose.orientation = Quaternion(*quat)
                return head_pose
            except Exception as e:
                rospy.loginfo("TF Exception:\r\n%s" %e)
                return None


if __name__ == '__main__':
    rospy.init_node('ar_servo_manager')
    mode = 'normal'  # options are 'manual' for manual base movement using motion capture positioning and auto otherwise
    manager = ServoingManager(mode=mode)
    import sys
    rospy.spin()
