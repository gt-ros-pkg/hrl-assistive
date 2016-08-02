#!/usr/bin/env python

from collections import deque

import rospy, rosparam, rospkg, roslib
import actionlib
from threading import RLock
import math as m
import rospy, rosparam, rospkg, roslib
from hrl_msgs.msg import FloatArrayBare
from std_msgs.msg import String, Int32, Int8, Bool

# from actionlib_msgs.msg import GoalStatus as GS
from geometry_msgs.msg import PoseStamped
import tf
from hrl_task_planning.msg import PDDLState
from hrl_pr2_ar_servo.msg import ARServoGoalData
from hrl_base_selection.srv import BaseMove_multi
from hrl_srvs.srv import None_Bool, None_BoolResponse
from pr2_controllers_msgs.msg import SingleJointPositionActionGoal
# pylint: disable=W0102
from task_smacher import PDDLSmachState


SPA = ["succeeded", "preempted", "aborted"]


def get_action_state(domain, problem, action, args, init_state, goal_state):
    if action == 'FIND_TAG':
        return FindTagState(domain=domain, model=args[0], problem=problem,
                            action=action, action_args=args,
                            init_state=init_state, goal_state=goal_state,
                            outcomes=SPA)
    if action == 'TRACK_TAG':
        return TrackTagState(domain=domain, model=args[0], problem=problem,
                             action=action, action_args=args,
                             init_state=init_state, goal_state=goal_state,
                             outcomes=SPA)
    elif action == 'CONFIGURE_MODEL_ROBOT':
        return ConfigureModelRobotState(domain=domain, task=args[0], model=args[1], problem=problem,
                                        action=action, action_args=args,
                                        init_state=init_state, goal_state=goal_state,
                                        outcomes=SPA)
    elif action == 'CHECK_OCCUPANCY':
        return CheckOccupancyState(domain=domain, model=args[0], problem=problem,
                                   action=action, action_args=args, init_state=init_state,
                                   goal_state=goal_state, outcomes=SPA)
    elif action == 'REGISTER_HEAD':
        return RegisterHeadState(domain=domain, model=args[0], problem=problem,
                                 action=action, action_args=args, init_state=init_state,
                                 goal_state=goal_state, outcomes=SPA)
    elif action == 'CALL_BASE_SELECTION':
        return CallBaseSelectionState(task=args[0], model=args[1], domain=domain, problem=problem,
                                      action=action, action_args=args, init_state=init_state,
                                      goal_state=goal_state, outcomes=SPA)
    elif action == 'MOVE_ROBOT':
        return MoveRobotState(domain=domain, task=args[0], model=args[1], problem=problem, action=action, action_args=args, init_state=init_state, goal_state=goal_state, outcomes=SPA)
    elif action == 'MOVE_ARM':
        return MoveArmState(task=args[0], model=args[1], domain=domain, problem=problem, action=action, action_args=args, init_state=init_state, goal_state=goal_state, outcomes=SPA)
    elif action == 'DO_TASK':
        return PDDLSmachState(domain=domain, problem=problem, action=action, action_args=args, init_state=init_state, goal_state=goal_state, outcomes=SPA)



class FindTagState(PDDLSmachState):
    def __init__(self, model, domain, *args, **kwargs):
        super(FindTagState, self).__init__(domain=domain, *args, **kwargs)
        self.start_finding_AR_publisher = rospy.Publisher('find_AR_now', Bool, queue_size=1)
        self.domain = domain
        self.state_pub = rospy.Publisher('/pddl_tasks/state_updates', PDDLState, queue_size=10, latch=True)
        self.model = model
        self.ar_tag_found = False

    def on_execute(self, ud):
        print "Start Looking For Tag"
        self.start_finding_AR_publisher.publish(True)
        print "Waiting to see if tag found"
        rospy.Subscriber('AR_acquired', Bool, self.found_ar_tag_cb)
        while not rospy.is_shutdown():
            if self.ar_tag_found:
                print "Tag FOUND"
                rospy.loginfo("AR Tag Found")
                state_update = PDDLState()
                state_update.domain = self.domain
                state_update.predicates = ['(FOUND-TAG %s)' % self.model]
                print "Publishing (FOUND-TAG) update"
                self.state_pub.publish(state_update)
                return
            rospy.sleep(1)

    def found_ar_tag_cb(self, msg):
        found_ar_tag = msg.data
        if found_ar_tag:
            self.ar_tag_found = True


class TrackTagState(PDDLSmachState):
    def __init__(self, model, domain, *args, **kwargs):
        super(TrackTagState, self).__init__(domain=domain, *args, **kwargs)
        self.start_tracking_AR_publisher = rospy.Publisher('track_AR_now', Bool, queue_size=1)
        self.model = model

    def on_execute(self, ud):
        print "Starting to track AR Tag"
        self.start_tracking_AR_publisher.publish(True)


class RegisterHeadState(PDDLSmachState):
    def __init__(self, model, domain, *args, **kwargs):
        super(RegisterHeadState, self).__init__(domain=domain, *args, **kwargs)
        self.listener = tf.TransformListener()
        self.state_pub = rospy.Publisher('/pddl_tasks/state_updates', PDDLState, queue_size=10, latch=True)
        self.model = model
        print "Looking for head of person on: %s" % model

    def on_execute(self, ud):
        head_registered = self.get_head_pose()
        if head_registered:
            print "Head Found."
            state_update = PDDLState()
            state_update.domain = self.domain
            state_update.predicates = ['(HEAD-REGISTERED %s)' % self.model]
            print "Publishing (HEAD-REGISTERED) update"
            self.state_pub.publish(state_update)
        else:
            print "Head NOT Found"
            return 'aborted'

    def get_head_pose(self, head_frame="/user_head_link"):
        try:
            now = rospy.Time.now()
            print "[%s] Register Head State Waiting for Head Transform" % rospy.get_name()
            self.listener.waitForTransform("/base_link", head_frame, now, rospy.Duration(5))
            pos, quat = self.listener.lookupTransform("/base_link", head_frame, now)
            return True
        except Exception as e:
            rospy.loginfo("TF Exception:\r\n%s" %e)
            return False


class CheckOccupancyState(PDDLSmachState):
    def __init__(self, model, domain, *args, **kwargs):
        super(CheckOccupancyState, self).__init__(domain=domain, *args, **kwargs)
        self.model = model
        self.state_pub = rospy.Publisher('/pddl_tasks/state_updates', PDDLState, queue_size=10, latch=True)
#        print "Check Occupancy of Model: %s" % model
        if model.upper() == 'AUTOBED':
            self.autobed_occupied_status = False

    def on_execute(self, ud):
        if self.model.upper() == 'AUTOBED':
            print "[%s] Check Occupancy State Waiting for Service" % rospy.get_name()
            rospy.wait_for_service('autobed_occ_status')
            try:
                self.AutobedOcc = rospy.ServiceProxy('autobed_occ_status', None_Bool)
                self.autobed_occupied_status = self.AutobedOcc().data
            except rospy.ServiceException, e:
                print "Service call failed: %s" % e
                return 'aborted'

            if self.autobed_occupied_status:
                state_update = PDDLState()
                state_update.domain = self.domain
                state_update.predicates = ['(OCCUPIED %s)' % self.model]
                self.state_pub.publish(state_update)
                self.goal_reached = False
            else:
                self.goal_reached = False
                return 'aborted'


class MoveArmState(PDDLSmachState):
    def __init__(self, task, model, domain, *args, **kwargs):
        super(MoveArmState, self).__init__(domain=domain, *args, **kwargs)
        self.l_arm_pose_pub = rospy.Publisher('/left_arm/haptic_mpc/goal_pose', PoseStamped, queue_size=1)
        self.domain = domain
        self.task = task
        self.model = model
        self.goal_reached = False
        self.state_pub = rospy.Publisher('/pddl_tasks/state_updates', PDDLState, queue_size=10, latch=True)

    def arm_reach_goal_cb(self, msg):
        self.goal_reached = msg.data

    def publish_goal(self):
        goal = PoseStamped()
        if self.task == 'scratching_knee_left':
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
        elif self.task == 'wiping_face':
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
        else:
            log_msg = 'I dont know where I should be reaching!!'
            return
        self.l_arm_pose_pub.publish(goal)

    def on_execute(self, ud):
        self.publish_goal()
        #Now that goal is published, we wait until goal is reached
        rospy.Subscriber("haptic_mpc/in_deadzone", std_msgs.msg.Bool, self.arm_reach_goal_cb)
        while not rospy.is_shutdown():
            if self.goal_reached:
                rospy.loginfo("Arm Goal Reached")
                state_update = PDDLState()
                state_update.domain = self.domain
                state_update.predicates = ['ARM-REACHED' + ' ' + str(self.task) + ' ' + str(self.model)]
                print "Publishing (ARM-REACHED) update"
                self.state_pub.publish(state_update)
                self.goal_reached = False
                return 'succeeded'
            rospy.sleep(1)


class MoveRobotState(PDDLSmachState):
    def __init__(self, task, model, domain, *args, **kwargs):
        super(MoveRobotState, self).__init__(domain=domain, *args, **kwargs)
        self.model = model
        self.task = task
        self.domain = domain
        self.goal_reached = False
        self.state_pub = rospy.Publisher('/pddl_tasks/state_updates', PDDLState, queue_size=10, latch=True)
        rospy.loginfo('Ready to move! Click to move PR2 base!')
        rospy.loginfo('Remember: The AR tag must be tracked before moving!')
        print 'Ready to move! Click to move PR2 base!'

    def on_execute(self, ud):
        log_msg = 'Moving PR2 base'
        print log_msg
        try:
            base_goals = rospy.get_param('/pddl_tasks/%s/base_goals' % self.domain, base_goals)
        except:
            rospy.logwarn("[%s] MoveRobotState - Cannot find base location on parameter server", rospy.get_name())
            return 'aborted'
        pr2_goal_pose = PoseStamped()
        pr2_goal_pose.header.stamp = rospy.Time.now()
        pr2_goal_pose.header.frame_id = 'base_footprint'
        trans_out = base_goals[:3]
        rot_out = base_goals[3:]
        pr2_goal_pose.pose.position.x = trans_out[0]
        pr2_goal_pose.pose.position.y = trans_out[1]
        pr2_goal_pose.pose.position.z = trans_out[2]
        pr2_goal_pose.pose.orientation.x = rot_out[0]
        pr2_goal_pose.pose.orientation.y = rot_out[1]
        pr2_goal_pose.pose.orientation.z = rot_out[2]
        pr2_goal_pose.pose.orientation.w = rot_out[3]
        goal = ARServoGoalData()
        goal.tag_id = 4
        goal.marker_topic = '/ar_pose_marker'
        goal.tag_goal_pose = pr2_goal_pose
        self.servo_goal_pub.publish(goal)
        rospy.Subscriber('/pr2_ar_servo/state_feedback', Int8, self.base_servoing_cb)
        while not rospy.is_shutdown():
            if self.goal_reached:
                rospy.loginfo("Base Goal Reached")
                state_update = PDDLState()
                state_update.domain = self.domain
                state_update.predicates = ['(BASE-REACHED %s %s)' % (self.task, self.model)]
                print "Publishing (BASE-REACHED) update"
                self.state_pub.publish(state_update)
                self.goal_reached = False
            rospy.sleep(1)

    def base_servoing_cb(self, msg):
        if msg.data == 5:
            self.goal_reached = True


class CallBaseSelectionState(PDDLSmachState):
    def __init__(self, task, model, domain, *args, **kwargs):
        super(CallBaseSelectionState, self).__init__(domain=domain, *args, **kwargs)
        self.state_pub = rospy.Publisher('/pddl_tasks/state_updates', PDDLState, queue_size=10, latch=True)
        print "Base Selection Called for task: %s and Model: %s" %(task, model)
        self.task = task
        self.model = model

    def call_base_selection(self):
        rospy.loginfo("[%s] Calling base selection. Please wait." %rospy.get_name())
        rospy.wait_for_service("select_base_position")
        self.base_selection_client = rospy.ServiceProxy("select_base_position", BaseMove_multi)
        self.domain = domain
        try:
            resp = self.base_selection_client(self.task, self.model)
        except rospy.ServiceException as se:
            rospy.logerr(se)
            return None
        return resp.base_goal, resp.configuration_goal

    def on_execute(self, ud):
        goal_array, config_array = self.call_base_selection()
        if goal_array == None or config_array == None:
            print "Base Selection Returned None"
            return 'aborted'
        for item in goal_array[:7]:
            base_goals.append(item)
        for item in config_array[:3]:
            configuration_goals.append(item)
        print "Base Goals returned:\r\n", base_goals
        print "Configuration Goals returned:\r\n", configuration_goals
        rospy.set_param('/pddl_tasks/%s/base_goals' % self.domain, base_goals)
        rospy.set_param('/pddl_tasks/%s/configuration_goals' % self.domain, configuration_goals)
        state_update = PDDLState()
        state_update.domain = self.domain
        state_update.predicates = ['(BASE-SELECTED %s %s)' % (self.task, self.model)]
        print "Publishing (BASE-SELECTED) update"
        self.state_pub.publish(state_update)


class ConfigureModelRobotState(PDDLSmachState):
    def __init__(self, task, model, domain, *args, **kwargs):
        super(ConfigureModelRobotState, self).__init__(domain=domain, *args, **kwargs)
        self.domain = domain
        self.task = task
        self.model = model
        print "Configuring Model and Robot for task: %s and Model: %s" %(task, model)
        if self.model.upper() == 'AUTOBED':
            self.model_reached = False
        else:
            self.model_reached = True
        self.torso_reached = False
        self.state_pub = rospy.Publisher('/pddl_tasks/state_updates', PDDLState, queue_size=10, latch=True)
        if self.model.upper() == 'AUTOBED':
            self.bed_state_leg_theta = None
            self.autobed_pub = rospy.Publisher('/abdin0', FloatArrayBare, queue_size=1, latch=True)
        self.torso_state_deque = deque([None]*24, 24)
        self.goal_level = -100.0

    def get_torso_height(self):
        rospy.loginfo("[%s] Waiting to get torso height." %rospy.get_name())
        rospy.wait_for_service("return_joint_states")
        try:
            s = rospy.ServiceProxy("return_joint_states", ReturnJointStates)
            resp = s('torso_lift_joint')
        except rospy.ServiceException, e:
            print "error when calling return_joint_states: %s"%e
            resp.position = None
        return resp.position

    def update_torso_state(self):
        torso_configured = None
        if self.goal_level == -100.0:
            self.torso_state_pub.publish(False)
            return
        torso_error = abs(self.goal_level - get_torso_height())
        if torso_error < 0.5:
            torso_configured = True if torso_configured is None else torso_configured
        else:
            torso_configured = False if torso_configured is None else torso_configured
        if torso_configured is None:
            return  # Nothing happening, skip ahead

        self.torso_state_deque.append(torso_configured)
        if None in self.torso_state_deque:
            return
        filtered_torso = True if sum(self.torso_state_deque) > self.torso_state_deque.maxlen/2 else False
        self.torso_reached = filtered_torso

    def torso_goal_cb(self, msg):
        self.goal_level = msg.position

    def bed_state_cb(self, data):
        self.bed_state_leg_theta = data.data[2]

    def bed_status_cb(self, data):
        self.model_reached = data.data

    def on_execute(self, ud):
        if self.model.upper() == 'AUTOBED':
            self.autobed_sub = rospy.Subscriber('/abdout0', FloatArrayBare, self.bed_state_cb)
            try:
                self.configuration_goal = rospy.get_param('/pddl_tasks/%s/configuration_goals' % domain, configuration_goals)
            except:
                rospy.logwarn("[%s] ConfigurationGoalState - Cannot find spine height and autobed config on parameter server", rospy.get_name())
                return 'aborted'
            while (self.bed_state_leg_theta is None):
                rospy.sleep(1)
            autobed_goal = FloatArrayBare()
            autobed_goal.data = ([self.configuration_goal[2],
                                  self.configuration_goal[1],
                                  self.bed_state_leg_theta])
            self.autobed_pub.publish(autobed_goal)
            if self.configuration_goal is not None:
                torso_lift_msg = SingleJointPositionActionGoal()
                torso_lift_msg.goal.position = self.configuration_goals[0]
                self.torso_lift_pub.publish(torso_lift_msg)
            else:
                rospy.loginfo("Some problem in getting TORSO HEIGHT from base selection")
                return 'aborted'
            rospy.Subscriber('abdstatus0', Bool, self.bed_status_cb)
            rospy.Subscriber('torso_controller/position_joint_action/goal',
                             SingleJointPositionActionGoal,
                             self.torso_goal_cb)
            while not rospy.is_shutdown():
                self.update_torso_state()
                if self.model_reached and self.torso_reached:
                    self.model_reached = False
                    self.torso_reached = False
                if self.model_reached:
                    rospy.loginfo("Bed Goal Reached")
                    state_update = PDDLState()
                    state_update.domain = self.domain
                    state_update.predicates = ['(CONFIGURED BED %s %s)' % (self.task, self.model)]
                    print "Publishing (CONFIGURED BED) update"
                    self.state_pub.publish(state_update)
                if self.torso_reached:
                    rospy.loginfo("Torso Goal Reached")
                    state_update = PDDLState()
                    state_update.domain = self.domain
                    state_update.predicates = ['(CONFIGURED SPINE %s %s)' % (self.task, self.model)]
                    print "Publishing (CONFIGURED SPINE) update"
                    self.state_pub.publish(state_update)
                rospy.sleep(1)
