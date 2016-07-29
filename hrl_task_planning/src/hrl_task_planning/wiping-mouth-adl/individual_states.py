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
    task = args[0]
    model = args[1]
    mode = args[2]
    if action == 'FIND_TAG':
        return FindTagState(domain=domain, problem=problem,
                                action=action, action_args=args,
                                init_state=init_state, goal_state=goal_state,
                                outcomes=SPA)
    if action == 'TRACK_TAG':
        return TrackTagState(domain=domain, problem=problem,
                                action=action, action_args=args,
                                init_state=init_state, goal_state=goal_state,
                                outcomes=SPA)
    elif action == 'CONFIGURE_BED_ROBOT':
        return ConfigureBedRobotState(domain=domain, problem=problem,
                                action=action, action_args=args,
                                init_state=init_state, goal_state=goal_state,
                                outcomes=SPA)
    elif action == 'CHECK_BED_OCCUPANCY':
        return CheckBedOccupancyState(domain=domain, problem=problem,
                                  action=action, action_args=args, init_state=init_state,
                                  goal_state=goal_state, outcomes=SPA)
    elif action == 'REGISTER_HEAD':
        return RegisterHeadState(domain=domain, problem=problem,
                                  action=action, action_args=args, init_state=init_state,
                                  goal_state=goal_state, outcomes=SPA)
    elif action == 'CALL_BASE_SELECTION':
        return CallBaseSelectionState(task=task, domain=domain, problem=problem,
                                  action=action, action_args=args, init_state=init_state,
                                  goal_state=goal_state, outcomes=SPA)
    elif action == 'MOVE_ROBOT':
        return MoveRobotState(domain=domain, problem=problem, action=action, action_args=args, init_state=init_state, goal_state=goal_state, outcomes=SPA)
    elif action == 'MOVE_ARM':
        return MoveArmState(task=task, domain=domain, problem=problem, action=action, action_args=args, init_state=init_state, goal_state=goal_state, outcomes=SPA)
    elif action in ['DO_TASK', 'FIND_TAG']:
        return PDDLSmachState(domain, problem, action, args, init_state, goal_state, outcomes=SPA)



class FindTagState(PDDLSmachState):
    def __init__(self, domain, *args, **kwargs):
        super(FindTagState, self).__init__(domain=domain, *args, **kwargs)
        self.start_finding_AR_publisher = rospy.Publisher('find_AR_now', Bool, queue_size=1)

    def on_execute(self):
        self.start_finding_AR_publisher.publish(True)


class TrackTagState(PDDLSmachState):
    def __init__(self, domain, *args, **kwargs):
        super(TrackTagState, self).__init__(domain=domain, *args, **kwargs)
        self.start_tracking_AR_publisher = rospy.Publisher('track_AR_now', Bool, queue_size=1)

    def on_execute(self):
        self.start_tracking_AR_publisher.publish(True)


class RegisterHeadState(PDDLSmachState):
    def __init__(self, domain, *args, **kwargs):
        super(RegisterHeadState, self).__init__(domain=domain, *args, **kwargs)
        self.listener = TransformListener()
        self.head_registered = self.get_head_pose()

    def on_execute(self):
        if self.head_registered:
            return 'succeeded'
        else:
            return 'aborted'

    def get_head_pose(self, head_frame="/user_head_link"):
        try:
            now = rospy.Time.now()
            self.listener.waitForTransform("/base_link", head_frame, now, rospy.Duration(5))
            pos, quat = self.listener.lookupTransform("/base_link", head_frame, now)
            return True
        except Exception as e:
            rospy.loginfo("TF Exception:\r\n%s" %e)
            return False


class CheckBedOccupancyState(PDDLSmachState):
    def __init__(self, domain, *args, **kwargs):
        super(CheckBedOccupancyState, self).__init__(domain=domain, *args, **kwargs)
        self.autobed_occupied_status = False
        rospy.wait_for_service('autobed_occ_status')
        try:
            self.AutobedOcc = rospy.ServiceProxy('autobed_occ_status', None_Bool)
            self.autobed_occupied_status = self.AutobedOcc().data
        except rospy.ServiceException, e:
            print "Service call failed: %s" % e

    def on_execute(self):
        if self.autobed_occ_status:
            return 'succeeded'
        else:
            return 'aborted'

class CheckBedOccupancyState(PDDLSmachState):
    def __init__(self, domain, *args, **kwargs):
        super(CheckBedOccupancyState, self).__init__(domain=domain, *args, **kwargs)
        self.autobed_occupied_status = False
        rospy.wait_for_service('autobed_occ_status')
        try:
            self.AutobedOcc = rospy.ServiceProxy('autobed_occ_status', None_Bool)
            self.autobed_occupied_status = self.AutobedOcc().data
        except rospy.ServiceException, e:
            print "Service call failed: %s" % e

    def on_execute(self):
        if self.autobed_occ_status:
            return 'succeeded'
        else:
            return 'aborted'

class MoveArmState(PDDLSmachState):
    def __init__(self, task, domain, *args, **kwargs):
        super(MoveArmState, self).__init__(domain=domain, *args, **kwargs)
        self.l_arm_pose_pub = rospy.Publisher('/left_arm/haptic_mpc/goal_pose', PoseStamped, queue_size=1)
        self.task = task

    def on_execute(self):
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
            return 'aborted'
        self.l_arm_pose_pub.publish(goal)
        return 'succeeded'


class MoveRobotState(PDDLSmachState):
    def __init__(self, domain, *args, **kwargs):
        super(MoveRobotState, self).__init__(domain=domain, *args, **kwargs)
        self.domain = domain
        base_goals = rospy.get_param('/pddl_tasks/%s/base_goals' % self.domain, base_goals)
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


    def on_execute(self):
        log_msg = 'Moving PR2 base'
        print log_msg
        goal = ARServoGoalData()
        goal.tag_id = 4
        goal.marker_topic = '/ar_pose_marker'
        goal.tag_goal_pose = self.pr2_goal_pose
        self.servo_goal_pub.publish(goal)
        return 'succeeded'


class CallBaseSelectionState(PDDLSmachState):
    def __init__(self, task, domain, *args, **kwargs):
        self.task = task 
        super(CallBaseSelectionState, self).__init__(domain=domain, *args, **kwargs)
        rospy.wait_for_service("select_base_position")
        self.base_selection_client = rospy.ServiceProxy("select_base_position", BaseMove_multi)
        self.domain = domain

    def call_base_selection(self):
        rospy.loginfo("[%s] Calling base selection. Please wait." %rospy.get_name())
        try:
            resp = self.base_selection_client(self.task, self.model)
        except rospy.ServiceException as se:
            rospy.logerr(se)
            return None
        return resp.base_goal, resp.configuration_goal

    def on_execute(self):
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
        return 'succeeded'

class ConfigureBedRobotState(PDDLSmachState):
    def __init__(self, domain, *args, **kwargs):
        super(ConfigureBedRobotState, self).__init__(domain=domain, *args, **kwargs)
        self.frame_lock = RLock()
        self.autobed_sub = rospy.Subscriber('/abdout0', FloatArrayBare, self.bed_state_cb)
        self.autobed_pub = rospy.Publisher('/abdin0', FloatArrayBare, queue_size=1, latch=True)
        self.bed_state_leg_theta = None
        self.configuration_goal = rospy.get_param('/pddl_tasks/%s/configuration_goals' % domain, configuration_goals)

    def bed_state_cb(self, data):
        with self.frame_lock:
            self.bed_state_leg_theta = data.data[2]

    def on_execute(self):
        print 'The autobed should be set to a height of: ', configuration_goals[1], ' cm'
        print 'The autobed should be set to a head rest angle of: ', configuration_goals[2], 'degrees'
        if self.bed_state_leg_theta is not None and self.configuration_goal is not None:
            autobed_goal = FloatArrayBare()
            autobed_goal.data = ([self.configuration_goal[2], 
                                  self.configuration_goal[1], 
                                  self.bed_state_leg_theta])
            self.autobed_pub.publish(autobed_goal)
        else:
            rospy.loginfo("Some problem in getting BED POSE from base selection")
            return 'aborted'
        if self.configuration_goal is not None:
            torso_lift_msg = SingleJointPositionActionGoal()
            torso_lift_msg.goal.position = self.configuration_goals[0]
            self.torso_lift_pub.publish(torso_lift_msg)
        else:
            rospy.loginfo("Some problem in getting TORSO HEIGHT from base selection")
            return 'aborted'
        return 'succeeded'
