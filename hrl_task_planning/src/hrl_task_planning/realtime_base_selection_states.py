#!/usr/bin/env python

# pylint: disable=W0102
import rospy
from task_smacher import PDDLSmachState
from hrl_task_planning.msg import PDDLState
from actionlib_msgs.msg import GoalStatus

SPA = ["succeeded", "preempted", "aborted"]


def get_action_state(domain, problem, action, args, init_state, goal_state):
    if action in ['GET_EE_GOAL', 'GET_FRAME']:
        return PDDLSmachState(domain=domain, problem=problem, action=action, action_args=args, init_state=init_state, goal_state=goal_state, outcomes=SPA)
    elif action == 'SCAN_ENVIRONMENT':
        return ScanEnvironmentState(domain=domain, problem=problem, action=action, action_args=args, init_state=init_state, goal_state=goal_state, outcomes=SPA)
    elif action == 'CLEAR_ENVIRONMENT':
        return ClearEnvironmentState(domain=domain, problem=problem, action=action, action_args=args, init_state=init_state, goal_state=goal_state, outcomes=SPA)
    elif action == 'CALL_BASE_SELECTION':
        ee_goal_param = "/pddl_tasks/%s/%s/%s" % (domain, 'KNOWN', args[0])
        ee_frame_param = "/pddl_tasks/%s/%s/%s" % (domain, 'KNOWN', args[1])
        base_goal_param = "/pddl_tasks/%s/%s/%s" % (domain, 'KNOWN', args[2])
        return CallBaseSelectionState(ee_goal_param, ee_frame_param, base_goal_param, domain=domain, problem=problem, action=action, action_args=args, init_state=init_state, goal_state=goal_state, outcomes=SPA)
    elif action == 'SERVO_OPEN_LOOP':
        base_goal_param = "/pddl_tasks/%s/%s/%s" % (domain, 'KNOWN', args[0])
        return ServoOpenLoopState(base_goal_param, domain=domain, problem=problem, action=action, action_args=args, init_state=init_state, goal_state=goal_state, outcomes=SPA)
    elif action == 'ADJUST_TORSO':
        base_goal_param = "/pddl_tasks/%s/%s/%s" % (domain, 'KNOWN', args[0])
        return AdjustTorsoState(base_goal_param, domain=domain, problem=problem, action=action, action_args=args, init_state=init_state, goal_state=goal_state, outcomes=SPA)
    elif action == 'CLEAR_TORSO_SET':
        return ClearTorsoSetState(domain=domain, problem=problem, action=action, action_args=args, init_state=init_state, goal_state=goal_state, outcomes=SPA)
    elif action == 'CLEAR_AT_GOAL':
        return ClearAtGoalState(domain=domain, problem=problem, action=action, action_args=args, init_state=init_state, goal_state=goal_state, outcomes=SPA)
    elif action in ['CLEAR_EE_GOAL', 'CLEAR_BASE_GOAL', 'CLEAR_FRAME']:
        param = "/pddl_tasks/%s/%s/%s" % (domain, 'KNOWN', args[0])
        return DeleteParamState(param, domain=domain, problem=problem,
                                action=action, action_args=args,
                                init_state=init_state, goal_state=goal_state,
                                outcomes=SPA)


from hrl_base_selection.srv import RealtimeBaseMove, RealtimeBaseMoveRequest
from geometry_msgs.msg import PoseStamped


class CallBaseSelectionState(PDDLSmachState):
    def __init__(self, ee_goal_param, ee_frame_param, base_goal_param, *args, **kwargs):
        super(CallBaseSelectionState, self).__init__(*args, **kwargs)
        self.ee_goal_param = ee_goal_param
        self.ee_frame_param = ee_frame_param
        self.base_goal_param = base_goal_param
        self.base_selection_service = rospy.ServiceProxy('/realtime_select_base_position', RealtimeBaseMove)
        self.ee_pose_pub = rospy.Publisher('/rtbs_ee_goal', PoseStamped, latch=True)

    @staticmethod
    def _dict_to_pose_stamped(ps_dict):
        ps = PoseStamped()
        ps.header.seq = ps_dict['header']['seq']
        ps.header.stamp.secs = ps_dict['header']['stamp']['secs']
        ps.header.stamp.nsecs = ps_dict['header']['stamp']['nsecs']
        ps.header.frame_id = ps_dict['header']['frame_id']
        ps.pose.position.x = ps_dict['pose']['position']['x']
        ps.pose.position.y = ps_dict['pose']['position']['y']
        ps.pose.position.z = ps_dict['pose']['position']['z']
        ps.pose.orientation.x = ps_dict['pose']['orientation']['x']
        ps.pose.orientation.y = ps_dict['pose']['orientation']['y']
        ps.pose.orientation.z = ps_dict['pose']['orientation']['z']
        ps.pose.orientation.w = ps_dict['pose']['orientation']['w']
        return ps

    def on_execute(self, ud):
        try:
            ee_goal_dict = rospy.get_param(self.ee_goal_param)
            ee_goal = self._dict_to_pose_stamped(ee_goal_dict)
            self.ee_pose_pub.publish(ee_goal)
        except (KeyError, rospy.ROSException):
            rospy.logerr("[%s] %s - Error trying to access param: %s", rospy.get_name(), self.__class__.__name__, self.ee_goal_param)
            return 'aborted'
        try:
            ee_frame = rospy.get_param(self.ee_frame_param)
        except (KeyError, rospy.ROSException):
            rospy.logerr("[%s] %s - Error trying to access param: %s", rospy.get_name(), self.__class__.__name__, self.ee_frame_param)
            return 'aborted'
        req = RealtimeBaseMoveRequest()
        req.ee_frame = ee_frame
        req.pose_target = ee_goal
        print "Request:\n", req
        try:
            res = self.base_selection_service.call(req)
        except rospy.ServiceException as se:
            rospy.logerr("[%s] CallBaseSelectionState - Service call failed: %s", rospy.get_name(), se)
            return 'aborted'
        print "Response:\n", res
        goal_data = list(res.base_goal[0:7])  # X, Y, Theta
        goal_data.append(res.configuration_goal[0])  # Torso height
        try:
            rospy.set_param(self.base_goal_param, goal_data)
        except rospy.ROSException:
            rospy.logerr("[%s] %s - Failed to load base goal to parameter server", rospy.get_name(), self.__class__.__name__)
            return 'aborted'

from assistive_teleop.msg import ServoAction, ServoGoal
from geometry_msgs.msg import Point, Quaterion


class ServoOpenLoopState(PDDLSmachState):
    def __init__(self, base_goal_param, *args, **kwargs):
        super(ServoOpenLoopState, self).__init__(*args, **kwargs)
        self.base_goal_param = base_goal_param
        self.action_client =  actionlib.SimpleActionClient('servoing_action', ServoAction)

    def execute(self, ud):
        self._start_execute()
        try:
            base_goal = rospy.get_param(self.base_goal_param)
        except (KeyError, rospy.ROSException):
            rospy.logerr("[%s] %s - Error trying to access param %s", rospy.get_name(), self.__class__.__name__, self.base_goal_param)
            return 'aborted'
        ps = PoseStamped()
        ps.header.frame_id = '/odom_combined'
        ps.header.stamp = rospy.Time.now()
        ps.pose.position = Point(*base_goal[0:3])
        ps.pose.orientation = Quaterion(*base_goal[3:])
        servo_goal = ServoGoal()
        servo_goal.goal = ps
        self.action_client.send_goal(servo_goal)
        self.xyt = base_goal[0:3]
        rate = rospy.Rate(5)
        result_published = False
        while not rospy.is_shutdown():
            action_state = self.action_client.get_state()
            if action_state == GoalStatus.SUCCEEDED:
                if not result_published:
                    state_msg = PDDLState()
                    state_msg.domain = self.domain
                    state_msg.problem = self.problem
                    state_msg.predicates = ['(TORSO_SET %s)' % self.base_goal_arg]
                    self.pddl_pub.publish(state_msg)
                    result_published = True
            elif action_state not in [GoalStatus.ACTIVE, GoalStatus.PENDING]:
                rospy.logwarn("[%s] %s - Servo Action Failed", rospy.get_name(), self.__class__.__name__)
                return 'aborted'
            result = self._check_pddl_status()
            if result is not None:
                return result
            rate.sleep()


from pr2_controllers_msgs.msg import SingleJointPositionAction, SingleJointPositionGoal


class AdjustTorsoState(PDDLSmachState):
    def __init__(self, base_goal_param, *args, **kwargs):
        super(AdjustTorsoState, self).__init__(*args, **kwargs)
        self.domain = kwargs['domain']
        self.problem = kwargs['problem']
        self.pddl_pub = rospy.Publisher('/pddl_tasks/state_updates', PDDLState)
        self.torso_client = actionlib.SimpleActionClient('torso_controller/position_joint_action', SingleJointPositionAction)
        self.base_goal_param = base_goal_param
        self.base_goal_arg = kwargs['action_args'][0]

    def execute(self, ud):
        self._start_execute()
        try:
            base_goal = rospy.get_param(self.base_goal_param)
        except (KeyError, rospy.ROSException):
            rospy.logerr("[%s] %s - Error trying to access param %s", rospy.get_name(), self.__class__.__name__, self.base_goal_param)
            return 'aborted'
        sjpg = SingleJointPositionGoal()
        sjpg.position = base_goal[-1]  # Should always be the final entry...
        self.torso_client.send_goal(sjpg)
        rate = rospy.Rate(5)
        result_published = False
        while not rospy.is_shutdown():
            action_state = self.torso_client.get_state()
            if action_state == GoalStatus.SUCCEEDED:
                if not result_published:
                    state_msg = PDDLState()
                    state_msg.domain = self.domain
                    state_msg.problem = self.problem
                    state_msg.predicates = ['(TORSO_SET %s)' % self.base_goal_arg]
                    self.pddl_pub.publish(state_msg)
                    result_published = True
            elif action_state not in [GoalStatus.ACTIVE, GoalStatus.PENDING]:
                rospy.logwarn("[%s] %s - Move Torso Action Failed", rospy.get_name(), self.__class__.__name__)
                return 'aborted'
            result = self._check_pddl_status()
            if result is not None:
                return result
            rate.sleep()


class ClearTorsoSetState(PDDLSmachState):
    def __init__(self, *args, **kwargs):
        super(ClearTorsoSetState, self).__init__(self, *args, **kwargs)
        self.domain = kwargs['domain']
        self.problem = kwargs['problem']
        self.pddl_pub = rospy.Publisher('/pddl_tasks/state_updates', PDDLState)
        self.base_goal_arg = kwargs['action_args'][0]

    def on_execute(self, ud):
        state_msg = PDDLState()
        state_msg.domain = self.domain
        state_msg.problem = self.problem
        state_msg.predicates = ['(NOT (TORSO_SET %s))' % self.base_goal_arg]
        self.pddl_pub.publish(state_msg)


class ClearAtGoalState(PDDLSmachState):
    def __init__(self, *args, **kwargs):
        super(ClearAtGoalState, self).__init__(self, *args, **kwargs)
        self.domain = kwargs['domain']
        self.problem = kwargs['problem']
        self.pddl_pub = rospy.Publisher('/pddl_tasks/state_updates', PDDLState)
        self.at_goal_arg = kwargs['action_args'][0]

    def on_execute(self, ud):
        state_msg = PDDLState()
        state_msg.domain = self.domain
        state_msg.problem = self.problem
        state_msg.predicates = ['(NOT (AT_GOAL %s))', self.at_goal_arg]
        self.pddl_pub.publish(state_msg)

from assistive_teleop.msg import HeadSweepAction, HeadSweepGoal
import actionlib
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class ScanEnvironmentState(PDDLSmachState):
    def __init__(self, *args, **kwargs):
        super(ScanEnvironmentState, self).__init__(*args, **kwargs)
        self.domain = kwargs['domain']
        self.problem = kwargs['problem']
        self.pddl_pub = rospy.Publisher('/pddl_tasks/state_updates', PDDLState)
        self.scan_actioin_client = actionlib.SimpleActionClient('/head_sweep_action', HeadSweepAction)
        found = self.scan_actioin_client.wait_for_server(rospy.Duration(5))
        if not found:
            rospy.logerr("[%s] Failed to find head_sweep_action serer", rospy.get_name())
            raise RuntimeError("Scan Environment State failed to connect to head_sweep_action server")

    def run_sweep(self):
        # Create scan trajectory start + end points
        jtp_start = JointTrajectoryPoint()
        jtp_start.positions = [0.9, 0.73]
        jtp_end = JointTrajectoryPoint()
        jtp_end.positions = [-0.9, 0.73]
        jtp_end.time_from_start = rospy.Duration(3)
        # Create scan trajectory
        jt = JointTrajectory()
        jt.joint_names = ['head_pan_joint', 'head_tilt_joint']
        jt.points = [jtp_start, jtp_end]
        # Fill out goal with trajectory
        goal = HeadSweepGoal()
        goal.sweep_trajectory = jt
        # Send goal
        self.scan_actioin_client.send_goal(goal)

    def execute(self, ud):
        """ Call head Sweep Action in separate thread, publish state once done."""
        self._start_execute()
        self.run_sweep()
        # Monitor for response and/or PDDL state updates
        published_pddl = False
        rate = rospy.Rate(5)
        while not rospy.is_shutdown():
            scan_state = self.scan_actioin_client.get_state()
            if scan_state == GoalStatus.SUCCEEDED:
                if not published_pddl:
                    state_msg = PDDLState()
                    state_msg.domain = self.domain
                    state_msg.problem = self.problem
                    state_msg.predicates = ['(SCAN_COMPLETE)']
                    self.pddl_pub.publish(state_msg)
                    published_pddl = True  # Avoids repeated publishing
            elif scan_state not in [GoalStatus.ACTIVE, GoalStatus.PENDING]:
                rospy.logwarn("[%s] ScanEnvironmentState - Scan Environment Action Failed", rospy.get_name())
                return 'aborted'
            # Wait for updated state
            result = self._check_pddl_status()
            if result is not None:
                return result
            rate.sleep()


class ClearEnvironmentState(PDDLSmachState):
    def __init__(self, *args, **kwargs):
        super(ClearEnvironmentState, self).__init__(*args, **kwargs)
        self.domain = kwargs['domain']
        self.problem = kwargs['problem']
        self.pddl_pub = rospy.Publisher('/pddl_tasks/state_updates', PDDLState)

    def on_execute(self, ud):
        state_msg = PDDLState()
        state_msg.domain = self.domain
        state_msg.problem = self.problem
        state_msg.predicates = ['(NOT (SCAN_COMPLETE))']
        self.pddl_pub.publish(state_msg)


class DeleteParamState(PDDLSmachState):
    def __init__(self, param, *args, **kwargs):
        super(DeleteParamState, self).__init__(*args, **kwargs)
        self.param = param

    def on_execute(self, ud):
        if self.preempt_requested():
            self.service_preempt()
            return 'preempted'
        try:
            rospy.delete_param(self.param)
        except KeyError:
            pass
        except rospy.ROSException:
            rospy.warn("[%s] Error trying to delete param %s", rospy.get_name(), self.param)
            return 'aborted'
