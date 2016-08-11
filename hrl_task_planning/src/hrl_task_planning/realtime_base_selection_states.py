#!/usr/bin/env python

# pylint: disable=W0102
import rospy
from task_smacher import PDDLSmachState
from hrl_task_planning.msg import PDDLState

SPA = ["succeeded", "preempted", "aborted"]


def get_action_state(domain, problem, action, args, init_state, goal_state):
    if action in ['GET_EE_GOAL', 'GET_FRAME']:
        return PDDLSmachState(domain, problem, action, args, init_state, goal_state, outcomes=SPA)
    elif action == 'SCAN_ENVIRONMENT':
        return ScanEnvironmentState(domain, problem, action, args, init_state, goal_state, outcomes=SPA)
    elif action == 'CLEAR_ENVIRONMENT':
        return ClearEnvironmentState(domain, problem, action, args, init_state, goal_state, outcomes=SPA)
    elif action == 'CALL_BASE_SELECTION':
        ee_goal_param = "/pddl_tasks/%s/%s/%s" % (domain, 'KNOWN', args[0])
        ee_frame_param = "/pddl_tasks/%s/%s/%s" % (domain, 'KNOWN', args[1])
        base_goal_param = "/pddl_tasks/%s/%s/%s" % (domain, 'KNOWN', args[2])
        return CallBaseSelectionState(ee_goal_param, ee_frame_param, base_goal_param, domain, problem, action, args, init_state, goal_state, outcomes=SPA)
    elif action == 'SERVO_OPEN_LOOP':
        base_goal_param = "/pddl_tasks/%s/%s/%s" % (domain, 'KNOWN', args[0])
        return ServoOpenLoopState(base_goal_param, domain, problem, action, args, init_state, goal_state, outcomes=SPA)
    elif action == 'ADJUST_TORSO':
        base_goal_param = "/pddl_tasks/%s/%s/%s" % (domain, 'KNOWN', args[0])
        return AdjustTorsoState(base_goal_param, domain, problem, action, args, init_state, goal_state, outcomes=SPA)
    elif action == 'CLEAR_TORSO_SET':
        return ClearTorsoSetState(args[0], domain, problem, action, args, init_state, goal_state, outcomes=SPA)
    elif action in ['CLEAR_EE_GOAL', 'CLEAR_BASE_GOAL', 'CLEAR_FRAME']:
        param = "/pddl_tasks/%s/%s/%s" % (domain, 'KNOWN', args[0])
        return DeleteParamState(param, domain=domain, problem=problem,
                                action=action, action_args=args,
                                init_state=init_state, goal_state=goal_state,
                                outcomes=SPA)


from hrl_base_selection.srv import RealtimeBaseMove, RealtimeBaseMoveRequest


class CallBaseSelectionState(PDDLSmachState):
    def __init__(self, ee_goal_param, ee_frame_param, base_goal_param, *args, **kwargs):
        super(CallBaseSelectionState, self).__init__(*args, **kwargs)
        self.ee_goal_param = ee_goal_param
        self.ee_frame_param = ee_frame_param
        try:
            self.ee_goal = rospy.get_param(self.ee_goal_param)
            self.ee_frame = rospy.get_param(self.ee_frame_param)
        except (KeyError, rospy.ROSException):
            rospy.logerror("[%s] CallBaseSelectionState - Error trying to access params", rospy.get_name())
            return 'aborted'
        self.base_goal_param = base_goal_param
        self.bs_service = rospy.ServiceProxy('/realtime_select_base_position', RealtimeBaseMove)

    def on_execute(self, ud):
        req = RealtimeBaseMoveRequest()
        req.ee_frame = self.ee_frame
        req.pose_target = self.ee_goal
        try:
            res = self.base_selection_service.call(req)
        except rospy.ServiceException as se:
            rospy.logerror("[%s] CallBaseSelectionState - Service call failed: %s", rospy.get_name(), se)
            return 'aborted'
        goal_data = res.base_goal[0:3]  # X, Y, Theta
        goal_data.append(res.configuration_goal[0])  # Torso height
        try:
            rospy.set_param(self.base_goal_param, goal_data)
        except rospy.ROSException:
            rospy.logerror("[%s] %s - Failed to load base goal to parameter server", rospy.get_name(), self.__class__.__name__)
            return 'aborted'


class ServoOpenLoopState(PDDLSmachState):
    def __init__(self, base_goal_param, *args, **kwargs):
        super(ServoOpenLoopState, self).__init__(*args, **kwargs)
        try:
            self.base_goal = rospy.get_param(base_goal_param)
        except (KeyError, rospy.ROSException):
            rospy.logerror("[%s] %s - Error trying to access param %s", rospy.get_name(), self.__class__.__name__, base_goal_param)
            return 'aborted'
        self.xyt = self.base_goal[0:3]

    def execute(self, ud):
        self._start_execute()
        rate = rospy.Rate(5)
        while not rospy.is_shutdown():
            result = self._check_pddl_status()
            if result is not None:
                return result
        rate.sleep()


from pr2_controllers_msgs.msg import SingleJointPositionAction, SingleJointPositionGoal


class AdjustTorsoState(PDDLSmachState):
    def __init__(self, base_goal_param, *args, **kwargs):
        super(AdjustTorsoState, self).__init__(*args, **kwargs)
        self.pddl_pub = rospy.Publisher('/pddl_tasks/state_updates', PDDLState)
        self.torso_client = actionlib.SimpleActionClient('torso_controller/position_joint_action', SingleJointPositionAction)
        try:
            base_goal = rospy.get_param(base_goal_param)
        except (KeyError, rospy.ROSException):
            rospy.logerror("[%s] %s - Error trying to access param %s", rospy.get_name(), self.__class__.__name__, base_goal_param)
            return 'aborted'
        self.torso_goal = base_goal[3]

    def execute(self, ud):
        self._start_execute()
        sjpg = SingleJointPositionGoal()
        sjpg.position = self.torso_goal
        self.torso_client.send_goal(sjpg)
        rate = rospy.Rate(5)
        result_published = False
        while not rospy.is_shutdown():
            action_state = self.torso_client.get_state()
            if action_state == 'SUCCEEDED':
                if not result_published:
                    self.pddl_pub.publish('(AT_GOAL - TORSO)')
                    result_published = True
            elif action_state != 'ACTIVE':
                rospy.logwarn("[%s] %s - Move Torso Action Failed", rospy.get_name(), self.__class__.__name__)
                return 'aborted'
            result = self._check_pddl_status()
            if result is not None:
                return result
            rate.sleep()


class ClearTorsoSetState(PDDLSmachState):
    def __init__(self, base_goal_arg, *args, **kwargs):
        super(ClearTorsoSetState, self).__init__(self, *args, **kwargs)
        self.pddl_pub = rospy.Publisher('/pddl_tasks/state_updates', PDDLState)
        self.base_goal_arg = base_goal_arg

    def on_execute(self, ud):
        self.pddl_pub.publish('(NOT (TORSO_SET %s))', self.base_goal_arg)


from assistive_teleop.msg import HeadSweepAction, HeadSweepActionGoal
import actionlib
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class ScanEnvironmentState(PDDLSmachState):
    def __init__(self, *args, **kwargs):
        super(ScanEnvironmentState, self).__init__(*args, **kwargs)
        self.pddl_pub = rospy.Publisher('/pddl_tasks/state_updates', PDDLState)
        self.scan_actioin_client = actionlib.SimpleActionClient('/head_sweep_action', HeadSweepAction)
        found = self.scan_actioin_client.wait_for_server(rospy.Duration(5))
        if not found:
            rospy.logerror("[%s] Failed to find head_sweep_action serer", rospy.get_name())
            raise RuntimeError("Scan Environment State failed to connect to head_sweep_action server")

    def run_sweep(self):
        # Create scan trajectory start + end points
        jtp_start = JointTrajectoryPoint()
        jtp_start.positions = [1.1, 0.73]
        jtp_end = JointTrajectoryPoint()
        jtp_end.positions = [-1.1, 0.73]
        jtp_end.time_from_start = rospy.Duration(5)
        # Create scan trajectory
        jt = JointTrajectory()
        jt.joint_names = ['head_pan_joint', 'head_tilt_joint']
        jt.points = [jtp_start, jtp_end]
        # Fill out goal with trajectory
        goal = HeadSweepActionGoal()
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
            if scan_state == 'SUCCEEDED':
                if not published_pddl:
                    self.pddl_pub.publish('(SCAN-COMPLETE)')
                    published_pddl = True  # Avoids repeated publishing
            elif scan_state != 'ACTIVE':
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
        self.pddl_pub = rospy.Publisher('/pddl_tasks/state_updates', PDDLState)

    def on_execute(self, ud):
        self.pddl_pub.publish('(NOT (SCAN-COMPLETE))')


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
