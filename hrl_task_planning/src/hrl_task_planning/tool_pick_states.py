import rospy
from hrl_task_planning.msg import PDDLState, PDDLProblem
import actionlib
from actionlib_msgs.msg import GoalStatus

# pylint: disable=W0102
from task_smacher import PDDLSmachState, StartNewTaskState


SPA = ["succeeded", "preempted", "aborted"]


def get_action_state(domain, problem, action, args, init_state, goal_state):
    if action == 'AUTO-GRASP-TOOL':
        return ToolGraspState(tool=args[0], hand=args[1], domain=domain, problem=problem,
                              action=action, action_args=args, init_state=init_state,
                              goal_state=goal_state, outcomes=SPA)
    elif action == 'RESET-AUTO-TRIED':
        return ResetAutoTriedState(domain=domain, problem=problem, action=action, action_args=args, init_state=init_state, goal_state=goal_state, outcomes=SPA)
    elif action == 'PLACE':
        rospy.set_param('/pddl_tasks/%s/default_goal' % domain, '(PLACED %s)' % args[1])
        problem_msg = PDDLProblem()
        problem_msg.domain = 'place'
        problem_msg.name = problem
        return StartNewTaskState(problem_msg, domain=domain, problem=problem,
                                 action=action, action_args=args, init_state=init_state,
                                 goal_state=goal_state, outcomes=SPA)
    elif action in ['MANUAL-GRASP-TOOL', 'FIND-TAG']:
        return PDDLSmachState(domain, problem, action, args, init_state, goal_state, outcomes=SPA)


from hrl_pr2_tool_grasp.msg import ARToolGraspAction, ARToolGraspGoal


class ToolGraspState(PDDLSmachState):
    def __init__(self, tool, hand, *args, **kwargs):
        super(ToolGraspState, self).__init__(*args, **kwargs)
        self.tool = tool
        print "TGS receved tool [%s], hand [%s]" % (tool, hand)
        self.arm = "right_arm" if hand[0].upper() == 'R' else "left_arm"
        self.published_pddl = False
        self.pddl_pub = rospy.Publisher('/pddl_tasks/state_updates', PDDLState, queue_size=2)
        self.action_client = actionlib.SimpleActionClient('/%s/ar_tool_grasp_action' % self.arm, ARToolGraspAction)
        rospy.loginfo("[%s] Waiting for ar_tool_grasp_action server" % (rospy.get_name()))
        if not self.action_client.wait_for_server(rospy.Duration(7)):
            rospy.logerr("[%s] Could not find ar_tool_grasp_action server", rospy.get_name())
        else:
            rospy.loginfo("[%s] ar_tool_grasp_action server found", rospy.get_name())

    def publish_done_state(self):
        state_msg = PDDLState()
        state_msg.domain = self.domain
        state_msg.problem = self.problem
        state_msg.predicates = ['(AUTO-GRASP-DONE)']
        self.pddl_pub.publish(state_msg)
        self.published_pddl = True

    def execute(self, ud):
        """ Call head Sweep Action in separate thread, publish state once done."""
        self._start_execute()
        # Send Goal
        goal = ARToolGraspGoal()
        try:
            param = '/tools/%s/tag_id' % self.tool.lower()
            goal.tag_id = rospy.get_param(param)
        except KeyError:
            rospy.logwarn("[%s] %s - Error loading param %s", rospy.get_name(), self.__class__.__name__, param)
            return 'aborted'

        # Monitor goal progress
        self.action_client.send_goal(goal)
        # Monitor for response and/or PDDL state updates
        self.published_pddl = False
        rate = rospy.Rate(5)
        while not rospy.is_shutdown():
            grasp_state = self.action_client.get_state()
            print "Status: ", grasp_state
            if grasp_state == GoalStatus.SUCCEEDED:
                print "SUCCEEDED!"
                if not self.published_pddl:
                    self.publish_done_state()
            elif grasp_state not in [GoalStatus.ACTIVE, GoalStatus.PENDING]:
                rospy.logwarn("[%s] %s - Grasp Action Failed", rospy.get_name(), self.__class__.__name__)
                self.publish_done_state()
                return 'aborted'
            # Wait for updated state
            result = self._check_pddl_status()
            if result is not None:
                return result
            rate.sleep()


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


class ResetAutoTriedState(PDDLSmachState):
    def __init__(self, *args, **kwargs):
        super(ResetAutoTriedState, self).__init__(*args, **kwargs)
        self.domain = kwargs['domain']
        self.problem = kwargs['problem']
        self.state_update_pub = rospy.Publisher('/pddl_tasks/state_updates', PDDLState, queue_size=3)

    def on_execute(self, ud):
        state_update = PDDLState()
        state_update.domain = self.domain
        state_update.problem = self.problem
        state_update.predicates = ['(NOT (AUTO-GRASP-DONE))']
        print "Publishing (AUTO-GRASP-DONE) update"
        self.state_update_pub.publish(state_update)
