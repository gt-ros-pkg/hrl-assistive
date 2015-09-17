import rospy
from std_msgs.msg import String, Bool
import smach
import smach_ros

from hrl_task_planning.msg import PDDLProblem, PDDLSolution
from pddl_utils import PDDLPredicate, PDDLObject, PDDLPlanStep

SPA = ["succeeded", "preempted", "aborted"]


class ManualMoveObjectManager(object):
    task_name = "move-object"

    def __init__(self):
        self.task_req_sub = rospy.Subscriber("task_planning/request", String, self.req_cb)
        self.task_problem_pub = rospy.Publisher("/task_planner/problem", PDDLProblem)
        self.task_state = {'empty': [],
                           'can-grasp': [["right-gripper", "object-to-move"],
                                         ["left-gripper", "object-to-move"]]}
        self.solutions = {}
        self.problem_count = 0
        self.task_solution_sub = rospy.Subscriber("/task_planner/solution", PDDLSolution, self.solution_cb)
        self.l_gripper_grasp_state_sub = rospy.Subscriber("/grasping/left_gripper", Bool, self.grasp_state_cb, "left-gripper")
        self.r_gripper_grasp_state_sub = rospy.Subscriber("/grasping/right_gripper", Bool, self.grasp_state_cb, "right-gripper")
        rospy.loginfo("[%s] Ready" % rospy.get_name())

    def grasp_state_cb(self, msg, gripper):
        already_empty_list = [i for i, list_ in enumerate(self.task_state['empty']) if gripper == list_[0]]
        if not msg.data:
            if not already_empty_list:  # (Hand is empty, not known to be so)
                self.task_state["empty"].append([gripper])
        else:
            if already_empty_list:  # (Hand full, known to be empty)
                for item in already_empty_list:
                    self.task_state["empty"].pop(item)

    def solution_cb(self, sol):
        self.solutions[sol.problem] = [PDDLPlanStep.from_string(act) for act in sol.actions]
        if not solution.solved:

        rospy.loginfo("[%s] Received Plan:\n %s" % (rospy.get_name(), '\n'.join(map(str, self.solutions[sol.problem]))))
        plan = map(PDDLPlanStep.from_string, sol.actions)
        sm = build_sm(plan)
        sis = smach_ros.IntrospectionServer('smach_introspection', sm, sol.problem)
        sis.start()
        sm.execute()
        sis.stop()

    def req_cb(self, req):
        if req.data != self.task_name:
            return
        problem = PDDLProblem()
        self.problem_count += 1
        problem.name = "manual-move-object-"+str(self.problem_count)
        problem.domain = "move-object"
        problem.objects = [str(PDDLObject("object-to-move", "object"))]
        problem.init = []
        for act, arg_sets in self.task_state.iteritems():
            for args in arg_sets:
                problem.init.append(str(PDDLPredicate(act, args)))
        problem.goal = []
        problem.goal.append(str(PDDLPredicate('at', ["object-to-move", "goal"])))
        self.task_problem_pub.publish(problem)


def build_sm(plan):
    sm = smach.StateMachine(outcomes=SPA)
    states = []
    for step in plan:
        states.append(get_action_state(step.name))
    with sm:
        try:
            for i, state in enumerate(states):
                next_state = states[i + 1]
                print "State: %s --> Next State: %s" % (state, next_state)
                sm.add(state[0], state[1], transitions={'succeeded': next_state[0]})
        except IndexError:
            sm.add(states[-1][0], states[-1][1])
    return sm


class WaitForGraspState(smach.State):
    def __init__(self, topic, side,  outcomes=SPA, input_keys=[], output_keys=['grasp_side']):
        super(WaitForGraspState, self).__init__(outcomes=outcomes, input_keys=input_keys, output_keys=output_keys)
        self.side = side
        self.running = False
        self.grasped = False
        self.sub = rospy.Subscriber(topic, Bool, self.msg_cb)

    def msg_cb(self, msg):
        if self.running and msg.data:
            print "%s Gripper Grasped!" % self.side.capitalize()
            self.grasped = True

    def execute(self, ud):
        print "Running Wait for Grasp %s" % self.side
        self.running = True
        while not self.preempt_requested() and not self.grasped:
            rospy.sleep(0.05)
        if self.preempt_requested():
            self.service_preempt()
            return "preempted"
        if self.grasped:
            ud['grasp_side'] = self.side
            print ud
            return "succeeded"
        return "aborted"


class WaitForReleaseState(smach.State):
    def __init__(self, topic, side,  outcomes=SPA, input_keys=['grasp_side'], output_keys=[]):
        super(WaitForReleaseState, self).__init__(outcomes=outcomes, input_keys=input_keys, output_keys=output_keys)
        self.side = side
        self.running = False
        self.released = False
        self.sub = rospy.Subscriber(topic, Bool, self.msg_cb)

    def msg_cb(self, msg):
        if self.running and not msg.data:
            print "%s Gripper Released!" % self.side
            self.released = True

    def execute(self, ud):
        print ud
        if self.side == ud.grasp_side:
            print "Running Wait for Release %s" % self.side
            self.running = True
            while not self.preempt_requested() and not self.released:
                rospy.sleep(0.05)
            if self.preempt_requested():
                self.service_preempt()
                return "preempted"
            if self.released:
                return "succeeded"
        return "aborted"


def get_action_state(plan_step):
    if plan_step == 'PICK':

        def outcome_cb(outcomes):
            if 'aborted' in outcomes.itervalues():
                return 'aborted'
            if 'succeeded' in outcomes.itervalues():
                return 'succeeded'
            return 'preempted'

        concurrence = smach.Concurrence(outcomes=SPA,
                                        default_outcome='aborted',
                                        output_keys=['grasp_side'],
                                        child_termination_cb=lambda so: True,
                                        outcome_cb=outcome_cb
                                        )

        grasp_state_left = WaitForGraspState("/grasping/left_gripper", side="left", outcomes=SPA, output_keys=['grasp_side'])
        grasp_state_right = WaitForGraspState("/grasping/right_gripper", side="right", outcomes=SPA, output_keys=['grasp_side'])
        with concurrence:
            concurrence.add('grasp-left', grasp_state_left)
            concurrence.add('grasp-right', grasp_state_right)
        return ("pick", concurrence)

    elif plan_step == 'PLACE':
        concurrence = smach.Concurrence(outcomes=SPA,
                                        default_outcome='aborted',
                                        input_keys=['grasp_side'],
                                        outcome_map={'succeeded': {'release-right': 'succeeded'},
                                                     'succeeded': {'release-left': 'succeeded'},
                                                     'aborted': {'release-left': 'aborted', 'release-right': 'aborted'},
                                                     'preempted': {'release-left': 'preempted'},
                                                     'preempted': {'release-right': 'preempted'}
                                                     }
                                        )

        release_state_left = WaitForReleaseState("/grasping/left_gripper", side="left", outcomes=SPA, input_keys=['grasp_side'])
        release_state_right = WaitForReleaseState("/grasping/right_gripper", side="right", outcomes=SPA, input_keys=['grasp_side'])
        with concurrence:
            concurrence.add('release-left', release_state_left)
            concurrence.add('release-right', release_state_right)
        return ("place", concurrence)


def main():
    rospy.init_node("move_object_task_manager")
    manager = ManualMoveObjectManager()
    rospy.spin()
