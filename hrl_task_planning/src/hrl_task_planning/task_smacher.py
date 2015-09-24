import importlib

import rospy
from std_msgs.msg import String
import smach
import smach_ros

from hrl_task_planning.msg import PDDLProblem, PDDLSolution
from hrl_task_planning.srv import PreprocessProblemRequest, PDDLPlannerRequest
from hrl_task_planning.pddl_utils import PDDLPredicate, PDDLObject, PDDLPlanStep

SPA = ["succeeded", "preempted", "aborted"]


class TaskSmacher(object):
    def __init__(self):
        self.problem_count = 0
        self.modules = {}
        self.task_req_sub = rospy.Subscriber("task_planning/request", String, self.req_cb)
        self.task_solution_sub = rospy.Subscriber("~solution", PDDLSolution, self.solution_cb)
        rospy.loginfo("[%s] Ready", rospy.get_name())

    def solution_cb(self, sol):
        if not sol.solved:
            return False
        self.solutions[sol.problem] = [PDDLPlanStep.from_string(act) for act in sol.actions]

        rospy.loginfo("[%s] Received Plan:\n %s", rospy.get_name(), '\n'.join(map(str, self.solutions[sol.problem])))
        plan = map(PDDLPlanStep.from_string, sol.actions)
        sm = build_sm(plan)
        sis = smach_ros.IntrospectionServer('smach_introspection', sm, sol.problem)
        sis.start()
        sm.execute()
        sis.stop()

    def req_cb(self, req):
        try:
            self.modules[req.domain] = importlib.import_module("hrl_task_planning.%s_states" % req.domain)
        except ImportError:
            rospy.logerr("[%s] Cannot load State Machine data for task: %s", rospy.get_name(), req.domain)
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

    def build_sm(self, plan):
        sm = smach.StateMachine(outcomes=SPA)
        states = []
        for step in plan:
            states.append(self.modules[plan.domain].get_action_state(step.name))
        with sm:
            try:
                for i, state in enumerate(states):
                    next_state = states[i + 1]
                    print "State: %s --> Next State: %s" % (state, next_state)
                    # TODO: Add state publisher for undo here at each transition
                    sm.add(state[0], state[1], transitions={'succeeded': next_state[0]})
            except IndexError:
                sm.add(states[-1][0], states[-1][1])
        return sm


def main():
    rospy.init_node("move_object_task_manager")
    task_smacher = TaskSmacher()
    rospy.spin()
