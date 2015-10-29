# pylint: disable=F0401,E0611

import importlib
from threading import Thread

import rospy
import smach
import smach_ros

from hrl_task_planning.msg import PDDLProblem, PDDLState, PDDLSolution
from hrl_task_planning.srv import PreprocessProblem, PDDLPlanner
from hrl_task_planning.pddl_utils import PlanStep

SPA = ["succeeded", "preempted", "aborted"]


class TaskSmacher(object):
    def __init__(self):
        self.modules = {}
        self.preprocess_services = {}
        self.running_sm_threads = {}
        self.state_pub = rospy.Publisher('task_state', PDDLState, latch=True)
        self.solution_pub = rospy.Publisher('task_solution', PDDLSolution, latch=True)
        self.planner_service = rospy.ServiceProxy("/pddl_planner", PDDLPlanner)
        self.task_req_sub = rospy.Subscriber("perform_task", PDDLProblem, self.req_cb)
        rospy.loginfo("[%s] Ready", rospy.get_name())

    def req_cb(self, req):
        print "Received Request:",  req
#        req.name = req.name.split('+')[0] # Throw out anything after the + sign
        # Make sure we have state machine definitions for this domain
        try:
            self.modules[req.domain] = importlib.import_module("hrl_task_planning.%s_states" % req.domain)
        except ImportError:
            rospy.logerr("[%s] Cannot load State Machine data for task: %s", rospy.get_name(), req.domain)
            return

        # If we don't have a preprocessor service set up for this domain, do so now
        if req.domain not in self.preprocess_services:
            self.preprocess_services[req.domain] = rospy.ServiceProxy("/preprocess_problem/%s" % req.domain, PreprocessProblem)

        # Try to preprocess the given problem (fills out current state, known objects, defaults goals as applicable)
        try:
            full_problem = self.preprocess_services[req.domain].call(req).problem
        except rospy.ServiceException as e:
            rospy.logerr("[%s] Service error when preprocesssing problem %s in domain %s: %s",
                         rospy.get_name(), req.name, req.domain, e.message)
            return

        # Get a planned solution based on the full problem
        try:
            solution = self.planner_service.call(full_problem)
        except rospy.ServiceException as e:
            rospy.logerr("[%s] Error when planning solution to problem %s in domain %s: %s",
                         rospy.get_name(), req.name, req.domain, e.message)
            return

        # Verify solution
        if not solution.solved:
            rospy.logwarn("[%s] Planner could not find a solution to problem %s in domain %s: %s",
                          rospy.get_name(), req.name, req.domain, e.message)
            return

        if 'UNDO' in req.name.upper() and self.running_sm_threads[req.domain].is_alive():
            next_state_req = self.running_sm_threads[req.domain].request
        else:
            next_state_req = None

        state_machine = self.build_sm(solution, self.modules[req.domain].get_action_state, next_state_req)

        try:
            if self.running_sm_threads[req.domain].is_alive():
                self.running_sm_threads[req.domain].preempt()
                rospy.loginfo("[%s] Preempt requested. Waiting for State Machine for %s to finish.",
                              rospy.get_name(), self.running_sm_threads[req.domain].problem_name)
                self.running_sm_threads[req.domain].join()
        except KeyError:
            pass

        self.solution_pub.publish(PDDLSolution(req.name, req.domain, solution.solved, solution.steps, solution.states))
        self.running_sm_threads[req.domain] = StateMachineThread(state_machine, req)
        self.running_sm_threads[req.domain].start()

    def build_sm(self, solution, get_state_fn, next_task_request=None):
        plan = map(PlanStep.from_string, solution.steps)
        pddl_states = solution.states
        domain = solution.states[0].domain
        problem = solution.states[0].problem.split('+')[0]

        sm = smach.StateMachine(outcomes=SPA)
        sm_states = []
        for i, step in enumerate(plan):
            sm_states.append(("_PDDL_STATE_PUB+%d" % i, PDDLStatePublisherState(pddl_states[i], self.state_pub, outcomes=SPA)))
            sm_states.append((step.name + "+%d" % i, get_state_fn(step, domain, problem)))
        sm_states.append(("_PDDL_STATE_PUB+FINAL", PDDLStatePublisherState(pddl_states[-1], self.state_pub, outcomes=SPA)))
        if next_task_request is None:
            sm_states.append(("_CLEANUP", CleanupState(problem=problem, outcomes=SPA, input_keys=["problem_name"])))
        else:
            sm_states.append(("_NextTask", StartNewTaskState(next_task_request, outcomes=SPA)))  # Keep old info if we're continuing on with this task...
        with sm:
            try:
                for i, sm_state in enumerate(sm_states):
                    next_sm_state = sm_states[i + 1]
                    # print "State: %s --> Next State: %s" % (sm_state, next_sm_state)
                    sm.add(sm_state[0], sm_state[1], transitions={'succeeded': next_sm_state[0]})
            except IndexError:
                sm.add(sm_states[-1][0], sm_states[-1][1], transitions={'succeeded': 'succeeded'})
        return sm


class StateMachineThread(Thread):
    def __init__(self, state_machine, request, *args, **kwargs):
        super(StateMachineThread, self).__init__(*args, **kwargs)
        self.state_machine = state_machine
        self.problem_name = request.name
        self.request = request
        self.sis = smach_ros.IntrospectionServer('smach_introspection', state_machine, self.problem_name)
        self.daemon = True

    def run(self):
        rospy.loginfo("[%s] Starting State Machine for %s", rospy.get_name(), self.problem_name)
        self.sis.start()
        self.state_machine.execute()
        self.sis.stop()

    def preempt(self):
        return self.state_machine.request_preempt()


class CleanupState(smach.State):
    def __init__(self, problem, *args, **kwargs):
        super(CleanupState, self).__init__(*args, **kwargs)
        self.problem = problem

    def execute(self, ud):
        if self.preempt_requested():
            self.service_preempt()
            return 'preempted'
        rospy.delete_param(self.problem)
        return 'succeeded'


class PDDLStatePublisherState(smach.State):
    def __init__(self, state, publisher, *args, **kwargs):
        super(PDDLStatePublisherState, self).__init__(*args, **kwargs)
        self.state = state
        self.publisher = publisher

    def execute(self, ud):
        self.publisher.publish(self.state)
        if self.preempt_requested():
            self.service_preempt()
            return 'preempted'
        return 'succeeded'


class StartNewTaskState(smach.State):
    def __init__(self, problem_msg, *args, **kwargs):
        super(StartNewTaskState, self).__init__(*args, **kwargs)
        self.request = problem_msg
        self.problem_pub = rospy.Publisher("perform_task", PDDLProblem)
        rospy.sleep(1)  # make sure subscribers can connect...

    def execute(self, ud):
        self.problem_pub.publish(self.request)
        if self.preempt_requested():
            self.service_preempt()
            return 'preempted'
        return 'succeeded'


def main():
    rospy.init_node("move_object_task_manager")
    task_smacher = TaskSmacher()
    rospy.spin()
