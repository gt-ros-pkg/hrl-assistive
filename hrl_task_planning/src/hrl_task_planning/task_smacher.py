# pylint: disable=F0401,E0611

import importlib
from threading import Thread
import copy

import rospy
import smach

from hrl_task_planning.msg import PDDLProblem, PDDLState, PDDLSolution, DomainList, PDDLPlanStep
from hrl_task_planning.srv import PDDLPlanner, PreemptTask
from hrl_task_planning.pddl_utils import PlanStep, State, GoalState, Predicate

SPA = ["succeeded", "preempted", "aborted"]


class TaskSmacher(object):
    def __init__(self):
        self._sm_threads = []
        self.last_running_set = set([])
        self.active_domains_pub = rospy.Publisher('pddl_tasks/active_domains', DomainList, latch=True)
        self.preempt_service = rospy.Service("preempt_pddl_task", PreemptTask, self.preempt_service_cb)
        self.task_req_sub = rospy.Subscriber("perform_task", PDDLProblem, self.req_cb)
        rospy.loginfo("[%s] Ready", rospy.get_name())

    def req_cb(self, req):
        # Find any running tasks for this domain, kill them and their peers
        running = [thread for thread in self._sm_threads if thread.is_alive()]
        kill_ids = set([thread.problem_name for thread in running])
        for problem_name in kill_ids:
            self.preempt_threads(problem_name)
        thread = self.create_thread(req)
        thread.start()

    def preempt_service_cb(self, preempt_request):
        self.preempt_threads(preempt_request.problem_name)
        return True

    def preempt_threads(self, problem_name):
        for thread in self._sm_threads:
            if thread.problem_name == problem_name:
                if thread.is_alive():
                    thread.preempt()
                    rospy.loginfo("Killing %s thread", thread.problem_name)
                    thread.join()  # DANGEROUS BLOCKING CALL HERE
                    rospy.loginfo("Killed %s thread", thread.problem_name)
        self._sm_threads = [thread for thread in self._sm_threads if thread.problem_name != problem_name]  # cut out now-preempted thread objects

    def create_thread(self, problem_msg):
        # If we're given a subgoal, prep a second thread for going to the default goal to call once we get to the subgoal
        default_goal_thread = None
        if problem_msg.goal:
            default_goal_problem = copy.copy(problem_msg)
            default_goal_problem.init = problem_msg.goal
            default_goal_problem.goal = []
            default_goal_thread = PDDLTaskThread(default_goal_problem)
            self._sm_threads.append(default_goal_thread)
        thread = PDDLTaskThread(problem_msg, next_thread=default_goal_thread)
        self._sm_threads.append(thread)
        return thread

    def check_threads(self):
        """ Check for stopped threads, remove any, and re-publish updates list if any changes occur """
        unstarted = []
        running = []
        finished = []
        for thread in self._sm_threads:
            if thread.is_alive():
                running.append(thread.domain)
                continue
            try:
                thread.join(0)
                finished.append(thread.domain)
            except RuntimeError:
                unstarted.append(thread.domain)
        running_set = set(running)
        if running_set != self.last_running_set:
            self.active_domains_pub.publish(running_set)
            self.last_running_set = running_set

    def spin(self, hz=25):
        rate = rospy.Rate(hz)
        while not rospy.is_shutdown():
            self.check_threads()
            rate.sleep()
        # All sub-threads are deamons, should die with this main thread


class PDDLTaskThread(Thread):
    def __init__(self, problem_msg, next_thread=None, *args, **kwargs):
        super(PDDLTaskThread, self).__init__(*args, **kwargs)
        self.problem_msg = problem_msg
        self.next_thread = next_thread
        self.problem_name = problem_msg.name
        self.domain = problem_msg.domain
        self.result = None
        self.constant_predicates = rospy.get_param('/pddl_tasks/%s/constant_predicates' % self.domain, [])
        self.default_goal = rospy.get_param('/pddl_tasks/%s/default_goal' % self.domain)
        self.solution_pub = rospy.Publisher('task_solution', PDDLSolution, latch=True)
        self.action_pub = rospy.Publisher('/pddl_tasks/%s/current_action' % self.domain, PDDLPlanStep, latch=True)
        self.planner_service = rospy.ServiceProxy("/pddl_planner", PDDLPlanner)
        self.domain_smach_states = importlib.import_module("hrl_task_planning.%s_states" % self.domain)
        self.domain_state = None
        self.domain_status_sub = rospy.Subscriber('/pddl_tasks/%s/state' % self.domain, PDDLState, self.domain_state_cb)
        self.state_machine = None
        self.daemon = True

    def domain_state_cb(self, pddl_state_msg):
        self.domain_state = pddl_state_msg.predicates

    def run(self):
        # Build out problem (init, goal), check for sub-goals, plan, check for irrecoverable actions, publish plan, compose sm, run sm
        if not self.problem_msg.goal:
            self.problem_msg.goal = self.default_goal
        while self.result != 'succeeded':
            # For the current problem get initial state and
            self.problem_msg.init.extend(self.constant_predicates)
            while self.domain_state is None:
                rospy.loginfo("Waiting for state of %s domain.", self.domain)
                rospy.sleep(1)
            print "Extending initial state with ", self.domain_state
            self.problem_msg.init.extend(self.domain_state)

            # Get solution from planner
            try:
                solution = self.planner_service.call(self.problem_msg)
                sol_msg = PDDLSolution()
                sol_msg.domain = self.domain
                sol_msg.problem = self.problem_name
                sol_msg.solved = solution.solved
                sol_msg.actions = solution.steps
                sol_msg.states = solution.states
                self.solution_pub.publish(sol_msg)
                print "Solution:\n", solution
                if solution.solved:
                    if not solution.steps:  # Already solved, no action retquired
                        rospy.loginfo("[%s] %s domain already in goal state, no action required.", rospy.get_name(), self.domain)
                        break
                else:
                    rospy.loginfo("[%s] Planner could not find a solution to problem %s in %s domain.",
                                  rospy.get_name(), self.problem_name, self.domain)
                    return
            except rospy.ServiceException as e:
                rospy.logerr("[%s] Error when planning solution to problem %s in %s domain: %s",
                             rospy.get_name(), self.problem_name, self.domain, e.message)
                return

            steps = map(PlanStep.from_string, solution.steps)
            state_preds = [state.predicates for state in solution.states]
            states = [State(preds) for preds in [map(Predicate.from_string, preds) for preds in state_preds]]
            n_steps = len(steps)
            self.state_machine = smach.StateMachine(outcomes=SPA)
            with self.state_machine:
                for i in range(n_steps):  # TODO: Catch index errors at end of list
                    smach_state = self.domain_smach_states.get_action_state(self.domain, self.problem_name,
                                                                            steps[i].name, steps[i].args,
                                                                            states[i], states[i+1])
                    if i == n_steps-1:
                        transitions = {'preempted': 'preempted',
                                       'aborted': 'aborted',
                                       'succeeded': 'succeeded'}
                    else:
                        transitions = {'preempted': 'preempted',
                                       'aborted': 'aborted',
                                       'succeeded': '%d-%s' % (i+1, steps[i+1].name)}
                    self.state_machine.add('%d-%s' % (i, steps[i].name), smach_state, transitions=transitions)
            try:
                self.result = self.state_machine.execute()
                print "Result: ", self.result
            except Exception as e:
                raise e
            if self.result == 'preempted':
                if self.next_thread is not None:
                    self.next_thread.preempt()
                break

        # Publish empty action to current action topic (since we're done)
        plan_step_msg = PDDLPlanStep()
        plan_step_msg.domain = self.domain
        plan_step_msg.problem = self.problem
        self.action_pub.publish(plan_step_msg)

        if self.next_thread is not None:
            self.next_thread.start()

    def preempt(self):
        if self.state_machine is not None:
            self.state_machine.request_preempt()


class PDDLSmachState(smach.State):
    def __init__(self, domain, problem, action, action_args, init_state, goal_state, outcomes=[], *args, **kwargs):
        super(PDDLSmachState, self).__init__(outcomes=outcomes, *args, **kwargs)
        self.domain = domain
        self.problem = problem
        self.action = action
        self.action_args = action_args
        self.init_state = init_state
        self.goal_state = GoalState(goal_state.predicates)
        self.state_delta = self.init_state.difference(self.goal_state)
        self.action_pub = rospy.Publisher('/pddl_tasks/%s/current_action' % self.domain, PDDLPlanStep, latch=True)
        self.current_state = None
        self.domain_state_sub = rospy.Subscriber("/pddl_tasks/%s/state" % self.domain, PDDLState, self.domain_state_cb)

    def domain_state_cb(self, state_msg):
        self.current_state = State(map(Predicate.from_string, state_msg.predicates))

    def on_execute(self, ud):
        """ Override to create task-specific functionality before waiting for state update in main execute."""
        pass

    def execute(self, ud):
        # Watch for task state to match goal state, then return successful
        plan_step_msg = PDDLPlanStep()
        plan_step_msg.domain = self.domain
        plan_step_msg.problem = self.problem
        plan_step_msg.action = self.action
        plan_step_msg.args = self.action_args
        self.action_pub.publish(plan_step_msg)
        self.on_execute(ud)
        rate = rospy.Rate(20)
        print "Starting PDDLSmachState: %s" % self.action
        print "Initial State: ", str(self.init_state)
        print "Goal State: ", str(self.goal_state)
        while self.current_state is None:
            rospy.loginfo("State %s waiting for current state", self.action)
            rospy.sleep(1)
        print "Current State: ", str(self.current_state)
        while not rospy.is_shutdown():
            if self.preempt_requested():
                rospy.loginfo("[%s] Preempted requested for %s(%s).", rospy.get_name(), self.action, ' '.join(self.action_args))
                self.service_preempt()
                return 'preempted'
            if self.goal_state.is_satisfied(self.current_state):
                return 'success'
            progress = self.init_state.difference(self.current_state)
            for pred in progress:
                if pred not in self.state_delta:
                    return 'aborted'
            rate.sleep()
        return 'preempted'


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
    task_smacher.spin()
