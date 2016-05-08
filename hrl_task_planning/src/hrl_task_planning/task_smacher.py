# pylint: disable=F0401,E0611

import importlib
from threading import Thread
import copy

import rospy
import smach
from std_msgs.msg import String

from hrl_task_planning.msg import PDDLProblem, PDDLState, PDDLSolution, DomainList, PDDLPlanStep
from hrl_task_planning.srv import PDDLPlanner, PreemptTask
from hrl_task_planning.pddl_utils import PlanStep, State, GoalState, Predicate


class TaskSmacher(object):
    def __init__(self):
        self._sm_threads = []
        self.last_running_set = set([])
        self.active_domains_pub = rospy.Publisher('pddl_tasks/active_domains', DomainList, queue_size=10, latch=True)
        self.active_problem = rospy.Publisher('pddl_tasks/current_problem', String, queue_size=10, latch=True)
        self.preempt_service = rospy.Service("preempt_pddl_task", PreemptTask, self.cancel_service_cb)
        self.task_req_sub = rospy.Subscriber("perform_task", PDDLProblem, self.req_cb)
        self.active_problem.publish('')  # Initialize to empty string
        self.active_domains_pub.publish(DomainList([]))  # Initialize to empty list
        rospy.loginfo("[%s] Ready", rospy.get_name())

    def req_cb(self, req):
        # Find any running tasks for this domain, kill them and their peers
        running = [thread for thread in self._sm_threads if thread.is_alive()]
        for thread in running:
            if thread.domain == req.domain:
                if thread.problem_name == req.name:
                    thread.set_new_goal(req)
                    return
                else:
                    thread.abort()
        new_thread = self.create_thread(req)
        new_thread.start()
        self.active_problem.publish(thread.problem_name)

    def create_thread(self, problem_msg):
        # If we're given a subgoal, prep a second thread for going to the default goal to call once we get to the subgoal
        thread = PDDLTaskThread()
        thread.set_problem(problem_msg)
        self._sm_threads.append(thread)
        return thread

    def cancel_service_cb(self, preempt_request):
        self.abort_problem_threads(preempt_request.problem_name)
        return True

    def abort_problem_threads(self, problem_name):
        for thread in self._sm_threads:
            if thread.problem_name == problem_name:
                if thread.is_alive():
                    thread.abort()
                    rospy.loginfo("Abort Problem %s -- Killing %s thread", thread.problem_name, thread.domain)
                    thread.join()  # DANGEROUS BLOCKING CALL HERE
                    rospy.loginfo("Abort Problem %s -- Killed %s thread", thread.problem_name, thread.domain)
        self._sm_threads = [thread for thread in self._sm_threads if thread.problem_name != problem_name]  # cut out now-preempted thread objects

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
            if not running_set:
                self.active_problem.publish('')  # If all domains ended, set problem to empty

    def spin(self, hz=25):
        rate = rospy.Rate(hz)
        while not rospy.is_shutdown():
            self.check_threads()
            rate.sleep()
        # All sub-threads are deamons, should die with this main thread


class PDDLTaskThread(Thread):
    def __init__(self, *args, **kwargs):
        super(PDDLTaskThread, self).__init__(*args, **kwargs)
        self.problem_msg = None
        self.abort_requested = False
        self.traversed_solution = {'steps': [], 'states': []}
        self.solution_history = []
        self.constant_predicates = rospy.get_param('/pddl_tasks/%s/constant_predicates' % self.domain, [])
        self.solution_pub = rospy.Publisher('/pddl_tasks/%s/solution' % self.domain, PDDLSolution, queue_size=10, latch=True)
        self.action_sub = rospy.Subscriber('/pddl_tasks/%s/current_action' % self.domain, PDDLPlanStep, self.current_action_cb)
        # TODO: Catch close signal, publish empty action before stopping...
        self.planner_service = rospy.ServiceProxy("/pddl_planner", PDDLPlanner)
        self.domain_smach_states = importlib.import_module("hrl_task_planning.%s_states" % self.domain)
        self.domain_state = None
        self.domain_status_sub = rospy.Subscriber('/pddl_tasks/%s/state' % self.domain, PDDLState, self.domain_state_cb)
        self.state_machine = None
        self.daemon = True

    def set_problem(self, problem_msg):
        self.problem_msg = problem_msg
        self.problem_name = problem_msg.name
        self.domain = problem_msg.domain

    def current_action_cb(self, plan_step_msg):
        sol = self.solution_history[-1]
        for i, action in enumerate(sol.steps):
            if plan_step_msg.action in action and all([arg in action for arg in plan_step_msg.args]):
                self.traversed_solution['steps'].append(action)
                self.traversed_solution['states'].append(sol.states[i])

    def domain_state_cb(self, pddl_state_msg):
        self.domain_state = pddl_state_msg.predicates

    def abort(self):
        self.abort_requested = True
        if self.state_machine is not None:
            self.state_machine.request_preempt()

    def set_new_goal(self, problem_msg):
        self.set_problem(problem_msg)
        if self.state_machine is not None:
            self.state_machine.request_preempt()

    def run(self):
        # Wait for domain state to become available
        rospy.loginfo("[%s] Starting Thread for %s domain in problem: %s", rospy.get_name(), self.domain, self.problem_name)
        while self.domain_state is None:
            rospy.sleep(0.5)
        # If no goal is specified, use the default problem goal
        if not self.problem_msg.goal:
            self.problem_msg.goal = rospy.get_param('/pddl_tasks/%s/default_goal' % self.domain)
        # Plan + execute (and re-plan and re-execute) until task complete to default goal
        result = None
        while not rospy.is_shutdown():
            # For the current problem get initial state and
            # [self.problem_msg.init.append(pred) for pred in self.constant_predicates if pred not in self.problem_msg.init]
            self.problem_msg.init = self.domain_state
            # Get solution from planner
            try:
                solution = self.planner_service.call(self.problem_msg)
                self.solution_history.append(solution)
                steps = copy.copy(self.traversed_solution['steps'])
                steps.extend(solution.steps)
                states = copy.copy(self.traversed_solution['states'])
                states.extend(solution.states)
                sol_msg = PDDLSolution()
                sol_msg.domain = self.domain
                sol_msg.problem = self.problem_name
                sol_msg.solved = solution.solved
                sol_msg.actions = steps
                sol_msg.states = states
                self.solution_pub.publish(sol_msg)
                print "Solution:\n", solution.steps
                if solution.solved:
                    if not solution.steps:  # Already solved, no action retquired
                        rospy.loginfo("[%s] %s domain already in goal state, no action required.", rospy.get_name(), self.domain)
                        result = 'succeeded'
                else:
                    rospy.loginfo("[%s] Planner could not find a solution to problem %s in %s domain.",
                                  rospy.get_name(), self.problem_name, self.domain)
                    result = 'aborted'
            except rospy.ServiceException as e:
                rospy.logerr("[%s] Error when planning solution to problem %s in %s domain: %s",
                             rospy.get_name(), self.problem_name, self.domain, e.message)
                result = 'aborted'

            # TODO: Check for irreversible actions and add a confirmation state.
            # Build smach state machine based on domain data
            steps = map(PlanStep.from_string, solution.steps)
            state_preds = [state.predicates for state in solution.states]
            states = [State(preds) for preds in [map(Predicate.from_string, preds) for preds in state_preds]]
            n_steps = len(steps)
            transitions = {'preempted': 'preempted', 'aborted': 'aborted'}
            self.state_machine = smach.StateMachine(outcomes=["succeeded", "preempted", "aborted"])
            with self.state_machine:
                for i in range(n_steps):
                    smach_state = self.domain_smach_states.get_action_state(self.domain, self.problem_name,
                                                                            steps[i].name, steps[i].args,
                                                                            states[i], states[i+1])
                    transitions['succeeded'] = 'succeeded' if (i == n_steps-1) else '%d-%s' % (i+1, steps[i+1].name)
                    self.state_machine.add('%d-%s' % (i, steps[i].name), smach_state, transitions=transitions)

            # Run the SMACH State-machine
            result = self.state_machine.execute()
            print "Exceution of %s SMACH plan: " % self.domain, result

            # Evaluate results, break if completely succeeded or aborted
            if result == 'aborted' or (result == 'preempted' and self.abort_requested):
                break
            if result == 'preempted':
                continue  # Interrupted, but not aborted, so probably have new goal. Re-try.
            if result == 'succeeded':
                default_goal_now = rospy.get_param('/pddl_tasks/%s/default_goal' % self.domain)
                if (self.problem_msg.goal == default_goal_now):
                    break
        print "Domain %s: %s" % (self.domain, result)


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
        self.action_pub = rospy.Publisher('/pddl_tasks/%s/current_action' % self.domain, PDDLPlanStep, queue_size=10, latch=True)
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
        while self.current_state is None:
            rospy.loginfo("State %s waiting for current state", self.action)
            rospy.sleep(1)
        # print "Current State: ", str(self.current_state)
        while not rospy.is_shutdown():
            if self.preempt_requested():
                rospy.loginfo("[%s] Preempted requested for %s(%s).", rospy.get_name(), self.action, ' '.join(self.action_args))
                self.service_preempt()
                return 'preempted'
            if self.goal_state.is_satisfied(self.current_state):
                return 'succeeded'
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


class StartNewTaskState(PDDLSmachState):
    def __init__(self, problem_msg, *args, **kwargs):
        super(StartNewTaskState, self).__init__(*args, **kwargs)
        self.request = problem_msg
        self.problem_pub = rospy.Publisher("perform_task", PDDLProblem, queue_size=10)
        rospy.sleep(1)  # make sure subscribers can connect...

    def on_execute(self, ud):
        if self.preempt_requested():
            self.service_preempt()
            return 'preempted'
        self.problem_pub.publish(self.request)


def main():
    rospy.init_node("move_object_task_manager")
    task_smacher = TaskSmacher()
    task_smacher.spin()
