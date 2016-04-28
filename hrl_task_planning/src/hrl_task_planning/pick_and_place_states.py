#!/usr/bin/env python

# pylint: disable=W0102
from task_smacher import StartNewTaskState
from hrl_task_planning.msg import PDDLProblem

SPA = ["succeeded", "preempted", "aborted"]


def get_action_state(domain, problem, action, args, init_state, goal_state):
    if action == 'PICK':
        problem_msg = PDDLProblem()
        problem_msg.domain = 'pick'
        problem_msg.problem = problem
        return StartNewTaskState(problem_msg, domain=domain, problem=problem,
                                 action=action, action_args=args, init_state=init_state,
                                 goal_state=goal_state, outcomes=SPA)
    elif action == 'PLACE':
        problem_msg = PDDLProblem()
        problem_msg.domain = 'place'
        problem_msg.problem = problem
        return StartNewTaskState(problem_msg, domain=domain, problem=problem,
                                 action=action, action_args=args, init_state=init_state,
                                 goal_state=goal_state, outcomes=SPA)
