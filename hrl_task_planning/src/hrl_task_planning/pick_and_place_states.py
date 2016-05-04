#!/usr/bin/env python

# pylint: disable=W0102
import rospy
from task_smacher import StartNewTaskState
from hrl_task_planning.msg import PDDLProblem

SPA = ["succeeded", "preempted", "aborted"]


def set_default_goal(domain, goal_list):
    rospy.set_param('/pddl_tasks/%s/default_goal' % domain, goal_list)


def get_action_state(domain, problem, action, args, init_state, goal_state):
    if action == 'PICK':
        set_default_goal('pick', ['(GRASPING %s %s)' % (args[0], args[1])])
        problem_msg = PDDLProblem()
        problem_msg.domain = 'pick'
        problem_msg.name = problem
        return StartNewTaskState(problem_msg, domain=domain, problem=problem,
                                 action=action, action_args=args, init_state=init_state,
                                 goal_state=goal_state, outcomes=SPA)
    elif action == 'PLACE':
        set_default_goal('place', ['(PLACED %s)' % args[1]])
        problem_msg = PDDLProblem()
        problem_msg.domain = 'place'
        problem_msg.name = problem
        return StartNewTaskState(problem_msg, domain=domain, problem=problem,
                                 action=action, action_args=args, init_state=init_state,
                                 goal_state=goal_state, outcomes=SPA)
