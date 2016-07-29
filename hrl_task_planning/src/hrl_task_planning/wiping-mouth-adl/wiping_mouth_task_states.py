#!/usr/bin/env python

# pylint: disable=W0102
import rospy
from task_smacher import StartNewTaskState
from hrl_task_planning.msg import PDDLProblem

SPA = ["succeeded", "preempted", "aborted"]


def set_default_goal(domain, goal_list):
    rospy.set_param('/pddl_tasks/%s/default_goal' % domain, goal_list)


def get_action_state(domain, problem, action, args, init_state, goal_state):
    if action == 'FIND_TAG':
        set_default_goal('find_tag', ['(FOUND-TAG)'])
        problem_msg = PDDLProblem()
        problem_msg.domain = 'find_tag'
        problem_msg.name = problem
        return StartNewTaskState(problem_msg, domain=domain, problem=problem,
                                 action=action, action_args=args, init_state=init_state,
                                 goal_state=goal_state, outcomes=SPA)
    elif action == 'TRACK_TAG':
        set_default_goal('track_tag', ['(IS-TRACKING-TAG)'])
        problem_msg = PDDLProblem()
        problem_msg.domain = 'place'
        problem_msg.name = problem
        return StartNewTaskState(problem_msg, domain=domain, problem=problem,
                                 action=action, action_args=args, init_state=init_state,
                                 goal_state=goal_state, outcomes=SPA)
    elif action == 'CALL_BASE_SELECTION':
        set_default_goal('call_base_selection', ['(BASE-SELECTED)'])
        problem_msg = PDDLProblem()
        problem_msg.domain = 'place'
        problem_msg.name = problem
        return StartNewTaskState(problem_msg, domain=domain, problem=problem,
                                 action=action, action_args=args, init_state=init_state,
                                 goal_state=goal_state, outcomes=SPA)
    elif action == 'CONFIGURE_BED_ROBOT':
        set_default_goal('configure_bed_robot', ['(CONFIGURED BED)'['(CONFIGURED SPINE)']])
        problem_msg = PDDLProblem()
        problem_msg.domain = 'place'
        problem_msg.name = problem
        return StartNewTaskState(problem_msg, domain=domain, problem=problem,
                                 action=action, action_args=args, init_state=init_state,
                                 goal_state=goal_state, outcomes=SPA)
    elif action == 'MOVE_ROBOT':
        set_default_goal('move_robot', ['(BASE-REACHED)'])
        problem_msg = PDDLProblem()
        problem_msg.domain = 'place'
        problem_msg.name = problem
        return StartNewTaskState(problem_msg, domain=domain, problem=problem,
                                 action=action, action_args=args, init_state=init_state,
                                 goal_state=goal_state, outcomes=SPA)
    elif action == 'MOVE_ARM':
        set_default_goal('move_arm', ['(ARM-REACHED)'])
        problem_msg = PDDLProblem()
        problem_msg.domain = 'place'
        problem_msg.name = problem
        return StartNewTaskState(problem_msg, domain=domain, problem=problem,
                                 action=action, action_args=args, init_state=init_state,
                                 goal_state=goal_state, outcomes=SPA)
    elif action == 'DO_TASK':
        set_default_goal('do_task', ['(TASK-COMPLETED)'])
        problem_msg = PDDLProblem()
        problem_msg.domain = 'place'
        problem_msg.name = problem
        return StartNewTaskState(problem_msg, domain=domain, problem=problem,
                                 action=action, action_args=args, init_state=init_state,
                                 goal_state=goal_state, outcomes=SPA)

