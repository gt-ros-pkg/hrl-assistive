#!/usr/bin/env python

import sys

import rospy

import hrl_task_planning.pddl_utils as pddl
from hrl_task_planning.srv import PDDLPlanner, PDDLPlannerResponse
from hrl_task_planning.msg import PDDLState


class TaskPlannerNode(object):
    """ A ROS node wrapping an instance of a task planner. """
    def __init__(self, planner):
        self.planner = planner
        self.planner_service = rospy.Service('pddl_planner', PDDLPlanner, self.plan_req_cb)
        rospy.loginfo("[%s] Ready", rospy.get_name())

    def plan_req_cb(self, req):
        try:
            domain_file_param = '/'.join([req.problem.domain, 'domain_file'])
            domain_file = rospy.get_param(domain_file_param)
        except KeyError as e:
            rospy.logerr("[%s] Could not find parameter: %s", rospy.get_name(), e.message)
            return (False, [], [])
        except Exception as e:
            raise rospy.ServiceException(e.message)
        rospy.loginfo("[%s] Planner Solving problem:\n%s", rospy.get_name(), req.problem)
        # Create PDDL Domain object from domain file
        domain = pddl.Domain.from_file(domain_file)
        # Create PDDL Problem object from incoming message
        problem = pddl.Problem.from_msg(req.problem)
        # Define the planning situation (domain and problem)
        situation = pddl.Situation(domain, problem)
        result = PDDLPlannerResponse()
        try:
            situation.solution = self.planner.solve(domain, problem)
            result.solved = True
            result.steps = map(str, situation.solution)
            result.states = [PDDLState(problem.name, map(str, state)) for state in situation.get_plan_intermediary_states()]
        except pddl.PlanningError:
            result.solved = False
        except Exception as e:
            raise rospy.ServiceException(e.message)
        return result


def main():
    rospy.init_node("ff_task_planner")
    args = rospy.myargv(argv=sys.argv)
    planner = pddl.FF(args[1])  # Only arg is the ff executable file
    planner_node = TaskPlannerNode(planner)
    rospy.spin()
