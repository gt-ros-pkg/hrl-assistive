#!/usr/bin/env python

from tempfile import NamedTemporaryFile
from subprocess import check_output, CalledProcessError
from os import remove
import sys

import rospy

from pddl_utils import PDDLProblem, PDDLPredicate, PDDLObject, PDDLPlanStep
import hrl_task_planning.msg as planner_msgs



class TaskPlannerNode(object):
    """ A ROS node wrapping an instance of a task planner. """
    def __init__(self, planner=FF):
        self.planner = planner
        self.problem_sub = rospy.Subscriber('~problem', planner_msgs.PDDLProblem, self.plan_req_cb)
        self.solution_pub = rospy.Publisher('~solution', planner_msgs.PDDLSolution)
        rospy.loginfo("[%s] Ready" % rospy.get_name())

    def plan_req_cb(self, req):
        try:
            domain_file_param = '/'.join([req.domain, 'domain_file'])
            domain_file = rospy.get_param(''.join(['~', domain_file_param]))
            # const_obj_param = '/'.join([req.domain, 'objects'])
            # objects = rospy.get_param(''.join(['~', const_obj_param]))
            const_preds_param = '/'.join([req.domain, 'predicates'])
            init = rospy.get_param(''.join(['~', const_preds_param]), [])
            if not req.goal:
                default_goal_param = '/'.join([req.domain, 'default_goal'])
                req.goal = rospy.get_param(''.join(['~', default_goal_param]), [])
        except KeyError as e:
            rospy.logerr("[%s] Could not find parameter: %s" % (rospy.get_name(), e.message))
            return []
        # objects.extend(req.objects)
        # objects = map(PDDLObject.from_string, objects)
        objects = map(PDDLObject.from_string, req.objects)
        init.extend(req.init)
        init = map(PDDLPredicate.from_string, init)
        goal = map(PDDLPredicate.from_string, req.goal)
        problem = PDDLProblem(req.name, req.domain, objects, init, goal)
        solution = self.planner.solve(problem, domain_file)
        sol_msg = planner_msgs.PDDLSolution()
        sol_msg.problem = req.name
        sol_msg.solved = bool(solution)
        sol_msg.actions = map(str, solution)
        self.solution_pub.publish(sol_msg)


def main():
    rospy.init_node("ff_task_planner")
    args = rospy.myargv(argv=sys.argv)
    planner = FF(args[1])  # Only arg is the ff executable file
    planner_node = TaskPlannerNode(planner)
    rospy.spin()
