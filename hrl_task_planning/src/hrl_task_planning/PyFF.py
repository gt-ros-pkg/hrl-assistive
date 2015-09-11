#!/usr/bin/env python

from tempfile import NamedTemporaryFile
from subprocess import check_output, CalledProcessError
from os import remove
import sys

import rospy

from pddl_utils import PDDLProblem, PDDLPredicate, PDDLObject, PDDLPlanStep
import hrl_task_planning.msg as planner_msgs


class FF(object):
    """ A solver instance based on an FF executable. """
    def __init__(self, ff_executable='./ff'):
        self.ff_executable = ff_executable

    def _parse_solution(self, soln_txt):
        """ Extract list of solution steps from FF output. """
        sol = []
        soln_txt = soln_txt.split('step')[1].strip()
        soln_txt = soln_txt.split('time spent')[0].strip()
        steps = [step.strip() for step in soln_txt.splitlines()]
        for step in steps:
            args = step.split(':')[1].lstrip().split()
            act = args.pop(0)  # Remove action, leave all args
            sol.append(PDDLPlanStep(act, args))
        return sol

    def solve(self, problem, domain_file):
        """ Create a temporary problem file and call FF to solve. """
        with NamedTemporaryFile() as problem_file:
            problem.to_file(problem_file.name)
            print ' '.join([self.ff_executable, '-o', domain_file, '-f', problem_file.name])
            print problem.to_string()
            try:
                soln_txt = check_output([self.ff_executable, '-o', domain_file, '-f', problem_file.name])
                print soln_txt
            except CalledProcessError as cpe:
                if "goal can be simplified to TRUE." in cpe.output:
                    return []
                else:
                    rospy.logwarn("[%s] FF Could not find a solution to problem: %s"
                                  % (rospy.get_name(), problem.name))
                    return False
            finally:
                # clean up the soln file produced by ff (avoids large dumps of files in /tmp)
                try:
                    remove('.'.join([problem_file.name, 'soln']))
                except OSError as ose:
                    if ose.errno != 2:
                        raise ose
        return self._parse_solution(soln_txt)


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
            #const_obj_param = '/'.join([req.domain, 'objects'])
            #objects = rospy.get_param(''.join(['~', const_obj_param]))
            const_preds_param = '/'.join([req.domain, 'predicates'])
            init = rospy.get_param(''.join(['~', const_preds_param]), [])
            if not req.goal:
                default_goal_param = '/'.join([req.domain, 'default_goal'])
                req.goal = rospy.get_param(''.join(['~', default_goal_param]), [])
        except KeyError as e:
            rospy.logerr("[%s] Could not find parameter: %s" % (rospy.get_name(), e.message))
            return []
        #objects.extend(req.objects)
        #objects = map(PDDLObject.from_string, objects)
        objects = map(PDDLObject.from_string, req.objects)
        init.extend(req.init)
        init = map(PDDLPredicate.from_string, init)
        goal = map(PDDLPredicate.from_string, req.goal)
        problem = PDDLProblem(req.name, req.domain, objects, init, goal)
        solution = self.planner.solve(problem, domain_file)
        sol_list = map(str, solution)
        self.solution_pub.publish(sol_list)


def main():
    rospy.init_node("ff_task_planner")
    args = rospy.myargv(argv=sys.argv)
    planner = FF(args[1])  # Only arg is the ff executable file
    planner_node = TaskPlannerNode(planner)
    rospy.spin()
