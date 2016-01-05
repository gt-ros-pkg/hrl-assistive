#!/usr/bin/env python

import argparse
import sys

import rospy
from std_msgs.msg import Bool

from hrl_task_planning import pddl_utils as pddl
from hrl_task_planning.msg import PDDLState


class GraspStateMonitor(object):
    def __init__(self, domains):
        self.domains = domains
        self.grasping = {'right': None,
                         'left': None}
        self.l_grasp_state = None
        self.r_grasp_state_sub = rospy.Subscriber('/grasping/right_gripper', Bool, self.grasp_state_cb, 'right')
        self.l_grasp_state_sub = rospy.Subscriber('/grasping/left_gripper', Bool, self.grasp_state_cb, 'left')
        self.state_pubs = {}
        for domain in self.domains:
            self.state_pubs[domain] = rospy.Publisher('/pddl_tasks/%s/state_updates' % domain, PDDLState)

    def grasp_state_cb(self, grasping_msg, side):
        update = False
        if grasping_msg.data:
            if self.grasping[side] is None:
                pred = pddl.Predicate('GRASPING', ['HAND', 'TARGET'])
                self.grasping[side] = 'TARGET'
                update = True
        else:
            if self.grasping[side] is not None:
                pred = pddl.Predicate('GRASPING', ['HAND', self.grasping[side]], neg=True)
                self.grasping[side] = None
                update = True
        if update:
            state_msg = PDDLState()
            state_msg.predicates = [str(pred)]
            for domain, pub in self.state_pubs.iteritems():
                state_msg.domain = domain
                pub.publish(state_msg)


def main():
    rospy.init_node('grasping_state_monitor')
    parser = argparse.ArgumentParser(description="Update the PDDLState when items are grasped/released.")
    parser.add_argument('--domains', '-d', nargs='+', help="The domain(s) this monitor is updating.")
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])

    monitor = GraspStateMonitor(args.domains)
    rospy.spin()
