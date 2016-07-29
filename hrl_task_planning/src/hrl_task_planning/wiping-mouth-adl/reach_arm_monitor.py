#!/usr/bin/env python

import sys
import argparse
from collections import deque

import rospy
from std_msgs.msg import Bool


from hrl_task_planning import pddl_utils as pddl
from hrl_task_planning.msg import PDDLState


class ArmReachMonitor(object):
    def __init__(self, domain):
        rospy.Subscriber("haptic_mpc/in_deadzone", std_msgs.msg.Bool, self.arm_reach_goal_cb)
        self.state_pub = rospy.Publisher('/pddl_tasks/state_updates', PDDLState, queue_size=10, latch=True)
        self.domain = domain

    def arm_reach_goal_cb(self, msg):
        preds = []
        goal_reached = msg.data
        if goal_reached:
            preds.append(pddl.Predicate('ARM-REACHED'))
        else:
            preds.append(pddl.Predicate('ARM-REACHED', neg=True))
        state_msg = PDDLState()
        state_msg.domain = self.domain
        state_msg.predicates = map(str, preds)
        self.state_pub.publish(state_msg)

   
def main():
    rospy.init_node('arm_reached_detection')
    parser = argparse.ArgumentParser(description="Update the PDDLState when arm reaches final pose.")
    parser.add_argument('--domain', '-d', help="The domain this monitor is updating.")
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])

    monitor = ArmReachMonitor(args.domain)
