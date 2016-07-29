#!/usr/bin/env python

import sys
import argparse
from collections import deque

import rospy
from std_msgs.msg import Bool

from hrl_task_planning import pddl_utils as pddl
from hrl_task_planning.msg import PDDLState


class ARTagTracker(object):
    def __init__(self, domain):
        self.domain = domain
        self.state_pub = rospy.Publisher('/pddl_tasks/state_updates', PDDLState, queue_size=10, latch=True)
        rospy.Subscriber('AR_tracking', Bool, self.tracking_ar_tag_cb)

    def tracking_ar_tag_cb(self, msg):
        preds = []
        found_ar_tag = msg.data
        if found_ar_tag:
            preds.append(pddl.Predicate('IS-TRACKING-TAG'))
        else:
            preds.append(pddl.Predicate('IS-TRACKING-TAG', neg=True))
        state_msg = PDDLState()
        state_msg.domain = self.domain
        state_msg.predicates = map(str, preds)
        self.state_pub.publish(state_msg)


def main():
    rospy.init_node('ar_tag_tracking_monitor')
    parser = argparse.ArgumentParser(description="Report when the PR2 detects an AR Tag")
    parser.add_argument('--domain', '-d', help="The domain this monitor is updating.")
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])

    monitor = ARTagTracker(args.domain)
    rospy.spin()
