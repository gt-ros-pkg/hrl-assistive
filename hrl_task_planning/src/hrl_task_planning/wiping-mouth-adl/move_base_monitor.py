#!/usr/bin/env python

import sys
import argparse

import rospy
from std_msgs.msg import Int8

from hrl_task_planning import pddl_utils as pddl
from hrl_task_planning.msg import PDDLState



class MoveBaseMonitor(object):
    def __init__(self, domain):
        self.domain = domain
        self.state_pub = rospy.Publisher('/pddl_tasks/state_updates', PDDLState, queue_size=10, latch=True)
        rospy.Subscriber('/pr2_ar_servo/state_feedback', Int8, self.base_servoing_cb)
        rospy.loginfo("[%s] PR2 Servoing Monitor Ready.", rospy.get_name())

    def base_servoing_cb(self, msg):
        preds = []
        if msg.data == 5:
            preds.append(pddl.Predicate('BASE-REACHED'))
        else:
            preds.append(pddl.Predicate('BASE-REACHED', neg=True))
        state_msg = PDDLState()
        state_msg.domain = self.domain
        state_msg.predicates = map(str, preds)
        self.state_pub.publish(state_msg)

def main():
    rospy.init_node('move_base_monitor')
    parser = argparse.ArgumentParser(description="Update the PDDLState when base is reached")
    parser.add_argument('--domain', '-d', help="The domain this monitor is updating.")
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])

    monitor = MoveBaseMonitor(args.domain)
    rospy.spin()
