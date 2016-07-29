#!/usr/bin/env python

import argparse
import sys

import rospy
from std_msgs.msg import Bool

from hrl_task_planning import pddl_utils as pddl
from hrl_task_planning.msg import PDDLState


class BedRobotMonitor(object):
    def __init__(self, domain):
        self.domain = domain
        self.state_pub = rospy.Publisher('/pddl_tasks/state_updates', PDDLState, queue_size=10, latch=True)
        self.torso_state_sub = rospy.Subscriber('/configuration/torso', Bool, self.torso_state_cb)
        self.torso_reached = False
        rospy.loginfo("[%s] Torso Height Monitor Ready.", rospy.get_name())

    def torso_state_cb(self, torso_msg):
        self.torso_reached = torso_msg.data
        preds = []
        if self.torso_reached:
            preds.append(pddl.Predicate('CONFIGURED SPINE'))
        else:
            preds.append(pddl.Predicate('CONFIGURED SPINE', neg=True))
        state_msg = PDDLState()
        state_msg.domain = self.domain
        state_msg.predicates = map(str, preds)
        self.state_pub.publish(state_msg)


def main():
    rospy.init_node('bed_robot_config_monitor')
    parser = argparse.ArgumentParser(description="Update the PDDLState when bed and robot are configured") 
    parser.add_argument('--domain', '-d', help="The domain this monitor is updating.")
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])

    monitor = BedRobotMonitor(args.domain)
    rospy.spin()
