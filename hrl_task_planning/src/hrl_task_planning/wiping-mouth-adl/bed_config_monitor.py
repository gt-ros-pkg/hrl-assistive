#!/usr/bin/env python

import sys
import argparse

import rospy
from std_msgs.msg import Bool

class BedConfiguredMonitor(object):
    def __init__(self):
        self.domain = domain
        self.state_pub = rospy.Publisher('/pddl_tasks/state_updates', PDDLState, queue_size=10, latch=True)
        rospy.Subscriber('abdstatus0', Bool, self.bed_status_cb)
        rospy.loginfo("[%s] Bed Ready.", rospy.get_name())

    def bed_status_cb(self, msg):
        preds = []
        if msg.data:
            preds.append(pddl.Predicate('CONFIGURED BED'))
        else:
            preds.append(pddl.Predicate('CONFIGURED BED', neg=True))
            return  # nothing new here
        state_msg = PDDLState()
        state_msg.domain = self.domain
        state_msg.predicates = map(str, preds)
        self.state_pub.publish(state_msg)


def main():
    rospy.init_node('bed_configured_detection')
    parser = argparse.ArgumentParser(description="Update the PDDLState when bed is configured") 
    parser.add_argument('--domain', '-d', help="The domain this monitor is updating.")
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])

    monitor = BedConfiguredMonitor(args.domain)
    rospy.spin()
