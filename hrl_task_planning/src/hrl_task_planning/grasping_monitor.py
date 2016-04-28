#!/usr/bin/env python

import argparse
import sys

import rospy
from std_msgs.msg import Bool

from hrl_task_planning import pddl_utils as pddl
from hrl_task_planning.msg import PDDLState


class GraspStateMonitor(object):
    def __init__(self, domain, side):
        self.domain = domain
        self.side = side
        self.gripper_name = 'RIGHT_HAND' if 'R' in self.side.upper() else 'LEFT_HAND'
        self.item_name = self.gripper_name+'_OBJECT'
        self.grasped_item = None
        self.state_pub = rospy.Publisher('/pddl_tasks/%s/state_updates' % domain, PDDLState, queue_size=10, latch=True)
        self.grasp_state_sub = rospy.Subscriber('/grasping/%s_gripper' % side, Bool, self.grasp_state_cb)
        rospy.loginfo("[%s] %s Grasp State Monitor Ready.", rospy.get_name(), self.side.capitalize())

    def grasp_state_cb(self, grasping_msg):
        if (grasping_msg.data and self.grasped_item is None):
                pred = pddl.Predicate('GRASPING', [self.gripper_name, self.item_name])
                self.grasped_item = self.item_name
        elif (not grasping_msg.data and self.grasped_item is not None):
                pred = pddl.Predicate('GRASPING', [self.gripper_name, self.grasped_item], neg=True)
                self.grasped_item = None
        else:
            return  # nothing new here
        state_msg = PDDLState()
        state_msg.domain = self.domain
        state_msg.predicates = [str(pred)]
        self.state_pub.publish(state_msg)


def main():
    rospy.init_node('grasping_state_monitor')
    parser = argparse.ArgumentParser(description="Update the PDDLState when items are grasped/released.")
    parser.add_argument('--domain', '-d', help="The domain this monitor is updating.")
    parser.add_argument('--side', '-s', help="The side of the robot his monitor is updating")
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])

    monitor = GraspStateMonitor(args.domain, args.side)
    rospy.spin()
