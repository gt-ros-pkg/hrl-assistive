#!/usr/bin/env python

import sys
import argparse
from collections import deque

import rospy
from std_msgs.msg import Bool

from pr2_controllers_msgs.msg import SingleJointPositionActionGoal

class TorsoConfiguredMonitor(object):
    def __init__(self):
        rospy.Subscriber('torso_controller/position_joint_action/goal', SingleJointPositionActionGoal, self.torso_goal_cb)
        self.torso_state_pub = rospy.Publisher('/configuration/torso', Bool, queue_size=10, latch=True)
        self.torso_state_deque = deque([None]*24, 24)
        self.goal_level = -100.0

    def torso_goal_cb(self, msg):
        self.goal_level = msg.position

    def get_torso_height(self):
        rospy.wait_for_service("return_joint_states")
        try:
            s = rospy.ServiceProxy("return_joint_states", ReturnJointStates)
            resp = s('torso_lift_joint')
        except rospy.ServiceException, e:
            print "error when calling return_joint_states: %s"%e
            sys.exit(1)
        return resp.position

    def update_torso_state(self):
        torso_configured = None
        if self.goal_level == -100.0:
            self.torso_state_pub.publish(False)
            return
        torso_error = abs(self.goal_level - get_torso_height())
        if torso_error < 0.5:
            torso_configured = True if torso_configured is None else torso_configured
        else:
            torso_configured = False if torso_configured is None else torso_configured
        if torso_configured is None:
            return  # Nothing happening, skip ahead

        self.torso_state_deque.append(torso_configured)
        if None in self.torso_state_deque:
            return
        filtered_torso = True if sum(self.torso_state_deque) > self.torso_state_deque.maxlen/2 else False
        self.torso_state_pub.publish(filtered_torso)


def main():
    rospy.init_node('torso_configured_detection')
    monitor = TorsoConfiguredMonitor()
    rospy.sleep(1.5)
    rate = rospy.Rate(12)
    while not rospy.is_shutdown():
        monitor.update_torso_state()
        rate.sleep()
