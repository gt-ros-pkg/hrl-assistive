#!/usr/bin/env python

import sys
import argparse

import rospy
from std_msgs.msg import Bool

class BedConfiguredMonitor(object):
    def __init__(self):
        rospy.Subscriber('abdstatus0', Bool, self.bed_status_cb)
        self.bed_state_pub = rospy.Publisher('/configuration/bed', Bool, queue_size=10, latch=True)

    def bed_status_cb(self, msg):
        self.bed_state_pub.publish(msg)

def main():
    rospy.init_node('bed_configured_detection')
    monitor = BedConfiguredMonitor()
    rospy.sleep(1.5)
    rate = rospy.Rate(24)
    while not rospy.is_shutdown():
        rate.sleep()
