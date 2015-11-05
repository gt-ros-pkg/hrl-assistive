#!/usr/bin/env python

import rospy
from smach_msgs.msg import SmachContainerStatus


def clean(msg, publisher):
    msg.local_data = ''
    publisher.publish(msg)


def main():
    rospy.init_node('smach_status_cleaning_relay')
    pub = rospy.Publisher('/smach_introspection/smach/container_status_cleaned', SmachContainerStatus)
    sub = rospy.Subscriber('/smach_introspection/smach/container_status', SmachContainerStatus, clean, pub)
    rospy.spin()
