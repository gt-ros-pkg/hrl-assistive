#!/usr/bin/env python

import rospy
import numpy as np
from sensor_msgs.msg import *
from geometry_msgs.msg import PoseStamped

def callback(data):
    print data
    print np.asarray(data.K).reshape((3,3))


def listener():
    rospy.init_node('listener')
    pub = rospy.Publisher("/right/haptic_mpc/gripper_pose", PoseStamped, queue_size=10)
    rate=rospy.Rate(10)
    while not rospy.is_shutdown():
        pose=PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pub.publish(pose)
        rate.sleep()

if __name__ == '__main__':
    listener()
