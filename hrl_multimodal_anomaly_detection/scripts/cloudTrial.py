#!/usr/bin/env python

import time
import rospy
from sensor_msgs.msg import PointCloud2

__author__ = 'zerickson'

class cloudTrial:
    def __init__(self):
        self.cloudTime = time.time()
        rospy.init_node('listener_cloud')
        rospy.Subscriber('/head_mount_kinect/depth_registered/points', PointCloud2, self.cloudCallback)
        rospy.spin()

    def cloudCallback(self, data):
        print 'Time between cloud calls:', time.time() - self.cloudTime
        self.cloudTime = time.time()

if __name__ == '__main__':
    cloudTrial()
