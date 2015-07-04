#!/usr/bin/env python

import time
import rospy
from sensor_msgs.msg import PointCloud2, Image

__author__ = 'zerickson'

class cloudTrial:
    def __init__(self):
        self.cloudTime = time.time()
        rospy.init_node('listener_cloud')
        # rospy.Subscriber('/head_mount_kinect/depth_registered/points', PointCloud2, self.cloudCallback)
        rospy.Subscriber('/head_mount_kinect/ir/image', Image, self.cloudCallback)
        rospy.spin()

    def cloudCallback(self, data):
        print 'Time between cloud calls:', time.time() - self.cloudTime
        self.cloudTime = time.time()
        print help(data)

if __name__ == '__main__':
    cloudTrial()
