#!/usr/bin/env python

import time
import rospy
from sensor_msgs.msg import Image

__author__ = 'zerickson'

class rgbTrial:
    def __init__(self):
        self.rgbTime = time.time()
        rospy.init_node('listener_rgb')
        rospy.Subscriber('/head_mount_kinect/rgb_lowres/image', Image, self.imageCallback)
        rospy.spin()

    def imageCallback(self, data):
        print 'Time between rgb calls:', time.time() - self.rgbTime
        self.rgbTime = time.time()

if __name__ == '__main__':
    rgbTrial()
