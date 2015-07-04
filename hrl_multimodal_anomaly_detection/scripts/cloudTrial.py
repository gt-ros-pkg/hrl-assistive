#!/usr/bin/env python

import time
import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2, Image

import roslib
roslib.load_manifest('hrl_multimodal_anomaly_detection')
from cv_bridge import CvBridge, CvBridgeError

__author__ = 'zerickson'

class cloudTrial:
    def __init__(self):
        self.cloudTime = time.time()
        self.bridge = CvBridge()
        rospy.init_node('listener_cloud')
        # rospy.Subscriber('/head_mount_kinect/depth_registered/points', PointCloud2, self.cloudCallback)
        rospy.Subscriber('/head_mount_kinect/ir/image', Image, self.cloudCallback)
        rospy.spin()

    def cloudCallback(self, data):
        print 'Time between cloud calls:', time.time() - self.cloudTime
        self.cloudTime = time.time()
        print data.width, data.height, type(data.data)

        # Grab image from Kinect sensor
        try:
            image = self.bridge.imgmsg_to_cv(data)
            image = np.asarray(image[:,:])
        except CvBridgeError, e:
            print e
            return

        print len(data.data), image.shape

if __name__ == '__main__':
    cloudTrial()
