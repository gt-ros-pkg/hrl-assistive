#!/usr/bin/env python

import cv2
import rospy
import numpy as np
from sensor_msgs.msg import Image

import roslib
roslib.load_manifest('hrl_multimodal_anomaly_detection')
from cv_bridge import CvBridge, CvBridgeError
from hrl_multimodal_anomaly_detection.msg import ImageFeatures

__author__ = 'zerickson'

class displayCameraData:
    def __init__(self):
        self.featureData = None
        self.bridge = CvBridge()

        rospy.init_node('listener_features')

        rospy.Subscriber('image_features', ImageFeatures, self.featuresCallback)

        # XBox 360 Kinect
        # rospy.Subscriber('/camera/rgb/image_color', Image, self.imageCallback)
        # Kinect 2
        rospy.Subscriber('/head_mount_kinect/rgb_lowres/image', Image, self.imageCallback)
        # PR2 Simulated
        # rospy.Subscriber('/head_mount_kinect/rgb/image_color', Image, self.imageCallback)
        print 'Connected to Kinect images'

        rospy.spin()

    def imageCallback(self, data):
        try:
            image = self.bridge.imgmsg_to_cv(data)
            image = np.asarray(image[:,:])
        except CvBridgeError, e:
            print e
            return

        if self.featureData is not None:
            # Draw all rectangles (bounding boxes)
            for rect in self.featureData.rectangles:
                rgb = [rect.b, rect.g, rect.r]
                cv2.rectangle(image, (rect.lowX, rect.lowY), (rect.highX, rect.highY), color=rgb, thickness=rect.thickness)

            # Draw all circles (features)
            for circle in self.featureData.circles:
                rgb = [circle.b, circle.g, circle.r]
                cv2.circle(image, (circle.x, circle.y), circle.radius, rgb, -1)

        cv2.imshow('Image window', image)
        cv2.waitKey(30)

    def featuresCallback(self, data):
        self.featureData = data

if __name__ == '__main__':
    displayCameraData()
