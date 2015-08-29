#!/usr/bin/env python

# System
import gc
from pylab import *

from onlineAnomalyDetection import onlineAnomalyDetection

# ROS
import roslib
roslib.load_manifest('hrl_multimodal_anomaly_detection')
import rospy, optparse
import tf

def log_parse():
    parser = optparse.OptionParser('Input the Pose node name and the ft sensor node name')

    parser.add_option("-t", "--tracker", action="store", type="string",\
    dest="tracker_name", default="adl2")
    parser.add_option("-f", "--force" , action="store", type="string",\
    dest="ft_sensor_name",default="/netft_data")

    (options, args) = parser.parse_args()

    return options.tracker_name, options.ft_sensor_name


class ADL_log:
    def __init__(self, subject='s1', task='s'):
        self.init_time = 0
        self.tf_listener = tf.TransformListener()
        rospy.logout('ADLs_log node subscribing..')
        self.isScooping = task == 's' or task == 'b'

        self.detector = onlineAnomalyDetection(subject=subject, task=task, targetFrame='/torso_lift_link', tfListener=self.tf_listener, isScooping=self.isScooping)
        self.detector.start()

        self.detector2 = None
        if task == 'b':
            self.detector2 = onlineAnomalyDetection(subject=subject, task=task, targetFrame='/torso_lift_link', tfListener=self.tf_listener, isScooping=False)
            self.detector2.start()

    def log_start(self, secondDetector=False):
        self.init_time = rospy.get_time()
        if not secondDetector:
            self.detector.reset()
            self.detector.init_time = self.init_time
        else:
            self.detector2.reset()
            self.detector2.init_time = self.init_time

    def close_log_file(self, secondDetector=False):
        if not secondDetector:
            self.detector.cancel()
        else:
            self.detector2.cancel()

        # Reinitialize all sensors
        # self.detector = onlineAnomalyDetection(targetFrame='/torso_lift_link', tfListener=self.tf_listener, isScooping=self.isScooping)

        gc.collect()

if __name__ == '__main__':
    log = ADL_log()

    log.log_start()

    rate = rospy.Rate(1000) # 25Hz, nominally.
    while not rospy.is_shutdown():
        rate.sleep()

    log.close_log_file()

