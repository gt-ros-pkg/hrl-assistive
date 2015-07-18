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
    def __init__(self):
        self.init_time = 0
        self.tf_listener = tf.TransformListener()
        rospy.logout('ADLs_log node subscribing..')

        self.detector = onlineAnomalyDetection(targetFrame='/torso_lift_link', tfListener=self.tf_listener)

    def log_start(self):
        self.init_time = rospy.get_time()
        self.detector.init_time = self.init_time
        self.detector.start()

    def close_log_file(self):
        self.detector.cancel()

        # Reinitialize all sensors
        self.detector = onlineAnomalyDetection(targetFrame='/torso_lift_link', tfListener=self.tf_listener)

        gc.collect()

if __name__ == '__main__':
    log = ADL_log()

    log.log_start()

    rate = rospy.Rate(1000) # 25Hz, nominally.
    while not rospy.is_shutdown():
        rate.sleep()

    log.close_log_file()

