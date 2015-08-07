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
from hrl_srvs.srv import None_Bool, None_BoolResponse, Int_Int
from hrl_multimodal_anomaly_detection.srv import String_String

def log_parse():
    parser = optparse.OptionParser('Input the Pose node name and the ft sensor node name')

    parser.add_option("-t", "--tracker", action="store", type="string",\
    dest="tracker_name", default="adl2")
    parser.add_option("-f", "--force" , action="store", type="string",\
    dest="ft_sensor_name",default="/netft_data")

    (options, args) = parser.parse_args()

    return options.tracker_name, options.ft_sensor_name


class ADL_log:
    def __init__(self, isScooping=True, useAudio=False):
        self.init_time = 0
        self.tf_listener = tf.TransformListener()
        rospy.logout('ADLs_log node subscribing..')
        self.isScooping = isScooping

        self.detector = onlineAnomalyDetection(targetFrame='/torso_lift_link', tfListener=self.tf_listener, isScooping=isScooping, useAudio=useAudio)

        self.scooping_steps_times = []
        self.scoopingStepsService = rospy.Service('/scooping_steps_service', None_Bool, self.scoopingStepsTimesCallback)

    def log_start(self):
        self.init_time = rospy.get_time()
        self.detector.init_time = self.init_time
        self.detector.start()

    def close_log_file(self):
        self.detector.cancel()

        # Reinitialize all sensors
        self.detector = onlineAnomalyDetection(targetFrame='/torso_lift_link', tfListener=self.tf_listener, isScooping=self.isScooping)

        gc.collect()

    def scoopingStepsTimesCallback(self, data):
        self.scooping_steps_times.append(rospy.get_time() - self.init_time)
        return None_BoolResponse(True)

if __name__ == '__main__':
    log = ADL_log()

    log.log_start()

    rate = rospy.Rate(1000) # 25Hz, nominally.
    while not rospy.is_shutdown():
        rate.sleep()

    log.close_log_file()

