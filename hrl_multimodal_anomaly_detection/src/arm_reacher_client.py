#!/usr/bin/env python
import roslib; roslib.load_manifest('sandbox_dpark_darpa_m3')
roslib.load_manifest('hrl_multimodal_anomaly_detection')
import rospy
import numpy as np, math
import time

from hrl_srvs.srv import None_Bool, None_BoolResponse



if __name__ == '__main__':

    rospy.init_node('feed_client')

    rospy.wait_for_service("/arm_reach_enable")
    armReachAction = rospy.ServiceProxy("/arm_reach_enable", None_Bool)
    #armMovements = rospy.ServiceProxy("/arm_reach_enable", None_Bool)

    ret = armReachAction()
    #ret = armMovements()
    print ret
