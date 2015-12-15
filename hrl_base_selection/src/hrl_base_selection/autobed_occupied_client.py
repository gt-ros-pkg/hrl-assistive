#!/usr/bin/env python  
import roslib 
roslib.load_manifest('hrl_base_selection')

import sys
import rospy
from hrl_srvs.srv import None_Bool, None_BoolResponse

def autobed_occupied_status_client():
    rospy.wait_for_service('autobed_occ_status')
    try:
        AutobedOcc = rospy.ServiceProxy('autobed_occ_status', None_Bool)
        resp1 = AutobedOcc()
        return resp1
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e


if __name__ == "__main__":
    print "Occupied Status:{}".format(autobed_occupied_status_client())
