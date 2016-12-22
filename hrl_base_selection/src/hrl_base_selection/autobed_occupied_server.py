#!/usr/bin/env python  

import roslib
roslib.load_manifest('hrl_base_selection')
import rospy
import numpy as np
import time
from hrl_msgs.msg import FloatArrayBare
from hrl_srvs.srv import None_Bool, None_BoolResponse

NUMOFTAXELS_X = 64#73 #taxels
NUMOFTAXELS_Y = 27#30 
TOTAL_TAXELS = NUMOFTAXELS_X*NUMOFTAXELS_Y


class AutobedOccupied():
    def __init__(self):
        self.pressure_map = []
        self.sub_once = rospy.Subscriber("/fsascan", FloatArrayBare, self.p_map_cb)
        rospy.sleep(1)
        # service for robot's request
        self.occ_status = rospy.Service('autobed_occ_status', None_Bool, self.autobed_occ_cb)


    def autobed_occ_cb(self, req):
        print "Autobed Data Sampled"
        total_weight = np.sum(np.asarray(self.pressure_map)) 
        print "Total Weight on the mat:{}".format(total_weight)
        if total_weight >= 800:
            return None_BoolResponse(True)
        else:
            return None_BoolResponse(False)


    def p_map_cb(self, data):
        '''This callback accepts incoming pressure map from 
        the Vista Medical Pressure Mat and sends it out. 
        Remember, this array needs to be binarized to be used'''
        if len(data.data) == TOTAL_TAXELS:
            self.pressure_map = data.data


if __name__ == '__main__':
    rospy.init_node('autobed_occ_status_server')
    ara = AutobedOccupied()
    rospy.spin()
