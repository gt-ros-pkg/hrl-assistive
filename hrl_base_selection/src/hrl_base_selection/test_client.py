#!/usr/bin/env python

#!/usr/bin/env python
import sys, optparse

import rospy, rospkg
import openravepy as op
import numpy as np
import math as m
import copy
from hrl_base_selection.srv import BaseMove, setBaseModel
import roslib
import time
roslib.load_manifest('hrl_base_selection')
roslib.load_manifest('hrl_haptic_mpc')
import rospy, rospkg
import tf
roslib.load_manifest('hrl_base_selection')
import hrl_lib.transforms as tr
import rospy
from visualization_msgs.msg import Marker
import time
from sensor_msgs.msg import PointCloud2
from hrl_msgs.msg import FloatArrayBare
from helper_functions import createBMatrix
from pr2_controllers_msgs.msg import SingleJointPositionActionGoal


class TestClient(object):
    def __init__(self):
        self.myPointCloud = None
        self.pc_sub = rospy.Subscriber('/head_mount_kinect/sd/points', PointCloud2, self.pc_cb)
        self.run_client()


    def pc_cb(self, msg):
        # print 'got message'
        self.myPointCloud = msg

    def run_client(self):
        rospy.wait_for_service('read_environment_model')
        rospy.wait_for_service('select_base_position')
        print 'services are available!'



        # rospy.sleep(4)
        while self.myPointCloud is None and not rospy.is_shutdown():
            rospy.sleep(1)
        # print myPointCloud
        read_environment_model = rospy.ServiceProxy('set_environment_model', setBaseModel)
        response = read_environment_model(self.myPointCloud)
        print response
        if response.success:
            print 'Environment has been read!'
        # select_base_position = rospy.ServiceProxy('select_base_position', BaseMove_multi)

if __name__ == "__main__":
    rospy.init_node('client_node')
    tc = TestClient()
    rospy.spin()

