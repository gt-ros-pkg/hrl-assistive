#!/usr/bin/env python

import copy

import roslib; roslib.load_manifest('hrl_face_adls')
import rospy
from tf import TransformBroadcaster

from hrl_ellipsoidal_control.msg import EllipsoidParams

class Rebroadcaster(object):
    def __init__(self):
        self.broadcaster = TransformBroadcaster()
        self.ell_param_sub = rospy.Subscriber('ellipsoid_params', EllipsoidParams, self.ell_cb)
        self.transform = None

    def ell_cb(self, ell_msg):
        print "Got transform"
        self.transform = copy.copy(ell_msg.e_frame)

    def send_transform(self):
        print "spinning", self.transform
        if self.transform is not None:
            print "Sending frame"
            self.broadcaster.sendTransform(self.transform)

if __name__=='__main__':
    rospy.init_node("ell_rebroadcaster")
    recast = Rebroadcaster()
    rate = rospy.Rate(10) 
    while not rospy.is_shutdown():
        recast.send_transform()
        rate.sleep()
        
