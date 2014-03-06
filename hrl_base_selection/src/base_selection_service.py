import sys, optparse

import rospy
import openravepy as op
import numpy as np
import math as m
import roslib
roslib.load_manifest('hrl_base_selection')
from hrl_base_selection.srv import *



def handle_select_base(req):
    print 'I got things!', req.current_loc, req.goal, req.head
    return req.goal

def select_base_server():
    rospy.init_node('select_base_server')
    s = rospy.Service('select_base_position', BaseMove, handle_select_base)
    print "Ready to select base."
    rospy.spin()

if __name__ == "__main__":
    select_base_server()
