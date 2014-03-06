import sys, optparse

import rospy
import openravepy as op
import numpy as np
import math as m
from hrl_base_selection.srv import *
import roslib; roslib.load_manifest('hrl_haptic_mpc')
import rospy

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist

def select_base_client(current_loc, goal, head):
    rospy.wait_for_service('select_base_position')
    try:
        select_base_position = rospy.ServiceProxy('select_base_position', BaseMove)
        response = select_base_position(current_loc, goal, head)
        return response.BaseGoal
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e

def usage():
    return "%s [current_loc goal head]"%sys.argv[0]



if __name__ == "__main__":
    if len(sys.argv) == 3:
        current_loc = PoseStamped(sys.argv[0])
        goal = PoseStamped(sys.argv[1])
	head = PoseStamped(sys.argv[2])
    else:
        print usage()
        sys.exit(1)
    print "Requesting Base Goal Position"
    print "Base Goal Position is:"%(select_base_client(current_loc, goal, head))
