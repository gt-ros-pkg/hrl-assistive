import sys, optparse

import rospy
#import openravepy as op
import numpy as np
import math as m
from hrl_base_selection.srv import *
import roslib; roslib.load_manifest('hrl_haptic_mpc')
import rospy
from hrl_msgs.msg import FloatArrayBare

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist

# pos_goal=[ 2,5, 0]
#
# ori_goal = [0, 0,0, 1]
# psm = PoseStamped()
# psm.header.frame_id = '/torso_lift_link'
# psm.pose.position.x=pos_goal[0]
# psm.pose.position.y=pos_goal[1]
# psm.pose.position.z=pos_goal[2]
# psm.pose.orientation.x=ori_goal[0]
# psm.pose.orientation.y=ori_goal[1]
# psm.pose.orientation.z=ori_goal[2]
# psm.pose.orientation.w=ori_goal[3]

# base_selection_client(psm, psm, psm)

rospy.init_node('test_publishing_node')
autobed_pub = rospy.Publisher('/abdin0', FloatArrayBare, queue_size=1)
rospy.sleep(1)
msg = FloatArrayBare()
msg.data = [45., 0., 0.]
autobed_pub.publish(msg)

