#!/usr/bin/env python

import openravepy as op
import numpy

import os

#from openravepy.misc import InitOpenRAVELogging 
#InitOpenRAVELogging() 

# from openravepy import *
import numpy, time
import rospkg
import math as m
import numpy as np
import rospy
import roslib
roslib.load_manifest('hrl_base_selection')
from hrl_base_selection.helper_functions import createBMatrix, Bmat_to_pos_quat
from hrl_base_selection.srv import IKService


rospy.init_node('test_service')

print 'waiting for service'
rospy.wait_for_service('ikfast_service')
print 'found service'
serv = rospy.ServiceProxy('ikfast_service', IKService)
resp = serv([0.6, -0.25, 1.05], [0., 0., 0., 1.], 0.1, 'rightarm')
print resp
