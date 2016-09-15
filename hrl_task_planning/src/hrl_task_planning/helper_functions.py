#!/usr/bin/env python


import numpy as np
import roslib; roslib.load_manifest('hrl_base_selection')
import rospy

import tf.transformations as tft

def createBMatrix(pos, ori):
    goalB = np.zeros([4, 4])
    goalB[3, 3] = 1

    goalB[0:3, 0:3] = np.array(tft.quaternion_matrix(ori))[0:3, 0:3]
    for i in xrange(0, 3):
        goalB[i, 3] = pos[i]
    return np.matrix(goalB)

def Bmat_to_pos_quat(Bmat):
    pos  = np.array([Bmat[0,3],Bmat[1,3],Bmat[2,3]])
    quat = tft.quaternion_from_matrix(Bmat) # order is xyzw because ROS uses xyzw order.    

    return pos, quat

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

































