#!/usr/bin/env python


import numpy as np
import roslib; roslib.load_manifest('hrl_base_selection')
import rospy
import math as m

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
    quat = np.array(tft.quaternion_from_matrix(Bmat)) # order is xyzw because ROS uses xyzw order.

    return pos, quat

# Calculate an axis-angle from a quaternion
def calc_axis_angle(quat):
    quat /= np.linalg.norm(quat)
    angle = 2 * m.acos(quat[3])
    s = m.sqrt(1 - quat[3] * quat[3])
    # test to avoid divide by zero, s is always positive due to sqrt
    # if s close to zero then direction of axis not important
    if (s < 0.001) :
        x = 1.#q1.x // if it is important that axis is normalised then replace with x=1; y=z=0;
        y = 0.#q1.y
        z = 0.#q1.z
    else:
        x = quat[0] / s  # normalize axis
        y = quat[1] / s
        z = quat[2] / s
    return np.array([x, y, z]), angle

# # Calculate an axis-angle from a quaternion
# def rot_to_axis_angle(Bmat):
#     x = Bmat[2, 1] - Bmat[1, 2]
#     quat /= np.linalg.norm(quat)
#     angle = 2 * m.acos(quat[3])
#     s = m.sqrt(1 - quat[3] * quat[3])
#     # test to avoid divide by zero, s is always positive due to sqrt
#     # if s close to zero then direction of axis not important
#     if (s < 0.001) :
#         x = 1.#q1.x // if it is important that axis is normalised then replace with x=1; y=z=0;
#         y = 0.#q1.y
#         z = 0.#q1.z
#     else:
#         x = quat[0] / s  # normalize axis
#         y = quat[1] / s
#         z = quat[2] / s
#     return np.array([x, y, z]), angle

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

































