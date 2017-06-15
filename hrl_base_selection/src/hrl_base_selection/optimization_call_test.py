#!/usr/bin/env python

import random, time
import numpy as np
import matplotlib.pyplot as plt
import math as m
import copy, threading, os
import rospy


from helper_functions import createBMatrix, Bmat_to_pos_quat
# from hmm_outcome_estimator import HMMOutcomeEstimator

# from basesim import BaseSim
# import util, datapreprocess

import roslib, rospkg
roslib.load_manifest('hrl_lib')
from hrl_lib.util import save_pickle, load_pickle
from std_msgs.msg import String
from hrl_msgs.msg import FloatArrayBare

from matplotlib.cbook import flatten

# from sleeve_pull_simulator import DressingTest


class OptimizationTest(object):
    def __init__(self):
        # dt = DressingTest()
        th = m.radians(0.)
        test_world_should_B_sleeve_start_trans = np.matrix([[  m.cos(th), 0.,  m.sin(th),     0.38],
                                                             [         0., 1.,         0.,    -0.32],
                                                             [ -m.sin(th), 0.,  m.cos(th),     0.1],
                                                             [         0., 0.,         0.,     1.]])

        th = m.radians(180.)
        test_world_shoulder_B_sleeve_start_rotz = np.matrix([[ m.cos(th), -m.sin(th),     0.,      0.],
                                                             [ m.sin(th),  m.cos(th),     0.,      0.],
                                                             [               0.,         0.,     1.,      0.],
                                                             [        0.,         0.,     0.,        1.]])
        th = m.radians(-30.)
        test_world_shoulder_B_sleeve_start_roty = np.matrix([[  m.cos(th), 0.,  m.sin(th),     0.],
                                                             [         0., 1.,         0.,    0.],
                                                             [ -m.sin(th), 0.,  m.cos(th),     0.],
                                                             [         0., 0.,         0.,     1.]])
        th = m.radians(0.)
        test_world_shoulder_B_sleeve_start_rotx = np.matrix([[1.,          0.,          0., 0.],
                                                             [0.,   m.cos(th),  -m.sin(th), 0.],
                                                             [0.,   m.sin(th),   m.cos(th), 0.],
                                                             [0.,          0.,          0., 1.]])
        th = m.radians(-90)
        shoulder_origin_B_rotated_shoulder_rotx = np.matrix([[1.,          0.,          0., 0.],
                                                             [0.,   m.cos(th),  -m.sin(th), 0.],
                                                             [0.,   m.sin(th),   m.cos(th), 0.],
                                                             [0.,          0.,          0., 1.]])
        pos, rot = Bmat_to_pos_quat(test_world_should_B_sleeve_start_trans*test_world_shoulder_B_sleeve_start_rotz*test_world_shoulder_B_sleeve_start_roty)
        pos_r, rot_r = Bmat_to_pos_quat(shoulder_origin_B_rotated_shoulder_rotx)
        self.simulation_setting = list(flatten([pos, rot, 0.85, m.radians(90), rot_r]))
        # self.set_simulation([0., 0., 0.], [.15, 0., 0.], 0, [0., 0., 0., 0.])
        self.traj_to_simulator_pub = rospy.Publisher('physx_simulator_input', FloatArrayBare, queue_size=1, latch=True)
        self.simulator_result_sub = rospy.Subscriber('physx_simulator_result', String, self.simulator_result_cb)
        # dt.set_simulation(pos, rot, 0.85, m.radians(90), rot_r)
        # dt.simulating = True
        # dt.start()

    def call_simulator(self):
        out_data = FloatArrayBare()
        out_data.data = self.simulation_setting
        self.traj_to_simulator_pub.publish(out_data)

    def simulator_result_cb(self, msg):
        print 'The result my optimization call test received was: ', msg.data

if __name__ == "__main__":
    rospy.init_node('optimization_test_node')
    h = OptimizationTest()
    h.call_simulator()
    rospy.spin()
    # output = h.determine_outcome(viz=True)
    # print output