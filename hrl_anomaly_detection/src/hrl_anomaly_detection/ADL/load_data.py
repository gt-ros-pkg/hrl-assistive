#!/usr/bin/env python

# System
import numpy as np
import time, sys, threading
import cPickle as pkl


# ROS
import roslib
roslib.load_manifest('hrl_anomaly_detection')
roslib.load_manifest('geometry_msgs')
roslib.load_manifest('hrl_lib')
import rospy, optparse, math, time
import tf
from geometry_msgs.msg import Wrench
from geometry_msgs.msg import TransformStamped, WrenchStamped
from std_msgs.msg import Bool

# HRL
import hrl_lib.util as ut
import matplotlib.pyplot as pp
import matplotlib as mpl


if __name__ == '__main__':

    pkl_file = './test_cup_human_t1.pkl'
    d = ut.load_pickle(pkl_file)

    print d.keys()
    ftime = d['ftime']
    force = d['force_raw']

    aForce = np.squeeze(force).T
    print aForce.shape
    
    pp.figure()
    pp.plot(ftime, aForce[0])
    pp.show()
