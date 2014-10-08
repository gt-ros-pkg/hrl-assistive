#!/usr/local/bin/python                                                                                          

# System library
import sys
import os
import time
import numpy as np

# ROS library
import roslib; roslib.load_manifest('hrl_anomaly_detection') 
import hrl_lib.util as ut

#import arm_trajectories as at


if __name__ == '__main__':

    import scipy.io
    mat = scipy.io.loadmat('../../matlab/test.mat')

    print mat
