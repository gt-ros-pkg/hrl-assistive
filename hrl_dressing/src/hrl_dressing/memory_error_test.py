#!/usr/bin/env python

import numpy as np
import math as m
import copy

import roslib

import rospy, rospkg
import tf
from geometry_msgs.msg import PoseStamped, Pose, PoseArray
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.cbook import flatten

from sensor_msgs.msg import JointState
from std_msgs.msg import String

roslib.load_manifest('hrl_base_selection')
from hrl_base_selection.helper_functions import createBMatrix, Bmat_to_pos_quat, calc_axis_angle
from hrl_base_selection.msg import PhysxOutcome
from hrl_base_selection.srv import InitPhysxBodyModel, PhysxInput, IKService, PhysxOutput, PhysxInputWaypoints

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from hrl_msgs.msg import FloatArrayBare

import random, threading

import openravepy as op
from openravepy.misc import InitOpenRAVELogging

import tf.transformations as tft

import pickle as pkl
roslib.load_manifest('hrl_lib')
from hrl_lib.util import save_pickle, load_pickle

rospy.init_node('test_memory_leak')

rospack = rospkg.RosPack()
pkg_path = rospack.get_path('hrl_dressing')

open(pkg_path+'/data/empty_text.txt', 'w').close()

start_time = rospy.Time.now()
elapsed_time = rospy.Time.now() - start_time
i = 1
save_list = []
save_dict = dict()
temp_text_file = pkg_path+'/data/upper_arm_configuration_evaluation/temp_results.txt'
open(temp_text_file, 'w').close()
alldata = load_pickle(pkg_path+'/data/upper_arm_configuration_evaluation/temp_results.pkl')
for item in alldata:
    with open(temp_text_file, 'a') as myfile:
        i +=1
        # myfile.write(str(np.round(item[0],12)) + ','+str(np.round(item[1],12)) + ',' + str(np.round(item[2],12)) + ',' + str(np.round(item[3],12)) + ','
        #              + str(item[4]) + ',' + str(item[5]) + ','
        #              + str(np.round(item[6], 12)) + '\n')
        myfile.write("{:.12f}".format(item[0]) + ',' + "{:.12f}".format(item[1]) + ',' + "{:.12f}".format(item[2])
                     + ',' + "{:.12f}".format(item[3]) + ','
                     + str(item[4]) + ',' + str(item[5]) + ','
                     + "{:.12f}".format(item[6]) + '\n')
        # print "{:.12f}".format(item[0]) + ',' + "{:.12f}".format(item[1]) + ',' + "{:.12f}".format(item[2])\
        #              + ',' + "{:.12f}".format(item[3]) + ','\
        #              + str(item[4]) + ',' + str(item[5]) + ','\
        #              + "{:.12f}".format(item[6]) + '\n'
        # if i > 10:
        #     break


# while i < 100000 and not rospy.is_shutdown() and elapsed_time.to_sec() < 10.:
    # elapsed_time = rospy.Time.now() - start_time
    # save_list.append([i, i+1])
    # save_dict[i] = [i, i+1]
    # save_pickle(save_list, pkg_path+'/data/empty_pickle.pkl')
    # with open(pkg_path+'/data/empty_text.txt', 'a') as myfile:
    #     myfile.write(str(i)+'\n')
    # i += 1
# with open(pkg_path+'/data/empty_text.txt', 'r') as f:
#     content = f.readlines()
# content = [x.strip() for x in content]

# print content
# print save_list
# print save_dict.keys()











