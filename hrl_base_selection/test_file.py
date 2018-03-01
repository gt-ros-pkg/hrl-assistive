#!/usr/bin/env python

import numpy as np
import math as m
import copy

import pydart2 as pydart

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
from hrl_base_selection.dart_setup import DartDressingWorld
from hrl_base_selection.graph_search_functions import SimpleGraph, a_star_search, reconstruct_path
from hrl_base_selection.msg import PhysxOutcome
from hrl_base_selection.srv import InitPhysxBodyModel, PhysxInput, IKService, PhysxOutput, PhysxInputWaypoints

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from hrl_msgs.msg import FloatArrayBare

import random, threading

import openravepy as op
from openravepy.misc import InitOpenRAVELogging

from sklearn.neighbors import NearestNeighbors

import tf.transformations as tft

import pickle as pkl
roslib.load_manifest('hrl_lib')
from hrl_lib.util import save_pickle, load_pickle


rospack = rospkg.RosPack()
pkg_path = rospack.get_path('hrl_base_selection')
save_file_path = pkg_path + '/data/'
save_file_name = 'arm_config_scores.log'
save_file_name_only_good = 'arm_configs_feasible.log'

rospy.init_node('test_file')

subtask_step = 1

feasible_configs = [line.rstrip('\n').split(',')
                for line in open(save_file_path + save_file_name_only_good)]


for j in xrange(len(feasible_configs)):
    feasible_configs[j] = [float(i) for i in feasible_configs[j]]
feasible_configs = np.array(feasible_configs)
feasible_configs = np.array([x for x in feasible_configs if int(x[0]) == subtask_step])
print 'feasible total:', len(feasible_configs)
cluster_count = 0
clusters = [[]]
while len(feasible_configs) > 0 and not rospy.is_shutdown():
    if len(clusters) < cluster_count + 1:
        print 'adding'
        clusters.append([])
    print 'feasible_configs:\n',feasible_configs
    print 'feasible total:', len(feasible_configs)
    queue = []
    visited = []
    queue.append(list(feasible_configs[0][1:5]))
    delete_list = []
    while len(queue) > 0  and not rospy.is_shutdown():
        #print 'queue:\n',queue
	#print 'visited:\n',visited
        current_node = list(queue.pop(0))
        # print 'current node:\n',current_node
        # print 'visited:\n',visited
        if current_node not in visited:
            visited.append(list(current_node))
            clusters[cluster_count].append(current_node)
            delete_list.append(0)
        for node_i in xrange(len(feasible_configs)):
            if np.max(np.abs(np.array(current_node) - np.array(feasible_configs[node_i])[1:5])) <     m.radians(5.1) and list(feasible_configs[node_i][1:5]) not in visited:
                close_node = list(feasible_configs[node_i][1:5])
                queue.append(list(close_node))
                delete_list.append(node_i)
                # clusters[cluster_count.append(close_node)]
    feasible_configs = np.delete(feasible_configs,delete_list,axis=0)
    cluster_count += 1
clusters = np.array(clusters)
print 'clusters:\n',clusters
print 'number of clusters:',len(clusters)
print 'number of clusters1:',len(clusters[0])
print 'number of clusters2:',len(clusters[1])
init_start_arm_configs = []
for cluster in clusters:
    init_start_arm_configs.append(np.array(cluster).mean(axis=0))
print init_start_arm_configs
