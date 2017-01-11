import numpy as np
import math as m
import copy

import time
import roslib
roslib.load_manifest('hrl_base_selection')
roslib.load_manifest('hrl_haptic_mpc')
import rospy, rospkg
import tf
from geometry_msgs.msg import PoseStamped
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.cbook import flatten
from itertools import combinations as comb
from operator import itemgetter

from sensor_msgs.msg import JointState
from std_msgs.msg import String
# import hrl_lib.transforms as tr
from hrl_base_selection.srv import BaseMove#, BaseMove_multi
from visualization_msgs.msg import Marker, MarkerArray
from helper_functions import createBMatrix, Bmat_to_pos_quat
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import random

import pickle as pkl
roslib.load_manifest('hrl_lib')
from hrl_lib.util import save_pickle



l_min = np.array([-40., -30., -44., -45., -400., -130., -400.])
l_max = np.array([130., 80., 224., 0., 400., 0., 400.])
l_range = l_max - l_min
weights_list = []  # = [[], [], [], [], [], [], []]
q_list = []
for i in xrange(3001):
    q = i*l_range/3000.+l_min
    # print 'q', q
    q_list.append(q)
    weights = np.zeros(7)
    for joint in xrange(len(weights)):
        weights[joint] = (1. - m.pow(0.5, ((l_range[joint])/2. - np.abs((l_range[joint])/2. - q[joint] + l_min[joint]))/(l_range[joint]/40.)))
        # weights[joint] = (1. - m.pow(0.5, (np.abs((l_max[joint] - l_min[joint])/2. - q[joint] - l_min[joint]))))
        # print weights[joint]
        # if weights[joint] < 0.01:
        #     weights[joint]=0.01
    # print 'weights', weights
    weights_list.append(weights)
q_list = np.array(q_list)
weights_list = np.array(weights_list)

q_percent = []
for item in q_list:
    perc = np.zeros(7)
    for j in xrange(len(item)):
        perc[j] = (-(l_range[j]/2. - item[j] + l_min[j]))/(l_range[j]/2.)
    q_percent.append(perc)
    # print perc
q_percent = np.array(q_percent)

# print weights_list[:, 0]
# print q_list[:,0]
fig1 = plt.figure(1)
joint = 0
ax1 = fig1.add_subplot(231)
ax1.set_xlabel('Percent of Angle Range')
ax1.set_ylabel('Weight')
ax1.set_ylim(0.0, 1.05)
# ax1.set_xlim(l_min[joint], l_max[joint])
ax1.set_xlim(-1.0, 1.0)
# surf11 = ax1.scatter(q_list[:, joint], weights_list[:, joint], color="green", s=1, alpha=1)
surf11 = ax1.scatter(q_percent[:, joint], weights_list[:, joint], color="green", s=1, alpha=1)

joint = 1
ax2 = fig1.add_subplot(232)
ax2.set_xlabel('Percent of Angle Range')
ax2.set_ylabel('Weight')
ax2.set_ylim(0.0, 1.05)
ax2.set_xlim(l_min[joint], l_max[joint])
ax2.set_xlim(-1.0, 1.0)
# surf12 = ax2.scatter(q_list[:, joint], weights_list[:, joint], color="green", s=1, alpha=1)
surf12 = ax2.scatter(q_percent[:, joint], weights_list[:, joint], color="green", s=1, alpha=1)

joint = 2
ax3 = fig1.add_subplot(233)
ax3.set_xlabel('Percent of Angle Range')
ax3.set_ylabel('Weight')
ax3.set_ylim(0.0, 1.05)
ax3.set_xlim(l_min[joint], l_max[joint])
ax3.set_xlim(-1.0, 1.0)
# surf13 = ax3.scatter(q_list[:, joint], weights_list[:, joint], color="green", s=1, alpha=1)
surf13 = ax3.scatter(q_percent[:, joint], weights_list[:, joint], color="green", s=1, alpha=1)

joint = 3
ax4 = fig1.add_subplot(234)
ax4.set_xlabel('Percent of Angle Range')
ax4.set_ylabel('Weight')
ax4.set_ylim(0.0, 1.05)
ax4.set_xlim(l_min[joint], l_max[joint])
ax4.set_xlim(-1.0, 1.0)
# surf14 = ax4.scatter(q_list[:, joint], weights_list[:, joint], color="green", s=1, alpha=1)
surf14 = ax4.scatter(q_percent[:, joint], weights_list[:, joint], color="green", s=1, alpha=1)

joint = 5
ax5 = fig1.add_subplot(235)
ax5.set_xlabel('Percent of Angle Range')
ax5.set_ylabel('Weight')
ax5.set_ylim(0.0, 1.05)
ax5.set_xlim(l_min[joint], l_max[joint])
ax5.set_xlim(-1.0, 1.0)
# surf14 = ax4.scatter(q_list[:, joint], weights_list[:, joint], color="green", s=1, alpha=1)
surf15 = ax5.scatter(q_percent[:, joint], weights_list[:, joint], color="green", s=1, alpha=1)



plt.show()