#!/usr/bin/env python

import numpy as np
import roslib; roslib.load_manifest('hrl_haptic_mpc')
import roslib
roslib.load_manifest('hrl_base_selection')
roslib.load_manifest('hrl_haptic_mpc')
import rospkg

roslib.load_manifest('hrl_base_selection')
import rospy
import time
from data_reader_cma import DataReader as DataReader_cma
from data_reader_comparisons import DataReader as DataReader_comparisons
from data_reader_task import DataReader_Task
# from score_generator import ScoreGenerator
from score_generator_comparisons import ScoreGenerator
from hrl_base_selection.inverse_reachability_setup import InverseReachabilitySetup
# from config_visualize import ConfigVisualize
import scipy.stats

import gc

import random
import os

# import sPickle as pkl
roslib.load_manifest('hrl_lib')
from hrl_lib.util import load_pickle

import matplotlib as mpl
import matplotlib.pyplot as plt

import csv

rospy.init_node('toc_comparison_plotting')
rospack = rospkg.RosPack()
pkg_path = rospack.get_path('hrl_base_selection')
data_path = '/home/ari/svn/robot1_data/usr/ari/data/base_selection'

load_file_name = 'toc_correlation_results.log'
load_file_path = pkg_path + '/data/'

with open(load_file_path+load_file_name, 'r') as f:
    reader = csv.reader(f)
    your_list = list(reader)

# print your_list

data = np.zeros([len(your_list),len(your_list[0])-2])
for i in xrange(len(your_list)):
    data[i] = your_list[i][2:]
# print data[:,4]
# print data[:,2]

xedges = np.arange(-0.5,0.2,0.01)
yedges = np.arange(0.,1.,0.1)
from pylab import hist2d
# plt.subplots(3, 1, sharex=True, sharey=True,
#                         tight_layout=True)
# fig = plt.figure(1)
# plt.show()
# fig.set_size_inches(18.5, 10.5, forward=True)
# ax = fig.add_subplot(111)
plt.hist2d(data[:, 4], data[:, 2],bins=40,cmin=1)
# ax.set_xlim(-0.4, -0.2)
# ax.set_ylim(0., 1.0)
# plt.colorbar()
# ax.set_title('Histogram!')

# ax.set_xlabel('TOC Score', fontsize=20)
# ax.set_ylabel('Success', fontsize=20)
# ax.tick_params(axis='x', labelsize=20)
# ax.tick_params(axis='y', labelsize=20)
plt.colorbar()
# ax.set_title(''.join(['Force in upward direction vs Position']), fontsize=20)







# H, xedges, yedges = np.histogram2d(data[:, 4], data[:, 2], bins=20)
# H = H.T  # Let each row list bins with common y range.

fig = plt.figure()

# fig = plt.figure(figsize=(7, 3))
# ax = fig.add_subplot(131, title='imshow: square bins')
# plt.imshow(H, interpolation='nearest', origin='low', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

# ax = fig.add_subplot(132, title='pcolormesh: actual edges', aspect='equal')
# X, Y = np.meshgrid(xedges, yedges)
# ax.pcolormesh(X, Y, H)

# ax = fig.add_subplot(133, title='NonUniformImage: interpolated', aspect='equal', xlim=xedges[[0, -1]], ylim=yedges[[0, -1]])
# im = mpl.image.NonUniformImage(ax, interpolation='bilinear')
# xcenters = (xedges[:-1] + xedges[1:]) / 2
# ycenters = (yedges[:-1] + yedges[1:]) / 2
# im.set_data(xcenters, ycenters, H)
# ax.images.append(im)



rospy.spin()







