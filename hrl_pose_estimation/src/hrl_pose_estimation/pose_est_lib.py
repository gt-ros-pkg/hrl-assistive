#!/usr/bin/env python
import sys
import os
import numpy as np
import cPickle as pkl
import random

# ROS
import roslib; roslib.load_manifest('autobed_physical_trainer')

# Graphics
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Machine Learning
from scipy import ndimage
from scipy.ndimage.filters import gaussian_filter
## from skimage import data, color, exposure
from sklearn.decomposition import PCA

# HRL libraries
import hrl_lib.util as ut
import pickle
roslib.load_manifest('hrl_lib')
from hrl_lib.util import load_pickle


MAT_WIDTH = 0.762 #metres
MAT_HEIGHT = 1.854 #metres
MAT_HALF_WIDTH = MAT_WIDTH/2 
NUMOFTAXELS_X = 64#73 #taxels
NUMOFTAXELS_Y = 27#30 
INTER_SENSOR_DISTANCE = 0.0286#metres
LOW_TAXEL_THRESH_X = 0
LOW_TAXEL_THRESH_Y = 0
HIGH_TAXEL_THRESH_X = (NUMOFTAXELS_X - 1) 
HIGH_TAXEL_THRESH_Y = (NUMOFTAXELS_Y - 1) 

def world_to_mat(w_data, p_world_mat, R_world_mat):

    '''Converts a vector in the world frame to a vector in the map frame.
    Depends on the calibration of the MoCap room. Be sure to change this 
    when the calibration file changes. This function mainly helps in
    visualizing the joint coordinates on the pressure mat.
    Input: w_data: which is a 3 x 1 vector in the world frame'''
    #The homogenous transformation matrix from world to mat
    #O_m_w = np.matrix([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    O_m_w = np.matrix(np.reshape(R_world_mat, (3, 3)))
    p_mat_world = O_m_w.dot(-np.asarray(p_world_mat))
    B_m_w = np.concatenate((O_m_w, p_mat_world.T), axis=1)
    last_row = np.array([[0, 0, 0, 1]])
    B_m_w = np.concatenate((B_m_w, last_row), axis=0)
    w_data = np.hstack([w_data, np.ones([len(w_data),1])])
        
    #Convert input to the mat frame vector
    m_data = B_m_w * w_data.T

    return np.squeeze(np.asarray(m_data[:3,:].T))
