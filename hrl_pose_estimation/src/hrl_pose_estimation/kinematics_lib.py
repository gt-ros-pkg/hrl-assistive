#!/usr/bin/env python
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import *

import cPickle as pkl
import random
from scipy import ndimage
import scipy.stats as ss
from scipy.misc import imresize
from scipy.ndimage.interpolation import zoom
from skimage.feature import hog
from skimage import data, color, exposure

from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn import svm, linear_model, decomposition, kernel_ridge, neighbors
from sklearn import metrics, cross_validation
from sklearn.utils import shuffle


import pickle
from hrl_lib.util import load_pickle
import rospkg
import roslib
import rospy
import tf.transformations as tft


# PyTorch libraries
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchvision import transforms
from torch.autograd import Variable

MAT_WIDTH = 0.762 #metres
MAT_HEIGHT = 1.854 #metres
MAT_HALF_WIDTH = MAT_WIDTH/2
NUMOFTAXELS_X = 84#73 #taxels
NUMOFTAXELS_Y = 47#30
INTER_SENSOR_DISTANCE = 0.0286#metres
LOW_TAXEL_THRESH_X = 0
LOW_TAXEL_THRESH_Y = 0
HIGH_TAXEL_THRESH_X = (NUMOFTAXELS_X - 1)
HIGH_TAXEL_THRESH_Y = (NUMOFTAXELS_Y - 1)




class KinematicsLib():


    def forward_arm_kinematics(self, images, torso_lengths, angles):
        #print images.shape, 'images shape'
        #print torso_lengths.shape, 'torso lengths shape'
        #print angles.shape, 'angles shape'


        try: #this happens when we call it from the convnet which has a tensor var we need to convert
            images = images.data.numpy()
            torso_lengths = torso_lengths.data.numpy()
            angles = angles.data.numpy()
            lengths = torso_lengths[:, 3:11] / 100
            angles = angles[:, 0:8]
            angles[:, 2:4] = angles[:, 2:4] * 0.25
            torso = torso_lengths[:, 0:3] / 100

        except: #this happens when we call it from create dataset and sent it a numpy array instead of a tensor var
            lengths = np.expand_dims(torso_lengths[3:11], axis = 0)
            angles = np.expand_dims(angles[0:8], axis = 0)
            torso = np.expand_dims(torso_lengths[0:3], axis = 0)


        #print lengths[0,:], angles[0,:], torso[0,:]
        targets = np.zeros((images.shape[0], 15))
        queue = np.zeros((5,3))
        for set in range(0, images.shape[0]):
            try: #this happens when the images are actually the images
                bedangle = images[set, 1, 10, 10]
            except: #this happens when you just throw an angle in there
                bedangle = images
            TrelO = tft.identity_matrix()
            TprelT = tft.identity_matrix()

            rSrelTp = tft.rotation_matrix(np.deg2rad(bedangle * 0.75), (1, 0, 0))
            lSrelTp = tft.rotation_matrix(np.deg2rad(bedangle * 0.75), (1, 0, 0))

            TrelO[0:3, 3] = torso[set, :].T
            TprelT[2, 3] = -lengths[set, 0]
            rSrelTp[0, 3] = -lengths[set, 2]
            rSrelTp[1, 3] = lengths[set, 1] * np.cos(np.deg2rad(bedangle * 0.75))
            rSrelTp[2, 3] = lengths[set, 1] * np.sin(np.deg2rad(bedangle * 0.75))
            lSrelTp[0, 3] = lengths[set, 2]
            lSrelTp[1, 3] = lengths[set, 1] * np.cos(np.deg2rad(bedangle * 0.75))
            lSrelTp[2, 3] = lengths[set, 1] * np.sin(np.deg2rad(bedangle * 0.75))

            rErelrS = np.matmul(np.matmul(tft.rotation_matrix(-np.deg2rad(-angles[set, 4] + 180), (0, 1, 0)),
                                          tft.rotation_matrix(np.deg2rad(180 + angles[set, 2]), (0, 0, 1))),
                                tft.rotation_matrix(np.deg2rad((angles[set, 0]) + 90 + angles[set, 4]), (-1, 0, 0)))
            lErellS = np.matmul(np.matmul(tft.rotation_matrix(-np.deg2rad(angles[set, 5] + 180), (0, 1, 0)),
                                          tft.rotation_matrix(np.deg2rad(-angles[set, 3]), (0, 0, 1))),
                                tft.rotation_matrix(np.deg2rad((-angles[set, 1]) + 90 - angles[set, 5]), (-1, 0, 0)))


            Pr_SE = np.matmul(rErelrS, np.array([[lengths[set, 4]], [0], [0], [1]]))
            Pl_SE = np.matmul(lErellS, np.array([[lengths[set, 5]], [0], [0], [1]]))

            rErelrS[0:3, 3] = -Pr_SE[0:3, 0]
            lErellS[0:3, 3] = -Pl_SE[0:3, 0]

            # rHrelrE = np.matmul(tft.rotation_matrix(np.deg2rad(-(angles[0])), (-1,0,0)),tft.rotation_matrix(np.deg2rad(angles[6]), (0, 0, 1)))
            rHrelrE = tft.rotation_matrix(np.deg2rad(angles[set, 6]), (0, 0, 1))
            lHrellE = tft.rotation_matrix(np.deg2rad(angles[set, 7]), (0, 0, 1))

            # print rHrelrE, 'rhrelre'
            # print lHrellE, 'lhrelle'

            Pr_EH = np.matmul(rHrelrE, np.array([[lengths[set, 6]], [0], [0], [1]]))
            Pl_EH = np.matmul(lHrellE, np.array([[lengths[set, 7]], [0], [0], [1]]))

            Pr_SE = -Pr_SE
            Pr_SE[3, 0] = 1
            Pl_SE = -Pl_SE
            Pl_SE[3, 0] = 1

            queue[0, :] = torso[set, :]

            pred_r_E = np.matrix(np.matmul(np.matmul(np.matmul(TrelO, TprelT), rSrelTp), Pr_SE))
            queue[1, :] = np.squeeze(pred_r_E[0:3, 0].T)

            pred_l_E = np.matrix(np.matmul(np.matmul(np.matmul(TrelO, TprelT), lSrelTp), Pl_SE))
            queue[2, :] = np.squeeze(pred_l_E[0:3, 0].T)

            pred_r_H = np.matrix(np.matmul(np.matmul(np.matmul(np.matmul(TrelO, TprelT), rSrelTp), rErelrS), Pr_EH))
            queue[3, :] = np.squeeze(pred_r_H[0:3, 0].T)

            pred_l_H = np.matrix(np.matmul(np.matmul(np.matmul(np.matmul(TrelO, TprelT), lSrelTp), lErellS), Pl_EH))
            queue[4, :] = np.squeeze(pred_l_H[0:3, 0].T)
            #print targets, 'targets'

            targets[set, :] = queue.flatten()

            #print targets[set, :], 'target set'

        scores = targets*1000
        return scores

