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

    def get_bed_distance(self, images, targets, bedangle = None):
        mat_size = (NUMOFTAXELS_X, NUMOFTAXELS_Y)

        try:
            images = images.data.numpy()
            targets = targets.data.numpy()/1000
            try:
                test = targets.shape[2]
            except:
                targets = np.reshape(targets, (targets.shape[0], targets.shape[1]/3, 3))
            bedangle = images[:, -1, 10, 10]


        except:
            images = np.expand_dims(images, axis = 0)
            targets = np.reshape(targets, (10, 3))
            targets = np.expand_dims(targets, axis = 0)
            bedangle = np.expand_dims(bedangle, axis = 0)



        distances = np.zeros((images.shape[0], targets.shape[1]))
        queue_frame = np.zeros((targets.shape[0], targets.shape[1], 4))
        queue_head = np.zeros((targets.shape[0], targets.shape[1], 4))

        # get the shortest distance from the main frame of the bed. it's just the z.
        queue_frame[:, :, 0] = targets[:, :, 2]

        # get the shortest distance from the head of the bed. you have to rotate about the bending point.


        By = (51) * 0.0286 - 0.0286 * 3 * np.sin(np.deg2rad(np.expand_dims(bedangle[:], axis = 1)))
        queue_head[:, :, 0] = targets[:, :, 2] * np.cos(np.deg2rad(np.expand_dims(bedangle[:], axis = 1))) - (targets[:, :,1] - By) * np.sin(np.deg2rad(np.expand_dims(bedangle[:], axis = 1) ))

        # Get the distance off the side of the bed.  The bed is 27 pixels wide, so things over hanging this are added
        queue_frame[:, :, 1] = (-targets[:, :, 0] + 10 * 0.0286).clip(min=0)
        queue_frame[:, :, 1] = queue_frame[:, :, 1] + (targets[:, :, 0] - 37 * 0.0286).clip(min=0)
        queue_head[:, :, 1] = np.copy(queue_frame[:, :, 1])  # the x distance does not depend on bed angle.

        # Now take the Euclidean for each frame and head set
        queue_frame[:, :, 2] = np.sqrt(np.square(queue_frame[:, :, 0]) + np.square(queue_frame[:, :, 1]))
        queue_head[:, :, 2] = np.sqrt(np.square(queue_head[:, :, 0]) + np.square(queue_head[:, :, 1]))

        # however, there is still a problem.  We should zero out the distance if the x position is within the bounds
        # of the pressure mat and the z is negative. This just indicates the person is pushing into the mat.
        queue_frame[:, :, 3] = (queue_frame[:, :, 2] - queue_frame[:, :, 0] - queue_frame[:, :, 1] * 1000000).clip(min=0)
        queue_head[:, :, 3] = (queue_head[:, :, 2] - queue_head[:, :, 0] - queue_frame[:, :, 1] * 1000000).clip(min=0)
        queue_frame[:, :, 2] = (queue_frame[:, :, 2] - queue_frame[:, :, 3]).clip(min=0)  # corrected Euclidean
        queue_head[:, :, 2] = (queue_head[:, :, 2] - queue_head[:, :, 3]).clip(min=0)  # corrected Euclidean

        # Now take the minimum of the Euclideans from the head and the frame planes
        distances[:, :] = np.amin([queue_frame[:, :, 2], queue_head[:, :, 2]], axis=0)


        return distances




    def forward_upper_kinematics(self, images, torso_lengths, angles):
        #print images.shape, 'images shape'
        #print torso_lengths.shape, 'torso lengths shape'
        #print angles.shape, 'angles shape'


        try: #this happens when we call it from the convnet which has a tensor var we need to convert
            images = images.data.numpy()
            torso_lengths = torso_lengths.data.numpy()
            angles = angles.data.numpy()
            lengths = torso_lengths[:, 3:12] / 100
            angles = angles[:, 0:10]
            angles[:, 2:4] = angles[:, 2:4] * 0.25
            torso = torso_lengths[:, 0:3] / 100

        except: #this happens when we call it from create dataset and sent it a numpy array instead of a tensor var
            lengths = np.expand_dims(torso_lengths[3:12], axis = 0)
            angles = np.expand_dims(angles[0:10], axis = 0)
            torso = np.expand_dims(torso_lengths[0:3], axis = 0)


        #print lengths[0,:], angles[0,:], torso[0,:]
        targets = np.zeros((images.shape[0], 18))
        queue = np.zeros((6,3))
        for set in range(0, images.shape[0]):
            try: #this happens when the images are actually the images
                bedangle = images[set, -1, 10, 10]
            except: #this happens when you just throw an angle in there
                bedangle = images
            TrelO = tft.identity_matrix()
            TprelT = tft.identity_matrix()

            NrelTp = tft.rotation_matrix(np.deg2rad(bedangle * 0.75), (1, 0, 0))

            TrelO[0:3, 3] = torso[set, :].T
            TprelT[2, 3] = -lengths[set, 0]
            NrelTp[1, 3] = lengths[set, 1] * np.cos(np.deg2rad(bedangle * 0.75))
            NrelTp[2, 3] = lengths[set, 1] * np.sin(np.deg2rad(bedangle * 0.75))

            rSrelN = tft.identity_matrix()
            rSrelN[0, 3] = -lengths[set, 2]

            lSrelN = tft.identity_matrix()
            lSrelN[0, 3] = lengths[set, 2]



            Pr_NS = np.array([[-lengths[set, 2]], [0], [0], [1]])
            Pl_NS = np.array([[lengths[set, 2]], [0], [0], [1]])

            HrelN = np.matmul(tft.rotation_matrix(np.deg2rad(-angles[set, 8] + 90), (0, 0, 1)),
                                          tft.rotation_matrix(np.deg2rad(angles[set, 9] - 90), (0, 1, 0)))
            rErelrS = np.matmul(np.matmul(tft.rotation_matrix(-np.deg2rad(-angles[set, 4] + 180), (0, 1, 0)),
                                          tft.rotation_matrix(np.deg2rad(180 + angles[set, 2]), (0, 0, 1))),
                                tft.rotation_matrix(np.deg2rad((angles[set, 0]) + 90 + angles[set, 4]), (-1, 0, 0)))
            lErellS = np.matmul(np.matmul(tft.rotation_matrix(-np.deg2rad(angles[set, 5] + 180), (0, 1, 0)),
                                          tft.rotation_matrix(np.deg2rad(-angles[set, 3]), (0, 0, 1))),
                                tft.rotation_matrix(np.deg2rad((-angles[set, 1]) + 90 - angles[set, 5]), (-1, 0, 0)))


            P_NH = np.matmul(HrelN, np.array([[lengths[set, 8]], [0], [0], [1]]))

            #print angles[set, 8:10]
            #print lengths[set, 8]
            #print P_NH,'Pnh'

            Pr_SE = np.matmul(rErelrS, np.array([[lengths[set, 4]], [0], [0], [1]]))
            Pl_SE = np.matmul(lErellS, np.array([[lengths[set, 5]], [0], [0], [1]]))



            HrelN[0:3, 3] = -P_NH[0:3, 0]
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


            pred_H = np.matrix(np.matmul(np.matmul(np.matmul(TrelO, TprelT), NrelTp), P_NH))
            queue[0, :] = np.squeeze(pred_H[0:3, 0].T)

            queue[1, :] = torso[set, :]

            pred_r_S = np.matrix(np.matmul(np.matmul(np.matmul(TrelO, TprelT), NrelTp), Pr_NS))
            #print pred_r_S, 'pred r S'

            pred_l_S = np.matrix(np.matmul(np.matmul(np.matmul(TrelO, TprelT), NrelTp), Pl_NS))
            #print pred_l_S, 'pred l S'

            pred_r_E = np.matrix(np.matmul(np.matmul(np.matmul(np.matmul(TrelO, TprelT), NrelTp), rSrelN), Pr_SE))
            queue[2, :] = np.squeeze(pred_r_E[0:3, 0].T)

            pred_l_E = np.matrix(np.matmul(np.matmul(np.matmul(np.matmul(TrelO, TprelT), NrelTp), lSrelN), Pl_SE))
            queue[3, :] = np.squeeze(pred_l_E[0:3, 0].T)

            pred_r_H = np.matrix(np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(TrelO, TprelT), NrelTp), rSrelN), rErelrS), Pr_EH))
            queue[4, :] = np.squeeze(pred_r_H[0:3, 0].T)

            pred_l_H = np.matrix(np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(TrelO, TprelT), NrelTp), lSrelN), lErellS), Pl_EH))
            queue[5, :] = np.squeeze(pred_l_H[0:3, 0].T)
            #print targets, 'targets'

            targets[set, :] = queue.flatten()

            #print targets[set, :], 'target set'

        targets = targets*1000
        return targets


    def forward_lower_kinematics(self, images, torso_lengths, angles):
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
                bedangle = images[set, -1, 10, 10]
            except: #this happens when you just throw an angle in there
                bedangle = images
            TrelO = tft.identity_matrix()
            TprelT = tft.identity_matrix()

            BrelTp = tft.identity_matrix()

            TrelO[0:3, 3] = torso[set, :].T
            TprelT[2, 3] = -lengths[set, 0]
            BrelTp[1, 3] = -lengths[set, 1]


            rGrelB = tft.identity_matrix()
            rGrelB[0, 3] = -lengths[set, 2]

            lGrelB = tft.identity_matrix()
            lGrelB[0, 3] = lengths[set, 2]



            Pr_BG = np.array([[-lengths[set, 2]], [0], [0], [1]])
            Pl_BG = np.array([[lengths[set, 2]], [0], [0], [1]])


            rKrelrG = np.matmul(np.matmul(tft.rotation_matrix(-np.deg2rad(angles[set, 4]), (0, 1, 0)),
                                          tft.rotation_matrix(np.deg2rad(180 + angles[set, 2]), (0, 0, 1))),
                                tft.rotation_matrix(np.deg2rad((-angles[set, 0]) + 90 + angles[set, 4]), (-1, 0, 0)))
            lKrellG = np.matmul(np.matmul(tft.rotation_matrix(-np.deg2rad(angles[set, 5] - 180), (0, 1, 0)),
                                          tft.rotation_matrix(np.deg2rad(-angles[set, 3]), (0, 0, 1))),
                                tft.rotation_matrix(np.deg2rad((-angles[set, 1]) + 90 - angles[set, 5]), (-1, 0, 0)))


            Pr_GK = np.matmul(rKrelrG, np.array([[lengths[set, 4]], [0], [0], [1]]))
            Pl_GK = np.matmul(lKrellG, np.array([[lengths[set, 5]], [0], [0], [1]]))

            rKrelrG[0:3, 3] = -Pr_GK[0:3, 0]
            lKrellG[0:3, 3] = -Pl_GK[0:3, 0]

            # rHrelrE = np.matmul(tft.rotation_matrix(np.deg2rad(-(angles[0])), (-1,0,0)),tft.rotation_matrix(np.deg2rad(angles[6]), (0, 0, 1)))
            rArelrK = tft.rotation_matrix(np.deg2rad(angles[set, 6]), (0, 0, 1))
            lArellK = tft.rotation_matrix(np.deg2rad(angles[set, 7]), (0, 0, 1))

            # print rHrelrE, 'rhrelre'
            # print lHrellE, 'lhrelle'

            Pr_KA = np.matmul(rArelrK, np.array([[lengths[set, 6]], [0], [0], [1]]))
            Pl_KA = np.matmul(lArellK, np.array([[lengths[set, 7]], [0], [0], [1]]))

            Pr_GK = -Pr_GK
            Pr_GK[3, 0] = 1
            Pl_GK = -Pl_GK
            Pl_GK[3, 0] = 1

            queue[0, :] = torso[set, :]

            pred_r_G = np.matrix(np.matmul(np.matmul(np.matmul(TrelO, TprelT), BrelTp), Pr_BG))
            #print pred_r_G, 'pred r G'

            pred_l_G = np.matrix(np.matmul(np.matmul(np.matmul(TrelO, TprelT), BrelTp), Pl_BG))
            #print pred_l_G, 'pred l G'

            pred_r_K = np.matrix(np.matmul(np.matmul(np.matmul(np.matmul(TrelO, TprelT), BrelTp), rGrelB), Pr_GK))
            queue[1, :] = np.squeeze(pred_r_K[0:3, 0].T)

            pred_l_K = np.matrix(np.matmul(np.matmul(np.matmul(np.matmul(TrelO, TprelT), BrelTp), lGrelB), Pl_GK))
            queue[2, :] = np.squeeze(pred_l_K[0:3, 0].T)
            #print pred_l_K, 'pred_l_K'

            pred_r_A = np.matrix(np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(TrelO, TprelT), BrelTp), rGrelB), rKrelrG), Pr_KA))
            queue[3, :] = np.squeeze(pred_r_A[0:3, 0].T)

            pred_l_A = np.matrix(np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(TrelO, TprelT), BrelTp), lGrelB), lKrellG), Pl_KA))
            queue[4, :] = np.squeeze(pred_l_A[0:3, 0].T)
            #print targets, 'targets'

            targets[set, :] = queue.flatten()

            #print targets[set, :], 'target set'

        targets = targets*1000
        return targets



    def forward_kinematics_pytorch(self, images_v, torso_lengths_angles_v, targets_v, loss_vector_type, kincons_v = None, prior_cascade = None, forward_only = False, body_side = None):

        test_ground_truth = False
        loop = False
        pseudotargets = None

        if loss_vector_type == 'upper_angles':
            torso_lengths_angles_v = torso_lengths_angles_v.unsqueeze(0)
            torso_lengths_angles_v = torso_lengths_angles_v.unsqueeze(0)
            torso_lengths_angles_v = F.pad(torso_lengths_angles_v, (0, 15, 0, 0)) #make more space for the head and arm x, y, z coords
            # print torso_lengths_angles_v.size()

            torso_lengths_angles_v = torso_lengths_angles_v.squeeze(0)
            torso_lengths_angles_v = torso_lengths_angles_v.squeeze(0)

            if test_ground_truth == True:
                # print kincons_v.size()
                torso_lengths_angles_v[:, 0:19] = kincons_v #upper arm angles and lengths
                torso_lengths_angles_v[:, 19:22] = targets_v[:, 3:6] / 1000 #torso
                # print targets_v[0, :], 'targets'
                # print torso_lengths_angles_v[0, :]

            torso_lengths_angles = torso_lengths_angles_v.data.numpy()
            # print torso_lengths_angles.shape

            # lengths_v = torso_lengths_angles_v[:, 3:11] # raw lengths coming out of network are in m.
            angles = torso_lengths_angles[:, 0:10] * 100
            # angles_v = torso_lengths_angles_v[:, 11:19] * 100 * np.pi / 180 #raw signal coming out of network is in hundredths of degrees
            lengths = torso_lengths_angles[:, 10:19]
            # torso_v = torso_lengths_angles_v[:, 0:3] #raw positions coming out of network are in m
            torso = torso_lengths_angles[:, 19:22]

            # targets = np.zeros((images.shape[0], 15)) #WE CAN"T DO THIS WITH THE TENSOR VARIBLE !!! FIX THIS
            # queue = np.zeros((5,3))
            # images_v[:, -1, 10, 10] = images_v[:, -1, 10, 10] * np.pi / 180
            images = images_v.data.numpy() * np.pi / 180
            bedangle = images[:, -1, 10, 10] * 0.75


            if loop == True:

                for set in range(0, torso_lengths_angles.shape[0]):
                    #print bedangle[set] * 180 / np.pi, 'bedangle converted to degrees'



                    TrelO = tft.identity_matrix()
                    TrelO[0:3, 3] = torso[set, :].T

                    TprelT = tft.identity_matrix()
                    TprelT[2, 3] = -lengths[set, 0]

                    NrelTp = tft.rotation_matrix(bedangle[set], (1, 0, 0))
                    NrelTp[1, 3] = lengths[set, 1] * np.cos(bedangle[set])
                    NrelTp[2, 3] = lengths[set, 1] * np.sin(bedangle[set])

                    HrelN = np.matmul(tft.rotation_matrix(np.deg2rad(-angles[set, 8] + 90), (0, 0, 1)),
                                      tft.rotation_matrix(np.deg2rad(angles[set, 9] - 90), (0, 1, 0)))

                    rSrelN = tft.identity_matrix()
                    rSrelN[0, 3] = -lengths[set, 2]

                    lSrelN = tft.identity_matrix()
                    lSrelN[0, 3] = lengths[set, 2]

                    Pr_NS = np.array([[-lengths[set, 2]], [0], [0], [1]])
                    Pl_NS = np.array([[lengths[set, 2]], [0], [0], [1]])

                    P_NH = np.matmul(HrelN, np.array([[lengths[set, 8]], [0], [0], [1]]))


                    print P_NH, 'Pnh'

                    pred_H = np.matrix(np.matmul(np.matmul(np.matmul(TrelO, TprelT), NrelTp), P_NH))

                    print pred_H
                    print targets_v[set,0:3]




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

                    # print lHrellE, 'lhrelle'

                    Pr_EH = np.matmul(rHrelrE, np.array([[lengths[set, 6]], [0], [0], [1]]))
                    Pl_EH = np.matmul(lHrellE, np.array([[lengths[set, 7]], [0], [0], [1]]))

                    Pr_SE = -Pr_SE
                    Pr_SE[3, 0] = 1
                    Pl_SE = -Pl_SE
                    Pl_SE[3, 0] = 1


                    #print np.matrix(np.matmul(np.matmul(TprelT, NrelTp), Pr_NS)), 'blah'

                    #print lengths[set, :]
                    #print torso[set, :]

                    pred_r_S = np.matrix(np.matmul(np.matmul(np.matmul(TrelO, TprelT), NrelTp), Pr_NS))
                    #print pred_r_S, 'pred r S'


                    #print out the right shoulder in vectorized form
                    #print torso_lengths_angles_v[set, 16] - torso_lengths_angles_v[set, 2]
                    #print torso_lengths_angles_v[set, 17] + torso_lengths_angles_v[set, 1] * np.cos(float(bedangle[set]))
                    #print torso_lengths_angles_v[set, 18] - torso_lengths_angles_v[set, 0] + torso_lengths_angles_v[set, 1] * np.sin(float(bedangle[set]))



                    pred_l_S = np.matrix(np.matmul(np.matmul(np.matmul(TrelO, TprelT), NrelTp), Pl_NS))
                    #print pred_l_S, 'pred l S'


                    #print out the left shoulder in vectorized form
                    #print torso_lengths_angles_v[set, 16] + torso_lengths_angles_v[set, 2]
                    #print torso_lengths_angles_v[set, 17] + torso_lengths_angles_v[set, 1] * np.cos(float(bedangle[set]))
                    #print torso_lengths_angles_v[set, 18] - torso_lengths_angles_v[set, 0] + torso_lengths_angles_v[set, 1] * np.sin(float(bedangle[set]))


                    torso_lengths_angles_v[set, 22] = float(torso_lengths_angles[set, 19]) + float(torso_lengths_angles[set, 18])*((np.pi/2. - torso_lengths_angles_v[set, 8]*100*np.pi/180).cos())*((-np.pi/2. + torso_lengths_angles_v[set, 9]*100*np.pi/180).cos())
                    torso_lengths_angles_v[set, 23] = float(torso_lengths_angles[set, 20]) + float(torso_lengths_angles[set, 18])*((np.pi/2. - torso_lengths_angles_v[set, 8]*100*np.pi/180).sin())*((-np.pi/2. + torso_lengths_angles_v[set, 9]*100*np.pi/180).cos())*np.cos(float(bedangle[set])) + float(torso_lengths_angles[set, 18])*((-np.pi/2. + torso_lengths_angles_v[set, 9]*100*np.pi/180).sin())*np.sin(float(bedangle[set]))+ float(torso_lengths_angles[set, 11]) * np.cos(float(bedangle[set]))
                    torso_lengths_angles_v[set, 24] = float(torso_lengths_angles[set, 21]) - float(torso_lengths_angles[set, 10]) + float(torso_lengths_angles[set, 18])*((np.pi/2. - torso_lengths_angles_v[set, 8]*100*np.pi/180).sin())*((-np.pi/2. + torso_lengths_angles_v[set, 9]*100*np.pi/180).cos())*np.sin(float(bedangle[set])) - float(torso_lengths_angles[set, 18])*((-np.pi/2. + torso_lengths_angles_v[set, 9]*100*np.pi/180).sin())*np.cos(float(bedangle[set]))+ float(torso_lengths_angles[set, 11]) * np.sin(float(bedangle[set]))

                    pred_r_E = np.matrix(np.matmul(np.matmul(np.matmul(np.matmul(TrelO, TprelT), NrelTp), rSrelN), Pr_SE))

                    #right elbow in vectorized form
                    torso_lengths_angles_v[set, 25] = float(torso_lengths_angles[set, 19]) - float(torso_lengths_angles[set, 10]) + (-(np.pi + torso_lengths_angles_v[set, 4]*100*np.pi/180).cos()*float(torso_lengths_angles[set, 12])*(np.pi + torso_lengths_angles_v[set, 2]*100*np.pi/180).cos())
                    torso_lengths_angles_v[set, 26] = float(torso_lengths_angles[set, 20]) + (-float(torso_lengths_angles[set, 12])*(np.pi + torso_lengths_angles_v[set, 2]*100*np.pi/180).sin())*np.cos(float(bedangle[set])) - ((np.pi + torso_lengths_angles_v[set, 4] * 100 * np.pi / 180).sin() * float(torso_lengths_angles[set, 12]) * (np.pi + torso_lengths_angles_v[set, 2] * 100 * np.pi / 180).cos())*np.sin(float(bedangle[set]))+ float(torso_lengths_angles[set, 9]) * np.cos(float(bedangle[set]))
                    torso_lengths_angles_v[set, 27] = float(torso_lengths_angles[set, 21]) - float(torso_lengths_angles[set, 8]) + (-float(torso_lengths_angles[set, 12])*(np.pi + torso_lengths_angles_v[set, 2]*100*np.pi/180).sin())*np.sin(float(bedangle[set])) + ((np.pi + torso_lengths_angles_v[set, 4] * 100 * np.pi / 180).sin() * float(torso_lengths_angles[set, 12]) * (np.pi + torso_lengths_angles_v[set, 2] * 100 * np.pi / 180).cos())*np.cos(float(bedangle[set]))+ float(torso_lengths_angles[set, 9]) * np.sin(float(bedangle[set]))


                    #print Pr_SE, 'prse'
                    #Pr_SE in vectorized form
                    #print (-(np.pi + torso_lengths_angles_v[set, 15]*100*np.pi/180).cos()*torso_lengths_angles_v[set, 7]*(np.pi + torso_lengths_angles_v[set, 13]*100*np.pi/180).cos()), 'prse x'
                    #print (-torso_lengths_angles_v[set, 7]*(np.pi + torso_lengths_angles_v[set, 13]*100*np.pi/180).sin()), 'prse y'
                    #print ((np.pi + torso_lengths_angles_v[set, 15] * 100 * np.pi / 180).sin() * torso_lengths_angles_v[set, 7] * (np.pi + torso_lengths_angles_v[set, 13] * 100 * np.pi / 180).cos()), 'prse z'



                    pred_l_E = np.matrix(np.matmul(np.matmul(np.matmul(np.matmul(TrelO, TprelT), NrelTp), lSrelN), Pl_SE))

                    #left elbow in vectorized form
                    torso_lengths_angles_v[set, 28] = float(torso_lengths_angles[set, 19]) + float(torso_lengths_angles[set, 11]) + (-(np.pi + torso_lengths_angles_v[set, 5]*100*np.pi/180).cos()*float(torso_lengths_angles[set, 13])*(-torso_lengths_angles_v[set, 3]*100*np.pi/180).cos())
                    torso_lengths_angles_v[set, 29] = float(torso_lengths_angles[set, 20]) + (-float(torso_lengths_angles[set, 13])*(-torso_lengths_angles_v[set, 3]*100*np.pi/180).sin())*np.cos(float(bedangle[set])) + ((np.pi + torso_lengths_angles_v[set, 5] * 100 * np.pi / 180).sin() * float(torso_lengths_angles[set, 13]) * (- torso_lengths_angles_v[set, 3] * 100 * np.pi / 180).cos())*np.sin(float(bedangle[set]))+ float(torso_lengths_angles[set, 9]) * np.cos(float(bedangle[set]))
                    torso_lengths_angles_v[set, 30] = float(torso_lengths_angles[set, 21]) - float(torso_lengths_angles[set, 8]) + (-float(torso_lengths_angles[set, 13])*(- torso_lengths_angles_v[set, 3]*100*np.pi/180).sin())*np.sin(float(bedangle[set])) - ((np.pi + torso_lengths_angles_v[set, 5] * 100 * np.pi / 180).sin() * float(torso_lengths_angles[set, 13]) * (- torso_lengths_angles_v[set, 3] * 100 * np.pi / 180).cos())*np.cos(float(bedangle[set]))+ float(torso_lengths_angles[set, 9]) * np.sin(float(bedangle[set]))


                    pred_r_H = np.matrix(np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(TrelO, TprelT), NrelTp), rSrelN), rErelrS), Pr_EH))

                    #right hand in vectorized form
                    torso_lengths_angles_v[set, 31] = float(torso_lengths_angles[set, 19]) - float(torso_lengths_angles[set, 10]) + ((1.8 - torso_lengths_angles_v[set, 4])*100*np.pi/180).cos()*(((1.8 + torso_lengths_angles_v[set, 2])*100*np.pi/180).cos()*((torso_lengths_angles_v[set, 6] * 100 * np.pi / 180).cos() * float(torso_lengths_angles[set, 14]) - float(torso_lengths_angles[set, 12])) - ((1.8 + torso_lengths_angles_v[set, 2])*100*np.pi/180).sin()*((torso_lengths_angles_v[set, 0] + torso_lengths_angles_v[set, 4] + 0.9)*100*np.pi/180).cos() * (torso_lengths_angles_v[set, 6] * 100 * np.pi / 180).sin() * float(torso_lengths_angles[set, 14])) + ((1.8 - torso_lengths_angles_v[set, 4])*100*np.pi/180).sin() * ((torso_lengths_angles_v[set, 0] + torso_lengths_angles_v[set, 4] + 0.9)*100*np.pi/180).sin() * (torso_lengths_angles_v[set, 6] * 100 * np.pi / 180).sin() * float(torso_lengths_angles[set, 14])
                    torso_lengths_angles_v[set, 32] = float(torso_lengths_angles[set, 20]) + (((1.8 + torso_lengths_angles_v[set, 2]) * 100 * np.pi / 180).sin() * ((torso_lengths_angles_v[set, 6] * 100 * np.pi / 180).cos() * float(torso_lengths_angles[set, 14]) - float(torso_lengths_angles[set, 12])) + ((1.8 + torso_lengths_angles_v[set, 2]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[set, 0] + torso_lengths_angles_v[set, 4] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[set, 6] * 100 * np.pi / 180).sin() * float(torso_lengths_angles[set, 14]))*np.cos(float(bedangle[set])) - (((1.8 - torso_lengths_angles_v[set, 4])*100*np.pi/180).sin()*(((1.8 + torso_lengths_angles_v[set, 2])*100*np.pi/180).cos()*((torso_lengths_angles_v[set, 6] * 100 * np.pi / 180).cos() * float(torso_lengths_angles[set, 14]) - float(torso_lengths_angles[set, 12])) - ((1.8 + torso_lengths_angles_v[set, 2])*100*np.pi/180).sin()*((torso_lengths_angles_v[set, 0] + torso_lengths_angles_v[set, 4] + 0.9)*100*np.pi/180).cos() * (torso_lengths_angles_v[set, 6] * 100 * np.pi / 180).sin() * float(torso_lengths_angles[set, 14])) - ((1.8 - torso_lengths_angles_v[set, 4])*100*np.pi/180).cos() * ((torso_lengths_angles_v[set, 0] + torso_lengths_angles_v[set, 4] + 0.9)*100*np.pi/180).sin() * (torso_lengths_angles_v[set, 6] * 100 * np.pi / 180).sin() * float(torso_lengths_angles[set, 14])) * np.sin(float(bedangle[set])) + float(torso_lengths_angles[set, 9]) * np.cos(float(bedangle[set]))
                    torso_lengths_angles_v[set, 33] = float(torso_lengths_angles[set, 21]) - float(torso_lengths_angles[set, 8]) + (((1.8 + torso_lengths_angles_v[set, 2]) * 100 * np.pi / 180).sin() * ((torso_lengths_angles_v[set, 6] * 100 * np.pi / 180).cos() * float(torso_lengths_angles[set, 14]) - float(torso_lengths_angles[set, 12])) + ((1.8 + torso_lengths_angles_v[set, 2]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[set, 0] + torso_lengths_angles_v[set, 4] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[set, 6] * 100 * np.pi / 180).sin() * float(torso_lengths_angles[set, 14]))*np.sin(float(bedangle[set])) + (((1.8 - torso_lengths_angles_v[set, 4])*100*np.pi/180).sin()*(((1.8 + torso_lengths_angles_v[set, 2])*100*np.pi/180).cos()*((torso_lengths_angles_v[set, 6] * 100 * np.pi / 180).cos() * float(torso_lengths_angles[set, 14]) - float(torso_lengths_angles[set, 12])) - ((1.8 + torso_lengths_angles_v[set, 2])*100*np.pi/180).sin()*((torso_lengths_angles_v[set, 0] + torso_lengths_angles_v[set, 4] + 0.9)*100*np.pi/180).cos() * (torso_lengths_angles_v[set, 6] * 100 * np.pi / 180).sin() * float(torso_lengths_angles[set, 14])) - ((1.8 - torso_lengths_angles_v[set, 4])*100*np.pi/180).cos() * ((torso_lengths_angles_v[set, 0] + torso_lengths_angles_v[set, 4] + 0.9)*100*np.pi/180).sin() * (torso_lengths_angles_v[set, 6] * 100 * np.pi / 180).sin() * float(torso_lengths_angles[set, 14])) * np.cos(float(bedangle[set])) + float(torso_lengths_angles[set, 9]) * np.sin(float(bedangle[set]))


                    pred_l_H = np.matrix(np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(TrelO, TprelT), NrelTp), lSrelN), lErellS), Pl_EH))

                    #left hand in vectorized form
                    torso_lengths_angles_v[set, 34] = float(torso_lengths_angles[set, 19]) + float(torso_lengths_angles[set, 11]) + ((1.8 + torso_lengths_angles_v[set, 5])*100*np.pi/180).cos()*(((-torso_lengths_angles_v[set, 3])*100*np.pi/180).cos()*((torso_lengths_angles_v[set, 7] * 100 * np.pi / 180).cos() * float(torso_lengths_angles[set, 15]) - float(torso_lengths_angles[set, 13])) - ((-torso_lengths_angles_v[set, 3])*100*np.pi/180).sin()*((-torso_lengths_angles_v[set, 1] - torso_lengths_angles_v[set, 5] + 0.9)*100*np.pi/180).cos() * (torso_lengths_angles_v[set, 7] * 100 * np.pi / 180).sin() * float(torso_lengths_angles[set, 15])) + ((1.8 + torso_lengths_angles_v[set, 5])*100*np.pi/180).sin() * ((-torso_lengths_angles_v[set, 1] - torso_lengths_angles_v[set, 5] + 0.9)*100*np.pi/180).sin() * (torso_lengths_angles_v[set, 7] * 100 * np.pi / 180).sin() * float(torso_lengths_angles[set, 15])
                    torso_lengths_angles_v[set, 35] = float(torso_lengths_angles[set, 20]) + (((-torso_lengths_angles_v[set, 3]) * 100 * np.pi / 180).sin() * ((torso_lengths_angles_v[set, 7] * 100 * np.pi / 180).cos() * float(torso_lengths_angles[set, 15]) - float(torso_lengths_angles[set, 13])) + ((-torso_lengths_angles_v[set, 3]) * 100 * np.pi / 180).cos() * ((-torso_lengths_angles_v[set, 1] - torso_lengths_angles_v[set, 5] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[set, 7] * 100 * np.pi / 180).sin() * float(torso_lengths_angles[set, 15]))*np.cos(float(bedangle[set])) - (((1.8 + torso_lengths_angles_v[set, 5])*100*np.pi/180).sin()*(((-torso_lengths_angles_v[set, 3])*100*np.pi/180).cos()*((torso_lengths_angles_v[set, 7] * 100 * np.pi / 180).cos() * float(torso_lengths_angles[set, 15]) - float(torso_lengths_angles[set, 13])) - ((-torso_lengths_angles_v[set, 3])*100*np.pi/180).sin()*((-torso_lengths_angles_v[set, 1] - torso_lengths_angles_v[set, 5] + 0.9)*100*np.pi/180).cos() * (torso_lengths_angles_v[set, 7] * 100 * np.pi / 180).sin() * float(torso_lengths_angles[set, 15])) - ((1.8 + torso_lengths_angles_v[set, 5])*100*np.pi/180).cos() * ((-torso_lengths_angles_v[set, 1] - torso_lengths_angles_v[set, 5] + 0.9)*100*np.pi/180).sin() * (torso_lengths_angles_v[set, 7] * 100 * np.pi / 180).sin() * float(torso_lengths_angles[set, 15])) * np.sin(float(bedangle[set])) + float(torso_lengths_angles[set, 9]) * np.cos(float(bedangle[set]))
                    torso_lengths_angles_v[set, 36] = float(torso_lengths_angles[set, 21]) - float(torso_lengths_angles[set, 8]) + (((-torso_lengths_angles_v[set, 3]) * 100 * np.pi / 180).sin() * ((torso_lengths_angles_v[set, 7] * 100 * np.pi / 180).cos() * float(torso_lengths_angles[set, 15]) - float(torso_lengths_angles[set, 13])) + ((-torso_lengths_angles_v[set, 3]) * 100 * np.pi / 180).cos() * ((-torso_lengths_angles_v[set, 1] - torso_lengths_angles_v[set, 5] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[set, 7] * 100 * np.pi / 180).sin() * float(torso_lengths_angles[set, 15]))*np.sin(float(bedangle[set])) + (((1.8 + torso_lengths_angles_v[set, 5])*100*np.pi/180).sin()*(((-torso_lengths_angles_v[set, 3])*100*np.pi/180).cos()*((torso_lengths_angles_v[set, 7] * 100 * np.pi / 180).cos() * float(torso_lengths_angles[set, 15]) - float(torso_lengths_angles[set, 13])) - ((-torso_lengths_angles_v[set, 3])*100*np.pi/180).sin()*((-torso_lengths_angles_v[set, 1] - torso_lengths_angles_v[set, 5] + 0.9)*100*np.pi/180).cos() * (torso_lengths_angles_v[set, 7] * 100 * np.pi / 180).sin() * float(torso_lengths_angles[set, 15])) - ((1.8 + torso_lengths_angles_v[set, 5])*100*np.pi/180).cos() * ((-torso_lengths_angles_v[set, 1] - torso_lengths_angles_v[set, 5] + 0.9)*100*np.pi/180).sin() * (torso_lengths_angles_v[set, 7] * 100 * np.pi / 180).sin() * float(torso_lengths_angles[set, 15])) * np.cos(float(bedangle[set])) + float(torso_lengths_angles[set, 9]) * np.sin(float(bedangle[set]))



                    #print torso_lengths_angles_v[set, 20:31].data.numpy() * 1000
                    # print torso_lengths_angles_v[set, :]
                    #print targets_v[set, 4:15].data.numpy(), 'l elbow'
                    # print kincons_v[set, 0:8]




                    #print pred_l_H
                    #queue[4, :] = np.squeeze(pred_l_H[0:3, 0].T)
                    # print targets, 'targets'

                    #targets[set, :] = queue.flatten()

                    # print targets[set, :], 'target set'



            elif loop == False:

                torso_lengths_angles = Variable(torch.Tensor(torso_lengths_angles))
                bedangle = Variable(torch.Tensor(bedangle))


                #head in vectorized form
                torso_lengths_angles_v[:, 22] = torso_lengths_angles_v[:, 19] + torso_lengths_angles[:, 18] * ((np.pi / 2. - torso_lengths_angles_v[:, 8] * 100 * np.pi / 180).cos()) * ((-np.pi / 2. + torso_lengths_angles_v[:, 9] * 100 * np.pi / 180).cos())
                torso_lengths_angles_v[:, 23] = torso_lengths_angles_v[:, 20] + torso_lengths_angles[:, 18] * ((np.pi / 2. - torso_lengths_angles_v[:, 8] * 100 * np.pi / 180).sin()) * ((-np.pi / 2. + torso_lengths_angles_v[:, 9] * 100 * np.pi / 180).cos()) * bedangle[:].cos() + torso_lengths_angles[:, 18] * ((-np.pi / 2. + torso_lengths_angles_v[:, 9] * 100 * np.pi / 180).sin()) * bedangle[:].sin() + torso_lengths_angles[:, 11] * bedangle[:].cos()
                torso_lengths_angles_v[:, 24] = torso_lengths_angles_v[:, 21] - torso_lengths_angles[:, 10] + torso_lengths_angles[:, 18] * ((np.pi / 2. - torso_lengths_angles_v[:, 8] * 100 * np.pi / 180).sin()) * ((-np.pi / 2. + torso_lengths_angles_v[:, 9] * 100 * np.pi / 180).cos()) * bedangle[:].sin() - torso_lengths_angles[:, 18] * ((-np.pi / 2. + torso_lengths_angles_v[:, 9] * 100 * np.pi / 180).sin()) * bedangle[:].cos() + torso_lengths_angles[:, 11] * bedangle[:].sin()

                # right elbow in vectorized form
                torso_lengths_angles_v[:, 25] = torso_lengths_angles_v[:, 19] - torso_lengths_angles[:, 12] + (-(np.pi + torso_lengths_angles_v[:, 4] * 100 * np.pi / 180).cos() * torso_lengths_angles[:, 14] * (np.pi + torso_lengths_angles_v[:, 2] * 100 * np.pi / 180).cos())
                torso_lengths_angles_v[:, 26] = torso_lengths_angles_v[:, 20] + (-torso_lengths_angles[:, 14] * (np.pi + torso_lengths_angles_v[:, 2] * 100 * np.pi / 180).sin()) * bedangle[:].cos() - ((np.pi + torso_lengths_angles_v[:, 4] * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 14] * (np.pi + torso_lengths_angles_v[:, 2] * 100 * np.pi / 180).cos()) * bedangle[:].sin() + torso_lengths_angles[:, 11] * bedangle[:].cos()
                torso_lengths_angles_v[:, 27] = torso_lengths_angles_v[:, 21] - torso_lengths_angles[:, 10] + (-torso_lengths_angles[:, 14] * (np.pi + torso_lengths_angles_v[:, 2] * 100 * np.pi / 180).sin()) * bedangle[:].sin() + ((np.pi + torso_lengths_angles_v[:, 4] * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 14] * (np.pi + torso_lengths_angles_v[:, 2] * 100 * np.pi / 180).cos()) * bedangle[:].cos() + torso_lengths_angles[:, 11] * bedangle[:].sin()

                # left elbow in vectorized form
                torso_lengths_angles_v[:, 28] = torso_lengths_angles_v[:, 19] + torso_lengths_angles[:, 13] + (-(np.pi + torso_lengths_angles_v[:, 5] * 100 * np.pi / 180).cos() * torso_lengths_angles[:, 15] * (-torso_lengths_angles_v[:, 3] * 100 * np.pi / 180).cos())
                torso_lengths_angles_v[:, 29] = torso_lengths_angles_v[:, 20] + (-torso_lengths_angles[:, 15] * (-torso_lengths_angles_v[:, 3] * 100 * np.pi / 180).sin()) * bedangle[:].cos() + ((np.pi + torso_lengths_angles_v[:, 5] * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 15] * (- torso_lengths_angles_v[:, 3] * 100 * np.pi / 180).cos()) * bedangle[:].sin() + torso_lengths_angles[:, 11] * bedangle[:].cos()
                torso_lengths_angles_v[:, 30] = torso_lengths_angles_v[:, 21] - torso_lengths_angles[:, 10] + (-torso_lengths_angles[:, 15] * (- torso_lengths_angles_v[:, 3] * 100 * np.pi / 180).sin()) * bedangle[:].sin() - ((np.pi +torso_lengths_angles_v[:, 5] * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 15] * (- torso_lengths_angles_v[:, 3] * 100 * np.pi / 180).cos()) * bedangle[:].cos() + torso_lengths_angles[:, 11] * bedangle[:].sin()

                # right hand in vectorized form
                torso_lengths_angles_v[:, 31] = torso_lengths_angles_v[:, 19] - torso_lengths_angles[:, 12] + ((1.8 - torso_lengths_angles_v[:, 4]) * 100 * np.pi / 180).cos() * (((1.8 + torso_lengths_angles_v[:, 2]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[:, 6] * 100 * np.pi / 180).cos() *  torso_lengths_angles[:, 16] -  torso_lengths_angles[:, 14]) - ((1.8 + torso_lengths_angles_v[:, 2]) * 100 * np.pi / 180).sin() * ((torso_lengths_angles_v[:, 0] + torso_lengths_angles_v[:, 4] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[:, 6] * 100 * np.pi / 180).sin() *  torso_lengths_angles[:, 16]) + ((1.8 -torso_lengths_angles_v[:, 4]) * 100 * np.pi / 180).sin() * ((torso_lengths_angles_v[:, 0] + torso_lengths_angles_v[:, 4] + 0.9) * 100 * np.pi / 180).sin() * (torso_lengths_angles_v[:, 6] * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 16]
                torso_lengths_angles_v[:, 32] = torso_lengths_angles_v[:, 20] + (((1.8 + torso_lengths_angles_v[:, 2]) * 100 * np.pi / 180).sin() * ((torso_lengths_angles_v[:, 6] * 100 * np.pi / 180).cos() *  torso_lengths_angles[:, 16] -  torso_lengths_angles[:, 14]) + ((1.8 + torso_lengths_angles_v[:, 2]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[:, 0] + torso_lengths_angles_v[:, 4] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[:, 6] * 100 * np.pi / 180).sin() *torso_lengths_angles[:, 16]) * bedangle[:].cos() - (((1.8 - torso_lengths_angles_v[:, 4]) * 100 * np.pi / 180).sin() * (((1.8 + torso_lengths_angles_v[:, 2]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[:, 6] * 100 * np.pi / 180).cos() *torso_lengths_angles[:, 16] -  torso_lengths_angles[:, 14]) - ((1.8 + torso_lengths_angles_v[:, 2]) * 100 * np.pi / 180).sin() * ((torso_lengths_angles_v[:, 0] + torso_lengths_angles_v[:, 4] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[:, 6] * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 16]) - ((1.8 - torso_lengths_angles_v[:, 4]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[:, 0] +torso_lengths_angles_v[:, 4] + 0.9) * 100 * np.pi / 180).sin() * (torso_lengths_angles_v[:, 6] * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 16]) * bedangle[:].sin() + torso_lengths_angles[:, 11] * bedangle[:].cos()
                torso_lengths_angles_v[:, 33] = torso_lengths_angles_v[:, 21] - torso_lengths_angles[:, 10] + (((1.8 + torso_lengths_angles_v[:, 2]) * 100 * np.pi / 180).sin() * ((torso_lengths_angles_v[:, 6] * 100 * np.pi / 180).cos() *  torso_lengths_angles[:, 16] - torso_lengths_angles[:, 14]) + ((1.8 + torso_lengths_angles_v[:, 2]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[:, 0] + torso_lengths_angles_v[:, 4] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[:, 6] * 100 * np.pi / 180).sin() *torso_lengths_angles[:, 16]) * bedangle[:].sin() + (((1.8 - torso_lengths_angles_v[:, 4]) * 100 * np.pi / 180).sin() * (((1.8 + torso_lengths_angles_v[:, 2]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[:, 6] * 100 * np.pi / 180).cos() * torso_lengths_angles[:, 16] -  torso_lengths_angles[:, 14]) - ((1.8 + torso_lengths_angles_v[:, 2]) * 100 * np.pi / 180).sin() * ((torso_lengths_angles_v[:, 0] + torso_lengths_angles_v[:, 4] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[:, 6] * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 16]) - ((1.8 - torso_lengths_angles_v[:, 4]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[:, 0] + torso_lengths_angles_v[:, 4] + 0.9) * 100 * np.pi / 180).sin() * (torso_lengths_angles_v[:, 6] * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 16]) * bedangle[:].cos() + torso_lengths_angles[:, 11] * bedangle[:].sin()

                # left hand in vectorized form
                torso_lengths_angles_v[:, 34] = torso_lengths_angles_v[:, 19] + torso_lengths_angles[:, 13] + ((1.8 + torso_lengths_angles_v[:, 5]) * 100 * np.pi / 180).cos() * (((-torso_lengths_angles_v[:, 3]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[:, 7] * 100 * np.pi / 180).cos() *  torso_lengths_angles[:, 17] - torso_lengths_angles[:, 15]) - ((-torso_lengths_angles_v[:, 3]) * 100 * np.pi / 180).sin() * ((-torso_lengths_angles_v[:, 1] - torso_lengths_angles_v[:, 5] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[:, 7] * 100 * np.pi / 180).sin() *  torso_lengths_angles[:, 17]) + ((1.8 +torso_lengths_angles_v[:, 5]) * 100 * np.pi / 180).sin() * ((-torso_lengths_angles_v[:, 1] -torso_lengths_angles_v[:, 5] + 0.9) * 100 * np.pi / 180).sin() * (torso_lengths_angles_v[:, 7] * 100 * np.pi / 180).sin() *  torso_lengths_angles[:, 17]
                torso_lengths_angles_v[:, 35] = torso_lengths_angles_v[:, 20] + (((-torso_lengths_angles_v[:, 3]) * 100 * np.pi / 180).sin() * ((torso_lengths_angles_v[:, 7] * 100 * np.pi / 180).cos() * torso_lengths_angles[:, 17] -  torso_lengths_angles[:, 15]) + ((-torso_lengths_angles_v[:, 3]) * 100 * np.pi / 180).cos() * ((-torso_lengths_angles_v[:, 1] - torso_lengths_angles_v[:, 5] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[:, 7] * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 17]) * bedangle[:].cos() - (((1.8 + torso_lengths_angles_v[:, 5]) * 100 * np.pi / 180).sin() * (((-torso_lengths_angles_v[:, 3]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[:, 7] * 100 * np.pi / 180).cos() *  torso_lengths_angles[:, 17] - torso_lengths_angles[:, 15]) - ((-torso_lengths_angles_v[:, 3]) * 100 * np.pi / 180).sin() * ((-torso_lengths_angles_v[:, 1] -torso_lengths_angles_v[:, 5] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[:, 7] * 100 * np.pi / 180).sin() *torso_lengths_angles[:, 17]) - ((1.8 + torso_lengths_angles_v[:, 5]) * 100 * np.pi / 180).cos() * ((-torso_lengths_angles_v[:, 1] - torso_lengths_angles_v[:, 5] + 0.9) * 100 * np.pi / 180).sin() * (torso_lengths_angles_v[:, 7] * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 17]) * bedangle[:].sin() + torso_lengths_angles[:, 11] * bedangle[:].cos()
                torso_lengths_angles_v[:, 36] = torso_lengths_angles_v[:, 21] - torso_lengths_angles[:, 10] + (((-torso_lengths_angles_v[:, 3]) * 100 * np.pi / 180).sin() * ((torso_lengths_angles_v[:, 7] * 100 * np.pi / 180).cos() *  torso_lengths_angles[:, 17] -torso_lengths_angles[:, 15]) + ((-torso_lengths_angles_v[:, 3]) * 100 * np.pi / 180).cos() * ((-torso_lengths_angles_v[:, 1] - torso_lengths_angles_v[:, 5] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[:, 7] * 100 * np.pi / 180).sin() *torso_lengths_angles[:, 17]) * bedangle[:].sin() + (((1.8 + torso_lengths_angles_v[:, 5]) * 100 * np.pi / 180).sin() * (((-torso_lengths_angles_v[:, 3]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[:, 7] * 100 * np.pi / 180).cos() *  torso_lengths_angles[:, 17] -torso_lengths_angles[:, 15]) - ((-torso_lengths_angles_v[:, 3]) * 100 * np.pi / 180).sin() * ((-torso_lengths_angles_v[:, 1] -torso_lengths_angles_v[:, 5] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[:, 7] * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 17]) - ((1.8 + torso_lengths_angles_v[:, 5]) * 100 * np.pi / 180).cos() * ((-torso_lengths_angles_v[:, 1] - torso_lengths_angles_v[:, 5] + 0.9) * 100 * np.pi / 180).sin() * (torso_lengths_angles_v[:, 7] * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 17]) * bedangle[:].cos() + torso_lengths_angles[:, 11] * bedangle[:].sin()


            torso_lengths_angles_v = torso_lengths_angles_v.unsqueeze(0)
            torso_lengths_angles_v = torso_lengths_angles_v.unsqueeze(0)
            torso_lengths_angles_v = F.pad(torso_lengths_angles_v, (-10, 0, 0, 0)) #cut off the angles
            torso_lengths_angles_v = torso_lengths_angles_v.squeeze(0)
            torso_lengths_angles_v = torso_lengths_angles_v.squeeze(0)


        elif loss_vector_type == 'angles':
            torso_lengths_angles_v = torso_lengths_angles_v.unsqueeze(0)
            torso_lengths_angles_v = torso_lengths_angles_v.unsqueeze(0)
            torso_lengths_angles_v = F.pad(torso_lengths_angles_v, (0, 27, 0, 0)) #make more room for head, arms, and legs x, y, z coords.  torso already is in the network.
            # print torso_lengths_angles_v.size()

            torso_lengths_angles_v = torso_lengths_angles_v.squeeze(0)
            torso_lengths_angles_v = torso_lengths_angles_v.squeeze(0)

            if test_ground_truth == True:
                # print kincons_v.size()
                torso_lengths_angles_v[:, 0:18] = kincons_v[:, 0:18] #this is the upper angles, lower angles, upper lengths, lower lengths in that order
                torso_lengths_angles_v[:, 20:37] = kincons_v[:, 18:35]
                torso_lengths_angles_v[:, 37:40] = targets_v[:, 3:6] / 1000 #this is the torso x, y, z coords
                # print targets_v[0, :], 'targets'
                # print torso_lengths_angles_v[0, :]

            torso_lengths_angles = torso_lengths_angles_v.data.numpy()
            # print torso_lengths_angles.shape

            images = images_v.data.numpy() * np.pi / 180
            bedangle = images[:, -1, 10, 10] * 0.75

            if loop == True:
                pass

            elif loop == False:

                torso_lengths_angles = Variable(torch.Tensor(torso_lengths_angles))
                bedangle = Variable(torch.Tensor(bedangle))

                angle_noise = False  # add noise to the output of the convolutions.  Only add it to the non-zero outputs, because most are zero.
                if angle_noise == True:
                    x = np.arange(-6, 6)
                    xU, xL = x + 0.5, x - 0.5
                    prob = ss.norm.cdf(xU, scale=2) - ss.norm.cdf(xL, scale=2)  # scale is the standard deviation using a cumulative density function
                    prob = prob / prob.sum()  # normalize the probabilities so their sum is 1
                    image_noise = np.random.choice(x, size=(1, 17), p=prob) / 100.
                    image_noise = Variable(torch.Tensor(image_noise), volatile=True)
                    #print image_noise.size()
                    #print torso_lengths_angles_v[:, 0:17].size()

                    torso_lengths_angles_v[:, 0:17] = torch.add(torso_lengths_angles_v[:, 0:17], image_noise)
                    #print torso_lengths_angles_v[:, 0:17]


                #print torso_lengths_angles_v[0, 17], torso_lengths_angles_v[0, 18], torso_lengths_angles_v.size()

                torso_lengths_angles_v[:, 0] = torch.clamp(torso_lengths_angles_v[:, 0], -1.8, 1.8)
                torso_lengths_angles_v[:, 1] = torch.clamp(torso_lengths_angles_v[:, 1], -1.8, 1.8)
                torso_lengths_angles_v[:, 2] = torch.clamp(torso_lengths_angles_v[:, 2], -1.35, 1.35)
                torso_lengths_angles_v[:, 3] = torch.clamp(torso_lengths_angles_v[:, 3], -1.35, 1.35)
                torso_lengths_angles_v[:, 4] = torch.clamp(torso_lengths_angles_v[:, 4], -1.35, 1.35)
                torso_lengths_angles_v[:, 5] = torch.clamp(torso_lengths_angles_v[:, 5], -1.35, 1.35)
                torso_lengths_angles_v[:, 6] = torch.clamp(torch.add(torso_lengths_angles_v[:, 6], 1.5), 0.4, 1.8)
                torso_lengths_angles_v[:, 7] = torch.clamp(torch.add(torso_lengths_angles_v[:, 7], 1.5), 0.4, 1.8)
                #torso_lengths_angles_v[:, 8] = torch.clamp(torso_lengths_angles_v[:, 6], -1.8, 1.8)
                #torso_lengths_angles_v[:, 9] = torch.clamp(torso_lengths_angles_v[:, 7], -1.5, 1.5)
                torso_lengths_angles_v[:, 10] = torch.clamp(torso_lengths_angles_v[:, 10], -1.8, 1.8)
                torso_lengths_angles_v[:, 11] = torch.clamp(torso_lengths_angles_v[:, 11], -1.8, 1.8)
                torso_lengths_angles_v[:, 12] = torch.clamp(torch.add(torso_lengths_angles_v[:, 12], -0.6), -1.8, 0.)
                torso_lengths_angles_v[:, 13] = torch.clamp(torch.add(torso_lengths_angles_v[:, 13], -0.6), -1.8, 0.)
                torso_lengths_angles_v[:, 14] = torch.clamp(torch.add(torso_lengths_angles_v[:, 14], -0.6), -1.35, 1.35)
                torso_lengths_angles_v[:, 15] = torch.clamp(torch.add(torso_lengths_angles_v[:, 15], -0.6), -1.35, 1.35)
                torso_lengths_angles_v[:, 16] = torch.clamp(torch.add(torso_lengths_angles_v[:, 16], 1.5), 0.4, 1.8)
                torso_lengths_angles_v[:, 17] = torch.clamp(torch.add(torso_lengths_angles_v[:, 17], 1.5), 0.4, 1.8)

                torso_lengths_angles_v[:, 18] = torch.clamp(torch.add(torso_lengths_angles_v[:, 18], 0.2), -0.5, 1.1) #torso angle for upper
                torso_lengths_angles_v[:, 19] = torch.clamp(torch.add(torso_lengths_angles_v[:, 19], 0.0), -0.5, 0.5) #torso angle for lower

                torso_lengths_angles_v[:, 20] = torch.add(torso_lengths_angles_v[:, 20], 0.1)
                torso_lengths_angles_v[:, 21] = torch.add(torso_lengths_angles_v[:, 21], 0.26)
                torso_lengths_angles_v[:, 22] = torch.add(torso_lengths_angles_v[:, 22], 0.17)
                torso_lengths_angles_v[:, 23] = torch.add(torso_lengths_angles_v[:, 23], 0.17)
                torso_lengths_angles_v[:, 24] = torch.add(torso_lengths_angles_v[:, 24], 0.28)
                torso_lengths_angles_v[:, 25] = torch.add(torso_lengths_angles_v[:, 25], 0.28)
                torso_lengths_angles_v[:, 26] = torch.add(torso_lengths_angles_v[:, 26], 0.19)
                torso_lengths_angles_v[:, 27] = torch.add(torso_lengths_angles_v[:, 27], 0.19)
                torso_lengths_angles_v[:, 28] = torch.add(torso_lengths_angles_v[:, 28], 0.28)
                torso_lengths_angles_v[:, 29] = torch.add(torso_lengths_angles_v[:, 29], 0.14)
                torso_lengths_angles_v[:, 30] = torch.add(torso_lengths_angles_v[:, 30], 0.19)
                torso_lengths_angles_v[:, 31] = torch.add(torso_lengths_angles_v[:, 31], 0.10)
                torso_lengths_angles_v[:, 32] = torch.add(torso_lengths_angles_v[:, 32], 0.10)
                torso_lengths_angles_v[:, 33] = torch.add(torso_lengths_angles_v[:, 33], 0.40)
                torso_lengths_angles_v[:, 34] = torch.add(torso_lengths_angles_v[:, 34], 0.40)
                torso_lengths_angles_v[:, 35] = torch.add(torso_lengths_angles_v[:, 35], 0.30)
                torso_lengths_angles_v[:, 36] = torch.add(torso_lengths_angles_v[:, 36], 0.30)
                torso_lengths_angles_v[:, 37] = torch.add(torso_lengths_angles_v[:, 37], 0.6)
                torso_lengths_angles_v[:, 38] = torch.add(torso_lengths_angles_v[:, 38], 1.3)
                torso_lengths_angles_v[:, 39] = torch.add(torso_lengths_angles_v[:, 39], 0.1)


                #head in vectorized form
                torso_lengths_angles_v[:, 40] = torso_lengths_angles_v[:, 37] + torso_lengths_angles[:, 28] * ((np.pi / 2. - torso_lengths_angles_v[:, 8] * 100 * np.pi / 180).cos()) * ((-np.pi / 2. + torso_lengths_angles_v[:, 9] * 100 * np.pi / 180).cos())
                torso_lengths_angles_v[:, 41] = torso_lengths_angles_v[:, 38] + torso_lengths_angles[:, 28] * ((np.pi / 2. - torso_lengths_angles_v[:, 8] * 100 * np.pi / 180).sin()) * ((-np.pi / 2. + torso_lengths_angles_v[:, 9] * 100 * np.pi / 180).cos()) * (0. + torso_lengths_angles_v[:, 18]).cos() + torso_lengths_angles[:, 28] * ((-np.pi / 2. + torso_lengths_angles_v[:, 9] * 100 * np.pi / 180).sin()) * (0. + torso_lengths_angles_v[:, 18]).sin() + torso_lengths_angles[:, 21] * (0. + torso_lengths_angles_v[:, 18]).cos()
                torso_lengths_angles_v[:, 42] = torso_lengths_angles_v[:, 39] - torso_lengths_angles[:, 20] + torso_lengths_angles[:, 28] * ((np.pi / 2. - torso_lengths_angles_v[:, 8] * 100 * np.pi / 180).sin()) * ((-np.pi / 2. + torso_lengths_angles_v[:, 9] * 100 * np.pi / 180).cos()) * (0. + torso_lengths_angles_v[:, 18]).sin() - torso_lengths_angles[:, 28] * ((-np.pi / 2. + torso_lengths_angles_v[:, 9] * 100 * np.pi / 180).sin()) * (0. + torso_lengths_angles_v[:, 18]).cos() + torso_lengths_angles[:, 21] * (0. + torso_lengths_angles_v[:, 18]).sin()

                # right elbow in vectorized form
                torso_lengths_angles_v[:, 43] = torso_lengths_angles_v[:, 37] - torso_lengths_angles[:, 22] + (-(np.pi + torso_lengths_angles_v[:, 4] * 100 * np.pi / 180).cos() * torso_lengths_angles[:, 24] * (np.pi + torso_lengths_angles_v[:, 2] * 100 * np.pi / 180).cos())
                torso_lengths_angles_v[:, 44] = torso_lengths_angles_v[:, 38] + (-torso_lengths_angles[:, 24] * (np.pi + torso_lengths_angles_v[:, 2] * 100 * np.pi / 180).sin()) * (0. + torso_lengths_angles_v[:, 18]).cos() - ((np.pi + torso_lengths_angles_v[:, 4] * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 24] * (np.pi + torso_lengths_angles_v[:, 2] * 100 * np.pi / 180).cos()) * (0. + torso_lengths_angles_v[:, 18]).sin() + torso_lengths_angles[:, 21] * (0. + torso_lengths_angles_v[:, 18]).cos()
                torso_lengths_angles_v[:, 45] = torso_lengths_angles_v[:, 39] - torso_lengths_angles[:, 20] + (-torso_lengths_angles[:, 24] * (np.pi + torso_lengths_angles_v[:, 2] * 100 * np.pi / 180).sin()) * (0. + torso_lengths_angles_v[:, 18]).sin() + ((np.pi + torso_lengths_angles_v[:, 4] * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 24] * (np.pi + torso_lengths_angles_v[:, 2] * 100 * np.pi / 180).cos()) * (0. + torso_lengths_angles_v[:, 18]).cos() + torso_lengths_angles[:, 21] * (0. + torso_lengths_angles_v[:, 18]).sin()

                # left elbow in vectorized form
                torso_lengths_angles_v[:, 46] = torso_lengths_angles_v[:, 37] + torso_lengths_angles[:, 23] + (-(np.pi + torso_lengths_angles_v[:, 5] * 100 * np.pi / 180).cos() * torso_lengths_angles[:, 25] * (-torso_lengths_angles_v[:, 3] * 100 * np.pi / 180).cos())
                torso_lengths_angles_v[:, 47] = torso_lengths_angles_v[:, 38] + (-torso_lengths_angles[:, 25] * (-torso_lengths_angles_v[:, 3] * 100 * np.pi / 180).sin()) * (0. + torso_lengths_angles_v[:, 18]).cos() + ((np.pi + torso_lengths_angles_v[:, 5] * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 25] * (- torso_lengths_angles_v[:, 3] * 100 * np.pi / 180).cos()) * (0. + torso_lengths_angles_v[:, 18]).sin() + torso_lengths_angles[:, 21] * (0. + torso_lengths_angles_v[:, 18]).cos()
                torso_lengths_angles_v[:, 48] = torso_lengths_angles_v[:, 39] - torso_lengths_angles[:, 20] + (-torso_lengths_angles[:, 25] * (- torso_lengths_angles_v[:, 3] * 100 * np.pi / 180).sin()) * (0. + torso_lengths_angles_v[:, 18]).sin() - ((np.pi +torso_lengths_angles_v[:, 5] * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 25] * (- torso_lengths_angles_v[:, 3] * 100 * np.pi / 180).cos()) * (0. + torso_lengths_angles_v[:, 18]).cos() + torso_lengths_angles[:, 21] * (0. + torso_lengths_angles_v[:, 18]).sin()

                # right hand in vectorized form
                torso_lengths_angles_v[:, 49] = torso_lengths_angles_v[:, 37] - torso_lengths_angles[:, 22] + ((1.8 - torso_lengths_angles_v[:, 4]) * 100 * np.pi / 180).cos() * (((1.8 + torso_lengths_angles_v[:, 2]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[:, 6] * 100 * np.pi / 180).cos() *  torso_lengths_angles[:, 26] -  torso_lengths_angles[:, 21]) - ((1.8 + torso_lengths_angles_v[:, 2]) * 100 * np.pi / 180).sin() * ((torso_lengths_angles_v[:, 0] + torso_lengths_angles_v[:, 4] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[:, 6] * 100 * np.pi / 180).sin() *  torso_lengths_angles[:, 26]) + ((1.8 -torso_lengths_angles_v[:, 4]) * 100 * np.pi / 180).sin() * ((torso_lengths_angles_v[:, 0] + torso_lengths_angles_v[:, 4] + 0.9) * 100 * np.pi / 180).sin() * (torso_lengths_angles_v[:, 6] * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 26]
                torso_lengths_angles_v[:, 50] = torso_lengths_angles_v[:, 38] + (((1.8 + torso_lengths_angles_v[:, 2]) * 100 * np.pi / 180).sin() * ((torso_lengths_angles_v[:, 6] * 100 * np.pi / 180).cos() *  torso_lengths_angles[:, 26] -  torso_lengths_angles[:, 21]) + ((1.8 + torso_lengths_angles_v[:, 2]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[:, 0] + torso_lengths_angles_v[:, 4] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[:, 6] * 100 * np.pi / 180).sin() *torso_lengths_angles[:, 26]) * (0. + torso_lengths_angles_v[:, 18]).cos() - (((1.8 - torso_lengths_angles_v[:, 4]) * 100 * np.pi / 180).sin() * (((1.8 + torso_lengths_angles_v[:, 2]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[:, 6] * 100 * np.pi / 180).cos() *torso_lengths_angles[:, 26] -  torso_lengths_angles[:, 21]) - ((1.8 + torso_lengths_angles_v[:, 2]) * 100 * np.pi / 180).sin() * ((torso_lengths_angles_v[:, 0] + torso_lengths_angles_v[:, 4] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[:, 6] * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 26]) - ((1.8 - torso_lengths_angles_v[:, 4]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[:, 0] +torso_lengths_angles_v[:, 4] + 0.9) * 100 * np.pi / 180).sin() * (torso_lengths_angles_v[:, 6] * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 26]) * (0. + torso_lengths_angles_v[:, 18]).sin() + torso_lengths_angles[:, 21] * (0. + torso_lengths_angles_v[:, 18]).cos()
                torso_lengths_angles_v[:, 51] = torso_lengths_angles_v[:, 39] - torso_lengths_angles[:, 20] + (((1.8 + torso_lengths_angles_v[:, 2]) * 100 * np.pi / 180).sin() * ((torso_lengths_angles_v[:, 6] * 100 * np.pi / 180).cos() *  torso_lengths_angles[:, 26] - torso_lengths_angles[:, 21]) + ((1.8 + torso_lengths_angles_v[:, 2]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[:, 0] + torso_lengths_angles_v[:, 4] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[:, 6] * 100 * np.pi / 180).sin() *torso_lengths_angles[:, 26]) * (0. + torso_lengths_angles_v[:, 18]).sin() + (((1.8 - torso_lengths_angles_v[:, 4]) * 100 * np.pi / 180).sin() * (((1.8 + torso_lengths_angles_v[:, 2]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[:, 6] * 100 * np.pi / 180).cos() * torso_lengths_angles[:, 26] -  torso_lengths_angles[:, 21]) - ((1.8 + torso_lengths_angles_v[:, 2]) * 100 * np.pi / 180).sin() * ((torso_lengths_angles_v[:, 0] + torso_lengths_angles_v[:, 4] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[:, 6] * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 26]) - ((1.8 - torso_lengths_angles_v[:, 4]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[:, 0] + torso_lengths_angles_v[:, 4] + 0.9) * 100 * np.pi / 180).sin() * (torso_lengths_angles_v[:, 6] * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 26]) * (0. + torso_lengths_angles_v[:, 18]).cos() + torso_lengths_angles[:, 21] * (0. + torso_lengths_angles_v[:, 18]).sin()

                # left hand in vectorized form
                torso_lengths_angles_v[:, 52] = torso_lengths_angles_v[:, 37] + torso_lengths_angles[:, 23] + ((1.8 + torso_lengths_angles_v[:, 5]) * 100 * np.pi / 180).cos() * (((-torso_lengths_angles_v[:, 3]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[:, 7] * 100 * np.pi / 180).cos() *  torso_lengths_angles[:, 27] - torso_lengths_angles[:, 25]) - ((-torso_lengths_angles_v[:, 3]) * 100 * np.pi / 180).sin() * ((-torso_lengths_angles_v[:, 1] - torso_lengths_angles_v[:, 5] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[:, 7] * 100 * np.pi / 180).sin() *  torso_lengths_angles[:, 27]) + ((1.8 +torso_lengths_angles_v[:, 5]) * 100 * np.pi / 180).sin() * ((-torso_lengths_angles_v[:, 1] -torso_lengths_angles_v[:, 5] + 0.9) * 100 * np.pi / 180).sin() * (torso_lengths_angles_v[:, 7] * 100 * np.pi / 180).sin() *  torso_lengths_angles[:, 27]
                torso_lengths_angles_v[:, 53] = torso_lengths_angles_v[:, 38] + (((-torso_lengths_angles_v[:, 3]) * 100 * np.pi / 180).sin() * ((torso_lengths_angles_v[:, 7] * 100 * np.pi / 180).cos() * torso_lengths_angles[:, 27] -  torso_lengths_angles[:, 25]) + ((-torso_lengths_angles_v[:, 3]) * 100 * np.pi / 180).cos() * ((-torso_lengths_angles_v[:, 1] - torso_lengths_angles_v[:, 5] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[:, 7] * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 27]) * (0. + torso_lengths_angles_v[:, 18]).cos() - (((1.8 + torso_lengths_angles_v[:, 5]) * 100 * np.pi / 180).sin() * (((-torso_lengths_angles_v[:, 3]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[:, 7] * 100 * np.pi / 180).cos() *  torso_lengths_angles[:, 27] - torso_lengths_angles[:, 25]) - ((-torso_lengths_angles_v[:, 3]) * 100 * np.pi / 180).sin() * ((-torso_lengths_angles_v[:, 1] -torso_lengths_angles_v[:, 5] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[:, 7] * 100 * np.pi / 180).sin() *torso_lengths_angles[:, 27]) - ((1.8 + torso_lengths_angles_v[:, 5]) * 100 * np.pi / 180).cos() * ((-torso_lengths_angles_v[:, 1] - torso_lengths_angles_v[:, 5] + 0.9) * 100 * np.pi / 180).sin() * (torso_lengths_angles_v[:, 7] * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 27]) * (0. + torso_lengths_angles_v[:, 18]).sin() + torso_lengths_angles[:, 21] * (0. + torso_lengths_angles_v[:, 18]).cos()
                torso_lengths_angles_v[:, 54] = torso_lengths_angles_v[:, 39] - torso_lengths_angles[:, 20] + (((-torso_lengths_angles_v[:, 3]) * 100 * np.pi / 180).sin() * ((torso_lengths_angles_v[:, 7] * 100 * np.pi / 180).cos() *  torso_lengths_angles[:, 27] -torso_lengths_angles[:, 25]) + ((-torso_lengths_angles_v[:, 3]) * 100 * np.pi / 180).cos() * ((-torso_lengths_angles_v[:, 1] - torso_lengths_angles_v[:, 5] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[:, 7] * 100 * np.pi / 180).sin() *torso_lengths_angles[:, 27]) * (0. + torso_lengths_angles_v[:, 18]).sin() + (((1.8 + torso_lengths_angles_v[:, 5]) * 100 * np.pi / 180).sin() * (((-torso_lengths_angles_v[:, 3]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[:, 7] * 100 * np.pi / 180).cos() *  torso_lengths_angles[:, 27] -torso_lengths_angles[:, 25]) - ((-torso_lengths_angles_v[:, 3]) * 100 * np.pi / 180).sin() * ((-torso_lengths_angles_v[:, 1] -torso_lengths_angles_v[:, 5] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[:, 7] * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 27]) - ((1.8 + torso_lengths_angles_v[:, 5]) * 100 * np.pi / 180).cos() * ((-torso_lengths_angles_v[:, 1] - torso_lengths_angles_v[:, 5] + 0.9) * 100 * np.pi / 180).sin() * (torso_lengths_angles_v[:, 7] * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 27]) * (0. + torso_lengths_angles_v[:, 18]).cos() + torso_lengths_angles[:, 21] * (0. + torso_lengths_angles_v[:, 18]).sin()


                # right knee in vectorized form
                torso_lengths_angles_v[:, 55] = torso_lengths_angles_v[:, 37] - torso_lengths_angles[:, 31] + ((torso_lengths_angles_v[:, 14] * 100 * np.pi / 180).cos() * torso_lengths_angles[:, 33] * (np.pi + torso_lengths_angles_v[:, 12] * 100 * np.pi / 180).cos())
                #torso_lengths_angles_v[:, 56] = torso_lengths_angles_v[:, 38] + (-torso_lengths_angles[:, 33] * (np.pi + torso_lengths_angles_v[:, 12] * 100 * np.pi / 180).sin()) - torso_lengths_angles[:, 30]
                #torso_lengths_angles_v[:, 57] = torso_lengths_angles_v[:, 39] - torso_lengths_angles[:, 29] + ((torso_lengths_angles_v[:, 14] * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 33] * (np.pi + torso_lengths_angles_v[:, 12] * 100 * np.pi / 180).cos())
                torso_lengths_angles_v[:, 56] = torso_lengths_angles_v[:, 38] + (-torso_lengths_angles[:, 33] * (np.pi + torso_lengths_angles_v[:, 12] * 100 * np.pi / 180).sin()) * (0. + torso_lengths_angles_v[:, 19]).cos() - ((torso_lengths_angles_v[:, 14] * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 33] * (np.pi + torso_lengths_angles_v[:, 12] * 100 * np.pi / 180).cos()) * (0. + torso_lengths_angles_v[:, 19]).sin() - torso_lengths_angles[:, 30] * (0. + torso_lengths_angles_v[:, 19]).cos()
                torso_lengths_angles_v[:, 57] = torso_lengths_angles_v[:, 39] - torso_lengths_angles[:, 29] + (-torso_lengths_angles[:, 33] * (np.pi + torso_lengths_angles_v[:, 12] * 100 * np.pi / 180).sin()) * (0. + torso_lengths_angles_v[:, 18]).sin() + ((torso_lengths_angles_v[:, 14] * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 33] * (np.pi + torso_lengths_angles_v[:, 12] * 100 * np.pi / 180).cos()) * (0. + torso_lengths_angles_v[:, 19]).cos() + torso_lengths_angles[:, 30] * (0. + torso_lengths_angles_v[:, 19]).sin()


                # left knee in vectorized form
                torso_lengths_angles_v[:, 58] = torso_lengths_angles_v[:, 37] + torso_lengths_angles[:, 32] + (-((-1.8 - torso_lengths_angles_v[:, 15]) * 100 * np.pi / 180).cos() * torso_lengths_angles[:, 34] * (-torso_lengths_angles_v[:, 13] * 100 * np.pi / 180).cos())
                #torso_lengths_angles_v[:, 59] = torso_lengths_angles_v[:, 38] + (-torso_lengths_angles[:, 34] * (-torso_lengths_angles_v[:, 13] * 100 * np.pi / 180).sin()) - torso_lengths_angles[:, 30]
                #torso_lengths_angles_v[:, 60] = torso_lengths_angles_v[:, 39] - torso_lengths_angles[:, 29] - (((-1.8 - torso_lengths_angles_v[:, 15]) * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 34] * (- torso_lengths_angles_v[:, 13] * 100 * np.pi / 180).cos())
                torso_lengths_angles_v[:, 59] = torso_lengths_angles_v[:, 38] + (-torso_lengths_angles[:, 34] * (-torso_lengths_angles_v[:, 13] * 100 * np.pi / 180).sin()) * (0. + torso_lengths_angles_v[:, 19]).cos() + (((-1.8 - torso_lengths_angles_v[:, 15]) * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 34] * (- torso_lengths_angles_v[:, 13] * 100 * np.pi / 180).cos()) * (0. + torso_lengths_angles_v[:, 19]).sin() - torso_lengths_angles[:, 30] * (0. + torso_lengths_angles_v[:, 19]).cos()
                torso_lengths_angles_v[:, 60] = torso_lengths_angles_v[:, 39] - torso_lengths_angles[:, 29] + (-torso_lengths_angles[:, 34] * (- torso_lengths_angles_v[:, 13] * 100 * np.pi / 180).sin()) * (0. + torso_lengths_angles_v[:, 19]).sin() - (((-1.8 - torso_lengths_angles_v[:, 15]) * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 34] * (- torso_lengths_angles_v[:, 13] * 100 * np.pi / 180).cos()) * (0. + torso_lengths_angles_v[:, 19]).cos()  + torso_lengths_angles[:, 30] * (0. + torso_lengths_angles_v[:, 19]).sin()

                # right ankle in vectorized form
                torso_lengths_angles_v[:, 61] = torso_lengths_angles_v[:, 37] - torso_lengths_angles[:, 31] + (-(torso_lengths_angles_v[:, 14]) * 100 * np.pi / 180).cos() * (((1.8 + torso_lengths_angles_v[:, 12]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[:, 16] * 100 * np.pi / 180).cos() *  torso_lengths_angles[:, 35] -  torso_lengths_angles[:, 33]) - ((1.8 + torso_lengths_angles_v[:, 12]) * 100 * np.pi / 180).sin() * ((-torso_lengths_angles_v[:, 10] + torso_lengths_angles_v[:, 14] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[:, 16] * 100 * np.pi / 180).sin() *  torso_lengths_angles[:, 35]) - ((torso_lengths_angles_v[:, 14]) * 100 * np.pi / 180).sin() * ((-torso_lengths_angles_v[:, 10] + torso_lengths_angles_v[:, 14] + 0.9) * 100 * np.pi / 180).sin() * (torso_lengths_angles_v[:, 16] * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 35]
                #torso_lengths_angles_v[:, 62] = torso_lengths_angles_v[:, 38] + (((1.8 + torso_lengths_angles_v[:, 12]) * 100 * np.pi / 180).sin() * ((torso_lengths_angles_v[:, 16] * 100 * np.pi / 180).cos() *  torso_lengths_angles[:, 35] -  torso_lengths_angles[:, 33]) + ((1.8 + torso_lengths_angles_v[:, 12]) * 100 * np.pi / 180).cos() * ((-torso_lengths_angles_v[:, 10] + torso_lengths_angles_v[:, 14] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[:, 16] * 100 * np.pi / 180).sin() *torso_lengths_angles[:, 35]) - torso_lengths_angles[:, 30]
                #torso_lengths_angles_v[:, 63] = torso_lengths_angles_v[:, 39] - torso_lengths_angles[:, 29] + (-(torso_lengths_angles_v[:, 14]) * 100 * np.pi / 180).sin() * (((1.8 + torso_lengths_angles_v[:, 12]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[:, 16] * 100 * np.pi / 180).cos() * torso_lengths_angles[:, 35] -  torso_lengths_angles[:, 33]) - ((1.8 + torso_lengths_angles_v[:, 12]) * 100 * np.pi / 180).sin() * ((-torso_lengths_angles_v[:, 10] + torso_lengths_angles_v[:, 14] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[:, 16] * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 35]) +  ((torso_lengths_angles_v[:, 14]) * 100 * np.pi / 180).cos() * ((-torso_lengths_angles_v[:, 10] + torso_lengths_angles_v[:, 14] + 0.9) * 100 * np.pi / 180).sin() * (torso_lengths_angles_v[:, 16] * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 35]

                torso_lengths_angles_v[:, 62] = torso_lengths_angles_v[:, 38] + (((1.8 + torso_lengths_angles_v[:, 12]) * 100 * np.pi / 180).sin() * ((torso_lengths_angles_v[:, 16] * 100 * np.pi / 180).cos() *  torso_lengths_angles[:, 35] -  torso_lengths_angles[:, 33]) + ((1.8 + torso_lengths_angles_v[:, 12]) * 100 * np.pi / 180).cos() * ((-torso_lengths_angles_v[:, 10] + torso_lengths_angles_v[:, 14] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[:, 16] * 100 * np.pi / 180).sin() *torso_lengths_angles[:, 35]) * (0. + torso_lengths_angles_v[:, 19]).cos() - (((torso_lengths_angles_v[:, 14]) * 100 * np.pi / 180).sin() * (((1.8 + torso_lengths_angles_v[:, 12]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[:, 16] * 100 * np.pi / 180).cos() *torso_lengths_angles[:, 35] -  torso_lengths_angles[:, 30]) - ((1.8 + torso_lengths_angles_v[:, 12]) * 100 * np.pi / 180).sin() * ((-torso_lengths_angles_v[:, 10] + torso_lengths_angles_v[:, 14] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[:, 16] * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 35]) -  ((torso_lengths_angles_v[:, 14]) * 100 * np.pi / 180).cos() * ((-torso_lengths_angles_v[:, 10] + torso_lengths_angles_v[:, 14] + 0.9) * 100 * np.pi / 180).sin() * (torso_lengths_angles_v[:, 16] * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 35]) * (0. + torso_lengths_angles_v[:, 19]).sin() - torso_lengths_angles[:, 30] * (0. + torso_lengths_angles_v[:, 19]).cos()
                torso_lengths_angles_v[:, 63] = torso_lengths_angles_v[:, 39] - torso_lengths_angles[:, 29] + (((1.8 + torso_lengths_angles_v[:, 12]) * 100 * np.pi / 180).sin() * ((torso_lengths_angles_v[:, 16] * 100 * np.pi / 180).cos() *  torso_lengths_angles[:, 35] - torso_lengths_angles[:, 30]) + ((1.8 + torso_lengths_angles_v[:, 12]) * 100 * np.pi / 180).cos() * ((-torso_lengths_angles_v[:, 10] + torso_lengths_angles_v[:, 14] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[:, 16] * 100 * np.pi / 180).sin() *torso_lengths_angles[:, 35]) * (0. + torso_lengths_angles_v[:, 19]).sin() + ((-(torso_lengths_angles_v[:, 14]) * 100 * np.pi / 180).sin() * (((1.8 + torso_lengths_angles_v[:, 12]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[:, 16] * 100 * np.pi / 180).cos() * torso_lengths_angles[:, 35] -  torso_lengths_angles[:, 33]) - ((1.8 + torso_lengths_angles_v[:, 12]) * 100 * np.pi / 180).sin() * ((-torso_lengths_angles_v[:, 10] + torso_lengths_angles_v[:, 14] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[:, 16] * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 35]) +  ((torso_lengths_angles_v[:, 14]) * 100 * np.pi / 180).cos() * ((-torso_lengths_angles_v[:, 10] + torso_lengths_angles_v[:, 14] + 0.9) * 100 * np.pi / 180).sin() * (torso_lengths_angles_v[:, 16] * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 35]) * (0. + torso_lengths_angles_v[:, 19]).cos() - torso_lengths_angles[:, 30] * (0. + torso_lengths_angles_v[:, 19]).sin()


                # left ankle in vectorized form
                torso_lengths_angles_v[:, 64] = torso_lengths_angles_v[:, 37] + torso_lengths_angles[:, 32] + (((-1.8 - torso_lengths_angles_v[:, 15])) * 100 * np.pi / 180).cos() * (((-torso_lengths_angles_v[:, 13]) * 100 * np.pi / 180).cos() * (((3.6 - torso_lengths_angles_v[:, 17]) * 100 * np.pi / 180).cos() *  torso_lengths_angles[:, 36] - torso_lengths_angles[:, 34]) - ((-torso_lengths_angles_v[:, 13]) * 100 * np.pi / 180).sin() * ((-torso_lengths_angles_v[:, 11] + (-1.8 - torso_lengths_angles_v[:, 15]) + 0.9) * 100 * np.pi / 180).cos() * ((3.6 - torso_lengths_angles_v[:, 17]) * 100 * np.pi / 180).sin() *  torso_lengths_angles[:, 36]) + (((-1.8 - torso_lengths_angles_v[:, 15])) * 100 * np.pi / 180).sin() * ((-torso_lengths_angles_v[:, 11] + (-1.8 - torso_lengths_angles_v[:, 15]) + 0.9) * 100 * np.pi / 180).sin() * ((3.6 - torso_lengths_angles_v[:, 17]) * 100 * np.pi / 180).sin() *  torso_lengths_angles[:, 36]
                #torso_lengths_angles_v[:, 65] = torso_lengths_angles_v[:, 38] + (((-torso_lengths_angles_v[:, 13]) * 100 * np.pi / 180).sin() * (((3.6 - torso_lengths_angles_v[:, 17]) * 100 * np.pi / 180).cos() * torso_lengths_angles[:, 36] -  torso_lengths_angles[:, 34]) + ((-torso_lengths_angles_v[:, 13]) * 100 * np.pi / 180).cos() * ((-torso_lengths_angles_v[:, 11] + (-1.8 - torso_lengths_angles_v[:, 15]) + 0.9) * 100 * np.pi / 180).cos() * ((3.6 - torso_lengths_angles_v[:, 17]) * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 36]) - torso_lengths_angles[:, 30]
                #torso_lengths_angles_v[:, 66] = torso_lengths_angles_v[:, 39] - torso_lengths_angles[:, 29] + (((-1.8 - torso_lengths_angles_v[:, 15])) * 100 * np.pi / 180).sin() * (((-torso_lengths_angles_v[:, 13]) * 100 * np.pi / 180).cos() * (((3.6 - torso_lengths_angles_v[:, 17]) * 100 * np.pi / 180).cos() *  torso_lengths_angles[:, 36] -torso_lengths_angles[:, 34]) - ((-torso_lengths_angles_v[:, 13]) * 100 * np.pi / 180).sin() * ((-torso_lengths_angles_v[:, 11] + (-1.8 - torso_lengths_angles_v[:, 15]) + 0.9) * 100 * np.pi / 180).cos() * ((3.6 - torso_lengths_angles_v[:, 17]) * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 36]) - (((-1.8 - torso_lengths_angles_v[:, 15])) * 100 * np.pi / 180).cos() * ((-torso_lengths_angles_v[:, 11] + (-1.8 - torso_lengths_angles_v[:, 15]) + 0.9) * 100 * np.pi / 180).sin() * ((3.6 - torso_lengths_angles_v[:, 17]) * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 36]
                torso_lengths_angles_v[:, 65] = torso_lengths_angles_v[:, 38] + (((-torso_lengths_angles_v[:, 13]) * 100 * np.pi / 180).sin() * (((3.6 - torso_lengths_angles_v[:, 17]) * 100 * np.pi / 180).cos() * torso_lengths_angles[:, 36] -  torso_lengths_angles[:, 34]) + ((-torso_lengths_angles_v[:, 13]) * 100 * np.pi / 180).cos() * ((-torso_lengths_angles_v[:, 11] + (-1.8 - torso_lengths_angles_v[:, 15]) + 0.9) * 100 * np.pi / 180).cos() * ((3.6 - torso_lengths_angles_v[:, 17]) * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 36]) * (0. + torso_lengths_angles_v[:, 19]).cos()  - ((((-1.8 - torso_lengths_angles_v[:, 15])) * 100 * np.pi / 180).sin() * (((-torso_lengths_angles_v[:, 13]) * 100 * np.pi / 180).cos() * (((3.6 - torso_lengths_angles_v[:, 17]) * 100 * np.pi / 180).cos() *  torso_lengths_angles[:, 36] - torso_lengths_angles[:, 34]) - ((-torso_lengths_angles_v[:, 13]) * 100 * np.pi / 180).sin() * ((-torso_lengths_angles_v[:, 11] + (-1.8 - torso_lengths_angles_v[:, 15]) + 0.9) * 100 * np.pi / 180).cos() * ((3.6 - torso_lengths_angles_v[:, 17]) * 100 * np.pi / 180).sin() *torso_lengths_angles[:, 36]) - (((-1.8 - torso_lengths_angles_v[:, 15])) * 100 * np.pi / 180).cos() * ((-torso_lengths_angles_v[:, 11] + (-1.8 - torso_lengths_angles_v[:, 15]) + 0.9) * 100 * np.pi / 180).sin() * ((3.6 - torso_lengths_angles_v[:, 17]) * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 36]) * (0. + torso_lengths_angles_v[:, 19]).sin() - torso_lengths_angles[:, 30] * (0. + torso_lengths_angles_v[:, 19]).cos()
                torso_lengths_angles_v[:, 66] = torso_lengths_angles_v[:, 39] - torso_lengths_angles[:, 29] + (((-torso_lengths_angles_v[:, 13]) * 100 * np.pi / 180).sin() * (((3.6 - torso_lengths_angles_v[:, 17]) * 100 * np.pi / 180).cos() *  torso_lengths_angles[:, 36] -torso_lengths_angles[:, 34]) + ((-torso_lengths_angles_v[:, 13]) * 100 * np.pi / 180).cos() * ((-torso_lengths_angles_v[:, 11] + (-1.8 - torso_lengths_angles_v[:, 15]) + 0.9) * 100 * np.pi / 180).cos() * ((3.6 - torso_lengths_angles_v[:, 17]) * 100 * np.pi / 180).sin() *torso_lengths_angles[:, 36]) * (0. + torso_lengths_angles_v[:, 19]).sin() + ((((-1.8 - torso_lengths_angles_v[:, 15])) * 100 * np.pi / 180).sin() * (((-torso_lengths_angles_v[:, 13]) * 100 * np.pi / 180).cos() * (((3.6 - torso_lengths_angles_v[:, 17]) * 100 * np.pi / 180).cos() *  torso_lengths_angles[:, 36] -torso_lengths_angles[:, 34]) - ((-torso_lengths_angles_v[:, 13]) * 100 * np.pi / 180).sin() * ((-torso_lengths_angles_v[:, 11] + (-1.8 - torso_lengths_angles_v[:, 15]) + 0.9) * 100 * np.pi / 180).cos() * ((3.6 - torso_lengths_angles_v[:, 17]) * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 36]) - (((-1.8 - torso_lengths_angles_v[:, 15])) * 100 * np.pi / 180).cos() * ((-torso_lengths_angles_v[:, 11] + (-1.8 - torso_lengths_angles_v[:, 15]) + 0.9) * 100 * np.pi / 180).sin() * ((3.6 - torso_lengths_angles_v[:, 17]) * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 36]) * (0. + torso_lengths_angles_v[:, 19]).cos() - torso_lengths_angles[:, 30] * (0. + torso_lengths_angles_v[:, 19]).sin()


                if forward_only == True:
                    #let's get the neck, shoulders, and glutes pseudotargets
                    pseudotargets = Variable(torch.Tensor(np.zeros((images.shape[0], 15))))

                    #get the neck in vectorized form
                    pseudotargets[:, 0] = torso_lengths_angles_v[:, 37]
                    pseudotargets[:, 1] = torso_lengths_angles_v[:, 38] + torso_lengths_angles[:, 21] * (0. + torso_lengths_angles_v[:, 18]).cos()
                    pseudotargets[:, 2] = torso_lengths_angles_v[:, 39] - torso_lengths_angles[:, 20] + torso_lengths_angles[:, 21] * (0. + torso_lengths_angles_v[:, 18]).sin()

                    #get the right shoulder in vectorized form
                    pseudotargets[:, 3] = torso_lengths_angles_v[:, 37] - torso_lengths_angles[:, 22]
                    pseudotargets[:, 4] = torso_lengths_angles_v[:, 38] + torso_lengths_angles[:, 21] * (0. + torso_lengths_angles_v[:, 18]).cos()
                    pseudotargets[:, 5] = torso_lengths_angles_v[:, 39] - torso_lengths_angles[:, 20] + torso_lengths_angles[:, 21] * (0. + torso_lengths_angles_v[:, 18]).sin()

                    #print \the left shoulder in vectorized form
                    pseudotargets[:, 6] = torso_lengths_angles_v[:, 37] + torso_lengths_angles[:, 23]
                    pseudotargets[:, 7] = torso_lengths_angles_v[:, 38] + torso_lengths_angles[:, 21] * (0. + torso_lengths_angles_v[:, 18]).cos()
                    pseudotargets[:, 8] = torso_lengths_angles_v[:, 39] - torso_lengths_angles[:, 20] + torso_lengths_angles[:, 21] * (0. + torso_lengths_angles_v[:, 18]).sin()

                    #get the right glute in vectorized form
                    pseudotargets[:, 9] = torso_lengths_angles_v[:, 37] - torso_lengths_angles[:, 31]
                    pseudotargets[:, 10] = torso_lengths_angles_v[:, 38] - torso_lengths_angles[:, 30] * (0. + torso_lengths_angles_v[:, 19]).cos()
                    pseudotargets[:, 11] = torso_lengths_angles_v[:, 39] - torso_lengths_angles[:, 29] + torso_lengths_angles[:, 30] * (0. + torso_lengths_angles_v[:, 19]).sin()

                    #print \the left glute in vectorized form
                    pseudotargets[:, 12] = torso_lengths_angles_v[:, 37] + torso_lengths_angles[:, 32]
                    pseudotargets[:, 13] = torso_lengths_angles_v[:, 38] - torso_lengths_angles[:, 30] * (0. + torso_lengths_angles_v[:, 19]).cos()
                    pseudotargets[:, 14] = torso_lengths_angles_v[:, 39] - torso_lengths_angles[:, 29] + torso_lengths_angles[:, 30] * (0. + torso_lengths_angles_v[:, 19]).sin()

                    pseudotargets = pseudotargets.data.numpy() * 1000

            angles = torso_lengths_angles_v[:, 0:20].data.numpy()*100
            torso_lengths_angles_v = torso_lengths_angles_v.unsqueeze(0)
            torso_lengths_angles_v = torso_lengths_angles_v.unsqueeze(0)
            torso_lengths_angles_v = F.pad(torso_lengths_angles_v, (-20, 0, 0, 0)) #cut off all the angles
            torso_lengths_angles_v = torso_lengths_angles_v.squeeze(0)
            torso_lengths_angles_v = torso_lengths_angles_v.squeeze(0)


        elif loss_vector_type == 'arms_cascade':
            torso_lengths_angles_v = torso_lengths_angles_v.unsqueeze(0)
            torso_lengths_angles_v = torso_lengths_angles_v.unsqueeze(0)
            torso_lengths_angles_v = F.pad(torso_lengths_angles_v, (0, 6, 0, 0))  #make space for right elbow and hand x, y, z coords
            # print torso_lengths_angles_v.size()

            torso_lengths_angles_v = torso_lengths_angles_v.squeeze(0)
            torso_lengths_angles_v = torso_lengths_angles_v.squeeze(0)

            if test_ground_truth == True:
                #print kincons_v.size(), 'kincon size'
                torso_lengths_angles_v[:, 0:4] = torch.cat((kincons_v[:, 0:1], kincons_v[:, 2:3], kincons_v[:, 4:5], kincons_v[:, 6:7]), dim = 1)

            # print torso_lengths_angles.shape

            lengths_v = prior_cascade[:, 3:11] # raw lengths coming out of network are in m.
            torso_v = prior_cascade[:, 0:3] / 1000 #raw positions coming out of network are in mm.

            #print lengths_v[0, :], 'lengths'
            #print torso_v[0, :]

            images = images_v.data.numpy() * np.pi / 180
            bedangle = images[:, -1, 10, 10] * 0.75

            if loop == True:
                pass

            elif loop == False:

                bedangle = Variable(torch.Tensor(bedangle))



                torso_lengths_angles_v[:, 0] = torch.clamp(torso_lengths_angles_v[:, 0], -1.8, 1.8)
                torso_lengths_angles_v[:, 1] = torch.clamp(torso_lengths_angles_v[:, 1], -1.35, 1.35)
                torso_lengths_angles_v[:, 2] = torch.clamp(torso_lengths_angles_v[:, 2], -1.35, 1.35)
                torso_lengths_angles_v[:, 3] = torch.clamp(torch.add(torso_lengths_angles_v[:, 3], 1.5), 0.4, 1.8)

                if body_side == 'right':
                    # right elbow in vectorized form
                    torso_lengths_angles_v[:, 4] = torso_v[:, 0] - lengths_v[:, 2] + (-(np.pi + torso_lengths_angles_v[:, 2] * 100 * np.pi / 180).cos() * lengths_v[:, 4] * (np.pi + torso_lengths_angles_v[:, 1] * 100 * np.pi / 180).cos())
                    torso_lengths_angles_v[:, 5] = torso_v[:, 1] + (-lengths_v[:, 4] * (np.pi + torso_lengths_angles_v[:, 1] * 100 * np.pi / 180).sin()) * bedangle[:].cos() - ((np.pi + torso_lengths_angles_v[:, 2] * 100 * np.pi / 180).sin() * lengths_v[:, 4] * (np.pi + torso_lengths_angles_v[:, 1] * 100 * np.pi / 180).cos()) * bedangle[:].sin() + lengths_v[:, 1] * bedangle[:].cos()
                    torso_lengths_angles_v[:, 6] = torso_v[:, 2] - lengths_v[:, 0] + (-lengths_v[:, 4] * (np.pi + torso_lengths_angles_v[:, 1] * 100 * np.pi / 180).sin()) * bedangle[:].sin() + ((np.pi + torso_lengths_angles_v[:, 2] * 100 * np.pi / 180).sin() * lengths_v[:, 4] * (np.pi + torso_lengths_angles_v[:, 1] * 100 * np.pi / 180).cos()) * bedangle[:].cos() + lengths_v[:, 1] * bedangle[:].sin()

                    # right hand in vectorized form
                    torso_lengths_angles_v[:, 7] = torso_v[:, 0] - lengths_v[:, 2] + ((1.8 - torso_lengths_angles_v[:, 2]) * 100 * np.pi / 180).cos() * (((1.8 + torso_lengths_angles_v[:, 1]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[:, 3] * 100 * np.pi / 180).cos() *  lengths_v[:, 6] -  lengths_v[:, 4]) - ((1.8 + torso_lengths_angles_v[:, 1]) * 100 * np.pi / 180).sin() * ((torso_lengths_angles_v[:, 0] + torso_lengths_angles_v[:, 2] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[:, 3] * 100 * np.pi / 180).sin() *  lengths_v[:, 6]) + ((1.8 -torso_lengths_angles_v[:, 2]) * 100 * np.pi / 180).sin() * ((torso_lengths_angles_v[:, 0] + torso_lengths_angles_v[:, 2] + 0.9) * 100 * np.pi / 180).sin() * (torso_lengths_angles_v[:, 3] * 100 * np.pi / 180).sin() * lengths_v[:, 6]
                    torso_lengths_angles_v[:, 8] = torso_v[:, 1] + (((1.8 + torso_lengths_angles_v[:, 1]) * 100 * np.pi / 180).sin() * ((torso_lengths_angles_v[:, 3] * 100 * np.pi / 180).cos() *  lengths_v[:, 6] -  lengths_v[:, 4]) + ((1.8 + torso_lengths_angles_v[:, 1]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[:, 0] + torso_lengths_angles_v[:, 2] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[:, 3] * 100 * np.pi / 180).sin() *lengths_v[:, 6]) * bedangle[:].cos() - (((1.8 - torso_lengths_angles_v[:, 2]) * 100 * np.pi / 180).sin() * (((1.8 + torso_lengths_angles_v[:, 1]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[:, 3] * 100 * np.pi / 180).cos() *lengths_v[:, 6] -  lengths_v[:, 4]) - ((1.8 + torso_lengths_angles_v[:, 1]) * 100 * np.pi / 180).sin() * ((torso_lengths_angles_v[:, 0] + torso_lengths_angles_v[:, 2] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[:, 3] * 100 * np.pi / 180).sin() * lengths_v[:, 6]) - ((1.8 - torso_lengths_angles_v[:, 2]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[:, 0] +torso_lengths_angles_v[:, 2] + 0.9) * 100 * np.pi / 180).sin() * (torso_lengths_angles_v[:, 3] * 100 * np.pi / 180).sin() * lengths_v[:, 6]) * bedangle[:].sin() + lengths_v[:, 1] * bedangle[:].cos()
                    torso_lengths_angles_v[:, 9] = torso_v[:, 2] - lengths_v[:, 0] + (((1.8 + torso_lengths_angles_v[:, 1]) * 100 * np.pi / 180).sin() * ((torso_lengths_angles_v[:, 3] * 100 * np.pi / 180).cos() *  lengths_v[:, 6] - lengths_v[:, 4]) + ((1.8 + torso_lengths_angles_v[:, 1]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[:, 0] + torso_lengths_angles_v[:, 2] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[:, 3] * 100 * np.pi / 180).sin() *lengths_v[:, 6]) * bedangle[:].sin() + (((1.8 - torso_lengths_angles_v[:, 2]) * 100 * np.pi / 180).sin() * (((1.8 + torso_lengths_angles_v[:, 1]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[:, 3] * 100 * np.pi / 180).cos() * lengths_v[:, 6] -  lengths_v[:, 4]) - ((1.8 + torso_lengths_angles_v[:, 1]) * 100 * np.pi / 180).sin() * ((torso_lengths_angles_v[:, 0] + torso_lengths_angles_v[:, 2] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[:, 3] * 100 * np.pi / 180).sin() * lengths_v[:, 6]) - ((1.8 - torso_lengths_angles_v[:, 2]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[:, 0] + torso_lengths_angles_v[:, 2] + 0.9) * 100 * np.pi / 180).sin() * (torso_lengths_angles_v[:, 3] * 100 * np.pi / 180).sin() * lengths_v[:, 6]) * bedangle[:].cos() + lengths_v[:, 1] * bedangle[:].sin()
                elif body_side == 'left':
                    # left elbow in vectorized form
                    torso_lengths_angles_v[:, 4] = torso_v[:, 0] + lengths_v[:, 3] + (-(np.pi + torso_lengths_angles_v[:, 2] * 100 * np.pi / 180).cos() * lengths_v[:, 5] * (-torso_lengths_angles_v[:, 1] * 100 * np.pi / 180).cos())
                    torso_lengths_angles_v[:, 5] = torso_v[:, 1] + (-lengths_v[:, 5] * (-torso_lengths_angles_v[:, 1] * 100 * np.pi / 180).sin()) * bedangle[:].cos() + ((np.pi + torso_lengths_angles_v[:,2] * 100 * np.pi / 180).sin() * lengths_v[:,5] * (- torso_lengths_angles_v[:,1] * 100 * np.pi / 180).cos()) * bedangle[:].sin() + lengths_v[:, 1] * bedangle[:].cos()
                    torso_lengths_angles_v[:, 6] = torso_v[:, 2] - lengths_v[:, 0] + (-lengths_v[:,5] * (- torso_lengths_angles_v[:, 1] * 100 * np.pi / 180).sin()) * bedangle[:].sin() - ((np.pi + torso_lengths_angles_v[:,2] * 100 * np.pi / 180).sin() * lengths_v[:,5] * (- torso_lengths_angles_v[:,1] * 100 * np.pi / 180).cos()) * bedangle[:].cos() + lengths_v[:,1] * bedangle[:].sin()

                    # left hand in vectorized form
                    torso_lengths_angles_v[:, 7] = torso_v[:, 0] + lengths_v[:, 3] + ((1.8 + torso_lengths_angles_v[:, 5]) * 100 * np.pi / 180).cos() * (((-torso_lengths_angles_v[:, 3]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[:, 7] * 100 * np.pi / 180).cos() * torso_lengths_angles[:, 25] - torso_lengths_angles[:, 25]) - ((-torso_lengths_angles_v[:,3]) * 100 * np.pi / 180).sin() * ((-torso_lengths_angles_v[:, 1] - torso_lengths_angles_v[:,5] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[:, 7] * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 25]) + ((1.8 + torso_lengths_angles_v[:,5]) * 100 * np.pi / 180).sin() * ((-torso_lengths_angles_v[:,1] - torso_lengths_angles_v[:,5] + 0.9) * 100 * np.pi / 180).sin() * (torso_lengths_angles_v[:, 7] * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 25]
                    torso_lengths_angles_v[:, 8] = torso_v[:, 1] + (((-torso_lengths_angles_v[:, 3]) * 100 * np.pi / 180).sin() * ((torso_lengths_angles_v[:, 7] * 100 * np.pi / 180).cos() * torso_lengths_angles[:, 25] - torso_lengths_angles[:, 25]) + ((-torso_lengths_angles_v[:, 3]) * 100 * np.pi / 180).cos() * ((-torso_lengths_angles_v[:,1] - torso_lengths_angles_v[:, 5] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[:, 7] * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 25]) * bedangle[:].cos() - (((1.8 + torso_lengths_angles_v[ :,5]) * 100 * np.pi / 180).sin() * (((-torso_lengths_angles_v[:, 3]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[:, 7] * 100 * np.pi / 180).cos() * torso_lengths_angles[:,25] - torso_lengths_angles[:, 25]) - ((-torso_lengths_angles_v[:,3]) * 100 * np.pi / 180).sin() * ((-torso_lengths_angles_v[:,1] - torso_lengths_angles_v[:, 5] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[:, 7] * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 25]) - ((1.8 + torso_lengths_angles_v[:, 5]) * 100 * np.pi / 180).cos() * ((-torso_lengths_angles_v[:, 1] - torso_lengths_angles_v[:, 5] + 0.9) * 100 * np.pi / 180).sin() * (torso_lengths_angles_v[:,7] * 100 * np.pi / 180).sin() * torso_lengths_angles[:,25]) * bedangle[:].sin() + torso_lengths_angles[:,19] * bedangle[:].cos()
                    torso_lengths_angles_v[:, 9] = torso_v[:, 2] - lengths_v[:, 0] + (((-torso_lengths_angles_v[:, 3]) * 100 * np.pi / 180).sin() * ((torso_lengths_angles_v[:, 7] * 100 * np.pi / 180).cos() * torso_lengths_angles[:, 25] - torso_lengths_angles[:,25]) + ((-torso_lengths_angles_v[:, 3]) * 100 * np.pi / 180).cos() * ((-torso_lengths_angles_v[:,1] - torso_lengths_angles_v[:, 5] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[:, 7] * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 25]) * bedangle[:].sin() + (((1.8 + torso_lengths_angles_v[:,5]) * 100 * np.pi / 180).sin() * (((-torso_lengths_angles_v[:, 3]) * 100 * np.pi / 180).cos() * ((torso_lengths_angles_v[:, 7] * 100 * np.pi / 180).cos() * torso_lengths_angles[:, 25] - torso_lengths_angles[:,25]) - ((-torso_lengths_angles_v[:, 3]) * 100 * np.pi / 180).sin() * ((-torso_lengths_angles_v[:, 1] - torso_lengths_angles_v[:, 5] + 0.9) * 100 * np.pi / 180).cos() * (torso_lengths_angles_v[:, 7] * 100 * np.pi / 180).sin() * torso_lengths_angles[:, 25]) - ((1.8 + torso_lengths_angles_v[:,5]) * 100 * np.pi / 180).cos() * ((-torso_lengths_angles_v[:, 1] - torso_lengths_angles_v[:, 5] + 0.9) * 100 * np.pi / 180).sin() * (torso_lengths_angles_v[:,7] * 100 * np.pi / 180).sin() * torso_lengths_angles[:,25]) * bedangle[:].cos() + torso_lengths_angles[:,19] * bedangle[:].sin()

            angles = torso_lengths_angles_v[:, 0:4].data.numpy()*100
            torso_lengths_angles_v = torso_lengths_angles_v.unsqueeze(0)
            torso_lengths_angles_v = torso_lengths_angles_v.unsqueeze(0)
            torso_lengths_angles_v = F.pad(torso_lengths_angles_v, (-4, 0, 0, 0)) #cut off 4 angles
            torso_lengths_angles_v = torso_lengths_angles_v.squeeze(0)
            torso_lengths_angles_v = torso_lengths_angles_v.squeeze(0)

        return torso_lengths_angles_v, angles, pseudotargets


    def forward_kinematics_3fc_pytorch(self, images_v, torso_v, lengths_v, angles_v, targets_v, loss_vector_type, kincons_v = None, prior_cascade = None, forward_only = False):

        test_ground_truth = False
        loop = False
        pseudotargets = None


        if loss_vector_type == 'angles':
            angles_v = angles_v.unsqueeze(0)
            angles_v = angles_v.unsqueeze(0)
            #print angles_v.size()
            angles_v = F.pad(angles_v, (0, 27, 0, 0)) #make more room for head, arms, and legs x, y, z coords.  torso already is in the network.
            #print angles_v.size()

            angles_v = angles_v.squeeze(0)
            angles_v = angles_v.squeeze(0)

            if test_ground_truth == True:
                # print kincons_v.size()
                angles_v[:, 0:18] = kincons_v[:, 0:18] #this is the upper angles, lower angles, upper lengths, lower lengths in that order
                lengths_v[:, 0:17] = kincons_v[:, 18:35]
                torso_v[:, 0:3] = targets_v[:, 3:6] / 1000 #this is the torso x, y, z coords
                # print targets_v[0, :], 'targets'
                # print torso_lengths_angles_v[0, :]

            lengths = lengths_v.data.numpy()
            # print torso_lengths_angles.shape

            images = images_v.data.numpy() * np.pi / 180
            bedangle = images[:, -1, 10, 10] * 0.75

            if loop == True:
                pass

            elif loop == False:

                lengths = Variable(torch.Tensor(lengths))
                bedangle = Variable(torch.Tensor(bedangle))

                #head in vectorized form
                angles_v[:, 18] = torso_v[:, 0] + lengths[:, 8] * ((np.pi / 2. - angles_v[:, 8] * 100 * np.pi / 180).cos()) * ((-np.pi / 2. + angles_v[:, 9] * 100 * np.pi / 180).cos())
                angles_v[:, 19] = torso_v[:, 1] + lengths[:, 8] * ((np.pi / 2. - angles_v[:, 8] * 100 * np.pi / 180).sin()) * ((-np.pi / 2. + angles_v[:, 9] * 100 * np.pi / 180).cos()) * bedangle[:].cos() + lengths[:, 8] * ((-np.pi / 2. + angles_v[:, 9] * 100 * np.pi / 180).sin()) * bedangle[:].sin() + lengths[:, 1] * bedangle[:].cos()
                angles_v[:, 20] = torso_v[:, 2] - lengths[:, 0] + lengths[:, 8] * ((np.pi / 2. - angles_v[:, 8] * 100 * np.pi / 180).sin()) * ((-np.pi / 2. + angles_v[:, 9] * 100 * np.pi / 180).cos()) * bedangle[:].sin() - lengths[:, 8] * ((-np.pi / 2. + angles_v[:, 9] * 100 * np.pi / 180).sin()) * bedangle[:].cos() + lengths[:, 1] * bedangle[:].sin()

                # right elbow in vectorized form
                angles_v[:, 21] = torso_v[:, 0] - lengths[:, 2] + (-(np.pi + angles_v[:, 4] * 100 * np.pi / 180).cos() * lengths[:, 4] * (np.pi + angles_v[:, 2] * 100 * np.pi / 180).cos())
                angles_v[:, 22] = torso_v[:, 1] + (-lengths[:, 4] * (np.pi + angles_v[:, 2] * 100 * np.pi / 180).sin()) * bedangle[:].cos() - ((np.pi + angles_v[:, 4] * 100 * np.pi / 180).sin() * lengths[:, 4] * (np.pi + angles_v[:, 2] * 100 * np.pi / 180).cos()) * bedangle[:].sin() + lengths[:, 1] * bedangle[:].cos()
                angles_v[:, 23] = torso_v[:, 2] - lengths[:, 0] + (-lengths[:, 4] * (np.pi + angles_v[:, 2] * 100 * np.pi / 180).sin()) * bedangle[:].sin() + ((np.pi + angles_v[:, 4] * 100 * np.pi / 180).sin() * lengths[:, 4] * (np.pi + angles_v[:, 2] * 100 * np.pi / 180).cos()) * bedangle[:].cos() + lengths[:, 1] * bedangle[:].sin()

                # left elbow in vectorized form
                angles_v[:, 24] = torso_v[:, 0] + lengths[:, 3] + (-(np.pi + angles_v[:, 5] * 100 * np.pi / 180).cos() * lengths[:, 5] * (-angles_v[:, 3] * 100 * np.pi / 180).cos())
                angles_v[:, 25] = torso_v[:, 1] + (-lengths[:, 5] * (-angles_v[:, 3] * 100 * np.pi / 180).sin()) * bedangle[:].cos() + ((np.pi + angles_v[:, 5] * 100 * np.pi / 180).sin() * lengths[:, 5] * (- angles_v[:, 3] * 100 * np.pi / 180).cos()) * bedangle[:].sin() + lengths[:, 1] * bedangle[:].cos()
                angles_v[:, 26] = torso_v[:, 2] - lengths[:, 0] + (-lengths[:, 5] * (- angles_v[:, 3] * 100 * np.pi / 180).sin()) * bedangle[:].sin() - ((np.pi +angles_v[:, 5] * 100 * np.pi / 180).sin() * lengths[:, 5] * (- angles_v[:, 3] * 100 * np.pi / 180).cos()) * bedangle[:].cos() + lengths[:, 1] * bedangle[:].sin()

                # right hand in vectorized form
                angles_v[:, 27] = torso_v[:, 0] - lengths[:, 2] + ((1.8 - angles_v[:, 4]) * 100 * np.pi / 180).cos() * (((1.8 + angles_v[:, 2]) * 100 * np.pi / 180).cos() * ((angles_v[:, 6] * 100 * np.pi / 180).cos() *  lengths[:, 6] -  lengths[:, 1]) - ((1.8 + angles_v[:, 2]) * 100 * np.pi / 180).sin() * ((angles_v[:, 0] + angles_v[:, 4] + 0.9) * 100 * np.pi / 180).cos() * (angles_v[:, 6] * 100 * np.pi / 180).sin() *  lengths[:, 6]) + ((1.8 -angles_v[:, 4]) * 100 * np.pi / 180).sin() * ((angles_v[:, 0] + angles_v[:, 4] + 0.9) * 100 * np.pi / 180).sin() * (angles_v[:, 6] * 100 * np.pi / 180).sin() * lengths[:, 6]
                angles_v[:, 28] = torso_v[:, 1] + (((1.8 + angles_v[:, 2]) * 100 * np.pi / 180).sin() * ((angles_v[:, 6] * 100 * np.pi / 180).cos() *  lengths[:, 6] -  lengths[:, 1]) + ((1.8 + angles_v[:, 2]) * 100 * np.pi / 180).cos() * ((angles_v[:, 0] + angles_v[:, 4] + 0.9) * 100 * np.pi / 180).cos() * (angles_v[:, 6] * 100 * np.pi / 180).sin() *lengths[:, 6]) * bedangle[:].cos() - (((1.8 - angles_v[:, 4]) * 100 * np.pi / 180).sin() * (((1.8 + angles_v[:, 2]) * 100 * np.pi / 180).cos() * ((angles_v[:, 6] * 100 * np.pi / 180).cos() *lengths[:, 6] -  lengths[:, 1]) - ((1.8 + angles_v[:, 2]) * 100 * np.pi / 180).sin() * ((angles_v[:, 0] + angles_v[:, 4] + 0.9) * 100 * np.pi / 180).cos() * (angles_v[:, 6] * 100 * np.pi / 180).sin() * lengths[:, 6]) - ((1.8 - angles_v[:, 4]) * 100 * np.pi / 180).cos() * ((angles_v[:, 0] +angles_v[:, 4] + 0.9) * 100 * np.pi / 180).sin() * (angles_v[:, 6] * 100 * np.pi / 180).sin() * lengths[:, 6]) * bedangle[:].sin() + lengths[:, 1] * bedangle[:].cos()
                angles_v[:, 29] = torso_v[:, 2] - lengths[:, 0] + (((1.8 + angles_v[:, 2]) * 100 * np.pi / 180).sin() * ((angles_v[:, 6] * 100 * np.pi / 180).cos() *  lengths[:, 6] - lengths[:, 1]) + ((1.8 + angles_v[:, 2]) * 100 * np.pi / 180).cos() * ((angles_v[:, 0] + angles_v[:, 4] + 0.9) * 100 * np.pi / 180).cos() * (angles_v[:, 6] * 100 * np.pi / 180).sin() *lengths[:, 6]) * bedangle[:].sin() + (((1.8 - angles_v[:, 4]) * 100 * np.pi / 180).sin() * (((1.8 + angles_v[:, 2]) * 100 * np.pi / 180).cos() * ((angles_v[:, 6] * 100 * np.pi / 180).cos() * lengths[:, 6] -  lengths[:, 1]) - ((1.8 + angles_v[:, 2]) * 100 * np.pi / 180).sin() * ((angles_v[:, 0] + angles_v[:, 4] + 0.9) * 100 * np.pi / 180).cos() * (angles_v[:, 6] * 100 * np.pi / 180).sin() * lengths[:, 6]) - ((1.8 - angles_v[:, 4]) * 100 * np.pi / 180).cos() * ((angles_v[:, 0] + angles_v[:, 4] + 0.9) * 100 * np.pi / 180).sin() * (angles_v[:, 6] * 100 * np.pi / 180).sin() * lengths[:, 6]) * bedangle[:].cos() + lengths[:, 1] * bedangle[:].sin()

                # left hand in vectorized form
                angles_v[:, 30] = torso_v[:, 0] + lengths[:, 3] + ((1.8 + angles_v[:, 5]) * 100 * np.pi / 180).cos() * (((-angles_v[:, 3]) * 100 * np.pi / 180).cos() * ((angles_v[:, 7] * 100 * np.pi / 180).cos() *  lengths[:, 7] - lengths[:, 5]) - ((-angles_v[:, 3]) * 100 * np.pi / 180).sin() * ((-angles_v[:, 1] - angles_v[:, 5] + 0.9) * 100 * np.pi / 180).cos() * (angles_v[:, 7] * 100 * np.pi / 180).sin() *  lengths[:, 7]) + ((1.8 +angles_v[:, 5]) * 100 * np.pi / 180).sin() * ((-angles_v[:, 1] -angles_v[:, 5] + 0.9) * 100 * np.pi / 180).sin() * (angles_v[:, 7] * 100 * np.pi / 180).sin() *  lengths[:, 7]
                angles_v[:, 31] = torso_v[:, 1] + (((-angles_v[:, 3]) * 100 * np.pi / 180).sin() * ((angles_v[:, 7] * 100 * np.pi / 180).cos() * lengths[:, 7] -  lengths[:, 5]) + ((-angles_v[:, 3]) * 100 * np.pi / 180).cos() * ((-angles_v[:, 1] - angles_v[:, 5] + 0.9) * 100 * np.pi / 180).cos() * (angles_v[:, 7] * 100 * np.pi / 180).sin() * lengths[:, 7]) * bedangle[:].cos() - (((1.8 + angles_v[:, 5]) * 100 * np.pi / 180).sin() * (((-angles_v[:, 3]) * 100 * np.pi / 180).cos() * ((angles_v[:, 7] * 100 * np.pi / 180).cos() *  lengths[:, 7] - lengths[:, 5]) - ((-angles_v[:, 3]) * 100 * np.pi / 180).sin() * ((-angles_v[:, 1] -angles_v[:, 5] + 0.9) * 100 * np.pi / 180).cos() * (angles_v[:, 7] * 100 * np.pi / 180).sin() *lengths[:, 7]) - ((1.8 + angles_v[:, 5]) * 100 * np.pi / 180).cos() * ((-angles_v[:, 1] - angles_v[:, 5] + 0.9) * 100 * np.pi / 180).sin() * (angles_v[:, 7] * 100 * np.pi / 180).sin() * lengths[:, 7]) * bedangle[:].sin() + lengths[:, 1] * bedangle[:].cos()
                angles_v[:, 32] = torso_v[:, 2] - lengths[:, 0] + (((-angles_v[:, 3]) * 100 * np.pi / 180).sin() * ((angles_v[:, 7] * 100 * np.pi / 180).cos() *  lengths[:, 7] -lengths[:, 5]) + ((-angles_v[:, 3]) * 100 * np.pi / 180).cos() * ((-angles_v[:, 1] - angles_v[:, 5] + 0.9) * 100 * np.pi / 180).cos() * (angles_v[:, 7] * 100 * np.pi / 180).sin() *lengths[:, 7]) * bedangle[:].sin() + (((1.8 + angles_v[:, 5]) * 100 * np.pi / 180).sin() * (((-angles_v[:, 3]) * 100 * np.pi / 180).cos() * ((angles_v[:, 7] * 100 * np.pi / 180).cos() *  lengths[:, 7] -lengths[:, 5]) - ((-angles_v[:, 3]) * 100 * np.pi / 180).sin() * ((-angles_v[:, 1] -angles_v[:, 5] + 0.9) * 100 * np.pi / 180).cos() * (angles_v[:, 7] * 100 * np.pi / 180).sin() * lengths[:, 7]) - ((1.8 + angles_v[:, 5]) * 100 * np.pi / 180).cos() * ((-angles_v[:, 1] - angles_v[:, 5] + 0.9) * 100 * np.pi / 180).sin() * (angles_v[:, 7] * 100 * np.pi / 180).sin() * lengths[:, 7]) * bedangle[:].cos() + lengths[:, 1] * bedangle[:].sin()



                # right knee in vectorized form
                angles_v[:, 33] = torso_v[:, 0] - lengths[:, 11] + (-(angles_v[:, 14] * 100 * np.pi / 180).cos() * lengths[:, 13] * (np.pi + angles_v[:, 12] * 100 * np.pi / 180).cos())
                angles_v[:, 34] = torso_v[:, 1] + (-lengths[:, 13] * (np.pi + angles_v[:, 12] * 100 * np.pi / 180).sin()) - lengths[:, 10]
                angles_v[:, 35] = torso_v[:, 2] - lengths[:, 9] + (-(angles_v[:, 14] * 100 * np.pi / 180).sin() * lengths[:, 13] * (np.pi + angles_v[:, 12] * 100 * np.pi / 180).cos())

                # left knee in vectorized form
                angles_v[:, 36] = torso_v[:, 0] + lengths[:, 12] + (-(-np.pi + angles_v[:, 15] * 100 * np.pi / 180).cos() * lengths[:, 14] * (-angles_v[:, 13] * 100 * np.pi / 180).cos())
                angles_v[:, 37] = torso_v[:, 1] + (-lengths[:, 14] * (-angles_v[:, 13] * 100 * np.pi / 180).sin()) - lengths[:, 10]
                angles_v[:, 38] = torso_v[:, 2] - lengths[:, 9] - ((-np.pi +angles_v[:, 15] * 100 * np.pi / 180).sin() * lengths[:, 14] * (- angles_v[:, 13] * 100 * np.pi / 180).cos())

                # right ankle in vectorized form
                angles_v[:, 39] = torso_v[:, 0] - lengths[:, 11] + ((angles_v[:, 14]) * 100 * np.pi / 180).cos() * (((1.8 + angles_v[:, 12]) * 100 * np.pi / 180).cos() * ((angles_v[:, 16] * 100 * np.pi / 180).cos() *  lengths[:, 15] -  lengths[:, 13]) - ((1.8 + angles_v[:, 12]) * 100 * np.pi / 180).sin() * ((-angles_v[:, 10] + angles_v[:, 14] + 0.9) * 100 * np.pi / 180).cos() * (angles_v[:, 16] * 100 * np.pi / 180).sin() *  lengths[:, 15]) + ((angles_v[:, 14]) * 100 * np.pi / 180).sin() * ((-angles_v[:, 10] + angles_v[:, 14] + 0.9) * 100 * np.pi / 180).sin() * (angles_v[:, 16] * 100 * np.pi / 180).sin() * lengths[:, 15]
                angles_v[:, 40] = torso_v[:, 1] + (((1.8 + angles_v[:, 12]) * 100 * np.pi / 180).sin() * ((angles_v[:, 16] * 100 * np.pi / 180).cos() *  lengths[:, 15] -  lengths[:, 13]) + ((1.8 + angles_v[:, 12]) * 100 * np.pi / 180).cos() * ((-angles_v[:, 10] + angles_v[:, 14] + 0.9) * 100 * np.pi / 180).cos() * (angles_v[:, 16] * 100 * np.pi / 180).sin() *lengths[:, 15]) - lengths[:, 10]
                angles_v[:, 41] = torso_v[:, 2] - lengths[:, 9] + ((angles_v[:, 14]) * 100 * np.pi / 180).sin() * (((1.8 + angles_v[:, 12]) * 100 * np.pi / 180).cos() * ((angles_v[:, 16] * 100 * np.pi / 180).cos() * lengths[:, 15] -  lengths[:, 13]) - ((1.8 + angles_v[:, 12]) * 100 * np.pi / 180).sin() * ((-angles_v[:, 10] + angles_v[:, 14] + 0.9) * 100 * np.pi / 180).cos() * (angles_v[:, 16] * 100 * np.pi / 180).sin() * lengths[:, 15]) - ((angles_v[:, 14]) * 100 * np.pi / 180).cos() * ((-angles_v[:, 10] + angles_v[:, 14] + 0.9) * 100 * np.pi / 180).sin() * (angles_v[:, 16] * 100 * np.pi / 180).sin() * lengths[:, 15]

                # left ankle in vectorized form
                angles_v[:, 42] = torso_v[:, 0] + lengths[:, 12] + ((angles_v[:, 15] - 1.8) * 100 * np.pi / 180).cos() * (((-angles_v[:, 13]) * 100 * np.pi / 180).cos() * ((angles_v[:, 17] * 100 * np.pi / 180).cos() *  lengths[:, 16] - lengths[:, 14]) - ((-angles_v[:, 13]) * 100 * np.pi / 180).sin() * ((-angles_v[:, 11] + angles_v[:, 15] + 0.9) * 100 * np.pi / 180).cos() * (angles_v[:, 17] * 100 * np.pi / 180).sin() *  lengths[:, 16]) + ((angles_v[:, 15] - 1.8) * 100 * np.pi / 180).sin() * ((-angles_v[:, 11] + angles_v[:, 15] + 0.9) * 100 * np.pi / 180).sin() * (angles_v[:, 17] * 100 * np.pi / 180).sin() *  lengths[:, 16]
                angles_v[:, 43] = torso_v[:, 1] + (((-angles_v[:, 13]) * 100 * np.pi / 180).sin() * ((angles_v[:, 17] * 100 * np.pi / 180).cos() * lengths[:, 16] -  lengths[:, 14]) + ((-angles_v[:, 13]) * 100 * np.pi / 180).cos() * ((-angles_v[:, 11] + angles_v[:, 15] + 0.9) * 100 * np.pi / 180).cos() * (angles_v[:, 17] * 100 * np.pi / 180).sin() * lengths[:, 16]) - lengths[:, 10]
                angles_v[:, 44] = torso_v[:, 2] - lengths[:, 9] + ((angles_v[:, 15] - 1.8) * 100 * np.pi / 180).sin() * (((-angles_v[:, 13]) * 100 * np.pi / 180).cos() * ((angles_v[:, 17] * 100 * np.pi / 180).cos() *  lengths[:, 16] -lengths[:, 14]) - ((-angles_v[:, 13]) * 100 * np.pi / 180).sin() * ((-angles_v[:, 11] + angles_v[:, 15] + 0.9) * 100 * np.pi / 180).cos() * (angles_v[:, 17] * 100 * np.pi / 180).sin() * lengths[:, 16]) - ((angles_v[:, 15] - 1.8) * 100 * np.pi / 180).cos() * ((-angles_v[:, 11] + angles_v[:, 15] + 0.9) * 100 * np.pi / 180).sin() * (angles_v[:, 17] * 100 * np.pi / 180).sin() * lengths[:, 16]

                if forward_only == True:
                    #let's get the neck, shoulders, and glutes pseudotargets
                    pseudotargets = Variable(torch.Tensor(np.zeros((images.shape[0], 15))))

                    #get the neck in vectorized form
                    pseudotargets[:, 0] = torso_v[:, 0]
                    pseudotargets[:, 1] = torso_v[:, 1] + lengths[:, 1] * bedangle[:].cos()
                    pseudotargets[:, 2] = torso_v[:, 2] - lengths[:, 0] + lengths[:, 1] * bedangle[:].sin()

                    #get the right shoulder in vectorized form
                    pseudotargets[:, 3] = torso_v[:, 0] - lengths[:, 2]
                    pseudotargets[:, 4] = torso_v[:, 1] + lengths[:, 1] * bedangle[:].cos()
                    pseudotargets[:, 5] = torso_v[:, 2] - lengths[:, 0] + lengths[:, 1] * bedangle[:].sin()

                    #print \the left shoulder in vectorized form
                    pseudotargets[:, 6] = torso_v[:, 0] + lengths[:, 3]
                    pseudotargets[:, 7] = torso_v[:, 1] + lengths[:, 1] * bedangle[:].cos()
                    pseudotargets[:, 8] = torso_v[:, 2] - lengths[:, 0] + lengths[:, 1] * bedangle[:].sin()

                    #get the right glute in vectorized form
                    pseudotargets[:, 9] = torso_v[:, 0] - lengths[:, 11]
                    pseudotargets[:, 10] = torso_v[:, 1] - lengths[:, 10]
                    pseudotargets[:, 11] = torso_v[:, 2] - lengths[:, 9]

                    #print \the left glute in vectorized form
                    pseudotargets[:, 12] = torso_v[:, 0] + lengths[:, 12]
                    pseudotargets[:, 13] = torso_v[:, 1] - lengths[:, 10]
                    pseudotargets[:, 14] = torso_v[:, 2] - lengths[:, 9]

                    pseudotargets = pseudotargets.data.numpy() * 1000

            angles_v = angles_v.unsqueeze(0)
            angles_v = angles_v.unsqueeze(0)
            angles_v = F.pad(angles_v, (-18, 0, 0, 0)) #cut off all the angles
            angles_v = angles_v.squeeze(0)
            angles_v = angles_v.squeeze(0)


        return torso_v, angles_v, pseudotargets

