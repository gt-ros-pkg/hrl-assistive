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




class SyntheticLib():

    def synthetic_scale(self, images, targets, bedangles, pcons = None):

        x = np.arange(-10 ,11)
        xU, xL = x + 0.5, x - 0.05
        prob = ss.norm.cdf(xU, scale=3) - ss.norm.cdf(xL, scale=3)
        prob = prob / prob.sum()  # normalize the probabilities so their sum is 1
        multiplier = np.random.choice(x, size=images.shape[0], p=prob)
        multiplier = (multiplier *0.020) +1
        #plt.hist(multiplier)
        #plt.show()
        #multiplier[:] = 1.2
        if self.include_inter == True:
            multiplier[bedangles[:,0,0] > 10] = 1 #we have to cut out the scaling where the bed isn't flat

        # print multiplier
        tar_mod = np.reshape(targets, (targets.shape[0], targets.shape[1] / 3, 3) ) /1000

        #print tar_mod[0,:,:], 'prior'
        for i in np.arange(images.shape[0]):
            if self.include_inter == True:
                resized = zoom(images[i, :, :, :], multiplier[i])
                resized = np.clip(resized, 0, 100)

                if pcons is not None:
                    pcons[i,18:35] *= multiplier[i]


                rl_diff = resized.shape[2] - images[i, :, :, :].shape[2]
                ud_diff = resized.shape[1] - images[i, :, :, :].shape[1]
                l_clip = np.int(math.ceil((rl_diff) / 2))
                # r_clip = rl_diff - l_clip
                u_clip = np.int(math.ceil((ud_diff) / 2))
                # d_clip = ud_diff - u_clip

                if rl_diff < 0:  # if less than 0, we'll have to add some padding in to get back up to normal size
                    resized_adjusted = np.zeros_like(images[i, :, :, :])
                    resized_adjusted[:, -u_clip:-u_clip + resized.shape[1], -l_clip:-l_clip + resized.shape[2]] = np.copy(
                        resized)
                    images[i, :, :, :] = resized_adjusted
                    shift_factor_x = INTER_SENSOR_DISTANCE * -l_clip
                elif rl_diff > 0:  # if greater than 0, we'll have to cut the sides to get back to normal size
                    resized_adjusted = np.copy \
                        (resized[:, u_clip:u_clip + images[i, :, :, :].shape[1], l_clip:l_clip + images[i, :, :, :].shape[2]])
                    images[i, :, :, :] = resized_adjusted
                    shift_factor_x = INTER_SENSOR_DISTANCE * -l_clip
                else:
                    shift_factor_x = 0

            else:
                #multiplier[i] = 0.8
                resized = zoom(images[i ,: ,:], multiplier[i])
                resized = np.clip(resized, 0, 100)

                rl_diff = resized.shape[1] - images[i ,: ,:].shape[1]
                ud_diff = resized.shape[0] - images[i ,: ,:].shape[0]
                l_clip = np.int(math.ceil((rl_diff) / 2))
                # r_clip = rl_diff - l_clip
                u_clip = np.int(math.ceil((ud_diff) / 2))
                # d_clip = ud_diff - u_clip

                if rl_diff < 0:  # if less than 0, we'll have to add some padding in to get back up to normal size
                    resized_adjusted = np.zeros_like(images[i ,: ,:])
                    resized_adjusted[-u_clip:-u_clip + resized.shape[0], -l_clip:-l_clip + resized.shape[1]] = np.copy(resized)
                    images[i ,: ,:] = resized_adjusted
                    shift_factor_x = INTER_SENSOR_DISTANCE * -l_clip
                elif rl_diff > 0: # if greater than 0, we'll have to cut the sides to get back to normal size
                    resized_adjusted = np.copy \
                        (resized[u_clip:u_clip + images[i ,: ,:].shape[0], l_clip:l_clip + images[i ,: ,:].shape[1]])
                    images[i ,: ,:] = resized_adjusted
                    shift_factor_x = INTER_SENSOR_DISTANCE * -l_clip
                else:
                    shift_factor_x = 0

            if ud_diff < 0:
                shift_factor_y = INTER_SENSOR_DISTANCE * u_clip
            elif ud_diff > 0:
                shift_factor_y = INTER_SENSOR_DISTANCE * u_clip
            else:
                shift_factor_y = 0
            # print shift_factor_y, shift_factor_x

            resized_tar = np.copy(tar_mod[i ,: ,:])
            # resized_tar = np.reshape(resized_tar, (len(resized_tar) / 3, 3))
            # print resized_tar.shape/
            resized_tar = (resized_tar + INTER_SENSOR_DISTANCE ) * multiplier[i]

            resized_tar[:, 0] = resized_tar[:, 0] + shift_factor_x  - INTER_SENSOR_DISTANCE #- 10 * INTER_SENSOR_DISTANCE * (1 - multiplier[i])
            # resized_tar2 = np.copy(resized_tar)
            resized_tar[:, 1] = resized_tar[:, 1] + NUMOFTAXELS_X * (1 - multiplier[i]) * INTER_SENSOR_DISTANCE + shift_factor_y  - INTER_SENSOR_DISTANCE #- 10 * INTER_SENSOR_DISTANCE * (1 - multiplier[i])
            # resized_tar[7,:] = [-0.286,0,0]
            resized_tar[:, 2] = np.copy(tar_mod[i, :, 2]) * multiplier[i]



            tar_mod[i, :, :] = resized_tar

        #print tar_mod[0,:,:], 'post'
        targets = np.reshape(tar_mod, (targets.shape[0], targets.shape[1])) * 1000

        return images, targets, pcons


    def synthetic_shiftxy(self, images, targets, bedangles):

        #use bed angles to keep it from shifting in the x and y directions

        x = np.arange(-5, 6)
        xU, xL = x + 0.5, x - 0.5
        prob = ss.norm.cdf(xU, scale=2) - ss.norm.cdf(xL, scale=2) #scale is the standard deviation using a cumulative density function
        prob = prob / prob.sum()  # normalize the probabilities so their sum is 1
        modified_x = np.random.choice(x, size=images.shape[0], p=prob)
        #plt.hist(modified_x)
        #plt.show()

        y = np.arange(-5, 6)
        yU, yL = y + 0.5, y - 0.5
        prob = ss.norm.cdf(yU, scale=2) - ss.norm.cdf(yL, scale=2)
        prob = prob / prob.sum()  # normalize the probabilities so their sum is 1
        modified_y = np.random.choice(y, size=images.shape[0], p=prob)
        modified_y[bedangles[:,0,0] > 10] = 0 #we have to cut out the vertical shifts where the bed is not flat
        #plt.hist(modified_y)
        #plt.show()

        tar_mod = np.reshape(targets, (targets.shape[0], targets.shape[1] / 3, 3))

        # print images[0,30:34,10:14]
        # print modified_x[0]
        for i in np.arange(images.shape[0]):
            if self.include_inter == True:
                if modified_x[i] > 0:
                    images[i, :, :, modified_x[i]:] = images[i, :, :, 0:-modified_x[i]]
                elif modified_x[i] < 0:
                    images[i, :, :, 0:modified_x[i]] = images[i, :, :, -modified_x[i]:]

                if modified_y[i] > 0:
                    images[i, :, modified_y[i]:, :] = images[i, :, 0:-modified_y[i], :]
                elif modified_y[i] < 0:
                    images[i, :, 0:modified_y[i], :] = images[i, :, -modified_y[i]:, :]

            else:
                if modified_x[i] > 0:
                    images[i, :, modified_x[i]:] = images[i, :, 0:-modified_x[i]]
                elif modified_x[i] < 0:
                    images[i, :, 0:modified_x[i]] = images[i, :, -modified_x[i]:]

                if modified_y[i] > 0:
                    images[i, modified_y[i]:, :] = images[i, 0:-modified_y[i], :]
                elif modified_y[i] < 0:
                    images[i, 0:modified_y[i], :] = images[i, -modified_y[i]:, :]

            tar_mod[i, :, 0] += modified_x[i] * INTER_SENSOR_DISTANCE * 1000
            tar_mod[i, :, 1] -= modified_y[i] * INTER_SENSOR_DISTANCE * 1000

        # print images[0, 30:34, 10:14]
        targets = np.reshape(tar_mod, (targets.shape[0], targets.shape[1]))

        return images, targets


    def synthetic_fliplr(self, images, targets, pcons = None):
        coin = np.random.randint(2, size=images.shape[0])
        modified = coin
        original = 1 - coin

        if self.include_inter == True:
            im_orig = np.multiply(images, original[:, np.newaxis, np.newaxis, np.newaxis])
            im_mod = np.multiply(images, modified[:, np.newaxis, np.newaxis, np.newaxis])
            # flip the x axis on all the modified pressure mat images
            im_mod = im_mod[:, :, :, ::-1]
        else:
            im_orig = np.multiply(images, original[:, np.newaxis, np.newaxis])
            im_mod = np.multiply(images, modified[:, np.newaxis, np.newaxis])
            # flip the x axis on all the modified pressure mat images
            im_mod = im_mod[:, :, ::-1]



        tar_orig = np.multiply(targets, original[:, np.newaxis])
        tar_mod = np.multiply(targets, modified[:, np.newaxis])

        #print pcons.shape, 'pconshape'

        # change the left and right tags on the target in the z, flip x target left to right
        tar_mod = np.reshape(tar_mod, (tar_mod.shape[0], tar_mod.shape[1] / 3, 3))

        # flip the x left to right
        tar_mod[:, :, 0] = (tar_mod[:, :, 0] - 657.8) * -1 + 657.8

        # swap in the z
        dummy = zeros((tar_mod.shape))

        if self.loss_vector_type == 'anglesCL' or self.loss_vector_type == 'anglesVL' or self.loss_vector_type == 'anglesSTVL':

            dummy[:, [2, 4, 6, 8], :] = tar_mod[:, [2, 4, 6, 8], :]
            tar_mod[:, [2, 4, 6, 8], :] = tar_mod[:, [3, 5, 7, 9], :]
            tar_mod[:, [3, 5, 7, 9], :] = dummy[:, [2, 4, 6, 8], :]
            if pcons is not None:
                pcons_orig = np.multiply(pcons, original[:, np.newaxis])
                pcons_mod = np.multiply(pcons, modified[:, np.newaxis])
                dummy2 = zeros((pcons_mod.shape))

                dummy2[:, [0, 2, 4, 6, 10, 12, 14, 16, 22, 24, 31, 33]] = pcons_mod[:, [0, 2, 4, 6, 10, 12, 14, 16, 22, 24, 31, 33]]
                pcons_mod[:, [0, 2, 4, 6, 10, 12, 14, 16, 22, 24, 31, 33]] = pcons_mod[:, [1, 3, 5, 7, 11, 13, 15, 17, 23, 25, 32, 34]]
                pcons_mod[:, [1, 3, 5, 7, 11, 13, 15, 17, 23, 25, 32, 34]] = dummy2[:, [0, 2, 4, 6, 10, 12, 14, 16, 22, 24, 31, 33]]
                pcons_mod = np.multiply(pcons_mod, modified[:, np.newaxis])
                pcons = pcons_orig + pcons_mod


        elif self.loss_vector_type == 'direct':
            dummy[:, [2, 4, 6, 8], :] = tar_mod[:, [2, 4, 6, 8], :]
            tar_mod[:, [2, 4, 6, 8], :] = tar_mod[:, [3, 5, 7, 9], :]
            tar_mod[:, [3, 5, 7, 9], :] = dummy[:, [2, 4, 6, 8], :]
        # print dummy[0,:,2], tar_mod[0,:,2]

        tar_mod = np.reshape(tar_mod, (tar_mod.shape[0], tar_orig.shape[1]))
        tar_mod = np.multiply(tar_mod, modified[:, np.newaxis])

        images = im_orig + im_mod
        targets = tar_orig + tar_mod
        return images, targets, pcons


    def synthetic_master(self, images_tensor, targets_tensor, pcons_tensor = None, flip=False, shift=False, scale=False, bedangle = False, include_inter = False, loss_vector_type = False):
        self.loss_vector_type = loss_vector_type
        self.include_inter = include_inter
        self.t1 = time.time()
        images_tensor = torch.squeeze(images_tensor)
        # images_tensor.torch.Tensor.permute(1,2,0)
        imagesangles = images_tensor.numpy()
        targets = targets_tensor.numpy()

        if pcons_tensor is not None: pcons = pcons_tensor.numpy()
        else: pcons = None
        if len(imagesangles.shape) < 4:
            imagesangles = np.expand_dims(imagesangles, 0)


        if bedangle == True:
            if include_inter == True:
                images = imagesangles[:, 0:2, :, :]
                bedangles = imagesangles[:, 2, :, :]
            else:
                images = imagesangles[:,0,:,:]
                bedangles = imagesangles[:,1,20,20]
                #print bedangles.shape
                #print targets.shape,'targets for synthetic code'
        else:
            images = imagesangles
            bedangles = None
        #print images.shape, targets.shape, 'shapes'


        if scale == True:
            images, targets, pcons = self.synthetic_scale(images, targets, bedangles, pcons)
        if flip == True:
            images, targets, pcons = self.synthetic_fliplr(images, targets, pcons)
        if shift == True:
            images, targets = self.synthetic_shiftxy(images, targets, bedangles)

        # print images[0, 10:15, 20:25]

        if bedangle == True:
            if include_inter == True:
                imagesangles[:,0:2,:,:] = images
            else:
                imagesangles[:,0,:,:] = images
            images_tensor = torch.Tensor(imagesangles)
        else:
            imagesangles = images
            images_tensor = torch.Tensor(imagesangles)
            images_tensor = images_tensor.unsqueeze(1)

        if pcons is not None: pcons_tensor = torch.Tensor(pcons)
        targets_tensor = torch.Tensor(targets)
        # images_tensor.torch.Tensor.permute(2, 0, 1)
        try:
            self.t2 = time.time() - self.t1
        except:
            self.t2 = 0
        # print self.t2, 'elapsed time'
        return images_tensor, targets_tensor, pcons_tensor

