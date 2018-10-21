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
from skimage import data, color, exposure, feature

from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn import svm, linear_model, decomposition, kernel_ridge, neighbors
from sklearn import metrics
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




class PreprocessingLib():
    def chi2_distance(self, histA, histB, eps = 1e-10):
        # compute the chi-squared distance
        d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
                for (a, b) in zip(histA, histB)])
        # return the chi-squared distance
        return d



    def compute_HoG(self, data):
        '''Computes a HoG(Histogram of gradients for a list of images provided
        to it. Returns a list of flattened lists'''
        flat_hog = []
        print "*****Begin HoGing the dataset*****"
        count = 0
        for index in range(len(data)):
            count += 1
            if count == 200:
                count = 0
                print "HoG being applied over image number: {}".format(index)

            #fd = data[index].flatten()
            #Compute HoG of the current pressure map
            fd, hog_image = hog(data[index], orientations=8,
                    pixels_per_cell=(4,4), cells_per_block = (1, 1),
                    visualise=True)

            flat_hog.append(fd)
        return flat_hog


    def compute_pixel_variance(self, data):
        self.verbose = True
        weight_matrix = np.std(data, axis=0)
        if self.verbose == True: print len(weight_matrix),'x', len(weight_matrix[0]), 'size of weight matrix'
        weight_matrix = weight_matrix/weight_matrix.max()
        print len(weight_matrix)
        print weight_matrix[0].shape

        x = np.zeros((20, 54))
        y = np.hstack((
                np.hstack((np.ones((60,10)), np.zeros((60, 32)))),
                np.ones((60,12))))
        z = np.ones((48, 54))
        weight_matrix = np.vstack((np.vstack((x,y)), z))

        print len(weight_matrix)
        print weight_matrix[0].shape

        matshow(weight_matrix, fignum=100, cmap=cm.gray)
        show()
        if self.verbose == True: print len(x),'x', len(x[0]), 'size of x matrix'
        if self.verbose == True: print len(y),'x', len(y[0]), 'size of y matrix'
        if self.verbose == True: print len(z),'x', len(z[0]), 'size of z matrix'
        return weight_matrix

    def preprocessing_add_image_noise(self, images):

        queue = np.copy(images[:, 0:2, :, :])
        queue[queue != 0] = 1.

        x = np.arange(-10, 10)
        xU, xL = x + 0.5, x - 0.5
        prob = ss.norm.cdf(xU, scale=1) - ss.norm.cdf(xL,scale=1)  # scale is the standard deviation using a cumulative density function
        prob = prob / prob.sum()  # normalize the probabilities so their sum is 1
        image_noise = np.random.choice(x, size=(images.shape[0], images.shape[1]-1, images.shape[2], images.shape[3]), p=prob)


        #image_noise = image_noise*queue
        images[:, 0:2, :, :] += image_noise

        #print images[0, 0, 50, 10:25], 'added noise'

        #clip noise so we dont go outside sensor limits
        images[:, 0, :, :] = np.clip(images[:, 0, :, :], 0, 100)
        images[:, 1, :, :] = np.clip(images[:, 1, :, :], 0, 10000)
        return images


    def preprocessing_pressure_array_resize(self, data, mat_size, verbose):
        '''Will resize all elements of the dataset into the dimensions of the
        pressure map'''
        p_map_dataset = []
        for map_index in range(len(data)):
            #print map_index, self.mat_size, 'mapidx'
            #Resize mat to make into a matrix
            p_map = np.reshape(data[map_index], mat_size)
            #print p_map
            p_map_dataset.append(p_map)
            #print p_map.shape
        if verbose: print len(data[0]),'x',1, 'size of an incoming pressure map'
        if verbose: print len(p_map_dataset[0]),'x',len(p_map_dataset[0][0]), 'size of a resized pressure map'
        return p_map_dataset

    def preprocessing_create_pressure_angle_stack_realtime(self, p_map, bedangle, mat_size):
        '''This is for creating a 2-channel input using the height of the bed. '''
        p_map = np.reshape(p_map, mat_size)
        print np.shape(p_map)
        print p_map.shape
        print np.shape(bedangle), 'angle dat'

        print 'calculating height matrix and sobel filter'
        p_map_dataset = []


        height_strip = np.zeros(np.shape(p_map)[0])
        height_strip[0:25] = np.flip(np.linspace(0, 1, num=25) * 25 * 2.86 * np.sin(np.deg2rad(bedangle)),
                                      axis=0)
        height_strip = np.repeat(np.expand_dims(height_strip, axis=1), 27, 1)
        a_map = height_strip


        # this makes a sobel edge on the image
        sx = ndimage.sobel(p_map, axis=0, mode='constant')
        sy = ndimage.sobel(p_map, axis=1, mode='constant')
        p_map_inter = np.hypot(sx, sy)

        print np.shape(p_map_inter)
        p_map_dataset.append([p_map, p_map_inter, a_map])

        return p_map_dataset


    def preprocessing_create_pressure_angle_stack(self,x_data, a_data, include_inter, mat_size, verbose):
        '''This is for creating a 2-channel input using the height of the bed. '''
        print np.shape(x_data)
        print np.shape(a_data), 'angle dat'

        print 'calculating height matrix and sobel filter'
        p_map_dataset = []
        for map_index in range(len(x_data)):
            # print map_index, self.mat_size, 'mapidx'
            # Resize mat to make into a matrix
            p_map = np.reshape(x_data[map_index], mat_size)
            height_strip = np.zeros(np.shape(p_map)[0])
            height_strip[10:35] = np.flip(np.linspace(0, 1, num=25) * 25 * 2.86 * np.sin(np.deg2rad(a_data[map_index])), axis = 0)
            height_strip = np.repeat(np.expand_dims(height_strip, axis = 1), 47, 1)
            a_map = height_strip

            #ZACKORY: ALTER THE VARIABLE "p_map" HERE TO STANDARDIZE. IT IS AN 84x47 MATRIX WITHIN EACH LOOP.
            #THIS ALTERATION WILL ALSO CHANGE HOW THE EDGE IS CALCULATED. IF YOU WANT TO DO THEM SEPARATELY,
            #THEN DO IT AFTER THE 'INCLUDE INTER' IF STATEMENT.
            #p_map = standardize(p_map)


            if include_inter == True:
                # this makes a sobel edge on the image
                sx = ndimage.sobel(p_map, axis=0, mode='constant')
                sy = ndimage.sobel(p_map, axis=1, mode='constant')
                p_map_inter = np.hypot(sx, sy)
                p_map_dataset.append([p_map, p_map_inter, a_map])
            else:
                p_map_dataset.append([p_map, a_map])
        if verbose: print len(x_data[0]), 'x', 1, 'size of an incoming pressure map'
        if verbose: print len(p_map_dataset[0][0]), 'x', len(p_map_dataset[0][0][0]), 'size of a resized pressure map'
        if verbose: print len(p_map_dataset[0][1]), 'x', len(p_map_dataset[0][1][0]), 'size of the stacked angle mat'
        return p_map_dataset


    def preprocessing_pressure_map_upsample(self, data, multiple=4, order=1):
        '''Will upsample an incoming pressure map dataset'''
        p_map_highres_dataset = []

        if len(np.shape(data)) == 3:
            for map_index in range(len(data)):
                #Upsample the current map using bilinear interpolation
                p_map_highres_dataset.append(
                        ndimage.zoom(data[map_index], multiple, order=order))
        elif len(np.shape(data)) == 4:
            for map_index in range(len(data)):
                p_map_highres_dataset_subindex = []
                for map_subindex in range(len(data[map_index])):
                    #Upsample the current map using bilinear interpolation
                    p_map_highres_dataset_subindex.append(ndimage.zoom(data[map_index][map_subindex], multiple, order=order))
                p_map_highres_dataset.append(p_map_highres_dataset_subindex)

        return p_map_highres_dataset



    def pad_pressure_mats(self,NxHxWimages):
        padded = np.zeros((NxHxWimages.shape[0],NxHxWimages.shape[1]+20,NxHxWimages.shape[2]+20))
        padded[:,10:74,10:37] = NxHxWimages
        NxHxWimages = padded
        return NxHxWimages



    def person_based_loocv(self):
        '''Computes Person Based Leave One Out Cross Validation. This means
        that if we have 10 participants, we train using 9 participants and test
        on 1 participant, and so on.
        To run this function, make sure that each subject_* directory in the
        dataset/ directory has a pickle file called individual_database.p
        If you don't have it in some directory that means you haven't run,
        create_raw_database.py on that subject's dataset. So create it and
        ensure that the pkl file is created successfully'''
        #Entire pressure dataset with coordinates in world frame
        dataset_dirname = os.path.dirname(os.path.realpath(training_database_file))
        print dataset_dirname
        subject_dirs = [x[0] for x in os.walk(dataset_dirname)]
        subject_dirs.pop(0)
        print subject_dirs
        dat = []
        for i in range(len(subject_dirs)):
            try:
                dat.append(pkl.load(open(os.path.join(subject_dirs[i],
                    'individual_database.p'), "rb")))
            except:
                print "Following dataset directory not formatted correctly. Is there an individual_dataset pkl file for every subject?"
                print os.path.join(subject_dirs[i], 'individual_database.p')
                sys.exit()
        print "Inserted all individual datasets into a list of dicts"
        print "Number of subjects:"
        print len(dat)
        mean_joint_error = np.zeros((len(dat), 10))
        std_joint_error = np.zeros((len(dat), 10))
        for i in range(len(dat)):
            train_dat = {}
            test_dat = dat[i]
            for j in range(len(dat)):
                if j == i:
                    print "#of omitted data points"
                    print len(dat[j].keys())
                    pass
                else:
                    print len(dat[j].keys())
                    print j
                    train_dat.update(dat[j])
            rand_keys = train_dat.keys()
            print "Training Dataset Size:"
            print len(rand_keys)
            print "Testing dataset size:"
            print len(test_dat.keys())
            self.train_y = [] #Initialize the training coordinate list
            self.dataset_y = [] #Initialization for the entire dataset
            self.train_x_flat = rand_keys[:]#Pressure maps
            [self.train_y.append(train_dat[key]) for key in self.train_x_flat]#Coordinates
            self.test_x_flat = test_dat.keys()#Pressure maps(test dataset)
            self.test_y = [] #Initialize the ground truth list
            [self.test_y.append(test_dat[key]) for key in self.test_x_flat]#ground truth
            self.dataset_x_flat = rand_keys[:]#Pressure maps
            [self.dataset_y.append(train_dat[key]) for key in self.dataset_x_flat]
            self.cv_fold = 3 # Value of k in k-fold cross validation
            self.mat_frame_joints = []
            p.train_hog_knn()
            (mean_joint_error[i][:], std_joint_error[i][:]) = self.test_learning_algorithm(self.regr)
            print "Mean Error:"
            print mean_joint_error
        print "MEAN ERROR AFTER PERSON LOOCV:"
        total_mean_error = np.mean(mean_joint_error, axis=0)
        total_std_error = np.mean(std_joint_error, axis=0)
        print total_mean_error
        print "STD DEV:"
        print total_std_error
        pkl.dump(mean_joint_error, open('./dataset/mean_loocv_results.p', 'w'))
        pkl.dump(mean_joint_error, open('./dataset/std_loocv_results.p', 'w'))

