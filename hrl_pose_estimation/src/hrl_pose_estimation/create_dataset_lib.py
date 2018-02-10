#!/usr/bin/env python
#Compiled by Henry M. Clever
#The purpose of this library is to shorten the lengths of other database creator scripts to make them easier to edit and peruse

import sys
import os
import numpy as np
import cPickle as pkl
import random

# ROS
#import roslib; roslib.load_manifest('hrl_pose_estimation')

# Graphics
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Machine Learning
from scipy import ndimage
from scipy.ndimage.filters import gaussian_filter
## from skimage import data, color, exposure
from sklearn.decomposition import PCA


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

class CreateDatasetLib():

    def world_to_mat(self,w_data, p_world_mat, R_world_mat):

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
        m_data = np.squeeze(np.asarray(m_data[:3, :].T))

        try:
            m_data[:,0] = m_data[:,0]+10*INTER_SENSOR_DISTANCE
            m_data[:, 1] = m_data[:, 1] + 10* INTER_SENSOR_DISTANCE
        except:
            m_data[0] = m_data[0] + 10*INTER_SENSOR_DISTANCE
            m_data[1] = m_data[1] + 10*INTER_SENSOR_DISTANCE

        return m_data


    def mat_to_taxels(self, m_data):
        '''
        Input:  Nx2 array
        Output: Nx2 array
        '''
        #Convert coordinates in 3D space in the mat frame into taxels
        taxels = m_data / INTER_SENSOR_DISTANCE

        '''Typecast into int, so that we can highlight the right taxel
        in the pressure matrix, and threshold the resulting values'''
        taxels = np.rint(taxels)

        #Thresholding the taxels_* array
        for i, taxel in enumerate(taxels):
            if taxel[1] < LOW_TAXEL_THRESH_X: taxels[i,1] = LOW_TAXEL_THRESH_X
            if taxel[0] < LOW_TAXEL_THRESH_Y: taxels[i,0] = LOW_TAXEL_THRESH_Y
            if taxel[1] > HIGH_TAXEL_THRESH_X: taxels[i,1] = HIGH_TAXEL_THRESH_X
            if taxel[0] > HIGH_TAXEL_THRESH_Y: taxels[i,0] = HIGH_TAXEL_THRESH_Y
        return taxels

    def rotate_3D_space(self, target):
        ''' Rotate the 3D target values (the 3D position of the markers
        attached to subject) using PCA'''
        # This isn't really PCA, it's more of a method to normalize to some home position through translation and rotation

        # We need only X,Y coordinates in the mat frame
        targets_mat = target[:, :2]

        # The output of PCA needs rotation by -90
        rot_targets_mat = self.pca_pixels.transform(targets_mat / INTER_SENSOR_DISTANCE) * INTER_SENSOR_DISTANCE
        rot_targets_mat = np.dot(rot_targets_mat, np.array([[0., -1.], [-1., 0.]]))

        # Translate the targets to the center of the mat so that they match the
        # pressure map
        rot_targets_mat += INTER_SENSOR_DISTANCE * np.array([float(NUMOFTAXELS_Y) / 2.0 - self.y_offset, \
                                                             float(NUMOFTAXELS_X) / 2.0 - self.x_offset])

        transformed_target = np.hstack([rot_targets_mat, target[:, 2:3]])
        return transformed_target

    def rotate_taxel_space(self, p_map):
        '''Rotates pressure map given to it using PCA and translates it back to
        center of the pressure mat.'''
        if np.shape(p_map) != self.mat_size:
            p_map = np.asarray(np.reshape(p_map, self.mat_size))
        # Get the nonzero indices
        nzero_indices = np.nonzero(p_map)
        # Perform PCA on the non-zero elements of the pressure map
        pca_x_tuples = zip(nzero_indices[1],
                           nzero_indices[0] * (-1) + (NUMOFTAXELS_X - 1))
        pca_x_pixels = np.asarray([list(elem) for elem in pca_x_tuples])
        pca_y_pixels = [p_map[elem] for elem in zip(nzero_indices[0],
                                                    nzero_indices[1])]

        # Perform PCA in the space of pressure mat pixels
        self.pca_pixels = PCA(n_components=2)
        self.pca_pixels.fit(pca_x_pixels)
        # The output of PCA needs rotation by -90
        rot_x_pixels = self.pca_pixels.transform(pca_x_pixels)
        rot_x_pixels = np.dot(rot_x_pixels, np.array([[0., -1.], [-1., 0.]]))

        # Daehyung: Adjust the center using the existence of real value pixels
        min_y = 1000
        max_y = 0
        min_x = 1000
        max_x = 0
        min_pressure = 0.3
        for i in xrange(len(rot_x_pixels)):

            if rot_x_pixels[i][0] < min_y and pca_y_pixels[i] > min_pressure:
                min_y = rot_x_pixels[i][0]
            if rot_x_pixels[i][0] > max_y and pca_y_pixels[i] > min_pressure:
                max_y = rot_x_pixels[i][0]
            if rot_x_pixels[i][1] < min_x and pca_y_pixels[i] > min_pressure:
                min_x = rot_x_pixels[i][1]
            if rot_x_pixels[i][1] > max_x and pca_y_pixels[i] > min_pressure:
                max_x = rot_x_pixels[i][1]

        self.y_offset = (max_y + min_y) / 2.0
        self.x_offset = (max_x + min_x) / 2.0

        rot_trans_x_pixels = np.asarray(
            [np.asarray(elem) + np.array([NUMOFTAXELS_Y / 2. - self.y_offset, \
                                          NUMOFTAXELS_X / 2. - self.x_offset])
             for elem in rot_x_pixels])

        # Convert the continuous pixel location into integer format
        rot_trans_x_pixels = np.rint(rot_trans_x_pixels)

        # Thresholding the rotated matrices
        rot_trans_x_pixels[rot_trans_x_pixels < LOW_TAXEL_THRESH_X] = (
            LOW_TAXEL_THRESH_X)
        rot_trans_x_pixels[rot_trans_x_pixels[:, 1] >= NUMOFTAXELS_X] = (
            NUMOFTAXELS_X - 1)

        rot_trans_x_pixels[rot_trans_x_pixels[:, 0] >= NUMOFTAXELS_Y] = (
            NUMOFTAXELS_Y - 1)

        rotated_p_map_coord = ([tuple([(-1) * (elem[1] - (NUMOFTAXELS_X - 1)),
                                       elem[0]]) for elem in rot_trans_x_pixels])
        # Creating rotated p_map
        rotated_p_map = np.zeros([NUMOFTAXELS_X, NUMOFTAXELS_Y])
        # print rotated_p_map[55][15]
        # print rotated_p_map[15][55]

        for i in range(len(pca_y_pixels)):
            # print i
            # print len(pca_y_pixels)
            # print rotated_p_map_coord[i][0],rotated_p_map_coord[i][1]
            # print pca_y_pixels[i]
            # print rotated_p_map
            # print rotated_p_map[(0,13)]
            rotated_p_map[int(rotated_p_map_coord[i][0])][int(rotated_p_map_coord[i][1])] = pca_y_pixels[i]
            # rotated_p_map[int(rotated_p_map_coord[i])] = pca_y_pixels[i]
        # print rotated_p_map[30]
        return rotated_p_map

