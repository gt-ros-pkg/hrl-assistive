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

#ROS
import rospy
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray


import pickle
from hrl_lib.util import load_pickle


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




class CascadeLib():

    def print_error(self, target, score, output_size, loss_vector_type = None, data = None, printerror = True):

        error = (score - target)
        error = np.reshape(error, (error.shape[0], output_size[0], output_size[1]))
        error_norm = np.expand_dims(np.linalg.norm(error, axis = 2),2)
        error = np.concatenate((error, error_norm), axis = 2)

        error_avg = np.mean(error, axis=0) / 10
        error_avg = np.reshape(error_avg, (output_size[0], output_size[1]+1))
        error_avg = np.reshape(np.array(["%.2f" % w for w in error_avg.reshape(error_avg.size)]),
                                     (output_size[0], output_size[1] + 1))

        if loss_vector_type == 'upper_angles':
            error_avg = np.transpose(np.concatenate(([['Average Error for Last Batch', '       ', ' Head  ', ' Torso ', 'R Elbow', 'L Elbow',
                                                       'R Hand ', 'L Hand ']], np.transpose(
                np.concatenate(([['', '', '', ''], [' x, cm ', ' y, cm ', ' z, cm ', '  norm ']], error_avg))))))
        elif loss_vector_type == 'all_joints' or loss_vector_type == 'angles':
            error_avg = np.transpose(np.concatenate(([['Average Error for Last Batch', '       ', 'Head   ',
                                                       'Torso  ', 'R Elbow', 'L Elbow', 'R Hand ', 'L Hand ',
                                                       'R Knee ', 'L Knee ', 'R Foot ', 'L Foot ']], np.transpose(
                np.concatenate(([['', '', '', ''], [' x, cm ', ' y, cm ', ' z, cm ', '  norm ']], error_avg))))))
        elif loss_vector_type == 'arms_cascade':
            error_avg = np.transpose(np.concatenate(([['Average Error for Last Batch', '       ', 'R Elbow', 'R Hand ']], np.transpose(
                np.concatenate(([['', '', '', ''], [' x, cm ', ' y, cm ', ' z, cm ', '  norm ']], error_avg))))))
        else:
            error_avg = np.transpose(np.concatenate(([['Average Error for Last Batch', '       ', ' Torso ', 'R Elbow', 'L Elbow',
                                  'R Hand ', 'L Hand ']], np.transpose(np.concatenate(
                        ([['', '', '', ''], [' x, cm ', ' y, cm ', ' z, cm ', '  norm ']], error_avg))))))
        if printerror == True:
            print data, error_avg


        error_std = np.std(error, axis=0) / 10
        error_std = np.reshape(error_std, (output_size[0], output_size[1] + 1))
        error_std = np.reshape(np.array(["%.2f" % w for w in error_std.reshape(error_std.size)]),
                                     (output_size[0], output_size[1] + 1))

        if loss_vector_type == 'upper_angles':
            error_std = np.transpose(np.concatenate(([['Error Standard Deviation for Last Batch', '       ', ' Head  ', ' Torso ', 'R Elbow',
                                                       'L Elbow', 'R Hand ', 'L Hand ']], np.transpose(
                np.concatenate(([['', '', '',''], ['x, cm', 'y, cm', 'z, cm', '  norm ']], error_std))))))
        elif loss_vector_type == 'all_joints' or loss_vector_type == 'angles':
            error_std = np.transpose(np.concatenate(([['Error Standard Deviation for Last Batch', '       ', 'Head   ', 'Torso  ',
                                  'R Elbow', 'L Elbow', 'R Hand ', 'L Hand ', 'R Knee ', 'L Knee ',
                                  'R Foot ', 'L Foot ']], np.transpose(
                    np.concatenate(([['', '', '', ''], ['x, cm', 'y, cm', 'z, cm', '  norm ']], error_std))))))
        elif loss_vector_type == 'arms_cascade':
            error_std = np.transpose(
                np.concatenate(([['Error Standard Deviation for Last Batch', '       ', 'R Elbow', 'R Hand ']], np.transpose(
                    np.concatenate(([['', '', '', ''], ['x, cm', 'y, cm', 'z, cm', '  norm ']], error_std))))))
        else:
            error_std = np.transpose(np.concatenate(([['Error Standard Deviation for Last Batch', '       ', ' Torso ', 'R Elbow',
                                                       'L Elbow', 'R Hand ', 'L Hand ']], np.transpose(
                np.concatenate(([['', '', '',''], ['x, cm', 'y, cm', 'z, cm', '  norm ']], error_std))))))
        #if printerror == True:
        #    print data, error_std
        error_norm = np.squeeze(error_norm, axis = 2)
        return error_norm



    def visualize_pressure_map(self, p_map, targets_raw=None, scores_raw = None, p_map_val = None, targets_val = None, scores_val = None, block = False):
        p_map = p_map[0,:,:] #select the original image matrix from the intermediate amplifier matrix and the height matrix

        plt.close()
        plt.pause(0.0001)

        fig = plt.figure()
        mngr = plt.get_current_fig_manager()
        # to put it into the upper left corner for example:
        #mngr.window.setGeometry(50, 100, 840, 705)

        plt.pause(0.0001)

        # set options
        if p_map_val is not None:
            p_map_val = p_map_val[0, :, :] #select the original image matrix from the intermediate amplifier matrix and the height matrix
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2)
            xlim = [-2.0, 49.0]
            ylim = [86.0, -2.0]
            ax1.set_xlim(xlim)
            ax1.set_ylim(ylim)
            ax2.set_xlim(xlim)
            ax2.set_ylim(ylim)
            ax1.set_axis_bgcolor('cyan')
            ax2.set_axis_bgcolor('cyan')
            ax1.imshow(p_map, interpolation='nearest', cmap=
            plt.cm.bwr, origin='upper', vmin=0, vmax=100)
            ax2.imshow(p_map_val, interpolation='nearest', cmap=
            plt.cm.bwr, origin='upper', vmin=0, vmax=100)
            ax1.set_title('Training Sample \n Targets and Estimates')
            ax2.set_title('Validation Sample \n Targets and Estimates')


        else:
            ax1 = fig.add_subplot(1, 1, 1)
            xlim = [-2.0, 49.0]
            ylim = [86.0, -2.0]
            ax1.set_xlim(xlim)
            ax1.set_ylim(ylim)
            ax1.set_axis_bgcolor('cyan')
            ax1.imshow(p_map, interpolation='nearest', cmap=
            plt.cm.bwr, origin='upper', vmin=0, vmax=100)
            ax1.set_title('Validation Sample \n Targets and Estimates')

        # Visualize targets of training set
        if targets_raw is not None:
            if len(np.shape(targets_raw)) == 1:
                targets_raw = np.reshape(targets_raw, (len(targets_raw) / 3, 3))
            target_coord = targets_raw[:, :2] / INTER_SENSOR_DISTANCE
            target_coord[:, 1] -= (NUMOFTAXELS_X - 1)
            target_coord[:, 1] *= -1.0
            ax1.plot(target_coord[:, 0], target_coord[:, 1], marker = 'o', linestyle='None', markerfacecolor = 'green',markeredgecolor='black', ms=8)
        plt.pause(0.0001)

        #Visualize estimated from training set
        if scores_raw is not None:
            if len(np.shape(scores_raw)) == 1:
                scores_raw = np.reshape(scores_raw, (len(scores_raw) / 3, 3))
            target_coord = scores_raw[:, :2] / INTER_SENSOR_DISTANCE
            target_coord[:, 1] -= (NUMOFTAXELS_X - 1)
            target_coord[:, 1] *= -1.0
            ax1.plot(target_coord[:, 0], target_coord[:, 1], marker = 'o', linestyle='None', markerfacecolor = 'white',markeredgecolor='black', ms=8)
        plt.pause(0.0001)

        # Visualize targets of validation set
        if targets_val is not None:
            if len(np.shape(targets_val)) == 1:
                targets_val = np.reshape(targets_val, (len(targets_val) / 3, 3))
            target_coord = targets_val[:, :2] / INTER_SENSOR_DISTANCE
            target_coord[:, 1] -= (NUMOFTAXELS_X - 1)
            target_coord[:, 1] *= -1.0
            ax2.plot(target_coord[:, 0], target_coord[:, 1], marker = 'o', linestyle='None', markerfacecolor = 'green',markeredgecolor='black', ms=8)
        plt.pause(0.0001)

        # Visualize estimated from training set
        if scores_val is not None:
            if len(np.shape(scores_val)) == 1:
                scores_val = np.reshape(scores_val, (len(scores_val) / 3, 3))
            target_coord = scores_val[:, :2] / INTER_SENSOR_DISTANCE
            target_coord[:, 1] -= (NUMOFTAXELS_X - 1)
            target_coord[:, 1] *= -1.0
            ax2.plot(target_coord[:, 0], target_coord[:, 1], marker = 'o', linestyle='None', markerfacecolor = 'white',markeredgecolor='black', ms=8)
        plt.pause(0.0001)
        plt.show(block = block)
        return

    def generate_bounded_box_input(self, images, box_centers, box_length = 17):

        image_boxes = np.zeros((box_centers.shape[0], 4, box_length, box_length))
        image_box_coords = np.zeros((box_centers.shape[0], 4, 4))

        #NEED TO VECTORIZE!!!!
        for i in range(box_centers.shape[0]):
            #i = 0

            y1, y2 = (84 - box_centers[i, 11, 1] - 8), (84 - box_centers[i, 11, 1] + 9)
            y1p = (-y1).clip(min=0)
            y2p = (y2 - images.shape[2]).clip(min=0)
            x1, x2 = box_centers[i, 11, 0] - 8, box_centers[i, 11, 0] + 9
            x1p = (-x1).clip(min=0)
            x2p = (x2 - images.shape[3]).clip(min=0)
            image_boxes[i, 0, int(y2p):box_length - int(y1p), int(x1p):box_length - int(x2p)] = images[i, 0,int(y1) + int(y1p):int(y2) - int(y2p),int(x1) + int(x1p):int(x2) - int(x2p)]

            y1, y2 = (84 - box_centers[i, 2, 1] - 8), (84 - box_centers[i, 2, 1] + 9)
            y1p = (-y1).clip(min=0)
            y2p = (y2 - images.shape[2]).clip(min=0)
            x1, x2 = box_centers[i, 2, 0] - 8, box_centers[i, 2, 0] + 9
            x1p = (-x1).clip(min=0)
            x2p = (x2 - images.shape[3]).clip(min=0)
            #print x1, x1p, x2, x2p
            #print y1, y1p, y2, y2p
            #print int(x1p), box_length-int(x2p), 'box x'
            #print int(y1p), box_length-int(y2p), 'box y'
            #print int(x1)+int(x1p), int(x2)-int(x2p), 'image x'
            #print int(y1)+int(y1p), int(y2)-int(y2p), 'image y', 'elbow'
            image_boxes[i, 1, int(y2p):box_length-int(y1p), int(x1p):box_length - int(x2p)] = images[i, 0, int(y1)+int(y1p):int(y2)-int(y2p), int(x1)+int(x1p):int(x2)-int(x2p)]

            y1, y2 = (84 - box_centers[i, 4, 1] - 8), (84 - box_centers[i, 4, 1] + 9)
            y1p = (-y1).clip(min=0)
            y2p = (y2 - images.shape[2]).clip(min=0)
            x1, x2 = box_centers[i, 4, 0] - 8, box_centers[i, 4, 0] + 9
            x1p = (-x1).clip(min=0)
            x2p = (x2 - images.shape[3]).clip(min=0)
            #print x1, x1p, x2, x2p
            #print y1, y1p, y2, y2p
            #print int(x1p), box_length - int(x2p), 'box x'
            #print int(y1p), box_length - int(y2p), 'box y'
            #print int(x1) + int(x1p), int(x2) - int(x2p), 'image x'
            #print int(y1) + int(y1p), int(y2) - int(y2p), 'image y', 'shoulder'
            image_boxes[i, 2, int(y2p):box_length - int(y1p), int(x1p):box_length - int(x2p)] = images[i, 0,int(y1) + int(y1p):int(y2) - int(y2p),int(x1) + int(x1p):int(x2) - int(x2p)]

            y1, y2 = (84 - box_centers[i, 11, 1] - 8), (84 - box_centers[i, 11, 1] + 9)
            y1p = (-y1).clip(min=0)
            y2p = (y2 - images.shape[2]).clip(min=0)
            x1, x2 = box_centers[i, 11, 0] - 8, box_centers[i, 11, 0] + 9
            x1p = (-x1).clip(min=0)
            x2p = (x2 - images.shape[3]).clip(min=0)
            image_boxes[i, 3, int(y2p):box_length - int(y1p), 0:box_length] = images[i, 2,int(y1) + int(y1p):int(y2) - int(y2p), 0:box_length]


        return image_boxes




    def get_2D_projection(self, images, targets):
        mat_size = (NUMOFTAXELS_X, NUMOFTAXELS_Y)
        bedangle = images[:, 2, 10, 10]

        queue_frame = np.zeros_like(targets)
        middleman = np.zeros_like(targets)
        middleman2 = np.zeros_like(targets)
        queue_head = np.zeros_like(targets)
        final_coords = np.zeros_like(targets)

        targets = targets/28.6

        final_coords[:, :, 0] = targets[:, :, 0]

        queue_frame[:, :, 0] = targets[:, :, 0] #x coord projected onto flat bed
        queue_frame[:, :, 1] = targets[:, :, 1] #y coord projected onto flat bed
        queue_frame[:, :, 2] = targets[:, :, 2] #z coord projected onto flat bed


        #find the distance in x/y between the bed bending point and each target
        middleman[:, :, 0] = np.sqrt(np.square(targets[:, :, 2]) + np.square(51-targets[:, :, 1]))

        #calculate the angle from 0 degrees (the point is in the plane of a flat bed) about the bed's bend and up to where it is.
        middleman2[:, :, 0] = targets[:, :, 1]
        middleman2[:, :, 0][middleman2[:, :, 0] <= 51.] = 1.
        middleman2[:, :, 0][middleman2[:, :, 0] > 51.] = 0.
        middleman2[:, :, 1] = 1 - middleman2[:, :, 0]

        middleman2[:, :, 0] *= np.pi/2 + np.arccos(targets[:, :, 2]/middleman[:, :, 0])
        middleman2[:, :, 1] *= np.pi/2 - np.arccos(targets[:, :, 2]/middleman[:, :, 0])
        middleman[:, :, 1] = middleman2[:, :, 0] + middleman2[:, :, 1]

        #For the y distance, you have to subtract the bed angle from the point rotated about the bed's bend until flat
        queue_head[:, :, 1] = 51 + middleman[:, :, 0]*np.cos(middleman[:, :, 1] - np.deg2rad(np.expand_dims(bedangle[:], axis=1))) + 6.*np.sin(np.deg2rad(np.expand_dims(bedangle[:], axis=1)))

        #z. this one is good to go.
        By = (51)  -  3 * np.sin(np.deg2rad(np.expand_dims(bedangle[:], axis=1)))
        queue_head[:, :, 2] = targets[:, :, 2] * np.cos(np.deg2rad(np.expand_dims(bedangle[:], axis=1))) - (targets[:, :, 1] - By) * np.sin(np.deg2rad(np.expand_dims(bedangle[:], axis=1)))

        queue_head[:, :, 1] = queue_head[:, :, 1]
        queue_frame[:, :, 1] = queue_frame[:, :, 1]

        #find which plane the point is closest to. If it's closest to the head of the bed, we need to select the rotated point.

        queue_frame[:, :, 2][bedangle[:] == 0.] = 100. #this is so we don't duplicate  things by having the mins equal

        middleman[:, :, 2] = np.amin([queue_frame[:, :, 2], queue_head[:, :, 2]], axis = 0)
        queue_frame[:, :, 2] -= middleman[:, :, 2]
        queue_frame[:, :, 2][queue_frame[:,:,2] == 0] = 1
        queue_frame[:, :, 2][queue_frame[:,:,2] != 1] = 0
        queue_head[:, :, 2] -= middleman[:, :, 2]
        queue_head[:, :, 2][queue_head[:,:,2] == 0] = 1
        queue_head[:, :, 2][queue_head[:,:,2] != 1] = 0
        final_coords[:, :, 1] = queue_frame[:,:,1]*queue_frame[:,:,2] + queue_head[:,:,1]*queue_head[:,:,2]

        final_coords = final_coords*28.6
        return final_coords

