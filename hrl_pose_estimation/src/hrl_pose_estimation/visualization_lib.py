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




class VisualizationLib():

    def print_error(self, target, score, output_size, loss_vector_type = None, data = None, printerror = True):
        if loss_vector_type == 'torso_lengths':
            error = (score[:, 0:3] - target[:, 0:3])
            error = error.data.numpy()
            error_norm = np.expand_dims(np.linalg.norm(error, axis = 1),1)
            error = np.concatenate((error, error_norm), axis = 1)


            error_avg = np.mean(error, axis=0)
            error_avg = np.reshape(error_avg, (output_size[0], output_size[1]+1))
            error_avg = np.reshape(np.array(["%.2f" % w for w in error_avg.reshape(error_avg.size)]),
                                         (output_size[0], output_size[1] + 1))
            error_avg = np.transpose(
                np.concatenate(([['Average Torso Error for Last Batch', '       ', ' Torso ']], np.transpose(
                    np.concatenate(([['', '', '', ''], [' x, cm ', ' y, cm ', ' z, cm ', '  norm ']], error_avg))))))

            lengths_error = (score[:, 3:11] - target[:, 3:11])
            lengths_error = lengths_error.data.numpy()
            lengths_error_avg = np.mean(lengths_error, axis = 0)
            lengths_error_avg = np.reshape(lengths_error_avg, (4,2))
            lengths_error_avg = np.reshape(np.array(["%.2f" % w for w in lengths_error_avg.reshape(lengths_error_avg.size)]),
                                           (4,2))
            lengths_error_avg = np.transpose(
                np.concatenate(([['Average Lengths Error for Last Batch', '                 ', '     Z + Vert    ','Spine to Shoulder', '   Upper Arm     ', '   Forearm       ']], np.transpose(
                    np.concatenate(([['', ''], [' R, cm ', ' L, cm ']], lengths_error_avg))))))
            if printerror ==True:
                print data, error_avg
                print lengths_error_avg


            error_std = np.std(error, axis = 0)
            error_std = np.reshape(error_std, (output_size[0], output_size[1]+1))
            error_std = np.reshape(np.array(["%.2f" % w for w in error_std.reshape(error_std.size)]),
                                         (output_size[0], output_size[1] + 1))
            error_std = np.transpose(
                np.concatenate(([['Average Torso Standard Deviation for Last Batch', '       ', ' Torso ']], np.transpose(
                    np.concatenate(([['', '', '', ''], [' x, cm ', ' y, cm ', ' z, cm ', '  norm ']], error_std))))))

            lengths_error_std = np.std(lengths_error, axis=0)
            lengths_error_std = np.reshape(lengths_error_std, (4, 2))
            lengths_error_std = np.reshape(
                np.array(["%.2f" % w for w in lengths_error_std.reshape(lengths_error_std.size)]),
                (4, 2))
            lengths_error_std = np.transpose(
                np.concatenate(([['Average Lengths Standard Deviation for Last Batch', '                 ', '     Z + Vert    ',
                                  'Spine to Shoulder', '   Upper Arm     ', '   Forearm       ']], np.transpose(
                    np.concatenate(([['', ''], [' R, cm ', ' L, cm ']], lengths_error_std))))))
            if printerror == True:
                print data, error_std
                print lengths_error_std


        else:
            error = (score - target)
            error = error.data.numpy()
            error = np.reshape(error, (error.shape[0], output_size[0], output_size[1]))
            error_norm = np.expand_dims(np.linalg.norm(error, axis = 2),2)
            error = np.concatenate((error, error_norm), axis = 2)

            error_avg = np.mean(error, axis=0) / 10
            error_avg = np.reshape(error_avg, (output_size[0], output_size[1]+1))
            error_avg = np.reshape(np.array(["%.2f" % w for w in error_avg.reshape(error_avg.size)]),
                                         (output_size[0], output_size[1] + 1))

            if loss_vector_type == 'arm_angles':
                error_avg = np.transpose(np.concatenate(([['Average Error for Last Batch', '       ', ' Torso ', 'R Elbow', 'L Elbow',
                                                           'R Hand ', 'L Hand ']], np.transpose(
                    np.concatenate(([['', '', '', ''], [' x, cm ', ' y, cm ', ' z, cm ', '  norm ']], error_avg))))))
            elif loss_vector_type == 'all_joints':
                error_avg = np.transpose(np.concatenate(([['Average Error for Last Batch', '       ', 'Head   ',
                                                           'Torso  ', 'R Elbow', 'L Elbow', 'R Hand ', 'L Hand ',
                                                           'R Knee ', 'L Knee ', 'R Foot ', 'L Foot ']], np.transpose(
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

            if loss_vector_type == 'arm_angles':
                error_std = np.transpose(np.concatenate(([['Error Standard Deviation for Last Batch', '       ', ' Torso ', 'R Elbow',
                                                           'L Elbow', 'R Hand ', 'L Hand ']], np.transpose(
                    np.concatenate(([['', '', '',''], ['x, cm', 'y, cm', 'z, cm', '  norm ']], error_std))))))
            elif loss_vector_type == 'all_joints':
                error_std = np.transpose(
                    np.concatenate(([['Error Standard Deviation for Last Batch', '       ', 'Head   ', 'Torso  ',
                                      'R Elbow', 'L Elbow', 'R Hand ', 'L Hand ', 'R Knee ', 'L Knee ',
                                      'R Foot ', 'L Foot ']], np.transpose(
                        np.concatenate(([['', '', ''], ['x, cm', 'y, cm', 'z, cm', '  norm ']], error_std))))))
            else:
                error_std = np.transpose(np.concatenate(([['Error Standard Deviation for Last Batch', '       ', ' Torso ', 'R Elbow',
                                                           'L Elbow', 'R Hand ', 'L Hand ']], np.transpose(
                    np.concatenate(([['', '', '',''], ['x, cm', 'y, cm', 'z, cm', '  norm ']], error_std))))))
            if printerror == True:
                print data, error_std
            error_norm = np.squeeze(error_norm, axis = 2)
        return error_norm


    def visualize_error_from_distance(self, bed_distance, error_norm):
        plt.close()
        fig = plt.figure()
        ax = []
        for joint in range(0, bed_distance.shape[1]):
            ax.append(joint)
            if bed_distance.shape[1] <= 5:
                ax[joint] = fig.add_subplot(1, bed_distance.shape[1], joint + 1)
            else:
                print math.ceil(bed_distance.shape[1]/2.)
                ax[joint] = fig.add_subplot(2, math.ceil(bed_distance.shape[1]/2.), joint + 1)
            ax[joint].set_xlim([0, 0.7])
            ax[joint].set_ylim([0, 1])
            ax[joint].set_title('Joint ' + str(joint) + ' error')
            ax[joint].plot(bed_distance[:, joint], error_norm[:, joint], 'r.')
        plt.show()


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
            ax2.plot(target_coord[:, 0], target_coord[:, 1], 'y*', ms=8)
        plt.pause(0.0001)

        # Visualize estimated from training set
        if scores_val is not None:
            if len(np.shape(scores_val)) == 1:
                scores_val = np.reshape(scores_val, (len(scores_val) / 3, 3))
            target_coord = scores_val[:, :2] / INTER_SENSOR_DISTANCE
            target_coord[:, 1] -= (NUMOFTAXELS_X - 1)
            target_coord[:, 1] *= -1.0
            ax2.plot(target_coord[:, 0], target_coord[:, 1], 'g*', ms=8)

        plt.show(block = block)
        return



    def rviz_publish_input(self, image, angle):
        mat_size = (NUMOFTAXELS_X, NUMOFTAXELS_Y)

        image = np.reshape(image, mat_size)

        markerArray = MarkerArray()
        for j in range(10, image.shape[0]-10):
            for i in range(10, image.shape[1]-10):
                imagePublisher = rospy.Publisher("/pressure_image", MarkerArray)

                marker = Marker()
                marker.header.frame_id = "autobed/base_link"
                marker.type = marker.SPHERE
                marker.action = marker.ADD
                marker.scale.x = 0.05
                marker.scale.y = 0.05
                marker.scale.z = 0.05
                marker.color.a = 1.0
                if image[j,i] > 60:
                    marker.color.r = 1.
                    marker.color.b = (100 - image[j, i])*.9 / 60.
                else:
                    marker.color.r = image[j, i] / 40.
                    marker.color.b = 1.
                marker.color.g = (70-np.abs(image[j,i]-50))/100.

                marker.pose.orientation.w = 1.0

                marker.pose.position.x = i*0.0286
                if j > 33:
                    marker.pose.position.y = (84-j)*0.0286 - 0.0286*3*np.sin(np.deg2rad(angle))
                    marker.pose.position.z = 0.#-0.1
                    #print marker.pose.position.x, 'x'
                else:

                    marker.pose.position.y = (51) * 0.0286 + (33 - j) * 0.0286 * np.cos(np.deg2rad(angle)) - 0.0286*3*np.sin(np.deg2rad(angle))
                    marker.pose.position.z = (33-j)*0.0286*np.sin(np.deg2rad(angle)) #-0.1
                    #print j, marker.pose.position.z, marker.pose.position.y, 'head'

                # We add the new marker to the MarkerArray, removing the oldest
                # marker from it when necessary
                #if (self.count > 100):
                 #   markerArray.markers.pop(0)

                markerArray.markers.append(marker)

                #print self.count

                # Renumber the marker IDs
                id = 0
                for m in markerArray.markers:
                    m.id = id
                    id += 1
        imagePublisher.publish(markerArray)


    def rviz_publish_output(self, targets, scores, pseudotargets = None):
        TargetArray = MarkerArray()
        for joint in range(0, targets.shape[0]):
            targetPublisher = rospy.Publisher("/targets", MarkerArray)
            Tmarker = Marker()
            Tmarker.header.frame_id = "autobed/base_link"
            Tmarker.type = Tmarker.SPHERE
            Tmarker.action = Tmarker.ADD
            Tmarker.scale.x = 0.05
            Tmarker.scale.y = 0.05
            Tmarker.scale.z = 0.05
            Tmarker.color.a = 1.0
            Tmarker.color.r = 1.0
            Tmarker.color.g = 1.0
            Tmarker.color.b = 0.0
            Tmarker.pose.orientation.w = 1.0
            Tmarker.pose.position.x = targets[joint, 0]
            Tmarker.pose.position.y = targets[joint, 1]
            Tmarker.pose.position.z = targets[joint, 2]
            TargetArray.markers.append(Tmarker)
            tid = 0
            for m in TargetArray.markers:
                m.id = tid
                tid += 1
        targetPublisher.publish(TargetArray)

        ScoresArray = MarkerArray()
        for joint in range(0, scores.shape[0]):
            scoresPublisher = rospy.Publisher("/scores", MarkerArray)
            Smarker = Marker()
            Smarker.header.frame_id = "autobed/base_link"
            Smarker.type = Smarker.SPHERE
            Smarker.action = Smarker.ADD
            Smarker.scale.x = 0.04
            Smarker.scale.y = 0.04
            Smarker.scale.z = 0.04
            Smarker.color.a = 1.0
            Smarker.color.r = 0.
            Smarker.color.g = 1.0
            Smarker.color.b = 1.0
            Smarker.pose.orientation.w = 1.0
            Smarker.pose.position.x = scores[joint, 0]
            Smarker.pose.position.y = scores[joint, 1]
            Smarker.pose.position.z = scores[joint, 2]
            ScoresArray.markers.append(Smarker)
            sid = 0
            for m in ScoresArray.markers:
                m.id = sid
                sid += 1
        scoresPublisher.publish(ScoresArray)

        if pseudotargets is not None:
            PTargetArray = MarkerArray()
            for joint in range(0, pseudotargets.shape[0]):
                ptargetPublisher = rospy.Publisher("/pseudotargets", MarkerArray)
                PTmarker = Marker()
                PTmarker.header.frame_id = "autobed/base_link"
                PTmarker.type = PTmarker.SPHERE
                PTmarker.action = PTmarker.ADD
                PTmarker.scale.x = 0.02
                PTmarker.scale.y = 0.02
                PTmarker.scale.z = 0.02
                PTmarker.color.a = 1.0
                PTmarker.color.r = 1.0
                PTmarker.color.g = 1.0
                PTmarker.color.b = 0.0
                PTmarker.pose.orientation.w = 1.0
                PTmarker.pose.position.x = pseudotargets[joint, 0]
                PTmarker.pose.position.y = pseudotargets[joint, 1]
                PTmarker.pose.position.z = pseudotargets[joint, 2]
                PTargetArray.markers.append(PTmarker)
                ptid = 0
                for m in PTargetArray.markers:
                    m.id = ptid
                    ptid += 1
            ptargetPublisher.publish(PTargetArray)


