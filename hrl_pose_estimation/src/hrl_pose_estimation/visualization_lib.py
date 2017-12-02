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

    def print_error(self, target, score, output_size, physical_constraints = None, data = None):

        if physical_constraints == 'torso_lengths':
            torso_error = (score[:, 0:3] - target[:, 0:3])
            torso_error = torso_error.data.numpy()
            torso_error_norm = np.expand_dims(np.linalg.norm(torso_error, axis = 1),1)
            torso_error = np.concatenate((torso_error, torso_error_norm), axis = 1)


            torso_error_avg = np.mean(torso_error, axis=0)
            torso_error_avg = np.reshape(torso_error_avg, (output_size[0], output_size[1]+1))
            torso_error_avg = np.reshape(np.array(["%.2f" % w for w in torso_error_avg.reshape(torso_error_avg.size)]),
                                         (output_size[0], output_size[1] + 1))
            torso_error_avg = np.transpose(
                np.concatenate(([['Average Torso Error for Last Batch', '       ', ' Torso ']], np.transpose(
                    np.concatenate(([['', '', '', ''], [' x, cm ', ' y, cm ', ' z, cm ', '  norm ']], torso_error_avg))))))

            lengths_error = (score[:, 3:11] - target[:, 3:11])
            lengths_error = lengths_error.data.numpy()
            lengths_error_avg = np.mean(lengths_error, axis = 0)
            lengths_error_avg = np.reshape(lengths_error_avg, (4,2))
            lengths_error_avg = np.reshape(np.array(["%.2f" % w for w in lengths_error_avg.reshape(lengths_error_avg.size)]),
                                           (4,2))
            lengths_error_avg = np.transpose(
                np.concatenate(([['Average Lengths Error for Last Batch', '                 ', '     Z + Vert    ','Spine to Shoulder', '   Upper Arm     ', '   Forearm       ']], np.transpose(
                    np.concatenate(([['', ''], [' R, cm ', ' L, cm ']], lengths_error_avg))))))
            print data, torso_error_avg
            print lengths_error_avg


            torso_error_std = np.std(torso_error, axis = 0)
            torso_error_std = np.reshape(torso_error_std, (output_size[0], output_size[1]+1))
            torso_error_std = np.reshape(np.array(["%.2f" % w for w in torso_error_std.reshape(torso_error_std.size)]),
                                         (output_size[0], output_size[1] + 1))
            torso_error_std = np.transpose(
                np.concatenate(([['Average Torso Standard Deviation for Last Batch', '       ', ' Torso ']], np.transpose(
                    np.concatenate(([['', '', '', ''], [' x, cm ', ' y, cm ', ' z, cm ', '  norm ']], torso_error_std))))))

            lengths_error_std = np.std(lengths_error, axis=0)
            lengths_error_std = np.reshape(lengths_error_std, (4, 2))
            lengths_error_std = np.reshape(
                np.array(["%.2f" % w for w in lengths_error_std.reshape(lengths_error_std.size)]),
                (4, 2))
            lengths_error_std = np.transpose(
                np.concatenate(([['Average Lengths Standard Deviation for Last Batch', '                 ', '     Z + Vert    ',
                                  'Spine to Shoulder', '   Upper Arm     ', '   Forearm       ']], np.transpose(
                    np.concatenate(([['', ''], [' R, cm ', ' L, cm ']], lengths_error_std))))))
            print data, torso_error_std
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

            if physical_constraints == 'arm_angles':
                error_avg = np.transpose(np.concatenate(([['Average Error for Last Batch', '       ', ' Torso ', 'R Elbow', 'L Elbow',
                                                           'R Hand ', 'L Hand ']], np.transpose(
                    np.concatenate(([['', '', '', ''], [' x, cm ', ' y, cm ', ' z, cm ', '  norm ']], error_avg))))))
            elif physical_constraints == 'all_joints':
                error_avg = np.transpose(np.concatenate(([['Average Error for Last Batch', '       ', 'Head   ',
                                                           'Torso  ', 'R Elbow', 'L Elbow', 'R Hand ', 'L Hand ',
                                                           'R Knee ', 'L Knee ', 'R Foot ', 'L Foot ']], np.transpose(
                    np.concatenate(([['', '', '', ''], [' x, cm ', ' y, cm ', ' z, cm ', '  norm ']], error_avg))))))
            else:
                error_avg = np.transpose(np.concatenate(([['Average Error for Last Batch', '       ', ' Torso ', 'R Elbow', 'L Elbow',
                                      'R Hand ', 'L Hand ']], np.transpose(np.concatenate(
                            ([['', '', '', ''], [' x, cm ', ' y, cm ', ' z, cm ', '  norm ']], error_avg))))))
            print data, error_avg


            error_std = np.std(error, axis=0) / 10
            error_std = np.reshape(error_std, (output_size[0], output_size[1] + 1))
            error_std = np.reshape(np.array(["%.2f" % w for w in error_std.reshape(error_std.size)]),
                                         (output_size[0], output_size[1] + 1))

            if physical_constraints == 'arm_angles':
                error_std = np.transpose(np.concatenate(([['Error Standard Deviation for Last Batch', '       ', ' Torso ', 'R Elbow',
                                                           'L Elbow', 'R Hand ', 'L Hand ']], np.transpose(
                    np.concatenate(([['', '', '',''], ['x, cm', 'y, cm', 'z, cm', '  norm ']], error_std))))))
            elif physical_constraints == 'all_joints':
                error_std = np.transpose(
                    np.concatenate(([['Error Standard Deviation for Last Batch', '       ', 'Head   ', 'Torso  ',
                                      'R Elbow', 'L Elbow', 'R Hand ', 'L Hand ', 'R Knee ', 'L Knee ',
                                      'R Foot ', 'L Foot ']], np.transpose(
                        np.concatenate(([['', '', ''], ['x, cm', 'y, cm', 'z, cm', '  norm ']], error_std))))))
            else:
                error_std = np.transpose(np.concatenate(([['Error Standard Deviation for Last Batch', '       ', ' Torso ', 'R Elbow',
                                                           'L Elbow', 'R Hand ', 'L Hand ']], np.transpose(
                    np.concatenate(([['', '', '',''], ['x, cm', 'y, cm', 'z, cm', '  norm ']], error_std))))))

            print data, error_std



    def rviz_publish_input(self, image, angle):
        mat_size = (NUMOFTAXELS_X, NUMOFTAXELS_Y)

        image = np.reshape(image, mat_size)
        print image.shape,'im shape'

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
                if j > 34:
                    marker.pose.position.y = (84-j)*0.0286 - 0.0286*3*np.sin(np.deg2rad(angle))
                    marker.pose.position.z = -0.1
                else:
                    marker.pose.position.y = (50) * 0.0286 + (34 - j) * 0.0286 * np.cos(np.deg2rad(angle)) - 0.0286*3*np.sin(np.deg2rad(angle))
                    marker.pose.position.z = -0.1 + (34-j)*0.0286*np.sin(np.deg2rad(angle))

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
        print targets
        print scores
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
            print joint, 'joint'
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


