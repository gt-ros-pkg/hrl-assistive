#!/usr/bin/env python
import sys
import os
import numpy as np
import cPickle as pkl
import random
import math

# ROS
#import roslib; roslib.load_manifest('hrl_pose_estimation')

# Graphics
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d import Axes3D

# Machine Learning
from scipy import ndimage
from scipy.ndimage.filters import gaussian_filter
from scipy import interpolate
from scipy.misc import imresize
from scipy.ndimage.interpolation import zoom
import scipy.stats as ss
## from skimage import data, color, exposure
from sklearn.decomposition import PCA

# HRL libraries
import hrl_lib.util as ut
import pickle
#roslib.load_manifest('hrl_lib')
from hrl_lib.util import load_pickle

# Pose Estimation Libraries
from create_dataset_lib import CreateDatasetLib

#PyTorch libraries
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable



MAT_WIDTH = 0.762 #metres
MAT_HEIGHT = 1.854 #metres
MAT_HALF_WIDTH = MAT_WIDTH/2
NUMOFTAXELS_X = 64#73 #taxels
NUMOFTAXELS_Y = 27#30
NUMOFOUTPUTDIMS = 3
NUMOFOUTPUTNODES = 10
INTER_SENSOR_DISTANCE = 0.0286#metres
LOW_TAXEL_THRESH_X = 0
LOW_TAXEL_THRESH_Y = 0
HIGH_TAXEL_THRESH_X = (NUMOFTAXELS_X - 1)
HIGH_TAXEL_THRESH_Y = (NUMOFTAXELS_Y - 1)


class DataVisualizer():
    '''Gets the directory of pkl database and iteratively go through each file,
    cutting up the pressure maps and creating synthetic database'''
    def __init__(self, pkl_directory):

        self.sitting =False
        self.subject = 1
        self.armsup = False
        # Set initial parameters
        self.dump_path = pkl_directory.rstrip('/')

        self.mat_size = (NUMOFTAXELS_X, NUMOFTAXELS_Y)
        self.output_size = (NUMOFOUTPUTNODES, NUMOFOUTPUTDIMS)

        train_val_loss = load_pickle(self.dump_path + '/train_val_losses.p')

        #print train_val_loss

        if self.subject == 1:
            #plt.plot(train_val_loss['epoch_flip_1'], train_val_loss['train_flip_1'],'b')
            #plt.plot(train_val_loss['epoch_1'], train_val_loss['train_1'], 'g')
            plt.plot(train_val_loss['epoch_1'], train_val_loss['val_1'], 'k')
            #plt.plot(train_val_loss['epoch_flip_1'], train_val_loss['val_flip_1'], 'r')
            #plt.plot(train_val_loss['epoch_flip_shift_1'], train_val_loss['train_flip_shift_1'], 'g')
            #plt.plot(train_val_loss['epoch_flip_shift_1'], train_val_loss['val_flip_shift_1'], 'g')
            #plt.plot(train_val_loss['epoch_flip_shift_nd_1'], train_val_loss['val_flip_shift_nd_1'], 'g')
            plt.plot(train_val_loss['epoch_flip_shift_nd_nohome_1'], train_val_loss['val_flip_shift_nd_nohome_1'], 'y')
            #plt.plot(train_val_loss['epoch_armsup_flip_shift_scale5_nd_nohome_1'], train_val_loss['train_armsup_flip_shift_scale5_nd_nohome_1'], 'b')
            #plt.plot(train_val_loss['epoch_armsup_flip_shift_scale5_nd_nohome_1'], train_val_loss['val_armsup_flip_shift_scale5_nd_nohome_1'], 'r')

        if self.subject == 4:
            #plt.plot(train_val_loss['epoch_flip_4'], train_val_loss['train_flip_4'], 'g')
            #plt.plot(train_val_loss['epoch_flip_4'], train_val_loss['val_flip_4'], 'y')
            #plt.plot(train_val_loss['epoch_4'], train_val_loss['train_4'], 'b')
            #plt.plot(train_val_loss['epoch_4'], train_val_loss['val_4'], 'r')
            #plt.plot(train_val_loss['epoch_flip_shift_nd_4'], train_val_loss['val_flip_shift_nd_4'], 'b')
            #plt.plot(train_val_loss['epoch_flip_shift_nd_nohome_4'], train_val_loss['val_flip_shift_nd_nohome_4'], 'y')

            plt.plot(train_val_loss['epoch_sitting_flip_shift_nd_700e4'], train_val_loss['val_sitting_flip_shift_nd_700e4'],'g')
            plt.plot(train_val_loss['epoch_sitting_flip_shift_scale5_nd_4'], train_val_loss['val_sitting_flip_shift_scale5_nd_4'],'y')

            #plt.plot(train_val_loss['epoch_flip_2'], train_val_loss['train_flip_2'], 'y')
            #plt.plot(train_val_loss['epoch_flip_2'], train_val_loss['val_flip_2'], 'g')
            #plt.plot(train_val_loss['epoch_flip_shift_nd_2'], train_val_loss['val_flip_shift_nd_2'], 'y')

        plt.axis([0,701,0,15000])
        plt.show()




        for key in train_val_loss:
            print key

    def validate_model(self, shorttrain = False):

        if self.sitting == True:
            validation_set = load_pickle(self.dump_path + '/subject_'+str(self.subject)+'/p_files/trainval_sitting_120rh_lh_rl_ll.p')
        elif self.armsup == True:
            validation_set = load_pickle(self.dump_path + '/subject_' + str(6) + '/p_files/trainval_200rh1_lh1_rl_ll_100rh23_lh23_head.p')
        else:
            validation_set = load_pickle(self.dump_path + '/subject_'+str(5)+'/p_files/trainval_200rh1_lh1_rl_ll.p')

        test_dat = validation_set

        self.test_x_flat = []  # Initialize the testing pressure mat list
        for entry in range(len(test_dat)):
            self.test_x_flat.append(test_dat[entry][0])
        test_x = self.preprocessing_pressure_array_resize(self.test_x_flat)
        test_x = np.array(test_x)
        #print test_x.shape
        test_x = self.pad_pressure_mats(test_x)


        self.test_x_tensor = torch.Tensor(test_x)
        print self.test_x_tensor.shape

        self.test_y_flat = []  # Initialize the ground truth list
        for entry in range(len(test_dat)):
            self.test_y_flat.append(test_dat[entry][1])
        self.test_y_tensor = torch.Tensor(self.test_y_flat)
        self.test_y_tensor = torch.mul(self.test_y_tensor, 1000)


        #print len(validation_set)
        batch_size = 1

        self.height_error = np.zeros((800,10,6))
        self.arm_angle_error = np.zeros((800,4,6))

        self.test_x_tensor = self.test_x_tensor.unsqueeze(1)
        self.test_dataset = torch.utils.data.TensorDataset(self.test_x_tensor, self.test_y_tensor)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size, shuffle=True)


        if self.sitting == True:
            model = torch.load(self.dump_path + '/subject_'+str(self.subject)+'/p_files/convnet_sitting_1to8_flip_shift_700e.pt')
        elif self.armsup == True:
            model = torch.load(self.dump_path + '/subject_' + str(self.subject) + '/p_files/convnet_1to8_flip_shift_scale5_armsup.pt')
        else:
            if shorttrain == True:
                model = torch.load(self.dump_path + '/subject_'+str(self.subject)+'/p_files/convnet_1to8_flip_shift_nodrop_nohome.pt')
            else:
                model = torch.load(self.dump_path + '/subject_' + str(self.subject) + '/p_files/convnet_1to8_flip_shift_nodrop_nohome_1000e.pt')

        count = 0
        for batch_idx, batch in enumerate(self.test_loader):
            count += 1
            #print count
            # prepare data
            #batch[0], batch[1] = self.synthesizetest(batch[0],batch[1])


            images, targets = Variable(batch[0]), Variable(batch[1])

            #print targets.size()

            scores = model(images)

            #print scores.size()
            self.print_error(targets,scores)

            self.im_sample = batch[0].numpy()
            self.im_sample = np.squeeze(self.im_sample[0, :])
            self.tar_sample = batch[1].numpy()
            self.tar_sample = np.squeeze(self.tar_sample[0, :]) / 1000
            self.sc_sample = scores.data.numpy()
            self.sc_sample = np.squeeze(self.sc_sample[0, :]) / 1000
            self.sc_sample = np.reshape(self.sc_sample, self.output_size)

            self.tar_sample = np.reshape(self.tar_sample, (len(self.tar_sample) / 3, 3))


            #log the estimated height of the right elbow
            for joint in xrange(10):
                self.height_error[count-1,joint,0]=self.tar_sample[joint, 2]*100
                self.height_error[count-1,joint,1]=self.sc_sample[joint,2]*100
                self.height_error[count-1,joint,2]=(self.sc_sample[joint, 0] - self.tar_sample[joint, 0])*100
                self.height_error[count-1,joint,3]=(self.sc_sample[joint, 1] - self.tar_sample[joint, 1])*100
                self.height_error[count-1,joint,4]=(self.sc_sample[joint, 2] - self.tar_sample[joint, 2])*100
                self.height_error[count-1,joint,5]=(np.sqrt(np.square(self.sc_sample[joint, 0] - self.tar_sample[joint, 0])+np.square(self.sc_sample[joint, 1] - self.tar_sample[joint, 1])+np.square(self.sc_sample[joint, 2] - self.tar_sample[joint, 2])))*100

            for joint in xrange(4):
                self.arm_angle_error[count-1,joint,0] = self.tar_sample[joint+2,1]*100
                self.arm_angle_error[count-1,joint,1] = self.sc_sample[joint+2,1]*100
                self.arm_angle_error[count - 1, joint, 2] = (self.sc_sample[joint+2, 0] - self.tar_sample[joint+2, 0]) * 100
                self.arm_angle_error[count - 1, joint, 3] = (self.sc_sample[joint+2, 1] - self.tar_sample[joint+2, 1]) * 100
                self.arm_angle_error[count - 1, joint, 4] = (self.sc_sample[joint+2, 2] - self.tar_sample[joint+2, 2]) * 100
                self.arm_angle_error[count - 1, joint, 5] = (np.sqrt(np.square(self.sc_sample[joint+2, 0] - self.tar_sample[joint+2, 0]) + np.square(self.sc_sample[joint+2, 1] - self.tar_sample[joint+2, 1]) + np.square(self.sc_sample[joint+2, 2] - self.tar_sample[joint+2, 2]))) * 100

            #self.visualize_pressure_map(self.im_sample, self.tar_sample, self.sc_sample)

        #self.height_error_plotter()
        #self.arm_angle_error_plotter()

        return self.arm_angle_error




    def height_error_plotter(self):

        fig = plt.figure()

        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(self.height_error[:,2,0], np.abs(self.height_error[:,2,2]), 'k.')
        ax1.plot(self.height_error[:,2,1], np.abs(self.height_error[:,2,2]), 'r.')
        ax1.set_title('x error as a function of height.')


        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(self.height_error[:, 2, 0], np.abs(self.height_error[:, 2, 3]), 'k.')
        ax2.plot(self.height_error[:, 2, 1], np.abs(self.height_error[:, 2, 3]), 'r.')
        ax2.set_title('y error as a function of height.')

        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(self.height_error[:, 2, 0], np.abs(self.height_error[:, 2, 4]), 'k.')
        ax3.plot(self.height_error[:, 2, 1], np.abs(self.height_error[:, 2, 4]), 'r.')
        ax3.set_title('z error as a function of height.')

        ax4 = fig.add_subplot(2, 2, 4)
        ax4.plot(self.height_error[:, 2, 0], np.abs(self.height_error[:, 2, 5]), 'k.')
        ax4.plot(self.height_error[:, 2, 1], np.abs(self.height_error[:, 2, 5]), 'r.')
        ax4.set_title('absolute error as a function of height.')
        plt.suptitle('Error as a function of the z-distance (height above bed)')
        plt.show()


        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        #ax1.plot(self.height_error[:,0,0], np.abs(self.height_error[:,0,5]), 'k.')
        ax1.plot(self.height_error[:,0,1], np.abs(self.height_error[:,0,5]), 'k.')
        ax1.set_title('Head')

        ax2 = fig.add_subplot(1, 2, 2)
        #ax2.plot(self.height_error[:, 1, 0], np.abs(self.height_error[:, 1, 5]), 'k.')
        ax2.plot(self.height_error[:, 1, 1], np.abs(self.height_error[:, 1, 5]), 'k.')
        ax2.set_title('Torso')
        plt.suptitle('Error as a function of the z-distance (height above bed)')
        plt.show()


        fig = plt.figure()
        ax1 = fig.add_subplot(2, 2, 1)
        #ax1.plot(self.height_error[:,3,0], np.abs(self.height_error[:,3,5]), 'k.')
        ax1.plot(self.height_error[:,3,1], np.abs(self.height_error[:,3,5]), 'k.')
        ax1.set_title('Left Elbow')
        ax1.set_xlim([-3,15])
        ax1.set_ylim([0, 40])

        ax2 = fig.add_subplot(2, 2, 2)
        #ax2.plot(self.height_error[:, 2, 0], np.abs(self.height_error[:, 2, 5]), 'k.')
        ax2.plot(self.height_error[:, 2, 1], np.abs(self.height_error[:, 2, 5]), 'k.')
        ax2.set_title('Right Elbow')
        ax2.set_xlim([-3,15])
        ax2.set_ylim([0, 40])

        ax3 = fig.add_subplot(2, 2, 3)
        #ax3.plot(self.height_error[:, 5, 0], np.abs(self.height_error[:, 5, 5]), 'k.')
        ax3.plot(self.height_error[:, 5, 1], np.abs(self.height_error[:, 5, 5]), 'k.')
        ax3.set_title('Left Hand')
        ax3.set_xlim([-2,20])
        ax3.set_ylim([0, 60])

        ax4 = fig.add_subplot(2, 2, 4)
        #ax4.plot(self.height_error[:, 4, 0], np.abs(self.height_error[:, 4, 5]), 'k.')
        ax4.plot(self.height_error[:, 4, 1], np.abs(self.height_error[:, 4, 5]), 'k.')
        ax4.set_title('Right Hand')
        ax4.set_xlim([-2,20])
        ax4.set_ylim([0, 60])
        plt.suptitle('Error as a function of the z-distance (height above bed)')
        plt.show()



        fig = plt.figure()
        ax1 = fig.add_subplot(2, 2, 1)
        #ax1.plot(self.height_error[:, 7, 0], np.abs(self.height_error[:, 7, 5]), 'k.')
        ax1.plot(self.height_error[:, 7, 1], np.abs(self.height_error[:, 7, 5]), 'k.')
        ax1.set_title('Left Knee')
        ax1.set_xlim([0,40])
        ax1.set_ylim([0, 40])

        ax2 = fig.add_subplot(2, 2, 2)
        #ax2.plot(self.height_error[:, 6, 0], np.abs(self.height_error[:, 6, 5]), 'k.')
        ax2.plot(self.height_error[:, 6, 1], np.abs(self.height_error[:, 6, 5]), 'k.')
        ax2.set_title('Right Knee')
        ax2.set_xlim([0,40])
        ax2.set_ylim([0, 40])

        ax3 = fig.add_subplot(2, 2, 3)
        #ax3.plot(self.height_error[:, 9, 0], np.abs(self.height_error[:, 9, 5]), 'k.')
        ax3.plot(self.height_error[:, 9, 1], np.abs(self.height_error[:, 9, 5]), 'k.')
        ax3.set_title('Left Foot')
        ax3.set_xlim([-3,30])
        ax3.set_ylim([0, 50])

        ax4 = fig.add_subplot(2, 2, 4)
        #ax4.plot(self.height_error[:, 8, 0], np.abs(self.height_error[:, 8, 5]), 'k.')
        ax4.plot(self.height_error[:, 8, 1], np.abs(self.height_error[:, 8, 5]), 'k.')
        ax4.set_title('Right Foot')
        ax4.set_xlim([-3, 30])
        ax4.set_ylim([0, 50])
        plt.suptitle('Error as a function of the z-distance (height above bed)')
        plt.show()


    def arm_angle_error_plotter(self,arm_angle_error1, arm_angle_error2 = None):
        fig = plt.figure()
        ax1 = fig.add_subplot(2, 2, 1)
        if arm_angle_error2 is not None:
            ax1.plot(arm_angle_error2[:, 1, 1], np.abs(arm_angle_error2[:,1,5]), 'r.')
        ax1.plot(arm_angle_error1[:, 1, 1], np.abs(arm_angle_error1[:, 1, 5]), 'k.')
        ax1.set_title('Left Elbow')
        ax1.set_xlim([100, 190])
        ax1.set_ylim([0, 40])

        ax2 = fig.add_subplot(2, 2, 2)
        if arm_angle_error2 is not None:
            ax2.plot(arm_angle_error2[:, 0, 1], np.abs(arm_angle_error2[:, 0, 5]), 'r.')
        ax2.plot(arm_angle_error1[:, 0, 1], np.abs(arm_angle_error1[:, 0, 5]), 'k.')
        ax2.set_title('Right Elbow')
        ax2.set_xlim([100, 190])
        ax2.set_ylim([0, 40])

        ax3 = fig.add_subplot(2, 2, 3)
        if arm_angle_error2 is not None:
            ax3.plot(arm_angle_error2[:, 3, 1], np.abs(arm_angle_error2[:, 3, 5]), 'r.')
        ax3.plot(arm_angle_error1[:, 3, 1], np.abs(arm_angle_error1[:, 3, 5]), 'k.')
        ax3.set_title('Left Hand')
        ax3.set_xlim([60, 210])
        ax3.set_ylim([0, 60])

        ax4 = fig.add_subplot(2, 2, 4)
        if arm_angle_error2 is not None:
            ax4.plot(arm_angle_error1[:, 2, 1], np.abs(arm_angle_error1[:, 2, 5]), 'r.')
        ax4.plot(arm_angle_error1[:, 2, 1], np.abs(arm_angle_error1[:, 2, 5]), 'k.')
        ax4.set_title('Right Hand')
        ax4.set_xlim([60, 210])
        ax4.set_ylim([0, 60])
        plt.suptitle('Error as a function of the y-distance (toe to head) of arm joints')
        plt.show()

    def pad_pressure_mats(self,NxHxWimages):
        padded = np.zeros((NxHxWimages.shape[0],NxHxWimages.shape[1]+20,NxHxWimages.shape[2]+20))
        padded[:,10:74,10:37] = NxHxWimages
        NxHxWimages = padded
        return NxHxWimages

    def preprocessing_pressure_array_resize(self, data):
        '''Will resize all elements of the dataset into the dimensions of the
        pressure map'''
        p_map_dataset = []
        for map_index in range(len(data)):
            #print map_index, self.mat_size, 'mapidx'
            #Resize mat to make into a matrix
            p_map = np.reshape(data[map_index], self.mat_size)
            p_map_dataset.append(p_map)
        print len(data[0]),'x',1, 'size of an incoming pressure map'
        print len(p_map_dataset[0]),'x',len(p_map_dataset[0][0]), 'size of a resized pressure map'
        return p_map_dataset

    def synthesizetest(self, image_tensor,target_tensor):
        return image_tensor, target_tensor



    def print_error(self, target, score, data = None):
        error = (score - target)
        error = error.data.numpy()
        error_avg = np.mean(np.abs(error), axis=0) / 10
        error_avg = np.reshape(error_avg, self.output_size)
        print np.sum(np.square(error_avg)), np.sum(np.abs(error_avg))
        error_avg = np.reshape(np.array(["%.2f" % w for w in error_avg.reshape(error_avg.size)]),
                               self.output_size)
        error_avg = np.transpose(np.concatenate(([['Average Error for Last Batch', '       ', 'Head   ',
                                                   'Torso  ', 'R Elbow', 'L Elbow', 'R Hand ', 'L Hand ',
                                                   'R Knee ', 'L Knee ', 'R Foot ', 'L Foot ']], np.transpose(
            np.concatenate(([['', '', ''], [' x, cm ', ' y, cm ', ' z, cm ']], error_avg))))))
        print data, error_avg

        error_std = np.std(error, axis=0) / 10
        error_std = np.reshape(error_std, self.output_size)
        print np.sum(np.abs(error_std))
        error_std = np.reshape(np.array(["%.2f" % w for w in error_std.reshape(error_std.size)]),
                               self.output_size)
        error_std = np.transpose(
            np.concatenate(([['Error Standard Deviation for Last Batch', '       ', 'Head   ', 'Torso  ',
                              'R Elbow', 'L Elbow', 'R Hand ', 'L Hand ', 'R Knee ', 'L Knee ',
                              'R Foot ', 'L Foot ']], np.transpose(
                np.concatenate(([['', '', ''], ['x, cm', 'y, cm', 'z, cm']], error_std))))))
        print data, error_std


    def visualize_pressure_map(self, p_map, targets_raw=None, scores_raw = None, p_map_val = None, targets_val = None, scores_val = None):
        print p_map.shape, 'pressure mat size', targets_raw.shape, 'target shape'
        #p_map = fliplr(p_map)

        plt.close()
        plt.pause(0.0001)

        fig = plt.figure()
        mngr = plt.get_current_fig_manager()
        # to put it into the upper left corner for example:
        mngr.window.setGeometry(50, 100, 840, 705)

        plt.pause(0.0001)

        # set options
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)


        xlim = [-2.0, 49.0]
        ylim = [86.0, -2.0]
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim)
        ax2.set_xlim(xlim)
        ax2.set_ylim(ylim)

        # background
        ax1.set_axis_bgcolor('cyan')
        ax2.set_axis_bgcolor('cyan')

        # Visualize pressure maps
        ax1.imshow(p_map, interpolation='nearest', cmap=
        plt.cm.bwr, origin='upper', vmin=0, vmax=100)

        if p_map_val is not None:
            ax2.imshow(p_map_val, interpolation='nearest', cmap=
            plt.cm.bwr, origin='upper', vmin=0, vmax=100)

        # Visualize targets of training set
        if targets_raw is not None:

            if len(np.shape(targets_raw)) == 1:
                targets_raw = np.reshape(targets_raw, (len(targets_raw) / 3, 3))

            #targets_raw[:, 0] = ((targets_raw[:, 0] - 0.3718) * -1) + 0.3718
            #print targets_raw
            #extra_point = np.array([[0.,0.3718,0.7436],[0.,0.,0.]])
            #extra_point = extra_point/INTER_SENSOR_DISTANCE
            #ax1.plot(extra_point[0,:],extra_point[1,:], 'r*', ms=8)

            target_coord = targets_raw[:, :2] / INTER_SENSOR_DISTANCE
            target_coord[:, 1] -= (NUMOFTAXELS_X - 1)
            target_coord[:, 1] *= -1.0
            ax1.plot(target_coord[:, 0]+10, target_coord[:, 1]+10, 'y*', ms=8)

        plt.pause(0.0001)

        #Visualize estimated from training set
        if scores_raw is not None:
            if len(np.shape(scores_raw)) == 1:
                scores_raw = np.reshape(scores_raw, (len(scores_raw) / 3, 3))
            target_coord = scores_raw[:, :2] / INTER_SENSOR_DISTANCE
            target_coord[:, 1] -= (NUMOFTAXELS_X - 1)
            target_coord[:, 1] *= -1.0
            ax1.plot(target_coord[:, 0]+10, target_coord[:, 1]+10, 'g*', ms=8)
        ax1.set_title('Training Sample \n Targets and Estimates')
        plt.pause(0.0001)

        # Visualize targets of validation set
        if targets_val is not None:
            if len(np.shape(targets_val)) == 1:
                targets_val = np.reshape(targets_val, (len(targets_val) / 3, 3))
            target_coord = targets_val[:, :2] / INTER_SENSOR_DISTANCE
            target_coord[:, 1] -= (NUMOFTAXELS_X - 1)
            target_coord[:, 1] *= -1.0
            ax2.plot(target_coord[:, 0]+10, target_coord[:, 1]+10, 'y*', ms=8)
        plt.pause(0.0001)

        # Visualize estimated from training set
        if scores_val is not None:
            if len(np.shape(scores_val)) == 1:
                scores_val = np.reshape(scores_val, (len(scores_val) / 3, 3))
            target_coord = scores_val[:, :2] / INTER_SENSOR_DISTANCE
            target_coord[:, 1] -= (NUMOFTAXELS_X - 1)
            target_coord[:, 1] *= -1.0
            ax2.plot(target_coord[:, 0]+10, target_coord[:, 1]+10, 'g*', ms=8)
        ax2.set_title('Validation Sample \n Targets and Estimates')
        plt.pause(0.5)


        #targets_raw_z = []
        #for idx in targets_raw: targets_raw_z.append(idx[2])
        #x = np.arange(0,10)
        #ax3.bar(x, targets_raw_z)
        #plt.xticks(x+0.5, ('Head', 'Torso', 'R Elbow', 'L Elbow', 'R Hand', 'L Hand', 'R Knee', 'L Knee', 'R Foot', 'L Foot'), rotation='vertical')
        #plt.title('Distance above Bed')
        #plt.pause(0.0001)

        plt.show()
        #plt.show(block = False)

        return




    def run(self):
        '''Runs either the synthetic database creation script or the
        raw dataset creation script to create a dataset'''
        arm_angle_err1 = self.validate_model()
        arm_angle_err2 = self.validate_model(shorttrain=True)
        self.arm_angle_error_plotter(arm_angle_err1,arm_angle_err2)
        return





if __name__ == "__main__":

    import optparse
    p = optparse.OptionParser()

    p.add_option('--training_data_path', '--path',  action='store', type='string', \
                 dest='trainingPath',\
                 default='/home/henryclever/hrl_file_server/Autobed/pose_estimation_data/subject_', \
                 help='Set path to the training database.')



    Path =  '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/'
    #Path = '/home/henryclever/hrl_file_server/Autobed/'

    print Path

    #Initialize trainer with a training database file
    p = DataVisualizer(pkl_directory=Path)
    p.run()
    sys.exit()
