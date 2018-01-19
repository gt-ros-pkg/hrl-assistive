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
from sklearn.preprocessing import normalize

#ROS libs
import rospkg
import roslib
import rospy
import tf.transformations as tft
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray


# HRL libraries
import hrl_lib.util as ut
import pickle
#roslib.load_manifest('hrl_lib')
from hrl_lib.util import load_pickle

# Pose Estimation Libraries
from create_dataset_lib import CreateDatasetLib
from visualization_lib import VisualizationLib
from kinematics_lib import KinematicsLib
from preprocessing_lib import PreprocessingLib
from cascade_lib import CascadeLib

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
NUMOFTAXELS_X = 84#73 #taxels
NUMOFTAXELS_Y = 47#30
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
    def __init__(self, pkl_directory, pathkey, opt):
        self.opt = opt
        self.sitting = False
        self.subject = 4
        self.armsup = False
        self.alldata = False
        self.verbose = True
        self.old = False
        self.normalize = True
        self.include_inter = True
        self.loss_vector_type = 'angles'
        # Set initial parameters
        self.dump_path = pkl_directory.rstrip('/')

        self.mat_size = (NUMOFTAXELS_X, NUMOFTAXELS_Y)


        self.mat_size = (NUMOFTAXELS_X, NUMOFTAXELS_Y)
        if self.loss_vector_type == 'upper_angles':
            self.output_size = (NUMOFOUTPUTNODES - 4, NUMOFOUTPUTDIMS)
        elif self.loss_vector_type == 'arms_cascade':
            self.output_size = (NUMOFOUTPUTNODES - 8, NUMOFOUTPUTDIMS) #because of symmetry, we can train on just one side via synthetic flipping
            self.val_output_size = (NUMOFOUTPUTNODES - 6, NUMOFOUTPUTDIMS) #however, we still want to make a validation double forward pass through both sides
        elif self.loss_vector_type == 'angles' or self.loss_vector_type == 'direct':
            self.output_size = (NUMOFOUTPUTNODES, NUMOFOUTPUTDIMS)
        elif self.loss_vector_type == None:
            self.output_size = (NUMOFOUTPUTNODES - 5, NUMOFOUTPUTDIMS)


        if pathkey == 'lab_hd':
            train_val_loss = load_pickle(self.dump_path + '/train_val_losses.p')
            train_val_loss_desk = load_pickle(self.dump_path + '/train_val_losses_hcdesktop.p')
            train_val_loss_all = load_pickle(self.dump_path + '/train_val_losses_all.p')
            train_val_loss_171106 = load_pickle(self.dump_path + '/train_val_losses_171106.p')
            train_val_loss_171202 = load_pickle(self.dump_path + '/train_val_losses_171202.p')
            for key in train_val_loss:
                print key
            print '###########################  done with laptop #################'
            for key in train_val_loss_desk:
                print key
            print '###########################  done with desktop ################'
            for key in train_val_loss_171106:
                print key
            print '###########################  done with 171106 ################'
            for key in train_val_loss_all:
                print key
            print '###########################  done with mixed sitting/laying ################'



        elif pathkey == 'hc_desktop':
            train_val_loss_all = load_pickle(self.dump_path + '/train_val_losses_all.p')
            for key in train_val_loss_all:
                print key
            print '###########################  done with desktop ################'

        #handles, labels = plt.get_legend_handles_labels()
        #plt.legend(handles, labels)
        #print train_val_loss



        if self.subject == 1:

            plt.plot(train_val_loss_desk['epoch_armsup_700e_1'], train_val_loss_desk['val_armsup_700e_1'], 'k',label='Raw Pressure Map Input')
            #plt.plot(train_val_loss['epoch_sitting_flip_700e_4'], train_val_loss['val_sitting_flip_700e_4'], 'c',label='Synthetic Flipping: $Pr(X=flip)=0.5$')
            #plt.plot(train_val_loss_desk['epoch_armsup_flip_shift_scale10_700e_1'],train_val_loss_desk['val_armsup_flip_shift_scale10_700e_1'], 'g',label='Synthetic Flipping+Shifting: $X,Y \sim N(\mu,\sigma), \mu = 0 cm, \sigma \~= 9 cm$')
            plt.plot(train_val_loss_desk['epoch_armsup_flip_shift_scale5_nd_nohome_700e_1'],train_val_loss_desk['val_armsup_flip_shift_scale5_nd_nohome_700e_1'], 'y', label='Synthetic Flipping+Shifting+Scaling: $S_C \sim N(\mu,\sigma), \mu = 1, \sigma \~= 1.02$')
            plt.legend()
            plt.ylabel('Mean squared error loss over 30 joint vectors')
            plt.xlabel('Epochs, where 700 epochs ~ 4 hours')
            plt.title('Subject 1 laying validation Loss, training performed on subjects 2, 3, 4, 5, 6, 7, 8')


        elif self.subject == 4:
            if pathkey == 'lab_hd': #results presented to hrl dressing 171106


                #plt.plot(train_val_loss_all['epoch_2to8_alldata_angles_115b_adam_200e_44'], train_val_loss_all['train_2to8_alldata_angles_115b_adam_200e_44'], 'k')
                plt.plot(train_val_loss_all['epoch_2to8_alldata_angles_115b_adam_200e_44'], train_val_loss_all['val_2to8_alldata_angles_115b_adam_200e_44'], 'y')
                #plt.plot(train_val_loss_all['epoch_2to8_alldata_angles_constrained_noise_115b_100e_44'], train_val_loss_all['train_2to8_alldata_angles_constrained_noise_115b_100e_44'], 'b')
                plt.plot(train_val_loss_all['epoch_2to8_alldata_angles_constrained_noise_115b_100e_44'], train_val_loss_all['val_2to8_alldata_angles_constrained_noise_115b_100e_44'], 'r')
                #plt.plot(train_val_loss_all['epoch_2to8_alldata_angles_s10to18_115b_50e_44'], train_val_loss_all['val_2to8_alldata_angles_s10to18_115b_50e_44'], 'g')
                plt.plot(train_val_loss_all['epoch_2to8_alldata_angles_implbedang_115b_100e_44'], train_val_loss_all['val_2to8_alldata_angles_implbedang_115b_100e_44'], 'g')



                if False:
                    plt.plot(train_val_loss['epoch_sitting_700e_4'],train_val_loss['val_sitting_700e_4'],'k', label='Raw Pressure Map Input')
                    plt.plot(train_val_loss['epoch_sitting_flip_700e_4'], train_val_loss['val_sitting_flip_700e_4'], 'c', label='Synthetic Flipping: $Pr(X=flip)=0.5$')
                    plt.plot(train_val_loss['epoch_sitting_flip_shift_nd_700e4'],train_val_loss['val_sitting_flip_shift_nd_700e4'], 'g', label='Synthetic Flipping+Shifting: $X,Y \sim N(\mu,\sigma), \mu = 0 cm, \sigma \~= 9 cm$')
                    plt.plot(train_val_loss['epoch_sitting_flip_shift_scale10_700e_4'],train_val_loss['val_sitting_flip_shift_scale10_700e_4'], 'y', label='Synthetic Flipping+Shifting+Scaling: $S_C \sim N(\mu,\sigma), \mu = 1, \sigma \~= 1.02$')
                    plt.plot(train_val_loss['epoch_alldata_flip_shift_scale5_nd_nohome_500e_4'],train_val_loss['val_alldata_flip_shift_scale5_nd_nohome_500e_4'], 'r',label='Standing+Sitting: Synthetic Flipping+Shifting+Scaling: $S_C \sim N(\mu,\sigma), \mu = 1, \sigma \~= 1.02$')
                #plt.plot(train_val_loss['epoch_sitting_flip_shift_scale5_700e_4'],train_val_loss['val_sitting_flip_shift_scale_700e_4'], 'y',label='Synthetic Flipping+Shifting+Scaling: $S_C \sim N(\mu,\sigma), \mu = 1, \sigma \~= 1.02$')

                    plt.plot(train_val_loss_171106['epoch_sitting_flip_shift_scale5_b50_700e_4'],train_val_loss_171106['train_sitting_flip_shift_scale5_b50_700e_4'], 'b', label='Synthetic Flipping+Shifting+Scaling: $S_C \sim N(\mu,\sigma), \mu = 1, \sigma \~= 1.02$')
                    plt.legend()
                    plt.ylabel('Mean squared error loss over 30 joint vectors')
                    plt.xlabel('Epochs, where 700 epochs ~ 4 hours')
                    plt.title('Subject 4 sitting validation Loss, training performed on subjects 2, 3, 5, 6, 7, 8')



        #plt.axis([0,410,0,30000])
        plt.axis([0, 200, 0, 20])
        plt.show()

        self.count = 0

        rospy.init_node('plot_loss')



        self.validation_set = load_pickle(self.dump_path + '/subject_' + str(4) + '/p_files/trainval_150rh1_lh1_rl_ll_100rh23_lh23_sit120rh_lh_rl_ll.p')

        test_dat = self.validation_set
        for key in test_dat:
            print key, np.array(test_dat[key]).shape


        self.test_x_flat = []  # Initialize the testing pressure mat list
        for entry in range(len(test_dat['images'])):
            self.test_x_flat.append(test_dat['images'][entry])
        #test_x = PreprocessingLib().preprocessing_pressure_array_resize(self.test_x_flat, self.mat_size, self.verbose)
        #test_x = np.array(test_x)

        self.test_a_flat = []  # Initialize the testing pressure mat angle list
        for entry in range(len(test_dat['images'])):
            self.test_a_flat.append(test_dat['bed_angle_deg'][entry])
        test_xa = PreprocessingLib().preprocessing_create_pressure_angle_stack(self.test_x_flat, self.test_a_flat, self.include_inter, self.mat_size, self.verbose)
        test_xa = np.array(test_xa)
        self.test_x_tensor = torch.Tensor(test_xa)

        print 'Finished converting inputs to a torch tensor'

        self.test_y_flat = []  # Initialize the ground truth list
        for entry in range(len(test_dat['images'])):
            if self.loss_vector_type == 'angles':
                c = np.concatenate((test_dat['markers_xyz_m'][entry][0:30] * 1000,
                                    test_dat['joint_lengths_U_m'][entry][0:9] * 100,
                                    test_dat['joint_angles_U_deg'][entry][0:10],
                                    test_dat['joint_lengths_L_m'][entry][0:8] * 100,
                                    test_dat['joint_angles_L_deg'][entry][0:8]), axis=0)
                self.test_y_flat.append(c)
            elif self.loss_vector_type == 'arms_cascade':
                c = np.concatenate((test_dat['markers_xyz_m'][entry][0:30] * 1000,
                                    test_dat['joint_lengths_U_m'][entry][0:9] * 100,
                                    test_dat['joint_angles_U_deg'][entry][0:10],
                                    test_dat['joint_lengths_L_m'][entry][0:8] * 100,
                                    test_dat['joint_angles_L_deg'][entry][0:8],
                                    test_dat['pseudomarkers_xyz_m'][entry][:] * 1000), axis=0)
                self.test_y_flat.append(c)
            elif self.loss_vector_type == 'upper_angles':
                c = np.concatenate((test_dat['markers_xyz_m'][entry][0:18] * 1000,
                                    test_dat['joint_lengths_U_m'][entry][0:9] * 100,
                                    test_dat['joint_angles_U_deg'][entry][0:10]), axis=0)
                self.test_y_flat.append(c)
            else:
                self.test_y_flat.append(test_dat['markers_xyz_m'][entry] * 1000)
        self.test_y_tensor = torch.Tensor(self.test_y_flat)

        print 'Finished converting outputs to a torch tensor'



    def validate_baseline(self):

        print len(self.validation_set), 'size of validation set'
        batch_size = 1

        self.test_dataset = torch.utils.data.TensorDataset(self.test_x_tensor, self.test_y_tensor)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size, shuffle=True)
        regr = load_pickle(self.dump_path + '/subject_' + str(4) + '/p_files/HoG_Linear.p')


        count = 0
        for batch_idx, batch in enumerate(self.test_loader):
            images = batch[0].numpy()[:, 0, :, :]
            targets = batch[1].numpy()

            # upsample the images
            images_up = PreprocessingLib().preprocessing_pressure_map_upsample(images)
            # targets = list(targets)
            print images_up[0].shape

            # Compute HoG of the current(training) pressure map dataset
            images_up = PreprocessingLib().compute_HoG(images_up)
            scores = regr.predict(images_up)

            VisualizationLib().print_error(scores, targets, self.output_size, loss_vector_type=self.loss_vector_type,
                                           data='test', printerror=True)

            self.im_sample = np.squeeze(images[0, :])
            print self.im_sample.shape

            self.tar_sample = np.squeeze(targets[0, :]) / 1000
            self.sc_sample = np.copy(scores)
            self.sc_sample = np.squeeze(self.sc_sample[0, :]) / 1000
            self.sc_sample = np.reshape(self.sc_sample, self.output_size)
            VisualizationLib().visualize_pressure_map(self.im_sample, self.tar_sample, self.sc_sample, block=True)

    def validate_convnet(self):


        print len(self.validation_set), 'size of validation set'
        batch_size = 1

        self.test_dataset = torch.utils.data.TensorDataset(self.test_x_tensor, self.test_y_tensor)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size, shuffle=True)



        #torso_length_model = torch.load(self.dump_path + '/subject_' + str(self.subject) + '/p_files/convnet_2to8_alldata_armsonly_torso_lengths_115b_adam_100e_4.pt')
        #angle_model = torch.load(self.dump_path + '/subject_' + str(self.subject) + '/p_files/convnet_2to8_alldata_armsonly_upper_angles_115b_adam_200e_4.pt')

        if self.loss_vector_type == 'angles':
            model = torch.load(self.dump_path + '/subject_' + str(4) + '/p_files/convnet_2to8_alldata_angles_implbedang_115b_100e_4.pt')
        elif self.loss_vector_type == 'arms_cascade':
            model_cascade_prior = torch.load(self.dump_path + '/subject_' + str(4) + '/p_files/convnet_2to8_alldata_angles_constrained_noise_115b_100e_4.pt')
            model = torch.load(self.dump_path + '/subject_' + str(4) + '/p_files/convnet_2to8_alldata_armanglescascade_constrained_noise_115b_100e_4.pt')

        count = 0
        x2, y2, x3, y3, x4, y4, x5, y5 = [], [], [], [], [], [], [], []


        for batch_idx, batch in enumerate(self.test_loader):
            count += 1
            #print count

            if self.loss_vector_type == 'angles':

                # append upper joint angles, lower joint angles, upper joint lengths, lower joint lengths, in that order
                batch.append(torch.cat((batch[1][:, 39:49], batch[1][:, 57:65], batch[1][:, 30:39], batch[1][:, 49:57]), dim=1))

                # get the direct joint locations
                batch[1] = batch[1][:, 0:30]

                count2 = 0
                cum_error = []
                cum_distance = []
                while count2 < 1:
                    count2 += 1

                    images_up = batch[0].numpy()
                    images_up = images_up[:, :, 5:79, 5:42]
                    images_up = PreprocessingLib().preprocessing_pressure_map_upsample(images_up)
                    images_up = np.array(images_up)
                    #images_up = PreprocessingLib().preprocessing_add_image_noise(images_up)
                    images_up = Variable(torch.Tensor(images_up), volatile = True, requires_grad=False)
                    #print images_up.size()

                    images, targets, constraints = Variable(batch[0], volatile = True, requires_grad=False), Variable(batch[1], volatile = True, requires_grad=False), Variable(batch[2], volatile = True, requires_grad=False)


                    _, targets_est, angles_est, lengths_est, pseudotargets_est = model.forward_kinematic_jacobian(images_up, targets, constraints, forward_only = True)

                    bed_distances = KinematicsLib().get_bed_distance(images, targets)


                    targets = targets.data.numpy()

                    error_norm = VisualizationLib().print_error(targets, targets_est, self.output_size, self.loss_vector_type, data=str(count), printerror =  True)/1000
                    print angles_est
                    cum_error.append(error_norm[0])
                    cum_distance=bed_distances[0]*50

                error = np.mean(np.array(cum_error)*100, axis = 0)
                std_error = np.std(np.array(cum_error)*100, axis=0)
                self.im_sample = batch[0].numpy()
                self.im_sample = np.squeeze(self.im_sample[0, :])
                self.tar_sample = targets
                self.tar_sample = np.squeeze(self.tar_sample[0, :]) / 1000
                self.sc_sample = targets_est
                self.sc_sample = np.squeeze(self.sc_sample[0, :]) / 1000
                self.sc_sample = np.reshape(self.sc_sample, self.output_size)
                self.pseudo_sample = pseudotargets_est
                self.pseudo_sample = np.squeeze(self.pseudo_sample[0, :])/1000
                self.pseudo_sample = np.reshape(self.pseudo_sample, (5, 3))
                VisualizationLib().rviz_publish_input(self.im_sample[0, :, :], self.im_sample[-1, 10, 10])
                VisualizationLib().rviz_publish_output(np.reshape(self.tar_sample, self.output_size), self.sc_sample, self.pseudo_sample)
                VisualizationLib().visualize_pressure_map(self.im_sample, self.tar_sample, self.sc_sample, block=True)
                # VisualizationLib().visualize_error_from_distance(bed_distances, error_norm)

                if False:

                    x2.append(std_error[2])
                    y2.append(error[2])
                    x3.append(std_error[3])
                    y3.append(error[3])
                    x4.append(std_error[4])
                    y4.append(error[4])
                    x5.append(std_error[5])
                    y5.append(error[5])
                    print batch_idx
                    if batch_idx > 500:
                        xlim = [0, 4]
                        ylim = [0, 50]
                        fig = plt.figure()
                        plt.suptitle('Subject 4 Validation. Euclidean Error as a function of Gaussian noise perturbations to output angles', fontsize = 16)

                        ax1 = fig.add_subplot(2, 2, 1)
                        ax1.set_xlim(xlim)
                        ax1.set_ylim(ylim)
                        ax1.set_xlabel('Std. Dev of 3D joint position following 15 noise perturbations per image')
                        ax1.set_ylabel('Mean Euclidean Error across 15 forward passes')
                        ax1.plot(x2, y2, 'ro')
                        ax1.set_title('Right Elbow')

                        ax2 = fig.add_subplot(2, 2, 2)
                        ax2.set_xlim(xlim)
                        ax2.set_ylim(ylim)
                        ax2.set_xlabel('Std. Dev of 3D joint position following 15 noise perturbations per image')
                        ax2.set_ylabel('Mean Euclidean Error across 15 forward passes')
                        ax2.plot(x3, y3, 'bo')
                        ax2.set_title('Left Elbow')

                        ax3 = fig.add_subplot(2, 2, 3)
                        ax3.set_xlim(xlim)
                        ax3.set_ylim(ylim)
                        ax3.set_xlabel('Std. Dev of 3D joint position following 15 noise perturbations per image')
                        ax3.set_ylabel('Mean Euclidean Error across 15 forward passes')
                        ax3.plot(x4, y4, 'ro')
                        ax3.set_title('Right Hand')

                        ax4 = fig.add_subplot(2, 2, 4)
                        ax4.set_xlim(xlim)
                        ax4.set_ylim(ylim)
                        ax4.set_xlabel('Std. Dev of 3D joint position following 15 noise perturbations per image')
                        ax4.set_ylabel('Mean Euclidean Error across 15 forward passes')
                        ax4.plot(x5, y5, 'bo')
                        ax4.set_title('Left Hand')

                        plt.show()

            elif self.loss_vector_type == 'arms_cascade':
                # append upper joint angles, lower joint angles, upper joint lengths, lower joint lengths, in that order
                batch.append(torch.cat((batch[1][:, 39:49], batch[1][:, 57:65], batch[1][:, 30:39], batch[1][:, 49:57]), dim=1))

                # get the targets and pseudotargets
                batch[1] = torch.cat((batch[1][:, 0:30], batch[1][:, 65:80]), dim=1)

                images, targets, constraints = Variable(batch[0], volatile=True, requires_grad=False), Variable(batch[1], volatile=True, requires_grad=False), Variable(batch[2], volatile=True,requires_grad=False)

                #prior_cascade = torch.cat((targets[:, 3:6], constraints[:, 18:26] / 100), dim=1)  # just use this method to check, do a forward pass through the prior cascade when its trained
                #print prior_cascade[0, :]


                images_up = Variable(torch.Tensor(np.array(PreprocessingLib().preprocessing_pressure_map_upsample(batch[0].numpy()[:, :, 5:79, 5:42]))),requires_grad=False)
                _, targets_prior, angles_prior, lengths_prior, pseudotargets_prior = model_cascade_prior.forward_kinematic_jacobian(images_up, targets, forward_only=True)

                prior_cascade = torch.cat((Variable(torch.Tensor(targets_prior[:, 3:6])), Variable(torch.Tensor(lengths_prior[:, 0:8]))), dim = 1)


                print targets.data.numpy()[:, 0:30].shape
                print targets_prior.shape
                VisualizationLib().print_error(targets.data.numpy()[:, 0:30], targets_prior, (10, 3), 'angles', data = 'validate')

                targets_prior = np.concatenate((targets_prior, pseudotargets_prior), axis=1)

                # find the pressure mat coordinates where the projected markers lie
                targets_2D = CascadeLib().get_2D_projection(images.data.numpy(), np.reshape(targets.data.numpy(),(targets.size()[0], 15, 3)))
                targets_2D_prior = CascadeLib().get_2D_projection(images.data.numpy(),np.reshape(targets_prior, (targets.size()[0], 15, 3)))

                box_centers = np.round(targets_2D_prior[:, :, 0:2] / 28.6, 0)
                image_cascade_right = CascadeLib().generate_bounded_box_input(batch[0].numpy(), box_centers, limb = 'right_arm')
                image_cascade_right_up = PreprocessingLib().preprocessing_pressure_map_upsample(image_cascade_right)
                image_cascade_right_tensor = Variable(torch.Tensor(image_cascade_right_up), requires_grad=False)

                scores, targets_est, angles_est, _, _ = model.forward_kinematic_jacobian(image_cascade_right_tensor, targets, constraints, prior_cascade=prior_cascade, body_side = 'right')

                print angles_est[0,:]


                ground_truth = np.zeros((batch[0].numpy().shape[0], 6))  # 6 is 6 joint locations for the x y z right side elbow and hand
                ground_truth = Variable(torch.Tensor(ground_truth))
                targets2 = targets  # if we want to visualize the 3D targets using only x /y and not the normal projection on the mat, visualize this
                targets = torch.cat((targets[:, 6:9], targets[:, 12:15]), dim=1)
                ground_truth[:, 0:6] = targets / 1000.
                VisualizationLib().print_error(targets.data.numpy(), targets_est, self.output_size, self.loss_vector_type, data='cascade validate')


                print angles_est[0, :], 'angles'
                im_sample = np.squeeze(batch[0].numpy()[0, 0, :])
                tar_sample = targets_2D[0, :, :].flatten() / 1000  # targets2.data.numpy()#batch[1].numpy()
                tar_prior_sample = targets_2D_prior[0, :, :, ].flatten() / 1000
                full_im_tar_prior = []
                full_im_tar_prior.append(im_sample)
                full_im_tar_prior.append(tar_sample)
                full_im_tar_prior.append(tar_prior_sample)

                cascade_im_sample = image_cascade[0, :, :, :]
                cascade_tar_sample = targets.data.numpy()[0, :]
                cascade_sc_sample = targets_est[0, :]
                cascade_im_tar_sc = []
                cascade_im_tar_sc.append(cascade_im_sample)
                cascade_im_tar_sc.append(cascade_tar_sample)
                cascade_im_tar_sc.append(cascade_sc_sample)
                cascade_im_tar_sc.append(box_centers)
                VisualizationLib().visualize_pressure_map_cascade(full_im_tar_prior, cascade_im_tar_sc, block = True)



            elif self.loss_vector_type == None:
                _, targets_est = model.forward_direct(images, targets)

        return mean, stdev

    def run(self):
        '''Runs either the synthetic database creation script or the
        raw dataset creation script to create a dataset'''
        self.validate_convnet()






if __name__ == "__main__":

    import optparse
    p = optparse.OptionParser()

    p.add_option('--training_data_path', '--path',  action='store', type='string', \
                 dest='trainingPath',\
                 default='/home/henryclever/hrl_file_server/Autobed/pose_estimation_data/subject_', \
                 help='Set path to the training database.')
    p.add_option('--lab_hd', action='store_true',
                 dest='lab_harddrive', \
                 default=False, \
                 help='Set path to the training database on lab harddrive.')
    p.add_option('--upper_only', action='store_true',
                 dest='upper_only', \
                 default=False, \
                 help='Train only on data from the arms, both sitting and laying.')

    opt, args = p.parse_args()

    PathKey = 'lab_hd'

    if PathKey == 'lab_hd':
        Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/'
    elif PathKey == 'hc_desktop':
        Path = '/home/henryclever/hrl_file_server/Autobed/'
    else:
        Path = None

    print Path

    #Initialize trainer with a training database file
    p = DataVisualizer(pkl_directory=Path, pathkey = PathKey, opt = opt)
    p.run()
    sys.exit()
