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
from synthetic_lib import SyntheticLib

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
    def __init__(self, pkl_directory,  opt):
        self.opt = opt
        self.sitting = False
        self.subject = self.opt.leave_out
        self.armsup = False
        self.alldata = False
        self.verbose = True
        self.old = False
        self.normalize = True
        self.include_inter = True
        self.loss_vector_type = self.opt.losstype
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

        if self.opt.computer == 'lab_harddrive':
            train_val_loss = load_pickle(self.dump_path + '/train_val_losses.p')
            train_val_loss_desk = load_pickle(self.dump_path + '/train_val_losses_hcdesktop.p')
            train_val_loss_13 = load_pickle(self.dump_path + '/train_val_losses_all_13.p')
            train_val_loss_GPU2 = load_pickle(self.dump_path + '/train_val_losses_GPU2.p')
            train_val_loss_GPU3 = load_pickle(self.dump_path + '/train_val_losses_GPU3.p')
            train_val_loss_GPU4 = load_pickle(self.dump_path + '/train_val_losses_GPU4.p')
            train_val_loss_13 = load_pickle(self.dump_path + '/train_val_losses_all_13.p')
            train_val_loss_GPU2new = load_pickle(self.dump_path + '/train_val_lossesGPU2_021018.p')
            train_val_loss_GPU3new = load_pickle(self.dump_path + '/train_val_lossesGPU3_021018.p')
            train_val_loss_GPU4new = load_pickle(self.dump_path + '/train_val_lossesGPU4_021018.p')
            for key in train_val_loss:
                print key
            print '###########################  done with laptop #################'
            for key in train_val_loss_desk:
                print key
            print '###########################  done with desktop ################'
            for key in train_val_loss_13:
                print key
            print '###########################  done with 13 cL ################'
            for key in train_val_loss_GPU4:
                print key
            print '###########################  done with GPU4 ################'
            for key in train_val_loss_GPU3:
                print key
            print '###########################  done with GPU3 ################'
            for key in train_val_loss_GPU2:
                print key
            print '###########################  done with GPU2 ################'
            for key in train_val_loss_GPU4new:
                print key
            print '###########################  done with GPU4new ################'
            for key in train_val_loss_GPU3new:
                print key
            print '###########################  done with GPU3new ################'
            for key in train_val_loss_GPU2new:
                print key
            print '###########################  done with GPU2new ################'

            if self.subject == 1:

                plt.plot(train_val_loss_desk['epoch_armsup_700e_1'], train_val_loss_desk['val_armsup_700e_1'], 'k',label='Raw Pressure Map Input')
                plt.plot(train_val_loss['epoch_sitting_flip_700e_4'], train_val_loss['val_sitting_flip_700e_4'], 'c',label='Synthetic Flipping: $Pr(X=flip)=0.5$')
                #plt.plot(train_val_loss_desk['epoch_armsup_flip_shift_scale10_700e_1'],train_val_loss_desk['val_armsup_flip_shift_scale10_700e_1'], 'g',label='Synthetic Flipping+Shifting: $X,Y \sim N(\mu,\sigma), \mu = 0 cm, \sigma \~= 9 cm$')
                #plt.plot(train_val_loss_desk['epoch_armsup_flip_shift_scale5_nd_nohome_700e_1'],train_val_loss_desk['val_armsup_flip_shift_scale5_nd_nohome_700e_1'], 'y', label='Synthetic Flipping+Shifting+Scaling: $S_C \sim N(\mu,\sigma), \mu = 1, \sigma \~= 1.02$')
                plt.legend()
                plt.ylabel('Mean squared error loss over 30 joint vectors')
                plt.xlabel('Epochs, where 700 epochs ~ 4 hours')
                plt.title('Subject 1 laying validation Loss, training performed on subjects 2, 3, 4, 5, 6, 7, 8')


            elif self.subject == 4:
                if pathkey == 'lab_hd': #results presented to hrl dressing 171106


                    #plt.plot(train_val_loss_all['epoch_2to8_alldata_angles_115b_adam_200e_44'], train_val_loss_all['train_2to8_alldata_angles_115b_adam_200e_44'], 'k')
                    #plt.plot(train_val_loss_all['epoch_2to8_alldata_angles_115b_adam_200e_44'], train_val_loss_all['val_2to8_alldata_angles_115b_adam_200e_44'], 'y')
                    #plt.plot(train_val_loss_all['epoch_2to8_alldata_angles_constrained_noise_115b_100e_44'], train_val_loss_all['train_2to8_alldata_angles_constrained_noise_115b_100e_44'], 'b')
                    #plt.plot(train_val_loss_all['epoch_2to8_alldata_angles_constrained_noise_115b_100e_44'], train_val_loss_all['val_2to8_alldata_angles_constrained_noise_115b_100e_44'], 'r')
                    #plt.plot(train_val_loss_all['epoch_2to8_alldata_angles_s10to18_115b_50e_44'], train_val_loss_all['val_2to8_alldata_angles_s10to18_115b_50e_44'], 'g')
                    #plt.plot(train_val_loss_all['epoch_2to8_alldata_angles_implbedang_115b_100e_44'], train_val_loss_all['val_2to8_alldata_angles_implbedang_115b_100e_44'], 'g')
                    #plt.plot(train_val_loss_all['epoch_2to8_angles_direct128b_200e_44'], train_val_loss_all['train_2to8_angles_direct128b_200e_44'], 'k')
                    #plt.plot(train_val_loss_GPU3['epoch_2to8_angles_angles128b_200e_44'], train_val_loss_GPU3['train_2to8_angles_angles128b_200e_44'], 'b')
                    #plt.plot(train_val_loss_GPU4['epoch_2to8_angles128b_200e_44'], train_val_loss_GPU4['val_2to8_angles128b_200e_44'], 'k') #pull out lengths weighted 1
                    #plt.plot(train_val_loss_GPU3['epoch_2to8_angles128b_200e_44'], train_val_loss_GPU3['val_2to8_angles128b_200e_44'], 'r') #propogated lengths weighted 1
                    plt.plot(train_val_loss_GPU3new['epoch_2to8_angles128b_200e_44'], train_val_loss_GPU3new['val_2to8_angles128b_200e_44'], 'y') #propogated lengths weighted 0.2
                    plt.plot(train_val_loss_GPU2['epoch_2to8_angles_direct128b_200e_44'],train_val_loss_GPU2['val_2to8_angles_direct128b_200e_44'], 'g')
                    plt.plot(train_val_loss_GPU4new['epoch_2to8_direct128b_200e_411'],train_val_loss_GPU4new['val_2to8_direct128b_200e_411'], 'c')
                    #plt.plot(train_val_loss_GPU2new['epoch_2to8_angles128b_250e_44'],train_val_loss_GPU2new['val_2to8_angles128b_250e_44'], 'b') #pull out lengths weighted 2

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

            elif self.subject == 13:

                plt.plot(train_val_loss_13['epoch_2to8_angles128b_200e_413'], train_val_loss_13['train_2to8_angles128b_200e_413'], 'k')
                plt.plot(train_val_loss_13['epoch_2to8_angles128b_200e_413'], train_val_loss_13['val_2to8_angles128b_200e_413'], 'y')



            #plt.axis([0,410,0,30000])
            plt.axis([0, 200, 10, 30])
            #if self.opt.visualize == True:
            #    plt.show()
            plt.close()


            rospy.init_node('plot_loss')
        self.count = 0


    def init(self, subject_num):
        print 'loading subject ', subject_num
        if self.opt.computer == 'aws':
            self.validation_set = load_pickle(self.dump_path + '/subject_' + str(subject_num) + '/trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p')
        else:
            self.validation_set = load_pickle(self.dump_path + '/subject_' + str(subject_num) + '/p_files/trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p')
            #self.validation_set = load_pickle(self.dump_path + '/subject_' + str(subject_num) + '/p_files/trainval_200rlh1_115rlh2_75rlh3_150rll.p')
            #self.validation_set = load_pickle(self.dump_path + '/subject_' + str(subject_num) + '/p_files/trainval_sit175rlh_sit120rll.p')


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
            c = np.concatenate((test_dat['markers_xyz_m'][entry][0:30] * 1000,
                                test_dat['joint_lengths_U_m'][entry][0:9] * 100,
                                test_dat['joint_angles_U_deg'][entry][0:10],
                                test_dat['joint_lengths_L_m'][entry][0:8] * 100,
                                test_dat['joint_angles_L_deg'][entry][0:8]), axis=0)
            self.test_y_flat.append(c)
        self.test_y_tensor = torch.Tensor(self.test_y_flat)

        print 'Finished converting outputs to a torch tensor'




    def validate(self):


        print len(self.validation_set), 'size of validation set'
        batch_size = 1
        generate_confidence = True
        self.test_dataset = torch.utils.data.TensorDataset(self.test_x_tensor, self.test_y_tensor)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size, shuffle=True)



        #torso_length_model = torch.load(self.dump_path + '/subject_' + str(self.subject) + '/p_files/convnet_2to8_alldata_armsonly_torso_lengths_115b_adam_100e_4.pt')
        #angle_model = torch.load(self.dump_path + '/subject_' + str(self.subject) + '/p_files/convnet_2to8_alldata_armsonly_upper_angles_115b_adam_200e_4.pt')

        if self.loss_vector_type == 'angles' and self.opt.mltype == 'convnet':
            print 'loading kinematic CNN'
            if self.opt.computer == 'aws':
                model_kin = torch.load(self.dump_path + '/subject_' + str(self.subject) + '/convnet_2to8_angles128b_200e_' + str(self.subject) + '.pt')
            else:
                model_kin = torch.load(self.dump_path + '/subject_' + str(self.subject) + '/p_files/convnet_2to8_angles128b_200e_cL_' + str(self.subject) + '.pt', map_location=lambda storage, loc: storage)
            pp = 0
            for p in list(model_kin.parameters()):
                nn = 1
                for s in list(p.size()):
                    nn = nn * s
                pp += nn
            print pp, 'num params'
        if self.loss_vector_type == 'direct' and self.opt.mltype == 'convnet':
            print 'loading direct CNN'
            if self.opt.computer == 'aws':
                model_dir = torch.load(self.dump_path + '/subject_' + str(self.subject) + '/convnet_2to8_direct128b_200e_'+str(self.subject)+'.pt')
            else:
                model_dir = torch.load(self.dump_path + '/subject_' + str(self.subject) + '/p_files/convnet_2to8_direct128b_200e_'+str(self.subject)+'.pt', map_location=lambda storage, loc: storage)
            pp = 0
            for p in list(model_dir.parameters()):
                nn = 1
                for s in list(p.size()):
                    nn = nn * s
                pp += nn
            print pp, 'num params'

        all_eval = False

        if all_eval == True: self.opt.mltype = 'KNN'
        if self.opt.mltype == 'KNN':
            print 'loading KNN'
            if self.opt.computer == 'aws':
                regr_KNN = load_pickle(self.dump_path + '/subject_' + str(self.subject) + '/HoG_KNN_p'+str(self.opt.leave_out)+'.p')
            else:
                print self.dump_path + '/subject_' + str(self.subject) + '/p_files/HoG_KNN_p'+str(self.opt.leave_out)+'.p'
                regr_KNN = load_pickle(self.dump_path + '/subject_' + str(self.subject) + '/p_files/HoG_KNN_p'+str(self.opt.leave_out)+'.p')
        if all_eval == True: self.opt.mltype = 'Ridge'
        if self.opt.mltype == 'Ridge':
            print 'loading Ridge'
            if self.opt.computer == 'aws':
                regr_Ridge = load_pickle(self.dump_path + '/subject_' + str(self.subject) + '/HoG_Ridge_p'+str(self.opt.leave_out)+'.p')
            else:
                regr_Ridge = load_pickle(self.dump_path + '/subject_' + str(self.subject) + '/p_files/HoG_Ridge_p'+str(self.opt.leave_out)+'.p')

        if all_eval == True: self.opt.mltype = 'KRidge'
        if self.opt.mltype == 'KRidge':
            print 'loading Kernel Ridge'
            if self.opt.computer == 'aws':
                regr_KRidge = load_pickle(self.dump_path + '/subject_' + str(self.subject) + '/HoG_KRidge_p'+str(self.opt.leave_out)+'.p')
            else:
                regr_KRidge = load_pickle(self.dump_path + '/subject_' + str(self.subject) + '/p_files/HoG_KRidge_p'+str(self.opt.leave_out)+'.p')




        self.count = 0
        self.x2, self.y2, self.x3, self.y3, self.x4, self.y4, self.x5, self.y5 = [], [], [], [], [], [], [], []


        for batch_idx, batch in enumerate(self.test_loader):

            # append upper joint angles, lower joint angles, upper joint lengths, lower joint lengths, in that order
            batch.append(torch.cat((batch[1][:, 39:49], batch[1][:, 57:65], batch[1][:, 30:39], batch[1][:, 49:57]), dim=1))

            # get the direct joint locations
            batch[1] = batch[1][:, 0:30]

            self.count += 1
            #print count

            #model_kin.eval()

            if self.loss_vector_type == 'angles' and self.opt.mltype == 'convnet':


                cum_error = []
                limbArray = None

                if generate_confidence == True:
                    limit = 25
                else:
                    limit = 1


                if generate_confidence == True:
                    batch0 = batch[0].clone()
                    batch1 = batch[1].clone()
                    batch0, batch1,_ = SyntheticLib().synthetic_master(batch0, batch1,flip=False, shift=False, scale=False,bedangle=True, include_inter=self.include_inter, loss_vector_type=self.loss_vector_type)
                    batch0 = batch0.expand(limit, 3, 84, 47)
                    batch1 = batch1.expand(limit, 30)

                else:
                    batch0 = batch[0].clone()
                    batch1 = batch[1].clone()

                print batch0.size()
                print batch1.size(), 'size'

                images_up = batch0.numpy()
                images_up = images_up[:, :, 10:74, 10:37]
                images_up = PreprocessingLib().preprocessing_pressure_map_upsample(images_up)
                images_up = np.array(images_up)
                #images_up = PreprocessingLib().preprocessing_add_image_noise(images_up)
                images_up = Variable(torch.Tensor(images_up), volatile = True, requires_grad=False)
                #print images_up.size()

                images, targets, constraints = Variable(batch0, volatile = True, requires_grad=False), Variable(batch1, volatile = True, requires_grad=False), Variable(batch[2], volatile = True, requires_grad=False)

                _, targets_est, angles_est, lengths_est, pseudotargets_est = model_kin.forward_kinematic_jacobian(images_up, targets, constraints, prior_cascade = None, forward_only = True, subject = self.opt.leave_out)

                #print lengths_est, 'lengths'
                #print targets_est

                bed_distances = KinematicsLib().get_bed_distance(images, targets)


                targets = targets.data

                error_norm, error_avg, error_avg_std = VisualizationLib().print_error(targets, targets_est, self.output_size, self.loss_vector_type, data=str(self.count), printerror =  True)
                error_norm = error_norm/1000

                #print 'blah'
                if generate_confidence == True:
                    print batch_idx #angles_est
                    cum_error.append(error_norm[0])
                    #cum_distance=bed_distances[0]*50

                    try:
                        cum_distance = torch.cat((cum_distance, (targets_est - targets)), 0)
                        cum_angles = torch.cat((cum_angles, angles_est), 0)
                    except:
                        cum_distance = (targets_est - targets)
                        cum_angles = angles_est.clone()
                        #cum_distance = cum_distance.unsqueeze(0)

                    try:
                        self.error_avg_list.append(error_avg)
                        self.error_std_list.append(error_avg_std)
                    except:
                        self.error_avg_list = []
                        self.error_std_list = []


                count2 = 0
                while count2 < limit:
                    print angles_est[0,:]

                    self.im_sample = batch0.numpy()
                    self.im_sample = np.squeeze(self.im_sample[count2, :])
                    self.tar_sample = targets
                    self.tar_sample = np.squeeze(self.tar_sample[count2, :]) / 1000
                    self.sc_sample = targets_est
                    self.sc_sample = np.squeeze(self.sc_sample[count2, :]) / 1000
                    self.sc_sample = np.reshape(self.sc_sample, self.output_size)
                    self.pseudo_sample = pseudotargets_est
                    self.pseudo_sample = np.squeeze(self.pseudo_sample[count2, :]) / 1000
                    self.pseudo_sample = np.reshape(self.pseudo_sample, (5, 3))


                    if self.opt.visualize == True:
                        if count2 <= 1: VisualizationLib().rviz_publish_input(self.im_sample[0, :, :]*1.3, self.im_sample[-1, 10, 10])
                        #else: VisualizationLib().rviz_publish_input(self.im_sample[1, :, :]/2, self.im_sample[-1, 10, 10])

                        VisualizationLib().rviz_publish_output(np.reshape(self.tar_sample, self.output_size), self.sc_sample, self.pseudo_sample)
                        limbArray = VisualizationLib().rviz_publish_output_limbs(np.reshape(self.tar_sample, self.output_size), self.sc_sample, self.pseudo_sample, LimbArray=limbArray, count = count2)

                        skip_image = VisualizationLib().visualize_pressure_map(self.im_sample, self.tar_sample, self.sc_sample,block=True, title = 'Kinematic Embedding')
                        #VisualizationLib().visualize_error_from_distance(bed_distances, error_norm)
                        if skip_image == 1:
                            count2 = limit

                    count2 += 1

                if generate_confidence == True:
                    cum_distance = np.array(cum_distance)
                    mean_distance = np.mean(cum_distance, axis = 0)


                    cum_distance = np.reshape(cum_distance, (cum_distance.shape[0], 10, 3))


                    mean_distance = np.reshape(mean_distance, (10, 3))
                    std_distance = 0.
                    for point in np.arange(cum_distance.shape[0]):
                        #print np.square(cum_distance[point,:,0] - mean_distance[:,0])
                        std_distance += (np.square(cum_distance[point,:,0] - mean_distance[:,0]) + np.square(cum_distance[point,:,1] - mean_distance[:,1]) + np.square(cum_distance[point,:,2] - mean_distance[:,2]))/cum_distance.shape[0]
                    std_distance = np.sqrt(std_distance)/10
                    #print angles_est


                if generate_confidence == True:

                    error = np.mean(np.array(cum_error) * 100, axis=0)
                    std_error = std_distance

                    self.x2.append(std_error[2])


                    self.y2.append(error[2])
                    self.x3.append(std_error[3])
                    self.y3.append(error[3])
                    self.x4.append(std_error[4])
                    self.y4.append(error[4])
                    self.x5.append(std_error[5])
                    self.y5.append(error[5])
                    print batch_idx
                    if batch_idx == 1669:
                        #VisualizationLib().visualize_error_threshold(self.error_avg_list)
                        #VisualizationLib().visualize_variance_threshold(self.error_avg_list, self.error_std_list)
                        #pkl.dump([self.error_avg_list, self.error_std_list], open('/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/Final_Data/error_avg_std_T25_subject'+str(self.subject)+'_kinvL.p', 'wb'))

                        xlim = [0, 20]
                        ylim = [0, 60]
                        fig = plt.figure()
                        plt.suptitle('Subject 4 Validation. Euclidean Error as a function of Gaussian noise perturbations to input images and shifting augmentation', fontsize = 16)

                        ax1 = fig.add_subplot(2, 2, 1)
                        ax1.set_xlim(xlim)
                        ax1.set_ylim(ylim)
                        ax1.set_xlabel('Std. Dev of 3D joint position following 15 noise perturbations per image')
                        ax1.set_ylabel('Mean Euclidean Error across 15 forward passes')
                        ax1.plot(self.x2, self.y2, 'ro')
                        ax1.set_title('Right Elbow')

                        ax2 = fig.add_subplot(2, 2, 2)
                        ax2.set_xlim(xlim)
                        ax2.set_ylim(ylim)
                        ax2.set_xlabel('Std. Dev of 3D joint position following 15 noise perturbations per image')
                        ax2.set_ylabel('Mean Euclidean Error across 15 forward passes')
                        ax2.plot(self.x3, self.y3, 'bo')
                        ax2.set_title('Left Elbow')

                        ax3 = fig.add_subplot(2, 2, 3)
                        ax3.set_xlim(xlim)
                        ax3.set_ylim(ylim)
                        ax3.set_xlabel('Std. Dev of 3D joint position following 15 noise perturbations per image')
                        ax3.set_ylabel('Mean Euclidean Error across 15 forward passes')
                        ax3.plot(self.x4, self.y4, 'ro')
                        ax3.set_title('Right Hand')

                        ax4 = fig.add_subplot(2, 2, 4)
                        ax4.set_xlim(xlim)
                        ax4.set_ylim(ylim)
                        ax4.set_xlabel('Std. Dev of 3D joint position following 15 noise perturbations per image')
                        ax4.set_ylabel('Mean Euclidean Error across 15 forward passes')
                        ax4.plot(self.x5, self.y5, 'bo')
                        ax4.set_title('Left Hand')

                        if self.opt.visualize == True:
                            plt.show()

            if self.loss_vector_type == 'direct' and self.opt.mltype == 'convnet':

                count2 = 0
                limbArray = None

                if generate_confidence == True:
                    limit = 25
                else:
                    limit = 1

                if generate_confidence == True:
                    batch0 = batch[0].clone()
                    batch1 = batch[1].clone()
                    batch0, batch1,_ = SyntheticLib().synthetic_master(batch0, batch1,flip=False, shift=False, scale=False,bedangle=True, include_inter=self.include_inter, loss_vector_type=self.loss_vector_type)
                    batch0 = batch0.expand(limit, 3, 84, 47)
                    batch1 = batch1.expand(limit, 30)

                else:
                    batch0 = batch[0].clone()
                    batch1 = batch[1].clone()



                images_up = batch0.numpy()
                images_up = images_up[:, :, 10:74, 10:37]
                images_up = PreprocessingLib().preprocessing_pressure_map_upsample(images_up)
                images_up = np.array(images_up)
                images_up = Variable(torch.Tensor(images_up), volatile=True, requires_grad=False)

                images, targets = Variable(batch0, volatile=True, requires_grad=False), Variable(batch1,volatile=True,requires_grad=False)

                _, targets_est = model_dir.forward_direct(images_up, targets)

                print torch.norm(targets.data[0,6:9]-targets.data[0,12:15]), 'ground truth R forearm length'
                print torch.norm(targets_est[0,6:9]-targets_est[0,12:15]), 'estimated R forearm length'
                print torch.norm(targets.data[0,9:12]-targets.data[0,15:18]), 'ground truth L forearm length'
                print torch.norm(targets_est[0,9:12]-targets_est[0,15:18]), 'estimated L forearm length'

                error_norm, error_avg, error_avg_std = VisualizationLib().print_error(targets.data, targets_est, self.output_size,self.loss_vector_type, data=str(self.count),printerror=True)
                error_norm = error_norm /1000
                print error_norm.shape



                if generate_confidence == True:
                    try:
                        cum_distance = torch.cat((cum_distance, (targets_est - targets.data)), 0)
                    except:
                        cum_distance = (targets_est - targets.data)
                        #cum_distance = cum_distance.unsqueeze(0)

                    try:
                        self.error_avg_list.append(error_avg)
                        self.error_std_list.append(error_avg_std)
                    except:
                        self.error_avg_list = []
                        self.error_std_list = []

                while count2 < limit:
                    self.im_sample = batch0.numpy()
                    self.im_sample = self.im_sample[count2, :].squeeze()
                    self.tar_sample = targets.data
                    self.tar_sample = self.tar_sample[count2, :].squeeze() / 1000
                    self.sc_sample = targets_est
                    self.sc_sample = self.sc_sample[count2, :].squeeze() / 1000
                    self.sc_sample = self.sc_sample.view(self.output_size)

                    if False:#self.opt.visualize == True:
                        VisualizationLib().rviz_publish_input(self.im_sample[0, :, :] * 1.3, self.im_sample[-1, 10, 10])
                        # else: VisualizationLib().rviz_publish_input(self.im_sample[1, :, :]/2, self.im_sample[-1, 10, 10])

                        VisualizationLib().rviz_publish_output(np.reshape(self.tar_sample, self.output_size), self.sc_sample)
                        limbArray = VisualizationLib().rviz_publish_output_limbs_direct(np.reshape(self.tar_sample, self.output_size), self.sc_sample, LimbArray=limbArray, count=count2)

                        skip_image = VisualizationLib().visualize_pressure_map(self.im_sample, self.tar_sample,self.sc_sample, block=True, title = 'Direct Regression')
                        if skip_image == 1:
                            count2 = limit

                    count2 += 1



                if generate_confidence == True:
                    if batch_idx == 1669:
                        pkl.dump([self.error_avg_list, self.error_std_list], open('/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/Final_Data/error_avg_std_T25_subject' + str(self.subject) + '_direct.p', 'wb'))

                    print cum_distance.size()
                    cum_distance = np.array(cum_distance)
                    mean_distance = np.mean(cum_distance, axis = 0)


                    cum_distance = np.reshape(cum_distance, (cum_distance.shape[0], 10, 3))

                    mean_distance = np.reshape(mean_distance, (10, 3))
                    std_distance = 0.
                    for point in np.arange(cum_distance.shape[0]):
                        #print np.square(cum_distance[point,:,0] - mean_distance[:,0])
                        std_distance += (np.square(cum_distance[point,:,0] - mean_distance[:,0]) + np.square(cum_distance[point,:,1] - mean_distance[:,1]) + np.square(cum_distance[point,:,2] - mean_distance[:,2]))/cum_distance.shape[0]
                    std_distance = np.sqrt(std_distance)/10
                    #print angles_est


            if self.opt.mltype == 'KNN' or self.opt.mltype == 'Ridge' or self.opt.mltype == 'KRidge':

                batch0 = batch[0].clone()
                batch1 = batch[1].clone()
                images = batch0.numpy()[:, 0, :, :]
                targets = batch1.numpy()

                # upsample the images
                images_up = PreprocessingLib().preprocessing_pressure_map_upsample(images)

                print np.shape(images_up)
                # Compute HoG of the current(training) pressure map dataset
                images_up = PreprocessingLib().compute_HoG(images_up)

                if all_eval == True: self.opt.mltype = 'KNN'
                if self.opt.mltype == 'KNN':
                    scores = regr_KNN.predict(images_up)

                    error_norm,_, _ =VisualizationLib().print_error(scores, targets, self.output_size,loss_vector_type=self.loss_vector_type, data='test', printerror=True)

                    pkl.dump(error_norm, open( '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/Final_Data/error_avg_subject' + str(self.subject) + '_KNN.p', 'wb'))

                    self.im_sample = np.squeeze(images[0, :])
                    print self.im_sample.shape

                    self.tar_sample = np.squeeze(targets[0, :]) / 1000
                    self.sc_sample = np.copy(scores)
                    self.sc_sample = np.squeeze(self.sc_sample[0, :]) / 1000
                    self.sc_sample = np.reshape(self.sc_sample, self.output_size)
                    if self.opt.visualize == True:
                        VisualizationLib().visualize_pressure_map(self.im_sample, self.tar_sample, self.sc_sample,block=True, title = 'KNN')

                if all_eval == True: self.opt.mltype = 'Ridge'
                if self.opt.mltype == 'Ridge':
                    scores = regr_Ridge.predict(images_up)

                    error_norm, _, _ = VisualizationLib().print_error(scores, targets, self.output_size,loss_vector_type=self.loss_vector_type, data='test', printerror=True)

                    pkl.dump(error_norm, open('/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/Final_Data/error_avg_subject' + str(self.subject) + '_LRR.p', 'wb'))

                    self.im_sample = np.squeeze(images[0, :])
                    print self.im_sample.shape

                    self.tar_sample = np.squeeze(targets[0, :]) / 1000
                    self.sc_sample = np.copy(scores)
                    self.sc_sample = np.squeeze(self.sc_sample[0, :]) / 1000
                    self.sc_sample = np.reshape(self.sc_sample, self.output_size)
                    if self.opt.visualize == True:
                        VisualizationLib().visualize_pressure_map(self.im_sample, self.tar_sample, self.sc_sample,block=True, title = 'Linear Ridge')

                if all_eval == True: self.opt.mltype = 'KRidge'
                if self.opt.mltype == 'KRidge':
                    scores = regr_KRidge.predict(images_up)

                    error_norm, _, _ = VisualizationLib().print_error(scores, targets, self.output_size,loss_vector_type=self.loss_vector_type, data='test', printerror=True)

                    pkl.dump(error_norm, open('/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/Final_Data/error_avg_subject' + str(self.subject) + '_KRR.p', 'wb'))

                    self.im_sample = np.squeeze(images[0, :])
                    print self.im_sample.shape

                    self.tar_sample = np.squeeze(targets[0, :]) / 1000
                    self.sc_sample = np.copy(scores)
                    self.sc_sample = np.squeeze(self.sc_sample[0, :]) / 1000
                    self.sc_sample = np.reshape(self.sc_sample, self.output_size)
                    if self.opt.visualize == True:
                        VisualizationLib().visualize_pressure_map(self.im_sample, self.tar_sample, self.sc_sample,block=True, title = 'Kernel Ridge')


        return


   # def forward_batch(self, batch_idx, batch):



    def run(self):
        '''Runs either the synthetic database creation script or the
        raw dataset creation script to create a dataset'''
        for subject in [self.subject]:
            self.init(subject)
            self.validate()






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
    p.add_option('--leave_out', action='store', type=int, \
                 dest='leave_out', \
                 help='Specify which subject to leave out for validation')
    p.add_option('--computer', action='store', type = 'string',
                 dest='computer', \
                 default='lab_harddrive', \
                 help='Set path to the training database on lab harddrive.')
    p.add_option('--mltype', action='store', type = 'string',
                 dest='mltype', \
                 default='convnet', \
                 help='Set if you want to do baseline ML or convnet.')
    p.add_option('--losstype', action='store', type = 'string',
                 dest='losstype', \
                 default='direct', \
                 help='Set if you want to do baseline ML or convnet.')
    p.add_option('--viz', action='store_true',
                 dest='visualize', \
                 default=False, \
                 help='Train only on data from the arms, both sitting and laying.')

    opt, args = p.parse_args()


    if opt.computer == 'lab_harddrive':
        Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/'
    elif opt.computer == 'aws':
        Path = '/home/ubuntu/Autobed_OFFICIAL_Trials/'
    elif opt.computer == 'hc_desktop':
        Path = '/home/henryclever/hrl_file_server/Autobed/'
    else:
        Path = None

    print Path

    #Initialize trainer with a training database file
    p = DataVisualizer(pkl_directory=Path, opt = opt)
    p.run()
    sys.exit()
