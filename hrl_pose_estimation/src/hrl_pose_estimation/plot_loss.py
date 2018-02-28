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
#roslib.load_manifest('hrl_lib')
import cPickle as pickle
def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

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

torch.set_num_threads(1)
if torch.cuda.is_available():
    # Use for GPU
    dtype = torch.cuda.FloatTensor
    print '######################### CUDA is available! #############################'
else:
    # Use for CPU
    dtype = torch.FloatTensor
    print '############################## USING CPU #################################'


class DataVisualizer():
    '''Gets the directory of pkl database and iteratively go through each file,
    cutting up the pressure maps and creating synthetic database'''
    def __init__(self, pkl_directory,  opt):
        self.opt = opt
        self.verbose = True
        self.include_inter = True
        self.loss_vector_type = self.opt.losstype
        # Set initial parameters
        self.dump_path = pkl_directory.rstrip('/')

        self.mat_size = (NUMOFTAXELS_X, NUMOFTAXELS_Y)
        self.output_size = (NUMOFOUTPUTNODES, NUMOFOUTPUTDIMS)

        if self.opt.computer == 'lab_harddrive':


            train_val_lossdir = load_pickle(self.dump_path + '/subject_'+str(self.opt.leave_out)+'/convnets/losses_9to18_direct_sTrue_128b_300e_'+str(self.opt.leave_out)+'.p')
            for key in train_val_lossdir:
                print key
            print '###########################  done with subject ',str(self.opt.leave_out),', direct ##############################'


            train_val_lossCL = load_pickle(self.dump_path + '/subject_'+str(self.opt.leave_out)+'/convnets/losses_9to18_anglesCL_sTrue_128b_300e_'+str(self.opt.leave_out)+'.p')
            for key in train_val_lossCL:
                print key
            print '###########################  done with subject ',str(self.opt.leave_out),', anglesCL ##############################'

            train_val_lossSTVL = load_pickle(self.dump_path + '/subject_'+str(self.opt.leave_out)+'/convnets/losses_9to18_anglesSTVL_sTrue_128b_300e_'+str(self.opt.leave_out)+'.p')
            for key in train_val_lossSTVL:
                print key
            print '###########################  done with subject ',str(self.opt.leave_out),', anglesSTVL ##############################'

            plt.plot(train_val_lossdir['epoch_9to18_direct_sTrue_128b_300e_' + str(self.opt.leave_out)],train_val_lossdir['val_9to18_direct_sTrue_128b_300e_' + str(self.opt.leave_out)], 'r',label='Direct CNN')
            plt.plot(train_val_lossCL['epoch_9to18_anglesCL_sTrue_128b_300e_'+str(self.opt.leave_out)],train_val_lossCL['val_9to18_anglesCL_sTrue_128b_300e_'+str(self.opt.leave_out)],'g',label='Kinematic CNN, const L')
            plt.plot(train_val_lossSTVL['epoch_9to18_anglesSTVL_sTrue_128b_300e_'+str(self.opt.leave_out)],train_val_lossSTVL['val_9to18_anglesSTVL_sTrue_128b_300e_'+str(self.opt.leave_out)],'b',label='Kinematic CNN, var L')


            if self.opt.leave_out == 1:

                plt.plot(train_val_loss_desk['epoch_armsup_700e_1'], train_val_loss_desk['val_armsup_700e_1'], 'k',label='Raw Pressure Map Input')
                plt.plot(train_val_loss['epoch_sitting_flip_700e_4'], train_val_loss['val_sitting_flip_700e_4'], 'c',label='Synthetic Flipping: $Pr(X=flip)=0.5$')
                #plt.plot(train_val_loss_desk['epoch_armsup_flip_shift_scale10_700e_1'],train_val_loss_desk['val_armsup_flip_shift_scale10_700e_1'], 'g',label='Synthetic Flipping+Shifting: $X,Y \sim N(\mu,\sigma), \mu = 0 cm, \sigma \~= 9 cm$')
                #plt.plot(train_val_loss_desk['epoch_armsup_flip_shift_scale5_nd_nohome_700e_1'],train_val_loss_desk['val_armsup_flip_shift_scale5_nd_nohome_700e_1'], 'y', label='Synthetic Flipping+Shifting+Scaling: $S_C \sim N(\mu,\sigma), \mu = 1, \sigma \~= 1.02$')


            plt.legend()


            plt.ylabel('L1 loss over 10 joint Euclidean errors')
            plt.xlabel('Epochs, where 200 epochs ~ 10 hours')
            plt.title('Subject '+str(self.opt.leave_out)+' validation Loss')


            #plt.axis([0,410,0,30000])
            plt.axis([0, 300, 0, 30])
            if self.opt.visualize == True:
                plt.show()
            plt.close()

        elif opt.computer == 'aws':
            if self.loss_vector_type == 'direct':
                print self.dump_path + '/subject_' + str(self.opt.leave_out) + '/losses_9to18_direct_sTrue_128b_300e_'+str(self.opt.leave_out)+'.p'


        if self.opt.visualize == True:
            rospy.init_node('plot_loss')
        self.count = 0


    def init(self, subject_num):
        print 'loading subject ', subject_num
        if self.opt.computer == 'aws':
            self.validation_set = load_pickle(self.dump_path + '/subject_' + str(subject_num) + '/trainval_200rlh1_115rlh2_75rlh3_175rllair.p')
            #self.validation_set = load_pickle(self.dump_path + '/subject_' + str(subject_num) + '/trainval_sit175rlh_sit120rll.p')

        else:

            #self.validation_set = load_pickle(self.dump_path + '/subject_' + str(subject_num) + '/p_files/trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p')
            #self.validation_set = load_pickle(self.dump_path + '/subject_' + str(subject_num) + '/p_files/trainval_200rlh1_115rlh2_75rlh3_175rllair.p')
            self.validation_set = load_pickle(self.dump_path + '/subject_' + str(subject_num) + '/p_files/trainval_sit175rlh_sit120rll.p')
            #self.validation_set = load_pickle(self.dump_path + '/subject_' + str(subject_num) + '/p_files/150RL_LL_air.p')
            #self.validation_set = load_pickle(self.dump_path + '/subject_' + str(subject_num) + '/p_files/trainvalLL.p')


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
        batch_size = 1#always do 1! you have to do 25 forward passes with a single labeled image

        self.test_dataset = torch.utils.data.TensorDataset(self.test_x_tensor, self.test_y_tensor)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size, shuffle=True)

        models = []
        all_eval = False

        dropout = True


        #if all_eval == True: self.loss_vector_type = 'anglesCL'
        if self.loss_vector_type == 'anglesCL':
            models.append('anglesCL')
            print 'loading kinematic constant bone lengths CNN, subject ', self.opt.leave_out
            if self.opt.computer == 'aws':
                model_kinCL = torch.load(self.dump_path + '/subject_' + str(self.opt.leave_out) + '/convnets/convnet_9to18_anglesCL_sTrue_128b_300e_'+str(self.opt.leave_out)+'.pt', map_location=lambda storage, loc: storage)
            else:
                model_kinCL = torch.load(self.dump_path + '/subject_' + str(self.opt.leave_out) + '/convnets/convnet_9to18_anglesCL_sTrue_128b_300e_' + str(self.opt.leave_out) + '.pt', map_location=lambda storage, loc: storage)
            model_kinCL.train()
            pp = 0
            for p in list(model_kinCL.parameters()):
                nn = 1
                for s in list(p.size()):
                    nn = nn * s
                pp += nn
            print pp, 'num params'

       # if all_eval == True: self.loss_vector_type = 'anglesSTVL'
        if self.loss_vector_type == 'anglesSTVL':
            models.append('anglesSTVL')
            print 'loading kinematic variable bone lengths straight through CNN, subject ', self.opt.leave_out
            if self.opt.computer == 'aws':
                model_kinSTVL = torch.load(self.dump_path + '/subject_' + str(self.opt.leave_out) + '/convnets/convnet_9to18_anglesSTVL_sTrue_128b_300e_'+str(self.opt.leave_out)+'.pt', map_location=lambda storage, loc: storage)
            else:
                model_kinSTVL = torch.load(self.dump_path + '/subject_' + str(self.opt.leave_out) + '/convnets/convnet_9to18_anglesSTVL_sTrue_128b_300e_' + str(self.opt.leave_out) + '.pt', map_location=lambda storage, loc: storage)
            model_kinSTVL.train()
            pp = 0
            for p in list(model_kinSTVL.parameters()):
                nn = 1
                for s in list(p.size()):
                    nn = nn * s
                pp += nn
            print pp, 'num params'

        #if all_eval == True: self.loss_vector_type = 'direct'
        if self.loss_vector_type == 'direct':
            models.append('direct')
            print 'loading direct CNN'
            if self.opt.computer == 'aws':
                model_dir = torch.load(self.dump_path + '/subject_' + str(self.opt.leave_out) + '/convnets/convnet_9to18_direct_sTrue_128b_300e_'+str(self.opt.leave_out)+'.pt', map_location=lambda storage, loc: storage)
            else:
                model_dir = torch.load(self.dump_path + '/subject_' + str(self.opt.leave_out) + '/convnets/convnet_9to18_direct_sTrue_128b_300e_'+str(self.opt.leave_out)+'.pt', map_location=lambda storage, loc: storage)
            model_dir.train()
            pp = 0
            for p in list(model_dir.parameters()):
                nn = 1
                for s in list(p.size()):
                    nn = nn * s
                pp += nn
            print pp, 'num params'


        if all_eval == True: self.loss_vector_type = 'KNN'
        if self.loss_vector_type == 'KNN':
            models.append('KNN')
            print 'loading KNN'
            if self.opt.computer == 'aws':
                regr_KNN = load_pickle(self.dump_path + '/subject_' + str(self.opt.leave_out) + '/HoG_KNN_p'+str(self.opt.leave_out)+'.p')
            else:
                print self.dump_path + '/subject_' + str(self.opt.leave_out) + '/p_files/HoGshift_KNN_p'+str(self.opt.leave_out)+'.p'
                regr_KNN = load_pickle(self.dump_path + '/subject_' + str(self.opt.leave_out) + '/p_files/HoGshift_KNN_p'+str(self.opt.leave_out)+'.p')

        if all_eval == True: self.loss_vector_type = 'Ridge'
        if self.loss_vector_type == 'Ridge':
            models.append('Ridge')
            print 'loading Ridge'
            if self.opt.computer == 'aws':
                regr_Ridge = load_pickle(self.dump_path + '/subject_' + str(self.opt.leave_out) + '/HoGshift0.7_Ridge_p'+str(self.opt.leave_out)+'.p')
            else:
                regr_Ridge = load_pickle(self.dump_path + '/subject_' + str(self.opt.leave_out) + '/p_files/HoGshift0.7_Ridge_p'+str(self.opt.leave_out)+'.p')

        if all_eval == True: self.loss_vector_type = 'KRidge'
        if self.loss_vector_type == 'KRidge':
            models.append('KRidge')
            print 'loading Kernel Ridge'
            if self.opt.computer == 'aws':
                regr_KRidge = load_pickle(self.dump_path + '/subject_' + str(self.opt.leave_out) + '/HoGshift0.4_KRidge_p'+str(self.opt.leave_out)+'.p')
            else:
                regr_KRidge = load_pickle(self.dump_path + '/subject_' + str(self.opt.leave_out) + '/p_files/HoGshift0.4_KRidge_p'+str(self.opt.leave_out)+'.p')



        self.count = 0

        print '############################### STARTING ON MODELS', models, ' ##########################################'

        for batch_idx, batch in enumerate(self.test_loader):

            # append upper joint angles, lower joint angles, upper joint lengths, lower joint lengths, in that order
            batch.append(torch.cat((batch[1][:, 39:49], batch[1][:, 57:65], batch[1][:, 30:39], batch[1][:, 49:57]), dim=1))

            # get the direct joint locations
            batch[1] = batch[1][:, 0:30]

            self.count += 1
            # print count
            for model_key in models:

                print model_key, 'MODEL KEY'


                limbArray = None

                T = 25 #STOCHASTIC FORWARD PASSES


                batch0 = batch[0].clone()
                batch1 = batch[1].clone()
                batch0, batch1,_ = SyntheticLib().synthetic_master(batch0, batch1,flip=False, shift=False, scale=False,bedangle=True, include_inter=self.include_inter, loss_vector_type=self.loss_vector_type)

                if model_key == 'anglesVL' or model_key == 'anglesCL' or model_key == 'anglesSTVL' or model_key == 'direct':
                    batch0 = batch0.expand(T, 3, 84, 47)
                    batch1 = batch1.expand(T, 30)

                print batch0.size()
                print batch1.size(), 'size'

                images, targets, constraints = Variable(batch0, volatile=True, requires_grad=False), Variable(batch1, volatile=True, requires_grad=False), Variable(batch[2], volatile=True, requires_grad=False)
                images_up = batch0.numpy()[:, :, 10:74, 10:37]
                images_up = PreprocessingLib().preprocessing_pressure_map_upsample(images_up)

                if model_key == 'anglesVL' or model_key == 'anglesCL' or model_key == 'anglesSTVL' or model_key == 'direct':
                    images_up = np.array(images_up)
                    images_up = Variable(torch.Tensor(images_up), volatile = False, requires_grad=False)
                    #print images_up.size()


                    if model_key == 'anglesVL':
                        _, targets_est, angles_est, lengths_est, pseudotargets_est = model_kinVL.forward_kinematic_jacobian(images_up, targets, constraints, forward_only = True, subject = self.opt.leave_out, loss_vector_type = self.loss_vector_type)
                    elif model_key == 'anglesCL':
                        _, targets_est, angles_est, lengths_est, pseudotargets_est = model_kinCL.forward_kinematic_jacobian(images_up, targets, constraints, forward_only = True, subject = self.opt.leave_out, loss_vector_type = self.loss_vector_type)
                    elif model_key == 'anglesSTVL':
                        _, targets_est, angles_est, lengths_est, pseudotargets_est = model_kinSTVL.forward_kinematic_jacobian(images_up, targets, constraints, forward_only = True, subject = self.opt.leave_out, loss_vector_type = self.loss_vector_type)

                        #print targets_est[0, :]
                        #print lengths_est[0, :]
                    elif model_key == 'direct':
                        _, targets_est = model_dir.forward_direct(images_up, targets)

                        print torch.norm(targets.data[0, 6:9] - targets.data[0, 12:15]), 'ground truth R forearm length'
                        print torch.norm(targets_est[0, 6:9] - targets_est[0, 12:15]), 'estimated R forearm length'
                        print torch.norm(targets.data[0, 9:12] - targets.data[0, 15:18]), 'ground truth L forearm length'
                        print torch.norm(targets_est[0, 9:12] - targets_est[0, 15:18]), 'estimated L forearm length'


                elif model_key == 'KNN':
                    images_HOG = PreprocessingLib().compute_HoG(np.array(images_up)[:, 0, :, :])
                    targets_est = torch.Tensor(regr_KNN.predict(images_HOG))
                elif model_key == 'Ridge':
                    images_HOG = PreprocessingLib().compute_HoG(np.array(images_up)[:, 0, :, :])
                    print np.shape(images_HOG)
                    targets_est = torch.Tensor(regr_Ridge.predict(images_HOG))
                elif model_key == 'KRidge':
                    images_HOG = PreprocessingLib().compute_HoG(np.array(images_up)[:, 0, :, :])
                    targets_est = torch.Tensor(regr_KRidge.predict(images_HOG))

                targets = targets.data
                error_norm, error_avg, error_avg_std = VisualizationLib().print_error(targets, targets_est, self.output_size, model_key, data=str(self.count), printerror =  True)
                error_norm = error_norm/1000

                if model_key == 'anglesVL' or model_key == 'anglesCL' or model_key == 'anglesSTVL' or model_key == 'direct':
                    print '################## PERFORMED ',str(T),' STOCHASTIC FWD PASSES ON ',str(model_key),' CT. ',batch_idx, ' ##################'
                if model_key == 'KNN' or model_key == 'Ridge' or model_key == 'KRidge':
                    print '################## PERFORMED 1 FWD PASSES ON ',str(model_key),' CT. ',batch_idx, ' ##################'









                try:
                    self.error_avg_list.append(error_avg) #change this for the 3 first baselines! just append the error norm, targets, and targets est
                    self.targets_list.append(targets[0,:].numpy())
                    self.targets_est_list.append(np.mean(targets_est.numpy(), axis = 0))
                    xyz_std = np.reshape(np.std((targets_est.numpy() - np.mean(targets_est.numpy(), axis=0)), axis=0),self.output_size)
                    #print targets_est.numpy()[0, :]

                    #print np.mean(targets_est.numpy(), axis = 0)
                    #print (targets_est.numpy() - np.mean(targets_est.numpy(), axis = 0))[:,0]


                    norm_std = np.linalg.norm(xyz_std, axis=1)
                    self.variance_est_list.append(norm_std)
                    #print error_avg, 'error_avg'
                    #print norm_std, 'norm std'
                except:
                    self.error_avg_list = []
                    self.targets_list = []
                    self.targets_est_list = []
                    self.variance_est_list = []


                    self.error_avg_list.append(error_avg)
                    self.targets_list.append(targets[0,:].numpy())
                    self.targets_est_list.append(np.mean(targets_est.numpy(), axis = 0))
                    xyz_std = np.reshape(np.std((targets_est.numpy() - np.mean(targets_est.numpy(), axis = 0)), axis =0), self.output_size)
                    norm_std = np.linalg.norm(xyz_std, axis = 1)
                    self.variance_est_list.append(norm_std)
                    #print norm_std
                    #self.variance_est_list.append()

                print np.shape(self.variance_est_list)






                    #leg_patchR = np.squeeze(batch0.numpy()[0,:])[0, 58:74, 10:23]
                    #leg_patchL = np.squeeze(batch0.numpy()[0,:])[0, 58:74, 24:37]
                    #print np.sum(leg_patchR), np.sum(leg_patchL), 'leg patch sum'
                    #try:
                    #    self.leg_patch_sums.append([np.sum(leg_patchR), np.sum(leg_patchL)])
                    #xcept:
                    #   self.leg_patch_sums = []
                    #   self.leg_patch_sums.append([np.sum(leg_patchR), np.sum(leg_patchL)])


                count2 = 0
                if model_key == 'anglesVL' or model_key == 'anglesCL' or model_key == 'anglesSTVL':
                    print np.concatenate((np.expand_dims(np.mean(angles_est.data.numpy(), axis=0), axis=0), np.expand_dims(np.std(angles_est.data.numpy(), axis=0), axis = 0)),axis = 0)
                if model_key == 'anglesVL' or model_key == 'anglesCL' or model_key == 'anglesSTVL' or model_key == 'direct':
                    print np.array(self.variance_est_list)[batch_idx, 0:10]

                if self.opt.visualize == True:
                    self.im_sample = batch0.numpy()
                    self.im_sample = np.squeeze(self.im_sample[count2, :])
                    self.tar_sample = targets
                    self.tar_sample = np.squeeze(self.tar_sample[count2, :]) / 1000

                    self.sc_sample_mean = targets_est.numpy()
                    self.sc_sample_mean = np.squeeze(self.sc_sample_mean[:, :]) / 1000
                    self.sc_sample_mean = np.mean(self.sc_sample_mean, axis=0)
                    self.sc_sample_mean = np.reshape(self.sc_sample_mean, self.output_size)
                    print np.array(self.error_avg_list)[batch_idx, 0:10]*10
                    #print np.concatenate((np.expand_dims(np.mean(angles_est.data.numpy(), axis=0), axis=0), np.expand_dims(np.std(angles_est.data.numpy(), axis=0), axis = 0)),axis = 0)


                    if count2 <= 1: VisualizationLib().rviz_publish_input(self.im_sample[0, :, :] * 1.3,
                                                                          self.im_sample[-1, 10, 10])
                    # else: VisualizationLib().rviz_publish_input(self.im_sample[1, :, :]/2, self.im_sample[-1, 10, 10])

                    if self.loss_vector_type == 'anglesVL' or self.loss_vector_type == 'anglesCL' or self.loss_vector_type == 'anglesSTVL':

                        self.pseudo_sample_mean = pseudotargets_est
                        self.pseudo_sample_mean = np.squeeze(self.pseudo_sample_mean[:, :]) / 1000
                        self.pseudo_sample_mean = np.mean(self.pseudo_sample_mean, axis=0)
                        self.pseudo_sample_mean = np.reshape(self.pseudo_sample_mean, (5, 3))

                        VisualizationLib().rviz_publish_output(np.reshape(self.tar_sample, self.output_size),
                                                               self.sc_sample_mean, self.pseudo_sample_mean)
                        limbArray = VisualizationLib().rviz_publish_output_limbs(
                            np.reshape(self.tar_sample, self.output_size), self.sc_sample_mean, self.pseudo_sample_mean,
                            LimbArray=limbArray, count=0)
                    else:
                        VisualizationLib().rviz_publish_output(np.reshape(self.tar_sample, self.output_size),
                                                               self.sc_sample_mean)
                        limbArray = VisualizationLib().rviz_publish_output_limbs_direct(
                            np.reshape(self.tar_sample, self.output_size), self.sc_sample_mean, LimbArray=limbArray,
                            count=-1)

                    skip_image = VisualizationLib().visualize_pressure_map(self.im_sample, self.tar_sample, self.sc_sample_mean,
                                                              block=True, title=str(model_key))
                    if skip_image == 1:
                        count2 = T
                    count2 += 1

                    while count2 < T:
                        #print angles_est[0,:]


                        self.sc_sample = targets_est
                        self.sc_sample = np.squeeze(self.sc_sample[count2, :]) / 1000
                        self.sc_sample = np.reshape(self.sc_sample, self.output_size)


                        if count2 <= 1: VisualizationLib().rviz_publish_input(self.im_sample[0, :, :] * 1.3,self.im_sample[-1, 10, 10])
                        # else: VisualizationLib().rviz_publish_input(self.im_sample[1, :, :]/2, self.im_sample[-1, 10, 10])

                        if self.loss_vector_type == 'anglesVL' or self.loss_vector_type == 'anglesCL' or self.loss_vector_type == 'anglesSTVL':

                            self.pseudo_sample = pseudotargets_est
                            self.pseudo_sample = np.squeeze(self.pseudo_sample[count2, :]) / 1000
                            self.pseudo_sample = np.reshape(self.pseudo_sample, (5, 3))
                            limbArray = VisualizationLib().rviz_publish_output_limbs(np.reshape(self.tar_sample, self.output_size), self.sc_sample, self.pseudo_sample,LimbArray=limbArray, count=count2)
                        else:
                            limbArray = VisualizationLib().rviz_publish_output_limbs_direct(np.reshape(self.tar_sample, self.output_size), self.sc_sample, LimbArray=limbArray,count=count2)

                        skip_image = VisualizationLib().visualize_pressure_map(self.im_sample, self.tar_sample, self.sc_sample,block=True, title = str(model_key))
                        #VisualizationLib().visualize_error_from_distance(bed_distances, error_norm)
                        if skip_image == 1:
                            count2 = T

                        count2 += 1
                        if count2 == T:
                            VisualizationLib().rviz_publish_output_limbs(np.reshape(self.tar_sample, self.output_size), self.sc_sample_mean, self.pseudo_sample_mean, LimbArray=limbArray, count=-1)
                            VisualizationLib().visualize_pressure_map(self.im_sample, self.tar_sample, self.sc_sample, block=True, title=str(model_key))

                            VisualizationLib().rviz_publish_input(self.im_sample[0, :, :] * 1.3, self.im_sample[-1, 10, 10])
                            VisualizationLib().visualize_pressure_map(self.im_sample, self.tar_sample, self.sc_sample, block=True, title=str(model_key))



                batch_idx_limit = np.shape(self.validation_set['images'])[0]-1

                if batch_idx == batch_idx_limit and self.opt.visualize == False:
                    if model_key == 'anglesVL' or model_key == 'anglesCL' or model_key == 'anglesSTVL' or model_key == 'direct':
                        print "DUMPING!!!!!"
                        pkl.dump([self.error_avg_list, self.variance_est_list, self.targets_list, self.targets_est_list], open(self.dump_path+'/Final_Data_V2/error_avg_std_T'+str(T)+'_subject'+str(self.opt.leave_out)+'_'+str(model_key)+'seated.p', 'wb'))
                        #pkl.dump([self.error_avg_list, self.variance_est_list, self.targets_list, self.targets_est_list], open(self.dump_path+'/Feet_Variance/error_avg_std_T'+str(T)+'_subject'+str(self.opt.leave_out)+'_'+str(model_key)+'.p', 'wb'))

                if self.opt.visualize == False:
                    if model_key == 'KNN' or model_key == 'Ridge' or model_key == 'KRidge':
                        pkl.dump(error_norm, open(self.dump_path+'/Final_Data/error_avg_subject' + str(self.opt.leave_out) + '_'+str(model_key)+'seated.p', 'wb'))

                        #print batch_idx, np.array(self.error_std_list)[batch_idx, 6:10], np.array(self.leg_patch_sums)[
                     #                                                                batch_idx, :]
                    #pkl.dump([np.array(self.leg_patch_sums), np.array(self.error_std_list)[:,6:10]], open(
                    #    '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/Final_Data/sumRL_sumLL_stdKA_T25_subject' + str(
                    #        self.opt.leave_out) + '_kinvL.p', 'wb'))


        return




    def run(self):
        '''Runs either the synthetic database creation script or the
        raw dataset creation script to create a dataset'''
        for subject in [self.opt.leave_out]:

            self.opt.leave_out = subject
            self.init(self.opt.leave_out)
            self.validate()






if __name__ == "__main__":

    import optparse
    p = optparse.OptionParser()

    p.add_option('--leave_out', action='store', type=int, \
                 dest='leave_out', \
                 help='Specify which subject to leave out for validation')
    p.add_option('--computer', action='store', type = 'string',
                 dest='computer', \
                 default='lab_harddrive', \
                 help='Set path to the training database on lab harddrive.')
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
