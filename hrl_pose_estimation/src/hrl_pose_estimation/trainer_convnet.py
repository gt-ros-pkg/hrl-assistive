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
from sklearn.preprocessing import normalize
from sklearn import svm, linear_model, decomposition, kernel_ridge, neighbors
from sklearn import metrics, cross_validation
from sklearn.utils import shuffle

import convnet as convnet
import convnet_cascade as convnet_cascade
import tf.transformations as tft

import pickle
from hrl_lib.util import load_pickle

# Pose Estimation Libraries
from create_dataset_lib import CreateDatasetLib
from synthetic_lib import SyntheticLib
from visualization_lib import VisualizationLib
from kinematics_lib import KinematicsLib
from cascade_lib import CascadeLib
from preprocessing_lib import PreprocessingLib
 
#PyTorch libraries
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable



np.set_printoptions(threshold='nan')

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

 
class PhysicalTrainer():
    '''Gets the dictionary of pressure maps from the training database, 
    and will have API to do all sorts of training with it.'''
    def __init__(self, training_database_file, test_file, opt):
        '''Opens the specified pickle files to get the combined dataset:
        This dataset is a dictionary of pressure maps with the corresponding
        3d position and orientation of the markers associated with it.'''
        self.verbose = opt.verbose
        self.opt = opt
        self.batch_size = 115
        self.num_epochs = 100
        self.include_inter = True


        print test_file
        #Entire pressure dataset with coordinates in world frame

        if self.opt.upper_only == True:
            self.save_name = '_2to8_alldata_angles_' + str(self.batch_size) + 'b_' + str(self.num_epochs) + 'e_4'
            self.loss_vector_type = 'upper_angles'

        else:
            self.save_name = '_2to8_alldata_angles_implbedang_' + str(self.batch_size) + 'b_' + str(self.num_epochs) + 'e_4'
            self.loss_vector_type = 'angles'  # 'arms_cascade'#'upper_angles' #this is so you train the set to joint lengths and angles

        #we'll be loading this later
        if self.opt.computer == 'lab_harddrive':
            #try:
            self.train_val_losses_all = load_pickle('/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/train_val_losses_all.p')
        else:
            try:
                self.train_val_losses_all = load_pickle('/home/henryclever/hrl_file_server/Autobed/train_val_losses_all.p')
            except:
                print 'starting anew'

        if self.opt.quick_test == False:
            print 'appending to','train'+self.save_name+str(self.opt.leaveOut)
            self.train_val_losses_all['train'+self.save_name+str(self.opt.leaveOut)] = []
            self.train_val_losses_all['val'+self.save_name+str(self.opt.leaveOut)] = []
            self.train_val_losses_all['epoch'+self.save_name + str(self.opt.leaveOut)] = []




        # TODO:Write code for the dataset to store these vals
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





        #load in the training files.  This may take a while.
        for some_subject in training_database_file:
            print some_subject
            dat_curr = load_pickle(some_subject)
            for key in dat_curr:
                if np.array(dat_curr[key]).shape[0] != 0:
                    for inputgoalset in np.arange(len(dat_curr['images'])):
                        try:
                            dat[key].append(dat_curr[key][inputgoalset])
                        except:
                            try:
                                dat[key] = []
                                dat[key].append(dat_curr[key][inputgoalset])
                            except:
                                dat = {}
                                dat[key] = []
                                dat[key].append(dat_curr[key][inputgoalset])




        #create a tensor for our training dataset.  First print out how many input/output sets we have and what data we have
        for key in dat:
            print 'training set: ', key, np.array(dat[key]).shape

        self.train_x_flat = []  # Initialize the testing pressure mat list
        for entry in range(len(dat['images'])):
            self.train_x_flat.append(dat['images'][entry])
        # test_x = PreprocessingLib().preprocessing_pressure_array_resize(self.test_x_flat, self.mat_size, self.verbose)
        # test_x = np.array(test_x)

        self.train_a_flat = []  # Initialize the testing pressure mat angle list
        for entry in range(len(dat['images'])):
            self.train_a_flat.append(dat['bed_angle_deg'][entry])
        train_xa = PreprocessingLib().preprocessing_create_pressure_angle_stack(self.train_x_flat, self.train_a_flat, self.include_inter, self.mat_size, self.verbose)
        train_xa = np.array(train_xa)
        self.train_x_tensor = torch.Tensor(train_xa)

        self.train_y_flat = [] #Initialize the training ground truth list
        for entry in range(len(dat['images'])):
            if self.loss_vector_type == 'upper_angles':
                c = np.concatenate((dat['markers_xyz_m'][entry][0:18] * 1000,
                                    dat['joint_lengths_U_m'][entry][0:9] * 100,
                                    dat['joint_angles_U_deg'][entry][0:10]), axis=0)
                self.train_y_flat.append(c)
            elif self.loss_vector_type == 'arms_cascade':
                c = np.concatenate((dat['markers_xyz_m'][entry][0:30] * 1000,
                                    dat['joint_lengths_U_m'][entry][0:9] * 100,
                                    dat['joint_angles_U_deg'][entry][0:10],
                                    dat['joint_lengths_L_m'][entry][0:8] * 100,
                                    dat['joint_angles_L_deg'][entry][0:8],
                                    dat['pseudomarkers_xyz_m'][entry][:] * 1000), axis=0)
                self.train_y_flat.append(c)
            elif self.loss_vector_type == 'angles':
                c = np.concatenate((dat['markers_xyz_m'][entry][0:30] * 1000,
                                    dat['joint_lengths_U_m'][entry][0:9] * 100,
                                    dat['joint_angles_U_deg'][entry][0:10],
                                    dat['joint_lengths_L_m'][entry][0:8] * 100,
                                    dat['joint_angles_L_deg'][entry][0:8]), axis=0)
                self.train_y_flat.append(c)
            else:
                self.train_y_flat.append(dat['markers_xyz_m'][entry] * 1000)
        self.train_y_tensor = torch.Tensor(self.train_y_flat)



        #load in the test file
        test_dat = load_pickle(test_file)

        # create a tensor for our testing dataset.  First print out how many input/output sets we have and what data we have
        for key in test_dat:
            print 'testing set: ', key, np.array(test_dat[key]).shape

        self.test_x_flat = []  # Initialize the testing pressure mat list
        for entry in range(len(test_dat['images'])):
            self.test_x_flat.append(test_dat['images'][entry])
        # test_x = PreprocessingLib.preprocessing_pressure_array_resize(self.test_x_flat, self.mat_size, self.verbose)
        # test_x = np.array(test_x)

        self.test_a_flat = []  # Initialize the testing pressure mat angle list
        for entry in range(len(test_dat['images'])):
            self.test_a_flat.append(test_dat['bed_angle_deg'][entry])
        test_xa = PreprocessingLib().preprocessing_create_pressure_angle_stack(self.test_x_flat, self.test_a_flat, self.include_inter, self.mat_size, self.verbose)
        test_xa = np.array(test_xa)
        self.test_x_tensor = torch.Tensor(test_xa)

        self.test_y_flat = []  # Initialize the ground truth list
        for entry in range(len(test_dat['images'])):
            if self.loss_vector_type == 'upper_angles':
                c = np.concatenate((test_dat['markers_xyz_m'][entry][0:18] * 1000,
                                    test_dat['joint_lengths_U_m'][entry][0:9] * 100,
                                    test_dat['joint_angles_U_deg'][entry][0:10]), axis=0)
                self.test_y_flat.append(c)
            elif self.loss_vector_type == 'arms_cascade':
                c = np.concatenate((test_dat['markers_xyz_m'][entry][0:30] * 1000,
                                    test_dat['joint_lengths_U_m'][entry][0:9] * 100,
                                    test_dat['joint_angles_U_deg'][entry][0:10],
                                    test_dat['joint_lengths_L_m'][entry][0:8] * 100,
                                    test_dat['joint_angles_L_deg'][entry][0:8],
                                    test_dat['pseudomarkers_xyz_m'][entry][:] * 1000), axis=0)
                self.test_y_flat.append(c)
            elif self.loss_vector_type == 'angles':
                c = np.concatenate((test_dat['markers_xyz_m'][entry][0:30] * 1000,
                                    test_dat['joint_lengths_U_m'][entry][0:9] * 100,
                                    test_dat['joint_angles_U_deg'][entry][0:10],
                                    test_dat['joint_lengths_L_m'][entry][0:8] * 100,
                                    test_dat['joint_angles_L_deg'][entry][0:8]), axis=0)
                self.test_y_flat.append(c)
            elif self.loss_vector_type == 'direct':
                self.test_y_flat.append(test_dat['markers_xyz_m'][entry] * 1000)
            else:
                print "ERROR! SPECIFY A VALID LOSS VECTOR TYPE."
        self.test_y_flat = np.array(self.test_y_flat)
        self.test_y_tensor = torch.Tensor(self.test_y_flat)


    def baseline_train(self):
        n_neighbors = 5
        cv_fold = 3

        #for knn we don't really care about the variable function in pytorch, but it's a nice utility for shuffling the data.
        self.batch_size = self.train_y_tensor.numpy().shape[0]
        self.batchtest_size = 1#self.test_y_tensor.numpy().shape[0]


        self.train_dataset = torch.utils.data.TensorDataset(self.train_x_tensor, self.train_y_tensor)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, self.batch_size, shuffle=True)

        self.test_dataset = torch.utils.data.TensorDataset(self.test_x_tensor, self.test_y_tensor)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, self.batchtest_size, shuffle=True)

        for batch_idx, batch in enumerate(self.train_loader):
            batch[0], batch[1], _ = SyntheticLib().synthetic_master(batch[0], batch[1], flip=True,
                                                                           shift=True, scale=False,
                                                                           bedangle=True,
                                                                           include_inter=self.include_inter,
                                                                           loss_vector_type=self.loss_vector_type)


            images = batch[0].numpy()[:,0,:,:]
            targets = batch[1].numpy()


            #upsample the images
            images_up = PreprocessingLib().preprocessing_pressure_map_upsample(images)
            #targets = list(targets)
            print images_up[0].shape



            # Compute HoG of the current(training) pressure map dataset
            images_up = PreprocessingLib().compute_HoG(images_up)

            #images_up = [[0], [1], [2], [3]]
            #targets = [0, 0, 1, 1]

            print np.shape(images_up)
            print np.shape(targets)

            print 'fitting'

            baseline = 'Linear'
            if baseline == 'KNN':
                regr = neighbors.KNeighborsRegressor(15, weights='distance')

            elif baseline == 'kmeans_SVM':
                k_means = KMeans(n_clusters=10, n_init=4)
                k_means.fit(images_up)
                labels = k_means.labels_
                svm_classifier = svm.SVC()
                svm_classifier.fit(images_up, labels)
                regr = linear_model.LinearRegression()

            elif baseline == 'Ridge':
                regr = linear_model.Ridge(alpha=1.0)

            elif baseline == 'KRidge':
                regr = kernel_ridge.KernelRidge(alpha=1, kernel='chi2', gamma= 10)
            elif baseline == 'Linear':
                regr = linear_model.LinearRegression()

            regr.fit(images_up, targets)
            print 'done fitting'

            pkl.dump(regr, open('/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_' + str(self.opt.leaveOut) + '/p_files/HoG_Linear.p', 'wb'))

            #validation
            for batchtest_idx, batchtest in enumerate(self.test_loader):
                images_test = batchtest[0].numpy()[:, 0, :, :]
                targets = batchtest[1].numpy()

                images_up_test = PreprocessingLib().preprocessing_pressure_map_upsample(images_test)
                #targets = list(targets)

                images_up_test = PreprocessingLib().compute_HoG(images_up_test)
                scores = regr.predict(images_up_test)

                print scores.shape
                print targets.shape
                print scores[0]
                print targets[0]

                print regr.predict(images_up_test[0]) - targets[0]
                VisualizationLib().print_error(scores, targets, self.output_size, loss_vector_type=self.loss_vector_type, data='test', printerror=True)

                self.im_sample = np.squeeze(images_test[0, :])
                print self.im_sample.shape

                self.tar_sample = np.squeeze(targets[0, :]) / 1000
                self.sc_sample = np.copy(scores)
                self.sc_sample = np.squeeze(self.sc_sample[0, :]) / 1000
                self.sc_sample = np.reshape(self.sc_sample, self.output_size)
                VisualizationLib().visualize_pressure_map(self.im_sample, self.tar_sample, self.sc_sample, block=True)

            print len(scores)
            print scores[0].shape
            print scores.shape
            print targets.shape



    def init_convnet_train(self):
        #indices = torch.LongTensor([0])
        #self.train_y_tensor = torch.index_select(self.train_y_tensor, 1, indices)

        if self.verbose: print self.train_x_tensor.size(), 'size of the training database'
        if self.verbose: print self.train_y_tensor.size(), 'size of the training database output'
        print self.train_y_tensor
        if self.verbose: print self.test_x_tensor.size(), 'length of the testing dataset'
        if self.verbose: print self.test_y_tensor.size(), 'size of the training database output'



        num_epochs = self.num_epochs
        hidden_dim = 12
        kernel_size = 10



        #self.train_x_tensor = self.train_x_tensor.unsqueeze(1)
        self.train_dataset = torch.utils.data.TensorDataset(self.train_x_tensor, self.train_y_tensor)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, self.batch_size, shuffle=True)

        #self.test_x_tensor = self.test_x_tensor.unsqueeze(1)
        self.test_dataset = torch.utils.data.TensorDataset(self.test_x_tensor, self.test_y_tensor)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, self.batch_size, shuffle=True)


        output_size = self.output_size[0]*self.output_size[1]

        if self.loss_vector_type == 'upper_angles':
            fc_output_size = 22 #10 angles for arms and head, 9 lengths for arms and head, 3 torso coordinates
            self.model = convnet.CNN(self.mat_size, fc_output_size, hidden_dim, kernel_size, self.loss_vector_type)
        elif self.loss_vector_type == 'angles':
            fc_output_size = 40#38 #18 angles for body, 17 lengths for body, 3 torso coordinates
            self.model = convnet.CNN(self.mat_size, fc_output_size, hidden_dim, kernel_size, self.loss_vector_type)
        elif self.loss_vector_type == 'arms_cascade':
            #we'll make a double pass through this network for the validation for each arm.
            fc_output_size = 4 #4 angles for arms
            self.model = convnet_cascade.CNN(self.mat_size, fc_output_size, hidden_dim, kernel_size, self.loss_vector_type)
            self.model_cascade_prior = torch.load('/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_' + str(self.opt.leaveOut) + '/p_files/convnet_2to8_alldata_angles_constrained_noise_115b_100e_4.pt')
        elif self.loss_vector_type == 'direct':
            fc_output_size = 30
            self.model = convnet.CNN(self.mat_size, fc_output_size, hidden_dim, kernel_size, self.loss_vector_type)


        self.criterion = F.cross_entropy



        if self.loss_vector_type == None:
            self.optimizer2 = optim.Adam(self.model.parameters(), lr=0.000025, weight_decay=0.0005)
        elif self.loss_vector_type == 'upper_angles' or self.loss_vector_type == 'arms_cascade' or self.loss_vector_type == 'angles' or self.loss_vector_type == 'direct':
            self.optimizer2 = optim.Adam(self.model.parameters(), lr=0.00001, weight_decay=0.0005)  #0.000002 does not converge even after 100 epochs on subjects 2-8 kin cons. use .00001
        elif self.loss_vector_type == 'direct':
            self.optimizer2 = optim.Adam(self.model.parameters(), lr=0.00005, weight_decay=0.0005)
        #self.optimizer = optim.RMSprop(self.model.parameters(), lr=0.000001, momentum=0.7, weight_decay=0.0005)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.00005, weight_decay=0.0005) #start with .00005


        # train the model one epoch at a time
        for epoch in range(1, num_epochs + 1):
            self.t1 = time.time()

            self.train_convnet(epoch)

            if epoch > 5: self.optimizer = self.optimizer2

            try:
                self.t2 = time.time() - self.t1
            except:
                self.t2 = 0
            print 'Time taken by epoch',epoch,':',self.t2,' seconds'



        print 'done with epochs, now evaluating'
        self.validate_convnet('test')

        print self.train_val_losses_all, 'trainval'
        # Save the model (architecture and weights)

        if self.opt.quick_test == False:
            if self.opt.computer == 'lab_harddrive':
                torch.save(self.model, '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_'+str(self.opt.leaveOut)+'/p_files/convnet'+self.save_name+'.pt')
                pkl.dump(self.train_val_losses_all,
                         open(os.path.join('/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/train_val_losses_all.p'), 'wb'))

            else:
                torch.save(self.model, '/home/henryclever/hrl_file_server/Autobed/subject_'+str(self.opt.leaveOut)+'/p_files/convnet'+self.save_name+'.pt')
                pkl.dump(self.train_val_losses_all,
                         open(os.path.join('/home/henryclever/hrl_file_server/Autobed/train_val_losses_all.p'), 'wb'))


    def train_convnet(self, epoch):
        '''
        Train the model for one epoch.
        '''
        # Some models use slightly different forward passes and train and test
        # time (e.g., any model with Dropout). This puts the model in train mode
        # (as opposed to eval mode) so it knows which one to use.
        self.model.train()
        scores = 0


        #This will loop a total = training_images/batch_size times
        for batch_idx, batch in enumerate(self.train_loader):


            if self.loss_vector_type == 'upper_angles':

                #append upper joint angles, upper joint lengths, in that order
                batch.append(torch.cat((batch[1][:,27:37], batch[1][:, 18:27]), dim = 1))

                #get the upper body marker x y z
                batch[1] = batch[1][:, 0:18]

                batch[0], batch[1], batch[2]= SyntheticLib().synthetic_master(batch[0], batch[1], batch[2], flip=True, shift=True, scale=False,
                                                           bedangle=True,
                                                           include_inter=self.include_inter,
                                                           loss_vector_type=self.loss_vector_type)

                images, targets, constraints = Variable(batch[0], requires_grad = False), Variable(batch[1], requires_grad = False), Variable(batch[2], requires_grad = False)


                self.optimizer.zero_grad()

                ground_truth = np.zeros((batch[0].numpy().shape[0], 27)) #27 is 9 joint lengths and 18 joint locations for x y z
                ground_truth = Variable(torch.Tensor(ground_truth))
                ground_truth[:, 0:9] = constraints[:, 10:19]/100
                ground_truth[:, 9:27] = targets[:, 0:18]/1000


                scores_zeros = np.zeros((batch[0].numpy().shape[0], 15)) #15 is  6 euclidean errors and 9 joint lengths
                scores_zeros = Variable(torch.Tensor(scores_zeros))
                scores_zeros[:, 6:15] = constraints[:, 10:19]/100


                scores, targets_est, angles_est, _, _ = self.model.forward_kinematic_jacobian(images, targets, constraints)
                self.criterion = nn.MSELoss()
                loss = self.criterion(scores, scores_zeros)

            elif self.loss_vector_type == 'angles':

                # append upper joint angles, lower joint angles, upper joint lengths, lower joint lengths, in that order
                batch.append(torch.cat((batch[1][:,39:49], batch[1][:, 57:65], batch[1][:, 30:39], batch[1][:, 49:57]), dim = 1))

                #get the whole body x y z
                batch[1] = batch[1][:, 0:30]

                batch[0], batch[1], batch[2]= SyntheticLib().synthetic_master(batch[0], batch[1], batch[2], flip=True, shift=True, scale=False, bedangle=True, include_inter=self.include_inter, loss_vector_type=self.loss_vector_type)


                images_up = Variable(torch.Tensor(np.array(PreprocessingLib().preprocessing_pressure_map_upsample(batch[0].numpy()[:, :, 5:79, 5:42]))),requires_grad=False)
                images, targets, constraints = Variable(batch[0], requires_grad = False), Variable(batch[1], requires_grad = False), Variable(batch[2], requires_grad = False)


                self.optimizer.zero_grad()

                ground_truth = np.zeros((batch[0].numpy().shape[0], 47)) #47 is 17 joint lengths and 30 joint locations for x y z
                ground_truth = Variable(torch.Tensor(ground_truth))
                ground_truth[:, 0:17] = constraints[:, 18:35]/100
                ground_truth[:, 17:47] = targets[:, 0:30]/1000



                scores_zeros = np.zeros((batch[0].numpy().shape[0], 27)) #27 is  10 euclidean errors and 17 joint lengths
                scores_zeros = Variable(torch.Tensor(scores_zeros))
                scores_zeros[:, 10:27] = constraints[:, 18:35]/100


                scores, targets_est, angles_est, lengths_est, _ = self.model.forward_kinematic_jacobian(images_up, targets, constraints) # scores is a variable with 27 for 10 euclidean errors and 17 lengths in meters. targets est is a numpy array in mm.
                #print lengths_est[0,0:10], 'lengths est'
                #print batch[0][0,2,10,10], 'angle'

                #print scores_zeros[0, :]

                self.criterion = nn.MSELoss()
                loss = self.criterion(scores, scores_zeros)

            elif self.loss_vector_type == 'arms_cascade':

                # append upper joint angles, lower joint angles, upper joint lengths, lower joint lengths, in that order
                #print batch[1].size()
                batch.append(torch.cat((batch[1][:,39:49], batch[1][:, 57:65], batch[1][:, 30:39], batch[1][:, 49:57]), dim = 1))

                #get the torso, shoulder pseudotargets, and the arm targets for elbow and hand in that order
                batch[1] = torch.cat((batch[1][:, 0:30], batch[1][:, 65:80]), dim = 1)

                batch[0], batch[1], batch[2]= SyntheticLib().synthetic_master(batch[0], batch[1], batch[2], flip=True, shift=True, scale=False,
                                                           bedangle=True,
                                                           include_inter=self.include_inter,
                                                           loss_vector_type=self.loss_vector_type)

                images, targets, constraints = Variable(batch[0], requires_grad = False), Variable(batch[1], requires_grad = False), Variable(batch[2], requires_grad = False)


                self.optimizer.zero_grad()


                scores_zeros = np.zeros((batch[0].numpy().shape[0], 2)) #2 is 2 euclidean errors
                scores_zeros = Variable(torch.Tensor(scores_zeros))

                images_up = Variable(torch.Tensor(np.array(PreprocessingLib().preprocessing_pressure_map_upsample(batch[0].numpy()[:, :, 5:79, 5:42]))), requires_grad=False)
                prior_cascade = torch.cat((targets[:, 3:6], constraints[:, 18:26]/100), dim = 1) #just use this method to check, do a forward pass through the prior cascade when its trained
                ##print prior_cascade[0, :], '1st prior'

                _, targets_prior, angles_prior, lengths_prior, pseudotargets_prior = self.model_cascade_prior.forward_kinematic_jacobian(images_up, targets, forward_only=True)
                targets_prior = np.concatenate((targets_prior, pseudotargets_prior), axis = 1)
                #print targets_prior.shape

                #find the pressure mat coordinates where the projected markers lie
                targets_2D = CascadeLib().get_2D_projection(images.data.numpy(), np.reshape(targets.data.numpy(), (targets.size()[0], 15, 3)))
                targets_2D_prior = CascadeLib().get_2D_projection(images.data.numpy(), np.reshape(targets_prior, (targets.size()[0], 15, 3)))

                box_centers = np.round(targets_2D_prior[:, :, 0:2] / 28.6, 0)

                image_cascade = CascadeLib().generate_bounded_box_input(batch[0].numpy(), box_centers, limb = 'right_arm')
                image_cascade_up = PreprocessingLib().preprocessing_pressure_map_upsample(image_cascade)
                image_cascade_tensor = Variable(torch.Tensor(image_cascade_up), requires_grad = False)

                scores, targets_est, angles_est, lengths_est, _ = self.model.forward_kinematic_jacobian(image_cascade_tensor, targets, constraints, prior_cascade=prior_cascade, body_side = 'right')


                ground_truth = np.zeros((batch[0].numpy().shape[0], 6)) #6 is 6 joint locations for the x y z right side elbow and hand
                ground_truth = Variable(torch.Tensor(ground_truth))
                targets2 = targets#if we want to visualize the 3D targets using only x /y and not the normal projection on the mat, visualize this
                targets = torch.cat((targets[:, 6:9], targets[:, 12:15]), dim = 1)
                ground_truth[:, 0:6] = targets/1000.

                self.criterion = nn.MSELoss()

                loss = self.criterion(scores, scores_zeros)


            elif self.loss_vector_type == 'direct':

                batch[0],batch[1], _ = SyntheticLib().synthetic_master(batch[0], batch[1], flip=True, shift=True, scale=False, bedangle=True, include_inter = self.include_inter, loss_vector_type = self.loss_vector_type)

                images, targets, scores_zeros = Variable(batch[0], requires_grad = False), Variable(batch[1], requires_grad = False), Variable(torch.Tensor(np.zeros((batch[1].shape[0], batch[1].shape[1]/3))), requires_grad = False)

                self.optimizer.zero_grad()
                scores, targets_est = self.model.forward_direct(images, targets)

                self.criterion = nn.MSELoss()
                loss = self.criterion(scores/1000, scores_zeros/1000)

            #print loss.data.numpy() * 1000, 'loss'

            loss.backward()
            self.optimizer.step()
            loss *= 1000


            if batch_idx % opt.log_interval == 0:
                if self.loss_vector_type == 'upper_angles' or self.loss_vector_type == 'angles':
                    VisualizationLib().print_error(targets.data.numpy(), targets_est, self.output_size, self.loss_vector_type, data = 'train')
                    print angles_est[0, :], 'angles'
                    print batch[0][0,2,10,10], 'bed angle'
                    self.im_sample = images.data.numpy()
                    #self.im_sample = self.im_sample[:,0,:,:]
                    self.im_sample = np.squeeze(self.im_sample[0, :])
                    self.tar_sample = targets.data.numpy()
                    self.tar_sample = np.squeeze(self.tar_sample[0, :])/1000
                    self.sc_sample = np.copy(targets_est)
                    self.sc_sample = np.squeeze(self.sc_sample[0, :]) / 1000
                    self.sc_sample = np.reshape(self.sc_sample, self.output_size)

                elif self.loss_vector_type == 'direct':
                    VisualizationLib().print_error(targets.data.numpy(), targets_est, self.output_size, self.loss_vector_type, data='train')
                    self.im_sample = batch[0].numpy()
                    #self.im_sample = self.im_sample[:, 1, :, :]
                    self.im_sample = np.squeeze(self.im_sample[0, :])
                    self.tar_sample = targets.data.numpy()#batch[1].numpy()
                    self.tar_sample = np.squeeze(self.tar_sample[0, :]) / 1000
                    self.sc_sample = targets_est#.data.numpy()
                    self.sc_sample = np.squeeze(self.sc_sample[0, :]) / 1000
                    self.sc_sample = np.reshape(self.sc_sample, self.output_size)

                elif self.loss_vector_type == 'arms_cascade':
                    VisualizationLib().print_error(targets.data.numpy(), targets_est, self.output_size, self.loss_vector_type, data='train')

                    print angles_est[0, :], 'angles'
                    im_sample = np.squeeze(batch[0].numpy()[0,0, :])
                    tar_sample = targets_2D[0, :, :].flatten() / 1000#targets2.data.numpy()#batch[1].numpy()
                    tar_prior_sample = targets_2D_prior[0, :, :,].flatten() / 1000
                    self.full_im_tar_prior = []
                    self.full_im_tar_prior.append(im_sample)
                    self.full_im_tar_prior.append(tar_sample)
                    self.full_im_tar_prior.append(tar_prior_sample)

                    cascade_im_sample = np.concatenate((image_cascade[0:1, 0, :, :], image_cascade[0:1, 2, :, :], image_cascade[0:1, 4, :, :], image_cascade[0:1, 6, :, :]), axis = 0)
                    print cascade_im_sample.shape, 'cascade shape'

                    cascade_tar_sample = targets.data.numpy()[0, :]
                    cascade_sc_sample = targets_est[0, :]
                    self.cascade_im_tar_sc = []
                    self.cascade_im_tar_sc.append(cascade_im_sample)
                    self.cascade_im_tar_sc.append(cascade_tar_sample)
                    self.cascade_im_tar_sc.append(cascade_sc_sample)
                    self.cascade_im_tar_sc.append(box_centers)

                val_loss = self.validate_convnet(n_batches=4)
                train_loss = loss.data[0]
                examples_this_epoch = batch_idx * len(images)
                epoch_progress = 100. * batch_idx / len(self.train_loader)
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t'
                      'Train Loss: {:.6f}\tVal Loss: {:.6f}'.format(
                    epoch, examples_this_epoch, len(self.train_loader.dataset),
                    epoch_progress, train_loss, val_loss))


                print 'appending to alldata losses'
                if self.opt.quick_test == False:
                    self.train_val_losses_all['train'+self.save_name + str(self.opt.leaveOut)].append(train_loss)
                    self.train_val_losses_all['val'+self.save_name + str(self.opt.leaveOut)].append(val_loss)
                    self.train_val_losses_all['epoch'+self.save_name + str(self.opt.leaveOut)].append(epoch)




    def validate_convnet(self, verbose=False, n_batches=None):
        '''
        Compute loss on val or test data.
        '''
        #print 'eval', split


        self.model.eval()
        loss = 0.
        n_examples = 0
        for batch_i, batch in enumerate(self.test_loader):

            self.model.train()

            if self.loss_vector_type == 'upper_angles':

                #append upper joint angles, upper joint lengths, in that order
                batch.append(torch.cat((batch[1][:,27:37], batch[1][:, 18:27]), dim = 1))


                #get the direct joint locations
                batch[1] = batch[1][:, 0:18]

                images, targets, constraints = Variable(batch[0], volatile = True, requires_grad=False), Variable(batch[1],volatile = True, requires_grad=False), Variable(batch[2], volatile = True,  requires_grad=False)

                self.optimizer.zero_grad()

                ground_truth = np.zeros((batch[0].numpy().shape[0], 27))
                ground_truth = Variable(torch.Tensor(ground_truth))
                ground_truth[:, 0:9] = constraints[:, 10:19]/100
                ground_truth[:, 9:27] = targets[:, 0:18]/1000


                scores_zeros = np.zeros((batch[0].numpy().shape[0], 15))
                scores_zeros = Variable(torch.Tensor(scores_zeros))
                scores_zeros[:, 6:15] = constraints[:, 10:19]/100

                scores, targets_est, angles_est, _, _ = self.model.forward_kinematic_jacobian(images, targets, constraints)
                self.criterion = nn.MSELoss()


                loss = self.criterion(scores[:, 0:6], scores_zeros[:, 0:6])
                loss = loss.data[0]

            elif self.loss_vector_type == 'angles':

                #append upper joint angles, lower joint angles, upper joint lengths, lower joint lengths, in that order
                batch.append(torch.cat((batch[1][:,39:49], batch[1][:, 57:65], batch[1][:, 30:39], batch[1][:, 49:57]), dim = 1))

                #get the direct joint locations
                batch[1] = batch[1][:, 0:30]


                images_up = batch[0].numpy()
                images_up = images_up[:, :, 5:79, 5:42]
                images_up = PreprocessingLib().preprocessing_pressure_map_upsample(images_up)
                images_up = np.array(images_up)
                images_up = Variable(torch.Tensor(images_up), volatile = True, requires_grad = False)
                #print images_up.size()



                images, targets, constraints = Variable(batch[0], volatile = True, requires_grad=False), Variable(batch[1],volatile = True, requires_grad=False), Variable(batch[2], volatile = True, requires_grad=False)

                self.optimizer.zero_grad()


                ground_truth = np.zeros((batch[0].numpy().shape[0], 47))
                ground_truth = Variable(torch.Tensor(ground_truth))
                ground_truth[:, 0:17] = constraints[:, 18:35]/100
                ground_truth[:, 17:47] = targets[:, 0:30]/1000

                scores_zeros = np.zeros((batch[0].numpy().shape[0], 27))
                scores_zeros = Variable(torch.Tensor(scores_zeros))
                scores_zeros[:, 10:27] = constraints[:, 18:35]/100

                scores, targets_est, angles_est, _, _ = self.model.forward_kinematic_jacobian(images_up, targets, constraints)


                self.criterion = nn.MSELoss()
                loss = self.criterion(scores[:, 0:8], scores_zeros[:, 0:8])
                loss = loss.data[0]

            elif self.loss_vector_type == 'arms_cascade':

                #append upper joint angles, lower joint angles, upper joint lengths, lower joint lengths, in that order
                batch.append(torch.cat((batch[1][:,39:49], batch[1][:, 57:65], batch[1][:, 30:39], batch[1][:, 49:57]), dim = 1))

                # get the targets and pseudotargets
                batch[1] = torch.cat((batch[1][:, 0:30], batch[1][:, 65:80]), dim=1)

                images, targets, constraints = Variable(batch[0], volatile = True, requires_grad = False), Variable(batch[1], volatile = True, requires_grad = False), Variable(batch[2], volatile = True, requires_grad = False)

                self.optimizer.zero_grad()

                scores_zeros = np.zeros((batch[0].numpy().shape[0], 2)) #2 is 2 euclidean errors
                scores_zeros = Variable(torch.Tensor(scores_zeros))

                prior_cascade = torch.cat((targets[:, 3:6], constraints[:, 18:26]/100), dim = 1) #just use this method to check, do a forward pass through the prior cascade when its trained

                images_up = Variable(torch.Tensor(np.array(PreprocessingLib().preprocessing_pressure_map_upsample(batch[0].numpy()[:, :, 5:79, 5:42]))),requires_grad=False)
                _, targets_prior, angles_prior, lengths_prior, pseudotargets_prior = self.model_cascade_prior.forward_kinematic_jacobian(images_up, targets, forward_only=True)
                targets_prior = np.concatenate((targets_prior, pseudotargets_prior), axis=1)

                # find the pressure mat coordinates where the projected markers lie
                targets_2D = CascadeLib().get_2D_projection(images.data.numpy(), np.reshape(targets.data.numpy(),(targets.size()[0], 15, 3)))
                targets_2D_prior = CascadeLib().get_2D_projection(images.data.numpy(), np.reshape(targets_prior, (targets.size()[0], 15, 3)))

                box_centers = np.round(targets_2D[:, :, 0:2] / 28.6, 0)
                image_cascade_right = CascadeLib().generate_bounded_box_input(batch[0].numpy(), box_centers, limb = 'right_arm')
                image_cascade_right_up = PreprocessingLib().preprocessing_pressure_map_upsample(image_cascade_right)
                image_cascade_right_tensor = Variable(torch.Tensor(image_cascade_right_up), requires_grad=False)




                scores, targets_est, angles_est, _, _ = self.model.forward_kinematic_jacobian(image_cascade_right_tensor, targets, constraints, prior_cascade=prior_cascade, forward_only= True, body_side = 'right')


                ground_truth = np.zeros((batch[0].numpy().shape[0], 6)) #6 is 6 joint locations for the x y z right side elbow and hand
                ground_truth = Variable(torch.Tensor(ground_truth))
                targets2 = targets #if we want to visualize the 3D targets using only x /y and not the normal projection on the mat, visualize this
                targets = torch.cat((targets[:, 6:9], targets[:, 12:15]), dim = 1)
                ground_truth[:, 0:6] = targets/1000.

                self.criterion = nn.MSELoss()
                loss = self.criterion(scores[:, 0:2], scores_zeros[:, 0:2])
                loss = loss.data[0]



            elif self.loss_vector_type == 'direct':
                images, targets, scores_zeros = Variable(batch[0], volatile = True), Variable(batch[1], volatile = True), Variable(torch.Tensor(np.zeros((batch[1].shape[0], batch[1].shape[1]/3))), requires_grad = False)


                scores, targets_est = self.model.forward_direct(images, targets)

                loss = self.criterion(scores/1000., scores_zeros/1000.)
                loss = loss.data[0]


            n_examples += self.batch_size
            #print n_examples

            if n_batches and (batch_i >= n_batches):
                break

        loss /= n_examples
        loss *= 100
        loss *= 1000


        VisualizationLib().print_error(targets.data.numpy(), targets_est, self.output_size, self.loss_vector_type, data='validate')

        if self.loss_vector_type == 'angles' or self.loss_vector_type == 'direct' or self.loss_vector_type == 'upper_angles':
            print angles_est[0, :], 'angles'
            print batch[0][0,2,10,10], 'bed angle'
            self.im_sampleval = images.data.numpy()
            #self.im_sampleval = self.im_sampleval[:,0,:,:]
            self.im_sampleval = np.squeeze(self.im_sampleval[0, :])
            self.tar_sampleval = targets.data.numpy()
            self.tar_sampleval = np.squeeze(self.tar_sampleval[0, :]) / 1000
            self.sc_sampleval = np.copy(targets_est)
            self.sc_sampleval = np.squeeze(self.sc_sampleval[0, :]) / 1000
            self.sc_sampleval = np.reshape(self.sc_sampleval, self.output_size)

            #VisualizationLib().visualize_pressure_map(self.im_sample, self.tar_sample, self.sc_sample,self.im_sampleval, self.tar_sampleval, self.sc_sampleval, block=False)

        elif self.loss_vector_type == 'arms_cascade':
            im_sample = np.squeeze(batch[0].numpy()[0, :])
            tar_sample = targets_2D[0, :, :].flatten() / 1000  # targets2.data.numpy()#batch[1].numpy()
            tar_prior_sample = targets_2D_prior[0, :, :, ].flatten() / 1000
            #print im_sample.shape, tar_sample.shape, tar_prior_sample.shape
            self.full_im_tar_prior_val = []
            self.full_im_tar_prior_val.append(im_sample)
            self.full_im_tar_prior_val.append(tar_sample)
            self.full_im_tar_prior_val.append(tar_prior_sample)

            cascade_im_sample = image_cascade_right[0, :, :, :]
            cascade_tar_sample = targets.data.numpy()[0, :]
            cascade_sc_sample = targets_est[0, :]
            #print cascade_im_sample.shape, cascade_tar_sample.shape, cascade_sc_sample.shape
            self.cascade_im_tar_sc_val = []
            self.cascade_im_tar_sc_val.append(cascade_im_sample)
            self.cascade_im_tar_sc_val.append(cascade_tar_sample)
            self.cascade_im_tar_sc_val.append(cascade_sc_sample)

           # VisualizationLib().visualize_pressure_map_cascade(self.full_im_tar_prior, self.cascade_im_tar_sc, block = True)




        #if verbose:
        #    print('\n{} set: Average loss: {:.4f}\n'.format(
        #        split, loss))
        return loss



if __name__ == "__main__":
    #Initialize trainer with a training database file
    import optparse
    p = optparse.OptionParser()
    p.add_option('--training_dataset', '--train_dataset',  action='store', type='string', \
                 dest='trainPath',\
                 default='/home/henryclever/hrl_file_server/Autobed/pose_estimation_data/basic_train_dataset.p', \
                 help='Specify path to the training database.')
    p.add_option('--leave_out', action='store', type=int, \
                 dest='leaveOut', \
                 help='Specify which subject to leave out for validation')
    p.add_option('--only_test','--t',  action='store_true', dest='only_test',
                 default=False, help='Whether you want only testing of previously stored model')
    p.add_option('--training_model', '--model',  action='store', type='string', \
                 dest='modelPath',\
                 default = '/home/henryclever/hrl_file_server/Autobed/pose_estimation_data', \
                 help='Specify path to the trained model')
    p.add_option('--testing_dataset', '--test_dataset',  action='store', type='string', \
                 dest='testPath',\
                 default='/home/henryclever/hrl_file_server/Autobed/pose_estimation_data/basic_test_dataset.p', \
                 help='Specify path to the training database.')
    p.add_option('--lab_hd', action='store', type = 'string',
                 dest='computer', \
                 default='lab_harddrive', \
                 help='Set path to the training database on lab harddrive.')
    p.add_option('--upper_only', action='store_true',
                 dest='upper_only', \
                 default=False, \
                 help='Train only on data from the arms, both sitting and laying.')
    p.add_option('--qt', action='store_true',
                 dest='quick_test', \
                 default=False, \
                 help='Train only on data from the arms, both sitting and laying.')

    p.add_option('--verbose', '--v',  action='store_true', dest='verbose',
                 default=False, help='Printout everything (under construction).')
    p.add_option('--log_interval', type=int, default=10, metavar='N',
                        help='number of batches between logging train status')

    opt, args = p.parse_args()

    if opt.computer == 'lab_harddrive':

        opt.subject2Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_2/p_files/trainval_150rh1_lh1_rl_ll_100rh23_lh23_sit120rh_lh_rl_ll.p'
        opt.subject3Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_3/p_files/trainval_150rh1_lh1_rl_ll_100rh23_lh23_sit120rh_lh_rl_ll.p'
        opt.subject4Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_4/p_files/trainval_150rh1_lh1_rl_ll_100rh23_lh23_sit120rh_lh_rl_ll.p'
        opt.subject5Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_5/p_files/trainval_150rh1_lh1_rl_ll_100rh23_lh23_sit120rh_lh_rl_ll.p'
        opt.subject6Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_6/p_files/trainval_150rh1_lh1_rl_ll_100rh23_lh23_sit120rh_lh_rl_ll.p'
        opt.subject7Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_7/p_files/trainval_150rh1_lh1_rl_ll_100rh23_lh23_sit120rh_lh_rl_ll.p'
        opt.subject8Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_8/p_files/trainval_150rh1_lh1_rl_ll_100rh23_lh23_sit120rh_lh_rl_ll.p'
        opt.subject9Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_9/p_files/trainval_150rh1_lh1_rl_ll_100rh23_lh23_sit120rh_lh_rl_ll.p'
        opt.subject10Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_10/p_files/trainval_150rh1_lh1_rl_ll_100rh23_lh23_sit120rh_lh_rl_ll.p'
        opt.subject11Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_11/p_files/trainval_150rh1_lh1_rl_ll_100rh23_lh23_sit120rh_lh_rl_ll.p'
        opt.subject12Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_12/p_files/trainval_150rh1_lh1_rl_ll_100rh23_lh23_sit120rh_lh_rl_ll.p'
        opt.subject13Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_13/p_files/trainval_150rh1_lh1_rl_ll_100rh23_lh23_sit120rh_lh_rl_ll.p'
        opt.subject14Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_14/p_files/trainval_150rh1_lh1_rl_ll_100rh23_lh23_sit120rh_lh_rl_ll.p'
        opt.subject15Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_15/p_files/trainval_150rh1_lh1_rl_ll_100rh23_lh23_sit120rh_lh_rl_ll.p'
        opt.subject16Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_16/p_files/trainval_150rh1_lh1_rl_ll_100rh23_lh23_sit120rh_lh_rl_ll.p'
        opt.subject17Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_17/p_files/trainval_150rh1_lh1_rl_ll_100rh23_lh23_sit120rh_lh_rl_ll.p'
        opt.subject18Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_18/p_files/trainval_150rh1_lh1_rl_ll_100rh23_lh23_sit120rh_lh_rl_ll.p'

        #shortcut:

        if opt.quick_test == True:
            opt.subject4Path = '/home/henryclever/test/trainval4_150rh1_sit120rh.p'
            opt.subject8Path = '/home/henryclever/test/trainval8_150rh1_sit120rh.p'



        training_database_file = []
    else:

        opt.subject2Path = '/home/henryclever/hrl_file_server/Autobed/subject_2/p_files/trainval_200rh1_lh1_rl_ll_100rh23_lh23_head_sit120rh_lh_rl_ll.p'
        opt.subject3Path = '/home/henryclever/hrl_file_server/Autobed/subject_3/p_files/trainval_200rh1_lh1_rl_ll_100rh23_lh23_head_sit120rh_lh_rl_ll.p'
        opt.subject4Path = '/home/henryclever/hrl_file_server/Autobed/subject_4/p_files/trainval_200rh1_lh1_rl_ll_100rh23_lh23_head_sit120rh_lh_rl_ll.p'
        opt.subject5Path = '/home/henryclever/hrl_file_server/Autobed/subject_5/p_files/trainval_200rh1_lh1_rl_ll_100rh23_lh23_head_sit120rh_lh_rl_ll.p'
        opt.subject6Path = '/home/henryclever/hrl_file_server/Autobed/subject_6/p_files/trainval_200rh1_lh1_rl_ll_100rh23_lh23_head_sit120rh_lh_rl_ll.p'
        opt.subject7Path = '/home/henryclever/hrl_file_server/Autobed/subject_7/p_files/trainval_200rh1_lh1_rl_ll_100rh23_lh23_head_sit120rh_lh_rl_ll.p'
        opt.subject8Path = '/home/henryclever/hrl_file_server/Autobed/subject_8/p_files/trainval_200rh1_lh1_rl_ll_100rh23_lh23_head_sit120rh_lh_rl_ll.p'
        opt.subject9Path = '/home/henryclever/hrl_file_server/Autobed/subject_9/p_files/trainval_200rh1_lh1_rl_ll_100rh23_lh23_head_sit120rh_lh_rl_ll.p'
        opt.subject10Path = '/home/henryclever/hrl_file_server/Autobed/subject_10/p_files/trainval_200rh1_lh1_rl_ll_100rh23_lh23_head_sit120rh_lh_rl_ll.p'
        opt.subject11Path = '/home/henryclever/hrl_file_server/Autobed/subject_11/p_files/trainval_200rh1_lh1_rl_ll_100rh23_lh23_head_sit120rh_lh_rl_ll.p'
        opt.subject12Path = '/home/henryclever/hrl_file_server/Autobed/subject_12/p_files/trainval_200rh1_lh1_rl_ll_100rh23_lh23_head_sit120rh_lh_rl_ll.p'
        opt.subject13Path = '/home/henryclever/hrl_file_server/Autobed/subject_13/p_files/trainval_200rh1_lh1_rl_ll_100rh23_lh23_head_sit120rh_lh_rl_ll.p'
        opt.subject14Path = '/home/henryclever/hrl_file_server/Autobed/subject_14/p_files/trainval_200rh1_lh1_rl_ll_100rh23_lh23_head_sit120rh_lh_rl_ll.p'
        opt.subject15Path = '/home/henryclever/hrl_file_server/Autobed/subject_15/p_files/trainval_200rh1_lh1_rl_ll_100rh23_lh23_head_sit120rh_lh_rl_ll.p'
        opt.subject16Path = '/home/henryclever/hrl_file_server/Autobed/subject_16/p_files/trainval_200rh1_lh1_rl_ll_100rh23_lh23_head_sit120rh_lh_rl_ll.p'
        opt.subject17Path = '/home/henryclever/hrl_file_server/Autobed/subject_17/p_files/trainval_200rh1_lh1_rl_ll_100rh23_lh23_head_sit120rh_lh_rl_ll.p'
        opt.subject18Path = '/home/henryclever/hrl_file_server/Autobed/subject_18/p_files/trainval_200rh1_lh1_rl_ll_100rh23_lh23_head_sit120rh_lh_rl_ll.p'

        training_database_file = []






    if opt.leaveOut == 4:
        test_database_file = opt.subject4Path
        #training_database_file.append(opt.subject1Path)
        if opt.quick_test == True:
            training_database_file.append(opt.subject8Path)
        else:
            training_database_file.append(opt.subject2Path)
            training_database_file.append(opt.subject3Path)
            training_database_file.append(opt.subject5Path)
            training_database_file.append(opt.subject6Path)
            training_database_file.append(opt.subject7Path)
            training_database_file.append(opt.subject8Path)
            #training_database_file.append(opt.subject9Path)
            #training_database_file.append(opt.subject10Path)
            #training_database_file.append(opt.subject11Path)
            #training_database_file.append(opt.subject12Path)
            #training_database_file.append(opt.subject13Path)
            ##training_database_file.append(opt.subject14Path)
            #training_database_file.append(opt.subject15Path)
            #training_database_file.append(opt.subject16Path)
            #training_database_file.append(opt.subject17Path)
            #training_database_file.append(opt.subject18Path)

    elif opt.leaveOut == 1:
        test_database_file = opt.subject1Path
        training_database_file.append(opt.subject2Path)
        training_database_file.append(opt.subject3Path)
        training_database_file.append(opt.subject4Path)
        training_database_file.append(opt.subject5Path)
        training_database_file.append(opt.subject6Path)
        training_database_file.append(opt.subject7Path)
        training_database_file.append(opt.subject8Path)

    elif opt.leaveOut == 2:
        test_database_file = opt.subject2Path
        training_database_file.append(opt.subject1Path)
        training_database_file.append(opt.subject3Path)
        training_database_file.append(opt.subject4Path)
        training_database_file.append(opt.subject5Path)
        training_database_file.append(opt.subject6Path)
        training_database_file.append(opt.subject7Path)
        training_database_file.append(opt.subject8Path)

    elif opt.leaveOut == 10:
        test_database_file = opt.subject10Path
        training_database_file.append(opt.subject9Path)
        training_database_file.append(opt.subject11Path)
        training_database_file.append(opt.subject12Path)
        training_database_file.append(opt.subject13Path)
        training_database_file.append(opt.subject14Path)
        training_database_file.append(opt.subject15Path)
        training_database_file.append(opt.subject16Path)
        training_database_file.append(opt.subject17Path)
        training_database_file.append(opt.subject18Path)

    else:
        print 'please specify which subject to leave out for validation using --leave_out _'



    print opt.testPath, 'testpath'
    print opt.modelPath, 'modelpath'



    test_bool = opt.only_test#Whether you want only testing done


    print test_bool, 'test_bool'
    print test_database_file, 'test database file'

    p = PhysicalTrainer(training_database_file, test_database_file, opt)

    if test_bool == True:
        trained_model = load_pickle(opt.modelPath+'/'+training_type+'.p')#Where the trained model is 
        p.test_learning_algorithm(trained_model)
        sys.exit()
    else:
        if opt.verbose == True: print 'Beginning Learning'



        #if training_type == 'convnet_2':
        p.init_convnet_train()
        #p.baseline_train()

        #else:
        #    print 'Please specify correct training type:1. HoG_KNN 2. convnet_2'
