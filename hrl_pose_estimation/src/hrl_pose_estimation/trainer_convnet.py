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

import convnet as convnet_armsonly
import tf.transformations as tft

import pickle
from hrl_lib.util import load_pickle

# Pose Estimation Libraries
from create_dataset_lib import CreateDatasetLib
from synthetic_lib import SyntheticLib
from visualization_lib import VisualizationLib
from kinematics_lib import KinematicsLib
 
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
        self.loss_vector_type = None#'torso_lengths'#'arm_angles' #this is so you train the set to joint lengths and angles

        print test_file
        #Entire pressure dataset with coordinates in world frame

        if self.opt.arms_only == True:
            self.save_name = '_2to8_alldata_armsonly_direct_' + str(self.batch_size) + 'b_adam_' + str(self.num_epochs) + 'e_4'

        else:
            self.save_name = '_2to8_fss_'+str(self.batch_size)+'b_adam_'+str(self.num_epochs)+'e_'




        #we'll be loading this later
        if self.opt.lab_harddrive == True:
            #try:
            self.train_val_losses_all = load_pickle('/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/train_val_losses_all.p')
        else:
            try:
                self.train_val_losses_all = load_pickle('/home/henryclever/hrl_file_server/Autobed/train_val_losses_all.p')
            except:
                print 'starting anew'


        print 'appending to','train'+self.save_name+str(self.opt.leaveOut)
        self.train_val_losses_all['train'+self.save_name+str(self.opt.leaveOut)] = []
        self.train_val_losses_all['val'+self.save_name+str(self.opt.leaveOut)] = []
        self.train_val_losses_all['epoch'+self.save_name + str(self.opt.leaveOut)] = []




        # TODO:Write code for the dataset to store these vals
        self.mat_size = (NUMOFTAXELS_X, NUMOFTAXELS_Y)
        if self.loss_vector_type == 'torso_lengths':
            self.output_size = (NUMOFOUTPUTNODES - 9, NUMOFOUTPUTDIMS)
        elif self.loss_vector_type == 'arm_angles':
            self.output_size = (NUMOFOUTPUTNODES - 5, NUMOFOUTPUTDIMS)
        elif self.loss_vector_type == 'euclidean_error':
            self.output_size = (NUMOFOUTPUTNODES - 5, NUMOFOUTPUTDIMS)
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
        # test_x = self.preprocessing_pressure_array_resize(self.test_x_flat)
        # test_x = np.array(test_x)

        self.train_a_flat = []  # Initialize the testing pressure mat angle list
        for entry in range(len(dat['images'])):
            self.train_a_flat.append(dat['bed_angle_deg'][entry])
        train_xa = self.preprocessing_create_pressure_angle_stack(self.train_x_flat, self.train_a_flat)
        train_xa = np.array(train_xa)
        self.train_x_tensor = torch.Tensor(train_xa)

        self.train_y_flat = [] #Initialize the training ground truth list
        for entry in range(len(dat['images'])):
            if self.opt.arms_only == True:
                c = np.concatenate((dat['markers_xyz_m'][entry][6:18] * 1000,
                                    dat['joint_lengths_U_m'][entry] * 100,
                                    dat['joint_angles_U_deg'][entry],
                                    dat['markers_xyz_m'][entry][3:6] * 100), axis=0)
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
        # test_x = self.preprocessing_pressure_array_resize(self.test_x_flat)
        # test_x = np.array(test_x)

        self.test_a_flat = []  # Initialize the testing pressure mat angle list
        for entry in range(len(test_dat['images'])):
            self.test_a_flat.append(test_dat['bed_angle_deg'][entry])
        test_xa = self.preprocessing_create_pressure_angle_stack(self.test_x_flat, self.test_a_flat)
        test_xa = np.array(test_xa)
        self.test_x_tensor = torch.Tensor(test_xa)

        self.test_y_flat = []  # Initialize the ground truth list
        for entry in range(len(test_dat['images'])):
            if self.opt.arms_only == True:
                c = np.concatenate((test_dat['markers_xyz_m'][entry][6:18] * 1000,
                                    test_dat['joint_lengths_U_m'][entry] * 100,
                                    test_dat['joint_angles_U_deg'][entry],
                                    test_dat['markers_xyz_m'][entry][3:6] * 100), axis=0)
                self.test_y_flat.append(c)
            else:
                self.test_y_flat.append(test_dat['markers_xyz_m'][entry] * 1000)
        self.test_y_flat = np.array(self.test_y_flat)
        self.test_y_tensor = torch.Tensor(self.test_y_flat)


    def preprocessing_pressure_array_resize(self, data):
        '''Will resize all elements of the dataset into the dimensions of the 
        pressure map'''
        p_map_dataset = []
        for map_index in range(len(data)):
            #print map_index, self.mat_size, 'mapidx'
            #Resize mat to make into a matrix
            p_map = np.reshape(data[map_index], self.mat_size)
            #print p_map
            p_map_dataset.append(p_map)
            #print p_map.shape
        if self.verbose: print len(data[0]),'x',1, 'size of an incoming pressure map'
        if self.verbose: print len(p_map_dataset[0]),'x',len(p_map_dataset[0][0]), 'size of a resized pressure map'
        return p_map_dataset


    def preprocessing_create_pressure_angle_stack(self,x_data,a_data):
        '''This is for creating a 2-channel input using the height of the bed. '''
        p_map_dataset = []
        for map_index in range(len(x_data)):
            # print map_index, self.mat_size, 'mapidx'
            # Resize mat to make into a matrix
            p_map = np.reshape(x_data[map_index], self.mat_size)
            a_map = zeros_like(p_map) + a_data[map_index]
            if self.include_inter == True:
                p_map_inter = (100-2*np.abs(p_map - 50))*4
                p_map_dataset.append([p_map, p_map_inter, a_map])
            else:
                p_map_dataset.append([p_map, a_map])
        if self.verbose: print len(x_data[0]), 'x', 1, 'size of an incoming pressure map'
        if self.verbose: print len(p_map_dataset[0][0]), 'x', len(p_map_dataset[0][0][0]), 'size of a resized pressure map'
        if self.verbose: print len(p_map_dataset[0][1]), 'x', len(p_map_dataset[0][1][0]), 'size of the stacked angle mat'
        return p_map_dataset



    def convnet_2layer(self):
        #indices = torch.LongTensor([0])
        #self.train_y_tensor = torch.index_select(self.train_y_tensor, 1, indices)

        if self.verbose: print self.train_x_tensor.size(), 'size of the training database'
        if self.verbose: print self.train_y_tensor.size(), 'size of the training database output'
        print self.train_y_tensor
        if self.verbose: print self.test_x_tensor.size(), 'length of the testing dataset'
        if self.verbose: print self.test_y_tensor.size(), 'size of the training database output'



        batch_size = self.batch_size
        num_epochs = self.num_epochs
        hidden_dim = 12
        kernel_size = 10



        #self.train_x_tensor = self.train_x_tensor.unsqueeze(1)
        self.train_dataset = torch.utils.data.TensorDataset(self.train_x_tensor, self.train_y_tensor)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size, shuffle=True)

        #self.test_x_tensor = self.test_x_tensor.unsqueeze(1)
        self.test_dataset = torch.utils.data.TensorDataset(self.test_x_tensor, self.test_y_tensor)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size, shuffle=True)


        output_size = self.output_size[0]*self.output_size[1]
        if self.opt.arms_only == True:
            if self.loss_vector_type == 'torso_lengths':
                output_size = 11
                self.model = convnet_armsonly.CNN(self.mat_size, output_size, hidden_dim, kernel_size)
            elif self.loss_vector_type == 'arm_angles':
                output_size = 8
                self.model = convnet_armsonly.CNN(self.mat_size, output_size, hidden_dim, kernel_size)
                self.model_torso_lengths = torch.load('/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_' + str(self.opt.leaveOut) + '/p_files/convnet_2to8_alldata_armsonly_torso_lengths_115b_adam_100e_4.pt')
            elif self.loss_vector_type == 'euclidean_error':
                output_size = 5
                self.model = convnet_armsonly.CNN(self.mat_size, output_size, hidden_dim, kernel_size)
                self.model_direct_marker = torch.load('/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_' + str(self.opt.leaveOut) + '/p_files/convnet_2to8_alldata_armsonly_direct_inclinter_115b_adam_200e_4.pt')
            elif self.loss_vector_type == None:
                output_size = 15
                self.model = convnet_armsonly.CNN(self.mat_size, output_size, hidden_dim, kernel_size)
        else:
            self.model = convnet.CNN(self.mat_size, output_size, hidden_dim, kernel_size)
        self.criterion = F.cross_entropy



        if self.loss_vector_type == None or self.loss_vector_type == 'euclidean_error':
            self.optimizer2 = optim.Adam(self.model.parameters(), lr=0.000025, weight_decay=0.0005)
        elif self.loss_vector_type == 'arm_angles' or self.loss_vector_type == 'torso_lengths':
            self.optimizer2 = optim.Adam(self.model.parameters(), lr=0.000004, weight_decay=0.0005)
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=0.000015, momentum=0.7, weight_decay=0.0005)



        # train the model one epoch at a time
        for epoch in range(1, num_epochs + 1):
            self.t1 = time.time()

            self.train(epoch)

            if epoch > 5: self.optimizer = self.optimizer2

            try:
                self.t2 = time.time() - self.t1
            except:
                self.t2 = 0
            print 'Time taken by epoch',epoch,':',self.t2,' seconds'



        print 'done with epochs, now evaluating'
        self.evaluate('test', verbose=True)

        print self.train_val_losses_all, 'trainval'
        # Save the model (architecture and weights)

        if self.opt.lab_harddrive == True:
            torch.save(self.model, '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_'+str(self.opt.leaveOut)+'/p_files/convnet'+self.save_name+'.pt')
            pkl.dump(self.train_val_losses_all,
                     open(os.path.join('/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/train_val_losses_all.p'), 'wb'))

        else:
            torch.save(self.model, '/home/henryclever/hrl_file_server/Autobed/subject_'+str(self.opt.leaveOut)+'/p_files/convnet'+self.save_name+'.pt')
            pkl.dump(self.train_val_losses_all,
                     open(os.path.join('/home/henryclever/hrl_file_server/Autobed/train_val_losses_all.p'), 'wb'))


    def train(self, epoch):
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

            if self.loss_vector_type == 'torso_lengths':
                batch.append(batch[1][:, 12:28])  # get the constraints we'll be training on, this makes for batch[2]

                batch[1] = torch.cat((torch.mul(batch[1][:, 28:31], 10), batch[1][:, 0:12]), dim=1) #direct joints

                #need to feed into synthetic in the following order: input images, direct joints, kinematic constraints.
                batch[0], batch[1], batch[2] = SyntheticLib().synthetic_master(batch[0], batch[1], batch[2], flip=True,
                                                                     shift=True, scale=False,
                                                                     bedangle=True, arms_only=self.opt.arms_only,
                                                                     include_inter=self.include_inter,
                                                                     p_cons=self.loss_vector_type)

                #now make the new set we'll be taking the loss over, in this case, the torso position and joint lengths
                batch[2] = torch.cat((torch.mul(batch[1][:, 0:3], 0.1), batch[2][:, 0:8]), dim=1)

                images, targets, torso_lengths = Variable(batch[0]), Variable(batch[1]), Variable(batch[2])

                self.optimizer.zero_grad()
                torso_lengths_scores = self.model(images)

                self.criterion = nn.MSELoss()
                loss = self.criterion(torso_lengths_scores, torso_lengths)


            elif self.loss_vector_type == 'arm_angles':

                batch.append(batch[1][:, 12:28]) #get the constraints we'll be training on, this makes for batch[2]

                #combine the torso marker with the elbow and hand direct joint markers
                batch[1] = torch.cat((torch.mul(batch[1][:,28:31],10),batch[1][:, 0:12]), dim = 1)
                #print batch[1].shape

                batch[0], batch[1], batch[2]= SyntheticLib().synthetic_master(batch[0], batch[1], batch[2], flip=True, shift=True, scale=False,
                                                           bedangle=True, arms_only=self.opt.arms_only,
                                                           include_inter=self.include_inter,
                                                           p_cons=self.loss_vector_type)

                #print batch[2].shape
                batch[2] = batch[2][:, 8:16]
                #print batch[2].shape
                batch[2] = batch[2].numpy()
                #print batch[2][0,:]
                batch[2][:,2:4] = batch[2][:,2:4] * 4
                #print batch[2][0,:]
                batch[2] = torch.Tensor(batch[2])


                images, targets, constraints = Variable(batch[0]), Variable(batch[1]), Variable(batch[2])
                self.optimizer.zero_grad()
                # print images.size(), 'im size'
                # print targets.size(), 'target size'
                constraint_scores = self.model(images)
                # print constraint_scores.size(), 'constraint scores'
                # print scores-sc_last

                torso_lengths_scores = self.model_torso_lengths(images)
                #print torso_lengths_scores.data.shape, 'torso length scores'
                #print torso_lengths_scores.data.numpy()[0,:]
                #print constraints.data.shape
                #print constraint_scores.data.shape, 'constraint shape'

                self.criterion = nn.MSELoss()
                loss = self.criterion(constraint_scores, constraints)

            elif self.loss_vector_type == 'euclidean_error':

                batch[1] = torch.cat((torch.mul(batch[1][:, 28:31], 10), batch[1][:, 0:12]), dim=1)
                batch[0],batch[1], _ = SyntheticLib().synthetic_master(batch[0], batch[1], flip=True, shift=True, scale=False, bedangle=True, arms_only = self.opt.arms_only, include_inter = self.include_inter, p_cons = self.loss_vector_type)

                images, targets = Variable(batch[0], volatile = True), Variable(batch[1], volatile = True)
                self.optimizer.zero_grad()

                scores = self.model_direct_marker(images)

                images = Variable(batch[0], volatile = False)
                errors_scores = self.model(images)


                errors = VisualizationLib().print_error(targets, scores, self.output_size, self.loss_vector_type, data = 'train', printerror = False)
                errors = Variable(torch.Tensor(errors), volatile = False)

                self.criterion = nn.MSELoss()
                loss = self.criterion(errors_scores,errors)

                print errors.data.numpy()[0] / 10, 'errors example'
                print errors_scores.data.numpy()[0] / 10, 'errors prediction example'

            elif self.loss_vector_type == None:

                batch[1] = torch.cat((torch.mul(batch[1][:, 28:31], 10), batch[1][:, 0:12]), dim=1)
                batch[0],batch[1], _ = SyntheticLib().synthetic_master(batch[0], batch[1], flip=True, shift=True, scale=False, bedangle=True, arms_only = self.opt.arms_only, include_inter = self.include_inter, p_cons = self.loss_vector_type)

                images, targets, euclidean_targets = Variable(batch[0], requires_grad = False), Variable(batch[1], requires_grad = False), Variable(torch.Tensor(np.zeros((batch[1].numpy().shape[0],batch[1].numpy().shape[1]/3))), requires_grad = False)

                self.optimizer.zero_grad()
                #print images.size(), 'im size'
                #print targets.size(), 'target size'
                scores, targets_est = self.model.forward_direct(images, targets)
                #print scores.size(), 'scores'

                self.criterion = nn.MSELoss()

                #print scores.size(), 'scores'
                #print targets_est.size(), 'targets est'
                #print targets.size(), 'targets'
                #print euclidean_targets.size()
                #loss = self.criterion(targets_est,targets)


                loss = self.criterion(scores, euclidean_targets)

            loss.backward()
            self.optimizer.step()
            scores = targets_est


            if batch_idx % opt.log_interval == 0:
                if self.loss_vector_type == 'torso_lengths':
                    VisualizationLib().print_error(torso_lengths, torso_lengths_scores, self.output_size, self.loss_vector_type, data = 'train')

                elif self.loss_vector_type == 'arm_angles':
                    scores = KinematicsLib().forward_arm_kinematics(images, torso_lengths_scores, constraint_scores)
                    scores = Variable(torch.Tensor(scores))

                    VisualizationLib().print_error(targets, scores, self.output_size, self.loss_vector_type, data = 'train')
                    self.im_sample = batch[0].numpy()
                    #self.im_sample = self.im_sample[:,0,:,:]
                    self.im_sample = np.squeeze(self.im_sample[0, :])
                    self.tar_sample = batch[1].numpy()
                    self.tar_sample = np.squeeze(self.tar_sample[0, :])/1000
                    self.sc_sample = scores.data.numpy()
                    self.sc_sample = np.squeeze(self.sc_sample[0, :]) / 1000
                    self.sc_sample = np.reshape(self.sc_sample, self.output_size)

                elif self.loss_vector_type == 'euclidean_error':
                    VisualizationLib().print_error(targets, scores, self.output_size, self.loss_vector_type, data='train')
                    self.im_sample = batch[0].numpy()
                    #self.im_sample = self.im_sample[:, 1, :, :]
                    self.im_sample = np.squeeze(self.im_sample[0, :])
                    self.tar_sample = batch[1].numpy()
                    self.tar_sample = np.squeeze(self.tar_sample[0, :]) / 1000
                    self.sc_sample = scores.data.numpy()
                    self.sc_sample = np.squeeze(self.sc_sample[0, :]) / 1000
                    self.sc_sample = np.reshape(self.sc_sample, self.output_size)

                elif self.loss_vector_type == None:
                    VisualizationLib().print_error(targets, scores, self.output_size, self.loss_vector_type, data='train')
                    self.im_sample = batch[0].numpy()
                    #self.im_sample = self.im_sample[:, 1, :, :]
                    self.im_sample = np.squeeze(self.im_sample[0, :])
                    self.tar_sample = batch[1].numpy()
                    self.tar_sample = np.squeeze(self.tar_sample[0, :]) / 1000
                    self.sc_sample = scores.data.numpy()
                    self.sc_sample = np.squeeze(self.sc_sample[0, :]) / 1000
                    self.sc_sample = np.reshape(self.sc_sample, self.output_size)


                val_loss = self.evaluate('test', n_batches=4)
                train_loss = loss.data[0]
                examples_this_epoch = batch_idx * len(images)
                epoch_progress = 100. * batch_idx / len(self.train_loader)
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t'
                      'Train Loss: {:.6f}\tVal Loss: {:.6f}'.format(
                    epoch, examples_this_epoch, len(self.train_loader.dataset),
                    epoch_progress, train_loss, val_loss))


                print 'appending to alldata losses'
                self.train_val_losses_all['train'+self.save_name + str(self.opt.leaveOut)].append(train_loss)
                self.train_val_losses_all['val'+self.save_name + str(self.opt.leaveOut)].append(val_loss)
                self.train_val_losses_all['epoch'+self.save_name + str(self.opt.leaveOut)].append(epoch)




    def evaluate(self, split, verbose=False, n_batches=None):
        '''
        Compute loss on val or test data.
        '''
        #print 'eval', split


        self.model.eval()
        loss = 8.0
        n_examples = 0
        if split == 'val':
            loader = val_loader
        elif split == 'test':
            loader = self.test_loader
        for batch_i, batch in enumerate(loader):

            self.model.train()
            if self.loss_vector_type == 'torso_lengths':
                batch.append(batch[1][:, 12:28])  # get the constraints we'll be training on, this makes for batch[2]

                batch[1] = torch.cat((torch.mul(batch[1][:, 28:31], 10), batch[1][:, 0:12]), dim=1)  # direct joints
                # now make the new set we'll be taking the loss over, in this case, the torso position and joint lengths
                batch[2] = torch.cat((torch.mul(batch[1][:, 0:3], 0.1), batch[2][:, 0:8]), dim=1)

                image, target, torso_length = Variable(batch[0], volatile = True), Variable(batch[1], volatile = True), Variable(batch[2], volatile = True)

                target = torso_length
                output = self.model(image)
                loss += self.criterion(output, target).data[0]


            elif self.loss_vector_type == 'arm_angles':
                batch.append(batch[1][:, 12:31])  # get the constraints we'll be training on
                #print batch[1][:, 28:31].shape, 'val'
                #print batch[1][:, 0:12].shape, 'val'
                batch[1] = torch.cat((torch.mul(batch[1][:, 28:31], 10), batch[1][:, 0:12]), dim=1)
                #print batch[1].shape, 'val'
                #print batch[2].shape, 'val'
                batch[2] = batch[2][:, 8:16]
                batch[2] = batch[2].numpy()
                batch[2][:,2:4] = batch[2][:,2:4] * 4
                batch[2] = torch.Tensor(batch[2])

                image, target, constraint = Variable(batch[0], volatile = True), Variable(batch[1], volatile = True), Variable(batch[2], volatile = True)

                #print batch[0].shape
                #print image[0].shape
                constraint_score = self.model(image)
                torso_length_score = self.model_torso_lengths(image)

                output = KinematicsLib().forward_arm_kinematics(image, torso_length_score, constraint_score) #remember to change this to constraint scores.
                output = Variable(torch.Tensor(output), volatile = True)
                loss += self.criterion(output, target).data[0]


            elif self.loss_vector_type == 'euclidean_error':
                batch[1] = torch.cat((torch.mul(batch[1][:, 28:31], 10), batch[1][:, 0:12]), dim=1)
                image, target = Variable(batch[0], volatile = True), Variable(batch[1], volatile = True)
                output = self.model_direct_marker(image)

                error = VisualizationLib().print_error(target, output, self.output_size, self.loss_vector_type, data='validate', printerror = False)
                error = Variable(torch.Tensor(error))
                error_output = self.model(image)
                loss += self.criterion(error_output, error).data[0]
                print error.data.numpy()[0] / 10, 'validation error example'
                print error_output.data.numpy()[0] / 10, 'validation error prediction example'

            elif self.loss_vector_type == None:
                batch[1] = torch.cat((torch.mul(batch[1][:, 28:31], 10), batch[1][:, 0:12]), dim=1)
                image, target, euclidean_target = Variable(batch[0], volatile = True), Variable(batch[1], volatile = True), Variable(torch.Tensor(np.zeros((batch[1].numpy().shape[0],batch[1].numpy().shape[1]/3))), requires_grad = False)


                output, target_est = self.model.forward_direct(image, target)
                loss += self.criterion(output, euclidean_target).data[0]
                output = target_est


            # predict the argmax of the log-probabilities
            pred = output.data.max(1, keepdim=True)[1]
            n_examples += pred.size(0)

            if n_batches and (batch_i >= n_batches):
                break

        loss /= n_examples
        loss *= 100



        VisualizationLib().print_error(target, output, self.output_size, self.loss_vector_type, data='validate')


        if self.loss_vector_type is not 'euclidean_error':
            if self.loss_vector_type is not 'torso_lengths':
                self.im_sampleval = image.data.numpy()
                #self.im_sampleval = self.im_sampleval[:,0,:,:]
                self.im_sampleval = np.squeeze(self.im_sampleval[0, :])
                self.tar_sampleval = target.data.numpy()
                self.tar_sampleval = np.squeeze(self.tar_sampleval[0, :]) / 1000
                self.sc_sampleval = output.data.numpy()
                self.sc_sampleval = np.squeeze(self.sc_sampleval[0, :]) / 1000
                self.sc_sampleval = np.reshape(self.sc_sampleval, self.output_size)

                VisualizationLib().visualize_pressure_map(self.im_sample, self.tar_sample, self.sc_sample, self.im_sampleval, self.tar_sampleval, self.sc_sampleval, block = False)



        if verbose:
            print('\n{} set: Average loss: {:.4f}\n'.format(
                split, loss))
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
    p.add_option('--lab_hd', action='store_true',
                 dest='lab_harddrive', \
                 default=False, \
                 help='Set path to the training database on lab harddrive.')
    p.add_option('--arms_only', action='store_true',
                 dest='arms_only', \
                 default=False, \
                 help='Train only on data from the arms, both sitting and laying.')

    p.add_option('--verbose', '--v',  action='store_true', dest='verbose',
                 default=False, help='Printout everything (under construction).')
    p.add_option('--log_interval', type=int, default=10, metavar='N',
                        help='number of batches between logging train status')

    opt, args = p.parse_args()

    if opt.lab_harddrive == True:

        if opt.arms_only == True:
            opt.subject2Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_2/p_files/trainval_150rh1_lh1_rl_ll_100rh23_lh23_sit120rh_lh_rl_ll.p'
            opt.subject3Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_3/p_files/trainval_150rh1_lh1_rl_ll_100rh23_lh23_sit120rh_lh_rl_ll.p'
            opt.subject4Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_4/p_files/trainval_150rh1_lh1_rl_ll_100rh23_lh23_sit120rh_lh_rl_ll.p'
            opt.subject5Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_5/p_files/trainval_150rh1_lh1_rl_ll_100rh23_lh23_sit120rh_lh_rl_ll.p'
            opt.subject6Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_6/p_files/trainval_150rh1_lh1_rl_ll_100rh23_lh23_sit120rh_lh_rl_ll.p'
            opt.subject7Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_7/p_files/trainval_150rh1_lh1_rl_ll_100rh23_lh23_sit120rh_lh_rl_ll.p'
            opt.subject8Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_8/p_files/trainval_150rh1_lh1_rl_ll_100rh23_lh23_sit120rh_lh_rl_ll.p'
            opt.subject9Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_9/p_files/trainval_200rh1_lh1_rl_ll_100rh23_lh23_sit120rh_lh_rl_ll.p'
            opt.subject10Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_10/p_files/trainval_200rh1_lh1_100rh23_lh23_sit120rh_lh.p'
            opt.subject11Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_11/p_files/trainval_200rh1_lh1_100rh23_lh23_sit120rh_lh.p'
            opt.subject12Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_12/p_files/trainval_200rh1_lh1_100rh23_lh23_sit120rh_lh.p'
            opt.subject13Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_13/p_files/trainval_200rh1_lh1_100rh23_lh23_sit120rh_lh.p'
            opt.subject14Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_14/p_files/trainval_200rh1_lh1_100rh23_lh23_sit120rh_lh.p'
            opt.subject15Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_15/p_files/trainval_200rh1_lh1_100rh23_lh23_sit120rh_lh.p'
            opt.subject16Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_16/p_files/trainval_200rh1_lh1_100rh23_lh23_sit120rh_lh.p'
            opt.subject17Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_17/p_files/trainval_200rh1_lh1_100rh23_lh23_sit120rh_lh.p'
            opt.subject18Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_18/p_files/trainval_200rh1_lh1_100rh23_lh23_sit120rh_lh.p'

        else:
            opt.subject2Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_2/p_files/trainval_200rh1_lh1_rl_ll_100rh23_lh23_head_sit120rh_lh_rl_ll.p'
            opt.subject3Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_3/p_files/trainval_200rh1_lh1_rl_ll_100rh23_lh23_head_sit120rh_lh_rl_ll.p'
            opt.subject4Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_4/p_files/trainval_200rh1_lh1_rl_ll_100rh23_lh23_head_sit120rh_lh_rl_ll.p'
            opt.subject5Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_5/p_files/trainval_200rh1_lh1_rl_ll_100rh23_lh23_head_sit120rh_lh_rl_ll.p'
            opt.subject6Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_6/p_files/trainval_200rh1_lh1_rl_ll_100rh23_lh23_head_sit120rh_lh_rl_ll.p'
            opt.subject7Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_7/p_files/trainval_200rh1_lh1_rl_ll_100rh23_lh23_head_sit120rh_lh_rl_ll.p'
            opt.subject8Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_8/p_files/trainval_200rh1_lh1_rl_ll_100rh23_lh23_head_sit120rh_lh_rl_ll.p'

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
        training_database_file.append(opt.subject2Path)
        training_database_file.append(opt.subject3Path)
        training_database_file.append(opt.subject5Path)
        training_database_file.append(opt.subject6Path)
        training_database_file.append(opt.subject7Path)
        training_database_file.append(opt.subject8Path)
        # training_database_file.append(opt.subject9Path)
        # training_database_file.append(opt.subject10Path)
        # training_database_file.append(opt.subject11Path)
        # training_database_file.append(opt.subject12Path)
        # training_database_file.append(opt.subject13Path)
        # training_database_file.append(opt.subject14Path)
        # training_database_file.append(opt.subject15Path)
        # training_database_file.append(opt.subject16Path)
        # training_database_file.append(opt.subject17Path)
        # training_database_file.append(opt.subject18Path)

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
        p.convnet_2layer()

        #else:
        #    print 'Please specify correct training type:1. HoG_KNN 2. convnet_2'
