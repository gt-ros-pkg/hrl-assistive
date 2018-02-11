#!/usr/bin/env python
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import *


#PyTorch libraries
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable


import convnet as convnet
import convnet_cascade as convnet_cascade
import tf.transformations as tft

# import hrl_lib.util as ut
import cPickle as pickle
# from hrl_lib.util import load_pickle
def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Pose Estimation Libraries
from create_dataset_lib import CreateDatasetLib
from synthetic_lib import SyntheticLib
from visualization_lib import VisualizationLib
from kinematics_lib import KinematicsLib
from cascade_lib import CascadeLib
from preprocessing_lib import PreprocessingLib


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
from sklearn.multioutput import MultiOutputRegressor


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

if torch.cuda.is_available():
    # Use for GPU
    dtype = torch.cuda.FloatTensor
else:
    # Use for CPU
    dtype = torch.FloatTensor

class PhysicalTrainer():
    '''Gets the dictionary of pressure maps from the training database,
    and will have API to do all sorts of training with it.'''
    def __init__(self, training_database_file, test_file, opt):
        '''Opens the specified pickle files to get the combined dataset:
        This dataset is a dictionary of pressure maps with the corresponding
        3d position and orientation of the markers associated with it.'''
        self.verbose = opt.verbose
        self.opt = opt
        self.batch_size = 128
        self.num_epochs = 200
        self.include_inter = True

        self.count = 0


        print test_file
        #Entire pressure dataset with coordinates in world frame

        self.save_name = '_2to8_' + opt.losstype + str(self.batch_size) + 'b_' + str(self.num_epochs) + 'e_4'


        #change this to 'direct' when you are doing baseline methods
        self.loss_vector_type = opt.losstype  # 'arms_cascade'#'upper_angles' #this is so you train the set to joint lengths and angles

        #we'll be loading this later
        if self.opt.computer == 'lab_harddrive':
            #try:
            self.train_val_losses_all = load_pickle('/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/train_val_losses_all.p')
        elif self.opt.computer == 'aws':
            self.train_val_losses_all = load_pickle('/home/ubuntu/Autobed_OFFICIAL_Trials/train_val_losses_all.p')

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
        elif self.loss_vector_type == 'angles' or self.loss_vector_type == 'direct' or self.loss_vector_type == 'confidence':
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

        # Standardize data to values between [0, 1]
        # print 'Standardizing training data'
        # mins = [np.min(train_xa[:, ii]) for ii in xrange(np.shape(train_xa)[1])]
        # maxs = [np.max(train_xa[:, ii]) for ii in xrange(np.shape(train_xa)[1])]
        # print 'mins:', mins, 'maxs:', maxs
        # for ii in xrange(np.shape(train_xa)[1]):
        #     train_xa[:, ii] = (train_xa[:, ii] - mins[ii]) / (maxs[ii] - mins[ii])

        self.train_x_tensor = torch.Tensor(train_xa)

        self.train_y_flat = [] #Initialize the training ground truth list
        for entry in range(len(dat['images'])):
            if self.loss_vector_type == 'upper_angles':
                c = np.concatenate((dat['markers_xyz_m'][entry][0:18] * 1000,
                                    dat['joint_lengths_U_m'][entry][0:9] * 100,
                                    dat['joint_angles_U_deg'][entry][0:10]), axis=0)
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

        # Standardize data to values between [0, 1]
        # print 'Standardizing test data'
        # for ii in xrange(np.shape(test_xa)[1]):
        #     test_xa[:, ii] = (test_xa[:, ii] - mins[ii]) / (maxs[ii] - mins[ii])

        self.test_x_tensor = torch.Tensor(test_xa)

        self.test_y_flat = []  # Initialize the ground truth list
        for entry in range(len(test_dat['images'])):
            if self.loss_vector_type == 'upper_angles':
                c = np.concatenate((test_dat['markers_xyz_m'][entry][0:18] * 1000,
                                    test_dat['joint_lengths_U_m'][entry][0:9] * 100,
                                    test_dat['joint_angles_U_deg'][entry][0:10]), axis=0)
                self.test_y_flat.append(c)
            elif self.loss_vector_type == 'angles':
                c = np.concatenate((test_dat['markers_xyz_m'][entry][0:30] * 1000,
                                    test_dat['joint_lengths_U_m'][entry][0:9] * 100,
                                    test_dat['joint_angles_U_deg'][entry][0:10],
                                    test_dat['joint_lengths_L_m'][entry][0:8] * 100,
                                    test_dat['joint_angles_L_deg'][entry][0:8]), axis=0)
                self.test_y_flat.append(c)
            elif self.loss_vector_type == 'direct' or self.loss_vector_type == 'confidence':
                self.test_y_flat.append(test_dat['markers_xyz_m'][entry] * 1000)
            else:
                print "ERROR! SPECIFY A VALID LOSS VECTOR TYPE."
        self.test_y_flat = np.array(self.test_y_flat)
        self.test_y_tensor = torch.Tensor(self.test_y_flat)


    def baseline_train(self, baseline):
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

            # get the whole body x y z
            batch[1] = batch[1][:, 0:30]

            batch[0], batch[1], _ = SyntheticLib().synthetic_master(batch[0], batch[1], flip=True,
                                                                           shift=True, scale=True,
                                                                           bedangle=True,
                                                                           include_inter=self.include_inter,
                                                                           loss_vector_type=self.loss_vector_type)


            images = batch[0].numpy()[:,0,:,:]
            targets = batch[1].numpy()


            #upsample the images
            images_up = PreprocessingLib().preprocessing_pressure_map_upsample(images)
            #targets = list(targets)
            #print images[0].shape



            # Compute HoG of the current(training) pressure map dataset
            images_up = PreprocessingLib().compute_HoG(images_up)

            #images_up = np.reshape(images, (images.shape[0], images.shape[1]*images.shape[2]))

            #images_up = [[0], [1], [2], [3]]
            #targets = [0, 0, 1, 1]

            print np.shape(images_up)
            print np.shape(targets)

            print 'fitting KNN'
            baseline = 'KNN'

            #if baseline == 'KNN':
            regr = neighbors.KNeighborsRegressor(10, weights='distance')
            regr.fit(images_up, targets)

            print 'done fitting KNN'

            if self.opt.computer == 'lab_harddrive':
                print 'saving to ', '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_' + str(
                    self.opt.leaveOut) + '/p_files/HoG_' + baseline + '_p' + str(self.opt.leaveOut) + '.p'
                pkl.dump(regr, open(
                    '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_' + str(
                        self.opt.leaveOut) + '/p_files/HoG_' + baseline + '_p' + str(self.opt.leaveOut) + '.p',
                    'wb'))
                print 'saved successfully'
            elif self.opt.computer == 'aws':
                pkl.dump(regr, open('/home/ubuntu/Autobed_OFFICIAL_Trials/subject_' + str(
                    self.opt.leaveOut) + '/HoG_' + baseline + '_p' + str(self.opt.leaveOut) + '.p', 'wb'))
                print 'saved successfully'


            print 'fitting Ridge'
            baseline = 'Ridge'
            #elif baseline == 'Ridge':
            regr = linear_model.Ridge(alpha=0.01)
            regr.fit(images_up, targets)

            print 'done fitting Ridge'

            if self.opt.computer == 'lab_harddrive':
                print 'saving to ', '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_' + str(
                    self.opt.leaveOut) + '/p_files/HoG_' + baseline + '_p' + str(self.opt.leaveOut) + '.p'
                pkl.dump(regr, open(
                    '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_' + str(
                        self.opt.leaveOut) + '/p_files/HoG_' + baseline + '_p' + str(self.opt.leaveOut) + '.p',
                    'wb'))
                print 'saved successfully'
            elif self.opt.computer == 'aws':
                pkl.dump(regr, open('/home/ubuntu/Autobed_OFFICIAL_Trials/subject_' + str(
                    self.opt.leaveOut) + '/HoG_' + baseline + '_p' + str(self.opt.leaveOut) + '.p', 'wb'))
                print 'saved successfully'




            print 'fitting KRidge'
            baseline = 'KRidge'
            #elif baseline == 'KRidge':
            regr = kernel_ridge.KernelRidge(alpha=0.01, kernel='rbf')
            regr.fit(images_up, targets)

            print 'done fitting KRidge'

            if self.opt.computer == 'lab_harddrive':
                print 'saving to ', '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_' + str(
                    self.opt.leaveOut) + '/p_files/HoG_' + baseline + '_p' + str(self.opt.leaveOut) + '.p'
                pkl.dump(regr, open(
                    '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_' + str(
                        self.opt.leaveOut) + '/p_files/HoG_' + baseline + '_p' + str(self.opt.leaveOut) + '.p',
                    'wb'))
                print 'saved successfully'
            elif self.opt.computer == 'aws':
                pkl.dump(regr, open('/home/ubuntu/Autobed_OFFICIAL_Trials/subject_' + str(
                    self.opt.leaveOut) + '/HoG_' + baseline + '_p' + str(self.opt.leaveOut) + '.p', 'wb'))
                print 'saved successfully'



            if baseline == 'SVM':
                regr = MultiOutputRegressor(estimator=svm.SVR(C=1.0, kernel='rbf', verbose = True))
                regr.fit(images_up, targets)
                #SVR(C=1.0, kernel='rbf', verbose = True)
                #SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto',
                #                    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)

            elif baseline == 'kmeans_SVM':
                k_means = KMeans(n_clusters=10, n_init=4)
                k_means.fit(images_up)
                labels = k_means.labels_
                print labels.shape, 'label shape'
                print targets.shape, 'target shape'
                print 'done fitting kmeans'
                svm_classifier = svm.SVC(kernel='rbf', verbose = True)
                svm_classifier.fit(labels, targets)
                print 'done fitting svm'
                regr = linear_model.LinearRegression()
                regr.fit(labels, targets)
                print 'done fitting linear model'

            elif baseline == 'Linear':
                regr = linear_model.LinearRegression()
                regr.fit(images_up, targets)



            #validation
            for batchtest_idx, batchtest in enumerate(self.test_loader):
                images_test = batchtest[0].numpy()[:, 0, :, :]
                targets = batchtest[1].numpy()

                images_up_test = PreprocessingLib().preprocessing_pressure_map_upsample(images_test)
                #targets = list(targets)

                images_up_test = PreprocessingLib().compute_HoG(images_up_test)
                #images_up_test = np.reshape(images_test, (images_test.shape[0], images_test.shape[1] * images_test.shape[2]))


                scores = regr.predict(images_up_test)

                #print scores.shape
                #print targets.shape
                #print scores[0]
                #print targets[0]

                #print regr.predict(images_up_test[0]) - targets[0]
                #VisualizationLib().print_error(scores, targets, self.output_size, loss_vector_type=self.loss_vector_type, data='test', printerror=True)

                self.im_sample = np.squeeze(images_test[0, :])
                #print self.im_sample.shape

                self.tar_sample = np.squeeze(targets[0, :]) / 1000
                self.sc_sample = np.copy(scores)
                self.sc_sample = np.squeeze(self.sc_sample[0, :]) / 1000
                self.sc_sample = np.reshape(self.sc_sample, self.output_size)
                if self.opt.visualize == True:
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

        if self.loss_vector_type == 'angles':
            fc_output_size = 40#38 #18 angles for body, 17 lengths for body, 3 torso coordinates
            self.model = convnet.CNN(self.mat_size, fc_output_size, hidden_dim, kernel_size, self.loss_vector_type)
            pp = 0
            for p in list(self.model.parameters()):
                nn = 1
                for s in list(p.size()):
                    nn = nn * s
                pp += nn
            print pp, 'num params'
        elif self.loss_vector_type == 'arms_cascade':
            #we'll make a double pass through this network for the validation for each arm.
            fc_output_size = 4 #4 angles for arms
            self.model = convnet_cascade.CNN(self.mat_size, fc_output_size, hidden_dim, kernel_size, self.loss_vector_type)
            self.model_cascade_prior = torch.load('/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_' + str(self.opt.leaveOut) + '/p_files/convnet_2to8_alldata_angles_constrained_noise_115b_100e_4.pt')
        elif self.loss_vector_type == 'direct' or self.loss_vector_type == 'confidence':
            fc_output_size = 30
            self.model = convnet.CNN(self.mat_size, fc_output_size, hidden_dim, kernel_size, self.loss_vector_type)

        # Run model on GPU if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()


        self.criterion = F.cross_entropy



        if self.loss_vector_type == None:
            self.optimizer2 = optim.Adam(self.model.parameters(), lr=0.00002, weight_decay=0.0005)
        elif self.loss_vector_type == 'upper_angles' or self.loss_vector_type == 'arms_cascade' or self.loss_vector_type == 'angles' or self.loss_vector_type == 'direct':
            self.optimizer2 = optim.Adam(self.model.parameters(), lr=0.00002, weight_decay=0.0005)  #0.000002 does not converge even after 100 epochs on subjects 2-8 kin cons. use .00001
        elif self.loss_vector_type == 'direct':
            self.optimizer2 = optim.Adam(self.model.parameters(), lr=0.00002, weight_decay=0.0005)
        #self.optimizer = optim.RMSprop(self.model.parameters(), lr=0.000001, momentum=0.7, weight_decay=0.0005)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.00002, weight_decay=0.0005) #start with .00005


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

            elif self.opt.computer == 'aws':
                torch.save(self.model, '/home/ubuntu/Autobed_OFFICIAL_Trials/subject_'+str(self.opt.leaveOut)+'/convnet'+self.save_name+'.pt')
                pkl.dump(self.train_val_losses_all,
                         open(os.path.join('/home/ubuntu/Autobed_OFFICIAL_Trials/train_val_losses_all.p'), 'wb'))

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

            if self.loss_vector_type == 'angles':

                # append upper joint angles, lower joint angles, upper joint lengths, lower joint lengths, in that order
                batch.append(torch.cat((batch[1][:,39:49], batch[1][:, 57:65], batch[1][:, 30:39], batch[1][:, 49:57]), dim = 1))

                #get the whole body x y z
                batch[1] = batch[1][:, 0:30]

                batch[0], batch[1], batch[2]= SyntheticLib().synthetic_master(batch[0], batch[1], batch[2], flip=True, shift=True, scale=True, bedangle=True, include_inter=self.include_inter, loss_vector_type=self.loss_vector_type)

                images_up = Variable(torch.Tensor(PreprocessingLib().preprocessing_add_image_noise(np.array(PreprocessingLib().preprocessing_pressure_map_upsample(batch[0].numpy()[:, :, 10:74, 10:37])))).type(dtype), requires_grad=False)
                images, targets, constraints = Variable(batch[0].type(dtype), requires_grad = False), Variable(batch[1].type(dtype), requires_grad = False), Variable(batch[2].type(dtype), requires_grad = False)


                # targets_2D = CascadeLib().get_2D_projection(images.data.numpy(), np.reshape(targets.data.numpy(), (targets.size()[0], 10, 3)))
                targets_2D = CascadeLib().get_2D_projection(images.data, targets.data.view(targets.size()[0], 10, 3))

                #image_coords = np.round(targets_2D[:, :, 0:2] / 28.6, 0)
                #print image_coords[0, :, :]

                self.optimizer.zero_grad()

                ground_truth = np.zeros((batch[0].numpy().shape[0], 47)) #47 is 17 joint lengths and 30 joint locations for x y z
                ground_truth = Variable(torch.Tensor(ground_truth).type(dtype))
                ground_truth[:, 0:17] = constraints[:, 18:35]/100
                ground_truth[:, 17:47] = targets[:, 0:30]/1000



                scores_zeros = np.zeros((batch[0].numpy().shape[0], 27)) #27 is  10 euclidean errors and 17 joint lengths
                scores_zeros = Variable(torch.Tensor(scores_zeros).type(dtype))
                scores_zeros[:, 10:27] = constraints[:, 18:35]/200 #divide by 100 for direct output. divide by 10 if you multiply the estimate length by 10.


                scores, targets_est, angles_est, lengths_est, _ = self.model.forward_kinematic_jacobian(images_up, targets, constraints) # scores is a variable with 27 for 10 euclidean errors and 17 lengths in meters. targets est is a numpy array in mm.
                #print lengths_est[0,0:10], 'lengths est'
                #print batch[0][0,2,10,10], 'angle'

                #print scores_zeros[0, :]

                self.criterion = nn.L1Loss()
                loss = self.criterion(scores, scores_zeros)


            elif self.loss_vector_type == 'direct':

                batch[0],batch[1], _ = SyntheticLib().synthetic_master(batch[0], batch[1], flip=True, shift=True, scale=True, bedangle=True, include_inter = self.include_inter, loss_vector_type = self.loss_vector_type)

                images_up = Variable(torch.Tensor(PreprocessingLib().preprocessing_add_image_noise(np.array(PreprocessingLib().preprocessing_pressure_map_upsample(batch[0].numpy()[:, :, 10:74, 10:37])))).type(dtype), requires_grad=False)
                images, targets, scores_zeros = Variable(batch[0].type(dtype), requires_grad = False), Variable(batch[1].type(dtype), requires_grad = False), Variable(torch.Tensor(np.zeros((batch[1].shape[0], batch[1].shape[1]/3))).type(dtype), requires_grad = False)

                self.optimizer.zero_grad()
                scores, targets_est = self.model.forward_direct(images_up, targets)

                self.criterion = nn.L1Loss()
                loss = self.criterion(scores, scores_zeros)

            #print loss.data.numpy() * 1000, 'loss'

            loss.backward()
            self.optimizer.step()
            loss *= 1000


            if batch_idx % opt.log_interval == 0:
                if self.loss_vector_type == 'upper_angles' or self.loss_vector_type == 'angles':
                    VisualizationLib().print_error(targets.data, targets_est, self.output_size, self.loss_vector_type, data = 'train')
                    print angles_est[0, :], 'angles'
                    print batch[0][0,2,10,10], 'bed angle'
                    self.im_sample = images.data
                    # #self.im_sample = self.im_sample[:,0,:,:]
                    self.im_sample = self.im_sample[0, :].squeeze()
                    self.tar_sample = targets.data
                    self.tar_sample = self.tar_sample[0, :].squeeze()/1000
                    self.sc_sample = targets_est.clone()
                    self.sc_sample = self.sc_sample[0, :].squeeze() / 1000
                    self.sc_sample = self.sc_sample.view(self.output_size)

                elif self.loss_vector_type == 'direct':
                    VisualizationLib().print_error(targets.data, targets_est, self.output_size, self.loss_vector_type, data='train')
                    self.im_sample = images.data
                    #self.im_sample = self.im_sample[:, 1, :, :]
                    self.im_sample = self.im_sample[0, :].squeeze()
                    self.tar_sample = targets.data
                    self.tar_sample = self.tar_sample[0, :].squeeze()/1000
                    self.sc_sample = targets_est.clone()
                    self.sc_sample = self.sc_sample[0, :].squeeze() / 1000
                    self.sc_sample = self.sc_sample.view(self.output_size)

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

        self.model.eval()
        loss = 0.
        n_examples = 0
        for batch_i, batch in enumerate(self.test_loader):

            self.model.train()


            if self.loss_vector_type == 'angles':

                #append upper joint angles, lower joint angles, upper joint lengths, lower joint lengths, in that order
                batch.append(torch.cat((batch[1][:,39:49], batch[1][:, 57:65], batch[1][:, 30:39], batch[1][:, 49:57]), dim = 1))

                #get the direct joint locations
                batch[1] = batch[1][:, 0:30]

                images_up = Variable(torch.Tensor(np.array(PreprocessingLib().preprocessing_pressure_map_upsample(batch[0].numpy()[:, :, 10:74, 10:37]))).type(dtype), requires_grad=False)
                images, targets, constraints = Variable(batch[0].type(dtype), volatile = True, requires_grad=False), Variable(batch[1].type(dtype),volatile = True, requires_grad=False), Variable(batch[2].type(dtype), volatile = True, requires_grad=False)

                self.optimizer.zero_grad()


                ground_truth = np.zeros((batch[0].numpy().shape[0], 47))
                ground_truth = Variable(torch.Tensor(ground_truth).type(dtype))
                ground_truth[:, 0:17] = constraints[:, 18:35]/100
                ground_truth[:, 17:47] = targets[:, 0:30]/1000

                scores_zeros = np.zeros((batch[0].numpy().shape[0], 27))
                scores_zeros = Variable(torch.Tensor(scores_zeros).type(dtype))
                scores_zeros[:, 10:27] = constraints[:, 18:35]/200

                scores, targets_est, angles_est, lengths_est, _ = self.model.forward_kinematic_jacobian(images_up, targets, constraints)


                self.criterion = nn.L1Loss()
                loss = self.criterion(scores[:, 0:10], scores_zeros[:, 0:10])
                loss = loss.data[0]



            elif self.loss_vector_type == 'direct':

                images_up = Variable(torch.Tensor(np.array(PreprocessingLib().preprocessing_pressure_map_upsample(batch[0].numpy()[:, :, 10:74, 10:37]))).type(dtype), requires_grad=False)
                images, targets, scores_zeros = Variable(batch[0].type(dtype), requires_grad=False), Variable(batch[1].type(dtype), requires_grad=False), Variable(torch.Tensor(np.zeros((batch[1].shape[0], batch[1].shape[1] / 3))).type(dtype), requires_grad=False)

                scores, targets_est = self.model.forward_direct(images_up, targets)
                self.criterion = nn.L1Loss()

                loss = self.criterion(scores, scores_zeros)
                loss = loss.data[0]


            n_examples += self.batch_size
            #print n_examples

            if n_batches and (batch_i >= n_batches):
                break

        loss /= n_examples
        loss *= 100
        loss *= 1000


        VisualizationLib().print_error(targets.data, targets_est, self.output_size, self.loss_vector_type, data='validate')

        if self.loss_vector_type == 'angles' or self.loss_vector_type == 'direct':
            if self.loss_vector_type == 'angles':
                print angles_est[0, :], 'validation angles'
                print lengths_est[0, :], 'validation lengths'

            print batch[0][0,2,10,10], 'validation bed angle'
            self.im_sampleval = images.data
            # #self.im_sampleval = self.im_sampleval[:,0,:,:]
            self.im_sampleval = self.im_sampleval[0, :].squeeze()
            self.tar_sampleval = targets.data
            self.tar_sampleval = self.tar_sampleval[0, :].squeeze() / 1000
            self.sc_sampleval = targets_est.clone()
            self.sc_sampleval = self.sc_sampleval[0, :].squeeze() / 1000
            self.sc_sampleval = self.sc_sampleval.view(self.output_size)

            if self.opt.visualize == True:
                VisualizationLib().visualize_pressure_map(self.im_sample, self.tar_sample, self.sc_sample,self.im_sampleval, self.tar_sampleval, self.sc_sampleval, block=False)


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
    p.add_option('--qt', action='store_true',
                 dest='quick_test', \
                 default=False, \
                 help='Train only on data from the arms, both sitting and laying.')
    p.add_option('--viz', action='store_true',
                 dest='visualize', \
                 default=False, \
                 help='Train only on data from the arms, both sitting and laying.')

    p.add_option('--verbose', '--v',  action='store_true', dest='verbose',
                 default=False, help='Printout everything (under construction).')
    p.add_option('--log_interval', type=int, default=10, metavar='N',
                        help='number of batches between logging train status')

    opt, args = p.parse_args()

    if opt.computer == 'lab_harddrive':

        opt.subject2Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_2/p_files/trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p'
        opt.subject3Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_3/p_files/trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p'
        opt.subject4Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_4/p_files/trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p'
        opt.subject5Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_5/p_files/trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p'
        opt.subject6Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_6/p_files/trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p'
        opt.subject7Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_7/p_files/trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p'
        opt.subject8Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_8/p_files/trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p'
        opt.subject9Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_9/p_files/trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p'
        opt.subject10Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_10/p_files/trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p'
        opt.subject11Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_11/p_files/trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p'
        opt.subject12Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_12/p_files/trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p'
        opt.subject13Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_13/p_files/trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p'
        opt.subject14Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_14/p_files/trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p'
        opt.subject15Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_15/p_files/trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p'
        opt.subject16Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_16/p_files/trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p'
        opt.subject17Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_17/p_files/trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p'
        opt.subject18Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_18/p_files/trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p'

        #shortcut:

        if opt.quick_test == True:
            opt.subject4Path = '/home/henryclever/test/trainval4_150rh1_sit120rh.p'
            opt.subject8Path = '/home/henryclever/test/trainval8_150rh1_sit120rh.p'


    elif opt.computer == 'aws':

        opt.subject2Path = '/home/ubuntu/Autobed_OFFICIAL_Trials/subject_2/trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p'
        opt.subject3Path = '/home/ubuntu/Autobed_OFFICIAL_Trials/subject_3/trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p'
        opt.subject4Path = '/home/ubuntu/Autobed_OFFICIAL_Trials/subject_4/trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p'
        opt.subject5Path = '/home/ubuntu/Autobed_OFFICIAL_Trials/subject_5/trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p'
        opt.subject6Path = '/home/ubuntu/Autobed_OFFICIAL_Trials/subject_6/trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p'
        opt.subject7Path = '/home/ubuntu/Autobed_OFFICIAL_Trials/subject_7/trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p'
        opt.subject8Path = '/home/ubuntu/Autobed_OFFICIAL_Trials/subject_8/trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p'
        opt.subject9Path = '/home/ubuntu/Autobed_OFFICIAL_Trials/subject_9/trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p'
        opt.subject10Path = '/home/ubuntu/Autobed_OFFICIAL_Trials/subject_10/trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p'
        opt.subject11Path = '/home/ubuntu/Autobed_OFFICIAL_Trials/subject_11/trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p'
        opt.subject12Path = '/home/ubuntu/Autobed_OFFICIAL_Trials/subject_12/trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p'
        opt.subject13Path = '/home/ubuntu/Autobed_OFFICIAL_Trials/subject_13/trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p'
        opt.subject14Path = '/home/ubuntu/Autobed_OFFICIAL_Trials/subject_14/trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p'
        opt.subject15Path = '/home/ubuntu/Autobed_OFFICIAL_Trials/subject_15/trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p'
        opt.subject16Path = '/home/ubuntu/Autobed_OFFICIAL_Trials/subject_16/trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p'
        opt.subject17Path = '/home/ubuntu/Autobed_OFFICIAL_Trials/subject_17/trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p'
        opt.subject18Path = '/home/ubuntu/Autobed_OFFICIAL_Trials/subject_18/trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p'

        #shortcut:

        if opt.quick_test == True:
            opt.subject4Path = '/home/ubuntu/test/trainval4_150rh1_sit120rh.p'
            opt.subject8Path = '/home/ubuntu/test/trainval8_150rh1_sit120rh.p'




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

    elif opt.leaveOut == 9:
        test_database_file = opt.subject9Path
        training_database_file.append(opt.subject10Path)
        training_database_file.append(opt.subject11Path)
        training_database_file.append(opt.subject12Path)
        training_database_file.append(opt.subject13Path)
        training_database_file.append(opt.subject14Path)
        training_database_file.append(opt.subject15Path)
        training_database_file.append(opt.subject16Path)
        training_database_file.append(opt.subject17Path)
        training_database_file.append(opt.subject18Path)
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
    elif opt.leaveOut == 11:
        test_database_file = opt.subject11Path
        training_database_file.append(opt.subject9Path)
        training_database_file.append(opt.subject10Path)
        training_database_file.append(opt.subject12Path)
        training_database_file.append(opt.subject13Path)
        training_database_file.append(opt.subject14Path)
        training_database_file.append(opt.subject15Path)
        training_database_file.append(opt.subject16Path)
        training_database_file.append(opt.subject17Path)
        training_database_file.append(opt.subject18Path)
    elif opt.leaveOut == 12:
        test_database_file = opt.subject12Path
        training_database_file.append(opt.subject9Path)
        training_database_file.append(opt.subject10Path)
        training_database_file.append(opt.subject11Path)
        training_database_file.append(opt.subject13Path)
        training_database_file.append(opt.subject14Path)
        training_database_file.append(opt.subject15Path)
        training_database_file.append(opt.subject16Path)
        training_database_file.append(opt.subject17Path)
        training_database_file.append(opt.subject18Path)
    elif opt.leaveOut == 13:
        test_database_file = opt.subject13Path
        training_database_file.append(opt.subject9Path)
        training_database_file.append(opt.subject10Path)
        training_database_file.append(opt.subject11Path)
        training_database_file.append(opt.subject12Path)
        training_database_file.append(opt.subject14Path)
        training_database_file.append(opt.subject15Path)
        training_database_file.append(opt.subject16Path)
        training_database_file.append(opt.subject17Path)
        training_database_file.append(opt.subject18Path)
    elif opt.leaveOut == 14:
        test_database_file = opt.subject14Path
        training_database_file.append(opt.subject9Path)
        training_database_file.append(opt.subject10Path)
        training_database_file.append(opt.subject11Path)
        training_database_file.append(opt.subject12Path)
        training_database_file.append(opt.subject13Path)
        training_database_file.append(opt.subject15Path)
        training_database_file.append(opt.subject16Path)
        training_database_file.append(opt.subject17Path)
        training_database_file.append(opt.subject18Path)
    elif opt.leaveOut == 15:
        test_database_file = opt.subject15Path
        training_database_file.append(opt.subject9Path)
        training_database_file.append(opt.subject10Path)
        training_database_file.append(opt.subject11Path)
        training_database_file.append(opt.subject12Path)
        training_database_file.append(opt.subject13Path)
        training_database_file.append(opt.subject14Path)
        training_database_file.append(opt.subject16Path)
        training_database_file.append(opt.subject17Path)
        training_database_file.append(opt.subject18Path)
    elif opt.leaveOut == 16:
        test_database_file = opt.subject16Path
        training_database_file.append(opt.subject9Path)
        training_database_file.append(opt.subject10Path)
        training_database_file.append(opt.subject11Path)
        training_database_file.append(opt.subject12Path)
        training_database_file.append(opt.subject13Path)
        training_database_file.append(opt.subject14Path)
        training_database_file.append(opt.subject15Path)
        training_database_file.append(opt.subject17Path)
        training_database_file.append(opt.subject18Path)
    elif opt.leaveOut == 17:
        test_database_file = opt.subject17Path
        training_database_file.append(opt.subject9Path)
        training_database_file.append(opt.subject10Path)
        training_database_file.append(opt.subject11Path)
        training_database_file.append(opt.subject12Path)
        training_database_file.append(opt.subject13Path)
        training_database_file.append(opt.subject14Path)
        training_database_file.append(opt.subject15Path)
        training_database_file.append(opt.subject16Path)
        training_database_file.append(opt.subject18Path)
    elif opt.leaveOut == 18:
        test_database_file = opt.subject18Path
        training_database_file.append(opt.subject9Path)
        training_database_file.append(opt.subject10Path)
        training_database_file.append(opt.subject11Path)
        training_database_file.append(opt.subject12Path)
        training_database_file.append(opt.subject13Path)
        training_database_file.append(opt.subject14Path)
        training_database_file.append(opt.subject15Path)
        training_database_file.append(opt.subject16Path)
        training_database_file.append(opt.subject17Path)

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



        if opt.mltype == 'convnet':
            p.init_convnet_train()
        elif opt.mltype != 'convnet':
            p.baseline_train(opt.mltype)

        #else:
        #    print 'Please specify correct training type:1. HoG_KNN 2. convnet_2'
