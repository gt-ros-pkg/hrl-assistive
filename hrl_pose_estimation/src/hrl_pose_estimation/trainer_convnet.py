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

import convnet

import pickle
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



np.set_printoptions(threshold='nan')

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

 
class PhysicalTrainer():
    '''Gets the dictionary of pressure maps from the training database, 
    and will have API to do all sorts of training with it.'''
    def __init__(self, training_database_file, test_file, opt):
        '''Opens the specified pickle files to get the combined dataset:
        This dataset is a dictionary of pressure maps with the corresponding
        3d position and orientation of the markers associated with it.'''
        self.verbose = opt.verbose
        self.opt = opt

        print test_file
        #Entire pressure dataset with coordinates in world frame


        #we'll be loading this later
        if self.opt.lab_harddrive == True:
            self.train_val_losses = load_pickle('/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/train_val_losses.p')
        else:
            self.train_val_losses = load_pickle('/home/henryclever/hrl_file_server/Autobed/train_val_losses.p')

        #print self.train_val_losses

        if self.opt.sitting == True:
            print 'appending to sitting losses'
            self.train_val_losses['train_sitting_flip_shift_scale10_700e_' + str(self.opt.leaveOut)] = []
            self.train_val_losses['val_sitting_flip_shift_scale10_700e_' + str(self.opt.leaveOut)] = []
            self.train_val_losses['epoch_sitting_flip_shift_scale10_700e_' + str(self.opt.leaveOut)] = []
        elif self.opt.armsup == True:
            print 'appending to armsup losses'
            self.train_val_losses['train_armsup_flip_shift_scale5_nd_nohome_700e_'+str(self.opt.leaveOut)] = []
            self.train_val_losses['val_armsup_flip_shift_scale5_nd_nohome_700e_'+str(self.opt.leaveOut)] = []
            self.train_val_losses['epoch_armsup_flip_shift_scale5_nd_nohome_700e_' + str(self.opt.leaveOut)] = []
        else:
            print 'appending to laying losses'
            self.train_val_losses['train_flip_shift_nd_nohome_1000e_'+str(self.opt.leaveOut)] = []
            self.train_val_losses['val_flip_shift_nd_nohome_1000e_'+str(self.opt.leaveOut)] = []
            self.train_val_losses['epoch_flip_shift_nd_nohome_1000e_' + str(self.opt.leaveOut)] = []






        #Here we concatenate all subjects in the training database in to one file
        dat = []
        for some_subject in training_database_file:
            dat_curr = load_pickle(some_subject)
            for inputgoalset in np.arange(len(dat_curr)):
                dat.append(dat_curr[inputgoalset])
        #dat = load_pickle(training_database_file)
        test_dat = load_pickle(test_file)

        print len(dat), len(test_dat)



       
        #TODO:Write code for the dataset to store these vals
        self.mat_size = (NUMOFTAXELS_X, NUMOFTAXELS_Y)
        self.output_size = (NUMOFOUTPUTNODES, NUMOFOUTPUTDIMS)
        #Remove empty elements from the dataset, that may be due to Motion
        #Capture issues.
        print "Checking database for empty values."
        empty_count = 0
        for entry in range(len(dat)):
            if len(dat[entry][1]) < (30) or (len(dat[entry][0]) <
                    self.mat_size[0]*self.mat_size[1]):
                empty_count += 1
                del dat[entry]
        print "Empty value check results for training set: {} rogue entries found".format(
                empty_count)


        for entry in range(len(test_dat)):
            if len(test_dat[entry][1]) < (30) or (len(test_dat[entry][0]) <
                    self.mat_size[0]*self.mat_size[1]):
                empty_count += 1
                del test_dat[entry]
        print "Empty value check results for test set: {} rogue entries found".format(empty_count)

        #Randomize the dataset entries
        dat_rand = []
        randentryset = shuffle(np.arange(len(dat)))
        for entry in range(len(dat)):
            dat_rand.append(dat[randentryset[entry]])


        rand_keys = dat
        random.shuffle(rand_keys)
        self.dataset_y = [] #Initialization for the entire dataset

        
        self.train_x_flat = [] #Initialize the training pressure mat list
        for entry in range(len(dat_rand)):
            self.train_x_flat.append(dat_rand[entry][0])
        train_x = self.preprocessing_pressure_array_resize(self.train_x_flat)
        train_x = np.array(train_x)
        train_x = self.pad_pressure_mats(train_x)
        self.train_x_tensor = torch.Tensor(train_x)


        self.train_y_flat = [] #Initialize the training ground truth list
        for entry in range(len(dat_rand)):
            self.train_y_flat.append(dat_rand[entry][1])
        #train_y = self.preprocessing_output_resize(self.train_y_flat)
        self.train_y_tensor = torch.Tensor(self.train_y_flat)
        self.train_y_tensor = torch.mul(self.train_y_tensor, 1000)
        

        self.test_x_flat = [] #Initialize the testing pressure mat list
        for entry in range(len(test_dat)):
            self.test_x_flat.append(test_dat[entry][0])
        test_x = self.preprocessing_pressure_array_resize(self.test_x_flat)
        test_x = np.array(test_x)
        test_x = self.pad_pressure_mats(test_x)
        self.test_x_tensor = torch.Tensor(test_x)

        self.test_y_flat = [] #Initialize the ground truth list
        for entry in range(len(test_dat)):
            self.test_y_flat.append(test_dat[entry][1])
        #test_y = self.preprocessing_output_resize(self.test_y_flat)
        self.test_y_tensor = torch.Tensor(self.test_y_flat)
        self.test_y_tensor = torch.mul(self.test_y_tensor, 1000)
        
        self.dataset_x_flat = self.train_x_flat#Pressure maps
        self.dataset_y = self.train_y_flat
        # [self.dataset_y.append(dat[key]) for key in self.dataset_x_flat]
        self.cv_fold = 3 # Value of k in k-fold cross validation 
        self.mat_frame_joints = []


    def pad_pressure_mats(self,NxHxWimages):
        padded = np.zeros((NxHxWimages.shape[0],NxHxWimages.shape[1]+20,NxHxWimages.shape[2]+20))
        padded[:,10:74,10:37] = NxHxWimages
        NxHxWimages = padded
        return NxHxWimages

    def chi2_distance(self, histA, histB, eps = 1e-10):
        # compute the chi-squared distance
        d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
                for (a, b) in zip(histA, histB)])
        # return the chi-squared distance
        return d


    def preprocessing_pressure_array_resize(self, data):
        '''Will resize all elements of the dataset into the dimensions of the 
        pressure map'''
        p_map_dataset = []
        for map_index in range(len(data)):
            #print map_index, self.mat_size, 'mapidx'
            #Resize mat to make into a matrix
            p_map = np.reshape(data[map_index], self.mat_size)
            p_map_dataset.append(p_map)
        if self.verbose: print len(data[0]),'x',1, 'size of an incoming pressure map'
        if self.verbose: print len(p_map_dataset[0]),'x',len(p_map_dataset[0][0]), 'size of a resized pressure map'
        return p_map_dataset

    def preprocessing_output_resize(self, data):

        p_map_dataset = []
        for map_index in range(len(data)):
            #Resize mat to make into a matrix
            p_map = np.reshape(data[map_index], self.output_size)
            p_map_dataset.append(p_map)
        if self.verbose: print len(data[0]),'x',1, 'size of an incoming output'
        if self.verbose: print len(p_map_dataset[0]),'x',len(p_map_dataset[0][0]), 'size of a resized output'
        return p_map_dataset


    def compute_pixel_variance(self, data):
        weight_matrix = np.std(data, axis=0)
        if self.verbose == True: print len(weight_matrix),'x', len(weight_matrix[0]), 'size of weight matrix'
        weight_matrix = weight_matrix/weight_matrix.max()

        x = np.zeros((20, 54))
        y = np.hstack((
                np.hstack((np.ones((60,10)), np.zeros((60, 32)))),
                np.ones((60,12))))
        z = np.ones((48, 54))
        weight_matrix = np.vstack((np.vstack((x,y)), z))
        matshow(weight_matrix, fignum=100, cmap=cm.gray)
        show()
        if self.verbose == True: print len(x),'x', len(x[0]), 'size of x matrix'
        if self.verbose == True: print len(y),'x', len(y[0]), 'size of y matrix'
        if self.verbose == True: print len(z),'x', len(z[0]), 'size of z matrix'
        return weight_matrix


    def synthetic_scale(self, images, targets):

        x = np.arange(-10,11)
        xU, xL = x + 0.5, x - 0.05
        prob = ss.norm.cdf(xU, scale=3) - ss.norm.cdf(xL, scale=3)
        prob = prob / prob.sum()  # normalize the probabilities so their sum is 1
        multiplier = np.random.choice(x, size=images.shape[0], p=prob)
        multiplier = (multiplier+100)*0.01
        #plt.hist(multiplier)
        #plt.show()

        #print multiplier
        tar_mod = np.reshape(targets, (targets.shape[0], targets.shape[1] / 3, 3))/1000

        for i in np.arange(images.shape[0]):
            #multiplier[i] = 1
            resized = zoom(images[i,:,:], multiplier[i])
            resized = np.clip(resized, 0, 100)

            rl_diff = resized.shape[1] - images[i,:,:].shape[1]
            ud_diff = resized.shape[0] - images[i,:,:].shape[0]
            l_clip = np.int(math.ceil((rl_diff) / 2))
            #r_clip = rl_diff - l_clip
            u_clip = np.int(math.ceil((ud_diff) / 2))
            #d_clip = ud_diff - u_clip

            if rl_diff < 0:  # if less than 0, we'll have to add some padding in to get back up to normal size
                resized_adjusted = np.zeros_like(images[i,:,:])
                resized_adjusted[-u_clip:-u_clip + resized.shape[0], -l_clip:-l_clip + resized.shape[1]] = np.copy(resized)
                images[i,:,:] = resized_adjusted
                shift_factor_x = INTER_SENSOR_DISTANCE * -l_clip
            elif rl_diff > 0: # if greater than 0, we'll have to cut the sides to get back to normal size
                resized_adjusted = np.copy(resized[u_clip:u_clip + images[i,:,:].shape[0], l_clip:l_clip + images[i,:,:].shape[1]])
                images[i,:,:] = resized_adjusted
                shift_factor_x = INTER_SENSOR_DISTANCE * -l_clip
            else:
                shift_factor_x = 0

            if ud_diff < 0:
                shift_factor_y = INTER_SENSOR_DISTANCE * u_clip
            elif ud_diff > 0:
                shift_factor_y = INTER_SENSOR_DISTANCE * u_clip
            else:
                shift_factor_y = 0
            #print shift_factor_y, shift_factor_x

            resized_tar = np.copy(tar_mod[i,:,:])
            #resized_tar = np.reshape(resized_tar, (len(resized_tar) / 3, 3))
            # print resized_tar.shape/
            resized_tar = (resized_tar + INTER_SENSOR_DISTANCE ) * multiplier[i]

            resized_tar[:, 0] = resized_tar[:, 0] + shift_factor_x  - 10*INTER_SENSOR_DISTANCE*(1-multiplier[i]) - INTER_SENSOR_DISTANCE
            # resized_tar2 = np.copy(resized_tar)
            resized_tar[:, 1] = resized_tar[:, 1] + 84 * (1 - multiplier[i]) * INTER_SENSOR_DISTANCE  + shift_factor_y - 10*INTER_SENSOR_DISTANCE*(1-multiplier[i]) - INTER_SENSOR_DISTANCE
            # resized_tar[7,:] = [-0.286,0,0]
            tar_mod[i,:,:] = resized_tar

        targets = np.reshape(tar_mod, (targets.shape[0], targets.shape[1]))*1000

        return images, targets




    def synthetic_shiftxy(self, images, targets):
        x = np.arange(-10, 11)
        xU, xL = x + 0.5, x - 0.5
        prob = ss.norm.cdf(xU, scale=3) - ss.norm.cdf(xL, scale=3)
        prob = prob / prob.sum()  # normalize the probabilities so their sum is 1
        modified_x = np.random.choice(x, size=images.shape[0], p=prob)
        #plt.hist(modified_x)
        #plt.show()

        y = np.arange(-10, 11)
        yU, yL = y + 0.5, y - 0.5
        prob = ss.norm.cdf(yU, scale=3) - ss.norm.cdf(yL, scale=3)
        prob = prob / prob.sum()  # normalize the probabilities so their sum is 1
        modified_y = np.random.choice(y, size=images.shape[0], p=prob)

        tar_mod = np.reshape(targets, (targets.shape[0], targets.shape[1] / 3, 3))

        #print images[0,30:34,10:14]
        #print modified_x[0]
        for i in np.arange(images.shape[0]):
            if modified_x[i] > 0:
                images[i, :, modified_x[i]:] = images[i, :, 0:-modified_x[i]]
            elif modified_x[i] < 0:
                images[i, :, 0:modified_x[i]] = images[i, :, -modified_x[i]:]

            if modified_y[i] > 0:
                images[i, modified_y[i]:,:] = images[i, 0:-modified_y[i], :]
            elif modified_y[i] < 0:
                images[i, 0:modified_y[i], :] = images[i, -modified_y[i]:, :]

            tar_mod[i, :, 0] += modified_x[i]*INTER_SENSOR_DISTANCE*1000
            tar_mod[i, :, 1] -= modified_y[i] * INTER_SENSOR_DISTANCE * 1000


        #print images[0, 30:34, 10:14]
        targets = np.reshape(tar_mod, (targets.shape[0], targets.shape[1]))

        return images, targets

    def synthetic_fliplr(self, images, targets):

        coin = np.random.randint(2, size = images.shape[0])
        modified = coin
        original = 1-coin

        im_orig = np.multiply(images,original[:,np.newaxis,np.newaxis])
        im_mod = np.multiply(images,modified[:,np.newaxis,np.newaxis])

        #flip the x axis on all the modified pressure mat images
        im_mod = im_mod[:,:,::-1]

        tar_orig = np.multiply(targets,original[:,np.newaxis])
        tar_mod = np.multiply(targets,modified[:,np.newaxis])

        #change the left and right tags on the target in the z, flip x target left to right
        tar_mod = np.reshape(tar_mod, (tar_mod.shape[0], tar_mod.shape[1] / 3, 3))

        #flip the x left to right
        tar_mod[:, :, 0] = (tar_mod[:, :, 0] -371.8) * -1 + 371.8

        #swap in the z
        dummy = zeros((tar_mod.shape))
        dummy[:,[2,4,6,8], :] = tar_mod[:, [2,4,6,8], :]
        tar_mod[:, [2,4,6,8], :] = tar_mod[:, [3,5,7,9], :]
        tar_mod[:, [3,5,7,9], :] = dummy[:, [2,4,6,8], :]
        #print dummy[0,:,2], tar_mod[0,:,2]

        tar_mod = np.reshape(tar_mod, (tar_mod.shape[0], tar_orig.shape[1]))
        tar_mod = np.multiply(tar_mod, modified[:, np.newaxis])

        images = im_orig+im_mod
        targets = tar_orig+tar_mod
        return images,targets

    def synthetic_master(self, images_tensor, targets_tensor):
        self.t1 = time.time()
        images_tensor = torch.squeeze(images_tensor)
        #images_tensor.torch.Tensor.permute(1,2,0)
        images = images_tensor.numpy()
        targets = targets_tensor.numpy()
        #print images.shape, targets.shape, 'shapes'

        images, targets = self.synthetic_scale(images, targets)
        images, targets = self.synthetic_fliplr(images,targets)
        images, targets = self.synthetic_shiftxy(images,targets)

        #print images[0, 10:15, 20:25]

        images_tensor = torch.Tensor(images)
        targets_tensor = torch.Tensor(targets)
        #images_tensor.torch.Tensor.permute(2, 0, 1)
        images_tensor = torch.unsqueeze(images_tensor,1)
        try:
            self.t2 = time.time() - self.t1
        except:
            self.t2 = 0
        #print self.t2, 'elapsed time'
        return images_tensor,targets_tensor




    def find_dataset_deviation(self):
        '''Should return the standard deviation of each joint in the (x,y,z) 
        axis'''
        return np.std(self.dataset_y, axis = 0)


    def convnet_2layer(self):
        #indices = torch.LongTensor([0])
        #self.train_y_tensor = torch.index_select(self.train_y_tensor, 1, indices)

        if self.verbose: print self.train_x_tensor.size(), 'size of the training database'
        if self.verbose: print self.train_y_tensor.size(), 'size of the training database output'
        print self.train_y_tensor
        if self.verbose: print self.test_x_tensor.size(), 'length of the testing dataset'
        if self.verbose: print self.test_y_tensor.size(), 'size of the training database output'



        batch_size = 150
        num_epochs = 700
        hidden_dim = 12
        kernel_size = 10



        self.train_x_tensor = self.train_x_tensor.unsqueeze(1)
        self.train_dataset = torch.utils.data.TensorDataset(self.train_x_tensor, self.train_y_tensor)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size, shuffle=True)

        self.test_x_tensor = self.test_x_tensor.unsqueeze(1)
        self.test_dataset = torch.utils.data.TensorDataset(self.test_x_tensor, self.test_y_tensor)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size, shuffle=True)


        self.model = convnet.CNN(self.mat_size, self.output_size, hidden_dim, kernel_size)
        self.criterion = F.cross_entropy
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.00000015, momentum=0.7, weight_decay=0.0005)

        # train the model one epoch at a time
        for epoch in range(1, num_epochs + 1):
            self.train(epoch)

        #print self.sc
        #print self.tg
        print self.sc - self.tg

        print 'done with epochs, now evaluating'
        self.evaluate('test', verbose=True)

        print self.train_val_losses, 'trainval'
        # Save the model (architecture and weights)

        if self.opt.lab_harddrive == True:
            if self.opt.sitting == True:
                torch.save(self.model,'/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_' + str(self.opt.leaveOut) + '/p_files/' + opt.trainingType + '_sitting' + '.pt')

            else:
                torch.save(self.model, '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_'+str(self.opt.leaveOut)+'/p_files/'+opt.trainingType + '.pt')
            pkl.dump(self.train_val_losses,
                     open(os.path.join('/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/train_val_losses.p'), 'wb'))


        else:
            if self.opt.sitting == True:
                torch.save(self.model,'/home/henryclever/hrl_file_server/Autobed/subject_' + str(self.opt.leaveOut) + '/p_files/' + opt.trainingType + '_sitting' + '.pt')
            else:
                torch.save(self.model, '/home/henryclever/hrl_file_server/Autobed/subject_'+str(self.opt.leaveOut)+'/p_files/'+opt.trainingType + '.pt')
            pkl.dump(self.train_val_losses,
                     open(os.path.join('/home/henryclever/hrl_file_server/Autobed/train_val_losses.p'), 'wb'))


    def train(self, epoch):
        '''
        Train the model for one epoch.
        '''
        # Some models use slightly different forward passes and train and test
        # time (e.g., any model with Dropout). This puts the model in train mode
        # (as opposed to eval mode) so it knows which one to use.
        self.model.train()
        scores = 0

        print opt.trainingType, 'opt model'

        #This will loop a total = training_images/batch_size times
        for batch_idx, batch in enumerate(self.train_loader):

            # im_sample = np.copy(batch[0].numpy())
            # im_sample = np.squeeze(im_sample[0, :])
            # tar_sample = np.copy(batch[1].numpy())
            # tar_sample = np.squeeze(tar_sample[0, :]) / 1000

            batch[0],batch[1] = self.synthetic_master(batch[0], batch[1])
            #
            # im_sample2 = np.copy(batch[0].numpy())
            # im_sample2 = np.squeeze(im_sample2[0, :])
            # tar_sample2 = np.copy(batch[1].numpy())
            # tar_sample2 = np.squeeze(tar_sample2[0, :]) / 1000
            #
            # self.visualize_pressure_map(p_map = im_sample, targets_raw=tar_sample, p_map_val=im_sample2, targets_val=tar_sample2)

            # prepare data
            sc_last = scores
            images, targets = Variable(batch[0]), Variable(batch[1])


            self.optimizer.zero_grad()

            #print images.size(), 'im size'
            #print targets.size(), 'target size'

            scores = self.model(images)

            #print scores.size(), 'scores'
            self.sc = scores
            self.tg = targets

            #print scores-sc_last

            self.criterion = nn.MSELoss()

            loss = self.criterion(scores,targets)

            loss.backward()
            self.optimizer.step()


            if batch_idx % opt.log_interval == 0:
                self.print_error(self.tg, self.sc, data = 'train')


                self.im_sample = batch[0].numpy()
                self.im_sample = np.squeeze(self.im_sample[0, :])
                self.tar_sample = batch[1].numpy()
                self.tar_sample = np.squeeze(self.tar_sample[0, :])/1000
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

                if self.opt.sitting == True:
                    print 'appending to sitting losses'
                    self.train_val_losses['train_sitting_flip_shift_scale10_700e_' + str(self.opt.leaveOut)].append(train_loss)
                    self.train_val_losses['val_sitting_flip_shift_scale10_700e_' + str(self.opt.leaveOut)].append(val_loss)
                    self.train_val_losses['epoch_sitting_flip_shift_scale10_700e_' + str(self.opt.leaveOut)].append(epoch)
                elif self.opt.armsup == True:
                    print 'appending to armsup losses'
                    self.train_val_losses['train_armsup_flip_shift_scale5_nd_nohome_700e_' + str(self.opt.leaveOut)].append(train_loss)
                    self.train_val_losses['val_armsup_flip_shift_scale5_nd_nohome_700e_' + str(self.opt.leaveOut)].append(val_loss)
                    self.train_val_losses['epoch_armsup_flip_shift_scale5_nd_nohome_700e_' + str(self.opt.leaveOut)].append(epoch)
                else:
                    self.train_val_losses['train_flip_shift_scale5_nd_nohome_700e_' + str(self.opt.leaveOut)].append(train_loss)
                    self.train_val_losses['val_flip_shift_scale5_nd_nohome_700e_' + str(self.opt.leaveOut)].append(val_loss)
                    self.train_val_losses['epoch_flip_shift_scale5_nd_nohome_700e_' + str(self.opt.leaveOut)].append(epoch)



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
            data, target = batch
            #if args.cuda:
            #    data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = self.model(data)
            loss += self.criterion(output, target).data[0]


            # predict the argmax of the log-probabilities
            pred = output.data.max(1, keepdim=True)[1]
            n_examples += pred.size(0)

            if n_batches and (batch_i >= n_batches):
                break

        loss /= n_examples
        loss *= 100

        self.print_error(target, output, data='validate')

        self.im_sampleval = data.data.numpy()
        self.im_sampleval = np.squeeze(self.im_sampleval[0, :])
        self.tar_sampleval = target.data.numpy()
        self.tar_sampleval = np.squeeze(self.tar_sampleval[0, :]) / 1000
        self.sc_sampleval = output.data.numpy()
        self.sc_sampleval = np.squeeze(self.sc_sampleval[0, :]) / 1000
        self.sc_sampleval = np.reshape(self.sc_sampleval, self.output_size)
        #self.visualize_pressure_map(self.im_sample, self.tar_sample, self.sc_sample, self.im_sampleval, self.tar_sampleval, self.sc_sampleval)



        if verbose:
            print('\n{} set: Average loss: {:.4f}\n'.format(
                split, loss))
        return loss



    def print_error(self, target, score, data = None):
        error = (score - target)
        error = error.data.numpy()
        error_avg = np.mean(error, axis=0) / 10
        error_avg = np.reshape(error_avg, self.output_size)
        error_avg = np.reshape(np.array(["%.2f" % w for w in error_avg.reshape(error_avg.size)]),
                               self.output_size)
        error_avg = np.transpose(np.concatenate(([['Average Error for Last Batch', '       ', 'Head   ',
                                                   'Torso  ', 'R Elbow', 'L Elbow', 'R Hand ', 'L Hand ',
                                                   'R Knee ', 'L Knee ', 'R Foot ', 'L Foot ']], np.transpose(
            np.concatenate(([['', '', ''], [' x, cm ', ' y, cm ', ' z, cm ']], error_avg))))))
        print data, error_avg

        error_std = np.std(error, axis=0) / 10
        error_std = np.reshape(error_std, self.output_size)
        error_std = np.reshape(np.array(["%.2f" % w for w in error_std.reshape(error_std.size)]),
                               self.output_size)
        error_std = np.transpose(
            np.concatenate(([['Error Standard Deviation for Last Batch', '       ', 'Head   ', 'Torso  ',
                              'R Elbow', 'L Elbow', 'R Hand ', 'L Hand ', 'R Knee ', 'L Knee ',
                              'R Foot ', 'L Foot ']], np.transpose(
                np.concatenate(([['', '', ''], ['x, cm', 'y, cm', 'z, cm']], error_std))))))
        print data, error_std


    def chunks(self, l, n):
        """ Yield successive n-sized chunks from l.
        """
        for i in xrange(0, len(l), n):
            yield l[i:i+n]


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
            print targets_raw
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
        plt.pause(0.0001)


        #targets_raw_z = []
        #for idx in targets_raw: targets_raw_z.append(idx[2])
        #x = np.arange(0,10)
        #ax3.bar(x, targets_raw_z)
        #plt.xticks(x+0.5, ('Head', 'Torso', 'R Elbow', 'L Elbow', 'R Hand', 'L Hand', 'R Knee', 'L Knee', 'R Foot', 'L Foot'), rotation='vertical')
        #plt.title('Distance above Bed')
        #plt.pause(0.0001)

        #plt.show()
        plt.show(block = False)

        return



    def person_based_loocv(self):
        '''Computes Person Based Leave One Out Cross Validation. This means
        that if we have 10 participants, we train using 9 participants and test
        on 1 participant, and so on.
        To run this function, make sure that each subject_* directory in the 
        dataset/ directory has a pickle file called individual_database.p
        If you don't have it in some directory that means you haven't run,
        create_raw_database.py on that subject's dataset. So create it and
        ensure that the pkl file is created successfully'''
        #Entire pressure dataset with coordinates in world frame
        dataset_dirname = os.path.dirname(os.path.realpath(training_database_file))
        print dataset_dirname
        subject_dirs = [x[0] for x in os.walk(dataset_dirname)]
        subject_dirs.pop(0)
        print subject_dirs
        dat = []
        for i in range(len(subject_dirs)):
            try:
                dat.append(pkl.load(open(os.path.join(subject_dirs[i],
                    'individual_database.p'), "rb"))) 
            except:
                print "Following dataset directory not formatted correctly. Is there an individual_dataset pkl file for every subject?"
                print os.path.join(subject_dirs[i], 'individual_database.p')
                sys.exit()
        print "Inserted all individual datasets into a list of dicts"
        print "Number of subjects:"
        print len(dat)
        mean_joint_error = np.zeros((len(dat), 10))
        std_joint_error = np.zeros((len(dat), 10))
        for i in range(len(dat)):
            train_dat = {}
            test_dat = dat[i]
            for j in range(len(dat)):
                if j == i:
                    print "#of omitted data points"
                    print len(dat[j].keys())
                    pass
                else:
                    print len(dat[j].keys())
                    print j
                    train_dat.update(dat[j])
            rand_keys = train_dat.keys()
            print "Training Dataset Size:"
            print len(rand_keys)
            print "Testing dataset size:"
            print len(test_dat.keys())
            self.train_y = [] #Initialize the training coordinate list
            self.dataset_y = [] #Initialization for the entire dataset 
            self.train_x_flat = rand_keys[:]#Pressure maps
            [self.train_y.append(train_dat[key]) for key in self.train_x_flat]#Coordinates 
            self.test_x_flat = test_dat.keys()#Pressure maps(test dataset)
            self.test_y = [] #Initialize the ground truth list
            [self.test_y.append(test_dat[key]) for key in self.test_x_flat]#ground truth
            self.dataset_x_flat = rand_keys[:]#Pressure maps
            [self.dataset_y.append(train_dat[key]) for key in self.dataset_x_flat]
            self.cv_fold = 3 # Value of k in k-fold cross validation 
            self.mat_frame_joints = []
            p.train_hog_knn()
            (mean_joint_error[i][:], std_joint_error[i][:]) = self.test_learning_algorithm(self.regr)
            print "Mean Error:"
            print mean_joint_error
        print "MEAN ERROR AFTER PERSON LOOCV:"
        total_mean_error = np.mean(mean_joint_error, axis=0)
        total_std_error = np.mean(std_joint_error, axis=0)
        print total_mean_error
        print "STD DEV:"
        print total_std_error
        pkl.dump(mean_joint_error, open('./dataset/mean_loocv_results.p', 'w'))
        pkl.dump(mean_joint_error, open('./dataset/std_loocv_results.p', 'w'))


if __name__ == "__main__":
    #Initialize trainer with a training database file
    import optparse
    p = optparse.OptionParser()
    p.add_option('--training_dataset', '--train_dataset',  action='store', type='string', \
                 dest='trainPath',\
                 default='/home/henryclever/hrl_file_server/Autobed/pose_estimation_data/basic_train_dataset.p', \
                 help='Specify path to the training database.')
    p.add_option('--training_type', '--type',  action='store', type='string', \
                 dest='trainingType',\
                 help='Specify what type of training model to use')
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
    p.add_option('--sitting', action='store_true',
                 dest='sitting', \
                 default=False, \
                 help='Set path to train on sitting data.')
    p.add_option('--armsup', action='store_true',
                 dest='armsup', \
                 default=False, \
                 help='Set path to train on larger dataset where arms and head move upwards.')
    p.add_option('--verbose', '--v',  action='store_true', dest='verbose',
                 default=False, help='Printout everything (under construction).')
    p.add_option('--log_interval', type=int, default=10, metavar='N',
                        help='number of batches between logging train status')

    opt, args = p.parse_args()

    if opt.lab_harddrive == True:

        if opt.sitting == True:
            opt.subject2Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_2/p_files/trainval_sitting_120rh_lh_rl_ll.p'
            opt.subject3Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_3/p_files/trainval_sitting_120rh_lh_rl_ll.p'
            opt.subject4Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_4/p_files/trainval_sitting_120rh_lh_rl_ll.p'
            opt.subject5Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_5/p_files/trainval_sitting_120rh_lh_rl_ll.p'
            opt.subject6Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_6/p_files/trainval_sitting_120rh_lh_rl_ll.p'
            opt.subject7Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_7/p_files/trainval_sitting_120rh_lh_rl_ll.p'
            opt.subject8Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_8/p_files/trainval_sitting_120rh_lh_rl_ll.p'

        else:
            opt.testPath = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/basic_test_dataset_select.p'
            opt.trainPath = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/basic_train_dataset_select.p'
            opt.subject1Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_1/p_files/trainval_200rh1_lh1_rl_ll.p'
            opt.subject2Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_2/p_files/trainval_200rh1_lh1_rl_ll.p'
            opt.subject3Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_3/p_files/trainval_200rh1_lh1_rl_ll.p'
            opt.subject4Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_4/p_files/trainval_200rh1_lh1_rl_ll.p'
            opt.subject5Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_5/p_files/trainval_200rh1_lh1_rl_ll.p'
            opt.subject6Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_6/p_files/trainval_200rh1_lh1_rl_ll.p'
            opt.subject7Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_7/p_files/trainval_200rh1_lh1_rl_ll.p'
            opt.subject8Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_8/p_files/trainval_200rh1_lh1_rl_ll.p'

        training_database_file = []
    else:
        if opt.sitting == True:
            opt.subject2Path = '/home/henryclever/hrl_file_server/Autobed/subject_2/p_files/trainval_sitting_120rh_lh_rl_ll.p'
            opt.subject3Path = '/home/henryclever/hrl_file_server/Autobed/subject_3/p_files/trainval_sitting_120rh_lh_rl_ll.p'
            opt.subject4Path = '/home/henryclever/hrl_file_server/Autobed/subject_4/p_files/trainval_sitting_120rh_lh_rl_ll.p'
            opt.subject5Path = '/home/henryclever/hrl_file_server/Autobed/subject_5/p_files/trainval_sitting_120rh_lh_rl_ll.p'
            opt.subject6Path = '/home/henryclever/hrl_file_server/Autobed/subject_6/p_files/trainval_sitting_120rh_lh_rl_ll.p'
            opt.subject7Path = '/home/henryclever/hrl_file_server/Autobed/subject_7/p_files/trainval_sitting_120rh_lh_rl_ll.p'
            opt.subject8Path = '/home/henryclever/hrl_file_server/Autobed/subject_8/p_files/trainval_sitting_120rh_lh_rl_ll.p'

        elif opt.armsup == True:
            opt.subject1Path = '/home/henryclever/hrl_file_server/Autobed/subject_1/p_files/trainval_200rh1_lh1_rl_ll_100rh23_lh23_head.p'
            opt.subject2Path = '/home/henryclever/hrl_file_server/Autobed/subject_2/p_files/trainval_200rh1_lh1_rl_ll_100rh23_lh23_head.p'
            opt.subject3Path = '/home/henryclever/hrl_file_server/Autobed/subject_3/p_files/trainval_200rh1_lh1_rl_ll_100rh23_lh23_head.p'
            opt.subject4Path = '/home/henryclever/hrl_file_server/Autobed/subject_4/p_files/trainval_200rh1_lh1_rl_ll_100rh23_lh23_head.p'
            opt.subject5Path = '/home/henryclever/hrl_file_server/Autobed/subject_5/p_files/trainval_200rh1_lh1_rl_ll_100rh23_lh23_head.p'
            opt.subject6Path = '/home/henryclever/hrl_file_server/Autobed/subject_6/p_files/trainval_200rh1_lh1_rl_ll_100rh23_lh23_head.p'
            opt.subject7Path = '/home/henryclever/hrl_file_server/Autobed/subject_7/p_files/trainval_200rh1_lh1_rl_ll_100rh23_lh23_head.p'
            opt.subject8Path = '/home/henryclever/hrl_file_server/Autobed/subject_8/p_files/trainval_200rh1_lh1_rl_ll_100rh23_lh23_head.p'

        else:
            opt.subject1Path = '/home/henryclever/hrl_file_server/Autobed/subject_1/p_files/trainval_200rh1_lh1_rl_ll.p'
            opt.subject2Path = '/home/henryclever/hrl_file_server/Autobed/subject_2/p_files/trainval_200rh1_lh1_rl_ll.p'
            opt.subject3Path = '/home/henryclever/hrl_file_server/Autobed/subject_3/p_files/trainval_200rh1_lh1_rl_ll.p'
            opt.subject4Path = '/home/henryclever/hrl_file_server/Autobed/subject_4/p_files/trainval_200rh1_lh1_rl_ll.p'
            opt.subject5Path = '/home/henryclever/hrl_file_server/Autobed/subject_5/p_files/trainval_200rh1_lh1_rl_ll.p'
            opt.subject6Path = '/home/henryclever/hrl_file_server/Autobed/subject_6/p_files/trainval_200rh1_lh1_rl_ll.p'
            opt.subject7Path = '/home/henryclever/hrl_file_server/Autobed/subject_7/p_files/trainval_200rh1_lh1_rl_ll.p'
            opt.subject8Path = '/home/henryclever/hrl_file_server/Autobed/subject_8/p_files/trainval_200rh1_lh1_rl_ll.p'

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

    else:
        print 'please specify which subject to leave out for validation using --leave_out _'



    print opt.testPath, 'testpath'
    print opt.modelPath, 'modelpath'



    training_type = opt.trainingType #Type of algorithm you want to train with
    test_bool = opt.only_test#Whether you want only testing done


    print test_bool, 'test_bool'
    print training_type, 'training_type'

    p = PhysicalTrainer(training_database_file, test_database_file, opt)

    if test_bool == True:
        trained_model = load_pickle(opt.modelPath+'/'+training_type+'.p')#Where the trained model is 
        p.test_learning_algorithm(trained_model)
        sys.exit()
    else:
        if opt.verbose == True: print 'Beginning Learning'

        if training_type == 'HoG_KNN':
            #p.person_based_loocv()
            regr = p.train_hog_knn()
            #if opt.testPath is not None:
                #p.test_learning_algorithm(regr)
#                sys.exit()
            sys.exit()


        if training_type == 'convnet_2':
            p.convnet_2layer()

        else:
            print 'Please specify correct training type:1. HoG_KNN 2. convnet_2'
