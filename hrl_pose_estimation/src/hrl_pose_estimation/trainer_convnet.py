#!/usr/bin/env python
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import *

import cPickle as pkl
import random
from scipy import ndimage
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


    def compute_HoG(self, data):
        '''Computes a HoG(Histogram of gradients for a list of images provided
        to it. Returns a list of flattened lists'''
        flat_hog = []
        if self.verbose==True: print np.shape(data), 'Size of dataset we are performing HoG on'
        print "*****Begin HoGing the dataset*****"
        for index in range(len(data)):
            print "HoG being applied over image number: {}".format(index)
            #Compute HoG of the current pressure map
            fd, hog_image = hog(data[index], orientations=8, 
                    pixels_per_cell=(4,4), cells_per_block = (1, 1), 
                    visualise=True)
            flat_hog.append(fd) 
            #self.visualize_pressure_map(data[index])
            #print hog_image[30], data[index][30]
   
        return flat_hog


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


    def preprocessing_pressure_map_upsample(self, data, multiple=2, order=1):
        '''Will upsample an incoming pressure map dataset'''
        p_map_highres_dataset = []
        for map_index in range(len(data)):
            #Upsample the current map using bilinear interpolation
            p_map_highres_dataset.append(
                    ndimage.zoom(data[map_index], multiple, order=order))
        if self.verbose: print len(p_map_highres_dataset[0]),'x',len(p_map_highres_dataset[0][0]), 'size of an upsampled pressure map'
        return p_map_highres_dataset

 

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
        num_epochs = 500
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

        # Save the model (architecture and weights)
        torch.save(self.model, opt.trainingType + '.pt')

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

    def train_hog_knn(self):
        '''Runs training on the dataset using the Upsample+ HoG+
        + K Nearest Neighbor Regression technique'''
        #Number of neighbors
        n_neighbors = 5
        #Resize incoming pressure map
        pressure_map_dataset_lowres_train = (
            self.preprocessing_pressure_array_resize(self.dataset_x_flat))
        #Upsample the lowres training dataset 
        pressure_map_dataset_highres_train = (
            self.preprocessing_pressure_map_upsample(
                pressure_map_dataset_lowres_train))

        self.weight_vector = (self.compute_pixel_variance(pressure_map_dataset_highres_train))

        weighted_p_map = np.multiply(pressure_map_dataset_highres_train,
                self.weight_vector)

        if self.verbose==True: print np.shape(weighted_p_map)[0], np.shape(weighted_p_map)[1], np.shape(weighted_p_map)[2], 'dims of weighted_p_map'


        weighted_dataset_flat = np.zeros((np.shape(weighted_p_map)[0],
                np.shape(weighted_p_map)[1]*np.shape(weighted_p_map)[2]))

        if self.verbose == True: print np.shape(weighted_dataset_flat)[0],np.shape(weighted_dataset_flat)[1], 'dims of flattened weighted_p_map'

        #here enumerate the weighted_dataset_flat with flattened entries of weighted_p_map
        for i in range(np.shape(weighted_p_map)[0]): #i goes from 0 to whatever number of pressure mat readings we have
            curr_map = weighted_p_map[i][:][:]
            weighted_dataset_flat[i][:] = np.reshape(curr_map,
                    (np.shape(curr_map)[0]*np.shape(curr_map)[1]))


        #p_map_normalized = self.normalize_data(
                #pressure_map_dataset_highres_train)
        #Compute HoG of the current(training) pressure map dataset
        pressure_hog_train = (self.compute_HoG(pressure_map_dataset_highres_train))

        #OPTIONAL: PCA STAGE
        #X = self.pca_pressure_map( self.train_y, False)
        #Now we train a Ridge regression on the dataset of HoGs
        #self.regr = neighbors.KNeighborsRegressor(n_neighbors,
        #weights='distance')
        #self.regr = (neighbors.KNeighborsRegressor(n_neighbors,
        #weights='distance', metric='pyfunc', func=self.chi2_distance))
        
        self.regr = (neighbors.KNeighborsRegressor(n_neighbors,
                weights='distance', metric='pyfunc'))
       
        print "JOINT STANDARD DEVIATION"
        joint_std_dev = self.find_dataset_deviation()
        print joint_std_dev
        sys.exit()
        scores = cross_validation.cross_val_score(
                self.regr, weighted_dataset_flat, self.dataset_y, cv=self.cv_fold)
        print("Accuracy after k-fold cross validation: %0.2f (+/- %0.2f)" 
                % (scores.mean(), scores.std() * 2))

        predicted = cross_validation.cross_val_predict(
                self.regr, np.asarray(pressure_hog_train), self.dataset_y, cv=self.cv_fold)
        ##Mean Squared Error
        print("Residual sum of squares: %.8f"
              % np.mean((predicted - self.dataset_y) **2))
        diff = predicted - self.dataset_y
        mean_error = np.mean(np.linalg.norm(diff, axis = 1))
        joint_error = np.zeros([np.shape(diff)[0],np.shape(diff)[1]/3])
        for i in range(np.shape(diff)[0]):
            diff_row = diff[i][:].reshape(np.shape(diff)[1]/3, 3)
            joint_error[i][:] = np.linalg.norm(diff_row, axis = 1)
        mean_joint_error =  np.mean(joint_error, axis = 0)
        std_joint_error = np.std(joint_error, axis = 0)
        print "MEAN JOINT ERROR AFTER {} FOLD CV".format(self.cv_fold)
        print mean_joint_error
        print "STANDARD DEVIATION AFTER {} FOLD CV".format(self.cv_fold) 
        print std_joint_error
        try:
            total_mean_error = pkl.load(open('./dataset/total_mean_error.p', 'rb'))
        except:
            total_mean_error = np.asarray(mean_joint_error)
            total_std_dev = np.asarray(std_joint_error)
            pkl.dump(total_mean_error, open('./dataset/total_mean_error.p', 'wb'))
            pkl.dump(total_std_dev, open('./dataset/total_std_dev.p', 'wb'))
            sys.exit()

        total_std_dev = pkl.load(open('./dataset/total_std_dev.p', 'rb'))
        total_mean_error = np.vstack((total_mean_error,
                                                np.asarray(mean_joint_error)))
        total_std_dev = np.vstack((total_std_dev, std_joint_error))
        pkl.dump(total_mean_error, open('./dataset/total_mean_error.p', 'wb'))
        pkl.dump(total_std_dev, open('./dataset/total_std_dev.p', 'wb'))
        ## Train the model using the training sets
        self.regr.fit(weighted_dataset_flat, self.dataset_y)
        #Pickle the trained model
        #pkl.dump(self.regr, open('./dataset/trained_model_'+'HoG_KNN.p', 'wb'))
        return self.regr 


    def test_learning_algorithm(self, trained_model):
        '''Tests the learning algorithm we're trying to implement'''
        test_x_lowres = (
            self.preprocessing_pressure_array_resize(self.test_x_flat))
        #Upsample the current map using bilinear interpolation
        test_x_highres = self.preprocessing_pressure_map_upsample(
                test_x_lowres)

        self.weight_vector = (self.compute_pixel_variance(test_x_highres))

        weighted_p_map = np.multiply(test_x_highres,
                self.weight_vector)
        weighted_dataset_flat = np.zeros((np.shape(weighted_p_map)[0],
                np.shape(weighted_p_map)[1]*np.shape(weighted_p_map)[2]))

        for i in range(np.shape(weighted_p_map)[0]):
            curr_map = weighted_p_map[i][:][:]
            weighted_dataset_flat[i][:] = np.reshape(curr_map,
                    (np.shape(curr_map)[0]*np.shape(curr_map)[1]))

        #Compute HoG of the current(test) pressure map dataset
        test_hog = self.compute_HoG(test_x_highres)

        #Load training model
        regr = trained_model

        # The coefficients
        try:
            print('Coefficients: \n', regr.coef_)
        except AttributeError:
            pass

        print np.shape(regr.coef_), 'size of regression coefficients'
        print np.shape(weighted_dataset_flat), 'size of weighted flat test dataset'
        print np.shape(test_hog), 'size of the HOG of the test set'
        print np.shape(self.test_y), 'size of test y matrix'

        # The mean square error
        #diff = regr.predict(weighted_dataset_flat) - self.test_y
        diff = regr.predict(test_hog) - self.test_y

#        print "Max absolute distance in each axis"
        #mean_indiv_error = np.ndarray.max(np.absolute(diff), axis = 0)
        #mean_indiv_error = mean_indiv_error.reshape(np.shape(mean_indiv_error)[0]/3, 3)
        #std_indiv_error = np.std(np.absolute(diff), axis = 0)
        #print mean_indiv_error
        #print "Std dev in absolute distance"
        #print std_indiv_error
        #sys.exit()
        mean_error = np.mean(np.linalg.norm(diff, axis = 1))
        joint_error = np.zeros([np.shape(diff)[0],np.shape(diff)[1]/3])
        for i in range(np.shape(diff)[0]):
            diff_row = diff[i][:].reshape(np.shape(diff)[1]/3, 3)
            joint_error[i][:] = np.linalg.norm(diff_row, axis = 1)
        mean_joint_error =  np.mean(joint_error, axis = 0)
        std_joint_error = np.std(joint_error, axis = 0)
        # Explained variance score: 1 is perfect prediction
        #print("Residual sum of squares: %.8f"
              #% np.mean((regr.predict(test_hog) - self.test_y) **2))
        #print('Variance score: %.8f' % regr.score(test_hog, self.test_y))
        #print ('Mean Euclidian Error (meters): %.8f' % mean_error)
        print 'Mean Joint Error (meters):' 
        print mean_joint_error
        print ('Head, Torso, Elbows, Hands, Knees, Ankles')
        print "Variance Joint Error"
        print std_joint_error
        print ('Head, Torso, Elbows, Hands, Knees, Ankles')
        ##Plot n test poses at random
        #estimated_y = regr.predict(weighted_dataset_flat)
        #distances, indices = regr.kneighbors(weighted_dataset_flat)
        estimated_y = regr.predict(test_hog)
        distances, indices = regr.kneighbors(test_hog)
        train_x_lowres = self.preprocessing_pressure_array_resize(self.train_x_flat)

        for i in range(np.shape(indices)[0]):
            taxel_real = []
            plt.subplot(161)
            [taxel_real.append(self.mat_to_taxels(item)) for item in (
                list(self.chunks(self.test_y[i], 3)))]
            for item in taxel_real:
                test_x_lowres[i][(NUMOFTAXELS_X-1) - item[1], item[0]] = 300
            self.visualize_pressure_map(test_x_lowres[i])

            taxel_real = []
            plt.subplot(162)
            [taxel_real.append(self.mat_to_taxels(item)) for item in (
                list(self.chunks(self.train_y[indices[i][0]], 3)))]
            for item in taxel_real:
                train_x_lowres[indices[i][0]][(NUMOFTAXELS_X-1) - item[1], item[0]] = 300
            self.visualize_pressure_map(train_x_lowres[indices[i][0]])

            taxel_real = []
            plt.subplot(163)
            [taxel_real.append(self.mat_to_taxels(item)) for item in (
                list(self.chunks(self.train_y[indices[i][1]], 3)))]
            for item in taxel_real:
                train_x_lowres[indices[i][1]][(NUMOFTAXELS_X-1) - item[1], item[0]] = 300
            self.visualize_pressure_map(train_x_lowres[indices[i][1]])

            taxel_real = []
            plt.subplot(164)
            [taxel_real.append(self.mat_to_taxels(item)) for item in (
                list(self.chunks(self.train_y[indices[i][2]], 3)))]
            for item in taxel_real:
                train_x_lowres[indices[i][2]][(NUMOFTAXELS_X-1) - item[1], item[0]] = 300
            self.visualize_pressure_map(train_x_lowres[indices[i][2]])

            taxel_real = []
            plt.subplot(165)
            [taxel_real.append(self.mat_to_taxels(item)) for item in (
                list(self.chunks(self.train_y[indices[i][3]], 3)))]
            for item in taxel_real:
                train_x_lowres[indices[i][3]][(NUMOFTAXELS_X-1) - item[1], item[0]] = 300
            self.visualize_pressure_map(train_x_lowres[indices[i][3]])

            taxel_real = []
            plt.subplot(166)
            [taxel_real.append(self.mat_to_taxels(item)) for item in (
                list(self.chunks(self.train_y[indices[i][4]], 3)))]
            for item in taxel_real:
                train_x_lowres[indices[i][4]][(NUMOFTAXELS_X-1) - item[1], item[0]] = 300
            self.visualize_pressure_map(train_x_lowres[indices[i][4]])
            #plt.show()
#        plt.subplot(131)
        #taxel_est = []
        #taxel_real = []
        #img = random.randint(1, len(test_x_lowres)-1)
        #for item in (list(self.chunks(estimated_y[img], 3))):
            #print item

        #[taxel_est.append(self.mat_to_taxels(item)) for item in (
           #list(self.chunks(estimated_y[img], 3)))]
        #for item in taxel_est:
            #test_x_lowres[img][(NUMOFTAXELS_X-1) - item[1], item[0]] = 200
        #[taxel_real.append(self.mat_to_taxels(item)) for item in (
            #list(self.chunks(self.test_y[img], 3)))]
        #for item in taxel_real:
            #test_x_lowres[img][(NUMOFTAXELS_X-1) - item[1], item[0]] = 300
        #self.visualize_pressure_map(test_x_lowres[img])
        
        #plt.subplot(132)
        #taxel_est = []
        #taxel_real = []
        #img = random.randint(1, len(test_x_lowres)-1)
        #[taxel_est.append(self.mat_to_taxels(item)) for item in (
            #list(self.chunks(estimated_y[img], 3)))]
        #for item in taxel_est:
            #test_x_lowres[img][(NUMOFTAXELS_X-1) - item[1], item[0]] = 200
        #[taxel_real.append(self.mat_to_taxels(item)) for item in (
            #list(self.chunks(self.test_y[img], 3)))]
        #for item in taxel_real:
            #print item
            #test_x_lowres[img][(NUMOFTAXELS_X - 1) - item[1], item[0]] = 300
        #self.visualize_pressure_map(test_x_lowres[img])

        #plt.subplot(133)
        #taxel_est = []
        #taxel_real = []
        #img = random.randint(1, len(test_x_lowres)-1)
        #[taxel_est.append(self.mat_to_taxels(item)) for item in (list(self.chunks(estimated_y[img], 3)))]
        #for item in taxel_est:
            #test_x_lowres[img][(NUMOFTAXELS_X-1) - item[1], item[0]] = 200
        #[taxel_real.append(self.mat_to_taxels(item)) for item in (
            #list(self.chunks(self.test_y[img], 3)))]
        #for item in taxel_real:
            #test_x_lowres[img][(NUMOFTAXELS_X-1) - item[1], item[0]] = 300
        #self.visualize_pressure_map(test_x_lowres[img])
        #plt.show()
        return mean_joint_error, std_joint_error



    def chunks(self, l, n):
        """ Yield successive n-sized chunks from l.
        """
        for i in xrange(0, len(l), n):
            yield l[i:i+n]


    def pca_pressure_map(self, data, visualize = True):
        '''Computing the 3D PCA of the dataset given to it. If visualize is set
        to True, we can also visualize the output of this function'''
        X = data
        if visualize:
            fig = plt.figure(1, figsize=(4, 3))
            plt.clf()
            ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
            plt.cla()

        pca = decomposition.PCA(n_components=3)
        pca.fit(X)
        X = pca.transform(X)
        if visualize:
            ax.scatter(X[:, 0], X[:, 1], X[:, 2], cmap=plt.cm.spectral)
            x_surf = [X[:, 0].min(), X[:, 0].max(),
                              X[:, 0].min(), X[:, 0].max()]
            y_surf = [X[:, 0].max(), X[:, 0].max(),
                              X[:, 0].min(), X[:, 0].min()]
            x_surf = np.array(x_surf)
            y_surf = np.array(y_surf)
            v0 = pca.transform(pca.components_[0])
            v0 /= v0[-1]
            v1 = pca.transform(pca.components_[1])
            v1 /= v1[-1]

            ax.w_xaxis.set_ticklabels([])
            ax.w_yaxis.set_ticklabels([])
            ax.w_zaxis.set_ticklabels([])
            plt.show()
        return X


    def mat_to_taxels(self, m_data):
        ''' 
        Input:  Nx2 array 
        Output: Nx2 array
        '''       
        #Convert coordinates in 3D space in the mat frame into taxels
        taxels = np.asarray(m_data) / INTER_SENSOR_DISTANCE
        '''Typecast into int, so that we can highlight the right taxel 
        in the pressure matrix, and threshold the resulting values'''
        taxels = np.rint(taxels)
        #Thresholding the taxels_* array
        if taxels[1] < LOW_TAXEL_THRESH_X: taxels[1] = LOW_TAXEL_THRESH_X
        if taxels[0] < LOW_TAXEL_THRESH_Y: taxels[0] = LOW_TAXEL_THRESH_Y
        if taxels[1] > HIGH_TAXEL_THRESH_X: taxels[1] = HIGH_TAXEL_THRESH_X
        if taxels[0] > HIGH_TAXEL_THRESH_Y: taxels[0] = HIGH_TAXEL_THRESH_Y
        return taxels


    def visualize_pressure_map(self, p_map_raw, targets_raw=None, scores_raw = None, p_map_val = None, targets_val = None, scores_val = None):
        print p_map_raw.shape, 'pressure mat size'

        plt.close()
        plt.pause(0.0001)
        p_map = np.asarray(np.reshape(p_map_raw, self.mat_size))
        p_map_val = np.asarray(np.reshape(p_map_val, self.mat_size))
        fig = plt.figure()
        mngr = plt.get_current_fig_manager()
        # to put it into the upper left corner for example:
        mngr.window.setGeometry(50, 100, 840, 705)

        plt.pause(0.0001)

        # set options
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)


        xlim = [-10.0, 35.0]
        ylim = [70.0, -10.0]
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
            if type(targets_raw) == list:
                targets_raw = np.array(targets_raw)
            if len(np.shape(targets_raw)) == 1:
                targets_raw = np.reshape(targets_raw, (len(targets_raw) / 3, 3))
            target_coord = targets_raw[:, :2] / INTER_SENSOR_DISTANCE
            target_coord[:, 1] -= (NUMOFTAXELS_X - 1)
            target_coord[:, 1] *= -1.0
            ax1.plot(target_coord[:, 0], target_coord[:, 1], 'y*', ms=8)
        plt.pause(0.0001)

        #Visualize estimated from training set
        if scores_raw is not None:
            if type(scores_raw) == list:
                scores_raw = np.array(scores_raw)
            if len(np.shape(scores_raw)) == 1:
                scores_raw = np.reshape(scores_raw, (len(scores_raw) / 3, 3))
            target_coord = scores_raw[:, :2] / INTER_SENSOR_DISTANCE
            target_coord[:, 1] -= (NUMOFTAXELS_X - 1)
            target_coord[:, 1] *= -1.0
            ax1.plot(target_coord[:, 0], target_coord[:, 1], 'g*', ms=8)
        ax1.set_title('Training Sample \n Targets and Estimates')
        plt.pause(0.0001)

        # Visualize targets of validation set
        if targets_val is not None:
            if type(targets_val) == list:
                targets_val = np.array(targets_val)
            if len(np.shape(targets_val)) == 1:
                targets_val = np.reshape(targets_val, (len(targets_val) / 3, 3))
            target_coord = targets_val[:, :2] / INTER_SENSOR_DISTANCE
            target_coord[:, 1] -= (NUMOFTAXELS_X - 1)
            target_coord[:, 1] *= -1.0
            ax2.plot(target_coord[:, 0], target_coord[:, 1], 'y*', ms=8)
        plt.pause(0.0001)

        # Visualize estimated from training set
        if scores_val is not None:
            if type(scores_val) == list:
                scores_val = np.array(scores_val)
            if len(np.shape(scores_val)) == 1:
                scores_val = np.reshape(scores_val, (len(scores_val) / 3, 3))
            target_coord = scores_val[:, :2] / INTER_SENSOR_DISTANCE
            target_coord[:, 1] -= (NUMOFTAXELS_X - 1)
            target_coord[:, 1] *= -1.0
            ax2.plot(target_coord[:, 0], target_coord[:, 1], 'g*', ms=8)
        ax2.set_title('Validation Sample \n Targets and Estimates')
        plt.pause(0.0001)


        #targets_raw_z = []
        #for idx in targets_raw: targets_raw_z.append(idx[2])
        #x = np.arange(0,10)
        #ax3.bar(x, targets_raw_z)
        #plt.xticks(x+0.5, ('Head', 'Torso', 'R Elbow', 'L Elbow', 'R Hand', 'L Hand', 'R Knee', 'L Knee', 'R Foot', 'L Foot'), rotation='vertical')
        #plt.title('Distance above Bed')
        #plt.pause(0.0001)


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
    p.add_option('--verbose', '--v',  action='store_true', dest='verbose',
                 default=False, help='Printout everything (under construction).')
    p.add_option('--log_interval', type=int, default=10, metavar='N',
                        help='number of batches between logging train status')

    opt, args = p.parse_args()

    if opt.lab_harddrive == True:
        opt.testPath = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/basic_test_dataset_select.p'
        opt.trainPath = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/basic_train_dataset_select.p'
        opt.subject4Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_4_dataset.p'
        opt.subject9Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_9_dataset.p'
        opt.subject10Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_10_dataset.p'
        opt.subject11Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_11_dataset.p'
        opt.subject12Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_12_dataset.p'
        opt.subject13Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_13_dataset.p'
        opt.subject14Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_14_dataset.p'
        opt.subject15Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_15_dataset.p'
        opt.subject16Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_16_dataset.p'
        opt.subject17Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_17_dataset.p'
        opt.subject18Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_18_dataset.p'

        training_database_file = []
        if opt.leaveOut == 4:
            test_database_file = opt.subject4Path
            training_database_file.append(opt.subject9Path)
            training_database_file.append(opt.subject10Path)
            training_database_file.append(opt.subject11Path)
            training_database_file.append(opt.subject12Path)
            training_database_file.append(opt.subject13Path)
            training_database_file.append(opt.subject14Path)
            training_database_file.append(opt.subject15Path)
            training_database_file.append(opt.subject16Path)
            training_database_file.append(opt.subject17Path)
            training_database_file.append(opt.subject18Path)

        elif opt.leaveOut == 9:
            test_database_file = opt.subject9Path
            training_database_file.append(opt.subject4Path)
            training_database_file.append(opt.subject10Path)
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
    else:
        training_database_file = opt.trainPath #Where is the training database is

        if opt.testPath is None:
            test_database_file = opt.testPath#Make it the same as train dataset
        else:
            test_database_file = opt.testPath#Where the test dataset is





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
