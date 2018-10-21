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
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.multioutput import MultiOutputRegressor

#keras stuff
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input


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

torch.set_num_threads(1)
if torch.cuda.is_available():
    # Use for GPU
    dtype = torch.cuda.FloatTensor
    print '######################### CUDA is available! #############################'
else:
    # Use for CPU
    dtype = torch.FloatTensor
    print '############################## USING CPU #################################'

class PhysicalTrainer():
    '''Gets the dictionary of pressure maps from the training database,
    and will have API to do all sorts of training with it.'''
    def __init__(self, training_database_file, test_database_file, opt, name_prefix, subject):
        '''Opens the specified pickle files to get the combined dataset:
        This dataset is a dictionary of pressure maps with the corresponding
        3d position and orientation of the markers associated with it.'''


        #change this to 'direct' when you are doing baseline methods
        self.loss_vector_type = opt.losstype

        self.verbose = opt.verbose
        self.opt = opt
        self.batch_size = 1130


        self.num_epochs = 1
        self.include_inter = True
        self.shuffle = True

        self.count = 0


        print test_database_file
        print self.num_epochs, 'NUM EPOCHS!'
        #Entire pressure dataset with coordinates in world frame


        self.mat_size = (NUMOFTAXELS_X, NUMOFTAXELS_Y)
        self.output_size = (NUMOFOUTPUTNODES, NUMOFOUTPUTDIMS)

        self.imagenet_model = VGG16(weights='imagenet', include_top=False)




        print 'loading training files'
        #load in the training files.  This may take a while.
        for some_subject in training_database_file:
            print some_subject
            dat_curr = load_pickle(some_subject)
            for key in dat_curr:
                print key
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

        print np.shape(dat['images'])
        print len(dat['images'])


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
            c = np.concatenate((dat['markers_xyz_m'][entry][0:30] * 1000,
                                dat['joint_lengths_U_m'][entry][0:9] * 100,
                                dat['joint_angles_U_deg'][entry][0:10],
                                dat['joint_lengths_L_m'][entry][0:8] * 100,
                                dat['joint_angles_L_deg'][entry][0:8]), axis=0)
            self.train_y_flat.append(c)

        self.train_y_tensor = torch.Tensor(self.train_y_flat)

        print 'loading testing files'
        #load in the test file
        for some_subject in test_database_file:
            print some_subject
            dat_curr = load_pickle(some_subject)
            for key in dat_curr:
                if np.array(dat_curr[key]).shape[0] != 0:
                    for inputgoalset in np.arange(len(dat_curr['images'])):
                        try:
                            test_dat[key].append(dat_curr[key][inputgoalset])
                        except:
                            try:
                                test_dat[key] = []
                                test_dat[key].append(dat_curr[key][inputgoalset])
                            except:
                                test_dat = {}
                                test_dat[key] = []
                                test_dat[key].append(dat_curr[key][inputgoalset])



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
            c = np.concatenate((test_dat['markers_xyz_m'][entry][0:30] * 1000,
                                test_dat['joint_lengths_U_m'][entry][0:9] * 100,
                                test_dat['joint_angles_U_deg'][entry][0:10],
                                test_dat['joint_lengths_L_m'][entry][0:8] * 100,
                                test_dat['joint_angles_L_deg'][entry][0:8]), axis=0)
            self.test_y_flat.append(c)

        self.test_y_flat = np.array(self.test_y_flat)
        self.test_y_tensor = torch.Tensor(self.test_y_flat)



        #self.train_x_tensor = self.train_x_tensor.unsqueeze(1)
        self.train_dataset = torch.utils.data.TensorDataset(self.train_x_tensor, self.train_y_tensor)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, self.batch_size, shuffle=True)

        #self.test_x_tensor = self.test_x_tensor.unsqueeze(1)
        self.test_dataset = torch.utils.data.TensorDataset(self.test_x_tensor, self.test_y_tensor)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, self.batch_size, shuffle=True)



        for batch_idx, batch in enumerate(self.train_loader):
            # append upper joint angles, lower joint angles, upper joint lengths, lower joint lengths, in that order
            batch.append(torch.cat((batch[1][:, 39:49], batch[1][:, 57:65], batch[1][:, 30:39], batch[1][:, 49:57]), dim=1))

            # get the whole body x y z
            batch[1] = batch[1][:, 0:30]

            batch[0], batch[1], batch[2] = SyntheticLib().synthetic_master(batch[0], batch[1], batch[2], flip=True,
                                                                           shift=True, scale=True, bedangle=True,
                                                                           include_inter=self.include_inter,
                                                                           loss_vector_type=self.loss_vector_type)

            images_up_non_tensor = PreprocessingLib().preprocessing_add_image_noise(np.array(PreprocessingLib().preprocessing_pressure_map_upsample(batch[0].numpy()[:, :, 10:74, 10:37])))

            #prior = time.time()
            vgg_images_up_non_tensor = preprocess_input(np.transpose(images_up_non_tensor, (0,2,3,1)))
            vgg16_image_features_non_tensor = self.imagenet_model.predict(vgg_images_up_non_tensor)

            print vgg16_image_features_non_tensor.shape, 'features shape train'


            print np.array(batch[1][0,:]).shape
            print type(np.array(batch[1][0,:]))
            print np.array(batch[2]).shape

            dat['features'] = []

            for entry in range(len(dat['images'])):
                dat['features'].append(vgg16_image_features_non_tensor[entry, :, :, :])

                dat['markers_xyz_m'][entry] = np.array(batch[1][entry,:]) / 1000

                dat['joint_lengths_U_m'][entry] = np.array(batch[2][entry,18:27]) / 100
                dat['joint_angles_U_deg'][entry] = np.array(batch[2][entry,0:10])
                dat['joint_lengths_L_m'][entry] = np.array(batch[2][entry,27:35]) / 100
                dat['joint_angles_L_deg'][entry] = np.array(batch[2][entry,10:18])



            del dat['images']
            del dat['pseudomarkers_xyz_m']
            del dat['marker_bed_euclideans_m']

            for key in dat:
                print 'training set: ', key, np.array(dat[key]).shape

            pkl.dump(dat, open(os.path.join(name_prefix+'/subject_'+str(subject) + '/p_files/trainfeat4xup_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p'), 'wb'))  # _trainval_200rlh1_115rlh2_75rlh3_175rllair_sit175rlh_sit120rll #200rlh1_115rlh2_75rlh3_175rllair_sit175rlh_sit120rll

            #pkl.dump(dat,open(os.path.join('/home/henryclever/test/trainfeat6xup_s8_150rh1_sit120rh.p'),'wb'))  # _trainval_200rlh1_115rlh2_75rlh3_175rllair_sit175rlh_sit120rll #200rlh1_115rlh2_75rlh3_175rllair_sit175rlh_sit120rll

            print 'done saving training set'



        for batch_idx, batch in enumerate(self.test_loader):
            # append upper joint angles, lower joint angles, upper joint lengths, lower joint lengths, in that order
            batch.append(torch.cat((batch[1][:, 39:49], batch[1][:, 57:65], batch[1][:, 30:39], batch[1][:, 49:57]), dim=1))

            # get the whole body x y z
            batch[1] = batch[1][:, 0:30]

            batch[0], batch[1], batch[2] = SyntheticLib().synthetic_master(batch[0], batch[1], batch[2], flip=False,
                                                                           shift=False, scale=False, bedangle=True,
                                                                           include_inter=self.include_inter,
                                                                           loss_vector_type=self.loss_vector_type)

            images_up_non_tensor = np.array(PreprocessingLib().preprocessing_pressure_map_upsample(batch[0].numpy()[:, :, 10:74, 10:37]))

            prior = time.time()
            vgg_images_up_non_tensor = preprocess_input(np.transpose(images_up_non_tensor, (0,2,3,1)))
            vgg16_image_features_non_tensor = self.imagenet_model.predict(vgg_images_up_non_tensor)

            print vgg16_image_features_non_tensor.shape, 'features shape test'

            print np.array(batch[1][0,:]).shape
            print type(np.array(batch[1][0,:]))
            print np.array(batch[2]).shape

            test_dat['features'] = []

            for entry in range(len(test_dat['images'])):
                test_dat['features'].append(vgg16_image_features_non_tensor[entry, :, :, :])

                test_dat['markers_xyz_m'][entry] = np.array(batch[1][entry,:]) / 1000

                test_dat['joint_lengths_U_m'][entry] = np.array(batch[2][entry,18:27]) / 100
                test_dat['joint_angles_U_deg'][entry] = np.array(batch[2][entry,0:10])
                test_dat['joint_lengths_L_m'][entry] = np.array(batch[2][entry,27:35]) / 100
                test_dat['joint_angles_L_deg'][entry] = np.array(batch[2][entry,10:18])



            del test_dat['images']
            del test_dat['pseudomarkers_xyz_m']
            del test_dat['marker_bed_euclideans_m']

            for key in test_dat:
                print 'testing set: ', key, np.array(test_dat[key]).shape

            pkl.dump(test_dat, open(os.path.join(name_prefix+'/subject_'+str(subject) + '/p_files/testfeat4xup_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p'), 'wb'))  # _trainval_200rlh1_115rlh2_75rlh3_175rllair_sit175rlh_sit120rll #200rlh1_115rlh2_75rlh3_175rllair_sit175rlh_sit120rll

            #pkl.dump(test_dat, open(os.path.join('/home/henryclever/test/testfeat6xup_s4_150rh1_sit120rh.p'), 'wb'))  # _trainval_200rlh1_115rlh2_75rlh3_175rllair_sit175rlh_sit120rll #200rlh1_115rlh2_75rlh3_175rllair_sit175rlh_sit120rll


if __name__ == "__main__":
    #Initialize trainer with a training database file
    import optparse
    p = optparse.OptionParser()
    p.add_option('--training_dataset', '--train_dataset',  action='store', type='string', \
                 dest='trainPath',\
                 default='/home/henryclever/hrl_file_server/Autobed/pose_estimation_data/basic_train_dataset.p', \
                 help='Specify path to the training database.')
    p.add_option('--subject', action='store', type=int, \
                 dest='subject', \
                 help='Specify which subject')
    p.add_option('--only_test','--t',  action='store_true', dest='only_test',
                 default=False, help='Whether you want only testing of previously stored model')
    p.add_option('--testing_dataset', '--test_dataset',  action='store', type='string', \
                 dest='testPath',\
                 default='/home/henryclever/hrl_file_server/Autobed/pose_estimation_data/basic_test_dataset.p', \
                 help='Specify path to the training database.')
    p.add_option('--computer', action='store', type = 'string',
                 dest='computer', \
                 default='lab_harddrive', \
                 help='Set path to the training database on lab harddrive.')
    p.add_option('--gpu', action='store', type = 'string',
                 dest='gpu', \
                 default='0', \
                 help='Set the GPU you will use.')
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
        name_prefix = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials'
        name_prefix_qt = '/home/henryclever/test'

    elif opt.computer == 'aws':
        name_prefix = '/home/ubuntu/Autobed_OFFICIAL_Trials'
        name_prefix_qt = '/home/ubuntu/test'

    elif opt.computer == 'baymax':
        name_prefix = '/home/henryclever/IROS_Data'
        name_prefix_qt = '/home/henryclever/test'

    elif opt.computer == 'gorilla':
        name_prefix = '/home/henry/IROS_Data'
        name_prefix_qt = '/home/henry/test'


    # subjects 2 to 8
    pathSuffixAll = 'trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p'

    # subjects 9 to 18
    pathSuffixSupine = 'trainval_200rlh1_115rlh2_75rlh3_175rllair.p'
    pathSuffixSitting = 'trainval_sit175rlh_sit120rll.p'

    for subject in [10, 11, 12, 13, 14, 15, 16, 17, 18]:
        opt.subjectPath = name_prefix+'/subject_'+str(subject)+'/p_files/'+pathSuffixSupine


        opt.subject4Path = name_prefix_qt+'/trainval4_150rh1_sit120rh.p'
        opt.subject8Path = name_prefix_qt+'/trainval8_150rh1_sit120rh.p'


        test_database_file = []
        training_database_file = []

        test_database_file.append(opt.subjectPath)
        training_database_file.append(opt.subjectPath)

        #test_database_file.append(opt.subject4Path)
        #training_database_file.append(opt.subject8Path)

        print opt.testPath, 'testpath'


        print test_database_file, 'test database file'

        p = PhysicalTrainer(training_database_file, test_database_file, opt, name_prefix, subject)


