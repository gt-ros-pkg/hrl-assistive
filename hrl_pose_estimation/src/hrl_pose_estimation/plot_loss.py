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
        self.opt.arms_only = True
        self.include_inter = True
        self.loss_vector_type = None#'arm_angles'
        # Set initial parameters
        self.dump_path = pkl_directory.rstrip('/')

        self.mat_size = (NUMOFTAXELS_X, NUMOFTAXELS_Y)


        if self.opt.arms_only == True:
            self.output_size = (NUMOFOUTPUTNODES-5, NUMOFOUTPUTDIMS)
        else:
            self.output_size = (NUMOFOUTPUTNODES, NUMOFOUTPUTDIMS)

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
            #plt.plot(train_val_loss['epoch_flip_1'], train_val_loss['train_flip_1'],'b')
            #plt.plot(train_val_loss['epoch_1'], train_val_loss['train_1'], 'g')
            #plt.plot(train_val_loss['epoch_1'], train_val_loss['val_1'], 'k')
            #plt.plot(train_val_loss['epoch_flip_1'], train_val_loss['val_flip_1'], 'r')
            #plt.plot(train_val_loss['epoch_flip_shift_1'], train_val_loss['train_flip_shift_1'], 'g')
            #plt.plot(train_val_loss['epoch_flip_shift_1'], train_val_loss['val_flip_shift_1'], 'g')
            #plt.plot(train_val_loss['epoch_flip_shift_nd_1'], train_val_loss['val_flip_shift_nd_1'], 'g')
            #plt.plot(train_val_loss['epoch_flip_shift_nd_nohome_1'], train_val_loss['val_flip_shift_nd_nohome_1'], 'y')
            #plt.plot(train_val_loss['epoch_armsup_flip_shift_scale5_nd_nohome_1'], train_val_loss['train_armsup_flip_shift_scale5_nd_nohome_1'], 'b')
            #plt.plot(train_val_loss['epoch_armsup_flip_shift_scale5_nd_nohome_1'], train_val_loss['val_armsup_flip_shift_scale5_nd_nohome_1'], 'r')

            plt.plot(train_val_loss_desk['epoch_armsup_700e_1'], train_val_loss_desk['val_armsup_700e_1'], 'k',label='Raw Pressure Map Input')
            #plt.plot(train_val_loss['epoch_sitting_flip_700e_4'], train_val_loss['val_sitting_flip_700e_4'], 'c',label='Synthetic Flipping: $Pr(X=flip)=0.5$')
            #plt.plot(train_val_loss_desk['epoch_armsup_flip_shift_scale10_700e_1'],train_val_loss_desk['val_armsup_flip_shift_scale10_700e_1'], 'g',label='Synthetic Flipping+Shifting: $X,Y \sim N(\mu,\sigma), \mu = 0 cm, \sigma \~= 9 cm$')
            plt.plot(train_val_loss_desk['epoch_armsup_flip_shift_scale5_nd_nohome_700e_1'],train_val_loss_desk['val_armsup_flip_shift_scale5_nd_nohome_700e_1'], 'y', label='Synthetic Flipping+Shifting+Scaling: $S_C \sim N(\mu,\sigma), \mu = 1, \sigma \~= 1.02$')
            plt.legend()
            plt.ylabel('Mean squared error loss over 30 joint vectors')
            plt.xlabel('Epochs, where 700 epochs ~ 4 hours')
            plt.title('Subject 1 laying validation Loss, training performed on subjects 2, 3, 4, 5, 6, 7, 8')


        elif self.subject == 4:
            #plt.plot(train_val_loss['epoch_flip_4'], train_val_loss['train_flip_4'], 'g')
            #plt.plot(train_val_loss['epoch_flip_4'], train_val_loss['val_flip_4'], 'y')
            #plt.plot(train_val_loss['epoch_4'], train_val_loss['train_4'], 'b')
            #plt.plot(train_val_loss['epoch_4'], train_val_loss['val_4'], 'r')
            #plt.plot(train_val_loss['epoch_flip_shift_nd_4'], train_val_loss['val_flip_shift_nd_4'], 'b')
            #plt.plot(train_val_loss['epoch_flip_shift_nd_nohome_4'], train_val_loss['val_flip_shift_nd_nohome_4'], 'y')

            if pathkey == 'lab_hd': #results presented to hrl dressing 171106

                if self.opt.arms_only == True:
                    #plt.plot(train_val_loss_all['epoch_2to8_all_armsonly_fss_100b_adam_300e_4'],train_val_loss_all['train_2to8_all_armsonly_fss_100b_adam_300e_4'],'k', label='Synthetic Flipping+Shifting+Scaling')
                    # plt.plot(train_val_loss_all['epoch_2to8_all_armsonly_fss_100b_adam_300e_4'],train_val_loss_all['val_2to8_all_armsonly_fss_100b_adam_300e_4'], 'c',label='Synthetic Flipping+Shifting+Scaling')
                    # plt.plot(train_val_loss_all['epoch_2to8_all_armsonly_fss_115b_adam_100e_4'],train_val_loss_all['val_2to8_all_armsonly_fss_115b_adam_100e_4'], 'g',label='Synthetic Flipping+Shifting+Scaling')
                    # plt.plot(train_val_loss_all['epoch_2to8_all_armsonly_fss_130b_adam_120e_4'],train_val_loss_all['val_2to8_all_armsonly_fss_130b_adam_120e_4'], 'y',label='Synthetic Flipping+Shifting+Scaling')
                    # plt.plot(train_val_loss_all['epoch_2to8_all_armsonly_fss_115b_adam_350e_4'],train_val_loss_all['val_2to8_all_armsonly_fss_115b_adam_350e_4'], 'r',label='Synthetic Flipping+Shifting+Scaling')
                    # plt.plot(train_val_loss_all['epoch_2to8_all_armsonly_fss_115b_adam_120e_lg1_4'],train_val_loss_all['val_2to8_all_armsonly_fss_115b_adam_120e_lg1_4'], 'b',label='Synthetic Flipping+Shifting+Scaling')
                    # plt.plot(train_val_loss_all['epoch_2to8_all_armsonly_fss_115b_adam_200e_sm1_4'],
                    #          train_val_loss_all['val_2to8_all_armsonly_fss_115b_adam_200e_sm1_4'], 'm',
                    #          label='Synthetic Flipping+Shifting+Scaling')
                    #plt.plot(train_val_loss_all['epoch_2to8_alldata_armsonly_direct_115b_adam_200e_4'],train_val_loss_all['val_2to8_alldata_armsonly_direct_115b_adam_200e_4'], 'k',label='Direct Joint, Synthetic Flipping+Shifting')
                    #plt.plot(train_val_loss_all['epoch_2to8_all_armsonly_kincons_115b_adam_200e_4'],train_val_loss_all['val_2to8_all_armsonly_kincons_115b_adam_200e_4'], 'b',label='Kinematics, Synthetic Flipping+Shifting')
                    #plt.plot(train_val_loss_all['epoch_2to8_all_armsonly_kincons_fs_5xp_115b_adam_200e_4'],train_val_loss_all['val_2to8_all_armsonly_kincons_fs_5xp_115b_adam_200e_4'], 'y',label='Kinematics, 4x pitch, Synthetic Flipping+Shifting')
                    #plt.plot(train_val_loss_all['epoch_2to8_alldata_armsonly_direct_115b_adam_100e_4_smkernel4'], train_val_loss_all['val_2to8_alldata_armsonly_direct_115b_adam_100e_4_smkernel4'], 'g', label='Kinematics, Synthetic Flipping+Shifting')
                    #plt.plot(train_val_loss_all['epoch_2to8_alldata_armsonly_direct_115b_adam_200e_4_smkernel24'],
                    #         train_val_loss_all['val_2to8_alldata_armsonly_direct_115b_adam_200e_4_smkernel24'], 'y',
                    #         label='Kinematics, Synthetic Flipping+Shifting')



                    #plt.plot(train_val_loss_all['epoch_2to8_alldata_armsonly_direct_115b_adam_400e_44'], train_val_loss_all['val_2to8_alldata_armsonly_direct_115b_adam_400e_44'], 'g', label='2-channel input')
                    #plt.plot(train_val_loss_all['epoch_2to8_alldata_armsonly_direct_inclinter_115b_adam_200e_44'],
                    #         train_val_loss_all['val_2to8_alldata_armsonly_direct_inclinter_115b_adam_200e_44'], 'y',
                    #         label='3-channel input')
                    #plt.plot([0, 410], [3000, 3000], 'k')
                    plt.plot(train_val_loss_all['epoch_2to8_alldata_armsonly_euclidean_error_115b_adam_100e_44'], train_val_loss_all['train_2to8_alldata_armsonly_euclidean_error_115b_adam_100e_44'], 'k')
                    plt.plot(train_val_loss_all['epoch_2to8_alldata_armsonly_euclidean_error_115b_adam_100e_44'], train_val_loss_all['val_2to8_alldata_armsonly_euclidean_error_115b_adam_100e_44'], 'y')



                else:
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




        elif self.subject == 10:
            if pathkey == 'hc_desktop':
                plt.plot(train_val_loss_all['epoch_alldata_flip_shift_scale5_700e_10'],train_val_loss_all['train_alldata_flip_shift_scale5_700e_10'], 'g',label='Synthetic Flipping+Shifting+Scaling: $S_C \sim N(\mu,\sigma), \mu = 1, \sigma \~= 1.02$')
                plt.plot(train_val_loss_all['epoch_alldata_flip_shift_scale5_700e_10'],train_val_loss_all['val_alldata_flip_shift_scale5_700e_10'], 'y',label='Synthetic Flipping+Shifting+Scaling: $S_C \sim N(\mu,\sigma), \mu = 1, \sigma \~= 1.02$')

            #plt.plot(train_val_loss['epoch_flip_2'], train_val_loss['train_flip_2'], 'y')
            #plt.plot(train_val_loss['epoch_flip_2'], train_val_loss['val_flip_2'], 'g')
            #plt.plot(train_val_loss['epoch_flip_shift_nd_2'], train_val_loss['val_flip_shift_nd_2'], 'y')

        #plt.axis([0,410,0,30000])
        plt.axis([0, 100, 0, 10000])
        plt.show()

        self.count = 0

        rospy.init_node('calc_mean_std_of_head_detector_node')






    def validate_model(self):

        validation_set = load_pickle(self.dump_path + '/subject_' + str(3) + '/p_files/trainval_150rh1_lh1_rl_ll_100rh23_lh23_sit120rh_lh_rl_ll.p')

        test_dat = validation_set
        for key in test_dat:
            print key, np.array(test_dat[key]).shape


        self.test_x_flat = []  # Initialize the testing pressure mat list
        for entry in range(len(test_dat['images'])):
            self.test_x_flat.append(test_dat['images'][entry])
        #test_x = self.preprocessing_pressure_array_resize(self.test_x_flat)
        #test_x = np.array(test_x)

        self.test_a_flat = []  # Initialize the testing pressure mat angle list
        for entry in range(len(test_dat['images'])):
            self.test_a_flat.append(test_dat['bed_angle_deg'][entry])
        test_xa = self.preprocessing_create_pressure_angle_stack(self.test_x_flat, self.test_a_flat)
        test_xa = np.array(test_xa)
        self.test_x_tensor = torch.Tensor(test_xa)

        print 'Finished converting inputs to a torch tensor'

        self.test_y_flat = []  # Initialize the ground truth list
        for entry in range(len(test_dat['images'])):
            if self.opt.arms_only == True:
                c = np.concatenate((test_dat['markers_xyz_m'][entry][6:18] * 1000, test_dat['joint_lengths_U_m'][entry] * 100, test_dat['joint_angles_U_deg'][entry], test_dat['markers_xyz_m'][entry][3:6] * 100), axis=0)
                self.test_y_flat.append(c)
            else:
                self.test_y_flat.append(test_dat['markers_xyz_m'][entry] * 1000)
        self.test_y_tensor = torch.Tensor(self.test_y_flat)

        print 'Finished converting outputs to a torch tensor'

        print len(validation_set), 'size of validation set'
        batch_size = 1

        self.test_dataset = torch.utils.data.TensorDataset(self.test_x_tensor, self.test_y_tensor)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size, shuffle=False)



        #torso_length_model = torch.load(self.dump_path + '/subject_' + str(self.subject) + '/p_files/convnet_2to8_alldata_armsonly_torso_lengths_115b_adam_100e_4.pt')
        #angle_model = torch.load(self.dump_path + '/subject_' + str(self.subject) + '/p_files/convnet_2to8_alldata_armsonly_arm_angles_115b_adam_200e_4.pt')
        model = torch.load(self.dump_path + '/subject_' + str(self.subject) + '/p_files/convnet_2to8_alldata_armsonly_direct_inclinter_115b_adam_200e_4.pt')

        count = 0
        for batch_idx, batch in enumerate(self.test_loader):
            count += 1
            #print count

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


            images, targets, constraints = Variable(batch[0]), Variable(batch[1]), Variable(batch[2])
            bed_distances = KinematicsLib().get_bed_distance(images, targets)


            #print batch[0].shape
            #print image[0].shape




            print test_dat['marker_direct_error_euclideans_m'][count-1]




            scores = model(images)
            error_norm = VisualizationLib().print_error(targets, scores, self.output_size, self.loss_vector_type, data=str(count))/1000
            self.im_sample = batch[0].numpy()
            self.im_sample = np.squeeze(self.im_sample[0, :])
            self.tar_sample = batch[1].numpy()
            self.tar_sample = np.squeeze(self.tar_sample[0, :]) / 1000
            self.sc_sample = scores.data.numpy()
            self.sc_sample = np.squeeze(self.sc_sample[0, :]) / 1000
            self.sc_sample = np.reshape(self.sc_sample, self.output_size)
            VisualizationLib().rviz_publish_input(self.im_sample[0, :, :], self.im_sample[-1, 10, 10])
            VisualizationLib().rviz_publish_output(np.reshape(self.tar_sample, self.output_size), self.sc_sample)
            VisualizationLib().visualize_pressure_map(self.im_sample, self.tar_sample, self.sc_sample, block = True)
            #VisualizationLib().visualize_error_from_distance(bed_distances, error_norm)




            if self.loss_vector_type == 'arm_angles':
                constraint_scores = angle_model(images)
                torso_length_scores = torso_length_model(images)

                output = KinematicsLib().forward_arm_kinematics(images, torso_length_scores, constraint_scores) #remember to change this to constraint scores.
                scores = Variable(torch.Tensor(output))

            else:
                #batch[1] = torch.cat((torch.mul(batch[1][:, 28:31], 10), batch[1][:, 0:12]), dim=1)
                #images, targets = Variable(batch[0]), Variable(batch[1])
                scores = model(images)
            error_norm = VisualizationLib().print_error(targets, scores, self.output_size, self.loss_vector_type, data=str(count))/1000
            self.im_sample = batch[0].numpy()
            self.im_sample = np.squeeze(self.im_sample[0, :])
            self.tar_sample = batch[1].numpy()
            self.tar_sample = np.squeeze(self.tar_sample[0, :]) / 1000
            self.sc_sample = scores.data.numpy()
            self.sc_sample = np.squeeze(self.sc_sample[0, :]) / 1000
            self.sc_sample = np.reshape(self.sc_sample, self.output_size)
            #VisualizationLib().rviz_publish_input(self.im_sample[0,:,:], self.im_sample[-1,10,10])
            #VisualizationLib().rviz_publish_output(np.reshape(self.tar_sample, self.output_size), self.sc_sample)
            #VisualizationLib().visualize_pressure_map(self.im_sample, self.tar_sample, self.sc_sample, block = True)
            #VisualizationLib().visualize_error_from_distance(bed_distances, error_norm)

        return mean, stdev

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
            #if self.normalize == True:
            #    p_map = normalize(p_map, norm='l2')

            p_map_dataset.append(p_map)
        if self.verbose: print len(data[0]),'x',1, 'size of an incoming pressure map'
        if self.verbose: print len(p_map_dataset[0]),'x',len(p_map_dataset[0][0]), 'size of a resized pressure map'
        return p_map_dataset

    def preprocessing_create_pressure_angle_stack(self,x_data,a_data):
        p_map_dataset = []
        for map_index in range(len(x_data)):
            # print map_index, self.mat_size, 'mapidx'
            # Resize mat to make into a matrix
            p_map = np.reshape(x_data[map_index], self.mat_size)
            a_map = np.zeros_like(p_map) + a_data[map_index]

            if self.include_inter == True:
                p_map_inter = (100-2*np.abs(p_map - 50))*4
                p_map_dataset.append([p_map, p_map_inter, a_map])
            else:
                p_map_dataset.append([p_map, a_map])

        if self.verbose: print len(x_data[0]), 'x', 1, 'size of an incoming pressure map'
        if self.verbose: print len(p_map_dataset[0][0]), 'x', len(p_map_dataset[0][0][0]), 'size of a resized pressure map'
        if self.verbose: print len(p_map_dataset[0][1]), 'x', len(p_map_dataset[0][1][0]), 'size of the stacked angle mat'

        return p_map_dataset



    def run(self):
        '''Runs either the synthetic database creation script or the
        raw dataset creation script to create a dataset'''
        self.validate_model()
        return





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
    p.add_option('--arms_only', action='store_true',
                 dest='arms_only', \
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
