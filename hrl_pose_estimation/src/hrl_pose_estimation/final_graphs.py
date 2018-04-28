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
import pylab as pylab

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.unicode'] = True

# Machine Learning
from scipy import ndimage
from scipy.ndimage.filters import gaussian_filter
from scipy import interpolate
from scipy.misc import imresize
from scipy.ndimage.interpolation import zoom
import scipy.stats as ss
from scipy.stats import ttest_ind
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
        self.old = False
        self.normalize = True
        self.include_inter = True
        # Set initial parameters
        self.subject = 4
        self.dump_path = pkl_directory.rstrip('/')

        self.mat_size = (NUMOFTAXELS_X, NUMOFTAXELS_Y)

        self.output_size = (10, 3)




        train_val_loss = load_pickle(self.dump_path + '/train_val_losses.p')
        train_val_loss_desk = load_pickle(self.dump_path + '/train_val_losses_hcdesktop.p')
        train_val_loss_GPU2 = load_pickle(self.dump_path + '/train_val_losses_GPU2.p')
        train_val_loss_GPU3 = load_pickle(self.dump_path + '/train_val_losses_GPU3.p')
        train_val_loss_GPU4 = load_pickle(self.dump_path + '/train_val_losses_GPU4.p')
        train_val_loss_GPU2new = load_pickle(self.dump_path + '/train_val_lossesGPU2_021018.p')
        train_val_loss_GPU3new = load_pickle(self.dump_path + '/train_val_lossesGPU3_021018.p')
        train_val_loss_GPU4new = load_pickle(self.dump_path + '/train_val_lossesGPU4_021018.p')
        for key in train_val_loss:
            print key
        print '###########################  done with laptop #################'
        for key in train_val_loss_desk:
            print key
        print '###########################  done with desktop ################'
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



        #plt.axis([0,410,0,30000])
        #plt.axis([0, 200, 10, 15])
        #if self.opt.visualize == True:
        #    plt.show()
        plt.close()


        rospy.init_node('final_graphs')
        self.count = 0

    def final_error(self):
        flat_errors = {}

        for modeltype in ['anglesSTVL']:#,'anglesSTVL', 'anglesCL']: #'KNN','LRR','KRR',
        #for modeltype in ['anglesCL']:
            error_norm_flat = None
            for posture in ['seated']:
                for subject in [9,10,11,12,13, 14, 15, 16, 17, 18]:#, 10, 11, 12, 13, 14, 15, 16, 17, 18]:
                    if modeltype == 'anglesCL' or modeltype == 'anglesSTVL' or modeltype == 'direct':
                        #if posture == 'supine':
                        error_avg, error_std, _, _ = load_pickle(
                            self.dump_path + '/Final_Data_V2/error_avg_std_T25_subject' + str(
                                subject) + '_anglesSTVLsupine.p')

                        _, _, targets, targets_est = load_pickle(self.dump_path + '/Final_Data_V2/error_avg_std_T25_subject' + str(subject) + '_' + str(modeltype) + 'supine.p')
                        #elif posture == 'seated':
                        _, _, targets_sit, targets_est_sit = load_pickle(self.dump_path + '/Final_Data_V2/error_avg_std_T25_subject' + str(subject) + '_' + str(modeltype) + 'seated.p')
                        # size (P x N) each: number of images, number of joints


                        targets = np.array(targets)
                        targets_est = np.array(targets_est)
                        print targets.shape
                        #print targets_sit.shape

                        #targets = np.concatenate((targets, targets_sit), axis = 0)
                        #targets_est = np.concatenate((targets_est, targets_est_sit), axis = 0)


                        #print np.reshape(targets, (targets.shape[0], self.output_size[0], self.output_size[1]))[0, :, :]
                        #print np.reshape(targets,(targets.shape[0],self.output_size[0], self.output_size[1]))[0, :, :] - np.reshape(targets_est, (targets.shape[0],self.output_size[0], self.output_size[1]))[0, :, :]

                        print subject, 'subject'
                        error_norm, _, _ = VisualizationLib().print_error(targets, targets_est, self.output_size, modeltype, data=str(subject), printerror=True)

                        #error_norm = error_norm[:, 2:4]
                        print error_norm.shape

                        try:
                            error_norm_flat = np.concatenate((error_norm_flat, error_norm.flatten()), axis=0)
                        except:
                            error_norm_flat = error_norm.flatten()




                    elif modeltype == 'KNN' or modeltype == 'LRR' or modeltype == 'KRR':
                        error_avg = load_pickle(self.dump_path + '/Final_Data/error_avg_subject' + str(subject) + '_' + str(modeltype) + '.p')



                        try:
                            error_avg_flat = np.concatenate((error_avg_flat, np.array(error_avg).flatten() / 10), axis=0)
                        except:
                            error_avg_flat = np.array(error_avg).flatten() / 10
                        print np.shape(error_avg_flat), subject, modeltype


            flat_errors[modeltype] = error_norm_flat


        for modeltype in flat_errors:
            print modeltype
            print np.mean(flat_errors[modeltype]), 'mean'
            print np.std(flat_errors[modeltype]), 'std'
            print flat_errors[modeltype].shape, 'size'
            print np.shape(flat_errors[modeltype]), 'count'

        print ttest_ind(flat_errors['anglesSTVL'], flat_errors['anglesCL'], equal_var=True), 'Students T test on direct and anglesCL'
        print ttest_ind(flat_errors['anglesSTVL'], flat_errors['direct'], equal_var=True), 'Students T test on direct and anglesSTVL'
        print ttest_ind(flat_errors['anglesSTVL'], flat_errors['anglesCL'], equal_var=False), 'Welchs T test on direct and anglesCL'
        print ttest_ind(flat_errors['direct'], flat_errors['direct'], equal_var=True), 'Students T test on direct and direct'


        plt.show()


    def final_foot_variance(self):
        flat_errors = {}

        for modeltype in ['anglesSTVL']:
            error_norm_flat = None
            for subject in [9, 10, 11, 12, 13, 14, 15, 16, 17, 18]:
                if modeltype == 'anglesCL' or modeltype == 'anglesSTVL' or modeltype == 'direct':

                    _, error_std_air, _, _ = load_pickle(self.dump_path + '/Feet_Variance2/error_avg_std_T25_subject' + str(subject) + '_' + str(modeltype) + 'air.p')
                    _, error_std_gnd, _, _ = load_pickle(self.dump_path + '/Feet_Variance2/error_avg_std_T25_subject' + str(subject) + '_' + str(modeltype) + 'snow.p')

                    print np.shape(error_std_air), 'shape!'

                    error_std_airR = np.array(error_std_air)[0:25, :]
                    error_std_airL = np.array(error_std_air)[25:50, :]
                    error_std_gndR = np.array(error_std_gnd)[0:25, :]
                    error_std_gndL = np.array(error_std_gnd)[25:50, :]
                    #_, error_std_gndR, _, _ = load_pickle(self.dump_path + '/Feet_Variance/error_avg_std_T25_subject' + str(subject) + '_' + str(modeltype) + '_RLgnd_only.p')
                    #_, error_std_airL, _, _ = load_pickle(self.dump_path + '/Feet_Variance/error_avg_std_T25_subject' + str(subject) + '_' + str(modeltype) + '_LLair_only.p')
                    #_, error_std_gndL, _, _ = load_pickle(self.dump_path + '/Feet_Variance/error_avg_std_T25_subject' + str(subject) + '_' + str(modeltype) + '_LLgnd_only.p')

                    error_std_airR = np.concatenate((np.array(error_std_airR)[:,6:7], np.array(error_std_airR)[:,8:9]), axis = 1)
                    error_std_gndR = np.concatenate((np.array(error_std_gndR)[:,6:7], np.array(error_std_gndR)[:,8:9]), axis = 1)
                    error_std_airL = np.concatenate((np.array(error_std_airL)[:,7:8], np.array(error_std_airL)[:,9:10]), axis = 1)
                    error_std_gndL = np.concatenate((np.array(error_std_gndL)[:,7:8], np.array(error_std_gndL)[:,9:10]), axis = 1)



                    try:
                        error_std_airR_all = np.concatenate((error_std_airR_all, error_std_airR), axis=0)
                        error_std_gndR_all = np.concatenate((error_std_gndR_all, error_std_gndR), axis=0)
                        error_std_airL_all = np.concatenate((error_std_airL_all, error_std_airL), axis=0)
                        error_std_gndL_all = np.concatenate((error_std_gndL_all, error_std_gndL), axis=0)
                    except:
                        error_std_airR_all = error_std_airR
                        error_std_gndR_all = error_std_gndR
                        error_std_airL_all = error_std_airL
                        error_std_gndL_all = error_std_gndL

                    print np.shape(error_std_airR_all)
                    print np.shape(error_std_gndR_all)
                    print np.shape(error_std_airL_all)
                    print np.shape(error_std_gndL_all)



        print np.mean(error_std_airR_all[:,0]), 'mean R knee air, ', np.mean(error_std_airR_all[:,1]), 'mean R ankle air', np.std(error_std_airR_all[:,0]), 'std R knee air, ', np.std(error_std_airR_all[:,1]), 'std R ankle air, '

        print np.mean(error_std_gndR_all[:,0]), 'mean R knee gnd, ', np.mean(error_std_gndR_all[:,1]), 'mean R ankle gnd, ', np.std(error_std_gndR_all[:,0]), 'std R knee gnd, ', np.std(error_std_gndR_all[:,1]), 'std R ankle gnd, '

        print np.mean(error_std_airL_all[:,0]), 'mean L knee air, ', np.mean(error_std_airL_all[:,1]), 'mean L ankle air, ', np.std(error_std_airL_all[:,0]), 'std L knee air, ', np.std(error_std_airL_all[:,1]), 'std L ankle air, '

        print np.mean(error_std_gndL_all[:,0]), 'mean L knee gnd, ', np.mean(error_std_gndL_all[:,1]), 'mean L ankle gnd, ', np.std(error_std_gndL_all[:,0]), 'std L knee gnd, ', np.std(error_std_gndL_all[:,1]), 'std L ankle gnd, '


        print ttest_ind(error_std_airR_all[:,0], error_std_gndR_all[:,0], equal_var=False), 'Welchs T test on R knee air and R knee gnd'
        print ttest_ind(error_std_airR_all[:,1], error_std_gndR_all[:,1], equal_var=False), 'Welchs T test on R ankle air and R ankle gnd'
        print ttest_ind(error_std_airL_all[:,0], error_std_gndL_all[:,0], equal_var=False), 'Welchs T test on L knee air and L knee gnd'
        print ttest_ind(error_std_airL_all[:,1], error_std_gndL_all[:,1], equal_var=False), 'Welchs T test on L ankle air and L ankle gnd'

        error_std_air_all = np.concatenate((error_std_airR_all,error_std_airL_all), axis = 0)
        error_std_gnd_all = np.concatenate((error_std_gndR_all,error_std_gndL_all), axis = 0)


        print ttest_ind(error_std_air_all[:,0], error_std_gnd_all[:,0], equal_var=False), 'Welchs T test on all knee air and all knee gnd'
        print ttest_ind(error_std_air_all[:,1], error_std_gnd_all[:,1], equal_var=False), 'Welchs T test on all ankle air and all ankle gnd'


        plt.show()



    def error_threshold(self):
        threshold_error = {}
        joint_percent = {}
        joint_percent_keep = {}
        for modeltype in ['KNN','Ridge','KRidge','anglesSTVL','anglesCL','direct']:
        #for modeltype in ['anglesSTVL']:
            error_avg_flat = None
            for subject in [9,10,11,12,13,14,15,16,17,18]:
                if modeltype == 'anglesSTVL' or modeltype == 'anglesCL' or modeltype == 'direct':
                    _, _, targets, targets_est = load_pickle(self.dump_path+'/Final_Data_V2/error_avg_std_T25_subject'+str(subject)+'_'+str(modeltype)+'seated.p')
                    _, _, targets_sit, targets_est_sit = load_pickle(self.dump_path+'/Final_Data_V2/error_avg_std_T25_subject'+str(subject)+'_'+str(modeltype)+'supine.p')

                    targets = np.array(targets)
                    targets_sit = np.array(targets_sit)
                    targets_est = np.array(targets_est)
                    targets_est_sit = np.array(targets_est_sit)

                    # print targets.shape

                    targets = np.concatenate((targets, targets_sit), axis=0)
                    # print targets.shape
                    #targets_est = np.concatenate((targets_est, targets_est_sit), axis=0)

                    targets = np.reshape(targets, (targets.shape[0], 10, 3))
                    targets_est = np.array(targets_est)
                    targets_est = np.reshape(targets_est, (targets_est.shape[0], 10, 3))
                    error_avg = np.linalg.norm((targets_est - targets), axis=2)





                    # size (P x N) each: number of images, number of joints
                    try:
                        error_avg_flat = np.concatenate((error_avg_flat, np.array(error_avg).flatten()), axis=0)
                    except:
                        error_avg_flat = np.array(error_avg).flatten()
                    print np.shape(error_avg_flat), subject, modeltype

                elif modeltype == 'KNN' or modeltype == 'Ridge' or modeltype == 'KRidge':
                    error_avg = load_pickle(self.dump_path+'/Final_Data/error_avg_subject'+str(subject)+'_'+str(modeltype)+'.p')
                    print np.shape(error_avg)

                    try:
                        error_avg_flat = np.concatenate((error_avg_flat, np.array(error_avg).flatten()/10), axis=0)
                    except:
                        error_avg_flat = np.array(error_avg).flatten()/10
                    print np.shape(error_avg_flat), subject, modeltype


            error_avg = np.array(error_avg_flat).flatten()
            print modeltype
            threshold_error[modeltype] = np.flip(np.linspace(0,np.max(error_avg), num = 200), axis = 0)
            joint_percent[modeltype] = np.zeros_like(threshold_error[modeltype])
            joint_percent_keep[modeltype] = np.zeros_like(threshold_error[modeltype])

            #print np.mean(error_avg), 'GMPJPE'

            for i in range(threshold_error[modeltype].shape[0]):
                joint_percent_queue = 0.
                joint_percent_queue_keep = 0.

                for j in range(error_avg.shape[0]):
                    #print error_var[j], threshold_variance[i], 'std and threshold'
                    if error_avg[j] <= threshold_error[modeltype][i]: #if our variance is less than the threshold, keep the value
                        joint_percent_queue += 1
                    if error_avg[j] >= threshold_error[modeltype][i]:
                        joint_percent_queue_keep += 1

                joint_percent[modeltype][i] = joint_percent_queue
                joint_percent_keep[modeltype][i] = joint_percent_queue_keep


            joint_percent[modeltype] = joint_percent[modeltype]/np.max(joint_percent[modeltype])
            joint_percent_keep[modeltype] = joint_percent_keep[modeltype]/np.max(joint_percent_keep[modeltype])



        xlim = [0, 300]
        ylim1 = [0, 1.]
        fig, ax1 = plt.subplots()
        #plt.suptitle('Subject '+str(subject)+ ' Error Thresholding.  All joints.', fontsize=16)
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim1)
        ax1.set_xlabel('Error Threshold (mm)', fontsize=16)
        ax1.set_ylabel('Fraction of joints \n below the error threshold', color='k', fontsize=16)
        ratioKNN = ax1.plot(threshold_error['KNN']*10000, joint_percent['KNN'], 'c-', lw=2, label='KNN')
        ratioLRR = ax1.plot(threshold_error['Ridge']*10000, joint_percent['Ridge'], 'm--', lw=2, label='LRR')
        ratioKRR = ax1.plot(threshold_error['KRidge']*10000, joint_percent['KRidge'], 'y-', lw=2, label='KRR')
        ratiodirect = ax1.plot(threshold_error['direct'], joint_percent['direct'], 'r--', lw=2, label='CNN direct')
        ratiokincL = ax1.plot(threshold_error['anglesCL'], joint_percent['anglesCL'], 'g-', lw=2, label='CNN kin. avg. '+r"$\boldsymbol{l}$")
        ratiokinvL = ax1.plot(threshold_error['anglesSTVL'], joint_percent['anglesSTVL'], 'b--', lw=2, label='CNN kin. regr. '+r"$\boldsymbol{l}$")
        lns = ratioKNN+ratioLRR+ratioKRR+ratiodirect+ratiokincL+ratiokinvL
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc=0)
        plt.show()

        plt.subplot(1,2,2)
        ax2.set_xlim(xlim)
        ax2.set_ylim(ylim1)
        ax2.set_xlabel('Error Threshold (mm)', fontsize=16)
        ax2.set_ylabel('Fraction of joints \n below the error threshold', color='k', fontsize=16)
        ratioKNN = ax2.plot(threshold_error['KNN']*10000, joint_percent['KNN'], 'c-', lw=3, label='KNN')
        ratioLRR = ax2.plot(threshold_error['Ridge']*10000, joint_percent['Ridge'], 'm-', lw=3, label='LRR')
        ratioKRR = ax2.plot(threshold_error['KRidge']*10000, joint_percent['KRidge'], 'y-', lw=3, label='KRR')
        ratiodirect = ax2.plot(threshold_error['direct'] * 10, joint_percent['direct'], 'r-', lw=3, label='CNN direct')
        ratiokincL = ax2.plot(threshold_error['anglesCL'] * 10, joint_percent['anglesCL'], 'g-', lw=3, label='CNN kin. avg. '+r"$\boldsymbol{l}$")
        ratiokinvL = ax2.plot(threshold_error['anglesSTVL'] * 10, joint_percent['anglesSTVL'], 'b-', lw=3, label='CNN kin. regr. '+r"$\boldsymbol{l}$")
        lns = ratioKNN+ratioLRR+ratioKRR+ratiodirect+ratiokincL+ratiokinvL
        labs = [l.get_label() for l in lns]
        ax2.legend(lns, labs, loc=0)

        plt.show()



        xlim = [0, 300]
        ylim1 = [0, 1.0]
        fig, ax1 = plt.subplots()
        #plt.suptitle('Subject 10 Std. Dev. Thresholding for keeping points.  All joints.', fontsize=16)
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim1)
        ax1.set_xlabel('Error Threshold (mm)', fontsize=16)
        ax1.set_ylabel('Fraction of joints \n above the error threshold', color='k', fontsize=16)
        ratioKNN = ax1.plot(threshold_error['KNN'] * 10, joint_percent_keep['KNN'], 'c-', lw=3, label='KNN')
        ratioLRR = ax1.plot(threshold_error['LRR'] * 10, joint_percent_keep['LRR'], 'm-', lw=3, label='LRR')
        ratioKRR = ax1.plot(threshold_error['KRR'] * 10, joint_percent_keep['KRR'], 'y-', lw=3, label='KRR')
        ratiodirect = ax1.plot(threshold_error['direct'] * 10, joint_percent_keep['direct'], 'r-', lw=3, label='CNN direct')
        ratiokincL = ax1.plot(threshold_error['kincL'] * 10, joint_percent_keep['kincL'], 'g-', lw=3, label='CNN kin. avg. '+r"$\boldsymbol{l}$")
        ratiokinvL = ax1.plot(threshold_error['kinvL'] * 10, joint_percent_keep['kinvL'], 'b-', lw=3, label='CNN kin. regr. ' + r"$\boldsymbol{l}$")
        lns = ratioKNN + ratioLRR + ratioKRR + ratiodirect+ratiokincL+ratiokinvL
        labs = [l.get_label() for l in lns]
        plt.legend(lns, labs, loc=0)
        plt.show()


        print np.max(error_avg)



    def dropout_std_threshold(self):
        for subject in [9, 10, 11, 12, 13, 14, 15, 16, 17, 18]:#,10,11,12,13,14,15,16,17,18]:
            _, error_std,targets, targets_est = load_pickle(self.dump_path + '/Final_Data_V2/error_avg_std_T25_subject'+str(subject)+'_anglesSTVLsupine.p')
            _, error_std_sit, targets_sit, targets_est_sit = load_pickle(self.dump_path + '/Final_Data_V2/error_avg_std_T25_subject'+str(subject)+'_anglesSTVLseated.p')
            #print np.shape(error_avg)
            #print np.shape(error_std)

            #error_avg = np.concatenate((error_avg, error_avg_sit), axis = 0)
            error_std = np.concatenate((error_std, error_std_sit), axis = 0)

            targets =np.array(targets)
            targets_sit = np.array(targets_sit)
            targets_est = np.array(targets_est)
            targets_est_sit = np.array(targets_est_sit)

            #print targets.shape

            targets = np.concatenate((targets, targets_sit), axis = 0)
            #print targets.shape
            targets_est = np.concatenate((targets_est, targets_est_sit), axis = 0)

            targets = np.reshape(targets, (targets.shape[0], 10, 3))
            targets_est = np.array(targets_est)
            targets_est = np.reshape(targets_est, (targets_est.shape[0], 10 ,3))
            error_avg = np.linalg.norm((targets_est - targets), axis = 2)

            print error_avg[0, :]
            #print targets_est.shape
            #print error_avg.shape

            #print targets.shape
            #print np.reshape(targets_est.numpy(), axis=0) - targets[0, :].numpy(), self.output_size)
            #print np.linalg.norm(np.reshape(np.mean(targets_est.numpy(), axis=0) - targets[0, :].numpy(), self.output_size), axis=1)

            try:
                error_avg_flat = np.concatenate((error_avg_flat, np.array(error_avg).flatten()), axis = 0)
                error_std_flat = np.concatenate((error_std_flat, np.array(error_std).flatten()), axis = 0)

            except:
                error_avg_flat = np.array(error_avg).flatten()
                error_std_flat = np.array(error_std).flatten()

            print np.shape(error_avg_flat)



        threshold_variance = np.flip(np.linspace(0,np.max(error_std_flat), num = 200), axis = 0)
        joint_percent = np.zeros_like(threshold_variance)
        GMPJPE = np.zeros_like(threshold_variance)
        joint_percent_keep = np.zeros_like(threshold_variance)
        GMPJPE_keep = np.zeros_like(threshold_variance)

        #print np.mean(error_avg_flat), 'GMPJPE'

        for i in range(threshold_variance.shape[0]):
            joint_percent_queue = 0.
            joint_percent_queue_keep = 0.
            GMPJPE_queue = []
            GMPJPE_queue_keep = []

            for j in range(error_avg_flat.shape[0]):
                #print error_std_flat[j], threshold_variance[i], 'std and threshold'
                if error_std_flat[j] <= threshold_variance[i]: #if our variance is less than the threshold, keep the value
                    joint_percent_queue += 1
                    GMPJPE_queue.append(error_avg_flat[j])
                if error_std_flat[j] >= threshold_variance[i]:
                    joint_percent_queue_keep += 1
                    GMPJPE_queue_keep.append(error_avg_flat[j])

            joint_percent[i] = joint_percent_queue
            joint_percent_keep[i] = joint_percent_queue_keep
            if GMPJPE_queue:
                GMPJPE[i] = np.mean(np.array(GMPJPE_queue))
            if GMPJPE_queue_keep:
                GMPJPE_keep[i] = np.mean(np.array(GMPJPE_queue_keep))


        joint_percent = joint_percent/np.max(joint_percent)

        joint_percent_keep = joint_percent_keep/np.max(joint_percent_keep)

        #print threshold_variance
        #print joint_percent, 'number kept. should decrease with decreasing threshold'
        #print GMPJPE, 'decreasing GMPJPE'

        xlim = [0, 300]
        ylim1 = [0, 1.0]
        ylim2 = [0, 100]
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        #plt.suptitle('Subject 10 Std. Dev. Thresholding for removing points.  All joints.',fontsize=16)
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim1)
        ax1.tick_params(labelsize=16)
        ax2.tick_params(labelsize=16)
        ax1.set_xlabel('Std. dev. threshold (mm) for discarding data \n $V=25$ MC dropout passes',fontsize=16)
        ax1.set_ylabel('Fraction of joints remaining',color='g',fontsize=16)

        #ax1.set_title('Right Elbow')

        ax2.set_xlim(xlim)
        ax2.set_ylim(ylim2)
        ax2.set_xlabel('Std. dev. threshold (mm) for discarding data \n $V=25$ MC dropout passes',fontsize=16)
        ax2.set_ylabel('MPJPE error across remaining joints',color='b',fontsize=16)
        ratio = ax1.plot(threshold_variance, joint_percent, 'g--', lw=4, label='Fraction of Joints Remaining')
        gmpjpe = ax2.plot(threshold_variance, GMPJPE, 'b-', lw = 4, label = 'MPJPE of Remaining Joints')

        lns = ratio + gmpjpe
        labs = [l.get_label() for l in lns]
        plt.legend(lns, labs, loc=4)
        plt.gcf().subplots_adjust(bottom=0.15)
        #ax2.set_title('Left Elbow')

        plt.show()

        xlim = [0, 300]
        ylim1 = [0, 1.0]
        ylim2 = [0, 500]
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        #plt.suptitle('Subject 10 Std. Dev. Thresholding for keeping points.  All joints.',fontsize=16)
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim1)
        ax1.tick_params(labelsize=16)
        ax2.tick_params(labelsize=16)
        ax1.set_xlabel('Std. dev. threshold (mm) for discarding data \n $V=25$ MC dropout passes',fontsize=16)
        ax1.set_ylabel('Fraction of joints discarded',color='g',fontsize=16)
        ratio = ax1.plot(threshold_variance, joint_percent_keep, 'g--',lw=4, label='Fraction of Joints Discarded')
        #ax1.set_title('Right Elbow')

        ax2.set_xlim(xlim)
        ax2.set_ylim(ylim2)
        ax2.set_xlabel('Std. dev. threshold (mm) for discarding data \n $V=25$ MC dropout passes',fontsize=16)
        ax2.set_ylabel('MPJPE error across discarded joints',color='b',fontsize=16)
        gmpjpe = ax2.plot(threshold_variance, GMPJPE_keep, 'b-',lw=4, label = 'MPJPE of Discarded Joints')
        #ax2.set_title('Left Elbow')

        lns = ratio + gmpjpe
        labs = [l.get_label() for l in lns]
        plt.legend(lns, labs, loc=0)

        plt.gcf().subplots_adjust(bottom=0.15)
        plt.show()

        xlim = [0, np.max(error_std_flat)*10]
        ylim1 = [0, 1.0]
        ylim2 = [0, 500]
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        #plt.suptitle('Subject 10 Std. Dev. Thresholding for keeping points.  All joints.',fontsize=16)
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim1)
        ax1.set_xlabel('Std. dev. threshold (mm) for discarding data, $V=25$ MC dropout passes',fontsize=16)
        ax1.set_ylabel('Fraction of joints discarded',color='g',fontsize=16)
        ratio = ax1.plot(threshold_variance, joint_percent_keep, 'g--',lw=4, label='Fraction of Joints Discarded')
        #ax1.set_title('Right Elbow')

        ax2.set_xlim(xlim)
        ax2.set_ylim(ylim2)
        ax2.set_xlabel('Std. dev. threshold (mm) for discarding data, $V=25$ MC dropout passes',fontsize=16)
        ax2.set_ylabel('MPJPE error across discarded joints',color='b',fontsize=16)
        gmpjpe = ax2.plot(threshold_variance, GMPJPE_keep*10, 'b-',lw=4, label = 'MPJPE of Discarded Joints')
        #ax2.set_title('Left Elbow')

        lns = ratio + gmpjpe
        labs = [l.get_label() for l in lns]
        plt.legend(lns, labs, loc=0)
        plt.show()

        print np.max(error_std)



    def p_information_std(self):

        for subject in [9]:
            p_info_sum_rl, knee_ankle_std = load_pickle(self.dump_path + '/Final_Data/sumRL_sumLL_stdKA_T25_subject' + str(subject) + '_kinvL.p')

            try:
                error_avg_flat = np.concatenate((error_avg_flat, np.array(error_avg).flatten()), axis = 0)
                error_std_flat = np.concatenate((error_std_flat, np.array(error_std).flatten()), axis = 0)

            except:
                right_knee_std = knee_ankle_std[:,0]
                right_ankle_std = knee_ankle_std[:,2]
                right_std = np.mean([right_knee_std,right_ankle_std], axis = 0)

                left_knee_std = knee_ankle_std[:,1]
                left_ankle_std = knee_ankle_std[:,3]
                left_std = np.mean([left_knee_std,left_ankle_std], axis = 0)

                sums = np.concatenate((p_info_sum_rl[:, 0], p_info_sum_rl[:, 1]), axis = 0)
                std = np.concatenate((right_std, left_std), axis = 0)


                print np.mean(sums), np.std(sums), 'mean, std of right side sums'

            fig = plt.figure()
            ax1 = fig.add_subplot(1, 1, 1)
            ax1.plot(sums, std, 'ro')
            plt.show()

            error_avg_flat = np.array(error_avg).flatten()
            error_std_flat = np.array(error_std).flatten()

            print np.shape(error_avg_flat)


    def all_joint_error(self):
        # here is some example to for plotting
        posture = 'sitting'

        Label_L4L5_DP = ['Head', 'Chest', 'Right Elbow', 'Left Elbow', 'Right Wrist', 'Left Wrist', 'Right Knee', 'Left Knee', 'Right Ankle', 'Left Ankle','Overall']



        # Setting the positions and width for the bars
        pos = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        width = 0.13

        # Plotting the bars
        fig, ax = plt.subplots(figsize=(10,3),dpi = 110)

        #############ALL
        #KNN     = [87.63,52.54,103.07,105.55,187.81,189.33,83.39,94.20,80.36,72.86]
        #Std_KNN = [69.62,20.61,81.93,90.03,144.87,152.80,48.23,48.66,56.49,53.94]
        #LRR      = [134.64,56.37,109.70,117.78,196.24,207.88,97.19,108.78,79.12,77.08]
        #Std_LRR = [66.49,24.67,57.51,63.15,101.47,112.01,37.36,38.95,38.00,36.90]


        if posture == 'supine':   ##############SUPINE
            KNN = [61.11,62.56,100.10,97.28,170.97,172.12,81.18,101.55,102.27,70.97]
            Std_KNN = [32.23,17.27,80.11,80.74,149.21,147.87,33.44,34.41,57.77,59.93]
            LRR = [129.33,60.72,105.37,105.38,181.24,189.00,93.87,117.35,96.15,91.69]
            Std_LRR = [57.18,21.38,51.86,52.17,94.37,93.75,30.92,33.35,52.84,50.83]
            KRR = [110.91,57.09,87.25,90.23,156.71,158.92,84.33,98.20,74.68,74.15]
            Std_KRR = [46.31,17.56,61.51,64.63,111.10,117.26,28.73,30.71,59.11,57.35]
            DirectCNN = [66.52,53.15,54.02,64.24,97.93,111.55,61.13,90.57,74.21,61.55]
            Std_DirectCNN = [17.77,12.66,38.89,42.21,82.03,90.62,20.82,21.95,35.85,41.64]
            KinCNNcL = [83.98,57.32,84.95,90.66,155.10,158.79,91.71,104.12,83.17,87.57]
            Std_KinCNNcL = [25.42,19.14,63.63,78.40,114.66,125.56,34.00,45.93,50.05,63.34]
            KinCNNvL = [68.36,48.82,59.12,66.81,107.95,122.88,63.19,87.78,73.51,55.85]
            Std_KinCNNvL = [18.01,9.80,42.16,49.46,81.55,98.28,21.08,21.23,34.20,43.96]
            ax.set_ylabel('Supine Posture\n Joint Position Error (mm)')




        elif posture == 'sitting':  ##############SITTING
            KNN = [96.84,54.31,86.68,84.30,145.73,139.11,80.53,107.48,68.77,70.43]
            Std_KNN = [45.48,12.46,62.98,67.10,117.35,113.52,33.11,33.58,40.79,42.65]
            LRR = [141.47,58.64,105.84,108.83,182.52,186.27,94.86,120.07,92.50,88.00]
            Std_LRR = [53.69,22.06,52.89,50.11,90.75,87.13,35.57,33.28,35.17,33.96]
            KRR = [123.62,50.30,88.22,84.68,149.81,141.67,80.65,104.80,74.38,72.89]
            Std_KRR = [45.30,15.26,48.61,59.54,85.05,98.86,29.74,30.90,37.10,38.58]
            DirectCNN = [82.95,50.18,67.41,68.00,116.50,110.99,67.80,91.93,88.31,80.37]
            Std_DirectCNN = [31.45,10.58,43.80,36.22,86.77,75.75,23.18,22.03,25.35,25.41]
            KinCNNcL = [84.95,59.05,94.89,84.52,140.84,140.24,82.18,103.06,110.36,96.87]
            Std_KinCNNcL = [31.52,19.68,52.94,70.53,87.65,110.83,37.09,44.03,47.34,53.56]
            KinCNNvL = [84.79,41.81,74.18,56.91,128.03,102.27,67.01,86.49,79.51,70.94]
            Std_KinCNNvL = [30.65,9.04,44.43,39.16,78.86,76.69,24.02,22.96,32.38,33.10]
            ax.set_ylabel('Sitting Posture \n Joint Position Error (mm)')

        plt.bar(pos, KNN, width,
                         alpha=0.5,
                         color='c',
                            yerr= Std_KNN,
                            ecolor = 'k'
                         #label=Label_L4L5_DP[3]
                         )

        plt.bar([p + width for p in pos], LRR, width,
                         alpha=0.5,
                         color='m',
                            yerr= Std_LRR,
                            ecolor = 'k'
                         #label=Label_L4L5_DP[0]
                         )

        plt.bar([p + 2*width for p in pos], KRR, width,
                         alpha=0.5,
                         color='y',
                            yerr= Std_KRR,
                            ecolor = 'k'
                         #label=Label_L4L5_DP[1]
                         )

        plt.bar([p + 3*width for p in pos], DirectCNN, width,
                         alpha=0.5,
                         color='r',
                            yerr= Std_DirectCNN,
                            ecolor = 'k',
                         #label=Label_L4L5_DP[2]
                         )

        plt.bar([p + 4*width for p in pos], KinCNNcL, width,
                         alpha=0.5,
                         color='g',
                            yerr = Std_KinCNNcL,
                            ecolor = 'k',
                         #yerr= 0,
                         #label=Label_L4L5_DP[2]
                         )

        plt.bar([p + 5*width for p in pos], KinCNNvL, width,
                         alpha=0.5,
                         color='b',
                            yerr= Std_KinCNNvL,
                            ecolor = 'k',
                         #label=Label_L4L5_DP[2]
                         )

        #plt.rc('text', usetex=True)
        # plt.plot([10,10,14,14,10],[2,4,4,2,2],'r')
        #col_labels = ['col1', 'col2', 'col3']
        #row_labels = ['row1', 'row2', 'row3']
        #table_vals = [11, 12, 13, 21, 22, 23, 31, 32, 33]
        #table = r'''\begin{tabular}{ c | c | c | c } & col1 & col2 & col3 \\\hline row1 & 11 & 12 & 13 \\\hline row2 & 21 & 22 & 23 \\\hline  row3 & 31 & 32 & 33 \end{tabular}'''
        #plt.text(1, 3.4, table, size=12)


        # Setting axis labels and ticks
        ax.set_xticks([p + 3.0 * width for p in pos])
        ax.set_xticklabels(Label_L4L5_DP)

        # Setting the x-axis and y-axis limits
        plt.xlim(min(pos)-width, max(pos)+width*4+0.4)
        plt.ylim([0, max(KNN + LRR + KRR + DirectCNN + KinCNNcL + KinCNNvL) * 1.2])

        #ax.text(max(pos)+0.13*0+0.015, KNN[9]+25, str(np.round(KNN[9],2)), color='black', fontweight = 'normal',rotation = 'vertical',fontsize=8)
        #ax.text(max(pos)+0.13*1+0.015, LRR[9]+25, str(np.round(LRR[9],2)), color='black', fontweight = 'normal',rotation = 'vertical',fontsize=8)
        #ax.text(max(pos)+0.13*2+0.015, KRR[9]+25, str(np.round(KRR[9],2)), color='black', fontweight = 'normal',rotation = 'vertical',fontsize=8)
        #ax.text(max(pos)+0.13*3+0.02, DirectCNN[9]+20, str(format(DirectCNN[9], '.2f')), color='black', fontweight='heavy', rotation = 'vertical',fontsize=9)
        #ax.text(max(pos)+0.13*4+0.02, KinCNNcL[9]+20, str(format(KinCNNcL[9], '.2f')), color='black', fontweight = 'normal', rotation = 'vertical',fontsize=8)
        #ax.text(max(pos)+0.13*5+0.02, KinCNNvL[9]+20, str(np.round(KinCNNvL[9],2)), color='black', fontweight='heavy', rotation = 'vertical',fontsize=9)

        # Adding the legend and showing the plot
        if posture == 'supine':
            plt.legend(['KNN', 'LRR', 'KRR', 'Direct ConvNet ',
                        'Kin. ConvNet const. len.', 'Kin. ConvNet regr. len.'], loc=9,
                       bbox_to_anchor=(0., 1.032, 1., .102), ncol=6)  # +r"$\boldsymbol{l}$"

        plt.grid()
        plt.show()






if __name__ == "__main__":

    import optparse
    p = optparse.OptionParser()

    p.add_option('--viz', action='store_true',
                 dest='visualize', \
                 default=False, \
                 help='Train only on data from the arms, both sitting and laying.')

    opt, args = p.parse_args()

    Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/'

    #Initialize trainer with a training database file
    p = DataVisualizer(pkl_directory=Path, opt = opt)

    #p.all_joint_error()
    #p.dropout_std_threshold()
    #p.error_threshold()
    #p.p_information_std()
    #p.final_foot_variance()
    p.all_joint_error()
    #p.final_error()
    sys.exit()
