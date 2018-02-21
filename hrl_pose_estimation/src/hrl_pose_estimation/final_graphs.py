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
        for modeltype in ['anglesSTVL']:
            error_avg_flat = None
            for subject in [9, 10, 11, 12, 13, 14, 15, 16, 17, 18]:
                if modeltype == 'anglesCL' or modeltype == 'anglesSTVL' or modeltype == 'direct':
                    _, _, targets, targets_est = load_pickle(self.dump_path + '/Final_Data/error_avg_std_T25_subject' + str(subject) + '_' + str(modeltype) + '.p')
                    # size (P x N) each: number of images, number of joints

                    print np.shape(targets)
                    print np.shape(targets_est)
                    try:
                        error_avg_flat = np.concatenate((error_avg_flat, np.array(error_avg).flatten()), axis=0)
                    except:
                        error_avg_flat = np.array(error_avg).flatten()
                    print np.shape(error_avg_flat), subject, modeltype

                elif modeltype == 'KNN' or modeltype == 'LRR' or modeltype == 'KRR':
                    error_avg = load_pickle(
                        self.dump_path + '/Final_Data/error_avg_subject' + str(subject) + '_' + str(modeltype) + '.p')
                    try:
                        error_avg_flat = np.concatenate((error_avg_flat, np.array(error_avg).flatten() / 10), axis=0)
                    except:
                        error_avg_flat = np.array(error_avg).flatten() / 10
                    print np.shape(error_avg_flat), subject, modeltype

            error_avg = np.array(error_avg_flat).flatten()

            threshold_error[modeltype] = np.flip(np.linspace(0, np.max(error_avg), num=200), axis=0)
            joint_percent[modeltype] = np.zeros_like(threshold_error[modeltype])
            joint_percent_keep[modeltype] = np.zeros_like(threshold_error[modeltype])

            # print np.mean(error_avg), 'GMPJPE'

            for i in range(threshold_error[modeltype].shape[0]):
                joint_percent_queue = 0.
                joint_percent_queue_keep = 0.

                for j in range(error_avg.shape[0]):
                    # print error_var[j], threshold_variance[i], 'std and threshold'
                    if error_avg[j] <= threshold_error[modeltype][
                        i]:  # if our variance is less than the threshold, keep the value
                        joint_percent_queue += 1
                    if error_avg[j] >= threshold_error[modeltype][i]:
                        joint_percent_queue_keep += 1

                joint_percent[modeltype][i] = joint_percent_queue
                joint_percent_keep[modeltype][i] = joint_percent_queue_keep

            joint_percent[modeltype] = joint_percent[modeltype] / np.max(joint_percent[modeltype])
            joint_percent_keep[modeltype] = joint_percent_keep[modeltype] / np.max(joint_percent_keep[modeltype])


        plt.show()



    def error_threshold(self):
        threshold_error = {}
        joint_percent = {}
        joint_percent_keep = {}
        for modeltype in ['KNN','LRR','KRR','direct','kincL','kinvL']:
            error_avg_flat = None
            for subject in [9,10,11,12,13,14,15,16,17,18]:
                if modeltype == 'kinvL' or modeltype == 'kincL' or modeltype == 'direct':
                    error_avg, _ = load_pickle(self.dump_path+'/Final_Data/error_avg_std_T25_subject'+str(subject)+'_'+str(modeltype)+'.p')
                    # size (P x N) each: number of images, number of joints
                    try:
                        error_avg_flat = np.concatenate((error_avg_flat, np.array(error_avg).flatten()), axis=0)
                    except:
                        error_avg_flat = np.array(error_avg).flatten()
                    print np.shape(error_avg_flat), subject, modeltype

                elif modeltype == 'KNN' or modeltype == 'LRR' or modeltype == 'KRR':
                    error_avg = load_pickle(self.dump_path+'/Final_Data/error_avg_subject'+str(subject)+'_'+str(modeltype)+'.p')
                    try:
                        error_avg_flat = np.concatenate((error_avg_flat, np.array(error_avg).flatten()/10), axis=0)
                    except:
                        error_avg_flat = np.array(error_avg).flatten()/10
                    print np.shape(error_avg_flat), subject, modeltype


            error_avg = np.array(error_avg_flat).flatten()

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
        ylim1 = [0, 1.0]
        fig, ax1 = plt.subplots()
        #plt.suptitle('Subject '+str(subject)+ ' Error Thresholding.  All joints.', fontsize=16)
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim1)
        ax1.set_xlabel('Error Threshold (mm)', fontsize=16)
        ax1.set_ylabel('Fraction of joints \n below the error threshold', color='k', fontsize=16)
        ratioKNN = ax1.plot(threshold_error['KNN'] * 10, joint_percent['KNN'], 'c-', lw=3, label='KNN')
        ratioLRR = ax1.plot(threshold_error['LRR'] * 10, joint_percent['LRR'], 'm-', lw=3, label='LRR')
        ratioKRR = ax1.plot(threshold_error['KRR'] * 10, joint_percent['KRR'], 'y-', lw=3, label='KRR')
        ratiodirect = ax1.plot(threshold_error['direct'] * 10, joint_percent['direct'], 'r-', lw=3, label='CNN direct')
        ratiokincL = ax1.plot(threshold_error['kincL'] * 10, joint_percent['kincL'], 'g-', lw=3, label='CNN kin. avg. '+r"$\boldsymbol{l}$")
        ratiokinvL = ax1.plot(threshold_error['kinvL'] * 10, joint_percent['kinvL'], 'b-', lw=3, label='CNN kin. regr. '+r"$\boldsymbol{l}$")
        lns = ratioKNN+ratioLRR+ratioKRR+ratiodirect+ratiokincL+ratiokinvL
        labs = [l.get_label() for l in lns]
        plt.legend(lns, labs, loc=0)
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
        for subject in [13]:
            error_avg, error_std = load_pickle(self.dump_path + '/Final_Data/error_avg_std_T25_subject'+str(subject)+'_kinvL.p')

            print np.shape(error_avg)
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

        xlim = [0, 70]
        ylim1 = [0, 1.0]
        ylim2 = [0, 100]
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        plt.suptitle('Subject 10 Std. Dev. Thresholding for removing points.  All joints.',fontsize=16)
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim1)
        ax1.set_xlabel('Std. dev. threshold (mm) for discarding data, T=25 MC dropout passes',fontsize=16)
        ax1.set_ylabel('Fraction of joints remaining',color='g',fontsize=16)

        #ax1.set_title('Right Elbow')

        ax2.set_xlim(xlim)
        ax2.set_ylim(ylim2)
        ax2.set_xlabel('Std. dev. threshold (mm) for discarding data, T=25 MC dropout passes',fontsize=16)
        ax2.set_ylabel('GMPJPE error across remaining joints',color='b',fontsize=16)
        ratio = ax1.plot(threshold_variance * 10, joint_percent, 'g-', lw=4, label='Ratio of Joints Remaining')
        gmpjpe = ax2.plot(threshold_variance * 10, GMPJPE * 10, 'b-', lw = 4, label = 'GMPJPE of Remaining Joints')

        lns = ratio + gmpjpe
        labs = [l.get_label() for l in lns]
        plt.legend(lns, labs, loc=4)
        #ax2.set_title('Left Elbow')

        plt.show()

        xlim = [0, 70]
        ylim1 = [0, 1.0]
        ylim2 = [0, 500]
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        #plt.suptitle('Subject 10 Std. Dev. Thresholding for keeping points.  All joints.',fontsize=16)
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim1)
        ax1.set_xlabel('Std. dev. threshold (mm) for discarding data, T=25 MC dropout passes',fontsize=16)
        ax1.set_ylabel('Fraction of joints discarded',color='g',fontsize=16)
        ratio = ax1.plot(threshold_variance*10, joint_percent_keep, 'g-',lw=4, label='Ratio of Joints Discarded')
        #ax1.set_title('Right Elbow')

        ax2.set_xlim(xlim)
        ax2.set_ylim(ylim2)
        ax2.set_xlabel('Std. dev. threshold (mm) for discarding data, T=25 MC dropout passes',fontsize=16)
        ax2.set_ylabel('GMPJPE error across discarded joints',color='b',fontsize=16)
        gmpjpe = ax2.plot(threshold_variance*10, GMPJPE_keep*10, 'b-',lw=4, label = 'GMPJPE of Discarded Joints')
        #ax2.set_title('Left Elbow')

        lns = ratio + gmpjpe
        labs = [l.get_label() for l in lns]
        plt.legend(lns, labs, loc=0)
        plt.show()

        xlim = [0, np.max(error_std_flat)*10]
        ylim1 = [0, 1.0]
        ylim2 = [0, 500]
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        #plt.suptitle('Subject 10 Std. Dev. Thresholding for keeping points.  All joints.',fontsize=16)
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim1)
        ax1.set_xlabel('Std. dev. threshold (mm) for discarding data, T=25 MC dropout passes',fontsize=16)
        ax1.set_ylabel('Fraction of joints discarded',color='g',fontsize=16)
        ratio = ax1.plot(threshold_variance*10, joint_percent_keep, 'g-',lw=4, label='Ratio of Joints Discarded')
        #ax1.set_title('Right Elbow')

        ax2.set_xlim(xlim)
        ax2.set_ylim(ylim2)
        ax2.set_xlabel('Std. dev. threshold (mm) for discarding data, T=25 MC dropout passes',fontsize=16)
        ax2.set_ylabel('GMPJPE error across discarded joints',color='b',fontsize=16)
        gmpjpe = ax2.plot(threshold_variance*10, GMPJPE_keep*10, 'b-',lw=4, label = 'GMPJPE of Discarded Joints')
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
        pos = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        width = 0.13

        # Plotting the bars
        fig, ax = plt.subplots(figsize=(11,4),dpi = 110)

        #############ALL
        #KNN     = [87.63,52.54,103.07,105.55,187.81,189.33,83.39,94.20,80.36,72.86]
        #Std_KNN = [69.62,20.61,81.93,90.03,144.87,152.80,48.23,48.66,56.49,53.94]
        #LRR      = [134.64,56.37,109.70,117.78,196.24,207.88,97.19,108.78,79.12,77.08]
        #Std_LRR = [66.49,24.67,57.51,63.15,101.47,112.01,37.36,38.95,38.00,36.90]


        if posture == 'supine':   ##############SUPINE
            KNN = [59.28,50.51,108.37,111.02,200.77,202.19,81.73,85.26,82.19,68.48,104.98]
            Std_KNN = [27.77,12.95,88.57,99.09,157.01,170.84,44.91,42.59,52.72,51.55,0]
            LRR = [126.94,56.92,107.02,115.50,192.14,207.40,100.43,103.73,75.67,74.60,116.04]
            Std_LRR = [60.83,24.44,57.08,61.06,102.56,110.31,30.30,36.69,36.66,33.51,0]
            KRR = [109.93,50.85,88.28,93.71,162.74,165.14,85.90,90.91,65.99,62.04,97.55]
            Std_KRR = [50.21,17.44,55.54,62.49,103.94,112.84,25.28,29.67,31.08,28.45,0]
            DirectCNN  = [60.43,47.80,53.04,62.11,93.74,108.37,58.79,80.23,62.64,51.95,67.91]
            Std_DirectCNN = [21.28,14.57,46.13,48.66,94.90,101.16,20.21,25.31,23.88,23.431,0]
            KinCNNcL  = [66.60,47.45,70.97,71.62,125.92,122.52,74.30,82.91,71.99,57.10,79.14]
            Std_KinCNNcL = [21.93,15.54,58.56,63.22,120.55,114.32,25.17,24.72,32.71,27.06,0]
            KinCNNvL  = [61.72,46.72,59.81,66.99,108.99,116.04,68.03,79.50,68.01,49.97,72.58]
            Std_KinCNNvL = [19.31,14.11,46.43,58.37,98.78,120.66,24.64,24.91,26.80,24.66,0]
            ax.set_ylabel('Supine Posture: GMPJPE (mm)')

        elif posture == 'sitting':  ##############SITTING
            KNN = [139.83, 56.88, 94.51, 95.29, 164.75, 165.57, 86.91, 109.91, 76.44,80.05,107.01]
            Std_KNN = [64.83,18.20,63.62,63.33,109.18,99.20,44.78,41.30,56.19,52.56,0]
            LRR = [150.14,55.82,114.56,122.39,203.13,209.41,90.89,118.15,85.28,81.02,123.08]
            Std_LRR = [59.78,20.69,54.60,58.90,95.87,103.69,35.61,37.51,34.09,38.14,0]
            KRR = [125.49,52.81,92.56,97.32,159.14,158.83,83.35,103.60,76.38,72.34,102.18]
            Std_KRR = [45.83,15.17,49.90,52.99,86.88,89.51,30.07,28.53,31.62,32.57,0]
            DirectCNN = [80.01,43.62,65.55,62.20,110.98,102.38,62.78,83.67,75.03,70.20,75.64]
            Std_DirectCNN = [34.36,9.66,50.45,44.90,93.93,85.01,23.89,23.79,23.40,23.59,0]
            KinCNNcL = [82.23,52.12,72.47,65.63,123.63,116.01,99.39,110.78,71.29,74.06,86.76]
            Std_KinCNNcL = [34.98,13.90,53.12,65.57,104.36,103.51,25.83,23.15,29.04,28.55,0]
            KinCNNvL = [84.63,50.06,69.18,60.21,119.44,100.02,85.51,98.70,70.13,66.01,80.39]
            Std_KinCNNvL = [32.60,12.54,51.54,45.89,94.74,87.94,26.97,22.69,23.46,24.11,0]
            ax.set_ylabel('Sitting Posture: GMPJPE (mm)')

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
        plt.ylim([0, max(KNN + LRR + KRR + DirectCNN + KinCNNcL + KinCNNvL) * 1.07])

        ax.text(max(pos)+0.13*0+0.015, KNN[10]+25, str(np.round(KNN[10],2)), color='black', fontweight = 'normal',rotation = 'vertical',fontsize=8)
        ax.text(max(pos)+0.13*1+0.015, LRR[10]+25, str(np.round(LRR[10],2)), color='black', fontweight = 'normal',rotation = 'vertical',fontsize=8)
        ax.text(max(pos)+0.13*2+0.015, KRR[10]+25, str(np.round(KRR[10],2)), color='black', fontweight = 'normal',rotation = 'vertical',fontsize=8)
        ax.text(max(pos)+0.13*3+0.02, DirectCNN[10]+20, str(format(DirectCNN[10], '.2f')), color='black', fontweight='heavy', rotation = 'vertical',fontsize=9)
        ax.text(max(pos)+0.13*4+0.02, KinCNNcL[10]+20, str(format(KinCNNcL[10], '.2f')), color='black', fontweight = 'normal', rotation = 'vertical',fontsize=8)
        ax.text(max(pos)+0.13*5+0.02, KinCNNvL[10]+20, str(np.round(KinCNNvL[10],2)), color='black', fontweight='heavy', rotation = 'vertical',fontsize=9)

        # Adding the legend and showing the plot
        if posture == 'supine':
            plt.legend(['KNN', 'LRR', 'KRR', 'CNN direct ' + r"$\boldsymbol{s}_{j=1..N}$",
                        'CNN kin. avg. ' + r"$\boldsymbol{l}$", 'CNN kin. regr. ' + r"$\boldsymbol{l}$"], loc=9,
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
    p.p_information_std()
    sys.exit()
