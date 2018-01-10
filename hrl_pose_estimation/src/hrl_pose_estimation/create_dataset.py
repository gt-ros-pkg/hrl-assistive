#!/usr/bin/env python
import sys
import os
import numpy as np
import cPickle as pkl
import random
import math
from scipy.stats import mode
from time import sleep
# ROS
import roslib#; roslib.load_manifest('hrl_pose_estimation')
import rospy
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

# Graphics
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d import Axes3D

# Machine Learning
from scipy import ndimage
from scipy.ndimage.filters import gaussian_filter
## from skimage import data, color, exposure
from sklearn.decomposition import PCA

# HRL libraries
import hrl_lib.util as ut
import pickle
#roslib.load_manifest('hrl_lib')
from hrl_lib.util import load_pickle

# Pose Estimation Libraries
from create_dataset_lib import CreateDatasetLib
from kinematics_lib import KinematicsLib
from visualization_lib import VisualizationLib



import tf.transformations as tft

MAT_WIDTH = 0.762 #metres
MAT_HEIGHT = 1.854 #metres
MAT_HALF_WIDTH = MAT_WIDTH/2 
NUMOFTAXELS_X = 84#73 #taxels
NUMOFTAXELS_Y = 47#30
INTER_SENSOR_DISTANCE = 0.0286#metres
LOW_TAXEL_THRESH_X = 0
LOW_TAXEL_THRESH_Y = 0
HIGH_TAXEL_THRESH_X = (NUMOFTAXELS_X - 1) 
HIGH_TAXEL_THRESH_Y = (NUMOFTAXELS_Y - 1) 

 
class DatabaseCreator():
    '''Gets the directory of pkl database and iteratively go through each file,
    cutting up the pressure maps and creating synthetic database'''
    def __init__(self, training_database_pkl_directory, save_pdf=False, verbose=False, select=False):

        # Set initial parameters
        self.training_dump_path = training_database_pkl_directory.rstrip('/')
        self.final_database_path = (
        os.path.abspath(os.path.join(self.training_dump_path, os.pardir)))

        self.verbose = verbose
        self.select = select
        self.world_to_mat = CreateDatasetLib().world_to_mat
        self.mat_to_taxels = CreateDatasetLib().mat_to_taxels

        if self.verbose: print 'The final database path is: ',self.final_database_path
        try:
            self.final_dataset = load_pickle(self.final_database_path+'/basic_train_dataset.p')
            print 'y'
        except IOError:
            print 'x'

        self.final_dataset = {}


        print self.training_dump_path
        [self.p_world_mat, self.R_world_mat] = load_pickle('/home/henryclever/hrl_file_server/Autobed/pose_estimation_data/mat_axes15.p')
        self.mat_size_orig = (NUMOFTAXELS_X - 20, NUMOFTAXELS_Y - 20)
        self.mat_size = (NUMOFTAXELS_X,NUMOFTAXELS_Y)
        self.individual_dataset = {}


        home_sup_dat = load_pickle(self.training_dump_path + '4/home_sup.p')
        if self.verbose: print "Checking database for empty values."
        empty_count = 0
        for entry in range(len(home_sup_dat)):
            home_joint_val = home_sup_dat[entry][1]
            if len(home_joint_val.flatten()) < (30) or (home_sup_dat[entry][0] < self.mat_size[0]*self.mat_size[1]):
                empty_count += 1
                del home_sup_dat[entry]

        if self.verbose: print "Empty value check results: {} rogue entries found".format(
            empty_count)

        # Targets in the mat frame
        home_sup_pressure_map = home_sup_dat[0][0]
        home_sup_joint_pos_world = home_sup_dat[0][1]
        home_sup_joint_pos = self.world_to_mat(home_sup_joint_pos_world, self.p_world_mat, self.R_world_mat)  # N x 3

        # print home_sup_joint_pos
        print len(home_sup_pressure_map)
        print len(home_sup_joint_pos), 'sizes'

        self.T = np.zeros((4, 1))
        self.H = np.zeros((4, 1))

        self.r_S = np.zeros((4, 1))
        self.r_E = np.zeros((4, 1))
        self.r_H = np.zeros((4, 1))
        self.T = np.zeros((4, 1))
        self.H = np.zeros((4, 1))
        self.N = np.zeros((4, 1))

        self.r_S = np.zeros((4, 1))
        self.r_E = np.zeros((4, 1))
        self.r_H = np.zeros((4, 1))
        self.Pr_SE = np.zeros((4, 1))
        self.Pr_EH = np.zeros((4, 1))
        self.r_SE = np.zeros((3, 1))
        self.r_SEn = np.zeros((3, 1))
        self.r_SH = np.zeros((3, 1))
        self.r_SHn = np.zeros((3, 1))
        self.r_Spxn = np.zeros((3, 1))
        self.r_Spxm = np.zeros((3, 1))
        self.r_Spxzn = np.zeros((3, 1))
        self.r_Spx = np.zeros((3, 1))
        self.r_SH_pp = np.zeros((3, 1))
        self.r_Spx_pp = np.zeros((3, 1))
        self.r_ROT_OS = np.matrix('0.,0.,-1.;0.,1.,0.;-1.,0.,0.')
        self.ROT_ON = np.matrix('1.,0.,0.;0.,0.,1.;0.,1.,0.')
        self.ROT_bed = np.matrix('1.,0.,0.;0.,1.,0.;0.,0.,1.')

        self.r_G = np.zeros((4,1))
        self.r_K = np.zeros((4,1))
        self.r_A = np.zeros((4,1))
        self.Pr_GK = np.zeros((4,1))
        self.Pr_KA = np.zeros((4,1))
        self.r_GK = np.zeros((3, 1))
        self.r_GKn = np.zeros((3, 1))
        self.r_GA = np.zeros((3, 1))
        self.r_GAn = np.zeros((3, 1))
        self.r_Gpxn = np.zeros((3, 1))
        self.r_Gpxm = np.zeros((3, 1))
        self.r_Gpxzn = np.zeros((3, 1))
        self.r_Gpx = np.zeros((3, 1))
        self.r_GA_pp = np.zeros((3, 1))
        self.r_Gpx_pp = np.zeros((3, 1))


        self.l_S = np.zeros((4, 1))
        self.l_E = np.zeros((4, 1))
        self.l_H = np.zeros((4, 1))
        self.Pl_SE = np.zeros((4, 1))
        self.Pl_EH = np.zeros((4, 1))
        self.l_SE = np.zeros((3, 1))
        self.l_SEn = np.zeros((3, 1))
        self.l_SH = np.zeros((3, 1))
        self.l_SHn = np.zeros((3, 1))
        self.l_Spxn = np.zeros((3, 1))
        self.l_Spxm = np.zeros((3, 1))
        self.l_Spxzn = np.zeros((3, 1))
        self.l_Spx = np.zeros((3, 1))
        self.l_SH_pp = np.zeros((3, 1))
        self.l_Spx_pp = np.zeros((3, 1))
        self.l_ROT_OS = np.matrix('0.,0.,1.;0.,1.,0.;1.,0.,0.')

        self.l_G = np.zeros((4,1))
        self.l_K = np.zeros((4,1))
        self.l_A = np.zeros((4,1))
        self.Pl_GK = np.zeros((4,1))
        self.Pl_KA = np.zeros((4,1))
        self.l_GK = np.zeros((3, 1))
        self.l_GKn = np.zeros((3, 1))
        self.l_GA = np.zeros((3, 1))
        self.l_GAn = np.zeros((3, 1))
        self.l_Gpxn = np.zeros((3, 1))
        self.l_Gpxm = np.zeros((3, 1))
        self.l_Gpxzn = np.zeros((3, 1))
        self.l_Gpx = np.zeros((3, 1))
        self.l_GA_pp = np.zeros((3, 1))
        self.l_Gpx_pp = np.zeros((3, 1))

        self.pseudoheight = {'1': 1.53, '2': 1.42, '3': 1.52, '4': 1.63, '5': 1.66, '6': 1.59, '7': 1.49, '8': 1.53,
                             '9': 1.69, '10': 1.58, '11': 1.64, '12': 1.45, '13': 1.58, '14': 1.67, '15': 1.63,
                             '16': 1.48,
                             '17': 1.43, '18': 1.54}

        self.bedangle = 0.
        self.count = 0

        rospy.init_node('calc_mean_std_of_head_detector_node')


    def visualize_single_pressure_map(self, p_map_raw, targets_raw=None):
        print p_map_raw.shape, 'pressure mat size'
        print targets_raw, 'targets ra'

        p_map = np.asarray(np.reshape(p_map_raw, self.mat_size))
        fig = plt.figure()

        # set options
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        xlim = [-2.0, 49.0]
        ylim = [86.0, -2.0]
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim)

        # background
        ax1.set_axis_bgcolor('cyan')

        # Visualize pressure maps
        ax1.imshow(p_map, interpolation='nearest', cmap=
        plt.cm.bwr, origin='upper', vmin=0, vmax=100)

        # Visualize targets
        if targets_raw is not None:
            if type(targets_raw) == list:
                targets_raw = np.array(targets_raw)
            if len(np.shape(targets_raw)) == 1:
                targets_raw = np.reshape(targets_raw, (len(targets_raw) / 3, 3))

            target_coord = targets_raw[:, :2] / INTER_SENSOR_DISTANCE
            target_coord[:, 1] -= (NUMOFTAXELS_X - 1)
            target_coord[:, 1] *= -1.0
            ax1.plot(target_coord[:, 0], target_coord[:, 1], 'y*', ms=8)
        plt.title('Pressure Mat and Targets')

        if self.verbose ==True: print targets_raw

        file_size = len(self.final_dataset) * 0.08958031837
        print 'output file size: ~', int(file_size), 'Mb'

        targets_raw_z = []
        for idx in targets_raw: targets_raw_z.append(idx[2])


        x = np.arange(0,10)
        ax2.bar(x, targets_raw_z)
        plt.xticks(x+0.5, ('Head', 'Torso', 'R Elbow', 'L Elbow', 'R Hand', 'L Hand', 'R Knee', 'L Knee', 'R Foot', 'L Foot'), rotation='vertical')
        plt.title('Distance above Bed')
        axkeep = plt.axes([0.01, 0.05, 0.08, 0.075])
        axdisc = plt.axes([0.01, 0.15, 0.08, 0.075])
        bdisc = Button(axdisc, 'Discard')
        bdisc.on_clicked(self.discard)
        bkeep = Button(axkeep, 'Keep')
        bkeep.on_clicked(self.keep)

        plt.show()

        return



    def pad_pressure_mats(self,HxWimages):
        HxWimages = np.asarray(HxWimages)
        HxWimages = np.reshape(HxWimages, self.mat_size_orig)

        padded = np.zeros((HxWimages.shape[0]+20,HxWimages.shape[1]+20))
        padded[10:74,10:37] = HxWimages
        HxWimages = list(padded.flatten())

        return HxWimages

    def discard(self, event):
        plt.close()
        self.keep_image = False

    def keep(self, event):
        plt.close()
        self.keep_image = True

    def rand_index_p_length(self, p_file):
        #this makes a new list of integers in the range of the p file length. It shuffles them.
        indexList = range(0, len(p_file))
        random.shuffle(indexList)
        return indexList

    def head_origin_callback(self, data):
        '''This callback will sample data until its asked to stop'''
        self.head_pose = [data.transform.translation.x,
                         data.transform.translation.y,
                         data.transform.translation.z]



    def inverse_arm_kinematics(self, targets, subject, bedangle):

        lengths = np.zeros((9))  # torso height, torso vert, shoulder right, shoulder left, upper arm right, upper arm left, forearm right, forearm left
        angles = np.zeros((10))  # sh right roll, sh left roll, sh right pitch, sh left pitch, sh right yaw, sh left yaw, elbow right, elbow left
        pseudotargets = np.zeros((3,3))

        lengths[0] = 0.1  # torso height
        lengths[1] = 0.2065 * self.pseudoheight[str(subject)] - 0.0529  # about 0.25. torso vert
        lengths[2] = 0.13454 * self.pseudoheight[str(subject)] - 0.03547  # about 0.15. shoulder right
        lengths[3] = 0.13454 * self.pseudoheight[str(subject)] - 0.03547  # about 0.15. shoulder left

        self.H[0, 0] = targets[0, 0]
        self.H[1, 0] = targets[0, 1]
        self.H[2, 0] = targets[0, 2]
        self.H[3, 0] = 1.

        self.T[0, 0] = targets[1, 0]
        self.T[1, 0] = targets[1, 1]
        self.T[2, 0] = targets[1, 2]
        self.T[3, 0] = 1.

        self.r_E[0, 0] = targets[2, 0]
        self.r_E[1, 0] = targets[2, 1]
        self.r_E[2, 0] = targets[2, 2]
        self.r_E[3, 0] = 1.

        self.l_E[0, 0] = targets[3, 0]
        self.l_E[1, 0] = targets[3, 1]
        self.l_E[2, 0] = targets[3, 2]
        self.l_E[3, 0] = 1.

        self.r_H[0, 0] = targets[4, 0]
        self.r_H[1, 0] = targets[4, 1]
        self.r_H[2, 0] = targets[4, 2]
        self.r_H[3, 0] = 1.

        self.l_H[0, 0] = targets[5, 0]
        self.l_H[1, 0] = targets[5, 1]
        self.l_H[2, 0] = targets[5, 2]
        self.l_H[3, 0] = 1.

        # here we construct pseudo ground truths for the neck and shoulders by making fixed translations from the torso
        self.N[0, 0] = self.T[0, 0]
        self.N[1, 0] = self.T[1, 0] + lengths[1] * np.cos(np.deg2rad(bedangle * 0.75))
        self.N[2, 0] = self.T[2, 0] - lengths[0] + lengths[1] * np.sin(np.deg2rad(bedangle * 0.75))
        self.N[3, 0] = 1

        self.r_S[0, 0] = self.T[0, 0] - lengths[2]
        self.r_S[1, 0] = self.T[1, 0] + lengths[1] * np.cos(np.deg2rad(bedangle * 0.75))
        self.r_S[2, 0] = self.T[2, 0] - lengths[0] + lengths[1] * np.sin(np.deg2rad(bedangle * 0.75))
        self.r_S[3, 0] = 1

        self.l_S[0, 0] = self.T[0, 0] + lengths[2]
        self.l_S[1, 0] = self.T[1, 0] + lengths[1] * np.cos(np.deg2rad(bedangle * 0.75))
        self.l_S[2, 0] = self.T[2, 0] - lengths[0] + lengths[1] * np.sin(np.deg2rad(bedangle * 0.75))
        self.l_S[3, 0] = 1

        pseudotargets[0, 0] = self.N[0, 0]
        pseudotargets[0, 1] = self.N[1, 0]
        pseudotargets[0, 2] = self.N[2, 0]

        pseudotargets[1, 0] = self.r_S[0, 0]
        pseudotargets[1, 1] = self.r_S[1, 0]
        pseudotargets[1, 2] = self.r_S[2, 0]

        pseudotargets[2, 0] = self.l_S[0, 0]
        pseudotargets[2, 1] = self.l_S[1, 0]
        pseudotargets[2, 2] = self.l_S[2, 0]

        # get the length of the right shoulder to right elbow
        lengths[4] = np.linalg.norm(self.r_E - self.r_S)
        lengths[5] = np.linalg.norm(self.l_E - self.l_S)

        # parameter for the length between hand and elbow. Should be around 0.2 meters.
        lengths[6] = np.linalg.norm(self.r_H - self.r_E)
        lengths[7] = np.linalg.norm(self.l_H - self.l_E)

        # get the length between the neck and head
        lengths[8] = np.linalg.norm(self.H - self.N)


        # To find the angles we also need to rotate by the bed angle
        self.ROT_bed[1, 1] = np.cos(np.deg2rad(-bedangle * 0.75))
        self.ROT_bed[1, 2] = -np.sin(np.deg2rad(-bedangle * 0.75))
        self.ROT_bed[2, 1] = np.sin(np.deg2rad(-bedangle * 0.75))
        self.ROT_bed[2, 2] = np.cos(np.deg2rad(-bedangle * 0.75))

        # get the neck yaw
        NH_mag = np.copy(lengths[8])
        self.NH = self.H[0:3] - self.N[0:3]
        self.NH = np.matmul(np.matmul(self.ROT_ON, self.ROT_bed), self.NH)
        if NH_mag > 0: self.NHn = np.copy(self.NH) / NH_mag
        angles[8] = np.degrees(np.arcsin(self.NHn[0, 0]))

        # get the shoulder pitch
        rSE_mag = np.copy(lengths[4])
        self.r_SE = self.r_S[0:3] - self.r_E[0:3]
        self.r_SE = np.matmul(np.matmul(self.r_ROT_OS, self.ROT_bed), self.r_SE)
        if rSE_mag > 0: self.r_SEn = np.copy(self.r_SE) / rSE_mag
        angles[2] = -np.degrees(np.arcsin(self.r_SEn[1, 0]))

        lSE_mag = np.copy(lengths[5])
        self.l_SE = self.l_S[0:3] - self.l_E[0:3]
        self.l_SE = np.matmul(np.matmul(self.l_ROT_OS, self.ROT_bed), self.l_SE)
        if lSE_mag > 0: self.l_SEn = np.copy(self.l_SE) / lSE_mag
        angles[3] = -np.degrees(np.arcsin(self.l_SEn[1, 0]))


        #get the neck pitch
        angles[9] = np.degrees(np.arctan(self.NHn[2, 0] / self.NHn[1, 0]))
        print self.N - self.H

        # get shoulder yaw
        angles[4] = -np.degrees(np.arctan(self.r_SEn[0, 0] / self.r_SEn[2, 0]))
        if self.r_S[0] - self.r_E[0] < 0: #this is to correct for the arc tan flip when the shoulder and elbow x flip
            angles[4] = -angles[4]
        angles[5] = +np.degrees(np.arctan(self.l_SEn[0, 0] / self.l_SEn[2, 0]))
        if self.l_S[0] - self.l_E[0] > 0: #this is to correct for the arc tan flip when the shoulder and elbow x flip
            angles[5] = -angles[5]

        # get the elbow angle
        rSH_mag = np.linalg.norm(self.r_H - self.r_S)
        # now apply law of cosines
        angles[6] = np.degrees(np.arccos((np.square(lengths[4]) + np.square(
            lengths[6]) - np.square(rSH_mag)) / (2 * lengths[4] * lengths[6])))
        lSH_mag = np.linalg.norm(self.l_H - self.l_S)
        # now apply law of cosines
        angles[7] = np.degrees(np.arccos((np.square(lengths[5]) + np.square(
            lengths[7]) - np.square(lSH_mag)) / (2 * lengths[5] * lengths[7])))


        # calculate the shoulder roll
        self.r_SH = self.r_S[0:3] - self.r_H[0:3]  # first get distance between shoulder and hand
        self.r_SH = np.matmul(np.matmul(self.r_ROT_OS, self.ROT_bed), self.r_SH)
        self.r_SHn = np.copy(self.r_SH) / rSH_mag
        sEndotsHn = np.copy(
            self.r_SEn[0, 0] * self.r_SHn[0, 0] + self.r_SEn[1, 0] * self.r_SHn[1, 0] + self.r_SEn[2, 0] *
            self.r_SHn[2, 0] - self.r_SEn[0, 0])
        self.r_Spxm[0, 0] = 1.
        if np.linalg.norm(self.r_SEn) > 0:
            self.r_Spxm[1, 0] = sEndotsHn / (self.r_SEn[1, 0] + np.square(self.r_SEn[2, 0]) / self.r_SEn[1, 0])
            self.r_Spxm[2, 0] = sEndotsHn / (self.r_SEn[2, 0] + np.square(self.r_SEn[1, 0]) / self.r_SEn[2, 0])
        # print self.r_Spxm, self.r_SEn
        self.r_Spxn = np.copy(self.r_Spxm / np.linalg.norm(self.r_Spxm))
        self.orig = np.copy(self.r_Spxn)
        self.r_Spx = np.copy(self.r_Spxn) * rSH_mag
        if np.linalg.norm(self.r_SE) > 0:
            self.r_SH_pp = - self.r_SE * (
                self.r_SE[0, 0] * self.r_SH[0, 0] + self.r_SE[1, 0] * self.r_SH[1, 0] + self.r_SE[2, 0] * self.r_SH[
                    2, 0]) / (np.linalg.norm(self.r_SE) * np.linalg.norm(self.r_SE)) + self.r_SH
        self.r_SH_pp_mag = np.linalg.norm(self.r_SH_pp)
        if np.linalg.norm(self.r_SE) > 0:
            self.r_Spx_pp = - self.r_SE * (np.dot(self.r_SE.T, self.r_Spx)) / (
                np.linalg.norm(self.r_SE) * np.linalg.norm(self.r_SE)) + self.r_Spx
        self.r_Spx_pp_mag = np.linalg.norm(self.r_Spx_pp)
        angles[0] = np.degrees(np.arccos(np.dot(self.r_SH_pp.T, self.r_Spx_pp) / (
            self.r_SH_pp_mag * self.r_Spx_pp_mag)))  # np.degrees(np.arctan2(np.cross(self.r_Spx_pp.T,self.r_SH_pp.T)[0],np.dot(self.r_SH_pp.T,self.r_Spx_pp)[0]))#
        if np.cross(self.r_SH.T, self.r_SE.T)[0][0] < 0:
            angles[0] = -angles[0]

        #if self.r_E[1] - self.r_H[1] < 0:
        #    if self.r_E[2] - self.r_H[2] < 0:
                #print 'z neg'
                #angles[0] +=angles[2] + angles[4]
            #else:
                #angles[0] -= angles[2]
            #print self.r_S - self.r_H
            #print self.r_E[0] - self.r_H[2],'rEH x'
            #print self.r_E[2] - self.r_H[2]

            #print angles[2], angles[4]
            #angles[0] = angles[0] + 90

        self.l_SH = self.l_S[0:3] - self.l_H[0:3]  # first get distance between shoulder and hand
        self.l_SH = np.matmul(np.matmul(self.l_ROT_OS, self.ROT_bed), self.l_SH)
        self.l_SHn = np.copy(self.l_SH) / lSH_mag
        sEndotsHn = np.copy(
            self.l_SEn[0, 0] * self.l_SHn[0, 0] + self.l_SEn[1, 0] * self.l_SHn[1, 0] + self.l_SEn[2, 0] *
            self.l_SHn[2, 0] - self.l_SEn[0, 0])
        self.l_Spxm[0, 0] = 1.
        if np.linalg.norm(self.l_SEn) > 0:
            self.l_Spxm[1, 0] = sEndotsHn / (self.l_SEn[1, 0] + np.square(self.l_SEn[2, 0]) / self.l_SEn[1, 0])
            self.l_Spxm[2, 0] = sEndotsHn / (self.l_SEn[2, 0] + np.square(self.l_SEn[1, 0]) / self.l_SEn[2, 0])
        # print self.l_Spxm, self.l_SEn
        self.l_Spxn = np.copy(self.l_Spxm / np.linalg.norm(self.l_Spxm))
        self.orig = np.copy(self.l_Spxn)
        self.l_Spx = np.copy(self.l_Spxn) * lSH_mag
        if np.linalg.norm(self.l_SE) > 0:
            self.l_SH_pp = - self.l_SE * (
                self.l_SE[0, 0] * self.l_SH[0, 0] + self.l_SE[1, 0] * self.l_SH[1, 0] + self.l_SE[2, 0] * self.l_SH[
                    2, 0]) / (np.linalg.norm(self.l_SE) * np.linalg.norm(self.l_SE)) + self.l_SH
        self.l_SH_pp_mag = np.linalg.norm(self.l_SH_pp)
        if np.linalg.norm(self.l_SE) > 0:
            self.l_Spx_pp = - self.l_SE * (np.dot(self.l_SE.T, self.l_Spx)) / (
                np.linalg.norm(self.l_SE) * np.linalg.norm(self.l_SE)) + self.l_Spx
        self.l_Spx_pp_mag = np.linalg.norm(self.l_Spx_pp)
        angles[1] = np.degrees(np.arccos(np.dot(self.l_SH_pp.T, self.l_Spx_pp) / (
            self.l_SH_pp_mag * self.l_Spx_pp_mag)))  # np.degrees(np.arctan2(np.cross(self.l_Spx_pp.T,self.l_SH_pp.T)[0],np.dot(self.l_SH_pp.T,self.l_Spx_pp)[0]))#
        angles[1] = (180 - angles[1])
        if np.cross(self.l_SH.T, self.l_SE.T)[0][0] < 0:
            angles[1] = -angles[1]



        print angles, 'UEP'


        return lengths, angles, pseudotargets


    def inverse_leg_kinematics(self, targets, subject, bedangle):

        lengths = np.zeros((8))  # torso height, torso vert, glute right, glute left, thigh right, thigh left, calf right, calf left
        angles = np.zeros((8))  # hip right roll, hip left roll, hip right pitch, hip left pitch, hip right yaw, hip left yaw, knee right, knee left
        pseudotargets = np.zeros((2,3))
        print self.pseudoheight[str(subject)], 'subject ', subject, 'height'

        lengths[0] = 0.14  # torso height
        #lengths[1] = 0.2065 * self.pseudoheight[str(subject)] - 0.0529  # about 0.25. torso vert
        #.85 is .0354, .0241
        #.8 is .0328, .0239
        #.75 is .0314, .0239


        lengths[1] = 0.1549 * self.pseudoheight[str(subject)] - 0.03968 # about 0.25. torso vert
        lengths[2] = 0.08072 * self.pseudoheight[str(subject)] - 0.02128  # Equal to 0.6 times the equivalent neck to shoulder. glute right
        lengths[3] = 0.08072 * self.pseudoheight[str(subject)] - 0.02128 # Equal to 0.6 times the equivalent neck to shoulder. glute left

        self.T[0, 0] = targets[1, 0]
        self.T[1, 0] = targets[1, 1]
        self.T[2, 0] = targets[1, 2]
        self.T[3, 0] = 1.

        self.r_K[0, 0] = targets[6, 0]
        self.r_K[1, 0] = targets[6, 1]
        self.r_K[2, 0] = targets[6, 2]
        self.r_K[3, 0] = 1.

        self.l_K[0, 0] = targets[7, 0]
        self.l_K[1, 0] = targets[7, 1]
        self.l_K[2, 0] = targets[7, 2]
        self.l_K[3, 0] = 1.

        self.r_A[0, 0] = targets[8, 0]
        self.r_A[1, 0] = targets[8, 1]
        self.r_A[2, 0] = targets[8, 2]
        self.r_A[3, 0] = 1.

        self.l_A[0, 0] = targets[9, 0]
        self.l_A[1, 0] = targets[9, 1]
        self.l_A[2, 0] = targets[9, 2]
        self.l_A[3, 0] = 1.

        # here we construct pseudo ground truths for the shoulders by making fixed translations from the torso
        self.r_G[0, 0] = self.T[0, 0] - lengths[2]
        self.r_G[1, 0] = self.T[1, 0] - lengths[1] #* np.cos(np.deg2rad(bedangle * 0.75))
        self.r_G[2, 0] = self.T[2, 0] - lengths[0] #+ lengths[1] * np.sin(np.deg2rad(bedangle * 0.75))
        self.r_G[3, 0] = 1

        self.l_G[0, 0] = self.T[0, 0] + lengths[2]
        self.l_G[1, 0] = self.T[1, 0] - lengths[1] #* np.cos(np.deg2rad(bedangle * 0.75))
        self.l_G[2, 0] = self.T[2, 0] - lengths[0] #+ lengths[1] * np.sin(np.deg2rad(bedangle * 0.75))
        self.l_G[3, 0] = 1

        pseudotargets[0, 0] = self.r_G[0, 0]
        pseudotargets[0, 1] = self.r_G[1, 0]
        pseudotargets[0, 2] = self.r_G[2, 0]

        pseudotargets[1, 0] = self.l_G[0, 0]
        pseudotargets[1, 1] = self.l_G[1, 0]
        pseudotargets[1, 2] = self.l_G[2, 0]

        # get the length of the right shoulder to right elbow
        lengths[4] = np.linalg.norm(self.r_K - self.r_G)
        lengths[5] = np.linalg.norm(self.l_K - self.l_G)

        # parameter for the length between hand and elbow. Should be around 0.2 meters.
        lengths[6] = np.linalg.norm(self.r_A - self.r_K)
        lengths[7] = np.linalg.norm(self.l_A - self.l_K)



        # get the shoulder pitch
        rGK_mag = np.copy(lengths[4])
        self.r_GK = self.r_G[0:3] - self.r_K[0:3]
        self.r_GK = np.matmul(self.r_ROT_OS, self.r_GK)
        if rGK_mag > 0: self.r_GKn = np.copy(self.r_GK) / rGK_mag
        angles[2] = -np.degrees(np.arcsin(self.r_GKn[1, 0]))

        lGK_mag = np.copy(lengths[5])
        self.l_GK = self.l_G[0:3] - self.l_K[0:3]
        self.l_GK = np.matmul(self.l_ROT_OS, self.l_GK)
        if lGK_mag > 0: self.l_GKn = np.copy(self.l_GK) / lGK_mag
        angles[3] = -np.degrees(np.arcsin(self.l_GKn[1, 0]))

        # get glute yaw
        angles[4] = np.degrees(np.arctan(self.r_GKn[0, 0] / self.r_GKn[2, 0]))
        if self.r_G[0] - self.r_K[0] > 0: #this is to correct for the arc tan flip when the shoulder and elbow x flip
            angles[4] = angles[4] + 180

        angles[5] = np.degrees(np.arctan(self.l_GKn[0, 0] / self.l_GKn[2, 0]))
        if self.l_G[0] - self.l_K[0] > 0: #this is to correct for the arc tan flip when the shoulder and elbow x flip
            angles[5] = angles[5] + 180

        # get the knee angle
        rGA_mag = np.linalg.norm(self.r_A - self.r_G)
        # now apply law of cosines
        angles[6] = np.degrees(np.arccos((np.square(lengths[4]) + np.square(
            lengths[6]) - np.square(rGA_mag)) / (2 * lengths[4] * lengths[6])))
        lGA_mag = np.linalg.norm(self.l_A - self.l_G)
        # now apply law of cosines
        angles[7] = np.degrees(np.arccos((np.square(lengths[5]) + np.square(
            lengths[7]) - np.square(lGA_mag)) / (2 * lengths[5] * lengths[7])))


        # calculate the glute roll
        self.r_GA = self.r_G[0:3] - self.r_A[0:3]  # first get distance between shoulder and hand
        self.r_GA = np.matmul(self.r_ROT_OS, self.r_GA)
        self.r_GAn = np.copy(self.r_GA) / rGA_mag
        gKndotgAn = np.copy(
            self.r_GKn[0, 0] * self.r_GAn[0, 0] + self.r_GKn[1, 0] * self.r_GAn[1, 0] + self.r_GKn[2, 0] *
            self.r_GAn[2, 0] - self.r_GKn[0, 0])
        self.r_Gpxm[0, 0] = 1.
        if np.linalg.norm(self.r_GKn) > 0:
            self.r_Gpxm[1, 0] = gKndotgAn / (self.r_GKn[1, 0] + np.square(self.r_GKn[2, 0]) / self.r_GKn[1, 0])
            self.r_Gpxm[2, 0] = gKndotgAn / (self.r_GKn[2, 0] + np.square(self.r_GKn[1, 0]) / self.r_GKn[2, 0])
        # print self.r_Gpxm, self.r_GKn
        self.r_Gpxn = np.copy(self.r_Gpxm / np.linalg.norm(self.r_Gpxm))
        self.r_Gpx = np.copy(self.r_Gpxn) * rGA_mag
        if np.linalg.norm(self.r_GK) > 0:
            self.r_GA_pp = - self.r_GK * (
                self.r_GK[0, 0] * self.r_GA[0, 0] + self.r_GK[1, 0] * self.r_GA[1, 0] + self.r_GK[2, 0] * self.r_GA[
                    2, 0]) / (np.linalg.norm(self.r_GK) * np.linalg.norm(self.r_GK)) + self.r_GA
        self.r_GA_pp_mag = np.linalg.norm(self.r_GA_pp)
        if np.linalg.norm(self.r_GK) > 0:
            self.r_Gpx_pp = - self.r_GK * (np.dot(self.r_GK.T, self.r_Gpx)) / (
                np.linalg.norm(self.r_GK) * np.linalg.norm(self.r_GK)) + self.r_Gpx
        self.r_Gpx_pp_mag = np.linalg.norm(self.r_Gpx_pp)
        angles[0] = np.degrees(np.arccos(np.dot(self.r_GA_pp.T, self.r_Gpx_pp) / (
            self.r_GA_pp_mag * self.r_Gpx_pp_mag)))  # np.degrees(np.arctan2(np.cross(self.r_Gpx_pp.T,self.r_GA_pp.T)[0],np.dot(self.r_GA_pp.T,self.r_Gpx_pp)[0]))#
        if np.cross(self.r_GA.T, self.r_GK.T)[0][0] < 0:
            angles[0] = -angles[0]

        #if self.r_E[1] - self.r_H[1] < 0:
        #    if self.r_E[2] - self.r_H[2] < 0:
                #print 'z neg'
                #angles[0] +=angles[2] + angles[4]
            #else:
                #angles[0] -= angles[2]
            #print self.r_G - self.r_H
            #print self.r_E[0] - self.r_H[2],'rEH x'
            #print self.r_E[2] - self.r_H[2]

            #print angles[2], angles[4]
            #angles[0] = angles[0] + 90

        self.l_GA = self.l_G[0:3] - self.l_A[0:3]  # first get distance between shoulder and hand
        self.l_GA = np.matmul(self.l_ROT_OS, self.l_GA)
        self.l_GAn = np.copy(self.l_GA) / lGA_mag
        gKndotgAn = np.copy(
            self.l_GKn[0, 0] * self.l_GAn[0, 0] + self.l_GKn[1, 0] * self.l_GAn[1, 0] + self.l_GKn[2, 0] *
            self.l_GAn[2, 0] - self.l_GKn[0, 0])
        self.l_Gpxm[0, 0] = 1.
        if np.linalg.norm(self.l_GKn) > 0:
            self.l_Gpxm[1, 0] = gKndotgAn / (self.l_GKn[1, 0] + np.square(self.l_GKn[2, 0]) / self.l_GKn[1, 0])
            self.l_Gpxm[2, 0] = gKndotgAn / (self.l_GKn[2, 0] + np.square(self.l_GKn[1, 0]) / self.l_GKn[2, 0])
        # print self.l_Gpxm, self.l_GKn
        self.l_Gpxn = np.copy(self.l_Gpxm / np.linalg.norm(self.l_Gpxm))
        self.l_Gpx = np.copy(self.l_Gpxn) * lGA_mag
        if np.linalg.norm(self.l_GK) > 0:
            self.l_GA_pp = - self.l_GK * (
                self.l_GK[0, 0] * self.l_GA[0, 0] + self.l_GK[1, 0] * self.l_GA[1, 0] + self.l_GK[2, 0] * self.l_GA[
                    2, 0]) / (np.linalg.norm(self.l_GK) * np.linalg.norm(self.l_GK)) + self.l_GA
        self.l_GA_pp_mag = np.linalg.norm(self.l_GA_pp)
        if np.linalg.norm(self.l_GK) > 0:
            self.l_Gpx_pp = - self.l_GK * (np.dot(self.l_GK.T, self.l_Gpx)) / (
                np.linalg.norm(self.l_GK) * np.linalg.norm(self.l_GK)) + self.l_Gpx
        self.l_Gpx_pp_mag = np.linalg.norm(self.l_Gpx_pp)
        angles[1] = np.degrees(np.arccos(np.dot(self.l_GA_pp.T, self.l_Gpx_pp) / (
            self.l_GA_pp_mag * self.l_Gpx_pp_mag)))  # np.degrees(np.arctan2(np.cross(self.l_Gpx_pp.T,self.l_GA_pp.T)[0],np.dot(self.l_GA_pp.T,self.l_Gpx_pp)[0]))#
        angles[1] = (180 - angles[1])
        if np.cross(self.l_GA.T, self.l_GK.T)[0][0] < 0:
            angles[1] = -angles[1]

        #print lengths, 'LEP'

        return lengths, angles, pseudotargets


    def create_raw_database(self):
        '''Creates a database using the raw pressure values(full_body) and only
        transforms world frame coordinates to mat coordinates'''

        std_lengths = []
        #for subject in [4,9,10,11,12,13,14,15,16,17,18]:
        #for subject in [12]:
        for subject in [2, 3, 4, 5, 6, 7, 8]:

            self.final_dataset = {}
            self.final_dataset['images'] = []
            self.final_dataset['markers_xyz_m'] = []
            self.final_dataset['pseudomarkers_xyz_m'] = []
            self.final_dataset['marker_bed_euclideans_m'] = []
            self.final_dataset['bed_angle_deg'] = []
            self.final_dataset['joint_lengths_U_m'] = []
            self.final_dataset['joint_angles_U_deg'] = []
            self.final_dataset['joint_lengths_L_m'] = []
            self.final_dataset['joint_angles_L_deg'] = []



            for movement in ['RH_sitting','LH_sitting','RL_sitting','LL_sitting','RH1','LH1','RH2','RH3','LH2','LH3','RL','LL']:
            #for movement in ['RH_sitting', 'RH1']:
                std_lengths_i = []
            #for movement in ['RH_sitting','RH1']:#'LH_sitting','RL_sitting','LL_sitting']:
            #self.training_dump_path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_'+str(subject)
            #print self.training_dump_path

                p_file = load_pickle(self.training_dump_path+str(subject)+'/p_files/'+movement+'.p')
                count = 0

                indexlist = self.rand_index_p_length(p_file)


                if movement == 'head':
                    num_samp = 100
                elif movement == 'RH1' or movement == 'LH1' or movement == 'RL' or movement == 'LL':
                    num_samp = 150
                elif movement == 'RH_sitting' or movement == 'LH_sitting' :
                    num_samp = 120
                elif movement == 'RL_sitting' or movement == 'LL_sitting':
                    num_samp = 120
                else:
                    num_samp = 100

                print 'working on subject: ',subject, '  movement type:', movement, '  length: ',len(p_file), '  Number sampled: ',num_samp


                self.index_queue = []

                for i in np.arange(num_samp):


                #for [p_map_raw, target_raw, _] in p_file:
                    #print len(LH_sup) #100+, this is a list
                    #print len(LH_sup[4]) #2, this is a list
                    #print len(LH_sup[4][0]) #1728 #this is a list

                    #print len(p_file[i])
                    #print LH_sup[4][1].shape  #this is an array
                    #break

                    #this try/except block trys to keep popping things out of the first index, unless it runs out.
                    # if it runs out more than once, you've probably set your num samp too high


                    try:
                        index = indexlist.pop()
                    except:
                        print 'resetting index list'
                        indexlist = self.rand_index_p_length(p_file)
                        index = indexlist.pop()



                    #this little statement tries to filter the angle data. Some of the angle data is messed up, so we make a queue and take the mode.
                    if self.index_queue == []:
                        self.index_queue = np.zeros(5)
                        if p_file[index][2][0][0] > 350:
                            self.index_queue = self.index_queue + math.ceil(p_file[index][2][0][0])-360
                        else:
                            self.index_queue = self.index_queue + math.ceil(p_file[index][2][0][0])
                        angle = mode(self.index_queue)[0][0]
                    else:
                        self.index_queue[1:5] = self.index_queue[0:4]
                        if p_file[index][2][0][0] > 350:
                            self.index_queue[0] = math.ceil(p_file[index][2][0][0]) - 360
                        else:
                            self.index_queue[0] = math.ceil(p_file[index][2][0][0])
                        angle = mode(self.index_queue)[0][0]
                    if angle > 180: angle = angle - 360

                    p_map_raw, target_raw, _ = p_file[index]

                    p_map_raw = self.pad_pressure_mats(p_map_raw)

                    self.keep_image = False
                    target_mat = self.world_to_mat(target_raw, self.p_world_mat, self.R_world_mat)
                    rot_p_map = np.array(p_map_raw)



                    rot_target_mat = target_mat

                    torso = np.ones((4, 1))
                    torso[0:3, 0] = rot_target_mat[1, :]
                    # print torso


                    arm_joint_lengths, arm_joint_angles, arm_pseudotargets = self.inverse_arm_kinematics(rot_target_mat, subject, angle)

                    #the following returns the targets in mm
                    arm_targets = KinematicsLib().forward_upper_kinematics(np.expand_dims(angle, axis=0), np.concatenate((torso[0:3,0], arm_joint_lengths), axis = 0), arm_joint_angles)
                    arm_targets = np.squeeze(arm_targets, axis = 0)
                    arm_targets = np.reshape(arm_targets, (6,3))


                    leg_joint_lengths, leg_joint_angles, leg_pseudotargets = self.inverse_leg_kinematics(rot_target_mat, subject, angle)

                    std_lengths_i.append(leg_joint_lengths[4])

                    # the following returns the targets in mm
                    leg_targets = KinematicsLib().forward_lower_kinematics(np.expand_dims(angle, axis=0), np.concatenate((torso[0:3, 0], leg_joint_lengths), axis=0), leg_joint_angles)
                    leg_targets = np.squeeze(leg_targets, axis=0)
                    leg_targets = np.reshape(leg_targets[3:], (4,3))




                    kin_targets = np.concatenate((arm_targets, leg_targets), axis = 0)
                    pseudotargets = np.concatenate((arm_pseudotargets, leg_pseudotargets), axis = 0)

                    #VisualizationLib().rviz_publish_input(rot_p_map, angle)
                    #VisualizationLib().rviz_publish_output(rot_target_mat, kin_targets / 1000, pseudotargets)

                    #get the distances from the bed. this will help us to do an a per instance loss and for final error evaluation,
                    #so we can throw out joint poses that are too far away.
                    bed_distances = KinematicsLib().get_bed_distance(rot_p_map, rot_target_mat, angle)




                    sleep(0.01)

                    if i < 0:
                        self.visualize_single_pressure_map(rot_p_map, rot_target_mat)
                        if self.keep_image == True:
                            self.final_dataset['images'].append(list(rot_p_map.flatten()))
                            self.final_dataset['markers_xyz_m'].append(rot_target_mat.flatten())
                            self.final_dataset['pseudomarkers_xyz_m'].append(pseudotargets.flatten())
                            self.final_dataset['marker_bed_euclideans_m'].append(bed_distances[0])
                            self.final_dataset['bed_angle_deg'].append(angle)
                            self.final_dataset['joint_lengths_U_m'].append(arm_joint_lengths)
                            self.final_dataset['joint_lengths_L_m'].append(leg_joint_lengths)
                            self.final_dataset['joint_angles_U_deg'].append(arm_joint_angles)
                            self.final_dataset['joint_angles_L_deg'].append(leg_joint_angles)

                    elif self.select == True:
                        self.visualize_single_pressure_map(rot_p_map, rot_target_mat)
                        if self.keep_image == True:
                            self.final_dataset['images'].append(list(rot_p_map.flatten()))
                            self.final_dataset['markers_xyz_m'].append(rot_target_mat.flatten())
                            self.final_dataset['pseudomarkers_xyz_m'].append(pseudotargets.flatten())
                            self.final_dataset['marker_bed_euclideans_m'].append(bed_distances[0])
                            self.final_dataset['bed_angle_deg'].append(angle)
                            self.final_dataset['joint_lengths_U_m'].append(arm_joint_lengths)
                            self.final_dataset['joint_lengths_L_m'].append(leg_joint_lengths)
                            self.final_dataset['joint_angles_U_deg'].append(arm_joint_angles)
                            self.final_dataset['joint_angles_L_deg'].append(leg_joint_angles)
                    else:
                        self.final_dataset['images'].append(list(rot_p_map.flatten()))
                        self.final_dataset['markers_xyz_m'].append(rot_target_mat.flatten())
                        self.final_dataset['pseudomarkers_xyz_m'].append(pseudotargets.flatten())
                        self.final_dataset['marker_bed_euclideans_m'].append(bed_distances[0])
                        self.final_dataset['bed_angle_deg'].append(angle)
                        self.final_dataset['joint_lengths_U_m'].append(arm_joint_lengths)
                        self.final_dataset['joint_lengths_L_m'].append(leg_joint_lengths)
                        self.final_dataset['joint_angles_U_deg'].append(arm_joint_angles)
                        self.final_dataset['joint_angles_L_deg'].append(leg_joint_angles)
                    count += 1

                std_lengths.append(np.std(np.array(std_lengths_i)))
                print std_lengths, 'std'

                print 'images shape: ',np.array(self.final_dataset['images']).shape
                print 'marker xyz array shape: ', np.array(self.final_dataset['markers_xyz_m']).shape
                print 'marker bed Euclideans shape: ', np.array(self.final_dataset['marker_bed_euclideans_m']).shape
                print 'bed angle in degrees shape: ', np.array(self.final_dataset['bed_angle_deg']).shape
                print 'joint lengths upper body shape: ',  np.array(self.final_dataset['joint_lengths_U_m']).shape
                print 'joint angles upper body shape: ', np.array(self.final_dataset['joint_angles_U_deg']).shape
                print 'joint lengths lower body shape: ',  np.array(self.final_dataset['joint_lengths_L_m']).shape
                print 'joint angles lower body shape: ', np.array(self.final_dataset['joint_angles_L_deg']).shape

            print np.mean(np.array(std_lengths)), 'mean of standard devs'
            print 'Output file size: ~', int(len(self.final_dataset['images']) * 0.08958031837*3948/1728), 'Mb'
            print "Saving final_dataset"
            pkl.dump(self.final_dataset, open(os.path.join(self.training_dump_path+str(subject)+'/p_files/trainval_150rh1_lh1_rl_ll_100rh23_lh23_sit120rh_lh_rl_ll.p'), 'wb'))

            print 'Done.'
        return


    def run(self):
        '''Runs either the synthetic database creation script or the 
        raw dataset creation script to create a dataset'''
        self.create_raw_database()
        return





if __name__ == "__main__":

    import optparse
    p = optparse.OptionParser()

    p.add_option('--lab_hd', action='store_true',
                 dest='lab_harddrive', \
                 default=False, \
                 help='Set path to the training database on lab harddrive.')

    p.add_option('--select', action='store_true', dest='select',
                 default=False, help='Presents visualization of all images for user to select discard/keep.')

    p.add_option('--verbose', '--v',  action='store_true', dest='verbose',
                 default=False, help='Printout everything (under construction).')
    
    opt, args = p.parse_args()


    opt.trainingPath = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_' #this is for the lab hard drive
    #opt.trainingPath = '/home/henryclever/hrl_file_server/Autobed/pose_estimation_data/subject_' this is if you're on a lab computer

    
    #Initialize trainer with a training database file
    p = DatabaseCreator(training_database_pkl_directory=opt.trainingPath,verbose = opt.verbose, select = opt.select)
    p.run()
    sys.exit()
