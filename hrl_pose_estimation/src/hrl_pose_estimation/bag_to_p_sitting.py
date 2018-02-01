#!/usr/bin/env python

#By Henry M. Clever
#This code to make pickle files is a combination of head_detector_bagreading.py (written by Ari Kapusta) and bag_to_p.py (written by Yash Chitalia).  
#The original bag_to_p.py requires replaying the bag files at original speed, which is cumbersome. 
#This version speeds up the latter and makes a pickle file that is better annotated


import rospy, roslib
import sys, os
import random, math
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pkl
from scipy import ndimage
from skimage.feature import blob_doh
from hrl_msgs.msg import FloatArrayBare
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import PoseStamped
import rosbag
import copy

roslib.load_manifest('hrl_lib')
from hrl_lib.util import save_pickle, load_pickle

# Pose Estimation Libraries
from create_dataset_lib import CreateDatasetLib
 


MAT_WIDTH = 0.74#0.762 #metres
MAT_HEIGHT = 1.75 #1.854 #metres
MAT_HALF_WIDTH = MAT_WIDTH/2 
NUMOFTAXELS_X = 64#73 #taxels
NUMOFTAXELS_Y = 27#30 
INTER_SENSOR_DISTANCE = 0.0286#metres
LOW_TAXEL_THRESH_X = 0
LOW_TAXEL_THRESH_Y = 0
HIGH_TAXEL_THRESH_X = (NUMOFTAXELS_X - 1) 
HIGH_TAXEL_THRESH_Y = (NUMOFTAXELS_Y - 1) 

class BagfileToPickle():
    '''Detects the head of a person sleeping on the autobed'''
    def __init__(self):
        self.mat_size = (NUMOFTAXELS_X, NUMOFTAXELS_Y)
        self.elapsed_time = []
        self.database_path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials'
        self.head_center_2d = None
        self.zoom_factor = 2
        self.mat_sampled = False
        self.mat_pose = []
        self.head_pose = []
        self.zoom_factor = 2

        [self.p_world_mat, self.R_world_mat] = load_pickle(
            '/home/henryclever/hrl_file_server/Autobed/pose_estimation_data/mat_axes15.p')

        self.T = np.zeros((4,1))
        self.H = np.zeros((4,1))

        self.r_S = np.zeros((4,1))
        self.r_E = np.zeros((4,1))
        self.r_H = np.zeros((4,1))

        self.l_S = np.zeros((4,1))
        self.l_E = np.zeros((4,1))
        self.l_H = np.zeros((4,1))
        self.pseudoheight = {'1': 1.53, '2': 1.42, '3': 1.52, '4': 1.63, '5': 1.66, '6': 1.59, '7': 1.49, '8': 1.53,
                         '9': 1.69, '10': 1.58, '11': 1.64, '12': 1.45, '13': 1.58, '14': 1.67, '15': 1.63, '16': 1.48,
                         '17': 1.43, '18': 1.54}

        self.bedangle = 0.
        self.params_length = np.zeros((8))  # torso height, torso vert, shoulder right, shoulder left, upper arm right, upper arm left, forearm right, forearm left

       
        print "Ready to start reading bags."

    def read_bag(self, subject, filename):
        print 'Starting on subject ', subject, 'with the following trial: ', filename



        self.mat_sampled = False

        filepath = self.database_path + '/subject_' + str(subject) + '/subject' + str(subject) + filename
        # print filepath

        bag = rosbag.Bag(filepath, 'r')
        count = 0

        targets = np.zeros((10, 3))
        bed_pos = np.zeros((1, 3))
        p_mat = []


        if filename == '_full_trial_sitting_LL2.bag' or filename == '_full_trial_sitting_RL2.bag':
            pass
        else:
            self.mat_tar_pos = []
        single_mat_tar_pos = []



        self.params_length[0] = 0.1 #torso height
        self.params_length[1] = 0.2065*self.pseudoheight[str(subject)] - 0.0529 #about 0.25. torso vert
        self.params_length[2] = 0.13454*self.pseudoheight[str(subject)] - 0.03547 #about 0.15. shoulder right
        self.params_length[3] = 0.13454*self.pseudoheight[str(subject)] - 0.03547 #about 0.15. shoulder left



        # don't forget to clear out  the caches of all the labels when you log
        for topic, msg, t in bag.read_messages():
            if topic == '/fsascan':
                self.mat_sampled = True
                p_mat = msg.data
                count += 1
            elif topic == '/abdout0':
                bed_pos[0, 0] = msg.data[0]
                bed_pos[0, 1] = msg.data[1]
                bed_pos[0, 2] = msg.data[2]
                self.bedangle = np.round(msg.data[0], 0)
                if self.bedangle > 180: self.bedangle = self.bedangle - 360
            elif topic == '/head_o/pose':
                targets[0, 0] = msg.transform.translation.x
                targets[0, 1] = msg.transform.translation.y
                targets[0, 2] = msg.transform.translation.z
            elif topic == '/l_ankle_o/pose':
                targets[9, 0] = msg.transform.translation.x
                targets[9, 1] = msg.transform.translation.y
                targets[9, 2] = msg.transform.translation.z
            elif topic == 'l_elbow_o/pose':
                targets[3, 0] = msg.transform.translation.x
                targets[3, 1] = msg.transform.translation.y
                targets[3, 2] = msg.transform.translation.z
                self.l_elbow_msg = msg
                self.l_E[0, 0] = msg.transform.translation.x
                self.l_E[1, 0] = msg.transform.translation.y
                self.l_E[2, 0] = msg.transform.translation.z
                self.l_E[3, 0] = 1
            elif topic == '/l_hand_o/pose':
                targets[5, 0] = msg.transform.translation.x
                targets[5, 1] = msg.transform.translation.y
                targets[5, 2] = msg.transform.translation.z
                self.l_hand_msg = msg
                self.l_H[0, 0] = msg.transform.translation.x
                self.l_H[1, 0] = msg.transform.translation.y
                self.l_H[2, 0] = msg.transform.translation.z
                self.l_H[3, 0] = 1
            elif topic == '/l_knee_o/pose':
                targets[7, 0] = msg.transform.translation.x
                targets[7, 1] = msg.transform.translation.y
                targets[7, 2] = msg.transform.translation.z
            elif topic == '/r_ankle_o/pose':
                targets[8, 0] = msg.transform.translation.x
                targets[8, 1] = msg.transform.translation.y
                targets[8, 2] = msg.transform.translation.z
            elif topic == '/r_elbow_o/pose':
                targets[2, 0] = msg.transform.translation.x
                targets[2, 1] = msg.transform.translation.y
                targets[2, 2] = msg.transform.translation.z
                self.r_elbow_msg = msg
                self.r_E[0, 0] = msg.transform.translation.x
                self.r_E[1, 0] = msg.transform.translation.y
                self.r_E[2, 0] = msg.transform.translation.z
                self.r_E[3, 0] = 1
            elif topic == '/r_hand_o/pose':
                targets[4, 0] = msg.transform.translation.x
                targets[4, 1] = msg.transform.translation.y
                targets[4, 2] = msg.transform.translation.z
                self.r_hand_msg = msg
                self.r_H[0, 0] = msg.transform.translation.x
                self.r_H[1, 0] = msg.transform.translation.y
                self.r_H[2, 0] = msg.transform.translation.z
                self.r_H[3, 0] = 1
            elif topic == '/r_knee_o/pose':
                targets[6, 0] = msg.transform.translation.x
                targets[6, 1] = msg.transform.translation.y
                targets[6, 2] = msg.transform.translation.z
            elif topic == '/torso_o/pose':
                targets[1, 0] = msg.transform.translation.x
                targets[1, 1] = msg.transform.translation.y
                targets[1, 2] = msg.transform.translation.z
                self.T[0, 0] = msg.transform.translation.x
                self.T[1, 0] = msg.transform.translation.y
                self.T[2, 0] = msg.transform.translation.z
                self.T[3, 0] = 1.

            # self.pseudoheight = (self.r_A[1,0]+self.l_A[1,0])/2 - self.H[1,0]

            # here we construct pseudo ground truths for the shoulders by making fixed translations from the torso
            vert_torso = TransformStamped()
            vert_torso.transform.rotation.x = 0.
            vert_torso.transform.rotation.y = np.sin(np.deg2rad(self.bedangle * 0.75))
            vert_torso.transform.rotation.z = np.cos(np.deg2rad(self.bedangle * 0.75))
            vert_torso.transform.rotation.w = 1.
            vert_torso.transform.translation.x = self.T[0, 0]
            vert_torso.transform.translation.y = self.T[1, 0] - self.params_length[1] * np.cos(
                np.deg2rad(self.bedangle * 0.75))
            vert_torso.transform.translation.z = self.T[2, 0] - self.params_length[0] + self.params_length[
                                                                                            1] * np.sin(
                np.deg2rad(self.bedangle * 0.75))

            r_should_pose = TransformStamped()
            r_should_pose.transform.rotation.x = 1.
            r_should_pose.transform.rotation.y = 0.
            r_should_pose.transform.rotation.z = 0.
            r_should_pose.transform.rotation.w = 1.
            r_should_pose.transform.translation.x = self.T[0, 0] + self.params_length[2]
            r_should_pose.transform.translation.y = self.T[1, 0] - self.params_length[1] * np.cos(
                np.deg2rad(self.bedangle * 0.75))
            r_should_pose.transform.translation.z = self.T[2, 0] - self.params_length[0] + self.params_length[
                                                                                               1] * np.sin(
                np.deg2rad(self.bedangle * 0.75))

            self.r_S[0, 0] = r_should_pose.transform.translation.x
            self.r_S[1, 0] = r_should_pose.transform.translation.y
            self.r_S[2, 0] = r_should_pose.transform.translation.z
            self.r_S[3, 0] = 1

            l_should_pose = TransformStamped()
            l_should_pose.transform.rotation.x = 1.
            l_should_pose.transform.rotation.y = 0.
            l_should_pose.transform.rotation.z = 0.
            l_should_pose.transform.rotation.w = 1.
            l_should_pose.transform.translation.x = self.T[0, 0] - self.params_length[2]
            l_should_pose.transform.translation.y = self.T[1, 0] - self.params_length[1] * np.cos(
                np.deg2rad(self.bedangle * 0.75))
            l_should_pose.transform.translation.z = self.T[2, 0] - self.params_length[0] + self.params_length[
                                                                                               1] * np.sin(
                np.deg2rad(self.bedangle * 0.75))

            self.l_S[0, 0] = l_should_pose.transform.translation.x
            self.l_S[1, 0] = l_should_pose.transform.translation.y
            self.l_S[2, 0] = l_should_pose.transform.translation.z
            self.l_S[3, 0] = 1

            # get the length of the right shoulder to right elbow
            self.params_length[4] = np.linalg.norm(self.r_E - self.r_S)
            self.params_length[5] = np.linalg.norm(self.l_E - self.l_S)

            # parameter for the length between hand and elbow. Should be around 0.2 meters.
            self.params_length[6] = np.linalg.norm(self.r_H - self.r_E)
            self.params_length[7] = np.linalg.norm(self.l_H - self.l_E)

            if self.mat_sampled == True:
                # print self.params_length, 'length'
                if len(p_mat) == 1728:# and self.params_length[4] > 0.15 and self.params_length[4] < 0.5 and self.params_length[5] > 0.15 and self.params_length[5] < 0.5 and self.params_length[6] > 0.1 and self.params_length[6] < 0.35 and self.params_length[7] > 0.1 and self.params_length[7] < 0.35:
                    if np.count_nonzero(targets) == 30:#targets[0,1] != 0 and targets[1,1] != 0 and targets[2,1] != 0 and targets[3,1] != 0 and targets[5,1] != 0 and targets[6,1] > 1.2:#
                        # print 'pressure mat has been scanned'
                        #print targets
                        # print np.count_nonzero(targets)

                        if subject == 18 and filename == '_full_trial_sitting_RH.bag' or filename == '_full_trial_sitting_RL.bag': #add some fillers for right elbow
                            if targets[3, 0] == 0: #fill in for the left elbow
                                targets[3, 1] = 0.90076631
                                targets[3, 0] = -0.23236339
                                targets[3, 2] = 0.5566895
                        elif subject == 3 and filename == '_full_trial_sitting_LL.bag':
                            if targets[3, 0] == 0:  # fill in for the left elbow
                                targets[3, 1] =  0.91585761
                                targets[3, 0] = -0.28543478
                                targets[3, 2] = 0.56117147
                                print 'filled'
                        elif subject == 3:
                            if filename == '_full_trial_sitting_LL.bag' or filename == '_full_trial_sitting_LH.bag':
                                if targets[8,0] > 0.26:
                                    targets[8, 0] = 0.08311699
                                    targets[8, 1] = 1.8579005
                                    targets[8, 2] = 0.57703006
                                    print 'filled'


                        if subject == 3:# and filename == '_full_trial_LH3.bag': #fix flipped knees. also, sometimes the feet freak out in other ways
                            if CreateDatasetLib().world_to_mat(targets, self.p_world_mat, self.R_world_mat)[6, 0] > CreateDatasetLib().world_to_mat(targets, self.p_world_mat, self.R_world_mat)[7,0]:
                                queue = np.copy(targets[6, :])
                                targets[6, :] = np.copy(targets[7, :])
                                targets[7, :] = queue
                                print 'flipped knees'

                        if subject == 8:# and filename == '_full_trial_LH3.bag': #fix flipped knees. also, sometimes the feet freak out in other ways
                            if CreateDatasetLib().world_to_mat(targets, self.p_world_mat, self.R_world_mat)[4, 0] > CreateDatasetLib().world_to_mat(targets, self.p_world_mat, self.R_world_mat)[6,0]:
                                queue = np.copy(targets[4, :])
                                targets[4, :] = np.copy(targets[5, :])
                                targets[5, :] = queue
                                print 'flipped hands'




                        single_mat_tar_pos = []
                        single_mat_tar_pos.append(p_mat)
                        single_mat_tar_pos.append(targets)
                        single_mat_tar_pos.append(bed_pos)
                        self.mat_tar_pos.append(single_mat_tar_pos)

                        self.mat_sampled = False
                        p_mat = []
                        targets = np.zeros((10, 3))
                        bed_pos = np.zeros((1, 3))
                        # print mat_tar_pos[len(mat_tar_pos)-1][2], 'accelerometer reading', len(mat_tar_pos)
                        print subject, 'subject', count, len(self.mat_tar_pos), 'count discarded, count kept,'

        bag.close()

        print count, len(self.mat_tar_pos), len(self.mat_tar_pos[0]), 'count, count, number of datatypes (should be 3)'

        return self.mat_tar_pos

    


if __name__ == '__main__':
    rospy.init_node('bag_to_pickle')
    bagtopkl = BagfileToPickle()
    filename = '_full_trial_RH1.bag'



    
    file_details_dict = {}
    #print file_details
    for subject in [2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18]:
        file_details = []
        # x = []
        # x.append(subject)
        # x.append('_full_trial_sitting_head.bag')
        # x.append('head_sitting')
        # file_details.append(x)
        #
        # x = []
        # x.append(subject)
        # x.append('_full_trial_sitting_home.bag')
        # x.append('home_sitting')
        # file_details.append(x)

        x = []
        x.append(subject)
        x.append('_full_trial_sitting_LH.bag')
        x.append('LH_sitting')
        file_details.append(x)

        #x = []
        #x.append(subject)
        #x.append('_full_trial_sitting_RH.bag')
        #x.append('RH_sitting')
        #file_details.append(x)

        #x = []
        #x.append(subject)
        #x.append('_full_trial_sitting_LL.bag')
        #x.append('LL_sitting')
        #ile_details.append(x)

        #x = []
        #x.append(subject)
        #x.append('_full_trial_sitting_RL.bag')
        #x.append('RL_sitting')
        #file_details.append(x)

        file_details_dict[str(subject)] = file_details

    for subject in [8]:
        file_details = []
        # x = []
        # x.append(subject)
        # x.append('_full_trial_sitting_head.bag')
        # x.append('head_sitting')
        # file_details.append(x)
        #
        # x = []
        # x.append(subject)
        # x.append('_full_trial_sitting_home.bag')
        # x.append('home_sitting')
        # file_details.append(x)

        #x = []
        #x.append(subject)
        #x.append('_full_trial_sitting_LH.bag')
        #x.append('LH_sitting')
        #file_details.append(x)

        #x = []
        #x.append(subject)
        #.append('_full_trial_sitting_RH.bag')
        #x.append('RH_sitting')
        #file_details.append(x)
        #
        x = []
        x.append(subject)
        x.append('_full_trial_sitting_LL1.bag')
        x.append('LL1_sitting')
        file_details.append(x)

        x = []
        x.append(subject)
        x.append('_full_trial_sitting_LL2.bag')
        x.append('LL2_sitting')
        file_details.append(x)
        #
        #x = []
        #x.append(subject)
        #x.append('_full_trial_sitting_RL1.bag')
        #.append('RL1_sitting')
        #file_details.append(x)

        #x = []
        #x.append(subject)
        #x.append('_full_trial_sitting_RL2.bag')
       # x.append('RL2_sitting')
        #file_details.append(x)

        file_details_dict[str(subject)] = file_details



    #print file_details_dict['9']
    database_path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials'

    for subject in [8]:
    #for subject in [2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]:



        entry = file_details_dict[str(subject)] #the entry has all the filename information for a subject, such as RH1, RH2, etc
        subject_alldata = {}

        for detail in entry: #the detail is for a specific bag file such as RH1

            #print detail[2]

            subject_detaildata = bagtopkl.read_bag(detail[0], detail[1])
            #pkl.dump(database_path, open(database_path+detail[2],'.p', "wb"))
            pkl.dump(subject_detaildata,open(os.path.join(database_path,'subject_'+str(subject),'p_files',detail[2]+'.p'), 'wb'))


