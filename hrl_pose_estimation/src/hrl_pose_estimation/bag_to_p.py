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
       
        print "Ready to start reading bags."

    def read_bag(self, subject, filename):
        print 'Starting on subject ', subject, 'with the following trial: ', filename


        self.mat_sampled = False

        filepath = self.database_path+'/subject_'+str(subject)+'/subject'+str(subject)+filename
        #print filepath

        bag = rosbag.Bag(filepath, 'r')
        count = 0


        targets = np.zeros((10,3))
        bed_pos = np.zeros((1,3))
        p_mat = []

        mat_tar_pos = []
        single_mat_tar_pos = []

        #don't forget to clear out  the caches of all the labels when you log
        for topic, msg, t in bag.read_messages():
            if topic == '/fsascan':
                self.mat_sampled = True
                p_mat = msg.data
                count += 1
            elif topic == '/abdout0':
                bed_pos[0,0] = msg.data[0]
                bed_pos[0,1] = msg.data[1]
                bed_pos[0,2] = msg.data[2]
            elif topic == '/head_o/pose':
                targets[0,0] = msg.transform.translation.x
                targets[0,1] = msg.transform.translation.y
                targets[0,2] = msg.transform.translation.z
            elif topic == '/l_ankle_o/pose':
                targets[9, 0] = msg.transform.translation.x
                targets[9, 1] = msg.transform.translation.y
                targets[9, 2] = msg.transform.translation.z
            elif topic == 'l_elbow_o/pose':
                targets[3, 0] = msg.transform.translation.x
                targets[3, 1] = msg.transform.translation.y
                targets[3, 2] = msg.transform.translation.z
            elif topic == '/l_hand_o/pose':
                targets[5, 0] = msg.transform.translation.x
                targets[5, 1] = msg.transform.translation.y
                targets[5, 2] = msg.transform.translation.z
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
            elif topic == '/r_hand_o/pose':
                targets[4, 0] = msg.transform.translation.x
                targets[4, 1] = msg.transform.translation.y
                targets[4, 2] = msg.transform.translation.z
            elif topic == '/r_knee_o/pose':
                targets[6, 0] = msg.transform.translation.x
                targets[6, 1] = msg.transform.translation.y
                targets[6, 2] = msg.transform.translation.z
            elif topic == '/torso_o/pose':
                targets[1,0] = msg.transform.translation.x
                targets[1,1] = msg.transform.translation.y
                targets[1,2] = msg.transform.translation.z
            if self.mat_sampled == True:
                if np.count_nonzero(targets) == 30 and len(p_mat) == 1728:
                    #print 'pressure mat has been scanned'
                    #print targets
                    #print np.count_nonzero(targets)
                    single_mat_tar_pos = []
                    single_mat_tar_pos.append(p_mat)
                    single_mat_tar_pos.append(targets)
                    single_mat_tar_pos.append(bed_pos)
                    mat_tar_pos.append(single_mat_tar_pos)


                    self.mat_sampled = False
                    p_mat = []
                    targets = np.zeros((10,3))
                    bed_pos = np.zeros((1,3))
                    #print targets
                    #print mat_tar_pos[len(mat_tar_pos)-1][2], 'accelerometer reading', len(mat_tar_pos)
               
        bag.close()


        print count, len(mat_tar_pos), len(mat_tar_pos[0]), 'count, count, number of datatypes (should be 3)'
        return mat_tar_pos

    


if __name__ == '__main__':
    rospy.init_node('bag_to_pickle')
    bagtopkl = BagfileToPickle()
    filename = '_full_trial_RH1.bag'



    
    file_details_dict = {}
    #print file_details
    for subject in [1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,18]:
        file_details = []
        x = []
        x.append(subject)
        x.append('_full_trial_head.bag')
        x.append('head')
        file_details.append(x)

        x = []
        x.append(subject)
        x.append('_full_trial_home.bag')
        x.append('home')
        file_details.append(x)

        x = []
        x.append(subject)
        x.append('_full_trial_LH1.bag')
        x.append('LH1')
        file_details.append(x)

        x = []
        x.append(subject)
        x.append('_full_trial_LH2.bag')
        x.append('LH2')
        file_details.append(x)

        x = []
        x.append(subject)
        x.append('_full_trial_LH3.bag')
        x.append('LH3')
        file_details.append(x)

        x = []
        x.append(subject)
        x.append('_full_trial_RH1.bag')
        x.append('RH1')
        file_details.append(x)

        x = []
        x.append(subject)
        x.append('_full_trial_RH2.bag')
        x.append('RH2')
        file_details.append(x)

        x = []
        x.append(subject)
        x.append('_full_trial_RH3.bag')
        x.append('RH3')
        file_details.append(x)

        x = []
        x.append(subject)
        x.append('_full_trial_LL.bag')
        x.append('LL')
        file_details.append(x)

        x = []
        x.append(subject)
        x.append('_full_trial_RL.bag')
        x.append('RL')
        file_details.append(x)

        file_details_dict[str(subject)] = file_details

    for subject in [8]:
        file_details = []
        x = []
        x.append(subject)
        x.append('_full_trial_head.bag')
        x.append('head')
        file_details.append(x)

        x = []
        x.append(subject)
        x.append('_full_trial_home.bag')
        x.append('home')
        file_details.append(x)

        x = []
        x.append(subject)
        x.append('_full_trial_LH1.bag')
        x.append('LH1')
        file_details.append(x)

        x = []
        x.append(subject)
        x.append('_full_trial_LH2.bag')
        x.append('LH2')
        file_details.append(x)

        x = []
        x.append(subject)
        x.append('_full_trial_LH3.bag')
        x.append('LH3')
        file_details.append(x)

        x = []
        x.append(subject)
        x.append('_full_trial_RH1.bag')
        x.append('RH1')
        file_details.append(x)

        x = []
        x.append(subject)
        x.append('_full_trial_RH2.bag')
        x.append('RH2')
        file_details.append(x)

        x = []
        x.append(subject)
        x.append('_full_trial_RH3.bag')
        x.append('RH3')
        file_details.append(x)

        x = []
        x.append(subject)
        x.append('_full_trial_LL1.bag')
        x.append('LL1')
        file_details.append(x)

        x = []
        x.append(subject)
        x.append('_full_trial_LL2.bag')
        x.append('LL2')
        file_details.append(x)

        x = []
        x.append(subject)
        x.append('_full_trial_RL1.bag')
        x.append('RL1')
        file_details.append(x)

        x = []
        x.append(subject)
        x.append('_full_trial_RL2.bag')
        x.append('RL2')
        file_details.append(x)

        file_details_dict[str(subject)] = file_details


    for subject in [17]:
        file_details = []
        x = []
        x.append(subject)
        x.append('_full_trial_head.bag')
        x.append('head')
        file_details.append(x)

        x = []
        x.append(subject)
        x.append('_full_trial_LH1.bag')
        x.append('LH1')
        file_details.append(x)

        x = []
        x.append(subject)
        x.append('_full_trial_LH2.bag')
        x.append('LH2')
        file_details.append(x)

        x = []
        x.append(subject)
        x.append('_full_trial_LH3.bag')
        x.append('LH3')
        file_details.append(x)

        x = []
        x.append(subject)
        x.append('_full_trial_RH1.bag')
        x.append('RH1')
        file_details.append(x)

        x = []
        x.append(subject)
        x.append('_full_trial_RH2.bag')
        x.append('RH2')
        file_details.append(x)

        x = []
        x.append(subject)
        x.append('_full_trial_RH3.bag')
        x.append('RH3')
        file_details.append(x)

        x = []
        x.append(subject)
        x.append('_full_trial_LL.bag')
        x.append('LL')
        file_details.append(x)

        x = []
        x.append(subject)
        x.append('_full_trial_RL.bag')
        x.append('RL')
        file_details.append(x)

        file_details_dict[str(subject)] = file_details


    #print file_details_dict['9']
    database_path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials'

    for subject in [1]:
    #for subject in [2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]:



        entry = file_details_dict[str(subject)] #the entry has all the filename information for a subject, such as RH1, RH2, etc
        subject_alldata = {}

        for detail in entry: #the detail is for a specific bag file such as RH1

            #print detail[2]

            subject_detaildata = bagtopkl.read_bag(detail[0], detail[1])
            #pkl.dump(database_path, open(database_path+detail[2],'.p', "wb"))
            pkl.dump(subject_detaildata,open(os.path.join(database_path,'subject_'+str(subject),'p_files',detail[2]+'.p'), 'wb'))


