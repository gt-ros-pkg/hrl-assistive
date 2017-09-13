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


        bag = rosbag.Bag(self.database_path+'/subject_'+str(subject)+'/subject'+str(subject)+filename, 'r')
        count = 0

        for topic, msg, t in bag.read_messages():
            if topic == '/fsascan':
                self.mat_sampled = True
                print len(msg.data)
                print topic
                count += 1
            elif topic == '/abdout0':
                print topic
            elif topic == '/head_o/pose':
                print topic
            elif topic == '/l_ankle_o/pose':
                print topic
            elif topic == 'l_elbow_o/pose':
                print topic
            elif topic == '/l_hand_o/pose':
                print topic
            elif topic == '/l_knee_o/pose':
                print topic
            elif topic == '/r_ankle_o/pose':
                print topic
            elif topic == '/r_elbow_o/pose':
                print topic
            elif topic == '/r_hand_o/pose':
                print topic
            elif topic == '/r_knee_o/pose':
                print topic
            elif topic == '/torso_o/pose':
                print topic
            if self.mat_sampled == True:
                print 'pressure mat has been scanned'
                self.mat_sampled = False
               
        bag.close()

        print count
        return

    


if __name__ == '__main__':
    rospy.init_node('bag to pickle')
    bagtopkl = BagfileToPickle()
    filename = '_full_trial_RH1.bag'

    filename_list = []

    x = []
    x.append(4)
    x.append('_full_trial_head.bag')
    x.append(0)
    x.append(50)
    filename_list.append(x)

    x = []
    x.append(4)
    x.append('_full_trial_home.bag')
    x.append(0)
    x.append(50)
    filename_list.append(x)

    x = []
    x.append(4)
    x.append('_full_trial_LH1.bag')
    x.append(0)
    x.append(50)
    filename_list.append(x)

    x = []
    x.append(4)
    x.append('_full_trial_LH2.bag')
    x.append(0)
    x.append(50)
    filename_list.append(x)

    x = []
    x.append(4)
    x.append('_full_trial_LH3.bag')
    x.append(0)
    x.append(50)
    filename_list.append(x)

    x = []
    x.append(4)
    x.append('_full_trial_RH1.bag')
    x.append(0)
    x.append(50)
    filename_list.append(x)

    x = []
    x.append(4)
    x.append('_full_trial_RH2.bag')
    x.append(0)
    x.append(50)
    filename_list.append(x)

    x = []
    x.append(4)
    x.append('_full_trial_RH3.bag')
    x.append(0)
    x.append(50)
    filename_list.append(x)

    x = []
    x.append(4)
    x.append('_full_trial_LL.bag')
    x.append(0)
    x.append(50)
    filename_list.append(x)

    x = []
    x.append(4)
    x.append('_full_trial_RL.bag')
    x.append(0)
    x.append(50)
    filename_list.append(x)
    



    x = []
    x.append(9)
    x.append('_full_trial_head.bag')
    x.append(0)
    x.append(50)
    filename_list.append(x)

    x = []
    x.append(9)
    x.append('_full_trial_home.bag')
    x.append(0)
    x.append(50)
    filename_list.append(x)

    x = []
    x.append(9)
    x.append('_full_trial_LH1.bag')
    x.append(0)
    x.append(50)
    filename_list.append(x)

    x = []
    x.append(9)
    x.append('_full_trial_LH2.bag')
    x.append(0)
    x.append(50)
    filename_list.append(x)

    x = []
    x.append(9)
    x.append('_full_trial_LH3.bag')
    x.append(0)
    x.append(50)
    filename_list.append(x)

    x = []
    x.append(9)
    x.append('_full_trial_RH1.bag')
    x.append(0)
    x.append(50)
    filename_list.append(x)

    x = []
    x.append(9)
    x.append('_full_trial_RH2.bag')
    x.append(0)
    x.append(50)
    filename_list.append(x)

    x = []
    x.append(9)
    x.append('_full_trial_RH3.bag')
    x.append(0)
    x.append(50)
    filename_list.append(x)

    x = []
    x.append(9)
    x.append('_full_trial_LL.bag')
    x.append(0)
    x.append(50)
    filename_list.append(x)

    x = []
    x.append(9)
    x.append('_full_trial_RL.bag')
    x.append(0)
    x.append(50)
    filename_list.append(x)
    



    x = []
    x.append(10)
    x.append('_full_trial_head.bag')
    x.append(0)
    x.append(50)
    filename_list.append(x)

    x = []
    x.append(10)
    x.append('_full_trial_home.bag')
    x.append(0)
    x.append(50)
    filename_list.append(x)

    x = []
    x.append(10)
    x.append('_full_trial_LH1.bag')
    x.append(0)
    x.append(50)
    filename_list.append(x)

    x = []
    x.append(10)
    x.append('_full_trial_LH2.bag')
    x.append(0)
    x.append(50)
    filename_list.append(x)

    x = []
    x.append(10)
    x.append('_full_trial_LH3.bag')
    x.append(0)
    x.append(50)
    filename_list.append(x)

    x = []
    x.append(10)
    x.append('_full_trial_RH1.bag')
    x.append(0)
    x.append(50)
    filename_list.append(x)

    x = []
    x.append(10)
    x.append('_full_trial_RH2.bag')
    x.append(0)
    x.append(50)
    filename_list.append(x)

    x = []
    x.append(10)
    x.append('_full_trial_RH3.bag')
    x.append(0)
    x.append(50)
    filename_list.append(x)

    x = []
    x.append(10)
    x.append('_full_trial_LL.bag')
    x.append(0)
    x.append(50)
    filename_list.append(x)

    x = []
    x.append(10)
    x.append('_full_trial_RL.bag')
    x.append(0)
    x.append(50)
    filename_list.append(x)
    
 
    
    
    print filename_list
    #for subject in [4,9,10,11,12,13,14,15]:
    #for filename in filename_list:
    #    print filename
    #    for subject in [13]:
    #        bagtopkl.read_bag(subject, filename, start, stop)     
