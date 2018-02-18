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
from time import sleep
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

class RealTimePose():
    '''Detects the head of a person sleeping on the autobed'''
    def __init__(self, database_path):
        self.mat_size = (NUMOFTAXELS_X, NUMOFTAXELS_Y)
        self.elapsed_time = []
        self.database_path = database_path
        [self.p_world_mat, self.R_world_mat] = load_pickle('/home/henryclever/hrl_file_server/Autobed/pose_estimation_data/mat_axes15.p')

        self.mat_pose_sampled = False
        self.ok_to_read_pose = False
        self.subject = 9
        self.filename = filename

        self.index_queue = []

        rospy.init_node('listener', anonymous=True)
        rospy.Subscriber("/fsascan", FloatArrayBare, self.pressure_map_callback)
        rospy.Subscriber("/abdout0", FloatArrayBare, self.bed_config_callback)

        try:
            self.kin_model = torch.load(self.database_path+'/subject_9/p_files/convnet_2to8_angles128b_200e_cL_18.pt')
        except:
            print "Model doesn't exist."
        self.count = 0  # When to sample the mat_origin



    def pressure_map_callback(self, data):
        '''This callback accepts incoming pressure map from
        the Vista Medical Pressure Mat and sends it out.'''

        p_mat_raw  = data.data
        self.p_mat = np.array(p_mat_raw)



        self.ok_to_read_pose = True

    def bed_config_callback(self, data):
        '''This callback accepts incoming pressure map from
        the Vista Medical Pressure Mat and sends it out.
        Remember, this array needs to be binarized to be used'''
        bed_pos[0, 0] = msg.data[0]
        bed_pos[0, 1] = msg.data[1]
        bed_pos[0, 2] = msg.data[2]
        bedangle = np.round(msg.data[0], 0)

        # this little statement tries to filter the angle data. Some of the angle data is messed up, so we make a queue and take the mode.
        if self.index_queue == []:
            self.index_queue = np.zeros(5)
            if bedangle > 350:
                self.index_queue = self.index_queue + math.ceil(bedangle) - 360
            else:
                self.index_queue = self.index_queue + math.ceil(bedangle)
            bedangle = mode(self.index_queue)[0][0]
        else:
            self.index_queue[1:5] = self.index_queue[0:4]
            if bedangle > 350:
                self.index_queue[0] = math.ceil(bedangle) - 360
            else:
                self.index_queue[0] = math.ceil(bedangle)
            bedangle = mode(self.index_queue)[0][0]

        if bedangle > 180: bedangle = bedangle - 360
        self.bedangle = bedangle


    def run(self):

        '''This code just collects the first 1200 samples of the
                pressure mat that will come in through the bagfile and
                will label them with the label'''
        while not rospy.is_shutdown():

            if self.ok_to_read_pose == True:
                self.count += 1

                input_stack = PreprocessingLib().preprocessing_create_pressure_angle_stack_realtime(self.p_mat, self.bedangle)
                input_stack = PreprocessingLib().preprocessing_pressure_map_upsample(input_stack)
                input_tensor = Variable(torch.Tensor(input_stack),volatile=True, requires_grad = False)

                _, targets_est, angles_est, lengths_est, pseudotargets_est = model_kin.forward_kinematic_jacobian(input_tensor, targets, constraints, prior_cascade=None, forward_only=True, subject=self.subject)

                print input_stack.shape
                print targets_est.shape


                self.ok_to_read_pose = False


if __name__ == '__main__':
    rospy.init_node('real_time_pose')

    #print file_details_dict['9']
    database_path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials'

    getpose = RealTimePose(database_path)

    getpose.run()

