#!/usr/bin/env python

#By Henry M. Clever


import rospy, roslib
import sys, os
import random, math
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pkl
import tf
from time import sleep
from scipy import ndimage
from scipy.stats import mode
from skimage.feature import blob_doh
from hrl_msgs.msg import FloatArrayBare
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Point
import rosbag
import copy


#PyTorch libraries
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable


# Pose Estimation Libraries
from create_dataset_lib import CreateDatasetLib
from synthetic_lib import SyntheticLib
from visualization_lib import VisualizationLib
from kinematics_lib import KinematicsLib
from cascade_lib import CascadeLib
from preprocessing_lib import PreprocessingLib

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
NUMOFOUTPUTDIMS = 3
NUMOFOUTPUTNODES = 10
HIGH_TAXEL_THRESH_X = (NUMOFTAXELS_X - 1) 
HIGH_TAXEL_THRESH_Y = (NUMOFTAXELS_Y - 1) 

class RealTimePose():
    '''Detects the head of a person sleeping on the autobed'''
    def __init__(self, database_path):
        self.mat_size = (NUMOFTAXELS_X, NUMOFTAXELS_Y)
        self.elapsed_time = []
        self.database_path = database_path
        #[self.p_world_mat, self.R_world_mat] = load_pickle('/home/henryclever/hrl_file_server/Autobed/pose_estimation_data/mat_axes15.p')

        self.mat_pose_sampled = False
        self.ok_to_read_pose = False
        self.subject = 9
        self.bedangle = 0.
        self.output_size = (NUMOFOUTPUTNODES, NUMOFOUTPUTDIMS)
        self.T = 25 #stochastic forward passes
        self.limbArray = None
        self.mat_size = (NUMOFTAXELS_X, NUMOFTAXELS_Y)
        self.index_queue = []

        rospy.Subscriber("/fsascan", FloatArrayBare, self.pressure_map_callback)
        rospy.Subscriber("/abdout0", FloatArrayBare, self.bed_config_callback)

        print self.database_path+'/subject_18/p_files/convnet_2to8_anglesvL_128b_200e_18.pt'
        try:
            self.kin_model = torch.load(self.database_path+'/subject_18/p_files/convnet_2to8_anglesvL_128b_200e_18.pt', map_location=lambda storage, loc: storage)
            print '###################################### MODEL SUCCESSFULLY LOADED ####################################'
        except:
            print '######################################## MODEL DOES NOT EXIST #######################################'
        self.count = 0  # When to sample the mat_origin



    def pressure_map_callback(self, data):
        '''This callback accepts incoming pressure map from
        the Vista Medical Pressure Mat and sends it out.'''

        p_mat_raw  = data.data
        self.p_mat = np.array(p_mat_raw)



        self.ok_to_read_pose = True

    def bed_config_callback(self, msg):
        '''This callback accepts incoming pressure map from
        the Vista Medical Pressure Mat and sends it out.
        Remember, this array needs to be binarized to be used'''
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

    def broadcast_tf_pose(self, msg, all_joints):
        br = tf.TransformBroadcaster()
        br.sendTransform((msg[0], msg[1], msg[2]), tf.transformations.quaternion_from_euler(0, 0, 0), rospy.Time.now(), all_joints, "autobed/base_link")

    def run(self):

        '''This code just collects the first 1200 samples of the
                pressure mat that will come in through the bagfile and
                will label them with the label'''

        while not rospy.is_shutdown():

            if self.ok_to_read_pose == True:
                self.count += 1

                input_stack = PreprocessingLib().preprocessing_create_pressure_angle_stack_realtime(self.p_mat, self.bedangle, self.mat_size)
                input_stack_up = PreprocessingLib().preprocessing_pressure_map_upsample(input_stack)
                input_tensor = Variable(torch.Tensor(input_stack_up),volatile=True, requires_grad = False)

                input_tensor = input_tensor.expand(self.T, 3, 128, 54)
                print input_tensor.size(),'input tensor'
                print self.bedangle, 'bedangle'

                _, targets_est, angles_est, lengths_est, pseudotargets_est = self.kin_model.forward_kinematic_jacobian(input_tensor, prior_cascade=None, forward_only=True, subject=self.subject)

                #targets_est = np.mean(error, axis=0) / 10

                viz_image = np.array(input_stack)[0,:,:,:]
                viz_image = np.pad(viz_image[0,:,:], 10, 'constant', constant_values=(0.))

                scores = np.reshape(np.mean(targets_est.numpy()/1000, axis = 0), self.output_size)
                scores_std = np.std(np.linalg.norm(np.reshape(targets_est.numpy()/1000, (self.T, 10, 3)), axis = 2), axis = 0)

                self.broadcast_tf_pose(scores[0,:], 'alljoints')


                pseudotarget_scores = np.reshape(np.mean(pseudotargets_est/1000, axis = 0), (5, 3))
                pseudotarget_scores_std = np.std(np.linalg.norm(np.reshape(pseudotargets_est/1000, (self.T, 5, 3)), axis = 2), axis = 0)

                #print scores
                print scores_std
                #print pseudotarget_scores
                #print pseudotarget_scores_std
                scores_all = np.concatenate((scores, pseudotarget_scores), axis = 0)


                if True: #publish in rviz
                    VisualizationLib().rviz_publish_input(viz_image * 1.3, self.bedangle)
                    VisualizationLib().rviz_publish_output(None, scores, pseudotarget_scores, scores_std, pseudotarget_scores_std)
                    self.limbArray = VisualizationLib().rviz_publish_output_limbs(None, np.reshape(scores, self.output_size), np.reshape(pseudotarget_scores, (5, 3)),
                                                                             LimbArray=self.limbArray, count=0)

                #VisualizationLib().visualize_pressure_map(viz_image, targets_raw=None, scores_raw=scores_all)


                self.ok_to_read_pose = False


if __name__ == '__main__':
    rospy.init_node('real_time_pose')

    #print file_details_dict['9']
    #database_path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials'
    database_path = '/home/henryclever/Autobed_OFFICIAL'

    getpose = RealTimePose(database_path)

    getpose.run()

