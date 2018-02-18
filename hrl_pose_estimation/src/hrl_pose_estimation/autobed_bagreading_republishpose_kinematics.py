#!/usr/bin/env python
import rospy, roslib
import sys, os
import random, math
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pkl
from scipy import ndimage
from scipy.optimize import fsolve
from skimage.feature import blob_doh
from hrl_msgs.msg import FloatArrayBare
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import PoseStamped
import rosbag
import copy
import tf.transformations as tft
import tf

# Pose Estimation Libraries
from create_dataset_lib import CreateDatasetLib

roslib.load_manifest('hrl_lib')
from hrl_lib.util import save_pickle, load_pickle

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

class HeadDetector:
    '''Detects the head of a person sleeping on the autobed'''
    def __init__(self):
        self.mat_size = (NUMOFTAXELS_X, NUMOFTAXELS_Y)
        self.elapsed_time = []
        # rospy.init_node('head_pose_estimator', anonymous=True)
        # rospy.Subscriber("/fsascan", FloatArrayBare,
        #         self.current_physical_pressure_map_callback)
        # rospy.Subscriber("/head_o/pose", TransformStamped,
        #         self.head_origin_callback)
        # self.database_path = '/home/yashc/Desktop/dataset/subject_4'
        self.database_path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials'
        #self.database_path = '/home/henryclever/hrl_file_server/Autobed'
        self.head_center_2d = None

        self.world_to_mat = CreateDatasetLib().world_to_mat
        [self.p_world_mat, self.R_world_mat] = load_pickle('/home/henryclever/hrl_file_server/Autobed/pose_estimation_data/mat_axes15.p')

        self.tf_broadcaster = tf.TransformBroadcaster()
        self.tf_listener = tf.TransformListener()

        self.T = np.zeros((4,1))
        self.H = np.zeros((4,1))

        self.r_S = np.zeros((4,1))
        self.r_E = np.zeros((4,1))
        self.r_H = np.zeros((4,1))
        self.Pr_SE = np.zeros((4,1))
        self.Pr_EH = np.zeros((4,1))
        self.r_elbow_msg = None
        self.r_hand_msg = None
        self.r_SE = np.zeros((3,1))
        self.r_SEn = np.zeros((3,1))
        self.r_SH = np.zeros((3, 1))
        self.r_SHn = np.zeros((3, 1))
        self.r_Spxn = np.zeros((3, 1))
        self.r_Spxm = np.zeros((3, 1))
        self.r_Spxzn = np.zeros((3,1))
        self.r_SPxz = np.zeros((3,1))
        self.r_Spx = np.zeros((3, 1))
        self.r_SH_pp = np.zeros((3, 1))
        self.r_Spx_pp = np.zeros((3, 1))

        self.pred_r_H = np.zeros((3, 1))
        self.r_ROT_OS = np.matrix('0.,0.,-1.;0.,1.,0.;-1.,0.,0.')
        self.ROT_bed = np.matrix('1.,0.,0.;0.,1.,0.;0.,0.,1.')


        self.l_S = np.zeros((4,1))
        self.l_E = np.zeros((4,1))
        self.l_H = np.zeros((4,1))
        self.Pl_SE = np.zeros((4,1))
        self.Pl_EH = np.zeros((4,1))
        self.l_elbow_msg = None
        self.l_hand_msg = None
        self.l_SE = np.zeros((3,1))
        self.l_SEn = np.zeros((3,1))
        self.l_SH = np.zeros((3, 1))
        self.l_SHn = np.zeros((3, 1))
        self.l_Spxn = np.zeros((3, 1))
        self.l_Spxm = np.zeros((3, 1))
        self.l_Spxzn = np.zeros((3,1))
        self.l_SPxz = np.zeros((3,1))
        self.l_Spx = np.zeros((3, 1))
        self.l_SH_pp = np.zeros((3, 1))
        self.l_Spx_pp = np.zeros((3, 1))

        self.l_ROT_OS = np.matrix('0.,0.,1.;0.,1.,0.;1.,0.,0.')

        self.r_K = np.zeros((4,1))
        self.r_A = np.zeros((4,1))

        self.l_K = np.zeros((4,1))
        self.l_A = np.zeros((4,1))

        self.pseudoheight = {'1': 1.53, '2': 1.42, '3': 1.52, '4': 1.63, '5': 1.66, '6': 1.59, '7': 1.49, '8': 1.53,
                         '9': 1.69, '10': 1.58, '11': 1.64, '12': 1.45, '13': 1.58, '14': 1.67, '15': 1.63, '16': 1.48,
                         '17': 1.43, '18': 1.54}

        self.bedangle = 0.

        self.zoom_factor = 2
        self.mat_sampled = False
        self.mat_pose = []
        self.head_pose = []
        self.zoom_factor = 2
        self.params_length = np.zeros((8))#torso height, torso vert, shoulder right, shoulder left, upper arm right, upper arm left, forearm right, forearm left
        self.params_angle = np.zeros((8))  # sh right roll, sh left roll, sh right pitch, sh left pitch, sh right yaw, sh left yaw, elbow right, elbow left


        print "Ready to start reading bags."

    def read_bag(self, subject, filename, method, visualize=False):
        print 'Starting on subject ', subject
        [self.p_world_mat, self.R_world_mat] = load_pickle(self.database_path+'/subject_15/mat_axes.p')
        # +'/subject_' + str(subject) +
        # pkl.load(open(os.path.join(self.database_path, '/subject_',str(subject),'/','mat_axes.p'), "r"))
        # self.p_world_mat = np.array([0, 0, 0])
        # self.R_world_mat = np.eye(3)
        # print self.p_world_mat
        # print self.R_world_mat
        head_center = [0, 0, 0]
        self.pos = 0
        self.total_count = 0
        count = 0
        self.error_array = []

        self.mat_sampled = False
        self.ground_truth_sampled = False


        head_pose = rospy.Publisher("/head_o/pose", PoseStamped, queue_size = 10)
        l_ankle_pose  = rospy.Publisher("/l_ankle_o/pose", PoseStamped, queue_size = 10)
        l_shoulder_pose = rospy.Publisher("/l_shoulder/pose", PoseStamped, queue_size = 10)
        l_elbow_pose = rospy.Publisher("/l_elbow_o/pose", PoseStamped, queue_size = 10)
        l_hand_pose = rospy.Publisher("/l_hand_o/pose", PoseStamped, queue_size = 10)
        l_knee_pose = rospy.Publisher("/l_knee_o/pose", PoseStamped, queue_size = 10)
        r_ankle_pose = rospy.Publisher("/r_ankle_o/pose", PoseStamped, queue_size = 10)
        r_shoulder_pose = rospy.Publisher("/r_shoulder/pose", PoseStamped, queue_size = 10)
        r_elbow_pose = rospy.Publisher("/r_elbow_o/pose", PoseStamped, queue_size = 10)
        r_hand_pose = rospy.Publisher("/r_hand_o/pose", PoseStamped, queue_size = 10)
        self.r_elbowpred_pose = rospy.Publisher("/r_elbowpred/pose", PoseStamped, queue_size = 10)
        self.l_elbowpred_pose = rospy.Publisher("/l_elbowpred/pose", PoseStamped, queue_size = 10)
        self.r_handpred_pose = rospy.Publisher("/r_handpred/pose", PoseStamped, queue_size = 10)
        self.l_handpred_pose = rospy.Publisher("/l_handpred/pose", PoseStamped, queue_size = 10)
        r_knee_pose = rospy.Publisher("/r_knee_o/pose", PoseStamped, queue_size = 10)
        torso_pose = rospy.Publisher("/torso_o/pose", PoseStamped, queue_size = 10)
        torso_vert_pose = rospy.Publisher("/torso_vert/pose", PoseStamped, queue_size = 10)
        abdout = rospy.Publisher("/abdout0", FloatArrayBare, queue_size = 10)
        mat_o_pose = rospy.Publisher("/mat_o/pose", PoseStamped, queue_size = 10)
        mat_x_pose = rospy.Publisher("/mat_x/pose", PoseStamped, queue_size = 10)
        mat_y_pose = rospy.Publisher("/mat_y/pose", PoseStamped, queue_size = 10)

        self.mocap_frame = 'autobed/base_link'

        bag = rosbag.Bag(self.database_path+'/subject_'+str(subject)+'/subject'+str(subject)+filename, 'r')
        latest_scan_time = None
        self.latest_ground_truth_time = None
        counter = 0
        rando = 0





        self.params_length[0] = 0.1 #torso height
        self.params_length[1] = 0.2065*self.pseudoheight[str(subject)] - 0.0529 #about 0.25. torso vert
        self.params_length[2] = 0.13454*self.pseudoheight[str(subject)] - 0.03547 #about 0.15. shoulder right
        self.params_length[3] = 0.13454*self.pseudoheight[str(subject)] - 0.03547 #about 0.15. shoulder left


        for topic, msg, t in bag.read_messages():

            if topic == '/fsascan':
                self.mat_sampled = True
                latest_scan_time = t
                self.current_physical_pressure_map_callback(msg)
                counter += 1
                print t, 'count: ',counter, rando
            elif topic == '/abdout0':
                self.publish_floatarr(msg,abdout)
                self.bedangle = np.round(msg.data[0],0)
                if self.bedangle > 180: self.bedangle = self.bedangle - 360
            elif topic == '/head_o/pose':
                w2m = self.world_to_mat(np.expand_dims(
                    np.array([msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z]),
                    0), self.p_world_mat, self.R_world_mat)
                self.H[0, 0], self.H[1, 0], self.H[2, 0], self.H[3, 0] = w2m[0], w2m[1], w2m[2], 1.
                msg.transform.translation.x = self.H[0, 0]
                msg.transform.translation.y = self.H[1, 0]
                msg.transform.translation.z = self.H[2, 0]
                self.publish_pose(msg, head_pose)
                self.head_msg = msg
            elif topic == '/torso_o/pose':
                w2m = self.world_to_mat(np.expand_dims(
                    np.array([msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z]),
                    0), self.p_world_mat, self.R_world_mat)
                self.T[0, 0], self.T[1, 0], self.T[2, 0], self.T[3, 0] = w2m[0], w2m[1], w2m[2], 1.
                msg.transform.translation.x = self.T[0, 0]
                msg.transform.translation.y = self.T[1, 0]
                msg.transform.translation.z = self.T[2, 0]
                self.publish_pose(msg,torso_pose)
                self.torso_msg = msg
            elif topic == '/r_elbow_o/pose':
                w2m = self.world_to_mat(np.expand_dims(
                    np.array([msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z]),
                    0), self.p_world_mat, self.R_world_mat)
                self.r_E[0, 0], self.r_E[1, 0], self.r_E[2, 0], self.r_E[3, 0] = w2m[0], w2m[1], w2m[2], 1.
                msg.transform.translation.x = self.r_E[0, 0]
                msg.transform.translation.y = self.r_E[1, 0]
                msg.transform.translation.z = self.r_E[2, 0]
                self.publish_pose(msg,r_elbow_pose)
                self.r_elbow_msg = msg
            elif topic == 'l_elbow_o/pose':
                w2m = self.world_to_mat(np.expand_dims(
                    np.array([msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z]),
                    0), self.p_world_mat, self.R_world_mat)
                self.l_E[0, 0], self.l_E[1, 0], self.l_E[2, 0], self.l_E[3, 0] = w2m[0], w2m[1], w2m[2], 1.
                msg.transform.translation.x = self.l_E[0, 0]
                msg.transform.translation.y = self.l_E[1, 0]
                msg.transform.translation.z = self.l_E[2, 0]
                self.publish_pose(msg,l_elbow_pose)
                self.l_elbow_msg = msg
            elif topic == '/r_hand_o/pose':
                w2m = self.world_to_mat(np.expand_dims(
                    np.array([msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z]),
                    0), self.p_world_mat, self.R_world_mat)
                self.r_H[0, 0], self.r_H[1, 0], self.r_H[2, 0], self.r_H[3, 0] = w2m[0], w2m[1], w2m[2], 1.
                msg.transform.translation.x = self.r_H[0, 0]
                msg.transform.translation.y = self.r_H[1, 0]
                msg.transform.translation.z = self.r_H[2, 0]
                self.publish_pose(msg,r_hand_pose)
                self.r_hand_msg = msg
            elif topic == '/l_hand_o/pose':
                w2m = self.world_to_mat(np.expand_dims(
                    np.array([msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z]),
                    0), self.p_world_mat, self.R_world_mat)
                self.l_H[0, 0], self.l_H[1, 0], self.l_H[2, 0], self.l_H[3, 0] = w2m[0], w2m[1], w2m[2], 1.
                msg.transform.translation.x = self.l_H[0, 0]
                msg.transform.translation.y = self.l_H[1, 0]
                msg.transform.translation.z = self.l_H[2, 0]
                self.publish_pose(msg,l_hand_pose)
                self.l_hand_msg = msg
            elif topic == '/r_knee_o/pose':
                w2m = self.world_to_mat(np.expand_dims(
                    np.array([msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z]),
                    0), self.p_world_mat, self.R_world_mat)
                self.r_K[0, 0], self.r_K[1, 0], self.r_K[2, 0], self.r_K[3, 0] = w2m[0], w2m[1], w2m[2], 1.
                msg.transform.translation.x = self.r_K[0, 0]
                msg.transform.translation.y = self.r_K[1, 0]
                msg.transform.translation.z = self.r_K[2, 0]
                self.publish_pose(msg,r_knee_pose)
                self.r_knee_msg = msg
            elif topic == '/l_knee_o/pose':
                w2m = self.world_to_mat(np.expand_dims(
                    np.array([msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z]),
                    0), self.p_world_mat, self.R_world_mat)
                self.l_K[0, 0], self.l_K[1, 0], self.l_K[2, 0], self.l_K[3, 0] = w2m[0], w2m[1], w2m[2], 1.
                msg.transform.translation.x = self.l_K[0, 0]
                msg.transform.translation.y = self.l_K[1, 0]
                msg.transform.translation.z = self.l_K[2, 0]
                self.publish_pose(msg,l_knee_pose)
                self.l_knee_msg = msg
            elif topic == '/r_ankle_o/pose':
                w2m = self.world_to_mat(np.expand_dims(
                    np.array([msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z]),
                    0), self.p_world_mat, self.R_world_mat)
                self.r_A[0, 0], self.r_A[1, 0], self.r_A[2, 0], self.r_A[3, 0] = w2m[0], w2m[1], w2m[2], 1.
                msg.transform.translation.x = self.r_A[0, 0]
                msg.transform.translation.y = self.r_A[1, 0]
                msg.transform.translation.z = self.r_A[2, 0]
                self.publish_pose(msg,r_ankle_pose)
                self.r_ankle_msg = msg
            elif topic == '/l_ankle_o/pose':
                w2m = self.world_to_mat(np.expand_dims(
                    np.array([msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z]),
                    0), self.p_world_mat, self.R_world_mat)
                self.l_A[0, 0], self.l_A[1, 0], self.l_A[2, 0], self.l_A[3, 0] = w2m[0], w2m[1], w2m[2], 1.
                msg.transform.translation.x = self.l_A[0, 0]
                msg.transform.translation.y = self.l_A[1, 0]
                msg.transform.translation.z = self.l_A[2, 0]
                self.publish_pose(msg, l_ankle_pose)
                self.l_ankle_msg = msg
            elif topic == '/mat_o/pose':
                self.publish_pose(msg,mat_o_pose)
            elif topic == '/mat_x/pose':
                self.publish_pose(msg,mat_x_pose)
            elif topic == '/mat_y/pose':
                self.publish_pose(msg,mat_y_pose)


            #self.pseudoheight = (self.r_A[1,0]+self.l_A[1,0])/2 - self.H[1,0]

            #here we construct pseudo ground truths for the shoulders by making fixed translations from the torso
            vert_torso = TransformStamped()
            vert_torso.transform.rotation.x = 0.
            vert_torso.transform.rotation.y = np.sin(np.deg2rad(self.bedangle*0.75))
            vert_torso.transform.rotation.z = -np.cos(np.deg2rad(self.bedangle*0.75))
            vert_torso.transform.rotation.w = 1.
            vert_torso.transform.translation.x = self.T[0,0]
            vert_torso.transform.translation.y = self.T[1,0] + self.params_length[1]*np.cos(np.deg2rad(self.bedangle*0.75))
            vert_torso.transform.translation.z = self.T[2,0] - self.params_length[0] + self.params_length[1]*np.sin(np.deg2rad(self.bedangle*0.75))
            self.publish_pose(vert_torso,torso_vert_pose)

            r_should_pose = TransformStamped()
            r_should_pose.transform.rotation.x = 1.
            r_should_pose.transform.rotation.y = 0.
            r_should_pose.transform.rotation.z = 0.
            r_should_pose.transform.rotation.w = -1.
            r_should_pose.transform.translation.x = self.T[0,0] - self.params_length[2]
            r_should_pose.transform.translation.y = self.T[1,0] + self.params_length[1]*np.cos(np.deg2rad(self.bedangle*0.75))
            r_should_pose.transform.translation.z = self.T[2,0] - self.params_length[0] + self.params_length[1]*np.sin(np.deg2rad(self.bedangle*0.75))
            self.publish_pose(r_should_pose, r_shoulder_pose)

            self.r_S[0,0] = r_should_pose.transform.translation.x
            self.r_S[1,0] = r_should_pose.transform.translation.y
            self.r_S[2,0] = r_should_pose.transform.translation.z
            self.r_S[3,0] = 1

            l_should_pose = TransformStamped()
            l_should_pose.transform.rotation.x = 1.
            l_should_pose.transform.rotation.y = 0.
            l_should_pose.transform.rotation.z = 0.
            l_should_pose.transform.rotation.w = -1.
            l_should_pose.transform.translation.x = self.T[0, 0] + self.params_length[2]
            l_should_pose.transform.translation.y = self.T[1, 0] + self.params_length[1]*np.cos(np.deg2rad(self.bedangle*0.75))
            l_should_pose.transform.translation.z = self.T[2, 0] - self.params_length[0] + self.params_length[1]*np.sin(np.deg2rad(self.bedangle*0.75))
            self.publish_pose(l_should_pose, l_shoulder_pose)

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


            #To find the angles we also need to rotate by the bed angle
            self.ROT_bed[1,1] = np.cos(np.deg2rad(-self.bedangle*0.75))
            self.ROT_bed[1,2] = -np.sin(np.deg2rad(-self.bedangle*0.75))
            self.ROT_bed[2,1] = np.sin(np.deg2rad(-self.bedangle*0.75))
            self.ROT_bed[2,2] = np.cos(np.deg2rad(-self.bedangle*0.75))


            #get the shoulder pitch
            rSE_mag = np.copy(self.params_length[4])
            self.r_SE = self.r_S[0:3] - self.r_E[0:3]
            self.r_SE = np.matmul(np.matmul(self.r_ROT_OS, self.ROT_bed), self.r_SE)
            if rSE_mag > 0: self.r_SEn = np.copy(self.r_SE)/rSE_mag
            self.params_angle[2] = -np.degrees(np.arcsin(self.r_SEn[1,0]))

            lSE_mag = np.copy(self.params_length[5])
            self.l_SE = self.l_S[0:3] - self.l_E[0:3]
            self.l_SE = np.matmul(np.matmul(self.l_ROT_OS, self.ROT_bed), self.l_SE)
            if lSE_mag > 0: self.l_SEn = np.copy(self.l_SE)/lSE_mag
            self.params_angle[3] = -np.degrees(np.arcsin(self.l_SEn[1,0]))

            #get shoulder yaw
            self.params_angle[4] = -np.degrees(np.arctan(self.r_SEn[0,0]/self.r_SEn[2,0]))
            self.params_angle[5] = np.degrees(np.arctan(self.l_SEn[0,0]/self.l_SEn[2,0]))

            # get the elbow angle
            rSH_mag = np.linalg.norm(self.r_H - self.r_S)
            # now apply law of cosines
            self.params_angle[6] = np.degrees(np.arccos((np.square(self.params_length[4]) + np.square(
                self.params_length[6]) - np.square(rSH_mag)) / (2 * self.params_length[4] * self.params_length[6])))
            lSH_mag = np.linalg.norm(self.l_H - self.l_S)
            # now apply law of cosines
            self.params_angle[7] = np.degrees(np.arccos((np.square(self.params_length[5]) + np.square(
                self.params_length[7]) - np.square(lSH_mag)) / (2 * self.params_length[5] * self.params_length[7])))


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
            self.params_angle[0] = np.degrees(np.arccos(np.dot(self.r_SH_pp.T, self.r_Spx_pp) / (
            self.r_SH_pp_mag * self.r_Spx_pp_mag)))  # np.degrees(np.arctan2(np.cross(self.r_Spx_pp.T,self.r_SH_pp.T)[0],np.dot(self.r_SH_pp.T,self.r_Spx_pp)[0]))#
            if np.cross(self.r_SH.T, self.r_SE.T)[0][0] < 0:
                self.params_angle[0] = -self.params_angle[0]


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
            self.params_angle[1] = np.degrees(np.arccos(np.dot(self.l_SH_pp.T, self.l_Spx_pp) / (
            self.l_SH_pp_mag * self.l_Spx_pp_mag)))  # np.degrees(np.arctan2(np.cross(self.l_Spx_pp.T,self.l_SH_pp.T)[0],np.dot(self.l_SH_pp.T,self.l_Spx_pp)[0]))#
            self.params_angle[1] = (180-self.params_angle[1])
            if np.cross(self.l_SH.T, self.l_SE.T)[0][0] < 0:
                self.params_angle[1] = -self.params_angle[1]





            #
            #
            # self.predRSH = self.r_S[0:3] - self.pred_r_H[0:3]
            # self.predRSH = np.matmul(self.r_ROT_OS, self.predRSH)
            # self.predRSHn = np.copy(self.predRSH) / rSH_mag
            # sEndotsHn = np.copy(self.r_SEn[0, 0] * self.predRSHn[0, 0] + self.r_SEn[1, 0] * self.predRSHn[1, 0] + self.r_SEn[2, 0] * self.predRSHn[2, 0] - self.r_SEn[0,0])
            # self.r_Spxm[0,0] = 1.
            # if np.linalg.norm(self.r_SEn) > 0:
            #     self.r_Spxm[1,0] = sEndotsHn / (self.r_SEn[1,0]+np.square(self.r_SEn[2,0])/self.r_SEn[1,0])
            #     self.r_Spxm[2,0] = sEndotsHn / (self.r_SEn[2,0]+np.square(self.r_SEn[1,0])/self.r_SEn[2,0])
            #
            # #print self.r_Spxm, self.r_SEn
            # self.r_Spxn = np.copy(self.r_Spxm/np.linalg.norm(self.r_Spxm))
            #
            #
            # # #
            # self.r_Spx = np.copy(self.r_Spxn) * rSH_mag
            # if np.linalg.norm(self.r_SE) > 0:
            #     self.r_SH_pp = - self.r_SE * (self.r_SE[0, 0] * self.predRSH[0, 0] + self.r_SE[1, 0] * self.predRSH[1, 0] + self.r_SE[2, 0] * self.predRSH[2, 0]) / (np.linalg.norm(self.r_SE) * np.linalg.norm(self.r_SE)) + self.predRSH
            # self.predRSH_pp_mag = np.linalg.norm(self.r_SH_pp)
            # if np.linalg.norm(self.r_SE) > 0:
            #     self.r_Spx_pp = - self.r_SE * (np.dot(self.r_SE.T, self.r_Spx)) / (np.linalg.norm(self.r_SE) * np.linalg.norm(self.r_SE)) + self.r_Spx
            # self.r_Spx_pp_mag = np.linalg.norm(self.r_Spx_pp)
            # self.params_angle[1] = np.degrees(np.arccos(np.dot(self.r_SH_pp.T, self.r_Spx_pp) / (self.predRSH_pp_mag * self.r_Spx_pp_mag)))  # np.degrees(np.arctan2(np.cross(self.r_Spx_pp.T,self.r_SH_pp.T)[0],np.dot(self.r_SH_pp.T,self.r_Spx_pp)[0]))#
            # if np.cross(self.predRSH.T, self.r_SE.T)[0][0] < 0:
            #     self.params_angle[1]=-self.params_angle[1]
            #
            # #self.params_angle[1] = ((self.params_angle[1] + 35)*-1)-35



            self.forward_kinematics(self.params_length,self.params_angle,self.T, self.bedangle)


            if self.mat_sampled and self.ground_truth_sampled and np.abs(latest_scan_time.to_sec() - self.latest_ground_truth_time.to_sec())<0.1:
                print '#############################################################################'
                #print self.params_angle[0], self.params_angle[1], self.params_angle[2], self.params_angle[4], self.params_angle[6], self.params_angle[0]-self.params_angle[1], self.params_angle[0]-self.params_angle[1]-self.params_angle[4]


                #print self.T

                if count == 0:
                    start_time_range = t
                    start_time_range = self.latest_ground_truth_time
                if count < 50:
                    count += 1
                    start_time = rospy.Time.now()
                    # self.count += 1
                    # print "Iteration:{}".format(self.count)
                    if method == 'center_of_mass':
                        headx, heady = self.detect_head()
                        head_center = np.array([headx, heady, 1.])
                        #print headx, heady
                    elif method == 'blob':
                        blobs = self.detect_blob()
                        if blobs.any():
                            head_center = blobs[0, :]
                    else:
                        print 'I dont know what method to use'
                        return None
                    taxels_to_meters_coeff = np.array([MAT_HEIGHT/(NUMOFTAXELS_X*self.zoom_factor),
                                                -MAT_WIDTH/(NUMOFTAXELS_Y*self.zoom_factor),
                                                1])
                    #
                    taxels_to_meters_offset = np.array([MAT_HEIGHT, 0.0, 0.0])
                    y, x, r = (taxels_to_meters_offset - taxels_to_meters_coeff*head_center)
                    r = 5.
                    self.elapsed_time.append(rospy.Time.now() - start_time)
                    # print 'Estimated x, y'
                    # print x, ', ', y
                    # print "X:{}, Y:{}".format(x,y)
                    # print "Radius:{}".format(r)
                    ground_truth = np.array(self.get_ground_truth())
                    # print "Final Ground Truth:"
                    # print ground_truth
                    # self.visualize_pressure_map(self.pressure_map, rotated_targets=[headx, heady, 1],\
                    #                            plot_3d=False)
                    if visualize:
                        self.visualize_pressure_map(self.pressure_map, rotated_targets=[x, y, r], \
                                                    plot_3d=False)
                        rospy.sleep(1)
                    error = np.abs(x-ground_truth[0])
                    error = np.linalg.norm(np.array([x]) - np.array(ground_truth[0]))
                    # print 'Error:', error
                    self.error_array.append(error)
                    self.mat_sampled = False
                    self.ground_truth_sampled = False

                if count == 50:
                    time_range = t - start_time_range
                    time_range = self.latest_ground_truth_time - start_time_range

        bag.close()

        mean_err = np.mean(self.error_array)
        std_err = np.std(self.error_array)
        print 'For subject ', subject
        # print 'And file ', filename
        print "Average Error: {}".format(mean_err)
        print "Standard Deviation : {}".format(std_err)
        # print 'Count: ', count
        return #mean_err, std_err, count, time_range

    def publish_pose(self,msg,pose_publisher):
        self.ground_truth_sampled = True
        self.latest_ground_truth_time = msg.header.stamp
        self.head_origin_callback(msg)
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = self.mocap_frame
        pose.pose.position = msg.transform.translation
        pose.pose.orientation = msg.transform.rotation
        pose_publisher.publish(pose)


    def forward_kinematics(self,lengths, angles, torso, bedangle):
        #print angles, 'angles' #self.orig.T,  self.r_Spxn.T
        #print lengths,
        print angles[2:6], bedangle, self.r_E, self.l_E #self.pseudoheight, self.r_E

        #print '_________', np.cross(self.orig.T,self.r_SE.T),  self.r_SEn[2, 0] * self.orig[1,0] - self.r_SEn[1, 0] * self.orig[2,0]
        #if np.abs(self.r_SEn[2, 0] * self.orig[1,0] - self.r_SEn[1, 0] * self.orig[2,0]) > 0.001: sys.exit('Roll solver failed. enter different initial conditions.')

        TrelO = tft.identity_matrix()
        TprelT = tft.identity_matrix()


        rSrelTp = tft.rotation_matrix(np.deg2rad(bedangle*0.75), (1, 0, 0))
        lSrelTp = tft.rotation_matrix(np.deg2rad(bedangle*0.75), (1, 0, 0))


        TrelO[:, 3] = torso.T
        TprelT[2,3] = -lengths[0]
        rSrelTp[0,3] = -lengths[2]
        rSrelTp[1,3] = lengths[1] * np.cos(np.deg2rad(bedangle * 0.75))
        rSrelTp[2,3] = lengths[1] * np.sin(np.deg2rad(bedangle * 0.75))
        lSrelTp[0,3] = lengths[2]
        lSrelTp[1,3] = lengths[1] * np.cos(np.deg2rad(bedangle * 0.75))
        lSrelTp[2,3] = lengths[1] * np.sin(np.deg2rad(bedangle * 0.75))


        Pr_TS = np.matmul(np.matmul(TprelT, rSrelTp), np.array([[np.linalg.norm(self.r_S - self.T)], [0], [0], [1]]))

        #shoulder to elbow
        #rErelrS = np.matmul(tft.rotation_matrix(-np.deg2rad(angles[4]+180),(0,1,0)), tft.rotation_matrix(np.deg2rad(angles[2]), (0,0,1)))
        rErelrS = np.matmul(np.matmul(tft.rotation_matrix(-np.deg2rad(-angles[4] + 180), (0, 1, 0)),tft.rotation_matrix(np.deg2rad(180+angles[2]), (0, 0, 1))),tft.rotation_matrix(np.deg2rad((angles[0]) + 90 + angles[4]), (-1,0,0)))
        lErellS = np.matmul(np.matmul(tft.rotation_matrix(-np.deg2rad(angles[5] + 180), (0, 1, 0)),tft.rotation_matrix(np.deg2rad(-angles[3]), (0, 0, 1))),tft.rotation_matrix(np.deg2rad((-angles[1]) + 90 - angles[5]), (-1, 0, 0)))


        Pr_SE = np.matmul(rErelrS, np.array([[lengths[4]], [0], [0], [1]]))
        Pl_SE = np.matmul(lErellS, np.array([[lengths[5]], [0], [0], [1]]))

        rErelrS[0:3, 3] = -Pr_SE[0:3, 0]
        lErellS[0:3, 3] = -Pl_SE[0:3, 0]

        #rHrelrE = np.matmul(tft.rotation_matrix(np.deg2rad(-(angles[0])), (-1,0,0)),tft.rotation_matrix(np.deg2rad(angles[6]), (0, 0, 1)))
        rHrelrE = tft.rotation_matrix(np.deg2rad(angles[6]), (0, 0, 1))
        lHrellE = tft.rotation_matrix(np.deg2rad(angles[7]), (0, 0, 1))

        Pr_EH = np.matmul(rHrelrE, np.array([[lengths[6]], [0], [0], [1]]))
        Pl_EH = np.matmul(lHrellE, np.array([[lengths[7]], [0], [0], [1]]))

        Pr_SE = -Pr_SE
        Pr_SE[3, 0] = 1
        Pl_SE = -Pl_SE
        Pl_SE[3, 0] = 1

        self.pred_r_E = np.matrix(np.matmul(np.matmul(np.matmul(TrelO, TprelT), rSrelTp), Pr_SE))
        r_elbow_pose = TransformStamped()
        r_elbow_pose.transform.translation.x = self.pred_r_E[0, 0]
        r_elbow_pose.transform.translation.y = self.pred_r_E[1, 0]
        r_elbow_pose.transform.translation.z = self.pred_r_E[2, 0]
        self.publish_pose(r_elbow_pose, self.r_elbowpred_pose)

        self.pred_l_E = np.matrix(np.matmul(np.matmul(np.matmul(TrelO, TprelT), lSrelTp), Pl_SE))
        l_elbow_pose = TransformStamped()
        l_elbow_pose.transform.translation.x = self.pred_l_E[0, 0]
        l_elbow_pose.transform.translation.y = self.pred_l_E[1, 0]
        l_elbow_pose.transform.translation.z = self.pred_l_E[2, 0]
        self.publish_pose(l_elbow_pose, self.l_elbowpred_pose)


        self.pred_r_H = np.matrix(np.matmul(np.matmul(np.matmul(np.matmul(TrelO, TprelT), rSrelTp), rErelrS), Pr_EH))
        r_hand_pose = TransformStamped()
        r_hand_pose.transform.translation.x = self.pred_r_H[0, 0]
        r_hand_pose.transform.translation.y = self.pred_r_H[1, 0]
        r_hand_pose.transform.translation.z = self.pred_r_H[2, 0]
        self.publish_pose(r_hand_pose, self.r_handpred_pose)

        self.pred_l_H = np.matrix(np.matmul(np.matmul(np.matmul(np.matmul(TrelO, TprelT), lSrelTp), lErellS), Pl_EH))
        l_hand_pose = TransformStamped()
        l_hand_pose.transform.translation.x = self.pred_l_H[0, 0]
        l_hand_pose.transform.translation.y = self.pred_l_H[1, 0]
        l_hand_pose.transform.translation.z = self.pred_l_H[2, 0]
        self.publish_pose(l_hand_pose, self.l_handpred_pose)

        targetpred = np.zeros((4, 3))
        targetpred[0, :] = np.squeeze(self.pred_r_E[0:3, 0].T)
        targetpred[1, :] = np.squeeze(self.pred_l_E[0:3, 0].T)
        targetpred[2, :] = np.squeeze(self.pred_r_H[0:3, 0].T)
        targetpred[3, :] = np.squeeze(self.pred_l_H[0:3, 0].T)
        print np.round(targetpred,3), 'targetpred'



        targets = np.zeros((4, 3))
        targets[0, :] = np.squeeze(self.r_E[0:3, 0].T)
        targets[1, :] = np.squeeze(self.l_E[0:3, 0].T)
        targets[2, :] = np.squeeze(self.r_H[0:3, 0].T)
        targets[3, :] = np.squeeze(self.l_H[0:3, 0].T)
        print np.round(targets,3), 'targets'



        # The handler function for the turtle pose message broadcasts this turtle's translation and rotation, and
        # publishes it as a transform from frame "shoulder" to frame "elbow".
        #self, tf_broadcaster.sendTransform((msg.x, msg.y, msg.z),
        #                                   [0,0,0,1], #rotation in quaternions
        #                                   # tft.quaternion_from_euler(0, 0, msg.theta),
        #                                   rospy.Time.now(),
        #                                   "elbow",
        #                                   "shoulder")


    def publish_floatarr(self,msg,floatarr_publisher):
        floatarr_publisher.publish(msg)
    

    def get_elapsed_time(self):
        return self.elapsed_time


    def mat_to_taxels(self, m_data):
        ''' 
        Input:  Nx2 array 
        Output: Nx2 array
        '''       
        self.zoom_factor = 2
        #Convert coordinates in 3D space in the mat frame into taxels
        meters = m_data[0] 
        meters_to_taxels = np.array([(NUMOFTAXELS_Y*self.zoom_factor)/MAT_WIDTH, 
                                     (NUMOFTAXELS_X*self.zoom_factor)/MAT_HEIGHT,
                                     1])
        '''Typecast into int, so that we can highlight the right taxel 
        in the pressure matrix, and threshold the resulting values'''
        taxel = np.rint(meters_to_taxels*meters)
        #Shift origin of mat frame to top of mat, and threshold the negative taxel values
        taxel[1] = self.zoom_factor*NUMOFTAXELS_X - taxel[1]
        taxel = taxel[:2]
        taxel[taxel < 0] = 0.0
        return taxel

    def mat_origin_callback(self, data):
        '''This callback will sample data until its asked to stop'''
        if not self.mat_pose_sampled:
            self.mat_pose = [data.transform.translation.x,
                             data.transform.translation.y,
                             data.transform.translation.z]

    def head_origin_callback(self, data):
        '''This callback will sample data until its asked to stop'''
        self.head_pose = [data.transform.translation.x,
                         data.transform.translation.y,
                         data.transform.translation.z]

    def current_physical_pressure_map_callback(self, data):
        '''This callback accepts incoming pressure map from 
        the Vista Medical Pressure Mat and sends it out.'''
        # p_array = data.data
        # p_map_raw = np.reshape(p_array, self.mat_size)
        # p_map_hres=ndimage.zoom(p_map_raw, 2, order=1)
        # self.pressure_map=p_map_hres
        # self.mat_sampled = True
        if list(data.data):
            p_array = copy.copy(data.data)
            p_map_raw = np.reshape(p_array, self.mat_size)
            p_map_hres = ndimage.zoom(p_map_raw, self.zoom_factor, order=1)
            self.pressure_map = p_map_hres
            # self.mat_sampled = True
            # self.head_rest_B_head = self.detect_head()
            if not self.mat_sampled:
                self.mat_sampled = True
        else:
            print 'SOMETHING HAS GONE WRONG. PRESSURE MAT IS DEAD!'




    def visualize_pressure_map(self, pressure_map_matrix, rotated_targets=None, fileNumber=0, plot_3d=False):
        '''Visualizing a plot of the pressure map'''        
        fig = plt.gcf()
        plt.ion()
        if plot_3d == False:            
            plt.imshow(pressure_map_matrix, interpolation='nearest', cmap=
                plt.cm.bwr, origin='upper', vmin=0, vmax=100)
        else:
            ax1= fig.add_subplot(121, projection='3d')
            ax2= fig.add_subplot(122, projection='3d')
   
            n,m = np.shape(pressure_map_matrix)
            X,Y = np.meshgrid(range(m), range(n))
            ax1.contourf(X,Y,pressure_map_matrix, zdir='z', offset=0.0, cmap=plt.cm.bwr)
            ax2.contourf(X,Y,pressure_map_matrix, zdir='z', offset=0.0, cmap=plt.cm.bwr)

        if rotated_targets is not None:
            rotated_target_coord = np.array(copy.copy(rotated_targets))
            rotated_target_coord[0] = rotated_target_coord[0]/0.74*54.
            rotated_target_coord[1] = 128 - rotated_target_coord[1] / 1.75 * 128.

            xlim = [0.0, 54.0]
            ylim = [128.0, 0.0]                     
            
            if plot_3d == False:
                plt.plot(rotated_target_coord[0], rotated_target_coord[1],\
                         'y*', ms=10)
                plt.xlim(xlim)
                plt.ylim(ylim)                         
                circle2 = plt.Circle((rotated_target_coord[0], rotated_target_coord[1]),\
                        rotated_target_coord[2],\
                        color='r', fill=False, linewidth=4)
                fig.gca().add_artist(circle2)
            else:
                ax1.plot(np.squeeze(rotated_target_coord[:,0]), \
                         np.squeeze(rotated_target_coord[:,1]),\
                         np.squeeze(rotated_targets[:,2]), 'y*', ms=10)
                ax1.set_xlim(xlim)
                ax1.set_ylim(ylim)
                ax1.view_init(20,-30)

                ax2.plot(np.squeeze(rotated_target_coord[:,0]), \
                         np.squeeze(rotated_target_coord[:,1]),\
                         np.squeeze(rotated_targets[:,2]), 'y*', ms=10)
                ax2.view_init(1,10)
                ax2.set_xlim(xlim)
                ax2.set_ylim(ylim)
                ax2.set_zlim([-0.1,0.4])
            plt.draw()
            plt.pause(0.05) 
            plt.clf()




if __name__ == '__main__':
    rospy.init_node('calc_mean_std_of_head_detector_node')
    head_blob = HeadDetector()
    subject_means = []
    subject_std = []
    subject_scan_count = 0
    time_ranges = []
    filename = '_full_trial_RH3.bag'
    #filename = '_full_trial_sitting_LH.bag'
    method = 'center_of_mass'  # options are: 'blob', 'center_of_mass'
    #for subject in [5,6,7,8]:

    for subject in [5]:
        #new_mean, new_std, new_count, new_time_range = \
        head_blob.read_bag(subject, filename, method, visualize=False)
        #subject_means.append(new_mean)
        #subject_std.append(new_std)
        #subject_scan_count += new_count
        #a_range = new_time_range.to_sec()
        #time_ranges.append(a_range)
    print 'Total error mean  over subjects is: ', np.mean(subject_means)
    print 'Total error standard deviation over subjects is: ', np.std(subject_means)
    print 'Total pressure mat scans examined: ', subject_scan_count
    print 'Mean time range: ', np.mean(time_ranges)
    print 'Standard deviation time range: ', np.std(time_ranges)
    print 'Used method:', method
    all_times = head_blob.get_elapsed_time()
    out_time = []
    for t in all_times:
        a_time = t.to_sec()
        out_time.append(a_time)
    # all_times = np.array(all_times)
    print 'Average time: ', np.mean(out_time)
    print 'Standard Deviation time: ', np.std(out_time)
