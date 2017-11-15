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
        self.T = np.zeros((4,1))
        self.r_S = np.zeros((4,1))
        self.r_E = np.zeros((4,1))
        self.r_H = np.zeros((4,1))
        self.l_S = np.zeros((4,1))
        self.l_E = np.zeros((4,1))
        self.l_H = np.zeros((4,1))

        self.tf_broadcaster = tf.TransformBroadcaster()
        self.tf_listener = tf.TransformListener()

        self.TrelO = tft.identity_matrix()
        self.rSrelT = tft.identity_matrix()
        self.rSrotrS = tft.identity_matrix()
        self.rErelrS = tft.identity_matrix()
        self.rHrelrE = tft.identity_matrix()
        self.Pr_SE = np.zeros((4,1))
        self.Pr_EH = np.zeros((4,1))


        self.r_elbow_msg = None
        self.l_elbow_msg = None
        self.r_hand_msg = None
        self.l_hand_msg = None
        self.r_SE = np.zeros((3,1))
        self.r_SEn = np.zeros((3,1))
        self.r_SH = np.zeros((3, 1))
        self.r_SHn = np.zeros((3, 1))
        self.r_Spxn = np.zeros((3, 1))
        self.r_Spx = np.zeros((3, 1))
        self.r_SH_pp = np.zeros((3, 1))
        self.r_Spx_pp = np.zeros((3, 1))
        self.r_ROT_OS = np.matrix('0,0,-1;0,1,0;-1,0,0')
        self.rSrotrS[0:3,0:3] = self.r_ROT_OS#np.array([[1,0,0],[0,1,0],[0,0,1]])


        self.l_SE = np.zeros((3, 1))
        self.l_SEn = np.zeros((3, 1))
        self.l_SH = np.zeros((3, 1))
        self.l_SHn = np.zeros((3, 1))
        self.l_ROT_OS = np.matrix('0,0,1;0,1,0;1,0,0')

        self.zoom_factor = 2
        self.mat_sampled = False
        self.mat_pose = []
        self.head_pose = []
        self.zoom_factor = 2
        self.params_length = np.zeros((8))#torso height, torso vert, shoulder right, shoulder left, upper arm right, upper arm left, forearm right, forearm left
        self.params_length[0] = 0.1 #torso height
        self.params_length[1] = 0.3 #torso vert
        self.params_length[2] = 0.15 #shoulder right
        self.params_length[3] = 0.15 #shoulder left
        self.params_angle = np.zeros((8))  # sh right roll, sh left roll, sh right pitch, sh left pitch, sh right yaw, sh left yaw, elbow right, elbow left

        self.rSrelT[0,3] = self.params_length[2]
        self.rSrelT[1,3] = -self.params_length[1]
        self.rSrelT[2,3] = -self.params_length[0]



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
        r_forearm_pose = rospy.Publisher("/r_forearm/pose", PoseStamped, queue_size = 10)
        self.r_elbowpred_pose = rospy.Publisher("/r_elbowpred/pose", PoseStamped, queue_size = 10)
        self.r_handpred_pose = rospy.Publisher("/r_handpred/pose", PoseStamped, queue_size = 10)
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

        for topic, msg, t in bag.read_messages():

            if topic == '/fsascan':
                self.mat_sampled = True
                latest_scan_time = t
                self.current_physical_pressure_map_callback(msg)
                counter += 1
                print t, 'count: ',counter, rando
            elif topic == '/abdout0':
                self.publish_floatarr(msg,abdout)
            elif topic == '/head_o/pose':
                self.publish_pose(msg,head_pose)
            elif topic == '/l_ankle_o/pose':
                self.publish_pose(msg,l_ankle_pose)
                rando = msg.transform.translation.x
            elif topic == 'l_elbow_o/pose':
                self.publish_pose(msg,l_elbow_pose)
                self.l_elbow_msg = msg
                self.l_E[0, 0] = msg.transform.translation.x
                self.l_E[1, 0] = msg.transform.translation.y
                self.l_E[2, 0] = msg.transform.translation.z
                self.l_E[3, 0] = 1
            elif topic == '/l_hand_o/pose':
                self.publish_pose(msg,l_hand_pose)
                self.l_hand_msg = msg
                self.l_H[0, 0] = msg.transform.translation.x
                self.l_H[1, 0] = msg.transform.translation.y
                self.l_H[2, 0] = msg.transform.translation.z
                self.l_H[3, 0] = 1
            elif topic == '/l_knee_o/pose':
                self.publish_pose(msg,l_knee_pose)
            elif topic == '/r_ankle_o/pose':
                self.publish_pose(msg,r_ankle_pose)
            elif topic == '/r_elbow_o/pose':
                self.publish_pose(msg,r_elbow_pose)
                self.r_elbow_msg = msg
                self.r_E[0, 0] = msg.transform.translation.x
                self.r_E[1, 0] = msg.transform.translation.y
                self.r_E[2, 0] = msg.transform.translation.z
                self.r_E[3, 0] = 1
            elif topic == '/r_hand_o/pose':
                self.publish_pose(msg,r_hand_pose)
                self.r_hand_msg = msg
                self.r_H[0, 0] = msg.transform.translation.x
                self.r_H[1, 0] = msg.transform.translation.y
                self.r_H[2, 0] = msg.transform.translation.z
                self.r_H[3, 0] = 1

            elif topic == '/r_knee_o/pose':
                self.publish_pose(msg,r_knee_pose)
            elif topic == '/torso_o/pose':
                msg_right_sampled = True
                self.publish_pose(msg,torso_pose)
                self.T[0, 0] = msg.transform.translation.x
                self.T[1, 0] = msg.transform.translation.y
                self.T[2, 0] = msg.transform.translation.z
                self.T[3, 0] = 1.

                self.TrelO[:,3] = self.T.T

                msg_mod = msg
                msg_mod.transform.rotation.x = 0.
                msg_mod.transform.rotation.y = 0.
                msg_mod.transform.rotation.z = 1.
                msg_mod.transform.rotation.w = 1.
                msg_mod.transform.translation.y = msg.transform.translation.y - self.params_length[1]
                msg_mod.transform.translation.z = msg.transform.translation.z - self.params_length[0]
                self.publish_pose(msg_mod,torso_vert_pose)

                msg_right = msg_mod
                msg_right.transform.rotation.x = 1.
                msg_right.transform.rotation.y = 0.
                msg_right.transform.rotation.z = 0.
                msg_right.transform.rotation.w = 1.
                msg_right.transform.translation.x = msg_mod.transform.translation.x + self.params_length[2]
                self.publish_pose(msg_right, r_shoulder_pose)

                self.r_S[0,0] = msg_right.transform.translation.x
                self.r_S[1,0] = msg_right.transform.translation.y
                self.r_S[2,0] = msg_right.transform.translation.z
                self.r_S[3,0] = 1

                # parameter for the length between hand and elbow. Should be around 0.2 meters.
                self.params_length[6] = np.linalg.norm(self.r_H - self.r_E)

                # get the length of the right shoulder to right elbow
                self.params_length[4] = np.linalg.norm(self.r_E - self.r_S)

                #get the shoulder pitch
                SE_mag = np.copy(self.params_length[4])
                self.r_SE = self.r_S[0:3] - self.r_E[0:3]
                self.r_SE = np.matmul(self.r_ROT_OS, self.r_SE)
                self.r_SEn = np.copy(self.r_SE)/SE_mag
                self.params_angle[2] = np.degrees(np.arcsin(self.r_SEn[1,0]))

                #get shoulder yaw
                self.params_angle[4] = np.degrees(np.arctan(self.r_SEn[0,0]/self.r_SEn[2,0]))

                # calculate the shoulder roll
                SH_mag = np.linalg.norm(self.r_H - self.r_S) # first get distance between shoulder and hand
                self.r_SH = self.r_S[0:3] - self.r_H[0:3]
                self.r_SH = np.matmul(self.r_ROT_OS, self.r_SH)
                self.r_SHn = np.copy(self.r_SH) / SH_mag
                self.r_Spxn[0,0], self.r_Spxn[1,0], self.r_Spxn[2,0] = fsolve(self.equations_s_RSpx, (.5, .5, .5 ))
                self.r_Spx = np.copy(self.r_Spxn)*SH_mag
                self.r_SH_pp = self.r_SE*(np.dot(self.r_SE.T,self.r_SH))/(np.linalg.norm(self.r_SE)*np.linalg.norm(self.r_SE)) - self.r_SH
                self.r_SH_pp_mag = np.linalg.norm(self.r_SH_pp)
                self.r_Spx_pp = self.r_SE * (np.dot(self.r_SE.T, self.r_Spx)) / (np.linalg.norm(self.r_SE) * np.linalg.norm(self.r_SE)) - self.r_Spx
                self.r_Spx_pp_mag = np.linalg.norm(self.r_Spx_pp)
                self.params_angle[0] = np.degrees(np.arccos(np.dot(self.r_SH_pp.T,self.r_Spx_pp)/(self.r_SH_pp_mag*self.r_Spx_pp_mag)))

                # get the elbow angle with the law of cosines between lengths
                self.params_angle[6] = np.degrees(np.arccos((np.square(self.params_length[4]) + np.square(
                    self.params_length[6]) - np.square(SH_mag)) / (2 * self.params_length[4] * self.params_length[6])))






            #convert the the RPH back to XYZ so we can visualize a vector and make sure the calculations are correct.
                #self.ROT_rph, self.rErelrS = self.RPH_to_XYZ(self.params_angle[0],self.params_angle[2],self.params_angle[4])

                #self.Pr_SE = np.matmul(self.rErelrS,np.array([[self.params_length[4]],[0],[0],[1]]))

                #self.rErelrS[0:3, 3] = -self.Pr_SE[0:3,0]

                #self.rHrelrE = tft.rotation_matrix(np.deg2rad(self.params_angle[6]), (0,0,1))
                #self.Pr_EH = np.matmul(self.rHrelrE, np.array([[self.params_length[6]], [0], [0],  [1]]))
                #print self.Pr_EH
                #print self.rSrelT
                #print self.rErelrS

                #print np.linalg.norm(self.r_SE[:,0]), np.sqrt(np.square(self.r_SE[1,0])+np.square(self.r_SE[0,0])+np.square(self.r_SE[2,0]))
                #print self.r_SE[:,0], self.r_SH[:,0]
                #print self.r_SE
                #print self.ROT_rph
                #print np.cross(self.r_SE[:,0].T,self.r_SH[:,0].T)/(np.linalg.norm(self.r_SE)*np.linalg.norm(self.r_SH)), 'rse cross rsh'
                #S_H = np.cross(self.r_SE[:,0].T,self.r_SH[:,0].T)/(np.linalg.norm(np.cross(self.r_SE[:,0].T,self.r_SH[:,0].T)))
                #S_E = self.r_SE[:,0].T/np.linalg.norm(self.r_SE[:,0].T)
                #print S_H
                #print S_E
                #print np.cross(S_H,S_E)/(np.linalg.norm(np.cross(S_H,S_E))), 'end'
                #print self.ROT_rph
                #
               # print self.rErelrS
                #print self.r_S
                #print np.matmul(np.matmul(np.matmul(self.rErelrS, self.rSrotrS), self.rSrelT), self.T)

                #self.Pr_SE[0:3, 0] =  np.matrix(self.r_SE.T)




                msg_left = msg_mod
                msg_left.transform.rotation.x = 1
                msg_left.transform.rotation.y = 0
                msg_left.transform.rotation.z = 0
                msg_left.transform.rotation.w = 1
                msg_left.transform.translation.x = msg_mod.transform.translation.x - self.params_length[3] - self.params_length[2]
                self.publish_pose(msg_left, l_shoulder_pose)


                # parameter for the length between hand and elbow. Should be around 0.2 meters.
                self.params_length[7] = np.linalg.norm(self.l_H - self.l_E)

                if self.l_elbow_msg is not None:
                    # get the length of the left shoulder to left elbow
                    self.params_length[5] = np.sqrt(
                        np.square(msg_left.transform.translation.x - self.l_elbow_msg.transform.translation.x) + np.square(
                            msg_left.transform.translation.y - self.l_elbow_msg.transform.translation.y) + np.square(
                            msg_left.transform.translation.z - self.l_elbow_msg.transform.translation.z))

                    # get the shoulder pitch
                    SE_mag = np.copy(self.params_length[5])
                    self.l_SE[0, 0] = msg_left.transform.translation.x - self.l_elbow_msg.transform.translation.x
                    self.l_SE[1, 0] = msg_left.transform.translation.y - self.l_elbow_msg.transform.translation.y
                    self.l_SE[2, 0] = msg_left.transform.translation.z - self.l_elbow_msg.transform.translation.z
                    self.l_SE = np.matmul(self.l_ROT_OS, self.l_SE)
                    self.l_SEn = np.copy(self.l_SE) / SE_mag
                    self.params_angle[3] = np.degrees(np.arcsin(self.l_SEn[1, 0]))

                    # get shoulder yaw
                    self.params_angle[5] = np.degrees(np.arctan(self.l_SEn[0, 0] / self.l_SEn[2, 0]))

                    #get the elbow angle
                    if self.l_hand_msg is not None:
                        #first get distance between shoulder and hand
                        SH_mag = np.sqrt(
                            np.square(msg_left.transform.translation.x - self.l_hand_msg.transform.translation.x) + np.square(
                                msg_left.transform.translation.y - self.l_hand_msg.transform.translation.y) + np.square(
                                msg_left.transform.translation.z - self.l_hand_msg.transform.translation.z))
                        #now apply law of cosines
                        self.params_angle[7] = np.degrees(np.arccos((np.square(self.params_length[5]) + np.square(
                            self.params_length[7]) - np.square(SH_mag)) / (
                                                         2 * self.params_length[5] * self.params_length[7])))

                    # calculate the shoulder roll
                        self.l_SH[0, 0] = msg_left.transform.translation.x - self.l_hand_msg.transform.translation.x
                        self.l_SH[1, 0] = msg_left.transform.translation.y - self.l_hand_msg.transform.translation.y
                        self.l_SH[2, 0] = msg_left.transform.translation.z - self.l_hand_msg.transform.translation.z
                        self.l_SH = np.matmul(self.l_ROT_OS, self.l_SH)
                        self.l_SHn = np.copy(self.l_SH) / SH_mag





            elif topic == '/mat_o/pose':
                self.publish_pose(msg,mat_o_pose)
            elif topic == '/mat_x/pose':
                self.publish_pose(msg,mat_x_pose)
            elif topic == '/mat_y/pose':
                self.publish_pose(msg,mat_y_pose)

            if self.r_elbow_msg is not None and self.l_hand_msg is not None:
                self.forward_kinematics(self.params_length,self.params_angle,self.T, msg_right)
                msg_right_sampled = False


            if self.mat_sampled and self.ground_truth_sampled and np.abs(latest_scan_time.to_sec() - self.latest_ground_truth_time.to_sec())<0.1:
                print '#############################################################################'
                print self.params_angle[0], self.params_angle[2], self.params_angle[4]
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
        return mean_err, std_err, count, time_range

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


    def equations_s_RSpx(self,p):
        x, y, z = p
        return (x ** 2 + y ** 2 + z ** 2 - 1, self.r_SEn[2, 0] * y - self.r_SEn[1, 0] * z,
                self.r_SEn[0, 0] * x + self.r_SEn[1, 0] * y + self.r_SEn[2, 0] * z - self.r_SEn[0, 0] * self.r_SHn[
                    0, 0] - self.r_SEn[1, 0] * self.r_SHn[1, 0] - self.r_SEn[2, 0] * self.r_SHn[2, 0])

    def RPH_to_XYZ(self,roll,pitch,yaw):
        roll = np.deg2rad(roll+180)
        pitch = np.deg2rad(pitch)
        yaw = np.deg2rad(yaw)
        R = np.zeros((3,3))
        T = np.zeros((4,4))
        R[0, 0] = np.sin(yaw) * np.cos(pitch)
        R[0, 1] = - np.sin(yaw) * np.sin(pitch) * np.sin(roll) + np.cos(yaw) * np.cos(roll)
        R[0, 2] = - np.sin(yaw) * np.sin(pitch) * np.cos(roll) - np.cos(yaw) * np.sin(roll)
        R[1, 0] = np.sin(pitch)
        R[1, 1] = np.cos(pitch) * np.sin(roll)
        R[1, 2] = np.cos(pitch) * np.sin(roll)
        R[2, 0] = np.cos(yaw) * np.cos(pitch)
        R[2, 1] = - np.cos(yaw) * np.sin(pitch) * np.sin(roll) - np.sin(yaw) * np.cos(roll)
        R[2, 2] = - np.cos(yaw) * np.sin(pitch) * np.cos(roll) + np.sin(yaw) * np.sin(roll)

        T[0:3,0:3]=R
        T[3,3] = 1

        return R, T

    def forward_kinematics(self,lengths, angles, torso, msg):

        TrelO = tft.identity_matrix()
        rSrelT = tft.identity_matrix()
        rSrotrS = tft.identity_matrix()
        rErelrS = tft.identity_matrix()
        Pr_SE = np.zeros((4,1))
        Pr_EH = np.zeros((4,1))



        TrelO[:, 3] = torso.T
        rSrelT[0,3] = lengths[2]
        rSrelT[1,3] = -lengths[1]
        rSrelT[2,3] = -lengths[0]
        rSrotrS[0:3, 0:3] = np.matrix('0,0,-1;0,1,0;-1,0,0')

        _, rErelrS = self.RPH_to_XYZ(angles[0], angles[2], angles[4])

        Pr_SE = np.matmul(rErelrS, np.array([[lengths[4]], [0], [0], [1]]))

        rErelrS[0:3, 3] = -Pr_SE[0:3, 0]

        rHrelrE = tft.rotation_matrix(np.deg2rad(angles[6]), (0, 0, 1))
        Pr_EH = np.matmul(rHrelrE, np.array([[lengths[6]], [0], [0], [1]]))

        Pr_SE = -Pr_SE
        Pr_SE[3, 0] = 1

        msg.transform.translation.x = np.matrix(np.matmul(np.matmul(np.matmul(TrelO, rSrelT), rSrotrS), Pr_SE))[0, 0]
        msg.transform.translation.y = np.matrix(np.matmul(np.matmul(np.matmul(TrelO, rSrelT), rSrotrS), Pr_SE))[1, 0]
        msg.transform.translation.z = np.matrix(np.matmul(np.matmul(np.matmul(TrelO, rSrelT), rSrotrS), Pr_SE))[2, 0]
        # The handler function for the turtle pose message broadcasts this turtle's translation and rotation, and
        # publishes it as a transform from frame "shoulder" to frame "elbow".
        #self, tf_broadcaster.sendTransform((msg.x, msg.y, msg.z),
        #                                   [0,0,0,1], #rotation in quaternions
        #                                   # tft.quaternion_from_euler(0, 0, msg.theta),
        #                                   rospy.Time.now(),
        #                                   "elbow",
        #                                   "shoulder")

        self.publish_pose(msg, self.r_elbowpred_pose)

        msg.transform.translation.x = np.matrix(np.matmul(np.matmul(np.matmul(np.matmul(TrelO, rSrelT), rSrotrS), rErelrS), Pr_EH))[0, 0]
        msg.transform.translation.y = np.matrix(np.matmul(np.matmul(np.matmul(np.matmul(TrelO, rSrelT), rSrotrS), rErelrS), Pr_EH))[1, 0]
        msg.transform.translation.z = np.matrix(np.matmul(np.matmul(np.matmul(np.matmul(TrelO, rSrelT), rSrotrS), rErelrS), Pr_EH))[2, 0]
        self.publish_pose(msg, self.r_handpred_pose)




        #should = np.zeros((4, 2))
        #should = np.matrix(should)
        #should[:, 0] = np.matrix(np.matmul(self.TrelO, self.rSrelT[:, 3].T)).T
        #should[:, 1] = np.matrix(np.matmul(np.matmul(self.TrelO, self.rSrotrS), self.rSrelT[:, 3].T)).T
        # print should,'shoulder in the origin and rotated frame'



        #elbpred = np.zeros((4, 2))
        #elbpred = np.matrix(elbpred)
        # elbpred[:, 0] = np.matrix(np.matmul(np.matmul(np.matmul(np.matmul(self.rSrotrS.T,self.rErelrS),self.rSrotrS), self.TrelO), self.rSrelT[:,3].T)).T
        #elbpred[:, 0] = np.matrix(np.matmul(np.matmul(np.matmul(self.TrelO, self.rSrelT), self.rSrotrS), self.Pr_SE))
        # print elbpred, 'elbow predicted in the origin and reference frame'


        #elb = np.zeros((4, 2))
        #elb = np.matrix(elb)
        #elb[:, 0] = self.r_E
        #elb[:, 1] = np.matmul(self.rSrotrS, self.r_E)
        # print elb, 'elbow in the origin and reference frame'


        #handpred = np.zeros((4, 2))
        #handpred = np.matrix(handpred)
        # elbpred[:, 0] = np.matrix(np.matmul(np.matmul(np.matmul(np.matmul(self.rSrotrS.T,self.rErelrS),self.rSrotrS), self.TrelO), self.rSrelT[:,3].T)).T
        #handpred[:, 0] = np.matrix(
        #    np.matmul(np.matmul(np.matmul(np.matmul(self.TrelO, self.rSrelT), self.rSrotrS), self.rErelrS), self.Pr_EH))
        # print handpred, self.params_length[6], np.linalg.norm(self.Pr_EH[0:3]), 'hand predicted in the origin and reference frame'


        #hand = np.zeros((4, 2))
        #hand = np.matrix(hand)
        #hand[:, 0] = self.r_H
        #hand[:, 1] = np.matmul(self.rSrotrS, self.r_H)
        # print hand, 'hand in the origin and reference frame'


        # shouldelbdiff = np.zeros((4,2))
        # shouldelbdiff = np.matrix(shouldelbdiff)
        # shouldelbdiff[:,0] = np.matmul(self.rSrelT, self.T) - self.r_E
        # shouldelbdiff[:,1]= np.matmul(np.matmul(self.rSrotrS,self.rSrelT), self.T) - np.matmul(self.rSrotrS,self.r_E)
        # print shouldelbdiff, np.linalg.norm(shouldelbdiff[:,1]), self.params_length[4], 'shoulder and elbow difference'





        # shouldelbpreddiff = np.zeros((4, 2))
        # shouldelbpreddiff = np.matrix(shouldelbpreddiff)
        # shouldelbpreddiff[:, 0] = np.matrix(np.matmul(self.TrelO, self.rSrelT[:,3].T)).T - np.matrix(np.matmul(np.matmul(np.matmul(self.TrelO, self.rSrelT), self.rSrotrS), self.Pr_SE))
        # shouldelbpreddiff[:, 1] = 0
        # print shouldelbpreddiff, np.linalg.norm(shouldelbpreddiff[:,1]), 'shoulder and elbow predicted difference'
        # print self.rErelrS,self.rSrotrS,self.rSrelT


    def publish_floatarr(self,msg,floatarr_publisher):
        floatarr_publisher.publish(msg)
    

    def get_elapsed_time(self):
        return self.elapsed_time

    def relu(self, x):
        if x < 0:
            return 0.0
        else:
            return x

    def sigmoid(self, x):
        #return 1 / (1 + math.exp(-x))
        return ((x / (1 + abs(x))) + 1)/2

    def world_to_mat(self, w_data):
        '''Converts a vector in the world frame to a vector in the map frame.
        Depends on the calibration of the MoCap room. Be sure to change this 
        when the calibration file changes. This function mainly helps in
        visualizing the joint coordinates on the pressure mat.
        Input: w_data: which is a 3 x 1 vector in the world frame'''
        #The homogenous transformation matrix from world to mat
        O_w_m = np.matrix([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        #O_w_m = np.matrix(np.reshape(self.R_world_mat, (3, 3)))
        O_m_w = O_w_m.T
        p_mat_world = O_m_w.dot(-np.asarray(self.p_world_mat))
        B_m_w = np.concatenate((O_m_w, p_mat_world.T), axis=1)
        last_row = np.array([[0, 0, 0, 1]])
        B_m_w = np.concatenate((B_m_w, last_row), axis=0)
        w_data = np.append(w_data, [1.0])
        #Convert input to the mat frame vector
        m_data = B_m_w.dot(w_data)
        return np.squeeze(np.asarray(m_data[0, :3]))

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

    def detect_head(self):
        '''Computes blobs in pressure map and return top
        blob as head'''
        # start_time = rospy.Time.now()
        # Select top 20 pixels of pressure map
        p_map = self.pressure_map
        # plt.matshow(p_map)
        # plt.show()

        com = np.array(ndimage.measurements.center_of_mass(p_map))
        com[0] = 10.0
        # print com
        # print "In discrete coordinates"
        # print self.head_center_2d[0, :]
        taxels_to_meters = np.array([MAT_HEIGHT / (NUMOFTAXELS_X * self.zoom_factor),
                                     MAT_WIDTH / (NUMOFTAXELS_Y * self.zoom_factor),
                                     1])

        self.head_center_2d = np.append(com, 1.0)
        # Median Filter
        # self.head_center_2d = taxels_to_meters * self.head_center_2d
        # print self.head_center_2d
        # positions = self.head_pos_buf.get_array()
        # pos = positions[positions[:, 1].argsort()]
        y, x, r = self.head_center_2d

        # mat_B_head = np.eye(4)
        # mat_B_head[0:3, 3] = np.array([x, y, -0.05])
        # mat_B_head[0:3, 3] = np.array([0,0,0])
        # print "In Mat Coordinates:"
        # print y, x
        # head_rest_B_head = np.matrix(self.head_rest_B_mat) * np.matrix(mat_B_head)
        # print "In head_rest_link coordinates:"
        # print head_rest_B_head[0:3, 3]
        # self.elapsed_time.append(rospy.Time.now() - start_time)
        return y, x

    def detect_blob(self):
        '''Computes blobs in pressure map'''
        #p_map = self.pressure_map[:20,:]
        p_map = self.pressure_map
        weights = np.zeros(np.shape(p_map))
        for i in range(np.shape(p_map)[0]):
            weights[i, :] = self.sigmoid((np.shape(p_map)[0]/8.533 - i))
        p_map = np.array(weights)*np.array(p_map)
        #plt.matshow(p_map)
        #plt.show()
        blobs = blob_doh(p_map, 
                         min_sigma=1, 
                         max_sigma=7, 
                         threshold=20,
                         overlap=0.1) 
        numofblobs = np.shape(blobs)[0] 
        # print "Number of Blobs Detected:{}".format(numofblobs)
        return blobs

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

    def get_ground_truth(self):
        target_raw = np.array(self.head_pose)
        target_mat = self.world_to_mat(target_raw)
        target_discrete = self.mat_to_taxels(target_mat) + np.array([0,3])
        target_cont = target_mat #+ np.array([0.0, -0.0410, 0.0])
        return target_cont[:2]

    def run(self):
        '''Runs pose estimation''' 
        head_center = [0, 0, 0]
        self.pos = 0
        self.total_count = 0
        self.count = 0 
        self.error_array = []
        while not rospy.is_shutdown():
            if self.mat_sampled:
                self.count += 1 
                print "Iteration:{}".format(self.count)
                blobs = self.detect_blob()
                if blobs.any():
                    head_center = blobs[0, :]
                taxels_to_meters_coeff = np.array([MAT_HEIGHT/(NUMOFTAXELS_X*self.zoom_factor), 
                                            -MAT_WIDTH/(NUMOFTAXELS_Y*self.zoom_factor), 
                                            1])

                taxels_to_meters_offset = np.array([MAT_HEIGHT, 0.0, 0.0])
                y, x, r = (taxels_to_meters_offset - taxels_to_meters_coeff*head_center)
                print "X:{}, Y:{}".format(x,y)
                print "Radius:{}".format(r)
                ground_truth = np.array(self.get_ground_truth()) 
                print "Final Ground Truth:"
                print ground_truth
                #self.visualize_pressure_map(self.pressure_map, rotated_targets=[x, y, r],\
                #                            plot_3d=False)
                error = np.linalg.norm(np.array([x,y]) - np.array(ground_truth))
                self.error_array.append(error)
                if self.count == 100:
                    mean_err = np.mean(self.error_array)
                    std_err = np.std(self.error_array)
                    print "Average Error: {}".format(mean_err)
                    print "Standard Deviation : {}".format(std_err)
                    sys.exit()
                self.mat_sampled = False
            else:
                pass

if __name__ == '__main__':
    rospy.init_node('calc_mean_std_of_head_detector_node')
    head_blob = HeadDetector()
    subject_means = []
    subject_std = []
    subject_scan_count = 0
    time_ranges = []
    filename = '_full_trial_RH1.bag'
    #filename = '_mat_o.bag'
    method = 'center_of_mass'  # options are: 'blob', 'center_of_mass'
    for subject in [9,10,11,12,13,14,15,16,17,18]:
    #for subject in [8]:
        new_mean, new_std, new_count, new_time_range = head_blob.read_bag(subject, filename, method, visualize=False)
        subject_means.append(new_mean)
        subject_std.append(new_std)
        subject_scan_count += new_count
        a_range = new_time_range.to_sec()
        time_ranges.append(a_range)
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
