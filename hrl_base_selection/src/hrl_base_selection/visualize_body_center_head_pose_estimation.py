#!/usr/bin/env python
import rospy, roslib
import sys, os
import random
import numpy as np
import threading, copy
import matplotlib.pyplot as plt
import cPickle as pkl
import hrl_lib.circular_buffer as cb
import tf
from scipy import ndimage
from skimage.feature import blob_doh
from hrl_msgs.msg import FloatArrayBare
from helper_functions import createBMatrix, Bmat_to_pos_quat

MAT_WIDTH = 0.74#0.762 #metres
MAT_HEIGHT = 1.75 #1.854 #metres
MAT_HALF_WIDTH = MAT_WIDTH/2 
NUMOFTAXELS_X = 64#73 #taxels
NUMOFTAXELS_Y = 27#30 
LOW_TAXEL_THRESH_X = 0
LOW_TAXEL_THRESH_Y = 0
HIGH_TAXEL_THRESH_X = (NUMOFTAXELS_X - 1) 
HIGH_TAXEL_THRESH_Y = (NUMOFTAXELS_Y - 1) 
INTER_SENSOR_DISTANCE = 0.0286

class HeadDetector:
    '''Detects the head of a person sleeping on the autobed'''
    def __init__(self):
        self.frame_lock = threading.RLock()
        self.mat_size = (NUMOFTAXELS_X, NUMOFTAXELS_Y)

        self.mat_pose = []
        self.head_pose = []
        self.head_center_2d = None
        self.zoom_factor = 2
        self.hist_size = 30
        self.width_offset = 0.05
        self.head_rest_B_head = None
        self.head_pos_buf = cb.CircularBuffer(self.hist_size, (3,))
        rospy.sleep(2)
        self.mat_sampled = False
        #Initialize some constant transforms
        self.head_rest_B_mat = np.eye(4)
        self.head_rest_B_mat[0:3, 0:3] = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
        self.head_rest_B_mat[0:3, 3] = np.array([0.735-0.2286, 0, -(MAT_HALF_WIDTH - self.width_offset)])
        rospy.Subscriber("/fsascan", FloatArrayBare, self.current_physical_pressure_map_callback)
        
        rospy.sleep(1)
        print 'Body Center tf broadcaster is up and running'
        # rospy.sleep(1)
        self.run()

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
                    
    def sigmoid(self, x):
        #return 1 / (1 + math.exp(-x))
        return ((x / (1 + abs(x))) + 1)/2

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

    def run(self):
        '''Runs pose estimation''' 
        rate = rospy.Rate(10.0)
        while not rospy.is_shutdown():
            if self.mat_sampled:
                headx, heady = self.detect_head()
                head_center = np.array([headx, heady, 1.])
                taxels_to_meters_coeff = np.array([MAT_HEIGHT/(NUMOFTAXELS_X*self.zoom_factor),
                                                  -MAT_WIDTH/(NUMOFTAXELS_Y*self.zoom_factor),
                                                  1])
                taxels_to_meters_offset = np.array([MAT_HEIGHT, 0.0, 0.0])
                y, x, r = (taxels_to_meters_offset - taxels_to_meters_coeff*head_center)
                r = 5.
                # self.head_rest_B_head = self.detect_head()
                #out_trans, out_rot = Bmat_to_pos_quat(self.head_rest_B_head)
                #x = out_trans[0]
                #y = out_trans[1]
                #r = 5.
                self.visualize_pressure_map(self.pressure_map, rotated_targets=[x, y, r], plot_3d=False)
            # else:
            #     print 'I have no blobs visible on the pressure mat by the head'
            else:
                pass
            rate.sleep()

if __name__ == '__main__':
    rospy.init_node('head_pose_broadcaster')
    head_blob = HeadDetector()
    #head_blob.run()


