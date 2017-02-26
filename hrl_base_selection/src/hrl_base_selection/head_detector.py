#!/usr/bin/env python
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
        rospy.init_node('head_pose_estimator', anonymous=True)
        rospy.Subscriber("/fsascan", FloatArrayBare, 
                self.current_physical_pressure_map_callback)
        rospy.Subscriber("/head_o/pose", TransformStamped,
                self.head_origin_callback)
        # self.database_path = '/home/yashc/Desktop/dataset/subject_4'
        self.database_path = '/home/yashc/Desktop/dataset/subject_4'
        # [self.p_world_mat, self.R_world_mat] = pkl.load(
        #         open(os.path.join(self.database_path,'mat_axes.p'), "r"))
        self.p_world_mat = np.array([0,0,0])
        self.R_world_mat = np.eye(3)
        print self.p_world_mat
        print self.R_world_mat
        self.mat_sampled = False
        self.mat_pose = []
        self.head_pose = []
        self.zoom_factor = 2
        print "Ready to start listening..."

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
        p_array = data.data
        p_map_raw = np.reshape(p_array, self.mat_size)
        p_map_hres=ndimage.zoom(p_map_raw, 2, order=1)
        self.pressure_map=p_map_hres
        self.mat_sampled = True

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
        print "Number of Blobs Detected:{}".format(numofblobs)
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
            rotated_target_coord = rotated_targets           

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
    head_blob = HeadDetector()
    head_blob.run()


