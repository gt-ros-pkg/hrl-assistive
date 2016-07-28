#!/usr/bin/env python
import rospy, roslib
import os, sys
import numpy as np
import random, math
import pickle as pkl
import atexit
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.feature import hog
from sklearn import svm
from matplotlib.patches import Rectangle   
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


class DetectHeadFromShoulders:
    '''Detects the head of a person sleeping on the autobed'''
    def __init__(self, dataset_directory):
        self.mat_size = (NUMOFTAXELS_X, NUMOFTAXELS_Y)
        rospy.init_node('head_shoulder_pose', anonymous=True)
        self.feature_params = {'template_size': [24, 36], 'hog_cell_size': 6}
        self.clf = pkl.load(open('./svm_classifier.p', 'rb'))
        rospy.Subscriber("/fsascan", FloatArrayBare, self.current_physical_pressure_map_callback)
        rospy.Subscriber("/head_o/pose", TransformStamped,
                self.head_origin_callback)
        self.database_path = dataset_directory
        [self.p_world_mat, self.R_world_mat] = pkl.load(
                open(os.path.join(self.database_path,'mat_axes.p'), "r"))         
        self.mat_sampled = False
        self.mat_pose = []
        self.head_pose = []
        self.zoom_factor = 2
        atexit.register(self.compute_mean_error)
        print "Ready to start listening..."


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


    def compute_mean_error(self):
        mean_err = np.mean(self.error_array)
        std_err = np.std(self.error_array)
        print "Average Error: {}".format(mean_err)
        print "Standard Deviation : {}".format(std_err)
        sys.exit()

    def run_detector(self):
	bboxes = np.zeros((0,4))
	confidences = np.zeros((0,1))
	num_scales = 1
        feature_params = self.feature_params
	L = feature_params['template_size']
	step = np.array(feature_params['template_size']) / feature_params['hog_cell_size']
	hog_cell_size = feature_params['hog_cell_size']
	THRESH = 0.0
	count = 0
        curr_image = self.pressure_map
        curr_image = curr_image/100.
        temp_bboxes = np.zeros((0, 4))
        temp_confidences = np.zeros((0,1))
        for row in xrange(0,np.shape(curr_image)[0] - L[0], hog_cell_size):
            for col in xrange(0,np.shape(curr_image)[1] - L[1], hog_cell_size):
                cropped_img = curr_image[row:row+L[0], col:col+L[1]]
                HOG, viz = hog(cropped_img, 
                               orientations=9, 
                               pixels_per_cell=(6,6), 
                               cells_per_block=(6,4), 
                               visualise=True)
                
                test_score = self.clf.predict(HOG)
                if test_score > THRESH:
                    bbox = np.zeros((1,4))
                    bbox[:, 0] = np.floor(col)
                    bbox[:, 1] = np.floor(row)
                    bbox[:, 2] = np.floor(np.floor(col) + L[1] -1)
                    bbox[:, 3] = np.floor(np.floor(row) + L[0] -1)
                    temp_bboxes = np.vstack((temp_bboxes, bbox))   
            try:
                bboxes = np.vstack((bboxes, temp_bboxes[1 , :])) 
            except:
                temp_bbox = np.array([0., 0., 23., 35.])
                bboxes = np.vstack((bboxes, temp_bbox))
        return bboxes[1, :]

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


    def head_center_from_bbox(self, bboxes):
        '''Compute head Center from bounding box'''
        box_start= np.array([bboxes[0], bboxes[1]])
        box_size = self.feature_params['template_size']
        print box_start, box_size
        head_center = np.array([(box_start[0] + (box_size[1]*0.5)),
                               (box_start[1] + (box_size[0]*0.15)),
                                0])
        return head_center


    def get_ground_truth(self):
        target_raw = np.array(self.head_pose)
        target_mat = self.world_to_mat(target_raw)
        target_discrete = self.mat_to_taxels(target_mat) + np.array([0,3])
        target_cont = target_mat #+ np.array([0.0, -0.0410, 0.0])
        return target_cont[:2]


    def visualize_pressure_map(self, pressure_map_matrix, rotated_targets=None, bbox=None):
        '''Visualizing a plot of the pressure map'''        
        feature_params = self.feature_params
        fig = plt.gcf()
        plt.ion()
        ax = fig.add_subplot(111, aspect='equal')
        ax.matshow(pressure_map_matrix, cmap=plt.cm.bwr)
        if rotated_targets is not None:
            rotated_target_coord = rotated_targets           
            ax.plot(rotated_target_coord[0], rotated_target_coord[1],\
                         'y*', ms=10)
            ax.add_patch(Rectangle((bbox[0], bbox[1]), 
                        feature_params['template_size'][1], 
                        feature_params['template_size'][0], 
                        fill=False, 
                        alpha=1,
                        edgecolor="green",
                        linewidth=2
                        ))
        plt.draw()
        plt.pause(0.05) 
        plt.clf()


    def run(self):
        '''Detects head location using bounding box'''
        head_center = [0, 0, 0]
        self.pos = 0
        self.count = 0 
        self.error_array = []
        while not rospy.is_shutdown():
            if self.mat_sampled:
                self.count += 1 
                print "Iteration:{}".format(self.count)
                bboxes = self.run_detector()
                print bboxes
                head_center = self.head_center_from_bbox(bboxes)
                self.visualize_pressure_map(self.pressure_map, rotated_targets=head_center, bbox = bboxes)
                plt.show()
                taxels_to_meters_coeff = np.array([-MAT_WIDTH/(NUMOFTAXELS_Y*self.zoom_factor), 
                                                    MAT_HEIGHT/(NUMOFTAXELS_X*self.zoom_factor), 
                                                    1])
                taxels_to_meters_offset = np.array([0.0, MAT_HEIGHT, 0.0])
                x, y, r = (taxels_to_meters_offset - taxels_to_meters_coeff*head_center)
                print "X:{}, Y:{}".format(x,y)
                print "Radius:{}".format(r)
                ground_truth = np.array(self.get_ground_truth()) 
                print "Final Ground Truth:"
                print ground_truth
                error = np.linalg.norm(np.array([x,y]) - np.array(ground_truth))
                self.error_array.append(error)
                self.mat_sampled = False
            else:
                pass



if __name__ == '__main__':
    dataset_dir = sys.argv[1]
    head_shoulder = DetectHeadFromShoulders(dataset_dir)
    head_shoulder.run()


