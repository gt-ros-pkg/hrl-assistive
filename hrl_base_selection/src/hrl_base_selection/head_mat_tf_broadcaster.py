#!/usr/bin/env python
import rospy, roslib
import sys, os
import random
import numpy as np
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
        self.mat_size = (NUMOFTAXELS_X, NUMOFTAXELS_Y)
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.tf_listener = tf.TransformListener()
        self.head_center_2d = None
        self.zoom_factor = 2
        self.hist_size = 30
        self.width_offset = 0.05
        self.head_pos_buf  = cb.CircularBuffer(self.hist_size, (3,))
        rospy.sleep(2)
        rospy.Subscriber("/fsascan", FloatArrayBare, self.current_physical_pressure_map_callback)
        self.mat_sampled = False
        #while (not self.tf_listener.canTransform('map', 'autobed/head_rest_link', rospy.Time(0))) and not rospy.is_shutdown():
        #while (not self.tf_listener.canTransform('base_link', 'torso_lift_link', rospy.Time(0))):
        #    print 'Waiting for head localization in world.'
        #    rospy.sleep(1)
        #Initialize some constant transforms
        self.head_rest_B_mat = np.eye(4)
        self.head_rest_B_mat[0:3, 0:3] = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
        self.head_rest_B_mat[0:3, 3] = np.array([0.735-0.2286, 0, -(MAT_HALF_WIDTH - self.width_offset)])
        
        rospy.sleep(1)
        print 'Head mat tf broadcaster is up and running'
        rospy.sleep(1)
        self.run()

    def current_physical_pressure_map_callback(self, data):
        '''This callback accepts incoming pressure map from 
        the Vista Medical Pressure Mat and sends it out.'''
        p_array = data.data
        p_map_raw = np.reshape(p_array, self.mat_size)
        p_map_hres=ndimage.zoom(p_map_raw, self.zoom_factor, order=1)
        self.pressure_map=p_map_hres
        self.mat_sampled = True


    def sigmoid(self, x):
        #return 1 / (1 + math.exp(-x))
        return ((x / (1 + abs(x))) + 1)/2

    def detect_head(self):
        '''Computes blobs in pressure map and return top 
        blob as head'''
        #plt.matshow(self.pressure_map)
        #plt.show()
        #Select top 20 pixels of pressure map
        p_map = self.pressure_map
        weights = np.zeros(np.shape(p_map))
        for i in range(np.shape(p_map)[0]):
            weights[i, :] = self.sigmoid((np.shape(p_map)[0]/8.533 - i))
        p_map = np.array(weights)*np.array(p_map)
        #plt.matshow(p_map)
        #plt.show()
        try:
            blobs = blob_doh(p_map, 
                             min_sigma=4,#1, 
                             max_sigma=6,#7, 
                             threshold=20,
                             overlap=0.1) 
        except:
            blobs = np.copy(self.head_center_2d)
            print "Head Not On Mat!"
        if blobs.any():
            self.head_center_2d = blobs
        #print "In discrete coordinates"
        #print self.head_center_2d[0, :]
        taxels_to_meters = np.array([MAT_HEIGHT/(NUMOFTAXELS_X*self.zoom_factor), 
                                     MAT_WIDTH/(NUMOFTAXELS_Y*self.zoom_factor), 
                                     1])
        if self.head_center_2d is None:
            return None


        #No Filters
        #y, x, r = taxels_to_meters*self.head_center_2d[0, :]

        #Median Filter
        self.head_pos_buf.append(taxels_to_meters*self.head_center_2d[0, :])
        positions = self.head_pos_buf.get_array()
        pos = positions[positions[:, 1].argsort()]
        y, x, r = pos[pos.shape[0]/2]

        mat_B_head = np.eye(4)
        mat_B_head[0:3, 3] = np.array([x, y, -0.05])
        #mat_B_head[0:3, 3] = np.array([0,0,0])
        #print "In Mat Coordinates:"
        #print x, y
        head_rest_B_head = np.matrix(self.head_rest_B_mat)*np.matrix(mat_B_head)
        #print "In head_rest_link coordinates:"
        #print head_rest_B_head[0:3, 3]
        
        return head_rest_B_head

    def run(self):
        '''Runs pose estimation''' 
        rate = rospy.Rate(50.0)
        while not rospy.is_shutdown():
            if self.mat_sampled:
                self.mat_sampled = False
                head_rest_B_head = self.detect_head()
                if head_rest_B_head is not None:
                    # (newtrans, newrot) = self.tf_listener.lookupTransform('autobed/base_link', \
                    #                                                       'autobed/head_rest_link', rospy.Time(0))
                    #(newtrans, newrot) = self.tf_listener.lookupTransform('base_link', \
                    #                                                      'torso_lift_link', rospy.Time(0))
                    # bedbase_B_head_rest = createBMatrix(newtrans, newrot)
                    # bedbase_B_head = bedbase_B_head_rest*head_rest_B_head
                    # (out_trans, out_rot) = Bmat_to_pos_quat(bedbase_B_head)
                    (out_trans, out_rot) = Bmat_to_pos_quat(head_rest_B_head)
                    try:
                        self.tf_broadcaster.sendTransform(out_trans, 
                                                          out_rot,
                                                          rospy.Time.now(),
                                                          'user_head_link',
                                                          'autobed/head_rest_link')
                                                          #'torso_lift_link')
                    
                    except:
                        print 'Head TF broadcaster crashed trying to broadcast!'
                        break
                else:
                    print 'I have no blobs visible on the pressure mat by the head'
            else:
                pass
            rate.sleep()

if __name__ == '__main__':
    rospy.init_node('head_pose_broadcaster', anonymous=True)
    head_blob = HeadDetector()
    #head_blob.run()


