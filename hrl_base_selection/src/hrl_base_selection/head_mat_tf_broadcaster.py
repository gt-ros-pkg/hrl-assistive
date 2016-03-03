#!/usr/bin/env python
import rospy, roslib
import sys, os
import random
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pkl
import tf
from scipy import ndimage
from skimage.feature import blob_doh
from hrl_msgs.msg import FloatArrayBare
from helper_functions import createBMatrix, Bmat_to_pos_quat

MAT_WIDTH = 0.762 #metres
MAT_HEIGHT = 1.854 #metres
MAT_HALF_WIDTH = MAT_WIDTH/2 
NUMOFTAXELS_X = 64#73 #taxels
NUMOFTAXELS_Y = 27#30 
LOW_TAXEL_THRESH_X = 0
LOW_TAXEL_THRESH_Y = 0
HIGH_TAXEL_THRESH_X = (NUMOFTAXELS_X - 1) 
HIGH_TAXEL_THRESH_Y = (NUMOFTAXELS_Y - 1) 
INTER_SENSOR_DISTANCE = 0.0286#metres

class HeadDetector:
    '''Detects the head of a person sleeping on the autobed'''
    def __init__(self):
        self.mat_size = (NUMOFTAXELS_X, NUMOFTAXELS_Y)
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.tf_listener = tf.TransformListener()
        self.head_center_2d = [0., 0., 0.]
        rospy.sleep(2)
        rospy.Subscriber("/fsascan", FloatArrayBare, self.current_physical_pressure_map_callback)
        self.mat_sampled = False
        while (not self.tf_listener.canTransform('map', 'autobed/head_rest_link', rospy.Time(0))):
            print 'Waiting for head localization in world.'
            rospy.sleep(1)
        #Initialize some constant transforms
        self.head_rest_B_mat = np.eye(4)
        self.head_rest_B_mat[0:3, 0:3] = np.array([[0, -1, 0], [0, 0, 1], [1, 0, 0]])
        self.head_rest_B_mat[0:3, 3] = np.array([0.735, 0, -0.445])
        rospy.sleep(1)
        self.run()

    def current_physical_pressure_map_callback(self, data):
        '''This callback accepts incoming pressure map from 
        the Vista Medical Pressure Mat and sends it out.'''
        p_array = data.data
        p_map_raw = np.reshape(p_array, self.mat_size)
        p_map_hres=ndimage.zoom(p_map_raw, 2, order=1)
        self.pressure_map=p_map_hres
        self.mat_sampled = True

    def detect_head(self):
        '''Computes blobs in pressure map and return top 
        blob as head'''
        #plt.matshow(self.pressure_map)
        #plt.show()
        #Select top 20 pixels of pressure map
        p_map = self.pressure_map[:20,:]
        try:
            blobs = blob_doh(p_map,
                             min_sigma=1, 
                             max_sigma=4, 
                             threshold=30,
                             overlap=0.1) 
        except:
            blobs = np.copy(self.head_center_2d)
            print "Head Not On Mat!"
        if blobs.any():
            self.head_center_2d = blobs[0, :]
        y, x, r = INTER_SENSOR_DISTANCE*self.head_center_2d
        mat_B_head = np.eye(4)
        mat_B_head[0:3, 3] = np.array([x, y, 0.05])
        head_rest_B_head = self.head_rest_B_mat*mat_B_head
        return head_rest_B_head

    def run(self):
        '''Runs pose estimation''' 
        rate = rospy.Rate(5.0)
        while not rospy.is_shutdown():
            if self.mat_sampled:
                self.mat_sampled = False
                head_rest_B_head = self.detect_head()
                self.tf_listener.waitForTransform('map', 'autobed/head_rest_link',\
                                                   rospy.Time(0), rospy.Duration(1))
                (newtrans, newrot) = self.tf_listener.lookupTransform('map', \
                                                                      'autobed/head_rest_link', rospy.Time(0))
                map_B_head_rest = createBMatrix(newtrans, newrot)
                map_B_head = map_B_head_rest*head_rest_B_head
                (out_trans, out_rot) = Bmat_to_pos_quat(map_B_head)
                try:
                    self.tf_broadcaster.sendTransform(out_trans, 
                                                      out_rot,
                                                      rospy.Time(0),
                                                      'user_head_link',
                                                      'map')
                    rate.sleep()
                except:
                    print 'Head TF broadcaster crashed trying to broadcast!'
                    break
            else:
                pass

if __name__ == '__main__':
    rospy.init_node('head_pose_broadcaster', anonymous=True)
    head_blob = HeadDetector()
    #head_blob.run()


