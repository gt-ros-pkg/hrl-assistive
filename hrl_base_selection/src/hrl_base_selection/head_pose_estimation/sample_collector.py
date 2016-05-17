#!/usr/bin/env python
import rospy, roslib
import sys, os
import random, math
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pkl
from scipy import ndimage
from hrl_msgs.msg import FloatArrayBare


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

class CollectSamples:
    def __init__(self, xmin, ymin):
        self.mat_size = (NUMOFTAXELS_X, NUMOFTAXELS_Y)
        rospy.init_node('samples_hog_detector', anonymous=True)
        rospy.Subscriber("/fsascan", FloatArrayBare, 
                self.current_physical_pressure_map_callback)
        self.mat_sampled = False
        self.xmin = int(xmin)
        self.ymin = int(ymin)
        self.x_range = 36
        self.y_range = 25
        self.zoom_factor = 2
        self.database_path = '/home/yashc/Desktop/dataset/'
        self.pos_sample_path = os.path.join(self.database_path,'pos_samples.p')
        self.neg_sample_path = os.path.join(self.database_path,'neg_samples.p')
        try:
            self.pos_array = pkl.load(open(self.pos_sample_path, "r"))         
        except:
            self.pos_array = []
        try:
            self.neg_array = pkl.load(open(self.neg_sample_path, "r"))         
        except:
            self.neg_array = []
        print "Initialization Done"


    def current_physical_pressure_map_callback(self, data):
        '''This callback accepts incoming pressure map from 
        the Vista Medical Pressure Mat and sends it out.'''
        p_array = data.data
        p_map_raw = np.reshape(p_array, self.mat_size)
        p_map_hres=ndimage.zoom(p_map_raw, 2, order=1)
        self.pressure_map=p_map_hres
        self.mat_sampled = True
        
    def take_n_sample(self):
        '''Capture some negative samples'''
        start_x = random.randint(0, NUMOFTAXELS_Y*self.zoom_factor - self.x_range - 1)
        start_y = random.randint(self.ymin+self.y_range, NUMOFTAXELS_X*self.zoom_factor - self.y_range - 1)
        print start_x, start_y
        n_map = self.pressure_map[start_y:start_y+self.y_range, start_x:start_x+self.x_range]
        #plt.matshow(n_map)
        #plt.show()
        self.neg_array.append(n_map)

    def collect_whole_body_samples(self):
        '''Runs pose estimation''' 
        self.count = 0 
        self.whole_body_path = os.path.join(self.database_path,'whole_body_sitting_test_dat.p')
        try:
            self.whole_body_array = pkl.load(open(self.whole_body_path, "r"))         
        except:
            self.whole_body_array = []
        r = rospy.Rate(5)
        while not rospy.is_shutdown():
            if self.mat_sampled:
                p_map = self.pressure_map
                self.whole_body_array.append(p_map)
                self.count = self.count + 1
                if self.count >= 1:
                    print self.count
                    pkl.dump(self.whole_body_array, open(self.whole_body_path, 'w'))
                    sys.exit()
            else:
                pass
            r.sleep()

    def run(self):
        '''Runs pose estimation''' 
        self.count = 0 
        r = rospy.Rate(5)
        while not rospy.is_shutdown():
            if self.mat_sampled:
                p_map = self.pressure_map[self.ymin:self.ymin+self.y_range, self.xmin:self.xmin+self.x_range]
                self.pos_array.append(p_map)
                self.take_n_sample()
                self.count = self.count + 1
                if self.count >= 200:
                    print self.count
                    pkl.dump(self.pos_array, open(self.pos_sample_path, 'w'))
                    pkl.dump(self.neg_array, open(self.neg_sample_path, 'w'))
                    sys.exit()
            else:
                pass
            r.sleep()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Samples Positive and Negative Pressure Map images for Dalal-Triggs algorithm")
    parser.add_argument("xmin", type=str, 
            help=" The Minimum X-Axis (column) of the pressure map where you want the positive sample to start")
    parser.add_argument("ymin", 
            type=int, help="The Minimum Y-Axis (column) of the pressure map where you want the positive sample to start")
    args = parser.parse_args(rospy.myargv()[1:])
    head_blob = CollectSamples(args.xmin, args.ymin)
    #head_blob.run()
    head_blob.collect_whole_body_samples()

