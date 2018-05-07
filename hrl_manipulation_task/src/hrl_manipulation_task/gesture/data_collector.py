#!/usr/local/bin/python

import sys
import os
import dlib
import threading, subprocess 
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing

import cv2
from cv_bridge import CvBridge, CvBridgeError
import tf.transformations as tft

import rospy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header, String
from geometry_msgs.msg import PoseStamped

import hrl_lib.util as ut
from hrl_anomaly_detection.hmm import learning_hmm as hmm


class data_collector:
    def __init__(self, sample_size = 40.0, save_loc="/home/hkim/rosbag_test/pkls/"):
        self.time=None
        self.detect=False
        self.detecting=False
        self.t, self.x, self.y = [[]], [[]], [[]]
        self.cnt = 0
        self.save_loc = save_loc
        self.features_sub = rospy.Subscriber("/gesture_control/features", String, self.cb, queue_size=10)
        self.start_sub    = rospy.Subscriber("/gesture_control/start", String, self.start_cb, queue_size=10)

        self.scaler = preprocessing.MinMaxScaler()
        #self.scaler = preprocessing.StandardScaler()
        self.sample_size = sample_size
        self.ion=False
        if self.ion:
            plt.ion()
        #self.run()

    def enable_detect(self):
        self.detect = True

    def disable_detect(self):
        self.detect = False

    def start_cb(self, data):
        if self.detect and data.data=='Start':
            if not self.detecting:
                self.detecting = True
                self.reset_data()

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.detecting and len(self.t) > 0:
                print "detecting"
                if abs(self.t[0][-1] - self.t[0][0]) >= 5.0:
                    if len(self.t[0]) >= self.sample_size:
                        print "finsihed detecting"
                        self.loglikelihoods(plot=True, show=True, color='y--')
                        self.detecting = False
            rate.sleep()

    def cb(self, data):
        features = eval(data.data)
        if self.time is None:
            self.time = features[0]
        t, x, y = features[0] - self.time, features[1], features[2]
        self.t[self.cnt].append(t)
        self.x[self.cnt].append(x)
        self.y[self.cnt].append(y)

    def reset_data(self):
        '''
        resets data stored
        '''
        self.reset_time()
        self.t, self.x, self.y = [[]], [[]], [[]]
        self.cnt = 0

    def reset_time(self):
        self.time = None
        self.t.append([])
        self.x.append([])
        self.y.append([])
        self.cnt = self.cnt + 1

    def grab_datas(self, scale=True, down_sample=True):
        datas = [[], []]
        temp_ts = []
        sample_size = self.sample_size
        for i in xrange(len(self.t)):
            if len(self.t[i])>0:
                if down_sample:
                    per_x = len(self.t[i])/ sample_size
                else:
                    per_x = 1
                temp = self.t[i][-1]
                scaled_t = []
                for t in self.t[i]:
                    scaled_t.append(t/temp * 100.0)
                data = np.asarray([self.x[i], self.y[i]]).transpose()
                if scale:
                    data = self.scaler.transform(data)
                data = data.transpose()
                for j in xrange(len(data[0])):
                    self.x[i][j] = data[0][j]
                    self.y[i][j] = data[1][j]
                print len(self.x[i])
                temp_data = [[], []]
                temp_t = []
                for j in xrange(len(self.t[i])):
                    if int(per_x * j) >=len(self.t[i]) or j >= sample_size:
                        if down_sample:
                            break
                    temp_data[0].append(data[0][int(per_x * j)])
                    temp_data[1].append(data[1][int(per_x * j)])
                    temp_t.append(scaled_t[int(per_x * j)])
                datas[0].append(temp_data[0])
                datas[1].append(temp_data[1])
                temp_ts.append(temp_t)
        datas = np.asarray(datas)
        return datas, temp_ts

    def save_data(self, f_name="data.pkl"):
        '''
        saves the collected data as dimension x samples x length
        '''
        datas, t = self.grab_datas(scale=False)
        d = {}
        d['data'] = datas
        d['t'] = t
        ut.save_pickle(d, os.path.join(self.save_loc, f_name))
        
    def load_data(self, f_name="data.pkl", y=None):
        '''
        load collected data as dimension x samples x length
        '''
        if type(f_name)== type([]):
            y_new = []
            X = None
            t = []
            if y is not None:
                if type(y) == type([]):
                    assert len(y) == len(f_name)
                    for i, file_name in enumerate(f_name):
                        #TODO: change "/home/hkim/rosbag_test/" to a variable called save_loc
                        print os.path.join(self.save_loc, file_name)
                        print os.path.isfile(os.path.join(self.save_loc, file_name))
                        d = ut.load_pickle(os.path.join(self.save_loc, file_name))
                        if type(d['t']) is type([]):
                            t = t + d['t']
                        else:
                            t = t + d['t'].tolist()
                        if X is None:
                            X = [[] for dimension in d['data']]
                        data = d['data']
                        if type(data) is not type([]):
                            data = data.tolist()
                        for j in xrange(len(X)):
                            X[j] = X[j] + data[j]
                        n_samples = np.asarray(d['data']).shape[1]
                        y_new = y_new + [y[i] for sample in xrange(n_samples)]
                    d = {}
                    d['data'] = X
                    d['y'] = y_new
                    d['t'] = t
                    return d
                else:
                    print "set of indices y must be a list"
            else:
                print "no y was given"
        else:
            d = ut.load_pickle(os.path.join(self.save_loc, f_name))
            d['y'] = None
            return d


    def process_dir(self, path):
        if os.path.isdir(path):
            for file_name in os.listdir(path):
                if ".bag" in file_name:
                    bag_path = os.path.join(path, file_name)
                    p = subprocess.Popen("rosbag play " + bag_path, stdin=subprocess.PIPE, shell=True)
                    p.wait()
                    self.reset_time()
        else:
            print "path ", path, " is not a directory"
        
def main():
    rospy.init_node('data_collector')
    collector = data_collector()
    #process_dir(plotter, "/home/hkim/rosbag_test/stay")
    #d = plotter.load_data(f_name="rotate_open_3_22_18.pkl")
    collector.process_dir("/home/hkim/rosbag_test/mouth_open/data/4_11_18")
    collector.save_data(f_name="m_open/m_open_4_11_18_take1.pkl")
    collector.reset_data()
    #plotter.run()

if __name__ == '__main__':
    main()
