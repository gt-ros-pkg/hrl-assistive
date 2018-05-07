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


class features_plotter:
    def __init__(self, sample_size = 40.0):
        self.time=None
        self.detect=False
        self.detecting=False
        self.t, self.x, self.y = [[]], [[]], [[]]
        self.cnt = 0
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
        reset currently stored data (does not reset ml)
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
        datas, t = self.grab_datas()
        d = {}
        d['data'] = datas
        d['t'] = t
        ut.save_pickle(d, os.path.join("/home/hkim/rosbag_test/", f_name))
        
    def load_data(self, f_name="data.pkl"):
        d = ut.load_pickle(os.path.join("/home/hkim/rosbag_test/", f_name))
        return d

    def scale(self, X):
        '''
        scale X to current scaler. Changes legnth to specific length by downsampling
        @input: dimension x samples x length
        @output: dimension x samples x length (scaled and downsampled)
        '''
        win_size=3
        datas = np.asarray(X)
        scaled_datas = [[] for d in datas]
        sample_size = self.sample_size
        for i in xrange(len(datas[0])):
            if len(datas[0][i])>0:
                per_x = len(datas[0][i])/ sample_size
                data = np.asarray([d[i] for d in datas]).transpose()
                data = self.scaler.transform(data)
                data = data.transpose()
                for j in xrange(len(datas)):
                    for k in xrange(len(data[0])):
                        datas[j][i][k] = data[j][k]
                temp_data = [[] for d in datas]
                for k in xrange(len(datas)):
                    for j in xrange(len(datas[0][i])):
                        if int(per_x * j) >=len(datas[0][i]) or j >= sample_size:
                            break
                        if int(per_x*j)+win_size >= len(datas[0][i]):
                            data_pt = np.sum(data[k][int(per_x*j)-win_size:int(per_x*j)])/float(win_size)
                        else:
                            data_pt = np.sum(data[k][int(per_x*j):int(per_x*j)+win_size])/float(win_size)
                        #temp_data[k].append(data[k][int(per_x * j)])
                        #print data_pt, data[k][int(per_x * j)]
                        temp_data[k].append(data_pt)
                    scaled_datas[k].append(temp_data[k])
        return scaled_datas


    def preprocessing(self, X=None):
        '''
        makes the data magnitude invariant (by standarizing the sample data)
        (e.g. given 1 data sample (a time series of features), it standarizes
        each features to 0 mean unit variance.)
        @input: X (dimension x samples x length)
        @output: X (dimension x samples x length)
        '''
        if X is None:
            datas, t = self.grab_datas()
            return self.preprocessing(datas)
        else:
            datas = np.asarray(X)
            for i in xrange(len(datas)):
                for j in xrange(len(datas[0])):
                    datas[i][j] = datas[i][j] - np.mean(datas[i][j])#datas[i][j][0]
                    datas[i][j] = datas[i][j] / np.std(datas[i][j])
            return datas

    def fit(self, X=None):
        '''
        set scaler to X and fit HMM to that data
        @input: X (dimension x samples x length)
        @output: True if successful
        '''
        if X is None:
            datas, t = self.grab_datas(scale=False)
            return self.fit(datas)
            """
            flattened_x = [item for sublist in self.x for item in sublist]
            flattened_y = [item for sublist in self.y for item in sublist]
            flattened_x = np.asarray(flattened_x).transpose()
            flattened_y = np.asarray(flattened_y).transpose()
            flattened_data= np.asarray([flattened_x, flattened_y]).transpose()
            self.scaler.fit_transform(flattened_data)
            datas, t = self.grab_datas()
            """
        else:
            X = self.preprocessing(X)
            datas = np.asarray(X)
            flattened = []
            for i in xrange(len(datas)):
                flattened.append([item for sublist in datas[i] for item in sublist])
            flattened_data = np.asarray(flattened).transpose()
            print flattened_data.shape
            self.scaler.fit_transform(flattened_data)
            datas = self.scale(datas)
        datas = np.asarray(datas)
        self.ml = hmm.learning_hmm(nState=15, nEmissionDim=2)
        success = self.ml.fit(datas)
        return success != 'Failure'


    def loglikelihoods(self, X=None, t=None, plot=False, show=False, color=None):
        '''
        return sequence of loglikelihoods for given X datas
        @input: dimension x samples x length, t (sample x length)
        @ouput: samples x length (downsampled)
        '''
        ret = None
        if X is None:
            datas, t = self.grab_datas(scale=False)
            if datas is not None:
                return self.loglikelihoods(X=datas, t=t, plot=plot, show=show, color=color)
            else:
                return None
        else:
            X = self.preprocessing(X)
            datas = self.scale(X)
        ret = self.ml.loglikelihoods(datas)
        if plot and (X is None or t is not None):
            for i in xrange(len(datas[0])):
                if color is None:
                    plt.subplot(311)
                    plt.plot(t[i], datas[0][i])
                    plt.subplot(312)
                    plt.plot(t[i], datas[1][i])
                    plt.subplot(313)
                    plt.plot(xrange(len(ret[i])), ret[i])
                else:
                    plt.subplot(311)
                    plt.plot(t[i], datas[0][i], color)
                    plt.subplot(312)
                    plt.plot(t[i], datas[1][i], color)
                    plt.subplot(313)
                    plt.plot(xrange(len(ret[i])), ret[i], color)
            if show:
                if self.ion:
                    plt.draw()
                else:
                    plt.show()
        return ret

def process_dir(plotter, path):
    if os.path.isdir(path):
        for file_name in os.listdir(path):
            if ".bag" in file_name:
                bag_path = os.path.join(path, file_name)
                p = subprocess.Popen("rosbag play " + bag_path, stdin=subprocess.PIPE, shell=True)
                p.wait()
                plotter.reset_time()
    else:
        print "path ", path, " is not a directory"
        
def main():
    rospy.init_node('feature_plotter')
    plotter = features_plotter()
    #process_dir(plotter, "/home/hkim/rosbag_test/stay")
    #d = plotter.load_data(f_name="rotate_open_3_22_18.pkl")
    process_dir(plotter, "/home/hkim/rosbag_test/rotate_open/left/data/3_22_18")
    plotter.fit()
    plotter.reset_data()
    process_dir(plotter, "/home/hkim/rosbag_test/rotate_open/left/data/3_22_18")
    plotter.loglikelihoods(plot=True)
    plotter.reset_data()
    process_dir(plotter, "/home/hkim/rosbag_test/rotate_open/left/data/3_25_18")
    plotter.loglikelihoods(plot=True)
    plotter.reset_data()
    process_dir(plotter, "/home/hkim/rosbag_test/mouth_open/data/3_25_18")
    plotter.loglikelihoods(plot=True, show=True)
    sys.exit()


    d = plotter.load_data(f_name='rotate_open_3_22_18.pkl')
    #d = plotter.load_data(f_name='mouth_open_3_25_18.pkl')
    plotter.fit(d['data'])
    d = plotter.load_data(f_name="rotate_open_3_22_18.pkl")
    plotter.loglikelihoods(d['data'], d['t'], plot=True, show=False, color='g')
    d = plotter.load_data(f_name="rotate_open_3_25_18.pkl")
    plotter.loglikelihoods(d['data'], d['t'], plot=True, show=False, color='b')
    d = plotter.load_data(f_name="stay_3_28_18.pkl")
    plotter.loglikelihoods(d['data'], d['t'], plot=True, show=False, color='y')
    d = plotter.load_data(f_name='mouth_open_3_25_18.pkl')
    plotter.loglikelihoods(d['data'], d['t'], plot=True, show=True, color='r')
    plotter.reset_data()
    """
    #using time as a feature as well
    d = plotter.load_data(f_name="rotate_open_3_22_18.pkl")
    a = d['data'].tolist()
    a.append(d['t'])
    a= np.asarray(a)
    plotter.fit(a)
    d = plotter.load_data(f_name="rotate_open_3_22_18.pkl")
    a = d['data'].tolist()
    a.append(d['t'])
    a= np.asarray(a)
    plotter.loglikelihoods(a, d['t'], plot=True, show=False, color='g')
    plotter.reset_data()
    d = plotter.load_data(f_name="rotate_open_3_25_18.pkl")
    a = d['data'].tolist()
    a.append(d['t'])
    a= np.asarray(a)
    plotter.loglikelihoods(a, d['t'], plot=True, show=False, color='b')
    d = plotter.load_data(f_name='mouth_open_3_25_18.pkl')
    #d = plotter.load_data(f_name="stay_3_28_18.pkl")
    a = d['data'].tolist()
    a.append(d['t'])
    a= np.asarray(a)
    print a.shape
    plotter.loglikelihoods(a, d['t'], plot=True, show=True, color='r')
    plotter.reset_data()
    """
    #plotter.run()

if __name__ == '__main__':
    main()
