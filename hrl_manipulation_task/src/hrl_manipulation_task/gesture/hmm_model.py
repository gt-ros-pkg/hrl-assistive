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


class hmm_model:
    def __init__(self, sample_size = 40.0, mode=1):
        '''
        sample_size: size to downsample to. 
        mode: 0 - does not standardize each feature in individual sample (However, it does standardize over entire sample)
              1 - does standardize individual sample. It also standardize each feature over entire sample
        '''
        self.scaler = preprocessing.MinMaxScaler()
        #self.scaler = preprocessing.StandardScaler()
        self.sample_size = sample_size
        self.mode = mode
        self.ion=False
        if self.ion:
            plt.ion()
        #self.run()

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


    def preprocessing(self, X):
        '''
        makes the data magnitude invariant (by standarizing the sample data)
        (e.g. given 1 data sample (a time series of features), it standarizes
        each features to 0 mean unit variance.)
        @input: X (dimension x samples x length)
        @output: X (dimension x samples x length)
        '''
        if self.mode == 0:
            return np.asarray(X)
        elif self.mode == 1:
            datas = np.asarray(X)
            for i in xrange(len(datas)):
                for j in xrange(len(datas[0])):
                    datas[i][j] = datas[i][j] - np.mean(datas[i][j])#datas[i][j][0]
                    datas[i][j] = datas[i][j] / np.std(datas[i][j])
            return datas

    def fit(self, X):
        '''
        set scaler to X and fit HMM to that data
        @input: X (dimension x samples x length)
        @output: True if successful
        '''
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


    def loglikelihoods(self, X, t=None, plot=False, show=False, color=None):
        '''
        return sequence of loglikelihoods for given X datas
        @input: dimension x samples x length, t (sample x length)
        @ouput: samples x length (downsampled)
        '''
        ret = None
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

def main():
    rospy.init_node('hmm_plotter')
    plotter = hmm_model(mode=1)
    from data_collector import data_collector
    #process_dir(plotter, "/home/hkim/rosbag_test/stay")
    #d = plotter.load_data(f_name="rotate_open_3_22_18.pkl")
    collector = data_collector()

    #d = collector.load_data(f_name='m_open/mouth_open_3_25_18.pkl')
    #d = collector.load_data(f_name='left_rotate/rotate_open_3_25_18.pkl')
    d = collector.load_data(f_name='right_rotate/rotate_open_4_11_18_take1.pkl')
    plotter.fit(d['data'])
    d = collector.load_data(f_name='right_rotate/rotate_open_4_11_18_take2.pkl')
    plotter.loglikelihoods(d['data'], d['t'], plot=True, show=False, color='y')
    d = collector.load_data(f_name="left_rotate/rotate_open_3_22_18.pkl")
    plotter.loglikelihoods(d['data'], d['t'], plot=True, show=False, color='g')
    d = collector.load_data(f_name="left_rotate/rotate_open_3_25_18.pkl")
    plotter.loglikelihoods(d['data'], d['t'], plot=True, show=False, color='b')
    d = collector.load_data(f_name="misc/stay_3_28_18.pkl")
    #d = collector.load_data(f_name='right_rotate/rotate_open_4_11_18_take1.pkl')
    #plotter.loglikelihoods(d['data'][:,8:9,:], d['t'][8:9], plot=True, show=True, color='y')
    #print d['data'][:, 8:9, :]
    d = collector.load_data(f_name='m_open/mouth_open_3_25_18.pkl')
    plotter.loglikelihoods(d['data'], d['t'], plot=True, show=True, color='r')

if __name__ == '__main__':
    main()
