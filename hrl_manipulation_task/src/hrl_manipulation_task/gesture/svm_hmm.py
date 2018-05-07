#!/usr/local/bin/python

import sys
import os
import threading, subprocess 
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn import svm

import rospy

from hmm_model import hmm_model

class svm_hmm:
    def __init__(self, nCommand = 2, nSample = 40, mode=2):
        '''
        nCommand : number of command to model with HMM
        nSample  : number of sample to downsize timeseries to for HMM
        mode: 0 (use the data with scaling)
            : 1 (use the data after preprocessing each sample to make data magnitude-invariant
                Scale afterwards)
            : 2 (same as 1 but pass in magnitude of each sample to SVM)
            : 3 (use both 0, 1 for each command)
        '''
        self.nCommand = nCommand
        self.mode = mode
        self.hmms = []
        self.svm = svm.SVR(kernel='linear')
        self.scaler = preprocessing.StandardScaler()
        for i in xrange(nCommand):
            if self.mode == 3:
                self.hmms.append(hmm_model(sample_size=nSample, mode=0))
                self.hmms.append(hmm_model(sample_size=nSample, mode=1))
            else:
                if self.mode == 0:
                    self.hmms.append(hmm_model(sample_size = nSample, mode=0))
                elif self.mode == 1 or self.mode==2:
                    self.hmms.append(hmm_model(sample_size = nSample, mode=1))

        self.stds  = []
        self.neg_inf = 0

    def fit(self, X, y):
        '''
        fits HMMs given each samples, and then fits SVM given HMM output
        (HMM first fits using samples that corresponds to its command (y),
        then each HMM gives loglikelihoods for each sample, which is used
        as feature for SVM)
        @input: X (dimension x samples x length), y (label of time series)
        '''
        X = np.asarray(X)
        #Sort datas to each label
        indices = [[] for command in xrange(self.nCommand)]
        for i in xrange(len(X[0])):
            indices[y[i]].append(i)
        #fit corresponding HMMs
        for i in xrange(self.nCommand):
            if self.mode == 3:
                X_temp = X.copy()
                self.hmms[2*i].fit(X[:, indices[i], :])
                X = X_temp.copy()
                self.hmms[2*i + 1].fit(X[:, indices[i], :])
                X = X_temp
            else:
                X_temp = X.copy()
                self.hmms[i].fit(X[:, indices[i], :])
                X = X_temp
        lls = np.asarray(self.extract_lls(X))
        svm_X = lls.copy()
        self.svm.fit(svm_X, y)
        svm_X = lls.copy()
        reg_y = self.svm.predict(svm_X)
        for i in xrange(self.nCommand):
            self.stds.append([np.mean(reg_y[indices[i]]), np.std(reg_y[indices[i]])])
        reg_y = np.asarray(reg_y)
        reg_y = reg_y.reshape(-1, 1)

    def extract_lls(self, X):
        '''
        extracts loglikelihoods from multiple HMMs
        @input X (dimension x samples x length)
        @Output lls (samples x lls) returns concatenated loglikelihoods (lls from HMM1 + lls from HMM2 ...)
        '''
        X = np.asarray(X)
        X = X.copy()
        lls = [[] for sample in X[0]]
        for i in xrange(len(self.hmms)):
            X_temp = X.copy()
            ll = self.hmms[i].loglikelihoods(X)
            for j in xrange(len(lls)):
                if type(ll[j]) is type([]):
                    lls[j] = lls[j] + ll[j][-5:] #Concatenate last 5 loglikelhoods
                else:
                    lls[j] = lls[j] + ll[j][-5:].tolist() #concatenate last 5 loglikelihoods
            X = X_temp
        if self.mode == 2:
            for i, dimension in enumerate(X):
                for j, sample in enumerate(dimension):
                    lls[j].append(np.std(sample))
                    lls[j].append(np.mean(sample))
        lls = np.asarray(lls)
        lls[lls == -np.inf] = self.neg_inf
        return lls

        
    def predict(self, X):
        '''
        predict the command
        @input X (dimension x samples x length)
        @output y (samples) returns list of -1 (non match), 0~n, each representing diff command
        '''
        lls = np.asarray(self.extract_lls(X))
        y_list = self.svm.predict(lls)
        y_list2 = np.asarray(y_list).copy().reshape(-1, 1)
        ret = []
        for y in y_list:
            lowest = 99999
            best_command = -1
            for command in xrange(self.nCommand):
                mean = self.stds[command][0]
                std = self.stds[command][1]
                val = abs((y-mean)/std)
                if val < 1.7:
                    if val < lowest:
                        best_command = command
                        lowest = val
            ret.append(best_command)
        return ret
        #return self.svm.predict_log_proba(lls)

    def score(self, X, y):
        '''
        evaluate the prediction of X to actual value y
        @input X, y
        @output percentage (of correct answers)
        '''
        y2 = self.predict(X)
        assert len(y) == len(y2)
        score = 0
        for i in xrange(len(y)):
            if y[i] == y2[i]:
                score = score + 1
            elif y2[i] == -1 and y2[i] not in xrange(self.nCommand):
                score = score + 1
        score = float(score)/ len(y)
        return score

def main():
    rospy.init_node('hmm_plotter')
    plotter = svm_hmm()
    from data_collector import data_collector
    #process_dir(plotter, "/home/hkim/rosbag_test/stay")
    #d = plotter.load_data(f_name="rotate_open_3_22_18.pkl")
    collector = data_collector()
    f_list = ['rotate_open_3_25_18.pkl', 'mouth_open_3_25_18.pkl']
    #f_list = ['rotate_open_3_22_18.pkl', 'rotate_open_3_25_18.pkl', 'mouth_open_3_25_18.pkl']
    y_list = [0, 1]
    d = collector.load_data(f_name=f_list, y=y_list)
    plotter.fit(d['data'], d['y'])
    f_list = ['rotate_open_3_22_18.pkl', 'rotate_open_3_25_18.pkl', 'mouth_open_3_25_18.pkl', 'stay_3_28_18.pkl']
    y_list = [2, 0, 1, 3]
    d = collector.load_data(f_name=f_list, y=y_list)
    lls = plotter.extract_lls(d['data'])
    color=['b', 'r', 'y', 'g']
    for i, pt in enumerate(np.asarray(lls)):
        plt.scatter(pt[-4], pt[-2], c=color[d['y'][i]])
    plt.show()
    d = collector.load_data(f_name=f_list, y=y_list)

if __name__ == '__main__':
    main()
