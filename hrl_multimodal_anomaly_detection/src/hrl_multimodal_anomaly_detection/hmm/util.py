#!/usr/bin/env python

import roslib
roslib.load_manifest('hrl_multimodal_anomaly_detection')

import os, sys
import math
import struct
import numpy as np
import cPickle as pickle
from scipy import interpolate
import matplotlib.pyplot as plt
from sklearn import preprocessing
from mvpa2.datasets.base import Dataset
from plotGenerator import plotGenerator
from learning_hmm_multi_1d import learning_hmm_multi_1d
from learning_hmm_multi_2d import learning_hmm_multi_2d
from learning_hmm_multi_4d import learning_hmm_multi_4d
from joblib import Parallel, delayed # note

import tf

def extrapolateData(data, maxsize):
    return [x if len(x) >= maxsize else x + [x[-1]]*(maxsize-len(x)) for x in data]

def extrapolateAllData(allData, maxsize):
    return [extrapolateData(data, maxsize) for data in allData]

def get_rms(block):
    # RMS amplitude is defined as the square root of the
    # mean over time of the square of the amplitude.
    # so we need to convert this string of bytes into
    # a string of 16-bit samples...

    # we will get one short out for each
    # two chars in the string.
    count = len(block)/2
    structFormat = '%dh' % count
    shorts = struct.unpack(structFormat, block)

    # iterate over the block.
    sum_squares = 0.0
    for sample in shorts:
        # sample is a signed short in +/- 32768.
        # normalize it to 1.0
        n = sample / 32768.0
        sum_squares += n*n

    return math.sqrt(sum_squares / count)

def scaling(X, minVal, maxVal, scale=1.0):
    X = np.array(X)
    return (X - minVal) / (maxVal - minVal) * scale






def displayExpLikelihoods(hmm, trainData, normalTestData, abnormalTestData, ths_mult, save_pdf=False):


    fig = plt.figure()

    n = len(normalTestData[0])
    log_ll = []
    exp_log_ll = []
        
    for i in range(n):
        m = len(normalTestData[0][i])

        log_ll.append([])
        exp_log_ll.append([])
        for j in range(2, m):
            
            X_test = hmm.convert_sequence(normalTestData[0][i][:j], normalTestData[1][i][:j], 
                                          normalTestData[2][i][:j], normalTestData[3][i][:j])
            try:
                logp = hmm.loglikelihood(X_test)
            except:
                print "Too different input profile that cannot be expressed by emission matrix"
                return [], 0.0 # error

            log_ll[i].append(logp)


            exp_logp = hmm.expLikelihoods(normalTestData[0][i][:j], normalTestData[1][i][:j], 
                                          normalTestData[2][i][:j], normalTestData[3][i][:j],
                                          ths_mult)
            exp_log_ll[i].append(exp_logp)
            

        plt.plot(log_ll[i], 'g-')
        plt.plot(exp_log_ll[i], 'r-')


    ## plt.ylim([-500, 500])
        

    if save_pdf == True:
        fig.savefig('test.pdf')
        fig.savefig('test.png')
        os.system('cp test.p* ~/Dropbox/HRL/')
    else:
        plt.show()        


def displayData(hmm, trainData, normalTestData, abnormalTestData, save_pdf=False):

    fig = plt.figure()
    ax1 = plt.subplot(412)
    ax1.set_ylabel('Force\nMagnitude (N)', fontsize=16)
    ax1.set_xticks(np.arange(0, 25, 5))
    # ax1.set_yticks(np.arange(8, 10, 0.5))
    # ax1.set_yticks(np.arange(np.min(self.forcesTrue), np.max(self.forcesTrue), 1.0))
    # ax1.grid()
    ax2 = plt.subplot(411)
    ax2.set_ylabel('Kinematic\nDistance (m)', fontsize=16)
    ax2.set_xticks(np.arange(0, 25, 5))
    # ax2.set_yticks(np.arange(0, 1.0, 0.2))
    # ax2.set_ylim([0, 1.0])
    # ax2.set_yticks(np.arange(np.min(self.distancesTrue), np.max(self.distancesTrue), 0.2))
    # ax2.grid()
    ax3 = plt.subplot(414)
    ax3.set_ylabel('Kinematic\nAngle (rad)', fontsize=16)
    ax3.set_xlabel('Time (sec)', fontsize=16)
    ax3.set_xticks(np.arange(0, 25, 5))
    # ax3.set_yticks(np.arange(0, 1.5, 0.3))
    # ax3.set_ylim([0, 1.5])
    # ax3.set_yticks(np.arange(np.min(self.anglesTrue), np.max(self.anglesTrue), 0.2))
    # ax3.grid()
    ax4 = plt.subplot(413)
    ax4.set_ylabel('Audio\nMagnitude (dec)', fontsize=16)
    ax4.set_xticks(np.arange(0, 25, 5))

    for i in xrange(len(trainData[0])):
        ax1.plot(trainData[0][i], c='b')
        ax2.plot(trainData[1][i], c='b')
        ax3.plot(trainData[2][i], c='b')
        ax4.plot(trainData[3][i], c='b')

    for i in xrange(len(normalTestData[0])):
        ax1.plot(normalTestData[0][i], c='r')
        ax2.plot(normalTestData[1][i], c='r')
        ax3.plot(normalTestData[2][i], c='r')
        ax4.plot(normalTestData[3][i], c='r')
        

    if save_pdf == True:
        fig.savefig('test.pdf')
        fig.savefig('test.png')
        os.system('cp test.p* ~/Dropbox/HRL/')
    else:
        plt.show()        
        
    

def displayLikelihoods(hmm, trainData, normalTestData, abnormalTestData, save_pdf=False):


    fig = plt.figure()

    n = len(trainData[0])
    log_ll = []
    
    for i in range(n):
        m = len(trainData[0][i])

        log_ll.append([])
        for j in range(2, m):

            X_test = hmm.convert_sequence(trainData[0][i][:j], trainData[1][i][:j], 
                                          trainData[2][i][:j], trainData[3][i][:j])
                        
            try:
                logp = hmm.loglikelihood(X_test)
            except:
                print "Too different input profile that cannot be expressed by emission matrix"
                return [], 0.0 # error

            log_ll[i].append(logp)

        plt.plot(log_ll[i], 'b-')
    

    n = len(normalTestData[0])
    log_ll = []

    print "0000000000000000000000"
    print n
    print "0000000000000000000000"
        
    for i in range(n):
        m = len(normalTestData[0][i])

        log_ll.append([])
        for j in range(2, m):

            X_test = hmm.convert_sequence(normalTestData[0][i][:j], normalTestData[1][i][:j], 
                                          normalTestData[2][i][:j], normalTestData[3][i][:j])

            try:
                logp = hmm.loglikelihood(X_test)
            except:
                print "Too different input profile that cannot be expressed by emission matrix"
                return [], 0.0 # error

            log_ll[i].append(logp)

        plt.plot(log_ll[i], 'g-')


    ## n = len(abnormalTestData[0])
    ## log_ll = []
        
    ## for i in range(n):
    ##     m = len(abnormalTestData[0][i])

    ##     log_ll.append([])
    ##     for j in range(2, m):

    ##         X_test = hmm.convert_sequence(abnormalTestData[0][i][:j], abnormalTestData[1][i][:j], 
    ##                                       abnormalTestData[2][i][:j], abnormalTestData[3][i][:j])

    ##         try:
    ##             logp = hmm.loglikelihood(X_test)
    ##         except:
    ##             print "Too different input profile that cannot be expressed by emission matrix"
    ##             return [], 0.0 # error

    ##         log_ll[i].append(logp)

    ##     ax = plt.plot(log_ll[i], 'r-')


    plt.ylim([-500, 500])
        

    if save_pdf == True:
        fig.savefig('test.pdf')
        fig.savefig('test.png')
        os.system('cp test.p* ~/Dropbox/HRL/')
    else:
        plt.show()        
