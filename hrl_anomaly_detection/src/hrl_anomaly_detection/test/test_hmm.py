#!/usr/bin/env python
#
# Copyright (c) 2014, Georgia Tech Research Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the Georgia Tech Research Corporation nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY GEORGIA TECH RESEARCH CORPORATION ''AS IS'' AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL GEORGIA TECH BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

#  \author Daehyung Park (Healthcare Robotics Lab, Georgia Tech.)

import sys

# visualization
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec

# util
import numpy as np
import random

import ghmm
from hrl_anomaly_detection.hmm import learning_hmm as hmm
from hrl_anomaly_detection.hmm import learning_util as util

def genData():

    # trainingData
    x_t = []
    y_t = []
    for i in xrange(10):
        xx = np.linspace(0, np.pi, 100)
        yy = np.cos(xx)
        r = random.uniform(1.0, 3.0)
        for j in xrange(len(xx)):
            yy[j] += random.uniform(-0.1, 0.1)*r

        x_t.append(xx)
        y_t.append(yy)

    # normalTestData
    x_s = []
    y_s = []
    for i in xrange(10):
        xx = np.linspace(0, np.pi, 100)
        yy = np.cos(xx)
        r = random.uniform(1.0, 3.0)
        for j in xrange(len(xx)):
            yy[j] += random.uniform(-0.1, 0.1)*r

        x_s.append(xx)
        y_s.append(yy)
    
    # abnormalTestData
    x_f = []
    y_f = []
    for i in xrange(10):
        xx = np.linspace(0, np.pi, 100)
        yy = np.cos(xx/2.0)
        r = random.uniform(1.0, 3.0)
        for j in xrange(len(xx)):
            yy[j] += random.uniform(-0.1, 0.1)*r

        x_f.append(xx)
        y_f.append(yy)

    #-------------------------------------

    # Custom data (success)
    xx = np.linspace(0, np.pi, 100)
    yy = np.cos(xx/1.5)
    r = random.uniform(1.0, 3.0)
    for j in xrange(len(xx)):
        yy[j] += random.uniform(-0.1, 0.1)*r

    x_s.append(xx)
    y_s.append(yy)
    
    # Custom data (failure)
    xx = np.linspace(0, np.pi, 100)
    yy = np.cos(xx/1.5)
    r = random.uniform(1.0, 3.0)
    for j in xrange(len(xx)):
        yy[j] += random.uniform(-0.1, 0.1)*r

    x_f.append(xx)
    y_f.append(yy)
    #-------------------------------------

    ## X1 = np.array([x_s + x_f]).flatten()
    ## X2 = np.array([y_s + y_f]).flatten()
    ## X = np.vstack([X1,X2]).T
    ## Y = np.hstack([np.ones(len(np.array(x_s).flatten())), np.zeros(len(np.array(x_f).flatten()))])
    
    trainingData = []
    trainingData.append(x_t)
    trainingData.append(y_t)

    normalTestData = []
    normalTestData.append(x_s)
    normalTestData.append(y_s)
    
    abnormalTestData = []
    abnormalTestData.append(x_f)
    abnormalTestData.append(y_f)

    return np.array(trainingData), np.array(normalTestData), np.array(abnormalTestData)


def raw_plot(fig, x_s, x_f):
    ax = fig.add_subplot(111)
    ax.plot(np.array(x_s).T,np.array(y_s).T, 'bo')
    ax.plot(np.array(x_f).T,np.array(y_f).T, 'rx')
    
    # step size in the mesh
    h = .01

    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
    y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)

    print Z.min(), Z.max()

    ## plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.Blues_r)
    plt.contourf(xx, yy, Z, levels=[Z.min(),0], colors='white')
    plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='orange')
    plt.axis('off')
    

if __name__ == '__main__':

    trainingData, normalTestData, abnormalTestData = genData()

    nState = 10
    nEmissionDim = len(trainingData)
    cov_mult = [10.0]*(nEmissionDim**2)
    detection_param_pkl = 'hmm_test.pkl'
    ths = -1.0
    
    ml  = hmm.learning_hmm(nState, nEmissionDim, verbose=False)
    ret = ml.fit(trainingData, cov_mult=cov_mult, ml_pkl=detection_param_pkl, use_pkl=False) # not(renew))


    fig = plt.figure(figsize=(8, 6))

    gs = gridspec.GridSpec(3, 1, height_ratios=[1,1,2]) 

    ax = fig.add_subplot(gs[0])
    ax.plot(trainingData[0].T, 'b')
    ax.plot(normalTestData[0].T, 'g')
    
    ax = fig.add_subplot(gs[1])
    ax.plot(trainingData[1].T, 'b')
    ax.plot(normalTestData[1].T, 'g')
    
    ax = fig.add_subplot(gs[2])    
    useTrain_color=False
    min_logp = 0.0
    max_logp = 0.0
    log_ll = []
    exp_log_ll = []        
    for i in xrange(len(trainingData[0])):

        log_ll.append([])
        exp_log_ll.append([])
        for j in range(2, len(trainingData[0][i])):

            X = [x[i,:j] for x in trainingData]                
            X_test = util.convert_sequence(X)
            try:
                logp = ml.loglikelihood(X_test)
            except:
                print "Too different input profile that cannot be expressed by emission matrix"
                #return [], 0.0 # error

            log_ll[i].append(logp)

        if min_logp > np.amin(log_ll): min_logp = np.amin(log_ll)
        if max_logp < np.amax(log_ll): max_logp = np.amax(log_ll)

        # disp
        if useTrain_color:
            plt.plot(log_ll[i], label=str(i))
            ## print i, " : ", trainFileList[i], log_ll[i][-1]                
        else:
            plt.plot(log_ll[i], 'b-')

    if useTrain_color: 
        plt.legend(loc=3,prop={'size':16})
    
    
    # abnormal test data
    log_ll = []
    exp_log_ll = []        
    for i in xrange(len(abnormalTestData[0])):

        log_ll.append([])
        exp_log_ll.append([])

        for j in range(2, len(abnormalTestData[0][i])):
            X = [x[i,:j] for x in abnormalTestData]                
            X_test = util.convert_sequence(X)
            try:
                logp = ml.loglikelihood(X_test)
            except:
                print "Too different input profile that cannot be expressed by emission matrix"
                ## return [], 0.0 # error

            log_ll[i].append(logp)
            exp_logp = ml.expLoglikelihood(X, ths)
            exp_log_ll[i].append(exp_logp)

        if min_logp > np.amin(log_ll): min_logp = np.amin(log_ll)
        if max_logp < np.amax(log_ll): max_logp = np.amax(log_ll)

        # disp 
        plt.plot(log_ll[i], 'm-')

        plt.plot(exp_log_ll[i], 'r*-')

    ## class_weight = {0: 1.0,
    ##                 1: 5.0}
    
    ## fig = plt.figure()
    plt.ylim([min_logp, max_logp])
    plt.show()
