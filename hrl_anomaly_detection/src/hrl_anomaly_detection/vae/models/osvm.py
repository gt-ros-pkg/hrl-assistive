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

# system & utils
import os, sys, copy, random
import numpy
import numpy as np
import scipy
np.random.seed(3334)

from hrl_anomaly_detection.vae import util as vutil
from hrl_anomaly_detection.vae import keras_util as ku

from sklearn  import svm

def osvm_detector(trainData, testData, weights_file=None, batch_size=32, nb_epoch=500, \
                  patience=20, fine_tuning=False, save_weights_file=None, \
                  noise_mag=0.0, timesteps=4, sam_epoch=1, \
                  renew=False, plot=True, trainable=None, **kwargs):

    x_train = trainData[0]
    x_test = testData[0]

    nSample = len(x_train)
    input_dim = len(x_train[0][0])
    length = len(x_train[0])

    x_train, y_train = create_dataset(x_train, timesteps, 0)
    x_test, y_test   = create_dataset(x_test, timesteps, 0)

    x_train = x_train.reshape((-1, input_dim*timesteps))
    x_test  = x_test.reshape((-1, input_dim*timesteps))
   

    clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    clf.fit(x_train)

    return clf


def anomaly_detection(clf, vae_mean, vae_logvar, enc_z_mean, enc_z_logvar, generator,\
                      normalTrainData, normalValData, \
                      normalTestData, abnormalTestData,
                      ad_method, method, window_size, alpha,
                      ths_l=None, save_pkl=None, plot=False, renew=False, **kwargs):

    nSample = len(normalTrainData)
    input_dim = np.shape(normalTrainData)[-1]
    length = np.shape(normalTrainData)[1]
    timesteps = window_size


    if os.path.isfile(save_pkl) and renew is False :
        d = ut.load_pickle(save_pickle)
        fp_ll = d['fp_ll']
        tn_ll = d['tn_ll']
        tp_ll = d['tp_ll']
        fn_ll = d['fn_ll']  
    else:

        x_train, y_train   = create_dataset(normalTrainData, timesteps, 0)
        x_test_n, y_test_n = create_dataset(normalTestData, timesteps, 0)
        x_test_a, y_test_a = create_dataset(abnormalTestData, timesteps, 0)

        x_train = x_train.reshape((-1, input_dim*timesteps))
        #x_test_n  = x_test_n.reshape((-1, input_dim*timesteps))
        #x_test_a  = x_test_a.reshape((-1, input_dim*timesteps))

        fp_ll = []; tn_ll = []
        tp_ll = []; fn_ll = []
        for ths in ths_l:

            clf = svm.OneClassSVM(nu=ths, kernel="rbf", gamma=0.1)
            clf.fit(x_train)

            fp_l=[]; tn_l=[]
            for x in x_test_n: # per sample

                xx = x.reshape((-1, input_dim*timesteps))
                yy = clf.predict(xx)

                if any( label>0 for label in yy): fp_l.append(1)
                else:                             tn_l.append(1)

            tp_l=[]; fn_l=[]
            for x in x_test_a:
                xx = x.reshape((-1, input_dim*timesteps))
                yy = clf.predict(xx)
                if any( label>0 for label in yy): tp_l.append(1)
                else:                             fn_l.append(1) 
                
        
            fp_ll.append(fp_l)
            tn_ll.append(tn_l)
            tp_ll.append(tp_l)
            fn_ll.append(fn_l)

        d = {}
        d['fp_ll'] = fp_ll
        d['tn_ll'] = tn_ll
        d['tp_ll'] = tp_ll
        d['fn_ll'] = fn_ll
        ut.save_pickle(d, save_pkl) 

    #--------------------------------------------------------------------
    tpr_l = []
    fpr_l = []
    for i in xrange(len(ths_l)):
        tpr_l.append( float(np.sum(tp_ll[i]))/float(np.sum(tp_ll[i])+np.sum(fn_ll[i]))*100.0 )
        fpr_l.append( float(np.sum(fp_ll[i]))/float(np.sum(fp_ll[i])+np.sum(tn_ll[i]))*100.0 )

    from sklearn import metrics
    roc = metrics.auc(fpr_l, tpr_l, True)
    return tp_ll, tn_ll, fp_ll, fn_ll, roc    

    

    

def create_dataset(dataset, window_size=5, step=5):
    '''
    Input: dataset= sample x timesteps x dim
    Output: dataX = sample x timesteps? x window x dim
    '''
    
    dataX, dataY = [], []
    for i in xrange(len(dataset)):
        x = []
        y = []
        for j in range(len(dataset[i])-step-window_size):
            x.append(dataset[i,j:(j+window_size), :].tolist())
            y.append(dataset[i,j+step:(j+step+window_size), :].tolist())
        dataX.append(x)
        dataY.append(y)
    return numpy.array(dataX), numpy.array(dataY)


