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

# system
## import rospy, roslib
import os, sys, copy
import random
import socket

# visualization
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec

# util
import numpy as np
import scipy
import hrl_lib.util as ut
from hrl_anomaly_detection.util import *
from hrl_anomaly_detection.util_viz import *
from joblib import Parallel, delayed

# learning
from hrl_anomaly_detection.hmm import learning_hmm_multi_n as hmm
import hrl_anomaly_detection.classifiers.classifier as cf
from sklearn import svm

# data
from mvpa2.datasets.base import Dataset
from hrl_anomaly_detection import data_manager as dm
from hrl_anomaly_detection.feature_extractors import theano_util as tutil


import itertools
colors = itertools.cycle(['r', 'g', 'b', 'm', 'c', 'k', 'y'])
shapes = itertools.cycle(['x','v', 'o', '+'])


def RunAutoEncoder(X, filename, viz=False):

    mlp_features = tutil.load_params(filename)

    # Generate training features
    feature_list = []
    count = 0    
    for idx in xrange(0, len(X[0]), nSingleData):
        count += 1
        test_features = mlp_features( X[:,idx:idx+nSingleData].astype('float32') )
        feature_list.append(test_features)

    # Filter by variances
    feature_list = np.swapaxes(feature_list, 0,1)
    print np.shape(feature_list)
    
    new_feature_list = []
    for i in xrange(len(feature_list)):

        all_std    = np.std(feature_list[i])
        ea_std     = np.std(feature_list[i], axis=0)
        avg_ea_std = np.mean(ea_std)

        print all_std, avg_ea_std, np.shape(ea_std)
        if all_std > 0.2 and avg_ea_std < 0.2:
            new_feature_list.append(feature_list[i])

    if viz:

        n_cols = 2
        n_rows = int(len(feature_list)/2)        
        colors = itertools.cycle(['r', 'g', 'b', 'm', 'c', 'k', 'y'])
        
        #--------------------------------------------------------------
        fig1 = plt.figure(1)
        for i in xrange(len(feature_list)):
            n_col = int(i/n_rows)
            n_row = i%n_rows
            ax = fig1.add_subplot(n_rows,n_cols,i+1)
            color = colors.next()

            for j in xrange(len(feature_list[i])):
                ax.plot(feature_list[i][j,:], ':', c=color)

            ax.set_ylim([0,1])

        fig1.suptitle('Bottleneck features')

        #--------------------------------------------------------------
        fig2 = plt.figure(2)
        for i in xrange(len(new_feature_list)):
            n_col = int(i/n_rows)
            n_row = i%n_rows
            ax = fig2.add_subplot(n_rows,n_cols,i+1)
            color = colors.next()

            for j in xrange(len(new_feature_list[i])):
                ax.plot(new_feature_list[i][j,:], ':', c=color)

            ax.set_ylim([0,1])

        fig2.suptitle('Bottleneck low-variance features')

        plt.show()

    return new_feature_list

def getLikelihoods(trainingData, save_pdf=False):

    print np.shape(trainingData)
    
    # training hmm
    nEmissionDim = len(trainingData)
    detection_param_pkl = os.path.join(processed_data_path, 'hmm_'+task_name+'.pkl')
    scale = 1.0
    cov_mult = [1.0]*(nEmissionDim**2)

    ml  = hmm.learning_hmm_multi_n(nState, nEmissionDim, scale=scale, cluster_type=cluster_type, verbose=False)
    ret = ml.fit(trainingData, cov_mult=cov_mult, ml_pkl=detection_param_pkl, use_pkl=True) # not(renew))
    


    
        

if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    p.add_option('--dataRenew', '--dr', action='store_true', dest='bDataRenew',
                 default=False, help='Renew pickle files.')
    p.add_option('--aeRenew', '--ar', action='store_true', dest='bAERenew',
                 default=False, help='Renew autoencoder parameters. -- not available?')
    p.add_option('--hmmRenew', '--hr', action='store_true', dest='bHMMRenew',
                 default=False, help='Renew HMM parameters.')
    p.add_option('--cfRenew', '--cr', action='store_true', dest='bCFRenew',
                 default=False, help='Renew classifier parameters.')

    p.add_option('--ae', action='store_true', dest='bRunAutoEncoder',
                 default=False, help='RunAutoEncoderOnly.')
    p.add_option('--hmm', action='store_true', dest='bRunHMM',
                 default=False, help='RunAutoEncoder and HMM.')

    p.add_option('--viz', action='store_true', dest='bViz',
                 default=False, help='Visualize data.')
    p.add_option('--savepdf', '--sp', action='store_true', dest='bSavePdf',
                 default=False, help='Save pdf files.')    
    p.add_option('--verbose', '--v', action='store_true', dest='bVerbose',
                 default=False, help='Print out.')

    opt, args = p.parse_args()



    subject_names       = ['gatsbii']
    task                = 'pushing'
    raw_data_path       = os.path.expanduser('~')+'/hrl_file_server/dpark_data/anomaly/RSS2016/'    
    processed_data_path = os.path.expanduser('~')+'/hrl_file_server/dpark_data/anomaly/RSS2016/'+task+'_data'
    save_pkl            = os.path.join(processed_data_path, 'ae_data.pkl')
    ## h5py_file           = os.path.join(processed_data_path, 'test.h5py')
    rf_center           = 'kinEEPos'
    local_range         = 1.25    
    nSet                = 1
    downSampleSize      = 200
    nAugment            = 4
    time_window         = 4
    ## layers              = [256,64,16]

    feature_list = ['relativePose_artag_EE', \
                    'relativePose_artag_artag', \
                    ## 'kinectAudio',\
                    'wristAudio', \
                    'ft', \
                    ## 'pps', \
                    ## 'visionChange', \
                    ## 'fabricSkin', \
                    ]

    AE_filename     = '/home/dpark/catkin_ws/src/hrl-assistive/hrl_anomaly_detection/src/hrl_anomaly_detection/feature_extractors/simple_model.pkl'
                    

    # Load data
    X_normalTrain, X_abnormalTrain, X_normalTest, X_abnormalTest, nSingleData \
      = dm.get_time_window_data(subject_names, task, raw_data_path, processed_data_path, save_pkl, \
                                rf_center, local_range, downSampleSize, time_window, feature_list, \
                                nAugment, renew=opt.bDataRenew)


    if opt.bRunAutoEncoder:
        # Load AE
        normTrainFeatures = RunAutoEncoder(X_normalTrain, AE_filename, opt.bViz)

        ## X_train = np.vstack([X_normalTrain, X_abnormalTrain])
        ## X_test  = np.vstack([X_normalTest, X_abnormalTest])
        ## trainFeatures = RunAutoEncoder(X_train, filename, opt.bViz)
        ## testFeatures  = RunAutoEncoder(X_test, filename, opt.bViz)

    if opt.bRunHMM:
        normTrainFeatures = RunAutoEncoder(X_normalTrain, AE_filename, opt.bViz)
        getLikelihoods(normTrainFeatures, save_pdf=opt.bSavePdf)
        ## trainHMMFeatures = RunHMM(nTrainFeatures, bTrain=False, opt.bViz)

    ## if opt.bRunSVM:
    ##     RunSVM(trainHMMFeatures, bTrain=True, opt.bViz)
    ##     RunSVM()
        
