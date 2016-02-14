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

# util
import numpy as np
import hrl_lib.util as ut
from hrl_anomaly_detection.util import *
from hrl_anomaly_detection.util_viz import *
from hrl_anomaly_detection import data_manager as dm

import csv

if __name__ == '__main__':

    subject_names       = ['gatsbii']
    task                = 'pushing'
    raw_data_path       = os.path.expanduser('~')+'/hrl_file_server/dpark_data/anomaly/RSS2016/'    
    processed_data_path = os.path.expanduser('~')+'/hrl_file_server/dpark_data/anomaly/RSS2016/'+task+'_data'
    rf_center           = 'kinEEPos'
    local_range         = 0.25    
    nSet                = 1
    downSampleSize      = 200

    feature_list = ['unimodal_audioPower',\
                    'unimodal_kinVel',\
                    'unimodal_ftForce',\
                    ##'unimodal_visionChange',\
                    ##'unimodal_ppsForce',\
                    ##'unimodal_fabricForce',\
                    'crossmodal_targetEEDist', \
                    'crossmodal_targetEEAng']
    ## feature_list = ['artagEE']



    _, successData, failureData,_ = dm.getDataSet(subject_names, task, raw_data_path, \
                                                  processed_data_path, rf_center, local_range,\
                                                  nSet=nSet, \
                                                  downSampleSize=downSampleSize, \
                                                  raw_data=True, \
                                                  feature_list=feature_list, \
                                                  data_renew=True)

    # index selection
    success_idx  = range(len(successData[0]))
    failure_idx  = range(len(failureData[0]))

    nTrain       = int( 0.7*len(success_idx) )    
    train_idx    = random.sample(success_idx, nTrain)
    success_test_idx = [x for x in success_idx if not x in train_idx]
    failure_test_idx = failure_idx

    # data structure: dim x sample x sequence
    trainingData     = successData[:, train_idx, :]
    normalTestData   = successData[:, success_test_idx, :]
    abnormalTestData = failureData[:, failure_test_idx, :]

    print "======================================"
    print "Training data: ", np.shape(trainingData)
    print "Normal test data: ", np.shape(normalTestData)
    print "Abnormal test data: ", np.shape(abnormalTestData)
    print "======================================"

    new_trainingData = []
    for i in xrange(len(trainingData[0])):
        singleSample = []
        for j in xrange(len(trainingData)):
            singleSample.append(trainingData[j][i,:])
            
        new_trainingData.append(singleSample)

    new_testData = []
    for i in xrange(len(normalTestData[0])):
        singleSample = []
        for j in xrange(len(normalTestData)):
            singleSample.append(normalTestData[j][i,:])
            
        new_testData.append(singleSample)

    for i in xrange(len(abnormalTestData[0])):
        singleSample = []
        for j in xrange(len(abnormalTestData)):
            singleSample.append(abnormalTestData[j][i,:])
            
        new_testData.append(singleSample)

        
    ## np.savetxt('test.txt', new_trainingData, delimiter=" ", fmt="%s")
    ## np.savetxt('test.csv', new_trainingData, delimiter=",", fmt="%10.5f")

    import h5py
    f = h5py.File('test.h5py', "w")
    f['trainingData'] = np.array(new_trainingData)
    f['testData']     = np.array(new_testData)
    f.close()
