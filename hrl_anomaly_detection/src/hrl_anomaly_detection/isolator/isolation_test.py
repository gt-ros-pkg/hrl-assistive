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


# system
import os, sys, copy
import random

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
from hrl_anomaly_detection import data_manager as dm

from hrl_anomaly_detection.ICRA2017_params import *
from hrl_anomaly_detection.optimizeParam import *
from hrl_anomaly_detection import util as util

# learning
from hrl_anomaly_detection.hmm import learning_hmm as hmm
from mvpa2.datasets.base import Dataset
## from sklearn import svm
from joblib import Parallel, delayed
from sklearn import metrics

# private learner
import hrl_anomaly_detection.classifiers.classifier as cf
import hrl_anomaly_detection.data_viz as dv

import itertools
import svmutil as svm
from sklearn import preprocessing
import matplotlib.pyplot as plt
colors = itertools.cycle(['g', 'm', 'c', 'k', 'y','r', 'b', ])
shapes = itertools.cycle(['x','v', 'o', '+'])

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42 


def isolation_test(subject_names, task_name, raw_data_path, processed_data_path, \
                     param_dict, verbose=False, data_renew=False):
    '''
    processed_data_path: please, use this folder as your data location
    '''

    ## Parameters
    # data
    data_dict  = param_dict['data_param']
    
    #-----------------------------------------------------------------------------------------


    if os.path.isdir(processed_data_path) is False:
        os.system('mkdir -p '+processed_data_path)

    crossVal_pkl = os.path.join(processed_data_path, 'cv_'+task_name+'.pkl')
    
    type_sets = []
    type_sets.append([42,43,44,45,46,47,48])
    type_sets.append([21,22,23,24,25,26])
    type_sets.append([12,13,14,27,28,29])
    type_sets.append([0,1,2,3,4,5,6,7,8,9,10,11])

    if os.path.isfile(crossVal_pkl) and data_renew is False:
        print "CV data exists and no renew"
        d = ut.load_pickle(crossVal_pkl)
        kFold_list = d['kFoldList'] 
    else:
        '''
        Use augmented data? if nAugment is 0, then aug_successData = successData
        '''        
        d = dm.getDataSet(subject_names, task_name, raw_data_path, \
                           processed_data_path, data_dict['rf_center'], data_dict['local_range'],\
                           downSampleSize=data_dict['downSampleSize'], scale=1.0,\
                           handFeatures=data_dict['handFeatures'], \
                           cut_data=data_dict['cut_data'], \
                           data_renew=data_renew, max_time=data_dict['max_time'])
        
        new_failureData = [[], [],[],[]]
        for sampleIdx in xrange(0, len(d['failureData'][0])):
            if formatY(sampleIdx, type_sets) is not 0:
                for dimIdx in xrange(0, len(d['failureData'])):
                    new_failureData[dimIdx].append(d['failureData'][0,sampleIdx,:])
        print np.asarray(new_failureData).shape
        d['failureData']=np.asarray(new_failureData)
        
        # Task-oriented hand-crafted features        
        kFold_list = dm.kFold_data_index2(len(d['successData'][0]), len(d['failureData'][0]), \
                                          data_dict['nNormalFold'], data_dict['nAbnormalFold'] )
        d['kFoldList']   = kFold_list
        #formated_X = formatX(d['failureData'][:,:,:])
        #print np.asarray(formated_X).shape
        #for sampleIdx in xrange(0, len(formated_X)):
            
        ut.save_pickle(d, crossVal_pkl)

    #print d['failureData']

    successData = d['successData']
    failureData = d['failureData']
    successFiles = d['successFiles']
    failureFiles = d['failureFiles']
    #print failureData.shape
    
    all_type = []
    for type_set in type_sets:
        all_type.extend(type_set)

    count = 0
    index_d = {}
    for i in xrange(0, 100):
        if i in all_type:
            index_d[str(i)] = count
            count += 1
    new_type_sets = []
    for type_set in type_sets:
        new_type_set = []
        for idx in type_set:
            new_type_set.append(index_d[str(idx)])
        new_type_sets.append(new_type_set)
    type_sets = new_type_sets

    #print "new_type_sets "
    #print type_sets


    # Training svm, and getting classifier training and testing data
    for idx, (normalTrainIdx, abnormalTrainIdx, normalTestIdx, abnormalTestIdx) \
      in enumerate(kFold_list):

        # dim x sample x length
        normalTrainData   = successData[:, normalTrainIdx, :] 
        abnormalTrainData = failureData[:, abnormalTrainIdx, :] 
        normalTestData    = successData[:, normalTestIdx, :] 
        abnormalTestData  = failureData[:, abnormalTestIdx, :]
        
        # code here
        test_classifier=cf.classifier()
        X = []
        y = []        
        mean = [0]* 40
        var = [0]*40
        #for sampleIdx in xrange(0, len(d['failureData'][0])):
        X = []
        for trainIdx in xrange(0, abnormalTrainData.shape[1]):
            curr_type = formatY(abnormalTrainIdx[trainIdx], type_sets)
            if curr_type is 0:
                continue
            formated_X = formatX(abnormalTrainData[:, trainIdx, :])
            #print np.asarray(formated_X).shape
            #print formated_X
            X.extend(formated_X)
            for windowIdx in xrange(0, len(formated_X)):
                y.append(curr_type)
        scaler = preprocessing.StandardScaler()
        X_scaled = scaler.fit_transform(X)
        #print np.asarray(X).shape
        #print np.asarray(y).shape
        #print np.mean(X_scaled, axis=0).shape
        #print np.std(X_scaled, axis=0).shape
        mean = np.mean(X, axis=0)
        var = np.std(X, axis=0)
        print np.mean(X_scaled, axis=0), np.std(X_scaled, axis=0)
        #print X_scaled.shape
        model = svm.svm_train(y, X_scaled.tolist(), '-b 1')
        for testIdx in xrange(0, abnormalTestData.shape[1]):
            X = []
            y = []
            curr_type = formatY(abnormalTestIdx[testIdx], type_sets)
            if curr_type is 0:
                continue
            formated_X = formatX(abnormalTestData[:, testIdx, :], scaler=scaler)
            X.extend(formated_X)
            for windowIdx in xrange(0, len(formated_X)):
                y.append(curr_type)
            prediction =  svm.svm_predict(y, X, model, '-b 1')
            #print prediction
            #print np.asarray(prediction[2])[:,0]
            probability = np.asarray(prediction[2])
            t = np.arange(0., len(prediction[2]), 1)
            plt.figure(1)
            """
            plt.subplot2grid((3,2),(0,0))
            plt.plot(probability[:,0])
            plt.subplot2grid((3,2),(0,1))
            plt.plot(probability[:,1])
            plt.subplot2grid((3,2),(1,0))
            plt.plot(probability[:,2])
            plt.subplot2grid((3,2),(1,1))
            plt.plot(probability[:,3])
            """
            color_list = ['', 'b', 'g', 'r', 'k']
            title_list = ['', 'sound', 'arm', 'spoon miss', 'spoon hit']
            for i in xrange(0, 4):
                color = color_list[int(model.get_labels()[i])]
                title = title_list[int(model.get_labels()[i])]
                plt.subplot2grid((3,2),(int(i/2), i % 2))
                plt.subplot2grid((3,2),(int(i/2), i %2)).set_title(title)
                plt.plot(probability[:,i], color)
            plt.subplot2grid((3,2),(2,0), colspan=2)
            for i in xrange(0,4):
                color = color_list[int(model.get_labels()[i])]
                if model.get_labels()[i] is curr_type:
                    plt.plot(probability[:,i], color + '>')
                else:
                    plt.plot(probability[:,i], color + '-')
            print model.label
            plt.show()
            #raw_input("press enter to continue")
            print " "
            plt.clf
    
def formatX(data, time_window=10, scaler=None):
    X = []
    for idx in xrange(0, data.shape[1]-time_window + 1):
        x = []
        for dimIdx in xrange(0, data.shape[0]):
            temp_x = data[dimIdx][idx:idx+time_window][:]
            x.extend(temp_x)
        X.append(x)
    if scaler is not None:
        X_scaled = scaler.transform(X, copy=True)
        X= X_scaled.tolist()
    return X

def formatY(index, type_sets):
    for setId, type_set in enumerate(type_sets):
        if index in type_set:
            return setId + 1
    return 0

 
if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    p.add_option('--dataRenew', '--dr', action='store_true', dest='bDataRenew',
                 default=False, help='Renew pickle files.')
    p.add_option('--hmmRenew', '--hr', action='store_true', dest='bHMMRenew',
                 default=False, help='Renew HMM parameters.')

    p.add_option('--task', action='store', dest='task', type='string', default='feeding',
                 help='type the desired task name')
    p.add_option('--dim', action='store', dest='dim', type=int, default=4,
                 help='type the desired dimension')

    p.add_option('--rawplot', '--rp', action='store_true', dest='bRawDataPlot',
                 default=False, help='Plot raw data.')
    p.add_option('--interplot', '--ip', action='store_true', dest='bInterpDataPlot',
                 default=False, help='Plot raw data.')
    p.add_option('--feature', '--ft', action='store_true', dest='bFeaturePlot',
                 default=False, help='Plot features.')
    p.add_option('--likelihoodplot', '--lp', action='store_true', dest='bLikelihoodPlot',
                 default=False, help='Plot the change of likelihood.')
                 
    
    p.add_option('--debug', '--dg', action='store_true', dest='bDebug',
                 default=False, help='Set debug mode.')
    p.add_option('--renew', action='store_true', dest='bRenew',
                 default=False, help='Renew pickle files.')
    p.add_option('--savepdf', '--sp', action='store_true', dest='bSavePdf',
                 default=False, help='Save pdf files.')    
    p.add_option('--noplot', '--np', action='store_true', dest='bNoPlot',
                 default=False, help='No Plot.')    
    p.add_option('--verbose', '--v', action='store_true', dest='bVerbose',
                 default=False, help='Print out.')

    
    opt, args = p.parse_args()

    #---------------------------------------------------------------------------           
    # Run evaluation
    #---------------------------------------------------------------------------           
    rf_center     = 'kinEEPos'        
    scale         = 1.0
    # Dectection TEST 
    local_range    = 10.0    

    #---------------------------------------------------------------------------
    if opt.task == 'scooping':
        subjects = ['park', 'test'] #'Henry', 
    #---------------------------------------------------------------------------
    elif opt.task == 'feeding':
        subjects = [ 'unexpected', 'unexpected2' ]
    else:
        print "Selected task name is not available."
        sys.exit()

    raw_data_path, save_data_path, param_dict = getParams(opt.task, opt.bDataRenew, \
                                                          False, False, opt.dim,\
                                                          rf_center, local_range)



    #---------------------------------------------------------------------------
    save_data_path = os.path.expanduser('~')+\
      '/hrl_file_server/dpark_data/anomaly/ICRA2017/'+opt.task+'_data_isolation/'+\
      str(param_dict['data_param']['downSampleSize'])+'_'+str(opt.dim)

    isolation_test(subjects, opt.task, raw_data_path, save_data_path, \
                   param_dict, verbose=opt.bVerbose, data_renew=opt.bDataRenew)
