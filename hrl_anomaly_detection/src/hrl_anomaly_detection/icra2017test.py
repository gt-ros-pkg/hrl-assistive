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
from hrl_anomaly_detection import data_manager as dm
## from hrl_anomaly_detection.scooping_feeding import util as sutil
## import PyKDL
## import sandbox_dpark_darpa_m3.lib.hrl_check_util as hcu
## import sandbox_dpark_darpa_m3.lib.hrl_dh_lib as hdl
## import hrl_lib.circular_buffer as cb
from hrl_anomaly_detection.ICRA2017_params import *
from hrl_anomaly_detection.optimizeParam import *
from hrl_anomaly_detection import util as util

# learning
## from hrl_anomaly_detection.hmm import learning_hmm_multi_n as hmm
from hrl_anomaly_detection.hmm import learning_hmm as hmm
from mvpa2.datasets.base import Dataset
## from sklearn import svm
from joblib import Parallel, delayed
from sklearn import metrics

# private learner
import hrl_anomaly_detection.classifiers.classifier as cf

import itertools
colors = itertools.cycle(['g', 'm', 'c', 'k', 'y','r', 'b', ])
shapes = itertools.cycle(['x','v', 'o', '+'])

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42 


def evaluation_all(subject_names, task_name, raw_data_path, processed_data_path, param_dict,\
                   data_renew=False, save_pdf=False, verbose=False, debug=False,\
                   no_plot=False, delay_plot=True, find_param=False, data_gen=False):

    ## Parameters
    # data
    data_dict  = param_dict['data_param']
    data_renew = data_dict['renew']
    # AE
    AE_dict     = param_dict['AE']
    # HMM
    HMM_dict   = param_dict['HMM']
    nState     = HMM_dict['nState']
    cov        = HMM_dict['cov']
    add_logp_d = HMM_dict.get('add_logp_d', False)
    # SVM
    SVM_dict   = param_dict['SVM']

    # ROC
    ROC_dict = param_dict['ROC']
    
    #------------------------------------------

   
    if os.path.isdir(processed_data_path) is False:
        os.system('mkdir -p '+processed_data_path)

    crossVal_pkl = os.path.join(processed_data_path, 'cv_'+task_name+'.pkl')
    
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
                           ae_data=AE_dict['switch'],\
                           handFeatures=data_dict['handFeatures'], \
                           rawFeatures=AE_dict['rawFeatures'],\
                           cut_data=data_dict['cut_data'], \
                           data_renew=data_renew)

        # TODO: need leave-one-person-out
        # Task-oriented hand-crafted features        
        kFold_list = dm.kFold_data_index2(len(d['successData'][0]), len(d['failureData'][0]), \
                                          data_dict['nNormalFold'], data_dict['nAbnormalFold'] )
        d['kFoldList']   = kFold_list
        ut.save_pickle(d, crossVal_pkl)
    if data_gen: sys.exit()

    #-----------------------------------------------------------------------------------------
    # parameters
    startIdx    = 4
    method_list = ROC_dict['methods'] 
    nPoints     = ROC_dict['nPoints']

    successData = d['successData']
    failureData = d['failureData']
    param_dict2  = d['param_dict']
    if 'timeList' in param_dict2.keys():
        timeList    = param_dict2['timeList'][startIdx:]
    else: timeList = None

    #-----------------------------------------------------------------------------------------
    # Training HMM, and getting classifier training and testing data
    for idx, (normalTrainIdx, abnormalTrainIdx, normalTestIdx, abnormalTestIdx) \
      in enumerate(kFold_list):

        if verbose: print idx, " : training hmm and getting classifier training and testing data"
        modeling_pkl = os.path.join(processed_data_path, 'hmm_'+task_name+'_'+str(idx)+'.pkl')

        if not (os.path.isfile(modeling_pkl) is False or HMM_dict['renew'] or data_renew): continue

        # dim x sample x length
        normalTrainData   = successData[:, normalTrainIdx, :] 
        abnormalTrainData = failureData[:, abnormalTrainIdx, :] 
        normalTestData    = successData[:, normalTestIdx, :] 
        abnormalTestData  = failureData[:, abnormalTestIdx, :] 

        # scaling
        if verbose: print "scaling data"
        normalTrainData   *= HMM_dict['scale']
        abnormalTrainData *= HMM_dict['scale']
        normalTestData    *= HMM_dict['scale']
        abnormalTestData  *= HMM_dict['scale']

        # training hmm
        if verbose: print "start to fit hmm"
        nEmissionDim = len(normalTrainData)
        cov_mult     = [cov]*(nEmissionDim**2)
        nLength      = len(normalTrainData[0][0]) - startIdx

        ml  = hmm.learning_hmm(nState, nEmissionDim, verbose=verbose) 
        if data_dict['handFeatures_noise']:
            ret = ml.fit(normalTrainData+\
                         np.random.normal(0.0, 0.03, np.shape(normalTrainData) )*HMM_dict['scale'], \
                         cov_mult=cov_mult, use_pkl=False)
        else:
            ret = ml.fit(normalTrainData, cov_mult=cov_mult, use_pkl=False)

        if ret == 'Failure': 
            print "-------------------------"
            print "HMM returned failure!!   "
            print "-------------------------"
            sys.exit()
            return (-1,-1,-1,-1)

        #-----------------------------------------------------------------------------------------
        # Classifier training data
        #-----------------------------------------------------------------------------------------
        testDataX = []
        testDataY = []
        for i in xrange(nEmissionDim):
            temp = np.vstack([normalTrainData[i], abnormalTrainData[i]])
            testDataX.append( temp )

        testDataY = np.hstack([ -np.ones(len(normalTrainData[0])), \
                                np.ones(len(abnormalTrainData[0])) ])

        r = Parallel(n_jobs=-1)(delayed(hmm.computeLikelihoods)(i, ml.A, ml.B, ml.pi, ml.F, \
                                                                [ testDataX[j][i] for j in xrange(nEmissionDim) ], \
                                                                ml.nEmissionDim, ml.nState,\
                                                                startIdx=startIdx, \
                                                                bPosterior=True)
                                                                for i in xrange(len(testDataX[0])))
        _, ll_classifier_train_idx, ll_logp, ll_post = zip(*r)

        ll_classifier_train_X, ll_classifier_train_Y = \
          hmm.getHMMinducedFeatures(ll_logp, ll_post, testDataY, c=1.0, add_delta_logp=add_logp_d)

        #-----------------------------------------------------------------------------------------
        # Classifier test data
        #-----------------------------------------------------------------------------------------
        testDataX = []
        testDataY = []
        for i in xrange(nEmissionDim):
            temp = np.vstack([normalTestData[i], abnormalTestData[i]])
            testDataX.append( temp )

        testDataY = np.hstack([ -np.ones(len(normalTestData[0])), \
                                np.ones(len(abnormalTestData[0])) ])

        r = Parallel(n_jobs=-1)(delayed(hmm.computeLikelihoods)(i, ml.A, ml.B, ml.pi, ml.F, \
                                                                [ testDataX[j][i] for j in xrange(nEmissionDim) ], \
                                                                ml.nEmissionDim, ml.nState,\
                                                                startIdx=startIdx, \
                                                                bPosterior=True)
                                                                for i in xrange(len(testDataX[0])))
        _, ll_classifier_test_idx, ll_logp, ll_post = zip(*r)

        # nSample x nLength
        ll_classifier_test_X, ll_classifier_test_Y = \
          hmm.getHMMinducedFeatures(ll_logp, ll_post, testDataY, c=1.0, add_delta_logp=add_logp_d)

        #-----------------------------------------------------------------------------------------
        d = {}
        d['nEmissionDim'] = ml.nEmissionDim
        d['A']            = ml.A 
        d['B']            = ml.B 
        d['pi']           = ml.pi
        d['F']            = ml.F
        d['nState']       = nState
        d['startIdx']     = startIdx
        d['ll_classifier_train_X']  = ll_classifier_train_X
        d['ll_classifier_train_Y']  = ll_classifier_train_Y            
        d['ll_classifier_train_idx']= ll_classifier_train_idx
        d['ll_classifier_test_X']   = ll_classifier_test_X
        d['ll_classifier_test_Y']   = ll_classifier_test_Y            
        d['ll_classifier_test_idx'] = ll_classifier_test_idx
        d['nLength']      = nLength
        ut.save_pickle(d, modeling_pkl)



def evaluation_unexp(subject_names, unexpected_subjects, task_name, raw_data_path, processed_data_path, \
                     param_dict,\
                     data_renew=False, save_pdf=False, verbose=False, debug=False,\
                     no_plot=False, delay_plot=False, find_param=False, data_gen=False):

    ## Parameters
    # data
    data_dict  = param_dict['data_param']
    data_renew = data_dict['renew']
    # AE
    AE_dict     = param_dict['AE']
    # HMM
    HMM_dict   = param_dict['HMM']
    nState     = HMM_dict['nState']
    cov        = HMM_dict['cov']
    add_logp_d = HMM_dict.get('add_logp_d', True)
    # SVM
    SVM_dict   = param_dict['SVM']

    # ROC
    ROC_dict = param_dict['ROC']
    
    #------------------------------------------
    if os.path.isdir(processed_data_path) is False:
        os.system('mkdir -p '+processed_data_path)

    crossVal_pkl = os.path.join(processed_data_path, 'cv_'+task_name+'.pkl')
    
    if os.path.isfile(crossVal_pkl) and data_renew is False:
        print "CV data exists and no renew"
    else:
        '''
        Use augmented data? if nAugment is 0, then aug_successData = successData
        '''        
        d = dm.getDataSet(subject_names, task_name, raw_data_path, \
                           processed_data_path, data_dict['rf_center'], data_dict['local_range'],\
                           downSampleSize=data_dict['downSampleSize'], scale=1.0,\
                           ae_data=AE_dict['switch'],\
                           handFeatures=data_dict['handFeatures'], \
                           rawFeatures=AE_dict['rawFeatures'],\
                           cut_data=data_dict['cut_data'], \
                           data_renew=data_renew)

        # TODO: need leave-one-person-out
        # Task-oriented hand-crafted features        
        kFold_list = dm.kFold_data_index2(len(d['successData'][0]), len(d['failureData'][0]), \
                                          data_dict['nNormalFold'], data_dict['nAbnormalFold'] )
        d['kFoldList']   = kFold_list
        ut.save_pickle(d, crossVal_pkl)
        if data_gen: sys.exit()

    #-----------------------------------------------------------------------------------------
    # parameters
    startIdx    = 4
    method_list = ROC_dict['methods'] 
    nPoints     = ROC_dict['nPoints']

    # Training HMM, and getting classifier training and testing data
    idx = 0
    modeling_pkl = os.path.join(processed_data_path, 'hmm_'+task_name+'_'+str(idx)+'.pkl')
    if not (os.path.isfile(modeling_pkl) is False or HMM_dict['renew'] or data_renew):
        print "learned hmm exists"
    else:
        d = ut.load_pickle(crossVal_pkl)

        successData = d['successData']
        failureData = d['failureData']
        handFeatureParams  = d['param_dict']
        if 'timeList' in handFeatureParams.keys():
            timeList    = handFeatureParams['timeList'][startIdx:]
        else: timeList = None
        
        # dim x sample x length
        normalTrainData   = successData 
        abnormalTrainData = failureData

        # scaling
        if verbose: print "scaling data"
        normalTrainData   *= HMM_dict['scale']
        abnormalTrainData *= HMM_dict['scale']

        # training hmm
        if verbose: print "start to fit hmm"
        nEmissionDim = len(normalTrainData)
        cov_mult     = [cov]*(nEmissionDim**2)
        nLength      = len(normalTrainData[0][0]) - startIdx

        ml  = hmm.learning_hmm(nState, nEmissionDim, verbose=verbose) 
        if data_dict['handFeatures_noise']:
            ret = ml.fit(normalTrainData+\
                         np.random.normal(0.0, 0.03, np.shape(normalTrainData) )*HMM_dict['scale'], \
                         cov_mult=cov_mult, use_pkl=False)
        else:
            ret = ml.fit(normalTrainData, cov_mult=cov_mult, use_pkl=False)

        if ret == 'Failure': 
            print "-------------------------"
            print "HMM returned failure!!   "
            print "-------------------------"
            sys.exit()
            return (-1,-1,-1,-1)

        #-----------------------------------------------------------------------------------------
        # Classifier training data
        #-----------------------------------------------------------------------------------------
        testDataX = []
        testDataY = []
        for i in xrange(nEmissionDim):
            temp = np.vstack([normalTrainData[i], abnormalTrainData[i]])
            testDataX.append( temp )

        testDataY = np.hstack([ -np.ones(len(normalTrainData[0])), \
                                np.ones(len(abnormalTrainData[0])) ])

        r = Parallel(n_jobs=-1)(delayed(hmm.computeLikelihoods)(i, ml.A, ml.B, ml.pi, ml.F, \
                                                                [ testDataX[j][i] for j in xrange(nEmissionDim) ], \
                                                                ml.nEmissionDim, ml.nState,\
                                                                startIdx=startIdx, \
                                                                bPosterior=True)
                                                                for i in xrange(len(testDataX[0])))
        _, ll_classifier_train_idx, ll_logp, ll_post = zip(*r)

        ll_classifier_train_X, ll_classifier_train_Y = \
          hmm.getHMMinducedFeatures(ll_logp, ll_post, testDataY, c=1.0, add_delta_logp=add_logp_d)

        #-----------------------------------------------------------------------------------------
        # Classifier test data
        #-----------------------------------------------------------------------------------------
        fileList = util.getSubjectFileList(raw_data_path, \
                                           unexpected_subjects, \
                                           task_name, no_split=True)                
                                           

        testDataX = dm.getDataList(fileList, data_dict['rf_center'], data_dict['local_range'],\
                                   handFeatureParams,\
                                   downSampleSize = data_dict['downSampleSize'], \
                                   cut_data       = data_dict['cut_data'],\
                                   handFeatures   = data_dict['handFeatures'])

        # scaling and applying offset            
        testDataX = np.array(testDataX)*HMM_dict['scale']
        testDataX = applying_offset(testDataX, normalTrainData, startIdx, nEmissionDim)

        testDataY = []
        for f in fileList:
            if f.find("success")>=0:
                testDataY.append(-1)
            elif f.find("failure")>=0:
                testDataY.append(1)

        r = Parallel(n_jobs=-1)(delayed(hmm.computeLikelihoods)(i, ml.A, ml.B, ml.pi, ml.F, \
                                                                [ testDataX[j][i] for j in xrange(nEmissionDim) ], \
                                                                ml.nEmissionDim, ml.nState,\
                                                                startIdx=startIdx, \
                                                                bPosterior=True)
                                                                for i in xrange(len(testDataX[0])))
        _, ll_classifier_test_idx, ll_logp, ll_post = zip(*r)

        # nSample x nLength
        ll_classifier_test_X, ll_classifier_test_Y = \
          hmm.getHMMinducedFeatures(ll_logp, ll_post, testDataY, c=1.0, add_delta_logp=add_logp_d)

        #-----------------------------------------------------------------------------------------
        d = {}
        d['nEmissionDim'] = ml.nEmissionDim
        d['A']            = ml.A 
        d['B']            = ml.B 
        d['pi']           = ml.pi
        d['F']            = ml.F
        d['nState']       = nState
        d['startIdx']     = startIdx
        d['ll_classifier_train_X']  = ll_classifier_train_X
        d['ll_classifier_train_Y']  = ll_classifier_train_Y            
        d['ll_classifier_train_idx']= ll_classifier_train_idx
        d['ll_classifier_test_X']   = ll_classifier_test_X
        d['ll_classifier_test_Y']   = ll_classifier_test_Y            
        d['ll_classifier_test_idx'] = ll_classifier_test_idx
        d['nLength']      = nLength
        ut.save_pickle(d, modeling_pkl)


    #-----------------------------------------------------------------------------------------
    roc_pkl = os.path.join(processed_data_path, 'roc_'+task_name+'.pkl')
    if os.path.isfile(roc_pkl) is False or HMM_dict['renew']:        
        ROC_data = {}
    else:
        ROC_data = ut.load_pickle(roc_pkl)
        
    for i, method in enumerate(method_list):
        if method not in ROC_data.keys() or method in ROC_dict['update_list']:            
            ROC_data[method] = {}
            ROC_data[method]['complete'] = False 
            ROC_data[method]['tp_l'] = [ [] for j in xrange(nPoints) ]
            ROC_data[method]['fp_l'] = [ [] for j in xrange(nPoints) ]
            ROC_data[method]['tn_l'] = [ [] for j in xrange(nPoints) ]
            ROC_data[method]['fn_l'] = [ [] for j in xrange(nPoints) ]
            ROC_data[method]['delay_l'] = [ [] for j in xrange(nPoints) ]

    # parallelization
    if debug: n_jobs=1
    else: n_jobs=-1
    r = Parallel(n_jobs=n_jobs, verbose=50)(delayed(cf.run_classifiers)( idx, processed_data_path, task_name, \
                                                                 method, ROC_data, \
                                                                 ROC_dict, AE_dict, \
                                                                 SVM_dict, HMM_dict, \
                                                                 startIdx=startIdx, nState=nState,)\
                                                                 for method in method_list )
                                                                  
    l_data = r
    print "finished to run run_classifiers"

    for i in xrange(len(l_data)):
        for j in xrange(nPoints):
            try:
                method = l_data[i].keys()[0]
            except:
                print l_data[i]
                sys.exit()
            if ROC_data[method]['complete'] == True: continue
            ROC_data[method]['tp_l'][j] += l_data[i][method]['tp_l'][j]
            ROC_data[method]['fp_l'][j] += l_data[i][method]['fp_l'][j]
            ROC_data[method]['tn_l'][j] += l_data[i][method]['tn_l'][j]
            ROC_data[method]['fn_l'][j] += l_data[i][method]['fn_l'][j]
            ROC_data[method]['delay_l'][j] += l_data[i][method]['delay_l'][j]

    for i, method in enumerate(method_list):
        ROC_data[method]['complete'] = True

    ut.save_pickle(ROC_data, roc_pkl)
        
    #-----------------------------------------------------------------------------------------
    # ---------------- ROC Visualization ----------------------
    roc_info(method_list, ROC_data, nPoints, delay_plot=delay_plot, no_plot=no_plot, save_pdf=save_pdf, \
             only_tpr=False)


def evaluation_online(subject_names, task_name, raw_data_path, processed_data_path, \
                      param_dict,\
                      data_renew=False, save_pdf=False, verbose=False, debug=False,\
                      no_plot=False, delay_plot=False, find_param=False, data_gen=False):

    ## Parameters
    # data
    data_dict  = param_dict['data_param']
    data_renew = data_dict['renew']
    # AE
    AE_dict     = param_dict['AE']
    # HMM
    HMM_dict   = param_dict['HMM']
    nState     = HMM_dict['nState']
    cov        = HMM_dict['cov']
    add_logp_d = False #HMM_dict.get('add_logp_d', True)
    # SVM
    SVM_dict   = param_dict['SVM']

    # ROC
    ROC_dict = param_dict['ROC']
    
    #------------------------------------------
    if os.path.isdir(processed_data_path) is False:
        os.system('mkdir -p '+processed_data_path)

    '''
    Use augmented data? if nAugment is 0, then aug_successData = successData
    '''
    # Get a data set with a leave-one-person-out
    d = dm.getDataLOPO(subject_names, task_name, raw_data_path, \
                       processed_data_path, data_dict['rf_center'], data_dict['local_range'],\
                       downSampleSize=data_dict['downSampleSize'], scale=1.0,\
                       handFeatures=data_dict['handFeatures'], \
                       cut_data=data_dict['cut_data'], \
                       data_renew=data_renew)
    if data_gen: sys.exit()

    #-----------------------------------------------------------------------------------------
    # parameters
    startIdx    = 4
    method_list = ROC_dict['methods'] 
    nPoints     = ROC_dict['nPoints']
    nPtrainData = 10

    # TODO: need leave-one-person-out
    # Task-oriented hand-crafted features
    kFold_list = []
    for idx in xrange(len(subject_names)):
        idx_list = range(len(subject_names))
        train_idx = idx_list[:idx]+idx_list[idx+1:]
        test_idx  = idx_list[idx:idx+1]        
        kFold_list.append([train_idx, test_idx])
           
        # Training HMM, and getting classifier training and testing data
        modeling_pkl = os.path.join(processed_data_path, 'hmm_'+task_name+'_'+str(idx)+'.pkl')
        if not (os.path.isfile(modeling_pkl) is False or HMM_dict['renew'] or data_renew):
            print "learned hmm exists"
        else:
            # person x dim x sample x length => sample x dim x length
            for i, tidx in enumerate(train_idx):
                if i == 0:
                    normalTrainData = np.swapaxes(d['successDataList'][tidx], 0, 1)
                    abnormalTrainData = np.swapaxes(d['failureDataList'][tidx], 0, 1)
                else:
                    normalTrainData = np.vstack([normalTrainData, np.swapaxes(d['successDataList'][tidx], 0, 1)])
                    abnormalTrainData = np.vstack([abnormalTrainData, np.swapaxes(d['failureDataList'][tidx], 0, 1)])

            for i, tidx in enumerate(test_idx):
                if i == 0:
                    normalTestData = np.swapaxes(d['successDataList'][tidx], 0, 1)
                    abnormalTestData = np.swapaxes(d['failureDataList'][tidx], 0, 1)
                else:
                    normalTestData = np.vstack([normalTestData, np.swapaxes(d['successDataList'][tidx], 0, 1)])
                    abnormalTestData = np.vstack([abnormalTestData, np.swapaxes(d['failureDataList'][tidx], 0, 1)])
            

            normalTrainData = np.swapaxes(normalTrainData, 0, 1)
            abnormalTrainData = np.swapaxes(abnormalTrainData, 0, 1)
            normalTestData = np.swapaxes(normalTestData, 0, 1)
            abnormalTestData = np.swapaxes(abnormalTestData, 0, 1)
            handFeatureParams = d['param_dict']

            # scaling
            if verbose: print "scaling data"
            normalTrainData   *= HMM_dict['scale']
            abnormalTrainData *= HMM_dict['scale']
            normalTestData    *= HMM_dict['scale']
            abnormalTestData  *= HMM_dict['scale']

            # training hmm
            if verbose: print "start to fit hmm"
            nEmissionDim = len(normalTrainData)
            cov_mult     = [cov]*(nEmissionDim**2)
            nLength      = len(normalTrainData[0][0]) - startIdx

            ml  = hmm.learning_hmm(nState, nEmissionDim, verbose=verbose) 
            if data_dict['handFeatures_noise']:
                ret = ml.fit(normalTrainData+\
                             np.random.normal(0.0, 0.03, np.shape(normalTrainData) )*HMM_dict['scale'], \
                             cov_mult=cov_mult, use_pkl=False)
            else:
                ret = ml.fit(normalTrainData, cov_mult=cov_mult, use_pkl=False)

            if ret == 'Failure': 
                print "-------------------------"
                print "HMM returned failure!!   "
                print "-------------------------"
                sys.exit()
                return (-1,-1,-1,-1)

            #-----------------------------------------------------------------------------------------
            # Classifier training data
            #-----------------------------------------------------------------------------------------
            testDataX = []
            testDataY = []
            for i in xrange(nEmissionDim):
                temp = np.vstack([normalTrainData[i], abnormalTrainData[i]])
                testDataX.append( temp )

            testDataY = np.hstack([ -np.ones(len(normalTrainData[0])), \
                                    np.ones(len(abnormalTrainData[0])) ])

            r = Parallel(n_jobs=-1)(delayed(hmm.computeLikelihoods)(i, ml.A, ml.B, ml.pi, ml.F, \
                                                                    [ testDataX[j][i] for j in xrange(nEmissionDim) ], \
                                                                    ml.nEmissionDim, ml.nState,\
                                                                    startIdx=startIdx, \
                                                                    bPosterior=True)
                                                                    for i in xrange(len(testDataX[0])))
            _, ll_classifier_train_idx, ll_logp, ll_post = zip(*r)

            ll_classifier_train_X, ll_classifier_train_Y = \
              hmm.getHMMinducedFeatures(ll_logp, ll_post, testDataY, c=1.0, add_delta_logp=add_logp_d)

            #-----------------------------------------------------------------------------------------
            # Classifier partial train/test data
            #-----------------------------------------------------------------------------------------
            l = range(len(normalTestData[0]))
            random.shuffle(l)
            normalPtrainData = normalTestData[:,l[:nPtrainData],:]

            #-----------------------------------------------------------------------------------------
            [A, B, pi, out_a_num, vec_num, mat_num, u_denom] = ml.get_hmm_object()

            dd = {}
            dd['nEmissionDim'] = ml.nEmissionDim
            dd['F']            = ml.F
            dd['nState']       = nState
            dd['A']            = A 
            dd['B']            = B 
            dd['pi']           = pi
            dd['out_a_num']    = out_a_num
            dd['vec_num']      = vec_num
            dd['mat_num']      = mat_num
            dd['u_denom']      = u_denom
            dd['startIdx']     = startIdx
            dd['ll_classifier_train_X']  = ll_classifier_train_X
            dd['ll_classifier_train_Y']  = ll_classifier_train_Y            
            dd['ll_classifier_train_idx']= ll_classifier_train_idx
            dd['normalPtrainData'] = normalPtrainData
            dd['nLength']      = nLength
            ut.save_pickle(dd, modeling_pkl)


    #-----------------------------------------------------------------------------------------
    roc_pkl = os.path.join(processed_data_path, 'roc_'+task_name+'.pkl')
    if os.path.isfile(roc_pkl) is False or HMM_dict['renew']:        
        ROC_data = {}
    else:
        ROC_data = ut.load_pickle(roc_pkl)


    nTrainOffset = 2
    nTrainTimes  = 3
    for i, method in enumerate(method_list):
        for j in xrange(nTrainTimes+1):
            if method+'_'+str(j) not in ROC_data.keys() or method in ROC_dict['update_list']:            
                data = {}
                data['complete'] = False 
                data['tp_l']     = [ [] for jj in xrange(nPoints) ]
                data['fp_l']     = [ [] for jj in xrange(nPoints) ]
                data['tn_l']     = [ [] for jj in xrange(nPoints) ]
                data['fn_l']     = [ [] for jj in xrange(nPoints) ]
                data['delay_l']  = [ [] for jj in xrange(nPoints) ]
                data['tp_idx_l']  = [ [] for jj in xrange(nPoints) ]
                ROC_data[method+'_'+str(j)] = data

    # Incremental evaluation
    normalData    = np.array([d['successDataList'][i] for i in test_idx])[0]
    abnormalData  = np.array([d['failureDataList'][i] for i in test_idx])[0]        
    
    print "Start the incremental evaluation"
    for idx, (train_idx, test_idx) in enumerate(kFold_list):
        if idx > 1: continue
        data = run_online_classifier(idx, processed_data_path, task_name, HMM_dict,\
                                     nPtrainData, nTrainOffset, nTrainTimes, ROC_data, ROC_dict,\
                                     normalData, abnormalData)

        for i, method in enumerate(method_list):
            for j in xrange(nTrainTimes+1):
                for key in ROC_data[method+'_'+str(j)].keys():
                    if key is 'complete': continue
                    for jj in xrange(nPoints):
                        ROC_data[method+'_'+str(j)][key][jj] += data[method+'_'+str(j)][key][jj]
                
    for i, method in enumerate(method_list):
        for j in xrange(nTrainTimes+1):
            print len(ROC_data[method+'_'+str(j)]), j
            ROC_data[method+'_'+str(j)]['complete'] = True

    ut.save_pickle(ROC_data, roc_pkl)
        
    #-----------------------------------------------------------------------------------------
    # ---------------- ROC Visualization ----------------------
    roc_info(method_list, ROC_data, nPoints, delay_plot=delay_plot, no_plot=no_plot, save_pdf=save_pdf, \
             only_tpr=False)
             

def run_online_classifier(idx, processed_data_path, task_name, HMM_dict, nPtrainData,\
                          nTrainOffset, nTrainTimes, ROC_data, ROC_dict, \
                          normalData, abnormalData, verbose=True):
    '''
    '''
    method_list = ROC_dict['methods'] 
    nPoints     = ROC_dict['nPoints']
    
    ROC_data_cur = {}
    for i, method in enumerate(method_list):
        for j in xrange(nTrainTimes+1):
            data = {}
            data['complete'] = False 
            data['tp_l']     = [ [] for jj in xrange(nPoints) ]
            data['fp_l']     = [ [] for jj in xrange(nPoints) ]
            data['tn_l']     = [ [] for jj in xrange(nPoints) ]
            data['fn_l']     = [ [] for jj in xrange(nPoints) ]
            data['delay_l']  = [ [] for jj in xrange(nPoints) ]
            data['tp_idx_l']  = [ [] for jj in xrange(nPoints) ]
            ROC_data_cur[method+'_'+str(j)] = data

    #
    modeling_pkl = os.path.join(processed_data_path, 'hmm_'+task_name+'_'+str(idx)+'.pkl')
    dd = ut.load_pickle(modeling_pkl)

    nEmissionDim = dd['nEmissionDim']
    nState    = dd['nState']       
    A         = dd['A']      
    B         = dd['B']      
    pi        = dd['pi']     
    out_a_num = dd['out_a_num']
    vec_num   = dd['vec_num']  
    mat_num   = dd['mat_num']  
    u_denom   = dd['u_denom']  
    startIdx  = dd['startIdx']
    nLength   = dd['nLength']
    normalPtrainData = dd['normalPtrainData']

    normalData   = normalData * HMM_dict['scale']
    abnormalData = abnormalData * HMM_dict['scale']

    # random split into two groups
    normalDataIdx   = range(len(normalData[0]))
    abnormalDataIdx = range(len(normalData[0]))
    random.shuffle(normalDataIdx)
    random.shuffle(abnormalDataIdx)

    normalTrainData = normalData[:,:len(normalDataIdx)/2,:]
    normalTestData  = normalData[:,len(normalDataIdx)/2:,:]
    abnormalTrainData = abnormalData[:,:len(abnormalDataIdx)/2,:]
    abnormalTestData  = abnormalData[:,len(abnormalDataIdx)/2:,:]

    testDataX = np.vstack([ np.swapaxes(normalTestData,0,1), np.swapaxes(abnormalTestData,0,1) ])
    testDataX = np.swapaxes(testDataX, 0,1)
    testDataY = np.hstack([-np.ones(len(normalTestData[0])), np.ones(len(abnormalTestData[0])) ])

    if len(normalPtrainData[0]) < nPtrainData:
        print "size of normal train data: ", len(normalPtrainData[0])
        sys.exit()
    if len(normalTrainData[0]) < nTrainOffset*nTrainTimes:
        print "size of normal partial fitting data for hmm: ", len(normalTrainData[0])
        print subject_names[test_idx[0]]
        sys.exit()


    ml = hmm.learning_hmm(nState, nEmissionDim, verbose=verbose) 
    ml.set_hmm_object(A,B,pi,out_a_num,vec_num,mat_num,u_denom)

    for i in xrange(nTrainTimes+1): #-nTrainOffset, len(normalTrainData[0]),nTrainOffset):

        if ROC_data[method+'_'+str(i)]['complete']: continue
        # partial fitting with
        if i > 0:
            print "Run partial fitting with online HMM : ", i
            ml.partial_fit( normalTrainData[:,(i-1)*nTrainOffset:i*nTrainOffset] )
            # Update last 10 samples
            normalPtrainData = np.delete(normalPtrainData, np.s_[:nTrainOffset],1)
            normalPtrainData = np.vstack([ np.swapaxes(normalPtrainData,0,1), \
                                           np.swapaxes(normalTrainData[:,(i-1)*nTrainOffset:i*nTrainOffset],0,1)\
                                           ])
            normalPtrainData = np.swapaxes(normalPtrainData, 0,1)


        # Get classifier training data using last 10 samples
        ll_logp, ll_post, ll_classifier_train_idx = ml.loglikelihoods(normalPtrainData, True, True,\
                                                                      startIdx=startIdx)
        ll_classifier_train_X, ll_classifier_train_Y = \
          hmm.getHMMinducedFeatures(ll_logp, ll_post, -np.ones(len(normalPtrainData[0])), \
                                    c=1.0, add_delta_logp=add_logp_d)

        if method.find('svm')>=0 or method.find('sgd')>=0: remove_fp=True
        else: remove_fp = False
        X_train_org, Y_train_org, idx_train_org = dm.flattenSample(ll_classifier_train_X, \
                                                                   ll_classifier_train_Y, \
                                                                   ll_classifier_train_idx,\
                                                                   remove_fp=remove_fp)

        # -------------------------------------------------------------------------------

        # Test data
        ll_logp, ll_post, ll_classifier_test_idx = ml.loglikelihoods(testDataX, True, True, \
                                                                     startIdx=startIdx)
        ll_classifier_test_X, ll_classifier_test_Y = \
          hmm.getHMMinducedFeatures(ll_logp, ll_post, testDataY, c=1.0, add_delta_logp=add_logp_d)
        X_test = ll_classifier_test_X
        Y_test = ll_classifier_test_Y

        # -------------------------------------------------------------------------------
        # update kmean
        # classification
        dtc = cf.classifier( method=method, nPosteriors=nState, nLength=nLength )
        for j in xrange(nPoints):
            dtc.set_params( **SVM_dict )

            print "Update classifier"
            if method == 'progress_time_cluster' or method == 'progress' or method == 'kmean':
                if method == 'progress_time_cluster':
                    thresholds = ROC_dict['progress_param_range']
                else:
                    thresholds = ROC_dict[method+'_param_range']
                dtc.set_params( ths_mult = thresholds[j] )
                if j==0: ret = dtc.fit(X_train_org, Y_train_org, idx_train_org, parallel=False)
            else:
                print "Not available method = ", method
                sys.exit()

            # evaluate the classifier wrt new never seen data
            tp_l = []
            fp_l = []
            tn_l = []
            fn_l = []
            delay_l = []
            delay_idx = 0
            tp_idx_l = []
            for ii in xrange(len(X_test)):
                if len(Y_test[ii])==0: continue

                if method == 'osvm' or method == 'cssvm' or method == 'hmmosvm':
                    est_y = dtc.predict(X_test[ii], y=np.array(Y_test[ii])*-1.0)
                    est_y = np.array(est_y)* -1.0
                else:
                    est_y    = dtc.predict(X_test[ii], y=Y_test[ii])

                anomaly = False
                for jj in xrange(len(est_y)):
                    if est_y[jj] > 0.0:

                        if ll_classifier_test_idx is not None and Y_test[ii][0]>0:
                            try:
                                delay_idx = ll_classifier_test_idx[ii][jj]
                            except:
                                print "Error!!!!!!!!!!!!!!!!!!"
                                print np.shape(ll_classifier_test_idx), ii, jj
                            delay_l.append(delay_idx)
                        if Y_test[ii][0] > 0:
                            tp_idx_l.append(ii)

                        anomaly = True
                        break        

                if Y_test[ii][0] > 0.0:
                    if anomaly: tp_l.append(1)
                    else: fn_l.append(1)
                elif Y_test[ii][0] <= 0.0:
                    if anomaly: fp_l.append(1)
                    else: tn_l.append(1)

            ROC_data_cur[method+'_'+str(i)]['tp_l'][j] += tp_l
            ROC_data_cur[method+'_'+str(i)]['fp_l'][j] += fp_l
            ROC_data_cur[method+'_'+str(i)]['fn_l'][j] += fn_l
            ROC_data_cur[method+'_'+str(i)]['tn_l'][j] += tn_l
            ROC_data_cur[method+'_'+str(i)]['delay_l'][j] += delay_l
            ROC_data_cur[method+'_'+str(i)]['tp_idx_l'][j] += tp_idx_l

    return ROC_data_cur

def applying_offset(data, normalTrainData, startOffsetSize, nEmissionDim):

    # get offset
    refData = np.reshape( np.mean(normalTrainData[:,:,:startOffsetSize], axis=(1,2)), \
                          (nEmissionDim,1,1) ) # 4,1,1

    curData = np.reshape( np.mean(data[:,:,:startOffsetSize], axis=(1,2)), \
                          (nEmissionDim,1,1) ) # 4,1,1
    offsetData = refData - curData

    for i in xrange(nEmissionDim):
        data[i] = (np.array(data[i]) + offsetData[i][0][0]).tolist()

    return data



if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    p.add_option('--dataRenew', '--dr', action='store_true', dest='bDataRenew',
                 default=False, help='Renew pickle files.')
    p.add_option('--AERenew', '--ar', action='store_true', dest='bAERenew',
                 default=False, help='Renew AE data.')
    p.add_option('--hmmRenew', '--hr', action='store_true', dest='bHMMRenew',
                 default=False, help='Renew HMM parameters.')
    p.add_option('--cfRenew', '--cr', action='store_true', dest='bClassifierRenew',
                 default=False, help='Renew Classifiers.')

    p.add_option('--task', action='store', dest='task', type='string', default='pushing_microwhite',
                 help='type the desired task name')
    p.add_option('--dim', action='store', dest='dim', type=int, default=3,
                 help='type the desired dimension')
    p.add_option('--aeswtch', '--aesw', action='store_true', dest='bAESwitch',
                 default=False, help='Enable AE data.')

    p.add_option('--rawplot', '--rp', action='store_true', dest='bRawDataPlot',
                 default=False, help='Plot raw data.')
    p.add_option('--interplot', '--ip', action='store_true', dest='bInterpDataPlot',
                 default=False, help='Plot raw data.')
    p.add_option('--feature', '--ft', action='store_true', dest='bFeaturePlot',
                 default=False, help='Plot features.')
    p.add_option('--likelihoodplot', '--lp', action='store_true', dest='bLikelihoodPlot',
                 default=False, help='Plot the change of likelihood.')
    p.add_option('--dataselect', '--ds', action='store_true', dest='bDataSelection',
                 default=False, help='Plot data and select it.')
    
    p.add_option('--evaluation_all', '--ea', action='store_true', dest='bEvaluationAll',
                 default=False, help='Evaluate a classifier with cross-validation.')
    p.add_option('--evaluation_unexp', '--eu', action='store_true', dest='bEvaluationUnexpected',
                 default=False, help='Evaluate a classifier with cross-validation.')
    p.add_option('--evaluation_online', '--eo', action='store_true', dest='bOnlineEval',
                 default=False, help='Evaluate a classifier with cross-validation with onlineHMM.')
    p.add_option('--data_generation', action='store_true', dest='bDataGen',
                 default=False, help='Data generation before evaluation.')
                 
    
    p.add_option('--debug', '--dg', action='store_true', dest='bDebug',
                 default=False, help='Set debug mode.')
    p.add_option('--renew', action='store_true', dest='bRenew',
                 default=False, help='Renew pickle files.')
    p.add_option('--savepdf', '--sp', action='store_true', dest='bSavePdf',
                 default=False, help='Save pdf files.')    
    p.add_option('--noplot', '--np', action='store_true', dest='bNoPlot',
                 default=False, help='No Plot.')    
    p.add_option('--noupdate', '--nu', action='store_true', dest='bNoUpdate',
                 default=False, help='No update.')    
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
        subjects = [ 'zack', 'hkim', 'ari', 'park', 'jina', 'linda']
        ## subjects = [ 'zack']
        ## subjects = [ ]
    else:
        print "Selected task name is not available."
        sys.exit()

    raw_data_path, save_data_path, param_dict = getParams(opt.task, opt.bDataRenew, \
                                                          opt.bAERenew, opt.bHMMRenew, opt.dim,\
                                                          rf_center, local_range, \
                                                          bAESwitch=opt.bAESwitch)
        
    
    #---------------------------------------------------------------------------           
    if opt.bRawDataPlot or opt.bInterpDataPlot:
        '''
        Before localization: Raw data plot
        After localization: Raw or interpolated data plot
        '''
        successData = True
        failureData = False
        modality_list   = ['kinematics', 'audio', 'ft', 'vision_artag'] # raw plot

        import hrl_anomaly_detection.data_viz as dv
        dv.data_plot(subjects, opt.task, raw_data_path, save_data_path,\
                  downSampleSize=param_dict['data_param']['downSampleSize'], \
                  local_range=local_range, rf_center=rf_center, \
                  raw_viz=opt.bRawDataPlot, interp_viz=opt.bInterpDataPlot, save_pdf=opt.bSavePdf,\
                  successData=successData, failureData=failureData,\
                  modality_list=modality_list, data_renew=opt.bDataRenew, verbose=opt.bVerbose)

    elif opt.bFeaturePlot:
        success_viz = True
        failure_viz = True
        
        dm.getDataSet(subjects, opt.task, raw_data_path, save_data_path,
                      param_dict['data_param']['rf_center'], param_dict['data_param']['local_range'],\
                      downSampleSize=param_dict['data_param']['downSampleSize'], scale=scale, \
                      success_viz=success_viz, failure_viz=failure_viz,\
                      ae_data=False,\
                      cut_data=param_dict['data_param']['cut_data'],\
                      save_pdf=opt.bSavePdf, solid_color=True,\
                      handFeatures=param_dict['data_param']['handFeatures'], data_renew=opt.bDataRenew)

    elif opt.bLikelihoodPlot:
        import hrl_anomaly_detection.data_viz as dv        
        dv.vizLikelihoods(subjects, opt.task, raw_data_path, save_data_path, param_dict,\
                          decision_boundary_viz=False, \
                          useTrain=False, useNormalTest=True, useAbnormalTest=True,\
                          useTrain_color=False, useNormalTest_color=False, useAbnormalTest_color=False,\
                          hmm_renew=opt.bHMMRenew, data_renew=opt.bDataRenew, save_pdf=opt.bSavePdf,\
                          verbose=opt.bVerbose)
                              
    elif opt.bEvaluationAll or opt.bDataGen:
        if opt.bHMMRenew: param_dict['ROC']['methods'] = ['fixed', 'progress_time_cluster'] 
        if opt.bNoUpdate: param_dict['ROC']['update_list'] = []
                    
        evaluation_all(subjects, opt.task, raw_data_path, save_data_path, param_dict, save_pdf=opt.bSavePdf, \
                       verbose=opt.bVerbose, debug=opt.bDebug, no_plot=opt.bNoPlot, \
                       find_param=False, data_gen=opt.bDataGen)

    elif opt.bEvaluationUnexpected:
        unexp_subjects = ['unexpected', 'unexpected2']
        save_data_path = os.path.expanduser('~')+\
          '/hrl_file_server/dpark_data/anomaly/ICRA2017/'+opt.task+'_data_unexp/'+\
          str(param_dict['data_param']['downSampleSize'])+'_'+str(opt.dim)

        evaluation_unexp(subjects, unexp_subjects, opt.task, raw_data_path, save_data_path, \
                         param_dict, save_pdf=opt.bSavePdf, \
                         verbose=opt.bVerbose, debug=opt.bDebug, no_plot=opt.bNoPlot, \
                         find_param=False, data_gen=opt.bDataGen)

    elif opt.bOnlineEval:
        subjects        = ['zack', 'hkim', 'ari', 'park', 'jina']        
        save_data_path = os.path.expanduser('~')+\
          '/hrl_file_server/dpark_data/anomaly/ICRA2017/'+opt.task+'_data_online/'+\
          str(param_dict['data_param']['downSampleSize'])+'_'+str(opt.dim)

        param_dict['ROC']['methods'] = ['progress_time_cluster']
        param_dict['ROC']['nPoints'] = 10

        evaluation_online(subjects, opt.task, raw_data_path, save_data_path, \
                         param_dict, save_pdf=opt.bSavePdf, \
                         verbose=opt.bVerbose, debug=opt.bDebug, no_plot=opt.bNoPlot, \
                         find_param=False, data_gen=opt.bDataGen)
