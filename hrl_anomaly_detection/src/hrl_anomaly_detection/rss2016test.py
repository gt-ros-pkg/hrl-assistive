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

# learning
## from hrl_anomaly_detection.hmm import learning_hmm_multi_n as hmm
from hrl_anomaly_detection.hmm import learning_hmm as hmm
from mvpa2.datasets.base import Dataset
from sklearn import svm
from joblib import Parallel, delayed

# private learner
import hrl_anomaly_detection.classifiers.classifier as cf

import itertools
colors = itertools.cycle(['r', 'g', 'b', 'm', 'c', 'k', 'y'])
shapes = itertools.cycle(['x','v', 'o', '+'])

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42 
   
def likelihoodOfSequences(subject_names, task_name, raw_data_path, processed_data_path, param_dict,\
                          decision_boundary_viz=False, \
                          useTrain=True, useNormalTest=True, useAbnormalTest=False,\
                          useTrain_color=False, useNormalTest_color=False, useAbnormalTest_color=False,\
                          data_renew=False, hmm_renew=False, save_pdf=False, verbose=False):

    ## Parameters
    # data
    data_dict  = param_dict['data_param']
    # AE
    AE_dict     = param_dict['AE']
    # HMM
    HMM_dict = param_dict['HMM']
    nState   = HMM_dict['nState']
    cov      = HMM_dict['cov']
    # SVM
    
    #------------------------------------------

    if AE_dict['switch']:
        
        AE_proc_data = os.path.join(processed_data_path, 'ae_processed_data_0.pkl')
        d = ut.load_pickle(AE_proc_data)
        if AE_dict['filter']:
            # Bottle features with variance filtering
            successData = d['normTrainDataFiltered']
            failureData = d['abnormTrainDataFiltered']
        else:
            # Bottle features without filtering
            successData = d['normTrainData']
            failureData = d['abnormTrainData']

        ## print np.shape(d['normTrainDataFiltered']), np.shape(d['normTrainData'])
        ## print d.keys()

        if AE_dict['add_option'] is 'featureToBottleneck':
            print "add feature is not implemented..."
            ##     dd = dm.getDataSet(subject_names, task_name, raw_data_path, \
            ##                        processed_data_path, data_dict['rf_center'], \
            ##                        data_dict['local_range'],\
            ##                        downSampleSize=data_dict['downSampleSize'], \
            ##                        scale=1.0,\
            ##                        ae_data=AE_dict['switch'],\
            ##                        data_ext=data_dict['lowVarDataRemv'],\
            ##                        nAugment=data_dict['nAugment'],\
            ##                        feature_list=data_dict['feature_list'], \
            ##                        cut_data=data_dict['cut_data'],\
            ##                        data_renew=data_dict['renew'])

            ##     print dd.keys()

            ##     org_successData = dd.get('successData', dd['trainingData'])
            ##     org_failureData = dd.get('failureData', dd['abnormalTestData'])
            ##     org_aug_successData = dd['successData_augmented']
            ##     org_aug_failureData = dd['failureData_augmented']

            ##     print np.shape(org_successData), np.shape(org_aug_successData)
            ##     sys.exit()

        successData *= HMM_dict['scale']
        failureData *= HMM_dict['scale']
        
    else:
        dd = dm.getDataSet(subject_names, task_name, raw_data_path, \
                           processed_data_path, data_dict['rf_center'], \
                           data_dict['local_range'],\
                           downSampleSize=data_dict['downSampleSize'], \
                           scale=1.0,\
                           ae_data=False,\
                           data_ext=data_dict['lowVarDataRemv'],\
                           nAugment=data_dict['nAugment'],\
                           feature_list=data_dict['feature_list'], \
                           cut_data=data_dict['cut_data'],\
                           data_renew=data_dict['renew'])
                           
        successData = dd['successData'] * HMM_dict['scale']
        failureData = dd['failureData'] * HMM_dict['scale']
                           

    normalTestData = None                                    
    print "======================================"
    print "Success data: ", np.shape(successData)
    ## print "Normal test data: ", np.shape(normalTestData)
    print "Failure data: ", np.shape(failureData)
    print "======================================"

    # training hmm
    nEmissionDim = len(successData)
    hmm_param_pkl = os.path.join(processed_data_path, 'hmm_'+task_name+'.pkl')    
    cov_mult = [cov]*(nEmissionDim**2)

    # generative model
    ml  = hmm.learning_hmm(nState, nEmissionDim, verbose=False)
    ret = ml.fit(successData, cov_mult=cov_mult, ml_pkl=hmm_param_pkl, use_pkl=False) # not(renew))
    ## ths = threshold
    startIdx = 4
        
    if ret == 'Failure': 
        print "-------------------------"
        print "HMM returned failure!!   "
        print "-------------------------"
        return (-1,-1,-1,-1)

    if decision_boundary_viz:
        testDataX = np.vstack([np.swapaxes(successData, 0, 1), np.swapaxes(failureData, 0, 1)])
        testDataX = np.swapaxes(testDataX, 0, 1)
        testDataY = np.hstack([ -np.ones(len(successData[0])), \
                                np.ones(len(failureData[0])) ])

        ## testDataX = successData
        ## testDataY = -np.ones(len(successData[0]))

        r = Parallel(n_jobs=-1)(delayed(hmm.computeLikelihoods)(i, ml.A, ml.B, ml.pi, ml.F, \
                                                                [testDataX[j][i] for j in \
                                                                 xrange(nEmissionDim)], \
                                                                ml.nEmissionDim, ml.nState,\
                                                                startIdx=startIdx, \
                                                                bPosterior=True)
                                                                for i in xrange(len(testDataX[0])))
        _, ll_classifier_train_idx, ll_logp, ll_post = zip(*r)

        ll_classifier_train_X = []
        ll_classifier_train_Y = []
        for i in xrange(len(ll_logp)):
            l_X = []
            l_Y = []
            for j in xrange(len(ll_logp[i])):        
                l_X.append( [ll_logp[i][j]] + ll_post[i][j].tolist() )

                if testDataY[i] > 0.0: l_Y.append(1)
                else: l_Y.append(-1)

            ll_classifier_train_X.append(l_X)
            ll_classifier_train_Y.append(l_Y)

        # flatten the data
        X_train_org = []
        Y_train_org = []
        idx_train_org = []
        for i in xrange(len(ll_classifier_train_X)):
            for j in xrange(len(ll_classifier_train_X[i])):
                X_train_org.append(ll_classifier_train_X[i][j])
                Y_train_org.append(ll_classifier_train_Y[i][j])
                idx_train_org.append(ll_classifier_train_idx[i][j])

        # discriminative classifier
        dtc = cf.classifier( method='progress_time_cluster', nPosteriors=nState, \
                             nLength=len(successData[0,0]), ths_mult=-1.0 )
        dtc.fit(X_train_org, Y_train_org, idx_train_org)

    
    fig = plt.figure()
    min_logp = 0.0
    max_logp = 0.0
        
    # training data
    if useTrain:

        log_ll = []
        exp_log_ll = []        
        for i in xrange(len(successData[0])):

            log_ll.append([])
            exp_log_ll.append([])
            for j in range(startIdx, len(successData[0][i])):

                X = [x[i,:j] for x in successData]
                logp = ml.loglikelihood(X)
                log_ll[i].append(logp)

                if decision_boundary_viz:
                    if j>=np.shape(ll_logp)[1]: continue
                    l_X = [ll_logp[i][j]] + ll_post[i][j].tolist()
                    
                    exp_logp = dtc.predict(l_X) + ll_logp[i][j]
                    exp_log_ll[i].append(exp_logp)


            if min_logp > np.amin(log_ll): min_logp = np.amin(log_ll)
            if max_logp < np.amax(log_ll): max_logp = np.amax(log_ll)
                
            # disp
            if useTrain_color: plt.plot(log_ll[i], label=str(i))
            else: plt.plot(log_ll[i], 'b-')

            ## # temp
            ## if show_plot:
            ##     plt.plot(log_ll[i], 'b-', lw=3.0)
            ##     plt.plot(exp_log_ll[i], 'm-')                            
            ##     plt.show()
            ##     fig = plt.figure()

        if useTrain_color: 
            plt.legend(loc=3,prop={'size':16})
            
        plt.plot(log_ll[i], 'k-', lw=3.0)
        if decision_boundary_viz:
            plt.plot(exp_log_ll[i], 'm-', lw=3.0)            
                                             
    # normal test data
    ## if useNormalTest and False:

    ##     log_ll = []
    ##     ## exp_log_ll = []        
    ##     for i in xrange(len(normalTestData[0])):

    ##         log_ll.append([])
    ##         ## exp_log_ll.append([])

    ##         for j in range(2, len(normalTestData[0][i])):
    ##             X = [x[i,:j] for x in normalTestData]                

    ##             logp = ml.loglikelihood(X)
    ##             log_ll[i].append(logp)

    ##             ## exp_logp, logp = ml.expLoglikelihood(X, ths, bLoglikelihood=True)
    ##             ## log_ll[i].append(logp)
    ##             ## exp_log_ll[i].append(exp_logp)

    ##         if min_logp > np.amin(log_ll): min_logp = np.amin(log_ll)
    ##         if max_logp < np.amax(log_ll): max_logp = np.amax(log_ll)

    ##         # disp 
    ##         if useNormalTest_color: plt.plot(log_ll[i], label=str(i))
    ##         else: plt.plot(log_ll[i], 'g-')

    ##         ## plt.plot(exp_log_ll[i], 'r*-')

    ##     if useNormalTest_color: 
    ##         plt.legend(loc=3,prop={'size':16})

    # abnormal test data
    if useAbnormalTest:
        log_ll = []
        ## exp_log_ll = []        
        for i in xrange(len(failureData[0])):

            log_ll.append([])
            ## exp_log_ll.append([])

            for j in range(startIdx, len(failureData[0][i])):
                X = [x[i,:j] for x in failureData]                
                try:
                    logp = ml.loglikelihood(X)
                except:
                    print "Too different input profile that cannot be expressed by emission matrix"
                    return [], 0.0 # error

                log_ll[i].append(logp)

            # disp 
            plt.plot(log_ll[i], 'r-')
            ## plt.plot(exp_log_ll[i], 'r*-')


    plt.ylim([min_logp, max_logp])
    if save_pdf == True:
        fig.savefig('test.pdf')
        fig.savefig('test.png')
        os.system('cp test.p* ~/Dropbox/HRL/')
    else:
        plt.show()        

    return


def stateLikelihoodPlot(subject_names, task_name, raw_data_path, processed_data_path, rf_center, \
                        local_range, \
                        downSampleSize=200, \
                        feature_list=['crossmodal_targetEEDist'], \
                        nState=10, threshold=-1.0, smooth=False, cluster_type='time', \
                        classifier_type='time', \
                        useTrain=True, useNormalTest=True, useAbnormalTest=False,\
                        useTrain_color=False, useNormalTest_color=False, useAbnormalTest_color=False,\
                        hmm_renew=False, data_renew=False, save_pdf=False, show_plot=True):

    dd = dm.getDataSet(subject_names, task_name, raw_data_path, \
                       processed_data_path, rf_center, local_range,\
                       downSampleSize=downSampleSize, \
                       feature_list=feature_list, \
                       data_renew=data_renew)
    successData = data_dict['successData']
    failureData = data_dict['failureData']
                       
                       

    # index selection
    success_idx  = range(len(successData[0]))
    failure_idx  = range(len(failureData[0]))
    
    nTrain       = int( 0.7*len(success_idx) )    
    train_idx    = random.sample(success_idx, nTrain)
    success_test_idx = [x for x in success_idx if not x in train_idx]
    failure_test_idx = failure_idx

    # data structure: dim x nData x sequence
    trainingData     = successData[:, train_idx, :]
    normalTestData   = successData[:, success_test_idx, :]
    abnormalTestData = failureData[:, failure_test_idx, :]
    
    print "======================================"
    print "Training data: ", np.shape(trainingData)
    print "Normal test data: ", np.shape(normalTestData)
    print "Abnormal test data: ", np.shape(abnormalTestData)
    print "======================================"

    # training hmm
    nEmissionDim = len(trainingData)
    detection_param_pkl = os.path.join(processed_data_path, 'hmm_'+task_name+'.pkl')
    cov_mult = [10.0]*(nEmissionDim**2)

    ml  = hmm.learning_hmm(nState, nEmissionDim, scale=1.0, cluster_type=cluster_type, verbose=False)
    ret = ml.fit(trainingData, cov_mult=cov_mult, ml_pkl=detection_param_pkl, \
                 use_pkl=True) # not(renew))
    ths = [threshold]*nState

    if ret == 'Failure': 
        print "-------------------------"
        print "HMM returned failure!!   "
        print "-------------------------"
        return (-1,-1,-1,-1)

    # discriminative classifier
    dtc = cb.classifier( ml, method='progress_time_cluster', nPosteriors=nState, \
                         nLength=len(trainingData[0,0]) )       
 

    # Visualization parameters
    fig = plt.figure()
    ax = fig.add_subplot(111)
    min_logp = 0.0
    max_logp = 100.0

    # normal test data
    _, l_logp_1, l_post_1 = ml.getLoglikelihoods(normalTestData, posterior=True)

    l_idx_1 = []
    for post in l_post_1:
        min_idx, _ = ml.findBestPosteriorDistribution(post)
        l_idx_1.append(min_idx)

    plt.plot(l_idx_1, l_logp_1, 'bo')

    
    # abnormal test data
    _, l_logp_2, l_post_2 = ml.getLoglikelihoods(abnormalTestData, posterior=True)
    
    l_idx_2 = []
    for post in l_post_2:
        min_idx, _ = ml.findBestPosteriorDistribution(post)
        l_idx_2.append(min_idx)

    plt.plot(l_idx_2, l_logp_2, 'rx')


    if min_logp > np.amin(l_logp_1)*1.3: min_logp = np.amin(l_logp_1)*1.3
    if max_logp < np.amax(l_logp_1)*1.3: max_logp = np.amax(l_logp_1)*1.3
    plt.ylim([min_logp, max_logp])

    if save_pdf == True:
        fig.savefig('test.pdf')
        fig.savefig('test.png')
        os.system('cp test.p* ~/Dropbox/HRL/')
        ## ut.get_keystroke('Hit a key to proceed next')
    else:
        plt.show()        

    
def aeDataExtraction(subject_names, task_name, raw_data_path, \
                    processed_data_path, param_dict,\
                    success_viz=False, failure_viz=False,\
                    cuda=True, verbose=False):

    ## Parameters
    # data
    data_dict  = param_dict['data_param']
    data_renew = data_dict['renew']
    feature_list = data_dict['feature_list']
    # AE
    AE_dict     = param_dict['AE']
    #------------------------------------------
    assert AE_dict['switch'] == True
                   
    crossVal_pkl = os.path.join(processed_data_path, 'cv_'+task_name+'.pkl')
    if os.path.isfile(crossVal_pkl):
        d = ut.load_pickle(crossVal_pkl)
    else:
        dd = dm.getDataSet(subject_names, task_name, raw_data_path, processed_data_path, \
                           data_dict['rf_center'], data_dict['local_range'],\
                           downSampleSize=data_dict['downSampleSize'], scale=1.0,\
                           ae_data=AE_dict['switch'], data_ext=data_dict['lowVarDataRemv'], \
                           nAugment=data_dict['nAugment'], feature_list=feature_list, \
                           cut_data=data_dict['cut_data'],
                           data_renew=data_renew)

        # Task-oriented raw features        
        successData = dd['aeSuccessData']
        failureData = dd['aeFailureData']
        aug_successData = dd['aeSuccessData_augmented']
        aug_failureData = dd['aeFailureData_augmented']
        
        kFold_list = dm.kFold_data_index2(len(aug_successData[0]), len(aug_failureData[0]),\
                                          data_dict['nNormalFold'], data_dict['nAbnormalFold'] )

        d = {}
        d['successData'] = successData
        d['failureData'] = failureData
        d['aug_successData'] = aug_successData
        d['aug_failureData'] = aug_failureData
        
        d['kFoldList']   = kFold_list                                             
        ut.save_pickle(d, crossVal_pkl)

    # Training HMM, and getting classifier training and testing data
    for idx, (normalTrainIdx, abnormalTrainIdx, normalTestIdx, abnormalTestIdx) \
      in enumerate( d['kFoldList'] ):

        if verbose: print "Start "+str(idx)+"/"+str(len( d['kFoldList'] ))+"th iteration"

        AE_proc_data = os.path.join(processed_data_path, 'ae_processed_data_'+str(idx)+'.pkl')

        # From dim x sample x length
        # To reduced_dim x sample
        dd = dm.getAEdataSet(idx, d['aug_successData'], d['aug_failureData'], \
                             normalTrainIdx, abnormalTrainIdx, normalTestIdx, abnormalTestIdx,
                             AE_dict['time_window'], \
                             AE_proc_data, \
                             # data param
                             processed_data_path, \
                             # AE param
                             layer_sizes=AE_dict['layer_sizes'], learning_rate=AE_dict['learning_rate'], \
                             learning_rate_decay=AE_dict['learning_rate_decay'], \
                             momentum=AE_dict['momentum'], dampening=AE_dict['dampening'], \
                             lambda_reg=AE_dict['lambda_reg'], \
                             max_iteration=AE_dict['max_iteration'], min_loss=AE_dict['min_loss'], \
                             cuda=AE_dict['cuda'], \
                             filtering=AE_dict['filter'], filteringDim=AE_dict['filterDim'],\
                             add_feature=AE_dict.get('add_feature', None),\
                             verbose=verbose, renew=AE_dict['renew'] )

        if AE_dict['filter']:
            # NOTE: pooling dimension should vary on each auto encoder.
            # Filtering using variances
            normalTrainData   = dd['normTrainDataFiltered']
            abnormalTrainData = dd['abnormTrainDataFiltered']
            normalTestData    = dd['normTestDataFiltered']
            abnormalTestData  = dd['abnormTestDataFiltered']
        else:
            normalTrainData   = dd['normTrainData']
            abnormalTrainData = dd['abnormTrainData']
            normalTestData    = dd['normTestData']
            abnormalTestData  = dd['abnormTestData']

        if success_viz or failure_viz:
            import data_viz as dv
            dv.viz(normalTrainData, abnormalTrainData)
            continue



# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------

def evaluation_all(subject_names, task_name, raw_data_path, processed_data_path, param_dict,\
                   data_renew=False, save_pdf=False, show_plot=True, verbose=False):

    ## Parameters
    # data
    data_dict  = param_dict['data_param']
    data_renew = data_dict['renew']
    # AE
    AE_dict     = param_dict['AE']
    autoEncoder = AE_dict['switch']
    # HMM
    HMM_dict = param_dict['HMM']
    nState   = HMM_dict['nState']
    cov      = HMM_dict['cov']
    # SVM

    # ROC
    ROC_dict = param_dict['ROC']
    
    #------------------------------------------

    

    if os.path.isdir(processed_data_path) is False:
        os.system('mkdir -p '+processed_data_path)

    crossVal_pkl = os.path.join(processed_data_path, 'cv_'+task_name+'.pkl')
    
    if os.path.isfile(crossVal_pkl) and data_renew is False:
        d = ut.load_pickle(crossVal_pkl)

        ## d['aeSuccessData'] = d['successData']
        ## d['aeFailureData'] = d['failureData']
        ## d['aeSuccessData_augmented'] = d['aug_successData']
        ## d['aeFailureData_augmented'] = d['aug_failureData']
        ## print d.keys()
        ## d = ut.save_pickle(d, crossVal_pkl)
        
        if AE_dict['switch']:
            successData = d['aeSuccessData']
            failureData = d['aeFailureData']
            aug_successData = d['aeSuccessData_augmented']
            aug_failureData = d['aeFailureData_augmented']
        else:
            successData = d['successData']
            failureData = d['failureData']
            aug_successData = d.get('successData_augmented', d['successData'])
            aug_failureData = d.get('failureData_augmented', d['failureData'])
        kFold_list  = d['kFoldList']



    else:
        '''
        Use augmented data? if nAugment is 0, then aug_successData = successData
        '''        
        d = dm.getDataSet(subject_names, task_name, raw_data_path, \
                           processed_data_path, data_dict['rf_center'], data_dict['local_range'],\
                           downSampleSize=data_dict['downSampleSize'], scale=1.0,\
                           ae_data=autoEncoder,\
                           data_ext=data_dict['lowVarDataRemv'], \
                           nAugment=data_dict['nAugment'], \
                           feature_list=data_dict['feature_list'], \
                           cut_data=data_dict['cut_data'], \
                           data_renew=data_renew)
        if AE_dict['switch']:
            successData     = d['aeSuccessData']
            failureData     = d['aeFailureData']
            aug_successData = d['aeSuccessData_augmented']
            aug_failureData = d['aeFailureData_augmented']
        else:
            successData     = d['successData']
            failureData     = d['failureData']
            aug_successData = d['successData_augmented']
            aug_failureData = d['failureData_augmented']

                          
        if data_dict['nAugment']>0:
            kFold_list = dm.kFold_data_index2(len(aug_successData[0]), len(aug_failureData[0]), \
                                              data_dict['nNormalFold'], data_dict['nAbnormalFold'] )
        else:
            kFold_list = dm.kFold_data_index2(len(successData[0]), len(failureData[0]), \
                                              data_dict['nNormalFold'], data_dict['nAbnormalFold'] )

        d['kFoldList']   = kFold_list

        
        ut.save_pickle(d, crossVal_pkl)


    #-----------------------------------------------------------------------------------------
    # parameters
    startIdx    = 4
    method_list = ROC_dict['methods'] 
    nPoints     = ROC_dict['nPoints']

    #-----------------------------------------------------------------------------------------
    # Training HMM, and getting classifier training and testing data
    for idx, (normalTrainIdx, abnormalTrainIdx, normalTestIdx, abnormalTestIdx) \
      in enumerate(kFold_list):

        if verbose: print idx, " : training hmm and getting classifier training and testing data"

        modeling_pkl = os.path.join(processed_data_path, 'hmm_'+task_name+'_'+str(idx)+'.pkl')

        if os.path.isfile(modeling_pkl) is False or HMM_dict['renew'] or data_renew:

            if autoEncoder:
                if verbose: print "Start "+str(idx)+"/"+str(len(kFold_list))+"th iteration"

                AE_proc_data = os.path.join(processed_data_path, 'ae_processed_data_'+str(idx)+'.pkl')
                
                # From dim x sample x length
                # To reduced_dim x sample x length
                d = dm.getAEdataSet(idx, aug_successData, aug_failureData, \
                                    normalTrainIdx, abnormalTrainIdx, normalTestIdx, abnormalTestIdx,\
                                    AE_dict['time_window'], \
                                    AE_proc_data, \
                                    # data param
                                    processed_data_path, \
                                    # AE param
                                    layer_sizes=AE_dict['layer_sizes'], learning_rate=AE_dict['learning_rate'], \
                                    learning_rate_decay=AE_dict['learning_rate_decay'], \
                                    momentum=AE_dict['momentum'], dampening=AE_dict['dampening'], \
                                    lambda_reg=AE_dict['lambda_reg'], \
                                    max_iteration=AE_dict['max_iteration'], min_loss=AE_dict['min_loss'], \
                                    filtering=AE_dict['filter'], filteringDim=AE_dict['filterDim'],\
                                    add_feature=AE_dict.get('add_feature', None),\
                                    cuda=False, verbose=False)

                if AE_dict['filter']:
                    # NOTE: pooling dimension should vary on each auto encoder.
                    # Filtering using variances
                    normalTrainData   = d['normTrainDataFiltered']
                    abnormalTrainData = d['abnormTrainDataFiltered']
                    normalTestData    = d['normTestDataFiltered']
                    abnormalTestData  = d['abnormTestDataFiltered']
                    ## import data_viz as dv
                    ## dv.viz(normalTrainData)
                    ## continue                   
                else:
                    normalTrainData   = d['normTrainData']
                    abnormalTrainData = d['abnormTrainData']
                    normalTestData    = d['normTestData']
                    abnormalTestData  = d['abnormTestData']
                
            else:
                # dim x sample x length
                ## trainingData      = successData[:, trainIdx, :] 
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
            nLength      = len(normalTrainData[0][0])

            ml  = hmm.learning_hmm(nState, nEmissionDim, verbose=verbose) 
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

            ll_classifier_train_X = []
            ll_classifier_train_Y = []
            for i in xrange(len(ll_logp)):
                l_X = []
                l_Y = []
                for j in xrange(len(ll_logp[i])):        
                    l_X.append( [ll_logp[i][j]] + ll_post[i][j].tolist() )

                    if testDataY[i] > 0.0: l_Y.append(1)
                    else: l_Y.append(-1)

                ll_classifier_train_X.append(l_X)
                ll_classifier_train_Y.append(l_Y)

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
            ll_classifier_test_X = []
            ll_classifier_test_Y = []
            for i in xrange(len(ll_logp)):
                l_X = []
                l_Y = []
                for j in xrange(len(ll_logp[i])):        
                    l_X.append( [ll_logp[i][j]] + ll_post[i][j].tolist() )

                    if testDataY[i] > 0.0: l_Y.append(1)
                    else: l_Y.append(-1)

                ll_classifier_test_X.append(l_X)
                ll_classifier_test_Y.append(l_Y)

                ## if len(l_Y) < 10:
                ##     print ">> ", np.shape(ll_logp[i]), np.shape(ll_post[i])
                ##     print i, np.shape(l_X), np.shape(l_Y)

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
        # temp
        if method not in ROC_data.keys() or method in ROC_dict['update_list']: # or method=='progress_time_cluster':# or method=='cssvm': # or  #or method=='cssvm_standard':# 
            ROC_data[method] = {}
            ROC_data[method]['complete'] = False 
            ROC_data[method]['tp_l'] = [ [] for j in xrange(nPoints) ]
            ROC_data[method]['fp_l'] = [ [] for j in xrange(nPoints) ]
            ROC_data[method]['tn_l'] = [ [] for j in xrange(nPoints) ]
            ROC_data[method]['fn_l'] = [ [] for j in xrange(nPoints) ]
            ROC_data[method]['delay_l'] = [ [] for j in xrange(nPoints) ]

    ## ROC_data['svm']['complete']                   = True
    ## ROC_data['progress_time_cluster']['complete'] = True
    
    # parallelization
    r = Parallel(n_jobs=-1)(delayed(run_classifiers)( idx, processed_data_path, task_name, ROC_data, ROC_dict ) for idx in xrange(len(kFold_list)) )    
    l_data = zip(*r)

    for i in xrange(len(l_data)):
        for j in xrange(nPoints):
            for k, method in enumerate(method_list):
                if ROC_data[method]['complete'] == True: continue
                ROC_data[method]['tp_l'][j] += l_data[method]['tp_l'][j]
                ROC_data[method]['fp_l'][j] += l_data[method]['fp_l'][j]
                ROC_data[method]['tn_l'][j] += l_data[method]['tn_l'][j]
                ROC_data[method]['fn_l'][j] += l_data[method]['fn_l'][j]
                ROC_data[method]['delay_l'][j] += l_data[method]['delay_l'][j]

    for i, method in enumerate(method_list):
        ROC_data[method]['complete'] = True

    ut.save_pickle(ROC_data, roc_pkl)
        
    #-----------------------------------------------------------------------------------------
    # ---------------- ROC Visualization ----------------------
    if True:
        
        ROC_data = ut.load_pickle(roc_pkl)        

        fig = plt.figure()

        for method in method_list:

            tp_ll = ROC_data[method]['tp_l']
            fp_ll = ROC_data[method]['fp_l']
            tn_ll = ROC_data[method]['tn_l']
            fn_ll = ROC_data[method]['fn_l']
            delay_ll = ROC_data[method]['delay_l']

            tpr_l = []
            fpr_l = []
            fnr_l = []
            delay_mean_l = []
            delay_std_l  = []

            print np.shape(tp_ll), np.shape(fn_ll), nPoints

            for i in xrange(nPoints):
                tpr_l.append( float(np.sum(tp_ll[i]))/float(np.sum(tp_ll[i])+np.sum(fn_ll[i]))*100.0 )
                fpr_l.append( float(np.sum(fp_ll[i]))/float(np.sum(fp_ll[i])+np.sum(tn_ll[i]))*100.0 )
                fnr_l.append( 100.0 - tpr_l[-1] )
                delay_mean_l.append( np.mean(delay_ll[i]) )
                delay_std_l.append( np.std(delay_ll[i]) )

            print "--------------------------------"
            print method
            print tpr_l
            print fpr_l
            print "--------------------------------"

            if method == 'svm': label='HMM-SVM'
            elif method == 'progress_time_cluster': label='HMMs with a dynamic threshold'
            elif method == 'fixed': label='HMMs with a fixed threshold'
                
            # visualization
            color = colors.next()
            shape = shapes.next()
            ax1 = fig.add_subplot(111)            
            plt.plot(fpr_l, tpr_l, '-'+shape+color, label=label, mec=color, ms=6, mew=2)
            plt.xlim([-1, 101])
            plt.ylim([-1, 101])
            plt.ylabel('True positive rate (percentage)', fontsize=22)
            plt.xlabel('False positive rate (percentage)', fontsize=22)

            ## font = {'family' : 'normal',
            ##         'weight' : 'bold',
            ##         'size'   : 22}
            ## matplotlib.rc('font', **font)
            ## plt.tick_params(axis='both', which='major', labelsize=12)
            plt.xticks([0, 50, 100], fontsize=22)
            plt.yticks([0, 50, 100], fontsize=22)
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            
            ## x = range(len(delay_mean_l))
            ## ax1 = fig.add_subplot(122)
            ## plt.errorbar(x, delay_mean_l, yerr=delay_std_l, c=color, label=method)

        plt.legend(loc='lower right', prop={'size':20})

        if save_pdf:
            fig.savefig('test.pdf')
            fig.savefig('test.png')
            os.system('cp test.p* ~/Dropbox/HRL/')        
        else:
            plt.show()
                   

def run_classifiers(idx, processed_data_path, task_name, ROC_data, ROC_dict ):

    ## if verbose: print idx, " : training classifier and evaluate testing data"
    # train a classifier and evaluate it using test data.
    from hrl_anomaly_detection.classifiers import classifier as cb
    from sklearn import preprocessing

    modeling_pkl = os.path.join(processed_data_path, 'hmm_'+task_name+'_'+str(idx)+'.pkl')
    d            = ut.load_pickle(modeling_pkl)
    nEmissionDim = d['nEmissionDim']
    A            = d['A']
    B            = d['B']
    pi           = d['pi']
    F            = d['F']
    nState       = d['nState']        
    ## startIdx = d['startIdx']
    ll_classifier_train_X   = d['ll_classifier_train_X']
    ll_classifier_train_Y   = d['ll_classifier_train_Y']         
    ll_classifier_train_idx = d['ll_classifier_train_idx']
    ll_classifier_test_X    = d['ll_classifier_test_X']  
    ll_classifier_test_Y    = d['ll_classifier_test_Y']
    ll_classifier_test_idx  = d['ll_classifier_test_idx']
    nLength      = d['nLength']

    method_list = ROC_dict['methods'] 
    nPoints     = ROC_dict['nPoints']

    #-----------------------------------------------------------------------------------------
    ## ## trainingData = successData[:, trainIdx, :]
    ## hmm_obs = hmm.learning_hmm(nState, nEmissionDim=nEmissionDim)
    ## hmm_obs.set_hmm_object(A,B,pi)
    ## ml = hmm_obs.ml

    #-----------------------------------------------------------------------------------------
    # flatten the data
    X_train_org = []
    Y_train_org = []
    idx_train_org = []
    for i in xrange(len(ll_classifier_train_X)):
        for j in xrange(len(ll_classifier_train_X[i])):
            X_train_org.append(ll_classifier_train_X[i][j])
            Y_train_org.append(ll_classifier_train_Y[i][j])
            idx_train_org.append(ll_classifier_train_idx[i][j])

    data = {}
    for i, method in enumerate(method_list):    
        # pass method if there is existing result
        if ROC_data[method]['complete'] == True: continue        
        data[method] = {}
        data[method]['tp_l'] = [ [] for j in xrange(nPoints) ]
        data[method]['fp_l'] = [ [] for j in xrange(nPoints) ]
        data[method]['tn_l'] = [ [] for j in xrange(nPoints) ]
        data[method]['fn_l'] = [ [] for j in xrange(nPoints) ]
        data[method]['delay_l'] = [ [] for j in xrange(nPoints) ]
        

    #-----------------------------------------------------------------------------------------
    # Generate parameter list for ROC curve
    for i, method in enumerate(method_list):

        # pass method if there is existing result
        if ROC_data[method]['complete'] == True: continue

        # data preparation
        if 'svm' in method:
            scaler = preprocessing.StandardScaler()
            ## scaler = preprocessing.scale()
            X_scaled = scaler.fit_transform(X_train_org)
        else:
            X_scaled = X_train_org
        print method, " : Before classification : ", np.shape(X_scaled), np.shape(Y_train_org)

        # classifier
        dtc = cb.classifier( method=method, nPosteriors=nState, nLength=nLength )        
        for j in xrange(nPoints):
            if method == 'svm':
                weights = ROC_dict['svm_param_range']
                ## weights = np.logspace(-2, 0.8, nPoints)
                dtc.set_params( class_weight=weights[j] )
                ## weights = np.linspace(0.5, 60.0, nPoints)
                ## dtc.set_params( class_weight= {1: 1.0, -1: weights[j]} )
            elif method == 'cssvm_standard':
                weights = np.logspace(-2, 0.1, nPoints)
                dtc.set_params( class_weight=weights[j] )
            elif method == 'cssvm':
                weights = ROC_dict['cssvm_param_range']
                dtc.set_params( class_weight=weights[j] )
            elif method == 'progress_time_cluster':
                ## thresholds = -np.linspace(1., 50, nPoints)+2.0
                thresholds = ROC_dict['progress_param_range']
                dtc.set_params( ths_mult = thresholds[j] )
            elif method == 'fixed':
                thresholds = ROC_dict['fixed_param_range']
                ## thresholds = np.linspace(0.1, -3.5, nPoints)
                dtc.set_params( ths_mult = thresholds[j] )

            ret = dtc.fit(X_scaled, Y_train_org, idx_train_org, parallel=False)

            ## X_scaled = scaler.transform(X_test_org)
            ## est_y = dtc.predict(X_scaled, Y_test_org)
            ## print est_y[:10]

            ## for jj in xrange(len(ll_classifier_test_X[0])):
            ##     X = scaler.transform([ll_classifier_test_X[0][jj]])
            ##     est_y = dtc.predict(X, y=ll_classifier_test_Y[0][jj:jj+1])
            ##     print est_y
            ##     if jj>10: break

            ## sys.exit()

            # evaluate the classifier
            tp_l = []
            fp_l = []
            tn_l = []
            fn_l = []
            delay_l = []
            delay_idx = 0

            for ii in xrange(len(ll_classifier_test_X)):

                if len(ll_classifier_test_Y[ii])==0: continue

                for jj in xrange(len(ll_classifier_test_X[ii])):
                    if 'svm' in method:
                        X = scaler.transform([ll_classifier_test_X[ii][jj]])
                    elif method == 'progress_time_cluster' or method == 'fixed':
                        X = ll_classifier_test_X[ii][jj]

                    est_y    = dtc.predict(X, y=ll_classifier_test_Y[ii][jj:jj+1])
                    if type(est_y) == list: est_y = est_y[0]
                    if type(est_y) == list: est_y = est_y[0]
                    ## X = X[0]

                    if est_y > 0.0:
                        delay_idx = ll_classifier_test_idx[ii][jj]
                        print "Break ", ii, " ", jj, " in ", est_y, " = ", ll_classifier_test_Y[ii][jj]
                        break        

                if ll_classifier_test_Y[ii][0] > 0.0:
                    if est_y > 0.0:
                        tp_l.append(1)
                        delay_l.append(delay_idx)
                    else: fn_l.append(1)
                elif ll_classifier_test_Y[ii][0] <= 0.0:
                    if est_y > 0.0: fp_l.append(1)
                    else: tn_l.append(1)

            data[method]['tp_l'][j] += tp_l
            data[method]['fp_l'][j] += fp_l
            data[method]['fn_l'][j] += fn_l
            data[method]['tn_l'][j] += tn_l
            data[method]['delay_l'][j] += delay_l

    return data


def evaluation(subject_names, task_name, raw_data_path, processed_data_path, rf_center, \
               local_range, \
               downSampleSize=200, \
               feature_list=['crossmodal_targetEEDist'], \
               nState=10, threshold=-1.0, smooth=False, cluster_type='time', \
               classifier_type='time', \
               hmm_renew=False, data_renew=False, save_pdf=False, show_plot=True, verbose=False):

    trainClassifier_pkl = os.path.join(processed_data_path, 'tc_'+task_name+'.pkl')
    if os.path.isfile(trainClassifier_pkl):
        d            = ut.load_pickle(trainClassifier_pkl)
        trainingData     = d['trainingData']
        normalTestData   = d['normalTestData']
        abnormalTestData = d['abnormalTestData']

        nEmissionDim = d['nEmissionDim']
        scale        = d['scale']
        A  = d['A']
        B  = d['B']
        pi = d['pi']
        F  = d['F']
        startIdx = d['startIdx']
        ll_X     = d['ll_X']
        ll_Y     = d['ll_Y']
        ll_idx   = d['ll_idx']
        ll_train_X = d['ll_train_X']
        
        import ghmm        
        if nEmissionDim >= 2:
            ml = ghmm.HMMFromMatrices(F, ghmm.MultivariateGaussianDistribution(F), \
                                      A, B, pi)
        else:
            ml = ghmm.HMMFromMatrices(F, ghmm.GaussianDistribution(F), A, B, pi)
        
    else:

        successData, failureData,_ ,_, _ = dm.getDataSet(subject_names, task_name, raw_data_path, \
                                                         processed_data_path, rf_center, local_range,\
                                                         downSampleSize=downSampleSize, \
                                                         feature_list=feature_list, \
                                                         data_renew=data_renew)

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

        d = {}
        d['trainingData'] = trainingData
        d['normalTestData'] = normalTestData   
        d['abnormalTestData'] = abnormalTestData
        
        # training hmm
        nEmissionDim = len(trainingData)
        scale        = 10.0
        detection_param_pkl = os.path.join(processed_data_path, 'hmm_'+task_name+'.pkl')
        cov_mult = [10.0]*(nEmissionDim**2)

        ml  = hmm.learning_hmm(nState, nEmissionDim, scale=scale, cluster_type=cluster_type, verbose=False)
        ret = ml.fit(trainingData, cov_mult=cov_mult, ml_pkl=detection_param_pkl, use_pkl=True) # not(renew))
        ## ths = [threshold]*nState

        if ret == 'Failure': 
            print "-------------------------"
            print "HMM returned failure!!   "
            print "-------------------------"
            return (-1,-1,-1,-1)

        #-----------------------------------------------------------------------------------------
        startIdx = 4
        r = Parallel(n_jobs=-1)(delayed(hmm.computeLikelihoods)(i, ml.A, ml.B, ml.pi, ml.F, \
                                                                [ trainingData[j][i] for j in xrange(nEmissionDim) ], \
                                                                ml.nEmissionDim, ml.scale, ml.nState,\
                                                                startIdx=startIdx, \
                                                                bPosterior=True)
                                                                for i in xrange(len(trainingData[0])))
        _, ll_idx, ll_logp, ll_post = zip(*r)

        ll_train_X = []
        for i in xrange(len(ll_logp)):
            l_X = []
            for j in xrange(len(ll_logp[i])):        
                l_X.append( [ll_logp[i][j]] + ll_post[i][j].tolist() )

            ll_train_X.append(l_X)


        #-----------------------------------------------------------------------------------------
        # Mix normal and abnormal test data
        testDataX = []
        testDataY = []
        for i in xrange(nEmissionDim):
            temp = np.vstack([normalTestData[i], abnormalTestData[i]])
            testDataX.append( temp )

        testDataY = np.hstack([ np.zeros(len(normalTestData[0])), np.ones(len(abnormalTestData[0])) ])

        # generate feature vectors for disriminative classifiers
        r = Parallel(n_jobs=-1)(delayed(hmm.computeLikelihoods)(i, ml.A, ml.B, ml.pi, ml.F, \
                                                                [ testDataX[j][i] for j in xrange(nEmissionDim) ], \
                                                                ml.nEmissionDim, ml.scale, ml.nState,\
                                                                startIdx=startIdx, \
                                                                bPosterior=True)
                                                                for i in xrange(len(testDataY)))
        _, ll_idx, ll_logp, ll_post = zip(*r)

        ll_X = []
        ll_Y = []
        for i in xrange(len(ll_logp)):
            l_X = []
            l_Y = []
            for j in xrange(len(ll_logp[i])):        
                l_X.append( [ll_logp[i][j]] + ll_post[i][j].tolist() )
                ## l_X.append( [ll_logp[i][j]] + [ll_idx[i][j]] )

                if testDataY[i] > 0.5: l_Y.append(1.0)
                else: l_Y.append(-1.0)

            ll_X.append(l_X)
            ll_Y.append(l_Y)               


        d['nEmissionDim'] = ml.nEmissionDim
        d['scale']        = scale        
        d['A'] = ml.A 
        d['B'] = ml.B 
        d['pi']= ml.pi
        d['F'] = ml.F
        d['startIdx'] = startIdx
        d['ll_train_X'] = ll_train_X
        d['ll_X'] = ll_X 
        d['ll_Y'] = ll_Y
        d['ll_idx']= ll_idx
        ut.save_pickle(d, trainClassifier_pkl)


    #-----------------------------------------------------------------------------------------
    print "Start to train SVM"
    #-----------------------------------------------------------------------------------------
    X_org = []
    Y_org = []
    for i in xrange(len(ll_X)):
        for j in xrange(len(ll_X[i])):
            X_org.append(ll_X[i][j])
            Y_org.append(ll_Y[i][j])

    # train svm
    method_list = ['svm', 'cssvm', 'progress_time_cluster', 'fixed']
    method_list = ['cssvm']
    ROC_data = {}
    from hrl_anomaly_detection.classifiers import classifier as cb
    
    # Generate parameter list for ROC curve
    for i, method in enumerate(method_list):

        # data preparation
        if method == 'svm' or method=='cssvm':
            from sklearn import preprocessing
            scaler = preprocessing.StandardScaler()
            ## scaler = preprocessing.scale()
            X_scaled = scaler.fit_transform(X_org)
        else:
            X_scaled = X_org
        print method, " : Before classification : ", np.shape(X_scaled), np.shape(Y_org)

        # containers or parameters
        tpr_l = []
        fpr_l = []
        ## tn_ll = []
        fnr_l = []
        delay_mean_l = []
        delay_std_l  = []

        # classifier
        dtc = cb.classifier( ml, method=method, nPosteriors=nState, nLength=len(trainingData[0,0]) )        
        nPoints = 10
        for j in xrange(nPoints):
        
            if method == 'svm':
                weights = np.linspace(1.0, 15.0, nPoints)
                dtc.set_params( class_weight= {1.0: 1.0, -1.0: weights[j]} )
            elif method == 'cssvm':
                weights = np.linspace(1, 100, nPoints)
                dtc.set_params( class_weight=weights[j] )
            elif method == 'progress_time_cluster':
                thresholds = np.linspace(-60, -1.0, nPoints)
                dtc.set_params( ths_mult = thresholds[j] )

            ret = dtc.fit(X_scaled, Y_org, ll_idx)
            ## print "Score: ", dtc.score(X_scaled, Y)

            #-----------------------------------------------------------------------------------------
            print "Start to evaluate the classifiers ", j
            #-----------------------------------------------------------------------------------------
            tp_l = []
            fp_l = []
            tn_l = []
            fn_l = []
            delay_l = []
            delay_idx = 0
            for ii in xrange(len(ll_X)):
                for jj in xrange(len(ll_X[ii])):

                    if method == 'svm' or method == 'cssvm':
                        X = scaler.transform([ll_X[ii][jj]])
                        ## X_scaled = ll_X[i][j]
                        ## X_scaled[0] = (X_scaled[0]-xmin)/(xmax-xmin)
                        ## ## X_scaled = np.array(X_scaled)[:1]
                    elif method == 'progress_time_cluster':
                        X = ll_X[ii][jj]

                    est_y    = dtc.predict(X)
                    if type(est_y) == list: est_y = est_y[0]
                    X = X[0]

                    if est_y > 0.0:
                        delay_idx = ll_idx[ii][jj]
                        ## print "Break ", i, " ", j, " in ", est_y, " = ", ll_Y[i][j]                 
                        break        

                if ll_Y[ii][0] > 0.0:
                    if est_y > 0.0:
                        tp_l.append(1)
                        delay_l.append(delay_idx)
                    else: fn_l.append(1)
                elif ll_Y[ii][0] <= 0.0:
                    if est_y > 0.0: fp_l.append(1)
                    else: tn_l.append(1)

            tpr_l.append( float(np.sum(tp_l))/float(np.sum(tp_l)+np.sum(fn_l)) )
            fpr_l.append( float(np.sum(fp_l))/float(np.sum(fp_l)+np.sum(tn_l)) )
            fnr_l.append( 1.0 - tpr_l[-1] )
            delay_mean_l.append( np.mean(delay_l) )
            delay_std_l.append( np.std(delay_l) )

        ROC_data[method] = {'tpr': tpr_l,
                            'fpr': fpr_l,
                            'fnr': fnr_l,
                            'delay_mean_l': delay_mean_l,
                            'delay_std_l': delay_std_l}    
            
    #-----------------------------------------------------------------------------------------
    # ---------------- ROC Visualization ----------------------
    if False:

        fig = plt.figure()
        ax1 = fig.add_subplot(121)

        for method in method_list:
            fpr_l = ROC_data[method]['fpr']
            tpr_l = ROC_data[method]['tpr']
            color = colors.next()
            plt.plot(fpr_l, tpr_l, c=color, label=method)

            print "------------------------------"
            print method
            print fpr_l
            print tpr_l
            
            
        plt.legend(loc='upper right')

        ax2 = fig.add_subplot(122)
        for method in method_list:            
            delay_mean_l = ROC_data[method]['delay_mean_l']
            delay_std_l  = ROC_data[method]['delay_std_l']
            x = range(len(delay_mean_l))
            color = colors.next()

            plt.errorbar(x, delay_mean_l, yerr=delay_std_l, c=color, label=method)

        plt.legend(loc='upper right')
        
        
    # ---------------- Boundary Visualization ----------------------
    if True:
        fig = plt.figure()
        for i, y in enumerate(Y_org):
            if y > 0:
                plt.plot(X_scaled[i,1], X_scaled[i,0], 'r.')
            else:
                plt.plot(X_scaled[i,1], X_scaled[i,0], 'b.')

        # create a mesh to plot in
        h = 0.1
        x_min, x_max = X_scaled[:, 1].min() - 0.1, X_scaled[:, 1].max() + 0.1
        y_min, y_max = X_scaled[:, 0].min() - 0.1, X_scaled[:, 0].max() + 0.1
        x_min = -2.0
        x_max = 2.0
        y_min = 0.0
        y_max = 1.5
        
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        Z = dtc.decision_function(np.c_[yy.ravel(), xx.ravel()])

        # Put the result into a color plot
        if type(Z) is list: Z=np.array(Z)
        Z = Z.reshape(np.shape(xx))

        plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.Blues_r)
        plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='orange')
        ## plt.axis('off')
    

    ## print "false negative rate (FNR): ", (1.0-tpr)*100.0
    ## print "false positive rate (FPR): ", fpr*100.0
    ## print "Detection delay ", np.mean(delay_l), np.std(delay_l)


    if save_pdf:
        fig.savefig('test.pdf')
        fig.savefig('test.png')
        os.system('cp test.p* ~/Dropbox/HRL/')        
    else:
        plt.show()
    
            
                
    





def trainClassifierSVM(subject_names, task_name, raw_data_path, processed_data_path, rf_center, \
                       local_range, \
                       downSampleSize=200, \
                       feature_list=['crossmodal_targetEEDist'], \
                       nState=10, threshold=-1.0, smooth=False, cluster_type='time', \
                       classifier_type='time', \
                       hmm_renew=False, data_renew=False, save_pdf=False, show_plot=True):

    trainClassifier_pkl = os.path.join(processed_data_path, 'tc_'+task_name+'.pkl')
    if os.path.isfile(trainClassifier_pkl):
        d            = ut.load_pickle(trainClassifier_pkl)
        trainingData     = d['trainingData']
        normalTestData   = d['normalTestData']
        abnormalTestData = d['abnormalTestData']
    else:

        successData, failureData,_,_,_ = dm.getDataSet(subject_names, task_name, raw_data_path, \
                                                       processed_data_path, rf_center, local_range,\
                                                       downSampleSize=downSampleSize, \
                                                       feature_list=feature_list, \
                                                       data_renew=data_renew)

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

        d = {}
        d['trainingData'] = trainingData
        d['normalTestData'] = normalTestData   
        d['abnormalTestData'] = abnormalTestData
        ut.save_pickle(d, trainClassifier_pkl)

        
    # training hmm
    nEmissionDim = len(trainingData)
    detection_param_pkl = os.path.join(processed_data_path, 'hmm_'+task_name+'.pkl')
    cov_mult = [10.0]*(nEmissionDim**2)

    ml  = hmm.learning_hmm(nState, nEmissionDim, scale=10.0, cluster_type=cluster_type, verbose=False)
    ret = ml.fit(trainingData, cov_mult=cov_mult, ml_pkl=detection_param_pkl, use_pkl=True) # not(renew))
    ## ths = [threshold]*nState

    if ret == 'Failure': 
        print "-------------------------"
        print "HMM returned failure!!   "
        print "-------------------------"
        return (-1,-1,-1,-1)


    # Mix normal and abnormal test data
    testDataX = []
    testDataY = []
    for i in xrange(nEmissionDim):
        temp = np.vstack([normalTestData[i], abnormalTestData[i]])
        testDataX.append(temp)

    testDataY   = np.hstack([ np.zeros(len(normalTestData[0])), np.ones(len(abnormalTestData[0])) ])
    testIdxList = random.sample( range(len(testDataY)), len(testDataY) )

    from hrl_anomaly_detection.classifiers import classifier_base as cb
    dtc = cb.classifier(ml, method='svm')
    from sklearn.decomposition import KernelPCA # Too bad
    dr = KernelPCA(n_components=2, kernel=cb.custom_kernel, gamma=5.0)

    # NOTE !!!!!!!!!!!!!!!!!!!!!!!!!!!
    eps = 1e-6

    # train pca
    ## print "Reducing the dimension of training data"
    ## X = []
    ## for i in xrange(len(trainingData[0])):
    ##     data = [ trainingData[j][i] for j in xrange(nEmissionDim) ]        
    ##     l_logp, l_post = ml.loglikelihoods(data, bPosterior=True, startIdx=startIdx)

    ##     # set feature vector
    ##     for j in xrange(len(l_logp)):
    ##         X.append( [l_logp[j]] + l_post[j].tolist() )

    ## ## print np.shape(X), np.shape(Y)
    ## training_rx = dr.fit_transform(X)
    ## from sklearn import preprocessing    

    # update the classifier when new data comes in    
    fig = plt.figure()
    X = []
    Y = []
    for i, idx in enumerate(testIdxList):
        print "updating classifier : ", i

        ll_post = []
        ll_logp = []
        ll_label = []

        # get hmm induced features: 1 x dim x length
        testData = [ testDataX[j][idx] for j in xrange(nEmissionDim) ]
        ls_logp, ls_post = ml.loglikelihoods(testData, bPosterior=True, startIdx=startIdx)
        l_logp, l_post   = l_logp[0], l_post[0]
        
        # set feature vector
        for j in xrange(len(l_logp)):
            X.append( [l_logp[j]] + l_post[j].tolist() )
        
            if testDataY[idx] > 0.5: Y.append(1.0)
            else: Y.append(0.0)

        # ToDo: how to normalize the features?
        # loglikelihood
        x_max = max(np.array(X)[:,0])
        x_min = min(np.array(X)[:,0])
        x_std = (np.array(X)[:,0] - x_min) / (x_max - x_min)
        X_scaled = x_std / (x_max - x_min) + x_min

        X_scaled = np.hstack([np.array([X_scaled]).T, np.array(X)[:,1:]])
        X_scaled[:,1:] += eps
        print "finished to scale the feature vector"

        # train svm
        ret = dtc.fit(X_scaled,Y)
        print "finished svm training: ", ret

        ## visualize the boundary
        # --------------- Dimension Reduction --------------------------
        ## X_scaled = preprocessing.scale(X)
        test_rx = dr.fit_transform(X_scaled)

        # ---------------- Boundary Visualization ----------------------
        # create a mesh to plot in
        h = 1.0
        x_min, x_max = test_rx[:, 0].min() - 0.2, test_rx[:, 0].max() + 0.2
        y_min, y_max = test_rx[:, 1].min() - 0.2, test_rx[:, 1].max() + 0.2

        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        ## Z = dtc.dt.decision_function(np.c_[xx.ravel(), yy.ravel()])
        ## Z = Z.reshape(xx.shape)
        
        ## plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.Blues_r)
        ## plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='orange')
        plt.axis('off')
        
        # ---------------- Sample Visualization ------------------------
        #
        abnormal_rx = [x for x, y in zip(test_rx, Y) if y > 0 ]
        normal_rx = [x for x, y in zip(test_rx, Y) if y <= 0]

        if len(normal_rx) > 0:
            plt.scatter(np.array(normal_rx)[:,0], np.array(normal_rx)[:,1], c='b', label=None)
        if len(abnormal_rx) > 0:            
            plt.scatter(np.array(abnormal_rx)[:,0], np.array(abnormal_rx)[:,1], c='r', label=None)

        if save_pdf:
            fig.savefig('test.pdf')
            fig.savefig('test.png')
            os.system('cp test.p* ~/Dropbox/HRL/')        
        else:
            plt.show()
            fig = plt.figure()

    
def trainClassifier(subject_names, task_name, raw_data_path, processed_data_path, rf_center, \
                    local_range, \
                    downSampleSize=200, \
                    feature_list=['crossmodal_targetEEDist'], \
                    nState=10, threshold=-1.0, smooth=False, cluster_type='time', \
                    classifier_type='time', \
                    useTrain=True, useNormalTest=True, useAbnormalTest=False,\
                    useTrain_color=False, useNormalTest_color=False, useAbnormalTest_color=False,\
                    hmm_renew=False, data_renew=False, save_pdf=False, show_plot=True):

    successData, failureData,_ ,_,_= dm.getDataSet(subject_names, task_name, raw_data_path, \
                                                   processed_data_path, rf_center, local_range,\
                                                   downSampleSize=downSampleSize, \
                                                   feature_list=feature_list, \
                                                   data_renew=data_renew)

    # index selection
    success_idx  = range(len(successData[0]))
    failure_idx  = range(len(failureData[0]))
    
    nTrain       = int( 0.7*len(success_idx) )    
    train_idx    = random.sample(success_idx, nTrain)
    success_test_idx = [x for x in success_idx if not x in train_idx]
    failure_test_idx = failure_idx

    # data structure: dim x nData x sequence
    trainingData     = successData[:, train_idx, :]
    normalTestData   = successData[:, success_test_idx, :]
    abnormalTestData = failureData[:, failure_test_idx, :]
    
    print "======================================"
    print "Training data: ", np.shape(trainingData)
    print "Normal test data: ", np.shape(normalTestData)
    print "Abnormal test data: ", np.shape(abnormalTestData)
    print "======================================"

    # training hmm
    nEmissionDim = len(trainingData)
    detection_param_pkl = os.path.join(processed_data_path, 'hmm_'+task_name+'.pkl')
    cov_mult = [10.0]*(nEmissionDim**2)

    ml  = hmm.learning_hmm(nState, nEmissionDim, scale=10.0, cluster_type=cluster_type, verbose=False)
    ret = ml.fit(trainingData, cov_mult=cov_mult, ml_pkl=detection_param_pkl, \
                 use_pkl=True) # not(renew))
    ths = [threshold]*nState

    if ret == 'Failure': 
        print "-------------------------"
        print "HMM returned failure!!   "
        print "-------------------------"
        return (-1,-1,-1,-1)

    # initialize internal classifier update parameter
    ml.opt_A   = []
    ml.opt_B   = []
    ml.opt_idx = []
    ml.opt_y   = []
    ml.opt_logp= []

    log_ll     = []
    exp_log_ll = []
    dec_log_ll = []
    c1 = 1.0
    c2 = 1.0        

    # Mix normal and abnormal test data
    testDataX = []
    testDataY = []
    for i in xrange(len(trainingData)): # dimension
        temp = np.vstack([normalTestData[i], abnormalTestData[i]])
        testDataX.append(temp)

    testDataY   = np.hstack([ np.zeros(len(normalTestData[0])), np.ones(len(abnormalTestData[0])) ])
    testIdxList = random.sample( range(len(testDataY)), len(testDataY) )

    min_logp = 0.0
    max_logp = 100.0

    # update the classifier when new data comes in    
    for i, idx in enumerate(testIdxList):
    
        prev_exp_log_l = []
        l_prev_cost = []
        for j in range(2, len(testDataX[0][idx])):
            X = [x[idx,:j] for x in testDataX]                

            exp_logp, logp = ml.expLoglikelihood(X, ml.l_ths_mult, smooth=smooth, bLoglikelihood=True)
            prev_exp_log_l.append(exp_logp)
            l_prev_cost.append( max(0, 1.0+(2.0*c2-1.0)*(exp_logp-logp)) )

        #----------------------------------------------------------
        X = [x[idx,:] for x in testDataX]
        y = testDataY[idx]
        cf.updateClassifierCoff(ml, X, y, c1=c1, c2=c2, classifier_type=classifier_type)
        #----------------------------------------------------------

        log_ll.append([])
        exp_log_ll.append([])
        dec_log_ll.append([])
        l_cost = []
        for j in range(2, len(testDataX[0][idx])):
            X = [x[idx,:j] for x in testDataX]                

            exp_logp, logp = ml.expLoglikelihood(X, ml.l_ths_mult, smooth=smooth, bLoglikelihood=True)
            log_ll[i].append(logp)
            exp_log_ll[i].append(exp_logp)
            dec_log_ll[i].append(exp_logp-logp)
            l_cost.append( max(0, 1.0+(2.0*c2-1.0)*(exp_logp-logp)) )

        # visualization
        fig = plt.figure()
        ax = fig.add_subplot(211)

        for j in xrange(len(log_ll)):
            if testDataY[ testIdxList[j] ] == 0.0:
                if j == len(log_ll)-1:
                    plt.plot(log_ll[j], 'b-', lw=3.0)
                else:
                    plt.plot(log_ll[j], 'b-')

                if min_logp > np.amin(log_ll[j])*1.3: min_logp = np.amin(log_ll[j])*1.3
                if max_logp < np.amax(log_ll[j])*1.3: max_logp = np.amax(log_ll[j])*1.3
                    
            else:
                if j == len(log_ll)-1:
                    plt.plot(log_ll[j], 'r-', lw=3.0)
                else:
                    plt.plot(log_ll[j], 'r-')
                    
        plt.plot(prev_exp_log_l, 'm*--', lw=3.0)            
        plt.plot(exp_log_ll[-1], 'm-', lw=3.0)            
       # plt.plot(dec_log_ll[-1], 'g-', lw=1.0)            
        plt.ylim([min_logp, max_logp])

        ax = fig.add_subplot(212)
        plt.plot(l_prev_cost, 'b-')
        plt.plot(l_cost, 'r-')

        if save_pdf == True:
            fig.savefig('test.pdf')
            fig.savefig('test.png')
            os.system('cp test.p* ~/Dropbox/HRL/')
            ut.get_keystroke('Hit a key to proceed next')
        else:
            plt.show()        



    


    
    
                
        
def data_plot(subject_names, task_name, raw_data_path, processed_data_path, \
              downSampleSize=200, \
              local_range=0.3, rf_center='kinEEPos', global_data=False, \
              success_viz=True, failure_viz=False, \
              raw_viz=False, interp_viz=False, save_pdf=False, \
              successData=False, failureData=True,\
              continuousPlot=False, \
              ## trainingData=True, normalTestData=False, abnormalTestData=False,\
              modality_list=['audio'], data_renew=False, verbose=False):    

    if os.path.isdir(processed_data_path) is False:
        os.system('mkdir -p '+processed_data_path)

    success_list, failure_list = getSubjectFileList(raw_data_path, subject_names, task_name)

    fig = plt.figure('all')
    time_lim    = [0.01, 0] 
    nPlot       = len(modality_list)

    for idx, file_list in enumerate([success_list, failure_list]):
        if idx == 0 and successData is not True: continue
        elif idx == 1 and failureData is not True: continue        

        ## fig = plt.figure('loadData')                        
        # loading and time-sync
        if idx == 0:
            if verbose: print "Load success data"
            data_pkl = os.path.join(processed_data_path, task+'_success_'+rf_center+\
                                    '_'+str(local_range))
            raw_data_dict, interp_data_dict = loadData(success_list, isTrainingData=True,
                                                       downSampleSize=downSampleSize,\
                                                       local_range=local_range, rf_center=rf_center,\
                                                       global_data=global_data, \
                                                       renew=data_renew, save_pkl=data_pkl, verbose=verbose)
        else:
            if verbose: print "Load failure data"
            data_pkl = os.path.join(processed_data_path, task+'_failure_'+rf_center+\
                                    '_'+str(local_range))
            raw_data_dict, interp_data_dict = loadData(failure_list, isTrainingData=False,
                                                       downSampleSize=downSampleSize,\
                                                       local_range=local_range, rf_center=rf_center,\
                                                       global_data=global_data,\
                                                       renew=data_renew, save_pkl=data_pkl, verbose=verbose)
            
        ## plt.show()
        ## sys.exit()
        if raw_viz: target_dict = raw_data_dict
        else: target_dict = interp_data_dict

        # check only training data to get time limit (TEMP)
        if idx == 0:
            for key in interp_data_dict.keys():
                if 'timesList' in key:
                    time_list = interp_data_dict[key]
                    if len(time_list)==0: continue
                    for tl in time_list:
                        ## print tl[-1]
                        time_lim[-1] = max(time_lim[-1], tl[-1])
            ## continue

        # for each file in success or failure set
        for fidx in xrange(len(file_list)):
                        
            count = 0
            for modality in modality_list:
                count +=1

                if 'audioWrist' in modality:
                    time_list = target_dict['audioWristTimesList']
                    data_list = target_dict['audioWristRMSList']
                    
                elif 'audio' in modality:
                    time_list = target_dict['audioTimesList']
                    data_list = target_dict['audioPowerList']

                elif 'kinematics' in modality:
                    time_list = target_dict['kinTimesList']
                    data_list = target_dict['kinPosList']

                    # distance
                    new_data_list = []
                    for d in data_list:
                        new_data_list.append( np.linalg.norm(d, axis=0) )
                    data_list = new_data_list

                elif 'ft' in modality:
                    time_list = target_dict['ftTimesList']
                    data_list = target_dict['ftForceList']

                    # distance
                    if len(np.shape(data_list[0])) > 1:
                        new_data_list = []
                        for d in data_list:
                            new_data_list.append( np.linalg.norm(d, axis=0) )
                        data_list = new_data_list

                elif 'vision_artag' in modality:
                    time_list = target_dict['visionArtagTimesList']
                    data_list = target_dict['visionArtagPosList']

                    # distance
                    new_data_list = []
                    for d in data_list:                    
                        new_data_list.append( np.linalg.norm(d[:3], axis=0) )
                    data_list = new_data_list

                elif 'vision_change' in modality:
                    time_list = target_dict['visionChangeTimesList']
                    data_list = target_dict['visionChangeMagList']

                elif 'pps' in modality:
                    time_list = target_dict['ppsTimesList']
                    data_list1 = target_dict['ppsLeftList']
                    data_list2 = target_dict['ppsRightList']

                    # magnitude
                    new_data_list = []
                    for i in xrange(len(data_list1)):
                        d1 = np.array(data_list1[i])
                        d2 = np.array(data_list2[i])
                        d = np.vstack([d1, d2])
                        new_data_list.append( np.sum(d, axis=0) )

                    data_list = new_data_list

                elif 'fabric' in modality:
                    time_list = target_dict['fabricTimesList']
                    ## data_list = target_dict['fabricValueList']
                    data_list = target_dict['fabricMagList']


                    ## for ii, d in enumerate(data_list):
                    ##     print np.max(d), target_dict['fileNameList'][ii]

                    ## # magnitude
                    ## new_data_list = []
                    ## for d in data_list:

                    ##     # d is 3xN-length in which each element has multiple float values
                    ##     sample = []
                    ##     if len(d) != 0 and len(d[0]) != 0:
                    ##         for i in xrange(len(d[0])):
                    ##             if d[0][i] == []:
                    ##                 sample.append( 0 )
                    ##             else:                                                               
                    ##                 s = np.array([d[0][i], d[1][i], d[2][i]])
                    ##                 v = np.mean(np.linalg.norm(s, axis=0)) # correct?
                    ##                 sample.append(v)
                    ##     else:
                    ##         print "WRONG data size in fabric data"

                    ##     new_data_list.append(sample)
                    ## data_list = new_data_list

                    ## fig_fabric = plt.figure('fabric')
                    ## ax_fabric = fig_fabric.add_subplot(111) #, projection='3d')
                    ## for d in data_list:
                    ##     color = colors.next()
                    ##     for i in xrange(len(d[0])):
                    ##         if d[0][i] == []: continue
                    ##         ax_fabric.scatter(d[1][i], d[0][i], c=color)
                    ##         ## ax_fabric.scatter(d[0][i], d[1][i], d[2][i])
                    ## ax_fabric.set_xlabel('x')
                    ## ax_fabric.set_ylabel('y')
                    ## ## ax_fabric.set_zlabel('z')
                    ## if save_pdf is False:
                    ##     plt.show()
                    ## else:
                    ##     fig_fabric.savefig('test_fabric.pdf')
                    ##     fig_fabric.savefig('test_fabric.png')
                    ##     os.system('mv test*.p* ~/Dropbox/HRL/')

                ax = fig.add_subplot(nPlot*100+10+count)
                if idx == 0: color = 'b'
                else: color = 'r'            

                if raw_viz:
                    combined_time_list = []
                    if data_list == []: continue

                    ## for t in time_list:
                    ##     temp = np.array(t[1:])-np.array(t[:-1])
                    ##     combined_time_list.append([ [0.0]  + list(temp)] )
                    ##     print modality, " : ", np.mean(temp), np.std(temp), np.max(temp)
                    ##     ## ax.plot(temp, label=modality)

                    for i in xrange(len(time_list)):
                        if len(time_list[i]) > len(data_list[i]):
                            ax.plot(time_list[i][:len(data_list[i])], data_list[i], c=color)
                        else:
                            ax.plot(time_list[i], data_list[i][:len(time_list[i])], c=color)

                    if continuousPlot:
                        new_color = 'm'
                        i         = fidx
                        if len(time_list[i]) > len(data_list[i]):
                            ax.plot(time_list[i][:len(data_list[i])], data_list[i], c=new_color, lw=3.0)
                        else:
                            ax.plot(time_list[i], data_list[i][:len(time_list[i])], c=new_color, lw=3.0)
                                                    
                else:
                    interp_time = np.linspace(time_lim[0], time_lim[1], num=downSampleSize)
                    for i in xrange(len(data_list)):
                        ax.plot(interp_time, data_list[i], c=color)                
                
                ax.set_xlim(time_lim)
                ax.set_title(modality)

            #------------------------------------------------------------------------------    
            if continuousPlot is False: break
            else:
                        
                print "-----------------------------------------------"
                print file_list[fidx]
                print "-----------------------------------------------"

                plt.tight_layout(pad=0.1, w_pad=0.5, h_pad=0.0)

                if save_pdf is False:
                    plt.show()
                else:
                    print "Save pdf to Dropbox folder"
                    fig.savefig('test.pdf')
                    fig.savefig('test.png')
                    os.system('mv test.p* ~/Dropbox/HRL/')

                fig = plt.figure('all')

                
    plt.tight_layout(pad=0.1, w_pad=0.5, h_pad=0.0)

    if save_pdf is False:
        plt.show()
    else:
        print "Save pdf to Dropbox folder"
        fig.savefig('test.pdf')
        fig.savefig('test.png')
        os.system('mv test.p* ~/Dropbox/HRL/')


    ## # training set
    ## trainingData, param_dict = extractFeature(data_dict['trainData'], feature_list, local_range)

    ## # test set
    ## normalTestData, _ = extractFeature(data_dict['normalTestData'], feature_list, local_range, \
    ##                                         param_dict=param_dict)        
    ## abnormalTestData, _ = extractFeature(data_dict['abnormalTestData'], feature_list, local_range, \
    ##                                         param_dict=param_dict)

    ## print "======================================"
    ## print "Training data: ", np.shape(trainingData)
    ## print "Normal test data: ", np.shape(normalTestData)
    ## print "Abnormal test data: ", np.shape(abnormalTestData)
    ## print "======================================"

    ## visualization_hmm_data(feature_list, trainingData=trainingData, \
    ##                        normalTestData=normalTestData,\
    ##                        abnormalTestData=abnormalTestData, save_pdf=save_pdf)        
    

def data_selection(subject_names, task_name, raw_data_path, processed_data_path, \
                  downSampleSize=200, \
                  local_range=0.3, rf_center='kinEEPos', \
                  success_viz=True, failure_viz=False, \
                  raw_viz=False, save_pdf=False, \
                  modality_list=['audio'], data_renew=False, verbose=False):    

    ## success_list, failure_list = getSubjectFileList(raw_data_path, subject_names, task_name)
    
    # Success data
    successData = success_viz
    failureData = failure_viz

    count = 0
    while True:
        
        ## success_list, failure_list = getSubjectFileList(raw_data_path, subject_names, task_name)        
        ## print "-----------------------------------------------"
        ## print success_list[count]
        ## print "-----------------------------------------------"
        
        data_plot(subject_names, task_name, raw_data_path, processed_data_path,\
                  downSampleSize=downSampleSize, \
                  local_range=local_range, rf_center=rf_center, \
                  raw_viz=True, interp_viz=False, save_pdf=save_pdf,\
                  successData=successData, failureData=failureData,\
                  continuousPlot=True, \
                  modality_list=modality_list, data_renew=data_renew, verbose=verbose)

        break

        ## feedback  = raw_input('Do you want to exclude the data? (e.g. y:yes n:no else: exit): ')
        ## if feedback == 'y':
        ##     print "move data"
        ##     ## os.system('mv '+subject_names+' ')
        ##     data_renew = True

        ## elif feedback == 'n':
        ##     print "keep data"
        ##     data_renew = False
        ##     count += 1
        ## else:
        ##     break
   

    

## def pca_plot(subject_names, task_name, raw_data_path, processed_data_path, rf_center, local_range, \
##              downSampleSize=200, success_viz=True, failure_viz=False, \
##              save_pdf=False, \
##              feature_list=['crossmodal_targetEEDist'], data_renew=False):


##     allData, trainingData, abnormalTestData, abnormalTestNameList\
##       = dm.getDataSet(subject_names, task_name, raw_data_path, \
##                       processed_data_path, rf_center, local_range,\
##                       downSampleSize=downSampleSize, \
##                       feature_list=feature_list, \
##                       data_renew=data_renew)

##     print "---------------------------------------------------"
##     print np.shape(trainingData), np.shape(abnormalTestData)
##     print "---------------------------------------------------"
    
##     m,n,k = np.shape(allData)
##     all_data_array = None
##     for i in xrange(n):
##         for j in xrange(k):
##             if all_data_array is None: all_data_array = allData[:,i,j]
##             else: all_data_array = np.vstack([all_data_array, allData[:,i,j]])
                
##     m,n,k = np.shape(trainingData)
##     success_data_array = None
##     for i in xrange(n):
##         for j in xrange(k):
##             if success_data_array is None: success_data_array = trainingData[:,i,j]
##             else: success_data_array = np.vstack([success_data_array, trainingData[:,i,j]])

##     m,n,k = np.shape(abnormalTestData)
##     failure_data_array = None
##     for i in xrange(n):
##         for j in xrange(k):
##             if failure_data_array is None: failure_data_array = abnormalTestData[:,i,j]
##             else: failure_data_array = np.vstack([failure_data_array, abnormalTestData[:,i,j]])

##     #--------------------- Parameters -------------------------------
##     fig = plt.figure()
##     # step size in the mesh
##     h = .01

##     # ------------------- Visualization using different PCA? --------
##     dr = {}
##     from sklearn.manifold import Isomap
##     ## dr['isomap4'] = Isomap(n_neighbors=4, n_components=2)
##     ## dr['isomap5'] = Isomap(n_neighbors=5, n_components=2)
##     dr['isomap4'] = Isomap(n_neighbors=4, n_components=2)
##     dr['isomap7'] = Isomap(n_neighbors=7, n_components=2)
##     from sklearn.decomposition import KernelPCA # Too bad
##     dr['kpca_gamma5'] = KernelPCA(n_components=2, kernel="linear", gamma=5.0)
##     dr['kpca_gamma2'] = KernelPCA(n_components=2, kernel="rbf", gamma=2.0)
##     ## dr['kpca_gamma3'] = KernelPCA(n_components=2, kernel="sigmoid", gamma=0.3)
##     ## dr['kpca_gamma5'] = KernelPCA(n_components=2, kernel="cosine", gamma=0.3)
##     from sklearn.manifold import LocallyLinearEmbedding # Too bad
##     ## dr['lle3'] = LocallyLinearEmbedding(n_neighbors=3, n_components=2, eigen_solver='dense')
##     ## dr['lle5'] = LocallyLinearEmbedding(n_neighbors=5, n_components=2, eigen_solver='dense')
##     ## dr['lle7'] = LocallyLinearEmbedding(n_neighbors=7, n_components=2, eigen_solver='dense')

##     bv = {}
##     from sklearn import svm
##     bv['svm_gamma1'] = svm.OneClassSVM(nu=0.1, kernel='rbf', gamma=0.4)
##     bv['svm_gamma2'] = svm.OneClassSVM(nu=0.1, kernel='rbf', gamma=2.0)
##     bv['svm_gamma3'] = svm.OneClassSVM(nu=0.1, kernel='rbf', gamma=3.0)
##     bv['svm_gamma4'] = svm.OneClassSVM(nu=0.1, kernel='rbf', gamma=4.0)


##     # title for the plots
##     for idx, key in enumerate(dr.keys()):
##     ## for idx, key in enumerate(bv.keys()):
##         ml  = dr[key]
##         clf = bv['svm_gamma1'] #[key]
##         plt.subplot(2, 2, idx + 1)

##         # --------------- Dimension Reduction --------------------------
##         success_x = ml.fit_transform(success_data_array)
##         success_y = [1.0]*len(success_data_array)

##         failure_x = ml.transform(failure_data_array)
##         failure_y = [0.0]*len(failure_data_array)

##         all_x = ml.transform(all_data_array)

##         # ---------------- Boundary Visualization ----------------------
##         clf.fit(success_x, success_y)

##         # create a mesh to plot in
##         x_min, x_max = all_x[:, 0].min() - 0.2, all_x[:, 0].max() + 0.2
##         y_min, y_max = all_x[:, 1].min() - 0.2, all_x[:, 1].max() + 0.2
##         xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
##                              np.arange(y_min, y_max, h))
        
##         Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])

##         # Put the result into a color plot
##         Z = Z.reshape(xx.shape)

##         plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.Blues_r)
##         plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='orange')
##         plt.axis('off')

##         plt.title(key)

##         # ---------------- Sample Visualization ------------------------
##         if success_viz:
##             plt.scatter(success_x[:,0], success_x[:,1], c='b', label=None)

##         # Abnormal
##         if failure_viz:
##             legend_handles = []
##             m,n,k = np.shape(abnormalTestData)
##             for i in xrange(n):
##                 data_array = None
##                 for j in xrange(k):
##                     if data_array is None: data_array = abnormalTestData[:,i,j]
##                     else: data_array = np.vstack([data_array, abnormalTestData[:,i,j]])

##                 res = ml.transform(data_array)
##                 ## color = colors.next()

##                 cause = os.path.split(abnormalTestNameList[i])[-1].split('.pkl')[0].split('failure_')[-1]
##                 if 'falling' in cause: color = 'k'
##                 elif 'touching' == cause: color = 'r'
##                 elif 'slip' in cause: color = 'm'
##                 ## elif 'sound' in cause: color = 'g'
##                 else: color = 'k'
                    
                
##                 plt.scatter(res[:,0], res[:,1], c=color, marker='x', label=cause)
##                 ## legend_handles.append( h )

##             ## plt.legend(loc='upper right') #handles=legend_handles) #loc='upper right', 
            
##     if save_pdf:
##         fig.savefig('test.pdf')
##         fig.savefig('test.png')
##         os.system('cp test.p* ~/Dropbox/HRL/')        
##     else:
##         plt.show()

if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    p.add_option('--dataRenew', '--dr', action='store_true', dest='bDataRenew',
                 default=False, help='Renew pickle files.')
    p.add_option('--hmmRenew', '--hr', action='store_true', dest='bHMMRenew',
                 default=False, help='Renew HMM parameters.')

    p.add_option('--task', action='store', dest='task', type='string', default='pushing',
                 help='type the desired task name')

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
    
    p.add_option('--aeDataExtraction', '--ae', action='store_true', dest='bAEDataExtraction',
                 default=False, help='Extract auto-encoder data.')
    p.add_option('--aeDataExtractionPlot', '--aep', action='store_true', dest='bAEDataExtractionPlot',
                 default=False, help='Extract auto-encoder data and plot it.')

    p.add_option('--evaluation_all', '--ea', action='store_true', dest='bEvaluationAll',
                 default=False, help='Evaluate a classifier with cross-validation.')
    
    p.add_option('--renew', action='store_true', dest='bRenew',
                 default=False, help='Renew pickle files.')
    p.add_option('--savepdf', '--sp', action='store_true', dest='bSavePdf',
                 default=False, help='Save pdf files.')    
    p.add_option('--verbose', '--v', action='store_true', dest='bVerbose',
                 default=False, help='Print out.')

    # deprecated
    p.add_option('--statelikelihoodplot', '--slp', action='store_true', dest='bStateLikelihoodPlot',
                 default=False, help='Plot the log likelihoods over states.')
    p.add_option('--evaluation', '--e', action='store_true', dest='bEvaluation',
                 default=False, help='Evaluate a classifier.')
    p.add_option('--trainClassifier', '--tc', action='store_true', dest='bTrainClassifier',
                 default=False, help='Train a cost sensitive classifier.')
    p.add_option('--localization', '--ll', action='store_true', dest='bLocalization',
                 default=False, help='Extract local feature.')
    
    p.add_option('--pca', action='store_true', dest='bPCAPlot',
                 default=False, help='Plot pca result.')


    opt, args = p.parse_args()

    #---------------------------------------------------------------------------           
    # Run evaluation
    #---------------------------------------------------------------------------           
    rf_center     = 'kinEEPos'        
    scale         = 1.0
    # Dectection TEST 
    local_range    = 10.0    

    if opt.task == 'scooping':
        ## subjects = ['gatsbii']
        subjects = ['Wonyoung', 'Tom', 'lin', 'Ashwin', 'Song', 'Henry2'] #'Henry', 
        task     = opt.task    
        ## feature_list = ['unimodal_ftForce', 'crossmodal_targetEEDist', \
        ##                 'crossmodal_targetEEAng']
        feature_list = ['unimodal_audioWristRMS',\
                        ## 'unimodal_audioPower',\
                        ## 'unimodal_kinVel',\
                        'unimodal_ftForce',\
                        ##'unimodal_visionChange',\
                        ## 'unimodal_ppsForce',\
                        ##'unimodal_fabricForce',\
                        'crossmodal_targetEEDist', \
                        'crossmodal_targetEEAng']
        ## modality_list = ['kinematics', 'audioWrist', 'audio', 'fabric', 'ft', 'vision_artag', \
        ##                  'vision_change', 'pps']
        modality_list = ['kinematics', 'audioWrist', 'ft', 'vision_artag', \
                         'vision_change', 'pps']
        downSampleSize = 200

        save_data_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/RSS2016/'+task+'_data'
        raw_data_path  = '/home/dpark/hrl_file_server/dpark_data/anomaly/RSS2016/'

        data_param_dict= {'renew': opt.bDataRenew, 'rf_center': rf_center, 'local_range': local_range,\
                          'downSampleSize': 200, 'cut_data': [0,130], 'nNormalFold':4, 'nAbnormalFold':4,\
                          'feature_list': feature_list, 'nAugment': 0, 'lowVarDataRemv': False}
        AE_param_dict  = {'renew': False, 'switch': False, 'time_window': 4, 'filter': True, \
                          'layer_sizes':[64,32,16], 'learning_rate':1e-6, 'learning_rate_decay':1e-6, \
                          'momentum':1e-6, 'dampening':1e-6, 'lambda_reg':1e-6, \
                          'max_iteration':30000, 'min_loss':0.1, 'cuda':True, 'filter':True, 'filterDim':4}
        HMM_param_dict = {'renew': opt.bHMMRenew, 'nState': 20, 'cov': 5.0, 'scale': 4.0}
        SVM_param_dict = {'renew': False,}

        nPoints        = 20
        ROC_param_dict = {'methods': ['progress_time_cluster', 'svm','fixed'],\
                          'update_list': [],\
                          'nPoints': nPoints,\
                          'progress_param_range':-np.linspace(0., 10.0, nPoints), \
                          'svm_param_range': np.logspace(-4, 1.2, nPoints),\
                          'fixed_param_range': -np.logspace(0.0, 0.9, nPoints)+1.2,\
                          'cssvm_param_range': np.logspace(0.0, 2.0, nPoints) }
        param_dict = {'data_param': data_param_dict, 'AE': AE_param_dict, 'HMM': HMM_param_dict, \
                      'SVM': SVM_param_dict, 'ROC': ROC_param_dict}


    #---------------------------------------------------------------------------
    elif opt.task == 'feeding':
        
        subjects = ['Tom', 'lin', 'Ashwin', 'Song'] #'Wonyoung']
        task     = opt.task 
        feature_list = ['unimodal_audioWristRMS', 'unimodal_ftForce', 'crossmodal_artagEEDist', \
                        'crossmodal_artagEEAng'] #'unimodal_audioPower'
        modality_list   = ['ft'] #'kinematics', 'audioWrist', , 'vision_artag'

        save_data_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/RSS2016/'+task+'_data'
        raw_data_path  = '/home/dpark/hrl_file_server/dpark_data/anomaly/RSS2016/'
        downSampleSize = 200

        data_param_dict= {'renew': opt.bDataRenew, 'rf_center': rf_center, 'local_range': local_range,\
                          'downSampleSize': downSampleSize, 'cut_data': [0,170], \
                          'nNormalFold':4, 'nAbnormalFold':4,\
                          'feature_list': feature_list, 'nAugment': 0, 'lowVarDataRemv': False}
        AE_param_dict  = {'renew': False, 'switch': False, 'time_window': 4, 'filter': True, \
                          'layer_sizes':[64,32,16], 'learning_rate':1e-6, 'learning_rate_decay':1e-6, \
                          'momentum':1e-6, 'dampening':1e-6, 'lambda_reg':1e-6, \
                          'max_iteration':30000, 'min_loss':0.1, 'cuda':True, 'filter':True, 'filterDim':4,\
                          'add_option': 'bottleneck', 'add_feature': feature_list} 
        HMM_param_dict = {'renew': opt.bHMMRenew, 'nState': 25, 'cov': 5.0, 'scale': 4.0}
        SVM_param_dict = {'renew': False,}
        
        nPoints        = 20
        ROC_param_dict = {'methods': ['progress_time_cluster', 'svm','fixed'],\
                          'update_list': [],\
                          'nPoints': nPoints,\
                          'progress_param_range':-np.linspace(0., 10.0, nPoints), \
                          'svm_param_range': np.logspace(-4, 1.2, nPoints),\
                          'fixed_param_range': np.linspace(1.0, -3.0, nPoints),\
                          'cssvm_param_range': np.logspace(0.0, 2.0, nPoints) }
        param_dict = {'data_param': data_param_dict, 'AE': AE_param_dict, 'HMM': HMM_param_dict, \
                      'SVM': SVM_param_dict, 'ROC': ROC_param_dict}
            
    #---------------------------------------------------------------------------           
    elif opt.task == 'pushing':
        subjects = ['gatsbii']
        task     = opt.task    
        ## task    = 'pushing_microwave_black'    
        ## task    = 'pushing_microwave_white'    
        ## task    = 'pushing_lab_cabinet'    
        ## feature_list = ['unimodal_audioPower',\
        ##                 'unimodal_kinVel',\
        ##                 'unimodal_ftForce',\
        ##                 ##'unimodal_visionChange',\
        ##                 'unimodal_ppsForce',\
        ##                 ##'unimodal_fabricForce',\
        ##                 'crossmodal_targetEEDist', \
        ##                 'crossmodal_targetEEAng']
        ## feature_list  = ['unimodal_ftForce',\
        ##                  'crossmodal_targetEEDist',\
        ##                  'crossmodal_targetEEAng',\
        ##                  'unimodal_audioWristRMS'] #'unimodal_audioPower', ,
        feature_list = ['relativePose_artag_EE', \
                        'relativePose_artag_artag', \
                        'wristAudio', \
                        'ft', \
                        ]       
                         
        #modality_list = ['ft'] #'audio', , 'audioWrist' # only for data plot
        modality_list   = ['kinematics', 'audio', 'ft']

        save_data_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/RSS2016/'+task+'_data'
        save_data_path = os.path.expanduser('~')+\
          '/hrl_file_server/dpark_data/anomaly/RSS2016/'+task+'_data/AE'        
        raw_data_path  = '/home/dpark/hrl_file_server/dpark_data/anomaly/RSS2016/'
        downSampleSize = 200      

        data_param_dict= {'renew': opt.bDataRenew, 'rf_center': rf_center, 'local_range': local_range,\
                          'downSampleSize': downSampleSize, 'cut_data': [0,200], \
                          'nNormalFold':3, 'nAbnormalFold':3,\
                          'feature_list': feature_list, 'nAugment': 1, 'lowVarDataRemv': False }
        AE_param_dict  = {'renew': False, 'switch': True, 'time_window': 4, 'filter': True, \
                          'layer_sizes':[64,32,16], 'learning_rate':1e-6, 'learning_rate_decay':1e-6, \
                          'momentum':1e-6, 'dampening':1e-6, 'lambda_reg':1e-6, \
                          'max_iteration':30000, 'min_loss':0.1, 'cuda':True, 'filter':True, 'filterDim':4, \
                          'add_option': 'featureToBottleneck'}
        HMM_param_dict = {'renew': opt.bHMMRenew, 'nState': 25, 'cov': 4.0, 'scale': 5.0}
        SVM_param_dict = {'renew': False,}

        nPoints        = 20
        ROC_param_dict = {'methods': ['progress_time_cluster', 'svm','fixed'],\
                          'update_list': [],\
                          'nPoints': nPoints,\
                          'progress_param_range':np.linspace(-1., -10., nPoints), \
                          'svm_param_range': np.logspace(-4, 1.2, nPoints),\
                          'fixed_param_range': np.linspace(1.0, -3.0, nPoints),\
                          'cssvm_param_range': np.logspace(0.0, 2.0, nPoints) }        
        param_dict = {'data_param': data_param_dict, 'AE': AE_param_dict, 'HMM': HMM_param_dict, \
                      'SVM': SVM_param_dict, 'ROC': ROC_param_dict}
    else:
        print "Selected task name is not available."
        sys.exit()

    #---------------------------------------------------------------------------           
    #---------------------------------------------------------------------------           
    #---------------------------------------------------------------------------           
    #---------------------------------------------------------------------------           
    #---------------------------------------------------------------------------           
    

    if opt.bRawDataPlot or opt.bInterpDataPlot:
        '''
        Before localization: Raw data plot
        After localization: Raw or interpolated data plot
        '''
        successData = True
        failureData = True
        
        data_plot(subjects, task, raw_data_path, save_data_path,\
                  downSampleSize=downSampleSize, \
                  local_range=local_range, rf_center=rf_center, \
                  raw_viz=opt.bRawDataPlot, interp_viz=opt.bInterpDataPlot, save_pdf=opt.bSavePdf,\
                  successData=successData, failureData=failureData,\
                  modality_list=modality_list, data_renew=opt.bDataRenew, verbose=opt.bVerbose)

    elif opt.bDataSelection:
        '''
        Manually select and filter bad data out
        '''
        ## modality_list   = ['kinematics', 'audioWrist','audio', 'fabric', 'ft', \
        ##                    'vision_artag', 'vision_change', 'pps']
        success_viz = True
        failure_viz = True

        data_selection(subjects, task, raw_data_path, save_data_path,\
                       downSampleSize=downSampleSize, \
                       local_range=local_range, rf_center=rf_center, \
                       success_viz=success_viz, failure_viz=failure_viz,\
                       raw_viz=opt.bRawDataPlot, save_pdf=opt.bSavePdf,\
                       modality_list=modality_list, data_renew=opt.bDataRenew, verbose=opt.bVerbose)        

    elif opt.bFeaturePlot:
        success_viz = True
        failure_viz = True

        dm.getDataSet(subjects, task, raw_data_path, save_data_path,
                      data_param_dict['rf_center'], data_param_dict['local_range'],\
                      downSampleSize=data_param_dict['downSampleSize'], scale=scale, \
                      success_viz=success_viz, failure_viz=failure_viz,\
                      ae_data=AE_param_dict['switch'],\
                      data_ext=data_param_dict['lowVarDataRemv'],\
                      cut_data=data_param_dict['cut_data'],
                      save_pdf=opt.bSavePdf, solid_color=True,\
                      feature_list=data_param_dict['feature_list'], data_renew=opt.bDataRenew)

    elif opt.bAEDataExtraction:
        aeDataExtraction(subjects, task, raw_data_path, save_data_path, param_dict, verbose=opt.bVerbose)

    elif opt.bAEDataExtractionPlot:
        success_viz = True
        failure_viz = True
        aeDataExtraction(subjects, task, raw_data_path, save_data_path, param_dict,\
                         success_viz=success_viz, failure_viz=failure_viz,\
                         verbose=opt.bVerbose)


    elif opt.bLikelihoodPlot:
        threshold    = 0.0
        smooth       = False

        likelihoodOfSequences(subjects, task, raw_data_path, save_data_path, param_dict,\
                              decision_boundary_viz=True, \
                              useTrain=True, useNormalTest=False, useAbnormalTest=True,\
                              useTrain_color=False, useNormalTest_color=False, useAbnormalTest_color=False,\
                              hmm_renew=opt.bHMMRenew, data_renew=opt.bDataRenew, save_pdf=opt.bSavePdf)
                              
    elif opt.bStateLikelihoodPlot:
        scale           = 100.0
        threshold       = 0.0
        smooth          = False #only related with expLoglikelihood
        ## cluster_type    = 'time'
        cluster_type    = 'state'
        classifier_type = 'new'
        nState        = 10
        
        stateLikelihoodPlot(subjects, task, raw_data_path, save_data_path, rf_center, local_range,\
                            downSampleSize=downSampleSize, \
                            feature_list=feature_list, \
                            nState=nState, threshold=threshold, smooth=smooth, cluster_type=cluster_type,\
                            classifier_type=classifier_type,\
                            useTrain=True, useNormalTest=False, useAbnormalTest=True,\
                            useTrain_color=False, useNormalTest_color=False, useAbnormalTest_color=False,\
                            hmm_renew=opt.bHMMRenew, data_renew=opt.bDataRenew, save_pdf=opt.bSavePdf)




    elif opt.bEvaluationAll:                
        evaluation_all(subjects, task, raw_data_path, save_data_path, param_dict, save_pdf=opt.bSavePdf, \
                       verbose=opt.bVerbose)


    #-------------------------------------------------------------------------------------------------
    elif opt.bTrainClassifier:
        rf_center    = 'kinEEPos'
        ## rf_center    = 'kinForearmPos'
        feature_list = ['unimodal_audioPower',\
                        #'unimodal_kinVel',\
                        'unimodal_ftForce',\
                        #'unimodal_visionChange',\
                        'unimodal_ppsForce',\
                        'unimodal_fabricForce',\
                        'crossmodal_targetEEDist', \
                        'crossmodal_targetEEAng'
                        ]
        local_range = 0.15

        nState    = 10
        threshold = 0.0
        smooth          = False #only related with expLoglikelihood
        ## cluster_type    = 'time'
        ## cluster_type = 'state'
        cluster_type = 'none'
        classifier_type = 'new'
        
        trainClassifierSVM(subjects, task, raw_data_path, save_data_path, rf_center, local_range,\
                           downSampleSize=downSampleSize, \
                           feature_list=feature_list, \
                           nState=nState, threshold=threshold, smooth=smooth, cluster_type=cluster_type,\
                           classifier_type=classifier_type,\
                           useTrain=True, useNormalTest=False, useAbnormalTest=True,\
                           useTrain_color=False, useNormalTest_color=False, useAbnormalTest_color=False,\
                           hmm_renew=opt.bHMMRenew, data_renew=opt.bDataRenew, save_pdf=opt.bSavePdf)

        ## trainClassifier([subject], task, raw_data_path, save_data_path, rf_center, local_range,\
        ##                 downSampleSize=downSampleSize, \
        ##                 feature_list=feature_list, \
        ##                 nState=nState, threshold=threshold, smooth=smooth, cluster_type=cluster_type,\
        ##                 classifier_type=classifier_type,\
        ##                 useTrain=True, useNormalTest=False, useAbnormalTest=True,\
        ##                 useTrain_color=False, useNormalTest_color=False, useAbnormalTest_color=False,\
        ##                 hmm_renew=opt.bHMMRenew, data_renew=opt.bDataRenew, save_pdf=opt.bSavePdf)

    elif opt.bEvaluation:
        rf_center    = 'kinEEPos'
        ## rf_center    = 'kinForearmPos'
        local_range  = 0.15

        threshold       = 0.0
        smooth          = False #only related with expLoglikelihood
        ## cluster_type = 'time'
        ## cluster_type = 'state'
        cluster_type    = 'none'
        classifier_type = 'cssvm'
        
        evaluation(subjects, task, raw_data_path, save_data_path, rf_center, local_range,\
                   downSampleSize=downSampleSize, \
                   feature_list=feature_list, \
                   nState=nState, threshold=threshold, smooth=smooth, cluster_type=cluster_type,\
                   classifier_type=classifier_type,\
                   hmm_renew=opt.bHMMRenew, data_renew=opt.bDataRenew, save_pdf=opt.bSavePdf, \
                   verbose=opt.bVerbose)
        
        
    ## else:
    ##     nState         = 10
    ##     cov_mult       = 5.0       
    ##     anomaly_offset = -20.0        
    ##     check_methods = ['progress']
    ##     evaluation_all([subject], task, check_methods, feature_list, \
    ##                    save_data_path, downSampleSize=downSampleSize, \
    ##                    nState=nState, cov_mult=cov_mult, anomaly_offset=anomaly_offset, local_range=local_range,\
    ##                    data_renew=opt.bDataRenew, hmm_renew=opt.bHMMRenew, viz=viz)    

    elif opt.bPCAPlot:
        '''
        deprecated?
        '''       
        ## rf_center    = 'kinEEPos'
        ## feature_list = ['unimodal_audioPower',\
        ##                 'unimodal_kinVel',\
        ##                 'unimodal_ftForce',\
        ##                 'unimodal_visionChange',\
        ##                 'unimodal_ppsForce',\
        ##                 'unimodal_fabricForce',\
        ##                 'crossmodal_targetEEDist', \
        ##                 'crossmodal_targetEEAng']
        task         = 'touching'    
        rf_center    = 'kinForearmPos'
        feature_list = ['unimodal_audioPower',\
                        'unimodal_kinVel',\
                        'unimodal_visionChange',\
                        'unimodal_fabricForce']
        local_range = 0.15
        success_viz = True
        failure_viz = True
                        
        pca_plot(subjects, task, raw_data_path, save_data_path, rf_center, local_range,\
                  downSampleSize=downSampleSize, \
                  success_viz=success_viz, failure_viz=failure_viz,\
                  save_pdf=opt.bSavePdf,
                  feature_list=feature_list, data_renew=opt.bDataRenew)


    else:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        from scipy.stats import poisson
        mu = 0.6
        mean, var, skew, kurt = poisson.stats(mu, moments='mvsk')
        x = np.arange(0.0, 30.0)
        ax.plot(x, poisson.pmf(x, mu), 'bo', ms=8, label='poisson pmf')
        
        plt.show()
