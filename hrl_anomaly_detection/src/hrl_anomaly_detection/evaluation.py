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
import os, sys, copy
import random
import numpy as np
import scipy
from joblib import Parallel, delayed

# util
import hrl_lib.util as ut
from hrl_anomaly_detection.util import *
from hrl_anomaly_detection.util_viz import *
from hrl_anomaly_detection import data_manager as dm
from hrl_anomaly_detection import util as util
## from hrl_anomaly_detection.optimizeParam import *

# learning
from hrl_anomaly_detection.hmm import learning_hmm as hmm
import hrl_anomaly_detection.classifiers.classifier as cf

# visualization
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec

import itertools
colors = itertools.cycle(['g', 'm', 'c', 'k', 'y','r', 'b', ])
shapes = itertools.cycle(['x','v', 'o', '+'])

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42 


def evaluation_step_noise(subject_names, task_name, raw_data_path, processed_data_path, param_dict,\
                          step_mag, pkl_prefix,\
                          data_renew=False, save_pdf=False, verbose=False, debug=False,\
                          no_plot=False, delay_plot=False, find_param=False, all_plot=False):

    ## Parameters
    # data
    data_dict  = param_dict['data_param']
    data_renew = data_dict['renew']
    dim        = len(data_dict['handFeatures'])
    # AE
    AE_dict    = param_dict['AE']
    # HMM
    HMM_dict   = param_dict['HMM']
    nState     = HMM_dict['nState']
    cov        = HMM_dict['cov']
    add_logp_d = HMM_dict.get('add_logp_d', False)
    # SVM
    SVM_dict   = param_dict['SVM']
    # ROC
    ROC_dict = param_dict['ROC']

    # reference data #TODO
    ref_data_path = os.path.join(processed_data_path, '../'+str(data_dict['downSampleSize'])+\
                                 '_'+str(dim))

    #------------------------------------------
    # Get features
    if os.path.isdir(processed_data_path) is False:
        os.system('mkdir -p '+processed_data_path)

    crossVal_pkl = os.path.join(ref_data_path, 'cv_'+task_name+'.pkl')
    
    if os.path.isfile(crossVal_pkl) and data_renew is False:
        d = ut.load_pickle(crossVal_pkl)
        kFold_list  = d['kFoldList']
    else: sys.exit()

    #-----------------------------------------------------------------------------------------
    # parameters
    startIdx    = 4
    method_list = ROC_dict['methods'] 
    nPoints     = ROC_dict['nPoints']

    successData = d['successData']
    failureData = d['failureData']
    param_dict2  = d['param_dict']
    if 'timeList' in param_dict2.keys():
        timeList = param_dict2['timeList'][startIdx:]
    else: timeList = None

    #-----------------------------------------------------------------------------------------
    # Training HMM, and getting classifier training and testing data
    for idx, (normalTrainIdx, abnormalTrainIdx, normalTestIdx, abnormalTestIdx) \
      in enumerate(kFold_list):

        if verbose: print idx, " : training hmm and getting classifier training and testing data"            
        ref_modeling_pkl = os.path.join(ref_data_path, 'hmm_'+task_name+'_'+str(idx)+'.pkl')
        if os.path.isfile(ref_modeling_pkl) is False:
            print ref_modeling_pkl
            print "No reference modeling file exists"
            sys.exit()
        
        modeling_pkl = os.path.join(processed_data_path, 'hmm_'+pkl_prefix+'_'+str(idx)+'.pkl')
        if not (os.path.isfile(modeling_pkl) is False or HMM_dict['renew'] or data_renew): continue

        # dim x sample x length
        normalTestData    = successData[:, normalTestIdx, :] * HMM_dict['scale']

        # training hmm
        if verbose: print "start to fit hmm"
        dd = ut.load_pickle(ref_modeling_pkl)
        nEmissionDim = dd['nEmissionDim']
        nLength      = len(normalTestData[0][0]) - startIdx

        # Classifier test data
        # random step noise
        abnormalTestData = copy.copy(normalTestData)
        step_idx_l = []
        for i in xrange(len(normalTestData[0])):
            step_idx_l.append(None)
        for i in xrange(len(abnormalTestData[0])):
            start_idx = np.random.randint(startIdx, nLength*2/3, 1)[0]
            dim_idx   = np.random.randint(0, len(abnormalTestData))
            
            abnormalTestData[dim_idx,i,start_idx:] += step_mag
            step_idx_l.append(start_idx)

        ml = hmm.learning_hmm(nState, nEmissionDim, verbose=False)
        ml.set_hmm_object(dd['A'], dd['B'], dd['pi'])            

        # Classifier test data
        ll_classifier_test_X, ll_classifier_test_Y, ll_classifier_test_idx =\
          hmm.getHMMinducedFeaturesFromRawFeatures(ml, normalTestData, abnormalTestData, startIdx, \
                                                   add_logp_d=add_logp_d)

        #-----------------------------------------------------------------------------------------
        d = {}
        d['nEmissionDim'] = nEmissionDim
        d['A']            = dd['A'] 
        d['B']            = dd['B'] 
        d['pi']           = dd['pi']
        d['F']            = dd['F']
        d['nState']       = nState
        d['startIdx']     = startIdx
        d['ll_classifier_train_X']   = dd['ll_classifier_train_X']
        d['ll_classifier_train_Y']   = dd['ll_classifier_train_Y']
        d['ll_classifier_train_idx'] = dd['ll_classifier_train_idx']
        d['ll_classifier_test_X']    = ll_classifier_test_X
        d['ll_classifier_test_Y']    = ll_classifier_test_Y            
        d['ll_classifier_test_idx']  = ll_classifier_test_idx
        d['nLength']      = nLength
        d['step_idx_l']   = step_idx_l
        ut.save_pickle(d, modeling_pkl)

    ## fig = plt.figure()
    ## ## modeling_pkl = os.path.join(processed_data_path, modeling_pkl_prefix+'_'+str(0)+'.pkl')
    ## ## modeling_pkl = os.path.join(ref_data_path, 'hmm_'+task_name+'_'+str(0)+'.pkl')
    ## d = ut.load_pickle(modeling_pkl)
    ## ll_classifier_test_X = d['ll_classifier_test_X']
    ## ll_classifier_test_Y = d['ll_classifier_test_Y']
    ## print np.shape(ll_classifier_test_Y)
    ## for i in xrange(len(ll_classifier_test_X)):        
    ##     if ll_classifier_test_Y[i][0] > 0: continue
    ##     print "test normal: ", np.shape(ll_classifier_test_X[i])        
    ##     x = ll_classifier_test_X[i]
    ##     plt.plot(np.argmax(np.array(x)[:,2:],axis=1), np.array(x)[:,0], 'b-')
    ## plt.show()
    ## sys.exit()

    #-----------------------------------------------------------------------------------------
    roc_pkl = os.path.join(processed_data_path, 'roc_'+pkl_prefix+'.pkl')
    if os.path.isfile(roc_pkl) is False or HMM_dict['renew'] or SVM_dict['renew']: ROC_data = {}
    else: ROC_data = ut.load_pickle(roc_pkl)
    ROC_data = util.reset_roc_data(ROC_data, method_list, ROC_dict['update_list'], nPoints)

    osvm_data = None ; bpsvm_data = None
    if 'osvm' in method_list  and ROC_data['osvm']['complete'] is False:
        modeling_pkl_prefix = os.path.join(processed_data_path, 'hmm_'+pkl_prefix)
        osvm_data = dm.getPCAData(len(kFold_list), crossVal_pkl, \
                                  window=SVM_dict['raw_window_size'],
                                  use_test=True, use_pca=False, \
                                  step_anomaly_info=(modeling_pkl_prefix, step_mag/HMM_dict['scale']) )

    ## kFold_list = kFold_list[:1]
                                  
    # parallelization
    if debug: n_jobs=1
    else: n_jobs=-1
    r = Parallel(n_jobs=n_jobs, verbose=10)(delayed(cf.run_classifiers)( idx, processed_data_path, task_name, \
                                                                 method, ROC_data, \
                                                                 ROC_dict, AE_dict, \
                                                                 SVM_dict, HMM_dict, \
                                                                 raw_data=(osvm_data,bpsvm_data),\
                                                                 startIdx=startIdx, nState=nState,\
                                                                 modeling_pkl_prefix='hmm_'+pkl_prefix,\
                                                                 delay_estimation=True) \
                                                                 for idx in xrange(len(kFold_list)) \
                                                                 for method in method_list )
    print "finished to run run_classifiers"

    ROC_data = util.update_roc_data(ROC_data, r, nPoints, method_list)
    ut.save_pickle(ROC_data, roc_pkl)
        
    #-----------------------------------------------------------------------------------------
    # ---------------- ROC Visualization ----------------------
    if all_plot:
        task_list = ["pushing_microblack", "pushing_microwhite", "pushing_toolcase", "scooping", "feeding"]
    else:
        ## roc_info(method_list, ROC_data, nPoints, delay_plot=delay_plot, no_plot=no_plot, save_pdf=save_pdf)
        delay_info(method_list, ROC_data, nPoints, no_plot=no_plot, save_pdf=save_pdf, timeList=timeList)


def evaluation_acc_param(subject_names, task_name, raw_data_path, processed_data_path, param_dict,\
                         step_mag, pkl_prefix,\
                         data_renew=False, save_pdf=False, verbose=False, debug=False,\
                         no_plot=False, delay_plot=False, find_param=False, all_plot=False):

    ## Parameters
    # data
    data_dict  = param_dict['data_param']
    data_renew = data_dict['renew']
    dim        = len(data_dict['handFeatures'])
    # AE
    AE_dict    = param_dict['AE']
    # HMM
    HMM_dict   = param_dict['HMM']
    nState     = HMM_dict['nState']
    cov        = HMM_dict['cov']
    add_logp_d = HMM_dict.get('add_logp_d', False)
    # SVM
    SVM_dict   = param_dict['SVM']
    # ROC
    ROC_dict = param_dict['ROC']

    # reference data #TODO
    ref_data_path = os.path.join(processed_data_path, '../'+str(data_dict['downSampleSize'])+\
                                 '_'+str(dim))
    pkl_target_prefix = pkl_prefix.split('_')[0] #+'_0.05'

    #------------------------------------------
    # Get features
    if os.path.isdir(processed_data_path) is False:
        os.system('mkdir -p '+processed_data_path)

    crossVal_pkl = os.path.join(ref_data_path, 'cv_'+task_name+'.pkl')
    
    if os.path.isfile(crossVal_pkl) and data_renew is False:
        d = ut.load_pickle(crossVal_pkl)
        kFold_list  = d['kFoldList']
    else: sys.exit()

    #-----------------------------------------------------------------------------------------
    # parameters
    startIdx    = 4
    method_list = ROC_dict['methods'] 
    nPoints     = ROC_dict['nPoints']

    successData = d['successData']
    failureData = d['successData'] #d['failureData']
    param_dict2  = d['param_dict']
    if 'timeList' in param_dict2.keys():
        timeList = param_dict2['timeList'][startIdx:]
    else: timeList = None

    ## kFold_list = kFold_list[:1]
    
    score_list  = [ [[] for i in xrange(len(method_list))] for j in xrange(3) ]
    delay_list = [ [[] for i in xrange(len(method_list))] for j in xrange(3) ]

    nLength = len(failureData[0][0])
    step_idx_l = []
    for i in xrange(len(failureData[0])):
        start_idx = np.random.randint(startIdx, nLength*2/3, 1)[0]
        dim_idx   = np.random.randint(0, len(failureData))
        step_mag  = np.random.uniform(0.05, 0.3)
        
        failureData[dim_idx,i,start_idx:] += step_mag
        step_idx_l.append(start_idx)


    #-----------------------------------------------------------------------------------------
    # Training HMM, and getting classifier training and testing data
    for idx, (normalTrainIdx, abnormalTrainIdx, normalTestIdx, abnormalTestIdx) \
      in enumerate(kFold_list):

        if verbose: print idx, " : training hmm and getting classifier training and testing data"            
        
        # dim x sample x length
        normalTrainData   = successData[:, normalTrainIdx, :] * HMM_dict['scale']
        abnormalTrainData = failureData[:, normalTrainIdx, :] * HMM_dict['scale']
        step_idx_l_train  = np.array(step_idx_l)[normalTrainIdx]
        ## abnormalTrainData = failureData[:, abnormalTrainIdx, :] * HMM_dict['scale']        
        ## abnormalTrainData = copy.copy(normalTrainData)
        ## normalTestData    = successData[:, normalTestIdx, :] * HMM_dict['scale']


        # training hmm
        if verbose: print "start to fit hmm"
        nEmissionDim = len(normalTrainData)
        cov_mult     = [cov]*(nEmissionDim**2)
        nLength      = len(normalTrainData[0][0]) - startIdx

        ## # Random Step Noise to crossvalidation data
        ## for i in xrange(len(abnormalTrainData[0])):
        ##     start_idx = np.random.randint(0, nLength/2, 1)[0]
        ##     if start_idx < startIdx: start_idx=startIdx
        ##     abnormalTrainData[:,i,start_idx:] += step_mag
            

        from sklearn import cross_validation
        normal_folds = cross_validation.KFold(len(normalTrainData[0]), n_folds=2, shuffle=True)

        for iidx, (train_fold, test_fold) in enumerate(normal_folds):

            modeling_pkl = os.path.join(processed_data_path, \
                                        'hmm_'+pkl_target_prefix+'_'+str(idx)+'_'+str(iidx)+'.pkl')
            if not (os.path.isfile(modeling_pkl) is False or HMM_dict['renew'] or data_renew): continue
            
            t_normalTrainData   = normalTrainData[:,train_fold]
            t_abnormalTrainData = abnormalTrainData #[:,train_fold]
            t_normalTestData    = normalTrainData[:,test_fold]
            t_abnormalTestData  = abnormalTrainData #[:,test_fold]

            #-----------------------------------------------------------------------------------------
            # Full co-variance
            #-----------------------------------------------------------------------------------------
            ml  = hmm.learning_hmm(nState, nEmissionDim, verbose=verbose) 
            if data_dict['handFeatures_noise']:
                ret = ml.fit(t_normalTrainData+\
                             np.random.normal(0.0, 0.03, np.shape(t_normalTrainData) )*HMM_dict['scale'], \
                             cov_mult=cov_mult, use_pkl=False)
            else:
                ret = ml.fit(t_normalTrainData, cov_mult=cov_mult, use_pkl=False)
            if ret == 'Failure': sys.exit()

            # Classifier training data
            ll_classifier_train_X, ll_classifier_train_Y, ll_classifier_train_idx =\
              hmm.getHMMinducedFeaturesFromRawFeatures(ml, t_normalTrainData, t_abnormalTrainData, startIdx, add_logp_d)

            # Classifier test data
            ll_classifier_test_X, ll_classifier_test_Y, ll_classifier_test_idx =\
              hmm.getHMMinducedFeaturesFromRawFeatures(ml, t_normalTestData, t_abnormalTestData, startIdx, add_logp_d)
           
            #-----------------------------------------------------------------------------------------
            d = {}
            d['nEmissionDim'] = ml.nEmissionDim
            d['A']            = ml.A 
            d['B']            = ml.B 
            d['pi']           = ml.pi
            d['F']            = ml.F
            d['nState']       = nState
            d['startIdx']     = startIdx
            d['nLength']      = nLength
            d['ll_classifier_train_X']       = ll_classifier_train_X
            d['ll_classifier_train_Y']       = ll_classifier_train_Y            
            d['ll_classifier_train_idx']     = ll_classifier_train_idx
            d['ll_classifier_test_X']        = ll_classifier_test_X
            d['ll_classifier_test_Y']        = ll_classifier_test_Y            
            d['ll_classifier_test_idx']      = ll_classifier_test_idx
            d['step_idx_l']   = step_idx_l_train
            ut.save_pickle(d, modeling_pkl)


    #-----------------------------------------------------------------------------------------
        roc_pkl = os.path.join(processed_data_path, 'roc_'+pkl_target_prefix+'_'+str(idx)+'.pkl')

        if os.path.isfile(roc_pkl) is False or HMM_dict['renew'] or SVM_dict['renew']: ROC_data = {}
        else: ROC_data = ut.load_pickle(roc_pkl)
        ROC_data = util.reset_roc_data(ROC_data, method_list, ROC_dict['update_list'], nPoints)

        osvm_data = None ; bpsvm_data = None
        if 'osvm' in method_list  and ROC_data['osvm']['complete'] is False:
            normalTrainData   = successData[:, normalTrainIdx, :] 
            abnormalTrainData = failureData[:, normalTrainIdx, :]
            ## abnormalTrainData = failureData[:, abnormalTrainIdx, :]

            fold_list = []
            for train_fold, test_fold in normal_folds:
                fold_list.append([train_fold, test_fold])

            normalFoldData = (fold_list, normalTrainData, abnormalTrainData)
            
            osvm_data = dm.getPCAData(len(fold_list), normalFoldData=normalFoldData, \
                                      window=SVM_dict['raw_window_size'],
                                      use_test=True, use_pca=False, step_anomaly_info=step_idx_l)
            

        # parallelization
        if debug: n_jobs=1
        else: n_jobs=-1
        r = Parallel(n_jobs=n_jobs, verbose=50)(delayed(cf.run_classifiers)( iidx, processed_data_path, task_name, \
                                                                             method, ROC_data, \
                                                                             ROC_dict, AE_dict, \
                                                                             SVM_dict, HMM_dict, \
                                                                             raw_data=(osvm_data,bpsvm_data),\
                                                                             startIdx=startIdx, nState=nState, \
                                                                             modeling_pkl_prefix=\
                                                                             'hmm_'+pkl_target_prefix+'_'+str(idx))\
                                                                             for iidx in xrange(len(normal_folds))
                                                                             for method in method_list )

        l_data = r
        print "finished to run run_classifiers"

        ROC_data = util.update_roc_data(ROC_data, l_data, nPoints, method_list)
        ut.save_pickle(ROC_data, roc_pkl)

        #-----------------------------------------------------------------------------------------
        # ---------------- ROC Visualization ----------------------
        best_param_idx = getBestParamIdx(method_list, ROC_data, nPoints, verbose=False)

        print method_list
        print best_param_idx

        roc_pkl = os.path.join(processed_data_path, 'roc_'+pkl_prefix+'.pkl')
        ROC_data = ut.load_pickle(roc_pkl)
        scores, delays = cost_info(best_param_idx, method_list, ROC_data, nPoints, \
                                   timeList=timeList, verbose=False)
        ## scores, delays = cost_info_with_max_tpr(method_list, ROC_data, nPoints, \
        ##                                         timeList=timeList, verbose=False)

        for i in xrange(len(method_list)):
            for j in xrange(3):
                score_list[j][i].append( scores[j][i] )
                delay_list[j][i] += delays[j][i]
                ## print np.shape(score_list[j][i]), np.shape(delay_list[j][i])

    if no_plot is False:
        plotCostDelay(method_list, score_list, delay_list, save_pdf=save_pdf)
        
