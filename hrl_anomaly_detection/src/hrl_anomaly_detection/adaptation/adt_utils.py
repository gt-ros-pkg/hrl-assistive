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
import numpy as np
import scipy
import hrl_lib.util as ut
from joblib import Parallel, delayed

# Private utils
from hrl_anomaly_detection.util import *
from hrl_anomaly_detection.util_viz import *
from hrl_anomaly_detection import data_manager as dm
from hrl_anomaly_detection import util as util
from hrl_execution_monitor import util as autil

# Private learners
from hrl_anomaly_detection.hmm import learning_hmm as hmm


def saveAHMMFeatures(od, td, task_name, processed_data_path, HMM_dict, ADT_dict, noise_mag,
                     pkl_prefix, startIdx=4, tgt_hmm_idx=0):
    
    nState = HMM_dict['nState']
    random.seed(3334)
    np.random.seed(3334)

    kFold_list = od['kFoldList']
    normalTrainIdx, abnormalTrainIdx, normalTestIdx, abnormalTestIdx = kFold_list[tgt_hmm_idx]

    # person-wise indices from normal training data
    nor_train_inds = [ np.arange(len(kFold_list[i][2])) for i in xrange(len(kFold_list)) ]
    for i in xrange(1,len(nor_train_inds)):
        nor_train_inds[i] += (nor_train_inds[i-1][-1]+1)

    # Split test data to two groups
    n_AHMM_sample   = ADT_dict['n_pTrain']
    n_AHMM_test_idx = 10
    for idx in xrange(len(td['successDataList'])): # per person

        ## if idx != 4: continue
        inc_model_pkl = os.path.join(processed_data_path, pkl_prefix+'_'+str(idx)+'.pkl')
        if os.path.isfile(inc_model_pkl) and HMM_dict['renew'] is False and ADT_dict['HMM_renew'] is False :
            print idx, " : updated hmm exists"
            continue

        normalTestData   = np.array(td['successDataList'][idx]) * HMM_dict['scale'] 
        abnormalTestData = np.array(td['failureDataList'][idx]) * HMM_dict['scale']

        X_ptrain  = copy.deepcopy(normalTestData[:,:n_AHMM_sample])
        noise_arr = np.random.normal(0.0, noise_mag, np.shape(X_ptrain))*HMM_dict['scale']
        nLength   = len(normalTestData[0][0]) - startIdx

        # Load original hmm induced feature dictionary
        model_pkl = os.path.join(processed_data_path, 'hmm_'+task_name+'_'+str(tgt_hmm_idx)+'.pkl')
        d         = ut.load_pickle(model_pkl)

        # Update
        ml = hmm.learning_hmm(nState, d['nEmissionDim'])
        ml.set_hmm_object(d['A'], d['B'], d['pi'], d['out_a_num'], d['vec_num'], \
                          d['mat_num'], d['u_denom'])

        if ADT_dict['HMM'] == 'adapt':
            ret = ml.partial_fit(X_ptrain+noise_arr, learningRate=ADT_dict['lr'],
                                 max_iter=ADT_dict['max_iter'], nrSteps=ADT_dict['nrSteps'])
        elif ADT_dict['HMM'] == 'renew':
            ret = ml.fit(X_ptrain+noise_arr)
        else: ret = 0
            
        try:
            if np.isnan(ret):
                print "kFold_list ........ partial fit error... ", ret
                sys.exit()
        except:
            print ret
            return None
            sys.exit()

        # Classifier test data
        n_jobs=-1
        if ADT_dict['HMM'] is 'old' and ADT_dict['CLF'] is 'old':
            ll_classifier_train_X = copy.deepcopy(d['ll_classifier_train_X'])
            ll_classifier_train_Y = copy.deepcopy(d['ll_classifier_train_Y'])
            ll_classifier_train_idx = copy.deepcopy(d['ll_classifier_train_idx'])
        elif ADT_dict['CLF'] is not 'renew':        
            ll_classifier_train_X, ll_classifier_train_Y, ll_classifier_train_idx =\
              hmm.getHMMinducedFeaturesFromRawFeatures(ml, copy.deepcopy(od['successData'])*HMM_dict['scale'],
                                                       startIdx=startIdx, n_jobs=n_jobs)
              
        if ADT_dict['CLF'] is 'adapt' or ADT_dict['CLF'] is 'renew':        
            ll_classifier_ptrain_X, ll_classifier_ptrain_Y, ll_classifier_ptrain_idx =\
              hmm.getHMMinducedFeaturesFromRawFeatures(ml, copy.deepcopy(normalTestData[:,:n_AHMM_sample]),
                                                       startIdx=startIdx, \
                                                       n_jobs=n_jobs)
        else:
            ll_classifier_ptrain_X = None
            ll_classifier_ptrain_Y = None
            ll_classifier_ptrain_idx = None
            
        ll_classifier_test_X, ll_classifier_test_Y, ll_classifier_test_idx =\
          hmm.getHMMinducedFeaturesFromRawFeatures(ml, copy.deepcopy(normalTestData[:,n_AHMM_test_idx:]),
                                                   copy.deepcopy(abnormalTestData), \
                                                   startIdx, n_jobs=n_jobs)

        ## if success_files is not None:
        ##     ll_classifier_test_labels = [success_files[i] for i in normalTestIdx[n_AHMM_sample:]]
        ##     ll_classifier_test_labels += [failure_files[i] for i in abnormalTestIdx]
        ## else:
        ll_classifier_test_labels = None

        #-----------------------------------------------------------------------------------------
        dd = {}
        dd['nEmissionDim'] = ml.nEmissionDim
        dd['A']            = ml.A 
        dd['B']            = ml.B 
        dd['pi']           = ml.pi
        dd['F']            = ml.F
        dd['nState']       = nState
        dd['startIdx']     = startIdx

        if ADT_dict['CLF'] == 'renew':
            dd['ll_classifier_train_X']  = ll_classifier_ptrain_X
            dd['ll_classifier_train_Y']  = ll_classifier_ptrain_Y            
            dd['ll_classifier_train_idx']= ll_classifier_ptrain_idx
        else:
            dd['ll_classifier_train_X']  = ll_classifier_train_X
            dd['ll_classifier_train_Y']  = ll_classifier_train_Y            
            dd['ll_classifier_train_idx']= ll_classifier_train_idx
        
        dd['ll_classifier_ptrain_X']  = ll_classifier_ptrain_X
        dd['ll_classifier_ptrain_Y']  = ll_classifier_ptrain_Y            
        dd['ll_classifier_ptrain_idx']= ll_classifier_ptrain_idx
        dd['ll_classifier_test_X']   = ll_classifier_test_X
        dd['ll_classifier_test_Y']   = ll_classifier_test_Y            
        dd['ll_classifier_test_idx'] = ll_classifier_test_idx
        dd['ll_classifier_test_labels'] = ll_classifier_test_labels
        dd['nLength']      = nLength
        dd['scale']        = HMM_dict['scale']
        dd['cov']          = HMM_dict['cov']
        dd['nor_train_inds'] = nor_train_inds

        dd['ll_window_train_X'] = d.get('ll_window_train_X', None)
        dd['ll_window_train_Y'] = d.get('ll_window_train_Y', None)
        dd['ll_window_test_X'] = d.get('ll_window_test_X', None)
        dd['ll_window_test_Y'] = d.get('ll_window_test_Y', None)
        #-----------------------------------------------------------------------------------------

        ut.save_pickle(dd, inc_model_pkl)
        del ml, dd, d

    return True


def saveWindowFeatures(od, processed_data_path, pkl_prefix, win_size=5, WIN_renew=False):
    # sample x length x (dim x window)
    (successWinData, failureWinData)\
      = dm.getWindowData(od['successData'], od['failureData'], window=win_size )

    kFold_list = od['kFoldList']
    for idx, (normalTrainIdx, abnormalTrainIdx, normalTestIdx, abnormalTestIdx) in enumerate(kFold_list):

        model_pkl = os.path.join(processed_data_path, pkl_prefix+'_'+str(idx)+'.pkl')
        if os.path.isfile(model_pkl) and WIN_renew is False :
            print idx, " : no update the sliding window data"
            continue
        else:
            d = ut.load_pickle(model_pkl)

        # for window data,  sample x length x features
        n,m,k = np.shape(np.array(successWinData)[normalTrainIdx])
        d['ll_window_train_X'] = np.array(successWinData)[normalTrainIdx].reshape(n*m,k)
        d['ll_window_train_Y'] = -1*np.ones(n*m)
        
        n1,m1,k1 = np.shape(np.array(successWinData)[normalTestIdx])
        te_win_org_X1 = np.array(successWinData)[normalTestIdx]

        n2,m2,k2 = np.shape(np.array(failureWinData)[abnormalTestIdx])
        te_win_org_X2 = np.array(failureWinData)[abnormalTestIdx]

        d['ll_window_test_X'] = np.vstack([te_win_org_X1, te_win_org_X2])
        d['ll_window_test_Y'] = np.hstack([-1*np.ones(n1), np.ones(n2)])

        ut.save_pickle(d, model_pkl)

    return True


def saveWindowFeaturesForADP(td, processed_data_path, ADT_dict, pkl_prefix, win_size=5,
                             startIdx=4, tgt_hmm_idx=0):


    model_pkl = os.path.join(processed_data_path, pkl_prefix+'_'+str(tgt_hmm_idx)+'.pkl')
    d = ut.load_pickle(model_pkl)
    tr_win_org_X = d['ll_window_train_X']
    tr_win_org_Y = d['ll_window_train_Y']
    
    # Split test data to two groups
    n_AHMM_sample   = ADT_dict['n_pTrain']
    n_AHMM_test_idx = ADT_dict.get('test_idx',10)
    for idx in xrange(len(td['successDataList'])): # per person

        inc_model_pkl = os.path.join(processed_data_path, pkl_prefix+'_'+str(idx)+'.pkl')
        if os.path.isfile(inc_model_pkl) and ADT_dict['WIN_renew'] is False :
            print idx, " : no update the sliding window data"
            continue
        else:
            d = ut.load_pickle(inc_model_pkl)
        
        # sample x length x (dim x window)
        (successWinData, failureWinData)\
          = dm.getWindowData(td['successDataList'][idx], td['failureDataList'][idx], window=win_size )
        ## successWinData = np.array(td['successWinDataList'][idx])
        ## failureWinData = np.array(td['failureWinDataList'][idx])

        if n_AHMM_sample>0:
            tr_win_new_X = np.array(successWinData[:n_AHMM_sample])
            n,m,k = np.shape(tr_win_new_X)
            tr_win_new_X = tr_win_new_X.reshape(n*m,k)
            tr_win_new_Y = -1*np.ones(n*m)

        te_win_new_X = successWinData[n_AHMM_test_idx:]
        n1,m1,k1 = np.shape(te_win_new_X)
        n2,m2,k2 = np.shape(failureWinData)
        te_win_new_X = np.vstack([te_win_new_X, failureWinData])        
        te_win_new_Y = np.hstack([-1*np.ones(n1), np.ones(n2)])

        if ADT_dict['CLF'] == 'old':
            d['ll_window_train_X'] = tr_win_org_X
            d['ll_window_train_Y'] = tr_win_org_Y
        elif ADT_dict['CLF'] == 'adapt':
            d['ll_window_train_X'] = tr_win_org_X
            d['ll_window_train_Y'] = tr_win_org_Y
            d['ll_window_ptrain_X'] = tr_win_new_X
            d['ll_window_ptrain_Y'] = tr_win_new_Y
        elif ADT_dict['CLF'] == 'renew':
            d['ll_window_train_X'] = tr_win_new_X
            d['ll_window_train_Y'] = tr_win_new_Y
        else:
            print "Unknown classifier setting"
            sys.exit()

        # flattening the window data
        d['ll_window_test_X'] = te_win_new_X
        d['ll_window_test_Y'] = te_win_new_Y
        ut.save_pickle(d, inc_model_pkl)

    return True

