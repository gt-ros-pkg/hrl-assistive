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
                          step_mag, pkl_prefix=None,\
                          data_renew=False, save_pdf=False, verbose=False, debug=False,\
                          no_plot=False, delay_plot=False, find_param=False, all_plot=False):

    ## Parameters
    # data
    data_dict  = param_dict['data_param']
    data_renew = data_dict['renew']
    dim        = len(data_dict['handFeatures'])
    # HMM
    HMM_dict   = param_dict['HMM']
    nState     = HMM_dict['nState']
    cov        = HMM_dict['cov']
    # SVM
    SVM_dict   = param_dict['SVM']
    # ROC
    ROC_dict = param_dict['ROC']

    # reference data #TODO
    ref_data_path = os.path.join(processed_data_path, '../'+str(data_dict['downSampleSize'])+\
                                 '_'+str(dim))

    if pkl_prefix is None:
        pkl_prefix = 'step_'+"%0.4f" % step_mag
        step_mag   = step_mag*param_dict['HMM']['scale']

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
            ## dim_idx   = np.random.randint(0, len(abnormalTestData))

            #abnormalTestData[dim_idx,i,start_idx:] += step_mag
            abnormalTestData[:,i,start_idx:] += step_mag
            step_idx_l.append(start_idx)

        ml = hmm.learning_hmm(nState, nEmissionDim, verbose=False)
        ml.set_hmm_object(dd['A'], dd['B'], dd['pi'])            

        # Classifier test data
        ll_classifier_test_X, ll_classifier_test_Y, ll_classifier_test_idx =\
          hmm.getHMMinducedFeaturesFromRawFeatures(ml, normalTestData, abnormalTestData, startIdx)

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


    ## modeling_pkl = os.path.join(processed_data_path, 'hmm_'+pkl_prefix+'_'+str(0)+'.pkl')
    ## d = ut.load_pickle(modeling_pkl)
    ## ll_classifier_train_X = d['ll_classifier_train_X']
    ## ll_classifier_train_Y = d['ll_classifier_train_Y']
    ## ll_classifier_test_X = d['ll_classifier_test_X']
    ## ll_classifier_test_Y = d['ll_classifier_test_Y']

    ## print np.shape(ll_classifier_test_X)
    ## nNormal = len(kFold_list[0][2])
    ## nAbnormal = len(kFold_list[0][3])
    ## ll_logp_neg = np.array(ll_classifier_test_X)[:nNormal,:,0]
    ## ll_logp_pos = np.array(ll_classifier_test_X)[nNormal:,:,0]
    ## import hrl_anomaly_detection.data_viz as dv        
    ## dv.vizLikelihood(ll_logp_neg, ll_logp_pos)
    ## sys.exit()

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

        foldData = (kFold_list, successData, failureData)
        
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
                                                                 ROC_dict, \
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
    # HMM
    HMM_dict   = param_dict['HMM']
    nState     = HMM_dict['nState']
    cov        = HMM_dict['cov']
    # SVM
    SVM_dict   = param_dict['SVM']
    # ROC
    ROC_dict = param_dict['ROC']

    # reference data #TODO
    ref_data_path = os.path.join(processed_data_path, '../'+str(data_dict['downSampleSize'])+\
                                 '_'+str(dim))
    pkl_target_prefix = pkl_prefix.split('_')[0] 

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
    failureData = copy.copy(d['successData']) #d['failureData']
    param_dict2  = d['param_dict']
    if 'timeList' in param_dict2.keys():
        timeList = param_dict2['timeList'][startIdx:]
    else: timeList = None

    ## kFold_list = kFold_list[:1]
    
    score_list  = [ [[] for i in xrange(len(method_list))] for j in xrange(3) ]
    delay_list = [ [[] for i in xrange(len(method_list))] for j in xrange(3) ]
    det_rate_list = [ [[] for i in xrange(len(method_list))] for j in xrange(3) ]

    #-----------------------------------------------------------------------------------------
    # Training HMM, and getting classifier training and testing data
    for idx, (normalTrainIdx, abnormalTrainIdx, normalTestIdx, abnormalTestIdx) \
      in enumerate(kFold_list):

        if verbose: print idx, " : training hmm and getting classifier training and testing data"            
        
        # dim x sample x length
        normalTrainData   = successData[:, normalTrainIdx, :] * HMM_dict['scale']

        # training hmm
        if verbose: print "start to fit hmm"
        nEmissionDim = len(normalTrainData)
        cov_mult     = [cov]*(nEmissionDim**2)
        nLength      = len(normalTrainData[0][0]) - startIdx

        from sklearn import cross_validation
        normal_folds = cross_validation.KFold(len(normalTrainData[0]), n_folds=5, shuffle=True)

        for iidx, (train_fold, test_fold) in enumerate(normal_folds):

            modeling_pkl = os.path.join(processed_data_path, \
                                        'hmm_'+pkl_target_prefix+'_'+str(idx)+'_'+str(iidx)+'.pkl')
            if not (os.path.isfile(modeling_pkl) is False or HMM_dict['renew'] or data_renew): continue
            
            t_normalTrainData   = copy.copy(normalTrainData[:,train_fold])
            t_normalTestData    = copy.copy(normalTrainData[:,test_fold])
            t_abnormalTestData  = copy.copy(normalTrainData[:,test_fold])
            
            t_step_idx_l = []
            for i in xrange(len(t_normalTestData[0])):
                t_step_idx_l.append(None)
            for i in xrange(len(t_abnormalTestData[0])):
                start_idx = np.random.randint(startIdx, nLength/2, 1)[0]
                t_abnormalTestData[:,i,start_idx:] += step_mag
                t_step_idx_l.append(start_idx)

            #-----------------------------------------------------------------------------------------
            # Full co-variance
            #-----------------------------------------------------------------------------------------
            ml  = hmm.learning_hmm(nState, nEmissionDim, verbose=verbose) 
            ret = ml.fit(t_normalTrainData+\
                         np.random.normal(0.0, 0.03, np.shape(t_normalTrainData) )*HMM_dict['scale'], \
                         cov_mult=cov_mult, use_pkl=False)
            if ret == 'Failure': sys.exit()

            # Classifier training data
            ll_classifier_train_X, ll_classifier_train_Y, ll_classifier_train_idx =\
              hmm.getHMMinducedFeaturesFromRawFeatures(ml, t_normalTrainData, startIdx=startIdx)

            # Classifier test data
            ll_classifier_test_X, ll_classifier_test_Y, ll_classifier_test_idx =\
              hmm.getHMMinducedFeaturesFromRawFeatures(ml, t_normalTestData, t_abnormalTestData, startIdx)
           
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
            d['step_idx_l']   = t_step_idx_l
            ut.save_pickle(d, modeling_pkl)


        ## modeling_pkl = os.path.join(processed_data_path, \
        ##                             'hmm_'+pkl_target_prefix+'_'+str(0)+'_'+str(0)+'.pkl')
        ## d = ut.load_pickle(modeling_pkl)
        ## ll_classifier_train_X = d['ll_classifier_train_X']
        ## ll_classifier_train_Y = d['ll_classifier_train_Y']
        ## ll_classifier_test_X = d['ll_classifier_test_X']
        ## ll_classifier_test_Y = d['ll_classifier_test_Y']

        ## for iidx, (train_fold, test_fold) in enumerate(normal_folds):
        ##     print np.shape(ll_classifier_test_X)
        ##     nNormal = len(train_fold)
        ##     ll_logp_neg = np.array(ll_classifier_test_X)[:nNormal,:,0]
        ##     ll_logp_pos = np.array(ll_classifier_test_X)[nNormal:,:,0]
        ##     import hrl_anomaly_detection.data_viz as dv        
        ##     dv.vizLikelihood(ll_logp_neg, ll_logp_pos)
        ##     sys.exit()


        #-----------------------------------------------------------------------------------------
        roc_pkl = os.path.join(processed_data_path, 'roc_'+pkl_target_prefix+'_'+str(idx)+'.pkl')

        if os.path.isfile(roc_pkl) is False or HMM_dict['renew'] or SVM_dict['renew']: ROC_data = {}
        else: ROC_data = ut.load_pickle(roc_pkl)
        ROC_data = util.reset_roc_data(ROC_data, method_list, ROC_dict['update_list'], nPoints)

        osvm_data = None ; bpsvm_data = None
        if 'osvm' in method_list  and ROC_data['osvm']['complete'] is False:

            fold_list = []
            for train_fold, test_fold in normal_folds:
                fold_list.append([train_fold, test_fold])

            normalFoldData = (fold_list, normalTrainData/HMM_dict['scale'], normalTrainData/HMM_dict['scale'])
            modeling_pkl_prefix = os.path.join(processed_data_path, 'hmm_'+pkl_target_prefix+'_'+str(idx) )
            
            osvm_data = dm.getPCAData(len(fold_list), normalFoldData=normalFoldData, \
                                      window=SVM_dict['raw_window_size'],
                                      use_test=True, use_pca=False, \
                                      step_anomaly_info=(modeling_pkl_prefix,step_mag/HMM_dict['scale']))
            
 
        # parallelization
        if debug: n_jobs=1
        else: n_jobs=-1
        r = Parallel(n_jobs=n_jobs, verbose=50)(delayed(cf.run_classifiers)( iidx, processed_data_path, task_name, \
                                                                             method, ROC_data, \
                                                                             ROC_dict, \
                                                                             SVM_dict, HMM_dict, \
                                                                             raw_data=(osvm_data,bpsvm_data),\
                                                                             startIdx=startIdx, nState=nState, \
                                                                             delay_estimation=True,\
                                                                             modeling_pkl_prefix=\
                                                                             'hmm_'+pkl_target_prefix+'_'+str(idx))\
                                                                             for iidx in xrange(len(normal_folds))
                                                                             for method in method_list )

        l_data = r
        print "finished to run run_classifiers"

        ROC_data = util.update_roc_data(ROC_data, l_data, nPoints, method_list)
        ut.save_pickle(ROC_data, roc_pkl)
        ## delay_info(method_list, ROC_data, nPoints, no_plot=no_plot, save_pdf=save_pdf, timeList=timeList)

        #-----------------------------------------------------------------------------------------
        # ---------------- ROC Visualization ----------------------
        best_param_idx = getBestParamIdx(method_list, ROC_data, nPoints, nLength=nLength)

        print method_list
        print best_param_idx

        roc_pkl = os.path.join(processed_data_path, 'roc_'+pkl_prefix+'.pkl')
        ROC_data = ut.load_pickle(roc_pkl)
        scores, delays, det_rate = cost_info(best_param_idx, method_list, ROC_data, nPoints, \
                                             timeList=timeList, verbose=False)
        ## scores, delays = cost_info_with_max_tpr(method_list, ROC_data, nPoints, \
        ##                                         timeList=timeList, verbose=False)
        
        for i in xrange(len(method_list)):
            for j in xrange(3):
                score_list[j][i].append( scores[j][i] )
                delay_list[j][i] += delays[j][i]
                det_rate_list[j][i].append( det_rate[j][i])
                ## print np.shape(score_list[j][i]), np.shape(delay_list[j][i])

    if no_plot is False:
        plotCostDelay(method_list, score_list, delay_list, save_pdf=save_pdf)
        ## for i in xrange(len(det_rate_list)):
        ##     print i, det_rate_list[i]


def evaluation_acc_param2(subject_names, task_name, raw_data_path, processed_data_path, param_dict,\
                          step_mag_list,\
                          data_renew=False, save_pdf=False, verbose=False, debug=False,\
                          no_plot=False, delay_plot=False, find_param=False, all_plot=False):

    ## Parameters
    # data
    data_dict  = param_dict['data_param']
    data_renew = data_dict['renew']
    dim        = len(data_dict['handFeatures'])
    # HMM
    HMM_dict   = param_dict['HMM']
    nState     = HMM_dict['nState']
    cov        = HMM_dict['cov']
    # SVM
    SVM_dict   = param_dict['SVM']
    # ROC
    ROC_dict = param_dict['ROC']

    # reference data #TODO
    ref_data_path = os.path.join(processed_data_path, '../'+str(data_dict['downSampleSize'])+\
                                 '_'+str(dim))
    crossVal_pkl = os.path.join(ref_data_path, 'cv_'+task_name+'.pkl')

    #-----------------------------------------------------------------------------------------
    # parameters
    startIdx    = 4
    method_list = ROC_dict['methods'] 
    nPoints     = ROC_dict['nPoints']

    d = ut.load_pickle(crossVal_pkl)
    param_dict2  = d['param_dict']
    if 'timeList' in param_dict2.keys():
        timeList = param_dict2['timeList'][startIdx:]
        time_step = (timeList[-1]-timeList[0])/float(len(timeList)-1)
    else:
        timeList = None
        time_step = 1.0

    nLength = np.shape(timeList)[-1]

    #-----------------------------------------------------------------------------------------
    method    = 'hmmgp'
    s_tpr_l   = []
    s_delay_mean_l = []
    s_delay_std_l = []

    for step_mag in step_mag_list:

        pkl_prefix = 'step_'+"%0.4f" % step_mag #str(step_mag)        
        step_mag   = step_mag*param_dict['HMM']['scale']
        
        roc_pkl  = os.path.join(processed_data_path, 'roc_'+pkl_prefix+'.pkl')
        ROC_data = ut.load_pickle(roc_pkl)
        print roc_pkl
        
        tp_ll = ROC_data[method]['tp_l']
        fp_ll = ROC_data[method]['fp_l']
        tn_ll = ROC_data[method]['tn_l']
        fn_ll = ROC_data[method]['fn_l']
        delay_ll = ROC_data[method]['delay_l']

        tp_l = []
        tn_l = []
        fn_l = []
        fp_l = []
        delay_mean_l = []
        delay_std_l  = []
        for i in xrange(nPoints):
            tp_l.append( float(np.sum(tp_ll[i])) )
            tn_l.append( float(np.sum(tn_ll[i])) )
            fn_l.append( float(np.sum(fn_ll[i])) )
            fp_l.append( float(np.sum(fp_ll[i])) )

            delay_list = [ delay_ll[i][ii]*time_step for ii in xrange(len(delay_ll[i])) ]
            ## delay_list = [ delay_ll[i][ii]*time_step for ii in xrange(len(delay_ll[i])) \
            ##                if delay_ll[i][ii]>=0 ]

            ## # to handle.....
            ## tot_pos = int(np.sum(tp_ll[i]) + np.sum(fn_ll[i]))
            ## n_true_detection = float(len(delay_list))/float(tot_pos)
            ## if len(delay_list) < tot_pos:
            ##     for k in xrange(tot_pos-len(delay_list)):
            ##         delay_list.append(nLength*time_step)

            delay_list = [ delay_list[ii] for ii in xrange(len(delay_list)) if delay_list[ii]>=0 ]

            delay_mean_l.append( np.mean(delay_list) )
            delay_std_l.append( np.std(delay_list) )

        tp_l = np.array(tp_l)
        fp_l = np.array(fp_l)
        tn_l = np.array(tn_l)
        fn_l = np.array(fn_l)

        acc_l = (tp_l+tn_l)/( tp_l+tn_l+fp_l+fn_l )
        fpr_l = fp_l/(fp_l+tn_l)
        tpr_l = tp_l/(tp_l+fn_l)

        best_idx = (np.abs(fpr_l-0.1)).argmin()
        print "acc: ", acc_l[best_idx], "tpr: ", tpr_l[best_idx], "fpr: ", fpr_l[best_idx]
        print "best idx: ", best_idx

        s_tpr_l.append( tpr_l[best_idx] )
        s_delay_mean_l.append( delay_mean_l[best_idx] )
        s_delay_std_l.append( delay_std_l[best_idx] )

    s_tpr_l = np.array(s_tpr_l)

    ## s_delay_mean_l = s_delay_mean_l[-1:]
    ## s_delay_std_l = s_delay_std_l[-1:]
    ## s_tpr_l       = s_tpr_l[-1:]
    ## step_mag_list = step_mag_list[-1:] 
    
    if no_plot is False:

        ## x_ticks = []
        ## for step_mag in step_mag_list:
        ##     x_ticks.append( str(round(step_mag,3)) )

        fig, ax1 = plt.subplots()
        plt.rc('text', usetex=True)

        ax1.errorbar(step_mag_list, s_delay_mean_l, yerr=s_delay_std_l, fmt='-o', ms=10, lw=2)
        #ax1.plot(step_mag_list, s_delay_mean_l, 'bo-', ms=10, lw=2)        
        ax1.set_ylim([0.0,9.0])
        ## ax1.set_xticks(x_ticks)
        ax1.set_xlim([step_mag_list[0]-0.01, step_mag_list[-1]+0.01])
        ax1.set_ylabel(r'Detection Delay [sec]', fontsize=22)
        ax1.set_xlabel(r'Amplitude of Step Signals [$\%$]', fontsize=22)
        ax1.yaxis.label.set_color('blue')
        for tl in ax1.get_xticklabels():
            tl.set_fontsize(18)

        for tl in ax1.get_yticklabels():
            tl.set_color('b')
            tl.set_fontsize(18)

        ax2 = ax1.twinx()
        ax2.plot(step_mag_list, s_tpr_l*100.0, 'r^--', ms=10, lw=2)
        ax2.set_ylabel(r'True Positive Rate [$\%$]', fontsize=22)
        ax2.set_ylim([0.0,100.0])
        ax2.yaxis.label.set_color('red')
        for tl in ax2.get_yticklabels():
            tl.set_color('r')
            tl.set_fontsize(18)

        plt.tight_layout()

        ## m1= ax1.plot([],[], 'bx-', markersize=15, label='HMM-D')
        ## m2= ax1.plot([],[], 'bo-', markersize=15, label='HMM-GP')        
        ## ax1.legend(loc=2, prop={'size':20}, ncol=2)

        ## m3= ax2.plot([],[], 'rx--', markersize=15, label='HMM-D')
        ## m4= ax2.plot([],[], 'ro--', markersize=15, label='HMM-GP')        
        ## ax2.legend(loc=4, prop={'size':20} )

        if save_pdf == True:
            fig.savefig('test_'+method+'.pdf')
            fig.savefig('test_'+method+'.png')
            os.system('mv test_'+method+'.p* ~/Dropbox/HRL/')
        else:
            plt.show()        
        del fig, ax1, ax2

        
        ## plotCostDelay(method_list, score_list, delay_list, save_pdf=save_pdf)
        ## for i in xrange(len(det_rate_list)):
        ##     print i, det_rate_list[i]
