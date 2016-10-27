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
from hrl_anomaly_detection.AURO2016_params import *
from hrl_anomaly_detection.optimizeParam import *
from hrl_anomaly_detection import util as util

# learning
## from hrl_anomaly_detection.hmm import learning_hmm_multi_n as hmm
from hrl_anomaly_detection.hmm import learning_hmm as hmm
from mvpa2.datasets.base import Dataset
## from sklearn import svm
from joblib import Parallel, delayed
from sklearn import metrics
from sklearn.grid_search import ParameterGrid

# private learner
import hrl_anomaly_detection.classifiers.classifier as cf
import hrl_anomaly_detection.data_viz as dv

import itertools
colors = itertools.cycle(['g', 'm', 'c', 'k', 'y','r', 'b', ])
shapes = itertools.cycle(['x','v', 'o', '+'])

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42 
random.seed(3334)
np.random.seed(3334)



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
    
    if os.path.isfile(crossVal_pkl) and data_renew is False and data_gen is False:
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
                           rawFeatures=AE_dict['rawFeatures'],\
                           data_renew=data_renew, max_time=data_dict['max_time'])

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
    dm.saveHMMinducedFeatures(kFold_list, successData, failureData,\
                              task_name, processed_data_path,\
                              HMM_dict, data_renew, startIdx, nState, cov, scale, \
                              add_logp_d=add_logp_d, verbose=verbose)

    #-----------------------------------------------------------------------------------------
    roc_pkl = os.path.join(processed_data_path, 'roc_'+task_name+'.pkl')

    if os.path.isfile(roc_pkl) is False or HMM_dict['renew'] or SVM_dict['renew']: ROC_data = {}
    else: ROC_data = ut.load_pickle(roc_pkl)
    ROC_data = util.reset_roc_data(ROC_data, method_list, ROC_dict['update_list'], nPoints)

    osvm_data = None ; bpsvm_data = None
    if 'osvm' in method_list  and ROC_data['osvm']['complete'] is False:
        normalTrainData   = successData[:, normalTrainIdx, :] 
        abnormalTrainData = failureData[:, abnormalTrainIdx, :]

        fold_list = []
        for train_fold, test_fold in normal_folds:
            fold_list.append([train_fold, test_fold])

        normalFoldData = (fold_list, normalTrainData, abnormalTrainData)

        osvm_data = dm.getPCAData(len(fold_list), normalFoldData=normalFoldData, \
                                  window=SVM_dict['raw_window_size'],
                                  use_test=True, use_pca=False )

    # parallelization
    if debug: n_jobs=1
    else: n_jobs=-1
    l_data = Parallel(n_jobs=n_jobs, verbose=10)(delayed(cf.run_classifiers)( idx, processed_data_path, \
                                                                         task_name, \
                                                                         method, ROC_data, \
                                                                         ROC_dict, AE_dict, \
                                                                         SVM_dict, HMM_dict, \
                                                                         raw_data=(osvm_data,bpsvm_data),\
                                                                         startIdx=startIdx, nState=nState) \
                                                                         for idx in xrange(len(kFold_list)) \
                                                                         for method in method_list )


    print "finished to run run_classifiers"
    ROC_data = util.update_roc_data(ROC_data, l_data, nPoints, method_list)
    ut.save_pickle(ROC_data, roc_pkl)

    #-----------------------------------------------------------------------------------------
    ## best_param_idx = getBestParamIdx(method_list, ROC_data, nPoints, verbose=False)
    ## scores, delays = cost_info(best_param_idx, method_list, ROC_data, nPoints, \
    ##                            timeList=timeList, verbose=False)

    
    # ---------------- ROC Visualization ----------------------
    roc_info(method_list, ROC_data, nPoints, no_plot=True)


def evaluation_unexp(subject_names, task_name, raw_data_path, processed_data_path, param_dict,\
                     data_renew=False, save_pdf=False, verbose=False, debug=False,\
                     no_plot=False, delay_plot=True, find_param=False, data_gen=False):
    """Get a list of failures that executin monitoring systems cannot detect."""

    ## Parameters
    # data
    data_dict  = param_dict['data_param']
    data_renew = data_dict['renew']
    # AE
    AE_dict     = param_dict['AE']
    # HMM
    HMM_dict   = param_dict['HMM']
    nState     = HMM_dict['nState']
    scale      = HMM_dict['scale']
    cov        = HMM_dict['cov']
    add_logp_d = HMM_dict.get('add_logp_d', False)
    # SVM
    SVM_dict   = param_dict['SVM']

    # ROC
    ROC_dict = param_dict['ROC']

    # parameters
    startIdx    = 4
    method_list = ROC_dict['methods'] 
    nPoints     = ROC_dict['nPoints']
    
    #------------------------------------------

   
    if os.path.isdir(processed_data_path) is False:
        os.system('mkdir -p '+processed_data_path)

    crossVal_pkl = os.path.join(processed_data_path, 'cv_'+task_name+'.pkl')
    
    if os.path.isfile(crossVal_pkl) and data_renew is False and data_gen is False:
        print "CV data exists and no renew"
        d = ut.load_pickle(crossVal_pkl)
        print d.keys()
        kFold_list  = d['kFoldList']
        successData = d['successData']
        failureData = d['failureData']        
        success_files = d['success_files']
        failure_files = d['failure_files']
    else:
        '''
        Use augmented data? if nAugment is 0, then aug_successData = successData
        '''
        # Get a data set with a leave-one-person-out
        print "Extract data using getDataLOPO"
        d = dm.getDataLOPO(subject_names, task_name, raw_data_path, \
                           processed_data_path, data_dict['rf_center'], data_dict['local_range'],\
                           downSampleSize=data_dict['downSampleSize'], scale=1.0,\
                           handFeatures=data_dict['handFeatures'], \
                           cut_data=data_dict['cut_data'], \
                           data_renew=data_renew, max_time=data_dict['max_time'])
        successData, failureData, success_files, failure_files, kFold_list \
          = dm.LOPO_data_index(d['successDataList'], d['failureDataList'],\
                               d['successFileList'], d['failureFileList'])

        d['successData']   = successData
        d['failureData']   = failureData
        d['success_files'] = success_files
        d['failure_files'] = failure_files
        d['kFoldList']     = kFold_list
        ut.save_pickle(d, crossVal_pkl)
        if data_gen: sys.exit()


    param_dict2  = d['param_dict']
    if 'timeList' in param_dict2.keys():
        timeList    = param_dict2['timeList'][startIdx:]
    else: timeList = None

    #-----------------------------------------------------------------------------------------
    # HMM-induced vector with LOPO
    dm.saveHMMinducedFeatures(kFold_list, successData, failureData,\
                              task_name, processed_data_path,\
                              HMM_dict, data_renew, startIdx, nState, cov, scale, \
                              success_files=success_files, failure_files=failure_files,\
                              add_logp_d=add_logp_d, verbose=verbose)
                              
    #-----------------------------------------------------------------------------------------
    roc_pkl = os.path.join(processed_data_path, 'roc_'+task_name+'.pkl')
    if os.path.isfile(roc_pkl) is False or HMM_dict['renew'] or SVM_dict['renew']: ROC_data = {}
    else: ROC_data = ut.load_pickle(roc_pkl)
    ROC_data = util.reset_roc_data(ROC_data, method_list, ROC_dict['update_list'], nPoints)


    # parallelization
    if debug: n_jobs=1
    else: n_jobs=-1
    l_data = Parallel(n_jobs=n_jobs, verbose=10)(delayed(cf.run_classifiers)( idx, processed_data_path, \
                                                                         task_name, \
                                                                         method, ROC_data, \
                                                                         ROC_dict, AE_dict, \
                                                                         SVM_dict, HMM_dict, \
                                                                         startIdx=startIdx, nState=nState) \
                                                                         for idx in xrange(len(kFold_list)) \
                                                                         for method in method_list )


    print "finished to run run_classifiers"
    ROC_data = util.update_roc_data(ROC_data, l_data, nPoints, method_list)
    ut.save_pickle(ROC_data, roc_pkl)


    #----------------- List up anomaly cases ------------------
    ## for method in method_list:
    ##     max_idx = np.argmax(acc_rates[method])

    ##     print "-----------------------------------"
    ##     print "Method: ", method
    ##     print acc_rates[method][max_idx]
    ## if nPoints > 1:
    ##     print "Wrong number of points"
    ##     sys.exit()
    
    for method in method_list:
        print "---------- ", method, " -----------"

        tp_l = []
        fn_l = []
        fp_l = []
        for i in xrange(nPoints):
            tp_l.append( float(np.sum(ROC_data[method]['tp_l'][i])) )
            fn_l.append( float(np.sum(ROC_data[method]['fn_l'][i])) )
            fp_l.append( float(np.sum(ROC_data[method]['fp_l'][i])) )
        fscore_l = 2.0*np.array(tp_l)/(2.0*np.array(tp_l)+np.array(fp_l)+np.array(fn_l))

        ##################################3
        ## best_idx = np.argmin(fp_l)
        best_idx = np.argmax(fscore_l)
        ##################################3
        
        print 'fp_l:', fp_l
        print 'fscore: ', fscore_l
        print "F1-score: ", fscore_l[best_idx]
        print "best idx: ", best_idx

        # fscore
        ## tp = float(np.sum(ROC_data[method]['tp_l'][0]))
        ## fn = float(np.sum(ROC_data[method]['fn_l'][0]))
        ## fp = float(np.sum(ROC_data[method]['fp_l'][0]))
        ## fscore_1 = 2.0*tp/(2.0*tp+fn+fp)
        
        # false negatives
        ## n = len(ROC_data[method]['fn_labels'])
        labels = ROC_data[method]['fn_labels'][best_idx]            
        anomalies = [label.split('/')[-1].split('_')[0] for label in labels] # extract class
            
        d = {x: anomalies.count(x) for x in anomalies}
        l_idx = np.array(d.values()).argsort()[-10:]

        print "Max count is ", len(kFold_list)*2
        for idx in l_idx:
            print "Class: ", np.array(d.keys())[idx], "Count: ", np.array(d.values())[idx]




if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    util.initialiseOptParser(p)
    opt, args = p.parse_args()
    

    #---------------------------------------------------------------------------           
    # Run evaluation
    #---------------------------------------------------------------------------           
    rf_center     = 'kinEEPos'        
    scale         = 1.0
    local_range   = 10.0
    ## nPoints = 1 if opt.bEvaluationUnexpected else None
    nPoints = 40 #None

    raw_data_path, save_data_path, param_dict = getParams(opt.task, opt.bDataRenew, \
                                                          opt.bHMMRenew, opt.bClassifierRenew, opt.dim,\
                                                          rf_center, local_range, nPoints=nPoints)
    if opt.bNoUpdate: param_dict['ROC']['update_list'] = []
    # Mikako - bad camera
    # s1 - kaci - before camera calibration
    subjects = ['s2', 's3','s4','s5', 's6','s7','s8', 's9']
    ## subjects = ['s1', 's5', 's6']


    save_data_path = os.path.expanduser('~')+\
      '/hrl_file_server/dpark_data/anomaly/AURO2016/'+opt.task+'_data_unexp/'+\
      str(param_dict['data_param']['downSampleSize'])+'_'+str(opt.dim)
    param_dict['ROC']['methods'] = ['fixed']
    param_dict['HMM']['nState'] = 20
    param_dict['HMM']['scale']  = 15.55
    param_dict['HMM']['cov']    = 3.75


    
    #---------------------------------------------------------------------------           
    if opt.bRawDataPlot or opt.bInterpDataPlot:
        '''
        Before localization: Raw data plot
        After localization: Raw or interpolated data plot
        '''
        successData = True
        failureData = False
        modality_list   = ['kinematics', 'kinematics_des', 'audioWrist', 'ft', 'vision_landmark'] # raw plot

        dv.data_plot(subjects, opt.task, raw_data_path, save_data_path,\
                  downSampleSize=param_dict['data_param']['downSampleSize'], \
                  local_range=local_range, rf_center=rf_center, global_data=True, \
                  raw_viz=opt.bRawDataPlot, interp_viz=opt.bInterpDataPlot, save_pdf=opt.bSavePdf,\
                  successData=successData, failureData=failureData,\
                  continuousPlot=True, \
                  modality_list=modality_list, data_renew=opt.bDataRenew, verbose=opt.bVerbose)

    elif opt.bFeaturePlot:
        success_viz = True
        failure_viz = False
        ## param_dict['data_param']['handFeatures'] = ['crossmodal_targetEEDist', \
        ##                                             'crossmodal_targetEEAng']
        
        dm.getDataLOPO(subjects, opt.task, raw_data_path, save_data_path,
                       param_dict['data_param']['rf_center'], param_dict['data_param']['local_range'],\
                       downSampleSize=param_dict['data_param']['downSampleSize'], scale=scale, \
                       success_viz=success_viz, failure_viz=failure_viz,\
                       ae_data=False,\
                       cut_data=param_dict['data_param']['cut_data'],\
                       save_pdf=opt.bSavePdf, solid_color=True,\
                       handFeatures=param_dict['data_param']['handFeatures'], data_renew=opt.bDataRenew, \
                       max_time=param_dict['data_param']['max_time'])

    elif opt.bLikelihoodPlot:
        import hrl_anomaly_detection.data_viz as dv        
        dv.vizLikelihoods(subjects, opt.task, raw_data_path, save_data_path, param_dict,\
                          decision_boundary_viz=False, \
                          useTrain=True, useNormalTest=True, useAbnormalTest=True,\
                          useTrain_color=False, useNormalTest_color=False, useAbnormalTest_color=False,\
                          hmm_renew=opt.bHMMRenew, data_renew=opt.bDataRenew, save_pdf=opt.bSavePdf,\
                          verbose=opt.bVerbose)

    elif opt.HMM_param_search:
        from hrl_anomaly_detection.hmm import run_hmm_cpu as hmm_opt
        parameters = {'nState': [15, 20, 25], 'scale': np.linspace(10.0,20.0,10), \
                      'cov': np.linspace(1.5,6.0,5) }
        max_check_fold = len(subjects) #5 #None
        no_cov = False
        method = 'hmmgp'
        
        hmm_opt.tune_hmm(parameters, opt.task, param_dict, save_data_path, verbose=False, n_jobs=opt.n_jobs, \
                         bSave=opt.bSave, method=method, max_check_fold=max_check_fold, no_cov=no_cov)

    elif opt.CLF_param_search:
        from hrl_anomaly_detection.classifiers import opt_classifier as clf_opt
        method = 'progress'
        clf_opt.tune_classifier(save_data_path, opt.task, method, param_dict, file_idx=2,\
                                n_jobs=opt.n_jobs, n_iter_search=1000, save=opt.bSave)

    elif opt.bEvaluationAll or opt.bDataGen:
        if opt.bHMMRenew: param_dict['ROC']['methods'] = ['fixed'] 
        ## param_dict['ROC']['update_list'] = ['svm']
                    
        evaluation_all(subjects, opt.task, raw_data_path, save_data_path, param_dict, save_pdf=opt.bSavePdf, \
                       verbose=opt.bVerbose, debug=opt.bDebug, no_plot=opt.bNoPlot, \
                       find_param=False, data_gen=opt.bDataGen)

    elif opt.bEvaluationUnexpected:
        param_dict['ROC']['methods'] = ['progress', 'hmmgp']
        param_dict['ROC']['update_list'] = []

        evaluation_unexp(subjects, opt.task, raw_data_path, save_data_path, \
                         param_dict, save_pdf=opt.bSavePdf, \
                         verbose=opt.bVerbose, debug=opt.bDebug, no_plot=opt.bNoPlot, \
                         find_param=False, data_gen=opt.bDataGen)

    elif opt.bEvaluationAccParam or opt.bEvaluationWithNoise:
        param_dict['ROC']['methods'] = ['osvm', 'fixed', 'change', 'hmmosvm', 'progress', 'hmmgp']
        if opt.bNoUpdate: param_dict['ROC']['update_list'] = []        
        nPoints = param_dict['ROC']['nPoints']

        save_data_path = os.path.expanduser('~')+\
          '/hrl_file_server/dpark_data/anomaly/AURO2016/'+opt.task+'_data_unexp/'+\
          str(param_dict['data_param']['downSampleSize'])+'_'+str(opt.dim)+'_acc_param'

        if opt.task == 'feeding':
            nPoints = 50
            ## param_dict['SVM']['hmmosvm_nu'] = 0.1
            param_dict['ROC']['hmmgp_param_range']  = -np.logspace(0.0, 3.0, nPoints)+2.0
            param_dict['ROC']['kmean_param_range']  = np.logspace(0.16, 0.8, nPoints)*-1.0
            param_dict['ROC']['progress_param_range'] = -np.logspace(0.0, 2.5, nPoints)+2.0            
            param_dict['ROC']['osvm_param_range']     = np.logspace(-5,1,nPoints)
            param_dict['ROC']['hmmosvm_param_range']  = np.logspace(-4,1,nPoints)
            param_dict['ROC']['fixed_param_range']  = np.linspace(2.0, -2.5, nPoints)
            param_dict['ROC']['change_param_range'] = np.linspace(5.0, -55.0, nPoints)
        else:
            sys.exit()

        if False:
            step_mag =0.01*param_dict['HMM']['scale'] # need to varying it
            pkl_prefix = 'step_0.01'
        elif 0:
            step_mag =0.05*param_dict['HMM']['scale'] # need to varying it
            pkl_prefix = 'step_0.05'
        elif 1:
            step_mag = 0.1*param_dict['HMM']['scale'] # need to varying it
            pkl_prefix = 'step_0.1'
        ## elif 0:
        ##     step_mag = 0.15*param_dict['HMM']['scale'] # need to varying it
        ##     pkl_prefix = 'step_0.15'
        elif 1:
            step_mag = 0.2*param_dict['HMM']['scale'] # need to varying it
            pkl_prefix = 'step_0.2'
        elif 0:
            step_mag = 0.25*param_dict['HMM']['scale'] # need to varying it
            pkl_prefix = 'step_0.25'
        elif True:
            step_mag = 0.5*param_dict['HMM']['scale'] # need to varying it
            pkl_prefix = 'step_0.5'
        elif True:
            step_mag =1.0*param_dict['HMM']['scale'] # need to varying it
            pkl_prefix = 'step_1.0'
        else:
            step_mag = 10000000*param_dict['HMM']['scale'] # need to varying it
            pkl_prefix = 'step_10000000'

        import hrl_anomaly_detection.evaluation as ev 
        if opt.bEvaluationAccParam:
            ev.evaluation_acc_param(subjects, opt.task, raw_data_path, save_data_path, param_dict,\
                                    step_mag, pkl_prefix,\
                                    save_pdf=opt.bSavePdf, verbose=opt.bVerbose, debug=opt.bDebug, \
                                    no_plot=opt.bNoPlot, delay_plot=True)
        else:        
            ev.evaluation_step_noise(subjects, opt.task, raw_data_path, save_data_path, param_dict,\
                                     step_mag, pkl_prefix,\
                                     save_pdf=opt.bSavePdf, verbose=opt.bVerbose, debug=opt.bDebug, \
                                     no_plot=opt.bNoPlot, delay_plot=True)
