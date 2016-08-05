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

    hmm.computeHMMfeatures(task_name, processed_data_path, param_dict, data_renew=data_renew)


def evaluation_unexp(subject_names, unexpected_subjects, task_name, raw_data_path, processed_data_path, \
                     param_dict,\
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
