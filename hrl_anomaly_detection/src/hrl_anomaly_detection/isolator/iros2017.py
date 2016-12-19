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
from joblib import Parallel, delayed
import hrl_lib.util as ut

# Private utils
from hrl_anomaly_detection.util import *
from hrl_anomaly_detection.util_viz import *
from hrl_anomaly_detection import data_manager as dm
from hrl_anomaly_detection import util as util

# Private learners
from hrl_anomaly_detection.hmm import learning_hmm as hmm
import hrl_anomaly_detection.classifiers.classifier as cf
import hrl_anomaly_detection.data_viz as dv

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
random.seed(3334)
np.random.seed(3334)


def evaluation_all(subject_names, task_name, raw_data_path, processed_data_path, param_dict,\
                   data_renew=False, save_pdf=False, verbose=False, debug=False,\
                   no_plot=False, delay_plot=True, find_param=False, data_gen=False, target_class=None):
    # data
    data_dict  = param_dict['data_param']
    data_renew = data_dict['renew']
    # HMM
    HMM_dict   = param_dict['HMM']
    nState     = HMM_dict['nState']
    cov        = HMM_dict['cov']
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
        kFold_list = d['kFoldList'] 
        success_data = d['successData']
        failure_data = d['failureData']        
        success_files = d['success_files']
        failure_files = d['failure_files']
    else:
        '''
        Use augmented data? if nAugment is 0, then aug_successData = successData
        '''        
        d = dm.getDataLOPO(subject_names, task_name, raw_data_path, \
                           processed_data_path, data_dict['rf_center'], data_dict['local_range'],\
                           downSampleSize=data_dict['downSampleSize'],\
                           handFeatures=param_dict['data_param']['isolationFeatures'], \
                           cut_data=data_dict['cut_data'], \
                           data_renew=data_renew, max_time=data_dict['max_time'])
        success_data, failure_data, success_files, failure_files, kFold_list \
          = dm.LOPO_data_index(d['successDataList'], d['failureDataList'],\
                               d['successFileList'], d['failureFileList'])

        d['successData'] = success_data
        d['failureData'] = failure_data
        d['success_files']   = success_files
        d['failure_files']   = failure_files
        d['kFoldList']       = kFold_list
        ut.save_pickle(d, crossVal_pkl)
        if data_gen: sys.exit()

    #-----------------------------------------------------------------------------------------

    print processed_data_path
    print d.keys()
    print d['param_dict']['feature_names']
    print len(d['param_dict']['feature_names'])
    print d['param_dict']['feature_names'][10]
    ## sys.exit()

    #temp
    kFold_list = kFold_list[:8]

    ## x_classes = ['Object collision', 'Noisy environment', 'Spoon miss by a user', 'Spoon collision by a user', 'Robot-body collision by a user', 'Aggressive eating', 'Anomalous sound from a user', 'Unreachable mouth pose', 'Face occlusion by a user', 'Spoon miss by system fault', 'Spoon collision by system fault', 'Freeze by system fault']

    org_processed_data_path = copy.copy(processed_data_path)
    for i in xrange(len(success_data)):

        successData = copy.deepcopy(d['successData'][[10,i]])
        failureData = copy.deepcopy(d['failureData'][[10,i]])
        ## successData = copy.copy(d['successIsolData'][i:i+1])
        ## failureData = copy.copy(d['failureIsolData'][i:i+1])

        success_files = d['success_files']
        failure_files = d['failure_files']

        processed_data_path = os.path.join(org_processed_data_path, str(i))
        if os.path.isdir(processed_data_path) is False:
            os.system('mkdir -p '+processed_data_path)

        roc_pkl = os.path.join(processed_data_path, 'roc_'+task_name+'.pkl')
        if os.path.isfile(roc_pkl) and HMM_dict['renew'] is False and SVM_dict['renew'] is False and \
          data_renew is False :
            print "ppppppppppppppass"
            continue

        #-----------------------------------------------------------------------------------------    
        # Training HMM, and getting classifier training and testing data
        try:
            dm.saveHMMinducedFeatures(kFold_list, successData, failureData,\
                                      task_name, processed_data_path,\
                                      HMM_dict, data_renew, startIdx, nState, cov, \
                                      success_files=success_files, failure_files=failure_files,\
                                      noise_mag=0.03, verbose=verbose)
        except:
            ## raise ValueError("hmm induced feature error")
            print "Feature ", i, " does not work"
            continue
            ## sys.exit()

        #-----------------------------------------------------------------------------------------
        if os.path.isfile(roc_pkl) is False or HMM_dict['renew'] or SVM_dict['renew']: ROC_data = {}
        else: ROC_data = ut.load_pickle(roc_pkl)
        ROC_data = util.reset_roc_data(ROC_data, method_list, ROC_dict['update_list'], nPoints)

        osvm_data = None ; bpsvm_data = None
        if 'osvm' in method_list  and ROC_data['osvm']['complete'] is False:
            osvm_data = dm.getPCAData(len(kFold_list), crossVal_pkl, \
                                      window=SVM_dict['raw_window_size'],
                                      use_test=True, use_pca=False )

        # parallelization
        if debug: n_jobs=1
        else: n_jobs=-1
        l_data = Parallel(n_jobs=n_jobs, verbose=10)(delayed(cf.run_classifiers)( idx, processed_data_path, \
                                                                             task_name, \
                                                                             method, ROC_data, \
                                                                             ROC_dict, \
                                                                             SVM_dict, HMM_dict, \
                                                                             raw_data=(osvm_data,bpsvm_data),\
                                                                             startIdx=startIdx, nState=nState,\
                                                                             n_jobs=n_jobs) \
                                                                             for idx in xrange(len(kFold_list)) \
                                                                             for method in method_list )


        print "finished to run run_classifiers"
        ROC_data = util.update_roc_data(ROC_data, l_data, nPoints, method_list)
        ut.save_pickle(ROC_data, roc_pkl)

        ## if detection_rate: detection_info(method_list, ROC_data, nPoints, kFold_list,zero_fp_flag=True)
        
    # ---------------- ROC Visualization ----------------------
    ## if detection_rate: sys.exit()
    for idx in xrange(len(success_data)):
        processed_data_path = os.path.join(org_processed_data_path, str(idx))
        roc_pkl = os.path.join(processed_data_path, 'roc_'+task_name+'.pkl')
        if os.path.isfile(roc_pkl) is False: continue
        ROC_data = ut.load_pickle(roc_pkl)        
        ## auc = roc_info(method_list, ROC_data, nPoints, no_plot=True, verbose=False)


        modeling_pkl = os.path.join(processed_data_path, 'hmm_'+task_name+'_'+str(0)+'.pkl')
        d            = ut.load_pickle(modeling_pkl)
        ll_classifier_test_labels = d['ll_classifier_test_labels']
        method = method_list[0]
        
        if target_class is not None:
            tot_pos = 0
            for c in target_class:
                for l in ll_classifier_test_labels:
                    if c == int(l.split('/')[-1].split('_')[0]):
                        tot_pos += 1.0
            tot_pos *= float(len(kFold_list))


            ## print "Total failures:", tot_pos
            fn_ll = []
            tp_ll = []
            for i in xrange(len(ROC_data['hmmgp']['tp_l'])):
                fn = 0
                for l in ROC_data['hmmgp']['fn_labels'][i]:
                    for c in target_class:
                        if c == int(l.split('/')[-1].split('_')[0]):
                            fn += 1.0
                            break

                fn_ll.append(fn)
                tp_ll.append(tot_pos- fn)
        else:
            tp_ll = ROC_data[method]['tp_l']
            fn_ll = ROC_data[method]['fn_l']

        
        fp_ll = ROC_data[method]['fp_l']
        tn_ll = ROC_data[method]['tn_l']

        tpr_l = []
        fpr_l = []
        fnr_l = []
        for i in xrange(nPoints):
            tpr_l.append( float(np.sum(tp_ll[i]))/float(np.sum(tp_ll[i])+np.sum(fn_ll[i]))*100.0 )
            fnr_l.append( 100.0 - tpr_l[-1] )
            fpr_l.append( float(np.sum(fp_ll[i]))/float(np.sum(fp_ll[i])+np.sum(tn_ll[i]))*100.0 )

        from sklearn import metrics 
        auc = metrics.auc(fpr_l, tpr_l, True)
        print idx , auc


def evaluation_single_ad(subject_names, task_name, raw_data_path, processed_data_path, param_dict,\
                         data_renew=False, save_pdf=False, verbose=False, debug=False,\
                         no_plot=False, delay_plot=True, find_param=False, data_gen=False,\
                         target_class=None):

    ## Parameters
    # data
    data_dict  = param_dict['data_param']
    data_renew = data_dict['renew']
    # HMM
    HMM_dict   = param_dict['HMM']
    nState     = HMM_dict['nState']
    cov        = HMM_dict['cov']
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
        kFold_list = d['kFoldList'] 
        successData = d['successData']
        failureData = d['failureData']
        success_files = d['success_files']
        failure_files = d['failure_files']        
    else:
        '''
        Use augmented data? if nAugment is 0, then aug_successData = successData
        '''        
        d = dm.getDataLOPO(subject_names, task_name, raw_data_path, \
                           processed_data_path, data_dict['rf_center'], data_dict['local_range'],\
                           downSampleSize=data_dict['downSampleSize'],\
                           handFeatures=data_dict['handFeatures'], \
                           cut_data=data_dict['cut_data'], \
                           data_renew=data_renew, max_time=data_dict['max_time'])


        successData, failureData, success_files, failure_files, kFold_list \
          = dm.LOPO_data_index(d['successDataList'], d['failureDataList'],\
                               d['successFileList'], d['failureFileList'],\
                               target_class=target_class)

        d['successData']   = successData
        d['failureData']   = failureData
        d['success_files']   = success_files
        d['failure_files']   = failure_files        
        d['kFoldList']     = kFold_list
        ut.save_pickle(d, crossVal_pkl)
        if data_gen: sys.exit()

    # temp
    kFold_list = kFold_list[:8]

    #-----------------------------------------------------------------------------------------    
    # Training HMM, and getting classifier training and testing data
    dm.saveHMMinducedFeatures(kFold_list, successData, failureData,\
                              task_name, processed_data_path,\
                              HMM_dict, data_renew, startIdx, nState, cov, \
                              noise_mag=0.03, diag=False, \
                              verbose=verbose)

    #-----------------------------------------------------------------------------------------
    roc_pkl = os.path.join(processed_data_path, 'roc_'+task_name+'.pkl')

    if os.path.isfile(roc_pkl) is False or HMM_dict['renew'] or SVM_dict['renew']: ROC_data = {}
    else: ROC_data = ut.load_pickle(roc_pkl)
    ROC_data = util.reset_roc_data(ROC_data, method_list, ROC_dict['update_list'], nPoints)

    osvm_data = None ; bpsvm_data = None
    if 'osvm' in method_list  and ROC_data['osvm']['complete'] is False:
        osvm_data = dm.getPCAData(len(kFold_list), crossVal_pkl, \
                                  window=SVM_dict['raw_window_size'],
                                  use_test=True, use_pca=False )

    # parallelization
    if debug: n_jobs=1
    else: n_jobs=-1
    l_data = Parallel(n_jobs=n_jobs, verbose=10)(delayed(cf.run_classifiers)( idx, processed_data_path, \
                                                                         task_name, \
                                                                         method, ROC_data, \
                                                                         ROC_dict, \
                                                                         SVM_dict, HMM_dict, \
                                                                         raw_data=(osvm_data,bpsvm_data),\
                                                                         startIdx=startIdx, nState=nState) \
                                                                         for idx in xrange(len(kFold_list)) \
                                                                         for method in method_list )


    print "finished to run run_classifiers"
    ROC_data = util.update_roc_data(ROC_data, l_data, nPoints, method_list)
    ut.save_pickle(ROC_data, roc_pkl)

    # ---------------- ROC Visualization ----------------------
    roc_info(method_list, ROC_data, nPoints, no_plot=True)


def evaluation_double_ad(subject_names, task_name, raw_data_path, processed_data_path, param_dict,\
                         data_renew=False, save_pdf=False, verbose=False, debug=False,\
                         no_plot=False, delay_plot=True, find_param=False, data_gen=False):

    ## Parameters
    # data
    data_dict  = param_dict['data_param']
    data_renew = data_dict['renew']
    # HMM
    HMM_dict   = param_dict['HMM']
    nState     = HMM_dict['nState']
    cov        = HMM_dict['cov']
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
        kFold_list = d['kFoldList'] 
        success_isol_data = d['successIsolData']
        failure_isol_data = d['failureIsolData']        
        success_files = d['success_files']
        failure_files = d['failure_files']
    else:
        '''
        Use augmented data? if nAugment is 0, then aug_successData = successData
        '''        
        d = dm.getDataLOPO(subject_names, task_name, raw_data_path, \
                           processed_data_path, data_dict['rf_center'], data_dict['local_range'],\
                           downSampleSize=data_dict['downSampleSize'],\
                           handFeatures=data_dict['isolationFeatures'], \
                           cut_data=data_dict['cut_data'], \
                           data_renew=data_renew, max_time=data_dict['max_time'])
                           
        success_isol_data, failure_isol_data, success_files, failure_files, kFold_list \
          = dm.LOPO_data_index(d['successDataList'], d['failureDataList'],\
                               d['successFileList'], d['failureFileList'])

        d['successIsolData'] = success_isol_data
        d['failureIsolData'] = failure_isol_data
        d['success_files']   = success_files
        d['failure_files']   = failure_files
        d['kFoldList']       = kFold_list
        ut.save_pickle(d, crossVal_pkl)
        if data_gen: sys.exit()

    #-----------------------------------------------------------------------------------------
    # feature selection
    print d['param_dict']['feature_names']
    
    feature_idx_list = []
    for i in xrange(2):
        print param_dict['data_param']['handFeatures'][i]
        
        feature_idx_list.append([])
        for feature in param_dict['data_param']['handFeatures'][i]:
            feature_idx_list[i].append(d['param_dict']['feature_names'].index(feature))
        
        successData = copy.copy(success_isol_data[feature_idx_list[i]])
        failureData = copy.copy(failure_isol_data[feature_idx_list[i]])
        HMM_dict_local = copy.deepcopy(HMM_dict)
        HMM_dict_local['scale'] = param_dict['HMM']['scale'][i]

        # Training HMM, and getting classifier training and testing data
        dm.saveHMMinducedFeatures(kFold_list, successData, failureData,\
                                  task_name, processed_data_path,\
                                  HMM_dict_local, data_renew, startIdx, nState, cov, \
                                  noise_mag=0.03, diag=False, suffix=str(i),\
                                  verbose=verbose)

    #-----------------------------------------------------------------------------------------
    roc_pkl = os.path.join(processed_data_path, 'roc_'+task_name+'.pkl')

    if os.path.isfile(roc_pkl) is False or HMM_dict['renew'] or SVM_dict['renew']: ROC_data = {}
    else: ROC_data = ut.load_pickle(roc_pkl)
    ROC_data = util.reset_roc_data(ROC_data, [method_list[0]], ROC_dict['update_list'], nPoints)

    print "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    # temp
    kFold_list = kFold_list[:8]

    # parallelization
    if debug: n_jobs=1
    else: n_jobs=-1
    l_data = Parallel(n_jobs=n_jobs, verbose=10)\
      (delayed(cf.run_classifiers_boost)( idx, processed_data_path, \
                                          task_name, \
                                          method_list, ROC_data, \
                                          param_dict,\
                                          startIdx=startIdx, nState=nState) \
      for idx in xrange(len(kFold_list)) )

    print "finished to run run_classifiers"
    ROC_data = util.update_roc_data(ROC_data, l_data, nPoints, method_list)
    ut.save_pickle(ROC_data, roc_pkl)

    
    # ---------------- ROC Visualization ----------------------
    roc_info(method_list, ROC_data, nPoints, no_plot=True)


def evaluation_isolation(subject_names, task_name, raw_data_path, processed_data_path, param_dict,\
                         data_renew=False, save_pdf=False, verbose=False, debug=False,\
                         no_plot=False, delay_plot=True, find_param=False, data_gen=False, \
                         save_viz_data=False):

    ## Parameters
    # data
    data_dict  = param_dict['data_param']
    data_renew = data_dict['renew']
    # HMM
    HMM_dict   = param_dict['HMM']
    nState     = HMM_dict['nState']
    cov        = HMM_dict['cov']
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
        kFold_list = d['kFoldList'] 
        successData = d['successIsolData']
        failureData = d['failureIsolData']        
        success_files = d['success_files']
        failure_files = d['failure_files']
    else:
        '''
        Use augmented data? if nAugment is 0, then aug_successData = successData
        '''        
        d = dm.getDataLOPO(subject_names, task_name, raw_data_path, \
                           processed_data_path, data_dict['rf_center'], data_dict['local_range'],\
                           downSampleSize=data_dict['downSampleSize'],\
                           handFeatures=data_dict['isolationFeatures'], \
                           cut_data=data_dict['cut_data'], \
                           data_renew=data_renew, max_time=data_dict['max_time'])
                           
        successData, failureData, success_files, failure_files, kFold_list \
          = dm.LOPO_data_index(d['successDataList'], d['failureDataList'],\
                               d['successFileList'], d['failureFileList'])

        d['successIsolData'] = successData
        d['failureIsolData'] = failureData
        d['success_files']   = success_files
        d['failure_files']   = failure_files
        d['kFoldList']       = kFold_list
        ut.save_pickle(d, crossVal_pkl)
        if data_gen: sys.exit()

    failure_labels = []
    for f in failure_files:
        failure_labels.append( int( f.split('/')[-1].split('_')[0] ) )
    failure_labels = np.array( failure_labels )


    ## if save_viz_data or True:
    ##     LOG_DIR = os.path.join(processed_data_path, 'tensorflow' )
    ##     if os.path.isdir(LOG_DIR) is False:
    ##         os.system('mkdir -p '+LOG_DIR)
        
    ##     n_features = failureData.shape[0]
    ##     n_samples  = failureData.shape[1]
    ##     n_length  = failureData.shape[2]
    ##     training_data   = copy.copy(failureData)
    ##     training_data   = np.swapaxes(training_data, 0, 1).reshape((n_samples, n_features*n_length))        
    ##     training_labels = copy.copy(failure_labels)

    ##     import csv
    ##     tgt_csv = os.path.join(LOG_DIR, 'data.tsv')
    ##     with open(tgt_csv, 'w') as csvfile:
    ##         for row in training_data:
    ##             string = None
    ##             for col in row:
    ##                 if string is None:
    ##                     string = str(col)
    ##                 else:
    ##                     string += '\t'+str(col)
                        
    ##             csvfile.write(string+"\n")
            
    ##     tgt_csv = os.path.join(LOG_DIR, 'labels.tsv')
    ##     with open(tgt_csv, 'w') as csvfile:
    ##         for row in training_labels:
    ##             csvfile.write(str(row)+"\n")


    ## from sklearn.linear_model import OrthogonalMatchingPursuit
    from ksvd import KSVD

    # k-fold cross validation
    for idx, (normalTrainIdx, abnormalTrainIdx, normalTestIdx, abnormalTestIdx) \
      in enumerate(kFold_list):

        # dim x sample x length
        ## normalTrainData   = copy.copy(successData[:, normalTrainIdx, :]) 
        ## normalTestData    = copy.copy(successData[:, normalTestIdx, :])
        abnormalTrainData = copy.copy(failureData[:, abnormalTrainIdx, :])
        abnormalTestData  = copy.copy(failureData[:, abnormalTestIdx, :])

        abnormalTrainLabel = copy.copy(failure_labels[abnormalTrainIdx])
        abnormalTestLabel  = copy.copy(failure_labels[abnormalTestIdx])

        # ---------------------------------------------------------------------------
        ## # random sampling?
        ## ## nSample     = 30
        ## window_size = 30
        ## window_step = 5
        ## X_train = []
        ## Y_train = []
        ## ml_dict = {}
        ## for i in xrange(len(abnormalTrainData[0])): # per sample

        ##     s_l = np.arange(startIdx, len(abnormalTrainData[0][i])-int(window_size*1.5), window_step)            
        ##     ## s_l = np.random.randint(startIdx, len(abnormalTrainData[0][i])-window_size*2, nSample)

        ##     for j in s_l:
        ##         block = abnormalTrainData[:,i,j:j+window_size]

        ##         # zero mean to resolve signal displacements
        ##         block -= np.mean(block, axis=1)[:,np.newaxis]

        ##         X_train.append( block )
        ##         Y_train.append( abnormalTrainLabel[i] )

        # ---------------------------------------------------------------------------
        X_train = []
        for i in xrange(len(abnormalTrainData[0])): # per sample
            X_train.append(abnormalTrainData[:,i,:]) #-np.mean(abnormalTrainData[:,i,:], axis=1)[:, np.newaxis])
        Y_train = copy.copy(abnormalTrainLabel)

        del abnormalTrainData
        del abnormalTrainLabel

        # train feature-wise omp
        dimension = len(X_train[0][0]) #window_size
        dict_size = int(dimension*10) ##1.5)
        n_examples = len(X_train)
        target_sparsity = int(0.1*dimension)

        Gammas = None
        ml_dict = {}
        X_train = np.array(X_train)
        for i in xrange(len(X_train[0])): # per feature
            print i, ' / ', len(X_train[0])
            # X \simeq Gamma * D
            # D is the dictionary with `dict_size` by `dimension`
            # Gamma is the code book with `n_examples` by `dict_size`
            D, Gamma = KSVD(X_train[:,i,:], dict_size, target_sparsity, 1000,
                            print_interval = 25,
                            enable_printing = True, enable_threading = True)
            ## print np.shape(D), np.shape(Gamma), dict_size, dimension, n_examples

            ml_dict[i] = (D, Gamma)

            if Gammas is None:
                Gammas = Gamma
            else:
                Gammas = np.hstack([Gammas, Gamma])

            del D
            del Gamma
        # ---------------------------------------------------------------------------
        X_train = []
        Y_train = []
        for i in xrange(len(abnormalTrainData[0])): # per sample
            for j in xrange(len(abnormalTrainData[0][i])): # per time
                X_train.append(abnormalTrainData[:,i,j]) 
                Y_train.append(abnormalTrainLabel[i])

        del abnormalTrainData
        del abnormalTrainLabel

        # train time-wise omp
        dimension = len(X_train[0]) 
        dict_size = int(dimension*10) ##1.5)
        n_examples = len(X_train)
        target_sparsity = int(0.1*dimension)
        
        window_size = 30 # for max pooling?
        window_step = 5  # for max pooling?
        Gammas = None
        ml_dict = {}
        ## for i in xrange(window_size, len(X_train[0][0])-1, window_step): # per length
            
        # X \simeq Gamma * D
        # D is the dictionary with `dict_size` by `dimension`
        # Gamma is the code book with `n_examples` by `dict_size`
        D, Gamma = KSVD(X_train, dict_size, target_sparsity, 1000,
                        print_interval = 25,
                        enable_printing = True, enable_threading = True)
        ## ml_dict[i] = (D, Gamma)

        ## if Gammas is None:  Gammas = Gamma
        ## else:               Gammas = np.hstack([Gammas, Gamma])
        ## del D, Gamma
        Gammas = gamma

        save_data_labels(Gammas, Y_train, processed_data_path)

        sys.exit()

        X_test = []
        Y_test = []
        for i in xrange(len(abnormalTestData[0])): # per sample
            s_l = np.arange(startIdx, len(abnormalTestData[0][i])-int(window_size*1.5), window_step)            
            ## s_l = np.random.randint(startIdx, len(abnormalTestData[0][i])-window_size*2, nSample)

            for j in s_l:
                block = abnormalTestData[:,i,j:j+window_size]

                # zero mean and unit scale(?) to resolve signal displacements
                block -= np.mean(block, axis=1)[:,np.newaxis]

                X_test.append( block )
                Y_test.append( abnormalTestLabel[i] )

        X_test = np.array(X_test)
        for i in xrange(len(X_test[0])):
            ml = ml_dict[i]
            ml.fit(X_train[:,i,:], Y_train)



        scores.append( omp.score(X_test, Y_test) )
        ## Y_pred = omp.predict(X_test)
        sys.exit()
            
    print scores





        # vectorize
      

        # omp feature extraction?



        # svm classification


def         

def save_data_labels(data, labels, processed_data_path='./'):
    LOG_DIR = os.path.join(processed_data_path, 'tensorflow' )
    if os.path.isdir(LOG_DIR) is False:
        os.system('mkdir -p '+LOG_DIR)

    
    if len(np.shape(data)) > 2:
        n_features = np.shape(data)[0]
        n_samples  = np.shape(data)[1]
        n_length  = np.shape(data)[2]
        training_data   = copy.copy(data)
        training_data   = np.swapaxes(training_data, 0, 1).reshape((n_samples, n_features*n_length))
    else:
        training_data   = copy.copy(data)
    training_labels = copy.copy(labels)

    import csv
    tgt_csv = os.path.join(LOG_DIR, 'data.tsv')
    with open(tgt_csv, 'w') as csvfile:
        for row in training_data:
            string = None
            for col in row:
                if string is None:
                    string = str(col)
                else:
                    string += '\t'+str(col)

            csvfile.write(string+"\n")

    tgt_csv = os.path.join(LOG_DIR, 'labels.tsv')
    with open(tgt_csv, 'w') as csvfile:
        for row in training_labels:
            csvfile.write(str(row)+"\n")
    
    






if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    util.initialiseOptParser(p)

    p.add_option('--eval_single', '--es', action='store_true', dest='evaluation_single',
                 default=False, help='Evaluate with single detector.')
    p.add_option('--eval_double', '--ed', action='store_true', dest='evaluation_double',
                 default=False, help='Evaluate with double detectors.')
    p.add_option('--eval_isol', '--ei', action='store_true', dest='evaluation_isolation',
                 default=False, help='Evaluate anomaly isolation with double detectors.')
    
    opt, args = p.parse_args()

    #---------------------------------------------------------------------------           
    # Run evaluation
    #---------------------------------------------------------------------------           
    rf_center     = 'kinEEPos'        
    scale         = 1.0
    local_range   = 10.0
    nPoints = 40 #None

    from hrl_anomaly_detection.isolator.IROS2017_params import *
    raw_data_path, save_data_path, param_dict = getParams(opt.task, opt.bDataRenew, \
                                                          opt.bHMMRenew, opt.bClassifierRenew, opt.dim,\
                                                          rf_center, local_range, nPoints=nPoints)
    if opt.bNoUpdate: param_dict['ROC']['update_list'] = []
    # Mikako - bad camera
    # s1 - kaci - before camera calibration
    subjects = ['s2', 's3','s4','s5', 's6','s7','s8', 's9']

    save_data_path = os.path.expanduser('~')+\
      '/hrl_file_server/dpark_data/anomaly/AURO2016/'+opt.task+'_data_isolation3/'+\
      str(param_dict['data_param']['downSampleSize'])+'_'+str(opt.dim)

    target_class = [11,4,13,10]


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
        failure_viz = True
        save_data_path = os.path.expanduser('~')+\
          '/hrl_file_server/dpark_data/anomaly/AURO2016/'+opt.task+'_data_isolation4/'+\
          str(param_dict['data_param']['downSampleSize'])+'_'+str(opt.dim)
        param_dict['data_param']['handFeatures'] = ['unimodal_kinDesEEChange',\
                                                    ## 'unimodal_kinEEChange',\
                                                    ## 'unimodal_audioWristRMS', \
                                                    ## 'unimodal_audioWristFrontRMS', \
                                                    ## 'unimodal_audioWristAzimuth',\
                                                    'unimodal_kinJntEff', \
                                                    'unimodal_ftForce_integ', \
                                                    'unimodal_ftForce_delta', \
                                                    'unimodal_ftForce_zero', \
                                                    ## 'unimodal_ftForce', \
                                                    ## 'unimodal_ftForceX', \
                                                    ## 'unimodal_ftForceY', \
                                                    ## 'unimodal_ftForceZ', \
                                                    'crossmodal_landmarkEEDist', \
                                                    'crossmodal_landmarkEEAng',\
                                                    'unimodal_fabricForce',\
                                                    'unimodal_landmarkDist']
                                                    
        dm.getDataLOPO(subjects, opt.task, raw_data_path, save_data_path,
                       param_dict['data_param']['rf_center'], param_dict['data_param']['local_range'],\
                       downSampleSize=param_dict['data_param']['downSampleSize'], \
                       success_viz=success_viz, failure_viz=failure_viz,\
                       cut_data=param_dict['data_param']['cut_data'],\
                       save_pdf=opt.bSavePdf, solid_color=True,\
                       handFeatures=param_dict['data_param']['handFeatures'], data_renew=opt.bDataRenew, \
                       max_time=param_dict['data_param']['max_time'])
                       ## max_time=param_dict['data_param']['max_time'], target_class=target_class)

    elif opt.bLikelihoodPlot:
        save_data_path = os.path.expanduser('~')+\
          '/hrl_file_server/dpark_data/anomaly/AURO2016/'+opt.task+'_data_isolation4/'+\
          str(param_dict['data_param']['downSampleSize'])+'_'+str(opt.dim)
        param_dict['HMM']['scale'] = 4.0
        param_dict['data_param']['handFeatures'] = ['unimodal_ftForce_zero', 'crossmodal_landmarkEEDist']
        #'crossmodal_landmarkEEDist'        
        param_dict['ROC']['hmmgp_param_range'] = np.logspace(-0.6, 2.3, nPoints)*-1.0

        import hrl_anomaly_detection.data_viz as dv        
        dv.vizLikelihoods(subjects, opt.task, raw_data_path, save_data_path, param_dict,\
                          decision_boundary_viz=False, method='progress', \
                          useTrain=True, useNormalTest=False, useAbnormalTest=True,\
                          useTrain_color=False, useNormalTest_color=False, useAbnormalTest_color=False,\
                          hmm_renew=opt.bHMMRenew, data_renew=opt.bDataRenew, save_pdf=opt.bSavePdf,\
                          verbose=opt.bVerbose, lopo=True, plot_feature=False)

    elif opt.bEvaluationAll:
        '''
        feature-wise evaluation
        '''        
        save_data_path = os.path.expanduser('~')+\
          '/hrl_file_server/dpark_data/anomaly/AURO2016/'+opt.task+'_data_isolation3/'+\
          str(param_dict['data_param']['downSampleSize'])+'_'+str(opt.dim)

        param_dict['ROC']['methods'] = ['hmmgp']
        param_dict['HMM']['scale'] = 6.11
        if opt.bNoUpdate: param_dict['ROC']['update_list'] = []        
        evaluation_all(subjects, opt.task, raw_data_path, save_data_path, param_dict, save_pdf=opt.bSavePdf, \
                       verbose=opt.bVerbose, debug=opt.bDebug, no_plot=opt.bNoPlot, \
                       find_param=False, data_gen=opt.bDataGen, target_class=target_class)

    elif opt.evaluation_single:
        '''
        evaluation with selected feature set
        '''
        # 74%
        save_data_path = os.path.expanduser('~')+\
          '/hrl_file_server/dpark_data/anomaly/AURO2016/'+opt.task+'_data_isolation5/'+\
          str(param_dict['data_param']['downSampleSize'])+'_'+str(opt.dim)
        param_dict['data_param']['handFeatures'] = ['unimodal_ftForce_integ', 'crossmodal_landmarkEEDist']
        # 68
        ## param_dict['data_param']['handFeatures'] = ['unimodal_audioWristRMS', 'unimodal_ftForce_integ', \
        ##                                             'crossmodal_landmarkEEDist', 'unimodal_kinJntEff_1']


        # 84.5% scale1, 83.81% scale?
        ## save_data_path = os.path.expanduser('~')+\
        ##   '/hrl_file_server/dpark_data/anomaly/AURO2016/'+opt.task+'_data_isolation6/'+\
        ##   str(param_dict['data_param']['downSampleSize'])+'_'+str(opt.dim)
        ## param_dict['data_param']['handFeatures'] = ['unimodal_audioWristRMS', 'unimodal_ftForce_integ', \
        ##                                             'crossmodal_landmarkEEDist', 'unimodal_kinJntEff_1']

        # 78% scale?,  82% scale 1
        ## save_data_path = os.path.expanduser('~')+\
        ##   '/hrl_file_server/dpark_data/anomaly/AURO2016/'+opt.task+'_data_isolation7/'+\
        ##   str(param_dict['data_param']['downSampleSize'])+'_'+str(opt.dim)
        ## param_dict['data_param']['handFeatures'] = ['unimodal_audioWristRMS', 'unimodal_ftForceZ', \
        ##                                             'crossmodal_landmarkEEDist', 'unimodal_kinJntEff_1']

        param_dict['ROC']['methods'] = ['hmmgp']
        nPoints = param_dict['ROC']['nPoints']
        param_dict['ROC']['hmmgp_param_range'] = np.logspace(-0.6, 2.3, nPoints)*-1.0
        param_dict['HMM']['scale'] = 6.111 #7.0
        if opt.bNoUpdate: param_dict['ROC']['update_list'] = []        
        evaluation_single_ad(subjects, opt.task, raw_data_path, save_data_path, param_dict, \
                             save_pdf=opt.bSavePdf, \
                             verbose=opt.bVerbose, debug=opt.bDebug, no_plot=opt.bNoPlot, \
                             find_param=False, data_gen=opt.bDataGen, target_class=target_class)
                             ## find_param=False, data_gen=opt.bDataGen)


    elif opt.evaluation_double:


        # TODO: change feature name
        param_dict['ROC']['methods'] = ['hmmgp', 'hmmgp']
        param_dict['HMM']['scale'] = [6.111, 6.111]
        param_dict['ROC']['hmmgp_param_range'] = np.logspace(-0.6, 2.3, nPoints)*-1.0
        param_dict['SVM']['hmmgp_logp_offset'] = 70.0 #50.0
        param_dict['SVM']['nugget'] = 10.0

        # 80%
        save_data_path = os.path.expanduser('~')+\
          '/hrl_file_server/dpark_data/anomaly/AURO2016/'+opt.task+'_data_isolation6/'+\
          str(param_dict['data_param']['downSampleSize'])+'_'+str(opt.dim)        
        param_dict['data_param']['handFeatures'] = [['audioWristRMS', 'ftForce_z', \
                                                      'landmarkEEDist', 'kinJntEff_1'],
                                                      ['ftForce_mag_integ', 'landmarkEEDist']  ]
        param_dict['SVM']['hmmgp_logp_offset'] = 30.0 #50.0
        param_dict['ROC']['hmmgp_param_range'] = np.logspace(-0.9, 2.0, nPoints)*-1.0+1.0
        
        save_data_path = os.path.expanduser('~')+\
          '/hrl_file_server/dpark_data/anomaly/AURO2016/'+opt.task+'_data_isolation8/'+\
          str(param_dict['data_param']['downSampleSize'])+'_'+str(opt.dim)
        param_dict['data_param']['handFeatures'] = [['audioWristRMS', 'ftForce_z', \
                                                      'landmarkEEDist', 'kinJntEff_1'],
                                                    ['audioWristRMS', 'ftForce_z', \
                                                      'landmarkEEDist', 'kinJntEff_1']]
        param_dict['SVM']['hmmgp_logp_offset'] = 30.0 #50.0
        param_dict['ROC']['hmmgp_param_range'] = np.logspace(-0.6, 2.3, nPoints)*-1.0+1.0

        ## save_data_path = os.path.expanduser('~')+\
        ##   '/hrl_file_server/dpark_data/anomaly/AURO2016/'+opt.task+'_data_isolation7/'+\
        ##   str(param_dict['data_param']['downSampleSize'])+'_'+str(opt.dim)        
        ## param_dict['data_param']['handFeatures'] = [['audioWristRMS', 'ftForce_mag_integ', \
        ##                                               'landmarkEEDist', 'kinJntEff_1'],
        ##                                             ['audioWristRMS', 'ftForce_mag_integ', \
        ##                                               'landmarkEEDist', 'kinJntEff_1']]
        ## param_dict['SVM']['hmmgp_logp_offset'] = 30.0 #50.0
        ## param_dict['ROC']['hmmgp_param_range'] = np.logspace(-2.0, 2.3, nPoints)*-1.0+2.0

        ## save_data_path = os.path.expanduser('~')+\
        ##   '/hrl_file_server/dpark_data/anomaly/AURO2016/'+opt.task+'_data_isolation7/'+\
        ##   str(param_dict['data_param']['downSampleSize'])+'_'+str(opt.dim)
        ## param_dict['data_param']['handFeatures'] = [ ['ftForce_mag_integ', 'landmarkEEDist'],
        ##                                              ['ftForce_mag_integ', 'landmarkEEDist']  ]
                                                     
        
        if opt.bNoUpdate: param_dict['ROC']['update_list'] = []        
        evaluation_double_ad(subjects, opt.task, raw_data_path, save_data_path, param_dict, \
                             save_pdf=opt.bSavePdf, \
                             verbose=opt.bVerbose, debug=opt.bDebug, no_plot=opt.bNoPlot, \
                             find_param=False, data_gen=opt.bDataGen)

    elif opt.evaluation_isolation:

        save_data_path = os.path.expanduser('~')+\
          '/hrl_file_server/dpark_data/anomaly/AURO2016/'+opt.task+'_data_isolation10/'+\
          str(param_dict['data_param']['downSampleSize'])+'_'+str(opt.dim)        
        param_dict['data_param']['handFeatures'] = [['audioWristRMS', 'ftForce_z', \
                                                      'landmarkEEDist', 'kinJntEff_1'],
                                                      ['ftForce_mag_integ', 'landmarkEEDist']  ]
        param_dict['SVM']['hmmgp_logp_offset'] = 30.0 #50.0
        param_dict['ROC']['hmmgp_param_range'] = np.logspace(-0.6, 2.3, nPoints)*-1.0+1.0

        evaluation_isolation(subjects, opt.task, raw_data_path, save_data_path, param_dict, \
                             save_pdf=opt.bSavePdf, \
                             verbose=opt.bVerbose, debug=opt.bDebug, no_plot=opt.bNoPlot, \
                             find_param=False, data_gen=opt.bDataGen)
