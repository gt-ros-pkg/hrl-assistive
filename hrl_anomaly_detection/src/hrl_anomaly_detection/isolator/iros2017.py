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

# Private utils
from hrl_anomaly_detection.util import *
from hrl_anomaly_detection.util_viz import *
from hrl_anomaly_detection import data_manager as dm
from hrl_anomaly_detection import util as util
from hrl_execution_monitor import util as autil

# Private learners
from hrl_anomaly_detection.hmm import learning_hmm as hmm
import hrl_anomaly_detection.classifiers.classifier as cf
import hrl_anomaly_detection.data_viz as dv
import hrl_anomaly_detection.isolator.isolation_util as iutil

from joblib import Parallel, delayed

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

    #temp
    kFold_list = kFold_list[:8]

    ## x_classes = ['Object collision', 'Noisy environment', 'Spoon miss by a user', 'Spoon collision by a user', 'Robot-body collision by a user', 'Aggressive eating', 'Anomalous sound from a user', 'Unreachable mouth pose', 'Face occlusion by a user', 'Spoon miss by system fault', 'Spoon collision by system fault', 'Freeze by system fault']

    # select feature for detection #[0,1,2,11,19]
    base_idx = []
    for feature in param_dict['data_param']['handFeatures']:
        idx = [ i for i, x in enumerate(param_dict['data_param']['isolationFeatures']) if feature == x][0]
        base_idx.append(idx)

    org_processed_data_path = copy.copy(processed_data_path)
    for i in xrange(len(success_data)):

        if i in base_idx: continue
        successData = copy.deepcopy(d['successData'][base_idx+[i]] )
        failureData = copy.deepcopy(d['failureData'][base_idx+[i]] )

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

        # parallelization
        if debug: n_jobs=1
        else: n_jobs=-1
        l_data = Parallel(n_jobs=n_jobs, verbose=10)(delayed(cf.run_classifiers)( idx, processed_data_path, \
                                                                             task_name, \
                                                                             method, ROC_data, \
                                                                             ROC_dict, \
                                                                             SVM_dict, HMM_dict, \
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
    for idx in xrange(len(d['successData'])):
        processed_data_path = os.path.join(org_processed_data_path, str(idx))
        roc_pkl = os.path.join(processed_data_path, 'roc_'+task_name+'.pkl')
        if os.path.isfile(roc_pkl) is False: continue
        ROC_data = ut.load_pickle(roc_pkl)        
        ## auc = roc_info(ROC_data, nPoints, no_plot=True, verbose=False)


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
        print idx , auc, " - ", fpr_l[0], fpr_l[-1]
        ## print idx , auc, " - ", fpr_l


def evaluation_omp_isolation(subject_names, task_name, raw_data_path, processed_data_path, param_dict,\
                             data_renew=False, svd_renew=False, save_pdf=False, verbose=False, debug=False,\
                             no_plot=False, delay_plot=True, find_param=False, data_gen=False, \
                             save_viz_data=False, weight=-5.0):
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
                           handFeatures=param_dict['data_param']['isolationFeatures'], \
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

    # ---------------------------------------------------------------
    # select feature for detection
    feature_list = []
    for feature in param_dict['data_param']['handFeatures']:
        idx = [ i for i, x in enumerate(param_dict['data_param']['isolationFeatures']) if feature == x][0]
        feature_list.append(idx)
    
    successData_ad = successData[feature_list]
    failureData_ad = failureData[feature_list]

    dm.saveHMMinducedFeatures(kFold_list, successData_ad, failureData_ad,\
                              task_name, processed_data_path,\
                              HMM_dict, data_renew, startIdx, nState, cov, \
                              success_files=success_files, failure_files=failure_files,\
                              noise_mag=0.03, verbose=verbose)
    
    # ---------------------------------------------------------------
    # select features for isolation
    feature_list = [0,1,2,11,15,16,17,18,20,21]
    successData_ai = successData[feature_list]
    failureData_ai = failureData[feature_list]

    #temp
    kFold_list = kFold_list[:8]

    # k-fold cross validation
    data_pkl = os.path.join(processed_data_path, 'isol_data.pkl')
    if os.path.isfile(data_pkl) is False or svd_renew:
        n_jobs = -1
        l_data = Parallel(n_jobs=n_jobs, verbose=10)\
          (delayed(iutil.get_isolation_data)( idx, kFold_list[idx],\
                                              os.path.join(processed_data_path, \
                                                           'hmm_'+task_name+'_'+str(idx)+'.pkl'),\
                                                nState, failureData_ad, failureData_ai, failure_files,\
                                                failure_labels,\
                                                task_name, processed_data_path, param_dict, weight,\
                                                n_jobs=1)
          for idx in xrange(len(kFold_list)) )
        data_dict = {}
        for i in xrange(len(l_data)):
            idx = l_data[i][0]
            data_dict[idx] = (l_data[i][1],l_data[i][2],l_data[i][3],l_data[i][4] )
            
        print "save pkl: ", data_pkl
        ut.save_pickle(data_dict, data_pkl)            
    else:
        data_dict = ut.load_pickle(data_pkl)


    # ---------------------------------------------------------------
    scores = []
    for idx, (normalTrainIdx, abnormalTrainIdx, normalTestIdx, abnormalTestIdx) \
      in enumerate(kFold_list):
        print "kFold_list: ", idx

        (gs_train, y_train, gs_test, y_test) = data_dict[idx]

        if type(gs_train) is list:
            gs_train = gs_train.tolist()
            y_train  = y_train.tolist()
            gs_test  = gs_test.tolist()
            y_test   = y_test.tolist()
        print np.shape( gs_train ), np.shape( y_train ), np.shape( gs_test ), np.shape( y_test )
        
        from sklearn.svm import SVC
        clf = SVC(C=1.0, kernel='linear') #, decision_function_shape='ovo')
        clf.fit(gs_train, y_train)
        ## y_pred = clf.predict(gs_test.tolist())
        score = clf.score(gs_test, y_test)
        scores.append( score )
        print idx, " = ", score
            
    print scores
    print np.mean(scores), np.std(scores)



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
                           handFeatures=data_dict['isolationFeatures'], \
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


    # select feature for detection
    feature_list = []
    for feature in param_dict['data_param']['handFeatures']:
        idx = [ i for i, x in enumerate(param_dict['data_param']['isolationFeatures']) if feature == x][0]
        feature_list.append(idx)
    
    successData = successData[feature_list]
    failureData = failureData[feature_list]

    # temp
    kFold_list = kFold_list[:8]

    #-----------------------------------------------------------------------------------------    
    # Training HMM, and getting classifier training and testing data
    dm.saveHMMinducedFeatures(kFold_list, successData, failureData,\
                              task_name, processed_data_path,\
                              HMM_dict, data_renew, startIdx, nState, cov, \
                              success_files=success_files, failure_files=failure_files,\
                              noise_mag=0.03, diag=False, cov_type='full', \
                              verbose=verbose)

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
                                                                         ROC_dict, \
                                                                         SVM_dict, HMM_dict, \
                                                                         startIdx=startIdx, nState=nState,\
                                                                         n_jobs=n_jobs) \
                                                                         for idx in xrange(len(kFold_list)) \
                                                                         for method in method_list )

    print "finished to run run_classifiers"
    ROC_data = util.update_roc_data(ROC_data, l_data, nPoints, method_list)
    ut.save_pickle(ROC_data, roc_pkl)

    # ---------------- ROC Visualization ----------------------
    roc_info(ROC_data, nPoints, no_plot=no_plot, ROC_dict=ROC_dict)
    class_info(method_list, ROC_data, nPoints, kFold_list)


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
        kFold_list    = d['kFoldList'] 
        successData   = d['successData']
        failureData   = d['failureData']        
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
                           data_renew=data_renew, max_time=data_dict['max_time'],
                           ros_bag_image=True)
                           
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

    #-----------------------------------------------------------------------------------------
    # feature selection
    print d['param_dict']['feature_names']    
    feature_idx_list = []
    for i in xrange(2):

        feature_idx_list.append([])
        for feature in param_dict['data_param']['handFeatures'][i]:
            feature_idx_list[i].append(data_dict['isolationFeatures'].index(feature))

        success_data_ad = copy.copy(successData[feature_idx_list[i]])
        failure_data_ad = copy.copy(failureData[feature_idx_list[i]])
        HMM_dict_local = copy.deepcopy(HMM_dict)
        HMM_dict_local['scale'] = param_dict['HMM']['scale'][i]

        ## #temp
        ## if i==0: continue

        # Training HMM, and getting classifier training and testing data
        dm.saveHMMinducedFeatures(kFold_list, success_data_ad, failure_data_ad,\
                                  task_name, processed_data_path,\
                                  HMM_dict_local, data_renew, startIdx, nState, cov, \
                                  success_files=success_files, failure_files=failure_files,\
                                  noise_mag=0.03, diag=False, suffix=str(i),\
                                  verbose=verbose)

    print "Finished to save hmm data"

    #-----------------------------------------------------------------------------------------
    roc_pkl = os.path.join(processed_data_path, 'roc_'+task_name+'.pkl')

    if os.path.isfile(roc_pkl) is False or HMM_dict['renew'] or SVM_dict['renew']: ROC_data = {}
    else: ROC_data = ut.load_pickle(roc_pkl)
    ROC_data = util.reset_roc_data(ROC_data, [method_list[0][:-1]], ROC_dict['update_list'], nPoints)

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
    ROC_data = util.update_roc_data(ROC_data, l_data, nPoints, [method_list[0][:-1]])
    ut.save_pickle(ROC_data, roc_pkl)

    
    # ---------------- ROC Visualization ----------------------
    roc_info(ROC_data, nPoints, no_plot=True, multi_ad=True, ROC_dict=ROC_dict)
    ## class_info(method_list, ROC_data, nPoints, kFold_list)




def evaluation_isolation2(subject_names, task_name, raw_data_path, processed_data_path, param_dict,\
                          data_renew=False, svd_renew=False, save_pdf=False, verbose=False, debug=False,\
                          no_plot=False, delay_plot=True, find_param=False, \
                          save_viz_data=False, weight=-5.0, window_steps=10, single_detector=False):
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

    if os.path.isfile(crossVal_pkl) and data_renew is False :
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
                           handFeatures=param_dict['data_param']['isolationFeatures'], \
                           cut_data=data_dict['cut_data'], \
                           data_renew=data_renew, max_time=data_dict['max_time'],\
                           ros_bag_image=True)
                           
        successData, failureData, success_files, failure_files, kFold_list \
          = dm.LOPO_data_index(d['successDataList'], d['failureDataList'],\
                               d['successFileList'], d['failureFileList'])

        d['successData'] = successData
        d['failureData'] = failureData
        d['success_files']   = success_files
        d['failure_files']   = failure_files
        d['kFoldList']       = kFold_list
        ut.save_pickle(d, crossVal_pkl)

    # flattening image list
    success_image_list = autil.image_list_flatten( d.get('success_image_list',[]) )
    failure_image_list = autil.image_list_flatten( d.get('failure_image_list',[]) )

    failure_labels = []
    for f in failure_files:
        failure_labels.append( int( f.split('/')[-1].split('_')[0] ) )
    failure_labels = np.array( failure_labels )



    #temp
    kFold_list = kFold_list[:8]


    #-----------------------------------------------------------------------------------------
    # Dynamic feature selection for detection and isolation
    print d['param_dict']['feature_names']    
    feature_idx_list = []
    success_data_ad = []
    failure_data_ad = []
    nDetector = len(param_dict['data_param']['handFeatures'])
    for i in xrange(nDetector):
        
        feature_idx_list.append([])
        for feature in param_dict['data_param']['handFeatures'][i]:
            feature_idx_list[i].append(data_dict['isolationFeatures'].index(feature))

        success_data_ad.append( copy.copy(successData[feature_idx_list[i]]) )
        failure_data_ad.append( copy.copy(failureData[feature_idx_list[i]]) )
        HMM_dict_local = copy.deepcopy(HMM_dict)
        HMM_dict_local['scale'] = param_dict['HMM']['scale'][i]
        
        # Training HMM, and getting classifier training and testing data
        dm.saveHMMinducedFeatures(kFold_list, success_data_ad[i], failure_data_ad[i],\
                                  task_name, processed_data_path,\
                                  HMM_dict_local, data_renew, startIdx, nState, cov, \
                                  noise_mag=0.03, diag=False, suffix=str(i),\
                                  verbose=verbose, one_class=False)

    # Static feature selection for isolation
    feature_list = []
    for feature in param_dict['data_param']['staticFeatures']:
        idx = [ i for i, x in enumerate(param_dict['data_param']['isolationFeatures']) if feature == x][0]
        feature_list.append(idx)
    successData_static = np.array(successData)[feature_list]
    failureData_static = np.array(failureData)[feature_list]


    #-----------------------------------------------------------------------------------------
    roc_pkl = os.path.join(processed_data_path, 'roc_'+task_name+'.pkl')

    if os.path.isfile(roc_pkl) is False or HMM_dict['renew'] or SVM_dict['renew']: ROC_data = {}
    else: ROC_data = ut.load_pickle(roc_pkl)
    ROC_data = util.reset_roc_data(ROC_data, [method_list[0][:-1]], ROC_dict['update_list'], nPoints)

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
    ROC_data = util.update_roc_data(ROC_data, l_data, nPoints, [method_list[0][:-1]])
    ut.save_pickle(ROC_data, roc_pkl)

    weight = util.get_best_weight(ROC_data, nPoints, ROC_dict)
    #-----------------------------------------------------------------------------------------
    # Training HMM, and getting classifier training and testing data
    data_dict = {}
    data_pkl = os.path.join(processed_data_path, 'isol_data.pkl')
    if os.path.isfile(data_pkl) is False or HMM_dict['renew'] or SVM_dict['renew'] or svd_renew:

        l_data = Parallel(n_jobs=1, verbose=10)\
          (delayed(iutil.get_hmm_isolation_data)(idx, kFold_list[idx], failure_data_ad, \
                                                 failureData_static, \
                                                 failure_labels,\
                                                 failure_image_list,\
                                                 task_name, processed_data_path, param_dict, weight,\
                                                 single_detector=single_detector,\
                                                 n_jobs=-1, window_steps=window_steps, verbose=verbose\
                                                 ) for idx in xrange(len(kFold_list)) )
        
        data_dict = {}
        for i in xrange(len(l_data)):
            idx = l_data[i][0]
            data_dict[idx] = (l_data[i][1],l_data[i][2],l_data[i][3],l_data[i][4] )
            
        print "save pkl: ", data_pkl
        ut.save_pickle(data_dict, data_pkl)            
    else:
        data_dict = ut.load_pickle(data_pkl)


    # ---------------------------------------------------------------
    scores = []
    for idx, (normalTrainIdx, abnormalTrainIdx, normalTestIdx, abnormalTestIdx) \
      in enumerate(kFold_list):
        print "kFold_list: ", idx

        (x_trains, y_train, x_tests, y_test) = data_dict[idx]         
        x_train = x_trains[0] 
        x_test  = x_tests[0]
        print np.shape(x_trains[0]), np.shape(x_trains[1]), np.shape(y_train)

        from sklearn import preprocessing
        scaler = preprocessing.StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test  = scaler.transform(x_test)

        if type(x_train) is np.ndarray:
            x_train = x_train.tolist()
            x_test  = x_test.tolist()
        if type(y_train) is np.ndarray:
            y_train  = y_train.tolist()
            y_test   = y_test.tolist()
        
        ## from sklearn.svm import SVC
        ## clf = SVC(C=1.0, kernel='rbf') #, decision_function_shape='ovo')
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=400, n_jobs=-1)

        clf.fit(x_train, y_train)
        ## y_pred = clf.predict(x_test.tolist())
        score = clf.score(x_test, y_test)
        scores.append( score )
        print idx, " : score = ", score


    # ---------------- ROC Visualization ----------------------
    roc_info(ROC_data, nPoints, no_plot=True, multi_ad=True, ROC_dict=ROC_dict)
    
    print scores
    print "Classification Score mean = ", np.mean(scores), np.std(scores)

    #temp
    ## iutil.save_data_labels(x_train, y_train)










if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    util.initialiseOptParser(p)

    p.add_option('--eval_single', '--es', action='store_true', dest='evaluation_single',
                 default=False, help='Evaluate with single detector.')
    p.add_option('--eval_double', '--ed', action='store_true', dest='evaluation_double',
                 default=False, help='Evaluate with double detectors.')
    p.add_option('--eval_omp_isol', '--eoi', action='store_true', dest='evaluation_omp_isolation',
                 default=False, help='Evaluate anomaly isolation with omp.')
    p.add_option('--eval_isol', '--ei', action='store_true', dest='evaluation_isolation',
                 default=False, help='Evaluate anomaly isolation with double detectors.')
    p.add_option('--eval_isol2', '--ei2', action='store_true', dest='evaluation_isolation2',
                 default=False, help='Evaluate anomaly isolation with double detectors.')
    p.add_option('--svd_renew', '--sr', action='store_true', dest='svd_renew',
                 default=False, help='Renew ksvd')
    
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
                                                          opt.bHMMRenew, opt.bCLFRenew, opt.dim,\
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
          '/hrl_file_server/dpark_data/anomaly/AURO2016/'+opt.task+'_data_isolation6/'+\
          str(param_dict['data_param']['downSampleSize'])+'_'+str(opt.dim)
        param_dict['data_param']['handFeatures'] = [## 'unimodal_audioWristRMS', \
                                                    ## 'unimodal_audioWristFrontRMS', \
                                                    ## 'unimodal_audioWristAzimuth',\
                                                    'unimodal_kinVel',\
                                                    ## 'unimodal_kinJntEff', \
                                                    'unimodal_ftForce_integ', \
                                                    ## 'unimodal_ftForce_delta', \
                                                    ## 'unimodal_ftForce_zero', \
                                                    ## 'unimodal_ftForce', \
                                                    ## 'unimodal_ftForceX', \
                                                    ## 'unimodal_ftForceY', \
                                                    ## 'unimodal_ftForceZ', \
                                                    ## 'unimodal_kinEEChange', \
                                                    'unimodal_kinDesEEChange', \
                                                    'crossmodal_landmarkEEDist', \
                                                    ## 'crossmodal_landmarkEEAng',\
                                                    ## 'unimodal_fabricForce',\
                                                    ## 'unimodal_landmarkDist'
                                                    ]
        ## target_class = [13]

                                                    
        dm.getDataLOPO(subjects, opt.task, raw_data_path, save_data_path,
                       param_dict['data_param']['rf_center'], param_dict['data_param']['local_range'],\
                       downSampleSize=param_dict['data_param']['downSampleSize'], \
                       success_viz=success_viz, failure_viz=failure_viz,\
                       cut_data=param_dict['data_param']['cut_data'],\
                       save_pdf=opt.bSavePdf, solid_color=True,\
                       handFeatures=param_dict['data_param']['handFeatures'], data_renew=opt.bDataRenew, \
                       ## max_time=param_dict['data_param']['max_time'])
                       max_time=param_dict['data_param']['max_time']) #, target_class=target_class)

    elif opt.bLikelihoodPlot:
        save_data_path = os.path.expanduser('~')+\
          '/hrl_file_server/dpark_data/anomaly/AURO2016/'+opt.task+'_data_isolation4/'+\
          str(param_dict['data_param']['downSampleSize'])+'_'+str(opt.dim)
        param_dict['HMM']['scale'] = 7.11
        param_dict['data_param']['handFeatures'] = ['unimodal_audioWristRMS',  \
                                                    'unimodal_kinJntEff_1',\
                                                    'unimodal_ftForce_integ',\
                                                    'unimodal_ftForceZ',\
                                                    'unimodal_kinEEChange', \
                                                    'crossmodal_landmarkEEDist', \
                                                    'unimodal_landmarkDist',\
                                                    ]
        ## param_dict['data_param']['handFeatures'] = ['unimodal_audioWristRMS', 'unimodal_ftForceZ', \
        ##                                             'crossmodal_landmarkEEDist', 'unimodal_kinJntEff_1']
        #'crossmodal_landmarkEEDist'        
        param_dict['ROC']['hmmgp_param_range'] = np.logspace(-0.6, 2.3, nPoints)*-1.0

        import hrl_anomaly_detection.data_viz as dv        
        dv.vizLikelihoods(subjects, opt.task, raw_data_path, save_data_path, param_dict,\
                          decision_boundary_viz=False, method='progress', \
                          useTrain=True, useNormalTest=False, useAbnormalTest=True,\
                          useTrain_color=False, useNormalTest_color=False, useAbnormalTest_color=False,\
                          hmm_renew=opt.bHMMRenew, data_renew=opt.bDataRenew, save_pdf=opt.bSavePdf,\
                          verbose=opt.bVerbose, lopo=True, plot_feature=False)

    elif opt.evaluation_omp_isolation:

        # c11 offset 0 weight -8 spar 0.05, dict 8, win_size 130 #74%
        save_data_path = os.path.expanduser('~')+\
          '/hrl_file_server/dpark_data/anomaly/AURO2016/'+opt.task+'_data_isolation10/'+\
          str(param_dict['data_param']['downSampleSize'])+'_'+str(opt.dim)
        weight = -18.0
        param_dict['SVM']['hmmgp_logp_offset'] = 0.0 #30.0 

        param_dict['data_param']['handFeatures'] = ['unimodal_audioWristRMS',  \
                                                    'unimodal_kinJntEff_1',\
                                                    'unimodal_ftForce_integ',\
                                                    'unimodal_kinEEChange', \
                                                    'crossmodal_landmarkEEDist', \
                                                    ]

        param_dict['ROC']['methods'] = ['hmmgp']
        nPoints = param_dict['ROC']['nPoints']
        param_dict['ROC']['hmmgp_param_range'] = np.logspace(-0.6, 2.3, nPoints)*-1.0
        param_dict['HMM']['scale'] = 6.111 
        if opt.bNoUpdate: param_dict['ROC']['update_list'] = []        
            
        evaluation_omp_isolation(subjects, opt.task, raw_data_path, save_data_path, param_dict, \
                                 data_renew=opt.bDataRenew, svd_renew=opt.svd_renew,\
                                 save_pdf=opt.bSavePdf, \
                                 verbose=opt.bVerbose, debug=opt.bDebug, no_plot=opt.bNoPlot, \
                                 find_param=False, data_gen=opt.bDataGen, weight=weight)


    elif opt.bEvaluationAll:
        '''
        feature-wise evaluation
        '''        
        save_data_path = os.path.expanduser('~')+\
          '/hrl_file_server/dpark_data/anomaly/AURO2016/'+opt.task+'_data_isolation3/'+\
          str(param_dict['data_param']['downSampleSize'])+'_'+str(opt.dim)

        # 87.95
        param_dict['data_param']['handFeatures'] = ['unimodal_audioWristRMS',  \
                                                    'unimodal_audioWristAzimuth',\
                                                    'unimodal_kinJntEff_1',\
                                                    'unimodal_ftForce',\
                                                    'unimodal_ftForce_integ',\
                                                    'unimodal_landmarkDist',\
                                                    'unimodal_kinEEChange',\
                                                    'crossmodal_landmarkEEDist', \
                                                    ]

        param_dict['ROC']['methods'] = ['hmmgp']
        param_dict['HMM']['scale'] = 7.11
        param_dict['ROC']['hmmgp_param_range'] = np.logspace(-0.6, 2.5, nPoints)*-1.0
        if opt.bNoUpdate: param_dict['ROC']['update_list'] = []        
        evaluation_all(subjects, opt.task, raw_data_path, save_data_path, param_dict, save_pdf=opt.bSavePdf, \
                       verbose=opt.bVerbose, debug=opt.bDebug, no_plot=opt.bNoPlot, \
                       find_param=False, data_gen=opt.bDataGen)
                       ## find_param=False, data_gen=opt.bDataGen, target_class=target_class)


    elif opt.evaluation_single:
        '''
        evaluation with selected feature set 5,6
        '''
        nPoints = param_dict['ROC']['nPoints']
        

        # c12  13-76 9-76 11-77 15-78 17-79 20-79
        ## save_data_path = os.path.expanduser('~')+\
        ##   '/hrl_file_server/dpark_data/anomaly/AURO2016/'+opt.task+'_data_isolation6/'+\
        ##   str(param_dict['data_param']['downSampleSize'])+'_'+str(opt.dim)
        ## param_dict['data_param']['handFeatures'] = ['unimodal_kinVel',\
        ##                                              'unimodal_ftForce_zero',\
        ##                                              ## 'unimodal_kinDesEEChange', \
        ##                                              'crossmodal_landmarkEEDist', \
        ##                                             ]        
        ## param_dict['HMM']['scale'] = 15.0
        ## param_dict['ROC']['hmmgp_param_range'] = np.logspace(-0.0, 2.5, nPoints)*-1.0 +0.5
        ## param_dict['ROC']['methods'] = ['hmmgp']
       
        ## ep 12-89.5 7-82
        save_data_path = os.path.expanduser('~')+\
          '/hrl_file_server/dpark_data/anomaly/AURO2016/'+opt.task+'_data_isolation4/'+\
          str(param_dict['data_param']['downSampleSize'])+'_'+str(opt.dim)
        param_dict['data_param']['handFeatures'] = ['unimodal_audioWristRMS',  \
                                                    'unimodal_kinJntEff_1',\
                                                    'unimodal_ftForce_integ',\
                                                    'unimodal_kinEEChange',\
                                                    'crossmodal_landmarkEEDist', \
                                                    ]
        param_dict['HMM']['scale'] = 7.0
        param_dict['ROC']['progress_param_range'] = -np.logspace(-0.2, 1.4, nPoints)+1.0
        param_dict['ROC']['methods'] = ['progress']

        ## c11 13-90.48 9-85.8
        ## save_data_path = os.path.expanduser('~')+\
        ##   '/hrl_file_server/dpark_data/anomaly/AURO2016/'+opt.task+'_data_isolation5/'+\
        ##   str(param_dict['data_param']['downSampleSize'])+'_'+str(opt.dim)
        ## param_dict['data_param']['handFeatures'] = ['unimodal_audioWristRMS',  \
        ##                                             'unimodal_kinJntEff_1',\
        ##                                             'unimodal_ftForce_integ',\
        ##                                             'unimodal_kinEEChange',\
        ##                                             'crossmodal_landmarkEEDist', \
        ##                                             ]
        ## param_dict['HMM']['scale'] = 9.0
        ## param_dict['ROC']['progress_param_range'] = -np.logspace(-0.2, 1.5, nPoints)
        ## param_dict['ROC']['methods'] = ['progress']

        # c12 12-82 14-83.54 15-84.55
        ## save_data_path = os.path.expanduser('~')+\
        ##   '/hrl_file_server/dpark_data/anomaly/AURO2016/'+opt.task+'_data_isolation6/'+\
        ##   str(param_dict['data_param']['downSampleSize'])+'_'+str(opt.dim)
        ## param_dict['data_param']['handFeatures'] = ['unimodal_kinVel',\
        ##                                              'unimodal_ftForce_zero',\
        ##                                              'crossmodal_landmarkEEDist', \
        ##                                             ]        
        ## param_dict['HMM']['scale'] = 12.0
        ## param_dict['ROC']['methods'] = ['progress']
        ## param_dict['ROC']['progress_param_range'] = -np.logspace(-0.4, 1.2, nPoints)

        # c8 10.0-79
        ## save_data_path = os.path.expanduser('~')+\
        ##   '/hrl_file_server/dpark_data/anomaly/AURO2016/'+opt.task+'_data_isolation7/'+\
        ##   str(param_dict['data_param']['downSampleSize'])+'_'+str(opt.dim)
        ## param_dict['data_param']['handFeatures'] = ['unimodal_kinVel',\
        ##                                              'unimodal_ftForce_zero',\
        ##                                              'crossmodal_landmarkEEDist', \
        ##                                             ]        
        ## param_dict['HMM']['scale'] = 13.0
        ## param_dict['ROC']['methods'] = ['progress']
        ## param_dict['ROC']['progress_param_range'] = -np.logspace(-0.2, 1.1, nPoints)



        if opt.bNoUpdate: param_dict['ROC']['update_list'] = []        
        evaluation_single_ad(subjects, opt.task, raw_data_path, save_data_path, param_dict, \
                             save_pdf=opt.bSavePdf, \
                             verbose=opt.bVerbose, debug=opt.bDebug, no_plot=opt.bNoPlot, \
                             find_param=False, data_gen=opt.bDataGen)
                             ## find_param=False, data_gen=opt.bDataGen, target_class=target_class)


    elif opt.evaluation_double:

        # TODO: change feature name
        param_dict['ROC']['methods'] = ['hmmgp0', 'hmmgp1']
        param_dict['SVM']['nugget']  = 10.0

        # -------------------------------------------------------------------------------------
        ## c8 77-83.5
        save_data_path = os.path.expanduser('~')+\
          '/hrl_file_server/dpark_data/anomaly/AURO2016/'+opt.task+'_data_isolation8/'+\
          str(param_dict['data_param']['downSampleSize'])+'_'+str(opt.dim)
        param_dict['ROC']['methods'] = ['progress0', 'progress1']
        param_dict['HMM']['scale']   = [7.0, 7.0]
        param_dict['HMM']['cov']     = 1.0
        
        ## c11 0713-89
        save_data_path = os.path.expanduser('~')+\
          '/hrl_file_server/dpark_data/anomaly/AURO2016/'+opt.task+'_data_isolation5/'+\
          str(param_dict['data_param']['downSampleSize'])+'_'+str(opt.dim)
        param_dict['ROC']['methods'] = ['progress0', 'progress1']
        param_dict['HMM']['scale']   = [7.0, 13.0]
        param_dict['HMM']['cov']     = 1.0

        ## c12
        save_data_path = os.path.expanduser('~')+\
          '/hrl_file_server/dpark_data/anomaly/AURO2016/'+opt.task+'_data_isolation6/'+\
          str(param_dict['data_param']['downSampleSize'])+'_'+str(opt.dim)
        param_dict['ROC']['methods'] = ['progress0', 'progress1']
        param_dict['HMM']['scale']   = [9.0, 9.0]
        param_dict['HMM']['cov']     = 1.0

        ## ## ep
        ## save_data_path = os.path.expanduser('~')+\
        ##   '/hrl_file_server/dpark_data/anomaly/AURO2016/'+opt.task+'_data_isolation4/'+\
        ##   str(param_dict['data_param']['downSampleSize'])+'_'+str(opt.dim)
        ## param_dict['ROC']['methods'] = ['progress0', 'progress1']
        ## param_dict['HMM']['scale']   = [7.0, 5.0]
        ## param_dict['HMM']['cov']     = 1.0

        
        param_dict['data_param']['handFeatures'] = [['unimodal_audioWristRMS',  \
                                                    'unimodal_kinJntEff_1',\
                                                    'unimodal_ftForce_integ',\
                                                    'unimodal_kinEEChange',\
                                                    'crossmodal_landmarkEEDist', \
                                                    ],
                                                    ['unimodal_kinVel',\
                                                     'unimodal_ftForce_zero',\
                                                     'crossmodal_landmarkEEDist', \
                                                    ]]
        param_dict['SVM']['hmmgp_logp_offset'] = 0 
        param_dict['ROC']['progress0_param_range'] = -np.logspace(-0.3, 1.3, nPoints)
        param_dict['ROC']['progress1_param_range'] = -np.logspace(-0.3, 1.3, nPoints)
        param_dict['ROC']['hmmgp0_param_range'] = np.logspace(0.2, 2.5, nPoints)*-1.0+1.0
        param_dict['ROC']['hmmgp1_param_range'] = np.logspace(0.2, 2.5, nPoints)*-1.0+0.5 #2.
        # -------------------------------------------------------------------------------------
        
        if opt.bNoUpdate: param_dict['ROC']['update_list'] = []        
        evaluation_double_ad(subjects, opt.task, raw_data_path, save_data_path, param_dict, \
                             save_pdf=opt.bSavePdf, \
                             verbose=opt.bVerbose, debug=opt.bDebug, no_plot=opt.bNoPlot, \
                             find_param=False, data_gen=opt.bDataGen)


    elif opt.evaluation_isolation2:

        ## c11, #nodes 69 ============== BEST
        ## save_data_path = os.path.expanduser('~')+\
        ##   '/hrl_file_server/dpark_data/anomaly/AURO2016/'+opt.task+'_data_isolation11/'+\
        ##   str(param_dict['data_param']['downSampleSize'])+'_'+str(opt.dim)
        ## param_dict['ROC']['methods'] = ['hmmgp0', 'hmmgp1']
        ## weight = [-23.0, -35.0]  #23
        ## param_dict['HMM']['scale'] = [7.0, 13.0]
        ## single_detector = False #True

        ## ## c12 68 = maybe.. best?
        ## save_data_path = os.path.expanduser('~')+\
        ##   '/hrl_file_server/dpark_data/anomaly/AURO2016/'+opt.task+'_data_isolation10/'+\
        ##   str(param_dict['data_param']['downSampleSize'])+'_'+str(opt.dim)
        ## param_dict['ROC']['methods'] = ['hmmgp0', 'hmmgp1']
        ## weight = [-23.0, -50.0] #23
        ## param_dict['HMM']['scale'] = [7.0, 15.0]
        ## single_detector = False

        ## c11 86 (-3,-3) 65 (-3,-4.5) for demo version
        ## save_data_path = os.path.expanduser('~')+\
        ##   '/hrl_file_server/dpark_data/anomaly/AURO2016/'+opt.task+'_data_isolation9/'+\
        ##   str(param_dict['data_param']['downSampleSize'])+'_'+str(opt.dim)
        ## param_dict['ROC']['methods'] = ['progress0', 'progress1']
        ## weight = [-3.0, -4.5]
        ## param_dict['HMM']['scale'] = [2.0, 2.0]
        ## param_dict['HMM']['cov']   = 1.0
        ## single_detector = False
        
        #-----------------------------------------------------------------------------------
        # 0407-
        # 0408-80-74
        # 0410
        # 0505-79-70
        # 0507-80-72.5
        # 0508-81-72
        # 0509-82-71.7
        # 0510-83-70.6
        # 0511-           86-71
        # 0606-81-66
        # 0607-82-64
        # 0608-82-69
        # 0609-83-71
        # 0610-84-68
        # 0611-87-60
        # 0613-89-64
        # 0705-77-72
        # 0706-80-71.5
        # 0707-82-72.4
        # 07075-
        # 0708-82-65
        # 0709-82-70
        # 0713-89-63
        # 0909-83-67
        # 0913-58
        # 1212-67.5 
        
        ## c8  
        save_data_path = os.path.expanduser('~')+\
          '/hrl_file_server/dpark_data/anomaly/AURO2016/'+opt.task+'_data_isolation8/'+\
          str(param_dict['data_param']['downSampleSize'])+'_'+str(opt.dim)
        param_dict['ROC']['methods'] = ['progress0', 'progress1']
        weight = [-4.8, -4.8]
        param_dict['HMM']['scale'] = [7.0, 7.0]
        param_dict['HMM']['cov']   = 1.0
        single_detector = False 

        ## c11 
        ## save_data_path = os.path.expanduser('~')+\
        ##   '/hrl_file_server/dpark_data/anomaly/AURO2016/'+opt.task+'_data_isolation5/'+\
        ##   str(param_dict['data_param']['downSampleSize'])+'_'+str(opt.dim)
        ## param_dict['ROC']['methods'] = ['progress0', 'progress1']
        ## weight = [-4., -4.]
        ## param_dict['HMM']['scale'] = [4.0, 10.0]
        ## param_dict['HMM']['cov']   = 1.0
        ## single_detector = False


        ## c12 1010-70
        ## save_data_path = os.path.expanduser('~')+\
        ##   '/hrl_file_server/dpark_data/anomaly/AURO2016/'+opt.task+'_data_isolation6/'+\
        ##   str(param_dict['data_param']['downSampleSize'])+'_'+str(opt.dim)
        ## param_dict['ROC']['methods'] = ['progress0', 'progress1']
        ## weight = [-4.9, -4.9]
        ## param_dict['HMM']['scale'] = [7.0, 7.5]
        ## param_dict['HMM']['cov']   = 1.0
        ## single_detector = False

        ## ep
        ## save_data_path = os.path.expanduser('~')+\
        ##   '/hrl_file_server/dpark_data/anomaly/AURO2016/'+opt.task+'_data_isolation4/'+\
        ##   str(param_dict['data_param']['downSampleSize'])+'_'+str(opt.dim)
        ## param_dict['ROC']['methods'] = ['progress0', 'progress1']
        ## weight = [-3., -3.]
        ## param_dict['HMM']['scale'] = [5.0, 11.0]
        ## param_dict['HMM']['cov']   = 1.0
        ## single_detector = False



        
        window_steps= 5
        nPoints = param_dict['ROC']['nPoints']
        param_dict['SVM']['hmmgp_logp_offset'] = 0.0 #30.0
        param_dict['SVM']['nugget']  = 10.0

        param_dict['data_param']['handFeatures'] = [['unimodal_audioWristRMS',  \
                                                    'unimodal_kinJntEff_1',\
                                                    'unimodal_ftForce_integ',\
                                                    'unimodal_kinEEChange',\
                                                    'crossmodal_landmarkEEDist'
                                                    ],\
                                                    ['unimodal_kinVel',\
                                                     'unimodal_ftForce_zero',\
                                                     ## 'unimodal_kinDesEEChange',\
                                                     'crossmodal_landmarkEEDist'
                                                    ]]


        param_dict['ROC']['hmmgp_param_range'] = np.logspace(-0., 2.3, nPoints)*-1.0+1.0
        param_dict['ROC']['progress0_param_range'] = -np.logspace(0., 0.9, nPoints)
        param_dict['ROC']['progress1_param_range'] = -np.logspace(0., 0.9, nPoints)
        param_dict['ROC']['hmmgp1_param_range'] = np.logspace(-0., 2.3, nPoints)*-1.0+1.0
        param_dict['ROC']['hmmgp2_param_range'] = np.logspace(-0., 2.5, nPoints)*-1.0+0.5
        
        param_dict['data_param']['staticFeatures'] = ['unimodal_audioWristFrontRMS',\
                                                      'unimodal_audioWristAzimuth',\
                                                      'unimodal_ftForce_XY',\
                                                      ## 'unimodal_ftForceX',\
                                                      ## 'unimodal_ftForceY',\
                                                      ## 'unimodal_ftForceZ',\
                                                      'unimodal_fabricForce',  \
                                                      'unimodal_landmarkDist',\
                                                      'crossmodal_landmarkEEAng',\
                                                      ]                                                  

        if opt.bNoUpdate: param_dict['ROC']['update_list'] = []        
        evaluation_isolation2(subjects, opt.task, raw_data_path, save_data_path, param_dict, \
                              data_renew=opt.bDataRenew, svd_renew=opt.svd_renew,\
                              save_pdf=opt.bSavePdf, \
                              verbose=opt.bVerbose, debug=opt.bDebug, no_plot=opt.bNoPlot, \
                              find_param=False, weight=weight, \
                              window_steps=window_steps, single_detector=single_detector)

