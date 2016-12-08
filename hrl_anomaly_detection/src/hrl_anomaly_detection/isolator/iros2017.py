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
        successData = d['successData']
        failureData = d['failureData']        
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
                           downSampleSize=data_dict['downSampleSize'], scale=1.0,\
                           handFeatures=data_dict['handFeatures'], \
                           cut_data=data_dict['cut_data'], \
                           isolationFeatures=param_dict['data_param']['isolationFeatures'], \
                           data_renew=data_renew, max_time=data_dict['max_time'])
        successData, failureData, success_files, failure_files, kFold_list \
          = dm.LOPO_data_index(d['successDataList'], d['failureDataList'],\
                               d['successFileList'], d['failureFileList'])

        for i in xrange(len(subject_names)):
            if i==0:
                success_isol_data = d['successIsolDataList'][i]
                failure_isol_data = d['failureIsolDataList'][i]
            else:
                success_isol_data = np.vstack([ np.swapaxes(success_isol_data,0,1), \
                                                np.swapaxes(d['successIsolDataList'][i], 0,1)])
                failure_isol_data = np.vstack([ np.swapaxes(failure_isol_data,0,1), \
                                                np.swapaxes(d['failureIsolDataList'][i], 0,1)])
                success_isol_data = np.swapaxes(success_isol_data, 0, 1)
                failure_isol_data = np.swapaxes(failure_isol_data, 0, 1)

        d['successData']     = successData
        d['failureData']     = failureData
        d['successIsolData'] = success_isol_data
        d['failureIsolData'] = failure_isol_data
        d['success_files']   = success_files
        d['failure_files']   = failure_files
        d['kFoldList']       = kFold_list
        ut.save_pickle(d, crossVal_pkl)
        if data_gen: sys.exit()

    #-----------------------------------------------------------------------------------------

    #temp
    kFold_list = kFold_list[:8]

    ## x_classes = ['Object collision', 'Noisy environment', 'Spoon miss by a user', 'Spoon collision by a user', 'Robot-body collision by a user', 'Aggressive eating', 'Anomalous sound from a user', 'Unreachable mouth pose', 'Face occlusion by a user', 'Spoon miss by system fault', 'Spoon collision by system fault', 'Freeze by system fault']

    org_processed_data_path = copy.copy(processed_data_path)
    for i in xrange(len(success_isol_data)):

        successData = copy.copy(d['successIsolData'][[12,i]])
        failureData = copy.copy(d['failureIsolData'][[12,i]])
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
            continue


        #-----------------------------------------------------------------------------------------    
        # Training HMM, and getting classifier training and testing data
        try:
            dm.saveHMMinducedFeatures(kFold_list, successData, failureData,\
                                      task_name, processed_data_path,\
                                      HMM_dict, data_renew, startIdx, nState, cov, scale, \
                                      success_files=success_files, failure_files=failure_files,\
                                      noise_mag=0.03, verbose=verbose)
        except:
            print "Feature: ", i
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
    for idx in xrange(len(success_isol_data)):
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

        
        ## auc_rates = {}
        ## tp_ll = ROC_data[method]['tp_l']
        fp_ll = ROC_data[method]['fp_l']
        tn_ll = ROC_data[method]['tn_l']
        ## fn_ll = ROC_data[method]['fn_l']

        tpr_l = []
        fpr_l = []
        fnr_l = []
        for i in xrange(nPoints):
            tpr_l.append( float(np.sum(tp_ll[i]))/float(np.sum(tp_ll[i])+np.sum(fn_ll[i]))*100.0 )
            fnr_l.append( 100.0 - tpr_l[-1] )
            fpr_l.append( float(np.sum(fp_ll[i]))/float(np.sum(fp_ll[i])+np.sum(tn_ll[i]))*100.0 )

        ## print tpr_l
        ## print fpr_l

        from sklearn import metrics 
        auc = metrics.auc(fpr_l, tpr_l, True)
        ## auc_rates[method] = auc
               
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
                           downSampleSize=data_dict['downSampleSize'], scale=1.0,\
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
                              HMM_dict, data_renew, startIdx, nState, cov, scale, \
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

    (param_dict1, param_dict2) = param_dict

    ## Parameters
    # data
    data_dict  = param_dict1['data_param']
    data_renew = data_dict['renew']
    # HMM
    HMM_dict   = param_dict1['HMM']
    nState     = HMM_dict['nState']
    cov        = HMM_dict['cov']
    # SVM
    SVM_dict   = param_dict1['SVM']

    # ROC
    ROC_dict = param_dict1['ROC']

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
                           downSampleSize=data_dict['downSampleSize'], scale=1.0,\
                           handFeatures=data_dict['handFeatures'], \
                           cut_data=data_dict['cut_data'], \
                           isolationFeatures=param_dict['data_param']['isolationFeatures'], \
                           data_renew=data_renew, max_time=data_dict['max_time'])
                           
        success_isol_data, failure_isol_data, success_files, failure_files, kFold_list \
          = dm.LOPO_data_index(d['successIsolDataList'], d['failureIsolDataList'],\
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

    print d['param_dict'].keys()
    sys.exit()
    ## param_dict1
    
    #-----------------------------------------------------------------------------------------    
    # Training HMM, and getting classifier training and testing data
    dm.saveHMMinducedFeatures(kFold_list, successData, failureData,\
                              task_name, processed_data_path,\
                              HMM_dict, data_renew, startIdx, nState, cov, scale, \
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




if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    util.initialiseOptParser(p)

    p.add_option('--eval_single', '--es', action='store_true', dest='evaluation_single',
                 default=False, help='Evaluate with single detector.')
    p.add_option('--eval_double', '--ed', action='store_true', dest='evaluation_double',
                 default=False, help='Evaluate with double detectors.')
    
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
        ## param_dict['data_param']['handFeatures'] = ['unimodal_landmarkDist', 'crossmodal_landmarkEEDist', \
        ##                                             'unimodal_kinJntEff_4', 'unimodal_kinDesEEChange']        
        param_dict['data_param']['handFeatures'] = ['unimodal_kinDesEEChange',\
                                                    ## 'unimodal_kinEEChange',\
                                                    ## 'unimodal_audioWristRMS', \
                                                    ## 'unimodal_audioWristFrontRMS', \
                                                    ## 'unimodal_audioWristAzimuth',\
                                                    ## 'unimodal_kinJntEff', \
                                                    'unimodal_ftForce_delta', \
                                                    'unimodal_ftForce_zero', \
                                                    'unimodal_ftForce', \
                                                    ## 'unimodal_ftForceX', \
                                                    ## 'unimodal_ftForceY', \
                                                    ## 'unimodal_ftForceZ', \
                                                    'crossmodal_landmarkEEDist', \
                                                    'crossmodal_landmarkEEAng',\
                                                    ## 'unimodal_fabricForce',\
                                                    'unimodal_landmarkDist']
                                                    
        dm.getDataLOPO(subjects, opt.task, raw_data_path, save_data_path,
                       param_dict['data_param']['rf_center'], param_dict['data_param']['local_range'],\
                       downSampleSize=param_dict['data_param']['downSampleSize'], scale=scale, \
                       success_viz=success_viz, failure_viz=failure_viz,\
                       cut_data=param_dict['data_param']['cut_data'],\
                       save_pdf=opt.bSavePdf, solid_color=True,\
                       handFeatures=param_dict['data_param']['handFeatures'], data_renew=opt.bDataRenew, \
                       max_time=param_dict['data_param']['max_time'], target_class=target_class)

    elif opt.bLikelihoodPlot:
        save_data_path = os.path.expanduser('~')+\
          '/hrl_file_server/dpark_data/anomaly/AURO2016/'+opt.task+'_data_isolation4/'+\
          str(param_dict['data_param']['downSampleSize'])+'_'+str(opt.dim)
        param_dict['HMM']['scale'] = 8.0
        param_dict['data_param']['handFeatures'] = ['unimodal_ftForce_delta', 'unimodal_kinDesEEChange',\
                                                    'crossmodal_landmarkEEAng']
        #'crossmodal_landmarkEEDist'        

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
        param_dict['HMM']['scale'] = 7.0
        if opt.bNoUpdate: param_dict['ROC']['update_list'] = []        
        evaluation_all(subjects, opt.task, raw_data_path, save_data_path, param_dict, save_pdf=opt.bSavePdf, \
                       verbose=opt.bVerbose, debug=opt.bDebug, no_plot=opt.bNoPlot, \
                       find_param=False, data_gen=opt.bDataGen, target_class=target_class)

    elif opt.evaluation_single:
        '''
        evaluation with selected feature set
        '''
        save_data_path = os.path.expanduser('~')+\
          '/hrl_file_server/dpark_data/anomaly/AURO2016/'+opt.task+'_data_isolation6/'+\
          str(param_dict['data_param']['downSampleSize'])+'_'+str(opt.dim)
        ## save_data_path = os.path.expanduser('~')+\
        ##   '/hrl_file_server/dpark_data/anomaly/AURO2016/'+opt.task+'_data_isolation5/'+\
        ##   str(param_dict['data_param']['downSampleSize'])+'_'+str(opt.dim)
        ## save_data_path = os.path.expanduser('~')+\
        ##   '/hrl_file_server/dpark_data/anomaly/AURO2016/'+opt.task+'_data_isolation7/'+\
        ##   str(param_dict['data_param']['downSampleSize'])+'_'+str(opt.dim)


        ## # 51%
        ## param_dict['data_param']['handFeatures'] = ['unimodal_ftForce_zero', 'unimodal_kinDesEEChange']
        ## # 52%
        ## param_dict['data_param']['handFeatures'] = ['unimodal_ftForce_zero', 'crossmodal_landmarkEEAng']
        ## # 58%
        ## param_dict['data_param']['handFeatures'] = ['unimodal_ftForce_zero', 'crossmodal_landmarkEEDist',\
        ##                                             'crossmodal_landmarkEEAng']        
        ## # 57%
        ## param_dict['data_param']['handFeatures'] = ['unimodal_ftForce_zero', 'unimodal_kinDesEEChange',\
        ##                                             'crossmodal_landmarkEEAng']

        # 54%
        param_dict['data_param']['handFeatures'] = ['unimodal_ftForce_delta', 'unimodal_kinDesEEChange',\
                                                    'crossmodal_landmarkEEAng']
        ## 62
        param_dict['data_param']['handFeatures'] = ['unimodal_ftForce_delta', 'crossmodal_landmarkEEDist',\
                                                    'crossmodal_landmarkEEAng']        
        # 48%
        ## param_dict['data_param']['handFeatures'] = ['unimodal_ftForce_delta', 'unimodal_kinJntEff_3']


        param_dict['data_param']['handFeatures'] = ['unimodal_ftForce_delta', 'unimodal_kinJntEff_5']

        ## param_dict['data_param']['handFeatures'] = ['unimodal_ftForce_delta', 'unimodal_kinJntEff_5',\
        ##                                             'crossmodal_landmarkEEAng']
        
        ## param_dict['data_param']['handFeatures'] = ['unimodal_ftForce_delta', 'unimodal_kinJntEff_5',\
        ##                                             'crossmodal_landmarkEEAng', 'crossmodal_landmarkEEDist']


        #------------------------------------------------------------------------------------------------
        ## param_dict['data_param']['handFeatures'] = ['unimodal_ftForce_delta', 'unimodal_kinJntEff_3', \
        ##                                             'unimodal_audioWristRMS',\
        ##                                             'unimodal_landmarkDist', 'unimodal_kinJntEff_5']
                                                     
        ## # 56%
        ## param_dict['data_param']['handFeatures'] = ['unimodal_audioWristRMS', 'unimodal_ftForceZ', \
        ##                                             'crossmodal_landmarkEEDist', 'unimodal_kinJntEff_1']
        ## # 
        ## param_dict['data_param']['handFeatures'] = ['unimodal_audioWristRMS', 'unimodal_ftForceZ', \
        ##                                             'crossmodal_landmarkEEDist', 'unimodal_kinJntEff_1']
        
        param_dict['ROC']['methods'] = ['hmmgp']
        nPoints = param_dict['ROC']['nPoints']
        param_dict['ROC']['hmmgp_param_range'] = np.logspace(-1, 2.3, nPoints)*-1.0
        param_dict['HMM']['scale'] = 7.0
        if opt.bNoUpdate: param_dict['ROC']['update_list'] = []        
        evaluation_single_ad(subjects, opt.task, raw_data_path, save_data_path, param_dict, \
                             save_pdf=opt.bSavePdf, \
                             verbose=opt.bVerbose, debug=opt.bDebug, no_plot=opt.bNoPlot, \
                             find_param=False, data_gen=opt.bDataGen, target_class=target_class)


    elif opt.evaluation_double:

        save_data_path = os.path.expanduser('~')+\
          '/hrl_file_server/dpark_data/anomaly/AURO2016/'+opt.task+'_data_isolation5/'+\
          str(param_dict['data_param']['downSampleSize'])+'_'+str(opt.dim)

        param_dict['ROC']['methods'] = ['hmmgp']
        param_dict1 = copy.copy(param_dict)
        param_dict1['data_param']['handFeatures1'] = ['unimodal_landmarkDist', 'crossmodal_landmarkEEDist', \
                                                     'unimodal_kinJntEff_4', 'unimodal_kinDesEEChange']
        param_dict1['HMM']['scale'] = 14.0
                                                     
        param_dict2 = copy.copy(param_dict)
        param_dict2['data_param']['handFeatures2'] = ['unimodal_audioWristRMS', 'unimodal_ftForceZ', \
                                                     'crossmodal_landmarkEEDist', 'unimodal_kinJntEff_1']
        param_dict2['HMM']['scale'] = 6.11
        
        if opt.bNoUpdate: param_dict['ROC']['update_list'] = []        
        evaluation_double_ad(subjects, opt.task, raw_data_path, save_data_path, (param_dict1,param_dict2), \
                             save_pdf=opt.bSavePdf, \
                             verbose=opt.bVerbose, debug=opt.bDebug, no_plot=opt.bNoPlot, \
                             find_param=False, data_gen=opt.bDataGen)

