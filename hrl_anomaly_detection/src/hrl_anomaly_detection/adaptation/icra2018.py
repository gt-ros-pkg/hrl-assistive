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
from hrl_anomaly_detection.adaptation import adt_utils as adutil


# Private learners
from hrl_anomaly_detection.hmm import learning_hmm as hmm
import hrl_anomaly_detection.classifiers.classifier as cf
import hrl_anomaly_detection.data_viz as dv

# visualization
import matplotlib
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



def evaluation_single_ad(subject_names, task_name, raw_data_path, processed_data_path, param_dict,\
                         data_renew=False, save_pdf=False, verbose=False, debug=False,\
                         no_plot=False, delay_plot=True, find_param=False, \
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

    # Adaptation
    ADT_dict = param_dict['ADT']

    # parameters
    startIdx    = 4
    method_list = ROC_dict['methods'] 
    nPoints     = ROC_dict['nPoints']
    
    #------------------------------------------

    if os.path.isdir(processed_data_path) is False:
        os.system('mkdir -p '+processed_data_path)

    crossVal_pkl = os.path.join(processed_data_path, 'cv_'+task_name+'.pkl')
    
    if os.path.isfile(crossVal_pkl) and data_renew is False:
        print "CV data exists and no renew"
        d = ut.load_pickle(crossVal_pkl)
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

        d['successData'], d['failureData'], d['success_files'], d['failure_files'], d['kFoldList'] \
          = dm.LOPO_data_index(d['successDataList'], d['failureDataList'],\
                               d['successFileList'], d['failureFileList'])

        ut.save_pickle(d, crossVal_pkl)


    # select feature for detection
    feature_list = []
    for feature in param_dict['data_param']['handFeatures']:
        idx = [ i for i, x in enumerate(param_dict['data_param']['isolationFeatures']) if feature == x][0]
        feature_list.append(idx)
    print np.shape(d['successData'])

    d['successData']    = d['successData'][feature_list]
    d['failureData']    = d['failureData'][feature_list]

    #-----------------------------------------------------------------------------------------    
    tgt_raw_data_path  = os.path.expanduser('~')+'/hrl_file_server/dpark_data/anomaly/RAW_DATA/CORL2017/'
    tgt_subjects = ['Andrew', 'Britteney', 'Joshua', 'Jun', 'Kihan', 'Lichard', 'Shingshing', 'Sid', 'Tao']

    crossVal_pkl = os.path.join(processed_data_path, 'cv_td_'+task_name+'.pkl')
    if os.path.isfile(crossVal_pkl) and data_renew is False and ADT_dict['data_renew'] is False:
        print "CV data exists and no renew"
        td = ut.load_pickle(crossVal_pkl)
    else:
        d['param_dict']['feature_names'] = np.array(d['param_dict']['feature_names'])[feature_list].tolist()
        d['param_dict']['feature_min'] = np.array(d['param_dict']['feature_min'])[feature_list].tolist()
        d['param_dict']['feature_max'] = np.array(d['param_dict']['feature_max'])[feature_list].tolist()
        
        # Extract data from designated location
        td = dm.getDataLOPO(tgt_subjects, task_name, tgt_raw_data_path, save_data_path,\
                            downSampleSize=data_dict['downSampleSize'],\
                            init_param_dict=d['param_dict'],\
                            handFeatures=param_dict['data_param']['handFeatures'], \
                            data_renew=ADT_dict['data_renew'], max_time=data_dict['max_time'],
                            pkl_prefix='tgt_', depth=True)

        ut.save_pickle(td, crossVal_pkl)

    #-----------------------------------------------------------------------------------------    
    # Training HMM, and getting classifier training and testing data
    noise_mag = 0.03
    dm.saveHMMinducedFeatures(d['kFoldList'], d['successData'], d['failureData'],\
                              task_name, processed_data_path,\
                              HMM_dict, data_renew, startIdx, nState, cov, \
                              success_files=d['success_files'], failure_files=d['failure_files'],\
                              noise_mag=noise_mag, diag=False, cov_type='full', \
                              inc_hmm_param=True, verbose=verbose)

    pkl_prefix = 'hmm_'+task_name
    for method in method_list:
        if method.find('ipca')>=0 or method.find('mlp')>=0:            
            adutil.saveWindowFeatures(d, processed_data_path, pkl_prefix, \
                                      win_size=SVM_dict['raw_window_size'], \
                                      WIN_renew=ADT_dict['data_renew'])
            break

    #-------------------------------------------------------------------------------------
    if HMM_dict['renew'] or SVM_dict['renew'] or ADT_dict['data_renew']: ADT_dict['HMM_renew'] = True

    # old, adapt, or renew hmm
    pkl_prefix = 'hmm_'+ADT_dict['HMM']+'_'+task_name
    ret = adutil.saveAHMMFeatures(d, td, task_name, processed_data_path, HMM_dict, ADT_dict,
                                  noise_mag, pkl_prefix)
    if ret is None: return ret

    # do we need for HMM?
    for method in method_list:
        if method.find('ipca')>=0 or method.find('mlp')>=0:            
            ret = adutil.saveWindowFeaturesForADP(td, processed_data_path, ADT_dict, pkl_prefix,
                                                  win_size=SVM_dict['raw_window_size'])
            break
    if ret is None: return ret


    ## # Comparison of
    ## from hrl_anomaly_detection import data_viz as dv
    ## import hmm_viz as hv   
    ## hv.data_viz(successData, td['successDataList'][0], raw_viz=True)
    ## hv.data_viz(successData, td['successDataList'][0], raw_viz=True,
    ##             minmax=(d['param_dict']['feature_min'], d['param_dict']['feature_max'] ))
    ## dv.viz( successData, minmax=(d['param_dict']['feature_min'], d['param_dict']['feature_max'] ) )
    ## dv.viz( td['successDataList'][0], minmax=(d['param_dict']['feature_min'], d['param_dict']['feature_max']))

    #-----------------------------------------------------------------------------------------
    roc_pkl = os.path.join(processed_data_path, 'roc_update_'+task_name+\
                           '_npTrain_'+str(ADT_dict['n_pTrain'])+\
                           '_nrSteps_'+str(ADT_dict['nrSteps'])+\
                           '_lr_'+str(ADT_dict['lr'])+'.pkl')
    if os.path.isfile(roc_pkl) is False or HMM_dict['renew'] or SVM_dict['renew'] \
      or ADT_dict['HMM_renew'] or ADT_dict['CLF_renew']:
        ROC_data = {}
    else:
        ROC_data = ut.load_pickle(roc_pkl)
    ROC_data = util.reset_roc_data(ROC_data, method_list, ROC_dict['update_list'], nPoints)

    if ADT_dict['CLF'] == 'adapt': adapt=True
    else: adapt=False

    # parallelization
    if debug: n_jobs=1
    else: n_jobs=-1
    l_data = Parallel(n_jobs=n_jobs, verbose=10)(delayed(cf.run_classifiers)( idx, processed_data_path, \
                                                                         task_name, \
                                                                         method, ROC_data, \
                                                                         ROC_dict, \
                                                                         SVM_dict, HMM_dict, \
                                                                         startIdx=startIdx, nState=nState,\
                                                                         n_jobs=n_jobs,\
                                                                         modeling_pkl_prefix=pkl_prefix,\
                                                                         adaptation=adapt) \
                                                                         for idx in xrange(len(td['successDataList']))
                                                                         for method in method_list)

    print "finished to run run_classifiers"
    ROC_data = util.update_roc_data(ROC_data, l_data, nPoints, method_list)
    ut.save_pickle(ROC_data, roc_pkl)
    d = roc_info(ROC_data, nPoints, no_plot=no_plot, ROC_dict=ROC_dict)

    from sklearn import metrics 
    for method in method_list:
        auc_raw_list=[]
        for i in xrange(len(l_data)):            
            tp_ll = l_data[i][method]['tp_l']
            fp_ll = l_data[i][method]['fp_l']
            tn_ll = l_data[i][method]['tn_l']
            fn_ll = l_data[i][method]['fn_l']
            tpr_l = []
            fpr_l = []
            for j in xrange(nPoints):
                tpr_l.append( float(np.sum(tp_ll[j]))/float(np.sum(tp_ll[j])+np.sum(fn_ll[j]))*100.0 )
                fpr_l.append( float(np.sum(fp_ll[j]))/float(np.sum(fp_ll[j])+np.sum(tn_ll[j]))*100.0 )

            auc = metrics.auc(fpr_l, tpr_l, True)
            auc_raw_list.append(auc)

        d[method+'_auc_raw'] = auc_raw_list
    return d
    ## class_info(method_list, ROC_data, nPoints, kFold_list)





if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    util.initialiseOptParser(p)

    p.add_option('--gen_likelihood', '--gl', action='store_true', dest='gen_likelihoods',
                 default=False, help='Generate likelihoods.')
    p.add_option('--eval_standard', '--esd', action='store_true', dest='evaluation_std',
                 default=False, help='Evaluate a standard single detector.')
    p.add_option('--eval_single', '--es', action='store_true', dest='evaluation_single',
                 default=False, help='Evaluate an adaptive single detector.')
         
    opt, args = p.parse_args()

    #---------------------------------------------------------------------------           
    # Run evaluation
    #---------------------------------------------------------------------------           
    rf_center     = 'kinEEPos'        
    scale         = 1.0
    local_range   = 10.0
    nPoints = 40 #None

    from hrl_anomaly_detection.adaptation.ICRA2018_params import *
    raw_data_path, save_data_path, param_dict = getParams(opt.task, opt.bDataRenew, \
                                                          opt.bHMMRenew, opt.bCLFRenew, opt.dim,\
                                                          rf_center, local_range, nPoints=nPoints)
    if opt.bNoUpdate: param_dict['ROC']['update_list'] = []

    subjects = ['s2', 's3','s4','s5', 's6','s7','s8', 's9']        
    save_data_path = os.path.expanduser('~')+\
      '/hrl_file_server/dpark_data/anomaly/TCDS2017/'+opt.task+'_data_adaptation3/'

    #---------------------------------------------------------------------------           
    if opt.bRawDataPlot or opt.bInterpDataPlot:
        '''
        Before localization: Raw data plot
        After localization: Raw or interpolated data plot
        '''
        raw_data_path  = os.path.expanduser('~')+'/hrl_file_server/dpark_data/anomaly/RAW_DATA/ICRA2017/'    
        subjects = ['jina']
        
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

        raw_data_path  = os.path.expanduser('~')+'/hrl_file_server/dpark_data/anomaly/RAW_DATA/ICRA2017/'    
        subjects = ['jina']
        
        ## br
        save_data_path = os.path.expanduser('~')+\
          '/hrl_file_server/dpark_data/anomaly/TCDS2017/'+opt.task+'_data3_adaptation/'
          
        success_viz = True
        failure_viz = False
        param_dict['data_param']['handFeatures'] = ['unimodal_audioWristRMS', \
                                                    ## 'unimodal_audioWristFrontRMS', \
                                                    ## 'unimodal_audioWristAzimuth',\
                                                    'unimodal_kinVel',\
                                                    'unimodal_kinJntEff_1', \
                                                    ## 'unimodal_kinJntEff', \
                                                    'unimodal_ftForce_integ', \
                                                    ## 'unimodal_ftForce_delta', \
                                                    'unimodal_ftForce_zero', \
                                                    ## 'unimodal_ftForce', \
                                                    ## 'unimodal_ftForceX', \
                                                    ## 'unimodal_ftForceY', \
                                                    ## 'unimodal_ftForceZ', \
                                                    'unimodal_kinEEChange', \
                                                    'unimodal_kinDesEEChange', \
                                                    'crossmodal_landmarkEEDist', \
                                                    ## 'crossmodal_landmarkEEAng',\
                                                    ## 'unimodal_fabricForce',\
                                                    ## 'unimodal_landmarkDist'
                                                    ]
        ## target_class = [13]
        ## param_dict['data_param']['handFeatures'] = ['unimodal_kinJntEff_1']
        
        dm.getDataLOPO(subjects, opt.task, raw_data_path, save_data_path,
                       param_dict['data_param']['rf_center'], param_dict['data_param']['local_range'],\
                       downSampleSize=param_dict['data_param']['downSampleSize'], \
                       success_viz=success_viz, failure_viz=failure_viz,\
                       cut_data=param_dict['data_param']['cut_data'],\
                       save_pdf=opt.bSavePdf, solid_color=True,\
                       handFeatures=param_dict['data_param']['handFeatures'], data_renew=opt.bDataRenew, \
                       max_time=param_dict['data_param']['max_time']) #, target_class=target_class)

    elif opt.bLikelihoodPlot:
        param_dict['HMM']['scale'] = 5.0
        param_dict['data_param']['handFeatures'] = ['unimodal_audioWristRMS',  \
                                                    'unimodal_kinJntEff_1',\
                                                    'unimodal_ftForce_integ',\
                                                    'unimodal_kinEEChange', \
                                                    'crossmodal_landmarkEEDist', \
                                                    ]
        param_dict['ROC']['hmmgp_param_range'] = np.logspace(-0.6, 2.3, nPoints)*-1.0

        import hrl_anomaly_detection.data_viz as dv        
        dv.vizLikelihoods(subjects, opt.task, raw_data_path, save_data_path, param_dict,\
                          decision_boundary_viz=False, method='progress', \
                          useTrain=True, useNormalTest=False, useAbnormalTest=True,\
                          useTrain_color=False, useNormalTest_color=False, useAbnormalTest_color=False,\
                          hmm_renew=opt.bHMMRenew, data_renew=opt.bDataRenew, save_pdf=opt.bSavePdf,\
                          verbose=opt.bVerbose, lopo=True, plot_feature=False)


    elif opt.gen_likelihoods:
        ## ep 12-89.5 7-82
        save_data_path = os.path.expanduser('~')+\
          '/hrl_file_server/dpark_data/anomaly/TCDS2017/'+opt.task+'_data_adaptation/'+\
          str(param_dict['data_param']['downSampleSize'])+'_'+str(opt.dim)
        param_dict['data_param']['handFeatures'] = ['unimodal_audioWristRMS',  \
                                                     'unimodal_kinJntEff_1',\
                                                     'unimodal_ftForce_integ',\
                                                     'crossmodal_landmarkEEDist']
        param_dict['HMM']['scale'] = 5.0
        
        gen_likelihoods(subjects, opt.task, raw_data_path, save_data_path, param_dict, \
                        save_pdf=opt.bSavePdf, verbose=opt.bVerbose )


    elif opt.evaluation_std:
        '''
        evaluation with selected feature set 5,6
        '''
        nPoints = param_dict['ROC']['nPoints'] = 40
        param_dict['data_param']['handFeatures'] = ['unimodal_audioWristRMS',  \
                                                     'unimodal_kinJntEff_1',\
                                                     'unimodal_ftForce_integ',\
                                                     'crossmodal_landmarkEEDist']
        param_dict['HMM']['scale'] = 5.0
        param_dict['ROC']['progress_param_range'] = -np.logspace(-1.2, 2.4, nPoints)+1.0
        param_dict['ROC']['ipca_param_range'] = np.logspace(0, 1.0, nPoints)-1
        param_dict['ROC']['mlp_param_range'] = np.logspace(-0.8, 0.0, nPoints)
        param_dict['ROC']['methods'] = ['mlp'] #['ipca'] #['progress']

        if opt.bNoUpdate: param_dict['ROC']['update_list'] = []

        ret = evaluation_std(subjects, opt.task, raw_data_path, save_data_path, param_dict, \
                            save_pdf=opt.bSavePdf, \
                            verbose=opt.bVerbose, debug=opt.bDebug, no_plot=opt.bNoPlot, \
                            find_param=False)




    elif opt.evaluation_single:
        '''
        evaluation with selected feature set 5,6
        '''
        nPoints = param_dict['ROC']['nPoints'] = 40
        param_dict['data_param']['handFeatures'] = ['unimodal_audioWristRMS',  \
                                                     'unimodal_kinJntEff_1',\
                                                     'unimodal_ftForce_integ',\
                                                     'crossmodal_landmarkEEDist']
        param_dict['HMM']['scale'] = 5.0
        param_dict['ROC']['progress_param_range'] = -np.logspace(-1.2, 2.4, nPoints)+1.0
        param_dict['ROC']['ipca_param_range'] = np.logspace(0, 2.5, nPoints)-1
        param_dict['ROC']['mlp_param_range'] = np.logspace(-0.8, 0.2, nPoints)-0.2
        param_dict['ROC']['methods'] = ['progress'] #['mlp'] #['ipca'] #['progress']

        if opt.bNoUpdate: param_dict['ROC']['update_list'] = []

        param_dict['ADT'] = {}
        param_dict['ADT']['data_renew'] = False


        auc_complete = []
        auc_list = []
        auc_raw_list = []
        for method in param_dict['ROC']['methods']:
            auc_complete.append([])
            auc_list.append([])
            auc_raw_list.append([])

            
        for nrSteps in [20]:
            #for hmm in ['adapt']: #'old', 'renew'
                #for lr in [0.05, 0.1, 0.6, 0.8]:
            for n_pTrain in [2,4,8,10]:
                for clf in ['adapt']: #'old', 'renew'
                    param_dict['ADT']['lr']       = 0.2 #lr 
                    param_dict['ADT']['max_iter'] = 1
                    param_dict['ADT']['n_pTrain'] = 10 #n_pTrain
                    param_dict['ADT']['nrSteps']  = nrSteps
                    param_dict['ADT']['HMM']      = 'adapt' #hmm #'old'
                    param_dict['ADT']['CLF']      = clf
                    param_dict['ADT']['HMM_renew'] = True
                    param_dict['ADT']['WIN_renew'] = True
                    param_dict['ADT']['CLF_renew'] = True

                    ret = evaluation_single_ad(subjects, opt.task, raw_data_path, save_data_path, param_dict, \
                                               save_pdf=opt.bSavePdf, \
                                               verbose=opt.bVerbose, debug=opt.bDebug, no_plot=opt.bNoPlot, \
                                               find_param=False)

                    for i, method in enumerate(param_dict['ROC']['methods']):

                        if ret is None:
                            auc_list[i].append(None)
                            auc_raw_list[i].append(None)
                            auc_complete[i].append(None)                        
                        else:
                            auc_list[i].append(ret[method])
                            auc_raw_list[i].append(ret[method+'_auc_raw'])
                            auc_complete[i].append(ret[method+'_complete'])

        print "-------------------------------"
        for i, method in enumerate(param_dict['ROC']['methods']):
            print auc_complete[i]
            print auc_raw_list[i]
            print auc_list[i]


