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
from hrl_anomaly_detection.params import *

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

        if AE_dict['add_option'] is not None:
            newHandSuccessData = handSuccessData = d['handNormTrainData']
            newHandFailureData = handFailureData = d['handAbnormTrainData']
            
            ## for i in xrange(AE_dict['nAugment']):
            ##     newHandSuccessData = stackSample(newHandSuccessData, handSuccessData)
            ##     newHandFailureData = stackSample(newHandFailureData, handFailureData)

            successData = combineData( successData, newHandSuccessData, \
                                       AE_dict['add_option'], d['handFeatureNames'] )
            failureData = combineData( failureData, newHandFailureData, \
                                       AE_dict['add_option'], d['handFeatureNames'] )

            ## # reduce dimension by pooling
            ## pooling_param_dict  = {'dim': AE_dict['filterDim']} # only for AE        
            ## successData, pooling_param_dict = dm.variancePooling(successData, \
            ##                                                   pooling_param_dict)
            ## failureData, _ = dm.variancePooling(failureData, pooling_param_dict)
            
            
        successData *= HMM_dict['scale']
        failureData *= HMM_dict['scale']
        
    else:
        dd = dm.getDataSet(subject_names, task_name, raw_data_path, \
                           processed_data_path, data_dict['rf_center'], \
                           data_dict['local_range'],\
                           downSampleSize=data_dict['downSampleSize'], \
                           scale=1.0,\
                           ae_data=False,\
                           handFeatures=data_dict['handFeatures'], \
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

    kFold_list = dm.kFold_data_index2(len(successData[0]),\
                                      len(failureData[0]),\
                                      data_dict['nNormalFold'], data_dict['nAbnormalFold'] )
    normalTrainIdx, abnormalTrainIdx, normalTestIdx, abnormalTestIdx = kFold_list[0]
    normalTrainData   = successData[:, normalTrainIdx, :] 
    abnormalTrainData = failureData[:, abnormalTrainIdx, :] 
    normalTestData    = successData[:, normalTestIdx, :] 
    abnormalTestData  = failureData[:, abnormalTestIdx, :] 
    
    # training hmm
    nEmissionDim = len(normalTrainData)
    ## hmm_param_pkl = os.path.join(processed_data_path, 'hmm_'+task_name+'.pkl')    
    cov_mult = [cov]*(nEmissionDim**2)

    # generative model
    ml  = hmm.learning_hmm(nState, nEmissionDim, verbose=verbose)
    if data_dict['handFeatures_noise']:
        ret = ml.fit(normalTrainData+\
                     np.random.normal(0.0, 0.03, np.shape(normalTrainData) )*HMM_dict['scale'], \
                     cov_mult=cov_mult, ml_pkl=None, use_pkl=False) # not(renew))
    else:
        ret = ml.fit(normalTrainData, cov_mult=cov_mult, ml_pkl=None, use_pkl=False) # not(renew))
        
    ## ths = threshold
    startIdx = 4
        
    if ret == 'Failure': 
        print "-------------------------"
        print "HMM returned failure!!   "
        print "-------------------------"
        return (-1,-1,-1,-1)

    if decision_boundary_viz:
        ## testDataX = np.vstack([np.swapaxes(normalTrainData, 0, 1), np.swapaxes(abnormalTrainData, 0, 1)])
        ## testDataX = np.swapaxes(testDataX, 0, 1)
        ## testDataY = np.hstack([ -np.ones(len(normalTrainData[0])), \
        ##                         np.ones(len(abnormalTrainData[0])) ])
        testDataX = normalTrainData
        testDataY = -np.ones(len(normalTrainData[0]))
                                

        r = Parallel(n_jobs=-1)(delayed(hmm.computeLikelihoods)(i, ml.A, ml.B, ml.pi, ml.F, \
                                                                [testDataX[j][i] for j in \
                                                                 xrange(nEmissionDim)], \
                                                                ml.nEmissionDim, ml.nState,\
                                                                startIdx=startIdx, \
                                                                bPosterior=True)
                                                                for i in xrange(len(testDataX[0])))
        _, ll_classifier_train_idx, ll_logp, ll_post = zip(*r)


        ## if True:
        ##     from hrl_anomaly_detection.hmm import learning_util as hmm_util                        
        ##     ll_classifier_test_X, ll_classifier_test_Y = \
        ##       hmm.getHMMinducedFeatures(ll_logp, ll_post, c=1.0)

        ##     fig = plt.figure()
        ##     ax1 = fig.add_subplot(211)
        ##     plt.plot(np.swapaxes( np.array(ll_classifier_test_X)[:,:,0], 0,1) )
        ##     ax1 = fig.add_subplot(212)
        ##     plt.plot(np.swapaxes( np.array(ll_classifier_test_X)[:,:,1], 0,1) )
        ##     plt.show()
        ##     sys.exit()
              
        ##     ll_delta_logp = []
        ##     ll_delta_post = []
        ##     ll_delta_logp_post = []
        ##     ll_delta_logp_post2 = []
        ##     for i in xrange(len(ll_post)):
        ##         l_delta_logp = []
        ##         l_delta_post = []
        ##         l_delta_logp_post = []
        ##         for j in xrange(len(ll_post[i])-1):
        ##             l_delta_logp.append( ll_logp[i][j+1] - ll_logp[i][j] )
        ##             l_delta_post.append( hmm_util.symmetric_entropy(ll_post[i][j], ll_post[i][j+1]) )
        ##         ll_delta_logp.append( l_delta_logp )
        ##         ll_delta_post.append( l_delta_post )
        ##         ll_delta_logp_post.append( np.array(l_delta_logp)/(np.array(l_delta_post)+0.1) )
        ##         ll_delta_logp_post2.append( np.array(l_delta_logp)/(np.array(l_delta_post)+1.0) )


        ##     fig = plt.figure()
        ##     ax1 = fig.add_subplot(411)            
        ##     plt.plot(np.swapaxes(ll_delta_logp,0,1))
        ##     ax1 = fig.add_subplot(412)            
        ##     plt.plot(np.swapaxes(ll_delta_post,0,1))
        ##     ax1 = fig.add_subplot(413)            
        ##     plt.plot(np.swapaxes(ll_delta_logp_post,0,1))
        ##     ax1 = fig.add_subplot(414)            
        ##     plt.plot(np.swapaxes(ll_delta_logp_post2,0,1))
        ##     ## plt.plot(np.swapaxes(ll_delta_post,0,1))
        ##     ## plt.plot( np.swapaxes( np.array(ll_delta_logp)/np.array(ll_delta_post), 0,1) )
        ##     plt.show()
        ##     sys.exit()


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
        X_train_org, Y_train_org, idx_train_org = flattenSample(ll_classifier_train_X, \
                                                                ll_classifier_train_Y, \
                                                                ll_classifier_train_idx)
        
        # discriminative classifier
        dtc = cf.classifier( method='progress_time_cluster', nPosteriors=nState, \
                             nLength=len(normalTestData[0,0]), ths_mult=-0.0 )
        dtc.fit(X_train_org, Y_train_org, idx_train_org, parallel=True)





    print "----------------------------------------------------------------------------"
    fig = plt.figure()
    min_logp = 0.0
    max_logp = 0.0
    target_idx = 1

    # training data
    if useTrain and False:

        log_ll = []
        exp_log_ll = []        
        for i in xrange(len(normalTrainData[0])):

            log_ll.append([])
            exp_log_ll.append([])
            for j in range(startIdx, len(normalTrainData[0][i])):

                X = [x[i,:j] for x in normalTrainData]
                logp = ml.loglikelihood(X)
                log_ll[i].append(logp)

                if decision_boundary_viz and i==target_idx:
                    if j>=len(ll_logp[i]): continue
                    l_X = [ll_logp[i][j]] + ll_post[i][j].tolist()

                    exp_logp = dtc.predict(l_X)[0] + ll_logp[i][j]
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
            
        ## plt.plot(log_ll[target_idx], 'k-', lw=3.0)
        if decision_boundary_viz:
            plt.plot(exp_log_ll[target_idx], 'm-', lw=3.0)            

            
    # normal test data
    if useNormalTest:

        log_ll = []
        ## exp_log_ll = []        
        for i in xrange(len(normalTestData[0])):

            log_ll.append([])
            ## exp_log_ll.append([])
            for j in range(startIdx, len(normalTestData[0][i])):
                X = [x[i,:j] for x in normalTestData] # by dim
                logp = ml.loglikelihood(X)
                log_ll[i].append(logp)

                ## exp_logp, logp = ml.expLoglikelihood(X, ths, bLoglikelihood=True)
                ## log_ll[i].append(logp)
                ## exp_log_ll[i].append(exp_logp)

            if min_logp > np.amin(log_ll): min_logp = np.amin(log_ll)
            if max_logp < np.amax(log_ll): max_logp = np.amax(log_ll)

            # disp 
            if useNormalTest_color: plt.plot(log_ll[i], label=str(i))
            else: plt.plot(log_ll[i], 'b-')

            ## plt.plot(exp_log_ll[i], 'r*-')

        if useNormalTest_color: 
            plt.legend(loc=3,prop={'size':16})

    # abnormal test data
    if useAbnormalTest:
        log_ll = []
        exp_log_ll = []        
        for i in xrange(len(abnormalTestData[0])):

            log_ll.append([])
            exp_log_ll.append([])

            for j in range(startIdx, len(abnormalTestData[0][i])):
                X = [x[i,:j] for x in abnormalTestData]                
                try:
                    logp = ml.loglikelihood(X)
                except:
                    print "Too different input profile that cannot be expressed by emission matrix"
                    return [], 0.0 # error

                log_ll[i].append(logp)

                if decision_boundary_viz and i==target_idx:
                    if j>=len(ll_logp[i]): continue
                    l_X = [ll_logp[i][j]] + ll_post[i][j].tolist()
                    exp_logp = dtc.predict(l_X)[0] + ll_logp[i][j]
                    exp_log_ll[i].append(exp_logp)


            # disp
            plt.plot(log_ll[i], 'r-')
            plt.plot(exp_log_ll[i], 'r*-')
        plt.plot(log_ll[target_idx], 'k-', lw=3.0)            


    plt.ylim([min_logp, max_logp])
    if save_pdf == True:
        fig.savefig('test.pdf')
        fig.savefig('test.png')
        os.system('cp test.p* ~/Dropbox/HRL/')
    else:
        plt.show()        

    return


    
def aeDataExtraction(subject_names, task_name, raw_data_path, \
                    processed_data_path, param_dict,\
                    handFeature_viz=False, success_viz=False, failure_viz=False,\
                    verbose=False):

    ## Parameters
    # data
    data_dict  = param_dict['data_param']
    data_renew = data_dict['renew']
    handFeatures = data_dict['handFeatures']
    # AE
    AE_dict     = param_dict['AE']
    rawFeatures = AE_dict['rawFeatures']
    #------------------------------------------
    assert AE_dict['switch'] == True
                   
    crossVal_pkl = os.path.join(processed_data_path, 'cv_'+task_name+'.pkl')
    if os.path.isfile(crossVal_pkl) and data_renew is False: 
        print "Loading cv data"
        d = ut.load_pickle(crossVal_pkl)
        if 'aeSuccessData' not in d.keys():
            print "Reload data!!"
            sys.exit()
    else:
        d = dm.getDataSet(subject_names, task_name, raw_data_path, processed_data_path, \
                           data_dict['rf_center'], data_dict['local_range'],\
                           downSampleSize=data_dict['downSampleSize'], scale=1.0,\
                           ae_data=AE_dict['switch'], \
                           handFeatures=handFeatures, rawFeatures=rawFeatures,\
                           cut_data=data_dict['cut_data'],
                           data_renew=data_renew)

        if os.path.isfile(crossVal_pkl):
            dd = ut.load_pickle(crossVal_pkl)
            d['kFoldList'] = dd['kFoldList'] 
        else:
            kFold_list = dm.kFold_data_index2(len(d['aeSuccessData'][0]),\
                                              len(d['aeFailureData'][0]),\
                                              data_dict['nNormalFold'], data_dict['nAbnormalFold'] )

            d['kFoldList']       = kFold_list                                             
        ut.save_pickle(d, crossVal_pkl)

    # Training HMM, and getting classifier training and testing data
    for idx, (normalTrainIdx, abnormalTrainIdx, normalTestIdx, abnormalTestIdx) \
      in enumerate( d['kFoldList'] ):

        if verbose: print "Start "+str(idx)+"/"+str(len( d['kFoldList'] ))+"th iteration"

        if AE_dict['method'] == 'ae':
            AE_proc_data = os.path.join(processed_data_path, 'ae_processed_data_'+str(idx)+'.pkl')
        else:
            AE_proc_data = os.path.join(processed_data_path, 'pca_processed_data_'+str(idx)+'.pkl')
            
        # From dim x sample x length
        # To reduced_dim x sample
        dd = dm.getAEdataSet(idx, d['aeSuccessData'], d['aeFailureData'], \
                             d['successData'], d['failureData'], d['param_dict'], \
                             normalTrainIdx, abnormalTrainIdx, normalTestIdx, abnormalTestIdx,
                             AE_dict['time_window'], AE_dict['nAugment'], \
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
                             method=AE_dict['method'],\
                             # PCA param
                             pca_gamma=AE_dict['pca_gamma'],\
                             verbose=verbose, renew=AE_dict['renew'], \
                             preTrainModel=AE_dict['preTrainModel'])

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

            
        if success_viz or failure_viz and False:
            import data_viz as dv
            print dd.keys()
            dv.viz(dd['normTrainData'], normTest=dd['normTestData'], \
                   abnormTest=dd['abnormTestData'],skip=True)
            if AE_dict['filter']:
                dv.viz(dd['normTrainDataFiltered'], abnormTest=dd['abnormTrainDataFiltered'])
            else: dv.viz(dd['normTrainData'], dd['abnormTrainData'])

        if handFeature_viz:
            print AE_dict['add_option'], dd['handFeatureNames']
            handNormalTrainData = combineData( normalTrainData, dd['handNormTrainData'],\
                                               AE_dict['add_option'], dd['handFeatureNames'])
            handAbnormalTrainData = combineData( abnormalTrainData, dd['handAbnormTrainData'],\
                                                 AE_dict['add_option'], dd['handFeatureNames'])

            
            import data_viz as dv
            dv.viz(handNormalTrainData, abnormTest=handAbnormalTrainData)

            ## normalTrainData   = stackSample(normalTrainData, handNormalTrainData)
            ## abnormalTrainData = stackSample(abnormalTrainData, handAbnormalTrainData)



# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------

def evaluation_all(subject_names, task_name, raw_data_path, processed_data_path, param_dict,\
                   data_renew=False, save_pdf=False, verbose=False, debug=False,\
                   no_plot=False, delay_plot=True):

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
        d = ut.load_pickle(crossVal_pkl)
        kFold_list  = d['kFoldList']
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
                           
        if AE_dict['switch']:
            # Task-oriented raw features        
            kFold_list = dm.kFold_data_index2(len(d['aeSuccessData'][0]), len(d['aeFailureData'][0]), \
                                              data_dict['nNormalFold'], data_dict['nAbnormalFold'] )
        else:
            # Task-oriented hand-crafted features        
            kFold_list = dm.kFold_data_index2(len(d['successData'][0]), len(d['failureData'][0]), \
                                              data_dict['nNormalFold'], data_dict['nAbnormalFold'] )
        d['kFoldList']   = kFold_list
        ut.save_pickle(d, crossVal_pkl)

    #-----------------------------------------------------------------------------------------
    # parameters
    startIdx    = 4
    method_list = ROC_dict['methods'] 
    nPoints     = ROC_dict['nPoints']

    successData = d['successData']
    failureData = d['failureData']
    param_dict  = d['param_dict']
    aeSuccessData = d.get('aeSuccessData', None)
    aeFailureData = d.get('aeFailureData', None)
    if 'timeList' in param_dict.keys():
        timeList    = param_dict['timeList'][startIdx:]
    else: timeList = None

    #-----------------------------------------------------------------------------------------
    # Training HMM, and getting classifier training and testing data
    for idx, (normalTrainIdx, abnormalTrainIdx, normalTestIdx, abnormalTestIdx) \
      in enumerate(kFold_list):

        if verbose: print idx, " : training hmm and getting classifier training and testing data"
            

        if AE_dict['switch'] and AE_dict['add_option'] is not None:
            tag = ''
            for ft in AE_dict['add_option']:
                tag += ft[:2]
            modeling_pkl = os.path.join(processed_data_path, 'hmm_'+task_name+'_raw_'+tag+'_'+str(idx)+'.pkl')
        elif AE_dict['switch'] and AE_dict['add_option'] is None:
            modeling_pkl = os.path.join(processed_data_path, 'hmm_'+task_name+'_raw_'+str(idx)+'.pkl')
        else:
            modeling_pkl = os.path.join(processed_data_path, 'hmm_'+task_name+'_'+str(idx)+'.pkl')

        if not (os.path.isfile(modeling_pkl) is False or HMM_dict['renew'] or data_renew): continue

        if AE_dict['switch']:
            if verbose: print "Start "+str(idx)+"/"+str(len(kFold_list))+"th iteration"

            AE_proc_data = os.path.join(processed_data_path, 'ae_processed_data_'+str(idx)+'.pkl')

            # From dim x sample x length
            # To reduced_dim x sample x length
            d = dm.getAEdataSet(idx, aeSuccessData, aeFailureData, \
                                successData, failureData, param_dict,\
                                normalTrainIdx, abnormalTrainIdx, normalTestIdx, abnormalTestIdx,\
                                AE_dict['time_window'], AE_dict['nAugment'], \
                                AE_proc_data, \
                                # data param
                                processed_data_path, \
                                # AE param
                                layer_sizes=AE_dict['layer_sizes'], learning_rate=AE_dict['learning_rate'], \
                                learning_rate_decay=AE_dict['learning_rate_decay'], \
                                momentum=AE_dict['momentum'], dampening=AE_dict['dampening'], \
                                lambda_reg=AE_dict['lambda_reg'], \
                                max_iteration=AE_dict['max_iteration'], min_loss=AE_dict['min_loss'], \
                                cuda=False, \
                                filtering=AE_dict['filter'], filteringDim=AE_dict['filterDim'],\
                                verbose=False)

            if AE_dict['filter']:
                # NOTE: pooling dimension should vary on each auto encoder.
                # Filtering using variances
                normalTrainData   = d['normTrainDataFiltered']
                abnormalTrainData = d['abnormTrainDataFiltered']
                normalTestData    = d['normTestDataFiltered']
                abnormalTestData  = d['abnormTestDataFiltered']
            else:
                normalTrainData   = d['normTrainData']
                abnormalTrainData = d['abnormTrainData']
                normalTestData    = d['normTestData']
                abnormalTestData  = d['abnormTestData']
        else:
            # dim x sample x length
            normalTrainData   = successData[:, normalTrainIdx, :] 
            abnormalTrainData = failureData[:, abnormalTrainIdx, :] 
            normalTestData    = successData[:, normalTestIdx, :] 
            abnormalTestData  = failureData[:, abnormalTestIdx, :] 


        if AE_dict['switch'] and AE_dict['add_option'] is not None:
            print "add hand-crafted features.."
            newHandSuccTrData = handSuccTrData = d['handNormTrainData']
            newHandFailTrData = handFailTrData = d['handAbnormTrainData']
            handSuccTeData = d['handNormTestData']
            handFailTeData = d['handAbnormTestData']

            ## for i in xrange(AE_dict['nAugment']):
            ##     newHandSuccTrData = stackSample(newHandSuccTrData, handSuccTrData)
            ##     newHandFailTrData = stackSample(newHandFailTrData, handFailTrData)

            normalTrainData   = combineData( normalTrainData, newHandSuccTrData,\
                                             AE_dict['add_option'], d['handFeatureNames'], \
                                             add_noise_features=AE_dict['add_noise_option'] )
            abnormalTrainData = combineData( abnormalTrainData, newHandFailTrData,\
                                             AE_dict['add_option'], d['handFeatureNames'])
            normalTestData   = combineData( normalTestData, handSuccTeData,\
                                            AE_dict['add_option'], d['handFeatureNames'])
            abnormalTestData  = combineData( abnormalTestData, handFailTeData,\
                                             AE_dict['add_option'], d['handFeatureNames'])

            ## # reduce dimension by pooling
            ## pooling_param_dict  = {'dim': AE_dict['filterDim']} # only for AE        
            ## normalTrainData, pooling_param_dict = dm.variancePooling(normalTrainData, \
            ##                                                          pooling_param_dict)
            ## abnormalTrainData, _ = dm.variancePooling(abnormalTrainData, pooling_param_dict)
            ## normalTestData, _    = dm.variancePooling(normalTestData, pooling_param_dict)
            ## abnormalTestData, _  = dm.variancePooling(abnormalTestData, pooling_param_dict)

        ## # add noise
        ##     normalTrainData += np.random.normal(0.0, 0.03, np.shape(normalTrainData) ) 

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
    if AE_dict['switch'] and AE_dict['add_option'] is not None:
        tag = ''
        for ft in AE_dict['add_option']:
            tag += ft[:2]
        roc_pkl = os.path.join(processed_data_path, 'roc_'+task_name+'_raw_'+tag+'.pkl')
    elif AE_dict['switch'] and AE_dict['add_option'] is None:
        roc_pkl = os.path.join(processed_data_path, 'roc_'+task_name+'_raw.pkl')
    else:
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

    osvm_data = None ; bpsvm_data = None
    if 'osvm' in method_list  and ROC_data['osvm']['complete'] is False:
        ## nFiles = data_dict['nNormalFold']*data_dict['nAbnormalFold']
        osvm_data = dm.getPCAData(len(kFold_list), crossVal_pkl, \
                                  window=SVM_dict['raw_window_size'],
                                  use_test=True, use_pca=False)
    if 'bpsvm' in method_list and ROC_data['bpsvm']['complete'] is False:

        # get ll_cut_idx only for pos data
        pos_dict = []
        for idx in xrange(len(kFold_list)):
            modeling_pkl = os.path.join(processed_data_path, 'hmm_'+task_name+'_'+str(idx)+'.pkl')
            d            = ut.load_pickle(modeling_pkl)
            ll_classifier_train_X   = d['ll_classifier_train_X']
            ll_classifier_train_Y   = d['ll_classifier_train_Y']         
            ll_classifier_train_idx = d['ll_classifier_train_idx']
            l_cut_idx = dm.getHMMCuttingIdx(ll_classifier_train_X, \
                                         ll_classifier_train_Y, \
                                         ll_classifier_train_idx)
            idx_dict={'abnormal_train_cut_idx': l_cut_idx}
            pos_dict.append(idx_dict)
                    
        bpsvm_data = dm.getPCAData(len(kFold_list), crossVal_pkl, \
                                   window=SVM_dict['raw_window_size'], \
                                   pos_dict=pos_dict, use_test=True, use_pca=False)
        
    
    # parallelization
    if debug: n_jobs=1
    else: n_jobs=-1
    r = Parallel(n_jobs=n_jobs, verbose=50)(delayed(run_classifiers)( idx, processed_data_path, task_name, \
                                                                 method, ROC_data, \
                                                                 ROC_dict, AE_dict, \
                                                                 SVM_dict, raw_data=(osvm_data,bpsvm_data),\
                                                                 startIdx=startIdx, nState=nState) \
                                                                 for idx in xrange(len(kFold_list)) \
                                                                 for method in method_list )
                                                                  
    #l_data = zip(*r)
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
             timeList=timeList)
                       

def evaluation_noise(subject_names, task_name, raw_data_path, processed_data_path, param_dict,\
                     data_renew=False, save_pdf=False, verbose=False, debug=False,\
                     no_plot=False):

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

    crossVal_pkl = os.path.join(processed_data_path, 'cv_'+task_name+'.pkl')    
    if os.path.isfile(crossVal_pkl) and data_renew is False:
        d = ut.load_pickle(crossVal_pkl)
        kFold_list  = d['kFoldList']
    else:
        '''
        Use augmented data? if nAugment is 0, then aug_successData = successData
        '''        
        sys.exit()
        
    #-----------------------------------------------------------------------------------------
    # parameters
    startIdx    = 4
    method_list = ROC_dict['methods'] 
    nPoints     = ROC_dict['nPoints']

    # temp
    ## kFold_list = kFold_list[0:1]

    #-----------------------------------------------------------------------------------------
    # add noise
    modeling_pkl_prefix = 'hmm_'+task_name+'_noise'
    for idx in xrange(len(kFold_list)):
        modeling_noise_pkl = os.path.join(processed_data_path, modeling_pkl_prefix+'_'+str(idx)+'.pkl')
        if os.path.isfile(modeling_noise_pkl) and HMM_dict['renew'] is False: continue
        
        modeling_pkl = os.path.join(processed_data_path, 'hmm_'+task_name+'_'+str(idx)+'.pkl')

        d = ut.load_pickle(modeling_pkl)
        ll_classifier_train_X   = d['ll_classifier_train_X']
        ll_classifier_train_Y   = d['ll_classifier_train_Y']         
        ll_classifier_train_idx = d['ll_classifier_train_idx']
        ll_classifier_test_X    = d['ll_classifier_test_X']  
        ll_classifier_test_Y    = d['ll_classifier_test_Y']
        ll_classifier_test_idx  = d['ll_classifier_test_idx']

        ll_classifier_test_X    = ll_classifier_train_X
        ll_classifier_test_Y    = ll_classifier_train_Y
        ll_classifier_test_idx  = ll_classifier_train_idx
        
        # exclude only normal data
        l_normal_test_X = []
        l_normal_test_Y = []
        l_normal_test_idx = []
        for i in xrange(len(ll_classifier_test_Y)):
            if ll_classifier_test_Y[i][0] > 0.0:
                continue
            l_normal_test_X.append( ll_classifier_test_X[i] )
            l_normal_test_Y.append( ll_classifier_test_Y[i] )
            l_normal_test_idx.append( ll_classifier_test_idx[i] )

        # get abnormal minimum likelihood
        logp_min = np.amin(np.array(ll_classifier_test_X)[:,:,0])
        if logp_min > 0:
            print "min loglikelihood is positive!!!!!!!!!!!!!!!!!!!!"
            sys.exit()

        #temp
        logp_min = -1000
        
        # add random extreme noise
        # maybe sample x length x features
        offset = 30
        l_abnormal_test_X = []
        l_abnormal_test_Y = (np.array(l_normal_test_Y)*-1.0).tolist()
        
        l_abnormal_test_idx = copy.deepcopy(l_normal_test_idx)
        for i in xrange(len(l_normal_test_X)):
            length  = len(l_normal_test_X[i])
            rnd_idx = random.randint(0+offset,length-1-offset)

            l_x = copy.deepcopy(l_normal_test_X[i])
            l_x = np.array(l_x)
            l_x[rnd_idx:rnd_idx+1,0] += logp_min
            ## l_x[rnd_idx][0]   += random.uniform(10.0*logp_min, 20.0*logp_min)
            ## print l_x[rnd_idx][0], np.shape(l_x)
            
            ## if add_logp_d:
            ##     l_x[rnd_idx][1] /= l_normal_test_X[i][rnd_idx][0] - l_normal_test_X[i][rnd_idx-1][0]
            ##     l_x[rnd_idx][1] *= l_x[rnd_idx][0]-l_x[rnd_idx-1][0] 

            ##     l_x[rnd_idx+1][1] /= l_normal_test_X[i][rnd_idx+1][0] - l_normal_test_X[i][rnd_idx][0]
            ##     l_x[rnd_idx+1][1] *= l_x[rnd_idx+1][0]-l_x[rnd_idx][0]
            ##     print "added random noise!!!"

            l_abnormal_test_X.append(l_x)
            
        new_test_X   = l_normal_test_X + l_abnormal_test_X 
        new_test_Y   = l_normal_test_Y + l_abnormal_test_Y 
        new_test_idx = l_normal_test_idx + l_abnormal_test_idx 

        d['ll_classifier_test_X']  = new_test_X
        d['ll_classifier_test_Y']  = new_test_Y
        d['ll_classifier_test_idx']= new_test_idx        
        ut.save_pickle(d, modeling_noise_pkl)
                 
    #-----------------------------------------------------------------------------------------
    roc_pkl = os.path.join(processed_data_path, 'roc_noise_'+task_name+'.pkl')
        
    if os.path.isfile(roc_pkl) is False or HMM_dict['renew']:        
        ROC_data = {}
    else:
        ROC_data = ut.load_pickle(roc_pkl)

    for ii, method in enumerate(method_list):
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
            r = Parallel(n_jobs=n_jobs, verbose=50)(delayed(run_classifiers)( idx, processed_data_path, \
                                                                              task_name, \
                                                                              method, ROC_data, \
                                                                              ROC_dict, AE_dict, \
                                                                              SVM_dict,\
                                                                              startIdx=startIdx, nState=nState,\
                                                                              modeling_pkl_prefix=modeling_pkl_prefix) \
                                                                              for idx in xrange(len(kFold_list)) \
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
    # ---------------- ROC Result ----------------------
    ## modeling_noise_pkl = os.path.join(processed_data_path, modeling_pkl_prefix+'_'+str(0)+'.pkl')
    ## d = ut.load_pickle(modeling_noise_pkl)
    ## ll_classifier_test_Y    = d['ll_classifier_test_Y']
    roc_info(method_list, ROC_data, nPoints, delay_plot=delay_plot, no_plot=no_plot, save_pdf=save_pdf)


def evaluation_drop(subject_names, task_name, raw_data_path, processed_data_path, param_dict,\
                    data_renew=False, save_pdf=False, verbose=False, debug=False,\
                    no_plot=False, delay_plot=False):

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
    modeling_pkl_prefix = 'hmm_drop_'+task_name


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
    param_dict  = d['param_dict']
    if 'timeList' in param_dict.keys():
        timeList = param_dict['timeList'][startIdx:]
    else: timeList = None


    #-----------------------------------------------------------------------------------------
    # Training HMM, and getting classifier training and testing data
    for idx, (normalTrainIdx, abnormalTrainIdx, normalTestIdx, abnormalTestIdx) \
      in enumerate(kFold_list):

        if verbose: print idx, " : training hmm and getting classifier training and testing data"
            
        ref_modeling_pkl = os.path.join(ref_data_path, 'hmm_'+task_name+'_'+str(idx)+'.pkl')
        if os.path.isfile(ref_modeling_pkl) is False:
            print "No reference modeling file exists"
            sys.exit()
        
        modeling_pkl = os.path.join(processed_data_path, modeling_pkl_prefix+'_'+str(idx)+'.pkl')
        if not (os.path.isfile(modeling_pkl) is False or HMM_dict['renew'] or data_renew): continue

        # dim x sample x length
        normalTestData    = successData[:, normalTestIdx, :] 
        abnormalTestData  = failureData[:, abnormalTestIdx, :] 

        # scaling
        if verbose: print "scaling data"
        normalTestData    *= HMM_dict['scale']
        abnormalTestData  *= HMM_dict['scale']

        # training hmm
        if verbose: print "start to fit hmm"
        dd = ut.load_pickle(ref_modeling_pkl)
        nEmissionDim = dd['nEmissionDim']
        A  = dd['A']
        B  = dd['B']
        pi = dd['pi']
        F  = dd['F']
        
        nLength      = len(normalTestData[0][0]) - startIdx
        
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


        # random drop
        samples = []
        drop_idx_l = []
        drop_length = 20
        for i in xrange(len(testDataX[0])):
            ## rnd_idx_l = np.unique( np.random.randint(0, nLength-1, 20) )
            start_idx = np.random.randint(0, nLength-1, 1)[0]
            if start_idx < startIdx: start_idx=startIdx
            end_idx   = start_idx+drop_length
            if end_idx > nLength-1: end_idx = nLength-1
            rnd_idx_l = range(start_idx, end_idx)

            sample = []
            for j in xrange(len(testDataX)):
                sample.append( np.delete( testDataX[j][i], rnd_idx_l ) )

            samples.append(sample)
            drop_idx_l.append(start_idx)
        testDataX = np.swapaxes(samples, 0, 1)

        r = Parallel(n_jobs=-1)(delayed(hmm.computeLikelihoods)(i, A, B, pi, F, \
                                                                [ testDataX[j][i] for j in xrange(nEmissionDim) ], \
                                                                nEmissionDim, nState,\
                                                                startIdx=startIdx, \
                                                                bPosterior=True)
                                                                for i in xrange(len(testDataX[0])))
        _, ll_classifier_test_idx, ll_logp, ll_post = zip(*r)

        # nSample x nLength
        ll_classifier_test_X, ll_classifier_test_Y = \
          hmm.getHMMinducedFeatures(ll_logp, ll_post, testDataY, c=1.0, add_delta_logp=add_logp_d)

        #-----------------------------------------------------------------------------------------
        d = {}
        d['nEmissionDim'] = nEmissionDim
        d['A']            = A 
        d['B']            = B 
        d['pi']           = pi
        d['F']            = F
        d['nState']       = nState
        d['startIdx']     = startIdx
        d['ll_classifier_train_X']   = dd['ll_classifier_train_X']
        d['ll_classifier_train_Y']   = dd['ll_classifier_train_Y']
        d['ll_classifier_train_idx'] = dd['ll_classifier_train_idx']
        d['ll_classifier_test_X']    = ll_classifier_test_X
        d['ll_classifier_test_Y']    = ll_classifier_test_Y            
        d['ll_classifier_test_idx']  = ll_classifier_test_idx
        d['nLength']      = nLength
        d['drop_idx_l']   = drop_idx_l
        d['drop_length']  = drop_length
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
    roc_pkl = os.path.join(processed_data_path, 'roc_drop_'+task_name+'.pkl')
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


    osvm_data = None ; bpsvm_data = None
    if 'bpsvm' in method_list and ROC_data['bpsvm']['complete'] is False:

        # get ll_cut_idx only for pos data
        pos_dict  = []
        drop_dict = []
        for idx in xrange(len(kFold_list)):
            modeling_pkl = os.path.join(processed_data_path, 'hmm_drop_'+task_name+'_'+str(idx)+'.pkl')
            d            = ut.load_pickle(modeling_pkl)
            ll_classifier_train_X   = d['ll_classifier_train_X']
            ll_classifier_train_Y   = d['ll_classifier_train_Y']         
            ll_classifier_train_idx = d['ll_classifier_train_idx']
            l_cut_idx = dm.getHMMCuttingIdx(ll_classifier_train_X, \
                                         ll_classifier_train_Y, \
                                         ll_classifier_train_idx)
            idx_dict={'abnormal_train_cut_idx': l_cut_idx}
            pos_dict.append(idx_dict)
            drop_dict.append([d['drop_idx_l'], d['drop_length']])
                    
        bpsvm_data = dm.getPCAData(len(kFold_list), crossVal_pkl, \
                                   window=SVM_dict['raw_window_size'], \
                                   pos_dict=pos_dict, use_test=True, use_pca=False,
                                   test_drop_elements=drop_dict)

    # parallelization
    if debug: n_jobs=1
    else: n_jobs=-1
    r = Parallel(n_jobs=n_jobs, verbose=50)(delayed(run_classifiers)( idx, processed_data_path, task_name, \
                                                                 method, ROC_data, \
                                                                 ROC_dict, AE_dict, \
                                                                 SVM_dict, raw_data=(osvm_data,bpsvm_data),\
                                                                 startIdx=startIdx, nState=nState,\
                                                                 modeling_pkl_prefix=modeling_pkl_prefix) \
                                                                 for idx in xrange(len(kFold_list)) \
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
    roc_info(method_list, ROC_data, nPoints, delay_plot=delay_plot, no_plot=no_plot, save_pdf=save_pdf)



def evaluation_freq(subject_names, task_name, raw_data_path, processed_data_path, param_dict,\
                    refSampleSize,\
                    data_renew=False, save_pdf=False, verbose=False, debug=False,\
                    no_plot=False, delay_plot=False):

    ## Parameters
    # data
    data_dict  = param_dict['data_param']
    data_renew = data_dict['renew']
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
    ref_data_path = os.path.join(processed_data_path, '../'+str(refSampleSize)+'_4')
    modeling_pkl_prefix = 'hmm_freq_'+task_name


    #------------------------------------------
    # Get features
    if os.path.isdir(processed_data_path) is False:
        os.system('mkdir -p '+processed_data_path)

    crossVal_pkl = os.path.join(processed_data_path, 'cv_'+task_name+'.pkl')
    
    if os.path.isfile(crossVal_pkl) and data_renew is False:
        d = ut.load_pickle(crossVal_pkl)
        kFold_list  = d['kFoldList']
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

        # TODO: hardcoded...
        refCrossVal_pkl = os.path.join(ref_data_path, 'cv_'+task_name+'.pkl')
        dd = ut.load_pickle(refCrossVal_pkl)
        kFold_list  = dd['kFoldList']
                           
        d['kFoldList']   = kFold_list
        ut.save_pickle(d, crossVal_pkl)


    #-----------------------------------------------------------------------------------------
    # parameters
    startIdx    = 4
    method_list = ROC_dict['methods'] 
    nPoints     = ROC_dict['nPoints']

    successData = d['successData']
    failureData = d['failureData']
    param_dict  = d['param_dict']
    if 'timeList' in param_dict.keys():
        timeList    = param_dict['timeList'][startIdx:]
    else: timeList = None


    #-----------------------------------------------------------------------------------------
    # Training HMM, and getting classifier training and testing data
    for idx, (normalTrainIdx, abnormalTrainIdx, normalTestIdx, abnormalTestIdx) \
      in enumerate(kFold_list):

        if verbose: print idx, " : training hmm and getting classifier training and testing data"
            

        ref_modeling_pkl = os.path.join(ref_data_path, 'hmm_'+task_name+'_'+str(idx)+'.pkl')
        if os.path.isfile(ref_modeling_pkl) is False:
            print "No reference modeling file exists"
            sys.exit()
        
        modeling_pkl = os.path.join(processed_data_path, modeling_pkl_prefix+'_'+str(idx)+'.pkl')
        if not (os.path.isfile(modeling_pkl) is False or HMM_dict['renew'] or data_renew): continue

        # dim x sample x length
        normalTestData    = successData[:, normalTestIdx, :] 
        abnormalTestData  = failureData[:, abnormalTestIdx, :] 

        # scaling
        if verbose: print "scaling data"
        normalTestData    *= HMM_dict['scale']
        abnormalTestData  *= HMM_dict['scale']

        # training hmm
        if verbose: print "start to fit hmm"
        dd = ut.load_pickle(ref_modeling_pkl)
        nEmissionDim = dd['nEmissionDim']
        A  = dd['A']
        B  = dd['B']
        pi = dd['pi']
        F  = dd['F']
        
        nLength      = len(normalTestData[0][0]) - startIdx
        
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

        r = Parallel(n_jobs=-1)(delayed(hmm.computeLikelihoods)(i, A, B, pi, F, \
                                                                [ testDataX[j][i] for j in xrange(nEmissionDim) ], \
                                                                nEmissionDim, nState,\
                                                                startIdx=startIdx, \
                                                                bPosterior=True)
                                                                for i in xrange(len(testDataX[0])))
        _, ll_classifier_test_idx, ll_logp, ll_post = zip(*r)

        # nSample x nLength
        ll_classifier_test_X, ll_classifier_test_Y = \
          hmm.getHMMinducedFeatures(ll_logp, ll_post, testDataY, c=1.0, add_delta_logp=add_logp_d)

        #-----------------------------------------------------------------------------------------
        d = {}
        d['nEmissionDim'] = nEmissionDim
        d['A']            = A 
        d['B']            = B 
        d['pi']           = pi
        d['F']            = F
        d['nState']       = nState
        d['startIdx']     = startIdx
        d['ll_classifier_train_X']   = dd['ll_classifier_train_X']
        d['ll_classifier_train_Y']   = dd['ll_classifier_train_Y']
        d['ll_classifier_train_idx'] = dd['ll_classifier_train_idx']
        d['ll_classifier_test_X']    = ll_classifier_test_X
        d['ll_classifier_test_Y']    = ll_classifier_test_Y            
        d['ll_classifier_test_idx']  = ll_classifier_test_idx
        d['nLength']      = nLength
        ut.save_pickle(d, modeling_pkl)


    #-----------------------------------------------------------------------------------------
    roc_pkl = os.path.join(processed_data_path, 'roc_freq_'+task_name+'.pkl')
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
    r = Parallel(n_jobs=n_jobs, verbose=50)(delayed(run_classifiers)( idx, processed_data_path, task_name, \
                                                                 method, ROC_data, \
                                                                 ROC_dict, AE_dict, \
                                                                 SVM_dict, \
                                                                 startIdx=startIdx, nState=nState,\
                                                                 modeling_pkl_prefix=modeling_pkl_prefix) \
                                                                 for idx in xrange(len(kFold_list)) \
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
    roc_info(method_list, ROC_data, nPoints, delay_plot=delay_plot, no_plot=no_plot, save_pdf=save_pdf)



def run_classifiers(idx, processed_data_path, task_name, method,\
                    ROC_data, ROC_dict, AE_dict, SVM_dict,\
                    raw_data=None, startIdx=4, nState=25, \
                    modeling_pkl_prefix=None):

    #-----------------------------------------------------------------------------------------
    nPoints     = ROC_dict['nPoints']

    data = {}
    # pass method if there is existing result
    data[method] = {}
    data[method]['tp_l'] = [ [] for j in xrange(nPoints) ]
    data[method]['fp_l'] = [ [] for j in xrange(nPoints) ]
    data[method]['tn_l'] = [ [] for j in xrange(nPoints) ]
    data[method]['fn_l'] = [ [] for j in xrange(nPoints) ]
    data[method]['delay_l'] = [ [] for j in xrange(nPoints) ]

    if ROC_data[method]['complete'] == True: return data
    #-----------------------------------------------------------------------------------------

    ## print idx, " : training classifier and evaluate testing data"
    # train a classifier and evaluate it using test data.
    from hrl_anomaly_detection.classifiers import classifier as cb
    from sklearn import preprocessing

    if method == 'osvm' or method == 'bpsvm':
        if method == 'osvm': raw_data_idx = 0
        elif method == 'bpsvm': raw_data_idx = 1
            
        X_train_org = raw_data[raw_data_idx][idx]['X_scaled']
        Y_train_org = raw_data[raw_data_idx][idx]['Y_train_org']
        idx_train_org = raw_data[raw_data_idx][idx]['idx_train_org']
        ll_classifier_test_X    = raw_data[raw_data_idx][idx]['X_test']
        ll_classifier_test_Y    = raw_data[raw_data_idx][idx]['Y_test']
        ll_classifier_test_idx  = raw_data[raw_data_idx][idx]['idx_test']

        nLength = 200
    else:

        if modeling_pkl_prefix is not None:
            modeling_pkl = os.path.join(processed_data_path, modeling_pkl_prefix+'_'+str(idx)+'.pkl')            
        else:        
            if AE_dict['switch'] and AE_dict['add_option'] is not None:
                tag = ''
                for ft in AE_dict['add_option']: tag += ft[:2]
                modeling_pkl = os.path.join(processed_data_path, 'hmm_'+task_name+'_raw_'+tag+'_'+\
                                            str(idx)+'.pkl')
            elif AE_dict['switch'] and AE_dict['add_option'] is None:
                modeling_pkl = os.path.join(processed_data_path, 'hmm_'+task_name+'_raw_'+str(idx)+'.pkl')
            else:
                modeling_pkl = os.path.join(processed_data_path, 'hmm_'+task_name+'_'+str(idx)+'.pkl')

        print "start to load hmm data, ", modeling_pkl
        d            = ut.load_pickle(modeling_pkl)
        nState       = d['nState']        
        ll_classifier_train_X   = d['ll_classifier_train_X']
        ll_classifier_train_Y   = d['ll_classifier_train_Y']         
        ll_classifier_train_idx = d['ll_classifier_train_idx']
        ll_classifier_test_X    = d['ll_classifier_test_X']  
        ll_classifier_test_Y    = d['ll_classifier_test_Y']
        ll_classifier_test_idx  = d['ll_classifier_test_idx']
        nLength      = d['nLength']

        if method == 'hmmosvm':
            normal_idx = [x for x in range(len(ll_classifier_train_X)) if ll_classifier_train_Y[x][0]<0 ]
            ll_classifier_train_X = np.array(ll_classifier_train_X)[normal_idx]
            ll_classifier_train_Y = np.array(ll_classifier_train_Y)[normal_idx]
            ll_classifier_train_idx = np.array(ll_classifier_train_idx)[normal_idx]
        elif method == 'hmmsvm_dL':
            # replace dL/(ds+e) to dL
            for i in xrange(len(ll_classifier_train_X)):
                for j in xrange(len(ll_classifier_train_X[i])):
                    if j == 0:
                        ll_classifier_train_X[i][j][1] = 0.0
                    else:
                        ll_classifier_train_X[i][j][1] = ll_classifier_train_X[i][j][0] - \
                          ll_classifier_train_X[i][j-1][0]

            for i in xrange(len(ll_classifier_test_X)):
                for j in xrange(len(ll_classifier_test_X[i])):
                    if j == 0:
                        ll_classifier_test_X[i][j][1] = 0.0
                    else:
                        ll_classifier_test_X[i][j][1] = ll_classifier_test_X[i][j][0] - \
                          ll_classifier_test_X[i][j-1][0]
        elif method == 'hmmsvm_LSLS':
            # reconstruct data into LS(t-1)+LS(t)
            if type(ll_classifier_train_X) is list:
                ll_classifier_train_X = np.array(ll_classifier_train_X)

            x = np.dstack([ll_classifier_train_X[:,:,:1], ll_classifier_train_X[:,:,2:]] )
            x = x.tolist()

            new_x = []
            for i in xrange(len(x)):
                new_x.append([])
                for j in xrange(len(x[i])):
                    if j == 0:
                        new_x[i].append( x[i][j]+x[i][j] )
                    else:
                        new_x[i].append( x[i][j-1]+x[i][j] )

            ll_classifier_train_X = new_x

            # test data
            if len(np.shape(ll_classifier_test_X))<3:
                x = []
                for sample in ll_classifier_test_X:
                    x.append( np.hstack( [np.array(sample)[:,:1], np.array(sample)[:,2:]] ).tolist() )
            else:
                if type(ll_classifier_test_X) is list:
                    ll_classifier_test_X = np.array(ll_classifier_test_X)

                x = np.dstack([ll_classifier_test_X[:,:,:1], ll_classifier_test_X[:,:,2:]] )
                x = x.tolist()

            new_x = []
            for i in xrange(len(x)):
                new_x.append([])
                for j in xrange(len(x[i])):
                    if j == 0:
                        new_x[i].append( x[i][j]+x[i][j] )
                    else:
                        new_x[i].append( x[i][j-1]+x[i][j] )

            ll_classifier_test_X = new_x
        elif method == 'hmmsvm_no_dL':
            # remove dL related things
            ll_classifier_train_X = np.array(ll_classifier_train_X)
            ll_classifier_train_X = np.delete(ll_classifier_train_X, 1, 2).tolist()

            if len(np.shape(ll_classifier_test_X))<3:
                x = []
                for sample in ll_classifier_test_X:
                    x.append( np.hstack( [np.array(sample)[:,:1], np.array(sample)[:,2:]] ).tolist() )
                ll_classifier_test_X = x
            else:
                ll_classifier_test_X = np.array(ll_classifier_test_X)
                ll_classifier_test_X = np.delete(ll_classifier_test_X, 1, 2).tolist()
            
                          
        # flatten the data
        if method.find('svm')>=0 or method.find('sgd')>=0: remove_fp=True
        else: remove_fp = False
        X_train_org, Y_train_org, idx_train_org = dm.flattenSample(ll_classifier_train_X, \
                                                                   ll_classifier_train_Y, \
                                                                   ll_classifier_train_idx,\
                                                                   remove_fp=remove_fp)
                                                                   


    #-----------------------------------------------------------------------------------------
    # Generate parameter list for ROC curve
    # pass method if there is existing result
    # data preparation
    if method == 'osvm' or method == 'bpsvm':
        X_scaled = X_train_org
    elif method.find('svm')>=0 or method.find('sgd')>=0:
        scaler = preprocessing.StandardScaler()
        X_scaled = scaler.fit_transform(X_train_org)
    else:
        X_scaled = X_train_org
    print method, " : Before classification : ", np.shape(X_scaled), np.shape(Y_train_org)

    X_test = []
    Y_test = [] 
    for j in xrange(len(ll_classifier_test_X)):
        if len(ll_classifier_test_X[j])==0: continue

        try:
            if method == 'osvm' or method == 'bpsvm':
                X = ll_classifier_test_X[j]
            elif method.find('svm')>=0 or method.find('sgd')>=0:
                X = scaler.transform(ll_classifier_test_X[j])                                
            else:
                X = ll_classifier_test_X[j]
        except:
            print "failed to scale ", np.shape(ll_classifier_test_X[j])
            continue

        X_test.append(X)
        Y_test.append(ll_classifier_test_Y[j])


    # classifier # TODO: need to make it efficient!!
    dtc = cb.classifier( method=method, nPosteriors=nState, nLength=nLength )
    for j in xrange(nPoints):
        ## _, tp_l, fp_l, fn_l, tn_l, delay_l = cb.run_classifier(j, X_scaled, Y_train_org, idx_train_org,\
        ##                                                        X_test, Y_test, ll_classifier_test_idx,\
        ##                                                        method, nState, nLength, nPoints, \
        ##                                                        SVM_dict, ROC_dict, dtc=dtc)

        ## cb.run_classifier(j)
        dtc.set_params( **SVM_dict )
        if method == 'svm' or method == 'hmmsvm_diag' or method == 'hmmsvm_dL' or method == 'hmmsvm_LSLS' or \
          method == 'bpsvm' or method == 'hmmsvm_no_dL':
            weights = ROC_dict[method+'_param_range']
            dtc.set_params( class_weight=weights[j] )
            ret = dtc.fit(X_scaled, Y_train_org, idx_train_org, parallel=False)                
        elif method == 'hmmosvm' or method == 'osvm':
            weights = ROC_dict[method+'_param_range']
            dtc.set_params( svm_type=2 )
            dtc.set_params( gamma=weights[j] )
            ret = dtc.fit(X_scaled, np.array(Y_train_org)*-1.0, parallel=False)
        elif method == 'cssvm':
            weights = ROC_dict[method+'_param_range']
            dtc.set_params( class_weight=weights[j] )
            ret = dtc.fit(X_scaled, np.array(Y_train_org)*-1.0, idx_train_org, parallel=False)                
        elif method == 'progress_time_cluster':
            thresholds = ROC_dict['progress_param_range']
            dtc.set_params( ths_mult = thresholds[j] )
            if j==0: ret = dtc.fit(X_scaled, Y_train_org, idx_train_org, parallel=False)                
        elif method == 'progress_state':
            thresholds = ROC_dict[method+'_param_range']
            dtc.set_params( ths_mult = thresholds[j] )
            if j==0: ret = dtc.fit(X_scaled, Y_train_org, idx_train_org, parallel=False)                
        elif method == 'fixed':
            thresholds = ROC_dict[method+'_param_range']
            dtc.set_params( ths_mult = thresholds[j] )
            if j==0: ret = dtc.fit(X_scaled, Y_train_org, idx_train_org, parallel=False)
        elif method == 'change':
            thresholds = ROC_dict[method+'_param_range']
            dtc.set_params( ths_mult = thresholds[j] )
            if j==0: ret = dtc.fit(ll_classifier_train_X, ll_classifier_train_Y, ll_classifier_train_idx)
        elif method == 'sgd':
            weights = ROC_dict[method+'_param_range']
            dtc.set_params( class_weight=weights[j] )
            ret = dtc.fit(X_scaled, Y_train_org, idx_train_org, parallel=False)                
        else:
            print "Not available method"
            return "Not available method", -1, params

        if ret is False:
            print "fit failed, ", weights[j]
            sys.exit()
            return 'fit failed', [],[],[],[],[]
        
        # evaluate the classifier
        tp_l = []
        fp_l = []
        tn_l = []
        fn_l = []
        delay_l = []
        delay_idx = 0
        for ii in xrange(len(X_test)):
            if len(Y_test[ii])==0: continue

            if method == 'osvm' or method == 'cssvm' or method == 'hmmosvm':
                est_y = dtc.predict(X_test[ii], y=np.array(Y_test[ii])*-1.0)
                est_y = np.array(est_y)* -1.0
            else:
                est_y    = dtc.predict(X_test[ii], y=Y_test[ii])

            anomaly = False
            for jj in xrange(len(est_y)):
                if est_y[jj] > 0.0:
                    if Y_test[ii][0] <0:
                        print "anomaly idx", jj, " true label: ", Y_test[ii][0] #, X_test[ii][jj]

                    ## if method == 'hmmosvm':
                    ##     window_size = 5 #3
                    ##     if jj < len(est_y)-window_size:
                    ##         if np.sum(est_y[jj:jj+window_size])>=window_size:
                    ##             anomaly = True                            
                    ##             break
                    ##     continue                        
                    
                    if ll_classifier_test_idx is not None and Y_test[ii][0]>0:
                        try:
                            delay_idx = ll_classifier_test_idx[ii][jj]
                        except:
                            print "Error!!!!!!!!!!!!!!!!!!"
                            print np.shape(ll_classifier_test_idx), ii, jj
                        delay_l.append(delay_idx)
                            
                    anomaly = True
                    break        

            if Y_test[ii][0] > 0.0:
                if anomaly: tp_l.append(1)
                else: fn_l.append(1)
            elif Y_test[ii][0] <= 0.0:
                if anomaly: fp_l.append(1)
                else: tn_l.append(1)

        data[method]['tp_l'][j] += tp_l
        data[method]['fp_l'][j] += fp_l
        data[method]['fn_l'][j] += fn_l
        data[method]['tn_l'][j] += tn_l
        data[method]['delay_l'][j] += delay_l

    print "finished ", idx, method
    return data
                       
        
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
            data_pkl = os.path.join(processed_data_path, task_name+'_success_'+rf_center+\
                                    '_'+str(local_range))
            raw_data_dict, interp_data_dict = loadData(success_list, isTrainingData=True,
                                                       downSampleSize=downSampleSize,\
                                                       local_range=local_range, rf_center=rf_center,\
                                                       global_data=global_data, \
                                                       renew=data_renew, save_pkl=data_pkl, verbose=verbose)
        else:
            if verbose: print "Load failure data"
            data_pkl = os.path.join(processed_data_path, task_name+'_failure_'+rf_center+\
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
                if key.find('timesList')>=0:
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
   

def plotDecisionBoundaries(subjects, task, raw_data_path, save_data_path, param_dict,\
                           methods,\
                           success_viz=True, failure_viz=False, save_pdf=False,\
                           db_renew=False):
    from sklearn import preprocessing
    from sklearn.externals import joblib

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
    SVM_dict = param_dict['SVM']
    # ROC
    ROC_dict = param_dict['ROC']
    nPoints  = 4 #ROC_dict['nPoints']

    foldIdx = 0

    # temp
    ROC_dict['progress_param_range'] = np.linspace(-5, 1., nPoints)
    ROC_dict['svm_param_range']      = np.logspace(-1.5, 0., nPoints)

    # Generative model ----------------------------------------------------------------------------
    if AE_dict['switch'] and AE_dict['add_option'] is not None:
        tag = ''
        for ft in AE_dict['add_option']:
            tag += ft[:2]
        modeling_pkl = os.path.join(save_data_path, 'hmm_'+task+'_raw_'+tag+'_'+str(foldIdx)+'.pkl')
    elif AE_dict['switch'] and AE_dict['add_option'] is None:
        modeling_pkl = os.path.join(save_data_path, 'hmm_'+task+'_raw_'+str(foldIdx)+'.pkl')
    else:
        modeling_pkl = os.path.join(save_data_path, 'hmm_'+task+'_'+str(foldIdx)+'.pkl')

    if os.path.isfile(modeling_pkl) is False:
        print "No HMM modeling file exists:", modeling_pkl
    else:
        d = ut.load_pickle(modeling_pkl)
        nEmissionDim = d['nEmissionDim'] 
        nState       = d['nState']    
        startIdx     = d['startIdx']     
        ll_classifier_train_X   = d['ll_classifier_train_X']  
        ll_classifier_train_Y   = d['ll_classifier_train_Y']             
        ll_classifier_train_idx = d['ll_classifier_train_idx']
        ll_classifier_test_X    = d['ll_classifier_test_X'] 
        ll_classifier_test_Y    = d['ll_classifier_test_Y']            
        ll_classifier_test_idx  = d['ll_classifier_test_idx'] 
        nLength                 = d['nLength']

    # Get time
    all_data_pkl     = os.path.join(save_data_path, task+'_all_'+data_dict['rf_center']+\
                                    '_'+str(data_dict['local_range'])+'_interp.pkl' )
    all_data_dict = ut.load_pickle(all_data_pkl)
    timeList  = all_data_dict['timesList'][0][startIdx:]
    print np.shape(timeList)

    # ----------------------------------------------------------
    bd_data    = os.path.join(save_data_path, 'hmm_bd_data_'+task+'_'+str(foldIdx)+'.pkl')
    scaler_model = os.path.join(save_data_path, 'hmm_bd_scaler_'+task+'_'+str(foldIdx)+'.pkl')
    if os.path.isfile(bd_data) and db_renew is False:
        dd = ut.load_pickle(bd_data)
        X_train_flat        = dd['X_train_flat']
        Y_train_flat        = dd['Y_train_flat']
        idx_train_flat      = dd['idx_train_flat']
        X_train_flat_scaled = dd['X_train_flat_scaled']            
        X_test_flat        = dd['X_test_flat']
        Y_test_flat        = dd['Y_test_flat'] 
        X_test_flat_scaled = dd['X_test_flat_scaled']
        X_bg        = dd['X_bg']
        x1          = dd['x1']
        x2          = dd['x2']
        xx_normal   = dd['xx_normal']
        xx_abnormal = dd['xx_abnormal']
        
        if os.path.isfile(scaler_model):
            ml_scaler = joblib.load(scaler_model)
        else:
            ml_scaler = preprocessing.StandardScaler()
            X_train_flat_scaled = ml_scaler.fit_transform(X_train_flat)         
    else:    
        # flatten train data
        X_train_flat = [] #
        Y_train_flat = [] #
        idx_train_flat = []
        post_list = []
        for i in xrange(len(ll_classifier_train_X)):
            for j in xrange(len(ll_classifier_train_X[i])):
                X_train_flat.append(ll_classifier_train_X[i][j])
                Y_train_flat.append(ll_classifier_train_Y[i][j])
                idx_train_flat.append(ll_classifier_train_idx[i][j])
                post_list.append(ll_classifier_train_X[i][j][1:])

        # flatten test data
        X_test_flat, Y_test_flat, _ = flattenSample(ll_classifier_test_X, \
                                                    ll_classifier_test_Y, \
                                                    ll_classifier_test_idx)


        # ------------------ scaling -----------------------------------------------
        if os.path.isfile(scaler_model):
            ml_scaler = joblib.load(scaler_model)
            X_train_flat_scaled = ml_scaler.transform(X_train_flat)
        else:
            ml_scaler = preprocessing.StandardScaler()
            X_train_flat_scaled = ml_scaler.fit_transform(X_train_flat) 
            joblib.dump(ml_scaler, scaler_model)
        X_test_flat_scaled  = ml_scaler.transform(X_test_flat)
        X_train_flat_scaled = np.array(X_train_flat_scaled) #
        X_test_flat_scaled  = np.array(X_test_flat_scaled) #
        
        # --------------------------------------------------------------------------
        # Generate hidden-state distribution axis over center...
        n_logp = 100
        ## n_post = 100
        ## post_exp_list = np.zeros((n_post,nState))
        ## mean_list     = np.linspace(0,nState-1,n_post)
        ## std_list      = [0.3]*n_post #[0.5]*n_post

        ## from scipy.stats import norm
        ## for i in xrange(n_post):
        ##     rv = norm(loc=mean_list[i],scale=std_list[i])
        ##     post_exp_list[i] = rv.pdf(np.linspace(0,nState-1,nState))
        ##     post_exp_list[i] /=np.sum(post_exp_list[i])

        n_post = len(ll_classifier_train_X[0])
        post_exp_list = []
        for i in xrange(len(ll_classifier_train_X[0])):
            post = np.array(ll_classifier_train_X)[:,i,1:]
            post = np.mean(post, axis=0)
            post /= np.sum(post)
            post_exp_list.append(post)
        post_exp_list = np.array(post_exp_list)

        # Discriminative classifier ---------------------------------------------------
        # Adjusting range
        ## x1_min, x1_max = X_train_flat_pca[:, 0].min() , X_train_flat_pca[:, 0].max() 
        x2_min, x2_max = np.array(X_train_flat)[:,:1].min() , np.array(X_train_flat)[:,:1].max()

        # create a mesh to plot in
        x1, x2 = np.meshgrid(range(n_post),
                             np.linspace(x2_min, x2_max, n_logp) )

        # Background
        print "Run background data"
        data = np.c_[x1.ravel(), x2.ravel()]
        X_bg = np.hstack([ np.array([x2.ravel()]).T, post_exp_list[x1.ravel().tolist()] ])
        print np.shape(X_bg)

        # test points
        print "Run test data"
        ## Y_test_flat_est = dtc.predict(np.array(X_test_flat))
        xx_normal = []
        xx_abnormal = []
        for x,y in zip(X_test_flat, Y_test_flat):
            post = x[1:]
            ## min_index, min_dist = cf.findBestPosteriorDistribution(post, post_exp_list)
            if y > 0: xx_abnormal.append([min_index,x[0]])
            else:     xx_normal.append([min_index,x[0]])
        xx_normal   = np.array(xx_normal)
        xx_abnormal = np.array(xx_abnormal)

        dd = {}
        dd['X_train_flat']       = X_train_flat 
        dd['Y_train_flat']       = Y_train_flat
        dd['idx_train_flat']     = idx_train_flat
        dd['X_train_flat_scaled']= X_train_flat_scaled
        dd['X_test_flat']        = X_test_flat
        dd['Y_test_flat']        = Y_test_flat
        dd['X_test_flat_scaled'] = X_test_flat_scaled
        dd['X_bg']               = X_bg
        dd['x1']                 = x1
        dd['x2']                 = x2
        dd['xx_normal']          = xx_normal
        dd['xx_abnormal']        = xx_abnormal        
        ut.save_pickle(dd, bd_data)

    # -----------------------------------------------------------------------------
    print "Run classifier"
    methods = ['svm']
    methods = ['progress_time_cluster']

    from matplotlib import rc
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    rc('text', usetex=True)
    fig = plt.figure(1)
    for method in methods:
        
        lines  = []
        labels = []
        print "Current method ", method
        bd_method_data = os.path.join(save_data_path, 'hmm_bd_'+task+'_'+str(foldIdx)+'_'+method+'.pkl')
        if os.path.isfile(bd_method_data):
            d = ut.load_pickle(bd_method_data)
        else:
            d = {}
            d[method] = method
            d['data'] = []

            # scaling?
            if method.find('svm')>=0: X_scaled = X_train_flat_scaled
            else: X_scaled = X_train_flat            
            dtc = cf.classifier( method=method, nPosteriors=nState, nLength=nLength)

            # weight number
            startPoint = 0
            for j in xrange(nPoints):
                dtc.set_params( **SVM_dict )
                if method == 'svm':
                    weights = ROC_dict['svm_param_range']
                    dtc.set_params( class_weight=weights[j] )
                    dtc.set_params( kernel_type=0 ) # temp
                    ret = dtc.fit(X_scaled, Y_train_flat, idx_train_flat, parallel=False)                
                elif method == 'cssvm':
                    weights = ROC_dict['cssvm_param_range']
                    dtc.set_params( class_weight=weights[j] )
                    ret = dtc.fit(X_scaled, Y_train_flat, idx_train_flat, parallel=False)                
                elif method == 'progress_time_cluster':
                    weights = ROC_dict['progress_param_range']
                    dtc.set_params( ths_mult=weights[j] )
                    if j==startPoint:
                        ret = dtc.fit(X_scaled, Y_train_flat, idx_train_flat, parallel=False)                
                elif method == 'fixed':
                    weights = ROC_dict['fixed_param_range']
                    dtc.set_params( ths_mult=weights[j] )
                    if j==startPoint:
                        ret = dtc.fit(X_scaled, Y_train_flat, idx_train_flat, parallel=False)                

                if method.find('svm')>=0:
                    print "SVM Weight: ", weights[j], np.shape(X_bg)
                    X_bg_scaled = ml_scaler.transform(X_bg)
                else:
                    print "Progress? Weight: ", weights[j]
                    X_bg_scaled = X_bg

                z = dtc.predict(np.array(X_bg_scaled))                
                print np.amin(z), np.amax(z), " : ", np.amin(Y_train_flat), np.amax(Y_train_flat)

                if np.amax(z) == np.amin(z):
                    print "Max equals to min. Wrong classification!"
                z = np.array(z)
                z = z.reshape(np.shape(x1)) 
                ## plt.contourf(x1, x2, z, cmap=plt.cm.Paired)
                #plt.contourf(x1, x2, z, levels=np.linspace(z.min(), 0, 7), cmap=plt.cm.Blues_r) # -1: blue, 1.0: red
                #plt.contourf(x1, x2, z, levels=[0, z.max()], colors='orange')
                d['data'].append( {'x1': x1, 'x2': x2, 'z':z, 'weight':weights[j] } )

            ut.save_pickle(d, bd_method_data)

        for data in d['data']:
            x1     = data['x1']
            x2     = data['x2']
            z      = data['z']
            weight = data['weight']
            
            color = colors.next()
            CS=plt.contour(x1, x2, z, levels=[0], linewidths=2, colors=color)
            ## plt.clabel(CS, inline=1, fontsize=10)
            lines.append(CS.collections[0])
            if method.find('svm')>=0:
                labels.append( r'$w_{+1}$ = '+'{0:.3f}'.format(weight) )
            else:
                labels.append( r'$c$ = '+'{0:.3f}'.format(weight) )

        plt.scatter(xx_abnormal[:,0],xx_abnormal[:,1],c='red', marker='x', \
                    label="Anomalous data")
        plt.scatter(xx_normal[:,0],xx_normal[:,1],c='blue', marker='o', lw=0,\
                    label="Non-anomalous data")
        plt.axis('tight')
        ## plt.axis('off')
        if np.amin(xx_normal[:,1])>0:
            plt.ylim([ np.amin(xx_normal[:,1])-150, np.amax(xx_normal[:,1])*1. ])
        else:
            plt.ylim([ np.amin(xx_normal[:,1])*1.3, np.amax(xx_normal[:,1])*1. ])
            
        plt.legend(lines, labels, loc=4, prop={'size':12})
        plt.ylabel("Log-likelihood", fontsize=22)
        plt.xlabel("Progress Vector Change [Index]", fontsize=22)
        ## plt.xlabel("Time [s]", fontsize=22)

        n_post = len(ll_classifier_train_X[0])
        time_array = np.linspace(0,n_post-1,3)
        plt.xticks(np.linspace(0,n_post-1,3).astype(int))
        ## time_array = np.linspace(timeList[0], timeList[-1],3)
        ## tick_list = []
        ## for i in xrange(3):
        ##     tick_list.append( "{0:.3f}".format(time_array[i]) )
        ## plt.xticks(np.linspace(0,n_post-1,3), tick_list )

        if save_pdf is False:
            plt.show()
        else:
            print "Save pdf to Dropbox folder "
            fig.savefig('test.pdf')
            fig.savefig('test.png')
            os.system('mv test.* ~/Dropbox/HRL/')


def roc_info(method_list, ROC_data, nPoints, delay_plot=False, no_plot=False, save_pdf=False,\
             timeList=None):
    # ---------------- ROC Visualization ----------------------
    
    print "Start to visualize ROC curves!!!"
    ## ROC_data = ut.load_pickle(roc_pkl)        

    if no_plot is False:
        if delay_plot:
            fig = plt.figure(figsize=(5,8))
            colors = itertools.cycle(['y', 'g', 'b', 'k', 'y','r', 'b', ])
            
        else:
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
        acc_l = []

        if timeList is not None:
            time_step = (timeList[-1]-timeList[0])/float(len(timeList)-1)
            ## print np.shape(timeList), timeList[0], timeList[-1], (timeList[-1]-timeList[0])/float(len(timeList))
            print "time_step[s] = ", time_step, " length: ", timeList[-1]-timeList[0]
        else:
            time_step = 1.0

        for i in xrange(nPoints):
            tpr_l.append( float(np.sum(tp_ll[i]))/float(np.sum(tp_ll[i])+np.sum(fn_ll[i]))*100.0 )
            fpr_l.append( float(np.sum(fp_ll[i]))/float(np.sum(fp_ll[i])+np.sum(tn_ll[i]))*100.0 )
            fnr_l.append( 100.0 - tpr_l[-1] )

            delay_mean_l.append( np.mean(np.array(delay_ll[i])*time_step) )
            delay_std_l.append( np.std(np.array(delay_ll[i])*time_step) )
            acc_l.append( float(np.sum(tp_ll[i]+tn_ll[i])) / float(np.sum(tp_ll[i]+fn_ll[i]+fp_ll[i]+tn_ll[i])) * 100.0 )

        # add edge
        ## fpr_l = [0] + fpr_l + [100]
        ## tpr_l = [0] + tpr_l + [100]

        print "--------------------------------"
        print " AUC and delay "
        print "--------------------------------"
        print method
        print tpr_l
        print fpr_l
        print metrics.auc([0] + fpr_l + [100], [0] + tpr_l + [100], True)
        print "--------------------------------"

        if method == 'svm': label='HMM-BPSVM'
        elif method == 'progress_time_cluster': label='HMM-D'
        elif method == 'progress_state': label='HMMs with a dynamic threshold + state_clsutering'
        elif method == 'fixed': label='HMM-F'
        elif method == 'change': label='HMM-C'
        elif method == 'cssvm': label='HMM-CSSVM'
        elif method == 'sgd': label='SGD'
        elif method == 'hmmosvm': label='HMM-OneClassSVM'
        elif method == 'hmmsvm_diag': label='HMM-SVM with diag cov'
        elif method == 'osvm': label='Kernel-SVM'
        elif method == 'bpsvm': label='BPSVM'
        else: label = method

        if no_plot is False:
            # visualization
            color = colors.next()
            shape = shapes.next()
            ax1 = fig.add_subplot(111)

            if delay_plot:
                if method not in ['fixed', 'progress_time_cluster', 'svm']: continue
                if method == 'fixed': color = 'y'
                if method == 'progress_time_cluster': color = 'g'
                if method == 'svm': color = 'b'
                plt.plot(acc_l, delay_mean_l, '-'+color, label=label, linewidth=2.0)
                ## plt.plot(acc_l, delay_mean_l, '-'+shape+color, label=label, mec=color, ms=6, mew=2)
                
                ## rate = np.array(tpr_l)/(np.array(fpr_l)+0.001)
                ## for i in xrange(len(rate)):
                ##     if rate[i] > 100: rate[i] = 100.0
                cut_idx = np.argmax(acc_l)
                if delay_mean_l[0] < delay_mean_l[-1]:
                    acc_l = acc_l[:cut_idx+1]                
                    delay_mean_l = np.array(delay_mean_l[:cut_idx+1])
                    delay_std_l  = np.array(delay_std_l[:cut_idx+1])
                else:
                    acc_l = acc_l[cut_idx:]                
                    delay_mean_l = np.array(delay_mean_l[cut_idx:])
                    delay_std_l  = np.array(delay_std_l[cut_idx:])

                ## delay_mean_l = np.array(delay_mean_l)
                delay_std_l  = np.array(delay_std_l) #*0.674
                    
                
                ## plt.plot(acc_l, delay_mean_l-delay_std_l, '--'+color)
                ## plt.plot(acc_l, delay_mean_l+delay_std_l, '--'+color)
                plt.fill_between(acc_l, delay_mean_l-delay_std_l, delay_mean_l+delay_std_l, \
                                 facecolor=color, alpha=0.15, lw=0.0, interpolate=True)
                plt.xlim([49, 101])
                plt.ylim([0, 7.0])
                plt.ylabel('Detection Time [s]', fontsize=24)
                plt.xlabel('Accuracy (percentage)', fontsize=24)

                plt.xticks([50, 100], fontsize=22)
                ## plt.yticks([0, 50, 100], fontsize=22)
            else:                
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

    ## if no_plot is False:
    ##     if delay_plot:
    ##         plt.legend(loc='upper right', prop={'size':24})
    ##     else:
    ##         plt.legend(loc='lower right', prop={'size':24})

    if save_pdf:
        ## task = 'feeding'
        ## fig.savefig('delay_'+task+'.pdf')
        ## fig.savefig('delay_'+task+'.png')
        fig.savefig('test.pdf')
        fig.savefig('test.png')
        os.system('cp test.p* ~/Dropbox/HRL/')
    elif no_plot is False:
        plt.show()
    





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
    p.add_option('--decision_boundary', '--db', action='store_true', dest='bDecisionBoundary',
                 default=False, help='Plot decision boundaries.')
    
    p.add_option('--aeDataExtraction', '--ae', action='store_true', dest='bAEDataExtraction',
                 default=False, help='Extract auto-encoder data.')
    p.add_option('--aeDataExtractionPlot', '--aep', action='store_true', dest='bAEDataExtractionPlot',
                 default=False, help='Extract auto-encoder data and plot it.')
    p.add_option('--aeDataAddFeature', '--aea', action='store_true', dest='bAEDataAddFeature',
                 default=False, help='Add hand-crafted data.')

    p.add_option('--evaluation_all', '--ea', action='store_true', dest='bEvaluationAll',
                 default=False, help='Evaluate a classifier with cross-validation.')
    p.add_option('--evaluation_drop', '--ead', action='store_true', dest='bEvaluationWithDrop',
                 default=False, help='Evaluate a classifier with cross-validation plus drop.')
    p.add_option('--evaluation_noise', '--ean', action='store_true', dest='bEvaluationWithNoise',
                 default=False, help='Evaluate a classifier with cross-validation plus noise.')
    p.add_option('--plot_progress_hmmosvm', '--pph', action='store_true', dest='bPlotProgressVSHMMOSVM',
                 default=False, help='plot.')
    p.add_option('--evaluation_freq', '--eaf', action='store_true', dest='bEvaluationWithDiffFreq',
                 default=False, help='Evaluate a classifier with cross-validation and different sampling\
                 frequency.')
    
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
        subjects = ['Wonyoung', 'Tom', 'lin', 'Ashwin', 'Song', 'Henry2'] #'Henry', 
        raw_data_path, save_data_path, param_dict = getScooping(opt.task, opt.bDataRenew, \
                                                                opt.bAERenew, opt.bHMMRenew,\
                                                                rf_center, local_range,\
                                                                ae_swtch=opt.bAESwitch, dim=opt.dim)
        
    #---------------------------------------------------------------------------
    elif opt.task == 'feeding':
        subjects = [ 'Ashwin', 'Song', 'tom' , 'lin', 'wonyoung']
        raw_data_path, save_data_path, param_dict = getFeeding(opt.task, opt.bDataRenew, \
                                                               opt.bAERenew, opt.bHMMRenew,\
                                                               rf_center, local_range,\
                                                               ae_swtch=opt.bAESwitch, dim=opt.dim)
        
    #---------------------------------------------------------------------------           
    elif opt.task == 'pushing_microwhite':
        subjects = ['gatsbii']
        raw_data_path, save_data_path, param_dict = getPushingMicroWhite(opt.task, opt.bDataRenew, \
                                                                         opt.bAERenew, opt.bHMMRenew,\
                                                                         rf_center, local_range, \
                                                                         ae_swtch=opt.bAESwitch, dim=opt.dim)
                                                                         
    #---------------------------------------------------------------------------           
    elif opt.task == 'pushing_microblack':
        subjects = ['gatsbii']
        raw_data_path, save_data_path, param_dict = getPushingMicroBlack(opt.task, opt.bDataRenew, \
                                                                         opt.bAERenew, opt.bHMMRenew,\
                                                                         rf_center, local_range, \
                                                                         ae_swtch=opt.bAESwitch, dim=opt.dim)
        
    #---------------------------------------------------------------------------           
    elif opt.task == 'pushing_toolcase':
        subjects = ['gatsbii']
        raw_data_path, save_data_path, param_dict = getPushingToolCase(opt.task, opt.bDataRenew, \
                                                                       opt.bAERenew, opt.bHMMRenew,\
                                                                       rf_center, local_range, \
                                                                       ae_swtch=opt.bAESwitch, dim=opt.dim)
        
    else:
        print "Selected task name is not available."
        sys.exit()

    #---------------------------------------------------------------------------
    ## if opt.bAEDataAddFeature:
    ##     param_dict['AE']['add_option'] = ['wristAudio'] #'featureToBottleneck'
    ##     param_dict['AE']['switch']     = True
    
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
        failureData = False
        modality_list   = ['kinematics', 'audio', 'ft', 'vision_artag'] # raw plot
        
        data_plot(subjects, opt.task, raw_data_path, save_data_path,\
                  downSampleSize=param_dict['data_param']['downSampleSize'], \
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
        modality_list   = ['kinematics', 'ft']
        success_viz = True
        failure_viz = False

        data_selection(subjects, opt.task, raw_data_path, save_data_path,\
                       downSampleSize=param_dict['data_param']['downSampleSize'], \
                       local_range=local_range, rf_center=rf_center, \
                       success_viz=success_viz, failure_viz=failure_viz,\
                       raw_viz=opt.bRawDataPlot, save_pdf=opt.bSavePdf,\
                       modality_list=modality_list, data_renew=opt.bDataRenew, verbose=opt.bVerbose)        

    elif opt.bFeaturePlot:
        success_viz = True
        failure_viz = False
        
        dm.getDataSet(subjects, opt.task, raw_data_path, save_data_path,
                      param_dict['data_param']['rf_center'], param_dict['data_param']['local_range'],\
                      downSampleSize=param_dict['data_param']['downSampleSize'], scale=scale, \
                      success_viz=success_viz, failure_viz=failure_viz,\
                      ae_data=False,\
                      cut_data=param_dict['data_param']['cut_data'],\
                      save_pdf=opt.bSavePdf, solid_color=False,\
                      handFeatures=param_dict['data_param']['handFeatures'], data_renew=opt.bDataRenew)

    elif opt.bDecisionBoundary:
        success_viz = True
        failure_viz = False
        methods     = ['svm', 'progress_time_cluster']

        plotDecisionBoundaries(subjects, opt.task, raw_data_path, save_data_path, param_dict,\
                               methods,\
                               success_viz, failure_viz, save_pdf=opt.bSavePdf, db_renew=opt.bClassifierRenew)

    elif opt.bAEDataExtraction:
        param_dict['AE']['switch']     = True
        aeDataExtraction(subjects, opt.task, raw_data_path, save_data_path, param_dict, verbose=opt.bVerbose)

    elif opt.bAEDataExtractionPlot:
        success_viz = True
        failure_viz = True
        handFeature_viz = False
        param_dict['AE']['switch']     = True        
        aeDataExtraction(subjects, opt.task, raw_data_path, save_data_path, param_dict,\
                         handFeature_viz=handFeature_viz,\
                         success_viz=success_viz, failure_viz=failure_viz,\
                         verbose=opt.bVerbose)


    elif opt.bLikelihoodPlot:
        likelihoodOfSequences(subjects, opt.task, raw_data_path, save_data_path, param_dict,\
                              decision_boundary_viz=False, \
                              useTrain=False, useNormalTest=True, useAbnormalTest=True,\
                              useTrain_color=False, useNormalTest_color=False, useAbnormalTest_color=False,\
                              hmm_renew=opt.bHMMRenew, data_renew=opt.bDataRenew, save_pdf=opt.bSavePdf,\
                              verbose=opt.bVerbose)
                              
    elif opt.bEvaluationAll or opt.bPlotProgressVSHMMOSVM:
        if opt.bHMMRenew: param_dict['ROC']['methods'] = ['fixed', 'progress_time_cluster'] #, 'change']
        if opt.bNoUpdate: param_dict['ROC']['update_list'] = []
        if opt.bPlotProgressVSHMMOSVM:
            param_dict['ROC']['methods'] = ['hmmosvm', 'progress_time_cluster'] 
            param_dict['ROC']['update_list'] = []
            param_dict['HMM']['renew'] = False
            param_dict['SVM']['renew'] = False
        
        evaluation_all(subjects, opt.task, raw_data_path, save_data_path, param_dict, save_pdf=opt.bSavePdf, \
                       verbose=opt.bVerbose, debug=opt.bDebug, no_plot=opt.bNoPlot)

    elif opt.bEvaluationWithNoise:
        ## param_dict['ROC']['methods'] = ['progress_time_cluster']
        ## param_dict['ROC']['update_list'] = ['progress_time_cluster']
        param_dict['ROC']['methods']     = ['svm']
        param_dict['ROC']['update_list'] = []
        param_dict['ROC']['nPoints']     = 20
        param_dict['ROC']['svm_param_range'] = np.linspace(0.0001, 1.8, param_dict['ROC']['nPoints'])
        param_dict['ROC']['progress_param_range'] = np.linspace(-1, -16., param_dict['ROC']['nPoints'])
        
        evaluation_noise(subjects, opt.task, raw_data_path, save_data_path, param_dict, save_pdf=opt.bSavePdf, \
                         verbose=opt.bVerbose, debug=opt.bDebug, no_plot=opt.bNoPlot)

    elif opt.bEvaluationWithDrop:

        param_dict['ROC']['methods']     = ['svm', 'hmmsvm_LSLS', 'hmmsvm_dL', 'hmmsvm_no_dL', 'bpsvm']
        param_dict['ROC']['update_list'] = ['svm', 'hmmsvm_LSLS', 'hmmsvm_dL', 'hmmsvm_no_dL', 'bpsvm']
        if opt.bNoUpdate: param_dict['ROC']['update_list'] = []

        save_data_path = os.path.expanduser('~')+\
          '/hrl_file_server/dpark_data/anomaly/RSS2016/'+opt.task+'_data/'+\
          str(param_dict['data_param']['downSampleSize'])+'_'+str(opt.dim)+'_drop'

        evaluation_drop(subjects, opt.task, raw_data_path, save_data_path, param_dict, \
                        save_pdf=opt.bSavePdf, \
                        verbose=opt.bVerbose, debug=opt.bDebug, no_plot=opt.bNoPlot)

    elif opt.bEvaluationWithDiffFreq:
        '''
        Change into different sampling frequency or sample drop
        '''
        param_dict['ROC']['methods'] = ['svm', 'hmmsvm_LSLS', 'hmmsvm_dL', 'hmmsvm_no_dL', 'bpsvm']
        param_dict['ROC']['update_list'] = []
        if opt.bNoUpdate: param_dict['ROC']['update_list'] = []
        param_dict['HMM']['renew'] = False
        param_dict['SVM']['renew'] = False
        nPoints = param_dict['ROC']['nPoints']
        refSampleSize = param_dict['data_param']['downSampleSize']
        
        
        for sampleSize in [50, 400]:
            print "============================="
            print "Sample Size: ", sampleSize
            print "============================="
            param_dict['data_param']['downSampleSize'] = sampleSize
            save_data_path = os.path.expanduser('~')+\
              '/hrl_file_server/dpark_data/anomaly/RSS2016/'+opt.task+'_data/'+\
              str(param_dict['data_param']['downSampleSize'])+'_'+str(opt.dim)

            if sampleSize == 50:
                param_dict['ROC']['update_list'] = ['hmmsvm_no_dL', 'hmmsvm_dL', 'hmmsvm_LSLS', 'svm']
                if opt.bNoUpdate: param_dict['ROC']['update_list'] = []
                if opt.task == "pushing_microwhite":
                    param_dict['ROC']['hmmsvm_dL_param_range'] *= 1.0
                    param_dict['ROC']['hmmsvm_LSLS_param_range'] *= 1.0
                    param_dict['ROC']['svm_param_range'] = np.logspace(-2.0, -0.1, nPoints)
                if opt.task == "pushing_toolcase":
                    param_dict['ROC']['hmmsvm_dL_param_range'] *= 1.0
                    param_dict['ROC']['hmmsvm_LSLS_param_range'] = np.logspace(-1.5, 0.0, nPoints)
                    param_dict['ROC']['svm_param_range'] = np.logspace(-2.5, 0.0, nPoints)
                if opt.task == "scooping":
                    param_dict['ROC']['hmmsvm_dL_param_range'] = np.logspace(-2.5, 0.0, nPoints) 
                    param_dict['ROC']['hmmsvm_LSLS_param_range'] = np.logspace(-4, 0.0, nPoints)
                    param_dict['ROC']['svm_param_range'] = np.logspace(-4, -1.5, nPoints) 
                    param_dict['ROC']['hmmsvm_no_dL_param_range'] = np.logspace(-4.5, -3.0, nPoints)
                if opt.task == "feeding":
                    param_dict['ROC']['hmmsvm_dL_param_range'] *= 1.0
                    param_dict['ROC']['hmmsvm_LSLS_param_range'] = np.logspace(-3, 0.8, nPoints)
                    param_dict['ROC']['svm_param_range'] = np.logspace(0.9, -3.0, nPoints)
                    
            elif sampleSize == 100:
                param_dict['ROC']['update_list'] = ['hmmsvm_dL', 'hmmsvm_LSLS', 'svm']
                if opt.bNoUpdate: param_dict['ROC']['update_list'] = []
                if opt.task == "pushing_microwhite":
                    param_dict['ROC']['hmmsvm_dL_param_range'] *= 1.0
                    param_dict['ROC']['hmmsvm_LSLS_param_range'] *= 1.0
                    param_dict['ROC']['svm_param_range'] = np.logspace(-2.0, -0.1, nPoints)
                if opt.task == "pushing_toolcase":
                    param_dict['ROC']['hmmsvm_dL_param_range'] *= 1.0
                    param_dict['ROC']['hmmsvm_LSLS_param_range'] = np.logspace(-1.5, 0.0, nPoints)
                    param_dict['ROC']['svm_param_range'] = np.logspace(-2.5, 0.0, nPoints)
                if opt.task == "scooping":
                    param_dict['ROC']['hmmsvm_dL_param_range'] = np.logspace(-2.5, 1.0, nPoints) 
                    param_dict['ROC']['hmmsvm_LSLS_param_range'] *= 0.1
                    param_dict['ROC']['svm_param_range'] = np.logspace(-3, 0.0, nPoints) 
                if opt.task == "feeding":
                    param_dict['ROC']['hmmsvm_dL_param_range'] *= 1.0
                    param_dict['ROC']['hmmsvm_LSLS_param_range'] = np.logspace(-3, 0.8, nPoints)
                    param_dict['ROC']['svm_param_range'] = np.logspace(0.9, -3.0, nPoints)
                    
            elif sampleSize > 200:
                param_dict['ROC']['update_list'] = ['hmmsvm_no_dL', 'hmmsvm_dL', 'hmmsvm_LSLS' , 'svm']
                if opt.bNoUpdate: param_dict['ROC']['update_list'] = []
                if opt.task == "pushing_microblack":
                    param_dict['ROC']['hmmsvm_dL_param_range'] = np.logspace(-4, 0.0, nPoints)
                    param_dict['ROC']['hmmsvm_LSLS_param_range'] *= 3.0
                    param_dict['ROC']['svm_param_range'] = np.logspace(-3.0, 0.0, nPoints)
                if opt.task == "pushing_microwhite":
                    param_dict['ROC']['hmmsvm_dL_param_range'] = np.logspace(-3.0, 1.2, nPoints)
                    param_dict['ROC']['hmmsvm_LSLS_param_range'] *= 15.0
                    param_dict['ROC']['svm_param_range'] *= 10.0
                if opt.task == "pushing_toolcase":
                    param_dict['ROC']['hmmsvm_dL_param_range'] *= 1.0
                    param_dict['ROC']['hmmsvm_LSLS_param_range'] = np.logspace(-4.0, 0.0, nPoints) 
                    param_dict['ROC']['svm_param_range'] = np.logspace(-3.2, 0.2, nPoints)
                if opt.task == "scooping":
                    param_dict['ROC']['hmmsvm_dL_param_range']    = np.logspace(-4.0, 3.2, nPoints) 
                    param_dict['ROC']['hmmsvm_LSLS_param_range'] *= 1.0
                    param_dict['ROC']['svm_param_range'] = np.logspace(-2.0, 1.8, nPoints) 
                    param_dict['ROC']['hmmsvm_no_dL_param_range'] = np.logspace(-1.0, 0.7, nPoints)
                if opt.task == "feeding":
                    param_dict['ROC']['hmmsvm_dL_param_range'] *= 15.0
                    param_dict['ROC']['hmmsvm_LSLS_param_range'] = np.logspace(-3, 1.5, nPoints)
                    param_dict['ROC']['svm_param_range'] = np.logspace(1.5, -2.5, nPoints)


            evaluation_freq(subjects, opt.task, raw_data_path, save_data_path, param_dict, \
                            refSampleSize,\
                            save_pdf=opt.bSavePdf, \
                            verbose=opt.bVerbose, debug=opt.bDebug, no_plot=opt.bNoPlot)

                            
