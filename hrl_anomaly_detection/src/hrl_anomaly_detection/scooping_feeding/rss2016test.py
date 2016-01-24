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
from hrl_anomaly_detection.data_manager import *
from hrl_anomaly_detection.scooping_feeding import util as sutil
## import PyKDL
## import sandbox_dpark_darpa_m3.lib.hrl_check_util as hcu
## import sandbox_dpark_darpa_m3.lib.hrl_dh_lib as hdl
## import hrl_lib.circular_buffer as cb

# learning
from hrl_anomaly_detection.hmm import learning_hmm_multi_n as hmm
from mvpa2.datasets.base import Dataset
from sklearn import svm
from joblib import Parallel, delayed

# private learner
import hrl_anomaly_detection.hmm.classifier as cf

import itertools
colors = itertools.cycle(['r', 'g', 'b', 'm', 'c', 'k', 'y'])
shapes = itertools.cycle(['x','v', 'o', '+'])


   
def likelihoodOfSequences(subject_names, task_name, raw_data_path, processed_data_path, rf_center, local_range, \
                          nSet=1, downSampleSize=200, \
                          feature_list=['crossmodal_targetRelativeDist'], \
                          nState=10, threshold=-1.0, smooth=False, cluster_type='time', \
                          useTrain=True, useNormalTest=True, useAbnormalTest=False,\
                          useTrain_color=False, useNormalTest_color=False, useAbnormalTest_color=False,\
                          hmm_renew=False, data_renew=False, save_pdf=False, show_plot=True):

    _, trainingData, abnormalTestData,_ = sutil.feature_extraction(subject_names, task_name, raw_data_path, \
                                                                   processed_data_path, rf_center, local_range,\
                                                                   nSet=nSet, \
                                                                   downSampleSize=downSampleSize, \
                                                                   feature_list=feature_list, \
                                                                   data_renew=data_renew)

    normalTestData = None                                    
    print "======================================"
    print "Training data: ", np.shape(trainingData)
    print "Normal test data: ", np.shape(normalTestData)
    print "Abnormal test data: ", np.shape(abnormalTestData)
    print "======================================"

    # training hmm
    nEmissionDim = len(trainingData)
    detection_param_pkl = os.path.join(processed_data_path, 'hmm_'+task_name+'.pkl')
    cov_mult = [10.0]*(nEmissionDim**2)
    scale = 10.0

    ml  = hmm.learning_hmm_multi_n(nState, nEmissionDim, scale=scale, cluster_type=cluster_type, verbose=False)
    ret = ml.fit(trainingData, cov_mult=cov_mult, ml_pkl=detection_param_pkl, use_pkl=True) # not(renew))
    ths = threshold

    if ret == 'Failure': 
        print "-------------------------"
        print "HMM returned failure!!   "
        print "-------------------------"
        return (-1,-1,-1,-1)
    
    if show_plot: fig = plt.figure()
    min_logp = 0.0
    max_logp = 0.0
        
    # training data
    if useTrain:

        log_ll = []
        exp_log_ll = []        
        for i in xrange(len(trainingData[0])):

            log_ll.append([])
            exp_log_ll.append([])
            for j in range(2, len(trainingData[0][i])):

                X = [x[i,:j] for x in trainingData]

                exp_logp, logp = ml.expLoglikelihood(X, ths, smooth=smooth, bLoglikelihood=True)
                log_ll[i].append(logp)
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
            
        plt.plot(log_ll[i], 'b-', lw=3.0)
        plt.plot(exp_log_ll[i], 'm-')            
                                             
    # normal test data
    if useNormalTest and False:

        log_ll = []
        exp_log_ll = []        
        for i in xrange(len(normalTestData[0])):

            log_ll.append([])
            exp_log_ll.append([])

            for j in range(2, len(normalTestData[0][i])):
                X = [x[i,:j] for x in normalTestData]                

                exp_logp, logp = ml.expLoglikelihood(X, ths, bLoglikelihood=True)
                log_ll[i].append(logp)
                exp_log_ll[i].append(exp_logp)

            if min_logp > np.amin(log_ll): min_logp = np.amin(log_ll)
            if max_logp < np.amax(log_ll): max_logp = np.amax(log_ll)

            # disp 
            if useNormalTest_color: plt.plot(log_ll[i], label=str(i))
            else: plt.plot(log_ll[i], 'g-')

            plt.plot(exp_log_ll[i], 'r*-')

        if useNormalTest_color: 
            plt.legend(loc=3,prop={'size':16})

    # abnormal test data
    if useAbnormalTest and False:
        log_ll = []
        exp_log_ll = []        
        for i in xrange(len(abnormalTestData[0])):

            log_ll.append([])
            exp_log_ll.append([])

            for j in range(2, len(abnormalTestData[0][i])):
                X = [x[i,:j] for x in abnormalTestData]                
                try:
                    logp = ml.loglikelihood(X)
                except:
                    print "Too different input profile that cannot be expressed by emission matrix"
                    return [], 0.0 # error

                log_ll[i].append(logp)

            # disp 
            plt.plot(log_ll[i], 'r-')
            ## plt.plot(exp_log_ll[i], 'r*-')


    plt.ylim([min_logp, max_logp])
    if save_pdf == True:
        fig.savefig('test.pdf')
        fig.savefig('test.png')
        os.system('cp test.p* ~/Dropbox/HRL/')
    else:
        if show_plot: plt.show()        

    return


def stateLikelihoodPlot(subject_names, task_name, raw_data_path, processed_data_path, rf_center, \
                        local_range, \
                        nSet=1, downSampleSize=200, \
                        feature_list=['crossmodal_targetRelativeDist'], \
                        nState=10, threshold=-1.0, smooth=False, cluster_type='time', \
                        classifier_type='time', \
                        useTrain=True, useNormalTest=True, useAbnormalTest=False,\
                        useTrain_color=False, useNormalTest_color=False, useAbnormalTest_color=False,\
                        hmm_renew=False, data_renew=False, save_pdf=False, show_plot=True):

    _, successData, failureData,_ = sutil.feature_extraction(subject_names, task_name, raw_data_path, \
                                                             processed_data_path, rf_center, local_range,\
                                                             nSet=nSet, \
                                                             downSampleSize=downSampleSize, \
                                                             feature_list=feature_list, \
                                                             data_renew=data_renew)

    # index selection
    success_idx  = range(len(successData[0]))
    failure_idx  = range(len(failureData[0]))
    
    nTrain       = int( 0.7*len(success_idx) )    
    train_idx    = random.sample(success_idx, nTrain)
    success_test_idx = [x for x in success_idx if not x in train_idx]
    failure_test_idx = failure_idx

    # data structure: dim x nData x sequence
    trainingData     = successData[:, train_idx, :]
    normalTestData   = successData[:, success_test_idx, :]
    abnormalTestData = failureData[:, failure_test_idx, :]
    
    print "======================================"
    print "Training data: ", np.shape(trainingData)
    print "Normal test data: ", np.shape(normalTestData)
    print "Abnormal test data: ", np.shape(abnormalTestData)
    print "======================================"

    # training hmm
    nEmissionDim = len(trainingData)
    detection_param_pkl = os.path.join(processed_data_path, 'hmm_'+task_name+'.pkl')
    cov_mult = [10.0]*(nEmissionDim**2)

    ml  = hmm.learning_hmm_multi_n(nState, nEmissionDim, scale=10.0, cluster_type=cluster_type, verbose=False)
    ret = ml.fit(trainingData, cov_mult=cov_mult, ml_pkl=detection_param_pkl, \
                 use_pkl=True) # not(renew))
    ths = [threshold]*nState

    if ret == 'Failure': 
        print "-------------------------"
        print "HMM returned failure!!   "
        print "-------------------------"
        return (-1,-1,-1,-1)


    # Visualization parameters
    fig = plt.figure()
    ax = fig.add_subplot(111)
    min_logp = 0.0
    max_logp = 100.0

    # normal test data
    _, l_logp_1, l_post_1 = ml.getPostLoglikelihoods(normalTestData)

    l_idx_1 = []
    for post in l_post_1:
        min_idx, _ = ml.findBestPosteriorDistribution(post)
        l_idx_1.append(min_idx)

    plt.plot(l_idx_1, l_logp_1, 'bo')

    
    # abnormal test data
    _, l_logp_2, l_post_2 = ml.getPostLoglikelihoods(abnormalTestData)
    
    l_idx_2 = []
    for post in l_post_2:
        min_idx, _ = ml.findBestPosteriorDistribution(post)
        l_idx_2.append(min_idx)

    plt.plot(l_idx_2, l_logp_2, 'rx')


    if min_logp > np.amin(l_logp_1)*1.3: min_logp = np.amin(l_logp_1)*1.3
    if max_logp < np.amax(l_logp_1)*1.3: max_logp = np.amax(l_logp_1)*1.3
    plt.ylim([min_logp, max_logp])

    if save_pdf == True:
        fig.savefig('test.pdf')
        fig.savefig('test.png')
        os.system('cp test.p* ~/Dropbox/HRL/')
        ## ut.get_keystroke('Hit a key to proceed next')
    else:
        plt.show()        

    
    


# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------

def evaluation(subject_names, task_name, raw_data_path, processed_data_path, rf_center, \
               local_range, \
               nSet=1, downSampleSize=200, \
               feature_list=['crossmodal_targetRelativeDist'], \
               nState=10, threshold=-1.0, smooth=False, cluster_type='time', \
               classifier_type='time', \
               hmm_renew=False, data_renew=False, save_pdf=False, show_plot=True, verbose=False):

    trainClassifier_pkl = os.path.join(processed_data_path, 'tc_'+task_name+'.pkl')
    if os.path.isfile(trainClassifier_pkl):
        d            = ut.load_pickle(trainClassifier_pkl)
        trainingData     = d['trainingData']
        normalTestData   = d['normalTestData']
        abnormalTestData = d['abnormalTestData']

        nEmissionDim = d['nEmissionDim']
        scale        = d['scale']
        A  = d['A']
        B  = d['B']
        pi = d['pi']
        F  = d['F']
        startIdx = d['startIdx']
        ll_X     = d['ll_X']
        ll_Y     = d['ll_Y']
        ll_idx   = d['ll_idx']
        ll_train_X = d['ll_train_X']
        
        import ghmm        
        if nEmissionDim >= 2:
            ml = ghmm.HMMFromMatrices(F, ghmm.MultivariateGaussianDistribution(F), \
                                      A, B, pi)
        else:
            ml = ghmm.HMMFromMatrices(F, ghmm.GaussianDistribution(F), A, B, pi)
        
    else:

        _, successData, failureData,_ = sutil.feature_extraction(subject_names, task_name, raw_data_path, \
                                                                 processed_data_path, rf_center, local_range,\
                                                                 nSet=nSet, \
                                                                 downSampleSize=downSampleSize, \
                                                                 feature_list=feature_list, \
                                                                 data_renew=data_renew)

        # index selection
        success_idx  = range(len(successData[0]))
        failure_idx  = range(len(failureData[0]))

        nTrain       = int( 0.7*len(success_idx) )    
        train_idx    = random.sample(success_idx, nTrain)
        success_test_idx = [x for x in success_idx if not x in train_idx]
        failure_test_idx = failure_idx

        # data structure: dim x sample x sequence
        trainingData     = successData[:, train_idx, :]
        normalTestData   = successData[:, success_test_idx, :]
        abnormalTestData = failureData[:, failure_test_idx, :]

        print "======================================"
        print "Training data: ", np.shape(trainingData)
        print "Normal test data: ", np.shape(normalTestData)
        print "Abnormal test data: ", np.shape(abnormalTestData)
        print "======================================"

        d = {}
        d['trainingData'] = trainingData
        d['normalTestData'] = normalTestData   
        d['abnormalTestData'] = abnormalTestData
        
        # training hmm
        nEmissionDim = len(trainingData)
        scale        = 10.0
        detection_param_pkl = os.path.join(processed_data_path, 'hmm_'+task_name+'.pkl')
        cov_mult = [scale]*(nEmissionDim**2)

        ml  = hmm.learning_hmm_multi_n(nState, nEmissionDim, scale=scale, cluster_type=cluster_type, verbose=False)
        ret = ml.fit(trainingData, cov_mult=cov_mult, ml_pkl=detection_param_pkl, use_pkl=True) # not(renew))
        ## ths = [threshold]*nState

        if ret == 'Failure': 
            print "-------------------------"
            print "HMM returned failure!!   "
            print "-------------------------"
            return (-1,-1,-1,-1)

        #-----------------------------------------------------------------------------------------
        startIdx = 4
        r = Parallel(n_jobs=-1)(delayed(hmm.computeLikelihoods)(i, ml.A, ml.B, ml.pi, ml.F, \
                                                                [ trainingData[j][i] for j in xrange(nEmissionDim) ], \
                                                                ml.nEmissionDim, ml.scale, ml.nState,\
                                                                startIdx=startIdx, \
                                                                bPosterior=True)
                                                                for i in xrange(len(trainingData)))
        _, ll_idx, ll_logp, ll_post = zip(*r)

        ll_train_X = []
        for i in xrange(len(ll_logp)):
            l_X = []
            for j in xrange(len(ll_logp[i])):        
                l_X.append( [ll_logp[i][j]] + ll_post[i][j].tolist() )

            ll_train_X.append(l_X)


        #-----------------------------------------------------------------------------------------
        # Mix normal and abnormal test data
        testDataX = []
        testDataY = []
        for i in xrange(nEmissionDim):
            temp = np.vstack([normalTestData[i], abnormalTestData[i]])
            testDataX.append( temp )

        testDataY = np.hstack([ np.zeros(len(normalTestData[0])), np.ones(len(abnormalTestData[0])) ])

        # generate feature vectors for disriminative classifiers
        r = Parallel(n_jobs=-1)(delayed(hmm.computeLikelihoods)(i, ml.A, ml.B, ml.pi, ml.F, \
                                                                [ testDataX[j][i] for j in xrange(nEmissionDim) ], \
                                                                ml.nEmissionDim, ml.scale, ml.nState,\
                                                                startIdx=startIdx, \
                                                                bPosterior=True)
                                                                for i in xrange(len(testDataY)))
        _, ll_idx, ll_logp, ll_post = zip(*r)

        ll_X = []
        ll_Y = []
        for i in xrange(len(ll_logp)):
            l_X = []
            l_Y = []
            for j in xrange(len(ll_logp[i])):        
                l_X.append( [ll_logp[i][j]] + ll_post[i][j].tolist() )
                ## l_X.append( [ll_logp[i][j]] + [ll_idx[i][j]] )

                if testDataY[i] > 0.5: l_Y.append(1.0)
                else: l_Y.append(-1.0)

            ll_X.append(l_X)
            ll_Y.append(l_Y)               


        d['nEmissionDim'] = ml.nEmissionDim
        d['scale']        = scale        
        d['A'] = ml.A 
        d['B'] = ml.B 
        d['pi']= ml.pi
        d['F'] = ml.F
        d['startIdx'] = startIdx
        d['ll_train_X'] = ll_train_X
        d['ll_X'] = ll_X 
        d['ll_Y'] = ll_Y
        d['ll_idx']= ll_idx
        ut.save_pickle(d, trainClassifier_pkl)


    #-----------------------------------------------------------------------------------------
    print "Start to train SVM"
    #-----------------------------------------------------------------------------------------
    X_org = []
    Y_org = []
    for i in xrange(len(ll_X)):
        for j in xrange(len(ll_X[i])):
            X_org.append(ll_X[i][j])
            Y_org.append(ll_Y[i][j])

    # train svm
    method_list = ['svm', 'progress_time_cluster']
    ROC_data = {}
    from hrl_anomaly_detection.classifiers import classifier_base as cb
    
    # Generate parameter list for ROC curve
    for i, method in enumerate(method_list):

        # data preparation
        if method == 'svm':
            from sklearn import preprocessing
            scaler = preprocessing.StandardScaler()
            ## scaler = preprocessing.scale()
            X_scaled = scaler.fit_transform(X_org)
        else:
            X_scaled = X_org
        print method, " : Before classification : ", np.shape(X_scaled), np.shape(Y_org)

        # containers or parameters
        tpr_l = []
        fpr_l = []
        ## tn_ll = []
        fnr_l = []
        delay_mean_l = []
        delay_std_l  = []

        # classifier
        dtc = cb.classifier( ml, method=method, nPosteriors=nState, nLength=len(trainingData[0,0]) )        
        nPoints = 10
        for j in xrange(nPoints):
        
            if method == 'svm':
                weights = np.linspace(1.0, 15.0, nPoints)
                dtc.set_params( class_weight= {1.0: 1.0, -1.0: weights[j]} )
            elif method == 'progress_time_cluster':
                thresholds = np.linspace(-60, -1.0, nPoints)
                dtc.set_params( ths_mult = thresholds[j] )

            ret = dtc.fit(X_scaled, Y_org, ll_idx)
            ## print "Score: ", dtc.score(X_scaled, Y)

            #-----------------------------------------------------------------------------------------
            print "Start to evaluate the classifiers ", j
            #-----------------------------------------------------------------------------------------
            tp_l = []
            fp_l = []
            tn_l = []
            fn_l = []
            delay_l = []
            delay_idx = 0
            for i in xrange(len(ll_X)):
                for j in xrange(len(ll_X[i])):

                    if method == 'svm':
                        X = scaler.transform([ll_X[i][j]])
                        ## X_scaled = ll_X[i][j]
                        ## X_scaled[0] = (X_scaled[0]-xmin)/(xmax-xmin)
                        ## ## X_scaled = np.array(X_scaled)[:1]
                    elif method == 'progress_time_cluster':
                        X = ll_X[i][j]

                    est_y    = dtc.predict(X)
                    if type(est_y) == list: est_y = est_y[0]
                    X = X[0]

                    if est_y > 0.0:
                        delay_idx = ll_idx[i][j]
                        ## print "Break ", i, " ", j, " in ", est_y, " = ", ll_Y[i][j]                 
                        break        

                if ll_Y[i][0] > 0.0:
                    if est_y > 0.0:
                        tp_l.append(1)
                        delay_l.append(delay_idx)
                    else: fn_l.append(1)
                elif ll_Y[i][0] <= 0.0:
                    if est_y > 0.0: fp_l.append(1)
                    else: tn_l.append(1)

            tpr_l.append( float(np.sum(tp_l))/float(np.sum(tp_l)+np.sum(fn_l)) )
            fpr_l.append( float(np.sum(fp_l))/float(np.sum(fp_l)+np.sum(tn_l)) )
            fnr_l.append( 1.0 - tpr_l[-1] )
            delay_mean_l.append( np.mean(delay_l) )
            delay_std_l.append( np.std(delay_l) )

        ROC_data[method] = {'tpr': tpr_l,
                            'fpr': fpr_l,
                            'fnr': fnr_l,
                            'delay_mean_l': delay_mean_l,
                            'delay_std_l': delay_std_l}    
            
    #-----------------------------------------------------------------------------------------
    # ---------------- ROC Visualization ----------------------
    if True:

        fig = plt.figure()
        ax1 = fig.add_subplot(121)

        for method in method_list:
            fpr_l = ROC_data[method]['fpr']
            tpr_l = ROC_data[method]['tpr']
            color = colors.next()
            plt.plot(fpr_l, tpr_l, c=color, label=method)

            print "------------------------------"
            print method
            print fpr_l
            print tpr_l
            
            
        plt.legend(loc='upper right')

        ax2 = fig.add_subplot(122)
        for method in method_list:            
            delay_mean_l = ROC_data[method]['delay_mean_l']
            delay_std_l  = ROC_data[method]['delay_std_l']
            x = range(len(delay_mean_l))
            color = colors.next()

            plt.errorbar(x, delay_mean_l, yerr=delay_std_l, c=color, label=method)

        plt.legend(loc='upper right')
        
        
    # ---------------- Boundary Visualization ----------------------
    if False:
        fig = plt.figure()
        ## for i, y in enumerate(Y):
        ##     if y > 0:
        ##         plt.plot(X_scaled[i,1], X_scaled[i,0], 'r.')
        ##     else:
        ##         plt.plot(X_scaled[i,1], X_scaled[i,0], 'b.')

        # create a mesh to plot in
        h = 0.1
        x_min, x_max = X_scaled[:, 1].min() - 0.1, X_scaled[:, 1].max() + 0.1
        y_min, y_max = X_scaled[:, 0].min() - 0.1, X_scaled[:, 0].max() + 0.1
        x_min = -2.0
        x_max = 2.0
        y_min = 0.0
        y_max = 1.5
        
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        Z = dtc.decision_function(np.c_[yy.ravel(), xx.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.Blues_r)
        plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='orange')
        ## plt.axis('off')
    

    ## print "false negative rate (FNR): ", (1.0-tpr)*100.0
    ## print "false positive rate (FPR): ", fpr*100.0
    ## print "Detection delay ", np.mean(delay_l), np.std(delay_l)


    if save_pdf:
        fig.savefig('test.pdf')
        fig.savefig('test.png')
        os.system('cp test.p* ~/Dropbox/HRL/')        
    else:
        plt.show()
    
            
                
    





def trainClassifierSVM(subject_names, task_name, raw_data_path, processed_data_path, rf_center, \
                       local_range, \
                       nSet=1, downSampleSize=200, \
                       feature_list=['crossmodal_targetRelativeDist'], \
                       nState=10, threshold=-1.0, smooth=False, cluster_type='time', \
                       classifier_type='time', \
                       hmm_renew=False, data_renew=False, save_pdf=False, show_plot=True):

    trainClassifier_pkl = os.path.join(processed_data_path, 'tc_'+task_name+'.pkl')
    if os.path.isfile(trainClassifier_pkl):
        d            = ut.load_pickle(trainClassifier_pkl)
        trainingData     = d['trainingData']
        normalTestData   = d['normalTestData']
        abnormalTestData = d['abnormalTestData']
    else:

        _, successData, failureData,_ = sutil.feature_extraction(subject_names, task_name, raw_data_path, \
                                                                 processed_data_path, rf_center, local_range,\
                                                                 nSet=nSet, \
                                                                 downSampleSize=downSampleSize, \
                                                                 feature_list=feature_list, \
                                                                 data_renew=data_renew)

        # index selection
        success_idx  = range(len(successData[0]))
        failure_idx  = range(len(failureData[0]))

        nTrain       = int( 0.7*len(success_idx) )    
        train_idx    = random.sample(success_idx, nTrain)
        success_test_idx = [x for x in success_idx if not x in train_idx]
        failure_test_idx = failure_idx

        # data structure: dim x sample x sequence
        trainingData     = successData[:, train_idx, :]
        normalTestData   = successData[:, success_test_idx, :]
        abnormalTestData = failureData[:, failure_test_idx, :]

        print "======================================"
        print "Training data: ", np.shape(trainingData)
        print "Normal test data: ", np.shape(normalTestData)
        print "Abnormal test data: ", np.shape(abnormalTestData)
        print "======================================"

        d = {}
        d['trainingData'] = trainingData
        d['normalTestData'] = normalTestData   
        d['abnormalTestData'] = abnormalTestData
        ut.save_pickle(d, trainClassifier_pkl)

        
    # training hmm
    nEmissionDim = len(trainingData)
    detection_param_pkl = os.path.join(processed_data_path, 'hmm_'+task_name+'.pkl')
    cov_mult = [10.0]*(nEmissionDim**2)

    ml  = hmm.learning_hmm_multi_n(nState, nEmissionDim, scale=10.0, cluster_type=cluster_type, verbose=False)
    ret = ml.fit(trainingData, cov_mult=cov_mult, ml_pkl=detection_param_pkl, use_pkl=True) # not(renew))
    ## ths = [threshold]*nState

    if ret == 'Failure': 
        print "-------------------------"
        print "HMM returned failure!!   "
        print "-------------------------"
        return (-1,-1,-1,-1)


    # Mix normal and abnormal test data
    testDataX = []
    testDataY = []
    for i in xrange(nEmissionDim):
        temp = np.vstack([normalTestData[i], abnormalTestData[i]])
        testDataX.append(temp)

    testDataY   = np.hstack([ np.zeros(len(normalTestData[0])), np.ones(len(abnormalTestData[0])) ])
    testIdxList = random.sample( range(len(testDataY)), len(testDataY) )

    from hrl_anomaly_detection.classifiers import classifier_base as cb
    dtc = cb.classifier(ml, method='svm')
    from sklearn.decomposition import KernelPCA # Too bad
    dr = KernelPCA(n_components=2, kernel=cb.custom_kernel, gamma=5.0)

    # NOTE !!!!!!!!!!!!!!!!!!!!!!!!!!!
    eps = 1e-6

    # train pca
    ## print "Reducing the dimension of training data"
    ## X = []
    ## for i in xrange(len(trainingData[0])):
    ##     data = [ trainingData[j][i] for j in xrange(nEmissionDim) ]        
    ##     l_logp, l_post = ml.loglikelihoods(data, bPosterior=True, startIdx=startIdx)

    ##     # set feature vector
    ##     for j in xrange(len(l_logp)):
    ##         X.append( [l_logp[j]] + l_post[j].tolist() )

    ## ## print np.shape(X), np.shape(Y)
    ## training_rx = dr.fit_transform(X)
    ## from sklearn import preprocessing    

    # update the classifier when new data comes in    
    fig = plt.figure()
    X = []
    Y = []
    for i, idx in enumerate(testIdxList):
        print "updating classifier : ", i

        ll_post = []
        ll_logp = []
        ll_label = []

        # get hmm induced features
        testData = [ testDataX[j][idx] for j in xrange(nEmissionDim) ]
        l_logp, l_post = ml.loglikelihoods(testData, bPosterior=True, startIdx=startIdx)
        
        # set feature vector
        for j in xrange(len(l_logp)):
            X.append( [l_logp[j]] + l_post[j].tolist() )
        
            if testDataY[idx] > 0.5: Y.append(1.0)
            else: Y.append(0.0)

        # ToDo: how to normalize the features?
        # loglikelihood
        x_max = max(np.array(X)[:,0])
        x_min = min(np.array(X)[:,0])
        x_std = (np.array(X)[:,0] - x_min) / (x_max - x_min)
        X_scaled = x_std / (x_max - x_min) + x_min

        X_scaled = np.hstack([np.array([X_scaled]).T, np.array(X)[:,1:]])
        X_scaled[:,1:] += eps
        print "finished to scale the feature vector"

        # train svm
        ret = dtc.fit(X_scaled,Y)
        print "finished svm training: ", ret

        ## visualize the boundary
        # --------------- Dimension Reduction --------------------------
        ## X_scaled = preprocessing.scale(X)
        test_rx = dr.fit_transform(X_scaled)

        # ---------------- Boundary Visualization ----------------------
        # create a mesh to plot in
        h = 1.0
        x_min, x_max = test_rx[:, 0].min() - 0.2, test_rx[:, 0].max() + 0.2
        y_min, y_max = test_rx[:, 1].min() - 0.2, test_rx[:, 1].max() + 0.2

        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        ## Z = dtc.dt.decision_function(np.c_[xx.ravel(), yy.ravel()])
        ## Z = Z.reshape(xx.shape)
        
        ## plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.Blues_r)
        ## plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='orange')
        plt.axis('off')
        
        # ---------------- Sample Visualization ------------------------
        #
        abnormal_rx = [x for x, y in zip(test_rx, Y) if y > 0 ]
        normal_rx = [x for x, y in zip(test_rx, Y) if y <= 0]

        if len(normal_rx) > 0:
            plt.scatter(np.array(normal_rx)[:,0], np.array(normal_rx)[:,1], c='b', label=None)
        if len(abnormal_rx) > 0:            
            plt.scatter(np.array(abnormal_rx)[:,0], np.array(abnormal_rx)[:,1], c='r', label=None)

        if save_pdf:
            fig.savefig('test.pdf')
            fig.savefig('test.png')
            os.system('cp test.p* ~/Dropbox/HRL/')        
        else:
            plt.show()
            fig = plt.figure()

    
def trainClassifier(subject_names, task_name, raw_data_path, processed_data_path, rf_center, \
                    local_range, \
                    nSet=1, downSampleSize=200, \
                    feature_list=['crossmodal_targetRelativeDist'], \
                    nState=10, threshold=-1.0, smooth=False, cluster_type='time', \
                    classifier_type='time', \
                    useTrain=True, useNormalTest=True, useAbnormalTest=False,\
                    useTrain_color=False, useNormalTest_color=False, useAbnormalTest_color=False,\
                    hmm_renew=False, data_renew=False, save_pdf=False, show_plot=True):

    _, successData, failureData,_ = sutil.feature_extraction(subject_names, task_name, raw_data_path, \
                                                             processed_data_path, rf_center, local_range,\
                                                             nSet=nSet, \
                                                             downSampleSize=downSampleSize, \
                                                             feature_list=feature_list, \
                                                             data_renew=data_renew)

    # index selection
    success_idx  = range(len(successData[0]))
    failure_idx  = range(len(failureData[0]))
    
    nTrain       = int( 0.7*len(success_idx) )    
    train_idx    = random.sample(success_idx, nTrain)
    success_test_idx = [x for x in success_idx if not x in train_idx]
    failure_test_idx = failure_idx

    # data structure: dim x nData x sequence
    trainingData     = successData[:, train_idx, :]
    normalTestData   = successData[:, success_test_idx, :]
    abnormalTestData = failureData[:, failure_test_idx, :]
    
    print "======================================"
    print "Training data: ", np.shape(trainingData)
    print "Normal test data: ", np.shape(normalTestData)
    print "Abnormal test data: ", np.shape(abnormalTestData)
    print "======================================"

    # training hmm
    nEmissionDim = len(trainingData)
    detection_param_pkl = os.path.join(processed_data_path, 'hmm_'+task_name+'.pkl')
    cov_mult = [10.0]*(nEmissionDim**2)

    ml  = hmm.learning_hmm_multi_n(nState, nEmissionDim, scale=10.0, cluster_type=cluster_type, verbose=False)
    ret = ml.fit(trainingData, cov_mult=cov_mult, ml_pkl=detection_param_pkl, \
                 use_pkl=True) # not(renew))
    ths = [threshold]*nState

    if ret == 'Failure': 
        print "-------------------------"
        print "HMM returned failure!!   "
        print "-------------------------"
        return (-1,-1,-1,-1)

    # initialize internal classifier update parameter
    ml.opt_A   = []
    ml.opt_B   = []
    ml.opt_idx = []
    ml.opt_y   = []
    ml.opt_logp= []

    log_ll     = []
    exp_log_ll = []
    dec_log_ll = []
    c1 = 1.0
    c2 = 1.0        

    # Mix normal and abnormal test data
    testDataX = []
    testDataY = []
    for i in xrange(len(trainingData)): # dimension
        temp = np.vstack([normalTestData[i], abnormalTestData[i]])
        testDataX.append(temp)

    testDataY   = np.hstack([ np.zeros(len(normalTestData[0])), np.ones(len(abnormalTestData[0])) ])
    testIdxList = random.sample( range(len(testDataY)), len(testDataY) )

    min_logp = 0.0
    max_logp = 100.0

    # update the classifier when new data comes in    
    for i, idx in enumerate(testIdxList):
    
        prev_exp_log_l = []
        l_prev_cost = []
        for j in range(2, len(testDataX[0][idx])):
            X = [x[idx,:j] for x in testDataX]                

            exp_logp, logp = ml.expLoglikelihood(X, ml.l_ths_mult, smooth=smooth, bLoglikelihood=True)
            prev_exp_log_l.append(exp_logp)
            l_prev_cost.append( max(0, 1.0+(2.0*c2-1.0)*(exp_logp-logp)) )

        #----------------------------------------------------------
        X = [x[idx,:] for x in testDataX]
        y = testDataY[idx]
        cf.updateClassifierCoff(ml, X, y, c1=c1, c2=c2, classifier_type=classifier_type)
        #----------------------------------------------------------

        log_ll.append([])
        exp_log_ll.append([])
        dec_log_ll.append([])
        l_cost = []
        for j in range(2, len(testDataX[0][idx])):
            X = [x[idx,:j] for x in testDataX]                

            exp_logp, logp = ml.expLoglikelihood(X, ml.l_ths_mult, smooth=smooth, bLoglikelihood=True)
            log_ll[i].append(logp)
            exp_log_ll[i].append(exp_logp)
            dec_log_ll[i].append(exp_logp-logp)
            l_cost.append( max(0, 1.0+(2.0*c2-1.0)*(exp_logp-logp)) )

        # visualization
        fig = plt.figure()
        ax = fig.add_subplot(211)

        for j in xrange(len(log_ll)):
            if testDataY[ testIdxList[j] ] == 0.0:
                if j == len(log_ll)-1:
                    plt.plot(log_ll[j], 'b-', lw=3.0)
                else:
                    plt.plot(log_ll[j], 'b-')

                if min_logp > np.amin(log_ll[j])*1.3: min_logp = np.amin(log_ll[j])*1.3
                if max_logp < np.amax(log_ll[j])*1.3: max_logp = np.amax(log_ll[j])*1.3
                    
            else:
                if j == len(log_ll)-1:
                    plt.plot(log_ll[j], 'r-', lw=3.0)
                else:
                    plt.plot(log_ll[j], 'r-')
                    
        plt.plot(prev_exp_log_l, 'm*--', lw=3.0)            
        plt.plot(exp_log_ll[-1], 'm-', lw=3.0)            
       # plt.plot(dec_log_ll[-1], 'g-', lw=1.0)            
        plt.ylim([min_logp, max_logp])

        ax = fig.add_subplot(212)
        plt.plot(l_prev_cost, 'b-')
        plt.plot(l_cost, 'r-')

        if save_pdf == True:
            fig.savefig('test.pdf')
            fig.savefig('test.png')
            os.system('cp test.p* ~/Dropbox/HRL/')
            ut.get_keystroke('Hit a key to proceed next')
        else:
            plt.show()        



    


    
    
                
        
def data_plot(subject_names, task_name, raw_data_path, processed_data_path, \
              nSet=1, downSampleSize=200, \
              local_range=0.3, rf_center='kinEEPos', \
              success_viz=True, failure_viz=False, \
              raw_viz=False, interp_viz=False, save_pdf=False, \
              successData=False, failureData=True,\
              continuousPlot=False, \
              ## trainingData=True, normalTestData=False, abnormalTestData=False,\
              modality_list=['audio'], data_renew=False, verbose=False):    

    success_list, failure_list = getSubjectFileList(raw_data_path, subject_names, task_name)

    fig = plt.figure('all')

    for idx, file_list in enumerate([success_list, failure_list]):
        if idx == 0 and successData is not True: continue
        elif idx == 1 and failureData is not True: continue        

        ## fig = plt.figure('loadData')                        
        # loading and time-sync
        if idx == 0:
            if verbose: print "Load success data"
            data_pkl = os.path.join(processed_data_path, subject+'_'+task+'_success_'+rf_center+\
                                    '_'+str(local_range))
            raw_data_dict, interp_data_dict = loadData(success_list, isTrainingData=False,
                                                       downSampleSize=downSampleSize,\
                                                       local_range=local_range, rf_center=rf_center,\
                                                       renew=data_renew, save_pkl=data_pkl, verbose=verbose)
        else:
            if verbose: print "Load failure data"
            data_pkl = os.path.join(processed_data_path, subject+'_'+task+'_failure_'+rf_center+\
                                    '_'+str(local_range))
            raw_data_dict, interp_data_dict = loadData(failure_list, isTrainingData=False,
                                                       downSampleSize=downSampleSize,\
                                                       local_range=local_range, rf_center=rf_center,\
                                                       renew=data_renew, save_pkl=data_pkl, verbose=verbose)
        ## plt.show()
        ## sys.exit()
                                                       
        nPlot       = len(modality_list)
        time_lim    = [0, 16] #?????????????????????????????
   
        if raw_viz: target_dict = raw_data_dict
        else: target_dict = interp_data_dict

        for fidx in xrange(len(file_list)):
                        
            count = 0
            for modality in modality_list:
                count +=1

                if 'audio' in modality:
                    time_list = target_dict['audioTimesList']
                    data_list = target_dict['audioPowerList']

                elif 'kinematics' in modality:
                    time_list = target_dict['kinTimesList']
                    data_list = target_dict['kinVelList']

                    # distance
                    new_data_list = []
                    for d in data_list:
                        new_data_list.append( np.linalg.norm(d, axis=0) )
                    data_list = new_data_list

                elif 'ft' in modality:
                    time_list = target_dict['ftTimesList']
                    data_list = target_dict['ftForceList']

                    # distance
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
                        new_data_list.append( np.linalg.norm(d, axis=0) )
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
    ## trainingData, param_dict = extractLocalFeature(data_dict['trainData'], feature_list, local_range)

    ## # test set
    ## normalTestData, _ = extractLocalFeature(data_dict['normalTestData'], feature_list, local_range, \
    ##                                         param_dict=param_dict)        
    ## abnormalTestData, _ = extractLocalFeature(data_dict['abnormalTestData'], feature_list, local_range, \
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
                  nSet=1, downSampleSize=200, \
                  local_range=0.3, rf_center='kinEEPos', \
                  success_viz=True, failure_viz=False, \
                  raw_viz=False, save_pdf=False, \
                  modality_list=['audio'], data_renew=False, verbose=False):    

    ## success_list, failure_list = getSubjectFileList(raw_data_path, subject_names, task_name)
    
    # Success data
    successData = True
    failureData = False

    count = 0
    while True:
        
        ## success_list, failure_list = getSubjectFileList(raw_data_path, subject_names, task_name)        

        ## print "-----------------------------------------------"
        ## print success_list[count]
        ## print "-----------------------------------------------"
        data_plot(subject_names, task_name, raw_data_path, processed_data_path,\
                  nSet=nSet, downSampleSize=downSampleSize, \
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
   

    

def pca_plot(subject_names, task_name, raw_data_path, processed_data_path, rf_center, local_range, \
             nSet=1, downSampleSize=200, success_viz=True, failure_viz=False, \
             save_pdf=False, \
             feature_list=['crossmodal_targetRelativeDist'], data_renew=False):


    allData, trainingData, abnormalTestData, abnormalTestNameList\
      = sutil.feature_extraction(subject_names, task_name, raw_data_path, \
                                 processed_data_path, rf_center, local_range,\
                                 nSet=nSet, \
                                 downSampleSize=downSampleSize, \
                                 feature_list=feature_list, \
                                 data_renew=data_renew)

    print "---------------------------------------------------"
    print np.shape(trainingData), np.shape(abnormalTestData)
    print "---------------------------------------------------"
    
    m,n,k = np.shape(allData)
    all_data_array = None
    for i in xrange(n):
        for j in xrange(k):
            if all_data_array is None: all_data_array = allData[:,i,j]
            else: all_data_array = np.vstack([all_data_array, allData[:,i,j]])
                
    m,n,k = np.shape(trainingData)
    success_data_array = None
    for i in xrange(n):
        for j in xrange(k):
            if success_data_array is None: success_data_array = trainingData[:,i,j]
            else: success_data_array = np.vstack([success_data_array, trainingData[:,i,j]])

    m,n,k = np.shape(abnormalTestData)
    failure_data_array = None
    for i in xrange(n):
        for j in xrange(k):
            if failure_data_array is None: failure_data_array = abnormalTestData[:,i,j]
            else: failure_data_array = np.vstack([failure_data_array, abnormalTestData[:,i,j]])

    #--------------------- Parameters -------------------------------
    fig = plt.figure()
    # step size in the mesh
    h = .01

    # ------------------- Visualization using different PCA? --------
    dr = {}
    from sklearn.manifold import Isomap
    ## dr['isomap4'] = Isomap(n_neighbors=4, n_components=2)
    ## dr['isomap5'] = Isomap(n_neighbors=5, n_components=2)
    dr['isomap4'] = Isomap(n_neighbors=4, n_components=2)
    dr['isomap7'] = Isomap(n_neighbors=7, n_components=2)
    from sklearn.decomposition import KernelPCA # Too bad
    dr['kpca_gamma5'] = KernelPCA(n_components=2, kernel="linear", gamma=5.0)
    dr['kpca_gamma2'] = KernelPCA(n_components=2, kernel="rbf", gamma=2.0)
    ## dr['kpca_gamma3'] = KernelPCA(n_components=2, kernel="sigmoid", gamma=0.3)
    ## dr['kpca_gamma5'] = KernelPCA(n_components=2, kernel="cosine", gamma=0.3)
    from sklearn.manifold import LocallyLinearEmbedding # Too bad
    ## dr['lle3'] = LocallyLinearEmbedding(n_neighbors=3, n_components=2, eigen_solver='dense')
    ## dr['lle5'] = LocallyLinearEmbedding(n_neighbors=5, n_components=2, eigen_solver='dense')
    ## dr['lle7'] = LocallyLinearEmbedding(n_neighbors=7, n_components=2, eigen_solver='dense')

    bv = {}
    from sklearn import svm
    bv['svm_gamma1'] = svm.OneClassSVM(nu=0.1, kernel='rbf', gamma=0.4)
    bv['svm_gamma2'] = svm.OneClassSVM(nu=0.1, kernel='rbf', gamma=2.0)
    bv['svm_gamma3'] = svm.OneClassSVM(nu=0.1, kernel='rbf', gamma=3.0)
    bv['svm_gamma4'] = svm.OneClassSVM(nu=0.1, kernel='rbf', gamma=4.0)


    # title for the plots
    for idx, key in enumerate(dr.keys()):
    ## for idx, key in enumerate(bv.keys()):
        ml  = dr[key]
        clf = bv['svm_gamma1'] #[key]
        plt.subplot(2, 2, idx + 1)

        # --------------- Dimension Reduction --------------------------
        success_x = ml.fit_transform(success_data_array)
        success_y = [1.0]*len(success_data_array)

        failure_x = ml.transform(failure_data_array)
        failure_y = [0.0]*len(failure_data_array)

        all_x = ml.transform(all_data_array)

        # ---------------- Boundary Visualization ----------------------
        clf.fit(success_x, success_y)

        # create a mesh to plot in
        x_min, x_max = all_x[:, 0].min() - 0.2, all_x[:, 0].max() + 0.2
        y_min, y_max = all_x[:, 1].min() - 0.2, all_x[:, 1].max() + 0.2
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.Blues_r)
        plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='orange')
        plt.axis('off')

        plt.title(key)

        # ---------------- Sample Visualization ------------------------
        if success_viz:
            plt.scatter(success_x[:,0], success_x[:,1], c='b', label=None)

        # Abnormal
        if failure_viz:
            legend_handles = []
            m,n,k = np.shape(abnormalTestData)
            for i in xrange(n):
                data_array = None
                for j in xrange(k):
                    if data_array is None: data_array = abnormalTestData[:,i,j]
                    else: data_array = np.vstack([data_array, abnormalTestData[:,i,j]])

                res = ml.transform(data_array)
                ## color = colors.next()

                cause = os.path.split(abnormalTestNameList[i])[-1].split('.pkl')[0].split('failure_')[-1]
                if 'falling' in cause: color = 'k'
                elif 'touching' == cause: color = 'r'
                elif 'slip' in cause: color = 'm'
                ## elif 'sound' in cause: color = 'g'
                else: color = 'k'
                    
                
                plt.scatter(res[:,0], res[:,1], c=color, marker='x', label=cause)
                ## legend_handles.append( h )

            ## plt.legend(loc='upper right') #handles=legend_handles) #loc='upper right', 
            
    if save_pdf:
        fig.savefig('test.pdf')
        fig.savefig('test.png')
        os.system('cp test.p* ~/Dropbox/HRL/')        
    else:
        plt.show()

if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    p.add_option('--dataRenew', '--dr', action='store_true', dest='bDataRenew',
                 default=False, help='Renew pickle files.')
    p.add_option('--hmmRenew', '--hr', action='store_true', dest='bHMMRenew',
                 default=False, help='Renew HMM parameters.')

    p.add_option('--rawplot', '--rp', action='store_true', dest='bRawDataPlot',
                 default=False, help='Plot raw data.')
    p.add_option('--interplot', '--ip', action='store_true', dest='bInterpDataPlot',
                 default=False, help='Plot raw data.')
    p.add_option('--feature', '--ft', action='store_true', dest='bFeaturePlot',
                 default=False, help='Plot features.')
    p.add_option('--dataselect', '--ds', action='store_true', dest='bDataSelection',
                 default=False, help='Plot data and select it.')
    p.add_option('--pca', action='store_true', dest='bPCAPlot',
                 default=False, help='Plot pca result.')
    p.add_option('--likelihoodplot', '--lp', action='store_true', dest='bLikelihoodPlot',
                 default=False, help='Plot the change of likelihood.')
    p.add_option('--statelikelihoodplot', '--slp', action='store_true', dest='bStateLikelihoodPlot',
                 default=False, help='Plot the log likelihoods over states.')

    p.add_option('--evaluation', '--e', action='store_true', dest='bEvaluation',
                 default=False, help='Evaluate a classifier.')
    
    p.add_option('--trainClassifier', '--tc', action='store_true', dest='bTrainClassifier',
                 default=False, help='Train a cost sensitive classifier.')
    p.add_option('--localization', '--ll', action='store_true', dest='bLocalization',
                 default=False, help='Extract local feature.')
    
    p.add_option('--renew', action='store_true', dest='bRenew',
                 default=False, help='Renew pickle files.')
    p.add_option('--savepdf', '--sp', action='store_true', dest='bSavePdf',
                 default=False, help='Save pdf files.')    
    p.add_option('--verbose', '--v', action='store_true', dest='bVerbose',
                 default=False, help='Print out.')

    opt, args = p.parse_args()

    save_data_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/RSS2016'
    raw_data_path  = '/home/dpark/hrl_file_server/dpark_data/anomaly/RSS2016/'

    #---------------------------------------------------------------------------           
    # Run evaluation
    #---------------------------------------------------------------------------           
    subject = 'gatsbii'
    task    = 'scooping'    
    ## feature_list = ['unimodal_ftForce', 'crossmodal_targetRelativeDist', \
    ##                 'crossmodal_targetRelativeAng']
    feature_list = ['unimodal_ftForce', 'crossmodal_targetRelativeDist']

    ## subject = 'gatsbii'
    ## task    = 'feeding' 
    ## feature_list = ['unimodal_audioPower', 'unimodal_ftForce', 'crossmodal_artagRelativeDist', \
    ##                 'crossmodal_artagRelativeAng']

    ## task    = 'touching'    
    #---------------------------------------------------------------------------           
    
    # Dectection TEST 
    nSet           = 1
    local_range    = 0.25    
    viz            = False
    renew          = False
    downSampleSize = 200

    if opt.bRawDataPlot or opt.bInterpDataPlot:
        '''
        Before localization: Raw data plot
        After localization: Raw or interpolated data plot
        '''
        target_data_set = 0
        rf_center       = 'kinEEPos'
        modality_list   = ['kinematics', 'audio', 'fabric', 'ft', 'vision_artag', 'vision_change', 'pps']
        ## rf_center       = 'kinForearmPos'
        ## modality_list   = ['kinematics', 'audio', 'fabric', 'vision_change']
        successData     = True #True
        failureData     = True
        local_range     = 0.15
        
        data_plot([subject], task, raw_data_path, save_data_path,\
                  nSet=target_data_set, downSampleSize=downSampleSize, \
                  local_range=local_range, rf_center=rf_center, \
                  raw_viz=opt.bRawDataPlot, interp_viz=opt.bInterpDataPlot, save_pdf=opt.bSavePdf,\
                  successData=successData, failureData=failureData,\
                  modality_list=modality_list, data_renew=opt.bDataRenew, verbose=opt.bVerbose)

    elif opt.bDataSelection:
        '''
        Manually select and filter bad data out
        '''
        target_data_set = 0
        rf_center       = 'kinEEPos'
        modality_list   = ['kinematics', 'audio', 'fabric', 'ft', 'vision_artag', 'vision_change', 'pps']
        local_range     = 0.15

        data_selection([subject], task, raw_data_path, save_data_path,\
                       nSet=target_data_set, downSampleSize=downSampleSize, \
                       local_range=local_range, rf_center=rf_center, \
                       raw_viz=opt.bRawDataPlot, save_pdf=opt.bSavePdf,\
                       modality_list=modality_list, data_renew=opt.bDataRenew, verbose=opt.bVerbose)        

    elif opt.bFeaturePlot:
        target_data_set = 0
        rf_center    = 'kinEEPos'
        ## rf_center    = 'kinForearmPos'
        feature_list = ['unimodal_audioPower',\
                        'unimodal_kinVel',\
                        'unimodal_ftForce',\
                        ##'unimodal_visionChange',\
                        'unimodal_ppsForce',\
                        'unimodal_fabricForce',\
                        'crossmodal_targetRelativeDist', \
                        'crossmodal_targetRelativeAng']
        local_range = 0.15
        success_viz = True
        failure_viz = False

        sutil.feature_extraction([subject], task, raw_data_path, save_data_path, rf_center, local_range,\
                                 nSet=target_data_set, downSampleSize=downSampleSize, \
                                 success_viz=success_viz, failure_viz=failure_viz,\
                                 save_pdf=opt.bSavePdf, solid_color=True,\
                                 feature_list=feature_list, data_renew=opt.bDataRenew)

    elif opt.bPCAPlot:
        '''
        deprecated?
        '''       
        target_data_set = 0
        ## rf_center    = 'kinEEPos'
        ## feature_list = ['unimodal_audioPower',\
        ##                 'unimodal_kinVel',\
        ##                 'unimodal_ftForce',\
        ##                 'unimodal_visionChange',\
        ##                 'unimodal_ppsForce',\
        ##                 'unimodal_fabricForce',\
        ##                 'crossmodal_targetRelativeDist', \
        ##                 'crossmodal_targetRelativeAng']
        task         = 'touching'    
        rf_center    = 'kinForearmPos'
        feature_list = ['unimodal_audioPower',\
                        'unimodal_kinVel',\
                        'unimodal_visionChange',\
                        'unimodal_fabricForce']
        local_range = 0.15
        success_viz = True
        failure_viz = True
                        
        pca_plot([subject], task, raw_data_path, save_data_path, rf_center, local_range,\
                  nSet=target_data_set, downSampleSize=downSampleSize, \
                  success_viz=success_viz, failure_viz=failure_viz,\
                  save_pdf=opt.bSavePdf,
                  feature_list=feature_list, data_renew=opt.bDataRenew)

    elif opt.bLikelihoodPlot:
        target_data_set = 0
        rf_center    = 'kinEEPos'
        ## rf_center    = 'kinForearmPos'
        feature_list = ['unimodal_audioPower',\
                        ##'unimodal_kinVel',\
                        'unimodal_ftForce',\
                        ##'unimodal_visionChange',\
                        'unimodal_ppsForce',\
                        'unimodal_fabricForce',\
                        'crossmodal_targetRelativeDist', \
                        'crossmodal_targetRelativeAng'
                        ]
        local_range = 0.15

        nState       = 20
        threshold    = 0.0
        smooth       = False
        cluster_type = 'time'
        cluster_type = 'state'

        likelihoodOfSequences([subject], task, raw_data_path, save_data_path, rf_center, local_range,\
                              nSet=target_data_set, downSampleSize=downSampleSize, \
                              feature_list=feature_list, \
                              nState=nState, threshold=threshold, smooth=smooth, cluster_type=cluster_type,\
                              useTrain=True, useNormalTest=False, useAbnormalTest=True,\
                              useTrain_color=False, useNormalTest_color=False, useAbnormalTest_color=False,\
                              hmm_renew=opt.bHMMRenew, data_renew=opt.bDataRenew, save_pdf=opt.bSavePdf)
                              
    elif opt.bStateLikelihoodPlot:
        target_data_set = 0
        rf_center    = 'kinEEPos'
        ## rf_center    = 'kinForearmPos'
        feature_list = ['unimodal_audioPower',\
                        #'unimodal_kinVel',\
                        'unimodal_ftForce',\
                        #'unimodal_visionChange',\
                        'unimodal_ppsForce',\
                        'unimodal_fabricForce',\
                        'crossmodal_targetRelativeDist', \
                        'crossmodal_targetRelativeAng'
                        ]
        local_range = 0.15

        nState    = 10
        threshold = 0.0
        smooth          = False #only related with expLoglikelihood
        ## cluster_type    = 'time'
        cluster_type = 'state'
        classifier_type = 'new'
        
        stateLikelihoodPlot([subject], task, raw_data_path, save_data_path, rf_center, local_range,\
                            nSet=target_data_set, downSampleSize=downSampleSize, \
                            feature_list=feature_list, \
                            nState=nState, threshold=threshold, smooth=smooth, cluster_type=cluster_type,\
                            classifier_type=classifier_type,\
                            useTrain=True, useNormalTest=False, useAbnormalTest=True,\
                            useTrain_color=False, useNormalTest_color=False, useAbnormalTest_color=False,\
                            hmm_renew=opt.bHMMRenew, data_renew=opt.bDataRenew, save_pdf=opt.bSavePdf)

    elif opt.bEvaluation:
        target_data_set = 0
        rf_center    = 'kinEEPos'
        ## rf_center    = 'kinForearmPos'
        feature_list = ['unimodal_audioPower',\
                        #'unimodal_kinVel',\
                        'unimodal_ftForce',\
                        #'unimodal_visionChange',\
                        'unimodal_ppsForce',\
                        'unimodal_fabricForce',\
                        'crossmodal_targetRelativeDist', \
                        'crossmodal_targetRelativeAng'
                        ]
        local_range = 0.15

        nState    = 10
        threshold = 0.0
        smooth          = False #only related with expLoglikelihood
        ## cluster_type    = 'time'
        ## cluster_type = 'state'
        cluster_type    = 'none'
        classifier_type = 'svm'
        
        evaluation([subject], task, raw_data_path, save_data_path, rf_center, local_range,\
                   nSet=target_data_set, downSampleSize=downSampleSize, \
                   feature_list=feature_list, \
                   nState=nState, threshold=threshold, smooth=smooth, cluster_type=cluster_type,\
                   classifier_type=classifier_type,\
                   hmm_renew=opt.bHMMRenew, data_renew=opt.bDataRenew, save_pdf=opt.bSavePdf, \
                   verbose=opt.bVerbose)
        


    elif opt.bTrainClassifier:
        target_data_set = 0
        rf_center    = 'kinEEPos'
        ## rf_center    = 'kinForearmPos'
        feature_list = ['unimodal_audioPower',\
                        #'unimodal_kinVel',\
                        'unimodal_ftForce',\
                        #'unimodal_visionChange',\
                        'unimodal_ppsForce',\
                        'unimodal_fabricForce',\
                        'crossmodal_targetRelativeDist', \
                        'crossmodal_targetRelativeAng'
                        ]
        local_range = 0.15

        nState    = 10
        threshold = 0.0
        smooth          = False #only related with expLoglikelihood
        ## cluster_type    = 'time'
        ## cluster_type = 'state'
        cluster_type = 'none'
        classifier_type = 'new'
        
        trainClassifierSVM([subject], task, raw_data_path, save_data_path, rf_center, local_range,\
                           nSet=target_data_set, downSampleSize=downSampleSize, \
                           feature_list=feature_list, \
                           nState=nState, threshold=threshold, smooth=smooth, cluster_type=cluster_type,\
                           classifier_type=classifier_type,\
                           useTrain=True, useNormalTest=False, useAbnormalTest=True,\
                           useTrain_color=False, useNormalTest_color=False, useAbnormalTest_color=False,\
                           hmm_renew=opt.bHMMRenew, data_renew=opt.bDataRenew, save_pdf=opt.bSavePdf)

        ## trainClassifier([subject], task, raw_data_path, save_data_path, rf_center, local_range,\
        ##                 nSet=target_data_set, downSampleSize=downSampleSize, \
        ##                 feature_list=feature_list, \
        ##                 nState=nState, threshold=threshold, smooth=smooth, cluster_type=cluster_type,\
        ##                 classifier_type=classifier_type,\
        ##                 useTrain=True, useNormalTest=False, useAbnormalTest=True,\
        ##                 useTrain_color=False, useNormalTest_color=False, useAbnormalTest_color=False,\
        ##                 hmm_renew=opt.bHMMRenew, data_renew=opt.bDataRenew, save_pdf=opt.bSavePdf)

        
    ## else:
    ##     nState         = 10
    ##     cov_mult       = 5.0       
    ##     anomaly_offset = -20.0        
    ##     check_methods = ['progress']
    ##     evaluation_all([subject], task, check_methods, feature_list, nSet,\
    ##                    save_data_path, downSampleSize=downSampleSize, \
    ##                    nState=nState, cov_mult=cov_mult, anomaly_offset=anomaly_offset, local_range=local_range,\
    ##                    data_renew=opt.bDataRenew, hmm_renew=opt.bHMMRenew, viz=viz)    

    else:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        from scipy.stats import poisson
        mu = 0.6
        mean, var, skew, kurt = poisson.stats(mu, moments='mvsk')
        x = np.arange(0.0, 30.0)
        ax.plot(x, poisson.pmf(x, mu), 'bo', ms=8, label='poisson pmf')
        
        plt.show()
