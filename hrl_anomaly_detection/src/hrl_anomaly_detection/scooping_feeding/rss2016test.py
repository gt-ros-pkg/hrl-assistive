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
import rospy, roslib
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
import PyKDL
import sandbox_dpark_darpa_m3.lib.hrl_check_util as hcu
import sandbox_dpark_darpa_m3.lib.hrl_dh_lib as hdl
import hrl_lib.circular_buffer as cb

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


def preprocessData(subject_names, task_name, raw_data_path, processed_data_path, nSet=1, \
                   folding_ratio=0.8, downSampleSize=200,\
                   raw_viz=False, interp_viz=False, renew=False, verbose=False, save_pdf=False):

    # Check if there is already scaled data
    for i in xrange(nSet):        
        target_file = os.path.join(processed_data_path, task_name+'_dataSet_'+str(i) )                    
        if os.path.isfile(target_file) is not True: renew=True
            
    if renew == False: return        

    success_list, failure_list = getSubjectFileList(raw_data_path, subject_names, task_name)

    nTrain = int(len(success_list) * folding_ratio)
    nTest  = len(success_list) - nTrain    

    if len(failure_list) < nTest: 
        print "Not enough failure data"
        sys.exit()

    # loading and time-sync
    _, data_dict = loadData(success_list, isTrainingData=False, downSampleSize=downSampleSize)
    
    ## data_min = {}
    ## data_max = {}
    ## for key in data_dict.keys():
    ##     if 'time' in key: continue
    ##     if data_dict[key] == []: continue
    ##     data_min[key] = np.min(data_dict[key])        
    ##     data_max[key] = np.max(data_dict[key])
        
    for i in xrange(nSet):

        # index selection
        success_idx  = range(len(success_list))
        failure_idx  = range(len(failure_list))
        train_idx    = random.sample(success_idx, nTrain)

        if nTest == 0: 
            success_test_idx = []
            failure_test_idx = []
        else: 
            success_test_idx = [x for x in success_idx if not x in train_idx]
            failure_test_idx = random.sample(failure_idx, nTest)

        # get training data
        trainFileList = [success_list[x] for x in train_idx]
        _, trainData = loadData(trainFileList, isTrainingData=True, \
                                downSampleSize=downSampleSize)

        # get test data
        if nTest != 0:        
            normalTestFileList = [success_list[x] for x in success_test_idx]
            _, normalTestData = loadData([success_list[x] for x in success_test_idx], 
                                                          isTrainingData=False, downSampleSize=downSampleSize)
            abnormalTestFileList = [failure_list[x] for x in failure_test_idx]
            _, abnormalTestData = loadData([failure_list[x] for x in failure_test_idx], \
                                        isTrainingData=False, downSampleSize=downSampleSize)

        # scaling data
        ## trainData_scaled = scaleData(trainData, scale=scale, data_min=data_min, 
        ##                              data_max=data_max, verbose=verbose)
        ## normalTestData_scaled = scaleData(normalTestData, scale=scale, data_min=data_min, 
        ##                                   data_max=data_max, verbose=verbose)
        ## abnormalTestData_scaled = scaleData(abnormalTestData, scale=scale, data_min=data_min, 
        ##                                     data_max=data_max, verbose=verbose)

        # cutting data (only traing and thresTest data)
        ## start_idx = int(float(len(trainData_scaled[0][0]))*train_cutting_ratio[0])
        ## end_idx   = int(float(len(trainData_scaled[0][0]))*train_cutting_ratio[1])

        ## for j in xrange(len(trainData_scaled)):
        ##     for k in xrange(len(trainData_scaled[j])):
        ##         trainData_scaled[j][k] = trainData_scaled[j][k][start_idx:end_idx]
                
        ## for j in xrange(len(normalTestData_scaled)):
        ##     for k in xrange(len(normalTestData_scaled[j])):                
        ##         normalTestData_scaled[j][k] = normalTestData_scaled[j][k][start_idx:end_idx]
                
        ## for j in xrange(len(abnormalTestData_scaled)):
        ##     for k in xrange(len(abnormalTestData_scaled[j])):                
        ##         abnormalTestData_scaled[j][k] = abnormalTestData_scaled[j][k][start_idx:end_idx]

        # Save data using dictionary
        d = {}
        d['trainData']        = trainData
        d['normalTestData']   = normalTestData
        d['abnormalTestData'] = abnormalTestData

        d['trainFileList']        = trainFileList
        d['normalTestFileList']   = normalTestFileList
        d['abnormalTestFileList'] = abnormalTestFileList        
        
        # Save data using dictionary
        target_file = os.path.join(processed_data_path, task_name+'_dataSet_'+str(i) )

        try:
            ut.save_pickle(d, target_file)        
        except:
            print "There is already target file: "
        

## def updateMinMax(param_dict, feature_name, feature_array):

##     if feature_name in param_dict.keys():
##         maxVal = np.max(feature_array)
##         minVal = np.min(feature_array)
##         if param_dict[feature_name+'_max'] < maxVal:
##             param_dict[feature_name+'_max'] = maxVal
##         if param_dict[feature_name+'_min'] > minVal:
##             param_dict[feature_name+'_min'] = minVal
##     else:
##         param_dict[feature_name+'_max'] = -100000000000
##         param_dict[feature_name+'_min'] =  100000000000        
    

def likelihoodOfSequences(subject_names, task_name, raw_data_path, processed_data_path, rf_center, local_range, \
                          nSet=1, downSampleSize=200, \
                          feature_list=['crossmodal_targetRelativeDist'], \
                          nState=10, threshold=-1.0, \
                          useTrain=True, useNormalTest=True, useAbnormalTest=False,\
                          useTrain_color=False, useNormalTest_color=False, useAbnormalTest_color=False,\
                          hmm_renew=False, data_renew=False, save_pdf=False, show_plot=True):

    _, trainingData, abnormalTestData,_ = feature_extraction(subject_names, task_name, raw_data_path, \
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

    ml  = hmm.learning_hmm_multi_n(nState, nEmissionDim, scale=10.0, verbose=False)
    ret = ml.fit(trainingData, cov_mult=cov_mult, ml_pkl=detection_param_pkl, use_pkl=False) # not(renew))
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

                exp_logp, logp = ml.expLoglikelihood(X, ths, smooth=True)
                log_ll[i].append(logp)
                exp_log_ll[i].append(exp_logp)

            if min_logp > np.amin(log_ll): min_logp = np.amin(log_ll)
            if max_logp < np.amax(log_ll): max_logp = np.amax(log_ll)
                
            # disp
            if useTrain_color:
                plt.plot(log_ll[i], label=str(i))
                ## print i, " : ", trainFileList[i], log_ll[i][-1]                
            else:
                plt.plot(log_ll[i], 'b-')


        if useTrain_color: 
            plt.legend(loc=3,prop={'size':16})
            
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

                exp_logp, logp = ml.expLoglikelihood(X, ths)
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
    if useAbnormalTest:
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



def trainClassifier(subject_names, task_name, raw_data_path, processed_data_path, rf_center, \
                             local_range, \
                             nSet=1, downSampleSize=200, \
                             feature_list=['crossmodal_targetRelativeDist'], \
                             nState=10, threshold=-1.0, \
                             useTrain=True, useNormalTest=True, useAbnormalTest=False,\
                             useTrain_color=False, useNormalTest_color=False, useAbnormalTest_color=False,\
                             hmm_renew=False, data_renew=False, save_pdf=False, show_plot=True):

    _, successData, failureData,_ = feature_extraction(subject_names, task_name, raw_data_path, \
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

    ml  = hmm.learning_hmm_multi_n(nState, nEmissionDim, scale=10.0, verbose=False)
    ret = ml.fit(trainingData, cov_mult=cov_mult, ml_pkl=detection_param_pkl, use_pkl=True) # not(renew))
    ths = [threshold]*nState

    # initialize internal classifier update parameter
    ml.opt_A   = []
    ml.opt_B   = []
    ml.opt_idx = []
    ml.opt_y   = []
    ml.opt_logp= []

    log_ll     = []
    exp_log_ll = []
    smooth = False
    c1 = 1.0
    c2 = 1.0        

    # update the classifier when new data comes in
    for i in xrange(len(normalTestData[0])):
        
        prev_exp_log_l = []
        l_prev_cost = []
        for j in range(2, len(normalTestData[0][i])):
            X = [x[i,:j] for x in normalTestData]                

            exp_logp, logp = ml.expLoglikelihood(X, ml.l_ths_mult, smooth=smooth)
            prev_exp_log_l.append(exp_logp)
            l_prev_cost.append( max(0, 1.0+(2.0*c2-1.0)*(exp_logp-logp)) )

        #----------------------------------------------------------
        X = [x[i,:] for x in normalTestData]
        y = -1
        if i>3:
            cf.updateClassifierCoff(ml, X, y, c1=c1, c2=c2, smooth=smooth)
        else:
            cf.updateClassifierCoff(ml, X, y, c1=c1, c2=c2, smooth=smooth, optimization=False)
        #----------------------------------------------------------

        log_ll.append([])
        exp_log_ll.append([])
        l_cost = []
        for j in range(2, len(normalTestData[0][i])):
            X = [x[i,:j] for x in normalTestData]                

            exp_logp, logp = ml.expLoglikelihood(X, ml.l_ths_mult, smooth=smooth)
            log_ll[i].append(logp)
            exp_log_ll[i].append(exp_logp)
            l_cost.append( max(0, 1.0+(2.0*c2-1.0)*(exp_logp-logp)) )

        if len(log_ll)>4:
            fig = plt.figure()
            ax = fig.add_subplot(211)

            for j in xrange(len(log_ll)):
                plt.plot(log_ll[j], 'b-')
            plt.plot(log_ll[-1], 'b-', lw=3.0)
            plt.plot(prev_exp_log_l, 'm*--', lw=3.0)            
            plt.plot(exp_log_ll[-1], 'm-', lw=3.0)            

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


    for i in xrange(len(abnormalTestData[0])):
        
        prev_exp_log_l = []
        for j in range(2, len(abnormalTestData[0][i])):
            X = [x[i,:j] for x in abnormalTestData]                

            exp_logp, _ = ml.expLoglikelihood(X, ml.l_ths_mult, smooth=smooth)
            prev_exp_log_l.append(exp_logp)

        #----------------------------------------------------------
        X      = [x[i,:] for x in abnormalTestData]
        y      = [1]
        cf.updateClassifierCoff(ml, X, y, c1=c1, c2=c2, smooth=smooth)
        #----------------------------------------------------------
        # visualization

        log_ll.append([])
        exp_log_ll.append([])
        for j in range(2, len(abnormalTestData[0][i])):
            X = [x[i,:j] for x in abnormalTestData]                

            exp_logp, logp = ml.expLoglikelihood(X, ml.l_ths_mult, smooth=smooth)
            log_ll[-1].append(logp)
            exp_log_ll[-1].append(exp_logp)


        fig = plt.figure()
        ax = fig.add_subplot(211)

        for j in xrange(len(log_ll)):
            if j < len(normalTestData[0]):
                plt.plot(log_ll[j], 'b-')
            else:
                plt.plot(log_ll[j], 'r-')
        plt.plot(log_ll[-1], 'r-', lw=3.0)
        plt.plot(prev_exp_log_l, 'm*--', lw=3.0)            
        plt.plot(exp_log_ll[-1], 'm-', lw=3.0)            

        if save_pdf == True:
            fig.savefig('test.pdf')
            fig.savefig('test.png')
            os.system('cp test.p* ~/Dropbox/HRL/')
        else:
            plt.show()        

    


def evaluation_all(subject_names, task_name, check_methods, feature_list, nSet, \
                   processed_data_path, downSampleSize=100, \
                   nState=10, cov_mult=1.0, anomaly_offset=0.0, local_range=0.25,\
                   data_renew=False, hmm_renew=False, save_pdf=False, viz=False):

    # For parallel computing
    strMachine = socket.gethostname()+"_"+str(os.getpid())    

    count = 0
    for method in check_methods:        

        # Check the existance of workspace
        method_path = os.path.join(processed_data_path, task_name, method)
        if os.path.isdir(method_path) == False:
            os.system('mkdir -p '+method_path)

        for idx, subject_name in enumerate(subject_names):

            ## For parallel computing
            # save file name
            res_file        = task_name+'_'+subject_name+'_'+method+'.pkl'
            mutex_file_part = 'running_'+task_name+'_'+subject_name+'_'+method

            res_file        = os.path.join(method_path, res_file)
            mutex_file_full = mutex_file_part+'_'+strMachine+'.txt'
            mutex_file      = os.path.join(method_path, mutex_file_full)

            if os.path.isfile(res_file): 
                count += 1            
                continue
            elif hcu.is_file(method_path, mutex_file_part) and \
              not hcu.is_file(method_path, mutex_file_part+'_'+socket.gethostname() ): 
                print "Mutex file exists"
                continue
            ## elif os.path.isfile(mutex_file): continue
            os.system('touch '+mutex_file)

            preprocessData(subject_names, task_name, processed_data_path, processed_data_path, \
                           renew=data_renew, downSampleSize=downSampleSize)

            (truePos, falseNeg, trueNeg, falsePos)\
              = evaluation(task_name, processed_data_path, nSet=nSet, nState=nState, cov_mult=cov_mult,\
                           anomaly_offset=anomaly_offset, check_method=method,\
                           hmm_renew=hmm_renew, viz=False, verbose=True)


            truePositiveRate = float(truePos) / float(truePos + falseNeg) * 100.0
            if trueNeg == 0 and falsePos == 0:            
                trueNegativeRate = "Not available"
            else:
                trueNegativeRate = float(trueNeg) / float(trueNeg + falsePos) * 100.0
                
            print 'True Negative Rate:', trueNegativeRate, 'True Positive Rate:', truePositiveRate
                           
            if truePos!=-1 :                 
                d = {}
                d['subject'] = subject_name
                d['tp'] = truePos
                d['fn'] = falseNeg
                d['tn'] = trueNeg
                d['fp'] = falsePos
                d['nSet'] = nSet

                try:
                    ut.save_pickle(d,res_file)        
                except:
                    print "There is already the targeted pkl file"
            else:
                target_file = os.path.join(method_path, task_name+'_dataSet_%d_eval_'+str(idx) ) 
                for j in xrange(nSet):
                    os.system('rm '+target_file % j)
                

            os.system('rm '+mutex_file)

            if truePos==-1: 
                print "truePos is -1"
                sys.exit()

    if count == len(check_methods)*len(subject_names):
        print "#############################################################################"
        print "All file exist ", count
        print "#############################################################################"        
    else:
        return
                

def evaluation(task_name, processed_data_path, nSet=1, nState=20, cov_mult=5.0, anomaly_offset=0.0,\
               check_method='progress', hmm_renew=False, save_pdf=False, viz=False, verbose=False):

    tot_truePos = 0
    tot_falseNeg = 0
    tot_trueNeg = 0 
    tot_falsePos = 0

    for i in xrange(nSet):        
        target_file = os.path.join(processed_data_path, task_name+'_dataSet_'+str(i) )                    
        if os.path.isfile(target_file) is not True: 
            print "There is no saved data"
            sys.exit()

        data_dict = ut.load_pickle(target_file)
        if viz: visualization_raw_data(data_dict, save_pdf=save_pdf)

        # training set
        trainingData, param_dict = extractLocalFeature(data_dict['trainData'], feature_list, local_range)

        # test set
        normalTestData, _ = extractLocalFeature(data_dict['normalTestData'], feature_list, local_range, \
                                                param_dict=param_dict)        
        abnormalTestData, _ = extractLocalFeature(data_dict['abnormalTestData'], feature_list, local_range, \
                                                param_dict=param_dict)

        print "======================================"
        print "Training data: ", np.shape(trainingData)
        print "Normal test data: ", np.shape(normalTestData)
        print "Abnormal test data: ", np.shape(abnormalTestData)
        print "======================================"

        if True: visualization_hmm_data(feature_list, trainingData=trainingData, \
                                        normalTestData=normalTestData,\
                                        abnormalTestData=abnormalTestData, save_pdf=save_pdf)        

        # training hmm
        nEmissionDim = len(trainingData)
        detection_param_pkl = os.path.join(processed_data_path, 'hmm_'+task_name+'.pkl')

        ml = hmm.learning_hmm_multi_n(nState, nEmissionDim, scale=1000.0, verbose=True)

        print "Start to fit hmm", np.shape(trainingData)
        ret = ml.fit(trainingData, cov_mult=[cov_mult]*nEmissionDim**2, ml_pkl=detection_param_pkl, \
                     use_pkl=hmm_renew)

        if ret == 'Failure': 
            print "-------------------------"
            print "HMM returned failure!!   "
            print "-------------------------"
            return (-1,-1,-1,-1)


        ## minThresholds = None                  
        ## if hmm_renew:
        ##     minThresholds1 = tuneSensitivityGain(ml, trainingData, method=check_method, verbose=verbose)
        ##     ## minThresholds2 = tuneSensitivityGain(ml, thresTestData, method=check_method, verbose=verbose)
        ##     minThresholds = minThresholds1

        ##     if type(minThresholds) == list or type(minThresholds) == np.ndarray:
        ##         for i in xrange(len(minThresholds1)):
        ##             if minThresholds1[i] < minThresholds2[i]:
        ##                 minThresholds[i] = minThresholds1[i]
        ##     else:
        ##         if minThresholds1 < minThresholds2:
        ##             minThresholds = minThresholds1

        ##     d = ut.load_pickle(detection_param_pkl)
        ##     if d is None: d = {}
        ##     d['minThresholds'] = minThresholds                
        ##     ut.save_pickle(d, detection_param_pkl)                
        ## else:
        ##     d = ut.load_pickle(detection_param_pkl)
        ##     minThresholds = d['minThresholds']
        minThresholds=-5.0

        truePos, falseNeg, trueNeg, falsePos = \
          onlineEvaluation(ml, normalTestData, abnormalTestData, c=minThresholds, verbose=True)
        if truePos == -1: 
            print "Error with task ", task_name
            print "Error with nSet ", i
            print "Error with crossEval ID: ", crossEvalID
            return (-1,-1,-1,-1)

        tot_truePos += truePos
        tot_falseNeg += falseNeg
        tot_trueNeg += trueNeg 
        tot_falsePos += falsePos
            
    truePositiveRate = float(tot_truePos) / float(tot_truePos + tot_falseNeg) * 100.0
    if tot_trueNeg == 0 and tot_falsePos == 0:
        trueNegativeRate = "not available"
    else:
        trueNegativeRate = float(tot_trueNeg) / float(tot_trueNeg + tot_falsePos) * 100.0
    print "------------------------------------------------"
    print "Total set of data: ", nSet
    print "------------------------------------------------"
    print 'True Negative Rate:', trueNegativeRate, 'True Positive Rate:', truePositiveRate
    print "------------------------------------------------"

    return (tot_truePos, tot_falseNeg, tot_trueNeg, tot_falsePos)
        
        ## tp_l = []
        ## fn_l = []
        ## fp_l = []
        ## tn_l = []
        ## ths_l = []

        ## # evaluation
        ## threshold_list = -(np.logspace(-1.0, 1.5, nThres, endpoint=True)-1.0 )        
        ## ## threshold_list = [-5.0]
        ## for ths in threshold_list:        
        ##     tp, fn, tn, fp = onlineEvaluation(ml, normalTestData, abnormalTestData, c=ths, 
        ##                                       verbose=True)
        ##     if tp == -1:
        ##         tp_l.append(0)
        ##         fn_l.append(0)
        ##         fp_l.append(0)
        ##         tn_l.append(0)
        ##         ths_l.append(ths)
        ##     else:                       
        ##         tp_l.append(tp)
        ##         fn_l.append(fn)
        ##         fp_l.append(fp)
        ##         tn_l.append(tn)
        ##         ths_l.append(ths)

        ## dd = {}
        ## dd['fn_l']    = fn_l
        ## dd['tn_l']    = tn_l
        ## dd['tp_l']    = tp_l
        ## dd['fp_l']    = fp_l
        ## dd['ths_l']   = ths_l

        ## try:
        ##     ut.save_pickle(dd,res_file)        
        ## except:
        ##     print "There is the targeted pkl file"

    
    
                
def onlineEvaluation(hmm, normalTestData, abnormalTestData, c=-5, verbose=False):
    truePos = 0
    trueNeg = 0
    falsePos = 0
    falseNeg = 0

    # positive is anomaly
    # negative is non-anomaly
    if verbose: print '\nBeginning anomaly testing for test set\n'

    # for normal test data
    if normalTestData != []:    
        for i in xrange(len(normalTestData[0])):
            if verbose: print 'Anomaly Error for test set ', i

            for j in range(20, len(normalTestData[0][i])):

                try:    
                    anomaly, error = hmm.anomaly_check(normalTestData[:][i][:j], c)
                except:
                    print "anomaly_check failed: ", i, j
                    ## return (-1,-1,-1,-1)
                    falsePos += 1
                    break

                if np.isnan(error):
                    print "anomaly check returned nan"
                    falsePos += 1
                    break
                    ## return (-1,-1,-1,-1)

                if verbose: print "Normal: ", j, " => ", anomaly, error

                # This is a successful nonanomalous attempt
                if anomaly:
                    falsePos += 1
                    if verbose: print 'Success Test', i,',',j, ' in ',len(normalTestData[0][i]), ' |', anomaly, 
                    error
                    break
                elif j == len(normalTestData[0][i]) - 1:
                    trueNeg += 1
                    break


    # for abnormal test data
    for i in xrange(len(abnormalTestData[0])):
        if verbose: print 'Anomaly Error for test set ', i

        for j in range(20, len(abnormalTestData[0][i])):
            try:                    
                anomaly, error = hmm.anomaly_check(abnormalTestData[:][i][:j], c)
            except:
                truePos += 1
                break

            if verbose: print anomaly, error
                
            if anomaly:
                truePos += 1
                break
            elif j == len(abnormalTestData[0][i]) - 1:
                falseNeg += 1
                if verbose: print 'Failure Test', i,',',j, ' in ',len(abnormalTestData[0][i]), ' |', anomaly, error
                break

    return truePos, falseNeg, trueNeg, falsePos

        
def data_plot(subject_names, task_name, raw_data_path, processed_data_path, \
              nSet=1, downSampleSize=200, \
              local_range=0.3, rf_center='kinEEPos', \
              success_viz=True, failure_viz=False, \
              raw_viz=False, interp_viz=False, save_pdf=False, \
              successData=False, failureData=True,\
              vizDataIdx=None,\
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
                                                       
        count       = 0
        nPlot       = len(modality_list)
        time_lim    = [0, 16] #?????????????????????????????
   
        if raw_viz: target_dict = raw_data_dict
        else: target_dict = interp_data_dict

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
                    if vizDataIdx is not None:
                        if i == vizDataIdx: new_color = 'm'
                        else: new_color = color
                    else:
                        new_color = color
                    
                    if len(time_list[i]) > len(data_list[i]):
                        ax.plot(time_list[i][:len(data_list[i])], data_list[i], c=new_color)
                    else:
                        ax.plot(time_list[i], data_list[i][:len(time_list[i])], c=new_color)                    
            else:
                interp_time = np.linspace(time_lim[0], time_lim[1], num=downSampleSize)
                for i in xrange(len(data_list)):
                    ax.plot(interp_time, data_list[i], c=color)                

            ax.set_xlim(time_lim)
            ax.set_title(modality)
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

    success_list, failure_list = getSubjectFileList(raw_data_path, subject_names, task_name)

    # Success data
    successData = True
    failureData = False

    for i in xrange(len(success_list)):

        print "-----------------------------------------------"
        print success_list[i]
        print "-----------------------------------------------"
        data_plot(subject_names, task_name, raw_data_path, processed_data_path,\
                  nSet=nSet, downSampleSize=downSampleSize, \
                  local_range=local_range, rf_center=rf_center, \
                  raw_viz=True, interp_viz=False, save_pdf=save_pdf,\
                  successData=successData, failureData=failureData,\
                  vizDataIdx=i,\
                  modality_list=modality_list, data_renew=data_renew, verbose=verbose)

        ## feedback  = raw_input('Do you want to exclude the data? (e.g. y:yes n:no else: exit): ')
        ## if feedback == 'y':
        ##     print "move data"
        ##     ## os.system('mv '+subject_names+' ')
        ## elif feedback == 'n':
        ##     print "keep data"
        ## else:
        ##     break




    

    
def feature_extraction(subject_names, task_name, raw_data_path, processed_data_path, rf_center, local_range, \
             nSet=1, downSampleSize=200, success_viz=False, failure_viz=False, \
             save_pdf=False, solid_color=True, \
             feature_list=['crossmodal_targetRelativeDist'], data_renew=False):

    save_pkl = os.path.join(processed_data_path, 'pca_'+rf_center+'_'+str(local_range) )
    if os.path.isfile(save_pkl) and data_renew is not True :
        data_dict = ut.load_pickle(save_pkl)
        allData          = data_dict['allData']
        trainingData     = data_dict['trainingData'] 
        abnormalTestData = data_dict['abnormalTestData']
        abnormalTestNameList = data_dict['abnormalTestNameList']
        param_dict       = data_dict['param_dict']
    else:
        ## data_renew = False #temp
        
        success_list, failure_list = getSubjectFileList(raw_data_path, subject_names, task_name)

        # loading and time-sync    
        all_data_pkl     = os.path.join(processed_data_path, subject+'_'+task+'_all_'+rf_center+\
                                        '_'+str(local_range))
        _, all_data_dict = loadData(success_list+failure_list, isTrainingData=False,
                                    downSampleSize=downSampleSize,\
                                    local_range=local_range, rf_center=rf_center,\
                                    ##global_data=True,\
                                    renew=data_renew, save_pkl=all_data_pkl)

        success_data_pkl     = os.path.join(processed_data_path, subject+'_'+task+'_success_'+rf_center+\
                                            '_'+str(local_range))
        _, success_data_dict = loadData(success_list, isTrainingData=True,
                                        downSampleSize=downSampleSize,\
                                        local_range=local_range, rf_center=rf_center,\
                                        renew=data_renew, save_pkl=success_data_pkl)

        failure_data_pkl     = os.path.join(processed_data_path, subject+'_'+task+'_failure_'+rf_center+\
                                            '_'+str(local_range))
        _, failure_data_dict = loadData(failure_list, isTrainingData=False,
                                        downSampleSize=downSampleSize,\
                                        local_range=local_range, rf_center=rf_center,\
                                        renew=data_renew, save_pkl=failure_data_pkl)

        # data set
        allData, param_dict = extractLocalFeature(all_data_dict, feature_list)
        trainingData, _     = extractLocalFeature(success_data_dict, feature_list, param_dict=param_dict)
        abnormalTestData, _ = extractLocalFeature(failure_data_dict, feature_list, param_dict=param_dict)

        allData          = np.array(allData)
        trainingData     = np.array(trainingData)
        abnormalTestData = np.array(abnormalTestData)

        data_dict = {}
        data_dict['allData'] = allData
        data_dict['trainingData'] = trainingData
        data_dict['abnormalTestData'] = abnormalTestData
        data_dict['abnormalTestNameList'] = abnormalTestNameList = failure_data_dict['fileNameList']
        data_dict['param_dict'] = param_dict
        ut.save_pickle(data_dict, save_pkl)


    ## # test
    ## success_list, failure_list = getSubjectFileList(raw_data_path, subject_names, task_name)
    ## _, success_data_dict = loadData(success_list, isTrainingData=True,
    ##                                 downSampleSize=downSampleSize,\
    ##                                 local_range=local_range, rf_center=rf_center)
    ## trainingData, _      = extractLocalFeature(success_data_dict, feature_list, \
    ##                                            param_dict=data_dict['param_dict'])
    ## sys.exit()
    
    ## All data
    nPlot = None
    feature_names = np.array(param_dict['feature_names'])

    if True:

        # 1) exclude stationary data
        thres = 0.025
        n,m,k = np.shape(trainingData)
        diff_all_data = trainingData[:,:,1:] - trainingData[:,:,:-1]
        add_idx    = []
        remove_idx = []
        std_list = []
        for i in xrange(n):
            std = np.max(np.max(diff_all_data[i], axis=1))
            std_list.append(std)
            if  std < thres: remove_idx.append(i)
            else: add_idx.append(i)

        allData          = allData[add_idx]
        trainingData     = trainingData[add_idx]
        abnormalTestData = abnormalTestData[add_idx]
        AddFeature_names    = feature_names[add_idx]
        RemoveFeature_names = feature_names[remove_idx]

        print "--------------------------------"
        print "STD list: ", std_list
        print "Add features: ", AddFeature_names
        print "Remove features: ", RemoveFeature_names
        print "--------------------------------"
        ## sys.exit()


    # -------------------- Display ---------------------
    fig = None
    if success_viz:
        fig = plt.figure()
        n,m,k = np.shape(trainingData)
        if nPlot is None:
            if n%2==0: nPlot = n
            else: nPlot = n+1

        for i in xrange(n):
            ax = fig.add_subplot((nPlot/2)*100+20+i)
            if solid_color: ax.plot(trainingData[i].T, c='b')
            else: ax.plot(trainingData[i].T)
            ax.set_title( AddFeature_names[i] )

    if failure_viz:
        if fig is None: fig = plt.figure()
        n,m,k = np.shape(abnormalTestData)
        if nPlot is None:
            if n%2==0: nPlot = n
            else: nPlot = n+1

        for i in xrange(n):
            ax = fig.add_subplot((nPlot/2)*100+20+i)
            if solid_color: ax.plot(abnormalTestData[i].T, c='r')
            else: ax.plot(abnormalTestData[i].T)
            ax.set_title( AddFeature_names[i] )

    if success_viz or failure_viz:
        plt.tight_layout(pad=3.0, w_pad=0.5, h_pad=0.5)

        if save_pdf:
            fig.savefig('test.pdf')
            fig.savefig('test.png')
            os.system('cp test.p* ~/Dropbox/HRL/')        
        else:
            plt.show()

    print "---------------------------------------------------"
    print np.shape(trainingData), np.shape(abnormalTestData)
    print "---------------------------------------------------"

    return allData, trainingData, abnormalTestData, abnormalTestNameList


def pca_plot(subject_names, task_name, raw_data_path, processed_data_path, rf_center, local_range, \
             nSet=1, downSampleSize=200, success_viz=True, failure_viz=False, \
             save_pdf=False, \
             feature_list=['crossmodal_targetRelativeDist'], data_renew=False):


    allData, trainingData, abnormalTestData, abnormalTestNameList\
      = feature_extraction(subject_names, task_name, raw_data_path, \
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

    p.add_option('--likelihoodplot', '--lp', action='store_true', dest='bLikelihoodPlot',
                 default=False, help='Plot the change of likelihood.')
    p.add_option('--localization', '--ll', action='store_true', dest='bLocalization',
                 default=False, help='Extract local feature.')
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
    p.add_option('--trainClassifier', '--tc', action='store_true', dest='bTrainClassifier',
                 default=False, help='Train a cost sensitive classifier.')
    
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
                        #'unimodal_visionChange',\
                        'unimodal_ppsForce',\
                        'unimodal_fabricForce',\
                        'crossmodal_targetRelativeDist', \
                        'crossmodal_targetRelativeAng']
        local_range = 0.15
        success_viz = True
        failure_viz = True

        feature_extraction([subject], task, raw_data_path, save_data_path, rf_center, local_range,\
                           nSet=target_data_set, downSampleSize=downSampleSize, \
                           success_viz=success_viz, failure_viz=failure_viz,\
                           save_pdf=opt.bSavePdf, solid_color=True,\
                           feature_list=feature_list, data_renew=opt.bDataRenew)

    elif opt.bPCAPlot:
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
        ## preprocessData([subject], task, raw_data_path, save_data_path, renew=opt.bDataRenew, \
        ##                downSampleSize=downSampleSize)
        likelihoodOfSequences([subject], task, raw_data_path, save_data_path, rf_center, local_range,\
                              nSet=target_data_set, downSampleSize=downSampleSize, \
                              feature_list=feature_list, \
                              nState=nState, threshold=threshold,\
                              useTrain=True, useNormalTest=False, useAbnormalTest=True,\
                              useTrain_color=False, useNormalTest_color=False, useAbnormalTest_color=False,\
                              hmm_renew=opt.bHMMRenew, data_renew=opt.bDataRenew, save_pdf=opt.bSavePdf)
                              

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
        trainClassifier([subject], task, raw_data_path, save_data_path, rf_center, local_range,\
                        nSet=target_data_set, downSampleSize=downSampleSize, \
                        feature_list=feature_list, \
                        nState=nState, threshold=threshold,\
                        useTrain=True, useNormalTest=False, useAbnormalTest=True,\
                        useTrain_color=False, useNormalTest_color=False, useAbnormalTest_color=False,\
                        hmm_renew=opt.bHMMRenew, data_renew=opt.bDataRenew, save_pdf=opt.bSavePdf)

        
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
