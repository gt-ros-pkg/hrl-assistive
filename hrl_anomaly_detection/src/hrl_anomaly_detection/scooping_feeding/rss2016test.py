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
## matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec

# util
import numpy as np
import hrl_lib.util as ut
from hrl_anomaly_detection.util import *
import PyKDL
import sandbox_dpark_darpa_m3.lib.hrl_check_util as hcu

# learning
from hrl_anomaly_detection.hmm import learning_hmm_multi_n as hmm




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
    
    data_min = {}
    data_max = {}
    for key in data_dict.keys():
        if 'time' in key: continue
        if data_dict[key] == []: continue
        data_min[key] = np.min(data_dict[key])        
        data_max[key] = np.max(data_dict[key])
        
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
        
        

        


def updateMinMax(param_dict, feature_name, feature_array):

    if feature_name in param_dict.keys():
        maxVal = np.max(feature_array)
        minVal = np.min(feature_array)
        if param_dict[feature_name+'_max'] < maxVal:
            param_dict[feature_name+'_max'] = maxVal
        if param_dict[feature_name+'_min'] > minVal:
            param_dict[feature_name+'_min'] = minVal
    else:
        param_dict[feature_name+'_max'] = -100000000000
        param_dict[feature_name+'_min'] =  100000000000
        
    

def likelihoodOfSequences(processed_data_path, task_name, feature_list, local_range, \
                          nSet=0, nState=10, threshold=-1.0, \
                          useTrain=True, useNormalTest=True, useAbnormalTest=False,\
                          useTrain_color=False, useNormalTest_color=False, useAbnormalTest_color=False,\
                          renew=False, save_pdf=False, show_plot=True):

    target_file = os.path.join(processed_data_path, task_name+'_dataSet_'+str(nSet) )                    
    if os.path.isfile(target_file) is not True: 
        print "There is no saved data"
        sys.exit()

    data_dict = ut.load_pickle(target_file)

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

    # training hmm
    nEmissionDim = len(trainingData)
    detection_param_pkl = os.path.join(processed_data_path, 'hmm_'+task_name+'.pkl')

    ml  = hmm.learning_hmm_multi_n(nState, nEmissionDim, verbose=False)
    ret = ml.fit(trainingData, ml_pkl=detection_param_pkl, use_pkl=not(renew))
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
        count = 0
        for i in xrange(len(trainingData[0])):

            log_ll.append([])
            exp_log_ll.append([])
            for j in range(2, len(trainingData[0][i])):

                X = [x[i,:j] for x in trainingData]                
                X_test = ml.convert_sequence(X)
                try:
                    logp = ml.loglikelihood(X_test)
                except:
                    print "Too different input profile that cannot be expressed by emission matrix"
                    return [], 0.0 # error

                log_ll[i].append(logp)

            if min_logp > np.amin(log_ll): min_logp = np.amin(log_ll)
            if max_logp < np.amax(log_ll): max_logp = np.amax(log_ll)
                
            # disp
            if useTrain_color:
                plt.plot(log_ll[i], label=str(i))
                print i, " : ", trainFileList[i], log_ll[i][-1]                
            else:
                plt.plot(log_ll[i], 'b-')

        if useTrain_color: 
            plt.legend(loc=3,prop={'size':16})
            
        ## plt.plot(exp_log_ll[i], 'r-')            
                                             
    # normal test data
    if useNormalTest:

        log_ll = []
        exp_log_ll = []        
        count = 0
        for i in xrange(len(normalTestData[0])):

            log_ll.append([])
            exp_log_ll.append([])

            for j in range(2, len(normalTestData[0][i])):
                X = [x[i,:j] for x in normalTestData]                
                X_test = ml.convert_sequence(X)
                try:
                    logp = ml.loglikelihood(X_test)
                except:
                    print "Too different input profile that cannot be expressed by emission matrix"
                    return [], 0.0 # error

                log_ll[i].append(logp)

                ## exp_logp = ml.expLikelihoods(X_test, ths)
                exp_logp = ml.expLikelihoods(X, ths)
                exp_log_ll[i].append(exp_logp)

            if min_logp > np.amin(log_ll): min_logp = np.amin(log_ll)
            if max_logp < np.amax(log_ll): max_logp = np.amax(log_ll)

            # disp 
            if useNormalTest_color:
                print i, " : ", normalTestFileList[i]                
                plt.plot(log_ll[i], label=str(i))
            else:
                plt.plot(log_ll[i], 'g-')

            plt.plot(exp_log_ll[i], 'r*-')


        if useNormalTest_color: 
            plt.legend(loc=3,prop={'size':16})

    # abnormal test data
    if useAbnormalTest:
        log_ll = []
        exp_log_ll = []        
        count = 0
        for i in xrange(len(abnormalTestData[0])):

            log_ll.append([])
            exp_log_ll.append([])

            for j in range(2, len(abnormalTestData[0][i])):
                X = [x[i,:j] for x in abnormalTestData]                
                X_test = ml.convert_sequence(X)
                try:
                    logp = ml.loglikelihood(X_test)
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
            print "-----------------------------------------------"

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

        ml = hmm.learning_hmm_multi_n(nState, nEmissionDim, verbose=True)

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
              nSet=1, downSampleSize=200, success_viz=True, failure_viz=False, \
              raw_viz=False, interp_viz=False, save_pdf=False, \
              ## trainingData=True, normalTestData=False, abnormalTestData=False,\
              modality_list=['kinEEPosList', 'audioPowerList'], data_renew=False):    


    success_list, failure_list = getSubjectFileList(raw_data_path, subject_names, task_name)
    
    # loading and time-sync
    success_data_pkl = os.path.join(processed_data_path, subject+'_'+task+'_success')
    raw_data_dict, interp_data_dict = loadData(success_list, isTrainingData=False,
                                               downSampleSize=downSampleSize,\
                                               renew=data_renew, save_pkl=success_data_pkl)

    if raw_viz: target_dict = raw_data_dict
    else: target_dict = data_dict
            
    ## target_file = os.path.join(processed_data_path, task_name+'_dataSet_'+str(nSet) )                    
    ## if os.path.isfile(target_file) is not True: 
    ##     print "There is no saved data"
    ##     sys.exit()
    ## data_dict = ut.load_pickle(target_file)
    count    = 0
    nPlot = len(modality_list)
    
    fig = plt.figure()

    if raw_viz:
        for modality in modality_list:
            count +=1
            
            if 'audio' in modality:
                time_list = target_dict['audioTimesList']
                data_list = target_dict['audioPowerList']

            if 'kinematics' in modality:
                time_list = target_dict['kinTimesList']
                data_list = target_dict['kinEEPosList']

                # distance
                new_data_list = []
                for d in data_list:
                    new_data_list.append( np.linalg.norm(d, axis=0) )
                data_list = new_data_list

            combined_time_list = []
            for t in time_list:
                temp = np.array(t[1:])-np.array(t[:-1])
                combined_time_list += ([0.0]  + list(temp) )
                
            print modality, " : ", np.mean(combined_time_list), np.std(combined_time_list),\
              np.max(combined_time_list), len(combined_time_list)            


            ax = fig.add_subplot(nPlot*100+10+count)
            ## ax.plot(combined_time_list, label=modality)

            for i in xrange(len(time_list)):

                if len(time_list[i]) > len(data_list[i]):
                    ax.plot(time_list[i][:len(data_list[i])], data_list[i])
                else:
                    ax.plot(time_list[i], data_list[i][:len(time_list[i])])
                            
    else:
        
        count    = 0
                
            
    if save_pdf is False:
        plt.show()
    else:
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
    
            

def visualization_hmm_data(feature_list, trainingData=None, normalTestData=None, abnormalTestData=None, save_pdf=False):

    if trainingData is not None:
        nDimension = len(trainingData)
    elif normalTestData is not None:
        nDimension = len(normalTestData)
    elif abnormalTestData is not None:
        nDimension = len(abnormalTestData)
    else:
        print "no data"
        sys.exit()
        
    fig = plt.figure()            
    # --------------------------------------------------
    for i in xrange(nDimension):
        ax = fig.add_subplot(100*nDimension+10+(i+1))
        if trainingData is not None:
            ax.plot(np.array(trainingData[i]).T, 'b')
        ## elif normalTestData is not None:
        ##     ax.plot(np.array(normalTestData[i]).T, 'k')
        ## elif abnormalTestData is not None:
        ##     ax.plot(abnormalTestData[i], 'r')

        ax.set_title(feature_list[i])

    if save_pdf:
        fig.savefig('test.pdf')
        fig.savefig('test.png')
        os.system('cp test.p* ~/Dropbox/HRL/')        
    else:
        plt.show()
    #sys.exit()


def visualization_raw_data(data_dict, modality='ft', save_pdf=False):

    data_key = 'trainData'
    file_key = 'trainFileList'
    ## data_key = 'normalTestData'
    ## file_key = 'normalTestFileList'
    for key in data_dict[data_key].keys():

        if not('Pos' in key): continue
        
        print "key: ", key
        
        dataList = data_dict[data_key][key]
        fileList = data_dict[file_key]
        
        if len(np.shape(dataList)) < 3: continue
        nSample, nDim, k = np.shape(dataList)

        fig = plt.figure()            
        for i in xrange(nDim):
            ax = fig.add_subplot(nDim*100+10+i)

            for j in xrange(nSample):
                fileName = fileList[j].split('/')[-1] 
                ax.plot(dataList[j][i,:], label=fileName)

        ## plt.legend(loc=3,prop={'size':8})
        ## ax1.set_ylim([0.0, 2.0])
        if save_pdf:
            fig.savefig('test'+str(count)+'.pdf')
            fig.savefig('test'+str(count)+'.png')
            os.system('cp test'+str(count)+'.p* ~/Dropbox/HRL/')        
        else:
            plt.show()


        ## count = 0
        ## d_list = []
        ## f_list = []

        ## for idx, data in enumerate(dataList):

        ##     d_list.append( np.linalg.norm(data, axis=0) )
        ##     f_list.append( fileList[idx].split('/')[-1] )

        ##     if idx%10 == 9:

        ##         fig = plt.figure()            
        ##         ax1 = fig.add_subplot(111)

        ##         for j, d in enumerate(d_list):
        ##             ax1.plot(d_list[j], label=f_list[j])

        ##         plt.legend(loc=3,prop={'size':8})
        ##         ## ax1.set_ylim([0.0, 2.0])

        ##         if save_pdf:
        ##             fig.savefig('test'+str(count)+'.pdf')
        ##             fig.savefig('test'+str(count)+'.png')
        ##             os.system('cp test'+str(count)+'.p* ~/Dropbox/HRL/')        
        ##         else:
        ##             plt.show()

        ##         d_list = []
        ##         f_list = []
        ##         count += 1

            

if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    p.add_option('--dataRenew', '--dr', action='store_true', dest='bDataRenew',
                 default=False, help='Renew pickle files.')
    p.add_option('--hmmRenew', '--hr', action='store_true', dest='bHMMRenew',
                 default=False, help='Renew HMM parameters.')

    p.add_option('--likelihoodplot', '--lp', action='store_true', dest='bLikelihoodPlot',
                 default=False, help='Plot the change of likelihood.')
    p.add_option('--rawplot', '--rp', action='store_true', dest='bRawDataPlot',
                 default=False, help='Plot raw data.')
    p.add_option('--interplot', '--ip', action='store_true', dest='bInterpDataPlot',
                 default=False, help='Plot raw data.')
    
    p.add_option('--renew', action='store_true', dest='bRenew',
                 default=False, help='Renew pickle files.')
    p.add_option('--savepdf', '--sp', action='store_true', dest='bSavePdf',
                 default=False, help='Save pdf files.')    

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
    

    # Dectection TEST 
    nSet         = 1
    local_range  = 0.25    
    viz          = False
    renew        = False
    downSampleSize=100

    if opt.bLikelihoodPlot:
        nState    = 15
        threshold = 0.0
        preprocessData([subject], task, raw_data_path, save_data_path, renew=opt.bDataRenew, \
                       downSampleSize=downSampleSize)
        likelihoodOfSequences(save_data_path, task, feature_list, local_range, \
                              nState=nState, threshold=threshold,\
                              useTrain=True, useNormalTest=False, useAbnormalTest=True,\
                              useTrain_color=False, useNormalTest_color=False, useAbnormalTest_color=False,\
                              renew=renew, save_pdf=opt.bSavePdf)
                              
    elif opt.bRawDataPlot or opt.bInterpDataPlot:
        target_data_set = 0
        modality_list   = ['kinematics', 'audio']
        
        data_plot([subject], task, raw_data_path, save_data_path,\
                  nSet=target_data_set, downSampleSize=downSampleSize, \
                  raw_viz=opt.bRawDataPlot, interp_viz=opt.bInterpDataPlot, save_pdf=opt.bSavePdf,
                  modality_list=modality_list, data_renew=opt.bDataRenew)
                              
    else:
        nState         = 10
        cov_mult       = 5.0       
        anomaly_offset = -20.0        
        check_methods = ['progress']
        evaluation_all([subject], task, check_methods, feature_list, nSet,\
                       save_data_path, downSampleSize=downSampleSize, \
                       nState=nState, cov_mult=cov_mult, anomaly_offset=anomaly_offset, local_range=local_range,\
                       data_renew=opt.bDataRenew, hmm_renew=opt.bHMMRenew, viz=viz)
