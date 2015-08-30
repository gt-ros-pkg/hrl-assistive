#!/usr/bin/python

import sys, os, copy
import numpy as np, math
import glob
import socket
import time
import random 

import roslib; roslib.load_manifest('hrl_multimodal_anomaly_detection')
import rospy

# Util
import hrl_lib.util as ut

#
from util import *
from learning_hmm_multi_4d import *

def distributionOfSequences(task_name, target_path, setID=0, scale=1.0,\
                            useTrain=True, useThsTest=False, useNormalTest=False, useAbnormalTest=False, \
                            useTrain_color=True,
                            save_pdf=False, verbose=False):

    # get data
    trainData, thresTestData, normalTestData, abnormalTestData, \
      trainTimeList, thresTestTimeList, normalTestTimeList, abnormalTestTimeList, \
      trainFileList, thsTestFileList, normalTestFileList, abnormalTestFileList \
    = getData(task_name, target_path, setID)

    fig = plt.figure()
    ax1 = plt.subplot(412)
    ax1.set_ylabel('Force\nMagnitude (N)', fontsize=16)
    ax1.set_xticks(np.arange(0, 25, 5))
    ax1.set_ylim([-scale*0.1, scale*1.1])
    
    ax2 = plt.subplot(411)
    ax2.set_ylabel('Kinematic\nDistance (m)', fontsize=16)
    ax2.set_xticks(np.arange(0, 25, 5))
    ax2.set_ylim([-scale*0.1, scale*1.1])

    ax3 = plt.subplot(414)
    ax3.set_ylabel('Kinematic\nAngle (rad)', fontsize=16)
    ax3.set_xlabel('Time (sec)', fontsize=16)
    ax3.set_xticks(np.arange(0, 25, 5))
    ax3.set_ylim([-scale*0.1, scale*1.1])

    ax4 = plt.subplot(413)
    ax4.set_ylabel('Audio\nMagnitude (dec)', fontsize=16)
    ax4.set_xticks(np.arange(0, 25, 5))
    ax4.set_ylim([-scale*0.1, scale*1.1])

    # training data
    if useTrain:
    
        count = 0
        for i in xrange(len(trainData[0])):
            if useTrain_color:
                ax1.plot(trainTimeList[i], trainData[0][i])
                ax2.plot(trainTimeList[i], trainData[1][i])
                ax3.plot(trainTimeList[i], trainData[2][i])
                ax4.plot(trainTimeList[i], trainData[3][i])            
            else:
                ax1.plot(trainTimeList[i], trainData[0][i], 'b')
                ax2.plot(trainTimeList[i], trainData[1][i], 'b')
                ax3.plot(trainTimeList[i], trainData[2][i], 'b')
                ax4.plot(trainTimeList[i], trainData[3][i], 'b', label=str(count))            
            count = count + 1
            if verbose: print i, trainFileList[i]

        if not useTrain_color: ax4.legend(loc=3,prop={'size':16})

    # threshold-test data
    if useThsTest:
    
        count = 0
        for i in xrange(len(thresTestData[0])):
            ax1.plot(thresTestTimeList[i], thresTestData[0][i], 'k--')
            ax2.plot(thresTestTimeList[i], thresTestData[1][i], 'k--')
            ax3.plot(thresTestTimeList[i], thresTestData[2][i], 'k--')
            ax4.plot(thresTestTimeList[i], thresTestData[3][i], 'k--')            
            count = count + 1

    # normal test data
    if useNormalTest:
    
        count = 0
        for i in xrange(len(normalTestData[0])):
            ax1.plot(normalTestTimeList[i], normalTestData[0][i], 'm--')
            ax2.plot(normalTestTimeList[i], normalTestData[1][i], 'm--')
            ax3.plot(normalTestTimeList[i], normalTestData[2][i], 'm--')
            ax4.plot(normalTestTimeList[i], normalTestData[3][i], 'm--')
            count = count + 1

    # normal test data
    if useAbnormalTest:
    
        count = 0
        for i in xrange(len(abnormalTestData[0])):
            ax1.plot(abnormalTestTimeList[i], abnormalTestData[0][i])
            ax2.plot(abnormalTestTimeList[i], abnormalTestData[1][i])
            ax3.plot(abnormalTestTimeList[i], abnormalTestData[2][i])
            ax4.plot(abnormalTestTimeList[i], abnormalTestData[3][i])
            count = count + 1
            
    if save_pdf == True:
        fig.savefig('test.pdf')
        fig.savefig('test.png')
        os.system('cp test.p* ~/Dropbox/HRL/')
    else:
        plt.show()        


def evaluation(task_name, target_path, nSet=1, nState=20, cov_mult=5.0, hmm_renew=False, 
               verbose=False):

    tot_truePos = 0
    tot_falseNeg = 0
    tot_trueNeg = 0 
    tot_falsePos = 0
    
    # Check if there is already scaled data
    for i in xrange(nSet):        

        trainData, thresTestData, normalTestData, abnormalTestData, \
          trainTimeList, thresTestTimeList, normalTestTimeList, abnormalTestTimeList, \
          trainFileList, thsTestFileList, normalTestFileList, abnormalTestFileList \
          = getData(task_name, target_path, i)

        dynamic_thres_pkl = os.path.join(target_path, "ml_"+task_name+"_"+str(i)+".pkl")
          
        nDimension = len(trainData)

        # Create and train multivariate HMM
        hmm = learning_hmm_multi_4d(nState=nState, nEmissionDim=nDimension, verbose=False)
        ret = hmm.fit(xData1=trainData[0], xData2=trainData[1], xData3=trainData[2], xData4=trainData[3],\
                      ml_pkl=dynamic_thres_pkl, use_pkl=(not hmm_renew), cov_mult=[cov_mult]*16)

        minThresholds1 = tuneSensitivityGain(hmm, trainData, verbose=verbose)
        minThresholds2 = tuneSensitivityGain(hmm, thresTestData, verbose=verbose)
        minThresholds = minThresholds2
        for i in xrange(len(minThresholds1)):
            if minThresholds1[i] < minThresholds2[i]:
                minThresholds[i] = minThresholds1[i]
        minThresholds = minThresholds                

        truePos, falseNeg, trueNeg, falsePos = \
        tableOfConfusionOnline(hmm, normalTestData, abnormalTestData, c=minThresholds, verbose=verbose)

        tot_truePos += truePos
        tot_falseNeg += falseNeg
        tot_trueNeg += trueNeg 
        tot_falsePos += falsePos


    truePositiveRate = float(tot_truePos) / float(tot_truePos + tot_falseNeg) * 100.0
    trueNegativeRate = float(tot_trueNeg) / float(tot_trueNeg + tot_falsePos) * 100.0
    print "------------------------------------------------"
    print "Total set of data: ", nSet
    print "------------------------------------------------"
    print 'True Negative Rate:', trueNegativeRate, 'True Positive Rate:', truePositiveRate
    print "------------------------------------------------"

    return 


def getData(task_name, target_path, setID=0):
    print "start to getting data"
    
    # Check if there is already scaled data
    target_file = os.path.join(target_path, task_name+'_dataSet_'+str(setID) )        
    if os.path.isfile(target_file) is not True: 
        print "Missing data: ", setID
        return

    print "file: ", target_file
    d = ut.load_pickle(target_file)
    trainData            = d['trainData']
    thresTestData        = d['thresTestData']
    normalTestData       = d['normalTestData'] 
    abnormalTestData     = d['abnormalTestData']
    trainTimeList        = d['trainTimeList'] 
    thresTestTimeList    = d['thresTestTimeList'] 
    normalTestTimeList   = d['normalTestTimeList'] 
    abnormalTestTimeList = d['abnormalTestTimeList'] 

    trainFileList        = d['trainFileList'] 
    thsTestFileList      = d['thsTestFileList'] 
    normalTestFileList   = d['normalTestFileList'] 
    abnormalTestFileList = d['abnormalTestFileList'] 
    
    print "Load complete"
    return [trainData, thresTestData, normalTestData, abnormalTestData,\
      trainTimeList, thresTestTimeList, normalTestTimeList, abnormalTestTimeList,\
      trainFileList, thsTestFileList, normalTestFileList, abnormalTestFileList]


def likelihoodOfSequences(task_name, target_path, setID=0, \
                          nState=20, cov_mult=5.0,\
                          useTrain=True, useThsTest=True, useNormalTest=True, useAbnormalTest=True, \
                          useTrain_color=False, useThsTest_color=False, useNormalTest_color=True,\
                          hmm_renew=False, save_pdf=False, verbose=False):

    # get data
    trainData, thresTestData, normalTestData, abnormalTestData, \
      trainTimeList, thresTestTimeList, normalTestTimeList, abnormalTestTimeList, \
      trainFileList, thsTestFileList, normalTestFileList, abnormalTestFileList \
    = getData(task_name, target_path, setID)

    dynamic_thres_pkl = os.path.join(target_path, "ml_"+task_name+"_"+str(setID)+".pkl")

    nDimension = len(trainData)

    # Create and train multivariate HMM
    hmm = learning_hmm_multi_4d(nState=nState, nEmissionDim=nDimension, verbose=False)
    ret = hmm.fit(xData1=trainData[0], xData2=trainData[1], xData3=trainData[2], xData4=trainData[3],\
                  ml_pkl=dynamic_thres_pkl, use_pkl=(not hmm_renew), cov_mult=[cov_mult]*16)

    minThresholds1 = tuneSensitivityGain(hmm, trainData, verbose=verbose)
    minThresholds2 = tuneSensitivityGain(hmm, thresTestData, verbose=verbose)
    minThresholds = minThresholds2
    for i in xrange(len(minThresholds1)):
        if minThresholds1[i] < minThresholds2[i]:
            minThresholds[i] = minThresholds1[i]
    
    fig = plt.figure()

    # training data
    if useTrain:

        log_ll = []
        exp_log_ll = []        
        count = 0
        for i in xrange(len(trainData[0])):

            log_ll.append([])
            exp_log_ll.append([])
            for j in range(2, len(trainData[0][i])):
                X_test = hmm.convert_sequence(trainData[0][i][:j], trainData[1][i][:j], 
                                              trainData[2][i][:j], trainData[3][i][:j])
                try:
                    logp = hmm.loglikelihood(X_test)
                except:
                    print "Too different input profile that cannot be expressed by emission matrix"
                    return [], 0.0 # error

                log_ll[i].append(logp)


                exp_logp = hmm.expLikelihoods(trainData[0][i][:j], trainData[1][i][:j], 
                                              trainData[2][i][:j], trainData[3][i][:j],
                                              minThresholds)
                exp_log_ll[i].append(exp_logp)
                
            # disp
            if useTrain_color:
                plt.plot(log_ll[i], label=str(i))
                print i, " : ", trainFileList[i]                
            else:
                plt.plot(log_ll[i], 'b-')

        if useTrain_color: 
            plt.legend(loc=3,prop={'size':16})
            
        ## plt.plot(exp_log_ll[i], 'r-')            
            
          
    # threshold-test data
    if useThsTest:

        log_ll = []
        exp_log_ll = []        
        count = 0
        for i in xrange(len(thresTestData[0])):

            log_ll.append([])
            exp_log_ll.append([])
            for j in range(2, len(thresTestData[0][i])):
                X_test = hmm.convert_sequence(thresTestData[0][i][:j], thresTestData[1][i][:j], 
                                              thresTestData[2][i][:j], thresTestData[3][i][:j])
                try:
                    logp = hmm.loglikelihood(X_test)
                except:
                    print "Too different input profile that cannot be expressed by emission matrix"
                    return [], 0.0 # error

                log_ll[i].append(logp)

                exp_logp = hmm.expLikelihoods(thresTestData[0][i][:j], thresTestData[1][i][:j], 
                                              thresTestData[2][i][:j], thresTestData[3][i][:j],
                                              minThresholds)
                exp_log_ll[i].append(exp_logp)
                
            # disp 
            if useThsTest_color:
                print i, " : ", thsTestFileList[i]                
                plt.plot(log_ll[i], label=str(i))
            else:
                plt.plot(log_ll[i], 'k-')

        if useThsTest_color: 
            plt.legend(loc=3,prop={'size':16})

            ## plt.plot(log_ll[i], 'b-')
            ## plt.plot(exp_log_ll[i], 'r--')
                        

    # normal test data
    if useNormalTest:

        log_ll = []
        exp_log_ll = []        
        count = 0
        for i in xrange(len(normalTestData[0])):

            log_ll.append([])
            exp_log_ll.append([])
            for j in range(2, len(normalTestData[0][i])):
                X_test = hmm.convert_sequence(normalTestData[0][i][:j], normalTestData[1][i][:j], 
                                              normalTestData[2][i][:j], normalTestData[3][i][:j])
                try:
                    logp = hmm.loglikelihood(X_test)
                except:
                    print "Too different input profile that cannot be expressed by emission matrix"
                    return [], 0.0 # error

                log_ll[i].append(logp)

                exp_logp = hmm.expLikelihoods(normalTestData[0][i][:j], normalTestData[1][i][:j], 
                                              normalTestData[2][i][:j], normalTestData[3][i][:j],
                                              minThresholds)
                exp_log_ll[i].append(exp_logp)
                
            # disp 
            if verbose: 
                print i, normalTestFileList[i], np.amin(log_ll[i]), log_ll[i][-1]
                

            # disp 
            if useNormalTest_color:
                print i, " : ", normalTestFileList[i]                
                plt.plot(log_ll[i], label=str(i))
            else:
                plt.plot(log_ll[i], 'g-')

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
                X_test = hmm.convert_sequence(abnormalTestData[0][i][:j], abnormalTestData[1][i][:j], 
                                              abnormalTestData[2][i][:j], abnormalTestData[3][i][:j])
                try:
                    logp = hmm.loglikelihood(X_test)
                except:
                    print "Too different input profile that cannot be expressed by emission matrix"
                    return [], 0.0 # error

                log_ll[i].append(logp)

                exp_logp = hmm.expLikelihoods(abnormalTestData[0][i][:j], abnormalTestData[1][i][:j], 
                                              abnormalTestData[2][i][:j], abnormalTestData[3][i][:j],
                                              minThresholds)
                exp_log_ll[i].append(exp_logp)
                
            # disp 
            plt.plot(log_ll[i], 'r-')
            
    if save_pdf == True:
        fig.savefig('test.pdf')
        fig.savefig('test.png')
        os.system('cp test.p* ~/Dropbox/HRL/')
    else:
        plt.show()        
      

def preprocessData(subject_names, task_name, root_path, target_path, nSet=1, folding_ratio=[0.6, 0.2, 0.2], 
                   scale=1.0, downSampleSize=200, train_cutting_ratio=[0.0, 0.65], renew=False, verbose=False):


    # Check if there is already scaled data
    for i in xrange(nSet):        
        target_file = os.path.join(target_path, task_name+'_dataSet_'+str(i) )        
        if os.path.isfile(target_file) is not True: renew=True
            
    if renew == False: return        
    
    # List up recorded files
    folder_list = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path,d))]        

    success_list = []
    failure_list = []
    for d in folder_list:

        name_flag = False
        for name in subject_names:
            if d.find(name) >= 0: name_flag = True
                                    
        if name_flag and d.find(task_name) >= 0:
            files = os.listdir(os.path.join(root_path,d))

            for f in files:
                # pickle file name with full path
                pkl_file = os.path.join(root_path,d,f)
                
                if f.find('success') >= 0:
                    if len(success_list)==0: success_list = [pkl_file]
                    else: success_list.append(pkl_file)
                elif f.find('failure') >= 0:
                    if len(failure_list)==0: failure_list = [pkl_file]
                    else: failure_list.append(pkl_file)
                else:
                    print "It's not success/failure file: ", f

    print "--------------------------------------------"
    print "# of Success files: ", len(success_list)
    print "# of Failure files: ", len(failure_list)
    print "--------------------------------------------"
    
    # random training, threshold-test, test set selection
    nTrain   = int(len(success_list) * folding_ratio[0])
    nThsTest = int(len(success_list) * folding_ratio[1])
    nTest    = len(success_list) - nTrain - nThsTest

    if len(failure_list) < nTest: 
        print "Not enough failure data"
        sys.exit()

    # minimum and maximum vales for scaling
    ## dataList, _ = loadData(failure_list, isTrainingData=False, downSampleSize=downSampleSize)
    dataList, _ = loadData(success_list, isTrainingData=False, downSampleSize=downSampleSize)
    minVals = []
    maxVals = []
    for modality in dataList:
        minVals.append(np.min(modality))
        maxVals.append(np.max(modality))
        
    for i in xrange(nSet):

        # index selection
        success_idx  = range(len(success_list))
        failure_idx  = range(len(failure_list))
        train_idx    = random.sample(success_idx, nTrain)
        ths_test_idx = random.sample([x for x in success_idx if not x in train_idx], nThsTest)
        success_test_idx = [x for x in success_idx if not (x in train_idx or x in ths_test_idx)]
        failure_test_idx = random.sample(failure_idx, nTest)

        # get training data
        trainFileList = [success_list[x] for x in train_idx]
        trainData, trainTimeList = loadData(trainFileList, isTrainingData=True, downSampleSize=downSampleSize)

        # get threshold-test data
        thsTestFileList = [success_list[x] for x in ths_test_idx]
        thresTestData, thresTestTimeList = loadData([success_list[x] for x in ths_test_idx], 
                                                    isTrainingData=True, downSampleSize=downSampleSize)

        # get test data
        normalTestFileList = [success_list[x] for x in success_test_idx]
        abnormalTestFileList = [failure_list[x] for x in failure_test_idx]
        normalTestData, normalTestTimeList = loadData([success_list[x] for x in success_test_idx], 
                                                      isTrainingData=False, downSampleSize=downSampleSize)
        abnormalTestData, abnormalTestTimeList = loadData([failure_list[x] for x in failure_test_idx], 
                                                          isTrainingData=False, downSampleSize=downSampleSize)

        # scaling data
        trainData_scaled,_ ,_  = scaleData(trainData, scale=scale, minVals=minVals, 
                                                 maxVals=maxVals, verbose=verbose)
        thresTestData_scaled,_ ,_ = scaleData(thresTestData, scale=scale, minVals=minVals, maxVals=maxVals, 
                                            verbose=verbose)
        normalTestData_scaled,_ ,_ = scaleData(normalTestData, scale=scale, minVals=minVals, 
                                               maxVals=maxVals, verbose=verbose)
        abnormalTestData_scaled,_ ,_ = scaleData(abnormalTestData, scale=scale, minVals=minVals, 
                                               maxVals=maxVals, verbose=verbose)

        # cutting data (only traing and thresTest data)
        start_idx = int(float(len(trainData_scaled[0][0]))*train_cutting_ratio[0])
        end_idx   = int(float(len(trainData_scaled[0][0]))*train_cutting_ratio[1])

        for j in xrange(len(trainData_scaled)):
            for k in xrange(len(trainData_scaled[j])):
                trainData_scaled[j][k] = trainData_scaled[j][k][start_idx:end_idx]
                trainTimeList[k]       = trainTimeList[k][start_idx:end_idx]
                
        for j in xrange(len(thresTestData_scaled)):
            for k in xrange(len(thresTestData_scaled[j])):                
                thresTestData_scaled[j][k] = thresTestData_scaled[j][k][start_idx:end_idx]
                thresTestTimeList[k]       = thresTestTimeList[k][start_idx:end_idx]

        for j in xrange(len(normalTestData_scaled)):
            for k in xrange(len(normalTestData_scaled[j])):                
                normalTestData_scaled[j][k] = normalTestData_scaled[j][k][start_idx:end_idx]
                normalTestTimeList[k]       = normalTestTimeList[k][start_idx:end_idx]
                
        for j in xrange(len(abnormalTestData_scaled)):
            for k in xrange(len(abnormalTestData_scaled[j])):                
                abnormalTestData_scaled[j][k] = abnormalTestData_scaled[j][k][start_idx:end_idx]
                abnormalTestTimeList[k]       = abnormalTestTimeList[k][start_idx:end_idx]
        
            
        # Save data using dictionary
        d = {}
        d['trainData'] = trainData_scaled
        d['thresTestData'] = thresTestData_scaled
        d['normalTestData'] = normalTestData_scaled
        d['abnormalTestData'] = abnormalTestData_scaled
        d['trainTimeList'] = trainTimeList
        d['thresTestTimeList'] = thresTestTimeList
        d['normalTestTimeList'] = normalTestTimeList
        d['abnormalTestTimeList'] = abnormalTestTimeList

        d['trainFileList'] = trainFileList
        d['thsTestFileList'] = thsTestFileList
        d['normalTestFileList'] = normalTestFileList
        d['abnormalTestFileList'] = abnormalTestFileList 

        d['minVals'] = minVals
        d['maxVals'] = maxVals

        target_file = os.path.join(target_path, task_name+'_dataSet_'+str(i) )
        try:
            ut.save_pickle(d, target_file)        
        except:
            print "There is already target file: "
        
    
    return 


def scaleData(dataList, scale=10, minVals=None, maxVals=None, verbose=False):

    # Determine max and min values
    if minVals is None:
        minVals = []
        maxVals = []
        for modality in dataList:
            minVals.append(np.min(modality))
            maxVals.append(np.max(modality))
        if verbose:
            print 'minValues', minVals
            print 'maxValues', maxVals

    nDimension = len(dataList)
    dataList_scaled = []
    for i in xrange(nDimension):
        dataList_scaled.append([])

    # Scale features
    for i in xrange(nDimension):
        if i==3: new_scale=scale #*0.2
        else: new_scale = scale
        
        for j in xrange(len(dataList[i])):
            dataList_scaled[i].append( scaling( dataList[i][j], minVals[i], maxVals[i], new_scale).tolist() )
            
    return dataList_scaled, minVals, maxVals


def tuneSensitivityGain(hmm, dataSample, verbose=False):
    minThresholds = np.zeros(hmm.nGaussian) + 10000

    n = len(dataSample[0])
    for i in range(n):
        m = len(dataSample[0][i])

        for j in range(2, m):
            threshold, index = hmm.get_sensitivity_gain(dataSample[0][i][:j], dataSample[1][i][:j], 
                                                        dataSample[2][i][:j], dataSample[3][i][:j])
            if not threshold:
                continue

            if minThresholds[index] > threshold:
                minThresholds[index] = threshold
                if verbose: print '(',i,',',n,')', 'Minimum threshold: ', minThresholds[index], index

    print minThresholds
    return minThresholds


def tableOfConfusionOnline(hmm, normalTestData, abnormalTestData, c=-5, verbose=False):
    truePos = 0
    trueNeg = 0
    falsePos = 0
    falseNeg = 0

    # positive is anomaly
    # negative is non-anomaly
    if verbose: print '\nBeginning anomaly testing for test set\n'

    # for normal test data
    for i in xrange(len(normalTestData[0])):
        if verbose: print 'Anomaly Error for test set %d' % i

        for j in range(2, len(normalTestData[0][i])):
            anomaly, error = hmm.anomaly_check(normalTestData[0][i][:j], 
                                               normalTestData[1][i][:j], 
                                               normalTestData[2][i][:j],
                                               normalTestData[3][i][:j], c)

            if verbose: print anomaly, error

            # This is a successful nonanomalous attempt
            if anomaly:
                falsePos += 1
                print 'Success Test', i,',',j, ' in ',len(normalTestData[0][i]), ' |', anomaly, error
                break
            elif not anomaly and j == len(normalTestData[0][i]) - 1:
                trueNeg += 1
                break


    # for abnormal test data
    for i in xrange(len(abnormalTestData[0])):
        if verbose: print 'Anomaly Error for test set %d' % i

        for j in range(2, len(abnormalTestData[0][i])):
            anomaly, error = hmm.anomaly_check(abnormalTestData[0][i][:j], 
                                               abnormalTestData[1][i][:j], 
                                               abnormalTestData[2][i][:j],
                                               abnormalTestData[3][i][:j], c)

            if verbose: print anomaly, error


            else:
                if anomaly:
                    truePos += 1
                    break
                elif not anomaly and j == len(abnormalTestData[0][i]) - 1:
                    falseNeg += 1
                    print 'Failure Test', i,',',j, ' in ',len(abnormalTestData[0][i]), ' |', anomaly, error
                    break

    truePositiveRate = float(truePos) / float(truePos + falseNeg) * 100.0
    trueNegativeRate = float(trueNeg) / float(trueNeg + falsePos) * 100.0
    return truePos, falseNeg, trueNeg, falsePos
    


if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    p.add_option('--dataRenew', '--dr', action='store_true', dest='bDataRenew',
                 default=False, help='Renew pickle files.')
    p.add_option('--hmmRenew', '--hr', action='store_true', dest='bHMMRenew',
                 default=False, help='Renew HMM parameters.')
    p.add_option('--verbose', '--v', action='store_true', dest='bVerbose',
                 default=False, help='Print descriptions.')
    p.add_option('--plot', '--p', action='store_true', dest='bPlot',
                 default=False, help='Plot distribution of data.')
    p.add_option('--likelihoodplot', '--lp', action='store_true', dest='bLikelihoodPlot',
                 default=False, help='Plot the change of likelihood.')
    opt, args = p.parse_args()

    subject_names = ['s2'] #'personal', 
    task_name     = 'feeding' #['scooping', 'feeding']
    ## subject_names = ['pr2'] #'personal', 
    ## task_name     = 'scooping'

    data_root_path   = '/home/dpark/git/hrl-assistive/hrl_multimodal_anomaly_detection/src/recordings'
    data_target_path = '/home/dpark/git/hrl-assistive/hrl_multimodal_anomaly_detection/src/hrl_multimodal_anomaly_detection/hmm/data'

    # Scooping
    ## nSet           = 1
    ## folding_ratio  = [0.4, 0.2, 0.3]
    ## downSampleSize = 100
    ## nState         = 10
    ## cov_mult       = 5.0
    ## scale          = 1.0
    ## cutting_ratio  = [0.0, 1.0] #[0.0, 0.7]

    # Feeding
    nSet           = 1
    folding_ratio  = [0.5, 0.2, 0.3]
    downSampleSize = 100
    nState         = 10
    cov_mult       = 5.0
    scale          = 1.0
    cutting_ratio  = [0.0, 1.0] #[0.0, 0.7]
    
    preprocessData(subject_names, task_name, data_root_path, data_target_path, nSet=nSet, scale=scale,\
                   folding_ratio=folding_ratio, downSampleSize=downSampleSize, \
                   train_cutting_ratio=cutting_ratio, renew=opt.bDataRenew, verbose=opt.bVerbose)

    if opt.bPlot:
        distributionOfSequences(task_name, data_target_path, setID=0, scale=scale,\
                                useTrain=True, useThsTest=False, useNormalTest=False, useAbnormalTest=False,\
                                useTrain_color=False,\
                                save_pdf=False, verbose=True)        
    elif opt.bLikelihoodPlot:
        if opt.bDataRenew == True: opt.bHMMRenew=True
        likelihoodOfSequences(task_name, data_target_path, setID=0, nState=nState, cov_mult=cov_mult,\
                              useTrain=True, useThsTest=True, useNormalTest=False, useAbnormalTest=True,\
                              useTrain_color=False, useThsTest_color=True, useNormalTest_color=False,\
                              hmm_renew=opt.bHMMRenew, save_pdf=False, verbose=True)        
    else:            
        if opt.bDataRenew == True: opt.bHMMRenew=True
        evaluation(task_name, data_target_path, nSet=nSet, nState=nState, cov_mult=cov_mult,\
                   hmm_renew = opt.bHMMRenew, verbose=False)
