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

def getData(subject_names, task_name, root_path, target_path, nSet=1, folding_ratio=[0.5, 0.2, 0.3], scale=1.0,\
            downSampleSize=200, renew=False, verbose=False):


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
    
    for i in xrange(nSet):

        # index selection
        success_idx  = range(len(success_list))
        failure_idx  = range(len(failure_list))
        train_idx    = random.sample(success_idx, nTrain)
        ths_test_idx = random.sample([x for x in success_idx if not x in train_idx], nThsTest)
        success_test_idx = [x for x in success_idx if not (x in train_idx or x in ths_test_idx)]
        failure_test_idx = random.sample(failure_idx, nTest)

        # get training data
        trainData, trainTimeList = loadData([success_list[x] for x in train_idx], 
                                            isTrainingData=True, downSampleSize=downSampleSize)

        # get threshold-test data
        thresTestData, thresTestTimeList = loadData([success_list[x] for x in ths_test_idx], 
                                                    isTrainingData=True, downSampleSize=downSampleSize)

        # get test data
        normalTestData, normalTestTimeList = loadData([success_list[x] for x in success_test_idx], 
                                                      isTrainingData=False, downSampleSize=downSampleSize)
        abnormalTestData, abnormalTestTimeList = loadData([success_list[x] for x in failure_test_idx], 
                                                          isTrainingData=False, downSampleSize=downSampleSize)

        # scaling data
        trainData_scaled, minVals, maxVals = scaleData(trainData, scale=scale, verbose=verbose)
        thresTestData_scaled,_ ,_ = scaleData(thresTestData, scale=scale, minVals=minVals, maxVals=maxVals, 
                                            verbose=verbose)
        normalTestData_scaled,_ ,_ = scaleData(normalTestData, scale=scale, minVals=minVals, maxVals=maxVals, 
                                             verbose=verbose)
        abnormalTestData_scaled,_ ,_ = scaleData(abnormalTestData, scale=scale, minVals=minVals, maxVals=maxVals, 
                                               verbose=verbose)

        # Save data using dictionary
        d = {}
        d['trainData'] = trainData
        d['thresTestData'] = thresTestData
        d['normalTestData'] = normalTestData
        d['abnormalTestData'] = abnormalTestData
        d['trainTimeList'] = trainTimeList
        d['thresTestTimeList'] = thresTestTimeList
        d['normalTestTimeList'] = normalTestTimeList
        d['abnormalTestTimeList'] = abnormalTestTimeList

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
        for j in xrange(len(dataList[i])):
            dataList_scaled[i].append( scaling( dataList[i][j], minVals[i], maxVals[i], scale).tolist() )
            
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
    print 'True Negative Rate:', trueNegativeRate, 'True Positive Rate:', truePositiveRate

    return 
    
def evaluation(task_name, target_path, nSet=1, nState=20, cov_mult=1.0, renew=False, verbose=False):

    # Check if there is already scaled data
    for i in xrange(nSet):        
        target_file = os.path.join(target_path, task_name+'_dataSet_'+str(i) )        
        if os.path.isfile(target_file) is not True: 
            print "Missing data: ", i
            return
        
        d = ut.load_pickle(target_file)
        trainData            = d['trainData']
        thresTestData        = d['thresTestData']
        normalTestData       = d['normalTestData'] 
        abnormalTestData     = d['abnormalTestData']
        trainTimeList        = d['trainTimeList'] 
        thresTestTimeList    = d['thresTestTimeList'] 
        normalTestTimeList   = d['normalTestTimeList'] 
        abnormalTestTimeList = d['abnormalTestTimeList'] 

        nDimension = len(trainData)
        use_pkl    = False

        # Create and train multivariate HMM
        hmm = learning_hmm_multi_4d(nState=nState, nEmissionDim=nDimension, verbose=False)
        ret = hmm.fit(xData1=trainData[0], xData2=trainData[1], xData3=trainData[2], xData4=trainData[3],
                      use_pkl=use_pkl, cov_mult=[cov_mult]*16)
    
        minThresholds = tuneSensitivityGain(hmm, thresTestData, verbose=verbose)

        tableOfConfusionOnline(hmm, normalTestData, abnormalTestData, c=minThresholds, verbose=verbose)
        

    return 



if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    p.add_option('--renew', action='store_true', dest='bRenew',
                 default=False, help='Renew pickle files.')
    p.add_option('--plot', '--p', action='store_true', dest='bPlot',
                 default=False, help='Plot distribution of data.')
    opt, args = p.parse_args()

    subject_names = ['s1']
    task_name     = 'feeding' #['scooping', 'feeding']
    data_root_path   = '/home/dpark/git/hrl-assistive/hrl_multimodal_anomaly_detection/src/recordings'
    data_target_path = '/home/dpark/git/hrl-assistive/hrl_multimodal_anomaly_detection/src/hrl_multimodal_anomaly_detection/hmm/data'


    getData(subject_names, task_name, data_root_path, data_target_path, nSet=1, 
            downSampleSize=100, renew=opt.bRenew)

    if opt.bPlot:
        plots = plotGenerator(forcesList, distancesList, anglesList, audioList, timesList, forcesTrueList,\
                              distancesTrueList,
                              anglesTrueList,
                audioTrueList, testForcesList, testDistancesList, testAnglesList, testAudioList, testTimesList,
                testForcesTrueList, testDistancesTrueList, testAnglesTrueList, testAudioTrueList)
        # Plot modalities
        plots.distributionOfSequences(useTest=False)

    else:            
        evaluation(task_name, data_target_path, renew = opt.bRenew, verbose=True)
