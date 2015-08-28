#!/usr/bin/env python

import glob
import random
from hmm.learning_hmm_multi_4d import learning_hmm_multi_4d

from util import *

# -- Deprecated -- Please use icra2015Batch.py

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
            dataList_scaled[i].append( scaling(dataList[i][j], minVals[i], maxVals[i], scale).tolist() )

    return dataList_scaled, minVals, maxVals


def getData(successPath, failurePath, folding_ratio=(0.5, 0.2, 0.3), downSampleSize=200, verbose=False):
    success_list = glob.glob(successPath)
    failure_list = glob.glob(failurePath)

    if verbose:
        print "--------------------------------------------"
        print "# of Success files: ", len(success_list)
        print "# of Failure files: ", len(failure_list)
        print "--------------------------------------------"

    # random training, threshold-test, test set selection
    nTrain   = int(len(success_list) * folding_ratio[0])
    nThsTest = int(len(success_list) * folding_ratio[1])
    nTest    = len(success_list) - nTrain - nThsTest

    if len(failure_list) < nTest:
        print 'Not enough failure data'
        print 'Number of successful test iterations:', nTest
        print 'Number of failure test iterations:', len(failure_list)
        sys.exit()

    # index selection
    success_idx  = range(len(success_list))
    failure_idx  = range(len(failure_list))
    train_idx    = random.sample(success_idx, nTrain)
    ths_test_idx = random.sample([x for x in success_idx if not x in train_idx], nThsTest)
    success_test_idx = [x for x in success_idx if not (x in train_idx or x in ths_test_idx)]
    failure_test_idx = random.sample(failure_idx, nTest)

    # get training data
    trainData, trainTimeList = loadData([success_list[x] for x in train_idx],
                                        isTrainingData=True, downSampleSize=downSampleSize,
                                        verbose=verbose)

    # get threshold-test data
    thresTestData, thresTestTimeList = loadData([success_list[x] for x in ths_test_idx],
                                                isTrainingData=True, downSampleSize=downSampleSize,
                                                verbose=verbose)

    # get test data
    normalTestData, normalTestTimeList = loadData([success_list[x] for x in success_test_idx],
                                                  isTrainingData=False, downSampleSize=downSampleSize,
                                                  verbose=verbose)
    abnormalTestData, abnormalTestTimeList = loadData([success_list[x] for x in failure_test_idx],
                                                      isTrainingData=False, downSampleSize=downSampleSize,
                                                      verbose=verbose)

    return trainData, thresTestData, normalTestData, abnormalTestData, trainTimeList, thresTestTimeList, normalTestTimeList, abnormalTestTimeList

def iteration(downSampleSize=200, scale=10, nState=20, cov_mult=1.0, verbose=False,
              isScooping=True, use_pkl=False):
    task = ('pr2_scooping' if isScooping else 's*_feeding')
    successPath = '/home/dpark/git/hrl-assistive/hrl_multimodal_anomaly_detection/src/recordings/%s_success/*' % task
    failurePath = '/home/dpark/git/hrl-assistive/hrl_multimodal_anomaly_detection/src/recordings/%s_failure/*' % task

    trainDataTrue, thresTestDataTrue, normalTestDataTrue, abnormalTestDataTrue, trainTimeList, \
    thresTestTimeList, normalTestTimeList, abnormalTestTimeList = getData(successPath, failurePath,
                                                                          downSampleSize=downSampleSize, verbose=verbose)

    # Scale data
    trainData, minVals, maxVals = scaleData(trainDataTrue, scale=scale, verbose=verbose)
    thresTestData, _ , _ = scaleData(thresTestDataTrue, scale=scale, minVals=minVals, maxVals=maxVals, verbose=verbose)
    normalTestData, _ , _ = scaleData(normalTestDataTrue, scale=scale, minVals=minVals, maxVals=maxVals, verbose=verbose)
    abnormalTestData, _ , _ = scaleData(abnormalTestDataTrue, scale=scale, minVals=minVals, maxVals=maxVals, verbose=verbose)

    hmm = learning_hmm_multi_4d(nState=nState, nEmissionDim=4, verbose=verbose)
    ret = hmm.fit(xData1=trainData[0], xData2=trainData[1], xData3=trainData[2], xData4=trainData[3],
                  use_pkl=use_pkl, cov_mult=[cov_mult]*16)

    if ret == 'Failure':
        print 'HMM returned NaN for Baum-Welch'
        sys.exit()

    return hmm, minVals, maxVals

    # return hmm, minVals, maxVals, np.mean(forcesList, axis=0), np.mean(distancesList, axis=0), np.mean(anglesList, axis=0), np.mean(audioList, axis=0), timesList[0], forcesList, distancesList, anglesList, audioList, timesList
