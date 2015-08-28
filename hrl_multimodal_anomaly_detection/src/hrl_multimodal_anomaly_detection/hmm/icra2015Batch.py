#!/usr/bin/env python

import glob
import random
import multiprocessing
from learning_hmm_multi_4d import learning_hmm_multi_4d

from util import *

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
    abnormalTestData, abnormalTestTimeList = loadData([failure_list[x] for x in failure_test_idx],
                                                      isTrainingData=False, downSampleSize=downSampleSize,
                                                      verbose=verbose)

    return trainData, thresTestData, normalTestData, abnormalTestData, trainTimeList, thresTestTimeList, normalTestTimeList, abnormalTestTimeList


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
            with suppress_output():
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
            with suppress_output():
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


def iteration(downSampleSize=200, scale=10, nState=20, cov_mult=1.0, verbose=False,
              isScooping=True, use_pkl=False, findThresholds=True, train_cutting_ratio=[0.0, 0.65],
              ml_pkl='ml_temp_4d.pkl', saveData=False, savedDataFile=None):

    if savedDataFile is not None:
        with open(ml_pkl, 'rb') as f:
            d = pickle.load(f)
            trainData = d['trainData']
            hmm = learning_hmm_multi_4d(nState=nState, nEmissionDim=4, verbose=verbose)
            hmm.fit(xData1=trainData[0], xData2=trainData[1], xData3=trainData[2], xData4=trainData[3],
                          ml_pkl=ml_pkl, use_pkl=use_pkl, cov_mult=[cov_mult]*16)
            return hmm, d['minVals'], d['maxVals'], d['minThresholds']

    task = 'pr2_scooping' if isScooping else 's*_feeding'
    successPath = '/home/dpark/git/hrl-assistive/hrl_multimodal_anomaly_detection/src/recordings/%s_success/*' % task
    failurePath = '/home/dpark/git/hrl-assistive/hrl_multimodal_anomaly_detection/src/recordings/%s_failure/*' % task

    trainDataTrue, thresTestDataTrue, normalTestDataTrue, abnormalTestDataTrue, trainTimeList, \
    thresTestTimeList, normalTestTimeList, abnormalTestTimeList = getData(successPath, failurePath,
                                                                          downSampleSize=downSampleSize, verbose=verbose)

    # minimum and maximum vales for scaling from Daehyung
    success_list = glob.glob(successPath)
    dataList, _ = loadData(success_list, isTrainingData=False, downSampleSize=downSampleSize)
    minVals = []
    maxVals = []
    for modality in dataList:
        minVals.append(np.min(modality))
        maxVals.append(np.max(modality))
    
    # Scale data
    trainData, _, _ = scaleData(trainDataTrue, scale=scale, minVals=minVals, maxVals=maxVals, 
                                            verbose=verbose)
    thresTestData, _ , _ = scaleData(thresTestDataTrue, scale=scale, minVals=minVals, maxVals=maxVals, 
                                     verbose=verbose)
    normalTestData, _ , _ = scaleData(normalTestDataTrue, scale=scale, minVals=minVals, maxVals=maxVals, 
                                      verbose=verbose)
    abnormalTestData, _ , _ = scaleData(abnormalTestDataTrue, scale=scale, minVals=minVals, maxVals=maxVals, 
                                        verbose=verbose)

    # cutting data (only traing and thresTest data)
    start_idx = int(float(len(trainData[0][0]))*train_cutting_ratio[0])
    end_idx   = int(float(len(trainData[0][0]))*train_cutting_ratio[1])

    for j in xrange(len(trainData)):
        for k in xrange(len(trainData[j])):
            trainData[j][k] = trainData[j][k][start_idx:end_idx]
            trainTimeList[k] = trainTimeList[k][start_idx:end_idx]

    for j in xrange(len(thresTestData)):
        for k in xrange(len(thresTestData[j])):
            thresTestData[j][k] = thresTestData[j][k][start_idx:end_idx]
            thresTestTimeList[k] = thresTestTimeList[k][start_idx:end_idx]

    hmm = learning_hmm_multi_4d(nState=nState, nEmissionDim=4, verbose=verbose)
    ret = hmm.fit(xData1=trainData[0], xData2=trainData[1], xData3=trainData[2], xData4=trainData[3],
                  ml_pkl=ml_pkl, use_pkl=use_pkl, cov_mult=[cov_mult]*16)

    if ret == 'Failure':
        return

    if not findThresholds:
        return hmm, minVals, maxVals

    else:
        with suppress_output():
            minThresholds = tuneSensitivityGain(hmm, thresTestData, verbose=verbose)

        if verbose:
            print 'Min threshold size:', np.shape(minThresholds)
            print minThresholds

        if not saveData:
            return hmm, minVals, maxVals, minThresholds
        else:
            tableOfConfusionOnline(hmm, normalTestData, abnormalTestData, c=minThresholds, verbose=verbose)

            # Save data into file for later use (since it was randomly sampled)
            d = dict()
            d['minVals'] = minVals
            d['maxVals'] = maxVals
            d['minThresholds'] = minThresholds
            d['trainData'] = trainData
            d['thresTestData'] = thresTestData
            d['normalTestData'] = normalTestData
            d['abnormalTestData'] = abnormalTestData
            d['trainDataTrue'] = trainDataTrue
            d['thresTestDataTrue'] = thresTestDataTrue
            d['normalTestDataTrue'] = normalTestDataTrue
            d['abnormalTestDataTrue'] = abnormalTestDataTrue
            d['trainTimeList'] = trainTimeList
            d['thresTestTimeList'] = thresTestTimeList
            d['normalTestTimeList'] = normalTestTimeList
            d['abnormalTestTimeList'] = abnormalTestTimeList
            taskName = 'scooping' if isScooping else 'feeding'
            fileName = 'batchDataFiles/%s_%d_%d_%d_%d.pkl' % (taskName, downSampleSize, scale, nState, int(cov_mult))
            with open(fileName, 'wb') as f:
                pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)


def batchTrain(parallel=True):
    for isScooping in [False, True]:
        for downSampleSize in [100, 200, 300]:
            for scale in [1, 5, 10]:
                for nState in [20, 30]:
                    for covMult in [1.0, 3.0, 5.0, 10.0]:
                        print 'Beginning iteration | isScooping: %s, downSampleSize: %d, scale: %d, nState: %d, covMult: %d' % (isScooping, downSampleSize, scale, nState, covMult)
                        if parallel:
                            p = multiprocessing.Process(target=iteration, args=(downSampleSize, scale, nState, covMult, False, isScooping, False, False))
                            p.start()
                        else:
                            iteration(downSampleSize=downSampleSize, scale=scale, nState=nState, cov_mult=covMult,
                                      verbose=False, isScooping=isScooping, use_pkl=False, saveData=True)
                        print 'End of iteration'


def plotData(isScooping=False):
    if isScooping:
        fileName = '/home/dpark/git/hrl-assistive/hrl_multimodal_anomaly_detection/src/recordings/pr2_scooping_success/*'
    else:
        fileName = '/home/dpark/git/hrl-assistive/hrl_multimodal_anomaly_detection/src/recordings/s*_feeding_success/*'

    successList = glob.glob(fileName)
    iterations = [os.path.basename(f).split('_')[1] for f in glob.glob(fileName)]

    dataList, timeList = loadData(successList, isTrainingData=True, downSampleSize=200, verbose=False)

    for line, times, num in zip(dataList[0], timeList, iterations):
        plt.plot(times, line, label='%s' % num)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    plotData(isScooping=False)

    iteration(downSampleSize=100, scale=1.0, nState=10, cov_mult=1.0, train_cutting_ratio=[0.0, 1.0],
                                      verbose=False, isScooping=True, use_pkl=False, saveData=True)

    sys.exit()

    orig_stdout = sys.stdout
    f = file('out.txt', 'w')
    sys.stdout = f

    batchTrain(parallel=False)

    sys.stdout = orig_stdout
    f.close()
