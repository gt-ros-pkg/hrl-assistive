#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')

import glob
import random
import multiprocessing
from learning_hmm_multi_4d import learning_hmm_multi_4d
# from learning_hmm_multi_n import learning_hmm_multi_n

from utilOld import *

def scaleData(dataList, scale=[10,10,10,10], minVals=None, maxVals=None, verbose=False):
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
            dataList_scaled[i].append( scaling(dataList[i][j], minVals[i], maxVals[i], scale[i]).tolist() )

    return dataList_scaled, minVals, maxVals



def getData(success_list, failure_list, folding_ratio=(0.5, 0.3, 0.2), downSampleSize=200, verbose=False):

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


def likelihoodOfSequences(hmm, trainData, thresTestData=None, normalTestData=None, abnormalTestData=None,
                          save_pdf=False, verbose=False, minThresholds=None):
    print 'Plotting likelihoods'

    if minThresholds is None:
        minThresholds1 = tuneSensitivityGain(hmm, trainData, verbose=verbose)
        minThresholds2 = tuneSensitivityGain(hmm, thresTestData, verbose=verbose)
        minThresholds = minThresholds2
        for i in xrange(len(minThresholds1)):
            if minThresholds1[i] < minThresholds2[i]:
                minThresholds[i] = minThresholds1[i]

    fig = plt.figure()

    dataSet = [x for x in [trainData, thresTestData, normalTestData, abnormalTestData] if x is not None]
    colors = ['b', 'k', 'g', 'r'][:len(dataSet)]

    min_logp = 0.0
    max_logp = 0.0
        
    # training data
    for data, color in zip(dataSet, colors):
        log_ll = []
        exp_log_ll = []
        for i in xrange(len(data[0])):
            log_ll.append([])
            exp_log_ll.append([])
            for j in range(2, len(data[0][i])):
                X_test = hmm.convert_sequence(data[0][i][:j], data[1][i][:j],
                                              data[2][i][:j], data[3][i][:j])
                try:
                    logp = hmm.loglikelihood(X_test)
                except:
                    print "Too different input profile that cannot be expressed by emission matrix"
                    return [], 0.0 # error

                log_ll[i].append(logp)

                ## exp_logp = hmm.expLikelihoods(data[0][i][:j], data[1][i][:j],
                ##                               data[2][i][:j], data[3][i][:j],
                ##                               minThresholds)
                ## exp_log_ll[i].append(exp_logp)

            if max_logp < np.amax(log_ll): max_logp = np.amax(log_ll)
                
            plt.plot(log_ll[i], color + '-')

    plt.ylim([min_logp, max_logp])
            
    if save_pdf:
        fig.savefig('test.pdf')
        fig.savefig('test.png')
    else:
        plt.show()



trainData = None
trainTimeList = None
def iteration(downSampleSize=200, scale=[10,10,10,10], nState=20, cov_mult=1.0, anomaly_offset=0.0, verbose=False,
              isScooping=True, use_pkl=False, findThresholds=True, train_cutting_ratio=[0.0, 0.65],
              ml_pkl='ml_temp_4d.pkl', saveData=False, savedDataFile=None, plotLikelihood=False):
    global trainData, trainTimeList

    # Daehyung: where are you using savedDataFile?, what is different with ml_pkl?
    #           if savedDataFile is updated, ml_pkl should be replaced!!         
    if savedDataFile is not None:
        ## with open(ml_pkl, 'rb') as f:
        with open(savedDataFile, 'rb') as f:
            d = pickle.load(f)
            trainData = d['trainData']
            trainTimeList = d['trainTimeList']
            # hmm = learning_hmm_multi_4d(nState=nState, nEmissionDim=4, anomaly_offset=anomaly_offset, verbose=verbose)
            hmm = learning_hmm_multi_4d(nState=nState, nEmissionDim=4, anomaly_offset=anomaly_offset, verbose=verbose)
            hmm.fit(xData1=trainData[0], xData2=trainData[1], xData3=trainData[2], xData4=trainData[3],
                          ml_pkl=ml_pkl, use_pkl=use_pkl, cov_mult=[cov_mult]*16)
            return hmm, d['minVals'], d['maxVals'], d['minThresholds']

    ## task = 'pr2_scooping' if isScooping else 's*_feeding'
    ## successPath = '/home/dpark/git/hrl-assistive/hrl_multimodal_anomaly_detection/src/recordings/%s_success/*' % task
    ## failurePath = '/home/dpark/git/hrl-assistive/hrl_multimodal_anomaly_detection/src/recordings/%s_failure/*' % task

    # Daehyung: Modified to select multiple subjects
    #           folder name should include subject name, task name, and success/failure words.
    if isScooping:
        subject_names = ['pr2']
        task_name     = 'scooping'
    else:
        subject_names = ['s2','s3','s4'] #'personal',
        task_name     = 'feeding'
    
    # Loading success and failure data
    root_path = '/home/zerickson/feeding'
    success_list, failure_list = getSubjectFileList(root_path, subject_names, task_name)

    trainDataTrue, thresTestDataTrue, normalTestDataTrue, abnormalTestDataTrue, trainTimeList, \
    thresTestTimeList, normalTestTimeList, abnormalTestTimeList = getData(success_list, failure_list,
                                                                          downSampleSize=downSampleSize, verbose=verbose)

    # minimum and maximum vales for scaling from Daehyung
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

    # hmm = learning_hmm_multi_4d(nState=nState, nEmissionDim=4, anomaly_offset=anomaly_offset, verbose=verbose)
    hmm = learning_hmm_multi_4d(nState=nState, nEmissionDim=4, anomaly_offset=anomaly_offset, verbose=verbose)
    ret = hmm.fit(xData1=trainData[0], xData2=trainData[1], xData3=trainData[2], xData4=trainData[3],
                  ml_pkl=ml_pkl, use_pkl=use_pkl, cov_mult=[cov_mult]*16)

    if ret == 'Failure':
        return

    if not findThresholds:
        return hmm, minVals, maxVals

    else:
        with suppress_output():
            # minThresholds = tuneSensitivityGain(hmm, thresTestData, verbose=verbose)
            # #thresTest is not enough, so we also use training data.
            minThresholds1 = tuneSensitivityGain(hmm, trainData, verbose=verbose)
            minThresholds2 = tuneSensitivityGain(hmm, thresTestData, verbose=verbose)
            minThresholds3 = tuneSensitivityGain(hmm, normalTestData, verbose=verbose)
            minThresholds = minThresholds3
            for i in xrange(len(minThresholds1)):
                if minThresholds1[i] < minThresholds[i]:
                    minThresholds[i] = minThresholds1[i]
                if minThresholds2[i] < minThresholds[i]:
                    minThresholds[i] = minThresholds2[i]

        if verbose:
            print 'Min threshold size:', np.shape(minThresholds)
            print minThresholds

        if plotLikelihood:
            likelihoodOfSequences(hmm, trainData, thresTestData, normalTestData, abnormalTestData,
                                  save_pdf=True, verbose=False, minThresholds=minThresholds)

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
            fileName = os.path.join(os.path.dirname(__file__), 'dataFiles/%s_%d_%d_%d_%d.pkl') % (taskName, downSampleSize, scale[0], nState, int(cov_mult))
            # fileName = '/home/dpark/git/hrl-assistive/hrl_multimodal_anomaly_detection/src/hrl_multimodal_anomaly_detection/hmm/batchDataFiles/%s_%d_%d_%d_%d.pkl' % (taskName, downSampleSize, scale[0], nState, int(cov_mult))
            with open(fileName, 'wb') as f:
                pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)


def batchTrain(parallel=True):
    for isScooping in [False, True]:
        for downSampleSize in [100, 200, 300]:
            for scale in [1, 5, 10]: # scale is not anymore single value
                for nState in [20, 30]:
                    for covMult in [1.0, 3.0, 5.0, 10.0]:
                        print 'Beginning iteration | isScooping: %s, downSampleSize: %d, scale: %d, nState: %d, covMult: %d' % (isScooping, downSampleSize, scale, nState, covMult)
                        if parallel:
                            p = multiprocessing.Process(target=iteration, args=(downSampleSize, scale, nState, covMult, False, isScooping, False, False))
                            p.start()
                        else:
                            # TODO Needs to be updates
                            iteration(downSampleSize=downSampleSize, scale=scale, nState=nState, cov_mult=covMult,
                                      verbose=False, isScooping=isScooping, use_pkl=False, saveData=True)
                        print 'End of iteration'


def plotData(isScooping=False):
    if isScooping:
        fileName = '/home/dpark/git/hrl-assistive/hrl_multimodal_anomaly_detection/src/recordings/pr2_scooping_success/*'
    else:
        fileName = '/home/dpark/git/hrl-assistive/hrl_multimodal_anomaly_detection/src/recordings/s*_feeding_success/*'

    successList = [f for f in glob.glob(fileName) if not os.path.isdir(f)]
    iterations = [os.path.basename(f).split('_')[1] for f in glob.glob(fileName) if not os.path.isdir(f)]

    dataList, timeList = loadData(successList, isTrainingData=True, downSampleSize=200, verbose=False)

    for i, title in zip(xrange(len(dataList)), ['Forces', 'Distances', 'Angles', 'Audio']):
        for line, times, num in zip(dataList[i], timeList, iterations):
            plt.plot(times, line, label='%s' % num)
        plt.title(title)
        plt.legend()
        plt.show()


if __name__ == '__main__':
    # plotData(isScooping=False)

    # scooping
    isScooping = True
    nState=10
    anomaly_offset = -25.0
    cutting_ratio  = [0.0, 0.9]
    scale = [1.0,1.0,1.0,0.7]

    # feeding
    ## isScooping = False
    ## nState=15
    ## anomaly_offset = -20.0
    ## cutting_ratio  = [0.0, 0.7]
    ## scale = [1.0,1.0,0.7,1.0]
        
    iteration(downSampleSize=100, scale=scale, nState=nState, cov_mult=5.0, train_cutting_ratio=cutting_ratio,
              anomaly_offset=anomaly_offset, verbose=False, isScooping=isScooping, use_pkl=False, saveData=True,
              plotLikelihood=True)

    sys.exit()

    orig_stdout = sys.stdout
    f = file('out.txt', 'w')
    sys.stdout = f

    batchTrain(parallel=False)

    sys.stdout = orig_stdout
    f.close()
