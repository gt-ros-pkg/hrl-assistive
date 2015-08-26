#!/usr/bin/env python

import multiprocessing
import launcher_4d as launcher

def dataFiles(isScooping=False):
    if isScooping:
        fileNamesTrain = ['/home/dpark/git/hrl-assistive/hrl_multimodal_anomaly_detection/src/recordings/scoopingTraining2_scooping_fak_08-19-2015_10-25-58/iteration_%d_success.pkl']
        iterationSetsTrain = [xrange(10)]

        fileNamesTest = ['/home/dpark/git/hrl-assistive/hrl_multimodal_anomaly_detection/src/recordings/scoopingTraining2_scooping_fak_08-19-2015_10-25-58/iteration_%d_success.pkl',
                         '/home/dpark/git/hrl-assistive/hrl_multimodal_anomaly_detection/src/recordings/scoopingTraining_scooping_fak_08-19-2015_10-17-52/iteration_%d_success.pkl',
                         '/home/dpark/git/hrl-assistive/hrl_multimodal_anomaly_detection/src/recordings/scoopingTest_scooping_fak_08-19-2015_10-46-26/iteration_%d_failure.pkl']
        iterationSetsTest = [xrange(10, 15), xrange(5), xrange(4)]
        numSuccess = 10
    else:
        fileNamesTrain = ['/home/dpark/git/hrl-assistive/hrl_multimodal_anomaly_detection/src/recordings/feedingTraining/iteration_%d_success.pkl']
        iterationSetsTrain = [xrange(10)]

        fileNamesTest = ['/home/dpark/git/hrl-assistive/hrl_multimodal_anomaly_detection/src/recordings/feedingTest/iteration_%d_success.pkl',
                         '/home/dpark/git/hrl-assistive/hrl_multimodal_anomaly_detection/src/recordings/feedingTest/iteration_%d_failure.pkl']
        iterationSetsTest = [xrange(13), xrange(13)]
        numSuccess = 13

    return fileNamesTrain, iterationSetsTrain, fileNamesTest, iterationSetsTest, numSuccess

def batchTrain(parallel=True):
    for isScooping in [False, True]:
        for downSampleSize in [100, 200, 300]:
            for scale in [1, 5, 10]:
                for nState in [20, 30]:
                    for covMult in [1.0, 3.0, 5.0, 10.0]:
                        fileNamesTrain, iterationSetsTrain, fileNamesTest, iterationSetsTest, numSuccess = dataFiles(isScooping=isScooping)

                        if parallel:
                            p = multiprocessing.Process(target=launcher.trainMultiHMM, args=(fileNamesTrain, iterationSetsTrain, fileNamesTest, iterationSetsTest,
                                          downSampleSize, scale, nState, covMult, 10, False, isScooping, False, False))
                            p.start()
                        else:
                            launcher.trainMultiHMM(fileNamesTrain, iterationSetsTrain, fileNamesTest, iterationSetsTest,
                                          downSampleSize=downSampleSize, scale=scale, nState=nState, cov_mult=covMult, numSuccess=10,
                                          verbose=False, isScooping=isScooping, use_pkl=False, usePlots=False)

batchTrain(parallel=False)
