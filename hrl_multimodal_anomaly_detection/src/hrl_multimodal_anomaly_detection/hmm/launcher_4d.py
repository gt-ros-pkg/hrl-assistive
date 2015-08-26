#!/usr/bin/env python

import os, sys
import math
import struct
import numpy as np
import cPickle as pickle
from scipy import interpolate
import matplotlib.pyplot as plt
from sklearn import preprocessing
from mvpa2.datasets.base import Dataset
from plotGenerator import plotGenerator
from learning_hmm_multi_1d import learning_hmm_multi_1d
from learning_hmm_multi_2d import learning_hmm_multi_2d
from learning_hmm_multi_4d import learning_hmm_multi_4d

import roslib
roslib.load_manifest('hrl_multimodal_anomaly_detection')
import tf

def forceKinematics(fileName):
    with open(fileName, 'rb') as f:
        data = pickle.load(f)
        kinematics = data['kinematics_data']
        kinematicsTimes = data['kinematics_time']
        force = data['ft_force_raw']
        forceTimes = data['ft_time']

        # Use magnitude of forces
        forces = np.linalg.norm(force, axis=1).flatten()
        distances = []
        angles = []

        # Compute kinematic distances and angles
        for mic, spoon, objectCenter in kinematics:
            # Determine distance between mic and center of object
            distances.append(np.linalg.norm(mic - objectCenter))
            # Find angle between gripper-object vector and gripper-spoon vector
            micSpoonVector = spoon - mic
            micObjectVector = objectCenter - mic
            angle = np.arccos(np.dot(micSpoonVector, micObjectVector) / (np.linalg.norm(micSpoonVector) * np.linalg.norm(micObjectVector)))
            angles.append(angle)

        return forces, distances, angles, kinematicsTimes, forceTimes

def get_rms(block):
    # RMS amplitude is defined as the square root of the
    # mean over time of the square of the amplitude.
    # so we need to convert this string of bytes into
    # a string of 16-bit samples...

    # we will get one short out for each
    # two chars in the string.
    count = len(block)/2
    structFormat = '%dh' % count
    shorts = struct.unpack(structFormat, block)

    # iterate over the block.
    sum_squares = 0.0
    for sample in shorts:
        # sample is a signed short in +/- 32768.
        # normalize it to 1.0
        n = sample / 32768.0
        sum_squares += n*n

    return math.sqrt(sum_squares / count)

def audioFeatures(fileName):
    with open(fileName, 'rb') as f:
        data = pickle.load(f)
        audios = data['audio_data_raw']
        audioTimes = data['audio_time']
        magnitudes = []
        for audio in audios:
            magnitudes.append(get_rms(audio))

        return magnitudes, audioTimes

def create_mvpa_dataset(aXData1, aXData2, aXData3, aXData4, chunks, labels):
    feat_list = []
    for x1, x2, x3, x4, chunk in zip(aXData1, aXData2, aXData3, aXData4, chunks):
        feat_list.append([x1, x2, x3, x4])

    data = Dataset(samples=feat_list)
    data.sa['id'] = range(0, len(labels))
    data.sa['chunks'] = chunks
    data.sa['targets'] = labels

    return data

def extrapolateData(data, maxsize):
    return [x if len(x) >= maxsize else x + [x[-1]]*(maxsize-len(x)) for x in data]

def extrapolateAllData(allData, maxsize):
    return [extrapolateData(data, maxsize) for data in allData]

def scaling(X, minVal, maxVal, scale=1.0):
    X = np.array(X)
    return (X - minVal) / (maxVal - minVal) * scale

def loadData(fileNames, iterationSets, isTrainingData=False, downSampleSize=100, verbose=False):
    timesList = []

    forcesTrueList = []
    distancesTrueList = []
    anglesTrueList = []
    audioTrueList = []
    for fileName, iterations in zip(fileNames, iterationSets):
        for i in iterations:
            name = fileName % i # Insert iteration value into filename
            audio, audioTimes = audioFeatures(name)
            forces, distances, angles, kinematicsTimes, forceTimes = forceKinematics(name)

            # There will be much more kinematics data than force or audio, so interpolate to fill in the gaps
            # print 'Force shape:', np.shape(forces), 'Distance shape:', np.shape(distances), 'Angles shape:', np.shape(angles), 'Audio shape:', np.shape(audio)

            newTimes = np.linspace(0.01, max(kinematicsTimes), downSampleSize)
            forceInterp = interpolate.splrep(forceTimes, forces, s=0)
            forces = interpolate.splev(newTimes, forceInterp, der=0)
            distanceInterp = interpolate.splrep(kinematicsTimes, distances, s=0)
            distances = interpolate.splev(newTimes, distanceInterp, der=0)
            angleInterp = interpolate.splrep(kinematicsTimes, angles, s=0)
            angles = interpolate.splev(newTimes, angleInterp, der=0)
            # audioInterp = interpolate.splrep(audioTimes, audio, s=0)
            # audio = interpolate.splev(newTimes, audioInterp, der=0)

            # Downsample audio (nicely), by finding the closest time sample in audio for each new time stamp
            audioTimes = np.array(audioTimes)
            audio = [audio[np.abs(audioTimes - t).argmin()] for t in newTimes]

            # print 'Shapes after downsampling'
            # print 'Force shape:', np.shape(forces), 'Distance shape:', np.shape(distances), 'Angles shape:', np.shape(angles), 'Audio shape:', np.shape(audio)

            # Constant (horizontal linear) interpolation for audio data
            # tempAudio = []
            # audioIndex = 0
            # for t in kinematicsTimes:
            #     if t > audioTimes[audioIndex + 1] and audioIndex < len(audioTimes) - 2:
            #         audioIndex += 1
            #     tempAudio.append(audio[audioIndex])
            # audio = tempAudio

            forcesTrueList.append(forces.tolist())
            distancesTrueList.append(distances.tolist())
            anglesTrueList.append(angles.tolist())
            audioTrueList.append(audio)
            timesList.append(newTimes.tolist())

    if verbose: print 'Load shapes pre extrapolation:', np.shape(forcesTrueList), np.shape(distancesTrueList), np.shape(anglesTrueList), np.shape(audioTrueList)

    # Each iteration may have a different number of time steps, so we extrapolate so they are all consistent
    if isTrainingData:
        # Find the largest iteration
        maxsize = max([len(x) for x in forcesTrueList])
        # Extrapolate each time step
        forcesTrueList, distancesTrueList, anglesTrueList, audioTrueList, timesList = extrapolateAllData([forcesTrueList, distancesTrueList,
                                                                                            anglesTrueList, audioTrueList, timesList], maxsize)

    if verbose: print 'Load shapes post extrapolation:', np.shape(forcesTrueList), np.shape(distancesTrueList), np.shape(anglesTrueList), np.shape(audioTrueList)

    return forcesTrueList, distancesTrueList, anglesTrueList, audioTrueList, timesList

def createSampleSet(forcesList, distancesList, anglesList, pdfList, init=0):
    testDataSet = create_mvpa_dataset(forcesList, distancesList, anglesList, pdfList, [10]*len(forcesList), [True]*len(forcesList))
    return [testDataSet.samples[init:, i, :] for i in xrange(4)]

def tableOfConfusion(hmm, forcesList, distancesList=None, anglesList=None, audioList=None, testForcesList=None,
                     testDistancesList=None, testAnglesList=None, testAudioList=None, numOfSuccess=5, c=-5, verbose=False):
    truePos = 0
    trueNeg = 0
    falsePos = 0
    falseNeg = 0

    # if verbose: print '\nBeginning anomaly testing for nonanomalous training set\n'
    # for i in xrange(len(forcesList)):
    #     if verbose: print 'Anomaly Error for training set %d' % i
    #     if distancesList is None:
    #         anomaly, error = hmm.anomaly_check(forcesList[i], c)
    #     elif anglesList is None:
    #         anomaly, error = hmm.anomaly_check(forcesList[i], distancesList[i], c)
    #     else:
    #         anomaly, error = hmm.anomaly_check(forcesList[i], distancesList[i], anglesList[i], audioList[i], c)
    #
    #     if verbose: print anomaly, error
    #
    #     if not anomaly:
    #         trueNeg += 1
    #     else:
    #         falsePos += 1

    if verbose: print '\nBeginning anomaly testing for test set\n'
    for i in xrange(len(testForcesList)):
        if verbose: print 'Anomaly Error for test set %d' % i
        if distancesList is None:
            anomaly, error = hmm.anomaly_check(testForcesList[i], c)
        elif anglesList is None:
            anomaly, error = hmm.anomaly_check(testForcesList[i], testDistancesList[i], c)
        else:
            anomaly, error = hmm.anomaly_check(testForcesList[i], testDistancesList[i], testAnglesList[i], 
                                               testAudioList[i], c)

        if verbose: print anomaly, error

        if i < numOfSuccess:
            # This is a successful nonanomalous attempt
            if not anomaly:
                trueNeg += 1
            else:
                falsePos += 1
                print 'Test', i, '|', anomaly, error
        else:
            if anomaly:
                truePos += 1
            else:
                falseNeg += 1
                print 'Test', i, '|', anomaly, error

    print 'True Positive:', truePos, 'True Negative:', trueNeg, 'False Positive:', falsePos, 'False Negative', falseNeg

    return (truePos + trueNeg) / float(len(testForcesList)) * 100.0


def tableOfConfusionOnline(hmm, forcesList, distancesList=None, anglesList=None, audioList=None, testForcesList=None,
                           testDistancesList=None, testAnglesList=None, testAudioList=None, numOfSuccess=5, c=-5, verbose=False):
    truePos = 0
    trueNeg = 0
    falsePos = 0
    falseNeg = 0

    # positive is anomaly
    # negative is non-anomaly
    if verbose: print '\nBeginning anomaly testing for test set\n'
    for i in xrange(len(testForcesList)):
        if verbose: print 'Anomaly Error for test set %d' % i

        for j in range(2, len(testForcesList[i])):
                
            if not verbose: sys.stdout = os.devnull

            if distancesList is None:
                anomaly, error = hmm.anomaly_check(testForcesList[i][:j], c)
            elif anglesList is None:
                anomaly, error = hmm.anomaly_check(testForcesList[i][:j], testDistancesList[i][:j], c)
            else:
                anomaly, error = hmm.anomaly_check(testForcesList[i][:j], testDistancesList[i][:j], testAnglesList[i][:j], 
                                                   testAudioList[i][:j], c)

            if not verbose: sys.stdout = sys.__stdout__

            if verbose: print anomaly, error

            if i < numOfSuccess:
                # This is a successful nonanomalous attempt
                if anomaly:
                    falsePos += 1
                    print 'Test', i, '|', anomaly, error
                    break
                elif not anomaly and j == len(testForcesList[i]) - 1:
                    trueNeg += 1
                    break
            else:
                if anomaly:
                    truePos += 1
                    break
                elif not anomaly and j == len(testForcesList[i]) - 1:
                    falseNeg += 1
                    print 'Test', i, '|', anomaly, error
                    break

                # if anomaly:
                #     truePos += 1
                #     break

    # trueNegativeRate = float(numOfSuccess - falsePos) / float(numOfSuccess) * 100.0
    # truePositiveRate = float(truePos) / float(len(testForcesList) - numOfSuccess) * 100.0

    truePositiveRate = float(truePos) / float(truePos + falseNeg) * 100.0
    trueNegativeRate = float(trueNeg) / float(trueNeg + falsePos) * 100.0
    print 'True Negative Rate:', trueNegativeRate, 'True Positive Rate:', truePositiveRate

    return 

def tuneSensitivityGain(hmm, forcesTestSample, distancesTestSample, anglesTestSample, audioTestSample, verbose=False):
    minThresholds = np.zeros(hmm.nGaussian) + 10000

    n = len(forcesTestSample)
    for i in range(n):
        m = len(forcesTestSample[i])

        for j in range(2, m):
            threshold, index = hmm.get_sensitivity_gain(forcesTestSample[i][:j], distancesTestSample[i][:j], 
                                                        anglesTestSample[i][:j], audioTestSample[i][:j])
            if not threshold:
                continue

            if minThresholds[index] > threshold:
                minThresholds[index] = threshold
                if verbose: print '(',i,',',n,')', 'Minimum threshold: ', minThresholds[index], index

    return minThresholds

minVals = None
maxVals = None
def scaleData(forcesTrueList, distancesTrueList, anglesTrueList, audioTrueList, scale=10, verbose=False):
    # Determine max and min values
    global minVals, maxVals
    if minVals is None:
        minVals = []
        maxVals = []
        for modality in [forcesTrueList, distancesTrueList, anglesTrueList, audioTrueList]:
            minVals.append(np.min(modality))
            maxVals.append(np.max(modality))
        if verbose:
            print 'minValues', minVals
            print 'maxValues', maxVals

    forcesList = []
    distancesList = []
    anglesList = []
    audioList = []

    # Scale features
    for forces, distances, angles, audio in zip(forcesTrueList, distancesTrueList, anglesTrueList, audioTrueList):
        forcesList.append(scaling(forces, minVals[0], maxVals[0], scale).tolist())
        distancesList.append(scaling(distances, minVals[1], maxVals[1], scale).tolist())
        anglesList.append(scaling(angles, minVals[2], maxVals[2], scale).tolist())
        audioList.append(scaling(audio, minVals[3], maxVals[3], scale).tolist())

    if verbose: print 'Forces: Min', np.min(forcesList), ', Max', np.max(forcesList), 'Distances: Min', np.min(distancesList), ', Max', np.max(distancesList), \
        'Angles: Min', np.min(anglesList), ', Max', np.max(anglesList), 'Audio: Min', np.min(audioList), ', Max', np.max(audioList)

    return forcesList, distancesList, anglesList, audioList

def trainMultiHMM(fileNamesTrain, iterationSetsTrain, fileNamesTest, iterationSetsTest, downSampleSize=200, scale=10, nState=20, cov_mult=1.0, numSuccess=10, verbose=False, isScooping=True, use_pkl=False, usePlots=False):
    fileName = os.path.join(os.path.dirname(__file__), 'data/Data%s.pkl' % ('' if isScooping else 'Feeding'))

    if use_pkl and os.path.isfile(fileName):
        with open(fileName, 'rb') as f:
            forcesList, distancesList, anglesList, audioList, timesList, forcesTrueList, distancesTrueList, anglesTrueList, \
            audioTrueList, testForcesList, testDistancesList, testAnglesList, testAudioList, testTimesList, \
            testForcesTrueList, testDistancesTrueList, testAnglesTrueList, testAudioTrueList = pickle.load(f)
    else:
        if verbose: print 'Loading training data'
        forcesTrueList, distancesTrueList, anglesTrueList, audioTrueList, timesList = loadData(fileNamesTrain, iterationSetsTrain, isTrainingData=True, downSampleSize=downSampleSize)

        if verbose: print 'Loading test data'
        testForcesTrueList, testDistancesTrueList, testAnglesTrueList, testAudioTrueList, testTimesList = loadData(fileNamesTest, iterationSetsTest, isTrainingData=True, downSampleSize=downSampleSize, verbose=verbose)

        forcesList, distancesList, anglesList, audioList = scaleData(forcesTrueList, distancesTrueList, anglesTrueList, audioTrueList, scale=scale, verbose=verbose)
        testForcesList, testDistancesList, testAnglesList, testAudioList = scaleData(testForcesTrueList, testDistancesTrueList, testAnglesTrueList, testAudioTrueList, scale=scale, verbose=verbose)

        if use_pkl:
            with open(fileName, 'wb') as f:
                pickle.dump((forcesList, distancesList, anglesList, audioList, timesList, forcesTrueList, distancesTrueList,
                             anglesTrueList, audioTrueList, testForcesList, testDistancesList, testAnglesList, testAudioList,
                             testTimesList, testForcesTrueList, testDistancesTrueList, testAnglesTrueList, testAudioTrueList),
                             f, protocol=pickle.HIGHEST_PROTOCOL)

    if verbose: print np.shape(forcesList), np.shape(distancesList), np.shape(anglesList), np.shape(audioList), np.shape(timesList)

    if usePlots:
        plots = plotGenerator(forcesList, distancesList, anglesList, audioList, timesList, forcesTrueList, distancesTrueList,
                              anglesTrueList,
                audioTrueList, testForcesList, testDistancesList, testAnglesList, testAudioList, testTimesList,
                testForcesTrueList, testDistancesTrueList, testAnglesTrueList, testAudioTrueList)
        # Plot modalities
        plots.distributionOfSequences(useTest=False)
        plots.distributionOfSequences(useTest=True, numSuccess=numSuccess)
        # plots.plotOneTrueSet()
        # plots.quickPlotModalities()
    
    # Setup training data
    forcesSample, distancesSample, anglesSample, audioSample = createSampleSet(forcesList, distancesList, anglesList, audioList)
    forcesTestSample, distancesTestSample, anglesTestSample, audioTestSample = createSampleSet(testForcesList, testDistancesList,
                                                                                               testAnglesList, testAudioList)

    # Daehyung: Quite high covariance. It may converge to local minima. I don't know whether the fitting result is reliable or not.
    #           If there is any error message in training, we have to fix. If we ignore, the result will be incorrect.
    # Create and train multivariate HMM
    hmm = learning_hmm_multi_4d(nState=nState, nEmissionDim=4, verbose=verbose)
    ret = hmm.fit(xData1=forcesSample, xData2=distancesSample, xData3=anglesSample, xData4=audioSample,
            ml_pkl='modals/ml_4d%s.pkl' % ('' if isScooping else '_Feeding'), use_pkl=use_pkl, cov_mult=[cov_mult]*16)

    if ret == 'Failure':
        return
    
    # 20 States, 1 cov_mult, scale 10

    if not verbose: sys.stdout = os.devnull
    minThresholds = tuneSensitivityGain(hmm, forcesTestSample, distancesTestSample, anglesTestSample, audioTestSample, verbose=verbose)
    if not verbose: sys.stdout = sys.__stdout__
    # minThresholds = tuneSensitivityGain(hmm, forcesSample, distancesSample, anglesSample, audioSample, verbose=verbose)
    if verbose:
        print 'Min threshold size:', np.shape(minThresholds)
        print minThresholds

    # Daehyung: here is online check method. It takes too long time. Probably, we need parallelization.
    tableOfConfusionOnline(hmm, forcesList, distancesList, anglesList, audioList, testForcesList, testDistancesList, testAnglesList, testAudioList, numOfSuccess=numSuccess, c=minThresholds, verbose=verbose)

    # Daehyung: Why do you execute offline check? If you use full-length of data, there will be almost no difference between
    #           fixed threshold, dynamic threshold with single coefficient, and dynamtic threshold with multiple coefficients, 
    #           since only last threshold will be used for the anomaly detection (expecially, the number of coefficients will not,
    #           almost affect the results). Subtle anomaly cannot be detected by offline method since the slightly dropped likelihood
    #           will be recovered at the end of data. I recommend to draw the change of likelihood for both normal and abnormal data.     
    ## tableOfConfusion(hmm, forcesList, distancesList, anglesList, audioList, testForcesList, testDistancesList, testAnglesList, testAudioList, numOfSuccess=10 if isScooping else 13, c=minThresholds)

    # c = 14 or c 18
    #for c in np.arange(0, 20, 0.5):
    #    print 'Table of Confusion for c=', c
    #    tableOfConfusion(hmm, forcesList, distancesList, anglesList, audioList, testForcesList, testDistancesList, testAnglesList, testAudioList, numOfSuccess=10 if isScooping else 13, c=(-1*c))

    # hmm.path_disp(forcesList, distancesList, anglesList, audioList)

    # figName = os.path.join(os.path.dirname(__file__), 'plots/likelihood_success.png')
    # hmm.likelihood_disp(forcesSample[1:], distancesSample[1:], anglesSample[1:], audioSample[1:], forcesTrueSample[1:], distancesTrueSample[1:], anglesTrueSample[1:], audioTrueSample[1:],
    #                     -5.0, figureSaveName=None)

    # for i in [4, 18, 19, 20]:
    #     forcesTestSample, distancesTestSample, anglesTestSample, audioTestSample = createSampleSet(testForcesList, testDistancesList, testAnglesList, testAudioList, init=i)
    #     forcesTrueTestSample, distancesTrueTestSample, anglesTrueTestSample, audioTrueTestSample = createSampleSet(testForcesTrueList, testDistancesTrueList, testAnglesTrueList, testAudioTrueList, init=i)
    #
    #     hmm.likelihood_disp(forcesTestSample, distancesTestSample, anglesTestSample, audioTestSample, forcesTrueTestSample, distancesTrueTestSample, anglesTrueTestSample, audioTrueTestSample,
    #                         forcesSample, distancesSample, anglesSample, audioSample, forcesTrueSample, distancesTrueSample, anglesTrueSample, audioTrueSample, -3.0, figureSaveName=None)

    ## # -- Global threshold approach --
    ## print '\n---------- Global Threshold ------------\n'
    ## hmmGlobal = learning_hmm_multi_4d(nState=20, nEmissionDim=4, check_method='global')
    ## hmmGlobal.fit(xData1=forcesSample, xData2=distancesSample, xData3=anglesSample, xData4=audioSample, ml_pkl='modals/ml_4d_global.pkl', use_pkl=True, cov_mult=[10.0]*16)

    ## for c in xrange(10):
    ##     print 'Table of Confusion for c=', c
    ##     tableOfConfusion(hmmGlobal, forcesList, distancesList, anglesList, audioList, testForcesList, testDistancesList, testAnglesList, testAudioList, numOfSuccess=10 if isScooping else 13, c=(-1*c))


    ## # -- 1 dimensional force hidden Markov model --
    ## print '\n\nBeginning testing for 1 dimensional force hidden Markov model\n\n'

    ## hmm1d = learning_hmm_multi_1d(nState=20, nEmissionDim=1)
    ## hmm1d.fit(xData1=forcesSample, ml_pkl='modals/ml_1d_force.pkl', use_pkl=True)

    ## for c in xrange(10):
    ##     print 'Table of Confusion for c=', c
    ##     tableOfConfusion(hmm1d, forcesList, testForcesList=testForcesList, numOfSuccess=16, c=(-1*c))

    ## # c=3 is the best fit
    ## accuracyForces = tableOfConfusion(hmm1d, forcesList, testForcesList=testForcesList, numOfSuccess=16, c=-3)

    ## # figName = os.path.join(os.path.dirname(__file__), 'plots/likelihood_success.png')
    ## # hmm1d.likelihood_disp(forcesSample, forcesTrueSample, -3.0, figureSaveName=None)


    ## # -- 1 dimensional distance hidden Markov model --
    ## print '\n\nBeginning testing for 1 dimensional distance hidden Markov model\n\n'

    ## hmm1d = learning_hmm_multi_1d(nState=20, nEmissionDim=1)
    ## hmm1d.fit(xData1=distancesSample, ml_pkl='modals/ml_1d_distance.pkl', use_pkl=True)

    ## for c in xrange(10):
    ##     print 'Table of Confusion for c=', c
    ##     tableOfConfusion(hmm1d, distancesList, testForcesList=testDistancesList, numOfSuccess=16, c=(-1*c))
    ## # c=1 is the best fit
    ## accuracyDistances = tableOfConfusion(hmm1d, distancesList, testForcesList=testDistancesList, numOfSuccess=16, c=-1)


    ## # -- 1 dimensional angle hidden Markov model --
    ## print '\n\nBeginning testing for 1 dimensional angle hidden Markov model\n\n'

    ## hmm1d = learning_hmm_multi_1d(nState=20, nEmissionDim=1)
    ## hmm1d.fit(xData1=anglesSample, ml_pkl='modals/ml_1d_angle.pkl', use_pkl=True)

    ## for c in xrange(10):
    ##     print 'Table of Confusion for c=', c
    ##     tableOfConfusion(hmm1d, anglesList, testForcesList=testAnglesList, numOfSuccess=16, c=(-1*c))
    ## # c=0 is the best fit
    ## accuracyAngles = tableOfConfusion(hmm1d, anglesList, testForcesList=testAnglesList, numOfSuccess=16, c=0)


    ## # -- 1 dimensional visual hidden Markov model --
    ## print '\n\nBeginning testing for 1 dimensional audio hidden Markov model\n\n'

    ## hmm1d = learning_hmm_multi_1d(nState=20, nEmissionDim=1)
    ## hmm1d.fit(xData1=audioSample, ml_pkl='modals/ml_1d_audio.pkl', use_pkl=True)

    ## for c in xrange(10):
    ##     print 'Table of Confusion for c=', c
    ##     tableOfConfusion(hmm1d, audioList, testForcesList=testAudioList, numOfSuccess=16, c=(-1*c))
    ## # c=2 is the best fit
    ## accuracyVision = tableOfConfusion(hmm1d, audioList, testForcesList=testAudioList, numOfSuccess=16, c=-2)


    ## # -- 2 dimensional distance/angle kinematics hidden Markov model --
    ## print '\n\nBeginning testing for 2 dimensional kinematics hidden Markov model\n\n'

    ## hmm2d = learning_hmm_multi_2d(nState=20, nEmissionDim=2)
    ## hmm2d.fit(xData1=forcesSample, xData2=distancesSample, ml_pkl='modals/ml_2d_kinematics_fd.pkl', use_pkl=True)

    ## for c in xrange(10):
    ##     print 'Table of Confusion for c=', c
    ##     tableOfConfusion(hmm2d, forcesList, distancesList, testForcesList=testForcesList, testDistancesList=testDistancesList, numOfSuccess=16, c=(-1*c))
    ## # c=1 is the best fit
    ## accuracyKinematics = tableOfConfusion(hmm2d, forcesList, distancesList, testForcesList=testForcesList, testDistancesList=testDistancesList, numOfSuccess=16, c=-1)

    ## # -- 2 dimensional distance/angle kinematics hidden Markov model --
    ## print '\n\nBeginning testing for 2 dimensional kinematics hidden Markov model with global threshold\n\n'

    ## hmm2d = learning_hmm_multi_2d(nState=20, nEmissionDim=2, check_method='global')
    ## hmm2d.fit(xData1=distancesSample, xData2=anglesSample, ml_pkl='modals/ml_2d_kinematics.pkl', use_pkl=True)

    ## for c in xrange(10):
    ##     print 'Table of Confusion for c=', c
    ##     tableOfConfusion(hmm2d, distancesList, anglesList, testForcesList=testDistancesList, testDistancesList=testAnglesList, numOfSuccess=16, c=(-1*c))
    ## # c=0 is the best fit
    ## tableOfConfusion(hmm2d, distancesList, anglesList, testForcesList=testDistancesList, testDistancesList=testAnglesList, numOfSuccess=16, c=0)


    ## fig = plt.figure()
    ## indices = np.arange(5)
    ## width = 0.5
    ## bars = plt.bar(indices + width/4.0, [accuracyForces, accuracyDistances, accuracyAngles, accuracyVision, accuracyKinematics], width, alpha=0.5, color='g')
    ## plt.ylabel('Accuracy (%)', fontsize=16)
    ## plt.xticks(indices + width*3/4.0, ('Force', 'Distance', 'Angle', 'Vision', 'Distance\n& Angle'), fontsize=16)
    ## plt.ylim([0, 100])

    ## def autolabel(rects):
    ##     # attach some text labels
    ##     for rect in rects:
    ##         height = rect.get_height()
    ##         plt.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%d'%int(height),
    ##                 ha='center', va='bottom')
    ## autolabel(bars)

    ## plt.show()

# trainMultiHMM(isScooping=True)
# trainMultiHMM(isScooping=False)

