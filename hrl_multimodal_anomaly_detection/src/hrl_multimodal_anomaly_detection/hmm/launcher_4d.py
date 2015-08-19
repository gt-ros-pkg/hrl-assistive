#!/usr/bin/env python

import os
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

def forceKinematics(fileName, audioTimes):
    with open(fileName, 'rb') as f:
        data = pickle.load(f)
        kinematics = data['kinematics_data']
        kinematicsTimes = data['kinematics_time']
        force = data['ft_force_raw']
        forceTimes = data['ft_time']

        # Use magnitude of forces
        forces = np.linalg.norm(force, axis=1).flatten()
        # print 'forces shape:', forces.shape
        # print forces[:10]
        # print forces[-10:]
        # print np.shape(kinematicsTimes)
        # print kinematicsTimes[-10:]
        # print np.shape(forceTimes)
        # print forceTimes[-10:]
        # print np.shape(audioTimes)
        # print audioTimes[-10:]
        distances = []
        angles = []

        # print forces.shape

        # Compute kinematic distances and angles
        for mic, spoon, objectCenter in kinematics:
            # TODO Make sure objectCenter is transformed to torso_lift_link frame

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

        # TODO There may be a lot more audio data than other modalities, so interpote other modalities accordingly

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

minVals = None
maxVals = None
def scaling(X, minVal, maxVal, scale=1.0):
    X = np.array(X)
    return (X - minVal) / (maxVal - minVal) * scale

def loadData(fileNames, iterationSets, isTrainingData=False):
    global minVals, maxVals
    forcesList = []
    distancesList = []
    anglesList = []
    audioList = []
    timesList = []

    forcesTrueList = []
    distancesTrueList = []
    anglesTrueList = []
    audioTrueList = []
    for fileName, iterations in zip(fileNames, iterationSets):
        for i in iterations:
            name = fileName % i # Insert iteration value into filename
            audio, audioTimes = audioFeatures(name)
            forces, distances, angles, kinematicsTimes, forceTimes = forceKinematics(name, audioTimes)

            # There will be much more kinematics data than force or audio, so interpolate to fill in the gaps
            print 'Force shape:', np.shape(forces), 'Distance shape:', np.shape(distances), 'Angles shape:', np.shape(angles), 'Audio shape:', np.shape(audio)
            forceInterp = interpolate.splrep(forceTimes, forces, s=0)
            # audioInterp = interpolate.splrep(audioTimes, audio, s=0)
            forces = interpolate.splev(kinematicsTimes, forceInterp, der=0)
            # audio = interpolate.splev(kinematicsTimes, audioInterp, der=0)

            # print 'Audio'
            # print audio[:10], audio[-10:]
            # Constant (horizontal linear) interpolation for audio data
            tempAudio = []
            audioIndex = 0
            for t in kinematicsTimes:
                if t > audioTimes[audioIndex + 1] and audioIndex < len(audioTimes) - 2:
                    audioIndex += 1
                tempAudio.append(audio[audioIndex])
            audio = tempAudio
            # print 'Audio New'
            # print audio[:10], audio[-10:]

            forcesTrueList.append(forces.tolist())
            distancesTrueList.append(distances)
            anglesTrueList.append(angles)
            audioTrueList.append(audio)

            if minVals is None:
                minVals = []
                maxVals = []
                for modality in [forces, distances, angles, audio]:
                    minVals.append(np.min(modality))
                    maxVals.append(np.max(modality))
                print 'minValues', minVals
                print 'maxValues', maxVals

            scale = 1

            # Scale features
            # forces = preprocessing.scale(forces) * scale
            # distances = preprocessing.scale(distances) * scale
            # angles = preprocessing.scale(angles) * scale
            # audio = preprocessing.scale(audio) * scale
            forces = scaling(forces, minVals[0], maxVals[0], scale)
            distances = scaling(distances, minVals[1], maxVals[1], scale)
            angles = scaling(angles, minVals[2], maxVals[2], scale)
            # print 'Audio before scale', audio[0]
            audio = scaling(audio, minVals[3], maxVals[3], scale)
            # print 'Audio after scale', audio[0], 'minVal', minVals[3], 'maxVal', maxVals[3]

            forcesList.append(forces.tolist())
            distancesList.append(distances.tolist())
            anglesList.append(angles.tolist())
            audioList.append(audio.tolist())
            timesList.append(kinematicsTimes)

    print 'Load shapes pre extrapolation:', np.shape(forcesList), np.shape(distancesList), np.shape(anglesList), np.shape(audioList)

    # Each iteration may have a different number of time steps, so we extrapolate so they are all consistent
    if isTrainingData:
        # Find the largest iteration
        maxsize = max([len(x) for x in forcesList])
        # Extrapolate each time step
        forcesList, distancesList, anglesList, audioList, timesList, forcesTrueList, distancesTrueList, anglesTrueList, \
        audioTrueList = extrapolateAllData([forcesList, distancesList, anglesList, audioList, timesList,
                                                             forcesTrueList, distancesTrueList, anglesTrueList, audioTrueList], maxsize)

    print 'Load shapes post extrapolation:', np.shape(forcesList), np.shape(distancesList), np.shape(anglesList), np.shape(audioList)

    return forcesList, distancesList, anglesList, audioList, timesList, forcesTrueList, distancesTrueList, anglesTrueList, audioTrueList

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
            anomaly, error = hmm.anomaly_check(testForcesList[i], testDistancesList[i], testAnglesList[i], testAudioList[i], c)

        if verbose: print anomaly, error

        if i < numOfSuccess:
            # This is a successful nonanomalous attempt
            if not anomaly:
                trueNeg += 1
            else:
                falsePos += 1
        else:
            if anomaly:
                truePos += 1
            else:
                falseNeg += 1

    print 'True Positive:', truePos, 'True Negative:', trueNeg, 'False Positive:', falsePos, 'False Negative', falseNeg

    return (truePos + trueNeg) / float(len(testForcesList)) * 100.0

def dataFiles():
    fileNamesTrain = ['/home/dpark/git/hrl-assistive/hrl_multimodal_anomaly_detection/src/recordings/scoopingTraining2_scooping_fak_08-19-2015_10-25-58/iteration_%d_success.pkl']
    iterationSetsTrain = [xrange(10)]

    fileNamesTest = ['/home/dpark/git/hrl-assistive/hrl_multimodal_anomaly_detection/src/recordings/scoopingTraining2_scooping_fak_08-19-2015_10-25-58/iteration_%d_success.pkl',
                     '/home/dpark/git/hrl-assistive/hrl_multimodal_anomaly_detection/src/recordings/scoopingTraining_scooping_fak_08-19-2015_10-17-52/iteration_%d_success.pkl',
                     '/home/dpark/git/hrl-assistive/hrl_multimodal_anomaly_detection/src/recordings/scoopingTest_scooping_fak_08-19-2015_10-46-26/iteration_%d_failure.pkl']
    iterationSetsTest = [xrange(10, 15), xrange(5), xrange(4)]

    # fileNamesTrain = ['/home/zerickson/Recordings/bowl3Stage1Test_scooping_fvk_07-27-2015_14-10-47/iteration_%d_success.pkl',
    #              '/home/zerickson/Recordings/bowl3Stage2Test_scooping_fvk_07-27-2015_14-25-13/iteration_%d_success.pkl',
    #              '/home/zerickson/Recordings/bowl3Stage3Test_scooping_fvk_07-27-2015_14-39-53/iteration_%d_success.pkl',
    #              '/home/zerickson/Recordings/bowl3Stage4Test_scooping_fvk_07-27-2015_15-01-32/iteration_%d_success.pkl',
    #              '/home/zerickson/Recordings/bowl3Stage5Test_scooping_fvk_07-27-2015_15-18-58/iteration_%d_success.pkl',
    #              '/home/zerickson/Recordings/bowl3Stage6Test_scooping_fvk_07-27-2015_15-40-37/iteration_%d_success.pkl']
    # iterationSetsTrain = [xrange(3), xrange(3), xrange(3), xrange(3), xrange(3), xrange(3)]

    # fileNamesTest = ['/home/zerickson/Recordings/bowl3Stage1Test_scooping_fvk_07-27-2015_14-10-47/iteration_%d_success.pkl',
    #              '/home/zerickson/Recordings/bowl3Stage2Test_scooping_fvk_07-27-2015_14-25-13/iteration_%d_success.pkl',
    #              '/home/zerickson/Recordings/bowl3Stage3Test_scooping_fvk_07-27-2015_14-39-53/iteration_%d_success.pkl',
    #              '/home/zerickson/Recordings/bowl3Stage4Test_scooping_fvk_07-27-2015_15-01-32/iteration_%d_success.pkl',
    #              '/home/zerickson/Recordings/bowl3Stage5Test_scooping_fvk_07-27-2015_15-18-58/iteration_%d_success.pkl',
    #              '/home/zerickson/Recordings/bowl3Stage6Test_scooping_fvk_07-27-2015_15-40-37/iteration_%d_success.pkl',
    #              '/home/zerickson/Recordings/bowl3Stage1Anomalous_scooping_fvk_07-27-2015_14-17-52/iteration_%d_failure.pkl',
    #              '/home/zerickson/Recordings/bowl3Stage2Anomalous_scooping_fvk_07-27-2015_14-31-42/iteration_%d_failure.pkl',
    #              '/home/zerickson/Recordings/bowl3Stage3Anomalous_scooping_fvk_07-27-2015_14-47-24/iteration_%d_failure.pkl',
    #              '/home/zerickson/Recordings/bowl3Stage4Anomalous_scooping_fvk_07-27-2015_15-09-00/iteration_%d_failure.pkl',
    #              '/home/zerickson/Recordings/bowl3Stage5Anomalous_scooping_fvk_07-27-2015_15-27-51/iteration_%d_failure.pkl',
    #              '/home/zerickson/Recordings/bowl3Stage6Anomalous_scooping_fvk_07-27-2015_15-49-27/iteration_%d_failure.pkl']
    # iterationSetsTest = [xrange(3, 6), xrange(3, 6), xrange(3, 6), xrange(3, 6), xrange(3, 6), [3],
    #                  xrange(2, 4), xrange(4), xrange(4), xrange(1, 4), xrange(4), [0, 1, 3]]

    return fileNamesTrain, iterationSetsTrain, fileNamesTest, iterationSetsTest


def trainMultiHMM():
    fileName = os.path.join(os.path.dirname(__file__), 'data/Data.pkl')

    if not os.path.isfile(fileName):
        fileNamesTrain, iterationSetsTrain, fileNamesTest, iterationSetsTest = dataFiles()

        print 'Loading training data'
        forcesList, distancesList, anglesList, audioList, timesList, forcesTrueList, distancesTrueList, \
            anglesTrueList, audioTrueList = loadData(fileNamesTrain, iterationSetsTrain, isTrainingData=True)

        print 'Loading test data'
        testForcesList, testDistancesList, testAnglesList, testAudioList, testTimesList, testForcesTrueList, testDistancesTrueList, \
            testAnglesTrueList, testAudioTrueList = loadData(fileNamesTest, iterationSetsTest, isTrainingData=True)

        with open(fileName, 'wb') as f:
            pickle.dump((forcesList, distancesList, anglesList, audioList, timesList, forcesTrueList, distancesTrueList, anglesTrueList,
                         audioTrueList, testForcesList, testDistancesList, testAnglesList, testAudioList, testTimesList,
                         testForcesTrueList, testDistancesTrueList, testAnglesTrueList, testAudioTrueList), f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(fileName, 'rb') as f:
            forcesList, distancesList, anglesList, audioList, timesList, forcesTrueList, distancesTrueList, anglesTrueList, \
            audioTrueList, testForcesList, testDistancesList, testAnglesList, testAudioList, testTimesList, \
            testForcesTrueList, testDistancesTrueList, testAnglesTrueList, testAudioTrueList = pickle.load(f)

    print np.shape(forcesList), np.shape(distancesList), np.shape(anglesList), np.shape(audioList)
    print np.shape(forcesTrueList), np.shape(audioTrueList), np.shape(timesList)

    plots = plotGenerator(forcesList, distancesList, anglesList, audioList, timesList, forcesTrueList, distancesTrueList, anglesTrueList,
            audioTrueList, testForcesList, testDistancesList, testAnglesList, testAudioList, testTimesList,
            testForcesTrueList, testDistancesTrueList, testAnglesTrueList, testAudioTrueList)
    # plots.plotOneTrueSet()
    # plots.distributionOfSequences()

    # Plot modalities
    # plots.quickPlotModalities()

    # Setup training data
    forcesSample, distancesSample, anglesSample, audioSample = createSampleSet(forcesList, distancesList, anglesList, audioList)
    forcesTrueSample, distancesTrueSample, anglesTrueSample, audioTrueSample = createSampleSet(forcesTrueList, distancesTrueList, anglesTrueList, audioTrueList)

    # Create and train multivariate HMM
    hmm = learning_hmm_multi_4d(nState=20, nEmissionDim=4)
    hmm.fit(xData1=forcesSample, xData2=distancesSample, xData3=anglesSample, xData4=audioSample, ml_pkl='modals/ml_4d.pkl', use_pkl=True)

    # testSet = hmm.convert_sequence(forcesList[0], distancesList[0], anglesList[0], audioList[0])
    # print 'Log likelihood of testset:', hmm.loglikelihood(testSet)

    # tableOfConfusion(hmm, forcesList, distancesList, anglesList, audioList, testForcesList, testDistancesList, testAnglesList, testAudioList, numOfSuccess=14, verbose=True)

    # print 'c=2 is the best so far'
    # np.tile(np.append(audioList[0], audioList[0][-1]), (len(testForcesList), 1))
    # tableOfConfusion(hmm, forcesList, distancesList, anglesList, audioList, testForcesList, testDistancesList, testAnglesList, testAudioList, numOfSuccess=16, c=-2, verbose=True)

    for c in np.arange(0, 10, 0.5):
        print 'Table of Confusion for c=', c
        tableOfConfusion(hmm, forcesList, distancesList, anglesList, audioList, testForcesList, testDistancesList, testAnglesList, testAudioList, numOfSuccess=10, c=(-1*c))

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

    # -- Global threshold approach --
    print '\n---------- Global Threshold ------------\n'
    hmm = learning_hmm_multi_4d(nState=20, nEmissionDim=4, check_method='global')
    hmm.fit(xData1=forcesSample, xData2=distancesSample, xData3=anglesSample, xData4=audioSample, ml_pkl='modals/ml_4d_global.pkl', use_pkl=True)

    for c in xrange(10):
        print 'Table of Confusion for c=', c
        tableOfConfusion(hmm, forcesList, distancesList, anglesList, audioList, testForcesList, testDistancesList, testAnglesList, testAudioList, numOfSuccess=16, c=(-1*c))


    # -- 1 dimensional force hidden Markov model --
    print '\n\nBeginning testing for 1 dimensional force hidden Markov model\n\n'

    hmm1d = learning_hmm_multi_1d(nState=20, nEmissionDim=1)
    hmm1d.fit(xData1=forcesSample, ml_pkl='modals/ml_1d_force.pkl', use_pkl=True)

    for c in xrange(10):
        print 'Table of Confusion for c=', c
        tableOfConfusion(hmm1d, forcesList, testForcesList=testForcesList, numOfSuccess=16, c=(-1*c))

    # c=3 is the best fit
    accuracyForces = tableOfConfusion(hmm1d, forcesList, testForcesList=testForcesList, numOfSuccess=16, c=-3)

    # figName = os.path.join(os.path.dirname(__file__), 'plots/likelihood_success.png')
    # hmm1d.likelihood_disp(forcesSample, forcesTrueSample, -3.0, figureSaveName=None)


    # -- 1 dimensional distance hidden Markov model --
    print '\n\nBeginning testing for 1 dimensional distance hidden Markov model\n\n'

    hmm1d = learning_hmm_multi_1d(nState=20, nEmissionDim=1)
    hmm1d.fit(xData1=distancesSample, ml_pkl='modals/ml_1d_distance.pkl', use_pkl=True)

    for c in xrange(10):
        print 'Table of Confusion for c=', c
        tableOfConfusion(hmm1d, distancesList, testForcesList=testDistancesList, numOfSuccess=16, c=(-1*c))
    # c=1 is the best fit
    accuracyDistances = tableOfConfusion(hmm1d, distancesList, testForcesList=testDistancesList, numOfSuccess=16, c=-1)


    # -- 1 dimensional angle hidden Markov model --
    print '\n\nBeginning testing for 1 dimensional angle hidden Markov model\n\n'

    hmm1d = learning_hmm_multi_1d(nState=20, nEmissionDim=1)
    hmm1d.fit(xData1=anglesSample, ml_pkl='modals/ml_1d_angle.pkl', use_pkl=True)

    for c in xrange(10):
        print 'Table of Confusion for c=', c
        tableOfConfusion(hmm1d, anglesList, testForcesList=testAnglesList, numOfSuccess=16, c=(-1*c))
    # c=0 is the best fit
    accuracyAngles = tableOfConfusion(hmm1d, anglesList, testForcesList=testAnglesList, numOfSuccess=16, c=0)


    # -- 1 dimensional visual hidden Markov model --
    print '\n\nBeginning testing for 1 dimensional audio hidden Markov model\n\n'

    hmm1d = learning_hmm_multi_1d(nState=20, nEmissionDim=1)
    hmm1d.fit(xData1=audioSample, ml_pkl='modals/ml_1d_audio.pkl', use_pkl=True)

    for c in xrange(10):
        print 'Table of Confusion for c=', c
        tableOfConfusion(hmm1d, audioList, testForcesList=testAudioList, numOfSuccess=16, c=(-1*c))
    # c=2 is the best fit
    accuracyVision = tableOfConfusion(hmm1d, audioList, testForcesList=testAudioList, numOfSuccess=16, c=-2)


    # -- 2 dimensional distance/angle kinematics hidden Markov model --
    print '\n\nBeginning testing for 2 dimensional kinematics hidden Markov model\n\n'

    hmm2d = learning_hmm_multi_2d(nState=20, nEmissionDim=2)
    hmm2d.fit(xData1=forcesSample, xData2=distancesSample, ml_pkl='modals/ml_2d_kinematics_fd.pkl', use_pkl=True)

    for c in xrange(10):
        print 'Table of Confusion for c=', c
        tableOfConfusion(hmm2d, forcesList, distancesList, testForcesList=testForcesList, testDistancesList=testDistancesList, numOfSuccess=16, c=(-1*c))
    # c=1 is the best fit
    accuracyKinematics = tableOfConfusion(hmm2d, forcesList, distancesList, testForcesList=testForcesList, testDistancesList=testDistancesList, numOfSuccess=16, c=-1)

    # -- 2 dimensional distance/angle kinematics hidden Markov model --
    print '\n\nBeginning testing for 2 dimensional kinematics hidden Markov model with global threshold\n\n'

    hmm2d = learning_hmm_multi_2d(nState=20, nEmissionDim=2, check_method='global')
    hmm2d.fit(xData1=distancesSample, xData2=anglesSample, ml_pkl='modals/ml_2d_kinematics.pkl', use_pkl=True)

    for c in xrange(10):
        print 'Table of Confusion for c=', c
        tableOfConfusion(hmm2d, distancesList, anglesList, testForcesList=testDistancesList, testDistancesList=testAnglesList, numOfSuccess=16, c=(-1*c))
    # c=0 is the best fit
    tableOfConfusion(hmm2d, distancesList, anglesList, testForcesList=testDistancesList, testDistancesList=testAnglesList, numOfSuccess=16, c=0)


    fig = plt.figure()
    indices = np.arange(5)
    width = 0.5
    bars = plt.bar(indices + width/4.0, [accuracyForces, accuracyDistances, accuracyAngles, accuracyVision, accuracyKinematics], width, alpha=0.5, color='g')
    plt.ylabel('Accuracy (%)', fontsize=16)
    plt.xticks(indices + width*3/4.0, ('Force', 'Distance', 'Angle', 'Vision', 'Distance\n& Angle'), fontsize=16)
    plt.ylim([0, 100])

    def autolabel(rects):
        # attach some text labels
        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%d'%int(height),
                    ha='center', va='bottom')
    autolabel(bars)

    plt.show()

trainMultiHMM()

