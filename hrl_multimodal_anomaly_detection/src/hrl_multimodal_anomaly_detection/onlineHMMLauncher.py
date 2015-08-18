#!/usr/bin/env python

import os
import math
import struct
import numpy as np
import cPickle as pickle
from scipy import interpolate
from sklearn import preprocessing
from mvpa2.datasets.base import Dataset
from hmm.learning_hmm_multi_4d import learning_hmm_multi_4d

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
        distances = []
        angles = []

        print forces.shape

        # Compute kinematic distances and angles
        for mic, spoon, objectCenter, in kinematics:
            # TODO Make sure objectCenter is transformed to torso_lift_link frame

            # Determine distance between mic and center of object
            distances.append(np.linalg.norm(mic - objectCenter))
            # Find angle between gripper-object vector and gripper-spoon vector
            micSpoonVector = spoon - mic
            micObjectVector = objectCenter - mic
            angle = np.arccos(np.dot(micSpoonVector, micObjectVector) / (np.linalg.norm(micSpoonVector) * np.linalg.norm(micObjectVector)))
            angles.append(angle)

        # There will be much more audio data than force and kinematics, so interpolate to fill in the gaps
        distInterp = interpolate.splrep(kinematicsTimes, distances, s=0)
        angleInterp = interpolate.splrep(kinematicsTimes, angles, s=0)
        forceInterp = interpolate.splrep(forceTimes, forces, s=0)
        distances = interpolate.splev(audioTimes, distInterp, der=0)
        angles = interpolate.splev(audioTimes, angleInterp, der=0)
        forces = interpolate.splev(audioTimes, forceInterp, der=0)

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

        # Constant (horizontal linear) interpolation
        # tempPdf = []
        # visualIndex = 0
        # for forceTime in forceTimes:
        #     if forceTime > visualTimes[visualIndex + 1] and visualIndex < len(visualTimes) - 2:
        #         visualIndex += 1
        #     tempPdf.append(pdf[visualIndex])
        # pdf = tempPdf

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

    for fileName, iterations in zip(fileNames, iterationSets):
        for i in iterations:
            name = fileName % i # Insert iteration value into filename
            audio, audioTimes = audioFeatures(name)
            forces, distances, angles, kinematicsTimes, forceTimes = forceKinematics(name, audioTimes)

            # There will be much more kinematics data than force or audio, so interpolate to fill in the gaps
            print 'Force shape:', np.shape(forces), 'Distance shape:', np.shape(distances), 'Angles shape:', np.shape(angles), 'Audio shape:', np.shape(audio)
            forceInterp = interpolate.splrep(forceTimes, forces, s=0)
            audioInterp = interpolate.splrep(audioTimes, audio, s=0)
            forces = interpolate.splev(kinematicsTimes, forceInterp, der=0)
            audio = interpolate.splev(kinematicsTimes, audioInterp, der=0)

            if minVals is None:
                minVals = []
                maxVals = []
                for modality in [forces, distances, angles, audio]:
                    minVals.append(np.min(modality))
                    maxVals.append(np.max(modality))
                # audioDiff = maxVals[3] - minVals[3]
                # minVals[3] -= audioDiff / 2.0
                # maxVals[3] += audioDiff / 2.0
                # forceDiff = maxVals[0] - minVals[0]
                # minVals[0] -= forceDiff / 4.0
                # maxVals[0] += forceDiff / 4.0
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
            print 'Audio before scale', audio[0]
            audio = scaling(audio, minVals[3], maxVals[3], scale)
            print 'Audio after scale', audio[0], 'minVal', minVals[3], 'maxVal', maxVals[3]

            forcesList.append(forces.tolist())
            distancesList.append(distances.tolist())
            anglesList.append(angles.tolist())
            audioList.append(audio.tolist())
            timesList.append(kinematicsTimes)

    # Each iteration may have a different number of time steps, so we extrapolate so they are all consistent
    if isTrainingData:
        # Find the largest iteration
        maxsize = max([len(x) for x in forcesList])
        # Extrapolate each time step
        forcesList, distancesList, anglesList, audioList, timesList = extrapolateAllData([forcesList, distancesList, anglesList, audioList, timesList], maxsize)

    return forcesList, distancesList, anglesList, audioList, timesList, minVals, maxVals

def setupMultiHMM(isScooping=True):
    if isScooping:
        fileName = os.path.join(os.path.dirname(__file__), 'onlineData.pkl')
    else:
        fileName = os.path.join(os.path.dirname(__file__), 'onlineFeedingData.pkl')

    if not os.path.isfile(fileName):
        print 'Loading training data'
        if isScooping:
            fileNames = ['/home/dpark/git/hrl-assistive/hrl_multimodal_anomaly_detection/recordings/bowl3Stage1Test_scooping_fvk_07-27-2015_14-10-47/iteration_%d_success.pkl',
                         '/home/dpark/git/hrl-assistive/hrl_multimodal_anomaly_detection/recordings/bowl3Stage2Test_scooping_fvk_07-27-2015_14-25-13/iteration_%d_success.pkl',
                         '/home/dpark/git/hrl-assistive/hrl_multimodal_anomaly_detection/recordings/bowl3Stage3Test_scooping_fvk_07-27-2015_14-39-53/iteration_%d_success.pkl',
                         '/home/dpark/git/hrl-assistive/hrl_multimodal_anomaly_detection/recordings/bowl3Stage4Test_scooping_fvk_07-27-2015_15-01-32/iteration_%d_success.pkl',
                         '/home/dpark/git/hrl-assistive/hrl_multimodal_anomaly_detection/recordings/bowl3Stage5Test_scooping_fvk_07-27-2015_15-18-58/iteration_%d_success.pkl',
                         '/home/dpark/git/hrl-assistive/hrl_multimodal_anomaly_detection/recordings/bowl3Stage6Test_scooping_fvk_07-27-2015_15-40-37/iteration_%d_success.pkl']
            iterationSets = [xrange(3), xrange(3), xrange(3), xrange(3), xrange(3), xrange(3)]
        else:
            fileNames = ['/home/dpark/git/hrl-assistive/hrl_multimodal_anomaly_detection/recordings/recordFeeding1_feeding_fvk_07-29-2015_16-08-29/iteration_%d_success.pkl']
            iterationSets = [xrange(10)]
        forcesList, distancesList, anglesList, audioList, timesList, minVals, maxVals = loadData(fileNames, iterationSets, isTrainingData=True)

        with open(fileName, 'wb') as f:
            pickle.dump((forcesList, distancesList, anglesList, audioList, timesList, minVals, maxVals), f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(fileName, 'rb') as f:
            forcesList, distancesList, anglesList, audioList, timesList, minVals, maxVals = pickle.load(f)

    # Setup training data
    chunks = [10]*len(forcesList)
    labels = [True]*len(forcesList)
    trainDataSet = create_mvpa_dataset(forcesList, distancesList, anglesList, audioList, chunks, labels)

    print trainDataSet.samples.shape
    forcesSample = trainDataSet.samples[:, 0, :]
    distancesSample = trainDataSet.samples[:, 1, :]
    anglesSample = trainDataSet.samples[:, 2, :]
    audioSample = trainDataSet.samples[:, 3, :]

    ml_pkl = '../ml_4d_online.pkl' if isScooping else '../ml_4d_online_feeding.pkl'

    hmm = learning_hmm_multi_4d(nState=20, nEmissionDim=4)
    hmm.fit(xData1=forcesSample, xData2=distancesSample, xData3=anglesSample, xData4=audioSample, ml_pkl=ml_pkl, use_pkl=True)

    print 'minValues', minVals
    print 'maxValues', maxVals

    return hmm, minVals, maxVals, np.mean(forcesList, axis=0), np.mean(distancesList, axis=0), np.mean(anglesList, axis=0), np.mean(audioList, axis=0), timesList[0], forcesList, distancesList, anglesList, audioList, timesList

    # print '\nBeginning anomaly testing for nonanomalous training set\n'
    # for i in xrange(len(forcesList)):
    #     print 'Anomaly Error for training set %d' % i
    #     print hmm.anomaly_check(forcesList[i], distancesList[i], anglesList[i], pdfList[i], -5)
