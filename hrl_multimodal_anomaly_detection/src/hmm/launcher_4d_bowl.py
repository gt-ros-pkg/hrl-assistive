#!/usr/bin/env python

import os
import numpy as np
import cPickle as pickle
from scipy import interpolate
import matplotlib.pyplot as plt
from sklearn import preprocessing
from mvpa2.datasets.base import Dataset
from plotGenerator import plotGenerator
from learning_hmm_multi_1d import learning_hmm_multi_1d
from learning_hmm_multi_4d import learning_hmm_multi_4d

import roslib
roslib.load_manifest('hrl_multimodal_anomaly_detection')
import tf

def forceKinematics(fileName):
    with open(fileName, 'rb') as f:
        data = pickle.load(f)
        visual = data['visual_points']
        visualTimes = data['visual_time']
        force = data['ft_force_raw']
        forceTimes = data['ft_time']
        bowl = data['bowl_position']
        bowl = np.array([x[0, 0] for x in bowl])

        # Use magnitude of forces
        forces = np.linalg.norm(force, axis=1).flatten()
        distances = []
        angles = []

        print forces.shape

        # Compute kinematic distances and angles
        for pointSet, mic, spoon, bowlPosition, bowlPositionKinect, (bowlX, bowlY), bowlToKinectMat, (targetTrans, targetRot) in visual:
            # print 'Time:', timeStamp
            # Transform mic and spoon into torso_lift_link
            targetMatrix = np.dot(tf.transformations.translation_matrix(targetTrans), tf.transformations.quaternion_matrix(targetRot))
            mic = np.dot(targetMatrix, np.array([mic[0], mic[1], mic[2], 1.0]))[:3]
            spoon = np.dot(targetMatrix, np.array([spoon[0], spoon[1], spoon[2], 1.0]))[:3]

            # pointSet = np.c_[pointSet, np.ones(len(pointSet))]
            # pointSet = np.dot(targetMatrix, pointSet.T).T[:, :3]

            distances.append(np.linalg.norm(mic - bowl))
            # Find angle between gripper-bowl vector and gripper-spoon vector
            micSpoonVector = spoon - mic
            micBowlVector = bowl - mic
            angle = np.arccos(np.dot(micSpoonVector, micBowlVector) / (np.linalg.norm(micSpoonVector) * np.linalg.norm(micBowlVector)))
            angles.append(angle)

        # There will be much more force data than kinematics, so interpolate to fill in the gaps
        distInterp = interpolate.splrep(visualTimes, distances, s=0)
        angleInterp = interpolate.splrep(visualTimes, angles, s=0)
        distances = interpolate.splev(forceTimes, distInterp, der=0)
        angles = interpolate.splev(forceTimes, angleInterp, der=0)

        return forces, distances, angles, forceTimes

def visualFeatures(fileName, forceTimes):
    with open(fileName, 'rb') as f:
        data = pickle.load(f)
        visual = data['visual_points']
        visualTimes = data['visual_time']
        pdf = []
        for pointSet, mic, spoon, bowlPosition, bowlPositionKinect, (bowlX, bowlY), bowlToKinectMat, (targetTrans, targetRot) in visual:
            # Transformation matrix into torso_lift_link
            targetMatrix = np.dot(tf.transformations.translation_matrix(targetTrans), tf.transformations.quaternion_matrix(targetRot))

            pointSet = np.c_[pointSet, np.ones(len(pointSet))]
            pointSet = np.dot(targetMatrix, pointSet.T).T[:, :3]

            # Check for invalid points
            pointSet = pointSet[np.linalg.norm(pointSet, axis=1) < 5]

            # Find points within a sphere of radius 6 cm around the center of bowl
            nearbyPoints = np.linalg.norm(pointSet - bowlPosition, axis=1) < 0.11

            # Points near bowl
            points = pointSet[nearbyPoints]

            if len(points) <= 0:
                print 'ARGH, it happened on file', fileName

            # If no points found, try opening up to 8 cm
            if len(points) <= 0:
                # Find points within a sphere of radius 8 cm around the center of bowl
                nearbyPoints = np.linalg.norm(pointSet - bowlPosition, axis=1) < 0.13
                # Points near bowl
                points = pointSet[nearbyPoints]
                if len(points) <= 0:
                    print 'No points near bowl in dataset:', fileName
                    pdf.append(0)
                    continue

            # left = bowlPosition + [0, 0.06, 0]
            # right = bowlPosition - [0, 0.06, 0]
            # above = bowlPosition + [0.06, 0, 0]
            # below = bowlPosition - [0.06, 0, 0]

            # print 'Number of points:', len(points)
            # Try an exponential dropoff instead of Trivariate Gaussian Distribution, take sqrt to prevent overflow
            # pdfLeft = np.sum(np.exp(np.linalg.norm(points - left, axis=1) * -1.0))
            # pdfRight = np.sum(np.exp(np.linalg.norm(points - right, axis=1) * -1.0))
            # pdfAbove = np.sum(np.exp(np.linalg.norm(points - above, axis=1) * -1.0))
            # pdfBelow = np.sum(np.exp(np.linalg.norm(points - below, axis=1) * -1.0))
            # pdfLeft = np.sum(np.linalg.norm(points - left, axis=1))
            # pdfRight = np.sum(np.linalg.norm(points - right, axis=1))
            # pdfAbove = np.sum(np.linalg.norm(points - above, axis=1))
            # pdfBelow = np.sum(np.linalg.norm(points - below, axis=1))
            # pdfValue = np.power(pdfLeft + pdfRight + pdfAbove + pdfBelow, 1.0/4.0)
            # pdfValue = np.sqrt(np.sum(np.exp(np.linalg.norm(points - bowlPosition, axis=1) * -1.0))) / float(len(points))
            # pdf.append(pdfValue)

            # Scale all points to prevent division by small numbers and singular matrices
            newPoints = points * 20
            newBowlPosition = bowlPosition * 20

            # Define a receptive field within the bowl
            mu = [newBowlPosition]

            # Trivariate Gaussian Distribution
            pdfList = []
            for muSet in mu:
                n, m = newPoints.shape
                sigma = np.zeros((m, m))
                # Compute covariances
                for h in xrange(m):
                    for j in xrange(m):
                        sigma[h, j] = 1.0/n * np.dot((newPoints[:, h] - muSet[h]).T, newPoints[:, j] - muSet[j])
                        # Examples:
                        # sigma[0, 0] = 1/n * np.dot((xs - mux).T, xs - mux) # cov(X, X)
                        # sigma[0, 1] = 1/n * np.dot((xs - mux).T, ys - muy) # cov(X, Y)
                constant = 1.0 / np.sqrt((2*np.pi)**m * np.linalg.det(sigma))
                sigmaInv = np.linalg.inv(sigma)
                pdfValue = 0
                # Evaluate the Probability Density Function for each point
                for point in newPoints:
                    pointMu = point - muSet
                    # scalar = np.exp(np.abs(np.linalg.norm(point - newBowlPosition))*-2.0)
                    pdfValue += constant * np.exp(-1.0/2.0 * np.dot(np.dot(pointMu.T, sigmaInv), pointMu)) # Normally: np.exp(-1.0/2.0
                pdfList.append(pdfValue / float(len(points)))
            pdf.append(pdfList[0])

        # There will be much more force data than vision, so perform constant interpolation to fill in the gaps
        tempPdf = []
        visualIndex = 0
        for forceTime in forceTimes:
            if forceTime > visualTimes[visualIndex + 1] and visualIndex < len(visualTimes) - 2:
                visualIndex += 1
            tempPdf.append(pdf[visualIndex])
        pdf = tempPdf

        # There will be much more force data than vision, so interpolate to fill in the gaps
        # pdf1Interp = interpolate.splrep(visualTimes, pdf1, s=0)
        # pdf2Interp = interpolate.splrep(visualTimes, pdf2, s=0)
        # pdf1 = interpolate.splev(forceTimes, pdf1Interp, der=0)
        # pdf2 = interpolate.splev(forceTimes, pdf2Interp, der=0)

        return pdf

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
    pdfList = []
    timesList = []

    forcesTrueList = []
    distancesTrueList = []
    anglesTrueList = []
    pdfTrueList = []
    minList = []
    maxList = []
    for fileName, iterations in zip(fileNames, iterationSets):
        for i in iterations:
            name = fileName % i # Insert iteration value into filename
            forces, distances, angles, times = forceKinematics(name)
            pdf = visualFeatures(name, times)
            forcesTrueList.append(forces.tolist())
            distancesTrueList.append(distances.tolist())
            anglesTrueList.append(angles.tolist())
            pdfTrueList.append(pdf)

            # pdf = np.array(pdf) * 10
            # forces *= 10
            # distances *= 10
            # angles /= 10

            if minVals is None:
                minVals = []
                maxVals = []
                for modality in [forces, distances, angles, pdf]:
                    minVals.append(np.min(modality))
                    maxVals.append(np.max(modality))
                # pdfDiff = maxVals[3] - minVals[3]
                # minVals[3] -= pdfDiff / 2.0
                # maxVals[3] += pdfDiff / 2.0
                # forceDiff = maxVals[0] - minVals[0]
                # minVals[0] -= forceDiff / 4.0
                # maxVals[0] += forceDiff / 4.0
                print 'minValues', minVals
                print 'maxValues', maxVals

            scale = 1

            min_c1, max_c1 = np.min(forces), np.max(forces)
            min_c2, max_c2 = np.min(distances), np.max(distances)
            min_c3, max_c3 = np.min(angles), np.max(angles)
            min_c4, max_c4 = np.min(pdf), np.max(pdf)

            # Scale features
            # forces = preprocessing.scale(forces) * scale
            # distances = preprocessing.scale(distances) * scale
            # angles = preprocessing.scale(angles) * scale
            # pdf = preprocessing.scale(pdf) * scale
            forces = scaling(forces, minVals[0], maxVals[0], scale)
            distances = scaling(distances, minVals[1], maxVals[1], scale)
            angles = scaling(angles, minVals[2], maxVals[2], scale)
            # print 'Pdf before scale', pdf[0]
            pdf = scaling(pdf, minVals[3], maxVals[3], scale)
            # print 'Pdf after scale', pdf[0], 'minVal', minVals[3], 'maxVal', maxVals[3]

            # print 'Forces shape:', forces.shape
            # print 'Distances shape:', distances.shape

            forcesList.append(forces.tolist())
            distancesList.append(distances.tolist())
            anglesList.append(angles.tolist())
            pdfList.append(pdf.tolist())
            timesList.append(times)
            minList.append([min_c1, min_c2, min_c3, min_c4])
            maxList.append([max_c1, max_c2, max_c3, max_c4])

    # Each iteration may have a different number of time steps, so we extrapolate so they are all consistent
    if isTrainingData:
        # Find the largest iteration
        maxsize = max([len(x) for x in forcesList])
        # Extrapolate each time step
        forcesList, distancesList, anglesList, pdfList, timesList, forcesTrueList, distancesTrueList, anglesTrueList, \
        pdfTrueList, minList, maxList = extrapolateAllData([forcesList, distancesList, anglesList, pdfList, timesList,
                                                             forcesTrueList, distancesTrueList, anglesTrueList, pdfTrueList, minList, maxList], maxsize)

    return forcesList, distancesList, anglesList, pdfList, timesList, forcesTrueList, distancesTrueList, anglesTrueList, pdfTrueList, minList, maxList

def createSampleSet(forcesList, distancesList, anglesList, pdfList, init=0):
    testDataSet = create_mvpa_dataset(forcesList, distancesList, anglesList, pdfList, [10]*len(forcesList), [True]*len(forcesList))
    return [testDataSet.samples[init:, i, :] for i in xrange(4)]

def tableOfConfusion(hmm, forcesList, distancesList=None, anglesList=None, pdfList=None, testForcesList=None,
                     testDistancesList=None, testAnglesList=None, testPdfList=None, c=-5, verbose=False):
    truePos = 0
    trueNeg = 0
    falsePos = 0
    falseNeg = 0

    if verbose: print '\nBeginning anomaly testing for nonanomalous training set\n'
    for i in xrange(len(forcesList)):
        if verbose: print 'Anomaly Error for training set %d' % i
        if distancesList is not None:
            anomaly, error = hmm.anomaly_check(forcesList[i], distancesList[i], anglesList[i], pdfList[i], c)
        else:
            anomaly, error = hmm.anomaly_check(forcesList[i], c)
        if not anomaly:
            trueNeg += 1
        else:
            falsePos += 1

    if verbose: print '\nBeginning anomaly testing for test set\n'
    for i in xrange(len(testForcesList)):
        if verbose: print 'Anomaly Error for test set %d' % i
        if distancesList is not None:
            anomaly, error = hmm.anomaly_check(testForcesList[i], testDistancesList[i], testAnglesList[i], testPdfList[i], c)
        else:
            anomaly, error = hmm.anomaly_check(testForcesList[i], c)
        if anomaly:
            truePos += 1
        else:
            falseNeg += 1

    print 'True Positive:', truePos, 'True Negative:', trueNeg, 'False Positive:', falsePos, 'False Negative', falseNeg


def trainMultiHMM():
    fileName = os.path.join(os.path.dirname(__file__), 'data/bowl2Data2.pkl')

    if not os.path.isfile(fileName):
        print 'Loading training data'
        # fileNames = ['/home/zerickson/Recordings/bowlStage1Train_scooping_fvk_07-17-2015_16-03-36/iteration_%d_success.pkl',
        #              '/home/zerickson/Recordings/bowlStage2Train_scooping_fvk_07-17-2015_16-45-28/iteration_%d_success.pkl',
        #              '/home/zerickson/Recordings/bowlStage3Train_scooping_fvk_07-17-2015_17-13-56/iteration_%d_success.pkl']
        fileNames = ['/home/zerickson/Recordings/bowl2Stage1Train_scooping_fvk_07-24-2015_08-24-58/iteration_%d_success.pkl',
                     '/home/zerickson/Recordings/bowl2Stage2Train_scooping_fvk_07-24-2015_08-39-29/iteration_%d_success.pkl',
                     '/home/zerickson/Recordings/bowl2Stage3Train_scooping_fvk_07-24-2015_08-55-44/iteration_%d_success.pkl']
        iterationSets = [xrange(6), xrange(1, 5), xrange(6)]
        forcesList, distancesList, anglesList, pdfList, timesList, forcesTrueList, distancesTrueList, \
            anglesTrueList, pdfTrueList, minList, maxList = loadData(fileNames, iterationSets, isTrainingData=True)

        print 'Loading test data'
        # fileNames = ['/home/zerickson/Recordings/bowlStage1Test_scooping_fvk_07-17-2015_16-32-06/iteration_%d_success.pkl',
        #              '/home/zerickson/Recordings/bowlStage1Anomalous_scooping_fvk_07-17-2015_16-16-57/iteration_%d_failure.pkl',
        #              '/home/zerickson/Recordings/bowlStage2Test_scooping_fvk_07-17-2015_16-53-13/iteration_%d_success.pkl',
        #              '/home/zerickson/Recordings/bowlStage2Anomalous_scooping_fvk_07-17-2015_16-59-49/iteration_%d_failure.pkl',
        #              '/home/zerickson/Recordings/bowlStage3Test_scooping_fvk_07-17-2015_17-21-10/iteration_%d_success.pkl',
        #              '/home/zerickson/Recordings/bowlStage3Anomalous_scooping_fvk_07-17-2015_17-24-41/iteration_%d_failure.pkl',
        #              '/home/zerickson/Recordings/trial_scooping_fvk_07-24-2015_07-47-05/iteration_%d_success.pkl']
        fileNames = ['/home/zerickson/Recordings/bowl2Stage1Test_scooping_fvk_07-24-2015_08-30-59/iteration_%d_success.pkl',
                     '/home/zerickson/Recordings/bowl2Stage2Test_scooping_fvk_07-24-2015_08-46-01/iteration_%d_success.pkl',
                     '/home/zerickson/Recordings/bowl2Stage3Test_scooping_fvk_07-24-2015_09-04-36/iteration_%d_success.pkl',
                     '/home/zerickson/Recordings/trial_scooping_fvk_07-24-2015_07-47-05/iteration_%d_success.pkl',
                     '/home/zerickson/Recordings/bowl2Stage1Anomalous_scooping_fvk_07-24-2015_08-34-36/iteration_%d_failure.pkl',
                     '/home/zerickson/Recordings/bowl2Stage2Anomalous_scooping_fvk_07-24-2015_08-51-03/iteration_%d_failure.pkl',
                     '/home/zerickson/Recordings/bowl2Stage3Anomalous_scooping_fvk_07-24-2015_09-08-15/iteration_%d_failure.pkl']
        iterationSets = [xrange(3), xrange(3), xrange(3), xrange(5), xrange(3), xrange(3), xrange(3)]
        testForcesList, testDistancesList, testAnglesList, testPdfList, testTimesList, testForcesTrueList, testDistancesTrueList, \
            testAnglesTrueList, testPdfTrueList, testMinList, testMaxList = loadData(fileNames, iterationSets, isTrainingData=True)

        with open(fileName, 'wb') as f:
            pickle.dump((forcesList, distancesList, anglesList, pdfList, timesList, forcesTrueList, distancesTrueList, anglesTrueList,
                         pdfTrueList, minList, maxList, testForcesList, testDistancesList, testAnglesList, testPdfList, testTimesList,
                         testForcesTrueList, testDistancesTrueList, testAnglesTrueList, testPdfTrueList,
                         testMinList, testMaxList), f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(fileName, 'rb') as f:
            forcesList, distancesList, anglesList, pdfList, timesList, forcesTrueList, distancesTrueList, anglesTrueList, \
            pdfTrueList, minList, maxList, testForcesList, testDistancesList, testAnglesList, testPdfList, testTimesList, \
            testForcesTrueList, testDistancesTrueList, testAnglesTrueList, testPdfTrueList, testMinList, testMaxList = pickle.load(f)

    print np.shape(forcesTrueList), np.shape(pdfTrueList), np.shape(timesList)

    plots = plotGenerator(forcesList, distancesList, anglesList, pdfList, timesList, forcesTrueList, distancesTrueList, anglesTrueList,
            pdfTrueList, testForcesList, testDistancesList, testAnglesList, testPdfList, testTimesList,
            testForcesTrueList, testDistancesTrueList, testAnglesTrueList, testPdfTrueList)
    plots.plotOneTrueSet()
    plots.distributionOfSequences()


    # Plot modalities
    # for modality in [forcesTrueList, distancesTrueList, anglesTrueList, pdfTrueList]:
    # for modality in [forcesTrueList[:3] + testForcesTrueList[20:], distancesTrueList[:3] + testDistancesTrueList[20:], anglesTrueList[:3] + testAnglesTrueList[20:], pdfTrueList[:3] + testPdfTrueList[20:]]:
    # for modality in [forcesTrueList + testForcesTrueList[0:2], distancesTrueList + testDistancesTrueList[0:2], anglesTrueList + testAnglesTrueList[0:2], pdfTrueList + testPdfTrueList[0:2]]:
    #     for index, (modal, times) in enumerate(zip(modality, timesList + testTimesList[0:2])): # timesList + testTimesList[0:2]
    #         plt.plot(times, modal, label='%d' % index)
    #     plt.legend()
    #     plt.show()

    # Setup training data
    forcesSample, distancesSample, anglesSample, pdfSample = createSampleSet(forcesList, distancesList, anglesList, pdfList)
    forcesTrueSample, distancesTrueSample, anglesTrueSample, pdfTrueSample = createSampleSet(forcesTrueList, distancesTrueList, anglesTrueList, pdfTrueList)

    hmm = learning_hmm_multi_4d(nState=20, nEmissionDim=4)
    hmm.fit(xData1=forcesSample, xData2=distancesSample, xData3=anglesSample, xData4=pdfSample, ml_pkl='modals/ml_4d_bowl2.pkl', use_pkl=True)

    # testSet = hmm.convert_sequence(forcesList[0], distancesList[0], anglesList[0], pdfList[0])
    # print 'Log likelihood of testset:', hmm.loglikelihood(testSet)

    tableOfConfusion(hmm, forcesList, distancesList, anglesList, pdfList, testForcesList, testDistancesList, testAnglesList, testPdfList)

    hmm.path_disp(forcesList, distancesList, anglesList, pdfList)

    # figName = os.path.join(os.path.dirname(__file__), 'plots/likelihood_success.png')
    # hmm.likelihood_disp(forcesSample[1:], distancesSample[1:], anglesSample[1:], pdfSample[1:], forcesTrueSample[1:], distancesTrueSample[1:], anglesTrueSample[1:], pdfTrueSample[1:],
    #                     -5.0, figureSaveName=None)

    # # Find the largest iteration
    # maxsize = max([len(x) for x in testForcesList])
    # # Extrapolate each time step
    # testForcesList, testDistancesList, testAnglesList, testPdfList, \
    #     testForcesTrueList, testDistancesTrueList, testAnglesTrueList, testPdfTrueList = extrapolateAllData([testForcesList, testDistancesList, testAnglesList, testPdfList, testForcesTrueList, testDistancesTrueList, testAnglesTrueList, testPdfTrueList], maxsize)

    forcesTestSample, distancesTestSample, anglesTestSample, pdfTestSample = createSampleSet(testForcesList, testDistancesList, testAnglesList, testPdfList, init=17)
    forcesTrueTestSample, distancesTrueTestSample, anglesTrueTestSample, pdfTrueTestSample = createSampleSet(testForcesTrueList, testDistancesTrueList, testAnglesTrueList, testPdfTrueList, init=17)

    figName = os.path.join(os.path.dirname(__file__), 'plots/likelihood_anomaly.png')
    hmm.likelihood_disp(forcesTestSample, distancesTestSample, anglesTestSample, pdfTestSample, forcesTrueTestSample, distancesTrueTestSample, anglesTrueTestSample, pdfTrueTestSample,
                        forcesSample[1:], distancesSample[1:], anglesSample[1:], pdfSample[1:], forcesTrueSample[1:], distancesTrueSample[1:], anglesTrueSample[1:], pdfTrueSample[1:], -5.0, figureSaveName=None)

    # -- Global threshold approach --
    hmm = learning_hmm_multi_4d(nState=20, nEmissionDim=4, check_method='global')
    hmm.fit(xData1=forcesSample, xData2=distancesSample, xData3=anglesSample, xData4=pdfSample, ml_pkl='modals/ml_4d_bowl2.pkl', use_pkl=True)

    tableOfConfusion(hmm, forcesList, distancesList, anglesList, pdfList, testForcesList, testDistancesList, testAnglesList, testPdfList, c=-5, verbose=True)

    print '\n---------- Global Threshold ------------\n'
    for c in xrange(10):
        print 'Table of Confusion for c=', c
        tableOfConfusion(hmm, forcesList, distancesList, anglesList, pdfList, testForcesList, testDistancesList, testAnglesList, testPdfList, c=(-1*c))

    # -- 1 dimensional force hidden Markov model --
    print '\n\nBeginning testing for 1 dimensional force hidden Markov model\n\n'

    hmm1d = learning_hmm_multi_1d(nState=20, nEmissionDim=1)
    hmm1d.fit(xData1=forcesSample, ml_pkl='modals/ml_1d_bowl.pkl', use_pkl=True)

    tableOfConfusion(hmm1d, forcesList, testForcesList=testForcesList, c=-5, verbose=True)

    figName = os.path.join(os.path.dirname(__file__), 'plots/likelihood_success.png')
    hmm1d.likelihood_disp(forcesSample, forcesTrueSample, -5.0, figureSaveName=None)


trainMultiHMM()

