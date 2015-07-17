#!/usr/bin/env python

import os
import numpy as np
import cPickle as pickle
from scipy import interpolate
import matplotlib.pyplot as plt
from sklearn import preprocessing
from mvpa2.datasets.base import Dataset
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
            # Transform mic and spoon into torso_lift_link
            targetMatrix = np.dot(tf.transformations.translation_matrix(targetTrans), tf.transformations.quaternion_matrix(targetRot))

            # Transpose points into target frame
            pointSet = np.c_[pointSet, np.ones(len(pointSet))]
            pointSet = np.dot(targetMatrix, pointSet.T).T[:, :3]

            # Check for invalid points
            pointSet = pointSet[np.linalg.norm(pointSet, axis=1) < 5]

            # Find points within a sphere of radius 6 cm around the center of bowl
            nearbyPoints = np.linalg.norm(pointSet - bowlPosition, axis=1) < 0.06

            # Points near bowl
            points = pointSet[nearbyPoints]

            # If no points found, try opening up to 8 cm
            if len(points) <= 0:
                # Find points within a sphere of radius 8 cm around the center of bowl
                nearbyPoints = np.linalg.norm(pointSet - bowlPosition, axis=1) < 0.08
                # Points near bowl
                points = pointSet[nearbyPoints]
                if len(points) <= 0:
                    print 'No points near bowl in dataset:', fileName
                    pdf.append(0)
                    continue

            # Scale all points to prevent division by small numbers and singular matrices
            points *= 20
            bowlPosition *= 20

            # Define a receptive field within the bowl
            mu = [bowlPosition]

            pdfList = []
            for muSet in mu:
                n, m = points.shape
                sigma = np.zeros((m, m))
                # Compute covariances
                for h in xrange(m):
                    for j in xrange(m):
                        sigma[h, j] = 1.0/n * np.dot((points[:, h] - muSet[h]).T, points[:, j] - muSet[j])
                        # Examples:
                        # sigma[0, 0] = 1/n * np.dot((xs - mux).T, xs - mux) # cov(X, X)
                        # sigma[0, 1] = 1/n * np.dot((xs - mux).T, ys - muy) # cov(X, Y)
                constant = 1.0 / np.sqrt((2*np.pi)**m * np.linalg.det(sigma))
                sigmaInv = np.linalg.inv(sigma)
                pdfValue = 0
                # Evaluate the Probability Density Function for each point
                for point in points:
                    pointMu = point - muSet
                    pdfValue += constant * np.exp(-1.0/2.0 * np.dot(np.dot(pointMu.T, sigmaInv), pointMu))
                pdfList.append(pdfValue)
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

def loadData(fileNames, iterationSets, isTrainingData=False):
    forcesList = []
    distancesList = []
    anglesList = []
    pdf1List = []
    pdf2List = []
    timesList = []

    forcesTrueList = []
    distancesTrueList = []
    anglesTrueList = []
    pdf1TrueList = []
    pdf2TrueList = []
    minList = []
    maxList = []
    for fileName, iterations in zip(fileNames, iterationSets):
        for i in iterations:
            name = fileName % i # Insert iteration value into filename
            forces, distances, angles, times = forceKinematics(name)
            pdf1, pdf2 = visualFeatures(name, times)
            forcesTrueList.append(forces.tolist())
            distancesTrueList.append(distances.tolist())
            anglesTrueList.append(angles.tolist())
            pdf1TrueList.append(pdf1)
            pdf2TrueList.append(pdf2)

            scale = 1

            min_c1, max_c1 = np.min(forces), np.max(forces)
            min_c2, max_c2 = np.min(distances), np.max(distances)
            min_c3, max_c3 = np.min(angles), np.max(angles)
            # min_c3, max_c3 = np.min(pdf1), np.max(pdf1)
            min_c4, max_c4 = np.min(pdf2), np.max(pdf2)

            # Scale features
            forces = preprocessing.scale(forces) * scale
            distances = preprocessing.scale(distances) * scale
            angles = preprocessing.scale(angles) * scale
            pdf1 = preprocessing.scale(pdf1) * scale
            pdf2 = preprocessing.scale(pdf2) * scale

            # print 'Forces shape:', forces.shape
            # print 'Distances shape:', distances.shape

            forcesList.append(forces.tolist())
            distancesList.append(distances.tolist())
            anglesList.append(angles.tolist())
            pdf1List.append(pdf1.tolist())
            pdf2List.append(pdf2.tolist())
            timesList.append(times)
            minList.append([min_c1, min_c2, min_c3, min_c4])
            maxList.append([max_c1, max_c2, max_c3, max_c4])

    # Each iteration may have a different number of time steps, so we extrapolate so they are all consistent
    if isTrainingData:
        # Find the largest iteration
        maxsize = max([len(x) for x in forcesList])
        # Extrapolate each time step
        forcesList, distancesList, anglesList, pdf2List, timesList, forcesTrueList, distancesTrueList, anglesTrueList, \
        pdf2TrueList, minList, maxList = extrapolateAllData([forcesList, distancesList, anglesList, pdf2List, timesList,
                                                             forcesTrueList, distancesTrueList, anglesTrueList, pdf2TrueList, minList, maxList], maxsize)

    return forcesList, distancesList, pdf1List, pdf2List, timesList, forcesTrueList, distancesTrueList, pdf1TrueList, pdf2TrueList, minList, maxList

def trainMultiHMM():
    fileName = os.path.join(os.path.dirname(__file__), 'data/bowlData.pkl')

    if not os.path.isfile(fileName):
        print 'Loading training data'
        fileNames = ['/home/zerickson/Recordings/trainingDataVer2_scooping_fvk_07-15-2015_14-21-47/iteration_%d_success.pkl']
        iterationSets = [xrange(30)]
        forcesList, distancesList, anglesList, pdf2List, timesList, forcesTrueList, distancesTrueList, \
            anglesTrueList, pdf2TrueList, minList, maxList = loadData(fileNames, iterationSets, isTrainingData=True)

        print 'Loading test data'
        fileNames = ['/home/zerickson/Recordings/trainingDataVer2_scooping_fvk_07-15-2015_14-21-47/iteration_%d_success.pkl',
                     '/home/zerickson/Recordings/testDataOriginalBowlNoise_scooping_fvk_07-16-2015_18-24-43/iteration_%d_success.pkl',
                     '/home/zerickson/Recordings/testDataNewBowlVer1_scooping_fvk_07-16-2015_18-11-35/iteration_%d_success.pkl',
                     '/home/zerickson/Recordings/testDataAnomalyVer1_scooping_fvk_07-15-2015_12-14-48/iteration_%d_failure.pkl',
                     '/home/zerickson/Recordings/testDataAnomalousVer2_scooping_fvk_07-16-2015_17-29-48/iteration_%d_failure.pkl']
        iterationSets = [xrange(30, 40), [1, 2, 3], xrange(8), xrange(6), [i for i in xrange(19) if i not in [7, 9]]]
        testForcesList, testDistancesList, testAnglesList, testPdf2List, testTimesList, testForcesTrueList, testDistancesTrueList, \
            testAnglesTrueList, testPdf2TrueList, testMinList, testMaxList = loadData(fileNames, iterationSets)

        with open(fileName, 'wb') as f:
            pickle.dump((forcesList, distancesList, anglesList, pdf2List, timesList, forcesTrueList, distancesTrueList, anglesTrueList,
                         pdf2TrueList, minList, maxList, testForcesList, testDistancesList, testAnglesList, testPdf2List, testTimesList,
                         testForcesTrueList, testDistancesTrueList, testAnglesTrueList, testPdf2TrueList,
                         testMinList, testMaxList), f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(fileName, 'rb') as f:
            forcesList, distancesList, anglesList, pdf2List, timesList, forcesTrueList, distancesTrueList, anglesTrueList, \
            pdf2TrueList, minList, maxList, testForcesList, testDistancesList, testAnglesList, testPdf2List, testTimesList, \
            testForcesTrueList, testDistancesTrueList, testAnglesTrueList, testPdf2TrueList, testMinList, testMaxList = pickle.load(f)

    print np.shape(forcesTrueList), np.shape(pdf2TrueList), np.shape(timesList)

    # Plot modalities
    # for modality in [forcesTrueList, distancesTrueList, pdf1TrueList, pdf2TrueList]:
    # for modality in [testForcesTrueList[:3] + testForcesTrueList[40:], testDistancesTrueList[:3] + testDistancesTrueList[40:], testAnglesTrueList[:3] + testAnglesTrueList[40:], testPdf2TrueList[:3] + testPdf2TrueList[40:]]:
    #     for index, (modal, times) in enumerate(zip(modality, testTimesList[:3] + testTimesList[40:])):
    #         plt.plot(times, modal, label='%d' % index)
    #     plt.legend()
    #     plt.show()

    # Setup training data
    chunks = [10]*len(forcesList)
    labels = [True]*len(forcesList)
    trainDataSet = create_mvpa_dataset(forcesList, distancesList, anglesList, pdf2List, chunks, labels)
    trainTrueDataSet = create_mvpa_dataset(forcesTrueList, distancesTrueList, anglesTrueList, pdf2TrueList, chunks, labels)

    print trainDataSet.samples.shape
    forcesSample = trainDataSet.samples[:, 0, :]
    distancesSample = trainDataSet.samples[:, 1, :]
    anglesSample = trainDataSet.samples[:, 2, :]
    pdf2Sample = trainDataSet.samples[:, 3, :]
    forcesTrueSample = trainTrueDataSet.samples[:, 0, :]
    distancesTrueSample = trainTrueDataSet.samples[:, 1, :]
    anglesTrueSample = trainTrueDataSet.samples[:, 2, :]
    pdf2TrueSample = trainTrueDataSet.samples[:, 3, :]

    # print 'Forces Sample:', forcesSample[:, :5]
    # print 'Distances Sample:', distancesSample[:, :5]
    # print 'PDF1 Sample:', pdf1Sample[:, :5]
    # print 'PDF2 Sample:', pdf2Sample[:, :5]

    hmm = learning_hmm_multi_4d(nState=20, nEmissionDim=4)
    hmm.fit(xData1=forcesSample, xData2=distancesSample, xData3=anglesSample, xData4=pdf2Sample, ml_pkl='modals/ml_4d_angle.pkl', use_pkl=True)

    testSet = hmm.convert_sequence(forcesList[0], distancesList[0], anglesList[0], pdf2List[0])

    # print hmm.predict(testSet)
    print 'Log likelihood of testset:', hmm.loglikelihood(testSet)
    print '\nBeginning anomaly testing for nonanomalous training set\n'
    for i in xrange(len(forcesList)):
        print 'Anomaly Error for training set %d' % i
        print hmm.anomaly_check(forcesList[i], distancesList[i], anglesList[i], pdf2List[i], -5)

    print '\nBeginning anomaly testing for nonanomalous test set\n'
    for i in xrange(len(testForcesList)):
        if i == 10: print '\nBeginning anomaly testing for nonanomalous noise test set\n'
        elif i == 15 - 2: print '\nBeginning anomaly testing for nonanomalous new bowl location test set\n'
        elif i == 24 - 3: print '\nBeginning anomaly testing for anomalous test set\n'
        elif i == 30 - 3: print '\nBeginning anomaly testing for second anomalous test set\n'
        print 'Anomaly Error for test set %d' % i
        print hmm.anomaly_check(testForcesList[i], testDistancesList[i], testAnglesList[i], testPdf2List[i], -5)

    figName = os.path.join(os.path.dirname(__file__), 'plots/likelihood_success.png')
    hmm.likelihood_disp(forcesSample, distancesSample, anglesSample, pdf2Sample, forcesTrueSample, distancesTrueSample,
                        anglesTrueSample, pdf2TrueSample, -5.0, figureSaveName=None)

    # Find the largest iteration
    maxsize = max([len(x) for x in testForcesList])
    # Extrapolate each time step
    testForcesList, testDistancesList, testAnglesList, testPdf2List, \
        testForcesTrueList, testDistancesTrueList, testAnglesTrueList, testPdf2TrueList = extrapolateAllData([testForcesList, testDistancesList, testAnglesList, testPdf2List, testForcesTrueList, testDistancesTrueList, testAnglesTrueList, testPdf2TrueList], maxsize)
    testDataSet = create_mvpa_dataset(testForcesList, testDistancesList, testAnglesList, testPdf2List, [10]*len(testForcesList), [True]*len(testForcesList))
    forcesTestSample = testDataSet.samples[15:, 0, :]
    distancesTestSample = testDataSet.samples[15:, 1, :]
    anglesTestSample = testDataSet.samples[15:, 2, :]
    pdf2TestSample = testDataSet.samples[15:, 3, :]
    testTrueDataSet = create_mvpa_dataset(testForcesTrueList, testDistancesTrueList, testAnglesTrueList, testPdf2TrueList, [10]*len(testForcesList), [True]*len(testForcesList))
    forcesTrueTestSample = testTrueDataSet.samples[15:, 0, :]
    distancesTrueTestSample = testTrueDataSet.samples[15:, 1, :]
    anglesTrueTestSample = testTrueDataSet.samples[15:, 2, :]
    pdf2TrueTestSample = testTrueDataSet.samples[15:, 3, :]

    figName = os.path.join(os.path.dirname(__file__), 'plots/likelihood_anomaly.png')
    hmm.likelihood_disp(forcesTestSample, distancesTestSample, anglesTestSample, pdf2TestSample, forcesTrueTestSample, distancesTrueTestSample,
                        anglesTrueTestSample, pdf2TrueTestSample, -5.0, figureSaveName=None)

    # -- 1 dimensional force hidden Markov model --
    print '\n\nBeginning testing for 1 dimensional force hidden Markov model\n\n'

    hmm1d = learning_hmm_multi_1d(nState=20, nEmissionDim=1)
    hmm1d.fit(xData1=forcesSample, ml_pkl='modals/ml_1d_force.pkl', use_pkl=True)

    testSet = hmm1d.convert_sequence(forcesList[0])

    # print hmm.predict(testSet)
    print 'Log likelihood of testset:', hmm1d.loglikelihood(testSet)
    print '\nBeginning anomaly testing for nonanomalous training set\n'
    for i in xrange(len(forcesList)):
        print 'Anomaly Error for training set %d' % i
        print hmm1d.anomaly_check(forcesList[i], -5)

    print '\nBeginning anomaly testing for nonanomalous test set\n'
    for i in xrange(len(testForcesList)):
        if i == 10: print '\nBeginning anomaly testing for nonanomalous noise test set\n'
        elif i == 15 - 2: print '\nBeginning anomaly testing for nonanomalous new bowl location test set\n'
        elif i == 24 - 3: print '\nBeginning anomaly testing for anomalous test set\n'
        elif i == 30 - 3: print '\nBeginning anomaly testing for second anomalous test set\n'
        print 'Anomaly Error for test set %d' % i
        print hmm1d.anomaly_check(testForcesList[i], -5)

    figName = os.path.join(os.path.dirname(__file__), 'plots/likelihood_success.png')
    hmm1d.likelihood_disp(forcesSample, forcesTrueSample, -5.0, figureSaveName=None)


trainMultiHMM()

