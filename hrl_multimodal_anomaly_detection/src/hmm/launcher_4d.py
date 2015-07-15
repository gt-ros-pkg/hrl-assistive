#!/usr/bin/env python

import numpy as np
import cPickle as pickle
from scipy import interpolate
import matplotlib.pyplot as plt
from sklearn import preprocessing
from mvpa2.datasets.base import Dataset
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
        for pointSet, image, mic, spoon, (targetTrans, targetRot), (gripTrans, gripRot) in visual:
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
        pdf1 = []
        pdf2 = []
        for pointSet, image, mic, spoon, (targetTrans, targetRot), (gripTrans, gripRot) in visual:
            # Transform mic and spoon into torso_lift_link
            targetMatrix = np.dot(tf.transformations.translation_matrix(targetTrans), tf.transformations.quaternion_matrix(targetRot))
            mic = np.dot(targetMatrix, np.array([mic[0], mic[1], mic[2], 1.0]))[:3]
            spoon = np.dot(targetMatrix, np.array([spoon[0], spoon[1], spoon[2], 1.0]))[:3]

            # Transpose points into target frame
            pointSet = np.c_[pointSet, np.ones(len(pointSet))]
            pointSet = np.dot(targetMatrix, pointSet.T).T[:, :3]

            # Check for invalid points
            pointSet = pointSet[np.linalg.norm(pointSet, axis=1) < 5]

            # Determine a line between the gripper and spoon
            directionVector = spoon - mic
            linePoints = mic + [t*directionVector for t in np.linspace(0, 1, 5)]

            # Find points within a sphere of radius 6 cm around each point on the line
            nearbyPoints = None
            for linePoint in linePoints:
                pointsNear = np.linalg.norm(pointSet - linePoint, axis=1) < 0.06
                nearbyPoints = nearbyPoints + pointsNear if nearbyPoints is not None else pointsNear

            # Points near spoon
            points = pointSet[nearbyPoints]

            # Scale all points to prevent division by small numbers and singular matrices
            spoon *= 20
            mic *= 20
            points *= 20
            directionVector = spoon - mic

            # Define receptive fields along spoon
            k = 9 # 9 receptive fields
            mu = mic + [t*directionVector for t in np.linspace(0, 1, k)]
            # Use only the second and second to last receptive fields (near handle and spoon end)
            mu = mu[[1, -2]]

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
                # print 'New pdf evaluation'
                # print np.linalg.det(sigma)
                # print (2*np.pi)**m
                # print (2*np.pi)**m * np.linalg.det(sigma)
                constant = 1.0 / np.sqrt((2*np.pi)**m * np.linalg.det(sigma))
                sigmaInv = np.linalg.inv(sigma)
                pdfValue = 0
                # print constant
                # Evaluate the Probability Density Function for each point
                for point in points:
                    pointMu = point - muSet
                    pdfValue += constant * np.exp(-1.0/2.0 * np.dot(np.dot(pointMu.T, sigmaInv), pointMu))
                    # print 'Constant:', constant
                    # print 'Exponential:', np.exp(-1.0/2.0 * np.dot(np.dot(pointMu.T, sigmaInv), pointMu))
                    # print constant * np.exp(-1.0/2.0 * np.dot(np.dot(pointMu.T, sigmaInv), pointMu))
                pdfList.append(pdfValue)
            pdf1.append(pdfList[0])
            pdf2.append(pdfList[1])

        # There will be much more force data than vision, so perform constant interpolation to fill in the gaps
        tempPdf1 = []
        tempPdf2 = []
        visualIndex = 0
        for forceTime in forceTimes:
            if forceTime > visualTimes[visualIndex + 1] and visualIndex < len(visualTimes) - 2:
                visualIndex += 1
            tempPdf1.append(pdf1[visualIndex])
            tempPdf2.append(pdf2[visualIndex])
        pdf1 = tempPdf1
        pdf2 = tempPdf2

        # There will be much more force data than vision, so interpolate to fill in the gaps
        # pdf1Interp = interpolate.splrep(visualTimes, pdf1, s=0)
        # pdf2Interp = interpolate.splrep(visualTimes, pdf2, s=0)
        # pdf1 = interpolate.splev(forceTimes, pdf1Interp, der=0)
        # pdf2 = interpolate.splev(forceTimes, pdf2Interp, der=0)

        return pdf1, pdf2

def create_mvpa_dataset(aXData1, aXData2, aXData3, aXData4, chunks, labels):
    feat_list = []
    for x1, x2, x3, x4, chunk in zip(aXData1, aXData2, aXData3, aXData4, chunks):
        feat_list.append([x1, x2, x3, x4])

    data = Dataset(samples=feat_list)
    data.sa['id'] = range(0, len(labels))
    data.sa['chunks'] = chunks
    data.sa['targets'] = labels

    return data

def trainMultiHMM():
    hmm = learning_hmm_multi_4d(nState=20, nEmissionDim=4)

    forcesList = []
    distancesList = []
    # anglesList = []
    pdf1List = []
    pdf2List = []

    forcesTrueList = []
    distancesTrueList = []
    # anglesTrueList = []
    pdf1TrueList = []
    pdf2TrueList = []

    timesList = []
    minList = []
    maxList = []
    for i in [0, 1, 3, 5, 6, 7, 8, 9]:
        fileName = '/home/zerickson/Recordings/trainingDataVer1_scooping_fvk_07-14-2015_11-06-33/iteration_%d_success.pkl' % i
        forces, distances, angles, times = forceKinematics(fileName)
        pdf1, pdf2 = visualFeatures(fileName, times)
        forcesTrueList.append(forces)
        distancesTrueList.append(distances)
        # anglesTrueList.append(angles)
        pdf1TrueList.append(pdf1)
        pdf2TrueList.append(pdf2)

        scale = 100
        # forces, min_c1, max_c1 = hmm.scaling(forces, scale=scale)
        # distances, min_c2, max_c2 = hmm.scaling(distances, scale=scale)
        # angles, min_c3, max_c3 = hmm.scaling(angles, scale=scale)

        min_c1, max_c1 = np.min(forces), np.max(forces)
        min_c2, max_c2 = np.min(distances), np.max(distances)
        min_c3, max_c3 = np.min(pdf1), np.max(pdf1)
        min_c4, max_c4 = np.min(pdf2), np.max(pdf2)
        # min_c3, max_c3 = np.min(angles), np.max(angles)
        # Scale features
        forces = preprocessing.scale(forces) * scale
        distances = preprocessing.scale(distances) * scale
        pdf1 = preprocessing.scale(pdf1) * scale
        pdf2 = preprocessing.scale(pdf2) * scale
        # angles = preprocessing.scale(angles) * scale

        # print 'Forces shape:', forces.shape
        # print 'Distances shape:', distances.shape
        # print 'Angles shape:', angles.shape

        forcesList.append(forces)
        distancesList.append(distances)
        pdf1List.append(pdf1)
        pdf2List.append(pdf2)
        # anglesList.append(angles)
        timesList.append(times)
        minList.append([min_c1, min_c2, min_c3, min_c4])
        maxList.append([max_c1, max_c2, max_c3, max_c4])
        # print minList
        # print maxList

    # Each training iteration may have a different number of time steps (align by chopping)
    # Find the smallest iteration
    # minsize = min([len(x) for x in forcesList])
    # Drop extra time steps beyond minsize for each iteration
    # forcesList = [x[:minsize] for x in forcesList]
    # distancesList = [x[:minsize] for x in distancesList]
    # pdf1List = [x[:minsize] for x in pdf1List]
    # pdf2List = [x[:minsize] for x in pdf2List]
    # anglesList = [x[:minsize] for x in anglesList]
    # timesList = [x[:minsize] for x in timesList]

    # Plot modalities
    # for modality in [forcesList, distancesList, pdf1List, pdf2List]:
    #     for index, (forces, times) in enumerate(zip(modality, timesList)):
    #         plt.plot(times, forces, label='%d' % index)
    #     plt.legend()
    #     plt.show()

    # Setup training data
    chunks = [10]*len(forcesList)
    labels = [True]*len(forcesList)
    trainDataSet = create_mvpa_dataset(forcesList, distancesList, pdf1List, pdf2List, chunks, labels)
    trainTrueDataSet = create_mvpa_dataset(forcesTrueList, distancesTrueList, pdf1TrueList, pdf2TrueList, chunks, labels)

    print trainDataSet.samples.shape
    forcesSample = trainDataSet.samples[:, 0, :]
    distancesSample = trainDataSet.samples[:, 1, :]
    pdf1Sample = trainDataSet.samples[:, 2, :]
    pdf2Sample = trainDataSet.samples[:, 3, :]
    forcesTrueSample = trainTrueDataSet.samples[:, 0, :]
    distancesTrueSample = trainTrueDataSet.samples[:, 1, :]
    pdf1TrueSample = trainTrueDataSet.samples[:, 2, :]
    pdf2TrueSample = trainTrueDataSet.samples[:, 3, :]

    print 'Forces Sample:', forcesSample[:, :5]
    print 'Distances Sample:', distancesSample[:, :5]
    print 'PDF1 Sample:', pdf1Sample[:, :5]
    print 'PDF2 Sample:', pdf2Sample[:, :5]

    hmm.fit(xData1=forcesSample, xData2=distancesSample, xData3=pdf1Sample, xData4=pdf2Sample, use_pkl=True)

    testSet = hmm.convert_sequence(forcesList[0], distancesList[0], pdf1List[0], pdf2List[0])

    # print hmm.predict(testSet)
    print 'Log likelihood of testset:', hmm.loglikelihood(testSet)
    for i in xrange(len(forcesList)):
        print 'Anomaly Error for training set %d' % i
        print hmm.anomaly_check(forcesList[i], distancesList[i], pdf1List[i], pdf2List[i], -4)

    for ths in -1.0*np.arange(3, 5, 0.5):
        k = 0
        # chunks = [10]*len(forcesList[k])
        # labels = [True]*len(forcesList[k])
        # dataSet = create_mvpa_dataset(forcesList[k], distancesList[k], anglesList[k], chunks, labels)
        # forcesSample = dataSet.samples[:, 0]
        # distancesSample = dataSet.samples[:, 1]
        # anglesSample = dataSet.samples[:, 2]
        hmm.likelihood_disp(forcesSample, distancesSample, pdf1Sample, pdf2Sample, forcesTrueSample, distancesTrueSample,
                            pdf1TrueSample, pdf2TrueSample, ths, scale1=[minList[k][0], maxList[k][0], 1],
                            scale2=[minList[k][1], maxList[k][1], 1], scale3=[minList[k][2], maxList[k][2], 1],
                            scale4=[minList[k][3], maxList[k][3], 1])

trainMultiHMM()
