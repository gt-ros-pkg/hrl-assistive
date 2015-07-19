#!/usr/bin/env python

import os
import numpy as np
import cPickle as pickle
from scipy import interpolate
from sklearn import preprocessing
from mvpa2.datasets.base import Dataset
from hmm.learning_hmm_multi_4d import learning_hmm_multi_4d

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

            # Find points within a sphere of radius 8 cm around the center of bowl
            nearbyPoints = np.linalg.norm(pointSet - bowlPosition, axis=1) < 0.08

            # Points near bowl
            points = pointSet[nearbyPoints]

            if len(points) <= 0:
                print 'ARGH, it happened on file', fileName

            # If no points found, try opening up to 10 cm
            if len(points) <= 0:
                # Find points within a sphere of radius 10 cm around the center of bowl
                nearbyPoints = np.linalg.norm(pointSet - bowlPosition, axis=1) < 0.10
                # Points near bowl
                points = pointSet[nearbyPoints]
                if len(points) <= 0:
                    print 'No points near bowl in dataset:', fileName
                    pdf.append(0)
                    continue

            # Try an exponential dropoff instead of Trivariate Gaussian Distribution
            # pdfValue = np.sum(np.exp(np.linalg.norm(points - bowlPosition, axis=1) * -10.0))
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

    for fileName, iterations in zip(fileNames, iterationSets):
        for i in iterations:
            name = fileName % i # Insert iteration value into filename
            forces, distances, angles, times = forceKinematics(name)
            pdf = visualFeatures(name, times)

            print minVals is None
            if minVals is None:
                minVals = []
                maxVals = []
                print 'Min and max values'
                for modality in [forces, distances, angles, pdf]:
                    print 'Min value', np.min(modality)
                    minVals.append(np.min(modality))
                    maxVals.append(np.max(modality))
                print 'minValues', minVals
                print 'maxValues', maxVals

            scale = 1

            # Scale features
            # forces = preprocessing.scale(forces) * scale
            # distances = preprocessing.scale(distances) * scale
            # angles = preprocessing.scale(angles) * scale
            # pdf = preprocessing.scale(pdf) * scale
            forces = scaling(forces, minVals[0], maxVals[0], scale)
            distances = scaling(distances, minVals[1], maxVals[1], scale)
            angles = scaling(angles, minVals[2], maxVals[2], scale)
            pdf = scaling(pdf, minVals[3], maxVals[3], scale)

            # print 'Forces shape:', forces.shape
            # print 'Distances shape:', distances.shape

            forcesList.append(forces.tolist())
            distancesList.append(distances.tolist())
            anglesList.append(angles.tolist())
            pdfList.append(pdf.tolist())
            timesList.append(times)

    # Each iteration may have a different number of time steps, so we extrapolate so they are all consistent
    if isTrainingData:
        # Find the largest iteration
        maxsize = max([len(x) for x in forcesList])
        # Extrapolate each time step
        forcesList, distancesList, anglesList, pdfList, timesList = extrapolateAllData([forcesList, distancesList, anglesList, pdfList, timesList], maxsize)

    return forcesList, distancesList, anglesList, pdfList, timesList

def setupMultiHMM():
    fileName = os.path.join(os.path.dirname(__file__), 'onlineData.pkl')

    if not os.path.isfile(fileName):
        print 'Loading training data'
        fileNames = ['/home/dpark/git/hrl-assistive/hrl_multimodal_anomaly_detection/recordings/bowlStage1Train_scooping_fvk_07-17-2015_16-03-36/iteration_%d_success.pkl',
                     '/home/dpark/git/hrl-assistive/hrl_multimodal_anomaly_detection/recordings/bowlStage2Train_scooping_fvk_07-17-2015_16-45-28/iteration_%d_success.pkl',
                     '/home/dpark/git/hrl-assistive/hrl_multimodal_anomaly_detection/recordings/bowlStage3Train_scooping_fvk_07-17-2015_17-13-56/iteration_%d_success.pkl',
                     '/home/dpark/git/hrl-assistive/hrl_multimodal_anomaly_detection/recordings/bowlStage1Test_scooping_fvk_07-17-2015_16-32-06/iteration_%d_success.pkl',
                     '/home/dpark/git/hrl-assistive/hrl_multimodal_anomaly_detection/recordings/bowlStage2Test_scooping_fvk_07-17-2015_16-53-13/iteration_%d_success.pkl',
                     '/home/dpark/git/hrl-assistive/hrl_multimodal_anomaly_detection/recordings/bowlStage3Test_scooping_fvk_07-17-2015_17-21-10/iteration_%d_success.pkl']
        iterationSets = [xrange(6), xrange(6), xrange(6), xrange(3), xrange(3), xrange(3)]
        forcesList, distancesList, anglesList, pdfList, timesList = loadData(fileNames, iterationSets, isTrainingData=True)

        with open(fileName, 'wb') as f:
            pickle.dump((forcesList, distancesList, anglesList, pdfList, timesList), f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(fileName, 'rb') as f:
            forcesList, distancesList, anglesList, pdfList, timesList = pickle.load(f)

    # Setup training data
    chunks = [10]*len(forcesList)
    labels = [True]*len(forcesList)
    trainDataSet = create_mvpa_dataset(forcesList, distancesList, anglesList, pdfList, chunks, labels)

    print trainDataSet.samples.shape
    forcesSample = trainDataSet.samples[:, 0, :]
    distancesSample = trainDataSet.samples[:, 1, :]
    anglesSample = trainDataSet.samples[:, 2, :]
    pdfSample = trainDataSet.samples[:, 3, :]

    hmm = learning_hmm_multi_4d(nState=20, nEmissionDim=4)
    hmm.fit(xData1=forcesSample, xData2=distancesSample, xData3=anglesSample, xData4=pdfSample, ml_pkl='../ml_4d_online.pkl', use_pkl=True)

    print 'minValues', minVals
    print 'maxValues', maxVals

    return hmm, minVals, maxVals

    # print '\nBeginning anomaly testing for nonanomalous training set\n'
    # for i in xrange(len(forcesList)):
    #     print 'Anomaly Error for training set %d' % i
    #     print hmm.anomaly_check(forcesList[i], distancesList[i], anglesList[i], pdfList[i], -5)
