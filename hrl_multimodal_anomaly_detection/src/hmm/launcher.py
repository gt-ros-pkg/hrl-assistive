#!/usr/bin/env python

import numpy as np
import cPickle as pickle
from sklearn import preprocessing
from mvpa2.datasets.base import Dataset
from learning_hmm_multi_3d import learning_hmm_multi_3d

import roslib
roslib.load_manifest('hrl_multimodal_anomaly_detection')
import tf

def launch(fileName):
    dataPoints = []
    distances = []
    angles = []
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

        print forces.shape

        # Compute kinematic distances and angles
        for (pointSet, image, mic, spoon, (targetTrans, targetRot), (gripTrans, gripRot)), timeStamp in zip(visual, visualTimes):
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

        # We need to align the force and visual data. This does such
        tempVisualTimes = []
        tempDistances = []
        tempAngles = []
        visualIndex = 0
        for forceTime in forceTimes:
            if forceTime > visualTimes[visualIndex + 1] and visualIndex < len(visualTimes) - 2:
                visualIndex += 1
            tempVisualTimes.append(visualTimes[visualIndex])
            tempDistances.append(distances[visualIndex])
            tempAngles.append(angles[visualIndex])
        distances = tempDistances
        angles = tempAngles

        # Create set of data points for the hidden Markov model
        # dataPoints = np.array([[f, d, a] for f, d, a in zip(forces, distances, angles)])

        return forces, distances, angles

def create_mvpa_dataset(aXData1, aXData2, aXData3, chunks, labels):
    feat_list = []
    for x1, x2, x3, chunk in zip(aXData1, aXData2, aXData3, chunks):
        feat_list.append([x1, x2, x3])

    data = Dataset(samples=feat_list)
    data.sa['id'] = range(0, len(labels))
    data.sa['chunks'] = chunks
    data.sa['targets'] = labels

    return data

def trainMultiHMM():
    forcesList = []
    distancesList = []
    anglesList = []
    for i in xrange(3):
        fileName = '/home/zerickson/Recordings/trainingDataVer1_scooping_fvk_07-14-2015_11-06-33/iteration_%d_success.pkl' % i
        forces, distances, angles = launch(fileName)
        scale = 0.6
        # forces = np.array(forces) * scale
        # distances = np.array(distances) * scale
        # angles = np.array(angles) * scale
        # # Scale features
        forces = preprocessing.scale(forces) * scale
        distances = preprocessing.scale(distances) * scale
        angles = preprocessing.scale(angles) * scale

        # print 'Forces shape:', forces.shape
        # print 'Distances shape:', distances.shape
        # print 'Angles shape:', angles.shape

        forcesList.append(forces.tolist())
        distancesList.append(distances.tolist())
        anglesList.append(angles.tolist())

        # print np.shape(forces), np.shape(distances), np.shape(angles)

    # Each training iteration may have a different number of time steps (align by chopping)
    # Find the smallest iteration
    minsize = min([len(x) for x in forcesList])
    # Drop extra time steps beyond minsize for each iteration
    forcesList = [x[:minsize] for x in forcesList]
    distancesList = [x[:minsize] for x in distancesList]
    anglesList = [x[:minsize] for x in anglesList]

    # Setup training data
    chunks = [10]*len(forcesList)
    labels = [True]*len(forcesList)
    trainDataSet = create_mvpa_dataset(forcesList, distancesList, anglesList, chunks, labels)

    print trainDataSet.samples.shape
    forcesSample = trainDataSet.samples[:,0,:]
    distancesSample = trainDataSet.samples[:,1,:]
    anglesSample = trainDataSet.samples[:,2,:]

    print 'Forces Sample:', forcesSample[:, :5]
    print 'Distances Sample:', distancesSample[:, :5]
    print 'Angles Sample:', anglesSample[:, :5]

    hmm = learning_hmm_multi_3d(nState=6, nEmissionDim=3)

    hmm.fit(xData1=forcesSample, xData2=distancesSample, xData3=anglesSample)

    # testSet = hmm.convert_sequence(forcesList[0], distancesList[0], anglesList[0])

    # print hmm.predict(testSet)
    print hmm.anomaly_check(forcesList[0], distancesList[0], anglesList[0], 1)
    # print hmm.score(test_seq)

trainMultiHMM()
