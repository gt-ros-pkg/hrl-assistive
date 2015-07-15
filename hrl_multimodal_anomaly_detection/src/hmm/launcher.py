#!/usr/bin/env python

import numpy as np
import cPickle as pickle
from scipy import interpolate
import matplotlib.pyplot as plt
from sklearn import preprocessing
from mvpa2.datasets.base import Dataset
from learning_hmm_multi_3d import learning_hmm_multi_3d

import roslib
roslib.load_manifest('hrl_multimodal_anomaly_detection')
import tf

def launch(fileName):
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

        # There will be much more force data than kinematics, so interpolate to fill in the gaps
        distInterp = interpolate.splrep(visualTimes, distances, s=0)
        angleInterp = interpolate.splrep(visualTimes, angles, s=0)
        distances = interpolate.splev(forceTimes, distInterp, der=0)
        angles = interpolate.splev(forceTimes, angleInterp, der=0)

        return forces, distances, angles, forceTimes

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
    hmm = learning_hmm_multi_3d(nState=20, nEmissionDim=3)

    forcesList = []
    distancesList = []
    anglesList = []
    forcesTrueList = []
    distancesTrueList = []
    anglesTrueList = []
    timesList = []
    minList = []
    maxList = []
    for i in [0, 1, 3, 5, 6, 7, 8, 9]:
        fileName = '/home/zerickson/Recordings/trainingDataVer1_scooping_fvk_07-14-2015_11-06-33/iteration_%d_success.pkl' % i
        forces, distances, angles, times = launch(fileName)
        forcesTrueList.append(forces)
        distancesTrueList.append(distances)
        anglesTrueList.append(angles)
        scale = 100
        # forces, min_c1, max_c1 = hmm.scaling(forces, scale=scale)
        # distances, min_c2, max_c2 = hmm.scaling(distances, scale=scale)
        # angles, min_c3, max_c3 = hmm.scaling(angles, scale=scale)

        min_c1, max_c1 = np.min(forces), np.max(forces)
        min_c2, max_c2 = np.min(distances), np.max(distances)
        min_c3, max_c3 = np.min(angles), np.max(angles)
        # Scale features
        forces = preprocessing.scale(forces) * scale
        distances = preprocessing.scale(distances) * scale
        angles = preprocessing.scale(angles) * scale

        # print 'Forces shape:', forces.shape
        # print 'Distances shape:', distances.shape
        # print 'Angles shape:', angles.shape

        print any([f < 0 for f in forces])

        forcesList.append(forces)
        distancesList.append(distances)
        anglesList.append(angles)
        timesList.append(times)
        minList.append([min_c1, min_c2, min_c3])
        maxList.append([max_c1, max_c2, max_c3])
        # print minList
        # print maxList

        # print np.shape(forces), np.shape(distances), np.shape(angles)

    # Each training iteration may have a different number of time steps (align by chopping)
    # Find the smallest iteration
    minsize = min([len(x) for x in forcesList])
    # Drop extra time steps beyond minsize for each iteration
    forcesList = [x[:minsize] for x in forcesList]
    distancesList = [x[:minsize] for x in distancesList]
    anglesList = [x[:minsize] for x in anglesList]
    timesList = [x[:minsize] for x in timesList]

    # Plot modalities
    # for modality in [forcesList, distancesList, anglesList]:
    #     for index, (forces, times) in enumerate(zip(modality, timesList)):
    #         plt.plot(times, forces, label='%d' % index)
    #     plt.legend()
    #     plt.show()

    # Setup training data
    chunks = [10]*len(forcesList)
    labels = [True]*len(forcesList)
    trainDataSet = create_mvpa_dataset(forcesList, distancesList, anglesList, chunks, labels)
    trainTrueDataSet = create_mvpa_dataset(forcesTrueList, distancesTrueList, anglesTrueList, chunks, labels)

    print trainDataSet.samples.shape
    forcesSample = trainDataSet.samples[:,0,:]
    distancesSample = trainDataSet.samples[:,1,:]
    anglesSample = trainDataSet.samples[:,2,:]
    forcesTrueSample = trainTrueDataSet.samples[:,0,:]
    distancesTrueSample = trainTrueDataSet.samples[:,1,:]
    anglesTrueSample = trainTrueDataSet.samples[:,2,:]

    print 'Forces Sample:', forcesSample[:, :5]
    print 'Distances Sample:', distancesSample[:, :5]
    print 'Angles Sample:', anglesSample[:, :5]

    hmm.fit(xData1=forcesSample, xData2=distancesSample, xData3=anglesSample, use_pkl=True)

    testSet = hmm.convert_sequence(forcesList[0], distancesList[0], anglesList[0])

    # print hmm.predict(testSet)
    print 'Log likelihood of testset:', hmm.loglikelihood(testSet)
    for i in xrange(len(forcesList)):
        print 'Anomaly Error for training set %d' % i
        print hmm.anomaly_check(forcesList[i], distancesList[i], anglesList[i], -4)

    for ths in -1.0*np.arange(3, 5, 0.5):
        k = 4
        # chunks = [10]*len(forcesList[k])
        # labels = [True]*len(forcesList[k])
        # dataSet = create_mvpa_dataset(forcesList[k], distancesList[k], anglesList[k], chunks, labels)
        # forcesSample = dataSet.samples[:, 0]
        # distancesSample = dataSet.samples[:, 1]
        # anglesSample = dataSet.samples[:, 2]
        hmm.likelihood_disp(forcesSample, distancesSample, anglesSample, forcesTrueSample, distancesTrueSample, anglesTrueSample, ths,
                            scale1=[minList[k][0], maxList[k][0], 1], scale2=[minList[k][1], maxList[k][1], 1], scale3=[minList[k][2], maxList[k][2], 1])

trainMultiHMM()
