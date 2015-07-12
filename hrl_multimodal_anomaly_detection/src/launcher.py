#!/usr/bin/env python

import numpy as np
import cPickle as pickle
from learning_hmm import learning_hmm

import roslib
roslib.load_manifest('hrl_multimodal_anomaly_detection')
import tf

hmm = learning_hmm(aXData=None, nState=6, nMaxStep=15)

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
            print 'Time:', timeStamp
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
        dataPoints = np.array([[f, d, a] for f, d, a in zip(forces, distances, angles)])

        return dataPoints


fileName = '/home/zerickson/Recordings/pinkSpoon_scooping_fvk_07-10-2015_18-30-38/iteration_0_success.pkl'

dataPoints = launch(fileName)

print dataPoints.shape, dataPoints.dtype

hmm.fit(dataPoints)

# print len(test_seq)
#
# print hmm.score(test_seq)
