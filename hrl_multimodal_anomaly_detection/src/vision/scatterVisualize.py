#!/usr/bin/env python

import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt

import roslib
roslib.load_manifest('hrl_multimodal_anomaly_detection')
import tf

def plot3dPoints(fileName, title=None):
    xs = []
    ys = []
    zs = []
    ts = []
    distances = []
    angles = []

    with open(fileName, 'rb') as f:
        data = pickle.load(f)
        visual = data['visual_points']
        times = data['visual_time']
        bowl = data['bowl_position']
        # print np.array(visual).shape
        i = 0
        for (pointSet, mic, spoon, (targetTrans, targetRot), (gripTrans, gripRot)), timeStamp in zip(visual, times):
            print 'Time:', timeStamp
            # Transform mic and spoon into torso_lift_link
            targetMatrix = np.dot(tf.transformations.translation_matrix(targetTrans), tf.transformations.quaternion_matrix(targetRot))
            mic = np.dot(targetMatrix, np.array([mic[0], mic[1], mic[2], 1.0]))[:3]
            spoon = np.dot(targetMatrix, np.array([spoon[0], spoon[1], spoon[2], 1.0]))[:3]

            pointSet = np.c_[pointSet, np.ones(len(pointSet))]
            pointSet = np.dot(targetMatrix, pointSet.T).T[:, :3]

            distances.append(np.linalg.norm(mic - bowl))
            # Find angle between gripper-bowl vector and gripper-spoon vector
            micSpoonVector = spoon - mic
            micBowlVector = bowl - mic
            angle = np.arccos(np.dot(micSpoonVector, micBowlVector) / (np.linalg.norm(micSpoonVector) * np.linalg.norm(micBowlVector)))
            angles.append(angle)

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
            clusterPoints = pointSet[nearbyPoints]
            # Points outside of spoon radius
            # nonClusterPoints = pointSet[nearbyPoints == False]

            # Transpose all points to gripper frame
            gripMatrix = np.dot(tf.transformations.translation_matrix(gripTrans), tf.transformations.quaternion_matrix(gripRot))
            clusterPoints = np.c_[clusterPoints, np.ones(len(clusterPoints))]
            clusterPoints = np.dot(gripMatrix, clusterPoints.T).T[:, :3]

            xs = np.concatenate((xs, clusterPoints[:, 0]))
            ys = np.concatenate((ys, clusterPoints[:, 1]))
            zs = np.concatenate((zs, clusterPoints[:, 2]))
            ts = np.concatenate((ts, [timeStamp]*clusterPoints.shape[0]))
            # if i >= 2:
            #     break
            # i += 1

    return xs, ys, zs, ts, distances, angles

xs, ys, zs, ts, distances, angles = plot3dPoints('/home/zerickson/Recordings/RecordingScoopingTimes2DaehyungROXLOL_scooping_fvk_07-08-2015_14-17-42/iteration_0_success.pkl')
xss, yss, zss, tss, distances2, angles2 = plot3dPoints('/home/zerickson/Recordings/RecordingScoopingTimes2DaehyungROXLOL_scooping_fvk_07-08-2015_14-17-42/iteration_3_success.pkl')

f, ((ax1, ax4), (ax2, ax5), (ax3, ax6)) = plt.subplots(3, 2, sharex=True)
plt.subplots_adjust(hspace=0.1)
ax1.set_title('Position Plot Of Each Point On Spoon Throughout Time', y=1.1)
ax1.set_ylabel('x')
ax1.scatter(ts, xs, s=0.01, alpha=0.5)
ax2.set_ylabel('y')
ax2.scatter(ts, ys, s=0.01, alpha=0.5)
ax3.set_xlabel('time')
ax3.set_ylabel('z')
ax3.scatter(ts, zs, s=0.01, alpha=0.5)

ax4.set_title('Position Plot Of Each Point On Spoon Throughout Time - Failure Case', y=1.1)
ax4.set_ylabel('x')
ax4.scatter(tss, xss, s=0.01, alpha=0.5)
ax5.set_ylabel('y')
ax5.scatter(tss, yss, s=0.01, alpha=0.5)
ax6.set_xlabel('time')
ax6.set_ylabel('z')
ax6.scatter(tss, zss, s=0.01, alpha=0.5)

plt.show()
