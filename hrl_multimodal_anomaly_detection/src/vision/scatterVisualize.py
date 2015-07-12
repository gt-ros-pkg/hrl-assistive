#!/usr/bin/env python

import cv2
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import roslib
roslib.load_manifest('hrl_multimodal_anomaly_detection')
import tf

# listener = tf.TransformListener()
# help(listener)

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
        bowl = np.array([x[0, 0] for x in bowl])
        # print np.array(visual).shape
        i = 0
        for (pointSet, image, mic, spoon, (targetTrans, targetRot), (gripTrans, gripRot)), timeStamp in zip(visual, times):
            print 'Time:', timeStamp

            # cv2.imshow('Image window', image)
            # cv2.waitKey(200)

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

            pca = PCA(n_components=1)
            pcaPoints = pca.fit(clusterPoints).transform(clusterPoints)

            xs = np.concatenate((xs, pcaPoints.flatten()))
            ys = np.concatenate((ys, clusterPoints[:, 1]))
            zs = np.concatenate((zs, clusterPoints[:, 2]))
            ts = np.concatenate((ts, [timeStamp]*clusterPoints.shape[0]))
            # if i >= 2:
            #     break
            # i += 1

    return xs, ys, zs, ts, distances, angles, times

xs, ys, zs, ts, distances, angles, times = plot3dPoints('/home/zerickson/Recordings/pinkSpoon_scooping_fvk_07-10-2015_18-30-38/iteration_0_success.pkl')
xss, yss, zss, tss, distances2, angles2, times2 = plot3dPoints('/home/zerickson/Recordings/pinkSpoon_scooping_fvk_07-10-2015_18-30-38/iteration_2_failure.pkl')

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

f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
plt.subplots_adjust(hspace=0.1)
ax1.set_title('Position Plot Of Each Point On Spoon Throughout Time', y=1.1)
ax1.set_ylabel('x')
ax1.scatter(ts, xs, s=0.01, alpha=0.5)
ax1.scatter(tss, xss, s=0.01, color='r', alpha=0.5)
ax2.set_ylabel('y')
ax2.scatter(ts, ys, s=0.01, alpha=0.5)
ax2.scatter(tss, yss, s=0.01, color='r', alpha=0.5)
ax3.set_xlabel('time')
ax3.set_ylabel('z')
ax3.scatter(ts, zs, s=0.01, alpha=0.5)
ax3.scatter(tss, zss, s=0.01, color='r', alpha=0.5)

plt.show()

f, (ax1, ax2) = plt.subplots(2, sharex=True)
ax1.plot(times, distances)
ax1.plot(times2, distances2)
ax1.set_xlabel('time')
ax1.set_ylabel('distance')

ax2.plot(times, angles)
ax2.plot(times2, angles2)
ax2.set_xlabel('time')
ax2.set_ylabel('angle')

plt.show()
