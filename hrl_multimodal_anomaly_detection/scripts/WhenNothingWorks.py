#!/usr/bin/env python

import numpy as np
import matplotlib.mlab
import cPickle as pickle
from sklearn import mixture
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.optimize import leastsq

import roslib
roslib.load_manifest('hrl_multimodal_anomaly_detection')
import tf


def generate_GMM(fileName, n=3, plot=False):
    mus = [[] for i in xrange(n)]
    stds = [[] for i in xrange(n)]
    ts = []

    with open(fileName, 'rb') as f:
        data = pickle.load(f)
        visual = data['visual_points']
        times = data['visual_time']
        bowl = data['bowl_position']
        bowl = np.array([x[0, 0] for x in bowl])
        for (pointSet, image, mic, spoon, (targetTrans, targetRot), (gripTrans, gripRot)), timeStamp in zip(visual, times):
            # cv2.imshow('Image window', image)
            # cv2.waitKey(200)

            # Transform mic and spoon into torso_lift_link
            targetMatrix = np.dot(tf.transformations.translation_matrix(targetTrans), tf.transformations.quaternion_matrix(targetRot))
            mic = np.dot(targetMatrix, np.array([mic[0], mic[1], mic[2], 1.0]))[:3]
            spoon = np.dot(targetMatrix, np.array([spoon[0], spoon[1], spoon[2], 1.0]))[:3]

            pointSet = np.c_[pointSet, np.ones(len(pointSet))]
            pointSet = np.dot(targetMatrix, pointSet.T).T[:, :3]

            # distances.append(np.linalg.norm(mic - bowl))
            # # Find angle between gripper-bowl vector and gripper-spoon vector
            # micSpoonVector = spoon - mic
            # micBowlVector = bowl - mic
            # angle = np.arccos(np.dot(micSpoonVector, micBowlVector) / (np.linalg.norm(micSpoonVector) * np.linalg.norm(micBowlVector)))
            # angles.append(angle)

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

            # Distances from points to mic location
            dists = np.linalg.norm(clusterPoints - mic, axis=1)

            # Transpose all points to gripper frame
            gripMatrix = np.dot(tf.transformations.translation_matrix(gripTrans), tf.transformations.quaternion_matrix(gripRot))
            clusterPoints = np.c_[clusterPoints, np.ones(len(clusterPoints))]
            clusterPoints = np.dot(gripMatrix, clusterPoints.T).T[:, :3]

            xs = clusterPoints[:, 0]
            ys = clusterPoints[:, 1]
            zs = clusterPoints[:, 2]
            ts.append(timeStamp)

            # Use distances from points to mic for calculating Gaussian Mixture Model
            dists = (dists - np.average(dists, axis=0)) / np.std(dists, axis=0)

            clf = mixture.GMM(n_components=n, covariance_type='full')
            clf.fit(dists)

            # print np.exp(clf.score(xs))

            if plot:
                # Plot the histogram.
                plt.hist(dists, bins=25, normed=True, alpha=0.6, color='g')
                # Plot the PDF.
                xmin, xmax = plt.xlim()
                x = np.linspace(xmin, xmax, 100)
                title = 'Gaussian Mixture Model for Distance Values'
                plt.title(title)

            # mu, std = norm.fit(xs)

            # Sort list based on means
            order = np.argsort(clf.means_.flatten())

            for index, (m, w, c) in enumerate(zip(clf.means_[order], clf.weights_[order], clf.covars_[order])):
                mu, std = m[0], np.sqrt(c)[0, 0]
                mus[index].append(mu)
                stds[index].append(std)

            if plot:
                for m, w, c in zip(clf.means_[order], clf.weights_[order], clf.covars_[order]):
                    # Fit a normal distribution to the data
                    mu, std = m[0], np.sqrt(c)[0, 0]
                    p = w * norm.pdf(x, mu, std)
                    plt.plot(x, p, linewidth=2)
                plt.show()
    return mus, stds, ts

mus, stds, ts = generate_GMM('/home/zerickson/Recordings/Ahh_scooping_fvk_07-09-2015_02-04-05/iteration_0_success.pkl', n=4)
mus2, stds2, ts2 = generate_GMM('/home/zerickson/Recordings/AHH_scooping_fvk_07-08-2015_21-46-41/iteration_0_success.pkl', n=4)

f, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2, sharey=True)
ax1.set_title('Mu\'s')
ax1.set_ylabel('Mu')
ax1.set_xlabel('Time')
print np.array(mus).shape
for mu in mus:
    print np.array(ts).shape, np.array(mu).shape
    ax1.plot(ts, mu, label='Gripper')

ax2.set_title('Standard Deviation\'s')
ax2.set_ylabel('Standard Deviation')
ax2.set_xlabel('Time')
for std in stds:
    ax2.plot(ts, std, label='Gripper')

ax3.set_title('Mu\'s')
ax3.set_ylabel('Mu')
ax3.set_xlabel('Time')
for mu in mus2:
    ax3.plot(ts2, mu, label='Gripper')

ax4.set_title('Standard Deviation\'s')
ax4.set_ylabel('Standard Deviation')
ax4.set_xlabel('Time')
for std in stds2:
    ax4.plot(ts2, std, label='Gripper')

plt.show()
