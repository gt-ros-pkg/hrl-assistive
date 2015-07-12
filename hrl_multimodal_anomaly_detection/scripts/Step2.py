#!/usr/bin/env python

import numpy as np
import matplotlib.mlab
import cPickle as pickle
from sklearn import mixture
from scipy.stats import norm, entropy
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from scipy import interpolate
from sklearn.decomposition import PCA

import roslib
roslib.load_manifest('hrl_multimodal_anomaly_detection')
import tf


def generate_GMM(fileName, n=3, plot=False, testRBFs=None):
    # mus = [[] for i in xrange(n)]
    # stds = [[] for i in xrange(n)]
    mus = []
    stds = []
    ts = []
    distances = []
    angles = []
    pdfs = []
    rbfs = []
    linspace = None

    with open(fileName, 'rb') as f:
        data = pickle.load(f)
        visual = data['visual_points']
        times = data['visual_time']
        bowl = data['bowl_position']
        bowl = np.array([x[0, 0] for x in bowl])
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

            # Distances from points to mic location
            dists = np.linalg.norm(clusterPoints - mic, axis=1)

            # Transpose all points to gripper frame
            gripMatrix = np.dot(tf.transformations.translation_matrix(gripTrans), tf.transformations.quaternion_matrix(gripRot))
            clusterPoints = np.c_[clusterPoints, np.ones(len(clusterPoints))]
            clusterPoints = np.dot(gripMatrix, clusterPoints.T).T[:, :3]

            ts.append(timeStamp)

            # Define activation cells along spoon
            actCells = mic + [t*directionVector for t in np.linspace(0, 1, 9)]
            activations = []
            for index, cell in enumerate(actCells):
                # Cell activations defined by number of points in the cell
                pointsInCell = np.linalg.norm(pointSet - cell, axis=1) < 0.06
                count = sum([1 for v in pointsInCell if v])
                # Count the points in this activation cell and divide by 10
                activations = activations + [index] * int(count / 10.0)

            clf = mixture.GMM(n_components=3, covariance_type='full')
            clf.fit(activations)

            if plot:
                # Plot the histogram.
                plt.hist(activations, bins=9, normed=True, alpha=0.6, color='g')
                title = 'Gaussian Mixture Model for Distance Values'
                plt.title(title)

            # Setup bounds for probability density function
            if linspace is None:
                linspace = np.linspace(np.min(activations), np.max(activations), 100)

            # Sort list based on means
            order = np.argsort(clf.means_.flatten())
            pdfSet = []
            # Determine all (3) PDFs for activation cells
            for m, w, c in zip(clf.means_[order], clf.weights_[order], clf.covars_[order]):
                # Fit a normal distribution to the data
                mu, std = m[0], np.sqrt(c)[0, 0]
                # print mu, std
                pdf = w * norm.pdf(linspace, mu, std)
                pdfSet.append(pdf)
                if plot:
                    plt.plot(linspace, pdf, linewidth=2)
            pdfs.append(pdfSet)

            if plot:
                plt.show()

            # Radial Basis Function
            # xs = clusterPoints[:, 0]
            # ys = clusterPoints[:, 1]
            # zs = clusterPoints[:, 2]
            # d = np.ones(len(xs))
            # rbfs.append(interpolate.Rbf(xs, ys, zs, d))


            # pca = PCA(n_components=1)
            # pcaPoints = pca.fit(clusterPoints).transform(clusterPoints).flatten()
            # # Normalize points
            # # pcaPoints = (pcaPoints - np.average(pcaPoints, axis=0)) / np.std(pcaPoints, axis=0)
            #
            # # Fit a normal distribution to the data:
            # mu, std = norm.fit(pcaPoints)
            # if linspace is None:
            #     linspace = np.linspace(np.min(pcaPoints), np.max(pcaPoints), 100)
            # pdf = norm.pdf(linspace, mu, std)
            # # print pcaPoints
            # # print pdf
            # # exit()
            # pdfs.append(pdf)
            # mus.append(mu)
            # stds.append(std)
            #
            # if plot:
            #     # Plot the histogram.
            #     plt.hist(pcaPoints, bins=25, normed=True, alpha=0.6, color='g')
            #
            #     # Plot the PDF.
            #     xmin, xmax = plt.xlim()
            #     x = np.linspace(xmin, xmax, 100)
            #     p = norm.pdf(x, mu, std)
            #     plt.plot(x, p, 'k', linewidth=2)
            #     title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
            #     plt.title(title)
            #
            #     plt.show()

    return mus, stds, distances, angles, pdfs, ts, rbfs

print '-- Set of times one --'
mus1, stds1, distances1, angles1, successpdfs, ts1, rbfs1 = generate_GMM('/home/zerickson/Recordings/pinkSpoon_scooping_fvk_07-10-2015_18-30-38/iteration_0_success.pkl', n=1, plot=False)
print '-- Set of times two --'
mus2, stds2, distances2, angles2, successpdfs2, ts2, _ = generate_GMM('/home/zerickson/Recordings/pinkSpoon_scooping_fvk_07-10-2015_18-30-38/iteration_1_success.pkl', n=1, testRBFs=rbfs1)
print '-- Set of times three --'
mus3, stds3, distances3, angles3, failurepdfs, ts3, _ = generate_GMM('/home/zerickson/Recordings/pinkSpoon_scooping_fvk_07-10-2015_18-30-38/iteration_2_failure.pkl', n=1)
print len(ts1), len(ts2), len(ts3)

# Determine entropies when using activation cells
print 'Entropies for two successful trials'
entropies = []
for (pdf1, pdf2, pdf3), (pdf4, pdf5, pdf6) in zip(successpdfs, successpdfs2):
    ent1 = entropy(pdf1, pdf4)
    ent2 = entropy(pdf2, pdf5)
    ent3 = entropy(pdf3, pdf6)
    print ent1, ent2, ent3
    entropies.append(ent1 + ent2 + ent3)
print 'Average:', np.mean(entropies), 'Max:', np.max(entropies)

print 'Entropies for one successful and one failure trial'
entropies = []
for index, ((pdf1, pdf2, pdf3), (pdf4, pdf5, pdf6)) in enumerate(zip(successpdfs, failurepdfs)):
    ent1 = entropy(pdf1, pdf4)
    ent2 = entropy(pdf2, pdf5)
    ent3 = entropy(pdf3, pdf6)
    if np.isinf(ent1):
        print 'ent1 was infinity at index', index
        print pdf1
        print pdf4
    if np.isinf(ent2):
        print 'ent2 was infinity at index', index
        print pdf2
        print pdf5
    if np.isinf(ent3):
        print 'ent3 was infinity at index', index
        print pdf3
        print pdf6
    if np.isinf(ent1) or np.isinf(ent2) or np.isinf(ent3):
        exit()
    print 'Time:', ts1[index], 'Entropy:', ent1 + ent2 + ent3
    entropies.append(ent1 + ent2 + ent3)
print 'Average:', np.mean(entropies), 'Max:', np.max(entropies)

# Determine entropies when using PCA
# print 'Entropies for two successful trials'
# ent = []
# for pdf1, pdf2 in zip(successpdfs, successpdfs2):
#     ent.append(entropy(pdf1, pdf2))
#     # print entropy(pdf1, pdf2)
# print 'Average:', np.mean(ent), 'Max:', np.max(ent)
#
# print 'Entropies for one successful and one failure trial'
# ent = []
# for index, (pdf1, pdf2) in enumerate(zip(successpdfs, failurepdfs)):
#     ent.append(entropy(pdf1, pdf2))
#     print 'Time:', ts1[index], 'Entropy:', entropy(pdf1, pdf2)
# print 'Average:', np.mean(ent), 'Max:', np.max(ent)


# f, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2, sharey=True)
# ax1.set_title('Means of Two Gaussian Density Functions over Time\n(Trial 1)')
# ax1.set_ylabel('Mu (m)')
# ax1.set_xlabel('Time (sec)')
# print np.array(mus).shape
# for mu in mus:
#     print np.array(ts).shape, np.array(mu).shape
#     ax1.plot(ts, mu, label='Gripper')
#
# ax2.set_title('Standard Deviations of Two Gaussian Density Functions over Time\n(Trial 1)')
# ax2.set_ylabel('Standard Deviation (m)')
# ax2.set_xlabel('Time (sec)')
# for std in stds:
#     ax2.plot(ts, std, label='Gripper')
#
# ax3.set_title('Means of Two Gaussian Density Functions over Time\n(Trial 2)')
# ax3.set_ylabel('Mu (m)')
# ax3.set_xlabel('Time (sec)')
# for mu in mus2:
#     ax3.plot(ts2, mu, label='Gripper')
#
# ax4.set_title('Standard Deviations of Two Gaussian Density Functions over Time\n(Trial 2)')
# ax4.set_ylabel('Standard Deviation (m)')
# ax4.set_xlabel('Time (sec)')
# for std in stds2:
#     ax4.plot(ts2, std, label='Gripper')
#
# plt.show()


