#!/usr/bin/env python

import numpy as np
import matplotlib.mlab
import cPickle as pickle
from sklearn import mixture, preprocessing
from scipy.stats import norm, entropy
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from scipy import interpolate
from sklearn.decomposition import PCA

import roslib
roslib.load_manifest('hrl_multimodal_anomaly_detection')
import tf

def generate_GMM(fileName, plot=False):
    ts = []
    pdfs = []
    linspace = None

    with open(fileName, 'rb') as f:
        data = pickle.load(f)
        visual = data['visual_points']
        times = data['visual_time']
        for (pointSet, image, mic, spoon, (targetTrans, targetRot), (gripTrans, gripRot)), timeStamp in zip(visual, times):
            # print 'Time:', timeStamp
            # cv2.imshow('Image window', image)
            # cv2.waitKey(200)

            # Transform mic and spoon into torso_lift_link
            targetMatrix = np.dot(tf.transformations.translation_matrix(targetTrans), tf.transformations.quaternion_matrix(targetRot))
            mic = np.dot(targetMatrix, np.array([mic[0], mic[1], mic[2], 1.0]))[:3]
            spoon = np.dot(targetMatrix, np.array([spoon[0], spoon[1], spoon[2], 1.0]))[:3]

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
            # Points outside of spoon radius
            # nonClusterPoints = pointSet[nearbyPoints == False]

            # Distances from points to mic location
            dists = np.linalg.norm(points - mic, axis=1)

            # Transpose all points to gripper frame
            # gripMatrix = np.dot(tf.transformations.translation_matrix(gripTrans), tf.transformations.quaternion_matrix(gripRot))
            # clusterPoints = np.c_[clusterPoints, np.ones(len(clusterPoints))]
            # clusterPoints = np.dot(gripMatrix, clusterPoints.T).T[:, :3]

            ts.append(timeStamp)

            # Scale all points to prevent division by small numbers and singular matrices
            spoon *= 20
            mic *= 20
            points *= 20
            directionVector = spoon - mic

            # Define receptive fields along spoon
            k = 9
            mu = mic + [t*directionVector for t in np.linspace(0, 1, k)]
            # sigma = receptFields[1] - receptFields[0]

            pdfList = []
            for i in xrange(k):
                n, m = points.shape
                sigma = np.zeros((m, m))
                # Compute covariances
                for h in xrange(m):
                    for j in xrange(m):
                        sigma[h, j] = 1.0/n * np.dot((points[:, h] - mu[i, h]).T, points[:, j] - mu[i, j])
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
                    pointMu = point - mu[i]
                    pdfValue += constant * np.exp(-1.0/2.0 * np.dot(np.dot(pointMu.T, sigmaInv), pointMu))
                    # print constant * np.exp(-1.0/2.0 * np.dot(np.dot(pointMu.T, sigmaInv), pointMu))
                pdfList.append(pdfValue)

            # print pdfList
            # pdfList = np.array(pdfList)
            # pdfList = (pdfList - np.average(pdfList, axis=0)) / np.std(pdfList, axis=0)
            # pdfList = preprocessing.scale(pdfList)
            # print pdfList
            pdfs.append(pdfList)

    return pdfs, ts

def trainEntropies():
    allPDFs = []
    for i in xrange(0, 10):
        pdfs, ts = generate_GMM('/home/zerickson/Recordings/trainingDataVer1_scooping_fvk_07-14-2015_11-06-33/iteration_%d_success.pkl' % i)
        allPDFs.append(pdfs)
    print allPDFs
    avgPDF = np.mean(allPDFs, axis=0)
    stdPDF = np.std(allPDFs, axis=0)
    print avgPDF
    print stdPDF

def testEntropies():
    # Test entropies
    pdfs, ts = generate_GMM('/home/zerickson/Recordings/trainingDataVer1_scooping_fvk_07-14-2015_11-06-33/iteration_0_success.pkl')
    for i in xrange(1, 10):
        # Determine entropies using receptive fields
        pdfs2, ts2 = generate_GMM('/home/zerickson/Recordings/trainingDataVer1_scooping_fvk_07-14-2015_11-06-33/iteration_%d_success.pkl' % i)
        print 'Entropies for two successful trials'
        entropies = []
        for index, (pdf1, pdf2) in enumerate(zip(pdfs, pdfs2)):
            ent = round(entropy(pdf1, pdf2) * 10, 3)
            entropies.append(ent)
            print 'Time:', round(ts[index], 3), 'Entropy:', ent
        print 'Average:', np.mean(entropies), 'Max:', np.max(entropies)

# testEntropies()
trainEntropies()

# print '-- Set of times one --'
# mus1, stds1, distances1, angles1, successpdfs, ts1, rbfs1 = generate_GMM('/home/zerickson/Recordings/trainingDataVer1_scooping_fvk_07-14-2015_11-06-33/iteration_0_success.pkl', n=1, plot=False)
# print '-- Set of times two --'
# mus2, stds2, distances2, angles2, successpdfs2, ts2, _ = generate_GMM('/home/zerickson/Recordings/pinkSpoon_scooping_fvk_07-10-2015_18-30-38/iteration_1_success.pkl', n=1, testRBFs=rbfs1)
# print '-- Set of times three --'
# mus3, stds3, distances3, angles3, failurepdfs, ts3, _ = generate_GMM('/home/zerickson/Recordings/pinkSpoon_scooping_fvk_07-10-2015_18-30-38/iteration_2_failure.pkl', n=1)
# print len(ts1), len(ts2), len(ts3)
#
# # Determine entropies using receptive fields
# print 'Entropies for two successful trials'
# entropies = []
# for index, (pdf1, pdf2) in enumerate(zip(successpdfs, successpdfs2)):
#     ent = round(entropy(pdf1, pdf2) * 10, 3)
#     entropies.append(ent)
#     print 'Time:', round(ts1[index], 3), 'Entropy:', ent
# print 'Average:', np.mean(entropies), 'Max:', np.max(entropies)
#
# print 'Entropies for one successful and one failure trial'
# entropies = []
# for index, (pdf1, pdf2) in enumerate(zip(successpdfs, failurepdfs)):
#     ent = round(entropy(pdf1, pdf2) * 10, 4)
#     entropies.append(ent)
#     print 'Time:', round(ts1[index], 3), 'Entropy:', ent
# print 'Average:', np.mean(entropies), 'Max:', np.max(entropies)

# Determine entropies when using activation cells
# print 'Entropies for two successful trials'
# entropies = []
# for i in xrange(len(successpdfs)):
#     if i >= len(successpdfs2):
#         break
#     ent = 0
#     for pdf1, pdf2 in zip(successpdfs[i], successpdfs2[i]):
#         ent += entropy(pdf1, pdf2)
#     if ent > 2 and i > 0:
#         ent2 = 0
#         for pdf1, pdf2 in zip(successpdfs[i-1], successpdfs2[i]):
#             ent2 += entropy(pdf1, pdf2)
#         if ent2 < ent:
#             ent = ent2
#     entropies.append(ent)
#     print 'Time:', ts1[i], 'Entropy:', ent
# print 'Average:', np.mean(entropies), 'Max:', np.max(entropies)
#
# print 'Entropies for one successful and one failure trial'
# entropies = []
# for i in xrange(len(successpdfs)):
#     if i >= len(successpdfs2):
#         break
#     ent = 0
#     for pdf1, pdf2 in zip(successpdfs[i], failurepdfs[i]):
#         ent += entropy(pdf1, pdf2)
#     if ent > 2 and i > 0:
#         ent2 = 0
#         for pdf1, pdf2 in zip(successpdfs[i-1], failurepdfs[i]):
#             ent2 += entropy(pdf1, pdf2)
#         if ent2 < ent:
#             ent = ent2
#     entropies.append(ent)
#     print 'Time:', ts1[i], 'Entropy:', ent
# print 'Average:', np.mean(entropies), 'Max:', np.max(entropies)

