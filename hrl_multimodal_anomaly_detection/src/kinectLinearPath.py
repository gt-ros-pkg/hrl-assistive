__author__ = 'zerickson'

from math import sin, cos
import time
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as m3d

## Generates a circular path through a given set of points in 3d space.
# @param data - a list of 3d points in format [x, y, z]
# @param maxRadius - a float that represents the maximum expected radius (in meters) for the circular path
# @return endPoints, error
#   (endPoints consists of two endpoints in [x, y, z] format that define the line,
#   error is the average euclidean distance from data points to the best fit line)
def calcLinearPath(data, verbose=False, plot=False):
    # Calculate the mean of the points, the 'center' of the cloud
    # datamean = data.mean(axis=0)
    datamean = np.mean(data, axis=0)

    # Do an SVD on the mean-centered data.
    uu, dd, vv = np.linalg.svd(data - datamean)

    error = sum(dd)

    # Find farthest and closest data points to origin and add the distances
    distances = [np.sqrt(x[0]**2 + x[1]**2 + x[2]**2) for x in data]
    far = np.max(distances)
    close = np.min(distances)
    spread = (far + close)/2.0

    # Generate the endpoints of the best fit line
    endPoints = vv[0] * np.mgrid[-spread:spread:2j][:, np.newaxis]

    # shift by the mean to get the line in the right place
    endPoints += datamean
    if verbose:
        print 'End points:', endPoints
        print 'Error:', error

    if plot:
        ax = m3d.Axes3D(plt.figure())
        # Plot data points
        ax.scatter3D(*data.T)
        # Plot best fit line
        ax.plot3D(*endPoints.T)
        plt.show()

    return endPoints, error


# Test data (Generate points on circle with random deviation)
# r = 1.5
# i, j, k = [3.1, 3.2, 3.3]
# binormal = np.array([1.0, 0.0, 1.0])
# binormal = binormal / np.linalg.norm(binormal)
# tangent = np.array([0.0, 1.0, 0.0])
#
# x = np.array([i + (r*cos(phi)*binormal+r*sin(phi)*tangent)[0] + np.random.rand()/4 for phi in np.linspace(0, np.pi, 25)])
# y = np.array([j + (r*cos(phi)*binormal+r*sin(phi)*tangent)[1] + np.random.rand()/4 for phi in np.linspace(0, np.pi, 25)])
# z = np.array([k + (r*cos(phi)*binormal+r*sin(phi)*tangent)[2] + np.random.rand()/4 for phi in np.linspace(0, np.pi, 25)])

# x = np.mgrid[-2:5:120j]
# y = np.mgrid[1:9:120j]
# z = np.mgrid[-5:3:120j]

# data = np.concatenate((x[:, np.newaxis],
#                        y[:, np.newaxis],
#                        z[:, np.newaxis]),
#                       axis=1)

# Perturb with some Gaussian noise
# data += np.random.normal(size=data.shape) * 0.4

# startTime = time.time()
# print calcLinearPath(data, verbose=True, plot=True)
# print 'Execution time:', time.time() - startTime
