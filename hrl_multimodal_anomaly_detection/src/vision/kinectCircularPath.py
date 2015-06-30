__author__ = 'zerickson'

from math import sin, cos, pi
import time
import random
import numpy as np
from scipy import optimize
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

## Generates a circular path through a given set of points in 3d space.
# @param data - a list of 3d points in format [x, y, z]
# @param maxRadius - a float that represents the maximum expected radius (in meters) for the circular path
# @return radius, centerPoint, normal, error, synthetic
#   (radius is a float,
#   center point and normal vector are of form [x, y, z],
#   error is the average euclidean distance from data points to estimated circle,
#   synthetic is a list of points around the estimated circular path with components
#   found by 'circleX, circleY, circleZ = [x for x in zip(*synthetic)]')
def calcCircularPath(data, normal=None, maxRadius=10, verbose=False, plot=False, iteration=0):
    # A point p is a member of the plane if (p-p0).n0 = 0
    def distanceToPlane(p0, n0, p):
        return np.dot(np.array(n0), np.array(p) - np.array(p0))

    # Distance between points and estimated plane (residuals)
    def residualsPlane(parameters, dataPoint):
        if normal is None:
            px, py, pz, theta, phi = parameters
            nx, ny, nz = sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)
        else:
            px, py, pz = parameters
            nx, ny, nz = normal
        distances = [distanceToPlane([px, py, pz], [nx, ny, nz], [x, y, z]) for x, y, z in dataPoint]
        return distances

    # Find the best fit plane through the points
    xs, ys, zs = [[x[i] for x in data] for i in xrange(3)]
    if normal is None:
        estimate = [np.mean(xs), np.mean(ys), np.mean(zs), 0, 0]
        bestFitValues, ier = optimize.leastsq(residualsPlane, estimate, args=data)
        xF, yF, zF, tF, pF = bestFitValues
    else:
        estimate = [np.mean(xs), np.mean(ys), np.mean(zs)]
        bestFitValues, ier = optimize.leastsq(residualsPlane, estimate, args=data)
        xF, yF, zF = bestFitValues

    # Define a point in plane and the normal of plane
    point = [xF, yF, zF]
    if normal is None:
        normal = [sin(tF)*cos(pF), sin(tF)*sin(pF), cos(tF)]

    # Fitting a circle inside the plane by creating two in plane vectors
    # We assume that the normal is not parallel to x-axis
    # Tangent Vector
    tangent = np.cross(np.array([1, 0, 0]), np.array(normal))
    tangent = tangent / np.linalg.norm(tangent)
    # Binormal or Bitangent vector
    binormal = np.cross(tangent, np.array(normal))
    # Make sure second vector is normalized
    binormal = binormal / np.linalg.norm(binormal)

    def residualsCircle(parameters, dataPoint):
        r, s, Ri = parameters
        # Project points onto plane
        planePointArr = s*tangent + r*binormal + np.array(point)
        # Distance between projected points on plane and actual data points
        distance = [np.linalg.norm(planePointArr-np.array([x, y, z])) for x, y, z in dataPoint]
        # Residuals
        res = [(Ri-dist) for dist in distance]
        return res

    estimateCircle = [0, 0, 0]
    bestCircleFitValues, ier = optimize.leastsq(residualsCircle, estimateCircle, args=data)
    r, s, Ri = bestCircleFitValues
    planePointArr = s*tangent + r*binormal + np.array(point)
    error = np.mean([np.linalg.norm(planePointArr-np.array([x, y, z])) for x, y, z in data])
    # error = np.mean(residualsCircle(bestCircleFitValues, data))

    rF, sF, radius = bestCircleFitValues
    centerPoint = sF*tangent + rF*binormal + np.array(point)
    if verbose:
        print 'Radius:', radius, 'Center Point:', centerPoint, 'Error:', error

    # Verify that the approximated circle is close. If not, recreate a new path
    if (radius > maxRadius or any(c > maxRadius for c in centerPoint)) and iteration < 5:
        if verbose:
            print 'Unable to estimate a circle through data points. Attempting again.'
        # Slightly perturb a random data point in dataset
        i = random.randint(0, len(data) - 1)
        j = random.randint(0, 2)
        data[i][j] = data[i][j] * 1.05
        return calcCircularPath(data, maxRadius=maxRadius, verbose=verbose, plot=plot, iteration=iteration+1)
    elif iteration >= 5 and verbose:
        print 'Unable to converge to an adequate circular path.'

    # Synthetic Circle Data
    synthetic = [list(centerPoint + radius*cos(phi)*binormal+radius*sin(phi)*tangent) for phi in np.linspace(0, 2*pi, 50)]
    if plot:
        circleX, circleY, circleZ = [x for x in zip(*synthetic)]

        # Generate wireframe of plane through points
        d = - np.dot(np.array(point), np.array(normal))
        # Create plane mesh
        minX, maxX = np.min(circleX), np.max(circleX)
        minY, maxY = np.min(circleY), np.max(circleY)
        xx, yy = np.meshgrid(np.linspace(minX, maxX, 10), np.linspace(minY, maxY, 10))
        # Calculate corresponding z for mesh
        z = (-normal[0]*xx - normal[1]*yy - d)/normal[2]

        # Plot surface of plane, data, synthetic circle, and normal vector
        fig = plt.figure()
        ax = fig.add_subplot(211, projection='3d')
        # Plot surface of plane with data points (wireframe)
        ax.scatter(xs, ys, zs, c='b', marker='o')
        ax.plot_wireframe(xx, yy, z)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        bx = fig.add_subplot(212, projection='3d')
        # Plot data points and estimated circle
        bx.scatter(xs, ys, zs, c='b', marker='o')
        bx.plot(circleX, circleY, circleZ)
        # Plot the normal, binormal, and tangent vectors
        cX, cY, cZ = centerPoint
        bx.plot([cX, cX + normal[0]*radius], [cY, cY + normal[1]*radius], [cZ, cZ + normal[2]*radius], color='green')
        bx.plot([cX, cX + tangent[0]*radius], [cY, cY + tangent[1]*radius], [cZ, cZ + tangent[2]*radius], color='red')
        bx.plot([cX, cX + binormal[0]*radius], [cY, cY + binormal[1]*radius], [cZ, cZ + binormal[2]*radius], color='blue')
        bx.set_xlabel('X Label')
        bx.set_ylabel('Y Label')
        bx.set_zlabel('Z Label')
        # Force equal sized axes
        # bx.set_aspect('equal')
        # bx.set_xlim(0, 5)
        # bx.set_ylim(0, 5)
        # bx.set_zlim(0, 5)
        plt.show()

    return radius, centerPoint, normal, error, synthetic

if __name__ == '__main__':
    # Test data (Generate points on circle with random deviation)
    r = 1.5
    i, j, k = [3.1, 3.2, 3.3]
    binormal = np.array([1.0, 0.0, 1.0])
    binormal = binormal / np.linalg.norm(binormal)
    tangent = np.array([0.0, 1.0, 0.0])

    data = [list([i, j, k] + r*cos(phi)*binormal+r*sin(phi)*tangent + np.random.rand(3)/4) for phi in np.linspace(0, np.pi, 25)]

    # x = np.mgrid[-2:5:120j]
    # y = np.mgrid[1:9:120j]
    # z = np.mgrid[-5:3:120j]
    #
    # data = [list([x[i], y[i], z[i]] + np.random.rand(3)*0.4) for i in xrange(x.size)]

    startTime = time.time()
    vals = calcCircularPath(data, verbose=True, plot=True)
    print vals[3]
    print 'Execution time:', time.time() - startTime
