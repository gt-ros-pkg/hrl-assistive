import operator
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

dbscan = DBSCAN(eps=0.12, min_samples=10)
fileName = '/home/zerickson/Downloads/rgbfun_scooping_07-06-2015_11-07-36/iteration_0_success.pkl'

xs = []
ys = []
zs = []
ts = []

with open(fileName, 'rb') as f:
    data = pickle.load(f)
    visual = data['visual_points']
    times = data['visual_time']
    # print visual
    i = 0
    for (pointSet, gripper, spoon), t in zip(visual, times):
        print 'Time:', t
        # Check for invalid points
        pointSet = pointSet[np.linalg.norm(pointSet, axis=1) < 5]

        # Use distance from gripper location
        pointSet = pointSet - gripper

        # Perform dbscan clustering
        X = StandardScaler().fit_transform(pointSet)
        labels = dbscan.fit_predict(X)

        # index, closePoint = min(enumerate(np.linalg.norm(pointSet - np.array(gripper), axis=1)), key=operator.itemgetter(1))
        index, closePoint = min(enumerate(np.linalg.norm(pointSet, axis=1)), key=operator.itemgetter(1))
        closeLabel = labels[index]
        while closeLabel == -1 and pointSet.size > 0:
            np.delete(pointSet, [index])
            np.delete(labels, [index])
            # index, closePoint = min(enumerate(np.linalg.norm(pointSet - np.array(gripper), axis=1)), key=operator.itemgetter(1))
            index, closePoint = min(enumerate(np.linalg.norm(pointSet, axis=1)), key=operator.itemgetter(1))
            closeLabel = labels[index]
        if pointSet.size <= 0:
            continue
        clusterPoints = pointSet[labels==closeLabel]
        nonClusterPoints = pointSet[labels!=closeLabel]

        xs = np.concatenate((xs, clusterPoints[:, 0]))
        ys = np.concatenate((ys, clusterPoints[:, 1]))
        zs = np.concatenate((zs, clusterPoints[:, 2]))
        ts = np.concatenate((ts, [t]*clusterPoints.shape[0]))
        if i >= 40:
            break
        i += 1

f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
# ax1.set_title('X axis')
# ax1.set_xlabel('time')
ax1.set_ylabel('x')
ax1.scatter(ts, xs, s=0.01, alpha=0.5)
# ax2.set_title('Y axis')
# ax2.set_xlabel('time')
ax2.set_ylabel('y')
ax2.scatter(ts, ys, s=0.01, alpha=0.5)
# ax3.set_title('Z axis')
ax3.set_xlabel('time')
ax3.set_ylabel('z')
ax3.scatter(ts, zs, s=0.01, alpha=0.5)

plt.show()
