__author__ = 'zackory'

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np


mu = np.array([40, 60])
sigma = np.array([[100, 30], [30, 140]])

sigmaInv = np.linalg.inv(sigma)
# coeff = 1 / (2*np.pi*np.sqrt(np.linalg.det(sigma)))
coeff = 1

p = [[np.exp(-0.5*np.dot(np.dot(([i,j]-mu).T, sigmaInv), ([i,j]-mu))) for j in xrange(100)] for i in xrange(100)]
p = np.array(p)*coeff


fig = plt.figure()
ax = fig.gca(projection='3d')
X = np.arange(0, 100, 1)
Y = np.arange(0, 100, 1)
X, Y = np.meshgrid(X, Y)
# cmap = cm.Blues
cmap = cm.cool
# cmap = cm.winter
# surf = ax.plot_surface(X, Y, p, rstride=1, cstride=1, cmap=cmap, linewidth=0, antialiased=False)
surf = ax.plot_surface(X, Y, p, rstride=2, cstride=2, cmap=cmap, alpha=0.4, linewidth=0, antialiased=False, edgecolors='#52b4f8')
# ax.set_zlim(-1.01, 1.01)

x = np.array([60, 30])
ax.plot([60, x[0]], [0, x[1]], alpha=0.8, c='k', linewidth=2)
ax.plot([100, x[0]], [30, x[1]], alpha=0.8, c='k', linewidth=2)
# Plot line between points
ax.plot([x[0], x[0]], [x[1], x[1]], zs=[0, p[x[1], x[0]]], alpha=0.8, c='k', linewidth=2)

# Plot point on 2D grid
ax.plot([x[0]], [x[1]], 'r.', alpha=0.7, markersize=20)
# Plot point on gaussian distribution
ax.plot([x[0]], [x[1]], 'g.', zs=[p[x[1], x[0]]], alpha=1, markersize=20)

ax.set_xlabel('X')
ax.set_ylabel('Y')
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
