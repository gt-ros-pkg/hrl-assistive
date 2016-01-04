#!/usr/bin/env python
#
# Copyright (c) 2014, Georgia Tech Research Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the Georgia Tech Research Corporation nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY GEORGIA TECH RESEARCH CORPORATION ''AS IS'' AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL GEORGIA TECH BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

#  \author Daehyung Park (Healthcare Robotics Lab, Georgia Tech.)

import sys

# visualization
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec

# util
import numpy as np
import random

from sklearn.svm import SVC

if __name__ == '__main__':

    # success
    x_s = []
    y_s = []
    for i in xrange(10):
        xx = np.linspace(0, np.pi, 100)
        yy = np.cos(xx)
        r = random.uniform(1.0, 3.0)
        for j in xrange(len(xx)):
            yy[j] += random.uniform(-0.1, 0.1)*r

        x_s.append(xx)
        y_s.append(yy)

    # failure
    x_f = []
    y_f = []
    for i in xrange(10):
        xx = np.linspace(0, np.pi, 100)
        yy = np.cos(xx/2.0)
        r = random.uniform(1.0, 3.0)
        for j in xrange(len(xx)):
            yy[j] += random.uniform(-0.1, 0.1)*r

        x_f.append(xx)
        y_f.append(yy)


    # Custom data (success)
    xx = np.linspace(0, np.pi, 100)
    yy = np.cos(xx/1.5)
    r = random.uniform(1.0, 3.0)
    for j in xrange(len(xx)):
        yy[j] += random.uniform(-0.1, 0.1)*r

    x_s.append(xx)
    y_s.append(yy)
    
    # Custom data (failure)
    xx = np.linspace(0, np.pi, 100)
    yy = np.cos(xx/1.5)
    r = random.uniform(1.0, 3.0)
    for j in xrange(len(xx)):
        yy[j] += random.uniform(-0.1, 0.1)*r

    x_f.append(xx)
    y_f.append(yy)
    #-------------------------------------

    X1 = np.array([x_s + x_f]).flatten()
    X2 = np.array([y_s + y_f]).flatten()
    X = np.vstack([X1,X2]).T
    Y = np.hstack([np.ones(len(np.array(x_s).flatten())), np.zeros(len(np.array(x_f).flatten()))])

    print np.shape(X), np.shape(Y)
    class_weight = {0: 1.0,
                    1: 5.0}
    
    clf = SVC(class_weight=class_weight)
    clf.fit(X,Y)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.array(x_s).T,np.array(y_s).T, 'bo')
    ax.plot(np.array(x_f).T,np.array(y_f).T, 'rx')
    
    # step size in the mesh
    h = .01

    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
    y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)

    print Z.min(), Z.max()

    ## plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.Blues_r)
    plt.contourf(xx, yy, Z, levels=[Z.min(),0], colors='white')
    plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='orange')
    plt.axis('off')

    plt.show()
