#!/usr/bin/env python

import os, sys, copy

# visualization
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec
# util
import numpy as np

import itertools
colors = itertools.cycle(['r', 'b'])
shapes = itertools.cycle(['x','v', 'o', '+'])

sys.path.insert(0, '/usr/lib/pymodules/python2.7')
import svmutil as svm

def decision_boudnary(y_train, X_train, ml):
    h = 0.02

    # create a mesh to plot in
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    color_map = {0: (1, 1, 1), -1: (0, 0, .9), 2: (1, 0, 0), 1: (.8, .6, 0)}

    fig = plt.figure()            
    ax = fig.add_subplot(111)

    ## print np.shape(np.c_[xx.ravel(), yy.ravel()]), np.shape([0]*len(xx.ravel()))

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    Z, _, _ = svm.svm_predict([0]*len(xx.ravel()), np.c_[xx.ravel(), yy.ravel()].tolist(), ml)
    Z = np.array(Z)

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    plt.axis('off')

    # Plot also the training points
    colors = [color_map[y] for y in y_train]
    plt.scatter(X_train[:, 0], X_train[:, 1], c=colors, cmap=plt.cm.Paired)

    plt.xlim([-2, 8])
    plt.ylim([-2, 2])
    plt.show()



if __name__ == '__main__':

    s_x = np.array([[0.0, 0.0],
                    [2.0, 0.0],
                    [4.0, 0.0]])
    s_y = [-1,-1,-1]

    f_x = np.array([[1.0, 0.0],
                    [3.0, 0.0],
                    [5.0, 0.0]])
    f_y = [1,1,1]

    X   = np.vstack([s_x, f_x])
    y   = s_y + f_y

    print np.shape(X), np.shape(y)

    # train
    for gamma in [0.1, 0.5, 1.0]:
        for w_pos in np.linspace(0.1, 2.0, 4):
            commands = '-q -s 0 -t 2 -w-1 0.5 -w1 '+str(w_pos)+' -g '+str(gamma)+' -c 10.0'
            print commands
            ml = svm.svm_train(y, X.tolist(), commands )            
            decision_boudnary(y, X, ml)


    # viz
       
    ## color = colors.next()
    ## shape = shapes.next()    
    ## ax.scatter(s_x[:,0], s_x[:,1], marker=shape, c=color, s=400)
    
    ## color = colors.next()
    ## shape = shapes.next()    
    ## ax.scatter(f_x[:,0], f_x[:,1], marker=shape, c=color, s=400)



