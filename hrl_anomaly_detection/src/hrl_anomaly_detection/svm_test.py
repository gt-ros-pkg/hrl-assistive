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



if __name__ == '__main__':

    s_x = np.array([[0.0, 0.0],
                    [2.0, 0.0],
                    [4.0, 0.0]])
    s_y = [-1,-1,-1]

    f_x = np.array([[1.0, 0.0],
                    [3.0, 0.0],
                    [5.0, 0.0]])
    f_y = [1,1,1]


    sys.path.insert(0, '/usr/lib/pymodules/python2.7')
    import svmutil as svm

    

    
    

    import itertools
    colors = itertools.cycle(['r', 'b'])
    shapes = itertools.cycle(['x','v', 'o', '+'])

    # viz
    fig = plt.figure()            
    ax = fig.add_subplot(111)

    color = colors.next()
    shape = shapes.next()    
    ax.scatter(s_x[:,0], s_x[:,1], marker=shape, c=color, s=400)
    
    color = colors.next()
    shape = shapes.next()    
    ax.scatter(f_x[:,0], f_x[:,1], marker=shape, c=color, s=400)


    plt.xlim([-2, 8])
    plt.ylim([-2, 2])
    plt.show()

