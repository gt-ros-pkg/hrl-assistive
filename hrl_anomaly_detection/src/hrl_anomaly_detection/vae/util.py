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

import numpy as np

def sampleWithWindow(X, window=5):
    '''
    X : sample x length x features
    return: (sample x length-window+1) x features
    '''
    if window < 1:
        print "Wrong window size"
        sys.exit()

    X_new = []
    for i in xrange(len(X)): # per sample
        for j in xrange(len(X[i])-window+1): # per time
            X_new.append( X[i][j:j+window].tolist() ) # per sample
    
    return X_new


def graph_variations(x_true, x_pred_mean, x_pred_std=None):
    '''
    x_true: timesteps x dim
    '''

    # visualization
    import matplotlib
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import gridspec
    import itertools
    colors = itertools.cycle(['g', 'm', 'c', 'k', 'y','r', 'b', ])
    shapes = itertools.cycle(['x','v', 'o', '+'])

    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42 

    
    nDim = len(x_true[0])
    
    fig = plt.figure(figsize=(6, 6))
    for k in xrange(nDim):
        fig.add_subplot(nDim,1,k+1)
        plt.plot(np.array(x_true)[:,k], '-b')
        plt.plot(np.array(x_pred_mean)[:,k], '-r')
        if x_pred_std is not None and len(x_pred_std)>0:
            plt.plot(np.array(x_pred_mean)[:,k]+np.array(x_pred_std)[:,k], '--r')
            plt.plot(np.array(x_pred_mean)[:,k]-np.array(x_pred_std)[:,k], '--r')
        plt.ylim([-0.1,1.1])
    plt.show()
