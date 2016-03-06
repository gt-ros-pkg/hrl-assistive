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

import os
import numpy as np

# util
import hrl_lib.util as ut
from six.moves import cPickle

# visualization
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec

import itertools
colors = itertools.cycle(['r', 'g', 'b', 'm', 'c', 'k', 'y'])
shapes = itertools.cycle(['x','v', 'o', '+'])


def save_params(obj, filename):
    f = file(filename, 'wb')
    cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()


def load_params(filename):
    if os.path.isfile(filename) is not True:
        print "Not existing file!!!!"
        return False

    f = open(filename, 'rb')
    obj = cPickle.load(f)
    f.close()
    ## obj = ut.load_pickle(filename)
    return obj


def RunAutoEncoder(X, filename, viz=False):
    import theano
    import theano.tensor as T
    import layer as l
    from theano import function, config, shared, sandbox

    mlp_features = load_params(filename)

    # Generate training features
    feature_list = []
    count = 0    
    for idx in xrange(0, len(X[0]), nSingleData):
        count += 1
        test_features = mlp_features( X[:,idx:idx+nSingleData].astype('float32') )
        feature_list.append(test_features)

    # Filter by variances
    feature_list = np.swapaxes(feature_list, 0,1)
    
    new_feature_list = []
    for i in xrange(len(feature_list)):

        all_std    = np.std(feature_list[i])
        ea_std     = np.std(feature_list[i], axis=0)
        avg_ea_std = np.mean(ea_std)

        if all_std > 0.2 and avg_ea_std < 0.2:
            new_feature_list.append(feature_list[i])

    if viz:
        n_cols = 2
        n_rows = int(len(feature_list)/2)        
        colors = itertools.cycle(['r', 'g', 'b', 'm', 'c', 'k', 'y'])
        
        #--------------------------------------------------------------
        fig1 = plt.figure(1)
        for i in xrange(len(feature_list)):
            n_col = int(i/n_rows)
            n_row = i%n_rows
            ax = fig1.add_subplot(n_rows,n_cols,i+1)
            color = colors.next()

            for j in xrange(len(feature_list[i])):
                ax.plot(feature_list[i][j,:], ':', c=color)

            ax.set_ylim([0,1])

        fig1.suptitle('Bottleneck features')

        #--------------------------------------------------------------
        fig2 = plt.figure(2)
        for i in xrange(len(new_feature_list)):
            n_col = int(i/n_rows)
            n_row = i%n_rows
            ax = fig2.add_subplot(n_rows,n_cols,i+1)
            color = colors.next()

            for j in xrange(len(new_feature_list[i])):
                ax.plot(new_feature_list[i][j,:], ':', c=color)

            ax.set_ylim([0,1])

        fig2.suptitle('Bottleneck low-variance features')

        plt.show()

    return new_feature_list

