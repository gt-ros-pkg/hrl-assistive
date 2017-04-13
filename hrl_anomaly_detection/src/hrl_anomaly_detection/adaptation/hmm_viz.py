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

# system
## import rospy, roslib
import os, sys, copy
import random
import socket

# visualization
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec

# util
import numpy as np
import scipy
import hrl_lib.util as ut

from hrl_anomaly_detection.hmm import learning_hmm as hmm


def hmm_emission_viz(hmm1, hmm2):
    '''
    Plot the mean of emissions
    '''

    [A1, B1, pi1, out_a_num1, vec_num1, mat_num1, u_denom1] = hmm1.get_hmm_object()
    [A2, B2, pi2, out_a_num2, vec_num2, mat_num2, u_denom2] = hmm2.get_hmm_object()
    assert len(B1)==len(B2), "The number of states does not match."


    nState       = len(B1)
    nEmissionDim = len(B1[0][0])

    m1_list = []
    m2_list = []
    for i in xrange(nState):
        m1_list.append( B1[i][0] )
        m2_list.append( B2[i][0] )
    m1_list = np.array(m1_list)
    m2_list = np.array(m2_list)

    # display
    fig = plt.figure()

    for i in xrange(nEmissionDim):
        fig.add_subplot((nEmissionDim)*100+10+1+i)        
        plt.plot(m1_list[:,i], 'b-')
        plt.plot(m2_list[:,i], 'r-')

    plt.show()



def data_viz(X1, X2, raw_viz=False, minmax=None):
    '''
    Data comparizon
    @ X1: dim x samples x length
    @ X2: dim x samples x length
    '''
    assert len(X1)==len(X2), "The number of dimensions does not match."

    nDim = len(X1)
    t    = range(len(X1[0][0]))

    if minmax is not None:
        for i in xrange(len(X1)):
            X1[i] = np.array(X1[i])*(minmax[1][i]-minmax[0][i]) + minmax[0][i]
            X2[i] = np.array(X2[i])*(minmax[1][i]-minmax[0][i]) + minmax[0][i]
        

    # get mean and variance over time
    mu1_list  = []
    std1_list = []
    for i in xrange(nDim):
        mu1_list.append(np.mean(X1[i], axis=0))
        std1_list.append(np.std(X1[i], axis=0))
        
    mu2_list  = []
    std2_list = []
    for i in xrange(nDim):
        mu2_list.append(np.mean(X2[i], axis=0))
        std2_list.append(np.std(X2[i], axis=0))
    

    # display
    fig = plt.figure()

    for i in xrange(nDim):
        fig.add_subplot((nDim+1)*100+10+1+i)

        if raw_viz:
            plt.plot(X1[i].T, 'b-')
            plt.plot(X2[i].T, 'r-')
        else:
            plt.plot(mu1_list[i], 'b-')
            plt.fill_between(t, mu1_list[i]-std1_list[i], mu1_list[i]+std1_list[i],
                             facecolor='b', alpha=0.15, lw=0.0, interpolate=True)
        
            plt.plot(mu2_list[i], 'r-')
            plt.fill_between(t, mu2_list[i]-std2_list[i], mu2_list[i]+std2_list[i],
                             facecolor='r', alpha=0.15, lw=0.0, interpolate=True)


    plt.show()
