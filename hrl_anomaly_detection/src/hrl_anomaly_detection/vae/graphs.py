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
import os, sys
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec
from matplotlib import rc

import itertools
colors = itertools.cycle(['g', 'm', 'c', 'k', 'y','r', 'b', ])
shapes = itertools.cycle(['x','v', 'o', '+'])
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42 


def roc(save_pdf=False):

    methods = []
    fpr_l   = []
    tpr_l   = []

    # RND
    methods.append('Random')
    tpr_l.append([0.0, 19.791666666666664, 30.729166666666668, 47.91666666666667, 72.39583333333334, 52.083333333333336, 70.83333333333334, 77.08333333333334, 83.33333333333334, 84.375, 87.5, 91.66666666666666, 98.4375, 90.625, 92.70833333333334, 100.0, 95.3125, 98.95833333333334, 98.95833333333334, 97.91666666666666, 99.47916666666666, 99.47916666666666, 100.0, 99.47916666666666, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    fpr_l.append([0.0, 18.125, 28.125, 47.5, 57.49999999999999, 61.25000000000001, 73.75, 78.125, 78.125, 80.0, 81.25, 91.25, 92.5, 93.75, 95.625, 95.625, 96.25, 96.875, 97.5, 98.125, 98.75, 99.375, 99.375, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])

    # OSVM
    methods.append('OSVM')
    tpr_l.append([44.27083333333333, 44.27083333333333, 44.27083333333333, 45.83333333333333, 46.35416666666667, 46.35416666666667, 44.27083333333333, 45.83333333333333, 45.83333333333333, 46.875, 47.91666666666667, 49.47916666666667, 53.645833333333336, 58.333333333333336, 60.9375, 60.9375, 60.416666666666664, 62.5, 65.10416666666666, 67.1875, 67.70833333333334, 69.27083333333334, 72.91666666666666, 78.64583333333334, 83.85416666666666, 90.625, 94.79166666666666, 98.95833333333334, 99.47916666666666, 99.47916666666666, 99.47916666666666, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    fpr_l.append([2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 3.125, 3.125, 3.125, 5.0, 5.625, 5.625, 6.875000000000001, 10.625, 13.750000000000002, 17.5, 19.375, 22.5, 26.25, 28.125, 28.125, 31.874999999999996, 39.375, 48.75, 56.875, 71.875, 81.25, 90.0, 97.5, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])


    

    # HMM-D?
    methods.append('HMM-D')
    tpr_l.append([18.75, 19.791666666666664, 20.833333333333336, 20.833333333333336, 21.354166666666664, 23.4375, 23.4375, 24.479166666666664, 26.5625, 27.604166666666668, 29.6875, 32.29166666666667, 34.375, 36.97916666666667, 41.66666666666667, 46.35416666666667, 50.520833333333336, 56.25, 59.895833333333336, 63.020833333333336, 67.70833333333334, 69.27083333333334, 71.875, 73.4375, 76.5625, 77.08333333333334, 78.125, 78.64583333333334, 81.25, 83.33333333333334, 85.9375, 86.45833333333334, 89.0625, 90.625, 91.66666666666666, 91.66666666666666, 93.75, 100.0, 100.0, 100.0])
    fpr_l.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.625, 1.25, 1.875, 2.5, 3.75, 5.0, 6.25, 6.875000000000001, 8.75, 10.625, 14.374999999999998, 16.875, 18.125, 25.0, 28.749999999999996, 31.874999999999996, 34.375, 36.875, 39.375, 43.75, 48.125, 55.625, 63.74999999999999, 71.875, 77.5, 81.25, 88.75, 100.0, 100.0, 100.0])

    ## # HMM-GP
    ## methods.append('HMM-GP')
    ## tpr_l.append([30.208333333333332, 32.8125, 34.375, 34.375, 36.97916666666667, 40.625, 42.1875, 45.83333333333333, 49.47916666666667, 51.5625, 54.6875, 56.770833333333336, 60.416666666666664, 63.541666666666664, 65.10416666666666, 69.27083333333334, 73.4375, 77.08333333333334, 80.20833333333334, 82.29166666666666, 84.375, 87.5, 88.54166666666666, 89.0625, 90.10416666666666, 91.14583333333334, 92.70833333333334, 92.70833333333334, 93.22916666666666, 93.75, 95.83333333333334, 96.35416666666666, 96.35416666666666, 96.35416666666666, 96.875, 97.39583333333334, 97.91666666666666, 97.91666666666666, 97.91666666666666, 97.91666666666666])
    ## fpr_l.append([0.0, 0.0, 0.0, 0.625, 1.875, 1.875, 2.5, 2.5, 3.125, 3.75, 3.75, 3.75, 5.625, 8.125, 9.375, 12.5, 18.125, 28.125, 34.375, 43.75, 52.5, 59.375, 71.25, 73.75, 80.625, 83.75, 88.125, 90.625, 91.25, 95.625, 97.5, 99.375, 99.375, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])


    ## # LSTM-VAE
    ## methods.append('LSTM-VAE(N=4)')
    ## tpr_l.append([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 99.47916666666666, 99.47916666666666, 98.4375, 96.875, 93.22916666666666, 89.0625, 79.6875, 69.79166666666666, 61.979166666666664, 59.895833333333336, 57.8125, 57.8125, 56.770833333333336, 53.645833333333336, 52.604166666666664, 50.520833333333336, 46.35416666666667, 44.27083333333333, 41.14583333333333, 38.02083333333333, 35.41666666666667, 28.645833333333332, 25.0, 22.395833333333336, 22.395833333333336, 21.875, 20.833333333333336, 18.229166666666664, 17.1875, 14.0625, 13.541666666666666, 13.541666666666666, 12.5, 11.458333333333332])
    ## fpr_l.append([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 98.125, 88.125, 72.5, 55.00000000000001, 38.75, 28.125, 20.0, 15.0, 12.5, 11.25, 10.0, 8.75, 7.5, 7.5, 7.5, 6.875000000000001, 6.875000000000001, 6.875000000000001, 6.25, 5.625, 5.625, 4.375, 3.75, 3.75, 3.125, 2.5, 1.25, 0.625, 0.625, 0.0, 0.0, 0.0, 0.0])

    # LSTM-VAE 8
    methods.append('LSTM-VAE')
    tpr_l.append([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 99.47916666666666, 97.39583333333334, 95.3125, 91.66666666666666, 84.89583333333334, 78.125, 70.83333333333334, 63.020833333333336, 59.375, 56.770833333333336, 56.770833333333336, 54.6875, 51.041666666666664, 46.35416666666667, 43.75, 39.58333333333333, 35.9375, 30.729166666666668, 23.4375, 22.916666666666664, 22.395833333333336, 20.833333333333336, 20.3125, 18.75, 17.708333333333336, 16.666666666666664, 16.666666666666664, 15.625])
    fpr_l.append([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 99.375, 95.625, 83.75, 66.875, 48.75, 31.25, 17.5, 14.374999999999998, 11.875, 10.625, 10.0, 8.125, 6.875000000000001, 6.25, 6.25, 6.25, 6.25, 3.75, 3.75, 2.5, 2.5, 1.875, 1.25, 0.625, 0.625, 0.625, 0.625, 0.0, 0.0])

    # Computation

    from sklearn import metrics
    #print "roc: ", metrics.auc(fpr_l, tpr_l, True)  
    fig = plt.figure(figsize=(6,6))
    fig.add_subplot(1,1,1)

    color = colors.next()
    shape = shapes.next()
    
    for i in xrange(len(fpr_l)):
        plt.plot(fpr_l[i], tpr_l[i], '-'+shape+color, ms=6, mew=2, mec=color, label=methods[i])
        color = colors.next()
        shape = shapes.next()
        
    plt.xlim([-1,101])
    plt.ylim([-1,101])
    plt.ylabel('True positive rate (percentage)', fontsize=22)
    plt.xlabel('False positive rate (percentage)', fontsize=22)
    plt.xticks([0, 50, 100], fontsize=22)
    plt.yticks([0, 50, 100], fontsize=22)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    plt.legend(loc='lower right', prop={'size':18})

    if save_pdf:
        fig.savefig('test.pdf')
        fig.savefig('test.png')
        os.system('cp test.p* ~/Dropbox/HRL/')
    else:
        plt.show()


if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    p.add_option('--savepdf', '--sp', action='store_true', dest='bSavePdf',
                 default=False, help='Save pdf files.')    
    p.add_option('--roc', action='store_true', dest='roc',
                 default=False, help='ROC.')
    opt, args = p.parse_args()


    if opt.roc:
        roc(save_pdf=opt.bSavePdf)
    else:
        print "N/A"
    
    