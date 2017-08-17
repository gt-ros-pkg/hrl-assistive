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


    # EncDec-AD (81.34)
    methods.append('EncDec-AD')
    tpr_l.append([])
    fpr_l.append([])


    # LSTM-DVAE after fine tunning gamma 0.5 alpha 1, AUC: 83.12
    methods.append('LSTM-DVAE')
    tpr_l.append([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 99.47916666666666, 98.4375, 98.4375, 97.91666666666666, 97.39583333333334, 96.35416666666666, 92.70833333333334, 90.10416666666666, 86.45833333333334, 83.85416666666666, 79.6875, 75.0, 70.3125, 66.14583333333334, 61.979166666666664, 55.729166666666664, 51.041666666666664, 44.79166666666667, 40.625, 35.41666666666667, 30.729166666666668, 28.125, 26.041666666666668, 23.4375, 21.354166666666664, 18.75, 17.708333333333336, 17.1875])
    fpr_l.append([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 98.125, 96.25, 95.0, 91.875, 88.75, 79.375, 67.5, 58.12500000000001, 43.125, 34.375, 26.25, 20.625, 18.75, 15.0, 11.25, 8.75, 5.625, 4.375, 3.75, 3.75, 1.25, 0.625, 0.625, 0.625, 0.0, 0.0, 0.0, 0.0])



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
    
    
