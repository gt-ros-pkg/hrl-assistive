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



methods = []
fpr_l   = []
tpr_l   = []

# RND (50.52)
methods.append('Random')
tpr_l.append([0.0, 0.0, 4.166666666666666, 4.166666666666666, 8.333333333333332, 12.5, 12.5, 16.666666666666664, 16.666666666666664, 20.833333333333336, 25.0, 25.0, 29.166666666666668, 33.33333333333333, 33.33333333333333, 37.5, 37.5, 41.66666666666667, 45.83333333333333, 45.83333333333333, 50.0, 50.0, 54.166666666666664, 58.333333333333336, 58.333333333333336, 62.5, 66.66666666666666, 66.66666666666666, 70.83333333333334, 70.83333333333334, 75.0, 79.16666666666666, 79.16666666666666, 83.33333333333334, 83.33333333333334, 87.5, 91.66666666666666, 91.66666666666666, 95.83333333333334, 100.0])
fpr_l.append([0.0, 0.0, 5.0, 5.0, 10.0, 10.0, 15.0, 15.0, 20.0, 20.0, 25.0, 25.0, 30.0, 30.0, 35.0, 35.0, 40.0, 40.0, 45.0, 45.0, 50.0, 50.0, 55.00000000000001, 55.00000000000001, 60.0, 60.0, 65.0, 65.0, 70.0, 70.0, 75.0, 75.0, 80.0, 80.0, 85.0, 85.0, 90.0, 90.0, 95.0, 100.0])


# AE (80.12)
methods.append('Autoencoder')
tpr_l.append([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 99.47916666666666, 96.875, 93.75, 92.1875, 90.10416666666666, 88.54166666666666, 83.33333333333334, 77.60416666666666, 71.875, 66.66666666666666, 55.729166666666664, 48.95833333333333, 40.10416666666667, 34.89583333333333, 30.208333333333332, 24.479166666666664, 21.875, 16.145833333333336, 15.104166666666666, 13.020833333333334, 13.020833333333334, 11.458333333333332, 8.854166666666668, 6.25])
fpr_l.append([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 98.75, 95.625, 83.75, 67.5, 50.0, 42.5, 34.375, 28.125, 16.875, 7.5, 4.375, 3.125, 1.25, 0.625, 0.625, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# EncDec-AD (80.75)
methods.append('EncDec-AD')
tpr_l.append([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 98.4375, 96.875, 94.79166666666666, 92.1875, 88.54166666666666, 84.89583333333334, 81.25, 78.64583333333334, 75.52083333333334, 69.27083333333334, 61.458333333333336, 54.166666666666664, 45.3125, 40.10416666666667, 28.645833333333332, 25.520833333333332, 22.395833333333336, 17.708333333333336, 13.020833333333334, 12.5, 10.9375, 8.854166666666668, 8.333333333333332, 7.291666666666667])
fpr_l.append([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 98.75, 94.375, 85.0, 65.0, 48.125, 42.5, 30.625000000000004, 23.125, 15.625, 11.25, 6.25, 3.125, 1.25, 0.625, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


# LSTM-DVAE after fine tunning no dyn ths alpha 1 AUC: 83.88
methods.append('LSTM-VAE')
tpr_l.append([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 98.4375, 97.91666666666666, 96.35416666666666, 94.27083333333334, 90.625, 84.375, 76.43979057591623, 71.35416666666666, 65.10416666666666, 57.8125, 50.0, 42.70833333333333, 37.5, 31.770833333333332, 23.958333333333336, 19.791666666666664, 16.145833333333336, 13.541666666666666, 13.020833333333334, 10.9375, 10.9375, 9.895833333333332, 8.333333333333332, 6.770833333333333])
fpr_l.append([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 99.375, 93.75, 85.625, 71.25, 55.625, 38.125, 25.624999999999996, 19.375, 12.5, 7.5, 1.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# LSTM-DVAE after fine tunning gamma 2.5 alpha 1, AUC: 87.19
methods.append(r'LSTM-VAE with $f_\eta$')
tpr_l.append([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 98.95833333333334, 97.39583333333334, 95.83333333333334, 90.625, 81.67539267015707, 73.4375, 67.70833333333334, 63.541666666666664, 52.879581151832454, 45.83333333333333, 41.66666666666667, 34.375, 29.166666666666668, 24.479166666666664, 19.270833333333336, 15.625, 14.0625, 13.020833333333334, 10.9375, 10.416666666666668, 9.895833333333332, 7.8125, 6.770833333333333])
fpr_l.append([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 99.375, 97.5, 94.375, 88.75, 75.0, 56.875, 41.25, 30.0, 21.25, 12.5, 5.0, 0.625, 0.625, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])



def roc(save_pdf=False):

    from sklearn import metrics
    #print "roc: ", metrics.auc(fpr_l, tpr_l, True)
    plt.style.use('ggplot')
    matplotlib.rcParams['text.usetex'] = True
    
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
    plt.ylabel('True positive rate [%]', fontsize=22)
    plt.xlabel('False positive rate [%]', fontsize=22)
    plt.xticks([0, 25, 50, 75, 100], fontsize=22)
    plt.yticks([0, 25, 50, 75, 100], fontsize=22)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    plt.legend(loc='lower right', prop={'size':18})

    if save_pdf:
        fig.savefig('test.pdf')
        fig.savefig('test.png')
        os.system('cp test.p* ~/Dropbox/HRL/')
    else:
        plt.show()


def roc_ths(save_pdf=False):

    from sklearn import metrics
    plt.style.use('ggplot')
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['legend.edgecolor'] = 'inherit'
    
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1,1,1)

    color = colors.next()
    shape = shapes.next()
    
    for i in xrange(len(fpr_l)-2,len(fpr_l)):
        
        plt.plot(fpr_l[i], tpr_l[i], '-'+shape+color, ms=6, mew=2, mec=color, label=methods[i])
        color = colors.next()
        shape = shapes.next()
        
    plt.xlim([-1,101])
    plt.ylim([-1,101])
    plt.ylabel('True positive rate [%]', fontsize=22)
    plt.xlabel('False positive rate [%]', fontsize=22)
    plt.xticks([0, 25, 50, 75, 100], fontsize=22)
    plt.yticks([0, 25, 50, 75, 100], fontsize=22)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    plt.legend(loc='lower right', prop={'size':18})


    if ax.legend_ <> None:
        lg = ax.legend_
        lg.get_frame().set_linewidth(0)
        lg.get_frame().set_alpha(0.5)

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
    p.add_option('--ths', action='store_true', dest='ths',
                 default=False, help='ROC.')
    opt, args = p.parse_args()


    if opt.roc:
        roc(save_pdf=opt.bSavePdf)
    if opt.ths:
        roc_ths(save_pdf=opt.bSavePdf)
    else:
        print "N/A"
    
    
