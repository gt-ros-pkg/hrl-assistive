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

    # HMM-D?
    methods.append('HMM-D')
    fpr_l.append([0,100])
    tpr_l.append([0,100])

    # LSTM-VAE
    methods.append('LSTM-VAE')
    fpr_l.append([0,100])
    tpr_l.append([0,100])

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

    plt.legend(loc='lower right', prop={'size':24})

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
    
    
