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

# system & utils
import os, sys, copy, random
import numpy as np
import scipy
import hrl_lib.util as ut
from joblib import Parallel, delayed

# Private utils
from hrl_anomaly_detection.util import *
from hrl_anomaly_detection.util_viz import *
from hrl_anomaly_detection import data_manager as dm
from hrl_anomaly_detection import util as util
from hrl_execution_monitor import util as autil

# Private learners
from hrl_anomaly_detection.hmm import learning_hmm as hmm
import hrl_anomaly_detection.classifiers.classifier as cf
import hrl_anomaly_detection.data_viz as dv

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
random.seed(3334)
np.random.seed(3334)



def lr_auc_graph(save_pdf=False):

    # max iter: 10

    # Adaptation (max_iter=10)
    lr_list  = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    auc_list = np.array([7505.1666578347731, 7382.7566106724616, 7469.3974882094226, 7531.1324254146566, 7676.8586720364583, 7629.9613163054428, 7932.2770388426679, 7774.6277356790843, 7967.2513380318996, 7974.6701287690103, 7882.7301149912564, 8076.4135445922311])/100.0 # nptrain: 10 w/ adaptive threshold
    ## auc_list = np.array([6538.8691643262146, 6538.8691643262146, 6538.8691643262146, 6538.8691643262146, 6538.8691643262146, 6538.8691643262146, 6538.8691643262146, 6538.8691643262146, 6538.8691643262146, 6538.8691643262146, 6538.8691643262146, 6538.8691643262146])/100.0

    
    auc_list = np.array([7505.7275055028967, 7452.7200035937294, 7478.9991464893765, 7562.3287363550598, 7591.527784016891, 7575.5806118323526, 7855.8914693859224, 8127.4426126409417, 8132.3839899375589, 7885.7643412245634, 7952.0237186110226, 7914.7387808274561])/100.0 #nptrain: 5 w/ adaptive threshold
    
    # Maximum? by new HMM and new threshold
    lr_list_renew  = [0,0.9]
    auc_list_renew = [82.53,82.53] # 10 nptrain
    auc_list_renew = [80.1,80.1]   # 5 nptrain

    # Minimum? using old HMM and no adaptive threshold
    lr_list_old  = [0,0.9]
    auc_list_old = [71.55,71.55]

    # display
    fig = plt.figure()
    ax  = fig.add_subplot(111)

    plt.plot(lr_list, auc_list, 'b-', label='Adaptation', lw=2.0 )
    plt.plot(lr_list_renew, auc_list_renew, 'r--', label='Re-training with new data', lw=2.0)
    plt.plot(lr_list_old, auc_list_old, 'k--', label='No adaptation', lw=2.0)


    ax.set_xlabel('Learning rate', fontsize=18)
    ax.set_ylabel('AUC rate [%]', fontsize=18)
    ax.set_yticklabels([0,20,40,60,80,100], fontsize=18)
    ax.set_ylim([0,100])
    ax.yaxis.grid()

    plt.legend(loc='lower right', prop={'size':20})   
    plt.tight_layout()
    if save_pdf:
        fig.savefig('test.pdf')
        fig.savefig('test.png')
        os.system('cp test.p* ~/Dropbox/HRL/')
    else:
        plt.show()



def n_ptrain_auc_graph(save_pdf=False):

    # max iter: 10

    # Adaptation (max_iter=10, lr=0.8)
    n_list  = [2,3,4,5,6,7,8,9, 10]
    auc_list = [62.46, 75.65, 80.11, 81.38, 80.3, 83.95, 79.35, 84.22, 84.22]

    # Maximum? by Renew 
    lr_list_renew  = [4,5,6,7,8,9,10]
    auc_list_renew = [81.15,81.57,81.81,82.12,78.99,80.14,82.53]

    # Minimum? using old HMM
    lr_list_old  = [0,10]
    auc_list_old = [71.55,71.55]

    # display
    fig = plt.figure()
    ax  = fig.add_subplot(111)

    plt.plot(n_list, auc_list, 'b-', label='Adapted HMM + Adapted Thresholds', lw=2.0 )
    plt.plot(lr_list_renew, auc_list_renew, 'r-', label='New HMM + New Thresholds', lw=2.0)
    plt.plot(lr_list_old, auc_list_old, 'k-', label='Prev HMM + Prev Thresholds', lw=2.0)


    ax.set_xlabel('The number of adaptation data', fontsize=18)
    ax.set_ylabel('AUC rate [%]', fontsize=18)
    ax.set_yticklabels([0,20,40,60,80,100], fontsize=18)
    ax.set_ylim([0,100])
    ax.yaxis.grid()

    plt.legend(loc='lower right', prop={'size':20})   
    plt.tight_layout()
    if save_pdf:
        fig.savefig('test.pdf')
        fig.savefig('test.png')
        os.system('cp test.p* ~/Dropbox/HRL/')
    else:
        plt.show()



def auc_graph(save_pdf=False):

    # max iter: 10
    auc_list = [[5125.0, 7318.181818181818, 6266.4277180406207, 7970.2380952380954, 7478.0976220275352, 7318.181818181818],
                [8062.5, 8363.636363636364, 5328.5543608124244, 9505.9523809523816, 8197.7471839799746, 8454.545454545454],
                [5250.0, 7909.090909090909, 5137.3954599761055, 8761.9047619047633, 8207.1339173967463, 7818.1818181818171]]
    auc_list = np.array(auc_list)/100.0
    auc_list = list(auc_list)

    # display
    fig = plt.figure()
    ax  = fig.add_subplot(111)

    plt.boxplot(auc_list, showmeans=True)

    ## ax.set_xlabel('The number of adaptation data', fontsize=18)
    methods=['General \n monitor', 'Adapted \n monitor', 'New \n monitor']
    plt.xticks([1,2,3], methods)
    for tl in ax.get_xticklabels():
        tl.set_fontsize(18)
    
    
    ax.set_ylabel('AUC rate [%]', fontsize=18)
    ax.set_yticklabels([0,20,40,60,80,100], fontsize=18)
    ax.set_ylim([0,100])
    ax.yaxis.grid()

    ## plt.legend(loc='lower right', prop={'size':20})   
    plt.tight_layout()
    if save_pdf:
        fig.savefig('test.pdf')
        fig.savefig('test.png')
        fig.savefig('test.jpg')
        os.system('cp test.p* ~/Dropbox/HRL/')
        os.system('cp test.j* ~/Dropbox/HRL/')
    else:
        plt.show()



def acc_graph(save_pdf=False):

    # max iter: 10

    acc_list = [[55.55555555555556, 78.125, 55.172413793103445, 67.79661016949152, 74.07407407407408, 78.125],[61.111111111111114, 93.75, 62.06896551724138, 83.05084745762711, 76.5432098765432, 93.75], [55.55555555555556, 81.25, 58.620689655172406, 69.49152542372882, 75.30864197530865, 84.375]]

    # display
    fig = plt.figure()
    ax  = fig.add_subplot(111)

    plt.boxplot(acc_list, showmeans=True)

    ## ax.set_xlabel('The number of adaptation data', fontsize=18)
    methods=['General \n monitor', 'Adapted \n monitor', 'New \n monitor']
    plt.xticks([1,2,3], methods)
    for tl in ax.get_xticklabels():
        tl.set_fontsize(18)
    
    
    ax.set_ylabel('Accuracy [%]', fontsize=18)
    ax.set_yticklabels([0,20,40,60,80,100], fontsize=18)
    ax.set_ylim([0,100])
    ax.yaxis.grid()

    ## plt.legend(loc='lower right', prop={'size':20})   
    plt.tight_layout()
    if save_pdf:
        fig.savefig('test.pdf')
        fig.savefig('test.png')
        os.system('cp test.p* ~/Dropbox/HRL/')
    else:
        plt.show()


def bi_component_graph(save_pdf=False):

    # max iter: 10

    
    old_aucs = [[69.6,71.65,76.01],
                [69.6, 71.83,75.69],
                [69.6, 72.29,76.23],
                [69.6, 72.49,76.82],
                [69.6, 73.19,77.0],
                [69.6,74.58,78.0]]
    adp_aucs = [[68.95,74.71,76.06],
                [66.85,73.59,72.35],
                [67.25,73.37,73.80],
                [68.55,76.80,73.55],
                [70.38,80.16,74.86],
                [70.02,82.43,78.96]]

    # display

    #-------------------------------------------------------------------------------
    fig1 = plt.figure()
    ax  = fig1.add_subplot(111)
    colors = itertools.cycle(['g', 'm', 'c', 'k', 'y','r', 'b', ])
    shapes = itertools.cycle(['x','v', 'o', '+'])
    for i in xrange(len(old_aucs)):
        shape = shapes.next()
        color = colors.next()
        plt.plot(old_aucs[i], '-'+shape+color, label="# new data="+str(i+5), lw=2.0, ms=10.0)

    ## ax.set_xlabel('The number of adaptation data', fontsize=18)
    methods=['Old threshold', 'Adapted threshold', 'New threshold']
    plt.xticks([0,1,2], methods)
    for tl in ax.get_xticklabels():
        tl.set_fontsize(18)
    
    ax.set_xlim([-0.2,2.2])    
    ax.set_ylabel('AUC rate [%]', fontsize=18)
    ax.set_yticklabels([0,20,40,60,80,100], fontsize=18)
    ax.set_ylim([0,100])
    ax.yaxis.grid()

    labels = ax.get_xticklabels()
    plt.setp(labels, rotation=10, fontsize=18)
    plt.tight_layout()
    plt.legend(loc='lower right', prop={'size':20})

    # ------------------------------------------------------------------
    fig2 = plt.figure()
    ax  = fig2.add_subplot(111)
    colors = itertools.cycle(['g', 'm', 'c', 'k', 'y','r', 'b', ])
    shapes = itertools.cycle(['x','v', 'o', '+'])
    for i in xrange(len(adp_aucs)):
        shape = shapes.next()
        color = colors.next()
        plt.plot(adp_aucs[i], '-'+shape+color, label="# new data="+str(i+5), lw=2.0, ms=10.0)

    ## ax.set_xlabel('The number of adaptation data', fontsize=18)
    methods=['Old threshold', 'Adapted threshold', 'New threshold']
    plt.xticks([0,1,2], methods)
    for tl in ax.get_xticklabels():
        tl.set_fontsize(18)
    
    ax.set_xlim([-0.2,2.2])    
    ax.set_ylabel('AUC rate [%]', fontsize=18)
    ax.set_yticklabels([0,20,40,60,80,100], fontsize=18)
    ax.set_ylim([0,100])
    ax.yaxis.grid()

    labels = ax.get_xticklabels()
    plt.setp(labels, rotation=10, fontsize=18)
    plt.tight_layout()
    plt.legend(loc='lower right', prop={'size':20})

    
    if save_pdf:
        fig1.savefig('old_hmm_ths_test.pdf')
        fig1.savefig('old_hmm_ths_test.png')
        fig1.savefig('old_hmm_ths_test.jpg')
        fig2.savefig('adp_hmm_ths_test.pdf')
        fig2.savefig('adp_hmm_ths_test.png')
        fig2.savefig('adp_hmm_ths_test.jpg') 
        os.system('cp ???_hmm_ths_test.p* ~/Dropbox/HRL/')
        os.system('cp ???_hmm_ths_test.j* ~/Dropbox/HRL/')
    else:
        plt.show()

def ahmm_component_graph(save_pdf=False):

    # max iter: 10
    # old-adapt-renew (given renewed boudnary)
    aucs = [[76.01, 76.06,None],
            [75.69, 73.25,None],
            [76.23, 73.85,None],
            [76.82, 71.98,77.64],
            [77.00, 73.68,None],
            [78.00, 77.20,77.56]]

    # old-adapt (given adapted boudnary) OP, PP
    aucs = [[71.65, 74.71],
            [71.83, 73.59],
            [72.29, 73.37],
            [72.49, 76.80],
            [73.19, 80.16],
            [74.58, 82.43]]


    #-------------------------------------------------------------------------------
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    colors = itertools.cycle(['g', 'm', 'c', 'k', 'y','r', 'b', ])
    shapes = itertools.cycle(['x','v', 'o', '+'])
    for i in xrange(len(aucs)):
        shape = shapes.next()
        color = colors.next()
        plt.plot(aucs[i], '-'+shape+color, label="# new data="+str(i+5), lw=2.0, ms=10.0)

    ## methods=['Old threshold', 'Adapted threshold', 'New threshold']
    ## plt.xticks([0,1,2], methods)
    ## ax.set_xlim([-0.2,2.2])    

    methods=['General HMM', 'Adapted HMM']
    plt.xticks([0,1], methods)
    ax.set_xlim([-0.2,1.2])    

    
    for tl in ax.get_xticklabels():
        tl.set_fontsize(18)
    
    ax.set_ylabel('AUC rate [%]', fontsize=18)
    ax.set_yticklabels([0,20,40,60,80,100], fontsize=18)
    ax.set_ylim([0,100])
    ax.yaxis.grid()

    labels = ax.get_xticklabels()
    plt.setp(labels, rotation=0, fontsize=18)
    plt.tight_layout()
    plt.legend(loc='lower right', prop={'size':20})
    
    if save_pdf:
        fig.savefig('test.pdf')
        fig.savefig('test.png')
        fig.savefig('test.jpg')
        os.system('cp test.p* ~/Dropbox/HRL/')
        os.system('cp test.j* ~/Dropbox/HRL/')
    else:
        plt.show()


if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    util.initialiseOptParser(p)
    opt, args = p.parse_args()

    #lr_auc_graph(save_pdf=opt.bSavePdf)
    #n_ptrain_auc_graph(save_pdf=opt.bSavePdf)
    #auc_graph(save_pdf=opt.bSavePdf)
    ## acc_graph(save_pdf=opt.bSavePdf)
    bi_component_graph(save_pdf=opt.bSavePdf)
    #ahmm_component_graph(save_pdf=opt.bSavePdf)
