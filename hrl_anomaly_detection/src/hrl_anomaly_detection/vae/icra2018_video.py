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

# Private utils
from hrl_anomaly_detection.util import *
from hrl_anomaly_detection.util_viz import *
from hrl_anomaly_detection import data_manager as dm
from hrl_anomaly_detection import util as util
from hrl_execution_monitor import util as autil
from hrl_execution_monitor import preprocess as pp

# Private learners
from hrl_anomaly_detection.hmm import learning_hmm as hmm
import hrl_anomaly_detection.classifiers.classifier as cf
import hrl_anomaly_detection.data_viz as dv
import hrl_anomaly_detection.isolator.isolation_util as iutil

import rosbag
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed

# visualization
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec
from matplotlib import animation
from matplotlib import rc
import matplotlib.patches as patches
import itertools
colors = itertools.cycle(['g', 'm', 'c', 'k', 'y','r', 'b', ])
shapes = itertools.cycle(['x','v', 'o', '+'])

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42 




def animate_score(bSave=False):
    ''' Generate animation for ICRA2018 video (failure data)'''
    
    # pred_score_4dim_failure: idx==3, class 5-10, ths=1.5 
    # /home/dpark/hrl_file_server/dpark_data/anomaly/RAW_DATA/AURO2016/s5_feeding/5_10_failure.pkl

    # Save Data
    #prefix   = '5_10_failure'
    prefix   = '0_27_success'
    save_pkl = os.path.join('./'+prefix+'_data_score.pkl')    
    sd = ut.load_pickle(save_pkl)
    
    nDim = sd['nDim']  
    s    = sd['s']  
    s_pred_mean = sd['s_pred_mean'] 
    s_pred_bnd  = sd['s_pred_bnd'] 
    x_true      = sd['x_true'] 
    x_pred_mean = sd['x_pred_mean'] 
    x_pred_std  = sd['x_pred_std'] 
    param_dict  = sd['param_dict']  
    x_time      = range(len(x_pred_mean))

    if nDim>4: nDim=4
    
    ## matplotlib.rcParams['animation.bitrate'] = 2000
    plt.rc('text', usetex=True)    
    fig = plt.figure(figsize=(5, 6))
    gs = gridspec.GridSpec(6, 1, height_ratios=[1,1,1,1,0.4,2]) 
    
    #-------------------------- 1 ------------------------------------
    ax_list = []
    lines = []
    meanlines = []
    stdlines  = []
    filllines = []
    for k in xrange(nDim):
        ax = fig.add_subplot(gs[k])

        line,    = ax.plot([], [], '-b')
        lines.append(line)
        meanline, = ax.plot([],[], '-r')
        meanlines.append(meanline)
        ## fillline = ax.fill_between([],[],[],facecolor='red', alpha=0.3, linewidth=0)
        ## filllines.append(fillline)
        
        if k==0:
            ax.set_ylabel('Sound'+'\n'+'Energy', rotation='horizontal',
                          verticalalignment='center',
                          horizontalalignment='center', fontsize=12)
        elif k==1: 
            ax.set_ylabel('1st Joint'+'\n'+'Torque(Nm)', rotation='horizontal',
                          verticalalignment='center',
                         horizontalalignment='center', fontsize=12)
            ax.set_ylim([-2.1,3.])
        elif k==2: 
            ax.set_ylabel('Accumulated'+'\n'+'Force'+'\n'+'on Spoon(N)',
                          rotation='horizontal', verticalalignment='center',
                          horizontalalignment='center', fontsize=12)
            ax.set_ylim([-0.5, 4]) #11])
        elif k==3: 
            ax.set_ylabel('Spoon-Mouth'+'\n'+'Distance(m)', rotation='horizontal',
                          verticalalignment='center',
                          horizontalalignment='center', fontsize=12)
            ax.set_ylim([-0.15,0.13])
            
        ax.yaxis.set_label_coords(-0.22,0.5)            
        #ax.set_ylabel(param_dict['feature_names'][k])

        ax.locator_params(axis='y', nbins=3)
        #if k < nDim-1:
        ax.tick_params(axis='x', bottom='off', labelbottom='off')
        ax.set_xlim([0, len(x_pred_mean)])

    # Score visualization ------------------------------------------------------
    ax2 = fig.add_subplot(gs[k+1+1])
    line,    = ax2.plot([], [], '-b')
    lines.append(line)
    meanline, = ax2.plot([],[],'-r')
    meanlines.append(meanline)
    thresline, = ax2.plot([],[], ':r')

    ax2.set_ylabel('Anomaly \n Score',
                  rotation='horizontal', verticalalignment='center',
                  horizontalalignment='center', fontsize=12)
    ax2.yaxis.set_label_coords(-0.22,0.5)            
    ax2.locator_params(axis='y', nbins=3)
    ax2.set_ylim([-1, 5])
    
    ax2.set_xlim([0, len(s_pred_bnd)])

    anomaly = [True, ]

    # --------------------------------------------------------------------------

    # Legend
    ax = fig.add_subplot(gs[0])
    axbox = ax.get_position()
    import matplotlib.lines as mlines
    blue_line = mlines.Line2D([], [], color='blue', alpha=0.5, markersize=30, label='Observations')
    red_line  = mlines.Line2D([], [], color='red', markersize=15,
                              label=r'Predicted distributions $\mu_{\bf x} \pm \sigma_{\bf x}$')

    handles = [blue_line,red_line]
    labels = [h.get_label() for h in handles]
    lg1 = plt.legend(handles=handles, labels=labels, loc=(axbox.x0-0.45, axbox.y0+0.5), #loc='upper center',
               ncol=2, shadow=False, fancybox=False, edgecolor='k', prop={'size': 12})

    blue_line = mlines.Line2D([], [], color='blue', alpha=0.5, markersize=30, label='Current')
    red_line  = mlines.Line2D([], [], color='red', markersize=15, label=r'Expected')
    red_dash_line  = mlines.Line2D([], [], color='red', ls=':', markersize=15, label=r'Threshold')
    handles = [blue_line,red_line, red_dash_line]
    labels = [h.get_label() for h in handles]
    axbox = ax2.get_position()
    lg2 = plt.legend(handles=handles, labels=labels, ncol=3, loc=(axbox.x0-0.33, axbox.y0-4.4),
                     shadow=False, fancybox=False, edgecolor='k', prop={'size': 12})
    ax.add_artist(lg1)
    ax.add_artist(lg2)

    # Axis labels
    if param_dict is not None:
        x_tick = [0,
                  (param_dict['timeList'][-1]-0)/2.0,
                  param_dict['timeList'][-1]]
        ax2.set_xticks(np.linspace(0, len(x_pred_mean), len(x_tick)))        
        ax2.set_xticklabels(x_tick)
        ax2.set_xlabel('Time [s]', fontsize=16)
        fig.subplots_adjust(left=0.28, right=0.95) 

    #fig.tight_layout()
    ## leg_1, = ax_list[0].plot([], [], color='#FF0000', lw=6, label='Current execution')
    ## leg_2, = ax_list[0].plot([], [], color='#0000FF', lw=6, label=r'Successful executions (Mean $\pm$ Std)')
    ## legend = ax_list[0].legend(handles=[leg_1, leg_2], loc=3, fancybox=True, shadow=True,
    ##                            ncol=1, borderaxespad=0.,bbox_to_anchor=(0.0, 1.3), \
    ##                            prop={'size':24})


    #-------------------------- 2 ------------------------------------
    legend = None
    ax_list= fig.axes
    patch_list = []
    for i in range(len(s)):
        patch_list.append( ax2.add_patch( patches.Rectangle((-100,-1),1,6, facecolor='peru', edgecolor='none' ) ) )

    def init():
        for i in xrange(nDim+1):
            lines[i].set_data([],[])
            meanlines[i].set_data([],[])
            ## if i<nDim:
            ##     filllines[i].set_data([],[],[])
        thresline.set_data([],[])

        
        return lines, meanlines, thresline #, filllines


    def animate(i):

        x = x_time[:i]
        
        try:
            for j in xrange(nDim):
                
                if j==1: mc = 0.1
                else: mc = 1.
                for coll in (ax_list[j].collections):
                    ax_list[j].collections.remove(coll)
                f = ax_list[j].fill_between(x, np.array(x_pred_mean)[:i,j]+mc*np.array(x_pred_std)[:i,j],
                                            np.array(x_pred_mean)[:i,j]-mc*np.array(x_pred_std)[:i,j],
                                            facecolor='red', alpha=0.3, linewidth=0)                
                meanlines[j].set_data(x, x_pred_mean[:i,j])
                lines[j].set_data(x, x_true[:i,j])

            lines[j+1].set_data(x, s[:i])
            meanlines[j+1].set_data(x, s_pred_mean[:i])
            thresline.set_data(x, np.array(s_pred_bnd[:i])-0.6) #0.7 for failure

            if s[i]-s_pred_bnd[i]+0.6>0:
                ## for coll in (ax_list[-1].collections):
                ##     ax_list[-1].collections.remove(coll)
                patch_list[i].set_xy((i,-1))
                #p = ax_list[-1].add_patch( patches.Rectangle((i,-1),1,6, facecolor='peru', edgecolor='none' ))
            
        except:
            print "Length error for line_",j

        ## for j in range(len(ax_list)):
        ##     ax_list[j].set_xlim([i-20,i])

        return lines, meanlines, thresline #, #f #fillines

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(x_time), interval=100,
                                   repeat=False)

    if True: #False and bSave == True:
        print "Start to save"
        ## FFMpegWriter = animation.writers['ffmpeg']
        ## writer = FFMpegWriter(fps=23, bitrate=-1)
        ## anim.save('ani_test.mp4', writer=writer)
        #anim.save('ani_test.mp4', fps=30)
        anim.save('ani_test.mp4', fps=23, extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'], bitrate=-1)
        print "File saved??"
    else:
        plt.show()


    

if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    util.initialiseOptParser(p)
    ## p.add_option('--eval_isol', '--ei', action='store_true', dest='evaluation_isolation',
    ##               default=False, help='Evaluate anomaly isolation with double detectors.')    
    opt, args = p.parse_args()

    animate_score()

