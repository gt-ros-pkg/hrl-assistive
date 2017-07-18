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

import os, sys, copy, random
import numpy as np
import scipy

# Private utils
import hrl_lib.util as ut
from hrl_anomaly_detection.vae import util as vutil

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


def anomaly_detection(vae, vae_mean, vae_logvar, enc_z_mean, enc_z_logvar, generator,
                      normalTestData, abnormalTestData, window_size, \
                      alpha, save_pkl=None, stateful=False, x_std_div=1.0, x_std_offset=1e-10):

    if os.path.isfile(save_pkl) and False:
        d = ut.load_pickle(save_pkl)
        scores_n = d['scores_n']
        scores_a = d['scores_a']
    else:
        scores_n = get_anomaly_score(normalTestData, vae_mean, enc_z_mean, enc_z_logvar,
                                     window_size, alpha, stateful=stateful,
                                     x_std_div=x_std_div, x_std_offset=x_std_offset)
        scores_a = get_anomaly_score(abnormalTestData, vae_mean, enc_z_mean, enc_z_logvar,
                                     window_size, alpha, stateful=stateful,
                                     x_std_div=x_std_div, x_std_offset=x_std_offset)
        
        d = {}
        d['scores_n'] = scores_n
        d['scores_a'] = scores_a
        ut.save_pickle(d, save_pkl)


    #ths_l = -np.logspace(-1,0.8,40)+2.0
    ths_l = -np.logspace(-1,0.5,40)+1.5
    ths_l = np.linspace(127,133,40)
    
    tpr_l = []
    fpr_l = []

    for ths in ths_l:
        tp_l = []
        fp_l = []
        tn_l = []
        fn_l = []
        for s in scores_n:
            for i in xrange(len(s)):
                if s[i]>ths:
                    fp_l.append(1)
                    break
                elif i == len(s)-1:
                    tn_l.append(1)

        for s in scores_a:
            for i in xrange(len(s)):
                if s[i]>ths:
                    tp_l.append(1)
                    break
                elif i == len(s)-1:
                    fn_l.append(1)

        tpr_l.append( float(np.sum(tp_l))/float(np.sum(tp_l)+np.sum(fn_l))*100.0 )
        fpr_l.append( float(np.sum(fp_l))/float(np.sum(fp_l)+np.sum(tn_l))*100.0 )
    

    e_n_l  = np.amin(scores_n, axis=-1) #[val[-1] for val in scores_n if val != np.log(1e-50) ]
    e_ab_l = np.amin(scores_a, axis=-1) #[val[-1] for val in scores_a if val != np.log(1e-50) ]
    print np.mean(e_n_l), np.std(e_n_l)
    print np.mean(e_ab_l), np.std(e_ab_l)
    print "acc ", float(np.sum(tp_l)+np.sum(tn_l))/float(np.sum(tp_l+fp_l+tn_l+fn_l))

    print tpr_l
    print fpr_l

    from sklearn import metrics 
    print "roc: ", metrics.auc(fpr_l, tpr_l, True)

    fig = plt.figure(figsize=(12,6))
    fig.add_subplot(1,2,1)    
    plt.plot(fpr_l, tpr_l, '-*b', ms=5, mec='b')
    plt.xlim([0,1])
    plt.ylim([0,1])
    
    fig.add_subplot(1,2,2)    
    plt.plot(e_n_l, '*b', ms=5, mec='b')
    plt.plot(e_ab_l, '*r', ms=5, mec='r')
    plt.show()

    return 



def get_anomaly_score(X, vae, enc_z_mean, enc_z_logvar, window_size, alpha, nSample=1000,
                      stateful=False, x_std_div=1, x_std_offset=1e-10):

    x_dim = len(X[0][0])

    scores = []
    for i in xrange(len(X)): # per sample
        print "sample: ", i+1, " out of ", len(X)
        np.random.seed(3334 + i)

        if window_size>0: x = vutil.sampleWithWindow(X[i:i+1], window=window_size)
        else:             x = X[i:i+1]
        if type(x) is list: x = np.array(x)

        if stateful:
            vae.reset_states()
            enc_z_mean.reset_states()
            enc_z_logvar.reset_states()
            

        s = []
        for j in xrange(len(x)): # per window
            # anomaly score per timesteps in an window            
            # pdf prediction
            x_new  = vae.predict(x[j:j+1])[0]

            # length x dim
            x_mean = x_new[:,:x_dim]
            x_std  = np.sqrt(x_new[:,x_dim:]/x_std_div+x_std_offset)

            #---------------------------------------------------------------
            # Method 1: Reconstruction probability
            #s.append( get_reconstruction_err_prob(x[j], x_mean, x_std, alpha=alpha) )

            # Method 2: Lower bound
            s.append( get_lower_bound(x[j:j+1], x_mean, x_std, enc_z_mean, enc_z_logvar) )
            

        scores.append(s) # s is scalers
    return scores


def get_reconstruction_err_prob(x, x_mean, x_std, alpha=1.0):
    '''
    Return minimum value for alpha x \sum P(x_i(t) ; mu, std) over time 
    '''
    
    p_l     = []
    for k in xrange(len(x_mean[0])): # per dim
        p = []
        for l in xrange(len(x_mean)): # per length
            p.append(scipy.stats.norm(x_mean[l,k], x_std[l,k]).pdf(x[l,k])) # length

        p = [val if not np.isinf(val).any() and not np.isnan(val).any() and val > 0
             else 1e-50 for val in p]
        p_l.append(p) # dim x length

    # find min 
    ## return np.mean(alpha.dot( np.log(np.array(p_l)) )) 
    return -np.amin(alpha.dot( np.log(np.array(p_l)) ))


def get_lower_bound(x, x_mean, x_std, enc_z_mean, enc_z_logvar):
    '''
    x: length x dim
    '''
    nDim = len(x[0])

    z_mean    = enc_z_mean.predict(x)[0]
    z_log_var = enc_z_logvar.predict(x)[0]
        
    p_l     = []
    log_p_x_z = -0.5 * ( np.sum( ((x-x_mean)/x_std)**2, axis=-1) \
                         + float(nDim) * np.log(2.0*np.pi) + np.sum(np.log(x_std**2), axis=-1) )
    xent_loss = np.mean(-log_p_x_z, axis=-1)
    kl_loss = - 0.5 * np.sum(1 + z_log_var - z_mean**2 - np.exp(z_log_var), axis=-1)
    
    #return xent_loss + kl_loss
    return np.mean(xent_loss + kl_loss) 

