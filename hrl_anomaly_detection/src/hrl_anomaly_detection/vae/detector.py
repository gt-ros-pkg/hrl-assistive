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
                      normalTrainData, abnormalTrainData,\
                      normalTestData, abnormalTestData, ad_method, window_size, \
                      alpha, ths_l=None, save_pkl=None, stateful=False, \
                      x_std_div=1.0, x_std_offset=1e-10,
                      dyn_ths=False, plot=False, renew=False, batch_info=(False,None), **kwargs):

    if os.path.isfile(save_pkl) and renew is False:
        d = ut.load_pickle(save_pkl)
        scores_tr_n = d['scores_tr_n']
        zs_tr_n = d['zs_tr_n']
        scores_te_n = d['scores_te_n']
        scores_te_a = d['scores_te_a']
        zs_te_n = d['zs_te_n']
        zs_te_a = d['zs_te_a']
    else:
        scores_tr_n, zs_tr_n = get_anomaly_score(normalTrainData, vae_mean, enc_z_mean, enc_z_logvar,
                                                 window_size, alpha, ad_method, stateful=stateful,
                                                 x_std_div=x_std_div, x_std_offset=x_std_offset,
                                                 batch_info=batch_info)        
        scores_te_n, zs_te_n = get_anomaly_score(normalTestData, vae_mean, enc_z_mean, enc_z_logvar,
                                     window_size, alpha, ad_method, stateful=stateful,
                                     x_std_div=x_std_div, x_std_offset=x_std_offset,batch_info=batch_info)
        scores_te_a, zs_te_a = get_anomaly_score(abnormalTestData, vae_mean, enc_z_mean, enc_z_logvar,
                                     window_size, alpha, ad_method, stateful=stateful,
                                     x_std_div=x_std_div, x_std_offset=x_std_offset,batch_info=batch_info)

        d = {}
        d['scores_tr_n'] = scores_tr_n
        d['zs_tr_n']     = zs_tr_n
        d['scores_te_n'] = scores_te_n
        d['zs_te_n']     = zs_te_n
        d['scores_te_a'] = scores_te_a
        d['zs_te_a']     = zs_te_a
        ut.save_pickle(d, save_pkl)


    if dyn_ths:
        print "Start to fit SVR with gamma=", 0.5
        from sklearn.svm import SVR
        clf = SVR(C=1.0, epsilon=0.2, kernel='rbf', gamma=0.5)
        x = np.array(zs_tr_n).reshape(-1,np.shape(zs_tr_n)[-1])
        y = np.array(scores_tr_n).reshape(-1,np.shape(scores_tr_n)[-1])
        clf.fit(x, y)

    ## for i, s_true in enumerate(scores_n):
    ##     s_pred = clf.predict(zs_n[i])        
    ##     fig = plt.figure()  
    ##     plt.plot(s_true, '-b')
    ##     plt.plot(s_pred, '-r') 
    ##     plt.show()
        
    #for s in scores_a:
    #    fig = plt.figure()
    #    plt.plot(s, '-r')
    #    plt.show()

    
    tpr_l = []
    fpr_l = []

    tp_ll = []
    fp_ll = []
    tn_ll = []
    fn_ll = []

    for ths in ths_l:
        tp_l = []
        fp_l = []
        tn_l = []
        fn_l = []
        for i, s in enumerate(scores_te_n):
            for j in xrange(len(s)):
                if dyn_ths: s_pred = clf.predict(zs_te_n[i][j])
                else: s_pred = 0
                
                if s[j]>s_pred+ths:
                    fp_l.append(1)
                    break
                elif j == len(s)-1:
                    tn_l.append(1)

        for i, s in enumerate(scores_te_a):
            for j in xrange(len(s)):
                if dyn_ths: s_pred = clf.predict(zs_te_a[i][j])
                else: s_pred = 0
                
                if s[j]>s_pred+ths:
                    tp_l.append(1)
                    break
                elif j == len(s)-1:
                    fn_l.append(1)

        tpr_l.append( float(np.sum(tp_l))/float(np.sum(tp_l)+np.sum(fn_l))*100.0 )
        fpr_l.append( float(np.sum(fp_l))/float(np.sum(fp_l)+np.sum(tn_l))*100.0 )

        tp_ll.append(tp_l)
        tn_ll.append(tn_l)
        fp_ll.append(fp_l)
        fn_ll.append(fn_l)
    

    e_n_l  = np.amax(scores_te_n, axis=-1) #[val[-1] for val in scores_n if val != np.log(1e-50) ]
    e_ab_l = np.amax(scores_te_a, axis=-1) #[val[-1] for val in scores_a if val != np.log(1e-50) ]
    print np.mean(e_n_l), np.std(e_n_l)
    print np.mean(e_ab_l), np.std(e_ab_l)
    print "acc ", float(np.sum(tp_l)+np.sum(tn_l))/float(np.sum(tp_l+fp_l+tn_l+fn_l))

    print tpr_l
    print fpr_l

    from sklearn import metrics 
    print "roc: ", metrics.auc(fpr_l, tpr_l, True)

    if plot:
        fig = plt.figure(figsize=(12,6))
        fig.add_subplot(1,2,1)    
        plt.plot(fpr_l, tpr_l, '-*b', ms=5, mec='b')
        plt.xlim([0,100])
        plt.ylim([0,100])

        fig.add_subplot(1,2,2)    
        plt.plot(e_n_l, '*b', ms=5, mec='b')
        plt.plot(e_ab_l, '*r', ms=5, mec='r')
        plt.show()

    return tp_ll, tn_ll, fp_ll, fn_ll



def get_anomaly_score(X, vae, enc_z_mean, enc_z_logvar, window_size, alpha, ad_method, nSample=1000,
                      stateful=False, x_std_div=1, x_std_offset=1e-10,batch_info=(False,None), **kwargs):

    x_dim = len(X[0][0])

    scores = []
    zs = []
    for i in xrange(len(X)): # per sample
        #print "sample: ", i+1, " out of ", len(X)
        np.random.seed(3334 + i)
            
        if window_size>0: x = vutil.sampleWithWindow(X[i:i+1], window=window_size)
        else:             x = X[i:i+1]
        if type(x) is list: x = np.array(x)

        if stateful:
            vae.reset_states()
            enc_z_mean.reset_states()
            if enc_z_logvar is not None: enc_z_logvar.reset_states()                
            
        z = []
        s = []
        for j in xrange(len(x)): # per window
            # anomaly score per timesteps in an window            
            # pdf prediction
            if batch_info[0]:
                xx = np.expand_dims(x[j:j+1,0], axis=0)
                for k in xrange(batch_info[1]-1):
                    xx = np.vstack([xx, np.expand_dims(x[j:j+1,0], axis=0) ])
            else:
                xx = x[j:j+1]
            x_new  = vae.predict(xx, batch_size=batch_info[1])[0]
            ## x_new  = vae.predict(x[j:j+1])[0]

            # length x dim
            x_mean = x_new[:,:x_dim]
            if enc_z_logvar is not None:
                x_std_div    = kwargs['x_std_div']
                x_std_offset = kwargs['x_std_offset']
                x_std  = np.sqrt(x_new[:,x_dim:]/x_std_div+x_std_offset)
            else:
                x_std = None

            #---------------------------------------------------------------
            if ad_method == 'recon_prob':
                # Method 1: Reconstruction probability
                l,z = get_reconstruction_err_prob(xx, x_mean, x_std, alpha=alpha)
                s.append(l)
                z.append( z.tolist() )                
            elif ad_method == 'recon_err':
                # Method 1: Reconstruction probability
                l = get_reconstruction_err(xx, x_mean, alpha=alpha)
                s.append( l )
                z.append( enc_z_mean.predict(xx, batch_size=batch_info[1])[0].tolist() )                
            elif ad_method == 'lower_bound':
                # Method 2: Lower bound
                l, z_mean, z_log_var = get_lower_bound(xx, x_mean, x_std, enc_z_mean, enc_z_logvar,\
                                                       x_dim)
                s.append(l)
                z.append(z_mean.tolist() + z_log_var.tolist())

        scores.append(s) # s is scalers
        zs.append(z)

    return scores, zs


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


def get_reconstruction_err(x, x_mean, alpha=1.0):
    '''
    Return minimum value for alpha \sum alpha * |x(i)-x_mean(i)|^2 over time 
    '''

    if len(np.shape(x))>2:
        x = x[0]
        
    p_l     = []
    for k in xrange(len(x_mean[0])): # per dim
        p = []
        for l in xrange(len(x_mean)): # per length
            
            p.append( alpha * (x_mean[l,k] - x[l,k])**2 ) # length

        p_l.append(p) # dim x length
    p_l = np.sum(p_l, axis=0)

    # find min 
    return [np.amin(p_l)]


def get_lower_bound(x, x_mean, x_std, enc_z_mean, enc_z_logvar, nDim):
    '''
    No fixed batch
    x: length x dim
    '''
    if len(np.shape(x))>2:
        batch_size = len(x)
        z_mean    = enc_z_mean.predict(x, batch_size=batch_size)[0]
        z_log_var = enc_z_logvar.predict(x, batch_size=batch_size)[0]
        x = x[0]
    else:
        z_mean    = enc_z_mean.predict(x)[0]
        z_log_var = enc_z_logvar.predict(x)[0]

    p_l     = []
    log_p_x_z = -0.5 * ( np.sum( ((x-x_mean)/x_std)**2, axis=-1) \
                         + float(nDim) * np.log(2.0*np.pi) + np.sum(np.log(x_std**2), axis=-1) )
    if len(np.shape(log_p_x_z))>1:
        xent_loss = np.mean(-log_p_x_z, axis=-1)
    else:
        xent_loss = -log_p_x_z

    if len(np.shape(z_log_var))>1:
        kl_loss = - 0.5 * np.sum(1 + z_log_var - z_mean**2 - np.exp(z_log_var), axis=-1)
    else:
        kl_loss = - 0.5 * np.sum(1 + z_log_var - z_mean**2 - np.exp(z_log_var))
                                 
    return xent_loss + kl_loss, z_mean, z_log_var
    #return float(np.mean(xent_loss + kl_loss) )

