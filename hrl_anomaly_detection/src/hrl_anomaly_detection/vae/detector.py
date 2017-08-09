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
                      normalTestData, abnormalTestData, ad_method, method, window_size, \
                      alpha, ths_l=None, save_pkl=None, stateful=False, \
                      x_std_div=1.0, x_std_offset=1e-10, z_std=None,\
                      dyn_ths=False, plot=False, renew=False, batch_info=(False,None), **kwargs):
    
    print "Start to get anomaly scores"
    if os.path.isfile(save_pkl) and renew is False :
        d = ut.load_pickle(save_pkl)
        scores_tr_n = d['scores_tr_n']
        zs_tr_n = d['zs_tr_n']
        scores_te_n = d['scores_te_n']
        scores_te_a = d['scores_te_a']
        zs_te_n = d['zs_te_n']
        zs_te_a = d['zs_te_a']
    else:
        scores_tr_n, zs_tr_n = get_anomaly_score(normalTrainData, vae_mean, enc_z_mean, enc_z_logvar,
                                                 window_size, alpha, ad_method, method, stateful=stateful,
                                                 x_std_div=x_std_div, x_std_offset=x_std_offset,
                                                 z_std=z_std, batch_info=batch_info,
                                                 kwargs={'train': True})        
        scores_te_n, zs_te_n = get_anomaly_score(normalTestData, vae_mean, enc_z_mean, enc_z_logvar,
                                     window_size, alpha, ad_method, method, stateful=stateful,
                                     x_std_div=x_std_div, x_std_offset=x_std_offset, z_std=z_std,
                                     batch_info=batch_info)
        scores_te_a, zs_te_a = get_anomaly_score(abnormalTestData, vae_mean, enc_z_mean, enc_z_logvar,
                                     window_size, alpha, ad_method, method, stateful=stateful,
                                     x_std_div=x_std_div, x_std_offset=x_std_offset, z_std=z_std,
                                     batch_info=batch_info)

        d = {}
        d['scores_tr_n'] = scores_tr_n
        d['zs_tr_n']     = zs_tr_n #sample x length x dim
        d['scores_te_n'] = scores_te_n
        d['zs_te_n']     = zs_te_n
        d['scores_te_a'] = scores_te_a
        d['zs_te_a']     = zs_te_a
        ut.save_pickle(d, save_pkl)

    zs_tr_n = np.array(zs_tr_n)
    zs_te_n = np.array(zs_te_n)
    zs_te_a = np.array(zs_te_a)
    scores_tr_n = np.array(scores_tr_n)
    scores_te_n = np.array(scores_te_n)
    scores_te_a = np.array(scores_te_a)

    if dyn_ths:
        l = len(zs_tr_n[0])
        x = np.array(zs_tr_n).reshape(-1,np.shape(zs_tr_n)[-1])
        y = np.array(scores_tr_n).reshape(-1,np.shape(scores_tr_n)[-1])

        method = 'SVR'
        if method=='SVR':
            print "Start to fit SVR with gamma="
            from sklearn.svm import SVR
            clf = SVR(C=1.0, epsilon=0.2, kernel='rbf', degree=3, gamma=2.0)
        elif method=='RF':
            print "Start to fit RF : ", np.shape(x), np.shape(y)
            from sklearn.ensemble import RandomForestRegressor
            clf = RandomForestRegressor(n_estimators=100, min_samples_leaf=1, n_jobs=1)
        elif method == 'KNN':
            print "Start to fit KNN"
            from sklearn.neighbors import KNeighborsRegressor 
            clf = KNeighborsRegressor(n_neighbors=10, n_jobs=1)
        elif method=='GP':
            from sklearn import gaussian_process
            clf = gaussian_process.GaussianProcess(regr='linear', theta0=5.0, \
                                                   corr='squared_exponential', \
                                                   normalize=True, nugget=1)

            u, idx = np.unique(x,axis=0,return_index=True)
            print np.shape(x), len(idx)
            x = x[idx]
            y = y[idx]
                        
            if len(x)>10000:
                # random sampling
                idx_list = range(len(x))
                np.random.shuffle(idx_list)
                x = x[idx_list]
                x = x[:10000]
                y = y[:10000]

        from sklearn import preprocessing
        scaler = preprocessing.StandardScaler()
        #x = scaler.fit_transform(x)            
            
        print np.shape(x), np.shape(y)
        clf.fit(x, y)
        print "-----------------------------------------"

    if True and False:
        print np.shape(zs_tr_n), np.shape(scores_tr_n)
        
        fig = plt.figure() 
        ## from mpl_toolkits.mplot3d import Axes3D
        ## ax = fig.add_subplot(111, projection='3d')
        ## for i, s in enumerate(scores_tr_n):
        ##     ax.scatter(zs_tr_n[i,:,0], zs_tr_n[i,:,1], scores_tr_n[i,:,0], c='g', marker='o')
        ## for i, s in enumerate(scores_te_n):
        ##     ax.scatter(zs_te_n[i,:,0], zs_te_n[i,:,1], scores_te_n[i,:,0], c='b', marker='x')
        ## for i, s in enumerate(scores_te_a):
        ##     ax.scatter(zs_te_a[i,:,0], zs_te_a[i,:,1], scores_te_a[i,:,0], c='r', marker='^')
        for i, s in enumerate(scores_tr_n):
            plt.plot(s, '-g')
        for i, s in enumerate(scores_te_n):
            plt.plot(s, '-b')
        for i, s in enumerate(scores_te_a):
            plt.plot(s, '-r')
        plt.show()

        for i, s in enumerate(scores_te_n):
            fig = plt.figure()
            plt.plot(s, '-b')
            ths = 1

            s_pred_mu = []
            s_pred_bnd = []
            for j in xrange(len(s)):
                if dyn_ths:
                    #x = scaler.transform(zs_te_n[i][j])
                    x = zs_te_n[i][j]
                    if method == 'SVR' or method == 'KNN':
                        s_pred = np.squeeze( clf.predict( x ) )
                        s_pred_mu.append(s_pred)
                        s_pred = s_pred + ths
                    elif method == 'RF':
                        s_pred = clf.predict( x )
                        s_pred_mu.append(s_pred)
                        err_down, err_up = pred_rf(clf, x, 68)
                        s_pred = s_pred + ths*err_up
                    else:
                        s_pred, MSE = clf.predict(x, eval_MSE=True)
                        #s_pred = np.squeeze(s_pred)
                        #MSE    = np.squeeze(MSE)
                        s_pred = s_pred[0,0]
                        s_pred_mu.append(s_pred)
                        s_pred = s_pred + ths*np.sqrt(MSE[0])*10
                        
                else:
                    s_pred = 0+ths
                    s_pred_mu.append(s_pred)
                
                s_pred_bnd.append(s_pred)

            print np.shape(s_pred_mu), np.shape(s_pred_bnd)
            plt.plot(s_pred_mu, '-r')
            
            if len(s_pred_bnd)>0:
                plt.plot(s_pred_bnd, '--r') 
            plt.show()
        

    #--------------------------------------------------------------------
    def get_pos_neg(zs, scores):
        s_preds = []
        MSEs    = []
        err_ups = []
        for i, s in enumerate(scores):
            if dyn_ths:
                #x = scaler.transform(zs[i])
                x = zs[i]
                if method == 'SVR' or method == 'KNN':
                    s_preds.append( clf.predict(x) )
                elif method == 'RF':
                    s_preds.append( clf.predict(x) )
                    _, err_up = pred_rf(clf, x, 95)
                    err_ups.append(err_up)
                else:
                    s_pred, MSE = clf.predict(x, eval_MSE=True)
                    s_preds.append(s_pred)
                    MSEs.append(MSE)
            else:
                s_preds.append([0]*len(s))

        p_ll = []
        n_ll = [] 
        for ths in ths_l:
            p_l = []
            n_l = []

            for i, s in enumerate(scores):
                if dyn_ths:
                    if method == 'SVR' or method == 'KNN':
                        vals = s_preds[i]+ths
                    elif method == 'RF':
                        vals = s_preds[i] + ths*err_ups[i]
                    else:
                        vals = s_preds[i][:,0] + ths*np.sqrt(MSEs[i])
                else: vals = np.array(s_preds[i])+ths

                pos_cnt = 0
                for j in xrange(4,len(s)):
                    if s[j]>vals[j]:
                        if pos_cnt>1:
                            p_l.append(1)
                            break
                        else:
                            pos_cnt += 1
                    elif j == len(s)-1:
                        n_l.append(1)                    

            n_ll.append(n_l)
            p_ll.append(p_l)

        return p_ll, n_ll

    #--------------------------------------------------------------------
    fp_ll, tn_ll = get_pos_neg(zs_te_n, scores_te_n)
    tp_ll, fn_ll = get_pos_neg(zs_te_a, scores_te_a)
            
    #--------------------------------------------------------------------
    tpr_l = []
    fpr_l = []
    for i in xrange(len(ths_l)):
        tpr_l.append( float(np.sum(tp_ll[i]))/float(np.sum(tp_ll[i])+np.sum(fn_ll[i]))*100.0 )
        fpr_l.append( float(np.sum(fp_ll[i]))/float(np.sum(fp_ll[i])+np.sum(tn_ll[i]))*100.0 )
        
    from sklearn import metrics
    roc = metrics.auc(fpr_l, tpr_l, True)
    
    return tp_ll, tn_ll, fp_ll, fn_ll, roc


def get_roc(tp_l, tn_l, fp_l, fn_l):

    tp_ll = []
    fp_ll = []
    tn_ll = []
    fn_ll = []  
    for i in xrange(len(tp_l)):
        tp_ll.append( tp_l[i])
        fp_ll.append( fp_l[i])
        tn_ll.append( tn_l[i])
        fn_ll.append( fn_l[i])



    tpr_l = np.array(tp_ll).astype(float)/(np.array(tp_ll).astype(float)+
                                           np.array(fn_ll).astype(float))*100.0
    fpr_l = np.array(fp_ll).astype(float)/(np.array(fp_ll).astype(float)+
                                           np.array(tn_ll).astype(float))*100.0

    from sklearn import metrics 
    return metrics.auc(fpr_l, tpr_l, True)


def get_anomaly_score(X, vae, enc_z_mean, enc_z_logvar, window_size, alpha, ad_method, method,\
                      nSample=1000,\
                      stateful=False, x_std_div=1, x_std_offset=1e-10, z_std=0.5,\
                      batch_info=(False,None), **kwargs):

    x_dim = len(X[0][0])
    length = len(X[0])
    train_flag = kwargs.get('train', False)

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
                if window_size>1:
                    xx = x[j:j+1]
                    for k in xrange(batch_info[1]-1):
                        xx = np.vstack([xx, x[j:j+1] ])
                else:
                    xx = np.expand_dims(x[j:j+1,0], axis=0)
                    for k in xrange(batch_info[1]-1):
                        xx = np.vstack([xx, np.expand_dims(x[j:j+1,0], axis=0) ])
                    
            else:
                xx = x[j:j+1]

            if method == 'lstm_vae_custom':
                x_in = np.concatenate((xx, np.zeros((len(xx), len(xx[0]),1))), axis=-1)
            else:
                x_in = xx
                
            x_new  = vae.predict(x_in, batch_size=batch_info[1])[0]
            ## x_new  = vae.predict(x[j:j+1])[0]

            # length x dim
            x_mean = x_new[:,:x_dim]
            if enc_z_logvar is not None:
                x_std  = np.sqrt(x_new[:,x_dim:]/x_std_div+x_std_offset)
            else:
                x_std = None

            #---------------------------------------------------------------
            if ad_method == 'recon_prob':
                # Method 1: Reconstruction probability
                l = get_reconstruction_err_prob(xx, x_mean, x_std, alpha=alpha)
                s.append(l)
                z.append( enc_z_mean.predict(xx, batch_size=batch_info[1])[0].tolist() )                
            elif ad_method == 'recon_err':
                # Method 1: Reconstruction probability
                l = get_reconstruction_err(xx, x_mean, alpha=alpha)
                s.append( l )
                z.append( enc_z_mean.predict(xx, batch_size=batch_info[1])[0].tolist() )                
            elif ad_method == 'lower_bound':

                if method == 'lstm_vae_custom':
                    p = float(j)/float(length-window_size+1)
                    if train_flag:
                        p = p*np.ones((batch_info[1], window_size, 1))
                    else:
                        #temp
                        p = p*np.ones((batch_info[1], window_size, 1))
                        ## p = np.zeros((len(x), len(x[0]),1))
                else:
                    p = None
                
                # Method 2: Lower bound
                l, z_mean, z_log_var = get_lower_bound(xx, x_mean, x_std, z_std,
                                                       enc_z_mean, enc_z_logvar,\
                                                       x_dim, method, p)
                s.append(l)
                z.append(z_mean.tolist()) # + z_log_var.tolist())

        ## s = np.cumsum(s)
        ## if len(np.shape(s))<2: s = np.expand_dims(s, axis=1)
        scores.append(s) # s is scalers
        zs.append(z)

    return scores, zs


def get_reconstruction_err_prob(x, x_mean, x_std, alpha=1.0):
    '''
    Return minimum value for alpha x \sum P(x_i(t) ; mu, std) over time 
    '''
    if len(np.shape(x))>2: x = x[0]
    
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
    return [-np.amin(alpha.dot( np.log(np.array(p_l)) ))]


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


def get_reconstruction_err_lld(x, x_mean, alpha=1.0):
    '''
    Return reconstruction error likelihood
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


def get_lower_bound(x, x_mean, x_std, z_std, enc_z_mean, enc_z_logvar, nDim, method=None, p=None):
    '''
    No fixed batch
    x: length x dim
    '''
    if len(np.shape(x))>2:
        if method == 'lstm_vae_custom':
            x_in = np.concatenate((x, p), axis=-1)
        else:
            x_in = xx
        
        batch_size = len(x)
        z_mean    = enc_z_mean.predict(x_in, batch_size=batch_size)[0]
        z_log_var = enc_z_logvar.predict(x_in, batch_size=batch_size)[0]
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

    if method == 'lstm_vae_custom':
        p=0
        if len(np.shape(z_log_var))>1:            
            kl_loss = - 0.5 * np.sum(1 + z_log_var -np.log(z_std*z_std) - (z_mean-p)**2
                                    - np.exp(z_log_var)/(z_std*z_std), axis=-1)
        else:
            kl_loss = - 0.5 * np.sum(1 + z_log_var -np.log(z_std*z_std) - (z_mean-p)**2
                                    - np.exp(z_log_var)/(z_std*z_std))
    else:
        if len(np.shape(z_log_var))>1:
            kl_loss = - 0.5 * np.sum(1 + z_log_var - z_mean**2 - np.exp(z_log_var), axis=-1)
        else:
            kl_loss = - 0.5 * np.sum(1 + z_log_var - z_mean**2 - np.exp(z_log_var))

    if len(xent_loss + kl_loss)>1:
        return [np.mean(xent_loss + kl_loss)], z_mean, z_log_var
    else:
        return xent_loss + kl_loss, z_mean, z_log_var


def pred_rf(model, x, percentile=95):

    preds = []
    for pred in model.estimators_:
        preds.append(pred.predict(x)[0])
    err_down = np.percentile(preds, (100 - percentile) / 2. )
    err_up   = np.percentile(preds, 100 - (100 - percentile) / 2.)

    return err_down, err_up

