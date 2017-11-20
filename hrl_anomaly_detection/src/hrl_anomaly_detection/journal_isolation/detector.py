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
from hrl_anomaly_detection.RAL18_detection import util as vutil
from hrl_anomaly_detection.RAL18_detection import metrics as ad_metrics

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
                      normalTrainData, normalValData, abnormalTrainData,\
                      normalTestData, abnormalTestData, ad_method, method, window_size=1, \
                      alpha=None, ths_l=None, save_pkl=None, stateful=False, \
                      x_std_div=1.0, x_std_offset=1e-10, z_std=None, phase=1.0, \
                      dyn_ths=False, plot=False, renew=False, batch_info=(False,None), **kwargs):
                      
    print "Start to get anomaly scores"
    if (os.path.isfile(save_pkl) and renew is False):
        print "Load anomaly detection results"
        d = ut.load_pickle(save_pkl)
    else:
        scores_tr_n, zs_tr_n = get_anomaly_score(normalTrainData, vae_mean, enc_z_mean, enc_z_logvar,
                                                 window_size, alpha, ad_method, method, stateful=stateful,
                                                 x_std_div=x_std_div, x_std_offset=x_std_offset,
                                                 z_std=z_std, batch_info=batch_info, valData=normalValData,
                                                 phase=phase, train_flag=True)        
        scores_tr_a, zs_tr_a, err_tr_a = get_anomaly_score(abnormalTrainData, vae_mean, enc_z_mean, enc_z_logvar,
                                                           window_size, alpha, ad_method, method,
                                                           stateful=stateful, x_std_div=x_std_div,
                                                           x_std_offset=x_std_offset, z_std=z_std,
                                                           phase=phase,\
                                                           batch_info=batch_info, ref_scores=scores_tr_n,
                                                           return_err=True)
        scores_te_n, zs_te_n = get_anomaly_score(normalTestData, vae_mean, enc_z_mean, enc_z_logvar,
                                                 window_size, alpha, ad_method, method, stateful=stateful,
                                                 x_std_div=x_std_div, x_std_offset=x_std_offset, z_std=z_std,
                                                 phase=phase,\
                                                 batch_info=batch_info, ref_scores=scores_tr_n)
        scores_te_a, zs_te_a, err_te_a = get_anomaly_score(abnormalTestData, vae_mean, enc_z_mean, enc_z_logvar,
                                                           window_size, alpha, ad_method, method,
                                                           stateful=stateful,
                                                           x_std_div=x_std_div, x_std_offset=x_std_offset,
                                                           z_std=z_std,
                                                           phase=phase,\
                                                           batch_info=batch_info, ref_scores=scores_tr_n,
                                                           return_err=True)

        d = {}
        d['scores_tr_n'] = scores_tr_n
        d['zs_tr_n']     = zs_tr_n #sample x length x dim
        d['scores_tr_a'] = scores_tr_a
        d['zs_tr_a']     = zs_tr_a #sample x length x dim
        d['err_tr_a']    = err_tr_a
        d['scores_te_n'] = scores_te_n
        d['zs_te_n']     = zs_te_n
        d['scores_te_a'] = scores_te_a
        d['zs_te_a']     = zs_te_a
        d['err_te_a']    = err_te_a
        ut.save_pickle(d, save_pkl)


    zs_tr_n     = np.array(d['zs_tr_n'])
    zs_tr_a     = np.array(d['zs_tr_a'])
    zs_te_n     = np.array(d['zs_te_n'])
    zs_te_a     = np.array(d['zs_te_a'])
    scores_tr_n = np.array(d['scores_tr_n'])
    scores_tr_a = np.array(d['scores_tr_a'])
    scores_te_n = np.array(d['scores_te_n'])
    scores_te_a = np.array(d['scores_te_a'])
    
    if ad_method == 'recon_err_vec': dyn_ths=False

    if dyn_ths:
        l = len(zs_tr_n[0])
        x = np.array(zs_tr_n).reshape(-1,np.shape(zs_tr_n)[-1])
        y = np.array(scores_tr_n).reshape(-1,np.shape(scores_tr_n)[-1])

        clf_method = 'SVR'
        if clf_method=='SVR':
            print "Start to fit SVR with gamma="
            from sklearn.svm import SVR
            #clf = SVR(C=1.0, epsilon=0.2, kernel='rbf', degree=3, gamma=1.)
            clf = SVR(C=1.0, epsilon=0.2, kernel='rbf', degree=3, gamma=2.5)
        elif clf_method=='RF':
            print "Start to fit RF : ", np.shape(x), np.shape(y)
            from sklearn.ensemble import RandomForestRegressor
            clf = RandomForestRegressor(n_estimators=100, min_samples_leaf=1, n_jobs=1)
        elif clf_method == 'KNN':
            print "Start to fit KNN"
            from sklearn.neighbors import KNeighborsRegressor 
            clf = KNeighborsRegressor(n_neighbors=5, n_jobs=1)
        elif clf_method=='GP':
            from sklearn import gaussian_process
            clf = gaussian_process.GaussianProcess(regr='linear', theta0=5.0, \
                                                   corr='squared_exponential', \
                                                   normalize=True, nugget=1)

            u, idx = np.unique(x,axis=0,return_index=True)
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
        #scaler = preprocessing.StandardScaler()
        scaler = preprocessing.MinMaxScaler()
        x = scaler.fit_transform(x)            
            
        ## print np.shape(x), np.shape(y)
        clf.fit(x, y)

    #--------------------------------------------------------------------
    def get_pos_neg(zs, scores, method=None):
        s_preds = []
        MSEs    = []
        err_ups = []
        for i, s in enumerate(scores):
            if dyn_ths:
                x = scaler.transform(zs[i])
                #x = zs[i]
                if clf_method == 'SVR' or clf_method == 'KNN':
                    s_preds.append( clf.predict(x) )
                elif clf_method == 'RF':
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
        idx_ll = []
        for ths in ths_l:

            if method == 'rnd':
                p_ll.append([1]*int(len(scores)*ths))
                n_ll.append([1]*(len(scores)-int(len(scores)*ths)))                
                continue
            else:
                p_l = []
                n_l = []
            idx_l = []                    
                    

            for i, s in enumerate(scores):
                if dyn_ths:
                    if clf_method == 'SVR' or clf_method == 'KNN':
                        vals = s_preds[i]+ths
                    elif clf_method == 'RF':
                        vals = s_preds[i] + ths*err_ups[i]
                    else:
                        vals = s_preds[i][:,0] + ths*np.sqrt(MSEs[i])
                else:
                    #if method == 'rnd':
                    #    vals = np.ones(len(s_preds[i])) * np.random.choice([-1,1], size=1,
                    #                                                       p=[ths, 1.0-ths])
                    #else:
                    vals = np.array(s_preds[i])+ths

                pos_cnt = 0
                for j in xrange(0,len(s)):
                    if s[j]>vals[j]:
                        if pos_cnt>0:
                            p_l.append(1)
                            idx_l.append(j)
                            break
                        else:
                            pos_cnt += 1
                            
                    ## elif j == len(s)-1:
                    if j == len(s)-1:
                        n_l.append(1)
                        idx_l.append(None)

            n_ll.append(n_l)
            p_ll.append(p_l)
            idx_ll.append(idx_l)

        return p_ll, n_ll, idx_ll

    #--------------------------------------------------------------------
    _, _, tr_a_idx_ll         = get_pos_neg(zs_tr_a, scores_tr_a, method=method)
    fp_ll, tn_ll, _           = get_pos_neg(zs_te_n, scores_te_n, method=method)
    tp_ll, fn_ll, te_a_idx_ll = get_pos_neg(zs_te_a, scores_te_a, method=method)
            
    #--------------------------------------------------------------------
    tpr_l = []
    fpr_l = []
    for i in xrange(len(ths_l)):
        tpr_l.append( float(np.sum(tp_ll[i]))/float(np.sum(tp_ll[i])+np.sum(fn_ll[i]))*100.0 )
        fpr_l.append( float(np.sum(fp_ll[i]))/float(np.sum(fp_ll[i])+np.sum(tn_ll[i]))*100.0 )
        
    from sklearn import metrics
    roc = metrics.auc(fpr_l, tpr_l, True)

    return_idx = kwargs.get('return_idx', False)
    if return_idx:
        dd = {}
        dd['tr_a_idx'] = tr_a_idx_ll
        dd['te_a_idx'] = te_a_idx_ll
        dd['tr_a_err'] = d['err_tr_a']
        dd['te_a_err'] = d['err_te_a']
        return tp_ll, tn_ll, fp_ll, fn_ll, roc, dd
    else:
        return tp_ll, tn_ll, fp_ll, fn_ll, roc



def get_anomaly_score(X, vae, enc_z_mean, enc_z_logvar, window_size, alpha, ad_method, method,\
                      nSample=1000, stateful=False, x_std_div=1, x_std_offset=1e-10, z_std=0.5,\
                      phase=1.0, batch_info=(False,None), ref_scores=None, **kwargs):

    x_dim  = len(X[0][0])
    length = len(X[0])
    train_flag = kwargs.get('train_flag', False)
    valData    = kwargs.get('valData', None)
    return_err = kwargs.get('return_err', False)


    scores = []
    zs = []
    es = []
    for i in xrange(len(X)): # per sample
        #print "sample: ", i+1, " out of ", len(X)
        np.random.seed(3334 + i)
            
        if window_size>0:
            if method.find('pred')>=0:                
                x,y = vutil.create_dataset(X[i], window_size, 1)
            else:
                x = vutil.sampleWithWindow(X[i:i+1], window=window_size)
        else:
            x = X[i:i+1]
        if type(x) is list: x = np.array(x)

        if method == 'rnd':            
            scores.append( np.zeros(len(x)) )            
            continue

        if stateful:
            vae.reset_states()
            if enc_z_mean is not None: enc_z_mean.reset_states()
            if enc_z_logvar is not None: enc_z_logvar.reset_states()                

        z = []
        s = []
        e = []
        last_inputs = None
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

            # Get prediction
            if (method.find('lstm_vae_custom')>=0 or method.find('lstm_dvae_custom')>=0 or\
                method.find('phase')>=0) and method.find('pred')<0 and method.find('input')<0:
                x_true = np.concatenate((xx, np.zeros((len(xx), len(xx[0]),1))), axis=-1)
            elif method.find('input')>=0:
                if last_inputs is None: last_inputs = xx
                x_true = np.concatenate((xx, np.zeros((len(xx), len(xx[0]),1)), last_inputs),
                                        axis=-1)                
                last_inputs = xx
            elif method.find('pred')>=0:
                x_true = np.concatenate((xx, np.zeros((len(xx), len(xx[0]),1)),xx), axis=-1)
            elif method == 'ae' or method == 'vae':
                x_true = xx.reshape((-1,window_size*x_dim))
                batch_info = (None, 1)
            else:
                x_true = xx
            x_pred  = vae.predict(x_true, batch_size=batch_info[1])

            # Get mean and std
            x_mean = None; x_std = None
            if method == 'lstm_vae_custom3':
                x_mean = np.mean(x_pred, axis=0) #length x dim
                x_std  = np.std(x_pred, axis=0)
            elif method == 'ae':
                x_mean = x_pred.reshape((-1,x_dim))
                ## x_mean = np.expand_dims(x_pred.reshape((-1,x_dim)), axis=0)
                
            else:
                x_pred = x_pred[0]
                # length x dim
                x_mean = x_pred[:,:x_dim]
                if enc_z_logvar is not None or len(x_pred[0])>x_dim:
                    x_std  = np.sqrt(x_pred[:,x_dim:]/x_std_div+x_std_offset)

            #---------------------------------------------------------------
            if ad_method == 'recon_prob':
                # Method 1: Reconstruction probability
                l = get_reconstruction_err_prob(xx, x_mean, x_std, alpha=alpha)
                s.append(l)
                z.append( enc_z_mean.predict(xx, batch_size=batch_info[1])[0].tolist() )                
            elif ad_method == 'recon_err':
                # Method 1: Reconstruction error
                s.append( get_reconstruction_err(xx, x_mean, alpha=alpha) )
                z.append( enc_z_mean.predict(xx, batch_size=batch_info[1])[0].tolist() )                
            elif ad_method == 'recon_err_vec':
                # Method 1: Reconstruction prob from vectors?
                l = get_reconstruction_err_prob(y[j], x_mean, x_std, alpha=alpha)
                s.append(l)
            elif ad_method == 'recon_err_lld':
                # Method 1: Reconstruction likelihhod
                if ref_scores is not None:

                    #maximum likelihood-based fitting of MVN?
                    ## import mvn
                    ## mu, cov = mvn.fit_mvn_param( np.array(ref_scores).reshape((-1,x_dim)) )
                    ## print np.shape(mu)
                    ## print np.shape(cov)
                    
                    #instead we use diagonal co-variance                   
                    mu  = np.mean(np.array(ref_scores).reshape((-1,x_dim)), axis=0)
                    var = cov = np.var(np.array(ref_scores).reshape((-1,x_dim)), axis=0)
                    e   = abs(xx[0]-x_mean)

                    ## from scipy.stats import multivariate_normal
                    ## ss = multivariate_normal.pdf(e, mean=mu, cov=cov)
                                        
                    a_score = np.mean(np.sum( ((e-mu)**2)/var, axis=-1))
                    s.append( a_score )
                else:
                    e = abs(xx[0]-x_mean)
                    #print np.shape(xx[0]), np.shape(x_mean)                    
                    # it returns sample x (length x window_size) x dim?
                    # it should return sample x window
                    if len(s) == 0: s = e
                    else:           s = np.concatenate((s,e), axis=0)
            elif ad_method == 'lower_bound':

                if method.find('lstm_vae_custom')>=0 or method.find('lstm_dvae_custom')>=0 or\
                    method.find('phase')>=0 or method.find('pred')>=0:
                    if method.find('circle')>=0:
                        p = float(j)/float(length-window_size+1)
                    else:
                        p = float(j)/float(length-window_size+1)
                        ## p = float(j)/float(length-window_size+1) *2.0*phase-phase
                        
                    if train_flag:
                        p = p*np.ones((batch_info[1], window_size, 1))
                    else:
                        #temp
                        p = p*np.ones((batch_info[1], window_size, 1))
                        ## p = np.zeros((len(x), len(x[0]),1))
                else:
                    p = None
                
                # Method 2: Lower bound
                l, z_mean, z_log_var = ad_metrics.get_lower_bound(xx, x_mean, x_std, z_std,
                                                                  enc_z_mean, enc_z_logvar,\
                                                                  x_dim, method, p, alpha=alpha)
                if return_err:
                    ## print "Start to get error vectors"
                    err = ad_metrics.get_err_vec(xx, x_mean, x_std, x_dim, method, p=p, alpha=alpha)
                    err = np.squeeze(err)
                    e.append(err.tolist())
                                                                  
                s.append(l)
                z.append(z_mean.tolist()) # + z_log_var.tolist())

        ## s = np.cumsum(s)
        ## if len(np.shape(s))<2: s = np.expand_dims(s, axis=1)
        scores.append(s) # s is scalers
        zs.append(z)
        es.append(e)

    if return_err: return scores, zs, es
    else:          return scores, zs


def get_err_vec(X, vae, enc_z_mean, enc_z_logvar, window_size, alpha, ad_method, method,\
                nSample=1000,\
                stateful=False, x_std_div=1, x_std_offset=1e-10, z_std=0.5,\
                phase=1.0,\
                batch_info=(False,None), ref_scores=None, **kwargs):

    x_dim = len(X[0][0])
    length = len(X[0])
    train_flag = kwargs.get('train_flag', False)
    valData    = kwargs.get('valData', None)


    scores = []
    zs = []
    for i in xrange(len(X)): # per sample
        #print "sample: ", i+1, " out of ", len(X)
        np.random.seed(3334 + i)
            
        if window_size>0:
            if method.find('pred')>=0:                
                x,y = vutil.create_dataset(X[i], window_size, 1)
            else:
                x = vutil.sampleWithWindow(X[i:i+1], window=window_size)
        else:
            x = X[i:i+1]
        if type(x) is list: x = np.array(x)

        if method == 'rnd':            
            scores.append( np.zeros(len(x)) )            
            continue

        if stateful:
            vae.reset_states()
            if enc_z_mean is not None: enc_z_mean.reset_states()
            if enc_z_logvar is not None: enc_z_logvar.reset_states()                

        z = []
        s = []
        e = []
        last_inputs = None
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

            # Get prediction
            if (method.find('lstm_vae_custom')>=0 or method.find('lstm_dvae_custom')>=0 or\
                method.find('phase')>=0) and method.find('pred')<0 and method.find('input')<0:
                x_true = np.concatenate((xx, np.zeros((len(xx), len(xx[0]),1))), axis=-1)
            elif method.find('input')>=0:
                if last_inputs is None: last_inputs = xx
                x_true = np.concatenate((xx, np.zeros((len(xx), len(xx[0]),1)), last_inputs),
                                        axis=-1)                
                last_inputs = xx
            elif method.find('pred')>=0:
                x_true = np.concatenate((xx, np.zeros((len(xx), len(xx[0]),1)),xx), axis=-1)
            elif method == 'ae' or method == 'vae':
                x_true = xx.reshape((-1,window_size*x_dim))
                batch_info = (None, 1)
            else:
                x_true = xx
            x_pred  = vae.predict(x_true, batch_size=batch_info[1])

            # Get mean and std
            x_mean = None; x_std = None
            if method == 'lstm_vae_custom3':
                x_mean = np.mean(x_pred, axis=0) #length x dim
                x_std  = np.std(x_pred, axis=0)
            elif method == 'ae':
                x_mean = x_pred.reshape((-1,x_dim))
                ## x_mean = np.expand_dims(x_pred.reshape((-1,x_dim)), axis=0)
                
            else:
                x_pred = x_pred[0]
                # length x dim
                x_mean = x_pred[:,:x_dim]
                if enc_z_logvar is not None or len(x_pred[0])>x_dim:
                    x_std  = np.sqrt(x_pred[:,x_dim:]/x_std_div+x_std_offset)

            #---------------------------------------------------------------
            if ad_method == 'recon_prob':
                # Method 1: Reconstruction probability
                l = get_reconstruction_err_prob(xx, x_mean, x_std, alpha=alpha)
                s.append(l)
                z.append( enc_z_mean.predict(xx, batch_size=batch_info[1])[0].tolist() )                
            elif ad_method == 'recon_err':
                # Method 1: Reconstruction error
                s.append( get_reconstruction_err(xx, x_mean, alpha=alpha) )
                z.append( enc_z_mean.predict(xx, batch_size=batch_info[1])[0].tolist() )                
            elif ad_method == 'recon_err_vec':
                # Method 1: Reconstruction prob from vectors?
                l = get_reconstruction_err_prob(y[j], x_mean, x_std, alpha=alpha)
                s.append(l)
            elif ad_method == 'recon_err_lld':
                # Method 1: Reconstruction likelihhod
                if ref_scores is not None:

                    #maximum likelihood-based fitting of MVN?
                    ## import mvn
                    ## mu, cov = mvn.fit_mvn_param( np.array(ref_scores).reshape((-1,x_dim)) )
                    ## print np.shape(mu)
                    ## print np.shape(cov)
                    
                    #instead we use diagonal co-variance                   
                    mu  = np.mean(np.array(ref_scores).reshape((-1,x_dim)), axis=0)
                    var = cov = np.var(np.array(ref_scores).reshape((-1,x_dim)), axis=0)
                    e   = abs(xx[0]-x_mean)

                    ## from scipy.stats import multivariate_normal
                    ## ss = multivariate_normal.pdf(e, mean=mu, cov=cov)
                                        
                    a_score = np.mean(np.sum( ((e-mu)**2)/var, axis=-1))
                    s.append( a_score )
                else:
                    e = abs(xx[0]-x_mean)
                    #print np.shape(xx[0]), np.shape(x_mean)                    
                    # it returns sample x (length x window_size) x dim?
                    # it should return sample x window
                    if len(s) == 0: s = e
                    else:           s = np.concatenate((s,e), axis=0)
            elif ad_method == 'lower_bound':

                if method.find('lstm_vae_custom')>=0 or method.find('lstm_dvae_custom')>=0 or\
                    method.find('phase')>=0 or method.find('pred')>=0:
                    p = float(j)/float(length-window_size+1) *2.0*phase-phase
                    if train_flag:
                        p = p*np.ones((batch_info[1], window_size, 1))
                    else:
                        #temp
                        p = p*np.ones((batch_info[1], window_size, 1))
                        ## p = np.zeros((len(x), len(x[0]),1))
                else:
                    p = None
                
                # Method 2: Lower bound
                l, z_mean, z_log_var = ad_metrics.get_lower_bound(xx, x_mean, x_std, z_std,
                                                                  enc_z_mean, enc_z_logvar,\
                                                                  x_dim, method, p, alpha=alpha)
                s.append(l)
                z.append(z_mean.tolist()) # + z_log_var.tolist())

        ## s = np.cumsum(s)
        ## if len(np.shape(s))<2: s = np.expand_dims(s, axis=1)
        scores.append(s) # s is scalers
        zs.append(z)

    return scores, zs



def pred_rf(model, x, percentile=95):

    preds = []
    for pred in model.estimators_:
        preds.append(pred.predict(x)[0])
    err_down = np.percentile(preds, (100 - percentile) / 2. )
    err_up   = np.percentile(preds, 100 - (100 - percentile) / 2.)

    return err_down, err_up


def get_optimal_alpha(inputs, vae, vae_mean, ad_method, method, window_size, save_pkl,\
                      stateful=False, renew=False,\
                      x_std_div=1.0, x_std_offset=1e-10, z_std=None,\
                      dyn_ths=False, batch_info=(False,None), **kwargs):

    X_nor, X_abnor = inputs
    enc_z_logvar = None    
    nDim    = len(X_nor[0,0])
    ## nSample = len(X_nor)
    p_ll = [[] for i in xrange(nDim) ]

    ## labels = np.vstack([ np.ones((len(X_nor), len(X_nor[0]))), -np.ones((len(X_abnor), len(X_abnor[0])))  ])
    #labels = [1.]*len(X_nor)*len(X_nor[0])+[-1.]*len(X_abnor)*len(X_abnor[0])
    labels = [1.]*len(X_nor)

    if os.path.isfile(save_pkl) and renew is False:
        d = ut.load_pickle(save_pkl)
        ## nSample    = d['nSample']
        p_ll       = d['p_ll']
        ## labels     = d['labels']
    else:

        #X = np.vstack([X_nor, X_abnor])
        X = X_nor
                
        p_ll = []
        for i in xrange(len(X)): # per sample
            print "sample: ", i+1, " out of ", len(X), np.shape(p_ll)
            np.random.seed(3334 + i)

            if window_size>0: x = vutil.sampleWithWindow(X[i:i+1], window=window_size)
            else:             x = X[i:i+1]
            if type(x) is list: x = np.array(x)

            if stateful:
                vae.reset_states()
                vae_mean.reset_states()
                #if enc_z_mean is not None: enc_z_mean.reset_states()
                #if enc_z_logvar is not None: enc_z_logvar.reset_states()                

            p_l = []
            for j in xrange(len(x)): # per window

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

                if (method.find('lstm_vae')>=0 or method.find('lstm_dvae')>=0 or method.find('phase')>=0) and\
                    method.find('pred')<0 and method.find('input')<0:
                    x_true = np.concatenate((xx, np.zeros((len(xx), len(xx[0]),1))), axis=-1)
                elif method.find('pred')>=0 or method.find('input')>=0:
                    x_true = np.concatenate((xx, np.zeros((len(xx), len(xx[0]),1)), xx), axis=-1) 
                else:
                    x_true = xx

                # x_true : batch x timesteps x dim +1
                x_pred  = vae_mean.predict(x_true, batch_size=batch_info[1])

                # Get mean and std
                if method == 'lstm_vae_custom3':
                    # Real sampling
                    x_mean = np.mean(x_pred, axis=0) #length x dim
                    x_std  = np.std(x_pred, axis=0)
                else:
                    x_pred = x_pred[0]
                    # x_pred: length x dim
                    x_mean = x_pred[:,:nDim]
                    if enc_z_logvar is not None or len(x_pred[0])>nDim:
                        x_std  = np.sqrt(x_pred[:,nDim:]/x_std_div+x_std_offset)
                    else:
                        x_std = None

                #log_p_x_z = -0.5 * ( np.sum( (alpha*(x-x_mean)/x_std)**2, axis=-1) )
                #log_p_x_z = np.sum( (alpha*(x-x_mean)/x_std)**2, axis=-1) 
                #---------------------------------------------------------------
                # anomaly score : window x dim
                p_l.append( ((xx[0]-x_mean)/x_std)**2 )
                
            ## p_l     = []
            ## for l in xrange(len(x_mean)): # per length
            ##     p = ((xx[0,l]-x_mean[l])/x_std[l])**2
            ##     p_l.append(p.tolist()) # length x dim

            max_idx = np.argmax(np.sum(np.sum(p_l, axis=-1), axis=-1))
            p_ll.append( p_l[max_idx][0].tolist() )# sample x dim
                                
        d = {'p_ll': p_ll, 'labels': labels}
        ut.save_pickle(d, save_pkl)

    print "p_ll: ", np.shape(p_ll)
    print "labels: ", np.shape(labels)


    print np.linalg.norm(p_ll, axis=0)

    r  = np.sum(p_ll, axis=0)/np.sum(p_ll)
    r  = 1.0-r
    #r  = (r-np.amin(r))/(np.amax(r)-np.amin(r))*0.6+0.4
    print r
    return r

    
    def score_func(X, *args):
        '''
        X      : dim
        args[0]: dim
        '''
        return float(np.sum( X.dot(args[0])*args[1] ))
    
    def const(X): return np.sum(X)-0.5


    from scipy.optimize import minimize
    x0   = np.array([1.0]*nDim) /float(nDim)
    bnds = [[1.0/float(nDim)/4.,x0[0]] for i in xrange(nDim) ]
    res  = minimize(score_func, x0, args=(p_ll[:,:len(X_nor)*len(X_nor[0])],
                                          np.array(labels)[:len(X_nor)*len(X_nor[0])]),
                                          method='SLSQP', tol=1e-6, bounds=bnds,
                    constraints={'type':'ineq', 'fun': const}, options={'disp': False})

    print res
    sys.exit()
    
    return res.x
