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

# system
import os, sys, copy, time
from joblib import Parallel, delayed

# util
import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.stats import norm, entropy

from hrl_anomaly_detection.classifiers.clf_base import clf_base
from hrl_anomaly_detection.classifiers import classifier_util as cutil
import hrl_lib.util as ut

class hmmd(clf_base):
    def __init__(self, nPosteriors=10, nLength=None, startIdx=4, parallel=False,\
                 ths_mult=-1.0, std_coff=1.0, logp_offset=0.0, verbose=False, **kwargs):
        ''' '''        
        clf_base.__init__(self)

        self.nPosteriors = nPosteriors
        self.nLength     = nLength
        self.startIdx    = startIdx
        self.parallel    = parallel
        self.dt          = None
        self.verbose     = verbose

        self.ths_mult    = ths_mult
        self.std_coff    = std_coff
        self.logp_offset = logp_offset
        self.ll_mu  = np.zeros(nPosteriors)
        self.ll_std = np.zeros(nPosteriors)
        self.l_statePosterior = None


    def fit(self, X, y=None, ll_idx=None):

        if type(X) == list: X = np.array(X)
        if ll_idx is None:
            # Need to artificially generate ll_idx....
            nSample  = len(y)/(self.nLength-self.startIdx)
            idx_list = range(self.nLength)[self.startIdx:]
            ll_idx = [ idx_list[j] for j in xrange(len(idx_list)) for i in xrange(nSample) if y[i*(self.nLength-self.startIdx)+1]<0 ]
        else:
            if len(np.shape(y))>1: y = np.array(y)[:,0]
            ll_idx  = [ ll_idx[i] for i in xrange(len(ll_idx)) if y[i]<0 ]                

        ll_logp = [ X[i,0] for i in xrange(len(X)) if y[i]<0 ]
        ll_post = [ X[i,-self.nPosteriors:] for i in xrange(len(X)) if y[i]<0 ]

        self.g_mu_list = np.linspace(0, self.nLength-1, self.nPosteriors)
        self.g_sig = float(self.nLength) / float(self.nPosteriors) * self.std_coff

        if self.parallel:
            r = Parallel(n_jobs=-1)(delayed(learn_time_clustering)(i, ll_idx, ll_logp, ll_post, \
                                                                   self.g_mu_list[i],\
                                                                   self.g_sig, self.nPosteriors)
                                                                   for i in xrange(self.nPosteriors))
            _, self.l_statePosterior, self.ll_mu, self.ll_std = zip(*r)
        else:
            self.l_statePosterior = []
            self.ll_mu            = []
            self.ll_std           = []
            for i in xrange(self.nPosteriors):
                _,p,m,s = learn_time_clustering(i, ll_idx, ll_logp, ll_post, self.g_mu_list[i],\
                                              self.g_sig, self.nPosteriors)
                self.l_statePosterior.append(p)
                self.ll_mu.append(m)
                self.ll_std.append(s)

        return True
    

    def partial_fit(self, X, y=None, X_idx=None, **kwargs):

        assert len(X)==len(y), "X and y sizes should be same"
        assert len(X)==len(X_idx), "X and y sizes should be same"
        if type(X) == list: X = np.array(X)

        # Get params for new data
        ll_idx  = [ X_idx[i] for i in xrange(len(X_idx)) if y[i]<0 ]                            
        ll_logp = [ X[i,0] for i in xrange(len(X)) if y[i]<0 ]
        ll_post = [ X[i,-self.nPosteriors:] for i in xrange(len(X)) if y[i]<0 ]

        # Update using bayesian inference
        mu_mu   = kwargs['mu_mu']
        std_mu  = kwargs['std_mu']
        mu_std  = kwargs['mu_std']
        std_std = kwargs['std_std']

        ## # 1) Find all likelihood in ith cluster
        ## l_idx = []
        ## ll_c_post = [[] for i in xrange(self.nPosteriors)]
        ## ll_c_logp = [[] for i in xrange(self.nPosteriors)]
        ## for i, post in enumerate(ll_post):
        ##     min_index, min_dist = findBestPosteriorDistribution(post, self.l_statePosterior)
        ##     ll_c_post[min_index].append(post)
        ##     ll_c_logp[min_index].append(ll_logp[i])


        for i in xrange(self.nPosteriors):
            # 1-new) Find time-based weight per cluster
            ## weights = norm(loc=self.g_mu_list[i], scale=self.g_sig).pdf(ll_idx) 

            # 1-new-new) Find KL-based weight per cluster
            weights = []
            for j, post in enumerate(ll_post):
                weights.append( 1.0 / cutil.symmetric_entropy(post, self.l_statePosterior[i]) )
            weights = np.array(weights)**2
            weights = [w if w > 0.01 else 0.0 for w in weights ]
            weights = [w if w < 1.0 else 1.0 for w in weights ]

            # 2) Run optimization
            x0 = [mu_mu[i], mu_std[i]]
            res = minimize(param_posterior, x0, args=(ll_logp, weights,\
                                                      mu_mu[i], std_mu[i], mu_std[i], std_std[i]),
                           method='L-BFGS-B',
                           bounds=((mu_mu[i]-15.0*std_mu[i], mu_mu[i]+15.0*std_mu[i]),
                                   (1e-5, mu_std[i]+15.0*std_std[i]))
                                   )

            self.ll_mu = list(self.ll_mu); self.ll_std = list(self.ll_std)
            self.ll_mu[i]  = res.x[0]
            self.ll_std[i] = res.x[1]
        
        return

    def predict(self, X, y=None, debug=False):
        ''' '''

        if len(np.shape(X))==1: X = [X]

        l_err = []
        for i in xrange(len(X)):
            logp = X[i][0]
            post = X[i][-self.nPosteriors:]

            # Find the best posterior distribution
            try:
                min_index, min_dist = cutil.findBestPosteriorDistribution(post, self.l_statePosterior)
            except:
                print i
                print self.l_statePosterior
                sys.exit()
            nState = len(post)

            if (type(self.ths_mult) == list or type(self.ths_mult) == np.ndarray or \
                type(self.ths_mult) == tuple) and len(self.ths_mult)>1:
                err = (self.ll_mu[min_index] + self.ths_mult[min_index]*self.ll_std[min_index]) - logp - self.logp_offset                        
            else:
                err = (self.ll_mu[min_index] + self.ths_mult*self.ll_std[min_index]) - logp - self.logp_offset

            l_err.append(err)

        if debug:
            return l_err, self.ll_mu[min_index], self.ll_std[min_index]
        else:
            return l_err

    def score(self, X, y):
        return self.dt.score(X,y)


    def save_model(self, fileName):
        d = {'g_mu_list': self.g_mu_list, 'g_sig': self.g_sig, \
             'l_statePosterior': self.l_statePosterior,\
             'll_mu': self.ll_mu, 'll_std': self.ll_std}
        ut.save_pickle(d, fileName)            

    def load_model(self, fileName):        
        print "Start to load a progress based classifier"
        d = ut.load_pickle(fileName)
        self.g_mu_list = d['g_mu_list']
        self.g_sig     = d['g_sig']
        self.l_statePosterior = d['l_statePosterior']
        self.ll_mu            = d['ll_mu']
        self.ll_std           = d['ll_std']

        



####################################################################
# functions for paralell computation
####################################################################

def learn_time_clustering(i, ll_idx, ll_logp, ll_post, g_mu, g_sig, nState):

    weights     = norm(loc=g_mu, scale=g_sig).pdf(ll_idx)
    weight_sum  = np.sum(weights)
    weight2_sum = np.sum(weights**2)

    g_post = np.zeros(nState)
    g_lhood = 0.0
    ## g_post  = np.matmul(weights,ll_post) 
    ## g_lhood = np.sum( ll_logp * weights )
    for j in xrange(len(ll_idx)):
        if weights[j] < 1e-3: continue
        g_post   += ll_post[j] * weights[j]
        g_lhood  += ll_logp[j] * weights[j]

    if abs(weight_sum)<1e-3: weight_sum=1e-3
    l_statePosterior  = g_post / weight_sum 
    l_likelihood_mean = g_lhood / weight_sum 

    g_lhood2 = np.sum( weights * ( (ll_logp-l_likelihood_mean)**2 ) )
        
    ## print g_lhood2/(weight_sum - weight2_sum/weight_sum), weight_sum - weight2_sum/weight_sum, weight_sum 
    l_likelihood_std = np.sqrt(g_lhood2/(weight_sum - weight2_sum/weight_sum))

    return i, l_statePosterior, l_likelihood_mean, l_likelihood_std

def param_posterior(x, l, w, mu_mu, std_mu, mu_std, std_std):

    mu_n  = x[0]
    std_n = x[1]

    p = 0
    N = float(len(l))
    if isinstance(l, list): l = np.array(l)
    if isinstance(w, list): w = np.array(w)
    if std_n < 1e-5: p = 100000000000
    
    # 1st term
    p += np.log(std_n)*N + np.sum( w*(mu_n-l)**2 )/(2.0*std_n**2)
    
    # 2nd term
    p += (mu_n-mu_mu)**2 / (2.0*std_mu**2) #*0.5
    
    # 3rd term
    p += (std_n-mu_std)**2 / (2.0*std_std**2) #*0.5

    return p

    
