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

# util
import numpy as np
import scipy

from hrl_anomaly_detection.classifiers.clf_base import clf_base
from sklearn import gaussian_process

class hmmgp(clf_base):
    def __init__(self, nPosteriors=10, ths_mult=-1.0, nugget=100.0, theta0=1.0,\
                 hmmgp_logp_offset=0., verbose=False, **kwargs):
        ''' '''        
        clf_base.__init__(self)
        
        self.nPosteriors = nPosteriors
        self.ths_mult    = ths_mult
        self.verbose     = verbose

        self.regr = 'linear' #'constant' #'constant', 'linear', 'quadratic'
        self.corr = 'squared_exponential' 
        self.nugget = nugget
        self.theta0 = theta0
        self.hmmgp_logp_offset = hmmgp_logp_offset

        self.dt = gaussian_process.GaussianProcess(regr=self.regr, theta0=self.theta0, corr=self.corr, \
                                                   normalize=True, nugget=self.nugget)            
        
    def fit(self, X, y, ll_idx=None):
        
        if type(X) == list: X = np.array(X)

        # extract only negatives
        ll_logp = [ X[i,0] for i in xrange(len(X)) if y[i]<0 ]
        ll_post = [ X[i,-self.nPosteriors:] for i in xrange(len(X)) if y[i]<0 ]

        # to prevent multiple same input we add noise into X
        ll_post = np.array(ll_post) + np.random.normal(0.0, 0.001, np.shape(ll_post))

        if False:
            from sklearn.utils import check_array
            ll_logp = check_array(ll_logp).T
            import sandbox_dpark_darpa_m3.lib.gaussian_process.spgp.spgp as gp
            self.dt = gp.Gaussian_Process(ll_post,ll_logp,M=400)
            self.dt.training('./spgp_obs.pkl', renew=True)
        else:
            self.dt.fit( ll_post, ll_logp )          
            ## idx_list = range(len(ll_post))
            ## random.shuffle(idx_list)
            ## self.dt.fit( ll_post[idx_list[:600]], np.array(ll_logp)[idx_list[:600]])          

        return

    def partial_fit(self, X, y=None, X_idx=None, shuffle=True, **kwargs):
        return

    def predict(self, X, y=None):
        ''' '''

        if len(np.shape(X))==1: X = [X]
        if type(X) is list: X= np.array(X)

        logps = X[:,0]
        posts = X[:,-self.nPosteriors:]

        if False:
            y_pred, sigma = self.dt.predict(posts, True)
        else:
            try:
                y_pred, MSE = self.dt.predict(posts, eval_MSE=True)
                sigma = np.sqrt(MSE)
            except:
                print "posterior probability is weired"
                ## for i, post in enumerate(posts):                        
                ##     print i, post
                #sys.exit()
                return np.ones(len(posts))

        ## mult_coeff = []
        ## for post in posts:
        ##     min_index = np.argmax(post)
        ##     ## mult_coeff.append( 1.0 + 0.* float(min_index)/(float(self.nPosteriors)-1.0) )
        ##     mult_coeff.append( 1.0 + 3.* float(min_index)/(float(self.nPosteriors)-1.0) )
        ## mult_coeff = np.array(mult_coeff)

        ## l_err = y_pred + mult_coeff*self.ths_mult*sigma - logps #- self.logp_offset
        l_err = y_pred + self.ths_mult*sigma - logps - self.hmmgp_logp_offset
        if debug:
            return l_err, y_pred, sigma
        else:
            return l_err                


        return


    def save_model(self, fileName):
        import pickle
        with open(fileName, 'wb') as f:
            pickle.dump(self.dt, f)
            pickle.dump(self.ths_mult, f)
            pickle.dump(self.hmmgp_logp_offset, f)

    def load_model(self, fileName):        
        import pickle
        with open(fileName, 'rb') as f:
            self.dt       = pickle.load(f)
            self.ths_mult = pickle.load(f)
            self.hmmgp_logp_offset = pickle.load(f)

    
        


