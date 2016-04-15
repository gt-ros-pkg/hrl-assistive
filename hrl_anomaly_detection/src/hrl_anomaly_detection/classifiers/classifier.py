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
import os, sys, copy

# visualization
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec

# util
import numpy as np
import scipy
import hrl_lib.util as ut

from scipy.stats import norm, entropy
from joblib import Parallel, delayed
from hrl_anomaly_detection.hmm.learning_base import learning_base

class classifier(learning_base):
    def __init__(self, method='svm', nPosteriors=10, nLength=200, ths_mult=-1.0,\
                 class_weight=1.0, \
                 #progress
                 logp_offset = 0.0,\
                 # svm
                 svm_type    = 0,\
                 kernel_type = 2,\
                 degree      = 3,\
                 gamma       = 0.3,\
                 cost        = 4.,\
                 w_negative  = 7.0,\
                 verbose=False):
        '''
        class_weight : positive class weight for svm
        nLength : only for progress-based classifier
        ths_mult: only for progress-based classifier
        '''              
        self.method = method
        self.dt     = None
        self.verbose = verbose

        if self.method == 'svm':
            sys.path.insert(0, '/usr/lib/pymodules/python2.7')
            import svmutil as svm
            self.class_weight = class_weight
            self.svm_type    = svm_type
            self.kernel_type = kernel_type
            self.degree      = degree 
            self.gamma       = gamma 
            self.cost        = cost 
            self.w_negative  = w_negative             
        elif self.method == 'cssvm_standard' or self.method == 'cssvm':
            sys.path.insert(0, os.path.expanduser('~')+'/git/cssvm/python')
            import cssvmutil as cssvm
            self.class_weight = class_weight
            self.svm_type    = svm_type
            self.kernel_type = kernel_type
            self.degree      = degree 
            self.gamma       = gamma 
            self.cost        = cost 
            self.w_negative  = w_negative 
        elif self.method == 'progress_time_cluster':
            self.nLength   = nLength
            self.std_coff  = 1.0
            self.nPosteriors = nPosteriors
            self.ths_mult = ths_mult
            self.logp_offset = logp_offset
            self.ll_mu  = np.zeros(nPosteriors)
            self.ll_std = np.zeros(nPosteriors) 
        elif self.method == 'fixed':
            self.mu  = 0.0
            self.std = 0.0
            self.ths_mult = ths_mult
            
        learning_base.__init__(self)

    def fit(self, X, y, ll_idx=None, parallel=True):
        '''
        ll_idx is the index list of each sample in a sequence.
        '''

        # saved file check.

        if self.method == 'svm':
            sys.path.insert(0, '/usr/lib/pymodules/python2.7')
            import svmutil as svm
            ## print svm.__file__
            if type(X) is not list: X=X.tolist()
            commands = '-q -s '+str(self.svm_type)+' -t '+str(self.kernel_type)+' -d '+str(self.degree)\
              +' -g '+str(self.gamma)\
              +' -c '+str(self.cost)+' -w1 '+str(self.class_weight)\
              +' -w-1 '+str(self.w_negative)
            try: self.dt = svm.svm_train(y, X, commands )
            except: return False
            return True
        elif self.method == 'cssvm_standard':
            sys.path.insert(0, os.path.expanduser('~')+'/git/cssvm/python')
            import cssvmutil as cssvm
            if type(X) is not list: X=X.tolist()
            self.dt = cssvm.svm_train(y, X, '-C 0 -c 4.0 -t 2 -w1 '+str(self.class_weight)+' -w-1 5.0' )
            return True
        elif self.method == 'cssvm':
            sys.path.insert(0, os.path.expanduser('~')+'/git/cssvm/python')
            import cssvmutil as cssvm
            if type(X) is not list: X=X.tolist()
            commands = '-q -C 1 -s '+str(self.svm_type)+' -t '+str(self.kernel_type)+' -d '+str(self.degree)\
              +' -g '+str(self.gamma)\
              +' -c '+str(self.cost)+' -w1 '+str(self.class_weight)\
              +' -w-1 '+str(self.w_negative) \
              +' -m 200'
            try: self.dt = cssvm.svm_train(y, X, commands )
            except: return False
            return True
            
        elif self.method == 'progress_time_cluster':
            if type(X) == list: X = np.array(X)
            ## ll_logp = X[:,0:1]
            ## ll_post = X[:,1:]
            ll_idx  = [ ll_idx[i] for i in xrange(len(ll_idx)) if y[i]<0 ]
            ll_logp = [ X[i,0] for i in xrange(len(X)) if y[i]<0 ]
            ll_post = [ X[i,1:] for i in xrange(len(X)) if y[i]<0 ]

            g_mu_list = np.linspace(0, self.nLength-1, self.nPosteriors)
            g_sig = float(self.nLength) / float(self.nPosteriors) * self.std_coff

            if parallel:
                r = Parallel(n_jobs=-1)(delayed(learn_time_clustering)(i, ll_idx, ll_logp, ll_post, \
                                                                       g_mu_list[i],\
                                                                       g_sig, self.nPosteriors)
                                                                       for i in xrange(self.nPosteriors))
                _, self.l_statePosterior, self.ll_mu, self.ll_std = zip(*r)
            else:
                self.l_statePosterior = []
                self.ll_mu            = []
                self.ll_std           = []
                for i in xrange(self.nPosteriors):
                    _,p,m,s = learn_time_clustering(i, ll_idx, ll_logp, ll_post, g_mu_list[i],\
                                                  g_sig, self.nPosteriors)
                    self.l_statePosterior.append(p)
                    self.ll_mu.append(m)
                    self.ll_std.append(s)

            return True

        elif self.method == 'fixed':
            if type(X) == list: X = np.array(X)
            ll_logp = X[:,0:1]
            self.mu  = np.mean(ll_logp)
            self.std = np.std(ll_logp)
            return True
                

    def predict(self, X, y=None):
        '''
        X is single sample
        return predicted values (not necessarily binaries)
        '''

        if self.method == 'cssvm_standard' or self.method == 'cssvm' or self.method == 'svm':
            if self.method == 'svm':
                sys.path.insert(0, '/usr/lib/pymodules/python2.7')
                import svmutil as svm
            else:
                sys.path.insert(0, os.path.expanduser('~')+'/git/cssvm/python')
                import cssvmutil as svm

            if self.verbose:
                print svm.__file__
            if type(X) is not list: X=X.tolist()
            if y is not None:
                p_labels, _, p_vals = svm.svm_predict(y, X, self.dt)
            else:
                p_labels, _, p_vals = svm.svm_predict([0]*len(X), X, self.dt)
            return p_labels
        
        elif self.method == 'progress_time_cluster':
            if len(np.shape(X))==1: X = [X]

            l_err = []
            for i in xrange(len(X)):
                logp = X[i][0]
                post = X[i][1:]

                # Find the best posterior distribution
                min_index, min_dist = findBestPosteriorDistribution(post, self.l_statePosterior)
                nState = len(post)
                ## c_time = float(nState - (min_index+1) )/float(nState) + 1.0
                ## c_time = np.logspace(0,-0.9,nState)[min_index]

                if (type(self.ths_mult) == list or type(self.ths_mult) == np.ndarray or \
                    type(self.ths_mult) == tuple) and len(self.ths_mult)>1:
                    err = (self.ll_mu[min_index] + self.ths_mult[min_index]*self.ll_std[min_index]) - logp - self.logp_offset
                else:
                    err = (self.ll_mu[min_index] + self.ths_mult*self.ll_std[min_index]) - logp - self.logp_offset
                l_err.append(err)
            return l_err
        
        elif self.method == 'fixed':
            if len(np.shape(X))==1: X = [X]
                
            l_err = []
            for i in xrange(len(X)):
                logp = X[i][0]
                err = self.mu + self.ths_mult * self.std - logp
                l_err.append(err)
            return l_err 

    ## def predict_batch(self, X, y, idx):

    ##     tp_l = []
    ##     fp_l = []
    ##     tn_l = []
    ##     fn_l = []
    ##     delay_l = []

    ##     for ii in xrange(len(X)):

    ##         if len(y[ii])==0: continue

    ##         for jj in xrange(len(X[ii])):

    ##             est_y = dtc.predict(X[ii][jj], y=y[ii][jj:jj+1])
    ##             if type(est_y) == list: est_y = est_y[0]
    ##             if type(est_y) == list: est_y = est_y[0]
    ##             if est_y > 0.0:
    ##                 delay_idx = idx[ii][jj]
    ##                 print "Break ", ii, " ", jj, " in ", est_y, " = ", y[ii][jj]
    ##                 break        

    ##         if y[ii][0] > 0.0:
    ##             if est_y > 0.0:
    ##                 tp_l.append(1)
    ##                 delay_l.append(delay_idx)
    ##             else: fn_l.append(1)
    ##         elif y[ii][0] <= 0.0:
    ##             if est_y > 0.0: fp_l.append(1)
    ##             else: tn_l.append(1)

    ##     return tp_l, fp_l, tn_l, fn_l, delay_l


    def decision_function(self, X):

        ## return self.dt.decision_function(X)
        if self.method == 'cssvm_standard' or self.method == 'cssvm' or \
          self.method == 'fixed' or self.method == 'svm':
            if type(X) is not list:
                return self.predict(X.tolist())
            else:
                return self.predict(X)
        else:
            print "Not implemented"
            sys.exit()

        return 
        
    def score(self, X, y):
        if self.method == 'svm':
            return self.dt.score(X,y)
        else:
            print "Not implemented funciton Score"
            return 

        
    def save_model(fileName):
        if self.dt is None: 
            print "No trained classifier"
            return
        
        if self.method == 'svm':
            sys.path.insert(0, '/usr/lib/pymodules/python2.7')
            import svmutil as svm            
            svm.svm_save_model(use_pkl, self.dt) 
        else:
            print "Not available method"

            
    def load_model(fileName):        
        if self.method == 'svm':
            sys.path.insert(0, '/usr/lib/pymodules/python2.7')
            import svmutil as svm            
            self.dt = svm.svm_load_model(use_pkl) 
        else:
            print "Not available method"
        
            

        
####################################################################
# functions for distances
####################################################################

def custom_kernel(x1,x2):
    '''
    Similarity estimation between (loglikelihood, state distribution) feature vector.
    kernel must take as arguments two matrices of shape (n_samples_1, n_features), (n_samples_2, n_features)
    and return a kernel matrix of shape (n_samples_1, n_samples_2)
    '''

    if len(np.shape(x1)) == 2: 

        kernel_mat = np.zeros((len(x1), len(x2)))

        r = Parallel(n_jobs=4)(delayed(customDist)(i, j, x1[i,1:], x2[j,1:]) for i in xrange(len(x1)) for j in xrange(len(x2)))
        
        ## l_i, l_j, l_dist = zip(*r)
        for i,j,dist in zip(r):
            kernel_mat[i,j] = dist

        
        ## for i in xrange(len(x1)):
        ##     for j in xrange(len(x2)):

                ## kernel_mat[i,j] += (x1[i,0]-x2[j,0])**2            
                ## kernel_mat[i,j] += 1.0/symmetric_entropy(x1[i,1:], x2[j,1:])

                ## if np.isnan(kernel_mat[i,j]): print "wrong kernel result ", x1, x2

        # normalization?? How do we know bounds?

        return kernel_mat

    else:

        d1 = (x1[0] - x2[0])**2
        d2 = 1.0/symmetric_entropy(x1[1:], x2[1:])
        return d1+d2

def customDist(i,j, x1, x2):
    return i,j,(x1-x2)**2 + 1.0/symmetric_entropy(x1,x2)

def custom_kernel2(x1,x2):
    '''
    Similarity estimation between state distribution feature vector.
    kernel must take as arguments two matrices of shape (n_samples_1, n_features), (n_samples_2, n_features)
    and return a kernel matrix of shape (n_samples_1, n_samples_2)
    '''

    if len(np.shape(x1)) == 2: 

        ## print np.shape(x1), np.shape(x2)
        kernel_mat = scipy.spatial.distance.cdist(x1, x2, 'euclidean')        

        ## for i in xrange(len(x1)):
        ##     for j in xrange(len(x2)):
        ##         ## kernel_mat[i,j] = 1.0/symmetric_entropy(x1[i], x2[j])
        ##         kernel_mat[i,j] = np.linalg.norm(x1[i]-x2[j])

        return kernel_mat

    else:

        ## return 1.0/symmetric_entropy(x1, x2)
        return np.linalg.norm(x1[i]-x2[j])

def symmetric_entropy(p,q):
    '''
    Return the sum of KL divergences
    '''
    return min(entropy(p,np.array(q)+1e-6), entropy(q,np.array(p)+1e-6)) + 1e-6


def findBestPosteriorDistribution(post, l_statePosterior):
    # Find the best posterior distribution
    min_dist  = 100000000
    min_index = 0

    for j in xrange(len(l_statePosterior)):
        dist = symmetric_entropy(post, l_statePosterior[j])
            
        if min_dist > dist:
            min_index = j
            min_dist  = dist

    return min_index, min_dist


####################################################################
# functions for paralell computation
####################################################################

def learn_time_clustering(i, ll_idx, ll_logp, ll_post, g_mu, g_sig, nState):

    l_likelihood_mean = 0.0
    l_likelihood_mean2 = 0.0
    l_statePosterior = np.zeros(nState)
    n = len(ll_idx)

    g_post = np.zeros(nState)
    g_lhood = 0.0
    g_lhood2 = 0.0
    weight_sum  = 0.0
    weight2_sum = 0.0

    for j in xrange(n):

        idx  = ll_idx[j]
        logp = ll_logp[j]
        post = ll_post[j]

        weight    = norm(loc=g_mu, scale=g_sig).pdf(idx)

        if weight < 1e-3: continue
        g_post   += post * weight
        g_lhood  += logp * weight
        weight_sum += weight
        weight2_sum += weight**2

    if abs(weight_sum)<1e-3: weight_sum=1e-3
    l_statePosterior   = g_post / weight_sum 
    l_likelihood_mean  = g_lhood / weight_sum 

    for j in xrange(n):

        idx  = ll_idx[j]
        logp = ll_logp[j]

        weight    = norm(loc=g_mu, scale=g_sig).pdf(idx)    
        if weight < 1e-3: continue
        g_lhood2 += weight * ((logp - l_likelihood_mean )**2)
        
    l_likelihood_std = np.sqrt(g_lhood2/(weight_sum - weight2_sum/weight_sum))

    return i, l_statePosterior, l_likelihood_mean, l_likelihood_std
    ## return i, l_statePosterior, l_likelihood_mean, np.sqrt(l_likelihood_mean2 - l_likelihood_mean**2)
