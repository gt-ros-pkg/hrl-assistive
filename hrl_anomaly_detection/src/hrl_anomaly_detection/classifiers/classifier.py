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
    def __init__(self, ml, method='svm', nPosteriors=10, nLength=200, ths_mult=None,\
                 class_weight={1.0: 1.0, -1.0: 5.1}, \
                 verbose=False):
        learning_base.__init__(self)
        
        self.ml = ml
        self.method = method
        self.verbose = verbose

        if self.method == 'svm':
            import svmutil as svm
            self.class_weight = class_weight
            ## from sklearn.svm import SVC
            ## self.class_weight = class_weight
            ## ## self.dt = svm.OneClassSVM(nu=0.1, kernel=custom_kernel)
            ## ## self.dt = SVC(nu=0.1, kernel='linear')
            ## ## self.dt = SVC(kernel='linear', gamma=0.0001, kernel=symmetric_entropy, verbose=True)
            ## self.dt = SVC(kernel='linear', gamma=0.01, verbose=True, \
            ##               class_weight=self.class_weight)
            ## ## self.dt = SVC(kernel=custom_kernel, gamma=0.01, verbose=True, \
            ## ##               class_weight=self.class_weight)
            ## #self.dt = SVC(kernel=custom_kernel2, verbose=True)
        elif self.method == 'cssvm_standard' or self.method == 'cssvm':
            sys.path.insert(0, '/home/dpark/git/cssvm/python')
            import cssvmutil as cssvm
            self.class_weight = class_weight
        elif self.method == 'progress_time_cluster':
            self.nLength   = nLength
            self.std_coff  = 1.0
            self.nPosteriors = nPosteriors
            self.ths_mult = ths_mult
            self.ll_mu  = np.zeros(nPosteriors)
            self.ll_std = np.zeros(nPosteriors) 
        elif self.method == 'fixed':
            self.mu  = 0.0
            self.std = 0.0
            self.ths_mult = ths_mult
            

    def fit(self, X, y, ll_idx=None):
        '''
        ll_idx is the index list of each sample in a sequence.
        '''

        # saved file check.

        if self.method == 'svm':
            import svmutil as svm
            if type(X) is not list: X=X.tolist()
            self.dt = svm.svm_train(y, X, '-c 1.1 -t 2 -w1 '+str(self.class_weight)+' -w-1 0.2' )
            ## self.dt.set_params(class_weight=self.class_weight)
            ## return self.dt.fit(X, y)
            return True
        elif self.method == 'cssvm_standard':
            import cssvmutil as cssvm
            if type(X) is not list: X=X.tolist()
            self.dt = cssvm.svm_train(y, X, '-C 0 -c 1.1 -t 2 -w1 '+str(self.class_weight)+' -w-1 0.2' )
            return True
        elif self.method == 'cssvm':
            import cssvmutil as cssvm
            ## self.dt = svm_train(y, X, '-c 4')
            if type(X) is not list: X=X.tolist()
            ## self.dt = cssvm.svm_train(y, X, '-s 1 -t 0 -c '+str(self.class_weight) )
            ## self.dt = cssvm.svm_train(y, X, '-c '+str(self.class_weight) )
            self.dt = cssvm.svm_train(y, X, '-C 1 -c 1.1 -t 2 -w1 '+str(self.class_weight)+' -w-1 0.01' )
            ## print "aaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            ## self.dt = cssvm.svm_train(y, X, '-C 1 -c 3' )
            return True
            
        elif self.method == 'progress_time_cluster':
            if type(X) == list: X = np.array(X)
            ll_logp = X[:,0:1]
            ll_post = X[:,1:]

            g_mu_list = np.linspace(0, self.nLength-1, self.nPosteriors)
            g_sig = float(self.nLength) / float(self.nPosteriors) * self.std_coff

            r = Parallel(n_jobs=-1)(delayed(learn_time_clustering)(i, ll_idx, ll_logp, ll_post, \
                                                                   g_mu_list[i],\
                                                                   g_sig, self.nPosteriors)
                                                                   for i in xrange(self.nPosteriors))
            _, self.l_statePosterior, self.ll_mu, self.ll_std = zip(*r)

            ## for i in xrange(self.nPosteriors):            
            ##     learn_time_clustering(i, ll_idx, ll_logp, ll_post, g_mu_list[i], g_sig, self.nPosteriors)

            return self.l_statePosterior, self.ll_mu, self.ll_std

        elif self.method == 'fixed':
            if type(X) == list: X = np.array(X)
            ll_logp = X[:,0:1]
            self.mu  = np.mean(ll_logp)
            self.std = np.std(ll_logp)
            return True
                

    def predict(self, X, y=None):
        '''
        return predicted values (not necessarily binaries)
        '''

        if self.method == 'svm':
            import svmutil as svm            
            if type(X) is not list: X=X.tolist()
            if y is not None:
                p_labels, _, p_vals = svm.svm_predict(y, X, self.dt)
            else:
                p_labels, _, p_vals = svm.svm_predict([0]*len(X), X, self.dt)
            return p_labels
            ## return self.dt.predict(X)
        elif self.method == 'cssvm_standard' or self.method == 'cssvm':
            import cssvmutil as cssvm            
            if type(X) is not list: X=X.tolist()
            if y is not None:
                p_labels, _, p_vals = cssvm.svm_predict(y, X, self.dt)
            else:
                p_labels, _, p_vals = cssvm.svm_predict([0]*len(X), X, self.dt)
            return p_labels
        elif self.method == 'progress_time_cluster':
            self.ml.cluster_type = 'time'
            
            logp = X[0]
            post = X[1:]

            # Find the best posterior distribution
            min_index, min_dist = findBestPosteriorDistribution(post, self.l_statePosterior)

            if (type(self.ths_mult) == list or type(self.ths_mult) == np.ndarray or \
                type(self.ths_mult) == tuple) and len(self.ths_mult)>1:
                err = (self.ll_mu[min_index] + self.ths_mult[min_index]*self.ll_std[min_index]) - logp
            else:
                err = (self.ll_mu[min_index] + self.ths_mult*self.ll_std[min_index]) - logp
            return err 
        elif self.method == 'fixed':
            logp = X[0]
            err = self.mu + self.ths_mult * self.std - logp
            return err

            

    def decision_function(self, X):

        if self.method == 'svm':
            if type(X) is not list:
                return self.predict(X.tolist())
            else:
                return self.predict(X)
            ## return self.dt.decision_function(X)
        elif self.method == 'cssvm_standard' or self.method == 'cssvm' or self.method == 'fixed':
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

    return min(entropy(p,q), entropy(q,p)) + 1e-6


def findBestPosteriorDistribution(post, l_statePosterior):
    # Find the best posterior distribution
    min_dist  = 100000000
    min_index = 0

    for j in xrange(len(l_statePosterior)):
        dist = entropy(post, l_statePosterior[j])
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
    prop_sum = 0.0

    for j in xrange(n):

        idx  = ll_idx[j]
        logp = ll_logp[j][0]
        post = ll_post[j]

        k_prop    = norm(loc=g_mu, scale=g_sig).pdf(idx)
        g_post   += post * k_prop
        g_lhood  += logp * k_prop
        g_lhood2 += logp * logp * k_prop
        prop_sum += k_prop

    l_statePosterior   = g_post / prop_sum 
    l_likelihood_mean  = g_lhood / prop_sum 
    l_likelihood_mean2 = g_lhood2 / prop_sum 

    return i, l_statePosterior, l_likelihood_mean, np.sqrt(l_likelihood_mean2 - l_likelihood_mean**2)
