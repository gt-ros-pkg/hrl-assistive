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

# visualization
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec

# util
import numpy as np
import scipy
import hrl_lib.util as ut
import random, copy

from scipy.stats import norm, entropy
from joblib import Parallel, delayed
from hrl_anomaly_detection.hmm.learning_base import learning_base
from sklearn import metrics
from sklearn.externals import joblib

from hrl_anomaly_detection import data_manager as dm
from hrl_anomaly_detection import util as util


class classifier(learning_base):
    def __init__(self, method='svm', nPosteriors=10, nLength=200, startIdx=4,\
                 ths_mult=-1.0,\
                 #progress time or state?
                 logp_offset = 0.0,\
                 class_weight=1.0, \
                 # svm
                 svm_type    = 0,\
                 kernel_type = 2,\
                 degree      = 3,\
                 gamma       = 0.3,\
                 nu          = 0.5,\
                 cost        = 4.,\
                 coef0       = 0.,\
                 w_negative  = 7.0,\
                 # bpsvm
                 bpsvm_w_negative = 7.0,\
                 bpsvm_cost       = 4.,\
                 bpsvm_gamma      = 0.3,\
                 # hmmsvm_diag
                 hmmsvm_diag_nu = 0.5,\
                 hmmsvm_diag_w_negative  = 7.0,\
                 hmmsvm_diag_cost        = 4.,\
                 hmmsvm_diag_gamma       = 0.3,\
                 # hmmsvm_dL
                 hmmsvm_dL_nu         = 0.5,\
                 hmmsvm_dL_w_negative = 7.0,\
                 hmmsvm_dL_cost       = 4.,\
                 hmmsvm_dL_gamma      = 0.3,\
                 # hmmsvm_LSLS
                 hmmsvm_LSLS_nu         = 0.5,\
                 hmmsvm_LSLS_w_negative = 7.0,\
                 hmmsvm_LSLS_cost       = 4.,\
                 hmmsvm_LSLS_gamma      = 0.3,\
                 # hmmsvm_dL
                 hmmsvm_no_dL_nu         = 0.5,\
                 hmmsvm_no_dL_w_negative = 7.0,\
                 hmmsvm_no_dL_cost       = 4.,\
                 hmmsvm_no_dL_gamma      = 0.3,\
                 # hmmosvm
                 hmmosvm_nu  = 0.00316,\
                 # osvm
                 osvm_nu     = 0.00316,\
                 # cssvm
                 cssvm_degree      = 3,\
                 cssvm_gamma       = 0.3,\
                 cssvm_cost        = 4.,\
                 cssvm_w_negative  = 7.0,\
                 # sgd
                 sgd_gamma      = 2.0,\
                 sgd_w_negative = 1.0,\
                 sgd_n_iter     = 10,\
                 # minibatchkmean
                 mbkmean_batch_size = 100,\
                 # progress svm
                 progress_svm_w_negative = 7.0,\
                 progress_svm_cost       = 4.,\
                 progress_svm_gamma      = 0.3,\
                 # hmmgp
                 theta0 = 1.0,\
                 nugget = 100.0,\
                 verbose=False):
        '''
        class_weight : positive class weight for svm
        nLength : only for progress-based classifier
        ths_mult: only for progress-based classifier
        '''              
        self.method = method
        self.nPosteriors = nPosteriors
        self.dt     = None
        self.nLength = nLength
        self.startIdx = startIdx
        self.verbose = verbose
                
        # constants to adjust thresholds
        self.class_weight = class_weight
        self.ths_mult = ths_mult

        if self.method.find('svm')>=0 and self.method is not 'cssvm':
            sys.path.insert(0, '/usr/lib/pymodules/python2.7')
            import svmutil as svm
            self.svm_type    = svm_type
            self.kernel_type = kernel_type
            self.degree      = degree 
            self.gamma       = gamma
            self.cost        = cost
            self.coef0       = coef0
            self.w_negative  = w_negative
            self.bpsvm_w_negative = bpsvm_w_negative
            self.bpsvm_cost       = bpsvm_cost
            self.bpsvm_gamma      = bpsvm_gamma                        
            self.hmmsvm_diag_nu         = hmmsvm_diag_nu
            self.hmmsvm_diag_w_negative = hmmsvm_diag_w_negative
            self.hmmsvm_diag_cost       = hmmsvm_diag_cost
            self.hmmsvm_diag_gamma      = hmmsvm_diag_gamma
            self.hmmsvm_dL_nu         = hmmsvm_dL_nu
            self.hmmsvm_dL_w_negative = hmmsvm_dL_w_negative
            self.hmmsvm_dL_cost       = hmmsvm_dL_cost
            self.hmmsvm_dL_gamma      = hmmsvm_dL_gamma
            self.hmmsvm_LSLS_nu       = hmmsvm_LSLS_nu
            self.hmmsvm_LSLS_w_negative = hmmsvm_LSLS_w_negative
            self.hmmsvm_LSLS_cost       = hmmsvm_LSLS_cost
            self.hmmsvm_LSLS_gamma      = hmmsvm_LSLS_gamma                        
            self.hmmsvm_no_dL_nu         = hmmsvm_no_dL_nu
            self.hmmsvm_no_dL_w_negative = hmmsvm_no_dL_w_negative
            self.hmmsvm_no_dL_cost       = hmmsvm_no_dL_cost
            self.hmmsvm_no_dL_gamma      = hmmsvm_no_dL_gamma
            self.hmmosvm_nu  = hmmosvm_nu
            self.osvm_nu     = osvm_nu
            self.nu          = nu
            self.progress_svm_w_negative = progress_svm_w_negative
            self.progress_svm_cost       = progress_svm_cost
            self.progress_svm_gamma      = progress_svm_gamma
        elif self.method == 'cssvm':
            sys.path.insert(0, os.path.expanduser('~')+'/git/cssvm/python')
            import cssvmutil as cssvm
            self.svm_type    = svm_type
            self.kernel_type = kernel_type
            self.cssvm_degree     = cssvm_degree 
            self.cssvm_gamma      = cssvm_gamma 
            self.cssvm_cost       = cssvm_cost 
            self.cssvm_w_negative = cssvm_w_negative 
        elif self.method == 'progress' or self.method == 'progress_state' or self.method == 'progress_diag':
            self.nLength   = nLength
            self.std_coff  = 1.0
            self.logp_offset = logp_offset
            self.ll_mu  = np.zeros(nPosteriors)
            self.ll_std = np.zeros(nPosteriors)
            self.l_statePosterior = None
        elif self.method == 'fixed':
            self.mu  = 0.0
            self.std = 0.0
        elif self.method == 'change':
            self.nLength   = nLength
            self.mu  = 0.0
            self.std = 0.0
        elif self.method == 'sgd':
            self.sgd_w_negative = sgd_w_negative             
            self.sgd_gamma      = sgd_gamma
            self.sgd_n_iter     = sgd_n_iter 
            ## self.cost         = cost
        elif self.method == 'mbkmean' or self.method == 'kmean' or self.method == 'state_kmean':
            self.mbkmean_batch_size = mbkmean_batch_size
            self.ll_mu  = np.zeros(nPosteriors)
            self.ll_std = np.zeros(nPosteriors)
        elif self.method == 'hmmgp':
            from sklearn import gaussian_process
            self.regr = 'linear' #'linear' # 'constant', 'linear', 'quadratic'
            self.corr = 'squared_exponential' #'squared_exponential' #'absolute_exponential', 'squared_exponential','generalized_exponential', 'cubic', 'linear'
            self.nugget = nugget
            self.theta0 = theta0

            self.dt = gaussian_process.GaussianProcess(regr=self.regr, theta0=self.theta0, corr=self.corr, \
                                                       normalize=True, nugget=self.nugget)            
        elif self.method == 'hmmsvr':
            self.svm_type    = svm_type
            self.kernel_type = kernel_type
            self.degree      = degree 
            self.gamma       = gamma
            self.cost        = cost
            self.coef0       = coef0
            self.nu          = nu
                        
        learning_base.__init__(self)



    def fit(self, X, y, ll_idx=None, parallel=True, warm_start=False):
        '''
        ll_idx is the index list of each sample in a sequence.
        '''
        # get custom precomputed kernel for svms
        ## if 'svm' in self.method:
        ##     self.X_train=X
        ##     ## y_train=y
        ##     K_train = custom_kernel(self.X_train, self.X_train, gamma=self.gamma)

        if self.method.find('svm')>=0 and self.method is not 'cssvm':
            sys.path.insert(0, '/usr/lib/pymodules/python2.7')
            import svmutil as svm

            if type(X) is not list: X=X.tolist()
            if type(y) is not list: y=y.tolist()
            commands = '-q -s '+str(self.svm_type)+' -t '+str(self.kernel_type)+' -d '+str(self.degree)\
              +' -w1 '+str(self.class_weight) +' -r '+str(self.coef0)

            if self.method == 'osvm':
                commands = commands+' -n '+str(self.osvm_nu)+' -g '+str(self.gamma)\
                  +' -w-1 '+str(self.w_negative)+' -c '+str(self.cost)
            elif self.method == 'hmmosvm':
                commands = commands+' -n '+str(self.hmmosvm_nu)+' -g '+str(self.gamma)\
                  +' -c '+str(self.cost)
            elif self.method == 'hmmsvm_diag':
                commands = commands+' -n '+str(self.hmmsvm_diag_nu)+' -g '+str(self.hmmsvm_diag_gamma)\
                  +' -w-1 '+str(self.hmmsvm_diag_w_negative)+' -c '+str(self.hmmsvm_diag_cost)
            elif self.method == 'hmmsvm_dL':
                commands = commands+' -n '+str(self.hmmsvm_dL_nu)+' -g '+str(self.hmmsvm_dL_gamma)\
                  +' -w-1 '+str(self.hmmsvm_dL_w_negative)+' -c '+str(self.hmmsvm_dL_cost)
            elif self.method == 'hmmsvm_LSLS':
                commands = commands+' -n '+str(self.hmmsvm_LSLS_nu)+' -g '+str(self.hmmsvm_LSLS_gamma)\
                  +' -w-1 '+str(self.hmmsvm_LSLS_w_negative)+' -c '+str(self.hmmsvm_LSLS_cost)
            elif self.method == 'hmmsvm_no_dL':
                commands = commands+' -n '+str(self.hmmsvm_no_dL_nu)+' -g '+str(self.hmmsvm_no_dL_gamma)\
                  +' -w-1 '+str(self.hmmsvm_no_dL_w_negative)+' -c '+str(self.hmmsvm_no_dL_cost)
            elif self.method == 'bpsvm':
                commands = commands+' -n '+str(self.nu)+' -g '+str(self.bpsvm_gamma)\
                  +' -w-1 '+str(self.bpsvm_w_negative)+' -c '+str(self.bpsvm_cost)
            elif self.method == 'progress_osvm':
                commands = commands+' -n '+str(self.hmmosvm_nu)+' -g '+str(self.gamma)\
                  +' -w-1 '+str(self.w_negative)+' -c '+str(self.cost)
            elif self.method == 'progress_svm':
                commands = commands+' -n '+str(self.nu)+' -g '+str(self.progress_svm_gamma)\
                  +' -w-1 '+str(self.progress_svm_w_negative)+' -c '+str(self.progress_svm_cost)
            elif self.method == 'svm_fixed':
                commands = commands+' -n '+str(self.nu)+' -g '+str(self.gamma)\
                  +' -w-1 '+str(self.w_negative)+' -c '+str(self.cost)
            else:
                commands = commands+' -n '+str(self.nu)+' -g '+str(self.gamma)\
                  +' -w-1 '+str(self.w_negative)+' -c '+str(self.cost)

            try: self.dt = svm.svm_train(y, X, commands )
            except:
                print "svm training failure"
                print np.shape(y), np.shape(X)
                print commands                
                return False


            if self.method == 'svm_fixed':
                if type(X) == list: X = np.array(X)
                ll_logp = X[:,0:1]
                self.mu  = np.mean(ll_logp)
                self.std = np.std(ll_logp)
                
            return True

        elif self.method.find('svr')>=0:
            sys.path.insert(0, '/usr/lib/pymodules/python2.7')
            import svmutil as svm
            commands = '-q -s 4 -t 0'+' -n '+str(self.nu)+' -g 0.04' \
              +' -c '+str(self.cost) + ' -d 3' 
              #+str(self.gamma)\
            if type(X) == list: X = np.array(X)
            
            # extract only negatives
            ll_logp = [ [X[i,0]] for i in xrange(len(X)) if y[i]<0 ]
            ll_post = [ X[i,-self.nPosteriors:] for i in xrange(len(X)) if y[i]<0 ]


            try: self.dt = svm.svm_train(ll_logp, ll_post, commands )
            except:
                print self.dt
                print "svm training failure", np.shape(ll_logp), np.shape(ll_post)
                print commands                
                return False
            return True
              
        
        elif self.method == 'cssvm':
            sys.path.insert(0, os.path.expanduser('~')+'/git/cssvm/python')
            import cssvmutil as cssvm
            if type(X) is not list: X=X.tolist()
            commands = '-q -C 1 -s '+str(self.svm_type)+' -t '+str(self.kernel_type)\
              +' -d '+str(self.cssvm_degree)\
              +' -g '+str(self.cssvm_gamma)\
              +' -c '+str(self.cssvm_cost)+' -w1 '+str(self.class_weight)\
              +' -w-1 '+str(self.cssvm_w_negative) \
              +' -m 200'
            try: self.dt = cssvm.svm_train(y, X, commands )
            except: return False
            return True
            
        elif self.method == 'progress' or self.method == 'progress_diag':
            if type(X) == list: X = np.array(X)
            if ll_idx is None:
                # Need to artificially generate ll_idx....
                nSample  = len(y)/(self.nLength-self.startIdx)
                idx_list = range(self.nLength)[self.startIdx:]
                ll_idx = [ idx_list[j] for j in xrange(len(idx_list)) for i in xrange(nSample) if y[i*(self.nLength-self.startIdx)+1]<0 ]
                ## ll_logp = [ X[i,0] for i in xrange(len(X)) if y[i]<0 ]
                ## print np.shape(ll_logp), np.shape(ll_idx), np.shape(y), len(y)/(self.nLength-self.startIdx)
                ## sys.exit()
                ## print "Error>> ll_idx is not inserted"
                ## sys.exit()
            else: ll_idx  = [ ll_idx[i] for i in xrange(len(ll_idx)) if y[i]<0 ]
            ll_logp = [ X[i,0] for i in xrange(len(X)) if y[i]<0 ]
            ll_post = [ X[i,-self.nPosteriors:] for i in xrange(len(X)) if y[i]<0 ]

            self.g_mu_list = np.linspace(0, self.nLength-1, self.nPosteriors)
            self.g_sig = float(self.nLength) / float(self.nPosteriors) * self.std_coff

            if parallel:
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

        elif self.method == 'progress_state':
            '''
            state-based clustering using knn
            '''
            if type(X) == list: X = np.array(X)

            # extract only negatives
            ll_logp = [ X[i,0] for i in xrange(len(X)) if y[i]<0 ]
            ll_post = [ X[i,-self.nPosteriors:] for i in xrange(len(X)) if y[i]<0 ]
            
            ## init_center = np.eye(self.nPosteriors, self.nPosteriors)
            ## self.dt     = KMeans(self.nPosteriors, init=init_center)
            ## idx_list    = self.km.fit_predict(state_mat.transpose())
            self.progress_neighbors = 10
            
            from sklearn.neighbors import NearestNeighbors
            from hrl_anomaly_detection.hmm.learning_util import symmetric_entropy
            self.dt = NearestNeighbors(n_neighbors=self.progress_neighbors, metric=symmetric_entropy)
            self.dt.fit(ll_post)
            
            self.ll_logp = np.array(ll_logp)
            
            return True

        elif self.method == 'hmmgp':
            '''
            gaussian process
            '''
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

        elif self.method == 'fixed':
            if type(X) == list: X = np.array(X)
            ll_logp = X[:,0:1]
            self.mu  = np.mean(ll_logp)
            self.std = np.std(ll_logp)
            return True

        elif self.method == 'change':
            if type(X) == list: X = np.array(X)

            if len(np.shape(y))>1:
                l_idx = [ i for i in range(len(X)) if y[i][0]<0 ]
            else:
                l_idx = [ i for i in range(len(X)) if y[i]<0 ]

            X_logp    = X[l_idx,:,0:1]
            X_logp_d = X_logp[:,1:,0]-X_logp[:,:-1,0]

            self.mu  = np.mean(X_logp_d)
            self.std = np.std(X_logp_d)

            return True
                
        elif self.method == 'sgd':

            max_components = 1000 #196
            if len(X) < max_components:
                n_components =len(X)
            else:
                n_components = max_components
                
            ## from sklearn.kernel_approximation import RBFSampler
            ## self.rbf_feature = RBFSampler(gamma=self.gamma, n_components=1000, random_state=1)
            from sklearn.kernel_approximation import Nystroem
            self.rbf_feature = Nystroem(gamma=self.sgd_gamma, n_components=n_components, random_state=1)
                
            from sklearn.linear_model import SGDClassifier
            # get time-based clustering center? Not yet implemented
            X_features       = self.rbf_feature.fit_transform(X)
            if self.verbose: print "sgd classifier: ", np.shape(X), np.shape(X_features)
            # fitting
            print "Class weight: ", self.class_weight, self.sgd_w_negative
            d = {+1: self.class_weight, -1: self.sgd_w_negative}
            if warm_start and self.dt is not None:
                self.dt.set_params(class_weight=d)
                self.dt.set_params(warm_start=True)
            else:
                self.dt = SGDClassifier(verbose=0,class_weight=d,n_iter=self.sgd_n_iter, #learning_rate='constant',\
                                        eta0=1e-2, shuffle=True, average=True, fit_intercept=True)
            self.dt.fit(X_features, y)

        elif self.method == 'mbkmean':
            from sklearn.cluster import MiniBatchKMeans
            init_list = []
            for i in xrange(self.nPosteriors):
                init_array = np.zeros(self.nPosteriors)
                init_array[i] = 1.0
                init_list.append(init_array)
                
            if type(X) == list: X = np.array(X)
            posts = X[:,-self.nPosteriors:]
            logps = X[:,0]
                
            self.dt = MiniBatchKMeans(n_clusters=self.nPosteriors, \
                                      batch_size=self.mbkmean_batch_size,\
                                      init=np.array(init_list))
            labels = self.dt.fit_predict(posts)
            # clustering likelihoods
            ll_logp = [[] for i in xrange(self.nPosteriors)]
            for i, label in enumerate(labels):
                ll_logp[label].append(logps[i])
            self.ll_nData = [len(ll_logp[i]) for i in xrange(self.nPosteriors)]
            for i in xrange(self.nPosteriors):
                self.ll_mu[i]  = np.mean(ll_logp[i])
                self.ll_std[i] = np.std(ll_logp[i])
            
        elif self.method == 'kmean':
            from sklearn.cluster import KMeans
            init_list = []
            for i in xrange(self.nPosteriors):
                init_array = np.zeros(self.nPosteriors)
                init_array[i] = 1.0
                init_list.append(init_array)
                
            if type(X) == list: X = np.array(X)
            posts = X[:,-self.nPosteriors:]
            logps = X[:,0]
                
            self.dt = KMeans(n_clusters=self.nPosteriors, \
                             init=np.array(init_list), n_init=1)
            labels = self.dt.fit_predict(posts)
            # clustering likelihoods
            ll_logp = [[] for i in xrange(self.nPosteriors)]
            for i, label in enumerate(labels):
                ll_logp[label].append(logps[i])
            self.ll_nData = [len(ll_logp[i]) for i in xrange(self.nPosteriors)]
            for i in xrange(self.nPosteriors):
                self.ll_mu[i]  = np.mean(ll_logp[i])
                self.ll_std[i] = np.std(ll_logp[i])

        elif self.method == 'state_kmean':
            init_list = []
            for i in xrange(self.nPosteriors):
                init_list.append( float(i) )
                
            if type(X) == list: X = np.array(X)
            states = np.argmax(X[:,-self.nPosteriors:], axis=1)
            logps = X[:,0]
                
            ll_logp = [[] for i in xrange(self.nPosteriors)]
            for i, state in enumerate(states):
                ll_logp[state].append(logps[i])
            self.ll_nData = [len(ll_logp[i]) for i in xrange(self.nPosteriors)]
            for i in xrange(self.nPosteriors):
                self.ll_mu[i]  = np.mean(ll_logp[i])
                self.ll_std[i] = np.std(ll_logp[i])




    def partial_fit(self, X, y=None, classes=None, sample_weight=None, n_iter=1, shuffle=True):
        '''
        X: samples x hmm-feature vec
        y: sample
        '''

        if shuffle is True:
            idx_list = range(len(X))
            random.shuffle(idx_list)
            X = [X[ii] for ii in idx_list]
            y = [y[ii] for ii in idx_list]
            if sample_weight is not None:
                sample_weight = [sample_weight[ii] for ii in idx_list]

        if self.method == 'sgd':
            ## if sample_weight is None: sample_weight = [self.class_weight]*len(X)
            d = {+1: self.class_weight, -1: self.sgd_w_negative}
            self.dt.set_params(class_weight=d)
            
            X_features = self.rbf_feature.transform(X)
            for i in xrange(n_iter):
                self.dt.partial_fit(X_features,y, classes=classes, sample_weight=sample_weight)
        elif self.method == 'mbkmean':
            if type(X) == list: X = np.array(X)
            posts = X[:,-self.nPosteriors:]
            logps = X[:,0]
            labels = self.dt.predict(posts)
            # clustering likelihoods
            ll_logp = [[] for i in xrange(self.nPosteriors)]
            for i, label in enumerate(labels):
                ll_logp[label].append(logps[i])
            for i in xrange(self.nPosteriors):
                n = float(len(ll_logp[i]))
                N = float(self.ll_nData[i])
                last_mu  = self.ll_mu[i]
                last_std = self.ll_std[i] 
                self.ll_mu[i] = ((N)*last_mu + n*np.sum(ll_logp[i])) / N+n
                self.ll_std[i]= np.sqrt( ((N)*( last_std*last_std + last_mu*last_mu)+\
                                          n*self.ll_mu[i]*self.ll_mu[i])/\
                                          (N+n) - self.ll_mu[i]*self.ll_mu[i] )
                
            
        else:
            print "Not available method, ", self.method
            sys.exit()


    def predict(self, X, y=None, temp=True):
        '''
        X is single sample
        return predicted values (not necessarily binaries)
        '''
        
        if self.method.find('svm')>=0:
            
            if self.method.find('cssvm')>=0:
                sys.path.insert(0, os.path.expanduser('~')+'/git/cssvm/python')
                import cssvmutil as svm
            else:
                sys.path.insert(0, '/usr/lib/pymodules/python2.7')
                import svmutil as svm

            if self.verbose:
                print svm.__file__

            if type(X) is not list: X=X.tolist()
            if y is not None:
                p_labels, _, p_vals = svm.svm_predict(y, X, self.dt)
            else:
                p_labels, _, p_vals = svm.svm_predict([0]*len(X), X, self.dt)

            if self.method.find('fixed')>=0:
                if len(np.shape(X))==1: X = [X]
                self.ths_mult = -3

                for i in xrange(len(X)):
                    logp = X[i][0]
                    err = self.mu + self.ths_mult * self.std - logp
                    if p_labels[i] > 0: continue
                    else:
                        if err>0: p_labels[i] = 1.0
            return p_labels
        
        elif self.method == 'progress' or self.method == 'progress_diag':
            if len(np.shape(X))==1: X = [X]

            l_err = []
            for i in xrange(len(X)):
                logp = X[i][0]
                post = X[i][-self.nPosteriors:]

                # Find the best posterior distribution
                try:
                    min_index, min_dist = findBestPosteriorDistribution(post, self.l_statePosterior)
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
            return l_err

        elif self.method == 'progress_state':
            if len(np.shape(X))==1: X = [X]

            l_err = []
            for i in xrange(len(X)):
                logp = X[i][0]
                post = X[i][-self.nPosteriors:]

                _, l_idx = self.dt.kneighbors(post)

                l_logp= self.ll_logp[l_idx[0]]
                err = np.mean(l_logp) + self.ths_mult*np.std(l_logp) - logp - self.logp_offset
                l_err.append(err)

            return l_err            

        elif self.method == 'hmmgp':
            '''
            gaussian process
            '''
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
                    for i, post in enumerate(posts):                        
                        print i, post
                    sys.exit()

            l_err = y_pred + self.ths_mult*sigma - logps #- self.logp_offset
            return l_err

        elif self.method == 'fixed':
            if len(np.shape(X))==1: X = [X]
                
            l_err = []
            for i in xrange(len(X)):
                logp = X[i][0]
                err = self.mu + self.ths_mult * self.std - logp
                l_err.append(err)
            return l_err

        elif self.method == 'change':
            if len(np.shape(X))==1: X = [X]
                
            l_err = []
            for i in xrange(len(X)):
                if i==0: logp_d = 0.0
                else:    logp_d = X[i][0]-X[i-1][0]
                err = self.mu + self.ths_mult * self.std - logp_d
                l_err.append(err)
            return l_err

        elif self.method == 'sgd':
            X_features = self.rbf_feature.transform(X)
            return self.dt.predict(X_features)

        elif self.method == 'mbkmean' or self.method == 'kmean':
            if type(X) == list: X = np.array(X)
            posts  = X[:,-self.nPosteriors:]            
            labels = self.dt.predict(posts)

            l_err = []
            for i in xrange(len(X)):
                logp = X[i][0]                
                err = self.ll_mu[labels[i]]+self.ths_mult*self.ll_std[labels[i]] - logp
                l_err.append(err)

            return l_err

        elif self.method == 'state_kmean':
            if type(X) == list: X = np.array(X)
            states = np.argmax(X[:,-self.nPosteriors:], axis=1)

            l_err = []
            for i in xrange(len(X)):
                logp = X[i][0]                
                err = self.ll_mu[states[i]]+self.ths_mult*self.ll_std[states[i]] - logp
                l_err.append(err)

            return l_err

        elif self.method == 'hmmsvr':
            
            sys.path.insert(0, '/usr/lib/pymodules/python2.7')
            import svmutil as svm

            if len(np.shape(X))==1: X = [X]
            if type(X) is list: X= np.array(X)
            
            logps = X[:,0]
            posts = X[:,-self.nPosteriors:].tolist()
                
            p_labels, (ACC, MSE, SCC), p_vals = svm.svm_predict(logps, posts, self.dt, options='-q')
            sigma = np.sqrt(MSE)

            l_err = p_vals + self.ths_mult*sigma - logps #- self.logp_offset
            return l_err

        elif self.method == 'rnd':
            if len(np.shape(X))==1: X = [X]

            l_err = np.random.choice([-1, 1], size=len(X), p=[self.class_weight, 1.0-self.class_weight])
            return l_err


    def decision_function(self, X):

        ## return self.dt.decision_function(X)
        if self.method.find('svm')>=0  or self.method == 'fixed':
            if type(X) is not list:
                return self.predict(X.tolist())
            else:
                return self.predict(X)
        elif self.method.find('sgd')>=0:
            X_features = self.rbf_feature.transform(X)
            return self.dt.decision_function(X_features)
        else:
            print "Not implemented"
            sys.exit()

        return 
        
    def score(self, X, y):
        if self.method.find('svm')>=0 and self.method is not 'cssvm':
            return self.dt.score(X,y)
        else:
            print "Not implemented funciton Score"
            return 

        
    def save_model(self, fileName):
        if self.dt is None: 
            print "No trained classifier"
            return
        
        if self.method.find('svm')>=0 and self.method is not 'cssvm':       
            sys.path.insert(0, '/usr/lib/pymodules/python2.7')
            import svmutil as svm            
            svm.svm_save_model(fileName, self.dt)
        elif self.method.find('sgd')>=0:            
            import pickle
            with open(fileName, 'wb') as f:
                pickle.dump(self.dt, f)
                pickle.dump(self.rbf_feature, f)
            ## joblib.dump(self.dt, fileName)
        elif self.method.find('progress')>=0 :
            d = {'g_mu_list': self.g_mu_list, 'g_sig': self.g_sig, \
                 'l_statePosterior': self.l_statePosterior,\
                 'll_mu': self.ll_mu, 'll_std': self.ll_std}
            ut.save_pickle(d, fileName)            
        elif self.method.find('mbkmean')>=0 or self.method.find('kmean')>=0:
            ## d = {'ll_mu': self.ll_mu, 'll_std': self.ll_std}
            ## ut.save_pickle(d, fileName)            
            ## import pickle
            ## with open(fileName, 'wb') as f:
            ##     pickle.dump(self.dt, f)
            print "Not able to save mbkmean or kmean"
        elif self.method.find('hmmgp')>=0:            
            import pickle
            with open(fileName, 'wb') as f:
                pickle.dump(self.dt, f)
            
        else:
            print "Not available method"

            
    def load_model(self, fileName):        
        if self.method.find('svm')>=0 and self.method is not 'cssvm':
            sys.path.insert(0, '/usr/lib/pymodules/python2.7')
            import svmutil as svm            
            self.dt = svm.svm_load_model(fileName) 
        elif self.method.find('sgd')>=0:
            import pickle
            with open(fileName, 'rb') as f:
                self.dt = pickle.load(f)
                self.rbf_feature = pickle.load(f)
            ## self.dt = joblib.load(fileName)
        elif self.method.find('progress')>=0:
            print "Start to load a progress based classifier"
            d = ut.load_pickle(fileName)
            self.g_mu_list = d['g_mu_list']
            self.g_sig     = d['g_sig']
            self.l_statePosterior = d['l_statePosterior']
            self.ll_mu            = d['ll_mu']
            self.ll_std           = d['ll_std']
        elif self.method.find('hmmgp')>=0:
            import pickle
            with open(fileName, 'rb') as f:
                self.dt = pickle.load(f)
        else:
            print "Not available method"
        
            

        
####################################################################
# functions for distances
####################################################################

def custom_kernel(x1,x2, gamma=1.0):
    '''
    Similarity estimation between (loglikelihood, state distribution) feature vector.
    kernel must take as arguments two matrices of shape (n_samples_1, n_features), (n_samples_2, n_features)
    and return a kernel matrix of shape (n_samples_1, n_samples_2)
    '''

    if len(np.shape(x1)) == 2: 

        kernel_mat       = np.zeros((len(x1), len(x2)+1))
        kernel_mat[:,:1] = np.arange(len(x1))[:,np.newaxis]+1
        kernel_mat[:,1:] = metrics.pairwise.pairwise_distances(x1[:,0],x2[:,0], metric='l1') 
        kernel_mat[:,1:] += gamma*metrics.pairwise.euclidean_distances(x1[:,1:],x2[:,1:])*\
          (metrics.pairwise.pairwise_distances(np.argmax(x1[:,1:],axis=1),\
                                               np.argmax(x2[:,1:],axis=1),
                                               metric='l1') + 1.0)
        return np.exp(-kernel_mat)
    else:
        d1 = abs(x1[0] - x2[0]) 
        d2 = np.linalg.norm(x1[1:]-x2[1:])*( abs(np.argmax(x1[1:])-np.argmax(x2[1:])) + 1.0)
        return np.exp(-(d1 + gamma*d2))

def customDist(i,j, x1, x2, gamma):
    return i,j,(x1-x2)**2 + gamma*1.0/symmetric_entropy(x1,x2)


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

def KLS(p,q, gamma=3.0):
    return np.exp(-gamma*( entropy(p,np.array(q)+1e-6) + entropy(q,np.array(p)+1e-6) ) )

def symmetric_entropy(p,q):
    '''
    Return the sum of KL divergences
    '''
    pp = np.array(p)+1e-6
    qq = np.array(q)+1e-6
    
    ## return min(entropy(p,np.array(q)+1e-6), entropy(q,np.array(p)+1e-6))
    return min(entropy(pp,qq), entropy(qq,pp))


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

        weight = norm(loc=g_mu, scale=g_sig).pdf(idx)

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

## def learn_state_clustering(i, ll_logp, ll_post, g_mu, g_sig, nState):
##     return i, 


def update_time_cluster(i, ll_idx, ll_logp, ll_post, rbf_mu, rbf_sig, mu, sig, nState, N, \
                        update_weight=1.0):

    g_lhood = 0.0
    weight_sum  = 0.0

    n = len(ll_idx)
    for j in xrange(n): # per execution

        idx  = ll_idx[j]
        logp = ll_logp[j]
        post = ll_post[j]

        weight    = norm(loc=rbf_mu, scale=rbf_sig).pdf(idx)

        ## if weight < 1e-3: continue
        g_lhood  = np.sum(logp * weight)
        weight_sum = np.sum(weight)
        if abs(weight_sum)<1e-3: weight_sum=1e-3        

        x_new   = g_lhood / weight_sum
        mu_new  = ( float(N-update_weight)*mu + update_weight*x_new )/(N)
        try:
            sig_new = np.sqrt( (float(N-update_weight)*( sig*sig + mu*mu)+update_weight*mu_new*mu_new)/float(N) \
                               - mu_new*mu_new )
        except:
            print (float(N-update_weight)*( sig*sig + mu*mu)+update_weight*mu_new*mu_new)/float(N) - mu_new*mu_new
            print (float(N-update_weight)*( sig*sig + mu*mu)+update_weight*mu_new*mu_new)/float(N), mu_new*mu_new


        mu  = mu_new
        sig = sig_new

    return i, mu, sig


def run_classifier(j, X_train, Y_train, idx_train, X_test, Y_test, idx_test, \
                   method, nState, nLength, nPoints, param_dict, ROC_dict, dtc=None):

    # classifier # TODO: need to make it efficient!!
    if dtc is None:
        dtc = classifier( method=method, nPosteriors=nState, nLength=nLength )        
    dtc.set_params( **param_dict )
    if method == 'svm' or method == 'hmmsvm_diag' or method == 'hmmsvm_dL' or method == 'hmmsvm_LSLS' or\
      method == 'hmmsvm_no_dL' or method == 'sgd' or method == 'progress_svm' or method == 'svm_fixed':
        if method == 'svm_fixed': 
            weights = ROC_dict['svm_param_range']
        else:
            weights = ROC_dict[method+'_param_range']
            
        dtc.set_params( class_weight=weights[j] )
        ret = dtc.fit(X_train, Y_train, parallel=False)
    elif method == 'bpsvm':
        weights = ROC_dict[method+'_param_range']
        ## dtc.set_params( kernel_type=0 )
        dtc.set_params( class_weight=weights[j] )
        ret = dtc.fit(X_train, Y_train, parallel=False)
    elif method == 'hmmosvm' or method == 'osvm' or method == 'progress_osvm':
        weights = ROC_dict[method+'_param_range']
        dtc.set_params( svm_type=2 )
        dtc.set_params( gamma=weights[j] )
        ret = dtc.fit(X_train, np.array(Y_train)*-1.0, parallel=False)
    elif method == 'cssvm':
        weights = ROC_dict[method+'_param_range']
        dtc.set_params( class_weight=weights[j] )
        ret = dtc.fit(X_train, np.array(Y_train)*-1.0, idx_train, parallel=False)                
    elif method == 'progress' or method == 'progress_diag' or method == 'progress_state' or method == 'fixed' \
      or method == 'kmean' or method == 'hmmgp':
        thresholds = ROC_dict[method+'_param_range']
        dtc.set_params( ths_mult = thresholds[j] )
        if j==0: ret = dtc.fit(X_train, Y_train, idx_train, parallel=False)                
    elif method == 'rfc':
        weights = ROC_dict[method+'_param_range']
        dtc.set_params( svm_type=2 )
        ret = dtc.fit(X_train, np.array(Y_train)*-1.0, parallel=False)
    else:
        print "Not available method: ", method
        return "Not available method", -1

    if ret is False:
        print "fit failed, ", weights[j]
        sys.exit()
        return 'fit failed', [],[],[],[],[]

    # evaluate the classifier
    tp_l = []
    fp_l = []
    tn_l = []
    fn_l = []
    delay_l = []
    delay_idx = 0
    for ii in xrange(len(X_test)):
        if len(Y_test[ii])==0: continue

        if method.find('osvm')>=0 or method == 'cssvm':
            est_y = dtc.predict(X_test[ii], y=np.array(Y_test[ii])*-1.0)
            est_y = np.array(est_y)* -1.0
        else:
            est_y = dtc.predict(X_test[ii], y=Y_test[ii])
        
        anomaly = False
        for jj in xrange(len(est_y)):
            if est_y[jj] > 0.0:

                ## if method == 'hmmosvm':
                ##     window_size = 5
                ##     if jj >= window_size:
                ##         if np.sum(est_y[jj-window_size:jj])>=window_size:
                ##             anomaly = True                            
                ##             break
                ##     continue                        

                ## if Y_test[ii][0] < 0:
                ## print jj, est_y[jj], Y_test[ii][0] #, " - ", X_test[ii][jj]
                    
                if idx_test is not None:
                    try:
                        delay_idx = idx_test[ii][jj]
                    except:
                        print "Error!!!!!!!!!!!!!!!!!!"
                        print np.shape(idx_test), ii, jj
                anomaly = True                            
                break        

        if Y_test[ii][0] > 0.0:
            if anomaly:
                tp_l.append(1)
                delay_l.append(delay_idx)
            else: fn_l.append(1)
        elif Y_test[ii][0] <= 0.0:
            if anomaly: fp_l.append(1)
            else: tn_l.append(1)

    return j, tp_l, fp_l, fn_l, tn_l, delay_l



def run_classifiers(idx, processed_data_path, task_name, method,\
                    ROC_data, ROC_dict, AE_dict, SVM_dict, HMM_dict,\
                    raw_data=None, startIdx=4, nState=25, \
                    modeling_pkl_prefix=None, failsafe=False, delay_estimation=False):

    #-----------------------------------------------------------------------------------------
    nPoints    = ROC_dict['nPoints']
    add_logp_d = HMM_dict.get('add_logp_d', False)

    # pass method if there is existing result
    data = {}
    data = util.reset_roc_data(data, [method], [], nPoints)
    if ROC_data[method]['complete'] == True: return data
    #-----------------------------------------------------------------------------------------

    ## print idx, " : training classifier and evaluate testing data"
    # train a classifier and evaluate it using test data.
    from sklearn import preprocessing

    if method == 'osvm' or method == 'bpsvm':
        if method == 'osvm': raw_data_idx = 0
        elif method == 'bpsvm': raw_data_idx = 1

        ## if modeling_pkl_prefix is not None and delay_estimation is False:
        ##     idx = int(modeling_pkl_prefix.split('_')[-1])
            
        X_train_org   = raw_data[raw_data_idx][idx]['X_scaled']
        Y_train_org   = raw_data[raw_data_idx][idx]['Y_train_org']
        idx_train_org = raw_data[raw_data_idx][idx]['idx_train_org']
        ll_classifier_test_X    = raw_data[raw_data_idx][idx]['X_test']
        ll_classifier_test_Y    = raw_data[raw_data_idx][idx]['Y_test']
        ll_classifier_test_idx  = raw_data[raw_data_idx][idx]['idx_test']
        ll_classifier_test_labels = None
        step_idx_l = raw_data[raw_data_idx][idx]['step_idx_l']

        # TODO: set automatically!
        nLength = 200
    else:

        if modeling_pkl_prefix is not None:
            modeling_pkl = os.path.join(processed_data_path, modeling_pkl_prefix+'_'+str(idx)+'.pkl')            
        else:        
            modeling_pkl = os.path.join(processed_data_path, 'hmm_'+task_name+'_'+str(idx)+'.pkl')

        print "start to load hmm data, ", modeling_pkl
        d            = ut.load_pickle(modeling_pkl)
        for k, v in d.iteritems():
            exec '%s = v' % k        
        ## nState, ll_classifier_train_?, ll_classifier_test_?, nLength    
        ll_classifier_test_labels = d.get('ll_classifier_test_labels', None)
    
        if 'diag' in method:
            ll_classifier_train_X   = ll_classifier_diag_train_X
            ll_classifier_train_Y   = ll_classifier_diag_train_Y
            ll_classifier_train_idx = ll_classifier_diag_train_idx
            ll_classifier_test_X    = ll_classifier_diag_test_X
            ll_classifier_test_Y    = ll_classifier_diag_test_Y
            ll_classifier_test_idx  = ll_classifier_diag_test_idx
        elif method =='progress_osvm' or method == 'progress_svm':
            ## # temp
            from hrl_anomaly_detection.hmm import learning_hmm as hmm
            ll_classifier_ep_train_X, ll_classifier_ep_train_Y, ll_classifier_ep_train_idx =\
              hmm.getEntropyFeaturesFromHMMInducedFeatures(ll_classifier_train_X, \
                                                           ll_classifier_train_Y, \
                                                           ll_classifier_train_idx, nState)
            ll_classifier_ep_test_X, ll_classifier_ep_test_Y, ll_classifier_ep_test_idx =\
              hmm.getEntropyFeaturesFromHMMInducedFeatures(ll_classifier_test_X, \
                                                           ll_classifier_test_Y, \
                                                           ll_classifier_test_idx, nState)

            ll_classifier_train_X   = ll_classifier_ep_train_X
            ll_classifier_train_Y   = ll_classifier_ep_train_Y
            ll_classifier_train_idx = ll_classifier_ep_train_idx
            ll_classifier_test_X    = ll_classifier_ep_test_X
            ll_classifier_test_Y    = ll_classifier_ep_test_Y
            ll_classifier_test_idx  = ll_classifier_ep_test_idx


        if method == 'hmmosvm' or method == 'progress_osvm' or method == 'hmmgp':            
            normal_idx = [x for x in range(len(ll_classifier_train_X)) if ll_classifier_train_Y[x][0]<0 ]
            ll_classifier_train_X = np.array(ll_classifier_train_X)[normal_idx]
            ll_classifier_train_Y = np.array(ll_classifier_train_Y)[normal_idx]
            ll_classifier_train_idx = np.array(ll_classifier_train_idx)[normal_idx]
        elif method == 'hmmsvm_dL':
            # replace dL/(ds+e) to dL
            for i in xrange(len(ll_classifier_train_X)):
                for j in xrange(len(ll_classifier_train_X[i])):
                    if j == 0:
                        ll_classifier_train_X[i][j][1] = 0.0
                    else:
                        ll_classifier_train_X[i][j][1] = ll_classifier_train_X[i][j][0] - \
                          ll_classifier_train_X[i][j-1][0]

            for i in xrange(len(ll_classifier_test_X)):
                for j in xrange(len(ll_classifier_test_X[i])):
                    if j == 0:
                        ll_classifier_test_X[i][j][1] = 0.0
                    else:
                        ll_classifier_test_X[i][j][1] = ll_classifier_test_X[i][j][0] - \
                          ll_classifier_test_X[i][j-1][0]
        elif method == 'hmmsvm_LSLS':
            # reconstruct data into LS(t-1)+LS(t)
            if type(ll_classifier_train_X) is list:
                ll_classifier_train_X = np.array(ll_classifier_train_X)

            x = np.dstack([ll_classifier_train_X[:,:,:1], ll_classifier_train_X[:,:,2:]] )
            x = x.tolist()

            new_x = []
            for i in xrange(len(x)):
                new_x.append([])
                for j in xrange(len(x[i])):
                    if j == 0:
                        new_x[i].append( x[i][j]+x[i][j] )
                    else:
                        new_x[i].append( x[i][j-1]+x[i][j] )

            ll_classifier_train_X = new_x

            # test data
            if len(np.shape(ll_classifier_test_X))<3:
                x = []
                for sample in ll_classifier_test_X:
                    x.append( np.hstack( [np.array(sample)[:,:1], np.array(sample)[:,2:]] ).tolist() )
            else:
                if type(ll_classifier_test_X) is list:
                    ll_classifier_test_X = np.array(ll_classifier_test_X)

                x = np.dstack([ll_classifier_test_X[:,:,:1], ll_classifier_test_X[:,:,2:]] )
                x = x.tolist()

            new_x = []
            for i in xrange(len(x)):
                new_x.append([])
                for j in xrange(len(x[i])):
                    if j == 0:
                        new_x[i].append( x[i][j]+x[i][j] )
                    else:
                        new_x[i].append( x[i][j-1]+x[i][j] )

            ll_classifier_test_X = new_x
        elif (method == 'hmmsvm_no_dL' or add_logp_d is False) and \
          len(ll_classifier_train_X[0][0]) > 1+nState:
            # remove dL related things
            ll_classifier_train_X = np.array(ll_classifier_train_X)
            ll_classifier_train_X = np.delete(ll_classifier_train_X, 1, 2).tolist()

            if len(np.shape(ll_classifier_test_X))<3:
                x = []
                for sample in ll_classifier_test_X:
                    x.append( np.hstack( [np.array(sample)[:,:1], np.array(sample)[:,2:]] ).tolist() )
                ll_classifier_test_X = x
            else:
                ll_classifier_test_X = np.array(ll_classifier_test_X)
                ll_classifier_test_X = np.delete(ll_classifier_test_X, 1, 2).tolist()

        if method == 'hmmgp':
            ## nSubSample = 40 #temp!!!!!!!!!!!!!
            nSubSample = 20 #20 # 20 
            nMaxData   = 50 # 40 100
            rnd_sample = True #False
            
            ll_classifier_train_X, ll_classifier_train_Y, ll_classifier_train_idx =\
              dm.subsampleData(ll_classifier_train_X, ll_classifier_train_Y, ll_classifier_train_idx,\
                               nSubSample=nSubSample, nMaxData=nMaxData, rnd_sample=rnd_sample)
            
        # flatten the data
        if method.find('svm')>=0 or method.find('sgd')>=0: remove_fp=True
        else: remove_fp = False
        X_train_org, Y_train_org, idx_train_org = dm.flattenSample(ll_classifier_train_X, \
                                                                   ll_classifier_train_Y, \
                                                                   ll_classifier_train_idx,\
                                                                   remove_fp=remove_fp)

        # Add failure safe data
        if failsafe and False:
            if ( (method.find('svm')>=0 or method.find('sgd')>=0) ) and False:
                for i in xrange(nState):
                    if len(X_train_org[0])>nState+1:
                        v                     = np.zeros(nState*2+1)
                        v[0]                  = -500
                        v[i+1]                = 1.0
                        v[i+1+nState] = 1.0
                    else:
                        v                     = np.zeros(nState+1)
                        v[0]                  = -500
                        v[i+1]                = 1.0
                    print np.shape(v), np.shape(X_train_org[0])
                    X_train_org.append(v.tolist())
                    Y_train_org.append(1)
                    idx_train_org.append(i)
            elif method == 'progress_svm':
                min_likelihood = np.amin(ll_classifier_train_X[:,:,0])
                min_selfInfo   = np.amin(ll_classifier_train_X[:,:,2])
                
                for i in xrange(nState):
                    v    = np.zeros(3)
                    v[0] = min_likelihood
                    v[1] = float(i)
                    v[2] = min_selfInfo
                    
                    X_train_org.append(v.tolist())
                    Y_train_org.append(1)
                    idx_train_org.append(i)
                    

    #-----------------------------------------------------------------------------------------
    # Generate parameter list for ROC curve
    # pass method if there is existing result
    # data preparation
    if (method.find('svm')>=0 or method.find('sgd')>=0) and not(method == 'osvm' or method == 'bpsvm'):
        scaler = preprocessing.StandardScaler()
        X_scaled = scaler.fit_transform(X_train_org)
    else:
        X_scaled = X_train_org
    print method, " : Before classification : ", np.shape(X_scaled), np.shape(Y_train_org)

    X_test = []
    Y_test = [] 
    for j in xrange(len(ll_classifier_test_X)):
        if len(ll_classifier_test_X[j])==0: continue

        if (method.find('svm')>=0 or method.find('sgd')>=0) and not(method == 'osvm' or method == 'bpsvm'):
            X = scaler.transform(ll_classifier_test_X[j])                                
        else:
            X = ll_classifier_test_X[j]

        X_test.append(X)
        Y_test.append(ll_classifier_test_Y[j])


    # classifier # TODO: need to make it efficient!!
    dtc = classifier( method=method, nPosteriors=nState, nLength=nLength )
    for j in xrange(nPoints):

        dtc.set_params( **SVM_dict )
        if method == 'svm' or method == 'hmmsvm_diag' or method == 'hmmsvm_dL' or method == 'hmmsvm_LSLS' or \
          method == 'bpsvm' or method == 'hmmsvm_no_dL' or method == 'sgd' or method == 'progress_svm' or \
          method == 'svm_fixed':
            if method == 'svm_fixed': 
                weights = ROC_dict['svm_param_range']
            else:
                weights = ROC_dict[method+'_param_range']
            dtc.set_params( class_weight=weights[j] )
            ret = dtc.fit(X_scaled, Y_train_org, idx_train_org, parallel=False)
        elif method == 'hmmosvm' or method == 'osvm' or method == 'progress_osvm':
            weights = ROC_dict[method+'_param_range']
            dtc.set_params( svm_type=2 )
            dtc.set_params( kernel_type=2 )
            dtc.set_params( gamma=weights[j] )
            ret = dtc.fit(X_scaled, np.array(Y_train_org)*-1.0, parallel=False)
        elif method == 'cssvm':
            weights = ROC_dict[method+'_param_range']
            dtc.set_params( class_weight=weights[j] )
            ret = dtc.fit(X_scaled, np.array(Y_train_org)*-1.0, idx_train_org, parallel=False)                
        elif method == 'progress' or method == 'progress_diag' or method == 'progress_state' or \
          method == 'fixed' or method == 'kmean' or method == 'hmmgp' or method == 'state_kmean':
            thresholds = ROC_dict[method+'_param_range']
            dtc.set_params( ths_mult = thresholds[j] )
            if j==0: ret = dtc.fit(X_scaled, Y_train_org, idx_train_org, parallel=False)
        elif method == 'change':
            thresholds = ROC_dict[method+'_param_range']
            dtc.set_params( ths_mult = thresholds[j] )
            if j==0: ret = dtc.fit(ll_classifier_train_X, ll_classifier_train_Y, ll_classifier_train_idx)
        elif method == 'rnd':
            weights = ROC_dict[method+'_param_range']
            dtc.set_params( class_weight=weights[j] )
            ret = True
        else:
            print "Not available method", method
            return "Not available method", -1, params

        if ret is False:
            print "fit failed, ", weights[j]
            sys.exit()
            return 'fit failed', [],[],[],[],[]
        
        # evaluate the classifier
        tp_l = []
        fp_l = []
        tn_l = []
        fn_l = []
        delay_l   = []
        delay_idx = 0
        tp_idx_l  = []
        fn_labels = []
        for ii in xrange(len(X_test)):
            if len(Y_test[ii])==0: continue

            if method.find('osvm')>=0 or method == 'cssvm':
                est_y = dtc.predict(X_test[ii], y=np.array(Y_test[ii])*-1.0)
                est_y = np.array(est_y)* -1.0
            else:
                est_y = dtc.predict(X_test[ii], y=Y_test[ii])

            anomaly = False
            for jj in xrange(len(est_y)):
                if est_y[jj] > 0.0:
                    if ll_classifier_test_idx is not None and Y_test[ii][0]>0:
                        try:
                            delay_idx = ll_classifier_test_idx[ii][jj]
                        except:
                            print "Error!!!!!!!!!!!!!!!!!!"
                            print np.shape(ll_classifier_test_idx), ii, jj
                        if delay_estimation:
                            delay_l.append(delay_idx-step_idx_l[ii])
                        else:
                            delay_l.append(delay_idx)
                    if Y_test[ii][0] > 0:
                        tp_idx_l.append(ii)

                    ## if Y_test[ii][0] < 0:
                    ##     print jj, Y_test[ii][0]
                    anomaly = True
                    break        

            if Y_test[ii][0] > 0.0:
                if anomaly: tp_l.append(1)
                else:
                    fn_l.append(1)
                    if ll_classifier_test_labels is not None:
                        fn_labels.append(ll_classifier_test_labels[ii])
            elif Y_test[ii][0] <= 0.0:
                if anomaly: fp_l.append(1)
                else: tn_l.append(1)

        data[method]['tp_l'][j] += tp_l
        data[method]['fp_l'][j] += fp_l
        data[method]['fn_l'][j] += fn_l
        data[method]['tn_l'][j] += tn_l
        data[method]['delay_l'][j] += delay_l
        data[method]['tp_idx_l'][j] += tp_idx_l
        data[method]['fn_labels'][j] += fn_labels

    print "finished ", idx, method
    return data


