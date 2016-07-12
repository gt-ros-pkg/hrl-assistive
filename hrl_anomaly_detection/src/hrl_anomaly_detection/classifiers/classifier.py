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
from sklearn import metrics

class classifier(learning_base):
    def __init__(self, method='svm', nPosteriors=10, nLength=200, ths_mult=-1.0,\
                 #progress time or state?
                 logp_offset = 0.0,\
                 # svm
                 class_weight=1.0, \
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
                 hmmosvm_nu  = 0.5,\
                 # osvm
                 osvm_nu     = 0.5,\
                 # cssvm
                 cssvm_degree      = 3,\
                 cssvm_gamma       = 0.3,\
                 cssvm_cost        = 4.,\
                 cssvm_w_negative  = 7.0,\
                 # sgd
                 sgd_gamma      = 2.0,\
                 sgd_w_negative = 1.0,\
                 sgd_n_iter     = 10,\
                 verbose=False):
        '''
        class_weight : positive class weight for svm
        nLength : only for progress-based classifier
        ths_mult: only for progress-based classifier
        '''              
        self.method = method
        self.dt     = None
        self.verbose = verbose

        if self.method.find('svm')>=0 and self.method is not 'cssvm':
            sys.path.insert(0, '/usr/lib/pymodules/python2.7')
            import svmutil as svm
            self.class_weight = class_weight
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
            self.hmmsvm_diag_nu = hmmsvm_diag_nu
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
        elif self.method == 'cssvm':
            sys.path.insert(0, os.path.expanduser('~')+'/git/cssvm/python')
            import cssvmutil as cssvm
            self.class_weight = class_weight
            self.svm_type    = svm_type
            self.kernel_type = kernel_type
            self.cssvm_degree     = cssvm_degree 
            self.cssvm_gamma      = cssvm_gamma 
            self.cssvm_cost       = cssvm_cost 
            self.cssvm_w_negative = cssvm_w_negative 
        elif self.method == 'progress_time_cluster' or self.method == 'progress_state':
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
        elif self.method == 'change':
            self.nLength   = nLength
            self.mu  = 0.0
            self.std = 0.0
        elif self.method == 'sgd':
            self.class_weight = class_weight
            self.sgd_w_negative = sgd_w_negative             
            self.sgd_gamma      = sgd_gamma
            self.sgd_n_iter     = sgd_n_iter 
            ## self.cost         = cost
                        
        learning_base.__init__(self)

    def fit(self, X, y, ll_idx=None, parallel=True):
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
              +' -w1 '+str(self.class_weight)\
              +' -r '+str(self.coef0)

            if self.method == 'osvm':
                commands = commands+' -n '+str(self.osvm_nu)+' -g '+str(self.gamma)\
                  +' -w-1 '+str(self.w_negative)+' -c '+str(self.cost)
            elif self.method == 'hmmosvm':
                commands = commands+' -n '+str(self.hmmosvm_nu)+' -g '+str(self.gamma)\
                  +' -w-1 '+str(self.w_negative)+' -c '+str(self.cost)
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
            else:
                commands = commands+' -n '+str(self.nu)+' -g '+str(self.gamma)\
                  +' -w-1 '+str(self.w_negative)+' -c '+str(self.cost)

            try: self.dt = svm.svm_train(y, X, commands )
            except:
                print "svm training failure"
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
            
        elif self.method == 'progress_time_cluster':
            if type(X) == list: X = np.array(X)
            ## ll_logp = X[:,0:1]
            ## ll_post = X[:,1:]
            if ll_idx is None:
                print "Error>> ll_idx is not inserted"
                sys.exit()
            else: ll_idx  = [ ll_idx[i] for i in xrange(len(ll_idx)) if y[i]<0 ]
            ll_logp = [ X[i,0] for i in xrange(len(X)) if y[i]<0 ]
            ll_post = [ X[i,-self.nPosteriors:] for i in xrange(len(X)) if y[i]<0 ]

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
            
            ## # mean and variance of likelihoods
            ## l = []
            ## for i in xrange(self.nPosteriors):
            ##     l.append([])

            ## for i, idx in enumerate(idx_list):
            ##     l[idx].append(likelihood_mat[0][i]) 

            ## self.ll_mu = []
            ## self.ll_std = []
            ## for i in xrange(self.nState):
            ##     self.ll_mu.append( np.mean(l[i]) )
            ##     self.ll_std.append( np.std(l[i]) )
            
            return True
        
        elif self.method == 'fixed':
            if type(X) == list: X = np.array(X)
            ll_logp = X[:,0:1]
            self.mu  = np.mean(ll_logp)
            self.std = np.std(ll_logp)
            return True

        elif self.method == 'change':
            if type(X) == list: X = np.array(X)

            l_idx = range(len(X))
            l_idx = [ i for i in l_idx if y[i][0]<0 ]

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
            self.dt = SGDClassifier(verbose=0,class_weight=d,n_iter=self.sgd_n_iter, #learning_rate='constant',\
                                    eta0=1e-2, shuffle=True, average=True)
            self.dt.fit(X_features, y)


    def partial_fit(self, X, y, classes=None, sample_weight=None, n_iter=1):
        '''
        X: samples x hmm-feature vec
        y: sample
        '''

        if self.method == 'sgd':
            ## if sample_weight is None: sample_weight = [self.class_weight]*len(X)
            X_features = self.rbf_feature.transform(X)
            for i in xrange(n_iter):
                self.dt.partial_fit(X_features,y, classes=classes, sample_weight=sample_weight)
        else:
            print "Not available method, ", self.method
            sys.exit()


    def predict(self, X, y=None):
        '''
        X is single sample
        return predicted values (not necessarily binaries)
        '''

        if self.method.find('svm')>=0:
            ## K_test = custom_kernel(X, self.X_train, gamma=self.gamma)
            
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
            return p_labels
        
        elif self.method == 'progress_time_cluster':
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
                ## c_time = float(nState - (min_index+1) )/float(nState) + 1.0
                ## c_time = np.logspace(0,-0.9,nState)[min_index]

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
        if self.method.find('svm')>=0  or self.method == 'fixed':
            if type(X) is not list:
                return self.predict(X.tolist())
            else:
                return self.predict(X)
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

        
    def save_model(fileName):
        if self.dt is None: 
            print "No trained classifier"
            return
        
        if self.method.find('svm')>=0 and self.method is not 'cssvm':
            sys.path.insert(0, '/usr/lib/pymodules/python2.7')
            import svmutil as svm            
            svm.svm_save_model(use_pkl, self.dt) 
        else:
            print "Not available method"

            
    def load_model(fileName):        
        if self.method.find('svm')>=0 and self.method is not 'cssvm':
            sys.path.insert(0, '/usr/lib/pymodules/python2.7')
            import svmutil as svm            
            self.dt = svm.svm_load_model(use_pkl) 
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

## def learn_state_clustering(i, ll_logp, ll_post, g_mu, g_sig, nState):
##     return i, 


def run_classifier(j, X_train, Y_train, idx_train, X_test, Y_test, idx_test, \
                   method, nState, nLength, nPoints, param_dict, ROC_dict, dtc=None):

    # classifier # TODO: need to make it efficient!!
    if dtc is None:
        dtc = classifier( method=method, nPosteriors=nState, nLength=nLength )        
    dtc.set_params( **param_dict )
    if method == 'svm' or method == 'hmmsvm_diag' or method == 'hmmsvm_dL' or method == 'hmmsvm_LSLS' or\
      method == 'hmmsvm_no_dL' :
        weights = ROC_dict[method+'_param_range']
        dtc.set_params( class_weight=weights[j] )
        ret = dtc.fit(X_train, Y_train, parallel=False)
    elif method == 'bpsvm':
        weights = ROC_dict[method+'_param_range']
        ## dtc.set_params( kernel_type=0 )
        dtc.set_params( class_weight=weights[j] )
        ret = dtc.fit(X_train, Y_train, parallel=False)
    elif method == 'hmmosvm' or method == 'osvm':
        weights = ROC_dict[method+'_param_range']
        dtc.set_params( svm_type=2 )
        dtc.set_params( gamma=weights[j] )
        ret = dtc.fit(X_train, np.array(Y_train)*-1.0, parallel=False)
    elif method == 'cssvm':
        weights = ROC_dict[method+'_param_range']
        dtc.set_params( class_weight=weights[j] )
        ret = dtc.fit(X_train, np.array(Y_train)*-1.0, idx_train, parallel=False)                
    elif method == 'progress_time_cluster':
        thresholds = ROC_dict['progress_param_range']
        dtc.set_params( ths_mult = thresholds[j] )
        if j==0: ret = dtc.fit(X_train, Y_train, idx_train, parallel=False)                
    elif method == 'progress_state':
        thresholds = ROC_dict[method+'_param_range']
        dtc.set_params( ths_mult = thresholds[j] )
        if j==0: ret = dtc.fit(X_train, Y_train, idx_train, parallel=False)                
    elif method == 'fixed':
        thresholds = ROC_dict['fixed_param_range']
        dtc.set_params( ths_mult = thresholds[j] )
        if j==0: ret = dtc.fit(X_train, Y_train, idx_train, parallel=False)                
    elif method == 'sgd':
        weights = ROC_dict[method+'_param_range']
        dtc.set_params( class_weight=weights[j] )
        ret = dtc.fit(X_train, Y_train, idx_train, parallel=False)                
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

        if method == 'osvm' or method == 'cssvm' or method == 'hmmosvm':
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

        ## print jj, est_y[jj], Y_test[ii][0], " - ", np.unique(Y_test[ii])
        if Y_test[ii][0] > 0.0:
            if anomaly:
                tp_l.append(1)
                delay_l.append(delay_idx)
            else: fn_l.append(1)
        elif Y_test[ii][0] <= 0.0:
            if anomaly: fp_l.append(1)
            else: tn_l.append(1)

    ##     print jj, est_y[jj], Y_test[ii][0], " - ", anomaly

    ## if np.sum(tp_l)+np.sum(fn_l) > 0 and np.sum(fp_l)+np.sum(tn_l)>0:
    ##     print "tpr, fpr: ", float(np.sum(tp_l))/float(np.sum(tp_l)+np.sum(fn_l))*100.0, float(np.sum(fp_l))/float(np.sum(fp_l)+np.sum(tn_l))*100.0

    return j, tp_l, fp_l, fn_l, tn_l, delay_l
