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


import numpy as np
import sys, os, copy
from scipy.stats import norm, entropy

# Util
import roslib
import hrl_lib.util as ut

# Matplot
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.collections as collections

import ghmm
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
from joblib import Parallel, delayed

os.system("taskset -p 0xff %d" % os.getpid())

class learning_hmm_multi_n:
    def __init__(self, nState, nEmissionDim=4, check_method='progress', anomaly_offset=0.0, \
                 cluster_type='time', verbose=False):
        self.ml = None
        self.verbose = verbose

        ## Tunable parameters
        self.nState         = nState # the number of hidden states
        self.nGaussian      = nState
        self.nEmissionDim   = nEmissionDim
        self.anomaly_offset = anomaly_offset
        
        ## Un-tunable parameters
        self.trans_type = 'left_right' # 'left_right' 'full'
        self.A  = None # transition matrix        
        self.B  = None # emission matrix
        self.pi = None # Initial probabilities per state
        self.check_method = check_method # ['global', 'progress']
        self.cluster_type = cluster_type

        self.l_statePosterior = None
        self.ll_mu = None
        self.ll_std = None
        self.l_mean_delta = None
        self.l_std_delta = None
        self.l_mu = None
        self.l_std = None
        self.std_coff = None
        self.km = None


        # emission domain of this model        
        self.F = ghmm.Float()  

        # print 'HMM initialized for', self.check_method

    def fit(self, xData, A=None, B=None, pi=None, cov_mult=None,
            ml_pkl=None, use_pkl=False):

        if ml_pkl is None:
            ml_pkl = os.path.join(os.path.dirname(__file__), 'ml_temp_n.pkl')            
        
        if cov_mult is None:
            cov_mult = [1.0]*(self.nEmissionDim**2)

        # Daehyung: What is the shape and type of input data?
        X = [np.array(data) for data in xData]

        if A is None:
            if self.verbose: print "Generating a new A matrix"
            # Transition probability matrix (Initial transition probability, TODO?)
            A = self.init_trans_mat(self.nState).tolist()

        if B is None:
            if self.verbose: print "Generating a new B matrix"
            # We should think about multivariate Gaussian pdf.  

            mus, cov = self.vectors_to_mean_cov(X, self.nState)

            for i in xrange(self.nEmissionDim):
                for j in xrange(self.nEmissionDim):
                    cov[:, j, i] *= cov_mult[self.nEmissionDim*i + j]

            if self.verbose:
                for i, mu in enumerate(mus):
                    print 'mu%i' % i, mu
                print 'cov', cov
                
            # Emission probability matrix
            B = [0] * self.nState
            for i in range(self.nState):
                B[i] = [[mu[i] for mu in mus]]
                B[i].append(cov[i].flatten())
        if pi is None:
            # pi - initial probabilities per state 
            ## pi = [1.0/float(self.nState)] * self.nState
            pi = [0.0] * self.nState
            pi[0] = 1.0

        # print 'Generating HMM'
        # HMM model object
        self.ml = ghmm.HMMFromMatrices(self.F, ghmm.MultivariateGaussianDistribution(self.F), A, B, pi)
        # print 'Creating Training Data'
        X_train = self.convert_sequence(X) # Training input
        X_train = X_train.tolist()
        
        if self.verbose: print 'Run Baum Welch method with (samples, length)', np.shape(X_train)
        final_seq = ghmm.SequenceSet(self.F, X_train)
        ## ret = self.ml.baumWelch(final_seq, loglikelihoodCutoff=2.0)
        ret = self.ml.baumWelch(final_seq, 10000)
        print 'Baum Welch return:', ret
        if np.isnan(ret): return 'Failure'

        [self.A, self.B, self.pi] = self.ml.asMatrices()
        self.A = np.array(self.A)
        self.B = np.array(self.B)

        #--------------- learning for anomaly detection ----------------------------
        [A, B, pi] = self.ml.asMatrices()
        n, m = np.shape(X[0])
        self.nGaussian = self.nState

        if self.check_method == 'change' or self.check_method == 'globalChange':
            # Get maximum change of loglikelihood over whole time
            ll_delta_logp = []
            for j in xrange(n):    
                l_logp = []                
                for k in xrange(1, m):
                    final_ts_obj = ghmm.EmissionSequence(self.F, X_train[j][:k*self.nEmissionDim])
                    logp         = self.ml.loglikelihoods(final_ts_obj)[0]

                    l_logp.append(logp)
                l_delta_logp = np.array(l_logp[1:]) - np.array(l_logp[:-1])                    
                ll_delta_logp.append(l_delta_logp)

            self.l_mean_delta = np.mean(abs(np.array(ll_delta_logp).flatten()))
            self.l_std_delta = np.std(abs(np.array(ll_delta_logp).flatten()))

            if self.verbose: 
                print "mean_delta: ", self.l_mean_delta, " std_delta: ", self.l_std_delta
        
        
        if self.check_method == 'global' or self.check_method == 'globalChange':
            # Get average loglikelihood threshold over whole time

            l_logp = []
            for j in xrange(n):
                for k in xrange(1, m):
                    final_ts_obj = ghmm.EmissionSequence(self.F, X_train[j][:k*self.nEmissionDim])
                    logp         = self.ml.loglikelihoods(final_ts_obj)[0]

                    l_logp.append(logp)

            self.l_mu = np.mean(l_logp)
            self.l_std = np.std(l_logp)
            
        elif self.check_method == 'progress':
            # Get average loglikelihood threshold wrt progress

            if os.path.isfile(ml_pkl) and use_pkl:
                if self.verbose: print 'Load detector parameters'
                d = ut.load_pickle(ml_pkl)
                self.l_statePosterior = d['state_post'] # time x state division
                self.ll_mu            = d['ll_mu']
                self.ll_std           = d['ll_std']
            else:
                if self.cluster_type == 'time':                
                    if self.verbose: print 'Begining parallel job'
                    self.std_coff  = 1.0
                    g_mu_list = np.linspace(0, m-1, self.nGaussian) #, dtype=np.dtype(np.int16))
                    g_sig = float(m) / float(self.nGaussian) * self.std_coff
                    r = Parallel(n_jobs=-1)(delayed(learn_likelihoods_progress)(i, n, m, A, B, pi, self.F, X_train,
                                                                           self.nEmissionDim, g_mu_list[i], g_sig, self.nState)
                                                                           for i in xrange(self.nGaussian))
                    if self.verbose: print 'Completed parallel job'
                    l_i, self.l_statePosterior, self.ll_mu, self.ll_std = zip(*r)

                elif self.cluster_type == 'state':
                    self.km = None                    
                    self.ll_mu = None
                    self.ll_std = None
                    self.ll_mu, self.ll_std = self.state_clustering(X)
                    path_mat  = np.zeros((self.nState, m*n))
                    likelihood_mat = np.zeros((1, m*n))
                    self.l_statePosterior=None
                    
                d = dict()
                d['state_post'] = self.l_statePosterior
                d['ll_mu'] = self.ll_mu
                d['ll_std'] = self.ll_std
                ut.save_pickle(d, ml_pkl)
                            
                    
    def get_sensitivity_gain(self, X):
        X_test = self.convert_sequence(X, emission=False)

        try:
            final_ts_obj = ghmm.EmissionSequence(self.F, X_test[0].tolist())
            logp = self.ml.loglikelihood(final_ts_obj)
        except:
            if self.verbose: print "Too different input profile that cannot be expressed by emission matrix"
            return [], 0.0 # error

        if self.check_method == 'progress':
            try:
                post = np.array(self.ml.posterior(final_ts_obj))
            except:
                if self.verbose: print "Unexpected profile!! GHMM cannot handle too low probability. Underflow?"
                return [], 0.0 # anomaly

            n = len(np.squeeze(X[0]))

            # Find the best posterior distribution
            min_index, min_dist = self.findBestPosteriorDistribution(post[n-1])

            ths = (logp - self.ll_mu[min_index])/self.ll_std[min_index]
            ## if logp >= 0.:                
            ##     ths = (logp*0.95 - self.ll_mu[min_index])/self.ll_std[min_index]
            ## else:
            ##     ths = (logp*1.05 - self.ll_mu[min_index])/self.ll_std[min_index]
                        
            return ths, min_index

        elif self.check_method == 'global':
            ths = (logp - self.l_mu) / self.l_std
            return ths, 0

        elif self.check_method == 'change':
            if len(X[0])<3: return [], 0.0 #error

            X_test = self.convert_sequence([x[:-1] for x in X], emission=False)

            try:
                final_ts_obj = ghmm.EmissionSequence(self.F, X_test[0].tolist())
                last_logp         = self.ml.loglikelihood(final_ts_obj)
            except:
                print "Too different input profile that cannot be expressed by emission matrix"
                return -1, 0.0 # error
            
            ths = -(( abs(logp-last_logp) - self.l_mean_delta) / self.l_std_delta)
            return ths, 0

        elif self.check_method == 'globalChange':
            if len(X[0])<3: return [], 0.0 #error

            X_test = self.convert_sequence([x[:-1] for x in X], emission=False)

            try:
                final_ts_obj = ghmm.EmissionSequence(self.F, X_test[0].tolist())
                last_logp         = self.ml.loglikelihood(final_ts_obj)
            except:
                print "Too different input profile that cannot be expressed by emission matrix"
                return [], 0.0 # error
            
            ths_c = -(( abs(logp-last_logp) - self.l_mean_delta) / self.l_std_delta)

            ths_g = (logp - self.l_mu) / self.l_std
            
            return [ths_c, ths_g], 0
        

    def path_disp(self, X):
        X = [np.array(x) for x in X]
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        n, m = np.shape(X[0])
        if self.verbose: print n, m
        x = np.arange(0., float(m))*(1./43.)
        path_mat  = np.zeros((self.nState, m))
        zbest_mat = np.zeros((self.nState, m))

        path_l = []
        for i in xrange(n):
            x_test = [x[i:i+1,:] for x in X]

            if self.nEmissionDim == 1:
                X_test = x_test[0]
            else:
                X_test = self.convert_sequence(x_test, emission=False)

            final_ts_obj = ghmm.EmissionSequence(self.F, X_test[0].tolist())
            path,_    = self.ml.viterbi(final_ts_obj)
            post = self.ml.posterior(final_ts_obj)

            use_last = False
            for j in xrange(m):
                ## sum_post = np.sum(post[j*2+1])
                ## if sum_post <= 0.1 or sum_post > 1.1 or sum_post == float('Inf') or use_last == True:
                ##     use_last = True
                ## else:
                add_post = np.array(post[j])/float(n)
                path_mat[:, j] += add_post

            path_l.append(path)
            for j in xrange(m):
                zbest_mat[path[j], j] += 1.0

        path_mat /= np.sum(path_mat, axis=0)

        # maxim = np.max(path_mat)
        # path_mat = maxim - path_mat

        zbest_mat /= np.sum(zbest_mat, axis=0)

        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42

        fig = plt.figure()
        plt.rc('text', usetex=True)

        ax1 = plt.subplot(111)
        im  = ax1.imshow(path_mat, cmap=plt.cm.Reds, interpolation='none', origin='upper',
                         extent=[0, float(m)*(1.0/10.), 20, 1], aspect=0.85)

        ## divider = make_axes_locatable(ax1)
        ## cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, fraction=0.031, ticks=[0.0, 1.0], pad=0.01)
        ax1.set_xlabel("Time (sec)", fontsize=18)
        ax1.set_ylabel("Hidden State Index", fontsize=18)
        ax = plt.gca()
        ax.set_ylim(ax.get_ylim()[::-1])

        ## for p in path_l:
        ##     ax1.plot(x, p, '*')

        ## ax2 = plt.subplot(212)
        ## im2 = ax2.imshow(zbest_mat, cmap=plt.cm.Reds, interpolation='none', origin='upper',
        ##                  extent=[0,float(m)*(1.0/43.),20,1], aspect=0.1)
        ## plt.colorbar(im2, fraction=0.031, ticks=[0.0, 1.0], pad=0.01)
        ## ax2.set_xlabel("Time [sec]", fontsize=18)
        ## ax2.set_ylabel("Hidden State", fontsize=18)


        ## ax3 = plt.subplot(313)
        # fig.savefig('test.pdf')
        # fig.savefig('test.png')
        plt.grid()
        plt.show()

    def predict(self, X):
        X = np.squeeze(X)
        X_test = X.tolist()

        mu_l = np.zeros(self.nEmissionDim)
        cov_l = np.zeros(self.nEmissionDim**2)

        if self.verbose: print self.F
        final_ts_obj = ghmm.EmissionSequence(self.F, X_test) # is it neccessary?

        try:
            # alpha: X_test length y # latent States at the moment t when state i is ended
            # test_profile_length x number_of_hidden_state
            (alpha, scale) = self.ml.forward(final_ts_obj)
            alpha = np.array(alpha)
        except:
            if self.verbose: print "No alpha is available !!"
            
        f = lambda x: round(x, 12)
        for i in range(len(alpha)):
            alpha[i] = map(f, alpha[i])
        alpha[-1] = map(f, alpha[-1])
        
        n = len(X_test)
        pred_numerator = 0.0

        for j in xrange(self.nState): # N+1
            total = np.sum(self.A[:,j]*alpha[n/self.nEmissionDim-1,:]) #* scaling_factor
            [mus, covars] = self.B[j]

            ## print mu1, mu2, cov11, cov12, cov21, cov22, total
            pred_numerator += total

            for i in xrange(mu_l.size):
                mu_l[i] += mus[i]*total
            for i in xrange(cov_l.size):
                cov_l[i] += covars[i] * (total**2)

        return mu_l, cov_l

    def loglikelihood(self, X):
        X = np.squeeze(X)
        X_test = X.tolist()        

        final_ts_obj = ghmm.EmissionSequence(self.F, X_test)

        try:    
            p = self.ml.loglikelihood(final_ts_obj)
        except:
            if self.verbose: print 'Likelihood error!!!!'
            sys.exit()

        return p

    def likelihoods(self, X):
        X_test = self.convert_sequence(X, emission=False)

        final_ts_obj = ghmm.EmissionSequence(self.F, X_test[0].tolist())
        logp = self.ml.loglikelihood(final_ts_obj)
        post = np.array(self.ml.posterior(final_ts_obj))

        n = len(np.squeeze(X[0]))

        # Find the best posterior distribution
        min_index, min_dist = self.findBestPosteriorDistribution(post[n-1])

        ll_likelihood = logp
        ll_state_idx  = min_index
        ll_likelihood_mu  = self.ll_mu[min_index]
        ll_likelihood_std = self.ll_std[min_index] #self.ll_mu[min_index] + ths_mult*self.ll_std[min_index]

        return ll_likelihood, ll_state_idx, ll_likelihood_mu, ll_likelihood_std

    def allLikelihoods(self, X):
        X_test = self.convert_sequence(X, emission=False)

        m = len(np.squeeze(X[0]))

        ll_likelihood = np.zeros(m)
        ll_state_idx  = np.zeros(m)
        ll_likelihood_mu  = np.zeros(m)
        ll_likelihood_std = np.zeros(m)
        for i in xrange(1, m):
            final_ts_obj = ghmm.EmissionSequence(self.F, X_test[0,:i*self.nEmissionDim].tolist())
            logp = self.ml.loglikelihood(final_ts_obj)
            post = np.array(self.ml.posterior(final_ts_obj))

            # Find the best posterior distribution
            min_index, min_dist = self.findBestPosteriorDistribution(post[i-1])

            ll_likelihood[i] = logp
            ll_state_idx[i]  = min_index
            ll_likelihood_mu[i]  = self.ll_mu[min_index]
            ll_likelihood_std[i] = self.ll_std[min_index] #self.ll_mu[min_index] + ths_mult*self.ll_std[min_index]

        return ll_likelihood, ll_state_idx, ll_likelihood_mu, ll_likelihood_std

    # Returns mu,sigma for n hidden-states from feature-vector
    @staticmethod
    def vectors_to_mean_sigma(vec, nState):
        index = 0
        m,n = np.shape(vec)
        mu  = np.zeros(nState)
        sig = np.zeros(nState)
        DIVS = n/nState

        while index < nState:
            m_init = index*DIVS
            temp_vec = vec[:, m_init:(m_init+DIVS)]
            temp_vec = np.reshape(temp_vec, (1, DIVS*m))
            mu[index]  = np.mean(temp_vec)
            sig[index] = np.std(temp_vec)
            index += 1

        return mu, sig
        
    # Returns mu,sigma for n hidden-states from feature-vector
    def vectors_to_mean_cov(self, vecs, nState):
        index = 0
        m, n = np.shape(vecs[0])
        #print m,n
        mus  = [np.zeros(nState) for i in xrange(self.nEmissionDim)]
        cov  = np.zeros((nState, self.nEmissionDim, self.nEmissionDim))
        DIVS = n/nState

        while index < nState:
            m_init = index*DIVS
            temp_vecs = [np.reshape(vec[:, m_init:(m_init+DIVS)], (1, DIVS*m)) for vec in vecs]
            for i, mu in enumerate(mus):
                mu[index] = np.mean(temp_vecs[i])
            cov[index, :, :] = np.cov(np.concatenate(temp_vecs, axis=0))
            index += 1

        return mus, cov

    @staticmethod
    def init_trans_mat(nState):
        # Reset transition probability matrix
        trans_prob_mat = np.zeros((nState, nState))

        for i in xrange(nState):
            # Exponential function
            # From y = a*e^(-bx)
            ## a = 0.4
            ## b = np.log(0.00001/a)/(-(nState-i))
            ## f = lambda x: a*np.exp(-b*x)

            # Linear function
            # From y = -a*x + b
            b = 0.4
            a = b/float(nState)
            f = lambda x: -a*x+b

            for j in np.array(range(nState-i))+i:
                trans_prob_mat[i, j] = f(j)

            # Gaussian transition probability
            ## z_prob = norm.pdf(float(i),loc=u_mu_list[i],scale=u_sigma_list[i])

            # Normalization
            trans_prob_mat[i,:] /= np.sum(trans_prob_mat[i,:])

        return trans_prob_mat

    @staticmethod
    def convert_sequence(data, emission=False):
        # change into array from other types
        X = [copy.copy(np.array(d)) if type(d) is not np.ndarray else copy.copy(d) for d in data]

        # Change into 2-dimensional array
        X = [np.reshape(x, (1, len(x))) if len(np.shape(x)) == 1 else x for x in X]

        n, m = np.shape(X[0])

        Seq = []
        for i in xrange(n):
            Xs = []
                
            if emission:
                for j in xrange(m):
                    Xs.append([x[i, j] for x in X])
                Seq.append(Xs)
            else:
                for j in xrange(m):
                    Xs.append([x[i, j] for x in X])
                Seq.append(np.array(Xs).flatten().tolist())

        return np.array(Seq)
        
    def anomaly_check(self, X, ths_mult=None):

        if self.nEmissionDim == 1: X_test = np.array([X[0]])
        else: X_test = self.convert_sequence(X, emission=False)

        try:
            final_ts_obj = ghmm.EmissionSequence(self.F, X_test[0].tolist())
            logp = self.ml.loglikelihood(final_ts_obj)
        except:
            if self.verbose: print "Too different input profile that cannot be expressed by emission matrix"
            return True, 0.0 # error

        if self.check_method == 'change' or self.check_method == 'globalChange':

            ## if len(X1)<3: 
            ##     if self.verbose: print "Too short profile!"
            ##     return -1, 0.0 #error

            X_test = self.convert_sequence([x[:-1] for x in X], emission=False)
            try:
                final_ts_obj = ghmm.EmissionSequence(self.F, X_test[0].tolist())
                last_logp         = self.ml.loglikelihood(final_ts_obj)
            except:
                print "Too different input profile that cannot be expressed by emission matrix"
                return True, 0.0 # error

            ## print self.l_mean_delta + ths_mult*self.l_std_delta, abs(logp-last_logp)
            if type(ths_mult) == list or type(ths_mult) == np.ndarray or type(ths_mult) == tuple:
                err = (self.l_mean_delta + (-1.0*ths_mult[0])*self.l_std_delta ) - abs(logp-last_logp)
            else:
                err = (self.l_mean_delta + (-1.0*ths_mult)*self.l_std_delta ) - abs(logp-last_logp)
            ## if err < self.anomaly_offset: return 1.0, 0.0 # anomaly            
            if err < 0.0: return True, 0.0 # anomaly            
            
        if self.check_method == 'global' or self.check_method == 'globalChange':
            if type(ths_mult) == list or type(ths_mult) == np.ndarray or type(ths_mult) == tuple:
                err = logp - (self.l_mu + ths_mult[1]*self.l_std)
            else:
                err = logp - (self.l_mu + ths_mult*self.l_std)

            if err<0.0: return True, err
            else: return False, err
            ## return err < 0.0, err
                
        elif self.check_method == 'progress':
            try:
                post = np.array(self.ml.posterior(final_ts_obj))
            except:
                if self.verbose: print "Unexpected profile!! GHMM cannot handle too low probability. Underflow?"
                return True, 0.0 # anomaly

            if len(X[0]) == 1:
                n = 1
            else:
                n = len(np.squeeze(X[0]))

            # Find the best posterior distribution
            min_index, min_dist = self.findBestPosteriorDistribution(post[n-1])

            print "Min index: ", min_index, " logp: ", logp, " ths_mult: ", ths_mult

            if (type(ths_mult) == list or type(ths_mult) == np.ndarray or type(ths_mult) == tuple) and len(ths_mult)>1:
                err = logp - (self.ll_mu[min_index] + ths_mult[min_index]*self.ll_std[min_index])
            else:
                err = logp - (self.ll_mu[min_index] + ths_mult*self.ll_std[min_index])

            if err < self.anomaly_offset: return True, err
            else: return False, err
            
        else:
            if err < 0.0: return True, err
            else: return False, err
            

            
    def expLikelihoods(self, X, ths_mult=None):
        if self.nEmissionDim == 1: X_test = np.array([X[0]])
        else: X_test = self.convert_sequence(X, emission=False)

        try:
            final_ts_obj = ghmm.EmissionSequence(self.F, X_test[0].tolist())
            logp = self.ml.loglikelihood(final_ts_obj)
        except:
            print "Too different input profile that cannot be expressed by emission matrix"
            return -1, 0.0 # error

        try:
            post = np.array(self.ml.posterior(final_ts_obj))
        except:
            print "Unexpected profile!! GHMM cannot handle too low probability. Underflow?"
            return 1.0, 0.0 # anomaly

        n = len(np.squeeze(X[0]))

        # Find the best posterior distribution
        min_index, min_dist = self.findBestPosteriorDistribution(post[n-1])

        # print 'Computing anomaly'
        # print logp
        # print self.ll_mu[min_index]
        # print self.ll_std[min_index]

        # print 'logp:', logp, 'll_mu', self.ll_mu[min_index], 'll_std', self.ll_std[min_index], 'mult_std', ths_mult*self.ll_std[min_index]

        if (type(ths_mult) == list or type(ths_mult) == np.ndarray or type(ths_mult) == tuple) and len(ths_mult)>1:
            ## print min_index, self.ll_mu[min_index], self.ll_std[min_index], ths_mult[min_index], " = ", (self.ll_mu[min_index] + ths_mult[min_index]*self.ll_std[min_index]) 
            return self.ll_mu[min_index] + ths_mult[min_index]*self.ll_std[min_index]
        else:
            return self.ll_mu[min_index] + ths_mult*self.ll_std[min_index]


        
    @staticmethod
    def scaling(X, min_c=None, max_c=None, scale=10.0, verbose=False):
        '''
        scale should be over than 10.0(?) to avoid floating number problem in ghmm.
        Return list type
        '''
        ## X_scaled = preprocessing.scale(np.array(X))

        if min_c is None or max_c is None:
            min_c = np.min(X)
            max_c = np.max(X)

        X_scaled = []
        for x in X:
            if verbose is True: print min_c, max_c, " : ", np.min(x), np.max(x)
            X_scaled.append(((x-min_c) / (max_c-min_c) * scale))

        return X_scaled, min_c, max_c

    def likelihood_disp(self, X, X_true, Z, Z_true, axisTitles, ths_mult, figureSaveName=None):
        n, m = np.shape(X[0])
        n2, m2 = np.shape(Z[0])
        if self.verbose: print "Input sequence X1: ", n, m
        if self.verbose: print 'Anomaly: ', self.anomaly_check(X, ths_mult)

        X_test = self.convert_sequence(X, emission=False)
        Z_test = self.convert_sequence(Z, emission=False)

        x = np.arange(0., float(m))
        z = np.arange(0., float(m2))
        ll_likelihood = np.zeros(m)
        ll_state_idx  = np.zeros(m)
        ll_likelihood_mu  = np.zeros(m)
        ll_likelihood_std = np.zeros(m)
        ll_thres_mult = np.zeros(m)
        for i in xrange(1, m):
            final_ts_obj = ghmm.EmissionSequence(self.F, X_test[0,:i*self.nEmissionDim].tolist())
            logp = self.ml.loglikelihood(final_ts_obj)
            post = np.array(self.ml.posterior(final_ts_obj))

            # Find the best posterior distribution
            min_index, min_dist = self.findBestPosteriorDistribution(post[i-1])

            ll_likelihood[i] = logp
            ll_state_idx[i]  = min_index
            ll_likelihood_mu[i]  = self.ll_mu[min_index]
            ll_likelihood_std[i] = self.ll_std[min_index] #self.ll_mu[min_index] + ths_mult*self.ll_std[min_index]
            ll_thres_mult[i] = ths_mult

        # state blocks
        block_flag = []
        block_x    = []
        block_state= []
        text_n     = []
        text_x     = []
        for i, p in enumerate(ll_state_idx):
            if i is 0:
                block_flag.append(0)
                block_state.append(0)
                text_x.append(0.0)
            elif ll_state_idx[i] != ll_state_idx[i-1]:
                if block_flag[-1] is 0: block_flag.append(1)
                else: block_flag.append(0)
                block_state.append( int(p) )
                text_x[-1] = (text_x[-1]+float(i-1))/2.0 - 0.5 #
                text_x.append(float(i))
            else:
                block_flag.append(block_flag[-1])
            block_x.append(float(i))
        text_x[-1] = (text_x[-1]+float(m-1))/2.0 - 0.5 #

        block_flag_interp = []
        block_x_interp    = []
        for i in xrange(len(block_flag)):
            block_flag_interp.append(block_flag[i])
            block_flag_interp.append(block_flag[i])
            block_x_interp.append( float(block_x[i]) )
            block_x_interp.append(block_x[i]+0.5)


        # y1 = (X1_true[0]/scale1[2])*(scale1[1]-scale1[0])+scale1[0]
        # y2 = (X2_true[0]/scale2[2])*(scale2[1]-scale2[0])+scale2[0]
        # y3 = (X3_true[0]/scale3[2])*(scale3[1]-scale3[0])+scale3[0]
        # y4 = (X4_true[0]/scale4[2])*(scale4[1]-scale4[0])+scale4[0]
        Y = [x_true[0] for x_true in X_true]

        ZY = [np.mean(z_true, axis=0) for z_true in Z_true]

        ## matplotlib.rcParams['figure.figsize'] = 8,7
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42

        fig = plt.figure()
        plt.rc('text', usetex=True)

        for index, (y, zy, title) in enumerate(zip(Y, ZY, axisTitles)):
            ax = plt.subplot('%i1%i' % (len(X) + 1, index + 1))
            ax.plot(x*(1./10.), y)
            ax.plot(z*(1./10.), zy, 'r')
            y_min = np.amin(y)
            y_max = np.amax(y)
            collection = collections.BrokenBarHCollection.span_where(np.array(block_x_interp)*(1./10.),
                                                                     ymin=0, ymax=y_max+0.5,
                                                                     # ymin=y_min - y_min/15.0, ymax=y_max + y_min/15.0,
                                                                     where=np.array(block_flag_interp)>0,
                                                                     facecolor='green',
                                                                     edgecolor='none', alpha=0.3)
            ax.add_collection(collection)
            ax.set_ylabel(title, fontsize=16)
            ax.set_xlim([0, x[-1]*(1./10.)])
            ax.set_ylim([y_min - 0.25, y_max + 0.5])
            # ax.set_ylim([y_min - y_min/15.0, y_max + y_min/15.0])

            # Text for progress
            if index == 0:
                for i in xrange(len(block_state)):
                    if i%2 is 0:
                        if i<10:
                            ax.text((text_x[i])*(1./10.), y_max+0.15, str(block_state[i]+1))
                        else:
                            ax.text((text_x[i]-1.0)*(1./10.), y_max+0.15, str(block_state[i]+1))
                    else:
                        if i<10:
                            ax.text((text_x[i])*(1./10.), y_max+0.06, str(block_state[i]+1))
                        else:
                            ax.text((text_x[i]-1.0)*(1./10.), y_max+0.06, str(block_state[i]+1))

        ax = plt.subplot('%i1%i' % (len(X) + 1, len(X) + 1))
        ax.plot(x*(1./10.), ll_likelihood, 'b', label='Actual from \n test data')
        ax.plot(x*(1./10.), ll_likelihood_mu, 'r', label='Expected from \n trained model')
        ax.plot(x*(1./10.), ll_likelihood_mu + ll_thres_mult*ll_likelihood_std, 'r--', label='Threshold')
        # ax.set_ylabel(r'$log P({\mathbf{X}} | {\mathbf{\theta}})$',fontsize=18)
        ax.set_ylabel('Log-likelihood', fontsize=16)
        ax.set_xlim([0, x[-1]*(1./10.)])

        # ax.legend(loc='upper left', fancybox=True, shadow=True, ncol=3, prop={'size':14})
        lgd = ax.legend(loc='upper center', fancybox=True, shadow=True, ncol=3, bbox_to_anchor=(0.5, -0.5), prop={'size':14})
        ax.set_xlabel('Time (sec)', fontsize=16)

        plt.subplots_adjust(bottom=0.15)

        if figureSaveName is None:
            plt.show()
        else:
            # fig.savefig('test.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
            fig.savefig(figureSaveName, bbox_extra_artists=(lgd,), bbox_inches='tight')

    def learn_likelihoods_progress_par(self, i, n, m, A, B, pi, X_train, g_mu, g_sig):
        l_likelihood_mean = 0.0
        l_likelihood_mean2 = 0.0
        l_statePosterior = np.zeros(self.nState)

        for j in xrange(n):
            results = Parallel(n_jobs=-1)(delayed(computeLikelihood)(self.F, k, X_train[j][:k*self.nEmissionDim], g_mu, g_sig, self.nEmissionDim, A, B, pi) for k in xrange(1, m))

            g_post = np.sum([r[0] for r in results], axis=0)
            g_lhood, g_lhood2, prop_sum = np.sum([r[1:] for r in results], axis=0)

            l_statePosterior += g_post / prop_sum / float(n)
            l_likelihood_mean += g_lhood / prop_sum / float(n)
            l_likelihood_mean2 += g_lhood2 / prop_sum / float(n)

        return i, l_statePosterior, l_likelihood_mean, np.sqrt(l_likelihood_mean2 - l_likelihood_mean**2)


    def state_clustering(self, X):
        n, m = np.shape(X[0])

        print n, m
        x = np.arange(0., float(m))*(1./43.)
        state_mat  = np.zeros((self.nState, m*n))
        likelihood_mat = np.zeros((1, m*n))

        count = 0           
        for i in xrange(n):
            for j in xrange(1, m):
                X_test = self.convert_sequence([x[i:i+1,:j] for x in X], emission=False)

                final_ts_obj = ghmm.EmissionSequence(self.F, X_test[0].tolist())
                ## path,_    = self.ml.viterbi(final_ts_obj)        
                post      = self.ml.posterior(final_ts_obj)
                logp      = self.ml.loglikelihood(final_ts_obj)

                state_mat[:, count] = np.array(post[j-1])
                likelihood_mat[0,count] = logp
                count += 1

        # k-means
        init_center = np.eye(self.nState, self.nState)
        self.km = KMeans(self.nState, init=init_center)
        idx_list = self.km.fit_predict(state_mat.transpose())

        # mean and variance of likelihoods
        l = []
        for i in xrange(self.nState):
            l.append([])

        for i, idx in enumerate(idx_list):
            l[idx].append(likelihood_mat[0][i]) 

        l_mean = []
        l_std = []
        for i in xrange(self.nState):
            l_mean.append( np.mean(l[i]) )
            l_std.append( np.std(l[i]) )
                
        return l_mean, l_std

    def findBestPosteriorDistribution(self, post):
        # Find the best posterior distribution
        min_dist  = 100000000
        min_index = 0

        if self.cluster_type == 'time':
            for j in xrange(self.nGaussian):
                dist = entropy(post, self.l_statePosterior[j])
                if min_dist > dist:
                    min_index = j
                    min_dist  = dist 
        else:
            print "state based clustering"
            min_index = self.km.predict(post)
            min_dist  = -1

        return min_index, min_dist
        
        
####################################################################
# functions for paralell computation
####################################################################

def learn_likelihoods_progress(i, n, m, A, B, pi, F, X_train, nEmissionDim, g_mu, g_sig, nState):
    if nEmissionDim >= 2:
        ml = ghmm.HMMFromMatrices(F, ghmm.MultivariateGaussianDistribution(F), A, B, pi)
    else:
        ml = ghmm.HMMFromMatrices(F, ghmm.GaussianDistribution(F), A, B, pi)

    l_likelihood_mean = 0.0
    l_likelihood_mean2 = 0.0
    l_statePosterior = np.zeros(nState)

    for j in xrange(n):    

        g_post = np.zeros(nState)
        g_lhood = 0.0
        g_lhood2 = 0.0
        prop_sum = 0.0

        for k in xrange(1, m):
            final_ts_obj = ghmm.EmissionSequence(F, X_train[j][:k*nEmissionDim])
            logp = ml.loglikelihoods(final_ts_obj)[0]
            # print 'Log likelihood:', logp
            post = np.array(ml.posterior(final_ts_obj))

            k_prop = norm(loc=g_mu, scale=g_sig).pdf(k)
            g_post += post[k-1] * k_prop
            g_lhood += logp * k_prop
            g_lhood2 += logp * logp * k_prop

            prop_sum  += k_prop

        l_statePosterior += g_post / prop_sum / float(n)
        l_likelihood_mean += g_lhood / prop_sum / float(n)
        l_likelihood_mean2 += g_lhood2 / prop_sum / float(n)

    return i, l_statePosterior, l_likelihood_mean, np.sqrt(l_likelihood_mean2 - l_likelihood_mean**2)
    
def computeLikelihood(F, k, data, g_mu, g_sig, nEmissionDim, A, B, pi):
    if nEmissionDim >= 2:
        hmm_ml = ghmm.HMMFromMatrices(F, ghmm.MultivariateGaussianDistribution(F), A, B, pi)
    else:
        hmm_ml = ghmm.HMMFromMatrices(F, ghmm.GaussianDistribution(F), A, B, pi)

    final_ts_obj = ghmm.EmissionSequence(F, data)
    logp = hmm_ml.loglikelihoods(final_ts_obj)[0]
    post = np.array(hmm_ml.posterior(final_ts_obj))

    k_prop = norm(loc=g_mu, scale=g_sig).pdf(k)
    g_post = post[k-1] * k_prop
    g_lhood = logp * k_prop
    g_lhood2 = logp * logp * k_prop
    prop_sum = k_prop

    # print np.shape(g_post), np.shape(g_lhood), np.shape(g_lhood2), np.shape(prop_sum)

    return g_post, g_lhood, g_lhood2, prop_sum

