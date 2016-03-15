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

#################### WARNING ###################################
# This file will be deprecated soon. Please, don't add anything. 
################################################################

import numpy as np
import sys, os, math
from scipy.stats import norm, entropy
import warnings

from sklearn.base import BaseEstimator, ClassifierMixin

import ghmm
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
from joblib import Parallel, delayed
from util import getSubjectFiles
from hrl_anomaly_detection import util as dataUtil
from hrl_anomaly_detection import data_manager as dataMng

import learning_util as util
from hrl_anomaly_detection.hmm.learning_base import learning_base

# Util
import roslib
import hrl_lib.util as ut

# Matplot
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.collections as collections

os.system("taskset -p 0xff %d" % os.getpid())

# Optimization variables
rootPath = None
paramSet = ''
# Current downSampleSize
sampleSize = 0
scores = []
# Training Data is reloaded only when downSampleSize changes
trainData = None
results = None

class learning_hmm_multi_n(learning_base, BaseEstimator, ClassifierMixin):
    def __init__(self, downSampleSize=200, scale=1.0, nState=20, cov_mult=1.0, nEmissionDim=4,
                 check_method='none', anomaly_offset=0.0, cluster_type='time', verbose=False,
                 optimDataPath=None, folds=3, resultsList=None):
        '''
        check_method and cluter_type will be deprecated
        '''
        global rootPath, results
                 
        # parent class
        learning_base.__init__(self)
                 
        warnings.simplefilter("always", DeprecationWarning)
        
        self.ml = None
        self.verbose = verbose

        ## Tunable parameters
        self.downSampleSize = downSampleSize
        self.scale          = scale
        self.nState         = nState # the number of hidden states
        self.cov_mult       = cov_mult
        self.nEmissionDim   = nEmissionDim

        ## Un-tunable parameters
        self.trans_type = 'left_right' # 'left_right' 'full'
        self.A  = None # transition matrix        
        self.B  = None # emission matrix
        self.pi = None # Initial probabilities per state

        ## Optimization parameters
        self.params = None
        self.isFitted = False
        if rootPath is None: rootPath = optimDataPath
        if results is None: results = resultsList


        ####################################################################
        ## Uner this line, every parameter will be deprecated soon.
        self.anomaly_offset = anomaly_offset
        self.check_method = check_method # ['global', 'progress']

        # For non-parametric decision boundary estimation
        # use: knn
        # For parametric decision boundary estimation (ICRA 2016)
        # use: time
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
        self.l_ths_mult = [-1.0]*self.nState

        # emission domain of this model        
        self.F = ghmm.Float()

        # print 'HMM initialized for', self.check_method

    def set_hmm_object(self, A, B, pi):

        self.ml = ghmm.HMMFromMatrices(self.F, ghmm.MultivariateGaussianDistribution(self.F),
                                       A, B, pi)
        return self.ml

    def set_params(self, **parameters):
        global sampleSize, results
        # self.params = ''
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
            # self.params += '%s: %s, ' % (parameter, str(value))
            # Check if downSampleSize has changed. If so, reload training data
            if parameter == 'downSampleSize' and value != sampleSize:
                print 'Loading new data with', parameter, value
                self.loadData()
                sampleSize = value
                # Determine nEmission from new data
                self.nEmissionDim = len(trainData)
        if parameters != self.params:
            # Add new results list for parameter set
            results.append([self.params, [], 0])

        self.params = parameters
        print ', '.join(['%s: %s' % (p, str(v)) for p, v in self.params])
        return self

    def fit(self, xData=None, A=None, B=None, pi=None, cov_mult=None,
            ml_pkl=None, use_pkl=False, X=None, y=None):
        '''
        TODO: explanation of the shape and type of xData
        xData: dimension x sample x length  
        '''
        global trainData, results

        if X is not None:
            # We are currently optimizing. Begin set up for training HMM
            # Check if this parameter set has already caused one of the k-folds to return NaN as a result
            if results[-1][1] == 'NaN':
                return self

            xData = trainData[:, X]
            cov_mult = [self.cov_mult]*(self.nEmissionDim**2)

        # Daehyung: What is the shape and type of input data?
        X = [np.array(data)*self.scale for data in xData]
        
        param_dict = {}

        # Load pre-trained HMM without training
        if use_pkl and ml_pkl is not None and os.path.isfile(ml_pkl):
            if self.verbose: print "Load HMM parameters without train the hmm"
                
            param_dict = ut.load_pickle(ml_pkl)
            self.A  = param_dict['A']
            self.B  = param_dict['B']
            self.pi = param_dict['pi']                       
            self.ml = ghmm.HMMFromMatrices(self.F, ghmm.MultivariateGaussianDistribution(self.F),
                                           self.A, self.B, self.pi)

        else:
           
            if ml_pkl is None:
                ml_pkl = os.path.join(os.path.dirname(__file__), 'ml_temp_n.pkl')            

            if cov_mult is None:
                cov_mult = [1.0]*(self.nEmissionDim**2)

            if A is None:
                if self.verbose: print "Generating a new A matrix"
                # Transition probability matrix (Initial transition probability, TODO?)
                A = util.init_trans_mat(self.nState).tolist()

            if B is None:
                if self.verbose: print "Generating a new B matrix"
                # We should think about multivariate Gaussian pdf.  

                mus, cov = util.vectors_to_mean_cov(X, self.nState, self.nEmissionDim)
                ## print np.shape(mus), np.shape(cov)

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
            X_train = util.convert_sequence(X) # Training input
            X_train = X_train.tolist()
            if self.verbose: print "training data size: ", np.shape(X_train)

            if self.verbose: print 'Run Baum Welch method with (samples, length)', np.shape(X_train)
            final_seq = ghmm.SequenceSet(self.F, X_train)
            ## ret = self.ml.baumWelch(final_seq, loglikelihoodCutoff=2.0)
            ret = self.ml.baumWelch(final_seq, 10000)
            print 'Baum Welch return:', ret
            if np.isnan(ret):
                if X is not None:
                    # Speed up optimization
                    print 'HMM return was a failure!'
                    results[-1][1] = 'NaN'
                    self.isFitted = False
                return 'Failure'

            [self.A, self.B, self.pi] = self.ml.asMatrices()
            self.A = np.array(self.A)
            self.B = np.array(self.B)

            param_dict['A'] = self.A
            param_dict['B'] = self.B
            param_dict['pi'] = self.pi

        #--------------- learning for anomaly detection ----------------------------
        ## [A, B, pi] = self.ml.asMatrices()
        n, m = np.shape(X[0])

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
        
        
        elif self.check_method == 'global' or self.check_method == 'globalChange':
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
            # Get average loglikelihood threshold wrt progress used for ICRA2016
            if os.path.isfile(ml_pkl) and use_pkl:
                if self.verbose: print 'Load detector parameters'
                d = ut.load_pickle(ml_pkl)
                self.l_statePosterior = d['state_post'] # time x state division
                self.ll_mu            = d['ll_mu']
                self.ll_std           = d['ll_std']
            else:

                # Estimate loglikelihoods and corresponding posteriors
                r = Parallel(n_jobs=-1)(delayed(computeLikelihood)(i, self.A, self.B, self.pi, self.F,
                                                                   X_train[i],
                                                                   self.nEmissionDim, self.nState,
                                                                   bPosterior=True, converted_X=True)
                                                                   for i in xrange(n))
                _, ll_idx, ll_logp, ll_post = zip(*r)

                l_idx = []
                l_logp = []
                l_post = []
                for i in xrange(len(ll_logp)):
                    l_idx  += ll_idx[i]
                    l_logp += ll_logp[i]
                    l_post += ll_post[i]

                
                if self.cluster_type == 'time':                
                    if self.verbose: print 'Begining parallel job'
                    self.std_coff  = 1.0
                    g_mu_list = np.linspace(0, m-1, self.nState) #, dtype=np.dtype(np.int16))
                    g_sig = float(m) / float(self.nState) * self.std_coff
                    ## r = Parallel(n_jobs=-1)(delayed(learn_time_clustering)(i, n, m, A, B, pi, self.F, X_train,
                    ##                                                        self.nEmissionDim, g_mu_list[i], \
                    ##                                                        g_sig, self.nState)
                    ##                                                        for i in xrange(self.nState))
                    r = Parallel(n_jobs=-1)(delayed(learn_time_clustering)(i, ll_idx, ll_logp, ll_post,
                                                                           g_mu_list[i],
                                                                           g_sig, self.nState)
                                            for i in xrange(self.nState))
                    
                    if self.verbose: print 'Completed parallel job'
                    _, self.l_statePosterior, self.ll_mu, self.ll_std = zip(*r)

                elif self.cluster_type == 'kmean':
                    self.km = None                    
                    self.ll_mu = None
                    self.ll_std = None
                    self.ll_mu, self.ll_std = self.state_clustering(X)
                    path_mat  = np.zeros((self.nState, m*n))
                    likelihood_mat = np.zeros((1, m*n))
                    self.l_statePosterior=None
                    
                elif self.cluster_type == 'state':
                    if self.verbose: print 'Begining parallel job'

                    ## # temp
                    ## for i in xrange(self.nState):
                    ##     learn_state_clustering(i, ll_idx, ll_logp, ll_post, self.nState)
                    ##     print '-------------- ', i , ' ---------------------------'
                                                
                    r = Parallel(n_jobs=-1)(delayed(learn_state_clustering)(i, ll_idx, ll_logp, ll_post,
                                                                            self.nState)
                                            for i in xrange(self.nState))
                    if self.verbose: print 'Completed parallel job'
                        
                    _, self.l_statePosterior, self.ll_mu, self.ll_std = zip(*r)
                                            
                param_dict['state_post'] = self.l_statePosterior
                param_dict['ll_mu'] = self.ll_mu
                param_dict['ll_std'] = self.ll_std

        ## elif self.check_method == 'none':
        if ml_pkl is not None: ut.save_pickle(param_dict, ml_pkl)
        self.isFitted = True
        sys.stdout.flush()
        return self


    def score(self, X, y, sample_weight=None):
        global scores, trainData, results
        score = 0

        if self.isFitted and results[-1][1] != 'NaN':
            trainingData = trainData[:, X]

            log_ll = []
            for i in xrange(len(trainingData[0])):
                log_ll.append([])
                # Compute likelihood values for data
                for j in range(2, len(trainingData[0][i])):
                    X = [x[i,:j] for x in trainingData]
                    logp = self.loglikelihood(X)
                    log_ll[i].append(logp)

            # Return average log-likelihood
            logs = [x[-1] for x in log_ll]
            score = sum(logs) / float(len(logs))
            if math.isnan(score):
                results[-1][1] = 'NaN'
            else:
                results[-1][1].append(score)
                scores.append(score)
                print 'loglikelihood() log_ll:', np.shape(log_ll), score

        sys.stdout.flush()
        return score

    # Load training data similar to the approach taken in data_manager.py
    def loadData(self, excludeStationary=True):
        global trainData, rootPath
        # Loading success and failure data
        success_list, _ = getSubjectFiles(rootPath)

        feature_list = ['unimodal_audioPower',
                        # 'unimodal_audioWristRMS',
                        'unimodal_kinVel',
                        'unimodal_ftForce',
                        'unimodal_ppsForce',
                        # 'unimodal_visionChange',
                        'unimodal_fabricForce',
                        'crossmodal_targetEEDist',
                        'crossmodal_targetEEAng',
                        'crossmodal_artagEEDist']
                        # 'crossmodal_artagEEAng']

        # t = time.time()
        rawDataDict, dataDict = dataUtil.loadData(success_list, isTrainingData=True, downSampleSize=self.downSampleSize, local_range=0.15, verbose=self.verbose)
        trainData, _ = dataMng.extractFeature(dataDict, feature_list, scale=1.0)
        trainData = np.array(trainData)

        if excludeStationary:
            # exclude stationary data
            thres = 0.025
            n,m,k = np.shape(trainData)
            diff_all_data = trainData[:,:,1:] - trainData[:,:,:-1]
            add_idx = []
            for i in xrange(n):
                std = np.max(np.max(diff_all_data[i], axis=1))
                if std >= thres:
                    add_idx.append(i)
            trainData  = trainData[add_idx]


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

    def get_sensitivity_gain(self, X):
        X_test = util.convert_sequence(X, emission=False)

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
            ## if logp >= 0.: ths = (logp*0.95 - self.ll_mu[min_index])/self.ll_std[min_index]
            ## else: ths = (logp*1.05 - self.ll_mu[min_index])/self.ll_std[min_index]
                        
            return ths, min_index

        elif self.check_method == 'global':
            ths = (logp - self.l_mu) / self.l_std
            return ths, 0

        elif self.check_method == 'change':
            if len(X[0])<3: return [], 0.0 #error

            X_test = util.convert_sequence([x[:-1] for x in X], emission=False)

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

            X_test = util.convert_sequence([x[:-1] for x in X], emission=False)

            try:
                final_ts_obj = ghmm.EmissionSequence(self.F, X_test[0].tolist())
                last_logp         = self.ml.loglikelihood(final_ts_obj)
            except:
                print "Too different input profile that cannot be expressed by emission matrix"
                return [], 0.0 # error
            
            ths_c = -(( abs(logp-last_logp) - self.l_mean_delta) / self.l_std_delta)

            ths_g = (logp - self.l_mu) / self.l_std
            
            return [ths_c, ths_g], 0
        

    def loglikelihood(self, X):
        X_test = util.convert_sequence(X, emission=False)
        X_test = np.squeeze(X_test)*self.scale
        final_ts_obj = ghmm.EmissionSequence(self.F, X_test.tolist())

        try:    
            p = self.ml.loglikelihood(final_ts_obj)
        except:
            print 'Likelihood error!!!!'
            sys.exit()

        return p


    def loglikelihoods(self, X, bPosterior=False, startIdx=1):
        X_test = util.convert_sequence(X, emission=False)
        X_test = np.squeeze(X_test)*self.scale

        l_likelihood = []
        l_posterior   = []        
        
        for i in xrange(startIdx, len(X[0])):
            final_ts_obj = ghmm.EmissionSequence(self.F, X_test[:i*self.nEmissionDim].tolist())

            try:
                logp = self.ml.loglikelihood(final_ts_obj)
                if bPosterior: post = np.array(self.ml.posterior(final_ts_obj))
            except:
                print "Unexpected profile!! GHMM cannot handle too low probability. Underflow?"
                ## return False, False # anomaly
                continue

            l_likelihood.append( logp )
            if bPosterior: l_posterior.append( post[i-1] )

        if bPosterior:
            return l_likelihood, l_posterior
        else:
            return l_likelihood
            

    def expLoglikelihood(self, X, ths_mult=None, smooth=False, bLoglikelihood=False):

        X_test = util.convert_sequence(X, emission=False)
        X_test = np.squeeze(X_test)*self.scale

        logp = None

        try:
            final_ts_obj = ghmm.EmissionSequence(self.F, X_test.tolist())
            if bLoglikelihood: logp = self.ml.loglikelihood(final_ts_obj)
        except:
            print "Too different input profile that cannot be expressed by emission matrix"
            return False, False # error

        try:
            post = np.array(self.ml.posterior(final_ts_obj))
        except:
            print "Unexpected profile!! GHMM cannot handle too low probability. Underflow?"
            return False, False # anomaly

        n = len(np.squeeze(X[0]))

        if smooth:
            # The version for IROS 2016.
            # The expected log-likelihood is estimated using weighted average.
            # We may not use this
            sum_w = 0.
            sum_l = 0.
            l_dist = []
            for i in xrange(self.nState):
                if self.cluster_type == 'state':
                    dist = np.linalg.norm(post[n-1] - self.l_statePosterior[i])
                else:
                    dist = util.symmetric_entropy(post[n-1], self.l_statePosterior[i])
                    ## weight = 1.0/entropy(post[n-1], self.l_statePosterior[i])

                l_dist.append(dist)
                    
                if dist < 1e-6: weight = 1e+6
                elif np.isinf(dist): weight = 1e-6
                else: weight = 1.0/dist
                
                sum_w += weight
                
                if (type(ths_mult) == list or type(ths_mult) == np.ndarray or type(ths_mult) == tuple) \
                  and len(ths_mult)>1:
                    sum_l += weight * (self.ll_mu[i] + ths_mult[i] * self.ll_std[i])
                else:
                    sum_l += weight * (self.ll_mu[i] + ths_mult * self.ll_std[i])                    

            if bLoglikelihood: return sum_l/sum_w, logp
            else: return sum_l/sum_w

        else:
            # The version of ICRA 2016
            # Find the best posterior distribution
            min_index, min_dist = self.findBestPosteriorDistribution(post[n-1])

            if (type(ths_mult) == list or type(ths_mult) == np.ndarray or type(ths_mult) == tuple) and \
              len(ths_mult)>1:
                if bLoglikelihood:
                    return self.ll_mu[min_index] + ths_mult[min_index]*self.ll_std[min_index], logp
                else:
                    return self.ll_mu[min_index] + ths_mult[min_index]*self.ll_std[min_index]
            else:
                if bLoglikelihood:
                    return self.ll_mu[min_index] + ths_mult*self.ll_std[min_index], logp
                else:
                    return self.ll_mu[min_index] + ths_mult*self.ll_std[min_index]


    '''
    Deprecated. Please use classifier.py
    '''
    def anomaly_check(self, X, ths_mult=None):
        print 'anomaly_check() is deprecated. Please use classifier.py instead.'
        if self.nEmissionDim == 1: X_test = np.array([X[0]])
        else: X_test = util.convert_sequence(X, emission=False)
        X_test *= self.scale

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

            X_test = util.convert_sequence([x[:-1] for x in X], emission=False)
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

            if self.verbose: print "Min index: ", min_index, " logp: ", logp, " ths_mult: ", ths_mult

            if (type(ths_mult) == list or type(ths_mult) == np.ndarray or type(ths_mult) == tuple) and len(ths_mult)>1:
                err = logp - (self.ll_mu[min_index] + ths_mult[min_index]*self.ll_std[min_index])
            else:
                err = logp - (self.ll_mu[min_index] + ths_mult*self.ll_std[min_index])

            if err < self.anomaly_offset: return True, err
            else: return False, err
            
        else:
            if err < 0.0: return True, err
            else: return False, err
            
    #-------------------------------------------------------------------

    def state_clustering(self, X):
        n, m = np.shape(X[0])

        x = np.arange(0., float(m))*(1./43.)
        state_mat  = np.zeros((self.nState, m*n))
        likelihood_mat = np.zeros((1, m*n))

        count = 0           
        for i in xrange(n):
            for j in xrange(1, m):
                X_test = util.convert_sequence([x[i:i+1,:j] for x in X], emission=False)

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

        if self.cluster_type == 'time' :
            for j in xrange(len(self.l_statePosterior)):
                dist = entropy(post, self.l_statePosterior[j])
                if min_dist > dist:
                    min_index = j
                    min_dist  = dist
                    
        elif self.cluster_type == 'kmean':
            min_index = self.km.predict(post)
            min_dist  = -1
            
        elif self.cluster_type == 'state':
            for j in xrange(len(self.l_statePosterior)):
                dist = np.linalg.norm( post - self.l_statePosterior[j] )
                ## dist = entropy(post, self.l_statePosterior[j])
                if min_dist > dist:
                    min_index = j
                    min_dist  = dist
            
            

        return min_index, min_dist

    def getPostLoglikelihoods(self, xData):

        X = [np.array(data)*self.scale for data in xData]
        X_test = util.convert_sequence(X) # Training input
        X_test = X_test.tolist()

        n, m = np.shape(X[0])

        # Estimate loglikelihoods and corresponding posteriors
        r = Parallel(n_jobs=-1)(delayed(computeLikelihood)(i, self.A, self.B, self.pi, self.F, X_test[i],
                                                           self.nEmissionDim, self.nState,
                                                           bPosterior=True, converted_X=True)
                                                           for i in xrange(n))
        _, ll_idx, ll_logp, ll_post = zip(*r)

        l_idx  = []
        l_logp = []
        l_post = []
        for i in xrange(len(ll_logp)):
            l_idx  += ll_idx[i]
            l_logp += ll_logp[i]
            l_post += ll_post[i]

        return l_idx, l_logp, l_post
    

#################### WARNING ###################################
# This file will be deprecated soon. Please, don't add anything. 
################################################################
    
####################################################################
# functions for paralell computation
####################################################################

def learn_time_clustering(i, ll_idx, ll_logp, ll_post, g_mu, g_sig, nState):

    l_likelihood_mean = 0.0
    l_likelihood_mean2 = 0.0
    l_statePosterior = np.zeros(nState)
    n = len(ll_idx)

    for j in xrange(n):

        g_post = np.zeros(nState)
        g_lhood = 0.0
        g_lhood2 = 0.0
        prop_sum = 0.0

        for idx, logp, post in zip(ll_idx[j], ll_logp[j], ll_post[j]):

            k_prop    = norm(loc=g_mu, scale=g_sig).pdf(idx)
            g_post   += post * k_prop
            g_lhood  += logp * k_prop
            g_lhood2 += logp * logp * k_prop

            prop_sum  += k_prop

        l_statePosterior += g_post / prop_sum / float(n)
        l_likelihood_mean += g_lhood / prop_sum / float(n)
        l_likelihood_mean2 += g_lhood2 / prop_sum / float(n)

    return i, l_statePosterior, l_likelihood_mean, np.sqrt(l_likelihood_mean2 - l_likelihood_mean**2)


def learn_state_clustering(i, ll_idx, ll_logp, ll_post, nState):

    l_statePosterior = np.zeros(nState)
    l_statePosterior[i] = 1.0

    l_likelihood_mean = 0.0
    l_likelihood_mean2 = 0.0
    n = len(ll_idx)

    for j in xrange(n):
        sum_weight = 0.0
        sum_logp   = 0.0
        sum_logp2  = 0.0

        for idx, logp, post in zip(ll_idx[j], ll_logp[j], ll_post[j]):

            dist = np.linalg.norm(post-l_statePosterior)
            if dist < 1e-6: weight = 1e+6
            else: weight = 1.0/dist
            ## weight = 1.0/entropy(post, l_statePosterior)
            ## if np.isnan(weight): continue
            
            sum_weight += weight
            sum_logp   += weight * logp
            sum_logp2  += weight * logp*logp

        ## if sum_weight < 1e-5: continue
        l_likelihood_mean  += sum_logp / sum_weight / float(n)
        l_likelihood_mean2 += sum_logp2 / sum_weight / float(n)

    return i, l_statePosterior, l_likelihood_mean, np.sqrt(l_likelihood_mean2 - l_likelihood_mean**2)


def computeLikelihood(idx, A, B, pi, F, X, nEmissionDim, nState, startIdx=1,
                      bPosterior=False, converted_X=False):
    '''
    This function will be deprecated. Please, use computeLikelihoods.
    '''

    if nEmissionDim >= 2:
        ml = ghmm.HMMFromMatrices(F, ghmm.MultivariateGaussianDistribution(F), A, B, pi)
    else:
        ml = ghmm.HMMFromMatrices(F, ghmm.GaussianDistribution(F), A, B, pi)
    
    if converted_X is False:
        X_test = util.convert_sequence(X, emission=False)
        X_test = np.squeeze(X_test)*ml.scale
        X_test = X_test.tolist()
    else:
        X_test = X

    l_idx        = []
    l_likelihood = []
    l_posterior  = []        

    for i in xrange(startIdx, len(X_test)/nEmissionDim):
        final_ts_obj = ghmm.EmissionSequence(F, X_test[:i*nEmissionDim])

        try:
            logp = ml.loglikelihood(final_ts_obj)
            if bPosterior: post = np.array(ml.posterior(final_ts_obj))
        except:
            print "Unexpected profile!! GHMM cannot handle too low probability. Underflow?"
            ## return False, False # anomaly
            continue

        l_idx.append( i )
        l_likelihood.append( logp )
        if bPosterior: l_posterior.append( post[i-1] )

    if bPosterior:
        return idx, l_idx, l_likelihood, l_posterior
    else:
        return idx, l_idx, l_likelihood


def computeLikelihoods(idx, A, B, pi, F, X, nEmissionDim, scale, nState, startIdx=2,
                       bPosterior=False, converted_X=False):
    '''
    Return
    '''

    if nEmissionDim >= 2:
        ml = ghmm.HMMFromMatrices(F, ghmm.MultivariateGaussianDistribution(F), A, B, pi)
    else:
        ml = ghmm.HMMFromMatrices(F, ghmm.GaussianDistribution(F), A, B, pi)

    X_test = util.convert_sequence(X, emission=False)
    X_test = np.squeeze(X_test)*scale

    l_idx        = []
    l_likelihood = []
    l_posterior  = []        

    for i in xrange(startIdx, len(X[0])):
        final_ts_obj = ghmm.EmissionSequence(F, X_test[:i*nEmissionDim].tolist())

        try:
            logp = ml.loglikelihood(final_ts_obj)
            if bPosterior: post = np.array(ml.posterior(final_ts_obj))
        except:
            print "Unexpected profile!! GHMM cannot handle too low probability. Underflow?"
            ## return False, False # anomaly
            continue

        l_idx.append( i )
        l_likelihood.append( logp )
        if bPosterior: l_posterior.append( post[i-1] )

    if bPosterior:
        return idx, l_idx, l_likelihood, l_posterior
    else:
        return idx, l_idx, l_likelihood

#################### WARNING ###################################
# This file will be deprecated soon. Please, don't add anything. 
################################################################

## def learn_time_clustering(i, n, m, A, B, pi, F, X_train, nEmissionDim, g_mu, g_sig, nState):
    ## if nEmissionDim >= 2:
    ##     ml = ghmm.HMMFromMatrices(F, ghmm.MultivariateGaussianDistribution(F), A, B, pi)
    ## else:
    ##     ml = ghmm.HMMFromMatrices(F, ghmm.GaussianDistribution(F), A, B, pi)
    
    ## l_likelihood_mean = 0.0
    ## l_likelihood_mean2 = 0.0
    ## l_statePosterior = np.zeros(nState)

    ## for j in xrange(n):    

    ##     g_post = np.zeros(nState)
    ##     g_lhood = 0.0
    ##     g_lhood2 = 0.0
    ##     prop_sum = 0.0

    ##     for k in xrange(1, m):
    ##         final_ts_obj = ghmm.EmissionSequence(F, X_train[j][:k*nEmissionDim])
    ##         logp = ml.loglikelihoods(final_ts_obj)[0]
    ##         # print 'Log likelihood:', logp
    ##         post = np.array(ml.posterior(final_ts_obj))

    ##         k_prop = norm(loc=g_mu, scale=g_sig).pdf(k)
    ##         g_post += post[k-1] * k_prop
    ##         g_lhood += logp * k_prop
    ##         g_lhood2 += logp * logp * k_prop

    ##         prop_sum  += k_prop

    ##     l_statePosterior += g_post / prop_sum / float(n)
    ##     l_likelihood_mean += g_lhood / prop_sum / float(n)
    ##     l_likelihood_mean2 += g_lhood2 / prop_sum / float(n)

    ## return i, l_statePosterior, l_likelihood_mean, np.sqrt(l_likelihood_mean2 - l_likelihood_mean**2)


## def computeLikelihood(F, k, data, g_mu, g_sig, nEmissionDim, A, B, pi):
##     if nEmissionDim >= 2:
##         hmm_ml = ghmm.HMMFromMatrices(F, ghmm.MultivariateGaussianDistribution(F), A, B, pi)
##     else:
##         hmm_ml = ghmm.HMMFromMatrices(F, ghmm.GaussianDistribution(F), A, B, pi)

##     final_ts_obj = ghmm.EmissionSequence(F, data)
##     logp = hmm_ml.loglikelihoods(final_ts_obj)[0]
##     post = np.array(hmm_ml.posterior(final_ts_obj))

##     k_prop = norm(loc=g_mu, scale=g_sig).pdf(k)
##     g_post = post[k-1] * k_prop
##     g_lhood = logp * k_prop
##     g_lhood2 = logp * logp * k_prop
##     prop_sum = k_prop

##     # print np.shape(g_post), np.shape(g_lhood), np.shape(g_lhood2), np.shape(prop_sum)

##     return g_post, g_lhood, g_lhood2, prop_sum

#################### WARNING ###################################
# This file will be deprecated soon. Please, don't add anything. 
################################################################

