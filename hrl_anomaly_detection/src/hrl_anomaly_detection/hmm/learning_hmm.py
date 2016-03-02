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

#system
import numpy as np
import sys, os, copy

# Util
import hrl_lib.util as ut
import learning_util as util

import ghmm
from joblib import Parallel, delayed

from hrl_anomaly_detection.hmm.learning_base import learning_base

os.system("taskset -p 0xff %d" % os.getpid())

class learning_hmm(learning_base):
    def __init__(self, nState=10, nEmissionDim=4, verbose=False):
        '''
        This class follows the policy of sklearn as much as possible.
        TODO: score function. NEED TO THINK WHAT WILL BE CRITERIA.
        '''
                 
        # parent class that provides sklearn related interfaces.
        learning_base.__init__(self)
                 
        self.ml = None
        self.verbose = verbose

        ## Tunable parameters
        self.nState         = nState # the number of hidden states
        self.nEmissionDim   = nEmissionDim
        
        ## Un-tunable parameters
        self.trans_type = 'left_right' # 'left_right' 'full'
        self.A  = None # transition matrix        
        self.B  = None # emission matrix
        self.pi = None # Initial probabilities per state

        # emission domain of this model        
        self.F = ghmm.Float()  


    def set_hmm_object(self, A, B, pi):

        self.ml = ghmm.HMMFromMatrices(self.F, ghmm.MultivariateGaussianDistribution(self.F), \
                                       A, B, pi)
        return self.ml


    def fit(self, xData, A=None, B=None, pi=None, cov_mult=None,
            ml_pkl=None, use_pkl=False):
        '''
        TODO: explanation of the shape and type of xData
        xData: dimension x sample x length  
        '''
        
        # Daehyung: What is the shape and type of input data?
        X = [np.array(data) for data in xData]
        
        param_dict = {}

        # Load pre-trained HMM without training
        if use_pkl and ml_pkl is not None and os.path.isfile(ml_pkl):
            if self.verbose: print "Load HMM parameters without train the hmm"
                
            param_dict = ut.load_pickle(ml_pkl)
            self.A  = param_dict['A']
            self.B  = param_dict['B']
            self.pi = param_dict['pi']                       
            self.ml = ghmm.HMMFromMatrices(self.F, ghmm.MultivariateGaussianDistribution(self.F), \
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
            if np.isnan(ret): return 'Failure'

            [self.A, self.B, self.pi] = self.ml.asMatrices()
            self.A = np.array(self.A)
            self.B = np.array(self.B)

            param_dict['A'] = self.A
            param_dict['B'] = self.B
            param_dict['pi'] = self.pi

                               

    ## def predict(self, X):
    ##     '''
    ##     ???????????????????
    ##     HMM is just a generative model. What will be prediction result?
    ##     Which code is using this fuction?
    ##     '''
    ##     X = np.squeeze(X)
    ##     X_test = X.tolist()

    ##     mu_l = np.zeros(self.nEmissionDim)
    ##     cov_l = np.zeros(self.nEmissionDim**2)

    ##     if self.verbose: print self.F
    ##     final_ts_obj = ghmm.EmissionSequence(self.F, X_test) # is it neccessary?

    ##     try:
    ##         # alpha: X_test length y # latent States at the moment t when state i is ended
    ##         # test_profile_length x number_of_hidden_state
    ##         (alpha, scale) = self.ml.forward(final_ts_obj)
    ##         alpha = np.array(alpha)
    ##     except:
    ##         if self.verbose: print "No alpha is available !!"
            
    ##     f = lambda x: round(x, 12)
    ##     for i in range(len(alpha)):
    ##         alpha[i] = map(f, alpha[i])
    ##     alpha[-1] = map(f, alpha[-1])
        
    ##     n = len(X_test)
    ##     pred_numerator = 0.0

    ##     for j in xrange(self.nState): # N+1
    ##         total = np.sum(self.A[:,j]*alpha[n/self.nEmissionDim-1,:]) #* scaling_factor
    ##         [mus, covars] = self.B[j]

    ##         ## print mu1, mu2, cov11, cov12, cov21, cov22, total
    ##         pred_numerator += total

    ##         for i in xrange(mu_l.size):
    ##             mu_l[i] += mus[i]*total
    ##         for i in xrange(cov_l.size):
    ##             cov_l[i] += covars[i] * (total**2)

    ##     return mu_l, cov_l


    def loglikelihood(self, X):
        X_test = util.convert_sequence(X, emission=False)
        X_test = np.squeeze(X_test)
        final_ts_obj = ghmm.EmissionSequence(self.F, X_test.tolist())

        try:    
            p = self.ml.loglikelihood(final_ts_obj)
        except:
            print 'Likelihood error!!!!'
            sys.exit()

        return p


    def loglikelihoods(self, X, bPosterior=False, startIdx=1):
        X_test = util.convert_sequence(X, emission=False)
        X_test = np.squeeze(X_test)

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
            
            
    def getPostLoglikelihoods(self, xData):

        X = [np.array(data) for data in xData]
        X_test = util.convert_sequence(X) # Training input
        X_test = X_test.tolist()

        n, m = np.shape(X[0])

        # Estimate loglikelihoods and corresponding posteriors
        r = Parallel(n_jobs=-1)(delayed(computeLikelihood)(i, self.A, self.B, self.pi, self.F, X_test[i], \
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
    

    
####################################################################
# functions for paralell computation
####################################################################

def computeLikelihood(idx, A, B, pi, F, X, nEmissionDim, nState, startIdx=1, \
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
        X_test = np.squeeze(X_test)
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


def computeLikelihoods(idx, A, B, pi, F, X, nEmissionDim, nState, startIdx=2, \
                       bPosterior=False, converted_X=False):
    '''
    Return
    '''

    if nEmissionDim >= 2:
        ml = ghmm.HMMFromMatrices(F, ghmm.MultivariateGaussianDistribution(F), A, B, pi)
    else:
        ml = ghmm.HMMFromMatrices(F, ghmm.GaussianDistribution(F), A, B, pi)

    X_test = util.convert_sequence(X, emission=False)
    X_test = np.squeeze(X_test)

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


