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

#system
import numpy as np
import sys, os, copy

# Util
import hrl_lib.util as ut
import learning_util as util

import ghmm
from joblib import Parallel, delayed
from scipy.stats import multivariate_normal

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
        self.A = A
        self.B = B
        self.pi = pi
        return self.ml


    def fit(self, xData, A=None, B=None, pi=None, cov_mult=None,
            ml_pkl=None, use_pkl=False, cov_type='full'):
        '''
        Input :
        - xData: dimension x sample x length
        Issues:
        - If NaN is returned, the reason can be one of followings,
        -- lower cov
        -- small range of xData (you have to scale it up.)
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
            return True
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

                mus, cov = util.vectors_to_mean_cov(X, self.nState, self.nEmissionDim, cov_type=cov_type)
                ## print np.shape(mus), np.shape(cov)

                # cov: state x dim x dim
                for i in xrange(self.nEmissionDim):
                    for j in xrange(self.nEmissionDim):
                        cov[:, i, j] *= cov_mult[self.nEmissionDim*i + j]

                if self.verbose:
                    for i, mu in enumerate(mus):
                        print 'mu%i' % i, mu
                    ## print 'cov', cov

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

            if ml_pkl is not None: ut.save_pickle(param_dict, ml_pkl)
            return ret

    def partial_fit(self, xData, nTrain, scale, weight=8.0):
        '''
        data: dimension x sample x length
        '''
        A  = copy.copy(self.A)
        B  = copy.copy(self.B)
        pi = copy.copy(self.pi)

        new_B = copy.copy(self.B)

        t_features = []
        mus        = []
        covs       = []
        for i in xrange(self.nState):
            t_features.append( B[i][0] + [ float(i) / float(self.nState)*scale/2.0 ])
            mus.append( B[i][0] )
            covs.append( B[i][1] )
        t_features = np.array(t_features)
        mus        = np.array(mus)
        covs       = np.array(covs)

        # update b ------------------------------------------------------------
        # mu
        x_l = [[] for i in xrange(self.nState)]
        X   = np.swapaxes(xData, 0, 1) # sample x dim x length
        seq_len = len(X[0][0])
        for i in xrange(len(X)):
            sample = np.swapaxes(X[i], 0, 1) # length x dim

            idx_l = []
            for j in xrange(len(sample)):
                feature = np.array( sample[j].tolist() + [float(j)/float(len(sample))*scale/2.0 ] )

                min_dist = 10000
                min_idx  = 0
                for idx, t_feature in enumerate(t_features):
                    dist = np.linalg.norm(t_feature-feature)
                    if dist < min_dist:
                        min_dist = dist
                        min_idx  = idx

                x_l[min_idx].append(feature[:-1].tolist())

        for i in xrange(len(mus)):
            if len(x_l[i]) > 0:
                avg_x = np.mean(x_l[i], axis=0)
                new_B[i][0] = list( ( float(nTrain-1)*mus[i] + avg_x*weight ) / float(nTrain+(weight-1) ) ) # specialized for single input


        # Normalize the state prior and transition values.
        A_sum = np.sum(A, axis=1)
        for i in xrange(self.nState):
            A[i,:] /= A_sum[i]
        pi /= np.sum(pi)

        # Daehyung: What is the shape and type of input data?
        xData = [np.array(data) for data in xData]
        X_ptrain = util.convert_sequence(xData) # Training input
        X_ptrain = np.squeeze(X_ptrain)

        final_ts_obj = ghmm.EmissionSequence(self.F, X_ptrain.tolist())        
        (alpha, scale) = self.ml.forward(final_ts_obj)
        beta = self.ml.backward(final_ts_obj, scale)

        ## print np.shape(alpha), np.shape(beta), type(alpha), type(beta)

        est_A = np.zeros((self.nState, self.nState))
        new_A = np.zeros((self.nState, self.nState))
        for i in xrange(self.nState):
            for j in xrange(self.nState):

                temp1 = 0.0
                temp2 = 0.0
                for t in xrange(seq_len):
                    p = multivariate_normal.pdf( X[0][:,t], mean=mus[j], \
                                                 cov=np.reshape(covs[j], \
                                                                (self.nEmissionDim, self.nEmissionDim)))
                    temp1 += alpha[t-1][i] * A[i,j] * p * beta[t][j]
                    temp2 += alpha[t-1][i] * beta[t][j]

                if temp1 == 0.0 or temp2 == 0.0: est_A[i,j] = 0
                else: est_A[i,j] = temp1/temp2
                    
                new_A[i,j] = (float(nTrain-len(xData))*A[i,j] + est_A[i,j]*weight) / float(nTrain + (weight-1.0) )

        # Normalize the state prior and transition values.
        A_sum = np.sum(new_A, axis=1)
        for i in xrange(self.nState):
            new_A[i,:] /= A_sum[i]
        pi /= np.sum(pi)
            
        self.set_hmm_object(new_A, new_B, pi)
        return new_A, new_B, pi
        

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


    def loglikelihood(self, X, bPosterior=False):
        '''        
        shape?
        return: the likelihood of a sequence
        '''
        X_test = util.convert_sequence(X, emission=False)
        X_test = np.squeeze(X_test)
        final_ts_obj = ghmm.EmissionSequence(self.F, X_test.tolist())

        try:    
            logp = self.ml.loglikelihood(final_ts_obj)
            if bPosterior: post = np.array(self.ml.posterior(final_ts_obj))
        except:
            print 'Likelihood error!!!!'          
            if bPosterior: return None, None
            return None

        if bPosterior: return logp, post
        return logp


    def loglikelihoods(self, X, bPosterior=False, startIdx=1):
        '''
        X: dimension x sample x length
        return: the likelihoods over time (in single data)
        '''
        # sample x some length
        X_test = util.convert_sequence(X, emission=False)
        ## X_test = np.squeeze(X_test)

        ll_likelihoods = []
        ll_posteriors  = []        
        for i in xrange(len(X[0])):
            l_likelihood = []
            l_posterior  = []        

            for j in xrange(startIdx, len(X[0][i])):

                try:
                    final_ts_obj = ghmm.EmissionSequence(self.F,X_test[i,:j*self.nEmissionDim].tolist())
                except:
                    if self.verbose: print "failed to make sequence"
                    continue

                try:
                    logp = self.ml.loglikelihood(final_ts_obj)
                    if bPosterior: post = np.array(self.ml.posterior(final_ts_obj))
                except:
                    if self.verbose: 
                        print "Unexpected profile!! GHMM cannot handle too low probability. Underflow?"
                    return False, False # anomaly
                    #continue

                l_likelihood.append( logp )
                if bPosterior: l_posterior.append( post[j-1] )

            ll_likelihoods.append(l_likelihood)
            if bPosterior: ll_posteriors.append(l_posterior)

        if bPosterior:
            return ll_likelihoods, ll_posteriors
        else:
            return ll_likelihoods
            
            
    def getLoglikelihoods(self, xData, posterior=False, startIdx=1, n_jobs=-1):
        '''
        shape?
        '''
        X = [np.array(data) for data in xData]
        X_test = util.convert_sequence(X) # Training input
        X_test = X_test.tolist()

        n, _ = np.shape(X[0])

        # Estimate loglikelihoods and corresponding posteriors
        r = Parallel(n_jobs=n_jobs)(delayed(computeLikelihood)(i, self.A, self.B, self.pi, self.F, X_test[i], \
                                                           self.nEmissionDim, self.nState,\
                                                           startIdx=startIdx,\
                                                           bPosterior=posterior, converted_X=True)
                                                           for i in xrange(n))
        if posterior:
            _, ll_idx, ll_logp, ll_post = zip(*r)
            return ll_idx, ll_logp, ll_post            
        else:
            _, ll_idx, ll_logp = zip(*r)
            return ll_idx, ll_logp
                

        ## ll_idx  = []
        ## ll_logp = []
        ## ll_post = []
        ## for i in xrange(len(ll_logp)):
        ##     l_idx.append( ll_idx[i] )
        ##     l_logp.append( ll_logp[i] )
        ##     if posterior: ll_post.append( ll_post[i] )


    
    def score(self, X, y=None, n_jobs=1):
        '''
        X: dim x sample x length
        
        If y exists, y can contains two kinds of labels, [-1, 1]
        If an input is close to training data, its label should be 1.
        If not, its label should be -1.
        '''
        assert y[0]==1
        nPos = 0
        for i in xrange(len(y)):
            if y[i] == -1:
                nPos = i
                break
        posIdxList = [i for i in xrange(len(y)) if y[i]==1 ]
        negIdxList = [i for i in xrange(len(y)) if y[i]==-1]
        posX       = X[:,posIdxList,:]
        negX       = X[:,negIdxList,:]
                    
        if n_jobs==1:
            ll_pos_logp = self.loglikelihoods(posX) 
            ll_neg_logp = self.loglikelihoods(negX) 
        else:
            # sample,            
            _, ll_pos_logp = self.getLoglikelihoods(posX, startIdx=len(X[0][0]-1), n_jobs=n_jobs)
            _, ll_neg_logp = self.getLoglikelihoods(negX, startIdx=len(X[0][0]-1), n_jobs=n_jobs)

        v = np.linalg.norm( ll_neg_logp - np.mean(ll_pos_logp) )
        ## v = np.mean( np.std(ll_logp, axis=0) )
        ## v = 0.0
        ## if y is not None:
        ##     for i, l_logp in enumerate(ll_logp):                
        ##         v += np.sum( np.array(l_logp) * y[i] )
        ## else:
        ##     v += np.sum(ll_logp)

        if self.verbose: print np.shape(ll_pos_logp), np.shape(ll_neg_logp)," : score = ", v 

        return v
                
            

def getHMMinducedFeatures(ll_logp, ll_post, l_labels=None, c=1.0, add_delta_logp=True):
    '''
    Convert a list of logps and posterior distributions to HMM-induced feature vectors.
    It returns [logp, d_logp/(d_post+c), post].
    '''

    X = []
    Y = []
    for i in xrange(len(ll_logp)):
        l_X = []
        l_Y = []
        for j in xrange(1,len(ll_logp[i])):
            if add_delta_logp:                    
                if j == 0:
                    l_X.append( [ll_logp[i][j]] + [0] + ll_post[i][j].tolist() )
                else:
                    d_logp = ll_logp[i][j]-ll_logp[i][j-1]
                    d_post = util.symmetric_entropy(ll_post[i][j-1], ll_post[i][j])
                    l_X.append( [ll_logp[i][j]] + [ d_logp/(d_post+c) ] + \
                                ll_post[i][j].tolist() )
            else:
                l_X.append( [ll_logp[i][j]] + ll_post[i][j].tolist() )

            if l_labels is not None:
                if l_labels[i] > 0.0: l_Y.append(1)
                else: l_Y.append(-1)

            if np.isnan(ll_logp[i][j]):
                print "nan values in ", i, j
                return [],[]
                ## sys.exit()

        X.append(l_X)
        if l_labels is not None: Y.append(l_Y)
    
    return X, Y

    
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
    Input:
    - X: dimension x length
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


