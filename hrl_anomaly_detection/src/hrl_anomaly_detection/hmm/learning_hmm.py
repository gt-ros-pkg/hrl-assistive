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
import random

# Util
import hrl_lib.util as ut
import learning_util as util

import ghmm
from joblib import Parallel, delayed
from scipy.stats import multivariate_normal

from hrl_anomaly_detection.hmm.learning_base import learning_base

os.system("taskset -p 0xff %d" % os.getpid())
random.seed(3334)

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


    def get_hmm_object(self):
        
        [A, B, pi] = self.ml.asMatrices()
        try:
            [out_a_num, vec_num, mat_num, u_denom] = self.ml.getBaumWelchParams()
        except:
            out_a_num=vec_num=mat_num=u_denom=None

        return [A, B, pi, out_a_num, vec_num, mat_num, u_denom]

    def set_hmm_object(self, A, B, pi, out_a_num=None, vec_num=None, mat_num=None, u_denom=None):

        if self.nEmissionDim == 1:
            self.ml = ghmm.HMMFromMatrices(self.F, ghmm.GaussianDistribution(self.F), \
                                           A, B, pi)
        else:        
            self.ml = ghmm.HMMFromMatrices(self.F, ghmm.MultivariateGaussianDistribution(self.F), \
                                           A, B, pi)
        self.A = A
        self.B = B
        self.pi = pi

        try:
            self.ml.setBaumWelchParams(out_a_num, vec_num, mat_num, u_denom)
        except:
            print "Install new ghmm!!"
            
        return self.ml


    def fit(self, xData, A=None, B=None, pi=None, cov_mult=None,
            ml_pkl=None, use_pkl=False, cov_type='full', fixed_trans=0,\
            shuffle=False):
        '''
        Input :
        - xData: dimension x sample x length
        Issues:
        - If NaN is returned, the reason can be one of followings,
        -- lower cov
        -- small range of xData (you have to scale it up.)
        '''
        
        # Daehyung: What is the shape and type of input data?
        if shuffle:
            X = xData
            X = np.swapaxes(X,0,1)
            id_list = range(len(X))
            random.shuffle(id_list)
            X = np.array(X)[id_list]
            X = np.swapaxes(X,0,1)
        else:
            X = [np.array(data) for data in xData]
        nData = len(xData[0])
        
        param_dict = {}

        # Load pre-trained HMM without training
        if use_pkl and ml_pkl is not None and os.path.isfile(ml_pkl):
            if self.verbose: print "Load HMM parameters without train the hmm"
                
            param_dict = ut.load_pickle(ml_pkl)
            self.A  = param_dict['A']
            self.B  = param_dict['B']
            self.pi = param_dict['pi']
            if self.nEmissionDim == 1:
                self.ml = ghmm.HMMFromMatrices(self.F, ghmm.GaussianDistribution(self.F), \
                                               self.A, self.B, self.pi)
            else:
                self.ml = ghmm.HMMFromMatrices(self.F, ghmm.MultivariateGaussianDistribution(self.F), \
                                               self.A, self.B, self.pi)

            out_a_num = param_dict.get('out_a_num', None)
            vec_num   = param_dict.get('vec_num', None)
            mat_num   = param_dict.get('mat_num', None)
            u_denom   = param_dict.get('u_denom', None)
            if out_a_num is not None:
                self.ml.setBaumWelchParams(out_a_num, vec_num, mat_num, u_denom)
                                           
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
                    if self.nEmissionDim>1:
                        B[i] = [[mu[i] for mu in mus]]
                        B[i].append(cov[i].flatten().tolist())
                    else:
                        B[i] = [np.squeeze(mus[0][i]), float(cov[i])]
            if pi is None:
                # pi - initial probabilities per state 
                ## pi = [1.0/float(self.nState)] * self.nState
                pi = [0.0] * self.nState
                pi[0] = 1.0

            # print 'Generating HMM'
            # HMM model object
            if self.nEmissionDim == 1:
                self.ml = ghmm.HMMFromMatrices(self.F, ghmm.GaussianDistribution(self.F), \
                                               A, B, pi)
            else:
                self.ml = ghmm.HMMFromMatrices(self.F, ghmm.MultivariateGaussianDistribution(self.F), \
                                               A, B, pi)
            if cov_type == 'diag': self.ml.setDiagonalCovariance(1)
                
            # print 'Creating Training Data'            
            X_train = util.convert_sequence(X) # Training input
            X_train = X_train.tolist()
            if self.verbose: print "training data size: ", np.shape(X_train)

            if self.verbose: print 'Run Baum Welch method with (samples, length)', np.shape(X_train)
            final_seq = ghmm.SequenceSet(self.F, X_train)
            ## ret = self.ml.baumWelch(final_seq, loglikelihoodCutoff=2.0)
            ret = self.ml.baumWelch(final_seq, 10000, fixedTrans=fixed_trans)
            if np.isnan(ret):
                print 'Baum Welch return:', ret
                return 'Failure'
            print 'Baum Welch return:', ret/float(nData)

            [self.A, self.B, self.pi] = self.ml.asMatrices()
            self.A = np.array(self.A)
            self.B = np.array(self.B)

            param_dict['A'] = self.A
            param_dict['B'] = self.B
            param_dict['pi'] = self.pi

            try:
                [out_a_num, vec_num, mat_num, u_denom] = self.ml.getBaumWelchParams()            
                param_dict['out_a_num'] = out_a_num
                param_dict['vec_num']   = vec_num
                param_dict['mat_num']   = mat_num
                param_dict['u_denom']   = u_denom
            except:
                print "Install new ghmm!!"

            if ml_pkl is not None: ut.save_pickle(param_dict, ml_pkl)
            return ret/float(nData)


    def partial_fit(self, xData, learningRate=0.2, nrSteps=100):
        ''' Online update of HMM using online Baum-Welch algorithm
        '''
        
        X     = [np.array(data) for data in xData]
        nData = len(X[0])

        # print 'Creating Training Data'            
        X_train = util.convert_sequence(X) # Training input
        X_train = X_train.tolist()

        if self.verbose: print 'Run Baum Welch method with (samples, length)', np.shape(X_train)
        if learningRate < 1e-5: learningRate = 1e-5

        final_seq = ghmm.SequenceSet(self.F, X_train)
        ret = self.ml.baumWelch(final_seq, nrSteps=nrSteps, learningRate=learningRate)

        if np.isnan(ret):
            print 'Baum Welch return:', ret
            return 'Failure'
        print 'Baum Welch return:', ret/float(nData)

        [self.A, self.B, self.pi] = self.ml.asMatrices()
        self.A = np.array(self.A)
        self.B = np.array(self.B)
        
        return ret

        
    def predict(self, X):
        '''
        Input
        @ X: dimension x sample x length #samples x known steps
        Output
        @ observation distribution: mu, var #samples x 1 [list]        
        '''

        # sample x some length
        X_test = util.convert_sequence(X, emission=False)

        mu_l  = []     
        cov_l = []      
            
        for i in xrange(len(X_test)):

            # Past profile
            final_ts_obj = ghmm.EmissionSequence(self.F,X_test[i].tolist()) 

            try:
                # alpha: X_test length y #latent States at the moment t when state i is ended
                #        test_profile_length x number_of_hidden_state
                (alpha,scale) = self.ml.forward(final_ts_obj)
                alpha         = np.array(alpha)
                scale         = np.array(scale)
            except:
                print "No alpha is available !!"
                sys.exit()
                ## continue

            f = lambda x: round(x,12)
            for j in range(len(alpha)):
                alpha[j] = map(f, alpha[j])
            alpha[-1] = map(f, alpha[-1])

            n     = len(X_test[i])
            t_mu  = np.zeros(self.nEmissionDim)
            t_cov = np.zeros(self.nEmissionDim*self.nEmissionDim)
            t_sum = 0.0
            for j in xrange(self.nState): # N+1

                total = np.sum(self.A[:,j]*alpha[n/self.nEmissionDim-1,:]) #* scaling_factor
                [mu, cov] = self.B[j]

                t_mu  += np.array(mu)*total
                t_cov += np.array(cov)*(total**2)
                t_sum += total

            mu_l.append( t_mu.tolist() )
            cov_l.append( t_cov.tolist() )

        return mu_l, cov_l


    def predict_from_single_seq(self, x, ref_num):
        '''
        Input
        @ x: length #samples x known steps
        Output
        @ observation distribution: nDimension
        '''
        
        # new emission for partial sequence
        B = []
        for i in xrange(self.nState):
            B.append( [ self.B[i][0][ref_num], self.B[i][1][ref_num*self.nEmissionDim+ref_num] ] )

        ml = ghmm.HMMFromMatrices(self.F, ghmm.GaussianDistribution(self.F), \
                                  self.A, B, self.pi)

        if type(x) is not list: x = x.tolist()            
        final_ts_obj = ghmm.EmissionSequence(self.F, x)
        
        try:
            (alpha, scale) = ml.forward(final_ts_obj)
        except:
            print "No alpha is available !!"
            sys.exit()

        x_pred = []
        for i in xrange(self.nEmissionDim):
            if i == ref_num:
                x_pred.append(x[-1])
            else:
                src_cov_idx = ref_num*self.nEmissionDim+ref_num
                tgt_cov_idx = ref_num*self.nEmissionDim+i

                t_o = 0.0
                for j in xrange(self.nState):
                    m_j = self.B[j][0][i] + \
                      self.B[j][1][tgt_cov_idx]/self.B[j][1][src_cov_idx]*\
                      (x[-1]-self.B[j][0][ref_num])
                    t_o += alpha[-1][j]*m_j
                x_pred.append(t_o)

        return x_pred


    def conditional_prob(self, x):
        '''
        Input
        @ x: dim x length
        Output
        @ A list of conditional probabilities P(x_t|x_s,lambda)

        Only single sample works
        '''
        from scipy.stats import norm, entropy

        # logp from all features
        X_test = util.convert_sequence2(x, emission=False)
        X_test = np.squeeze(X_test)
        final_ts_obj = ghmm.EmissionSequence(self.F, X_test.tolist())
        logp_all = self.ml.loglikelihood(final_ts_obj)

        # feature-wise conditional probability
        cond_prob = []
        for i in xrange(self.nEmissionDim): # per feature

            B = copy.copy(self.B)
            for j in xrange(self.nState):
                B[j][0] = [b for idx, b in enumerate(B[j][0]) if idx != i ]
                B_arr = copy.copy(B[j][1])
                B_arr = np.array(B_arr).reshape( (self.nEmissionDim, self.nEmissionDim) )
                B_arr = np.delete(B_arr, (i), axis=0)
                B_arr = np.delete(B_arr, (i), axis=1)
                B[j][1] = B_arr.flatten().tolist()
            ml_src = ghmm.HMMFromMatrices(self.F, ghmm.MultivariateGaussianDistribution(self.F), \
                                          self.A, B, self.pi)

            # logp from remains
            X_test = util.convert_sequence2([ x[j] for j in xrange(len(x)) if j != i ], \
                                            emission=False)
            X_test = np.squeeze(X_test)
            final_ts_obj = ghmm.EmissionSequence(self.F, X_test.tolist())
            logp_src = ml_src.loglikelihood(final_ts_obj)

            cond_prob.append(logp_all - logp_src)

            if np.isnan(cond_prob[-1]):
                print "NaN in conditional probabilities: ", np.shape(x)
                return None
        
        return np.array(cond_prob)


    def conditional_prob2(self, x):
        '''
        Input
        @ x: dim x length
        Output
        @ A list of conditional probabilities P(x_t|lambda)

        Only single sample works
        '''
        from scipy.stats import norm, entropy

        # feature-wise conditional probability
        cond_prob = []
        for i in xrange(self.nEmissionDim): # per feature

            B = [0] * self.nState
            for j in xrange(self.nState):
                B[j] = [ self.B[j][0][i], self.B[j][1][i*self.nEmissionDim+i] ]
                
            ml_src = ghmm.HMMFromMatrices(self.F, ghmm.GaussianDistribution(self.F), \
                                          self.A, B, self.pi)

            X_test = util.convert_sequence2(x[[i]], emission=False)
            X_test = np.squeeze(X_test)
            final_ts_obj = ghmm.EmissionSequence(self.F, X_test.tolist())
            logp = ml_src.loglikelihood(final_ts_obj)

            cond_prob.append(logp)

        ## # all
        ## X_test = util.convert_sequence2(x, emission=False)
        ## X_test = np.squeeze(X_test)
        ## final_ts_obj = ghmm.EmissionSequence(self.F, X_test.tolist())
        ## cond_prob.append( self.ml.loglikelihood(final_ts_obj) )
        
        # min-max normalization
        cond_prob = np.array(cond_prob)
            
        return cond_prob
    

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


    def loglikelihoods(self, X, bPosterior=False, bIdx=False, startIdx=1):
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
                    print "failed to make sequence"
                    continue

                try:
                    logp = self.ml.loglikelihood(final_ts_obj)
                    if bPosterior: post = np.array(self.ml.posterior(final_ts_obj))
                except:
                    print "Unexpected profile!! GHMM cannot handle too low probability. Underflow?"
                    continue
                    ## return False, False # anomaly
                    #continue

                l_likelihood.append( logp )
                if bPosterior: l_posterior.append( post[j-1] )

            ll_likelihoods.append(l_likelihood)
            if bPosterior: ll_posteriors.append(l_posterior)
        
        if bIdx:
            ll_idx = []
            for ii in xrange(len(X[0])):
                l_idx = []
                for jj in xrange(startIdx, len(X[0][ii])):
                    l_idx.append( jj )
                ll_idx.append(l_idx)
            
            if bPosterior: return ll_likelihoods, ll_posteriors, ll_idx
            else:          return ll_likelihoods, ll_idx
        else:
            if bPosterior: return ll_likelihoods, ll_posteriors
            else:          return ll_likelihoods
            
            
            
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
    It returns [logp, last_post, post].
    '''
    if type(ll_logp) is tuple: ll_logp = list(ll_logp)
    if type(ll_post) is tuple: ll_post = list(ll_post)

    X = []
    Y = []
    for i in xrange(len(ll_logp)):

        if add_delta_logp:                            
            l_X = []
            for j in xrange(len(ll_logp[i])):
                if j == 0:
                    ## l_X.append( [ll_logp[i][j]] + [0] + ll_post[i][j].tolist() )
                    l_X.append( [ll_logp[i][j]] + list(ll_post[i][j]) + list(ll_post[i][j]) )
                else:
                    ## d_logp = ll_logp[i][j]-ll_logp[i][j-1]
                    ## d_post = util.symmetric_entropy(ll_post[i][j-1], ll_post[i][j])
                    ## l_X.append( [ll_logp[i][j]] + [ d_logp/(d_post+c) ] + \
                    ##             ll_post[i][j].tolist() )
                    l_X.append( [ll_logp[i][j]] + list(ll_post[i][j-1]) + \
                                list(ll_post[i][j]) )
        else:
            l_X = np.hstack([ np.array(ll_logp[i]).reshape(len(ll_logp[i]),1), np.array(ll_post[i]) ]).tolist()

        if l_labels is not None:
            if l_labels[i] > 0.0: l_Y = [1 for j in xrange(len(l_X))]
            else: l_Y = [-1 for j in xrange(len(l_X))]

        if np.nan in ll_logp[i]:
            print "nan values in ", i
            return [],[]
            ## sys.exit()

        X.append(l_X)
        if l_labels is not None: Y.append(l_Y)
    
    return X, Y


def getHMMinducedFlattenFeatures(ll_logp, ll_post, ll_idx, l_labels=None, c=1.0, add_delta_logp=True,\
                                 remove_fp=False, remove_outlier=False):
    from hrl_anomaly_detection import data_manager as dm

    if len(ll_logp)>2 and remove_outlier:
        ll_logp, ll_post, ll_idx, l_labels = removeLikelihoodOutliers(ll_logp, ll_post, ll_idx, l_labels)
            

    ll_X, ll_Y = getHMMinducedFeatures(ll_logp, ll_post, l_labels, c=c, add_delta_logp=add_delta_logp)
    if ll_X == []: return [],[],[]
    
    X_flat, Y_flat, idx_flat = dm.flattenSample(ll_X, ll_Y, ll_idx, remove_fp=remove_fp)
    return X_flat, Y_flat, idx_flat


def getHMMinducedFeaturesFromRawFeatures(ml, normalTrainData, abnormalTrainData=None, startIdx=4, \
                                         add_logp_d=False,\
                                         cov_type='full'):

    if abnormalTrainData is not None:
        testDataX = np.vstack([ np.swapaxes(normalTrainData,0,1), np.swapaxes(abnormalTrainData,0,1) ])
        testDataX = np.swapaxes(testDataX, 0,1)
        testDataY = np.hstack([ -np.ones(len(normalTrainData[0])), \
                                np.ones(len(abnormalTrainData[0])) ])
    else:
        testDataX = normalTrainData
        testDataY = -np.ones(len(testDataX[0]))
        
    return getHMMinducedFeaturesFromRawCombinedFeatures(ml, testDataX, testDataY, startIdx, \
                                                        add_logp_d=add_logp_d, cov_type=cov_type)


def getHMMinducedFeaturesFromRawCombinedFeatures(ml, dataX, dataY, startIdx, add_logp_d=False, cov_type='full',\
                                                 nSubSample=None, nMaxData=100000, rnd_sample=True, n_jobs=-1):


    r = Parallel(n_jobs=n_jobs)(delayed(computeLikelihoods)(i, ml.A, ml.B, ml.pi, ml.F, \
                                                        [ dataX[j][i] for j in \
                                                          xrange(ml.nEmissionDim) ], \
                                                          ml.nEmissionDim, ml.nState,\
                                                          startIdx=startIdx, \
                                                          bPosterior=True, cov_type=cov_type)
                                                          for i in xrange(len(dataX[0])))
    _, ll_classifier_train_idx, ll_logp, ll_post = zip(*r)    

    ll_classifier_train_X, ll_classifier_train_Y = \
      getHMMinducedFeatures(ll_logp, ll_post, dataY, c=1.0, add_delta_logp=add_logp_d)

    if nSubSample is not None:
        from hrl_anomaly_detection import data_manager as dm
        ll_classifier_train_X, ll_classifier_train_Y, ll_classifier_train_idx =\
          dm.subsampleData(ll_classifier_train_X, ll_classifier_train_Y, ll_classifier_train_idx,\
                           nSubSample=nSubSample, nMaxData=nMaxData, rnd_sample=rnd_sample)

    return ll_classifier_train_X, ll_classifier_train_Y, ll_classifier_train_idx


def removeLikelihoodOutliers(ll_logp, ll_post, ll_idx, l_labels=None):        
    ''' remove outliers from normal data (upper 5% and lower 5%)
    '''
    if type(ll_logp) is tuple: ll_logp = list(ll_logp)
    if type(ll_post) is tuple: ll_post = list(ll_post)
    if type(ll_idx) is tuple: ll_idx = list(ll_idx)
    
    # Check only last likelihoods
    logp_lst = []
    idx_lst  = []
    for i in xrange(len(ll_logp)):
        if l_labels is None or l_labels[i] < 0:
            logp_lst.append(ll_logp[i][-1])
            idx_lst.append(i)

    logp_lst, idx_lst = zip(*sorted(zip(logp_lst, idx_lst)))
    nOutlier = int(0.05*len(ll_logp))
    if nOutlier < 1: nOutlier = 1

    lower_lst = idx_lst[:nOutlier]
    upper_lst = idx_lst[-nOutlier:]
    indices = lower_lst + upper_lst
    indices = sorted(list(indices), reverse=True)
    for i in indices:
        del ll_logp[i]
        del ll_post[i]
        del ll_idx[i]
        if l_labels is None: continue
        if type(l_labels) is not list: l_labels = l_labels.tolist()
        del l_labels[i]

    return ll_logp, ll_post, ll_idx, l_labels


def getEntropyFeaturesFromHMMInducedFeatures(ll_X, ll_Y, ll_idx, nPosteriors):
    from scipy.stats import norm, entropy

    direc_delta = np.zeros(nPosteriors)
    for i in xrange(len(direc_delta)):
        direc_delta[i] = np.exp(-float(i))
    direc_delta /= np.sum(direc_delta)

    lll_X   = []
    for k in xrange(len(ll_X)):

        # length x features
        ll_logp = np.array(ll_X[k])[:,0]
        ll_post = np.array(ll_X[k])[:,-nPosteriors:]

        new_X = []
        max_states = np.argmax(ll_post, axis=1)

        
        for i in xrange(len(ll_logp)):
            state = max_states[i]
            selfInfo = util.shannon_entropy(ll_post[i]+1e-6)
            ## selfInfo = util.symmetric_entropy(direc_delta+1e-6, ll_post[i]+1e-6)
            ## if selfInfo < 1e-10: selfInfo = 1e-10
            ## selfInfo = np.log(selfInfo)
                
            new_X.append([ll_logp[i], float(state), selfInfo ])

        lll_X.append(new_X)
        
    return lll_X, ll_Y, ll_idx

####################################################################
# functions for paralell computation
####################################################################

def computeLikelihood(idx, A, B, pi, F, X, nEmissionDim, nState, startIdx=1, \
                      bPosterior=False, converted_X=False, cov_type='full'):
    '''
    This function will be deprecated. Please, use computeLikelihoods.
    '''

    if nEmissionDim >= 2:
        ml = ghmm.HMMFromMatrices(F, ghmm.MultivariateGaussianDistribution(F), A, B, pi)
        if cov_type == 'diag' or cov_type.find('diag')>=0: ml.setDiagonalCovariance(1)        
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

            l_idx.append( i )
            l_likelihood.append( -100000000 )
            if bPosterior: 
                if len(l_posterior) == 0: l_posterior.append(list(pi))
                else: l_posterior.append( l_posterior[-1] )            
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
                       bPosterior=False, converted_X=False, cov_type='full'):
    '''
    Input:
    - X: dimension x length
    '''

    if nEmissionDim >= 2:
        ml = ghmm.HMMFromMatrices(F, ghmm.MultivariateGaussianDistribution(F), A, B, pi)
        if cov_type == 'diag': ml.setDiagonalCovariance(1)
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
            l_likelihood.append( logp )
            if bPosterior: l_posterior.append( post[i-1] )                
        except:
            print "Unexpected profile!! GHMM cannot handle too low probability. Underflow?"
            ## return False, False # anomaly            
            ## continue
            # we keep the state as the previous one
            l_likelihood.append( -1000000000000 )
            if bPosterior:
                if len(l_posterior) == 0: l_posterior.append(list(pi))
                else: l_posterior.append( l_posterior[-1] )

        l_idx.append( i )
            

    if bPosterior:
        return idx, l_idx, l_likelihood, l_posterior
    else:
        return idx, l_idx, l_likelihood


## ####################################################################
## # functions for data generation
## ####################################################################

## def computeHMMfeatures(task_name, processed_data_path, param_dict, data_renew=False, verbose=False):
##     ## Parameters
##     # data
##     data_dict  = param_dict['data_param']
##     data_renew = data_dict['renew']
##     # AE
##     AE_dict     = param_dict['AE']
##     # HMM
##     HMM_dict   = param_dict['HMM']
##     nState     = HMM_dict['nState']
##     cov        = HMM_dict['cov']
##     add_logp_d = HMM_dict.get('add_logp_d', False)
##     # SVM
##     SVM_dict   = param_dict['SVM']

##     # ROC
##     ROC_dict = param_dict['ROC']

##     crossVal_pkl = os.path.join(processed_data_path, 'cv_'+task_name+'.pkl')
##     if os.path.isfile(crossVal_pkl):
##         d = ut.load_pickle(crossVal_pkl)
##         kFold_list  = d['kFoldList']
##     else:
##         print "No cv data"
##         sys.exit()

##     #-----------------------------------------------------------------------------------------
##     # parameters
##     startIdx    = 4
##     method_list = ROC_dict['methods'] 
##     nPoints     = ROC_dict['nPoints']

##     successData = d['successData']
##     failureData = d['failureData']
##     param_dict  = d['param_dict']
##     aeSuccessData = d.get('aeSuccessData', None)
##     aeFailureData = d.get('aeFailureData', None)
##     if 'timeList' in param_dict.keys():
##         timeList    = param_dict['timeList'][startIdx:]
##     else: timeList = None

##     #-----------------------------------------------------------------------------------------
##     # Training HMM, and getting classifier training and testing data
##     for idx, (normalTrainIdx, abnormalTrainIdx, normalTestIdx, abnormalTestIdx) \
##       in enumerate(kFold_list):

##         if verbose: print idx, " : training hmm and getting classifier training and testing data"

##         if AE_dict['switch'] and AE_dict['add_option'] is not None:
##             tag = ''
##             for ft in AE_dict['add_option']:
##                 tag += ft[:2]
##             modeling_pkl = os.path.join(processed_data_path, 'hmm_'+task_name+'_raw_'+tag+'_'+str(idx)+'.pkl')
##         elif AE_dict['switch'] and AE_dict['add_option'] is None:
##             modeling_pkl = os.path.join(processed_data_path, 'hmm_'+task_name+'_raw_'+str(idx)+'.pkl')
##         else:
##             modeling_pkl = os.path.join(processed_data_path, 'hmm_'+task_name+'_'+str(idx)+'.pkl')

##         if not (os.path.isfile(modeling_pkl) is False or HMM_dict['renew'] or data_renew): continue

##         if AE_dict['switch']:
##             if verbose: print "Start "+str(idx)+"/"+str(len(kFold_list))+"th iteration"

##             AE_proc_data = os.path.join(processed_data_path, 'ae_processed_data_'+str(idx)+'.pkl')

##             # From dim x sample x length
##             # To reduced_dim x sample x length
##             d = dm.getAEdataSet(idx, aeSuccessData, aeFailureData, \
##                                 successData, failureData, param_dict,\
##                                 normalTrainIdx, abnormalTrainIdx, normalTestIdx, abnormalTestIdx,\
##                                 AE_dict['time_window'], AE_dict['nAugment'], \
##                                 AE_proc_data, \
##                                 # data param
##                                 processed_data_path, \
##                                 # AE param
##                                 layer_sizes=AE_dict['layer_sizes'], learning_rate=AE_dict['learning_rate'], \
##                                 learning_rate_decay=AE_dict['learning_rate_decay'], \
##                                 momentum=AE_dict['momentum'], dampening=AE_dict['dampening'], \
##                                 lambda_reg=AE_dict['lambda_reg'], \
##                                 max_iteration=AE_dict['max_iteration'], min_loss=AE_dict['min_loss'], \
##                                 cuda=False, \
##                                 filtering=AE_dict['filter'], filteringDim=AE_dict['filterDim'],\
##                                 verbose=False)

##             if AE_dict['filter']:
##                 # NOTE: pooling dimension should vary on each auto encoder.
##                 # Filtering using variances
##                 normalTrainData   = d['normTrainDataFiltered']
##                 abnormalTrainData = d['abnormTrainDataFiltered']
##                 normalTestData    = d['normTestDataFiltered']
##                 abnormalTestData  = d['abnormTestDataFiltered']
##             else:
##                 normalTrainData   = d['normTrainData']
##                 abnormalTrainData = d['abnormTrainData']
##                 normalTestData    = d['normTestData']
##                 abnormalTestData  = d['abnormTestData']
##         else:
##             # dim x sample x length
##             normalTrainData   = successData[:, normalTrainIdx, :] 
##             abnormalTrainData = failureData[:, abnormalTrainIdx, :] 
##             normalTestData    = successData[:, normalTestIdx, :] 
##             abnormalTestData  = failureData[:, abnormalTestIdx, :] 


##         if AE_dict['switch'] and AE_dict['add_option'] is not None:
##             print "add hand-crafted features.."
##             newHandSuccTrData = handSuccTrData = d['handNormTrainData']
##             newHandFailTrData = handFailTrData = d['handAbnormTrainData']
##             handSuccTeData = d['handNormTestData']
##             handFailTeData = d['handAbnormTestData']

##             normalTrainData   = combineData( normalTrainData, newHandSuccTrData,\
##                                              AE_dict['add_option'], d['handFeatureNames'], \
##                                              add_noise_features=AE_dict['add_noise_option'] )
##             abnormalTrainData = combineData( abnormalTrainData, newHandFailTrData,\
##                                              AE_dict['add_option'], d['handFeatureNames'])
##             normalTestData   = combineData( normalTestData, handSuccTeData,\
##                                             AE_dict['add_option'], d['handFeatureNames'])
##             abnormalTestData  = combineData( abnormalTestData, handFailTeData,\
##                                              AE_dict['add_option'], d['handFeatureNames'])

##             ## # reduce dimension by pooling
##             ## pooling_param_dict  = {'dim': AE_dict['filterDim']} # only for AE        
##             ## normalTrainData, pooling_param_dict = dm.variancePooling(normalTrainData, \
##             ##                                                          pooling_param_dict)
##             ## abnormalTrainData, _ = dm.variancePooling(abnormalTrainData, pooling_param_dict)
##             ## normalTestData, _    = dm.variancePooling(normalTestData, pooling_param_dict)
##             ## abnormalTestData, _  = dm.variancePooling(abnormalTestData, pooling_param_dict)

##         ## # add noise
##         ##     normalTrainData += np.random.normal(0.0, 0.03, np.shape(normalTrainData) ) 

##         # scaling
##         if verbose: print "scaling data"
##         normalTrainData   *= HMM_dict['scale']
##         abnormalTrainData *= HMM_dict['scale']
##         normalTestData    *= HMM_dict['scale']
##         abnormalTestData  *= HMM_dict['scale']

##         # training hmm
##         if verbose: print "start to fit hmm"
##         nEmissionDim = len(normalTrainData)
##         cov_mult     = [cov]*(nEmissionDim**2)
##         nLength      = len(normalTrainData[0][0]) - startIdx

##         ml  = learning_hmm(nState, nEmissionDim, verbose=verbose) 
##         if data_dict['handFeatures_noise']:
##             ret = ml.fit(normalTrainData+\
##                          np.random.normal(0.0, 0.03, np.shape(normalTrainData) )*HMM_dict['scale'], \
##                          cov_mult=cov_mult, use_pkl=False)
##         else:
##             ret = ml.fit(normalTrainData, cov_mult=cov_mult, use_pkl=False)

##         if ret == 'Failure' or np.isnan(ret): sys.exit()

##         #-----------------------------------------------------------------------------------------
##         # Classifier training data
##         #-----------------------------------------------------------------------------------------
##         testDataX = []
##         testDataY = []
##         for i in xrange(nEmissionDim):
##             temp = np.vstack([normalTrainData[i], abnormalTrainData[i]])
##             testDataX.append( temp )

##         testDataY = np.hstack([ -np.ones(len(normalTrainData[0])), \
##                                 np.ones(len(abnormalTrainData[0])) ])

##         r = Parallel(n_jobs=-1)(delayed(computeLikelihoods)(i, ml.A, ml.B, ml.pi, ml.F, \
##                                                                 [ testDataX[j][i] for j in xrange(nEmissionDim) ], \
##                                                                 ml.nEmissionDim, ml.nState,\
##                                                                 startIdx=startIdx, \
##                                                                 bPosterior=True)
##                                                                 for i in xrange(len(testDataX[0])))
##         _, ll_classifier_train_idx, ll_logp, ll_post = zip(*r)

##         ll_classifier_train_X, ll_classifier_train_Y = \
##           getHMMinducedFeatures(ll_logp, ll_post, testDataY, c=1.0, add_delta_logp=add_logp_d)

##         #-----------------------------------------------------------------------------------------
##         # Classifier test data
##         #-----------------------------------------------------------------------------------------
##         testDataX = []
##         testDataY = []
##         for i in xrange(nEmissionDim):
##             temp = np.vstack([normalTestData[i], abnormalTestData[i]])
##             testDataX.append( temp )

##         testDataY = np.hstack([ -np.ones(len(normalTestData[0])), \
##                                 np.ones(len(abnormalTestData[0])) ])

##         r = Parallel(n_jobs=-1)(delayed(computeLikelihoods)(i, ml.A, ml.B, ml.pi, ml.F, \
##                                                                 [ testDataX[j][i] for j in xrange(nEmissionDim) ], \
##                                                                 ml.nEmissionDim, ml.nState,\
##                                                                 startIdx=startIdx, \
##                                                                 bPosterior=True)
##                                                                 for i in xrange(len(testDataX[0])))
##         _, ll_classifier_test_idx, ll_logp, ll_post = zip(*r)

##         # nSample x nLength
##         ll_classifier_test_X, ll_classifier_test_Y = \
##           getHMMinducedFeatures(ll_logp, ll_post, testDataY, c=1.0, add_delta_logp=add_logp_d)

##         #-----------------------------------------------------------------------------------------
##         d = {}
##         d['nEmissionDim'] = ml.nEmissionDim
##         d['A']            = ml.A 
##         d['B']            = ml.B 
##         d['pi']           = ml.pi
##         d['F']            = ml.F
##         d['nState']       = nState
##         d['startIdx']     = startIdx
##         d['ll_classifier_train_X']  = ll_classifier_train_X
##         d['ll_classifier_train_Y']  = ll_classifier_train_Y            
##         d['ll_classifier_train_idx']= ll_classifier_train_idx
##         d['ll_classifier_test_X']   = ll_classifier_test_X
##         d['ll_classifier_test_Y']   = ll_classifier_test_Y            
##         d['ll_classifier_test_idx'] = ll_classifier_test_idx
##         d['nLength']      = nLength
##         ut.save_pickle(d, modeling_pkl)
    
##     return
