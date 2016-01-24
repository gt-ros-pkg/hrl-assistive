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

import sys, copy
import numpy as np
from scipy.stats import norm, entropy
import scipy
import ghmm

def updateClassifierCoff(model, X, y, c1, c2, classifier_type='time', optimization=True):
    
    l_logp, l_post = model.loglikelihoods(X, bPosterior=True)

    if classifier_type=='time':
        for i in xrange(len(l_logp)):

            # Find the best posterior distribution
            min_index, min_dist = model.findBestPosteriorDistribution(l_post[i])

            model.opt_B.append( model.ll_mu[min_index] )
            model.opt_A.append( model.ll_std[min_index] )
            model.opt_idx.append( min_index )
            model.opt_logp.append( l_logp[i] )
            
    elif classifier_type=='smooth':
        # compute w for new data
        #post_idx = range(0,len(post), model.nEmissionDim)
        for i in xrange(len(l_logp)):

            sum_w   = 0.
            sum_mu  = 0.
            sum_std = []        

            for j in xrange(len(model.ll_mu)):

                dist    = 1.0/entropy(l_post[i], model.l_statePosterior[j])
                sum_w   += dist
                sum_mu  += dist * model.ll_mu[j]
                sum_std.append( dist * model.ll_std[j] )

            # append ml.A, B, D
            model.opt_A.append(sum_std/sum_w)
            model.opt_B.append(sum_mu/sum_w)
            model.opt_logp.append(l_logp[i])
            
    else:
        print "UpdateClassifier> new"

        for i in xrange(len(l_logp)):

            sum_w   = 0.
            sum_mu  = 0.
            sum_std = []
            weight_l = []

            for j in xrange(len(model.ll_mu)):
                if model.cluster_type == 'time':
                    dist = entropy(l_post[i], model.l_statePosterior[j])
                else:
                    dist = np.linalg.norm(l_post[i]- model.l_statePosterior[j])

                if dist < 1e-6: weight = 1e+6
                else: weight = 1.0/dist
    
                ## weight = 1.0/(dist*dist)
                sum_w  += weight
                sum_mu += weight * model.ll_mu[j]
                ## sum_std.append( dist * model.ll_std[j] )
                weight_l.append(weight)

            est_mu   = sum_mu/sum_w
            ## est_std  = np.sum(sum_std)/sum_w
            ## est_mult = [x*est_std/sum_w for x in dist_l]
            est_mult = [x*model.ll_std[idx]/sum_w for idx, x in enumerate(weight_l)]

            # append ml.A, B, D
            model.opt_A.append(est_mult)
            model.opt_B.append(est_mu)
            model.opt_logp.append(l_logp[i])        
            
    model.opt_y += [y]*len(l_logp)

    #
    c  = 1.0/float(len(model.opt_y))
    x0 = model.l_ths_mult
    ## print classifierCost(x0, model, c, c1, c2, classifier_type)
    ## sys.exit()

    if optimization:
        ###############################################################                                     
        # Set bound for x data.
        lBounds = []

        # Fixed feature range
        for i in xrange(model.nState):
            lBounds.append([-20.0,0.5])

        ## constraints_dict = {}
        ## constraints_dict['']
        minimizer_kwargs={}
        minimizer_kwargs['args']   = (model, c, c1, c2, classifier_type)
        minimizer_kwargs['method'] = 'L-BFGS-B'
        minimizer_kwargs['bounds'] = tuple(lBounds)
        
        
        ## lBestCondition = scipy.optimize.minimize(classifierCost, x0, args=(model, c, c1, c2),
        ## method='Powell', options={'maxiter': 260})
        lBestCondition = scipy.optimize.minimize(classifierCost, x0, args=(model, c, c1, c2, classifier_type), \
                                                 method='SLSQP', bounds=tuple(lBounds), tol=0.0000001)
        ## lBestCondition = scipy.optimize.basinhopping(classifierCost, x0, minimizer_kwargs=minimizer_kwargs,\
        ##                                              stepsize=0.5)
        ## lBestCondition = scipy.optimize.minimize(classifierCost, x0, args=(model, c, c1, c2, smooth), \
        ##                                          method='Newton-CG', jac=classifierCostPrime)        
        model.l_ths_mult = lBestCondition['x']
        print lBestCondition
                
    ## print model.l_ths_mult, np.shape(model.opt_B), np.shape(model.opt_A), np.shape(model.opt_idx), np.shape(model.opt_logp), np.shape(model.opt_y)
    print "00000000000000000000000000000000000000000000000"
    ## sys.exit()
    
    return True

#--------------------------------------------------------------------------
def classifierCost(x, model, c=1.0, c1=1.0, c2=1.0, classifier_type='time'):

    new_x = np.array(x)

    cost1 = 0.5*np.sum(new_x*new_x)
    cost2 = 0.0
    cost_sum1 = 0.0
    cost_sum2 = 0.0    

    if classifier_type=='time':
        new_c = np.array([[x[idx] for idx in model.opt_idx]]).T
        exp_l = np.array(model.opt_A).dot( new_c) + np.array([model.opt_B]).T
        dec   = exp_l - np.array([model.opt_logp]).T
    else:
        exp_l = np.array(model.opt_A).dot( np.array([x]).T) + np.array([model.opt_B]).T 
        dec   = exp_l - np.array([model.opt_logp]).T

    if False:
        '''
        sigmoidal loss function
        '''    
        dec = np.sign( dec )

        for i in xrange(len(dec)):

            # mu, sigma
            eta = 1.0 - scipy.stats.norm.pdf(model.opt_logp[i], loc=model.opt_B[i], scale=np.sum(model.opt_A[i]))

            if model.opt_y[i] == 1.0:
                cost_sum1 += eta * (1. - dec[i])/2.0
            else:
                cost_sum2 += (1.0-eta) * (1. + dec[i])/2.0

        cost2 = c * (c1*cost_sum1 + c2*cost_sum2)

    else:
        '''
        hinge loss function
        '''    
        for i in xrange(len(model.opt_y)):

            if model.opt_y[i] > 0:
                cost_sum1 += max(0, c1 - c1*dec[i])
            else:
                cost_sum2 += max(0, 1.0 + (2.0*c2-1.0)*dec[i])

        cost1 = cost1*c
        cost2 = cost_sum1 + cost_sum2

    ## print cost1, cost2, np.shape(model.opt_y), np.shape(model.opt_A)

    cost = cost1 + cost2
    return cost

#--------------------------------------------------------------------------
def classifierCostPrime(x, model, c=1.0, c1=1.0, c2=1.0, classifier_type=False):

    new_x = np.array(x)

    # cost 0
    dJdw_0 = np.array(x)/c

    # cost 1 and 2
    dJdw_1 = 0.0
    dJdw_2 = 0.0
    new_c = np.array([[x[idx] for idx in model.opt_idx]]).T
    exp_l = np.array(model.opt_A).dot( new_c) + np.array([model.opt_B]).T
    dec   = exp_l - np.array([model.opt_logp]).T        
    coff  = 2*c2-1.0
    
    for i in xrange(len(model.opt_y)):
        if model.opt_y[i] > 0:
            if dec[i] < 1.0:
                dJdw_1 += -model.opt_A[i] 
        else:
            if coff*dec[i] < 1.0:
                dJdw_2 += coff * model.opt_A[i]
            
    return dJdw_0 + dJdw_1 + dJdw_2


## def expLikelihoods(post, ll_mu, ll_std, ths_mult, l_statePosterior):
##     '''
##     return estimated likelihood, estimated mean, estimated standard deviation
##     Need to implement sum of muliple gaussian distributions
##     '''

##     sum_w = 0.
##     sum_l = 0.
##     sum_mu  = 0.
##     sum_std = 0.

##     for i in xrange(len(ll_mu)):
##         dist = 1.0/entropy(post, l_statePosterior[i])
##         sum_w += dist
##         sum_l += dist * (ll_mu[i] + ths_mult[i] * ll_std[i])
##         sum_mu  += dist * ll_mu[i]
##         sum_std += dist * ll_std[i]

##     return sum_l/sum_w, sum_mu/sum_w, sum_std/sum_w
