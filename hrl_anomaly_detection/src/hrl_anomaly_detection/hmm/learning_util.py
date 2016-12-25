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


# Returns mu,sigma for n hidden-states from feature-vector
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
def vectors_to_mean_cov(vecs, nState, nEmissionDim, cov_type='full'):
    index = 0
    m, n = np.shape(vecs[0]) # samples x length
    mus  = [np.zeros(nState) for i in xrange(nEmissionDim)]
    cov  = np.zeros((nState, nEmissionDim, nEmissionDim))
    DIVS = n/nState

    while index < nState:
        m_init = index*DIVS

        ## temp_vecs = [vec[:, m_init:(m_init+DIVS)] for vec in vecs]
        temp_vecs = [np.reshape(vec[:, m_init:(m_init+DIVS)], (1, DIVS*m)) for vec in vecs]
        for i, mu in enumerate(mus):
            mu[index] = np.mean(temp_vecs[i].flatten())

        ## print np.shape(temp_vecs), np.shape(cov)
        cov[index, :, :] = np.cov(np.concatenate(temp_vecs, axis=0))
        if cov_type == 'diag':
            for i in xrange(nEmissionDim):
                for j in xrange(nEmissionDim):
                    if i==j: continue
                    cov[index,i, j] = 0.0
        index += 1

    return mus, cov


def convert_sequence(data, emission=False):
    '''
    @data: dimension x sample x length

    return: data with sample x ???
    TODO: need to replace entire code it looks too inefficient conversion.
    '''

    # change to array from other types => dimension x sample x length
    X = [copy.copy(np.array(d)) if type(d) is not np.ndarray else copy.copy(d) for d in data]

    # Change to two-dimensional array => dim x sample x length
    X = [np.reshape(x, (1, len(x))) if len(np.shape(x)) == 1 else x for x in X]

    # sample, length
    n, m = np.shape(X[0])

    Seq = []
    for i in xrange(n):
        Xs = []
        for j in xrange(m):
            Xs.append([x[i, j] for x in X]) # dim x j

        if emission:
            Seq.append(Xs) # sample x dim x ... 
        else:
            Seq.append(np.array(Xs).flatten().tolist())

    return np.array(Seq)


def scaling(X, min_c=None, max_c=None, scale=10.0, bMinMax=False, verbose=False):
    '''
    scale should be over than 10.0(?) to avoid floating number problem in ghmm.
    Return list type
    '''
    ## X_scaled = preprocessing.scale(np.array(X))

    if min_c is None or max_c is None:
        min_c = np.min(X)
        max_c = np.max(X)

    X_scaled = np.array(X)
    X_scaled = (X_scaled-min_c) / (max_c-min_c) * scale

    if verbose: print min_c, max_c, " : ", np.min(X_scaled), np.max(X_scaled)
    if bMinMax:
        return X_scaled.tolist(), min_c, max_c
    else:
        return X_scaled.tolist()
    
def symmetric_entropy(p,q):
    '''
    Return the sum of KL divergences
    '''

    return min(entropy(p,q) , entropy(q,p))

def shannon_entropy(p):
    if type(p) is list: p = np.array(p)

    ## p += 1e-6
    p /= np.linalg.norm(p)
    return -np.sum(p*np.log(p))

