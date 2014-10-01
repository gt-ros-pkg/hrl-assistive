#!/usr/local/bin/python

import sys, os
import numpy as np, math

import roslib; roslib.load_manifest('hrl_anomaly_detection')
import rospy
import inspect
import warnings
import random

# Util
import hrl_lib.util as ut

import door_open_data as dod
import ghmm

class learning_hmm():
    def __init__(self, data_path):

        ## self.model = hmm.GaussianHMM(3, "full", self.startprob, self.transmat)

        F = ghmm.Float()  # emission domain of this model
        
        # Confusion Matrix NOTE ???
        ## cmat = np.zeros((4,4))

        
        pass

    #----------------------------------------------------------------------        
    #    
    @classmethod
    def _get_param_names(cls):

        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any                                       
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
  
        if init is object.__init__:
            # No explicit constructor to introspect          
            return []
                                                                                                     
        
        # introspect the constructor arguments to find the model parameters                                      
        # to represent
                                                                                                    
        args, varargs, kw, default = inspect.getargspec(init)
                                                             
        if varargs is not None:
            raise RuntimeError("scikit-learn estimators should always " 
                                   "specify their parameters in the signature" 
                                   " of their __init__ (no varargs)."
                                   " %s doesn't follow this convention." 
                                   % (cls, )) 

        # Remove 'self'                                                                                         
        # XXX: This is going to fail if the init is a staticmethod, but
        # who would do this?
        args.pop(0)
        args.sort() 
        return args


    #----------------------------------------------------------------------        
    #
    def fit(self, X):

        # We should think about multivariate Gaussian pdf.
        
        mu, sigma = vectors_to_mean_vars(X,self.nState)
        
        ## # Initial Probability Matrix       
        ## self.start_prob = start_prob
        ## First state must be 1 !!!
       
        # Transition probabilities per state        
        self.trans_prob = trans_prob

        # Emission probability matrix
        B = np.hstack([mu, sigma])

        
        m = HMMFromMatrices(sigma, DiscreteDistribution(sigma), A, B, pi)
        
        pass

    #----------------------------------------------------------------------        
    #
    def vectors_to_mean_vars(self, vecs, nState):

        m,n,k = vecs.shape # features, length, samples
        mu    = np.zeros((m,nState,k))
        sigma = np.zeros((m,nState,k))

        nDivs = n/float(nState)

        for i in xrange(m):
            index = 0
            while (index < nState):
                m_init = index*nDivs
                temp_vec = vecs[i][(m_init):(m_init+nDivs),:]
                
                mu[i][index] = np.mean(temp_vec, axis=0)
                sigma[i][index] = np.std(temp_vec, axis=0)
                index = index+1

        return mu,sigma
        


    

if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    opt, args = p.parse_args()

    ## Init variables    
    data_path = os.getcwd()
    nState    = 10.0
    
    ######################################################    
    # Get Raw Data
    td = dod.traj_data()
    data_vecs = td.get_raw_data()

    ######################################################    
    # Training and Prediction
    lh = learning_hmm(data_path, nState)
    
    lh.fit(data_vecs)

    
    
