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
from ghmm import *

class learning_hmm():
    def __init__(self, data_path, start_prob, trans_prob, mean, covars):

        self.start_prob = start_prob
        self.trans_prob = trans_prob
        self.mean       = mean
        self.covars     = covars

        self.model = hmm.GaussianHMM(3, "full", self.startprob, self.transmat)

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
    def fit(self):

        pass


if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    opt, args = p.parse_args()

    ## Init variables    
    data_path = os.getcwd()
    mean   = 0.0
    covars = 1.0
    
    ######################################################    
    # Get Raw Data
    td = dod.traj_data()

    test_dict = td.discrete_profile(False)
    td.update_trans_mat_all(test_dict)

    ######################################################    
    # Training and Prediction
    lh = learning_hmm(data_path,td.start_prob_vec, td.trans_prob_mat, mean, covars)

    lh.fit()

    
    
