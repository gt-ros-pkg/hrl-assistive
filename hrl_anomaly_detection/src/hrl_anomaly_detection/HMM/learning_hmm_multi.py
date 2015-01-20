#!/usr/local/bin/python

import sys, os, copy
import numpy as np, math
import scipy as scp

import roslib; roslib.load_manifest('hrl_anomaly_detection')
import rospy
import inspect
import warnings
import random

# Util
import hrl_lib.util as ut
## import cPickle
## from sklearn.externals import joblib

# Matplot
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import animation

## import door_open_data as dod
import ghmm
from scipy.stats import norm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score
from sklearn.base import clone
from sklearn import cross_validation
from scipy import optimize
## from pysmac.optimize import fmin                
from joblib import Parallel, delayed
## from scipy.optimize import fsolve
## from scipy import interpolate

from learning_base import learning_base
import sandbox_dpark_darpa_m3.lib.hrl_dh_lib as hdl


class learning_hmm(learning_base):
    def __init__(self, aXData, nState, nMaxStep, nFutureStep=5, nCurrentStep=10, \
                 step_size_list=None, trans_type="left_right"):

        learning_base.__init__(self, aXData, trans_type)

        ## Tunable parameters                
        self.nState= nState # the number of hidden states
        self.nFutureStep = nFutureStep
        self.nCurrentStep = nCurrentStep
        self.step_size_list = step_size_list
        
        ## Un-tunable parameters
        ## self.trans_type = trans_type #"left_right" #"full"
        self.nMaxStep = nMaxStep  # the length of profile
        self.obsrv_range = [np.min(aXData), np.max(aXData)]
        self.A = None # transition matrix        
        self.B = None # emission matrix        
                
        # emission domain of this model        
        self.F = ghmm.Float()  
               
        # Assign local functions
        learning_base.__dict__['fit'] = self.fit        
        learning_base.__dict__['predict'] = self.predict
        learning_base.__dict__['score'] = self.score                
        pass

        
    #----------------------------------------------------------------------        
    #
    def fit(self, X_train, A=None, B=None, pi=None, B_dict=None, verbose=False):

        if A is None:        
            if verbose: print "Generate new A matrix"                
            # Transition probability matrix (Initial transition probability, TODO?)
            A = self.init_trans_mat(self.nState).tolist()

        if B is None:
            if verbose: print "Generate new B matrix"                                            
            # We should think about multivariate Gaussian pdf.        
            self.mu, self.sig = self.vectors_to_mean_sigma(X_train, self.nState)

            # Emission probability matrix
            B = np.hstack([self.mu, self.sig]).tolist() # Must be [i,:] = [mu, sig]
                
        if pi is None:            
            # pi - initial probabilities per state 
            ## pi = [1.0/float(self.nState)] * self.nState
            pi = [0.] * self.nState
            pi[0] = 1.0

        # HMM model object
        self.ml = ghmm.HMMFromMatrices(self.F, ghmm.GaussianDistribution(self.F), A, B, pi)
        
        ## print "Run Baum Welch method with (samples, length)", X_train.shape
        train_seq = X_train.tolist()
        final_seq = ghmm.SequenceSet(self.F, train_seq)        
        self.ml.baumWelch(final_seq, 10000)

        [self.A,self.B,self.pi] = self.ml.asMatrices()
        self.A = np.array(self.A)
        self.B = np.array(self.B)

        ## self.mean_path_plot(mu[:,0], sigma[:,0])        
        ## print "Completed to fitting", np.array(final_seq).shape
        
        # state range
        self.state_range = np.arange(0, self.nState, 1)

        # Pre-computation for PHMM variables
        self.mu_z   = np.zeros((self.nState))
        self.mu_z2  = np.zeros((self.nState))
        self.mu_z3  = np.zeros((self.nState))
        self.var_z  = np.zeros((self.nState))
        self.sig_z3 = np.zeros((self.nState))
        for i in xrange(self.nState):
            zp             = self.A[i,:]*self.state_range
            self.mu_z[i]   = np.sum(zp)
            self.mu_z2[i]  = self.mu_z[i]**2
            #self.mu_z3[i]  = self.mu_z[i]**3
            self.var_z[i]  = np.sum(zp*self.state_range) - self.mu_z[i]**2
            #self.sig_z3[i] = self.var_z[i]**(1.5)


    #----------------------------------------------------------------------
    # Returns the mean accuracy on the given test data and labels.
    def score(self, X_test, **kwargs):

        if self.ml is None: 
            print "No ml!!"
            return -5.0        
        
        # Get input
        if type(X_test) == np.ndarray:
            X=X_test.tolist()

        sample_weight=None # TODO: future input
        
        #
        n = len(X)
        nCurrentStep = [5,10,15,20,25]
        nFutureStep = 1

        total_score = np.zeros((len(nCurrentStep)))
        for j, nStep in enumerate(nCurrentStep):

            self.nCurrentStep = nStep
            X_next = np.zeros((n))
            X_pred = np.zeros((n))
            mu_pred  = np.zeros((n))
            var_pred = np.zeros((n))
            
            for i in xrange(n):
                if len(X[i]) > nStep+nFutureStep: #Full data                
                    X_past = X[i][:nStep]
                    X_next[i] = X[i][nStep]
                else:
                    print "Error: input should be full length data!!"
                    sys.exit()

                mu, var = self.one_step_predict(X_past)
                mu_pred[i] = mu[0]
                var_pred[i] = var[0]

            total_score[j] = r2_score(X_next, mu_pred, sample_weight=sample_weight)

        ## print "---------------------------------------------"
        ## print "Total Score"
        ## print total_score
        ## print "---------------------------------------------"
        return sum(total_score) / float(len(nCurrentStep))
        
        
