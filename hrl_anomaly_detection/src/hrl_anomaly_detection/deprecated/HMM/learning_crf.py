#!/usr/local/bin/python

import sys, os, copy
import numpy as np, math

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
import hrl_anomaly_detection.door_opening.mechanism_analyse_daehyung as mad
from scipy.stats import norm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score
from sklearn.base import clone
from sklearn import cross_validation
from scipy import optimize
from joblib import Parallel, delayed
from scipy.optimize import fsolve

from learning_base import learning_base
import sandbox_dpark_darpa_m3.lib.hrl_dh_lib as hdl


class learning_crf(learning_base):
    def __init__(self, aXData, nState, nMaxStep, nFutureStep=5, nCurrentStep=10, step_size_list=None, trans_type="left_right"):

        learning_base.__init__(self, aXData, trans_type)
        
        ## Tunable parameters                
        self.nState= nState # the number of hidden states
        self.nFutureStep = nFutureStep
        self.nCurrentStep = nCurrentStep
        
        ## Un-tunable parameters
        self.nMaxStep = nMaxStep  # the length of profile
        self.obsrv_range = [np.min(aXData), np.max(aXData)]
        self.A = None # transition matrix        
        self.B = None # emission matrix        
        
        # Assign local functions
        learning_base.__dict__['fit'] = self.fit        
        learning_base.__dict__['predict'] = self.predict
        learning_base.__dict__['score'] = self.score                
        pass
        

    #----------------------------------------------------------------------        
    #
    def fit(self, X_train, A=None, B=None, pi=None, B_dict=None, verbose=False):

        return


    #----------------------------------------------------------------------        
    #
    def predict(self, X):

        return
        
    #----------------------------------------------------------------------
    # Returns the mean accuracy on the given test data and labels.
    def score(self, X_test, **kwargs):
        
        return
