#!/usr/local/bin/python
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

import sys, os
import numpy as np, math
import inspect
import warnings

# Util
import hrl_lib.util as ut

## import scipy
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.externals import six

class learning_base():
    def __init__(self):
        self.ml = None
        pass

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
            raise RuntimeError("Estimators should always "
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

        
    def fit(self):
        '''
        '''
        print "No training method is defined."
        pass


    def score(self, X,y):
        '''
        '''
        return self.ml.score(X,y)


    def get_params(self, deep=False):
        '''
        '''
        out = dict()
        for key in self._get_param_names():
            # We need deprecation warnings to always be on in order to
            # catch deprecated param values.
            # This is set in utils/__init__.py but it gets overwritten
            # when running under python3 somehow.
            warnings.simplefilter("always", DeprecationWarning)
            try:
                with warnings.catch_warnings(record=True) as w:
                    value = getattr(self, key, None)
                if len(w) and w[0].category == DeprecationWarning:
                # if the parameter is deprecated, don't show it
                    continue
            finally:
                warnings.filters.pop(0)

            # XXX: should we rather test if instance of estimator?
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        
        return dict(out.items())

    
    def set_params(self, **params):
        '''
        '''

        if not params:                                                                                            
            # Simple optimisation to gain speed (inspect is slow)                                                 
            return self 

        valid_params = self.get_params(deep=True)            
        for key, value in six.iteritems(params):
            # simple objects case
            if not key in valid_params:
                continue
                ## raise ValueError('Invalid parameter %s ' 'for estimator %s'
                ##                  % (key, self.__class__.__name__))                
            setattr(self, key, value)

        return self

        
    def predict(self, X):
        '''
        '''
        print "No prediction method is defined."
        return

    
    def cross_validation(self, X, nFold):
        '''
        '''
        nSample = len(X)
        
        # Variable check
        if nFold > nSample:
            print "Wrong nVfold number"
            sys.exit()

        # K-fold CV
        from sklearn import cross_validation
        scores = cross_validation.cross_val_score(self, X, cv=nFold)

        print scores
        
        
    def param_estimation(self, X, parameters, y=None, nFold=2, n_jobs=-1, save_file=None):
        '''
        '''
        # nFold: integer and less than the total number of samples.

        nSample = len(X)
        
        # Variable check
        if nFold > nSample:
            print "Wrong nVfold number"
            sys.exit()

        # Split the dataset in two equal parts
        print("Tuning hyper-parameters for %s :", X.shape)
        print()        
        clf = GridSearchCV(self, parameters, cv=nFold, n_jobs=n_jobs, verbose=50)
        clf.fit(X) # [n_samples, n_features] 

        print("Best parameters set found on development set:")
        print()
        print(clf.best_estimator_)
        print()
        print("Grid scores on development set:")
        print()

        params_list = []
        mean_list = []
        std_list = []
        
        for params, mean_score, scores in clf.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean_score, scores.std() / 2, params))
            params_list.append(params)
            mean_list.append(mean_score)
            std_list.append(scores.std())
        print()

        return params_list, mean_list, std_list

