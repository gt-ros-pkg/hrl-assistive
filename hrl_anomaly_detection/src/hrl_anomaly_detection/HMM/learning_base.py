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
#from hrl_srvs.srv import FloatArray_FloatArray, FloatArray_FloatArrayResponse

## import scipy
## from scipy import optimize
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.externals import six

class learning_base():
    def __init__(self, data_path, aXData):

        # Common parameters
        self.data_path = data_path
        self.aXData = aXData
        ## self.aYData = aYData

        # Tunable parameters        
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
        print "No training method is defined."
        pass


    #----------------------------------------------------------------------        
    #
    def score(self):
        print "No score method is defined."
        pass


    #----------------------------------------------------------------------        
    #
    def get_params(self, deep=False):

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

    
    #----------------------------------------------------------------------        
    #
    def set_params(self, **params):

        if not params:                                                                                            
            # Simple optimisation to gain speed (inspect is slow)                                                 
            return self 

        valid_params = self.get_params(deep=True)            
        for key, value in six.iteritems(params): 
            # simple objects case
            if not key in valid_params:
                raise ValueError('Invalid parameter %s ' 'for estimator %s'
                                 % (key, self.__class__.__name__))                
            setattr(self, key, value)
                       
        return self

        
    #----------------------------------------------------------------------        
    #
    def predict(self, X_test, bBinary=True, sign=1.0):
        print "No prediction method is defined."
        return

    
    #----------------------------------------------------------------------        
    #
    def cross_validation(self, nFold):

        nSample = len(self.aXData)
        
        # Variable check
        if nFold > nSample:
            print "Wrong nVfold number"
            sys.exit()

        # K-fold CV
        from sklearn import cross_validation
        scores = cross_validation.cross_val_score(self, self.aXData, cv=nFold)

        print scores
        
        
    #----------------------------------------------------------------------        
    #
    def param_estimation(self, tuned_parameters, nFold):

        nSample = len(self.aXData)
        
        # Variable check
        if nFold > nSample:
            print "Wrong nVfold number"
            sys.exit()

        # Split the dataset in two equal parts
        X_train, X_test = train_test_split(self.aXData, test_size=0.5, random_state=0)
        #Y_train = [1.0]*X_train.shape[0] # Dummy


        clf = GridSearchCV(self, tuned_parameters, cv=nFold, scoring=self.score)
        clf.fit(X_train) # [n_samples, n_features] 
        
        
        ## scores = ['precision', 'recall']
        ## for score in scores:
        ##     print("# Tuning hyper-parameters for %s" % score)
        ##     print()

        ##     clf = GridSearchCV(self, tuned_parameters, cv=nFold, scoring=score)
        ##     clf.fit(X_train) # [n_samples, n_features] 
        ##     sys.exit()
            
        ##     print("Best parameters set found on development set:")
        ##     print()
        ##     print(clf.best_estimator_)
        ##     print()
        ##     print("Grid scores on development set:")
        ##     print()
        ##     for params, mean_score, scores in clf.grid_scores_:
        ##         print("%0.3f (+/-%0.03f) for %r"
        ##               % (mean_score, scores.std() / 2, params))
        ##     print()
            
        ##     print("Detailed classification report:")
        ##     print()
        ##     print("The model is trained on the full development set.")
        ##     print("The scores are computed on the full evaluation set.")
        ##     print()
        ##     y_true, y_pred = y_test, clf.predict(X_test)
        ##     print(classification_report(y_true, y_pred))
        ##     print()
                
    #----------------------------------------------------------------------        
    # Normalize along with each feature, where X is sample X feature
    def set_normalization_param(self, X):
        ## print "Set normalization parameter"
        # Mean
        self.mean =  np.average(X,axis=0)

        # Variance
        self.std  = np.std(X,axis=0)
        

    #----------------------------------------------------------------------
    # Normalize along with each feature, where X is sample X feature
    def get_normalization(self, X):
        ## print "Get normalization"
        
        # Normalizaed features
        normal_X = (X - self.mean) / self.std

        return normal_X

    
    ## #----------------------------------------------------------------------
    ## #
    ## def service(self, req):
    ##     print "Request: ", req.val

    ##     aGoal = np.array((req.val)[0:2])

    ##     # Get a best start
    ##     aBestCondition = self.optimization(aGoal)

    ##     lJoint = aBestCondition[-self.nMaxJoint:].tolist()

    ##     # Not implemented        
    ##     if self.use_mobile_base:
    ##         mBase  = np.matrix([0.0, aBestStart[-3], 0.0]).T
    ##         lJoint = lJoint + [mBase[1,0]]
            
    ##     print "Response: ", lJoint
    ##     return FloatArray_FloatArrayResponse(lJoint)
    
