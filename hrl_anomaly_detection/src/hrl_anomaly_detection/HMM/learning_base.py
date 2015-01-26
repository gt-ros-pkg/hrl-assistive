#!/usr/local/bin/python

import sys, os
import numpy as np, math
import roslib; roslib.load_manifest('hrl_anomaly_detection')
import rospy
import inspect
import warnings
import random
import scipy as scp

# Util
import hrl_lib.util as ut
#from hrl_srvs.srv import FloatArray_FloatArray, FloatArray_FloatArrayResponse

## import scipy
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.externals import six

class learning_base():
    def __init__(self, aXData=None, trans_type="left_right"):

        # Common parameters
        self.aXData = aXData
        ## self.aYData = aYData
        self.trans_type=trans_type

        # etc 
        self.ml = None
        
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
    def param_estimation(self, tuned_parameters, nFold, save_file=None):
        # nFold: integer and less than the total number of samples.

        nSample = len(self.aXData)
        
        # Variable check
        if nFold > nSample:
            print "Wrong nVfold number"
            sys.exit()

        # Split the dataset in two equal parts
        X_train = self.aXData

        print("Tuning hyper-parameters for %s :", X_train.shape)
        print()        
        clf = GridSearchCV(self, tuned_parameters, cv=nFold, n_jobs=8)
        clf.fit(X_train) # [n_samples, n_features] 

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

        # Save data
        data = {}
        data['mean'] = mean_list
        data['std'] = std_list
        data['params'] = params_list
        if save_file is None:
            save_file='tune_data.pkl'            
        ut.save_pickle(data, save_file)
        
        
                
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


    #----------------------------------------------------------------------        
    #                
    def init_trans_mat(self, nState):

        # Reset transition probability matrix
        trans_prob_mat = np.zeros((nState, nState))
        
        for i in xrange(nState):

            # Exponential function                
            # From y = a*e^(-bx)
            #a = 0.4
            #b = np.log(0.00001/a)/(-(nState-i))
            #f = lambda x: a*np.exp(-b*x)

            # Exponential function
            # From y = -a*x + b
            b = 0.4
            a = b/float(nState)
            f = lambda x: -a*x+b

            for j in np.array(range(nState-i))+i:
                trans_prob_mat[i,j] = f(j)

            # Gaussian transition probability
            ## z_prob = norm.pdf(float(i),loc=u_mu_list[i],scale=u_sigma_list[i])

            # Normalization 
            trans_prob_mat[i,:] /= np.sum(trans_prob_mat[i,:])
                
        return trans_prob_mat

    #----------------------------------------------------------------------        
    #                
    ## def init_emission_mat(self, nState):

    ##     n,m = self.aXData.shape
        
    ##     mu    = np.zeros((self.nState,1))
    ##     sigma = np.zeros((self.nState,1))


    ##     for i in xrange(self.nState):
    ##         mu[i], sigma[i] = self.feature_to_mu_sigma(self.aXData, i+1)

    ##     B = np.hstack([mu, sigma])

    ##     return B



    
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
    


