#!/usr/local/bin/python

import sys, os
import numpy as np, math
import roslib; roslib.load_manifest('sandbox_dpark_darpa_m3')
import rospy
import inspect
import warnings
import random

# Util
import hrl_lib.util as ut
from hrl_srvs.srv import FloatArray_FloatArray, FloatArray_FloatArrayResponse

import scipy
from scipy import optimize
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report

import dim_reduction_joint as drj
import dim_reduction as dr
import sandbox_dpark_darpa_m3.lib.ode_sim_lib.ode_sim_param as param

TOL=0.0001

class learning_base():
    def __init__(self, s_robot, data_path, drj_renew=False):

        # Tunable parameters
        self.s_robot = s_robot
        self.b_mobile_base = False
        self.data_path = data_path
        self.drj_renew = drj_renew
        
        self.aXData = None
        self.aYData = None

        self.use_mobile_base = False

        # Simulation Parameters
        self.l_robot = None
        if self.s_robot == 'sim3':
            self.l_robot = '--planar_three_link_capsule'
         
        self.pm  = param.parameters(self.l_robot)    
        param_pkl = os.path.join(data_path,'param','work_param.pkl')
        self.pm.load_param(param_pkl)
        
        ## Get joints with dimension reduction
        self.DRJ = drj.JOINT_DIM_REDUCTION(self.s_robot,self.l_robot,self.data_path, pm=self.pm)    
        pkl_file = os.path.join(data_path, 'param', 'planar_three_link_capsule_joints.pkl')
        self.DRJ.jointDimReduction(pkl_file, drj_renew)
        self.DR = dr.DIM_REDUCTION(s_robot, data_path, method='fa')

        self.nMaxJoint = len(self.DRJ.rbt.lJtsMax)
        
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
        print "No get_params method is defined."
        pass

    
    #----------------------------------------------------------------------        
    #
    def set_params(self, **params):
        print "No set_params method is defined."
        pass

        
    #----------------------------------------------------------------------        
    #
    def predict(self, X_test, bBinary=True, sign=1.0):
        print "No prediction method is defined."
        return

    
    #----------------------------------------------------------------------        
    # Estimated probability of success checking with 
    def predict_with_constraints(self, x, sign=1.0):

        lJoint = x[-self.nMaxJoint:]

        mEE,_ = self.DRJ.rbt.kinematics.FK(lJoint, self.nMaxJoint)

        if (mEE[0,0] < self.pm.lStartXlim[0] - TOL or
            mEE[0,0] > self.pm.lStartXlim[1] + TOL or
            mEE[1,0] < self.pm.lStartYlim[0] - TOL or
            mEE[1,0] > self.pm.lStartYlim[1] + TOL or
            mEE[2,0] < self.pm.lStartZlim[0] - TOL or
            mEE[2,0] > self.pm.lStartZlim[1] + TOL):
            return -100.0*sign

        else:
            return self.predict(x, bBinary=False)*sign

        
    #----------------------------------------------------------------------        
    # 
    def optimization(self, x_fixed):
            
        ###############################################################                         
        # Set initial point for optimization
        # 1) Grid search for initial x0
        joints    = self.DRJ.getAllJointsOnStartLine() # dim reduction for joints
        nSample,_ = joints.shape
        aGoal     = np.zeros((joints.shape[0],len(x_fixed))) + x_fixed
        
        raw_X     = np.hstack([aGoal,joints])
        normal_X  = self.DR.transform(raw_X) # dim reduction for all data        
        #y         = self.Likelihood_ratio(normal_X)
        y         = self.predict(raw_X, False)

        x0_ind = np.argmax(y)
        x0     = raw_X[x0_ind,:]
            
        # 2) Random selection for initial x0
        ## for nStart in xrange(nRndStart):            
            ## # Get random ee pose
            ## x = random.uniform((self.pm).lStartXlim[0], (self.pm).lStartXlim[1])
            ## y = random.uniform((self.pm).lStartYlim[0], (self.pm).lStartYlim[1])
            ## z = random.uniform((self.pm).lStartZlim[0], (self.pm).lStartZlim[1])
            ## mStart = np.matrix([x,y,z]).T

            ## # Get random joint angles over the above ee pose
            ## ret, lJointInit, _ = self.DRJ.rbt.getJointByPhi(mStart)
            ## x0 = np.hstack([x_fixed, np.array(lJointInit)])

            
        ###############################################################                                     
        # Set bound for x data.
        lBounds = []

        # Fixed feature range
        for d in x0[:-self.nMaxJoint]:
            lBounds.append([d,d])

        # Pseudo initial condition range        
        for i in xrange(self.nMaxJoint):
            lBounds.append([self.DRJ.rbt.lJtsMin[i], self.DRJ.rbt.lJtsMax[i]])


        ###############################################################                
        # Optimization part
        direct_optimization = True
        if direct_optimization:

            lBestCondition = optimize.minimize(self.predict_with_constraints,x0,args=(-1.0,), method='L-BFGS-B', bounds=tuple(lBounds), options={'maxiter': 60})

        else:
            # Bounds class to set x range
            class Bounds(object):
                def __init__(self, aBounds):
                    self.xmax = aBounds[:,0]
                    self.xmin = aBounds[:,1]
                def __call__(self, **kwargs):
                    x = kwargs["x_new"]
                    tmax = bool(np.all(x <= self.xmax))
                    tmin = bool(np.all(x >= self.xmin))
                    return tmax and tmin

            bound_test = Bounds(np.array(lBounds))

            # basinhopping uses minimize function
            minimizer_kwargs={}
            minimizer_kwargs['args'] = (-1.0,)
            minimizer_kwargs['method'] = 'L-BFGS-B'
            minimizer_kwargs['bounds'] = tuple(lBounds)
            #minimizer_kwargs['options'] = {'maxiter': 50}

            lBestCondition = optimize.basinhopping(self.predict_with_constraints,x0,niter=50,minimizer_kwargs=minimizer_kwargs, accept_test=bound_test, stepsize=0.5)

        ###############################################################                            
        ## if lBestCondition['fun'] > self.predict_with_constraints(x0, sign=-1.0):
        ##     print "Wrong optimization, return initial minimum point."
        ##     continue

        ## print "****************"
        ## print "Best Condition = "
        ## print lBestCondition
        ## print "****************"

        ###############################################################                            
        # Get a best condition (I have to simplify this part)
        lConditions = []        
        lConditions.append(lBestCondition['x'])
            
        # Find an optimum from multiple minimum.
        fLow = 1000.0
        lOptCondition = []
        for i, condition in enumerate(lConditions):

            fCurrent = self.predict(condition, bBinary=False, sign=-1.0)
            if fLow > fCurrent:
                fLow = fCurrent
                lOptCondition = condition
                
        print "Minimum: ", fLow, lOptCondition
        return lOptCondition                

        
    #----------------------------------------------------------------------        
    #
    def cross_validation(self, nFold):

        nSample = len(self.aYData)
        
        # Variable check
        if nFold > nSample:
            print "Wrong nVfold number"
            sys.exit()

        # K-fold CV
        from sklearn import cross_validation
        scores = cross_validation.cross_val_score(self, self.aXData, self.aYData, cv=nFold)

        print scores
        
        
    #----------------------------------------------------------------------        
    #
    def param_estimation(self, tuned_parameters, nFold):

        nSample = len(self.aYData)
        
        # Variable check
        if nFold > nSample:
            print "Wrong nVfold number"
            sys.exit()

        # Split the dataset in two equal parts
        X_train, X_test, y_train, y_test = train_test_split(self.aXData, self.aYData, test_size=0.5, random_state=0)

        scores = ['precision', 'recall']

        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print()

            clf = GridSearchCV(self, tuned_parameters, cv=nFold, scoring=score)
            clf.fit(X_train, y_train)
            
            print("Best parameters set found on development set:")
            print()
            print(clf.best_estimator_)
            print()
            print("Grid scores on development set:")
            print()
            for params, mean_score, scores in clf.grid_scores_:
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean_score, scores.std() / 2, params))
            print()
            
            print("Detailed classification report:")
            print()
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.")
            print()
            y_true, y_pred = y_test, clf.predict(X_test)
            print(classification_report(y_true, y_pred))
            print()
                
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
    def service(self, req):
        print "Request: ", req.val

        aGoal = np.array((req.val)[0:2])

        # Get a best start
        aBestCondition = self.optimization(aGoal)

        lJoint = aBestCondition[-self.nMaxJoint:].tolist()

        # Not implemented        
        if self.use_mobile_base:
            mBase  = np.matrix([0.0, aBestStart[-3], 0.0]).T
            lJoint = lJoint + [mBase[1,0]]
            
        print "Response: ", lJoint
        return FloatArray_FloatArrayResponse(lJoint)
    
