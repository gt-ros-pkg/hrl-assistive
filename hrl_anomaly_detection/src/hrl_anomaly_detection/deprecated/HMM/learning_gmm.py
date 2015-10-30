#!/usr/local/bin/python

import sys, os
import numpy as np, math
import roslib; roslib.load_manifest('sandbox_dpark_darpa_m3')
import rospy
import warnings

import hrl_lib.util as ut
from hrl_srvs.srv import FloatArray_FloatArray, FloatArray_FloatArrayResponse

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

import scipy
from sklearn.externals import joblib
from sklearn import mixture
from sklearn.externals import six

import sandbox_dpark_darpa_m3.planar_robot.sub_scripts.learning_data as ld
import sandbox_dpark_darpa_m3.lib.ode_sim_lib.ode_sim_param as param
from sandbox_dpark_darpa_m3.planar_robot.sub_scripts2.learning_base import learning_base


class learning_gmm_lic1(learning_base):
    def __init__(self, s_robot, data_path, b_dim_reduction=True, b_dim_reduction_renew=False, dr_method='fa', gmm_0__n_components=5, gmm_1__n_components=5, gmm_0__covariance_type='full', gmm_1__covariance_type='full', gmm_0__params=None, gmm_1__params=None, gmm_0__n_init=1, gmm_1__n_init=1, gmm_0__random_state=None, gmm_1__random_state=None, gmm_0__thresh=0.01, gmm_1__thresh=0.01, gmm_0__min_covar=0.001, gmm_1__min_covar=0.001, gmm_0__init_params='wmc', gmm_1__init_params='wmc', gmm_0__n_iter=100, gmm_1__n_iter=100, bRenew=False, kpca_gamma=1.5, isomap_neighbors=5):
        
        learning_base.__init__(self, s_robot, data_path)

        ## Get training data    
        LD = ld.LEARNING_DATA(s_robot, data_path)

        # Load data
        self.aXData, self.aYData = LD.get_gs_data(data_path, opt.renew)    
        ## new_X = DR.normalization(X)

        #
        self.gmm_0 = None
        self.gmm_1 = None       
        self.b_dim_reduction = b_dim_reduction
        self.b_dim_reduction_renew = b_dim_reduction_renew
        self.dr_method = dr_method
        self.gmm_0__n_components = gmm_0__n_components
        self.gmm_1__n_components = gmm_1__n_components        
        self.gmm_0__covariance_type = gmm_0__covariance_type
        self.gmm_1__covariance_type = gmm_1__covariance_type

        self.gmm_0 = mixture.GMM(n_components=gmm_0__n_components,covariance_type=gmm_0__covariance_type)
        self.gmm_1 = mixture.GMM(n_components=gmm_1__n_components,covariance_type=gmm_1__covariance_type)

        self.kpca_gamma = kpca_gamma
        self.isomap_neighbors = isomap_neighbors
        
        # Assign local functions
        learning_base.__dict__['fit'] = self.fit        
        learning_base.__dict__['predict'] = self.predict
        learning_base.__dict__['score'] = self.score        
        learning_base.__dict__['get_params'] = self.get_params        
        learning_base.__dict__['set_params'] = self.set_params        

        
        pass

        
    #----------------------------------------------------------------------
    #    
    def fit(self, X_train, y_train):
        
        if self.b_dim_reduction == True:
            # dim reduction for joints
            ## raw_joints = self.DRJ.get_normalization(self.aXData[:,2:])
            ## dr_joints  = self.DRJ.transform(raw_joints)
            X, y = self.DR.fit_transform(X_train,y_train,dim=2,prefix='all',method=self.dr_method,renew=self.b_dim_reduction_renew, kpca_gamma=self.kpca_gamma, isomap_neighbors=self.isomap_neighbors)
        else:
            # No dim reduction for joints
            self.set_normalization_param(X_train)
            X = self.get_normalization(X_train) 
            y = y_train                

        print X_train.shape, y_train.shape
        print X.shape, y.shape
            

        ## Two labeled data    
        idxs = np.where(y==True)[0]
        success_dr_X = X[idxs,:]
        self.gmm_0.fit(success_dr_X)

        idxs = np.where(y==False)[0]
        failure_dr_X = X[idxs,:]
        self.gmm_1.fit(failure_dr_X)

        
    #----------------------------------------------------------------------
    # Compute the estimated probability (0.0~1.0)
    def predict(self, X_test, bBinary=True, sign=1.0):
        
        if self.b_dim_reduction == True:        
            ## dr_joints = self.DRJ.transform(org_x[:,2:]) # dim reduction for joints
            ## raw_X     = np.hstack([org_x[:,0:2],dr_joints])
            X = self.DR.transform(X_test) # dim reduction for all data
        else:
            X = self.get_normalization(X_test)
            
        # P0/P1 = P/(1-P)
        fProb = np.exp(self.gmm_0.score(X)) / ( np.exp(self.gmm_0.score(X)) + np.exp(self.gmm_1.score(X)))
        
        if bBinary:
            return sign*np.around(fProb)
        else:
            return sign*fProb
            

        
    #----------------------------------------------------------------------
    # Returns the mean accuracy on the given test data and labels.
    def score(self, X_test, y_test, sample_weight=None):

        # P0/P1 = P/(1-P)
        ## return self.gmm_0.score(X) - np.log(np.exp(self.gmm_0.score(X)) + np.exp(self.gmm_1.score(X)))

        ## from sklearn.metrics import accuracy_score
        ## return accuracy_score(y_test, np.around(self.predict(X_test)), sample_weight=sample_weight)

        from sklearn.metrics import r2_score
        return r2_score(y_test, self.predict(X_test), sample_weight=sample_weight)


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
        
        if self.gmm_0 == None or self.gmm_1 == None: return out

        #-------------------------------------------------------------
        gmm_0_param_dict = self.gmm_0.get_params(deep)
        for key in gmm_0_param_dict.keys():
            gmm_0_param_dict['gmm_0__'+key] = gmm_0_param_dict.pop(key)

        gmm_1_param_dict = self.gmm_1.get_params(deep)
        for key in gmm_1_param_dict.keys():
            gmm_1_param_dict['gmm_1__'+key] = gmm_1_param_dict.pop(key)


        ## print self.gmm_0.get_params(True)
        ## print gmm_0_param_dict.items()
        ## print out.items()
        ## print dict(gmm_0_param_dict.items() + gmm_1_param_dict.items() + out.items())

        return dict(gmm_0_param_dict.items() + gmm_1_param_dict.items() + out.items())

    
    #----------------------------------------------------------------------        
    #
    def set_params(self, **params):
        if not params:                                                                                            
            # Simple optimisation to gain speed (inspect is slow)                                                 
            return self 

        # gmm_0
        gmm_0_param_dict = {}
        gmm_1_param_dict = {}

        valid_params = self.get_params(deep=True)            
        for key, value in six.iteritems(params): 
            split = key.split('__',1)
            if len(split) > 1:
                # combined key name
                prefix, name = split
                if prefix.find('gmm_0') >= 0:
                    gmm_0_param_dict[name] = value
                elif prefix.find('gmm_1') >= 0:
                    gmm_1_param_dict[name] = value
                else:
                    raise ValueError('Invalid parameter %s ' 'for estimator %s'
                                     % (key, self.__class__.__name__))                
            else:
                # simple objects case
                if not key in valid_params:
                    raise ValueError('Invalid parameter %s ' 'for estimator %s'
                                     % (key, self.__class__.__name__))                
                setattr(self, key, value)
                
        self.gmm_0.set_params(params=gmm_0_param_dict)
        self.gmm_1.set_params(params=gmm_0_param_dict)
        ## self.gmm_1.set_params(gmm_1_param_dict)
        
        return self
            
        
        
    #----------------------------------------------------------------------
    #    
    ## def Likelihood_ratio(self, X):

    ##     return 1.0 - np.exp(self.gmm_0.score(X) - self.gmm_1.score(X))

        
    #----------------------------------------------------------------------
    #    
    ## def Likelihood_ratio_given_goal(self, goal, joints):
    ##     nSample,_ = joints.shape
    ##     aGoal     = np.zeros((joints.shape[0],goal.shape[1])) + goal
        
    ##     dr_joints = self.DRJ.transform(joints) # dim reduction for joints
    ##     raw_X     = np.hstack([aGoal,dr_joints])
    ##     dr_X      = self.DR.transform(raw_X) # dim reduction for all data

    ##     H0_weight = np.zeros(self.n_components)
    ##     H1_weight = np.zeros(self.n_components)
        
    ##     # H0 parameters (success)
    ##     for i in xrange(self.n_components):
    ##         H0_weight[i] = self.gmm_0.weights_[i]/self.gmm_0.covars_[i,0] * np.exp(-1.0/(2.0*self.gmm_0.covars_[i,0]**2.0)*(dr_X[0,0:2]-)**2.0)

    ##     # H1 parameters (failure)

        
    ##     return np.exp(self.gmm_1.score(dr_X) - self.gmm_0.score(dr_X))

    #----------------------------------------------------------------------
    #    
    def test3(self):

        goal = np.array([[0.6,0.2]])
        
        joints    = self.DRJ.getAllJointsOnStartLine() # dim reduction for joints
        nSample,_ = joints.shape
        aGoal     = np.zeros((joints.shape[0],len(goal))) + goal
        
        raw_X     = np.hstack([aGoal,joints])
        #dr_X      = self.DR.transform(raw_X) # dim reduction for all data
        normal_X  = self.get_normalization(raw_X)
        
        y         = self.Likelihood_ratio(normal_X)


        fig = plt.figure()

        ## likelihood ratio
        ax1 = fig.add_subplot(131, aspect='equal', projection='3d')    
        ax1.scatter(np.degrees(joints[:,0]), np.degrees(joints[:,1]), np.degrees(joints[:,2]),c=y,linewidth=0)
        ## new_y = (y-np.amin(y))/(np.amax(y)-np.amin(y))
        ## ax1.plot_trisurf(np.degrees(joints[:,0]), np.degrees(joints[:,1]), np.degrees(joints[:,2]), color=new_y, cmap=plt.cm.Spectral,linewidth=0)
        
        lJtsMin = np.degrees(self.DRJ.rbt.lJtsMin)
        lJtsMax = np.degrees(self.DRJ.rbt.lJtsMax)
        ax1.set_xlim([lJtsMin[0],lJtsMax[0]])
        ax1.set_ylim([lJtsMin[1],lJtsMax[1]])
        ax1.set_zlim([lJtsMin[2],lJtsMax[2]])
        ax1.set_xlabel("$theta_1$")
        ax1.set_ylabel("$theta_2$")
        ax1.set_zlabel("$theta_3$")
        ax1.view_init(38,-158)
        
        # Success (H0)
        ax2 = fig.add_subplot(132, aspect='equal', projection='3d')    
        ax2.scatter(np.degrees(joints[:,0]), np.degrees(joints[:,1]), np.degrees(joints[:,2]),c=self.gmm_0.score(normal_X),linewidth=0,alpha=0.3)

        ax2.set_xlim([lJtsMin[0],lJtsMax[0]])
        ax2.set_ylim([lJtsMin[1],lJtsMax[1]])
        ax2.set_zlim([lJtsMin[2],lJtsMax[2]])        
        ax2.set_xlabel("$theta_1$")
        ax2.set_ylabel("$theta_2$")
        ax2.set_zlabel("$theta_3$")
        ax2.view_init(38,-158)

        # Success (H1)
        ax3 = fig.add_subplot(133, aspect='equal', projection='3d')    
        ax3.scatter(np.degrees(joints[:,0]), np.degrees(joints[:,1]), np.degrees(joints[:,2]),c=self.gmm_1.score(normal_X),linewidth=0,alpha=0.3)

        ax3.set_xlim([lJtsMin[0],lJtsMax[0]])
        ax3.set_ylim([lJtsMin[1],lJtsMax[1]])
        ax3.set_zlim([lJtsMin[2],lJtsMax[2]])        
        ax3.set_xlabel("$theta_1$")
        ax3.set_ylabel("$theta_2$")
        ax3.set_zlabel("$theta_3$")
        ax3.view_init(38,-158)

        # Success Raw Data
        idxs = np.where(self.aYData==True)[0]
        X = self.aXData[idxs,:]
        joints = None
        markersize  = None
        goal_radius = 0.1 
        for x in X:
            x    = x.reshape((1,x.shape[0]))
            dist = np.linalg.norm(x[:,0:2]-goal)
            if dist < goal_radius:
                if joints == None:
                    joints = x[:,2:]
                    markersize = np.array([100. * (goal_radius-dist)/goal_radius + 50.])
                else:
                    joints = np.vstack([joints, x[:,2:]])
                    markersize = np.vstack([markersize, np.array([100. * (goal_radius-dist)/goal_radius + 50.])])
         
        ax2.scatter(np.degrees(joints[:,0]), np.degrees(joints[:,1]), np.degrees(joints[:,2]),marker='*',s=markersize,c='k',linewidth=0)        

        # Failure Raw Data
        idxs = np.where(self.aYData==False)[0]
        X = self.aXData[idxs,:]
        joints = None
        markersize  = None
        goal_radius = 0.1 
        for x in X:
            x    = x.reshape((1,x.shape[0]))
            dist = np.linalg.norm(x[:,0:2]-goal)
            if dist < goal_radius:
                if joints == None:
                    joints = x[:,2:]
                    markersize = np.array([100. * (goal_radius-dist)/goal_radius + 50.])
                else:
                    joints = np.vstack([joints, x[:,2:]])
                    markersize = np.vstack([markersize, np.array([100. * (goal_radius-dist)/goal_radius + 50.])]) 
                
        ax3.scatter(np.degrees(joints[:,0]), np.degrees(joints[:,1]), np.degrees(joints[:,2]),marker='*',s=markersize,linewidth=0)        
        
        plt.show()

        
    #----------------------------------------------------------------------
    #    
    def test2(self):

        goal = np.array([0.6,0.0])
        
        joints    = self.DRJ.getAllJointsOnStartLine() # dim reduction for joints
        nSample,_ = joints.shape
        aGoal     = np.zeros((joints.shape[0],len(goal))) + goal
        
        raw_X     = np.hstack([aGoal,joints])
        normal_X  = self.DR.transform(raw_X) # dim reduction for all data        
        #y         = self.Likelihood_ratio(normal_X)
        y         = self.predict(raw_X, False)

        xlim=[-2.0,2.0]
        ylim=[-2.0,2.0]

        fig = plt.figure()

        ## likelihood ratio
        ax1 = fig.add_subplot(131, aspect='equal')    
        ax1.scatter(normal_X[:,0], normal_X[:,1],c=y,linewidth=0)
        ## new_y = (y-np.amin(y))/(np.amax(y)-np.amin(y))
        ## ax1.plot_trisurf(np.degrees(joints[:,0]), np.degrees(joints[:,1]), np.degrees(joints[:,2]), color=new_y, cmap=plt.cm.Spectral,linewidth=0)
        
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim)
        ax1.set_xlabel("$theta_1$")
        ax1.set_ylabel("$theta_2$")
        
        # Success (H0)
        ax2 = fig.add_subplot(132, aspect='equal')    
        ax2.scatter(normal_X[:,0], normal_X[:,1],c=np.exp(self.gmm_0.score(normal_X)),linewidth=0,alpha=0.3)

        ax2.set_xlim(xlim)
        ax2.set_ylim(ylim)
        ax2.set_xlabel("$theta_1$")
        ax2.set_ylabel("$theta_2$")

        # Success (H1)
        ax3 = fig.add_subplot(133, aspect='equal')    
        ax3.scatter(normal_X[:,0], normal_X[:,1],c=np.exp(self.gmm_1.score(normal_X)),linewidth=0,alpha=0.3)

        ax3.set_xlim(xlim)
        ax3.set_ylim(ylim)
        ax3.set_xlabel("$theta_1$")
        ax3.set_ylabel("$theta_2$")

        # Success Raw Data
        idxs = np.where(self.aYData==True)[0]
        X = self.aXData[idxs,:]
        normal_X = None
        markersize  = None
        goal_radius = 0.1 
        for x in X:
            x    = x.reshape((1,x.shape[0]))
            dist = np.linalg.norm(x[:,0:2]-goal)
            if dist < goal_radius:
                if normal_X == None:
                    normal_X = self.DR.transform(x)
                    markersize = np.array([130. * (goal_radius-dist)/goal_radius + 20.])
                else:
                    normal_X = np.vstack([normal_X, self.DR.transform(x)])
                    markersize = np.vstack([markersize, np.array([130. * (goal_radius-dist)/goal_radius + 20.])])
         
        ax2.scatter(normal_X[:,0], normal_X[:,1], marker='*',s=markersize,c='k',linewidth=0)        

        # Failure Raw Data
        idxs = np.where(self.aYData==False)[0]
        X = self.aXData[idxs,:]
        normal_X = None
        markersize  = None
        goal_radius = 0.1 
        for x in X:
            x    = x.reshape((1,x.shape[0]))
            dist = np.linalg.norm(x[:,0:2]-goal)
            if dist < goal_radius:
                if normal_X == None:
                    normal_X = self.DR.transform(x)
                    markersize = np.array([130. * (goal_radius-dist)/goal_radius + 20.])
                else:
                    normal_X = np.vstack([normal_X, self.DR.transform(x)])
                    markersize = np.vstack([markersize, np.array([130. * (goal_radius-dist)/goal_radius + 20.])]) 
                
        ax3.scatter(normal_X[:,0], normal_X[:,1], marker='*',s=markersize,linewidth=0)        


        # Show optimum goal!!
        lCondition = lg.optimization(goal)
        normal_X  = self.DR.transform(lCondition) # dim reduction for all data        
        ax1.scatter(normal_X[0,0], normal_X[0,1],c='k',linewidth=0.0,marker='*',s=130)
        
        
        plt.show()

        
        
    def test(self):
        xlim=[-2.0,2.0]
        ylim=[-2.0,2.0]
        
        resol = 0.025
        sample_x = np.arange(-2.0,2.0,resol)
        sample_y = np.arange(-2.0,2.0,resol)
        sample_X, sample_Y = np.meshgrid(sample_x, sample_y)
        sample_XX = np.hstack([sample_X.reshape(sample_X.size,1),sample_Y.reshape(sample_Y.size,1)])

        sample_Z1 = np.exp(self.gmm_0.score(sample_XX))
        sample_Z2 = np.exp(self.gmm_1.score(sample_XX))

        plt.figure()
        ax1 = plt.subplot(121, aspect='equal')    
        CS = plt.contour(sample_X, sample_Y, sample_Z1.reshape(sample_X.shape))
        plt.clabel(CS, inline=1, fontsize=10)

        ax2 = plt.subplot(122, aspect='equal')    
        CS = plt.contour(sample_X, sample_Y, sample_Z2.reshape(sample_X.shape))
        plt.clabel(CS, inline=1, fontsize=10)

        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim)

        ax2.set_xlim(xlim)
        ax2.set_ylim(ylim)

        plt.show()

        #print gmm_1.get_params(deep=True)
        print self.gmm_0.weights_
        print self.gmm_0.means_
        print self.gmm_0.covars_

            

        

if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()

    # For default options
    p.add_option('--dir', action='store', dest='data_path',
                 type='string', 
                 help='top level directory with sub-directories for each reach problem.')
    p.add_option('--server', action='store_true', dest='server',
                 help='Launch server')

    # For a robot selection
    p.add_option('--rbt', '--robot', action='store', dest='rbt',
                 type='string', default='sim3', help='Choose a robot')

    
    p.add_option('--renew', action='store_true', dest='renew',
                 default=False, help='Renew pickle files.')
    p.add_option('--cross_val', '--cv', action='store_true', dest='bCrossVal',
                 default=False, help='N-fold cross validation for parameter')             
    
    opt, args = p.parse_args()

    
    ## Init variables
    s_robot   = 'sim3'
    data_path = '/home/dpark/hrl_file_server/dpark_data/map/ICRA2015/2D/TRAIN_CYLIN_F40_M40_2'

    n_components = 8
    b_dim_reduction = True
    b_dim_reduction_renew = False
    dr_method = 'fa'
    covariance_type = 'diag'
    
    ######################################################    
    # Get Raw Data
    lg = learning_gmm_lic1(s_robot, data_path, b_dim_reduction=b_dim_reduction, b_dim_reduction_renew=b_dim_reduction_renew, dr_method=dr_method, gmm_0__n_components=n_components, gmm_1__n_components=n_components, gmm_0__covariance_type=covariance_type, gmm_1__covariance_type=covariance_type, bRenew=opt.renew)

    ######################################################    
    # Training and Prediction
    if opt.server:
        lg.fit(lg.aXData, lg.aYData)        
        
        rospy.init_node('learning_gmm_1st_node')       
        rospy.Service('/learning/gmm_1st', FloatArray_FloatArray, lg.service)
        print "Service Start!!"
        rospy.spin()

    elif opt.bCrossVal:

        tuned_parameters = [{'gmm_0__n_components': [4,8,12], 'gmm_0__covariance_type': ['full', 'diag'], 'b_dim_reduction': [True, False], 'b_dim_reduction_renew': [True],
                            'dr_method': ['fa', 'pca', 'kpca', 'isomap'], 'kpca_gamma': [0.1, 1.0, 10.0], 
                            'isomap_neighbors': [3,5,10]}]
            
        lg.param_estimation(tuned_parameters, 10)

    else:
        lg.fit(lg.aXData, lg.aYData)        
        ## lg.score(lg.aXData[1:10,:], lg.aYData[1:10])

        ## goal = np.array([0.6, 0.0])
        ## lg.optimization(goal)

        lg.test2()
        ## lg.test3()

    
    print "-----------------------------------"
    ## DR.plot_embedding(dr_X,y,xlim=[-1.0,1.0], ylim=[-1.0,1.0])
    ## xlim=[-2.0,2.0]
    ## ylim=[-2.0,2.0]
    ## DR.plot_embedding(dr_X,y,dim=2,xlim=xlim, ylim=ylim)



    
