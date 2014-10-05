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
import cPickle
from sklearn.externals import joblib

# Matplot
import matplotlib
import matplotlib.pyplot as plt

## import door_open_data as dod
import ghmm
import hrl_anomaly_detection.mechanism_analyse_daehyung as mad

class learning_hmm():
    def __init__(self, data_path, nState, nStep):

        ## self.model = hmm.GaussianHMM(3, "full", self.startprob, self.transmat)

        self.nState= nState
        self.nStep = nStep

        # emission domain of this model        
        self.F = ghmm.Float()  
        
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
        
        # Transition probability matrix
        A, _ = mad.get_trans_mat(X, self.nState)

        # We should think about multivariate Gaussian pdf.        
        mu, sigma = self.vectors_to_mean_vars(X)

        # Emission probability matrix
        B = np.hstack([mu, sigma])
        B = B.tolist()
        
        # pi - initial probabilities per state
        pi = [1.0/float(self.nState)] * self.nState
        
        # HMM model object
        self.ml = ghmm.HMMFromMatrices(self.F, ghmm.GaussianDistribution(self.F), A, B, pi)
        ## self.ml = ghmm.HMMFromMatrices(self.F, ghmm.DiscreteDistribution(self.F), A, B, pi)
        
        print "Run Baum Welch method with ", X.T.shape
        train_seq = X.T.tolist()
        final_seq = ghmm.SequenceSet(self.F, train_seq)        
        self.ml.baumWelch(final_seq)

        ## self.mean_path_plot(mu[:,0], sigma[:,0])        
        print "Completed to fitting"
        

    #----------------------------------------------------------------------        
    #
    def vectors_to_mean_vars(self, vecs):

        n,k   = vecs.shape # length, samples
        mu    = np.zeros((self.nStep,1))
        sigma = np.zeros((self.nStep,1))

        nDivs = int(n/float(self.nStep))

        index = 0
        while (index < self.nStep):
            m_init = index*nDivs
            temp_vec = vecs[(m_init):(m_init+nDivs)]

            mu[index] = np.mean(temp_vec)
            sigma[index] = np.std(temp_vec)
            index = index+1

        return mu,sigma
        

    #----------------------------------------------------------------------        
    #
    def predict(self, X_test):

        final_ts_obj = ghmm.EmissionSequence(self.F,X_test) # is it neccessary?

        alpha_Z_n = self.ml.forward(final_ts_obj)

        
        for x in alpha_Z_n:
            print np.array(x).shape
            
        print len(final_ts_obj)
        print len(alpha_Z_n)
        ## path_obj = self.ml.viterbi(final_ts_obj)

        return path_obj

    #----------------------------------------------------------------------        
    #
    def save_obs(self, obs_name):

        self.ml.write(obs_name)

    #----------------------------------------------------------------------        
    #
    def load_obs(self, obs_name):

        self.ml.write(obs_name)
        
    #----------------------------------------------------------------------        
    #
    def load_obs(self, obs_name):

        try:     
            self.ml = joblib.load(obs_name)        
        except:
            rospy.signal_shutdown('Failed to load learned object. Retry!')                            
            time.sleep(1.0)                
            sys.exit()

    
    #----------------------------------------------------------------------        
    #
    def path_plot(self, X_test, path_obj):

        fig = plt.figure(1)
        ax = fig.add_subplot(111)

        for i in xrange(X_test.shape[1]):
            ax.plot(X_test[:,i], '-')    
            
        ax.set_xlabel("Angle")
        ax.set_ylabel("Force")

        ## plt.show()
        fig.savefig('/home/dpark/Dropbox/HRL/collision_detection_hsi_kitchen_pr2.pdf', format='pdf')
        

    #----------------------------------------------------------------------        
    #
    def mean_path_plot(self, mu, var):

        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        
        ax.plot(mu, 'k-')    

        x_axis = np.linspace(0,len(mu),len(mu))
        ax.fill_between(x_axis, mu-var**2, mu+var**2, facecolor='green', alpha=0.5)
            
        ax.set_xlabel("Angle")
        ax.set_ylabel("Force")

        ## plt.show()
        fig.savefig('/home/dpark/Dropbox/HRL/collision_detection_hsi_kitchen_pr2.pdf', format='pdf')
        

        

if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    opt, args = p.parse_args()

    ## Init variables    
    data_path = os.getcwd()
    nState    = 10
    nStep     = 36
    pkl_file  = "door_opening_data.pkl"    

    ######################################################    
    # Get Raw Data
    if os.path.isfile(pkl_file):
        print "Saved pickle found"
        data = ut.load_pickle(pkl_file)
        data_vecs = data['data_vecs']
        data_mech = data['data_mech']
        data_chunks = data['data_chunks']
    else:        
        print "No saved pickle found"        
        data_vecs, data_mech, data_chunks = mad.get_all_blocked_detection()
        data = {}
        data['data_vecs'] = data_vecs
        data['data_mech'] = data_mech
        data['data_chunks'] = data_chunks
        ut.save_pickle(data,pkl_file)
        
        
    data_vecs = np.array([data_vecs])
    data_vecs[0] = mad.approx_missing_value(data_vecs[0])

    

    ######################################################    
    # Training and Prediction
    lh = learning_hmm(data_path, nState, nStep)

    lh.fit(data_vecs[0])    
    ## lh.path_plot(data_vecs[0], data_vecs[0,:,3])

    x_test = data_vecs[0][1:10,5].tolist()

    traj = lh.predict(x_test)
    ## print x_test
    ## print traj
