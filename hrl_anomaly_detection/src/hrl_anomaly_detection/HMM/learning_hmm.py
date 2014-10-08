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
## import cPickle
## from sklearn.externals import joblib

# Matplot
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec

## import door_open_data as dod
import ghmm
import hrl_anomaly_detection.mechanism_analyse_daehyung as mad
from scipy.stats import norm

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
        self.mu, self.sigma = self.vectors_to_mean_vars(X)

        # Emission probability matrix
        B = np.hstack([self.mu, self.sigma])
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
    def predict(self, X_test, X_predict):

        # Past profile
        final_ts_obj = ghmm.EmissionSequence(self.F,X_test) # is it neccessary?

        ## print "\nForward"
        ## logp1 = self.ml.loglikelihood(final_ts_obj)
        ## print "logp = " + str(logp1) + "\n"
        
        # alpha: X_test length x #latent States at the moment t when state i is ended
        (alpha,scale) = self.ml.forward(final_ts_obj)
        ## print "alpha: ", np.array(alpha).shape,"\n" + str(alpha) + "\n"
        ## print "scale = " + str(scale) + "\n"

        # beta
        beta = self.ml.backward(final_ts_obj,scale)
        ## print "beta", np.array(beta).shape, " = \n " + str(beta) + "\n"

        pred_numerator = 0.0
        pred_denominator = 0.0
        for i in xrange(self.nState): # N+1

            total = 0.0        
            for j in xrange(self.nState): # N                  
                total += self.ml.getTransition(j,i) * alpha[-1][j]
                
            (mu, sigma) = self.ml.getEmission(i)

            pred_numerator += norm(loc=mu,scale=sigma).pdf(X_predict) * total
            pred_denominator += alpha[-1][i]*beta[-1][i]

        ## for i in xrange(len(alpha)):
        ##     print X_test[i]

        ## for i in xrange(self.nState):            
        ##     print alpha[-1][i]
            
        return pred_numerator / pred_denominator
            

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
    def all_path_plot(self, X_test):
        self.fig = plt.figure(1)
        self.ax = self.fig.add_subplot(111)
        
        for i in xrange(X_test.shape[0]):
            self.ax.plot(X_test[i], '-')    
        
    #----------------------------------------------------------------------        
    #
    def predictive_path_plot(self, X_test, X_pred, X_pred_prob, x_test_next):

        self.fig = plt.figure(1)
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1]) 

        ## Main predictive distribution
        self.ax = self.fig.add_subplot(gs[0])

        # 1) Observation
        self.ax.plot(X_test, 'k-o')    
        ## self.ax.plot(X_test, '-o', ms=10, lw=1, alpha=0.5, mfc='orange')    

        # 2) Next true observation
        x_array = np.arange(len(X_test)-1,len(X_test)+0.0001,1.0)
        y_array = np.array([X_test[-1], x_test_next])                        
        self.ax.plot(x_array, y_array, 'k-', linewidth=1.0)    
        
        # 3) Prediction        
        for x, x_pred in zip(X_pred, X_pred_prob):
            y_array = np.array([X_test[-1], x])                        
            alpha   = x_pred / 2.0 + 0.5
            if x_pred > 1.0: x_pred = 1.0
            self.ax.plot(x_array, y_array, 'r-', alpha=x_pred**1., linewidth=1.0)    

        ## Side distribution
        self.ax1 = self.fig.add_subplot(gs[1])
        self.ax1.plot(X_pred_prob, X_pred, 'r-')

        ## self.ax1.tick_params(\
        ##                      axis='y',          # changes apply to the x-axis
        ##                      which='both',      # both major and minor ticks are affected
        ##                      left='off',        # ticks along the bottom edge are off                             
        ##                      bottom='off',      # ticks along the bottom edge are off
        ##                      top='off',         # ticks along the top edge are off
        ##                      labelleft='off',   # labels along the bottom edge are off                             
        ##                      labelbottom='off') # labels along the bottom edge are off

        self.ax1.set_xlabel("Probability of Next Obervation")
            

    #----------------------------------------------------------------------        
    #
    def mean_path_plot(self, mu, var):

        self.fig = plt.figure(1)
        self.ax = self.fig.add_subplot(111)
        
        self.ax.plot(mu, 'k-')    

        x_axis = np.linspace(0,len(mu),len(mu))

        ## print x_axis.shape, mu.shape, var.shape
        
        self.ax.fill_between(x_axis, mu[:,0]-var[:,0]**2, mu[:,0]+var[:,0]**2, facecolor='green', alpha=0.5)
                    

    #----------------------------------------------------------------------        
    #
    def final_plot(self):
        self.ax.set_xlabel("Angle")
        self.ax.set_ylabel("Force")
        
        plt.show()
        ## self.fig.savefig('/home/dpark/Dropbox/HRL/collision_detection_hsi_kitchen_pr2.pdf', format='pdf')

if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    opt, args = p.parse_args()

    ## Init variables    
    data_path = os.getcwd()
    nState    = 15
    nStep     = 36
    pkl_file  = "door_opening_data.pkl"    

    ######################################################    
    # Get Training Data
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

    # Filtering
    idxs = np.where(['Office Cabinet' in i for i in data_mech])[0].tolist()

    print data_mech
    print data_vecs.shape, np.array(data_mech).shape, np.array(data_chunks).shape
    data_vecs = data_vecs[:,idxs]
    data_mech = [data_mech[i] for i in idxs]
    data_chunks = [data_chunks[i] for i in idxs]
    print data_vecs.shape, np.array(data_mech).shape, np.array(data_chunks).shape
    
    data_vecs = np.array([data_vecs])
    data_vecs[0] = mad.approx_missing_value(data_vecs[0])
    

    ######################################################    
    # Training 
    lh = learning_hmm(data_path, nState, nStep)

    lh.fit(data_vecs[0])    
    ## lh.path_plot(data_vecs[0], data_vecs[0,:,3])

    ######################################################    
    # Test data
    h_config, h_ftan = mad.get_a_blocked_detection()
    ## print np.array(h_config)*180.0/3.14
    print len(h_ftan)
    
    
    for i in xrange(1,2,2):
        ## print i
        ## x_test      = data_vecs[0][:12,i].tolist()
        ## x_test_next = data_vecs[0][12:12+1,i]
        x_test = h_ftan[:15]
        x_test_next = h_ftan[15:16][0]
        
        # Future profile
        future_obsrv = np.arange(0.0, data_vecs[0][:,i].max()*1.5, 0.2)

        future_prob = np.zeros(future_obsrv.shape)
        for i in xrange(len(future_obsrv)):           
            future_prob[i] = lh.predict(x_test, future_obsrv[i]) 

        lh.predictive_path_plot(np.array(x_test), future_obsrv, future_prob, x_test_next)
        lh.final_plot()

    ## print lh.mean_path_plot(lh.mu, lh.sigma)
        
    ## print x_test
    ## print x_test[-4:]

    ## fig = plt.figure(1)
    ## ax = fig.add_subplot(111)

    ## ax.plot(future_obsrv, future_prob)
    ## plt.show()

