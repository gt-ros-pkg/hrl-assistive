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

## import door_open_data as dod
import ghmm
import hrl_anomaly_detection.mechanism_analyse_daehyung as mad
from scipy.stats import norm

from learning_base import learning_base


class learning_hmm(learning_base):
    def __init__(self, data_path, aXData, nState, nMaxStep, nFutureStep=5, fObsrvResol=0.2):

        learning_base.__init__(self, data_path, aXData)
        
        ## Tunable parameters                
        self.nState= nState # the number of hidden states
        self.nFutureStep = nFutureStep
        self.fObsrvResol = fObsrvResol

        ## Un-tunable parameters
        self.nMaxStep = nMaxStep  # the length of profile
        self.future_obsrv = None  # Future observation range
        
        # emission domain of this model        
        self.F = ghmm.Float()  
        
        # Confusion Matrix NOTE ???
        ## cmat = np.zeros((4,4))

        # Assign local functions
        learning_base.__dict__['fit'] = self.fit        
        learning_base.__dict__['predict'] = self.predict
        learning_base.__dict__['score'] = self.score                
        pass


    #----------------------------------------------------------------------        
    #
    def fit(self, X_train, Y_train=None):
        
        # Transition probability matrix (Initial transition probability, TODO?)
        A, _ = mad.get_trans_mat(X_train, self.nState)

        # We should think about multivariate Gaussian pdf.        
        self.mu, self.sigma = self.vectors_to_mean_vars(X_train)

        # Emission probability matrix
        B = np.hstack([self.mu, self.sigma]) # Must be [i,:] = [mu, sigma]
        ## B = B.T.tolist()
        
        # pi - initial probabilities per state 
        pi = [1.0/float(self.nState)] * self.nState

        # HMM model object
        self.ml = ghmm.HMMFromMatrices(self.F, ghmm.GaussianDistribution(self.F), A, B, pi)
        ## self.ml = ghmm.HMMFromMatrices(self.F, ghmm.DiscreteDistribution(self.F), A, B, pi)
        
        print "Run Baum Welch method with (samples, length)", X_train.shape
        train_seq = X_train.tolist()
        final_seq = ghmm.SequenceSet(self.F, train_seq)        
        self.ml.baumWelch(final_seq)

        ## self.mean_path_plot(mu[:,0], sigma[:,0])        
        print "Completed to fitting", np.array(final_seq).shape

        # Future observation range
        self.max_obsrv = X_train.max()
        self.obsrv_range = np.arange(0.0, self.max_obsrv*1.5, self.fObsrvResol)

        
    #----------------------------------------------------------------------        
    #
    def vectors_to_mean_vars(self, vecs):

        _,n   = vecs.shape # samples, length
        mu    = np.zeros((self.nState,1))
        sigma = np.zeros((self.nState,1))

        nDivs = int(n/float(self.nState))

        index = 0
        while (index < self.nState):
            m_init = index*nDivs
            temp_vec = vecs[:,(m_init):(m_init+nDivs)]

            mu[index] = np.mean(temp_vec)
            sigma[index] = np.std(temp_vec)
            index = index+1

        return mu,sigma
        

    #----------------------------------------------------------------------        
    # Compute the estimated probability (0.0~1.0)
    def multi_step_predict(self, X_test):
        # Input: X, X_{N+M-1}, P(X_{N+M-1} | X)
        # Output:  P(X_{N+M} | X)

        # Initialization            
        X_pred_prob = np.zeros((len(self.obsrv_range),self.nFutureStep))
        X = copy.copy(X_test)
        X_pred = [0.0]*self.nFutureStep

        # Recursive prediction
        for i in xrange(self.nFutureStep):

            # Get all probability
            for j in xrange(len(self.obsrv_range)):           
                X_pred_prob[j][i] += self.predict(X+[self.obsrv_range[j]])             

            # Select observation with maximum probability
            idx_list = [k[0] for k in sorted(enumerate(X_pred_prob[:,i]), key=lambda x:x[1], reverse=True)]
                              
            # Udate 
            X.append(self.obsrv_range[idx_list[0]])
            X_pred[i] = self.obsrv_range[idx_list[0]]
            
        return X_pred, X_pred_prob


    #----------------------------------------------------------------------        
    #
    def predict(self, X):
        # Input
        # @ X_test: N length array
        # @ x_pred: scalar
        # Output
        # @ probability

        X_test = X[:-1]
        x_pred = X[-1]

        # Past profile
        final_ts_obj = ghmm.EmissionSequence(self.F,X_test) # is it neccessary?

        ## print "\nForward"
        ## logp1 = self.ml.loglikelihood(final_ts_obj)
        ## print "logp = " + str(logp1) + "\n"
        
        # alpha: X_test length y #latent States at the moment t when state i is ended
        #        test_profile_length x number_of_hidden_state
        (alpha,scale) = self.ml.forward(final_ts_obj)
        ## print "alpha: ", np.array(alpha).shape,"\n" + str(alpha) + "\n"
        ## print "scale = " + str(scale) + "\n"
        ## print np.array(X_test).shape, np.array(alpha).shape
        
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

            pred_numerator += norm(loc=mu,scale=sigma).pdf(x_pred) * total
            pred_denominator += alpha[-1][i]*beta[-1][i]

        return pred_numerator / pred_denominator
                        

    #----------------------------------------------------------------------
    # Returns the mean accuracy on the given test data and labels.
    def score(self, X_test, X_test_next, sample_weight=None):

        ## from sklearn.metrics import accuracy_score
        ## return accuracy_score(y_test, np.around(self.predict(X_test)), sample_weight=sample_weight)

        from sklearn.metrics import r2_score
        X_pred, _ = self.multi_step_predict(X_test)
        print X_test_next, X_pred
        
        return r2_score(X_test_next, X_pred, sample_weight=sample_weight)


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
    def all_path_plot(self, X_test):
        self.fig = plt.figure(1)
        self.ax = self.fig.add_subplot(111)
        
        for i in xrange(X_test.shape[0]):
            self.ax.plot(X_test[i], '-')    

            
    #----------------------------------------------------------------------        
    #
    def predictive_path_plot(self, X_test, X_pred, X_pred_prob, X_test_next):

        self.fig = plt.figure(1)
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1]) 

        ## Main predictive distribution
        self.ax = self.fig.add_subplot(gs[0])

        # 1) Observation        
        self.ax.plot(X_test, 'k-o')    
        ## self.ax.plot(X_test, '-o', ms=10, lw=1, alpha=0.5, mfc='orange')    

        # 2) Next predictive & true observation
        y_array = np.hstack([X_test[-1], X_test_next])                        
        x_array = np.arange(0, len(X_test_next)+0.0001,1.0) + len(X_test) -1        
        self.ax.plot(x_array, y_array, 'k-', linewidth=1.0)    

        y_array = np.hstack([X_test[-1], X_pred])                        
        self.ax.plot(x_array, y_array, 'b-', linewidth=2.0)    
        
        # 3) Prediction        
        n,m = X_pred_prob.shape
        x_last = X_test[-1]
        for i in xrange(m):
            
            x_pred_max = 0.0
            x_best     = 0.0            
            for x, prob in zip(self.obsrv_range, X_pred_prob[:,i]):
                y_array = np.array([x_last, x])
                
                alpha   = prob / 2.0 + 0.5
                if prob > 1.0: prob = 1.0
                self.ax.plot(x_array[i:i+2], y_array, 'r-', alpha=prob**1., linewidth=1.0)    

            x_last = X_pred[i]
                

        ## Side distribution
        self.ax1 = self.fig.add_subplot(gs[1])
        self.ax1.plot(X_pred_prob[:,-1], self.obsrv_range, 'r-')

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
    p.add_option('--renew', action='store_true', dest='renew',
                 default=False, help='Renew pickle files.')
    p.add_option('--cross_val', '--cv', action='store_true', dest='bCrossVal',
                 default=False, help='N-fold cross validation for parameter')             
    opt, args = p.parse_args()

    ## Init variables    
    data_path = os.getcwd()
    nState    = 36
    nMaxStep     = 36 # total step of data. It should be automatically assigned...
    pkl_file  = "door_opening_data.pkl"    
    nFutureStep = 2
    ## data_column_idx = 1
    fObsrvResol = 0.2

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

    ## print data_mech
    print data_vecs.shape, np.array(data_mech).shape, np.array(data_chunks).shape
    data_vecs = data_vecs[:,idxs]
    data_mech = [data_mech[i] for i in idxs]
    data_chunks = [data_chunks[i] for i in idxs]
    ## print data_vecs.shape, np.array(data_mech).shape, np.array(data_chunks).shape

    ## X data
    data_vecs = np.array([data_vecs.T]) # category x number_of_data x profile_length
    data_vecs[0] = mad.approx_missing_value(data_vecs[0])    

    ## ## time step data
    ## m, n = data_vecs[0].shape
    ## aXData = np.array([np.arange(0.0,float(n)-0.0001,1.0).tolist()] * m)
    
    ######################################################    
    # Training 
    lh = learning_hmm(data_path=data_path, aXData=data_vecs[0], nState=nState, nMaxStep=nMaxStep, nFutureStep=nFutureStep, fObsrvResol=fObsrvResol)


    if opt.bCrossVal:
        print "Cross Validation"
        tuned_parameters = [{'nState': [15,20,25,30,35], 'nFutureStep': [2,4,6,8,10], 'fObsrvResol': [0.05,0.1,0.15,0.2]}]
        lh.param_estimation(tuned_parameters, 10)
        
    else:
        lh.fit(lh.aXData)    
        ## lh.path_plot(data_vecs[0], data_vecs[0,:,3])

        ######################################################    
        # Test data
        ## h_config, h_ftan = mad.get_a_blocked_detection()
        ## print np.array(h_config)*180.0/3.14
        ## print len(h_ftan)


        for i in xrange(2,3,2):
            nProgress = 10
            x_test      = data_vecs[0][i,:nProgress].tolist()
            x_test_next = data_vecs[0][i,nProgress:nProgress+lh.nFutureStep].tolist()
            x_test_all  = data_vecs[0][i,:].tolist()
            ## x_test = h_ftan[:15]
            ## x_test_next = h_ftan[15:15+lh.nFutureStep]

            x_pred, x_pred_prob = lh.multi_step_predict(x_test)
            lh.predictive_path_plot(np.array(x_test), np.array(x_pred), x_pred_prob, np.array(x_test_next))
            lh.final_plot()


    ## print lh.mean_path_plot(lh.mu, lh.sigma)
        
    ## print x_test
    ## print x_test[-4:]

    ## fig = plt.figure(1)
    ## ax = fig.add_subplot(111)

    ## ax.plot(obsrv_range, future_prob)
    ## plt.show()




















    
