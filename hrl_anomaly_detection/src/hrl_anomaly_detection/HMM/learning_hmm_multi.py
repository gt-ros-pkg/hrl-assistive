#!/usr/local/bin/python

import sys, os, copy
import numpy as np, math
import scipy as scp
from scipy import optimize, interpolate
from scipy.stats import norm

import roslib; roslib.load_manifest('hrl_anomaly_detection')
import rospy
import inspect
import warnings
import random

# Util
import hrl_lib.util as ut
## import cPickle

# Matplot
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import animation

## import door_open_data as dod
import ghmm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score
from sklearn.base import clone
from sklearn import cross_validation
from joblib import Parallel, delayed
## from pysmac.optimize import fmin                
## from scipy.optimize import fsolve
## from scipy import interpolate

from learning_base import learning_base
import sandbox_dpark_darpa_m3.lib.hrl_dh_lib as hdl


class learning_hmm_multi(learning_base):
    def __init__(self, nState, nFutureStep=5, nCurrentStep=10, \
                 trans_type="left_right"):

        learning_base.__init__(self, trans_type)

        ## Tunable parameters                
        self.nState= nState # the number of hidden states
        self.nFutureStep = nFutureStep
        self.nCurrentStep = nCurrentStep
        
        ## Un-tunable parameters
        self.trans_type = trans_type #"left_right" #"full"
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
    def fit(self, aXData1, aXData2, A=None, B=None, pi=None, B_dict=None, verbose=False):

        if A is None:        
            if verbose: print "Generate new A matrix"                
            # Transition probability matrix (Initial transition probability, TODO?)
            A = self.init_trans_mat(self.nState).tolist()

        if B is None:
            if verbose: print "Generate new B matrix"                                            
            # We should think about multivariate Gaussian pdf.  

            self.mu1, self.mu2, self.cov = self.vectors_to_mean_cov(aXData1, aXData2, self.nState)
            
            # Emission probability matrix
            B = [0.0] * self.nState
            for i in range(self.nState):
                B[i] = [[self.mu1[i],self.mu2[i]],[self.cov[i][0][0],self.cov[i][0][1], \
                                                   self.cov[i][1][0],self.cov[i][1][1]]]       
                            
        if pi is None:            
            # pi - initial probabilities per state 
            ## pi = [1.0/float(self.nState)] * self.nState
            pi = [0.] * self.nState
            pi[0] = 1.0


        # Training input
        X_train  = self.convert_sequence(aXData1, aXData2)
        X_scaled = self.scaling(X_train)

        # HMM model object
        self.ml = ghmm.HMMFromMatrices(self.F, ghmm.MultivariateGaussianDistribution(self.F), A, B, pi)

        print "Run Baum Welch method with (samples, length)", X_scaled.shape        
        train_seq = X_scaled.tolist()
        final_seq = ghmm.SequenceSet(self.F, train_seq)        
        self.ml.baumWelch(final_seq)
        ## self.ml.baumWelch(final_seq, 10000)

        [self.A,self.B,self.pi] = self.ml.asMatrices()
        self.A = np.array(self.A)
        self.B = np.array(self.B)

        ## print "----------------------"
        ## seq = self.ml.sample(20, len(aXData1[0]), seed=3586663)
        ## seq = np.array(seq)
        ## X1, X2 = self.convert_sequence_reverse(seq)


        ## mu = np.array(self.B[:,0])
        ## cov = np.array(self.B[:,1])
        ## mu1 = []
        ## mu2 = []
        ## cov11=[]
        ## cov12=[]
        ## cov21=[]
        ## cov22=[]
            
        ## for i in xrange(len(mu)):
        ##     mu1.append(mu[i][0])
        ##     mu2.append(mu[i][1])
        ##     cov11.append(cov[i][0])
        ##     cov22.append(cov[i][3])
            
        ## plt.figure()
        ## plt.subplot(211)
        ## plt.plot(aXData1[0],'r')
        ## ## plt.plot(self.mu1,'b')
        ## ## plt.plot(mu1+np.sqrt(cov11),'b')
        ## for j in xrange(len(X1)):
        ##     plt.plot(X1[j],'b')
        ## plt.subplot(212)
        ## plt.plot(aXData2[0],'r')        
        ## ## plt.plot(self.mu2,'b')
        ## ## plt.plot(mu2+np.sqrt(cov22),'b')
        ## for j in xrange(len(X2)):
        ##     plt.plot(X2[j],'b')
        ## ## plt.subplot(313)
        ## plt.show()
        ## sys.exit()
        
        # state range
        self.state_range = np.arange(0, self.nState, 1)
        return


    #----------------------------------------------------------------------        
    #
    def predict(self, X):

        n,m    = X.shape
        X_test = X[0].tolist()
        mu_l  = np.zeros(2) 
        cov_l = np.zeros(4)

        final_ts_obj = ghmm.EmissionSequence(self.F, X_test) # is it neccessary?

        try:
            # alpha: X_test length y #latent States at the moment t when state i is ended
            #        test_profile_length x number_of_hidden_state
            (alpha,scale) = self.ml.forward(final_ts_obj)
            alpha         = np.array(alpha)
            scale         = np.array(scale)
            print "alpha: ", np.array(alpha).shape,"\n" #+ str(alpha) + "\n"
            ## print "scale = " + str(scale) + "\n"
        except:
            print "No alpha is available !!"

        f = lambda x: round(x,12)
        for i in range(len(alpha)):
            alpha[i] = map(f, alpha[i])
        ## alpha[-1] = map(f, alpha[-1])

        ## print scale
        print alpha[-2,:]
        print alpha[-1,:]
            
        pred_numerator = 0.0
        ## pred_denominator = 0.0
        for j in xrange(self.nState): # N+1

            total = np.sum(self.A[:,j]*alpha[-1,:]) #* scaling_factor
            [[mu1, mu2], [cov11, cov12, cov21, cov22]] = self.B[j]

            ## print mu1, mu2, cov11, cov12, cov21, cov22, total
            
            mu_l[0] += mu1*total
            mu_l[1] += mu2*total
            cov_l[0] += (cov11)*(total**2)
            cov_l[1] += (cov12)*(total**2)
            cov_l[2] += (cov21)*(total**2)
            cov_l[3] += (cov22)*(total**2)

        return mu_l, cov_l
    

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
        
        
    #----------------------------------------------------------------------        
    #                
    # Returns mu,sigma for n hidden-states from feature-vector
    def vectors_to_mean_cov(self,vec1, vec2, nState): 

        index = 0
        m,n = np.shape(vec1)
        #print m,n
        mu_1 = np.zeros(nState)
        mu_2 = np.zeros(nState)
        cov = np.zeros((nState,2,2))
        DIVS = n/nState

        while (index < nState):
            m_init = index*DIVS
            temp_vec1 = vec1[:,(m_init):(m_init+DIVS)]
            temp_vec2 = vec2[:,(m_init):(m_init+DIVS)]
            temp_vec1 = np.reshape(temp_vec1,(1,DIVS*m))
            temp_vec2 = np.reshape(temp_vec2,(1,DIVS*m))
            mu_1[index] = np.mean(temp_vec1)
            mu_2[index] = np.mean(temp_vec2)
            cov[index,:,:] = np.cov(np.concatenate((temp_vec1,temp_vec2),axis=0))
            ## if index == 0:
            ##     print 'mean = ', mu_2[index]
            ##     print 'mean = ', scp.mean(vec2[0:,(m_init):(m_init+DIVS)])
            ##     print np.shape(np.concatenate((temp_vec1,temp_vec2),axis=0))
            ##     print cov[index,:,:]
            ##     print scp.std(vec2[0:,(m_init):(m_init+DIVS)])
            ##     print scp.std(temp_vec2)
            index = index+1

        return mu_1,mu_2,cov

    
    def vectors_to_mean_cov2(self,vec1, vec2, nState): 

        if len(vec1[0]) != len(vec2[0]):
            print "data length different!!! ", len(vec1[0]), len(vec2[0])
            sys.exit()

                    
        index = 0
        m,n = np.shape(vec1)
        ## print m,n
        mult = 2

        o_x    = np.arange(0.0, n, 1.0)
        o_mu1  = scp.mean(vec1, axis=0)
        o_sig1 = scp.std(vec1, axis=0)
        o_mu2  = scp.mean(vec2, axis=0)
        o_sig2 = scp.std(vec2, axis=0)
        o_cov  = np.zeros((n,2,2))

        for i in xrange(n):
            o_cov[i] = np.cov(np.concatenate((vec1[:,i],vec2[:,i]),axis=0))
        
        f_mu1  = interpolate.interp1d(o_x, o_mu1, kind='linear')
        f_sig1 = interpolate.interp1d(o_x, o_sig1, kind='linear')
        f_mu2  = interpolate.interp1d(o_x, o_mu2, kind='linear')
        f_sig2 = interpolate.interp1d(o_x, o_sig2, kind='linear')

        f_cov11 = interpolate.interp1d(o_x, o_cov[:,0,0], kind='linear')
        f_cov12 = interpolate.interp1d(o_x, o_cov[:,0,1], kind='linear')
        ## f_cov21 = interpolate.interp1d(o_x, o_cov[:,1,0], kind='linear')
        f_cov22 = interpolate.interp1d(o_x, o_cov[:,1,1], kind='linear')

            
        x = np.arange(0.0, float(n-1)+1.0/float(mult), 1.0/float(mult))
        mu1  = f_mu1(x)
        sig1 = f_sig1(x)
        mu2  = f_mu2(x)
        sig2 = f_sig2(x)

        cov11 = f_cov11(x)
        cov12 = f_cov12(x)
        cov21 = f_cov12(x)
        cov22 = f_cov22(x)
        
        while len(mu1) != nState:

            d_mu1  = np.abs(mu1[1:] - mu1[:-1]) # -1 length 
            d_sig1 = np.abs(sig1[1:] - sig1[:-1]) # -1 length 
            idx = d_sig1.tolist().index(min(d_sig1))
            
            mu1[idx]  = (mu1[idx]+mu1[idx+1])/2.0
            sig1[idx] = np.sqrt(sig1[idx]**2+sig1[idx+1]**2)
            mu2[idx]  = (mu2[idx]+mu2[idx+1])/2.0
            sig2[idx] = np.sqrt(sig2[idx]**2+sig2[idx+1]**2)
            
        
            mu  = scp.delete(mu,idx+1)
            sig = scp.delete(sig,idx+1)

        mu = mu.reshape((len(mu),1))
        sig = sig.reshape((len(sig),1))

        
        ## import matplotlib.pyplot as pp
        ## pp.figure()
        ## pp.plot(mu)
        ## pp.plot(mu+1.*sig)
        ## pp.plot(scp.mean(vec, axis=0), 'r')
        ## pp.show()
        ## sys.exit()

        return mu,sig


    #----------------------------------------------------------------------        
    #
    def init_plot(self, bAni=False):
        print "Start to print out"

        self.fig = plt.figure(1)
        gs = gridspec.GridSpec(2, 1) 
        
        self.ax1 = self.fig.add_subplot(gs[0])        
        self.ax2 = self.fig.add_subplot(gs[1])        

    #----------------------------------------------------------------------        
    #
    def pred_plot(self, X_test1, X_test2):

        X_test = self.convert_sequence(X_test1, X_test2)
        mu, cov = self.predict(X_test)

        ## print mu
        ## print cov
        
        ## Main predictive distribution        
        self.ax1.plot(np.hstack([X_test1[0], mu[0]]))
        self.ax2.plot(np.hstack([X_test2[0], mu[1]]))
        

    #----------------------------------------------------------------------        
    #
    def final_plot(self):
        plt.rc('text', usetex=True)
        
        ## self.ax1.set_xlabel(r'\textbf{Angle [}{^\circ}\textbf{]}', fontsize=22)
        ## self.ax1.set_ylabel(r'\textbf{Applied Opening Force [N]}', fontsize=22)
        ## self.ax.set_xlim([0, self.nMaxStep])
        ## self.ax.set_ylim(self.obsrv_range)
        
        plt.show()


    #----------------------------------------------------------------------        
    #
    def convert_sequence(self, X1, X2):

        n,m = X1.shape

        X = []
        for i in xrange(n):
            Xs = []
            for j in xrange(m):
                Xs.append([X1[i,j], X2[i,j]])
            X.append(Xs)
            
        return X
        
    #----------------------------------------------------------------------        
    #
    def convert_sequence_reverse(self, X):

        m,n = X.shape
        X1 = np.zeros((m,n/2))
        X2 = np.zeros((m,n/2))
        for i in xrange(n/2):
            X1[:,i:i+1] = X[:,i*2:i*2+1] 
            X2[:,i:i+1] = X[:,i*2+1:i*2+2] 

        return X1, X2
        
