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
from matplotlib import rc

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
                 trans_type="left_right", nEmissionDim=2):

        learning_base.__init__(self, trans_type)

        ## Tunable parameters                
        self.nState= nState # the number of hidden states
        self.nFutureStep = nFutureStep
        self.nCurrentStep = nCurrentStep
        self.nEmissionDim = nEmissionDim
        
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
    def fit(self, aXData1, aXData2=None, A=None, B=None, pi=None, cov_mult=[1.0, 1.0, 1.0, 1.0], \
            B_dict=None, verbose=False):

        X1 = np.array(aXData1)
        X2 = np.array(aXData2)
            
        if A is None:        
            if verbose: print "Generate new A matrix"                
            # Transition probability matrix (Initial transition probability, TODO?)
            A = self.init_trans_mat(self.nState).tolist()

        if B is None:
            if verbose: print "Generate new B matrix"                                            
            # We should think about multivariate Gaussian pdf.  

            if self.nEmissionDim == 1:
                self.mu, self.sig = self.vectors_to_mean_sigma(X1, self.nState)
                B = np.vstack([self.mu, self.sig*cov_mult[0]]).T.tolist() # Must be [i,:] = [mu, sig]
            else:
                self.mu1, self.mu2, self.cov = self.vectors_to_mean_cov(X1, X2, self.nState)
                self.cov[:,0,0] *= cov_mult[0] #1.5 # to avoid No convergence warning
                self.cov[:,1,0] *= cov_mult[1] #5.5 # to avoid No convergence warning
                self.cov[:,0,1] *= cov_mult[2] #5.5 # to avoid No convergence warning
                self.cov[:,1,1] *= cov_mult[3] #5.5 # to avoid No convergence warning

                # Emission probability matrix
                B = [0.0] * self.nState
                for i in range(self.nState):
                    B[i] = [[self.mu1[i],self.mu2[i]],[self.cov[i,0,0],self.cov[i,0,1], \
                                                       self.cov[i,1,0],self.cov[i,1,1]]]       
                                            
        if pi is None:            
            # pi - initial probabilities per state 
            ## pi = [1.0/float(self.nState)] * self.nState
            pi = [0.0] * self.nState
            pi[0] = 1.0

        # HMM model object
        if self.nEmissionDim==1:
            self.ml = ghmm.HMMFromMatrices(self.F, ghmm.GaussianDistribution(self.F), A, B, pi)
            X_train = X1.tolist()
        else:
            self.ml = ghmm.HMMFromMatrices(self.F, ghmm.MultivariateGaussianDistribution(self.F), A, B, pi)
            X_train = self.convert_sequence(X1, X2) # Training input
            X_train = X_train.tolist()
        
            
        print "Run Baum Welch method with (samples, length)", np.shape(X_train)                        
        final_seq = ghmm.SequenceSet(self.F, X_train)        
        ## ret = self.ml.baumWelch(final_seq, loglikelihoodCutoff=2.0)
        ret = self.ml.baumWelch(final_seq, 10000)
        print "baumwelch return : ", ret

        [self.A,self.B,self.pi] = self.ml.asMatrices()
        self.A = np.array(self.A)
        self.B = np.array(self.B)

        # Get average loglikelihood threshold
        ## print "Compute loglikihood threshold"
        n,m = np.shape(X1)
        ll_list = []
        for i in xrange(self.nState):
            ll_list.append([])
        self.ll_mu  = np.zeros(self.nState)
        self.ll_std = np.zeros(self.nState)

        for i in xrange(n):                                                          
            for j in xrange(m):
                if j==0: continue

                final_ts_obj = ghmm.EmissionSequence(self.F, X_train[i][:j*self.nEmissionDim])
                ## final_ts_obj = ghmm.EmissionSequence(self.F, X_train[i,:j*self.nEmissionDim].tolist())
                path,logp    = self.ml.viterbi(final_ts_obj)

                if len(path) == 0: continue
                ll_list[path[-1]].append(logp)
        print "Executed viterbi algorithm: ", np.shape(X1)

                
        for i, ll in enumerate(ll_list):
            if len(ll) > 0: 
                self.ll_mu[i]  = np.array(ll).mean()
                self.ll_std[i] = np.sqrt(np.array(ll).var())
            else:
                self.ll_mu[i]  = self.ll_mu[i-1]
                self.ll_std[i] = self.ll_std[i-1]

        ## print "---------------------"                
        ## print "Avg log likelihoods"
        ## print "---------------------"
        ## print self.ll_std
        ## print self.ll_mu
        ## print "---------------------"
        
            
            ## path          = self.ml.viterbi(final_ts_obj)
            ## (alpha,scale) = self.ml.forward(final_ts_obj)
            ## alpha         = np.array(alpha)
            ## scale         = np.array(scale)

            ## print "---------------------"
            ## temp = 100
            ## p = self.loglikelihood(X_train[i:i+1,:temp])
            ## final_ts_obj  = ghmm.EmissionSequence(self.F, X_train[i,:temp].tolist())                
            ## (alpha,scale) = self.ml.forward(final_ts_obj)
            ## alpha         = np.array(alpha)
            ## scale         = np.array(scale)

            ## print p, np.sum(np.log(scale[:temp/2]))

            ## a = np.log(np.sum(alpha[:temp/2],axis=1)*scale[:temp/2])
            ## print np.sum(a)


        # state range
        ## self.state_range = np.arange(0, self.nState, 1)
        ## self.x1_range = np.linspace(np.min(aXData1), np.max(aXData1), 100)
        ## self.x2_range = np.linspace(np.min(X2), np.max(aXData2), 100)
        return


    #----------------------------------------------------------------------        
    #
    def predict(self, X):

        ## n,m    = np.array(X).shape
        X = np.squeeze(X)
        X_test = X.tolist()
        ## print "Input: ", np.shape(X_test)        
        
        mu_l  = np.zeros(2) 
        cov_l = np.zeros(4)

        final_ts_obj = ghmm.EmissionSequence(self.F, X_test) # is it neccessary?

        try:
            # alpha: X_test length y #latent States at the moment t when state i is ended
            #        test_profile_length x number_of_hidden_state
            (alpha,scale) = self.ml.forward(final_ts_obj)
            alpha         = np.array(alpha)
            scale         = np.array(scale)
            ## print "alpha: ", np.array(alpha).shape,"\n" #+ str(alpha) + "\n"
            ## print "scale = " + str(scale) + "\n"
        except:
            print "No alpha is available !!"
            
        f = lambda x: round(x,12)
        for i in range(len(alpha)):
            alpha[i] = map(f, alpha[i])
        alpha[-1] = map(f, alpha[-1])
        
        n = len(X_test)
        pred_numerator = 0.0
        ## pred_denominator = 0.0
        
        for j in xrange(self.nState): # N+1

            total = np.sum(self.A[:,j]*alpha[n/self.nEmissionDim-1,:]) #* scaling_factor
            [[mu1, mu2], [cov11, cov12, cov21, cov22]] = self.B[j]

            ## print mu1, mu2, cov11, cov12, cov21, cov22, total
            pred_numerator += total
            
            mu_l[0] += mu1*total
            mu_l[1] += mu2*total
            cov_l[0] += (cov11)*(total**2)
            cov_l[1] += (cov12)*(total**2)
            cov_l[2] += (cov21)*(total**2)
            cov_l[3] += (cov22)*(total**2)

        ## if pred_numerator < 1.0:
        ##     print "PRED>> Low prediction numerator", pred_numerator

        return mu_l, cov_l


    #----------------------------------------------------------------------        
    #
    def predict2(self, X, x1, x2):

        ## n,m    = np.array(X).shape
        X = np.squeeze(X)
        X_test = X.tolist()        
        n = len(X_test)
        ## print "Input: ", np.shape(X_test)        

        mu_l  = np.zeros(2)
        cov_l = np.zeros(4)

        
        ## final_ts_obj = ghmm.EmissionSequence(self.F, X_test)

        ## try:
        ##     # alpha: X_test length y #latent States at the moment t when state i is ended
        ##     #        test_profile_length x number_of_hidden_state
        ##     (alpha,scale) = self.ml.forward(final_ts_obj)
        ##     beta          = self.ml.backward(final_ts_obj, scale)

        ## except:
        ##     print "No alpha is available !!"

        ## alpha = np.array(alpha)
        ## scale = np.array(scale)
        ## beta  = np.array(beta)

        ## alpha_beta = np.sum(alpha[n/self.nEmissionDim-2,:] * beta[n/self.nEmissionDim-2,:])

        #------------------------------------------------------
        ## max_p = 0.0        
        ## for i, x1 in enumerate(self.x1_range):
        ##     for j, x2 in enumerate(self.x2_range):

        ##         if abs(X_test[-2]-x1) > 0.5: continue
        ##         if abs(X_test[-1]-x2) > 0.5: continue
                
        ##         final_ts_obj = ghmm.EmissionSequence(self.F, X_test+[x1, x2])

        ##         try:                
        ##             ## p = self.ml.loglikelihood(final_ts_obj)
        ##             p = self.ml.posterior(final_ts_obj)
        ##         except:
        ##             continue
                
        ##         if max_p < p:
        ##             max_p = p
        ##             max_x1 = x1
        ##             max_x2 = x2
        
        ## #------------------------------------------------------
        ## final_ts_obj = ghmm.EmissionSequence(self.F, X_test)
        ## p = self.ml.posterior(final_ts_obj)

        ## for j in xrange(self.nState):
        ##     total = np.sum(self.A[:,j]*p[n/self.nEmissionDim-1]) #* scaling_factor
        ##     [[mu1, mu2], [cov11, cov12, cov21, cov22]] = self.B[j]
            
        ##     mu_l[0] += mu1*total
        ##     mu_l[1] += mu2*total
        ##     cov_l[0] += (cov11)*(total**2)
        ##     cov_l[1] += (cov12)*(total**2)
        ##     cov_l[2] += (cov21)*(total**2)
        ##     cov_l[3] += (cov22)*(total**2)


        #------------------------------------------------------
        final_ts_obj = ghmm.EmissionSequence(self.F, X_test + [x1, x2])

        try:
            (alpha,scale) = self.ml.forward(final_ts_obj)
        except:
            print "No alpha is available !!"

            
        alpha = np.array(alpha)
        ## print x2
        ## print alpha[n/self.nEmissionDim]

        for j in xrange(self.nState):
            
            [[mu1, mu2], [cov11, cov12, cov21, cov22]] = self.B[j]

            mu_l[0] = x1
            mu_l[1] += alpha[n/self.nEmissionDim,j]*(mu2 + cov21/cov11*(x1 - mu1) )
            ## cov_l[0] += (cov11)*(total**2)
            ## cov_l[1] += (cov12)*(total**2)
            ## cov_l[2] += (cov21)*(total**2)
            ## cov_l[3] += (cov22)*(total**2)

            

        return mu_l, cov_l
        

    #----------------------------------------------------------------------        
    #
    def loglikelihood(self, X):

        X = np.squeeze(X)
        X_test = X.tolist()        
        n = len(X_test)

        final_ts_obj = ghmm.EmissionSequence(self.F, X_test)

        try:    
            p = self.ml.loglikelihood(final_ts_obj)
            ## p = self.ml.posterior(final_ts_obj)
        except:
            print "Likelihood error!!!! "
            p = 0.0
            sys.exit()

        return p
        
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
    def vectors_to_mean_sigma(self,vec, nState): 

        index = 0
        m,n = np.shape(vec)
        mu  = np.zeros(nState)
        sig = np.zeros(nState)
        DIVS = n/nState

        while (index < nState):
            m_init = index*DIVS
            temp_vec = vec[:,(m_init):(m_init+DIVS)]
            temp_vec = np.reshape(temp_vec,(1,DIVS*m))
            mu[index]  = np.mean(temp_vec)
            sig[index] = np.std(temp_vec)
            index = index+1

        return mu, sig
        
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

        index = 0
        m,n = np.shape(vec1)
        #print m,n
        mu_1 = np.zeros(nState)
        mu_2 = np.zeros(nState)
        cov = np.zeros((nState,2,2))

        widths = np.zeros((nState, 1))        
        centers = np.zeros((nState, 1))        
        activation=0.1
        cutoff=0.001        
        last_input_x = 1.0
        alpha_x = -math.log(cutoff)        

        temp = 0.0
        for i in range(nState): 
            t = (i + 1)  / (nState - 1) * 1.0; # 1.0 is the default duration
            input_x = math.exp(-alpha_x * t)
            widths[i] = (input_x - last_input_x) ** 2 / -math.log(activation)
            centers[i] = last_input_x
            last_input_x = input_x
            
            print centers[i], widths[i]
            temp += widths[i]

        return mu_1,mu_2,cov
        
    
    def vectors_to_mean_cov3(self,vec1, vec2, nState): 

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

        
        f_mu1  = interpolate.interp1d(o_x, o_mu1, kind='linear')
        f_sig1 = interpolate.interp1d(o_x, o_sig1, kind='linear')
        f_mu2  = interpolate.interp1d(o_x, o_mu2, kind='linear')
        f_sig2 = interpolate.interp1d(o_x, o_sig2, kind='linear')

        x = np.arange(0.0, float(n-1)+1.0/float(mult), 1.0/float(mult))
        mu1  = f_mu1(x)
        sig1 = f_sig1(x)
        mu2  = f_mu2(x)
        sig2 = f_sig2(x)
               
        while len(mu1) != nState:

            d_mu1  = np.abs(mu1[1:] - mu1[:-1]) # -1 length 
            d_sig1 = np.abs(sig1[1:] - sig1[:-1]) # -1 length 
            idx = d_sig1.tolist().index(min(d_sig1))
            
            mu1[idx]  = (mu1[idx]+mu1[idx+1])/2.0
            sig1[idx] = 0.5*(mu1[idx]**2 + sig1[idx]**2 + mu1[idx+1]**2 + sig1[idx+1]**2) - ( 0.5*(mu1[idx]+mu1[idx+1]) )**2
            mu2[idx]  = (mu2[idx]+mu2[idx+1])/2.0
            sig2[idx] = 0.5*(mu2[idx]**2 + sig2[idx]**2 + mu2[idx+1]**2 + sig2[idx+1]**2) - ( 0.5*(mu2[idx]+mu2[idx+1]) )**2
        
            mu1  = scp.delete(mu1,idx+1)
            sig1 = scp.delete(sig1,idx+1)
            mu2  = scp.delete(mu2,idx+1)
            sig2 = scp.delete(sig2,idx+1)

        cov  = np.zeros((len(mu1),2,2))
        for i in xrange(len(mu1)):
            cov[i,0,0] = sig1[i]**2
            cov[i,0,1] = cov[i,1,0] = (mu1[i] + sig1[i])*(mu2[i] + sig2[i]) - mu1[i] * mu2[i]
            cov[i,1,1] = sig2[i]**2
                
        ## import matplotlib.pyplot as pp
        ## pp.figure()
        ## pp.subplot(211)
        ## pp.plot(mu1)
        ## pp.plot(mu1+1.*np.sqrt(cov[:,0,0]))
        ## ## pp.plot(scp.mean(vec1, axis=0), 'r')
        ## pp.subplot(212)
        ## pp.plot(mu2)
        ## pp.plot(mu2+1.*np.sqrt(cov[:,1,1]))        
        ## pp.show()
        ## sys.exit()

        return mu1, mu2, cov


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
    def data_plot(self, X_test1, X_test2, color='r'):

        
        ## Main predictive distribution        
        self.ax1.plot(np.hstack([X_test1[0]]), color)
        self.ax2.plot(np.hstack([X_test2[0]]), color)

        ## self.ax1.plot(np.hstack([X_test1[0], mu[0]]))
        ## self.ax2.plot(np.hstack([X_test2[0], mu[1]]))

        


        

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
    def convert_sequence(self, X1, X2, emission=False):
        
        n,m = np.array(X1).shape

        X = []
        for i in xrange(n):
            Xs = []
                
            if emission:
                for j in xrange(m):
                    Xs.append([X1[i,j], X2[i,j]])
                X.append(Xs)                    
            else:
                for j in xrange(m):
                    Xs.append([X1[i,j], X2[i,j]])                
                X.append(np.array(Xs).flatten().tolist())

        return np.array(X)
        
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
        

    #----------------------------------------------------------------------        
    #
    def anomaly_check(self, X1, X2=None, ths_mult=None):

        if self.nEmissionDim == 1:
            X_test = X1
        else:
            X_test = self.convert_sequence(X1, X2, emission=False)                
        n = len(np.squeeze(X1))
        
        final_ts_obj = ghmm.EmissionSequence(self.F, X_test[0].tolist())
        path,logp    = self.ml.viterbi(final_ts_obj)

        ## print path, logp, self.ll_mu[path[-1]], logp - (self.ll_mu[path[-1]] - ths_mult*self.ll_std[path[-1]])
        ## raw_input()

        if len(path) == 0: 
            print "zero path: ", logp - (self.ll_mu[0] + ths_mult*self.ll_std[0])
            return 0.0, logp - (self.ll_mu[0] + ths_mult*self.ll_std[0])
        err = logp - (self.ll_mu[path[-1]] - ths_mult*self.ll_std[path[-1]])

        ## print path, logp, (self.ll_mu[path[-1]] - ths_mult*self.ll_std[path[-1]])
        
        if err < 0.0: return 1.0, 0.0 # anomaly
        else: return 0.0, err # normal    

        
    #----------------------------------------------------------------------        
    #
    def simulation(self, X1, X2):

        X1= np.squeeze(X1)
        X2= np.squeeze(X2)
        
        X_time = np.arange(0.0, len(X1), 1.0)
        
        plt.rc('text', usetex=True)
        
        self.fig = plt.figure(1)
        self.gs = gridspec.GridSpec(2, 2, width_ratios=[6, 1]) 
        
        #-------------------------- 1 ------------------------------------
        self.ax11 = self.fig.add_subplot(self.gs[0,0])
        self.ax11.set_xlim([0, np.max(X_time)*1.05])
        self.ax11.set_ylim([0, np.max(X1)*1.4])
        ## self.ax11.set_xlabel(r'\textbf{Angle [}{^\circ}\textbf{]}', fontsize=22)
        ## self.ax11.set_ylabel(r'\textbf{Applied Opening Force [N]}', fontsize=22)

        self.ax12 = self.fig.add_subplot(self.gs[0,1])        
        lbar_1,    = self.ax12.bar(0.0001, 0.0, width=1.0, color='b', zorder=1)
        self.ax12.text(0.13, 0.02, 'Normal', fontsize='14', zorder=-1)            
        self.ax12.text(0.05, 0.95, 'Abnormal', fontsize='14', zorder=0)            
        self.ax12.set_xlim([0.0, 1.0])
        self.ax12.set_ylim([0, 1.0])        
        self.ax12.set_xlabel("Anomaly \n Gauge", fontsize=18)        
        plt.setp(self.ax12.get_xticklabels(), visible=False)
        plt.setp(self.ax12.get_yticklabels(), visible=False)

        self.ax21 = self.fig.add_subplot(self.gs[1,0])
        self.ax21.set_xlim([0, np.max(X_time)*1.05])
        self.ax21.set_ylim([0, np.max(X1)*1.4])
        ## self.ax21.set_xlabel(r'\textbf{Angle [}{^\circ}\textbf{]}', fontsize=22)
        ## self.ax21.set_ylabel(r'\textbf{Applied Opening Force [N]}', fontsize=22)

        self.ax22 = self.fig.add_subplot(self.gs[1,1])        
        lbar_2,    = self.ax22.bar(0.0001, 0.0, width=1.0, color='b', zorder=1)
        self.ax22.text(0.13, 0.02, 'Normal', fontsize='14', zorder=-1)            
        self.ax22.text(0.05, 0.95, 'Abnormal', fontsize='14', zorder=0)            
        self.ax22.set_xlim([0.0, 1.0])
        self.ax22.set_ylim([0, 1.0])        
        self.ax22.set_xlabel("Anomaly \n Gauge", fontsize=18)        
        plt.setp(self.ax22.get_xticklabels(), visible=False)
        plt.setp(self.ax22.get_yticklabels(), visible=False)
        
        #-------------------------- 1 ------------------------------------

        lAll_1, = self.ax11.plot([], [], color='#66FFFF', lw=2, label='Expected force history')
        line_1, = self.ax11.plot([], [], lw=2, label='Current force history')
        lmean_1, = self.ax11.plot([], [], 'm-', linewidth=2.0, label=r'Predicted mean \mu')    
        lvar1_1, = self.ax11.plot([], [], '--', color='0.75', linewidth=2.0, \
                                  label=r'Predicted bounds \mu \pm ( d_1 \sigma + d_2 )')    
        lvar2_1, = self.ax11.plot([], [], '--', color='0.75', linewidth=2.0, )    
        ## self.ax11.legend(loc=2,prop={'size':12})        

        lAll_2, = self.ax21.plot([], [], color='#66FFFF', lw=2, label='Expected force history')
        line_2, = self.ax21.plot([], [], lw=2, label='Current force history')
        lmean_2, = self.ax21.plot([], [], 'm-', linewidth=2.0, label=r'Predicted mean \mu')    
        lvar1_2, = self.ax21.plot([], [], '--', color='0.75', linewidth=2.0, \
                                  label=r'Predicted bounds \mu \pm ( d_1 \sigma + d_2 )')    
        lvar2_2, = self.ax21.plot([], [], '--', color='0.75', linewidth=2.0, )    
        ## self.ax21.legend(loc=2,prop={'size':12})               
        
        self.fig.subplots_adjust(wspace=0.02)        
        
        def init():
            lAll_1.set_data([],[])
            line_1.set_data([],[])
            lmean_1.set_data([],[])
            lvar1_1.set_data([],[])
            lvar2_1.set_data([],[])
            lbar_1.set_height(0.0)            

            lAll_2.set_data([],[])
            line_2.set_data([],[])
            lmean_2.set_data([],[])
            lvar1_2.set_data([],[])
            lvar2_2.set_data([],[])
            lbar_2.set_height(0.0)            
            
            return lAll_1, line_1, lmean_1, lvar1_1, lvar2_1, lbar_1, \
              lAll_2, line_2, lmean_2, lvar1_2, lvar2_2, lbar_2,

        def animate(i):
            lAll_1.set_data(X_time, X1)            
            lAll_2.set_data(X_time, X2)            
            
            x = X_time[:i]
            y1 = X1[:i]
            y2 = X2[:i]

            if i >= 30:
                y1[29] = 2.0
            
            x_nxt = X_time[:i+1]
            y_nxt1 = X1[:i+1]
            y_nxt2 = X2[:i+1]
            line_1.set_data(x, y1)
            line_2.set_data(x, y2)

            
            if i > 1:
            ##     mu_list, var_list = self.update_buffer(y)            
                ## # check anomaly score
                ## bFlag, err, fScore = self.check_anomaly(y_nxt[-1])
                
                X_test = self.convert_sequence(np.array([y1]), np.array([y2]), emission=False)                
                ## mu, cov = self.predict(X_test)
                p       = self.loglikelihood(X_test)

                final_ts_obj = ghmm.EmissionSequence(self.F, X_test[0].tolist())
                posterior = self.ml.posterior(final_ts_obj)
                state_idx = posterior[i-1].index(max(posterior[i-1]))
                threshold = self.likelihood_avg[state_idx]
                
            
            if i >= 3 and i < len(X1)-1:# -self.nFutureStep:

                ## x_sup, idx = hdl.find_nearest(self.aXRange, x[-1], sup=True)            
                ## a_X   = np.arange(x_sup, x_sup+(self.nFutureStep+1)*self.fXInterval, self.fXInterval)
                
                ## if x[-1]-x_sup < x[-1]-x[-2]:                    
                ##     y_idx = 1
                ## else:
                ##     y_idx = int((x[-1]-x_sup)/(x[-1]-x[-2]))+1
                ## a_time = [x[-1], x[-1]+1.0] 
                ## a_mu1  = np.hstack([y1[-1], mu[0]])
                ## a_mu2  = np.hstack([y2[-1], mu[1]])
                ## a_sig1 = np.hstack([0, np.sqrt(cov[0])])
                ## a_sig2 = np.hstack([0, np.sqrt(cov[3])])

                ## lmean_1.set_data( a_time, a_mu1)
                ## lmean_2.set_data( a_time, a_mu2)

                ## sig_mult = self.sig_mult*np.arange(self.nFutureStep) + self.sig_offset
                ## sig_mult = np.hstack([0, sig_mult])

                ## min_val = a_mu1 - a_sig1 
                ## max_val = a_mu1 + a_sig1 

                ## lvar1_1.set_data( a_time, min_val)
                ## lvar2_1.set_data( a_time, max_val)

                lbar_2.set_height(1)
                if p < threshold*0.7:
                    lbar_2.set_color('r')
                elif p < threshold*0.95:          
                    lbar_2.set_color('orange')
                else:
                    lbar_2.set_color('b')
                    
            else:
                lmean_1.set_data([],[])
                lvar1_1.set_data([],[])
                lvar2_1.set_data([],[])
                lbar_1.set_height(0.0)           

                lmean_2.set_data([],[])
                lvar1_2.set_data([],[])
                lvar2_2.set_data([],[])
                lbar_2.set_height(0.0)           
                
            ## if i>=0 or i<4 : 
            ##     self.ax1.legend(handles=[lAll, line, lmean, lvar1], loc=2,prop={'size':12})        
            ## else:
            ##     self.ax1.legend.set_visible(False)
                                
            ## if i%3 == 0 and i >0:
            ##     plt.savefig('roc_ani_'+str(i)+'.pdf')
                
                
            return lAll_1, line_1, lmean_1, lvar1_1, lvar2_1, lbar_1, \
              lAll_2, line_2, lmean_2, lvar1_2, lvar2_2, lbar_2,

           
        anim = animation.FuncAnimation(self.fig, animate, init_func=init,
                                       frames=len(X1), interval=300, blit=True)

        ## anim.save('ani_test.mp4', fps=6, extra_args=['-vcodec', 'libx264'])
        plt.show()



    def likelihood_disp(self, X1, X2, ths_mult):

        n,m = np.shape(X1)

        if self.nEmissionDim == 1:
            X_test = X1
        else:        
            X_test = self.convert_sequence(X1, X2, emission=False)                

        x      = range(m)
        ll     = np.zeros(m)
        ll_mu  = np.zeros(m)
        ll_ths = np.zeros(m)
        for i in xrange(m):
            if i == 0: continue
        
            final_ts_obj = ghmm.EmissionSequence(self.F, X_test[0,:i*self.nEmissionDim].tolist())
            path,logp    = self.ml.viterbi(final_ts_obj)

            if len(path) == 0: continue

            ll[i]     = logp
            ll_mu[i]  = self.ll_mu[path[-1]]
            ll_ths[i] = self.ll_mu[path[-1]] - ths_mult*self.ll_std[path[-1]]


        l = [0]
        for p in path:
            l.append(p%2)

        import matplotlib.collections as collections
        
        plt.figure()
        plt.rc('text', usetex=True)
        ax1 = plt.subplot(311)
        ax1.plot(x, X1[0])
        collection = collections.BrokenBarHCollection.span_where(x, ymin=0, ymax=10, where=np.array(l)>0, facecolor='green', edgecolor='none', alpha=0.3)
        ax1.add_collection(collection)
        ax1.set_ylabel("Force magnitude")
            
        ax2 = plt.subplot(312)
        ax2.plot(x, X2[0])
        collection = collections.BrokenBarHCollection.span_where(x, ymin=0, ymax=10, where=np.array(l)>0, facecolor='green', edgecolor='none', alpha=0.3)
        ax2.add_collection(collection)
        ax2.set_ylabel("Audio [RMS]")

        
        ax3 = plt.subplot(313)
        
        ax3.plot(x, ll, 'b', label='{log likelihood per current step}')
        ax3.plot(x, ll_mu, 'r', label=r'$\mu$')
        ax3.plot(x, ll_ths, 'r--', label=r'$\mu + c \sigma$')
        ax3.set_ylabel("Log likelihood")
        
        ax3.legend(loc='best',prop={'size':10})
        plt.show()
        
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
        
