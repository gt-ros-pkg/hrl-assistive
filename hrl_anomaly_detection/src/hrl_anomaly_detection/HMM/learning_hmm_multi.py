#!/usr/local/bin/python

import sys, os, copy, time
import numpy as np, math
import scipy as scp
from scipy import optimize, interpolate
from scipy.stats import norm, entropy

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
from sklearn.cluster import KMeans
## from pysmac.optimize import fmin                
## from scipy.optimize import fsolve
## from scipy import interpolate

from learning_base import learning_base
import sandbox_dpark_darpa_m3.lib.hrl_dh_lib as hdl


class learning_hmm_multi(learning_base):
    def __init__(self, nState, nFutureStep=5, nCurrentStep=10, \
                 trans_type="left_right", nEmissionDim=2, check_method='progress', cluster_type='time'):

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
        self.check_method = check_method # ['global', 'progress']
        self.cluster_type = cluster_type
                
        # emission domain of this model        
        self.F = ghmm.Float()  

        # Assign local functions
        learning_base.__dict__['fit'] = self.fit        
        learning_base.__dict__['predict'] = self.predict
        learning_base.__dict__['score'] = self.score                

        print "HMM initialized for ",self.check_method
        pass

        
    #----------------------------------------------------------------------        
    #
    def fit(self, aXData1, aXData2=None, A=None, B=None, pi=None, cov_mult=[1.0, 1.0, 1.0, 1.0], \
            B_dict=None, verbose=False, ml_pkl='ml_temp.pkl', use_pkl=False):

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
        if math.isnan(ret) :
            print "Return ", ret
            return False

        [self.A,self.B,self.pi] = self.ml.asMatrices()
        self.A = np.array(self.A)
        self.B = np.array(self.B)

        #--------------- learning for anomaly detection ----------------------------
        [A, B, pi] = self.ml.asMatrices()
        n,m = np.shape(X1)


        if self.check_method == 'change' or self.check_method == 'globalChange':
            # Get maximum change of loglikelihood over whole time
            ll_delta_logp = []
            for j in xrange(n):    
                l_logp = []                
                for k in xrange(1,m):
                    final_ts_obj = ghmm.EmissionSequence(self.F, X_train[j][:k*self.nEmissionDim])
                    logp         = self.ml.loglikelihoods(final_ts_obj)[0]

                    l_logp.append(logp)
                l_delta_logp = np.array(l_logp[1:]) - np.array(l_logp[:-1])                    
                ll_delta_logp.append(l_delta_logp)

            self.l_mean_delta = np.mean(abs(np.array(ll_delta_logp).flatten()))
            self.l_std_delta = np.std(abs(np.array(ll_delta_logp).flatten()))

            print "mean_delta: ", self.l_mean_delta, " std_delta: ", self.l_std_delta
        
        if self.check_method == 'global' or self.check_method == 'globalChange':
            # Get average loglikelihood threshold over whole time

            l_logp = []
            for j in xrange(n):    
                for k in xrange(1,m):
                    final_ts_obj = ghmm.EmissionSequence(self.F, X_train[j][:k*self.nEmissionDim])
                    logp         = self.ml.loglikelihoods(final_ts_obj)[0]

                    l_logp.append(logp)

            self.l_mu = np.mean(l_logp)
            self.l_std = np.std(l_logp)


        elif self.check_method == 'progress':
            # Get average loglikelihood threshold wrt progress
            ######################################################################################
            if os.path.isfile(ml_pkl) and use_pkl:
                d = ut.load_pickle(ml_pkl)
                self.l_statePosterior = d['state_post'] # time x state division
                self.ll_mu            = d['ll_mu']
                self.ll_std           = d['ll_std']            
            else:        

                if self.cluster_type == 'time':
                    self.nGaussian = self.nState
                    self.std_coff  = 1.0
                    g_mu_list = np.linspace(0, m-1, self.nGaussian) #, dtype=np.dtype(np.int16))
                    g_sig     = float(m) / float(self.nGaussian) * self.std_coff

                    n_jobs = -1
                    r = Parallel(n_jobs=n_jobs)(delayed(learn_likelihoods_progress)(i, n, m, A, B, pi, \
                                                                           self.F, X_train, \
                                                                           self.nEmissionDim, g_mu_list[i], g_sig, \
                                                                           self.nState) \
                                                                           for i in xrange(self.nGaussian))
                    l_i, self.l_statePosterior, self.ll_mu, self.ll_std = zip(*r)

                elif self.cluster_type == 'state':

                    self.km = None                    
                    self.ll_mu = None
                    self.ll_std = None
                    self.ll_mu, self.ll_std = self.state_clustering(X1, X2)
                    path_mat  = np.zeros((self.nState, m*n))
                    likelihood_mat = np.zeros((1, m*n))
                    self.l_statePosterior=None
                    

                d = {}
                d['state_post'] = self.l_statePosterior
                d['ll_mu'] = self.ll_mu
                d['ll_std'] = self.ll_std
                ut.save_pickle(d, ml_pkl)
                    

            ##########################################################################
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

        return True


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
    def convert_sequence(self, data1, data2, emission=False):

        # change into array from other types
        if type(data1) is not np.ndarray:
            X1 = copy.copy(np.array(data1))
        else:
            X1 = copy.copy(data1)
        if type(data2) is not np.ndarray:
            X2 = copy.copy(np.array(data2))
        else:
            X2 = copy.copy(data2)

        # Change into 2dimensional array
        dim = np.shape(X1)
        if len(dim) == 1:
            X1 = np.reshape(X1, (1,len(X1)))            
        dim = np.shape(X2)
        if len(dim) == 1:
            X2 = np.reshape(X2, (1,len(X2)))            

        n,m = np.shape(X1)

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

        if self.nEmissionDim == 1: X_test = np.array([X1])
        else: X_test = self.convert_sequence(X1, X2, emission=False)                

        try:
            final_ts_obj = ghmm.EmissionSequence(self.F, X_test[0].tolist())
            logp         = self.ml.loglikelihood(final_ts_obj)
        except:
            print "Too different input profile that cannot be expressed by emission matrix"
            return -1, 0.0 # error

        if self.check_method == 'change' or self.check_method == 'globalChange':

            if len(X1)<3: return -1, 0.0 #error

            if self.nEmissionDim == 1: X_test = np.array([X1[:-1]])
            else: X_test = self.convert_sequence(X1[:-1], X2[:-1], emission=False)                

            try:
                final_ts_obj = ghmm.EmissionSequence(self.F, X_test[0].tolist())
                last_logp         = self.ml.loglikelihood(final_ts_obj)
            except:
                print "Too different input profile that cannot be expressed by emission matrix"
                return -1, 0.0 # error

            ## print self.l_mean_delta + ths_mult*self.l_std_delta, abs(logp-last_logp)
            if type(ths_mult) == list or type(ths_mult) == np.ndarray or type(ths_mult) == tuple:
                err = (self.l_mean_delta + (-1.0*ths_mult[0])*self.l_std_delta ) - abs(logp-last_logp)
            else:                
                err = (self.l_mean_delta + (-1.0*ths_mult)*self.l_std_delta ) - abs(logp-last_logp)
            if err < 0.0: return 1.0, 0.0 # anomaly            
            
        if self.check_method == 'global' or self.check_method == 'globalChange':
            if type(ths_mult) == list or type(ths_mult) == np.ndarray or type(ths_mult) == tuple:
                err = logp - (self.l_mu + ths_mult[1]*self.l_std) 
            else:
                err = logp - (self.l_mu + ths_mult*self.l_std) 
                
            if err < 0.0: return 1.0, 0.0 # anomaly
            else: return 0.0, err # normal               
            
        elif self.check_method == 'progress':
            try:
                post = np.array(self.ml.posterior(final_ts_obj))            
            except:
                print "Unexpected profile!! GHMM cannot handle too low probability. Underflow?"
                return 1.0, 0.0 # anomaly

            n = len(np.squeeze(X1))
                
            # Find the best posterior distribution
            min_index, min_dist = self.findBestPosteriorDistribution(post[n-1])
            ## print logp, self.ll_mu[min_index], logp - (self.ll_mu[min_index] + ths_mult*self.ll_std[min_index])
            ## raw_input()
            if (type(ths_mult) == list or type(ths_mult) == np.ndarray or type(ths_mult) == tuple) and \
              len(ths_mult)>1:
                err = logp - (self.ll_mu[min_index] + ths_mult[min_index]*self.ll_std[min_index])
            else:
                err = logp - (self.ll_mu[min_index] + ths_mult*self.ll_std[min_index])
                    
            ## print logp, (self.ll_mu[min_index] - ths_mult*self.ll_std[min_index])
        ## else:
        ##     print "Not available anomaly check method"


        if err < 0.0: return 1.0, 0.0 # anomaly
        else: return 0.0, err # normal    


    #----------------------------------------------------------------------        
    #
    def get_sensitivity_gain_batch(self, x_test1, x_test2=None):

        min_ths = 0
        min_ind = 0
        if self.check_method == 'progress':
            min_ths = np.zeros(self.nGaussian)+10000
            min_ind = np.zeros(self.nGaussian)
        elif self.check_method == 'globalChange':
            min_ths = np.zeros(2)+10000

        n = len(x_test1)
        for i in range(n):
            m = len(x_test1[i])

            # anomaly_check only returns anomaly cases only
            for j in range(2,m):                    

                if self.nEmissionDim == 2:            
                    ths, ind = self.get_sensitivity_gain(x_test1[i][:j], x_test2[i][:j])   
                else:
                    ths, ind = self.get_sensitivity_gain(x_test1[i][:j])

                if ths == []: continue

                if self.check_method == 'progress':
                    if min_ths[ind] > ths:
                        min_ths[ind] = ths
                        print "Minimum threshold: ", min_ths[ind], ind                                
                elif self.check_method == 'globalChange':
                    if min_ths[0] > ths[0]:
                        min_ths[0] = ths[0]
                    if min_ths[1] > ths[1]:
                        min_ths[1] = ths[1]
                    print "Minimum threshold: ", min_ths[0], min_ths[1]                                
                else:
                    if min_ths > ths:
                        min_ths = ths
                        print "Minimum threshold: ", min_ths

        return min_ths, min_ind
        
    #----------------------------------------------------------------------        
    #
    def get_sensitivity_gain(self, X1, X2=None):

        if self.nEmissionDim == 1: X_test = np.array([X1])
        else: X_test = self.convert_sequence(X1, X2, emission=False)                

        try:
            final_ts_obj = ghmm.EmissionSequence(self.F, X_test[0].tolist())
            logp         = self.ml.loglikelihood(final_ts_obj)
        except:
            print "Too different input profile that cannot be expressed by emission matrix"
            return [], 0.0 # error

            
        if self.check_method == 'progress':
            try:
                post = np.array(self.ml.posterior(final_ts_obj))            
            except:
                print "Unexpected profile!! GHMM cannot handle too low probability. Underflow?"
                return [], 0.0 # anomaly

            n = len(np.squeeze(X1))
                
            # Find the best posterior distribution
            min_index, min_dist = self.findBestPosteriorDistribution(post[n-1])

            ths = (logp - self.ll_mu[min_index])/self.ll_std[min_index]
            return ths, min_index

        elif self.check_method == 'global':
            ths = (logp - self.l_mu) / self.l_std
            return ths, 0

        elif self.check_method == 'change':
            if len(X1)<3: return [], 0.0 #error

            if self.nEmissionDim == 1: X_test = np.array([X1[:-1]])
            else: X_test = self.convert_sequence(X1[:-1], X2[:-1], emission=False)                

            try:
                final_ts_obj = ghmm.EmissionSequence(self.F, X_test[0].tolist())
                last_logp         = self.ml.loglikelihood(final_ts_obj)
            except:
                print "Too different input profile that cannot be expressed by emission matrix"
                return -1, 0.0 # error
            
            ths = -(( abs(logp-last_logp) - self.l_mean_delta) / self.l_std_delta)
            return ths, 0

        elif self.check_method == 'globalChange':
            if len(X1)<3: return [], 0.0 #error

            if self.nEmissionDim == 1: X_test = np.array([X1[:-1]])
            else: X_test = self.convert_sequence(X1[:-1], X2[:-1], emission=False)                

            try:
                final_ts_obj = ghmm.EmissionSequence(self.F, X_test[0].tolist())
                last_logp         = self.ml.loglikelihood(final_ts_obj)
            except:
                print "Too different input profile that cannot be expressed by emission matrix"
                return [], 0.0 # error
            
            ths_c = -(( abs(logp-last_logp) - self.l_mean_delta) / self.l_std_delta)

            ths_g = (logp - self.l_mu) / self.l_std
            
            return [ths_c, ths_g], 0
            

    #----------------------------------------------------------------------        
    #
    def simulation(self, X1, X2, ths_mult, freq=43.0, bSave=False):

        X1= np.squeeze(X1)
        X2= np.squeeze(X2)
        
        X_time = np.arange(0.0, len(X1), 1.0) * (1./freq)
        self.l_ll = []
        self.l_ell = []
        self.l_thres = []
        min_llim = -70.0
        max_llim = 150.0
        
        
        plt.rc('text', usetex=True)
        
        self.fig = plt.figure(1)
        ## self.gs = gridspec.GridSpec(3, 2, width_ratios=[6, 1], height_ratios=[1,1,2]) 
        self.gs = gridspec.GridSpec(3, 1, height_ratios=[1,1,2]) 
        
        #-------------------------- 1 ------------------------------------
        self.ax11 = self.fig.add_subplot(self.gs[0,0])
        self.ax11.set_xlim([0, np.max(X_time)*1.05])
        self.ax11.set_ylim([0, np.max(X1)*1.4])
        ## self.ax11.set_xlabel(r'\textbf{Angle [}{^\circ}\textbf{]}', fontsize=22)
        self.ax11.set_ylabel('Magnitude [N]', fontsize=14)

        self.ax21 = self.fig.add_subplot(self.gs[1,0])
        self.ax21.set_xlim([0, np.max(X_time)*1.05])
        self.ax21.set_ylim([0, np.max(X2)*1.4])
        ## self.ax21.set_xlabel(r'\textbf{Angle [}{^\circ}\textbf{]}', fontsize=22)
        self.ax21.set_ylabel('Energy', fontsize=14)

        self.ax31 = self.fig.add_subplot(self.gs[2,0])
        self.ax31.set_xlim([0, np.max(X_time)*1.05])
        self.ax31.set_ylim([min_llim, max_llim])
        self.ax31.set_xlabel('Time [s]', fontsize=22)
        self.ax31.set_ylabel('log-likelihood', fontsize=14)
        
        ## self.ax32 = self.fig.add_subplot(self.gs[2,1])        
        ## self.ax32.text(0.13, 0.02, 'Normal', fontsize='14', zorder=-1)            
        ## self.ax32.text(0.05, 0.95, 'Abnormal', fontsize='14', zorder=0)            
        ## self.ax32.set_xlim([0.0, 1.0])
        ## self.ax32.set_ylim([0, 1.0])        
        ## self.ax32.set_xlabel("Anomaly \n Gauge", fontsize=18)        
        ## plt.setp(self.ax32.get_xticklabels(), visible=False)
        ## plt.setp(self.ax32.get_yticklabels(), visible=False)
        
        #-------------------------- 2 ------------------------------------

        line_1, = self.ax11.plot([], [], lw=2, label='Force')
        line_2, = self.ax21.plot([], [], lw=2, label='Sound')
        line_3, = self.ax31.plot([], [], lw=2, label='Log-likelihood')
        

        lmean, = self.ax31.plot([], [], 'm-', linewidth=2.0, label='Expected log-likelihood')    
        lvar, = self.ax31.plot([], [], '--', color='0.75', linewidth=2.0, \
                                  label='Threshold')                                      
        ## lbar,     = self.ax32.bar(0.0001, 0.0, width=1.0, color='b', zorder=1)

        self.legend = None
        
        ## lAll_1, = self.ax11.plot([], [], color='#66FFFF', lw=2, label='Expected force history')
        ## ## self.ax11.legend(loc=2,prop={'size':12})        

        ## lAll_2, = self.ax21.plot([], [], color='#66FFFF', lw=2, label='Expected force history')
        ## lmean_2, = self.ax21.plot([], [], 'm-', linewidth=2.0, label=r'Predicted mean \mu')    
        ## lvar1_2, = self.ax21.plot([], [], '--', color='0.75', linewidth=2.0, \
        ##                           label=r'Predicted bounds \mu \pm ( d_1 \sigma + d_2 )')    
        ## lvar2_2, = self.ax21.plot([], [], '--', color='0.75', linewidth=2.0, )    
        ## ## self.ax21.legend(loc=2,prop={'size':12})               
        
        self.fig.subplots_adjust(wspace=0.02)        
        
        def init():
            line_1.set_data([],[])
            line_2.set_data([],[])
            line_3.set_data([],[])
            
            lmean.set_data([],[])
            lvar.set_data([],[])
            ## lbar.set_height(0.0)            
            self.l_ll = []
            self.l_ell = []
            self.l_thres = []            
            
            return line_1, line_2, line_3, lmean, lvar # , lbar

        def animate(i):
            if i==0:
                self.l_ll = []
                self.l_ell = []
                self.l_thres = []            
                
            
            x = X_time[:i]
            y1 = X1[:i]
            y2 = X2[:i]

            try:
                line_1.set_data(x, y1)
                line_2.set_data(x, y2)
            except:
                print "Length error for line_1"
                    
                        
            if i > 1:
                
                X_test = self.convert_sequence(np.array([y1]), np.array([y2]), emission=False)                
                ## mu, cov = self.predict(X_test)
                p      = self.loglikelihood(X_test)
                
                final_ts_obj = ghmm.EmissionSequence(self.F, X_test[0].tolist())
                post = self.ml.posterior(final_ts_obj)

                n = len(np.squeeze(y1))

                # Find the best posterior distribution
                min_index, min_dist = self.findBestPosteriorDistribution(post[n-1])

                threshold = (self.ll_mu[min_index] + ths_mult[min_index]*self.ll_std[min_index])

                self.l_ll.append(p)
                self.l_ell.append(self.ll_mu[min_index])
                self.l_thres.append(threshold)
                
                x = X_time[1:i]

                max_lim = max([max(self.l_ll),max(self.l_ell), max(self.l_thres), max_llim])
                min_lim = min([min(self.l_ll),min(self.l_ell), min(self.l_thres), min_llim])
                self.ax31.set_ylim([min_lim, max_lim*1.2])
                
                try:
                    line_3.set_data(x, self.l_ll)                    
                    lmean.set_data(x, self.l_ell)                    
                    lvar.set_data(x, self.l_thres)                    
                except:
                    print len(x), len(self.l_ll)
                    print i
                    print "Length error"

                ## lbar.set_height(1)
                ## if p < threshold:
                ##     lbar.set_color('r')
                ## elif p < threshold*0.9:          
                ##     lbar.set_color('orange')
                ## else:
                ##     lbar.set_color('b')
                
            else:
                line_3.set_data([],[])
                lmean.set_data([],[])
                lvar.set_data([],[])
                ## lbar.set_height(0.0)           

                
            self.ax11.legend(handles=[line_1], loc=2,prop={'size':12})
            self.ax21.legend(handles=[line_2], loc=2,prop={'size':12})
                
            if i>=0 and i<10 or True: 
                self.legend = self.ax31.legend(handles=[line_3, lmean, lvar], loc=2,prop={'size':12})        
            else:
                self.legend.remove()
                                
            ## if i%3 == 0 and i >0:
            ##     plt.savefig('roc_ani_'+str(i)+'.pdf')
                
                
            return line_1, line_2, line_3, lmean, lvar #, lbar

           
        anim = animation.FuncAnimation(self.fig, animate, init_func=init,
                                       frames=len(X1), interval=300) #, blit=True

        if bSave == True:
            anim.save('ani_test.mp4', fps=6, extra_args=['-vcodec', 'libx264'])
        plt.show()



    def likelihood_disp(self, X1, X2, ths_mult, scale1=None, scale2=None):

        n,m = np.shape(X1)
        print "Input sequence X1: ", n,m

        if self.nEmissionDim == 1:
            X_test = X1
        else:        
            X_test = self.convert_sequence(X1, X2, emission=False)                

        x        = np.arange(0., float(m))
        ll_likelihood = np.zeros(m)
        ll_state_idx  = np.zeros(m)
        ll_likelihood_mu  = np.zeros(m)
        ll_likelihood_std = np.zeros(m)
        ll_thres_mult = np.zeros(m)
        for i in xrange(m):
            if i == 0: continue
        
            final_ts_obj = ghmm.EmissionSequence(self.F, X_test[0,:i*self.nEmissionDim].tolist())
            logp         = self.ml.loglikelihood(final_ts_obj)
            post         = np.array(self.ml.posterior(final_ts_obj))
            
            # Find the best posterior distribution
            min_index, min_dist = self.findBestPosteriorDistribution(post[i-1])
                
            ll_likelihood[i] = logp
            ll_state_idx[i]  = min_index
            ll_likelihood_mu[i]  = self.ll_mu[min_index]
            ll_likelihood_std[i] = self.ll_std[min_index] #self.ll_mu[min_index] + ths_mult*self.ll_std[min_index]
            if self.check_method == 'progress' and type(ths_mult) is not float and len(ths_mult)>1:
                ll_thres_mult[i] = ths_mult[min_index]
            else:
                ll_thres_mult[i] = ths_mult
            
        # temp to see offline state path
        ## path_l = path
            
        # state blocks
        block_flag = []
        block_x    = []
        block_state= []
        text_n     = []
        text_x     = []
        for i, p in enumerate(ll_state_idx):
            if i is 0: 
                block_flag.append(0)
                block_state.append(0)
                text_x.append(0.0)
            elif ll_state_idx[i] != ll_state_idx[i-1]:
                if block_flag[-1] is 0: block_flag.append(1)
                else: block_flag.append(0)
                block_state.append( int(p) )
                text_x[-1] = (text_x[-1]+float(i-1))/2.0 - 0.5 # 
                text_x.append(float(i))
            else:
                block_flag.append(block_flag[-1])
            block_x.append(float(i))
        text_x[-1] = (text_x[-1]+float(m-1))/2.0 - 0.5 # 

        block_flag_interp = []
        block_x_interp    = []
        for i in xrange(len(block_flag)):
            block_flag_interp.append(block_flag[i])
            block_flag_interp.append(block_flag[i])
            block_x_interp.append( float(block_x[i]) )
            block_x_interp.append(block_x[i]+0.5)


        y1 = (X1[0]/scale1[2])*(scale1[1]-scale1[0])+scale1[0]
        y2 = (X2[0]/scale2[2])*(scale2[1]-scale2[0])+scale2[0]
            
        import matplotlib.collections as collections

        ## matplotlib.rcParams['figure.figsize'] = 8,7
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        
        fig = plt.figure()
        plt.rc('text', usetex=True)
        
        ax1 = plt.subplot(311)
        ax1.plot(x*(1./43.), y1)
        y_max = np.amax(y1) #35.0
        collection = collections.BrokenBarHCollection.span_where(np.array(block_x_interp)*(1./43.), 
                                                                 ymin=0, ymax=y_max+4.0, 
                                                                 where=np.array(block_flag_interp)>0, 
                                                                 facecolor='green', 
                                                                 edgecolor='none', alpha=0.3)
        ax1.add_collection(collection)
        # Text for progress
        for i in xrange(len(block_state)):
            if i%2 is 0:
                if i<10:
                    ax1.text((text_x[i])*(1./43.), y_max-0.0, str(block_state[i]+1))
                else:
                    ax1.text((text_x[i]-1.0)*(1./43.), y_max-0.0, str(block_state[i]+1))
            else:
                if i<10:
                    ax1.text((text_x[i])*(1./43.), y_max-4.0, str(block_state[i]+1))
                else:
                    ax1.text((text_x[i]-1.0)*(1./43.), y_max-4.0, str(block_state[i]+1))
                        
        ax1.set_ylabel("Force [N]", fontsize=18)
        ax1.set_xlim([0, x[-1]*(1./43.)])
        ## ax1.set_ylim([0, np.amax(y1)*1.1])
        ax1.set_ylim([0, y_max+4.0])
        
        ax2 = plt.subplot(312)
        ax2.plot(x*(1./43.), y2)
        y_max = 0.01
        collection = collections.BrokenBarHCollection.span_where(np.array(block_x_interp)*(1./43.), 
                                                                 ymin=0, ymax=y_max, 
                                                                 where=np.array(block_flag_interp)>0, 
                                                                 facecolor='green', 
                                                                 edgecolor='none', alpha=0.3)
        ax2.add_collection(collection)
        ax2.set_ylabel("Sound [RMS]", fontsize=18)
        ax2.set_xlim([0, x[-1]*(1./43.)])
        ax2.set_ylim([0, y_max])
        
        ax3 = plt.subplot(313)        
        ax3.plot(x*(1./43.), ll_likelihood, 'b', label='Log-likelihood \n from test data')
        ax3.plot(x*(1./43.), ll_likelihood_mu, 'r', label='Expected log-likelihood \n from trained model')
        ax3.plot(x*(1./43.), ll_likelihood_mu - ll_thres_mult*ll_likelihood_std, 'r--', label='Threshold')
        ## ax3.set_ylabel(r'$log P({\mathbf{X}} | {\mathbf{\theta}})$',fontsize=18)
        ax3.set_ylabel('Log-likelihood',fontsize=18)
        ax3.set_xlim([0, x[-1]*(1./43.)])
        
        ## ax3.legend(loc='upper left', fancybox=True, shadow=True, ncol=3, prop={'size':14})
        lgd = ax3.legend(loc='upper center', fancybox=True, shadow=True, ncol=3, bbox_to_anchor=(0.5, -0.3), \
                   prop={'size':14})
        ax3.set_xlabel('Time [sec]', fontsize=18)

        plt.subplots_adjust(bottom=0.15)        
        fig.savefig('test.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
        fig.savefig('test.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
        
        os.system('cp test.p* ~/Dropbox/HRL/')
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
        
    def path_disp(self, X1, X2, scale1=None, scale2=None):

        from mpl_toolkits.axes_grid1 import make_axes_locatable
        
        n,m = np.shape(X1)
        print n,m
        x   = np.arange(0., float(m))*(1./43.)
        path_mat  = np.zeros((self.nState, m))
        zbest_mat = np.zeros((self.nState, m))

        path_l = []            
        for i in xrange(n):

            x_test1 = X1[i:i+1,:]
            x_test2 = X2[i:i+1,:]            

            if self.nEmissionDim == 1:
                X_test = x_test1
            else:        
                X_test = self.convert_sequence(x_test1, x_test2, emission=False)                

            final_ts_obj = ghmm.EmissionSequence(self.F, X_test[0].tolist())
            path,_    = self.ml.viterbi(final_ts_obj)        
            post = self.ml.posterior(final_ts_obj)

            use_last = False
            for j in xrange(m):
                ## sum_post = np.sum(post[j*2+1])
                ## if sum_post <= 0.1 or sum_post > 1.1 or sum_post == float('Inf') or use_last == True:
                ##     use_last = True
                ## else:
                add_post = np.array(post[j])/float(n)    
                path_mat[:, j] += add_post 

            path_l.append(path)
            for j in xrange(m):
                zbest_mat[path[j], j] += 1.0

        path_mat /= np.sum(path_mat, axis=0)
        zbest_mat /= np.sum(zbest_mat, axis=0)

        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
            
        fig = plt.figure()
        plt.rc('text', usetex=True)
        
        ax1 = plt.subplot(111)            
        im  = ax1.imshow(path_mat, cmap=plt.cm.Reds, interpolation='none', origin='upper', 
                         extent=[0,float(m)*(1.0/43.),20,1], aspect=0.1)

        ## divider = make_axes_locatable(ax1)
        ## cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, fraction=0.031, ticks=[0.0, 1.0], pad=0.01)
        ax1.set_xlabel("Time [sec]", fontsize=18)
        ax1.set_ylabel("Hidden State Index", fontsize=18)
        
        ## for p in path_l:
        ##     ax1.plot(x, p, '*')

        ## ax2 = plt.subplot(212)
        ## im2 = ax2.imshow(zbest_mat, cmap=plt.cm.Reds, interpolation='none', origin='upper', 
        ##                  extent=[0,float(m)*(1.0/43.),20,1], aspect=0.1)
        ## plt.colorbar(im2, fraction=0.031, ticks=[0.0, 1.0], pad=0.01)
        ## ax2.set_xlabel("Time [sec]", fontsize=18)
        ## ax2.set_ylabel("Hidden State", fontsize=18)
        
        
        ## ax3 = plt.subplot(313)
        fig.savefig('test.pdf')
        fig.savefig('test.png')
        os.system('cp test.p* ~/Dropbox/HRL/')
        ## plt.grid()            
        ## plt.show()

    def state_clustering(self, X1, X2=None):
        n,m = np.shape(X1)

        print n,m
        x   = np.arange(0., float(m))*(1./43.)
        state_mat  = np.zeros((self.nState, m*n))
        likelihood_mat = np.zeros((1, m*n))

        count = 0           
        for i in xrange(n):

            for j in xrange(1,m):            

                if self.nEmissionDim == 1:
                    x_test1 = X1[i:i+1,:j]
                    X_test = x_test1
                else:        
                    x_test1 = X1[i:i+1,:j]
                    x_test2 = X2[i:i+1,:j]            
                    X_test = self.convert_sequence(x_test1, x_test2, emission=False)                

                final_ts_obj = ghmm.EmissionSequence(self.F, X_test[0].tolist())
                ## path,_    = self.ml.viterbi(final_ts_obj)        
                post      = self.ml.posterior(final_ts_obj)
                logp      = self.ml.loglikelihood(final_ts_obj)

                state_mat[:, count] = np.array(post[j-1])
                likelihood_mat[0,count] = logp
                count += 1

        # k-means
        init_center = np.eye(self.nState, self.nState)
        self.km = KMeans(self.nState, init=init_center)
        idx_list = self.km.fit_predict(state_mat.transpose())

        # mean and variance of likelihoods
        l = []
        for i in xrange(self.nState):
            l.append([])

        for i, idx in enumerate(idx_list):
            l[idx].append(likelihood_mat[0][i]) 

        l_mean = []
        l_std = []
        for i in xrange(self.nState):
            l_mean.append( np.mean(l[i]) )
            l_std.append( np.std(l[i]) )
                
        return l_mean, l_std
        
    def path_cluster(self, X1, X2, res_file, scale1=None, scale2=None):

        
        if os.path.isfile(res_file): 
            d = ut.load_pickle(res_file)
            path_mat = d['path'] 
            likelihood_mat = d['likelihoods'] 
        else:                        
            state_mat, likelihood_mat = self.state_clustering(X1, X2)
            
            d = {}
            d['path'] = path_mat
            d['likelihoods'] = likelihood_mat
            ut.save_pickle(d, 'clustering.pkl')


        # clustering
        print n,m
        print np.shape(path_mat), np.shape(likelihood_mat)
        m=m-1
        
        raw_idx_list = []
        raw_l_list = []
        for i in xrange(n):
            single_path = path_mat[:,i*m:(i+1)*m]
            idx = np.argmax(single_path, axis=0)
            raw_idx_list.append(idx)
            raw_l_list.append(likelihood_mat[0][i*m:(i+1)*m])

        raw_l_mean = np.mean(raw_l_list, axis=0)
        raw_l_std = np.std(raw_l_list, axis=0)
        #------------------------------------------------

        # k-means
        init_center = np.eye(self.nState, self.nState)
        km = KMeans(self.nState, init=init_center)
        idx_list = km.fit_predict(path_mat.transpose())

        # mean and variance of likelihoods
        l = []
        for i in xrange(self.nState):
            l.append([])

        for i, idx in enumerate(idx_list):
            l[idx].append(likelihood_mat[0][i]) 

        l_mean = []
        l_std = []
        for i in xrange(self.nState):
            l_mean.append( np.mean(l[i]) )
            l_std.append( np.std(l[i]) )

        path_list = []
        path_l_list = []
        for i in xrange(n):
            #print idx_list[i*m:(i+1)*m]
            path_list.append(idx_list[i*m:(i+1)*m])

            path_l_list.append([])
            for j in idx_list[i*m:(i+1)*m]:
                path_l_list[-1].append(l_mean[j])

        ###############
        ## fig = plt.figure(1)
        ## ax1 = plt.subplot(211)
        ## plt.plot(np.array(raw_idx_list).transpose(), '*')

        ## ax2 = plt.subplot(212)        
        ## plt.plot(np.array(path_list).transpose(), '*')
        
        ###########
        fig = plt.figure(1)
        ax1 = plt.subplot(211)
        plt.plot(raw_l_mean, 'r-', linewidth=2.0)
        plt.plot(np.array(raw_l_list).transpose(), '-', linewidth=1.0)

        ax2 = plt.subplot(212)        
        plt.plot(raw_l_mean, 'r-', linewidth=2.0)        
        plt.plot(np.array(path_l_list).transpose(), '*')
        

        

        fig.savefig('test.pdf')
        fig.savefig('test.png')
        ## os.system('cp test.p* ~/Dropbox/HRL/')
        plt.show()
            
        

    def progress_analysis(self, X1, X2, scale1=None, scale2=None):

        n,m = np.shape(X1)
        if self.nEmissionDim == 1:
            X_test = X1
        else:        
            X_test = self.convert_sequence(X1, X2, emission=False)                


        final_ts_obj = ghmm.EmissionSequence(self.F, X_test[0,:].tolist())
        off_progress,_    = self.ml.viterbi(final_ts_obj)            
        
        on_progress = np.zeros(m)
        for i in xrange(m):
            if i == 0: continue
        
            final_ts_obj = ghmm.EmissionSequence(self.F, X_test[0,:i*self.nEmissionDim].tolist())
            path,_    = self.ml.viterbi(final_ts_obj)
        
            if len(path) == 0: 
                on_progress[i] = on_progress[i-1]
                continue
            else: 
                on_progress[i] = path[-1]

        return off_progress, on_progress


    def findBestPosteriorDistribution(self, post):
        # Find the best posterior distribution
        min_dist  = 100000000
        min_index = 0

        if self.cluster_type == 'time':
            for j in xrange(self.nGaussian):
                dist = entropy(post, self.l_statePosterior[j])
                if min_dist > dist:
                    min_index = j
                    min_dist  = dist 
        else:
            print "state based clustering"
            min_index = self.km.predict(post)
            min_dist  = -1

        return min_index, min_dist
        
        
####################################################################
# functions for paralell computation
####################################################################
        
def learn_likelihoods_progress(i, n, m, A, B, pi, F, X_train, nEmissionDim, g_mu, g_sig, nState):

    if nEmissionDim ==2:
        ml = ghmm.HMMFromMatrices(F, ghmm.MultivariateGaussianDistribution(F), A, B, pi)
    else:
        ml = ghmm.HMMFromMatrices(F, ghmm.GaussianDistribution(F), A, B, pi)
        
    l_likelihood_mean  = 0.0
    l_likelihood_mean2 = 0.0
    l_statePosterior = np.zeros(nState)

    for j in xrange(n):    

        g_post   = np.zeros(nState)
        g_lhood  = 0.0
        g_lhood2 = 0.0
        prop_sum = 0.0

        for k in xrange(1,m):
            final_ts_obj = ghmm.EmissionSequence(F, X_train[j][:k*nEmissionDim])
            logp         = ml.loglikelihoods(final_ts_obj)[0]
            post         = np.array(ml.posterior(final_ts_obj))

            k_prop       = norm(loc=g_mu, scale=g_sig).pdf(k)
            g_post      += post[k-1] * k_prop
            g_lhood     += logp * k_prop                    
            g_lhood2    += logp * logp * k_prop                    

            prop_sum  += k_prop

        l_statePosterior  += g_post / prop_sum / float(n)
        l_likelihood_mean += g_lhood / prop_sum / float(n)
        l_likelihood_mean2+= g_lhood2 / prop_sum / float(n)

    return i, l_statePosterior, l_likelihood_mean, np.sqrt(l_likelihood_mean2 - l_likelihood_mean**2)
    
