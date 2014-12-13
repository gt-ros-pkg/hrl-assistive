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
from matplotlib import animation

## import door_open_data as dod
import ghmm
import hrl_anomaly_detection.door_opening.mechanism_analyse_daehyung as mad
from scipy.stats import norm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score
from sklearn.base import clone
from sklearn import cross_validation
from scipy import optimize
## from pysmac.optimize import fmin                
from joblib import Parallel, delayed

from learning_base import learning_base
import sandbox_dpark_darpa_m3.lib.hrl_dh_lib as hdl




class learning_hmm(learning_base):
    def __init__(self, data_path, aXData, nState, nMaxStep, nFutureStep=5, fObsrvResol=0.2, nCurrentStep=10, step_size_list=None, trans_type="left_right"):

        learning_base.__init__(self, data_path, aXData, trans_type)
        
        ## Tunable parameters                
        self.nState= nState # the number of hidden states
        self.nFutureStep = nFutureStep
        self.fObsrvResol = fObsrvResol
        self.nCurrentStep = nCurrentStep
        self.step_size_list = step_size_list
        
        ## Un-tunable parameters
        ## self.trans_type = trans_type #"left_right" #"full"
        self.nMaxStep = nMaxStep  # the length of profile
        self.future_obsrv = None  # Future observation range
        self.A = None # transition matrix        

        self.B_lower=[]
        self.B_upper=[]
        for i in xrange(self.nState):
            self.B_lower.append([0.1])
            self.B_lower.append([0.1])
            self.B_upper.append([20.])
            self.B_upper.append([4.])

        self.B_upper =  np.array(self.B_upper).flatten()            
        self.B_lower =  np.array(self.B_lower).flatten()            
                
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
    def fit(self, X_train, A=None, B=None, pi=None, verbose=False):

        if A is None:        
            if verbose: print "Generate new A matrix"                
            # Transition probability matrix (Initial transition probability, TODO?)
            A = self.init_trans_mat(self.nState).tolist()
            #A,_ = mad.get_trans_mat(X_train, self.nState)

        if B is None:
            if verbose: print "Generate new B matrix"                            
            # We should think about multivariate Gaussian pdf.        
            self.mu, self.sigma = self.vectors_to_mean_vars(X_train, optimize=False)

            # Emission probability matrix
            B = np.hstack([self.mu, self.sigma]).tolist() # Must be [i,:] = [mu, sigma]
        else:
            if bool(np.all(B.flatten() >= self.B_lower)) == False:
                print "[Error]: negative component of B is not allowed"
                ## sys.exit()
                self.ml = None
                return
                
        if pi is None:            
            # pi - initial probabilities per state 
            ## pi = [1.0/float(self.nState)] * self.nState
            pi = [0.] * self.nState
            pi[0] = 1.0

        # HMM model object
        self.ml = ghmm.HMMFromMatrices(self.F, ghmm.GaussianDistribution(self.F), A, B, pi)
        ## self.ml = ghmm.HMMFromMatrices(self.F, ghmm.DiscreteDistribution(self.F), A, B, pi)
        
        ## print "Run Baum Welch method with (samples, length)", X_train.shape
        train_seq = X_train.tolist()
        final_seq = ghmm.SequenceSet(self.F, train_seq)        
        self.ml.baumWelch(final_seq)

        [self.A,self.B,self.pi] = self.ml.asMatrices()
        self.A = np.array(self.A)
        self.B = np.array(self.B)
        ## self.mean_path_plot(mu[:,0], sigma[:,0])        
        ## print "Completed to fitting", np.array(final_seq).shape

        # Future observation range
        self.max_obsrv = X_train.max()
        self.obsrv_range = np.arange(0.0, self.max_obsrv*1.2, self.fObsrvResol)
        self.state_range = np.arange(0, self.nState, 1)

        # Pre-computation for PHMM variables
        self.mu_z  = np.zeros((self.nState))
        self.mu_z2 = np.zeros((self.nState))
        self.var_z = np.zeros((self.nState))
        for i in xrange(self.nState):
            zp            = self.A[i,:]*self.state_range
            self.mu_z[i]  = np.sum(zp)
            self.mu_z2[i] = self.mu_z[i]**2
            self.var_z    = np.sum(zp*self.state_range) - self.mu_z[i]**2

        self.obs_prob = np.zeros((self.nState, len(self.obsrv_range)))
        # Get all probability over states
        for i in xrange(self.nState): 
            (x_mu, x_sigma) = self.B[i]
            self.obs_prob[i,:] = norm.pdf(self.obsrv_range,loc=x_mu,scale=x_sigma)
        

        if verbose:
            A = np.array(A)
            print "A: ", A.shape
            n,m = A.shape
            for i in xrange(n):
                a = None
                for j in xrange(m):
                    if a is None:
                        a = "%0.3f" % A[i,j]
                    else:
                        a += ",  "
                        a += "%0.3f" % A[i,j]
                print a
            print "----------------------------------------------"
            for i in xrange(n):
                a = None
                for j in xrange(m):
                    if a is None:
                        a = "[ %0.3f" % self.ml.getTransition(i,j)
                    else:
                        a += ",  "
                        a += "%0.3f" % self.ml.getTransition(i,j)
                print a + " ]"
                
            print "B: ", B.shape
            print B
            print "----------------------------------------------"
            ## for i in xrange(n):
            ##     print self.ml.getEmission(i)


    #----------------------------------------------------------------------        
    #
    def vectors_to_mean_vars(self, vecs, optimize=False):

        _,n   = vecs.shape # samples, length
        mu    = np.zeros((self.nState,1))
        sigma = np.zeros((self.nState,1))

        if self.step_size_list is None or len(self.step_size_list) != self.nState:
            print "Use new step size list!!"
            # Initial 
            self.step_size_list = [1] * self.nState
            while sum(self.step_size_list)!=self.nMaxStep:
                idx = int(random.gauss(float(self.nState)/2.0,float(self.nState)/2.0/2.0))
                if idx < 0 or idx >= self.nState: 
                    continue
                else:
                    self.step_size_list[idx] += 1                
        else:
            print "Use previous step size list!!"                            
            print self.step_size_list

        # Compute mean and std
        index = 0
        m_init = 0
        while (index < self.nState):
            temp_vec = vecs[:,(m_init):(m_init + int(self.step_size_list[index]))] 
            m_init = m_init + int(self.step_size_list[index])

            mu[index] = np.mean(temp_vec)
            sigma[index] = np.std(temp_vec)
            index = index+1

        return mu,sigma
               

    #----------------------------------------------------------------------        
    # B matrix optimization
    def param_optimization(self, save_file):

        _,n   = self.aXData.shape # samples, length
        mu    = np.zeros((self.nState,1))
        sigma = np.zeros((self.nState,1))


        # Initial 
        x0 = [1] * self.nState
        while sum(x0)!=self.nMaxStep:
            idx = int(random.gauss(float(self.nState)/2.0,float(self.nState)/2.0/2.0))
            if idx < 0 or idx >= self.nState: 
                continue
            else:
                x0[idx] += 1

        # Compute mean and std
        index = 0
        m_init = 0
        while (index < self.nState):
            temp_vec = self.aXData[:,(m_init):(m_init + int(x0[index]))] 
            m_init = m_init + int(x0[index])

            mu[index] = np.mean(temp_vec)
            sigma[index] = np.std(temp_vec)
            index = index+1

        B0 = np.hstack([mu, sigma]) # Must be [i,:] = [mu, sigma]

        ## hopping_step_size = np.zeros((self.nState*2))
        ## for i in xrange(self.nState):
        ##     hopping_step_size[i*2] = 2.0
        ##     hopping_step_size[i*2+1] = 0.5             

        class MyTakeStep(object):
            def __init__(self, stepsize=0.5, xmax=self.B_upper, xmin=self.B_lower):
                self.stepsize = stepsize
                self.xmax = xmax
                self.xmin = xmin
            def __call__(self, x):
                s = self.stepsize
                n = len(x)

                for i in xrange(n):
                    while True:

                        if i%2==0:                        
                            next_x = x[i] + np.random.uniform(-2.*s, 2.*s)                                
                        else:
                            next_x = x[i] + np.random.uniform(-0.5*s, 0.5*s)

                        if next_x > self.xmax[i] or next_x < self.xmin[i]:
                            continue
                        else:
                            x[i] = next_x
                            break
                                                                        
                ## for i in xrange(n/2):
                ##     x[i*2] += np.random.uniform(-2.*s, 2.*s)
                ##     x[i*2+1] += np.random.uniform(-0.5, 0.5)
                return x            

        class MyBounds(object):
            def __init__(self, xmax=self.B_upper, xmin=self.B_lower ):
                self.xmax = xmax
                self.xmin = xmin
            def __call__(self, **kwargs):
                x = kwargs["x_new"]
                tmax = bool(np.all(x <= self.xmax))
                tmin = bool(np.all(x >= self.xmin))
                return tmax and tmin

        def print_fun(x, f, accepted):
            print("at minima %.4f accepted %d" % (f, int(accepted)))


        bnds=[]
        for i in xrange(len(self.B_lower)):
            bnds.append([self.B_lower[i],self.B_upper[i]])
                        
        mytakestep = MyTakeStep()
        mybounds = MyBounds()
        minimizer_kwargs = {"method":"L-BFGS-B", "bounds":bnds}
        self.last_x = None

        # T
        res = optimize.basinhopping(self.mean_vars_score,B0.flatten(), minimizer_kwargs=minimizer_kwargs, niter=1000, take_step=mytakestep, accept_test=mybounds, callback=print_fun)
        # , stepsize=2.0, interval=2

        B = res['x'].reshape((self.nState,2))
        fval = res['fun']
        
        ## # Set range of params
        ## xmin = [0.0]* self.nState
        ## xmax = [self.nMaxStep-(self.nState-1)]*self.nState

        ## x_opt, fval = fmin(self.mean_vars_score,x0_int=x0, xmin_int=xmin, 
        ##                    xmax_int=xmax, max_evaluations=1000,
        ##                    custom_args={"self": self})

        ## print x_opt
        ## print fval

        ## self.step_size_list = x_opt['x_int'].tolist()
        ## print "Best step_size_list: "
        ## string = None
        ## for x in self.step_size_list:
        ##     if string == None:
        ##         string = str(x)+", " 
        ##     else:
        ##         string += str(x)+", "
        ## print string

        ## params_list = [{'nState': self.nState, 'fObsrvResol': self.fObsrvResol,'step_size_list': self.step_size_list}]
        ## params_list = [{'nState': self.nState, 'fObsrvResol': self.fObsrvResol,'B': B.flatten()}]

        # Save data
        data = {}
        data['score'] = [fval]
        data['nState'] = self.nState
        data['fObsrvResol'] = self.fObsrvResol
        data['B'] = B

        if save_file is None:
            save_file='tune_data.pkl'            
        ut.save_pickle(data, save_file)

        return 


    #----------------------------------------------------------------------        
    #
    def mean_vars_score(self, x, *args):

        # check limit
        if self.last_x is None or np.linalg.norm(self.last_x-x) > 0.05:
            tmax = bool(np.all(x <= self.B_upper))
            tmin = bool(np.all(x >= self.B_lower))
            if tmax and tmin == False: return 5            
            self.last_x = x
        else:
            return self.last_score
            
        B=x.reshape((self.nState,2))
        
        # K-fold CV: Split the dataset in two equal parts
        nFold = 8
        scores = cross_validation.cross_val_score(self, self.aXData, cv=nFold, fit_params={'B': B}, n_jobs=-1)
        ## scores = cross_validation.cross_val_score(self, self.aXData, cv=nFold, fit_params={'B': B})
        
        ## print x, " : ", -1.0 * sum(scores)/float(len(scores))
        self.last_score = -1.0 * sum(scores)/float(len(scores))
        return -1.0 * sum(scores)/float(len(scores))

        
    #----------------------------------------------------------------------        
    #
    def mean_vars_constraint1(self, x0):
        return np.sum(x0) - self.nMaxStep

    def mean_vars_constraint2(self, x0):
        for x in x0:
            if x < 1: return 1.0 
            if np.isnan(x)==True: return 1.0
        return 0.0
        
        
    #----------------------------------------------------------------------        
    #
    def predict(self, X):
        # Input
        # @ X_test: samples x known steps
        # @ x_pred: samples x 1
        # Output
        # @ probability: samples x 1 [list]

        if self.nCurrentStep > self.nMaxStep:
            print "ERROR: Current step over the max step"
            sys.exit()
        
        if type(X) == np.ndarray:
            X = X.tolist()

        n = len(X)
        prob = [0.0] * n            
            
        for i in xrange(n):

            ## if len(X[i]) > self.nCurrentStep+self.nFutureStep: #Full data                
            ##     X_test = X[i][:self.nCurrentStep]
            ##     X_pred = X[i][self.nCurrentStep:self.nCurrentStep+1]
            ## else:
            ##     X_test = X[i][:-1]
            ##     X_pred = X[i][-1]

            if len(X[i]) < self.nCurrentStep+1: 
                print "Why X is short??"
                sys.exit()
            X_test = X[i][:self.nCurrentStep]
            X_pred = X[i][self.nCurrentStep:self.nCurrentStep+1]
                

            bloglikelihood=False
            if bloglikelihood:
                
                # profile
                final_ts_obj = ghmm.EmissionSequence(self.F,X_test+[X_pred]) # is it neccessary?

                # log( P(O|param) )
                prob[i] = np.exp(self.ml.loglikelihood(final_ts_obj))
                
            else:

                # Past profile
                final_ts_obj = ghmm.EmissionSequence(self.F,X_test) # is it neccessary?
                #final_ts_obj = ghmm.EmissionSequence(self.F,X_test+[X_pred]) # is it neccessary?

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
                    prob[i] = 0.0
                    continue
                    
                # beta
                ## beta = self.ml.backward(final_ts_obj,scale)
                ## print "beta", np.array(beta).shape, " = \n " #+ str(beta) + "\n"

                ## scaling_factor = 1.0
                ## for k in xrange(len(scale)):
                ##     scaling_factor *= scale[k] 
                
                pred_numerator = 0.0
                ## pred_denominator = 0.0
                for j in xrange(self.nState): # N+1

                        
                    total = np.sum(self.A[:,j]*alpha[-1,:]) #* scaling_factor
                    [mu, sigma] = self.B[j]
                    
                    ## total = 0.0        
                    ## for k in xrange(self.nState): # N                  
                    ##     total += self.ml.getTransition(k,j) * alpha[self.nCurrentStep][k]

                    ## (mu, sigma) = self.ml.getEmission(j)

                    pred_numerator += norm.pdf(X_pred,loc=mu,scale=sigma) * total
                    ## pred_denominator += alpha[-1][j]*beta[self.nCurrentStep][j]

                prob[i] = pred_numerator #/ np.exp(self.ml.loglikelihood(final_ts_obj)) #/ pred_denominator
                
        return prob


    #----------------------------------------------------------------------        
    #
    def one_step_predict(self, X):
        # Input
        # @ X_test: 1 x known steps #samples x known steps
        # Output
        # @ X_pred: 1
        # @ X_pred_prob: obsrv_range

        # Initialization            
        X_pred_prob = np.zeros((len(self.obsrv_range)))
        if type(X) == np.ndarray:
            X = X.tolist()

        # Get all probability
        for i, obsrv in enumerate(self.obsrv_range):           
            if abs(X[-1]-obsrv) > 4.0: continue
            X_pred_prob[i] = self.predict([X+[obsrv]])[0]       

        # Select observation with maximum probability
        ## idx_list = [k[0] for k in sorted(enumerate(X_pred_prob), key=lambda x:x[1], reverse=True)]
        max_idx = X_pred_prob.argmax()

        sum_x_p = np.sum(X_pred_prob*X_pred_prob)
        if sum_x_p < 0.00001:
            return self.obsrv_range[max_idx], X_pred_prob*0.0
        else:
            ## return self.obsrv_range[idx_list[0]], X_pred_prob/np.sum(X_pred_prob)
            return self.obsrv_range[max_idx], X_pred_prob /np.sum(X_pred_prob)

        
    #----------------------------------------------------------------------        
    # Compute the estimated probability (0.0~1.0)
    def multi_step_predict(self, X_test, verbose=False):
        # Input: X, X_{N+M-1}, P(X_{N+M-1} | X)
        # Output:  P(X_{N+M} | X)

        # Initialization            
        X_pred_prob = np.zeros((len(self.obsrv_range),self.nFutureStep))
        X = copy.copy(X_test)
        X_pred = [0.0]*self.nFutureStep

        # Recursive prediction
        for i in xrange(self.nFutureStep):

            X_pred[i], X_pred_prob[:,i] = self.one_step_predict(X)

            # Udate 
            X.append(X_pred[i])

            if False: #verbose:
                print "-----------------"
                print X_pred_prob[:,i].shape
                a = None
                for p in X_pred_prob[:,i]:
                    if a is None:
                        a = "%0.3f" % p
                    else:
                        a += "  "
                        a += "%0.3f" % p
                print a

        ## # TEMP
        ## final_ts_obj = ghmm.EmissionSequence(self.F,X) # is it neccessary?                
        ## (path,log)   = self.ml.viterbi(final_ts_obj)
        ## print path,log
                
        return X_pred, X_pred_prob


    #----------------------------------------------------------------------        
    # Compute the estimated probability (0.0~1.0)
    def multi_step_approximated_predict(self, X_test, full_step=False, n_jobs=-1, verbose=False):
        ## print "Start to predict multistep approximated observations"

        ## Initialization            
        X_pred_prob = np.zeros((len(self.obsrv_range),self.nFutureStep))
        X_pred = [0.0]*self.nFutureStep

        ## Get mu, sigma from X such that u_{N+1}, sigma_{N+1}
        # Past profile
        final_ts_obj = ghmm.EmissionSequence(self.F,X_test) # is it neccessary?

        # alpha: X_test length y #latent States at the moment t when state i is ended
        #        test_profile_length x number_of_hidden_state
        #      : time step x nState (size)
        #        The sum of a row is 1.0 which means it is scaled one. Thus, original one
        #        must be divided by the scaling factor
        (alpha,scale) = self.ml.forward(final_ts_obj)
        alpha         = np.array(alpha)

        ## temp = alpha[-1,:]

        #scaling_factor = 1.0
        #for i in xrange(len(scale)):
        #    scaling_factor *= scale[i]

        p_z_x = np.zeros((self.nState))
        for i in xrange(self.nState):
            p_z_x[i] = np.sum(self.A[:,i]*alpha[-1,:]) #/ scaling_factor

        # Normalization?
        p_z_x /= np.sum(p_z_x)

        # (hidden space)
        if self.trans_type == "left_right":
            (u_xi, u_omega, u_alpha) = hdl.skew_normal_param_estimation(self.state_range, p_z_x)
            
            u_xi_list = [0.0]*self.nFutureStep
            u_omega_list = [0.0]*self.nFutureStep
            u_alpha_list = [0.0]*self.nFutureStep
            u_xi_list[0] = u_xi  # U_n+1 
            u_omega_list[0] = u_omega
            u_alpha_list[0] = u_alpha

            for i in xrange(self.nFutureStep-1):
                u_xi_list[i+1], u_omega_list[i+1], u_alpha_list[i+1] = self.skew_normal_approximation(u_xi_list[i], u_omega_list[i], u_alpha_list[i])
        else:
            (u_mu, u_var) = hdl.gaussian_param_estimation(self.state_range, p_z_x)

            u_mu_list  = [0.0]*self.nFutureStep
            u_sigma_list = [0.0]*self.nFutureStep
            u_mu_list[0]  = u_mu  # U_n+1 
            u_sigma_list[0] = np.sqrt(u_var)

            for i in xrange(self.nFutureStep-1):
                u_mu_list[i+1], u_sigma_list[i+1] = self.gaussian_approximation(u_mu_list[i], u_sigma_list[i])
                
                              
                
        # Compute all intermediate steps (observation space)
        if full_step:

            if self.trans_type == "left_right":
                r = Parallel(n_jobs=n_jobs)(delayed(f)(i, self.state_range, \
                                                       self.obsrv_range, \
                                                       self.obs_prob, \
                                                       u_xi_list[i], \
                                                       u_omega_list[i], \
                                                       u_alpha_list[i], \
                                                       self.trans_type) \
                                                       for i in xrange(self.nFutureStep) )
            else:
                r = Parallel(n_jobs=n_jobs)(delayed(f)(i, self.state_range, \
                                                       self.obsrv_range, \
                                                       self.obs_prob, \
                                                       u_mu_list[i], \
                                                       u_sigma_list[i], \
                                                       0, \
                                                       self.trans_type) \
                                                       for i in xrange(self.nFutureStep) )
                
                                              
            res, i = zip(*r)
            X_pred_prob = np.array(res).T 
            
            # Recursive prediction for each future step
            for i in xrange(self.nFutureStep):
                        
                ## max_idx = X_pred_prob[:,i].argmax()                    
                ## X_pred[i] = self.obsrv_range[max_idx]
                X_pred_prob[:,i] /= np.sum(X_pred_prob[:,i] + 0.000001)
                                
        else:
            print "Predict on last part!!"
            
            # Recursive prediction for each future step
            i = self.nFutureStep - 1

            # Get all probability over observations
            for j, obsrv in enumerate(self.obsrv_range):           

                # Get all probability over states
                for k in xrange(self.nState): 

                    z_prob = norm.pdf(float(k),loc=u_mu_list[i],scale=u_sigma_list[i])
                    (x_mu, x_sigma) = self.ml.getEmission(k)                        
                    X_pred_prob[j,i] += norm.pdf(self.obsrv_range[j],loc=x_mu,scale=x_sigma)*z_prob

                    if np.isnan(z_prob): sys.exit()

            max_idx = X_pred_prob[:,i].argmax()                    
            X_pred[i] = self.obsrv_range[max_idx]
            X_pred_prob[:,i] /= np.sum(X_pred_prob[:,i])

        ## return X_pred, X_pred_prob
        return None, X_pred_prob
        

    #----------------------------------------------------------------------        
    # Compute the estimated probability (0.0~1.0)
    def gaussian_approximation(self, u_mu, u_sigma):

        e_mu   = 0.0
        e_mu2  = 0.0
        e_var  = 0.0

        if u_sigma == 0.0: u_sigma = 0.00001
        p_z    = norm.pdf(np.arange(0.0,float(self.nState),1.0),loc=u_mu,scale=u_sigma)
        p_z    = p_z/np.sum(p_z)

        #
        e_mu  = np.sum(p_z*self.mu_z)
        e_mu2 = np.sum(p_z*self.mu_z2)
        e_var = np.sum(p_z*self.var_z)

        m = e_mu
        v = (e_var) + (e_mu2) - m**2

        ## print v, " = ", (e_var/sum(p_z)), (e_mu2/sum(p_z)), m**2

        if v < 0.0: print "Negative variance error: ", v
        if v == 0.0: v = 1.0e-6
        return m, np.sqrt(v)

        
    #----------------------------------------------------------------------        
    # Compute the estimated probability (0.0~1.0)
    def skew_normal_approximation(self, u_xi, u_omega, u_alpha=0.0):

        e_mu   = 0.0
        e_mu2  = 0.0
        e_var  = 0.0

        if u_omega == 0.0: u_omega = 0.00001
        p_z    = hdl.skew_normal_distribution(np.arange(0.0,float(self.nState),1.0),loc=u_xi,scale=u_omega,skewness=u_alpha)
        p_z    = p_z/np.sum(p_z)

        # Need to speed up!!
        e_mu  = np.sum(p_z*self.mu_z)
        e_mu2 = np.sum(p_z*self.mu_z2)
        e_var = np.sum(p_z*self.var_z)
        ## for i in xrange(self.nState):
            ## zp     = self.A[i,:]*self.state_range
            ## mu_z   = np.sum(zp)
            ## ## p_z[i] = norm.pdf(float(i),loc=u_mu,scale=u_sigma)
            
            ## e_mu   += p_z[i] * self.mu_z[i]
            ## e_mu2  += p_z[i] * mu_z**2
            ## e_var  += p_z[i] * ( np.sum(zp*self.state_range) - mu_z**2 )

        d = u_alpha / np.sqrt(1.0 + u_alpha**2)
        v = (e_var+e_mu2-e_mu**2) / (1. - 2.*d*d/np.pi)
        m = e_mu - np.sqrt(2.0 * v / np.pi) * d
        s = (4.0-np.pi)/2.0 * (e_mu)/(e_mu2-e_mu**2)

        ## print v, " = ", (e_var/sum(p_z)), (e_mu2/sum(p_z)), m**2

        ## if v < 0.0: print "Negative scale error: ", v
        if v == 0.0: v = 1.0e-6
        return m, np.sqrt(v), s
        
    #----------------------------------------------------------------------
    # Returns the mean accuracy on the given test data and labels.
    ## def score(self, X_test, **kwargs):
    ##     # Neccessary package
    ##     from sklearn.metrics import r2_score

    ##     # Get input
    ##     if type(X_test) == np.ndarray:
    ##         X=X_test.tolist()

    ##     sample_weight=None # TODO: future input

    ##     #
    ##     n = len(X)
    ##     score  = np.zeros((n))
    ##     X_next = np.zeros((n))
    ##     X_pred = np.zeros((n))

    ##     for i in xrange(n):

    ##         if len(X[i]) > self.nCurrentStep+self.nFutureStep: #Full data                
    ##             X_past = X[i][:self.nCurrentStep]
    ##             X_next[i] = X[i][self.nCurrentStep]
    ##         else:
    ##             X_past = X[i][:-1]
    ##             X_next[i] = X[i][-1]

    ##         X_pred[i], _ = self.one_step_predict(X_past)

    ##     return r2_score(X_next, X_pred, sample_weight=sample_weight)
            
    ##     ## from sklearn.metrics import accuracy_score
    ##     ## return accuracy_score(y_test, np.around(self.predict(X_test)), sample_weight=sample_weight)
        
    ##     ## return np.sum(score)/float(n)


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
            
            for i in xrange(n):
                if len(X[i]) > nStep+nFutureStep: #Full data                
                    X_past = X[i][:nStep]
                    X_next[i] = X[i][nStep]
                else:
                    print "Error: input should be full length data!!"
                    sys.exit()

                X_pred[i], _ = self.one_step_predict(X_past)

            total_score[j] = r2_score(X_next, X_pred, sample_weight=sample_weight)

        ## print "---------------------------------------------"
        ## print "Total Score"
        ## print total_score
        ## print "---------------------------------------------"
        return sum(total_score) / float(len(nCurrentStep))
        
        
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
    def predictive_path_plot(self, X_test, X_pred, X_pred_prob, X_test_next, X_test_all=None):
                
        # 1) Observation        
        self.ax.plot(X_test, 'k-o')    
        if X_test_all is not None:
            self.ax.plot(X_test_all, 'k-')    
        ## self.ax.plot(X_test, '-o', ms=10, lw=1, alpha=0.5, mfc='orange')    

        # 2) Next true & predictive observation
        y_array = np.hstack([X_test[-1], X_test_next])                        
        x_array = np.arange(0, len(X_test_next)+0.0001,1.0) + len(X_test) -1        
        self.ax.plot(x_array, y_array, 'k-', linewidth=1.0)    

        ## y_array = np.hstack([X_test[-1], X_pred])                        
        ## self.ax.plot(x_array, y_array, 'b-', linewidth=2.0)    
        
        # 3) Prediction        
        n,m = X_pred_prob.shape
        x_last = X_test[-1]
        mu = np.zeros((m))
        var = np.zeros((m))
        for i in xrange(m):
            
            ## x_pred_max = 0.0
            ## x_best     = 0.0            
            ## for x, prob in zip(self.obsrv_range, X_pred_prob[:,i]):
            ##     y_array = np.array([x_last, x])
                
            ##     alpha   = 1.0 - np.exp(-prob*50.0)
            ##     if alpha > 1.0: alpha = 1.0
            ##     self.ax.plot(x_array[i:i+2], y_array, 'r-', alpha=alpha, linewidth=1.0)    

            ## x_last = X_pred[i]
            (mu[i], var[i]) = hdl.gaussian_param_estimation(self.obsrv_range, X_pred_prob[:,i])

        # 4) mean var plot
        mu  = np.hstack([X_test,mu])
        sig = np.hstack([np.zeros((len(X_test))),np.sqrt(var)])
        X   = np.arange(0, len(mu),1.0)
        self.ax.fill_between(X, mu-1.*sig, mu+1.*sig, facecolor='yellow', alpha=0.5)
        self.ax.plot(X[len(X_test)-1:], mu[len(X_test)-1:], 'm-', linewidth=2.0)    
        self.ax.set_ylim([0.0, 1.2*self.obsrv_range[-1]])

        ## Side distribution
        self.ax1 = self.fig.add_subplot(self.gs[1])
        self.ax1.plot(X_pred_prob[:,-1], self.obsrv_range, 'r-')

        ## self.ax1.tick_params(\
        ##                      axis='y',          # changes apply to the x-axis
        ##                      which='both',      # both major and minor ticks are affected
        ##                      left='off',        # ticks along the bottom edge are off                             
        ##                      bottom='off',      # ticks along the bottom edge are off
        ##                      top='off',         # ticks along the top edge are off
        ##                      labelleft='off',   # labels along the bottom edge are off                             
        ##                      labelbottom='off') # labels along the bottom edge are off

        self.ax1.set_xlim([0.0, 1.0])
        self.ax1.set_ylim([0.0, 1.2*self.obsrv_range[-1]])
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
    def init_plot(self, bAni=False):
        print "Start to print out"
        
        self.fig = plt.figure(1)
        self.gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1]) 

        ## Main predictive distribution
        if bAni:
            self.ax = self.fig.add_subplot(111)
        else:            
            self.ax = self.fig.add_subplot(self.gs[0])
        

    #----------------------------------------------------------------------        
    #
    def final_plot(self):
        plt.rc('text', usetex=True)
        
        self.ax.set_xlabel(r'\textbf{Angle [}{^\circ}\textbf{]}', fontsize=22)
        self.ax.set_ylabel(r'\textbf{Applied Opening Force [N]}', fontsize=22)
        self.ax.set_xlim([0, self.nMaxStep])
        self.ax.set_ylim([0, max(self.obsrv_range)])
        
        plt.show()
        ## self.fig.savefig('/home/dpark/Dropbox/HRL/collision_detection_hsi_kitchen_pr2.pdf', format='pdf')

        
    #----------------------------------------------------------------------        
    #
    def animated_path_plot(self, X_test, Y_test, bReload=False):

        # Load data
        pkl_file = 'animation_data.pkl'
        if os.path.isfile(pkl_file) and bReload==False:
            print "Load saved pickle"
            data = ut.load_pickle(pkl_file)        
            X_test      = data['X_test']
            Y_test      = data['X_test']
            ## Y_pred      = data['Y_pred']
            Y_pred_prob = data['Y_pred_prob']
            mu          = data['mu']
            var         = data['var']
        else:        

            n = len(X_test)
            mu = np.zeros((n, self.nFutureStep))
            var = np.zeros((n, self.nFutureStep))
            
            for i in range(1,n,1):
                ## Y_pred, Y_pred_prob = self.multi_step_approximated_predict(Y_test[:i],full_step=True)
                _, Y_pred_prob = self.multi_step_approximated_predict(Y_test[:i],full_step=True)
                for j in range(self.nFutureStep):
                    print i,j, Y_pred_prob.shape
                    (mu[i,j], var[i,j]) = hdl.gaussian_param_estimation(self.obsrv_range, Y_pred_prob[:,j])

            print "Save pickle"                    
            data={}
            data['X_test'] = X_test
            data['X_test'] = Y_test                
            ## data['Y_pred'] =Y_pred
            data['Y_pred_prob']=Y_pred_prob
            data['mu']=mu
            data['var']=var
            ut.save_pickle(data, pkl_file)                
        print "---------------------------"

        
        
        ## fig = plt.figure()
        ## ax = plt.axes(xlim=(0, len(Y_test)), ylim=(0, 20))
        lAll, = self.ax.plot([], [], color='#66FFFF', lw=2)
        line, = self.ax.plot([], [], lw=2)
        lmean, = self.ax.plot([], [], 'm-', linewidth=2.0)    
        lvar1, = self.ax.plot([], [], '--', color='0.75', linewidth=2.0)    
        lvar2, = self.ax.plot([], [], '--', color='0.75', linewidth=2.0)    
        ## lvar , = self.ax.fill_between([], [], [], facecolor='yellow', alpha=0.5)

        
        def init():
            lAll.set_data([],[])
            line.set_data([],[])
            lmean.set_data([],[])
            lvar1.set_data([],[])
            lvar2.set_data([],[])
            ## lvar.set_data([],[], [])
            return lAll, line, lmean, lvar1, lvar2,

        def animate(i):
            x = np.arange(0.0, len(Y_test), 1.0)
            y = Y_test
            lAll.set_data(x, y)            
            
            x = np.arange(0.0, len(Y_test[:i]), 1.0)
            y = Y_test[:i]
            line.set_data(x, y)

            if i >= 1 and i < len(Y_test):# -self.nFutureStep:
                a_mu = np.hstack([y[-1], mu[i]])
                a_X  = np.arange(len(x)-1, len(x)+self.nFutureStep, 1.0)
                lmean.set_data( a_X, a_mu)
                
                a_sig = np.hstack([0, np.sqrt(var[i])])
                ## lvar.set_data( a_X, a_mu-1.*a_sig, a_mu+1.*a_sig)
                lvar1.set_data( a_X, a_mu-1.*a_sig)
                lvar2.set_data( a_X, a_mu+1.*a_sig)
            else:
                lmean.set_data([],[])
                lvar1.set_data([],[])
                lvar2.set_data([],[])
                ## lvar.set_data([],[],[])
           
            return lAll, line, lmean, lvar1, lvar2,

           
        anim = animation.FuncAnimation(self.fig, animate, init_func=init,
                                       frames=len(Y_test), interval=800, blit=True)
        plt.show()


## def f(i, nState, B, obsrv_range, u_mu, u_sigma, u_alpha, trans_type="full"):
def f(i, state_range, obsrv_range, obs_prob, u_mu, u_sigma, u_alpha, trans_type="full"):

    if trans_type == "left_right":        
        u_xi = u_mu
        u_omega = u_sigma

        # hidden state distribution
        z_prob = hdl.skew_normal_distribution(state_range, loc=u_xi, scale=u_omega, skewness=u_alpha)        
    else:
        # hidden state distribution
        z_prob = norm.pdf(state_range,loc=u_mu,scale=u_sigma)
        
    # Get all probability over observations
    X_pred_prob = np.zeros((len(obsrv_range)))    
    for j in xrange(len(obsrv_range)):
        X_pred_prob[j] = np.sum(obs_prob[:,j]*z_prob)
    
    
    ## # Get all probability over observations
    ## for j in xrange(len(obsrv_range)):
    
    ##     # Get all probability over states
    ##     for k in xrange(nState): 

    ##         if u_sigma != 0.0:
    ##             z_prob = norm.pdf(float(k),loc=u_mu,scale=u_sigma)
    ##         else:
    ##             if float(k) == u_mu: z_prob = 1.0
    ##             else: z_prob = 0.0

    ##         (x_mu, x_sigma) = B[k] #hmm.ml.getEmission(k)
    ##         X_pred_prob[j] += norm.pdf(obsrv_range[j],loc=x_mu,scale=x_sigma)*z_prob

    return X_pred_prob, i
                                              

