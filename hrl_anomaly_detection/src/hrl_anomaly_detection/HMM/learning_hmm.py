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
from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score
from sklearn import cross_validation
from scipy import optimize
## from pysmac.optimize import fmin                

from learning_base import learning_base
import sandbox_dpark_darpa_m3.lib.hrl_dh_lib as hdl




class learning_hmm(learning_base):
    def __init__(self, data_path, aXData, nState, nMaxStep, nFutureStep=5, fObsrvResol=0.2, nCurrentStep=10, step_size_list=None):

        learning_base.__init__(self, data_path, aXData)
        
        ## Tunable parameters                
        self.nState= nState # the number of hidden states
        self.nFutureStep = nFutureStep
        self.fObsrvResol = fObsrvResol
        self.nCurrentStep = nCurrentStep
        self.step_size_list = step_size_list
        
        ## Un-tunable parameters
        self.nMaxStep = nMaxStep  # the length of profile
        self.future_obsrv = None  # Future observation range
        self.A = None # transition matrix        

        self.B_lower=[]
        self.B_upper=[]
        for i in xrange(self.nState):
            self.B_lower.append([0.1])
            self.B_lower.append([0.01])
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
    def fit(self, X_train, A=None, B=None, verbose=False):

        if A==None:        
            if verbose: print "Generate new A matrix"                
            # Transition probability matrix (Initial transition probability, TODO?)
            A = self.init_trans_mat(self.nState).tolist()
            #A,_ = mad.get_trans_mat(X_train, self.nState)

        if B==None:
            if verbose: print "Generate new B matrix"                            
            # We should think about multivariate Gaussian pdf.        
            self.mu, self.sigma = self.vectors_to_mean_vars(X_train, optimize=False)

            # Emission probability matrix
            B = np.hstack([self.mu, self.sigma]).tolist() # Must be [i,:] = [mu, sigma]
        else:
            if bool(np.all(B.flatten() >= self.B_lower)) == False:
                print "[Error]: negative component of B is not allowed"
                ## sys.exit()
                return
                
        
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

        if verbose:
            A = np.array(A)
            print "A: ", A.shape
            n,m = A.shape
            for i in xrange(n):
                a = None
                for j in xrange(m):
                    if a==None:
                        a = "%0.3f" % A[i,j]
                    else:
                        a += ",  "
                        a += "%0.3f" % A[i,j]
                print a
            print "----------------------------------------------"
            for i in xrange(n):
                a = None
                for j in xrange(m):
                    if a==None:
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

        if self.step_size_list == None or len(self.step_size_list) != self.nState:
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


        class MyTakeStep(object):
            def __init__(self, stepsize=0.5):
                self.stepsize = stepsize
            def __call__(self, x):
                s = self.stepsize
                n = len(x)

                for i in xrange(n/2):
                    x[i*2] += np.random.uniform(-2.*s, 2.*s)
                    x[i*2+1] += np.random.uniform(-0.5, 0.5)
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
            
            
        ## res = optimize.minimize(self.mean_vars_score,B0.flatten(), method='SLSQP', bounds=tuple(bnds), 
        ##                         options={'maxiter': 50})

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
        params_list = [{'nState': self.nState, 'fObsrvResol': self.fObsrvResol,'B': B.flatten()}]

        # Save data
        data = {}
        data['mean'] = [fval]
        data['std'] = [0]            
        data['params'] = params_list
        if save_file == None:
            save_file='tune_data.pkl'            
        ut.save_pickle(data, save_file)

        return 


    #----------------------------------------------------------------------        
    #
    def mean_vars_score(self, x, *args):

        # check limit
        if self.last_x == None or np.linalg.norm(self.last_x-x) > 0.05:
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


        ## # constraint check
        ## if self.mean_vars_constraint1(x_int) != 0:
        ##     return 100000000000000000
        ## if self.mean_vars_constraint2(x_int) > 0:
        ##     return 100000000000000000

        ## # init
        ## mu    = np.zeros((self.nState,1))
        ## sigma = np.zeros((self.nState,1))

        ## index = 0
        ## m_init = 0
        ## while (index < self.nState):
        ##     temp_vec = self.aXData[:,(m_init):(m_init + int(x_int[index]))] 
        ##     m_init = m_init + int(x_int[index])

        ##     mu[index] = np.mean(temp_vec)
        ##     sigma[index] = np.std(temp_vec)
        ##     index = index+1

        ## print x
        ## n,_ = x.shape
        ## score_list = [0.0]*n

        ## for i in xrange(n):
        
            ## B = np.hstack([mu, sigma]).tolist() # Must be [i,:] = [mu, sigma]
        
        ## print "loop ",i, "/",n," : ",-1.0 * sum(scores)/float(len(scores)), x
        ## score_list[i] = -1.0 * sum(scores)/float(len(scores))

        
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

            if len(X[i]) > self.nCurrentStep+self.nFutureStep: #Full data                
                X_test = X[i][:self.nCurrentStep]
                X_pred = X[i][self.nCurrentStep:self.nCurrentStep+1]
            else:
                X_test = X[i][:-1]
                X_pred = X[i][-1]

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
            
                # alpha: X_test length y #latent States at the moment t when state i is ended
                #        test_profile_length x number_of_hidden_state
                (alpha,scale) = self.ml.forward(final_ts_obj)
                alpha         = np.array(alpha)
                scale         = np.array(scale)
                ## print "alpha: ", np.array(alpha).shape,"\n" #+ str(alpha) + "\n"
                ## print "scale = " + str(scale) + "\n"
                
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
                    if a==None:
                        a = "%0.3f" % p
                    else:
                        a += "  "
                        a += "%0.3f" % p
                print a


        # TEMP
        final_ts_obj = ghmm.EmissionSequence(self.F,X) # is it neccessary?                
        (path,log)   = self.ml.viterbi(final_ts_obj)
        print path,log

                
        return X_pred, X_pred_prob


    #----------------------------------------------------------------------        
    # Compute the estimated probability (0.0~1.0)
    def multi_step_approximated_predict(self, X_test, verbose=False):

        ## Initialization            
        X_pred_prob = np.zeros((len(self.obsrv_range),self.nFutureStep))
        X = copy.copy(X_test)
        X_pred = [0.0]*self.nFutureStep

        ## Get mu, sigma from X such that u_{N+1}, sigma_{N+1}
        # Past profile
        final_ts_obj = ghmm.EmissionSequence(self.F,X_test) # is it neccessary?

        # alpha: X_test length y #latent States at the moment t when state i is ended
        #        test_profile_length x number_of_hidden_state
        (alpha,scale) = self.ml.forward(final_ts_obj)
        alpha         = np.array(alpha)

        scaling_factor = 1.0
        for i in xrange(len(scale)):
            scaling_factor *= scale[i] 

        p_z_x = np.zeros((self.nState))
        for i in xrange(self.nState):
            p_z_x[i] = np.sum(self.A[:,i]*alpha[-1,:]) * scaling_factor

        (u_mu, u_var) = ldh.gaussian_param_estimation(self.state_range, p_z_x)

        ## for i in xrange(self.nFutureStep-1):
        ##     some thing...
        
        ## _, X_pred_prob[:,i] = self.one_step_predict(X)
        
            
        # Recursive prediction
        ## for i in xrange(self.nFutureStep):

            
            
        return X_pred, X_pred_prob

    #----------------------------------------------------------------------        
    # Compute the estimated probability (0.0~1.0)
    def gaussian_approximation(self, u_mu, u_var):

        u_sigma = np.sqrt(u_var)

        e_mu   = 0.0
        e_mu2  = 0.0
        e_sig2 = 0.0
        for i in xrange(self.nState):

            zp    = self.A[i,:]*self.state_range
            mu_z  = np.sum(zp)
            p_z   = norm.pdf(float(i),loc=u_mu,scale=u_sigma)
            
            e_mu   += p_z * mu_z
            e_mu2  += p_z * mu_z**2
            e_sig2 += p_z * ( np.sum(zp*self.state_range) - mu_z**2 )**2

        m = e_mu
        v = e_sig2*e_mu2 - m**2
            
        return m, v
       
        
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
    def predictive_path_plot(self, X_test, X_pred, X_pred_prob, X_test_next):
        print "Start to print out"
        
        self.fig = plt.figure(1)
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1]) 

        ## Main predictive distribution
        self.ax = self.fig.add_subplot(gs[0])

        # 1) Observation        
        self.ax.plot(X_test, 'k-o')    
        ## self.ax.plot(X_test, '-o', ms=10, lw=1, alpha=0.5, mfc='orange')    

        # 2) Next true & predictive observation
        y_array = np.hstack([X_test[-1], X_test_next])                        
        x_array = np.arange(0, len(X_test_next)+0.0001,1.0) + len(X_test) -1        
        self.ax.plot(x_array, y_array, 'k-', linewidth=1.0)    

        y_array = np.hstack([X_test[-1], X_pred])                        
        self.ax.plot(x_array, y_array, 'b-', linewidth=2.0)    
        
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
        var = np.hstack([np.zeros((len(X_test))),var])
        X   = np.arange(0, len(mu),1.0)
        self.ax.fill_between(X, mu-2.*var, mu+2.*var, facecolor='yellow', alpha=0.5)
        self.ax.set_ylim([0.0, 1.2*self.obsrv_range[-1]])

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
    def final_plot(self):
        self.ax.set_xlabel("Angle")
        self.ax.set_ylabel("Force")
        
        plt.show()
        ## self.fig.savefig('/home/dpark/Dropbox/HRL/collision_detection_hsi_kitchen_pr2.pdf', format='pdf')

        











    
