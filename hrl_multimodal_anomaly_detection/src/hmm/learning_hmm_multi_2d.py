#!/usr/local/bin/python

import numpy as np
import sys, os, copy
import cPickle as pickle
from scipy.stats import norm, entropy


# Matplot
import matplotlib.pyplot as plt
from matplotlib import gridspec

import ghmm
from sklearn.metrics import r2_score
from joblib import Parallel, delayed

class learning_hmm_multi_2d:
    def __init__(self, nState, nFutureStep=5, nCurrentStep=10, nEmissionDim=2, check_method='progress'):
        self.ml = None

        ## Tunable parameters
        self.nState = nState # the number of hidden states
        self.nGaussian = nState
        self.nFutureStep = nFutureStep
        self.nCurrentStep = nCurrentStep
        self.nEmissionDim = nEmissionDim
        
        ## Un-tunable parameters
        self.trans_type = 'left_right' # 'left_right' 'full'
        self.A = None # transition matrix        
        self.B = None # emission matrix
        self.pi = None # Initial probabilities per state
        self.check_method = check_method # ['global', 'progress']

        self.l_statePosterior = None
        self.ll_mu = None
        self.ll_std = None
        self.l_mean_delta = None
        self.l_std_delta = None
        self.l_mu = None
        self.l_std = None
        self.std_coff = None

        # emission domain of this model        
        self.F = ghmm.Float()  

        print 'HMM initialized for', self.check_method

    def fit(self, xData1, xData2=None, A=None, B=None, pi=None, cov_mult=(1.0, 1.0, 1.0, 1.0), verbose=False, ml_pkl='ml_temp.pkl', use_pkl=False):
        ml_pkl = os.path.join(os.path.dirname(__file__), ml_pkl)
        X1 = np.array(xData1)
        X2 = np.array(xData2)
            
        if A is None:        
            if verbose: print "Generating a new A matrix"
            # Transition probability matrix (Initial transition probability, TODO?)
            A = self.init_trans_mat(self.nState).tolist()

        if B is None:
            if verbose: print "Generating a new B matrix"
            # We should think about multivariate Gaussian pdf.  

            mu1, mu2, cov = self.vectors_to_mean_cov(X1, X2, self.nState)
            cov[:, 0, 0] *= cov_mult[0] #1.5 # to avoid No convergence warning
            cov[:, 1, 0] *= cov_mult[1] #5.5 # to avoid No convergence warning
            cov[:, 0, 1] *= cov_mult[2] #5.5 # to avoid No convergence warning
            cov[:, 1, 1] *= cov_mult[3] #5.5 # to avoid No convergence warning

            # Emission probability matrix
            B = [0.0] * self.nState
            for i in range(self.nState):
                B[i] = [[mu1[i], mu2[i]], [cov[i,0,0], cov[i,0,1], cov[i,1,0], cov[i,1,1]]]
                                            
        if pi is None:            
            # pi - initial probabilities per state 
            ## pi = [1.0/float(self.nState)] * self.nState
            pi = [0.0] * self.nState
            pi[0] = 1.0

        # HMM model object
        self.ml = ghmm.HMMFromMatrices(self.F, ghmm.MultivariateGaussianDistribution(self.F), A, B, pi)
        X_train = self.convert_sequence(X1, X2) # Training input
        X_train = X_train.tolist()
        
            
        print 'Run Baum Welch method with (samples, length)', np.shape(X_train)
        final_seq = ghmm.SequenceSet(self.F, X_train)        
        ## ret = self.ml.baumWelch(final_seq, loglikelihoodCutoff=2.0)
        ret = self.ml.baumWelch(final_seq, 10000)
        print 'Baum Welch return:', ret

        [self.A, self.B, self.pi] = self.ml.asMatrices()
        self.A = np.array(self.A)
        self.B = np.array(self.B)

        #--------------- learning for anomaly detection ----------------------------
        [A, B, pi] = self.ml.asMatrices()
        n, m = np.shape(X1)
        self.nGaussian = self.nState

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
            self.std_coff  = 1.0
            g_mu_list = np.linspace(0, m-1, self.nGaussian) #, dtype=np.dtype(np.int16))
            g_sig     = float(m) / float(self.nGaussian) * self.std_coff

            ######################################################################################
            if os.path.isfile(ml_pkl) and use_pkl:
                with open(ml_pkl, 'rb') as f:
                    d = pickle.load(f)
                    self.l_statePosterior = d['state_post'] # time x state division
                    self.ll_mu            = d['ll_mu']
                    self.ll_std           = d['ll_std']
            else:        
                n_jobs = -1
                r = Parallel(n_jobs=n_jobs)(delayed(learn_likelihoods_progress)(i, n, m, A, B, pi, self.F, X_train,
                                                                       self.nEmissionDim, g_mu_list[i], g_sig, self.nState)
                                                                       for i in xrange(self.nGaussian))
                l_i, self.l_statePosterior, self.ll_mu, self.ll_std = zip(*r)

                d = dict()
                d['state_post'] = self.l_statePosterior
                d['ll_mu'] = self.ll_mu
                d['ll_std'] = self.ll_std
                with open(ml_pkl, 'wb') as f:
                    pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)

    def predict(self, X):
        X = np.squeeze(X)
        X_test = X.tolist()

        mu_l  = np.zeros(2) 
        cov_l = np.zeros(4)

        print self.F
        final_ts_obj = ghmm.EmissionSequence(self.F, X_test) # is it neccessary?

        try:
            # alpha: X_test length y # latent States at the moment t when state i is ended
            # test_profile_length x number_of_hidden_state
            (alpha, scale) = self.ml.forward(final_ts_obj)
            alpha = np.array(alpha)
        except:
            print "No alpha is available !!"
            
        f = lambda x: round(x,12)
        for i in range(len(alpha)):
            alpha[i] = map(f, alpha[i])
        alpha[-1] = map(f, alpha[-1])
        
        n = len(X_test)
        pred_numerator = 0.0

        for j in xrange(self.nState): # N+1
            total = np.sum(self.A[:,j]*alpha[n/self.nEmissionDim-1,:]) #* scaling_factor
            [[mu1, mu2], [cov11, cov12, cov21, cov22]] = self.B[j]

            ## print mu1, mu2, cov11, cov12, cov21, cov22, total
            pred_numerator += total
            
            mu_l[0] += mu1*total
            mu_l[1] += mu2*total
            cov_l[0] += cov11 * (total**2)
            cov_l[1] += cov12 * (total**2)
            cov_l[2] += cov21 * (total**2)
            cov_l[3] += cov22 * (total**2)

        return mu_l, cov_l

    def predict2(self, X, x1, x2):
        X = np.squeeze(X)
        X_test = X.tolist()        
        n = len(X_test)

        mu_l  = np.zeros(2)
        cov_l = np.zeros(4)

        final_ts_obj = ghmm.EmissionSequence(self.F, X_test + [x1, x2])

        try:
            (alpha, scale) = self.ml.forward(final_ts_obj)
        except:
            print "No alpha is available !!"
            sys.exit()

        alpha = np.array(alpha)

        for j in xrange(self.nState):
            
            [[mu1, mu2], [cov11, cov12, cov21, cov22]] = self.B[j]

            mu_l[0] = x1
            mu_l[1] += alpha[n/self.nEmissionDim,j]*(mu2 + cov21/cov11*(x1 - mu1) )
            ## cov_l[0] += (cov11)*(total**2)
            ## cov_l[1] += (cov12)*(total**2)
            ## cov_l[2] += (cov21)*(total**2)
            ## cov_l[3] += (cov22)*(total**2)

        return mu_l, cov_l
        
    def loglikelihood(self, X):

        X = np.squeeze(X)
        X_test = X.tolist()        

        final_ts_obj = ghmm.EmissionSequence(self.F, X_test)

        try:    
            p = self.ml.loglikelihood(final_ts_obj)
        except:
            print 'Likelihood error!!!!'
            sys.exit()

        return p
        
    # Returns the mean accuracy on the given test data and labels.
    def score(self, X_test):
        if self.ml is None:
            print 'No ml!!'
            return -5.0        
        
        # Get input
        if type(X_test) == np.ndarray:
            X = X_test.tolist()
        else:
            X = X_test

        sample_weight = None # TODO: future input
        
        n = len(X)
        nCurrentStep = [5,10,15,20,25]
        nFutureStep = 1

        total_score = np.zeros((len(nCurrentStep)))
        for j, nStep in enumerate(nCurrentStep):

            self.nCurrentStep = nStep
            X_next = np.zeros(n)
            X_pred = np.zeros(n)
            mu_pred  = np.zeros(n)
            var_pred = np.zeros(n)
            
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

        return sum(total_score) / float(len(nCurrentStep))

    # Returns mu,sigma for n hidden-states from feature-vector
    @staticmethod
    def vectors_to_mean_sigma(vec, nState):
        index = 0
        m,n = np.shape(vec)
        mu  = np.zeros(nState)
        sig = np.zeros(nState)
        DIVS = n/nState

        while index < nState:
            m_init = index*DIVS
            temp_vec = vec[:, m_init:(m_init+DIVS)]
            temp_vec = np.reshape(temp_vec, (1, DIVS*m))
            mu[index]  = np.mean(temp_vec)
            sig[index] = np.std(temp_vec)
            index = index+1

        return mu, sig
        
    # Returns mu,sigma for n hidden-states from feature-vector
    @staticmethod
    def vectors_to_mean_cov(vec1, vec2, nState):
        index = 0
        m, n = np.shape(vec1)
        #print m,n
        mu_1 = np.zeros(nState)
        mu_2 = np.zeros(nState)
        cov = np.zeros((nState,2,2))
        DIVS = n/nState


        while index < nState:
            m_init = index*DIVS
            temp_vec1 = vec1[:, m_init:(m_init+DIVS)]
            temp_vec2 = vec2[:, m_init:(m_init+DIVS)]
            temp_vec1 = np.reshape(temp_vec1,(1,DIVS*m))
            temp_vec2 = np.reshape(temp_vec2,(1,DIVS*m))
            mu_1[index] = np.mean(temp_vec1)
            mu_2[index] = np.mean(temp_vec2)
            cov[index,:,:] = np.cov(np.concatenate((temp_vec1,temp_vec2),axis=0))
            index = index+1

        return mu_1,mu_2,cov

    @ staticmethod
    def init_trans_mat(nState):
        # Reset transition probability matrix
        trans_prob_mat = np.zeros((nState, nState))

        for i in xrange(nState):
            # Exponential function
            # From y = a*e^(-bx)
            #a = 0.4
            #b = np.log(0.00001/a)/(-(nState-i))
            #f = lambda x: a*np.exp(-b*x)

            # Exponential function
            # From y = -a*x + b
            b = 0.4
            a = b/float(nState)
            f = lambda x: -a*x+b

            for j in np.array(range(nState-i))+i:
                trans_prob_mat[i,j] = f(j)

            # Gaussian transition probability
            ## z_prob = norm.pdf(float(i),loc=u_mu_list[i],scale=u_sigma_list[i])

            # Normalization
            trans_prob_mat[i,:] /= np.sum(trans_prob_mat[i,:])

        return trans_prob_mat

    def init_plot(self):
        print "Start to print out"

        self.fig = plt.figure(1)
        gs = gridspec.GridSpec(2, 1) 
        
        self.ax1 = self.fig.add_subplot(gs[0])        
        self.ax2 = self.fig.add_subplot(gs[1])        

    def data_plot(self, X_test1, X_test2, color='r'):
        self.ax1.plot(np.hstack([X_test1[0]]), color)
        self.ax2.plot(np.hstack([X_test2[0]]), color)

    @staticmethod
    def final_plot():
        plt.rc('text', usetex=True)
        plt.show()

    @staticmethod
    def convert_sequence(data1, data2, emission=False):
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

        n, m = np.shape(X1)

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
            min_dist  = 100000000
            min_index = 0
            for j in xrange(self.nGaussian):
                dist = entropy(post[n-1], self.l_statePosterior[j])
                if min_dist > dist:
                    min_index = j
                    min_dist  = dist 

            if (type(ths_mult) == list or type(ths_mult) == np.ndarray or type(ths_mult) == tuple) and len(ths_mult)>1:
                err = logp - (self.ll_mu[min_index] + ths_mult[min_index]*self.ll_std[min_index])
            else:
                err = logp - (self.ll_mu[min_index] + ths_mult*self.ll_std[min_index])
                    
        return err < 0.0, err

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
    
