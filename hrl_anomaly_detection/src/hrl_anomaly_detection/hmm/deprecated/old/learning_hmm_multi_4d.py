#!/usr/local/bin/python

import numpy as np
import sys, os, copy
from scipy.stats import norm, entropy

# Util
import roslib
roslib.load_manifest('hrl_anomaly_detection')
import hrl_lib.util as ut

# Matplot
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.collections as collections

import ghmm
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
from joblib import Parallel, delayed

os.system("taskset -p 0xff %d" % os.getpid())

class learning_hmm_multi_4d:
    def __init__(self, nState, nEmissionDim=4, check_method='progress', anomaly_offset=0.0, \
                 cluster_type='time', verbose=False):
        self.ml = None

        ## Tunable parameters
        self.nState = nState # the number of hidden states
        self.nGaussian = nState
        self.nEmissionDim = nEmissionDim
        self.verbose = verbose
        
        ## Un-tunable parameters
        self.trans_type = 'left_right' # 'left_right' 'full'
        self.A = None # transition matrix        
        self.B = None # emission matrix
        self.pi = None # Initial probabilities per state
        self.check_method = check_method # ['global', 'progress']
        self.cluster_type = cluster_type

        self.l_statePosterior = None
        self.ll_mu = None
        self.ll_std = None
        self.l_mean_delta = None
        self.l_std_delta = None
        self.l_mu = None
        self.l_std = None
        self.std_coff = None

        self.anomaly_offset=anomaly_offset

        # emission domain of this model        
        self.F = ghmm.Float()  

        # print 'HMM initialized for', self.check_method

    def fit(self, xData1, xData2, xData3, xData4, A=None, B=None, pi=None, cov_mult=[1.0]*16, \
            ml_pkl='ml_temp_4d.pkl', use_pkl=False):
        ml_pkl = os.path.join(os.path.dirname(__file__), ml_pkl)
        X1 = np.array(xData1)
        X2 = np.array(xData2)
        X3 = np.array(xData3)
        X4 = np.array(xData4)

        if A is None:        
            if self.verbose: print "Generating a new A matrix"
            # Transition probability matrix (Initial transition probability, TODO?)
            A = self.init_trans_mat(self.nState).tolist()

            # print 'A', A

        if B is None:
            if self.verbose: print "Generating a new B matrix"
            # We should think about multivariate Gaussian pdf.  

            mu1, mu2, mu3, mu4, cov = self.vectors_to_mean_cov(X1, X2, X3, X4, self.nState)
            for i in xrange(self.nEmissionDim):
                for j in xrange(self.nEmissionDim):
                    cov[:, j, i] *= cov_mult[self.nEmissionDim*i + j]

            if self.verbose:
                print 'mu1:', mu1
                print 'mu2:', mu2
                print 'mu3:', mu3
                print 'mu4:', mu4
                print 'cov', cov
                
            # Emission probability matrix
            B = [0.0] * self.nState
            for i in range(self.nState):
                B[i] = [[mu1[i], mu2[i], mu3[i], mu4[i]], [cov[i,0,0], cov[i,0,1], cov[i,0,2], cov[i,0,3],
                                                           cov[i,1,0], cov[i,1,1], cov[i,1,2], cov[i,1,3],
                                                           cov[i,2,0], cov[i,2,1], cov[i,2,2], cov[i,2,3],
                                                           cov[i,3,0], cov[i,3,1], cov[i,3,2], cov[i,3,3]]]
        if pi is None:
            # pi - initial probabilities per state 
            ## pi = [1.0/float(self.nState)] * self.nState
            pi = [0.0] * self.nState
            pi[0] = 1.0
            # pi[0] = 0.3
            # pi[1] = 0.3
            # pi[2] = 0.2
            # pi[3] = 0.2

        # print 'Generating HMM'
        # HMM model object
        self.ml = ghmm.HMMFromMatrices(self.F, ghmm.MultivariateGaussianDistribution(self.F), A, B, pi)
        # print 'Creating Training Data'
        X_train = self.convert_sequence(X1, X2, X3, X4) # Training input
        X_train = X_train.tolist()
        
        if self.verbose: print 'Run Baum Welch method with (samples, length)', np.shape(X_train)
        final_seq = ghmm.SequenceSet(self.F, X_train)
        ## ret = self.ml.baumWelch(final_seq, loglikelihoodCutoff=2.0)
        ret = self.ml.baumWelch(final_seq, 10000)
        print 'Baum Welch return:', ret
        if np.isnan(ret): return 'Failure'

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

            if self.verbose: 
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

            if os.path.isfile(ml_pkl) and use_pkl:
                d = ut.load_pickle(ml_pkl)
                self.l_statePosterior = d['state_post'] # time x state division
                self.ll_mu            = d['ll_mu']
                self.ll_std           = d['ll_std']
            else:
                if self.cluster_type == 'time':                
                    if self.verbose: print 'Begining parallel job'
                    self.std_coff  = 1.0
                    g_mu_list = np.linspace(0, m-1, self.nGaussian) #, dtype=np.dtype(np.int16))
                    g_sig = float(m) / float(self.nGaussian) * self.std_coff
                    r = Parallel(n_jobs=-1)(delayed(learn_likelihoods_progress)(i, n, m, A, B, pi, self.F, X_train,
                                                                           self.nEmissionDim, g_mu_list[i], g_sig, self.nState)
                                                                           for i in xrange(self.nGaussian))
                    if self.verbose: print 'Completed parallel job'
                    l_i, self.l_statePosterior, self.ll_mu, self.ll_std = zip(*r)

                elif self.cluster_type == 'state':
                    self.km = None                    
                    self.ll_mu = None
                    self.ll_std = None
                    self.ll_mu, self.ll_std = self.state_clustering(X1, X2, X3, X4)
                    path_mat  = np.zeros((self.nState, m*n))
                    likelihood_mat = np.zeros((1, m*n))
                    self.l_statePosterior=None
                    
                d = dict()
                d['state_post'] = self.l_statePosterior
                d['ll_mu'] = self.ll_mu
                d['ll_std'] = self.ll_std
                ut.save_pickle(d, ml_pkl)
                            
                    
    def get_sensitivity_gain(self, X1, X2, X3, X4):

        X_test = self.convert_sequence(X1, X2, X3, X4, emission=False)

        try:
            final_ts_obj = ghmm.EmissionSequence(self.F, X_test[0].tolist())
            logp = self.ml.loglikelihood(final_ts_obj)
        except:
            if self.verbose: print "Too different input profile that cannot be expressed by emission matrix"
            return [], 0.0 # error

        if self.check_method == 'progress':
            try:
                post = np.array(self.ml.posterior(final_ts_obj))
            except:
                if self.verbose: print "Unexpected profile!! GHMM cannot handle too low probability. Underflow?"
                return [], 0.0 # anomaly

            n = len(np.squeeze(X1))

            # Find the best posterior distribution
            min_index, min_dist = self.findBestPosteriorDistribution(post[n-1])

            ths = (logp - self.ll_mu[min_index])/self.ll_std[min_index]
            ## if logp >= 0.:                
            ##     ths = (logp*0.95 - self.ll_mu[min_index])/self.ll_std[min_index]
            ## else:
            ##     ths = (logp*1.05 - self.ll_mu[min_index])/self.ll_std[min_index]
                        
            return ths, min_index

        elif self.check_method == 'global':
            ths = (logp - self.l_mu) / self.l_std
            return ths, 0

        elif self.check_method == 'change':
            if len(X1)<3: return [], 0.0 #error

            X_test = self.convert_sequence(X1[:-1], X2[:-1], X3[:-1], X4[:-1], emission=False)                

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

            X_test = self.convert_sequence(X1[:-1], X2[:-1], X3[:-1], X4[:-1], emission=False)                

            try:
                final_ts_obj = ghmm.EmissionSequence(self.F, X_test[0].tolist())
                last_logp         = self.ml.loglikelihood(final_ts_obj)
            except:
                print "Too different input profile that cannot be expressed by emission matrix"
                return [], 0.0 # error
            
            ths_c = -(( abs(logp-last_logp) - self.l_mean_delta) / self.l_std_delta)

            ths_g = (logp - self.l_mu) / self.l_std
            
            return [ths_c, ths_g], 0
        

    def path_disp(self, X1, X2, X3, X4):
        X1 = np.array(X1)
        X2 = np.array(X2)
        X3 = np.array(X3)
        X4 = np.array(X4)
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        n, m = np.shape(X1)
        if self.verbose: print n, m
        x = np.arange(0., float(m))*(1./43.)
        path_mat  = np.zeros((self.nState, m))
        zbest_mat = np.zeros((self.nState, m))

        path_l = []
        for i in xrange(n):
            x_test1 = X1[i:i+1,:]
            x_test2 = X2[i:i+1,:]
            x_test3 = X3[i:i+1,:]
            x_test4 = X4[i:i+1,:]

            if self.nEmissionDim == 1:
                X_test = x_test1
            else:
                X_test = self.convert_sequence(x_test1, x_test2, x_test3, x_test4, emission=False)

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

        # maxim = np.max(path_mat)
        # path_mat = maxim - path_mat

        zbest_mat /= np.sum(zbest_mat, axis=0)

        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42

        fig = plt.figure()
        plt.rc('text', usetex=True)

        ax1 = plt.subplot(111)
        im  = ax1.imshow(path_mat, cmap=plt.cm.Reds, interpolation='none', origin='upper',
                         extent=[0, float(m)*(1.0/10.), 20, 1], aspect=0.85)

        ## divider = make_axes_locatable(ax1)
        ## cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, fraction=0.031, ticks=[0.0, 1.0], pad=0.01)
        ax1.set_xlabel("Time (sec)", fontsize=18)
        ax1.set_ylabel("Hidden State Index", fontsize=18)
        ax = plt.gca()
        ax.set_ylim(ax.get_ylim()[::-1])

        ## for p in path_l:
        ##     ax1.plot(x, p, '*')

        ## ax2 = plt.subplot(212)
        ## im2 = ax2.imshow(zbest_mat, cmap=plt.cm.Reds, interpolation='none', origin='upper',
        ##                  extent=[0,float(m)*(1.0/43.),20,1], aspect=0.1)
        ## plt.colorbar(im2, fraction=0.031, ticks=[0.0, 1.0], pad=0.01)
        ## ax2.set_xlabel("Time [sec]", fontsize=18)
        ## ax2.set_ylabel("Hidden State", fontsize=18)


        ## ax3 = plt.subplot(313)
        # fig.savefig('test.pdf')
        # fig.savefig('test.png')
        plt.grid()
        plt.show()

    def predict(self, X):
        X = np.squeeze(X)
        X_test = X.tolist()

        mu_l = np.zeros(self.nEmissionDim)
        cov_l = np.zeros(self.nEmissionDim**2)

        if self.verbose: print self.F
        final_ts_obj = ghmm.EmissionSequence(self.F, X_test) # is it neccessary?

        try:
            # alpha: X_test length y # latent States at the moment t when state i is ended
            # test_profile_length x number_of_hidden_state
            (alpha, scale) = self.ml.forward(final_ts_obj)
            alpha = np.array(alpha)
        except:
            if self.verbose: print "No alpha is available !!"
            
        f = lambda x: round(x, 12)
        for i in range(len(alpha)):
            alpha[i] = map(f, alpha[i])
        alpha[-1] = map(f, alpha[-1])
        
        n = len(X_test)
        pred_numerator = 0.0

        for j in xrange(self.nState): # N+1
            total = np.sum(self.A[:,j]*alpha[n/self.nEmissionDim-1,:]) #* scaling_factor
            [mus, covars] = self.B[j]

            ## print mu1, mu2, cov11, cov12, cov21, cov22, total
            pred_numerator += total

            for i in xrange(mu_l.size):
                mu_l[i] += mus[i]*total
            for i in xrange(cov_l.size):
                cov_l[i] += covars[i] * (total**2)

        return mu_l, cov_l

    def predict2(self, X, x1, x2, x3, x4):
        X = np.squeeze(X)
        X_test = X.tolist()        
        n = len(X_test)

        mu_l = np.zeros(self.nEmissionDim)
        cov_l = np.zeros(self.nEmissionDim**2)

        final_ts_obj = ghmm.EmissionSequence(self.F, X_test + [x1, x2, x3, x4])

        try:
            (alpha, scale) = self.ml.forward(final_ts_obj)
        except:
            if self.verbose: print "No alpha is available !!"
            sys.exit()

        alpha = np.array(alpha)

        for j in xrange(self.nState):
            
            [[mu1, mu2, mu3, mu4], [cov11, cov12, cov13, cov14, cov21, cov22, cov23, cov24,
                                    cov31, cov32, cov33, cov34, cov41, cov42, cov43, cov44]] = self.B[j]

            mu_l[0] = x1
            mu_l[1] += alpha[n/self.nEmissionDim, j] * (mu2 + cov21/cov11*(x1 - mu1) )
            mu_l[2] += alpha[n/self.nEmissionDim, j] * (mu3 + cov31/cov21*(x2 - mu2)) # TODO Where does this come from?
            mu_l[3] += alpha[n/self.nEmissionDim, j] * (mu4 + cov41/cov31*(x3 - mu3)) # TODO Where does this come from?
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
            if self.verbose: print 'Likelihood error!!!!'
            sys.exit()

        return p

    def likelihoods(self, X1, X2, X3, X4):
        # n, m = np.shape(X1)
        X_test = self.convert_sequence(X1, X2, X3, X4, emission=False)
        # i = m - 1

        final_ts_obj = ghmm.EmissionSequence(self.F, X_test[0].tolist())
        logp = self.ml.loglikelihood(final_ts_obj)
        post = np.array(self.ml.posterior(final_ts_obj))

        n = len(np.squeeze(X1))

        # Find the best posterior distribution
        min_index, min_dist = self.findBestPosteriorDistribution(post[n-1])

        ll_likelihood = logp
        ll_state_idx  = min_index
        ll_likelihood_mu  = self.ll_mu[min_index]
        ll_likelihood_std = self.ll_std[min_index] #self.ll_mu[min_index] + ths_mult*self.ll_std[min_index]

        return ll_likelihood, ll_state_idx, ll_likelihood_mu, ll_likelihood_std

    def allLikelihoods(self, X1, X2, X3, X4):
        # n, m = np.shape(X1)
        X_test = self.convert_sequence(X1, X2, X3, X4, emission=False)
        # i = m - 1

        m = len(np.squeeze(X1))

        ll_likelihood = np.zeros(m)
        ll_state_idx  = np.zeros(m)
        ll_likelihood_mu  = np.zeros(m)
        ll_likelihood_std = np.zeros(m)
        for i in xrange(1, m):
            final_ts_obj = ghmm.EmissionSequence(self.F, X_test[0,:i*self.nEmissionDim].tolist())
            logp = self.ml.loglikelihood(final_ts_obj)
            post = np.array(self.ml.posterior(final_ts_obj))

            # Find the best posterior distribution
            min_index, min_dist = self.findBestPosteriorDistribution(post[i-1])

            ll_likelihood[i] = logp
            ll_state_idx[i]  = min_index
            ll_likelihood_mu[i]  = self.ll_mu[min_index]
            ll_likelihood_std[i] = self.ll_std[min_index] #self.ll_mu[min_index] + ths_mult*self.ll_std[min_index]

        return ll_likelihood, ll_state_idx, ll_likelihood_mu, ll_likelihood_std

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
    def vectors_to_mean_cov(self, vec1, vec2, vec3, vec4, nState):
        index = 0
        m, n = np.shape(vec1)
        #print m,n
        mu_1 = np.zeros(nState)
        mu_2 = np.zeros(nState)
        mu_3 = np.zeros(nState)
        mu_4 = np.zeros(nState)
        cov = np.zeros((nState, self.nEmissionDim, self.nEmissionDim))
        DIVS = n/nState

        while index < nState:
            m_init = index*DIVS
            temp_vec1 = vec1[:, m_init:(m_init+DIVS)]
            temp_vec2 = vec2[:, m_init:(m_init+DIVS)]
            temp_vec3 = vec3[:, m_init:(m_init+DIVS)]
            temp_vec4 = vec4[:, m_init:(m_init+DIVS)]
            temp_vec1 = np.reshape(temp_vec1, (1, DIVS*m))
            temp_vec2 = np.reshape(temp_vec2, (1, DIVS*m))
            temp_vec3 = np.reshape(temp_vec3, (1, DIVS*m))
            temp_vec4 = np.reshape(temp_vec4, (1, DIVS*m))
            mu_1[index] = np.mean(temp_vec1)
            mu_2[index] = np.mean(temp_vec2)
            mu_3[index] = np.mean(temp_vec3)
            mu_4[index] = np.mean(temp_vec4)
            cov[index, :, :] = np.cov(np.concatenate((temp_vec1, temp_vec2, temp_vec3, temp_vec4), axis=0))
            index = index+1

        return mu_1, mu_2, mu_3, mu_4, cov

    @staticmethod
    def init_trans_mat(nState):
        # Reset transition probability matrix
        trans_prob_mat = np.zeros((nState, nState))

        for i in xrange(nState):
            # Exponential function
            # From y = a*e^(-bx)
            ## a = 0.4
            ## b = np.log(0.00001/a)/(-(nState-i))
            ## f = lambda x: a*np.exp(-b*x)

            # Linear function
            # From y = -a*x + b
            b = 0.4
            a = b/float(nState)
            f = lambda x: -a*x+b

            for j in np.array(range(nState-i))+i:
                trans_prob_mat[i, j] = f(j)

            # Gaussian transition probability
            ## z_prob = norm.pdf(float(i),loc=u_mu_list[i],scale=u_sigma_list[i])

            # Normalization
            trans_prob_mat[i,:] /= np.sum(trans_prob_mat[i,:])

        return trans_prob_mat

    def init_plot(self):
        print "Start to print out"

        self.fig = plt.figure(1)
        gs = gridspec.GridSpec(4, 1)
        
        self.ax1 = self.fig.add_subplot(gs[0])        
        self.ax2 = self.fig.add_subplot(gs[1])        
        self.ax3 = self.fig.add_subplot(gs[2])
        self.ax4 = self.fig.add_subplot(gs[3])

    def data_plot(self, X_test1, X_test2, X_test3, X_test4, color='r'):
        self.ax1.plot(np.hstack([X_test1[0]]), color)
        self.ax2.plot(np.hstack([X_test2[0]]), color)
        self.ax3.plot(np.hstack([X_test3[0]]), color)
        self.ax4.plot(np.hstack([X_test4[0]]), color)

    @staticmethod
    def final_plot():
        plt.rc('text', usetex=True)
        plt.show()

    @staticmethod
    def convert_sequence(data1, data2, data3, data4, emission=False):
        # change into array from other types
        if type(data1) is not np.ndarray:
            X1 = copy.copy(np.array(data1))
        else:
            X1 = copy.copy(data1)
        if type(data2) is not np.ndarray:
            X2 = copy.copy(np.array(data2))
        else:
            X2 = copy.copy(data2)
        if type(data3) is not np.ndarray:
            X3 = copy.copy(np.array(data3))
        else:
            X3 = copy.copy(data3)
        if type(data4) is not np.ndarray:
            X4 = copy.copy(np.array(data4))
        else:
            X4 = copy.copy(data4)

        # Change into 2dimensional array
        dim = np.shape(X1)
        if len(dim) == 1:
            X1 = np.reshape(X1, (1, len(X1)))
        dim = np.shape(X2)
        if len(dim) == 1:
            X2 = np.reshape(X2, (1, len(X2)))
        dim = np.shape(X3)
        if len(dim) == 1:
            X3 = np.reshape(X3, (1, len(X3)))
        dim = np.shape(X4)
        if len(dim) == 1:
            X4 = np.reshape(X4, (1, len(X4)))

        n, m = np.shape(X1)

        X = []
        for i in xrange(n):
            Xs = []
                
            if emission:
                for j in xrange(m):
                    Xs.append([X1[i, j], X2[i, j], X3[i, j], X4[i, j]])
                X.append(Xs)
            else:
                for j in xrange(m):
                    Xs.append([X1[i, j], X2[i, j], X3[i, j], X4[i, j]])
                X.append(np.array(Xs).flatten().tolist())

        return np.array(X)
        
    def anomaly_check(self, X1, X2=None, X3=None, X4=None, ths_mult=None):

        if self.nEmissionDim == 1: X_test = np.array([X1])
        else: X_test = self.convert_sequence(X1, X2, X3, X4, emission=False)

        try:
            final_ts_obj = ghmm.EmissionSequence(self.F, X_test[0].tolist())
            logp = self.ml.loglikelihood(final_ts_obj)
        except:
            if self.verbose: print "Too different input profile that cannot be expressed by emission matrix"
            return True, 0.0 # error

        if self.check_method == 'change' or self.check_method == 'globalChange':

            ## if len(X1)<3: 
            ##     if self.verbose: print "Too short profile!"
            ##     return -1, 0.0 #error

            X_test = self.convert_sequence(X1[:-1], X2[:-1], X3[:-1], X4[:-1], emission=False)                
            try:
                final_ts_obj = ghmm.EmissionSequence(self.F, X_test[0].tolist())
                last_logp         = self.ml.loglikelihood(final_ts_obj)
            except:
                print "Too different input profile that cannot be expressed by emission matrix"
                return True, 0.0 # error

            ## print self.l_mean_delta + ths_mult*self.l_std_delta, abs(logp-last_logp)
            if type(ths_mult) == list or type(ths_mult) == np.ndarray or type(ths_mult) == tuple:
                err = (self.l_mean_delta + (-1.0*ths_mult[0])*self.l_std_delta ) - abs(logp-last_logp)
            else:                
                err = (self.l_mean_delta + (-1.0*ths_mult)*self.l_std_delta ) - abs(logp-last_logp)
            ## if err < self.anomaly_offset: return 1.0, 0.0 # anomaly            
            if err < 0.0: return True, 0.0 # anomaly            
            
        if self.check_method == 'global' or self.check_method == 'globalChange':
            if type(ths_mult) == list or type(ths_mult) == np.ndarray or type(ths_mult) == tuple:
                err = logp - (self.l_mu + ths_mult[1]*self.l_std)
            else:
                err = logp - (self.l_mu + ths_mult*self.l_std)

            if err<0.0: return True, err
            else: return False, err
            ## return err < 0.0, err
                
        elif self.check_method == 'progress':
            try:
                post = np.array(self.ml.posterior(final_ts_obj))
            except:
                if self.verbose: print "Unexpected profile!! GHMM cannot handle too low probability. Underflow?"
                return True, 0.0 # anomaly

            if len(X1) == 1:
                n = 1
            else:
                n = len(np.squeeze(X1))

            # Find the best posterior distribution
            min_index, min_dist = self.findBestPosteriorDistribution(post[n-1])

            if (type(ths_mult) == list or type(ths_mult) == np.ndarray or type(ths_mult) == tuple) and len(ths_mult)>1:
                err = logp - (self.ll_mu[min_index] + ths_mult[min_index]*self.ll_std[min_index])
            else:
                err = logp - (self.ll_mu[min_index] + ths_mult*self.ll_std[min_index])

            if err < self.anomaly_offset: return True, err
            else: return False, err
            
        else:
            if err < 0.0: return True, err
            else: return False, err
            

            
    def expLikelihoods(self, X1, X2=None, X3=None, X4=None, ths_mult=None):
        if self.nEmissionDim == 1: X_test = np.array([X1])
        else: X_test = self.convert_sequence(X1, X2, X3, X4, emission=False)

        try:
            final_ts_obj = ghmm.EmissionSequence(self.F, X_test[0].tolist())
            logp = self.ml.loglikelihood(final_ts_obj)
        except:
            print "Too different input profile that cannot be expressed by emission matrix"
            return -1, 0.0 # error

        try:
            post = np.array(self.ml.posterior(final_ts_obj))
        except:
            print "Unexpected profile!! GHMM cannot handle too low probability. Underflow?"
            return 1.0, 0.0 # anomaly

        n = len(np.squeeze(X1))

        # Find the best posterior distribution
        min_index, min_dist = self.findBestPosteriorDistribution(post[n-1])

        # print 'Computing anomaly'
        # print logp
        # print self.ll_mu[min_index]
        # print self.ll_std[min_index]

        # print 'logp:', logp, 'll_mu', self.ll_mu[min_index], 'll_std', self.ll_std[min_index], 'mult_std', ths_mult*self.ll_std[min_index]

        if (type(ths_mult) == list or type(ths_mult) == np.ndarray or type(ths_mult) == tuple) and len(ths_mult)>1:
            ## print min_index, self.ll_mu[min_index], self.ll_std[min_index], ths_mult[min_index], " = ", (self.ll_mu[min_index] + ths_mult[min_index]*self.ll_std[min_index]) 
            return (self.ll_mu[min_index] + ths_mult[min_index]*self.ll_std[min_index])
        else:
            return (self.ll_mu[min_index] + ths_mult*self.ll_std[min_index])


        
    @staticmethod
    def scaling(X, min_c=None, max_c=None, scale=10.0, verbose=False):
        '''
        scale should be over than 10.0(?) to avoid floating number problem in ghmm.
        Return list type
        '''
        ## X_scaled = preprocessing.scale(np.array(X))

        if min_c is None or max_c is None:
            min_c = np.min(X)
            max_c = np.max(X)

        X_scaled = []
        for x in X:
            if verbose is True: print min_c, max_c, " : ", np.min(x), np.max(x)
            X_scaled.append(((x-min_c) / (max_c-min_c) * scale))

        return X_scaled, min_c, max_c

    def likelihood_disp(self, X1, X2, X3, X4, X1_true, X2_true, X3_true, X4_true,
                        Z1, Z2, Z3, Z4, Z1_true, Z2_true, Z3_true, Z4_true, ths_mult, figureSaveName=None):
        ## print np.shape(X1)
        n, m = np.shape(X1)
        n2, m2 = np.shape(Z1)
        if self.verbose: print "Input sequence X1: ", n, m
        if self.verbose: print 'Anomaly: ', self.anomaly_check(X1, X2, X3, X4, ths_mult)

        X_test = self.convert_sequence(X1, X2, X3, X4, emission=False)
        Z_test = self.convert_sequence(Z1, Z2, Z3, Z4, emission=False)

        x = np.arange(0., float(m))
        z = np.arange(0., float(m2))
        ll_likelihood = np.zeros(m)
        ll_state_idx  = np.zeros(m)
        ll_likelihood_mu  = np.zeros(m)
        ll_likelihood_std = np.zeros(m)
        ll_thres_mult = np.zeros(m)
        for i in xrange(1, m):
            final_ts_obj = ghmm.EmissionSequence(self.F, X_test[0,:i*self.nEmissionDim].tolist())
            logp = self.ml.loglikelihood(final_ts_obj)
            post = np.array(self.ml.posterior(final_ts_obj))

            # Find the best posterior distribution
            min_index, min_dist = self.findBestPosteriorDistribution(post[i-1])

            ll_likelihood[i] = logp
            ll_state_idx[i]  = min_index
            ll_likelihood_mu[i]  = self.ll_mu[min_index]
            ll_likelihood_std[i] = self.ll_std[min_index] #self.ll_mu[min_index] + ths_mult*self.ll_std[min_index]
            ll_thres_mult[i] = ths_mult

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


        # y1 = (X1_true[0]/scale1[2])*(scale1[1]-scale1[0])+scale1[0]
        # y2 = (X2_true[0]/scale2[2])*(scale2[1]-scale2[0])+scale2[0]
        # y3 = (X3_true[0]/scale3[2])*(scale3[1]-scale3[0])+scale3[0]
        # y4 = (X4_true[0]/scale4[2])*(scale4[1]-scale4[0])+scale4[0]
        y1 = X1_true[0]
        y2 = X2_true[0]
        y3 = X3_true[0]
        y4 = X4_true[0]

        zy1 = np.mean(Z1_true, axis=0)
        zy2 = np.mean(Z2_true, axis=0)
        zy3 = np.mean(Z3_true, axis=0)
        zy4 = np.mean(Z4_true, axis=0)

        ## matplotlib.rcParams['figure.figsize'] = 8,7
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42

        fig = plt.figure()
        plt.rc('text', usetex=True)

        ax1 = plt.subplot(512)
        # print np.shape(x), np.shape(y1)
        ax1.plot(x*(1./10.), y1)
        # print np.shape(z), np.shape(zy1)
        ax1.plot(z*(1./10.), zy1, 'r')
        y_min = np.amin(y1)
        y_max = np.amax(y1)
        collection = collections.BrokenBarHCollection.span_where(np.array(block_x_interp)*(1./10.),
                                                                 ymin=0, ymax=y_max+0.5,
                                                                 where=np.array(block_flag_interp)>0,
                                                                 facecolor='green',
                                                                 edgecolor='none', alpha=0.3)
        ax1.add_collection(collection)
        ax1.set_ylabel("Force (N)", fontsize=16)
        ax1.set_xlim([0, x[-1]*(1./10.)])
        ## ax1.set_ylim([0, np.amax(y1)*1.1])
        ax1.set_ylim([y_min - 0.25, y_max + 0.5])

        # -----

        ax2 = plt.subplot(511)
        ax2.plot(x*(1./10.), y2)
        ax2.plot(z*(1./10.), zy2, 'r')
        y_max = np.amax(y2)
        collection = collections.BrokenBarHCollection.span_where(np.array(block_x_interp)*(1./10.),
                                                                 ymin=0, ymax=y_max + 0.25,
                                                                 where=np.array(block_flag_interp)>0,
                                                                 facecolor='green',
                                                                 edgecolor='none', alpha=0.3)
        ax2.add_collection(collection)

        # Text for progress
        for i in xrange(len(block_state)):
            if i%2 is 0:
                if i<10:
                    ax2.text((text_x[i])*(1./10.), y_max+0.15, str(block_state[i]+1))
                else:
                    ax2.text((text_x[i]-1.0)*(1./10.), y_max+0.15, str(block_state[i]+1))
            else:
                if i<10:
                    ax2.text((text_x[i])*(1./10.), y_max+0.06, str(block_state[i]+1))
                else:
                    ax2.text((text_x[i]-1.0)*(1./10.), y_max+0.06, str(block_state[i]+1))

        ax2.set_ylabel("Distance (m)", fontsize=16)
        ax2.set_xlim([0, x[-1]*(1./10.)])
        ax2.set_ylim([0, y_max + 0.25])

        # -----

        ax4 = plt.subplot(514)
        ax4.plot(x*(1./10.), y3)
        ax4.plot(z*(1./10.), zy3, 'r')
        y_max = np.amax(y3)
        collection = collections.BrokenBarHCollection.span_where(np.array(block_x_interp)*(1./10.),
                                                                 ymin=0, ymax=y_max + 0.1,
                                                                 where=np.array(block_flag_interp)>0,
                                                                 facecolor='green',
                                                                 edgecolor='none', alpha=0.3)
        ax4.add_collection(collection)
        ax4.set_ylabel("Angle (rad)", fontsize=16)
        ax4.set_xlim([0, x[-1]*(1./10.)])
        ax4.set_ylim([0, y_max + 0.1])

        # -----

        ax5 = plt.subplot(513)
        ax5.plot(x*(1./10.), y4)
        ax5.plot(z*(1./10.), zy4, 'r')
        y_min = np.amin(y4)
        y_max = np.amax(y4)
        collection = collections.BrokenBarHCollection.span_where(np.array(block_x_interp)*(1./10.),
                                                                 ymin=y_min - y_min/15.0, ymax=y_max + y_min/15.0,
                                                                 where=np.array(block_flag_interp)>0,
                                                                 facecolor='green',
                                                                 edgecolor='none', alpha=0.3)
        ax5.add_collection(collection)
        ax5.set_ylabel("Audio (dec)", fontsize=16)
        ax5.set_xlim([0, x[-1]*(1./10.)])
        ax5.set_ylim([y_min - y_min/15.0, y_max + y_min/15.0])

        # -----

        ax3 = plt.subplot(515)
        ax3.plot(x*(1./10.), ll_likelihood, 'b', label='Actual from \n test data')
        ax3.plot(x*(1./10.), ll_likelihood_mu, 'r', label='Expected from \n trained model')
        ax3.plot(x*(1./10.), ll_likelihood_mu + ll_thres_mult*ll_likelihood_std, 'r--', label='Threshold')
        ## ax3.set_ylabel(r'$log P({\mathbf{X}} | {\mathbf{\theta}})$',fontsize=18)
        ax3.set_ylabel('Log-likelihood',fontsize=16)
        ax3.set_xlim([0, x[-1]*(1./10.)])

        ## ax3.legend(loc='upper left', fancybox=True, shadow=True, ncol=3, prop={'size':14})
        lgd = ax3.legend(loc='upper center', fancybox=True, shadow=True, ncol=3, bbox_to_anchor=(0.5, -0.5), prop={'size':14})
        ax3.set_xlabel('Time (sec)', fontsize=16)

        plt.subplots_adjust(bottom=0.15)

        if figureSaveName is None:
            plt.show()
        else:
            # fig.savefig('test.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
            fig.savefig(figureSaveName, bbox_extra_artists=(lgd,), bbox_inches='tight')

    def learn_likelihoods_progress_par(self, i, n, m, A, B, pi, X_train, g_mu, g_sig):
        l_likelihood_mean = 0.0
        l_likelihood_mean2 = 0.0
        l_statePosterior = np.zeros(self.nState)

        for j in xrange(n):
            results = Parallel(n_jobs=-1)(delayed(computeLikelihood)(self.F, k, X_train[j][:k*self.nEmissionDim], g_mu, g_sig, self.nEmissionDim, A, B, pi) for k in xrange(1, m))

            g_post = np.sum([r[0] for r in results], axis=0)
            g_lhood, g_lhood2, prop_sum = np.sum([r[1:] for r in results], axis=0)

            l_statePosterior += g_post / prop_sum / float(n)
            l_likelihood_mean += g_lhood / prop_sum / float(n)
            l_likelihood_mean2 += g_lhood2 / prop_sum / float(n)

        return i, l_statePosterior, l_likelihood_mean, np.sqrt(l_likelihood_mean2 - l_likelihood_mean**2)


    def state_clustering(self, X1, X2, X3, X4):
        n,m = np.shape(X1)

        print n,m
        x   = np.arange(0., float(m))*(1./43.)
        state_mat  = np.zeros((self.nState, m*n))
        likelihood_mat = np.zeros((1, m*n))

        count = 0           
        for i in xrange(n):

            for j in xrange(1,m):            

                x_test1 = X1[i:i+1,:j]
                x_test2 = X2[i:i+1,:j]            
                x_test3 = X3[i:i+1,:j]            
                x_test4 = X4[i:i+1,:j]            
                X_test = self.convert_sequence(x_test1, x_test2, x_test3, x_test4, emission=False)

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
    if nEmissionDim >= 2:
        ml = ghmm.HMMFromMatrices(F, ghmm.MultivariateGaussianDistribution(F), A, B, pi)
    else:
        ml = ghmm.HMMFromMatrices(F, ghmm.GaussianDistribution(F), A, B, pi)

    l_likelihood_mean = 0.0
    l_likelihood_mean2 = 0.0
    l_statePosterior = np.zeros(nState)

    for j in xrange(n):    

        g_post = np.zeros(nState)
        g_lhood = 0.0
        g_lhood2 = 0.0
        prop_sum = 0.0

        for k in xrange(1, m):
            final_ts_obj = ghmm.EmissionSequence(F, X_train[j][:k*nEmissionDim])
            logp = ml.loglikelihoods(final_ts_obj)[0]
            # print 'Log likelihood:', logp
            post = np.array(ml.posterior(final_ts_obj))

            k_prop = norm(loc=g_mu, scale=g_sig).pdf(k)
            g_post += post[k-1] * k_prop
            g_lhood += logp * k_prop
            g_lhood2 += logp * logp * k_prop

            prop_sum  += k_prop

        l_statePosterior += g_post / prop_sum / float(n)
        l_likelihood_mean += g_lhood / prop_sum / float(n)
        l_likelihood_mean2 += g_lhood2 / prop_sum / float(n)

    return i, l_statePosterior, l_likelihood_mean, np.sqrt(l_likelihood_mean2 - l_likelihood_mean**2)
    
def computeLikelihood(F, k, data, g_mu, g_sig, nEmissionDim, A, B, pi):
    if nEmissionDim >= 2:
        hmm_ml = ghmm.HMMFromMatrices(F, ghmm.MultivariateGaussianDistribution(F), A, B, pi)
    else:
        hmm_ml = ghmm.HMMFromMatrices(F, ghmm.GaussianDistribution(F), A, B, pi)

    final_ts_obj = ghmm.EmissionSequence(F, data)
    logp = hmm_ml.loglikelihoods(final_ts_obj)[0]
    post = np.array(hmm_ml.posterior(final_ts_obj))

    k_prop = norm(loc=g_mu, scale=g_sig).pdf(k)
    g_post = post[k-1] * k_prop
    g_lhood = logp * k_prop
    g_lhood2 = logp * logp * k_prop
    prop_sum = k_prop

    # print np.shape(g_post), np.shape(g_lhood), np.shape(g_lhood2), np.shape(prop_sum)

    return g_post, g_lhood, g_lhood2, prop_sum

