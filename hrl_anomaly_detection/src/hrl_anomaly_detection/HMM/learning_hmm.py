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
    def fit(self, X_train, verbose=False):
        
        # Transition probability matrix (Initial transition probability, TODO?)
        A = self.init_trans_mat(self.nState).tolist()
                                
        # We should think about multivariate Gaussian pdf.        
        self.mu, self.sigma = self.vectors_to_mean_vars(X_train, optimize=False)

        # Emission probability matrix
        B = np.hstack([self.mu, self.sigma]).tolist() # Must be [i,:] = [mu, sigma]
        
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

        
        ## self.mean_path_plot(mu[:,0], sigma[:,0])        
        ## print "Completed to fitting", np.array(final_seq).shape

        # Future observation range
        self.max_obsrv = X_train.max()
        self.obsrv_range = np.arange(0.0, self.max_obsrv*1.2, self.fObsrvResol)


        if verbose:
            print "A: ", A.shape
            n,m = A.shape
            for i in xrange(n):
                a = None
                for j in xrange(m):
                    if a==None:
                        a = "%0.3f" % A[i,j]
                    else:
                        a += "  "
                        a += "%0.3f" % A[i,j]
                print a
            print "----------------------------------------------"
            for i in xrange(n):
                a = None
                for j in xrange(m):
                    if a==None:
                        a = "%0.3f" % self.ml.getTransition(i,j)
                    else:
                        a += "  "
                        a += "%0.3f" % self.ml.getTransition(i,j)
                print a
                
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

        if optimize==False:
            
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
            
            index = 0
            m_init = 0
            while (index < self.nState):
                temp_vec = vecs[:,(m_init):(m_init + int(self.step_size_list[index]))] 
                m_init = m_init + int(self.step_size_list[index])

                mu[index] = np.mean(temp_vec)
                sigma[index] = np.std(temp_vec)
                index = index+1

        else:
            from scipy import optimize

            # Initial 
            x0 = [1] * self.nState
            while sum(x0)!=self.nMaxStep:
                idx = int(random.gauss(float(self.nState)/2.0,float(self.nState)/2.0/2.0))
                if idx < 0 or idx >= self.nState: 
                    continue
                else:
                    x0[idx] += 1
                
            bnds=[]
            for i in xrange(self.nState):
                bnds.append([0,self.nMaxStep])
                
            ## res = optimize.minimize(self.mean_vars_score,x0,args=(vecs), method='SLSQP', bounds=bnds, constraints=({'type':'eq','fun':self.mean_vars_constraints}), options={'maxiter': 50})
            res = optimize.minimize(self.mean_vars_score,x0, method='SLSQP', bounds=bnds, constraints=({'type':'eq','fun':self.mean_vars_constraint1}, {'type':'eq','fun':self.mean_vars_constraint2}), options={'maxiter': 50})
            self.step_size_list = res['x'] 
            print "Best step_size_list: "
            string = None
            for x in self.step_size_list:
                if string == None:
                    string = str(x)+", " 
                else:
                    string += str(x)+", "
            print string
                
            
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
    #
    ## def mean_vars_score(self, x0, *args):
    def mean_vars_score(self, x0):

        vecs = self.aXData #args[0]        
        ## mu    = np.zeros((self.nState,1))
        sigma = np.zeros((self.nState,1))

        for i, nDivs in enumerate(x0):
            m_init = i*nDivs
            try:
                temp_vec = vecs[:,int(m_init):int(m_init+nDivs)]
            except:
                return 0.0

            ## mu[i] = np.mean(temp_vec)
            sigma[i] = np.std(temp_vec)
        return np.std(sigma)

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

                
            # profile
            final_ts_obj = ghmm.EmissionSequence(self.F,X_test+[X_pred]) # is it neccessary?

            prob[i] = self.ml.loglikelihood(final_ts_obj)
            ## continue

            ## # Past profile
            ## final_ts_obj = ghmm.EmissionSequence(self.F,X_test) # is it neccessary?
            
            ## ## print "\nForward"
            ## ## logp1 = self.ml.loglikelihood(final_ts_obj)
            ## ## print "logp = " + str(logp1) + "\n"

            ## # alpha: X_test length y #latent States at the moment t when state i is ended
            ## #        test_profile_length x number_of_hidden_state
            ## (alpha,scale) = self.ml.forward(final_ts_obj)
            ## ## print "alpha: ", np.array(alpha).shape,"\n" + str(alpha) + "\n"
            ## ## print "scale = " + str(scale) + "\n"
            ## ## print np.array(X_test).shape, np.array(alpha).shape

            ## # beta
            ## beta = self.ml.backward(final_ts_obj,scale)
            ## ## print "beta", np.array(beta).shape, " = \n " + str(beta) + "\n"

            ## pred_numerator = 0.0
            ## pred_denominator = 0.0
            ## for j in xrange(self.nState): # N+1

            ##     total = 0.0        
            ##     for k in xrange(self.nState): # N                  
            ##         total += self.ml.getTransition(k,j) * alpha[-1][k]

            ##     (mu, sigma) = self.ml.getEmission(j)

            ##     pred_numerator += norm(loc=mu,scale=sigma).pdf(X_pred) * total
            ##     pred_denominator += alpha[-1][j]*beta[-1][j]

            ## prob[i] = pred_numerator / pred_denominator

        return prob


    #----------------------------------------------------------------------        
    #
    def one_step_predict(self, X):
        # Input
        # @ X_test: 1 x known steps #samples x known steps
        # Output
        # @ X_pred: 1
        # @ X_pred_prob: samples x 1

        # Initialization            
        X_pred_prob = np.zeros((len(self.obsrv_range)))
        if type(X) == np.ndarray:
            X = X.tolist()

        # Get all probability
        for i, obsrv in enumerate(self.obsrv_range):           
            if abs(X[-1]-obsrv) > 4.0: continue
            X_pred_prob[i] += np.exp(self.predict([X+[obsrv]]))        #???? normalized??     

        # Select observation with maximum probability
        idx_list = [k[0] for k in sorted(enumerate(X_pred_prob), key=lambda x:x[1], reverse=True)]
            
        return self.obsrv_range[idx_list[0]], X_pred_prob/np.sum(X_pred_prob)

        
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
                        
        return X_pred, X_pred_prob


    #----------------------------------------------------------------------        
    # Compute the estimated probability (0.0~1.0)
    def multi_step_approximated_predict(self, X_test, verbose=False):



        return X_pred, X_pred_prob
        
        
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
        # Neccessary package
        from sklearn.metrics import r2_score

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

        print "---------------------------------------------"
        print "Total Score"
        print total_score
        print "---------------------------------------------"
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
                
                alpha   = 1.0 - np.exp(-prob*50.0)
                if alpha > 1.0: alpha = 1.0
                self.ax.plot(x_array[i:i+2], y_array, 'r-', alpha=alpha, linewidth=1.0)    

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
    p.add_option('--optimize_mv', '--mv', action='store_true', dest='bOptMeanVar',
                 default=False, help='Optimize mean and vars for B matrix')
    p.add_option('--verbose', '--v', action='store_true', dest='bVerbose',
                 default=False, help='Print out everything')
    opt, args = p.parse_args()

    ## Init variables    
    data_path = os.getcwd()
    nState    = 34
    nMaxStep     = 36 # total step of data. It should be automatically assigned...
    pkl_file  = "door_opening_data.pkl"    
    nFutureStep = 6
    ## data_column_idx = 1
    fObsrvResol = 0.1
    nCurrentStep = 28

    if nState == 28:
        step_size_list = [1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 2, 1, 2, 1, 3, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1]
            #step_size_list = [1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 3, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1] 
    ## elif nState == 30:
    ##     step_size_list = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    else:
        step_size_list = None
        
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
    ## print data_vecs.shape, np.array(data_mech).shape, np.array(data_chunks).shape
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
    lh = learning_hmm(data_path=data_path, aXData=data_vecs[0], nState=nState, nMaxStep=nMaxStep, nFutureStep=nFutureStep, fObsrvResol=fObsrvResol, nCurrentStep=nCurrentStep, step_size_list=step_size_list)    

    if opt.bCrossVal:
        print "Cross Validation"

        import socket, time
        host_name = socket.gethostname()
        t=time.gmtime()                
        save_file = os.path.join('/home/dpark/hrl_file_server/dpark_data/anomaly/RSS2015/door_tune',host_name+'_'+str(t[0])+str(t[1])+str(t[2])+'_'+str(t[3])+str(t[4])+'.pkl')

        #tuned_parameters = [{'nState': [20,25,30,35], 'nFutureStep': [1], 'fObsrvResol': [0.05,0.1,0.15,0.2,0.25], 'nCurrentStep': [5,10,15,20,25]}]
        tuned_parameters = [{'nState': [10,11,12,13,14], 'nFutureStep': [1], 'fObsrvResol': [0.05,0.1,0.15,0.2,0.2]}]
        tuned_parameters = [{'nState': [15,16,17,18,19], 'nFutureStep': [1], 'fObsrvResol': [0.05,0.1,0.15,0.2,0.2]}]
        tuned_parameters = [{'nState': [20,21,22,23,24], 'nFutureStep': [1], 'fObsrvResol': [0.05,0.1,0.15,0.2,0.2]}]
        ## tuned_parameters = [{'nState': [25,26,27,28,29], 'nFutureStep': [1], 'fObsrvResol': [0.05,0.1,0.15,0.2,0.2]}]
        ## tuned_parameters = [{'nState': [30,31,32,33,34], 'nFutureStep': [1], 'fObsrvResol': [0.05,0.1,0.15,0.2,0.2]}]
        ## tuned_parameters = [{'nState': [35,36], 'nFutureStep': [1], 'fObsrvResol': [0.05,0.1,0.15,0.2,0.2]}]

        ## tuned_parameters = [{'nState': [20,30], 'nFutureStep': [1], 'fObsrvResol': [0.1]}]

        
        ## step_size_list_set = []
        ## for i in xrange(10):
        ##     step_size_list = [1] * lh.nState
        ##     while sum(step_size_list)!=lh.nMaxStep:
        ##         idx = int(random.gauss(float(lh.nState)/2.0,float(lh.nState)/2.0/2.0))
        ##         if idx < 0 or idx >= lh.nState: 
        ##             continue
        ##         else:
        ##             step_size_list[idx] += 1                
        ##     step_size_list_set.append(step_size_list)                    
        
        ## tuned_parameters = [{'nState': [28], 'nFutureStep': [1], 'fObsrvResol': [0.1], 'nCurrentStep': [5,10,15,20,25], 'step_size_list': step_size_list_set}]
        
        lh.param_estimation(tuned_parameters, 20, save_file=save_file)

    elif opt.bOptMeanVar:
        print "Optimize B matrix"
        lh.vectors_to_mean_vars(lh.aXData, optimize=True)
        
    else:
        lh.fit(lh.aXData, verbose=opt.bVerbose)    
        ## lh.path_plot(data_vecs[0], data_vecs[0,:,3])

        ######################################################    
        # Test data
        ## h_config, h_ftan = mad.get_a_blocked_detection()
        ## print np.array(h_config)*180.0/3.14
        ## print len(h_ftan)

        for i in xrange(1,22,2):
            
            x_test      = data_vecs[0][i,:nCurrentStep].tolist()
            x_test_next = data_vecs[0][i,nCurrentStep:nCurrentStep+lh.nFutureStep].tolist()
            x_test_all  = data_vecs[0][i,:].tolist()
            ## x_test = h_ftan[:15]
            ## x_test_next = h_ftan[15:15+lh.nFutureStep]

            x_pred, x_pred_prob = lh.multi_step_predict(x_test, verbose=opt.bVerbose)
            lh.predictive_path_plot(np.array(x_test), np.array(x_pred), x_pred_prob, np.array(x_test_next))
            lh.final_plot()


    ## print lh.mean_path_plot(lh.mu, lh.sigma)
        
    ## print x_test
    ## print x_test[-4:]

    ## fig = plt.figure(1)
    ## ax = fig.add_subplot(111)

    ## ax.plot(obsrv_range, future_prob)
    ## plt.show()




















    
