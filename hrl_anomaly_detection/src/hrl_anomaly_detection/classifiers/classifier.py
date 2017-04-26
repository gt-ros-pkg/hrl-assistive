#!/usr/bin/env python
#
# Copyright (c) 2014, Georgia Tech Research Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the Georgia Tech Research Corporation nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY GEORGIA TECH RESEARCH CORPORATION ''AS IS'' AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL GEORGIA TECH BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

#  \author Daehyung Park (Healthcare Robotics Lab, Georgia Tech.)

# system
import os, sys, copy, time

# visualization
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec

# util
import numpy as np
import scipy
import hrl_lib.util as ut
import random, copy

from scipy.stats import norm, entropy
from joblib import Parallel, delayed
from hrl_anomaly_detection.classifiers.clf_base import clf_base
from sklearn import metrics
from sklearn.externals import joblib

from hrl_anomaly_detection import data_manager as dm
from hrl_anomaly_detection import util as util
from hrl_anomaly_detection.classifiers import classifier_util as cutil

from sklearn import preprocessing
import warnings


class classifier(clf_base):
    def __init__(self, method='svm', nPosteriors=10, nLength=None, startIdx=4, parallel=False,\
                 ths_mult=-1.0, class_weight=1.0, verbose=False, **kwargs):
        '''
        class_weight : positive class weight for svm
        nLength : only for progress-based classifier
        ths_mult: only for progress-based classifier
        '''
        warnings.simplefilter("always", DeprecationWarning)
        
        self.method = method
        self.nPosteriors = nPosteriors
        self.dt     = None
        self.nLength = nLength
        self.startIdx = startIdx
        self.parallel = parallel
        self.verbose = verbose
                
        # constants to adjust thresholds
        self.class_weight = class_weight
        self.ths_mult = ths_mult

        if self.method.find('svm')>=0:
            sys.path.insert(0, '/usr/lib/pymodules/python2.7')
            import svmutil as svm
            self.svm_type                = kwargs.get('svm_type',0)
            self.kernel_type             = kwargs.get('kernel_type',2)
            self.degree                  = kwargs.get('degree',3) 
            self.gamma                   = kwargs.get('gamma',0.3)
            self.cost                    = kwargs.get('cost',4.)
            self.coef0                   = kwargs.get('coef0',0)
            self.w_negative              = kwargs.get('w_negative',7.)
            self.bpsvm_w_negative        = kwargs.get('bpsvm_w_negative',7.)
            self.bpsvm_cost              = kwargs.get('bpsvm_cost',4.)
            self.bpsvm_gamma             = kwargs.get('bpsvm_gamma',0.3)                        
            self.hmmsvm_diag_nu          = kwargs.get('hmmsvm_diag_nu',0.5)
            self.hmmsvm_diag_w_negative  = kwargs.get('hmmsvm_diag_w_negative',7.)
            self.hmmsvm_diag_cost        = kwargs.get('hmmsvm_diag_cost',4.)
            self.hmmsvm_diag_gamma       = kwargs.get('hmmsvm_diag_gamma',0.3)
            self.hmmosvm_nu              = kwargs.get('hmmosvm_nu',0.00316)
            self.osvm_nu                 = kwargs.get('osvm_nu',0.00316)
            self.nu                      = kwargs.get('nu',0.5)
            self.progress_svm_w_negative = kwargs.get('progress_svm_w_negative',7.0)
            self.progress_svm_cost       = kwargs.get('progress_svm_cost',4.)
            self.progress_svm_gamma      = kwargs.get('progress_svm_gamma',0.3)
        elif self.method == 'progress_state':
            if nLength is None:
                print "Need to input nLength or automatically set to 200"
                self.nLength = 200
            else:
                ## print "Set data length to ", nLength
                self.nLength = nLength
            self.std_coff    = kwargs.get('std_coff',1.0)
            self.logp_offset = kwargs.get('logp_offset',0.0)
            self.ll_mu  = np.zeros(nPosteriors)
            self.ll_std = np.zeros(nPosteriors)
            self.l_statePosterior = None
        elif self.method == 'fixed':
            self.mu  = 0.0
            self.std = 0.0
        elif self.method == 'change':
            self.nLength   = nLength
            self.mu  = 0.0
            self.std = 0.0
        elif self.method == 'sgd':
            self.sgd_w_negative = kwargs.get('sgd_w_negative',1.0)
            self.sgd_gamma      = kwargs.get('sgd_gamma',2.0)
            self.sgd_n_iter     = kwargs.get('sgd_n_iter',10) 
            ## self.cost         = cost
        elif self.method == 'hmmsvr':
            self.svm_type    = kwargs.get('svm_type',0)
            self.kernel_type = kwargs.get('kernel_type',2)
            self.degree      = kwargs.get('degree',3)
            self.gamma       = kwargs.get('gamma',0.3)
            self.cost        = kwargs.get('cost',4.)
            self.coef0       = kwargs.get('coef0',0.)
            self.nu          = kwargs.get('nu',0.5)
                        
        clf_base.__init__(self)



    def fit(self, X, y, ll_idx=None, warm_start=False):
        '''
        ll_idx is the index list of each sample in a sequence.
        '''

        if self.method.find('svm')>=0 :
            sys.path.insert(0, '/usr/lib/pymodules/python2.7')
            import svmutil as svm

            if type(X) is not list: X=X.tolist()
            if type(y) is not list: y=y.tolist()
            commands = '-q -s '+str(self.svm_type)+' -t '+str(self.kernel_type)+' -d '+str(self.degree)\
              +' -w1 '+str(self.class_weight) +' -r '+str(self.coef0)

            if self.method == 'osvm':
                commands = commands+' -n '+str(self.osvm_nu)+' -g '+str(self.gamma)\
                  +' -w-1 '+str(self.w_negative)+' -c '+str(self.cost)
            elif self.method == 'hmmosvm':
                commands = commands+' -n '+str(self.hmmosvm_nu)+' -g '+str(self.gamma)\
                  +' -c '+str(self.cost)
            elif self.method == 'hmmsvm_diag':
                commands = commands+' -n '+str(self.hmmsvm_diag_nu)+' -g '+str(self.hmmsvm_diag_gamma)\
                  +' -w-1 '+str(self.hmmsvm_diag_w_negative)+' -c '+str(self.hmmsvm_diag_cost)
            elif self.method == 'bpsvm':
                commands = commands+' -n '+str(self.nu)+' -g '+str(self.bpsvm_gamma)\
                  +' -w-1 '+str(self.bpsvm_w_negative)+' -c '+str(self.bpsvm_cost)
            elif self.method == 'progress_osvm':
                commands = commands+' -n '+str(self.hmmosvm_nu)+' -g '+str(self.gamma)\
                  +' -w-1 '+str(self.w_negative)+' -c '+str(self.cost)
            elif self.method == 'progress_svm':
                commands = commands+' -n '+str(self.nu)+' -g '+str(self.progress_svm_gamma)\
                  +' -w-1 '+str(self.progress_svm_w_negative)+' -c '+str(self.progress_svm_cost)
            elif self.method == 'svm_fixed':
                commands = commands+' -n '+str(self.nu)+' -g '+str(self.gamma)\
                  +' -w-1 '+str(self.w_negative)+' -c '+str(self.cost)
            else:
                commands = commands+' -n '+str(self.nu)+' -g '+str(self.gamma)\
                  +' -w-1 '+str(self.w_negative)+' -c '+str(self.cost)

            try: self.dt = svm.svm_train(y, X, commands )
            except:
                print "svm training failure"
                print np.shape(y), np.shape(X)
                print commands                
                return False


            if self.method == 'svm_fixed':
                if type(X) == list: X = np.array(X)
                ll_logp = X[:,0:1]
                self.mu  = np.mean(ll_logp)
                self.std = np.std(ll_logp)
                
            return True

        elif self.method.find('svr')>=0:
            sys.path.insert(0, '/usr/lib/pymodules/python2.7')
            import svmutil as svm
            commands = '-q -s 4 -t 0'+' -n '+str(self.nu)+' -g 0.04' \
              +' -c '+str(self.cost) + ' -d 3' 
              #+str(self.gamma)\
            if type(X) == list: X = np.array(X)
            
            # extract only negatives
            ll_logp = [ [X[i,0]] for i in xrange(len(X)) if y[i]<0 ]
            ll_post = [ X[i,-self.nPosteriors:] for i in xrange(len(X)) if y[i]<0 ]


            try: self.dt = svm.svm_train(ll_logp, ll_post, commands )
            except:
                print self.dt
                print "svm training failure", np.shape(ll_logp), np.shape(ll_post)
                print commands                
                return False
            return True
        
        elif self.method == 'progress_state':
            '''
            state-based clustering using knn
            '''
            if type(X) == list: X = np.array(X)

            # extract only negatives
            ll_logp = [ X[i,0] for i in xrange(len(X)) if y[i]<0 ]
            ll_post = [ X[i,-self.nPosteriors:] for i in xrange(len(X)) if y[i]<0 ]
            
            self.progress_neighbors = 10
            
            from sklearn.neighbors import NearestNeighbors
            self.dt = NearestNeighbors(n_neighbors=self.progress_neighbors, metric=cutil.symmetric_entropy)
            self.dt.fit(ll_post)
            
            self.ll_logp = np.array(ll_logp)
            
            return True

        elif self.method == 'fixed':
            if type(X) == list: X = np.array(X)
            ll_logp = X[:,0:1]
            self.mu  = np.mean(ll_logp)
            self.std = np.std(ll_logp)
            return True

        elif self.method == 'change':
            if type(X) == list: X = np.array(X)

            if len(np.shape(y))>1:
                l_idx = [ i for i in range(len(X)) if y[i][0]<0 ]
            else:
                l_idx = [ i for i in range(len(X)) if y[i]<0 ]

            X_logp    = X[l_idx,:,0:1]
            X_logp_d = X_logp[:,1:,0]-X_logp[:,:-1,0]

            self.mu  = np.mean(X_logp_d)
            self.std = np.std(X_logp_d)

            return True
                
        elif self.method == 'sgd':

            max_components = 1000 #196
            if len(X) < max_components:
                n_components =len(X)
            else:
                n_components = max_components
                
            ## from sklearn.kernel_approximation import RBFSampler
            ## self.rbf_feature = RBFSampler(gamma=self.gamma, n_components=1000, random_state=1)
            from sklearn.kernel_approximation import Nystroem
            self.rbf_feature = Nystroem(gamma=self.sgd_gamma, n_components=n_components, random_state=1)
                
            from sklearn.linear_model import SGDClassifier
            # get time-based clustering center? Not yet implemented
            X_features       = self.rbf_feature.fit_transform(X)
            if self.verbose: print "sgd classifier: ", np.shape(X), np.shape(X_features)
            # fitting
            print "Class weight: ", self.class_weight, self.sgd_w_negative
            d = {+1: self.class_weight, -1: self.sgd_w_negative}
            if warm_start and self.dt is not None:
                self.dt.set_params(class_weight=d)
                self.dt.set_params(warm_start=True)
            else:
                self.dt = SGDClassifier(verbose=0,class_weight=d,n_iter=self.sgd_n_iter, #learning_rate='constant',\
                                        eta0=1e-2, shuffle=True, average=True, fit_intercept=True)
            self.dt.fit(X_features, y)
            
        else:
            print self.method, " is not supported in this classifier.py"
            sys.exit()


    def partial_fit(self, X, y=None, X_idx=None, shuffle=False, **kwargs):
        '''
        X: samples x hmm-feature vec
        y: sample
        '''

        if shuffle is True:
            idx_list = range(len(X))
            random.shuffle(idx_list)
            X = [X[ii] for ii in idx_list]
            y = [y[ii] for ii in idx_list]
            if X_idx is not None: X_idx = [X_idx[ii] for ii in idx_list]
            if sample_weight is not None:
                sample_weight = [sample_weight[ii] for ii in idx_list]

        if self.method == 'sgd':
            ## if sample_weight is None: sample_weight = [self.class_weight]*len(X)
            d = {+1: self.class_weight, -1: self.sgd_w_negative}
            self.dt.set_params(class_weight=d)
            
            X_features = self.rbf_feature.transform(X)
            for i in xrange(kwargs['n_iter']):
                self.dt.partial_fit(X_features,y,
                                    classes=kwargs['classes'],
                                    sample_weight=kwargs['sample_weight'])
        else:
            print "Not available method, ", self.method
            sys.exit()


    def predict(self, X, y=None, debug=False):
        '''
        X is single sample
        return predicted values (not necessarily binaries)
        '''

        if self.method.find('svm')>=0:
            
            sys.path.insert(0, '/usr/lib/pymodules/python2.7')
            import svmutil as svm

            if self.verbose:
                print svm.__file__

            if type(X) is not list: X=X.tolist()
            if y is not None:
                p_labels, _, p_vals = svm.svm_predict(y, X, self.dt)
            else:
                p_labels, _, p_vals = svm.svm_predict([0]*len(X), X, self.dt)

            if self.method.find('fixed')>=0:
                if len(np.shape(X))==1: X = [X]
                self.ths_mult = -3

                for i in xrange(len(X)):
                    logp = X[i][0]
                    err = self.mu + self.ths_mult * self.std - logp
                    if p_labels[i] > 0: continue
                    else:
                        if err>0: p_labels[i] = 1.0
            return p_labels
        
        elif self.method == 'progress_state':
            if len(np.shape(X))==1: X = [X]

            l_err = []
            for i in xrange(len(X)):
                logp = X[i][0]
                post = X[i][-self.nPosteriors:]

                _, l_idx = self.dt.kneighbors(post)

                l_logp= self.ll_logp[l_idx[0]]
                err = np.mean(l_logp) + self.ths_mult*np.std(l_logp) - logp - self.logp_offset
                l_err.append(err)

            return l_err            

        elif self.method == 'fixed':
            if len(np.shape(X))==1: X = [X]
                
            l_err = []
            for i in xrange(len(X)):
                logp = X[i][0]
                err = self.mu + self.ths_mult * self.std - logp
                l_err.append(err)
            return l_err

        elif self.method == 'change':
            if len(np.shape(X))==1: X = [X]
                
            l_err = []
            for i in xrange(len(X)):
                if i==0: logp_d = 0.0
                else:    logp_d = X[i][0]-X[i-1][0]
                err = self.mu + self.ths_mult * self.std - logp_d
                l_err.append(err)
            return l_err

        elif self.method == 'sgd':
            X_features = self.rbf_feature.transform(X)
            return self.dt.predict(X_features)

        elif self.method == 'hmmsvr':
            
            sys.path.insert(0, '/usr/lib/pymodules/python2.7')
            import svmutil as svm

            if len(np.shape(X))==1: X = [X]
            if type(X) is list: X= np.array(X)
            
            logps = X[:,0]
            posts = X[:,-self.nPosteriors:].tolist()
                
            p_labels, (ACC, MSE, SCC), p_vals = svm.svm_predict(logps, posts, self.dt, options='-q')
            sigma = np.sqrt(MSE)

            l_err = p_vals + self.ths_mult*sigma - logps #- self.logp_offset
            return l_err

        elif self.method == 'rnd':
            if len(np.shape(X))==1: X = [X]

            l_err = np.random.choice([-1, 1], size=len(X), p=[self.class_weight, 1.0-self.class_weight])
            return l_err


    def decision_function(self, X):

        ## return self.dt.decision_function(X)
        if self.method.find('svm')>=0  or self.method == 'fixed':
            if type(X) is not list:
                return self.predict(X.tolist())
            else:
                return self.predict(X)
        elif self.method.find('sgd')>=0:
            X_features = self.rbf_feature.transform(X)
            return self.dt.decision_function(X_features)
        else:
            print "Not implemented"
            sys.exit()

        return 
        
    def score(self, X, y):
        if self.method.find('svm')>=0:
            return self.dt.score(X,y)
        else:
            print "Not implemented funciton Score"
            return 

        
    def save_model(self, fileName):
        if self.dt is None and self.method.find('progress')<0: 
            print "No trained classifier"
            return
        
        if self.method.find('svm')>=0:       
            sys.path.insert(0, '/usr/lib/pymodules/python2.7')
            import svmutil as svm            
            svm.svm_save_model(fileName, self.dt)
        elif self.method.find('sgd')>=0:            
            import pickle
            with open(fileName, 'wb') as f:
                pickle.dump(self.dt, f)
                pickle.dump(self.rbf_feature, f)
        else:
            print "Not available method"

            
    def load_model(self, fileName):        
        if self.method.find('svm')>=0:
            sys.path.insert(0, '/usr/lib/pymodules/python2.7')
            import svmutil as svm            
            self.dt = svm.svm_load_model(fileName) 
        elif self.method.find('sgd')>=0:
            import pickle
            with open(fileName, 'rb') as f:
                self.dt = pickle.load(f)
                self.rbf_feature = pickle.load(f)
        else:
            print "Not available method"
        

####################################################################
# functions for paralell computation
####################################################################

def run_classifiers(idx, processed_data_path, task_name, method,\
                    ROC_data, ROC_dict, SVM_dict, HMM_dict,\
                    startIdx=4, nState=25, \
                    modeling_pkl_prefix=None, failsafe=False, delay_estimation=False,\
                    adaptation=False, \
                    save_model=False, load_model=False, n_jobs=-1):

    #-----------------------------------------------------------------------------------------
    nPoints    = ROC_dict['nPoints']

    # pass method if there is existing result
    data = {}
    data = util.reset_roc_data(data, [method], [], nPoints)
    if ROC_data[method]['complete'] == True: return data
    #-----------------------------------------------------------------------------------------

    if modeling_pkl_prefix is not None:
        modeling_pkl = os.path.join(processed_data_path, modeling_pkl_prefix+'_'+str(idx)+'.pkl')            
    else:        
        modeling_pkl = os.path.join(processed_data_path, 'hmm_'+task_name+'_'+str(idx)+'.pkl')

    print "start to load hmm data, ", modeling_pkl
    d            = ut.load_pickle(modeling_pkl)
    for k, v in d.iteritems():
        exec '%s = v' % k        

    # train a classifier and evaluate it using test data.
    if method == 'ipca' or method == 'osvm' or method == 'bpsvm' or method == 'sgd' or \
      method == 'mlp':
        if method == 'osvm': raw_data_idx = 0
        elif method == 'bpsvm': raw_data_idx = 1

        X_train                 = ll_window_train_X
        Y_train                 = ll_window_train_Y
        idx_train               = None        
        ll_classifier_test_X    = ll_window_test_X
        ll_classifier_test_Y    = ll_window_test_Y
        ll_classifier_test_idx  = None        
        ll_classifier_test_labels = None
        ## step_idx_l = raw_data[raw_data_idx][idx]['step_idx_l']

        if adaptation is True:
            X_train_p  = ll_window_ptrain_X
            Y_train_p  = ll_window_ptrain_Y
            idx_train_p= None

        # TODO: set automatically!
        if processed_data_path.find('feeding')>=0:
            nLength = 140
        else:            
            nLength = 200
    else:

        ## nState, ll_classifier_train_?, ll_classifier_test_?, nLength    
        ll_classifier_test_labels = d.get('ll_classifier_test_labels', None)

        
        if 'diag' in method:
            ll_classifier_train_X   = ll_classifier_diag_train_X
            ll_classifier_train_Y   = ll_classifier_diag_train_Y
            ll_classifier_train_idx = ll_classifier_diag_train_idx
            ll_classifier_test_X    = ll_classifier_diag_test_X
            ll_classifier_test_Y    = ll_classifier_diag_test_Y
            ll_classifier_test_idx  = ll_classifier_diag_test_idx
        elif method =='progress_osvm' or method == 'progress_svm':
            ## # temp
            from hrl_anomaly_detection.hmm import learning_hmm as hmm
            ll_classifier_ep_train_X, ll_classifier_ep_train_Y, ll_classifier_ep_train_idx =\
              hmm.getEntropyFeaturesFromHMMInducedFeatures(ll_classifier_train_X, \
                                                           ll_classifier_train_Y, \
                                                           ll_classifier_train_idx, nState)
            ll_classifier_ep_test_X, ll_classifier_ep_test_Y, ll_classifier_ep_test_idx =\
              hmm.getEntropyFeaturesFromHMMInducedFeatures(ll_classifier_test_X, \
                                                           ll_classifier_test_Y, \
                                                           ll_classifier_test_idx, nState)

            ll_classifier_train_X   = ll_classifier_ep_train_X
            ll_classifier_train_Y   = ll_classifier_ep_train_Y
            ll_classifier_train_idx = ll_classifier_ep_train_idx
            ll_classifier_test_X    = ll_classifier_ep_test_X
            ll_classifier_test_Y    = ll_classifier_ep_test_Y
            ll_classifier_test_idx  = ll_classifier_ep_test_idx

        if method == 'hmmosvm' or method == 'progress_osvm' or method == 'hmmgp':            
            normal_idx = [x for x in range(len(ll_classifier_train_X)) if ll_classifier_train_Y[x][0]<0 ]
            ll_classifier_train_X = np.array(ll_classifier_train_X)[normal_idx]
            ll_classifier_train_Y = np.array(ll_classifier_train_Y)[normal_idx]
            ll_classifier_train_idx = np.array(ll_classifier_train_idx)[normal_idx]

        if method == 'hmmgp':
            ## nSubSample = 50 #temp!!!!!!!!!!!!!
            nSubSample = 20 #20 # 20 
            nMaxData   = 50 #40 100
            rnd_sample = True #False
            
            ll_classifier_train_X, ll_classifier_train_Y, ll_classifier_train_idx =\
              dm.subsampleData(ll_classifier_train_X, ll_classifier_train_Y, ll_classifier_train_idx,\
                               nSubSample=nSubSample, nMaxData=nMaxData, rnd_sample=rnd_sample)
            
        # flatten the data
        if method.find('svm')>=0 or method.find('sgd')>=0: remove_fp=True
        else: remove_fp = False
        X_train, Y_train, idx_train = dm.flattenSample(ll_classifier_train_X, \
                                                                   ll_classifier_train_Y, \
                                                                   ll_classifier_train_idx,\
                                                                   remove_fp=remove_fp)

        if method.find('progress')>=0 and adaptation is True:

            mu_list  = [ [] for i in xrange(nState) ]
            std_list = [ [] for i in xrange(nState) ]

            for i in xrange(len(nor_train_inds)):
                # compute theta per person
                if n_jobs == 1: parallel = True
                else: parallel = False
                from hrl_anomaly_detection.classifiers import hmmd
                dtc = hmmd.hmmd( nPosteriors=nState, nLength=nLength, parallel=parallel )                

                X_train_p, Y_train_p, idx_train_p =\
                dm.flattenSample(np.array(ll_classifier_train_X)[nor_train_inds[i]], \
                                 np.array(ll_classifier_train_Y)[nor_train_inds[i]], \
                                 np.array(ll_classifier_train_idx)[nor_train_inds[i]],\
                                 remove_fp=remove_fp)
                dtc.fit( X_train_p, Y_train_p, idx_train_p )
                for j in xrange(nState):
                    mu_list[j].append( dtc.ll_mu[j] )
                    std_list[j].append( dtc.ll_std[j] )

            X_train_p, Y_train_p, idx_train_p =\
            dm.flattenSample(np.array(ll_classifier_ptrain_X), \
                             np.array(ll_classifier_ptrain_Y), \
                             np.array(ll_classifier_ptrain_idx),\
                             remove_fp=remove_fp)



    #-----------------------------------------------------------------------------------------
    # Generate parameter list for ROC curve
    # pass method if there is existing result
    # data preparation
    if (method.find('svm')>=0 or method.find('ipca')>=0 or method.find('mlp')>=0) and \
      not(method == 'osvm' or method == 'bpsvm'):
        scr_pkl  = os.path.join(processed_data_path, 'scr_'+method+'_'+str(idx)+'.pkl')

        import pickle
        if load_model:
            with open(scr_pkl, 'rb') as f:            
                scaler = pickle.load(f)

        print "Training data is scaled ", np.shape(X_train)
        if method.find('mlp')>=0: scaler = preprocessing.MinMaxScaler()
        else:                     scaler = preprocessing.StandardScaler()
        X_train = scaler.fit_transform(X_train)

        if save_model:
            with open(scr_pkl, 'wb') as f:
                pickle.dump(scaler, f)
                
        
    print method, " : Before classification : ", np.shape(X_train), np.shape(Y_train)

    # Get TEST data
    X_test = []
    Y_test = [] 
    for j in xrange(len(ll_classifier_test_X)):
        if len(ll_classifier_test_X[j])==0: continue

        if (method.find('svm')>=0 or method.find('ipca')>=0 or method.find('mlp')>=0) and \
          not(method == 'osvm' or method == 'bpsvm'):
            try:
                X = scaler.transform(ll_classifier_test_X[j])
            except:
                print "Feature: ", j, np.shape(ll_classifier_test_X)
                for k in xrange(len(ll_classifier_test_X[j])):
                    print ll_classifier_test_X[j][k]
                    if np.nan in ll_classifier_test_X[j][k]:
                        print "00000000000000000000000000000000000000000000000000"
                        print k, ll_classifier_test_X[j][k]
                        print "00000000000000000000000000000000000000000000000000"
                sys.exit()
        else:
            X = ll_classifier_test_X[j]

        X_test.append(X)
        Y_test.append(np.ones(len(X))*ll_classifier_test_Y[j])


    if modeling_pkl_prefix is not None:
        clf_pkl = os.path.join(processed_data_path, 'clf_'+modeling_pkl_prefix+'_'+method+'_'+\
                               str(idx)+'.pkl')
    else:
        clf_pkl = os.path.join(processed_data_path, 'clf_'+method+'_'+\
                               str(idx)+'.pkl')

    # classifier # TODO: need to make it efficient!!
    if n_jobs == 1: parallel = True
    else: parallel = False
    if method == 'progress' or method == 'progress_diag':
        from hrl_anomaly_detection.classifiers import hmmd
        dtc = hmmd.hmmd( nPosteriors=nState, nLength=nLength, parallel=parallel )
    elif method == 'hmmgp':
        from hrl_anomaly_detection.classifiers import hmmgp
        dtc = hmmgp.hmmgp( nPosteriors=nState, parallel=parallel )
    elif method == 'ipca':
        from hrl_anomaly_detection.classifiers import ipca
        dtc = ipca.ipca(n_components=len(X_train[0])/8, batch_size=1000 )        
    elif method == 'mlp':
        from hrl_anomaly_detection.classifiers import mlp
        dtc = mlp.mlp(enconding_dim=4, patience=10)        
    else:
        dtc = classifier( method=method, nPosteriors=nState, nLength=nLength, parallel=parallel )

        
    for j in xrange(nPoints):
            
        if load_model: dtc.load_model(clf_pkl)
        dtc.set_params( **SVM_dict )
        ret = True
        
        if method == 'svm' or method == 'hmmsvm_diag' or \
          method == 'bpsvm' or method == 'sgd' or method == 'progress_svm' or \
          method == 'svm_fixed':
            if method == 'svm_fixed': 
                weights = ROC_dict['svm_param_range']
            else:
                weights = ROC_dict[method+'_param_range']
            dtc.set_params( class_weight=weights[j] )
            if not load_model: ret = dtc.fit(X_train, Y_train, idx_train)
        elif method == 'hmmosvm' or method == 'osvm' or method == 'progress_osvm':
            weights = ROC_dict[method+'_param_range']
            dtc.set_params( svm_type=2 )
            dtc.set_params( kernel_type=2 )
            dtc.set_params( gamma=weights[j] )
            if not load_model: ret = dtc.fit(X_train, np.array(Y_train)*-1.0)
        elif method == 'progress' or method == 'progress_diag' or \
          method == 'fixed' or method == 'hmmgp' or method == 'ipca' or method == 'mlp':
            thresholds = ROC_dict[method+'_param_range']
            if method == 'ipca' or method == 'mlp':
                dtc.set_params( ths = thresholds[j] )
            else:
                dtc.set_params( ths_mult = thresholds[j] )
                
            if not load_model:
                if j==0:

                    if method == 'mlp':
                        ret = dtc.fit(X_train, y=Y_train, ll_idx=idx_train, Xv=X_test[1])
                    else:
                        ret = dtc.fit(X_train, y=Y_train, ll_idx=idx_train)

                    # Adaptation
                    if adaptation is True:
                        if method == 'ipca' or method == 'mlp':
                            dtc.partial_fit(X_train_p)
                        else:
                            dtc.partial_fit(X_train_p, Y_train_p, idx_train_p,
                                            mu_mu=np.mean(mu_list, axis=1),
                                            std_mu=np.std(mu_list, axis=1),
                                            mu_std=np.mean(std_list, axis=1),
                                            std_std=np.std(std_list, axis=1) )
                   
        elif method == 'change':
            thresholds = ROC_dict[method+'_param_range']
            dtc.set_params( ths_mult = thresholds[j] )
            if not load_model:                
                if j==0: ret = dtc.fit(ll_classifier_train_X, ll_classifier_train_Y, ll_classifier_train_idx)
        else:
            # rnd
            weights = ROC_dict[method+'_param_range']
            dtc.set_params( class_weight=weights[j] )
            dtc.set_params( ths_mult = weights[j] )
            ret = True

        if ret is False: raise ValueError("Classifier fitting error")
        if j==0 and save_model: dtc.save_model(clf_pkl)
            
        # evaluate the classifier
        tp_l = []
        fp_l = []
        tn_l = []
        fn_l = []
        delay_l   = []
        delay_idx = 0
        tp_idx_l  = []
        fn_labels = []
        for ii in xrange(len(X_test)):
            if len(Y_test[ii])==0: continue

            if method.find('osvm')>=0:
                est_y = dtc.predict(X_test[ii], y=np.array(Y_test[ii])*-1.0)
                est_y = np.array(est_y)* -1.0
            else:
                est_y = dtc.predict(X_test[ii], y=Y_test[ii])

            anomaly = False
            for jj in xrange(len(est_y)):
                if est_y[jj] > 0.0:
                    if ll_classifier_test_idx is not None and Y_test[ii][0]>0:
                        try:
                            delay_idx = ll_classifier_test_idx[ii][jj]
                        except:
                            print "Error!!!!!!!!!!!!!!!!!!"
                            print np.shape(ll_classifier_test_idx), ii, jj
                        if delay_estimation: # and False:
                            if step_idx_l[ii] is None:
                                print "Wrong step idx setting"
                                sys.exit()
                            if delay_idx-step_idx_l[ii]<0: continue
                            delay_l.append( delay_idx-step_idx_l[ii] )
                        else:
                            delay_l.append( delay_idx )
                    if Y_test[ii][0] > 0:
                        tp_idx_l.append(ii)

                    anomaly = True
                    break        

            if Y_test[ii][0] > 0.0:
                if anomaly:
                    if delay_estimation and False:
                        if delay_l[-1] >= 0:
                            tp_l.append(1)
                        else:
                            fp_l.append(1)
                            del delay_l[-1]
                    else:
                        tp_l.append(1)
                else:
                    fn_l.append(1)
                    if ll_classifier_test_labels is not None:
                        fn_labels.append(ll_classifier_test_labels[ii])
            elif Y_test[ii][0] <= 0.0:
                if anomaly: fp_l.append(1)
                else: tn_l.append(1)

        data[method]['tp_l'][j] += tp_l
        data[method]['fp_l'][j] += fp_l
        data[method]['fn_l'][j] += fn_l
        data[method]['tn_l'][j] += tn_l
        data[method]['delay_l'][j] += delay_l
        data[method]['tp_idx_l'][j] += tp_idx_l
        data[method]['fn_labels'][j] += fn_labels

    print "finished ", idx, method
    return data


def run_classifiers_boost(idx, processed_data_path, task_name, method_list,\
                          ROC_data, param_dict,\
                          raw_data=None, startIdx=4, nState=25, nSubSample=20, \
                          prefix=None, suffix=None,\
                          delay_estimation=False,\
                          save_model=False, n_jobs=-1):
                          
    HMM_dict = param_dict['HMM']
    SVM_dict = param_dict['SVM']
    ROC_dict = param_dict['ROC'] 
    nPoints  = ROC_dict['nPoints']
    method   = method_list[0][:-1]
    nDetector = len(method_list)

    #-----------------------------------------------------------------------------------------
    # pass method if there is existing result
    data = {}
    data = util.reset_roc_data(data, [method], [], nPoints)
    if ROC_data[method]['complete'] == True: return data
    #-----------------------------------------------------------------------------------------

    # train a classifier and evaluate it using test data.
    X_train = []
    Y_train = []
    Idx_train = []
    X_test  = []
    Y_test  = []
    Idx_test  = []

    for clf_idx in xrange(len(method_list)):
        
        modeling_pkl = os.path.join(processed_data_path, 'hmm_'+task_name+'_'+\
                                    str(idx)+'_c'+str(clf_idx)+'.pkl')            

        print "start to load hmm data, ", modeling_pkl
        d            = ut.load_pickle(modeling_pkl)
        for k, v in d.iteritems():
            exec '%s = v' % k        
        ## nState, ll_classifier_train_?, ll_classifier_test_?, nLength    
        ll_classifier_test_labels = d.get('ll_classifier_test_labels', None)

        if method_list[clf_idx].find('hmmgp')>=0:            
            normal_idx = [x for x in range(len(ll_classifier_train_X)) if ll_classifier_train_Y[x][0]<0 ]
            ll_classifier_train_X = np.array(ll_classifier_train_X)[normal_idx]
            ll_classifier_train_Y = np.array(ll_classifier_train_Y)[normal_idx]
            ll_classifier_train_idx = np.array(ll_classifier_train_idx)[normal_idx]

            ## nSubSample = 50 #temp!!!!!!!!!!!!!
            #nSubSample = 20 #20 # 20 
            nMaxData   = 50 #40 100
            rnd_sample = True #False

            print "nSubsample : ", nSubSample
            ll_classifier_train_X, ll_classifier_train_Y, ll_classifier_train_idx =\
              dm.subsampleData(ll_classifier_train_X, ll_classifier_train_Y, ll_classifier_train_idx,\
                               nSubSample=nSubSample, nMaxData=nMaxData, rnd_sample=rnd_sample)

        
        # flatten the data
        X_train_flat, Y_train_flat, idx_train_flat = dm.flattenSample(ll_classifier_train_X, \
                                                                      ll_classifier_train_Y, \
                                                                      ll_classifier_train_idx,\
                                                                      remove_fp=False)
        X_train.append(X_train_flat)
        Y_train.append(Y_train_flat)
        Idx_train.append(idx_train_flat)
    
        # Generate parameter list for ROC curve pass method if there is existing result
        # data preparation
        X_test_flat = []
        Y_test_flat = [] 
        for j in xrange(len(ll_classifier_test_X)):
            if len(ll_classifier_test_X[j])==0: continue

            X_test_flat.append(ll_classifier_test_X[j])
            Y_test_flat.append(ll_classifier_test_Y[j])

        X_test.append(X_test_flat)
        Y_test.append(Y_test_flat)

    #-----------------------------------------------------------------------------------------
    # classifier # TODO: need to make it efficient!!
    if n_jobs == 1: parallel = True
    else: parallel = False
    dtc = {}
    if method_list[0][:-1] == 'progress' or method_list[0][:-1] == 'progress_diag':
        from hrl_anomaly_detection.classifiers import hmmd
        dtc[0] = hmmd.hmmd( nPosteriors=nState, nLength=nLength, parallel=parallel )
    else:    
        dtc[0] = classifier( method=method_list[0][:-1], nPosteriors=nState, nLength=nLength,
                             parallel=parallel )
    clf_pkl = []
    clf_pkl.append(os.path.join(processed_data_path, 'clf_'+method_list[0]+'_'+str(idx)+'.pkl'))
    if nDetector>1:
        if method_list[1][:-1] == 'progress' or method_list[1][:-1] == 'progress_diag':
            from hrl_anomaly_detection.classifiers import hmmd
            dtc[1] = hmmd.hmmd( nPosteriors=nState, nLength=nLength, parallel=parallel )
        else:        
            dtc[1] = classifier( method=method_list[1][:-1], nPosteriors=nState, nLength=nLength,
                                 parallel=parallel )
        clf_pkl.append(os.path.join(processed_data_path, 'clf_'+method_list[1]+'_'+str(idx)+'.pkl'))
        
    for j in xrange(nPoints):

        # Training
        for clf_idx in xrange(nDetector):

            X = X_train[clf_idx]
            Y = Y_train[clf_idx]
            inds = Idx_train[clf_idx]
            
            dtc[clf_idx].set_params( **SVM_dict )
            if method_list[clf_idx].find('progress')>=0 or method_list[clf_idx] == 'fixed' or \
              method_list[clf_idx].find('hmmgp')>=0:
                thresholds = ROC_dict[method_list[clf_idx]+'_param_range']
                dtc[clf_idx].set_params( ths_mult = thresholds[j] )
                if not(j==0): continue
                ret = dtc[clf_idx].fit(X, Y, inds)
                if ret is False: raise ValueError("Classifier fitting error")
            else:
                raise ValueError("Not available method: "+method_list[clf_idx])

            if j==0 and save_model: dtc[clf_idx].save_model(clf_pkl[clf_idx])


        # evaluate the classifier
        tp_l = []
        fp_l = []
        tn_l = []
        fn_l = []
        delay_l   = []
        delay_idx = 0
        tp_idx_l  = []
        fn_labels = []
        for ii in xrange(len(X_test[0])): # per sample
            if len(Y_test[0][ii])==0: continue

            est_y = []
            for clf_idx in xrange(nDetector):
                est_y.append(dtc[clf_idx].predict(X_test[clf_idx][ii], y=Y_test[clf_idx][ii]))

            true_y = Y_test[0][ii][0]

            # Combine classification result
            anomaly = False
            for jj in xrange(len(est_y[0])): # per time length


                detection_flag = False
                for kk in xrange(nDetector):
                    if est_y[kk][jj] > 0.0:
                        detection_flag = True
                        break
                
                if detection_flag:
                    if ll_classifier_test_idx is not None and true_y>0:
                        try:
                            delay_idx = ll_classifier_test_idx[ii][jj]
                        except:
                            raise ValueError("Classifier test index error")

                        # Only work with simulated anomalies
                        if delay_estimation: 
                            if step_idx_l[ii] is None:
                                raise ValueError("Wrong step idx setting")
                            # Ignore early detection
                            if delay_idx-step_idx_l[ii]<0: continue
                            delay_l.append( delay_idx-step_idx_l[ii] )
                        else:
                            delay_l.append( delay_idx )
                    if true_y > 0:
                        tp_idx_l.append(ii)

                    anomaly = True
                    break        

            if true_y > 0.0:
                if anomaly:
                    if delay_estimation and False:
                        if delay_l[-1] >= 0:
                            tp_l.append(1)
                        else:
                            fp_l.append(1)
                            del delay_l[-1]
                    else:
                        tp_l.append(1)
                else:
                    fn_l.append(1)
                    if ll_classifier_test_labels is not None:
                        fn_labels.append(ll_classifier_test_labels[ii])
            elif true_y <= 0.0:
                if anomaly: fp_l.append(1)
                else: tn_l.append(1)

        data[method]['tp_l'][j] += tp_l
        data[method]['fp_l'][j] += fp_l
        data[method]['fn_l'][j] += fn_l
        data[method]['tn_l'][j] += tn_l
        data[method]['delay_l'][j] += delay_l
        data[method]['tp_idx_l'][j] += tp_idx_l
        data[method]['fn_labels'][j] += fn_labels

    print "finished ", idx, method
    return data


