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

# system & utils
import os, sys, copy, random
import numpy as np
from joblib import Parallel, delayed

from hrl_anomaly_detection import data_manager as dm
from hrl_anomaly_detection import util as util
import hrl_lib.util as ut
from hrl_anomaly_detection.hmm import learning_hmm as hmm
import hrl_anomaly_detection.classifiers.classifier as cf


def rnd_():
    
    # random sampling?
    ## nSample     = 30
    window_size = 30
    window_step = 5
    X_train = []
    Y_train = []
    ml_dict = {}
    for i in xrange(len(x[0])): # per sample

        s_l = np.arange(startIdx, len(x[0][i])-int(window_size*1.5), window_step)            
        ## s_l = np.random.randint(startIdx, len(x[0][i])-window_size*2, nSample)

        for j in s_l:
            block = x[:,i,j:j+window_size]

            # zero mean to resolve signal displacements
            block -= np.mean(block, axis=1)[:,np.newaxis]

            X_train.append( block )
            Y_train.append( label[i] )
    
def feature_omp(x, label, D0=None, n_iter=1000, sp_ratio=0.1):
    ''' Feature-wise omp '''
    from ksvd import KSVD, KSVD_Encode

    # train feature-wise omp
    X_ = []
    for i in xrange(len(x[0])): # per sample
        X_.append(x[:,i,:]) #-np.mean(x[:,i,:], axis=1)[:, np.newaxis])
    Y_ = copy.copy(label)

    dimension = len(X_[0][0]) #window_size
    dict_size = int(dimension*2) ##10, 1.5)
    n_examples = len(X_)
    target_sparsity = int(sp_ratio*dimension)

    gs = None
    Ds = {}
    X_ = np.array(X_)
    for i in xrange(len(X_[0])): # per feature
        print i, ' / ', len(X_[0])

        if D0 is None:
            # X \simeq g * D
            # D is the dictionary with `dict_size` by `dimension`
            # g is the code book with `n_examples` by `dict_size`
            D, g = KSVD(X_[:,i,:], dict_size, target_sparsity, n_iter,
                            print_interval = 25,
                            enable_printing = True, enable_threading = True)
            Ds[i] = D
        else:
            g = KSVD_Encode(X_[:,i,:], D0[i], target_sparsity)

        if gs is None:
            gs = g
        else:
            gs = np.hstack([gs, g])

    if D0 is None: return Ds, gs, Y_
    else:          return D0, gs, Y_


def m_omp(x, label, D0=None, n_iter=1000, sp_ratio=0.05, idx_list=None):
    ''' Multichannel OMP '''
    from ksvd import KSVD, KSVD_Encode

    #idx_list = None

    # train multichannel omp?
    if idx_list is None:
        X_ = []
        Y_ = []
        for i in xrange(len(x[0])): # per sample
            for j in xrange(len(x)): # per feature
                X_.append( x[j,i,:] - np.mean(x[j,i,:5]) ) 
        Y_ = copy.copy(label)
    else:
        X_ = []
        Y_ = []
        for i in xrange(len(x[0])): # per sample
            if idx_list[i] is None: continue

            for j in xrange(len(x)): # per feature
                x_j = x[j,i,:idx_list[i]+1].tolist()
                x_j = x_j + [x_j[0]]*(len(x[j,i])-len(x_j)) 
                ## x_j = x_j + [x_j[-1]]*(len(x[j,i])-len(x_j)) 
                ## x_j = x_j + [0]*(len(x[j,i])-len(x_j)) 
                X_.append( x_j ) 

            Y_.append(label[i])


    n_features = len(x)
    dimension  = len(x[0][0]) 
    dict_size  = int(dimension*8)
    n_examples = len(X_)
    ## target_sparsity = int(sp_ratio*dimension)
    target_sparsity = int(sp_ratio*dict_size)

    if D0 is None:
        # X \simeq g * D
        # D is the dictionary with `dict_size` by `dimension`
        # g is the code book with `n_examples` by `dict_size`
        D, g = KSVD(np.array(X_), dict_size, target_sparsity, n_iter,
                        print_interval = 25,
                        enable_printing = True, enable_threading = True)
    else:
        if idx_list is None:
            g = KSVD_Encode(np.array(X_), D0, target_sparsity)
        else:
            target_sparsity = int(sp_ratio*dict_size) if int(sp_ratio*dict_size) > 0 else 1
            
            g = []
            for i in xrange(len(X_)):
                g.append( KSVD_Encode(np.array(X_[i]).reshape(1,-1), D0, target_sparsity).tolist() )
            g = np.array(g)
            
    # Stacking?
    gs = None
    for i in xrange(len(Y_)): # per sample

        single_g = g[i*n_features:(i+1)*n_features,:].flatten()
        ## single_g /= np.linalg.norm(single_g)

        if gs is None: gs = single_g
        else: gs = np.vstack([gs, single_g])

    if D0 is None: return D, gs, Y_
    else:          return D0, gs, Y_


def window_omp(x, label, D0=None, n_iter=500, sp_ratio=0.05, idx_list=None):
    ''' Multichannel OMP with sliding window'''
    from ksvd import KSVD, KSVD_Encode

    ## idx_list = None
    window_size = 130 
    window_step = 10

    # train multichannel omp?
    X_ = []
    Y_ = []
    if idx_list is None:
        for i in xrange(len(x[0])): # per sample
            for k in xrange(window_size, len(x[0][i]), window_step):                
                for j in xrange(len(x)): # per feature
                    X_.append(x[j,i,k-window_size:k]) # -np.mean(x[:,i,j])) 
    else:
        for i in xrange(len(x[0])): # per sample
            if idx_list[i] is None: continue
            for j in xrange(len(x)): # per feature
                if idx_list[i]-window_size < -1:
                    x_j = [x[j,i,0]]*(abs(idx_list[i]-window_size)-1) + x[j,i,0:idx_list[i]+1].tolist()
                else:
                    x_j = x[j,i,idx_list[i]+1-window_size:idx_list[i]+1].tolist()
                X_.append( x_j ) 
            Y_.append( label[i] )


    n_features = len(x)
    dimension  = window_size 
    dict_size  = int(dimension*8)
    n_examples = len(X_)
    target_sparsity = int(sp_ratio*dict_size)

    if D0 is None:
        # X \simeq g * D
        # D is the dictionary with `dict_size` by `dimension`
        # g is the code book with `n_examples` by `dict_size`
        D, g = KSVD(np.array(X_), dict_size, target_sparsity, n_iter,
                        print_interval = 25,
                        enable_printing = True, enable_threading = True)
    else:
        D = D0
        g = KSVD_Encode(np.array(X_), D0, target_sparsity)
        g = np.array(g)
    
    # Stacking?
    gs = None
    if idx_list is None:
        n_window_per_sample = len(range(window_size, len(x[0][0]), window_step))
        Y_ = []
        
        for i in xrange(len(x[0])): # per sample
            for k in xrange(n_window_per_sample):
                single_g = g[i*(n_window_per_sample*n_features)+k*n_features: \
                             i*(n_window_per_sample*n_features)+(k+1)*n_features ]

                if gs is None: gs = single_g.flatten()
                else: gs = np.vstack([gs, single_g.flatten()])

                Y_.append(label[i])
    else:
        for i in xrange(len(X_)/n_features): # per sample
            single_g = g[i*n_features : (i+1)*n_features ]

            if gs is None: gs = single_g.flatten()
            else: gs = np.vstack([gs, single_g.flatten()])

    if D0 is None: return D, gs, Y_
    else:          return D0, gs, Y_



def w_omp(x, label, D0=None, n_iter=1000, sp_ratio=0.05):
    ''' Multichannel OMP with random wavelet dictionary'''
    from sklearn.decomposition import SparseCoder

    def ricker_function(resolution, center, width):
        """Discrete sub-sampled Ricker (Mexican hat) wavelet"""
        x = np.linspace(0, resolution - 1, resolution)
        x = ((2 / ((np.sqrt(3 * width) * np.pi ** 1 / 4)))
             * (1 - ((x - center) ** 2 / width ** 2))
             * np.exp((-(x - center) ** 2) / (2 * width ** 2)))
        return x


    def ricker_matrix(width, resolution, n_components):
        """Dictionary of Ricker (Mexican hat) wavelets"""
        centers = np.linspace(0, resolution - 1, n_components)
        D = np.empty((n_components, resolution))
        for i, center in enumerate(centers):
            D[i] = ricker_function(resolution, center, width)
        D /= np.sqrt(np.sum(D ** 2, axis=1))[:, np.newaxis]
        return D


    # train multichannel omp?
    X_ = []
    Y_ = []
    for i in xrange(len(x[0])): # per sample
        for j in xrange(len(x)): # per feature
            X_.append(x[j,i,:]) 
    Y_ = copy.copy(label)

    n_features = len(x)
    dimension  = len(X_[0]) 
    n_examples = len(X_)
    target_sparsity = int(sp_ratio*dimension)
    n_components = 60 #dimension / 2
    w_list = np.logspace(0, np.log10(dimension), dimension/8).astype(int)
    ## w_list = np.linspace(1, dimension-1, dimension/2)

    # Compute a wavelet dictionary
    if D0 is None:
        D = np.r_[tuple(ricker_matrix(width=w, resolution=dimension,
                                      n_components=n_components )
                                      for w in w_list)]
    else:
        D = D0

    gs = None
    X_ = np.array(X_)

    # X \simeq g * D
    coder = SparseCoder(dictionary=D, transform_n_nonzero_coefs=target_sparsity,
                        transform_alpha=None, transform_algorithm='omp', n_jobs=1)
    g = coder.transform(X_)

    ## X_est = np.ravel(np.dot(g, D))
    ## squared_error = np.sum((X_est - np.ravel(X_)) ** 2)
    ## print squared_error
    ## for i in xrange(len(g)):
    ##     print np.shape(D), np.shape(g), np.shape(np.dot(g[i:i+1,:],D))
    ##     plot_decoder(X_[i], np.dot(g[i:i+1,:],D)[0] )    
    
    # Stacking?
    for i in xrange(len(x[0])): # per sample

        single_g = g[i*n_features:(i+1)*n_features,:].flatten()

        if gs is None: gs = single_g
        else: gs = np.vstack([gs, single_g])

    return D, gs, Y_


    
def time_omp(x, label, D0=None, n_iter=2000, sp_ratio=0.1, idx_list=None):
    ''' Time-sample OMP with max pooling and contrast normalization'''
    from ksvd import KSVD, KSVD_Encode
    
    # train time-wise omp
    X_ = []
    Y_ = []
    for i in xrange(len(x[0])): # per sample
        for j in xrange(len(x[0][i])): # per time
            X_.append(x[:,i,j]) #-np.mean(x[:,i,j])) 
            ## Y_.append(label[i])

    dimension  = len(X_[0]) 
    n_examples = len(X_)
    dict_size  = int(dimension*2)
    target_sparsity = int(sp_ratio*dict_size)

    X_ = np.array(X_)
    if D0 is None:
        # X \simeq g * D
        # D is the dictionary with `dict_size` by `dimension`
        # g is the code book with `n_examples` by `dict_size`
        D, g = KSVD(X_, dict_size, target_sparsity, n_iter,
                        print_interval = 25,
                        enable_printing = True, enable_threading = True)
    else:        
        g = KSVD_Encode(X_, D0, target_sparsity)        


    # Fixed-size Max pooling?
    window_size = 30 # for max pooling?
    window_step = 5  # for max pooling?
    gs = None
    for i in xrange(len(x[0])): # per sample
        g_per_sample = g[i*len(x[0][i]):(i+1)*len(x[0][i]),:]

        if idx_list is None:
            for j in xrange(window_size, len(x[0][i]), window_step): # per time

                max_pool = np.amax( g_per_sample[j-window_size:j,:], axis=0 )
                ## max_pool /= np.linalg.norm(max_pool+1e-6)

                if gs is None: gs = max_pool
                else: gs = np.vstack([gs, max_pool])
                Y_.append(label[i])
        else:
            j = idx_list[i]
            if j is None: continue
            
            ## print i, len(idx_list), idx_list[i], " : ", window_size, j-window_size
            if j-window_size < 0: start_idx = 0
            else:start_idx = j-window_size 
            end_idx = j+1
            max_pool = np.amax( g_per_sample[start_idx:end_idx,:], axis=0 )
            if gs is None: gs = max_pool
            else: gs = np.vstack([gs, max_pool])
            Y_.append(label[i])

    print 

    if D0 is None: return D, gs, Y_
    else:          return D0, gs, Y_



def plot_decoder(x1,x2):
    # visualization
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(x1, 'b-')
    plt.plot(x2, 'r-', linewidth=3.0)
    plt.show()

    return


def anomaly_detection(X, Y, task_name, processed_data_path, param_dict, logp_viz=False, verbose=False,
                      weight=0.0, idx=0, n_jobs=-1):
    ''' Anomaly detector that return anomalous point on each data.
    '''
    HMM_dict = param_dict['HMM']
    SVM_dict = param_dict['SVM']
    ROC_dict = param_dict['ROC']
    
    # set parameters
    method  = 'hmmgp' #'progress'
    ## weights = ROC_dict[method+'_param_range']
    nMaxData   = 20 # The maximun number of executions to train GP
    nSubSample = 50 # The number of sub-samples from each execution to train GP

    # Load a generative model
    modeling_pkl = os.path.join(processed_data_path, 'hmm_'+task_name+'_'+str(idx)+'.pkl')

    if verbose: print "start to load hmm data, ", modeling_pkl
    d            = ut.load_pickle(modeling_pkl)
    ## Load local variables: nState, nEmissionDim, ll_classifier_train_?, ll_classifier_test_?, nLength    
    for k, v in d.iteritems():
        # Ignore predefined test data in the hmm object
        if not(k.find('test')>=0):
            exec '%s = v' % k

    ml = hmm.learning_hmm(nState, nEmissionDim, verbose=verbose) 
    ml.set_hmm_object(A,B,pi)
            
    # 1) Convert training data
    if method == 'hmmgp':

        idx_list = range(len(ll_classifier_train_X))
        random.shuffle(idx_list)
        ll_classifier_train_X = np.array(ll_classifier_train_X)[idx_list[:nMaxData]].tolist()
        ll_classifier_train_Y = np.array(ll_classifier_train_Y)[idx_list[:nMaxData]].tolist()
        ll_classifier_train_idx = np.array(ll_classifier_train_idx)[idx_list[:nMaxData]].tolist()

        new_X = []
        new_Y = []
        new_idx = []
        for i in xrange(len(ll_classifier_train_X)):
            idx_list = range(len(ll_classifier_train_X[i]))
            random.shuffle(idx_list)
            new_X.append( np.array(ll_classifier_train_X)[i,idx_list[:nSubSample]].tolist() )
            new_Y.append( np.array(ll_classifier_train_Y)[i,idx_list[:nSubSample]].tolist() )
            new_idx.append( np.array(ll_classifier_train_idx)[i,idx_list[:nSubSample]].tolist() )

        ll_classifier_train_X = new_X
        ll_classifier_train_Y = new_Y
        ll_classifier_train_idx = new_idx

        if len(ll_classifier_train_X)*len(ll_classifier_train_X[0]) > 1000:
            print "Too many input data for GP"
            sys.exit()

    X_train, Y_train, idx_train = dm.flattenSample(ll_classifier_train_X, \
                                                   ll_classifier_train_Y, \
                                                   ll_classifier_train_idx,\
                                                   remove_fp=False)
    if verbose: print method, " : Before classification : ", np.shape(X_train), np.shape(Y_train)

    # 2) Convert test data
    startIdx   = 4
    ll_classifier_test_X, ll_classifier_test_Y, ll_classifier_test_idx = \
      hmm.getHMMinducedFeaturesFromRawCombinedFeatures(ml, X * HMM_dict['scale'], Y, startIdx, \
                                                       n_jobs=n_jobs)

    if logp_viz:
        ll_logp_neg = np.array(ll_classifier_train_X)[:,:,0]
        ll_logp_pos = np.array(ll_classifier_test_X)[:,:,0]
        dv.vizLikelihood(ll_logp_neg, ll_logp_pos)
        sys.exit()

    # Create anomaly classifier
    dtc = cf.classifier( method=method, nPosteriors=nState, nLength=nLength, parallel=True )
    dtc.set_params( class_weight=weight )
    dtc.set_params( ths_mult = weight )    
    ret = dtc.fit(X_train, Y_train, idx_train)

    # anomaly detection
    detection_idx = [None for i in xrange(len(ll_classifier_test_X))]
    for ii in xrange(len(ll_classifier_test_X)):
        if len(ll_classifier_test_Y[ii])==0: continue

        est_y    = dtc.predict(ll_classifier_test_X[ii], y=ll_classifier_test_Y[ii])

        for jj in xrange(len(est_y)):
            if est_y[jj] > 0.0:                
                if ll_classifier_test_Y[ii][0] > 0:
                    detection_idx[ii] = ll_classifier_test_idx[ii][jj]
                    ## if ll_classifier_test_idx[ii][jj] ==4:
                    ##     print "Current likelihood: ", ll_classifier_test_X[ii][jj][0] 
                break

    return detection_idx


def get_isolation_data(idx, kFold_list, modeling_pkl, nState, \
                       failureData_ad, failureData_ai, failure_files, failure_labels, \
                       task_name, processed_data_path, param_dict, weight,\
                       verbose=False, n_jobs=-1):

    normalTrainIdx = kFold_list[0]
    abnormalTrainIdx = kFold_list[1]
    normalTestIdx = kFold_list[2]
    abnormalTestIdx = kFold_list[3]

    #-----------------------------------------------------------------------------------------
    # Anomaly Detection
    #-----------------------------------------------------------------------------------------
    dd = ut.load_pickle(modeling_pkl)
    nEmissionDim = dd['nEmissionDim']
    ml  = hmm.learning_hmm(nState, nEmissionDim, verbose=verbose) 
    ml.set_hmm_object(dd['A'],dd['B'],dd['pi'])

    # dim x sample x length
    abnormalTestData  = copy.copy(failureData_ad[:, abnormalTestIdx, :])
    abnormal_test_files  = np.array(failure_files)[abnormalTestIdx].tolist()

    testDataY = []
    abnormalTestIdxList  = []
    abnormalTestFileList = []
    for i, f in enumerate(abnormal_test_files):
        if f.find("failure")>=0:
            testDataY.append(1)
            abnormalTestIdxList.append(i)
            abnormalTestFileList.append(f.split('/')[-1])    

    detection_test_idx_list = anomaly_detection(abnormalTestData, testDataY, \
                                                task_name, processed_data_path, param_dict,\
                                                logp_viz=False, verbose=False, weight=weight,\
                                                idx=idx, n_jobs=n_jobs)

    #-----------------------------------------------------------------------------------------
    # Anomaly Isolation
    #-----------------------------------------------------------------------------------------
    # dim x sample x length
    abnormalTrainData = copy.copy(failureData_ai[:, abnormalTrainIdx, :])
    abnormalTestData  = copy.copy(failureData_ai[:, abnormalTestIdx, :])
    abnormalTrainLabel = copy.copy(failure_labels[abnormalTrainIdx])
    abnormalTestLabel  = copy.copy(failure_labels[abnormalTestIdx])

    ## omp feature extraction?
    # Train & test
    ## Ds, gs_train, y_train = feature_omp(abnormalTrainData, abnormalTrainLabel)
    ## _, gs_test, y_test = feature_omp(abnormalTestData, abnormalTestLabel, Ds)

    # Train & test
    print "Training: ", idx
    Ds, gs_train, y_train = m_omp(abnormalTrainData, abnormalTrainLabel)
    print "Testing: ", idx
    _, gs_test, y_test = m_omp(abnormalTestData, abnormalTestLabel, Ds,\
                               )
                                     ## idx_list=detection_test_idx_list)

    # Train & test
    ## Ds, gs_train, y_train = w_omp(abnormalTrainData, abnormalTrainLabel)
    ## _, gs_test, y_test = w_omp(abnormalTestData, abnormalTestLabel, Ds)

    # Train & test
    ## Ds, gs_train, y_train = time_omp(abnormalTrainData, abnormalTrainLabel)
    ## _, gs_test, y_test = time_omp(abnormalTestData, abnormalTestLabel, Ds, \
    ##                                     idx_list=detection_test_idx_list)

    # Train & test
    ## print "Training: ", idx
    ## Ds, gs_train, y_train = window_omp(abnormalTrainData, abnormalTrainLabel)
    ## print "Testing: ", idx
    ## _, gs_test, y_test = window_omp(abnormalTestData, abnormalTestLabel, Ds,\
    ##                                  idx_list=detection_test_idx_list)


    return idx, gs_train, y_train, gs_test, y_test


def get_hmm_isolation_data(idx, kFold_list, failureData_ad, failureData, failure_labels,
                           task_name, processed_data_path, param_dict, weight,\
                           ref_idx, n_jobs=-1 ):

    normalTrainIdx = kFold_list[0]
    abnormalTrainIdx = kFold_list[1]
    normalTestIdx = kFold_list[2]
    abnormalTestIdx = kFold_list[3]

    # dim x sample x length
    abnormalTrainData_ad  = copy.copy(failureData_ad[:, abnormalTrainIdx, :])
    abnormalTestData_ad   = copy.copy(failureData_ad[:, abnormalTestIdx, :])
    abnormalTrainData  = copy.copy(failureData[:, abnormalTrainIdx, :])
    abnormalTestData   = copy.copy(failureData[:, abnormalTestIdx, :])
    abnormalTrainLabel = copy.copy(failure_labels[abnormalTrainIdx])
    abnormalTestLabel  = copy.copy(failure_labels[abnormalTestIdx])

    #-----------------------------------------------------------------------------------------
    # Anomaly Detection
    #-----------------------------------------------------------------------------------------
    detection_train_idx_list = anomaly_detection(abnormalTrainData_ad, \
                                                 [1]*len(abnormalTrainData_ad[0]), \
                                                 task_name, processed_data_path, param_dict,\
                                                 logp_viz=False, verbose=False, \
                                                 weight=weight,\
                                                 idx=idx, n_jobs=n_jobs)
    detection_test_idx_list = anomaly_detection(abnormalTestData_ad, \
                                                [1]*len(abnormalTestData_ad[0]), \
                                                task_name, processed_data_path, param_dict,\
                                                logp_viz=False, verbose=False, \
                                                weight=weight,\
                                                idx=idx, n_jobs=n_jobs)

    #-----------------------------------------------------------------------------------------
    # Feature Extraction
    #-----------------------------------------------------------------------------------------
    x_train, y_train = get_cond_prob(idx, detection_train_idx_list, \
                                     abnormalTrainData, abnormalTrainLabel,\
                                     task_name, processed_data_path, param_dict, \
                                     ref_idx=ref_idx, plot=False, window=True, window_step=10 )
                                     
    x_test, y_test = get_cond_prob(idx, detection_test_idx_list, \
                                   abnormalTestData, abnormalTestLabel,\
                                   task_name, processed_data_path, param_dict, \
                                   ref_idx=ref_idx  )

    return idx, x_train, y_train, x_test, y_test

                                         

def get_cond_prob(idx, anomaly_idx_list, abnormalData, abnormalLabel, \
                  task_name, processed_data_path, param_dict,\
                  ref_idx, window_step=10, verbose=False, plot=False,\
                  window=False):
    ''' Get conditional probability vector when anomalies are detected
    '''

    # Load a generative model
    modeling_pkl = os.path.join(processed_data_path, 'hmm_'+task_name+'_'+str(idx)+'.pkl')
    d            = ut.load_pickle(modeling_pkl)
    ## Load local variables: nState, nEmissionDim, ll_classifier_train_?, ll_classifier_test_?, nLength    
    for k, v in d.iteritems():
        # Ignore predefined test data in the hmm object
        if not(k.find('test')>=0):
            exec '%s = v' % k

    ml = hmm.learning_hmm(nState, nEmissionDim, verbose=verbose) 
    ml.set_hmm_object(A,B,pi)
   
    x = []
    y = []
    for i, d_idx in enumerate(anomaly_idx_list):

        # Skip undetected anomaly
        if d_idx is None:
            continue

        if plot is False:
            if window:
                for j in range(-window_step, window_step):
                    if d_idx+1+j <= 4: continue
                    if d_idx+1+j > len(abnormalData[0,i]): continue
                    cp_vecs = ml.conditional_prob2( abnormalData[:,i,:d_idx+1+j]*\
                                                    param_dict['HMM']['scale'])
                    cp_vecs = (cp_vecs-np.amin(cp_vecs))/(np.amax(cp_vecs)-np.amin(cp_vecs))
                    x.append( cp_vecs )
                    y.append( abnormalLabel[i] )                                                    
            else:
                if d_idx+1 <= 0: continue
                if d_idx+1 > len(abnormalData[0,i]): continue
                cp_vecs = ml.conditional_prob2( abnormalData[:,i,:d_idx+1]*\
                                                param_dict['HMM']['scale'])
                cp_vecs = (cp_vecs-np.amin(cp_vecs))/(np.amax(cp_vecs)-np.amin(cp_vecs))

                x.append( cp_vecs )
                y.append( abnormalLabel[i] )
        else:
            cp_vecs = None
            for j in xrange(len(abnormalData[0,i,:])):
                if j<4: continue
                cp_vec = ml.conditional_prob2( abnormalData[:,i,:j+1]*\
                                               param_dict['HMM']['scale'] )
                if cp_vecs is None: cp_vecs = cp_vec
                else: cp_vecs = np.vstack([ cp_vecs, cp_vec])


            nPlot = len(cp_vecs[0])
            print "label: ", abnormalLabel[i], np.shape(cp_vecs)
            
            import matplotlib.pyplot as plt
            fig = plt.figure()

            for j in xrange(nPlot):
                ax = fig.add_subplot(nPlot*100+10+j+1)
                ax.plot(cp_vecs[:,j], 'r-')

            ## # ----------------------------------------------------
            ## ref_logps = np.array(ll_classifier_train_X)[:,:,0]
            ## ref_logps = np.swapaxes(ref_logps,0,1)
            ## for j in xrange(len(ll_classifier_train_Y[0])):
            ##     if ll_classifier_train_Y[0][j] > 0:
            ##         print "--------------------------------------"
            ##         print j, ' / ', len(ll_classifier_train_Y[0])
            ##         print "--------------------------------------"
            ##         break
                
            ## ref_logps_normal   = ref_logps[:j]
            ## ref_logps_abnormal = ref_logps[j:]
            ## # ----------------------------------------------------

            ## #temp
            ## ax = fig.add_subplot( nPlot*100+10+nPlot )
            ## ## ref_logps = np.array(ll_classifier_train_X)[:,i,0]
            ## ## ax = fig.add_subplot(111)
            ## ax.plot(ref_logps_normal, 'b-')
            ## ax.plot(cp_vecs[:,-1], 'r-')
            ax.plot([d_idx,d_idx], [np.amin(cp_vecs[:,-1]), np.amax(cp_vecs[:,-1])], 'k-')
                
            plt.show()
            
            # slice data
            x.append( cp_vecs )
            y.append( abnormalLabel[i] )
        
    return x, y


def get_single_cond_prob(d_idx, window_step, ml, abnormalData, param_dict, ref_idx=None):
    cp_vecs = None
    ## for j in xrange(d_idx-window_step, d_idx):
    for j in xrange(len(abnormalData[0])):
        if j<4: continue
        cp_vec = ml.conditional_prob2( abnormalData[:,:j+1]*\
                                      param_dict['HMM']['scale'])
        if cp_vecs is None: cp_vecs = cp_vec
        else: cp_vecs = np.vstack([ cp_vecs, cp_vec])
    return cp_vecs


def save_data_labels(data, labels, processed_data_path='./'):
    LOG_DIR = os.path.join(processed_data_path, 'tensorflow' )
    if os.path.isdir(LOG_DIR) is False:
        os.system('mkdir -p '+LOG_DIR)

    
    if len(np.shape(data)) > 2:
        n_features = np.shape(data)[0]
        n_samples  = np.shape(data)[1]
        n_length  = np.shape(data)[2]
        training_data   = copy.copy(data)
        training_data   = np.swapaxes(training_data, 0, 1).reshape\
          ((n_samples, n_features*n_length))
    else:
        training_data   = copy.copy(data)
    training_labels = copy.copy(labels)

    import csv
    ## tgt_csv = os.path.join(LOG_DIR, 'data.tsv')
    tgt_csv = './data.tsv'
    with open(tgt_csv, 'w') as csvfile:
        for row in training_data:
            string = None
            for col in row:
                if string is None:
                    string = str(col)
                else:
                    string += '\t'+str(col)

            csvfile.write(string+"\n")

    ## tgt_csv = os.path.join(LOG_DIR, 'labels.tsv')
    tgt_csv = './labels.tsv'
    with open(tgt_csv, 'w') as csvfile:
        for row in training_labels:
            csvfile.write(str(row)+"\n")
    
    os.system('mv *.tsv ~/Dropbox/HRL/')        

