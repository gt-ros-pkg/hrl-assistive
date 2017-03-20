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
## import rospy, roslib
import os, sys, copy
import random
import socket

# visualization
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec

# util
import numpy as np
import scipy
import hrl_lib.util as ut
from hrl_anomaly_detection.util import *
from hrl_anomaly_detection.util_viz import *
from joblib import Parallel, delayed

# learning
from hrl_anomaly_detection.feature_extractors import train as ae
from hrl_anomaly_detection.feature_extractors import theano_util as tu
from hrl_anomaly_detection.hmm import learning_hmm as hmm
import hrl_anomaly_detection.classifiers.classifier as cf
from sklearn import svm

# data
from mvpa2.datasets.base import Dataset
from hrl_anomaly_detection import data_manager as dm

import warnings



def getLikelihoods(normTrainFeatures, abnormTrainFeatures, normTestFeatures, abnormTestFeatures, \
                   nState=10, scale=1.0, \
                   useNormTrain=True, useNormTest=False, useAbnormTest=True,\
                   useNormTrain_color=False, useNormTest_color=False, useAbnormTest_color=False,\
                   hmm_param_pkl=None, save_pdf=False):
    '''
    Visualize likelihood distribution
    '''
    warnings.simplefilter("always", DeprecationWarning)

    print "======================================"
    print "Training data: ", np.shape(normTrainFeatures)
    print "======================================"
    
    # training hmm
    nEmissionDim = len(normTrainFeatures)
    cov_mult = [1.0]*(nEmissionDim**2)

    ml  = hmm.learning_hmm(nState, nEmissionDim, verbose=False)
    ret = ml.fit(normTrainFeatures, cov_mult=cov_mult, ml_pkl=hmm_param_pkl, use_pkl=False) # not(renew))
    
    if ret == 'Failure': 
        print "-------------------------"
        print "HMM returned failure!!   "
        print "-------------------------"
        return (-1,-1,-1,-1)


    fig = plt.figure()
    min_logp = 0.0
    max_logp = 0.0

    # training data
    if useNormTrain:

        log_ll = []
        ## exp_log_ll = []        
        for i in xrange(len(normTrainFeatures[0])):

            log_ll.append([])
            ## exp_log_ll.append([])
            for j in range(2, len(normTrainFeatures[0][i])):

                X = [x[i,:j] for x in normTrainFeatures]
                logp = ml.loglikelihood(X)
                log_ll[i].append(logp)

            if min_logp > np.amin(log_ll): min_logp = np.amin(log_ll)
            if max_logp < np.amax(log_ll): max_logp = np.amax(log_ll)
                
            # disp
            if useNormTrain_color: plt.plot(log_ll[i], label=str(i))
            else: plt.plot(log_ll[i], 'b-')

        if useNormTrain_color: 
            plt.legend(loc=3,prop={'size':16})
            
        ## plt.plot(log_ll[i], 'b-', lw=3.0)
        ## plt.plot(exp_log_ll[i], 'm-')            


    # training data
    if useNormTest:

        log_ll = []
        ## exp_log_ll = []        
        for i in xrange(len(normTestFeatures[0])):

            log_ll.append([])
            ## exp_log_ll.append([])
            for j in range(2, len(normTestFeatures[0][i])):

                X = [x[i,:j] for x in normTestFeatures]
                logp = ml.loglikelihood(X)
                log_ll[i].append(logp)

            if min_logp > np.amin(log_ll): min_logp = np.amin(log_ll)
            if max_logp < np.amax(log_ll): max_logp = np.amax(log_ll)
                
            # disp
            if useNormTest_color: plt.plot(log_ll[i], label=str(i))
            else: plt.plot(log_ll[i], 'g-')

        if useNormTest_color: 
            plt.legend(loc=3,prop={'size':16})
            
        ## plt.plot(log_ll[i], 'b-', lw=3.0)
        ## plt.plot(exp_log_ll[i], 'm-')            


    # training data
    if useAbnormTest:

        log_ll = []
        ## exp_log_ll = []        
        for i in xrange(len(abnormTestFeatures[0])):

            log_ll.append([])
            ## exp_log_ll.append([])
            for j in range(2, len(abnormTestFeatures[0][i])):

                X = [x[i,:j] for x in abnormTestFeatures]
                logp = ml.loglikelihood(X)
                log_ll[i].append(logp)

            if min_logp > np.amin(log_ll): min_logp = np.amin(log_ll)
            if max_logp < np.amax(log_ll): max_logp = np.amax(log_ll)
                
            # disp
            if useAbnormTest_color: plt.plot(log_ll[i], label=str(i))
            else: plt.plot(log_ll[i], 'r-')

        if useAbnormTest_color: 
            plt.legend(loc=3,prop={'size':16})
            
        ## plt.plot(log_ll[i], 'r-', lw=3.0)
        ## plt.plot(exp_log_ll[i], 'm-')            







    plt.ylim([min_logp, max_logp])
    if save_pdf == True:
        fig.savefig('test.pdf')
        fig.savefig('test.png')
        os.system('cp test.p* ~/Dropbox/HRL/')
    else:
        plt.show()        

    return



def Run_AE_HMM_SVM(X_normalTrain, X_abnormalTrain, X_normalTest, X_abnormalTest,\
                   AE_model, AE_data_pkl, HMM_data_pkl, \
                   nState=10, scale=1.0, hmm_param_pkl=None, save_pdf=False):
    '''
    Input: dimension x sample x length
    '''
    warnings.simplefilter("always", DeprecationWarning)

    print "Raw Data: ", np.shape(X_normalTrain)

    # Auto encoder
    if os.path.isfile(AE_data_pkl):
        d = ut.load_pickle(AE_data_pkl)
        normTrainFeatures   = d['normTrainFeatures']   
        abnormTrainFeatures = d['abnormTrainFeatures'] 
        normTestFeatures    = d['normTestFeatures']   
        abnormTestFeatures  = d['abnormTestFeatures'] 
    else:
        normTrainFeatures   = tu.RunAutoEncoder(X_normalTrain, AE_model)
        abnormTrainFeatures = tu.RunAutoEncoder(X_abnormalTrain, AE_model)
        normTestFeatures    = tu.RunAutoEncoder(X_normalTest, AE_model)
        abnormTestFeatures  = tu.RunAutoEncoder(X_abnormalTest, AE_model)

        d = {}
        d['normTrainFeatures']   = normTrainFeatures
        d['abnormTrainFeatures'] = abnormTrainFeatures
        d['normTestFeatures']   = normTestFeatures
        d['abnormTestFeatures'] = abnormTestFeatures
        ut.save_pickle(d, AE_data_pkl)

    print "AE data: ", np.shape(normTrainFeatures)

    # scaling
    normTrainFeatures   = np.array(normTrainFeatures) * scale
    abnormTrainFeatures = np.array(abnormTrainFeatures) * scale
    normTestFeatures    = np.array(normTestFeatures) * scale
    abnormTestFeatures  = np.array(abnormTestFeatures) * scale

    # Train HMM        
    if os.path.isfile(HMM_data_pkl):
        d = ut.load_pickle(HMM_data_pkl)
        normTrainHMMFeatures   = d['normTrainHMMFeatures']
        abnormTrainHMMFeatures = d['abnormTrainHMMFeatures']
        normTestHMMFeatures    = d['normTestHMMFeatures']     
        abnormTestHMMFeatures  = d['abnormTestHMMFeatures']
        nEmissionDim = len(normTrainHMMFeatures[0][0])
        A  = d['A']
        B  = d['B']
        pi = d['pi']
        ml = hmm.learning_hmm(nState, nEmissionDim=nEmissionDim)
        ml.set_hmm_object(A,B,pi)        
        
    else:        
        nEmissionDim = len(normTrainFeatures)
        cov_mult = [1.0]*(nEmissionDim**2)

        ml  = hmm.learning_hmm(nState, nEmissionDim, verbose=False)
        ret = ml.fit(normTrainFeatures, cov_mult=cov_mult, ml_pkl=hmm_param_pkl, use_pkl=False) # not(renew))

        if ret == 'Failure': 
            print "-------------------------"
            print "HMM returned failure!!   "
            print "-------------------------"
            return (-1,-1,-1,-1)

        # Get HMM-induced features
        normTrainHMMFeatures   = getHMMFeatures(ml, normTrainFeatures)
        abnormTrainHMMFeatures = getHMMFeatures(ml, abnormTrainFeatures)
        normTestHMMFeatures    = getHMMFeatures(ml, normTestFeatures)
        abnormTestHMMFeatures  = getHMMFeatures(ml, abnormTestFeatures)

        d = {}
        d['A']                      = ml.A
        d['B']                      = ml.B
        d['pi']                     = ml.pi
        
        d['normTrainHMMFeatures']   = normTrainHMMFeatures
        d['abnormTrainHMMFeatures'] = abnormTrainHMMFeatures
        d['normTestHMMFeatures']    = normTestHMMFeatures
        d['abnormTestHMMFeatures']  = abnormTestHMMFeatures
        ut.save_pickle(d, HMM_data_pkl)

    print "HMM features: ", np.shape(normTrainHMMFeatures)
    
    # stacking train data
    new_normTrainHMMFeatures = []
    for i in xrange(len(normTrainHMMFeatures)):
        for j in xrange(len(normTrainHMMFeatures[i])):
            new_normTrainHMMFeatures.append( normTrainHMMFeatures[i][j] )
    new_abnormTrainHMMFeatures = []
    for i in xrange(len(abnormTrainHMMFeatures)):
        for j in xrange(len(abnormTrainHMMFeatures[i])):
            new_abnormTrainHMMFeatures.append( abnormTrainHMMFeatures[i][j] )

    ############################## Classifier ##################################
    from hrl_anomaly_detection.classifiers import classifier as cb
    dtc = cb.classifier( ml, method='svm', nPosteriors=nState, nLength=len(new_normTrainHMMFeatures[0]) )        
    dtc.set_params( class_weight=0.4 )

    from sklearn import preprocessing
    scaler = preprocessing.StandardScaler()

    train_X      = np.vstack([new_normTrainHMMFeatures, new_abnormTrainHMMFeatures])
    train_Y      = [0]*len(new_normTrainHMMFeatures) + [1]*len(new_abnormTrainHMMFeatures)
    train_scaled_x = scaler.fit_transform(train_X)

    #
    ret = dtc.fit(train_scaled_x, train_Y)
    
    tp_l = []
    fp_l = []
    tn_l = []
    fn_l = []

    ## 
    for i in xrange(len(normTestHMMFeatures)):
        for j in xrange(len(normTestHMMFeatures[i])):
            X     = scaler.transform([normTestHMMFeatures[i][j]])
            est_y = dtc.predict(X, [0])

            if type(est_y) == list: est_y = est_y[0]
            if type(est_y) == list: est_y = est_y[0]

            if est_y > 0.0:
                print "Break ", i, j
                break

        if est_y > 0:
            fn_l.append(1)
        else:
            tn_l.append(1)
            
    for i in xrange(len(abnormTestHMMFeatures)):
        for j in xrange(len(abnormTestHMMFeatures[i])):
            X     = scaler.transform([abnormTestHMMFeatures[i][j]])
            est_y = dtc.predict(X, [1])

            if type(est_y) == list: est_y = est_y[0]
            if type(est_y) == list: est_y = est_y[0]

            if est_y > 0.0:
                print "Break ", i, j
                break

        if est_y > 0:
            tn_l.append(1)
        else:
            fn_l.append(1)

    print "tpr = ", float(np.sum(tp_l))/float(np.sum(tp_l)+np.sum(fn_l))
    print "fpr = ", float(np.sum(fp_l))/float(np.sum(fp_l)+np.sum(tn_l))
   
    return 


def getHMMFeatures(ml, X, startIdx=4, cpu_mode=False):
    '''
    Input: dimension x sample x length
    '''
    warnings.simplefilter("always", DeprecationWarning)

    nEmissionDim = len(X)

    # generate feature vectors for disriminative classifiers
    if cpu_mode:
        ll_X = []
        for i in xrange(len(X[0])):
            print i
            _, idx, logp, post = hmm.computeLikelihoods(i, ml.A, ml.B, ml.pi, ml.F, \
                                                        [ X[j][i] for j in xrange(nEmissionDim) ], \
                                                        ml.nEmissionDim, ml.nState, startIdx=startIdx, \
                                                        bPosterior=True)
            l_X = []
            for j in xrange(len(logp)):
                l_X.append( [logp[j]] + post[j].tolist() )
            ll_X.append(l_X)
                
                                                        
    else:
        # Parallel processing does not work with Theano and CUDA combination
        r = Parallel(n_jobs=-1)(delayed(hmm.computeLikelihoods)(i, ml.A, ml.B, ml.pi, ml.F, \
                                                                [ X[j][i] for j in xrange(nEmissionDim) ], \
                                                                ml.nEmissionDim, ml.nState, startIdx=startIdx, \
                                                                bPosterior=True)
                                                                for i in xrange(len(X[0])) )
        _, ll_idx, ll_logp, ll_post = zip(*r)

        ll_X = []
        for i in xrange(len(ll_logp)):
            l_X = []
            for j in xrange(len(ll_logp[i])):        
                l_X.append( [ll_logp[i][j]] + ll_post[i][j].tolist() )
            ll_X.append(l_X)
        
    return ll_X
        

if __name__ == '__main__':
    warnings.simplefilter("always", DeprecationWarning)

    import optparse
    p = optparse.OptionParser()
    p.add_option('--dataRenew', '--dr', action='store_true', dest='bDataRenew',
                 default=False, help='Renew pickle files.')
    ## p.add_option('--aeRenew', '--ar', action='store_true', dest='bAERenew',
    ##              default=False, help='Renew autoencoder parameters. -- not available?')
    p.add_option('--hmmRenew', '--hr', action='store_true', dest='bHMMRenew',
                 default=False, help='Renew HMM parameters.')
    p.add_option('--cfRenew', '--cr', action='store_true', dest='bCFRenew',
                 default=False, help='Renew classifier parameters.')

    p.add_option('--ae_train', '--at', action='store_true', dest='bTrainAutoEncoder',
                 default=False, help='Train AutoEncoderOnly.')
    p.add_option('--ae_run', '--ar', action='store_true', dest='bRunAutoEncoder',
                 default=False, help='Run AutoEncoderOnly.')
    p.add_option('--hmm', action='store_true', dest='bRunHMM',
                 default=False, help='Run AutoEncoder and HMM.')
    p.add_option('--svm', action='store_true', dest='bRunSVM',
                 default=False, help='Run AutoEncoder, HMM, and SVM.')

    p.add_option('--viz', action='store_true', dest='bViz',
                 default=False, help='Visualize data.')
    p.add_option('--savepdf', '--sp', action='store_true', dest='bSavePdf',
                 default=False, help='Save pdf files.')    
    p.add_option('--verbose', '--v', action='store_true', dest='bVerbose',
                 default=False, help='Print out.')

    opt, args = p.parse_args()



    subject_names       = ['gatsbii']
    task                = 'pushing'
    raw_data_path       = os.path.expanduser('~')+'/hrl_file_server/dpark_data/anomaly/RSS2016/'    
    processed_data_path = os.path.expanduser('~')+'/hrl_file_server/dpark_data/anomaly/RSS2016/'+task+'_data'
    save_pkl            = os.path.join(processed_data_path, 'ae_data.pkl')
    ## h5py_file           = os.path.join(processed_data_path, 'test.h5py')
    rf_center           = 'kinEEPos'
    local_range         = 1.25    
    nSet                = 1
    downSampleSize      = 200
    nAugment            = 4
    time_window         = 4
    ## layers              = [256,64,16]
    nState        = 10
    scale         = 10.0

    feature_list = ['relativePose_artag_EE', \
                    'relativePose_artag_artag', \
                    ## 'kinectAudio',\
                    'wristAudio', \
                    'ft', \
                    ## 'pps', \
                    ## 'visionChange', \
                    ## 'fabricSkin', \
                    ]

    AE_model      = '/home/dpark/catkin_ws/src/hrl-assistive/hrl_anomaly_detection/src/hrl_anomaly_detection/feature_extractors/simple_model.pkl'
    AE_processed_pkl = os.path.join(processed_data_path, 'ae_processed_data.pkl')
    HMM_processed_pkl = os.path.join(processed_data_path, 'hmm_processed_data.pkl')
                    

    # Load data
    X_normalTrain, X_abnormalTrain, X_normalTest, X_abnormalTest, nSingleData \
      = dm.get_time_window_data(subject_names, task, raw_data_path, processed_data_path, save_pkl, \
                                rf_center, local_range, downSampleSize, time_window, feature_list, \
                                nAugment, renew=opt.bDataRenew)


    if opt.bTrainAutoEncoder:
        print "Under construction"

    elif opt.bRunAutoEncoder:
        # Load AE
        normTrainFeatures = tu.RunAutoEncoder(X_normalTrain, AE_model, opt.bViz)

        ## X_train = np.vstack([X_normalTrain, X_abnormalTrain])
        ## X_test  = np.vstack([X_normalTest, X_abnormalTest])
        ## trainFeatures = RunAutoEncoder(X_train, model, opt.bViz)
        ## testFeatures  = RunAutoEncoder(X_test, model, opt.bViz)

    elif opt.bRunHMM:        
        normTrainFeatures   = tu.RunAutoEncoder(X_normalTrain, AE_model, opt.bViz)
        abnormTrainFeatures = tu.RunAutoEncoder(X_abnormalTrain, AE_model, opt.bViz)
        normTestFeatures    = tu.RunAutoEncoder(X_normalTest, AE_model, opt.bViz)
        abnormTestFeatures  = tu.RunAutoEncoder(X_abnormalTest, AE_model, opt.bViz)

        hmm_param_pkl = os.path.join(processed_data_path, 'hmm_'+task+'.pkl')
        
        getLikelihoods(normTrainFeatures, abnormTrainFeatures, normTestFeatures, abnormTestFeatures, \
                       nState=nState, scale=scale, \
                       useNormTrain=True, useNormTest=True, useAbnormTest=True, \
                       useNormTrain_color=False, useNormTest_color=False, useAbnormTest_color=False,\
                       hmm_param_pkl=hmm_param_pkl, save_pdf=opt.bSavePdf)
        ## trainHMMFeatures = RunHMM(nTrainFeatures, bTrain=False, opt.bViz)

    elif opt.bRunSVM:
        
        # HMM-SVM
        Run_AE_HMM_SVM(X_normalTrain, X_abnormalTrain, X_normalTest, X_abnormalTest, \
                       AE_model, AE_processed_pkl, HMM_processed_pkl, \
                       nState=nState, scale=scale, hmm_param_pkl=None, save_pdf=False)








    ## if opt.bRunSVM:
    ##     RunSVM(trainHMMFeatures, bTrain=True, opt.bViz)
    ##     RunSVM()
        
