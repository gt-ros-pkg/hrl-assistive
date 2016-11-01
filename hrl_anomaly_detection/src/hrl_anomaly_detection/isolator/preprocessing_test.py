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
import scipy
from scipy import stats
from joblib import Parallel, delayed

# sklearn
from sklearn import preprocessing


import hrl_lib.util as ut
from hrl_anomaly_detection.util import *
from hrl_anomaly_detection.util_viz import *
from hrl_anomaly_detection import data_manager as dm
from hrl_anomaly_detection import util as util
## import hrl_lib.circular_buffer as cb
import hrl_anomaly_detection.data_viz as dv

# private learner
from hrl_anomaly_detection.hmm import learning_hmm as hmm
import hrl_anomaly_detection.classifiers.classifier as cf

# visualization
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec
import itertools
colors = itertools.cycle(['g', 'm', 'c', 'k', 'y','r', 'b', ])
shapes = itertools.cycle(['x','v', 'o', '+'])

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42 
random.seed(3334)
np.random.seed(3334)

def evaluation_test(subject_names, task_name, raw_data_path, processed_data_path, param_dict,\
                    data_renew=False, save_pdf=False, verbose=False, debug=False,\
                    dim_viz=False,\
                    no_plot=False, delay_plot=True, find_param=False, data_gen=False):

    ## Parameters
    # data
    data_dict  = param_dict['data_param']
    data_renew = data_dict['renew']
    # AE
    AE_dict    = param_dict['AE']
    # HMM
    HMM_dict   = param_dict['HMM']
    nState     = HMM_dict['nState']
    cov        = HMM_dict['cov']
    add_logp_d = HMM_dict.get('add_logp_d', False)
    # SVM
    SVM_dict   = param_dict['SVM']

    # ROC
    ROC_dict = param_dict['ROC']


    #------------------------------------------
    if os.path.isdir(processed_data_path) is False:
        os.system('mkdir -p '+processed_data_path)

    crossVal_pkl = os.path.join(processed_data_path, 'cv_'+task_name+'.pkl')

    if os.path.isfile(crossVal_pkl) and data_renew is False and data_gen is False:
        print "CV data exists and no renew"
        d = ut.load_pickle(crossVal_pkl)
        kFold_list  = d['kFoldList']
        successData = d['successData']
        failureData = d['failureData']        
        success_isol_data = d['successIsolData']
        failure_isol_data = d['failureIsolData']        
        success_files = d['success_files']
        failure_files = d['failure_files']
    else:
        '''
        Use augmented data? if nAugment is 0, then aug_successData = successData
        '''
        # Get a data set with a leave-one-person-out
        print "Extract data using getDataLOPO"
        d = dm.getDataLOPO(subject_names, task_name, raw_data_path, \
                           processed_data_path, data_dict['rf_center'], data_dict['local_range'],\
                           downSampleSize=data_dict['downSampleSize'],\
                           handFeatures=data_dict['handFeatures'], \
                           cut_data=data_dict['cut_data'], \
                           isolationFeatures=param_dict['data_param']['isolationFeatures'], \
                           data_renew=data_renew, max_time=data_dict['max_time'])
        successData, failureData, success_files, failure_files, kFold_list \
          = dm.LOPO_data_index(d['successDataList'], d['failureDataList'],\
                               d['successFileList'], d['failureFileList'])

        for i in xrange(len(subject_names)):
            if i==0:
                success_isol_data = d['successIsolDataList'][i]
                failure_isol_data = d['failureIsolDataList'][i]
            else:
                success_isol_data = np.vstack([ np.swapaxes(success_isol_data,0,1), \
                                                np.swapaxes(d['successIsolDataList'][i], 0,1)])
                failure_isol_data = np.vstack([ np.swapaxes(failure_isol_data,0,1), \
                                                np.swapaxes(d['failureIsolDataList'][i], 0,1)])
                success_isol_data = np.swapaxes(success_isol_data, 0, 1)
                failure_isol_data = np.swapaxes(failure_isol_data, 0, 1)

        d['successData']     = successData
        d['failureData']     = failureData
        d['successIsolData'] = success_isol_data
        d['failureIsolData'] = failure_isol_data
        d['success_files']   = success_files
        d['failure_files']   = failure_files
        d['kFoldList']       = kFold_list
        ut.save_pickle(d, crossVal_pkl)
        if data_gen: sys.exit()

    #-----------------------------------------------------------------------------------------
    # parameters
    ref_num      = 2
    window_size = [10,20]
    startIdx    = 4
    weight      = -4.9 #-5.5 
    method_list = ROC_dict['methods'] 
    nPoints     = ROC_dict['nPoints']

    param_dict2  = d['param_dict']
    if 'timeList' in param_dict2.keys():
        timeList    = param_dict2['timeList'][startIdx:]
    else: timeList = None
    handFeatureParams = d['param_dict']
    normalTrainData   = d['successData'] * HMM_dict['scale']

    # 0 1 2 3 45678910 111213 14 15 16 17
    ## success_isol_data = success_isol_data[[0,1,2,3,9,10,11,12,13,14,15]]
    ## failure_isol_data = failure_isol_data[[0,1,2,3,9,10,11,12,13,14,15]]
    ## success_isol_data = success_isol_data[[0,3,4,5,6,7,8,9,10,11,12,13,14,15]]
    ## failure_isol_data = failure_isol_data[[0,3,4,5,6,7,8,9,10,11,12,13,14,15]]

    #-----------------------------------------------------------------------------------------
    # HMM-induced vector with LOPO
    dm.saveHMMinducedFeatures(kFold_list, successData, failureData,\
                              task_name, processed_data_path,\
                              HMM_dict, data_renew, startIdx, nState, cov, HMM_dict['scale'], \
                              success_files=success_files, failure_files=failure_files,\
                              add_logp_d=add_logp_d, verbose=verbose)

    #-----------------------------------------------------------------------------------------
    # Training HMM, and getting classifier training and testing data
    for idx, (normalTrainIdx, abnormalTrainIdx, normalTestIdx, abnormalTestIdx) \
      in enumerate(kFold_list):

        if verbose: print idx, " : training hmm and getting classifier training and testing data"
        feature_pkl = os.path.join(processed_data_path, 'isol_'+task_name+'_'+str(idx)+'.pkl')
        if not (os.path.isfile(feature_pkl) is False or HMM_dict['renew'] or data_renew or \
                SVM_dict['renew']): continue

        modeling_pkl = os.path.join(processed_data_path, 'hmm_'+task_name+'_'+str(idx)+'.pkl')
        dd = ut.load_pickle(modeling_pkl)
        nEmissionDim = dd['nEmissionDim']
        ml  = hmm.learning_hmm(nState, nEmissionDim, verbose=verbose) 
        ml.set_hmm_object(dd['A'],dd['B'],dd['pi'])

        # dim x sample x length
        normalTrainData   = successData[:, normalTrainIdx, :]
        abnormalTrainData = failureData[:, abnormalTrainIdx, :]
        normalTestData    = successData[:, normalTestIdx, :] 
        abnormalTestData  = failureData[:, abnormalTestIdx, :]
        abnormal_train_files = np.array(failure_files)[abnormalTrainIdx].tolist()
        abnormal_test_files = np.array(failure_files)[abnormalTestIdx].tolist()

        #-----------------------------------------------------------------------------------------
        # Classifier train data
        #-----------------------------------------------------------------------------------------
        trainDataX = abnormalTrainData*HMM_dict['scale']
        trainDataY = []
        abnormalTrainIdxList  = []
        abnormalTrainFileList = []
        for i, f in enumerate(abnormal_train_files):
            if f.find("failure")>=0:
                trainDataY.append(1)
                abnormalTrainIdxList.append(i)
                abnormalTrainFileList.append(f.split('/')[-1])    

        detection_train_idx_list = anomaly_detection(trainDataX/HMM_dict['scale'], trainDataY, \
                                                     task_name, save_data_path, param_dict,\
                                                     logp_viz=False, verbose=False, weight=weight)

        #-----------------------------------------------------------------------------------------
        # Classifier test data
        #-----------------------------------------------------------------------------------------
        testDataX = abnormalTestData*HMM_dict['scale']

        testDataY = []
        normalIdxList  = []
        normalFileList = []
        abnormalTestIdxList  = []
        abnormalTestFileList = []
        for i, f in enumerate(abnormal_test_files):
            ## if f.find("success")>=0:
            ##     testDataY.append(-1)
            ##     normalIdxList.append(i)
            ##     normalFileList.append(f.split('/')[-1])
            if f.find("failure")>=0:
                testDataY.append(1)
                abnormalTestIdxList.append(i)
                abnormalTestFileList.append(f.split('/')[-1])    

        detection_test_idx_list = anomaly_detection(testDataX/HMM_dict['scale'], testDataY, \
                                                    task_name, save_data_path, param_dict,\
                                                    logp_viz=False, verbose=False, weight=weight)

        #-----------------------------------------------------------------------------------------
        # Expected output - it should be replaced.... using theoretical stuff
        #-----------------------------------------------------------------------------------------        
        # get delta values...
        normal_isol_train_data   = success_isol_data[:, normalTrainIdx, :] 
        abnormal_isol_train_data = failure_isol_data[:, abnormalTrainIdx, :] 
        normal_isol_test_data    = success_isol_data[:, normalTestIdx, :] 
        abnormal_isol_test_data  = failure_isol_data[:, abnormalTestIdx, :]

        # get individual HMM
        A        = dd['A']
        pi       = dd['pi']
        nEmissionDim = 2
        cov_mult = [cov/2.0]*(nEmissionDim**2)
        scale    = HMM_dict['scale']/2.0

        ml_dict = {}
        ref_data = normal_isol_train_data[0:1]
        tgt_data = normal_isol_train_data[1:]
        for i in xrange(len(tgt_data)):
            x = np.vstack([ref_data, tgt_data[i:i+1]])

            ml = hmm.learning_hmm(nState, nEmissionDim, verbose=verbose)
            ret = ml.fit( (x+np.random.normal(0.0, 0.05, np.shape(x)))*scale, \
                          cov_mult=cov_mult, use_pkl=False)
            if ret == 'Failure':
                print "fitting failed... ", i, ml.B
                sys.exit()
            ml_dict[i] = ml

        train_feature_list, train_anomaly_list = extractFeature(normal_isol_train_data, \
                                                                abnormal_isol_train_data, \
                                                                detection_train_idx_list, \
                                                                abnormalTrainFileList, \
                                                                window_size, hmm_model=ml_dict,\
                                                                scale=scale)
        test_feature_list, test_anomaly_list = extractFeature(normal_isol_train_data, \
                                                              abnormal_isol_test_data, \
                                                              detection_test_idx_list, \
                                                              abnormalTestFileList, \
                                                              window_size, hmm_model=ml_dict,\
                                                              scale=scale)

        d = {}
        d['train_feature_list'] = train_feature_list
        d['train_anomaly_list'] = train_anomaly_list
        d['test_feature_list']  = test_feature_list
        d['test_anomaly_list']  = test_anomaly_list

        feature_pkl = os.path.join(processed_data_path, 'isol_'+task_name+'_'+str(idx)+'.pkl')
        ut.save_pickle(d, feature_pkl)

    y_test = []
    y_pred = []
    scores = []   
    # Training HMM, and getting classifier training and testing data
    for idx in xrange(len(kFold_list)):

        feature_pkl = os.path.join(processed_data_path, 'isol_'+task_name+'_'+str(idx)+'.pkl')
        d = ut.load_pickle(feature_pkl)
        train_feature_list = d['train_feature_list']
        train_anomaly_list = d['train_anomaly_list'] 
        test_feature_list  = d['test_feature_list']  
        test_anomaly_list  = d['test_anomaly_list']  

        ## print np.shape(train_feature_list)
        ## sys.exit()
        # 0 : 1 2 3 45678910 111213 14 15 16 17  # = total 18
        # (i-1)*2, (i-1)*2+1
        ## remove_list = [1,2,4,5,12,13]
        remove_list = [1,4,13]
        out_list = []
        for i in remove_list:
            out_list.append( (i-1)*2 )
            out_list.append( (i-1)*2 + 1 )

        def feature_remove(x, out_list):
            x = np.swapaxes(x, 0,1).tolist()
            x = [x[i] for i in xrange(len(x)) if i not in out_list]
            x = np.swapaxes(x, 0,1)
            return x

        print np.shape(train_feature_list), np.shape(test_feature_list)
        train_feature_list = feature_remove(train_feature_list, out_list)
        test_feature_list = feature_remove(test_feature_list, out_list)
        print np.shape(train_feature_list), np.shape(test_feature_list)
                   
        # scaling
        scaler = preprocessing.StandardScaler()
        train_feature_list = scaler.fit_transform(train_feature_list)
        test_feature_list = scaler.transform(test_feature_list)
        
        #-----------------------------------------------------------------------------------------
        # Classification
        #-----------------------------------------------------------------------------------------            
        ## from sklearn.neighbors import KNeighborsClassifier
        ## clf = KNeighborsClassifier(n_neighbors=10, weights='distance')
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=400, n_jobs=-1)
        clf.fit(train_feature_list, train_anomaly_list)
        pred_anomaly_list = clf.predict(test_feature_list)
        score = clf.score(test_feature_list, test_anomaly_list)

        y_test += list(test_anomaly_list)
        y_pred += list(pred_anomaly_list)
        scores.append(score)
        print idx, "'s score : ", score
        
        #-----------------------------------------------------------------------------------------
        # Visualization
        #-----------------------------------------------------------------------------------------            
        if dim_viz:
            low_dim_viz(train_feature_list, train_anomaly_list, \
                        test_feature_list, test_anomaly_list)
            sys.exit()
            
        ## raw_data_viz(ml, nEmissionDim, nState, testDataX, testDataY, normalTestData,\
        ##              detection_idx_list, abnormalFileList, timeList,\
        ##              HMM_dict, \
        ##              startIdx=4, ref_num=2, window_size=window_size)

    print np.shape(y_test), np.shape(y_pred)
    print "Score: ", np.mean(scores), np.std(scores)

    if no_plot is False:
        class_names = np.unique(y_test)
        from sklearn.metrics import confusion_matrix
        cnf_matrix = confusion_matrix(y_test, y_pred, labels=class_names)
        np.set_printoptions(precision=2)

        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                                                title='Normalized confusion matrix')
        plt.show()




def anomaly_detection(X, Y, task_name, processed_data_path, param_dict, logp_viz=False, verbose=False,
                      weight=0.0):
    ''' Anomaly detector that return anomalous point on each data.
    '''
    HMM_dict = param_dict['HMM']
    SVM_dict = param_dict['SVM']
    ROC_dict = param_dict['ROC']
    
    # set parameters
    method  = 'hmmgp' #'progress'
    ## weights = ROC_dict[method+'_param_range']
    nMaxData   = 20 # The maximun number of executions to train GP
    nSubSample = 40 # The number of sub-samples from each execution to train GP

    # Load a generative model
    idx = 0
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
      hmm.getHMMinducedFeaturesFromRawCombinedFeatures(ml, X * HMM_dict['scale'], Y, startIdx)

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


def extractFeature(normal_data, abnormal_data, anomaly_idx_list, abnormal_file_list, window_size,\
                   hmm_model=None, scale=1.0, startIdx=4):

    if hmm_model is None:
        normal_mean = []
        for i in xrange(len(normal_data)):
            normal_mean.append(np.mean(normal_data[i],axis=0))
    ## else:        
    ##     (A,pi,nState,scale) = hmm_param

    ##     for i in xrange(len(normal_data)):
    ##         x = normal_data[i:i+1]+np.random.normal(0.0, 0.03, np.shape(normalTrainData[i:i+1]) )
    ##         x *= scale
            
    ##         ml  = hmm.learning_hmm(nState, 1, verbose=verbose)
    ##         ml.fit(x, A=A, pi=pi, fixed_trans=1)
    ##         if ret == 'Failure' or np.isnan(ret):
    ##             print "hmm training failed"
    ##             sys.exit()
            
    ref_num = 0
         

    anomaly_list = []
    feature_list = []
    for i in xrange(len(abnormal_data[0])): # per sample
        # Anomaly point
        anomaly_idx = anomaly_idx_list[i]
        if anomaly_idx is None:
            print "Failed to detect anomaly", abnormal_file_list[i]
            continue

        # for each feature
        features = []
        for j in xrange(len(abnormal_data)): # per feature

            if j == 0: continue
            if hmm_model is not None:

                start_idx = anomaly_idx-window_size[0]
                end_idx   = anomaly_idx+window_size[1]
                if start_idx < 0: start_idx = 0
                if end_idx >= len(abnormal_data[j][i]): end_idx = len(abnormal_data[j][i])-1
                
                ml            = hmm_model[j-1]
                single_window = []
                for k in xrange(start_idx, end_idx+1):
                    if k<startIdx:
                        x_pred = ml.B[0][0][1]
                    else:
                        x_pred = ml.predict_from_single_seq(abnormal_data[ref_num,i,:k+1]*scale, \
                                                            ref_num=ref_num)[1]
                    ## print np.shape(abnormal_data), j,i,k, (abnormal_data[j,i,k] - x_pred)/scale
                    single_window.append( (abnormal_data[j,i,k] - x_pred)/scale )
            else:
                single_data   = abnormal_data[j,i] - normal_mean[j]
                if anomaly_idx-window_size[0] <0: start_idx = 0
                else: start_idx = anomaly_idx-window_size[0]
                single_window = single_data[start_idx:anomaly_idx+window_size[1]+1]

            features += [np.mean(single_window), np.amax(single_window)-np.amin(single_window)]
        feature_list.append(features)
        tid = int(abnormal_file_list[i].split('_')[0])
        anomaly_list.append(tid)

    return feature_list, anomaly_list



def raw_data_viz(ml, nEmissionDim, nState, testDataX, testDataY, normalTestData,\
                 detection_idx_list, abnormalFileList, timeList,\
                 HMM_dict, \
                 startIdx=4, ref_num=2, window_size = [10,20]):

    normalMean = []
    normalStd  = []
    for i in xrange(nEmissionDim):
        normalMean.append(np.mean(normalTestData[i],axis=0))
        normalStd.append(np.std(normalTestData[i],axis=0))
                 
    x = range(len(normalMean[0])) # length

    targetDataX = testDataX #abnormalTestData
    targetDataY = testDataY
    print "Target Data: ", np.shape(targetDataX), np.shape(targetDataY)

    abnormal_windows = []
    abnormal_class   = []
    for i in xrange(len(targetDataX[0])): # per sample

        if targetDataY[i] < 0:
            print "Ignored negative data: ", i
            continue

        # Expected output (prediction from partial observation)
        mu       = []
        for j in xrange(len(x)): # per time sample
            if j < startIdx:
                mu.append(ml.B[0][0])
            else:
                ## x_pred = ml.predict_from_single_seq(exp_interp_traj[i][:j]*HMM_dict['scale'], \
                ##                                     ref_num=2)
                x_pred = ml.predict_from_single_seq(targetDataX[ref_num,i,:j], ref_num=2)
                mu.append(x_pred)
        mu  = np.array(mu)

        # Estimated progress of execution
        _,_,l_logp, l_post = hmm.computeLikelihoods(0, ml.A, ml.B, ml.pi, ml.F,\
                                                    [ targetDataX[j][i] for j in xrange(nEmissionDim)],\
                                                    nEmissionDim, nState,\
                                                    startIdx=startIdx,\
                                                    bPosterior=True)
        max_idx = np.argmax(l_post, axis=1)
        max_idx = [max_idx[0]]*startIdx+max_idx.tolist()

        # Anomaly point
        anomaly_idx = detection_idx_list[i]
        if anomaly_idx is not None:

            # mean, range, slope
            abnormal_window = []
            for k in xrange(nEmissionDim):
                single_data   = (targetDataX[k,i]-mu[:,k])/HMM_dict['scale']
                single_window = single_data[anomaly_idx-window_size[0]:anomaly_idx+window_size[1]+1]
                ## slope,_,_,_,_ = stats.linregress(range(len(single_window)), single_window)
                abnormal_window += [np.mean(single_window),\
                                    np.amax(single_window)-np.amin(single_window)] 
            abnormal_windows.append(abnormal_window)
            tid = int(abnormalFileList[i].split('_')[0])
            abnormal_class.append(tid)
            ## for kk in xrange(len(classes)):
            ##     if tid in classes[kk]:
            ##         abnormal_class.append(kk)
            ##         break

        lim_list = []

        fig = plt.figure(1)
        for k in xrange(nEmissionDim):
            lim_list.append([ np.amin(normalMean[k]-1.0*normalStd[k])*0.75, np.amax(normalMean[k]+1.0*normalStd[k])*1.5 ])

            ax = fig.add_subplot(nEmissionDim*100+20+k*2+1)
            ax.fill_between(x, normalMean[k]-1.0*normalStd[k], \
                            normalMean[k]+1.0*normalStd[k], \
                            facecolor='green', alpha=0.3)
            ax.plot(x, targetDataX[k,i]/HMM_dict['scale'], 'r-')
            ax.plot(x, mu[:,k]/HMM_dict['scale'], 'b-')
            if anomaly_idx is not None:
                ax.plot([anomaly_idx, anomaly_idx], [0.0,2.0], 'm-')

            if k == 0:
                for l in xrange(len(x)):
                    if l%5==0:
                        ax.text(x[l], 0.4, str(max_idx[l]+1) )

            n      = len(x)
            xx     = [0, n/2, n-1]
            labels = [int(timeList[0]), int(timeList[n/2]), int(timeList[-1])]
            ax.set_xticks(xx)
            ax.set_xticklabels(labels)


            ax = fig.add_subplot(nEmissionDim*100+20+k*2+2)
            ax.plot(x, (targetDataX[k,i]-mu[:,k])/HMM_dict['scale'],'r-')
            if anomaly_idx is not None:
                ax.plot([anomaly_idx, anomaly_idx], [-2.0,2.0], 'm-')

                min_idx = anomaly_idx-window_size[0]
                max_idx = anomaly_idx+window_size[1]
                if min_idx <0 : min_idx = 0
                ax.plot([min_idx, min_idx], [-2.0,2.0], 'm-')
                ax.plot([max_idx, max_idx], [-2.0,2.0], 'm-')


        for k in xrange(nEmissionDim):
            ax = fig.add_subplot(nEmissionDim*100+20+k*2+1)
            ax.set_ylim(lim_list[k])
            ## if k==0: ax.set_ylim([-0.1,0.4])
            ## elif k==1: ax.set_ylim([0.15,0.4])
            ## elif k==2: ax.set_ylim([0.1,0.75])
            ## else: ax.set_ylim([0.1,0.7])
            ax = fig.add_subplot(nEmissionDim*100+20+k*2+2)
            ax.set_ylim([-0.2,0.2])

        plt.suptitle(abnormalFileList[i], fontsize=18)
        plt.show()


def low_dim_viz(x_train, y_train, x_test=None, y_test=None ):

    ## (x_train,y_train) = xy_train

    ## print np.shape(feature_list), np.shape(anomaly_list)
    from sklearn.manifold import TSNE
    model = TSNE(n_components=2, random_state=0, perplexity=17) #, init='pca')
    x = model.fit_transform(x_train,y_train)
    ## test_x_new = model.transform(test_feature_list)
    y_uni = np.unique(y_train)
    y     = y_train
    
    #colors = ['g', 'm', 'c', 'k', 'y','r', 'b', ]
    markers = ['x','v', 'o', '+', 'D', 'H', 's', '*', '1', '3', '8', '^', 'd']
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, len(y_uni))]

    c = []
    m = []
    for i in xrange(len(y)):
        j = y_uni.tolist().index(y[i])
        c.append( colors[j] )
        m.append( markers[j] )

    fig = plt.figure(1)
    for _x, _y, _c, _m in zip(x[:,0], x[:,1], c, m):
        plt.scatter(_x,_y, c=_c, marker=_m, cmap=plt.cm.Spectral, s=80)


    ## if x_test is not None:
    ##     (x_test, y_test) = xy_test    

    ##     c = []
    ##     m = []
    ##     for i in xrange(len(y_test)):
    ##         j = y_uni.tolist().index(y[i])
    ##         c.append( colors[j] )
    ##         m.append( markers[j] )

    ##     for _x, _y, _c, _m in zip(x_test[:,0], x_test[:,1], c, m):
    ##         plt.scatter(_x,_y, c=_c, marker=_m, cmap=plt.cm.Spectral, s=160)

            
    plt.axis('tight')
    plt.show()



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    ## thresh = cm.max() / 2.
    ## for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    ##     plt.text(j, i, cm[i, j],
    ##              horizontalalignment="center",
    ##              color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    util.initialiseOptParser(p)

    p.add_option('--low_dim_viz', '--lv', action='store_true', dest='low_dim_viz',
                 default=False, help='Plot low-dimensional embedding.')
    
    opt, args = p.parse_args()
    
    #---------------------------------------------------------------------------           
    # Run evaluation
    #---------------------------------------------------------------------------           
    rf_center   = 'kinEEPos'        
    scale       = 1.0
    local_range = 10.0    

    from hrl_anomaly_detection.AURO2016_params import *
    raw_data_path, save_data_path, param_dict = getParams(opt.task, opt.bDataRenew, \
                                                          opt.bHMMRenew, opt.bClassifierRenew, opt.dim,\
                                                          rf_center, local_range)
    save_data_path = os.path.expanduser('~')+\
      '/hrl_file_server/dpark_data/anomaly/AURO2016/'+opt.task+'_data_isolation/'+\
      str(param_dict['data_param']['downSampleSize'])+'_'+str(opt.dim)
                                                          
    #---------------------------------------------------------------------------
    if opt.task == 'scooping':
        subjects = ['park', 'test'] #'Henry', 
    #---------------------------------------------------------------------------
    elif opt.task == 'feeding':
        subjects = ['s2', 's3','s4','s5', 's6','s7','s8', 's9']
    elif opt.task == 'pushing':
        subjects = ['microblack', 'microwhite']        
    else:
        print "Selected task name is not available."
        sys.exit()
                                                          
    #---------------------------------------------------------------------------           
    if opt.bRawDataPlot or opt.bInterpDataPlot:
        '''
        Before localization: Raw data plot
        After localization: Raw or interpolated data plot
        '''
        successData = True
        failureData = True
        modality_list   = ['kinematics', 'audio', 'ft', 'vision_landmark'] # raw plot

        dv.data_plot(subjects, opt.task, raw_data_path, save_data_path,\
                  downSampleSize=param_dict['data_param']['downSampleSize'], \
                  local_range=local_range, rf_center=rf_center, global_data=True,\
                  raw_viz=opt.bRawDataPlot, interp_viz=opt.bInterpDataPlot, save_pdf=opt.bSavePdf,\
                  successData=successData, failureData=failureData,\
                  modality_list=modality_list, data_renew=opt.bDataRenew, verbose=opt.bVerbose)

    elif opt.bDataSelection:
        '''
        Manually select and filter bad data out
        '''
        ## modality_list   = ['kinematics', 'audioWrist','audio', 'fabric', 'ft', \
        ##                    'vision_artag', 'vision_change', 'pps']
        modality_list   = ['kinematics', 'ft']
        success_viz = True
        failure_viz = True

        data_selection(subjects, opt.task, raw_data_path, save_data_path,\
                       downSampleSize=param_dict['data_param']['downSampleSize'], \
                       local_range=local_range, rf_center=rf_center, \
                       success_viz=success_viz, failure_viz=failure_viz,\
                       raw_viz=opt.bRawDataPlot, save_pdf=opt.bSavePdf,\
                       modality_list=modality_list, data_renew=opt.bDataRenew, \
                       max_time=param_dict['data_param']['max_time'], verbose=opt.bVerbose)        

    elif opt.bFeaturePlot:
        success_viz = True
        failure_viz = False
        
        ## save_data_path = os.path.expanduser('~')+\
        ##   '/hrl_file_server/dpark_data/anomaly/ICRA2017/'+opt.task+'_data_online/'+\
        ##   str(param_dict['data_param']['downSampleSize'])+'_'+str(opt.dim)
        dm.getDataLOPO(subjects, opt.task, raw_data_path, save_data_path,
                       param_dict['data_param']['rf_center'], param_dict['data_param']['local_range'],\
                       downSampleSize=param_dict['data_param']['downSampleSize'], scale=scale, \
                       success_viz=success_viz, failure_viz=failure_viz,\
                       ae_data=False,\
                       cut_data=param_dict['data_param']['cut_data'],\
                       save_pdf=opt.bSavePdf, solid_color=True,\
                       handFeatures=param_dict['data_param']['handFeatures'], data_renew=opt.bDataRenew, \
                       isolationFeatures=param_dict['data_param']['isolationFeatures'], isolation_viz=True,
                       max_time=param_dict['data_param']['max_time'])

    elif opt.HMM_param_search:

        from hrl_anomaly_detection.hmm import run_hmm_cpy as hmm_opt
        parameters = {'nState': [20, 25], 'scale': np.linspace(3.0,15.0,10), \
                      'cov': np.linspace(0.5,10.0,5) }
        max_check_fold = 1 #None
        no_cov = False
        
        hmm_opt.tune_hmm(parameters, d, param_dict, save_data_path, verbose=True, n_jobs=opt.n_jobs, \
                         bSave=opt.bSave, method=opt.method, max_check_fold=max_check_fold, no_cov=no_cov)

    elif opt.CLF_param_search:
        from hrl_anomaly_detection.classifiers import opt_classifier as clf_opt
        method = 'hmmgp'
        clf_opt.tune_classifier(save_data_path, opt.task, method, param_dict, n_jobs=-1, n_iter_search=1000)
                         
    else:
        if opt.bHMMRenew: param_dict['ROC']['methods'] = ['fixed', 'progress'] 
        if opt.bNoUpdate: param_dict['ROC']['update_list'] = []
                    
        evaluation_test(subjects, opt.task, raw_data_path, save_data_path, param_dict, \
                        save_pdf=opt.bSavePdf, dim_viz=opt.low_dim_viz,\
                        verbose=opt.bVerbose, debug=opt.bDebug, no_plot=opt.bNoPlot, \
                        find_param=False, data_gen=opt.bDataGen)


        ## print np.shape(abnormal_windows)
        ## print abnormal_windows
        
        ## pca_gamma=5.0
        ## from sklearn.decomposition import KernelPCA
        ## ml = KernelPCA(n_components=2, kernel="rbf", fit_inverse_transform=False, \
        ##                gamma=pca_gamma)
        ## X_scaled = ml.fit_transform(abnormal_windows)
        ## fig = plt.figure(2)
        ## for kk in xrange(len(classes)):
        ##     color = colors.next()
        ##     shape = shapes.next()
            
        ##     idx_list = [nn for nn, c in enumerate(abnormal_class) if c == kk ]
        ##     xy_data = X_scaled[idx_list]
        ##     print np.shape(xy_data), color, shape
        ##     print idx_list
        ##     plt.scatter(xy_data[:,0], xy_data[:,1], c=color, marker=shape, label=str(kk))
        ## plt.legend(loc='lower left', prop={'size':12})            
        ## plt.show()

            
