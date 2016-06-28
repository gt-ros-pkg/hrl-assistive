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
import os, sys, copy
import random
import warnings

import matplotlib
## matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec

# util
import numpy as np
import scipy
import hrl_lib.util as ut
import hrl_lib.quaternion as qt
from hrl_anomaly_detection import util

from mvpa2.datasets.base import Dataset
from mvpa2.generators.partition import NFoldPartitioner
from mvpa2.generators import splitters

from sklearn import cross_validation
from sklearn.externals import joblib

import matplotlib.pyplot as plt

def create_mvpa_dataset(aXData, chunks, labels):
    data = Dataset(samples=aXData)
    data.sa['id']      = range(0,len(labels))
    data.sa['chunks']  = chunks
    data.sa['targets'] = labels

    return data

def kFold_data_index(nAbnormal, nNormal, nAbnormalFold, nNormalFold):

    normal_folds   = cross_validation.KFold(nNormal, n_folds=nNormalFold, shuffle=True)
    abnormal_folds = cross_validation.KFold(nAbnormal, n_folds=nAbnormalFold, shuffle=True)

    kFold_list = []

    for normal_temp_fold, normal_test_fold in normal_folds:

        normal_dc_fold = cross_validation.KFold(len(normal_temp_fold), \
                                                n_folds=nNormalFold-1, shuffle=True)
        for normal_train_fold, normal_classifier_fold in normal_dc_fold:

            normal_d_fold = normal_temp_fold[normal_train_fold]
            normal_c_fold = normal_temp_fold[normal_classifier_fold]

            for abnormal_c_fold, abnormal_test_fold in abnormal_folds:
                '''
                Normal training data for model
                Normal training data 
                Abnormal training data 
                Normal test data 
                Abnormal test data 
                '''
                index_list = [normal_d_fold, normal_c_fold, abnormal_c_fold, \
                              normal_test_fold, abnormal_test_fold]
                kFold_list.append(index_list)

    return kFold_list

def kFold_data_index2(nNormal, nAbnormal, nNormalFold, nAbnormalFold ):
    '''
    Output:
    Normal training data 
    Abnormal training data 
    Normal test data 
    Abnormal test data 
    '''

    normal_folds   = cross_validation.KFold(nNormal, n_folds=nNormalFold, shuffle=True)
    abnormal_folds = cross_validation.KFold(nAbnormal, n_folds=nAbnormalFold, shuffle=True)

    kFold_list = []

    for normal_train_fold, normal_test_fold in normal_folds:

        for abnormal_train_fold, abnormal_test_fold in abnormal_folds:
            index_list = [normal_train_fold, abnormal_train_fold, \
                          normal_test_fold, abnormal_test_fold]
            kFold_list.append(index_list)

    return kFold_list




#-------------------------------------------------------------------------------------------------
def getDataList(fileNames, rf_center, local_range, param_dict, downSampleSize=200, \
                cut_data=None, \
                handFeatures=['crossmodal_targetEEDist']):
    '''
    Load a list of files and return hand-engineered feature set of each file.
    The feature is scaled by the param dict.
    '''
    
    for fileName in fileName:
        if os.path.isfile(fileName) is False:
            print "Error>> there is no recorded file: ", fileName
            sys.exit()

    _, data_dict = util.loadData(fileNames, isTrainingData=False,
                                 downSampleSize=downSampleSize,
                                 local_range=local_range, rf_center=rf_center)
   
    features, _ = extractHandFeature(data_dict, handFeatures, \
                                     param_dict=param_dict, cut_data=cut_data)

    return features


def getDataSet(subject_names, task_name, raw_data_path, processed_data_path, rf_center, \
               local_range, \
               downSampleSize=200, scale=1.0, ae_data=False, \
               cut_data=None, init_param_dict=None, \
               success_viz=False, failure_viz=False, \
               save_pdf=False, solid_color=True, \
               handFeatures=['crossmodal_targetEEDist'], rawFeatures=None, data_renew=False):
    '''
    If ae_data is True, it returns additional task-oriented raw feature data for auto-encoders.
    '''

    if os.path.isdir(processed_data_path) is False:
        os.system('mkdir -p '+processed_data_path)

    save_pkl = os.path.join(processed_data_path, 'feature_extraction_'+rf_center+'_'+\
                            str(local_range) )
            
    if os.path.isfile(save_pkl) and data_renew is False:
        print "--------------------------------------"
        print "Load saved data"
        print "--------------------------------------"
        data_dict = ut.load_pickle(save_pkl)
        print data_dict.keys()
        if ae_data:
            # Task-oriented raw features
            successData     = data_dict['aeSuccessData'] 
            failureData     = data_dict['aeFailureData']
            failureNameList = None
            param_dict      = data_dict.get('aeParamDict', [])
        else:        
            # Task-oriented hand-crafted features
            allData         = data_dict['allData']
            successData     = data_dict['successData'] 
            failureData     = data_dict['failureData']
            failureNameList = None #data_dict['abnormalTestNameList']
            param_dict      = data_dict['param_dict']

    else:
        ## data_renew = False #temp        
        success_list, failure_list = util.getSubjectFileList(raw_data_path, subject_names, task_name)

        # loading and time-sync    
        all_data_pkl     = os.path.join(processed_data_path, task_name+'_all_'+rf_center+\
                                        '_'+str(local_range))
        _, all_data_dict = util.loadData(success_list+failure_list, isTrainingData=False,
                                         downSampleSize=downSampleSize,\
                                         local_range=local_range, rf_center=rf_center,\
                                         ##global_data=True,\
                                         renew=data_renew, save_pkl=all_data_pkl)

        # data set
        success_data_pkl     = os.path.join(processed_data_path, task_name+'_success_'+rf_center+\
                                            '_'+str(local_range))
        _, success_data_dict = util.loadData(success_list, isTrainingData=True,
                                             downSampleSize=downSampleSize,\
                                             local_range=local_range, rf_center=rf_center,\
                                             renew=data_renew, save_pkl=success_data_pkl)

        failure_data_pkl     = os.path.join(processed_data_path, task_name+'_failure_'+rf_center+\
                                            '_'+str(local_range))
        _, failure_data_dict = util.loadData(failure_list, isTrainingData=False,
                                             downSampleSize=downSampleSize,\
                                             local_range=local_range, rf_center=rf_center,\
                                             renew=data_renew, save_pkl=failure_data_pkl)

        # Task-oriented hand-crafted features
        if init_param_dict is not None:
            allData, _ = extractHandFeature(all_data_dict, handFeatures, scale=scale,\
                                            param_dict=init_param_dict, cut_data=cut_data)
            param_dict=init_param_dict                                            
        else:
            allData, param_dict = extractHandFeature(all_data_dict, handFeatures, scale=scale,\
                                                     cut_data=cut_data)            
        successData, _      = extractHandFeature(success_data_dict, handFeatures, scale=scale, \
                                                 param_dict=param_dict, cut_data=cut_data)
        failureData, _      = extractHandFeature(failure_data_dict, handFeatures, scale=scale, \
                                                 param_dict=param_dict, cut_data=cut_data)

        data_dict = {}
        data_dict['allData']      = allData = np.array(allData)
        data_dict['successData']  = successData = np.array(successData)
        data_dict['failureData']  = failureData = np.array(failureData)
        data_dict['dataNameList'] = failureNameList = None #failure_data_dict['fileNameList']
        data_dict['param_dict'] = param_dict
        
        if rawFeatures is not None: #ae_data and 
            # Task-oriented raw features
            ae_successData, ae_failureData, ae_param_dict = \
              extractRawFeature(all_data_dict, rawFeatures, nSuccess=len(success_list), \
                             nFailure=len(failure_list), cut_data=cut_data)

            data_dict['aeSuccessData'] = np.array(ae_successData)
            data_dict['aeFailureData'] = np.array(ae_failureData)
            data_dict['aeParamDict']   = ae_param_dict

        ut.save_pickle(data_dict, save_pkl)

    #-----------------------------------------------------------------------------
    ## All data
    nPlot = None

    # almost deprecated??
    feature_names = np.array(param_dict.get('feature_names', handFeatures))
    AddFeature_names    = feature_names

    # -------------------- Display ---------------------
    fig = None
    if success_viz:

        fig = plt.figure()
        n,m,k = np.shape(successData)
        ## if nPlot is None:
        ##     if n%2==0: nPlot = n
        ##     else: nPlot = n+1
        nPlot = n

        for i in xrange(n):
            ## ax = fig.add_subplot((nPlot/2)*100+20+i)
            ax = fig.add_subplot(n*100+10+i)
            if solid_color: ax.plot(successData[i].T, c='b')
            else: ax.plot(successData[i].T)

            print AddFeature_names[i]
            if AddFeature_names[i] == 'ftForce_mag': ax.set_ylabel('Force Magnitude (N)')
            elif AddFeature_names[i] == 'artagEEDist': ax.set_ylabel('Relative Distance (m)')
            elif AddFeature_names[i] == 'audioWristRMS': ax.set_ylabel('Sound Energy')
            else: ax.set_ylabel(AddFeature_names[i])
                ## ax.set_title( AddFeature_names[i] )

    if failure_viz:
        if fig is None: fig = plt.figure()
        n,m,k = np.shape(failureData)
        ## if nPlot is None:
        ##     if n%2==0: nPlot = n
        ##     else: nPlot = n+1
        nPlot = n

        for i in xrange(n):
            ## ax = fig.add_subplot((nPlot/2)*100+20+i)
            ax = fig.add_subplot(n*100+10+i)
            if solid_color: ax.plot(failureData[i].T, c='r')
            else: ax.plot(failureData[i].T)
            ax.set_title( AddFeature_names[i] )

    if success_viz or failure_viz:
        plt.tight_layout(pad=3.0, w_pad=0.5, h_pad=0.5)

        if save_pdf:
            fig.savefig('test.pdf')
            fig.savefig('test.png')
            os.system('cp test.p* ~/Dropbox/HRL/')        
        else:
            plt.show()

    print "---------------------------------------------------"
    print "s/f data: ", np.shape(successData), np.shape(failureData)
    print "---------------------------------------------------"
    return data_dict


def getAEdataSet(idx, rawSuccessData, rawFailureData, handSuccessData, handFailureData, handParam, \
                 normalTrainIdx, abnormalTrainIdx, normalTestIdx, abnormalTestIdx,
                 time_window, nAugment, \
                 AE_proc_data, \
                 # data param
                 processed_data_path, \
                 # AE param
                 layer_sizes=[256,128,16], learning_rate=1e-6, learning_rate_decay=1e-6, \
                 momentum=1e-6, dampening=1e-6, lambda_reg=1e-6, \
                 max_iteration=20000, min_loss=1.0, cuda=False, \
                 filtering=True, filteringDim=4, method='ae',\
                 # PCA param
                 pca_gamma=5.0,\
                 verbose=False, renew=False, preTrainModel=None ):

    ## if os.path.isfile(AE_proc_data) and not renew:        
    ##     d = ut.load_pickle(AE_proc_data)
    ##     ## d['handFeatureNames'] = handParam['feature_names']
    ##     ## ut.save_pickle(d, AE_proc_data)
    ##     return d

    # dim x sample x length
    normalTrainData   = rawSuccessData[:, normalTrainIdx, :] 
    abnormalTrainData = rawFailureData[:, abnormalTrainIdx, :] 
    normalTestData    = rawSuccessData[:, normalTestIdx, :] 
    abnormalTestData  = rawFailureData[:, abnormalTestIdx, :]

    # sample x dim x length
    normalTrainData   = np.swapaxes(normalTrainData, 0, 1)
    abnormalTrainData = np.swapaxes(abnormalTrainData, 0, 1)
    normalTestData    = np.swapaxes(normalTestData, 0, 1)
    abnormalTestData  = np.swapaxes(abnormalTestData, 0, 1)

    # data augmentation for auto encoder
    if nAugment>0:
        normalTrainDataAug, abnormalTrainDataAug = data_augmentation(normalTrainData, \
                                                                     abnormalTrainData, nAugment)
    else:
        normalTrainDataAug   = normalTrainData
        abnormalTrainDataAug = abnormalTrainData

    # sample x time_window_flatten_length
    normalTrainDataAugConv   = getTimeDelayData(normalTrainDataAug, time_window)
    abnormalTrainDataAugConv = getTimeDelayData(abnormalTrainDataAug, time_window)
    normalTrainDataConv      = getTimeDelayData(normalTrainData, time_window)
    abnormalTrainDataConv    = getTimeDelayData(abnormalTrainData, time_window)
    normalTestDataConv       = getTimeDelayData(normalTestData, time_window)
    abnormalTestDataConv     = getTimeDelayData(abnormalTestData, time_window)
    nSingleData              = len(normalTrainDataAug[0][0])-time_window+1
    nDim                     = len(normalTrainDataConv[1])

    # sample x time_window_flatten_length
    if nAugment>0:
        X_train  = np.vstack([normalTrainDataAugConv, abnormalTrainDataAugConv])
    else:
        X_train  = np.vstack([normalTrainDataConv, abnormalTrainDataConv])        

    # train ae
    if method == 'ae':
        print "Loading ae_model data"
        from hrl_anomaly_detection.feature_extractors import auto_encoder as ae
        ml = ae.auto_encoder([nDim]+layer_sizes, \
                             learning_rate, learning_rate_decay, momentum, dampening, \
                             lambda_reg, time_window, \
                             max_iteration=max_iteration, min_loss=min_loss, cuda=cuda, verbose=True)

        AE_model = os.path.join(processed_data_path, 'ae_model_'+str(idx)+'.pkl')
        if os.path.isfile(AE_model):
            print "AE model exists: ", AE_model
            ## ml.load_params(AE_model)
            ml.create_layers(load=True, filename=AE_model)
        else:
            if preTrainModel is not None:
                ml.fit(X_train, save_obs={'save': False, 'load': True, 'filename': preTrainModel})
            else:
                ml.fit(X_train)
            ml.save_params(AE_model)

        def predictFeatures(clf, X, nSingleData):
            # Generate training features
            feature_list = []
            for idx in xrange(0, len(X), nSingleData):
                test_features = clf.predict_features( X[idx:idx+nSingleData,:].astype('float32') )
                feature_list.append(test_features)
            return feature_list

        # test ae
        # sample x dim => dim x sample
        d = {}
        d['normTrainData']   = np.swapaxes(predictFeatures(ml, normalTrainDataConv, nSingleData), 0,1)
        d['abnormTrainData'] = np.swapaxes(predictFeatures(ml, abnormalTrainDataConv, nSingleData), 0,1) 
        d['normTestData']    = np.swapaxes(predictFeatures(ml, normalTestDataConv, nSingleData), 0,1)
        d['abnormTestData']  = np.swapaxes(predictFeatures(ml, abnormalTestDataConv, nSingleData), 0,1)
            
    else:
        print "Loading pca model data"
        from sklearn.decomposition import KernelPCA
        ml = KernelPCA(n_components=layer_sizes[-1], kernel="rbf", fit_inverse_transform=False, \
                       gamma=pca_gamma)

        print np.shape(normalTrainData), np.shape(abnormalTrainData)
        print np.shape(normalTrainDataConv), np.shape(abnormalTrainDataConv)
        print np.shape(X_train)
        print "Exit in pca data extraction"
        sys.exit()

        pca_model = os.path.join(processed_data_path, 'pca_model_'+str(idx)+'.pkl')
        if os.path.isfile(pca_model):
            print "PCA model exists: ", pca_model
            ml = joblib.load(pca_model)
        else:
            ml.fit(np.array(X_train))
            joblib.dump(ml, pca_model)

        def predictFeatures(clf, X, nSingleData):
            # Generate training features
            feature_list = []
            for idx in xrange(0, len(X), nSingleData):
                test_features = clf.transform( X[idx:idx+nSingleData,:] )
                feature_list.append(test_features)
                print np.shape(X[idx:idx+nSingleData,:]), np.shape(test_features)
            return feature_list

        # test ae
        # sample x dim => dim x sample
        d = {}
        d['normTrainData']   = np.swapaxes(predictFeatures(ml, normalTrainDataConv, nSingleData), 0,1)
        d['abnormTrainData'] = np.swapaxes(predictFeatures(ml, abnormalTrainDataConv, nSingleData), 0,1) 
        d['normTestData']    = np.swapaxes(predictFeatures(ml, normalTestDataConv, nSingleData), 0,1)
        d['abnormTestData']  = np.swapaxes(predictFeatures(ml, abnormalTestDataConv, nSingleData), 0,1)

        print np.shape(predictFeatures(ml, normalTrainDataConv, nSingleData))
        sys.exit()
    
    # dim x sample x length
    d['handNormTrainData']   = handSuccessData[:, normalTrainIdx, time_window-1:]
    d['handAbnormTrainData'] = handFailureData[:, abnormalTrainIdx, time_window-1:]
    d['handNormTestData']    = handSuccessData[:, normalTestIdx, time_window-1:]
    d['handAbnormTestData']  = handFailureData[:, abnormalTestIdx, time_window-1:]

    if filtering:
        pooling_param_dict  = {'dim': filteringDim} # only for AE        
        d['normTrainDataFiltered'], d['abnormTrainDataFiltered'],pooling_param_dict \
          = errorPooling(d['normTrainData'], d['abnormTrainData'], pooling_param_dict)
        d['normTestDataFiltered'], d['abnormTestDataFiltered'], _ \
          = errorPooling(d['normTestData'], d['abnormTestData'], pooling_param_dict)

    d['handFeatureNames'] = handParam['feature_names']
    ut.save_pickle(d, AE_proc_data)
    return d



def getHMMData(method, nFiles, processed_data_path, task_name, default_params, negTrain=False):
    import os
    from sklearn import preprocessing

    ## Default Parameters
    # data
    data_dict = default_params['data_param']
    # AE
    AE_dict = default_params['AE']
    # HMM
    HMM_dict = default_params['HMM']
    # ROC
    ROC_dict = default_params['ROC']
    #------------------------------------------

    # load data and preprocess it
    print "Start to get data"
    data = {}
    for file_idx in xrange(nFiles):
        if AE_dict['switch'] and AE_dict['add_option'] is not None:
            tag = ''
            for ft in AE_dict['add_option']:
                tag += ft[:2]
            modeling_pkl = os.path.join(processed_data_path, 'hmm_'+task_name+'_raw_'+tag+'_'+str(file_idx)+'.pkl')
        elif AE_dict['switch']:
            modeling_pkl = os.path.join(processed_data_path, \
                                        'hmm_'+task_name+'_raw_'+str(file_idx)+'.pkl')
        else:
            modeling_pkl = os.path.join(processed_data_path, \
                                        'hmm_'+task_name+'_'+str(file_idx)+'.pkl')

        if os.path.isfile(modeling_pkl) is False:
            print "Please run evaluation all first to get hmm files."
            sys.exit()
        # train a classifier and evaluate it using test data.
        d            = ut.load_pickle(modeling_pkl)
        ## startIdx = d['startIdx']
        
        # sample x length x feature vector
        ll_classifier_train_X   = d['ll_classifier_train_X']
        ll_classifier_train_Y   = d['ll_classifier_train_Y']         
        ll_classifier_train_idx = d['ll_classifier_train_idx']
        ## ll_classifier_test_X    = d['ll_classifier_test_X']  
        ## ll_classifier_test_Y    = d['ll_classifier_test_Y']
        ## ll_classifier_test_idx  = d['ll_classifier_test_idx']
        nLength      = d['nLength']
        nPoints      = ROC_dict['nPoints']

        if method == 'hmmsvm_dL':
            # replace dL/(ds+e) to dL
            for i in xrange(len(ll_classifier_train_X)):
                for j in xrange(len(ll_classifier_train_X[i])):
                    if j == 0:
                        ll_classifier_train_X[i][j][1] = 0.0
                    else:
                        ll_classifier_train_X[i][j][1] = ll_classifier_train_X[i][j][0] - \
                          ll_classifier_train_X[i][j-1][0]
        elif method == 'hmmsvm_LSLS':
            # reconstruct data into LS(t-1)+LS(t)
            if type(ll_classifier_train_X) is list:
                ll_classifier_train_X = np.array(ll_classifier_train_X)

            x = np.dstack([ll_classifier_train_X[:,:,:1], ll_classifier_train_X[:,:,2:]] )
            x = x.tolist()
            ## ll_classifier_train_X = np.hstack([ x, x ])

            new_x = []
            for i in xrange(len(x)):
                new_x.append([])
                for j in xrange(len(x[i])):
                    if j == 0:
                        new_x[i].append( x[i][j]+x[i][j] )
                    else:
                        new_x[i].append( x[i][j-1]+x[i][j] )

            ll_classifier_train_X = new_x
            

        # divide into training and param estimation set
        import random
        train_idx = random.sample(range(len(ll_classifier_train_X)), int( 0.7*len(ll_classifier_train_X)) )
        test_idx  = [x for x in range(len(ll_classifier_train_X)) if not x in train_idx] 

        train_X = np.array(ll_classifier_train_X)[train_idx]
        train_Y = np.array(ll_classifier_train_Y)[train_idx]
        train_idx = np.array(ll_classifier_train_idx)[train_idx]
        test_X  = np.array(ll_classifier_train_X)[test_idx]
        test_Y  = np.array(ll_classifier_train_Y)[test_idx]
        test_idx = np.array(ll_classifier_train_idx)[test_idx]

        if negTrain:
            normal_idx = [x for x in range(len(train_X)) if train_Y[x][0]<0 ]
            train_X = train_X[normal_idx]
            train_Y = train_Y[normal_idx]
            train_idx = train_idx[normal_idx]

        # flatten the data
        X_train_org, Y_train_org, idx_train_org = flattenSample(train_X, train_Y, train_idx)

        # training data preparation
        if 'svm' in method or 'sgd' in method:
            scaler = preprocessing.StandardScaler()
            ## scaler = preprocessing.scale()
            X_scaled = scaler.fit_transform(X_train_org)
        else:
            X_scaled = X_train_org
        ## print method, " : Before classification : ", np.shape(X_scaled), np.shape(Y_train_org)

        # test data preparation
        X_test = []
        Y_test = []
        idx_test = test_idx
        for ii in xrange(len(test_X)):
            if np.nan in test_X[ii] or len(test_X[ii]) == 0 \
              or np.nan in test_X[ii][0]:
                continue

            if 'svm' in method or 'sgd' in method:
                X = scaler.transform(test_X[ii])                                
            elif method == 'progress_time_cluster' or method == 'fixed':
                X = test_X[ii]
            X_test.append(X)
            Y_test.append(test_Y[ii])


        data[file_idx]={}
        data[file_idx]['X_scaled']      = X_scaled
        data[file_idx]['Y_train_org']   = Y_train_org
        data[file_idx]['idx_train_org'] = idx_train_org
        data[file_idx]['X_test']   = X_test
        data[file_idx]['Y_test']   = Y_test
        data[file_idx]['idx_test'] = idx_test
        data[file_idx]['nLength'] = nLength

    return data 


def getPCAData(nFiles, startIdx, data_pkl, window=1, posdata=False, gamma=1., pos_cut_indices=None):

    d = ut.load_pickle(data_pkl)
    kFold_list  = d['kFoldList']
    successData = d['successData']
    failureData = d['failureData']

    # load data and preprocess it
    print "Start to get data"
    data = {}
    for file_idx in xrange(nFiles):
    
        (normalTrainIdx, abnormalTrainIdx, normalTestIdx, abnormalTestIdx) = kFold_list[file_idx]

        # dim x sample x length
        normalTrainData   = successData[:, normalTrainIdx, :] 
        abnormalTrainData = failureData[:, abnormalTrainIdx, :] 
        normalTestData    = successData[:, normalTestIdx, :] 
        abnormalTestData  = failureData[:, abnormalTestIdx, :] 

        # sample x dim x length
        normalTrainData   = np.swapaxes(normalTrainData, 0, 1)
        abnomalTrainData  = np.swapaxes(abnormalTrainData, 1, 2)         
        normalTestData    = np.swapaxes(normalTestData, 0, 1)
        abnormalTestData  = np.swapaxes(abnormalTestData, 0, 1)

        # sample x length x dim 
        normalTrainData   = np.swapaxes(normalTrainData, 1, 2)
        abnormalTrainData = np.swapaxes(abnormalTrainData, 1, 2)         
        normalTestData    = np.swapaxes(normalTestData, 1, 2)
        abnormalTestData  = np.swapaxes(abnormalTestData, 1, 2)

        #--------------------------------------------------------------------------------
        # Training data
        ll_classifier_train_X = normalTrainData
        ll_classifier_train_Y = [[-1]*len(normalTrainData[0])]*len(normalTrainData)
                    
        # flatten the data
        if window == 0: sys.exit()
        elif window==1:
            X_train_org, Y_train_org, _ = flattenSample(ll_classifier_train_X, \
                                                        ll_classifier_train_Y)
        else:
            X_train_org, Y_train_org, _ = flattenSampleWithWindow(ll_classifier_train_X, \
                                                                  ll_classifier_train_Y, window=window)

        if posdata and pos_cut_indices is not None:
            abnormalTrainData_X = []
            abnormalTrainData_Y = []
            for i, cut_idx in enumerate(pos_cut_indices[file_idx]):
                abnormalTrainData_X.append( abnormalTrainData[i][cut_idx:].tolist() )
                abnormalTrainData_Y.append( [1]*len(abnormalTrainData_X[i]) )
                    
            if window == 0: sys.exit()
            elif window==1:
                X_abnorm_train, Y_abnorm_train, _ = flattenSample(abnormalTrainData_X, \
                                                                  abnormalTrainData_Y)
            else:
                X_abnorm_train, Y_abnorm_train, _ = flattenSampleWithWindow(abnormalTrainData_X, \
                                                                            abnormalTrainData_Y, window=window)

            print np.shape(X_train_org), np.shape(X_abnorm_train)
            X_train_org = X_train_org + X_abnorm_train
            Y_train_org = Y_train_org + Y_abnorm_train
            print np.shape(X_train_org)
            print "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            sys.exit()


                                                                  
        # scaling
        from sklearn import preprocessing
        scaler = preprocessing.StandardScaler()
        X_scaled = scaler.fit_transform(X_train_org)
                       
        ## # PCA
        ## from sklearn.decomposition import KernelPCA
        ## ml = KernelPCA(n_components=2, kernel="poly", fit_inverse_transform=False, \
        ##                    gamma=gamma, degree=5)
        ## X_scaled = ml.fit_transform(np.array(X_scaled))

        # LLE
        ## from sklearn.manifold import LocallyLinearEmbedding
        ## ml = LocallyLinearEmbedding(5,2, reg=gamma)        
        ## X_scaled = ml.fit_transform(np.array(X_scaled))

        #--------------------------------------------------------------------------------
        # Testing data
        ll_classifier_test_X   = np.vstack([normalTestData, abnormalTestData])
        ll_classifier_test_Y   = [[-1]*len(normalTestData[0])]*len(normalTestData)+\
          [[1]*len(abnormalTestData[0])]*len(abnormalTestData)
        ll_classifier_test_idx = [range(len(normalTestData[0]))]*len(normalTestData) + \
          [range(len(abnormalTestData[0]))]*len(abnormalTestData)
        ll_classifier_test_idx = np.array(ll_classifier_test_idx)

        # test data preparation
        X_test = []
        Y_test = []
        if window==1:
            for ii in xrange(len(ll_classifier_test_X)):
                if np.nan in ll_classifier_test_X[ii] or len(ll_classifier_test_X[ii]) == 0 \
                  or np.nan in ll_classifier_test_X[ii][0]:
                    continue

                X = scaler.transform(ll_classifier_test_X[ii])
                ## X = ml.transform(X)
                X_test.append(X)
                Y_test.append(ll_classifier_test_Y[ii])            
        else:
            ll_classifier_test_X = sampleWithWindow(ll_classifier_test_X, window=window)
            for ii in xrange(len(ll_classifier_test_X)):
                if np.nan in ll_classifier_test_X[ii] or len(ll_classifier_test_X[ii]) == 0 \
                  or np.nan in ll_classifier_test_X[ii][0]:
                    continue

                X = scaler.transform(ll_classifier_test_X[ii])
                ## X = ml.transform(X)
                X_test.append(X)
                Y_test.append(ll_classifier_test_Y[ii])

        
        ## fig = plt.figure(1)
        ## plt.scatter(X_scaled[:,0], X_scaled[:,1], c='blue')
        ## for i in xrange(len(X_test)):
        ##     if Y_test[i][0] == -1: continue
        ##     plt.scatter(np.array(X_test)[i,:,0], np.array(X_test)[i,:,1], c='red', marker='x')        
        ## plt.axis('tight')        
        ## plt.show()
        ## fig.savefig('test'+str(file_idx)+'.pdf')
        ## fig.savefig('test'+str(file_idx)+'.png')
        ## os.system('mv test*.png ~/Dropbox/HRL/')

        #--------------------------------------------------------------------------------
        data[file_idx]={}
        data[file_idx]['X_scaled']      = X_scaled
        data[file_idx]['Y_train_org']   = Y_train_org
        data[file_idx]['idx_train_org'] = None
        data[file_idx]['X_test']        = X_test
        data[file_idx]['Y_test']        = Y_test
        data[file_idx]['idx_test']      = None
        data[file_idx]['nLength']       = len(normalTrainData[0][0])

    return data 
    

def getHMMCuttingIdx(ll_X, ll_Y, ll_idx):
    '''
    ll : sample x length x hmm features
    It takes only positive data.
    '''
    
    l_X   = []
    l_Y   = []
    l_idx = []
    for i in xrange(len(ll_X)):
        if ll_Y[i][0] < 0:
            print "Error: negative data come in."
            sys.exit()
        else:
            _,_,idx = getEstTruePositive(ll_X[i], ll_idx[i])
            l_idx.append(idx)
            
    return l_idx
    

def errorPooling(norX, abnorX, param_dict):
    '''
    dim x samples
    Select non-stationary data
    Assuption: norX and abnorX should have the same phase
    '''
    dim         = param_dict['dim']

    if 'dim_idx' not in param_dict.keys():
        dim_idx    = []
        new_norX   = []
        new_abnorX = []

        err_list = []
        for i in xrange(len(norX)):
            # get mean curve
            meanNorCurve   = np.mean(norX[i], axis=0)
            ## meanAbnorCurve = np.mean(abnorX[i], axis=0)
            stdNorCurve   = np.std(norX[i], axis=0)
            ## stdAbnorCurve = np.std(abnorX[i], axis=0)
            ## if np.std(meanNorCurve) < 0.02 and np.std(meanAbnorCurve) < 0.02 and\
            ##   np.mean(stdNorCurve) < 0.02 and np.mean(stdAbnorCurve) < 0.02:
            ##     err_list.append(1e-9)
            ##     continue

            maxCurve = meanNorCurve+stdNorCurve
            minCurve = meanNorCurve-stdNorCurve

            # get error score
            score = 0.0
            for j in xrange(len(abnorX[i])):
                for k in xrange(len(abnorX[i][j])):
                    if abnorX[i][j][k] > maxCurve[k] or abnorX[i][j][k] < minCurve[k]:
                       score += 1.0

            # get mean range , mean std
            score /= np.max(meanNorCurve)-np.min(meanNorCurve)
            #score *= np.mean(stdNorCurve)
                       
            err_list.append(score)

        indices = np.argsort(err_list)

        for idx in indices[:dim]:
            new_norX.append(norX[idx])
            new_abnorX.append(abnorX[idx])
            dim_idx.append(idx)

            ## if all_std > min_all_std and avg_ea_std < max_avg_std:
            ##     new_X.append(X[i])
            ##     dim_idx.append(i)

        param_dict['dim_idx'] = dim_idx
    else:
        new_norX = [ norX[idx] for idx in param_dict['dim_idx'] ]
        new_abnorX = [ abnorX[idx] for idx in param_dict['dim_idx'] ]

    return np.array(new_norX), np.array(new_abnorX), param_dict
        

def variancePooling(X, param_dict):
    '''
    dim x samples
    Select non-stationary data
    
    TODO: can we select final dimension?
    
    '''
    dim         = param_dict['dim']
    ## min_all_std = param_dict['min_all_std']
    ## max_avg_std = param_dict['max_avg_std']

    if 'dim_idx' not in param_dict.keys():
        dim_idx = []
        new_X   = []

        std_list = []
        for i in xrange(len(X)):
            # for each dimension
            ## avg_std = np.mean( np.std(X[i], axis=0) )
            std_avg = np.std( np.mean(X[i], axis=0) )
            ## std_list.append(std_avg/avg_std)
            std_list.append(std_avg)

        indices = np.argsort(std_list)[::-1]

        for idx in indices[:dim]:
            new_X.append(X[idx])
            dim_idx.append(idx)

            ## if all_std > min_all_std and avg_ea_std < max_avg_std:
            ##     new_X.append(X[i])
            ##     dim_idx.append(i)

        param_dict['dim_idx'] = dim_idx
    else:
        new_X = [ X[idx] for idx in param_dict['dim_idx'] ]

    return np.array(new_X), param_dict
        
    
#-------------------------------------------------------------------------------------------------

def extractHandFeature(d, feature_list, scale=1.0, cut_data=None, param_dict=None, verbose=False):

    if param_dict is None:
        isTrainingData=True
        param_dict = {}
        param_dict['timeList'] = d['timesList'][0]

        if 'unimodal_audioPower' in feature_list:
            ## power_max = np.amax(d['audioPowerList'])
            ## power_min = np.amin(d['audioPowerList'])
            ## power_min = np.mean(np.array(d['audioPowerList'])[:,:10])
            power_min = 10000
            power_max = 0
            for pwr in d['audioPowerList']:
                p_min = np.amin(pwr)
                p_max = np.amax(pwr)
                if power_min > p_min:
                    power_min = p_min
                ## if p_max < 50 and power_max < p_max:
                if power_max < p_max:
                    power_max = p_max

            param_dict['unimodal_audioPower_power_max'] = power_max
            param_dict['unimodal_audioPower_power_min'] = power_min
                                
        ## if 'unimodal_ftForce' in feature_list:
        ##     force_array = None
        ##     start_force_array = None
        ##     for idx in xrange(len(d['ftForceList'])):
        ##         if force_array is None:
        ##             force_array = d['ftForceList'][idx]
        ##             ## start_force_array = d['ftForceList'][idx][:,:5]
        ##         else:
        ##             force_array = np.hstack([force_array, d['ftForceList'][idx] ])
        ##             ## start_force_array = np.hstack([start_force_array, d['ftForceList'][idx][:,:5]])

        ##     ftPCADim    = 2
        ##     ftForce_pca = PCA(n_components=ftPCADim)
        ##     res = ftForce_pca.fit_transform( force_array.T )            
        ##     param_dict['unimodal_ftForce_pca'] = ftForce_pca
        ##     param_dict['unimodal_ftForce_pca_dim'] = ftPCADim

        ##     ## res = ftForce_pca.transform(start_force_array.T)
        ##     ## param_dict['unimodal_ftForce_pca_init_avg'] = np.array([np.mean(res, axis=0)]).T
        ##     ## param_dict['unimodal_ftForce_init_avg'] = np.mean(start_force_array, axis=1)

        if 'unimodal_ppsForce' in feature_list:
            ppsLeft  = d['ppsLeftList']
            ppsRight = d['ppsRightList']

            pps_mag = []
            for i in xrange(len(ppsLeft)):                
                pps      = np.vstack([ppsLeft[i], ppsRight[i]])
                pps_mag.append( np.linalg.norm(pps, axis=0) )

            pps_max = np.max( np.array(pps_mag).flatten() )
            pps_min = np.min( np.array(pps_mag).flatten() )
            param_dict['unimodal_ppsForce_max'] = pps_max
            param_dict['unimodal_ppsForce_min'] = pps_min

        param_dict['feature_names'] = []
    else:
        isTrainingData=False
            

    # -------------------------------------------------------------        

    # extract local features
    startOffsetSize = 4
    dataList   = []
    for idx in xrange(len(d['timesList'])): # each sample

        timeList     = d['timesList'][idx]
        dataSample = None
        if len(timeList) < 2: offset_flag=False
        else: offset_flag=True
            

        # Unimoda feature - Audio --------------------------------------------
        if 'unimodal_audioPower' in feature_list:
            ## audioAzimuth = d['audioAzimuthList'][idx]
            audioPower   = d['audioPowerList'][idx]            
            unimodal_audioPower = audioPower
            
            if dataSample is None: dataSample = copy.copy(np.array(unimodal_audioPower))
            else: dataSample = np.vstack([dataSample, copy.copy(unimodal_audioPower)])
            if 'audioPower' not in param_dict['feature_names']:
                param_dict['feature_names'].append('audioPower')

        # Unimoda feature - AudioWrist ---------------------------------------
        if 'unimodal_audioWristRMS' in feature_list:
            audioWristRMS = d['audioWristRMSList'][idx]            
            unimodal_audioWristRMS = audioWristRMS
            if offset_flag:
                unimodal_audioWristRMS -= np.mean(audioWristRMS[:startOffsetSize])

            if dataSample is None: dataSample = copy.copy(np.array(unimodal_audioWristRMS))
            else: dataSample = np.vstack([dataSample, copy.copy(unimodal_audioWristRMS)])
            if 'audioWristRMS' not in param_dict['feature_names']:
                param_dict['feature_names'].append('audioWristRMS')

        # Unimodal feature - Kinematics --------------------------------------
        if 'unimodal_kinVel' in feature_list:
            kinVel  = d['kinVelList'][idx]
            unimodal_kinVel = kinVel

            if dataSample is None: dataSample = np.array(unimodal_kinVel)
            else: dataSample = np.vstack([dataSample, unimodal_kinVel])
            if 'kinVel_x' not in param_dict['feature_names']:
                param_dict['feature_names'].append('kinVel_x')
                param_dict['feature_names'].append('kinVel_y')
                param_dict['feature_names'].append('kinVel_z')

        # Unimodal feature - Force -------------------------------------------
        if 'unimodal_ftForce' in feature_list:
            ftForce = d['ftForceList'][idx]

            # magnitude
            if len(np.shape(ftForce)) > 1:
                unimodal_ftForce_mag = np.linalg.norm(ftForce, axis=0)
                # individual force
                ## unimodal_ftForce_ind = ftForce[2:3,:]
                if offset_flag:
                    unimodal_ftForce_mag -= np.mean(unimodal_ftForce_mag[:startOffsetSize])
                
                if dataSample is None: dataSample = np.array(unimodal_ftForce_mag)
                else: dataSample = np.vstack([dataSample, unimodal_ftForce_mag])

                ## if dataSample is None: dataSample = np.array(unimodal_ftForce_ind)
                ## else: dataSample = np.vstack([dataSample, unimodal_ftForce_ind])

                if 'ftForce_mag' not in param_dict['feature_names']:
                    param_dict['feature_names'].append('ftForce_mag')
                    ## param_dict['feature_names'].append('ftForce_x')
                    ## param_dict['feature_names'].append('ftForce_y')
                    ## param_dict['feature_names'].append('ftForce_z')
            else:                
                unimodal_ftForce_mag = ftForce
            
                if dataSample is None: dataSample = np.array(unimodal_ftForce_mag)
                else: dataSample = np.vstack([dataSample, unimodal_ftForce_mag])

                if 'ftForce_mag' not in param_dict['feature_names']:
                    param_dict['feature_names'].append('ftForce_mag')

            ## ftPos   = d['kinEEPosList'][idx]
            ## ftForce_pca = param_dict['unimodal_ftForce_pca']

            ## unimodal_ftForce = None
            ## for time_idx in xrange(len(timeList)):
            ##     if unimodal_ftForce is None:
            ##         unimodal_ftForce = ftForce_pca.transform(ftForce[:,time_idx:time_idx+1].T).T
            ##     else:
            ##         unimodal_ftForce = np.hstack([ unimodal_ftForce, \
            ##                                        ftForce_pca.transform(ftForce[:,time_idx:time_idx+1].T).T ])

            ## unimodal_ftForce -= np.array([np.mean(unimodal_ftForce[:,:5], axis=1)]).T
            
            ## if 'ftForce_1' not in param_dict['feature_names']:
            ##     param_dict['feature_names'].append('ftForce_1')
            ##     param_dict['feature_names'].append('ftForce_2')
            ## if 'ftForce_x' not in param_dict['feature_names']:
            ##     param_dict['feature_names'].append('ftForce_x')
            ##     param_dict['feature_names'].append('ftForce_y')
            ##     param_dict['feature_names'].append('ftForce_z')

        # Unimodal feature - pps -------------------------------------------
        if 'unimodal_ppsForce' in feature_list:
            ppsLeft  = d['ppsLeftList'][idx]
            ppsRight = d['ppsRightList'][idx]
            ppsPos   = d['kinTargetPosList'][idx]

            pps = np.vstack([ppsLeft, ppsRight])
            unimodal_ppsForce = pps

            # 2
            pps = np.vstack([np.sum(ppsLeft, axis=0), np.sum(ppsRight, axis=0)])
            unimodal_ppsForce = pps
            
            # 1
            ## unimodal_ppsForce = np.array([np.linalg.norm(pps, axis=0)])
            if offset_flag:
                unimodal_ppsForce -= np.array([np.mean(unimodal_ppsForce[:,:startOffsetSize], \
                                                       axis=1)]).T

            ## unimodal_ppsForce = []
            ## for time_idx in xrange(len(timeList)):
            ##     unimodal_ppsForce.append( np.linalg.norm(pps[:,time_idx]) )

            if dataSample is None: dataSample = unimodal_ppsForce
            else: dataSample = np.vstack([dataSample, unimodal_ppsForce])

            ## if 'ppsForce' not in param_dict['feature_names']:
            ##     param_dict['feature_names'].append('ppsForce')
            if 'ppsForce_1' not in param_dict['feature_names']:
                param_dict['feature_names'].append('ppsForce_1')
                param_dict['feature_names'].append('ppsForce_2')                
            ## if 'ppsForce_1' not in param_dict['feature_names']:
            ##     param_dict['feature_names'].append('ppsForce_1')
            ##     param_dict['feature_names'].append('ppsForce_2')
            ##     param_dict['feature_names'].append('ppsForce_3')
            ##     param_dict['feature_names'].append('ppsForce_4')
            ##     param_dict['feature_names'].append('ppsForce_5')
            ##     param_dict['feature_names'].append('ppsForce_6')


        # Unimodal feature - vision change ------------------------------------
        if 'unimodal_visionChange' in feature_list:
            visionChangeMag = d['visionChangeMagList'][idx]

            unimodal_visionChange = visionChangeMag

            if dataSample is None: dataSample = unimodal_visionChange
            else: dataSample = np.vstack([dataSample, unimodal_visionChange])
            if 'visionChange' not in param_dict['feature_names']:
                param_dict['feature_names'].append('visionChange')

                
        # Unimodal feature - fabric skin ------------------------------------
        if 'unimodal_fabricForce' in feature_list:
            fabricMag = d['fabricMagList'][idx]

            unimodal_fabricForce = fabricMag

            if dataSample is None: dataSample = unimodal_fabricForce
            else: dataSample = np.vstack([dataSample, unimodal_fabricForce])
            if 'fabricForce' not in param_dict['feature_names']:
                param_dict['feature_names'].append('fabricForce')

            
        # Crossmodal feature - relative dist --------------------------
        if 'crossmodal_targetEEDist' in feature_list:
            kinEEPos     = d['kinEEPosList'][idx]
            kinTargetPos  = d['kinTargetPosList'][idx]

            dist = np.linalg.norm(kinTargetPos - kinEEPos, axis=0)
            if offset_flag:
                dist -= np.mean(dist[:startOffsetSize])
            
            crossmodal_targetEEDist = []
            for time_idx in xrange(len(timeList)):
                crossmodal_targetEEDist.append( dist[time_idx])

            if dataSample is None: dataSample = np.array(crossmodal_targetEEDist)
            else: dataSample = np.vstack([dataSample, crossmodal_targetEEDist])
            if 'targetEEDist' not in param_dict['feature_names']:
                param_dict['feature_names'].append('targetEEDist')


        # Crossmodal feature - relative Velocity --------------------------
        if 'crossmodal_targetEEVel' in feature_list:
            kinEEPos     = d['kinEEPosList'][idx]
            kinTargetPos  = d['kinTargetPosList'][idx]

            dist = np.linalg.norm(kinTargetPos - kinEEPos, axis=0)
            if offset_flag:
                dist -= np.mean(dist[:startOffsetSize])

            print np.shape(dist)
            vel = dist[1:]-dist[:-1]
            ## vel =
            sys.exit()
            
            crossmodal_targetEEDist = []
            for time_idx in xrange(len(timeList)):
                crossmodal_targetEEDist.append( dist[time_idx])

            if dataSample is None: dataSample = np.array(crossmodal_targetEEDist)
            else: dataSample = np.vstack([dataSample, crossmodal_targetEEDist])
            if 'targetEEDist' not in param_dict['feature_names']:
                param_dict['feature_names'].append('targetEEDist')


        # Crossmodal feature - relative angle --------------------------
        if 'crossmodal_targetEEAng' in feature_list:                
            kinEEQuat    = d['kinEEQuatList'][idx]
            kinTargetQuat = d['kinTargetQuatList'][idx]

            ## kinEEPos     = d['kinEEPosList'][idx]
            ## kinTargetPos = d['kinTargetPosList'][idx]
            ## dist         = np.linalg.norm(kinTargetPos - kinEEPos, axis=0)
            
            crossmodal_targetEEAng = []
            for time_idx in xrange(len(timeList)):

                startQuat = kinEEQuat[:,time_idx]
                endQuat   = kinTargetQuat[:,time_idx]

                diff_ang = qt.quat_angle(startQuat, endQuat)
                crossmodal_targetEEAng.append( abs(diff_ang) )

            crossmodal_targetEEAng = np.array(crossmodal_targetEEAng)
            if offset_flag:
                crossmodal_targetEEAng -= np.mean(crossmodal_targetEEAng[:startOffsetSize])

            ## fig = plt.figure()
            ## ## plt.plot(crossmodal_targetEEAng)
            ## plt.plot( kinEEQuat[0] )
            ## plt.plot( kinEEQuat[1] )
            ## plt.plot( kinEEQuat[2] )
            ## plt.plot( kinEEQuat[3] )
            ## fig.savefig('test.pdf')
            ## fig.savefig('test.png')
            ## os.system('cp test.p* ~/Dropbox/HRL/')        
            ## sys.exit()
            
            if dataSample is None: dataSample = np.array(crossmodal_targetEEAng)
            else: dataSample = np.vstack([dataSample, crossmodal_targetEEAng])
            if 'targetEEAng' not in param_dict['feature_names']:
                param_dict['feature_names'].append('targetEEAng')

        # Crossmodal feature - vision relative dist with main(first) vision target----
        if 'crossmodal_artagEEDist' in feature_list:
            kinEEPos  = d['kinEEPosList'][idx]
            visionArtagPos = d['visionArtagPosList'][idx][:3] # originally length x 3*tags

            dist = np.linalg.norm(visionArtagPos - kinEEPos, axis=0)
            if offset_flag:
                dist -= np.mean(dist[:startOffsetSize])
            
            crossmodal_artagEEDist = []
            for time_idx in xrange(len(timeList)):
                crossmodal_artagEEDist.append(dist[time_idx])

            if dataSample is None: dataSample = np.array(crossmodal_artagEEDist)
            else: dataSample = np.vstack([dataSample, crossmodal_artagEEDist])
            if 'artagEEDist' not in param_dict['feature_names']:
                param_dict['feature_names'].append('artagEEDist')

        # Crossmodal feature - vision relative angle --------------------------
        if 'crossmodal_artagEEAng' in feature_list:                
            kinEEQuat    = d['kinEEQuatList'][idx]
            visionArtagQuat = d['visionArtagQuatList'][idx][:4]

            ## kinEEPos  = d['kinEEPosList'][idx]
            ## visionArtagPos = d['visionArtagPosList'][idx][:3]
            ## dist = np.linalg.norm(visionArtagPos - kinEEPos, axis=0)
            
            crossmodal_artagEEAng = []
            for time_idx in xrange(len(timeList)):

                startQuat = kinEEQuat[:,time_idx]
                endQuat   = visionArtagQuat[:,time_idx]

                diff_ang = qt.quat_angle(startQuat, endQuat)
                crossmodal_artagEEAng.append( abs(diff_ang) )

            crossmodal_artagEEAng = np.array(crossmodal_artagEEAng)
            if offset_flag:
                crossmodal_artagEEAng -= np.mean(crossmodal_artagEEAng[:startOffsetSize])

            if dataSample is None: dataSample = np.array(crossmodal_artagEEAng)
            else: dataSample = np.vstack([dataSample, crossmodal_artagEEAng])
            if 'artagEEAng' not in param_dict['feature_names']:
                param_dict['feature_names'].append('artagEEAng')


        # Crossmodal feature - vision relative dist with sub vision target----
        if 'crossmodal_subArtagEEDist' in feature_list:
            kinEEPos  = d['kinEEPosList'][idx]
            visionArtagPos = d['visionArtagPosList'][idx][3:6] # originally length x 3*tags

            dist = np.linalg.norm(visionArtagPos - kinEEPos, axis=0)
            if offset_flag:
                dist -= np.mean(dist[:startOffsetSize])
            
            crossmodal_artagEEDist = []
            for time_idx in xrange(len(timeList)):
                crossmodal_artagEEDist.append(dist[time_idx])

            if dataSample is None: dataSample = np.array(crossmodal_artagEEDist)
            else: dataSample = np.vstack([dataSample, crossmodal_artagEEDist])
            if 'subArtagEEDist' not in param_dict['feature_names']:
                param_dict['feature_names'].append('subArtagEEDist')

        # Crossmodal feature - vision relative angle --------------------------
        if 'crossmodal_subArtagEEAng' in feature_list:                
            kinEEQuat    = d['kinEEQuatList'][idx]
            visionArtagQuat = d['visionArtagQuatList'][idx][4:8]

            crossmodal_artagEEAng = []
            for time_idx in xrange(len(timeList)):

                startQuat = kinEEQuat[:,time_idx]
                endQuat   = visionArtagQuat[:,time_idx]

                diff_ang = qt.quat_angle(startQuat, endQuat)
                crossmodal_artagEEAng.append( abs(diff_ang) )

            crossmodal_artagEEAng = np.array(crossmodal_artagEEAng)
            if offset_flag:
                crossmodal_artagEEAng -= np.mean(crossmodal_artagEEAng[:startOffsetSize])

            if dataSample is None: dataSample = np.array(crossmodal_artagEEAng)
            else: dataSample = np.vstack([dataSample, crossmodal_artagEEAng])
            if 'subArtagEEAng' not in param_dict['feature_names']:
                param_dict['feature_names'].append('subArtagEEAng')

        # ----------------------------------------------------------------
        dataList.append(dataSample)


    # Convert data structure 
    # From nSample x dim x length
    # To dim x nSample x length
    nSample      = len(dataList)
    nEmissionDim = len(dataList[0])
    features = np.swapaxes(dataList, 0, 1)    

    # cut unnecessary part #temp
    if cut_data is not None:
        features = features[:,:,cut_data[0]:cut_data[1]]

    # Scaling ------------------------------------------------------------
    if isTrainingData:
        param_dict['feature_max'] = [ np.max(np.array(feature).flatten()) for feature in features ]
        param_dict['feature_min'] = [ np.min(np.array(feature).flatten()) for feature in features ]
        print "Before scaling, max is: ", param_dict['feature_max']
        print "Before scaling, min is: ", param_dict['feature_min']
                
    scaled_features = []
    for i, feature in enumerate(features):

        if abs( param_dict['feature_max'][i] - param_dict['feature_min'][i]) < 1e-3:
            scaled_features.append( np.array(feature) )
        else:
            scaled_features.append( scale*( np.array(feature) - param_dict['feature_min'][i] )\
                                    /( param_dict['feature_max'][i] - param_dict['feature_min'][i]))

    return scaled_features, param_dict


def extractRawFeature(d, raw_feature_list, nSuccess, nFailure, param_dict=None, \
                      cut_data=None, verbose=False, scaling=True):

    from sandbox_dpark_darpa_m3.lib import hrl_dh_lib as dh
    from hrl_lib import quaternion as qt
    
    if param_dict is None:
        isTrainingData=True
        param_dict = {}
    else:
        isTrainingData=False
            
    # -------------------------------------------------------------        
    # extract modality data
    dataList = []
    dataDim  = []
    nSample  = len(d['timesList'])
    startOffsetSize = 4
    for idx in xrange(nSample): # each sample

        timeList     = d['timesList'][idx]
        dataSample = None

        # rightEE-leftEE - relative dist ----
        if 'relativePose_target_EE' in raw_feature_list:
            kinEEPos      = d['kinEEPosList'][idx]
            kinEEQuat     = d['kinEEQuatList'][idx]
            kinTargetPos  = d['kinTargetPosList'][idx]
            kinTargetQuat = d['kinTargetQuatList'][idx]

            # pos and quat?
            relativePose = []
            for time_idx in xrange(len(timeList)):
                startFrame = dh.array2KDLframe( kinTargetPos[:,time_idx].tolist() +\
                                                kinTargetQuat[:,time_idx].tolist() )
                endFrame   = dh.array2KDLframe( kinEEPos[:,time_idx].tolist()+\
                                                kinEEQuat[:,time_idx].tolist() )
                diffFrame  = endFrame*startFrame.Inverse()                                
                relativePose.append( dh.KDLframe2List(diffFrame) )

            relativePose = np.array(relativePose).T[:-1]
            relativePose[:3,:] -= np.array([np.mean(relativePose[:,:startOffsetSize], axis=1)[:3]]).T
            
            if dataSample is None: dataSample = relativePose
            else: dataSample = np.vstack([dataSample, relativePose])
            if idx == 0: dataDim.append(['relativePos_target_EE', 3])
            if idx == 0: dataDim.append(['relativeAng_target_EE', 4])
                

        # main-artag EE - vision relative dist with main(first) vision target----
        if 'relativePose_artag_EE' in raw_feature_list:
            kinEEPos        = d['kinEEPosList'][idx]
            kinEEQuat       = d['kinEEQuatList'][idx]
            visionArtagPos  = d['visionArtagPosList'][idx][:3] # originally length x 3*tags
            visionArtagQuat = d['visionArtagQuatList'][idx][:4] # originally length x 3*tags

            # pos and quat?
            relativePose = []
            for time_idx in xrange(len(timeList)):
                startFrame = dh.array2KDLframe( visionArtagPos[:,time_idx].tolist() +\
                                                visionArtagQuat[:,time_idx].tolist() )
                endFrame   = dh.array2KDLframe( kinEEPos[:,time_idx].tolist()+\
                                                kinEEQuat[:,time_idx].tolist() )
                diffFrame  = endFrame*startFrame.Inverse()                                
                relativePose.append( dh.KDLframe2List(diffFrame) )
            
            relativePose = np.array(relativePose).T[:-1]
            relativePose[:3,:] -= np.array([np.mean(relativePose[:,:startOffsetSize], axis=1)[:3]]).T
            
            if dataSample is None: dataSample = relativePose
            else: dataSample = np.vstack([dataSample, relativePose])
            if idx == 0: dataDim.append(['relativePos_artag_EE', 3])
            if idx == 0: dataDim.append(['relativeAng_artag_EE', 4])
                

        # main-artag sub-artag - vision relative dist with main(first) vision target----
        if 'relativePose_artag_artag' in raw_feature_list:
            visionArtagPos1 = d['visionArtagPosList'][idx][:3] # originally length x 3*tags
            visionArtagQuat1 = d['visionArtagQuatList'][idx][:4] # originally length x 3*tags
            visionArtagPos2 = d['visionArtagPosList'][idx][3:6] # originally length x 3*tags
            visionArtagQuat2 = d['visionArtagQuatList'][idx][4:8] # originally length x 3*tags

            # pos and quat?
            relativePose = []
            for time_idx in xrange(len(timeList)):

                startFrame = dh.array2KDLframe( visionArtagPos1[:,time_idx].tolist() +\
                                                visionArtagQuat1[:,time_idx].tolist() )
                endFrame = dh.array2KDLframe( visionArtagPos2[:,time_idx].tolist() +\
                                              visionArtagQuat2[:,time_idx].tolist() )                
                diffFrame  = endFrame*startFrame.Inverse()                                
                relativePose.append( dh.KDLframe2List(diffFrame) )

            relativePose = np.array(relativePose).T[:-1]
            relativePose[:3,:] -= np.mean(relativePose[:,:startOffsetSize], axis=1)[:3,:]

            if dataSample is None: dataSample = relativePose
            else: dataSample = np.vstack([dataSample, relativePose])
            if idx == 0: dataDim.append(['relativePos_artag_artag', 3])
            if idx == 0: dataDim.append(['relativeAng_artag_artag', 4])

        # Audio --------------------------------------------
        if 'kinectAudio' in raw_feature_list:
            audioPower   = d['audioPowerList'][idx]                        
            if dataSample is None: dataSample = copy.copy(np.array(audioPower))
            else: dataSample = np.vstack([dataSample, copy.copy(audioPower)])
            if idx == 0: dataDim.append(['kinectAudio', len(audioPower)])

        # AudioWrist ---------------------------------------
        if 'wristAudio' in raw_feature_list:
            ## audioWristRMS  = d['audioWristRMSList'][idx]
            audioWristMFCC = d['audioWristMFCCList'][idx]            

            ## if dataSample is None: dataSample = copy.copy(np.array(audioWristRMS))
            ## else: dataSample = np.vstack([dataSample, copy.copy(audioWristRMS)])

            dataSample = np.vstack([dataSample, copy.copy(audioWristMFCC)])
            ## if idx == 0: dataDim.append(['wristAudio_RMS', 1])                
            if idx == 0: dataDim.append(['wristAudio_MFCC', len(audioWristMFCC)])                

        # FT -------------------------------------------
        if 'ft' in raw_feature_list:
            ftForce  = d['ftForceList'][idx]
            ftTorque = d['ftTorqueList'][idx]

            ftForce  -= np.array([np.mean(ftForce[:startOffsetSize,:], axis=1)]).T
            ftTorque -= np.array([np.mean(ftTorque[:startOffsetSize,:], axis=1)]).T

            if dataSample is None: dataSample = np.array(ftForce)
            else: dataSample = np.vstack([dataSample, ftForce])

            if dataSample is None: dataSample = np.array(ftTorque)
            else: dataSample = np.vstack([dataSample, ftTorque])
            if idx == 0: dataDim.append(['ft_force', len(ftForce)])
            if idx == 0: dataDim.append(['ft_torque', len(ftTorque)])

        # pps -------------------------------------------
        if 'pps' in raw_feature_list:
            ppsLeft  = d['ppsLeftList'][idx]
            ppsRight = d['ppsRightList'][idx]

            ppsLeft  -= np.array([np.mean(ppsLeft[:startOffsetSize,:], axis=1)]).T
            ppsRight -= np.array([np.mean(ppsRight[:startOffsetSize,:], axis=1)]).T

            if dataSample is None: dataSample = ppsLeft
            else: dataSample = np.vstack([dataSample, ppsLeft])

            if dataSample is None: dataSample = ppsRight
            else: dataSample = np.vstack([dataSample, ppsRight])
            if idx == 0: dataDim.append(['pps', len(ppsLeft)+len(ppsRight)])

        # Kinematics --------------------------------------
        if 'kinematics' in raw_feature_list:
            kinEEPos   = d['kinEEPosList'][idx]
            kinEEQuat  = d['kinEEQuatList'][idx]
            kinJntPos  = d['kinJntPosList'][idx]
            kinPos     = d['kinPosList'][idx]
            kinVel     = d['kinVelList'][idx]

            if dataSample is None: dataSample = np.array(kinEEPos)
            else: dataSample = np.vstack([dataSample, kinEEPos])
            if 'kinEEPos_x' not in param_dict['feature_names']:
                param_dict['feature_names'].append('kinEEPos_x')
                param_dict['feature_names'].append('kinEEPos_y')
                param_dict['feature_names'].append('kinEEPos_z')

            if dataSample is None: dataSample = np.array(kinEEQuat)
            else: dataSample = np.vstack([dataSample, kinEEQuat])
            if 'kinEEQuat_x' not in param_dict['feature_names']:
                param_dict['feature_names'].append('kinEEQuat_x')
                param_dict['feature_names'].append('kinEEQuat_y')
                param_dict['feature_names'].append('kinEEQuat_z')
                param_dict['feature_names'].append('kinEEQuat_w')

            if dataSample is None: dataSample = np.array(kinJntPos)
            else: dataSample = np.vstack([dataSample, kinJntPos])
            if 'kinJntPos_1' not in param_dict['feature_names']:
                param_dict['feature_names'].append('kinJntPos_1')
                param_dict['feature_names'].append('kinJntPos_2')
                param_dict['feature_names'].append('kinJntPos_3')
                param_dict['feature_names'].append('kinJntPos_4')
                param_dict['feature_names'].append('kinJntPos_5')
                param_dict['feature_names'].append('kinJntPos_6')
                param_dict['feature_names'].append('kinJntPos_7')

            if dataSample is None: dataSample = np.array(kinPos)
            else: dataSample = np.vstack([dataSample, kinPos])
            if 'kinPos_x' not in param_dict['feature_names']:
                param_dict['feature_names'].append('kinPos_x')
                param_dict['feature_names'].append('kinPos_y')
                param_dict['feature_names'].append('kinPos_z')

            if dataSample is None: dataSample = np.array(kinVel)
            else: dataSample = np.vstack([dataSample, kinVel])
            if 'kinVel_x' not in param_dict['feature_names']:
                param_dict['feature_names'].append('kinVel_x')
                param_dict['feature_names'].append('kinVel_y')
                param_dict['feature_names'].append('kinVel_z')
                

        ## # Unimodal feature - vision change ------------------------------------
        ## if 'unimodal_visionChange' in raw_feature_list:
        ##     visionChangeMag = d['visionChangeMagList'][idx]

        ##     unimodal_visionChange = visionChangeMag

        ##     if dataSample is None: dataSample = unimodal_visionChange
        ##     else: dataSample = np.vstack([dataSample, unimodal_visionChange])
        ##     if 'visionChange' not in param_dict['feature_names']:
        ##         param_dict['feature_names'].append('visionChange')
                
        ## # Unimodal feature - fabric skin ------------------------------------
        ## if 'unimodal_fabricForce' in raw_feature_list:
        ##     fabricMag = d['fabricMagList'][idx]

        ##     unimodal_fabricForce = fabricMag

        ##     if dataSample is None: dataSample = unimodal_fabricForce
        ##     else: dataSample = np.vstack([dataSample, unimodal_fabricForce])
        ##     if 'fabricForce' not in param_dict['feature_names']:
        ##         param_dict['feature_names'].append('fabricForce')

        # ----------------------------------------------------------------
        dataList.append(dataSample)

    # Augmentation -------------------------------------------------------
    # assuming there is no currupted file    
    assert len(dataList) == nSuccess+nFailure
    successDataList = dataList[0:nSuccess]
    failureDataList = dataList[nSuccess:]
    allDataList     = successDataList + failureDataList

    # Converting data structure & cutting unnecessary part ---------------
    features         = np.swapaxes(allDataList, 0, 1)
    success_features = np.swapaxes(successDataList, 0, 1)
    failure_features = np.swapaxes(failureDataList, 0, 1)

    ## Cut data
    if cut_data is not None:
        features         = features[:,:,cut_data[0]:cut_data[1]]
        success_features = success_features[:,:,cut_data[0]:cut_data[1]]
        failure_features = failure_features[:,:,cut_data[0]:cut_data[1]]
               
    # Scaling ------------------------------------------------------------
    if isTrainingData:
        param_dict['feature_max'] = [ np.max(np.array(feature).flatten()) for feature in features ]
        param_dict['feature_min'] = [ np.min(np.array(feature).flatten()) for feature in features ]
        param_dict['feature_mu']  = [ np.mean(np.array(feature).flatten()) for feature in features ]
        param_dict['feature_std'] = [ np.std(np.array(feature).flatten()) for feature in features ]
        ## print "max: ", param_dict['feature_max']
        ## print "min: ", param_dict['feature_min']

    if scaling is True: 
        success_features = scale( success_features, param_dict['feature_min'], param_dict['feature_max'] )
        failure_features = scale( failure_features, param_dict['feature_min'], param_dict['feature_max'] )
        ## success_features = normalization( success_features, param_dict['feature_mu'], \
        ## param_dict['feature_std'] )
        ## failure_features = normalization( failure_features, param_dict['feature_mu'], \
        ## param_dict['feature_std'] )

    param_dict['feature_names'] = raw_feature_list
    param_dict['dataDim']       = dataDim
   
    return success_features, failure_features, param_dict

#-------------------------------------------------------------------------------------------------

def normalization(x, mu, std):
    new_x = copy.copy(x)
    for i in xrange(len(x)):
        new_x[i] = (x[i]-mu[i])/std[i]
    return new_x

def scale(x, x_min, x_max):
    '''
    scale data between 0 and 1
    '''
    new_x = copy.copy(x)
    for i in xrange(len(x)):
        new_x[i] = (x[i]-x_min[i])/(x_max[i]-x_min[i])
    return new_x
    

## def changeDataStructure(dataList):
##     '''
##     From nSample x dim x length to dim x nSample x length
##     or
##     From dim x nSample x length to nSample x dim x length 
##     '''
    
##     n = len(dataList)
##     m = len(dataList[0])
##     features     = []
##     for i in xrange(m):
##         feature  = []

##         for j in xrange(n):
##             try:
##                 feature.append(dataList[j][i,:])
##             except:
##                 print "Failed to cut data", j,i, np.shape(dataList[j]), dataList[j][i]
##                 print np.shape(dataList), np.shape(dataList[j]), j, i
##                 sys.exit()

##         features.append( feature )

def data_augmentation(successes, failures, nAugment=1):

    '''
    nSamples x Dim x nLength
    '''
    c_scale  = [0.8, 1.2]
    c_shift  = [-10, 10]
    c_noise  = 20.0 # constant computing noise sgd, sample_std/constant
    c_filter = []

    nDim     = len(successes[0])
    np.random.seed(1342)

    if nAugment == 0: return successes, failures

    # for each sample
    for k in xrange(2):
        
        aug_data_list = []
        
        for x in [successes, failures][k]:
            
            # x is numpy 2D array
            for n in xrange(nAugment):

                # scaling (selective dim)
                ## idx_list = np.random.randint(0, 2, size=nDim)
                ## new_x = None
                ## for i, flag in zip( range(nDim), idx_list ):
                ##     if flag == 0: temp = x[i:i+1]
                ##     else: temp = x[i:i+1] * np.random.uniform(c_scale[0], c_scale[1])

                ##     if len(np.shape(temp)) == 1: temp = np.array([temp])

                ##     if new_x is None: new_x = temp
                ##     else: new_x = np.vstack([new_x, temp])

                ## aug_data_list.append(new_x)


                ## # shifting (selective dim)
                ## idx_list = np.random.randint(0, 2, size=nDim)
                ## new_x = None
                ## for i, flag in zip( range(nDim), idx_list ):
                ##     if flag == 0:
                ##         temp = x[i:i+1]
                ##     else:
                ##         shift = np.random.random_integers(c_shift[0], c_shift[1])
                ##         if shift >= 0:
                ##             temp = np.hstack([x[i][shift:], [x[i][-1]]*shift])
                ##         else:
                ##             temp = np.hstack([[x[i][0]]*abs(shift), x[i][:-abs(shift)]])

                ##     if len(np.shape(temp)) == 1: temp = np.array([temp])

                ##     if new_x is None: new_x = temp
                ##     else: new_x = np.vstack([new_x, temp])

                ## aug_data_list.append(new_x)

                # noise (all or selectively)
                idx_list = np.random.randint(0, 2, size=nDim)
                new_x = None
                for i, flag in zip( range(nDim), idx_list ):
                    if flag == 0: temp = x[i:i+1]
                    else: temp = x[i:i+1] + np.random.normal(0.0, np.std(x[i])/c_noise, len(x[i]))

                    if len(np.shape(temp)) == 1: temp = np.array([temp])

                    if new_x is None: new_x = temp
                    else: new_x = np.vstack([new_x, temp])

                aug_data_list.append(new_x)
                

                # filtering
                
        if k==0:
            if type(successes) == list: success_aug_list = successes + aug_data_list
            else:  success_aug_list = successes.tolist() + aug_data_list
        else:
            if type(failures) == list: failure_aug_list = failures + aug_data_list
            else: failure_aug_list = failures.tolist() + aug_data_list

    ## print "data auuuuuuuuuuuugmentation"
    ## print "From : ", np.shape(successes), np.shape(failures)
    ## print "To : ", np.shape(success_aug_list), np.shape(failure_aug_list)
    
    return np.array(success_aug_list), np.array(failure_aug_list)


def get_time_window_data(subject_names, task, raw_data_path, processed_data_path, save_pkl, \
                         rf_center, local_range, downSampleSize, time_window, handFeatures, rawFeatures, \
                         cut_data, nAugment=1, renew=False):

    if os.path.isfile(save_pkl) and renew is not True:
        d = ut.load_pickle(save_pkl)
        # Time-sliding window
        new_normalTrainingData   = getTimeDelayData( d['normalTrainingData'], time_window )
        new_abnormalTrainingData = getTimeDelayData( d['abnormalTrainingData'], time_window )        
        new_normalTestData       = getTimeDelayData( d['normalTestData'], time_window )
        new_abnormalTestData     = getTimeDelayData( d['abnormalTestData'], time_window )        
        nSingleData              = len(d['normalTestData'][0][0])-time_window+1

        return new_normalTrainingData, new_normalTrainingData, new_normalTestData, new_abnormalTestData, \
          nSingleData

    # dim x sample x length
    data_dict = getDataSet(subject_names, task, raw_data_path, processed_data_path, \
                           rf_center, local_range,\
                           downSampleSize=downSampleSize, scale=1.0,\
                           ae_data=True, \
                           handFeatures=handFeatures, rawFeatures=rawFeatures, \
                           cut_data=cut_data,\
                           data_renew=renew)
    successData = data_dict['aeSuccessData']
    failureData = data_dict['aeFailureData']
                           

    # index selection
    ratio        = 0.8
    success_idx  = range(len(successData[0]))
    failure_idx  = range(len(failureData[0]))

    s_train_idx  = random.sample(success_idx, int( ratio*len(success_idx)) )
    f_train_idx  = random.sample(failure_idx, int( ratio*len(failure_idx)) )
    
    s_test_idx = [x for x in success_idx if not x in s_train_idx]
    f_test_idx = [x for x in failure_idx if not x in f_train_idx]

    # data structure: dim x sample x sequence
    normalTrainingData   = successData[:, s_train_idx, :]
    abnormalTrainingData = failureData[:, f_train_idx, :]
    normalTestData       = successData[:, s_test_idx, :]
    abnormalTestData     = failureData[:, f_test_idx, :]

    # scaling by the number of dimensions in each feature
    # nSamples x Dim x nLength
    d = {}        
    d['normalTrainingData']   = np.swapaxes(normalTrainingData, 0, 1)
    d['abnormalTrainingData'] = np.swapaxes(abnormalTrainingData, 0, 1)
    d['normalTestData']       = np.swapaxes(normalTestData, 0, 1)
    d['abnormalTestData']     = np.swapaxes(abnormalTestData, 0, 1)
    ut.save_pickle(d, save_pkl)

    # data augmentation for auto encoder
    if nAugment>0:
        normalTrainDataAug, abnormalTrainDataAug = data_augmentation(d['normalTrainingData'], \
                                                                     d['abnormalTrainingData'], nAugment)
    else:
        normalTrainDataAug   = d['normalTrainData']
        abnormalTrainDataAug = d['abnormalTrainData']


    print "======================================"
    print "nSamples x Dim x nLength"
    print "--------------------------------------"
    print "Normal Train data: ",   np.shape(d['normalTrainingData'])
    print "Abnormal Train data: ", np.shape(d['abnormalTrainingData'])
    print "Normal test data: ",    np.shape(d['normalTestData'])
    print "Abnormal test data: ",  np.shape(d['abnormalTestData'])
    print "======================================"

    # Time-sliding window
    # sample x time_window_flatten_length
    new_normalTrainingData   = getTimeDelayData( d['normalTrainingData'], time_window )
    new_abnormalTrainingData = getTimeDelayData( d['abnormalTrainingData'], time_window )
    new_normalTestData       = getTimeDelayData( d['normalTestData'], time_window )
    new_abnormalTestData     = getTimeDelayData( d['abnormalTestData'], time_window )
    nSingleData       = len(d['normalTestData'][0][0])-time_window+1

    # sample x dim
    return new_normalTrainingData, new_abnormalTrainingData, \
      new_normalTestData, new_abnormalTestData, nSingleData


def getTimeDelayData(data, time_window):
    '''
    Input size is sample x dim x length.
    Output size is sample x time_window_flatten_length.
    '''
    new_data = []
    for i in xrange(len(data)):
        for j in xrange(len(data[i][0])-time_window+1):
            new_data.append( data[i][:,j:j+time_window].flatten() )

    return np.array(new_data)

def getEstTruePositive(ll_X, ll_idx=None, nOffset=5):
    '''
    Input size is nSamples x length x HMM-induced features or 
    Input size is length x HMM-induced features
    Output is nData x HMM-induced features
    Here the input should be positive data only.
    '''

    flatten_X   = []
    flatten_idx = []
    
    if len(np.shape(ll_X))==3:
        for i in xrange(len(ll_X)):
            for j in xrange(0, len(ll_X[i])-nOffset):
                if ll_X[i][j+nOffset][0]-ll_X[i][j][0] < 0 : #and X[i][j+1][0]-X[i][j][0] < 0:
                    flatten_X   += ll_X[i][j:]
                    if ll_idx is not None: flatten_idx += ll_idx[i][j]
                    break
    elif len(np.shape(ll_X))==2:
        for j in xrange(0, len(ll_X)-nOffset):
            if ll_X[j+nOffset][0]-ll_X[j][0] < 0 : #and X[j+1][0]-X[j][0] < 0:
                if type(ll_X[j:]) is list:
                    flatten_X += ll_X[j:]
                else:
                    flatten_X += ll_X[j:].tolist()
                if ll_idx is not None: flatten_idx = ll_idx[j]
                break
    else:
        warnings.warn("Not available dimension of data X")
                
    flatten_Y = [1]*len(flatten_X)

    if ll_idx is None:
        return flatten_X, flatten_Y
    else:
        return flatten_X, flatten_Y, flatten_idx        
        
        
def flattenSample(ll_X, ll_Y, ll_idx=None, remove_fp=False):
    '''
    ll : sample x length x hmm features
    l  : sample...  x hmm features
    '''

    if remove_fp:
        l_X   = []
        l_Y   = []
        l_idx = []
        for i in xrange(len(ll_X)):
            if ll_Y[i][0] < 0:
                if type(ll_X[i]) is list:
                    l_X += ll_X[i]
                else:
                    l_X += ll_X[i].tolist()

                if type(ll_Y[i]) is list:
                    l_Y += ll_Y[i]
                else:
                    l_Y += ll_Y[i].tolist()
            else:
                X,Y = getEstTruePositive(ll_X[i])
                l_X += X
                l_Y += Y
        if len(np.shape(l_X))==1:
            warnings.warn("wrong size vector in flatten function")
            sys.exit()
                
    else:
        l_X = []
        l_Y = []
        l_idx = []
        for i in xrange(len(ll_X)):
            for j in xrange(len(ll_X[i])):
                l_X.append(ll_X[i][j])
                l_Y.append(ll_Y[i][j])
                if ll_idx is not None:
                    l_idx.append(ll_idx[i][j])

    return l_X, l_Y, l_idx
    
def flattenSampleWithWindow(ll_X, ll_Y, ll_idx=None, window=2):
    '''
    ll : sample x length x features
    l  : sample...  x features
    '''

    if window < 2:
        print "Wrong window size"
        sys.exit()

    l_X = []
    l_Y = []
    l_idx = []
    for i in xrange(len(ll_X)):
        for j in xrange(len(ll_X[i])):

            X = []
            for k in range(window,0,-1):
                if j-k < 0:
                    X += ll_X[i][0].tolist()
                else:
                    X += ll_X[i][j-k].tolist()

            l_X.append(X)
            l_Y.append(ll_Y[i][j])
            if ll_idx is not None:
                l_idx.append(ll_idx[i][j])

    return l_X, l_Y, l_idx

def sampleWithWindow(ll_X, window=2):
    '''
    ll : sample x length x hmm features
    '''
    if window < 2:
        print "Wrong window size"
        sys.exit()

    ll_X_new = []
    for i in xrange(len(ll_X)):
        l_X_new = []
        for j in xrange(len(ll_X[i])):

            X = []
            for k in range(window,0,-1):
                if j-k < 0:
                    X+= ll_X[i][0].tolist()
                else:
                    X+= ll_X[i][j-k].tolist()

            l_X_new.append(X)
        ll_X_new.append(l_X_new)

    return ll_X_new



