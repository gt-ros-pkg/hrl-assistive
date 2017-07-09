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
import hrl_lib.util as ut
from joblib import Parallel, delayed

# Private utils
from hrl_anomaly_detection.util import *
from hrl_anomaly_detection.util_viz import *
from hrl_anomaly_detection import data_manager as dm
from hrl_anomaly_detection import util as util
from hrl_execution_monitor import util as autil

# Private learners
from hrl_anomaly_detection.hmm import learning_hmm as hmm
import hrl_anomaly_detection.classifiers.classifier as cf
import hrl_anomaly_detection.data_viz as dv
from hrl_anomaly_detection.vae import keras_models as km

# visualization
import matplotlib
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


def lstm_test(subject_names, task_name, raw_data_path, processed_data_path, param_dict, plot=False):
    ## Parameters
    data_dict  = param_dict['data_param']
    data_renew = data_dict['renew']
    
    #------------------------------------------
    if os.path.isdir(processed_data_path) is False:
        os.system('mkdir -p '+processed_data_path)

    crossVal_pkl = os.path.join(processed_data_path, 'cv_'+task_name+'.pkl')    
    if os.path.isfile(crossVal_pkl) and data_renew is False:
        print "CV data exists and no renew"
        d = ut.load_pickle(crossVal_pkl)         
    else:
        '''
        Use augmented data? if nAugment is 0, then aug_successData = successData
        '''        
        d = dm.getDataLOPO(subject_names, task_name, raw_data_path, \
                           processed_data_path, data_dict['rf_center'], data_dict['local_range'],\
                           downSampleSize=data_dict['downSampleSize'],\
                           handFeatures=data_dict['isolationFeatures'], \
                           cut_data=data_dict['cut_data'], \
                           data_renew=data_renew, max_time=data_dict['max_time'])

        d['successData'], d['failureData'], d['success_files'], d['failure_files'], d['kFoldList'] \
          = dm.LOPO_data_index(d['successDataList'], d['failureDataList'],\
                               d['successFileList'], d['failureFileList'])

        ut.save_pickle(d, crossVal_pkl)


    # select feature for detection
    feature_list = []
    for feature in param_dict['data_param']['handFeatures']:
        idx = [ i for i, x in enumerate(param_dict['data_param']['isolationFeatures']) if feature == x][0]
        feature_list.append(idx)
    d['successData']    = d['successData'][feature_list]
    d['failureData']    = d['failureData'][feature_list]

    ## subjects = ['Andrew', 'Britteney', 'Joshua', 'Jun', 'Kihan', 'Lichard', 'Shingshing', 'Sid', 'Tao']
    ## raw_data_path  = os.path.expanduser('~')+'/hrl_file_server/dpark_data/anomaly/RAW_DATA/CORL2017/'
    ## td1 = get_ext_data(subjects, task_name, raw_data_path, save_data_path, param_dict,
    ##                   init_param_dict=d['param_dict'], id_num=0)

    ## subjects = ['ari', 'park', 'jina', 'linda', 'sai', 'hyun']
    ## raw_data_path  = os.path.expanduser('~')+'/hrl_file_server/dpark_data/anomaly/RAW_DATA/ICRA2017/'
    ## td2 = get_ext_data(subjects, task_name, raw_data_path, save_data_path, param_dict,
    ##                   init_param_dict=d['param_dict'], id_num=1)

    # Parameters
    nDim = len(d['successData'])

    # split data
    # HMM-induced vector with LOPO
    for idx, (normalTrainIdx, abnormalTrainIdx, normalTestIdx, abnormalTestIdx) \
      in enumerate(d['kFoldList']):
        if idx == 0: continue

        # dim x sample x length
        ## normalTrainData   = copy.deepcopy(d['successData'][:, normalTrainIdx, :])
        ## abnormalTrainData = copy.deepcopy(d['failureData'][:, abnormalTrainIdx, :])
        ## normalTestData    = copy.deepcopy(d['successData'][:, normalTestIdx, :]) 
        ## abnormalTestData  = copy.deepcopy(d['failureData'][:, abnormalTestIdx, :])
        ## normalTrainData   = np.hstack([normalTrainData, copy.deepcopy(td1['successData']), copy.deepcopy(td2['successData'])])
        ## abnormalTrainData = np.hstack([abnormalTrainData, copy.deepcopy(td1['failureData']), copy.deepcopy(td2['failureData'])])


        ## normalData   = np.hstack([copy.deepcopy(d['successData']), copy.deepcopy(td1['successData']), \
        ##                           copy.deepcopy(td2['successData'])])
        ## abnormalData = np.hstack([copy.deepcopy(d['failureData']), copy.deepcopy(td1['failureData']), \
        ##                           copy.deepcopy(td2['failureData'])])
        normalData   = copy.deepcopy(d['successData'])
        abnormalData = copy.deepcopy(d['failureData'])
        # ------------------------------------------------------------------------------------------
        
        trainData, testData, window_size, raw_data, raw_data_ft = \
          get_batch_data(normalData, abnormalData, win=True)
        (normalTrainData, abnormalTrainData, normalTestData, abnormalTestData) = raw_data
        (normalTrainData_ft, abnormalTrainData_ft, normalTestData_ft, abnormalTestData_ft) = raw_data_ft
        batch_size  = 16
         
        weights_path = os.path.join(save_data_path,'tmp_weights_'+str(idx)+'.h5')
        ## weights_path = os.path.join(save_data_path,'tmp_fine_weights_'+str(idx)+'.h5')
        vae_mean   = None
        vae_logvar = None
        enc_z_mean = enc_z_std = None

        # ------------------------------------------------------------------------------------------
        ## autoencoder, enc_z_mean, enc_z_std, generator = km.lstm_vae2(trainData, testData, weights_path,
        ##                                                             patience=5, batch_size=batch_size)
        autoencoder, vae_mean, vae_logvar, enc_z_mean, enc_z_std, generator = \
          km.lstm_vae3(trainData, testData, weights_path, patience=3, batch_size=batch_size,
                       steps_per_epoch=2048)
        #autoencoder, vae_mean, vae_logvar, enc_z_mean, enc_z_std, generator = \
        #  km.lstm_vae4(trainData, testData, weights_path, patience=3, batch_size=batch_size)
        ## autoencoder = km.lstm_ae(trainData, testData, weights_path,
        ##                          patience=5, batch_size=batch_size)

        # ------------------------------------------------------------------------------------------
        ## # Fine tuning
        ## normalData   = copy.deepcopy(d['successData'])
        ## abnormalData = copy.deepcopy(d['failureData']) 
        ## trainData, testData, window_size, raw_data, raw_data_ft = get_batch_data(normalData, abnormalData)
        ## (normalTrainData, abnormalTrainData, normalTestData, abnormalTestData) = raw_data
        ## (normalTrainData_ft, abnormalTrainData_ft, normalTestData_ft, abnormalTestData_ft) = raw_data_ft
       
        ## save_weights_path = os.path.join(save_data_path,'tmp_fine_weights_'+str(idx)+'.h5')
        ## autoencoder, enc_z_mean, enc_z_std, generator = km.lstm_vae(trainData, testData, weights_path,
        ##                                                             fine_tuning=True, \
        ##                                                             save_weights_file=save_weights_path)


        if True and False:
            if True:
                # get optimized alpha
                save_pkl = os.path.join(save_data_path, 'tmp_data.pkl')
                alpha = get_optimal_alpha(autoencoder, vae_mean, vae_logvar, enc_z_mean, enc_z_std,
                                          generator, normalTrainData, window_size,\
                                          save_pkl=save_pkl)
            else:
                alpha = np.array([1.0]*nDim)/float(nDim)
                ## alpha = np.array([0.0]*nDim)/float(nDim)
                ## alpha[0] = 1.0
            
            save_pkl = os.path.join(save_data_path, 'tmp_test_scores.pkl')
            anomaly_detection(autoencoder, vae_mean, vae_logvar, enc_z_mean, enc_z_std, generator,
                              normalTestData, abnormalTestData, \
                              window_size, alpha, save_pkl=save_pkl)

        
        if plot:
            if enc_z_mean is not None:
                # display a 2D plot of classes in the latent space
                x_n_encoded  = enc_z_mean.predict(normalTrainData_ft)
                x_ab_encoded = enc_z_mean.predict(abnormalTrainData_ft)

                plt.figure(figsize=(6, 6))
                plt.plot(x_n_encoded[:, 0], x_n_encoded[:, 1], '.b', ms=5, mec='b', mew=0)
                plt.plot(x_ab_encoded[:, 0], x_ab_encoded[:, 1], '.r', ms=5, mec='r', mew=0)
                plt.show()
                
            if vae_mean is None: vae_mean = autoencoder


            # display generated data
            for i in xrange(len(normalTrainData)):
                if window_size is not None:
                    x = sampleWithWindow(normalTrainData[i:i+1], window=window_size)
                    x = np.array(x)


                    x_true = []
                    x_pred = []
                    for j in xrange(len(x)):
                        x_new = vae_mean.predict(x[j:j+1])
                        x_true.append(x[j][-1])
                        x_pred.append(x_new[0,-1])


                    fig = plt.figure(figsize=(6, 6))
                    for k in xrange(len(x_true[0])):
                        fig.add_subplot(6,2,k+1)
                        plt.plot(np.array(x_true)[:,k], '-b')
                        plt.plot(np.array(x_pred)[:,k], '-r')
                        plt.ylim([-0.1,1.1])
                    plt.show()
                        
                else:
                    x = normalTrainData[i:i+1]
                    x_new = autoencoder.predict(x)
                    print np.shape(x), np.shape(x[0])

                    for j in xrange(len(x)):
                        fig = plt.figure(figsize=(6, 6))
                        for k in xrange(len(x[j][0])):
                            fig.add_subplot(6,2,k+1)
                            ## fig.add_subplot(100*len(x[j][0])+10+k+1)
                            plt.plot(np.array(x)[j,:,k], '-b')
                            plt.plot(np.array(x_new)[j,:,k], '-r')
                            plt.ylim([-0.1,1.1])
                        plt.show()
        
        return
    
    # flatten data window 1
    #def train_vae_classifier() 

def gen_data(subject_names, task_name, raw_data_path, processed_data_path, param_dict):
    ## Parameters
    # data
    data_dict  = param_dict['data_param']
    data_renew = data_dict['renew']

    

    # Adaptation
    ## ADT_dict = param_dict['ADT']

    # parameters
    startIdx    = 4
    method_list = ROC_dict['methods'] 
    nPoints     = ROC_dict['nPoints']
    
    #------------------------------------------

    if os.path.isdir(processed_data_path) is False:
        os.system('mkdir -p '+processed_data_path)

    crossVal_pkl = os.path.join(processed_data_path, 'cv_'+task_name+'.pkl')
    
    if os.path.isfile(crossVal_pkl) and data_renew is False:
        print "CV data exists and no renew"
        d = ut.load_pickle(crossVal_pkl)         
    else:
        '''
        Use augmented data? if nAugment is 0, then aug_successData = successData
        '''        
        d = dm.getDataLOPO(subject_names, task_name, raw_data_path, \
                           processed_data_path, data_dict['rf_center'], data_dict['local_range'],\
                           downSampleSize=data_dict['downSampleSize'],\
                           handFeatures=data_dict['isolationFeatures'], \
                           cut_data=data_dict['cut_data'], \
                           data_renew=data_renew, max_time=data_dict['max_time'])

        d['successData'], d['failureData'], d['success_files'], d['failure_files'], d['kFoldList'] \
          = dm.LOPO_data_index(d['successDataList'], d['failureDataList'],\
                               d['successFileList'], d['failureFileList'])

        ut.save_pickle(d, crossVal_pkl)

    # select feature for detection
    feature_list = []
    for feature in param_dict['data_param']['handFeatures']:
        idx = [ i for i, x in enumerate(param_dict['data_param']['isolationFeatures']) if feature == x][0]
        feature_list.append(idx)
    print np.shape(d['successData'])

    d['successData'] = d['successData'][feature_list]
    d['failureData'] = d['failureData'][feature_list]
    print np.shape(d['successData'])

    td = get_ext_data(subject_names, task_name, raw_data_path, save_data_path, param_dict,
                      init_param_dict=d['param_dict'])

    # ------------------------------------------------------------------------------
    # split data
    # HMM-induced vector with LOPO
    for idx, (normalTrainIdx, abnormalTrainIdx, normalTestIdx, abnormalTestIdx) \
      in enumerate(d['kFoldList']):

        # dim x sample x length
        normalTrainData   = copy.deepcopy(d['successData'][:, normalTrainIdx, :])
        abnormalTrainData = copy.deepcopy(d['failureData'][:, abnormalTrainIdx, :])
        normalTestData    = copy.deepcopy(d['successData'][:, normalTestIdx, :]) 
        abnormalTestData  = copy.deepcopy(d['failureData'][:, abnormalTestIdx, :])

        normalTrainData   = np.hstack([normalTrainData, copy.deepcopy(td['successData'])])
        abnormalTrainData = np.hstack([abnormalTrainData, copy.deepcopy(td['failureData'])])
        batch_size = len(normalTrainData[0,0])

        normalTrainData   = np.swapaxes(normalTrainData, 0,1 )
        normalTrainData   = np.swapaxes(normalTrainData, 1,2 )
        abnormalTrainData = np.swapaxes(abnormalTrainData, 0,1 )
        abnormalTrainData = np.swapaxes(abnormalTrainData, 1,2 )

        normalTestData   = np.swapaxes(normalTestData, 0,1 )
        normalTestData   = np.swapaxes(normalTestData, 1,2 )
        abnormalTestData = np.swapaxes(abnormalTestData, 0,1 )
        abnormalTestData = np.swapaxes(abnormalTestData, 1,2 )


        # flatten the data (sample, length, dim)
        trainData = np.vstack([normalTrainData, abnormalTrainData])
        trainData = trainData.reshape(len(trainData)*len(trainData[0]), len(trainData[0,0]))

        testData = normalTestData.reshape(len(normalTestData)*len(normalTestData[0]), len(normalTestData[0,0]))
        print np.shape(trainData), np.shape(testData)

        ## print np.amin(trainData, axis=0)
        ## print np.amax(trainData, axis=0)
        ## sys.exit()

        if True:
            # get window data
            # sample x length x dim => sample x length x (dim x window)
            trainData = dm.sampleWithWindow(trainData, window=20)
            testData = dm.sampleWithWindow(testData, window=20)

        weights_path = os.path.join(save_data_path,'tmp_weights_'+str(idx)+'.h5')        
        vae = km.variational_autoencoder(trainData, testData, weights_path, batch_size=batch_size)


        return
    


def get_ext_data(subjects, task_name, raw_data_path, processed_data_path, param_dict,
                 init_param_dict=None, id_num=0):
    ## Parameters
    # data
    data_dict  = param_dict['data_param']
    data_renew = data_dict['renew']
    
    #------------------------------------------

    if os.path.isdir(processed_data_path) is False:
        os.system('mkdir -p '+processed_data_path)

    if init_param_dict is None:
        crossVal_pkl = os.path.join(processed_data_path, 'cv_'+task_name+'.pkl')
        print "CV data exists and no renew"
        d = ut.load_pickle(crossVal_pkl)
        init_param_dict = d['param_dict']
        

    #------------------------------------------
    
    crossVal_pkl = os.path.join(processed_data_path, 'cv_td_'+task_name+'_'+str(id_num)+'.pkl')
    if os.path.isfile(crossVal_pkl) and data_renew is False and False:
        print "CV data exists and no renew"
        td = ut.load_pickle(crossVal_pkl)
    else:
        # Extract data from designated location
        td = dm.getDataLOPO(subjects, task_name, raw_data_path, save_data_path,\
                            downSampleSize=data_dict['downSampleSize'],\
                            init_param_dict=init_param_dict,\
                            handFeatures=data_dict['isolationFeatures'], \
                            cut_data=data_dict['cut_data'],\
                            data_renew=data_renew, max_time=data_dict['max_time'],
                            pkl_prefix='tgt_', depth=True)

        td['successData'], td['failureData'], td['success_files'], td['failure_files'], td['kFoldList'] \
          = dm.LOPO_data_index(td['successDataList'], td['failureDataList'],\
                               td['successFileList'], td['failureFileList'])

        ut.save_pickle(td, crossVal_pkl)


    #------------------------------------------
    # select feature for detection
    feature_list = []
    for feature in param_dict['data_param']['handFeatures']:
        idx = [ i for i, x in enumerate(param_dict['data_param']['isolationFeatures']) if feature == x][0]
        feature_list.append(idx)
    print np.shape(td['successData'])

    td['successData']    = td['successData'][feature_list]
    td['failureData']    = td['failureData'][feature_list]
    print np.shape(td['successData'])

    return td




def sampleWithWindow(X, window=5):
    '''
    X : sample x length x features
    return: (sample x length-window+1) x features
    '''
    if window < 2:
        print "Wrong window size"
        sys.exit()

    X_new = []
    for i in xrange(len(X)): # per sample
        for j in xrange(len(X[i])-window+1): # per time
            X_new.append( X[i][j:j+window].tolist() ) # per sample
    
    return X_new


def get_batch_data(normalData, abnormalData, win=False):
    
    # dim x sample x length => sample x length x dim
    normalData   = np.swapaxes(normalData, 0,1 )
    normalData   = np.swapaxes(normalData, 1,2 )
    abnormalData = np.swapaxes(abnormalData, 0,1 )
    abnormalData = np.swapaxes(abnormalData, 1,2 )

    np.random.shuffle(normalData)
    np.random.shuffle(abnormalData)
    print np.shape(normalData), np.shape(abnormalData)

    ratio=0.8
    normalTrainData, normalTestData\
    = normalData[:int(len(normalData)*ratio)],normalData[int(len(normalData)*ratio):]
    abnormalTrainData, abnormalTestData\
    = abnormalData[:int(len(abnormalData)*ratio)],abnormalData[int(len(abnormalData)*ratio):]


    ## # dim x sample x length => sample x length x dim
    ## normalTrainData   = np.swapaxes(normalTrainData, 0,1 )
    ## normalTrainData   = np.swapaxes(normalTrainData, 1,2 )
    ## abnormalTrainData = np.swapaxes(abnormalTrainData, 0,1 )
    ## abnormalTrainData = np.swapaxes(abnormalTrainData, 1,2 )

    ## # dim x sample x length => sample x length x dim
    ## normalTestData   = np.swapaxes(normalTestData, 0,1 )
    ## normalTestData   = np.swapaxes(normalTestData, 1,2 )
    ## abnormalTestData = np.swapaxes(abnormalTestData, 0,1 )
    ## abnormalTestData = np.swapaxes(abnormalTestData, 1,2 )

    # normalization => (sample x dim) ----------------------------------
    from sklearn import preprocessing
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

    normalTrainData_scaled   = scaler.fit_transform(normalTrainData.reshape(-1,len(normalTrainData[0][0])))
    abnormalTrainData_scaled = scaler.transform(abnormalTrainData.reshape(-1,len(abnormalTrainData[0][0])))
    normalTestData_scaled    = scaler.transform(normalTestData.reshape(-1,len(normalTestData[0][0])))
    abnormalTestData_scaled  = scaler.transform(abnormalTestData.reshape(-1,len(abnormalTestData[0][0])))

    # reshape
    normalTrainData   = normalTrainData_scaled.reshape(np.shape(normalTrainData))
    abnormalTrainData = abnormalTrainData_scaled.reshape(np.shape(abnormalTrainData))
    normalTestData   = normalTestData_scaled.reshape(np.shape(normalTestData))
    abnormalTestData  = abnormalTestData_scaled.reshape(np.shape(abnormalTestData))
    #----------------------------------------------------------------------

    ## for i in xrange(len(normalTrainData[0][0])):
    ##     print np.amin(normalTrainData[:,:,i]), np.amax(normalTrainData[:,:,i]), np.amin(normalTestData[:,:,i]), np.amax(normalTestData[:,:,i])

    ## fig = plt.figure(figsize=(6, 6))
    ## normalData   = normalTrainData
    ## for i in xrange(len(normalData[0][0])):
    ##     fig.add_subplot(6,2,i+1)
    ##     for j in xrange(len(normalData)):
    ##         if j>20: break
    ##         plt.plot(np.array(normalData)[j][:,i], '-b')
    ## normalData   = normalTestData
    ## for i in xrange(len(normalData[0][0])):
    ##     fig.add_subplot(6,2,i+1)
    ##     for j in xrange(len(normalData)):
    ##         if j>20: break
    ##         plt.plot(np.array(normalData)[j][:,i], '-r')
    ## plt.show()
    ## sys.exit()


    if win:
        window_size = 20
        
        # get window data
        # sample x length x dim => (sample x length) x dim
        normalTrainData_ft   = sampleWithWindow(normalTrainData, window=window_size)
        abnormalTrainData_ft = sampleWithWindow(abnormalTrainData, window=window_size)
        normalTestData_ft    = sampleWithWindow(normalTestData, window=window_size)
        abnormalTestData_ft  = sampleWithWindow(abnormalTestData, window=window_size)

        # flatten the data (sample, length, dim)
        ## trainData = [np.vstack([normalTrainData, abnormalTrainData]),
        ##              [0]*len(normalTrainData)+[1]*len(abnormalTrainData)]
        ## testData  = [np.vstack([normalTestData, abnormalTestData]),
        ##              [0]*len(normalTestData)+[1]*len(abnormalTestData)]
        trainData_win = [normalTrainData_ft, [0]*len(normalTrainData_ft)]
        testData_win  = [normalTestData_ft, [0]*len(normalTestData_ft)]
    else:
        window_size = None
        normalTrainData_ft = normalTrainData
        abnormalTrainData_ft = abnormalTrainData
        normalTestData_ft = normalTestData
        abnormalTestData_ft = abnormalTestData
        trainData_win = [normalTrainData, [0]*len(normalTrainData)]
        testData_win  = [normalTestData, [0]*len(normalTestData)]

    raw_data = (normalTrainData, abnormalTrainData, normalTestData, abnormalTestData)
    raw_data_ft = (normalTrainData_ft, abnormalTrainData_ft, normalTestData_ft, abnormalTestData_ft)
    return trainData_win, testData_win, window_size, raw_data, raw_data_ft


def get_optimal_alpha(vae, vae_mean, vae_logvar, enc_z_mean, enc_z_std, generator,
                      normalTrainData, window_size, save_pkl=None):

    
    nDim    = len(normalTrainData[0,0])
    nSample = len(normalTrainData)
    nSubSample = 1
    p_ll = [[] for i in xrange(nDim) ]

    if os.path.isfile(save_pkl):
        d = ut.load_pickle(save_pkl)
        nSample    = d['nSample']
        nSubSample = d['nSubSample']
        p_ll       = d['p_ll']
    else:
        for i in xrange(len(normalTrainData)):
            print "sample: ", i+1, " out of ", len(normalTrainData), np.shape(p_ll)

            if window_size>0: x = sampleWithWindow(normalTrainData[i:i+1], window=window_size)
            else:             x = normalTrainData[i:i+1]
                
            #nSubSample = len(z_mean)

            p_ll = None
            for j in xrange(len(x)): # per window

                # sampling based method ----------------------------------------
                ## z_mean = enc_z_mean.predict(x[j:j+1]) # 1 x 2 (or 1 x (nwindow x 2))
                ## z_std  = enc_z_std.predict(x[j:j+1]) *100.0
                
                ## std  = [val if val > 0 else 1e-5 for val in z_std[j]]
                ## L = []
                ## for k in xrange(len(z_mean[j])):
                ##     L.append(np.random.normal(z_mean[j][k], std[k],nSample))
                ## L = np.array(L)
                ## L = np.swapaxes(L,0,1) # sample x z_dim

                ## x_rnd = generator.predict(L) # sample x (window_)length x dim
                ## x_rnd = np.swapaxes(x_rnd, 0, 2)

                ## x_mean = np.mean(x_rnd.reshape(len(x_rnd),-1), axis=1 )
                ## x_std  = np.std(x_rnd.reshape(len(x_rnd),-1), axis=1 )


                #---------------------------------------------------------------
                # prediction based method
                x_mean   = vae_mean.predict(x[j:j+1])
                x_logvar = vae_logvar.predict(x[j:j+1])
                x_std    = np.exp(x_logvar/2.0)

                x_mean = np.swapaxes(np.squeeze(x_mean), 0, 1)
                x_std  = np.swapaxes(np.squeeze(x_std), 0, 1)


                #---------------------------------------------------------------
                # anomaly score
                p_l     = []
                for k in xrange(len(x_mean)): # per dim
                    p = []
                    for l in xrange(len(x_mean[0])): # per length
                        p.append(scipy.stats.norm(x_mean[k][l], x_std[k][l]).pdf(x[j,l,k])) # length

                    p = [val if not np.isinf(val).any() and not np.isnan(val).any() and val > 0
                         else 1e-50 for val in p]
                    p_l.append(p) # dim x length

                if p_ll is None:
                    p_ll = np.log(np.array(p_l))
                else:
                    p_ll = np.hstack([p_ll, np.log(np.array(p_l)+1e-10)])
                ## p_ll.append(np.log(np.array(p_l)+1e-10))                    

                print np.shape(p_ll)

                # find min idx
                ## idx = np.argmin(p_l)%len(x[0])            
                ## for k in xrange(len(x_mean)): # per dim
                ##     p_ll[k].append( np.log(p_l[k][idx]) )
                                
        d = {'p_ll': p_ll, 'nSample': nSample, 'nSubSample': nSubSample }
        ut.save_pickle(d, save_pkl)


    print "p_ll: ", np.shape(p_ll)
    
    def score_func(X, args):
        '''
        X      : dim
        args[0]: dim
        '''
        return -np.sum(X.dot(args)) +10.0*np.sum( X**2 )
    

    def const(X):
        return np.sum(X)-1.0
        

    from scipy.optimize import minimize
    x0   = np.array([1]*nDim)/float(nDim)
    d    = p_ll #np.sum(p_ll, axis=1)/float(nSample)/float(nSubSample) # sample x dim x length
    bnds = [[0.01,0.99] for i in xrange(nDim) ]
    res  = minimize(score_func, x0, args=(d), method='SLSQP', tol=1e-6, bounds=bnds,
                    constraints={'type':'eq', 'fun': const}, options={'disp': False})
    print res
    
    return res.x

def anomaly_detection(vae, vae_mean, vae_logvar, enc_z_mean, enc_z_std, generator,
                      normalTestData, abnormalTestData, window_size, \
                      alpha, save_pkl=None):

    if os.path.isfile(save_pkl) and False:
        d = ut.load_pickle(save_pkl)
        scores_n = d['scores_n']
        scores_a = d['scores_a']
    else:
        x_dim = len(normalTestData[0][0])

        def get_anomaly_score(X, window_size, alpha, nSample=1000):

            scores = []
            for i in xrange(len(X)):
                print "sample: ", i+1, " out of ", len(X)
                np.random.seed(3334 + i)

                if window_size>0: x = sampleWithWindow(X[i:i+1], window=window_size)
                else:             x = X[i:i+1]

                s = []
                for j in xrange(len(x)): # per window


                    # sampling based method ----------------------------------------
                    ## z_mean = enc_z_mean.predict(x[j:j+1]) # 1 x 2 (or 1 x (nwindow x 2))
                    ## z_std  = enc_z_std.predict(x[j:j+1]) *100.0

                    ## nSample=1000
                    ## ## print z_mean
                    ## ## print z_std
                    
                    ## std  = [val if val > 0 else 1e-5 for val in z_std]
                    ## L = []
                    ## for k in xrange(len(z_mean[j])): # per dim
                    ##     L.append(np.random.normal(z_mean[k], std[k],nSample))
                    ## L = np.swapaxes(np.array(L),0,1) # nSample x z_dim

                    ## x_rnd = generator.predict(L) # sample x (window_)length x dim
                    ## x_rnd = np.swapaxes(x_rnd, 0, 2) # dim x (window_)length x nSample

                    ## # time wise
                    ## x_mean = []; x_std = []
                    ## for k in xrange(len(x_rnd)): # per dim
                    ##     x_mean.append( np.mean(x_rnd[k], axis=1) )
                    ##     x_std.append( np.std(x_rnd[k], axis=1) )

                    #---------------------------------------------------------------
                    # prediction based method
                    x_mean   = vae_mean.predict(x[j:j+1])
                    x_logvar = vae_logvar.predict(x[j:j+1])
                    x_std    = np.exp(x_logvar/2.0)

                    x_mean = np.swapaxes(np.squeeze(x_mean), 0, 1)
                    x_std  = np.swapaxes(np.squeeze(x_std), 0, 1)

                    #---------------------------------------------------------------

                    # temp
                    ## fig = plt.figure()
                    ## for k in xrange(len(x_mean)): # per dim                    
                    ##     print x_std[k]
                    ##     fig.add_subplot(len(x_mean),1,k+1)
                    ##     plt.plot(x_mean[k], '-b')
                    ##     plt.plot(x_mean[k]+0.2*x_std[k], '--b')
                    ##     plt.plot(x_mean[k]-0.2*x_std[k], '--b')
                    ##     plt.plot(x[j,:,k], '-r')
                    ## plt.show()
                    

                    # anomaly score
                    p_l     = []
                    for k in xrange(len(x_mean)): # per dim
                        p = []
                        for l in xrange(len(x_mean[0])): # per length
                            p.append(scipy.stats.norm(x_mean[k][l], x_std[k][l]).pdf(x[j,l,k])) # length

                        p = [val if not np.isinf(val).any() and not np.isnan(val).any() and val > 0
                             else 1e-50 for val in p]
                        p_l.append(p) # dim x length

                    # find min 
                    s.append( np.amin(alpha.dot( np.log(np.array(p_l)) )) )
                    #s.append( np.mean(alpha.dot( np.log(np.array(p_l)) )) )

                scores.append(s)
            return scores

        scores_n = get_anomaly_score(normalTestData, window_size, alpha )
        scores_a = get_anomaly_score(abnormalTestData, window_size, alpha )
        

        d = {}
        d['scores_n'] = scores_n
        d['scores_a'] = scores_a
        ut.save_pickle(d, save_pkl)


    ths_l = -np.logspace(-1,2.9,40)
    tpr_l = []
    fpr_l = []

    for ths in ths_l:
        tp_l = []
        fp_l = []
        tn_l = []
        fn_l = []
        for s in scores_n:
            for i in xrange(len(s)):
                if s[i]<ths:
                    fp_l.append(1)
                    break
                elif i == len(s)-1:
                    tn_l.append(1)

        for s in scores_a:
            for i in xrange(len(s)):
                if s[i]<ths:
                    tp_l.append(1)
                    break
                elif i == len(s)-1:
                    fn_l.append(1)

        tpr_l.append( float(np.sum(tp_l))/float(np.sum(tp_l)+np.sum(fn_l))*100.0 )
        fpr_l.append( float(np.sum(fp_l))/float(np.sum(fp_l)+np.sum(tn_l))*100.0 )
    

    e_n_l  = [val[-1] for val in scores_n if val != np.log(1e-50) ]
    e_ab_l = [val[-1] for val in scores_a if val != np.log(1e-50) ]
    print np.mean(e_n_l), np.std(e_n_l)
    print np.mean(e_ab_l), np.std(e_ab_l)
    print "acc ", float(np.sum(tp_l)+np.sum(tn_l))/float(np.sum(tp_l+fp_l+tn_l+fn_l))

    print ths_l
    print tpr_l
    print fpr_l

    from sklearn import metrics 
    print "roc: ", metrics.auc(fpr_l, tpr_l, True)

    fig = plt.figure(figsize=(12,6))
    fig.add_subplot(1,2,1)    
    plt.plot(fpr_l, tpr_l, '-*b', ms=5, mec='b')
    
    fig.add_subplot(1,2,2)    
    plt.plot(e_n_l, '*b', ms=5, mec='b')
    plt.plot(e_ab_l, '*r', ms=5, mec='r')
    plt.show()

    return 



def feature_plot(subject_names, task_name, raw_data_path, processed_data_path, param_dict):
    ## Parameters
    data_dict  = param_dict['data_param']
    data_renew = data_dict['renew']
    
    #------------------------------------------
    if os.path.isdir(processed_data_path) is False:
        os.system('mkdir -p '+processed_data_path)

    crossVal_pkl = os.path.join(processed_data_path, 'cv_'+task_name+'.pkl')    
    if os.path.isfile(crossVal_pkl) and data_renew is False:
        print "CV data exists and no renew"
        d = ut.load_pickle(crossVal_pkl)         
    else:
        '''
        Use augmented data? if nAugment is 0, then aug_successData = successData
        '''        
        d = dm.getDataLOPO(subject_names, task_name, raw_data_path, \
                           processed_data_path, data_dict['rf_center'], data_dict['local_range'],\
                           downSampleSize=data_dict['downSampleSize'],\
                           handFeatures=data_dict['isolationFeatures'], \
                           cut_data=data_dict['cut_data'], \
                           data_renew=data_renew, max_time=data_dict['max_time'])

        d['successData'], d['failureData'], d['success_files'], d['failure_files'], d['kFoldList'] \
          = dm.LOPO_data_index(d['successDataList'], d['failureDataList'],\
                               d['successFileList'], d['failureFileList'])

        ut.save_pickle(d, crossVal_pkl)


    # select feature for detection
    feature_list = []
    for feature in param_dict['data_param']['handFeatures']:
        idx = [ i for i, x in enumerate(param_dict['data_param']['isolationFeatures']) if feature == x][0]
        feature_list.append(idx)
    d['successData']    = d['successData'][feature_list]
    d['failureData']    = d['failureData'][feature_list]

    td = get_ext_data(subject_names, task_name, raw_data_path, save_data_path, param_dict,
                      init_param_dict=d['param_dict'])
    

    ## normalData   = np.hstack([copy.deepcopy(d['successData']), copy.deepcopy(td['successData'])])
    ## abnormalTrainData = np.hstack([abnormalTrainData, copy.deepcopy(td['failureData'])])

    fig = plt.figure(figsize=(6, 6))
    normalData   = d['successData']
    for i in xrange(len(normalData)):
        fig.add_subplot(6,2,i+1)
        for j in xrange(len(normalData[i])):
            if j>20: break
            plt.plot(np.array(normalData)[i][j], '-b')
        #plt.plot(np.array(x_new)[j,:,k], '-r')
        ## plt.ylim([0,1.0])
    normalData   = td['successData']
    for i in xrange(len(normalData)):
        fig.add_subplot(6,2,i+1)
        for j in xrange(len(normalData[i])):
            if j < 20: continue
            if j > 50: break
            plt.plot(np.array(normalData)[i][j], '-r')
    plt.show()




if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    util.initialiseOptParser(p)
    
    p.add_option('--gen_data', '--gd', action='store_true', dest='gen_data',
                 default=False, help='Generate data.')
    p.add_option('--ext_data', '--ed', action='store_true', dest='extra_data',
                 default=False, help='Add extra data.')
    p.add_option('--preprocess', '--p', action='store_true', dest='preprocessing',
                 default=False, help='Preprocess')
    p.add_option('--lstm_test', '--lt', action='store_true', dest='lstm_test',
                 default=False, help='Generate data.')
    
    opt, args = p.parse_args()

    #---------------------------------------------------------------------------           
    # Run evaluation
    #---------------------------------------------------------------------------           
    rf_center   = 'kinEEPos'        
    scale       = 1.0
    local_range = 10.0
    nPoints     = 40 #None

    from hrl_anomaly_detection.vae.vae_params import *
    raw_data_path, save_data_path, param_dict = getParams(opt.task, opt.bDataRenew, \
                                                          opt.bHMMRenew, opt.bCLFRenew, opt.dim,\
                                                          rf_center, local_range, nPoints=nPoints)
    if opt.bNoUpdate: param_dict['ROC']['update_list'] = []
    # Mikako - bad camera
    # s1 - kaci - before camera calibration
    subjects = ['s2', 's3','s4','s5', 's6','s7','s8', 's9']

    if os.uname()[1] == 'monty1':
        save_data_path = os.path.expanduser('~')+\
          '/hrl_file_server/dpark_data/anomaly/ICRA2018/'+opt.task+'_data_lstm'
    else:
        save_data_path = os.path.expanduser('~')+\
          '/hrl_file_server/dpark_data/anomaly/TCDS2017/'+opt.task+'_data_adaptation2'


    ## param_dict['data_param']['handFeatures'] = ['unimodal_kinVel',\
    ##                                             'unimodal_kinJntEff_1',\
    ##                                             'unimodal_ftForce_zero',\
    ##                                             'unimodal_ftForce_integ',\
    ##                                             'unimodal_kinEEChange',\
    ##                                             'unimodal_kinDesEEChange',\
    ##                                             'crossmodal_landmarkEEDist', \
    ##                                             'unimodal_audioWristRMS',\
    ##                                             'unimodal_fabricForce',\
    ##                                             'unimodal_landmarkDist',\
    ##                                             'crossmodal_landmarkEEAng']

    param_dict['data_param']['handFeatures'] = ['unimodal_audioWristRMS',  \
                                                'unimodal_kinJntEff_1',\
                                                'unimodal_ftForce_integ',\
                                                'crossmodal_landmarkEEDist']


    if opt.gen_data:
        if opt.bNoUpdate: param_dict['ROC']['update_list'] = []        
        gen_data(subjects, opt.task, raw_data_path, save_data_path, param_dict)
        
    elif opt.extra_data:
        
        if opt.bNoUpdate: param_dict['ROC']['update_list'] = []        
        get_ext_data(subjects, opt.task, raw_data_path, save_data_path, param_dict)
        

    elif opt.preprocessing:
        src_pkl = os.path.join(save_data_path, 'isol_data.pkl')
        from hrl_execution_monitor import preprocess as pp
        pp.preprocess_data(src_pkl, save_data_path, img_scale=0.25, nb_classes=12,
                            img_feature_type='vgg', nFold=nFold)

    elif opt.lstm_test:
        lstm_test(subjects, opt.task, raw_data_path, save_data_path, param_dict, plot=not opt.bNoPlot)

    elif opt.bFeaturePlot:
        
        feature_plot(subjects, opt.task, raw_data_path, save_data_path, param_dict)
