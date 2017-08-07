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

import os
import numpy as np
import hrl_lib.util as ut

def sampleWithWindow(X, window=5):
    '''
    X : sample x length x features
    return: (sample x length-window+1) x features
    '''
    if window < 1:
        print "Wrong window size"
        sys.exit()

    X_new = []
    for i in xrange(len(X)): # per sample
        for j in xrange(len(X[i])-window+1): # per time
            X_new.append( X[i][j:j+window].tolist() ) # per sample
    
    return X_new


def graph_variations(x_true, x_pred_mean, x_pred_std=None):
    '''
    x_true: timesteps x dim
    '''

    # visualization
    import matplotlib
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import gridspec
    from matplotlib import rc
    import itertools
    colors = itertools.cycle(['g', 'm', 'c', 'k', 'y','r', 'b', ])
    shapes = itertools.cycle(['x','v', 'o', '+'])

    #matplotlib.rcParams['pdf.fonttype'] = 42
    #matplotlib.rcParams['ps.fonttype'] = 42
    #rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    #rc('text', usetex=True)
    ## matplotlib.rcParams['text.usetex'] = True
    
    nDim = len(x_true[0])
    if nDim > 6: nDim = 6
    
    fig = plt.figure(figsize=(6, 6))
    for k in xrange(nDim):
        fig.add_subplot(nDim,1,k+1)
        #plt.rc('text', usetex=True) 
        plt.plot(np.array(x_true)[:,k], '-b', label='Inputs')
        plt.plot(np.array(x_pred_mean)[:,k], '-r', )#label=r'$\mu$')
        if x_pred_std is not None and len(x_pred_std)>0:
            plt.plot(np.array(x_pred_mean)[:,k]+np.array(x_pred_std)[:,k], '--r', )#label=r'$\mu\pm\sigma$')
            plt.plot(np.array(x_pred_mean)[:,k]-np.array(x_pred_std)[:,k], '--r')
        #plt.ylim([-0.1,1.1])
    plt.show()

def graph_latent_space(normalTestData, abnormalTestData, enc_z, timesteps=1, batch_size=None,
                       method='lstm_vae'):

    print "latent variable visualization"
    if method == 'lstm_vae_offline':
        z_mean_n = enc_z_mean.predict(normalTestData)
        z_mean_a = enc_z_mean.predict(abnormalTestData)
        viz_latent_space(z_mean_n, z_mean_a)
    else:
        #if batch_size is not None:
        z_mean_n = []
        for i in xrange(len(normalTestData)):

            x = normalTestData[i:i+1]
            for j in xrange(batch_size-1):
                x = np.vstack([x,normalTestData[i:i+1]])            

            for j in xrange(len(x[0])-timesteps+1):
                z = enc_z.predict(x[:,j:j+timesteps], batch_size=batch_size)
                z_mean_n.append( z[0] )

        z_mean_a = []
        for i in xrange(len(abnormalTestData)):

            x = abnormalTestData[i:i+1]
            for j in xrange(batch_size-1):
                x = np.vstack([x,abnormalTestData[i:i+1]])            

            for j in xrange(len(x[0])-timesteps+1):
                z = enc_z.predict(x[:,j:j+timesteps], batch_size=batch_size)
                z_mean_a.append( z[0] )

        viz_latent_space(z_mean_n, z_mean_a)
    


def viz_latent_space(z_n, z_a=None):
    '''
    z_n: latent variable from normal data
    z_n: latent variable from abnormal data
    '''
    import matplotlib
    import matplotlib.pyplot as plt
    if type(z_n) is list: z_n = np.array(z_n)
    if type(z_a) is list: z_a = np.array(z_a)

    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42 
    fig = plt.figure(figsize=(6, 6))

    s = 121
    plt.scatter(z_n[:,0], z_n[:,1], color='b', s=0.5*s, alpha=.4, label='Non-anomalous')
    if z_a is not None:
        plt.scatter(z_a[:,0], z_a[:,1], color='r', s=0.5*s, marker='^', alpha=.4, label='Anomalous')

    plt.legend(loc=3, ncol=2)
    plt.show()


def get_ext_data(subjects, task_name, raw_data_path, save_data_path, param_dict,
                 init_param_dict=None, init_raw_param_dict=None, id_num=0, raw_feature=False,
                 depth=False):
    from hrl_anomaly_detection import data_manager as dm
    
    ## Parameters # data
    data_dict  = param_dict['data_param']
    data_renew = data_dict['renew']
    if raw_feature:
        AE_dict = param_dict['AE']
    
    #------------------------------------------
    if os.path.isdir(save_data_path) is False:
        os.system('mkdir -p '+save_data_path)

    if init_param_dict is None:
        crossVal_pkl = os.path.join(save_data_path, 'cv_'+task_name+'.pkl')
        print "CV data exists and no renew"
        d = ut.load_pickle(crossVal_pkl)
        init_param_dict = d['param_dict']
        if raw_feature:
            init_raw_param_dict = d['raw_param_dict']

    #------------------------------------------
    crossVal_pkl = os.path.join(save_data_path, 'cv_td_'+task_name+'_'+str(id_num)+'.pkl')
    if os.path.isfile(crossVal_pkl) and data_renew is False:
        print "CV data exists and no renew"
        td = ut.load_pickle(crossVal_pkl)
    else:
        if raw_feature is False:
            # Extract data from designated location
            td = dm.getDataLOPO(subjects, task_name, raw_data_path, save_data_path,\
                                downSampleSize=data_dict['downSampleSize'],\
                                init_param_dict=init_param_dict,\
                                handFeatures=data_dict['isolationFeatures'], \
                                cut_data=data_dict['cut_data'],\
                                data_renew=data_renew, max_time=data_dict['max_time'],
                                pkl_prefix='tgt_', depth=depth, id_num=id_num)

            td['successData'], td['failureData'], td['success_files'], td['failure_files'], td['kFoldList'] \
              = dm.LOPO_data_index(td['successDataList'], td['failureDataList'],\
                                   td['successFileList'], td['failureFileList'])
        else:
            # Extract data from designated location
            td = dm.getRawDataLOPO(subjects, task_name, raw_data_path, save_data_path,\
                                   downSampleSize=data_dict['downSampleSize'],\
                                   init_param_dict=init_param_dict,\
                                   init_raw_param_dict=init_raw_param_dict,\
                                   handFeatures=data_dict['isolationFeatures'], \
                                   rawFeatures=AE_dict['rawFeatures'],\
                                   cut_data=data_dict['cut_data'],\
                                   data_renew=data_renew, max_time=data_dict['max_time'],
                                   pkl_prefix='tgt_', depth=depth, id_num=id_num)

            td['successData'], td['failureData'], td['success_files'], td['failure_files'], td['kFoldList'] \
              = dm.LOPO_data_index(td['successRawDataList'], td['failureRawDataList'],\
                                   td['successFileList'], td['failureFileList'])

        ut.save_pickle(td, crossVal_pkl)
    
    if raw_feature is False:
        #------------------------------------------
        # select feature for detection
        feature_list = []
        for feature in param_dict['data_param']['handFeatures']:
            idx = [ i for i, x in enumerate(param_dict['data_param']['isolationFeatures']) if feature == x][0]
            feature_list.append(idx)

        td['successData']    = np.array(td['successData'])[feature_list]
        td['failureData']    = np.array(td['failureData'])[feature_list]

    return td


def get_scaled_data(normalTrainData, abnormalTrainData, normalTestData, abnormalTestData, aligned=True):
    '''
    Remove outlier and scale into 0-1 range
    '''

    if aligned is False:
        # dim x sample x length => sample x length x dim
        normalTrainData   = np.swapaxes(normalTrainData, 0,1 )
        normalTrainData   = np.swapaxes(normalTrainData, 1,2 )
        abnormalTrainData = np.swapaxes(abnormalTrainData, 0,1 )
        abnormalTrainData = np.swapaxes(abnormalTrainData, 1,2 )

        # dim x sample x length => sample x length x dim
        normalTestData   = np.swapaxes(normalTestData, 0,1 )
        normalTestData   = np.swapaxes(normalTestData, 1,2 )
        abnormalTestData = np.swapaxes(abnormalTestData, 0,1 )
        abnormalTestData = np.swapaxes(abnormalTestData, 1,2 )
        

    # normalization => (sample x dim) ----------------------------------
    from sklearn import preprocessing
    ## scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaler = preprocessing.StandardScaler() 


    normalTrainData_scaled   = scaler.fit_transform(normalTrainData.reshape(-1,len(normalTrainData[0][0])))
    abnormalTrainData_scaled = scaler.transform(abnormalTrainData.reshape(-1,len(abnormalTrainData[0][0])))
    normalTestData_scaled    = scaler.transform(normalTestData.reshape(-1,len(normalTestData[0][0])))
    abnormalTestData_scaled  = scaler.transform(abnormalTestData.reshape(-1,len(abnormalTestData[0][0])))

    # rescale 95%of values into 0-1
    def rescaler(x, mean, var):
        
        max_val = 1.67 #8 #1.9#mean+3.0*np.sqrt(var)
        min_val = -1.67 #8 #mean-3.0*np.sqrt(var)
        return (x-min_val)/( max_val-min_val )
    
    normalTrainData_scaled   = rescaler(normalTrainData_scaled, scaler.mean_, scaler.var_)
    abnormalTrainData_scaled = rescaler(abnormalTrainData_scaled, scaler.mean_, scaler.var_)
    normalTestData_scaled    = rescaler(normalTestData_scaled, scaler.mean_, scaler.var_)
    abnormalTestData_scaled  = rescaler(abnormalTestData_scaled, scaler.mean_, scaler.var_)

    # reshape
    normalTrainData   = normalTrainData_scaled.reshape(np.shape(normalTrainData))
    abnormalTrainData = abnormalTrainData_scaled.reshape(np.shape(abnormalTrainData))
    normalTestData   = normalTestData_scaled.reshape(np.shape(normalTestData))
    abnormalTestData  = abnormalTestData_scaled.reshape(np.shape(abnormalTestData))

    return normalTrainData, abnormalTrainData, normalTestData, abnormalTestData, scaler


def get_scaled_data2(x, scaler, aligned=True):
    if aligned is False:
        # dim x sample x length => sample x length x dim
        x = np.swapaxes(x, 0, 1 )
        x = np.swapaxes(x, 1, 2 )

    x_scaled = scaler.transform(x.reshape(-1,len(x[0][0])))

    # rescale 95%of values into 0-1
    def rescaler(X, mean, var):
        
        max_val = 1.8 #1.9#mean+3.0*np.sqrt(var)
        min_val = -1.8 #mean-3.0*np.sqrt(var)
        return (X-min_val)/( max_val-min_val )

    x_scaled = rescaler(x_scaled, scaler.mean_, scaler.var_)
    x        = x_scaled.reshape(np.shape(x))

    return x

        

def get_ext_feeding_data(task_name, save_data_path, param_dict, d, raw_feature=False):
    if raw_feature is False:
        d['raw_param_dict'] = None
    
    subjects = ['Andrew', 'Britteney', 'Joshua', 'Jun', 'Kihan', 'Lichard', 'Shingshing', 'Sid', 'Tao']
    raw_data_path  = os.path.expanduser('~')+'/hrl_file_server/dpark_data/anomaly/RAW_DATA/CORL2017/'
    td1 = get_ext_data(subjects, task_name, raw_data_path, save_data_path, param_dict,
                       init_param_dict=d['param_dict'], init_raw_param_dict=d['raw_param_dict'],
                       depth=True, id_num=1, raw_feature=raw_feature)

    subjects = ['ari', 'park', 'jina', 'linda', 'sai', 'hyun']
    raw_data_path  = os.path.expanduser('~')+'/hrl_file_server/dpark_data/anomaly/RAW_DATA/ICRA2017/'
    td2 = get_ext_data(subjects, task_name, raw_data_path, save_data_path, param_dict,
                       init_param_dict=d['param_dict'], init_raw_param_dict=d['raw_param_dict'],
                       id_num=2, raw_feature=raw_feature)

    subjects = []
    #for i in xrange(13,14):
    for i in xrange(1,23):
        subjects.append('day'+str(i))
    raw_data_path  = os.path.expanduser('~')+'/hrl_file_server/dpark_data/anomaly/RAW_DATA/ICRA2018/'
    td3 = get_ext_data(subjects, task_name, raw_data_path, save_data_path, param_dict,
                       init_param_dict=d['param_dict'], init_raw_param_dict=d['raw_param_dict'],
                       id_num=3, raw_feature=raw_feature)

    return td1, td2, td3
    
