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


def create_dataset(X, window_size=5, step=5):
    '''
    dataset: timesteps x dim
    '''    
    x = []
    y = []
    for j in range(len(X)-step-window_size):
        x.append(X[j:(j+window_size), :].tolist())
        y.append(X[j+step:(j+step+window_size), :].tolist())
    return np.array(x), np.array(y)



def graph_variations(x_true, x_pred_mean, x_pred_std=None, scaler_dict={}, save_pdf=False):
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

    # unscale
    if scaler_dict is None: param_dict = None
    else: param_dict = scaler_dict.get('param_dict', None)

    def unscale(x, std=False):
        if type(x) is list: x = np.array(x)
        x = x*2.*scaler_dict['scale'] - scaler_dict['scale']
        x = scaler_dict['scaler'].inverse_transform(x)
        if std is False:
            x = x*(param_dict['feature_max']-param_dict['feature_min'])+\
              param_dict['feature_min']
        else:
            x = x*(param_dict['feature_max']-param_dict['feature_min'])
        return x

    print np.shape(x_true), np.shape(x_pred_mean)

    

    ## x_true      = unscale(x_true)
    ## x_pred_mean = unscale(x_pred_mean)
    ## x_pred_std  = unscale(x_pred_std, std=True)
    #--------------------------------------------------------------------

    
    nDim = len(x_true[0])
    if nDim > 6: nDim = 6
    
    fig = plt.figure(figsize=(6, 6))
    for k in xrange(nDim):
        ax = fig.add_subplot(nDim,1,k+1)
        #plt.rc('text', usetex=True) 
        ax.plot(np.array(x_true)[:,k], '-b', label='Inputs')
        ax.plot(np.array(x_pred_mean)[:,k], '-r', )#label=r'$\mu$')
        if x_pred_std is not None and len(x_pred_std)>0:
            ax.fill_between(range(len(x_pred_mean)),
                            np.array(x_pred_mean)[:,k]+np.array(x_pred_std)[:,k],
                            np.array(x_pred_mean)[:,k]-np.array(x_pred_std)[:,k],
                            facecolor='red', alpha=0.5, linewidth=0)
            ## plt.plot(np.array(x_pred_mean)[:,k]+np.array(x_pred_std)[:,k], '--r', )#label=r'$\mu\pm\sigma$')
            ## plt.plot(np.array(x_pred_mean)[:,k]-np.array(x_pred_std)[:,k], '--r')
        #plt.ylim([-0.1,1.1])

        if k==0:
            ax.set_ylabel('Sound'+'\n'+'Energy', rotation='horizontal',
                          verticalalignment='center',
                          horizontalalignment='center')
            ## ax.set_ylabel('Audio [RMS]')
        elif k==1: 
            ax.set_ylabel('1st Joint'+'\n'+'Torque(Nm)', rotation='horizontal',
                          verticalalignment='center',
                          horizontalalignment='center')
            ## ax.set_ylabel('1st Joint \n effort')            
        elif k==2: 
            ax.set_ylabel('Accumulated'+'\n'+'Force'+'\n'+'on Spoon(N)',
                          rotation='horizontal', verticalalignment='center',
                          horizontalalignment='center')
            ## ax.set_ylabel('Accumulated \n force [N]')
        elif k==3: 
            ax.set_ylabel('Spoon-Mouth'+'\n'+'Distance(m)', rotation='horizontal',
                          verticalalignment='center',
                          horizontalalignment='center')
            ## ax.set_ylabel('Distance [m]')
            
        ax.yaxis.set_label_coords(-0.2,0.5)            
        #ax.set_ylabel(param_dict['feature_names'][k])

        ax.locator_params(axis='y', nbins=3)
        if k < nDim-1: ax.tick_params(axis='x', bottom='off', labelbottom='off')

    if param_dict is not None:
        x_tick = [param_dict['timeList'][0],
                  (param_dict['timeList'][-1]-param_dict['timeList'][0])/2.0,
                  param_dict['timeList'][-1]]
        ax.set_xticks(np.linspace(0, len(x_pred_mean), len(x_tick)))        
        ax.set_xticklabels(x_tick)
        ax.set_xlabel('Time [s]', fontsize=18)
        fig.subplots_adjust(left=0.25) 

    if save_pdf :
        fig.savefig('test.pdf')
        fig.savefig('test.png')
        fig.savefig('test.eps')
        os.system('cp test.p* ~/Dropbox/HRL/')        
        os.system('cp test.e* ~/Dropbox/HRL/')        
    else:
        plt.show()

    ut.get_keystroke('Hit a key to proceed next')  


def graph_latent_space(normalTestData, abnormalTestData, enc_z, timesteps=1, batch_size=None,
                       method='lstm_vae', save_pdf=False):

    print "latent variable visualization"
    if method == 'lstm_vae_offline':
        z_mean_n = enc_z_mean.predict(normalTestData)
        z_mean_a = enc_z_mean.predict(abnormalTestData)
        viz_latent_space(z_mean_n, z_mean_a)
    else:
        #if batch_size is not None:
        z_mean_n = []
        z_mean_n_s = []
        z_mean_n_e = []
        for i in xrange(len(normalTestData)):

            x = normalTestData[i:i+1]
            for j in xrange(batch_size-1):
                x = np.vstack([x,normalTestData[i:i+1]])            

            z_mean=[]
            for j in xrange(len(x[0])-timesteps+1):
                if method.find('lstm_vae_custom') >=0 or method.find('lstm_dvae_custom') >=0
                    or method.find('phase')>=0:
                    x_in = np.concatenate((x[:,j:j+timesteps],
                                           np.zeros((len(x), timesteps,1))), axis=-1)
                else:
                    x_in = x[:,j:j+timesteps]
                z = enc_z.predict(x_in, batch_size=batch_size)
                z_mean.append( z[0] )
                if j==0: z_mean_n_s.append(z[0])
                if j==len(x[0])-timesteps: z_mean_n_e.append(z[0])
            z_mean_n.append(z_mean)

        z_mean_a = []
        z_mean_a_s = []
        z_mean_a_e = []
        for i in xrange(len(abnormalTestData)):

            x = abnormalTestData[i:i+1]
            for j in xrange(batch_size-1):
                x = np.vstack([x,abnormalTestData[i:i+1]])            

            z_mean=[]
            for j in xrange(len(x[0])-timesteps+1):
                if method.find('lstm_vae_custom')>=0 or method.find('lstm_vae_custom')>=0
                    or method.find('phase')>=0:
                    x_in = np.concatenate((x[:,j:j+timesteps],
                                           np.zeros((len(x), timesteps,1))), axis=-1)
                else:
                    x_in = x[:,j:j+timesteps]                    
                z = enc_z.predict(x_in, batch_size=batch_size)
                z_mean.append( z[0] )
                if j==0: z_mean_a_s.append(z[0])
                if j==len(x[0])-timesteps: z_mean_a_e.append(z[0])
            z_mean_a.append(z_mean)

        viz_latent_space(z_mean_n, z_mean_a, z_n_se=(z_mean_n_s, z_mean_n_e),
                         save_pdf=save_pdf)
    


def viz_latent_space(z_n, z_a=None, z_n_se=None, save_pdf=False):
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
    
    if z_a is not None :
        for z in z_a:
            ax2 = plt.scatter(z[:,0], z[:,1], color='r', s=0.5*s, marker='^', alpha=.4, label='Anomalous')
        #    plt.plot(z[:,0], z[:,1], 'r', marker='^', ms=5, alpha=.4, label='Anomalous')

    for z in z_n:
        ax1 = plt.scatter(z[:,0], z[:,1], color='b', s=0.5*s, alpha=.4, label='Non-anomalous')
        #    plt.plot(z[:,0], z[:,1], 'b', marker='o', ms=5, alpha=.4, label='Non-anomalous')

    if z_n_se is not None:
        z_n_s, z_n_e = z_n_se
        plt.scatter(np.array(z_n_s)[:,0], np.array(z_n_s)[:,1], color='g', s=1.*s, marker='x')
        plt.scatter(np.array(z_n_e)[:,0], np.array(z_n_e)[:,1], color='y', s=1.*s, marker='x') 
            

    ax = plt.gca()
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    
        
    #plt.legend(handles=[ax1, ax2], loc=3, ncol=2)

    if save_pdf:
        fig.savefig('test.pdf')
        fig.savefig('test.png')
        fig.savefig('test.eps')
        os.system('cp test.p* ~/Dropbox/HRL/')
    else:
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


def get_scaled_data(normalTrainData, abnormalTrainData, normalTestData, abnormalTestData, aligned=True,
                    scale=1.8):
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
        
        max_val = scale #1.9#mean+3.0*np.sqrt(var)
        min_val = -scale #mean-3.0*np.sqrt(var)
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


def get_scaled_data2(x, scaler, aligned=True, scale=1.8):
    if aligned is False:
        # dim x sample x length => sample x length x dim
        x = np.swapaxes(x, 0, 1 )
        x = np.swapaxes(x, 1, 2 )

    x_scaled = scaler.transform(x.reshape(-1,len(x[0][0])))

    # rescale 95%of values into 0-1
    def rescaler(X, mean, var):
        
        max_val = scale #1.9#mean+3.0*np.sqrt(var)
        min_val = -scale #mean-3.0*np.sqrt(var)
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
    

def get_roc(tp_l, tn_l, fp_l, fn_l):

    tp_ll = []
    fp_ll = []
    tn_ll = []
    fn_ll = []  
    for i in xrange(len(tp_l)):
        tp_ll.append( tp_l[i])
        fp_ll.append( fp_l[i])
        tn_ll.append( tn_l[i])
        fn_ll.append( fn_l[i])



    tpr_l = np.array(tp_ll).astype(float)/(np.array(tp_ll).astype(float)+
                                           np.array(fn_ll).astype(float))*100.0
    fpr_l = np.array(fp_ll).astype(float)/(np.array(fp_ll).astype(float)+
                                           np.array(tn_ll).astype(float))*100.0

    from sklearn import metrics 
    return metrics.auc(fpr_l, tpr_l, True)

