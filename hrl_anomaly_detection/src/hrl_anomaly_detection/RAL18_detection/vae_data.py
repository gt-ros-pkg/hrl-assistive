
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
from hrl_anomaly_detection.RAL18_detection import util as vutil
from hrl_anomaly_detection.RAL18_detection import detector as dt 

# Private learners
from hrl_anomaly_detection.hmm import learning_hmm as hmm
import hrl_anomaly_detection.classifiers.classifier as cf
import hrl_anomaly_detection.data_viz as dv
from hrl_anomaly_detection.RAL18_detection import keras_models as km

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


def lstm_test(subject_names, task_name, raw_data_path, processed_data_path, param_dict, plot=False,
              re_load=False, fine_tuning=False, dyn_ths=False, latent_plot=False):
    ## Parameters
    data_dict  = param_dict['data_param']
    data_renew = data_dict['renew']
    ae_renew   = param_dict['HMM']['renew']
    method     = param_dict['ROC']['methods'][0]
    
    if ae_renew: clf_renew = True
    else: clf_renew  = param_dict['SVM']['renew']
    
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
    d['successData']   = d['successData'][feature_list]
    d['failureData']   = d['failureData'][feature_list]
    d['param_dict']['feature_max']   = np.array(d['param_dict']['feature_max'])[feature_list]   
    d['param_dict']['feature_min']   = np.array(d['param_dict']['feature_min'])[feature_list]
    #d['param_dict']['feature_names'] = d['param_dict']['feature_names'][feature_list]

    
    # Parameters
    nDim = len(d['successData'])
    batch_size  = 1 #64
    scale = 1.8
    ths_l = np.logspace(-1.0,2.2,40) -0.1
    add_data = False

    if fine_tuning is False and add_data:
        td1, td2, td3 = vutil.get_ext_feeding_data(task_name, save_data_path, param_dict, d,
                                                   raw_feature=False)
                  

    tp_ll = [[] for i in xrange(len(ths_l))]
    fp_ll = [[] for i in xrange(len(ths_l))]
    tn_ll = [[] for i in xrange(len(ths_l))]
    fn_ll = [[] for i in xrange(len(ths_l))]
    roc_l = []

    # split data
    # HMM-induced vector with LOPO
    for idx, (normalTrainIdx, abnormalTrainIdx, normalTestIdx, abnormalTestIdx) \
      in enumerate(d['kFoldList']):
        #if idx != 5: continue
        #if not(idx == 0 or idx == 7): continue

        # pred_score_4dim_success: idx==5, ths=1.5, first data
        # pred_score_4dim_failure: idx==3, class 5-10, ths=1.5 
        # /home/dpark/hrl_file_server/dpark_data/anomaly/RAW_DATA/AURO2016/s5_feeding/5_10_failure.pkl
        
        # pred_score_4dim_failure: idx==2, class 7-14, ths=1. 
        # /home/dpark/hrl_file_server/dpark_data/anomaly/RAW_DATA/AURO2016/s4_feeding/7_14_failure.pkl
        
        
        print "==================== ", idx, " ========================"

        # dim x sample x length
        normalTrainData   = d['successData'][:, normalTrainIdx, :]
        abnormalTrainData = d['failureData'][:, abnormalTrainIdx, :]
        normalTestData    = d['successData'][:, normalTestIdx, :]
        abnormalTestData  = d['failureData'][:, abnormalTestIdx, :]
        if fine_tuning is False and add_data:
            normalTrainData   = np.hstack([normalTrainData,
                                           copy.deepcopy(td1['successData']),
                                           copy.deepcopy(td2['successData']),
                                           copy.deepcopy(td3['successData'])])
            abnormalTrainData = np.hstack([abnormalTrainData,
                                           copy.deepcopy(td1['failureData']),
                                           copy.deepcopy(td2['failureData']),
                                           copy.deepcopy(td3['failureData'])])

        # shuffle
        np.random.seed(3334+idx)
        idx_list = range(len(normalTrainData[0]))
        np.random.shuffle(idx_list)
        normalTrainData = normalTrainData[:,idx_list]            

        normalTrainData, abnormalTrainData, normalTestData, abnormalTestData, scaler =\
          vutil.get_scaled_data(normalTrainData, abnormalTrainData,
                                normalTestData, abnormalTestData, aligned=False, scale=scale)

        trainData = [normalTrainData[:int(len(normalTrainData)*0.7)],
                     [0]*len(normalTrainData[:int(len(normalTrainData)*0.7)])]
        valData   = [normalTrainData[int(len(normalTrainData)*0.7):],
                     [0]*len(normalTrainData[int(len(normalTrainData)*0.7):])]
        testData  = [normalTestData, [0]*len(normalTestData)]

        # scaling info to reconstruct the original scale of data
        scaler_dict = {'scaler': scaler, 'scale': scale, 'param_dict': d['param_dict']}

        # ------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------        
        method      = 'lstm_dvae_phase'
        #method      = 'rnd'
        #method      = 'osvm'
         
        weights_path = os.path.join(save_data_path,'model_weights_'+method+'_'+str(idx)+'.h5')
        vae_mean   = None
        vae_logvar = None
        enc_z_mean = enc_z_std = None
        generator  = None
        x_std_div   = None
        x_std_offset= None
        z_std      = None
        dyn_ths    = False
        stateful   = False
        
        window_size = 1
        batch_size  = 256
        fixed_batch_size = True
        noise_mag   = 0.05
        sam_epoch   = 20
        patience    = 4
        h1_dim      = nDim
        z_dim       = 2
        phase       = 1.0
        gamma       = 0.1
        nu          = 1.0

        if (method.find('lstm_vae')>=0 or method.find('lstm_dvae')>=0) and\
            method.find('offline')<0 and method.find('pred')<0:
            dyn_ths  = True
            stateful = True
            ad_method   = 'lower_bound'
            phase       = 1.0
            
            if method == 'lstm_vae':
                from hrl_anomaly_detection.RAL18_detection.models import lstm_vae_state_batch as km
                ths_l = np.logspace(-1.0,2.4,40) #-0.1
            elif method == 'lstm_vae_custom':
                from hrl_anomaly_detection.RAL18_detection.models import lstm_vae_custom as km
                if nDim == 4:
                    ths_l = np.logspace(-1.0,2.,40) -0.2
                    x_std_div   = 4.
                    x_std_offset= 0.05
                    z_std       = 0.3 #0.2
                elif nDim == 6:
                   ths_l = np.logspace(-1.0,2.,40) -0.2
                   x_std_div   = 4.
                   x_std_offset= 0.1
                   z_std       = 0.2
                else:
                   ths_l = np.logspace(-1.0,2.,40) -0.2
                   x_std_div   = 4.
                   x_std_offset= 0.1
                   z_std       = 0.3
                h1_dim      = nDim #8 #4 # raw
            elif method == 'lstm_dvae_custom':
                from hrl_anomaly_detection.RAL18_detection.models import lstm_dvae_custom as km
                ths_l = np.logspace(-1.0,1.7,40) -0.01
                x_std_div   = 4.
                x_std_offset= 0.1
                z_std       = 0.5
                h1_dim      = nDim #8 #4 # raw
                if add_data is False:
                    batch_size = 32
                    
            #------------------------------------------------------------------
            elif method == 'lstm_vae_phase':
                from hrl_anomaly_detection.RAL18_detection.models import lstm_vae_phase as km
                ths_l = np.logspace(-1.0,2.,40) -0.2
                x_std_div   = 4.
                x_std_offset= 0.1
                z_std       = 0.3 #0.2
                h1_dim      = nDim #8 #4 # raw
            elif method == 'lstm_dvae_phase':
                from hrl_anomaly_detection.RAL18_detection.models import lstm_dvae_phase2 as km
                ths_l = np.logspace(-1.0,2.4,40) -0.2
                x_std_div   = 4.
                x_std_offset= 0.1
                z_std       = 1.0 
                h1_dim      = nDim
                z_dim       = 3
                phase       = 1.0
                sam_epoch   = 40
                if add_data is False:
                    batch_size = 32
                    
            #------------------------------------------------------------------
            elif method == 'lstm_vae_custom3':
                from hrl_anomaly_detection.RAL18_detection.models import lstm_vae_custom3 as km
                ths_l = np.logspace(-1.0,2.,40) -0.2
                x_std_offset= 0.05
                z_std       = 0.6
                sam_epoch   = 1
            elif method == 'lstm_vae2':
                from hrl_anomaly_detection.RAL18_detection.models import lstm_vae_state_batch2 as km
                ths_l = np.logspace(-1.0,2.2,40) -0.5  
            else:
                from hrl_anomaly_detection.RAL18_detection.models import lstm_dvae_state_batch as km
                ths_l = np.logspace(-1.0,2.2,40) -0.1  

                

            autoencoder, vae_mean, _, enc_z_mean, enc_z_std, generator = \
              km.lstm_vae(trainData, valData, weights_path, patience=4, batch_size=batch_size,
                          noise_mag=noise_mag, timesteps=window_size, sam_epoch=sam_epoch,
                          x_std_div=x_std_div, x_std_offset=x_std_offset, z_std=z_std,                          
                          h1_dim = h1_dim, phase=phase, z_dim=z_dim,\
                          renew=ae_renew, fine_tuning=fine_tuning, plot=plot,
                          scaler_dict=scaler_dict)

        elif method == 'lstm_dvae_pred':
            from hrl_anomaly_detection.RAL18_detection.models import lstm_dvae_pred as km
            ths_l = np.logspace(-1.0,2.,40) -0.2
            window_size = 1
            x_std_div   = 2.
            x_std_offset= 0.1
            z_std       = 1. #0.5 #0.2
            h1_dim      = nDim #8 #4 # raw
            phase       = 0.5
            ad_method   = 'lower_bound'
            dyn_ths    = True
            stateful = True   
            
            if add_data is False:
                batch_size = 32
            autoencoder, vae_mean, _, enc_z_mean, enc_z_std, generator = \
              km.lstm_vae(trainData, valData, weights_path, patience=patience, batch_size=batch_size,
                          noise_mag=noise_mag, timesteps=window_size, sam_epoch=sam_epoch,
                          x_std_div=x_std_div, x_std_offset=x_std_offset, z_std=z_std,
                          h1_dim = h1_dim, phase=phase,\
                          renew=ae_renew, fine_tuning=fine_tuning, plot=plot,\
                          scaler_dict=scaler_dict)    

        #--------------------------------------------------------------------------------
        elif method == 'lstm_pred':
            from hrl_anomaly_detection.RAL18_detection.models import lstm_pred as km
            from hrl_anomaly_detection.RAL18_detection.models import lstm_pred_var as km
            stateful = True
            ths_l = np.logspace(-3.0,2.1,40) #-0.1
            ad_method   = 'recon_err'
            window_size = 5
            x_std_div   = 2.
            x_std_offset= 0.1
            
            autoencoder, vae_mean = \
              km.lstm_pred(trainData, valData, weights_path, patience=4, batch_size=batch_size,
                           noise_mag=noise_mag, timesteps=window_size, sam_epoch=sam_epoch,
                           x_std_div=x_std_div, x_std_offset=x_std_offset,
                           re_load=re_load, renew=ae_renew, fine_tuning=fine_tuning, plot=plot)

        elif method == 'ae':
            from hrl_anomaly_detection.RAL18_detection.models import ae
            window_size  = 3
            batch_size   = 256
            sam_epoch    = 20
            fixed_batch_size = False
            stateful     = False
            ad_method    = 'recon_err_lld' #'recon_err_lld'
            ths_l = np.logspace(-1.0,4.0,40)  
            autoencoder, enc_z_mean, generator = \
              ae.autoencoder(trainData, valData, weights_path, patience=5, batch_size=batch_size,
                             noise_mag=noise_mag, sam_epoch=sam_epoch, timesteps=window_size,\
                             renew=ae_renew, fine_tuning=fine_tuning, plot=plot)
            vae_mean = autoencoder
            
        elif method == 'lstm_ae':
            # LSTM-AE (Confirmed) %74.99
            from hrl_anomaly_detection.RAL18_detection.models import lstm_ae_state_batch as km
            stateful = True
            ad_method   = 'recon_err'
            ths_l = np.logspace(-1.0,1.8,40) -0.5 
            autoencoder, vae_mean,_, enc_z_mean = \
              km.lstm_ae(trainData, valData, weights_path, patience=4, batch_size=batch_size,
                         noise_mag=noise_mag, timesteps=window_size, sam_epoch=sam_epoch,
                         re_load=re_load, renew=ae_renew, fine_tuning=fine_tuning, plot=plot)
            vae_mean = autoencoder
        elif method == 'encdec_ad':
            # EncDec-AD from Malhortra
            from hrl_anomaly_detection.RAL18_detection.models import encdec_ad as km
            window_size = 3
            sam_epoch   = 40
            batch_size  = 256
            noise_mag   = 0.05
            fixed_batch_size = False
            stateful = False
            ad_method   = 'recon_err_lld'
            ths_l = np.logspace(-0.3,4.0,40) #-0.5 
            autoencoder = \
              km.lstm_ae(trainData, valData, weights_path, patience=4, batch_size=batch_size,
                         noise_mag=noise_mag, timesteps=window_size, sam_epoch=sam_epoch,
                         re_load=re_load, renew=ae_renew, fine_tuning=fine_tuning, plot=plot)
            vae_mean = autoencoder
        elif method == 'lstm_vae_offline':
            from hrl_anomaly_detection.RAL18_detection.models import lstm_vae_offline as km
            window_size  = 0
            batch_size   = 1024
            sam_epoch    = 100
            x_std_div    = 1
            x_std_offset = 0.05
            z_std        = 0.7
            fixed_batch_size = False
            stateful     = False
            ad_method    = 'lower_bound'
            ths_l = np.logspace(-1.0,0.6,40)-1.0  
            autoencoder, vae_mean, _, enc_z_mean, enc_z_std, generator = \
              km.lstm_vae(trainData, valData, weights_path, patience=5, batch_size=batch_size,
                          noise_mag=noise_mag, sam_epoch=sam_epoch,
                          x_std_div=x_std_div, x_std_offset=x_std_offset, z_std=z_std,\
                          re_load=re_load, renew=ae_renew, fine_tuning=fine_tuning, plot=plot)

        elif method == 'rnd':
            autoencoder = None
            dyn_ths     = False
            window_size = 1
            ths_l = np.linspace(0.0,1.0,40)

        elif method == 'osvm':
            window_size = 3
            fixed_batch_size = False
            ad_method   = None
            nu    = 0.05
            gamma = 1e-5
            ths_l = np.logspace(-4, -0.2, 40)
            #ths_l = np.logspace(-6.5, -1, 10)
            autoencoder = None

        #------------------------------------------------------------------------------------
        if  True and False: 
            vutil.graph_latent_space(normalTestData, abnormalTestData, enc_z_mean,
                                     timesteps=window_size, batch_size=batch_size,
                                     method=method, save_pdf=False)
            
        # -----------------------------------------------------------------------------------
        if True and False:
            # get optimized alpha
            if fine_tuning: alpha_renew = True
            else: alpha_renew = False
            save_pkl = os.path.join(save_data_path, 'model_alpha_'+method+'_'+str(idx)+'.pkl')
            alpha = dt.get_optimal_alpha((valData[0], abnormalTrainData), autoencoder, vae_mean,
                                         ad_method, method, window_size, save_pkl,\
                                         stateful=stateful, renew=alpha_renew,\
                                         x_std_div = x_std_div, x_std_offset=x_std_offset, z_std=z_std,
                                         dyn_ths=dyn_ths, batch_info=(fixed_batch_size,batch_size))
        else:
            alpha = np.array([1.0]*nDim)
            #alpha[0] = 0.5
            ## if nDim ==8:
            ##     alpha[-1] = 0.4
            ## else:
            ##     alpha[0] = 0.4

        if fine_tuning: clf_renew=True
        normalTrainData = vutil.get_scaled_data2(d['successData'][:, normalTrainIdx, :],
                                                 scaler, aligned=False)

        if method == 'osvm':
            from hrl_anomaly_detection.RAL18_detection.models import osvm as dt
        else:
            from hrl_anomaly_detection.RAL18_detection import detector as dt


        save_pkl = os.path.join(save_data_path, 'model_ad_scores_'+str(idx)+'.pkl')
        tp_l, tn_l, fp_l, fn_l, roc = \
          dt.anomaly_detection(autoencoder, vae_mean, vae_logvar, enc_z_mean, enc_z_std, generator,
                               normalTrainData, valData[0],\
                               normalTestData, abnormalTestData, \
                               ad_method, method,
                               window_size, alpha, nu=nu, gamma=gamma, ths_l=ths_l,
                               save_pkl=save_pkl, stateful=stateful,
                               x_std_div = x_std_div, x_std_offset=x_std_offset, z_std=z_std,
                               phase=phase, latent_plot=latent_plot,
                               renew=clf_renew, dyn_ths=dyn_ths, batch_info=(fixed_batch_size,batch_size),
                               param_dict=d['param_dict'], scaler_dict=scaler_dict,
                               filenames=(np.array(d['success_files'])[normalTestIdx],
                                          np.array(d['failure_files'])[abnormalTestIdx]))

        roc_l.append(roc)

        for i in xrange(len(ths_l)):
            tp_ll[i] += tp_l[i]
            fp_ll[i] += fp_l[i]
            tn_ll[i] += tn_l[i]
            fn_ll[i] += fn_l[i]

    print "roc list ", roc_l

    d = {}
    d['tp_ll'] = tp_ll
    d['fp_ll'] = fp_ll
    d['tn_ll'] = tn_ll
    d['fn_ll'] = fn_ll
    #roc_pkl = os.path.join(processed_data_path, 'roc_'+task_name+'.pkl')
    #ut.save_pickle(d, roc_pkl)

    tpr_l = []
    fpr_l = []
    for i in xrange(len(ths_l)):
        tpr_l.append( float(np.sum(tp_ll[i]))/float(np.sum(tp_ll[i])+np.sum(fn_ll[i]))*100.0 )
        fpr_l.append( float(np.sum(fp_ll[i]))/float(np.sum(fp_ll[i])+np.sum(tn_ll[i]))*100.0 )  

    print roc_l
    print "------------------------------------------------------"
    print tpr_l
    print fpr_l


    from sklearn import metrics
    print "roc: ", metrics.auc(fpr_l, tpr_l, True)  
    fig = plt.figure(figsize=(6,6))
    fig.add_subplot(1,1,1)
    plt.plot(fpr_l, tpr_l, '-*b', ms=5, mec='b')
    plt.xlim([0,100])
    plt.ylim([0,100])
    plt.show()





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

    td = vutil.get_ext_data(subject_names, task_name, raw_data_path, save_data_path, param_dict,
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
    






def get_batch_data(normalData, abnormalData, win=False):
    
    # dim x sample x length => sample x length x dim
    normalData   = np.swapaxes(normalData, 0,1 )
    normalData   = np.swapaxes(normalData, 1,2 )
    abnormalData = np.swapaxes(abnormalData, 0,1 )
    abnormalData = np.swapaxes(abnormalData, 1,2 )

    np.random.shuffle(normalData)
    np.random.shuffle(abnormalData)
    print np.shape(normalData), np.shape(abnormalData)

    ratio=0.7
    normalTrainData, normalTestData\
    = normalData[:int(len(normalData)*ratio)],normalData[int(len(normalData)*ratio):]
    abnormalTrainData, abnormalTestData\
    = abnormalData[:int(len(abnormalData)*ratio)],abnormalData[int(len(abnormalData)*ratio):]

    normalTrainData, abnormalTrainData, normalTestData, abnormalTestData =\
      vutil.get_scaled_data(normalTrainData, abnormalTrainData,
                            normalTestData, abnormalTestData, aligned=True)
    

    if win:
        window_size = 20
        
        # get window data
        # sample x length x dim => (sample x length) x dim
        normalTrainData_ft   = vutil.sampleWithWindow(normalTrainData, window=window_size)
        abnormalTrainData_ft = vutil.sampleWithWindow(abnormalTrainData, window=window_size)
        normalTestData_ft    = vutil.sampleWithWindow(normalTestData, window=window_size)
        abnormalTestData_ft  = vutil.sampleWithWindow(abnormalTestData, window=window_size)

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

            if window_size>0: x = vutil.sampleWithWindow(normalTrainData[i:i+1], window=window_size)
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

    td = vutil.get_ext_data(subject_names, task_name, raw_data_path, save_data_path, param_dict,
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
    p.add_option('--reload', '--rl', action='store_true', dest='bReLoad',
                 default=False, help='Reload previous parameters.')
    p.add_option('--fint_tuning', '--ftn', action='store_true', dest='bFineTune',
                 default=False, help='Run fine tuning.')
    p.add_option('--dyn_ths', '--dt', action='store_true', dest='bDynThs',
                 default=False, help='Run dynamic threshold.')
    p.add_option('--latent_space_plot', '--lsp', action='store_true', dest='bLatentPlot',
                 default=False, help='Show latent space.')

    opt, args = p.parse_args()

    #---------------------------------------------------------------------------           
    # Run evaluation
    #---------------------------------------------------------------------------           
    rf_center   = 'kinEEPos'        
    scale       = 1.0
    local_range = 10.0
    nPoints     = 40 #None
    opt.bHMMRenew = opt.bAERenew

    from hrl_anomaly_detection.RAL18_detection.vae_params import *
    raw_data_path, save_data_path, param_dict = getParams(opt.task, opt.bDataRenew, \
                                                          opt.bHMMRenew, opt.bCLFRenew, opt.dim,\
                                                          rf_center, local_range, nPoints=nPoints)
    if opt.bNoUpdate: param_dict['ROC']['update_list'] = []
    # Mikako - bad camera
    # s1 - kaci - before camera calibration
    subjects = ['s2', 's3','s4','s5', 's6','s7','s8', 's9']

    if os.uname()[1] == 'monty1':
        ## save_data_path = os.path.expanduser('~')+\
        ##   '/hrl_file_server/dpark_data/anomaly/ICRA2018/'+opt.task+'_data_lstm'
        save_data_path = os.path.expanduser('~')+\
          '/hrl_file_server/dpark_data/anomaly/ICRA2018/'+opt.task+'_data_lstm_4'    
    else:
        save_data_path = os.path.expanduser('~')+\
          '/hrl_file_server/dpark_data/anomaly/ICRA2018/'+opt.task+'_data_lstm_dvae_phase'    
        #save_data_path = os.path.expanduser('~')+\
        #  '/hrl_file_server/dpark_data/anomaly/ICRA2018/'+opt.task+'_data_osvm'    


    #save_data_path = os.path.expanduser('~')+\
    #  '/hrl_file_server/dpark_data/anomaly/ICRA2018/'+opt.task+'_data_lstm_4'

          
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

    '''
    param_dict['data_param']['handFeatures'] = ['unimodal_kinVel',\
                                                'unimodal_kinJntEff_1',\
                                                'unimodal_ftForce_zero',\
                                                'unimodal_ftForce_integ',\
                                                'unimodal_kinEEChange',\
                                                'unimodal_kinDesEEChange',\
                                                'crossmodal_landmarkEEDist', \
                                                'unimodal_audioWristRMS']


    param_dict['data_param']['handFeatures'] = ['unimodal_audioWristRMS',\
                                                'unimodal_kinJntEff_1',\
                                                'unimodal_ftForce_zero',\
                                                'unimodal_ftForce_integ',\
                                                'unimodal_kinDesEEChange',\
                                                'crossmodal_landmarkEEDist', \
                                                ]


    '''
    param_dict['data_param']['handFeatures'] = ['unimodal_audioWristRMS',  \
                                               'unimodal_kinJntEff_1',\
                                               'unimodal_ftForce_integ',\
                                               'crossmodal_landmarkEEDist']
    


    if opt.gen_data:
        if opt.bNoUpdate: param_dict['ROC']['update_list'] = []        
        gen_data(subjects, opt.task, raw_data_path, save_data_path, param_dict)
        
    elif opt.extra_data:
        
        if opt.bNoUpdate: param_dict['ROC']['update_list'] = []        
        vutil.get_ext_data(subjects, opt.task, raw_data_path, save_data_path, param_dict)
        

    elif opt.preprocessing:
        src_pkl = os.path.join(save_data_path, 'isol_data.pkl')
        from hrl_execution_monitor import preprocess as pp
        pp.preprocess_data(src_pkl, save_data_path, img_scale=0.25, nb_classes=12,
                            img_feature_type='vgg', nFold=nFold)

    elif opt.lstm_test:
        lstm_test(subjects, opt.task, raw_data_path, save_data_path, param_dict, plot=not opt.bNoPlot,
                  re_load=opt.bReLoad, fine_tuning=opt.bFineTune, dyn_ths=opt.bDynThs,\
                  latent_plot=opt.bLatentPlot)

    elif opt.bFeaturePlot:
        
        feature_plot(subjects, opt.task, raw_data_path, save_data_path, param_dict)
