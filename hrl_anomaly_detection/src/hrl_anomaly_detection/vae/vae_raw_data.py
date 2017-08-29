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
from hrl_anomaly_detection.vae import util as vutil


# Private learners
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

def lstm_test(subject_names, task_name, raw_data_path, processed_data_path, param_dict, plot=False,
              re_load=False, fine_tuning=False, dyn_ths=False):
    ## Parameters
    data_dict  = param_dict['data_param']
    AE_dict    = param_dict['AE']
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
        d = dm.getRawDataLOPO(subject_names, task_name, raw_data_path, \
                              processed_data_path, data_dict['rf_center'], data_dict['local_range'],\
                              downSampleSize=data_dict['downSampleSize'],\
                              handFeatures=data_dict['isolationFeatures'], \
                              rawFeatures=AE_dict['rawFeatures'], \
                              cut_data=data_dict['cut_data'], \
                              data_renew=data_renew, max_time=data_dict['max_time'])

        d['successData'], d['failureData'], d['success_files'], d['failure_files'], d['kFoldList'] \
          = dm.LOPO_data_index(d['successRawDataList'], d['failureRawDataList'],\
                               d['successFileList'], d['failureFileList'])

        ut.save_pickle(d, crossVal_pkl)

    if fine_tuning is False:
        td1, td2, td3 = vutil.get_ext_feeding_data(task_name, save_data_path, param_dict, d,
                                                   raw_feature=True)
        sys.exit()
        

    # Parameters
    nDim = len(d['successData'])
    batch_size  = 1 #64
    ths_l = np.logspace(-1.0,2.2,40) -0.1 

    tp_ll = [[] for i in xrange(len(ths_l))]
    fp_ll = [[] for i in xrange(len(ths_l))]
    tn_ll = [[] for i in xrange(len(ths_l))]
    fn_ll = [[] for i in xrange(len(ths_l))]
    roc_l = []

    # split data
    # HMM-induced vector with LOPO
    for idx, (normalTrainIdx, abnormalTrainIdx, normalTestIdx, abnormalTestIdx) \
      in enumerate(d['kFoldList']):
        #if (idx == 0 or idx==7): continue
        if idx != 0: continue
        print "==================== ", idx, " ========================"

        # dim x sample x length
        normalTrainData   = d['successData'][:, normalTrainIdx, :]
        abnormalTrainData = d['failureData'][:, abnormalTrainIdx, :]
        normalTestData    = d['successData'][:, normalTestIdx, :]
        abnormalTestData  = d['failureData'][:, abnormalTestIdx, :]
        if fine_tuning is False:
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
                                normalTestData, abnormalTestData, aligned=False)

        trainData = [normalTrainData[:int(len(normalTrainData)*0.7)],
                     [0]*len(normalTrainData[:int(len(normalTrainData)*0.7)])]
        valData   = [normalTrainData[int(len(normalTrainData)*0.7):],
                     [0]*len(normalTrainData[int(len(normalTrainData)*0.7):])]
        testData  = [normalTestData, [0]*len(normalTestData)]

        # scaling info to reconstruct the original scale of data
        scaler_dict = {'scaler': scaler, 'scale': scale, 'param_dict': d['raw_param_dict']}

        # ------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------        
        method      = 'lstm_dvae_phase'
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
        
        window_size = 1
        batch_size  = 256
        fixed_batch_size = True
        noise_mag   = 0.05
        sam_epoch   = 40
        patience    = 4
        h1_dim      = nDim
        z_dim       = 2
        phase       = 1.0
        stateful    = None
        ad_method   = None

        if (method.find('lstm_vae')>=0 or method.find('lstm_dvae')>=0) and\
            method.find('offline')<0:
            dyn_ths     = True
            stateful    = True
            ad_method   = 'lower_bound'

            if method == 'lstm_vae':
                from hrl_anomaly_detection.vae.models import lstm_vae_state_batch as km
                ths_l = np.logspace(-1.0,3.2,40) -0.1
            #------------------------------------------------------------------
            elif method == 'lstm_vae_custom':
                from hrl_anomaly_detection.vae.models import lstm_vae_custom as km
                ths_l = np.logspace(-1.0,2.,40) -0.2
                x_std_div   = 4.
                x_std_offset= 0.1
                z_std       = 0.3 #0.2
                h1_dim      = nDim #8 #4 # raw
                phase       = 1.0
            elif method == 'lstm_dvae_custom':
                from hrl_anomaly_detection.vae.models import lstm_dvae_custom as km
                ths_l = np.logspace(-1.0,2.,40) -0.02
                x_std_div   = 4.
                x_std_offset= 0.1
                z_std       = 0.4 #5
                h1_dim      = nDim #8 #4 # raw
                phase       = 1.0
            #------------------------------------------------------------------
            elif method == 'lstm_vae_phase':
                from hrl_anomaly_detection.vae.models import lstm_vae_phase as km
                ths_l = np.logspace(-1.0,2.,40) -0.2
                x_std_div   = 4.
                x_std_offset= 0.1
                z_std       = 0.3 #0.2
                h1_dim      = nDim #8 #4 # raw
                phase       = 1.0
            elif method == 'lstm_dvae_phase':
                from hrl_anomaly_detection.vae.models import lstm_dvae_phase2 as km
                ths_l = np.logspace(-1.0,2.4,40) -0.2
                x_std_div   = 4.
                x_std_offset= 0.1
                z_std       = 1.0 
                h1_dim      = 4 #nDim
                z_dim       = 3
                phase       = 1.0
                sam_epoch   = 40
            elif method == 'lstm_dvae_phase_lastinput':
                from hrl_anomaly_detection.vae.models import lstm_dvae_phase_lastinput as km
                ths_l = np.logspace(-1.0,2.4,40) -0.2
                x_std_div   = 4.
                x_std_offset= 0.1
                z_std       = 1.0
                h1_dim      = 4 #nDim
                z_dim       = 3
                phase       = 1.0
                sam_epoch   = 40
            #------------------------------------------------------------------
            elif method == 'lstm_dvae_pred':
                from hrl_anomaly_detection.vae.models import lstm_dvae_pred as km            
                ths_l = np.logspace(-1.0,2.,40) -0.2
                x_std_div   = 4.
                x_std_offset= 0.1
                z_std       = 1.0 #3 
                h1_dim      = 4 #nDim 
                phase       = 0.5
            elif method == 'lstm_dvae_pred_phase':
                from hrl_anomaly_detection.vae.models import lstm_dvae_pred_phase as km            
                ths_l = np.logspace(-1.0,2.,40) -0.2
                x_std_div   = 4.
                x_std_offset= 0.1
                z_std       = 1.0 #3 
                h1_dim      = 2 #nDim #8 #4 # raw
                z_dim       = 3
                phase       = 1.0
            #------------------------------------------------------------------
            elif method == 'lstm_vae_custom3':
                from hrl_anomaly_detection.vae.models import lstm_vae_custom3 as km
                ths_l = np.logspace(-1.0,2.,40) -0.2
                x_std_offset= 0.05
                z_std       = 1.0
                sam_epoch   = 10
            elif method == 'lstm_vae2':
                from hrl_anomaly_detection.vae.models import lstm_vae_state_batch2 as km
                ths_l = np.logspace(-1.0,2.2,40) -0.5  
            else:
                from hrl_anomaly_detection.vae.models import lstm_dvae_state_batch as km
                ths_l = np.logspace(-1.0,2.2,40) -0.1
                
            autoencoder, vae_mean, _, enc_z_mean, enc_z_std, generator = \
              km.lstm_vae(trainData, valData, weights_path, patience=patience, batch_size=batch_size,
                          noise_mag=noise_mag, timesteps=window_size, sam_epoch=sam_epoch,
                          x_std_div=x_std_div, x_std_offset=x_std_offset, z_std=z_std,\
                          phase=phase, z_dim=z_dim, h1_dim=h1_dim, \
                          renew=ae_renew, fine_tuning=fine_tuning, plot=plot,\
                          scaler_dict=scaler_dict)
                          
        elif method == 'ae':
            from hrl_anomaly_detection.vae.models import ae
            window_size  = 3
            batch_size   = 256
            sam_epoch    = 20
            noise_mag    = 0.05
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
            from hrl_anomaly_detection.vae.models import lstm_ae_state_batch as km
            stateful = True
            ad_method   = 'recon_err'
            ths_l = np.logspace(-1.0,1.8,40) -0.5 
            autoencoder,_,_, enc_z_mean = \
              km.lstm_ae(trainData, valData, weights_path, patience=4, batch_size=batch_size,
                         noise_mag=noise_mag, timesteps=window_size, sam_epoch=sam_epoch,
                         re_load=re_load, renew=ae_renew, fine_tuning=fine_tuning, plot=plot)
            vae_mean = autoencoder
        elif method == 'encdec_ad':
            # EncDec-AD from Malhortra
            from hrl_anomaly_detection.vae.models import encdec_ad as km
            window_size = 3
            sam_epoch   = 40
            batch_size  = 256
            noise_mag   = 0.05
            fixed_batch_size = False
            stateful = False
            ad_method   = 'recon_err_lld'
            ths_l = np.logspace(-1.0,4.0,40) 
            autoencoder = \
              km.lstm_ae(trainData, valData, weights_path, patience=4, batch_size=batch_size,
                         noise_mag=noise_mag, timesteps=window_size, sam_epoch=sam_epoch,
                         renew=ae_renew, fine_tuning=fine_tuning, plot=plot)
            vae_mean = autoencoder
            
        elif method == 'lstm_vae_offline':
            from hrl_anomaly_detection.vae.models import lstm_vae_offline as km
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
                          remo_load=re_load, renew=ae_renew, fine_tuning=fine_tuning, plot=plot)
        elif method == 'rnd':
            autoencoder = None
            dyn_ths     = False
            window_size = 1
            ths_l = np.linspace(0.0,1.0,40)
        elif method == 'osvm':
            window_size = 3
            fixed_batch_size = False
            ad_method   = None
            ths_l = np.linspace(3e-1, 1.0, 40)
            autoencoder = None
            #autoencoder = km.osvm(trainData, valData, weights_path, timesteps=window_size,
            #                      renew=ae_renew)
            
        
        #------------------------------------------------------------------------------------
        if  True : 
            vutil.graph_latent_space(normalTestData, abnormalTestData, enc_z_mean,
                                     timesteps=window_size, batch_size=batch_size,
                                     method=method)
            
        # -----------------------------------------------------------------------------------
        if True and False:
            # get optimized alpha
            if fine_tuning: alpha_renew = True
            else: alpha_renew = False
            save_pkl = os.path.join(save_data_path, 'model_alpha_'+method+'_'+str(idx)+'.pkl')
            from hrl_anomaly_detection.vae import detector as dt
            alpha = dt.get_optimal_alpha((valData[0], abnormalTrainData), autoencoder, vae_mean,
                                         ad_method, method, window_size, save_pkl,\
                                         stateful=stateful, renew=alpha_renew,\
                                         x_std_div = x_std_div, x_std_offset=x_std_offset, z_std=z_std,
                                         dyn_ths=dyn_ths, batch_info=(fixed_batch_size,batch_size))            
        else:
            alpha = np.array([1.0]*nDim) #/float(nDim)
            alpha[0] = 1.
            #alpha[1:] = 1.0
            #alpha[4:11] = 0.5


        if fine_tuning: clf_renew=True
        normalTrainData = vutil.get_scaled_data2(d['successData'][:, normalTrainIdx, :],
                                                 scaler, aligned=False)

        if method == 'osvm':
            from hrl_anomaly_detection.vae.models import osvm as dt
        else:
            from hrl_anomaly_detection.vae import detector as dt
        
        save_pkl = os.path.join(save_data_path, 'model_ad_scores_'+str(idx)+'.pkl')
        tp_l, tn_l, fp_l, fn_l, roc = \
          dt.anomaly_detection(autoencoder, vae_mean, vae_logvar, enc_z_mean, enc_z_std, generator,
                               normalTrainData, valData[0],\
                               normalTestData, abnormalTestData, \
                               ad_method, method,
                               window_size, alpha, ths_l=ths_l, save_pkl=save_pkl, stateful=stateful,
                               x_std_div = x_std_div, x_std_offset=x_std_offset, z_std=z_std, \
                               phase=phase,\
                               plot=plot, param_dict=d['param_dict'],\
                               renew=clf_renew, dyn_ths=dyn_ths, batch_info=(fixed_batch_size,batch_size),\
                               scaler_dict=scaler_dict)

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


def ad_score_viz(task_name, raw_data_path, processed_data_path, param_dict):

    crossVal_pkl = os.path.join(processed_data_path, 'cv_'+task_name+'.pkl')
    d = ut.load_pickle(crossVal_pkl)         

    for idx, (normalTrainIdx, abnormalTrainIdx, normalTestIdx, abnormalTestIdx) \
      in enumerate(d['kFoldList']):

        if idx != 1: continue
        save_pkl = os.path.join(processed_data_path, 'model_ad_scores_'+str(idx)+'.pkl')

        dd = ut.load_pickle(save_pkl)
        ## scores_tr_n = d['scores_tr_n']
        scores_te_n = dd['scores_te_n']
        scores_te_a = dd['scores_te_a']
        
        vutil.graph_score_distribution(scores_te_n, scores_te_a, d['param_dict'], save_pdf=True)
    

if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    util.initialiseOptParser(p)
    
    p.add_option('--lstm_test', '--lt', action='store_true', dest='lstm_test',
                 default=False, help='Generate data.')
    p.add_option('--reload', '--rl', action='store_true', dest='bReLoad',
                 default=False, help='Reload previous parameters.')
    p.add_option('--fint_tuning', '--ftn', action='store_true', dest='bFineTune',
                 default=False, help='Run fine tuning.')
    p.add_option('--dyn_ths', '--dt', action='store_true', dest='bDynThs',
                 default=False, help='Run dynamic threshold.')
    p.add_option('--ad_score', '--as', action='store_true', dest='ad_score_viz',
                 default=False, help='Visualize anomaly scores.')

    opt, args = p.parse_args()

    #---------------------------------------------------------------------------           
    # Run evaluation
    #---------------------------------------------------------------------------           
    rf_center   = 'kinEEPos'        
    scale       = 1.0
    local_range = 10.0
    nPoints     = 40 #None
    opt.bHMMRenew = opt.bAERenew

    from hrl_anomaly_detection.vae.vae_params import *
    raw_data_path, save_data_path, param_dict = getParams(opt.task, opt.bDataRenew, \
                                                          opt.bHMMRenew, opt.bCLFRenew, opt.dim,\
                                                          rf_center, local_range, nPoints=nPoints)
    if opt.bNoUpdate: param_dict['ROC']['update_list'] = []
    subjects = ['s2', 's3','s4','s5', 's6','s7','s8', 's9']

    if os.uname()[1] == 'monty1':
        save_data_path = os.path.expanduser('~')+\
          '/hrl_file_server/dpark_data/anomaly/ICRA2018/'+opt.task+'_data_osvm_raw'
    else:
        save_data_path = os.path.expanduser('~')+\
          '/hrl_file_server/dpark_data/anomaly/ICRA2018/'+opt.task+'_data_lstm_dvae_phase_raw'


    param_dict['data_param']['handFeatures'] = ['unimodal_audioWristRMS',  \
                                                'unimodal_kinJntEff_1',\
                                                'unimodal_ftForce_integ',\
                                                'crossmodal_landmarkEEDist']

    if opt.lstm_test:
        lstm_test(subjects, opt.task, raw_data_path, save_data_path, param_dict, plot=not opt.bNoPlot,
                  re_load=opt.bReLoad, fine_tuning=opt.bFineTune, dyn_ths=opt.bDynThs)
    elif opt.ad_score_viz:
        ad_score_viz(opt.task, raw_data_path, save_data_path, param_dict)

