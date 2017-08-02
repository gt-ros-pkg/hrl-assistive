
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


def lstm_test(subject_names, task_name, raw_data_path, processed_data_path, param_dict, plot=False,
              re_load=False, fine_tuning=False, dyn_ths=False):
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
    d['successData']    = d['successData'][feature_list]
    d['failureData']    = d['failureData'][feature_list]

    if fine_tuning is False :
        subjects = ['Andrew', 'Britteney', 'Joshua', 'Jun', 'Kihan', 'Lichard', 'Shingshing', 'Sid', 'Tao']
        raw_data_path  = os.path.expanduser('~')+'/hrl_file_server/dpark_data/anomaly/RAW_DATA/CORL2017/'
        td1 = vutil.get_ext_data(subjects, task_name, raw_data_path, save_data_path, param_dict,
                          init_param_dict=d['param_dict'], id_num=0)

        subjects = ['ari', 'park', 'jina', 'linda', 'sai', 'hyun']
        raw_data_path  = os.path.expanduser('~')+'/hrl_file_server/dpark_data/anomaly/RAW_DATA/ICRA2017/'
        td2 = vutil.get_ext_data(subjects, task_name, raw_data_path, save_data_path, param_dict,
                          init_param_dict=d['param_dict'], id_num=1)

    # Parameters
    nDim = len(d['successData'])
    batch_size  = 1 #64

    #ths_l = -np.logspace(-1,0.8,40)+2.0
    ths_l = -np.logspace(-1,0.5,40)+1.5
    ths_l = np.linspace(127,133,40)
    #ths_l = np.logspace(0.2,1.8,40) #2.0  
    ths_l = np.logspace(-1.0,2.2,40) -0.1 


    tp_ll = [[] for i in xrange(len(ths_l))]
    fp_ll = [[] for i in xrange(len(ths_l))]
    tn_ll = [[] for i in xrange(len(ths_l))]
    fn_ll = [[] for i in xrange(len(ths_l))]

    # split data
    # HMM-induced vector with LOPO
    for idx, (normalTrainIdx, abnormalTrainIdx, normalTestIdx, abnormalTestIdx) \
      in enumerate(d['kFoldList']):
        if idx != 7: continue
        ## np.random.shuffle(normalTrainIdx)  

        # dim x sample x length
        ## normalTrainData   = d['successData'][:, normalTrainIdx, :]
        ## abnormalTrainData = d['failureData'][:, abnormalTrainIdx, :]
        ## normalTestData    = d['successData'][:, normalTestIdx, :]
        ## abnormalTestData  = d['failureData'][:, abnormalTestIdx, :]
        ## if fine_tuning is False and False:
        ##     normalTrainData   = np.hstack([normalTrainData, copy.deepcopy(td1['successData']), copy.deepcopy(td2['successData'])])
        ##     abnormalTrainData = np.hstack([abnormalTrainData, copy.deepcopy(td1['failureData']), copy.deepcopy(td2['failureData'])])
        normalTrainData   = np.hstack([copy.deepcopy(td1['successData']),
                                       copy.deepcopy(td2['successData'])])
        abnormalTrainData = np.hstack([copy.deepcopy(td1['failureData']),
                                       copy.deepcopy(td2['failureData'])])

        idx_list = range(len(normalTrainData))
        np.random.shuffle(idx_list)
        normalTrainData = normalTrainData[:,idx_list]


        normalTrainData, abnormalTrainData, normalTestData, abnormalTestData =\
          vutil.get_scaled_data(normalTrainData, abnormalTrainData, aligned=False)

        ## trainData = [normalTrainData, [0]*len(normalTrainData)]
        ## valData   = [normalTestData, [0]*len(normalTestData)]
        trainData = [normalTrainData[:int(len(normalTrainData)*0.7)],
                     [0]*len(normalTrainData[:int(len(normalTrainData)*0.7)])]
        valData   = [normalTrainData[int(len(normalTrainData)*0.7):],
                     [0]*len(normalTrainData[int(len(normalTrainData)*0.7):])]

        method      = 'lstm_dvae'

        # ------------------------------------------------------------------------------------------        
        weights_path = os.path.join(save_data_path,'model_weights_'+method+'_'+str(idx)+'.h5')
        vae_mean   = None
        vae_logvar = None
        enc_z_mean = enc_z_std = None
        generator  = None
        x_std_div   = None
        x_std_offset= None

        # ------------------------------------------------------------------------------------------
        window_size = 5
        batch_size  = 64
        fixed_batch_size = True
        noise_mag   = 0.1
        sam_epoch   = 20

        if method == 'lstm_vae' or method == 'lstm_vae2' or method == 'lstm_dvae':
            if method == 'lstm_vae':
                from hrl_anomaly_detection.vae import lstm_vae_state_batch as km
                ths_l = np.logspace(-1.0,2.2,40) -0.1  
            elif method == 'lstm_vae2':
                from hrl_anomaly_detection.vae import lstm_vae_state_batch2 as km
                ths_l = np.logspace(-1.0,2.2,40) -0.5  
            else:
                from hrl_anomaly_detection.vae import lstm_dvae_state_batch as km
                ths_l = np.logspace(-1.0,2.2,40) -0.1  
            x_std_div   = 2
            x_std_offset= 0.05
            z_std       = 0.4
            stateful = True
            ad_method   = 'lower_bound'
            for i in xrange(1):
                autoencoder, vae_mean, _, enc_z_mean, enc_z_std, generator = \
                  km.lstm_vae(trainData, valData, weights_path, patience=4, batch_size=batch_size,
                              noise_mag=noise_mag, timesteps=window_size, sam_epoch=sam_epoch,
                              x_std_div=x_std_div, x_std_offset=x_std_offset, z_std=z_std,
                              re_load=True, plot=False)#, trainable=i)
                
        elif method == 'lstm_ae':
            # LSTM-AE (Confirmed) %74.99
            from hrl_anomaly_detection.vae import lstm_ae_state_batch as km
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
            from hrl_anomaly_detection.vae import lstm_ae_state_batch as km
            window_size = 10
            stateful = True
            ad_method   = 'recon_err_likelihood'
            ths_l = np.logspace(-1.0,1.8,40) -0.5 
            autoencoder,_,_, enc_z_mean = \
              km.lstm_ae(trainData, valData, weights_path, patience=4, batch_size=batch_size,
                         noise_mag=noise_mag, timesteps=window_size, sam_epoch=sam_epoch,
                         re_load=re_load, renew=ae_renew, fine_tuning=fine_tuning, plot=plot)
            vae_mean = autoencoder
        elif method == 'lstm_vae_offline':
            from hrl_anomaly_detection.vae import lstm_vae_offline as km
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
        
        #------------------------------------------------------------------------------------
        if  True and False: 
            graph_latent_space(normalTestData, abnormalTestData, enc_z_mean, batch_size=batch_size,
                               method=method)



            
if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    util.initialiseOptParser(p)

    p.add_option('--pre_train', '--pt', action='store_true', dest='lstm_pretrain',
                 default=False, help='Pre-training.')
    p.add_option('--dyn_ths', '--dt', action='store_true', dest='bDynThs',
                 default=False, help='Run dynamic threshold.')

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
    # Mikako - bad camera
    # s1 - kaci - before camera calibration
    subjects = ['s2', 's3','s4','s5', 's6','s7','s8', 's9']

    if os.uname()[1] == 'monty1':
        ## save_data_path = os.path.expanduser('~')+\
        ##   '/hrl_file_server/dpark_data/anomaly/ICRA2018/'+opt.task+'_data_lstm'
        save_data_path = os.path.expanduser('~')+\
          '/hrl_file_server/dpark_data/anomaly/ICRA2018/'+opt.task+'_data_lstm_pretrain'
    else:
        ## save_data_path = os.path.expanduser('~')+\
        ##   '/hrl_file_server/dpark_data/anomaly/ICRA2018/'+opt.task+'_data_lstm_pretrain'
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

    param_dict['data_param']['handFeatures'] = ['unimodal_kinVel',\
                                                'unimodal_kinJntEff_1',\
                                                'unimodal_ftForce_zero',\
                                                'unimodal_ftForce_integ',\
                                                'unimodal_kinEEChange',\
                                                'unimodal_kinDesEEChange',\
                                                'crossmodal_landmarkEEDist', \
                                                'unimodal_audioWristRMS']

    param_dict['data_param']['handFeatures'] = ['unimodal_audioWristRMS',  \
                                                'unimodal_kinJntEff_1',\
                                                'unimodal_ftForce_integ',\
                                                'crossmodal_landmarkEEDist']

    if opt.lstm_pretrain:
        lstm_test(subjects, opt.task, raw_data_path, save_data_path, param_dict, plot=not opt.bNoPlot,
                  dyn_ths=opt.bDynThs)
