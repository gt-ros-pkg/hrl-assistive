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

#  \author Daehyung Park (Healthcare Robotics Lab, Georgia Tech.)

# system & utils
import os, sys, copy, random
import scipy, numpy as np
import hrl_lib.util as ut

# Private utils
## from hrl_anomaly_detection import util as util
## from hrl_anomaly_detection.util_viz import *
from hrl_anomaly_detection import util as util
from hrl_anomaly_detection import data_manager as dm

import hrl_anomaly_detection.IROS17_isolation.isolation_util as iutil
from hrl_execution_monitor import util as autil
from hrl_anomaly_detection.RAL18_detection import util as vutil


# Private learners
from hrl_anomaly_detection.hmm import learning_hmm as hmm


from sklearn import preprocessing
from joblib import Parallel, delayed
import gc

random.seed(3334)
np.random.seed(3334)


def get_data(subject_names, task_name, raw_data_path, save_data_path, param_dict, fine_tuning=False):

    # load params (param_dict)
    data_dict  = param_dict['data_param']
    AE_dict    = param_dict['AE']
    data_renew = data_dict['renew']
    
    #------------------------------------------
    if os.path.isdir(save_data_path) is False:
        os.system('mkdir -p '+save_data_path)

    crossVal_pkl = os.path.join(save_data_path, 'cv_'+task_name+'.pkl')
    if os.path.isfile(crossVal_pkl) and data_renew is False:
        print "CV data exists and no renew"
        d = ut.load_pickle(crossVal_pkl)        
    else:
        '''
        Use augmented data? if nAugment is 0, then aug_successData = successData
        '''        
        d = dm.getRawDataLOPO(subject_names, task_name, raw_data_path, \
                              save_data_path,\
                              downSampleSize=data_dict['downSampleSize'],\
                              handFeatures=data_dict['isolationFeatures'], \
                              rawFeatures=AE_dict['rawFeatures'], \
                              cut_data=data_dict['cut_data'], \
                              data_renew=data_renew, max_time=data_dict['max_time'],
                              ros_bag_image=True)

        (d['successData'], d['success_image_list'], d['success_d_image_list']),\
            (d['failureData'], d['failure_image_list'], d['failure_d_image_list']), \
          d['success_files'], d['failure_files'], d['kFoldList'] \
          = dm.LOPO_data_index(d['successRawDataList'], d['failureRawDataList'],\
                               d['successFileList'], d['failureFileList'],\
                               success_image_list = d['success_image_list'], \
                               failure_image_list = d['failure_image_list'], \
                               success_d_image_list = d['success_d_image_list'], \
                               failure_d_image_list = d['failure_d_image_list'])

        d['failure_labels']  = get_label_from_filename(d['failure_files'])

        ut.save_pickle(d, crossVal_pkl)

    print "Main data"
    ## print np.shape(d['successData']), np.shape(d['success_image_list']), np.shape(d['success_d_image_list'])
    ## print np.shape(d['failureData']), np.shape(d['failure_image_list']), np.shape(d['failure_d_image_list'])
    d['failure_labels']  = get_label_from_filename(d['failure_files'])

    if fine_tuning is False:
        td1, td2, td3 = vutil.get_ext_feeding_data(task_name, save_data_path, param_dict, d,
                                                   raw_feature=True, ros_bag_image=False)

        ## subjects = ['Andrew', 'Britteney', 'Joshua', 'Jun', 'Kihan', 'Lichard', 'Shingshing', 'Sid', 'Tao']
        ## raw_data_path  = os.path.expanduser('~')+'/hrl_file_server/dpark_data/anomaly/RAW_DATA/CORL2017/'
        ## td1 = vutil.get_ext_data(subjects, task_name, raw_data_path, save_data_path, param_dict,
        ##                         init_param_dict=d['param_dict'], init_raw_param_dict=d['raw_param_dict'],
        ##                         depth=True, id_num=1, raw_feature=True, ros_bag_image=True, kfold_split=True)

        ## subjects = ['s1','s2','s3','s4','s5','s6']
        ## raw_data_path  = os.path.expanduser('~')+'/hrl_file_server/dpark_data/anomaly/RAW_DATA/HENRY2017/'
        ## td4 = vutil.get_ext_data(subjects, task_name, raw_data_path, save_data_path, param_dict,
        ##                         init_param_dict=d['param_dict'], init_raw_param_dict=d['raw_param_dict'],
        ##                         depth=False, id_num=4, raw_feature=True, ros_bag_image=True, kfold_split=True)

        # Manually selected data?

        # Get main and sub data dictionary
        td = {}
        for key in td1.keys():
            if key in ['success_image_list', 'failure_image_list',
                       'success_d_image_list', 'failure_d_image_list',
                       'successRawDataList', 'failureRawDataList',
                       'successFileList', 'failureFileList',
                       'success_files', 'failure_files',
                       'failure_labels']:
                td[key] = td1[key]+td2[key]+td3[key]
                #td[key] = td1[key]+td4[key]
            elif key in ['successData', 'failureData']:
                td[key] = np.vstack([np.swapaxes(td1[key],0,1),
                                     np.swapaxes(td2[key],0,1),
                                     np.swapaxes(td3[key],0,1)])
                ## td[key] = np.vstack([np.swapaxes(td1[key],0,1),
                ##                      np.swapaxes(td4[key],0,1)])
                td[key] = np.swapaxes(td[key],0,1)
    else:
        td = None

    return d, td


def get_label_from_filename(file_names):

    labels = []
    for f in file_names:
        labels.append( int(f.split('/')[-1].split('_')[0]) )

    return labels

    

def get_detection_idx(save_data_path, main_data, sub_data, param_dict, verbose=False,
                      fine_tuning=False):
    
    # load params (param_dict)
    method     = param_dict['ROC']['methods'][0]
    nPoints    = param_dict['ROC']['nPoints']
    
    if param_dict['HMM']['renew']: clf_renew = True
    else:                          clf_renew = param_dict['SVM']['renew']

    # Check the list of temporal data and images
    nDim = len(main_data['successData'])
    tp_ll = [[] for i in xrange(nPoints)]
    fp_ll = [[] for i in xrange(nPoints)]
    tn_ll = [[] for i in xrange(nPoints)]
    fn_ll = [[] for i in xrange(nPoints)]
    roc_l = []
    train_a_idx_ll = []
    test_a_idx_ll  = []
    train_a_err_ll = []
    test_a_err_ll  = []

    detection_pkl = os.path.join(save_data_path, 'anomaly_idx.pkl')

    #-----------------------------------------------------------------------------------------
    # Anomaly Detection using lstm-dvae-phase
    #-----------------------------------------------------------------------------------------        
    # Leave-one-person-out cross validation
    for idx, (normalTrainIdx, abnormalTrainIdx, normalTestIdx, abnormalTestIdx) \
      in enumerate(main_data['kFoldList']):

        if clf_renew is False and os.path.isfile(detection_pkl) : break
        print "==================== ", idx, " ========================"
        
        # ------------------------------------------------------------------------------------------         
        # dim x sample x length
        normalTrainData     = main_data['successData'][:, normalTrainIdx, :]
        abnormalTrainData   = main_data['failureData'][:, abnormalTrainIdx, :]
        abnormalTrainLabels = [main_data['failure_labels'][i] for i in abnormalTrainIdx]
        
        normalTestData      = main_data['successData'][:, normalTestIdx, :]
        abnormalTestData    = main_data['failureData'][:, abnormalTestIdx, :]
        abnormalTestLabels  = [main_data['failure_labels'][i] for i in abnormalTestIdx]
        
        normalTrainData     = np.hstack([normalTrainData,
                                         copy.deepcopy(sub_data['successData'])])
        abnormalTrainData   = np.hstack([abnormalTrainData,
                                         copy.deepcopy(sub_data['failureData'])])
        abnormalTrainLabels = abnormalTrainLabels + sub_data['failure_labels']

        # shuffle
        np.random.seed(3334+idx)
        idx_list = range(len(normalTrainData[0]))
        np.random.shuffle(idx_list)
        normalTrainData = normalTrainData[:,idx_list]

        method       = 'lstm_dvae_phase'
        scaler_file = os.path.join(save_data_path,'scaler_'+method+'_'+str(idx)+'.pkl')
        if os.path.isfile(scaler_file):
            scaler_dict = ut.load_pickle(scaler_file)            
            normalTrainData, abnormalTrainData, normalTestData, abnormalTestData, scaler =\
              vutil.get_scaled_data(normalTrainData, abnormalTrainData,
                                    normalTestData, abnormalTestData, aligned=False,
                                    scaler=scaler_dict['scaler'], )
        else:
            normalTrainData, abnormalTrainData, normalTestData, abnormalTestData, scaler =\
              vutil.get_scaled_data(normalTrainData, abnormalTrainData,
                                    normalTestData, abnormalTestData, aligned=False)

        trainData = [normalTrainData[:int(len(normalTrainData)*0.7)],
                     [0]*len(normalTrainData[:int(len(normalTrainData)*0.7)])]
        valData   = [normalTrainData[int(len(normalTrainData)*0.7):],
                     [0]*len(normalTrainData[int(len(normalTrainData)*0.7):])]
        ## testData  = [normalTestData, [0]*len(normalTestData)]

        # ------------------------------------------------------------------------------------------         
        # scaling info to reconstruct the original scale of data
        scaler_dict  = {'scaler': scaler, 'scale': 1, 'param_dict': main_data['raw_param_dict']}

        ae_renew   = False
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
        vae_logvar   = None

        weights_path = os.path.join(save_data_path,'model_weights_'+method+'_'+str(idx)+'.h5')
        
        dyn_ths   = True 
        ad_method = 'lower_bound'
        stateful    = True
        x_std_div   = 4.
        x_std_offset= 0.1
        z_std       = 1.0 
        h1_dim      = 4 #nDim
        z_dim       = 2 #3
        phase       = 1.0
        sam_epoch   = 40
        plot = False
        fixed_batch_size = True
        batch_size  = 256


        from hrl_anomaly_detection.journal_isolation.models import lstm_dvae_phase_circle as km
        ## from hrl_anomaly_detection.RAL18_detection.models import lstm_dvae_phase2 as km
        ## autoencoder, vae_mean, _, enc_z_mean, enc_z_std, generator = \
        ##   km.lstm_vae(trainData, valData, weights_path, patience=4, batch_size=256,
        ##               noise_mag=0.05, timesteps=1, sam_epoch=40,
        ##               x_std_div=4., x_std_offset=0.1, z_std=1.0,\
        ##               phase=1., z_dim=3, h1_dim=4, \
        ##               renew=param_dict['HMM']['renew'], fine_tuning=fine_tuning, plot=False,\
        ##               scaler_dict=scaler_dict)

        autoencoder, vae_mean, _, enc_z_mean, enc_z_std, generator = \
          km.lstm_vae(trainData, valData, weights_path, patience=patience, batch_size=batch_size,
                      noise_mag=noise_mag, timesteps=window_size, sam_epoch=sam_epoch,
                      x_std_div=x_std_div, x_std_offset=x_std_offset, z_std=z_std,\
                      phase=phase, z_dim=z_dim, h1_dim=h1_dim, \
                      renew=ae_renew, fine_tuning=fine_tuning, plot=plot,\
                      scaler_dict=scaler_dict)
                      
        #------------------------------------------------------------------------------------
        ## vutil.graph_latent_space(normalTestData, abnormalTestData, enc_z_mean,
        ##                          timesteps=window_size, batch_size=batch_size,
        ##                          method=method)
        ## sys.exit()

        alpha = np.array([1.0]*nDim) #/float(nDim)
        alpha[0] = 1.
        ths_l = np.logspace(0.,1.3,nPoints) #- 0.12
        ths_l = np.logspace(0.3,1.4,nPoints) #- 0.12

        from hrl_anomaly_detection.journal_isolation import detector as dt
        save_pkl = os.path.join(save_data_path, 'model_ad_scores_'+str(idx)+'.pkl')
        tp_l, tn_l, fp_l, fn_l, roc, ad_dict = \
          dt.anomaly_detection(autoencoder, vae_mean, vae_logvar, enc_z_mean, enc_z_std, generator,
                               normalTrainData, valData[0], abnormalTrainData,\
                               normalTestData, abnormalTestData, \
                               ad_method, method,
                               window_size, alpha, ths_l=ths_l, save_pkl=save_pkl, stateful=True,
                               x_std_div = x_std_div, x_std_offset=x_std_offset, z_std=z_std, \
                               phase=phase, plot=False, \
                               renew=clf_renew, dyn_ths=dyn_ths, batch_info=(True,batch_size),\
                               param_dict=main_data['param_dict'], scaler_dict=scaler_dict,\
                               filenames=(np.array(main_data['success_files'])[normalTestIdx],
                                          np.array(main_data['failure_files'])[abnormalTestIdx]),\
                               return_idx=True)

        ## from hrl_anomaly_detection.RAL18_detection import detector as dt
        ## save_pkl = os.path.join(save_data_path, 'model_ad_scores_'+str(idx)+'.pkl')
        ## tp_l, tn_l, fp_l, fn_l, roc = \
        ##   dt.anomaly_detection(autoencoder, vae_mean, vae_logvar, enc_z_mean, enc_z_std, generator,
        ##                        normalTrainData, valData[0],\
        ##                        normalTestData, abnormalTestData, \
        ##                        ad_method, method,
        ##                        1, alpha, ths_l=ths_l, save_pkl=save_pkl, stateful=stateful,
        ##                        x_std_div = x_std_div, x_std_offset=x_std_offset, z_std=z_std, \
        ##                        phase=phase, plot=plot, \
        ##                        renew=clf_renew, dyn_ths=dyn_ths, batch_info=(fixed_batch_size,batch_size),\
        ##                        param_dict=main_data['param_dict'], scaler_dict=scaler_dict,\
        ##                        filenames=(np.array(main_data['success_files'])[normalTestIdx],
        ##                                   np.array(main_data['failure_files'])[abnormalTestIdx]))
        

        roc_l.append(roc)        
        ## train_a_idx_ll.append(ad_dict['tr_a_idx'])
        ## test_a_idx_ll.append(ad_dict['te_a_idx'])
        ## train_a_err_ll.append(ad_dict['tr_a_err'])
        ## test_a_err_ll.append(ad_dict['te_a_err'])

        for i in xrange(len(ths_l)):
            tp_ll[i] += tp_l[i]
            fp_ll[i] += fp_l[i]
            tn_ll[i] += tn_l[i]
            fn_ll[i] += fn_l[i]
   
    #--------------------------------------------------------------------
    if clf_renew or os.path.isfile(detection_pkl) is False:
        print "roc list ", roc_l

        # f-scores
        fs_l = []
        for i in xrange(len(ths_l)):
            fs_l.append( (2.0*float(np.sum(tp_ll[i])))/ (2.0*float(np.sum(tp_ll[i])) + float(np.sum(fn_ll[i])) + float(np.sum(fp_ll[i]))) )

        dd = {}
        dd['tp_ll'] = tp_ll
        dd['fp_ll'] = fp_ll
        dd['tn_ll'] = tn_ll
        dd['fn_ll'] = fn_ll
        ## d['roc_l'] = roc_l
        dd['train_idx_ll'] = train_a_idx_ll
        dd['test_idx_ll']  = test_a_idx_ll
        dd['train_err_ll'] = train_a_err_ll
        dd['test_err_ll']  = test_a_err_ll
        dd['train_labels'] = abnormalTrainLabels
        dd['test_labels']  = abnormalTestLabels
        dd['fs_l']  = fs_l
        dd['kFoldList'] = main_data['kFoldList']
        ut.save_pickle(dd, detection_pkl)
    else:
        dd = ut.load_pickle(detection_pkl)


    tpr_l = []
    fpr_l = []
    for i in xrange(nPoints):
        tpr_l.append( float(np.sum(dd['tp_ll'][i]))/float(np.sum(dd['tp_ll'][i])+np.sum(dd['fn_ll'][i]))*100.0 )
        fpr_l.append( float(np.sum(dd['fp_ll'][i]))/float(np.sum(dd['fp_ll'][i])+np.sum(dd['tn_ll'][i]))*100.0 ) 

    print "------------------------------------------------------"
    print tpr_l
    print fpr_l

    from sklearn import metrics
    print "roc: ", metrics.auc(fpr_l, tpr_l, True)  

    return dd

    
def get_isolation_data(subject_names, task_name, raw_data_path, save_data_path, param_dict,
                       fine_tuning=False):
                       #main_data, sub_data,
                       #train_idx_list, test_idx_list, ths_idx, param_dict):

    # Get Raw Data
    main_data, sub_data = get_data(subject_names, task_name, raw_data_path, save_data_path, param_dict,
                                   fine_tuning=fine_tuning)
    # Get a trained detector

    #temp
    #main_data['kFoldList'] = main_data['kFoldList'][0:1]

    # Get detection indices and corresponding features
    dt_dict = get_detection_idx(save_data_path, main_data, sub_data, param_dict, fine_tuning=fine_tuning,
                                verbose=False)

    # Classification?
    x_train_s = []
    x_train_i = []
    x_train_d = []
    y_train   = []

    x_test_s = []
    x_test_i = []
    x_test_d = []
    y_test   = []
    
    #-----------------------------------------------------------------------------------------
    # Data extraction
    #-----------------------------------------------------------------------------------------        
    for idx, (normalTrainIdx, abnormalTrainIdx, normalTestIdx, abnormalTestIdx) \
      in enumerate(dt_dict['kFoldList']):
        print "==================== ", idx, " ========================"

        # Find an index given maximum f-score
        ths_idx = np.argmax(dt_dict['fs_l'][idx])


        print np.shape(dt_dict['train_err_ll'][idx]), np.shape(dt_dict['train_labels']),
        print np.shape(dt_dict['test_err_ll'][idx]), np.shape(dt_dict['test_labels'])        
        print "00000000000000000000000"        
        ## iutil.save_data_labels(dt_dict['train_err_ll'][idx], dt_dict['train_labels'])        

        # feature extraction by index based on IROS17_isolation/isolation_util.py's feature_extraction
        #1) Individual features reconstruction probability?
        #2) Image list
        feature_extraction(idx,\
                           (dt_dict['train_idx_ll'][idx][ths_idx],
                            dt_dict['train_err_ll'][idx],
                           [main_data['failure_d_image_list'][i] for i in abnormalTrainIdx]
                           + sub_data['failure_d_image_list'],
                           dt_dict['train_labels']),\
                           (dt_dict['test_idx_ll'][idx][ths_idx],
                            dt_dict['test_err_ll'][idx],
                           [main_data['failure_d_image_list'][i] for i in abnormalTestIdx],
                           dt_dict['test_labels']),\
                           save_data_path)



        sys.exit()


        


        # train sig net
        sig_weights_file=os.path.join(save_data_path,'sig_weights_'+str(idx)+'.h5')
        sig_net([x_train_s, y_train], [x_test_s, y_test], noise_mag=0,
                save_weights_file=sig_weights_file)
        
        

    return [x_train, x_train_img], y_train, [x_test, x_test_img], y_test    


def feature_extraction(idx, trainData, testData, save_data_path, window_step=10):

    ''' Get individual reconstruction probability vector when anomalies are detected '''
    train_d_idx_l  = trainData[0]
    train_err_l  = trainData[1] # sample x length x dim
    train_dimg_l = trainData[2]
    train_labels = trainData[3]

    test_d_idx_l  = testData[0]
    test_err_l  = testData[1]
    test_dimg_l = testData[2]
    test_labels = testData[3]

## anomaly_idx_list, abnormalData, abnormalData_s, \
##                        abnormalLabel, abnormalData_img,\
##                        task_name, processed_data_path, param_dict,\
##                        window_step=10, verbose=False, plot=False,\
##                        window=False, delta_flag=False):

    max_step = 8

    def features(d_idx_l, x_sig, x_img, y):
        for i, d_idx in enumerate(d_idx_l):
            # Skip undetected anomaly
            if d_idx is None: continue
        
            for j in range(-window_step, window_step):
                if d_idx+j <= 4: continue
                if d_idx+j > len(x_sig[0][0,i]): continue

                # pyramid pooling? (1,4,8)
                vs = np.amin(x_sig[i][j], axis=0)
                vs = np.vstack([ vs, np.amin(x_sig[i][:4], axis=0) ])
                vs = np.vstack([ vs, np.amin(x_sig[i][:8], axis=0) ])
                vs = vs.flatten()


    features(train_d_idx_l, train_err_l, train_dimg_l, train_labels)
    features(test_d_idx_l, test_err_l, test_dimg_l, test_labels)


    ## x = []
    ## y = []
    ## x_img = []
    ## for i, d_idx in enumerate(anomaly_idx_list):
    ##     # Skip undetected anomaly
    ##     if d_idx is None: continue

    ##     if window:
    ##         for j in range(-window_step, window_step):
    ##             ## for j in range(0, window_step):
    ##             if d_idx+j <= 4: continue
    ##             if d_idx+j > len(abnormalData[0][0,i]): continue

    ##             vs = None # step x feature
    ##             for ii in xrange(nDetector):
    ##                 v = temporal_features(abnormalData[ii][:,i], d_idx+j, max_step, ml_list[ii],
    ##                                       scale_list[ii])
    ##                 if vs is None: vs = v
    ##                 else: vs = np.hstack([vs, v])

    ##             if delta_flag:
    ##                 #2,4,8
    ##                 cp_vecs = np.amin(vs[:1], axis=0)
    ##                 cp_vecs = np.vstack([ cp_vecs, np.amin(vs[:4], axis=0) ])
    ##                 cp_vecs = np.vstack([ cp_vecs, np.amin(vs[:8], axis=0) ])
    ##                 cp_vecs = cp_vecs.flatten()
    ##             else:
    ##                 cp_vecs = np.amin(vs[:1], axis=0)

    ##             s_idx = d_idx+j-20
    ##             if s_idx <0: s_idx = 0
    ##             max_vals = np.amax(abnormalData_s[:,i,s_idx:d_idx+j], axis=1)
    ##             min_vals = np.amin(abnormalData_s[:,i,s_idx:d_idx+j], axis=1)
    ##             vals = [mx if abs(mx) > abs(mi) else mi for (mx, mi) in zip(max_vals, min_vals) ]

    ##             cp_vecs = cp_vecs.tolist()+ vals
    ##             if np.isnan(cp_vecs).any() or np.isinf(cp_vecs).any():
    ##                 print "NaN in cp_vecs ", i, d_idx
    ##                 sys.exit()

    ##             x.append( cp_vecs )
    ##             y.append( abnormalLabel[i] )
    ##             if abnormalData_img is not None and abnormalData_img[i] is not None:
    ##                 x_img.append( abnormalData_img[i][d_idx+j-1] )
    ##             else:
    ##                 x_img.append(None)

    ##     else:
    ##         if d_idx <= 0: continue
    ##         if d_idx > len(abnormalData[0][0,i]): continue                    

    ##         vs = None # step x feature
    ##         for ii in xrange(nDetector):
    ##             v = temporal_features(abnormalData[ii][:,i], d_idx, max_step, ml_list[ii],
    ##                                   scale_list[ii])
    ##             if vs is None: vs = v
    ##             else: vs = np.hstack([vs, v])

    ##         if delta_flag:
    ##             #1,4,8
    ##             cp_vecs = np.amin(vs[:1], axis=0)
    ##             cp_vecs = np.vstack([ cp_vecs, np.amin(vs[:4], axis=0) ])
    ##             cp_vecs = np.vstack([ cp_vecs, np.amin(vs[:8], axis=0) ])
    ##             cp_vecs = cp_vecs.flatten()
    ##         else:
    ##             cp_vecs = np.amin(vs[:1], axis=0)


    ##         s_idx = d_idx-20
    ##         if s_idx <0: s_idx = 0
    ##         max_vals = np.amax(abnormalData_s[:,i,s_idx:d_idx], axis=1)
    ##         min_vals = np.amin(abnormalData_s[:,i,s_idx:d_idx], axis=1)
    ##         vals = [mx if abs(mx) > abs(mi) else mi for (mx, mi) in zip(max_vals, min_vals) ]

    ##         cp_vecs = cp_vecs.tolist()+ vals
    ##         x.append( cp_vecs )
    ##         y.append( abnormalLabel[i] )
    ##         if abnormalData_img is not None and abnormalData_img[i] is not None:
    ##             x_img.append( abnormalData_img[i][d_idx-1] )
    ##         else:
    ##             x_img.append( None )


    ##     if len(x_img) == 0:
    ##         print "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    ##         print np.shape(abnormalData_img)
    ##         print x_img
    ##         print "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

    ## return x, y, x_img
    ## ## return sig, img, d_img

                      

if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    util.initialiseOptParser(p)
    opt, args = p.parse_args()

    from hrl_anomaly_detection.journal_isolation.isolation_param import *
    # IROS2017
    subject_names = ['s2', 's3','s4','s5', 's6','s7','s8', 's9']
    raw_data_path, save_data_path, param_dict = getParams(opt.task, opt.bDataRenew, \
                                                          opt.bHMMRenew, opt.bCLFRenew)
    save_data_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/JOURNAL_ISOL/'+opt.task+'_1'
    save_data_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/JOURNAL_ISOL/'+opt.task+'_2'


    window_steps= 5
    task_name = 'feeding'
    nb_classes = 12


    get_isolation_data(subject_names, task_name, raw_data_path, save_data_path, param_dict)

    #, weight=1.0, window_steps=window_steps, verbose=False)




