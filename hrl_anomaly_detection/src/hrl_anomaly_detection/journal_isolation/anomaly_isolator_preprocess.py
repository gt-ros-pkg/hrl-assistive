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

IROS_TEST = False
JOURNAL_TEST = None

def get_data(subject_names, task_name, raw_data_path, save_data_path, param_dict, fine_tuning=False):

    # load params (param_dict)
    data_dict  = param_dict['data_param']
    AE_dict    = param_dict['AE']
    data_renew = data_dict['renew']
    if IROS_TEST: ros_bag_image = False
    elif JOURNAL_TEST: ros_bag_image = True
    else: ros_bag_image = True
    
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
                              ros_bag_image=ros_bag_image)

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
    if IROS_TEST or JOURNAL_TEST is not None:
        d['failure_labels']  = get_label_from_filename(d['failure_files'])
    else:
        print np.shape(d['successData']), np.shape(d['success_image_list']), np.shape(d['success_d_image_list'])
        print np.shape(d['failureData']), np.shape(d['failure_image_list']), np.shape(d['failure_d_image_list'])

    if fine_tuning is False:
        if IROS_TEST or JOURNAL_TEST is not None:        
            td1, td2, td3 = vutil.get_ext_feeding_data(task_name, save_data_path, param_dict, d,
                                                       raw_feature=True, ros_bag_image=ros_bag_image)
            td1['failure_labels']  = get_label_from_filename(td1['failure_files'], offset=1)
            td2['failure_labels']  = get_label_from_filename(td2['failure_files'])
            td3['failure_labels']  = get_label_from_filename(td3['failure_files'])
        else:
            subjects = ['Andrew', 'Britteney', 'Joshua', 'Jun', 'Kihan', 'Lichard', 'Shingshing', 'Sid', 'Tao']
            raw_data_path  = os.path.expanduser('~')+'/hrl_file_server/dpark_data/anomaly/RAW_DATA/CORL2017/'
            td1 = vutil.get_ext_data(subjects, task_name, raw_data_path, save_data_path, param_dict,
                                    init_param_dict=d['param_dict'], init_raw_param_dict=d['raw_param_dict'],
                                    depth=True, id_num=1, raw_feature=True,
                                    ros_bag_image=ros_bag_image, kfold_split=True)
            td1['failure_labels']  = get_label_from_filename(td1['failure_files'], offset=1)

        subjects = ['s1','s2','s3','s4','s5','s6']
        raw_data_path  = os.path.expanduser('~')+'/hrl_file_server/dpark_data/anomaly/RAW_DATA/HENRY2017/'
        td4 = vutil.get_ext_data(subjects, task_name, raw_data_path, save_data_path, param_dict,
                                 init_param_dict=d['param_dict'], init_raw_param_dict=d['raw_param_dict'],
                                 depth=False, id_num=4, raw_feature=True,
                                 ros_bag_image=ros_bag_image, kfold_split=True)
        td4['failure_labels']  = get_label_from_filename(td4['failure_files'])
        
        
        # Manually selected data?
        ## print np.unique(d['failure_labels'])
        ## print np.unique(td1['failure_labels'])
        ## print np.unique(td2['failure_labels'])
        ## print np.unique(td3['failure_labels'])
        ## print np.unique(td4['failure_labels'])
        ## sys.exit()

        # Get main and sub data dictionary
        td = {}
        for key in td1.keys():
            if key in ['success_image_list', 'failure_image_list',
                       'success_d_image_list', 'failure_d_image_list',
                       'successRawDataList', 'failureRawDataList',
                       'successFileList', 'failureFileList',
                       'success_files', 'failure_files',
                       'failure_labels']:
                if IROS_TEST:
                    td[key] = td1[key]+td2[key]+td3[key]
                elif JOURNAL_TEST is not None:
                    td[key] = td1[key]+td2[key]+td3[key]+td4[key]
                else:
                    td[key] = td1[key]+td4[key]
            elif key in ['successData', 'failureData']:
                if IROS_TEST:
                    td[key] = np.vstack([np.swapaxes(td1[key],0,1),
                                         np.swapaxes(td2[key],0,1),
                                         np.swapaxes(td3[key],0,1)])
                elif JOURNAL_TEST is not None:
                    td[key] = np.vstack([np.swapaxes(td1[key],0,1),
                                         np.swapaxes(td2[key],0,1),
                                         np.swapaxes(td3[key],0,1),
                                         np.swapaxes(td4[key],0,1)])
                else:
                    td[key] = np.vstack([np.swapaxes(td1[key],0,1),
                                         np.swapaxes(td4[key],0,1)])
                td[key] = np.swapaxes(td[key],0,1)
    else:
        td = None

    return d, td


def get_label_from_filename(file_names, offset=0):

    labels = []
    for f in file_names:
        try:
            labels.append( int(f.split('/')[-1].split('_')[0])+offset )
        except:
            labels.append(None)

    return labels

    

def get_detection_idx(method, save_data_path, main_data, sub_data, param_dict, verbose=False,
                      dyn_ths=False, scale=1.8, fine_tuning=False, tr_only=False, te_only=False,
                      isol_only=False, latent_plot=False):
    
    # load params (param_dict)
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
    train_a_img_ll = []
    test_a_img_ll  = []

    if isol_only is False:
        detection_pkl = os.path.join(save_data_path, 'anomaly_idx.pkl')
    else:
        detection_pkl = os.path.join(save_data_path, 'anomaly_ai_idx.pkl')

    #-----------------------------------------------------------------------------------------
    # Anomaly Detection using lstm-dvae-phase
    #-----------------------------------------------------------------------------------------        
    # Leave-one-person-out cross validation
    for idx, (normalTrainIdx, abnormalTrainIdx, normalTestIdx, abnormalTestIdx) \
      in enumerate(main_data['kFoldList']):

        #if idx==0: continue
        #if clf_renew is False and os.path.isfile(detection_pkl) and latent_plot is False: break
        print "==================== ", idx, " ========================"
        
        # ------------------------------------------------------------------------------------         
        # dim x sample x length
        normalTrainData     = main_data['successData'][:, normalTrainIdx, :]
        abnormalTrainData   = main_data['failureData'][:, abnormalTrainIdx, :]
        abnormalTrainImg    = [main_data['failure_d_image_list'][i] for i in abnormalTrainIdx]
        abnormalTrainLabels = [main_data['failure_labels'][i] for i in abnormalTrainIdx]
        
        normalTestData      = main_data['successData'][:, normalTestIdx, :]
        abnormalTestData    = main_data['failureData'][:, abnormalTestIdx, :]
        abnormalTestImg     = [main_data['failure_d_image_list'][i] for i in abnormalTestIdx]
        abnormalTestLabels  = [main_data['failure_labels'][i] for i in abnormalTestIdx]

        if fine_tuning is False and te_only is False:
            normalTrainData     = np.hstack([normalTrainData,
                                             copy.deepcopy(sub_data['successData'])])
            abnormalTrainData   = np.hstack([abnormalTrainData,
                                             copy.deepcopy(sub_data['failureData'])])
            abnormalTrainImg    = abnormalTrainImg + sub_data['failure_d_image_list']
            abnormalTrainLabels = abnormalTrainLabels + sub_data['failure_labels']
        elif isol_only:
            abnormalTrainData   = np.hstack([abnormalTrainData,
                                             copy.deepcopy(sub_data['failureData'])])
            abnormalTrainImg    = abnormalTrainImg + sub_data['failure_d_image_list']
            abnormalTrainLabels = abnormalTrainLabels + sub_data['failure_labels']
            
            
        # shuffle
        np.random.seed(3334+idx)
        idx_list = range(len(normalTrainData[0]))
        np.random.shuffle(idx_list)
        normalTrainData = normalTrainData[:,idx_list]

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
                                    normalTestData, abnormalTestData, aligned=False, scale=scale)

        trainData = [normalTrainData[:int(len(normalTrainData)*0.7)],
                     [0]*len(normalTrainData[:int(len(normalTrainData)*0.7)])]
        valData   = [normalTrainData[int(len(normalTrainData)*0.7):],
                     [0]*len(normalTrainData[int(len(normalTrainData)*0.7):])]
        ## testData  = [normalTestData, [0]*len(normalTestData)]

        # ------------------------------------------------------------------------------------------         
        # scaling info to reconstruct the original scale of data
        scaler_dict = {'scaler': scaler, 'scale': 1, 'param_dict': main_data['raw_param_dict']}

        vae_logvar   = None
        window_size  = 1
        noise_mag    = 0.05
        patience     = 10 #4 #10
        
        ad_method    = 'lower_bound'
        stateful     = True
        x_std_div    = 4.
        x_std_offset = 0.1
        z_std        = 1.0 #2:1.0 
        h1_dim       = 4 #nDim
        z_dim        = 2 #3
        phase        = 1.0
        sam_epoch    = 40 #2:40
        plot         = False
        fixed_batch_size = True
        batch_size   = 256

        if method == 'lstm_dvae_phase_circle_kl':
            from hrl_anomaly_detection.journal_isolation.models import lstm_dvae_phase_circle_kl as km
        elif method == 'lstm_dvae_phase_circle':
            from hrl_anomaly_detection.journal_isolation.models import lstm_dvae_phase_circle as km
        else:
            from hrl_anomaly_detection.journal_isolation.models import lstm_dvae_phase2 as km 
            
        weights_path = os.path.join(save_data_path,'model_weights_'+method+'_'+str(idx)+'.h5')
        autoencoder, vae_mean, _, enc_z_mean, enc_z_std, generator = \
          km.lstm_vae(trainData, valData, weights_path, patience=patience, batch_size=batch_size,
                      noise_mag=noise_mag, timesteps=window_size, sam_epoch=sam_epoch,
                      x_std_div=x_std_div, x_std_offset=x_std_offset, z_std=z_std,\
                      phase=phase, z_dim=z_dim, h1_dim=h1_dim, \
                      renew=param_dict['HMM']['renew'], fine_tuning=fine_tuning, plot=plot,\
                      scaler_dict=scaler_dict)
        if tr_only: continue
                      
        #------------------------------------------------------------------------------------
        ## if latent_plot:
        ##     vutil.graph_latent_space(normalTestData, abnormalTestData, enc_z_mean,
        ##                              timesteps=window_size, batch_size=batch_size,
        ##                              method=method)
        ##     sys.exit()
        
        alpha    = np.array([1.0]*nDim) #/float(nDim)
        alpha[0] = 1.
        ths_l = np.logspace(0.,1.3,nPoints) #- 0.12
        ths_l = np.logspace(-0.4,1.8,nPoints) - 0.2 # 2
        #ths_l = np.logspace(-0.4,2.1,nPoints) - 0.2    
        clf_method = 'SVR'

        from hrl_anomaly_detection.journal_isolation import detector as dt
        if isol_only is False:
            save_pkl = os.path.join(save_data_path, 'model_ad_scores_'+str(idx)+'.pkl')
        else:
            save_pkl = os.path.join(save_data_path, 'model_ai_scores_'+str(idx)+'.pkl')
        tp_l, tn_l, fp_l, fn_l, roc, ad_dict = \
          dt.anomaly_detection(autoencoder, vae_mean, vae_logvar, enc_z_mean, enc_z_std, generator,
                               normalTrainData, valData[0], abnormalTrainData,\
                               normalTestData, abnormalTestData, \
                               ad_method, method,
                               window_size, alpha, ths_l=ths_l, save_pkl=save_pkl, stateful=True,
                               x_std_div = x_std_div, x_std_offset=x_std_offset, z_std=z_std, \
                               phase=phase, latent_plot=latent_plot, \
                               renew=clf_renew, dyn_ths=dyn_ths, batch_info=(True,batch_size),\
                               param_dict=main_data['param_dict'], scaler_dict=scaler_dict,\
                               filenames=(np.array(main_data['success_files'])[normalTestIdx],
                                          np.array(main_data['failure_files'])[abnormalTestIdx]),\
                               clf_method=clf_method,\
                               clf_file=os.path.join(save_data_path, 'clf_'+clf_method+'_'+str(idx)+'.sav'),\
                               return_idx=True)        

        roc_l.append(roc)        
        train_a_idx_ll.append(ad_dict['tr_a_idx'])
        test_a_idx_ll.append(ad_dict['te_a_idx'])
        train_a_err_ll.append(ad_dict['tr_a_err'])
        test_a_err_ll.append(ad_dict['te_a_err'])
        train_a_img_ll.append(abnormalTrainImg)
        test_a_img_ll.append(abnormalTestImg)

        for i in xrange(len(ths_l)):
            tp_ll[i] += tp_l[i]
            fp_ll[i] += fp_l[i]
            tn_ll[i] += tn_l[i]
            fn_ll[i] += fn_l[i]
   
    #--------------------------------------------------------------------
    if tr_only: return {}        
    if clf_renew or os.path.isfile(detection_pkl) is False or True:
        print "roc list ", roc_l
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
        dd['train_img_ll'] = train_a_img_ll
        dd['test_img_ll']  = test_a_img_ll
        dd['train_labels'] = abnormalTrainLabels
        dd['test_labels']  = abnormalTestLabels
        dd['ths_l'] = ths_l
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
    print "------------------------------------------------------"
    ths_idx = 20
    print tpr_l[ths_idx], fpr_l[ths_idx]
    print "------------------------------------------------------"

    

    from sklearn import metrics
    print "roc: ", metrics.auc(fpr_l, tpr_l, True)  

    return dd

    
def get_isolation_data(method, subject_names, task_name, raw_data_path, save_data_path, param_dict,
                       fine_tuning=False, dyn_ths=False, tr_only=False, te_only=False, isol_only=False,
                       latent_plot=False):

    # Get Raw Data
    main_data, sub_data = get_data(subject_names, task_name, raw_data_path, save_data_path, param_dict,
                                   fine_tuning=fine_tuning)
    
    # Get detection indices and corresponding features
    dt_dict = get_detection_idx(method, save_data_path, main_data, sub_data, param_dict,
                                fine_tuning=fine_tuning,
                                dyn_ths=dyn_ths, scale=1.8, tr_only=tr_only, te_only=te_only,
                                isol_only=isol_only,
                                latent_plot=latent_plot, verbose=False)


def preprocess_data(subject_names, task_name, raw_data_path, save_data_path, param_dict,
                    fine_tuning=False, latent_plot=False,\
                    img_scale=0.25, img_feature_type='vgg', **kwargs):

    # Get Raw Data
    main_data, sub_data = get_data(subject_names, task_name, raw_data_path, save_data_path, param_dict,
                                   fine_tuning=fine_tuning)
    
    detection_pkl = os.path.join(save_data_path, 'anomaly_ai_idx.pkl')
    dt_dict = ut.load_pickle(detection_pkl)
    
    # Find an index given maximum f-score
    ## # f-scores
    ## fs1_l = []
    ## fs05_l = []
    ## fs03_l = []
    ## for i in xrange(len(dt_dict['ths_l'])):
    ##     #f1_score
    ##     fs1_l.append( util.fscore(float(np.sum(dt_dict['tp_ll'][i])),
    ##                               float(np.sum(dt_dict['fn_ll'][i])),
    ##                               float(np.sum(dt_dict['fp_ll'][i])),
    ##                               beta=1.0) )
    ## ths_idx = np.argmax(fs03_l)
    ## print "THS_IDX(f1score): ", np.argmax(fs1_l), np.argmax(fs05_l), np.argmax(fs03_l)
    ths_idx = 20 #5

    from hrl_execution_monitor import preprocess as pp
    
    #-----------------------------------------------------------------------------------------
    # Data extraction
    #-----------------------------------------------------------------------------------------        
    for idx, (normalTrainIdx, abnormalTrainIdx, normalTestIdx, abnormalTestIdx) \
      in enumerate(main_data['kFoldList']):
        print "==================== ", idx, " ========================"
        print np.shape(dt_dict['train_err_ll'][idx]), np.shape(dt_dict['train_labels']), " / ",
        print np.shape(dt_dict['test_err_ll'][idx]), np.shape(dt_dict['test_labels'])        
        ## iutil.save_data_labels(dt_dict['train_err_ll'][idx], dt_dict['train_labels'])        

        trainData = (dt_dict['train_idx_ll'][idx],
                     dt_dict['train_err_ll'][idx],
                     [main_data['failure_d_image_list'][i] for i in abnormalTrainIdx]
                     + sub_data['failure_d_image_list'],
                     dt_dict['train_labels'])
        testData = (dt_dict['test_idx_ll'][idx],
                    dt_dict['test_err_ll'][idx],
                    [main_data['failure_d_image_list'][i] for i in abnormalTestIdx],
                    dt_dict['test_labels'])

        # feature extraction by index based on IROS17_isolation/isolation_util.py's feature_extraction
        ''' Get individual reconstruction probability vector when anomalies are detected '''
        train_d_idx_l = trainData[0]
        train_err_l   = trainData[1] # sample x length x dim
        train_dimg_l  = trainData[2]
        train_labels  = trainData[3]

        test_d_idx_l  = testData[0]
        test_err_l    = testData[1]
        test_dimg_l   = testData[2]
        test_labels   = testData[3]

        
        def features(d_idx_l, x_sig, x_img, y, window_step=(0,1)):
            x_sig_list = []
            x_img_list = []
            y_list = []
            for i, d_idx in enumerate(d_idx_l):
                # Skip undetected anomaly
                if d_idx is None: continue
                if y[i] > 13: continue
                if y[i] < 2: continue

                x_sig_acc = np.array(copy.copy(x_sig[i]))
                x_sig_acc = np.exp(x_sig_acc)                

                for j in range(window_step[0], window_step[1]):
                    if d_idx+j >= len(x_sig[i]) or d_idx+j-7 < 0: continue

                    if JOURNAL_TEST == 'img' or JOURNAL_TEST == 'all': 
                        if x_img[i] is None: continue
                        if len(x_img[i]) == 0: continue
                        if x_img[i][0] is None: continue
                    
                    p1 = np.array(x_sig_acc[d_idx+j])
                    p2 = np.amax(x_sig_acc[d_idx+j-3:d_idx+j+1], axis=0)
                    p3 = np.amax(x_sig_acc[d_idx+j-7:d_idx+j+1], axis=0)
                    
                    vs = np.hstack([p1, p3])
                    vs = vs.flatten() # 3x17=51

                    x_sig_list.append(vs.tolist())
                    ## print np.shape(x_img[i]), d_idx, j
                    
                    x_img_list.append(pp.extract_image(x_img[i][d_idx+j], img_feature_type=img_feature_type,
                                                       img_scale=img_scale))
                    y_list.append(y[i])

            return x_sig_list, x_img_list, y_list

        d = {}
        d['x_sig_tr'] = []; d['x_img_tr'] = []; d['y_tr'] = []
        d['x_sig_te'] = []; d['x_img_te'] = []; d['y_te'] = []
        ths_idx = kwargs.get('ths_idx', 25)        

        for i in xrange(ths_idx-5, ths_idx+5):
            x_sig, x_img, y = features(train_d_idx_l[i], train_err_l, train_dimg_l, train_labels,\
                                       window_step=(0,2))
            d['x_sig_tr'] += x_sig
            d['x_img_tr'] += x_img
            d['y_tr']     += y

        d['x_sig_te'], d['x_img_te'], d['y_te']\
        = features(test_d_idx_l[ths_idx], test_err_l, test_dimg_l, test_labels, window_step=(0,1))

        file_name = os.path.join(save_data_path, 'isol_features_'+str(idx)+'.pkl')
        ut.save_pickle(d, file_name)

    return #(d['x_sig_tr'], d['x_sig_tr'], d['y_tr']), (d['x_sig_te'], d['x_sig_te'], d['y_te'])



def sig_net_test(save_data_path, pkl_name='', renew=False):

    weights_path = os.path.join(save_data_path, 'weights')
    if os.path.isdir(weights_path) is False:
        os.system('mkdir -p '+weights_path)        

    scores = []
    for idx in xrange(8):

        np.random.seed(3334+idx)

        file_name = os.path.join(save_data_path, pkl_name+'_'+str(idx)+'.pkl')
        d = ut.load_pickle(file_name)
        trainData = (d['x_sig_tr'], d['y_tr'])
        testData = (d['x_sig_te'], d['y_te'])

        scaler  = preprocessing.StandardScaler()
        x_train = scaler.fit_transform(d['x_sig_tr'])
        x_test  = scaler.transform(d['x_sig_te'])
        ## #scaler  = preprocessing.MinMaxScaler()
        trainData = (x_train, d['y_tr'])
        testData = (x_test, d['y_te'])

        ## n,m,k = np.shape(d['x_sig_tr'])    
        ## x_train = np.array(d['x_sig_tr']).reshape(n,m*k)
        ## #x_train = scaler.fit_transform(np.array(d['x_sig_tr']).reshape(n*m,k)).reshape(n,m*k)
        ## n,m,k = np.shape(d['x_sig_te'])
        ## x_test  = np.array(d['x_sig_te']).reshape(n,m*k)
        ## #x_test  = scaler.transform(np.array(d['x_sig_te']).reshape(n*m,k)).reshape(n,m*k)

        ## # train svm
        ## from sklearn.svm import SVC
        ## clf = SVC(C=1.0, kernel='rbf', gamma=1e-3) #, decision_function_shape='ovo')
        ## #from sklearn.ensemble import RandomForestClassifier
        ## #clf = RandomForestClassifier(n_estimators=800, n_jobs=-1)
        ## ## from sklearn.neighbors import KNeighborsClassifier
        ## ## clf = KNeighborsClassifier(n_neighbors=10, n_jobs=-1)
        ## clf.fit(trainData[0], trainData[1])

        ## # classify and get scores
        ## score = clf.score(testData[0], testData[1])
        ## print "score: ", score
        ## sys.exit()



        method      = 'sig_net'
        noise_mag   = 0.01
        patience    = 5 #4 #10
        fine_tuning = False
        batch_size  = 4096

        # Save data for visualizer
        #from hrl_anomaly_detection.IROS17_isolation import isolation_util as iu
        #iu.save_data_labels(trainData[0], trainData[1])
        #sys.exit()

        from hrl_anomaly_detection.journal_isolation.models import sig_net             
        weights_file = os.path.join(weights_path,'model_weights_'+method+'_'+str(idx)+'.h5')
        ml, score = sig_net.sig_net(trainData, testData, weights_file=weights_file, patience=patience,
                                    batch_size=batch_size,\
                                    noise_mag=noise_mag, renew=True, fine_tuning=fine_tuning)
                                    
        # test
        scores.append(score)

    print scores
    print np.mean(scores), np.std(scores)

    return 


def img_net_test(save_data_path, pkl_name='', renew=False):

    weights_path = os.path.join(save_data_path, 'weights')
    if os.path.isdir(weights_path) is False:
        os.system('mkdir -p '+weights_path)        

    scores = []
    for idx in xrange(8):
        print "----------------------"
        print "Subject ", idx
        print "----------------------"
        if idx<6 or idx>6: continue

        np.random.seed(3334+idx)

        file_name = os.path.join(save_data_path, pkl_name+'_'+str(idx)+'.pkl')
        d = ut.load_pickle(file_name)
        trainData = (d['x_img_tr'], d['y_tr'])
        testData  = (d['x_img_te'], d['y_te'])

        # Extra images?

        method      = 'img_net'
        patience    = 4 #4 #10
        sam_epoch   = 40 #2:40
        fine_tuning = False
        batch_size  = 2048

        from hrl_anomaly_detection.journal_isolation.models import img_net             
        weights_file = os.path.join(weights_path,'model_weights_'+method+'_'+str(idx)+'.h5')
        ml, score = img_net.img_net(idx, trainData, testData, save_data_path,
                                    weights_file=weights_file, patience=patience,
                                    batch_size=batch_size,\
                                    sam_epoch=sam_epoch, renew=False,
                                    fine_tuning=fine_tuning)

        # test
        scores.append(score)
        #break

    print scores
    print np.mean(scores), np.std(scores)

    return 


def multi_net_test(save_data_path, pkl_name='', renew=False):

    weights_path = os.path.join(save_data_path, 'weights')
    if os.path.isdir(weights_path) is False:
        os.system('mkdir -p '+weights_path)        

    scores = []
    for idx in xrange(8):
        print "----------------------"
        print "Subject ", idx
        print "----------------------"

        # temp
        if idx >0: continue

        np.random.seed(3334+idx)

        file_name = os.path.join(save_data_path, pkl_name+'_'+str(idx)+'.pkl')
        d = ut.load_pickle(file_name)

        # scaling
        print "Scaling"
        scaler  = preprocessing.StandardScaler()
        x_sig_tr = scaler.fit_transform(d['x_sig_tr'])
        x_sig_te = scaler.transform(d['x_sig_te'])
        trainData = (x_sig_tr, d['x_img_tr'], d['y_tr'])
        testData  = (x_sig_te, d['x_img_te'], d['y_te'])
        

        method      = 'multi_net'
        noise_mag   = 0.05 #5
        patience    = 6 #4 #10
        #sam_epoch   = 40 #2:40
        fine_tuning = False
        batch_size  = 1024
        sam_epoch   = len(x_sig_tr)/batch_size

        print "Start to run multi_net"
        from hrl_anomaly_detection.journal_isolation.models import multi_net             
        sig_weights_file   = os.path.join(weights_path, 'model_weights_sig_net_'+str(idx)+'.h5')
        img_weights_file   = os.path.join(weights_path, 'model_weights_img_net_'+str(idx)+'.h5')
        multi_weights_file = os.path.join(weights_path, 'model_weights_multi_net_'+str(idx)+'.h5')
        ml, score = multi_net.multi_net(idx, trainData, testData,
                                        weights_file=(sig_weights_file, img_weights_file,
                                                      multi_weights_file),
                                        patience=patience, batch_size=batch_size,\
                                        noise_mag=noise_mag, sam_epoch=sam_epoch, renew=False,
                                        fine_tuning=fine_tuning)

        # test
        scores.append(score)
        #break

    print scores
    print np.mean(scores), np.std(scores)

    return 




## anomaly_idx_list, abnormalData, abnormalData_s, \
##                        abnormalLabel, abnormalData_img,\
##                        task_name, processed_data_path, param_dict,\
##                        window_step=10, verbose=False, plot=False,\
##                        window=False, delta_flag=False):
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
    p.add_option('--get_isol_data', '--gi', action='store_true', dest='getIsolData',
                 default=False, help='Preprocess')
    p.add_option('--preprocess', '--pp', action='store_true', dest='preprocessing',
                 default=False, help='Preprocess')
    
    p.add_option('--fint_tuning', '--ftn', action='store_true', dest='bFineTune',
                 default=False, help='Run fine tuning.')
    p.add_option('--dyn_ths', '--dt', action='store_true', dest='bDynThs',
                 default=True, help='Run dynamic threshold.')
    p.add_option('--training_only', '--to', action='store_true', dest='bTrainOnly',
                 default=False, help='Run dynamic threshold.')
    p.add_option('--testing_only', '--te', action='store_true', dest='bTestOnly',
                 default=False, help='Run dynamic threshold.')         
    p.add_option('--isol_only', '--ai', action='store', dest='isol_flag',
                 default='None', help='Run a classifier with anomaly isolation data.')         
    p.add_option('--latent_space_plot', '--lsp', action='store_true', dest='bLatentPlot',
                 default=False, help='Show latent space.')
    opt, args = p.parse_args()

    from hrl_anomaly_detection.journal_isolation.isolation_param import *
    # IROS2017
    subject_names = ['s2', 's3','s4','s5', 's6','s7','s8', 's9']
    raw_data_path, save_data_path, param_dict = getParams(opt.task, opt.bDataRenew, \
                                                          opt.bHMMRenew, opt.bCLFRenew)
    if os.uname()[1] == 'monty1':
        save_data_path = os.path.expanduser('~')+\
          '/hrl_file_server/dpark_data/anomaly/JOURNAL_ISOL/'+opt.task+'_4' #2 with dropout?
    elif os.uname()[1] == 'colossus12':
        save_data_path = os.path.expanduser('~')+\
          '/hrl_file_server/dpark_data/anomaly/JOURNAL_ISOL/'+opt.task+'_3'
    elif os.uname()[1] == 'colossus8':
        save_data_path = os.path.expanduser('~')+\
          '/hrl_file_server/dpark_data/anomaly/JOURNAL_ISOL/'+opt.task+'_4'
    else:
        save_data_path = os.path.expanduser('~')+\
          '/hrl_file_server/dpark_data/anomaly/JOURNAL_ISOL/'+opt.task+'_4'


    task_name = 'feeding'
    nb_classes = 12
    method       = 'lstm_dvae_phase_circle'
    IROS_TEST = False
    if opt.isol_flag == 'None':
        JOURNAL_TEST = None
        isol_only    = False
    else:
        JOURNAL_TEST = opt.isol_flag
        isol_only    = True
        


    if opt.getIsolData:
        get_isolation_data(method, subject_names, task_name, raw_data_path, save_data_path, param_dict,
                           fine_tuning=opt.bFineTune, dyn_ths=opt.bDynThs,
                           tr_only=opt.bTrainOnly, te_only=opt.bTestOnly, isol_only=isol_only,\
                           latent_plot=opt.bLatentPlot)
    elif opt.preprocessing:
        preprocess_data(subject_names, task_name, raw_data_path, save_data_path, param_dict,
                        fine_tuning=opt.bFineTune)
    else:                           
        ## sig_net_test(save_data_path, pkl_name='isol_features', renew=False)
        #img_net_test(save_data_path, pkl_name='isol_features', renew=False)
        multi_net_test(save_data_path, pkl_name='isol_features', renew=False)
