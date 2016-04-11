import os, sys
import numpy as np


def getPushingMicrowave(task, data_renew, AE_renew, HMM_renew, rf_center,local_range):

    handFeatures = ['unimodal_ftForce',\
                    'crossmodal_targetEEDist',\
                    'crossmodal_targetEEAng',\
                    'unimodal_audioWristRMS'] #'unimodal_audioPower', ,
    rawFeatures = ['relativePose_artag_EE', \
                   'relativePose_artag_artag', \
                   'wristAudio', \
                   'ft' ]                                
    modality_list   = ['kinematics', 'audio', 'ft', 'vision_artag'] # raw plot

    raw_data_path  = '/home/dpark/hrl_file_server/dpark_data/anomaly/RSS2016/'
    ## save_data_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/RSS2016/'+task+'_data'

    ## save_data_path = os.path.expanduser('~')+\
    ##   '/hrl_file_server/dpark_data/anomaly/RSS2016/'+task+'_data/AE'        
    ## downSampleSize = 100
    ## layers = [64,4]

    filterDim=3
    if filterDim==3: 
        # filtered dim 3
        save_data_path = os.path.expanduser('~')+\
          '/hrl_file_server/dpark_data/anomaly/RSS2016/'+task+'_data/AE150_3'
        downSampleSize = 150
        layers = [64,8]
        ## add_option = ['audioWristRMS']
        add_option = ['ftForce_mag']
        add_noise_option = ['ftForce_mag']
        
    else:
        # filtered dim 4
        save_data_path = os.path.expanduser('~')+\
          '/hrl_file_server/dpark_data/anomaly/RSS2016/'+task+'_data/AE150'
        downSampleSize = 150
        layers = [64,8]
        add_option = None


    data_param_dict= {'renew': data_renew, 'rf_center': rf_center, 'local_range': local_range,\
                      'downSampleSize': downSampleSize, 'cut_data': [0,downSampleSize], \
                      'nNormalFold':3, 'nAbnormalFold':3,\
                      'handFeatures': handFeatures, 'lowVarDataRemv': False }
    AE_param_dict  = {'renew': AE_renew, 'switch': False, 'time_window': 4,  \
                      'layer_sizes':layers, 'learning_rate':1e-4, \
                      'learning_rate_decay':1e-6, \
                      'momentum':1e-6, 'dampening':1e-6, 'lambda_reg':1e-6, \
                      'max_iteration':100000, 'min_loss':0.1, 'cuda':True, \
                      'filter':True, 'filterDim':filterDim, \
                      'nAugment': 1, \
                      'add_option': add_option , 'rawFeatures': rawFeatures,
                      'add_noise_option': add_noise_option}
                      ## 'add_option': ['targetEEDist'], 'rawFeatures': rawFeatures}
                      ## 'add_option': ['ftForce_mag'], 'rawFeatures': rawFeatures}

    if AE_param_dict['switch'] and AE_param_dict['add_option'] is ['audioWristRMS', 'ftForce_mag']:            
        SVM_param_dict = {'renew': False, 'w_negative': 6.0, 'gamma': 0.173, 'cost': 4.0}
        HMM_param_dict = {'renew': HMM_renew, 'nState': 20, 'cov': 3.0, 'scale': 1.5}
    elif AE_param_dict['switch'] and AE_param_dict['add_option'] is ['audioWristRMS']:            
        SVM_param_dict = {'renew': False, 'w_negative': 6.0, 'gamma': 0.334, 'cost': 2.0}
        HMM_param_dict = {'renew': HMM_renew, 'nState': 20, 'cov': 4.0, 'scale': 1.5}
        ## HMM_param_dict = {'renew': HMM_renew, 'nState': 20, 'cov': 3.0, 'scale': 1.5}
    elif AE_param_dict['switch'] and AE_param_dict['add_option'] is ['ftForce_mag']:            
        SVM_param_dict = {'renew': False, 'w_negative': 6.0, 'gamma': 0.173, 'cost': 4.0}
        HMM_param_dict = {'renew': HMM_renew, 'nState': 20, 'cov': 3.0, 'scale': 0.5}
    elif AE_param_dict['switch'] and AE_param_dict['add_option'] is ['targetEEDist']:            
        SVM_param_dict = {'renew': False, 'w_negative': 6.0, 'gamma': 0.173, 'cost': 4.0}
        HMM_param_dict = {'renew': HMM_renew, 'nState': 20, 'cov': 1.5, 'scale': 1.0}
    elif AE_param_dict['switch']:            
        SVM_param_dict = {'renew': False, 'w_negative': 3.0, 'gamma': 0.334, 'cost': 1.0}
        HMM_param_dict = {'renew': HMM_renew, 'nState': 20, 'cov': 2.0, 'scale': 2.0}
    else:
        SVM_param_dict = {'renew': False, 'w_negative': 6.0, 'gamma': 0.173, 'cost': 4.0}
        HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 4.0, 'scale': 5.0}

    nPoints        = 20  # 'progress_time_cluster',,'fixed' , 'svm' , 
    ROC_param_dict = {'methods': [ 'svm', 'progress_time_cluster', 'fixed' ],\
                      'update_list': [],\
                      'nPoints': nPoints,\
                      'progress_param_range':np.linspace(-1., -10., nPoints), \
                      'svm_param_range': np.logspace(-4, 1.2, nPoints),\
                      'fixed_param_range': np.linspace(1.0, -3.0, nPoints),\
                      'cssvm_param_range': np.logspace(0.0, 2.0, nPoints) }        
    param_dict = {'data_param': data_param_dict, 'AE': AE_param_dict, 'HMM': HMM_param_dict, \
                  'SVM': SVM_param_dict, 'ROC': ROC_param_dict}



    return raw_data_path, save_data_path, param_dict

