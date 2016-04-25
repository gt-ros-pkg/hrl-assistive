import os, sys
import numpy as np


def getScooping(task, data_renew, AE_renew, HMM_renew, rf_center,local_range, ae_swtch=False, dim=4):
    
    handFeatures = ['unimodal_audioWristRMS',\
                    'unimodal_ftForce',\
                    'crossmodal_targetEEDist', \
                    'crossmodal_targetEEAng']
    rawFeatures = ['relativePose_target_EE', \
                   'wristAudio', \
                   'ft' ]                                
    modality_list = ['kinematics', 'audioWrist', 'ft', 'vision_artag', \
                     'vision_change', 'pps']
    downSampleSize = 200

    save_data_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/RSS2016/'+task+'_data/TEST'
    raw_data_path  = '/home/dpark/hrl_file_server/dpark_data/anomaly/RSS2016/'

    data_param_dict= {'renew': data_renew, 'rf_center': rf_center, 'local_range': local_range,\
                      'downSampleSize': downSampleSize, 'cut_data': [0,200], 'nNormalFold':4, 'nAbnormalFold':4,\
                      'handFeatures': handFeatures, 'lowVarDataRemv': False,\
                      'handFeatures_noise': False}
    AE_param_dict  = {'renew': AE_renew, 'switch': False, 'time_window': 4, \
                      'layer_sizes':[64,dim], 'learning_rate':1e-6, 'learning_rate_decay':1e-6, \
                      'momentum':1e-6, 'dampening':1e-6, 'lambda_reg':1e-6, \
                      'max_iteration':30000, 'min_loss':0.1, 'cuda':True, \
                      'filter':True, 'filterDim':4, \
                      'nAugment': 1, \
                      'add_option': None, 'rawFeatures': rawFeatures,\
                      'add_noise_option': [], 'preTrainModel': None}
    HMM_param_dict = {'renew': HMM_renew, 'nState': 20, 'cov': 5.0, 'scale': 4.0}
    SVM_param_dict = {'renew': False, 'w_negative': 3.0, 'gamma': 0.3, 'cost': 6.0}

    nPoints        = 20
    ROC_param_dict = {'methods': ['progress_time_cluster', 'svm','fixed'],\
                      'update_list': [],\
                      'nPoints': nPoints,\
                      'progress_param_range':-np.linspace(0., 10.0, nPoints), \
                      'svm_param_range': np.logspace(-4, 1.2, nPoints),\
                      'fixed_param_range': -np.logspace(0.0, 0.9, nPoints)+1.2,\
                      'cssvm_param_range': np.logspace(0.0, 2.0, nPoints) }
    param_dict = {'data_param': data_param_dict, 'AE': AE_param_dict, 'HMM': HMM_param_dict, \
                  'SVM': SVM_param_dict, 'ROC': ROC_param_dict}

    return raw_data_path, save_data_path, param_dict


def getFeeding(task, data_renew, AE_renew, HMM_renew, rf_center,local_range, ae_swtch=False, dim=4):
    handFeatures = ['unimodal_audioWristRMS', 'unimodal_ftForce', \
                    'crossmodal_artagEEDist', 'crossmodal_artagEEAng'] 
    rawFeatures = ['relativePose_target_EE', \
                   'wristAudio', \
                   'ft' ]
                   #'relativePose_artag_EE', \

    modality_list   = ['ft' ,'kinematics', 'audioWrist', 'vision_artag']
    downSampleSize = 200

    save_data_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/RSS2016/'+task+'_data/TEST'
    raw_data_path  = '/home/dpark/hrl_file_server/dpark_data/anomaly/RSS2016/'

    data_param_dict= {'renew': data_renew, 'rf_center': rf_center, 'local_range': local_range,\
                      'downSampleSize': downSampleSize, 'cut_data': [0,200], \
                      'nNormalFold':4, 'nAbnormalFold':4,\
                      'handFeatures': handFeatures, 'lowVarDataRemv': False,\
                      'handFeatures_noise': True}
    AE_param_dict  = {'renew': AE_renew, 'switch': False, 'time_window': 4, \
                      'layer_sizes':[64,dim], 'learning_rate':1e-6, 'learning_rate_decay':1e-6, \
                      'momentum':1e-6, 'dampening':1e-6, 'lambda_reg':1e-6, \
                      'max_iteration':30000, 'min_loss':0.1, 'cuda':True, \
                      'filter':True, 'filterDim':4,\
                      'add_option': None, 'rawFeatures': rawFeatures,\
                      'add_noise_option': [], 'preTrainModel': None}                      
    HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 5.0, 'scale': 4.0}
    SVM_param_dict = {'renew': False, 'w_negative': 1.3, 'gamma': 0.0103, 'cost': 1.0}

    nPoints        = 20
    ROC_param_dict = {'methods': ['progress_time_cluster', 'svm','fixed'],\
                      'update_list': [],\
                      'nPoints': nPoints,\
                      'progress_param_range': -np.logspace(0., 1.5, nPoints),\
                      'svm_param_range': np.logspace(-1.8, 0.25, nPoints),\
                      'fixed_param_range': np.linspace(1.0, -3.0, nPoints),\
                      'cssvm_param_range': np.logspace(0.0, 2.0, nPoints) }
    param_dict = {'data_param': data_param_dict, 'AE': AE_param_dict, 'HMM': HMM_param_dict, \
                  'SVM': SVM_param_dict, 'ROC': ROC_param_dict}

    return raw_data_path, save_data_path, param_dict

def getPushingMicroWhite(task, data_renew, AE_renew, HMM_renew, rf_center,local_range, pre_train=False, \
                         ae_swtch=False, dim=3):

    if dim == 5:
        handFeatures = ['unimodal_ftForce',\
                        'crossmodal_artagEEDist',\
                        'crossmodal_artagEEAng',\
                        'crossmodal_subArtagEEDist',\
                        'unimodal_audioWristRMS'] #'unimodal_audioPower', ,
        SVM_param_dict = {'renew': False, 'w_negative': 0.325, 'gamma': 2., 'cost': 5.0}
        HMM_param_dict = {'renew': HMM_renew, 'nState': 35, 'cov': 0.73, 'scale': 3.25}

        nPoints        = 20  # 'progress_time_cluster',,'fixed' , 'svm' , 
        ROC_param_dict = {'methods': [ 'fixed', 'progress_time_cluster', 'svm' ],\
                          'update_list': [],\
                          'nPoints': nPoints,\
                          'progress_param_range':np.linspace(0.0, -8., nPoints), \
                          'svm_param_range': np.logspace(-2.5, 0, nPoints),\
                          'fixed_param_range': np.linspace(1.0, -3.0, nPoints),\
                          'cssvm_param_range': np.logspace(-4.0, 2.0, nPoints),\
                          'svm_param_range': np.logspace(-4, 1.2, nPoints),\        
                          'sgd_param_range': np.logspace(-4, 1.2, nPoints)}        
        
    elif dim == 4:
        handFeatures = ['unimodal_ftForce',\
                        'crossmodal_artagEEDist',\
                        'crossmodal_subArtagEEDist',\
                        'unimodal_audioWristRMS'] #'unimodal_audioPower', ,
        if ae_swtch:
            SVM_param_dict = {'renew': False, 'w_negative': 0.2, 'gamma': 2.5, 'cost': 4.0}
            HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 0.73, 'scale': 5.5}
        else:
            SVM_param_dict = {'renew': False, 'w_negative': 0.45, 'gamma': 1.5, 'cost': 4.0}
            HMM_param_dict = {'renew': HMM_renew, 'nState': 40, 'cov': 0.73, 'scale': 5.5}

        nPoints        = 20  # 'progress_time_cluster',,'fixed' , 'svm' , 
        ROC_param_dict = {'methods': [ 'fixed', 'progress_time_cluster', 'svm' ],\
                          'update_list': [],\
                          'nPoints': nPoints,\
                          'progress_param_range':np.linspace(-1, -10., nPoints), \
                          'svm_param_range': np.logspace(-2.5, 0, nPoints),\
                          'fixed_param_range': np.logspace(-2, 0.0, nPoints)*-5.0,\
                          'cssvm_param_range': np.logspace(-4.0, 2.0, nPoints),
                          'svm_param_range': np.logspace(-4, 1.2, nPoints),\        
                          'sgd_param_range': np.logspace(-4, 1.2, nPoints)}        
        
    elif dim == 3:
        handFeatures = ['unimodal_ftForce',\
                        'crossmodal_artagEEDist',\
                        'unimodal_audioWristRMS'] #'unimodal_audioPower', ,
        SVM_param_dict = {'renew': False, 'w_negative': 0.7, 'gamma': 1.5, 'cost': 5.0}
        HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 1.36, 'scale': 5.5}
        
        nPoints        = 20  # 'progress_time_cluster',,'fixed' , 'svm' , 
        ROC_param_dict = {'methods': [ 'fixed', 'progress_time_cluster', 'svm' ],\
                          'update_list': [],\
                          'nPoints': nPoints,\
                          'progress_param_range':np.linspace(0.0, -8.0, nPoints), \
                          'svm_param_range': np.logspace(-2, 0, nPoints),\
                          'fixed_param_range': np.linspace(1.0, -3.0, nPoints),\
                          'cssvm_param_range': np.logspace(-4.0, 2.0, nPoints),
                          'svm_param_range': np.logspace(-4, 1.2, nPoints)}        
    elif dim == 2:
        handFeatures = ['unimodal_ftForce',\
                        'unimodal_audioWristRMS'] #'unimodal_audioPower', ,
        SVM_param_dict = {'renew': False, 'w_negative': 0.325, 'gamma': 2.0, 'cost': 5.0}
        HMM_param_dict = {'renew': HMM_renew, 'nState': 30, 'cov': 1.36, 'scale': 7.75}

        nPoints        = 20  # 'progress_time_cluster',,'fixed' , 'svm' , 
        ROC_param_dict = {'methods': [ 'fixed', 'progress_time_cluster', 'svm' ],\
                          'update_list': [],\
                          'nPoints': nPoints,\
                          'progress_param_range':np.linspace(0.0, -8.0, nPoints), \
                          'svm_param_range': np.logspace(-2.5, 0, nPoints),\
                          'fixed_param_range': np.linspace(1.0, -3.0, nPoints),\
                          'cssvm_param_range': np.logspace(-4.0, 2.0, nPoints),
                          'svm_param_range': np.logspace(-4, 1.2, nPoints)}        
        
    rawFeatures = ['relativePose_artag_EE', \
                   'wristAudio', \
                   'ft' ]                                
    modality_list   = ['kinematics', 'audio', 'ft', 'vision_artag'] # raw plot
    raw_data_path  = '/home/dpark/hrl_file_server/dpark_data/anomaly/RSS2016/'

    AE_param_dict  = {'renew': AE_renew, 'switch': ae_swtch, 'method': 'ae', 'time_window': 4,  \
                      'layer_sizes':[], 'learning_rate':1e-4, \
                      'learning_rate_decay':1e-6, \
                      'momentum':1e-6, 'dampening':1e-6, 'lambda_reg':1e-6, \
                      'max_iteration':50000, 'min_loss':0.01, 'cuda':True, \
                      'pca_gamma': 1.0,\
                      'filter':False, 'filterDim':4, \
                      'nAugment': 1, \
                      'add_option': None, 'rawFeatures': rawFeatures,\
                      'add_noise_option': [], 'preTrainModel': None}

    data_param_dict= {'renew': data_renew, 'rf_center': rf_center, 'local_range': local_range,\
                      'downSampleSize': 200, 'cut_data': None, \
                      'nNormalFold':3, 'nAbnormalFold':3,\
                      'handFeatures': handFeatures, 'lowVarDataRemv': False,\
                      'handFeatures_noise': True}

    if AE_param_dict['method']=='pca':      
        # filtered dim 4
        save_data_path = os.path.expanduser('~')+\
          '/hrl_file_server/dpark_data/anomaly/RSS2016/'+task+'_data/AE150'
        data_param_dict['downSampleSize'] = 150
        AE_param_dict['layer_sizes']      = [64,dim]
        AE_param_dict['nAugment']         = 0

    elif AE_param_dict['method']=='ae' and pre_train:
        save_data_path = os.path.expanduser('~')+\
          '/hrl_file_server/dpark_data/anomaly/RSS2016/'+task+'_data/'
        data_param_dict['downSampleSize'] = 200
        AE_param_dict['layer_sizes'] = [64,dim]
        AE_param_dict['add_option']  = None
        AE_param_dict['learning_rate'] = 1e-6
        
    else:
        save_data_path = os.path.expanduser('~')+\
          '/hrl_file_server/dpark_data/anomaly/RSS2016/'+task+'_data/AE150_'+str(dim)
        data_param_dict['downSampleSize'] = 200
        AE_param_dict['layer_sizes'] = [64,dim]
        AE_param_dict['add_option'] = None
        AE_param_dict['add_noise_option'] = []
        AE_param_dict['learning_rate'] = 1e-6
        AE_param_dict['preTrainModel'] = os.path.join(save_data_path, 'ae_pretrain_model_'+str(dim)+'.pkl')
            
        
            ## # filtered dim 1
            ## save_data_path = os.path.expanduser('~')+\
            ##   '/hrl_file_server/dpark_data/anomaly/RSS2016/'+task+'_data/AE150_1'
            ## data_param_dict['downSampleSize'] = 150
            ## AE_param_dict['layer_sizes'] = [64,8]
            ## ## add_option = ['audioWristRMS']
            ## AE_param_dict['add_option'] = ['ftForce_mag','audioWristRMS','targetEEDist', 'targetEEAng']
            ## AE_param_dict['add_noise_option'] = ['ftForce_mag']
            

    ## if AE_param_dict['switch'] and AE_param_dict['add_option'] is ['audioWristRMS', 'ftForce_mag','targetEEDist','targetEEAng']:            
    ##     SVM_param_dict = {'renew': False, 'w_negative': 6.0, 'gamma': 0.173, 'cost': 4.0}
    ##     HMM_param_dict = {'renew': HMM_renew, 'nState': 20, 'cov': 5.0, 'scale': 1.5}
    ## elif AE_param_dict['switch'] and AE_param_dict['add_option'] is ['audioWristRMS', 'ftForce_mag','targetEEDist']:            
    ##     SVM_param_dict = {'renew': False, 'w_negative': 6.0, 'gamma': 0.173, 'cost': 4.0}
    ##     HMM_param_dict = {'renew': HMM_renew, 'nState': 20, 'cov': 3.0, 'scale': 1.5}
    ## elif AE_param_dict['switch'] and AE_param_dict['add_option'] is ['audioWristRMS', 'ftForce_mag']:            
    ##     SVM_param_dict = {'renew': False, 'w_negative': 6.0, 'gamma': 0.173, 'cost': 4.0}
    ##     HMM_param_dict = {'renew': HMM_renew, 'nState': 20, 'cov': 3.0, 'scale': 1.5}
    ## elif AE_param_dict['switch'] and AE_param_dict['add_option'] is ['audioWristRMS']:            
    ##     SVM_param_dict = {'renew': False, 'w_negative': 6.0, 'gamma': 0.334, 'cost': 2.0}
    ##     HMM_param_dict = {'renew': HMM_renew, 'nState': 20, 'cov': 4.0, 'scale': 1.5}
    ##     ## HMM_param_dict = {'renew': HMM_renew, 'nState': 20, 'cov': 3.0, 'scale': 1.5}
    ## elif AE_param_dict['switch'] and AE_param_dict['add_option'] is ['ftForce_mag']:            
    ##     SVM_param_dict = {'renew': False, 'w_negative': 6.0, 'gamma': 0.173, 'cost': 4.0}
    ##     HMM_param_dict = {'renew': HMM_renew, 'nState': 20, 'cov': 5.0, 'scale': 1.5}
    ## elif AE_param_dict['switch'] and AE_param_dict['add_option'] is ['targetEEDist']:            
    ##     SVM_param_dict = {'renew': False, 'w_negative': 6.0, 'gamma': 0.173, 'cost': 4.0}
    ##     HMM_param_dict = {'renew': HMM_renew, 'nState': 20, 'cov': 1.5, 'scale': 1.0}
    ## elif AE_param_dict['switch'] and AE_param_dict['method']=='pca':            
    ##     SVM_param_dict = {'renew': False, 'w_negative': 3.0, 'gamma': 0.334, 'cost': 1.0}
    ##     HMM_param_dict = {'renew': HMM_renew, 'nState': 20, 'cov': 5.0, 'scale': 0.5}
    ## elif AE_param_dict['switch'] and AE_param_dict['method']=='ae':            
    ##     SVM_param_dict = {'renew': False, 'w_negative': 3.0, 'gamma': 0.334, 'cost': 1.0}
    ##     HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 4.0, 'scale': 2.0}



    param_dict = {'data_param': data_param_dict, 'AE': AE_param_dict, 'HMM': HMM_param_dict, \
                  'SVM': SVM_param_dict, 'ROC': ROC_param_dict}



    return raw_data_path, save_data_path, param_dict


def getPushingMicroBlack(task, data_renew, AE_renew, HMM_renew, rf_center,local_range, pre_train=False,\
                         ae_swtch=False, dim=3):

    if dim == 5:
        handFeatures = ['unimodal_ftForce',\
                        'crossmodal_artagEEDist',\
                        'crossmodal_artagEEAng',\
                        'crossmodal_subArtagEEDist',\
                        'unimodal_audioWristRMS'] #'unimodal_audioPower', ,
        SVM_param_dict = {'renew': False, 'w_negative': 0.325, 'gamma': 2., 'cost': 5.0}
        HMM_param_dict = {'renew': HMM_renew, 'nState': 35, 'cov': 0.73, 'scale': 3.25}

        nPoints        = 20  # 'progress_time_cluster',,'fixed' , 'svm' , 
        ROC_param_dict = {'methods': [ 'fixed', 'progress_time_cluster', 'svm' ],\
                          'update_list': [],\
                          'nPoints': nPoints,\
                          'progress_param_range':np.linspace(0.0, -8., nPoints), \
                          'svm_param_range': np.logspace(-2.5, 0, nPoints),\
                          'fixed_param_range': np.linspace(1.0, -3.0, nPoints),\
                          'cssvm_param_range': np.logspace(-4.0, 2.0, nPoints),
                          'svm_param_range': np.logspace(-4, 1.2, nPoints)}        
        
    elif dim == 4:
        handFeatures = ['unimodal_ftForce',\
                        'crossmodal_artagEEDist',\
                        'crossmodal_subArtagEEDist',\
                        'unimodal_audioWristRMS'] #'unimodal_audioPower', ,
        if ae_swtch:
            SVM_param_dict = {'renew': False, 'w_negative': 0.2, 'gamma': 2.5, 'cost': 4.0}
            HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 0.73, 'scale': 5.5}
        else:
            SVM_param_dict = {'renew': False, 'w_negative': 0.45, 'gamma': 1.5, 'cost': 4.0}
            HMM_param_dict = {'renew': HMM_renew, 'nState': 40, 'cov': 0.73, 'scale': 5.5}

        nPoints        = 20  # 'progress_time_cluster',,'fixed' , 'svm' , 
        ROC_param_dict = {'methods': [ 'fixed', 'progress_time_cluster', 'svm' ],\
                          'update_list': [],\
                          'nPoints': nPoints,\
                          'progress_param_range':np.linspace(-1, -10., nPoints), \
                          'svm_param_range': np.logspace(-2.5, 0, nPoints),\
                          'fixed_param_range': np.logspace(-2, 0.0, nPoints)*-5.0,\
                          'cssvm_param_range': np.logspace(-4.0, 2.0, nPoints),
                          'svm_param_range': np.logspace(-4, 1.2, nPoints)}        
        
    elif dim == 3:
        handFeatures = ['unimodal_ftForce',\
                        'crossmodal_artagEEDist',\
                        'unimodal_audioWristRMS'] #'unimodal_audioPower', ,
        SVM_param_dict = {'renew': False, 'w_negative': 0.7, 'gamma': 1.5, 'cost': 5.0}
        HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 1.36, 'scale': 5.5}
        
        nPoints        = 20  # 'progress_time_cluster',,'fixed' , 'svm' , 
        ROC_param_dict = {'methods': [ 'fixed', 'progress_time_cluster', 'svm' ],\
                          'update_list': [],\
                          'nPoints': nPoints,\
                          'progress_param_range':np.linspace(0.0, -8.0, nPoints), \
                          'svm_param_range': np.logspace(-2, 0, nPoints),\
                          'fixed_param_range': np.linspace(1.0, -3.0, nPoints),\
                          'cssvm_param_range': np.logspace(-4.0, 2.0, nPoints),
                          'svm_param_range': np.logspace(-4, 1.2, nPoints)}        
    elif dim == 2:
        handFeatures = ['unimodal_ftForce',\
                        'unimodal_audioWristRMS'] #'unimodal_audioPower', ,
        SVM_param_dict = {'renew': False, 'w_negative': 0.325, 'gamma': 2.0, 'cost': 5.0}
        HMM_param_dict = {'renew': HMM_renew, 'nState': 30, 'cov': 1.36, 'scale': 7.75}

        nPoints        = 20  # 'progress_time_cluster',,'fixed' , 'svm' , 
        ROC_param_dict = {'methods': [ 'fixed', 'progress_time_cluster', 'svm' ],\
                          'update_list': [],\
                          'nPoints': nPoints,\
                          'progress_param_range':np.linspace(0.0, -8.0, nPoints), \
                          'svm_param_range': np.logspace(-2.5, 0, nPoints),\
                          'fixed_param_range': np.linspace(1.0, -3.0, nPoints),\
                          'cssvm_param_range': np.logspace(-4.0, 2.0, nPoints),
                          'svm_param_range': np.logspace(-4, 1.2, nPoints)}
            
    rawFeatures = ['relativePose_artag_EE', \
                   'wristAudio', \
                   'ft' ]                                
    modality_list   = ['kinematics', 'audio', 'ft', 'vision_artag'] # raw plot
    raw_data_path  = '/home/dpark/hrl_file_server/dpark_data/anomaly/RSS2016/'

    AE_param_dict  = {'renew': AE_renew, 'switch': ae_swtch, 'method': 'ae', 'time_window': 4,  \
                      'layer_sizes':[], 'learning_rate':1e-4, \
                      'learning_rate_decay':1e-6, \
                      'momentum':1e-6, 'dampening':1e-6, 'lambda_reg':1e-6, \
                      'max_iteration':100000, 'min_loss':0.01, 'cuda':True, \
                      'pca_gamma': 1.0,\
                      'filter':False, 'filterDim':4, \
                      'nAugment': 1, \
                      'add_option': None, 'rawFeatures': rawFeatures,\
                      'add_noise_option': [], 'preTrainModel': None}

    data_param_dict= {'renew': data_renew, 'rf_center': rf_center, 'local_range': local_range,\
                      'downSampleSize': 200, 'cut_data': None, \
                      'nNormalFold':3, 'nAbnormalFold':3,\
                      'handFeatures': handFeatures, 'lowVarDataRemv': False,\
                      'handFeatures_noise': True}

    if AE_param_dict['method']=='pca':
        save_data_path = os.path.expanduser('~')+\
          '/hrl_file_server/dpark_data/anomaly/RSS2016/'+task+'_data/AE200'
        data_param_dict['downSampleSize'] = 200
        AE_param_dict['layer_sizes']      = [64,dim]
        AE_param_dict['nAugment']         = 0
        
    elif AE_param_dict['method']=='ae' and pre_train:
        save_data_path = os.path.expanduser('~')+\
          '/hrl_file_server/dpark_data/anomaly/RSS2016/'+task+'_data/AE200'
        data_param_dict['downSampleSize'] = 200
        AE_param_dict['layer_sizes']      = [64,dim]
        AE_param_dict['add_option']       = None
        AE_param_dict['learning_rate'] = 1e-6            
    else:
        save_data_path = os.path.expanduser('~')+\
          '/hrl_file_server/dpark_data/anomaly/RSS2016/'+task+'_data/'
        data_param_dict['downSampleSize'] = 200
        AE_param_dict['layer_sizes'] = [64,dim]
        AE_param_dict['add_option']  = None
        AE_param_dict['learning_rate'] = 1e-6
        AE_param_dict['preTrainModel'] = os.path.join(save_data_path, 'ae_pretrain_model_'+str(dim)+'.pkl')
        

    param_dict = {'data_param': data_param_dict, 'AE': AE_param_dict, 'HMM': HMM_param_dict, \
                  'SVM': SVM_param_dict, 'ROC': ROC_param_dict}

    return raw_data_path, save_data_path, param_dict


def getPushingToolCase(task, data_renew, AE_renew, HMM_renew, rf_center,local_range, pre_train=False, \
                       ae_swtch=False, dim=3):

    
    nPoints        = 20  # 'progress_time_cluster',,'fixed' , 'svm' 
    if dim == 5:
        handFeatures = ['unimodal_ftForce',\
                        'crossmodal_artagEEDist',\
                        'crossmodal_artagEEAng',\
                        'crossmodal_subArtagEEDist',\
                        'unimodal_audioWristRMS'] #'unimodal_audioPower', ,
        SVM_param_dict = {'renew': False, 'w_negative': 2.0, 'gamma': 0.1, 'cost': 6.0,\
                          'cssvm_w_negative': 8.0, 'cssvm_gamma': 0.1, 'cssvm_cost': 8.0}
        HMM_param_dict = {'renew': HMM_renew, 'nState': 15, 'cov': 2.83, 'scale': 4.72}
        ROC_param_dict = {'methods': [ 'progress_time_cluster', 'svm', 'fixed' ],\
                          'update_list': [],\
                          'nPoints': nPoints,\
                          'progress_param_range':np.linspace(1., -3., nPoints), \
                          'svm_param_range': np.logspace(-2, 0.1, nPoints),\
                          'fixed_param_range': np.linspace(2.0, -1.0, nPoints),\
                          'cssvm_param_range': np.logspace(-4.0, 2.0, nPoints) }        
    elif dim == 4:
        handFeatures = ['unimodal_ftForce',\
                        'crossmodal_artagEEDist',\
                        'crossmodal_subArtagEEDist',\
                        'unimodal_audioWristRMS'] #'unimodal_audioPower', ,
        SVM_param_dict = {'renew': False, 'w_negative': 8.0, 'gamma': 0.1, 'cost': 8.0,\
                          'cssvm_w_negative': 8.0, 'cssvm_gamma': 0.1, 'cssvm_cost': 13.0}
        HMM_param_dict = {'renew': HMM_renew, 'nState': 20, 'cov': 2.83, 'scale': 4.72}
        ## HMM_param_dict = {'renew': HMM_renew, 'nState': 10, 'cov': 1.6, 'scale': 0.01}
        ROC_param_dict = {'methods': [ 'progress_time_cluster', 'svm', 'fixed', 'cssvm' ],\
                          'update_list': [],\
                          'nPoints': nPoints,\
                          'progress_param_range':np.linspace(0.5, -4., nPoints), \
                          'svm_param_range': np.logspace(-2, 0.8, nPoints),\
                          'fixed_param_range': np.linspace(-2.0, 1.0, nPoints),\
                          'cssvm_param_range': np.logspace(-4.0, 2.0, nPoints) }        
    elif dim == 3:
        handFeatures = ['unimodal_ftForce',\
                        'crossmodal_artagEEDist',\
                        'unimodal_audioWristRMS'] #'unimodal_audioPower', ,
        SVM_param_dict = {'renew': False, 'w_negative': 2.0, 'gamma': 2.0, 'cost': 2.0,\
                          'cssvm_w_negative': 8.0, 'cssvm_gamma': 0.1, 'cssvm_cost': 8.0}
        HMM_param_dict = {'renew': HMM_renew, 'nState': 15, 'cov': 1.66, 'scale': 4.72}
        ## HMM_param_dict = {'renew': HMM_renew, 'nState': 10, 'cov': 1.66, 'scale': 10}
        ROC_param_dict = {'methods': [ 'progress_time_cluster', 'svm', 'fixed' ],\
                          'update_list': [],\
                          'nPoints': nPoints,\
                          'progress_param_range':np.linspace(1., -3., nPoints), \
                          'svm_param_range': np.logspace(-2, 0.1, nPoints),\
                          'fixed_param_range': np.linspace(2.0, -1.0, nPoints),\
                          'cssvm_param_range': np.logspace(-4.0, 2.0, nPoints) }        
    elif dim == 2:
        handFeatures = ['unimodal_ftForce',\
                        'unimodal_audioWristRMS'] #'unimodal_audioPower', ,
        SVM_param_dict = {'renew': False, 'w_negative': 8.0, 'gamma': 0.01, 'cost': 8.0}
        HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 4, 'scale': 6.83}
        ROC_param_dict = {'methods': [ 'progress_time_cluster', 'svm', 'fixed' ],\
                          'update_list': [],\
                          'nPoints': nPoints,\
                          'progress_param_range':np.linspace(-4., 1., nPoints), \
                          'svm_param_range': np.logspace(-2, 0.8, nPoints),\
                          'fixed_param_range': np.linspace(2.0, -3.0, nPoints),\
                          'cssvm_param_range': np.logspace(-4.0, 2.0, nPoints) }        
        
                        
    rawFeatures = ['relativePose_artag_EE', \
                   'wristAudio', \
                   'ft' ]                                
    modality_list   = ['kinematics', 'audio', 'ft', 'vision_artag'] # raw plot
    raw_data_path  = '/home/dpark/hrl_file_server/dpark_data/anomaly/RSS2016/'

    AE_param_dict  = {'renew': AE_renew, 'switch': ae_swtch, 'method': 'ae', 'time_window': 4,  \
                      'layer_sizes':[], 'learning_rate':1e-4, \
                      'learning_rate_decay':1e-6, \
                      'momentum':1e-6, 'dampening':1e-6, 'lambda_reg':1e-6, \
                      'max_iteration':100000, 'min_loss':0.01, 'cuda':True, \
                      'pca_gamma': 1.0,\
                      'filter':False, 'filterDim':4, \
                      'nAugment': 1, \
                      'add_option': None, 'rawFeatures': rawFeatures,\
                      'add_noise_option': [], 'preTrainModel': None}

    data_param_dict= {'renew': data_renew, 'rf_center': rf_center, 'local_range': local_range,\
                      'downSampleSize': 200, 'cut_data': None, \
                      'nNormalFold':3, 'nAbnormalFold':3,\
                      'handFeatures': handFeatures, 'lowVarDataRemv': False,\
                      'handFeatures_noise': True}

    if AE_param_dict['method']=='pca':
        # filtered dim 5
        save_data_path = os.path.expanduser('~')+\
          '/hrl_file_server/dpark_data/anomaly/RSS2016/'+task+'_data/AE150'
        data_param_dict['downSampleSize'] = 150
        AE_param_dict['layer_sizes']      = [64,dim]
        AE_param_dict['nAugment']         = 0
        
    elif AE_param_dict['method']=='ae' and pre_train:
        # filtered dim 5
        save_data_path = os.path.expanduser('~')+\
          '/hrl_file_server/dpark_data/anomaly/RSS2016/'+task+'_data/'
        data_param_dict['downSampleSize'] = 200
        AE_param_dict['layer_sizes'] = [64,dim]
        AE_param_dict['add_option']  = None
        AE_param_dict['learning_rate'] = 1e-6        
    else:
        save_data_path = os.path.expanduser('~')+\
          '/hrl_file_server/dpark_data/anomaly/RSS2016/'+task+'_data/AE200_'+str(dim)
        data_param_dict['downSampleSize'] = 200
        AE_param_dict['layer_sizes']      = [64,dim]
        AE_param_dict['add_option']       = None
        AE_param_dict['preTrainModel'] = os.path.join(save_data_path, 'ae_pretrain_model.pkl')
        AE_param_dict['learning_rate'] = 1e-6            
        

    if AE_param_dict['switch'] and AE_param_dict['method']=='pca':            
        SVM_param_dict = {'renew': False, 'w_negative': 3.0, 'gamma': 0.334, 'cost': 1.0}
        HMM_param_dict = {'renew': HMM_renew, 'nState': 20, 'cov': 5.0, 'scale': 0.5}
    elif AE_param_dict['switch'] and AE_param_dict['method']=='ae':            
        SVM_param_dict = {'renew': False, 'w_negative': 3.0, 'gamma': 0.334, 'cost': 1.0}
        HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 4.0, 'scale': 1.5}


    param_dict = {'data_param': data_param_dict, 'AE': AE_param_dict, 'HMM': HMM_param_dict, \
                  'SVM': SVM_param_dict, 'ROC': ROC_param_dict}

    return raw_data_path, save_data_path, param_dict
