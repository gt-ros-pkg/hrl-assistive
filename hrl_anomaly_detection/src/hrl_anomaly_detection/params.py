import os, sys
import numpy as np


def getScooping(task, data_renew, AE_renew, HMM_renew, rf_center,local_range, pre_train=False,\
                ae_swtch=False, dim=4):

    if dim == 4:
        handFeatures = ['unimodal_ftForce',\
                        'crossmodal_targetEEDist', \
                        'crossmodal_targetEEAng', \
                        'unimodal_audioWristRMS']
        HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 3.566, 'scale': 1.0}
        SVM_param_dict = {'renew': False, 'w_negative': 4.0, 'gamma': 0.039, 'cost': 4.59,\
                          'hmmosvm_nu': 0.00316,\
                          'hmmsvm_diag_w_negative': 0.85, 'hmmsvm_diag_cost': 12.5, \
                          'hmmsvm_diag_gamma': 0.01,\
                          'osvm_nu': 0.000215, 'osvm_window_size': 10,\
                          'hmmsvm_dL_w_negative': 0.85, 'hmmsvm_dL_cost': 7.5, \
                          'hmmsvm_dL_gamma': 0.50749 }
        
        nPoints        = 20  # 'progress_time_cluster',,'fixed' , 'svm' , 
        ROC_param_dict = {'methods': [ 'change', 'fixed', 'progress_time_cluster', 'svm', 'hmmosvm', 'hmmsvm_diag', 'osvm', 'hmmsvm_dL' ],\
                          'update_list': ['svm'],\
                          'nPoints': nPoints,\
                          'progress_param_range':np.linspace(0.0, -7., nPoints), \
                          'svm_param_range': np.logspace(-1.8, 1.0, nPoints),\
                          'change_param_range': np.logspace(-0.8, 1.0, nPoints)*-1.0,\
                          'fixed_param_range': np.logspace(0.0, 0.5, nPoints)*-1.0+1.3,\
                          'cssvm_param_range': np.logspace(-4.0, 2.0, nPoints),\
                          'hmmosvm_param_range': np.logspace(-4.0, 1.0, nPoints),\
                          'hmmsvm_diag_param_range': np.logspace(-4, 1.2, nPoints),\
                          'hmmsvm_dL_param_range': np.logspace(-4, 1.2, nPoints),\
                          'osvm_param_range': np.logspace(-5., 0.0, nPoints),\
                          'sgd_param_range': np.logspace(-4, 1.2, nPoints)}        
        
    elif dim == 3:
        handFeatures = ['unimodal_ftForce',\
                        'crossmodal_targetEEDist', \
                        'crossmodal_targetEEAng']
        HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 3.566, 'scale': 3.0}
        SVM_param_dict = {'renew': False, 'w_negative': 0.825, 'gamma': 3.16, 'cost': 4.0,\
                          'hmmosvm_nu': 0.00316,\
                          'hmmsvm_diag_w_negative': 0.85, 'hmmsvm_diag_cost': 12.5, \
                          'hmmsvm_diag_gamma': 0.01}

        nPoints        = 20  # 'progress_time_cluster',,'fixed' , 'svm' , 
        ROC_param_dict = {'methods': [ 'fixed', 'progress_time_cluster', 'svm', 'hmmosvm'],\
                          'update_list': ['hmmosvm'],\
                          'nPoints': nPoints,\
                          'progress_param_range':np.linspace(-0.8, -5., nPoints), \
                          'svm_param_range': np.logspace(-2.5, 0, nPoints),\
                          'fixed_param_range': np.linspace(0.0, -1.5, nPoints),\
                          'cssvm_param_range': np.logspace(-4.0, 2.0, nPoints),\
                          ## 'svm_param_range': np.logspace(-4, 1.2, nPoints),\
                          'hmmosvm_param_range': np.logspace(-4.0, 1.0, nPoints),\
                          'osvm_param_range': np.linspace(0.1, 2.0, nPoints),\
                          'sgd_param_range': np.logspace(-4, 1.2, nPoints)}        
        
    elif dim == 2:
        handFeatures = ['unimodal_ftForce',\
                        'crossmodal_targetEEDist' ]
        HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 1.4, 'scale': 3.0}
        SVM_param_dict = {'renew': False, 'w_negative': 3.5, 'gamma': 0.0147, 'cost': 3.0,\
                          'hmmosvm_nu': 0.00316}

        nPoints        = 20  # 'progress_time_cluster',,'fixed' , 'svm' , 
        ROC_param_dict = {'methods': [ 'fixed', 'progress_time_cluster', 'svm', 'hmmosvm'],\
                          'update_list': ['hmmosvm'],\
                          'nPoints': nPoints,\
                          'progress_param_range':np.linspace(-0.8, -8., nPoints), \
                          'svm_param_range': np.logspace(-1.5, 1, nPoints),\
                          'fixed_param_range': np.linspace(0.2, -3.0, nPoints),\
                          'cssvm_param_range': np.logspace(-4.0, 2.0, nPoints),\
                          ## 'svm_param_range': np.logspace(-4, 1.2, nPoints),\
                          'hmmosvm_param_range': np.logspace(-4.0, 1.5, nPoints),\
                          'osvm_param_range': np.linspace(0.1, 2.0, nPoints),\
                          'sgd_param_range': np.logspace(-4, 1.2, nPoints)}        
        
    rawFeatures = ['relativePose_target_EE', \
                   'wristAudio', \
                   'ft' ]                                
    modality_list = ['kinematics', 'audioWrist', 'ft', 'vision_artag', \
                     'vision_change', 'pps']
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

    save_data_path = os.path.expanduser('~')+\
      '/hrl_file_server/dpark_data/anomaly/RSS2016/'+task+'_data/'+\
      str(data_param_dict['downSampleSize'])+'_'+str(dim)
      
    param_dict = {'data_param': data_param_dict, 'AE': AE_param_dict, 'HMM': HMM_param_dict, \
                  'SVM': SVM_param_dict, 'ROC': ROC_param_dict}
      
    return raw_data_path, save_data_path, param_dict


def getFeeding(task, data_renew, AE_renew, HMM_renew, rf_center,local_range, ae_swtch=False, dim=4):

    if dim == 4:

        handFeatures = ['unimodal_audioWristRMS', 'unimodal_ftForce', \
                        'crossmodal_artagEEDist', 'crossmodal_artagEEAng']
        HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 3.5, 'scale': 4.111}
        ## HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 3.5, 'scale': 4.111}
        SVM_param_dict = {'renew': False, 'w_negative': 0.825, 'gamma': 5.0, 'cost': 3.5,\
                          'hmmosvm_nu': 0.000316,\
                          'osvm_nu': 0.000359,\
                          'hmmsvm_diag_w_negative': 0.525, 'hmmsvm_diag_cost': 7.5, \
                          'hmmsvm_diag_gamma': 2.0,\
                          'osvm_window_size': 10,\
                          'hmmsvm_dL_w_negative': 0.525, 'hmmsvm_dL_cost': 12.5, \
                          'hmmsvm_dL_gamma': 4.0}
                          

        nPoints        = 20 #, 'hmmosvm',
        ROC_param_dict = {'methods': ['progress_time_cluster', 'svm','fixed', 'change', 'osvm', 'hmmsvm_diag', 'hmmsvm_dL' ],\
                          'update_list': ['svm'],\
                          'nPoints': nPoints,\
                          'progress_param_range': -np.logspace(0., 1.5, nPoints),\
                          'svm_param_range': np.logspace(-2.2, 0.8, nPoints),\
                          'hmmsvm_diag_param_range': np.logspace(-4, 1.2, nPoints),\
                          'hmmsvm_dL_param_range': np.logspace(-4, 1.2, nPoints),\
                          'hmmosvm_param_range': np.logspace(-4.0, 1.0, nPoints),\
                          'change_param_range': np.logspace(-0.8, 1.0, nPoints)*-1.0,\
                          'osvm_param_range': np.logspace(-5., 0.0, nPoints),\
                          'fixed_param_range': np.linspace(1.0, -3.0, nPoints),\
                          'cssvm_param_range': np.logspace(0.0, 2.0, nPoints) }
    elif dim == 3:

        handFeatures = ['unimodal_ftForce', \
                        'crossmodal_artagEEDist', 'crossmodal_artagEEAng']
        HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 6.0, 'scale': 8.777}
        SVM_param_dict = {'renew': False, 'w_negative': 0.825, 'gamma': 3.911, 'cost': 2.25,\
                          'hmmosvm_nu': 0.00316}

        nPoints        = 20 #'svm','hmmosvm'
        ROC_param_dict = {'methods': ['progress_time_cluster', 'fixed', 'svm', 'hmmosvm'],\
                          'update_list': ['hmmosvm'],\
                          'nPoints': nPoints,\
                          'progress_param_range': -np.logspace(0., 1.5, nPoints),\
                          'svm_param_range': np.logspace(-2.0, 0.25, nPoints),\
                          'hmmosvm_param_range': np.logspace(-4.0, 1.0, nPoints),\
                          'fixed_param_range': np.linspace(1.0, -3.0, nPoints),\
                          'cssvm_param_range': np.logspace(0.0, 2.0, nPoints) }
    elif dim == 2:

        handFeatures = ['unimodal_ftForce', \
                        'crossmodal_artagEEDist']
        HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 3.5, 'scale': 13.444}
        ## HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 6.0, 'scale': 3.0}
        SVM_param_dict = {'renew': False, 'w_negative': 5.0, 'gamma': 2.049, 'cost': 1.75,\
                          'hmmosvm_nu': 0.001}

        nPoints        = 20 #, 'hmmosvm'
        ROC_param_dict = {'methods': ['progress_time_cluster', 'svm','fixed', 'hmmosvm'],\
                          'update_list': ['svm'],\
                          'nPoints': nPoints,\
                          'progress_param_range': -np.logspace(0., 1.5, nPoints),\
                          'svm_param_range': np.logspace(-1.8, 0.25, nPoints),\
                          'hmmosvm_param_range': np.logspace(-4.0, 1.0, nPoints),\
                          'fixed_param_range': np.linspace(1.0, -3.0, nPoints),\
                          'cssvm_param_range': np.logspace(0.0, 2.0, nPoints) }

        
    rawFeatures = ['relativePose_target_EE', \
                   'wristAudio', \
                   'ft' ]
                   #'relativePose_artag_EE', \

    modality_list   = ['ft' ,'kinematics', 'audioWrist', 'vision_artag']
    raw_data_path  = os.path.expanduser('~')+'/hrl_file_server/dpark_data/anomaly/RSS2016/'

    AE_param_dict  = {'renew': AE_renew, 'switch': False, 'time_window': 4, \
                      'layer_sizes':[64,dim], 'learning_rate':1e-6, 'learning_rate_decay':1e-6, \
                      'momentum':1e-6, 'dampening':1e-6, 'lambda_reg':1e-6, \
                      'max_iteration':30000, 'min_loss':0.1, 'cuda':True, \
                      'filter':True, 'filterDim':4,\
                      'add_option': None, 'rawFeatures': rawFeatures,\
                      'add_noise_option': [], 'preTrainModel': None}                      

    data_param_dict= {'renew': data_renew, 'rf_center': rf_center, 'local_range': local_range,\
                      'downSampleSize': 200, 'cut_data': None, \
                      'nNormalFold':3, 'nAbnormalFold':3,\
                      'handFeatures': handFeatures, 'lowVarDataRemv': False,\
                      'handFeatures_noise': True}

    save_data_path = os.path.expanduser('~')+\
      '/hrl_file_server/dpark_data/anomaly/RSS2016/'+task+'_data/'+\
      str(data_param_dict['downSampleSize'])+'_'+str(dim)

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
        HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 1.577, 'scale': 4.0}
        SVM_param_dict = {'renew': False, 'w_negative': 0.56, 'gamma': 0.0527, 'cost': 1.75,\
                          'hmmosvm_nu': 0.01}

        nPoints        = 20  # 'progress_time_cluster',,'fixed' , 'svm' , 
        ROC_param_dict = {'methods': [ 'fixed', 'progress_time_cluster', 'svm','hmmosvm' ],\
                          'update_list': ['progress_time_cluster'],\
                          'nPoints': nPoints,\
                          'progress_param_range':np.logspace(-0.9, 1.2, nPoints)*-1.0 -1., \
                          'fixed_param_range': np.linspace(1.0, -5.0, nPoints),\
                          'svm_param_range': np.logspace(-2.2, 1.3, nPoints),\
                          'cssvm_param_range': np.logspace(-4.0, 2.0, nPoints),\
                          ## 'svm_param_range': np.logspace(-4, 1.2, nPoints),\
                          'hmmosvm_param_range': np.logspace(-4.0, 1.3, nPoints),\
                          'osvm_param_range': np.linspace(0.1, 2.0, nPoints),\
                          'sgd_param_range': np.logspace(-4, 1.2, nPoints)}        
        
    elif dim == 4:
        handFeatures = ['unimodal_ftForce',\
                        'crossmodal_artagEEDist',\
                        'crossmodal_subArtagEEDist',\
                        'unimodal_audioWristRMS'] #'unimodal_audioPower', ,
        if ae_swtch:
            HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 0.73, 'scale': 5.5}
            SVM_param_dict = {'renew': False, 'w_negative': 0.2, 'gamma': 2.5, 'cost': 4.0}
        else:
            HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 1.341, 'scale': 7.33}
            SVM_param_dict = {'renew': False, 'w_negative': 1.175, 'gamma': 0.006, 'cost': 5.,\
                              'sgd_gamma':0.32, 'sgd_w_negative':2.5,\
                              'hmmosvm_nu': 0.01,
                              'hmmsvm_diag_w_negative': 0.525, 'hmmsvm_diag_cost': 15.0, \
                              'hmmsvm_diag_gamma': 0.50749,\
                              'osvm_nu': 0.000215, 'osvm_window_size': 10,\
                              'hmmsvm_dL_w_negative': 0.85, 'hmmsvm_dL_cost': 15.0, \
                              'hmmsvm_dL_gamma': 0.01}

        nPoints        = 20   
        ROC_param_dict = {'methods': [ 'change','progress_time_cluster','fixed' , 'svm' , 'hmmosvm', \
                                       'hmmsvm_diag', 'osvm', 'hmmsvm_dL' ],\
                          'update_list': ['hmmsvm_dL' ],\
                          'nPoints': nPoints,\
                          'progress_param_range':np.linspace(-1, -13., nPoints), \
                          'svm_param_range': np.logspace(-2.0, 0.5, nPoints),\
                          'hmmsvm_diag_param_range': np.logspace(-4, 1.2, nPoints),\
                          'hmmsvm_dL_param_range': np.logspace(-3, 1.2, nPoints),\
                          'hmmosvm_param_range': np.logspace(-4.0, 1.0, nPoints),\
                          'cssvm_param_range': np.logspace(-4.0, 2.0, nPoints),
                          'fixed_param_range': np.logspace(-1, 0.0, nPoints)*-5.0+1.5,\
                          'change_param_range': np.logspace(0., 1.3, nPoints)*-1.0,\
                          'osvm_param_range': np.logspace(-6, 0.0, nPoints),\
                          'sgd_param_range': np.logspace(-4, 1.2, nPoints) }        
        
    elif dim == 3:
        handFeatures = ['unimodal_ftForce',\
                        'crossmodal_artagEEDist',\
                        'unimodal_audioWristRMS'] #'unimodal_audioPower', ,
        HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 1.57, 'scale': 8.}
        SVM_param_dict = {'renew': False, 'w_negative': 0.85, 'gamma': 0.006, 'cost':15.0,\
                          'hmmosvm_nu': 0.00316}
        
        nPoints        = 20  # 'progress_time_cluster',,'fixed' , 'svm' , 
        ROC_param_dict = {'methods': [ 'fixed', 'progress_time_cluster', 'svm','hmmosvm' ],\
                          'update_list': ['hmmosvm'],\
                          'nPoints': nPoints,\
                          'progress_param_range':np.linspace(0.0, -8.0, nPoints), \
                          'svm_param_range': np.logspace(-2.5, 0.8, nPoints),\
                          'fixed_param_range': np.linspace(1.0, -3.0, nPoints),\
                          'cssvm_param_range': np.logspace(-4.0, 2.0, nPoints),\
                          'hmmosvm_param_range': np.logspace(-2.0, 1.4, nPoints),\
                          'osvm_param_range': np.linspace(0.1, 2.0, nPoints)}        
    elif dim == 2:
        handFeatures = ['unimodal_ftForce',\
                        'unimodal_audioWristRMS'] #'unimodal_audioPower', ,
        HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 0.1, 'scale': 10.0}
        ## HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 3.11, 'scale': 6.0}
        SVM_param_dict = {'renew': False, 'w_negative': 0.293, 'gamma': 4.0, 'cost': 3.83,\
                          'hmmosvm_nu': 0.00316,
                          'hmmsvm_diag_w_negative': 1.5, 'hmmsvm_diag_cost': 12.5, \
                          'hmmsvm_diag_gamma': 0.01}
                          

        nPoints        = 20  #
        ROC_param_dict = {'methods': ['fixed', 'progress_time_cluster', 'svm', 'hmmosvm'],\
                          'update_list': [ 'svm' ],\
                          'nPoints': nPoints,\
                          'progress_param_range':np.linspace(0.0, -8.0, nPoints), \
                          'svm_param_range': np.logspace(-3.0, 0.5, nPoints),\
                          'fixed_param_range': np.linspace(1.0, -3.0, nPoints),\
                          'cssvm_param_range': np.logspace(-4.0, 2.0, nPoints),\
                          'hmmosvm_param_range': np.logspace(-4.0, 1.0, nPoints),\
                          'osvm_param_range': np.linspace(0.1, 1.0, nPoints),\
                          'hmmsvm_diag_param_range': np.logspace(-4, 1.2, nPoints)}        
        
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
        HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 2.0, 'scale': 6.66 }
        ## HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 3.5, 'scale': 5.5}
        SVM_param_dict = {'renew': False, 'w_negative': 0.1, 'gamma': 1.85, 'cost': 2.5,\
                          'hmmosvm_nu': 0.00316}

        nPoints        = 20  # 'progress_time_cluster',,'fixed' , 'svm' , 
        ROC_param_dict = {'methods': [ 'fixed', 'progress_time_cluster', 'svm', 'hmmosvm'],\
                          'update_list': ['hmmosvm'],\
                          'nPoints': nPoints,\
                          'progress_param_range':np.linspace(0.0, -8., nPoints), \
                          'svm_param_range': np.logspace(-2.5, 0, nPoints),\
                          'fixed_param_range': np.linspace(1.0, -3.0, nPoints),\
                          'cssvm_param_range': np.logspace(-4.0, 2.0, nPoints),
                          'hmmosvm_param_range': np.logspace(-4.0, 1.0, nPoints),\
                          'osvm_param_range': np.linspace(0.1, 2.0, nPoints),\
                          'svm_param_range': np.logspace(-4, 1.2, nPoints)}        
        
    elif dim == 4:
        handFeatures = ['unimodal_ftForce',\
                        'crossmodal_artagEEDist',\
                        'crossmodal_subArtagEEDist',\
                        'unimodal_audioWristRMS'] #'unimodal_audioPower', ,
        if ae_swtch:
            HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 0.73, 'scale': 5.5}
            SVM_param_dict = {'renew': False, 'w_negative': 0.2, 'gamma': 2.5, 'cost': 4.0}
        else:
            HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 5.0, 'scale': 7.333, \
                              'add_logp_d': True}
            SVM_param_dict = {'renew': False, 'w_negative': 0.177, 'gamma': 0.9777, 'cost': 3.25,\
                              'osvm_nu': 4.64e-05,\
                              'hmmosvm_nu': 0.001,
                              'hmmsvm_diag_w_negative': 0.85, 'hmmsvm_diag_cost': 15.0, \
                              'hmmsvm_diag_gamma': 0.01,\
                              'osvm_window_size': 10,\
                              'hmmsvm_dL_w_negative': 0.2, 'hmmsvm_dL_cost': 5.0, \
                              'hmmsvm_dL_gamma': 0.50749}                              

        nPoints        = 20  # 'progress_time_cluster',,'fixed' , 'svm' , 
        ROC_param_dict = {'methods': [ 'change', 'fixed', 'progress_time_cluster', 'svm', 'hmmsvm_diag', 'hmmosvm', 'osvm', 'hmmsvm_dL' ],\
                          'update_list': [ 'hmmsvm_dL' ],\
                          'nPoints': nPoints,\
                          'progress_param_range':np.logspace(0, 1.5, nPoints)*-1.0, \
                          'svm_param_range': np.logspace(-2.5, -0.8, nPoints),\
                          'fixed_param_range': (-np.logspace(0.0,1.,nPoints)**2)/10.0+0.4,\
                          'change_param_range': np.logspace(-0.8, 1.0, nPoints)*-1.0,\
                          'cssvm_param_range': np.logspace(-4.0, 2.0, nPoints),
                          'hmmosvm_param_range': np.logspace(-4.0, 1.0, nPoints),\
                          'hmmsvm_diag_param_range': np.logspace(-4, 1.5, nPoints),\
                          'hmmsvm_dL_param_range': np.logspace(-4, 1.2, nPoints),\
                          'osvm_param_range': np.logspace(-5., 0.0, nPoints)}        
        
    elif dim == 3:
        handFeatures = ['unimodal_ftForce',\
                        'crossmodal_artagEEDist',\
                        'unimodal_audioWristRMS'] #'unimodal_audioPower', ,
        HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 3.66, 'scale': 6.0, \
                          'add_logp_d': True}
        ## HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 6.83, 'scale': 7.0, \
        ##                   'add_logp_d': True}
        SVM_param_dict = {'renew': False, 'w_negative': 0.749, 'gamma': 0.1, 'cost': 4.0,\
                          'hmmosvm_nu': 0.003}
        
        nPoints        = 20  # 'progress_time_cluster',,'fixed' , 'svm' , 
        ROC_param_dict = {'methods': [ 'fixed', 'progress_time_cluster', 'svm','hmmosvm' ],\
                          'update_list': ['svm'],\
                          'nPoints': nPoints,\
                          'progress_param_range':np.linspace(0.0, -8.0, nPoints), \
                          'svm_param_range': np.logspace(-1.3, -0.2, nPoints),\
                          'fixed_param_range': np.linspace(1.0, -3.0, nPoints),\
                          'cssvm_param_range': np.logspace(-4.0, 2.0, nPoints),
                          'hmmosvm_param_range': np.logspace(-2.0, 2.0, nPoints),\
                          'osvm_param_range': np.linspace(0.1, 2.0, nPoints),\
                          'svm_param_range': np.logspace(-4, 1.2, nPoints)}        
    elif dim == 2:
        handFeatures = ['unimodal_ftForce',\
                        'unimodal_audioWristRMS'] #'unimodal_audioPower', ,
        HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 1.5, 'scale': 2.0}
        SVM_param_dict = {'renew': False, 'w_negative': 0.1778, 'gamma': 8.0, 'cost': 1.75,\
                          'hmmosvm_nu': 0.00316}

        nPoints        = 20  # 'progress_time_cluster',,'fixed' , 'svm' , 
        ROC_param_dict = {'methods': [ 'fixed', 'progress_time_cluster', 'svm','hmmosvm' ],\
                          'update_list': ['svm'],\
                          'nPoints': nPoints,\
                          'progress_param_range':np.linspace(0.0, -8.0, nPoints), \
                          'svm_param_range': np.logspace(-2.5, 0, nPoints),\
                          'fixed_param_range': np.linspace(1.0, -3.0, nPoints),\
                          'cssvm_param_range': np.logspace(-4.0, 2.0, nPoints),\
                          'hmmosvm_param_range': np.logspace(-4.0, 1.0, nPoints),\
                          'osvm_param_range': np.linspace(0.1, 2.0, nPoints),\
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
          '/hrl_file_server/dpark_data/anomaly/RSS2016/'+task+'_data/AE200_'+str(dim)
        data_param_dict['downSampleSize'] = 200
        AE_param_dict['layer_sizes']      = [64,dim]
        AE_param_dict['nAugment']         = 0
        
    elif AE_param_dict['method']=='ae' and pre_train:
        save_data_path = os.path.expanduser('~')+\
          '/hrl_file_server/dpark_data/anomaly/RSS2016/'+task+'_data/AE200_'+str(dim)
        data_param_dict['downSampleSize'] = 200
        AE_param_dict['layer_sizes']      = [64,dim]
        AE_param_dict['add_option']       = None
        AE_param_dict['learning_rate'] = 1e-6            
    else:
        save_data_path = os.path.expanduser('~')+\
          '/hrl_file_server/dpark_data/anomaly/RSS2016/'+task+'_data/200_'+str(dim)
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
        HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 3.61, 'scale': 8.0}
        SVM_param_dict = {'renew': False, 'w_negative': 0.2, 'gamma': 3.0, 'cost': 3.775,\
                          'cssvm_w_negative': 8.0, 'cssvm_gamma': 0.1, 'cssvm_cost': 8.0,\
                          'hmmosvm_nu': 0.001}
                          
        ROC_param_dict = {'methods': [ 'fixed', 'progress_time_cluster', 'svm', 'hmmosvm' ],\
                          'update_list': ['hmmosvm'],\
                          'nPoints': nPoints,\
                          'progress_param_range':np.linspace(0.2, -5., nPoints), \
                          'svm_param_range': np.logspace(-2, 0.5, nPoints),\
                          'fixed_param_range': np.linspace(0.5, -3.0, nPoints),\
                          'hmmosvm_param_range': np.logspace(-4.0, 1.0, nPoints),\
                          'osvm_param_range': np.linspace(0.1, 2.0, nPoints),\
                          'cssvm_param_range': np.logspace(-4.0, 2.0, nPoints) }        
    elif dim == 4:
        handFeatures = ['unimodal_ftForce',\
                        'crossmodal_artagEEDist',\
                        'crossmodal_subArtagEEDist',\
                        'unimodal_audioWristRMS'] #'unimodal_audioPower', ,
        HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 1.278, 'scale': 4.11}
        ## HMM_param_dict = {'renew': HMM_renew, 'nState': 10, 'cov': 1.6, 'scale': 0.01}
        SVM_param_dict = {'renew': False, 'w_negative': 0.825, 'gamma': 0.1, 'cost': 10.0,\
                          'cssvm_w_negative': 2.0, 'cssvm_gamma': 0.05, 'cssvm_cost': 9.75,\
                          'osvm_nu': 0.00001,\
                          'hmmosvm_nu': 0.00316,\
                          'hmmsvm_diag_w_negative': 0.2, 'hmmsvm_diag_cost': 15.0, \
                          'hmmsvm_diag_gamma': 0.5075,\
                          'osvm_window_size': 10,\
                          'hmmsvm_dL_w_negative': 1.175, 'hmmsvm_dL_cost': 7.5, \
                          'hmmsvm_dL_gamma': 0.50749}
                          
        ROC_param_dict = {'methods': [ 'progress_time_cluster', 'svm', 'fixed', 'change', 'osvm', 'hmmsvm_dL', 'hmmosvm', 'hmmsvm_diag' ],\
                          'update_list': [ 'hmmsvm_diag' ],\
                          'nPoints': nPoints,\
                          'progress_param_range':np.linspace(-0.5, -11., nPoints), \
                          'svm_param_range': np.logspace(-2.5, 0.8, nPoints),\
                          'fixed_param_range': np.linspace(-1.0, 0.8, nPoints),\
                          'change_param_range': np.logspace(-0.2, 1.5, nPoints)*-1.0,\
                          'hmmsvm_diag_param_range': np.logspace(-4, 1.2, nPoints),\
                          'hmmsvm_dL_param_range': np.logspace(-4, 1.2, nPoints),\
                          'hmmosvm_param_range': np.logspace(-4.0, 1.0, nPoints),\
                          'osvm_param_range': np.logspace(-6, 0.2, nPoints),\
                          'cssvm_param_range': np.logspace(-4.0, 2.0, nPoints) }        
    elif dim == 3:
        handFeatures = ['unimodal_ftForce',\
                        'crossmodal_artagEEDist',\
                        'unimodal_audioWristRMS'] #'unimodal_audioPower', ,
        HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 2.444, 'scale': 5.667}
        ## HMM_param_dict = {'renew': HMM_renew, 'nState': 10, 'cov': 1.66, 'scale': 10}
        SVM_param_dict = {'renew': False, 'w_negative': 4.0, 'gamma': 0.039, 'cost': 4.0,\
                          'cssvm_w_negative': 8.0, 'cssvm_gamma': 0.1, 'cssvm_cost': 8.0,\
                          'hmmosvm_nu': 0.00316}
        ROC_param_dict = {'methods': [ 'progress_time_cluster', 'svm', 'fixed', 'hmmosvm' ],\
                          'update_list': ['progress_time_cluster', 'fixed'],\
                          'nPoints': nPoints,\
                          'progress_param_range':np.linspace(1., -4., nPoints), \
                          'svm_param_range': np.logspace(-2, 0.4, nPoints),\
                          'fixed_param_range': np.linspace(2.0, -2.0, nPoints),\
                          'hmmosvm_param_range': np.logspace(-4.0, 1.0, nPoints),\
                          'osvm_param_range': np.linspace(0.1, 2.0, nPoints),\
                          'cssvm_param_range': np.logspace(-4.0, 2.0, nPoints) }        
    elif dim == 2:
        handFeatures = ['unimodal_ftForce',\
                        'unimodal_audioWristRMS'] #'unimodal_audioPower', ,
        HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 4.0, 'scale': 2.555}
        SVM_param_dict = {'renew': False, 'w_negative': 0.825, 'gamma': 0.1, 'cost': 7.75,\
                          'hmmosvm_nu': 0.001}                          
        ROC_param_dict = {'methods': [ 'progress_time_cluster', 'svm', 'fixed', 'hmmosvm' ],\
                          'update_list': ['fixed', 'svm'],\
                          'nPoints': nPoints,\
                          'progress_param_range':np.linspace(-8., -1.0, nPoints), \
                          'svm_param_range': np.logspace(-1.4, -0.1, nPoints),\
                          'fixed_param_range': np.linspace(-1.7, -2., nPoints),\
                          'hmmosvm_param_range': np.logspace(-4.0, 1.0, nPoints),\
                          'osvm_param_range': np.linspace(0.1, 2.0, nPoints),\
                          'cssvm_param_range': np.logspace(-4.0, 2.0, nPoints) }        
        
                        
    rawFeatures = ['relativePose_artag_EE', \
                   'wristAudio', \
                   'ft' ]                                
    modality_list   = ['kinematics', 'audio', 'ft', 'vision_artag'] # raw plot
    raw_data_path  = os.path.expanduser('~')+'/hrl_file_server/dpark_data/anomaly/RSS2016/'

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
