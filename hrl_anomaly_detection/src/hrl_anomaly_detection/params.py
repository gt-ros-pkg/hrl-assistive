import os, sys
import numpy as np


def getParams(task, bDataRenew, bHMMRenew, bCFRenew, dim, rf_center='kinEEPos',\
              local_range=10.0, bAESwitch=False, nPoints=None ):

    #---------------------------------------------------------------------------
    if task == 'scooping':
        raw_data_path, save_data_path, param_dict = getScooping(task, bDataRenew, \
                                                                bHMMRenew, bCFRenew, \
                                                                rf_center, local_range,\
                                                                ae_swtch=bAESwitch, dim=dim, \
                                                                nPoints=nPoints)
        
    #---------------------------------------------------------------------------
    elif task == 'feeding':
        raw_data_path, save_data_path, param_dict = getFeeding(task, bDataRenew, \
                                                               bHMMRenew, bCFRenew, \
                                                               rf_center, local_range,\
                                                               ae_swtch=bAESwitch, dim=dim,\
                                                               nPoints=nPoints)
        
    #---------------------------------------------------------------------------           
    elif task == 'pushing_microwhite':
        raw_data_path, save_data_path, param_dict = getPushingMicroWhite(task, bDataRenew, \
                                                                         bHMMRenew, bCFRenew, \
                                                                         rf_center, local_range, \
                                                                         ae_swtch=bAESwitch, dim=dim,\
                                                                         nPoints=nPoints)
                                                                         
    #---------------------------------------------------------------------------           
    elif task == 'pushing_microblack':
        raw_data_path, save_data_path, param_dict = getPushingMicroBlack(task, bDataRenew, \
                                                                         bHMMRenew, bCFRenew, \
                                                                         rf_center, local_range, \
                                                                         ae_swtch=bAESwitch, dim=dim,\
                                                                         nPoints=nPoints)
        
    #---------------------------------------------------------------------------           
    elif task == 'pushing_toolcase':
        raw_data_path, save_data_path, param_dict = getPushingToolCase(task, bDataRenew, \
                                                                       bHMMRenew, bCFRenew, \
                                                                       rf_center, local_range, \
                                                                       ae_swtch=bAESwitch, dim=dim,\
                                                                       nPoints=nPoints)
    else:
        print "Selected task name is not available."
        sys.exit()


    # common params
    if dim == 4:
        param_dict['ROC']['methods'] = [ 'fixed', 'change', 'progress', 'progress_diag', \
                                         'osvm', 'hmmosvm', 'kmean', 'progress_osvm', 'progress_svm',\
                                         'hmmgp', 'rnd', 'svm'] #'progress_state', 
        ## param_dict['ROC']['methods'] = [ 'hmmgp' ]
        ## param_dict['ROC']['update_list'] = [ 'progress_osvm', 'progress_svm']
        param_dict['ROC']['update_list'] = [ 'svm' ]
        ## param_dict['ROC']['update_list'] = [ ]
    else:
        param_dict['ROC']['methods'] = [ 'fixed', 'change', 'progress', 'osvm', 'hmmosvm', 'kmean', 'hmmgp',\
                                         ]
        ## param_dict['ROC']['update_list'] = [ 'change']
    param_dict['SVM']['raw_window_size'] = 5

    return raw_data_path, save_data_path, param_dict
    

def getScooping(task, data_renew, HMM_renew, CF_renew, rf_center,local_range, pre_train=False,\
                ae_swtch=False, dim=4, nPoints=None):

    if nPoints is None: nPoints = 40
    if dim == 4:
        handFeatures = ['unimodal_ftForce',\
                        'crossmodal_targetEEDist', \
                        'crossmodal_targetEEAng', \
                        'unimodal_audioWristRMS']
        HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 5.0, 'scale': 7.333,
                          'add_logp_d': False}
        SVM_param_dict = {'renew': CF_renew, 'w_negative': 1.9952, 'gamma': 0.0464, 'cost': 6.0,\
                          'hmmosvm_nu': 0.00316,\
                          'hmmsvm_diag_w_negative': 0.85, 'hmmsvm_diag_cost': 15.0, \
                          'hmmsvm_diag_gamma': 0.01,\
                          'osvm_nu': 0.0003, \
                          'hmmsvm_dL_w_negative': 1.5, 'hmmsvm_dL_cost': 12.5, \
                          'hmmsvm_dL_gamma': 0.01,
                          'hmmsvm_no_dL_w_negative': 1.25, 'hmmsvm_no_dL_cost': 15.0, \
                          'hmmsvm_no_dL_gamma': 0.0316,\
                          'bpsvm_cost': 5.0,\
                          'bpsvm_gamma': 0.507, \
                          'bpsvm_w_negative': 0.2,\
                          'progress_svm_w_negative': 1.5, 'progress_svm_cost': 15.0, \
                          'progress_svm_gamma': 0.01 }
                          
        # , 'hmmsvm_dL', 'hmmosvm', 'hmmsvm_diag', 'progress_state', 'svm', 'hmmsvm_no_dL'
        ROC_param_dict = {'methods': [ 'fixed', 'change', 'progress', 'progress_state', 'osvm', 'progress_diag', ],\
                          'update_list': [ 'progress_svm', 'change'],\
                          'nPoints': nPoints,\
                          'progress_param_range':np.logspace(-0.5, 1.3, nPoints)*-1.0 + 0.5, \
                          'progress_diag_param_range':np.logspace(-2.0, 1.3, nPoints)*-1.0, \
                          'progress_state_param_range':np.logspace(0, 2.5, nPoints)*-1+3.0, \
                          'progress_svm_param_range': np.linspace(0.005, 6.0, nPoints),\
                          'progress_osvm_param_range': np.logspace(-6.0, 1.0, nPoints),\
                          'hmmgp_param_range':np.logspace(-1., 1.8, nPoints)*-1.0+0.5, \
                          'kmean_param_range':np.logspace(-1.0, 1.1, nPoints)*-1.0 + 0.5, \
                          'svm_param_range': np.logspace(-2, 0.2553, nPoints),\
                          'change_param_range': np.linspace(-1.0, -22.0, nPoints),\
                          'fixed_param_range': np.linspace(-1.5, 0.3, nPoints),\
                          'cssvm_param_range': np.logspace(-4.0, 2.0, nPoints),\
                          'hmmosvm_param_range': np.logspace(-4.0, 0.3, nPoints),\
                          'hmmsvm_diag_param_range': np.logspace(-3.7, -0.1, nPoints),\
                          'hmmsvm_dL_param_range': np.logspace(-3.0, 0.25, nPoints),\
                          'hmmsvm_no_dL_param_range': np.logspace(-2.53, 0.176, nPoints),\
                          'osvm_param_range': np.logspace(-5., 0.1, nPoints),\
                          'bpsvm_param_range': np.logspace(-2.2, 0.5, nPoints),\
                          'rnd_param_range': 1.0-np.logspace(-1, -0.75, nPoints)+0.1,\
                          'sgd_param_range': np.logspace(-4, 1.2, nPoints)}        
        
    elif dim == 3:
        handFeatures = ['unimodal_ftForce',\
                        'crossmodal_targetEEDist', \
                        'crossmodal_targetEEAng']
        HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 2.0, 'scale': 7.333, \
                          'add_logp_d': False}
        SVM_param_dict = {'renew': CF_renew, 'w_negative': 2.25, 'gamma': 1.05925, 'cost': 5.4,\
                          'hmmosvm_nu': 0.00316,\
                          'hmmsvm_diag_w_negative': 0.85, 'hmmsvm_diag_cost': 12.5, \
                          'hmmsvm_diag_gamma': 0.01}

        ROC_param_dict = {'methods': [ 'fixed',  'kmean'],\
                          'update_list': [ 'progress' ],\
                          'nPoints': nPoints,\
                          'hmmgp_param_range':np.linspace(3., -40.0, nPoints), \
                          'progress_param_range':np.logspace(0.01, 1.2, nPoints)*-1.0, \
                          'kmean_param_range':np.linspace(-1.2, -3.5, nPoints), \
                          'svm_param_range': np.logspace(-2.523, 0.34, nPoints),\
                          'fixed_param_range': np.linspace(1.0, -0.43, nPoints),\
                          'cssvm_param_range': np.logspace(-4.0, 2.0, nPoints),\
                          'hmmosvm_param_range': np.logspace(-4.0, 0.5, nPoints),\
                          'change_param_range': np.linspace(-1.0, -10.0, nPoints),\
                          'osvm_param_range': np.logspace(-4, 0.5, nPoints),\
                          'sgd_param_range': np.logspace(-4, 1.2, nPoints)}        
        
    elif dim == 2:
        handFeatures = ['unimodal_ftForce',\
                        'crossmodal_targetEEDist' ]
        HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 4.0, 'scale': 6.44,
                          'add_logp_d': False}
        SVM_param_dict = {'renew': CF_renew, 'w_negative': 1.0, 'gamma': 5.011, 'cost': 4.599,\
                          'hmmosvm_nu': 0.00316}

        ROC_param_dict = {'methods': [ 'fixed', 'kmean'],\
                          'update_list': [ 'progress', 'kmean', 'hmmgp' ],\
                          'nPoints': nPoints,\
                          'hmmgp_param_range':np.logspace(-1, 1.9, nPoints)*-1.0, \
                          'progress_param_range':np.logspace(-0.0, 0.8, nPoints)*-1.0, \
                          'kmean_param_range':np.logspace(-0.2, 0.8, nPoints)*-1.0, \
                          'svm_param_range': np.logspace(-2.0, -0.3307, nPoints),\
                          'fixed_param_range': np.linspace(0.2, -0.6, nPoints),\
                          'cssvm_param_range': np.logspace(-4.0, 2.0, nPoints),\
                          'hmmosvm_param_range': np.logspace(-4.0, 1.0, nPoints),\
                          'change_param_range': np.linspace(-1.5, -20.0, nPoints),\
                          'osvm_param_range': np.logspace(-3, 0.8, nPoints),\
                          'sgd_param_range': np.logspace(-4, 1.2, nPoints)}        
        
    rawFeatures = ['relativePose_target_EE', \
                   'wristAudio', \
                   'ft' ]                                
    modality_list = ['kinematics', 'audioWrist', 'ft', 'vision_artag', \
                     'vision_change', 'pps']
    raw_data_path  = '/home/dpark/hrl_file_server/dpark_data/anomaly/RSS2016/'

    AE_param_dict  = {'renew': False, 'switch': ae_swtch, 'method': 'ae', 'time_window': 4,  \
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


def getFeeding(task, data_renew, HMM_renew, CF_renew, rf_center,local_range, ae_swtch=False, dim=4,\
               nPoints=None):
    if nPoints is None: nPoints = 40

    if dim == 4:

        handFeatures = ['unimodal_audioWristRMS', 'unimodal_ftForce', \
                        'crossmodal_artagEEDist', 'crossmodal_artagEEAng']
        HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 0.1, 'scale': 10.33, \
                          'add_logp_d': False}
        ## HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 7.0, 'scale': 7.0, \
        ##                   'add_logp_d': False}
        ## SVM_param_dict = {'renew': CF_renew, 'w_negative': 1.05, 'gamma': 7.122, 'cost': 2.066,\
        SVM_param_dict = {'renew': CF_renew, 'w_negative': 1.0, 'gamma': 10.0, 'cost': 1.0,\
                          'nu': 0.6896,\
                          'hmmosvm_nu': 0.001,\
                          'osvm_nu': 0.000359,\
                          'hmmsvm_diag_w_negative': 0.85, \
                          'hmmsvm_diag_cost': 12.5, \
                          'hmmsvm_diag_gamma': 0.01,\
                          'hmmsvm_dL_w_negative': 1.175, \
                          'hmmsvm_dL_cost': 7.5, \
                          'hmmsvm_dL_gamma': 0.01,\
                          'hmmsvm_no_dL_w_negative': 0.45, \
                          'hmmsvm_no_dL_cost': 3.25, \
                          'hmmsvm_no_dL_gamma': 4.625,\
                          'bpsvm_cost': 7.5,\
                          'bpsvm_gamma': 0.01, \
                          'bpsvm_w_negative': 1.175,\
                          'progress_svm_w_negative': 1.5, 'progress_svm_cost': 15.0, \
                          'progress_svm_gamma': 0.01 }

                          
        ## ROC_param_dict = {'methods': ['fixed', 'progress', 'hmmosvm', 'svm', 'change', 'hmmsvm_diag', 'hmmsvm_dL', 'hmmsvm_no_dL'],\
        ROC_param_dict = {'methods': [ 'fixed', 'change', 'progress', 'progress_state', 'osvm', 'progress_diag', ],\
                          'update_list': [ 'progress_svm', 'progress_state', 'progress_diag', 'progress', 'hmmosvm', 'hmmgp', 'fixed', 'change'],\
                          'nPoints': nPoints,\
                          'progress_param_range': -np.logspace(-1.5, 1.9, nPoints),\
                          'progress_diag_param_range': -np.logspace(-1.5, 1.6, nPoints),\
                          'kmean_param_range': -np.logspace(-1, 1.3, nPoints)+1.0,\
                          'progress_state_param_range':np.linspace(0, -115., nPoints), \
                          'progress_svm_param_range': np.linspace(0.005, 4.5, nPoints),\
                          'progress_osvm_param_range': np.logspace(-6.0, 1.0, nPoints),\
                          'hmmgp_param_range':np.logspace(-0.5, 1.8, nPoints)*-1.0, \
                          'svm_param_range': np.logspace(-2.15, -0.101, nPoints),\
                          'hmmsvm_diag_param_range': np.logspace(-4, 0.0, nPoints),\
                          'hmmsvm_dL_param_range': np.logspace(-4, 0.14, nPoints),\
                          'hmmsvm_no_dL_param_range': np.logspace(-2.52, -0.45, nPoints),\
                          'hmmosvm_param_range': np.logspace(-4., 0., nPoints),\
                          'change_param_range': np.linspace(0.0, -30.0, nPoints),\
                          'osvm_param_range': np.logspace(-5., 0.0, nPoints),\
                          'bpsvm_param_range': np.logspace(-2.2, 0.5, nPoints),\
                          'fixed_param_range': np.linspace(0.247, -0.4, nPoints),\
                          'rnd_param_range': 1.0-np.logspace(-1, -0.75, nPoints)+0.1,\
                          'cssvm_param_range': np.logspace(0.0, 2.0, nPoints) }
                          #np.logspace(-8.0, 1.0, nPoints)
    elif dim == 3:

        handFeatures = ['unimodal_ftForce', \
                        'crossmodal_artagEEDist', 'crossmodal_artagEEAng']
        HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 4.0, 'scale': 10.0, \
                          'add_logp_d': False}
        SVM_param_dict = {'renew': CF_renew, 'w_negative': 0.1, 'gamma': 3.911, 'cost': 1.625,\
                          'hmmosvm_nu': 0.001,\
                          'hmmsvm_bpsvm_cost': 12.5,\
                          'hmmsvm_bpsvm_gamma': 0.507, \
                          'hmmsvm_bpsvm_w_negative': 0.2
                          }
                          

        ROC_param_dict = {'methods': ['progress', 'fixed', 'kmean'],\
                          'update_list': [ 'hmmgp' ],\
                          'nPoints': nPoints,\
                          'hmmgp_param_range':np.logspace(0.3, 2.3, nPoints)*-1.0, \
                          'progress_param_range': -np.logspace(-2, 1.8, nPoints),\
                          'kmean_param_range': -np.logspace(-3, 1.8, nPoints),\
                          'svm_param_range': np.logspace(-1.8, 0.67, nPoints),\
                          'bpsvm_param_range': np.logspace(-2, 0, nPoints),\
                          'hmmosvm_param_range': np.logspace(-6.0, 0.0, nPoints),\
                          'change_param_range': np.linspace(1.0, -60.0, nPoints),\
                          'fixed_param_range': np.linspace(0.240, -0.007, nPoints),\
                          'osvm_param_range': np.logspace(-5., 0.0, nPoints),\
                          'cssvm_param_range': np.logspace(0.0, 2.0, nPoints) }
    elif dim == 2:

        handFeatures = ['unimodal_ftForce', \
                        'crossmodal_artagEEDist']
        HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 3.0, 'scale': 3.0, \
                          'add_logp_d': False}
        SVM_param_dict = {'renew': CF_renew, 'w_negative': 5.0, 'gamma': 0.1, 'cost': 2.5,\
                          'hmmosvm_nu': 0.001,\
                          'hmmsvm_bpsvm_cost': 15.0,\
                          'hmmsvm_bpsvm_gamma': 0.01, \
                          'hmmsvm_bpsvm_w_negative': 1.5
                          }

        ROC_param_dict = {'methods': ['progress', 'fixed', 'kmean'],\
                          'update_list': ['change' ],\
                          'nPoints': nPoints,\
                          'hmmgp_param_range':np.linspace(3., -40.0, nPoints), \
                          'progress_param_range': -np.logspace(-0.1, 1.6, nPoints)+0.3,\
                          'kmean_param_range': -np.logspace(-0.1, 1.2, nPoints)+0.3,\
                          'svm_param_range': np.logspace(-2.7, 5.850, nPoints),\
                          'hmmosvm_param_range': np.logspace(-4.0, 0.0, nPoints),\
                          'change_param_range': np.logspace(0, 1.7, nPoints)*-1.0,\
                          'fixed_param_range': np.linspace(0.18, 0.0, nPoints),\
                          'bpsvm_param_range': np.logspace(-2, 0, nPoints),\
                          'osvm_param_range': np.logspace(-5., 0.0, nPoints),\
                          'cssvm_param_range': np.logspace(0.0, 2.0, nPoints) }
    else:
        print "Not available dimension"
        sys.exit()

        
    rawFeatures = ['relativePose_target_EE', \
                   'wristAudio', \
                   'ft' ]
                   #'relativePose_artag_EE', \

    modality_list   = ['ft' ,'kinematics', 'audioWrist', 'vision_artag']
    raw_data_path  = os.path.expanduser('~')+'/hrl_file_server/dpark_data/anomaly/RSS2016/'

    AE_param_dict  = {'renew': False, 'switch': False, 'time_window': 4, \
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
                      'handFeatures_noise': True, 'max_time': 7.0}

    save_data_path = os.path.expanduser('~')+\
      '/hrl_file_server/dpark_data/anomaly/RSS2016/'+task+'_data/'+\
      str(data_param_dict['downSampleSize'])+'_'+str(dim)

    param_dict = {'data_param': data_param_dict, 'AE': AE_param_dict, 'HMM': HMM_param_dict, \
                  'SVM': SVM_param_dict, 'ROC': ROC_param_dict}

    return raw_data_path, save_data_path, param_dict

def getPushingMicroWhite(task, data_renew, HMM_renew, CF_renew, rf_center,local_range, pre_train=False, \
                         ae_swtch=False, dim=3, nPoints=None):
    if nPoints is None: nPoints = 40

    if dim == 5:
        handFeatures = ['unimodal_ftForce',\
                        'crossmodal_artagEEDist',\
                        'crossmodal_artagEEAng',\
                        'crossmodal_subArtagEEDist',\
                        'unimodal_audioWristRMS'] #'unimodal_audioPower', ,
        HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 1.788, 'scale': 10.0, \
                          'add_logp_d': False}
        SVM_param_dict = {'renew': CF_renew, 'w_negative': 0.525, 'gamma': 0.0316, 'cost': 4.0,\
                          'hmmosvm_nu': 0.00316}

        ROC_param_dict = {'methods': ['fixed', 'progress', 'kmean'],\
                          'update_list': [ 'hmmgp', 'fixed' ],\
                          'nPoints': nPoints,\
                          'hmmgp_param_range':np.logspace(-1, 2.0, nPoints)*-1.0, \
                          'progress_param_range':np.logspace(-1.0, 1.2, nPoints)*-1.0 +1., \
                          'kmean_param_range':np.logspace(-1.1, 1.0, nPoints)*-1.0 -0.5, \
                          'fixed_param_range': np.linspace(-7.0, 0.0, nPoints),\
                          'svm_param_range': np.logspace(-2.5, -0.276, nPoints),\
                          'cssvm_param_range': np.logspace(-4.0, 2.0, nPoints),\
                          'hmmosvm_param_range': np.logspace(-4.0, 1.8, nPoints),\
                          'change_param_range': np.linspace(-1.0, -20.0, nPoints),\
                          'osvm_param_range': np.logspace(-5, 0, nPoints),\
                          'sgd_param_range': np.logspace(-4, 1.2, nPoints)}        
        
    elif dim == 4:
        handFeatures = ['unimodal_ftForce',\
                        'crossmodal_artagEEDist',\
                        'crossmodal_subArtagEEDist',\
                        'unimodal_audioWristRMS'] #'unimodal_audioPower', ,
        HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 5.33, 'scale': 7.33, \
                          'add_logp_d': False}
        SVM_param_dict = {'renew': CF_renew, 'w_negative': 0.85, 'gamma': 0.001, 'cost': 12.5,\
                          'sgd_gamma':0.32, 'sgd_w_negative':2.5,\
                          'hmmosvm_nu': 0.00316,
                          'hmmsvm_diag_w_negative': 1.5, \
                          'hmmsvm_diag_cost': 10.0, \
                          'hmmsvm_diag_gamma': 0.01,\
                          'osvm_nu': 0.000359,
                          'hmmsvm_dL_w_negative': 1.5, \
                          'hmmsvm_dL_cost': 7.5, \
                          'hmmsvm_dL_gamma': 0.01,\
                          'hmmsvm_no_dL_w_negative': 3.25, \
                          'hmmsvm_no_dL_cost': 3.75, \
                          'hmmsvm_no_dL_gamma': 0.0133,\
                          'bpsvm_cost': 5.0,\
                          'bpsvm_gamma': 0.507, \
                          'bpsvm_w_negative': 1.5,\
                          'progress_svm_w_negative': 1.5, 'progress_svm_cost': 15.0, \
                          'progress_svm_gamma': 0.01 }

        # 'svm' , 'hmmosvm', 'hmmsvm_diag', 'hmmsvm_dL', 'hmmsvm_no_dL', , 
        ROC_param_dict = {'methods': [ 'change','fixed','progress', 'progress_state', \
                                       'progress_diag', 'kmean', 'osvm', 'hmmosvm',\
                                       'progress_osvm'],\
                          'update_list': [ 'progress' ],\
                          'nPoints': nPoints,\
                          'progress_param_range':np.logspace(0.0, 1.15, nPoints)*-1.0, \
                          'progress_diag_param_range':np.logspace(0, 1.2, nPoints)*-1.0, \
                          'kmean_param_range':np.logspace(-1.1, 1.2, nPoints)*-1.0 -1., \
                          'progress_state_param_range':np.logspace(-0.4, 3.3, nPoints)*-1.0+0.4, \
                          'progress_svm_param_range': np.linspace(0.005, 6.0, nPoints),\
                          'progress_osvm_param_range': np.logspace(-6.0, 1.0, nPoints),\
                          'hmmgp_param_range':np.logspace(-0.1, 2.1, nPoints)*-1.0, \
                          'fixed_param_range': np.linspace(-1.1, 0.171, nPoints),\
                          'change_param_range': np.logspace(0.2, 1.3, nPoints)*-1.0,\
                          'osvm_param_range': np.logspace(-6, 0.0, nPoints),\
                          'svm_param_range': np.logspace(-1.236, 0.7, nPoints),\
                          'bpsvm_param_range': np.logspace(-3., 0.4, nPoints),\
                          'hmmsvm_diag_param_range': np.logspace(-1.85, 0.486, nPoints),\
                          'hmmsvm_dL_param_range': np.logspace(-3.0, 0.7, nPoints),\
                          'hmmsvm_no_dL_param_range': np.logspace(-1.346, 0.8, nPoints),\
                          'hmmosvm_param_range': np.logspace(-4.0, 1.5, nPoints),\
                          'rnd_param_range': 1.0-np.logspace(-3, -0.9, nPoints)+0.001,\
                          'cssvm_param_range': np.logspace(-4.0, 2.0, nPoints),
                          'sgd_param_range': np.logspace(-4, 1.2, nPoints) }        
        
    elif dim == 3:
        handFeatures = ['unimodal_ftForce',\
                        'crossmodal_artagEEDist',\
                        'unimodal_audioWristRMS'] #'unimodal_audioPower', ,
        ## HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 1.0, 'scale': 8.,\
        ##                   'add_logp_d': False}
        HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 4.06, 'scale': 10.,\
                          'add_logp_d': False}
        SVM_param_dict = {'renew': CF_renew, 'w_negative': 1.175, 'gamma': 0.0063, 'cost':7.5,\
                          'hmmosvm_nu': 0.001}
        
        ROC_param_dict = {'methods': ['fixed', 'progress', 'kmean'],\
                          'update_list': [ 'progress' ],\
                          'nPoints': nPoints,\
                          'hmmgp_param_range':np.logspace(-0.6, 1.8, nPoints)*-1.0, \
                          'progress_param_range':np.logspace(-1, 1.3, nPoints)*-1.0 + 1.0, \
                          'kmean_param_range':np.logspace(-1.1, 1.0, nPoints)*-1.0 -0.5, \
                          'svm_param_range': np.logspace(-1.22, 0.2, nPoints),\
                          'fixed_param_range': np.linspace(0.3, -5.0, nPoints),\
                          'cssvm_param_range': np.logspace(-4.0, 2.0, nPoints),\
                          'hmmosvm_param_range': np.logspace(-3.0, 0.3, nPoints),\
                          'change_param_range': np.linspace(-1.5, -20.0, nPoints),\
                          'osvm_param_range': np.logspace(-5, 0., nPoints)}        
    elif dim == 2:
        handFeatures = ['unimodal_ftForce',\
                        'unimodal_audioWristRMS'] #'unimodal_audioPower', ,
        HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 0.3, 'scale': 10.0,\
                          'add_logp_d': False}
        SVM_param_dict = {'renew': CF_renew, 'w_negative': 1.584, 'gamma': 6.0, 'cost': 2.1666,\
                          'hmmosvm_nu': 0.001,
                          'hmmsvm_diag_w_negative': 1.5, 'hmmsvm_diag_cost': 12.5, \
                          'hmmsvm_diag_gamma': 0.01}
                          

        ROC_param_dict = {'methods': [ ],\
                          'update_list': [ 'kmean' , 'hmmgp'],\
                          'nPoints': nPoints,\
                          'hmmgp_param_range':np.linspace(3., -35.0, nPoints), \
                          'progress_param_range':np.linspace(0.0, -9.0, nPoints), \
                          'kmean_param_range':np.logspace(-0.3, 1.0, nPoints)*-1.0 , \
                          'svm_param_range': np.logspace(-3.0, -0.3458, nPoints),\
                          'fixed_param_range': np.linspace(1.0, -2.0, nPoints),\
                          'cssvm_param_range': np.logspace(-4.0, 2.0, nPoints),\
                          'hmmosvm_param_range': np.logspace(-6.0, 0.3, nPoints),\
                          'osvm_param_range': np.logspace(-5, 0, nPoints),\
                          'change_param_range': np.linspace(-1.0, -10.0, nPoints),\
                          'hmmsvm_diag_param_range': np.logspace(-4, 1.2, nPoints)}        
        
    rawFeatures = ['relativePose_artag_EE', \
                   'wristAudio', \
                   'ft' ]                                
    modality_list   = ['kinematics', 'audio', 'ft', 'vision_artag'] # raw plot
    raw_data_path  = '/home/dpark/hrl_file_server/dpark_data/anomaly/RSS2016/'

    AE_param_dict  = {'renew': False, 'switch': ae_swtch, 'method': 'ae', 'time_window': 4,  \
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
                      'handFeatures_noise': True, 'max_time': None}

    data_param_dict['downSampleSize'] = 200
    save_data_path = os.path.expanduser('~')+\
      '/hrl_file_server/dpark_data/anomaly/RSS2016/'+task+'_data/'+\
      str(data_param_dict['downSampleSize'])+'_'+str(dim)
    AE_param_dict['layer_sizes'] = [64,dim]
    AE_param_dict['add_option'] = None
    AE_param_dict['add_noise_option'] = []
    AE_param_dict['learning_rate'] = 1e-6
    AE_param_dict['preTrainModel'] = os.path.join(save_data_path, 'ae_pretrain_model_'+str(dim)+'.pkl')

    param_dict = {'data_param': data_param_dict, 'AE': AE_param_dict, 'HMM': HMM_param_dict, \
                  'SVM': SVM_param_dict, 'ROC': ROC_param_dict}

    return raw_data_path, save_data_path, param_dict


def getPushingMicroBlack(task, data_renew, HMM_renew, CF_renew, rf_center,local_range, pre_train=False,\
                         ae_swtch=False, dim=3, nPoints=None):

    if nPoints is None: nPoints        = 40  #
    if dim == 5:
        handFeatures = ['unimodal_ftForce',\
                        'crossmodal_artagEEDist',\
                        'crossmodal_artagEEAng',\
                        'crossmodal_subArtagEEDist',\
                        'unimodal_audioWristRMS'] #'unimodal_audioPower', ,
        HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 1.0, 'scale': 8.0,\
                          'add_logp_d': False}
        ## HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 3.5, 'scale': 5.5}
        SVM_param_dict = {'renew': CF_renew, 'w_negative': 1.0, 'gamma': 1.0, 'cost': 1.0,\
                          'nu': 0.5,\
                          'hmmosvm_nu': 0.001}

        ROC_param_dict = {'methods': [ ],\
                          'update_list': [ 'fixed'],\
                          'nPoints': nPoints,\
                          'hmmgp_param_range':np.linspace(3., -40.0, nPoints), \
                          'progress_param_range':np.logspace(0., 1.0, nPoints)*-1.0, \
                          'kmean_param_range':np.logspace(-0.2, 1.0, nPoints)*-1.0, \
                          'fixed_param_range': np.linspace(-1.2, 0.3, nPoints),\
                          'svm_param_range': np.logspace(-1.5, 0.45, nPoints),\
                          'cssvm_param_range': np.logspace(-4.0, 2.0, nPoints),
                          'hmmosvm_param_range': np.logspace(-6.0, 0.3, nPoints),\
                          'change_param_range': np.linspace(-1.0, -10.0, nPoints),\
                          'osvm_param_range': np.logspace(-3, 0.5, nPoints),\
                          }        
        
    elif dim == 4:
        handFeatures = ['unimodal_ftForce',\
                        'crossmodal_artagEEDist',\
                        'crossmodal_subArtagEEDist',\
                        'unimodal_audioWristRMS'] #'unimodal_audioPower', ,
                        
        HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 5.0, 'scale': 7.25, \
                          'add_logp_d': False}
        SVM_param_dict = {'renew': CF_renew, 'w_negative': 0.7498, 'gamma': 6.244, 'cost': 1.75,\
                          'osvm_nu': 0.01,\
                          'hmmosvm_nu': 0.001,
                          'hmmsvm_diag_w_negative': 0.2, 'hmmsvm_diag_cost': 15.0, \
                          'hmmsvm_diag_gamma': 1.005,\
                          'hmmsvm_dL_w_negative': 1.5, 'hmmsvm_dL_cost': 15.0, \
                          'hmmsvm_dL_gamma': 0.01,\
                          'hmmsvm_no_dL_w_negative': 2.0, 'hmmsvm_no_dL_cost': 15.0, \
                          'hmmsvm_no_dL_gamma': 0.01,\
                          'bpsvm_cost': 9.75,\
                          'bpsvm_gamma': 0.5075, \
                          'bpsvm_w_negative': 1.5,\
                          'progress_svm_w_negative': 1.5, 'progress_svm_cost': 15.0, \
                          'progress_svm_gamma': 0.01                              
                          }                              

        ## ROC_param_dict = {'methods': ['fixed', 'change','progress', 'svm', 'hmmsvm_dL', 'hmmosvm', 'hmmsvm_diag', 'hmmsvm_no_dL' ],\
        ROC_param_dict = {'methods': [ 'change','fixed','progress',\
                                       'progress_diag', 'kmean'],\
                          'update_list': [ ],\
                          'nPoints': nPoints,\
                          'progress_param_range':np.logspace(-1., 1.0, nPoints)*-1.0, \
                          'progress_diag_param_range':np.logspace(-1., 1.0, nPoints)*-1.0, \
                          'kmean_param_range':np.logspace(-1, 0.8, nPoints)*-1.0, \
                          'state_kmean_param_range':np.logspace(-1, 0.8, nPoints)*-1.0, \
                          'progress_state_param_range':np.logspace(-0.5, 3.0, nPoints)*-1.0, \
                          'progress_osvm_param_range': np.logspace(-6.0, 1.0, nPoints),\
                          'progress_svm_param_range': np.linspace(0.002, 3.809, nPoints),\
                          'hmmgp_param_range':np.logspace(-1, 1.6, nPoints)*-1.0, \
                          'svm_param_range': np.linspace(0.05, 0.5, nPoints),\
                          'bpsvm_param_range': np.logspace(-4., 0.5, nPoints),\
                          'fixed_param_range': np.linspace(-0.305, 0.85, nPoints ),\
                          'change_param_range': np.logspace(0.0, 1.3, nPoints)*-1.0,\
                          'cssvm_param_range': np.logspace(-4.0, 2.0, nPoints),
                          'hmmosvm_param_range': np.logspace(-6.0, 1.0, nPoints),\
                          'hmmsvm_diag_param_range': np.logspace(-3.0, -0.726, nPoints),\
                          'hmmsvm_dL_param_range': np.logspace(-2.301, 0.303, nPoints),\
                          'hmmsvm_no_dL_param_range': np.logspace(-1.886, 0.33, nPoints),\
                          'osvm_param_range': np.logspace(-4., 0.0, nPoints),\
                          'rnd_param_range': 1.0-np.logspace(-1, -0.75, nPoints)+0.1,\
                          }        
        
    elif dim == 3:
        handFeatures = ['unimodal_ftForce',\
                        'crossmodal_artagEEDist',\
                        'unimodal_audioWristRMS'] #'unimodal_audioPower', ,
        HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 3.0, 'scale': 8.0, \
                          'add_logp_d': False}
        SVM_param_dict = {'renew': CF_renew, 'w_negative': 3.1622, 'gamma': 0.1, 'cost': 2.5,\
                          'hmmosvm_nu': 0.001}
        
        ROC_param_dict = {'methods': [ ],\
                          'update_list': [ 'fixed'],\
                          'nPoints': nPoints,\
                          'hmmgp_param_range':np.logspace(-1, 1.8, nPoints)*-1.0, \
                          'progress_param_range':np.linspace(-0.8, -11.0, nPoints), \
                          'kmean_param_range':np.linspace(0.5, -6.6, nPoints), \
                          'fixed_param_range': np.linspace(-1.0, 1.0, nPoints),\
                          'svm_param_range': np.logspace(-2.0, 0.592, nPoints),\
                          'cssvm_param_range': np.logspace(-4.0, 2.0, nPoints),
                          'hmmosvm_param_range': np.logspace(-4.0, 0.3, nPoints),\
                          'change_param_range': np.linspace(-1.0, -10.0, nPoints),\
                          'osvm_param_range': np.logspace(-3, 0, nPoints),\
                          }        
    elif dim == 2:
        handFeatures = ['unimodal_ftForce',\
                        'unimodal_audioWristRMS'] #'unimodal_audioPower', ,
        ## HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 2.5, 'scale': 7.33,\
        HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 5.0, 'scale': 8.0,\
                          'add_logp_d': False}
        SVM_param_dict = {'renew': CF_renew, 'w_negative': 0.7498, 'gamma': 0.1, 'cost': 2.5,\
                          'hmmosvm_nu': 0.000316}

        ROC_param_dict = {'methods': [ ],\
                          'update_list': [ 'hmmgp', 'fixed' ],\
                          'nPoints': nPoints,\
                          'hmmgp_param_range':np.logspace(-1, 1.8, nPoints)*-1.0, \
                          'progress_param_range':np.logspace(0.0, 1.2, nPoints)*-1.0, \
                          'kmean_param_range':np.logspace(0.0, 1.5, nPoints)*-1.0, \
                          'fixed_param_range': np.linspace(-3.5, 0.5, nPoints),\
                          'svm_param_range': np.logspace(-2.523, 0.0484, nPoints),\
                          'cssvm_param_range': np.logspace(-4.0, 2.0, nPoints),\
                          'hmmosvm_param_range': np.logspace(-3.2, 0.7, nPoints),\
                          'change_param_range': np.linspace(-1.0, -10.0, nPoints),\
                          'osvm_param_range': np.logspace(-3, 0, nPoints)
                          }
            
    rawFeatures = ['relativePose_artag_EE', \
                   'wristAudio', \
                   'ft' ]                                
    modality_list   = ['kinematics', 'audio', 'ft', 'vision_artag'] # raw plot
    raw_data_path  = '/home/dpark/hrl_file_server/dpark_data/anomaly/RSS2016/'

    AE_param_dict  = {'renew': False, 'switch': ae_swtch, 'method': 'ae', 'time_window': 4,  \
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
                      'handFeatures_noise': True, 'max_time': None}

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
        data_param_dict['downSampleSize'] = 200
        save_data_path = os.path.expanduser('~')+\
          '/hrl_file_server/dpark_data/anomaly/RSS2016/'+task+'_data/'+\
          str(data_param_dict['downSampleSize'])+'_'+str(dim)
        AE_param_dict['layer_sizes'] = [64,dim]
        AE_param_dict['add_option']  = None
        AE_param_dict['learning_rate'] = 1e-6
        AE_param_dict['preTrainModel'] = os.path.join(save_data_path, 'ae_pretrain_model_'+str(dim)+'.pkl')
        

    param_dict = {'data_param': data_param_dict, 'AE': AE_param_dict, 'HMM': HMM_param_dict, \
                  'SVM': SVM_param_dict, 'ROC': ROC_param_dict}

    return raw_data_path, save_data_path, param_dict


def getPushingToolCase(task, data_renew, HMM_renew, CF_renew, rf_center,local_range, pre_train=False, \
                       ae_swtch=False, dim=3, nPoints=None):

    
    if nPoints is None: nPoints        = 40  # 'progress',,'fixed' , 'svm' 
    if dim == 5:
        handFeatures = ['unimodal_ftForce',\
                        'crossmodal_artagEEDist',\
                        'crossmodal_artagEEAng',\
                        'crossmodal_subArtagEEDist',\
                        'unimodal_audioWristRMS'] #'unimodal_audioPower', ,
        HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 4.5, 'scale': 5.5, \
                          'add_logp_d': False}
        SVM_param_dict = {'renew': CF_renew, 'w_negative': 2.0, 'gamma': 0.1, 'cost': 5.0,\
                          'hmmosvm_nu': 0.00316}
                          ## 'cssvm_w_negative': 8.0, 'cssvm_gamma': 0.1, 'cssvm_cost': 8.0,\
                          
        ROC_param_dict = {'methods': [ 'fixed', 'progress', 'kmean' ],\
                          'update_list': ['change' ],\
                          'nPoints': nPoints,\
                          'hmmgp_param_range':np.logspace(-1, 2.2, nPoints)*-1.0, \
                          'progress_param_range':np.logspace(0.3, 1.6, nPoints)*-1+1.5, \
                          'kmean_param_range':np.logspace(0.4, 1.3, nPoints)*-1+1.5, \
                          'svm_param_range': np.logspace(-2.16, 0.31, nPoints),\
                          'fixed_param_range': np.linspace(0.1, -2.5, nPoints),\
                          'hmmosvm_param_range': np.logspace(-6.0, 0.6, nPoints),\
                          'osvm_param_range': np.logspace(-5, -0.2, nPoints),\
                          'change_param_range': np.linspace(-.7, -25.0, nPoints),\
                          'cssvm_param_range': np.logspace(-4.0, 2.0, nPoints) }        
    elif dim == 4:
        handFeatures = ['unimodal_ftForce',\
                        'crossmodal_artagEEDist',\
                        'crossmodal_subArtagEEDist',\
                        'unimodal_audioWristRMS'] #'unimodal_audioPower', ,
        ## HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 3.775, 'scale': 6.66, \
        HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 1.73, 'scale': 13.25, \
                          'add_logp_d': False}
        SVM_param_dict = {'renew': CF_renew, 'w_negative': 1.711, 'gamma': 0.01, 'cost': 3.0,\
                          'cssvm_w_negative': 2.0, 'cssvm_gamma': 0.05, 'cssvm_cost': 9.75,\
                          'osvm_nu': 0.00316,\
                          'hmmosvm_nu': 0.001,\
                          'hmmsvm_diag_w_negative': 1.711, 'hmmsvm_diag_cost': 3.0, \
                          'hmmsvm_diag_gamma': 0.01,\
                          'hmmsvm_dL_w_negative': 2.1499, 'hmmsvm_dL_cost': 5.5, \
                          'hmmsvm_dL_gamma': 0.01,\
                          'hmmsvm_no_dL_w_negative': 1.5, 'hmmsvm_no_dL_cost': 8.0, \
                          'hmmsvm_no_dL_gamma': 0.0316,\
                          'bpsvm_cost': 15.25,\
                          'bpsvm_gamma': 1.0, \
                          'bpsvm_w_negative': 1.2589,\
                          'progress_svm_w_negative': 1.5, 'progress_svm_cost': 15.0, \
                          'progress_svm_gamma': 0.01
                          }

        # 'bpsvm', 'osvm', 
        ROC_param_dict = {'methods': ['change','fixed','progress',\
                                       'progress_diag', 'kmean' ],\
                          'update_list': [ 'hmmgp' ],\
                          'nPoints': nPoints,\
                          'progress_param_range':np.logspace(0.0, 1.2, nPoints)*-1.0+1.0, \
                          'progress_diag_param_range':np.logspace(0.0, 1.2, nPoints)*-1.0+1.0, \
                          'kmean_param_range':np.logspace(-0.0, 1.1, nPoints)*-1.0+1.0, \
                          'progress_state_param_range':np.logspace(-0.1, 3.3, nPoints)*-1.0, \
                          'progress_svm_param_range': np.linspace(0.002, 2.07, nPoints),\
                          'progress_osvm_param_range': np.logspace(-6.0, 1.0, nPoints),\
                          'hmmgp_param_range':np.logspace(-1, 2.2, nPoints)*-1.0+0.5, \
                          'svm_param_range': np.logspace(-1.0, 0.046, nPoints),\
                          'fixed_param_range': np.linspace(0.5, -4.0, nPoints),\
                          'change_param_range': np.linspace(-1.2, -20.0, nPoints),\
                          'hmmsvm_diag_param_range': np.logspace(-3, -0.023, nPoints),\
                          'hmmsvm_dL_param_range': np.logspace(-1.18, 0.132, nPoints),\
                          'hmmsvm_no_dL_param_range': np.logspace(-3, 0.045, nPoints),\
                          'hmmosvm_param_range': np.logspace(-4.0, 1.0, nPoints),\
                          'osvm_param_range': np.logspace(-6, 0.2, nPoints),\
                          'bpsvm_param_range': np.logspace(-4.0, 0.7, nPoints),\
                          'rnd_param_range': 1.0-np.logspace(-1, -0.75, nPoints)+0.1,\
                          'cssvm_param_range': np.logspace(-4.0, 2.0, nPoints) }        
    elif dim == 3:
        handFeatures = ['unimodal_ftForce',\
                        'crossmodal_artagEEDist',\
                        'unimodal_audioWristRMS'] #'unimodal_audioPower', ,
        ## HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 0.1, 'scale': 7.66, \
        HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 4.06, 'scale': 10.0, \
                          'add_logp_d': False}
        SVM_param_dict = {'renew': CF_renew, 'w_negative': 3.0, 'gamma': 0.01, 'cost': 5.0,\
                          'cssvm_w_negative': 8.0, 'cssvm_gamma': 0.1, 'cssvm_cost': 8.0,\
                          'hmmosvm_nu': 0.001}
        ROC_param_dict = {'methods': [ 'fixed', 'progress', 'kmean'],\
                          'update_list': [ 'hmmgp' ],\
                          'nPoints': nPoints,\
                          'hmmgp_param_range':np.logspace(-0.2, 2.5, nPoints)*-1.0, \
                          'progress_param_range':np.linspace(1., -20., nPoints), \
                          'kmean_param_range':np.linspace(1., -13., nPoints), \
                          'svm_param_range': np.logspace(-1.087, 0.89, nPoints),\
                          'fixed_param_range': np.linspace(+1.0, -5.0, nPoints),\
                          'hmmosvm_param_range': np.logspace(-4.0, 1.0, nPoints),\
                          'change_param_range': np.linspace(-1.0, -20.0, nPoints),\
                          'osvm_param_range': np.logspace(-4, 0.0, nPoints),\
                          'cssvm_param_range': np.logspace(-4.0, 2.0, nPoints) }        
    elif dim == 2:
        handFeatures = ['unimodal_ftForce',\
                        'unimodal_audioWristRMS'] #'unimodal_audioPower', ,
        HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 3.77, 'scale': 15.00, \
                          'add_logp_d': False}
        SVM_param_dict = {'renew': CF_renew, 'w_negative': 0.575, 'gamma': 0.1, 'cost': 7.75,\
                          'hmmosvm_nu': 0.01}                          
        ROC_param_dict = {'methods': [ 'fixed', 'progress', 'kmean' ],\
                          'update_list': [ 'progress', 'hmmgp','fixed' ],\
                          'nPoints': nPoints,\
                          'hmmgp_param_range':np.logspace(-1, 1.5, nPoints)*-1.0, \
                          'progress_param_range':np.linspace(-20., 0.5, nPoints), \
                          'kmean_param_range':np.logspace(-1, 1, nPoints)*-1.0, \
                          'svm_param_range': np.logspace(-1.087, -0.4, nPoints),\
                          'fixed_param_range': np.linspace(0.0, -15., nPoints),\
                          'change_param_range': np.linspace(-1.0, -10.0, nPoints),\
                          'hmmosvm_param_range': np.logspace(-4.0, 0.5, nPoints),\
                          'osvm_param_range': np.logspace(-3, 0, nPoints),\
                          'cssvm_param_range': np.logspace(-4.0, 2.0, nPoints) }        
        
                        
    rawFeatures = ['relativePose_artag_EE', \
                   'wristAudio', \
                   'ft' ]                                
    modality_list   = ['kinematics', 'audio', 'ft', 'vision_artag'] # raw plot
    raw_data_path  = os.path.expanduser('~')+'/hrl_file_server/dpark_data/anomaly/RSS2016/'

    AE_param_dict  = {'renew': False, 'switch': ae_swtch, 'method': 'ae', 'time_window': 4,  \
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
                      'handFeatures_noise': True, 'max_time': None}

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
        data_param_dict['downSampleSize'] = 200
        save_data_path = os.path.expanduser('~')+\
          '/hrl_file_server/dpark_data/anomaly/RSS2016/'+task+'_data/'+\
          str(data_param_dict['downSampleSize'])+'_'+str(dim)
        AE_param_dict['layer_sizes']      = [64,dim]
        AE_param_dict['add_option']       = None
        AE_param_dict['preTrainModel'] = os.path.join(save_data_path, 'ae_pretrain_model.pkl')
        AE_param_dict['learning_rate'] = 1e-6            
        

    if AE_param_dict['switch'] and AE_param_dict['method']=='pca':            
        SVM_param_dict = {'renew': CF_renew, 'w_negative': 3.0, 'gamma': 0.334, 'cost': 1.0}
        HMM_param_dict = {'renew': HMM_renew, 'nState': 20, 'cov': 5.0, 'scale': 0.5}
    elif AE_param_dict['switch'] and AE_param_dict['method']=='ae':            
        SVM_param_dict = {'renew': CF_renew, 'w_negative': 3.0, 'gamma': 0.334, 'cost': 1.0}
        HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 4.0, 'scale': 1.5}


    param_dict = {'data_param': data_param_dict, 'AE': AE_param_dict, 'HMM': HMM_param_dict, \
                  'SVM': SVM_param_dict, 'ROC': ROC_param_dict}

    return raw_data_path, save_data_path, param_dict
