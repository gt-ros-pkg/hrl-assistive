import os, sys
import numpy as np


def getParams(task, bDataRenew, bHMMRenew, bCFRenew, dim=0, rf_center='kinEEPos',\
              local_range=10.0, bAESwitch=False, nPoints=None ):

    #---------------------------------------------------------------------------
    if task == 'scooping':
        raw_data_path, save_data_path, param_dict = getScooping(task, bDataRenew, \
                                                                bHMMRenew, bCFRenew, \
                                                                rf_center, local_range,\
                                                                ae_swtch=bAESwitch, dim=dim,\
                                                                nPoints=nPoints)
        
    #---------------------------------------------------------------------------
    elif task == 'feeding':
        raw_data_path, save_data_path, param_dict = getFeeding(task, bDataRenew, \
                                                               bHMMRenew, bCFRenew, \
                                                               rf_center, local_range,\
                                                               ae_swtch=bAESwitch, dim=dim,\
                                                               nPoints=nPoints)

    else:
        print "Selected task name is not available."
        sys.exit()

    # common params
    param_dict['ROC']['methods'] = [ 'hmmgp']
    param_dict['ROC']['update_list'] = [ ]
    param_dict['SVM']['raw_window_size'] = 5

    if bDataRenew or bHMMRenew:
        param_dict['HMM']['renew'] = True
        param_dict['SVM']['renew'] = True

    return raw_data_path, save_data_path, param_dict


def getFeeding(task, data_renew, HMM_renew, CF_renew, rf_center='kinEEPos',local_range=10.0, \
               ae_swtch=False, dim=4, nPoints=None):

    if nPoints is None: nPoints = 40 

    handFeatures = [['unimodal_audioWristRMS',  \
                     'unimodal_kinJntEff_1',\
                     'unimodal_ftForce_integ',\
                     'unimodal_kinEEChange',\
                     'crossmodal_landmarkEEDist'
                     ],\
                     ['unimodal_kinVel',\
                      'unimodal_ftForce_zero',\
                      'crossmodal_landmarkEEDist'
                     ]]

    staticFeatures = ['unimodal_audioWristFrontRMS',\
                      'unimodal_audioWristAzimuth',\
                      'unimodal_ftForceX',\
                      'unimodal_ftForceY',\
                      'unimodal_fabricForce',  \
                      'unimodal_landmarkDist',\
                      'crossmodal_landmarkEEAng',\
                      ]

    HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 2.64, 'scale': 6.111,\
                      'add_logp_d': False }
    SVM_param_dict = {'renew': CF_renew, 'w_negative': 1.0, 'gamma': 5.0, 'cost': 1.0,\
                      'hmmosvm_nu': 0.5,\
                      'osvm_nu': 0.5,\
                      'logp_offset': 0,\
                      'nugget': 10.0, 'theta0': 1.0,\
                      'std_offset': 1.4464
                      }


    ROC_param_dict = {'nPoints': nPoints,\
                      'm2o': {'gp_nSubsample': 20, 'alpha_coeff': 0.15, 'hmm_scale': 9.0, 'hmm_cov': 9.0,\
                              'noise_max': 0.0 },\
                      'o2o': {'gp_nSubsample': 40, 'alpha_coeff': 0.05, 'hmm_scale': 3.0, 'hmm_cov': 1.0,\
                              'noise_max': 0.05 },\
                      'progress_param_range': -np.logspace(0.6, 0.9, nPoints)+1.0,\
                      'progress_diag_param_range': -np.logspace(-0.7, 1.4, nPoints),\
                      'svm_param_range': np.logspace(-2.4, 0.5, nPoints),\
                      'hmmgp_param_range':np.logspace(0.3, 1.9, nPoints)*-1.0, \
                      'hmmosvm_param_range': np.logspace(-4.0, 0.2, nPoints),\
                      'change_param_range': np.logspace(0.5, 2.1, nPoints)*-1.0,\
                      'osvm_param_range': np.logspace(-7., 0.5, nPoints),\
                      'fixed_param_range': np.linspace(0.1, 0.01, nPoints),\
                      'rnd_param_range': 1.0-np.logspace(-1, -0.75, nPoints)+0.1,\
                      'bpsvm_param_range': np.logspace(-2.2, 0.5, nPoints),\
                      'sgd_param_range': np.logspace(-1, 1., nPoints)}


    AD_param_dict = {'svm_w_positive': 1.0, 'sgd_w_positive': 1.0, 'sgd_n_iter': 20}

    rawFeatures = ['relativePose_target_EE', \
                   'wristAudio', \
                   'ft',\
                   'relativePose_landmark_EE']

    # 012 3 45678910 11121314 15161718 19 20 2122 2324
    isolationFeatures = ['unimodal_audioWristRMS', \
                         'unimodal_audioWristFrontRMS',\
                         'unimodal_audioWristAzimuth',\
                         'unimodal_kinVel',\
                         'unimodal_kinJntEff_1', \
                         'unimodal_kinJntEff_2', \
                         'unimodal_kinJntEff_3', \
                         'unimodal_kinJntEff_4', \
                         'unimodal_kinJntEff_5', \
                         'unimodal_kinJntEff_6', \
                         'unimodal_kinJntEff_7', \
                         'unimodal_ftForce',\
                         'unimodal_ftForce_zero',\
                         'unimodal_ftForce_integ',\
                         'unimodal_ftForce_delta',\
                         'unimodal_ftForceX',\
                         'unimodal_ftForceY',\
                         'unimodal_ftForceZ',\
                         'unimodal_ftForce_XY',\
                         'unimodal_fabricForce',\
                         'unimodal_landmarkDist',\
                         'unimodal_kinEEChange',\
                         'unimodal_kinDesEEChange',\
                         'crossmodal_landmarkEEDist', \
                         'crossmodal_landmarkEEAng',\
                         ] 
                   

    modality_list   = ['ft' ,'kinematics', 'audioWrist', 'vision_landmark']
    raw_data_path  = os.path.expanduser('~')+'/hrl_file_server/dpark_data/anomaly/AURO2016/'

    data_param_dict= {'renew': data_renew, 'rf_center': rf_center, 'local_range': local_range,\
                      'downSampleSize': 140, 'cut_data': None, \
                      'nNormalFold':2, 'nAbnormalFold':2,\
                      'handFeatures': handFeatures,\
                      'staticFeatures': staticFeatures,\
                      'lowVarDataRemv': False,\
                      'isolationFeatures': isolationFeatures,\
                      'handFeatures_noise': True, 'max_time': 7.0}

    save_data_path = os.path.expanduser('~')+\
      '/hrl_file_server/dpark_data/anomaly/AURO2016/'+task+'_data/'+\
      str(data_param_dict['downSampleSize'])+'_'+str(dim)

    param_dict = {'data_param': data_param_dict, 'HMM': HMM_param_dict, \
                  'SVM': SVM_param_dict, 'ROC': ROC_param_dict, 'AD': AD_param_dict}
    ## param_dict = {'data_param': data_param_dict, 'AE': AE_param_dict, 'HMM': HMM_param_dict, \
    ##               'SVM': SVM_param_dict, 'ROC': ROC_param_dict, 'AD': AD_param_dict}

    return raw_data_path, save_data_path, param_dict


def getScooping(task, data_renew, HMM_renew, CF_renew, rf_center='kinEEPos', local_range=10.0, \
                pre_train=False,\
                ae_swtch=False, dim=4, nPoints=None):

    if nPoints is None: nPoints = 20  

    handFeatures = ['unimodal_ftForce',\
                    'crossmodal_targetEEDist', \
                    'crossmodal_targetEEAng', \
                    'unimodal_audioWristRMS']
    HMM_param_dict = {'renew': HMM_renew, 'nState': 20, 'cov': 3.566, 'scale': 1.0,
                      'add_logp_d': False}
    SVM_param_dict = {'renew': CF_renew, 'w_negative': 0.2, 'gamma': 0.01, 'cost': 15.0,\
                      'hmmosvm_nu': 0.00316,\
                      'hmmsvm_diag_w_negative': 0.85, 'hmmsvm_diag_cost': 12.5, \
                      'hmmsvm_diag_gamma': 0.01,\
                      'osvm_nu': 0.000215,\
                      'hmmsvm_dL_w_negative': 0.85, 'hmmsvm_dL_cost': 7.5, \
                      'hmmsvm_dL_gamma': 0.50749,
                      }

    ROC_param_dict = {'nPoints': nPoints,\
                      'hmmgp_param_range':np.linspace(0, -40.0, nPoints), \
                      'progress_param_range':np.linspace(0.0, -7., nPoints), \
                      'svm_param_range': np.logspace(-4, 1.2, nPoints),\
                      'change_param_range': np.logspace(-0.8, 1.0, nPoints)*-1.0,\
                      'fixed_param_range': np.logspace(0.0, 0.5, nPoints)*-1.0+1.3,\
                      'cssvm_param_range': np.logspace(-4.0, 2.0, nPoints),\
                      'hmmosvm_param_range': np.logspace(-4.0, 1.0, nPoints),\
                      'hmmsvm_diag_param_range': np.logspace(-4, 1.2, nPoints),\
                      'hmmsvm_dL_param_range': np.logspace(-4, 1.2, nPoints),\
                      'osvm_param_range': np.logspace(-5., 0.0, nPoints),\
                      'sgd_param_range': np.logspace(-4.0, 1.3, nPoints)}

    AD_param_dict = {'svm_w_positive': 1.0, 'sgd_w_positive': 1.0, 'sgd_n_iter': 20}
       
        
    rawFeatures = ['relativePose_target_EE', \
                   'wristAudio', \
                   'ft' ]                                
    modality_list = ['kinematics', 'audioWrist', 'ft', \
                     'pps']
    raw_data_path  = os.path.expanduser('~')+'/hrl_file_server/dpark_data/anomaly/AURO2016/'

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
                      'downSampleSize': 140, 'cut_data': None, \
                      'nNormalFold':3, 'nAbnormalFold':3,\
                      'handFeatures': handFeatures, 'lowVarDataRemv': False,\
                      'handFeatures_noise': True, 'max_time': None}

    save_data_path = os.path.expanduser('~')+\
      '/hrl_file_server/dpark_data/anomaly/AURO2016/'+task+'_data/'+\
      str(data_param_dict['downSampleSize'])+'_'+str(dim)
      
    param_dict = {'data_param': data_param_dict, 'AE': AE_param_dict, 'HMM': HMM_param_dict, \
                  'SVM': SVM_param_dict, 'ROC': ROC_param_dict, 'AD': AD_param_dict}
      
    return raw_data_path, save_data_path, param_dict

