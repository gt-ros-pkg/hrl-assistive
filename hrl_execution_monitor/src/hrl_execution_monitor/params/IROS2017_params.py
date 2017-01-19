import os, sys
import numpy as np


def getParams(task, bDataRenew=False, bHMMRenew=False, bCFRenew=False, dim=0, rf_center='kinEEPos',\
              local_range=10.0, nPoints=None ):

    #---------------------------------------------------------------------------
    if task == 'scooping':
        raw_data_path, save_data_path, param_dict = getScooping(task, bDataRenew, \
                                                                bHMMRenew, bCFRenew, \
                                                                rf_center, local_range,\
                                                                dim=dim,\
                                                                nPoints=nPoints)
        
    #---------------------------------------------------------------------------
    elif task == 'feeding':
        raw_data_path, save_data_path, param_dict = getFeeding(task, bDataRenew, \
                                                               bHMMRenew, bCFRenew, \
                                                               rf_center, local_range,\
                                                               dim=dim,\
                                                               nPoints=nPoints)

    else:
        print "Selected task name is not available."
        sys.exit()

    # common params
    param_dict['ROC']['methods'] = [ 'hmmgp0', 'hmmgp1']
    param_dict['ROC']['update_list'] = [ ]
    param_dict['SVM']['raw_window_size'] = 5

    if bDataRenew or bHMMRenew:
        param_dict['HMM']['renew'] = True
        param_dict['SVM']['renew'] = True

    return raw_data_path, save_data_path, param_dict


def getFeeding(task, data_renew, HMM_renew, CF_renew, rf_center='kinEEPos',local_range=10.0, \
               dim=4, nPoints=None):

    if nPoints is None: nPoints = 40 

    handFeatures  = [['unimodal_audioWristRMS',  \
                     'unimodal_kinJntEff_1',\
                     'unimodal_ftForce_integ',\
                     'unimodal_kinEEChange',\
                     'crossmodal_landmarkEEDist'
                     ],
                     ['unimodal_kinVel',\
                      'unimodal_ftForce_zero',\
                      'crossmodal_landmarkEEDist']]


    staticFeatures = ['unimodal_audioWristFrontRMS',\
                      'unimodal_audioWristAzimuth',\
                      'unimodal_ftForceX',\
                      'unimodal_ftForceY',\
                      'unimodal_fabricForce',  \
                      'unimodal_landmarkDist',\
                      'crossmodal_landmarkEEAng',\
                      ]                                                  
        

    HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 2.645, 'scale': [7.0, 15.0],\
                      'add_logp_d': False }
    SVM_param_dict = {'renew': CF_renew,\
                      'logp_offset': 0.,\
                      'nugget': 10.0, 'theta0': 1.0,\
                      'std_offset': 1.4464
                      }

    ROC_param_dict = {'nPoints': nPoints,\
                      'progress_param_range': -np.logspace(0.6, 0.9, nPoints)+1.0,\
                      'hmmgp_param_range': np.logspace(-0., 2.3, nPoints)*-1.0+1.0, \
                      'change_param_range': np.logspace(0.5, 2.1, nPoints)*-1.0,\
                      'fixed_param_range': np.linspace(0.1, 0.01, nPoints),\
                      'rnd_param_range': 1.0-np.logspace(-1, -0.75, nPoints)+0.1,\
                      'hmmgp0_param_range': np.logspace(0.1, 2.3, nPoints)*-1.0+1.0,\
                      'hmmgp1_param_range': np.logspace(0.1, 2.5, nPoints)*-1.0+0.5}
        

    AD_param_dict = {'svm_w_positive': 1.0, 'sgd_w_positive': 1.0, 'sgd_n_iter': 20}


    # 012 3 45678910 11121314 151617 18 19 2021 2223
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
                         'unimodal_fabricForce',\
                         'unimodal_landmarkDist',\
                         'unimodal_kinEEChange',\
                         'unimodal_kinDesEEChange',\
                         'crossmodal_landmarkEEDist', \
                         'crossmodal_landmarkEEAng',\
                         ] 
                   

    raw_data_path  = os.path.expanduser('~')+'/hrl_file_server/dpark_data/anomaly/AURO2016/'

    data_param_dict= {'renew': data_renew, 'rf_center': rf_center, 'local_range': local_range,\
                      'downSampleSize': 140, 'cut_data': None, \
                      'nNormalFold':3, 'nAbnormalFold':3,\
                      'handFeatures': handFeatures, 'staticFeatures': staticFeatures,\
                      'lowVarDataRemv': False,\
                      'isolationFeatures': isolationFeatures,\
                      'handFeatures_noise': True, 'max_time': 7.0}

    save_data_path = None
    param_dict = {'data_param': data_param_dict, 'HMM': HMM_param_dict, \
                  'SVM': SVM_param_dict, 'ROC': ROC_param_dict, 'AD': AD_param_dict}

    return raw_data_path, save_data_path, param_dict


def getScooping(task, data_renew, HMM_renew, CF_renew, rf_center='kinEEPos', local_range=10.0, \
                pre_train=False, dim=4, nPoints=None):

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

    AD_param_dict = {}
       
    raw_data_path  = os.path.expanduser('~')+'/hrl_file_server/dpark_data/anomaly/AURO2016/'

    data_param_dict= {'renew': data_renew, 'rf_center': rf_center, 'local_range': local_range,\
                      'downSampleSize': 140, 'cut_data': None, \
                      'nNormalFold':3, 'nAbnormalFold':3,\
                      'handFeatures': handFeatures, 'lowVarDataRemv': False,\
                      'handFeatures_noise': True, 'max_time': None}

    save_data_path = os.path.expanduser('~')+\
      '/hrl_file_server/dpark_data/anomaly/AURO2016/'+task+'_data/'+\
      str(data_param_dict['downSampleSize'])+'_'+str(dim)
      
    param_dict = {'data_param': data_param_dict, 'HMM': HMM_param_dict, \
                  'SVM': SVM_param_dict, 'ROC': ROC_param_dict, 'AD': AD_param_dict}
      
    return raw_data_path, save_data_path, param_dict

