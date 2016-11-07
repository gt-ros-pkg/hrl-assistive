import os, sys
import numpy as np


def getParams(task, bDataRenew, bHMMRenew, bCFRenew, dim, rf_center='kinEEPos',\
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
    if dim == 5:
        param_dict['ROC']['methods'] = [ 'hmmgp']
        param_dict['ROC']['update_list'] = [ 'hmmgp' ]
    elif dim == 4:
        ## param_dict['ROC']['methods'] = [ 'fixed', 'change', 'progress', 'osvm', 'hmmosvm', 'rnd', 'hmmosvm', 'hmmgp' ]
        # 'fixed', 'change', 'progress', 'progress_diag', 'osvm', 'hmmosvm', 'rnd' 
        param_dict['ROC']['methods'] = [ 'hmmgp']
        param_dict['ROC']['update_list'] = [ 'hmmgp']
    else:
        param_dict['ROC']['methods'] = [ 'fixed', 'change', 'progress', 'osvm', 'hmmosvm', 'hmmgp']
        param_dict['ROC']['update_list'] = [ 'hmmgp']

    param_dict['SVM']['raw_window_size'] = 5

    return raw_data_path, save_data_path, param_dict


def getFeeding(task, data_renew, HMM_renew, CF_renew, rf_center='kinEEPos',local_range=10.0, \
               ae_swtch=False, dim=4, nPoints=None):

    if nPoints is None: nPoints = 40 

    if dim == 5:

        handFeatures = ['unimodal_audioWristRMS', 'unimodal_ftForceZ', 'crossmodal_landmarkEEAng', \
                        'crossmodal_landmarkEEDist', 'unimodal_kinJntEff_1']
                        
                        ## 'unimodal_fabricForce' ,\, 'unimodal_ftForceY'
                        ## 'unimodal_kinEEChange']
                        ## 

            ## ['unimodal_audioWristRMS', 'unimodal_ftForceZ', \
            ##             'crossmodal_landmarkEEDist', 'crossmodal_landmarkEEAng']

        ## HMM_param_dict = {'renew': HMM_renew, 'nState': 20, 'cov': 3.75, 'scale': 15.55,\
        ## HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 1.75, 'scale': 8.55,\
        HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 1.82, 'scale': 13.03,\
                          'add_logp_d': False}
        SVM_param_dict = {'renew': CF_renew, 'w_negative': 1.0, 'gamma': 5.0, 'cost': 1.0,\
                          'hmmosvm_nu': 0.000316,\
                          'osvm_nu': 0.000359,\
                          'logp_offset': 0,\
                          'nugget': 80.0, 'theta0': 0.91,\
                          'std_offset': 1.4464
                          }
                          ## 'nugget': 104.42, 'theta0': 1.42,\
                          ## 'std_offset': 1.4464
                          ## }

        
        ROC_param_dict = {'nPoints': nPoints,\
                          'm2o': {'gp_nSubsample': 20, 'alpha_coeff': 0.15, 'hmm_scale': 9.0, 'hmm_cov': 9.0,\
                                  'noise_max': 0.0 },\
                          'o2o': {'gp_nSubsample': 40, 'alpha_coeff': 0.05, 'hmm_scale': 3.0, 'hmm_cov': 1.0,\
                                  'noise_max': 0.05 },\
                          'progress_param_range': -np.logspace(0, 1.3, nPoints)+1.0,\
                          'kmean_param_range': -np.logspace(0, 3.0, nPoints),\
                          'svm_param_range': np.logspace(-2.4, 0.5, nPoints),\
                          'hmmgp_param_range':np.logspace(0.3, 1.9, nPoints)*-1.0+0.5, \
                          'hmmsvm_diag_param_range': np.logspace(-4, 1.2, nPoints),\
                          'hmmsvm_dL_param_range': np.logspace(-4, 1.2, nPoints),\
                          'hmmosvm_param_range': np.logspace(-4.0, 1.0, nPoints),\
                          'change_param_range': np.logspace(0.0, 2.6, nPoints)*-1.0,\
                          'osvm_param_range': np.logspace(-4., 1.0, nPoints),\
                          'bpsvm_param_range': np.logspace(-2.2, 0.5, nPoints),\
                          'fixed_param_range': np.linspace(0.3, -0.0, nPoints),\
                          'cssvm_param_range': np.logspace(0.0, 2.0, nPoints),\
                          'sgd_param_range': np.logspace(-1, 1., nPoints)}

        # Parameters should be determinded by optimizer.
        if nPoints == 1:
            ROC_param_dict['fixed_param_range'] = [-1.0]
            ROC_param_dict['progress_param_range'] = [-1.8413]
            ROC_param_dict['hmmgp_param_range'] = [-4.9]

        AD_param_dict = {'svm_w_positive': 1.0, 'sgd_w_positive': 1.0, 'sgd_n_iter': 20}
                          

    elif dim == 4:

        handFeatures = ['unimodal_audioWristRMS', 'unimodal_ftForceZ', \
                        'crossmodal_landmarkEEDist', 'unimodal_kinJntEff_1']

        ## handFeatures = ['unimodal_audioWristRMS', 'unimodal_ftForce', \
        ##                 'crossmodal_landmarkEEDist', 'crossmodal_landmarkEEAng']
        ## HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 3.75, 'scale': 15.55,\
        ## HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 2.645, 'scale': 3.111,\
        ## HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 0.615, 'scale': 6.512,\
        HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 2.645, 'scale': 3.111,\
                          'add_logp_d': False}
        SVM_param_dict = {'renew': CF_renew, 'w_negative': 1.0, 'gamma': 5.0, 'cost': 1.0,\
                          'hmmosvm_nu': 0.5,\
                          'osvm_nu': 0.5,\
                          'logp_offset': 0,\
                          'nugget': 50.0, 'theta0': 1.0,\
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
                          'hmmgp_param_range':np.logspace(0.0, 2.0, nPoints)*-1.0, \
                          'hmmosvm_param_range': np.logspace(-4.0, 0.2, nPoints),\
                          'change_param_range': np.logspace(0.5, 2.1, nPoints)*-1.0,\
                          'osvm_param_range': np.logspace(-7., 0.5, nPoints),\
                          'fixed_param_range': np.linspace(0.07, -0.0, nPoints),\
                          'rnd_param_range': 1.0-np.logspace(-1, -0.75, nPoints)+0.1,\
                          'bpsvm_param_range': np.logspace(-2.2, 0.5, nPoints),\
                          'sgd_param_range': np.logspace(-1, 1., nPoints)}

                          ## 'kmean_param_range': -np.logspace(0, 3.0, nPoints),\
                          ## 'hmmsvm_diag_param_range': np.logspace(-4, 1.2, nPoints),\
                          ## 'hmmsvm_dL_param_range': np.logspace(-4, 1.2, nPoints),\

        # Parameters should be determinded by optimizer.
        if nPoints == 1:
            ROC_param_dict['fixed_param_range'] = [-1.0]
            ROC_param_dict['progress_param_range'] = [-1.8413]
            ROC_param_dict['hmmgp_param_range'] = [-4.9]

        AD_param_dict = {'svm_w_positive': 1.0, 'sgd_w_positive': 1.0, 'sgd_n_iter': 20}
                          
    elif dim == 3:

        handFeatures = ['unimodal_audioWristRMS', 'unimodal_ftForceZ', \
                        'crossmodal_landmarkEEDist']
        ## HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 2.88, 'scale': 11.94,\
        HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 0.1, 'scale': 16.36,\
                          'add_logp_d': False}
        SVM_param_dict = {'renew': CF_renew, 'w_negative': 1.55, 'gamma': 3.911, 'cost': 1.0,\
                          'hmmosvm_nu': 0.01,\
                          'osvm_nu': 0.01,\
                          'nugget': 65.70, 'theta0': 1.097,\
                          'std_offset': 1.0
                          }
                          
        ROC_param_dict = {'nPoints': nPoints,\
                          'hmmgp_param_range': -np.logspace(-0.7, 1.3, nPoints), \
                          'progress_param_range': -np.logspace(-4., 1.5, nPoints)+1.0,\
                          'svm_param_range': np.logspace(-0.8, 2.5, nPoints),\
                          'bpsvm_param_range': np.logspace(-2, 0, nPoints),\
                          'hmmosvm_param_range': np.logspace(-3.0, 0.5, nPoints),\
                          'fixed_param_range': np.linspace(1.5, -0.1, nPoints),\
                          'change_param_range': np.logspace(-2., 5.0, nPoints)*-1.0,\
                          'osvm_param_range': np.logspace(-4., 0.5, nPoints),\
                          'cssvm_param_range': np.logspace(0.0, 2.0, nPoints) }

        AD_param_dict = {'svm_w_positive': 1.0, 'sgd_w_positive': 1.0}
                          
    elif dim == 2:

        ## handFeatures = ['unimodal_ftForce', \
        ##                 'crossmodal_landmarkEEDist']
        handFeatures = ['unimodal_ftForceZ', 'crossmodal_landmarkEEDist']
        ## HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 1.5, 'scale': 19.11,\
        ## HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 1.0, 'scale': 12.33,\
        HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 1.0, 'scale': 8.33,\
                          'add_logp_d': False}
        ## HMM_param_dict = {'renew': HMM_renew, 'nState': 25, 'cov': 6.0, 'scale': 3.0}
        SVM_param_dict = {'renew': CF_renew, 'w_negative': 5.0, 'gamma': 2.049, 'cost': 1.75,\
                          'hmmosvm_nu': 0.01,\
                          'osvm_nu': 0.01,\
                          'nugget': 10.0, 'theta0': 1.0,\
                          }

        ROC_param_dict = {'nPoints': nPoints,\
                          'hmmgp_param_range':np.logspace(1.1, 1.6, nPoints)*-1.0, \
                          'progress_param_range': -np.logspace(0.0, 1.2, nPoints),\
                          'svm_param_range': np.logspace(-2.5, 0.7, nPoints),\
                          'hmmosvm_param_range': np.logspace(-3.0, 0., nPoints),\
                          'fixed_param_range': np.linspace(-1.5, 1.5, nPoints),\
                          'change_param_range': np.logspace(0.0, 2.0, nPoints)*-1.0,\
                          'osvm_param_range': np.logspace(-6., 1., nPoints),\
                          'bpsvm_param_range': np.logspace(-2, 0, nPoints),\
                          'cssvm_param_range': np.logspace(0.0, 2.0, nPoints) }

        AD_param_dict = {'svm_w_positive': 1.0, 'sgd_w_positive': 1.0}

        
    rawFeatures = ['relativePose_target_EE', \
                   'wristAudio', \
                   'ft',\
                   'relativePose_landmark_EE']

    isolationFeatures = ['unimodal_kinEEChange',\
                         'unimodal_audioWristRMS', \
                         'unimodal_audioWristFrontRMS', \
                         'unimodal_audioWristAzimuth',                         
                         'unimodal_kinJntEff', \
                         'unimodal_ftForceX', \
                         'unimodal_ftForceY', \
                         'unimodal_ftForceZ', \
                         'crossmodal_landmarkEEDist', \
                         'crossmodal_landmarkEEAng',\
                         'unimodal_fabricForce',\
                         'unimodal_landmarkDist']
                   

    modality_list   = ['ft' ,'kinematics', 'audioWrist', 'vision_landmark']
    raw_data_path  = os.path.expanduser('~')+'/hrl_file_server/dpark_data/anomaly/AURO2016/'

    AE_param_dict  = {'renew': False, 'switch': False, 'time_window': 4, \
                      'layer_sizes':[64,dim], 'learning_rate':1e-6, 'learning_rate_decay':1e-6, \
                      'momentum':1e-6, 'dampening':1e-6, 'lambda_reg':1e-6, \
                      'max_iteration':30000, 'min_loss':0.1, 'cuda':True, \
                      'filter':True, 'filterDim':4,\
                      'add_option': None, 'rawFeatures': rawFeatures,\
                      'add_noise_option': [], 'preTrainModel': None}                      

    data_param_dict= {'renew': data_renew, 'rf_center': rf_center, 'local_range': local_range,\
                      'downSampleSize': 140, 'cut_data': None, \
                      'nNormalFold':2, 'nAbnormalFold':2,\
                      'handFeatures': handFeatures, 'lowVarDataRemv': False,\
                      'isolationFeatures': isolationFeatures,\
                      'handFeatures_noise': True, 'max_time': 7.0}

    save_data_path = os.path.expanduser('~')+\
      '/hrl_file_server/dpark_data/anomaly/AURO2016/'+task+'_data/'+\
      str(data_param_dict['downSampleSize'])+'_'+str(dim)

    param_dict = {'data_param': data_param_dict, 'AE': AE_param_dict, 'HMM': HMM_param_dict, \
                  'SVM': SVM_param_dict, 'ROC': ROC_param_dict, 'AD': AD_param_dict}

    return raw_data_path, save_data_path, param_dict


def getScooping(task, data_renew, HMM_renew, CF_renew, rf_center='kinEEPos', local_range=10.0, \
                pre_train=False,\
                ae_swtch=False, dim=4, nPoints=None):

    if nPoints is None: nPoints = 20  

    if dim == 4:
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
        
    elif dim == 3:
        handFeatures = ['unimodal_ftForce',\
                        'crossmodal_targetEEDist', \
                        'crossmodal_targetEEAng']
        HMM_param_dict = {'renew': HMM_renew, 'nState': 20, 'cov': 3.566, 'scale': 3.0,\
                          'add_logp_d': False}
        SVM_param_dict = {'renew': CF_renew, 'w_negative': 0.825, 'gamma': 3.16, 'cost': 4.0,\
                          'hmmosvm_nu': 0.00316,\
                          'hmmsvm_diag_w_negative': 0.85, 'hmmsvm_diag_cost': 12.5, \
                          'hmmsvm_diag_gamma': 0.01}

        ROC_param_dict = {'nPoints': nPoints,\
                          'progress_param_range':np.linspace(-0.8, -5., nPoints), \
                          'svm_param_range': np.logspace(-2.5, 0, nPoints),\
                          'fixed_param_range': np.linspace(0.0, -1.5, nPoints),\
                          'cssvm_param_range': np.logspace(-4.0, 2.0, nPoints),\
                          ## 'svm_param_range': np.logspace(-4, 1.2, nPoints),\
                          'hmmosvm_param_range': np.logspace(-4.0, 1.0, nPoints),\
                          'osvm_param_range': np.linspace(0.1, 2.0, nPoints),\
                          'sgd_param_range': np.logspace(-4, 1.2, nPoints)}        
        AD_param_dict = {'svm_w_positive': 0.1, 'sgd_w_positive': 1.0}
        
    elif dim == 2:
        handFeatures = ['unimodal_ftForce',\
                        'crossmodal_targetEEDist' ]
        HMM_param_dict = {'renew': HMM_renew, 'nState': 20, 'cov': 1.4, 'scale': 3.0,\
                          'add_logp_d': False}
        SVM_param_dict = {'renew': CF_renew, 'w_negative': 3.5, 'gamma': 0.0147, 'cost': 3.0,\
                          'hmmosvm_nu': 0.00316}

        ROC_param_dict = {'nPoints': nPoints,\
                          'progress_param_range':np.linspace(-0.8, -8., nPoints), \
                          'svm_param_range': np.logspace(-1.5, 1, nPoints),\
                          'fixed_param_range': np.linspace(0.2, -3.0, nPoints),\
                          'cssvm_param_range': np.logspace(-4.0, 2.0, nPoints),\
                          ## 'svm_param_range': np.logspace(-4, 1.2, nPoints),\
                          'hmmosvm_param_range': np.logspace(-4.0, 1.5, nPoints),\
                          'osvm_param_range': np.linspace(0.1, 2.0, nPoints),\
                          'sgd_param_range': np.logspace(-4, 1.2, nPoints)}        
        AD_param_dict = {'svm_w_positive': 1.0, 'sgd_w_positive': 1.0}
        
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

