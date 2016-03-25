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

# system
import os, sys
import numpy as np
import hrl_lib.util as ut

from sklearn.grid_search import ParameterGrid
from sklearn.cross_validation import KFold
import time

from hrl_anomaly_detection.hmm import learning_hmm as hmm
from hrl_anomaly_detection import data_manager as dm

# AWS
from hrl_anomaly_detection.aws.cloud_search import CloudSearch

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

class CloudSearchForClassifier(CloudSearch):
    def __init__(self, path_json, path_key, clust_name, user_name):
        CloudSearch.__init__(self, path_json, path_key, clust_name, user_name)

    #run data in cloud.
	#each node grabs file from their local path and runs the model
	#requires grab_data to be implemented correctly
	#n_inst is to create a fold. the way it generates fold can be changed
    def run_with_local_data(self, params, processed_data_path, task_name, nFiles, param_dict):
        
        all_param = list(ParameterGrid(params))

        for param_idx, param in enumerate(all_param):
            task = self.lb_view.apply(cross_validate_local, param_idx, nFiles, \
                                      processed_data_path, task_name, \
                                      default_params=param_dict, custom_params=param)
            self.all_tasks.append(task)
        return self.all_tasks


def cross_validate_local(param_idx, nFiles, processed_data_path, task_name, default_params, custom_params, \
                         n_jobs=2):
    '''
    
    '''
    from joblib import Parallel, delayed
    
    ## Default Parameters
    # data
    data_dict = default_params['data_param']
    # HMM
    HMM_dict = default_params['HMM']
    # ROC
    ROC_dict = default_params['ROC']
    #------------------------------------------

    ## Custom parameters
    method = custom_params['method']
    
    #------------------------------------------
    ROC_data = {}
    ROC_data[method] = {}
    ROC_data[method]['tp_l'] = [ [] for i in xrange(ROC_dict['nPoints']) ]
    ROC_data[method]['fp_l'] = [ [] for i in xrange(ROC_dict['nPoints']) ]
    ROC_data[method]['tn_l'] = [ [] for i in xrange(ROC_dict['nPoints']) ]
    ROC_data[method]['fn_l'] = [ [] for i in xrange(ROC_dict['nPoints']) ]
    ROC_data[method]['delay_l'] = [ [] for i in xrange(ROC_dict['nPoints']) ]
    
    ## for file_idx in xrange(nFiles):
    ##     run_classifier( os.path.join(processed_data_path, 'hmm_'+task_name+'_'+str(file_idx)+'.pkl'), \
    ##                     method, HMM_dict, ROC_dict, custom_params)
    ##     print "running fine"
    ##     return 0, -1, custom_params
    
    r = Parallel(n_jobs=n_jobs)(delayed(run_classifier)( os.path.join(processed_data_path, 'hmm_'+task_name+'_'+str(file_idx)+'.pkl'), method, HMM_dict, ROC_dict, custom_params) for file_idx in xrange(nFiles))
    l_ROC_data, l_param_idx = zip(*r)

    for i in xrange(len(l_ROC_data)):
        if l_param_idx[i]==-1: return ROC_data, -1, custom_params

        for j in xrange(ROC_dict['nPoints']):
            ROC_data[method]['tp_l'][j] += l_ROC_data[i]['tp_l'][j]
            ROC_data[method]['fp_l'][j] += l_ROC_data[i]['fp_l'][j]
            ROC_data[method]['fn_l'][j] += l_ROC_data[i]['fn_l'][j]
            ROC_data[method]['tn_l'][j] += l_ROC_data[i]['tn_l'][j]
            ROC_data[method]['delay_l'][j] += l_ROC_data[i]['delay_l'][j]
       
    return ROC_data, param_idx, custom_params

def run_classifier(modeling_pkl, method, HMM_dict, ROC_dict, params):

    import os
    ## from hrl_anomaly_detection.classifiers import run_classifier_aws as rca
    from hrl_anomaly_detection.classifiers import classifier as cb
    import hrl_lib.util as ut
    import numpy as np
    from sklearn import preprocessing

    print modeling_pkl
    # train a classifier and evaluate it using test data.
    d            = ut.load_pickle(modeling_pkl)
    nEmissionDim = d['nEmissionDim']
    A            = d['A']
    B            = d['B']
    pi           = d['pi']
    F            = d['F']
    nState       = d['nState']        
    ## startIdx = d['startIdx']
    ll_classifier_train_X   = d['ll_classifier_train_X']
    ll_classifier_train_Y   = d['ll_classifier_train_Y']         
    ll_classifier_train_idx = d['ll_classifier_train_idx']
    ll_classifier_test_X    = d['ll_classifier_test_X']  
    ll_classifier_test_Y    = d['ll_classifier_test_Y']
    ll_classifier_test_idx  = d['ll_classifier_test_idx']
    nLength      = d['nLength']
    nPoints      = ROC_dict['nPoints']

    # flatten the data
    X_train_org = []
    Y_train_org = []
    idx_train_org = []
    for i in xrange(len(ll_classifier_train_X)):
        for j in xrange(len(ll_classifier_train_X[i])):
            X_train_org.append(ll_classifier_train_X[i][j])
            Y_train_org.append(ll_classifier_train_Y[i][j])
            idx_train_org.append(ll_classifier_train_idx[i][j])

    # data preparation
    if 'svm' in method:
        scaler = preprocessing.StandardScaler()
        ## scaler = preprocessing.scale()
        X_scaled = scaler.fit_transform(X_train_org)
    else:
        X_scaled = X_train_org
    ## print method, " : Before classification : ", np.shape(X_scaled), np.shape(Y_train_org)

    tp_ll = [ [] for i in xrange(ROC_dict['nPoints']) ]
    fp_ll = [ [] for i in xrange(ROC_dict['nPoints']) ]
    fn_ll = [ [] for i in xrange(ROC_dict['nPoints']) ]
    tn_ll = [ [] for i in xrange(ROC_dict['nPoints']) ]
    delay_ll = [ [] for i in xrange(ROC_dict['nPoints']) ]

    # classifier
    dtc = cb.classifier( method=method, nPosteriors=HMM_dict['nState'], nLength=nLength )        
    for j in xrange(ROC_dict['nPoints']):
        if method == 'svm':
            weights = ROC_dict['svm_param_range']
            dtc.set_params( class_weight=weights[j] )
        elif method == 'cssvm_standard':
            weights = np.logspace(-2, 0.1, nPoints)
            dtc.set_params( class_weight=weights[j] )
        elif method == 'cssvm':
            weights = ROC_dict['cssvm_param_range']
            dtc.set_params( class_weight=weights[j] )
        elif method == 'progress_time_cluster':
            thresholds = ROC_dict['progress_param_range']
            dtc.set_params( ths_mult = thresholds[j] )
        elif method == 'fixed':
            thresholds = ROC_dict['fixed_param_range']
            dtc.set_params( ths_mult = thresholds[j] )
        else:
            print "Not available method"
            return "Not available method", -1, params

        dtc.set_params(**params)
        ret = dtc.fit(X_scaled, Y_train_org, idx_train_org)
        if ret is False: return 'fit failed', -1

        # evaluate the classifier
        tp_l = []
        fp_l = []
        tn_l = []
        fn_l = []
        delay_l = []
        delay_idx = 0

        for ii in xrange(len(ll_classifier_test_X)):
            if len(ll_classifier_test_Y[ii])==0: continue

            for jj in xrange(len(ll_classifier_test_X[ii])):
                if 'svm' in method:
                    X = scaler.transform([ll_classifier_test_X[ii][jj]])
                elif method == 'progress_time_cluster' or method == 'fixed':
                    X = ll_classifier_test_X[ii][jj]

                est_y    = dtc.predict(X, y=ll_classifier_test_Y[ii][jj:jj+1])
                if type(est_y) == list: est_y = est_y[0]
                if type(est_y) == list: est_y = est_y[0]
                ## X = X[0]

                if est_y > 0.0:
                    delay_idx = ll_classifier_test_idx[ii][jj]
                    ## print "Break ", ii, " ", jj, " in ", est_y, " = ", ll_classifier_test_Y[ii][jj]
                    break        

            if ll_classifier_test_Y[ii][0] > 0.0:
                if est_y > 0.0:
                    tp_l.append(1)
                    delay_l.append(delay_idx)
                else: fn_l.append(1)
            elif ll_classifier_test_Y[ii][0] <= 0.0:
                if est_y > 0.0: fp_l.append(1)
                else: tn_l.append(1)

        tp_ll[j] += tp_l
        fp_ll[j] += fp_l
        fn_ll[j] += fn_l
        tn_ll[j] += tn_l
        delay_ll[j] += delay_l

    ## return tp_ll, fp_ll, fn_ll, tn_ll, delay_ll

    if tp_ll is None or fp_ll is None or fn_ll is None or tn_ll is None:
        return tp_ll, -1

    ## for j in xrange(ROC_dict['nPoints']):
    ##     ROC_data[method]['tp_l'][j] += tp_ll[j]
    ##     ROC_data[method]['fp_l'][j] += fp_ll[j]
    ##     ROC_data[method]['fn_l'][j] += fn_ll[j]
    ##     ROC_data[method]['tn_l'][j] += tn_ll[j]
    ##     ROC_data[method]['delay_l'][j] += delay_ll[j]
    ROC_data = {}
    ROC_data['tp_l'] = tp_ll
    ROC_data['fp_l'] = fp_ll
    ROC_data['fn_l'] = fn_ll
    ROC_data['tn_l'] = tn_ll
    ROC_data['delay_l'] = delay_ll

    return ROC_data, param_idx





def cross_validate_cpu(processed_data_path, task_name, nFiles, param_dict, parameters):

    ## Parameters
    # data
    data_dict  = param_dict['data_param']
    data_renew = data_dict['renew']
    # AE
    AE_dict     = param_dict['AE']
    autoEncoder = AE_dict['switch']
    # HMM
    HMM_dict = param_dict['HMM']
    nState   = HMM_dict['nState']
    cov      = HMM_dict['cov']
    # Classifier

    # ROC
    ROC_dict = param_dict['ROC']
    method   = ROC_dict['methods'][0]
    
    #------------------------------------------

    # sample x dim x length
    param_list = list(ParameterGrid(parameters))
    score_list = []
    ROC_data = {}
    verbose = False
    
    for param in param_list:

        ROC_data[method] = {}
        ROC_data[method]['complete'] = False 
        ROC_data[method]['tp_l'] = []
        ROC_data[method]['fp_l'] = []
        ROC_data[method]['tn_l'] = []
        ROC_data[method]['fn_l'] = []
        ROC_data[method]['delay_l'] = []

        for i in xrange(ROC_dict['nPoints']):
            ROC_data[method]['tp_l'].append([])
            ROC_data[method]['fp_l'].append([])
            ROC_data[method]['tn_l'].append([])
            ROC_data[method]['fn_l'].append([])
            ROC_data[method]['delay_l'].append([])

        for idx in xrange(nFiles):

            if verbose: print idx, " : training classifier and evaluate testing data"
            modeling_pkl = os.path.join(processed_data_path, 'hmm_'+task_name+'_'+str(idx)+'.pkl')
            tp_ll, fp_ll, fn_ll, tn_ll, delay_ll = run_classifier(modeling_pkl, method, HMM_dict, ROC_dict)

            for j in xrange(ROC_dict['nPoints']):
                ROC_data[method]['tp_l'][j] += tp_ll[j]
                ROC_data[method]['fp_l'][j] += fp_ll[j]
                ROC_data[method]['fn_l'][j] += fn_ll[j]
                ROC_data[method]['tn_l'][j] += tn_ll[j]
                ROC_data[method]['delay_l'][j] += delay_ll[j]
        
        tp_ll = ROC_data[method]['tp_l']
        fp_ll = ROC_data[method]['fp_l']
        tn_ll = ROC_data[method]['tn_l']
        fn_ll = ROC_data[method]['fn_l']
        delay_ll = ROC_data[method]['delay_l']

        tpr_l = []
        fpr_l = []

        for i in xrange(ROC_dict['nPoints']):
            tpr_l.append( float(np.sum(tp_ll[i]))/float(np.sum(tp_ll[i])+np.sum(fn_ll[i]))*100.0 )
            fpr_l.append( float(np.sum(fp_ll[i]))/float(np.sum(fp_ll[i])+np.sum(tn_ll[i]))*100.0 )

        print "--------------------------------"
        print method
        print tpr_l
        print fpr_l
        print "--------------------------------"

        # get AUC
        score_list.append( getAUC(fpr_l, tpr_l) )
        
    for i, param in enumerate(param_list):
        print("%0.3f for %r" % (score_list[i], param))


def getAUC(fpr_l, tpr_l):
    area = 0.0
    for i in range(len(fpr_l)-1):        
        area += (fpr_l[i+1]-fpr_l[i])*(tpr_l[i]+tpr_l[i+1])*0.5
    return area


if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()

    p.add_option('--user', action='store', dest='user', type='string', default='dpark',
                 help='type the user name')
    p.add_option('--task', action='store', dest='task', type='string', default='pushing',
                 help='type the desired task name')

    opt, args = p.parse_args()

    rf_center     = 'kinEEPos'        
    local_range    = 10.0    

    if opt.task == 'scooping':
        subjects = ['Wonyoung', 'Tom', 'lin', 'Ashwin', 'Song', 'Henry2'] #'Henry', 
        task     = opt.task    
        feature_list = ['unimodal_audioWristRMS',\
                        'unimodal_ftForce',\
                        'crossmodal_targetEEDist', \
                        'crossmodal_targetEEAng']
        modality_list = ['kinematics', 'audioWrist', 'ft', 'vision_artag', \
                         'vision_change', 'pps']
        save_data_path = '/home/'+opt.user+'/hrl_file_server/dpark_data/anomaly/RSS2016/'+task+'_data'
        downSampleSize = 200



        data_param_dict= {'renew': False, 'rf_center': rf_center, 'local_range': local_range,\
                          'downSampleSize': 200, 'cut_data': [0,130], 'nNormalFold':4, 'nAbnormalFold':4,\
                          'feature_list': feature_list, 'nAugment': 0, 'lowVarDataRemv': False}
        AE_param_dict  = {'renew': False, 'switch': False, 'time_window': 4, 'filter': True, \
                          'layer_sizes':[64,32,16], 'learning_rate':1e-6, 'learning_rate_decay':1e-6, \
                          'momentum':1e-6, 'dampening':1e-6, 'lambda_reg':1e-6, \
                          'max_iteration':30000, 'min_loss':0.1, 'cuda':True, 'filter':True, 'filterDim':4}
        HMM_param_dict = {'renew': False, 'nState': 20, 'cov': 5.0, 'scale': 4.0}
        SVM_param_dict = {'renew': False,}

        nPoints        = 10
        ROC_param_dict = {'methods': ['progress_time_cluster', 'svm','fixed'],\
                          'update_list': [],\
                          'nPoints': nPoints,\
                          'progress_param_range':-np.linspace(0., 10.0, nPoints), \
                          'svm_param_range': np.logspace(-4, 1.2, nPoints),\
                          'fixed_param_range': -np.logspace(0.0, 0.9, nPoints)+1.2,\
                          'cssvm_param_range': np.logspace(0.0, 2.0, nPoints) }
        param_dict = {'data_param': data_param_dict, 'AE': AE_param_dict, 'HMM': HMM_param_dict, \
                      'SVM': SVM_param_dict, 'ROC': ROC_param_dict}

        nFiles = 16
        ## parameters = {'method': ['svm'], 'svm_type': [0], 'kernel_type': [2], \
        ##               'degree': [3], 'gamma': np.linspace(0.01, 0.5, 4).tolist(), \
        ##               'w_negative': np.arange(1.0, 10.0) }
        parameters = {'method': ['svm'], 'svm_type': [0], 'kernel_type': [2], \
                      'cost': [1.0, 2.0, 4.0, 6.0],\
                      'gamma': np.linspace(0.01, 0.3, 4).tolist(), \
                      'w_negative': np.arange(1.0, 7.0) }

    #---------------------------------------------------------------------------
    elif opt.task == 'feeding':
        
        subjects = ['Tom', 'lin', 'Ashwin', 'Song'] #'Wonyoung']
        task     = opt.task 
        feature_list = ['unimodal_audioWristRMS', 'unimodal_ftForce', 'crossmodal_artagEEDist', \
                        'crossmodal_artagEEAng'] #'unimodal_audioPower'
        modality_list   = ['ft'] #'kinematics', 'audioWrist', , 'vision_artag'

        save_data_path = '/home/'+opt.user+'/hrl_file_server/dpark_data/anomaly/RSS2016/'+task+'_data'
        downSampleSize = 200

        data_param_dict= {'renew': False, 'rf_center': rf_center, 'local_range': local_range,\
                          'downSampleSize': downSampleSize, 'cut_data': [0,170], \
                          'nNormalFold':4, 'nAbnormalFold':4,\
                          'feature_list': feature_list, 'nAugment': 0, 'lowVarDataRemv': False}
        AE_param_dict  = {'renew': False, 'switch': False, 'time_window': 4, 'filter': True, \
                          'layer_sizes':[64,32,16], 'learning_rate':1e-6, 'learning_rate_decay':1e-6, \
                          'momentum':1e-6, 'dampening':1e-6, 'lambda_reg':1e-6, \
                          'max_iteration':30000, 'min_loss':0.1, 'cuda':True, 'filter':True, 'filterDim':4}
        HMM_param_dict = {'renew': False, 'nState': 25, 'cov': 5.0, 'scale': 4.0}
        SVM_param_dict = {'renew': False,}
        
        nPoints        = 10
        ROC_param_dict = {'methods': ['progress_time_cluster', 'svm','fixed'],\
                          'update_list': [],\
                          'nPoints': nPoints,\
                          'progress_param_range':-np.linspace(0., 10.0, nPoints), \
                          'svm_param_range': np.logspace(-4, 1.2, nPoints),\
                          'fixed_param_range': np.linspace(1.0, -3.0, nPoints),\
                          'cssvm_param_range': np.logspace(0.0, 2.0, nPoints) }
        param_dict = {'data_param': data_param_dict, 'AE': AE_param_dict, 'HMM': HMM_param_dict, \
                      'SVM': SVM_param_dict, 'ROC': ROC_param_dict}

        nFiles = 16
        parameters = {'method': ['svm'], 'svm_type': [0], 'kernel_type': [2], \
                      'cost': [1.0, 2.0, 4.0, 6.0],\
                      'gamma': np.linspace(0.001, 0.015, 4).tolist(), \
                      'w_negative': np.linspace(0.01, 1.3, 5) }

    #---------------------------------------------------------------------------           
    elif opt.task == 'pushing':
    
        subjects = ['gatsbii']
        task     = opt.task 
        feature_list = ['relativePose_artag_EE', \
                        'relativePose_artag_artag', \
                        'wristAudio', \
                        'ft', \
                        ]

        modality_list  = ['kinematics', 'audio', 'ft']
        save_data_path = '/home/'+opt.user+'/hrl_file_server/dpark_data/anomaly/RSS2016/'+task+'_data/AE'
        downSampleSize = 200      

        data_param_dict= {'renew': False, 'rf_center': rf_center, 'local_range': local_range,\
                          'downSampleSize': downSampleSize, 'cut_data': [0,200], \
                          'nNormalFold':3, 'nAbnormalFold':3,\
                          'feature_list': feature_list, 'nAugment': 1, 'lowVarDataRemv': False }
        AE_param_dict  = {'renew': False, 'switch': True, 'time_window': 4, 'filter': True, \
                          'layer_sizes':[64,32,16], 'learning_rate':1e-6, 'learning_rate_decay':1e-6, \
                          'momentum':1e-6, 'dampening':1e-6, 'lambda_reg':1e-6, \
                          'max_iteration':30000, 'min_loss':0.1, 'cuda':True, 'filter':True, 'filterDim':4, \
                          'add_option': 'featureToBottleneck', 'add_feature': feature_list}
        HMM_param_dict = {'renew': False, 'nState': 25, 'cov': 4.0, 'scale': 5.0}
        SVM_param_dict = {'renew': False,}

        #temp
        nPoints        = 10
        ROC_param_dict = {'methods': ['cssvm'],\
                          'nPoints': nPoints,\
                          'progress_param_range':np.linspace(-1., -10., nPoints), \
                          'svm_param_range': np.logspace(-4, 1.2, nPoints),\
                          'fixed_param_range': np.linspace(1.0, -3.0, nPoints),\
                          'cssvm_param_range': np.logspace(0.0, 2.0, nPoints) }        
        param_dict = {'data_param': data_param_dict, 'AE': AE_param_dict, 'HMM': HMM_param_dict, \
                      'SVM': SVM_param_dict, 'ROC': ROC_param_dict}

        nFiles = 9
        parameters = {'method': ['svm'], 'svm_type': [0], 'kernel_type': [2], \
                      'gamma': np.linspace(0.01, 0.5, 4).tolist(), \
                      'w_negative': np.arange(1.0, 10.0) }

    else:
        print "Selected task name is not available."
        sys.exit()

    #--------------------------------------------------------------------------------------

    ## parameters = {'method': ['svm'], 'svm_type': [1], 'svn_kernel_type': [1,2], 'svn_degree': [2], \
    ##               'svm_w_negative': [1.0]}

    # Get combined results
    max_param_idx = len( list(ParameterGrid(parameters)) )
    method = parameters['method'][0]
    score_list = []
    print "max_param_idx = ", max_param_idx

    ##################################################################################################
    # cpu version
    if True:
        ## cross_validate_cpu(save_data_path, task, nFiles, param_dict, parameters)
        ## save_data_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/RSS2016/'+task+'_data'

        ## nFiles = 2
        ## parameters = {'method': ['svm'], 'svm_type': [0], 'svm_kernel_type': [1,2], \
        ##               'svm_degree': [3], 'svm_nu': [0.5], 'svm_w_negative': [7.0]}

        if os.path.isfile('./temp.pkl') is False:
            results = []
            for param_idx, param in enumerate( list(ParameterGrid(parameters)) ):
                ret_ROC_data, ret_param_idx, ret_params = cross_validate_local(param_idx, nFiles, \
                                                                               save_data_path, \
                                                                               task, param_dict, param, \
                                                                               n_jobs=-1)
                results.append([ret_ROC_data, ret_param_idx, ret_params])

            ut.save_pickle(results, './temp.pkl')
        else:
            results = ut.load_pickle('./temp.pkl')

        ## plt.figure()
        for result in results:
            ret_ROC_data = result[0]
            ret_param_idx = result[1]
            ret_params = result[2]
            
            if ret_param_idx == -1:
                score_list.append([0, ret_params])
                continue
            tp_ll = ret_ROC_data[method]['tp_l']
            fp_ll = ret_ROC_data[method]['fp_l']
            tn_ll = ret_ROC_data[method]['tn_l']
            fn_ll = ret_ROC_data[method]['fn_l']
            delay_ll = ret_ROC_data[method]['delay_l']

            tpr_l = []
            fpr_l = []
            try:
                for j in xrange(nPoints):
                    tpr_l.append( float(np.sum(tp_ll[j]))/float(np.sum(tp_ll[j])+np.sum(fn_ll[j]))*100.0 )
                    fpr_l.append( float(np.sum(fp_ll[j]))/float(np.sum(fp_ll[j])+np.sum(tn_ll[j]))*100.0 )
            except:
                print "failed to get TPR and FPR"
                break
            print fpr_l, tpr_l

            # get AUC
            ## score_list.append( [getAUC(fpr_l, tpr_l), ret_params] )

            ## plt.plot(fpr_l, tpr_l, '-')            
            # get max tp in fpr (0~20)
            max_tp = 0
            for i, fp in enumerate(fpr_l):
                if fp < 20.0:
                    if tpr_l[i] > max_tp: max_tp = tpr_l[i]
            score_list.append( [max_tp, ret_params] )

        ## plt.show()
        # Get sorted results
        from operator import itemgetter
        score_list.sort(key=itemgetter(0), reverse=False)

        for i in xrange(len(score_list)):
            print("%0.3f for %r" % (score_list[i][0], score_list[i][1]))
                                                                           
    else:

        if os.path.isfile('./temp.pkl') is False:

            cloud = CloudSearchForClassifier(os.path.expanduser('~')+\
                                             '/.starcluster/ipcluster/SecurityGroup:@sc-testdpark-us-east-1.json', \
                                             os.path.expanduser('~')+'/.ssh/HRL_ANOMALY.pem', 'testdpark', 'ubuntu')
            cloud.run_with_local_data(parameters, save_data_path, task, nFiles, param_dict )


            # wait until finishing parameter search
            while cloud.get_num_all_tasks() != cloud.get_num_tasks_completed():
                print "Processing tasks, ", cloud.get_num_tasks_completed(), ' / ', cloud.get_num_all_tasks()
                time.sleep(5)

            results = cloud.get_completed_results()
            print "===================================="
            print "Result"
            print "===================================="
            for result in results:        
                print result
            print "===================================="
            import hrl_lib.util as ut
            ut.save_pickle(results, './temp.pkl')            
        else:
            import hrl_lib.util as ut
            results = ut.load_pickle('./temp.pkl')

        
        for i in xrange(max_param_idx):

            ROC_data = {}
            ROC_data[method] = {}
            ROC_data[method]['tp_l']    = [ [] for j in xrange(nPoints) ]
            ROC_data[method]['fp_l']    = [ [] for j in xrange(nPoints) ]
            ROC_data[method]['tn_l']    = [ [] for j in xrange(nPoints) ]
            ROC_data[method]['fn_l']    = [ [] for j in xrange(nPoints) ]
            ROC_data[method]['delay_l'] = [ [] for j in xrange(nPoints) ]

            param = None
            for result in results:
                if result[1] == i:
                    param    = result[2]

                    for j in xrange(nPoints):
                        ROC_data[method]['tp_l'][j]    += result[0][method]['tp_l'][j]
                        ROC_data[method]['fp_l'][j]    += result[0][method]['fp_l'][j]
                        ROC_data[method]['fn_l'][j]    += result[0][method]['fn_l'][j]
                        ROC_data[method]['tn_l'][j]    += result[0][method]['tn_l'][j]
                        ROC_data[method]['delay_l'][j] += result[0][method]['delay_l'][j]


            tp_ll = ROC_data[method]['tp_l']
            fp_ll = ROC_data[method]['fp_l']
            tn_ll = ROC_data[method]['tn_l']
            fn_ll = ROC_data[method]['fn_l']
            delay_ll = ROC_data[method]['delay_l']

            tpr_l = []
            fpr_l = []

            try:
                for j in xrange(nPoints):
                    tpr_l.append( float(np.sum(tp_ll[j]))/float(np.sum(tp_ll[j])+np.sum(fn_ll[j]))*100.0 )
                    fpr_l.append( float(np.sum(fp_ll[j]))/float(np.sum(fp_ll[j])+np.sum(tn_ll[j]))*100.0 )
            except:
                print "failed to get TPR and FPR"
                ## print tp_ll, fn_ll
                ## cloud.stop()
                #cloud.flush()
                break

            # get AUC
            score_list.append( [getAUC(fpr_l, tpr_l), param] )

        for i in xrange(len(score_list)):
            print("%0.3f for %r" % (score_list[i][0], score_list[i][1]))
            
        cloud.stop()
        cloud.flush()
        print "Finished"
