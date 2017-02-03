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
from hrl_anomaly_detection import util as util
from hrl_anomaly_detection import data_manager as dm
import hrl_anomaly_detection.isolator.isolation_util as iutil
from hrl_execution_monitor import util as autil
from hrl_execution_monitor import preprocess as pp
from hrl_execution_monitor import anomaly_isolator_preprocess as aip

# Private learners
from hrl_anomaly_detection.hmm import learning_hmm as hmm
from sklearn import preprocessing

from joblib import Parallel, delayed

random.seed(3334)
np.random.seed(3334)



def train_isolator_modules(save_data_path, n_labels, verbose=False):
    '''
    Train networks
    '''
    d = ut.load_pickle(os.path.join(save_data_path, 'isol_data.pkl'))
    nFold = len(d.keys())
    del d

    train_rfc_with_signal(save_data_path, n_labels, nFold)
    return


def train_rfc_with_signal(save_data_path, n_labels, nFold):

    scores= []
    y_test_list = []
    y_pred_list = []
    for idx in xrange(nFold):

        # Loading data
        train_data, test_data = autil.load_data(idx, os.path.join(save_data_path,'keras'), viz=False)      
        x_train_sig = train_data[0]
        y_train     = train_data[2]
        x_test_sig = test_data[0]
        y_test     = test_data[2]

        # Scaling
        from sklearn import preprocessing
        scaler      = preprocessing.StandardScaler()
        x_train_sig = scaler.fit_transform(x_train_sig)
        x_test_sig  = scaler.transform(x_test_sig)

        fileName = os.path.join(save_data_path, 'scr_'+str(idx))
        print fileName
        import pickle
        with open(fileName, 'wb') as f:
            pickle.dump(scaler, f)

        # get classifier
        ## from sklearn.svm import SVC
        ## clf = SVC(C=1.0, kernel='rbf') #, decision_function_shape='ovo')
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=400, n_jobs=-1)
        clf.fit(x_train_sig, np.argmax(y_train, axis=1))
        score = clf.score(x_test_sig, np.argmax(y_test, axis=1))
        scores.append(score)
        print score
        
        fileName = os.path.join(save_data_path, 'clf_'+str(idx))
        print fileName
        import pickle
        with open(fileName, 'wb') as f:
            pickle.dump(clf, f)

    print 
    print np.mean(scores), np.std(scores)
    return



def get_isolator_modules(save_data_path, task_name, param_dict, fold_idx=0, \
                         nDetector=2, verbose=False):

    # load param
    feature_pkl = os.path.join(save_data_path, 'feature_extraction_kinEEPos_'+\
                            str(10.0) )

    d = ut.load_pickle(feature_pkl) 
    # TODO: need to get normal train data with specific feature names
    feature_idx_list = []
    for feature in param_dict['data_param']['staticFeatures']:
        feature_idx_list.append(param_dict['data_param']['isolationFeatures'].index(feature))

    d['param_dict']['feature_names'] = np.array(d['param_dict']['feature_names'])[feature_idx_list]
    d['param_dict']['feature_min'] = np.array(d['param_dict']['feature_min'])[feature_idx_list]
    d['param_dict']['feature_max'] = np.array(d['param_dict']['feature_max'])[feature_idx_list]   

    m_param_dict = {}
    m_param_dict['feature_params'] = d['param_dict']
    m_param_dict['successData']    = d['successData'][feature_idx_list]
    m_param_dict['failureData']    = d['failureData'][feature_idx_list]
                         

    # load param
    hmm_list = []
    for i in xrange(nDetector):
        hmm_pkl = os.path.join(save_data_path, 'hmm_'+task_name+'_'+str(fold_idx)+'_c'+str(i)+'.pkl')

        d     = ut.load_pickle(hmm_pkl)
        m_gen = hmm.learning_hmm(d['nState'], d['nEmissionDim'], verbose=verbose)
        m_gen.set_hmm_object(d['A'], d['B'], d['pi'])
        hmm_list.append(m_gen)

    # get scaler
    fileName = os.path.join(save_data_path, 'scr_'+str(fold_idx))
    import pickle
    with open(fileName, 'rb') as f:
        m_scr = pickle.load(f)
    

    # get classifier
    fileName = os.path.join(save_data_path, 'clf_'+str(fold_idx))
    import pickle
    with open(fileName, 'rb') as f:
        m_clf = pickle.load(f)

    return m_param_dict, hmm_list, m_scr, m_clf



    


if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    util.initialiseOptParser(p)

    p.add_option('--get_isol_data', '--gi', action='store_true', dest='getIsolData',
                 default=False, help='Preprocess')
    p.add_option('--preprocess', '--p', action='store_true', dest='preprocessing',
                 default=False, help='Preprocess')
    p.add_option('--preprocess_extra', '--pe', action='store_true', dest='preprocessing_extra',
                 default=False, help='Preprocess extra images')
    p.add_option('--train', '--tr', action='store_true', dest='train',
                 default=False, help='Train')
    
    opt, args = p.parse_args()

    from hrl_execution_monitor.params.IROS2017_params import *
    # IROS2017
    subject_names = ['s2', 's3','s4','s5', 's6','s7','s8', 's9']
    raw_data_path, save_data_path, param_dict = getParams(opt.task, opt.bDataRenew, \
                                                          opt.bHMMRenew, opt.bCLFRenew)

    # -----------------------------------------------------------------------
    # signal only best
    save_data_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/IROS2017/'+opt.task+'_demo1'
    ## weight    = [-3.0, -4.5]
    ## param_dict['HMM']['scale'] = [2.0, 2.0]

    #c11 
    save_data_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/IROS2017/'+opt.task+'_demo2'
    ## weight    = [-3.0, -4.5]
    ## param_dict['HMM']['scale'] = [2.0, 2.0]
    # -----------------------------------------------------------------------
    # ep
    save_data_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/IROS2017/'+opt.task+'_demo5'
    weight    = [-9.0, -9.0]
    
    ## # c8
    ## save_data_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/IROS2017/'+opt.task+'_demo4'
    ## weight    = [-7.44, -12.0]
    
    # c12
    save_data_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/IROS2017/'+opt.task+'_demo'
    weight    = [-5.2., -6.2]

    # c11 5.2,5.2-65  5.2,6.2-69 maybebest 
    ## save_data_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/IROS2017/'+opt.task+'_demo3'
    ## weight    = [-5.2, -5.2]
    param_dict['HMM']['scale'] = [5.0, 11.0]

    
    #7.44,7.44-57
    #7.44,10-66
    #7.44,12-62.5
    #8,8-60
    # -----------------------------------------------------------------------


    task_name = 'feeding'
    method    = ['progress0', 'progress1'] 
    param_dict['ROC']['methods'] = ['progress0', 'progress1'] #'hmmgp'
    param_dict['HMM']['cov']   = 1.0
    single_detector=False    
    nb_classes = 12
    window_steps= 5


    if opt.getIsolData:
        aip.get_isolation_data(subject_names, task_name, raw_data_path, save_data_path,
                               param_dict, weight, single_detector=single_detector,
                               window_steps=window_steps, verbose=False)        
    elif opt.preprocessing:    
        # preprocessing data
        data_pkl = os.path.join(save_data_path, 'isol_data.pkl')
        pp.preprocess_data(data_pkl, save_data_path, img_scale=0.25, nb_classes=nb_classes,
                              img_feature_type='vgg')
        
    elif opt.preprocessing_extra:
        raw_data_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/AURO2016/raw_data/manual_label'
        pp.preprocess_images(raw_data_path, save_data_path, img_scale=0.25, nb_classes=nb_classes,
                                img_feature_type='vgg')
        
    elif opt.train:
        # train signal only
        train_isolator_modules(save_data_path, nb_classes, verbose=False)
        
    else:
        get_isolator_modules(save_data_path, task_name, param_dict, fold_idx=0, \
                             nDetector=2)
