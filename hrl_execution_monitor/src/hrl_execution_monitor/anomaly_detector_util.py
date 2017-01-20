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

random.seed(3334)
np.random.seed(3334)

# Private utils
from hrl_anomaly_detection import util as util
from hrl_anomaly_detection import data_manager as dm
from hrl_execution_monitor import util as autil

# Private learners
from hrl_anomaly_detection.hmm import learning_hmm as hmm
import hrl_anomaly_detection.classifiers.classifier as cf

from joblib import Parallel, delayed



def train_detector_modules(subject_names, task_name, raw_data_path, save_data_path,
                            param_dict, verbose=False):

    # load params (param_dict)
    data_dict  = param_dict['data_param']
    data_renew = data_dict['renew']
    # HMM
    HMM_dict   = param_dict['HMM']
    nState     = HMM_dict['nState']
    cov        = HMM_dict['cov']
    # SVM
    SVM_dict   = param_dict['SVM']
    # ROC
    ROC_dict = param_dict['ROC']
    method_list = ROC_dict['methods']

    # parameters
    startIdx    = 4
    nPoints     = ROC_dict['nPoints']

    # load data (mix) -------------------------------------------------
    d = dm.getDataSet(subject_names, task_name, raw_data_path, \
                      save_data_path,\
                      downSampleSize=data_dict['downSampleSize'],\
                      handFeatures=data_dict['isolationFeatures'], \
                      data_renew=data_renew, max_time=data_dict['max_time'],\
                      ros_bag_image=True, rndFold=True)
                      
    # split data with 80:20 ratio, 3set
    kFold_list = d['kFold_list']
    
    # Train a generative model ----------------------------------------
    # Training HMM, and getting classifier training and testing data
    feature_idx_list = []
    success_data_ad = []
    failure_data_ad = []
    nDetector = len(param_dict['data_param']['handFeatures'])
    for i in xrange(nDetector):
        
        feature_idx_list.append([])
        for feature in param_dict['data_param']['handFeatures'][i]:
            feature_idx_list[i].append(data_dict['isolationFeatures'].index(feature))

        success_data_ad.append( copy.copy(d['successData'][feature_idx_list[i]]) )
        failure_data_ad.append( copy.copy(d['failureData'][feature_idx_list[i]]) )
        HMM_dict_local = copy.deepcopy(HMM_dict)
        HMM_dict_local['scale'] = param_dict['HMM']['scale'][i]
    
        # Training HMM, and getting classifier training and testing data
        dm.saveHMMinducedFeatures(kFold_list, success_data_ad[i], failure_data_ad[i],\
                                  task_name, save_data_path,\
                                  HMM_dict_local, data_renew, startIdx, nState, cov, \
                                  success_files=d['successFiles'], failure_files=d['failureFiles'],\
                                  noise_mag=0.1, suffix=str(i),\
                                  verbose=verbose, one_class=False)

    # Train a classifier ----------------------------------------------
    roc_pkl = os.path.join(save_data_path, 'roc_'+task_name+'.pkl')

    if os.path.isfile(roc_pkl) is False or HMM_dict['renew'] or SVM_dict['renew']: ROC_data = {}
    else: ROC_data = ut.load_pickle(roc_pkl)
    ROC_data = autil.reset_roc_data(ROC_data, [method_list[0][:-1]], ROC_dict['update_list'], nPoints)

    l_data = Parallel(n_jobs=-1, verbose=10)\
      (delayed(cf.run_classifiers_boost)( idx, save_data_path, task_name, \
                                          method_list, ROC_data, \
                                          param_dict,\
                                          startIdx=startIdx, nState=nState,\
                                          save_model=True) \
      for idx in xrange(len(kFold_list)) )
    
    ROC_data = autil.update_roc_data(ROC_data, l_data, nPoints, [method_list[0][:-1]])
    ut.save_pickle(ROC_data, roc_pkl)
    
    # ROC Visualization ------------------------------------------------
    util.roc_info(ROC_data, nPoints, no_plot=True, ROC_dict=ROC_dict,
                  multi_ad=True, verbose=True, padding=True)

    # TODO
    # need to print the best weight out
    # need to save acc list

    return 


## def test_detector_modules(save_data_path, task_name):
##     return
    

def get_detector_modules(save_data_path, task_name, param_dict, detector_id, fold_idx=0, \
                         verbose=False):

    ## nDetector   = len(param_dict['data_param']['handFeatures'])
    method_list = param_dict['ROC']['methods']
    method      = method_list[detector_id]
    i           = detector_id

    # load param
    feature_pkl = os.path.join(save_data_path, 'feature_extraction_kinEEPos_'+\
                            str(10.0) )
    ## scr_pkl = os.path.join(save_data_path, 'scr_'+method+'_'+str(fold_idx)+'_c'+str(i)+'.pkl')
    hmm_pkl = os.path.join(save_data_path, 'hmm_'+task_name+'_'+str(fold_idx)+'_c'+str(i)+'.pkl')
    clf_pkl = os.path.join(save_data_path, 'clf_'+method+'_'+str(fold_idx)+'.pkl')


    d = ut.load_pickle(feature_pkl) 
    # TODO: need to get normal train data with specific feature names
    feature_idx_list = []
    for feature in param_dict['data_param']['handFeatures'][detector_id]:
        feature_idx_list.append(param_dict['data_param']['isolationFeatures'].index(feature))

    d['param_dict']['feature_names'] = np.array(d['param_dict']['feature_names'])[feature_idx_list]
    d['param_dict']['feature_min'] = np.array(d['param_dict']['feature_min'])[feature_idx_list]
    d['param_dict']['feature_max'] = np.array(d['param_dict']['feature_max'])[feature_idx_list]   

    m_param_dict = {}
    m_param_dict['feature_params'] = d['param_dict']
    m_param_dict['successData']    = d['successData'][feature_idx_list]
    m_param_dict['failureData']    = d['failureData'][feature_idx_list]

    
    ## # load scaler
    ## import pickle
    ## if os.path.isfile(scr_pkl):
    ##     with open(scr_pkl, 'rb') as f:
    ##         m_scr = pickle.load(f)
    ## else: m_scr = None

    # load hmm
    if os.path.isfile(hmm_pkl) is False:
        print "No HMM pickle file: ", hmm_pkl
        sys.exit()

    d     = ut.load_pickle(hmm_pkl)
    m_gen = hmm.learning_hmm(d['nState'], d['nEmissionDim'], verbose=verbose)
    m_gen.set_hmm_object(d['A'], d['B'], d['pi'])

    # load classifier
    m_clf = cf.classifier( method=method, nPosteriors=d['nState'], parallel=True )
    m_clf.load_model(clf_pkl)

    return m_param_dict, m_gen, m_clf


def anomaly_detection_batch(X, save_data_path, task_name, method_list, param_dict,
                            Y=None, verbose=False):
    
    m_scr, m_gen, m_clf = get_detector_modules(save_data_path, task_name, method,
                                               param_dict, fold_idx=0,\
                                               verbose=False)

    if Y is None:
        Y = [1]*len(X)

    nDetector = 2 # temp
    startIdx  = 4
    ## scale     =
    n_jobs    = -1

    # get individual detection result
    y_preds_list = []
    for ii in xrange(nDetector):
        # Convert test data
        ll_classifier_test_X, ll_classifier_test_Y, ll_classifier_test_idx = \
          hmm.getHMMinducedFeaturesFromRawCombinedFeatures(m_gen[ii], X[ii] * scale, Y,
                                                           startIdx, \
                                                           n_jobs=n_jobs)

        y_preds = []
        for i in xrange(len(ll_classifier_test_X)):
            y_preds.append( m_clf[ii].predict(ll_classifier_test_X[i], y=ll_classifier_test_Y[i]) )
        y_preds_list.append(y_preds)

    ## if len(y_preds_list)>1:
        
        


if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    util.initialiseOptParser(p)
    opt, args = p.parse_args()

    from hrl_execution_monitor.params.IROS2017_params import *
    # IROS2017
    subject_names = ['s2', 's3','s4','s5', 's6','s7','s8', 's9']
    raw_data_path, save_data_path, param_dict = getParams(opt.task, opt.bDataRenew, \
                                                          opt.bHMMRenew, opt.bCLFRenew)
    save_data_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/IROS2017/'+opt.task+'_demo'

    task_name = 'feeding'
    param_dict['ROC']['methods'] = ['hmmgp0', 'hmmgp1']

    #c12 84
    save_data_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/IROS2017/'+opt.task+'_demo1'
    param_dict['HMM']['scale'] = [5.0, 5.0]

    #c11 85
    ## save_data_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/IROS2017/'+opt.task+'_demo2'
    ## param_dict['HMM']['scale'] = [7.,7.] #[1.0, 11.0]

    #c8 85
    save_data_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/IROS2017/'+opt.task+'_demo3'
    param_dict['HMM']['scale'] = [6.0, 6.0]
    

    train_detector_modules(subject_names, task_name, raw_data_path, save_data_path,\
                            param_dict, verbose=False)

    ## get_detector_modules(save_data_path, task_name, param_dict, detector_id=0,
    ##                      fold_idx=0, verbose=False)
    ## get_detector_modules(save_data_path, task_name, param_dict, detector_id=1,
    ##                      fold_idx=0, verbose=False)
