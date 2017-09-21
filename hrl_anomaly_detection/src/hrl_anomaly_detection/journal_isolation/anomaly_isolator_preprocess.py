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
## from hrl_anomaly_detection import util as util
## from hrl_anomaly_detection.util_viz import *
from hrl_anomaly_detection import util as util
from hrl_anomaly_detection import data_manager as dm
import hrl_anomaly_detection.IROS17_isolation.isolation_util as iutil
from hrl_execution_monitor import util as autil

from hrl_anomaly_detection.RA-L18_detection import util as vutil


# Private learners
from hrl_anomaly_detection.hmm import learning_hmm as hmm


from sklearn import preprocessing
from joblib import Parallel, delayed
import gc

random.seed(3334)
np.random.seed(3334)

def get_isolation_data(subject_names, task_name, raw_data_path, save_data_path,
                       param_dict, weight=1., window_steps=10, single_detector=False,
                       verbose=False):
    
    # load params (param_dict)
    data_dict  = param_dict['data_param']
    AE_dict    = param_dict['AE']
    data_renew = data_dict['renew']
    ae_renew   = param_dict['HMM']['renew']
    method     = param_dict['ROC']['methods'][0]
    
    if ae_renew: clf_renew = True
    else: clf_renew  = param_dict['SVM']['renew']

    #------------------------------------------
    if os.path.isdir(save_data_path) is False:
        os.system('mkdir -p '+save_data_path)

    crossVal_pkl = os.path.join(save_data_path, 'cv_'+task_name+'.pkl')
    if os.path.isfile(crossVal_pkl) and data_renew is False:
        print "CV data exists and no renew"
        d = ut.load_pickle(crossVal_pkl)         
    else:
        '''
        Use augmented data? if nAugment is 0, then aug_successData = successData
        '''        
        d = dm.getRawDataLOPO(subject_names, task_name, raw_data_path, \
                              save_data_path,\
                              downSampleSize=data_dict['downSampleSize'],\
                              handFeatures=data_dict['isolationFeatures'], \
                              rawFeatures=AE_dict['rawFeatures'], \
                              cut_data=data_dict['cut_data'], \
                              data_renew=data_renew, max_time=data_dict['max_time'],
                              ros_bag_image=True)

        ## print np.shape(d['successRawDataList'])
        ## print np.shape(d['failureRawDataList'])
        ## print np.shape(d['success_image_list']), type(d['success_image_list'][0])
        ## print np.shape(d['failure_image_list'])

        (d['successData'], d['success_image_list']), (d['failureData'], d['failure_image_list']), \
          d['success_files'], d['failure_files'], d['kFoldList'] \
          = dm.LOPO_data_index(d['successRawDataList'], d['failureRawDataList'],\
                               d['successFileList'], d['failureFileList'],\
                               success_image_list = d['success_image_list'], \
                               failure_image_list = d['failure_image_list'])

        print np.shape(d['successData']), np.shape(d['success_image_list'])
        print np.shape(d['failureData']), np.shape(d['failure_image_list'])
        ut.save_pickle(d, crossVal_pkl)

    if fine_tuning is False:
        td1, td2, td3 = vutil.get_ext_feeding_data(task_name, save_data_path, param_dict, d,
                                                   raw_feature=True)






    #==========================================================
    # parameters
    startIdx    = 4
    nPoints     = ROC_dict['nPoints']
    

    ## # load data (mix) -------------------------------------------------
    ## d = dm.getDataSet(subject_names, task_name, raw_data_path, \
    ##                   save_data_path,\
    ##                   downSampleSize=data_dict['downSampleSize'],\
    ##                   handFeatures=data_dict['isolationFeatures'], \
    ##                   data_renew=data_renew, max_time=data_dict['max_time'],\
    ##                   ros_bag_image=True, rndFold=True)
                      
    # split data with 80:20 ratio, 3set
    kFold_list = d['kFold_list'][:1]

    # flattening image list
    success_image_list = autil.image_list_flatten( d.get('success_image_list',[]) )
    failure_image_list = autil.image_list_flatten( d.get('failure_image_list',[]) )

    failure_labels = []
    for f in d['failureFiles']:
        failure_labels.append( int( f.split('/')[-1].split('_')[0] ) )
    failure_labels = np.array( failure_labels )

    # Static feature selection for isolation
    feature_list = []
    for feature in param_dict['data_param']['staticFeatures']:
        idx = [ i for i, x in enumerate(param_dict['data_param']['isolationFeatures']) if feature == x][0]
        feature_list.append(idx)
    successData_static = np.array(d['successData'])[feature_list]
    failureData_static = np.array(d['failureData'])[feature_list]
    
    #-----------------------------------------------------------------------------------------
    # Dynamic feature selection for detection and isolation
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
                                  noise_mag=noise_mag[i], suffix=str(i),\
                                  verbose=verbose, one_class=False)

    del d

    # ---------------------------------------------------------------
    # get data
    data_dict = {}
    data_pkl = os.path.join(save_data_path, 'isol_data.pkl')
    if os.path.isfile(data_pkl) is False or HMM_dict['renew'] or SVM_dict['renew']:

        l_data = Parallel(n_jobs=1, verbose=10)\
          (delayed(iutil.get_hmm_isolation_data)(idx, kFold_list[idx], failure_data_ad, \
                                                 failureData_static, \
                                                 failure_labels,\
                                                 failure_image_list,\
                                                 task_name, save_data_path, param_dict, weight,\
                                                 single_detector=single_detector,\
                                                 n_jobs=-1, window_steps=window_steps, verbose=verbose\
                                                 ) for idx in xrange(len(kFold_list)) )
        
        data_dict = {}
        for i in xrange(len(l_data)):
            idx = l_data[i][0]
            data_dict[idx] = (l_data[i][1],l_data[i][2],l_data[i][3],l_data[i][4] )
            
        print "save pkl: ", data_pkl
        ut.save_pickle(data_dict, data_pkl)            
    else:
        data_dict = ut.load_pickle(data_pkl)
    

    # ---------------------------------------------------------------
    scores = []
    for idx, (normalTrainIdx, abnormalTrainIdx, normalTestIdx, abnormalTestIdx) \
      in enumerate(kFold_list):
        print "kFold_list: ", idx

        (x_trains, y_train, x_tests, y_test) = data_dict[idx]         
        x_train = x_trains[0] 
        x_test  = x_tests[0] 
        print np.shape(x_train), np.shape(x_test)

        scaler = preprocessing.StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test  = scaler.transform(x_test)

        if type(x_train) is np.ndarray:
            x_train = x_train.tolist()
            x_test  = x_test.tolist()
        if type(y_train) is np.ndarray:
            y_train  = y_train.tolist()
            y_test   = y_test.tolist()
        
        ## from sklearn.svm import SVC
        ## clf = SVC(C=1.0, kernel='rbf') #, decision_function_shape='ovo')
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=400, n_jobs=-1)

        clf.fit(x_train, y_train)
        ## y_pred = clf.predict(x_test.tolist())
        score = clf.score(x_test, y_test)
        scores.append( score )
        print idx, " : score = ", score


    print scores
    print "Score mean = ", np.mean(scores), np.std(scores)

if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    util.initialiseOptParser(p)
    opt, args = p.parse_args()

    from hrl_anomaly_detection.journal_isolation.isolation_param import *
    # IROS2017
    subject_names = ['s2', 's3','s4','s5', 's6','s7','s8', 's9']
    raw_data_path, save_data_path, param_dict = getParams(opt.task, opt.bDataRenew, \
                                                          opt.bHMMRenew, opt.bCLFRenew)
    save_data_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/JOURNAL_ISOL/'+opt.task+'_1'


    window_steps= 5
    task_name = 'feeding'
    single_detector=True
    nb_classes = 12


    get_isolation_data(subject_names, task_name, raw_data_path, save_data_path,
                       param_dict, weight=1.0, single_detector=single_detector,
                       window_steps=window_steps, verbose=False)
