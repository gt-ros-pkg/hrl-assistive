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

# system & utils
import os, sys, copy, random
import numpy as np
import scipy
import hrl_lib.util as ut

# Private utils
from hrl_anomaly_detection.util import *
from hrl_anomaly_detection.util_viz import *
from hrl_anomaly_detection import data_manager as dm
from hrl_anomaly_detection import util as util
from hrl_execution_monitor import util as autil

# Private learners
from hrl_anomaly_detection.hmm import learning_hmm as hmm
import hrl_anomaly_detection.classifiers.classifier as cf
import hrl_anomaly_detection.data_viz as dv
import hrl_anomaly_detection.isolator.isolation_util as iutil

from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed

# visualization
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec
import itertools
colors = itertools.cycle(['g', 'm', 'c', 'k', 'y','r', 'b', ])
shapes = itertools.cycle(['x','v', 'o', '+'])

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42 
random.seed(3334)
np.random.seed(3334)


def evaluation_isolation(task_name, processed_data_path, method_list, n_labels=12,
                         save_pdf=False, verbose=False):
    ## Parameters
    crossVal_pkl = os.path.join(processed_data_path, 'cv_'+task_name+'.pkl')
    d = ut.load_pickle(crossVal_pkl)
    kFold_list = d['kFoldList'] 

    data_pkl  = os.path.join(processed_data_path, 'isol_data.pkl')
    data_dict = ut.load_pickle(data_pkl)

    # need to get anomaly idx?


    # ---------------------------------------------------------------
    scores_list = []
    ## for i in xrange(len(method_list)):
    ##     scores.append([])
        
    for idx, (normalTrainIdx, abnormalTrainIdx, normalTestIdx, abnormalTestIdx) \
      in enumerate(kFold_list):
        print "kFold_list: ", idx
        
        (x_trains, y_train, x_tests, y_test) = data_dict[idx]         
        x_train = x_trains[0] 
        x_test  = x_tests[0]

        from sklearn import preprocessing
        scaler = preprocessing.StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test  = scaler.transform(x_test)

        if type(x_train) is np.ndarray:
            x_train = x_train.tolist()
            x_test  = x_test.tolist()
        if type(y_train) is np.ndarray:
            y_train  = y_train.tolist()
            y_test   = y_test.tolist()

        scores = []
        if 'rnd' in method_list:
            y_pred = np.random.choice(range(n_labels), len(y_test))
            score = accuracy_score(np.array(y_test)-2, y_pred)
            scores.append( score )

        if 'svm_raw' in method_list:
            
            
        if 'svm_signal' in method_list:        
            from sklearn.svm import SVC
            clf = SVC(C=1.0, kernel='rbf') #, decision_function_shape='ovo')
            clf.fit(x_train, y_train)
            score = clf.score(x_test, y_test)
            scores.append( score )
            
        if 'rfc_signal' in method_list:            
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(n_estimators=400, n_jobs=-1)
            clf.fit(x_train, y_train)
            score = clf.score(x_test, y_test)
            scores.append( score )
            
        ## print idx, " : score = ", score
        scores_list.append(scores)

    print np.mean(scores_list, axis=0)
    print np.std(scores_list, axis=0)


if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    util.initialiseOptParser(p)

    p.add_option('--eval_isol', '--ei', action='store_true', dest='evaluation_isolation',
                 default=False, help='Evaluate anomaly isolation with double detectors.')
    p.add_option('--svd_renew', '--sr', action='store_true', dest='svd_renew',
                 default=False, help='Renew ksvd')
    
    opt, args = p.parse_args()

    #---------------------------------------------------------------------------           
    # Run evaluation
    #---------------------------------------------------------------------------           
    rf_center     = 'kinEEPos'        
    scale         = 1.0
    local_range   = 10.0
    nPoints = 40 #None

    from hrl_anomaly_detection.isolator.IROS2017_params import *
    raw_data_path, save_data_path, param_dict = getParams(opt.task, opt.bDataRenew, \
                                                          opt.bHMMRenew, opt.bCLFRenew, opt.dim,\
                                                          rf_center, local_range, nPoints=nPoints)
    if opt.bNoUpdate: param_dict['ROC']['update_list'] = []
    # Mikako - bad camera
    # s1 - kaci - before camera calibration
    subjects = ['s2', 's3','s4','s5', 's6','s7','s8', 's9']

    save_data_path = os.path.expanduser('~')+\
      '/hrl_file_server/dpark_data/anomaly/AURO2016/'+opt.task+'_data_isolation6/'+\
      str(param_dict['data_param']['downSampleSize'])+'_'+str(opt.dim)

    param_dict['ROC']['methods'] = ['progress0', 'progress1']
    param_dict['HMM']['scale'] = [5.0, 11.0]
    param_dict['HMM']['cov']   = 1.0
    single_detector = False

    param_dict['ROC']['progress0_param_range'] = -np.logspace(0., 0.9, nPoints)
    param_dict['ROC']['progress1_param_range'] = -np.logspace(0., 0.9, nPoints)


    ## if opt.evaluation_isolation:

    ## method_list = ['rnd', 'svm_signal', 'rfc_signal']
    method_list = ['svm_raw']
    evaluation_isolation(opt.task, save_data_path, method_list)
