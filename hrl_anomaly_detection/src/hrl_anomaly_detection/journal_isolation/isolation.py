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

from hrl_anomaly_detection.RAL18_detection import util as vutil
from hrl_anomaly_detection.journal_isolation import anomaly_isolator_preprocess as aip

# Private learners
from hrl_anomaly_detection.hmm import learning_hmm as hmm


from sklearn import preprocessing
from joblib import Parallel, delayed
import gc

random.seed(3334)
np.random.seed(3334)


def evaluation(subject_names, task_name, raw_data_path, save_data_path,
               param_dict, weight=1., window_steps=10, single_detector=False,\
               fine_tuning=False, verbose=False, renew=False):

    # Setup parameters
    ## if ae_renew: clf_renew = True
    ## else: clf_renew  = param_dict['SVM']['renew']
    
    
    main_data, sub_data = aip.get_data(subject_names, task_name, raw_data_path, save_data_path,
                                       param_dict)

    # Get all detection indices and f-scores
    train_idx_list, test_idx_list, fs_list = aip.get_detection_idx(save_data_path, main_data, sub_data,
                                                                   param_dict, renew=renew)

    # Select threshold index
    print "F-score: ", fs_list
    ths_idx = np.argmax(fs_list)
    print np.argmax(fs_list), np.amax(fs_list)


    # Extract isolation data
    train_data, test_data = aip.get_isolation_data(save_data_path, main_data, sub_data,
                                                   train_idx_list, test_idx_list, ths_idx, param_dict)

    ## train_isolator(train_data)
    ## test_isolator(test_data)

    # Summary


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

    evaluation(subject_names, task_name, raw_data_path, save_data_path,
               param_dict, weight=1.0, single_detector=single_detector,
               window_steps=window_steps, verbose=False)

