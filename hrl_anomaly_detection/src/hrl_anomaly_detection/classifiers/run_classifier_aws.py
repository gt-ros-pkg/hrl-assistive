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
from hrl_anomaly_detection.params import *

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

        method = params['method'][0]
        data = dm.getHMMData(method, nFiles, processed_data_path, task_name, param_dict)
        all_param = list(ParameterGrid(params))

        for param_idx, param in enumerate(all_param):
            task = self.lb_view.apply(cross_validate_local, param_idx, nFiles, \
                                      data, \
                                      default_params=param_dict, custom_params=param, n_jobs=1)
            self.all_tasks.append(task)
        return self.all_tasks




def cross_validate_local(param_idx, nFiles, data, default_params, custom_params, n_jobs=-1):
    '''
    
    '''
    print "in cross validate"
    from joblib import Parallel, delayed
    ## Default Parameters
    # data
    data_dict = default_params['data_param']
    # AE
    AE_dict = default_params['AE']
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

    from hrl_anomaly_detection.classifiers import classifier as cb
    r = Parallel(n_jobs=n_jobs, verbose=50)(delayed(cb.run_classifier)(j,\
                                                                       data[file_idx]['X_scaled'], \
                                                                       data[file_idx]['Y_train_org'], \
                                                                       data[file_idx]['idx_train_org'], \
                                                                       data[file_idx]['X_test'], \
                                                                       data[file_idx]['Y_test'], \
                                                                       data[file_idx]['idx_test'], \
                                                                       method, HMM_dict['nState'], \
                                                                       data[file_idx]['nLength'], \
                                                                       ROC_dict['nPoints'],\
                                                                       custom_params, ROC_dict )
                                                                       for j in xrange(ROC_dict['nPoints'])
                                                                       for file_idx in xrange(nFiles))

    ## r = Parallel(n_jobs=n_jobs, verbose=50)(delayed(run_ROC_eval)(j, data[file_idx]['X_scaled'], \
    ##                                                               data[file_idx]['Y_train_org'], \
    ##                                                               data[file_idx]['idx_train_org'], \
    ##                                                               data[file_idx]['X_test'], \
    ##                                                               data[file_idx]['Y_test'], \
    ##                                                               data[file_idx]['idx_test'], \
    ##                                                               method, ROC_dict, \
    ##                                                               HMM_dict, custom_params, \
    ##                                                               data[file_idx]['nLength'])
    ##                                                               for j in xrange(ROC_dict['nPoints'])
    ##                                                               for file_idx in xrange(nFiles))
    l_j, l_tp_l, l_fp_l, l_fn_l, l_tn_l, l_delay_l = zip(*r)
    for i, j in enumerate(l_j):
        if j == 'fit failed':
            print i,j
            continue
        ROC_data[method]['tp_l'][j] += l_tp_l[i]
        ROC_data[method]['fp_l'][j] += l_fp_l[i]
        ROC_data[method]['fn_l'][j] += l_fn_l[i]
        ROC_data[method]['tn_l'][j] += l_tn_l[i]
        ROC_data[method]['delay_l'][j] += l_delay_l[i]

    return ROC_data, param_idx, custom_params


## # classifier
## def run_ROC_eval(j, X_scaled, Y_train_org, idx_train_org, \
##                  X_test, Y_test, idx_test, method, ROC_dict, HMM_dict, params, nLength):
##     from hrl_anomaly_detection.classifiers import classifier as cb

##     dtc = cb.classifier( method=method, nPosteriors=HMM_dict['nState'], nLength=nLength )        
##     if method == 'svm':
##         weights = ROC_dict['svm_param_range']
##         dtc.set_params( class_weight=weights[j] )
##     elif method == 'cssvm_standard':
##         weights = np.logspace(-2, 0.1, nPoints)
##         dtc.set_params( class_weight=weights[j] )
##     elif method == 'cssvm':
##         weights = ROC_dict['cssvm_param_range']
##         dtc.set_params( class_weight=weights[j] )
##     elif method == 'progress_time_cluster':
##         thresholds = ROC_dict['progress_param_range']
##         dtc.set_params( ths_mult = thresholds[j] )
##     elif method == 'fixed':
##         thresholds = ROC_dict['fixed_param_range']
##         dtc.set_params( ths_mult = thresholds[j] )
##     elif method == 'sgd':
##         weights = ROC_dict['sgd_param_range']
##         dtc.set_params( class_weight=weights[j] )
##     else:
##         print "Not available method"
##         return "Not available method", -1, params

##     dtc.set_params(**params)

##     ret = dtc.fit(X_scaled, Y_train_org, idx_train_org)
##     if ret is False: return 'fit failed', -1

##     # evaluate the classifier
##     tp_l = []
##     fp_l = []
##     tn_l = []
##     fn_l = []
##     delay_l = []
##     delay_idx = 0

##     for ii in xrange(len(X_test)):
##         if len(Y_test[ii])==0: continue
##         est_y    = dtc.predict(X_test[ii], y=Y_test[ii])

##         for jj in xrange(len(est_y)):
##             if est_y[jj] > 0.0:                
##                 try:
##                     delay_idx = idx_test[ii][jj]
##                 except:
##                     print "Error!!!!!!!!!!!!!!!!!!"
##                     print np.shape(idx_test), ii, jj
##                 ## print "Break ", ii, " ", jj, " in ", est_y, " = ", ll_classifier_test_Y[ii][jj]
##                 break        

##         if Y_test[ii][0] > 0.0:
##             if est_y[jj] > 0.0:
##                 tp_l.append(1)
##                 delay_l.append(delay_idx)
##             else: fn_l.append(1)
##         elif Y_test[ii][0] <= 0.0:
##             if est_y[jj] > 0.0: fp_l.append(1)
##             else: tn_l.append(1)

##     return j, tp_l, fp_l, fn_l, tn_l, delay_l


def disp_score(results, method, nPoints):

    score_list = []
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
        for j in xrange(nPoints):
            try:
                tpr_l.append( float(np.sum(tp_ll[j]))/float(np.sum(tp_ll[j])+np.sum(fn_ll[j]))*100.0 )
                fpr_l.append( float(np.sum(fp_ll[j]))/float(np.sum(fp_ll[j])+np.sum(tn_ll[j]))*100.0 )
            except:
                tpr_l.append(0.0)
                fpr_l.append(0.0)
                print j, np.shape(tp_ll[j]), np.shape(fn_ll[j]), np.shape(fp_ll[j]), np.shape(tn_ll[j])
                print "failed to get TPR and FPR"
                sys.exit()
            ## break
        print "tpr: ", tpr_l
        print "fpr: ", fpr_l

        # get AUC
        from sklearn import metrics        
        score_list.append( [metrics.auc([0]+fpr_l+[100], [0]+tpr_l+[100], True), ret_params] )

        # get max tp in fpr (0~20)
        ## max_tp = 0
        ## for i, fp in enumerate(fpr_l):
        ##     if fp < 20.0:
        ##         if tpr_l[i] > max_tp: max_tp = tpr_l[i]
        ## score_list.append( [max_tp, ret_params] )

    ## plt.show()
    # Get sorted results
    from operator import itemgetter
    score_list.sort(key=itemgetter(0), reverse=False)

    for i in xrange(len(score_list)):
        print("%0.3f for %r" % (score_list[i][0], score_list[i][1]))

    


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
    p.add_option('--task', action='store', dest='task', type='string', default='pushing_microwhite',
                 help='type the desired task name')
    p.add_option('--dim', action='store', dest='dim', type=int, default=3,
                 help='type the desired dimension')
    p.add_option('--method', '--m', action='store', dest='method', type='string', default='svm',
                 help='type the method name')
    p.add_option('--n_jobs', action='store', dest='n_jobs', type=int, default=-1,
                 help='number of processes for multi processing')
    p.add_option('--aeswtch', '--aesw', action='store_true', dest='bAESwitch',
                 default=False, help='Enable AE data.')

    p.add_option('--rawplot', '--rp', action='store_true', dest='bRawDataPlot',
                 default=False, help='Plot raw data.')
    p.add_option('--cpu', '--c', action='store_true', dest='bCPU', default=True,
                 help='Enable cpu mode')
    p.add_option('--renew', action='store_true', dest='bRenew', default=False,
                 help='Renew result')
    opt, args = p.parse_args()

    rf_center     = 'kinEEPos'        
    local_range    = 10.0    

    if opt.task == 'scooping':
        subjects = ['Wonyoung', 'Tom', 'lin', 'Ashwin', 'Song', 'Henry2'] #'Henry', 
        raw_data_path, save_data_path, param_dict = getScooping(opt.task, False, \
                                                                False, False,\
                                                                rf_center, local_range,\
                                                                ae_swtch=opt.bAESwitch, dim=opt.dim)

        nPoints        = 10
        ROC_param_dict = {'methods': ['progress_time_cluster', 'svm','fixed', 'hmmosvm', 'hmmsvm_dL'],\
                          'update_list': [],\
                          'nPoints': nPoints,\
                          'progress_param_range':-np.linspace(0., 10.0, nPoints), \
                          'svm_param_range': np.logspace(-1.8, 1.0, nPoints),\
                          'osvm_param_range': np.logspace(-6, 0.2, nPoints),\
                          'hmmosvm_param_range': np.logspace(-4.0, 1.5, nPoints),\
                          'hmmsvm_diag_param_range': np.logspace(-4, 1.2, nPoints),\
                          'hmmsvm_dL_param_range': np.logspace(-4, 1.2, nPoints),\
                          'hmmsvm_LSLS_param_range': np.logspace(-4, 1.2, nPoints),\
                          'fixed_param_range': -np.logspace(0.0, 0.9, nPoints)+1.2,\
                          'cssvm_param_range': np.logspace(0.0, 2.0, nPoints) }
        param_dict['ROC'] = ROC_param_dict

        nFiles = param_dict['data_param']['nNormalFold']*param_dict['data_param']['nAbnormalFold']
        if opt.method == 'svm':
            parameters = {'method': ['svm'], 'svm_type': [0], 'kernel_type': [2], \
                          'cost': np.linspace(3.0, 7.0, 6),\
                          'gamma': np.logspace(-2, 0.7, 10), \
                          'w_negative': np.linspace(1.0, 6.0,10) }
        elif opt.method == 'hmmosvm':
            parameters = {'method': ['hmmosvm'], 'svm_type': [2], 'kernel_type': [2], \
                          'hmmosvm_nu': np.logspace(-5,0.,5)
                          }
        elif opt.method == 'hmmsvm_diag':
            parameters = {'method': ['hmmsvm_diag'], 'svm_type': [0], 'kernel_type': [2], \
                          'hmmsvm_diag_cost': np.linspace(5,15.0,5),\
                          'hmmsvm_diag_gamma': np.linspace(0.01,2.0,5), \
                          'hmmsvm_diag_w_negative': np.linspace(0.2,1.5,5)
                          }
        elif opt.method == 'osvm':
            parameters = {'method': ['osvm'], 'svm_type': [2], 'kernel_type': [2], \
                          'osvm_nu': np.logspace(-5,-3,10),
                          }
        elif opt.method == 'hmmsvm_dL':
            parameters = {'method': ['hmmsvm_dL'], 'svm_type': [0], 'kernel_type': [2], \
                          'hmmsvm_dL_cost': np.linspace(5,15.0,5),\
                          'hmmsvm_dL_gamma': np.linspace(0.01,2.0,5), \
                          'hmmsvm_dL_w_negative': np.linspace(0.2,1.5,5)
                          }
        elif opt.method == 'hmmsvm_LSLS':
            parameters = {'method': ['hmmsvm_LSLS'], 'svm_type': [0], 'kernel_type': [2], \
                          'hmmsvm_LSLS_cost': np.linspace(5,15.0,5),\
                          'hmmsvm_LSLS_gamma': np.linspace(0.01,2.0,5), \
                          'hmmsvm_LSLS_w_negative': np.linspace(0.2,1.5,5)
                          }
                
                      

    #---------------------------------------------------------------------------
    elif opt.task == 'feeding':
        
        subjects = ['Tom', 'lin', 'Ashwin', 'Song', 'wonyoung']
        raw_data_path, save_data_path, param_dict = getFeeding(opt.task, False, \
                                                               False, False,\
                                                               rf_center, local_range,\
                                                               ae_swtch=opt.bAESwitch, dim=opt.dim)
        nPoints        = 10
        ROC_param_dict = {'methods': ['svm'],\
                          'update_list': [],\
                          'nPoints': nPoints,\
                          'progress_param_range':-np.linspace(0., 10.0, nPoints), \
                          'svm_param_range': np.logspace(-2, 1.2, nPoints),\
                          'hmmsvm_diag_param_range': np.logspace(-4, 1.2, nPoints),\
                          'hmmsvm_dL_param_range': np.logspace(-4, 1.2, nPoints),\
                          'hmmosvm_param_range': np.logspace(-3.5, 0.5, nPoints),\
                          'hmmsvm_LSLS_param_range': np.logspace(-4, 1.2, nPoints),\
                          'fixed_param_range': np.linspace(1.0, -3.0, nPoints),\
                          'osvm_param_range': np.logspace(-6, 0.2, nPoints),\
                          'cssvm_param_range': np.logspace(0.0, 2.0, nPoints) }
        param_dict['ROC'] = ROC_param_dict

        nFiles = 4 #param_dict['data_param']['nNormalFold']*param_dict['data_param']['nAbnormalFold']
        if opt.method == 'svm':
            if opt.dim == 2:
                parameters = {'method': ['svm'], 'svm_type': [0], 'kernel_type': [2], \
                              'cost': np.linspace(1,4.0,5),\
                              'gamma': np.linspace(0.1,4.0,5), \
                              'w_negative': np.linspace(0.1,5.0,5) }
            elif opt.dim == 3:
                parameters = {'method': ['svm'], 'svm_type': [0], 'kernel_type': [2], \
                              'cost': np.linspace(1.0,6.0,5),\
                              'gamma': np.linspace(0.1,5.0,10), \
                              'w_negative': np.linspace(0.1,3.0,5) }
                param_dict['ROC']['svm_param_range'] = np.logspace(-2.0, 1.5, nPoints)
            else:
                parameters = {'method': ['svm'], 'svm_type': [0], 'kernel_type': [2], \
                              'cost': np.linspace(1.0,6.0,5),\
                              'gamma': np.linspace(0.1,5.0,10), \
                              'w_negative': np.linspace(0.1,3.0,5) }
        elif opt.method == 'hmmsvm_diag':
            parameters = {'method': ['hmmsvm_diag'], 'svm_type': [0], 'kernel_type': [2], \
                          'hmmsvm_diag_cost': np.linspace(5,15.0,5),\
                          'hmmsvm_diag_gamma': np.linspace(0.01,2.0,5), \
                          'hmmsvm_diag_w_negative': np.linspace(0.2,1.5,5)
                          }
        elif opt.method == 'hmmosvm':
            parameters = {'method': ['hmmosvm'], 'svm_type': [2], 'kernel_type': [2], \
                          'hmmosvm_nu': np.logspace(-4,-2.,5)
                         }
        elif opt.method == 'osvm':
            parameters = {'method': ['osvm'], 'svm_type': [2], 'kernel_type': [2], \
                          'osvm_nu': np.logspace(-5,-3,10),
                          }
        elif opt.method == 'hmmsvm_dL':
            parameters = {'method': ['hmmsvm_dL'], 'svm_type': [0], 'kernel_type': [2], \
                          'hmmsvm_dL_cost': np.linspace(5,15.0,5),\
                          'hmmsvm_dL_gamma': np.linspace(0.01,4.0,5), \
                          'hmmsvm_dL_w_negative': np.linspace(0.2,1.5,5)
                          }
        elif opt.method == 'hmmsvm_LSLS':
            parameters = {'method': ['hmmsvm_LSLS'], 'svm_type': [0], 'kernel_type': [2], \
                          'hmmsvm_LSLS_cost': np.linspace(5,15.0,5),\
                          'hmmsvm_LSLS_gamma': np.linspace(0.01,2.0,5), \
                          'hmmsvm_LSLS_w_negative': np.linspace(0.2,1.5,5)
                          }
                

    #---------------------------------------------------------------------------           
    elif opt.task == 'pushing_microwhite':
    
        subjects = ['gatsbii']
        raw_data_path, save_data_path, param_dict = getPushingMicroWhite(opt.task, False, \
                                                                         False, False,\
                                                                         rf_center, local_range,\
                                                                         ae_swtch=opt.bAESwitch, dim=opt.dim)
        
        nPoints        = 10
        ROC_param_dict = {'methods': ['hmmsvm_LSLS'],\
                          'nPoints': nPoints,\
                          'progress_param_range':np.linspace(-1., -10., nPoints), \
                          'svm_param_range': np.logspace(-2, 0, nPoints),\
                          'bpsvm_param_range': np.logspace(-2, 0, nPoints),\
                          'hmmsvm_diag_param_range': np.logspace(-4, 1.2, nPoints),\
                          'hmmsvm_dL_param_range': np.logspace(-4, 1.2, nPoints),\
                          'hmmsvm_LSLS_param_range': np.logspace(-4, 1.2, nPoints),\
                          'hmmosvm_param_range': np.logspace(-4.0, 1.5, nPoints),\
                          'fixed_param_range': np.linspace(1.0, -3.0, nPoints),\
                          'cssvm_param_range': np.logspace(-4, 1.2, nPoints),\
                          'osvm_param_range': np.logspace(-6, 0.2, nPoints),\
                          'sgd_param_range': np.logspace(-1.0, -0.0, nPoints)}
        param_dict['ROC'] = ROC_param_dict

        nFiles = 4 #9
        if opt.method == 'svm':
            if opt.dim == 2:
                parameters = {'method': ['svm'], 'svm_type': [0], 'kernel_type': [2], \
                              'cost': np.linspace(0.5,8.0,10),\
                              'gamma': np.linspace(2.0,6.0,5), \
                              'w_negative': np.logspace(-2,0.2,10) }
            elif opt.dim == 5:
                parameters = {'method': ['svm'], 'svm_type': [0], 'kernel_type': [2], \
                              'cost': np.linspace(0.5,4.0,5),\
                              'gamma': np.logspace(-3,0.0,5), \
                              'w_negative': np.linspace(0.2,1.5,5) }
            else:
                parameters = {'method': ['svm'], 'svm_type': [0], 'kernel_type': [2], \
                              'cost': np.linspace(5,15.0,5),\
                              'gamma': np.logspace(-3,0.2,5), \
                              'w_negative': np.linspace(0.2,1.5,5) }
        elif opt.method == 'hmmosvm':
            parameters = {'method': ['hmmosvm'], 'svm_type': [2], 'kernel_type': [2], \
                          'hmmosvm_nu': np.logspace(-4,-2.,5)
                          }
        elif opt.method == 'osvm':
            parameters = {'method': ['osvm'], 'svm_type': [2], 'kernel_type': [2], \
                          'osvm_nu': np.logspace(-5,-3,10),
                          }
        elif opt.method == 'hmmsvm_diag':
            parameters = {'method': ['hmmsvm_diag'], 'svm_type': [0], 'kernel_type': [2], \
                          'hmmsvm_diag_cost': np.linspace(5,15.0,5),\
                          'hmmsvm_diag_gamma': np.linspace(0.01,2.0,5), \
                          'hmmsvm_diag_w_negative': np.linspace(0.2,1.5,5)
                          }
        elif opt.method == 'hmmsvm_dL':
            parameters = {'method': ['hmmsvm_dL'], 'svm_type': [0], 'kernel_type': [2], \
                          'hmmsvm_dL_cost': np.linspace(5,15.0,5),\
                          'hmmsvm_dL_gamma': np.linspace(0.01,2.0,5), \
                          'hmmsvm_dL_w_negative': np.linspace(0.2,1.5,5)
                          }
        elif opt.method == 'hmmsvm_LSLS':
            parameters = {'method': ['hmmsvm_LSLS'], 'svm_type': [0], 'kernel_type': [2], \
                          'hmmsvm_LSLS_cost': np.linspace(5,15.0,5),\
                          'hmmsvm_LSLS_gamma': np.linspace(0.01,2.0,5), \
                          'hmmsvm_LSLS_w_negative': np.linspace(0.2,1.5,5)
                          }
        elif opt.method == 'bpsvm':
            parameters = {'method': ['bpsvm'], 'svm_type': [0], 'kernel_type': [2], \
                          'bpsvm_cost': np.linspace(5,15.0,5),\
                          'bpsvm_gamma': np.linspace(0.01,2.0,5), \
                          'bpsvm_w_negative': np.linspace(0.2,1.5,5)
                          }

                
                                      
        ## parameters = {'method': ['sgd'], \
        ##               'gamma': np.logspace(-1.5,-0.5,5), \
        ##               'w_negative': np.linspace(1.0,2.5,5) }

    #---------------------------------------------------------------------------           
    elif opt.task == 'pushing_toolcase':
    
        subjects = ['gatsbii']
        raw_data_path, save_data_path, param_dict = getPushingToolCase(opt.task, False, \
                                                                       False, False,\
                                                                       rf_center, local_range,\
                                                                       ae_swtch=opt.bAESwitch, dim=opt.dim)
        
        #temp
        nPoints        = 10
        ROC_param_dict = {'methods': ['svm'],\
                          'nPoints': nPoints,\
                          'progress_param_range':np.linspace(-1., -10., nPoints), \
                          'svm_param_range': np.logspace(-2, 0.1, nPoints),\
                          'osvm_param_range': np.logspace(-4, 1.0, nPoints),\
                          'fixed_param_range': np.linspace(1.0, -3.0, nPoints),\
                          'hmmosvm_param_range': np.logspace(-4.0, 1.5, nPoints),\
                          'hmmsvm_diag_param_range': np.logspace(-4, 1.2, nPoints),\
                          'hmmsvm_dL_param_range': np.logspace(-4, 1.2, nPoints),\
                          'hmmsvm_LSLS_param_range': np.logspace(-4, 1.2, nPoints),\
                          'cssvm_param_range': np.logspace(-3.0, -0.5, nPoints) }
        param_dict['ROC'] = ROC_param_dict

        nFiles = 4 #9

        if opt.method == 'svm':        
            if opt.dim == 5:
                parameters = {'method': ['svm'], 'svm_type': [0], 'kernel_type': [2], \
                              'cost': np.linspace(5.0,12.0,5),\
                              'gamma': [0.01, 0.1, 1.0], \
                              'w_negative': np.linspace(0.1,2.0,4) }
            elif opt.dim == 3:
                parameters = {'method': ['svm'], 'svm_type': [0], 'kernel_type': [2], \
                              'cost': np.linspace(1.0,5.0,5),\
                              'gamma': np.logspace(-2, 0.4, 5), \
                              'w_negative': np.linspace(1.0,5.0,5) }
            else:
                parameters = {'method': ['svm'], 'svm_type': [0], 'kernel_type': [2], \
                              'cost': np.linspace(1.0,10.0,5),\
                              'gamma': [0.1, 1.0, 2.0, 3.0, 4.0], \
                              'w_negative': np.linspace(0.1,3.0,5) }
        elif opt.method == 'hmmosvm':
            parameters = {'method': ['hmmosvm'], 'svm_type': [2], 'kernel_type': [2], \
                          'hmmosvm_nu': np.logspace(-4,-2.,5)
                          }
        elif opt.method == 'hmmsvm_diag':
            parameters = {'method': ['hmmsvm_diag'], 'svm_type': [0], 'kernel_type': [2], \
                          'hmmsvm_diag_cost': np.linspace(5,15.0,5),\
                          'hmmsvm_diag_gamma': np.linspace(0.01,2.0,5), \
                          'hmmsvm_diag_w_negative': np.linspace(0.2,1.5,5)
                          }
        elif opt.method == 'osvm':
            parameters = {'method': ['osvm'], 'svm_type': [2], 'kernel_type': [2], \
                          'osvm_nu': np.logspace(-5.5,-3.0,20),
                          }
        elif opt.method == 'hmmsvm_dL':
            parameters = {'method': ['hmmsvm_dL'], 'svm_type': [0], 'kernel_type': [2], \
                          'hmmsvm_dL_cost': np.linspace(5,15.0,5),\
                          'hmmsvm_dL_gamma': np.linspace(0.01,2.0,5), \
                          'hmmsvm_dL_w_negative': np.linspace(0.2,1.5,5)
                          }
        elif opt.method == 'hmmsvm_LSLS':
            parameters = {'method': ['hmmsvm_LSLS'], 'svm_type': [0], 'kernel_type': [2], \
                          'hmmsvm_LSLS_cost': np.linspace(5,15.0,5),\
                          'hmmsvm_LSLS_gamma': np.linspace(0.01,2.0,5), \
                          'hmmsvm_LSLS_w_negative': np.linspace(0.2,1.5,5)
                          }
            
        ## if opt.dim == 4:
        ##     parameters = {'method': ['cssvm'], 'svm_type': [0], 'kernel_type': [2], \
        ##                   'cssvm_cost': np.linspace(8.0,15.0,5),\
        ##                   'cssvm_gamma': [0.01, 0.05, 0.1], \
        ##                   'cssvm_w_negative': [1.0, 2.0, 3.0] }
        ## elif opt.dim == 3:
        ##     parameters = {'method': ['cssvm'], 'svm_type': [0], 'kernel_type': [2], \
        ##                   'cssvm_cost': np.linspace(1.,15.0,10),\
        ##                   'cssvm_gamma': [2.0], \
        ##                   'cssvm_w_negative': [2.0] }

    #---------------------------------------------------------------------------           
    elif opt.task == 'pushing_microblack':
    
        subjects = ['gatsbii']
        raw_data_path, save_data_path, param_dict = getPushingMicroBlack(opt.task, False, \
                                                                         False, False,\
                                                                         rf_center, local_range,\
                                                                         ae_swtch=opt.bAESwitch, dim=opt.dim)
        
        #temp
        nPoints        = 10
        ROC_param_dict = {'methods': ['svm'],\
                          'nPoints': nPoints,\
                          'progress_param_range':np.linspace(-1., -10., nPoints), \
                          'svm_param_range': np.logspace(-2, 0, nPoints),\
                          'fixed_param_range': np.linspace(1.0, -3.0, nPoints),\
                          'hmmosvm_param_range': np.logspace(-3.5, 0.5, nPoints),\
                          'osvm_param_range': np.logspace(-6., 0.2, nPoints),\
                          'cssvm_param_range': np.logspace(-4, 1.2, nPoints),\
                          'hmmsvm_diag_param_range': np.logspace(-4, 1.2, nPoints),\
                          'hmmsvm_dL_param_range': np.logspace(-4, 1.2, nPoints),\
                          'hmmsvm_LSLS_param_range': np.logspace(-4, 1.2, nPoints),\
                          'sgd_param_range': np.logspace(-1.0, -0.0, nPoints)}
        param_dict['ROC'] = ROC_param_dict

        nFiles = 4 #9
        ## parameters = {'method': ['sgd'], \
        ##               'gamma': np.logspace(-1.5,-0.5,5), \
        ##               'w_negative': np.linspace(1.0,2.5,5) }
        ## parameters = {'method': ['cssvm'], 'svm_type': [0], 'kernel_type': [2], \
        ##               'cost': [3.,4.,5.],\
        ##               'gamma': [1.5,2.0,2.5], \
        ##               'w_negative': np.linspace(0.2,0.7,5) }
        if opt.method == 'svm':
            if opt.dim == 5:
                parameters = {'method': ['svm'], 'svm_type': [0], 'kernel_type': [2], \
                              'cost': np.linspace(1.0,4.0,5),\
                              'gamma': np.logspace(-1.5,0.5,10), \
                              'w_negative': np.logspace(-0.5, 0.5, 5) }
                param_dict['ROC']['svm_param_range'] = np.logspace(-2, -0.5, nPoints)
            else:
                parameters = {'method': ['svm'], 'svm_type': [0], 'kernel_type': [2], \
                              'cost': np.linspace(1.0,4.0,5),\
                              'gamma': np.linspace(0.1,8.0,10), \
                              'w_negative': np.logspace(-2, 0.5, 5) }                
        elif opt.method == 'osvm':
            parameters = {'method': ['osvm'], 'svm_type': [2], 'kernel_type': [2], \
                          'osvm_nu': np.logspace(-6,-1,10),
                          }
        elif opt.method == 'hmmosvm':
            parameters = {'method': ['hmmosvm'], 'svm_type': [2], 'kernel_type': [2], \
                          'hmmosvm_nu': np.logspace(-4,-2.,5)
                         }
        elif opt.method == 'hmmsvm_diag':
            parameters = {'method': ['hmmsvm_diag'], 'svm_type': [0], 'kernel_type': [2], \
                          opt.method+'_cost': np.linspace(5,15.0,5),\
                          opt.method+'_gamma': np.linspace(0.01,2.0,5), \
                          opt.method+'_w_negative': np.linspace(0.2,1.5,5)
                          }
        elif opt.method == 'hmmsvm_dL':
            parameters = {'method': ['hmmsvm_dL'], 'svm_type': [0], 'kernel_type': [2], \
                          'hmmsvm_dL_cost': np.linspace(5,15.0,5),\
                          'hmmsvm_dL_gamma': np.linspace(0.01,2.0,5), \
                          'hmmsvm_dL_w_negative': np.linspace(0.2,1.5,5)
                          }
        elif opt.method == 'hmmsvm_LSLS':
            parameters = {'method': ['hmmsvm_LSLS'], 'svm_type': [0], 'kernel_type': [2], \
                          'hmmsvm_LSLS_cost': np.linspace(5,15.0,5),\
                          'hmmsvm_LSLS_gamma': np.linspace(0.01,2.0,5), \
                          'hmmsvm_LSLS_w_negative': np.linspace(0.2,1.5,5)
                          }


        ## if opt.dim > 2:
        ##     ROC_param_dict['hmmosvm_param_range'] = np.logspace(-2, 2.5, nPoints)
        ## elif opt.dim == 2:
        ##     ROC_param_dict['hmmosvm_param_range'] = np.logspace(-4, 1.5, nPoints)

    else:
        print "Selected task name is not available."
        sys.exit()

    #--------------------------------------------------------------------------------------

    ## parameters = {'method': ['svm'], 'svm_type': [1], 'svn_kernel_type': [1,2], 'svn_degree': [2], \
    ##               'svm_w_negative': [1.0]}

    # Get combined results
    max_param_idx = len( list(ParameterGrid(parameters)) )
    method = parameters['method'][0]
    print "max_param_idx = ", max_param_idx
    AE_param_dict = param_dict['AE']
    if AE_param_dict['switch'] == True and AE_param_dict['add_option'] is not None:
        result_pkl = os.path.join(save_data_path, 'result_'+opt.task+'_rawftb_'+str(opt.dim)+'.pkl')
    elif AE_param_dict['switch'] == True:
        result_pkl = os.path.join(save_data_path, 'result_'+opt.task+'_raw_'+str(opt.dim)+'.pkl')
    else:
        result_pkl = os.path.join(save_data_path, 'result_'+opt.task+'_'+str(opt.dim)+'.pkl')
        
    ##################################################################################################
    # cpu version
    if opt.bCPU:
        ## save_data_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/RSS2016/'+task+'_data'
        ## nFiles = 2
        ## parameters = {'method': ['svm'], 'svm_type': [0], 'kernel_type': [2], \
        ##               'cost': [1.0, 3.], 'w_negative': [3.0]}
        
        if os.path.isfile(result_pkl) is False or opt.bRenew is True:

            ## Custom parameters
            method = parameters['method'][0]
            if method is not 'osvm':
                if method is 'hmmosvm':
                    data = dm.getHMMData(method, nFiles, save_data_path, opt.task, param_dict, negTrain=True)
                else:
                    data = dm.getHMMData(method, nFiles, save_data_path, opt.task, param_dict)
                    if method is 'bpsvm':
                        # get cutting idx for pos data # need to fix!!!!!!!!!!!!!!!! TODO
                        l_idx = dm.getHMMCuttingIdx(data['X_scaled'],
                                                    data['Y_train_org'],
                                                    data['idx_train_org'])
                        
    
            results = []
            for param_idx, param in enumerate( list(ParameterGrid(parameters)) ):
                if method is 'osvm':
                    startIdx=4
                    data_pkl = os.path.join(save_data_path, 'cv_'+opt.task+'.pkl' )
                    data = dm.getPCAData(nFiles, startIdx, data_pkl, window=10, posdata=False)
                elif method is 'bpsvm':
                    startIdx=4
                    data_pkl = os.path.join(save_data_path, 'cv_'+opt.task+'.pkl' )
                    data = dm.getPCAData(nFiles, startIdx, data_pkl, posdata=True, pos_cut_indices=l_idx)
                elif method is 'rfc':
                    startIdx=4
                    data_pkl = os.path.join(save_data_path, 'cv_'+opt.task+'.pkl' )
                    data = dm.getPCAData(nFiles, startIdx, data_pkl, posdata=True)
                    
                print "running ", param_idx, " / ", len(list(ParameterGrid(parameters))) 
                start = time.time()
                ret_ROC_data, ret_param_idx, ret_params = cross_validate_local(param_idx, nFiles, \
                                                                               data, param_dict, param, \
                                                                               n_jobs=opt.n_jobs)
                end = time.time()
                print "-------------------------------------------------"
                print param_idx, " Elapsed time: ", end - start
                print "-------------------------------------------------"
                results.append([ret_ROC_data, ret_param_idx, ret_params])
                disp_score(results, method, nPoints)

            ut.save_pickle(results, result_pkl)
        else:
            results = ut.load_pickle(result_pkl)

    else:

        if os.path.isfile(result_pkl) is False:

            cloud = CloudSearchForClassifier(os.path.expanduser('~')+\
                                             '/.starcluster/ipcluster/SecurityGroup:@sc-testdpark-us-east-1.json', \
                                             os.path.expanduser('~')+'/.ssh/HRL_ANOMALY.pem', 'testdpark', 'ubuntu')
            cloud.run_with_local_data(parameters, save_data_path, opt.task, nFiles, param_dict )
            print len(cloud.client)

            # wait until finishing parameter search
            time1 = time.time()
            while cloud.get_num_all_tasks() != cloud.get_num_tasks_completed():
                print "Processing tasks, ", cloud.get_num_tasks_completed(), ' / ', cloud.get_num_all_tasks()
                time.sleep(60*5)
                print "Std out"
                print "===================================="
                cloud.print_stdout()
                results = cloud.get_completed_results()
                if len(results)>0:
                    disp_score(results, method, nPoints)
                print "===================================="
            results = cloud.get_completed_results()
            time2 = time.time()
            print "===================================="
            print "Result"
            print "===================================="
            for result in results:        
                print result
            print "===================================="
            print "time"
            print time2-time1
            print "===================================="
            #print "Std out"
            #print "===================================="
            #cloud.print_stdout()
            #print "===================================="
            import hrl_lib.util as ut
            ut.save_pickle(results, result_pkl)            
        else:
            import hrl_lib.util as ut
            results = ut.load_pickle(result_pkl)

        #cloud.stop()
        cloud.flush()
        print "Finished"

    # 000000000000000000000000000000000000000000000000000000000000000000
    disp_score(results, method, nPoints)




## def run_classifier(param_idx, modeling_pkl, method, HMM_dict, ROC_dict, params, n_jobs=-1):

##     from joblib import Parallel, delayed
##     import os
##     ## from hrl_anomaly_detection.classifiers import run_classifier_aws as rca
##     import hrl_lib.util as ut
##     import numpy as np
##     from sklearn import preprocessing

##     # train a classifier and evaluate it using test data.
##     d            = ut.load_pickle(modeling_pkl)
##     ## startIdx = d['startIdx']
##     ll_classifier_train_X   = d['ll_classifier_train_X']
##     ll_classifier_train_Y   = d['ll_classifier_train_Y']         
##     ll_classifier_train_idx = d['ll_classifier_train_idx']
##     ll_classifier_test_X    = d['ll_classifier_test_X']  
##     ll_classifier_test_Y    = d['ll_classifier_test_Y']
##     ll_classifier_test_idx  = d['ll_classifier_test_idx']
##     nLength      = d['nLength']
##     nPoints      = ROC_dict['nPoints']

##     # flatten the data
##     X_train_org = []
##     Y_train_org = []
##     idx_train_org = []
##     for i in xrange(len(ll_classifier_train_X)):
##         for j in xrange(len(ll_classifier_train_X[i])):
##             X_train_org.append(ll_classifier_train_X[i][j])
##             Y_train_org.append(ll_classifier_train_Y[i][j])
##             idx_train_org.append(ll_classifier_train_idx[i][j])

##     # training data preparation
##     if 'svm' in method:
##         scaler = preprocessing.StandardScaler()
##         ## scaler = preprocessing.scale()
##         X_scaled = scaler.fit_transform(X_train_org)
##     else:
##         X_scaled = X_train_org
##     ## print method, " : Before classification : ", np.shape(X_scaled), np.shape(Y_train_org)

##     # test data preparation
##     X_test = []
##     Y_test = ll_classifier_test_Y
##     idx_test = ll_classifier_test_idx
##     for ii in xrange(len(ll_classifier_test_X)):
##         if 'svm' in method:
##             X = scaler.transform(ll_classifier_test_X[ii])                                
##         elif method == 'progress_time_cluster' or method == 'fixed':
##             X = ll_classifier_test_X[ii]
##         X_test.append(X)


##     #================================================================================================

##     tp_ll = [ [] for i in xrange(ROC_dict['nPoints']) ]
##     fp_ll = [ [] for i in xrange(ROC_dict['nPoints']) ]
##     fn_ll = [ [] for i in xrange(ROC_dict['nPoints']) ]
##     tn_ll = [ [] for i in xrange(ROC_dict['nPoints']) ]
##     delay_ll = [ [] for i in xrange(ROC_dict['nPoints']) ]

##     ## print "started to run a classifier"
##     ## start = time.time()        

##     r = Parallel(n_jobs=n_jobs, verbose=50)(delayed(run_ROC_eval)(j, X_scaled, Y_train_org, idx_train_org, \
##                                                               X_test, Y_test, idx_test, method, ROC_dict, \
##                                                               HMM_dict, params, nLength) \
##                                                               for j in xrange(ROC_dict['nPoints']))    

##     l_j, l_tp_l, l_fp_l, l_fn_l, l_tn_l, l_delay_l = zip(*r)

##     for i, j in enumerate(l_j):
        
##         tp_ll[j] += l_tp_l[i]
##         fp_ll[j] += l_fp_l[i]
##         fn_ll[j] += l_fn_l[i]
##         tn_ll[j] += l_tn_l[i]
##         delay_ll[j] += l_delay_l[i]

##     ## end = time.time()
##     ## print " Elapsed time to eval: ", end - start
##     ## sys.exit()

##     ## return tp_ll, fp_ll, fn_ll, tn_ll, delay_ll
##     if tp_ll is None or fp_ll is None or fn_ll is None or tn_ll is None:
##         return tp_ll, -1

##     ROC_data = {}
##     ROC_data['tp_l'] = tp_ll
##     ROC_data['fp_l'] = fp_ll
##     ROC_data['fn_l'] = fn_ll
##     ROC_data['tn_l'] = tn_ll
##     ROC_data['delay_l'] = delay_ll

##     return ROC_data, param_idx

