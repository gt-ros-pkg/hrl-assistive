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
## import rospy, roslib
import os, sys, copy
import random

# util
import numpy as np
import scipy
import hrl_lib.util as ut
from hrl_anomaly_detection.util import *
from hrl_anomaly_detection.util_viz import *
from hrl_anomaly_detection import data_manager as dm

# learning
from hrl_anomaly_detection.hmm import learning_hmm as hmm
from mvpa2.datasets.base import Dataset
from joblib import Parallel, delayed
from sklearn import metrics

import hrl_anomaly_detection.classifiers.classifier as cf
from hrl_anomaly_detection import data_manager as dm


def find_ROC_param_range(method, task_name, processed_data_path, param_dict, debug=False,\
                         modeling_pkl_prefix=None, add_print=''):

    ## Parameters
    # data
    data_dict  = param_dict['data_param']
    data_renew = data_dict['renew']
    dim        = len(data_dict['handFeatures'])
    # AE
    AE_dict     = param_dict['AE']
    # HMM
    HMM_dict   = param_dict['HMM']
    nState     = HMM_dict['nState']
    cov        = HMM_dict['cov']
    add_logp_d = HMM_dict.get('add_logp_d', False)
    # SVM
    SVM_dict   = param_dict['SVM']

    # ROC
    ROC_dict = param_dict['ROC']
    
    nFiles = data_dict['nNormalFold']*data_dict['nAbnormalFold']

    #-----------------------------------------------------------------------------------------
    # parameters
    startIdx    = 4
    nPoints     = ROC_dict['nPoints']

    #-----------------------------------------------------------------------------------------
    n_iter = 10
    nPoints = ROC_dict['nPoints'] = 4

    org_start_param = ROC_dict[method+'_param_range'][0]
    org_end_param = ROC_dict[method+'_param_range'][-1]
        
    if org_start_param > org_end_param:
        temp = org_start_param
        org_start_param = org_end_param
        org_end_param = org_start_param
    
    start_param = org_start_param 
    end_param = org_end_param #(org_start_param+org_end_param)/2.0
    delta_p = 2.5
    ratio_p = 0.9
    
    # find min param
    ## if 'fixed' in method or 'progress' in method:   
    ##     r = scipy.optimize.minimize(optFunc, x0=start_param, args=(param_dict, True), \
    ##                             options={maxiter:100})
    ## else:
    ##     r = scipy.optimize.minimize(optFunc, x0=np.log(end_param), method='Powell',\
    ##                                 args=(method, task_name, processed_data_path, param_dict, startIdx,\
    ##                                       None, True, False, None), \
    ##                                 options={'maxiter':30, 'direc':np.array([-1])},)
    ##                                 ## tol=0.1)
    ##                                 ## constraints={'type': 'ineq', 'fun': cond_min})
    ## min_param = r.x

    ## print r
    ## print start_param
    ## print "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    ## sys.exit()
    
    for run_idx in xrange(n_iter):

        print "----------------------------------------"
        print run_idx, ' : ', start_param, end_param
        print "----------------------------------------"
        ROC_dict[method+'_param_range'] = np.linspace(start_param, end_param, ROC_dict['nPoints'])
        

        ROC_data = {}
        ROC_data[method] = {}
        ROC_data[method]['complete'] = False 
        ROC_data[method]['tp_l'] = [ [] for j in xrange(nPoints) ]
        ROC_data[method]['fp_l'] = [ [] for j in xrange(nPoints) ]
        ROC_data[method]['tn_l'] = [ [] for j in xrange(nPoints) ]
        ROC_data[method]['fn_l'] = [ [] for j in xrange(nPoints) ]
        ROC_data[method]['delay_l'] = [ [] for j in xrange(nPoints) ]

        
        if debug: n_jobs=1
        else: n_jobs=-1
        r = Parallel(n_jobs=n_jobs, verbose=50)(delayed(cf.run_classifiers)( idx, processed_data_path, \
                                                                             task_name, \
                                                                             method, ROC_data, \
                                                                             ROC_dict, AE_dict, \
                                                                             SVM_dict, HMM_dict, \
                                                                             startIdx=startIdx, nState=nState,\
                                                                             modeling_pkl_prefix=modeling_pkl_prefix\
                                                                             )for idx in xrange(nFiles) \
                                                                             )


        tp_ll = [[] for j in xrange(nPoints)]
        fp_ll = [[] for j in xrange(nPoints)]
        tn_ll = [[] for j in xrange(nPoints)]
        fn_ll = [[] for j in xrange(nPoints)]

        l_data = r
        for i in xrange(len(l_data)):
            for j in xrange(nPoints):
                tp_ll[j] += l_data[i][method]['tp_l'][j]
                fp_ll[j] += l_data[i][method]['fp_l'][j]
                tn_ll[j] += l_data[i][method]['tn_l'][j]
                fn_ll[j] += l_data[i][method]['fn_l'][j]

        tpr_l = []
        fpr_l = []
        for i in xrange(nPoints):
            tpr_l.append( float(np.sum(tp_ll[i]))/float(np.sum(tp_ll[i])+np.sum(fn_ll[i]))*100.0 )
            fpr_l.append( float(np.sum(fp_ll[i]))/float(np.sum(fp_ll[i])+np.sum(tn_ll[i]))*100.0 )

        if np.amin(fpr_l) > 0.5:
            if 'fixed' in method or 'progress' in method:
                end_param    = start_param
                start_param -= delta_p
            else:
                end_param    = start_param
                start_param  = end_param *(1.0-ratio_p)
        elif np.amax(fpr_l) <= 0.05:
            if 'fixed' in method or 'progress' in method:
                start_param = end_param
                end_param   += delta_p
            else:
                start_param = end_param
                end_param   = start_param*(1.+ratio_p)                        
        else:
            for i in xrange(len(fpr_l)-1):
                if fpr_l[i] <= 0.05 and fpr_l[i+1] > 0.05:
                    start_param = ROC_dict[method+'_param_range'][i]
                    end_param   = ROC_dict[method+'_param_range'][i+1]
                    break
            if (fpr_l[i] <= 0.05 and fpr_l[i+1] > 0.05) and abs(fpr_l[i]-fpr_l[i+1])<1.0:
                print "Converged!!!!!!!!"
                break

        delta_p *= 0.9
        ratio_p *= 0.9
            
        if abs(start_param-end_param) < 0.001: break

    min_param = start_param
    if i+1 > len(fpr_l)-1: fpr_l.append(fpr_l[-1])
    
    min_fpr_range = [fpr_l[i], fpr_l[i+1]]

    # find max param
    start_param = (org_start_param+org_end_param)/2.0
    end_param = org_end_param    
    delta_p = 2.5
    ratio_p = 0.9
    
    # find min param
    for run_idx in xrange(n_iter):

        print "----------------------------------------"
        print run_idx, ' : ', start_param, end_param
        print "----------------------------------------"
        ROC_dict[method+'_param_range'] = np.linspace(start_param, end_param, ROC_dict['nPoints'])
        

        ROC_data = {}
        ROC_data[method] = {}
        ROC_data[method]['complete'] = False 
        ROC_data[method]['tp_l'] = [ [] for j in xrange(nPoints) ]
        ROC_data[method]['fp_l'] = [ [] for j in xrange(nPoints) ]
        ROC_data[method]['tn_l'] = [ [] for j in xrange(nPoints) ]
        ROC_data[method]['fn_l'] = [ [] for j in xrange(nPoints) ]
        ROC_data[method]['delay_l'] = [ [] for j in xrange(nPoints) ]

        
        if debug: n_jobs=1
        else: n_jobs=-1
        r = Parallel(n_jobs=n_jobs, verbose=50)(delayed(cf.run_classifiers)( idx, processed_data_path, task_name, \
                                                                          method, ROC_data, \
                                                                          ROC_dict, AE_dict, \
                                                                          SVM_dict, HMM_dict, \
                                                                          startIdx=startIdx, nState=nState,\
                                                                          modeling_pkl_prefix=modeling_pkl_prefix) \
                                                                          for idx in xrange(nFiles) \
                                                                          )

        tp_ll = [[] for j in xrange(nPoints)]
        fp_ll = [[] for j in xrange(nPoints)]
        tn_ll = [[] for j in xrange(nPoints)]
        fn_ll = [[] for j in xrange(nPoints)]

        l_data = r
        for i in xrange(len(l_data)):
            for j in xrange(nPoints):
                tp_ll[j] += l_data[i][method]['tp_l'][j]
                fp_ll[j] += l_data[i][method]['fp_l'][j]
                tn_ll[j] += l_data[i][method]['tn_l'][j]
                fn_ll[j] += l_data[i][method]['fn_l'][j]

        tpr_l = []
        fpr_l = []
        for i in xrange(nPoints):
            tpr_l.append( float(np.sum(tp_ll[i]))/float(np.sum(tp_ll[i])+np.sum(fn_ll[i]))*100.0 )
            fpr_l.append( float(np.sum(fp_ll[i]))/float(np.sum(fp_ll[i])+np.sum(tn_ll[i]))*100.0 )

        if np.amin(fpr_l) > 99.5:
            if 'fixed' in method or 'progress' in method:
                end_param    = start_param
                start_param -= delta_p
            else:
                end_param    = start_param
                start_param  = end_param*(1.0-ratio_p)
        elif np.amax(fpr_l) <= 99.5:
            if 'fixed' in method or 'progress' in method:
                start_param = end_param
                end_param   += delta_p
            else:
                start_param = end_param
                end_param   = start_param*(1.0+ratio_p)                        
        else:
            for i in xrange(len(fpr_l)-1):
                if fpr_l[i] <= 99.5 and fpr_l[i+1] > 99.5:
                    start_param = ROC_dict[method+'_param_range'][i]
                    end_param   = ROC_dict[method+'_param_range'][i+1]
                    break
            delta_p *= 0.9
            ratio_p *= 0.9
            if (fpr_l[i] <= 99.5 and fpr_l[i+1] > 99.5) and abs(fpr_l[i]-fpr_l[i+1])<1.0:
                break
                            
        if abs(start_param-end_param) < 0.05: break
    
    max_param = end_param
    if i+1 > len(fpr_l)-1: fpr_l.append(fpr_l[-1])
    max_fpr_range = [fpr_l[i], fpr_l[i+1]]
    
    print "----------------------------------------"
    print run_idx, ' : ', min_param, max_param
    print "----------------------------------------"
    
    savefile = os.path.join(processed_data_path,'../','result_find_param_range.txt')
    if os.path.isfile(savefile) is False:
        with open(savefile, 'w') as file:
            file.write( "-----------------------------------------\n")
            file.write( add_print+" \n" )
            file.write( 'task: '+task_name+' method: '+method+' dim: '+str(dim)+'\n' )
            file.write( "%0.3f with %r" % (min_param, min_fpr_range)+'\n' )
            file.write( "%0.3f with %r" % (max_param, max_fpr_range)+'\n\n' )
    else:
        with open(savefile, 'a') as file:
            file.write( "-----------------------------------------\n")
            file.write( add_print+" \n" )
            file.write( 'task: '+task_name+' method: '+method+' dim: '+str(dim)+'\n' )
            file.write( "%0.3f with %r" % (min_param, min_fpr_range)+'\n' )
            file.write( "%0.3f with %r" % (max_param, max_fpr_range)+'\n\n' )



def optFunc(x, method, task_name, processed_data_path, param_dict, startIdx, \
            modeling_pkl_prefix=None, min_eval=False, debug=False, nFiles=None):

    ## Parameters
    # data
    data_dict  = param_dict['data_param']
    data_renew = data_dict['renew']
    dim        = len(data_dict['handFeatures'])
    # AE
    AE_dict     = param_dict['AE']
    # HMM
    HMM_dict   = param_dict['HMM']
    nState     = HMM_dict['nState']
    cov        = HMM_dict['cov']
    add_logp_d = HMM_dict.get('add_logp_d', False)
    # SVM
    SVM_dict   = param_dict['SVM']

    # ROC
    ROC_dict = param_dict['ROC']

    if nFiles is None:
        nFiles = data_dict['nNormalFold']*data_dict['nAbnormalFold']

    nPoints = ROC_dict['nPoints'] = 2
    val     = np.exp(x[0])

    if min_eval:
        ROC_dict[method+'_param_range'] = [val, val+0.01]
    else:
        ROC_dict[method+'_param_range'] = [val, val-0.01]


    ROC_data = {}
    ROC_data[method] = {}
    ROC_data[method]['complete'] = False 
    ROC_data[method]['tp_l'] = [ [] for j in xrange(nPoints) ]
    ROC_data[method]['fp_l'] = [ [] for j in xrange(nPoints) ]
    ROC_data[method]['tn_l'] = [ [] for j in xrange(nPoints) ]
    ROC_data[method]['fn_l'] = [ [] for j in xrange(nPoints) ]
    ROC_data[method]['delay_l'] = [ [] for j in xrange(nPoints) ]

    nFiles = 1
    ## n_jobs = 1
    if debug: n_jobs=1
    else: n_jobs=-1
    r = Parallel(n_jobs=n_jobs, verbose=50)(delayed(cf.run_classifiers)( idx, processed_data_path, task_name, \
                                                                      method, ROC_data, \
                                                                      ROC_dict, AE_dict, \
                                                                      SVM_dict, HMM_dict, \
                                                                      startIdx=startIdx, nState=nState,\
                                                                      modeling_pkl_prefix=modeling_pkl_prefix) \
                                                                      for idx in xrange(nFiles) \
                                                                      )

    ## r = []
    ## for idx in xrange(nFiles):        
    ##     r.append( cf.run_classifiers( idx, processed_data_path, task_name, \
    ##                                   method, ROC_data, \
    ##                                   ROC_dict, AE_dict, \
    ##                                   SVM_dict, HMM_dict, \
    ##                                   startIdx=startIdx, nState=nState,\
    ##                                   modeling_pkl_prefix=modeling_pkl_prefix ) )
    ## print r
                                                                      

    tp_ll = [[] for j in xrange(nPoints)]
    fp_ll = [[] for j in xrange(nPoints)]
    tn_ll = [[] for j in xrange(nPoints)]
    fn_ll = [[] for j in xrange(nPoints)]

    l_data = r
    for i in xrange(len(l_data)):
        for j in xrange(nPoints):
            tp_ll[j] += l_data[i][method]['tp_l'][j]
            fp_ll[j] += l_data[i][method]['fp_l'][j]
            tn_ll[j] += l_data[i][method]['tn_l'][j]
            fn_ll[j] += l_data[i][method]['fn_l'][j]

    tpr_l = []
    fpr_l = []
    for i in xrange(nPoints):
        tpr_l.append( float(np.sum(tp_ll[i]))/float(np.sum(tp_ll[i])+np.sum(fn_ll[i]))*100.0 )
        fpr_l.append( float(np.sum(fp_ll[i]))/float(np.sum(fp_ll[i])+np.sum(tn_ll[i]))*100.0 )

    print fpr_l, 1.0/val, x
    if fpr_l[0] > 0.0:
        return fpr_l[0]
    ## elif fpr_l[1] > 0.0:
    ##     return fpr_l[1]
    else:        
        return 1000.0/val
    

## def cond_min():

##     return
