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
from hrl_anomaly_detection.hmm.learning_base import learning_base
from hrl_anomaly_detection.hmm import learning_hmm as hmm
import hrl_anomaly_detection.classifiers.classifier as cf
from joblib import Parallel, delayed


class anomaly_detector(learning_base):
    def __init__(self, method, nState, nLength, nEmissionDim,\
                 scale=1.0, cov=1.0, noise_std=0.03,\
                 weight=1., w_negative=1., gamma=1., cost=1., nu=0.5,\
                 ths_mult=-1.0, nugget=100.0, theta0=1.0, step_mag_range=[0.05, 0.2], verbose=False):
        self.method = method
        self.nLength = nLength
        self.scaler = None
        self.nEmissionDim = nEmissionDim
        self.startIdx = 4

        # hmm
        self.nState = nState
        self.scale  = scale
        self.cov    = cov
        self.noise_std  = noise_std

        # classifier
        self.weight     = weight
        self.w_negative = w_negative
        self.gamma      = gamma
        self.cost       = cost
        self.nu         = nu

        self.ths_mult = ths_mult
        self.nugget = nugget
        self.theta0 = theta0

        # step noise
        self.step_mag_range = step_mag_range
        
        self.hmm = hmm.learning_hmm(nState, nEmissionDim, verbose=verbose)
        self.dtc = cf.classifier( method=method, nPosteriors=nState, nLength=nLength, parallel=True )        

        self.fit_complete = False
        return

    def fit(self, X, y):
        print "---------------NEW--------------------"
        self.fit_complete = False

        X_train = np.swapaxes(X,0,1)*self.scale
        cov_mult = [self.cov]*(self.nEmissionDim**2)

        # set hmm param
        ## d = {'scale': self.scale, 'cov': self.cov}
        ## self.hmm.set_params(**d)
        ret = self.hmm.fit(X_train+ np.random.normal(0.0, self.noise_std, np.shape(X_train)) , \
                           cov_mult=cov_mult)
        if ret == 'Failure' or np.isnan(ret):
            print "hmm training failed"
            return
            #sys.exit()
        elif ret > 1000 or ret<0:
            print "hmm training result is not good"
            return
        
        # Classifier training data
        ll_classifier_train_X, ll_classifier_train_Y, ll_classifier_train_idx =\
          hmm.getHMMinducedFeaturesFromRawFeatures(self.hmm, X_train, startIdx=self.startIdx)
        
        # set classifier param
        d = {'w_negative': self.w_negative, 'gamma': self.gamma,\
             'cost': self.cost, 'class_weight': self.weight, 'nu': self.nu,\
             'ths_mult': self.ths_mult, 'nugget': self.nugget, 'theta0': self.theta0}
        self.dtc.set_params(**d)

        if self.method.find('hmmgp')>=0:
            nSubSample = 20 #20 # 20 
            nMaxData   = 50 # 40 100
            rnd_sample = True #False
            train_X, train_Y, _ =\
              dm.subsampleData(ll_classifier_train_X, ll_classifier_train_Y, None,\
                               nSubSample=nSubSample, nMaxData=nMaxData, rnd_sample=rnd_sample)
        else:
            train_X = ll_classifier_train_X
            train_Y = ll_classifier_train_Y

        
        # flatten the data
        if self.method.find('svm')>=0 or self.method.find('sgd')>=0: remove_fp=True
        else: remove_fp = False
        X_train, y_train, _ = dm.flattenSample(train_X, train_Y, remove_fp=remove_fp)
        ## print self.w_negative, self.weight

        if (self.method.find('svm')>=0 or self.method.find('sgd')>=0) and \
          not(self.method == 'osvm' or self.method == 'bpsvm'):
            self.scaler = preprocessing.StandardScaler()
            X_train = self.scaler.fit_transform(X_train)

        # fit classifier
        ret = self.dtc.fit(X_train, y_train)
        if ret is False:
            print "Fitting failure"
            return
            ## sys.exit()

        self.fit_complete = True
        return

    def predict(self, X):

        # Classifier test data
        # random step noise
        abnormalTestData = copy.copy(X)
        step_idx_l = []
        for i in xrange(len(X)):
            step_idx_l.append(None)
        for i in xrange(len(X)):
            start_idx = np.random.randint(self.startIdx, self.nLength*2/3, 1)[0]
            dim_idx   = np.random.randint(0, len(X[0]))
            step_mag  = np.random.uniform(self.step_mag_range[0], self.step_mag_range[1])
            
            abnormalTestData[i,dim_idx,start_idx:] += step_mag
            step_idx_l.append(start_idx)

        # dim x sample x length
        normalTestData = np.swapaxes(X,0,1)*self.scale
        abnormalTestData = np.swapaxes(abnormalTestData,0,1)*self.scale

        # Classifier test data
        ll_classifier_test_X, ll_classifier_test_Y, ll_classifier_test_idx =\
          hmm.getHMMinducedFeaturesFromRawFeatures(self.hmm, normalTestData, abnormalTestData, self.startIdx)

        labels = []
        delays = []
        for i in xrange(len(ll_classifier_test_X)):
            
            if (self.method.find('svm')>=0 or self.method.find('sgd')>=0) and \
              not(self.method == 'osvm' or self.method == 'bpsvm'):
                X_scaled = self.scaler.transform(ll_classifier_test_X[i])
            else:
                X_scaled = ll_classifier_test_X[i]
            est_y    = self.dtc.predict(X_scaled)

            label = -1.0
            for j in xrange(len(est_y)):                
                if est_y[j]>0:
                    if ll_classifier_test_Y[i][0]>0:
                        delays.append( ll_classifier_test_idx[i][j] - step_idx_l[i] )
                    label = 1.0
                    break
            
            labels.append(label)

        # should return label and time
        return labels, delays

    ## def decision_function(self, X):
    ##     return

    def score(self, X, y):
        if self.fit_complete is False: return 0.0
        
        true_y = (-np.ones(len(y))).tolist()+(np.ones(len(y))).tolist()

        tp = 0.0
        fp = 0.0
        tn = 0.0
        fn = 0.0

        est_y, delays = self.predict(X)
        
        for i in xrange(len(est_y)):
            if true_y[i]>0:
                if est_y[i]>0: tp += 1.0
                else: fn += 1.0
            else:
                if est_y[i]>0: fp += 1.0
                else: tn += 1.0

        # f1-score
        ## fscore = 2.0*tp/(2.0*tp+fn+fp)
        # f0.5-score
        fscore = 1.25*tp/(1.25*tp+0.25*fn+fp)
        # f2-score
        ## fscore = 5.0*tp/(5.0*tp+4.0*fn+fp)

        delay_score = 1.0 - np.mean( np.abs( np.array(delays).astype(float) /float(self.nLength)) )
        ## print fscore, delay_score
        return fscore+2.0*delay_score



def tune_detector(parameters, task_name, param_dict, save_data_path, verbose=False, n_jobs=-1, \
                  save=False, method='hmmgp', n_iter_search=1000, cv=3):

    ## Parameters
    # data
    data_dict  = param_dict['data_param']
    # AE
    AE_dict    = param_dict['AE']
    # HMM
    HMM_dict = param_dict['HMM']
    nState   = HMM_dict['nState']
    
    # SVM
    SVM_dict = param_dict['SVM']

    ROC_dict = param_dict['ROC']
    #------------------------------------------

    crossVal_pkl        = os.path.join(save_data_path, 'cv_'+task_name+'.pkl')
    if os.path.isfile(crossVal_pkl):
        cv_dict = ut.load_pickle(crossVal_pkl)
        kFold_list  = cv_dict['kFoldList']
    else:
        print "no existing data file, ", crossVal_pkl
        sys.exit()

    ## print np.shape(kFold_list)
    normalTrainIdx, abnormalTrainIdx, normalTestIdx, abnormalTestIdx = kFold_list[1]
    normalTrainData   = cv_dict['successData']#[:, normalTrainIdx, :]   

    # sample x feature x length
    ## X = np.vstack([ np.swapaxes(normalTrainData,0,1), np.swapaxes(abnormalTrainData,0,1) ])
    ## y = [-1]*len(normalTrainData[0])+[1]*len(abnormalTrainData)
    X = np.swapaxes(normalTrainData,0,1)
    y = [-1]*len(normalTrainData[0])

    nEmissionDim = len(normalTrainData)
    nLength = len(normalTrainData[0][0])

    # run randomized search
    n_jobs = 1
    from sklearn.model_selection import RandomizedSearchCV
    clf           = anomaly_detector(method, nState, nLength, nEmissionDim)
    random_search = RandomizedSearchCV(clf, param_distributions=parameters,
                                       cv=cv, n_jobs=n_jobs,
                                       n_iter=n_iter_search)
    random_search.fit(X, y)

    print("Best parameters set found on development set:")
    print()
    print(random_search.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means  = random_search.cv_results_['mean_test_score']
    stds   = random_search.cv_results_['std_test_score']
    params = random_search.cv_results_['params']

    score_list = []
    for i in xrange(len(means)):
        score_list.append([means[i], stds[i], params[i]])

    from operator import itemgetter
    score_list.sort(key=itemgetter(0), reverse=False)
    
    for mean, std, param in score_list:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, param))
    print()
    score_list = score_list[-20:]

    if save:
        savefile = os.path.join(save_data_path,'../','result_run_opt.txt')       
        if os.path.isfile(savefile) is False:
            with open(savefile, 'w') as file:
                file.write( "-----------------------------------------\n")
                file.write( str(nEmissionDim)+'\n\n' )
                for mean, std, param in score_list:
                    file.write( "%0.3f (+/-%0.03f) for %r"
                                % (mean, std * 2, param)+'\n\n')
        else:
            with open(savefile, 'a') as file:
                file.write( "-----------------------------------------\n")
                file.write( str(nEmissionDim)+'\n\n' )
                for mean, std, param in score_list:
                    file.write( "%0.3f (+/-%0.03f) for %r"
                                % (mean, std * 2, param)+'\n\n')
        

    return score_list[-1][0], score_list[-1][1], score_list[-1][2]





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
            if ('fixed' in method or 'progress' in method) and method.find('svm') <0:
                end_param    = start_param
                start_param *= (1.+ratio_p)
            else:
                end_param    = start_param
                start_param  = end_param *(1.0-ratio_p)
        elif np.amax(fpr_l) <= 0.05:
            if ('fixed' in method or 'progress' in method) and method.find('svm') <0:
                start_param = end_param
                end_param   *= ratio_p
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
            if ('fixed' in method or 'progress' in method) and method.find('svm') <0:
                end_param    = start_param
                start_param *= 1.0+ratio_p
            else:
                end_param    = start_param
                start_param  = end_param*(1.0-ratio_p)
        elif np.amax(fpr_l) <= 99.5:
            if ('fixed' in method or 'progress' in method) and method.find('svm') <0:
                start_param = end_param
                end_param   *= ratio_p
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
    
