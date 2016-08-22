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

from sklearn.grid_search import ParameterGrid
from sklearn.cross_validation import KFold
import time
from sklearn import preprocessing

from hrl_anomaly_detection.hmm import learning_hmm as hmm
from hrl_anomaly_detection import data_manager as dm
import hrl_lib.util as ut
from hrl_anomaly_detection.util import *
from hrl_anomaly_detection.classifiers import classifier as cb

from joblib import Parallel, delayed

def tune_hmm(parameters, cv_dict, param_dict, processed_data_path, verbose=False, n_jobs=-1, \
             bSave=False, method='svm', max_check_fold=None, no_cov=False):

    ## Parameters
    # data
    data_dict  = param_dict['data_param']
    # AE
    AE_dict     = param_dict['AE']
    # HMM
    HMM_dict = param_dict['HMM']
    nState   = HMM_dict['nState']
    add_logp_d = HMM_dict['add_logp_d']
    
    ## cov      = HMM_dict['cov']
    # SVM
    SVM_dict = param_dict['SVM']

    ROC_dict = param_dict['ROC']
    
    #------------------------------------------
    kFold_list = cv_dict['kFoldList']
    if max_check_fold is not None:
        if max_check_fold < len(kFold_list):
            kFold_list = kFold_list[:max_check_fold]

    # sample x dim x length
    param_list = list(ParameterGrid(parameters))
    mean_list  = []
    std_list   = []
    
    for param in param_list:

        tp_l = [[] for i in xrange((ROC_dict['nPoints'])) ]
        fp_l = [[] for i in xrange((ROC_dict['nPoints'])) ]
        tn_l = [[] for i in xrange((ROC_dict['nPoints'])) ]
        fn_l = [[] for i in xrange((ROC_dict['nPoints'])) ]

        scores = []
        # Training HMM, and getting classifier training and testing data
        for idx, (normalTrainIdx, abnormalTrainIdx, normalTestIdx, abnormalTestIdx) \
          in enumerate(kFold_list):

            # dim x sample x length
            normalTrainData   = cv_dict['successData'][:, normalTrainIdx, :] * param['scale']
            abnormalTrainData = cv_dict['failureData'][:, abnormalTrainIdx, :] * param['scale'] 
            normalTestData    = cv_dict['successData'][:, normalTestIdx, :] * param['scale'] 
            abnormalTestData  = cv_dict['failureData'][:, abnormalTestIdx, :] * param['scale'] 

            #
            nEmissionDim = len(normalTrainData)
            if no_cov:
                cov_mult     = [HMM_dict['scale']]*(nEmissionDim**2)
            else:
                cov_mult     = [param['cov']]*(nEmissionDim**2)
            nLength      = len(normalTrainData[0][0])

            # scaling
            ml = hmm.learning_hmm( param['nState'], nEmissionDim )
            if (data_dict['handFeatures_noise'] and AE_dict['switch'] is False):
                ret = ml.fit( normalTrainData+\
                              np.random.normal(0.0, 0.03, np.shape(normalTrainData) )*HMM_dict['scale'], \
                              cov_mult=cov_mult )
            else:
                ret = ml.fit( normalTrainData, cov_mult=cov_mult )
                
            if ret == 'Failure' or np.isnan(ret):
                print "fitting failure", param['scale'], param['cov']
                scores.append(-1.0 * 1e+10)
                break
            ## if ret/float(len(normalTrainData[0])) < -100:
            print "Mean likelihoods: ", ret/float(len(normalTrainData[0]))

            #-----------------------------------------------------------------------------------------
            # Classifier train data
            #-----------------------------------------------------------------------------------------
            startIdx = 4
            # Classifier training data (parallelization)
            ll_classifier_train_X, ll_classifier_train_Y, ll_classifier_train_idx =\
              hmm.getHMMinducedFeaturesFromRawFeatures(ml, normalTrainData, abnormalTrainData, \
                                                       startIdx, add_logp_d)

            ## if len(np.shape(ll_classifier_train_X))<3:
            ##     print "feature extractin failed", np.shape(ll_classifier_train_X)
            ##     scores.append(-1.0 * 1e+10)
            ##     ret = 'Failure'
            ##     sys.exit()
            ##     break
            if np.amax( np.array(ll_classifier_train_X)[:,:,0] ) < 0:
                print "Negative likelihoods"
                scores.append(-1.0 * 1e+10)
                ret = 'Failure'
                break
            if True in np.isnan( np.array(ll_classifier_train_X).flatten().tolist() ):
                print "NaN in feature"
                scores.append(-1.0 * 1e+10)
                ret = 'Failure'
                break

            # split
            import random
            train_idx = random.sample(range(len(ll_classifier_train_X)), int( 0.5*len(ll_classifier_train_X)) )
            test_idx  = [x for x in range(len(ll_classifier_train_X)) if not x in train_idx]

            ll_classifier_test_X   = np.array(ll_classifier_train_X)[test_idx].tolist()
            ll_classifier_test_Y   = np.array(ll_classifier_train_Y)[test_idx].tolist()
            ll_classifier_test_idx = np.array(ll_classifier_train_idx)[test_idx].tolist()
            ll_classifier_train_X   = np.array(ll_classifier_train_X)[train_idx].tolist()
            ll_classifier_train_Y   = np.array(ll_classifier_train_Y)[train_idx].tolist()
            ll_classifier_train_idx = np.array(ll_classifier_train_idx)[train_idx].tolist()

            # nSample x nLength
            if ll_classifier_test_X == []:
                print "HMM-induced vector is wrong", param['scale'], param['cov']
                scores.append(-1.0 * 1e+10)
                ret = 'Failure'
                break

            if method == 'hmmgp':
                nSubSample = 20
                import random

                print "before: ", np.shape(ll_classifier_train_X), np.shape(ll_classifier_train_Y)
                new_X = []
                new_Y = []
                new_idx = []
                for i in xrange(len(ll_classifier_train_X)):
                    idx_list = range(len(ll_classifier_train_X[i]))
                    random.shuffle(idx_list)
                    new_X.append( np.array(ll_classifier_train_X)[i,idx_list[:nSubSample]].tolist() )
                    new_Y.append( np.array(ll_classifier_train_Y)[i,idx_list[:nSubSample]].tolist() )
                    new_idx.append( np.array(ll_classifier_train_idx)[i,idx_list[:nSubSample]].tolist() )

                ll_classifier_train_X = new_X
                ll_classifier_train_Y = new_Y
                ll_classifier_train_idx = new_idx
                print "After: ", np.shape(ll_classifier_train_X), np.shape(ll_classifier_train_Y)

                
            # flatten the data
            X_train_org, Y_train_org, idx_train_org = dm.flattenSample(ll_classifier_train_X, \
                                                                       ll_classifier_train_Y, \
                                                                       ll_classifier_train_idx,
                                                                       remove_fp=False)


            if X_train_org == []:
                print "HMM-induced vector is wrong", param['scale'], param['cov']
                scores.append(-1.0 * 1e+10)
                ret = 'Failure'
                break
            
            if method.find('svm')>=0:
                scaler = preprocessing.StandardScaler()
                try:
                    X_scaled = scaler.fit_transform(X_train_org)
                except:
                    scores.append(-1.0 * 1e+10)
                    ret = 'Failure'
                    break

                X_test = []
                Y_test = [] 
                for j in xrange(len(ll_classifier_test_X)):
                    if len(ll_classifier_test_X[j])==0: continue
                    X = scaler.transform(ll_classifier_test_X[j])                                

                    X_test.append(X)
                    Y_test.append(ll_classifier_test_Y[j])
            else:
                X_scaled = X_train_org
                X_test = ll_classifier_test_X
                Y_test = ll_classifier_test_Y
            weights = ROC_dict[method+'_param_range']


            if method == 'progress':
                print "Classifier fitting", method
                dtc = cb.classifier( method=method, nPosteriors=nState, nLength=nLength )
                ret = dtc.fit(X_scaled, Y_train_org, idx_train_org, parallel=True)
                cf_dict = {}
                cf_dict['method']      = dtc.method
                cf_dict['nPosteriors'] = dtc.nPosteriors
                cf_dict['l_statePosterior'] = dtc.l_statePosterior
                cf_dict['ths_mult']    = dtc.ths_mult
                cf_dict['ll_mu']       = dtc.ll_mu
                cf_dict['ll_std']      = dtc.ll_std
                cf_dict['logp_offset'] = dtc.logp_offset
            elif method == 'hmmgp':
                print "Classifier fitting", method
                dtc = cb.classifier( method=method, nPosteriors=nState, nLength=nLength )
                ret = dtc.fit(X_scaled, Y_train_org, idx_train_org, parallel=True)
                cf_dict = {}
                cf_dict['method']      = dtc.method
                cf_dict['nPosteriors'] = dtc.nPosteriors
                cf_dict['ths_mult']    = dtc.ths_mult
                dtc.save_model('./temp_hmmgp.pkl')
            

            print "Start to run classifiers"
            r = Parallel(n_jobs=n_jobs, verbose=50)(delayed(run_classifiers)(iii, X_scaled, Y_train_org, \
                                                                             idx_train_org, \
                                                                             X_test, Y_test, \
                                                                             nEmissionDim, nLength, \
                                                                             SVM_dict, weight=weights[iii], \
                                                                             method=method, cf_dict=cf_dict,\
                                                                             verbose=False)\
                                                                             for iii in xrange(len(weights)))
            idx_l, tp_ll, fn_ll, fp_ll, tn_ll = zip(*r)

            err_flag = False
            for iii, idx_point in enumerate(idx_l):
                if np.nan in tp_ll[iii]:
                    err_flag = True
                    break

                tp_l[idx_point] += tp_ll[iii]
                fn_l[idx_point] += fn_ll[iii]
                fp_l[idx_point] += fp_ll[iii]
                tn_l[idx_point] += tn_ll[iii]

            if err_flag:
                scores.append(-1.0 * 1e+10)
                ret = 'Failure'
                break

        if ret == 'Failure':
            mean_list.append(0)
            std_list.append(0)
        else:
            tpr_l = []
            fpr_l = []
            for i in xrange(ROC_dict['nPoints']):
                tpr_l.append( float(np.sum(tp_l[i]))/float(np.sum(tp_l[i])+np.sum(fn_l[i]))*100.0 )
                fpr_l.append( float(np.sum(fp_l[i]))/float(np.sum(fp_l[i])+np.sum(tn_l[i]))*100.0 )

            from sklearn import metrics
            mean_list.append( metrics.auc([0] + fpr_l + [100], [0] + tpr_l + [100], True) )
            std_list.append(0)
            
        ## print np.mean(scores), param
        ## mean_list.append( np.mean(scores) )
        ## std_list.append( np.std(scores) )
        
    print "mean: ", mean_list
    score_array = np.array(mean_list) #-np.array(std_list)
    idx_list = np.argsort(score_array)

    for i in idx_list:
        if np.isnan(score_array[i]): continue
            
        print("%0.3f : %0.3f (+/-%0.03f) for %r"
              % (score_array[i], mean_list[i], std_list[i], param_list[i]))

    if bSave: 
        savefile = os.path.join(processed_data_path,'../','result_run_hmm.txt')       
        if os.path.isfile(savefile) is False:
            with open(savefile, 'w') as file:
                file.write( "-----------------------------------------\n")
                file.write( 'dim: '+str(nEmissionDim)+'\n' )
                file.write( "%0.3f : %0.3f (+/-%0.03f) for %r"
                            % (score_array[i], mean_list[i], std_list[i], param_list[i])+'\n\n' )
        else:
            with open(savefile, 'a') as file:
                file.write( "-----------------------------------------\n")
                file.write( 'dim: '+str(nEmissionDim)+'\n' )
                file.write( "%0.3f : %0.3f (+/-%0.03f) for %r"
                            % (score_array[i], mean_list[i], std_list[i], param_list[i])+'\n\n' )



def run_classifiers(idx, X_scaled, Y_train_org, idx_train_org, X_test, Y_test, nEmissionDim, nLength, \
                    SVM_dict, weight, method='svm', cf_dict=None,\
                    verbose=False):

    if verbose: print "Run a classifier"
    dtc = cb.classifier( method=method, nPosteriors=nEmissionDim, nLength=nLength )
    dtc.set_params( **SVM_dict )
    if cf_dict is None:
        ret = dtc.fit(X_scaled, Y_train_org, idx_train_org, parallel=False)        
        if ret is False:
            print "SVM fitting failure!!"
            return idx, [np.nan], [np.nan], [np.nan], [np.nan]
    else:
        for k, v in cf_dict.iteritems():        
            exec 'dtc.%s = v' % k        
        if method == 'hmmgp':
            dtc.load_model('./temp_hmmgp.pkl')

    if method.find('svm')>=0:
        dtc.set_params( class_weight=weight )
    else:
        dtc.set_params( ths_mult=weight )

    tp_l = []
    fn_l = []
    fp_l = []
    tn_l = []
    for ii in xrange(len(X_test)):
        if len(Y_test[ii])==0: continue
        X = X_test[ii]
        est_y    = dtc.predict(X, y=Y_test[ii])

        for jj in xrange(len(est_y)):
            if est_y[jj] > 0.0: break        

        if Y_test[ii][0] > 0.0:
            if est_y[jj] > 0.0:
                tp_l.append(1)
            else: fn_l.append(1)
        elif Y_test[ii][0] <= 0.0:
            if est_y[jj] > 0.0: fp_l.append(1)
            else: tn_l.append(1)

    return idx, tp_l, fn_l, fp_l, tn_l

        


if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    p.add_option('--task', action='store', dest='task', type='string', default='pushing_microwhite',
                 help='type the desired task name')
    p.add_option('--dim', action='store', dest='dim', type=int, default=3,
                 help='type the desired dimension')
    p.add_option('--n_jobs', action='store', dest='n_jobs', type=int, default=-1,
                 help='number of processes for multi processing')
    p.add_option('--aeswtch', '--aesw', action='store_true', dest='bAESwitch',
                 default=False, help='Enable AE data.')
    p.add_option('--method', '--m', action='store', dest='method', type='string', default='progress',
                 help='type the desired method')

    p.add_option('--icra2017', action='store_true', dest='bICRA2017',
                 default=False, help='Enable ICRA2017.')
    
    p.add_option('--save', action='store_true', dest='bSave',
                 default=False, help='Save result.')
    opt, args = p.parse_args()
    
    rf_center     = 'kinEEPos'        
    local_range    = 10.0    

    if opt.bICRA2017 is False:
        from hrl_anomaly_detection.params import *
        raw_data_path, save_data_path, param_dict = getParams(opt.task, False, \
                                                              False, False, opt.dim,\
                                                              rf_center, local_range, \
                                                              bAESwitch=opt.bAESwitch, \
                                                              nPoints=8)


        if opt.task == 'scooping':
            parameters = {'nState': [25], 'scale': np.linspace(2.0,10.0,10), \
                          'cov': np.linspace(2.,5.0,10) }

        elif opt.task == 'feeding':
            if opt.dim == 2:
                parameters = {'nState': [25], 'scale': np.linspace(0.5,3.0,5), \
                              'cov': np.linspace(1.0,10.0,10) }
            elif opt.dim == 3:
                parameters = {'nState': [25], 'scale': np.linspace(1.0,10.0,10), \
                              'cov': np.linspace(1.0,10.0,10) }
            else:
                parameters = {'nState': [25], 'scale': np.linspace(1.0,15.0,7), \
                              'cov': np.linspace(0.1,10.0,3) }

        elif opt.task == 'pushing_microwhite':
            if opt.dim == 4:
                parameters = {'nState': [25], 'scale': np.linspace(2.0,8.0,10), \
                              'cov': np.linspace(0.01,6.0,10) }
            else:
                parameters = {'nState': [25], 'scale': np.linspace(1.0,10.0,10), \
                              'cov': np.linspace(0.1,2.0,10) }

        elif opt.task == 'pushing_microblack':
            parameters = {'nState': [25], 'scale': np.linspace(2.0,8.0,10), \
                          'cov': np.linspace(0.5,5.,10) }
        elif opt.task == 'pushing_toolcase':
            parameters = {'nState': [25], 'scale': np.linspace(5.0,15.0,7), \
                          'cov': np.linspace(0.1,5.0,5) }
        else:
            print "Not available task"
        max_check_fold = None #2
        no_cov = False

    else:

        from hrl_anomaly_detection.ICRA2017_params import *
        raw_data_path, save_data_path, param_dict = getParams(opt.task, False, \
                                                              False, False, opt.dim,\
                                                              rf_center, local_range, \
                                                              bAESwitch=opt.bAESwitch, \
                                                              nPoints=8)
        parameters = {'nState': [25], 'scale': np.linspace(3.0,15.0,10), \
                      'cov': np.linspace(1.0,15.0,1) }
        save_data_path = os.path.expanduser('~')+\
          '/hrl_file_server/dpark_data/anomaly/ICRA2017/'+opt.task+'_data_online/'+\
          str(param_dict['data_param']['downSampleSize'])+'_'+str(opt.dim)


        max_check_fold = None
        ## max_check_fold = 3
        no_cov = True

    #--------------------------------------------------------------------------------------
    # test change of logp
    
    crossVal_pkl        = os.path.join(save_data_path, 'cv_'+opt.task+'.pkl')
    if os.path.isfile(crossVal_pkl):
        d = ut.load_pickle(crossVal_pkl)
        kFold_list  = d['kFoldList']
    else:
        print "no existing data file, ", crossVal_pkl
        sys.exit()

    tune_hmm(parameters, d, param_dict, save_data_path, verbose=True, n_jobs=opt.n_jobs, \
             bSave=opt.bSave, method=opt.method, max_check_fold=max_check_fold, no_cov=no_cov)
