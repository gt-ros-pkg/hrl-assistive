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
             bSave=False, method='svm', max_check_fold=None):

    ## Parameters
    # data
    data_dict  = param_dict['data_param']
    # AE
    AE_dict     = param_dict['AE']
    # HMM
    HMM_dict = param_dict['HMM']
    nState   = HMM_dict['nState']
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

            if AE_dict['switch']:
                if verbose: print "Start "+str(idx)+"/"+str(len(kFold_list))+"th iteration"

                AE_proc_data = os.path.join(processed_data_path, 'ae_processed_data_'+str(idx)+'.pkl')
                d = ut.load_pickle(AE_proc_data)
                
                if AE_dict['filter']:
                    # NOTE: pooling dimension should vary on each auto encoder.
                    # Filtering using variances
                    normalTrainData   = d['normTrainDataFiltered']
                    abnormalTrainData = d['abnormTrainDataFiltered']
                    normalTestData    = d['normTestDataFiltered']
                    abnormalTestData  = d['abnormTestDataFiltered']
                    ## import data_viz as dv
                    ## dv.viz(normalTrainData)
                    ## continue                   
                else:
                    normalTrainData   = d['normTrainData']
                    abnormalTrainData = d['abnormTrainData']
                    normalTestData    = d['normTestData']
                    abnormalTestData  = d['abnormTestData']
                
            else:
                # dim x sample x length
                normalTrainData   = cv_dict['successData'][:, normalTrainIdx, :] 
                abnormalTrainData = cv_dict['failureData'][:, abnormalTrainIdx, :] 
                normalTestData    = cv_dict['successData'][:, normalTestIdx, :] 
                abnormalTestData  = cv_dict['failureData'][:, abnormalTestIdx, :] 


            if AE_dict['add_option'] is not None:
                print "add feature!!"
                newHandSuccTrData = handSuccTrData = d['handNormTrainData']
                newHandFailTrData = handFailTrData = d['handAbnormTrainData']
                handSuccTeData = d['handNormTestData']
                handFailTeData = d['handAbnormTestData']

                print d['handFeatureNames']
                ## sys.exit()
                normalTrainData   = combineData( normalTrainData, newHandSuccTrData,\
                                                 AE_dict['add_option'], d['handFeatureNames'], \
                                                 add_noise_features=AE_dict['add_noise_option'])
                abnormalTrainData = combineData( abnormalTrainData, newHandFailTrData,\
                                                 AE_dict['add_option'], d['handFeatureNames'])
                normalTestData    = combineData( normalTestData, handSuccTeData,\
                                                 AE_dict['add_option'], d['handFeatureNames'])
                abnormalTestData  = combineData( abnormalTestData, handFailTeData,\
                                                 AE_dict['add_option'], d['handFeatureNames'])

                
                ## pooling_param_dict  = {'dim': AE_dict['filterDim']} # only for AE
                ## normalTrainData, abnormalTrainData,pooling_param_dict \
                ##   = dm.errorPooling(d['normTrainData'], d['abnormTrainData'], pooling_param_dict)
                ## normalTestData, abnormalTestData, _ \
                ##   = dm.errorPooling(d['normTestData'], d['abnormTestData'], pooling_param_dict)
                
                ## normalTrainData, pooling_param_dict = dm.variancePooling(normalTrainData, \
                ##                                                          pooling_param_dict)
                ## abnormalTrainData, _ = dm.variancePooling(abnormalTrainData, pooling_param_dict)
                ## normalTestData, _    = dm.variancePooling(normalTestData, pooling_param_dict)
                ## abnormalTestData, _  = dm.variancePooling(abnormalTestData, pooling_param_dict)                

            # scaling
            if verbose: print "scaling data ", idx, " / ", len(kFold_list)
            normalTrainData   *= param['scale']
            abnormalTrainData *= param['scale']
            normalTestData    *= param['scale']
            abnormalTestData  *= param['scale']

            #
            nEmissionDim = len(normalTrainData)
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
                
            if ret == 'Failure':
                print "fitting failure", param['scale'], param['cov']
                scores.append(-1.0 * 1e+10)
                break

            #-----------------------------------------------------------------------------------------
            # Classifier train data
            #-----------------------------------------------------------------------------------------
            testDataX = []
            testDataY = []
            for i in xrange(nEmissionDim):
                temp = np.vstack([normalTrainData[i], abnormalTrainData[i]])
                testDataX.append( temp )

            testDataY = np.hstack([ -np.ones(len(normalTrainData[0])), \
                                    np.ones(len(abnormalTrainData[0])) ])

            # compute last three indices only
            r = Parallel(n_jobs=n_jobs)(delayed(hmm.computeLikelihoods)(i, ml.A, ml.B, ml.pi, ml.F, \
                                                                    [ testDataX[j][i] for j in xrange(nEmissionDim) ], \
                                                                    ml.nEmissionDim, ml.nState,\
                                                                    startIdx=4, \
                                                                    ## startIdx=nLength-3, \
                                                                    bPosterior=True)
                                                                    for i in xrange(len(testDataX[0])))
            _, ll_idx, ll_logp, ll_post = zip(*r)


            logp_l = []
            for i in xrange(len(normalTrainData[0])):
                logp_l.append(ll_logp[i][-1])
                
            if np.mean( logp_l ) < 0:
                print "Negative likelihoods"
                scores.append(-1.0 * 1e+10)
                ret = 'Failure'
                break


            # split
            import random
            train_idx = random.sample(range(len(ll_logp)), int( 0.5*len(ll_logp)) )
            test_idx  = [x for x in range(len(ll_logp)) if not x in train_idx]

            ll_logp_train = np.array(ll_logp)[train_idx].tolist()
            ll_post_train = np.array(ll_post)[train_idx].tolist()
            ll_idx_train  = np.array(ll_idx)[train_idx].tolist()
            l_label_train = testDataY[train_idx].tolist()
            ll_logp_test = np.array(ll_logp)[test_idx].tolist()
            ll_post_test = np.array(ll_post)[test_idx].tolist()
            l_label_test = testDataY[test_idx].tolist()

            X_train_org, Y_train_org, idx_train_org = \
              hmm.getHMMinducedFlattenFeatures(ll_logp_train, ll_post_train, ll_idx_train,\
                                               l_label_train, \
                                               c=1.0, add_delta_logp=True,\
                                               remove_fp=True, remove_outlier=True)

            if X_train_org == []:
                print "HMM-induced vector is wrong", param['scale'], param['cov']
                scores.append(-1.0 * 1e+10)
                ret = 'Failure'
                break
            
            # nSample x nLength
            test_X, test_Y = \
              hmm.getHMMinducedFeatures(ll_logp_test, ll_post_test, l_label_test, c=1.0, add_delta_logp=True)
            if test_X == []:
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
                for j in xrange(len(test_X)):
                    if len(test_X[j])==0: continue
                    X = scaler.transform(test_X[j])                                

                    X_test.append(X)
                    Y_test.append(test_Y[j])
                weights = ROC_dict['svm_param_range']
            else:
                X_scaled = X_train_org
                X_test = test_X
                Y_test = test_Y
                weights = ROC_dict['progress_param_range']
                

            r = Parallel(n_jobs=n_jobs, verbose=50)(delayed(run_classifiers)(iii, X_scaled, Y_train_org, \
                                                                             idx_train_org, \
                                                                             X_test, Y_test, \
                                                                             nEmissionDim, nLength, \
                                                                             SVM_dict, weight=weights[iii], \
                                                                             method=method,\
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

                                
            ## max_norm_logp = np.amax(norm_logp)
            ## min_norm_logp = np.amin(norm_logp)

            ## ll_norm_logp   = (np.array(ll_norm_logp)-min_norm_logp)/(max_norm_logp-min_norm_logp)
            ## ll_abnorm_logp = (np.array(ll_abnorm_logp)-min_norm_logp)/(max_norm_logp-min_norm_logp)

            ## #
            ## ## import MDAnalysis.analysis.psa as psa
            ## l_mean_logp = np.array([np.mean(ll_norm_logp, axis=0)])
            ## norm_dist = []
            ## abnorm_dist = []
            ## for i in xrange(len(ll_norm_logp)):
            ##     norm_dist.append( np.linalg.norm(l_mean_logp - ll_norm_logp[i:i+1] ) )
            ##     ## norm_dist.append(np.log(psa.hausdorff(l_mean_logp, ll_norm_logp[i:i+1] )))
            ## for i in xrange(len(ll_abnorm_logp)):
            ##     abnorm_dist.append( np.linalg.norm(l_mean_logp - ll_abnorm_logp[i:i+1] ) )
            ##     ## abnorm_dist.append(np.log(psa.hausdorff(l_mean_logp, ll_abnorm_logp[i:i+1] )))

            ## print param['scale'], param['cov'], " : ", np.mean(norm_dist)-np.mean(abnorm_dist), \
            ##   " : ", np.std(norm_dist)-np.std(abnorm_dist) 
            ## scores.append( abs(np.mean(abnorm_dist)/np.mean(norm_dist))/(1.0 + float(nEmissionDim)/3.0*np.std(norm_dist))  )

            #--------------------------------------------------------------
            ## logps = norm_logp + abnorm_logp
            ## if len(logps) == 0:
            ##     scores.append(-100000)
            ##     continue
            ## if np.mean(norm_logp) < 0 or np.amin(norm_logp) < 0:
            ##     continue

            ## # normalization
            ## max_logp     = np.amax(logps) 
            ## norm_logp   /= max_logp
            ## abnorm_logp /= max_logp

            ## # mu, sig
            ## l_mu  = np.mean(norm_logp)
            ## l_sig = np.std(norm_logp)
            ## new_abnorm_logp = [logp for logp in abnorm_logp if logp > 0.0]

            ## from scipy.stats import norm
            ## score = 0.0; c1=300.0; c2=300.0; c3=50. #1.e+2 c8
            ## ## score = 0.0; c1=1000.0; c2=1.0; c3=500. #1.e+2 pc1
            ## ## score = 0.0; c1=1000.0; c2=1.0; c3=1000. #1.e+2 c12
            ## ## score = 0.0; c1=1000.0; c2=1.0; c3=5000. #1.e+2 c11
            ## ## score = 0.0; c1=1000.0; c2=1.0; c3=5. #1.e+2 ep
            ## score += c1/l_sig
            ## score += c2/np.sum([ norm.pdf(logp,loc=l_mu,scale=l_sig) for logp in new_abnorm_logp ])
            ## ## score += c3/max_logp
            ## ## ## abnorm_logp = np.sort(abnorm_logp)[::-1][:len(abnorm_logp)/2]
            ## scores.append( 1000.0*score )

            #--------------------------------------------------------------
            ## # score 1 - c12
            ## diff_vals = -abnorm_logp + np.mean(norm_logp)
            ## diff_list = []
            ## for v in diff_vals:
            ##     if v is np.nan or v is np.inf: continue
            ##     diff_list.append(v)

            ## if len(diff_list)==0: continue
            ## score = np.median(diff_list)
            ## scores.append( score )                                    
            ## print scores

        ## print np.sum(tp_l)+np.sum(fn_l), np.sum(fp_l)+np.sum(tn_l)

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
                    SVM_dict, weight, method='svm',\
                    verbose=False):

    if verbose: print "Run a classifier"
    dtc = cb.classifier( method=method, nPosteriors=nEmissionDim, nLength=nLength )
    dtc.set_params( **SVM_dict )

    if method.find('svm')>=0:
        dtc.set_params( class_weight=weight )
    else:
        dtc.set_params( ths_mult=weight )
        
    ret = dtc.fit(X_scaled, Y_train_org, idx_train_org, parallel=False)
    if ret is False:
        print "SVM fitting failure!!"
        return idx, [np.nan], [np.nan], [np.nan], [np.nan]

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


def tune_hmm_classifier(parameters, kFold_list, param_dict, verbose=True):

    ## Parameters
    # data
    data_dict  = param_dict['data_param']
    # AE
    AE_dict     = param_dict['AE']
    # HMM
    HMM_dict = param_dict['HMM']
    ## nState   = HMM_dict['nState']
    ## cov      = HMM_dict['cov']
    # SVM
    SVM_dict = param_dict['SVM']
    
    #------------------------------------------

    param_list = list(ParameterGrid(parameters))
    startIdx   = 4
    scores     = []
    data       = {}

    # get data
    for idx, (normalTrainIdx, abnormalTrainIdx, normalTestIdx, abnormalTestIdx) \
      in enumerate(kFold_list):
        if AE_dict['switch']:
            if verbose: print "Start "+str(idx)+"/"+str(len(kFold_list))+"th iteration"

            AE_proc_data = os.path.join(processed_data_path, 'ae_processed_data_'+str(idx)+'.pkl')
            d = ut.load_pickle(AE_proc_data)

            if AE_dict['filter']:
                # NOTE: pooling dimension should vary on each auto encoder.
                # Filtering using variances
                normalTrainData   = d['normTrainDataFiltered']
                abnormalTrainData = d['abnormTrainDataFiltered']
                normalTestData    = d['normTestDataFiltered']
                abnormalTestData  = d['abnormTestDataFiltered']
#                ## import data_viz as dv
                ## dv.viz(normalTrainData)
                ## continue                   
            else:
                normalTrainData   = d['normTrainData']
                abnormalTrainData = d['abnormTrainData']
                normalTestData    = d['normTestData']
                abnormalTestData  = d['abnormTestData']

        else:
            # dim x sample x length
            normalTrainData   = successData[:, normalTrainIdx, :] 
            abnormalTrainData = failureData[:, abnormalTrainIdx, :] 
            normalTestData    = successData[:, normalTestIdx, :] 
            abnormalTestData  = failureData[:, abnormalTestIdx, :] 


        if AE_dict['add_option'] is not None:
            print "add feature!!"
            newHandSuccTrData = handSuccTrData = d['handNormTrainData']
            newHandFailTrData = handFailTrData = d['handAbnormTrainData']
            handSuccTeData = d['handNormTestData']
            handFailTeData = d['handAbnormTestData']

            normalTrainData   = combineData( normalTrainData, newHandSuccTrData,\
                                             AE_dict['add_option'], d['handFeatureNames'])
            abnormalTrainData = combineData( abnormalTrainData, newHandFailTrData,\
                                             AE_dict['add_option'], d['handFeatureNames'])
            normalTestData   = combineData( normalTestData, handSuccTeData,\
                                            AE_dict['add_option'], d['handFeatureNames'])
            abnormalTestData  = combineData( abnormalTestData, handFailTeData,\
                                             AE_dict['add_option'], d['handFeatureNames'])


            ## pooling_param_dict  = {'dim': AE_dict['filterDim']} # only for AE
            ## normalTrainData, abnormalTrainData,pooling_param_dict \
            ##   = dm.errorPooling(d['normTrainData'], d['abnormTrainData'], pooling_param_dict)
            ## normalTestData, abnormalTestData, _ \
            ##   = dm.errorPooling(d['normTestData'], d['abnormTestData'], pooling_param_dict)

            ## normalTrainData, pooling_param_dict = dm.variancePooling(normalTrainData, \
            ##                                                          pooling_param_dict)
            ## abnormalTrainData, _ = dm.variancePooling(abnormalTrainData, pooling_param_dict)
            ## normalTestData, _    = dm.variancePooling(normalTestData, pooling_param_dict)
            ## abnormalTestData, _  = dm.variancePooling(abnormalTestData, pooling_param_dict)

        data[idx] = {'normalTrainData': normalTrainData, 'abnormalTrainData': abnormalTrainData, \
                     'normalTestData': normalTestData, 'abnormalTestData': abnormalTestData }


    # Training HMM, and getting classifier training and testing data
    print "Start hmm - classifier"

    idx_list = []
    tp_list  = []
    fp_list  = []
    tn_list  = []
    fn_list  = []
    for param_idx, param in enumerate(param_list):
        for idx in xrange(len(kFold_list)):
            _, tp, fp, tn, fn = run_single_hmm_classifier(param_idx, data[idx], param, HMM_dict, SVM_dict, startIdx, n_jobs=-1)
            idx_list.append(param_idx)
            tp_list.append(tp)
            fp_list.append(fp)
            tn_list.append(tn)
            fn_list.append(fn)
            print "Finished ", param_idx*len(param_list)+idx, " / ", len(param_list)*len(kFold_list)
    
    ## r = Parallel(n_jobs=-1)(delayed(run_single_hmm_classifier)(param_idx, data[idx], param, HMM_dict, SVM_dict, startIdx, n_jobs=1) for idx in xrange(len(kFold_list)) for param_idx, param in enumerate(param_list) )
    ## idx_list, tp_list, fp_list, tn_list, fn_list = zip(*r)

    for i in xrange(len(param_list)):
        tp_l = []
        fn_l = []
        for j, idx in enumerate(idx_list):
            
            if idx == -1:
                print "Failed to fit HMM so ignore!!!"
                break
                
            if i==idx:
                tp_l += tp_list[j]
                fn_l += fn_list[j]

        if idx == -1:
            score.append(-1.0*1e+10)
        else:
            tpr = float(np.sum(tp_l))/float(np.sum(tp_l)+np.sum(fn_l))*100.0
            print "true positive rate : ", tpr
            scores.append( tpr )        

    for i, param in enumerate(param_list):
        print("%0.3f for %r"
              % (scores[i], param))




def run_single_hmm_classifier(param_idx, data, param, HMM_dict, SVM_dict, startIdx, n_jobs=-1, verbose=True):

    print "Start to run classifier with single data ", param_idx
    normalTrainData   = data['normalTrainData'] * param['scale']
    abnormalTrainData = data['abnormalTrainData'] * param['scale']
    normalTestData    = data['normalTestData'] * param['scale']
    abnormalTestData  = data['abnormalTestData'] * param['scale']

    #
    nEmissionDim = len(normalTrainData)
    cov_mult     = [param['cov']]*(nEmissionDim**2)
    nLength      = len(normalTrainData[0][0])

    print "start fit hmm"
    ml = hmm.learning_hmm( param['nState'], nEmissionDim )
    ret = ml.fit( normalTrainData, cov_mult=cov_mult )
    if ret == 'Failure':
        print 'failure with ', param
        return -1, [],[],[],[]
        ## scores.append(-1.0 * 1e+10)
        ## continue

    #-----------------------------------------------------------------------------------------
    # Classifier training data (dim x sample x length)
    #-----------------------------------------------------------------------------------------
    testDataX = np.vstack([ np.swapaxes(normalTrainData, 0, 1), \
                            np.swapaxes(abnormalTrainData, 0, 1) ])
    testDataX = np.swapaxes(testDataX, 0, 1)
    testDataY = np.hstack([ -np.ones(len(normalTrainData[0])), \
                            np.ones(len(abnormalTrainData[0])) ])

    r = Parallel(n_jobs=n_jobs)(delayed(hmm.computeLikelihoods)(i, ml.A, ml.B, ml.pi, ml.F, \
                                                            [ testDataX[j][i] for j in xrange(nEmissionDim) ], \
                                                            ml.nEmissionDim, ml.nState,\
                                                            startIdx=startIdx, \
                                                            bPosterior=True)
                                                            for i in xrange(len(testDataX[0])))
    _, ll_classifier_train_idx, ll_logp, ll_post = zip(*r)

    ll_classifier_train_X = []
    ll_classifier_train_Y = []
    for i in xrange(len(ll_logp)):
        l_X = []
        l_Y = []
        for j in xrange(len(ll_logp[i])):        
            l_X.append( [ll_logp[i][j]] + ll_post[i][j].tolist() )

            if testDataY[i] > 0.0: l_Y.append(1)
            else: l_Y.append(-1)

        ll_classifier_train_X.append(l_X)
        ll_classifier_train_Y.append(l_Y)

    #-----------------------------------------------------------------------------------------
    # Classifier test data (dim x sample x length)
    #-----------------------------------------------------------------------------------------
    testDataX = np.vstack([ np.swapaxes(normalTestData, 0, 1), \
                            np.swapaxes(abnormalTestData, 0, 1) ])
    testDataX = np.swapaxes(testDataX, 0, 1)
    testDataY = np.hstack([ -np.ones(len(normalTestData[0])), \
                            np.ones(len(abnormalTestData[0])) ])

    r = Parallel(n_jobs=n_jobs)(delayed(hmm.computeLikelihoods)(i, ml.A, ml.B, ml.pi, ml.F, \
                                                            [ testDataX[j][i] for j in xrange(nEmissionDim) ], \
                                                            ml.nEmissionDim, ml.nState,\
                                                            startIdx=startIdx, \
                                                            bPosterior=True)
                                                            for i in xrange(len(testDataX[0])))
    _, ll_classifier_test_idx, ll_logp, ll_post = zip(*r)

    # nSample x nLength
    ll_classifier_test_X = []
    ll_classifier_test_Y = []
    for i in xrange(len(ll_logp)):
        l_X = []
        l_Y = []
        for j in xrange(len(ll_logp[i])):        
            l_X.append( [ll_logp[i][j]] + ll_post[i][j].tolist() )

            if testDataY[i] > 0.0: l_Y.append(1)
            else: l_Y.append(-1)

        ll_classifier_test_X.append(l_X)
        ll_classifier_test_Y.append(l_Y)

    #-----------------------------------------------------------------------------------------
    # 
    #-----------------------------------------------------------------------------------------
    # flatten the data
    X_train = []
    Y_train = []
    idx_train = []
    for i in xrange(len(ll_classifier_train_X)):
        for j in xrange(len(ll_classifier_train_X[i])):
            X_train.append(ll_classifier_train_X[i][j])
            Y_train.append(ll_classifier_train_Y[i][j])
            idx_train.append(ll_classifier_train_idx[i][j])


    print "Start fit the classifier"
    dtc = cb.classifier( method='svm', nPosteriors=ml.nState, nLength=nLength )        
    dtc.set_params( **SVM_dict )
    ret = dtc.fit(X_train, Y_train, idx_train)


    tp_l = []
    fp_l = []
    tn_l = []
    fn_l = []            

    # We should maximize score,
    #   if normal, error is close to 0
    #   if abnormal, error is large
    for i in xrange(len(ll_classifier_test_X)):
        X     = ll_classifier_test_X[i]
        try:
            est_y = dtc.predict(X, y=ll_classifier_test_Y[i])
        except:
            continue

        for j in xrange(len(est_y)):
            if est_y[j] > 0.0:
                break

        if ll_classifier_test_Y[i][0] > 0.0:
            if est_y[j] > 0.0: tp_l.append(1)
            else: fn_l.append(1)
        elif ll_classifier_test_Y[i][0] <= 0.0:
            if est_y[j] > 0.0: fp_l.append(1)
            else: tn_l.append(1)

    print param_idx, param, len(tp_l), len(fn_l)
    return param_idx, tp_l, fp_l, tn_l, fn_l



        


if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    p.add_option('--task', action='store', dest='task', type='string', default='pushing_microwhite',
                 help='type the desired task name')
    p.add_option('--dim', action='store', dest='dim', type=int, default=3,
                 help='type the desired dimension')
    p.add_option('--n_jobs', action='store', dest='n_jobs', type=int, default=-1,
                 help='number of processes for multi processing')
    p.add_option('--add_logp_d', '--ld', action='store_false', dest='bAddLogpD',
                 default=True, help='Enable AE data.')
    p.add_option('--aeswtch', '--aesw', action='store_true', dest='bAESwitch',
                 default=False, help='Enable AE data.')
    p.add_option('--method', '--m', action='store', dest='method', type='string', default='svm',
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
                                                              nPoints=10)


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
                parameters = {'nState': [25], 'scale': np.linspace(3.0,10.0,10), \
                              'cov': np.linspace(1.0,10.0,10) }

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
            parameters = {'nState': [25], 'scale': np.linspace(1.0,8.0,10), \
                          'cov': np.linspace(0.5,4.0,10) }
        else:
            print "Not available task"

    else:

        from hrl_anomaly_detection.ICRA2017_params import *
        raw_data_path, save_data_path, param_dict = getParams(opt.task, False, \
                                                              False, False, opt.dim,\
                                                              rf_center, local_range, \
                                                              bAESwitch=opt.bAESwitch, \
                                                              nPoints=8)
        parameters = {'nState': [25], 'scale': np.linspace(9.0,15.0,10), \
                      'cov': np.linspace(1.0,1.0,5) }

        ## save_data_path = os.path.expanduser('~')+\
        ##   '/hrl_file_server/dpark_data/anomaly/ICRA2017/'+opt.task+'_data_online_hmm/'+\
        ##   str(param_dict['data_param']['downSampleSize'])+'_'+str(opt.dim)

    max_check_fold = None
    ## max_check_fold = 2

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
             bSave=opt.bSave, method=opt.method, max_check_fold=max_check_fold)
    ## tune_hmm_classifier(parameters, kFold_list, param_dict, verbose=True)
