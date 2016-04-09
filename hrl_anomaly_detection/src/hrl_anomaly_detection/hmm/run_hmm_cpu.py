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

from hrl_anomaly_detection.hmm import learning_hmm as hmm
from hrl_anomaly_detection import data_manager as dm
import hrl_lib.util as ut
from hrl_anomaly_detection.util import *
from hrl_anomaly_detection.classifiers import classifier as cb

from joblib import Parallel, delayed

def tune_hmm(parameters, kFold_list, param_dict, verbose=False):

    ## Parameters
    # data
    data_dict  = param_dict['data_param']
    # AE
    AE_dict     = param_dict['AE']
    # HMM
    HMM_dict = param_dict['HMM']
    nState   = HMM_dict['nState']
    cov      = HMM_dict['cov']
    # SVM
    
    #------------------------------------------

    # sample x dim x length
    param_list = list(ParameterGrid(parameters))
    mean_list  = []
    std_list   = []
    
    for param in param_list:

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
                normalTrainData   = successData[:, normalTrainIdx, :] 
                abnormalTrainData = failureData[:, abnormalTrainIdx, :] 
                normalTestData    = successData[:, normalTestIdx, :] 
                abnormalTestData  = failureData[:, abnormalTestIdx, :] 


            if AE_dict['add_option'] is 'featureToBottleneck':
                print "add feature!!"
                newHandSuccTrData = handSuccTrData = d['handNormTrainData']
                newHandFailTrData = handFailTrData = d['handAbnormTrainData']
                handSuccTeData = d['handNormTestData']
                handFailTeData = d['handAbnormTestData']

                for i in xrange(AE_dict['nAugment']):
                    newHandSuccTrData = stackSample(newHandSuccTrData, handSuccTrData)
                    newHandFailTrData = stackSample(newHandFailTrData, handFailTrData)

                normalTrainData   = combineData( normalTrainData, newHandSuccTrData )
                abnormalTrainData = combineData( abnormalTrainData, newHandFailTrData )
                normalTestData   = combineData( normalTestData, handSuccTeData )
                abnormalTestData  = combineData( abnormalTestData, handFailTeData )
                ## print np.shape(normalTrainData), np.shape(normalTestData), np.shape(abnormalTestData)
                ## sys.exit()

                
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
            if verbose: print "scaling data"
            normalTrainData   *= HMM_dict['scale']
            abnormalTrainData *= HMM_dict['scale']
            normalTestData    *= HMM_dict['scale']
            abnormalTestData  *= HMM_dict['scale']

            #
            nEmissionDim = len(normalTrainData)
            cov_mult     = [param['cov']]*(nEmissionDim**2)
            nLength      = len(normalTrainData[0][0])

            # scaling
            ml = hmm.learning_hmm( param['nState'], nEmissionDim )
            ret = ml.fit( normalTrainData, cov_mult=cov_mult )
            if ret == 'Failure':
                scores.append(-1.0 * 1e+10)
            else:           
                # evaluation:  dim x sample => sample x dim
                ## testData_x = np.swapaxes( normalTestData, 0, 1)
                testData_x = np.vstack([ np.swapaxes( normalTestData, 0, 1),
                                         np.swapaxes( abnormalTestData, 0, 1) ])
                testData_x = np.swapaxes( testData_x, 0, 1) #dim x sample
                                         
                ## testData_y = [1.0]*len( normalTestData[0] )
                testData_y = [1.0]*len( normalTestData[0] ) + [-1]*len( abnormalTestData[0] )
                scores.append( ml.score( testData_x, y=testData_y, n_jobs=-1 ) )


        mean_list.append( np.mean(scores) )
        std_list.append( np.std(scores) )


    for i, param in enumerate(param_list):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_list[i], std_list[i], param))


    ## # Get sorted results
    ## from operator import itemgetter
    ## mean_list.sort(key=itemgetter(0), reverse=False)

    ## for i in xrange(len(results)):
    ##     print results[i]



def tune_hmm_classifier(parameters, kFold_list, param_dict, verbose=True):

    ## Parameters
    # data
    data_dict  = param_dict['data_param']
    # AE
    AE_dict     = param_dict['AE']
    # HMM
    HMM_dict = param_dict['HMM']
    nState   = HMM_dict['nState']
    cov      = HMM_dict['cov']
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
            normalTrainData   = successData[:, normalTrainIdx, :] 
            abnormalTrainData = failureData[:, abnormalTrainIdx, :] 
            normalTestData    = successData[:, normalTestIdx, :] 
            abnormalTestData  = failureData[:, abnormalTestIdx, :] 


        if AE_dict['add_option'] is 'featureToBottleneck':
            print "add feature!!"
            newHandSuccTrData = handSuccTrData = d['handNormTrainData']
            newHandFailTrData = handFailTrData = d['handAbnormTrainData']
            handSuccTeData = d['handNormTestData']
            handFailTeData = d['handAbnormTestData']

            for i in xrange(AE_dict['nAugment']):
                newHandSuccTrData = stackSample(newHandSuccTrData, handSuccTrData)
                newHandFailTrData = stackSample(newHandFailTrData, handFailTrData)

            normalTrainData   = combineData( normalTrainData, newHandSuccTrData )
            abnormalTrainData = combineData( abnormalTrainData, newHandFailTrData )
            normalTestData   = combineData( normalTestData, handSuccTeData )
            abnormalTestData  = combineData( abnormalTestData, handFailTeData )
            ## print np.shape(normalTrainData), np.shape(normalTestData), np.shape(abnormalTestData)
            ## sys.exit()


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
    r = Parallel(n_jobs=-1)(delayed(run_single_hmm_classifier)(param_idx, data[idx], param, HMM_dict, SVM_dict, n_jobs=1) for idx in xrange(len(kFold_list)) for param_idx, param in enumerate(param_list) )
    idx_list, tp_list, fp_list, tn_list, fn_list = zip(*r)

    for i in xrange(len(param_list))
        tp_l = []
        fn_l = []
        for j, idx in enumerate(idx_list):
            if i==idx:
                tp_l += tp_list[j]
                fn_l += fn_list[j]

        tpr = float(np.sum(tp_l))/float(np.sum(tp_l)+np.sum(fn_l))*100.0
        print "true positive rate : ", tpr
        scores.append( tpr )        

    for i, param in enumerate(param_list):
        print("%0.3f for %r"
              % (scores[i], param))




def run_single_hmm_classifier(param_idx, data, param, HMM_dict, SVM_dict, n_jobs=-1):

    normalTrainData   = data['normalTrainData']
    abnormalTrainData = data['abnormalTrainData']
    normalTestData    = data['normalTestData']
    abnormalTestData  = data['abnormalTestData']

    # scaling
    if verbose: print "scaling data"
    normalTrainData   *= HMM_dict['scale']
    abnormalTrainData *= HMM_dict['scale']
    normalTestData    *= HMM_dict['scale']
    abnormalTestData  *= HMM_dict['scale']

    #
    nEmissionDim = len(normalTrainData)
    cov_mult     = [param['cov']]*(nEmissionDim**2)
    nLength      = len(normalTrainData[0][0])

    # scaling
    ml = hmm.learning_hmm( param['nState'], nEmissionDim )
    ret = ml.fit( normalTrainData, cov_mult=cov_mult )
    if ret == 'Failure':
        scores.append(-1.0 * 1e+10)
        continue

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

    print param_idx
    return param_idx, tp_l, fp_l, tn_l, fn_l



        


if __name__ == '__main__':
    rf_center     = 'kinEEPos'        
    scale         = 1.0
    # Dectection TEST 
    local_range    = 10.0    

    subjects  = ['gatsbii']
    task_name = 'pushing'

    handFeatures = ['unimodal_ftForce',\
                    'crossmodal_targetEEDist',\
                    'crossmodal_targetEEAng',\
                    'unimodal_audioWristRMS'] #'unimodal_audioPower', ,
    rawFeatures = ['relativePose_artag_EE', \
                   'relativePose_artag_artag', \
                   'wristAudio', \
                   'ft' ]       

    ## processed_data_path = os.path.expanduser('~')+\
    ##   '/hrl_file_server/dpark_data/anomaly/RSS2016/'+task_name+'_data/AE'        
    ## downSampleSize = 100
    ## layers = [64,4]

    processed_data_path = os.path.expanduser('~')+\
      '/hrl_file_server/dpark_data/anomaly/RSS2016/'+task_name+'_data/AE150'        
    downSampleSize = 150
    layers = [64,8]


    data_param_dict= {'renew': False, 'rf_center': rf_center, 'local_range': local_range,\
                      'downSampleSize': downSampleSize, 'cut_data': [0,downSampleSize], \
                      'nNormalFold':3, 'nAbnormalFold':3,\
                      'handFeatures': handFeatures, 'lowVarDataRemv': False }
    AE_param_dict  = {'renew': False, 'switch': True, 'time_window': 4, 'filter': True, \
                      'layer_sizes':layers, 'learning_rate':1e-6, \
                      'learning_rate_decay':1e-6, \
                      'momentum':1e-6, 'dampening':1e-6, 'lambda_reg':1e-6, \
                      'max_iteration':30000, 'min_loss':0.1, 'cuda':True, \
                      'filter':False, 'filterDim':4, \
                      'nAugment': 1, \
                      'add_option': None, 'rawFeatures': rawFeatures}
                      ## 'add_option': 'featureToBottleneck', 'rawFeatures': rawFeatures}
                      ##'add_option': True, 'rawFeatures': rawFeatures}
    HMM_param_dict = {'renew': False, 'nState': 25, 'cov': 4.0, 'scale': 5.0}
    SVM_param_dict = {'renew': False, 'w_negative': 6.0, 'gamma': 0.173, 'cost': 4.0, class_weight=0.001}

    param_dict = {'data_param': data_param_dict, 'AE': AE_param_dict, 'HMM': HMM_param_dict, \
                  'SVM': SVM_param_dict}
    
    parameters = {'nState': [20, 25, 30], 'scale':np.arange(1.0, 10.0, 1.0), \
                  'cov': [1.0, 2.0, 4.0, 8.0] }
    ## parameters = {'nState': [20, 25, 30], 'scale':np.arange(4.0, 6.0, 1.0), \
    ##               'cov': [4.0, 8.0] }

    #--------------------------------------------------------------------------------------
    crossVal_pkl        = os.path.join(processed_data_path, 'cv_'+task_name+'.pkl')
    if os.path.isfile(crossVal_pkl):
        d = ut.load_pickle(crossVal_pkl)
        kFold_list  = d['kFoldList']
    else:
        print "no existing data file"
        sys.exit()

    ## tune_hmm(parameters, kFold_list, param_dict, verbose=True)
    tune_hmm_classifier(parameters, kFold_list, param_dict, verbose=True)
