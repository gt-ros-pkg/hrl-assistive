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

def tune_hmm(parameters, kFold_list, filtering=True):

    # sample x dim x length
    param_list = list(ParameterGrid(parameters))
    mean_list  = []
    std_list   = []
    
    for param in param_list:

        scores = []
        # Training HMM, and getting classifier training and testing data
        for idx, (normalTrainIdx, abnormalTrainIdx, normalTestIdx, abnormalTestIdx) \
          in enumerate(kFold_list):

            AE_proc_data = os.path.join(processed_data_path, 'ae_processed_data_'+str(idx)+'.pkl')
            if os.path.isfile(AE_proc_data):
                d = ut.load_pickle(AE_proc_data)
                if filtering:
                    normalTrainData = d['normTrainDataFiltered'] * param['scale']
                    abnormalTrainData = d['abnormTrainDataFiltered'] * param['scale']
                    normalTestData  = d['normTestDataFiltered'] * param['scale']
                    abnormalTestData  = d['abnormTestDataFiltered'] * param['scale']
                else:
                    normalTrainData = d['normTrainData'] * param['scale']
                    abnormalTrainData = d['abnormTrainData'] * param['scale']
                    normalTestData  = d['normTestData'] * param['scale']
                    abnormalTestData  = d['abnormTestData'] * param['scale']

            #
            nEmissionDim = len(normalTrainData)
            cov_mult     = [param['cov']]*(nEmissionDim**2)
            nLength      = len(normalTrainData[0][0])

            # scaling
            model = hmm.learning_hmm( param['nState'], nEmissionDim )
            ret = model.fit( normalTrainData, cov_mult=cov_mult )
            if ret == 'Failure':
                scores.append(-1.0 * 1e+10)
            else:
                # evaluation:  dim x sample => sample x dim
                testData_x = np.vstack([ np.swapaxes( normalTestData, 0, 1),
                                         np.swapaxes( abnormalTestData, 0, 1) ])                                     
                testData_x = np.swapaxes( testData_x, 0, 1) #dim x sample
                testData_y = [1.0]*len( normalTestData[0] ) + [-1]*len( abnormalTestData[0] )
                scores.append( model.score( testData_x, y=testData_y, n_jobs=-1 ) )


        mean_list.append( np.mean(scores) )
        std_list.append( np.std(scores) )


    for i, param in enumerate(param_list):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_list[i], std_list[i], param))


def tune_hmm_progress(parameters, kFold_list, filtering=True):

    param_list = list(ParameterGrid(parameters))
    mean_list  = []
    std_list   = []
    
    for param in param_list:

        scores = []
        # Training HMM, and getting classifier training and testing data
        for idx, (normalTrainIdx, abnormalTrainIdx, normalTestIdx, abnormalTestIdx) \
          in enumerate(kFold_list):

            AE_proc_data = os.path.join(processed_data_path, 'ae_processed_data_'+str(idx)+'.pkl')
            if os.path.isfile(AE_proc_data):
                d = ut.load_pickle(AE_proc_data)
                # dim x sample x length
                if filtering:
                    normalTrainData = d['normTrainDataFiltered'] * param['scale']
                    abnormalTrainData = d['abnormTrainDataFiltered'] * param['scale']
                    normalTestData  = d['normTestDataFiltered'] * param['scale']
                    abnormalTestData  = d['abnormTestDataFiltered'] * param['scale']
                else:
                    normalTrainData = d['normTrainData'] * param['scale']
                    abnormalTrainData = d['abnormTrainData'] * param['scale']
                    normalTestData  = d['normTestData'] * param['scale']
                    abnormalTestData  = d['abnormTestData'] * param['scale']

            #
            nEmissionDim = len(normalTrainData)
            cov_mult     = [param['cov']]*(nEmissionDim**2)
            nLength      = len(normalTrainData[0][0])

            # scaling
            model = hmm.learning_hmm( param['nState'], nEmissionDim )
            ret = model.fit( normalTrainData, cov_mult=cov_mult )
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

            r = Parallel(n_jobs=-1)(delayed(hmm.computeLikelihoods)(i, ml.A, ml.B, ml.pi, ml.F, \
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

            r = Parallel(n_jobs=-1)(delayed(hmm.computeLikelihoods)(i, ml.A, ml.B, ml.pi, ml.F, \
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


            dtc = cb.classifier( method='progress_time_cluster', nPosteriors=ml.nState, nLength=nLength )        
            ret = dtc.fit(X_train, Y_train, idx_train)

            # We should maximize score,
            #   if normal, error is close to 0
            #   if abnormal, error is large
            score = 0.0
            for i in xrange(len(ll_classifier_test_X)):
                for j in xrange(len(ll_classifier_test_X[ii])):
                    err = abs( dtc.predict(ll_classifier_test_X[i][j]) )
                    y = ll_classifier_test_Y[i][j] * -1.0   
                    score +=  -np.log(err) * y
                    
            scores.append( score )        
        mean_list.append( np.mean(scores) )
        std_list.append( np.std(scores) )


    for i, param in enumerate(param_list):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_list[i], std_list[i], param))


    


if __name__ == '__main__':

    task_name           = 'pushing'
    processed_data_path = os.path.expanduser('~')+'/hrl_file_server/dpark_data/anomaly/RSS2016/'\
      +task_name+'_data/AE'
    filtering           = True
    
    parameters = {'nState': [10, 15, 20, 25, 30], 'scale':np.arange(1.0, 10.0, 1.0), \
                  'cov': [1.0, 2.0, 4.0, 8.0] }

    #--------------------------------------------------------------------------------------
    crossVal_pkl        = os.path.join(processed_data_path, 'cv_'+task_name+'.pkl')
    if os.path.isfile(crossVal_pkl):
        d = ut.load_pickle(crossVal_pkl)
        successData = d['successData']
        failureData = d['failureData']
        aug_successData = d['aug_successData']
        aug_failureData = d['aug_failureData']
        kFold_list  = d['kFoldList']
    else:
        print "no existing data file"
        sys.exit()


    ## tune_hmm(parameters, kFold_list, filtering=filtering)
    tune_hmm_progress(parameters, kFold_list, filtering=filtering)
