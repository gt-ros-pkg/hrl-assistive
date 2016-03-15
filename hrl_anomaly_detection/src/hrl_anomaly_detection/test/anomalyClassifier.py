#!/usr/bin/env python

import sys, random
import numpy as np
from sklearn import preprocessing
from joblib import Parallel, delayed
from hrl_anomaly_detection import util as dataUtil
from hrl_anomaly_detection import data_manager as dataMng
from hrl_anomaly_detection.classifiers import classifier as cb
from hrl_anomaly_detection.hmm import learning_hmm

downSampleSize = 300
cov_mult = 10.0
nState = 10
scale = 2.5
classifier_method = 'svm'
rf_center = 'kinEEPos'
local_range = 0.3

feature_list = ['unimodal_audioPower',
                # 'unimodal_audioWristRMS',
                'unimodal_kinVel',
                'unimodal_ftForce',
                'unimodal_ppsForce',
                # 'unimodal_visionChange',
                'unimodal_fabricForce',
                'crossmodal_targetEEDist',
                'crossmodal_targetEEAng',
                'crossmodal_artagEEDist']
                # 'crossmodal_artagEEAng']


subject = 'gatsbii'
task = 'scooping'
save_data_path = '/home/mycroft/RSS2016/' + task + '_data'
raw_data_path = '/home/mycroft/RSS2016'


# Load Data
print 'Loading data'
_, successData, failureData, _, _ = dataMng.getDataSet([subject], task, raw_data_path, save_data_path,
                                                   rf_center, local_range, downSampleSize=downSampleSize,
                                                   scale=scale, feature_list=feature_list)

print np.shape(successData), np.shape(failureData)
#successData = successData[:10]
#failureData = failureData[:10]

kFold_list = dataMng.kFold_data_index(len(failureData[0]), len(successData[0]), 2, 3)
print np.shape(kFold_list)
trainIdx, normalClassifierIdx, abnormalClassifierIdx, normalTestIdx, abnormalTestIdx = kFold_list[0]


trainingData           = successData[:, trainIdx, :]
normalClassifierData   = successData[:, normalClassifierIdx, :]
abnormalClassifierData = failureData[:, abnormalClassifierIdx, :]
normalTestData         = successData[:, normalTestIdx, :]
abnormalTestData       = failureData[:, abnormalTestIdx, :]

print "======================================"
print "HMM training data: ", np.shape(trainingData)
print "Normal classifier training data: ", np.shape(normalClassifierData)
print "Abnormal classifier training data: ", np.shape(abnormalClassifierData)
print "Normal classifier testing data: ", np.shape(normalTestData)
print "Abnormal classifier testing data: ", np.shape(abnormalTestData)
print "======================================"

# Train HMM
nEmissionDim = len(trainingData)
hmm = learning_hmm.learning_hmm(nState, nEmissionDim, verbose=False)
ret = hmm.fit(trainingData, cov_mult=[cov_mult]*nEmissionDim**2)

if ret == 'Failure':
    print "-------------------------"
    print "HMM returned failure!!   "
    print "-------------------------"
    sys.exit()

print 'HMM trained'




#-----------------------------------------------------------------------------------------
# Classifier training data
#-----------------------------------------------------------------------------------------
print 'Processing classifier training data'
testDataX = []
for i in xrange(nEmissionDim):
    temp = np.vstack([normalClassifierData[i], abnormalClassifierData[i]])
    testDataX.append( temp )

testDataY = np.hstack([ -np.ones(len(normalClassifierData[0])),
                        np.ones(len(abnormalClassifierData[0])) ])

startIdx = 4
r = Parallel(n_jobs=-1)(delayed(learning_hmm.computeLikelihoods)(i, hmm.A, hmm.B, hmm.pi, hmm.F,
                                                        [ testDataX[j][i] for j in xrange(nEmissionDim) ],
                                                        hmm.nEmissionDim, hmm.nState,
                                                        startIdx=startIdx, bPosterior=True)
                                                        for i in xrange(len(testDataX[0])))
_, ll_classifier_train_idx, ll_logp, ll_post = zip(*r)
print 'Data processed by HMM'

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


# flatten the data
X_train_org = []
Y_train_org = []
idx_train_org = []
for i in xrange(len(ll_classifier_train_X)):
    for j in xrange(len(ll_classifier_train_X[i])):
        X_train_org.append(ll_classifier_train_X[i][j])
        Y_train_org.append(ll_classifier_train_Y[i][j])
        idx_train_org.append(ll_classifier_train_idx[i][j])


# data preparation
scaler = preprocessing.StandardScaler()
if 'svm' in classifier_method:
    X_scaled = scaler.fit_transform(X_train_org)
else:
    X_scaled = X_train_org
print classifier_method, " : Before classification : ", np.shape(X_scaled), np.shape(Y_train_org)

# Train Classifier
classifier = cb.classifier(method=classifier_method, nPosteriors=nState, nLength=len(trainingData[0,0]))

# nPoints = 10
# if method == 'svm':
#     weights = np.logspace(-2, 0.1, nPoints)
#     classifier.set_params( class_weight=weights[j] )
# elif method == 'cssvm_standard':
#     weights = np.logspace(-2, 0.1, nPoints)
#     classifier.set_params( class_weight=weights[j] )
# elif method == 'cssvm':
#     weights = np.logspace(0.0, 2.0, nPoints)
#     classifier.set_params( class_weight=weights[j] )
# elif method == 'progress_time_cluster':
#     ## thresholds = -np.linspace(1., 50, nPoints)+2.0
#     thresholds = -np.linspace(1., 4, nPoints)+2.0
#     classifier.set_params( ths_mult = thresholds[j] )
# elif method == 'fixed':
#     thresholds = np.linspace(1., -3, nPoints)
#     classifier.set_params( ths_mult = thresholds[j] )

ret = classifier.fit(X_scaled, Y_train_org, idx_train_org)
print 'Classifier trained'


#-----------------------------------------------------------------------------------------
# Classifier test data
#-----------------------------------------------------------------------------------------
print 'Processing classifier testing data'
testDataX = []
for i in xrange(nEmissionDim):
    temp = np.vstack([normalTestData[i], abnormalTestData[i]])
    testDataX.append( temp )

testDataY = np.hstack([ -np.ones(len(normalTestData[0])),
                        np.ones(len(abnormalTestData[0])) ])

r = Parallel(n_jobs=-1)(delayed(learning_hmm.computeLikelihoods)(i, hmm.A, hmm.B, hmm.pi, hmm.F,
                                                        [ testDataX[j][i] for j in xrange(nEmissionDim) ],
                                                        hmm.nEmissionDim, hmm.nState,
                                                        startIdx=startIdx, bPosterior=True)
                                                        for i in xrange(len(testDataX[0])))
_, ll_classifier_test_idx, ll_logp, ll_post = zip(*r)
print 'Data processed by HMM'

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

print 'Shape of classifier test data'
print np.shape(ll_classifier_test_X), np.shape(ll_classifier_test_Y)
print len(ll_classifier_test_X), len(ll_classifier_test_X[0])
print ll_classifier_test_X[0][0]

# evaluate the classifier
tp_l = []
fp_l = []
tn_l = []
fn_l = []
delay_l = []
delay_idx = 0

print '-'*15, 'Evaluating Classifier', '-'*15
for ii in xrange(len(ll_classifier_test_X)):
    for jj in xrange(len(ll_classifier_test_X[ii])):
        if 'svm' in classifier_method:
            X = scaler.transform([ll_classifier_test_X[ii][jj]])
        elif classifier_method == 'progress_time_cluster' or classifier_method == 'fixed':
            X = ll_classifier_test_X[ii][jj]
        else:
            print 'Invalid classifier method. Exiting.'
            exit()

        est_y = classifier.predict(X, y=ll_classifier_test_Y[ii][jj:jj+1])
        if type(est_y) == list: est_y = est_y[0]
        if type(est_y) == list: est_y = est_y[0]

        if est_y > 0.0:
            print '-'*15, 'Anomaly has occured!', '-'*15
            delay_idx = ll_classifier_test_idx[ii][jj]
            print "Break", ii, jj, "in", est_y, "=", ll_classifier_test_Y[ii][jj]
            break

    if ll_classifier_test_Y[ii][0] > 0.0:
        if est_y > 0.0:
            tp_l.append(1)
            delay_l.append(delay_idx)
        else: fn_l.append(1)
    elif ll_classifier_test_Y[ii][0] <= 0.0:
        if est_y > 0.0: fp_l.append(1)
    else: tn_l.append(1)


print 'True pos:', tp_l
print 'False pos:', fp_l
print 'True neg:', tn_l
print 'False Negative:', fn_l
print 'Delay:', delay_l
print 'Delay index:', delay_idx

