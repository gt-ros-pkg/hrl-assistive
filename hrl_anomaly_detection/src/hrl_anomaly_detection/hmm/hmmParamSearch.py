#!/usr/bin/env python

import random
from scipy.stats import randint as sp_randint
from sklearn.grid_search import RandomizedSearchCV, GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin
from learning_hmm_multi_n import learning_hmm_multi_n
from util import *
import learning_util as util

class HmmClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, downSampleSize=200, scale=10, nState=20, cov_mult=1.0, isScooping=True):
        self.downSampleSize = downSampleSize
        self.scale = scale
        self.nState = nState
        self.cov_mult = cov_mult
        # print 'Testing', downSampleSize, scale, nState, cov_mult

        self.train_cutting_ratio = [0.0, 0.65]
        self.anomaly_offset = 0.0
        self.isScooping = isScooping
        self.verbose = False

        self.isFitted = False

        self.hmm = None
        self.minVals = None
        self.maxVals = None
        self.normalTestData = None
        self.abnormalTestData = None
        self.minThresholds = None

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
            print parameter, value
        return self

    def fit(self, X=None, y=None):
        """
        This should fit classifier. All the "work" should be done here.

        Note: assert is not a good choice here and you should rather
        use try/except blog with exceptions. This is just for short syntax.
        """

        if self.isScooping:
            subject_names = ['pr2']
            task_name = 'scooping'
        else:
            subject_names = ['s2', 's3', 's4']
            task_name = 'feeding'

        # Loading success and failure data
        root_path = '/home/mycroft/feeding'
        success_list, failure_list = getSubjectFileList(root_path, subject_names, task_name)

        trainDataTrue, thresTestDataTrue, normalTestDataTrue, abnormalTestDataTrue, trainTimeList, \
        thresTestTimeList, normalTestTimeList, abnormalTestTimeList = self.getData(success_list, failure_list)

        # minimum and maximum vales for scaling from Daehyung
        dataList, _ = loadData(success_list, isTrainingData=False, downSampleSize=self.downSampleSize)
        self.minVals = []
        self.maxVals = []
        for modality in dataList:
            self.minVals.append(np.min(modality))
            self.maxVals.append(np.max(modality))

        # Scale data
        trainData, _, _ = self.scaleData(trainDataTrue, minVals=self.minVals, maxVals=self.maxVals)
        thresTestData, _ , _ = self.scaleData(thresTestDataTrue, minVals=self.minVals, maxVals=self.maxVals)
        self.normalTestData, _ , _ = self.scaleData(normalTestDataTrue, minVals=self.minVals, maxVals=self.maxVals)
        self.abnormalTestData, _ , _ = self.scaleData(abnormalTestDataTrue, minVals=self.minVals, maxVals=self.maxVals)

        # cutting data (only training and thresTest data)
        start_idx = int(float(len(trainData[0][0]))*self.train_cutting_ratio[0])
        end_idx = int(float(len(trainData[0][0]))*self.train_cutting_ratio[1])

        for j in xrange(len(trainData)):
            for k in xrange(len(trainData[j])):
                trainData[j][k] = trainData[j][k][start_idx:end_idx]
                trainTimeList[k] = trainTimeList[k][start_idx:end_idx]

        for j in xrange(len(thresTestData)):
            for k in xrange(len(thresTestData[j])):
                thresTestData[j][k] = thresTestData[j][k][start_idx:end_idx]
                thresTestTimeList[k] = thresTestTimeList[k][start_idx:end_idx]

        # hmm = learning_hmm_multi_4d(nState=nState, nEmissionDim=4, anomaly_offset=anomaly_offset, verbose=verbose)
        self.hmm = learning_hmm_multi_n(nState=self.nState, nEmissionDim=4, anomaly_offset=self.anomaly_offset, verbose=self.verbose)
        ret = self.hmm.fit(xData=trainData, cov_mult=[self.cov_mult]*16)

        if ret == 'Failure':
            print 'HMM return was a failure!'
            return self

        with suppress_output():
            # minThresholds = tuneSensitivityGain(hmm, thresTestData, verbose=verbose)
            # #thresTest is not enough, so we also use training data.
            minThresholds1 = tuneSensitivityGain(self.hmm, trainData, verbose=self.verbose)
            minThresholds2 = tuneSensitivityGain(self.hmm, thresTestData, verbose=self.verbose)
            minThresholds3 = tuneSensitivityGain(self.hmm, self.normalTestData, verbose=self.verbose)
            self.minThresholds = minThresholds3
            for i in xrange(len(minThresholds1)):
                if minThresholds1[i] < self.minThresholds[i]:
                    self.minThresholds[i] = minThresholds1[i]
                if minThresholds2[i] < self.minThresholds[i]:
                    self.minThresholds[i] = minThresholds2[i]

        self.isFitted = True

        return self

    def score(self, X, y, sample_weight=None):
        if not self.isFitted:
            return 0
        c = self.minThresholds
        truePos = 0
        trueNeg = 0
        falsePos = 0
        falseNeg = 0

        # positive is anomaly
        # negative is non-anomaly
        if self.verbose: print '\nBeginning anomaly testing for test set\n'

        # for normal test data
        for i in xrange(len(self.normalTestData[0])):
            if self.verbose: print 'Anomaly Error for test set %d' % i

            for j in range(2, len(self.normalTestData[0][i])):
                with suppress_output():
                    anomaly, error = self.hmm.anomaly_check([x[i][:j] for x in self.normalTestData], c)

                if self.verbose: print anomaly, error

                # This is a successful nonanomalous attempt
                if anomaly:
                    falsePos += 1
                    print 'Success Test', i,',',j, ' in ',len(self.normalTestData[0][i]), ' |', anomaly, error
                    break
                elif not anomaly and j == len(self.normalTestData[0][i]) - 1:
                    trueNeg += 1
                    break

        # for abnormal test data
        for i in xrange(len(self.abnormalTestData[0])):
            if self.verbose: print 'Anomaly Error for test set %d' % i

            for j in range(2, len(self.abnormalTestData[0][i])):
                with suppress_output():
                    anomaly, error = self.hmm.anomaly_check([x[i][:j] for x in self.abnormalTestData], c)

                if self.verbose: print anomaly, error


                else:
                    if anomaly:
                        truePos += 1
                        break
                    elif not anomaly and j == len(self.abnormalTestData[0][i]) - 1:
                        falseNeg += 1
                        print 'Failure Test', i,',',j, ' in ',len(self.abnormalTestData[0][i]), ' |', anomaly, error
                        break

        truePositiveRate = float(truePos) / float(truePos + falseNeg) * 100.0
        trueNegativeRate = float(trueNeg) / float(trueNeg + falsePos) * 100.0
        print 'True Negative Rate:', trueNegativeRate, 'True Positive Rate:', truePositiveRate

        return truePositiveRate + trueNegativeRate
        # return truePos + trueNeg



    def getData(self, success_list, failure_list, folding_ratio=(0.5, 0.3, 0.2)):

        if self.verbose:
            print "--------------------------------------------"
            print "# of Success files: ", len(success_list)
            print "# of Failure files: ", len(failure_list)
            print "--------------------------------------------"

        # random training, threshold-test, test set selection
        nTrain   = int(len(success_list) * folding_ratio[0])
        nThsTest = int(len(success_list) * folding_ratio[1])
        nTest    = len(success_list) - nTrain - nThsTest

        if len(failure_list) < nTest:
            print 'Not enough failure data'
            print 'Number of successful test iterations:', nTest
            print 'Number of failure test iterations:', len(failure_list)
            exit()

        # index selection
        success_idx  = range(len(success_list))
        failure_idx  = range(len(failure_list))
        train_idx    = random.sample(success_idx, nTrain)
        ths_test_idx = random.sample([x for x in success_idx if not x in train_idx], nThsTest)
        success_test_idx = [x for x in success_idx if not (x in train_idx or x in ths_test_idx)]
        failure_test_idx = random.sample(failure_idx, nTest)

        # get training data
        trainData, trainTimeList = loadData([success_list[x] for x in train_idx],
                                            isTrainingData=True, downSampleSize=self.downSampleSize,
                                            verbose=self.verbose)

        # get threshold-test data
        thresTestData, thresTestTimeList = loadData([success_list[x] for x in ths_test_idx],
                                                    isTrainingData=True, downSampleSize=self.downSampleSize,
                                                    verbose=self.verbose)

        # get test data
        normalTestData, normalTestTimeList = loadData([success_list[x] for x in success_test_idx],
                                                      isTrainingData=False, downSampleSize=self.downSampleSize,
                                                      verbose=self.verbose)
        abnormalTestData, abnormalTestTimeList = loadData([failure_list[x] for x in failure_test_idx],
                                                          isTrainingData=False, downSampleSize=self.downSampleSize,
                                                          verbose=self.verbose)

        return trainData, thresTestData, normalTestData, abnormalTestData, trainTimeList, thresTestTimeList, normalTestTimeList, abnormalTestTimeList

    def scaleData(self, dataList, minVals=None, maxVals=None):
        # Determine max and min values
        if minVals is None:
            minVals = []
            maxVals = []
            for modality in dataList:
                minVals.append(np.min(modality))
                maxVals.append(np.max(modality))
            if self.verbose:
                print 'minValues', minVals
                print 'maxValues', maxVals

        nDimension = len(dataList)
        dataList_scaled = []
        for i in xrange(nDimension):
            dataList_scaled.append([])

        # Scale features
        for i in xrange(nDimension):
            for j in xrange(len(dataList[i])):
                dataList_scaled[i].append(util.scaling(dataList[i][j], minVals[i], maxVals[i], self.scale))

        return dataList_scaled, minVals, maxVals


# Specify parameters and possible parameter values
tuned_params = {'downSampleSize': [100, 200, 300], 'scale': [1, 5, 10], 'nState': [20, 30], 'cov_mult': [1.0, 3.0, 5.0, 10.0]}

# Run grid search
gs = GridSearchCV(HmmClassifier(), tuned_params)
gs.fit(X=[1,2,3,4], y=[1,1,1,1])

print 'Grid Search:'
print gs.best_params_, gs.best_score_, gs.grid_scores_


# specify parameters and distributions to sample from
param_dist = {'downSampleSize': sp_randint(100, 300),
              'scale': sp_randint(1, 10),
              'nState': sp_randint(20, 30),
              'cov_mult': sp_randint(1, 10)}

# Run randomized search
random_search = RandomizedSearchCV(HmmClassifier(), param_distributions=param_dist, n_iter=5)
random_search.fit(X=[1,2,3,4], y=[1,1,1,1])

print 'Randomized Search:'
print gs.best_params_, gs.best_score_, gs.grid_scores_
