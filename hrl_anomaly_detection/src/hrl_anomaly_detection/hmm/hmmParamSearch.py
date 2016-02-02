#!/usr/bin/env python

import random
from scipy.stats import randint as sp_randint
from sklearn.grid_search import RandomizedSearchCV, GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin
from learning_hmm_multi_n import learning_hmm_multi_n
from util import *
import learning_util as util

# catkin_make_isolated --only-pkg-with-deps hrl_anomaly_detection --merge
# nohup rosrun hrl_anomaly_detection hmmParamSearch.py > optimization.log &
# sed '/GHMM ghmm.py/d' optimization.log > optimizationClean.log

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
        self.trainData = None

    def set_params(self, **parameters):
        params = ''
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
            params += '%s: %s, ' % (parameter, str(value))
        print params
        return self

    def fit(self, X=None, y=None):
        """
        This should fit classifier. All the "work" should be done here.

        Note: assert is not a good choice here and you should rather
        use try/except blog with exceptions. This is just for short syntax.
        """

        # subject_names = ['s2', 's3', 's4', 's7', 's8', 's9', 's10', 's11', 's12', 's13']
        # subject_names = ['s2', 's3', 's4', 's7', 's8', 's9', 's10', 's11']
        subject_names = ['s2']
        task_name = 'feeding'

        # Loading success and failure data
        root_path = '/home/mycroft/feeding'
        success_list, _ = getSubjectFileList(root_path, subject_names, task_name)

        if self.verbose:
            print "--------------------------------------------"
            print "# of Success files: ", len(success_list)
            print "--------------------------------------------"

        trainData, _ = loadData(success_list, isTrainingData=True, downSampleSize=self.downSampleSize, verbose=self.verbose)

        # Possible pass in trainData through X (thus utilizing the cross-validation in sklearn)
        # print 'Number of modalities (dimensions):', len(trainData)
        # print 'Lengths of data:', [len(trainData[i]) for i in xrange(len(trainData))]
        # print 'Lengths of internal data:', [len(trainData[i][0]) for i in xrange(len(trainData))]
        # TODO: Notice above!!

        # minimum and maximum vales for scaling from Daehyung
        minVals = []
        maxVals = []
        for modality in trainData:
            minVals.append(np.min(modality))
            maxVals.append(np.max(modality))

        # Scale data
        self.trainData = self.scaleData(trainData, minVals=minVals, maxVals=maxVals)

        # # cutting data (only training and thresTest data)
        # start_idx = int(float(len(trainData[0][0]))*self.train_cutting_ratio[0])
        # end_idx = int(float(len(trainData[0][0]))*self.train_cutting_ratio[1])
        #
        # for j in xrange(len(trainData)):
        #     for k in xrange(len(trainData[j])):
        #         trainData[j][k] = trainData[j][k][start_idx:end_idx]

        # hmm = learning_hmm_multi_4d(nState=nState, nEmissionDim=4, anomaly_offset=anomaly_offset, verbose=verbose)
        self.hmm = learning_hmm_multi_n(nState=self.nState, nEmissionDim=4, check_method='progress', anomaly_offset=self.anomaly_offset, verbose=self.verbose)
        ret = self.hmm.fit(xData=self.trainData, cov_mult=[self.cov_mult]*16)

        if ret == 'Failure':
            print 'HMM return was a failure!'
            return self

        self.isFitted = True

        return self

    def score(self, X, y, sample_weight=None):
        if not self.isFitted:
            return 0

        log_ll = []
        # exp_log_ll = []        
        for i in xrange(len(self.trainData[0])):
            log_ll.append([])
            # exp_log_ll.append([])
            for j in range(2, len(self.trainData[0][i])):
                X = [x[i,:j] for x in np.array(self.trainData)]                
                exp_logp, logp = self.hmm.expLoglikelihood(X, self.hmm.l_ths_mult, bLoglikelihood=True)
                log_ll[i].append(logp)
                # exp_log_ll[i].append(exp_logp)

        logs = [x[-1] for x in log_ll]
        print 'expLoglikelihood() log_ll:', np.shape(log_ll), sum(logs) / float(len(logs))
        return sum(logs) / float(len(logs))

        # print 'expLoglikelihood() exp_log_ll:', np.shape(exp_log_ll), [x[-1] for x in exp_log_ll]

        # print 'loglikelihood()', self.hmm.loglikelihood(self.trainData)
        # print '-'*50
        # likelihood, posterior = self.hmm.loglikelihoods(self.trainData, bPosterior=True)
        # print 'loglikelihoods(), likelihood:', likelihood
        # print 'loglikelihoods(), posterior:', posterior
        # print '-'*50
        # exp_logp, logp = self.hmm.expLoglikelihood(self.trainData, self.hmm.l_ths_mult, bLoglikelihood=True)
        # print 'expLoglikelihood() exp_logp:', exp_logp
        # print 'expLoglikelihood() logp:', logp

        # return 0

    def scaleData(self, dataList, minVals=None, maxVals=None):
        nDimension = len(dataList)
        dataList_scaled = []
        for i in xrange(nDimension):
            dataList_scaled.append([])

        # Scale features
        for i in xrange(nDimension):
            for j in xrange(len(dataList[i])):
                dataList_scaled[i].append(util.scaling(dataList[i][j], minVals[i], maxVals[i], self.scale))

        return dataList_scaled


print '\n', '-'*50, '\nBeginning Grid Search\n', '-'*50, '\n'

# Specify parameters and possible parameter values
tuned_params = {'downSampleSize': [100, 200, 300], 'scale': [1, 5, 10], 'nState': [20, 30], 'cov_mult': [1.0, 3.0, 5.0, 10.0]}

# Run grid search
gs = GridSearchCV(HmmClassifier(), tuned_params, cv=2)
gs.fit(X=[1,2,3,4], y=[1,1,1,1])

print 'Grid Search:'
print 'Best params:', gs.best_params_
print 'Best Score:', gs.best_score_
# print 'Grid scores:', gs.grid_scores_

print '\n', '-'*50, '\nBeginning Randomized Search\n', '-'*50, '\n'

# specify parameters and distributions to sample from
param_dist = {'downSampleSize': sp_randint(100, 300),
              'scale': sp_randint(1, 10),
              'nState': sp_randint(20, 30),
              'cov_mult': sp_randint(1, 10)}

# Run randomized search
random_search = RandomizedSearchCV(HmmClassifier(), param_distributions=param_dist, cv=2, n_iter=50)
random_search.fit(X=[1,2,3,4], y=[1,1,1,1])

print 'Randomized Search:'
print 'Best params:', random_search.best_params_
print 'Best Score:', random_search.best_score_
# print 'Grid scores:', random_search.grid_scores_
