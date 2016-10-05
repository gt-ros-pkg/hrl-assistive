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
import time
import random

## from sklearn.grid_search import ParameterGrid
## from sklearn.cross_validation import KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn import preprocessing

import hrl_lib.util as ut
from hrl_anomaly_detection.hmm.learning_base import learning_base
from hrl_anomaly_detection.hmm import learning_hmm as hmm
from hrl_anomaly_detection import data_manager as dm
from hrl_anomaly_detection.params import *
import hrl_anomaly_detection.classifiers.classifier as cf


import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt


class anomaly_detector(learning_base):
    def __init__(self, method, nState, nLength=None,\
                 weight=1., w_negative=1., gamma=1., cost=1., nu=0.5):
        self.method = method
        self.nState = nState
        self.scaler = None

        self.weight     = weight
        self.w_negative = w_negative
        self.gamma      = gamma
        self.cost       = cost
        self.nu         = nu
        
        self.dtc = cf.classifier( method=method, nPosteriors=nState, nLength=nLength )        

        return

    def fit(self, X, y):

        # set param
        d = {'w_negative': self.w_negative, 'gamma': self.gamma,\
             'cost': self.cost, 'class_weight': self.weight, 'nu': self.nu}
        self.dtc.set_params(**d)
        
        # flatten the data
        if self.method.find('svm')>=0 or self.method.find('sgd')>=0: remove_fp=True
        else: remove_fp = False
        X_train, y_train, _ = dm.flattenSample(X, y, remove_fp=remove_fp)
        ## print self.w_negative, self.weight

        if (self.method.find('svm')>=0 or self.method.find('sgd')>=0) and \
          not(self.method == 'osvm' or self.method == 'bpsvm'):
            self.scaler = preprocessing.StandardScaler()
            X_train = self.scaler.fit_transform(X_train)

        # fit classifier
        ret = self.dtc.fit(X_train, y_train)
        if ret is False:
            print "Fitting failure"
            sys.exit()

        return

    def predict(self, X):

        labels = []
        for i in xrange(len(X)):
            
            X_scaled = self.scaler.transform(X[i])
            est_y    = self.dtc.predict(X_scaled)

            label = -1.0
            for j in xrange(len(est_y)):                
                if est_y[j]>0:
                    label = 1.0
                    break
            
            labels.append(label)
        return labels

    ## def decision_function(self, X):
    ##     return

    def score(self, X, y):

        tp = 0.0
        fp = 0.0
        tn = 0.0
        fn = 0.0

        est_y = self.predict(X)
        
        for i in xrange(len(est_y)):
            if y[i][0]>0:
                if est_y[i]>0: tp += 1.0
                else: fn += 1.0
            else:
                if est_y[i]>0: fp += 1.0
                else: tn += 1.0

        # f1-score
        fscore = 2.0*tp/(2.0*tp+fn+fp)
        # f0.5-score
        ## fscore = 1.25*tp/(1.25*tp+0.25*fn+fp)
        # f2-score
        ## fscore = 5.0*tp/(5.0*tp+4.0*fn+fp)
        
        return fscore

def getSamples(modeling_pkl):
    print "start to load hmm data, ", modeling_pkl
    d = ut.load_pickle(modeling_pkl)
    for k, v in d.iteritems():
        exec '%s = v' % k
    X_train = ll_classifier_train_X
    y_train = ll_classifier_train_Y
    X_test  = ll_classifier_test_X
    y_test  = ll_classifier_test_Y

    id_list = range(len(X_train))
    random.shuffle(id_list)
    X_train = np.array(X_train)[id_list] 
    y_train = np.array(y_train)[id_list] 

    ## data = dm.getHMMData(method, nFiles, save_data_path, opt.task, param_dict)
    ## X_train = data[file_idx]['X_scaled']
    ## y_train = data[file_idx]['Y_train_org']
    ## X_test  = data[file_idx]['X_test']
    ## y_test  = data[file_idx]['Y_test']

    return X_train, y_train, X_test, y_test
    


if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()

    p.add_option('--task', action='store', dest='task', type='string', default='pushing_microwhite',
                 help='type the desired task name')
    p.add_option('--dim', action='store', dest='dim', type=int, default=4,
                 help='type the desired dimension')
    p.add_option('--method', '--m', action='store', dest='method', type='string', default='svm',
                 help='type the method name')
    p.add_option('--n_jobs', action='store', dest='n_jobs', type=int, default=-1,
                 help='number of processes for multi processing')
    p.add_option('--save', action='store_true', dest='bSave',
                 default=False, help='Save result.')

    p.add_option('--icra2017', action='store_true', dest='bICRA2017',
                 default=False, help='Enable ICRA2017.')

    p.add_option('--rawplot', '--rp', action='store_true', dest='bRawDataPlot',
                 default=False, help='Plot raw data.')
    p.add_option('--renew', action='store_true', dest='bRenew', default=False,
                 help='Renew result')
    opt, args = p.parse_args()

    rf_center     = 'kinEEPos'        
    local_range    = 10.0    
    nPoints        = 10

    if opt.bICRA2017 is False:
        from hrl_anomaly_detection.params import *
        raw_data_path, save_data_path, param_dict = getParams(opt.task, False, \
                                                              False, False, opt.dim,\
                                                              rf_center, local_range, \
                                                              nPoints=nPoints)
    else:
        from hrl_anomaly_detection.ICRA2017_params import *
        raw_data_path, save_data_path, param_dict = getParams(opt.task, False, \
                                                              False, False, opt.dim,\
                                                              rf_center, local_range, \
                                                              nPoints=nPoints)
    nFiles     =  1 #param_dict['data_param']['nNormalFold']*param_dict['data_param']['nAbnormalFold']
    method     = opt.method
    result_pkl = os.path.join(save_data_path, 'result_'+opt.task+'_'+str(opt.dim)+'_'+method+'.pkl')

    
    # get training X,y
    file_idx = 1
    modeling_pkl = os.path.join(save_data_path, 'hmm_'+opt.task+'_'+str(file_idx)+'.pkl')
    X_train, y_train, X_test, y_test = getSamples(modeling_pkl)

    X = X_train
    y = y_train
    
    # specify parameters and distributions to sample from
    from scipy.stats import uniform, expon
    param_dist = {'cost': [1.0],\
                  'gamma': uniform(0.1, 3.0),\
                  'weight': expon(scale=0.3),\
                  'nu': [0.5]
                  }
        #, #uniform(0.1,0.9)
        # uniform(7.0,15.0)
        # 'cost': uniform(0.5,4.0)
        ## 'weight': uniform(np.exp(-2.15), np.exp(-0.1)),
    clf = anomaly_detector(method, param_dict['HMM']['nState'])
        
    # run randomized search
    n_iter_search = 3000 #1000 #20
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                       cv=2, n_jobs=8,
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

    ## print("Detailed classification report:")
    ## print()
    ## print("The model is trained on the full development set.")
    ## print("The scores are computed on the full evaluation set.")
    ## print()
    ## y_true, y_pred = y_test, random_search.predict(X_test)
    ## print(classification_report(y_true, y_pred))
    ## print()
    
