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

import sys
import random
import numpy as np
from util import getSubjectFiles
from scipy.stats import randint as sp_randint
from sklearn.grid_search import RandomizedSearchCV, GridSearchCV
from learning_hmm_multi_n import learning_hmm_multi_n

def optimizeHMM(root_path, params, kfolds=3, gridSearch=True, iterations=50, verbose=False):
    # List for all returned values of HMM optimization
    results = []

    # Loading success and failure data
    success_list, _ = getSubjectFiles(root_path)
    featureIndices = list(xrange(len(success_list)))
    random.shuffle(featureIndices)
    if verbose:
        print "--------------------------------------------"
        print "# of Success files: ", len(success_list)
        print "--------------------------------------------"

    hmm = learning_hmm_multi_n(check_method='progress', cluster_type='time', optimDataPath=root_path, folds=kfolds, resultsList=results)

    if gridSearch:
        if verbose:
            print '\n', '-'*50, '\nBeginning Grid Search\n', '-'*50, '\n'
            sys.stdout.flush()

        # Run grid search
        gs = GridSearchCV(hmm, params, cv=kfolds)
        gs.fit(X=featureIndices, y=[1]*len(featureIndices))

        if verbose:
            print 'Grid Search:'
            print 'Best params:', gs.best_params_
            print 'Best Score:', gs.best_score_
            sys.stdout.flush()
    else:
        if verbose:
            print '\n', '-'*50, '\nBeginning Randomized Search\n', '-'*50, '\n'
            sys.stdout.flush()

        # Run randomized search
        random_search = RandomizedSearchCV(hmm, param_distributions=params, n_iter=iterations, cv=kfolds)
        random_search.fit(X=featureIndices, y=[1]*len(featureIndices))

        if verbose:
            print 'Randomized Search:'
            print 'Best params:', random_search.best_params_
            print 'Best Score:', random_search.best_score_
            sys.stdout.flush()

    return results


def autoOptimize(root_path, kfolds=3, verbose=False):
    # Specify parameters and parameter values to search through
    params = {'downSampleSize': [100, 300, 500], 'scale': [1, 2, 4, 6], 'nState': [5, 10, 15, 20], 'cov_mult': [10.0, 40.0, 100.0]}

    res = optimizeHMM(root_path, params, kfolds=kfolds, gridSearch=True, verbose=verbose)

    notNaN = [x for x in res if x[1] != 'NaN' and x[2] != 0]
    if not notNaN:
        # No viable parameters found! Open up search space with random search
        print 'Unable to find viable parameters using grid search. Increasing search space with random search.'
        params = {'downSampleSize': sp_randint(100, 500), 'scale': sp_randint(1, 10), 'nState': sp_randint(5, 30), 'cov_mult': sp_randint(1, 200)}
        res = optimizeHMM(root_path, params, kfolds=kfolds, gridSearch=True, verbose=verbose)
        notNaN = [x for x in res if x[1] != 'NaN' and x[2] != 0]
        if not notNaN:
            print 'Unable to find any viable HMM parameters for this data. Please restructure and try again.'
            return None

    notNaN = sorted(notNaN, key=lambda x: x[-1], reverse=True)
    bestParams, _, bestScore = notNaN[0]

    if verbose:
        print 'Displaying results for initial autonomous parameter search'
        displayResults(res)

    # Perform refined search near best parameter set
    sampleSize = [bestParams['downSampleSize'] - 50, bestParams['downSampleSize'], bestParams['downSampleSize'] + 50]
    scale = [bestParams['scale'] - 0.5, bestParams['scale'], bestParams['scale'] + 0.5]
    nState = [bestParams['nState'] - 2, bestParams['nState'], bestParams['nState'] + 2]
    cov_mult = [bestParams['cov_mult']]
    params = {'downSampleSize': sampleSize, 'scale': scale, 'nState': nState, 'cov_mult': cov_mult}

    res2 = optimizeHMM(root_path, params, kfolds=kfolds, gridSearch=True, verbose=verbose)

    notNaN2 = [x for x in res2 if x[1] != 'NaN' and x[2] != 0]
    if notNaN2:
        notNaN2 = sorted(notNaN2, key=lambda x: x[-1], reverse=True)
        if notNaN2[0][2] > bestScore:
            bestParams, _, bestScore = notNaN2[0]
            if verbose:
                print 'Refined search found higher scoring parameters.'

    if verbose:
        print 'Displaying results for refined autonomous parameter search'
        displayResults(res2)

    return bestParams, bestScore

def displayResults(results):
    # Display Overview of results, with successful parameter sets then 'NaN' parameter sets
    print '\n', '-'*15, 'Results', '-'*15
    notNaN = [x for x in results if x[1] != 'NaN' and x[2] != 0]
    notNaN = sorted(notNaN, key=lambda x: x[-1], reverse=True)
    for x in notNaN:
        # Convert parameter list to string
        s = ', '.join(['%s: %s' % (p, str(v)) for p, v in x[0]])
        print x[-1], '-', s, '-', x[1]

    print '\n', '-'*15, 'NaNs', '-'*15
    for x in results:
        if x[1] == 'NaN':
            # Convert parameter list to string
            print ', '.join(['%s: %s' % (p, str(v)) for p, v in x[0]])


kfolds = 3
root_path = '/home/mycroft/gatsbii_scooping'

# Specify parameters and possible parameter values
tuned_params = {'downSampleSize': [300], 'scale': [2.5], 'nState': [10, 15], 'cov_mult': [10.0, 50.0, 100.0, 200.0]}

# # specify parameters and distributions to sample from
# param_dist = {'downSampleSize': [300, 350, 400, 450, 500], #sp_randint(300, 400),
#              'scale': [2.5],
#              'nState': sp_randint(5, 20),
#              'cov_mult': sp_randint(10, 200)}
#

res = optimizeHMM(root_path, tuned_params, kfolds=kfolds, gridSearch=True, verbose=True)

displayResults(res)

print '\n', '-'*50, '\nBeginning Auto Optimization\n', '-'*50, '\n'

finalParams, finalScore = autoOptimize(root_path, kfolds=kfolds, verbose=True)

print 'Final params:', finalParams
print 'Final Score:', finalScore
