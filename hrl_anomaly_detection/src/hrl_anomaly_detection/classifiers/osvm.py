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
import os, sys, copy, time

# util
import numpy as np
import scipy

from hrl_anomaly_detection.hmm.clf_base import clf_base

class osvm(clf_base):
    def __init__(self, svm_type=0, kernel_type=2, degree=3, gamma=0.3, cost=4., coef0=0.,
                 w_negative=7., osvm_nu=0.00316, verbose=False, **kwargs):
        ''' '''        
        clf_base.__init__(self)

        sys.path.insert(0, '/usr/lib/pymodules/python2.7')
        import svmutil as svm
        self.svm_type    = svm_type
        self.kernel_type = kernel_type
        self.degree      = degree
        self.gamma       = gamma
        self.cost        = cost
        self.coef0       = coef0
        self.w_negative  = w_negative
        self.osvm_nu     = osvm_nu
        ## self.nu          = nu
        self.dt          = None
        self.verbose = verbose

    def fit(self, X, y=None):

        sys.path.insert(0, '/usr/lib/pymodules/python2.7')
        import svmutil as svm

        if type(X) is not list: X=X.tolist()
        if type(y) is not list: y=y.tolist()
        commands = '-q -s '+str(self.svm_type)+' -t '+str(self.kernel_type)+' -d '+str(self.degree)\
          +' -w1 '+str(self.class_weight) +' -r '+str(self.coef0)

        commands = commands+' -n '+str(self.osvm_nu)+' -g '+str(self.gamma)\
          +' -w-1 '+str(self.w_negative)+' -c '+str(self.cost)

        try: self.dt = svm.svm_train(y, X, commands )
        except:
            print "svm training failure"
            print np.shape(y), np.shape(X)
            print commands                
            return False

        if self.method == 'svm_fixed':
            if type(X) == list: X = np.array(X)
            ll_logp = X[:,0:1]
            self.mu  = np.mean(ll_logp)
            self.std = np.std(ll_logp)

        return True
    

    def partial_fit(self, X, y=None, X_idx=None, shuffle=True, **kwargs):
        return

    def predict(self, X, y=None):
        ''' '''

        sys.path.insert(0, '/usr/lib/pymodules/python2.7')
        import svmutil as svm
        
        if self.verbose:
            print svm.__file__

        if type(X) is not list: X=X.tolist()
        if y is not None:
            p_labels, _, p_vals = svm.svm_predict(y, X, self.dt)
        else:
            p_labels, _, p_vals = svm.svm_predict([0]*len(X), X, self.dt)
        
        return p_labels

    def score(self, X, y):
        return self.dt.score(X,y)


    def save_model(self, fileName):
        sys.path.insert(0, '/usr/lib/pymodules/python2.7')
        import svmutil as svm            
        svm.svm_save_model(fileName, self.dt)

    def load_model(self, fileName):        
        sys.path.insert(0, '/usr/lib/pymodules/python2.7')
        import svmutil as svm            
        self.dt = svm.svm_load_model(fileName) 
    
        


