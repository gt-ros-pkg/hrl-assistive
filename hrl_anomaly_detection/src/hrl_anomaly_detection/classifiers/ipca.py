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

from hrl_anomaly_detection.classifiers.clf_base import clf_base
from sklearn.decomposition import IncrementalPCA

class ipca(clf_base):
    def __init__(self, n_components=2, batch_size=50, ths=0, verbose=False, **kwargs):
        ''' '''        
        clf_base.__init__(self)
        self.ths = ths
        self.ml  = IncrementalPCA(n_components=n_components, batch_size=batch_size)
        

    def fit(self, X, **kwargs):
        '''
        X: sample x feature
        '''
        self.ml.fit(X )

        # temp
        # Need to estimate reconstruction errors on training data to set up ths range
        ## errs = self.predict(X)+self.ths
        return

    def partial_fit(self, X, **kwargs):
        
        self.ml.partial_fit(X)        
        return

    def predict(self, X, y=None):
        ''' '''
        l_err = []
        for i in xrange(len(X)):
            x_org = X[i]
            x_prj = self.ml.inverse_transform(self.ml.transform(x_org))

            l_err.append( ((x_org-x_prj)**2).mean() )

        return np.array(l_err) - self.ths


    def save_model(self, fileName):
        import pickle
        with open(fileName, 'wb') as f:
            pickle.dump(self.ml, f)
        return

    def load_model(self, fileName):        
        import pickle
        with open(fileName, 'rb') as f:
            self.ml = pickle.load(f)
        return

    
        


