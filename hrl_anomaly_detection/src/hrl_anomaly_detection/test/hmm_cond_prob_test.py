import numpy as np
import sys, os
from hrl_anomaly_detection.hmm import learning_hmm as hmm
from hrl_anomaly_detection import util as util
## import learning_util as util



t = np.linspace(0.0, np.pi*2.0, 100)
n = 10 

X = []
for i in xrange(n):
    if i<1:
        x1 = np.cos(t) + np.random.normal(-0.3, 0.3, np.shape(t) )
        x2 = np.sin(t) + np.random.normal(-0.3, 0.3, np.shape(t) )
        x3 = np.sin(t+np.pi/4.0) + np.random.normal(-0.3, 0.3, np.shape(t) )
    else:
        x1 = np.cos(t) + np.random.normal(-0.2, 0.2, np.shape(t) )
        x2 = np.sin(t) + np.random.normal(-0.2, 0.2, np.shape(t) )        
        x3 = np.sin(t+np.pi/4.0) + np.random.normal(-0.2, 0.2, np.shape(t) )        
    X.append( np.vstack([ x1.reshape(1,len(t)), x2.reshape(1,len(t)), x3.reshape(1,len(t)) ]) )
X = np.swapaxes(X, 0,1)

print np.shape(X)

nEmissionDim = len(X)
nState       = 20
cov_mult = [0.01]*(nEmissionDim**2)
cov_type = 'full'

#-------------------------------------------------------------
ml  = hmm.learning_hmm(nState, nEmissionDim)
ret = ml.fit(X+\
             np.random.normal(0.0, 0.2, np.shape(X) ), \
             cov_mult=cov_mult, use_pkl=False)

#-------------------------------------------------------------

X_test = X[:,1,:50]
cp = ml.conditional_prob(X_test)

print cp #/np.sum(cp)


