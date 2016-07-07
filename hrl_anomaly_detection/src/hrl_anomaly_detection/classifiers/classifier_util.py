
from sklearn import preprocessing
import random, copy
import numpy as np

from hrl_anomaly_detection import data_manager as dm


def getProcessSGDdata(X, y, weight=1.0, remove_overlap=True):
    '''
    y: a list of sample label (1D)
    '''

    for k in xrange(len(X)):
        X_ptrain, Y_ptrain = X[k], y[k]
        if Y_ptrain[0] > 0 and remove_overlap:           
            X_ptrain, Y_ptrain = dm.getEstTruePositive(X_ptrain)

        ## sample_weight = np.array([1.0]*len(Y_ptrain))
        if Y_ptrain == -1:
            sample_weight = [1.0]*len(X_ptrain)
        else:
            sample_weight = [weight]*len(X_ptrain)

        if k==0:
            p_train_X = X_ptrain
            p_train_Y = Y_ptrain
            p_train_W = sample_weight
        else:
            p_train_X = np.vstack([p_train_X, X_ptrain])
            p_train_Y = np.hstack([p_train_Y, Y_ptrain])
            p_train_W = np.hstack([p_train_W, sample_weight])

    p_idx_list = range(len(p_train_X))
    random.shuffle(p_idx_list)
    p_train_X = [p_train_X[ii] for ii in p_idx_list]
    p_train_Y = [p_train_Y[ii] for ii in p_idx_list]
    p_train_W = [p_train_W[ii] for ii in p_idx_list]

    return p_train_X, p_train_Y, p_train_W
    
