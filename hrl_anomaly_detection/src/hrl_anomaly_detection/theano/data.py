import os
import random
import numpy as np
import matplotlib.pyplot as plt
import theano
# By convention, the tensor submodule is loaded as T
import theano.tensor as T


def getData():

    # Training data - two randomly-generated Gaussian-distributed clouds of points in 2d space
    np.random.seed(0)
    # Number of points
    N = 1000
    # Labels for each cluster
    y = np.random.random_integers(0, 1, N)
    # Mean of each cluster
    means = np.array([[-1, 1], [-1, 1]])
    # Covariance (in X and Y direction) of each cluster
    covariances = np.random.random_sample((2, 2)) + 1
    # Dimensions of each point
    X = np.vstack([np.random.randn(N)*covariances[0, y] + means[0, y],
                   np.random.randn(N)*covariances[1, y] + means[1, y]])
    # Plot the data
    #plt.figure(figsize=(8, 8))
    #plt.scatter(X[0, :], X[1, :], c=y, lw=.3, s=3, cmap=plt.cm.cool)
    #plt.axis([-6, 6, -6, 6])
    #plt.show()
    
    return X, y

def getData2(dataset='./SDA/mnist.pkl.gz'):

    from SDA.logistic_sgd import LogisticRegression, load_data

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    return train_set_x, train_set_y

def getData3(time_window, renew=False):

    from hrl_anomaly_detection import data_manager as dm

    subject_names       = ['gatsbii']
    task                = 'pushing'
    raw_data_path       = os.path.expanduser('~')+'/hrl_file_server/dpark_data/anomaly/RSS2016/'    
    processed_data_path = os.path.expanduser('~')+'/hrl_file_server/dpark_data/anomaly/RSS2016/'+task+'_data'
    h5py_file           = os.path.join(processed_data_path, 'test.h5py')
    rf_center           = 'kinEEPos'
    local_range         = 1.25    
    nSet                = 1
    downSampleSize      = 200

    feature_list = ['unimodal_audioPower',\
                    'unimodal_kinVel',\
                    'unimodal_ftForce',\
                    ##'unimodal_visionChange',\
                    ##'unimodal_ppsForce',\
                    ##'unimodal_fabricForce',\
                    'crossmodal_targetEEDist', \
                    'crossmodal_targetEEAng']
    feature_list = ['relativePose_artag_EE', \
                    'relativePose_artag_artag', \
                    'kinectAudio',\
                    'wristAudio', \
                    'ft', \
                    ## 'pps', \
                    ## 'visionChange', \
                    ## 'fabricSkin', \
                    ]



    _, successData, failureData,_ , param_dict = dm.getDataSet(subject_names, task, raw_data_path, \
                                                               processed_data_path, rf_center, local_range,\
                                                               nSet=nSet, \
                                                               downSampleSize=downSampleSize, \
                                                               scale=1.0,\
                                                               raw_data=True, data_ext=False, \
                                                               feature_list=feature_list, \
                                                               data_renew=renew)

    # index selection
    success_idx  = range(len(successData[0]))
    failure_idx  = range(len(failureData[0]))

    nTrain       = int( 0.8*len(success_idx) )    
    train_idx    = random.sample(success_idx, nTrain)
    success_test_idx = [x for x in success_idx if not x in train_idx]
    failure_test_idx = failure_idx
    ndim_list        = param_dict

    # data structure: dim x sample x sequence
    trainingData     = successData[:, train_idx, :]
    normalTestData   = successData[:, success_test_idx, :]
    abnormalTestData = failureData[:, failure_test_idx, :]

    print "======================================"
    print "Dim x nSamples x nLength"
    print "--------------------------------------"
    print "Training data: ", np.shape(trainingData)
    print "Normal test data: ", np.shape(normalTestData)
    print "Abnormal test data: ", np.shape(abnormalTestData)
    print "======================================"


    # scaling by the number of dimensions in each feature
    dataDim = param_dict['dataDim']
    index   = 0
    for feature_name, nDim in dataDim:
        pre_index = index
        index    += nDim

        trainingData[pre_index:index] /= np.sqrt(nDim)
        normalTestData[pre_index:index] /= np.sqrt(nDim)
        abnormalTestData[pre_index:index] /= np.sqrt(nDim)
        

    new_trainingData = []
    for i in xrange(len(trainingData[0])):
        singleSample = []
        for j in xrange(len(trainingData)):
            singleSample.append(trainingData[j][i,:])            
        new_trainingData.append(singleSample)

    new_testData = []
    for i in xrange(len(normalTestData[0])):
        singleSample = []
        for j in xrange(len(normalTestData)):
            singleSample.append(normalTestData[j][i,:])            
        new_testData.append(singleSample)

    for i in xrange(len(abnormalTestData[0])):
        singleSample = []
        for j in xrange(len(abnormalTestData)):
            singleSample.append(abnormalTestData[j][i,:])            
        new_testData.append(singleSample)


    print "======================================"
    print "nSamples x Dim x nLength"
    print "--------------------------------------"
    print "Training Data: ", np.shape(new_trainingData)
    print "Test Data: ", np.shape(new_testData)
    print "======================================"


    # reshaping
    new_trainingData = np.array(new_trainingData)
    new_trainingData2 = []
    for i in xrange(len(new_trainingData)):
        for j in xrange(len(new_trainingData)-time_window+1):
            new_trainingData2.append( new_trainingData[i][:,j:j+time_window].flatten() )        
    print np.shape(np.array(new_trainingData2).T)

    new_testData = np.array(new_testData)
    new_testData2 = []
    for i in xrange(len(new_testData)):
        for j in xrange(len(new_testData)-time_window+1):
            new_testData2.append( new_testData[i][:,j:j+time_window].flatten() )        
    print np.shape(np.array(new_testData2).T)

    nSingleData = len(new_testData)-time_window+1
    
    return np.array(new_trainingData2).T, None, np.array(new_testData2).T, None, nSingleData

