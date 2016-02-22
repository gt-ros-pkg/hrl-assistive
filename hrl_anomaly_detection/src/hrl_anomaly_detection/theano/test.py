import sys, os
import numpy as np
import matplotlib.pyplot as plt
import theano
# By convention, the tensor submodule is loaded as T
import theano.tensor as T
import layer as l

# util
import numpy as np
import hrl_lib.util as ut
from hrl_anomaly_detection.util import *
from hrl_anomaly_detection.util_viz import *
from hrl_anomaly_detection import data_manager as dm
import random

## import cPickle
from six.moves import cPickle

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

def getData3(time_window):

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
                                                               raw_data=True, data_ext=False, \
                                                               feature_list=feature_list, \
                                                               data_renew=False)

    # index selection
    success_idx  = range(len(successData[0]))
    failure_idx  = range(len(failureData[0]))

    nTrain       = int( 0.7*len(success_idx) )    
    train_idx    = random.sample(success_idx, nTrain)
    success_test_idx = [x for x in success_idx if not x in train_idx]
    failure_test_idx = failure_idx
    ndim_list        = param_dict

    # data structure: dim x sample x sequence
    trainingData     = successData[:, train_idx, :]
    normalTestData   = successData[:, success_test_idx, :]
    abnormalTestData = failureData[:, failure_test_idx, :]

    print "======================================"
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

    print np.shape(new_trainingData)


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
    
    return np.array(new_trainingData2).T, None, np.array(new_testData2).T, None


def save_params(obj, filename):
    f = file(filename, 'wb')
    cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
                            
if __name__ == '__main__':


    ## X, y = getData2()
    ## X = X.get_value(True).T

    ## X, y = getData()
    time_window = 2
    X_train, _, X_test, _ = getData3(time_window)

    layer_sizes = [X_train.shape[0], 20, 10, 5]
    print layer_sizes

    # Set initial parameter values
    W_init_en = []
    b_init_en = []
    activations_en = []
    W_init_de = []
    b_init_de = []
    activations_de = []

    # Encoder layers
    for n_input, n_output in zip(layer_sizes[:-1], layer_sizes[1:]):
        W_init_en.append(np.random.randn(n_output, n_input))
        b_init_en.append(np.ones(n_output))
        # We'll use sigmoid activation for all layers
        # Note that this doesn't make a ton of sense when using squared distance
        # because the sigmoid function is bounded on [0, 1].
        activations_en.append(T.nnet.sigmoid)

    # Decoder layers
    layer_sizes = list(reversed(layer_sizes))
    for n_input, n_output in zip(layer_sizes[:-1], layer_sizes[1:]):
        W_init_de.append(np.random.randn(n_output, n_input))
        b_init_de.append(np.ones(n_output))
        activations_de.append(T.nnet.sigmoid)
    activations_de[-1] = None

    # Create an instance of the MLP class
    mlp = l.AD(W_init_en, b_init_en, activations_en,
               W_init_de, b_init_de, activations_de, nEncoderLayers=len(layer_sizes))

    # Create Theano variables for the MLP input
    mlp_input = T.matrix('mlp_input')
    mlp_target = T.matrix('mlp_target')
    learning_rate = 0.001
    momentum = 0.01
    lambda_reg = 0.0
    batch_size = 2
    cost = mlp.squared_error(mlp_input, mlp_target)
    print 'Creating a theano function for training the network'
    train = theano.function([mlp_input, mlp_target], cost,
                            updates=l.gradient_updates_momentum(cost, mlp.params, learning_rate, momentum, \
                                                                lambda_reg))
    print 'Creating a theano function for computing the MLP\'s output given some input'
    mlp_output = theano.function([mlp_input], mlp.output(mlp_input))
    print 'Creating a theano function for computing the MLP\'s output given some input'
    mlp_features = theano.function([mlp_input], mlp.get_features(mlp_input))


    print 'Optimising'
    # Keep track of the number of training iterations performed
    iteration = 0
    max_iteration = 100000
    while iteration < max_iteration:

        current_cost = 0.0
        count = 0.0
        for i in range(0, X_train.shape[1]-batch_size+1, batch_size):
            count += 1.0
            current_cost += train(X_train[:,i:i+batch_size], X_train[:,i:i+batch_size])
            current_output = mlp_output(X_train[:,i:i+batch_size])

        current_output = mlp_output(X_test)

        loss = T.sum( (X_test-current_output)**2 )/len(X_test[0])
            
            
        print iteration, ' Train cost ' + str(current_cost/count)+ ' test loss: '+str(loss.eval())

        if iteration%10:
            f = open('simple_model.pkl', 'wb')
            cPickle.dump(mlp_features, f, protocol=cPickle.HIGHEST_PROTOCOL)
            f.close()
            
            ## for i in xrange(len(mlp.params)):
            ##     save_params(mlp.params[i], './params_'+str(i)+'.pkl')
            
        ## # Plot network output after this iteration
        ## plt.figure(figsize=(8, 8))
        ## plt.scatter(X_train[0, :], X_train[1, :], c='r',
        ##            lw=.6, s=5)
        ## plt.scatter(current_output[0, :], current_output[1, :], c='b',
        ##            lw=.6, s=5)
        ## plt.axis([-6, 6, -6, 6])
        ## plt.title('Cost: {:.3f}'.format(float(current_cost)))
        ## plt.show()
        iteration += 1
