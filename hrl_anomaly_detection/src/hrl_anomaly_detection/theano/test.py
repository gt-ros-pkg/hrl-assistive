import theano
import theano.tensor as T
import layer as l
from theano import function, config, shared, sandbox

# system util
import sys, os
import numpy as np
import matplotlib.pyplot as plt
## import cPickle
from six.moves import cPickle
import random

# util
import hrl_lib.util as ut
## from hrl_anomaly_detection.util import *
## from hrl_anomaly_detection.util_viz import *

import data as dd
import itertools
colors = itertools.cycle(['r', 'g', 'b', 'm', 'c', 'k', 'y'])
shapes = itertools.cycle(['x','v', 'o', '+'])

def save_params(obj, filename):
    f = file(filename, 'wb')
    cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

def load_params(filename):
    obj = ut.load_pickle(filename)
    return obj

def train(X_train, layer_sizes, learning_rate, momentum, lambda_reg, batch_size, time_window, \
          filename, nSingleData, \
          viz=False, max_iteration=100000, save=False, cuda=False):

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
        activations_en.append( T.tanh ) #T.nnet.tanh
            

    # Decoder layers
    layer_sizes = list(reversed(layer_sizes))
    for n_input, n_output in zip(layer_sizes[:-1], layer_sizes[1:]):
        W_init_de.append(np.random.randn(n_output, n_input))
        b_init_de.append(np.ones(n_output))
        activations_de.append(T.tanh)
            
    ## activations_de[-1] = None

    # Create an instance of the MLP class
    mlp = l.AD(W_init_en, b_init_en, activations_en,
               W_init_de, b_init_de, activations_de, nEncoderLayers=len(layer_sizes)-1)

    # Create Theano variables for the MLP input
    mlp_input = T.dmatrix('mlp_input')
    mlp_target = T.dmatrix('mlp_target')
    cost = mlp.squared_error(mlp_input, mlp_target)
    print 'Creating a theano function for training the network'
    train = theano.function([mlp_input, mlp_target], cost,
                            updates=l.gradient_updates_momentum(cost, mlp.params, learning_rate, momentum, \
                                                                lambda_reg))
    print 'Creating a theano function for computing the MLP\'s output given some input'
    mlp_output = theano.function([mlp_input], mlp.output(mlp_input))
    print 'Creating a theano function for computing the MLP\'s output given some input'
    mlp_features = theano.function([mlp_input], mlp.get_features(mlp_input))

    mlp_cost = theano.function([mlp_input, mlp_target], cost)


    print 'Optimising'
    # Keep track of the number of training iterations performed
    iteration = 0
    train_losses = []
    test_losses  = []

    if viz:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        plt.ion()
        fig.show()
        ## line1, =ax.plot(X_train[0,0:nSingleData], 'b-')
        ## line2, =ax.plot(X_train[0,0:nSingleData], 'r-')
    
    
    while iteration < max_iteration:

        # training with SGD
        train_loss   = 0.0
        count = 0.0
        for i in range(0, X_train.shape[1]-batch_size+1, batch_size):
            count += 1.0
            if cuda:
                train_loss += train( shared(X_train[:,i:i+batch_size], borrow=True), \
                                     shared(X_train[:,i:i+batch_size], borrow=True))
            else:
                train_loss += train(X_train[:,i:i+batch_size], X_train[:,i:i+batch_size])
                
        train_loss /= (count*batch_size)
        train_loss /= float(time_window)
        if np.isnan(train_loss) or np.isinf(train_loss):
            print "Train loss is NaN with iter ", iteration
            sys.exit()
        
        if iteration%20 == 0:
            # testing
            if cuda:
                test_loss = mlp_cost( shared(X_test, borrow=True),
                                      shared(X_test, borrow=True))/float(len(X_test[0]))
            else:
                test_loss = mlp_cost(X_test, X_test)/float(len(X_test[0]))
                
            test_loss /= float(time_window)
            if np.isnan(test_loss) or np.isinf(test_loss):
                print "Test loss is NaN with iter ", iteration
                sys.exit()
            
            print iteration, ' Train loss: ', train_loss, ' test loss: ', test_loss
            
        if viz and iteration%20 == 0:
            train_losses.append(train_loss)
            test_losses.append(test_loss)

            if len(train_losses)>3:
                ax.plot(train_losses, 'r-')
                ax.plot(test_losses, 'b-')
                plt.draw()
                ## plt.show()

        if save and iteration%100 ==0:
            f = open(filename, 'wb')
            cPickle.dump(mlp_features, f, protocol=cPickle.HIGHEST_PROTOCOL)
            f.close()
            

        ## feature_list = []
        ## pred = mlp_output( X_test[:,0:nSingleData] )

        ## line1.set_ydata(X_test[0,0:nSingleData])
        ## line2.set_ydata(pred[0,:])
        ## fig.canvas.draw()
        


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
    

def test(X_test, filename, nSingleData, time_window, bReconstruct=False):

    mlp_features = load_params(filename)
    err = 0.0

    print "X_test: ", np.shape( X_test )

    feature_list = []
    count = 0
    for i in xrange(0, len(X_test[0]), nSingleData):
        count += 1
        test_features = mlp_features( X_test[:,i:i+nSingleData] )
        feature_list.append(test_features)

        print "Total samples : ", count

        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        for i in xrange(len(feature_list)):

            # get mean, var        
            colors = itertools.cycle(['r', 'g', 'b', 'm', 'c', 'k', 'y'])        
            for j in xrange(3):
                color = colors.next()
                ax.plot(feature_list[i][j,:], c=color)

            ## if i==3:
            ##     break
        plt.show()

    
if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    p.add_option('--train', '--tr', action='store_true', dest='bTrain',
                 default=False, help='Train ....')
    p.add_option('--test', '--t', action='store_true', dest='bTest',
                 default=False, help='Test ....')
    p.add_option('--viz', action='store_true', dest='bViz',
                 default=False, help='Visualize ....')
    p.add_option('--save', action='store_true', dest='bSave',
                 default=False, help='Visualize ....')
    p.add_option('--renew', action='store_true', dest='bRenew',
                 default=False, help='Renew data ....')
    
    p.add_option('--time_window', '--tw', action='store', dest='nTimeWindow',
                 type="int", default=4, help='Size of time window ....')
    p.add_option('--learning_rate', '--lr', action='store', dest='fLearningRate',
                 type="float", default=0.001, help='Size of time window ....')
    p.add_option('--momentum', '--m', action='store', dest='fMomentum',
                 type="float", default=1e-5, help='Size of time window ....1e-5')
    p.add_option('--lambda', '--lb', action='store', dest='fLambda',
                 type="float", default=1e-5, help='Lambda for regularization 1e-8')
    p.add_option('--batch_size', '--bs', action='store', dest='nBatchSize',
                 type="int", default=64, help='Size of batches ....')

    p.add_option('--layer_size', '--ls', action='store', dest='lLayerSize',
                 default="[256, 64, 16]", help='Size of layers ....')
    p.add_option('--maxiter', '--mi', action='store', dest='nMaxIter',
                 type="int", default=100000, help='Max iteration ....')
    
    opt, args = p.parse_args()

    '''
    parameters
    '''
    time_window   = opt.nTimeWindow
    learning_rate = opt.fLearningRate
    momentum      = opt.fMomentum
    lambda_reg    = opt.fLambda
    batch_size    = opt.nBatchSize
    maxiteration  = opt.nMaxIter
    filename      = 'simple_model.pkl'

    ## X, y = getData2()
    ## X = X.get_value(True).T
    ## X, y = getData()    
    X_train, X_test, nSingleData = dd.getData3(time_window, renew=opt.bRenew)
    layer_sizes = [X_train.shape[0]] + eval(opt.lLayerSize) #, 20, 10, 5]
    print layer_sizes
    print "Max iteration : ", opt.nMaxIter

    if opt.bTrain:
        train(X_train, layer_sizes, learning_rate, momentum, lambda_reg, batch_size, time_window, filename,\
              nSingleData,\
              viz=opt.bViz, max_iteration=maxiteration, save=opt.bSave)
    else:
        test(X_test, filename, nSingleData, time_window)
        
