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
import matplotlib
matplotlib.use('Agg')

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

def train(X_train, layer_sizes, learning_rate, learning_rate_decay, momentum, dampening, \
          lambda_reg, batch_size, time_window, filename, nSingleData, \
          viz=False, rviz=False, max_iteration=100000, save=False, cuda=True, save_pdf=False):

    # Set initial parameter values
    W_init_en = []
    b_init_en = []
    activations_en = []
    W_init_de = []
    b_init_de = []
    activations_de = []

    # Encoder layers
    for n_input, n_output in zip(layer_sizes[:-1], layer_sizes[1:]):
        W_init_en.append(np.random.randn(n_output, n_input).astype('float32'))
        b_init_en.append(np.ones(n_output).astype('float32'))
            
        # We'll use sigmoid activation for all layers
        # Note that this doesn't make a ton of sense when using squared distance
        # because the sigmoid function is bounded on [0, 1].
        activations_en.append( T.nnet.sigmoid ) #T.tanh ) 
            

    # Decoder layers
    layer_sizes = list(reversed(layer_sizes))
    for n_input, n_output in zip(layer_sizes[:-1], layer_sizes[1:]):
        W_init_de.append(np.random.randn(n_output, n_input).astype('float32'))
        b_init_de.append(np.ones(n_output).astype('float32'))
        activations_de.append( T.nnet.sigmoid )  #T.tanh)
            
    ## activations_de[-1] = None

    # Create an instance of the MLP class
    mlp = l.AD(W_init_en, b_init_en, activations_en,
               W_init_de, b_init_de, activations_de, nEncoderLayers=len(layer_sizes)-1)

    # Create Theano variables for the MLP input
    mlp_input = T.fmatrix('mlp_input')
    mlp_target = T.fmatrix('mlp_target')
    cost = mlp.squared_error(mlp_input, mlp_target)
    ## cost = mlp.L1_error(mlp_input, mlp_target)
            
    print 'Creating a theano function for training the network'
    train = theano.function([mlp_input, mlp_target], cost,
                            updates=l.gradient_updates_momentum(cost, mlp.params, learning_rate, \
                                                                learning_rate_decay, momentum, \
                                                                dampening, lambda_reg))
            
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

    if viz or rviz:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)

        plt.ion()        
        if save_pdf == False:
            fig.show()

        if rviz:
            line1, =ax.plot(X_train[0,0:nSingleData], 'b-')
            line2, =ax.plot(X_train[0,0:nSingleData], 'r-')
    

    batch_size = X_train.shape[1]
    train_batch_data = X_train.astype('float32')
    test_batch_data = X_test.astype('float32')
    
    while iteration < max_iteration:

        # training with SGD
        train_loss   = 0.0
        count = 0.0
        for i in range(0, X_train.shape[1]-batch_size+1, batch_size):
            count += 1.0
            if cuda:
                train_loss += train( train_batch_data, train_batch_data )
            else:
                train_loss += train(X_train[:,i:i+batch_size], X_train[:,i:i+batch_size])
                
        train_loss /= np.float32((count*batch_size))
        train_loss /= np.float32(time_window)
        if np.isnan(train_loss) or np.isinf(train_loss):
            print "Train loss is NaN with iter ", iteration
            sys.exit()
        
        if iteration%20 == 0:
            if iteration%200 == 0:
                # testing
                test_loss = mlp_cost(test_batch_data, test_batch_data)/np.float32(len(X_test[0]))

                test_loss /= np.float32(time_window)
                if np.isnan(test_loss) or np.isinf(test_loss):
                    print "Test loss is NaN with iter ", iteration
                    sys.exit()

                print iteration, ' Train loss: ', train_loss, ' test loss: ', test_loss
            else:
                print iteration, ' Train loss: ', train_loss
            
            
        if viz and iteration%20 == 0:
            train_losses.append(train_loss)
            test_losses.append(test_loss)

            if len(train_losses)>3:
                ax.plot(train_losses, 'r-')
                ax.plot(test_losses, 'b-')
                plt.draw()
                ## plt.show()

        if rviz and iteration%100 == 0:

            feature_list = []
            pred = mlp_output( X_test[:,0:nSingleData].astype('float32') )

            line1.set_ydata(X_test[0,0:nSingleData])
            line2.set_ydata(pred[0,:])

            if save_pdf == True and iteration%4000:
                fig.savefig('test.pdf')
                fig.savefig('test.png')
                os.system('cp test.p* ~/Dropbox/HRL/')
            else:
                fig.canvas.draw()
                

            

        if save and iteration%100 ==0:
            f = open(filename, 'wb')
            cPickle.dump(mlp_features, f, protocol=cPickle.HIGHEST_PROTOCOL)
            f.close()
            

       
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

    
def train_pca(X_train, X_test, time_window, nSingleData, outputDim=4, save_pdf=False):

    from sklearn.decomposition import KernelPCA # Too bad
    clf = KernelPCA(n_components=outputDim, kernel="rbf", gamma=5.0)

    print "Train Input size: ", np.shape(X_train)
    print "Test Input size: ", np.shape(X_test)
    X_rd = clf.fit_transform(X_train.T)

    from sklearn.externals import joblib
    filename='./pca.pkl'
    if os.path.isfile(filename):
        clf = joblib.load(filename)
    else:
        joblib.dump(clf, filename)


    feature_list = []
    count = 0
    for i in xrange(0, len(X_test[0]), nSingleData):
        count += 1

        print "Test input size: ", np.shape(X_test[:,i:i+nSingleData].T)
        test_features = clf.transform( X_test[:,i:i+nSingleData].T )
        print "Test output size: ", np.shape(test_features)
        print 
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


def test(X_test, filename, nSingleData, time_window, bReconstruct=False, save_pdf=False):

    mlp_features = load_params(filename)
    print "X_test: ", np.shape( X_test )

    feature_list = []
    count = 0
    for i in xrange(0, len(X_test[0]), nSingleData):
        count += 1
        test_features = mlp_features( X_test[:,i:i+nSingleData].astype('float32') )
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
    p.add_option('--rviz', action='store_true', dest='bReconstructViz',
                 default=False, help='Visualize reconstructed signal')
    p.add_option('--save', action='store_true', dest='bSave',
                 default=False, help='Visualize ....')
    p.add_option('--save_pdf', '--sp', action='store_true', dest='bSavePDF',
                 default=False, help='Save the visualized result as a pdf')
    p.add_option('--renew', action='store_true', dest='bRenew',
                 default=False, help='Renew data ....')
    
    p.add_option('--time_window', '--tw', action='store', dest='nTimeWindow',
                 type="int", default=4, help='Size of time window')
    p.add_option('--learning_rate', '--lr', action='store', dest='fLearningRate',
                 type="float", default=1e-5, help='Learning rate weight')
    p.add_option('--learning_rate_decay', '--lrd', action='store', dest='fLearningRateDecay',
                 type="float", default=1e-5, help='Learning rate weight decay')
    p.add_option('--momentum', '--m', action='store', dest='fMomentum',
                 type="float", default=1e-6, help='momentum')
    p.add_option('--dampening', '--d', action='store', dest='fDampening',
                 type="float", default=1e-6, help='dampening for momentum')
    p.add_option('--lambda', '--lb', action='store', dest='fLambda',
                 type="float", default=1e-6, help='Lambda for regularization ')
    p.add_option('--batch_size', '--bs', action='store', dest='nBatchSize',
                 type="int", default=1024, help='Size of batches ....')
    p.add_option('--layer_size', '--ls', action='store', dest='lLayerSize',
                 ## default="[3]", help='Size of layers ....')
                 default="[256,128]", help='Size of layers ....')
    p.add_option('--maxiter', '--mi', action='store', dest='nMaxIter',
                 type="int", default=100000, help='Max iteration ....')
    
    opt, args = p.parse_args()

    '''
    parameters
    '''
    time_window   = opt.nTimeWindow
    learning_rate = np.float32(opt.fLearningRate)
    learning_rate_decay = np.float32(opt.fLearningRateDecay)
    momentum      = np.float32(opt.fMomentum)
    dampening     = np.float32(opt.fDampening)
    lambda_reg    = np.float32(opt.fLambda)
    batch_size    = opt.nBatchSize
    maxiteration  = opt.nMaxIter
    filename      = 'simple_model.pkl'

    ## X, y = getData2()
    ## X = X.get_value(True).T
    ## X, y = getData()    
    X_train, X_test, nSingleData = dd.getData3(time_window, renew=opt.bRenew, )
    layer_sizes = [X_train.shape[0]] + eval(opt.lLayerSize) #, 20, 10, 5]
    print layer_sizes
    ## print "Max iteration : ", opt.nMaxIter

    if opt.bTrain:        
        ## train_pca(X_train, X_test, time_window, nSingleData, \
        ##           outputDim=eval(opt.lLayerSize)[-1], save_pdf=opt.bSavePDF)
        train(X_train, layer_sizes, learning_rate, learning_rate_decay, momentum, dampening, \
              lambda_reg, batch_size, time_window, filename, nSingleData, \
              viz=opt.bViz, max_iteration=maxiteration, save=opt.bSave, rviz=opt.bReconstructViz)
    else:
        test(X_test, filename, nSingleData, time_window, save_pdf=opt.bSavePDF)
        
