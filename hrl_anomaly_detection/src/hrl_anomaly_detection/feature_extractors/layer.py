import sys
import numpy as np
import matplotlib.pyplot as plt
import theano
# By convention, the tensor submodule is loaded as T
import theano.tensor as T


class Layer(object):
    def __init__(self, W_init, b_init, activation):
        '''
        A layer of a neural network, computes s(Wx + b) where s is a nonlinearity and x is the input vector.
        
        :parameters:
        - W_init : np.ndarray, shape=(n_output, n_input)
                   Values to initialize the weight matrix to.
        - b_init : np.ndarray, shape=(n_output,)
                   Values to initialize the bias vector
        - activation : theano.tensor.elemwise.Elemwise
                       Activation function for layer output
        '''
        n_output, n_input = W_init.shape
        assert b_init.shape == (n_output,)
        self.W = theano.shared(value=W_init.astype('float32'),
                               name='W',
                               borrow=True)
        self.b = theano.shared(value=b_init.reshape(n_output, 1).astype('float32'),
                               name='b',
                               borrow=True,
                               broadcastable=(False, True))
        self.activation = activation
        self.params = [self.W, self.b]

    def output(self, x):
        '''
        Compute this layer's output given an input
        
        :parameters:
            - x : theano.tensor.var.TensorVariable
                  Theano symbolic variable for layer input
        
        :returns:
            - output : theano.tensor.var.TensorVariable
                Mixed, biased, and activated x
        '''
        # Compute linear mix
        lin_output = T.dot(self.W, x) + self.b
        return (lin_output if self.activation is None else self.activation(lin_output))


class MLP(object):
    def __init__(self, W_init, b_init, activations):
        '''
        Multi-layer perceptron class, computes the composition of a sequence of Layers

        :parameters:
            - W_init : list of np.ndarray, len=N
                Values to initialize the weight matrix in each layer to.
                The layer sizes will be inferred from the shape of each matrix in W_init
            - b_init : list of np.ndarray, len=N
                Values to initialize the bias vector in each layer to
            - activations : list of theano.tensor.elemwise.Elemwise, len=N
                Activation function for layer output for each layer
        '''
        # Make sure the input lists are all of the same length
        assert len(W_init) == len(b_init) == len(activations)
        
        self.layers = []
        for W, b, activation in zip(W_init, b_init, activations):
            self.layers.append(Layer(W, b, activation))

        self.params = []
        for layer in self.layers:
            self.params += layer.params
        
    def output(self, x):
        '''
        Compute the MLP's output given an input
        
        :parameters:
            - x : theano.tensor.var.TensorVariable
                Theano symbolic variable for network input

        :returns:
            - output : theano.tensor.var.TensorVariable
                x passed through the MLP
        '''
        # Recursively compute output
        for layer in self.layers:
            x = layer.output(x)
        return x

    def squared_error(self, x, y):
        '''
        Compute the squared euclidean error of the network output against the "true" output y
        
        :parameters:
            - x : theano.tensor.var.TensorVariable
                Theano symbolic variable for network input
            - y : theano.tensor.var.TensorVariable
                Theano symbolic variable for desired network output

        :returns:
            - error : theano.tensor.var.TensorVariable
                The squared Euclidian distance between the network output and y
        '''
        return T.sum((self.output(x) - y)**2) 


class AD(object):
    def __init__(self, W_init_en, b_init_en, activations_en,
                 W_init_de, b_init_de, activations_de, nEncoderLayers):
        '''
        Multi-layer perceptron class, computes the composition of a sequence of Layers

        :parameters:
            - W_init : list of np.ndarray, len=N
                Values to initialize the weight matrix in each layer to.
                The layer sizes will be inferred from the shape of each matrix in W_init
            - b_init : list of np.ndarray, len=N
                Values to initialize the bias vector in each layer to
            - activations : list of theano.tensor.elemwise.Elemwise, len=N
                Activation function for layer output for each layer
        '''
        # Make sure the input lists are all of the same length
        assert len(W_init_en) == len(b_init_en) == len(activations_en)
        assert len(W_init_de) == len(b_init_de) == len(activations_de)

        self.nEncoderLayers = nEncoderLayers
        self.layers = []
        for W, b, activation in zip(W_init_en, b_init_en, activations_en):
            self.layers.append(Layer(W, b, activation))
        for W, b, activation in zip(W_init_de, b_init_de, activations_de):
            self.layers.append(Layer(W, b, activation))

        self.params = []
        for layer in self.layers:
            self.params += layer.params
        
        
    def output(self, x):
        '''
        Compute the MLP's output given an input
        
        :parameters:
            - x : theano.tensor.var.TensorVariable
                Theano symbolic variable for network input

        :returns:
            - output : theano.tensor.var.TensorVariable
                x passed through the MLP
        '''
        # Recursively compute output
        for layer in self.layers:
            x = layer.output(x)
        return x

    def get_features(self, x):
        
        # Recursively compute output
        for i, layer in enumerate(self.layers):
            x = layer.output(x)
            if i == self.nEncoderLayers-1:
                break
        return x
        

    def squared_error(self, x, y):
        '''
        Compute the squared euclidean error of the network output against the "true" output y
        
        :parameters:
            - x : theano.tensor.var.TensorVariable
                Theano symbolic variable for network input
            - y : theano.tensor.var.TensorVariable
                Theano symbolic variable for desired network output

        :returns:
            - error : theano.tensor.var.TensorVariable
                The squared Euclidian distance between the network output and y
        '''
        return T.sum((self.output(x) - y)**2)


    def L1_error(self, x, y):
        '''
        Compute the L1 error of the network output against the "true" output y
        
        :parameters:
            - x : theano.tensor.var.TensorVariable
                Theano symbolic variable for network input
            - y : theano.tensor.var.TensorVariable
                Theano symbolic variable for desired network output

        :returns:
            - error : theano.tensor.var.TensorVariable
                The error between the network output and y
        '''
        return T.sum(abs(self.output(x) - y))



def gradient_updates_momentum(cost, params, learning_rate, learning_rate_decay, momentum, dampening, \
                              lambda_reg):
    '''
    Compute updates for gradient descent with momentum
    
    :parameters:
        - cost : theano.tensor.var.TensorVariable
            Theano cost function to minimize
        - params : list of theano.tensor.var.TensorVariable
            Parameters to compute gradient against
        - learning_rate : float
            Gradient descent learning rate
        - learning_rate_decay : float
            Gradient descent learning rate decay
        - momentum : float
            Momentum parameter, should be at least 0 (standard gradient descent) and less than 1
        - dampening : float
            dampening for momentum
        - lambda_reg : float
            Regularization weight
   
    :returns:
        updates : list
            List of updates, one for each parameter
    '''
    # Make sure momentum is a sane value
    assert momentum < 1 and momentum >= 0
    # List of update steps for each parameter
    updates = []
    # Just gradient descent on cost
    for param in params:

        param_update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
        updates.append((param, param - learning_rate/(np.float32(1.0)+learning_rate_decay)\
                        *( momentum*param_update \
                           +(np.float32(1.)-dampening)*(T.grad(cost, param) + lambda_reg*param))\
                           ))
        updates.append((param_update, momentum*param_update \
                        +(np.float32(1.)-dampening)*(T.grad(cost, param) + lambda_reg*param)
                        )) 
        
    return updates

