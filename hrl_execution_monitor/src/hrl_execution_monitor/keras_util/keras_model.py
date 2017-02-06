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

# system & utils
import os, sys, copy, random
import scipy, numpy as np
import hrl_lib.util as ut
import gc

import h5py 
import cv2

from keras.models import Sequential, Model
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, Merge, Input
from keras.layers import Activation, Dropout, Flatten, Dense, merge
from keras.layers.advanced_activations import PReLU
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD, Adagrad, Adadelta, RMSprop
from keras.utils.visualize_util import plot
from keras.layers.normalization import BatchNormalization
from keras.regularizers import EigenvalueRegularizer, L1L2Regularizer

from hrl_execution_monitor.keras_util import model_util as mutil
random.seed(3334)
np.random.seed(3334)

vgg_model_weights_path = os.path.expanduser('~')+'/git/keras_test/vgg16_weights.h5'


def cnn_net(input_shape, n_labels, weights_path=None, sig_weights_path=None,
            with_top=False, input_shape2=None,
            fine_tune=False, activ_type='relu'):

    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=input_shape, name='conv1_1'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(32, 3, 3, name='conv2_1'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 3, 3, name='conv3_1'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))


    if with_top or fine_tune:
        for layer in model.layers:
            layer.trainable = False

    if with_top:
        model.add(Flatten(input_shape=model.output_shape[1:]))
        model.add(Dense(64, name='fc1_1'))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))        
        if img_weights_path is not None:
            model.load_weights(img_weights_path, by_name=True)
        
        sig_model = Sequential()
        sig_model.add(Dense(128, activation='relu', init='uniform', input_shape=input_shape2,\
                            name='fc2_1'))
        sig_model.add(Dropout(0.5))
        if sig_weights_path is not None:
            sig_model.load_weights(sig_weights_path)

        merge = Merge([model, sig_model], mode='concat')
        c_model = Sequential()
        c_model.add(merge)        
        c_model.add(Dense(64, activation='relu', init='uniform', name='fc3_1'))
        c_model.add(Dropout(0.5))      
        c_model.add(Dense(n_labels, activation='softmax', name='fc_out'))
        if weights_path is not None:
            c_model.load_weights(weights_path)
            
        return c_model
    
    else:
        # the model so far outputs 3D feature maps (height, width, features)
        t_model = Sequential()
        t_model.add(Flatten(input_shape=model.output_shape[1:]))
        t_model.add(Dense(64, name='fc1_1'))
        t_model.add(Activation('relu'))
        t_model.add(Dropout(0.5))        
        t_model.add(Dense(n_labels, activation='softmax', name='fc_img_out'))
        model.add(t_model)
        if weights_path is not None:
            model.load_weights(weights_path)

        return model


def sig_net(input_shape, n_labels, weights_path=None, activ_type='relu' ):

    if activ_type == 'PReLU': activ_type = PReLU(init='zero', weights=None)

    model = Sequential()
    model.add(Dense(128, init='uniform', input_shape=input_shape,
                    W_regularizer=L1L2Regularizer(0.0,0.0),\
                    name='fc2_1'))
    model.add(Activation(activ_type))
    model.add(Dropout(0.2))
    ## model.add(Dense(128, init='uniform', input_shape=input_shape,
    ##                 W_regularizer=L1L2Regularizer(0.0,0.0),\
    ##                 name='fc2_2'))
    ## model.add(Activation(activ_type))
    ## model.add(Dropout(0.2))

    model.add(Dense(n_labels, activation='softmax',W_regularizer=L1L2Regularizer(0,0),
                    name='fc_sig_out'))

    if weights_path is not None:
        model.load_weights(weights_path)

    return model


def vgg16_net(input_shape, n_labels, weights_path=None,\
              sig_weights_path=None, img_weights_path=None,\
              with_multi_top=False, with_img_top=False, bottle_model=False, \
              input_shape2=None, fine_tune=False, viz=False):
    
    # build the VGG16 network
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=input_shape))

    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    if vgg_model_weights_path is not None:       
        # load the weights of the VGG16 networks
        # (trained on ImageNet, won the ILSVRC competition in 2014)
        # note: when there is a complete match between your model definition
        # and your weight savefile, you can simply call model.load_weights(filename)
        assert os.path.exists(vgg_model_weights_path), \
          'Model weights not found (see "weights_path" variable in script).'
        f = h5py.File(vgg_model_weights_path)
        for k in range(f.attrs['nb_layers']):
            if k >= len(model.layers):
                # we don't look at the last (fully-connected) layers in the savefile
                break
            g = f['layer_{}'.format(k)]
            weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
            model.layers[k].set_weights(weights)
        f.close()
        print('Model loaded.')
    # 31 layers---------------------------------------------------------------


    if fine_tune and False:
        for layer in model.layers[:25]:
            layer.trainable = False
    else:
        for layer in model.layers:
            layer.trainable = False


    if with_multi_top:
        model.add(Flatten())

        # -----------------------------------------------------------
        weights_file = None
        if img_weights_path:
            weights_file = h5py.File(img_weights_path)
        else:
            weights_file = h5py.File(weights_path)
            
        weight1_1 = mutil.get_layer_weights(weights_file, layer_name='fc1_1')                
        model.add(Dense(1024, init='uniform', weights=weight1_1, name='fc1_1',
                        W_regularizer=L1L2Regularizer(0.0,0.03)))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))        
        # -----------------------------------------------------------
        weight1_2 = mutil.get_layer_weights(weights_file, layer_name='fc1_2')        
        model.add(Dense(128, init='uniform', weights=weight1_2, name='fc1_2',
                        W_regularizer=L1L2Regularizer(0.0,0.01)))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))        
        ## # -----------------------------------------------------------
        ## weight1_3 = mutil.get_layer_weights(weights_file, layer_name='fc1_3')        
        ## model.add(Dense(64, init='uniform', weights=weight1_3, name='fc1_3',
        ##                 W_regularizer=L1L2Regularizer(0.0,0.01)))
        ## model.add(Activation('relu'))
        ## model.add(Dropout(0.5))        


        if not fine_tune:
            for layer in model.layers[31:]:
                layer.trainable = False
        ## else:
        ##     for layer in model.layers:
        ##         layer.trainable = False
            
        if weights_path: weights_file.close()        
        # -----------------------------------------------------------
        # -----------------------------------------------------------
        weights_file = None
        if sig_weights_path:
            weights_file = h5py.File(sig_weights_path)
        else:
            weights_file = h5py.File(weights_path, 'r')
        weight2_1 = mutil.get_layer_weights(weights_file, layer_name='fc2_1')
        
        sig_model = Sequential()
        sig_model.add(Dense(128, init='uniform', weights=weight2_1,
                            input_shape=input_shape2, name='fc2_1',
                            W_regularizer=L1L2Regularizer(0.0,0.0)))
        sig_model.add(Activation('relu'))        
        sig_model.add(Dropout(0.2))

        ## weight2_2 = mutil.get_layer_weights(weights_file, layer_name='fc_sig_out')
        ## sig_model.add(Dense(n_labels, init='uniform', weights=weight2_2, name='fc_sig_out'))
        ## sig_model.add(Activation('relu'))        

        if not fine_tune:
            for layer in sig_model.layers:
                layer.trainable = False
        
        if weights_path: weights_file.close()        
        # -----------------------------------------------------------
        merge   = Merge([model, sig_model], mode='concat') #mode='mul') 
        c_model = Sequential()
        c_model.add(merge)

        if bottle_model: return c_model            
        if weights_path: weights_file = h5py.File(weights_path, 'r')
        else: weights_file = None
        weight3_1 = mutil.get_layer_weights(weights_file, layer_name='fc3_1')
        ## weight3_2 = mutil.get_layer_weights(weights_file, layer_name='fc3_2')
        weight_fc_out = mutil.get_layer_weights(weights_file, layer_name='fc_out')        
                
        c_model.add(Dense(256, activation='relu', init='uniform', weights=weight3_1, name='fc3_1',
                          W_regularizer=L1L2Regularizer(0,0.05)))
        c_model.add(Dropout(0.0))
        ## c_model.add(Dense(256, activation='relu', init='uniform', weights=weight3_2, name='fc3_2',
        ##                   W_regularizer=L1L2Regularizer(0,0.05)))
        ## c_model.add(Dropout(0.6))
        
        c_model.add(Dense(n_labels, activation='softmax', name='fc_out', weights=weight_fc_out,
                    W_regularizer=L1L2Regularizer(0,0)))
        
        if weights_path: weights_file.close()        
        return c_model
    
    elif with_img_top:
        model.add(Flatten())
        model.add(Dense(1024, init='uniform', name='fc1_1', W_regularizer=L1L2Regularizer(0.0,0.03)))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))        
        model.add(Dense(64, init='uniform', name='fc1_2', W_regularizer=L1L2Regularizer(0.0,0.01)))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))        
        model.add(Dense(n_labels, activation='softmax', name='fc_img_out'))

        if weights_path is not None: model.load_weights(weights_path)
        return model
    else:
        return model




        ## if weights_path is not None:
        ##     weights_file = h5py.File(weights_path)
        ## else: weights_file = None
        ## weight = mutil.get_layer_weights(weights_file, layer_name='fc1_1')
        ## weight2 = mutil.get_layer_weights(weights_file, layer_name='conv5_3')
        ## print weight
        ## print weight2

        ## t_model = Sequential()
        ## t_model.add(Flatten(input_shape=model.output_shape[1:]))

        ## if weights_path is not None:
        ##     weights_file = h5py.File(weights_path)
        ## else: weights_file = None
        ## weight = mutil.get_layer_weights(weights_file, layer_name='fc1_1')
        ## print "aaaaaaaaaaaa"
        ## print weight
        ## print "aaaaaaaaaaaa"



def vgg_image_top_net(input_shape, n_labels, weights_path=None):

    model = Sequential()
    model.add(Flatten(input_shape=input_shape))


    model.add(Dense(1024, init='uniform', name='fc1_1', W_regularizer=L1L2Regularizer(0.0,0.03)))
    model.add(Activation('relu')) #0.03
    model.add(Dropout(0.5))        
    ## model.add(Dense(1024, init='uniform', name='fc1_2', W_regularizer=L1L2Regularizer(0.0,0.01)))
    ## model.add(Activation('relu'))
    ## model.add(Dropout(0.5))
    model.add(Dense(128, init='uniform', name='fc1_2', W_regularizer=L1L2Regularizer(0.0,0.01)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5)) #0.5

    
    ## model.add(Dense(256, init='uniform', name='fc1_1', W_regularizer=L1L2Regularizer(0.0,0.01)))
    ## model.add(Activation('relu'))
    ## ## model.add(Dropout(0.4))        
    ## model.add(Dense(64, init='uniform', name='fc1_2', W_regularizer=L1L2Regularizer(0.0,0.01)))
    ## model.add(Activation('relu'))
    ## ## model.add(Dropout(0.4))

    
    model.add(Dense(n_labels, activation='softmax', name='fc_img_out'))

    if weights_path is not None: model.load_weights(weights_path)
    return model


def vgg_multi_top_net(input_shape, n_labels, weights_path=None):

    model = Sequential()

    weights_file = None
    if weights_path is not None: weights_file = h5py.File(weights_path)
    weight = mutil.get_layer_weights(weights_file, layer_name='fc3_1')    
    model.add(Dense(256, activation='relu', init='uniform', name='fc3_1', input_shape=input_shape,
                    W_regularizer=L1L2Regularizer(0.0, 0.05), weights=weight))
    model.add(Dropout(0.0))

    # -------------------------------------------------------------------
    weight = mutil.get_layer_weights(weights_file, layer_name='fc3_2')    
    model.add(Dense(256, activation='relu', init='uniform', name='fc3_2',
                    W_regularizer=L1L2Regularizer(0,0.05 ), weights=weight))    
    model.add(Dropout(0.0))

    weight = mutil.get_layer_weights(weights_file, layer_name='fc_out')
    model.add(Dense(n_labels, activation='softmax', name='fc_out', input_shape=input_shape,
                    weights=weight, W_regularizer=L1L2Regularizer(0,0.0)))

    if weights_path is not None:  weights_file.close()
    return model
