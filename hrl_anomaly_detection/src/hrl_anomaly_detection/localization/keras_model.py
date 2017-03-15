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
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, Merge, Input, Lambda
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, merge
from keras.layers.advanced_activations import PReLU
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD, Adagrad, Adadelta, RMSprop
from keras.utils.visualize_util import plot
from keras.layers.normalization import BatchNormalization
from keras.regularizers import EigenvalueRegularizer, L1L2Regularizer
from keras import backend as K

from hrl_execution_monitor.keras_util import model_util as mutil
random.seed(3334)
np.random.seed(3334)

vgg_weights_path = os.path.expanduser('~')+'/git/keras_test/vgg16_weights.h5'
#TF_VGG_WEIGHTS_PATH = 'https://github.com/dpark_setup/keras_weights/vgg16_weights.h5'



def sig_net(input_shape, n_labels, weights_path=None, activ_type='relu', with_top=True ):

    if activ_type == 'PReLU': activ_type = PReLU(init='zero', weights=None)

    weights_file = None
    if weights_path is not None: weights_file = h5py.File(weights_path)

    model = Sequential()
    weight_1 = mutil.get_layer_weights(weights_file, layer_name='fc_sig_1')        
    model.add(Dense(64, init='uniform', input_shape=input_shape,
                    W_regularizer=L1L2Regularizer(0.0,0.01),\
                    weights=weight_1, name='fc_sig_1'))
    model.add(Activation(activ_type))
    model.add(Dropout(0.1))


    ## model.add(Dense(128, init='uniform', input_shape=input_shape,
    ##                 W_regularizer=L1L2Regularizer(0.0,0.0),\
    ##                 weights=weight_1, name='fc_sig_1'))
    ## model.add(Activation(activ_type))
    ## model.add(Dropout(0.2))
    
    weight_2 = mutil.get_layer_weights(weights_file, layer_name='fc_sig_2')        
    model.add(Dense(64, init='uniform', weights=weight_2,\
                    W_regularizer=L1L2Regularizer(0.0,0.01), \
                    name='fc_sig_2'))
    model.add(Activation(activ_type))
    model.add(Dropout(0.))

    if with_top is False:
        weight_out = mutil.get_layer_weights(weights_file, layer_name='fc_sig_out')        
        model.add(Dense(n_labels, activation='softmax',W_regularizer=L1L2Regularizer(0,0),
                        weights=weight_out, name='fc_sig_out'))

    ## if weights_path is not None: model.load_weights(weights_path)
    return model


def vgg16_net(input_shape, n_labels=None, weights_path=None,\
              fine_tune=False):

    weights_file = None
    if weights_path is not None: weights_file = h5py.File(weights_path)
              
    # build the VGG16 network
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=input_shape))

    weight_11 = mutil.get_layer_weights(weights_file, layer_name='conv1_1')        
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1', weights=weight_11))
    model.add(ZeroPadding2D((1, 1)))
    weight_12 = mutil.get_layer_weights(weights_file, layer_name='conv1_2')        
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2', weights=weight_12))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    weight_21 = mutil.get_layer_weights(weights_file, layer_name='conv2_1')        
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1', weights=weight_21))
    model.add(ZeroPadding2D((1, 1)))
    weight_22 = mutil.get_layer_weights(weights_file, layer_name='conv2_2')        
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2', weights=weight_22))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    weight_31 = mutil.get_layer_weights(weights_file, layer_name='conv3_1')        
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1', weights=weight_31))
    model.add(ZeroPadding2D((1, 1)))
    weight_32 = mutil.get_layer_weights(weights_file, layer_name='conv3_2')        
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2', weights=weight_32))
    model.add(ZeroPadding2D((1, 1)))
    weight_33 = mutil.get_layer_weights(weights_file, layer_name='conv3_3')        
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3', weights=weight_33))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    weight_41 = mutil.get_layer_weights(weights_file, layer_name='conv4_1')        
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1', weights=weight_41))
    model.add(ZeroPadding2D((1, 1)))
    weight_42 = mutil.get_layer_weights(weights_file, layer_name='conv4_2')        
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2', weights=weight_42))
    model.add(ZeroPadding2D((1, 1)))
    weight_43 = mutil.get_layer_weights(weights_file, layer_name='conv4_3')        
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3', weights=weight_43))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    weight_51 = mutil.get_layer_weights(weights_file, layer_name='conv5_1')        
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1', weights=weight_51))
    model.add(ZeroPadding2D((1, 1)))
    weight_52 = mutil.get_layer_weights(weights_file, layer_name='conv5_2')        
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2', weights=weight_52))
    model.add(ZeroPadding2D((1, 1)))
    weight_53 = mutil.get_layer_weights(weights_file, layer_name='conv5_3')        
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3', weights=weight_53))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    if weights_path is None:
        weights_path = vgg_weights_path 
        
        # load the weights of the VGG16 networks
        # (trained on ImageNet, won the ILSVRC competition in 2014)
        # note: when there is a complete match between your model definition
        # and your weight savefile, you can simply call model.load_weights(filename)
        assert os.path.exists(weights_path), \
          'Model weights not found (see "weights_path" variable in script).'
        f = h5py.File(weights_path)

        print weights_path
        print f.attrs.keys()

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

    if fine_tune:
        for layer in model.layers[:25]:
            layer.trainable = False
    else:
        for layer in model.layers:
            layer.trainable = False

    return model

def vgg16_net2(input_shape, n_labels=None, weights_path=None,\
              fine_tune=False):

    weights_file = None
    if weights_path is not None: weights_file = h5py.File(weights_path)
              
    # build the VGG16 network
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=input_shape))

    weight_11 = mutil.get_layer_weights(weights_file, layer_name='2_conv1_1')        
    model.add(Convolution2D(64, 3, 3, activation='relu', name='2_conv1_1', weights=weight_11))
    model.add(ZeroPadding2D((1, 1)))
    weight_12 = mutil.get_layer_weights(weights_file, layer_name='2_conv1_2')        
    model.add(Convolution2D(64, 3, 3, activation='relu', name='2_conv1_2', weights=weight_12))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    weight_21 = mutil.get_layer_weights(weights_file, layer_name='2_conv2_1')        
    model.add(Convolution2D(128, 3, 3, activation='relu', name='2_conv2_1', weights=weight_21))
    model.add(ZeroPadding2D((1, 1)))
    weight_22 = mutil.get_layer_weights(weights_file, layer_name='2_conv2_2')        
    model.add(Convolution2D(128, 3, 3, activation='relu', name='2_conv2_2', weights=weight_22))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    weight_31 = mutil.get_layer_weights(weights_file, layer_name='2_conv3_1')        
    model.add(Convolution2D(256, 3, 3, activation='relu', name='2_conv3_1', weights=weight_31))
    model.add(ZeroPadding2D((1, 1)))
    weight_32 = mutil.get_layer_weights(weights_file, layer_name='2_conv3_2')        
    model.add(Convolution2D(256, 3, 3, activation='relu', name='2_conv3_2', weights=weight_32))
    model.add(ZeroPadding2D((1, 1)))
    weight_33 = mutil.get_layer_weights(weights_file, layer_name='2_conv3_3')        
    model.add(Convolution2D(256, 3, 3, activation='relu', name='2_conv3_3', weights=weight_33))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    weight_41 = mutil.get_layer_weights(weights_file, layer_name='2_conv4_1')        
    model.add(Convolution2D(512, 3, 3, activation='relu', name='2_conv4_1', weights=weight_41))
    model.add(ZeroPadding2D((1, 1)))
    weight_42 = mutil.get_layer_weights(weights_file, layer_name='2_conv4_2')        
    model.add(Convolution2D(512, 3, 3, activation='relu', name='2_conv4_2', weights=weight_42))
    model.add(ZeroPadding2D((1, 1)))
    weight_43 = mutil.get_layer_weights(weights_file, layer_name='2_conv4_3')        
    model.add(Convolution2D(512, 3, 3, activation='relu', name='2_conv4_3', weights=weight_43))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    weight_51 = mutil.get_layer_weights(weights_file, layer_name='2_conv5_1')        
    model.add(Convolution2D(512, 3, 3, activation='relu', name='2_conv5_1', weights=weight_51))
    model.add(ZeroPadding2D((1, 1)))
    weight_52 = mutil.get_layer_weights(weights_file, layer_name='2_conv5_2')        
    model.add(Convolution2D(512, 3, 3, activation='relu', name='2_conv5_2', weights=weight_52))
    model.add(ZeroPadding2D((1, 1)))
    weight_53 = mutil.get_layer_weights(weights_file, layer_name='2_conv5_3')        
    model.add(Convolution2D(512, 3, 3, activation='relu', name='2_conv5_3', weights=weight_53))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    if weights_path is None:
        weights_path = vgg_weights_path 
        
        # load the weights of the VGG16 networks
        # (trained on ImageNet, won the ILSVRC competition in 2014)
        # note: when there is a complete match between your model definition
        # and your weight savefile, you can simply call model.load_weights(filename)
        assert os.path.exists(weights_path), \
          'Model weights not found (see "weights_path" variable in script).'
        f = h5py.File(weights_path)

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

    if fine_tune:
        for layer in model.layers[:25]:
            layer.trainable = False
    else:
        for layer in model.layers:
            layer.trainable = False

    return model




def vgg_image_top_net(input_shape, n_labels, weights_path=None):

    model = Sequential()

    
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(1024, init='uniform', name='fc1_1', W_regularizer=L1L2Regularizer(0.0,0.03)))
    model.add(Activation('relu')) 
    model.add(Dropout(0.5))        
    model.add(Dense(128, init='uniform', name='fc1_2', W_regularizer=L1L2Regularizer(0.0,0.01)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_labels, activation='softmax', name='fc_img_out'))


    ## model.add(GlobalAveragePooling2D(input_shape=input_shape, dim_ordering='th'))
    ## model.add(Dense(n_labels, activation='softmax', name='fc_img_out'))
    ## ## model.add(Activation('softmax'))
    

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
    ## weight = mutil.get_layer_weights(weights_file, layer_name='fc3_2')    
    ## model.add(Dense(256, activation='relu', init='uniform', name='fc3_2',
    ##                 W_regularizer=L1L2Regularizer(0,0.05 ), weights=weight))    
    ## model.add(Dropout(0.2))

    weight = mutil.get_layer_weights(weights_file, layer_name='fc_out')
    model.add(Dense(n_labels, activation='softmax', name='fc_out', input_shape=input_shape,
                    weights=weight, W_regularizer=L1L2Regularizer(0,0.0)))

    if weights_path is not None:  weights_file.close()
    return model


def vgg_multi_image_top_net(input_shape, n_labels, weights_path=None):

    weights_file = None
    if weights_path is not None: weights_file = h5py.File(weights_path)

    model1 = Sequential()
    model1.add(Flatten(input_shape=input_shape))
    weight_21 = mutil.get_layer_weights(weights_file, layer_name='fc_img2_1')        
    model1.add(Dense(128, init='uniform', name='fc_img2_1', weights=weight_21,
                     W_regularizer=L1L2Regularizer(0.0,0.0)))
    model1.add(Activation('relu')) 
    model1.add(Dropout(0.4))        
    weight_22 = mutil.get_layer_weights(weights_file, layer_name='fc_img2_2')        
    model1.add(Dense(n_labels, init='uniform', name='fc_img2_2', weights=weight_22,
                     W_regularizer=L1L2Regularizer(0.0,0.0)))
    model1.add(Activation('relu')) 
    model1.add(Dropout(0.))        

    model2 = Sequential()
    model2.add(Flatten(input_shape=input_shape))
    weight_31 = mutil.get_layer_weights(weights_file, layer_name='fc_img3_1')        
    model2.add(Dense(128, init='uniform', name='fc_img3_1', weights=weight_31,
                     W_regularizer=L1L2Regularizer(0.0,0.0)))
    model2.add(Activation('relu')) 
    model2.add(Dropout(0.4))        
    weight_32 = mutil.get_layer_weights(weights_file, layer_name='fc_img3_2')        
    model2.add(Dense(n_labels, init='uniform', name='fc_img3_2', weights=weight_32,
                     W_regularizer=L1L2Regularizer(0.0,0.0)))
    model2.add(Activation('relu')) 
    model2.add(Dropout(0.))        

    merge = Merge([model1, model2], mode='concat')
    model = Sequential()
    model.add(merge)
    weight_norm = mutil.get_layer_weights(weights_file, layer_name='batchnormalization_1')
    model.add(BatchNormalization(name='batchnormalization_1', weights=weight_norm))
    weight_out = mutil.get_layer_weights(weights_file, layer_name='fc_img_out')        
    model.add(Dense(n_labels, activation='softmax', weights=weight_out,
                    name='fc_img_out'))

    ## if weights_path is not None: model.load_weights(weights_path)    
    return model


def vgg_multi_image_net(input_shape, n_labels, full_weights_path=None, top_weights_path=None,\
                        fine_tune=False):

    model1 = vgg16_net(input_shape, weights_path=full_weights_path, fine_tune=fine_tune)        
    model2 = vgg16_net2(input_shape, weights_path=full_weights_path, fine_tune=fine_tune)


    if full_weights_path is not None: weights_path = full_weights_path
    elif top_weights_path is not None: weights_path = top_weights_path
    weights_file = None
    if weights_path is not None: weights_file = h5py.File(weights_path)

    model1.add(Flatten())

    weight_21 = mutil.get_layer_weights(weights_file, layer_name='fc_img2_1')        
    model1.add(Dense(128, init='uniform', name='fc_img2_1', weights=weight_21,
                     W_regularizer=L1L2Regularizer(0.0,0.0)))
    model1.add(Activation('relu')) 
    model1.add(Dropout(0.4))        
    weight_22 = mutil.get_layer_weights(weights_file, layer_name='fc_img2_2')        
    model1.add(Dense(n_labels, init='uniform', name='fc_img2_2', weights=weight_22,
                     W_regularizer=L1L2Regularizer(0.0,0.0)))
    model1.add(Activation('relu')) 
    model1.add(Dropout(0.))        

    model2.add(Flatten())
    weight_31 = mutil.get_layer_weights(weights_file, layer_name='fc_img3_1')        
    model2.add(Dense(128, init='uniform', name='fc_img3_1', weights=weight_31,
                     W_regularizer=L1L2Regularizer(0.0,0.0)))
    model2.add(Activation('relu')) 
    model2.add(Dropout(0.4))        
    weight_32 = mutil.get_layer_weights(weights_file, layer_name='fc_img3_2')        
    model2.add(Dense(n_labels, init='uniform', name='fc_img3_2', weights=weight_32,
                     W_regularizer=L1L2Regularizer(0.0,0.0)))
    model2.add(Activation('relu')) 
    model2.add(Dropout(0.))        

    merge = Merge([model1, model2], mode='concat')
    model = Sequential()
    model.add(merge)
    weight_norm = mutil.get_layer_weights(weights_file, layer_name='batchnormalization_1')    
    model.add(BatchNormalization(name='batchnormalization_1', weights=weight_norm))
   
    weight_out = mutil.get_layer_weights(weights_file, layer_name='fc_img_out')        
    model.add(Dense(n_labels, activation='softmax', weights=weight_out,
                    name='fc_img_out'))

    return model



def vgg_multi_net(input_shape1, input_shape2, n_labels,
                  sig_weights_path=None, img_weights_path=None, top_weights_path=None,
                  full_weights_path=None, \
                  fine_tune=False, with_top=False):
    '''
    input_shape1: signal
    input_shape2: image
    '''

    # for image ------------------------------------------------------------------
    if full_weights_path is not None: weights_path = full_weights_path
    elif img_weights_path is not None: weights_path = img_weights_path
    model_img1 = vgg16_net(input_shape2, weights_path=weights_path, fine_tune=fine_tune)
    model_img2 = vgg16_net2(input_shape2, weights_path=weights_path, fine_tune=fine_tune)

        
    weights_file = None
    if weights_path is not None: weights_file = h5py.File(weights_path)

    model_img1.add(Flatten())

    weight_21 = mutil.get_layer_weights(weights_file, layer_name='fc_img2_1')        
    model_img1.add(Dense(128, init='uniform', name='fc_img2_1', weights=weight_21,
                     W_regularizer=L1L2Regularizer(0.0,0.0)))
    model_img1.add(Activation('relu')) 
    model_img1.add(Dropout(0.4))        
    weight_22 = mutil.get_layer_weights(weights_file, layer_name='fc_img2_2')        
    model_img1.add(Dense(n_labels, init='uniform', name='fc_img2_2', weights=weight_22,
                     W_regularizer=L1L2Regularizer(0.0,0.0)))
    model_img1.add(Activation('relu')) 
    model_img1.add(Dropout(0.))        

    model_img2.add(Flatten())
    weight_31 = mutil.get_layer_weights(weights_file, layer_name='fc_img3_1')        
    model_img2.add(Dense(128, init='uniform', name='fc_img3_1', weights=weight_31,
                     W_regularizer=L1L2Regularizer(0.0,0.0)))
    model_img2.add(Activation('relu')) 
    model_img2.add(Dropout(0.4))        
    weight_32 = mutil.get_layer_weights(weights_file, layer_name='fc_img3_2')        
    model_img2.add(Dense(n_labels, init='uniform', name='fc_img3_2', weights=weight_32,
                     W_regularizer=L1L2Regularizer(0.0,0.0)))
    model_img2.add(Activation('relu')) 
    model_img2.add(Dropout(0.))        

    merge = Merge([model_img1, model_img2], mode='concat')
    model_img = Sequential()
    model_img.add(merge)
    weight_norm = mutil.get_layer_weights(weights_file, layer_name='batchnormalization_1')    
    model_img.add(BatchNormalization(name='batchnormalization_1', weights=weight_norm))
    ## weight_out = mutil.get_layer_weights(weights_file, layer_name='fc_img_out')        
    ## model_img.add(Dense(n_labels, activation='softmax', weights=weight_out,
    ##                 name='fc_img_out'))

    # for signal --------------------------------------------------------------
    if full_weights_path is not None: weights_path = full_weights_path
    elif sig_weights_path is not None: weights_path = sig_weights_path
    model_sig = sig_net(input_shape1, n_labels, weights_path=weights_path, with_top=False)

    # all ---------------------------------------------------------------------
    # Do we need normalization??
    
    # add 1.0 to each model
    ## merge_img = Merge([model_img, ], mode='mul')
    ## model_sig.add(Lambda(add_const, output_shape=output_of_add_const))
    ## model_img.add(Lambda(add_const, output_shape=output_of_add_const))

    ## if fine_tune is not True:
    ##     for layer in model_sig.layers:
    ##         layer.trainable = False
    ##     for layer in model_img.layers:
    ##         layer.trainable = False


    merge = Merge([model_sig, model_img], mode='concat')
    model = Sequential()
    model.add(merge)

    if with_top:

        if full_weights_path is not None: weights_file = h5py.File(full_weights_path)
        elif top_weights_path is not None: weights_file = h5py.File(top_weights_path)      
        else: weights_file = None
        weight_out = mutil.get_layer_weights(weights_file, layer_name='fc_out')        
        model.add(Dense(n_labels, activation='softmax',W_regularizer=L1L2Regularizer(0,0),
                        weights=weight_out, name='fc_out'))
    
    ## model.add(Activation('softmax')) 
    return model



def multi_top_net(input_shape, n_labels, weights_path=None):

    model = Sequential()

    weights_file = None
    if weights_path is not None: weights_file = h5py.File(weights_path)
    weight = mutil.get_layer_weights(weights_file, layer_name='fc_multi_1')    
    model.add(Dense(16, activation='relu', init='uniform', name='fc_multi_1', input_shape=input_shape,
                    W_regularizer=L1L2Regularizer(0.0, 0.0), weights=weight))
    model.add(Dropout(0.0))

    ## weight = mutil.get_layer_weights(weights_file, layer_name='fc_multi_2')    
    ## model.add(Dense(16, activation='relu', init='uniform', name='fc_multi_2',
    ##                 W_regularizer=L1L2Regularizer(0.0, 0.0), weights=weight))
    ## model.add(Dropout(0.0))


    weight = mutil.get_layer_weights(weights_file, layer_name='fc_out')
    model.add(Dense(n_labels, activation='softmax', name='fc_out', input_shape=input_shape,
                    weights=weight, W_regularizer=L1L2Regularizer(0,0.0)))

    if weights_path is not None:  weights_file.close()
    return model



def output_of_add_const(input_shape):
    return input_shape

def add_const(x):
    return x+1.0








## def euc_dist(x):
##     'Merge function: euclidean_distance(u,v)'
##     s = x[0] - x[1]
##     output = K.sum( s**2, axis=1)
##     output = K.expand_dims(output,1)
##     return output

## def euc_dist_shape(input_shape):
##     'Merge output shape'
##     shape = list(input_shape)
##     outshape = (shape[0][0],1)
##     return tuple(outshape)

## def sqrt_diff(X):
##     s = X[0]
##     for i in range(1, len(X)):
##         s -= X[i]
##         s = K.sqrt(K.square(s) + 1e-7)
##     return s
