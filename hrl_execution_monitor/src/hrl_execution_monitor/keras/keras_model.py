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

random.seed(3334)
np.random.seed(3334)

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


def cnn_net(input_shape, n_labels, weights_path=None, with_top=False, input_shape2=None,
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

    # the model so far outputs 3D feature maps (height, width, features)
    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64, name='fc1_1'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    ## model.add(Dense(1))
    ## model.add(Activation('sigmoid'))

    if with_top or fine_tune:
        for layer in model.layers:
            layer.trainable = False


    if with_top:
        sig_model = Sequential()
        sig_model.add(Dense(128, activation='relu', init='uniform', input_shape=input_shape2,\
                            name='fc2_1'))
        sig_model.add(Dropout(0.5))

        merge = Merge([model, sig_model], mode='concat')
        t_model = Sequential()
        t_model.add(merge)        
        t_model.add(Dense(64, activation='relu', init='uniform', name='fc3_1'))
        t_model.add(Dropout(0.5))      
        t_model.add(Dense(n_labels, activation='softmax', name='fc_out'))

        if weights_path is not None:
            t_model.load_weights(weights_path, by_name=True)
        return t_model
    
    else:
        model.add(Dense(n_labels, activation='softmax', name='fc_img_out'))

        if weights_path is not None:
            model.load_weights(weights_path, by_name=True)

        return model


def sig_net(input_shape, n_labels, weights_path=None, fine_tune=False, activ_type='relu'):

    if activ_type == 'PReLU':
        activ_type = PReLU(init='zero', weights=None)


    
    model = Sequential()
    model.add(Dense(128, init='uniform', input_shape=input_shape,\
                    name='fc2_1'))
    ## model.add(BatchNormalization())
    model.add(Activation(activ_type))
    model.add(Dropout(0.5))
    ## model.add(Dense(128, init='uniform', name='fc2_2'))
    ## ## model.add(BatchNormalization())
    ## model.add(Activation(activ_type))
    ## model.add(Dropout(0.5))

    if fine_tune:
        for layer in model.layers:
            layer.trainable = False
    
    model.add(Dense(n_labels, activation='softmax', name='fc_sig_out'))

    if weights_path is not None:
        model.load_weights(weights_path, by_name=True)

    return model


def vgg16_net(input_shape, n_labels, imagenet_weights_path=None, weights_path=None, \
              with_top=False, input_shape2=None, fine_tune=False, viz=False):
    
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

    if fine_tune:
        for layer in model.layers[:25]:
            layer.trainable = False
    else:
        for layer in model.layers:
            layer.trainable = False

    if imagenet_weights_path is not None:
        
        # load the weights of the VGG16 networks
        # (trained on ImageNet, won the ILSVRC competition in 2014)
        # note: when there is a complete match between your model definition
        # and your weight savefile, you can simply call model.load_weights(filename)
        assert os.path.exists(imagenet_weights_path), \
          'Model weights not found (see "weights_path" variable in script).'
        f = h5py.File(imagenet_weights_path)
        for k in range(f.attrs['nb_layers']):
            if k >= len(model.layers):
                # we don't look at the last (fully-connected) layers in the savefile
                break
            g = f['layer_{}'.format(k)]
            weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
            model.layers[k].set_weights(weights)
        f.close()
        print('Model loaded.')

    model.add(Flatten())
    model.add(Dense(256, init='uniform', name='fc1_1'))
    ## model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    if with_top:
        sig_model = Sequential()
        sig_model.add(Dense(128, activation='relu', init='uniform', input_shape=input_shape2,\
                            name='fc2_1'))
        sig_model.add(Dropout(0.5))

        merge = Merge([model, sig_model], mode='concat')
        t_model = Sequential()
        t_model.add(merge)        
        t_model.add(Dense(64, activation='relu', init='uniform', name='fc3_1'))
        t_model.add(Dropout(0.5))      
        t_model.add(Dense(n_labels, activation='softmax', name='fc_out'))

        if weights_path is not None:
            t_model.load_weights(weights_path, by_name=True)
        return t_model
    
    else:
        model.add(Dense(n_labels, activation='softmax', name='fc_img_out'))

        if weights_path is not None:
            model.load_weights(weights_path, by_name=True)

        return model


