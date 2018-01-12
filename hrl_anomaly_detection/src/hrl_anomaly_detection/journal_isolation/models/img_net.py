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
import numpy
import numpy as np
import scipy

# Keras
import h5py 
from keras.models import Sequential, Model
from keras.layers import Input, TimeDistributed, Layer
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, Merge, Input
from keras.layers import Activation, Dropout, Flatten, Dense, merge, Lambda, GaussianNoise
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD, Adagrad, Adadelta, RMSprop
from keras import backend as K
from keras import objectives
from keras.callbacks import *
import keras

from hrl_anomaly_detection.journal_isolation.models import keras_util as ku
## from hrl_anomaly_detection.vae import util as vutil

import gc

random.seed(3334)
np.random.seed(3334)
vgg_weights_path = os.path.expanduser('~')+'/git/keras_test/vgg16_weights.h5'


def img_net(trainData, testData, batch_size=512, nb_epoch=500, \
            patience=20, fine_tuning=False, noise_mag=0.0,\
            weights_file=None, save_weights_file='sig_weights.h5', renew=False, **kwargs):
    """
    Variational Autoencoder with two LSTMs and one fully-connected layer
    x_train is (sample x dim)
    x_test is (sample x dim)

    y_train should contain all labels
    """
    x_train = trainData[0]
    y_train = np.expand_dims(trainData[1], axis=-1)-2
    x_test = testData[0]
    y_test = np.expand_dims(testData[1], axis=-1)-2

    n_labels = len(np.unique(y_train))
    print "Labels: ", np.unique(y_train), " #Labels: ", n_labels
    print "Labels: ", np.unique(y_test), " #Labels: ", n_labels



    # Extract images -------------------------------------------------------------------
    from hrl_execution_monitor import preprocess as pp
    rm_idx = []
    x = []
    for j, f in enumerate(x_train):
        if f is None:
            print "None image ", j+1, '/', len(x_train)
            rm_idx.append(j)
            continue

        img = extract_image(f, img_feature_type=img_feature_type, img_scale=img_scale)
        x.append(img)

    y_train  = np.expand_dims(trainData[1], axis=-1)
    n_labels = len(np.unique(y_train))
    get_bottleneck_image(save_data_path, n_labels, idx)




    
    print np.shape(x_train), np.shape(y_train), np.shape(x_test), np.shape(y_test)

    # Convert labels to categorical one-hot encoding
    y_train = keras.utils.to_categorical(y_train, num_classes=n_labels)
    y_test  = keras.utils.to_categorical(y_test, num_classes=n_labels)

    from sklearn import preprocessing
    scaler  = preprocessing.StandardScaler()
    #scaler  = preprocessing.MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test  = scaler.transform(x_test)

    # Model construction (VGG16) ---------------------------------------------------------
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

    if vgg_weights_path is not None:       
        # load the weights of the VGG16 networks
        # (trained on ImageNet, won the ILSVRC competition in 2014)
        # note: when there is a complete match between your model definition
        # and your weight savefile, you can simply call model.load_weights(filename)        
        assert os.path.exists(vgg_weights_path), \
          'Model weights not found (see "weights_path" variable in script).'
        f = h5py.File(vgg_weights_path)
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



    
    print(model.summary())

    if weights_file is not None and os.path.isfile(weights_file) and fine_tuning is False and\
      renew is False:
        model.load_weights(weights_file)
    else:

        callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=patience,
                                   verbose=0, mode='auto'),
                     ModelCheckpoint(weights_file,
                                     save_best_only=True,
                                     save_weights_only=True,
                                     monitor='val_loss'),
                     ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                       patience=patience-1, min_lr=0.00001)]

        if fine_tuning:
            model.load_weights(weights_file)
            optimizer = SGD(lr=0.0001, decay=1e-7, momentum=0.9, nesterov=True)
            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        else:
            print "----------------------"
            print weights_file
            print "----------------------"
            if os.path.isfile(weights_file) and False: model.load_weights(weights_file)
            model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
            #model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])


        train_datagen   = ku.sigGenerator(augmentation=True, noise_mag=noise_mag )
        test_datagen    = ku.sigGenerator(augmentation=False, noise_mag=0.0)
        train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)
        test_generator  = test_datagen.flow(x_test, y_test, batch_size=batch_size)

        hist = model.fit_generator(train_generator,
                                   samples_per_epoch=len(y_train),
                                   nb_epoch=nb_epoch,
                                   validation_data=test_generator,
                                   nb_val_samples=len(y_test),
                                   callbacks=callbacks)

        ## hist = model.fit(x_train, y_train, nb_epoch=nb_epoch, batch_size=batch_size, shuffle=True,
        ##                  validation_data=(x_test, y_test), callbacks=callbacks)

        scores = []
        scores.append( hist.history['val_acc'][-1] )
        gc.collect()


        print "score : ", scores
        print 
        print np.mean(scores), np.std(scores)

    return model


## class ResetStatesCallback(Callback):
##     def __init__(self, max_len):
##         self.counter = 0
##         self.max_len = max_len
        
##     def on_batch_begin(self, batch, logs={}):
##         if self.counter % self.max_len == 0:
##             self.model.reset_states()
##         self.counter += 1

