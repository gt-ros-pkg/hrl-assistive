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

from sklearn.metrics import accuracy_score

from hrl_execution_monitor.keras_util import keras_util as ku
## from hrl_anomaly_detection.vae import util as vutil
from . import vgg16

import gc

random.seed(3334)
np.random.seed(3334)
vgg_weights_path = os.path.expanduser('~')+'/git/keras_test/vgg16_weights.h5'
y_group = [[3,5,10],[2,7,8,9,11],[1,6],[0,4]]


def img_net(idx, trainData, testData, save_data_path, batch_size=512, nb_epoch=500, \
            patience=20, fine_tuning=False, img_scale=0.25, img_feature_type='vgg',\
            weights_file=None, save_weights_file='sig_weights.h5', renew=False,
            load_weights=False, cause_class=True, **kwargs):
    """
    Variational Autoencoder with two LSTMs and one fully-connected layer
    x_train is (sample x dim)
    x_test is (sample x dim)

    y_train should contain all labels
    """
    if img_feature_type == 'vgg': vgg=True
    else: vgg=False
    if vgg: prefix = 'vgg_'
    else: prefix = ''

    
    x_train = trainData[0]
    y_train = np.expand_dims(trainData[1], axis=-1)-2
    x_test = testData[0]
    y_test = np.expand_dims(testData[1], axis=-1)-2

    n_labels = len(np.unique(y_train))
    print "Labels: ", np.unique(y_train), " #Labels: ", n_labels
    print "Labels: ", np.unique(y_test), " #Labels: ", n_labels
    print np.shape(x_train), np.shape(y_train), np.shape(x_test), np.shape(y_test)

    x_train, y_train, x_test, y_test = get_bottleneck_image(idx, (x_train, y_train), (x_test, y_test),
                                                            save_data_path, n_labels, renew=renew)    

    # Convert labels to categorical one-hot encoding
    y_train = keras.utils.to_categorical(y_train, num_classes=n_labels)
    y_test  = keras.utils.to_categorical(y_test, num_classes=n_labels)

    # split train data into training and validation data.
    idx_list = range(len(x_train))
    np.random.shuffle(idx_list)
    x = np.array(x_train)[idx_list]
    y = y_train[idx_list]

    x_train = x[:int(len(x)*0.7)]
    y_train = y[:int(len(x)*0.7)]

    x_val = x[int(len(x)*0.7):]
    y_val = y[int(len(x)*0.7):]
    

    # Model construction (VGG16) ---------------------------------------------------------
    if weights_file is not None and os.path.isfile(weights_file) and renew is False:
        model.load_weights(weights_file)
    else:
        callbacks = [EarlyStopping(monitor='val_loss', min_delta=0.001, patience=patience,
                                   verbose=0, mode='auto'),
                     ModelCheckpoint(weights_file,
                                     save_best_only=True,
                                     save_weights_only=True,
                                     monitor='val_loss'),
                     ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                       patience=2, min_lr=0.00001)]

        if load_weights is False:

            model = vgg16.vgg_image_top_net(np.shape(x_train)[1:], n_labels)
            ## optimizer = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
            optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.001)
            ## model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        else:
            if vgg: model = km.vgg_image_top_net(np.shape(x_train)[1:], n_labels, weights_path)
            else: sys.exit()
            optimizer = SGD(lr=0.0001, decay=1e-7, momentum=0.9, nesterov=True)                
            #model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
            model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


        class_weight={}
        for i in xrange(n_labels):
            class_weight[i] = 1.0
        ## class_weight[1]  = 0.1 # noisy env
        ## class_weight[6]  = 0.1 # anomalous snd
        ## class_weight[-3] = 0.5 # spoon miss by sys
        ## class_weight[-2] = 0.5 # spoon collision by sys
        ## class_weight[-1] = 0.5 # freeze

        model.fit(x_train, y_train, nb_epoch=nb_epoch, batch_size=4096, shuffle=True,
                  validation_data=(x_val, y_val), callbacks=callbacks,
                  class_weight=class_weight)


    y_pred = model.predict(x_test)
    y_test_list = []
    y_pred_list = []
    if cause_class:
        y_pred_list += np.argmax(y_pred, axis=1).tolist()
        y_test_list += np.argmax(y_test, axis=1).tolist()
        score = accuracy_score(np.argmax(y_test, axis=1).tolist(),
                               np.argmax(y_pred, axis=1).tolist() ) 

    else:
        for y in np.argmax(y_pred, axis=1):
            if y in y_group[0]: y_pred_list.append(0)
            elif y in y_group[1]: y_pred_list.append(1)
            elif y in y_group[2]: y_pred_list.append(2)
            elif y in y_group[3]: y_pred_list.append(3)

        for y in np.argmax(y_test, axis=1):
            if y in y_group[0]: y_test_list.append(0)
            elif y in y_group[1]: y_test_list.append(1)
            elif y in y_group[2]: y_test_list.append(2)
            elif y in y_group[3]: y_test_list.append(3)
        score = accuracy_score(y_test_list, y_pred_list) 

    print "score : ", score

    return model, score



def get_bottleneck_image(idx, train_data, test_data, save_data_path, n_labels, use_extra_img=False,
                         renew=False):

    bt_data_path = os.path.join(save_data_path, 'bt')
    if os.path.isdir(bt_data_path) is False:
        os.system('mkdir -p '+bt_data_path)

    tr_x_fname = os.path.join(bt_data_path,'x_train_bt_'+str(idx)+'.npy')
    tr_y_fname = os.path.join(bt_data_path,'y_train_bt_'+str(idx)+'.npy')
    te_x_fname = os.path.join(bt_data_path,'x_test_bt_'+str(idx)+'.npy')
    te_y_fname = os.path.join(bt_data_path,'y_test_bt_'+str(idx)+'.npy')

    if renew is False and os.path.isfile(tr_x_fname):
        x_tr_ = np.load(open(tr_x_fname))
        y_tr_ = np.load(open(tr_y_fname))
        x_te_ = np.load(open(te_x_fname))
        y_te_ = np.load(open(te_y_fname))
        return x_tr_, y_tr_, x_te_, y_te_


    # Loading data
    x_train = np.array(train_data[0])
    y_train = np.array(train_data[1])
    x_test  = np.array(test_data[0])
    y_test  = np.array(test_data[1])
        
    ## from hrl_execution_monitor.keras_util import keras_model as km
    ## model = km.vgg16_net(np.shape(x_train)[1:], n_labels)
    model = vgg16.VGG16(include_top=False, input_shape=np.shape(x_train)[1:], classes=n_labels,
                        weights='img_net')
            
    # ------------------------------------------------------------
    from keras.preprocessing.image import ImageDataGenerator
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        rescale=1./255,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,
        fill_mode='nearest',
        data_format=K.image_data_format())

    x_tr_ = None
    y_tr_ = None
    count = 0
    for x_batch, y_batch in train_datagen.flow(x_train, y_train, batch_size=len(x_train),
                                               shuffle=False):
        if x_tr_ is None: x_tr_ = model.predict(x_batch)
        else: x_tr_ = np.vstack([x_tr_, model.predict(x_batch)])
        if y_tr_ is None: y_tr_ = y_train
        else: y_tr_ = np.vstack([y_tr_, y_train])
        count += 1
        print count
        if count > 4: break

    np.save(open(tr_x_fname, 'w'), x_tr_)
    np.save(open(tr_y_fname, 'w'), y_tr_)
    del train_datagen

    # ------------------------------------------------------------
    test_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=False,
        data_format=K.image_data_format())

    x_te_ = None
    y_te_ = None
    count = 0
    for x_batch, y_batch in test_datagen.flow(x_test, y_test, batch_size=len(x_test), shuffle=False):
        if x_te_ is None: x_te_ = model.predict(x_batch)
        else: x_te_ = np.vstack([x_te_, model.predict(x_batch)])
        if y_te_ is None: y_te_ = y_test
        else: y_te_ = np.vstack([y_te_, y_test])
        count += 1
        print count
        if count > 0: break

    np.save(open(te_x_fname, 'w'), x_te_)
    np.save(open(te_y_fname, 'w'), y_te_)
    del test_datagen
        
    return x_tr_, y_tr_, x_te_, y_te_




