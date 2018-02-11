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
from keras.layers import Activation, Dropout, Flatten, Dense, merge, Lambda, GaussianNoise
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD, Adagrad, Adadelta, RMSprop
from keras import regularizers
from keras import backend as K
from keras import objectives
from keras.callbacks import *
import keras

from hrl_anomaly_detection.journal_isolation.models import keras_util as ku
## from hrl_anomaly_detection.vae import util as vutil

import gc



def multi_net(idx, trainData, testData, batch_size=512, sam_epoch=500, \
              patience=20, fine_tuning=False, noise_mag=0.0,\
              weights_file=None, renew=False, **kwargs):
    """
    Variational Autoencoder with two LSTMs and one fully-connected layer
    x_train is (sample x dim)
    x_test is (sample x dim)

    y_train should contain all labels
    """
    x_sig_train = trainData[0]
    x_img_train = trainData[1]
    y_train = np.expand_dims(trainData[2], axis=-1)-2
    x_sig_test = testData[0]
    x_img_test = testData[1]
    y_test = np.expand_dims(testData[2], axis=-1)-2

    n_labels = len(np.unique(y_train))
    print "Labels: ", np.unique(y_train), " #Labels: ", n_labels
    print "Labels: ", np.unique(y_test), " #Labels: ", n_labels
    ## print np.shape(x_train), np.shape(y_train), np.shape(x_test), np.shape(y_test)


    # Load weights
    if weights_file is not None:
        sig_weights_file = weights_file[0]
        img_weights_file = weights_file[1]
        multi_weights_file = weights_file[2]
    else:
        sig_weights_file = None
        img_weights_file = None
        multi_weights_file = None


    # split train data into training and validation data.
    idx_list = range(len(x_sig_train))
    np.random.shuffle(idx_list)
    x_sig = x_sig_train[idx_list]
    x_img = x_img_train[idx_list]
    y = y_train[idx_list]

    x_sig_train = x_sig[:int(len(x)*0.7)]
    x_img_train = x_img[:int(len(x)*0.7)]
    y_train = y[:int(len(x)*0.7)]

    x_sig_val = x_sig[int(len(x)*0.7):]
    x_img_val = x_img[int(len(x)*0.7):]
    y_val = y[int(len(x)*0.7):]

    # Convert labels to categorical one-hot encoding
    y_train = keras.utils.to_categorical(y_train, num_classes=n_labels)
    y_val   = keras.utils.to_categorical(y_val, num_classes=n_labels)
    ## y_test  = keras.utils.to_categorical(y_test)#, num_classes=n_labels)

    if multi_weights_file is not None and os.path.isfile(multi_weights_file) and fine_tuning is False and \
      renew is False:
        model.load_weights(multi_weights_file)
    else:
        callbacks = [EarlyStopping(monitor='val_loss', min_delta=0.0005, patience=patience,
                                   verbose=0, mode='auto'),
                     ModelCheckpoint(multi_weights_file,
                                     save_best_only=True,
                                     save_weights_only=True,
                                     monitor='val_loss'),
                     ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                       patience=3, min_lr=0.00001)]


        from . import vgg16
        model = vgg16.VGG16(include_top=False, include_multi_top=True,
                            input_shape=np.shape(x_train)[1:], classes=n_labels)
        ## model = km.vgg_multi_top_net(np.shape(x_train)[1:], n_labels, weights_path)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        print(model.summary())


        train_datagen   = ku.multiGenerator(augmentation=True, noise_mag=noise_mag )
        test_datagen    = ku.multiGenerator(augmentation=False, noise_mag=0.0)
        train_generator = train_datagen.flow(x_sig_train, x_img_train, y_train, batch_size=batch_size)
        test_generator  = test_datagen.flow(x_sig_val, x_img_val, y_val, batch_size=batch_size)

        hist = model.fit_generator(train_generator,
                                   samples_per_epoch=len(y_train),
                                   nb_epoch=nb_epoch,
                                   validation_data=test_generator,
                                   nb_val_samples=len(y_val),
                                   callbacks=callbacks)

    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1).tolist()

    print np.shape(y_pred), np.shape(y_test)

    from sklearn.metrics import accuracy_score
    score = accuracy_score(y_test, y_pred) 
    ## scores.append( hist.history['val_acc'][-1] )
    ## gc.collect()


    print "score : ", score
    ## print 
    ## print np.mean(scores), np.std(scores)

    return model, score


## class ResetStatesCallback(Callback):
##     def __init__(self, max_len):
##         self.counter = 0
##         self.max_len = max_len
        
##     def on_batch_begin(self, batch, logs={}):
##         if self.counter % self.max_len == 0:
##             self.model.reset_states()
##         self.counter += 1

