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

# Private utils
from hrl_anomaly_detection import data_manager as dm
from hrl_anomaly_detection import util as util
import hrl_anomaly_detection.isolator.isolation_util as iutil
from hrl_anomaly_detection.localization import keras_model as km
## from hrl_execution_monitor.keras_util import keras_model as km
from hrl_execution_monitor.keras_util import keras_util as kutil
from hrl_execution_monitor.keras_util import train_isol_net as kt
from hrl_execution_monitor import util as autil
from hrl_execution_monitor import viz as eviz
from hrl_execution_monitor import preprocess as pp
from joblib import Parallel, delayed

from hrl_execution_monitor.keras_util import keras_util as ku
from keras.optimizers import SGD, Adagrad, Adadelta, RMSprop
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from sklearn import preprocessing
from sklearn.metrics import accuracy_score


random.seed(3334)
np.random.seed(3334)

import h5py, cv2, gc

    
def multi_level_test(save_data_path, n_labels=12, n_folds=8, verbose=False):

    fold_list = range(nFold)
    fold_list = [1]

    save_data_path = os.path.join(save_data_path, 'keras')

    # training with signals ----------------------------------
    ## train_with_signal(save_data_path, n_labels, fold_list, nb_epoch=800, patience=10)
    ## train_with_signal(save_data_path, n_labels, fold_list, nb_epoch=800, patience=2, load_weights=True)
    ## train_with_signal(save_data_path, n_labels, fold_list, load_weights=True,
    ##                   test_only=True, cause_class=True) #75


    # training the top model with images -----------------------------------
    # 50
    ## get_multi_bottleneck_images(save_data_path, n_labels, fold_list)
    ## train_top_model_with_image(save_data_path, n_labels, fold_list, nb_epoch=1000, patience=10,
    ##                           multi=True)
    ## train_top_model_with_image(save_data_path, n_labels, fold_list, nb_epoch=1, patience=100,
    ##                            multi=True, load_weights=True)
    ## train_top_model_with_image(save_data_path, n_labels, fold_list,\
    ##                            multi=True, load_weights=True, test_only=True)
     
    # training_with images -----------------------------------
    ## train_model_with_image(save_data_path, n_labels, fold_list, nb_epoch=1000, patience=10)
    ## train_model_with_image(save_data_path, n_labels, fold_list, nb_epoch=1000, patience=1,
    ##                        load_weights=True)
    ## train_model_with_image(save_data_path, n_labels, fold_list, load_weights=True, test_only=True)

    # training_with all --------------------------------------
    ## get_multi_bottleneck(save_data_path, n_labels, fold_list)
    ## train_multi_top_model(save_data_path, n_labels, fold_list, patience=10)

    
    ## train_with_all(save_data_path, n_labels, fold_list, patience=1, nb_epoch=100, fine_tune=False)
    ## ## train_with_all(save_data_path, n_labels, fold_list, load_weights=True, patience=5)
    ## train_with_all(save_data_path, n_labels, fold_list, load_weights=True,
    ##                test_only=True)



def train_with_signal(save_data_path, n_labels, fold_list, nb_epoch=400, load_weights=False,
                      activ_type='relu', test_only=False, save_pdf=False, patience=50,
                      cause_class=True):

    scores= []
    y_test_list = []
    y_pred_list = []
    for idx in fold_list:

        # Loading data
        train_data, test_data = autil.load_data(idx, save_data_path, viz=False)      
        x_train_sig = train_data[0]
        y_train     = train_data[2]
        x_test_sig = test_data[0]
        y_test     = test_data[2]

        # Scaling
        scaler      = preprocessing.StandardScaler()
        x_train_sig = scaler.fit_transform(x_train_sig)
        x_test_sig  = scaler.transform(x_test_sig)

        weights_path = os.path.join(save_data_path,'sig_weights_'+str(idx)+'.h5')
        callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=patience,
                                   verbose=0, mode='auto'),
                     ModelCheckpoint(weights_path,
                                     save_best_only=True,
                                     save_weights_only=True,
                                     monitor='val_loss'),
                     ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                       patience=20, min_lr=0.00001)]
        
        ## # Load pre-trained vgg16 model
        if load_weights is False:
            model = km.sig_net(np.shape(x_train_sig)[1:], n_labels, activ_type=activ_type)                    
            #optimizer = RMSprop(lr=0.01, rho=0.9, epsilon=1e-08, decay=0.0)
            optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.005)            
            ## optimizer = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
            ## optimizer = Adagrad(lr=0.001, epsilon=1e-08, decay=0.001)
        else:
            model = km.sig_net(np.shape(x_train_sig)[1:], n_labels,\
                               weights_path = weights_path, activ_type=activ_type)
            optimizer = SGD(lr=0.0001, decay=1e-7, momentum=0.9, nesterov=True)
            optimizer = RMSprop(lr=0.0005, rho=0.9, epsilon=1e-08, decay=0.005)            

        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        ## model.compile(optimizer=optimizer, loss='categorical_crossentropy', \
        ##               metrics=['mean_squared_logarithmic_error', 'accuracy'])

        import collections
        bincnt = collections.Counter(np.argmax(y_train,axis=1))

        class_weight = {}
        for i in xrange(n_labels):
            ## class_weight[i] = float(len(y_train))/(float(n_labels)*float(bincnt[i]) )
            class_weight[i] = 1.0
        

        if test_only is False:
            train_datagen = ku.sigGenerator(augmentation=True, noise_mag=0.0 )
            test_datagen = ku.sigGenerator(augmentation=False)
            train_generator = train_datagen.flow(x_train_sig, y_train, batch_size=1024)
            test_generator = test_datagen.flow(x_test_sig, y_test, batch_size=1024)

            hist = model.fit_generator(train_generator,
                                       samples_per_epoch=len(y_train),
                                       nb_epoch=nb_epoch,
                                       validation_data=test_generator,
                                       nb_val_samples=len(y_test),
                                       callbacks=callbacks, class_weight=class_weight)

            scores.append( hist.history['val_acc'][-1] )
            del model
        else:
            model.load_weights(weights_path)
            y_pred = model.predict(x_test_sig)

            if cause_class:
                y_pred_list += np.argmax(y_pred, axis=1).tolist()
                y_test_list += np.argmax(y_test, axis=1).tolist()
                scores.append( accuracy_score(np.argmax(y_test, axis=1).tolist(),
                                              np.argmax(y_pred, axis=1).tolist() ) )
            else:
                y_test_list = []
                y_pred_list = []
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
                scores.append( accuracy_score(y_test_list, y_pred_list) )
                            
            print "score : ", scores

    print 
    print np.mean(scores), np.std(scores)
    if test_only: return y_test_list, y_pred_list
    return


def train_top_model_with_image(save_data_path, n_labels, fold_list, nb_epoch=400, load_weights=False,
                               vgg=True,\
                               patience=5, remove_label=[], test_only=False,\
                               cause_class=True, multi=False):
    prefix = 'vgg_'

    scores= []
    y_pred_list = []
    y_test_list = []
    for idx in fold_list:

        bt_data_path = os.path.join(save_data_path, 'bt')
        if test_only is False:
            x_train = np.load(open(os.path.join(bt_data_path,'x_train_bt_'+str(idx)+'.npy')))
            y_train = np.load(open(os.path.join(bt_data_path,'y_train_bt_'+str(idx)+'.npy')))
        x_test = np.load(open(os.path.join(bt_data_path,'x_test_bt_'+str(idx)+'.npy')))
        y_test = np.load(open(os.path.join(bt_data_path,'y_test_bt_'+str(idx)+'.npy')))

        weights_path = os.path.join(save_data_path,prefix+'cnn_weights_'+str(idx)+'.h5')
        if os.path.isfile(weights_path) is False:
            print weights_path
            sys.exit()
        
        callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=patience,
                                   verbose=0, mode='auto'),
                     ModelCheckpoint(weights_path,
                                     save_best_only=True,
                                     save_weights_only=True,
                                     monitor='val_loss'),
                     ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                       patience=5, min_lr=0.00001)]

        if load_weights is False:
            if multi: model = km.vgg_multi_image_top_net(np.shape(x_test[1])[1:], n_labels)
            else:     model = km.vgg_image_top_net(np.shape(x_test)[1:], n_labels)
            model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        else:
            print "Load weight!!!!!!!!!!!!!!!"
            if multi: model = km.vgg_multi_image_top_net(np.shape(x_test[1])[1:], n_labels, weights_path)
            else:     model = km.vgg_image_top_net(np.shape(x_test)[1:], n_labels, weights_path)
            optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.001)
            optimizer = SGD(lr=0.001, decay=1e-7, momentum=0.9, nesterov=True)                
            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        
        class_weight={}
        for i in xrange(n_labels): class_weight[i] = 1.0
            
        if test_only is False:
            if multi:
                hist = model.fit([x_train[1], x_train[2]], y_train,
                                 nb_epoch=nb_epoch, batch_size=1024, shuffle=True,
                                 validation_data=([x_test[1], x_test[2]], y_test),
                                 callbacks=callbacks,
                                 class_weight=class_weight)
            else:
                hist = model.fit(x_train, y_train, nb_epoch=nb_epoch, batch_size=4096, shuffle=True,
                                 validation_data=(x_test, y_test), callbacks=callbacks,
                                 class_weight=class_weight)

            scores.append( hist.history['val_acc'][-1] )
        else:
            if multi:
                y_pred = model.predict([x_test[1], x_test[2]])
            else:
                y_pred = model.predict(x_test)
                
            if cause_class:
                # cause classification
                y_pred_list += np.argmax(y_pred, axis=1).tolist()
                y_test_list += np.argmax(y_test, axis=1).tolist()
                scores.append( accuracy_score(np.argmax(y_test, axis=1).tolist(),
                                              np.argmax(y_pred, axis=1).tolist() ) )
            else:
                # type classification
                y_test_list = []
                y_pred_list = []
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
                scores.append( accuracy_score(y_test_list, y_pred_list) )

            print "score : ", scores        
        gc.collect()

    print 
    print np.mean(scores), np.std(scores)
    return


def train_model_with_image(save_data_path, n_labels, fold_list, nb_epoch=400, load_weights=False,
                           vgg=True, patience=5, use_extra_img=True, test_only=False,\
                           cause_class=True):
    if vgg: prefix = 'vgg_'
    else: prefix = ''

    scores= []
    y_pred_list = []
    y_test_list = []
    for idx in fold_list:

        # Loading data
        train_data, test_data = autil.load_data(idx, save_data_path, extra_img=use_extra_img, viz=False)      
        x_train_img = train_data[1] # sample x 3 x 224 x 224
        y_train     = train_data[2]
        x_test_img = test_data[1]
        y_test     = test_data[2]

        full_weights_path = os.path.join(save_data_path,prefix+'multi_img_weights_'+str(idx)+'.h5')
        callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=patience,
                                   verbose=0, mode='auto'),
                     ModelCheckpoint(full_weights_path,
                                     save_best_only=True,
                                     save_weights_only=True,
                                     monitor='val_loss'),
                     ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                       patience=5, min_lr=0.00001)]

        if load_weights is False:
            top_weights_path = os.path.join(save_data_path,prefix+'cnn_weights_'+str(idx)+'.h5')
            model = km.vgg_multi_image_net(np.shape(x_train_img)[1:], n_labels,
                                           top_weights_path=top_weights_path,
                                           fine_tune=True)
            ## model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        else:
            print "Load weight!!!!!!!!!!!!!!!"
            model = km.vgg_multi_image_net(np.shape(x_train_img)[1:], n_labels,
                                           full_weights_path=full_weights_path,
                                           fine_tune=True)
            ## optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.001)
        optimizer = SGD(lr=0.001, decay=1e-7, momentum=0.9, nesterov=True)                
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        
        class_weight={}
        for i in xrange(n_labels): class_weight[i] = 1.0

        # ------------------------------------------------------------
        
        if test_only is False:
            train_datagen = ku.multiImgGenerator(augmentation=True, rescale=1./255.)
            test_datagen = ku.multiImgGenerator(augmentation=False, rescale=1./255.)
            train_generator = train_datagen.flow(x_train_img, y_train, batch_size=32) 
            test_generator = test_datagen.flow(x_test_img, y_test, batch_size=32)

            hist = model.fit_generator(train_generator,
                                       samples_per_epoch=len(y_train),
                                       nb_epoch=nb_epoch,
                                       validation_data=test_generator,
                                       nb_val_samples=len(y_test),
                                       callbacks=callbacks,
                                       class_weight=class_weight)

            scores.append( hist.history['val_acc'][-1] )

        else:
            ## x_test_img = x_train_img
            ## y_test = y_train
            
            test_datagen = ku.multiImgGenerator(augmentation=False, rescale=1./255.)
            for x_batch, y_batch in test_datagen.flow(x_test_img, y_test, batch_size=len(x_test_img),
                                                      shuffle=False):
                x = x_batch
                y = y_batch
                break
            y_pred = model.predict([x[0], x[1]])
                
            if cause_class:
                # cause classification
                scores.append( accuracy_score(np.argmax(y_test, axis=1).tolist(),
                                              np.argmax(y_pred, axis=1).tolist() ) )
            else:
                # type classification
                y_test_list = []
                y_pred_list = []
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
                scores.append( accuracy_score(y_test_list, y_pred_list) )

            print "score : ", scores        
        gc.collect()

    print 
    print np.mean(scores), np.std(scores)
    return


def train_with_all(save_data_path, n_labels, fold_list, nb_epoch=100, load_weights=False,
                   test_only=False, save_pdf=False, vgg=True, patience=3, fine_tune=False):

    if vgg: prefix = 'vgg_'
    else: prefix = ''

    scores= []
    for idx in fold_list:
        # Loading data
        train_data, test_data = autil.load_data(idx, save_data_path, viz=False)      
        x_train_sig = train_data[0]
        x_train_img = train_data[1]
        y_train     = train_data[2]
        x_test_sig = test_data[0]
        x_test_img = test_data[1]
        y_test     = test_data[2]

        scaler      = preprocessing.StandardScaler()
        x_train_sig = scaler.fit_transform(x_train_sig)
        x_test_sig  = scaler.transform(x_test_sig)

        full_weights_path = os.path.join(save_data_path,prefix+'all_weights_'+str(idx)+'.h5')
        callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=patience,
                                   verbose=0, mode='auto'),
                     ModelCheckpoint(full_weights_path,
                                     save_best_only=True,
                                     save_weights_only=True,
                                     monitor='val_loss'),
                     ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                       patience=1, min_lr=0.00001)]

        if load_weights is False:

            sig_weights_path = os.path.join(save_data_path,'sig_weights_'+str(idx)+'.h5')
            img_weights_path = os.path.join(save_data_path,prefix+'multi_img_weights_'+str(idx)+'.h5')
            top_weights_path = os.path.join(save_data_path,prefix+'multi_top_weights_'+str(idx)+'.h5')
            
            # training
            model = km.vgg_multi_net(np.shape(x_train_sig)[1:], np.shape(x_train_img)[1:], n_labels,
                                     sig_weights_path=sig_weights_path,
                                     img_weights_path=img_weights_path,
                                     top_weights_path=top_weights_path,
                                     fine_tune=fine_tune)
        else:
            # fine tuning
            model = km.vgg_multi_net(np.shape(x_train_sig)[1:], np.shape(x_train_img)[1:], n_labels,
                                     full_weights_path=full_weights_path,
                                     fine_tune=fine_tune)
            
        optimizer = SGD(lr=0.005, decay=1e-7, momentum=0.9, nesterov=True)                
        ## optimizer = RMSprop(lr=0.1, rho=0.9, epsilon=1e-08, decay=0.00001)                        
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        ## model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        import collections
        bincnt = collections.Counter(np.argmax(y_train,axis=1))

        class_weight={}
        for i in xrange(n_labels): class_weight[i] = 1.0
        
        if test_only is False:
            train_datagen = ku.multiGenerator(augmentation=True, rescale=1./255.)
            test_datagen = ku.multiGenerator(augmentation=False, rescale=1./255.)
            train_generator = train_datagen.flow(x_train_sig, x_train_img, y_train, batch_size=32) #128)
            test_generator = test_datagen.flow(x_test_sig, x_test_img, y_test, batch_size=32) #128)
        
            hist = model.fit_generator(train_generator,
                                       samples_per_epoch=len(y_train),
                                       nb_epoch=nb_epoch,
                                       validation_data=test_generator,
                                       nb_val_samples=len(y_test),
                                       callbacks=callbacks,
                                       class_weight=class_weight)

            scores.append( hist.history['val_acc'][-1] )
        else:

            test_datagen = ku.multiGenerator(augmentation=False, rescale=1./255.)
            for x_batch, y_batch in test_datagen.flow(x_test_sig, x_test_img, y_test,
                                                      batch_size=len(x_test_img),
                                                      shuffle=False):
                x = x_batch
                y = y_batch
                break
            
            y_pred = model.predict([x[0], x[1], x[2]])
            y_pred_list = np.argmax(y_pred, axis=1).tolist()
            y_test_list = np.argmax(y_test, axis=1).tolist()
            
            from sklearn.metrics import accuracy_score
            print "score : ", accuracy_score(y_test_list, y_pred_list)
            scores.append( accuracy_score(y_test_list, y_pred_list) )
        gc.collect()


    print np.mean(scores), np.std(scores)
    if test_only: return y_test_list, y_pred_list
    return


def train_multi_top_model(save_data_path, n_labels, fold_list, nb_epoch=3000, load_weights=False, vgg=False,
                          patience=30, test_only=False):

    if vgg: prefix = 'vgg_'
    else: prefix = ''

    scores= []
    for idx in fold_list:

        bt_data_path = os.path.join(save_data_path, 'bt')
        x_train = np.load(open(os.path.join(bt_data_path,'x_train_btmt_'+str(idx)+'.npy')))
        y_train = np.load(open(os.path.join(bt_data_path,'y_train_btmt_'+str(idx)+'.npy')))
        x_test  = np.load(open(os.path.join(bt_data_path,'x_test_btmt_'+str(idx)+'.npy')))
        y_test  = np.load(open(os.path.join(bt_data_path,'y_test_btmt_'+str(idx)+'.npy')))
        print np.shape(x_train), np.shape(y_train), np.shape(x_test), np.shape(y_test)

        #----------------------------------------------------------------------------------                
        weights_path = os.path.join(save_data_path,prefix+'multi_top_weights_'+str(idx)+'.h5')
        callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=patience,
                                   verbose=0, mode='auto'),
                     ModelCheckpoint(weights_path,
                                     save_best_only=True,
                                     save_weights_only=True,
                                     monitor='val_loss'),
                     ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                       patience=5, min_lr=0.00001)]

        if load_weights is False:            
            model = km.multi_top_net(np.shape(x_train)[1:], n_labels)
            model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        else:
            model = km.multi_top_net(np.shape(x_train)[1:], n_labels, weights_path)
            optimizer = SGD(lr=0.005, decay=1e-7, momentum=0.9, nesterov=True)                
            #optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.001)
            ## optimizer = Adagrad(lr=0.0001, epsilon=1e-08, decay=0.001)
            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
            ## model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        ## from sklearn.svm import SVC
        ## clf = SVC(C=1.0, kernel='rbf', gamma=1e-5) #, decision_function_shape='ovo')
        ## from sklearn.ensemble import RandomForestClassifier
        ## clf = RandomForestClassifier(n_estimators=400, n_jobs=-1)
        ## clf.fit(x_train, np.argmax(y_train, axis=1))
        ## score = clf.score(x_test, np.argmax(y_test,axis=1))
        ## scores.append(score)   

        if test_only is False:
            hist = model.fit(x_train, y_train, nb_epoch=nb_epoch, batch_size=len(x_train), shuffle=True,
                             validation_data=(x_test, y_test), callbacks=callbacks)       
            scores.append( hist.history['val_acc'][-1] )
        else:
            y_pred = model.predict(x_test)
            y_pred_list = np.argmax(y_pred, axis=1).tolist()
            y_test_list = np.argmax(y_test, axis=1).tolist()
            from sklearn.metrics import accuracy_score
            print "score : ", accuracy_score(y_test_list, y_pred_list)
            scores.append( accuracy_score(y_test_list, y_pred_list) )
        
        gc.collect()

    print 
    print np.mean(scores), np.std(scores)
    return


    


# ------------------------------------------------------------------------------------------------
def get_multi_bottleneck_images(save_data_path, n_labels, fold_list, vgg=True, use_extra_img=True,
                                remove_label=[]):
    ''' Get 3-level bottleneck features from an image. '''

    if vgg: prefix = 'vgg_'
    else: prefix = ''

    scores= []
    for idx in fold_list:

        # Loading data
        train_data, test_data = autil.load_data(idx, save_data_path, extra_img=use_extra_img, viz=False)      
        x_train_img = train_data[1] # sample x 3 x 224 x 224
        y_train     = train_data[2]
        x_test_img = test_data[1]
        y_test     = test_data[2]

        ## for img in x_train_img:
        ##     img = img.transpose((1,2,0))
        ##     img[:,:,0] += 103.939
        ##     img[:,:,1] += 116.779
        ##     img[:,:,2] += 123.68

        ##     rows,cols = np.shape(img)[:2]
        ##     M   = cv2.getRotationMatrix2D((cols/2,rows/2),180,1)
        ##     img = cv2.warpAffine(img,M,(cols,rows))

        ##     gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        ##     faceDetectClassifier = cv2.CascadeClassifier("/home/dpark/util/opencv-2.4.11/data/haarcascades/haarcascade_frontalface_default.xml")

        ##     faces = faceDetectClassifier.detectMultiScale(gray, 1.3, 5)
        ##     for (x,y,w,h) in faces:
        ##         cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                
        ##     cv2.imshow('image',img.astype(np.uint8))
        ##     cv2.waitKey(0)
        ##     cv2.destroyAllWindows()
        ## sys.exit()
            
        # remove specific label --------------------
        add_idx = []
        for i, y in enumerate(y_train):
            if np.argmax(y) not in remove_label:
                add_idx.append(i)
        x_train_img = np.array(x_train_img)[add_idx]
        y_train = np.array(y_train)[add_idx]

        add_idx = []
        for i, y in enumerate(y_test):
            if np.argmax(y) not in remove_label:
                add_idx.append(i)
        x_test_img = np.array(x_test_img)[add_idx]
        y_test = np.array(y_test)[add_idx]
        #--------------------------------------------
        
        if vgg: model = km.vgg16_net(np.shape(x_train_img)[1:], n_labels)
        else: model = km.cnn_net(np.shape(x_train_img)[1:], n_labels)            

        bt_data_path = os.path.join(save_data_path, 'bt')
        if os.path.isdir(bt_data_path) is False:
            os.system('mkdir -p '+bt_data_path)
            
        # ------------------------------------------------------------
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            rescale=1./255,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=False,
            fill_mode='nearest',
            dim_ordering="th")

        x_ = [None, None]
        y_ = None
        width  = 70
        height = 70
        img_size = np.shape(x_train_img)[-2:]

        count = 0        
        for x_batch, y_batch in train_datagen.flow(x_train_img, y_train, batch_size=len(x_train_img),
                                                   shuffle=False):

            ## crop data using three levels
            # 1. face only
            ## x_batch_1 = None
            ## for img in x_batch:
            ##     img   = img.transpose((1,2,0))
            ##     img_1 = cv2.resize(img[(img_size[0]-width)/2:(img_size[0]+width)/2,
            ##                            (img_size[1]-height)/2:(img_size[1]+height)/2],
            ##                            np.shape(img)[:2])
            ##     img_1 = img_1.transpose((2,0,1))
            ##     if x_batch_1 is None:
            ##         x_batch_1 = np.expand_dims(img_1, axis=0)
            ##     else:
            ##         x_batch_1 = np.vstack([x_batch_1, np.expand_dims(img_1, axis=0)])

            ## if x_[0] is None: x_[0] = model.predict(x_batch_1)
            ## else: x_[0] = np.vstack([x_[0], model.predict(x_batch_1)])

            # 2. face and hand
            x_batch_2 = None
            for img in x_batch:
                img   = img.transpose((1,2,0))
                img_2 = cv2.resize(img[img_size[0]/4-width/4:((img_size[0]+width)/2+img_size[0])/2,
                                       img_size[1]/4-height/4:((img_size[1]+height/2)+img_size[1])/2],
                                       np.shape(img)[:2])
                img_2 = img_2.transpose((2,0,1))
                if x_batch_2 is None:
                    x_batch_2 = np.expand_dims(img_2, axis=0)
                else:
                    x_batch_2 = np.vstack([x_batch_2, np.expand_dims(img_2, axis=0)])
            if x_[1] is None: x_[1] = model.predict(x_batch_2)
            else: x_[1] = np.vstack([x_[1], model.predict(x_batch_2)])

            # 3. face, hand, and background
            if x_[2] is None: x_[2] = model.predict(x_batch)
            else: x_[2] = np.vstack([x_[2], model.predict(x_batch)])

            if y_ is None: y_ = y_train
            else: y_ = np.vstack([y_, y_train])
            count += 1
            print count
            if count > 4: break

        ## x_ = np.array(x_)
        ## x_ = np.swapaxes(x_, 0, 1)
        ## print np.shape(x_)
            
        np.save(open(os.path.join(bt_data_path,'x_train_bt_'+str(idx)+'.npy'), 'w'), x_)
        np.save(open(os.path.join(bt_data_path,'y_train_bt_'+str(idx)+'.npy'), 'w'), y_)
        del x_, y_, train_datagen, x_batch_2, x_batch

        # ------------------------------------------------------------
        test_datagen = ImageDataGenerator(
            rescale=1./255,
            horizontal_flip=False,
            dim_ordering="th")

        x_ = [None, None, None]
        y_ = None
        count = 0
        for x_batch, y_batch in test_datagen.flow(x_test_img, y_test, batch_size=len(x_test_img), shuffle=False):

            # 1. face only
            ## x_batch_1 = None
            ## for img in x_batch:
            ##     img   = img.transpose((1,2,0))
            ##     img_1 = cv2.resize(img[(img_size[0]-width)/2:(img_size[0]+width)/2,
            ##                            (img_size[1]-height)/2:(img_size[1]+height)/2],
            ##                            np.shape(img)[:2])
            ##     img_1 = img_1.transpose((2,0,1))
            ##     if x_batch_1 is None:
            ##         x_batch_1 = np.expand_dims(img_1, axis=0)
            ##     else:
            ##         x_batch_1 = np.vstack([x_batch_1, np.expand_dims(img_1, axis=0)])
            ## if x_[0] is None: x_[0] = model.predict(x_batch_1)
            ## else: x_[0] = np.vstack([x_[0], model.predict(x_batch_1)])

            # 2. face and hand
            x_batch_2 = None
            for img in x_batch:
                img   = img.transpose((1,2,0))
                img_2 = cv2.resize(img[img_size[0]/4-width/4:((img_size[0]+width)/2+img_size[0])/2,
                                       img_size[1]/4-height/4:((img_size[1]+height/2)+img_size[1])/2],
                                       np.shape(img)[:2])
                img_2 = img_2.transpose((2,0,1))
                if x_batch_2 is None:
                    x_batch_2 = np.expand_dims(img_2, axis=0)
                else:
                    x_batch_2 = np.vstack([x_batch_2, np.expand_dims(img_2, axis=0)])
            if x_[1] is None: x_[1] = model.predict(x_batch_2)
            else: x_[1] = np.vstack([x_[1], model.predict(x_batch_2)])

            # 3. face, hand, and background
            if x_[2] is None: x_[2] = model.predict(x_batch)
            else: x_[2] = np.vstack([x_[2], model.predict(x_batch)])
                
            if y_ is None: y_ = y_test
            else: y_ = np.vstack([y_, y_test])
            count += 1
            print count
            if count > 0: break

        ## x_ = np.array(x_)
        ## x_ = np.swapaxes(x_, 0, 1)

        np.save(open(os.path.join(bt_data_path,'x_test_bt_'+str(idx)+'.npy'), 'w'), x_)
        np.save(open(os.path.join(bt_data_path,'y_test_bt_'+str(idx)+'.npy'), 'w'), y_)
        del x_, y_, test_datagen, x_batch_2, x_batch
        
        gc.collect()

    return



def get_multi_bottleneck(save_data_path, n_labels, fold_list, vgg=True):

    if vgg: prefix = 'vgg_'
    else: prefix = ''

    scores      = []
    y_test_list = []
    y_pred_list = []
    for idx in fold_list:
        # Loading data
        train_data, test_data = autil.load_data(idx, save_data_path, viz=False)      
        x_train_sig = train_data[0]
        x_train_img = train_data[1]
        y_train     = train_data[2]
        x_test_sig = test_data[0]
        x_test_img = test_data[1]
        y_test     = test_data[2]

        scaler      = preprocessing.StandardScaler()
        x_train_sig = scaler.fit_transform(x_train_sig)
        x_test_sig  = scaler.transform(x_test_sig)

        sig_weights_path=os.path.join(save_data_path,'sig_weights_'+str(idx)+'.h5')
        img_weights_path=os.path.join(save_data_path,prefix+'multi_img_weights_'+str(idx)+'.h5')

        # training
        model = km.vgg_multi_net(np.shape(x_train_sig)[1:], np.shape(x_train_img)[1:], n_labels,
                                 sig_weights_path=sig_weights_path,
                                 img_weights_path=img_weights_path)

        train_datagen = ku.multiGenerator(augmentation=True, rescale=1./255.)
        test_datagen = ku.multiGenerator(augmentation=False, rescale=1./255.)

        bt_data_path = os.path.join(save_data_path, 'bt')
        if os.path.isdir(bt_data_path) is False:
            os.system('mkdir -p '+bt_data_path)

        # ------------------------------------------------------------
        x_ = None
        y_ = None
        count = 0
        for x_batch, y_batch in train_datagen.flow(x_train_sig, x_train_img, y_train,
                                                   batch_size=len(x_train_img)):
            if x_ is None: x_ = model.predict([x_batch[0], x_batch[1], x_batch[2]])
            else: x_ = np.vstack([x_, model.predict([x_batch[0], x_batch[1], x_batch[2]])])
            if y_ is None: y_ = y_batch
            else: y_ = np.vstack([y_, y_batch])
            count += 1
            print count
            if count > 4: break
        np.save(open(os.path.join(bt_data_path,'x_train_btmt_'+str(idx)+'.npy'), 'w'), x_)
        np.save(open(os.path.join(bt_data_path,'y_train_btmt_'+str(idx)+'.npy'), 'w'), y_)
        del x_, y_, train_datagen

        x_ = None
        y_ = None
        count = 0
        for x_batch, y_batch in test_datagen.flow(x_test_sig, x_test_img, y_test,
                                                  batch_size=len(x_test_img)):
            if x_ is None: x_ = model.predict([x_batch[0], x_batch[1], x_batch[2]])
            else: x_ = np.vstack([x_, model.predict([x_batch[0], x_batch[1], x_batch[2]])])
            if y_ is None: y_ = y_batch
            else: y_ = np.vstack([y_, y_batch])
            count += 1
            print count
            if count > 0: break
        np.save(open(os.path.join(bt_data_path,'x_test_btmt_'+str(idx)+'.npy'), 'w'), x_)
        np.save(open(os.path.join(bt_data_path,'y_test_btmt_'+str(idx)+'.npy'), 'w'), y_)
        del x_, y_, test_datagen
        
        gc.collect()

    return
    




if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    util.initialiseOptParser(p)

    p.add_option('--preprocess', '--p', action='store_true', dest='preprocessing',
                 default=False, help='Preprocess')
    p.add_option('--preprocess_extra', '--pe', action='store_true', dest='preprocessing_extra',
                 default=False, help='Preprocess extra images')
    p.add_option('--train', '--tr', action='store_true', dest='train',
                 default=False, help='Train')
    p.add_option('--test', '--te', action='store_true', dest='test',
                 default=False, help='Test')
    p.add_option('--viz', action='store_true', dest='viz',
                 default=False, help='Visualize')
    p.add_option('--viz_model', '--vm', action='store_true', dest='viz_model',
                 default=False, help='Visualize the current model')
    p.add_option('--save_viz_feature', '--svf', action='store_true', dest='save_viz_feature',
                 default=False, help='Save features for visualization')
    
    opt, args = p.parse_args()

    #---------------------------------------------------------------------------           
    # Run evaluation
    #---------------------------------------------------------------------------           
    nPoints = 40 #None

    from hrl_anomaly_detection.localization.ICRA2018_params import *
    raw_data_path, save_data_path, param_dict = getParams(opt.task, opt.bDataRenew, \
                                                          opt.bHMMRenew, opt.bCLFRenew, opt.dim,\
                                                          nPoints=nPoints)
    if opt.bNoUpdate: param_dict['ROC']['update_list'] = []
    # s1 - kaci - before camera calibration
    subjects = ['s2', 's3','s4','s5', 's6','s7','s8', 's9']
    n_labels = 12 #len(np.unique(y_train))
    nFold    = 8

    # IROS2017 - backup
    save_data_path = os.path.expanduser('~')+\
      '/hrl_file_server/dpark_data/anomaly/ICRA2018/'+opt.task+'_data_localization/'

    ## cause_class = False
    
    # ---------------------------------------------------------------------
    if opt.preprocessing:
        src_pkl = os.path.join(save_data_path, 'isol_data.pkl')
        pp.preprocess_data(src_pkl, save_data_path, img_scale=0.25, nb_classes=12,
                            img_feature_type='cascade', nFold=nFold)

    elif opt.preprocessing_extra:
        raw_data_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/AURO2016/raw_data/manual_label'
        pp.preprocess_images(raw_data_path, save_data_path, img_scale=0.25, nb_classes=n_labels,
                                img_feature_type='cascade')

    ## elif opt.train:
    ##     train_isolator_modules(save_data_path, n_labels, verbose=False, cause_class=cause_class)

    elif opt.viz:
        x_train, y_train, x_test, y_test = autil.load_data(save_data_path, True)
        
    elif opt.viz_model:
        model = km.vgg16_net((3,120,160), 12, with_top=True, input_shape2=(14,), viz=True)
        plot(model, to_file='model.png')
        
    else:
        multi_level_test(save_data_path)
        #test(save_data_path)
