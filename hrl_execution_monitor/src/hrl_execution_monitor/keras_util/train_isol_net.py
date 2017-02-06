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

#  \author Daehyung Park (Healthcare Robotics Lab, Georgia Tech.)

# system & utils
import os, sys, copy, random
import scipy, numpy as np
import hrl_lib.util as ut

# Private utils
from hrl_anomaly_detection import util as util
from hrl_execution_monitor import util as autil

#keras
from hrl_execution_monitor.keras_util import keras_model as km
from hrl_execution_monitor.keras_util import keras_util as ku
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import SGD, Adagrad, Adadelta, RMSprop
from keras.preprocessing.image import ImageDataGenerator
import h5py 

from sklearn import preprocessing

from joblib import Parallel, delayed
import gc

random.seed(3334)
np.random.seed(3334)



def train_isolator_modules(save_data_path, n_labels, verbose=False):
    '''
    Train networks
    '''
    d = ut.load_pickle(os.path.join(save_data_path, 'isol_data.pkl'))
    nFold = len(d.keys())
    del d
    fold_list = range(nFold)

    save_data_path = os.path.join(save_data_path, 'keras')

    # training with signals ----------------------------------
    ## train_with_signal(save_data_path, n_labels, fold_list, nb_epoch=800, patience=50)
    ## train_with_signal(save_data_path, n_labels, fold_list, nb_epoch=800, patience=50, load_weights=True)

    # training_with images -----------------------------------
    remove_label = [1]
    #get_bottleneck_image(save_data_path, n_labels, fold_list, vgg=True, remove_label=remove_label)
    #train_top_model_with_image(save_data_path, n_labels, fold_list, vgg=True)
    ## train_top_model_with_image(save_data_path, n_labels, fold_list, vgg=True, nb_epoch=1000, load_weights=True)
    
    ## train_with_image(save_data_path, n_labels, fold_list, patience=20)
    ## train_with_image(save_data_path, n_labels, fold_list, patience=20, fine_tune=True)

    ## train_with_image(save_data_path, n_labels, fold_list, patience=20, vgg=True, remove_label=remove_label)
    ## train_with_image(save_data_path, n_labels, fold_list, patience=20, vgg=True, remove_label=remove_label,
    ##                  load_weights=True)

    # training_with all --------------------------------------
    #get_bottleneck_mutil(save_data_path, n_labels, fold_list, vgg=True)
    ## train_multi_top_model(save_data_path, n_labels, fold_list, vgg=True)
    ## train_multi_top_model(save_data_path, n_labels, fold_list, vgg=True, load_weights=True) # noneed
    
    ## train_with_all(save_data_path, n_labels, fold_list, patience=1, nb_epoch=1, vgg=True)
    train_with_all(save_data_path, n_labels, fold_list, load_weights=True, patience=3, vgg=True) # almost no need


    return



def train_with_signal(save_data_path, n_labels, fold_list, nb_epoch=400, load_weights=False,
                      activ_type='relu',
                      test_only=False, save_pdf=False, patience=50):

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
        from sklearn import preprocessing
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
            ## optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
            optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.005)            
        else:
            model = km.sig_net(np.shape(x_train_sig)[1:], n_labels,\
                               weights_path = weights_path, activ_type=activ_type)
            ## optimizer = SGD(lr=0.001, decay=1e-7, momentum=0.9, nesterov=True)
            optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.005)            

        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        ## model.compile(optimizer=optimizer, loss='categorical_crossentropy', \
        ##               metrics=['mean_squared_logarithmic_error', 'accuracy'])


        if test_only is False:
            train_datagen = ku.sigGenerator(augmentation=True, noise_mag=0.05 )
            test_datagen = ku.sigGenerator(augmentation=False)
            train_generator = train_datagen.flow(x_train_sig, y_train, batch_size=512)
            test_generator = test_datagen.flow(x_test_sig, y_test, batch_size=512)

            hist = model.fit_generator(train_generator,
                                       samples_per_epoch=len(y_train),
                                       nb_epoch=nb_epoch,
                                       validation_data=test_generator,
                                       nb_val_samples=len(y_test),
                                       callbacks=callbacks)

            scores.append( hist.history['val_acc'][-1] )
            ## model.save_weights(weights_path)
            del model
        else:
            model.load_weights(weights_path)
            y_pred = model.predict(x_test_sig)
            y_pred_list += np.argmax(y_pred, axis=1).tolist()
            y_test_list += np.argmax(y_test, axis=1).tolist()
            
            from sklearn.metrics import accuracy_score
            print "score : ", accuracy_score(y_test_list, y_pred_list)
            scores.append( accuracy_score(y_test_list, y_pred_list) )

    print 
    print np.mean(scores), np.std(scores)
    if test_only: return y_test_list, y_pred_list
    return


def train_with_image(save_data_path, n_labels, fold_list, nb_epoch=1, load_weights=False, vgg=False,
                     patience=20, remove_label=[], use_extra_img=True):

    if vgg: prefix = 'vgg_'
    else: prefix = ''

    scores= []
    for idx in fold_list:

        # Loading data
        train_data, test_data = autil.load_data(idx, save_data_path, extra_img=use_extra_img, viz=False)      
        x_train_img = train_data[1]
        y_train     = train_data[2]
        x_test_img = test_data[1]
        y_test     = test_data[2]

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
                
        weights_path = os.path.join(save_data_path,prefix+'cnn_weights_'+str(idx)+'.h5')
        callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=patience,
                                   verbose=0, mode='auto'),
                     ModelCheckpoint(weights_path,
                                     save_best_only=True,
                                     save_weights_only=True,
                                     monitor='val_loss'),
                     ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                       patience=5, min_lr=0.00001)]

        if load_weights is False:            
            if vgg: model = km.vgg16_net(np.shape(x_train_img)[1:], n_labels, with_img_top=True)
            else: model = km.cnn_net(np.shape(x_train_img)[1:], n_labels)            
        else:
            if vgg: model = km.vgg16_net(np.shape(x_train_img)[1:], n_labels, weights_path, with_img_top=True)
            else: model = km.cnn_net(np.shape(x_train_img)[1:], n_labels, weights_path)
            ## optimizer = RMSprop(lr=0.00001, rho=0.9, epsilon=1e-08, decay=0.0)
            ## model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        optimizer = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        ## optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.005)                        
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        ## model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        train_datagen = ImageDataGenerator(
            rotation_range=20,
            rescale=1./255,
            width_shift_range=0.4,
            height_shift_range=0.2,
            zoom_range=0.1,
            horizontal_flip=False,
            fill_mode='nearest',
            dim_ordering="th")
        test_datagen = ImageDataGenerator(rescale=1./255,\
                                          dim_ordering="th")

        train_generator = train_datagen.flow(x_train_img, y_train, batch_size=32) #128)
        test_generator = test_datagen.flow(x_test_img, y_test, batch_size=32) #128)

        hist = model.fit_generator(train_generator,
                                   samples_per_epoch=len(y_train),
                                   nb_epoch=nb_epoch,
                                   validation_data=test_generator,
                                   nb_val_samples=len(y_test),
                                   callbacks=callbacks)

        scores.append( hist.history['val_acc'][-1] )
        gc.collect()

    print 
    print np.mean(scores), np.std(scores)
    return


def train_with_all(save_data_path, n_labels, fold_list, nb_epoch=100, load_weights=False,
                   test_only=False, save_pdf=False, vgg=False, patience=3):

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


        ## import cv2
        ## print np.shape(x_train_img)
        ## img = x_test_img[0].transpose((1,2,0))
        ## img[:,:,0] += 103.939
        ## img[:,:,1] += 116.779
        ## img[:,:,2] += 123.68
        ## print np.shape(img), np.amax(img), np.amin(img)
        
        ## cv2.imshow('image',img.astype(np.uint8))
        ## cv2.waitKey(0)
        ## cv2.destroyAllWindows()
        ## sys.exit()

        scaler      = preprocessing.StandardScaler()
        x_train_sig = scaler.fit_transform(x_train_sig)
        x_test_sig  = scaler.transform(x_test_sig)

        weights_path = os.path.join(save_data_path,prefix+'all_weights_'+str(idx)+'.h5')
        callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=patience,
                                   verbose=0, mode='auto'),
                     ModelCheckpoint(weights_path,
                                     save_best_only=True,
                                     save_weights_only=True,
                                     monitor='val_loss'),
                     ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                       patience=3, min_lr=0.00001)]

        if load_weights is False:

            top_weights_path = os.path.join(save_data_path,prefix+'cnn_fc_weights_'+str(idx)+'.h5')
            sig_weights_path=os.path.join(save_data_path,'sig_weights_'+str(idx)+'.h5')
            img_weights_path=os.path.join(save_data_path,prefix+'cnn_weights_'+str(idx)+'.h5')
            
            # training
            if vgg:
                model = km.vgg16_net(np.shape(x_train_img)[1:], n_labels, with_multi_top=True,
                                     input_shape2=np.shape(x_train_sig)[1:],
                                     sig_weights_path=sig_weights_path,
                                     img_weights_path=img_weights_path,
                                     weights_path=top_weights_path,
                                     fine_tune=True)
            else:
                model = km.cnn_net(np.shape(x_train_img)[1:], n_labels, with_multi_top=True,
                                   input_shape2=np.shape(x_train_sig)[1:],
                                   sig_weights_path=sig_weights_path,
                                   img_weights_path=img_weights_path,
                                   weights_path=top_weights_path,
                                   fine_tune=True)
            optimizer = SGD(lr=0.0001, decay=1e-7, momentum=0.9, nesterov=True)                
        else:

            # fine tuning
            if vgg:
                model = km.vgg16_net(np.shape(x_train_img)[1:], n_labels, with_multi_top=True,
                                     input_shape2=np.shape(x_train_sig)[1:],
                                     weights_path=weights_path,
                                     fine_tune=True)
            else:
                model = km.cnn_net(np.shape(x_train_img)[1:], n_labels, with_multi_top=True,
                                   input_shape2=np.shape(x_train_sig)[1:],
                                   weights_path=weights_path,
                                   fine_tune=True)
            optimizer = SGD(lr=0.0001, decay=1e-8, momentum=0.9, nesterov=True)                

        ## optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.005)                        
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


        class_weight={}
        for i in xrange(n_labels):
            class_weight[i] = 1.0
        ## class_weight[1]  = 0.1 # noisy env
        ## class_weight[6]  = 0.1 # anomalous snd
        ## class_weight[-3] = 0.1 # spoon miss by sys
        ## class_weight[-2] = 0.1 # spoon collision by sys
        ## class_weight[-1] = 0.1 # freeze

        

        if test_only is False:
            train_datagen = ku.myGenerator(augmentation=True, rescale=1./255.)
            test_datagen = ku.myGenerator(augmentation=False, rescale=1./255.)
            train_generator = train_datagen.flow(x_train_img, x_train_sig, y_train, batch_size=128)
            test_generator = test_datagen.flow(x_test_img, x_test_sig, y_test, batch_size=128)
        
            hist = model.fit_generator(train_generator,
                                       samples_per_epoch=len(y_train),
                                       nb_epoch=nb_epoch,
                                       validation_data=test_generator,
                                       nb_val_samples=len(y_test),
                                       callbacks=callbacks,
                                       class_weight=class_weight)

            scores.append( hist.history['val_acc'][-1] )
        else:
            ## #temp
            ## x_test_img = x_train_img
            ## x_test_sig = x_train_sig
            ## y_test = y_train
            y_test_list = []
            y_pred_list = []
            
            y_pred = model.predict([x_test_img/255., x_test_sig])
            y_pred_list += np.argmax(y_pred, axis=1).tolist()
            y_test_list += np.argmax(y_test, axis=1).tolist()
            
            from sklearn.metrics import accuracy_score
            print "score : ", accuracy_score(y_test_list, y_pred_list)
            scores.append( accuracy_score(y_test_list, y_pred_list) )
        gc.collect()


    print np.mean(scores), np.std(scores)
    if test_only: return y_test_list, y_pred_list
    return



def get_bottleneck_image(save_data_path, n_labels, fold_list, vgg=False, use_extra_img=True,
                         remove_label=[]):

    if vgg: prefix = 'vgg_'
    else: prefix = ''

    scores= []
    for idx in fold_list:

        # Loading data
        train_data, test_data = autil.load_data(idx, save_data_path, extra_img=use_extra_img, viz=False)      
        x_train_img = train_data[1]
        y_train     = train_data[2]
        x_test_img = test_data[1]
        y_test     = test_data[2]

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

        x_ = None
        y_ = None
        count = 0
        for x_batch, y_batch in train_datagen.flow(x_train_img, y_train, batch_size=len(x_train_img),
                                                   shuffle=False):
            if x_ is None: x_ = model.predict(x_batch)
            else: x_ = np.vstack([x_, model.predict(x_batch)])
            if y_ is None: y_ = y_train
            else: y_ = np.vstack([y_, y_train])
            count += 1
            print count
            if count > 4: break

        np.save(open(os.path.join(bt_data_path,'x_train_bt_'+str(idx)+'.npy'), 'w'), x_)
        np.save(open(os.path.join(bt_data_path,'y_train_bt_'+str(idx)+'.npy'), 'w'), y_)
        del x_, y_, train_datagen

        # ------------------------------------------------------------
        test_datagen = ImageDataGenerator(
            rescale=1./255,
            horizontal_flip=False,
            dim_ordering="th")

        x_ = None
        y_ = None
        count = 0
        for x_batch, y_batch in test_datagen.flow(x_test_img, y_test, batch_size=len(x_test_img), shuffle=False):
            if x_ is None: x_ = model.predict(x_batch)
            else: x_ = np.vstack([x_, model.predict(x_batch)])
            if y_ is None: y_ = y_test
            else: y_ = np.vstack([y_, y_test])
            count += 1
            print count
            if count > 0: break

        np.save(open(os.path.join(bt_data_path,'x_test_bt_'+str(idx)+'.npy'), 'w'), x_)
        np.save(open(os.path.join(bt_data_path,'y_test_bt_'+str(idx)+'.npy'), 'w'), y_)
        del x_, y_, test_datagen
        
        gc.collect()

    return


def train_top_model_with_image(save_data_path, n_labels, fold_list, nb_epoch=400, load_weights=False, vgg=False,
                               patience=5, remove_label=[], use_extra_img=True, test_only=False):

    if vgg: prefix = 'vgg_'
    else: prefix = ''

    y_pred_list = []
    y_test_list = []
    scores= []
    for idx in fold_list:

        bt_data_path = os.path.join(save_data_path, 'bt')

        x_train = np.load(open(os.path.join(bt_data_path,'x_train_bt_'+str(idx)+'.npy')))
        y_train = np.load(open(os.path.join(bt_data_path,'y_train_bt_'+str(idx)+'.npy')))
        x_test = np.load(open(os.path.join(bt_data_path,'x_test_bt_'+str(idx)+'.npy')))
        y_test = np.load(open(os.path.join(bt_data_path,'y_test_bt_'+str(idx)+'.npy')))

        print np.shape(x_train), np.shape(y_train), np.shape(x_test), np.shape(y_test)

        #--------------------------------------------
                
        weights_path = os.path.join(save_data_path,prefix+'cnn_weights_'+str(idx)+'.h5')
        callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=patience,
                                   verbose=0, mode='auto'),
                     ModelCheckpoint(weights_path,
                                     save_best_only=True,
                                     save_weights_only=True,
                                     monitor='val_loss'),
                     ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                       patience=5, min_lr=0.00001)]

        if load_weights is False:            
            if vgg: model = km.vgg_image_top_net(np.shape(x_train)[1:], n_labels)
            else: sys.exit()
            ## optimizer = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
            optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.001)                        
        else:
            if vgg: model = km.vgg_image_top_net(np.shape(x_train)[1:], n_labels, weights_path)
            else: sys.exit()
            optimizer = SGD(lr=0.0001, decay=1e-7, momentum=0.9, nesterov=True)
                
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        ## model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        
        class_weight={}
        for i in xrange(n_labels):
            class_weight[i] = 1.0
        class_weight[1]  = 0.1 # noisy env
        class_weight[6]  = 0.1 # anomalous snd
        class_weight[-3] = 0.5 # spoon miss by sys
        class_weight[-2] = 0.5 # spoon collision by sys
        class_weight[-1] = 0.5 # freeze

        if test_only is False:
            hist = model.fit(x_train, y_train, nb_epoch=nb_epoch, batch_size=4096, shuffle=True,
                             validation_data=(x_test, y_test), callbacks=callbacks,
                             class_weight=class_weight)

            scores.append( hist.history['val_acc'][-1] )
        else:
            ## # train
            ## x_test = x_train
            ## y_test = y_train
            
            y_pred = model.predict(x_test)
            y_pred_list += np.argmax(y_pred, axis=1).tolist()
            y_test_list += np.argmax(y_test, axis=1).tolist()

            from sklearn.metrics import accuracy_score
            print "score : ", accuracy_score(y_test_list, y_pred_list)
            scores.append( accuracy_score(y_test_list, y_pred_list) )
        
        gc.collect()

    print 
    print np.mean(scores), np.std(scores)
    return


def get_bottleneck_mutil(save_data_path, n_labels, fold_list, vgg=False):

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
        img_weights_path=os.path.join(save_data_path,prefix+'cnn_weights_'+str(idx)+'.h5')
            
        # training
        if vgg:
            model = km.vgg16_net(np.shape(x_train_img)[1:], n_labels, with_multi_top=True,
                                 bottle_model=True,
                                 input_shape2=np.shape(x_train_sig)[1:],
                                 sig_weights_path=sig_weights_path,
                                 img_weights_path=img_weights_path)
        else:
            model = km.cnn_net(np.shape(x_train_img)[1:], n_labels, with_multi_top=True,
                               bottle_model=True,
                               input_shape2=np.shape(x_train_sig)[1:],
                               sig_weights_path=sig_weights_path,
                               img_weights_path=img_weights_path)

        train_datagen = ku.myGenerator(augmentation=True, rescale=1./255.)
        test_datagen = ku.myGenerator(augmentation=False, rescale=1./255.)

        bt_data_path = os.path.join(save_data_path, 'bt')
        if os.path.isdir(bt_data_path) is False:
            os.system('mkdir -p '+bt_data_path)

        # ------------------------------------------------------------
        x_ = None
        y_ = None
        count = 0
        for x_batch, y_batch in train_datagen.flow(x_train_img, x_train_sig, y_train,
                                                   batch_size=len(x_train_img)):
            if x_ is None: x_ = model.predict(x_batch)
            else: x_ = np.vstack([x_, model.predict(x_batch)])
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
        for x_batch, y_batch in test_datagen.flow(x_test_img, x_test_sig, y_test,
                                                  batch_size=len(x_test_img)):
            if x_ is None: x_ = model.predict(x_batch)
            else: x_ = np.vstack([x_, model.predict(x_batch)])
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
        weights_path = os.path.join(save_data_path,prefix+'cnn_fc_weights_'+str(idx)+'.h5')
        callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=patience,
                                   verbose=0, mode='auto'),
                     ModelCheckpoint(weights_path,
                                     save_best_only=True,
                                     save_weights_only=True,
                                     monitor='val_loss'),
                     ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                       patience=10, min_lr=0.00001)]

        if load_weights is False:            
            if vgg: model = km.vgg_multi_top_net(np.shape(x_train)[1:], n_labels)
            else: sys.exit()
            ## optimizer = SGD(lr=0.001, decay=1e-8, momentum=0.9, nesterov=True)                
            #optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.001)                        
            model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
            #model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        else:
            if vgg: model = km.vgg_multi_top_net(np.shape(x_train)[1:], n_labels, weights_path)
            else: sys.exit()
            optimizer = SGD(lr=0.0005, decay=1e-7, momentum=0.9, nesterov=True)                
            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

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
            ## #temp
            ## x_test = x_train
            ## y_test = y_train
            y_pred_list = []
            y_test_list = []
            
            y_pred = model.predict(x_test)
            y_pred_list += np.argmax(y_pred, axis=1).tolist()
            y_test_list += np.argmax(y_test, axis=1).tolist()
            from sklearn.metrics import accuracy_score
            print "score : ", accuracy_score(y_test_list, y_pred_list)
            scores.append( accuracy_score(y_test_list, y_pred_list) )
        
        gc.collect()

    print 
    print np.mean(scores), np.std(scores)
    return



def test(save_data_path):

    d = ut.load_pickle(os.path.join(save_data_path, 'isol_data.pkl'))
    nFold = len(d.keys())
    del d

    save_data_path = os.path.join(save_data_path, 'keras')

    train_data, test_data = autil.load_data(0, save_data_path, viz=False)
    print np.shape(train_data[0]),np.shape(train_data[1]),np.shape(train_data[2])
    print np.shape(test_data[0]), np.shape(test_data[1]), np.shape(test_data[2])

    datagen = ImageDataGenerator(
        rotation_range=20,
        rescale=1.0,#1./255,
        width_shift_range=0.4,
        height_shift_range=0.2,
        zoom_range=0.1,
        horizontal_flip=False,
        fill_mode='nearest',
        dim_ordering="th")

    ## datagen = ImageDataGenerator(dim_ordering="th")
    save_data_path = os.path.join(save_data_path, 'temp')

    import cv2
    count = 0
    for x,y in datagen.flow(test_data[1], test_data[2], batch_size=1, shuffle=False):
        x[0][0] += 103.939
        x[0][1] += 116.779
        x[0][2] += 123.68
        img = x[0].transpose((1,2,0)).astype(int)
        f = os.path.join(save_data_path, str(count)+'_'+str( np.argmax(y[0])+2)+'.jpg' )
        print count, np.shape(x[0])
        cv2.imwrite(f, img)
        count +=1
        if count > 10: break
    

def test_hog(save_data_path, n_labels, verbose=False):
    '''
    Train networks
    '''
    ## d = ut.load_pickle(os.path.join(save_data_path, 'isol_data.pkl'))
    ## nFold = len(d.keys())
    ## del d

    save_data_path = os.path.join(save_data_path, 'keras')

    idx = 0
    train_data, test_data = autil.load_data(idx, save_data_path, viz=False)      
    x_train_sig = train_data[0]
    x_train_img = train_data[1]
    y_train     = train_data[2]
    x_test_sig = test_data[0]
    x_test_img = test_data[1]
    y_test     = test_data[2]

    print np.shape(x_train_img), np.shape(x_train_sig), np.shape(y_train)
    x_train = np.hstack([x_train_sig, x_train_img])
    x_test  = np.hstack([x_test_sig, x_test_img])

    scaler      = preprocessing.StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test  = scaler.transform(x_test)
    

    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=400, n_jobs=-1)
    clf.fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    print score
    


if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    util.initialiseOptParser(p)

    p.add_option('--train', '--tr', action='store_true', dest='train',
                 default=False, help='Train')
    p.add_option('--hog', action='store_true', dest='hog',
                 default=False, help='Test with hog')
    
    opt, args = p.parse_args()

    from hrl_execution_monitor.params.IROS2017_params import *
    # IROS2017
    subject_names = ['s2', 's3','s4','s5', 's6','s7','s8', 's9']
    raw_data_path, save_data_path, param_dict = getParams(opt.task, opt.bDataRenew, \
                                                          opt.bHMMRenew, opt.bCLFRenew)
                                                          
    save_data_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/IROS2017/'+opt.task+'_demo1'
    save_data_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/IROS2017/'+opt.task+'_demo2'

    # best one for (-5,-11)
    save_data_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/IROS2017/'+opt.task+'_demo'

    task_name = 'feeding'
    method    = ['progress0', 'progress1'] 
    param_dict['ROC']['methods'] = ['progress0', 'progress1'] #'hmmgp'
    single_detector=False    
    nb_classes = 12


    if opt.train:
        train_isolator_modules(save_data_path, nb_classes, verbose=False)
    elif opt.hog:
        test_hog(save_data_path, nb_classes, verbose=False)
    else:
        test(save_data_path)


    ## get_isolator_modules(save_data_path, task_name, method, param_dict, fold_idx=0,\
    ##                       verbose=False)
