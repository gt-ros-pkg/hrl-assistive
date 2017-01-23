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
## from hrl_anomaly_detection import util as util
## from hrl_anomaly_detection.util_viz import *
from hrl_anomaly_detection import util as util
from hrl_anomaly_detection import data_manager as dm
import hrl_anomaly_detection.isolator.isolation_util as iutil
from hrl_execution_monitor import util as autil

# Private learners
from hrl_anomaly_detection.hmm import learning_hmm as hmm
## import hrl_anomaly_detection.classifiers.classifier as cf

#keras
from hrl_execution_monitor.keras_util import keras_model as km
from hrl_execution_monitor.keras_util import keras_util as ku
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import SGD, Adagrad, Adadelta, RMSprop
from keras.preprocessing.image import ImageDataGenerator

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

    save_data_path = os.path.join(save_data_path, 'keras')

    # training with signals
    ## train_with_signal(save_data_path, n_labels, nFold, nb_epoch=800, patience=50)
    ## train_with_signal(save_data_path, n_labels, nFold, nb_epoch=800, patience=50, load_weights=True)

    # training_with images
    remove_label = [1]
    ## train_with_image(save_data_path, n_labels, nFold, patience=20)
    ## train_with_image(save_data_path, n_labels, nFold, patience=20, fine_tune=True)
    
    train_with_image(save_data_path, n_labels, nFold, patience=20, vgg=True, remove_label=remove_label)
    ## train_with_image(save_data_path, n_labels, nFold, patience=20, vgg=True, remove_label=remove_label,
    ##                  load_weights=True)

    # training_with all
    ## train_with_all(save_data_path, n_labels, nFold, patience=10)
    ## train_with_all(save_data_path, n_labels, nFold, load_weights=True, patience=20)

    return



def get_isolator_modules(save_data_path, task_name, method, param_dict, fold_idx=0, \
                         nDetector=1, verbose=False):

    # load param
    hmm_list = []
    for i in xrange(nDetector):
        hmm_pkl = os.path.join(save_data_path, 'hmm_'+task_name+'_'+str(fold_idx)+'_c'+str(i)+'.pkl')

        d     = ut.load_pickle(hmm_pkl)
        m_gen = hmm.learning_hmm(d['nState'], d['nEmissionDim'], verbose=verbose)
        m_gen.set_hmm_object(d['A'], d['B'], d['pi'])
        hmm_list.append(m_gen)

    return hmm_list




def load_data(idx, save_data_path, extra_img=False, viz=False):
    ''' Load selected fold's data '''

    assert os.path.isfile(os.path.join(save_data_path,'x_train_img_'+str(idx)+'.npy')) == True, \
      "No preprocessed data"
        
    x_train_sig = np.load(open(os.path.join(save_data_path,'x_train_sig_'+str(idx)+'.npy')))
    x_train_img = np.load(open(os.path.join(save_data_path,'x_train_img_'+str(idx)+'.npy')))
    y_train = np.load(open(os.path.join(save_data_path,'y_train_'+str(idx)+'.npy')))

    x_test_sig = np.load(open(os.path.join(save_data_path,'x_test_sig_'+str(idx)+'.npy')))
    x_test_img = np.load(open(os.path.join(save_data_path,'x_test_img_'+str(idx)+'.npy')))
    y_test = np.load(open(os.path.join(save_data_path,'y_test_'+str(idx)+'.npy')))

    if extra_img:
        x_train_img_extra = np.load(open(os.path.join(save_data_path,'x_train_img_extra.npy')))
        y_train_extra = np.load(open(os.path.join(save_data_path,'y_train_extra.npy')))

        x_train_img = np.vstack([x_train_img, x_train_img_extra])
        y_train     = np.vstack([y_train, y_train_extra])

    return (x_train_sig, x_train_img, y_train), (x_test_sig, x_test_img, y_test)


def train_with_signal(save_data_path, n_labels, nFold, nb_epoch=400, load_weights=False,
                      activ_type='relu',
                      test_only=False, save_pdf=False, patience=50):

    scores= []
    y_test_list = []
    y_pred_list = []
    for idx in xrange(nFold):

        # Loading data
        train_data, test_data = load_data(idx, save_data_path, viz=False)      
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

    print 
    print np.mean(scores), np.std(scores)
    if test_only: return y_test_list, y_pred_list
    return


def train_with_image(save_data_path, n_labels, nFold, nb_epoch=1000, load_weights=False, vgg=False,
                     patience=20, remove_label=[], use_extra_img=True):

    if vgg: prefix = 'vgg_'
    else: prefix = ''

    scores= []
    for idx in xrange(nFold):

        # Loading data
        train_data, test_data = load_data(idx, save_data_path, extra_img=use_extra_img, viz=False)      
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
            if vgg: model = km.vgg16_net(np.shape(x_train_img)[1:], n_labels)
            else: model = km.cnn_net(np.shape(x_train_img)[1:], n_labels)            
        else:
            if vgg: model = km.vgg16_net(np.shape(x_train_img)[1:], n_labels, weights_path)
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
            width_shift_range=0.2,
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


def train_with_all(save_data_path, n_labels, nFold, nb_epoch=1, load_weights=False,
                   test_only=False, save_pdf=False, vgg=False, patience=30):

    if vgg: prefix = 'vgg_'
    else: prefix = ''

    scores= []
    y_test_list = []
    y_pred_list = []
    for idx in xrange(nFold):
        # Loading data
        train_data, test_data = load_data(idx, save_data_path, viz=False)      
        x_train_sig = train_data[0]
        x_train_img = train_data[1]
        y_train     = train_data[2]
        x_test_sig = test_data[0]
        x_test_img = test_data[1]
        y_test     = test_data[2]

        scaler      = preprocessing.StandardScaler()
        x_train_sig = scaler.fit_transform(x_train_sig)
        x_test_sig  = scaler.transform(x_test_sig)

        weights_path = os.path.join(save_data_path,prefix+'cnn_fc_weights_'+str(idx)+'.h5')
        callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=patience,
                                   verbose=0, mode='auto'),
                     ModelCheckpoint(weights_path,
                                     save_best_only=True,
                                     save_weights_only=True,
                                     monitor='val_loss'),
                     ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                       patience=20, min_lr=0.00001)]

        if load_weights is False:

            sig_weights_path=os.path.join(save_data_path,'sig_weights_'+str(idx)+'.h5'),
            img_weights_path=os.path.join(save_data_path,prefix+'cnn_weights_'+str(idx)+'.h5')
            
            # training
            if vgg:
                model = km.vgg16_net(np.shape(x_train_img)[1:], n_labels, with_top=True,
                                     input_shape2=np.shape(x_train_sig)[1:],
                                     sig_weights_path=sig_weights_path,
                                     img_weights_path=img_weights_path)
            else:
                model = km.cnn_net(np.shape(x_train_img)[1:], n_labels, with_top=True,
                                   input_shape2=np.shape(x_train_sig)[1:],
                                   sig_weights_path=sig_weights_path,
                                   img_weights_path=img_weights_path)
            ## model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        else:
            # fine tuning
            if vgg:
                model = km.vgg16_net(np.shape(x_train_img)[1:], n_labels, with_top=True,
                                     input_shape2=np.shape(x_train_sig)[1:],
                                     weights_path=weights_path)
            else:
                model = km.cnn_net(np.shape(x_train_img)[1:], n_labels, with_top=True,
                                   input_shape2=np.shape(x_train_sig)[1:],
                                   weights_path=weights_path)
            ## optimizer = SGD(lr=0.0001, decay=1e-8, momentum=0.9, nesterov=True)
            ## optimizer = RMSprop(lr=0.00001, rho=0.9, epsilon=1e-08, decay=0.0)

        optimizer = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        ## optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.005)                        
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
                

        if test_only is False:
            train_datagen = ku.myGenerator(augmentation=True, rescale=1./255.)
            test_datagen = ku.myGenerator(augmentation=False, rescale=1./255.)
            train_generator = train_datagen.flow(x_train_img, x_train_sig, y_train, batch_size=32) #128)
            test_generator = test_datagen.flow(x_test_img, x_test_sig, y_test, batch_size=32) #128)
        
            hist = model.fit_generator(train_generator,
                                       samples_per_epoch=len(y_train),
                                       nb_epoch=nb_epoch,
                                       validation_data=test_generator,
                                       nb_val_samples=len(y_test),
                                       callbacks=callbacks)

            scores.append( hist.history['val_acc'][-1] )
        else:
            model.load_weights( os.path.join(save_data_path,prefix+'cnn_fc_weights_'+str(idx)+'.h5') )
            y_pred = model.predict([x_test_img/255., x_test_sig])
            y_pred_list += np.argmax(y_pred, axis=1).tolist()
            y_test_list += np.argmax(y_test, axis=1).tolist()
            
            from sklearn.metrics import accuracy_score
            print "score : ", accuracy_score(y_test_list, y_pred_list)
            ## break
        gc.collect()



    # temp --------------------------------------------------------------
    weights_path = os.path.join(save_data_path,prefix+'cnn_fc_weights_0.h5')
    weights_file = h5py.File(sig_weights_path)
    weight = mutil.get_layer_weights(weights_file, layer_name='fc1_1')
    # -------------------------------------------------------------------

    print np.mean(scores), np.std(scores)
    if test_only: return y_test_list, y_pred_list
    return


def test(save_data_path):

    d = ut.load_pickle(os.path.join(save_data_path, 'isol_data.pkl'))
    nFold = len(d.keys())
    del d

    save_data_path = os.path.join(save_data_path, 'keras')

    train_data, test_data = load_data(0, save_data_path, viz=False)
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
    
    opt, args = p.parse_args()

    from hrl_execution_monitor.params.IROS2017_params import *
    # IROS2017
    subject_names = ['s2', 's3','s4','s5', 's6','s7','s8', 's9']
    raw_data_path, save_data_path, param_dict = getParams(opt.task, opt.bDataRenew, \
                                                          opt.bHMMRenew, opt.bCLFRenew)
                                                          
    save_data_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/IROS2017/'+opt.task+'_demo1'
    #window 0-5
    ## save_data_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/IROS2017/'+opt.task+'_demo2'

    task_name = 'feeding'
    method    = ['progress0', 'progress1'] 
    param_dict['ROC']['methods'] = ['progress0', 'progress1'] #'hmmgp'
    weight    = [-3.0, -4.5]
    param_dict['HMM']['scale'] = [2.0, 2.0]
    param_dict['HMM']['cov']   = 1.0
    single_detector=False    
    nb_classes = 12


    if opt.preprocessing:    
        # preprocessing data
        data_pkl = os.path.join(save_data_path, 'isol_data.pkl')
        ku.preprocess_data(data_pkl, save_data_path, img_scale=0.25, nb_classes=nb_classes,
                        img_feature_type='vgg')
    elif opt.preprocessing_extra:
        raw_data_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/AURO2016/raw_data/manual_label'
        ku.preprocess_images(raw_data_path, save_data_path, img_scale=0.25, nb_classes=nb_classes,
                             img_feature_type='vgg')        
    elif opt.train:
        train_isolator_modules(save_data_path, nb_classes, verbose=False)
        
    else:
        test(save_data_path)


    ## get_isolator_modules(save_data_path, task_name, method, param_dict, fold_idx=0,\
    ##                       verbose=False)
