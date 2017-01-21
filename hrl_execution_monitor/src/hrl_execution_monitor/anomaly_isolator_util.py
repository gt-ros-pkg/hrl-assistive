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
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
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
    ## train_with_signal(save_data_path, n_labels, nFold, nb_epoch=200, patience=10)
    ## train_with_signal(save_data_path, n_labels, nFold, nb_epoch=200, patience=10, fine_tune=True)

    # training_with images
    ## train_with_image(save_data_path, n_labels, nFold, patience=20)
    ## train_with_image(save_data_path, n_labels, nFold, patience=20, fine_tune=True)

    # training_with all
    ## train_with_all(save_data_path, n_labels, nFold, patience=10)
    ## train_with_all(save_data_path, n_labels, nFold, fine_tune=True, patience=10)
    ## train_with_all(save_data_path, n_labels, nFold, fine_tune=True, patience=10)
    ## train_with_all(save_data_path, n_labels, nFold, fine_tune=True, patience=10)
    ## train_with_all(save_data_path, n_labels, nFold, fine_tune=True, patience=10)
    train_with_all(save_data_path, n_labels, nFold, fine_tune=True, patience=10,
                   nb_epoch=1)

    return



def get_isolator_modules(save_data_path, task_name, method, param_dict, fold_idx=0, \
                          verbose=False):

    # load param
    scr_pkl = os.path.join(save_data_path, 'ai_scr_'+method+'_'+str(fold_idx)+'.pkl')
    hmm_pkl = os.path.join(save_data_path, 'hmm_'+task_name+'_'+str(fold_idx)+'.pkl')
    clf_pkl = os.path.join(save_data_path, 'ai_clf_'+method+'_'+\
                           str(fold_idx)+'.pkl')

    # load scaler
    import pickle
    if os.path.isfile(scr_pkl):
        with open(scr_pkl, 'rb') as f:
            m_scr = pickle.load(f)
    else: m_scr = None

    # load hmm
    if os.path.isfile(hmm_pkl) is False:
        print "No HMM pickle file: ", hmm_pkl
        sys.exit()
        
    d     = ut.load_pickle(hmm_pkl)
    print d.keys()
    m_gen = hmm.learning_hmm(d['nState'], d['nEmissionDim'], verbose=verbose)
    m_gen.set_hmm_object(d['A'], d['B'], d['pi'])


    # load classifier
    ## m_clf = cf.classifier( method=method, nPosteriors=d['nState'], parallel=True )
    ## m_clf.load_model(clf_pkl)

    return m_scr, m_gen, m_clf


def get_isolation_data(subject_names, task_name, raw_data_path, save_data_path,
                       param_dict, weight, window_steps=10, single_detector=False,
                       verbose=False):
    
    # load params (param_dict)
    data_dict  = param_dict['data_param']
    data_renew = data_dict['renew']
    # HMM
    HMM_dict   = param_dict['HMM']
    nState     = HMM_dict['nState']
    cov        = HMM_dict['cov']
    # SVM
    SVM_dict   = param_dict['SVM']
    # ROC
    ROC_dict = param_dict['ROC']
    method_list = param_dict['ROC']['methods']

    # parameters
    startIdx    = 4
    nPoints     = ROC_dict['nPoints']
    

    # load data (mix) -------------------------------------------------
    d = dm.getDataSet(subject_names, task_name, raw_data_path, \
                      save_data_path,\
                      downSampleSize=data_dict['downSampleSize'],\
                      handFeatures=data_dict['isolationFeatures'], \
                      data_renew=data_renew, max_time=data_dict['max_time'],\
                      ros_bag_image=True, rndFold=True)
                      
    # split data with 80:20 ratio, 3set
    kFold_list = d['kFold_list'][:1]

    # flattening image list
    success_image_list = autil.image_list_flatten( d.get('success_image_list',[]) )
    failure_image_list = autil.image_list_flatten( d.get('failure_image_list',[]) )

    failure_labels = []
    for f in d['failureFiles']:
        failure_labels.append( int( f.split('/')[-1].split('_')[0] ) )
    failure_labels = np.array( failure_labels )

    # Static feature selection for isolation
    feature_list = []
    for feature in param_dict['data_param']['staticFeatures']:
        idx = [ i for i, x in enumerate(param_dict['data_param']['isolationFeatures']) if feature == x][0]
        feature_list.append(idx)
    successData_static = np.array(d['successData'])[feature_list]
    failureData_static = np.array(d['failureData'])[feature_list]
    
    #-----------------------------------------------------------------------------------------
    # Dynamic feature selection for detection and isolation
    feature_idx_list = []
    success_data_ad = []
    failure_data_ad = []
    nDetector = len(param_dict['data_param']['handFeatures'])
    for i in xrange(nDetector):
        
        feature_idx_list.append([])
        for feature in param_dict['data_param']['handFeatures'][i]:
            feature_idx_list[i].append(data_dict['isolationFeatures'].index(feature))

        success_data_ad.append( copy.copy(d['successData'][feature_idx_list[i]]) )
        failure_data_ad.append( copy.copy(d['failureData'][feature_idx_list[i]]) )
        HMM_dict_local = copy.deepcopy(HMM_dict)
        HMM_dict_local['scale'] = param_dict['HMM']['scale'][i]
        
        ## # Training HMM, and getting classifier training and testing data
        ## dm.saveHMMinducedFeatures(kFold_list, success_data_ad[i], failure_data_ad[i],\
        ##                           task_name, save_data_path,\
        ##                           HMM_dict_local, data_renew, startIdx, nState, cov, \
        ##                           success_files=d['successFiles'], failure_files=d['failureFiles'],\
        ##                           noise_mag=0.03, suffix=str(i),\
        ##                           verbose=verbose, one_class=False)

    del d

    # ---------------------------------------------------------------
    # get data
    data_dict = {}
    data_pkl = os.path.join(save_data_path, 'isol_data.pkl')
    if os.path.isfile(data_pkl) is False or HMM_dict['renew'] or SVM_dict['renew']:

        l_data = Parallel(n_jobs=1, verbose=10)\
          (delayed(iutil.get_hmm_isolation_data)(idx, kFold_list[idx], failure_data_ad, \
                                                 failureData_static, \
                                                 failure_labels,\
                                                 failure_image_list,\
                                                 task_name, save_data_path, param_dict, weight,\
                                                 single_detector=single_detector,\
                                                 n_jobs=-1, window_steps=window_steps, verbose=verbose\
                                                 ) for idx in xrange(len(kFold_list)) )
        
        data_dict = {}
        for i in xrange(len(l_data)):
            idx = l_data[i][0]
            data_dict[idx] = (l_data[i][1],l_data[i][2],l_data[i][3],l_data[i][4] )
            
        print "save pkl: ", data_pkl
        ut.save_pickle(data_dict, data_pkl)            
    else:
        data_dict = ut.load_pickle(data_pkl)
    

    # ---------------------------------------------------------------
    scores = []
    for idx, (normalTrainIdx, abnormalTrainIdx, normalTestIdx, abnormalTestIdx) \
      in enumerate(kFold_list):
        print "kFold_list: ", idx

        (x_trains, y_train, x_tests, y_test) = data_dict[idx]         
        x_train = x_trains[0] 
        x_test  = x_tests[0] 
        print np.shape(x_train), np.shape(x_test)

        scaler = preprocessing.StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test  = scaler.transform(x_test)

        if type(x_train) is np.ndarray:
            x_train = x_train.tolist()
            x_test  = x_test.tolist()
        if type(y_train) is np.ndarray:
            y_train  = y_train.tolist()
            y_test   = y_test.tolist()
        
        ## from sklearn.svm import SVC
        ## clf = SVC(C=1.0, kernel='rbf') #, decision_function_shape='ovo')
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=400, n_jobs=-1)

        clf.fit(x_train, y_train)
        ## y_pred = clf.predict(x_test.tolist())
        score = clf.score(x_test, y_test)
        scores.append( score )
        print idx, " : score = ", score


    print scores
    print "Score mean = ", np.mean(scores), np.std(scores)



def load_data(idx, save_data_path, viz=False):
    ''' Load selected fold's data '''

    assert os.path.isfile(os.path.join(save_data_path,'x_train_img_'+str(idx)+'.npy')) == True, \
      "No preprocessed data"
        
    x_train_sig = np.load(open(os.path.join(save_data_path,'x_train_sig_'+str(idx)+'.npy')))
    x_train_img = np.load(open(os.path.join(save_data_path,'x_train_img_'+str(idx)+'.npy')))
    y_train = np.load(open(os.path.join(save_data_path,'y_train_'+str(idx)+'.npy')))

    x_test_sig = np.load(open(os.path.join(save_data_path,'x_test_sig_'+str(idx)+'.npy')))
    x_test_img = np.load(open(os.path.join(save_data_path,'x_test_img_'+str(idx)+'.npy')))
    y_test = np.load(open(os.path.join(save_data_path,'y_test_'+str(idx)+'.npy')))

    return (x_train_sig, x_train_img, y_train), (x_test_sig, x_test_img, y_test)


def train_with_signal(save_data_path, n_labels, nFold, nb_epoch=400, fine_tune=False, activ_type='relu',
                      test_only=False, save_pdf=False, patience=30):

    callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')]
    ## callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')]
    ## callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.2,
    ##                                patience=5, min_lr=0.0001)]

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

        # Debug
        ## x_train_sig_dyn1 = x_train_sig[:,:15]
        ## x_train_sig_dyn2 = x_train_sig[:,15:-7]#[:,[0, 3, 6]]
        ## x_train_sig_stc = x_train_sig[:,-7:]#[:,[0,1,2,4,5,6,7]]

        ## ## x_train_sig_dyn1 /= np.amax(x_train_sig_dyn1, axis=1)[:,np.newaxis]
        ## ## x_train_sig_dyn2 /= np.amax(x_train_sig_dyn2, axis=1)[:,np.newaxis]
        ## x_train_sig = np.hstack([x_train_sig_dyn1, x_train_sig_dyn2, x_train_sig_stc])
        
        ## x_test_sig_dyn1 = x_test_sig[:,:15]
        ## x_test_sig_dyn2 = x_test_sig[:,15:-7]#[:,[0, 3, 6,]]
        ## x_test_sig_stc = x_test_sig[:,-7:]#[:,[0,1,2,4,5,6,7]]
        ## ## x_test_sig_dyn1 /= np.amax(x_test_sig_dyn1, axis=1)[:,np.newaxis]
        ## ## x_test_sig_dyn2 /= np.amax(x_test_sig_dyn2, axis=1)[:,np.newaxis]
        ## x_test_sig = np.hstack([x_test_sig_dyn1, x_test_sig_dyn2, x_test_sig_stc])

        # Scaling
        from sklearn import preprocessing
        scaler      = preprocessing.StandardScaler()
        x_train_sig = scaler.fit_transform(x_train_sig)
        x_test_sig  = scaler.transform(x_test_sig)


        full_weights_path = os.path.join(save_data_path,'sig_weights_'+str(idx)+'.h5')

        ## # Load pre-trained vgg16 model
        if fine_tune is False:
            model = km.sig_net(np.shape(x_train_sig)[1:], n_labels, activ_type=activ_type)
            ## optimizer = SGD(lr=0.001, decay=1e-5, momentum=0.9, nesterov=True)
            optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
            ## model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        else:
            model = km.sig_net(np.shape(x_train_sig)[1:], n_labels, fine_tune=True,\
                               weights_path = full_weights_path, activ_type=activ_type)
        
            ## optimizer = SGD(lr=0.01, decay=1e-5, momentum=0.9, nesterov=True)
            ## optimizer = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
            optimizer = SGD(lr=0.0001, decay=1e-7, momentum=0.9, nesterov=True)
            ## optimizer = RMSprop(lr=0.00001, rho=0.9, epsilon=1e-08, decay=0.0)
            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        if test_only is False:
            train_datagen = ku.sigGenerator(augmentation=True, noise_mag=0.05 )
            test_datagen = ku.sigGenerator(augmentation=False)
            train_generator = train_datagen.flow(x_train_sig, y_train, batch_size=128)
            test_generator = test_datagen.flow(x_test_sig, y_test, batch_size=128)

            hist = model.fit_generator(train_generator,
                                       samples_per_epoch=len(y_train),
                                       nb_epoch=nb_epoch,
                                       validation_data=test_generator,
                                       nb_val_samples=len(y_test),
                                       callbacks=callbacks)

            scores.append( hist.history['val_acc'][-1] )
            model.save_weights(full_weights_path)
            del model
        else:
            model.load_weights(full_weights_path)
            y_pred = model.predict(x_test_sig)
            y_pred_list += np.argmax(y_pred, axis=1).tolist()
            y_test_list += np.argmax(y_test, axis=1).tolist()
            
            from sklearn.metrics import accuracy_score
            print "score : ", accuracy_score(y_test_list, y_pred_list)

    print 
    print np.mean(scores), np.std(scores)
    if test_only: return y_test_list, y_pred_list
    return


def train_with_image(save_data_path, n_labels, nFold, nb_epoch=100, fine_tune=False, vgg=False,
                     patience=10):

    if vgg: prefix = 'vgg_'
    else: prefix = ''
    callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=0, mode='auto')]

    scores= []
    for idx in xrange(nFold):

        # Loading data
        train_data, test_data = load_data(idx, save_data_path, viz=False)      
        ## x_train_sig = train_data[0]
        x_train_img = train_data[1]
        y_train     = train_data[2]
        ## x_test_sig = test_data[0]
        x_test_img = test_data[1]
        y_test     = test_data[2]
        
        full_weights_path = os.path.join(save_data_path,prefix+'cnn_weights_'+str(idx)+'.h5')

        # Load pre-trained vgg16 model
        ## model = km.vgg16_net(np.shape(x_train_img)[1:], n_labels, vgg_model_weights_path, \
        ##                      full_weights_path, fine_tune=False)
        ## model = km.vgg16_net(np.shape(x_train_img)[1:], n_labels, vgg_model_weights_path, \
        ##                      fine_tune=False)

        if fine_tune is False:
            if vgg: model = km.vgg16_net(np.shape(x_train_img)[1:], n_labels, vgg_model_weights_path)
            else: model = km.cnn_net(np.shape(x_train_img)[1:], n_labels)            
            model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        else:
            if vgg: model = km.vgg16_net(np.shape(x_train_img)[1:], n_labels, vgg_model_weights_path,\
                                        full_weights_path)
            else: model = km.cnn_net(np.shape(x_train_img)[1:], n_labels, full_weights_path)
            optimizer = SGD(lr=0.0001, decay=1e-8, momentum=0.9, nesterov=True)
            ## optimizer = RMSprop(lr=0.00001, rho=0.9, epsilon=1e-08, decay=0.0)
            ## model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
            model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


        train_datagen = ImageDataGenerator(
            rotation_range=20,
            rescale=1./255,
            width_shift_range=0.3,
            height_shift_range=0.2,
            ## zoom_range=0.1,
            horizontal_flip=False,
            fill_mode='nearest',
            dim_ordering="th")
        test_datagen = ImageDataGenerator(rescale=1./255,\
                                          dim_ordering="th")

        train_generator = train_datagen.flow(x_train_img, y_train, batch_size=64) #128)
        test_generator = test_datagen.flow(x_test_img, y_test, batch_size=64) #128)

        hist = model.fit_generator(train_generator,
                                   samples_per_epoch=len(y_train),
                                   nb_epoch=nb_epoch,
                                   validation_data=test_generator,
                                   nb_val_samples=len(y_test),
                                   callbacks=callbacks)

        print "Saved weights!!!!!!!!!!!!!!!!!!!!!!!!"
        model.save_weights(full_weights_path)

        scores.append( hist.history['val_acc'][-1] )
        gc.collect()

    print 
    print np.mean(scores), np.std(scores)
    return


def train_with_all(save_data_path, n_labels, nFold, nb_epoch=100, fine_tune=False,
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

        if fine_tune is False:
            # training
            if vgg:
                model = km.vgg16_net(np.shape(x_train_img)[1:], n_labels, vgg_model_weights_path,\
                                     with_top=True, fine_tune=False,
                                     input_shape2=np.shape(x_train_sig)[1:] )
            else:
                model = km.cnn_net(np.shape(x_train_img)[1:], n_labels, \
                                   with_top=True, fine_tune=False,
                                   input_shape2=np.shape(x_train_sig)[1:] )
            model.load_weights( os.path.join(save_data_path,'sig_weights_'+str(idx)+'.h5'), by_name=True )
            model.load_weights( os.path.join(save_data_path,prefix+'cnn_weights_'+str(idx)+'.h5'), \
                                by_name=True )
            model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        else:
            # fine tuning
            if vgg:
                model = km.vgg16_net(np.shape(x_train_img)[1:], n_labels, vgg_model_weights_path,\
                                     with_top=True, fine_tune=True,
                                     input_shape2=np.shape(x_train_sig)[1:] )
            else:
                model = km.cnn_net(np.shape(x_train_img)[1:], n_labels, \
                                   with_top=True, fine_tune=True,
                                   input_shape2=np.shape(x_train_sig)[1:] )
            model.load_weights( os.path.join(save_data_path,prefix+'cnn_fc_weights_'+str(idx)+'.h5') )
            optimizer = SGD(lr=0.0001, decay=1e-8, momentum=0.9, nesterov=True)
            ## optimizer = RMSprop(lr=0.00001, rho=0.9, epsilon=1e-08, decay=0.0)
            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

                

        if test_only is False:
            train_datagen = ku.myGenerator(augmentation=True, rescale=1./255.)
            test_datagen = ku.myGenerator(augmentation=False, rescale=1./255.)
            train_generator = train_datagen.flow(x_train_img, x_train_sig, y_train, batch_size=32) #128)
            test_generator = test_datagen.flow(x_test_img, x_test_sig, y_test, batch_size=32) #128)
            callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=patience,
                                       verbose=0, mode='auto')]
        
            hist = model.fit_generator(train_generator,
                                       samples_per_epoch=len(y_train),
                                       nb_epoch=nb_epoch,
                                       validation_data=test_generator,
                                       nb_val_samples=len(y_test),
                                       callbacks=callbacks)

            full_weights_path = os.path.join(save_data_path,prefix+'cnn_fc_weights_'+str(idx)+'.h5')
            print "Saved weights"
            model.save_weights(full_weights_path)

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

    for x in test_data[0]:
        if np.any(np.equal(x, None)):
            print x
    for x in test_data[1]:
        if np.any(np.equal(x, None)):
            print x
    for x in test_data[2]:
        if np.any(np.equal(x, None)):
            print x

    


if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    util.initialiseOptParser(p)

    p.add_option('--get_data', '--gd', action='store_true', dest='get_data',
                 default=False, help='Get data')
    p.add_option('--preprocess', '--p', action='store_true', dest='preprocessing',
                 default=False, help='Preprocess')
    p.add_option('--train', '--tr', action='store_true', dest='train',
                 default=False, help='Train')
    
    opt, args = p.parse_args()

    from hrl_execution_monitor.params.IROS2017_params import *
    # IROS2017
    subject_names = ['s2', 's3','s4','s5', 's6','s7','s8', 's9']
    raw_data_path, save_data_path, param_dict = getParams(opt.task, opt.bDataRenew, \
                                                          opt.bHMMRenew, opt.bCLFRenew)
    ## save_data_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/IROS2017/'+opt.task+'_demo1'
    save_data_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/IROS2017/'+opt.task+'_demo2'

    task_name = 'feeding'
    method    = ['progress0', 'progress1'] 
    param_dict['ROC']['methods'] = ['progress0', 'progress1'] #'hmmgp'
    weight    = [-3.0, -4.5]
    param_dict['HMM']['scale'] = [2.0, 2.0]
    param_dict['HMM']['cov']   = 1.0
    single_detector=False    
    window_steps= 5
    nb_classes = 12


    if opt.get_data:
        get_isolation_data(subject_names, task_name, raw_data_path, save_data_path,
                           param_dict, weight, single_detector=single_detector,
                           window_steps=window_steps, verbose=False)

    elif opt.preprocessing:    
        # preprocessing data
        data_pkl = os.path.join(save_data_path, 'isol_data.pkl')
        ku.preprocess_data(data_pkl, save_data_path, img_scale=0.25, nb_classes=nb_classes,
                        img_feature_type='vgg')
        
    elif opt.train:
        train_isolator_modules(save_data_path, nb_classes, verbose=False)
        
    else:
        test(save_data_path)


    ## get_isolator_modules(save_data_path, task_name, method, param_dict, fold_idx=0,\
    ##                       verbose=False)
