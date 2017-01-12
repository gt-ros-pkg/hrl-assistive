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
import hrl_anomaly_detection.isolator.isolation_viz as iviz
from joblib import Parallel, delayed

random.seed(3334)
np.random.seed(3334)

from sklearn import preprocessing
import h5py
import cv2

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, Merge, Input
from keras.layers import Activation, Dropout, Flatten, Dense, merge
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD, Adagrad, Adadelta, RMSprop
from keras.utils.visualize_util import plot
from keras.layers.normalization import BatchNormalization

from hrl_anomaly_detection.isolator import keras_models as km

vgg_model_weights_path = '/home/dpark/git/keras_test/vgg16_weights.h5'
## nb_train_samples = 1000 #len(x_train)
## nb_validation_samples = 200 #len(x_test)
## nb_train_samples = 500 #2000
## nb_validation_samples = 200 #800
## nb_epoch = 200


def preprocess_data(save_data_path, scaler=4, n_labels=12, viz=False, hog_feature=False,
                    org_ratio=False):
    ''' Preprocessing data '''
    ## from sklearn import preprocessing
    ## scaler = preprocessing.StandardScaler()

    d = ut.load_pickle(os.path.join(save_data_path, 'isol_data.pkl'))
    nFold = len(d.keys())

    for idx in xrange(nFold):

        (x_trains, y_train, x_tests, y_test) = d[idx]         
        x_train_sig = x_trains[0] #signals
        x_test_sig  = x_tests[0]
        x_train_img = x_trains[1] #img
        x_test_img  = x_tests[1]
        print "Training data: ", np.shape(x_train_img), np.shape(x_train_sig), np.shape(y_train)
        print "Testing data: ", np.shape(x_test_img), np.shape(x_test_sig), np.shape(y_test)

        ## if idx == 0:
        ##     x = np.vstack([x_train_sig, x_test_sig])
        ##     x_train_sig = scaler.fit_transform(x_train_sig)
        ##     x_test_sig  = scaler.transform(x_test_sig)
            
        ## x_train_sig = scaler.fit_transform(x_train_sig)
        ## x_test_sig  = scaler.transform(x_test_sig)

        #--------------------------------------------------------------------
        # check images
        rm_idx = []
        x_train = []
        for j, f in enumerate(x_train_img):
            if f is None:
                print "None image ", j+1, '/', len(x_train_img)
                rm_idx.append(j)
                continue

            if hog_feature is False:
                if org_ratio:
                    img = cv2.imread(f)
                    height, width = np.array(img).shape[:2]
                    img = cv2.resize(img,(width/scaler, height/scaler), interpolation = cv2.INTER_CUBIC)
                    img = img.astype(np.float32)
                else:                    
                    img = cv2.resize(cv2.imread(f), (224, 224)).astype(np.float32)
                img[:,:,0] -= 103.939
                img[:,:,1] -= 116.779
                img[:,:,2] -= 123.68

                if viz:
                    print np.shape(img), type(img), type(img[0][0][0])
                    # visual check
                    cv2.imshow('image',img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    sys.exit()

                img = img.transpose((2,0,1))
            else:
                ## img = cv2.imread(f,0)
                ## height, width = np.array(img).shape[:2]
                ## scaler = 1
                ## img = cv2.resize(img,(width/scaler, height/scaler), interpolation = cv2.INTER_CUBIC)
                ## img = hog(img, bin_n=16)

                img = cv2.imread(f)
                gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                sift = cv2.xfeatures2d.SIFT_create()
                kp = sift.detect(gray,None)
                img=cv2.drawKeypoints(gray,kp,img)
                print np.shape(img)
                sys.exit()
                
            x_train.append(img)
        x_train_img = x_train

        # check signals and labels
        x_train_sig = [x_train_sig[i] for i in xrange(len(x_train_sig)) if i not in rm_idx ]
        y_train = [y_train[i] for i in xrange(len(y_train)) if i not in rm_idx ]
        y_train = np.array(y_train)-2 # make label start from zero
        if hog_feature is False:
            y_train = to_categorical(y_train, nb_classes=n_labels)

        np.save(open(os.path.join(save_data_path,'x_train_img_'+str(idx)+'.npy'), 'w'), x_train_img)
        np.save(open(os.path.join(save_data_path,'x_train_sig_'+str(idx)+'.npy'), 'w'), x_train_sig)
        np.save(open(os.path.join(save_data_path,'y_train_'+str(idx)+'.npy'), 'w'), y_train)

        #--------------------------------------------------------------------
        # check images
        rm_idx = []
        x_test = []
        for j, f in enumerate(x_test_img):
            if f is None:
                print "None image ", j+1, '/', len(x_test_img)
                rm_idx.append(j)
                continue

            if hog_feature is False:
                if org_ratio:
                    img = cv2.imread(f)
                    height, width = np.array(img).shape[:2]
                    img = cv2.resize(img,(width/scaler, height/scaler), interpolation = cv2.INTER_CUBIC)
                    img = img.astype(np.float32)
                else:                    
                    img = cv2.resize(cv2.imread(f), (224, 224)).astype(np.float32)
                img[:,:,0] -= 103.939
                img[:,:,1] -= 116.779
                img[:,:,2] -= 123.68
                img = img.transpose((2,0,1))
            else:
                ## img = cv2.imread(f,0)
                ## height, width = np.array(img).shape[:2]
                ## scaler = 1
                ## img = cv2.resize(img,(width/scaler, height/scaler), interpolation = cv2.INTER_CUBIC)
                ## img = hog(img, bin_n=16)

                img = cv2.imread(f)
                gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                sift = cv2.xfeatures2d.SIFT_create()
                kp = sift.detect(gray,None)
                img=cv2.drawKeypoints(gray,kp,img)

            x_test.append(img)
        x_test_img = x_test

        # check singlas
        x_test_sig = [x_test_sig[i] for i in xrange(len(x_test_sig)) if i not in rm_idx  ]
        y_test = [y_test[i] for i in xrange(len(y_test)) if i not in rm_idx  ]
        y_test = np.array(y_test)-2 # make label start from zero
        if hog_feature is False:
            y_test = to_categorical(y_test, nb_classes=n_labels)

        np.save(open(os.path.join(save_data_path,'x_test_img_'+str(idx)+'.npy'), 'w'), x_test_img)
        np.save(open(os.path.join(save_data_path,'x_test_sig_'+str(idx)+'.npy'), 'w'), x_test_sig)
        np.save(open(os.path.join(save_data_path,'y_test_'+str(idx)+'.npy'), 'w'), y_test)




def hog(img, bin_n=16):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     # hist is a 64 bit vector
    return hist


def bottom_feature_extraction(save_data_path, n_labels, augmentation=False, model_viz=False):

    d = ut.load_pickle(os.path.join(save_data_path, 'isol_data.pkl'))
    nFold = len(d.keys())
    del d

    for idx in xrange(nFold):
        train_data, test_data = km.load_data(idx, save_data_path, viz=False)      
        x_train_sig = train_data[0]
        x_train_img = train_data[1]
        y_train = train_data[2]
        x_test_sig  = test_data[0]
        x_test_img  = test_data[1]
        y_test  = test_data[2]        
        print "Data: ", np.shape(x_train_img), np.shape(y_train), np.shape(x_test_img), np.shape(y_test)
        
        img_width, img_height =  np.shape(x_train_img)[2:]
        print "width, height: ", img_width, img_height

        # Load pre-trained vgg16 model
        model = km.get_vgg16_model(img_width, img_height, vgg_model_weights_path)
        if model_viz:
            plot(model, to_file='model.png')
            sys.exit()

        if augmentation:
            x_trains_sig = None
            x_trains_img = None
            y_trains = None
            
            print "Extracting training features"
            for i in xrange(len(y_train)):
                datagen = ImageDataGenerator(
                    rotation_range=20,
                    rescale=1./255,
                    width_shift_range=0.4,
                    height_shift_range=0.2,
                    zoom_range=0.1,
                    horizontal_flip=False,
                    fill_mode='nearest',
                    dim_ordering="th")

                count = 0
                for x_batch, y_batch in datagen.flow(x_train_img[i:i+1], y_train[i:i+1], batch_size=32,\
                      save_to_dir='preview', save_prefix='cat', save_format='jpeg'):

                    # TODO: add noise to signal                    
                    x_feature = model.predict(x_batch)
                    if x_trains_img is None:
                        x_trains_img = x_feature
                        x_trains_sig = x_train_sig[i:i+1] + np.random.normal(0.0, 0.001, \
                                                                             np.shape(x_train_sig[i:i+1]))
                        y_trains = y_batch
                    else:
                        x_trains_img = np.vstack([x_trains_img, x_feature])
                        x_trains_sig = np.vstack([x_trains_sig, x_train_sig[i:i+1] + \
                                                  np.random.normal(0.0, 0.001, \
                                                                   np.shape(x_train_sig[i:i+1])) ])
                        y_trains = np.vstack([y_trains, y_batch])

                    ## ## temp = x_train_img[0]
                    ## temp = x_batch[0] #* 255
                    ## temp = temp.astype(np.uint8)                    
                    ## temp = np.swapaxes(temp,0,1)
                    ## temp = np.swapaxes(temp,1,2)
                    ## print np.amin(temp), np.amax(temp)
                    ## cv2.imshow('image',temp)
                    ## cv2.waitKey(0)
                    ## cv2.destroyAllWindows()        
                    ## sys.exit()
                    
                    count += 1
                    if count==3: break

            np.save(open(os.path.join(save_data_path,'x_train_img_bottom_'+str(idx)+'.npy'), 'w'),\
                    x_trains_img)
            np.save(open(os.path.join(save_data_path,'x_train_sig_bottom_'+str(idx)+'.npy'), 'w'),\
                    x_trains_sig)            
            np.save(open(os.path.join(save_data_path,'y_train_bottom_'+str(idx)+'.npy'), 'w'), y_trains)
        else:
            print "Extracting training features"
            x_train_feature = model.predict(x_train_img/255)
            np.save(open(os.path.join(save_data_path,'x_train_img_bottom_'+str(idx)+'.npy'), 'w'),\
                    x_train_feature)
            np.save(open(os.path.join(save_data_path,'x_train_sig_bottom_'+str(idx)+'.npy'), 'w'),\
                    x_train_sig)
            np.save(open(os.path.join(save_data_path,'y_train_bottom_'+str(idx)+'.npy'), 'w'), y_train)
            
        print "Extracting testing features"
        x_test_feature = model.predict(x_test_img/255)
        np.save(open(os.path.join(save_data_path,'x_test_img_bottom_'+str(idx)+'.npy'), 'w'), x_test_feature)
        np.save(open(os.path.join(save_data_path,'x_test_sig_bottom_'+str(idx)+'.npy'), 'w'), x_test_sig)
        np.save(open(os.path.join(save_data_path,'y_test_bottom_'+str(idx)+'.npy'), 'w'), y_test)

        #temp
        ## break
        


def top_model_train(n_labels, nb_epoch=200):

    d = ut.load_pickle(os.path.join(save_data_path, 'isol_data.pkl'))
    nFold = len(d.keys())
    del d

    scores = []
    for idx in xrange(nFold):
        ## if idx < nFold-1: continue
        ## if idx > 0: continue

        # Loading data
        x_train_sig = np.load(open(os.path.join(save_data_path,'x_train_sig_bottom_'+str(idx)+'.npy')))
        x_train_img = np.load(open(os.path.join(save_data_path,'x_train_img_bottom_'+str(idx)+'.npy')))
        y_train     = np.load(open(os.path.join(save_data_path,'y_train_bottom_'+str(idx)+'.npy')))
        x_test_sig  = np.load(open(os.path.join(save_data_path,'x_test_sig_bottom_'+str(idx)+'.npy')))
        x_test_img  = np.load(open(os.path.join(save_data_path,'x_test_img_bottom_'+str(idx)+'.npy')))
        y_test      = np.load(open(os.path.join(save_data_path,'y_test_bottom_'+str(idx)+'.npy')))

        # Preprocessing
        ## scaler = preprocessing.MinMaxScaler()
        scaler = preprocessing.StandardScaler()
        x_train_sig = scaler.fit_transform(x_train_sig)
        x_test_sig  = scaler.transform(x_test_sig)

        model = km.get_top_model(x_train_img.shape[1:], x_train_sig.shape[1:], n_labels=n_labels)
        ## plot(model, to_file='model.png')
        ## sys.exit()

        ## sgd = SGD(lr=0.0001, decay=5e-7, momentum=0.9, nesterov=True)
        ## optimizer = SGD(lr=0.001, decay=5e-7, momentum=0.9, nesterov=True)
        ## sgd = Adagrad(lr=0.001, epsilon=1e-08, decay=0.0)
        ## sgd = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
        ## optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.001)
        ## model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        hist = model.fit([x_train_img, x_train_sig], y_train,
                         nb_epoch=nb_epoch, batch_size=32, shuffle=True,
                         validation_data=([x_test_img, x_test_sig], y_test))

        top_model_weights_path = os.path.join(save_data_path,'fc_weights_'+str(idx)+'.h5')
        model.save_weights(top_model_weights_path)

        scores.append( hist.history['val_acc'][-1] )
        break

    print 
    print np.mean(scores), np.std(scores)
    return


def top_model_train2(save_data_path, n_labels, augmentation=False, nb_epoch=200):

    d = ut.load_pickle(os.path.join(save_data_path, 'isol_data.pkl'))
    nFold = len(d.keys())
    del d

    scores = []
    for idx in xrange(nFold):
        train_data, test_data = km.load_data(idx, save_data_path, viz=False)      
        x_train_sig = train_data[0]
        x_train_img = train_data[1]
        y_train     = train_data[2]
        x_test_sig  = test_data[0]
        x_test_img  = test_data[1]
        y_test      = test_data[2]        
        print "Train Data: ", np.shape(x_train_sig), np.shape(y_train)
        print "Test data: ", np.shape(x_test_sig), np.shape(y_test)
        
        img_width, img_height =  np.shape(x_train_img)[2:]
        print "width, height: ", img_width, img_height

        scaler = preprocessing.StandardScaler()
        x_train_sig = scaler.fit_transform(x_train_sig)
        x_test_sig  = scaler.transform(x_test_sig)
        
        # ---------------------------------------------------------------
        ## top_model_weights_path = os.path.join(save_data_path,'fc_weights_'+str(idx)+'.h5')
        model = km.get_vgg16_model(img_width, img_height, vgg_model_weights_path, \
                                x_train_sig.shape[1:],\
                                with_top=True, vgg_hold=True)
            
        ## sgd = SGD(lr=0.0001, decay=5e-7, momentum=0.9, nesterov=True)
        ## model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
        optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        hist = model.fit([x_train_img, x_train_sig], y_train,
                         nb_epoch=nb_epoch, batch_size=32, shuffle=True,
                         validation_data=([x_test_img, x_test_sig], y_test))
        
        full_model_weights_path = os.path.join(save_data_path,'full_weights_'+str(idx)+'.h5')
        model.save_weights(full_model_weights_path)

        scores.append( hist.history['val_acc'][-1] )

    print np.mean(scores), np.std(scores)
    return




def fine_tune(save_data_path, n_labels, augmentation=False, nb_epoch=10):

    d = ut.load_pickle(os.path.join(save_data_path, 'isol_data.pkl'))
    nFold = len(d.keys())
    del d

    scores = []
    for idx in xrange(nFold):
        train_data, test_data = km.load_data(idx, save_data_path, viz=False)      
        x_train_sig = train_data[0]
        x_train_img = train_data[1]
        y_train     = train_data[2]
        x_test_sig  = test_data[0]
        x_test_img  = test_data[1]
        y_test      = test_data[2]        
        print "Train Data: ", np.shape(x_train_sig), np.shape(y_train),
        print "Test data: ", np.shape(x_test_sig), np.shape(y_test)
        
        img_width, img_height =  np.shape(x_train_img)[2:]
        print "width, height: ", img_width, img_height

        scaler = preprocessing.StandardScaler()
        x_train_sig = scaler.fit_transform(x_train_sig)
        x_test_sig  = scaler.transform(x_test_sig)
        
        # ---------------------------------------------------------------
        top_model_weights_path = os.path.join(save_data_path,'full_weights_'+str(idx)+'.h5')
        model = km.get_vgg16_model(img_width, img_height, vgg_model_weights_path, \
                                x_train_sig.shape[1:],\
                                top_model_weights_path, with_top=True)
        ## plot(model, to_file='model.png')
        ## sys.exit()
        ## for layer in model.layers[:25]:
        ##     layer.trainable = False
            
        ## sgd = SGD(lr=0.0001, decay=5e-7, momentum=0.9, nesterov=True)
        ## model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
        ## optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        optimizer = RMSprop(lr=0.00001, rho=0.9, epsilon=1e-08, decay=0.0)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        hist = model.fit([x_train_img, x_train_sig], y_train,
                         nb_epoch=nb_epoch, batch_size=32, shuffle=True,
                         validation_data=([x_test_img, x_test_sig], y_test))
        scores.append( hist.history['val_acc'][-1] )
        
        ## from sklearn.metrics import accuracy_score
        ## y_pred = model.predict([x_train_img, x_train_sig])
        ## train_score = accuracy_score(np.argmax(y_train, axis=1), np.argmax(y_pred, axis=1))
        ## y_pred = model.predict([x_test_img, x_test_sig])
        ## test_score = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
        
        ## print "score: ", train_score, test_score
        ## scores.append(test_score)

        ## full_model_weights_path = os.path.join(save_data_path,'full_weights_'+str(idx)+'.h5')
        ## model.save_weights(full_model_weights_path)

        ## break

    print np.mean(scores), np.std(scores)
    return

def evaluate_svm(save_data_path):

    d = ut.load_pickle(os.path.join(save_data_path, 'isol_data.pkl'))
    nFold = len(d.keys())
    del d

    scores = []
    for idx in xrange(nFold):
        train_data, test_data = km.load_data(idx, save_data_path, viz=False)      
        x_train_sig = train_data[0]
        x_train_img = train_data[1].astype(np.float32)
        y_train = train_data[2]
        x_test_sig  = test_data[0]
        x_test_img  = test_data[1].astype(np.float32)
        y_test  = test_data[2]        
        print "Data: ", np.shape(x_train_img), np.shape(x_train_sig), np.shape(y_train)

        if len(np.shape(y_train))>1: y_train = np.argmax(y_train, axis=1)
        if len(np.shape(y_test))>1: y_test = np.argmax(y_test, axis=1)

        ## x_train = np.hstack([x_train_sig, x_train_img])
        ## x_test = np.hstack([x_test_sig, x_test_img])
        x_train = x_train_sig
        x_test  = x_test_sig
        ## x_train = x_train_img
        ## x_test  = x_test_img

        x_train_dyn1 = x_train[:,:8]
        x_train_dyn2 = x_train[:,16:-6][:,:6]
        x_train_stc = x_train[:,-6:]
        ## x_train_dyn1 -= np.mean(x_train_dyn1, axis=1)[:,np.newaxis]
        ## x_train_dyn2 -= np.mean(x_train_dyn2, axis=1)[:,np.newaxis]
        x_train = np.hstack([x_train_dyn1, x_train_dyn2, x_train_stc])
        ## x_train = np.hstack([x_train_dyn1, x_train_stc])

        x_test_dyn1 = x_test[:,:8]
        x_test_dyn2 = x_test[:,16:-6][:,:6]
        x_test_stc = x_test[:,-6:]
        ## x_test_dyn1 -= np.mean(x_test_dyn1, axis=1)[:,np.newaxis]
        ## x_test_dyn2 -= np.mean(x_test_dyn2, axis=1)[:,np.newaxis]
        x_test = np.hstack([x_test_dyn1, x_test_dyn2, x_test_stc])
        ## x_test = np.hstack([x_test_dyn1, x_test_stc])
        
        ## scaler = preprocessing.MinMaxScaler()
        scaler = preprocessing.StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test  = scaler.transform(x_test)

        # train svm
        ## from sklearn.svm import SVC
        ## clf = SVC(C=1.0, kernel='rbf', gamma=1e-5) #, decision_function_shape='ovo')
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=400, n_jobs=-1)
        ## from sklearn.neighbors import KNeighborsClassifier
        ## clf = KNeighborsClassifier(n_neighbors=10, n_jobs=-1)
        
        clf.fit(x_train, y_train)

        # classify and get scores
        score = clf.score(x_test, y_test)
        scores.append(score)
        print "score: ", score
        
    print scores
    print np.mean(scores), np.std(scores)
    


def evaluate(save_data_path):

    d = ut.load_pickle(os.path.join(save_data_path, 'isol_data.pkl'))
    nFold = len(d.keys())
    del d

    for idx in xrange(nFold):

        # Loading data
        train_data, test_data = km.load_data(idx, save_data_path, viz=False)      
        x_test_sig = test_data[0]
        x_test_img = test_data[1]
        y_test    = test_data[2]

        img_width, img_height =  np.shape(x_train)[2:]
        print "width, height: ", img_width, img_height

        top_model_weights_path = os.path.join(save_data_path,'fc_weights_'+str(idx)+'.h5')

        # Load pre-trained vgg16 model
        model = km.get_all_model(img_width, img_height, vgg_weights_path, fc_weights_path)
        
        y_pred = model.predict([x_test_img, x_test_sig])

        from sklearn.metrics import accuracy_score
        score = accuracy_score(y_test, y_pred)

        sys.exit()
    
    return
    

def unimodal_fc(save_data_path, n_labels, nb_epoch=400, fine_tune=False):

    d = ut.load_pickle(os.path.join(save_data_path, 'isol_data.pkl'))
    nFold = len(d.keys())
    del d

    scores= []
    for idx in xrange(nFold):

        # Loading data
        train_data, test_data = km.load_data(idx, save_data_path, viz=False)      
        x_train_sig = train_data[0]
        ## x_train_img = train_data[1]
        y_train     = train_data[2]
        x_test_sig = test_data[0]
        ## x_test_img = test_data[1]
        y_test     = test_data[2]



        x_train_sig_dyn1 = x_train_sig[:,:16]
        x_train_sig_dyn2 = x_train_sig[:,16:-8]
        x_train_sig_stc = x_train_sig[:,-8:] #[:,[1,2,4,5,6,7]]
        x_train_sig_dyn1 -= np.mean(x_train_sig_dyn1, axis=1)[:,np.newaxis]
        x_train_sig_dyn2 -= np.mean(x_train_sig_dyn2, axis=1)[:,np.newaxis]
        ## x_train_sig = np.hstack([x_train_sig_dyn1, x_train_sig_dyn2, x_train_sig_stc])
        x_train_sig = np.hstack([x_train_sig_dyn1, x_train_sig_stc])
        ## x_train_sig = np.hstack([x_train_sig_dyn1, x_train_sig_dyn2])
        
        x_test_sig_dyn1 = x_test_sig[:,:16]
        x_test_sig_dyn2 = x_test_sig[:,16:-8]
        x_test_sig_stc = x_test_sig[:,-8:] #[:,[1,2,4,5,6,7]]
        x_test_sig_dyn1 -= np.mean(x_test_sig_dyn1, axis=1)[:,np.newaxis]
        x_test_sig_dyn2 -= np.mean(x_test_sig_dyn2, axis=1)[:,np.newaxis]
        ## x_test_sig = np.hstack([x_test_sig_dyn1, x_test_sig_dyn2, x_test_sig_stc])
        x_test_sig = np.hstack([x_test_sig_dyn1, x_test_sig_stc])
        ## x_test_sig = np.hstack([x_test_sig_dyn1, x_test_sig_dyn2])

        ## print np.shape(x_train_sig)
        ## sys.exit()

        ## scaler = preprocessing.MinMaxScaler()
        scaler      = preprocessing.StandardScaler()
        x_train_sig = scaler.fit_transform(x_train_sig)
        x_test_sig  = scaler.transform(x_test_sig)


        full_weights_path = os.path.join(save_data_path,'sig_weights_'+str(idx)+'.h5')

        ## # Load pre-trained vgg16 model
        if fine_tune is False:
            model = km.sig_net(np.shape(x_train_sig)[1:], n_labels)
            optimizer = SGD(lr=0.001, decay=1e-5, momentum=0.9, nesterov=True)
            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
            ## model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        else:
            model = km.sig_net(np.shape(x_train_sig)[1:], n_labels, fine_tune=True,\
                               weights_path = full_weights_path)
        
            ## optimizer = SGD(lr=0.01, decay=1e-5, momentum=0.9, nesterov=True)
            ## optimizer = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
            optimizer = SGD(lr=0.00001, decay=1e-8, momentum=0.9, nesterov=True)
            ## optimizer = RMSprop(lr=0.00001, rho=0.9, epsilon=1e-08, decay=0.0)
            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        hist = model.fit(x_train_sig, y_train,
                         nb_epoch=nb_epoch, batch_size=32, shuffle=True,
                         validation_data=(x_test_sig, y_test))
        scores.append( hist.history['val_acc'][-1] )
        model.save_weights(full_weights_path)
        del model

        break

    print 
    print np.mean(scores), np.std(scores)
    return

    
def unimodal_cnn(save_data_path, n_labels, nb_epoch=100, fine_tune=False, vgg=False):

    d = ut.load_pickle(os.path.join(save_data_path, 'isol_data.pkl'))
    nFold = len(d.keys())
    del d

    if vgg: prefix = 'vgg_'
    else: prefix = ''

    scores= []
    for idx in xrange(nFold):

        # Loading data
        train_data, test_data = km.load_data(idx, save_data_path, viz=False)      
        x_train_sig = train_data[0]
        x_train_img = train_data[1]
        y_train     = train_data[2]
        x_test_sig = test_data[0]
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
            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


        train_datagen = ImageDataGenerator(
            rotation_range=20,
            rescale=1./255,
            width_shift_range=0.2,
            height_shift_range=0.2,
            ## zoom_range=0.1,
            horizontal_flip=False,
            fill_mode='nearest',
            dim_ordering="th")
        test_datagen = ImageDataGenerator(rescale=1./255,\
                                          dim_ordering="th")

        train_generator = train_datagen.flow(x_train_img, y_train, batch_size=32)
        test_generator = test_datagen.flow(x_test_img, y_test, batch_size=32)

        hist = model.fit_generator(train_generator,
                                   samples_per_epoch=len(y_train),
                                   nb_epoch=nb_epoch,
                                   validation_data=test_generator,
                                   nb_val_samples=len(y_test))

        model.save_weights(full_weights_path)

        scores.append( hist.history['val_acc'][-1] )

    print 
    print np.mean(scores), np.std(scores)
    return
    
def multimodal_cnn_fc(save_data_path, n_labels, nb_epoch=100, fine_tune=False,
                      test_only=False, save_pdf=False, vgg=False):

    d = ut.load_pickle(os.path.join(save_data_path, 'isol_data.pkl'))
    nFold = len(d.keys())
    del d

    if vgg: prefix = 'vgg_'
    else: prefix = ''

    scores= []
    y_test_list = []
    y_pred_list = []
    for idx in xrange(nFold):

        # Loading data
        train_data, test_data = km.load_data(idx, save_data_path, viz=False)      
        x_train_sig = train_data[0]
        x_train_img = train_data[1]
        y_train     = train_data[2]
        x_test_sig = test_data[0]
        x_test_img = test_data[1]
        y_test     = test_data[2]

        scaler      = preprocessing.StandardScaler()
        x_train_sig = scaler.fit_transform(x_train_sig)
        x_test_sig  = scaler.transform(x_test_sig)

        ## full_weights_path = os.path.join(save_data_path,'vgg16_weights_'+str(idx)+'.h5')
        ## full_weights_path = os.path.join(save_data_path,'cov_weights_'+str(idx)+'.h5')
        ## full_weights_path = os.path.join(save_data_path,'cnn_fc_weights_'+str(idx)+'.h5')

        # Load pre-trained model
        ## model = km.vgg16_net(np.shape(x_train_img)[1:], n_labels, vgg_model_weights_path, \
        ##                      full_weights_path, fine_tune=False)
        ## model = km.vgg16_net(np.shape(x_train_img)[1:], n_labels, vgg_model_weights_path, \
        ##                      fine_tune=False)
        ## model.save_weights(full_weights_path)--------------------------------------------------
        ## sys.exit()

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
            optimizer = SGD(lr=0.000001, decay=1e-8, momentum=0.9, nesterov=True)
            ## optimizer = RMSprop(lr=0.00001, rho=0.9, epsilon=1e-08, decay=0.0)
            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

                

        if test_only is False:
            train_datagen = km.myGenerator(augmentation=True, rescale=1./255.)
            test_datagen = km.myGenerator(augmentation=False, rescale=1./255.)
            train_generator = train_datagen.flow(x_train_img, x_train_sig, y_train, batch_size=32)
            test_generator = test_datagen.flow(x_test_img, x_test_sig, y_test, batch_size=32)
        
            hist = model.fit_generator(train_generator,
                                       samples_per_epoch=len(y_train),
                                       nb_epoch=50,
                                       validation_data=test_generator,
                                       nb_val_samples=len(y_test))

            full_weights_path = os.path.join(save_data_path,prefix+'cnn_fc_weights_'+str(idx)+'.h5')
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


    if test_only is False:
        print np.mean(scores), np.std(scores)
    else:
        classes = ['Object collision', 'Noisy environment', 'Spoon miss by a user', 'Spoon collision by a user', 'Robot-body collision by a user', 'Aggressive eating', 'Anomalous sound from a user', 'Unreachable mouth pose', 'Face occlusion by a user', 'Spoon miss by system fault', 'Spoon collision by system fault', 'Freeze by system fault']

        print len(np.unique(y_test_list)), np.shape(y_test_list)
        print len(np.unique(y_pred_list)), np.shape(y_pred_list)
        print np.shape(classes),
        
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test_list, y_pred_list)

        iviz.plot_confusion_matrix(cm, classes=classes, normalize=True,
                                   title='Anomaly Isolation', save_pdf=save_pdf)
    
    





if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    util.initialiseOptParser(p)

    p.add_option('--preprocess', '--p', action='store_true', dest='preprocessing',
                 default=False, help='Preprocess')
    p.add_option('--bottom_train', '--bt', action='store_true', dest='bottom_train',
                 default=False, help='Train the bottom layer')
    p.add_option('--top_train', '--tt', action='store_true', dest='top_train',
                 default=False, help='Train the top layer')
    p.add_option('--fine_tune', '--f', action='store_true', dest='fine_tune',
                 default=False, help='Fine tuning')
    p.add_option('--viz', action='store_true', dest='viz',
                 default=False, help='Visualize')
    p.add_option('--viz_model', '--vm', action='store_true', dest='viz_model',
                 default=False, help='Visualize the current model')

    p.add_option('--eval_isol', '--ei', action='store_true', dest='evaluation_isolation',
                 default=False, help='Evaluate anomaly isolation with double detectors.')
    p.add_option('--ai_renew', '--ai', action='store_true', dest='ai_renew',
                 default=False, help='Renew ai')
    p.add_option('--combined_train', '--ct', action='store_true', dest='combined_train',
                 default=False, help='Train all layer')
    
    opt, args = p.parse_args()

    #---------------------------------------------------------------------------           
    # Run evaluation
    #---------------------------------------------------------------------------           
    rf_center     = 'kinEEPos'        
    scale         = 1.0
    local_range   = 10.0
    nPoints = 40 #None

    from hrl_anomaly_detection.isolator.IROS2017_params import *
    raw_data_path, save_data_path, param_dict = getParams(opt.task, opt.bDataRenew, \
                                                          opt.bHMMRenew, opt.bClassifierRenew, opt.dim,\
                                                          rf_center, local_range, nPoints=nPoints)
    if opt.bNoUpdate: param_dict['ROC']['update_list'] = []
    # Mikako - bad camera
    # s1 - kaci - before camera calibration
    subjects = ['s2', 's3','s4','s5', 's6','s7','s8', 's9']

    save_data_path = os.path.expanduser('~')+\
      '/hrl_file_server/dpark_data/anomaly/AURO2016/'+opt.task+'_data_isolation3/'+\
      str(param_dict['data_param']['downSampleSize'])+'_'+str(opt.dim)

    n_labels = 12 #len(np.unique(y_train))
    
    # ---------------------------------------------------------------------
    # 1st pre_train top layer
    if opt.preprocessing:
        preprocess_data(save_data_path, viz=opt.viz)

    elif opt.bottom_train:
        bottom_feature_extraction(save_data_path, n_labels, augmentation=True)
                                 
    elif opt.top_train:
        top_model_train(n_labels)
        ## top_model_train2(save_data_path, n_labels, augmentation=False, nb_epoch=10)

    elif opt.fine_tune:
        fine_tune(save_data_path, n_labels)


        
    elif opt.viz:
        x_train, y_train, x_test, y_test = km.load_data(save_data_path, True)
    elif opt.viz_model:
        model = km.vgg16_net((3,120,160), 12, with_top=True, input_shape2=(14,), viz=True)
        plot(model, to_file='model.png')
        
    else:
        # ep 2,4,8 mean, -> 2,4,8 amin
        ## param_dict['HMM']['scale'] = [7.0, 9.0]
        save_data_path = os.path.expanduser('~')+\
          '/hrl_file_server/dpark_data/anomaly/AURO2016/'+opt.task+'_data_isolation4/'+\
          str(param_dict['data_param']['downSampleSize'])+'_'+str(opt.dim)
        ## preprocess_data(save_data_path, viz=opt.viz, hog_feature=False, org_ratio=True)


        unimodal_fc(save_data_path, n_labels, nb_epoch=1000)        
        unimodal_fc(save_data_path, n_labels, fine_tune=True)        
        ## unimodal_cnn(save_data_path, n_labels)        
        ## unimodal_cnn(save_data_path, n_labels, fine_tune=True)        
        ## multimodal_cnn_fc(save_data_path, n_labels)
        ## multimodal_cnn_fc(save_data_path, n_labels, fine_tune=True)
        ## multimodal_cnn_fc(save_data_path, n_labels, fine_tune=True, test_only=True,
        ##                   save_pdf=opt.bSavePdf)
        ## evaluate_svm(save_data_path)

        ## unimodal_cnn(save_data_path, n_labels, vgg=True)        
        ## unimodal_cnn(save_data_path, n_labels, fine_tune=True, vgg=True)        
        ## multimodal_cnn_fc(save_data_path, n_labels, vgg=True)
        ## multimodal_cnn_fc(save_data_path, n_labels, fine_tune=True, vgg=True)

        


        
        ## n = 102
        ## imgs = None
        ## for i in xrange(n):
        ##     img = np.array([[np.ones((10,10)).tolist(),
        ##                     np.ones((10,10)).tolist(),
        ##                     np.ones((10,10)).tolist()]])+i

        ##     if imgs is None: imgs = img
        ##     else: imgs = np.vstack([imgs, img])

        ## datagen = km.myGenerator(True)
        
        ## count = 0
        ## for x1,x2,y in datagen.flow(imgs.astype(float), imgs.astype(float), range(n), \
        ##                             batch_size=5, shuffle=True):
        ##     print x1[:,0,0,0], x2[:,0,0,0] #, x2.shape
        ##     count +=1
        ##     if count > 99: break


        
        ## save_data_path = os.path.expanduser('~')+\
        ##   '/hrl_file_server/dpark_data/anomaly/AURO2016/'+opt.task+'_data_isolation4/'+\
        ##   str(param_dict['data_param']['downSampleSize'])+'_'+str(opt.dim)
        ## ## preprocess_data(save_data_path, viz=opt.viz, hog_feature=True)
        ## evaluate_svm(save_data_path)

