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
from hrl_execution_monitor.keras_util import keras_model as km
from hrl_execution_monitor.keras_util import keras_util as kutil
from hrl_execution_monitor.keras_util import train_isol_net as kt
from hrl_execution_monitor import util as autil
from hrl_execution_monitor import viz as eviz
from hrl_execution_monitor import preprocess as pp
from joblib import Parallel, delayed

from keras.optimizers import SGD, Adagrad, Adadelta, RMSprop
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn import preprocessing
from sklearn.metrics import accuracy_score


random.seed(3334)
np.random.seed(3334)

import h5py, cv2, gc

def test(save_data_path, n_labels=12, n_folds=8, verbose=False):

    ## d = ut.load_pickle(os.path.join(save_data_path, 'isol_data.pkl'))
    ## print d.keys()
    ## nFold = len(d.keys())
    ## del d
    fold_list = range(nFold)

    #temp
    fold_list = [2,3,4,5,6,7]

    save_data_path = os.path.join(save_data_path, 'keras')

    ## kt.get_bottleneck_image(save_data_path, n_labels, fold_list, vgg=True)
    train_top_model_with_image(save_data_path, n_labels, fold_list, nb_epoch=1000, patience=50)
    train_top_model_with_image(save_data_path, n_labels, fold_list, nb_epoch=1000,
                               patience=100, load_weights=True)
    train_top_model_with_image(save_data_path, n_labels, fold_list, 
                               load_weights=True, test_only=True)
    

def get_model():
    model = km.vgg16_net()
    model = load_model_weights(model, "vgg16_weights.h5")

    model.add(Lambda(global_average_pooling,
                     output_shape=global_average_pooling_shape))
    model.add(Dense(2, activation = 'softmax', init='uniform'))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.5, nesterov=True)
    model.compile(loss = 'categorical_crossentropy', \
                  optimizer = sgd, metrics=['accuracy'])
    return model



def train_top_model_with_image(save_data_path, n_labels, fold_list, nb_epoch=400, load_weights=False,
                               vgg=True,\
                               patience=5, remove_label=[], use_extra_img=True, test_only=False,\
                               cause_class=True):
    prefix = 'vgg_'

    scores= []
    y_pred_list = []
    y_test_list = []
    for idx in fold_list:

        bt_data_path = os.path.join(save_data_path, 'bt')
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
            model = km.vgg_image_top_net(np.shape(x_train)[1:], n_labels)
            model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        else:
            print "Load weight!!!!!!!!!!!!!!!"
            model = km.vgg_image_top_net(np.shape(x_train)[1:], n_labels, weights_path)
            optimizer = RMSprop(lr=0.000001, rho=0.9, epsilon=1e-08, decay=0.001)
            #optimizer = SGD(lr=0.001, decay=1e-7, momentum=0.9, nesterov=True)                
            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        
        class_weight={}
        for i in xrange(n_labels): class_weight[i] = 1.0
            
        if test_only is False:
            hist = model.fit(x_train, y_train, nb_epoch=nb_epoch, batch_size=4096, shuffle=True,
                             validation_data=(x_test, y_test), callbacks=callbacks,
                             class_weight=class_weight)

            scores.append( hist.history['val_acc'][-1] )
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

    cause_class = False
    
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
        test(save_data_path)
