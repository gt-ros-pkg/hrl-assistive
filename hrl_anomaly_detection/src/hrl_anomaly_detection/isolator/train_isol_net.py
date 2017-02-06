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

random.seed(3334)
np.random.seed(3334)

from sklearn import preprocessing
import h5py
import cv2
import gc


def train_isolator_modules(save_data_path, n_labels, verbose=False):
    '''
    Train networks
    '''
    d = ut.load_pickle(os.path.join(save_data_path, 'isol_data.pkl'))
    nFold = len(d.keys())
    del d

    ## nFold = 1
    fold_list = range(nFold)
    ## fold_list = [4,7]

    save_data_path = os.path.join(save_data_path, 'keras')

    # training with signals ----------------------------------
    ## kt.train_with_signal(save_data_path, n_labels, fold_list, nb_epoch=800, patience=50)
    ## kt.train_with_signal(save_data_path, n_labels, fold_list, nb_epoch=800, patience=50, load_weights=True)
    ## kt.train_with_signal(save_data_path, n_labels, fold_list, nb_epoch=800, patience=5, load_weights=True,
    ##                      test_only=True) #70
    
    # training_with images -----------------------------------
    remove_label = [1]
    ## kt.get_bottleneck_image(save_data_path, n_labels, fold_list, vgg=True, remove_label=remove_label)
    ## kt.train_top_model_with_image(save_data_path, n_labels, fold_list, vgg=True, patience=30)
    ## kt.train_top_model_with_image(save_data_path, n_labels, fold_list, vgg=True, nb_epoch=1000, patience=30,
    ##                               load_weights=True)
    ## kt.train_top_model_with_image(save_data_path, n_labels, fold_list, vgg=True, nb_epoch=1000, load_weights=True,
    ##                               test_only=True)

    fold_list = [6]
    
    # training_with all --------------------------------------
    ## kt.get_bottleneck_mutil(save_data_path, n_labels, fold_list, vgg=True)
    kt.train_multi_top_model(save_data_path, n_labels, fold_list, vgg=True)
    kt.train_multi_top_model(save_data_path, n_labels, fold_list, vgg=True, patience=200, load_weights=True)
    ## kt.train_multi_top_model(save_data_path, n_labels, fold_list, vgg=True, load_weights=True,
    ##                          test_only=True) #74

    # 0.55 0.92 0.93 0.82    0.76 0.76 0.76 0.75
    ## kt.train_with_all(save_data_path, n_labels, fold_list, patience=1, nb_epoch=1, vgg=True)
    ## kt.train_with_all(save_data_path, n_labels, fold_list, load_weights=True, patience=1, vgg=True)
    ## kt.train_with_all(save_data_path, n_labels, fold_list, load_weights=True, patience=5, vgg=True,
    ##                   test_only=True)
    return




def evaluate_svm(save_data_path, viz=False):

    d = ut.load_pickle(os.path.join(save_data_path, 'isol_data.pkl'))
    nFold = len(d.keys())
    del d

    scores = []
    y_test_list = []
    y_pred_list = []
    for idx in xrange(nFold):
        train_data, test_data = autil.load_data(idx, save_data_path, viz=False)      
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


        print np.shape(x_train)

        x_train_dyn1 = x_train[:,:24]
        x_train_dyn2 = x_train[:,24:-7]#[:,:6]
        x_train_stc = x_train[:,-7:][:,[0,1,2,4,5,6]]
        ## x_train_dyn1 -= np.mean(x_train_dyn1, axis=1)[:,np.newaxis]
        ## x_train_dyn2 -= np.mean(x_train_dyn2, axis=1)[:,np.newaxis]
        x_train = np.hstack([x_train_dyn1, x_train_dyn2, x_train_stc])

        x_test_dyn1 = x_test[:,:24]
        x_test_dyn2 = x_test[:,24:-7]#[:,:6]
        x_test_stc = x_test[:,-7:][:,[0,1,2,4,5,6]]
        ## x_test_dyn1 -= np.mean(x_test_dyn1, axis=1)[:,np.newaxis]
        ## x_test_dyn2 -= np.mean(x_test_dyn2, axis=1)[:,np.newaxis]
        x_test = np.hstack([x_test_dyn1, x_test_dyn2, x_test_stc])
        
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

        if viz:
            y_pred = clf.predict(x_test)
            y_pred_list += y_pred.tolist()
            y_test_list += y_test.tolist()
            
        
    print scores
    print np.mean(scores), np.std(scores)
    if viz: plot_confusion_matrix(y_test_list, y_pred_list)


def plot_confusion_matrix(y_test_list, y_pred_list, save_pdf=False):
    classes = ['Object collision', 'Noisy environment', 'Spoon miss by a user', 'Spoon collision by a user', 'Robot-body collision by a user', 'Aggressive eating', 'Anomalous sound from a user', 'Unreachable mouth pose', 'Face occlusion by a user', 'Spoon miss by system fault', 'Spoon collision by system fault', 'Freeze by system fault']


    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test_list, y_pred_list)

    print np.sum(cm,axis=1)

    eviz.plot_confusion_matrix(cm, classes=classes, normalize=True,
                               title='Anomaly Isolation', save_pdf=save_pdf)
    





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
    p.add_option('--viz', action='store_true', dest='viz',
                 default=False, help='Visualize')
    p.add_option('--viz_model', '--vm', action='store_true', dest='viz_model',
                 default=False, help='Visualize the current model')

    
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
                                                          opt.bHMMRenew, opt.bCLFRenew, opt.dim,\
                                                          rf_center, local_range, nPoints=nPoints)
    if opt.bNoUpdate: param_dict['ROC']['update_list'] = []
    # Mikako - bad camera
    # s1 - kaci - before camera calibration
    subjects = ['s2', 's3','s4','s5', 's6','s7','s8', 's9']
    n_labels = 12 #len(np.unique(y_train))

    # 148 amin - nofz        
    save_data_path = os.path.expanduser('~')+\
      '/hrl_file_server/dpark_data/anomaly/AURO2016/'+opt.task+'_data_isolation6/'+\
      str(param_dict['data_param']['downSampleSize'])+'_'+str(opt.dim)

    
    # ---------------------------------------------------------------------
    if opt.preprocessing:
        src_pkl = os.path.join(save_data_path, 'isol_data.pkl')
        pp.preprocess_data(src_pkl, save_data_path, img_scale=0.25, nb_classes=12,
                            img_feature_type='vgg')

    elif opt.preprocessing_extra:
        raw_data_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/AURO2016/raw_data/manual_label'
        pp.preprocess_images(raw_data_path, save_data_path, img_scale=0.25, nb_classes=n_labels,
                                img_feature_type='vgg')

    elif opt.viz:
        x_train, y_train, x_test, y_test = autil.load_data(save_data_path, True)
    elif opt.viz_model:
        model = km.vgg16_net((3,120,160), 12, with_top=True, input_shape2=(14,), viz=True)
        plot(model, to_file='model.png')
        
    elif opt.train:

        train_isolator_modules(save_data_path, n_labels, verbose=False)
        ## evaluate_svm(save_data_path, viz=True)


