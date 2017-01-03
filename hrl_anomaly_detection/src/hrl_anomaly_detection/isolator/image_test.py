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
import numpy as np
import scipy
import hrl_lib.util as ut

# Private utils
from hrl_anomaly_detection.util import *
from hrl_anomaly_detection.util_viz import *
from hrl_anomaly_detection import data_manager as dm
from hrl_anomaly_detection import util as util

# Private learners
from hrl_anomaly_detection.hmm import learning_hmm as hmm
import hrl_anomaly_detection.classifiers.classifier as cf
import hrl_anomaly_detection.data_viz as dv
import hrl_anomaly_detection.isolator.isolation_util as iutil

from joblib import Parallel, delayed

# visualization
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec
import itertools
colors = itertools.cycle(['g', 'm', 'c', 'k', 'y','r', 'b', ])
shapes = itertools.cycle(['x','v', 'o', '+'])

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42 
random.seed(3334)
np.random.seed(3334)

import cv2
SZ=20
bin_n = 16 # Number of bins


def evaluation_all(subject_names, task_name, raw_data_path, processed_data_path, param_dict,\
                   data_renew=False, save_pdf=False, verbose=False, debug=False,\
                   no_plot=False, delay_plot=True, find_param=False, data_gen=False,\
                   weight=-5.0, ai_renew=False):
    # data
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
    # parameters
    startIdx    = 4
    method_list = ROC_dict['methods'] 
    nPoints     = ROC_dict['nPoints']
    
    #------------------------------------------
    if os.path.isdir(processed_data_path) is False:
        os.system('mkdir -p '+processed_data_path)
    crossVal_pkl = os.path.join(processed_data_path, 'cv_'+task_name+'.pkl')
    
    if os.path.isfile(crossVal_pkl) and data_renew is False and data_gen is False:
        print "CV data exists and no renew"
        d = ut.load_pickle(crossVal_pkl)
        kFold_list = d['kFoldList'] 
        success_data = d['successData']
        failure_data = d['failureData']        
        success_files = d['success_files']
        failure_files = d['failure_files']
    else:
        '''
        Use augmented data? if nAugment is 0, then aug_successData = successData
        '''        
        d = dm.getDataLOPO(subject_names, task_name, raw_data_path, \
                           processed_data_path, data_dict['rf_center'], data_dict['local_range'],\
                           downSampleSize=data_dict['downSampleSize'],\
                           handFeatures=param_dict['data_param']['isolationFeatures'], \
                           cut_data=data_dict['cut_data'], \
                           data_renew=data_renew, max_time=data_dict['max_time'], ros_bag_image=True)
        success_data, failure_data, success_files, failure_files, kFold_list \
          = dm.LOPO_data_index(d['successDataList'], d['failureDataList'],\
                               d['successFileList'], d['failureFileList'])

        d['successData'] = success_data
        d['failureData'] = failure_data
        d['success_files']   = success_files
        d['failure_files']   = failure_files
        d['kFoldList']       = kFold_list
        
        ut.save_pickle(d, crossVal_pkl)
        if data_gen: sys.exit()

    # flattening image list
    success_image_list  = d.get('success_image_list',[])
    failure_image_list  = d.get('failure_image_list',[])

    new_list = []
    for i in xrange(len(success_image_list)):
        for j in xrange(len(success_image_list[i])):
            new_list.append(success_image_list[i][j])
    success_image_list = np.array(copy.copy(new_list))

    new_list = []
    for i in xrange(len(failure_image_list)):
        for j in xrange(len(failure_image_list[i])):
            new_list.append(failure_image_list[i][j])
    failure_image_list = np.array(copy.copy(new_list))

    # label
    failure_labels = []
    for f in failure_files:
        failure_labels.append( int( f.split('/')[-1].split('_')[0] ) )
    failure_labels = np.array( failure_labels )

    print "----------------------------------------"

    #-----------------------------------------------------------------------------------------
    # Anomaly Detection
    #-----------------------------------------------------------------------------------------
    # HMM-induced vector with LOPO
    ad_pkl = os.path.join(processed_data_path, 'ad_result.pkl')
    if os.path.isfile(ad_pkl) and HMM_dict['renew'] is False:
        dd = ut.load_pickle(ad_pkl)
        detection_train_idx_list = dd['detection_train_idx_list']
        detection_test_idx_list  = dd['detection_test_idx_list']
    else:
        # select feature for detection
        feature_list = []
        for feature in param_dict['data_param']['handFeatures']:
            idx = [ i for i, x in enumerate(param_dict['data_param']['isolationFeatures']) if feature == x][0]
            feature_list.append(idx)
        
        ## feature_list 
        successData_ad = np.array(success_data)[feature_list]
        failureData_ad = np.array(failure_data)[feature_list]

        dm.saveHMMinducedFeatures(kFold_list, successData_ad, failureData_ad,\
                                  task_name, processed_data_path,\
                                  HMM_dict, data_renew, startIdx, nState, cov, \
                                  success_files=success_files, failure_files=failure_files,\
                                  noise_mag=0.03, verbose=verbose)
        
        detection_train_idx_list = []
        detection_test_idx_list = []
        for idx, (normalTrainIdx, abnormalTrainIdx, normalTestIdx, abnormalTestIdx) \
          in enumerate(kFold_list):

            abnormalTrainData_ad  = copy.copy(failureData_ad[:, abnormalTrainIdx, :])
            abnormalTestData_ad   = copy.copy(failureData_ad[:, abnormalTestIdx, :])

            n_jobs = -1
            detection_train_idxs = iutil.anomaly_detection(abnormalTrainData_ad, \
                                                           [1]*len(abnormalTrainData_ad[0]), \
                                                           task_name, processed_data_path, param_dict,\
                                                           logp_viz=False, verbose=False, \
                                                           weight=weight,\
                                                           idx=idx, n_jobs=n_jobs)
            detection_test_idxs = iutil.anomaly_detection(abnormalTestData_ad, \
                                                          [1]*len(abnormalTestData_ad[0]), \
                                                          task_name, processed_data_path, param_dict,\
                                                          logp_viz=False, verbose=False, \
                                                          weight=weight,\
                                                          idx=idx, n_jobs=n_jobs)
            detection_train_idx_list.append(detection_train_idxs)
            detection_test_idx_list.append(detection_test_idxs)
        dd = {}
        dd['detection_train_idx_list'] = detection_train_idx_list
        dd['detection_test_idx_list'] = detection_test_idx_list
        ut.save_pickle(dd, ad_pkl)

    print "----------------------------------------"

    #-----------------------------------------------------------------------------------------
    # Feature extraction for Anomaly Isolation
    #-----------------------------------------------------------------------------------------
    hog_pkl = os.path.join(processed_data_path, 'hog_data.pkl')
    if os.path.isfile(hog_pkl) and HMM_dict['renew'] is False and ai_renew is False: # and False:
        print "Start to loading"
        ## dd = ut.load_pickle(hog_pkl)
        ## print "Finished to loading"
        ## x_train_list = dd['x_train_list']
        ## y_train_list = dd['y_train_list']
        ## x_test_list  = dd['x_test_list']
        ## y_test_list  = dd['y_test_list']
    else:
        # get hog data first
        x_train_list = []
        y_train_list = []
        x_test_list = []
        y_test_list = []
        for idx, (normalTrainIdx, abnormalTrainIdx, normalTestIdx, abnormalTestIdx) \
          in enumerate(kFold_list):

            # 1 x sample x length
            normalTrainData   = success_image_list[normalTrainIdx]
            abnormalTrainData = failure_image_list[abnormalTrainIdx]
            normalTestData    = success_image_list[normalTestIdx]
            abnormalTestData  = failure_image_list[abnormalTestIdx]

            abnormal_train_labels = [failure_labels[i] for i in abnormalTrainIdx]
            abnormal_test_labels  = [failure_labels[i] for i in abnormalTestIdx]

            #-----------------------------------------------------------------------------------------
            # Feature Extraction
            #-----------------------------------------------------------------------------------------
            # load images and labels
            n_jobs = -1
            l_data = Parallel(n_jobs=n_jobs, verbose=10)\
              (delayed(get_hog_data)(idx, files, abnormal_train_labels[i] ) \
               for i, files in enumerate(abnormalTrainData))

            x_train = []
            y_train = []
            for i in xrange(len(l_data)):
                x_train.append( l_data[i][0] )
                y_train.append( l_data[i][1] )

            l_data = Parallel(n_jobs=n_jobs, verbose=10)\
              (delayed(get_hog_data)(idx, files, abnormal_test_labels[i] ) \
               for i, files in enumerate(abnormalTestData))

            x_test = []
            y_test = []
            for i in xrange(len(l_data)):
                x_test.append( l_data[i][0] )
                y_test.append( l_data[i][1] )

            x_train_list.append(x_train)
            y_train_list.append(y_train)
            x_test_list.append(x_test)
            y_test_list.append(y_test)
            break

        dd = {}
        dd['x_train_list'] = x_train_list
        dd['y_train_list'] = y_train_list
        dd['x_test_list'] = x_test_list
        dd['y_test_list'] = y_test_list
        ut.save_pickle(dd, hog_pkl)
    
    scores = []
    # Select feature and 
    test_data_pkl = os.path.join(processed_data_path, 'test_data.pkl')
    for idx, (normalTrainIdx, abnormalTrainIdx, normalTestIdx, abnormalTestIdx) in enumerate(kFold_list):

        if os.path.isfile(test_data_pkl):
            dd = ut.load_pickle(test_data_pkl)
            x_train = dd['x_train']
            y_train = dd['y_train']
            x_test  = dd['x_test']
            y_test  = dd['y_test']
        else:
            window_step = 5
            print np.shape(x_train_list), np.shape(detection_train_idx_list)
            print np.shape(x_test_list), np.shape(detection_test_idx_list)
            assert len(x_train_list[idx]) == len(detection_train_idx_list[idx])

            # feature extraction
            x_train = []
            y_train = []
            for i, x in enumerate(x_train_list[idx]):
                d_idx = detection_train_idx_list[idx][i]
                if d_idx is None: continue

                for j in range(-window_step, window_step):
                    if d_idx+1+j <= 4: continue
                    if d_idx+1+j >= len(x_train_list[idx][i]): continue

                    ## print y_train_list[idx][i][d_idx+1+j]
                    ## cv2.imshow('image',x_train_list[idx][i][d_idx+1+j])
                    ## cv2.waitKey(0)
                    ## cv2.destroyAllWindows()        
                    ## sys.exit()

                    x_train.append( x_train_list[idx][i][d_idx+1+j] )
                    y_train.append( y_train_list[idx][i][d_idx+1+j] )


            x_test = []
            y_test = []
            for i, x in enumerate(x_test_list[idx]):
                d_idx = detection_test_idx_list[idx][i]
                if d_idx is None: continue
                if d_idx+1 >= len(x_test_list[idx][i]): d_idx -= 1
                if x_test_list[idx][i] == []: continue
                x_test.append( x_test_list[idx][i][d_idx+1] )
                y_test.append( y_test_list[idx][i][d_idx+1] )

            dd = {}
            dd['x_train'] = x_train
            dd['y_train'] = y_train
            dd['x_test'] = x_test
            dd['y_test'] = y_test
            ut.save_pickle(dd, test_data_pkl)
        print "---------------------------------------- aaaaaaaaaaaaaaaaa"
        ## #temp
        ## sys.exit()

        # extract hog
        x_train_hog = []
        for x in x_train: x_train_hog.append( hog(x) )
        x_test_hog = []
        for x in x_test: x_test_hog.append( hog(x) )
        x_train = x_train_hog
        x_test = x_test_hog

        print np.shape(x_train), np.shape(y_train)
        print np.shape(x_test), np.shape(y_test)


        # normalization
        from sklearn import preprocessing
        scaler = preprocessing.StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test  = scaler.transform(x_test)
        ## iutil.save_data_labels(x_train, y_train)
        
        # train svm
        from sklearn.svm import SVC
        clf = SVC(C=1.0, kernel='linear', gamma=1e-1) #, decision_function_shape='ovo')
        clf.fit(x_train, y_train)
        
        # classify and get scores
        score = clf.score(x_test, y_test)
        scores.append(score)
        print "score: ", score
        sys.exit()

    print scores
    print np.mean(scores), np.std(scores)


def get_hog_data(idx, files, label, augmentation=True):
    if files is None: return [],[]

    from skimage.transform import rescale
    ## from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

    ## datagen = ImageDataGenerator(
    ##     rotation_range=40,
    ##     width_shift_range=0.2,
    ##     height_shift_range=0.2,
    ##     shear_range=0.2,
    ##     zoom_range=0.2,
    ##     horizontal_flip=True,
    ##     fill_mode='nearest')

    images = []
    for f in files:
        img = cv2.imread(f) # 480*640*3
        height, width = img.shape[:2]
        ## sys.exit()
        ## l = 224
        ## crop_img = img[height/2-l:height/2+l, width/2-l:width/2+l ]        
        ## img = cv2.resize(crop_img,(l, l), interpolation = cv2.INTER_CUBIC)
        img = cv2.resize(img,(width/4, height/4), interpolation = cv2.INTER_CUBIC)
        ## img = rescale(img, 0.25)
        ## print np.shape(img)
        ## images.append(hog(img))
        images.append(img)
    return images, [label]*len(images)


def hog(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     # hist is a 64 bit vector
    return hist



if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    util.initialiseOptParser(p)

    p.add_option('--eval_isol', '--ei', action='store_true', dest='evaluation_isolation',
                 default=False, help='Evaluate anomaly isolation with double detectors.')
    p.add_option('--ai_renew', '--ai', action='store_true', dest='ai_renew',
                 default=False, help='Renew ai')
    
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


    if opt.bEvaluationAll:
        '''
        feature-wise evaluation
        '''        
        save_data_path = os.path.expanduser('~')+\
          '/hrl_file_server/dpark_data/anomaly/AURO2016/'+opt.task+'_data_isolation3/'+\
          str(param_dict['data_param']['downSampleSize'])+'_'+str(opt.dim)

        # 87.95
        param_dict['data_param']['handFeatures'] = ['unimodal_audioWristRMS',  \
                                                    'unimodal_kinJntEff_1',\
                                                    'unimodal_ftForce_integ',\
                                                    'unimodal_kinEEChange',\
                                                    'crossmodal_landmarkEEDist', \
                                                    ]

        param_dict['ROC']['methods'] = ['hmmgp']
        param_dict['HMM']['scale'] = 7.11
        param_dict['ROC']['hmmgp_param_range'] = np.logspace(-0.6, 2.5, nPoints)*-1.0
        weight = -20.0

        
        if opt.bNoUpdate: param_dict['ROC']['update_list'] = []        
        evaluation_all(subjects, opt.task, raw_data_path, save_data_path, param_dict, save_pdf=opt.bSavePdf, \
                       verbose=opt.bVerbose, debug=opt.bDebug, no_plot=opt.bNoPlot, \
                       find_param=False, data_gen=opt.bDataGen, weight=weight, ai_renew=opt.ai_renew)
