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

# system
import os, sys, copy
import random
import warnings

import matplotlib
## matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec

# util
import numpy as np
import scipy
import hrl_lib.util as ut
import hrl_lib.quaternion as qt
from hrl_anomaly_detection import util

from mvpa2.datasets.base import Dataset
from mvpa2.generators.partition import NFoldPartitioner
from mvpa2.generators import splitters

from sklearn import cross_validation
from sklearn.externals import joblib

import matplotlib.pyplot as plt

def create_mvpa_dataset(aXData, chunks, labels):
    data = Dataset(samples=aXData)
    data.sa['id']      = range(0,len(labels))
    data.sa['chunks']  = chunks
    data.sa['targets'] = labels

    return data

def kFold_data_index(nNormal, nAbnormal, nNormalFold, nAbnormalFold ):
    """
    Output:
    Normal training data 
    Abnormal training data 
    Normal test data 
    Abnormal test data 
    """

    normal_folds   = cross_validation.KFold(nNormal, n_folds=nNormalFold, shuffle=True)
    abnormal_folds = cross_validation.KFold(nAbnormal, n_folds=nAbnormalFold, shuffle=True)

    kFold_list = []

    for normal_train_fold, normal_test_fold in normal_folds:

        for abnormal_train_fold, abnormal_test_fold in abnormal_folds:
            index_list = [normal_train_fold, abnormal_train_fold, \
                          normal_test_fold, abnormal_test_fold]
            kFold_list.append(index_list)

    return kFold_list

def LOPO_data_index(success_data_list, failure_data_list, \
                    success_file_list, failure_file_list, many_to_one=True,\
                    target_class=None):
    """
    Return completed set of success and failure data with LOPO cross-validatation fold list
    """
    nSubject = len(success_data_list)
    successIdx = []
    failureIdx = []
    success_files = []
    failure_files = []
    for i in xrange(nSubject):

        if i == 0:
            success_data = success_data_list[i]
            failure_data = failure_data_list[i]
            successIdx.append( range(len(success_data_list[i][0])) )
            failureIdx.append( range(len(failure_data_list[i][0])) )
        else:
            success_data = np.vstack([ np.swapaxes(success_data,0,1), \
                                      np.swapaxes(success_data_list[i], 0,1)])
            failure_data = np.vstack([ np.swapaxes(failure_data,0,1), \
                                      np.swapaxes(failure_data_list[i], 0,1)])
            success_data = np.swapaxes(success_data, 0, 1)
            failure_data = np.swapaxes(failure_data, 0, 1)
            successIdx.append( range(successIdx[-1][-1]+1, successIdx[-1][-1]+1+\
                                     len(success_data_list[i][0])) )
            failureIdx.append( range(failureIdx[-1][-1]+1, failureIdx[-1][-1]+1+\
                                     len(failure_data_list[i][0])) )

        success_files += success_file_list[i]
        failure_files += failure_file_list[i]


    # Select specific anomalies
    if target_class is not None:
        target_idx = []
        for i, f in enumerate(failure_files):
            if int(f.split('/')[-1].split('_')[0]) in target_class:
                target_idx.append(i)

        failure_data = failure_data[:,target_idx,:]
        failure_files = [failure_files[i] for i in target_idx]

        failureIdx = []
        for i in xrange(nSubject):

            target_idx = []
            for k, f in enumerate(failure_file_list[i]):
                if int(f.split('/')[-1].split('_')[0]) in target_class:
                    target_idx.append(k)

            if i == 0:
                failureIdx.append( range(len(target_idx)) )
            else:
                failureIdx.append( range(failureIdx[-1][-1]+1, failureIdx[-1][-1]+1+\
                                         len(target_idx)) )


    # only for hmm tuning
    kFold_list = []
    # leave-one-person-out
    for idx in xrange(nSubject):
        idx_list = range(nSubject)
        train_idx = idx_list[:idx]+idx_list[idx+1:]
        test_idx  = idx_list[idx:idx+1]        

        normalTrainIdx = []
        abnormalTrainIdx = []
        for tidx in train_idx:
            if many_to_one:
                normalTrainIdx   += successIdx[tidx]
                abnormalTrainIdx += failureIdx[tidx]
            else:                
                normalTrainIdx   = successIdx[tidx]
                abnormalTrainIdx = failureIdx[tidx]
                normalTestIdx    = successIdx[test_idx[0]]
                abnormalTestIdx  = failureIdx[test_idx[0]]
                kFold_list.append([ normalTrainIdx, abnormalTrainIdx, normalTestIdx, abnormalTestIdx])

        if many_to_one:
            normalTestIdx = []
            abnormalTestIdx = []
            for tidx in test_idx:
                normalTestIdx   += successIdx[tidx]
                abnormalTestIdx += failureIdx[tidx]

            kFold_list.append([ normalTrainIdx, abnormalTrainIdx, normalTestIdx, abnormalTestIdx])

    return success_data, failure_data, success_files, failure_files, kFold_list


def rnd_fold_index(nNormal, nAbnormal, train_ratio=0.8, nSet=1):
    """
    Return completed set of success and failure data with random fold list
    """

    kFold_list = []
    for i in xrange(nSet):
        # divide into training and param estimation set
        nor_train_idx = random.sample(range(nNormal), int( train_ratio*nNormal ) )
        nor_test_idx  = [x for x in range(nNormal) if not x in nor_train_idx] 
        
        abr_train_idx = random.sample(range(nAbnormal), int( train_ratio*nAbnormal ) )
        abr_test_idx  = [x for x in range(nAbnormal) if not x in abr_train_idx] 
        
        index_list = [nor_train_idx, abr_train_idx, nor_test_idx, abr_test_idx]
        kFold_list.append(index_list)

    return kFold_list


#-------------------------------------------------------------------------------------------------
def getDataList(fileNames, rf_center, local_range, param_dict, downSampleSize=200, \
                cut_data=None, \
                handFeatures=['crossmodal_targetEEDist'], \
                renew_minmax=False):
    '''
    Load a list of files and return hand-engineered feature set of each file.
    The feature is scaled by the param dict.
    '''
    
    for fileName in fileNames:
        if os.path.isfile(fileName) is False:
            print "Error>> there is no recorded file: ", fileName
            sys.exit()

    max_time = param_dict['timeList'][-1]
    print "max time is ", max_time
    
    _, data_dict = util.loadData(fileNames, isTrainingData=False,
                                 downSampleSize=downSampleSize,
                                 local_range=local_range, rf_center=rf_center, max_time=max_time)
   
    features, _ = extractHandFeature(data_dict, handFeatures, \
                                     init_param_dict=param_dict, cut_data=cut_data,\
                                     renew_minmax=renew_minmax)

    return features, data_dict



def getDataSet(subject_names, task_name, raw_data_path, processed_data_path,
               rf_center='kinEEPos', local_range=10.0, downSampleSize=200, \
               cut_data=None, init_param_dict=None, \
               success_viz=False, failure_viz=False, \
               save_pdf=False, solid_color=True, \
               handFeatures=[], data_renew=False,\
               time_sort=False, max_time=None, ros_bag_image=False, rndFold=False,
               verbose=False):
    '''
    '''

    if os.path.isdir(processed_data_path) is False:
        os.system('mkdir -p '+processed_data_path)

    save_pkl = os.path.join(processed_data_path, 'feature_extraction_'+rf_center+'_'+\
                            str(local_range) )
            
    if os.path.isfile(save_pkl) and data_renew is False:
        print "--------------------------------------"
        print "Load saved data"
        print "--------------------------------------"
        data_dict = ut.load_pickle(save_pkl)
        
        # Task-oriented hand-crafted features
        if 'successData' not in data_dict.keys():
            successDataList = data_dict['successDataList'] 
            failureDataList = data_dict['failureDataList']

            for i in xrange(len(successDataList)):
                if i == 0:
                    successData = successDataList[i]
                    failureData = failureDataList[i]
                else:
                    successData = np.vstack([ np.swapaxes(successData,0,1), \
                                              np.swapaxes(successDataList[i], 0,1)])
                    failureData = np.vstack([ np.swapaxes(failureData,0,1), \
                                              np.swapaxes(failureDataList[i], 0,1)])
                    successData = np.swapaxes(successData, 0, 1)
                    failureData = np.swapaxes(failureData, 0, 1)
            data_dict['successData'] = successData
            data_dict['failureData'] = failureData
        else:
            successData = data_dict['successData'] 
            failureData = data_dict['failureData']

        param_dict      = data_dict['param_dict']

    else:
        success_list, failure_list = util.getSubjectFileList(raw_data_path, subject_names, task_name,\
                                                             time_sort=time_sort)

        print "start to load data"
        all_data_pkl     = os.path.join(processed_data_path, task_name+'_all_'+rf_center+\
                                        '_'+str(local_range))
        _, all_data_dict = util.loadData(success_list+failure_list, isTrainingData=False,
                                         downSampleSize=downSampleSize,\
                                         renew=data_renew, save_pkl=all_data_pkl,\
                                         max_time=max_time)
        max_time = all_data_dict['timesList'][0][-1]
        if verbose: print "max time is ", max_time

        # data set
        _, success_data_dict = util.loadData(success_list, isTrainingData=True,
                                             downSampleSize=downSampleSize,\
                                             renew=data_renew,\
                                             max_time=max_time)

        _, failure_data_dict = util.loadData(failure_list, isTrainingData=False,
                                             downSampleSize=downSampleSize,\
                                             renew=data_renew,\
                                             max_time=max_time)

        # Task-oriented hand-crafted features
        if init_param_dict is not None:
            allData, _ = extractHandFeature(all_data_dict, handFeatures,\
                                            init_param_dict=init_param_dict, cut_data=cut_data)
            param_dict=init_param_dict                                            
        else:
            allData, param_dict = extractHandFeature(all_data_dict, handFeatures,\
                                                     cut_data=cut_data)
        print " --------------------- Success -----------------------------"  
        successData, _      = extractHandFeature(success_data_dict, handFeatures, \
                                                 init_param_dict=param_dict, cut_data=cut_data)
        print " --------------------- Failure -----------------------------"  
        failureData, _      = extractHandFeature(failure_data_dict, handFeatures, \
                                                 init_param_dict=param_dict, cut_data=cut_data)

        success_image_list = []
        failure_image_list = []
        if ros_bag_image:
            new_success_list = []
            for f in success_list:
                root_dir = os.path.split(f)[0]+'_rosbag'
                sub_dir  = os.path.split(f)[1].split('.pkl')[0]
                new_success_list.append( os.path.join(root_dir, sub_dir) )
            new_failure_list = []
            for f in failure_list:
                root_dir = os.path.split(f)[0]+'_rosbag'
                sub_dir  = os.path.split(f)[1].split('.pkl')[0]
                new_failure_list.append( os.path.join(root_dir, sub_dir) )

            success_image_list.append(export_images(new_success_list, success_data_dict, \
                                                    downSampleSize) )
            failure_image_list.append(export_images(new_failure_list, failure_data_dict, \
                                                    downSampleSize) )


        data_dict = {}
        data_dict['allData']      = allData = np.array(allData)
        data_dict['successData']  = successData = np.array(successData)
        data_dict['failureData']  = failureData = np.array(failureData)
        data_dict['successFiles'] = success_list
        data_dict['failureFiles'] = failure_list
        data_dict['success_image_list'] = success_image_list
        data_dict['failure_image_list'] = failure_image_list
        data_dict['param_dict'] = param_dict


        if rndFold:
            # split data with 80:20 ratio
            kFold_list = rnd_fold_index(len(successData[0]), len(failureData[0]), \
                                        train_ratio=0.8, nSet=3 )
            data_dict['kFold_list'] = kFold_list
        
        ut.save_pickle(data_dict, save_pkl)

    #-----------------------------------------------------------------------------
    ## All data
    nPlot = None

    # almost deprecated??
    feature_names = np.array(param_dict.get('feature_names', handFeatures))
    AddFeature_names    = feature_names

    # -------------------- Display ---------------------
    fig = None
    if success_viz:

        fig = plt.figure()
        n,m,k = np.shape(successData)
        nPlot = n

        for i in xrange(n):
            ## ax = fig.add_subplot((nPlot/2)*100+20+i)
            ax = fig.add_subplot(n*100+10+i)
            if solid_color: ax.plot(successData[i].T, c='b')
            else: ax.plot(successData[i].T)

            print AddFeature_names[i]
            if AddFeature_names[i] == 'ftForce_mag': ax.set_ylabel('Force Magnitude (N)')
            elif AddFeature_names[i] == 'artagEEDist': ax.set_ylabel('Relative Distance (m)')
            elif AddFeature_names[i] == 'audioWristRMS': ax.set_ylabel('Sound Energy')
            else: ax.set_ylabel(AddFeature_names[i])
                ## ax.set_title( AddFeature_names[i] )

    if failure_viz:
        if fig is None: fig = plt.figure()
        n,m,k = np.shape(failureData)
        nPlot = n

        for i in xrange(n):
            ax = fig.add_subplot(n*100+10+i)
            if solid_color: ax.plot(failureData[i].T, c='r')
            else: ax.plot(failureData[i].T)
            ax.set_title( AddFeature_names[i] )

    if success_viz or failure_viz:
        plt.tight_layout(pad=3.0, w_pad=0.5, h_pad=0.5)
        for i in xrange(n):
            ax = fig.add_subplot(n*100+10+i)

        if save_pdf:
            fig.savefig('test.pdf')
            fig.savefig('test.png')
            os.system('cp test.p* ~/Dropbox/HRL/')        
        else:
            plt.show()

    print "---------------------------------------------------"
    print "s/f data: ", np.shape(successData), np.shape(failureData)
    print "---------------------------------------------------"
    return data_dict


def getDataLOPO(subject_names, task_name, raw_data_path, processed_data_path,
                rf_center='kinEEPos', local_range=10.0, downSampleSize=200, \
                cut_data=None, init_param_dict=None, \
                success_viz=False, failure_viz=False, \
                save_pdf=False, solid_color=True, \
                handFeatures=[], data_renew=False,\
                time_sort=False, max_time=None, \
                target_class=None, ros_bag_image=False):
    """
    Get data per subject. It also returns leave-one-out cross-validataion indices.
    """

    if os.path.isdir(processed_data_path) is False:
        os.system('mkdir -p '+processed_data_path)

    save_pkl = os.path.join(processed_data_path, 'feature_extraction_'+rf_center+'_'+\
                            str(local_range) )
            
    if os.path.isfile(save_pkl) and data_renew is False:
        print "--------------------------------------"
        print "Load saved data"
        print "--------------------------------------"
        data_dict = ut.load_pickle(save_pkl)
        # Task-oriented hand-crafted features
        successDataList = data_dict['successDataList']
        failureDataList = data_dict['failureDataList']
        param_dict      = data_dict['param_dict']
        successFileList     = data_dict.get('successFileList',[])
        failureFileList     = data_dict.get('failureFileList',[])
        success_image_list  = data_dict.get('success_image_list',[])
        failure_image_list  = data_dict.get('failure_image_list',[])

    else:
        file_list = util.getSubjectFileList(raw_data_path, subject_names, task_name,\
                                            time_sort=time_sort, no_split=True)

        print "start to load data"
        # loading and time-sync    
        all_data_pkl     = os.path.join(processed_data_path, task_name+'_all_'+rf_center+\
                                        '_'+str(local_range))
        _, all_data_dict = util.loadData(file_list, isTrainingData=False,
                                         downSampleSize=downSampleSize,\
                                         renew=data_renew, save_pkl=all_data_pkl,\
                                         max_time=max_time)
        max_time = all_data_dict['timesList'][0][-1]
        print "max time is ", max_time

        # Task-oriented hand-crafted features
        if init_param_dict is not None:
            _, _ = extractHandFeature(all_data_dict, handFeatures,\
                                            init_param_dict=init_param_dict, cut_data=cut_data)
            param_dict=init_param_dict                                            
        else:
            _, param_dict = extractHandFeature(all_data_dict, handFeatures,\
                                                     cut_data=cut_data)

        # leave-one-person-out
        successDataList = []
        failureDataList = []
        successFileList = []
        failureFileList = []
        success_image_list = []
        failure_image_list = []
        for i in xrange(len(subject_names)):

            success_list, failure_list = util.getSubjectFileList(raw_data_path, [subject_names[i]], \
                                                                 task_name,\
                                                                 time_sort=time_sort)

            _, success_data_dict = util.loadData(success_list, isTrainingData=True,
                                                 downSampleSize=downSampleSize,\
                                                 renew=data_renew,\
                                                 max_time=max_time)

            _, failure_data_dict = util.loadData(failure_list, isTrainingData=False,
                                                 downSampleSize=downSampleSize,\
                                                 renew=data_renew,\
                                                 max_time=max_time)

            # Get data
            if len(handFeatures) > 0:

                print " --------------------- Success -----------------------------"  
                successData, _      = extractHandFeature(success_data_dict, handFeatures, \
                                                         init_param_dict=param_dict, cut_data=cut_data)
                print " --------------------- Failure -----------------------------"  
                failureData, _      = extractHandFeature(failure_data_dict, handFeatures, \
                                                         init_param_dict=param_dict, cut_data=cut_data)
                successDataList.append(successData)
                failureDataList.append(failureData)

            
            successFileList.append(success_list)
            failureFileList.append(failure_list)

            if ros_bag_image:
                new_success_list = []
                for f in success_list:
                    root_dir = os.path.split(f)[0]+'_rosbag'
                    sub_dir  = os.path.split(f)[1].split('.pkl')[0]
                    new_success_list.append( os.path.join(root_dir, sub_dir) )
                new_failure_list = []
                for f in failure_list:
                    root_dir = os.path.split(f)[0]+'_rosbag'
                    sub_dir  = os.path.split(f)[1].split('.pkl')[0]
                    new_failure_list.append( os.path.join(root_dir, sub_dir) )
                    
                success_image_list.append(export_images(new_success_list, success_data_dict, \
                                                        downSampleSize) )
                failure_image_list.append(export_images(new_failure_list, failure_data_dict, \
                                                        downSampleSize) )
            

        data_dict = {}        
        data_dict['successDataList'] = successDataList
        data_dict['failureDataList'] = failureDataList
        data_dict['param_dict']      = param_dict
        data_dict['successFileList'] = successFileList
        data_dict['failureFileList'] = failureFileList        
        data_dict['success_image_list'] = success_image_list
        data_dict['failure_image_list'] = failure_image_list
        
        ut.save_pickle(data_dict, save_pkl)

    #-----------------------------------------------------------------------------
    ## All data
    nPlot = None

    # almost deprecated??
    AddFeature_names = np.array(param_dict.get('feature_names', handFeatures))

    # -------------------- Display ---------------------
    
    fig = None
    if success_viz:

        import itertools
        colors = itertools.cycle(['g', 'm', 'c', 'k', 'y','r', 'b', ])
        shapes = itertools.cycle(['x','v', 'o', '+'])

        fig = plt.figure()        

        for successData in successDataList:
            n,m,k = np.shape(successData)
            nPlot = n
            color = colors.next()

            print "successData shape: ", n,m,k

            for i in xrange(n):
                if n>9:
                    ax = fig.add_subplot(nPlot/2,2,i+1)
                else:
                    ax = fig.add_subplot(n*100+10+i+1)
                if solid_color: ax.plot(successData[i].T, c='b')
                else: ax.plot(successData[i].T)

                if AddFeature_names[i] == 'ftForce_mag': ax.set_ylabel('Force Magnitude (N)')
                elif AddFeature_names[i] == 'artagEEDist': ax.set_ylabel('Relative Distance (m)')
                elif AddFeature_names[i] == 'audioWristRMS': ax.set_ylabel('Sound Energy')
                else: ax.set_ylabel(AddFeature_names[i])


    if failure_viz:
        if fig is None: fig = plt.figure()

        for fidx, failureData in enumerate(failureDataList):
            if len(failureData)==0: break
            n,m,k = np.shape(failureData)            
            nPlot = n

            failure_data = None
            if target_class is not None:
                for lidx, l in enumerate(failureFileList[fidx]):
                    if int(l.split('/')[-1].split('_')[0]) in target_class:
                        if failure_data is None:
                            failure_data = copy.copy(np.array(failureData)[:,lidx:lidx+1,:])
                        else:
                            failure_data = np.vstack([failure_data, np.array(failureData)[:,lidx:lidx+1,:] ])
            else:
                failure_data = failureData

            for i in xrange(n): # per feature                
                if n>9:
                    ax = fig.add_subplot(nPlot/2,2,i+1)
                else:
                    ax = fig.add_subplot(n*100+10+i+1)
                if solid_color: ax.plot(failure_data[i].T, c='r')
                else: ax.plot(failure_data[i].T)
                ax.set_title( AddFeature_names[i] )


    if success_viz or failure_viz:
        if save_pdf:
            fig.savefig('test.pdf')
            fig.savefig('test.png')
            os.system('cp test.p* ~/Dropbox/HRL/')        
        else:
            plt.show()

    print "---------------------------------------------------"
    print "s/f data: ", np.shape(successDataList), np.shape(failureDataList)
    print "---------------------------------------------------"
    return data_dict


def getAEdataSet(idx, rawSuccessData, rawFailureData, handSuccessData, handFailureData, handParam, \
                 normalTrainIdx, abnormalTrainIdx, normalTestIdx, abnormalTestIdx,
                 time_window, nAugment, \
                 AE_proc_data, \
                 # data param
                 processed_data_path, \
                 # AE param
                 layer_sizes=[256,128,16], learning_rate=1e-6, learning_rate_decay=1e-6, \
                 momentum=1e-6, dampening=1e-6, lambda_reg=1e-6, \
                 max_iteration=20000, min_loss=1.0, cuda=False, \
                 filtering=True, filteringDim=4, method='ae',\
                 # PCA param
                 pca_gamma=5.0,\
                 verbose=False, renew=False, preTrainModel=None ):

    ## if os.path.isfile(AE_proc_data) and not renew:        
    ##     d = ut.load_pickle(AE_proc_data)
    ##     ## d['handFeatureNames'] = handParam['feature_names']
    ##     ## ut.save_pickle(d, AE_proc_data)
    ##     return d

    # dim x sample x length
    normalTrainData   = rawSuccessData[:, normalTrainIdx, :] 
    abnormalTrainData = rawFailureData[:, abnormalTrainIdx, :] 
    normalTestData    = rawSuccessData[:, normalTestIdx, :] 
    abnormalTestData  = rawFailureData[:, abnormalTestIdx, :]

    # sample x dim x length
    normalTrainData   = np.swapaxes(normalTrainData, 0, 1)
    abnormalTrainData = np.swapaxes(abnormalTrainData, 0, 1)
    normalTestData    = np.swapaxes(normalTestData, 0, 1)
    abnormalTestData  = np.swapaxes(abnormalTestData, 0, 1)

    # data augmentation for auto encoder
    if nAugment>0:
        normalTrainDataAug, abnormalTrainDataAug = data_augmentation(normalTrainData, \
                                                                     abnormalTrainData, nAugment)
    else:
        normalTrainDataAug   = normalTrainData
        abnormalTrainDataAug = abnormalTrainData

    # sample x time_window_flatten_length
    normalTrainDataAugConv   = getTimeDelayData(normalTrainDataAug, time_window)
    abnormalTrainDataAugConv = getTimeDelayData(abnormalTrainDataAug, time_window)
    normalTrainDataConv      = getTimeDelayData(normalTrainData, time_window)
    abnormalTrainDataConv    = getTimeDelayData(abnormalTrainData, time_window)
    normalTestDataConv       = getTimeDelayData(normalTestData, time_window)
    abnormalTestDataConv     = getTimeDelayData(abnormalTestData, time_window)
    nSingleData              = len(normalTrainDataAug[0][0])-time_window+1
    nDim                     = len(normalTrainDataConv[1])

    # sample x time_window_flatten_length
    if nAugment>0:
        X_train  = np.vstack([normalTrainDataAugConv, abnormalTrainDataAugConv])
    else:
        X_train  = np.vstack([normalTrainDataConv, abnormalTrainDataConv])        

    # train ae
    if method == 'ae':
        print "Loading ae_model data"
        from hrl_anomaly_detection.feature_extractors import auto_encoder as ae
        ml = ae.auto_encoder([nDim]+layer_sizes, \
                             learning_rate, learning_rate_decay, momentum, dampening, \
                             lambda_reg, time_window, \
                             max_iteration=max_iteration, min_loss=min_loss, cuda=cuda, verbose=True)

        AE_model = os.path.join(processed_data_path, 'ae_model_'+str(idx)+'.pkl')
        if os.path.isfile(AE_model):
            print "AE model exists: ", AE_model
            ## ml.load_params(AE_model)
            ml.create_layers(load=True, filename=AE_model)
        else:
            if preTrainModel is not None:
                ml.fit(X_train, save_obs={'save': False, 'load': True, 'filename': preTrainModel})
            else:
                ml.fit(X_train)
            ml.save_params(AE_model)

        def predictFeatures(clf, X, nSingleData):
            # Generate training features
            feature_list = []
            for idx in xrange(0, len(X), nSingleData):
                test_features = clf.predict_features( X[idx:idx+nSingleData,:].astype('float32') )
                feature_list.append(test_features)
            return feature_list

        # test ae
        # sample x dim => dim x sample
        d = {}
        d['normTrainData']   = np.swapaxes(predictFeatures(ml, normalTrainDataConv, nSingleData), 0,1)
        d['abnormTrainData'] = np.swapaxes(predictFeatures(ml, abnormalTrainDataConv, nSingleData), 0,1) 
        d['normTestData']    = np.swapaxes(predictFeatures(ml, normalTestDataConv, nSingleData), 0,1)
        d['abnormTestData']  = np.swapaxes(predictFeatures(ml, abnormalTestDataConv, nSingleData), 0,1)
            
    else:
        print "Loading pca model data"
        from sklearn.decomposition import KernelPCA
        ml = KernelPCA(n_components=layer_sizes[-1], kernel="rbf", fit_inverse_transform=False, \
                       gamma=pca_gamma)

        print np.shape(normalTrainData), np.shape(abnormalTrainData)
        print np.shape(normalTrainDataConv), np.shape(abnormalTrainDataConv)
        print np.shape(X_train)
        print "Exit in pca data extraction"
        sys.exit()

        pca_model = os.path.join(processed_data_path, 'pca_model_'+str(idx)+'.pkl')
        if os.path.isfile(pca_model):
            print "PCA model exists: ", pca_model
            ml = joblib.load(pca_model)
        else:
            ml.fit(np.array(X_train))
            joblib.dump(ml, pca_model)

        def predictFeatures(clf, X, nSingleData):
            # Generate training features
            feature_list = []
            for idx in xrange(0, len(X), nSingleData):
                test_features = clf.transform( X[idx:idx+nSingleData,:] )
                feature_list.append(test_features)
                print np.shape(X[idx:idx+nSingleData,:]), np.shape(test_features)
            return feature_list

        # test ae
        # sample x dim => dim x sample
        d = {}
        d['normTrainData']   = np.swapaxes(predictFeatures(ml, normalTrainDataConv, nSingleData), 0,1)
        d['abnormTrainData'] = np.swapaxes(predictFeatures(ml, abnormalTrainDataConv, nSingleData), 0,1) 
        d['normTestData']    = np.swapaxes(predictFeatures(ml, normalTestDataConv, nSingleData), 0,1)
        d['abnormTestData']  = np.swapaxes(predictFeatures(ml, abnormalTestDataConv, nSingleData), 0,1)

        print np.shape(predictFeatures(ml, normalTrainDataConv, nSingleData))
        sys.exit()
    
    # dim x sample x length
    d['handNormTrainData']   = handSuccessData[:, normalTrainIdx, time_window-1:]
    d['handAbnormTrainData'] = handFailureData[:, abnormalTrainIdx, time_window-1:]
    d['handNormTestData']    = handSuccessData[:, normalTestIdx, time_window-1:]
    d['handAbnormTestData']  = handFailureData[:, abnormalTestIdx, time_window-1:]

    if filtering:
        pooling_param_dict  = {'dim': filteringDim} # only for AE        
        d['normTrainDataFiltered'], d['abnormTrainDataFiltered'],pooling_param_dict \
          = errorPooling(d['normTrainData'], d['abnormTrainData'], pooling_param_dict)
        d['normTestDataFiltered'], d['abnormTestDataFiltered'], _ \
          = errorPooling(d['normTestData'], d['abnormTestData'], pooling_param_dict)

    d['handFeatureNames'] = handParam['feature_names']
    ut.save_pickle(d, AE_proc_data)
    return d



def getHMMData(method, nFiles, processed_data_path, task_name, default_params, negTrain=False):
    '''
    This should be used only for training not for validation since it decomposes training data into
    internal training and test set.
    '''
    import os
    from sklearn import preprocessing

    ## Default Parameters
    # data
    data_dict = default_params['data_param']
    # AE
    AE_dict = default_params['AE']
    # HMM
    HMM_dict = default_params['HMM']
    # ROC
    ROC_dict = default_params['ROC']
    #------------------------------------------

    # load data and preprocess it
    print "Start to get data"
    data = {}
    for file_idx in xrange(nFiles):
        if AE_dict['switch'] and AE_dict['add_option'] is not None:
            tag = ''
            for ft in AE_dict['add_option']:
                tag += ft[:2]
            modeling_pkl = os.path.join(processed_data_path, 'hmm_'+task_name+'_raw_'+tag+'_'+str(file_idx)+'.pkl')
        elif AE_dict['switch']:
            modeling_pkl = os.path.join(processed_data_path, \
                                        'hmm_'+task_name+'_raw_'+str(file_idx)+'.pkl')
        else:
            modeling_pkl = os.path.join(processed_data_path, \
                                        'hmm_'+task_name+'_'+str(file_idx)+'.pkl')

        if os.path.isfile(modeling_pkl) is False:
            print "Please run evaluation all first to get hmm files."
            sys.exit()
        # train a classifier and evaluate it using test data.
        d            = ut.load_pickle(modeling_pkl)
        ## startIdx = d['startIdx']
        
        # sample x length x feature vector
        ll_classifier_train_X   = d['ll_classifier_train_X'] # [normal, abnormal]
        ll_classifier_train_Y   = d['ll_classifier_train_Y']         
        ll_classifier_train_idx = d['ll_classifier_train_idx']
        ## ll_classifier_test_X    = d['ll_classifier_test_X']  
        ## ll_classifier_test_Y    = d['ll_classifier_test_Y']
        ## ll_classifier_test_idx  = d['ll_classifier_test_idx']
        nLength      = d['nLength']
        nPoints      = ROC_dict['nPoints']

        if method == 'hmmsvm_dL':
            # replace dL/(ds+e) to dL
            for i in xrange(len(ll_classifier_train_X)):
                for j in xrange(len(ll_classifier_train_X[i])):
                    if j == 0:
                        ll_classifier_train_X[i][j][1] = 0.0
                    else:
                        ll_classifier_train_X[i][j][1] = ll_classifier_train_X[i][j][0] - \
                          ll_classifier_train_X[i][j-1][0]
        elif method == 'hmmsvm_LSLS':
            # reconstruct data into LS(t-1)+LS(t)
            if type(ll_classifier_train_X) is list:
                ll_classifier_train_X = np.array(ll_classifier_train_X)

            x = np.dstack([ll_classifier_train_X[:,:,:1], ll_classifier_train_X[:,:,2:]] )
            x = x.tolist()
            ## ll_classifier_train_X = np.hstack([ x, x ])

            new_x = []
            for i in xrange(len(x)):
                new_x.append([])
                for j in xrange(len(x[i])):
                    if j == 0:
                        new_x[i].append( x[i][j]+x[i][j] )
                    else:
                        new_x[i].append( x[i][j-1]+x[i][j] )

            ll_classifier_train_X = new_x
        elif (method == 'hmmsvm_no_dL' or HMM_dict['add_logp_d'] is False) and \
          len(ll_classifier_train_X) > HMM_dict['nState']+1:
            # remove dL/(ds+e)
            ll_classifier_train_X = np.array(ll_classifier_train_X)
            ll_classifier_train_X = np.delete(ll_classifier_train_X, 1, 2).tolist()
            

        # divide into training and param estimation set
        import random
        rnd_train_idx = random.sample(range(len(ll_classifier_train_X)), int( 0.7*len(ll_classifier_train_X)) )
        rnd_test_idx  = [x for x in range(len(ll_classifier_train_X)) if not x in rnd_train_idx] 

        train_X = np.array(ll_classifier_train_X)[rnd_train_idx]
        train_Y = np.array(ll_classifier_train_Y)[rnd_train_idx]
        train_idx = np.array(ll_classifier_train_idx)[rnd_train_idx]
        test_X  = np.array(ll_classifier_train_X)[rnd_test_idx]
        test_Y  = np.array(ll_classifier_train_Y)[rnd_test_idx]
        test_idx = np.array(ll_classifier_train_idx)[rnd_test_idx]

        if negTrain:
            normal_idx = [x for x in range(len(train_X)) if train_Y[x][0]<0 ]
            train_X = train_X[normal_idx]
            train_Y = train_Y[normal_idx]
            train_idx = train_idx[normal_idx]
        if method is 'bpsvm':
            l_abnorm_cut_idx = getHMMCuttingIdx(train_X, train_Y, \
                                                train_idx)
           
        # flatten the data
        X_train_org, Y_train_org, idx_train_org = flattenSample(train_X, train_Y, train_idx)

        # training data preparation
        if 'svm' in method or 'sgd' in method:
            scaler = preprocessing.StandardScaler()
            ## scaler = preprocessing.scale()
            X_scaled = scaler.fit_transform(X_train_org)
        else:
            X_scaled = X_train_org
        ## print method, " : Before classification : ", np.shape(X_scaled), np.shape(Y_train_org)

        # test data preparation
        X_test = []
        Y_test = []
        idx_test = test_idx
        for ii in xrange(len(test_X)):
            if np.nan in test_X[ii] or len(test_X[ii]) == 0 \
              or np.nan in test_X[ii][0]:
                continue

            if 'svm' in method or 'sgd' in method:
                X = scaler.transform(test_X[ii])                                
            elif method == 'progress_time_cluster' or method == 'fixed':
                X = test_X[ii]
            X_test.append(X)
            Y_test.append(test_Y[ii])


        data[file_idx]={}
        data[file_idx]['X_scaled']      = X_scaled
        data[file_idx]['Y_train_org']   = Y_train_org
        data[file_idx]['idx_train_org'] = idx_train_org
        data[file_idx]['X_test']   = X_test
        data[file_idx]['Y_test']   = Y_test
        data[file_idx]['idx_test'] = idx_test
        data[file_idx]['nLength'] = nLength
        if method is 'bpsvm':
            data[file_idx]['rnd_train_idx'] = rnd_train_idx
            data[file_idx]['rnd_test_idx'] = rnd_test_idx
            data[file_idx]['abnormal_train_cut_idx'] = l_abnorm_cut_idx

    return data 


def getPCAData(nFiles, data_pkl=None, window=1, gamma=1., pos_dict=None, use_test=True, use_pca=True,\
               test_drop_elements=None, step_anomaly_info=None, normalFoldData=None):

    if data_pkl is not None:
        d = ut.load_pickle(data_pkl)
        kFold_list  = d['kFoldList']
        successData = d['successData']
        failureData = d['failureData']
    else:
        (normal_folds, successData, failureData) = normalFoldData

    if window == 0:
        print "wrong window size"
        sys.exit()

    # load data and preprocess it
    print "Start to get data"
    data = {}
    for file_idx in xrange(nFiles):

        if data_pkl is not None:
            (normalTrainIdx, abnormalTrainIdx, normalTestIdx, abnormalTestIdx) = kFold_list[file_idx]

            # dim x sample x length
            normalTrainData   = successData[:, normalTrainIdx, :] 
            abnormalTrainData = failureData[:, abnormalTrainIdx, :] 
            normalTestData    = successData[:, normalTestIdx, :] 
            abnormalTestData  = failureData[:, abnormalTestIdx, :]
        else:
            (train_fold, test_fold) = normal_folds[file_idx]
             
            # dim x sample x length
            normalTrainData   = successData[:, train_fold] 
            abnormalTrainData = failureData
            normalTestData    = successData[:, test_fold] 
            abnormalTestData  = failureData

            

        # dim x sample x length => sample x dim x length
        normalTrainData   = np.swapaxes(normalTrainData, 0, 1)
        abnormalTrainData  = np.swapaxes(abnormalTrainData, 0, 1)         
        normalTestData    = np.swapaxes(normalTestData, 0, 1)
        abnormalTestData  = np.swapaxes(abnormalTestData, 0, 1)

        # sample x dim x length = > sample x length x dim 
        normalTrainData   = np.swapaxes(normalTrainData, 1, 2)
        abnormalTrainData = np.swapaxes(abnormalTrainData, 1, 2)         
        normalTestData    = np.swapaxes(normalTestData, 1, 2)
        abnormalTestData  = np.swapaxes(abnormalTestData, 1, 2)


        #--------------------------------------------------------------------------------
        if step_anomaly_info is not None:
            ## step_idx_l = [] 
            ## for i in xrange(len(normalTestData[0])):
            ##     step_idx_l.append(None) 
            ## for i in xrange(len(abnormalTestData[0])): 
            ##     step_idx_l.append(step_anomaly_info[i]) 

            ## step_idx_l = step_anomaly_info
            ## print "we do not use step_anomaly_info"
            ## sys.exit()
            
            modeling_pkl_prefix = step_anomaly_info[0]
            step_mag = step_anomaly_info[1]
            dd = ut.load_pickle(modeling_pkl_prefix+'_'+str(file_idx)+'.pkl')
            step_idx_l = dd['step_idx_l']

            abnormalTestData = copy.copy(normalTestData)
            for i in xrange(len(abnormalTestData)):
                abnormalTestData[i,step_idx_l[len(normalTestData)+i],:] += step_mag
        else:
            step_idx_l = None


        #--------------------------------------------------------------------------------
        # scaler
        from sklearn import preprocessing
        scaler = preprocessing.StandardScaler()
            
        # Training data
        if use_test:
            ll_classifier_train_X = normalTrainData
            ll_classifier_train_Y = [[-1]*len(normalTrainData[0])]*len(normalTrainData)
                    
            # flatten the data
            if window == 0: sys.exit()
            elif window==1:
                X_train_org, Y_train_org, _ = flattenSample(ll_classifier_train_X, \
                                                            ll_classifier_train_Y)
            else:
                X_train_org, Y_train_org, _ = flattenSampleWithWindow(ll_classifier_train_X, \
                                                                      ll_classifier_train_Y, window=window)

            X_scaled = scaler.fit_transform(X_train_org)

            if pos_dict is not None:
                abnormalTrainData_X = []
                abnormalTrainData_Y = []
                for i, cut_idx in enumerate(pos_dict[file_idx]['abnormal_train_cut_idx']):
                    abnormalTrainData_X.append( abnormalTrainData[i][cut_idx:].tolist() )
                    abnormalTrainData_Y.append( [1]*len(abnormalTrainData_X[i]) )

                if window == 0: sys.exit()
                elif window==1:
                    X_abnorm_train, Y_abnorm_train, _ = flattenSample(abnormalTrainData_X, \
                                                                      abnormalTrainData_Y)
                else:
                    X_abnorm_train, Y_abnorm_train, _ = flattenSampleWithWindow(abnormalTrainData_X, \
                                                                                abnormalTrainData_Y, \
                                                                                window=window)

                X_scaled = X_scaled.tolist() + scaler.transform(X_abnorm_train).tolist()
                Y_train_org = Y_train_org + Y_abnorm_train

            # Testing data
            ll_classifier_test_X   = np.vstack([normalTestData, abnormalTestData])
            ll_classifier_test_Y   = [[-1]*len(normalTestData[0])]*len(normalTestData)+\
              [[1]*len(abnormalTestData[0])]*len(abnormalTestData)
            ll_classifier_test_idx = [range(len(normalTestData[0]))]*len(normalTestData) + \
              [range(len(abnormalTestData[0]))]*len(abnormalTestData)
            ll_classifier_test_idx = np.array(ll_classifier_test_idx)
                
        else:
            ll_classifier_train_X = np.vstack([normalTrainData, abnormalTrainData])
            ll_classifier_train_Y = [[-1]*len(normalTrainData[0])]*len(normalTrainData)+\
              [[1]*len(abnormalTrainData[0])]*len(abnormalTrainData)
            ll_classifier_train_idx = [range(len(normalTrainData[0]))]*len(normalTrainData) + \
              [range(len(abnormalTrainData[0]))]*len(abnormalTrainData)
            ll_classifier_train_idx = np.array(ll_classifier_train_idx)

            if pos_dict is None:
                rnd_train_idx = random.sample(range(len(normalTrainData)), \
                                              int( 0.7*len(normalTrainData)) )
                rnd_test_idx  = [x for x in range(len(ll_classifier_train_X)) if not x in rnd_train_idx]
                
                X_train = np.array(ll_classifier_train_X)[rnd_train_idx]
                Y_train = np.array(ll_classifier_train_Y)[rnd_train_idx]

                # flatten the data
                if window==1:
                    X_train_org, Y_train_org, _ = flattenSample(X_train, Y_train)
                else:
                    X_train_org, Y_train_org, _ = flattenSampleWithWindow(X_train, Y_train, window=window)

                ll_classifier_test_X   = np.array(ll_classifier_train_X)[rnd_test_idx]
                ll_classifier_test_Y   = np.array(ll_classifier_train_Y)[rnd_test_idx]
                ll_classifier_test_idx = np.array(ll_classifier_train_idx)[rnd_test_idx]
                    
            else:
                rnd_train_idx = pos_dict[file_idx]['rnd_train_idx']
                rnd_test_idx  = pos_dict[file_idx]['rnd_test_idx']
            
                X = np.array(ll_classifier_train_X)[rnd_train_idx]
                Y = np.array(ll_classifier_train_Y)[rnd_train_idx]
                ll_classifier_test_X   = np.array(ll_classifier_train_X)[rnd_test_idx]
                ll_classifier_test_Y   = np.array(ll_classifier_train_Y)[rnd_test_idx]
                ll_classifier_test_idx = np.array(ll_classifier_train_idx)[rnd_test_idx]

                normalTrainData_X = []
                normalTrainData_Y = []
                abnormalTrainData_X = []
                abnormalTrainData_Y = []

                count = 0
                for i, x in enumerate(X):
                    if Y[i][0] < 0:
                        normalTrainData_X.append(x)
                        normalTrainData_Y.append(Y[i])
                    else:
                        abnormalTrainData_X.append(x)
                        abnormalTrainData_Y.append(Y[i])

                if len(pos_dict[file_idx]['abnormal_train_cut_idx'])-len(abnormalTrainData_X) is not 0:
                    print "wrong number of cutting data"
                    sys.exit()

                for i, cut_idx in enumerate(pos_dict[file_idx]['abnormal_train_cut_idx']):
                    abnormalTrainData_X[i] = abnormalTrainData_X[i][cut_idx:].tolist() 
                    abnormalTrainData_Y[i] = abnormalTrainData_Y[i][cut_idx:].tolist()

                # flatten the data
                if window==1:
                    X_train_org, Y_train_org, _ = flattenSample(normalTrainData_X,
                                                                normalTrainData_Y)
                    X_abnorm_train, Y_abnorm_train, _ = flattenSample(abnormalTrainData_X, \
                                                                      abnormalTrainData_Y)
                else:
                    X_train_org, Y_train_org, _ = flattenSampleWithWindow(normalTrainData_X,\
                                                                          normalTrainData_Y,\
                                                                          window=window)
                    X_abnorm_train, Y_abnorm_train, _ = flattenSampleWithWindow(abnormalTrainData_X, \
                                                                                abnormalTrainData_Y, \
                                                                                window=window)

                X_train_org = X_train_org + X_abnorm_train
                Y_train_org = Y_train_org + Y_abnorm_train                    
                
            X_scaled = scaler.fit_transform(X_train_org)

                       
        ## # PCA
        if use_pca:
            from sklearn.decomposition import KernelPCA
            ml = KernelPCA(n_components=2, kernel="rbf", fit_inverse_transform=False, \
                           gamma=gamma, degree=3)
            X_scaled = ml.fit_transform(np.array(X_scaled))

        # LLE
        ## from sklearn.manifold import LocallyLinearEmbedding
        ## ml = LocallyLinearEmbedding(5,2, reg=gamma)        
        ## X_scaled = ml.fit_transform(np.array(X_scaled))

        #--------------------------------------------------------------------------------

        if test_drop_elements is not None:
            drop_idx_l  = test_drop_elements[file_idx][0]
            drop_length = test_drop_elements[file_idx][1]

            ll_classifier_test_X = np.swapaxes(ll_classifier_test_X, 1,2) # sample x dim x length            
            nLength = len(ll_classifier_test_X[0][0])
            
            samples = []
            for i in xrange(len(ll_classifier_test_X)):
                start_idx = drop_idx_l[i]
                end_idx   = start_idx+drop_length
                if end_idx > nLength-1: end_idx = nLength-1
                rnd_idx_l = range(start_idx, end_idx)

                sample = []
                for j in xrange(len(ll_classifier_test_X[i])):
                    sample.append( np.delete( ll_classifier_test_X[i][j], rnd_idx_l ) )

                sample = np.swapaxes(sample, 0,1) # length x dim
                samples.append(sample)

                if ll_classifier_test_Y[i][0]>0:
                    ll_classifier_test_Y[i] = [1]*len(sample)
                else:
                    ll_classifier_test_Y[i] = [-1]*len(sample)

            ll_classifier_test_X = samples


        # test data preparation
        X_test = []
        Y_test = []
        if window==1:
            for ii in xrange(len(ll_classifier_test_X)):
                if np.nan in ll_classifier_test_X[ii] or len(ll_classifier_test_X[ii]) == 0 \
                  or np.nan in ll_classifier_test_X[ii][0]:
                    continue

                X = scaler.transform(ll_classifier_test_X[ii])
                if use_pca: X = ml.transform(X)
                X_test.append(X)
                Y_test.append(ll_classifier_test_Y[ii])            
        else:
            ll_classifier_test_X = sampleWithWindow(ll_classifier_test_X, window=window)
            for ii in xrange(len(ll_classifier_test_X)):
                if np.nan in ll_classifier_test_X[ii] or len(ll_classifier_test_X[ii]) == 0 \
                  or np.nan in ll_classifier_test_X[ii][0]:
                  print ii, " : nan in data "
                  continue

                X = scaler.transform(ll_classifier_test_X[ii])
                if use_pca: X = ml.transform(X)
                X_test.append(X)
                Y_test.append(ll_classifier_test_Y[ii])

        
        ## fig = plt.figure(1)
        ## plt.scatter(X_scaled[:,0], X_scaled[:,1], c='blue')
        ## for i in xrange(len(X_test)):
        ##     if Y_test[i][0] == -1: continue
        ##     plt.scatter(np.array(X_test)[i,:,0], np.array(X_test)[i,:,1], c='red', marker='x')        
        ## plt.axis('tight')        
        ## plt.show()
        ## fig.savefig('test'+str(file_idx)+'.pdf')
        ## fig.savefig('test'+str(file_idx)+'.png')
        ## os.system('mv test*.png ~/Dropbox/HRL/')

        #--------------------------------------------------------------------------------
        data[file_idx]={}
        data[file_idx]['X_scaled']      = X_scaled
        data[file_idx]['Y_train_org']   = Y_train_org
        data[file_idx]['idx_train_org'] = None
        data[file_idx]['X_test']        = X_test
        data[file_idx]['Y_test']        = Y_test
        data[file_idx]['idx_test']      = ll_classifier_test_idx
        data[file_idx]['nLength']       = len(normalTrainData[0][0])
        data[file_idx]['step_idx_l']    = step_idx_l
    return data 
    

def getHMMCuttingIdx(ll_X, ll_Y, ll_idx):
    '''
    ll_X : sample x length x hmm features
    ll_Y : sample x length
    ll_idx:
    '''
    ## print np.shape(ll_X), np.shape(ll_Y), np.shape(ll_idx)
    ## print ll_Y[-1][0], ll_X[-1][:,0]
    ## sys.exit()
    
    l_X   = []
    l_Y   = []
    l_idx = []
    for i in xrange(len(ll_X)):
        if ll_Y[i][0] < 0:
            ## l_idx.append(ll_idx[i][-1])
            continue
        else:
            _,_,idx = getEstTruePositive(ll_X[i], ll_idx=ll_idx[i])
            l_idx.append(idx)
            
    return l_idx
    

def getAnomalyInfo(task_name, processed_data_path, rf_center='kinEEPos', local_range=10.0):

    crossVal_pkl = os.path.join(processed_data_path, 'cv_'+task_name+'.pkl')
            
    if os.path.isfile(crossVal_pkl) is False:
        print "No data!!!!!!!!!!!!!"
        sys.exit()
        
    # get success / failure data
    print "CV data exists and no renew"
    d = ut.load_pickle(crossVal_pkl)
    kFold_list = d['kFoldList'] 
    successData = d['successData']
    failureData = d['failureData']
    param_dict  = d['param_dict']
    
    print np.shape(successData)
    print np.shape(failureData)
   
    # get difference
    avg_success = np.mean(successData, axis=1)
    print np.shape(avg_success)

    feature_anomaly_diff = []
    for i in xrange(len(failureData)): # per feature
        diff = []
        for j in xrange(len(failureData[i])): # per sample
            diff.append(failureData[i][j]-avg_success[i])
        feature_anomaly_diff.append(diff)
        
    print np.shape(feature_anomaly_diff)

    # get scaling factor
    print param_dict.keys()
    scale = np.array(param_dict['feature_max'])-np.array(param_dict['feature_min'])

    # inverse scale the difference
    avg_success_list = []
    max_success_list = []
    min_success_list = []
    max_failure_list = []
    min_failure_list = []
    max_diff_list = []
    min_diff_list = []
    for i in xrange(len(feature_anomaly_diff)):
        avg_success_list.append( np.mean(successData[i])*scale[i] + param_dict['feature_min'][i] )
        
        max_success_list.append( np.amax(successData[i])*scale[i] + param_dict['feature_min'][i] )
        min_success_list.append( np.amin(successData[i])*scale[i] + param_dict['feature_min'][i] )
        max_failure_list.append( np.amax(failureData[i])*scale[i] + param_dict['feature_min'][i] )
        min_failure_list.append( np.amin(failureData[i])*scale[i] + param_dict['feature_min'][i] )
        max_diff_list.append( np.amax(feature_anomaly_diff[i])*scale[i] )
        min_diff_list.append( np.amin(feature_anomaly_diff[i])*scale[i] )

    # print out
    print param_dict['feature_names']
    print "--------- average --------------"
    print avg_success_list
    print "--------- success --------------"
    print max_success_list
    print min_success_list
    print "--------- failure --------------"
    print max_failure_list
    print min_failure_list    
    print "--------- diff -----------------" 
    print max_diff_list
    print min_diff_list
    

def errorPooling(norX, abnorX, param_dict):
    '''
    dim x samples
    Select non-stationary data
    Assuption: norX and abnorX should have the same phase
    '''
    dim         = param_dict['dim']

    if 'dim_idx' not in param_dict.keys():
        dim_idx    = []
        new_norX   = []
        new_abnorX = []

        err_list = []
        for i in xrange(len(norX)):
            # get mean curve
            meanNorCurve   = np.mean(norX[i], axis=0)
            ## meanAbnorCurve = np.mean(abnorX[i], axis=0)
            stdNorCurve   = np.std(norX[i], axis=0)
            ## stdAbnorCurve = np.std(abnorX[i], axis=0)
            ## if np.std(meanNorCurve) < 0.02 and np.std(meanAbnorCurve) < 0.02 and\
            ##   np.mean(stdNorCurve) < 0.02 and np.mean(stdAbnorCurve) < 0.02:
            ##     err_list.append(1e-9)
            ##     continue

            maxCurve = meanNorCurve+stdNorCurve
            minCurve = meanNorCurve-stdNorCurve

            # get error score
            score = 0.0
            for j in xrange(len(abnorX[i])):
                for k in xrange(len(abnorX[i][j])):
                    if abnorX[i][j][k] > maxCurve[k] or abnorX[i][j][k] < minCurve[k]:
                       score += 1.0

            # get mean range , mean std
            score /= np.max(meanNorCurve)-np.min(meanNorCurve)
            #score *= np.mean(stdNorCurve)
                       
            err_list.append(score)

        indices = np.argsort(err_list)

        for idx in indices[:dim]:
            new_norX.append(norX[idx])
            new_abnorX.append(abnorX[idx])
            dim_idx.append(idx)

            ## if all_std > min_all_std and avg_ea_std < max_avg_std:
            ##     new_X.append(X[i])
            ##     dim_idx.append(i)

        param_dict['dim_idx'] = dim_idx
    else:
        new_norX = [ norX[idx] for idx in param_dict['dim_idx'] ]
        new_abnorX = [ abnorX[idx] for idx in param_dict['dim_idx'] ]

    return np.array(new_norX), np.array(new_abnorX), param_dict
        

def variancePooling(X, param_dict):
    '''
    dim x samples
    Select non-stationary data
    
    TODO: can we select final dimension?
    
    '''
    dim         = param_dict['dim']
    ## min_all_std = param_dict['min_all_std']
    ## max_avg_std = param_dict['max_avg_std']

    if 'dim_idx' not in param_dict.keys():
        dim_idx = []
        new_X   = []

        std_list = []
        for i in xrange(len(X)):
            # for each dimension
            ## avg_std = np.mean( np.std(X[i], axis=0) )
            std_avg = np.std( np.mean(X[i], axis=0) )
            ## std_list.append(std_avg/avg_std)
            std_list.append(std_avg)

        indices = np.argsort(std_list)[::-1]

        for idx in indices[:dim]:
            new_X.append(X[idx])
            dim_idx.append(idx)

            ## if all_std > min_all_std and avg_ea_std < max_avg_std:
            ##     new_X.append(X[i])
            ##     dim_idx.append(i)

        param_dict['dim_idx'] = dim_idx
    else:
        new_X = [ X[idx] for idx in param_dict['dim_idx'] ]

    return np.array(new_X), param_dict
        
    
#-------------------------------------------------------------------------------------------------

def extractHandFeature(d, feature_list, cut_data=None, init_param_dict=None, verbose=False, \
                       renew_minmax=False):

    if len(d['timesList']) == 0: return [], {}

    if init_param_dict is None:
        isTrainingData=True
        param_dict = {}
        param_dict['timeList'] = d['timesList'][0]

        if 'unimodal_audioPower' in feature_list:
            power_min = 10000
            power_max = 0
            for pwr in d['audioPowerList']:
                p_min = np.amin(pwr)
                p_max = np.amax(pwr)
                if power_min > p_min:
                    power_min = p_min
                if power_max < p_max:
                    power_max = p_max

            param_dict['unimodal_audioPower_power_max'] = power_max
            param_dict['unimodal_audioPower_power_min'] = power_min
                                
        if 'unimodal_ppsForce' in feature_list:
            ppsLeft  = d['ppsLeftList']
            ppsRight = d['ppsRightList']

            pps_mag = []
            for i in xrange(len(ppsLeft)):                
                pps      = np.vstack([ppsLeft[i], ppsRight[i]])
                pps_mag.append( np.linalg.norm(pps, axis=0) )

            pps_max = np.max( np.array(pps_mag).flatten() )
            pps_min = np.min( np.array(pps_mag).flatten() )
            param_dict['unimodal_ppsForce_max'] = pps_max
            param_dict['unimodal_ppsForce_min'] = pps_min

        param_dict['feature_names'] = []
    else:
        param_dict = copy.copy(init_param_dict)
        isTrainingData=False
            

    # -------------------------------------------------------------        

    # extract local features
    startOffsetSize = 4
    dataList   = []
    for idx in xrange(len(d['timesList'])): # each sample

        param_dict['timeList'] = timeList = d['timesList'][idx]
        dataSample = None
        if len(timeList) < 2: offset_flag=False
        else: offset_flag=True
            

        # Unimoda feature - Audio --------------------------------------------
        if 'unimodal_audioPower' in feature_list:
            ## audioAzimuth = d['audioAzimuthList'][idx]
            unimodal_audioPower = d['audioPowerList'][idx]
            
            if dataSample is None: dataSample = copy.copy(np.array(unimodal_audioPower))
            else: dataSample = np.vstack([dataSample, copy.copy(unimodal_audioPower)])
            if 'audioPower' not in param_dict['feature_names']:
                param_dict['feature_names'].append('audioPower')

        # Unimoda feature - AudioWrist ---------------------------------------
        if 'unimodal_audioWristRMS' in feature_list:
            unimodal_audioWristRMS = d['audioWristRMSList'][idx]
            if offset_flag:
                unimodal_audioWristRMS -= np.amin(unimodal_audioWristRMS)
                ## unimodal_audioWristRMS -= np.mean(audioWristRMS[:startOffsetSize])

            if dataSample is None: dataSample = copy.copy(np.array(unimodal_audioWristRMS))
            else: dataSample = np.vstack([dataSample, copy.copy(unimodal_audioWristRMS)])
            if 'audioWristRMS' not in param_dict['feature_names']:
                param_dict['feature_names'].append('audioWristRMS')

        # Unimoda feature - AudioWristFront------------------------------------
        if 'unimodal_audioWristFrontRMS' in feature_list:
            unimodal_audioWristFrontRMS = d['audioWristFrontRMSList'][idx]
            if offset_flag:
                unimodal_audioWristFrontRMS -= np.amin(unimodal_audioWristFrontRMS[:startOffsetSize])
                ## unimodal_audioWristFrontRMS -= np.mean(audioWristFrontRMS[:startOffsetSize])

            if dataSample is None: dataSample = copy.copy(np.array(unimodal_audioWristFrontRMS))
            else: dataSample = np.vstack([dataSample, copy.copy(unimodal_audioWristFrontRMS)])
            if 'audioWristFrontRMS' not in param_dict['feature_names']:
                param_dict['feature_names'].append('audioWristFrontRMS')

        # Unimoda feature - AudioWristAzimuth------------------------------------
        if 'unimodal_audioWristAzimuth' in feature_list:
            unimodal_audioWristAzimuth = d['audioWristAzimuthList'][idx]
            ## if offset_flag:
            ##     unimodal_audioWristAzimuth -= np.mean(unimodal_audioWristAzimuth[:startOffsetSize])
            unimodal_audioWristAzimuth = abs(unimodal_audioWristAzimuth)
            
            if dataSample is None: dataSample = copy.copy(np.array(unimodal_audioWristAzimuth))
            else: dataSample = np.vstack([dataSample, copy.copy(unimodal_audioWristAzimuth)])
            if 'audioWristAzimuth' not in param_dict['feature_names']:
                param_dict['feature_names'].append('audioWristAzimuth')

        # Unimodal feature - Kinematics --------------------------------------
        if 'unimodal_kinVel' in feature_list:

            kinEEPos        = d['kinEEPosList'][idx]            
            ## unimodal_kinVel = d['kinVelList'][idx]

            if len(kinEEPos[0])>2:
                vel = np.linalg.norm( kinEEPos[:,1:] - kinEEPos[:,:-1], axis=0 )
                vel = np.array( [0] + vel.tolist() )
            else:
                vel = np.linalg.norm( kinEEPos[:,-1:] -
                                      d['kinEEPosList_last'][idx][:,-1:], axis=0 )
                
            ## vel = np.linalg.norm(unimodal_kinVel, axis=0)

            if dataSample is None: dataSample = vel
            else: dataSample = np.vstack([dataSample, vel])
            if 'kinVel' not in param_dict['feature_names']:
                param_dict['feature_names'].append('kinVel')

        # Unimodal feature - Kinematics --------------------------------------
        for jnt_idx in xrange(7):
            if 'unimodal_kinJntEff_'+str(jnt_idx+1) in feature_list:
                unimodal_kinJntEff = d['kinJntEffList'][idx]

                # TODO: need to simplify!
                if offset_flag:
                    offset = np.mean(unimodal_kinJntEff[jnt_idx,:startOffsetSize])
                    unimodal_kinJntEff[jnt_idx] -= offset

                if dataSample is None: dataSample = np.array( unimodal_kinJntEff[jnt_idx:jnt_idx+1] )
                else: dataSample = np.vstack([ dataSample, unimodal_kinJntEff[jnt_idx:jnt_idx+1] ])
                if 'kinJntEff_'+str(jnt_idx+1) not in param_dict['feature_names']:           
                    param_dict['feature_names'].append( 'kinJntEff_'+str(jnt_idx+1) )

        ## # Unimodal feature - Kinematics --------------------------------------
        ## if 'unimodal_kinJntEff' in feature_list:
        ##     unimodal_kinJntEff = d['kinJntEffList'][idx]

        ##     if offset_flag:
        ##         offset = np.mean(unimodal_kinJntEff[:,:startOffsetSize], axis=1)
        ##         for i in xrange(len(offset)):
        ##             unimodal_kinJntEff[i] -= offset[i]


        ##     if dataSample is None: dataSample = np.array(unimodal_kinJntEff)
        ##     else: dataSample = np.vstack([dataSample, unimodal_kinJntEff])
        ##     if 'kinJntEff_1' not in param_dict['feature_names']:           
        ##         for i in xrange(len(unimodal_kinJntEff)):
        ##             param_dict['feature_names'].append('kinJntEff_'+str(i+1))

        # Unimodal feature - Force -------------------------------------------
        if 'unimodal_ftForce' in feature_list:
            ftForce = d['ftForceList'][idx]
            
            # magnitude
            ## if len(np.shape(ftForce)) > 1:
            unimodal_ftForce_mag = np.linalg.norm(ftForce, axis=0)
            if offset_flag: #correct???????
                unimodal_ftForce_mag -= np.mean(unimodal_ftForce_mag[:startOffsetSize])

            if dataSample is None: dataSample = np.array(unimodal_ftForce_mag)
            else: dataSample = np.vstack([dataSample, unimodal_ftForce_mag])

            if 'ftForce_mag' not in param_dict['feature_names']:
                param_dict['feature_names'].append('ftForce_mag')
            ## else:                
            ##     unimodal_ftForce_mag = ftForce
            
            ##     if dataSample is None: dataSample = np.array(unimodal_ftForce_mag)
            ##     else: dataSample = np.vstack([dataSample, unimodal_ftForce_mag])

            ##     if 'ftForce_mag' not in param_dict['feature_names']:
            ##         param_dict['feature_names'].append('ftForce_mag')

        # Unimodal feature - Force zeroing -------------------------------------------
        if 'unimodal_ftForce_zero' in feature_list:
            ftForce = d['ftForceList'][idx]

            unimodal_ftForce_mean = np.mean(ftForce[:,:startOffsetSize], axis=1)
            for i in xrange(len(ftForce)):
                ftForce[i] -= unimodal_ftForce_mean[i]
            
            # magnitude
            unimodal_ftForce_mag = np.linalg.norm(ftForce, axis=0)
            ## if offset_flag: #correct???????
            ##     unimodal_ftForce_mag -= np.mean(unimodal_ftForce_mag[:startOffsetSize])

            if dataSample is None: dataSample = np.array(unimodal_ftForce_mag)
            else: dataSample = np.vstack([dataSample, unimodal_ftForce_mag])

            if 'ftForce_mag_zero' not in param_dict['feature_names']:
                param_dict['feature_names'].append('ftForce_mag_zero')


        # Unimodal feature - Force zeroing -------------------------------------------
        if 'unimodal_ftForce_integ' in feature_list:
            ftForce = d['ftForceList'][idx]

            if offset_flag: 
                ftForce -= np.mean(ftForce[:,:startOffsetSize], axis=1)[:,np.newaxis]
            ## else:
            ##     ftForce -= d['ftForceList_init'][0]
                
            # magnitude
            unimodal_ftForce_mag = np.linalg.norm(ftForce, axis=0)
            if offset_flag: 
                unimodal_ftForce_mag -= np.mean(unimodal_ftForce_mag[:startOffsetSize])

            # cumulation
            if len(unimodal_ftForce_mag)>1:
                for i in xrange(1,len(unimodal_ftForce_mag)):
                    unimodal_ftForce_mag[i] += unimodal_ftForce_mag[i-1]
            ## else:
            ##     # last integ before scaling
                
            if dataSample is None: dataSample = np.array(unimodal_ftForce_mag)
            else: dataSample = np.vstack([dataSample, unimodal_ftForce_mag])

            if 'ftForce_mag_integ' not in param_dict['feature_names']:
                param_dict['feature_names'].append('ftForce_mag_integ')


        # Unimodal feature - Force zeroing -------------------------------------------
        if 'unimodal_ftForce_delta' in feature_list:
            ftForce = d['ftForceList'][idx]

            unimodal_ftForce_mean = np.mean(ftForce[:,:startOffsetSize], axis=1)
            for i in xrange(len(ftForce)):
                ftForce[i] -= unimodal_ftForce_mean[i]
                
            # magnitude
            unimodal_ftForce_mag = np.linalg.norm(ftForce, axis=0)
            ## unimodal_ftForce_mag = np.sum(ftForce**2, axis=0)
            if offset_flag: #correct???????
                unimodal_ftForce_mag -= np.mean(unimodal_ftForce_mag[:startOffsetSize])
            unimodal_ftForce_mag -= np.array([unimodal_ftForce_mag[0]]+unimodal_ftForce_mag.tolist()[:-1])

            if dataSample is None: dataSample = np.array(unimodal_ftForce_mag)
            else: dataSample = np.vstack([dataSample, unimodal_ftForce_mag])

            if 'ftForce_mag_delta' not in param_dict['feature_names']:
                param_dict['feature_names'].append('ftForce_mag_delta')


        # Unimodal feature - Force -------------------------------------------
        if 'unimodal_ftForceX' in feature_list:
            ftForce = d['ftForceList'][idx]
            
            # magnitude
            if len(np.shape(ftForce)) > 1:
                unimodal_ftForce_x = ftForce[0:1,:]
                if offset_flag:
                    unimodal_ftForce_x -= np.mean(unimodal_ftForce_x[:,:startOffsetSize])
            else:                
                unimodal_ftForce_x = ftForce

            if dataSample is None: dataSample = np.array(unimodal_ftForce_x)
            else: dataSample = np.vstack([dataSample, unimodal_ftForce_x])

            if 'ftForce_x' not in param_dict['feature_names']:
                param_dict['feature_names'].append('ftForce_x')


        # Unimodal feature - Force -------------------------------------------
        if 'unimodal_ftForceY' in feature_list:
            ftForce = d['ftForceList'][idx]
            
            # magnitude
            if len(np.shape(ftForce)) > 1:
                unimodal_ftForce_y = ftForce[1:2,:]
                if offset_flag:
                    unimodal_ftForce_y -= np.mean(unimodal_ftForce_y[:,:startOffsetSize])
            else:                
                unimodal_ftForce_y = ftForce

            if dataSample is None: dataSample = np.array(unimodal_ftForce_y)
            else: dataSample = np.vstack([dataSample, unimodal_ftForce_y])

            if 'ftForce_y' not in param_dict['feature_names']:
                param_dict['feature_names'].append('ftForce_y')


        # Unimodal feature - Force -------------------------------------------
        if 'unimodal_ftForceZ' in feature_list:
            ftForce = d['ftForceList'][idx]
            
            # magnitude
            if len(np.shape(ftForce)) > 1:
                unimodal_ftForce_z = ftForce[2:3,:]
                if offset_flag:
                    unimodal_ftForce_z -= np.mean(unimodal_ftForce_z[:,:startOffsetSize])
            else:                
                unimodal_ftForce_z = ftForce

            if dataSample is None: dataSample = np.array(unimodal_ftForce_z)
            else: dataSample = np.vstack([dataSample, unimodal_ftForce_z])

            if 'ftForce_z' not in param_dict['feature_names']:
                param_dict['feature_names'].append('ftForce_z')



        # Unimodal feature - pps -------------------------------------------
        if 'unimodal_ppsForce' in feature_list:
            ppsLeft  = d['ppsLeftList'][idx]
            ppsRight = d['ppsRightList'][idx]
            ppsPos   = d['kinTargetPosList'][idx]

            pps = np.vstack([ppsLeft, ppsRight])
            unimodal_ppsForce = pps

            # 2
            pps = np.vstack([np.sum(ppsLeft, axis=0), np.sum(ppsRight, axis=0)])
            unimodal_ppsForce = pps
            
            # 1
            ## unimodal_ppsForce = np.array([np.linalg.norm(pps, axis=0)])
            if offset_flag:
                unimodal_ppsForce -= np.array([np.mean(unimodal_ppsForce[:,:startOffsetSize], \
                                                       axis=1)]).T

            ## unimodal_ppsForce = []
            ## for time_idx in xrange(len(timeList)):
            ##     unimodal_ppsForce.append( np.linalg.norm(pps[:,time_idx]) )

            if dataSample is None: dataSample = unimodal_ppsForce
            else: dataSample = np.vstack([dataSample, unimodal_ppsForce])

            ## if 'ppsForce' not in param_dict['feature_names']:
            ##     param_dict['feature_names'].append('ppsForce')
            if 'ppsForce_1' not in param_dict['feature_names']:
                param_dict['feature_names'].append('ppsForce_1')
                param_dict['feature_names'].append('ppsForce_2')                
            ## if 'ppsForce_1' not in param_dict['feature_names']:
            ##     param_dict['feature_names'].append('ppsForce_1')
            ##     param_dict['feature_names'].append('ppsForce_2')
            ##     param_dict['feature_names'].append('ppsForce_3')
            ##     param_dict['feature_names'].append('ppsForce_4')
            ##     param_dict['feature_names'].append('ppsForce_5')
            ##     param_dict['feature_names'].append('ppsForce_6')


        # Unimodal feature - vision change ------------------------------------
        if 'unimodal_visionChange' in feature_list:
            unimodal_visionChange = d['visionChangeMagList'][idx]

            if dataSample is None: dataSample = unimodal_visionChange
            else: dataSample = np.vstack([dataSample, unimodal_visionChange])
            if 'visionChange' not in param_dict['feature_names']:
                param_dict['feature_names'].append('visionChange')

                
        # Unimodal feature - fabric skin ------------------------------------
        if 'unimodal_fabricForce' in feature_list:
            unimodal_fabricForce = d['fabricMagList'][idx]

            if offset_flag:
                unimodal_fabricForce -= np.amin(unimodal_fabricForce)

            if dataSample is None: dataSample = unimodal_fabricForce
            else: dataSample = np.vstack([dataSample, unimodal_fabricForce])
            if 'fabricForce' not in param_dict['feature_names']:
                param_dict['feature_names'].append('fabricForce')

        # Unimodal feature - landmark motion --------------------------
        if 'unimodal_landmarkDist' in feature_list:
            visionLandmarkPos = d['visionLandmarkPosList'][idx] # originally length x 3*tags

            if len(np.shape(visionLandmarkPos)) == 1:
                visionLandmarkPos = np.reshape(visionLandmarkPos, (3,1))

            ## if offset_flag:
            ##     visionLandmarkPos_mean = np.mean(visionLandmarkPos[:,:startOffsetSize], axis=1)
            ##     for i in xrange(len(visionLandmarkPos)):
            ##         visionLandmarkPos[i] -= visionLandmarkPos_mean[i]
            dist = np.linalg.norm(visionLandmarkPos, axis=0)
            if offset_flag:
                dist -= np.mean(dist[:startOffsetSize])

            crossmodal_landmarkDist = []
            for time_idx in xrange(len(timeList)):
                crossmodal_landmarkDist.append(dist[time_idx])
                
            if dataSample is None: dataSample = np.array(crossmodal_landmarkDist)
            else: dataSample = np.vstack([dataSample, crossmodal_landmarkDist])
            
            if 'landmarkDist' not in param_dict['feature_names']:
                param_dict['feature_names'].append('landmarkDist')


        # Unimodal feature - EE change --------------------------
        if 'unimodal_kinEEChange' in feature_list:
            kinEEPos     = d['kinEEPosList'][idx]

            ## if offset_flag:
            ##     offset = np.mean(kinEEPos[:,:startOffsetSize], axis=1)
            ##     for i in xrange(len(offset)):
            ##         kinEEPos[i] -= offset[i]
            if offset_flag:
                kinEEPos -= np.mean(kinEEPos[:,:startOffsetSize], axis=1)[:,np.newaxis]
            dist = np.linalg.norm(kinEEPos, axis=0)

            if dataSample is None: dataSample = np.array(dist)
            else: dataSample = np.vstack([dataSample, dist])
            if 'EEChange' not in param_dict['feature_names']:
                param_dict['feature_names'].append('EEChange')


        # Unimodal feature - Desired EE change --------------------------
        if 'unimodal_kinDesEEChange' in feature_list:
            kinEEPos     = d['kinEEPosList'][idx]
            kinDesEEPos  = d['kinDesEEPosList'][idx]

            ## kinDesEEPos -= np.mean(kinDesEEPos[:,:startOffsetSize], axis=1)[:,np.newaxis]
            ## dist = np.linalg.norm(kinDesEEPos, axis=0)
            
            dist = np.linalg.norm(kinEEPos-kinDesEEPos, axis=0)
            if offset_flag:
                dist -= np.mean(dist[:startOffsetSize])

            if dataSample is None: dataSample = np.array(dist)
            else: dataSample = np.vstack([dataSample, dist])
            if 'DesEEChange' not in param_dict['feature_names']:
                param_dict['feature_names'].append('DesEEChange')


        # Crossmodal feature - relative dist --------------------------
        if 'crossmodal_targetEEDist' in feature_list:
            kinEEPos     = d['kinEEPosList'][idx]
            kinTargetPos  = d['kinTargetPosList'][idx]

            dist = np.linalg.norm(kinTargetPos - kinEEPos, axis=0)
            if offset_flag:
                dist -= np.mean(dist[:startOffsetSize])
            
            crossmodal_targetEEDist = []
            for time_idx in xrange(len(timeList)):
                crossmodal_targetEEDist.append( dist[time_idx])

            if dataSample is None: dataSample = np.array(crossmodal_targetEEDist)
            else: dataSample = np.vstack([dataSample, crossmodal_targetEEDist])
            if 'targetEEDist' not in param_dict['feature_names']:
                param_dict['feature_names'].append('targetEEDist')


        # Crossmodal feature - relative Velocity --------------------------
        if 'crossmodal_targetEEVel' in feature_list:
            kinEEPos     = d['kinEEPosList'][idx]
            kinTargetPos  = d['kinTargetPosList'][idx]

            dist = np.linalg.norm(kinTargetPos - kinEEPos, axis=0)
            if offset_flag:
                dist -= np.mean(dist[:startOffsetSize])

            vel = dist[1:]-dist[:-1]
            ## vel =
            sys.exit()
            
            crossmodal_targetEEDist = []
            for time_idx in xrange(len(timeList)):
                crossmodal_targetEEDist.append( dist[time_idx])

            if dataSample is None: dataSample = np.array(crossmodal_targetEEDist)
            else: dataSample = np.vstack([dataSample, crossmodal_targetEEDist])
            if 'targetEEDist' not in param_dict['feature_names']:
                param_dict['feature_names'].append('targetEEDist')


        # Crossmodal feature - relative angle --------------------------
        if 'crossmodal_targetEEAng' in feature_list:                
            kinEEQuat    = d['kinEEQuatList'][idx]
            kinTargetQuat = d['kinTargetQuatList'][idx]

            ## kinEEPos     = d['kinEEPosList'][idx]
            ## kinTargetPos = d['kinTargetPosList'][idx]
            ## dist         = np.linalg.norm(kinTargetPos - kinEEPos, axis=0)
            
            crossmodal_targetEEAng = []
            for time_idx in xrange(len(timeList)):

                startQuat = kinEEQuat[:,time_idx]
                endQuat   = kinTargetQuat[:,time_idx]

                diff_ang = qt.quat_angle(startQuat, endQuat)
                crossmodal_targetEEAng.append( abs(diff_ang) )

            crossmodal_targetEEAng = np.array(crossmodal_targetEEAng)
            if offset_flag:
                crossmodal_targetEEAng -= np.mean(crossmodal_targetEEAng[:startOffsetSize])

            ## fig = plt.figure()
            ## ## plt.plot(crossmodal_targetEEAng)
            ## plt.plot( kinEEQuat[0] )
            ## plt.plot( kinEEQuat[1] )
            ## plt.plot( kinEEQuat[2] )
            ## plt.plot( kinEEQuat[3] )
            ## fig.savefig('test.pdf')
            ## fig.savefig('test.png')
            ## os.system('cp test.p* ~/Dropbox/HRL/')        
            ## sys.exit()
            
            if dataSample is None: dataSample = np.array(crossmodal_targetEEAng)
            else: dataSample = np.vstack([dataSample, crossmodal_targetEEAng])
            if 'targetEEAng' not in param_dict['feature_names']:
                param_dict['feature_names'].append('targetEEAng')

        # Crossmodal feature - vision relative dist with main(first) vision target----
        if 'crossmodal_artagEEDist' in feature_list:
            kinEEPos  = d['kinEEPosList'][idx]
            visionArtagPos = d['visionArtagPosList'][idx][:3] # originally length x 3*tags

            dist = np.linalg.norm(visionArtagPos - kinEEPos, axis=0)
            if offset_flag:
                dist -= np.mean(dist[:startOffsetSize])
            
            crossmodal_artagEEDist = []
            for time_idx in xrange(len(timeList)):
                crossmodal_artagEEDist.append(dist[time_idx])

            if dataSample is None: dataSample = np.array(crossmodal_artagEEDist)
            else: dataSample = np.vstack([dataSample, crossmodal_artagEEDist])
            if 'artagEEDist' not in param_dict['feature_names']:
                param_dict['feature_names'].append('artagEEDist')

        # Crossmodal feature - vision relative angle --------------------------
        if 'crossmodal_artagEEAng' in feature_list:                
            kinEEQuat    = d['kinEEQuatList'][idx]
            visionArtagQuat = d['visionArtagQuatList'][idx][:4]

            ## kinEEPos  = d['kinEEPosList'][idx]
            ## visionArtagPos = d['visionArtagPosList'][idx][:3]
            ## dist = np.linalg.norm(visionArtagPos - kinEEPos, axis=0)
            
            crossmodal_artagEEAng = []
            for time_idx in xrange(len(timeList)):

                startQuat = kinEEQuat[:,time_idx]
                endQuat   = visionArtagQuat[:,time_idx]

                diff_ang = qt.quat_angle(startQuat, endQuat)
                crossmodal_artagEEAng.append( abs(diff_ang) )

            crossmodal_artagEEAng = np.array(crossmodal_artagEEAng)
            if offset_flag:
                crossmodal_artagEEAng -= np.mean(crossmodal_artagEEAng[:startOffsetSize])

            if dataSample is None: dataSample = np.array(crossmodal_artagEEAng)
            else: dataSample = np.vstack([dataSample, crossmodal_artagEEAng])
            if 'artagEEAng' not in param_dict['feature_names']:
                param_dict['feature_names'].append('artagEEAng')


        # Crossmodal feature - vision relative dist with sub vision target----
        if 'crossmodal_subArtagEEDist' in feature_list:
            kinEEPos  = d['kinEEPosList'][idx]
            visionArtagPos = d['visionArtagPosList'][idx][3:6] # originally length x 3*tags

            dist = np.linalg.norm(visionArtagPos - kinEEPos, axis=0)
            if offset_flag:
                dist -= np.mean(dist[:startOffsetSize])
            
            crossmodal_artagEEDist = []
            for time_idx in xrange(len(timeList)):
                crossmodal_artagEEDist.append(dist[time_idx])

            if dataSample is None: dataSample = np.array(crossmodal_artagEEDist)
            else: dataSample = np.vstack([dataSample, crossmodal_artagEEDist])
            if 'subArtagEEDist' not in param_dict['feature_names']:
                param_dict['feature_names'].append('subArtagEEDist')

        # Crossmodal feature - vision relative angle --------------------------
        if 'crossmodal_subArtagEEAng' in feature_list:                
            kinEEQuat    = d['kinEEQuatList'][idx]
            visionArtagQuat = d['visionArtagQuatList'][idx][4:8]

            crossmodal_artagEEAng = []
            for time_idx in xrange(len(timeList)):

                startQuat = kinEEQuat[:,time_idx]
                endQuat   = visionArtagQuat[:,time_idx]

                diff_ang = qt.quat_angle(startQuat, endQuat)
                crossmodal_artagEEAng.append( abs(diff_ang) )

            crossmodal_artagEEAng = np.array(crossmodal_artagEEAng)
            if offset_flag:
                crossmodal_artagEEAng -= np.mean(crossmodal_artagEEAng[:startOffsetSize])

            if dataSample is None: dataSample = np.array(crossmodal_artagEEAng)
            else: dataSample = np.vstack([dataSample, crossmodal_artagEEAng])
            if 'subArtagEEAng' not in param_dict['feature_names']:
                param_dict['feature_names'].append('subArtagEEAng')


        # Crossmodal feature - vision relative dist with main(first) vision target----
        if 'crossmodal_landmarkEEDist' in feature_list:
            kinEEPos  = d['kinEEPosList'][idx]
            visionLandmarkPos = d['visionLandmarkPosList'][idx] # originally length x 3*tags

            if len(np.shape(visionLandmarkPos)) == 1:
                visionLandmarkPos = np.reshape(visionLandmarkPos, (3,1))
                
            ## dist = np.linalg.norm(visionLandmarkPos, axis=0)            
            dist = np.linalg.norm(visionLandmarkPos - kinEEPos, axis=0)
            if offset_flag:
                dist -= np.mean(dist[:startOffsetSize])
            
            crossmodal_landmarkEEDist = []
            for time_idx in xrange(len(timeList)):
                crossmodal_landmarkEEDist.append(dist[time_idx])

            if dataSample is None: dataSample = np.array(crossmodal_landmarkEEDist)
            else: dataSample = np.vstack([dataSample, crossmodal_landmarkEEDist])
            if 'landmarkEEDist' not in param_dict['feature_names']:
                param_dict['feature_names'].append('landmarkEEDist')


        # Crossmodal feature - vision relative angle --------------------------
        if 'crossmodal_landmarkEEAng' in feature_list:
            kinEEQuat    = d['kinEEQuatList'][idx]
            visionLandmarkQuat = d['visionLandmarkQuatList'][idx][:4]
            visionLandmarkPos  = d['visionLandmarkPosList'][idx] # originally length x 3*tags
            if len(np.shape(visionLandmarkPos)) == 1:
                visionLandmarkQuat = np.reshape(visionLandmarkQuat, (4,1))

            crossmodal_landmarkEEAng = []
            for time_idx in xrange(len(timeList)):

                startQuat = kinEEQuat[:,time_idx]
                endQuat   = visionLandmarkQuat[:,time_idx]

                diff_ang = qt.quat_angle(startQuat, endQuat)
                crossmodal_landmarkEEAng.append( abs(diff_ang) )

            crossmodal_landmarkEEAng = np.array(crossmodal_landmarkEEAng)
            if offset_flag:
                crossmodal_landmarkEEAng -= np.mean(crossmodal_landmarkEEAng[:startOffsetSize])
            crossmodal_landmarkEEAng = abs(crossmodal_landmarkEEAng)

            if dataSample is None: dataSample = np.array(crossmodal_landmarkEEAng)
            else: dataSample = np.vstack([dataSample, crossmodal_landmarkEEAng])
            if 'landmarkEEAng' not in param_dict['feature_names']:
                param_dict['feature_names'].append('landmarkEEAng')


        # ----------------------------------------------------------------
        if len(np.shape(dataSample)) < 2: dataSample = np.array([dataSample])
        dataList.append(dataSample)


    # Convert data structure 
    # From nSample x dim x length
    # To dim x nSample x length
    nSample      = len(dataList)
    nEmissionDim = len(dataList[0])
    features = np.swapaxes(dataList, 0, 1)    

    # cut unnecessary part #temp
    if cut_data is not None:
        features = features[:,:,cut_data[0]:cut_data[1]]

    # Scaling ------------------------------------------------------------
    if isTrainingData or renew_minmax:
        param_dict['feature_max'] = [ np.max(np.array(feature).flatten()) for feature in features ]
        param_dict['feature_min'] = [ np.min(np.array(feature).flatten()) for feature in features ]

        # find feature
        for feature_name in ['ftForce_mag_integ', 'ftForce_mag_zero']:
            if feature_name in param_dict['feature_names']:
                idx = param_dict['feature_names'].index(feature_name)
                # split success
                success_idx = d['success_idx_list']
                # update min/max
                param_dict['feature_max'][idx] = np.max(np.array(features[idx][success_idx]).flatten())
                param_dict['feature_min'][idx] = np.min(np.array(features[idx][success_idx]).flatten())

    scaled_features = []
    for i, feature in enumerate(features):
        if abs( param_dict['feature_max'][i] - param_dict['feature_min'][i]) < 1e-3:
            scaled_features.append( np.array(feature) )
        else:
            scaled_features.append( ( np.array(feature) - param_dict['feature_min'][i] )\
                                    /( param_dict['feature_max'][i] - param_dict['feature_min'][i]) )

    return scaled_features, param_dict


def extractRawFeature(d, raw_feature_list, nSuccess, nFailure, param_dict=None, \
                      cut_data=None, verbose=False, scaling=True):

    from sandbox_dpark_darpa_m3.lib import hrl_dh_lib as dh
    from hrl_lib import quaternion as qt
    
    if param_dict is None:
        isTrainingData=True
        param_dict = {}
    else:
        isTrainingData=False
            
    # -------------------------------------------------------------        
    # extract modality data
    dataList = []
    dataDim  = []
    nSample  = len(d['timesList'])
    startOffsetSize = 4
    for idx in xrange(nSample): # each sample

        timeList     = d['timesList'][idx]
        dataSample = None

        # rightEE-leftEE - relative dist ----
        if 'relativePose_target_EE' in raw_feature_list:
            kinEEPos      = d['kinEEPosList'][idx]
            kinEEQuat     = d['kinEEQuatList'][idx]
            kinTargetPos  = d['kinTargetPosList'][idx]
            kinTargetQuat = d['kinTargetQuatList'][idx]

            # pos and quat?
            relativePose = []
            for time_idx in xrange(len(timeList)):
                startFrame = dh.array2KDLframe( kinTargetPos[:,time_idx].tolist() +\
                                                kinTargetQuat[:,time_idx].tolist() )
                endFrame   = dh.array2KDLframe( kinEEPos[:,time_idx].tolist()+\
                                                kinEEQuat[:,time_idx].tolist() )
                diffFrame  = endFrame*startFrame.Inverse()                                
                relativePose.append( dh.KDLframe2List(diffFrame) )

            relativePose = np.array(relativePose).T[:-1]
            relativePose[:3,:] -= np.array([np.mean(relativePose[:,:startOffsetSize], axis=1)[:3]]).T
            
            if dataSample is None: dataSample = relativePose
            else: dataSample = np.vstack([dataSample, relativePose])
            if idx == 0: dataDim.append(['relativePos_target_EE', 3])
            if idx == 0: dataDim.append(['relativeAng_target_EE', 4])
                

        # main-artag EE - vision relative dist with main(first) vision target----
        if 'relativePose_artag_EE' in raw_feature_list:
            kinEEPos        = d['kinEEPosList'][idx]
            kinEEQuat       = d['kinEEQuatList'][idx]
            visionArtagPos  = d['visionArtagPosList'][idx][:3] # originally length x 3*tags
            visionArtagQuat = d['visionArtagQuatList'][idx][:4] # originally length x 3*tags

            # pos and quat?
            relativePose = []
            for time_idx in xrange(len(timeList)):
                startFrame = dh.array2KDLframe( visionArtagPos[:,time_idx].tolist() +\
                                                visionArtagQuat[:,time_idx].tolist() )
                endFrame   = dh.array2KDLframe( kinEEPos[:,time_idx].tolist()+\
                                                kinEEQuat[:,time_idx].tolist() )
                diffFrame  = endFrame*startFrame.Inverse()                                
                relativePose.append( dh.KDLframe2List(diffFrame) )
            
            relativePose = np.array(relativePose).T[:-1]
            relativePose[:3,:] -= np.array([np.mean(relativePose[:,:startOffsetSize], axis=1)[:3]]).T
            
            if dataSample is None: dataSample = relativePose
            else: dataSample = np.vstack([dataSample, relativePose])
            if idx == 0: dataDim.append(['relativePos_artag_EE', 3])
            if idx == 0: dataDim.append(['relativeAng_artag_EE', 4])
                

        # main-artag sub-artag - vision relative dist with main(first) vision target----
        if 'relativePose_artag_artag' in raw_feature_list:
            visionArtagPos1 = d['visionArtagPosList'][idx][:3] # originally length x 3*tags
            visionArtagQuat1 = d['visionArtagQuatList'][idx][:4] # originally length x 3*tags
            visionArtagPos2 = d['visionArtagPosList'][idx][3:6] # originally length x 3*tags
            visionArtagQuat2 = d['visionArtagQuatList'][idx][4:8] # originally length x 3*tags

            # pos and quat?
            relativePose = []
            for time_idx in xrange(len(timeList)):

                startFrame = dh.array2KDLframe( visionArtagPos1[:,time_idx].tolist() +\
                                                visionArtagQuat1[:,time_idx].tolist() )
                endFrame = dh.array2KDLframe( visionArtagPos2[:,time_idx].tolist() +\
                                              visionArtagQuat2[:,time_idx].tolist() )                
                diffFrame  = endFrame*startFrame.Inverse()                                
                relativePose.append( dh.KDLframe2List(diffFrame) )

            relativePose = np.array(relativePose).T[:-1]
            relativePose[:3,:] -= np.mean(relativePose[:,:startOffsetSize], axis=1)[:3,:]

            if dataSample is None: dataSample = relativePose
            else: dataSample = np.vstack([dataSample, relativePose])
            if idx == 0: dataDim.append(['relativePos_artag_artag', 3])
            if idx == 0: dataDim.append(['relativeAng_artag_artag', 4])

        # Audio --------------------------------------------
        if 'kinectAudio' in raw_feature_list:
            audioPower   = d['audioPowerList'][idx]                        
            if dataSample is None: dataSample = copy.copy(np.array(audioPower))
            else: dataSample = np.vstack([dataSample, copy.copy(audioPower)])
            if idx == 0: dataDim.append(['kinectAudio', len(audioPower)])

        # AudioWrist ---------------------------------------
        if 'wristAudio' in raw_feature_list:
            audioWristRMS  = d['audioWristRMSList'][idx]
            ## audioWristMFCC = d['audioWristMFCCList'][idx]            

            if dataSample is None: dataSample = copy.copy(np.array(audioWristRMS))
            else: dataSample = np.vstack([dataSample, copy.copy(audioWristRMS)])
            ## dataSample = np.vstack([dataSample, copy.copy(audioWristMFCC)])
            
            if idx == 0: dataDim.append(['wristAudio_RMS', 1])                
            ## if idx == 0: dataDim.append(['wristAudio_MFCC', len(audioWristMFCC)])                

        # FT -------------------------------------------
        if 'ft' in raw_feature_list:
            ftForce  = d['ftForceList'][idx]
            ftTorque = d['ftTorqueList'][idx]

            ftForce  -= np.array([np.mean(ftForce[:startOffsetSize,:], axis=1)]).T
            ftTorque -= np.array([np.mean(ftTorque[:startOffsetSize,:], axis=1)]).T

            if dataSample is None: dataSample = np.array(ftForce)
            else: dataSample = np.vstack([dataSample, ftForce])

            if dataSample is None: dataSample = np.array(ftTorque)
            else: dataSample = np.vstack([dataSample, ftTorque])
            if idx == 0: dataDim.append(['ft_force', len(ftForce)])
            if idx == 0: dataDim.append(['ft_torque', len(ftTorque)])

        # pps -------------------------------------------
        if 'pps' in raw_feature_list:
            ppsLeft  = d['ppsLeftList'][idx]
            ppsRight = d['ppsRightList'][idx]

            ppsLeft  -= np.array([np.mean(ppsLeft[:startOffsetSize,:], axis=1)]).T
            ppsRight -= np.array([np.mean(ppsRight[:startOffsetSize,:], axis=1)]).T

            if dataSample is None: dataSample = ppsLeft
            else: dataSample = np.vstack([dataSample, ppsLeft])

            if dataSample is None: dataSample = ppsRight
            else: dataSample = np.vstack([dataSample, ppsRight])
            if idx == 0: dataDim.append(['pps', len(ppsLeft)+len(ppsRight)])

        # Kinematics --------------------------------------
        if 'kinematics' in raw_feature_list:
            kinEEPos   = d['kinEEPosList'][idx]
            kinEEQuat  = d['kinEEQuatList'][idx]
            kinJntPos  = d['kinJntPosList'][idx]
            kinPos     = d['kinPosList'][idx]
            kinVel     = d['kinVelList'][idx]

            if dataSample is None: dataSample = np.array(kinEEPos)
            else: dataSample = np.vstack([dataSample, kinEEPos])
            if 'kinEEPos_x' not in param_dict['feature_names']:
                param_dict['feature_names'].append('kinEEPos_x')
                param_dict['feature_names'].append('kinEEPos_y')
                param_dict['feature_names'].append('kinEEPos_z')

            if dataSample is None: dataSample = np.array(kinEEQuat)
            else: dataSample = np.vstack([dataSample, kinEEQuat])
            if 'kinEEQuat_x' not in param_dict['feature_names']:
                param_dict['feature_names'].append('kinEEQuat_x')
                param_dict['feature_names'].append('kinEEQuat_y')
                param_dict['feature_names'].append('kinEEQuat_z')
                param_dict['feature_names'].append('kinEEQuat_w')

            if dataSample is None: dataSample = np.array(kinJntPos)
            else: dataSample = np.vstack([dataSample, kinJntPos])
            if 'kinJntPos_1' not in param_dict['feature_names']:
                param_dict['feature_names'].append('kinJntPos_1')
                param_dict['feature_names'].append('kinJntPos_2')
                param_dict['feature_names'].append('kinJntPos_3')
                param_dict['feature_names'].append('kinJntPos_4')
                param_dict['feature_names'].append('kinJntPos_5')
                param_dict['feature_names'].append('kinJntPos_6')
                param_dict['feature_names'].append('kinJntPos_7')

            if dataSample is None: dataSample = np.array(kinPos)
            else: dataSample = np.vstack([dataSample, kinPos])
            if 'kinPos_x' not in param_dict['feature_names']:
                param_dict['feature_names'].append('kinPos_x')
                param_dict['feature_names'].append('kinPos_y')
                param_dict['feature_names'].append('kinPos_z')

            if dataSample is None: dataSample = np.array(kinVel)
            else: dataSample = np.vstack([dataSample, kinVel])
            if 'kinVel_x' not in param_dict['feature_names']:
                param_dict['feature_names'].append('kinVel_x')
                param_dict['feature_names'].append('kinVel_y')
                param_dict['feature_names'].append('kinVel_z')
                

        ## # Unimodal feature - vision change ------------------------------------
        ## if 'unimodal_visionChange' in raw_feature_list:
        ##     visionChangeMag = d['visionChangeMagList'][idx]

        ##     unimodal_visionChange = visionChangeMag

        ##     if dataSample is None: dataSample = unimodal_visionChange
        ##     else: dataSample = np.vstack([dataSample, unimodal_visionChange])
        ##     if 'visionChange' not in param_dict['feature_names']:
        ##         param_dict['feature_names'].append('visionChange')
                
        ## # Unimodal feature - fabric skin ------------------------------------
        ## if 'unimodal_fabricForce' in raw_feature_list:
        ##     fabricMag = d['fabricMagList'][idx]

        ##     unimodal_fabricForce = fabricMag

        ##     if dataSample is None: dataSample = unimodal_fabricForce
        ##     else: dataSample = np.vstack([dataSample, unimodal_fabricForce])
        ##     if 'fabricForce' not in param_dict['feature_names']:
        ##         param_dict['feature_names'].append('fabricForce')

        # ----------------------------------------------------------------
        dataList.append(dataSample)

    # Augmentation -------------------------------------------------------
    # assuming there is no currupted file    
    assert len(dataList) == nSuccess+nFailure
    successDataList = dataList[0:nSuccess]
    failureDataList = dataList[nSuccess:]
    allDataList     = successDataList + failureDataList

    # Converting data structure & cutting unnecessary part ---------------
    features         = np.swapaxes(allDataList, 0, 1)
    success_features = np.swapaxes(successDataList, 0, 1)
    failure_features = np.swapaxes(failureDataList, 0, 1)

    ## Cut data
    if cut_data is not None:
        features         = features[:,:,cut_data[0]:cut_data[1]]
        success_features = success_features[:,:,cut_data[0]:cut_data[1]]
        failure_features = failure_features[:,:,cut_data[0]:cut_data[1]]
               
    # Scaling ------------------------------------------------------------
    if isTrainingData:
        param_dict['feature_max'] = [ np.max(np.array(feature).flatten()) for feature in features ]
        param_dict['feature_min'] = [ np.min(np.array(feature).flatten()) for feature in features ]
        param_dict['feature_mu']  = [ np.mean(np.array(feature).flatten()) for feature in features ]
        param_dict['feature_std'] = [ np.std(np.array(feature).flatten()) for feature in features ]
        ## print "max: ", param_dict['feature_max']
        ## print "min: ", param_dict['feature_min']

    if scaling is True: 
        success_features = scale( success_features, param_dict['feature_min'], param_dict['feature_max'] )
        failure_features = scale( failure_features, param_dict['feature_min'], param_dict['feature_max'] )
        ## success_features = normalization( success_features, param_dict['feature_mu'], \
        ## param_dict['feature_std'] )
        ## failure_features = normalization( failure_features, param_dict['feature_mu'], \
        ## param_dict['feature_std'] )

    param_dict['feature_names'] = raw_feature_list
    param_dict['dataDim']       = dataDim
   
    return success_features, failure_features, param_dict


def export_images(folder_list, data_dict, downSampleSize):
    ''' Get list of images between start and end time '''

    assert len(data_dict['timesList']) == len(folder_list)

    images = []
    for idx, f in enumerate(folder_list):

        des_time_list = data_dict['timesList'][idx]

        # get image folder
        files = os.listdir(f)
        if len(files) == 0:
            print "No images so skip: ", f
            images.append(None)
            continue            

        # get time list
        time_list = []
        for i, img in enumerate(files):
            if not(img.find('.jpg')>=0): continue
            t = float(img.split('_')[-1].split('.jpg')[0])
            time_list.append(t-data_dict['initTimeList'][idx])
        time_list = np.array(time_list)

        # Note: do we need to sort? probably not

        t_idx_list = []
        for t in des_time_list:
            t_idx = np.argmin(abs(time_list - t))
            t_idx_list.append(t_idx)

        # get list of iamges between start_time and end_time
        imgs = [os.path.join(f,files[i]) for i in t_idx_list ]
        images.append(imgs)

    return images

#-------------------------------------------------------------------------------------------------

def normalization(x, mu, std):
    new_x = copy.copy(x)
    for i in xrange(len(x)):
        new_x[i] = (x[i]-mu[i])/std[i]
    return new_x

def scale(x, x_min, x_max):
    '''
    scale data between 0 and 1
    '''
    new_x = copy.copy(x)
    for i in xrange(len(x)):
        new_x[i] = (x[i]-x_min[i])/(x_max[i]-x_min[i])
    return new_x
    

## def changeDataStructure(dataList):
##     '''
##     From nSample x dim x length to dim x nSample x length
##     or
##     From dim x nSample x length to nSample x dim x length 
##     '''
    
##     n = len(dataList)
##     m = len(dataList[0])
##     features     = []
##     for i in xrange(m):
##         feature  = []

##         for j in xrange(n):
##             try:
##                 feature.append(dataList[j][i,:])
##             except:
##                 print "Failed to cut data", j,i, np.shape(dataList[j]), dataList[j][i]
##                 print np.shape(dataList), np.shape(dataList[j]), j, i
##                 sys.exit()

##         features.append( feature )

def data_augmentation(successes, failures, nAugment=1):

    '''
    nSamples x Dim x nLength
    '''
    c_scale  = [0.8, 1.2]
    c_shift  = [-10, 10]
    c_noise  = 20.0 # constant computing noise sgd, sample_std/constant
    c_filter = []

    nDim     = len(successes[0])
    np.random.seed(1342)

    if nAugment == 0: return successes, failures

    # for each sample
    for k in xrange(2):
        
        aug_data_list = []
        
        for x in [successes, failures][k]:
            
            # x is numpy 2D array
            for n in xrange(nAugment):

                # scaling (selective dim)
                ## idx_list = np.random.randint(0, 2, size=nDim)
                ## new_x = None
                ## for i, flag in zip( range(nDim), idx_list ):
                ##     if flag == 0: temp = x[i:i+1]
                ##     else: temp = x[i:i+1] * np.random.uniform(c_scale[0], c_scale[1])

                ##     if len(np.shape(temp)) == 1: temp = np.array([temp])

                ##     if new_x is None: new_x = temp
                ##     else: new_x = np.vstack([new_x, temp])

                ## aug_data_list.append(new_x)


                ## # shifting (selective dim)
                ## idx_list = np.random.randint(0, 2, size=nDim)
                ## new_x = None
                ## for i, flag in zip( range(nDim), idx_list ):
                ##     if flag == 0:
                ##         temp = x[i:i+1]
                ##     else:
                ##         shift = np.random.random_integers(c_shift[0], c_shift[1])
                ##         if shift >= 0:
                ##             temp = np.hstack([x[i][shift:], [x[i][-1]]*shift])
                ##         else:
                ##             temp = np.hstack([[x[i][0]]*abs(shift), x[i][:-abs(shift)]])

                ##     if len(np.shape(temp)) == 1: temp = np.array([temp])

                ##     if new_x is None: new_x = temp
                ##     else: new_x = np.vstack([new_x, temp])

                ## aug_data_list.append(new_x)

                # noise (all or selectively)
                idx_list = np.random.randint(0, 2, size=nDim)
                new_x = None
                for i, flag in zip( range(nDim), idx_list ):
                    if flag == 0: temp = x[i:i+1]
                    else: temp = x[i:i+1] + np.random.normal(0.0, np.std(x[i])/c_noise, len(x[i]))

                    if len(np.shape(temp)) == 1: temp = np.array([temp])

                    if new_x is None: new_x = temp
                    else: new_x = np.vstack([new_x, temp])

                aug_data_list.append(new_x)
                

                # filtering
                
        if k==0:
            if type(successes) == list: success_aug_list = successes + aug_data_list
            else:  success_aug_list = successes.tolist() + aug_data_list
        else:
            if type(failures) == list: failure_aug_list = failures + aug_data_list
            else: failure_aug_list = failures.tolist() + aug_data_list

    ## print "data auuuuuuuuuuuugmentation"
    ## print "From : ", np.shape(successes), np.shape(failures)
    ## print "To : ", np.shape(success_aug_list), np.shape(failure_aug_list)
    
    return np.array(success_aug_list), np.array(failure_aug_list)


def get_time_window_data(subject_names, task, raw_data_path, processed_data_path, save_pkl, \
                         rf_center, local_range, downSampleSize, time_window, handFeatures, rawFeatures, \
                         cut_data, nAugment=1, renew=False):

    if os.path.isfile(save_pkl) and renew is not True:
        d = ut.load_pickle(save_pkl)
        # Time-sliding window
        new_normalTrainingData   = getTimeDelayData( d['normalTrainingData'], time_window )
        new_abnormalTrainingData = getTimeDelayData( d['abnormalTrainingData'], time_window )        
        new_normalTestData       = getTimeDelayData( d['normalTestData'], time_window )
        new_abnormalTestData     = getTimeDelayData( d['abnormalTestData'], time_window )        
        nSingleData              = len(d['normalTestData'][0][0])-time_window+1

        return new_normalTrainingData, new_normalTrainingData, new_normalTestData, new_abnormalTestData, \
          nSingleData

    # dim x sample x length
    data_dict = getDataSet(subject_names, task, raw_data_path, processed_data_path, \
                           rf_center, local_range,\
                           downSampleSize=downSampleSize, scale=1.0,\
                           ae_data=True, \
                           handFeatures=handFeatures, rawFeatures=rawFeatures, \
                           cut_data=cut_data,\
                           data_renew=renew)
    successData = data_dict['aeSuccessData']
    failureData = data_dict['aeFailureData']
                           

    # index selection
    ratio        = 0.8
    success_idx  = range(len(successData[0]))
    failure_idx  = range(len(failureData[0]))

    s_train_idx  = random.sample(success_idx, int( ratio*len(success_idx)) )
    f_train_idx  = random.sample(failure_idx, int( ratio*len(failure_idx)) )
    
    s_test_idx = [x for x in success_idx if not x in s_train_idx]
    f_test_idx = [x for x in failure_idx if not x in f_train_idx]

    # data structure: dim x sample x sequence
    normalTrainingData   = successData[:, s_train_idx, :]
    abnormalTrainingData = failureData[:, f_train_idx, :]
    normalTestData       = successData[:, s_test_idx, :]
    abnormalTestData     = failureData[:, f_test_idx, :]

    # scaling by the number of dimensions in each feature
    # nSamples x Dim x nLength
    d = {}        
    d['normalTrainingData']   = np.swapaxes(normalTrainingData, 0, 1)
    d['abnormalTrainingData'] = np.swapaxes(abnormalTrainingData, 0, 1)
    d['normalTestData']       = np.swapaxes(normalTestData, 0, 1)
    d['abnormalTestData']     = np.swapaxes(abnormalTestData, 0, 1)
    ut.save_pickle(d, save_pkl)

    # data augmentation for auto encoder
    if nAugment>0:
        normalTrainDataAug, abnormalTrainDataAug = data_augmentation(d['normalTrainingData'], \
                                                                     d['abnormalTrainingData'], nAugment)
    else:
        normalTrainDataAug   = d['normalTrainData']
        abnormalTrainDataAug = d['abnormalTrainData']


    print "======================================"
    print "nSamples x Dim x nLength"
    print "--------------------------------------"
    print "Normal Train data: ",   np.shape(d['normalTrainingData'])
    print "Abnormal Train data: ", np.shape(d['abnormalTrainingData'])
    print "Normal test data: ",    np.shape(d['normalTestData'])
    print "Abnormal test data: ",  np.shape(d['abnormalTestData'])
    print "======================================"

    # Time-sliding window
    # sample x time_window_flatten_length
    new_normalTrainingData   = getTimeDelayData( d['normalTrainingData'], time_window )
    new_abnormalTrainingData = getTimeDelayData( d['abnormalTrainingData'], time_window )
    new_normalTestData       = getTimeDelayData( d['normalTestData'], time_window )
    new_abnormalTestData     = getTimeDelayData( d['abnormalTestData'], time_window )
    nSingleData       = len(d['normalTestData'][0][0])-time_window+1

    # sample x dim
    return new_normalTrainingData, new_abnormalTrainingData, \
      new_normalTestData, new_abnormalTestData, nSingleData


def getTimeDelayData(data, time_window):
    '''
    Input size is sample x dim x length.
    Output size is sample x time_window_flatten_length.
    '''
    new_data = []
    for i in xrange(len(data)):
        for j in xrange(len(data[i][0])-time_window+1):
            new_data.append( data[i][:,j:j+time_window].flatten() )

    return np.array(new_data)

def getEstTruePositive(ll_X, ll_idx=None, nOffset=5):
    '''
    Input size is Samples x length x HMM-induced features or 
    Input size is length x HMM-induced features
    Output is nData x HMM-induced features
    Here the input should be positive data only.
    '''

    flatten_X   = []
    flatten_idx = []
    
    if len(np.shape(ll_X))==3:
        for i in xrange(len(ll_X)):
            for j in xrange(0, len(ll_X[i])-nOffset):
                if ll_X[i][j+nOffset][0]-ll_X[i][j][0] < 0 : #and X[i][j+1][0]-X[i][j][0] < 0:
                    flatten_X   += ll_X[i][j:]
                    if ll_idx is not None: flatten_idx += ll_idx[i][j] # if thee is no likelihood drop?
                    break
    elif len(np.shape(ll_X))==2:
        if ll_idx is not None: flatten_idx = ll_idx[-1]
        for j in xrange(0, len(ll_X)-nOffset):
            if ll_X[j+nOffset][0]-ll_X[j][0] < 0 : #and X[j+1][0]-X[j][0] < 0:
                if type(ll_X[j:]) is list:
                    flatten_X += ll_X[j:]
                else:
                    flatten_X += ll_X[j:].tolist()
                if ll_idx is not None: flatten_idx = ll_idx[j]
                break
    else:
        warnings.warn("Not available dimension of data X")

    flatten_Y = [1]*len(flatten_X)

    if ll_idx is None:
        return flatten_X, flatten_Y
    else:
        return flatten_X, flatten_Y, flatten_idx        
        
        
def flattenSample(ll_X, ll_Y, ll_idx=None, remove_fp=False):
    '''
    ll : sample x length x hmm features
    l  : sample...  x hmm features
    '''

    if remove_fp:

        pos_idx = []
        neg_idx = []        
        for i in xrange(len(ll_X)):
            if ll_Y[i][0] < 0:
                neg_idx.append(i)
            else:
                pos_idx.append(i)

        # sample x length
        ## ll_pos_X = np.array(ll_X)[pos_idx,:,0]
        ll_neg_X = np.array(ll_X)[neg_idx,:,0]

        # logp distribution
        means = np.mean(ll_neg_X, axis=0)
        stds = np.std(ll_neg_X, axis=0)
        
        l_X   = []
        l_Y   = []
        l_idx = []
        for i in xrange(len(ll_X)):
            if ll_Y[i][0] < 0:
                if type(ll_X[i]) is list:
                    l_X += ll_X[i]
                else:
                    l_X += ll_X[i].tolist()

                if type(ll_Y[i]) is list:
                    l_Y += ll_Y[i]
                else:
                    l_Y += ll_Y[i].tolist()
            else:
                for j in xrange(len(ll_X[i])):
                    if ll_X[i][j][0] < means[j]-1.0*stds[j]-10:
                        break
                ## X,Y = getEstTruePositive(ll_X[i])
                l_X += np.array(ll_X)[i][j:].tolist()
                l_Y += np.array(ll_Y)[i][j:].tolist()
        if len(np.shape(l_X))==1:
            warnings.warn("wrong size vector in flatten function")
            sys.exit()
                
    else:
        l_X = []
        l_Y = []
        l_idx = []
        for i in xrange(len(ll_X)):
            for j in xrange(len(ll_X[i])):
                l_X.append(ll_X[i][j])
                l_Y.append(ll_Y[i][j])
                if ll_idx is not None:
                    l_idx.append(ll_idx[i][j])

    return l_X, l_Y, l_idx
    
def flattenSampleWithWindow(ll_X, ll_Y, ll_idx=None, window=2):
    '''
    ll : sample x length x features
    l  : sample...  x features
    '''

    if window < 2:
        print "Wrong window size"
        sys.exit()

    l_X = []
    l_Y = []
    l_idx = []
    for i in xrange(len(ll_X)):
        for j in xrange(len(ll_X[i])):

            X = []
            for k in range(window,0,-1):
                if j-k < 0:
                    if type(ll_X[i][0]) is not list: X += ll_X[i][0].tolist()
                    else: X += ll_X[i][0]
                else:
                    if type(ll_X[i][j-k]) is not list: X += ll_X[i][j-k].tolist()
                    else: X += ll_X[i][j-k]

            l_X.append(X)
            l_Y.append(ll_Y[i][j])
            if ll_idx is not None:
                l_idx.append(ll_idx[i][j])

    return l_X, l_Y, l_idx

def sampleWithWindow(ll_X, window=2):
    '''
    ll : sample x length x features
    '''
    if window < 2:
        print "Wrong window size"
        sys.exit()

    ll_X_new = []
    for i in xrange(len(ll_X)):
        l_X_new = []
        for j in xrange(len(ll_X[i])):

            X = []
            for k in range(window,0,-1):
                if j-k < 0:
                    X+= ll_X[i][0].tolist()
                else:
                    X+= ll_X[i][j-k].tolist()

            l_X_new.append(X)
        ll_X_new.append(l_X_new)

    return ll_X_new


def subsampleData(X,Y,idx=None, nSubSample=40, nMaxData=50, startIdx=4, rnd_sample=False):

    import random

    sample_id_list = range(len(X))
    if len(X)*nSubSample > nMaxData*nSubSample:
        print "Too many training data, so we resample!!"
        sample_id_list = range(len(X))
        random.shuffle(sample_id_list)
        sample_id_list = sample_id_list[:nMaxData]

        X = np.array(X)[sample_id_list]
        Y = np.array(Y)[sample_id_list]
        if idx is not None:
            idx = np.array(idx)[sample_id_list]

    print "before: ", np.shape(X), np.shape(Y)
    new_X = []
    new_Y = []
    new_idx = []
    for i in xrange(len(X)):
        if rnd_sample is False:
            idx_list = np.linspace(startIdx, len(X[i])-1, nSubSample).astype(int)
            new_X.append( np.array(X)[i,idx_list].tolist() )
            new_Y.append( np.array(Y)[i,idx_list].tolist() )
            if idx is not None:
                new_idx.append( np.array(idx)[i,idx_list].tolist() )
        else:
            idx_list = range(len(X[i]))
            random.shuffle(idx_list)
            new_X.append( np.array(X)[i,idx_list[:nSubSample]].tolist() )
            new_Y.append( np.array(Y)[i,idx_list[:nSubSample]].tolist() )
            if idx is not None:            
                new_idx.append( np.array(idx)[i,idx_list[:nSubSample]].tolist() )

    return new_X, new_Y, new_idx


    
def applying_offset(data, normalTrainData, startOffsetSize, nEmissionDim):

    # get offset
    refData = np.reshape( np.mean(normalTrainData[:,:,:startOffsetSize], axis=(1,2)), \
                          (nEmissionDim,1,1) ) # 4,1,1

    curData = np.reshape( np.mean(data[:,:,:startOffsetSize], axis=(1,2)), \
                          (nEmissionDim,1,1) ) # 4,1,1
    offsetData = refData - curData

    for i in xrange(nEmissionDim):
        data[i] = (np.array(data[i]) + offsetData[i][0][0]).tolist()

    return data


def saveHMMinducedFeatures(kFold_list, successData, failureData,\
                           task_name, processed_data_path,\
                           HMM_dict, data_renew, startIdx, nState, cov, \
                           success_files=None, failure_files=None,\
                           noise_mag = 0.03, one_class=True, suffix=None, n_jobs=-1,\
                           add_logp_d=False, diag=False, cov_type='full', verbose=False):
    """
    Training HMM, and getting classifier training and testing data.
    """
    
    from hrl_anomaly_detection.hmm import learning_hmm as hmm

    if type(successData) is list:
        successData = np.array(successData)
        failureData = np.array(failureData)

    # HMM-induced vector with LOPO
    for idx, (normalTrainIdx, abnormalTrainIdx, normalTestIdx, abnormalTestIdx) \
      in enumerate(kFold_list):
        if verbose: print "dm.saveHMM ", idx

        # Training HMM, and getting classifier training and testing data
        if suffix is not None:
            modeling_pkl = os.path.join(processed_data_path, 'hmm_'+task_name+'_'+str(idx)+'_c'+suffix+'.pkl')
        else:
            modeling_pkl = os.path.join(processed_data_path, 'hmm_'+task_name+'_'+str(idx)+'.pkl')
            
        if not (os.path.isfile(modeling_pkl) is False or HMM_dict['renew'] or data_renew):
            print idx, " : learned hmm exists"
            continue

        # dim x sample x length
        normalTrainData   = copy.copy(successData[:, normalTrainIdx, :]) * HMM_dict['scale']
        abnormalTrainData = copy.copy(failureData[:, abnormalTrainIdx, :]) * HMM_dict['scale'] 
        normalTestData    = copy.copy(successData[:, normalTestIdx, :]) * HMM_dict['scale'] 
        abnormalTestData  = copy.copy(failureData[:, abnormalTestIdx, :]) * HMM_dict['scale'] 
        if one_class: abnormalTrainData = None

        # training hmm
        if verbose: print "start to fit hmm"
        nEmissionDim = len(normalTrainData)
        nLength      = len(normalTrainData[0][0]) - startIdx
        cov_mult     = [cov]*(nEmissionDim**2)

        ml  = hmm.learning_hmm(nState, nEmissionDim, verbose=verbose)
        ret = ml.fit(normalTrainData+\
                     np.random.normal(0.0, noise_mag, np.shape(normalTrainData) )*HMM_dict['scale'], \
                     cov_mult=cov_mult, use_pkl=False, cov_type=cov_type)
                     ## np.random.normal(0.0, noise_mag, np.shape(normalTrainData) )*1.0, \
        if ret == 'Failure' or np.isnan(ret):
            print "hmm training failed"
            sys.exit()

        if verbose: print "Start to extract features "
        # Classifier training data
        ll_classifier_train_X, ll_classifier_train_Y, ll_classifier_train_idx =\
          hmm.getHMMinducedFeaturesFromRawFeatures(ml, normalTrainData, abnormalTrainData, \
                                                   startIdx, add_logp_d, n_jobs=n_jobs)
        # Classifier test data
        ll_classifier_test_X, ll_classifier_test_Y, ll_classifier_test_idx =\
          hmm.getHMMinducedFeaturesFromRawFeatures(ml, normalTestData, abnormalTestData, \
                                                   startIdx, add_logp_d, n_jobs=n_jobs)

        if success_files is not None:
            ll_classifier_test_labels = [success_files[i] for i in normalTestIdx]
            ll_classifier_test_labels += [failure_files[i] for i in abnormalTestIdx]
        else:
            ll_classifier_test_labels = None


        #-----------------------------------------------------------------------------------------
        # Diagonal co-variance
        #-----------------------------------------------------------------------------------------
        if diag:
            ml  = hmm.learning_hmm(nState, nEmissionDim, verbose=verbose) 
            ret = ml.fit(normalTrainData+\
                         np.random.normal(0.0, noise_mag, np.shape(normalTrainData) )*HMM_dict['scale'], \
                         cov_mult=cov_mult, use_pkl=False, cov_type='diag')
            if ret == 'Failure' or np.isnan(ret): sys.exit()

            # Classifier training data
            ll_classifier_diag_train_X, ll_classifier_diag_train_Y, ll_classifier_diag_train_idx =\
              hmm.getHMMinducedFeaturesFromRawFeatures(ml, normalTrainData, abnormalTrainData, startIdx, \
                                                       add_logp_d, n_jobs=n_jobs, \
                                                       cov_type='diag')

            # Classifier test data
            ll_classifier_diag_test_X, ll_classifier_diag_test_Y, ll_classifier_diag_test_idx =\
              hmm.getHMMinducedFeaturesFromRawFeatures(ml, normalTestData, abnormalTestData, startIdx, \
                                                       add_logp_d, n_jobs=n_jobs,\
                                                       cov_type='diag')


        #-----------------------------------------------------------------------------------------
        d = {}
        d['nEmissionDim'] = ml.nEmissionDim
        d['A']            = ml.A 
        d['B']            = ml.B 
        d['pi']           = ml.pi
        d['F']            = ml.F
        d['nState']       = nState
        d['startIdx']     = startIdx
        d['ll_classifier_train_X']  = ll_classifier_train_X
        d['ll_classifier_train_Y']  = ll_classifier_train_Y            
        d['ll_classifier_train_idx']= ll_classifier_train_idx
        d['ll_classifier_test_X']   = ll_classifier_test_X
        d['ll_classifier_test_Y']   = ll_classifier_test_Y            
        d['ll_classifier_test_idx'] = ll_classifier_test_idx
        d['ll_classifier_test_labels'] = ll_classifier_test_labels
        d['nLength']      = nLength
        d['scale']        = HMM_dict['scale']
        d['cov']          = HMM_dict['cov']
        if diag:
            d['ll_classifier_diag_train_X']  = ll_classifier_diag_train_X
            d['ll_classifier_diag_train_Y']  = ll_classifier_diag_train_Y            
            d['ll_classifier_diag_train_idx']= ll_classifier_diag_train_idx
            d['ll_classifier_diag_test_X']   = ll_classifier_diag_test_X
            d['ll_classifier_diag_test_Y']   = ll_classifier_diag_test_Y            
            d['ll_classifier_diag_test_idx'] = ll_classifier_diag_test_idx
        
        ut.save_pickle(d, modeling_pkl)
