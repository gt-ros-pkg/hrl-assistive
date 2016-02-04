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

# util
import numpy as np
import scipy
import hrl_lib.util as ut

from mvpa2.datasets.base import Dataset
from mvpa2.generators.partition import NFoldPartitioner
from mvpa2.generators import splitters

from sklearn import cross_validation

def create_mvpa_dataset(aXData, chunks, labels):
    data = Dataset(samples=aXData)
    data.sa['id']      = range(0,len(labels))
    data.sa['chunks']  = chunks
    data.sa['targets'] = labels

    return data

def kFold_data_index(nAbnormal, nNormal, nAbnormalFold, nNormalFold):

    normal_folds   = cross_validation.KFold(nNormal, n_folds=nNormalFold, shuffle=True)
    abnormal_folds = cross_validation.KFold(nAbnormal, n_folds=nAbnormalFold, shuffle=True)

    kFold_list = []

    for normal_temp_fold, normal_test_fold in normal_folds:

        normal_dc_fold = cross_validation.KFold(len(normal_temp_fold), \
                                                n_folds=nNormalFold-1, shuffle=True)
        for normal_train_fold, normal_classifier_fold in normal_dc_fold:

            normal_d_fold = normal_temp_fold[normal_train_fold]
            normal_c_fold = normal_temp_fold[normal_classifier_fold]

            for abnormal_c_fold, abnormal_test_fold in abnormal_folds:
                '''
                Normal training data for model
                Normal training data for classifier
                Abnormal training data for classifier
                Normal test data 
                Abnormal test data 
                '''
                index_list = [normal_d_fold, normal_c_fold, abnormal_c_fold, \
                              normal_test_fold, abnormal_test_fold]
                kFold_list.append(index_list)

    return kFold_list
    
def feature_extraction(subject_names, task_name, raw_data_path, processed_data_path, rf_center, local_range, \
             nSet=1, downSampleSize=200, success_viz=False, failure_viz=False, \
             save_pdf=False, solid_color=True, \
             feature_list=['crossmodal_targetEEDist'], data_renew=False):

    if os.path.isdir(processed_data_path) is False:
        os.system('mkdir -p '+processed_data_path)

    save_pkl = os.path.join(processed_data_path, 'feature_extraction_'+rf_center+'_'+str(local_range) )
    if os.path.isfile(save_pkl) and data_renew is not True :
        data_dict = ut.load_pickle(save_pkl)
        allData          = data_dict['allData']
        trainingData     = data_dict['trainingData'] 
        abnormalTestData = data_dict['abnormalTestData']
        abnormalTestNameList = data_dict['abnormalTestNameList']
        param_dict       = data_dict['param_dict']
    else:
        ## data_renew = False #temp        
        success_list, failure_list = getSubjectFileList(raw_data_path, subject_names, task_name)

        # loading and time-sync    
        all_data_pkl     = os.path.join(processed_data_path, task_name+'_all_'+rf_center+\
                                        '_'+str(local_range))
        _, all_data_dict = loadData(success_list+failure_list, isTrainingData=False,
                                    downSampleSize=downSampleSize,\
                                    local_range=local_range, rf_center=rf_center,\
                                    ##global_data=True,\
                                    renew=data_renew, save_pkl=all_data_pkl)

        success_data_pkl     = os.path.join(processed_data_path, task_name+'_success_'+rf_center+\
                                            '_'+str(local_range))
        _, success_data_dict = loadData(success_list, isTrainingData=True,
                                        downSampleSize=downSampleSize,\
                                        local_range=local_range, rf_center=rf_center,\
                                        renew=data_renew, save_pkl=success_data_pkl)

        failure_data_pkl     = os.path.join(processed_data_path, task_name+'_failure_'+rf_center+\
                                            '_'+str(local_range))
        _, failure_data_dict = loadData(failure_list, isTrainingData=False,
                                        downSampleSize=downSampleSize,\
                                        local_range=local_range, rf_center=rf_center,\
                                        renew=data_renew, save_pkl=failure_data_pkl)

        # data set        
        allData, param_dict = extractLocalFeature(all_data_dict, feature_list)
        trainingData, _     = extractLocalFeature(success_data_dict, feature_list, param_dict=param_dict)
        abnormalTestData, _ = extractLocalFeature(failure_data_dict, feature_list, param_dict=param_dict)

        allData          = np.array(allData)
        trainingData     = np.array(trainingData)
        abnormalTestData = np.array(abnormalTestData)

        data_dict = {}
        data_dict['allData'] = allData
        data_dict['trainingData'] = trainingData
        data_dict['abnormalTestData'] = abnormalTestData
        data_dict['abnormalTestNameList'] = abnormalTestNameList = failure_data_dict['fileNameList']
        data_dict['param_dict'] = param_dict
        ut.save_pickle(data_dict, save_pkl)


    ## # test
    ## success_list, failure_list = getSubjectFileList(raw_data_path, subject_names, task_name)
    ## _, success_data_dict = loadData(success_list, isTrainingData=True,
    ##                                 downSampleSize=downSampleSize,\
    ##                                 local_range=local_range, rf_center=rf_center)
    ## trainingData, _      = extractLocalFeature(success_data_dict, feature_list, \
    ##                                            param_dict=data_dict['param_dict'])
    ## sys.exit()
    
    ## All data
    nPlot = None
    feature_names = np.array(param_dict['feature_names'])

    if True:

        # 1) exclude stationary data
        thres = 0.025
        n,m,k = np.shape(trainingData)
        diff_all_data = trainingData[:,:,1:] - trainingData[:,:,:-1]
        add_idx    = []
        remove_idx = []
        std_list = []
        for i in xrange(n):
            std = np.max(np.max(diff_all_data[i], axis=1))
            std_list.append(std)
            if  std < thres: remove_idx.append(i)
            else: add_idx.append(i)

        allData          = allData[add_idx]
        trainingData     = trainingData[add_idx]
        abnormalTestData = abnormalTestData[add_idx]
        AddFeature_names    = feature_names[add_idx]
        RemoveFeature_names = feature_names[remove_idx]

        print "--------------------------------"
        print "STD list: ", std_list
        print "Add features: ", AddFeature_names
        print "Remove features: ", RemoveFeature_names
        print "--------------------------------"
        ## sys.exit()


    # -------------------- Display ---------------------
    fig = None
    if success_viz:
        fig = plt.figure()
        n,m,k = np.shape(trainingData)
        if nPlot is None:
            if n%2==0: nPlot = n
            else: nPlot = n+1

        for i in xrange(n):
            ax = fig.add_subplot((nPlot/2)*100+20+i)
            if solid_color: ax.plot(trainingData[i].T, c='b')
            else: ax.plot(trainingData[i].T)
            ax.set_title( AddFeature_names[i] )

    if failure_viz:
        if fig is None: fig = plt.figure()
        n,m,k = np.shape(abnormalTestData)
        if nPlot is None:
            if n%2==0: nPlot = n
            else: nPlot = n+1

        for i in xrange(n):
            ax = fig.add_subplot((nPlot/2)*100+20+i)
            if solid_color: ax.plot(abnormalTestData[i].T, c='r')
            else: ax.plot(abnormalTestData[i].T)
            ax.set_title( AddFeature_names[i] )

    if success_viz or failure_viz:
        plt.tight_layout(pad=3.0, w_pad=0.5, h_pad=0.5)

        if save_pdf:
            fig.savefig('test.pdf')
            fig.savefig('test.png')
            os.system('cp test.p* ~/Dropbox/HRL/')        
        else:
            plt.show()

    print "---------------------------------------------------"
    print np.shape(trainingData), np.shape(abnormalTestData)
    print "---------------------------------------------------"

    return allData, trainingData, abnormalTestData, abnormalTestNameList

