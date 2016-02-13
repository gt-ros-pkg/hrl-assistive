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
from hrl_anomaly_detection import util

from mvpa2.datasets.base import Dataset
from mvpa2.generators.partition import NFoldPartitioner
from mvpa2.generators import splitters

from sklearn import cross_validation

import matplotlib.pyplot as plt

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
             nSet=1, downSampleSize=200, scale=10.0, cutting=False, success_viz=False, failure_viz=False, \
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
        success_list, failure_list = util.getSubjectFileList(raw_data_path, subject_names, task_name)

        # loading and time-sync    
        all_data_pkl     = os.path.join(processed_data_path, task_name+'_all_'+rf_center+\
                                        '_'+str(local_range))
        _, all_data_dict = util.loadData(success_list+failure_list, isTrainingData=False,
                                         downSampleSize=downSampleSize,\
                                         local_range=local_range, rf_center=rf_center,\
                                         ##global_data=True,\
                                         renew=data_renew, save_pkl=all_data_pkl)

        success_data_pkl     = os.path.join(processed_data_path, task_name+'_success_'+rf_center+\
                                            '_'+str(local_range))
        _, success_data_dict = util.loadData(success_list, isTrainingData=True,
                                             downSampleSize=downSampleSize,\
                                             local_range=local_range, rf_center=rf_center,\
                                             renew=data_renew, save_pkl=success_data_pkl)

        failure_data_pkl     = os.path.join(processed_data_path, task_name+'_failure_'+rf_center+\
                                            '_'+str(local_range))
        _, failure_data_dict = util.loadData(failure_list, isTrainingData=False,
                                             downSampleSize=downSampleSize,\
                                             local_range=local_range, rf_center=rf_center,\
                                             renew=data_renew, save_pkl=failure_data_pkl)

        # data set        
        allData, param_dict = extractLocalFeature(all_data_dict, feature_list, scale=scale)
        trainingData, _     = extractLocalFeature(success_data_dict, feature_list, scale=scale, \
                                                  param_dict=param_dict)
        abnormalTestData, _ = extractLocalFeature(failure_data_dict, feature_list, scale=scale, \
                                                       param_dict=param_dict)

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
    ## _, success_data_dict = util.loadData(success_list, isTrainingData=True,
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


def raw_data_extraction(subject_names, task_name, raw_data_path, processed_data_path, rf_center, local_range, \
                        nSet=1, downSampleSize=200, scale=10.0, cutting=False, \
                        success_viz=False, failure_viz=False, \
                        save_pdf=False, solid_color=True, data_renew=False):

    if os.path.isdir(processed_data_path) is False:
        os.system('mkdir -p '+processed_data_path)

    save_pkl = os.path.join(processed_data_path, 'raw_extraction_'+rf_center+'_'+str(local_range) )
    if os.path.isfile(save_pkl) and data_renew is not True :
        data_dict = ut.load_pickle(save_pkl)
        allData          = data_dict['allData']
        trainingData     = data_dict['trainingData'] 
        abnormalTestData = data_dict['abnormalTestData']
        abnormalTestNameList = data_dict['abnormalTestNameList']
        param_dict       = data_dict['param_dict']
    else:
        ## data_renew = False #temp        
        success_list, failure_list = util.getSubjectFileList(raw_data_path, subject_names, task_name)

        # loading and time-sync    
        all_data_pkl     = os.path.join(processed_data_path, task_name+'_all_'+rf_center+\
                                        '_'+str(local_range))
        _, all_data_dict = util.loadData(success_list+failure_list, isTrainingData=False,
                                         downSampleSize=downSampleSize,\
                                         local_range=local_range, rf_center=rf_center,\
                                         ##global_data=True,\
                                         renew=data_renew, save_pkl=all_data_pkl)

        success_data_pkl     = os.path.join(processed_data_path, task_name+'_success_'+rf_center+\
                                            '_'+str(local_range))
        _, success_data_dict = util.loadData(success_list, isTrainingData=True,
                                             downSampleSize=downSampleSize,\
                                             local_range=local_range, rf_center=rf_center,\
                                             renew=data_renew, save_pkl=success_data_pkl)

        failure_data_pkl     = os.path.join(processed_data_path, task_name+'_failure_'+rf_center+\
                                            '_'+str(local_range))
        _, failure_data_dict = util.loadData(failure_list, isTrainingData=False,
                                             downSampleSize=downSampleSize,\
                                             local_range=local_range, rf_center=rf_center,\
                                             renew=data_renew, save_pkl=failure_data_pkl)

        # data set        
        allData, param_dict = extractLocalFeature(all_data_dict, feature_list, scale=scale)
        trainingData, _     = extractLocalFeature(success_data_dict, feature_list, scale=scale, \
                                                       param_dict=param_dict)
        abnormalTestData, _ = extractLocalFeature(failure_data_dict, feature_list, scale=scale, \
                                                       param_dict=param_dict)

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


    ## All data
    nPlot = None
    feature_names = np.array(param_dict['feature_names'])

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


def extractLocalFeature(d, feature_list, scale=10.0, param_dict=None, verbose=False):

    if param_dict is None:
        isTrainingData=True
        param_dict = {}

        if 'unimodal_audioPower' in feature_list:
            ## power_max = np.amax(d['audioPowerList'])
            ## power_min = np.amin(d['audioPowerList'])
            ## power_min = np.mean(np.array(d['audioPowerList'])[:,:10])
            power_min = 10000
            power_max = 0
            for pwr in d['audioPowerList']:
                p_min = np.amin(pwr)
                p_max = np.amax(pwr)
                if power_min > p_min:
                    power_min = p_min
                ## if p_max < 50 and power_max < p_max:
                if power_max < p_max:
                    power_max = p_max

            param_dict['unimodal_audioPower_power_max'] = power_max
            param_dict['unimodal_audioPower_power_min'] = power_min
                                
        ## if 'unimodal_ftForce' in feature_list:
        ##     force_array = None
        ##     start_force_array = None
        ##     for idx in xrange(len(d['ftForceList'])):
        ##         if force_array is None:
        ##             force_array = d['ftForceList'][idx]
        ##             ## start_force_array = d['ftForceList'][idx][:,:5]
        ##         else:
        ##             force_array = np.hstack([force_array, d['ftForceList'][idx] ])
        ##             ## start_force_array = np.hstack([start_force_array, d['ftForceList'][idx][:,:5]])

        ##     ftPCADim    = 2
        ##     ftForce_pca = PCA(n_components=ftPCADim)
        ##     res = ftForce_pca.fit_transform( force_array.T )            
        ##     param_dict['unimodal_ftForce_pca'] = ftForce_pca
        ##     param_dict['unimodal_ftForce_pca_dim'] = ftPCADim

        ##     ## res = ftForce_pca.transform(start_force_array.T)
        ##     ## param_dict['unimodal_ftForce_pca_init_avg'] = np.array([np.mean(res, axis=0)]).T
        ##     ## param_dict['unimodal_ftForce_init_avg'] = np.mean(start_force_array, axis=1)

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
        isTrainingData=False
            

    # -------------------------------------------------------------        

    # extract local features
    dataList   = []
    for idx in xrange(len(d['timesList'])): # each sample

        timeList     = d['timesList'][idx]
        dataSample = None

        ## # Define receptive field center trajectory ---------------------------
        ## if rf_center == 'kinEEPos':
        ##     rf_traj = d['kinEEPosList'][idx]
        ## elif rf_center == 'kinForearmPos':
        ##     rf_traj = d['kinForearmPosList'][idx]
        ## ## elif rf_center == 'l_upper_arm_link':            
        ## else:
        ##     sys.exit()
        

        # Unimoda feature - Audio --------------------------------------------
        if 'unimodal_audioPower' in feature_list:
            ## audioAzimuth = d['audioAzimuthList'][idx]
            audioPower   = d['audioPowerList'][idx]            
            unimodal_audioPower = audioPower
            
            if dataSample is None: dataSample = copy.copy(np.array(unimodal_audioPower))
            else: dataSample = np.vstack([dataSample, copy.copy(unimodal_audioPower)])
            if 'audioPower' not in param_dict['feature_names']:
                param_dict['feature_names'].append('audioPower')

        # Unimoda feature - AudioWrist ---------------------------------------
        if 'unimodal_audioWristRMS' in feature_list:
            audioWristRMS = d['audioWristRMSList'][idx]            
            unimodal_audioWristRMS = audioWristRMS
            
            if dataSample is None: dataSample = copy.copy(np.array(unimodal_audioWristRMS))
            else: dataSample = np.vstack([dataSample, copy.copy(unimodal_audioWristRMS)])
            if 'audioWristRMS' not in param_dict['feature_names']:
                param_dict['feature_names'].append('audioWristRMS')

        # Unimodal feature - Kinematics --------------------------------------
        if 'unimodal_kinVel' in feature_list:
            kinVel  = d['kinVelList'][idx]
            unimodal_kinVel = kinVel

            if dataSample is None: dataSample = np.array(unimodal_kinVel)
            else: dataSample = np.vstack([dataSample, unimodal_kinVel])
            if 'kinVel_x' not in param_dict['feature_names']:
                param_dict['feature_names'].append('kinVel_x')
                param_dict['feature_names'].append('kinVel_y')
                param_dict['feature_names'].append('kinVel_z')

        # Unimodal feature - Force -------------------------------------------
        if 'unimodal_ftForce' in feature_list:
            ftForce = d['ftForceList'][idx]

            # magnitude
            if len(np.shape(ftForce)) > 1:
                unimodal_ftForce_mag = np.linalg.norm(ftForce, axis=0)
                # individual force
                unimodal_ftForce_ind = ftForce[2:3,:]                
                
                if dataSample is None: dataSample = np.array(unimodal_ftForce_mag)
                else: dataSample = np.vstack([dataSample, unimodal_ftForce_mag])

                if dataSample is None: dataSample = np.array(unimodal_ftForce_ind)
                else: dataSample = np.vstack([dataSample, unimodal_ftForce_ind])

                if 'ftForce_z' not in param_dict['feature_names']:
                    param_dict['feature_names'].append('ftForce_mag')
                    ## param_dict['feature_names'].append('ftForce_x')
                    ## param_dict['feature_names'].append('ftForce_y')
                    param_dict['feature_names'].append('ftForce_z')
            else:                
                unimodal_ftForce_mag = ftForce
            
                if dataSample is None: dataSample = np.array(unimodal_ftForce_mag)
                else: dataSample = np.vstack([dataSample, unimodal_ftForce_mag])

                if 'ftForce_mag' not in param_dict['feature_names']:
                    param_dict['feature_names'].append('ftForce_mag')

            ## ftPos   = d['kinEEPosList'][idx]
            ## ftForce_pca = param_dict['unimodal_ftForce_pca']

            ## unimodal_ftForce = None
            ## for time_idx in xrange(len(timeList)):
            ##     if unimodal_ftForce is None:
            ##         unimodal_ftForce = ftForce_pca.transform(ftForce[:,time_idx:time_idx+1].T).T
            ##     else:
            ##         unimodal_ftForce = np.hstack([ unimodal_ftForce, \
            ##                                        ftForce_pca.transform(ftForce[:,time_idx:time_idx+1].T).T ])

            ## unimodal_ftForce -= np.array([np.mean(unimodal_ftForce[:,:5], axis=1)]).T
            
            ## if 'ftForce_1' not in param_dict['feature_names']:
            ##     param_dict['feature_names'].append('ftForce_1')
            ##     param_dict['feature_names'].append('ftForce_2')
            ## if 'ftForce_x' not in param_dict['feature_names']:
            ##     param_dict['feature_names'].append('ftForce_x')
            ##     param_dict['feature_names'].append('ftForce_y')
            ##     param_dict['feature_names'].append('ftForce_z')

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

            unimodal_ppsForce -= np.array([np.mean(unimodal_ppsForce[:,:5], axis=1)]).T

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
            visionChangeMag = d['visionChangeMagList'][idx]

            unimodal_visionChange = visionChangeMag

            if dataSample is None: dataSample = unimodal_visionChange
            else: dataSample = np.vstack([dataSample, unimodal_visionChange])
            if 'visionChange' not in param_dict['feature_names']:
                param_dict['feature_names'].append('visionChange')

                
        # Unimodal feature - fabric skin ------------------------------------
        if 'unimodal_fabricForce' in feature_list:
            fabricMag = d['fabricMagList'][idx]

            unimodal_fabricForce = fabricMag

            if dataSample is None: dataSample = unimodal_fabricForce
            else: dataSample = np.vstack([dataSample, unimodal_fabricForce])
            if 'fabricForce' not in param_dict['feature_names']:
                param_dict['feature_names'].append('fabricForce')

            
        # Crossmodal feature - relative dist --------------------------
        if 'crossmodal_targetEEDist' in feature_list:
            kinEEPos     = d['kinEEPosList'][idx]
            kinTargetPos  = d['kinTargetPosList'][idx]

            dist = np.linalg.norm(kinTargetPos - kinEEPos, axis=0)
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

            kinEEPos  = d['kinEEPosList'][idx]
            visionArtagPos = d['visionArtagPosList'][idx][:3]
            dist = np.linalg.norm(visionArtagPos - kinEEPos, axis=0)
            
            crossmodal_artagEEAng = []
            for time_idx in xrange(len(timeList)):

                startQuat = kinEEQuat[:,time_idx]
                endQuat   = visionArtagQuat[:,time_idx]

                diff_ang = qt.quat_angle(startQuat, endQuat)
                crossmodal_artagEEAng.append( abs(diff_ang) )

            if dataSample is None: dataSample = np.array(crossmodal_artagEEAng)
            else: dataSample = np.vstack([dataSample, crossmodal_artagEEAng])
            if 'artagEEAng' not in param_dict['feature_names']:
                param_dict['feature_names'].append('artagEEAng')

        # ----------------------------------------------------------------
        dataList.append(dataSample)


    # Converting data structure & cutting unnecessary part
    nSample      = len(dataList)
    nEmissionDim = len(dataList[0])
    features     = []
    startIdx     = 50
    endIdx       = 150
    for i in xrange(nEmissionDim):
        feature  = []

        for j in xrange(nSample):
            try:
                ## feature.append(dataList[j][i])
                feature.append(dataList[j][i,:])
            except:
                ## print "Failed to cut data", j,i, np.shape(dataList[j]), dataList[j][i]
                print np.shape(dataList), np.shape(dataList[j]), j, i
                sys.exit()

        features.append( feature )


    # Scaling ------------------------------------------------------------
    if isTrainingData:
        param_dict['feature_max'] = [ np.max(np.array(feature).flatten()) for feature in features ]
        param_dict['feature_min'] = [ np.min(np.array(feature).flatten()) for feature in features ]
        print "max: ", param_dict['feature_max']
        print "min: ", param_dict['feature_min']
        
        
    scaled_features = []
    for i, feature in enumerate(features):

        if abs( param_dict['feature_max'][i] - param_dict['feature_min'][i]) < 1e-3:
            scaled_features.append( np.array(feature) )
        else:
            scaled_features.append( scale* ( np.array(feature) - param_dict['feature_min'][i] )\
                                    /( param_dict['feature_max'][i] - param_dict['feature_min'][i]) )

    ## import matplotlib.pyplot as plt
    ## plt.figure()
    ## plt.plot(np.array(scaled_features[0]).T)
    ## plt.show()
    ## sys.exit()
                                
    return scaled_features, param_dict

