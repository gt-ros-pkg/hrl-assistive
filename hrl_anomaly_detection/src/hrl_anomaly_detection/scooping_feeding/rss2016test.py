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
import rospy
import roslib
roslib.load_manifest('hrl_anomaly_detection')
import os, sys, copy
import random

# util
import numpy as np
import hrl_lib.util as ut
from hrl_anomaly_detection.util import *
import PyKDL
import hrl_lib.quaternion as qt

# learning
from sklearn.decomposition import PCA
from hrl_multimodal_anomaly_detection.hmm import learning_hmm_multi_n as hmm

# visualization
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec



def preprocessData(subject_names, task_name, raw_data_path, processed_data_path, nSet=1, \
                   folding_ratio=0.8, downSampleSize=200,\
                   renew=False, verbose=False):

    # Check if there is already scaled data
    for i in xrange(nSet):        
        target_file = os.path.join(processed_data_path, task_name+'_dataSet_'+str(i) )                    
        if os.path.isfile(target_file) is not True: renew=True
            
    if renew == False: return        

    success_list, failure_list = getSubjectFileList(raw_data_path, subject_names, task_name)

    nTrain = int(len(success_list) * folding_ratio)
    nTest  = len(success_list) - nTrain    

    if len(failure_list) < nTest: 
        print "Not enough failure data"
        sys.exit()

    # loading and time-sync
    data_dict = loadData(success_list, isTrainingData=False, downSampleSize=downSampleSize)
    
    data_min = {}
    data_max = {}
    for key in data_dict.keys():
        if 'time' in key: continue
        if data_dict[key] == []: continue
        data_min[key] = np.min(data_dict[key])        
        data_max[key] = np.max(data_dict[key])
        
    for i in xrange(nSet):

        # index selection
        success_idx  = range(len(success_list))
        failure_idx  = range(len(failure_list))
        train_idx    = random.sample(success_idx, nTrain)

        if nTest == 0: 
            success_test_idx = []
            failure_test_idx = []
        else: 
            success_test_idx = [x for x in success_idx if not x in train_idx]
            failure_test_idx = random.sample(failure_idx, nTest)

        # get training data
        trainFileList = [success_list[x] for x in train_idx]
        trainData = loadData(trainFileList, isTrainingData=True, \
                             downSampleSize=downSampleSize)

        # get test data
        if nTest != 0:        
            normalTestFileList = [success_list[x] for x in success_test_idx]
            normalTestData = loadData([success_list[x] for x in success_test_idx], 
                                                          isTrainingData=False, downSampleSize=downSampleSize)
            abnormalTestFileList = [failure_list[x] for x in failure_test_idx]
            abnormalTestData = loadData([failure_list[x] for x in failure_test_idx], \
                                        isTrainingData=False, downSampleSize=downSampleSize)

        # scaling data
        ## trainData_scaled = scaleData(trainData, scale=scale, data_min=data_min, 
        ##                              data_max=data_max, verbose=verbose)
        ## normalTestData_scaled = scaleData(normalTestData, scale=scale, data_min=data_min, 
        ##                                   data_max=data_max, verbose=verbose)
        ## abnormalTestData_scaled = scaleData(abnormalTestData, scale=scale, data_min=data_min, 
        ##                                     data_max=data_max, verbose=verbose)

        # cutting data (only traing and thresTest data)
        ## start_idx = int(float(len(trainData_scaled[0][0]))*train_cutting_ratio[0])
        ## end_idx   = int(float(len(trainData_scaled[0][0]))*train_cutting_ratio[1])

        ## for j in xrange(len(trainData_scaled)):
        ##     for k in xrange(len(trainData_scaled[j])):
        ##         trainData_scaled[j][k] = trainData_scaled[j][k][start_idx:end_idx]
                
        ## for j in xrange(len(normalTestData_scaled)):
        ##     for k in xrange(len(normalTestData_scaled[j])):                
        ##         normalTestData_scaled[j][k] = normalTestData_scaled[j][k][start_idx:end_idx]
                
        ## for j in xrange(len(abnormalTestData_scaled)):
        ##     for k in xrange(len(abnormalTestData_scaled[j])):                
        ##         abnormalTestData_scaled[j][k] = abnormalTestData_scaled[j][k][start_idx:end_idx]

        # Save data using dictionary
        d = {}
        d['trainData']        = trainData
        d['normalTestData']   = normalTestData
        d['abnormalTestData'] = abnormalTestData

        d['trainFileList']        = trainFileList
        d['normalTestFileList']   = normalTestFileList
        d['abnormalTestFileList'] = abnormalTestFileList        
        
        # Save data using dictionary
        target_file = os.path.join(processed_data_path, task_name+'_dataSet_'+str(i) )

        try:
            ut.save_pickle(d, target_file)        
        except:
            print "There is already target file: "
        
        

        
def extractLocalFeature(d, feature_list, local_range, param_dict=None, verbose=False):

    if param_dict is None:
        isTrainingData=True
        param_dict = {}

        if 'unimodal_audioPower' in feature_list:
            power_max   = np.amax(d['audioPowerList'])
            power_min   = np.amin(d['audioPowerList'])
            param_dict['unimodal_audioPower_power_min'] = power_min
            
        if 'unimodal_ftForce' in feature_list:
            force_array = None
            for idx in xrange(len(d['ftForceList'])):
                if force_array is None:
                    force_array = d['ftForceList'][idx]
                else:
                    force_array = np.hstack([force_array, d['ftForceList'][idx] ])

            ftForce_pca = PCA(n_components=1)
            res = ftForce_pca.fit_transform( force_array.T )
            param_dict['unimodal_ftForce_pca'] = ftForce_pca
    else:
        isTrainingData=False
        if 'unimodal_audioPower' in feature_list:
            power_min   = param_dict['unimodal_audioPower_power_min']
        
        if 'unimodal_ftForce' in feature_list:
            ftForce_pca = param_dict['unimodal_ftForce_pca']

    # -------------------------------------------------------------        

    # extract local features
    dataList   = []
    for idx in xrange(len(d['timesList'])):

        timeList     = d['timesList'][idx]
        dataSample = None

        # Unimoda feature - Audio --------------------------------------------
        if 'unimodal_audioPower' in feature_list:
            audioAzimuth = d['audioAzimuthList'][idx]
            audioPower   = d['audioPowerList'][idx]
            kinEEPos     = d['kinEEPosList'][idx]
            
            unimodal_audioPower = []
            for time_idx in xrange(len(timeList)):
                ang_max, ang_min = getAngularSpatialRF(kinEEPos[:,time_idx], local_range)

                if audioAzimuth[time_idx] > ang_min and audioAzimuth[time_idx] < ang_max:
                    unimodal_audioPower.append(audioPower[time_idx])
                else:
                    unimodal_audioPower.append(power_min) # or append white noise?

            if dataSample is None: dataSample = np.array(unimodal_audioPower)
            else: dataSample = np.vstack([dataSample, unimodal_audioPower])

            ## updateMinMax(param_dict, 'unimodal_audioPower', unimodal_audioPower)                
            ## self.audio_disp(timeList, audioAzimuth, audioPower, audioPowerLocal, \
            ##                 power_min=power_min, power_max=power_max)

        # Unimodal feature - Kinematics --------------------------------------
        if 'unimodal_kinVel' in feature_list:
            unimodal_kinVel = []
            if dataSample is None: dataSample = np.array(unimodal_kinVel)
            else: dataSample = np.vstack([dataSample, unimodal_kinVel])

        # Unimodal feature - Force -------------------------------------------
        if 'unimodal_ftForce' in feature_list:
            ftForce      = d['ftForceList'][idx]
            
            # ftForceLocal = np.linalg.norm(ftForce, axis=0) #* np.sign(ftForce[2])
            unimodal_ftForce = ftForce_pca.transform(ftForce.T).T
            ## self.ft_disp(timeList, ftForce, ftForceLocal)

            if dataSample is None: dataSample = np.array(unimodal_ftForce)
            else: dataSample = np.vstack([dataSample, unimodal_ftForce])
                        
        # Crossmodal feature - relative dist --------------------------
        if 'crossmodal_targetRelativeDist' in feature_list:
            kinEEPos     = d['kinEEPosList'][idx]
            kinTargetPos  = d['kinTargetPosList'][idx]
            
            crossmodal_targetRelativeDist = np.linalg.norm(kinTargetPos - kinEEPos, axis=0)

            if dataSample is None: dataSample = np.array(crossmodal_targetRelativeDist)
            else: dataSample = np.vstack([dataSample, crossmodal_targetRelativeDist])

        # Crossmodal feature - relative angle --------------------------
        if 'crossmodal_targetRelativeAng' in feature_list:                
            kinEEQuat    = d['kinEEQuatList'][idx]
            kinTargetQuat = d['kinTargetQuatList'][idx]
            
            crossmodal_targetRelativeAng = []
            for time_idx in xrange(len(timeList)):

                startQuat = kinEEQuat[:,time_idx]
                endQuat   = kinTargetQuat[:,time_idx]

                diff_ang = qt.quat_angle(startQuat, endQuat)
                crossmodal_targetRelativeAng.append( abs(diff_ang) )

            if dataSample is None: dataSample = np.array(crossmodal_targetRelativeAng)
            else: dataSample = np.vstack([dataSample, crossmodal_targetRelativeAng])

        # Crossmodal feature - vision relative dist --------------------------
        if 'crossmodal_artagRelativeDist' in feature_list:
            kinEEPos  = d['kinEEPosList'][idx]
            visionPos = d['visionPosList'][idx]
            
            crossmodal_artagRelativeDist = np.linalg.norm(visionPos - kinEEPos, axis=0)

            if dataSample is None: dataSample = np.array(crossmodal_artagRelativeDist)
            else: dataSample = np.vstack([dataSample, crossmodal_artagRelativeDist])

        # Crossmodal feature - vision relative angle --------------------------
        if 'crossmodal_artagRelativeAng' in feature_list:                
            kinEEQuat    = d['kinEEQuatList'][idx]
            visionQuat = d['visionQuatList'][idx]
            
            crossmodal_artagRelativeAng = []
            for time_idx in xrange(len(timeList)):

                startQuat = kinEEQuat[:,time_idx]
                endQuat   = visionQuat[:,time_idx]

                diff_ang = qt.quat_angle(startQuat, endQuat)
                crossmodal_artagRelativeAng.append( abs(diff_ang) )

            if dataSample is None: dataSample = np.array(crossmodal_artagRelativeAng)
            else: dataSample = np.vstack([dataSample, crossmodal_artagRelativeAng])

        # ----------------------------------------------------------------
        dataList.append(dataSample)

        
    # Converting data structure
    nSample      = len(dataList)
    nEmissionDim = len(dataList[0])
    features     = []
    for i in xrange(nEmissionDim):
        feature  = []

        for j in xrange(nSample):
            feature.append(dataList[j][i,:])

        features.append( feature )


    # Scaling ------------------------------------------------------------
    if isTrainingData:
        param_dict['feature_max'] = [ np.max(x) for x in features ]
        param_dict['feature_min'] = [ np.min(x) for x in features ]
        
    scaled_features = []
    for i, feature in enumerate(features):
        scaled_features.append( ( np.array(feature) - param_dict['feature_min'][i] )\
                                /( param_dict['feature_max'][i] - param_dict['feature_min'][i]) )

    return scaled_features, param_dict


def updateMinMax(param_dict, feature_name, feature_array):

    if feature_name in param_dict.keys():
        maxVal = np.max(feature_array)
        minVal = np.min(feature_array)
        if param_dict[feature_name+'_max'] < maxVal:
            param_dict[feature_name+'_max'] = maxVal
        if param_dict[feature_name+'_min'] > minVal:
            param_dict[feature_name+'_min'] = minVal
    else:
        param_dict[feature_name+'_max'] = -100000000000
        param_dict[feature_name+'_min'] =  100000000000
        
    

def test(processed_data_path, task_name, nSet, feature_list, local_range, viz=False):

    for i in xrange(nSet):        
        target_file = os.path.join(processed_data_path, task_name+'_dataSet_'+str(i) )                    
        if os.path.isfile(target_file) is not True: 
            print "There is no saved data"
            sys.exit()

        data_dict = ut.load_pickle(target_file)
        ## if viz: visualization_raw_data(data_dict)

        # training set
        trainingData, param_dict = extractLocalFeature(data_dict['trainData'], feature_list, local_range)
        
        # test set
        normalTestData, _ = extractLocalFeature(data_dict['normalTestData'], feature_list, local_range, \
                                                param_dict=param_dict)        
        abnormalTestData, _ = extractLocalFeature(data_dict['abnormalTestData'], feature_list, local_range, \
                                                param_dict=param_dict)

        print "======================================"
        print "Training data: ", np.shape(trainingData)
        print "Normal test data: ", np.shape(normalTestData)
        print "Abnormal test data: ", np.shape(abnormalTestData)
        print "======================================"


        if viz: visualization_hmm_data(feature_list, trainingData=trainingData, \
                                       normalTestData=normalTestData,\
                                       abnormalTestData=abnormalTestData)        
        
        # training hmm
        nState = 5
        nEmissionDim = len(trainingData)
                
        ml = hmm.learning_hmm_multi_n(nState, nEmissionDim)
        ml.fit(trainingData)

        # evaluation


def visualization_hmm_data(feature_list, trainingData=None, normalTestData=None, abnormalTestData=None):

    if trainingData is not None:
        nDimension = len(trainingData)
    elif normalTestData is not None:
        nDimension = len(normalTestData)
    elif abnormalTestData is not None:
        nDimension = len(abnormalTestData)
    else:
        print "no data"
        sys.exit()
        
    fig = plt.figure()            
    # --------------------------------------------------
    for i in xrange(nDimension):
        ax = fig.add_subplot(100*nDimension+10+(i+1))
        if trainingData is not None:
            ax.plot(np.array(trainingData[i]).T, 'b')
        elif normalTestData is not None:
            ax.plot(np.array(normalTestData[i]).T, 'k')
        ## elif abnormalTestData is not None:
        ##     ax.plot(abnormalTestData[i], 'r')

        ax.set_title(feature_list[i])

    fig.savefig('test.pdf')
    fig.savefig('test.png')
    os.system('cp test.p* ~/Dropbox/HRL/')        
    ## plt.show()
    sys.exit()


def visualization_raw_data(data_dict, modality='ft'):

    ## dataList = data_dict['trainData']['ftForceList']
    ## dataList = data_dict['trainData']['kinTargetPosList']
    dataList = data_dict['trainData']['kinEEPosList']
    dataList = data_dict['trainData']['kinEEQuatList']

    fileList = data_dict['trainFileList']
    
    
    ## # Converting data structure
    ## nSample      = len(dataList)
    ## nEmissionDim = len(dataList[0])
    ## features     = []
    ## for i in xrange(nEmissionDim):
    ##     feature  = []

    ##     for j in xrange(nSample):
    ##         feature.append(dataList[j][i,:])

    ##     features.append( feature )

    count = 0
    d_list = []
    f_list = []
    
    for idx, data in enumerate(dataList):

        d_list.append( np.mean(data, axis=0) )
        f_list.append( fileList[idx].split('/')[-1] )

        if idx%10 == 9:
                
            fig = plt.figure()            
            ax1 = fig.add_subplot(111)

            for j, d in enumerate(d_list):
                ax1.plot(d_list[j], label=f_list[j])
                
            plt.legend(loc=3,prop={'size':8})

            fig.savefig('test'+str(count)+'.pdf')
            fig.savefig('test'+str(count)+'.png')
            os.system('cp test'+str(count)+'.p* ~/Dropbox/HRL/')        
            d_list = []
            f_list = []
            count += 1
            

    fig = plt.figure()            
    ax1 = fig.add_subplot(111)

    for j, d in enumerate(d_list):
        ax1.plot(d_list[j], label=f_list[j])

    plt.legend(loc=3,prop={'size':8})

    fig.savefig('test'+str(count)+'.pdf')
    fig.savefig('test'+str(count)+'.png')
    os.system('cp test'+str(count)+'.p* ~/Dropbox/HRL/')        
            
    ## plt.show()
    sys.exit()

    

if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
        
    p.add_option('--renew', action='store_true', dest='bRenew',
                 default=False, help='Renew pickle files.')

    opt, args = p.parse_args()

    save_data_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/RSS2016'
    raw_data_path  = '/home/dpark/hrl_file_server/dpark_data/anomaly/RSS2016/'

    #---------------------------------------------------------------------------           
    # Run evaluation
    #---------------------------------------------------------------------------           
    subject = 'gatsbii'
    task    = 'scooping'    
    feature_list = ['unimodal_ftForce', 'crossmodal_targetRelativeDist', \
                    'crossmodal_targetRelativeAng']

    ## subject = 'gatsbii'
    ## task    = 'feeding' 
    ## feature_list = ['unimodal_audioPower', 'unimodal_ftForce', 'crossmodal_artagRelativeDist', \
    ##                 'crossmodal_artagRelativeAng']
    
    preprocessData([subject], task, raw_data_path, save_data_path, renew=opt.bRenew)

    # Dectection TEST 
    nSet         = 1
    local_range  = 0.25    
    viz          = False
        
    test(save_data_path, task, nSet, feature_list, local_range, viz=viz)
