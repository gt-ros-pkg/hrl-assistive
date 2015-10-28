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

# util
import numpy as np
import hrl_lib.util as ut
import hrl_lib.quaternion as qt

from scipy import interpolate
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import data_viz

def extrapolateData(data, maxsize):
    if len(np.shape(data[0])) > 1:     
        # need to implement incremental extrapolation
        return [x if len(x[0]) >= maxsize else x + [x[:,-1]]*(maxsize-len(x[0])) for x in data]
    else:
        # need to implement incremental extrapolation        
        return [x if len(x) >= maxsize else x + [x[-1]]*(maxsize-len(x)) for x in data]
        

def loadData(fileNames, isTrainingData=False, downSampleSize=100, raw_viz=False, interp_viz=False, \
             verbose=False):

    data_dict = {}
    data_dict['timesList']        = []
    data_dict['audioAzimuthList'] = []    
    data_dict['audioPowerList']   = []    
    data_dict['kinEEPosList']     = []
    data_dict['kinEEQuatList']    = []
    data_dict['kinJntPosList']    = []
    data_dict['ftForceList']      = []
    data_dict['kinTargetPosList']  = []
    data_dict['kinTargetQuatList'] = []
    data_dict['visionPosList']     = []
    data_dict['visionQuatList']    = []

    
    if raw_viz or interp_viz: fig = plt.figure()

    for idx, fileName in enumerate(fileNames):
        if os.path.isdir(fileName):
            continue

        print fileName
        d = ut.load_pickle(fileName)        

        max_time = 0
        for key in d.keys():
            if 'time' in key and 'init' not in key:
                feature_time = d[key]
                if max_time < feature_time[-1]: max_time = feature_time[-1]
        new_times = np.linspace(0.01, max_time, downSampleSize)
        data_dict['timesList'].append(new_times)

        # sound ----------------------------------------------------------------
        if 'audio_time' in d.keys():
            audio_time    = d['audio_time']
            audio_azimuth = d['audio_azimuth']
            audio_power   = d['audio_power']

            interp = interpolate.splrep(audio_time, audio_azimuth, s=0)
            data_dict['audioAzimuthList'].append(interpolate.splev(new_times, interp, der=0))

            interp = interpolate.splrep(audio_time, audio_power, s=0)
            data_dict['audioPowerList'].append(interpolate.splev(new_times, interp, der=0))
            
        # kinematics -----------------------------------------------------------
        if 'kinematics_time' in d.keys():
            kin_time = d['kinematics_time']
            kin_ee_pos  = d['kinematics_ee_pos'] # 3xN
            kin_ee_quat = d['kinematics_ee_quat'] # ?xN
            kin_target_pos  = d['kinematics_target_pos']
            kin_target_quat = d['kinematics_target_quat']
            kin_jnt_pos  = d['kinematics_jnt_pos'] # 7xN

            if raw_viz:
                ax = fig.add_subplot(312)
                if len(kin_time) > len(kin_ee_pos[2]):
                    ax.plot(kin_time[:len(kin_ee_pos[2])], kin_ee_pos[2])
                else:
                    ax.plot(kin_time, kin_ee_pos[2][:len(kin_time)])

                ax = fig.add_subplot(313)
                if len(kin_time) > len(kin_jnt_pos[2]):
                    ax.plot(kin_time[:len(kin_jnt_pos[2])], kin_jnt_pos[2])
                else:
                    ax.plot(kin_time, kin_jnt_pos[2][:len(kin_time)])
                    
            
            ee_pos_array = interpolationData(kin_time, kin_ee_pos, new_times)
            data_dict['kinEEPosList'].append(ee_pos_array)                                         

            ee_quat_array = interpolationQuatData(kin_time, kin_ee_quat, new_times)
            data_dict['kinEEQuatList'].append(ee_quat_array)                                         

            target_pos_array = interpolationData(kin_time, kin_target_pos, new_times)
            data_dict['kinTargetPosList'].append(target_pos_array)                                         

            target_quat_array = interpolationQuatData(kin_time, kin_target_quat, new_times)
            data_dict['kinTargetQuatList'].append(target_quat_array)                                         

            jnt_pos_array = interpolationData(kin_time, kin_jnt_pos, new_times)
            data_dict['kinJntPosList'].append(jnt_pos_array)                                         
            
        # ft -------------------------------------------------------------------
        if 'ft_time' in d.keys():
            ft_time        = d['ft_time']
            ft_force_array = d['ft_force']

            if raw_viz:
                ax = fig.add_subplot(311)
                if len(ft_time) > len(ft_force_array[2]):
                    ax.plot(ft_time[:len(ft_force_array[2])], ft_force_array[2])
                else:
                    ax.plot(ft_time, ft_force_array[2][:len(ft_time)])           

                print kin_time[0], kin_time[-1], ft_time[0], ft_time[-1]
                print np.shape(kin_time), np.shape(ft_time)
                plt.show()
                fig = plt.figure()
            ## if idx > 10: break
                
                    
            force_array = interpolationData(ft_time, ft_force_array, new_times)
            data_dict['ftForceList'].append(force_array)                                         
            
        # vision ---------------------------------------------------------------
        if 'vision_time' in d.keys():
            vision_time = d['vision_time']
            vision_pos  = d['vision_pos']
            vision_quat = d['vision_quat']

            vision_pos_array  = interpolationData(vision_time, vision_pos, new_times)
            data_dict['visionPosList'].append(vision_pos_array)                                         
            
            vision_quat_array = interpolationQuatData(vision_time, vision_quat, new_times)
            data_dict['visionQuatList'].append(vision_quat_array)                                         

        # pps ------------------------------------------------------------------
        if 'pps_skin_time' in d.keys():
            pps_skin_time  = d['pps_skin_time']
            pps_skin_left  = d['pps_skin_left']
            pps_skin_right = d['pps_skin_right']

        # ----------------------------------------------------------------------

    if raw_viz or interp_viz: plt.show()
        
    # Each iteration may have a different number of time steps, so we extrapolate so they are all consistent
    if isTrainingData:
        # Find the largest iteration
        max_size = max([ len(x) for x in data_dict['timesList'] ])
        # Extrapolate each time step
        for key in data_dict.keys():
            if data_dict[key] == []: continue
            data_dict[key] = extrapolateData(data_dict[key], max_size)

    return data_dict
    
    
def getSubjectFileList(root_path, subject_names, task_name):
    # List up recorded files
    folder_list = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path,d))]        

    success_list = []
    failure_list = []
    for d in folder_list:

        name_flag = False
        for name in subject_names:
            if d.find(name) >= 0: name_flag = True
                                    
        if name_flag and d.find(task_name) >= 0:
            files = os.listdir(os.path.join(root_path,d))

            for f in files:
                # pickle file name with full path
                pkl_file = os.path.join(root_path,d,f)
                
                if f.find('success') >= 0:
                    if len(success_list)==0: success_list = [pkl_file]
                    else: success_list.append(pkl_file)
                elif f.find('failure') >= 0:
                    if len(failure_list)==0: failure_list = [pkl_file]
                    else: failure_list.append(pkl_file)
                else:
                    print "It's not success/failure file: ", f

    print "--------------------------------------------"
    print "# of Success files: ", len(success_list)
    print "# of Failure files: ", len(failure_list)
    print "--------------------------------------------"
    
    return success_list, failure_list


def interpolationData(time_array, data_array, new_time_array):
    '''
    time_array: N - length array
    data_array: D x N - length array
    '''

    from scipy import interpolate

    n,m = np.shape(data_array)
    if len(time_array) > m:
        time_array = time_array[0:m]

    new_data_array = None    
    for i in xrange(n):
        interp = interpolate.splrep(time_array, data_array[i], s=0)
        interp_data = interpolate.splev(new_time_array, interp, der=0)
        
        if new_data_array is None:
            new_data_array = interp_data
        else:
            new_data_array = np.vstack([new_data_array, interp_data])

    return new_data_array
    
def interpolationQuatData(time_array, data_array, new_time_array):
    '''
    We have to use SLERP for start-goal quaternion interpolation.
    But, I cound not find any good library for quaternion array interpolation.
    time_array: N - length array
    data_array: 4 x N - length array
    '''
    from scipy import interpolate

    n,m = np.shape(data_array)
    if len(time_array) > m:
        time_array = time_array[0:m]
    
    new_data_array = None    

    l     = len(time_array)
    new_l = len(new_time_array)

    idx_list = np.linspace(0, l-1, new_l)

    for idx in idx_list:
        if new_data_array is None:
            new_data_array = qt.slerp( data_array[:,int(idx)], data_array[:,int(np.ceil(idx))], idx-int(idx) )
        else:
            new_data_array = np.vstack([new_data_array, 
                                        qt.slerp( data_array[:,int(idx)], data_array[:,int(np.ceil(idx))], \
                                                  idx-int(idx) )])                            
                                                  
    return new_data_array.T

    
def scaleData(data_dict, scale=10, data_min=None, data_max=None, verbose=False):

    if data_dict == {}: return {}
    
    # Determine max and min values
    if data_min is None or data_max is None:
        data_min = {}
        data_max = {}
        for key in data_dict.keys():
            if 'time' in key or 'Quat' in key: continue
            if data_dict[key] == []: continue            
            data_min[key] = np.min(data_dict[key])
            data_max[key] = np.max(data_dict[key])
            
        if verbose:
            print 'minValues', data_min
            print 'maxValues', data_max

    data_dict_scaled = {}
    for key in data_dict.keys():
        if data_dict[key] == []: continue
        if 'time' in key or 'Quat' in key: 
            data_dict_scaled[key] = data_dict[key]
        else:
            data_dict_scaled[key] = (data_dict[key] - data_min[key])/(data_max[key]-data_min[key]) * scale
        
    return data_dict_scaled


def getAngularSpatialRF(cur_pos, dist_margin ):

    dist = np.linalg.norm(cur_pos)
    ang_margin = np.arcsin(dist_margin/dist)

    cur_pos /= np.linalg.norm(cur_pos)
    ang_cur  = np.arccos(cur_pos[1]) - np.pi/2.0

    ang_margin = 10.0 * np.pi/180.0

    ang_max = ang_cur + ang_margin
    ang_min = ang_cur - ang_margin

    return ang_max, ang_min



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
            ## data_viz.ft_disp(timeList, ftForce, unimodal_ftForce)
            ## sys.exit()

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

    ## import matplotlib.pyplot as plt
    ## plt.figure()
    ## plt.plot(np.array(scaled_features[0]).T)
    ## plt.show()
    ## sys.exit()
                                
    return scaled_features, param_dict
