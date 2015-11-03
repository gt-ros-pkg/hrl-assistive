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
import rospy, roslib
import os, sys, copy

# util
import numpy as np
import hrl_lib.util as ut
import hrl_lib.quaternion as qt

from scipy import interpolate
from sklearn.decomposition import PCA

import matplotlib
## matplotlib.use('pdf')
import matplotlib.pyplot as plt
## import data_viz

def extrapolateData(data, maxsize):
    if len(np.shape(data[0])) > 1:     
        # need to implement incremental extrapolation
        return [x if len(x[0]) >= maxsize else x + [x[:,-1]]*(maxsize-len(x[0])) for x in data]
    else:
        # need to implement incremental extrapolation        
        return [x if len(x) >= maxsize else x + [x[-1]]*(maxsize-len(x)) for x in data]
        

def loadData(fileNames, isTrainingData=False, downSampleSize=100, \
             verbose=False, renew=True, save_pkl=None):

    if save_pkl is not None:
        if os.path.isfile(save_pkl+'_raw.pkl') is True and os.path.isfile(save_pkl+'_interp.pkl') is True \
          and renew is not True:
            raw_data_dict = ut.load_pickle(save_pkl+'_raw.pkl')
            data_dict = ut.load_pickle(save_pkl+'_interp.pkl')
            return raw_data_dict, data_dict

    key_list = ['timesList',\
                'audioTimesList', 'audioAzimuthList', 'audioPowerList',\
                'kinTimesList', 'kinEEPosList', 'kinEEQuatList', 'kinJntPosList', 'kinTargetPosList', \
                'kinTargetQuatList', \
                'ftTimesList', 'ftForceList', \
                'visionTimesList', 'visionPosList', 'visionQuatList', \
                'ppsTimesList', 'ppsLeftList', 'ppsRightList',\
                'fabricTimesList', 'fabricCenterList', 'fabricNormalList', 'fabricValueList' ]

    raw_data_dict = {}
    data_dict = {}
    for key in key_list:
        raw_data_dict[key] = []
        data_dict[key]     = []
    
    for idx, fileName in enumerate(fileNames):
        if os.path.isdir(fileName):
            continue

        if verbose: print fileName
        d = ut.load_pickle(fileName)        
        init_time = d['init_time']

        max_time = 0
        for key in d.keys():
            if 'time' in key and 'init' not in key:
                feature_time = d[key]
                if max_time < feature_time[-1]-init_time: max_time = feature_time[-1]-init_time
        new_times = np.linspace(0.01, max_time, downSampleSize)
        data_dict['timesList'].append(new_times)

        # sound ----------------------------------------------------------------
        if 'audio_time' in d.keys():
            audio_time    = (np.array(d['audio_time']) - init_time).tolist()
            audio_azimuth = d['audio_azimuth']
            audio_power   = d['audio_power']

            raw_data_dict['audioTimesList'].append(audio_time)
            raw_data_dict['audioAzimuthList'].append(audio_azimuth)
            raw_data_dict['audioPowerList'].append(audio_power)

            data_dict['audioAzimuthList'].append(interpolationData(audio_time, audio_azimuth, new_times))
            data_dict['audioPowerList'].append(interpolationData(audio_time, audio_power, new_times))


        # kinematics -----------------------------------------------------------
        if 'kinematics_time' in d.keys():
            kin_time        = (np.array(d['kinematics_time']) - init_time).tolist()
            kin_ee_pos      = d['kinematics_ee_pos'] # 3xN
            kin_ee_quat     = d['kinematics_ee_quat'] # ?xN
            kin_target_pos  = d['kinematics_target_pos']
            kin_target_quat = d['kinematics_target_quat']
            kin_jnt_pos     = d['kinematics_jnt_pos'] # 7xN

            raw_data_dict['kinTimesList'].append(kin_time)
            raw_data_dict['kinEEPosList'].append(kin_ee_pos)
            raw_data_dict['kinEEQuatList'].append(kin_ee_quat)
            raw_data_dict['kinTargetPosList'].append(kin_target_pos)
            raw_data_dict['kinTargetQuatList'].append(kin_target_quat)
            raw_data_dict['kinJntPosList'].append(kin_jnt_pos)
                    
            data_dict['kinEEPosList'].append(interpolationData(kin_time, kin_ee_pos, new_times))
            data_dict['kinEEQuatList'].append(interpolationData(kin_time, kin_ee_quat, new_times))
            data_dict['kinTargetPosList'].append(interpolationData(kin_time, kin_target_pos, new_times))
            data_dict['kinTargetQuatList'].append(interpolationData(kin_time, kin_target_quat, new_times))
            data_dict['kinJntPosList'].append(interpolationData(kin_time, kin_jnt_pos, new_times))
            

        # ft -------------------------------------------------------------------
        if 'ft_time' in d.keys():
            ft_time        = (np.array(d['ft_time']) - init_time).tolist()
            ft_force_array = d['ft_force']

            raw_data_dict['ftTimesList'].append(ft_time)
            raw_data_dict['ftForceList'].append(ft_force_array)

            force_array = interpolationData(ft_time, ft_force_array, new_times)
            data_dict['ftForceList'].append(force_array)                                         
            
                    
        # vision ---------------------------------------------------------------
        if 'vision_time' in d.keys():
            vision_time = (np.array(d['vision_time']) - init_time).tolist()
            vision_pos  = d['vision_pos']
            vision_quat = d['vision_quat']

            raw_data_dict['visionTimesList'].append(vision_time)
            raw_data_dict['visionPosList'].append(vision_pos)
            raw_data_dict['visionQuatList'].append(vision_quat)

            vision_pos_array  = interpolationData(vision_time, vision_pos, new_times)
            data_dict['visionPosList'].append(vision_pos_array)                                         
            
            vision_quat_array = interpolationQuatData(vision_time, vision_quat, new_times)
            data_dict['visionQuatList'].append(vision_quat_array)                                         


        # pps ------------------------------------------------------------------
        if 'pps_skin_time' in d.keys():
            pps_skin_time  = (np.array(d['pps_skin_time']) - init_time).tolist()
            pps_skin_left  = d['pps_skin_left']
            pps_skin_right = d['pps_skin_right']

            raw_data_dict['ppsTimesList'].append(pps_skin_time)
            raw_data_dict['ppsLeftList'].append(pps_skin_left)
            raw_data_dict['ppsRightList'].append(pps_skin_right)

            left_array = interpolationData(pps_skin_time, pps_skin_left, new_times)
            data_dict['ppsLeftList'].append(left_array)
            right_array = interpolationData(pps_skin_time, pps_skin_right, new_times)
            data_dict['ppsRightList'].append(right_array)


        # fabric skin ------------------------------------------------------------------
        if 'fabric_skin_time' in d.keys():
            fabric_skin_time      = (np.array(d['fabric_skin_time']) - init_time).tolist()
            fabric_skin_centers_x = d['fabric_skin_centers_x']
            fabric_skin_centers_y = d['fabric_skin_centers_y']
            fabric_skin_centers_z = d['fabric_skin_centers_z']
            fabric_skin_normals_x = d['fabric_skin_normals_x']
            fabric_skin_normals_y = d['fabric_skin_normals_y']
            fabric_skin_normals_z = d['fabric_skin_normals_z']
            fabric_skin_values_x  = d['fabric_skin_values_x']
            fabric_skin_values_y  = d['fabric_skin_values_y']
            fabric_skin_values_z  = d['fabric_skin_values_z']

            # time weighted sum?
            raw_data_dict['fabricTimesList'].append(fabric_skin_time)
            raw_data_dict['fabricCenterList'].append([fabric_skin_centers_x,\
                                                      fabric_skin_centers_y,\
                                                      fabric_skin_centers_z])
            raw_data_dict['fabricNormalList'].append([fabric_skin_normals_x,\
                                                      fabric_skin_normals_y,\
                                                      fabric_skin_normals_z])
            raw_data_dict['fabricValueList'].append([fabric_skin_values_x,\
                                                     fabric_skin_values_y,\
                                                     fabric_skin_values_z])
                                                     
            # skin interpolation
            center_array, normal_array, value_array \
              = interpolationSkinData(fabric_skin_time, \
                                      raw_data_dict['fabricCenterList'][-1],\
                                      raw_data_dict['fabricNormalList'][-1],\
                                      raw_data_dict['fabricValueList'][-1], new_times )
                
            data_dict['fabricCenterList'].append(center_array)
            data_dict['fabricNormalList'].append(normal_array)
            data_dict['fabricValueList'].append(value_array)

            
        # ----------------------------------------------------------------------

    # Each iteration may have a different number of time steps, so we extrapolate so they are all consistent
    if isTrainingData:
        # Find the largest iteration
        max_size = max([ len(x) for x in data_dict['timesList'] ])
        # Extrapolate each time step
        for key in data_dict.keys():
            if data_dict[key] == []: continue
            data_dict[key] = extrapolateData(data_dict[key], max_size)

    if save_pkl is not None:
        ut.save_pickle(raw_data_dict, save_pkl+'_raw.pkl')
        ut.save_pickle(data_dict, save_pkl+'_interp.pkl')

    return raw_data_dict, data_dict
    
    
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

    if len(np.shape(data_array)) == 1: data_array = np.array([data_array])

    n,m = np.shape(data_array)
    if len(time_array) > m:
        time_array = time_array[0:m]

    print np.shape(data_array), np.shape(time_array)

    # remove repeated data
    temp_time_array = [time_array[0]]
    temp_data_array = data_array[:,0:1]
    for i in xrange(1, len(time_array)):
        if time_array[i-1] != time_array[i]:
            temp_time_array.append(time_array[i])
            temp_data_array = np.hstack([temp_data_array, data_array[:,i:i+1]])

    time_array = temp_time_array
    data_array = temp_data_array

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

def interpolationSkinData(time_array, center_array, normal_array, value_array, new_time_array):
    '''
    Interpolate haptic msg
    '''
    from scipy import interpolate

    new_c_arr = []
    new_n_arr = []
    new_v_arr = []

    ths   = 0.025 
    l     = len(time_array)
    new_l = len(new_time_array)

    if l == 0 or len(center_array[0]) == 0: return [],[],[]

    if len(np.array(center_array[0]).flatten()) == 0: return [],[],[]

    idx_list = np.linspace(0, l-2, new_l)
    for idx in idx_list:
        w1 = idx-int(idx)
        w2 = 1.0 - w1

        idx1 = int(idx)
        idx2 = int(idx)+1

        c1 = np.array(center_array)[:,idx1]
        c2 = np.array(center_array)[:,idx2]
        n1 = np.array(normal_array)[:,idx1]
        n2 = np.array(normal_array)[:,idx2]
        v1 = np.array(value_array)[:,idx1]
        v2 = np.array(value_array)[:,idx2]      

        if c1[0] == []:            
            if c2[0] == []:
                c = []
                n = []
                v = []
            else:
                c = c2.tolist() #w2*np.array(c2)
                n = n2.tolist() #w2*np.array(n2)
                v = v2.tolist() #w2*np.array(v2)
        else:
            if c2[0] == []:
                c = c1.tolist() #w1*np.array(c1)
                n = n1.tolist() #w1*np.array(n1)
                v = v1.tolist() #w1*np.array(v1)
            else:
                c1 = np.array(c1.tolist())
                c2 = np.array(c2.tolist())
                n1 = np.array(n1.tolist())
                n2 = np.array(n2.tolist())
                v1 = np.array(v1.tolist())
                v2 = np.array(v2.tolist())
                
                c = None
                n = None
                v = None
                close_idxes = []
                for i in xrange(len(c1[0])):
                    close_idx = None
                    for j in xrange(len(c2[0])):
                        if np.linalg.norm(c1[:,i] - c2[:,j]) < ths:
                            close_idx = j
                            close_idxes.append(j)
                            break

                    if close_idx is None:                        
                        if c is None:
                            c = c1[:,i:i+1]
                            n = n1[:,i:i+1]
                            v = v1[:,i:i+1]
                        else:
                            c = np.hstack([c, c1[:,i:i+1]])
                            n = np.hstack([n, n1[:,i:i+1]])
                            v = np.hstack([v, v1[:,i:i+1]])
                    else:
                        if c is None:                        
                            c = w1*c1[:,i:i+1] + w2*c2[:,close_idx:close_idx+1]
                            n = w1*n1[:,i:i+1] + w2*n2[:,close_idx:close_idx+1]
                            v = w1*v1[:,i:i+1] + w2*v2[:,close_idx:close_idx+1]
                        else:
                            c = np.hstack([c, w1*c1[:,i:i+1] + w2*c2[:,close_idx:close_idx+1]])
                            n = np.hstack([n, w1*n1[:,i:i+1] + w2*n2[:,close_idx:close_idx+1]])
                            v = np.hstack([v, w1*v1[:,i:i+1] + w2*v2[:,close_idx:close_idx+1]])

                if len(close_idxes) < len(c2[0]):
                    for i in xrange(len(c2[0])):
                        if i not in close_idxes:
                            if c is None:
                                c = c2[:,i:i+1]
                                n = n2[:,i:i+1]
                                v = v2[:,i:i+1]
                            else:
                                c = np.hstack([c, c2[:,i:i+1]])
                                n = np.hstack([n, n2[:,i:i+1]])
                                v = np.hstack([v, v2[:,i:i+1]])
                          
                c = c.tolist()
                n = n.tolist()
                v - v.tolist()
                
        new_c_arr.append(c)
        new_n_arr.append(n)
        new_v_arr.append(v)
            
    return new_c_arr, new_n_arr, new_v_arr
    
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
