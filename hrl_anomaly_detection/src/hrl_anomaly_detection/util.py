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

from scipy import interpolate


def loadData(fileNames, isTrainingData=False, downSampleSize=100, verbose=False):

    data_dict = {}
    data_dict['timesList']        = []
    data_dict['audioAzimuthList'] = []    
    data_dict['audioPowerList']   = []    
    data_dict['kinEEPosList']     = []
    data_dict['kinEEQuatList']    = []
    data_dict['ftForceList']      = []
    data_dict['kinTargetPosList']  = []
    data_dict['kinTargetQuatList'] = []
    
    for idx, fileName in enumerate(fileNames):
        if os.path.isdir(fileName):
            continue

        d = ut.load_pickle(fileName)        
        print d.keys()

        kin_time = d['kinematics_time']
        new_times = np.linspace(0.01, kin_time[-1], downSampleSize)
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

            ee_pos_array = interpolationData(kin_time, kin_ee_pos, new_times)
            data_dict['kinEEPosList'].append(ee_pos_array)                                         

            ee_quat_array = interpolationQuatData(kin_time, kin_ee_quat, new_times)
            data_dict['kinEEQuatList'].append(ee_quat_array)                                         

            target_pos_array = interpolationData(kin_time, kin_target_pos, new_times)
            data_dict['kinTargetPosList'].append(target_pos_array)                                         

            target_quat_array = interpolationQuatData(kin_time, kin_target_quat, new_times)
            data_dict['kinTargetQuatList'].append(target_quat_array)                                         
            
        # ft -------------------------------------------------------------------
        if 'ft_time' in d.keys():
            ft_time        = d['ft_time']
            ft_force_array = d['ft_force']

            force_array = interpolationData(ft_time, ft_force_array, new_times)
            data_dict['ftForceList'].append(force_array)                                         
            
        # vision ---------------------------------------------------------------
        if 'vision_time' in d.keys():
            vision_time = d['vision_time']
            vision_pos  = d['vision_pos']
            vision_quat = d['vision_quat']

            vision_pos_array  = interpolationData(vision_time, vision_pos, new_times)
            vision_quat_array = interpolationQuatData(vision_time, vision_quat, new_times)
            
        # pps ------------------------------------------------------------------
        if 'pps_skin_time' in d.keys():
            pps_skin_time  = d['pps_skin_time']
            pps_skin_left  = d['pps_skin_left']
            pps_skin_right = d['pps_skin_right']

        # ----------------------------------------------------------------------
        
                
    ## if isTrainingData:
        
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
    We have to use SLERP, but I cound not find a good library for quaternion array.
    time_array: N - length array
    data_array: 4 x N - length array
    '''
    from scipy import interpolate

    n,m = np.shape(data_array)

    new_data_array = None    
    
    if len(time_array) > len(new_time_array)*2.0:

        l     = len(time_array)
        new_l = len(new_time_array)

        idx_list = np.linspace(0, l-1, new_l)

        for idx in idx_list:
            if new_data_array is None:
                new_data_array = data_array[:,0]
            else:
                new_data_array = np.hstack([new_data_array, data_array[:,idx]])        
    else:
        print "quaternion array interpolation is not implemented"
        sys.exit()

    return new_data_array
    
