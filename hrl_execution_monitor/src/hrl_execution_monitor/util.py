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

#  \author Daehyung Park (Healthcare Robotics Lab, Georgia Tech.)

# system & utils
import os, sys, copy
import scipy, numpy as np

from hrl_anomaly_detection import data_manager as dm



def extract_feature(msg, init_msg, last_msg, last_data, handFeatures, param_dict, count=0):
    ''' Run it on every time step '''
    dataSample = []

    dt = msg.header.stamp.to_sec() - last_msg.header.stamp.to_sec()

    # Unimodal feature - AudioWrist ---------------------------------------
    if 'unimodal_audioWristRMS' in handFeatures:
        dataSample.append(msg.audio_wrist_rms-init_msg.audio_wrist_rms)
        if 'audioWristRMS' not in param_dict['feature_names']:
            param_dict['feature_names'].append('audioWristRMS')

    # Unimoda feature - AudioWristFront------------------------------------
    if 'unimodal_audioWristFrontRMS' in handFeatures:
        ang_range = 15.0
        audio_front_rms = msg.audio_wrist_rms*stats.norm.pdf(
            msg.audio_wrist_azimuth,scale=ang_range)\
          /stats.norm.pdf(0.0,scale=ang_range)
        dataSample.append(audio_front_rms)           
        if 'audioWristFrontRMS' not in param_dict['feature_names']:
            param_dict['feature_names'].append('audioWristFrontRMS')

    # Unimoda feature - AudioWristAzimuth------------------------------------
    if 'unimodal_audioWristAzimuth' in handFeatures:
        dataSample.append( abs(msg.audio_azimuth) )
        if 'audioWristAzimuth' not in param_dict['feature_names']:
            param_dict['feature_names'].append('audioWristAzimuth')

    # Unimodal feature - Kinematics --------------------------------------
    if 'unimodal_kinVel' in handFeatures:
        vel = np.array(msg.kinematics_ee_pos) - np.array(last_msg.kinematics_ee_pos)
        vel = np.linalg.norm(vel)/dt
        dataSample.append( vel )
        if 'kinVel' not in param_dict['feature_names']:
            param_dict['feature_names'].append('kinVel')

    # Unimodal feature - Kinematics --------------------------------------
    if 'unimodal_kinJntEff_1' in handFeatures:
        jnt_idx = 0
        dataSample.append( msg.kinematics_jnt_eff[jnt_idx] - init_msg.kinematics_jnt_eff[jnt_idx] )
        if 'kinJntEff_1' not in param_dict['feature_names']:           
            param_dict['feature_names'].append( 'kinJntEff_1')

    # Unimodal feature - Force -------------------------------------------
    if 'unimodal_ftForce_zero' in handFeatures:      
        mag = np.linalg.norm(np.array(msg.ft_force) - np.array(init_msg.ft_force) )
        dataSample.append(mag)
        if 'ftForce_mag_zero' not in param_dict['feature_names']:
            param_dict['feature_names'].append('ftForce_mag_zero')

    # Unimodal feature - Force integ -------------------------------------------
    if 'unimodal_ftForce_integ' in handFeatures:
        mag = np.linalg.norm(np.array(msg.ft_force) - np.array(init_msg.ft_force))
        ## mag -= init_mag

        if count >0:
            fidx = param_dict['feature_names'].tolist().index('ftForce_mag_integ')

            last_mag = np.linalg.norm(np.array(last_msg.ft_force) - np.array(init_msg.ft_force))
            mag = last_data[fidx] + (mag + last_mag )*dt/2.0

            fidx = param_dict['feature_names'].tolist().index('ftForce_mag_integ')
            ## print count, " : ", last_data[fidx], dt, " => ", mag

        dataSample.append(mag)
        if 'ftForce_mag_integ' not in param_dict['feature_names']:
            param_dict['feature_names'].append('ftForce_mag_integ')

    # Unimodal feature - Force -------------------------------------------
    if 'unimodal_ftForceX' in handFeatures:
        dataSample.append( msg.ft_force[0]-init_msg.ft_force[0] )
        if 'ftForce_x' not in param_dict['feature_names']:
            param_dict['feature_names'].append('ftForce_x')

    # Unimodal feature - Force -------------------------------------------
    if 'unimodal_ftForceY' in handFeatures:
        dataSample.append( msg.ft_force[1]-init_msg.ft_force[1] )
        if 'ftForce_y' not in param_dict['feature_names']:
            param_dict['feature_names'].append('ftForce_y')

    # Unimodal feature - Force -------------------------------------------
    if 'unimodal_ftForceZ' in handFeatures:
        dataSample.append( msg.ft_force[2]-init_msg.ft_force[2] )
        if 'ftForce_z' not in param_dict['feature_names']:
            param_dict['feature_names'].append('ftForce_z')

    # Unimodal feature - fabric skin ------------------------------------
    if 'unimodal_fabricForce' in handFeatures:
        fabric_skin_values  = [msg.fabric_skin_values_x,
                               msg.fabric_skin_values_y, \
                               msg.fabric_skin_values_z]
        if not fabric_skin_values[0]:
            mag = 0
        else:
            mag = np.sum( np.linalg.norm(np.array(fabric_skin_values), axis=0) )

        fabric_skin_values  = [init_msg.fabric_skin_values_x,
                               init_msg.fabric_skin_values_y, \
                               init_msg.fabric_skin_values_z]
        if not fabric_skin_values[0]:
            init_mag = 0
        else:
            init_mag = np.sum( np.linalg.norm(np.array(fabric_skin_values), axis=0) )

        dataSample.append( mag - init_mag )
        if 'fabricForce' not in param_dict['feature_names']:
            param_dict['feature_names'].append('fabricForce')

    # Unimodal feature - landmark motion --------------------------
    if 'unimodal_landmarkDist' in handFeatures:
        dist = np.linalg.norm(msg.vision_landmark_pos[:3])-np.linalg.norm(init_msg.vision_landmark_pos[:3])
        dataSample.append(dist)
        if 'landmarkDist' not in param_dict['feature_names']:
            param_dict['feature_names'].append('landmarkDist')

    # Unimodal feature - EE change --------------------------
    if 'unimodal_kinEEChange' in handFeatures:
        dist = np.linalg.norm(np.array(msg.kinematics_ee_pos) - np.array(init_msg.kinematics_ee_pos))
        dataSample.append(dist)
        if 'EEChange' not in param_dict['feature_names']:
            param_dict['feature_names'].append('EEChange')

    # Unimodal feature - Desired EE change --------------------------
    if 'unimodal_kinDesEEChange' in handFeatures:
        dist = np.linalg.norm(np.array(msg.kinematics_ee_pos) -
                              np.array(msg.kinematics_des_ee_pos))
        init_dist = np.linalg.norm(np.array(init_msg.kinematics_ee_pos) -
                                   np.array(init_msg.kinematics_des_ee_pos))
        
        dataSample.append(dist-init_dist)
        if 'DesEEChange' not in param_dict['feature_names']:
            param_dict['feature_names'].append('DesEEChange')

    ## # Crossmodal feature - relative dist --------------------------
    ## if 'crossmodal_targetEEDist' in handFeatures:
    ##     d['kinEEPosList']     = [np.array([msg.kinematics_ee_pos]).T]
    ##     d['kinTargetPosList'] = [np.array([msg.kinematics_target_pos]).T]

    ## # Crossmodal feature - relative angle --------------------------
    ## if 'crossmodal_targetEEAng' in handFeatures:
    ##     d['kinEEQuatList'] = [np.array([msg.kinematics_ee_quat]).T]
    ##     d['kinTargetQuatList'] = [np.array([msg.kinematics_target_quat]).T]

    # Crossmodal feature - vision relative dist with main(first) vision target----
    if 'crossmodal_landmarkEEDist' in handFeatures:
        dist = np.linalg.norm(np.array(msg.vision_landmark_pos)[:3] -
                              np.array(msg.kinematics_ee_pos))
        dist -= np.linalg.norm(np.array(init_msg.vision_landmark_pos)[:3] -
                               np.array(init_msg.kinematics_ee_pos))
        dataSample.append(dist)
        if 'landmarkEEDist' not in param_dict['feature_names']:
            param_dict['feature_names'].append('landmarkEEDist')

    # Crossmodal feature - vision relative angle --------------------------
    if 'crossmodal_landmarkEEAng' in handFeatures:
        d['kinEEQuatList'] = [np.array([msg.kinematics_ee_quat]).T]
        d['visionLandmarkPosList'] = [np.array([msg.vision_landmark_pos]).T]
        d['visionLandmarkQuatList'] = [np.array([msg.vision_landmark_quat]).T]

        startQuat = np.array(msg.kinematics_ee_quat)
        endQuat   = np.array(msg.vision_landmark_quat)[:4]
        diff_ang  = abs(qt.quat_angle(startQuat, endQuat))

        startQuat = np.array(init_msg.kinematics_ee_quat)
        endQuat   = np.array(init_msg.vision_landmark_quat)[:4]
        diff_ang -= abs(qt.quat_angle(startQuat, endQuat))
        dataSample.append( abs(diff_ang) )

        if 'landmarkEEAng' not in param_dict['feature_names']:
            param_dict['feature_names'].append('landmarkEEAng')

    # --------------------------------------------------------------------
    # scaling?
    scaled_dataSample = []    
    for i in xrange(len(dataSample)):
        if abs( param_dict['feature_max'][i] - param_dict['feature_min'][i]) < 1e-3:
            scaled_dataSample.append( dataSample[i] )
        else:
            scaled_dataSample.append( ( dataSample[i] - param_dict['feature_min'][i] )\
                                    /( param_dict['feature_max'][i] - param_dict['feature_min'][i]) )

    return dataSample, scaled_dataSample


def reset_roc_data(ROC_data, method_list, update_list, nPoints):

    for i, method in enumerate(method_list):
        if method not in ROC_data.keys() or method in update_list: 
        ## if method not in ROC_data.keys() or method in ROC_dict['update_list']: 
            ROC_data[method] = {}
            ROC_data[method]['complete'] = False 
            ROC_data[method]['tp_l'] = [ [] for j in xrange(nPoints) ]
            ROC_data[method]['fp_l'] = [ [] for j in xrange(nPoints) ]
            ROC_data[method]['tn_l'] = [ [] for j in xrange(nPoints) ]
            ROC_data[method]['fn_l'] = [ [] for j in xrange(nPoints) ]
            ROC_data[method]['delay_l'] = [ [] for j in xrange(nPoints) ]
            ROC_data[method]['tp_delay_l'] = [ [] for j in xrange(nPoints) ]
            ROC_data[method]['tp_idx_l'] = [ [] for j in xrange(nPoints) ]
            ROC_data[method]['fn_labels'] = [ [] for j in xrange(nPoints) ]
    
    return ROC_data

def update_roc_data(ROC_data, new_data, nPoints, method_list):

    for i in xrange(len(new_data)):
        for j in xrange(nPoints):
            try:
                method = new_data[i].keys()[0]
            except:                
                print "Error when collect ROC data:", new_data[i]
                sys.exit()
            if ROC_data[method]['complete'] == True: continue
            ROC_data[method]['tp_l'][j] += new_data[i][method]['tp_l'][j]
            ROC_data[method]['fp_l'][j] += new_data[i][method]['fp_l'][j]
            ROC_data[method]['tn_l'][j] += new_data[i][method]['tn_l'][j]
            ROC_data[method]['fn_l'][j] += new_data[i][method]['fn_l'][j]
            ROC_data[method]['delay_l'][j] += new_data[i][method]['delay_l'][j]
            ROC_data[method]['tp_delay_l'][j].append( new_data[i][method]['delay_l'][j] )
            ROC_data[method]['tp_idx_l'][j].append( new_data[i][method]['tp_idx_l'][j] )
            ROC_data[method]['fn_labels'][j] += new_data[i][method]['fn_labels'][j]

    for i, method in enumerate(method_list):
        ROC_data[method]['complete'] = True

    return ROC_data

def image_list_flatten(image_list):
    ''' flatten image list '''
    if len(image_list) == 0: return []
    new_list = []
    for i in xrange(len(image_list)):
        for j in xrange(len(image_list[i])):
            new_list.append(image_list[i][j])
    return np.array(new_list)

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N


def temporal_features(X, max_step, ml, scale):
    ''' Return n_step x n_features'''

    d_idx = len(X[0])
    while True:
        v = ml.conditional_prob( X[:,:d_idx]*scale)
        if v is None: d_idx -= 1
        else: break

    vs = None
    for i in xrange(d_idx, d_idx-max_step,-1):
        if i<=0: continue
        v = ml.conditional_prob( X[:,:i]*scale)
        v = v.reshape((1,) + v.shape)
        if vs is None: vs = v
        else:          vs = np.vstack([vs, v])
    return vs
