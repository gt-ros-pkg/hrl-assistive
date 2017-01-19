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



def extract_feature(msg, last_msg, handFeatures, param_dict):
    ''' Run it on every time step '''

    d = {}
    d['timesList'] = [[0.]]

    # Unimodal feature - AudioWrist ---------------------------------------
    if 'unimodal_audioWristRMS' in handFeatures:
        d['audioWristRMSList'] = [msg.audio_wrist_rms]

    # Unimoda feature - AudioWristFront------------------------------------
    if 'unimodal_audioWristFrontRMS' in handFeatures:
        ang_range = 15.0
        audio_front_rms = msg.audio_wrist_rms*stats.norm.pdf(
            msg.audio_wrist_azimuth,scale=ang_range)\
          /stats.norm.pdf(0.0,scale=ang_range)            
        d['audioWristFrontRMSList'] = [audio_front_rms]

    # Unimoda feature - AudioWristAzimuth------------------------------------
    if 'unimodal_audioWristAzimuth' in handFeatures:
        d['audioWristAzimuthList'] = [msg.audio_azimuth]

    # Unimodal feature - Kinematics --------------------------------------
    if 'unimodal_kinVel' in handFeatures:
        d['kinEEPosList_last'] = [ np.array([last_msg.kinematics_ee_pos]).T ]

    # Unimodal feature - Kinematics --------------------------------------
    if 'unimodal_kinJntEff_1' in handFeatures:
        d['kinJntEffList'] = [msg.kinematics_jnt_eff]

    # Unimodal feature - Force -------------------------------------------
    if 'unimodal_ftForce' in handFeatures or 'unimodal_ftForce_zero' in handFeatures or\
        'unimodal_ftForce_integ' in handFeatures or 'unimodal_ftForceX' in handFeatures or\
        'unimodal_ftForceY' in handFeatures or 'unimodal_ftForceZ' in handFeatures:
        d['ftForceList']  = [np.array([msg.ft_force]).T]
        ## d['ftTorqueList'] = [np.array([msg.ft_torque]).T]

    # Unimodal feature - fabric skin ------------------------------------
    if 'unimodal_fabricForce' in handFeatures:
        fabric_skin_values  = [msg.fabric_skin_values_x,
                               msg.fabric_skin_values_y, \
                               msg.fabric_skin_values_z]
        if not fabric_skin_values[0]:
            d['fabricMagList'] = [0]
        else:
            d['fabricMagList'] = [np.sum( np.linalg.norm(np.array(fabric_skin_values), axis=0) )]

    # Unimodal feature - landmark motion --------------------------
    if 'unimodal_landmarkDist' in handFeatures:
        d['visionLandmarkPosList'] = [np.array([msg.vision_landmark_pos]).T]

    # Unimodal feature - EE change --------------------------
    if 'unimodal_kinEEChange' in handFeatures:
        d['kinEEPosList']  = [np.array([msg.kinematics_ee_pos]).T]

    # Unimodal feature - Desired EE change --------------------------
    if 'unimodal_kinDesEEChange' in handFeatures:
        d['kinEEPosList']  = [np.array([msg.kinematics_ee_pos]).T]
        d['kinDesEEPosList'] = [np.array([msg.kinematics_des_ee_pos]).T]
    
    # Crossmodal feature - relative dist --------------------------
    if 'crossmodal_targetEEDist' in handFeatures:
        d['kinEEPosList']     = [np.array([msg.kinematics_ee_pos]).T]
        d['kinTargetPosList'] = [np.array([msg.kinematics_target_pos]).T]

    # Crossmodal feature - relative Velocity --------------------------
    if 'crossmodal_targetEEVel' in handFeatures:
        rospy.loginfo( "Not available")

    # Crossmodal feature - relative angle --------------------------
    if 'crossmodal_targetEEAng' in handFeatures:
        d['kinEEQuatList'] = [np.array([msg.kinematics_ee_quat]).T]
        d['kinTargetQuatList'] = [np.array([msg.kinematics_target_quat]).T]

    # Crossmodal feature - vision relative dist with main(first) vision target----
    if 'crossmodal_artagEEDist' in handFeatures:
        d['kinEEPosList']     = [np.array([msg.kinematics_ee_pos]).T]
        d['visionArtagPosList'] = [np.array([msg.vision_artag_pos]).T]

    # Crossmodal feature - vision relative angle --------------------------
    if 'crossmodal_artagEEAng' in handFeatures:
        d['kinEEQuatList'] = [np.array([msg.kinematics_ee_quat]).T]
        d['visionArtagPosList'] = [np.array([msg.vision_artag_pos]).T]
        d['visionArtagQuatList'] = [np.array([msg.vision_artag_quat]).T]

    # Crossmodal feature - vision relative dist with sub vision target----
    if 'crossmodal_subArtagEEDist' in handFeatures:
        rospy.loginfo( "Not available" )

    # Crossmodal feature - vision relative angle --------------------------
    if 'crossmodal_subArtagEEAng' in handFeatures:                
        rospy.loginfo( "Not available" )

    # Crossmodal feature - vision relative dist with main(first) vision target----
    if 'crossmodal_landmarkEEDist' in handFeatures:
        d['kinEEPosList']     = [np.array([msg.kinematics_ee_pos]).T]
        d['visionLandmarkPosList'] = [np.array([msg.vision_landmark_pos]).T]

    # Crossmodal feature - vision relative angle --------------------------
    if 'crossmodal_landmarkEEAng' in handFeatures:
        d['kinEEQuatList'] = [np.array([msg.kinematics_ee_quat]).T]
        d['visionLandmarkPosList'] = [np.array([msg.vision_landmark_pos]).T]
        d['visionLandmarkQuatList'] = [np.array([msg.vision_landmark_quat]).T]

    data, _ = dm.extractHandFeature(d, handFeatures, init_param_dict = param_dict)

    return data


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
