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
import threading

# util
import numpy as np
import hrl_lib.util as ut
from hrl_anomaly_detection.util import *
from sound_play.libsoundplay import SoundClient

# learning
from sklearn.decomposition import PCA
from hrl_multimodal_anomaly_detection.hmm import learning_hmm_multi_n as hmm

# msg
from hrl_srvs.srv import Bool_None, Bool_NoneResponse
from hrl_anomaly_detection.msg import MultiModality
from std_msgs.msg import String

# viz
import matplotlib.pyplot as plt


class anomaly_detector:

    def __init__(self, task_name, feature_list, save_data_path, training_data_pkl, verbose=False, \
                 online_raw_viz=False):
        rospy.init_node(task_name)
        rospy.loginfo('Initializing anomaly detector')

        self.task_name         = task_name
        self.feature_list      = feature_list
        self.save_data_path    = save_data_path
        self.training_data_pkl = os.path.join(save_data_path, training_data_pkl)
        self.online_raw_viz    = online_raw_viz
        self.count = 0

        self.enable_detector = False
        self.soundHandle = SoundClient()

        # visualization
        if self.online_raw_viz: 
            self.fig = plt.figure()
            plt.ion()
            plt.show()
            self.plot_data = {}
        
        self.initParams()
        self.initComms()
        self.initDetector()
        self.reset()

    def initParams(self):
        '''
        Load feature list
        '''
        self.rf_radius = rospy.get_param('hrl_manipulation_task/receptive_field_radius')
        self.nState    = 10
        self.threshold = -5.0
    
    def initComms(self):
        '''
        Subscribe raw data
        '''
        # Publisher
        self.action_interruption_pub = rospy.Publisher('InterruptAction', String)
        
        # Subscriber
        rospy.Subscriber('/hrl_manipulation_task/raw_data', MultiModality, self.rawDataCallback)
        
        # Service
        self.detection_service = rospy.Service('anomaly_detector_enable/'+self.task_name, Bool_None, \
                                               self.enablerCallback)

    def initDetector(self):

        if os.path.isfile(self.training_data_pkl) is not True: 
            print "There is no saved data"
            sys.exit()
        
        data_dict = ut.load_pickle(self.training_data_pkl)
        trainingData, self.param_dict = extractLocalFeature(data_dict['trainData'], self.feature_list, \
                                                            self.rf_radius)

        # training hmm
        self.nEmissionDim = len(trainingData)
        detection_param_pkl = os.path.join(self.save_data_path, 'hmm_'+self.task_name+'.pkl')        
        self.ml = hmm.learning_hmm_multi_n(self.nState, self.nEmissionDim, verbose=False)
        
        ret = self.ml.fit(trainingData, ml_pkl=detection_param_pkl, use_pkl=True)

        if ret == 'Failure': 
            print "-------------------------"
            print "HMM returned failure!!   "
            print "-------------------------"
            sys.exit()

    
    def enablerCallback(self, msg):

        if msg.data is True: 
            self.enable_detector = True
        else:
            # Reset detector
            self.enable_detector = False
            self.reset()            
        
        return Bool_NoneResponse()

        
    def rawDataCallback(self, msg):

        self.audio_feature     = msg.audio_feature
        self.audio_power       = msg.audio_power
        self.audio_azimuth     = msg.audio_azimuth
        self.audio_head_joints = msg.audio_head_joints
        self.audio_cmd         = msg.audio_cmd

        self.kinematics_ee_pos      = msg.kinematics_ee_pos
        self.kinematics_ee_quat     = msg.kinematics_ee_quat
        self.kinematics_jnt_pos     = msg.kinematics_jnt_pos
        self.kinematics_jnt_vel     = msg.kinematics_jnt_vel
        self.kinematics_jnt_eff     = msg.kinematics_jnt_eff
        self.kinematics_target_pos  = msg.kinematics_target_pos
        self.kinematics_target_quat = msg.kinematics_target_quat

        self.ft_force  = msg.ft_force
        self.ft_torque = msg.ft_torque

        self.vision_pos  = msg.vision_pos
        self.vision_quat = msg.vision_quat

        self.pps_skin_left  = msg.pps_skin_left
        self.pps_skin_right = msg.pps_skin_right
        

    def extractLocalFeature(self):

        dataSample = []

        # Unimoda feature - Audio --------------------------------------------        
        if 'unimodal_audioPower' in self.feature_list:

            ang_max, ang_min = getAngularSpatialRF(self.kinematics_ee_pos, self.rf_radius)
            audio_power_min  = self.param_dict['unimodal_audioPower_power_min']          

            if self.audio_azimuth > ang_min and self.audio_azimuth < ang_max:
                unimodal_audioPower = self.audio_power
            else:
                unimodal_audioPower = audio_power_min
            
            dataSample.append(unimodal_audioPower)

        # Unimodal feature - Kinematics --------------------------------------
        if 'unimodal_kinVel' in self.feature_list:
            print "not implemented"

        # Unimodal feature - Force -------------------------------------------
        if 'unimodal_ftForce' in self.feature_list:
            ftForce_pca = self.param_dict['unimodal_ftForce_pca']            
            unimodal_ftForce = ftForce_pca.transform(self.ft_force)

            if len(np.array(unimodal_ftForce).flatten()) > 1:
                dataSample += list(np.squeeze(unimodal_ftForce))
            else: 
                dataSample.append( np.squeeze(unimodal_ftForce) )


        # Crossmodal feature - relative dist --------------------------
        if 'crossmodal_targetRelativeDist' in feature_list:

            crossmodal_targetRelativeDist = np.linalg.norm(np.array(self.kinematics_target_pos) - \
                                                           np.array(self.kinematics_ee_pos))

            dataSample.append( crossmodal_targetRelativeDist )

        # Crossmodal feature - relative angle --------------------------
        if 'crossmodal_targetRelativeAng' in feature_list:                
            
            diff_ang = qt.quat_angle(self.kinematics_ee_quat, self.kinematics_target_quat)
            crossmodal_targetRelativeAng = abs(diff_ang)

            dataSample.append( crossmodal_targetRelativeAng )

        # Scaling ------------------------------------------------------------
        scaled_features = []
        for i, feature in enumerate(dataSample):
            scaled_features.append( ( feature - self.param_dict['feature_min'][i] )\
                                    /( self.param_dict['feature_max'][i] - self.param_dict['feature_min'][i]) )
        
        
        return scaled_features
            
            
    def reset(self):
        '''
        Reset parameters
        '''
        self.dataList = []
        

    def run(self):
        '''
        Run detector
        '''            
        rospy.loginfo("Start to run anomaly detection: "+self.task_name)
        self.count = 0
        self.enable_detector = True
        
        rate = rospy.Rate(20) # 25Hz, nominally.
        while not rospy.is_shutdown():

            if self.enable_detector is False: continue
            self.count += 1

            # extract feature
            self.dataList.append( self.extractLocalFeature() ) 

            if len(np.shape(self.dataList)) == 1: continue
            if np.shape(self.dataList)[0] < 100: continue

            # visualization
            if self.online_raw_viz: 
                for i in xrange(len(self.dataList[-1])):
                    self.plot_data.setdefault(i, [])
                    self.plot_data[i].append(self.dataList[-1][i])
                    
                if self.count % 10 == 0:
                    
                    for i in xrange(len(self.plot_data.keys())):
                        if i == 0: plt.ioff()
                        else: plt.ion()
                        self.fig.add_subplot( len(self.plot_data.keys())*100+10+(i+1) )
                        plt.plot(self.plot_data[i])
                    plt.draw()
            
            # Run anomaly checker
            ## anomaly, error = self.ml.anomaly_check(np.array(self.dataList).T, self.threshold)
            ## print "anomaly check : ", anomaly, " " , error
            
            # anomaly decision
            ## if np.isnan(error): print "anomaly check returned nan"
            ## elif anomaly: 
            ##     self.action_interruption_pub.publish(self.task_name+'_anomaly')
            ##     self.soundHandle.play(2)
            ##     self.enable_detector = False
            ##     self.reset()                           
                
            
            rate.sleep()

        

if __name__ == '__main__':
        

    task_name    = 'scooping'
    feature_list = ['unimodal_ftForce', 'crossmodal_targetRelativeDist', \
                    'crossmodal_targetRelativeAng']
    save_data_path    = '/home/dpark/hrl_file_server/dpark_data/anomaly/RSS2016'
    training_data_pkl = task_name+'_dataSet_0'

    ad = anomaly_detector(task_name, feature_list, save_data_path, training_data_pkl, online_raw_viz=True)
    ad.run()
    
