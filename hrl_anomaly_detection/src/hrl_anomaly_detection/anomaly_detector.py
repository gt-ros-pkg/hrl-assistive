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
import os, sys, copy
import threading

# util
import numpy as np
import hrl_lib.util as ut
from hrl_anomaly_detection.util import *
from hrl_anomaly_detection import data_manager as dm
from sound_play.libsoundplay import SoundClient

# learning
from sklearn.decomposition import PCA
from hrl_multimodal_anomaly_detection.hmm import learning_hmm_multi_n as hmm

# msg
from hrl_srvs.srv import Bool_None, Bool_NoneResponse
from hrl_anomaly_detection.msg import MultiModality
from std_msgs.msg import String

# viz
import matplotlib
import matplotlib.pyplot as plt
## matplotlib.interactive(True)
## matplotlib.use('TkAgg')
## import pylab

class anomaly_detector:

    def __init__(self, subject_names, task_name, check_method, save_data_path, training_data_pkl, \
                 verbose=False, \
                 online_raw_viz=False):
        rospy.init_node(task_name)
        rospy.loginfo('Initializing anomaly detector')

        self.subject_names     = subject_names
        self.task_name         = task_name
        self.check_method      = check_method
        self.save_data_path    = save_data_path
        self.training_data_pkl = os.path.join(save_data_path, training_data_pkl)
        self.online_raw_viz    = online_raw_viz
        self.count = 0

        self.enable_detector = False
        self.soundHandle = SoundClient()

        # visualization
        if self.online_raw_viz: 
            ## plt.ion()
            self.fig = plt.figure(1)
            ## pylab.hold(False)
            self.plot_data = {}
            self.plot_len = 22000
        
        self.initParams()
        self.initComms()
        self.initDetector()
        self.reset()

    def initParams(self):
        '''
        Load feature list
        '''
        self.rf_radius = rospy.get_param('hrl_manipulation_task/'+self.task_name+'/rf_radius')
        self.rf_center = rospy.get_param('hrl_manipulation_task/'+self.task_name+'/rf_center')
        self.downSampleSize = rospy.get_param('hrl_manipulation_task/'+self.task_name+'/downSampleSize')
        self.feature_list = rospy.get_param('hrl_manipulation_task/'+self.task_name+'/feature_list')

        # Generative modeling
        self.nState    = rospy.get_param('hrl_anomaly_detection/'+self.task_name+'/states')
        self.cov_mult  = rospy.get_param('hrl_anomaly_detection/'+self.task_name+'/cov_mult')

        # Discriminative classifier
        self.cov_mult  = rospy.get_param('hrl_anomaly_detection/'+self.task_name+'/cov_mult')
        self.threshold = -200.0
    
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

        modeling_pkl = os.path.join(processed_data_path, 'hmm_'+task_name+'_exp.pkl')
        if os.path.isfile(modeling_pkl) is False:

            _, success_data, failure_data, _ = dm.feature_extraction(self.subject_names, self.task_name, \
                                                                     self.save_data_path, \
                                                                     self.save_data_path, self.rf_center, \
                                                                     self.rf_radius, \
                                                                     downSampleSize=self.downSampleSize, \
                                                                     feature_list=self.feature_list)

            # index selection
            success_idx  = range(len(successData[0]))
            failure_idx  = range(len(failureData[0]))

            nTrain       = int( 0.7*len(success_idx) )    
            train_idx    = random.sample(success_idx, nTrain)
            success_test_idx = [x for x in success_idx if not x in train_idx]
            failure_test_idx = failure_idx

            # data structure: dim x sample x sequence
            trainingData     = successData[:, train_idx, :]
            normalTestData   = successData[:, success_test_idx, :]
            abnormalTestData = failureData[:, failure_test_idx, :]

            print "======================================"
            print "Training data: ", np.shape(trainingData)
            print "Normal test data: ", np.shape(normalTestData)
            print "Abnormal test data: ", np.shape(abnormalTestData)
            print "======================================"

            # training hmm
            self.nEmissionDim = len(trainingData)
            detection_param_pkl = os.path.join(self.save_data_path, 'hmm_'+self.task_name+'.pkl')        
            self.ml = hmm.learning_hmm_multi_n(self.nState, self.nEmissionDim, verbose=False)        
            ret = self.ml.fit(self.success_data, cov_mult=[self.cov_mult]*self.nEmissionDim**2, \
                              ml_pkl=detection_param_pkl, use_pkl=True)

            if ret == 'Failure': 
                print "-------------------------"
                print "HMM returned failure!!   "
                print "-------------------------"
                sys.exit()

            #-----------------------------------------------------------------------------------------
            # Classifier training data
            #-----------------------------------------------------------------------------------------
            testDataX = []
            testDataY = []
            for i in xrange(nEmissionDim):
                temp = np.vstack([normalClassifierData[i], abnormalClassifierData[i]])
                testDataX.append( temp )

            testDataY = np.hstack([ -np.ones(len(normalClassifierData[0])), \
                                    np.ones(len(abnormalClassifierData[0])) ])

            r = Parallel(n_jobs=-1)(delayed(hmm.computeLikelihoods)(i, ml.A, ml.B, ml.pi, ml.F, \
                                                                    [ testDataX[j][i] for j in xrange(nEmissionDim) ], \
                                                                    ml.nEmissionDim, ml.scale, ml.nState,\
                                                                    startIdx=startIdx, \
                                                                    bPosterior=True)
                                                                    for i in xrange(len(testDataX[0])))
            _, ll_classifier_train_idx, ll_logp, ll_post = zip(*r)

            ll_classifier_train_X = []
            ll_classifier_train_Y = []
            for i in xrange(len(ll_logp)):
                l_X = []
                l_Y = []
                for j in xrange(len(ll_logp[i])):        
                    l_X.append( [ll_logp[i][j]] + ll_post[i][j].tolist() )

                    if testDataY[i] > 0.0: l_Y.append(1)
                    else: l_Y.append(-1)

                ll_classifier_train_X.append(l_X)
                ll_classifier_train_Y.append(l_Y)
                

        else:
            

                

        # training classifier
        dtc = cb.classifier( ml, method=method, nPosteriors=nState, nLength=len(trainingData[0,0]) )        



            
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

        self.audio_wrist_rms   = msg.audio_wrist_rms
        self.audio_wrist_mfcc  = msg.audio_wrist_mfcc

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
        
        self.fabric_skin_centers_x = msg.fabric_skin_centers_x
        self.fabric_skin_centers_y = msg.fabric_skin_centers_y
        self.fabric_skin_centers_z = msg.fabric_skin_centers_z
        self.fabric_skin_normals_x = msg.fabric_skin_normals_x
        self.fabric_skin_normals_y = msg.fabric_skin_normals_y
        self.fabric_skin_normals_z = msg.fabric_skin_normals_z
        self.fabric_skin_values_x  = msg.fabric_skin_values_x
        self.fabric_skin_values_y  = msg.fabric_skin_values_y
        self.fabric_skin_values_z  = msg.fabric_skin_values_z


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
        if 'crossmodal_targetEEDist' in feature_list:

            crossmodal_targetEEDist = np.linalg.norm(np.array(self.kinematics_target_pos) - \
                                                           np.array(self.kinematics_ee_pos))

            dataSample.append( crossmodal_targetEEDist )

        # Crossmodal feature - relative angle --------------------------
        if 'crossmodal_targetEEAng' in feature_list:                
            
            diff_ang = qt.quat_angle(self.kinematics_ee_quat, self.kinematics_target_quat)
            crossmodal_targetEEAng = abs(diff_ang)

            dataSample.append( crossmodal_targetEEAng )

        # Scaling ------------------------------------------------------------
        scaled_features = (np.array(dataSample) - np.array(self.param_dict['feature_min']) )\
          /( np.array(self.param_dict['feature_max']) - np.array(self.param_dict['feature_min']) ) 

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
        ## self.enable_detector = True
        
        rate = rospy.Rate(20) # 25Hz, nominally.
        while not rospy.is_shutdown():

            if self.enable_detector is False: continue
            self.count += 1

            # extract feature
            self.dataList.append( self.extractLocalFeature() ) 

            if len(np.shape(self.dataList)) == 1: continue
            if np.shape(self.dataList)[0] < 10: continue

            # visualization
            if self.online_raw_viz: 
                for i in xrange(len(self.dataList[-1])):
                    self.plot_data.setdefault(i, [])
                    self.plot_data[i].append(self.dataList[-1][i])
                    
                if self.count % 50 == 0:

                    for i in xrange(len(self.dataList[-1])):
                        ax = self.fig.add_subplot( len(self.plot_data.keys())*100+10+(i+1) )
                        ax.plot(self.plot_data[i], 'r')
                        
                if self.count > 280: 
                    for i, feature_name in enumerate(feature_list):
                        ax = self.fig.add_subplot( len(self.plot_data.keys())*100+10+(i+1) )
                        ax.plot(np.array(self.success_data[i]).T, 'b')                        
                    plt.show()
            
            # Run anomaly checker
            anomaly, error = self.ml.anomaly_check(np.array(self.dataList).T, self.threshold)
            print "anomaly check : ", anomaly, " " , error, " dat shape: ", np.array(self.dataList).T.shape

            # anomaly decision
            ## if np.isnan(error): continue #print "anomaly check returned nan"
            if anomaly or np.isnan(error) or np.isinf(error): 
                print "anoooooooooooooooooooooomaly"
                self.action_interruption_pub.publish(self.task_name+'_anomaly')
                self.soundHandle.play(2)
                self.enable_detector = False
                self.reset()                           
                
            
            rate.sleep()

        

if __name__ == '__main__':
        
    subject_names     = ['gatsbii']
    task_name         = 'scooping'
    ## feature_list      = ['unimodal_ftForce', 'crossmodal_targetEEDist']
    save_data_path    = '/home/dpark/hrl_file_server/dpark_data/anomaly/RSS2016/'+task_name+'_data'    
    training_data_pkl = task_name+'_dataSet'
    check_method      = 'progress_time_cluster' # cssvm

    ad = anomaly_detector(subject_names, task_name, check_method, save_data_path, training_data_pkl, \
                          online_raw_viz=False)
    ad.run()
    
