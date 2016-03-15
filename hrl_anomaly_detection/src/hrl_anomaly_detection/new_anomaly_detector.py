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

# system
import rospy
import roslib
import random
import os, sys

# util
import numpy as np
import hrl_lib.quaternion as qt
from hrl_anomaly_detection import util
from hrl_anomaly_detection import data_manager as dm
from sound_play.libsoundplay import SoundClient

# learning
from hrl_anomaly_detection.hmm import learning_hmm

# Classifier
from hrl_anomaly_detection.classifiers import classifier as cb
from sklearn import preprocessing
from joblib import Parallel, delayed

# msg
from hrl_srvs.srv import Bool_None, Bool_NoneResponse
from hrl_anomaly_detection.msg import MultiModality
from std_msgs.msg import String

class anomaly_detector:
    def __init__(self, subject_names, task_name, check_method, raw_data_path, save_data_path, training_data_pkl):
        rospy.init_node(task_name)
        rospy.loginfo('Initializing anomaly detector')

        self.subject_names     = subject_names
        self.task_name         = task_name
        self.check_method      = check_method
        self.raw_data_path     = raw_data_path
        self.save_data_path    = save_data_path
        self.training_data_pkl = os.path.join(save_data_path, training_data_pkl)

        self.enable_detector = False
        self.soundHandle = SoundClient()
        self.dataList = []

        # Params
        self.classifier_method = None
        self.rf_radius = None
        self.rf_center = None
        self.downSampleSize = None
        self.feature_list = None
        self.nState = None
        self.cov_mult = None
        self.scale = None
        self.threshold = None
        self.nEmissionDim = None
        self.ml = None
        self.classifier = None

        # Comms
        self.action_interruption_pub = None
        self.detection_service = None

        self.initParams()
        self.initComms()
        self.initDetector()
        self.reset()

    '''
    Load feature list
    '''
    def initParams(self):
        self.rf_radius = rospy.get_param('hrl_manipulation_task/'+self.task_name+'/rf_radius')
        self.rf_center = rospy.get_param('hrl_manipulation_task/'+self.task_name+'/rf_center')
        self.downSampleSize = rospy.get_param('hrl_manipulation_task/'+self.task_name+'/downSampleSize')
        self.feature_list = rospy.get_param('hrl_manipulation_task/'+self.task_name+'/feature_list')

        # Generative modeling
        self.nState    = rospy.get_param('hrl_anomaly_detection/'+self.task_name+'/states')
        self.cov_mult  = rospy.get_param('hrl_anomaly_detection/'+self.task_name+'/cov_mult')
        self.scale     = rospy.get_param('hrl_anomaly_detection/'+self.task_name+'/scale')

        # Discriminative classifier
        self.threshold = -200.0

    '''
    Subscribe raw data
    '''
    def initComms(self):
        # Publisher
        self.action_interruption_pub = rospy.Publisher('InterruptAction', String)

        # Subscriber
        rospy.Subscriber('/hrl_manipulation_task/raw_data', MultiModality, self.rawDataCallback)

        # Service
        self.detection_service = rospy.Service('anomaly_detector_enable/' + self.task_name, Bool_None, self.enablerCallback)

    def initDetector(self):
        _, successData, failureData, _ = dm.getDataSet(self.subject_names, self.task_name,
                                                         self.raw_data_path, self.save_data_path,
                                                         self.rf_center, self.rf_radius,
                                                         downSampleSize=self.downSampleSize,
                                                         scale=self.scale,
                                                         feature_list=self.feature_list)
        # index selection
        success_idx  = range(len(successData[0]))
        failure_idx  = range(len(failureData[0]))

        nTrain = int( 0.5*len(success_idx) )
        train_idx = random.sample(success_idx, nTrain)
        success_test_idx = [x for x in success_idx if not x in train_idx]
        failure_test_idx = failure_idx

        # data structure: dim x sample x sequence
        trainingData = successData[:, train_idx, :]
        normalClassifierData = successData[:, success_test_idx, :]
        abnormalClassifierData = failureData[:, failure_test_idx, :]

        print "======================================"
        print "Training data: ", np.shape(trainingData)
        print "Normal classifier training data: ", np.shape(normalClassifierData)
        print "Abnormal classifier training data: ", np.shape(abnormalClassifierData)
        print "======================================"

        # training hmm
        self.nEmissionDim = len(trainingData)
        detection_param_pkl = os.path.join(self.save_data_path, 'hmm_'+self.task_name+'.pkl')
        self.ml = learning_hmm.learning_hmm(self.nState, self.nEmissionDim, verbose=False)
        ret = self.ml.fit(trainingData, cov_mult=[self.cov_mult]*self.nEmissionDim**2,
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
        for i in xrange(self.nEmissionDim):
            temp = np.vstack([normalClassifierData[i], abnormalClassifierData[i]])
            testDataX.append( temp )

        testDataY = np.hstack([ -np.ones(len(normalClassifierData[0])),
                                np.ones(len(abnormalClassifierData[0])) ])

        startIdx = 4
        r = Parallel(n_jobs=-1)(delayed(learning_hmm.computeLikelihoods)(i, self.ml.A, self.ml.B, self.ml.pi, self.ml.F,
                                                                [ testDataX[j][i] for j in xrange(self.nEmissionDim) ],
                                                                self.ml.nEmissionDim, self.ml.nState,
                                                                startIdx=startIdx, bPosterior=True)
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


        # flatten the data
        X_train_org = []
        Y_train_org = []
        idx_train_org = []
        for i in xrange(len(ll_classifier_train_X)):
            for j in xrange(len(ll_classifier_train_X[i])):
                X_train_org.append(ll_classifier_train_X[i][j])
                Y_train_org.append(ll_classifier_train_Y[i][j])
                idx_train_org.append(ll_classifier_train_idx[i][j])


        # data preparation
        scaler = preprocessing.StandardScaler()
        if 'svm' in self.classifier_method:
            X_scaled = scaler.fit_transform(X_train_org)
        else:
            X_scaled = X_train_org
        print self.classifier_method, " : Before classification : ", np.shape(X_scaled), np.shape(Y_train_org)

        # Fit Classifier
        self.classifier = cb.classifier(method=self.classifier_method, nPosteriors=self.nState, nLength=len(trainingData[0,0]))

        self.classifier.fit(X_scaled, Y_train_org, idx_train_org)


    def enablerCallback(self, msg):

        if msg.data is True:
            self.enable_detector = True
        else:
            # Reset detector
            self.enable_detector = False
            self.reset()

        return Bool_NoneResponse()

    def rawDataCallback(self, msg):
        # self.d['audioPowerList'] = [msg.audio_power]
        # self.d['ppsLeftList'] = [msg.pps_skin_left]
        # self.d['ppsRightList'] = [msg.pps_skin_right]
        # self.d['timesList'] = [0]



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

        self.feature_list = ['unimodal_audioPower',
                             # 'unimodal_audioWristRMS',
                             'unimodal_kinVel',
                             'unimodal_ftForce',
                             'unimodal_ppsForce',
                             # 'unimodal_visionChange',
                             'unimodal_fabricForce',
                             'crossmodal_targetEEDist',
                             'crossmodal_targetEEAng',
                             'crossmodal_artagEEDist']
                             # 'crossmodal_artagEEAng']





        dataSample = []

        # Unimoda feature - Audio --------------------------------------------
        if 'unimodal_audioPower' in self.feature_list:

            ang_max, ang_min = util.getAngularSpatialRF(self.kinematics_ee_pos, self.rf_radius)
            audio_power_min  = self.param_dict['unimodal_audioPower_power_min']

            if ang_min < self.audio_azimuth < ang_max:
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
        if 'crossmodal_targetEEDist' in self.feature_list:

            crossmodal_targetEEDist = np.linalg.norm(np.array(self.kinematics_target_pos) - \
                                                           np.array(self.kinematics_ee_pos))

            dataSample.append( crossmodal_targetEEDist )

        # Crossmodal feature - relative angle --------------------------
        if 'crossmodal_targetEEAng' in self.feature_list:

            diff_ang = qt.quat_angle(self.kinematics_ee_quat, self.kinematics_target_quat)
            crossmodal_targetEEAng = abs(diff_ang)

            dataSample.append( crossmodal_targetEEAng )

        # Scaling ------------------------------------------------------------
        scaled_features = util.scaleData(dataSample, scale=self.scale)

        return scaled_features

    '''
    Reset parameters
    '''
    def reset(self):
        self.dataList = []

    '''
    Run detector
    '''
    def run(self):
        rospy.loginfo("Start to run anomaly detection: " + self.task_name)

        rate = rospy.Rate(20) # 25Hz, nominally.
        while not rospy.is_shutdown():
            if self.enable_detector is False: continue

            # extract feature
            self.dataList.append( self.extractLocalFeature() )

            if len(np.shape(self.dataList)) == 1: continue
            if np.shape(self.dataList)[0] < 10: continue


            l_logp, l_post = self.ml.loglikelihoods(self.dataList, bPosterior=True, startIdx=4)

            print 'Shape of l_logp:', np.shape(l_logp), 'l_post:', np.shape(l_post)

            ll_classifier_test_X = []
            for i in xrange(len(l_logp)):
                ll_classifier_test_X.append( [l_logp[i]] + l_post[i].tolist() )

            scaler = preprocessing.StandardScaler()

            for i in xrange(len(ll_classifier_test_X)):
                if 'svm' in self.classifier_method:
                    X = scaler.transform([ll_classifier_test_X[i]])
                elif self.classifier_method == 'progress_time_cluster' or self.classifier_method == 'fixed':
                    X = ll_classifier_test_X[i]
                else:
                    print 'Invalid classifier method. Exiting.'
                    exit()

                est_y = self.classifier.predict(X)
                if type(est_y) == list:
                    est_y = est_y[0]

                print 'Estimated classification', est_y

                if est_y > 0.0:
                    print '-'*15, 'Anomaly has occured!', '-'*15
                    self.action_interruption_pub.publish(self.task_name+'_anomaly')
                    self.soundHandle.play(2)
                    self.enable_detector = False
                    self.reset()

            rate.sleep()

