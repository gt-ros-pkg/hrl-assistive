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
import random
import os, sys, threading
from joblib import Parallel, delayed

# util
import numpy as np
import hrl_lib.quaternion as qt
from hrl_anomaly_detection import util
from hrl_anomaly_detection import data_manager as dm
from sound_play.libsoundplay import SoundClient
import hrl_lib.util as ut

# learning
from hrl_anomaly_detection.hmm import learning_hmm
from sklearn import preprocessing

# Classifier
from hrl_anomaly_detection.classifiers import classifier as cb

# msg
from hrl_anomaly_detection.msg import MultiModality
from std_msgs.msg import String, Float64
from hrl_srvs.srv import Bool_None, Bool_NoneResponse, StringArray_None


QUEUE_SIZE = 10

class anomaly_detector:
    def __init__(self, subject_names, task_name, check_method, raw_data_path, save_data_path,\
                 param_dict):
        rospy.loginfo('Initializing anomaly detector')

        self.subject_names     = subject_names
        self.task_name         = task_name
        self.raw_data_path     = raw_data_path
        self.save_data_path    = save_data_path

        self.enable_detector = False
        self.soundHandle = SoundClient()
        self.dataList = []

        # Params
        self.param_dict = param_dict        
        self.classifier_method = check_method
        
        self.nEmissionDim = None
        self.ml = None
        self.classifier = None

        # Comms
        self.lock = threading.Lock()        

        self.initParams()
        self.initComms()
        self.initDetector()
        self.reset()

    '''
    Load feature list
    '''
    def initParams(self):

        if False:
            # data
            self.rf_radius = rospy.get_param('/hrl_manipulation_task/'+self.task_name+'/rf_radius')
            self.rf_center = rospy.get_param('/hrl_manipulation_task/'+self.task_name+'/rf_center')
            self.downSampleSize = rospy.get_param('/hrl_manipulation_task/'+self.task_name+'/downSampleSize')
            self.handFeatures = rospy.get_param('/hrl_manipulation_task/'+self.task_name+'/feature_list')
            self.nNormalFold   = 2
            self.nAbnormalFold = 2

            # Generative modeling
            self.nState = rospy.get_param('/hrl_anomaly_detection/'+self.task_name+'/states')
            self.cov    = rospy.get_param('/hrl_anomaly_detection/'+self.task_name+'/cov_mult')
            self.scale  = rospy.get_param('/hrl_anomaly_detection/'+self.task_name+'/scale')
            self.add_logp_d = True

            self.SVM_dict = None
        else:
            self.rf_radius = self.param_dict['data_param']['local_range']
            self.rf_center = self.param_dict['data_param']['rf_center']
            self.downSampleSize = self.param_dict['data_param']['downSampleSize']
            self.handFeatures = self.param_dict['data_param']['handFeatures']
            self.cut_data     = self.param_dict['data_param']['cut_data']
            self.nNormalFold   = self.param_dict['data_param']['nNormalFold']
            self.nAbnormalFold = self.param_dict['data_param']['nAbnormalFold']

            self.nState = self.param_dict['HMM']['nState']
            self.cov    = self.param_dict['HMM']['cov']
            self.scale  = self.param_dict['HMM']['scale']
            self.add_logp_d = self.param_dict['HMM'].get('add_logp_d', False)

            self.SVM_dict  = self.param_dict['SVM']

            if 'svm' in self.classifier_method or 'sgd' in self.classifier_method:
                self.w_max = self.param_dict['ROC'][self.classifier_method+'_param_range'][-1]
                self.w_min = self.param_dict['ROC'][self.classifier_method+'_param_range'][0]
            elif self.classifier_method == 'progress_time_cluster':                    
                self.w_max = self.param_dict['ROC']['progress_param_range'][-1]
                self.w_min = self.param_dict['ROC']['progress_param_range'][0]
            else:
                print "sensitivity info is not available"
                sys.exit()


    '''
    Subscribe raw data
    '''
    def initComms(self):
        # Publisher
        self.action_interruption_pub = rospy.Publisher('/hrl_manipulation_task/InterruptAction', String,
                                                       queue_size=QUEUE_SIZE)
        self.task_interruption_pub   = rospy.Publisher("/manipulation_task/emergency", String,
                                                       queue_size=QUEUE_SIZE)
        self.sensitivity_pub         = rospy.Publisher("/manipulation_task/ad_sensitivity_state", \
                                                       Float64, queue_size=QUEUE_SIZE, latch=True)

        # Subscriber
        rospy.Subscriber('/hrl_manipulation_task/raw_data', MultiModality, self.rawDataCallback)
        rospy.Subscriber('/manipulation_task/status', String, self.statusCallback)
        rospy.Subscriber('/manipulation_task/user_feedback', String, self.userfbCallback)
        rospy.Subscriber('/manipulation_task/ad_sensitivity_request', Float64, self.sensitivityCallback)

        # Service
        self.detection_service = rospy.Service('anomaly_detector_enable', Bool_None, self.enablerCallback)
        self.update_service    = rospy.Service('anomaly_detector_update', StringArray_None, self.updateCallback)

    def initDetector(self):
        
        train_pkl = os.path.join(save_data_path, self.task_name + '_demo.pkl')
        startIdx  = 4
        
        if os.path.isfile(train_pkl):
            d = ut.load_pickle(train_pkl)
            # HMM
            self.nEmissionDim = d['nEmissionDim']
            self.A            = d['A']
            self.B            = d['B']
            self.pi           = d['pi']
            self.ml = learning_hmm.learning_hmm(self.nState, self.nEmissionDim, verbose=False)
            self.ml.set_hmm_object(self.A, self.B, self.pi)
            
            X_test_org   = d['X_test_org'] 
            Y_test_org   = d['Y_test_org']
            idx_test_org = d['idx_test_org']
            nLength      = d['nLength']
            self.handFeatureParams = d['param_dict']

        else:
            dd = dm.getDataSet(self.subject_names, self.task_name, self.raw_data_path, \
                               self.save_data_path, self.rf_center, \
                               self.rf_radius,\
                               downSampleSize=self.downSampleSize, \
                               scale=1.0,\
                               ae_data=False,\
                               handFeatures=self.handFeatures, \
                               cut_data=self.cut_data,\
                               data_renew=False)

            self.handFeatureParams = dd['param_dict']

            # Task-oriented hand-crafted features        
            kFold_list = dm.kFold_data_index2(len(dd['successData'][0]), len(dd['failureData'][0]), \
                                                  self.nNormalFold, self.nAbnormalFold )
            # Select the first fold as the training data (need to fix?)
            (normalTrainIdx, abnormalTrainIdx, normalTestIdx, abnormalTestIdx) = kFold_list[0]


            # dim x sample x length # TODO: what is the best selection?
            if self.check_method.find('svm')>=0:
                normalTrainData   = dd['successData'][:, normalTrainIdx, :]   * self.scale
                abnormalTrainData = dd['failureData'][:, abnormalTrainIdx, :] * self.scale # will not be used...?
                normalTestData    = dd['successData'][:, normalTestIdx, :]    * self.scale
                abnormalTestData  = dd['failureData'][:, abnormalTestIdx, :]  * self.scale
            elif self.check_method.find('sgd')>=0:
                normalTrainData   = dd['successData'][:, normalTrainIdx, :] * self.scale
                normalTestData    = dd['successData'][:, normalTestIdx, :]    * self.scale
                abnormalTestData  = dd['failureData'][:, abnormalTestIdx, :]  * self.scale

                

            # training hmm
            self.nEmissionDim   = len(normalTrainData)
            detection_param_pkl = os.path.join(self.save_data_path, 'hmm_'+self.task_name+'_demo.pkl')
            self.ml = learning_hmm.learning_hmm(self.nState, self.nEmissionDim, verbose=False)
            if self.param_dict['data_param']['handFeatures_noise']:
                ret = self.ml.fit(normalTrainData+
                                  np.random.normal(0.0, 0.03, np.shape(normalTrainData) )*self.scale, \
                                  cov_mult=[self.cov]*(self.nEmissionDim**2),
                                  ml_pkl=detection_param_pkl, use_pkl=True)
            else:
                ret = self.ml.fit(normalTrainData, cov_mult=[self.cov]*(self.nEmissionDim**2),
                                  ml_pkl=detection_param_pkl, use_pkl=True)

            if ret == 'Failure':
                print "-------------------------"
                print "HMM returned failure!!   "
                print "-------------------------"
                sys.exit()

            #-----------------------------------------------------------------------------------------
            # Classifier test data
            #-----------------------------------------------------------------------------------------
            testDataX = []
            testDataY = []
            for i in xrange(self.nEmissionDim):
                temp = np.vstack([normalTestData[i], abnormalTestData[i]])
                testDataX.append( temp )

            testDataY = np.hstack([ -np.ones(len(normalTestData[0])), \
                                    np.ones(len(abnormalTestData[0])) ])

            r = Parallel(n_jobs=-1)(delayed(learning_hmm.computeLikelihoods)(i, self.ml.A, self.ml.B, \
                                                                             self.ml.pi, self.ml.F,
                                                                             [ testDataX[j][i] for j in xrange(self.nEmissionDim) ],
                                                                    self.ml.nEmissionDim, self.ml.nState,
                                                                    startIdx=startIdx, bPosterior=True)
                                                                    for i in xrange(len(testDataX[0])))
            _, ll_classifier_test_idx, ll_logp, ll_post = zip(*r)

            # nSample x nLength
            ll_classifier_test_X, ll_classifier_test_Y = \
              learning_hmm.getHMMinducedFeatures(ll_logp, ll_post, testDataY, c=1.0, add_delta_logp=add_logp_d)

            # flatten the data
            X_test_org = []
            Y_test_org = []
            idx_test_org = []
            for i in xrange(len(ll_classifier_test_X)):
                for j in xrange(len(ll_classifier_test_X[i])):
                    X_test_org.append(ll_classifier_test_X[i][j])
                    Y_test_org.append(ll_classifier_test_Y[i][j])
                    idx_test_org.append(ll_classifier_test_idx[i][j])

            d = {}
            d['A']  = self.ml.A
            d['B']  = self.ml.B
            d['pi'] = self.ml.pi
            d['nEmissionDim'] = self.nEmissionDim
            d['X_test_org']   = X_test_org
            d['Y_test_org']   = Y_test_org
            d['idx_test_org'] = idx_test_org
            d['nLength']      = nLength = len(normalTrainData[0][0])
            d['param_dict']   = self.handFeatureParams
            ut.save_pickle(d, train_pkl)

        # data preparation
        self.scaler       = preprocessing.StandardScaler()
        self.Y_test_org   = Y_test_org
        self.idx_test_org = idx_test_org
        if 'svm' in self.classifier_method:
            self.X_scaled = self.scaler.fit_transform(X_test_org)
        else:
            self.X_scaled = X_test_org
            
        print self.classifier_method, " : Before classification : ", \
          np.shape(self.X_scaled), np.shape(self.Y_test_org)
            
        # Fit Classifier
        self.classifier = cb.classifier(method=self.classifier_method, nPosteriors=self.nState, \
                                        nLength=nLength - startIdx)
        self.classifier.set_params(**self.SVM_dict)
        self.classifier.fit(self.X_scaled, self.Y_test_org, self.idx_test_org, parallel=False)
        print "Finished to train SVM"

        if 'svm' in self.classifier_method or 'sgd' in self.classifier_method:
            # may be we have to manually set or adjust it through the GUI
            sensitivity = (self.classifier.class_weight-self.w_min)/(self.w_max-self.w_min)
        elif self.classifier_method == 'progress_time_cluster':                    
            sensitivity = (self.classifier.ths_mult-self.w_min)/(self.w_max-self.w_min)
        self.sensitivity_pub.publish(sensitivity)                                   

        print "Current sensitivity is ", sensitivity
        
        return

        
    def enablerCallback(self, msg):

        if msg.data is True:
            print "anomaly detector enabled"
            self.enable_detector = True
        else:
            print "anomaly detector disabled"
            # Reset detector
            self.enable_detector = False
            self.reset()

        return Bool_NoneResponse()


    def updateCallback(self, msg):
        fileNames = msg.data

        if len(fileNames) == 0 or os.path.isfile(fileName) is False:
            print "Warning>> there is no recorded file"
            return StringArray_NoneResponse()
              
        print "Start to update detector using ", fileName

        # Get label
        for f in fileNames:
            if 'success' in f: self.Y_test_org.append(0)
            else: Y_test_org.append(1)

        # Preprocessing
        trainData,_ = dm.getDataList(fileNames, self.rf_center, self.local_range,\
                                     self.handFeatureParams,\
                                     downSampleSize = self.downSampleSize, \
                                     cut_data       = self.cut_data,\
                                     handFeatures   = self.handFeatures)
        print "Preprocessing: ", np.shape(trainData), np.shape(Y_test_org)

        ## HMM
        ll_logp, ll_post = self.ml.loglikelihoods(trainData, bPosterior=True)
        X, Y = getHMMinducedFeatures(ll_logp, ll_post, Y_test_org)
        print "Features: ", np.shape(X), np.shape(Y)

        ## Remove unseparable region and scaling it
        if 'svm' in self.classifier_method or 'sgd' in self.classifier_method:
            X_train_org, Y_train_org, _ = dm.flattenSample(X, Y, remove_fp=True)
            self.X_scaled = np.vstack([ self.X_scaled, self.scaler.transform(X_train_org) ])
        else:
            self.X_scaled = np.vstack([ self.X_scaled, X ])

        # Run SGD? or SVM?
        if self.classifier_method.find('svm'):
            self.classifier.fit(self.X_scaled, self.Y_test_org)
        else:
            print "Not available update method"
            
        return StringArray_NoneResponse()
        

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

        self.vision_artag_pos  = msg.vision_artag_pos
        self.vision_artag_quat = msg.vision_artag_quat

        self.vision_change_centers_x = msg.vision_change_centers_x
        self.vision_change_centers_y = msg.vision_change_centers_y
        self.vision_change_centers_z = msg.vision_change_centers_z

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

        # If detector is disbled, detector does not fill out the dataList.
        if self.enable_detector is False: return
        
        self.lock.acquire()
        newData = self.extractLocalFeature()

        # get offset
        startOffsetSize = 4
        if len(self.dataList[0][0]) == startOffsetSize:
            self.offsetData = np.mean(self.dataList, axis=2)/self.scale
        elif len(self.dataList[0][0]) < startOffsetSize:
            self.offsetData = np.zeros(np.shape(newData))
        newData -= self.offsetData
        
        if len(self.dataList) == 0:
            self.dataList = (np.array(newData)*self.scale).tolist()
        else:                
            ## self.dataList = np.swapaxes(self.dataList, 0,1)
            for i in xrange(self.nEmissionDim):
                self.dataList[i][0] = self.dataList[i][0] + [newData[i][0][0]*self.scale]
            ## self.dataList = np.swapaxes(self.dataList, 0,1)
                       
        self.lock.release()


    def statusCallback(self, msg):
        self.cur_task = msg.data

    def userfbCallback(self, msg):
        self.user_feedback = msg.data

    def sensitivityCallback(self, msg):
        '''
        Requested value's range is 0~1.
        '''
        self.sensitivity_req = msg.data
        sensitivity = (self.classifier.ths_mult-self.w_min)/(self.w_max-self.w_min)
        print "Current sensitivity is ", sensitivity
        print "Requested sensitivity is ", self.sensitivity_req

        if 'svm' in self.classifier_method:           
            self.classifier.set_params(class_weight=self.sensitivity_req*(self.w_max-self.w_min)+self.w_min )
        elif self.classifier_method == 'progress_time_cluster':                    
            self.classifier.set_params(ths_mult=self.sensitivity_req*(self.w_max-self.w_min)+self.w_min)
            
        self.classifier.fit(self.X_scaled, self.Y_test_org, self.idx_test_org)
        print "Classifier is updated!"

        if 'svm' in self.classifier_method:
            sensitivity = (self.classifier.class_weight-self.w_min)/(self.w_max-self.w_min)
            print "Current sensitivity is ", sensitivity, self.classifier.class_weight
        elif self.classifier_method == 'progress_time_cluster':                    
            sensitivity = (self.classifier.ths_mult-self.w_min)/(self.w_max-self.w_min)
            print "Current sensitivity is ", sensitivity, self.classifier.ths_mult
        self.sensitivity_pub.publish(sensitivity)                                   

        

                    
    def extractLocalFeature(self):

        startOffsetSize = 4
        data_dict = {}
        data_dict['timesList'] = [[0.]]
            
        # Unimoda feature - Audio --------------------------------------------
        if 'unimodal_audioPower' in self.handFeatures:
            ang_max, ang_min = util.getAngularSpatialRF(self.kinematics_ee_pos, self.rf_radius)
            if ang_min < self.audio_azimuth < ang_max:
                data_dict['audioPowerList'] = [self.audio_power]
            else:
                data_dict['audioPowerList'] = [0.0]

        # Unimodal feature - AudioWrist ---------------------------------------
        if 'unimodal_audioWristRMS' in self.handFeatures:
            data_dict['audioWristRMSList'] = [self.audio_wrist_rms]
            ## self.data_dict['audioWristMFCCList'].append(audio_mfcc)

        # Unimodal feature - Kinematics --------------------------------------
        if 'unimodal_kinVel' in self.handFeatures:
            print 'unimodal_kinVel not implemented'

        # Unimodal feature - Force -------------------------------------------
        if 'unimodal_ftForce' in self.handFeatures:
            data_dict['ftForceList']  = [np.array([self.ft_force]).T]
            data_dict['ftTorqueList'] = [np.array([self.ft_torque]).T]

        # Unimodal feature - pps -------------------------------------------
        if 'unimodal_ppsForce' in self.handFeatures:
            data_dict['ppsLeftList'] = [self.pps_skin_left]
            data_dict['ppsRightList'] = [self.pps_skin_right]
            data_dict['kinTargetPosList'] = [self.kinematics_target_pos]

        # Unimodal feature - vision change ------------------------------------
        if 'unimodal_visionChange' in self.handFeatures:
            vision_centers = np.array([self.vision_change_centers_x, self.vision_change_centers_y, \
                                       self.vision_change_centers_z])
            data_dict['visionChangeMagList'] = [len(vision_centers[0])]
            print 'unimodal_visionChange may not be implemented properly'

        # Unimodal feature - fabric skin ------------------------------------
        if 'unimodal_fabricForce' in self.handFeatures:
            fabric_skin_values  = [self.fabric_skin_values_x, self.fabric_skin_values_y, \
                                   self.fabric_skin_values_z]
            if not fabric_skin_values[0]:
                data_dict['fabricMagList'] = [0]
            else:
                data_dict['fabricMagList'] = [np.sum( np.linalg.norm(np.array(fabric_skin_values), axis=0) )]

        # Crossmodal feature - relative dist --------------------------
        if 'crossmodal_targetEEDist' in self.handFeatures:
            data_dict['kinEEPosList']     = [np.array([self.kinematics_ee_pos]).T]
            data_dict['kinTargetPosList'] = [np.array([self.kinematics_target_pos]).T]

        # Crossmodal feature - relative angle --------------------------
        if 'crossmodal_targetEEAng' in self.handFeatures:
            data_dict['kinEEQuatList'] = [np.array([self.kinematics_ee_quat]).T]
            data_dict['kinTargetQuatList'] = [np.array([self.kinematics_target_quat]).T]

        # Crossmodal feature - vision relative dist with main(first) vision target----
        if 'crossmodal_artagEEDist' in self.handFeatures:
            data_dict['kinEEPosList']     = [np.array([self.kinematics_ee_pos]).T]
            data_dict['visionArtagPosList'] = [np.array([self.vision_artag_pos]).T]

        # Crossmodal feature - vision relative angle --------------------------
        if 'crossmodal_artagEEAng' in self.handFeatures:
            data_dict['kinEEQuatList'] = [np.array([self.kinematics_ee_quat]).T]
            data_dict['visionArtagPosList'] = [np.array([self.vision_artag_pos]).T]
            data_dict['visionArtagQuatList'] = [np.array([self.vision_artag_quat]).T]

        data, _ = dm.extractHandFeature(data_dict, self.handFeatures, scale=1.0, \
                                        param_dict = self.handFeatureParams)

                                        
        return data

    '''
    Reset parameters
    '''
    def reset(self):
        self.dataList = []
        self.enableDetector = False

    '''
    Run detector
    '''
    def run(self):
        rospy.loginfo("Start to run anomaly detection: " + self.task_name)
        rate = rospy.Rate(20) # 25Hz, nominally.
        while not rospy.is_shutdown():
            
            if self.enable_detector is False: 
                self.dataList = []
                continue            
            if len(self.dataList) == 0 or len(self.dataList[0][0]) < 10: continue

            #-----------------------------------------------------------------------
            self.lock.acquire()
            cur_length     = len(self.dataList[0][0])
            l_logp, l_post = self.ml.loglikelihood(self.dataList, bPosterior=True)
            self.lock.release()            
            if l_logp == None: 
                print "logp is None => anomaly"
                self.action_interruption_pub.publish(self.task_name+'_anomaly')
                self.task_interruption_pub.publish(self.task_name+'_anomaly')
                self.soundHandle.play(2)
                self.enable_detector = False
                self.reset()
                continue


            print "logp: ", l_logp, "  state: ", np.argmax(l_post[cur_length-1]), \
              " cutoff: ", self.param_dict['HMM']['nState']*0.85
            if np.argmax(l_post[cur_length-1])==0 and l_logp < 0.0: continue
            if np.argmax(l_post[cur_length-1])>self.param_dict['HMM']['nState']*0.85: continue
            ll_classifier_test_X = [l_logp] + l_post[cur_length-1].tolist() 

            if 'svm' in self.classifier_method:
                X = self.scaler.transform([ll_classifier_test_X])
            elif self.classifier_method == 'progress_time_cluster' or \
              self.classifier_method == 'fixed':
                X = ll_classifier_test_X
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
                self.task_interruption_pub.publish(self.task_name+'_anomaly')
                self.soundHandle.play(2)
                self.enable_detector = False
                self.reset()

            rate.sleep()

if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    p.add_option('--task', action='store', dest='task', type='string', default='scooping',
                 help='type the desired task name')
    p.add_option('--method', '--m', action='store', dest='method', type='string', default='svm',
                 help='type the method name')
    p.add_option('--dim', action='store', dest='dim', type=int, default=4,
                 help='type the desired dimension')
    p.add_option('--data_path', action='store', dest='sRecordDataPath',
                 default='/home/dpark/hrl_file_server/dpark_data/anomaly/ICRA2017', \
                 help='Enter a record data path')
    opt, args = p.parse_args()
    rospy.init_node(opt.task)


    rf_center     = 'kinEEPos'        
    scale         = 1.0
    local_range   = 10.0    



    # From ICRA 2016 to TRO2017
    if False:
        from hrl_anomaly_detection.params import *

        if opt.task == 'scooping':
            subject_names = ['Wonyoung', 'Tom', 'lin', 'Ashwin', 'Song', 'Henry2'] #'Henry',         
            raw_data_path, _, param_dict = getScooping(opt.task, False, \
                                                       False, False,\
                                                       rf_center, local_range, dim=opt.dim)
            check_method      = opt.method # cssvm
            save_data_path    = '/home/dpark/hrl_file_server/dpark_data/anomaly/RSS2016/'+opt.task+'_data/demo'
            param_dict['SVM'] = {'renew': False, 'w_negative': 3.0, 'gamma': 0.3, 'cost': 6.0, \
                                 'class_weight': 1.5e-2, 'logp_offset': 100, 'ths_mult': -2.0}

        elif opt.task == 'feeding':
            subject_names = ['Tom', 'lin', 'Ashwin', 'Song'] #'Wonyoung']        
            raw_data_path, _, param_dict = getFeeding(opt.task, False, \
                                                      False, False,\
                                                      rf_center, local_range, dim=opt.dim)
            check_method      = opt.method # cssvm
            save_data_path    = '/home/dpark/hrl_file_server/dpark_data/anomaly/RSS2016/'+opt.task+'_data/demo'
            param_dict['SVM'] = {'renew': False, 'w_negative': 1.3, 'gamma': 0.0103, 'cost': 1.0,\
                                 'class_weight': 0.05, 'logp_offset': 200, 'ths_mult': -2.5}

        elif opt.task == 'pushing_microwhite':
            subject_names = ['gatsbii']        
            raw_data_path, _, param_dict = getPushingMicroWhite(opt.task, False, \
                                                                False, False,\
                                                                rf_center, local_range, dim=opt.dim)
            check_method      = opt.method # cssvm
            save_data_path    = '/home/dpark/hrl_file_server/dpark_data/anomaly/RSS2016/'+opt.task+'_data/demo'
            param_dict['SVM'] = {'renew': False, 'w_negative': 3.0, 'gamma': 0.3, 'cost': 6.0, \
                                 'class_weight': 1.5e-2, 'logp_offset': 100, 'ths_mult': -2.0}

        else:
            sys.exit()
    else:
        
        if opt.task == 'scooping':
            subject_names = [] 
            raw_data_path, _, param_dict = getScooping(opt.task, False, \
                                                       False, False,\
                                                       rf_center, local_range, dim=opt.dim)
            check_method      = opt.method # cssvm
            save_data_path    = '/home/dpark/hrl_file_server/dpark_data/anomaly/ICRA2017'+opt.task+'_data/demo'
            param_dict['SVM'] = {'renew': False, 'w_negative': 3.0, 'gamma': 0.3, 'cost': 6.0, \
                                 'class_weight': 1.5e-2, 'logp_offset': 100, 'ths_mult': -2.0}


    ad = anomaly_detector(subject_names, opt.task, check_method, raw_data_path, save_data_path, \
                          param_dict)
    ad.run()

