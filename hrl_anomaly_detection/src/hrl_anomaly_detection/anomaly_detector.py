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
import datetime

# util
import numpy as np
import hrl_lib.quaternion as qt
from hrl_anomaly_detection import util
from hrl_anomaly_detection import data_manager as dm
from sound_play.libsoundplay import SoundClient
import hrl_lib.util as ut

# learning
from hrl_anomaly_detection.hmm import learning_hmm
from hrl_anomaly_detection.hmm import learning_util as hmm_util
from sklearn import preprocessing

# Classifier
from hrl_anomaly_detection.classifiers import classifier as cb
from hrl_anomaly_detection.classifiers.classifier_util import *

# msg
from hrl_anomaly_detection.msg import MultiModality
from std_msgs.msg import String, Float64
from hrl_srvs.srv import Bool_None, Bool_NoneResponse, StringArray_None

#
from matplotlib import pyplot as plt


QUEUE_SIZE = 10

class anomaly_detector:
    def __init__(self, subject_names, task_name, check_method, raw_data_path, save_data_path,\
                 param_dict, data_renew=False, hmm_renew=False, viz=False, auto_update=False, \
                 debug=False):
        rospy.loginfo('Initializing anomaly detector')

        self.subject_names   = subject_names
        self.task_name       = task_name.lower()
        self.raw_data_path   = raw_data_path
        self.save_data_path  = save_data_path
        self.debug           = debug

        self.enable_detector = False
        self.cur_task        = None
        self.soundHandle     = SoundClient()
        self.dataList        = []
        self.auto_update     = auto_update
        self.used_file_list  = []
        self.figure_flag     = False
        
        # Params
        self.param_dict = param_dict        
        self.classifier_method = check_method
        self.startOffsetSize = 4
        self.startCheckIdx   = 20
        self.exp_sensitivity = True
        self.nUpdateFreq = 3
        
        self.nEmissionDim = None
        self.ml = None
        self.classifier = None
        self.t1 = datetime.datetime.now()


        # Comms
        self.lock = threading.Lock()        

        self.initParams()
        self.initComms()
        self.initDetector(data_renew=data_renew, hmm_renew=hmm_renew)

        self.viz = viz
        if viz:
            print "Visualization enabled!!!"
            self.figure_flag = False
            ## import hrl_lib.circular_buffer as cb
            ## self.feature_buf = cb.CircularBuffer(100, (self.nEmissionDim,))
        
        self.reset()
        print "=========================================================="
        print "Initialization completed!! : ", self.task_name
        print "=========================================================="

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
            self.add_logp_d = self.param_dict['HMM'].get('add_logp_d', True)

            self.SVM_dict        = self.param_dict['SVM']

            if 'svm' in self.classifier_method or 'sgd' in self.classifier_method:
                self.init_w_positive = self.param_dict['AD'][self.classifier_method+'_w_positive']
                self.w_max = self.param_dict['ROC'][self.classifier_method+'_param_range'][-1]
                self.w_min = self.param_dict['ROC'][self.classifier_method+'_param_range'][0]
            elif self.classifier_method == 'progress_time_cluster':                    
                self.w_max = self.param_dict['ROC']['progress_param_range'][-1]
                self.w_min = self.param_dict['ROC']['progress_param_range'][0]
            else:
                print "sensitivity info is not available"
                sys.exit()

            if self.w_min > self.w_max:
                temp = self.w_min
                self.w_min = self.w_max
                self.w_max = temp
                
            # we use logarlism for the sensitivity
            if self.exp_sensitivity:
                self.w_max = np.log(self.w_max)
                self.w_min = np.log(self.w_min)


    def initComms(self):
        # Publisher
        self.action_interruption_pub = rospy.Publisher('/hrl_manipulation_task/InterruptAction', String,
                                                       queue_size=QUEUE_SIZE)
        self.task_interruption_pub   = rospy.Publisher("/manipulation_task/emergency", String,
                                                       queue_size=QUEUE_SIZE)
        self.sensitivity_pub         = rospy.Publisher("manipulation_task/ad_sensitivity_state", \
                                                       Float64, queue_size=QUEUE_SIZE, latch=True)

        # Subscriber # TODO: topic should include task name prefix?
        rospy.Subscriber('/hrl_manipulation_task/raw_data', MultiModality, self.rawDataCallback)
        rospy.Subscriber('/manipulation_task/status', String, self.statusCallback)
        rospy.Subscriber('/manipulation_task/user_feedback', String, self.userfbCallback)
        rospy.Subscriber('manipulation_task/ad_sensitivity_request', Float64, self.sensitivityCallback)

        # Service
        self.detection_service = rospy.Service('anomaly_detector_enable', Bool_None, self.enablerCallback)
        ## self.update_service    = rospy.Service('anomaly_detector_update', StringArray_None, self.updateCallback)
        # NOTE: when and how update?

    def initDetector(self, data_renew=False, hmm_renew=False):
        print "Initializing a detector with ", self.classifier_method
        
        train_pkl = os.path.join(save_data_path, self.task_name + '_demo.pkl')
        startIdx  = 4
        (success_list, failure_list) = \
          util.getSubjectFileList(self.raw_data_path, self.subject_names, self.task_name)
        self.used_file_list = success_list+failure_list

        
        if os.path.isfile(train_pkl) and hmm_renew is False:
            d = ut.load_pickle(train_pkl)
            # HMM
            self.nEmissionDim = d['nEmissionDim']
            self.A            = d['A']
            self.B            = d['B']
            self.pi           = d['pi']
            self.ml = learning_hmm.learning_hmm(self.nState, self.nEmissionDim, verbose=False)
            self.ml.set_hmm_object(self.A, self.B, self.pi)
            
            X_train_org   = d['X_train_org'] 
            Y_train_org   = d['Y_train_org']
            idx_train_org = d['idx_train_org']
            nLength      = d['nLength']            
            self.handFeatureParams = d['param_dict']
            self.normalTrainData   = d.get('normalTrainData', None)

            if self.debug:
                self.visualization()
                sys.exit()

        else:
            print "Started get data set"
            dd = dm.getDataSet(self.subject_names, self.task_name, self.raw_data_path, \
                               self.save_data_path, self.rf_center, \
                               self.rf_radius,\
                               downSampleSize=self.downSampleSize, \
                               scale=1.0,\
                               ae_data=False,\
                               handFeatures=self.handFeatures, \
                               cut_data=self.cut_data,\
                               data_renew=data_renew)

            self.handFeatureParams = dd['param_dict']

            # do we need folding????????????????
            if self.nNormalFold > 1:
                # Task-oriented hand-crafted features        
                kFold_list = dm.kFold_data_index2(len(dd['successData'][0]), len(dd['failureData'][0]), \
                                                      self.nNormalFold, self.nAbnormalFold )
                # Select the first fold as the training data (need to fix?)
                (normalTrainIdx, abnormalTrainIdx, normalTestIdx, abnormalTestIdx) = kFold_list[0]
            else:
                #TODO?
                normalTrainIdx   = range(len(dd['successData'][0]))
                abnormalTrainIdx = range(len(dd['failureData'][0]))
                normalTestIdx   = None
                abnormalTestIdx = None

            # dim x sample x length # TODO: what is the best selection?
            if self.classifier_method.find('svm')>=0:
                normalTrainData   = dd['successData'][:, normalTrainIdx, :]   * self.scale
                abnormalTrainData = dd['failureData'][:, abnormalTrainIdx, :] * self.scale 
                ## if normalTestIdx is not None:
                ##     normalTestData    = dd['successData'][:, normalTestIdx, :]    * self.scale
                ##     abnormalTestData  = dd['failureData'][:, abnormalTestIdx, :]  * self.scale
            elif self.classifier_method.find('sgd')>=0:
                normalTrainData   = dd['successData'][:, normalTrainIdx, :]   * self.scale
                abnormalTrainData = dd['failureData'][:, abnormalTrainIdx, :] * self.scale
                ## if normalTestIdx is not None:
                ##     normalTestData    = dd['successData'][:, normalTestIdx, :]    * self.scale
                ##     abnormalTestData  = dd['failureData'][:, abnormalTestIdx, :]  * self.scale


            if self.debug:
                self.normalTrainData = normalTrainData
                self.nEmissionDim   = len(normalTrainData)
                self.visualization()
                sys.exit()


            # training hmm
            self.nEmissionDim   = len(normalTrainData)
            detection_param_pkl = os.path.join(self.save_data_path, 'hmm_'+self.task_name+'_demo.pkl')
            self.ml = learning_hmm.learning_hmm(self.nState, self.nEmissionDim, verbose=False)
            if self.param_dict['data_param']['handFeatures_noise']:
                ret = self.ml.fit(normalTrainData+
                                  np.random.normal(0.0, 0.03, np.shape(normalTrainData) )*self.scale, \
                                  cov_mult=[self.cov]*(self.nEmissionDim**2),
                                  ml_pkl=detection_param_pkl, use_pkl=(not hmm_renew))
            else:
                ret = self.ml.fit(normalTrainData, cov_mult=[self.cov]*(self.nEmissionDim**2),
                                  ml_pkl=detection_param_pkl, use_pkl=(not hmm_renew))

            if ret == 'Failure':
                print "-------------------------"
                print "HMM returned failure!!   "
                print "-------------------------"
                sys.exit()

            #-----------------------------------------------------------------------------------------
            # Classifier train data
            #-----------------------------------------------------------------------------------------
            trainDataX = []
            trainDataY = []
            for i in xrange(self.nEmissionDim):
                temp = np.vstack([normalTrainData[i], abnormalTrainData[i]])
                trainDataX.append( temp )

            trainDataY = np.hstack([ -np.ones(len(normalTrainData[0])), \
                                    np.ones(len(abnormalTrainData[0])) ])

            r = Parallel(n_jobs=-1)(delayed(learning_hmm.computeLikelihoods)(i, self.ml.A, self.ml.B, \
                                                                             self.ml.pi, self.ml.F,
                                                                             [ trainDataX[j][i] for j in xrange(self.nEmissionDim) ],
                                                                             self.ml.nEmissionDim, self.ml.nState,
                                                                             startIdx=startIdx, bPosterior=True)
                                                                             for i in xrange(len(trainDataX[0])))
            _, ll_classifier_train_idx, ll_logp, ll_post = zip(*r)

            # nSample x nLength
            ll_classifier_train_X, ll_classifier_train_Y = \
              learning_hmm.getHMMinducedFeatures(ll_logp, ll_post, trainDataY, c=1.0, add_delta_logp=self.add_logp_d)

            # flatten the data
            X_train_org = []
            Y_train_org = []
            idx_train_org = []
            for i in xrange(len(ll_classifier_train_X)):
                for j in xrange(len(ll_classifier_train_X[i])):
                    X_train_org.append(ll_classifier_train_X[i][j])
                    Y_train_org.append(ll_classifier_train_Y[i][j])
                    idx_train_org.append(ll_classifier_train_idx[i][j])

            d = {}
            d['A']  = self.ml.A
            d['B']  = self.ml.B
            d['pi'] = self.ml.pi
            d['nEmissionDim']  = self.nEmissionDim
            d['X_train_org']   = X_train_org
            d['Y_train_org']   = Y_train_org
            d['idx_train_org'] = idx_train_org
            d['nLength']       = nLength = len(normalTrainData[0][0])
            d['param_dict']    = self.handFeatureParams
            d['normalTrainData'] = self.normalTrainData = normalTrainData
            ut.save_pickle(d, train_pkl)

        # data preparation
        self.scaler        = preprocessing.StandardScaler()
        self.Y_train_org   = Y_train_org
        self.idx_train_org = idx_train_org
        if 'svm' in self.classifier_method or 'sgd' in self.classifier_method:
            self.X_scaled = self.scaler.fit_transform(X_train_org)
        else:
            self.X_scaled = X_train_org
            
        print self.classifier_method, " : Before classification : ", \
          np.shape(self.X_scaled), np.shape(self.Y_train_org)
            
        # Fit Classifier
        self.classifier = cb.classifier(method=self.classifier_method, nPosteriors=self.nState, \
                                        nLength=nLength - startIdx)
        self.classifier.set_params(**self.SVM_dict)
        self.classifier.set_params( class_weight=self.init_w_positive )        # for svm , sgd
        self.classifier.fit(self.X_scaled, self.Y_train_org, self.idx_train_org, parallel=False)
        print "Finished to train "+self.classifier_method

        self.pubSensitivity()        
        return


    #-------------------------- Communication fuctions --------------------------
    def enablerCallback(self, msg):

        if msg.data is True:
            rospy.loginfo("anomaly detector enabled")
            self.enable_detector = True
            # visualize sensitivity
            self.pubSensitivity()                    
        else:
            rospy.loginfo("anomaly detector disabled")
            # Reset detector
            self.enable_detector = False
            self.reset() #TODO: may be it should be removed

        return Bool_NoneResponse()


    def rawDataCallback(self, msg):
        '''
        Subscribe raw data
        '''
        
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

        self.vision_landmark_pos  = msg.vision_landmark_pos
        self.vision_landmark_quat = msg.vision_landmark_quat

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
        newData = np.array(self.extractHandFeature()) * self.scale #array

        # get offset
        if self.dataList == [] or len(self.dataList[0][0]) < self.startOffsetSize:
            self.offsetData = np.zeros(np.shape(newData))            
        elif len(self.dataList[0][0]) == self.startOffsetSize:
            refData = np.reshape( np.mean(self.normalTrainData[:,:,:self.startOffsetSize], axis=(1,2)), \
                                  (self.nEmissionDim,1,1) ) # 4,1,1
            curData = np.reshape( np.mean(self.dataList, axis=(1,2)), (self.nEmissionDim,1,1) ) # 4,1,1
            self.offsetData = refData - curData
                                  
            for i in xrange(self.nEmissionDim):
                self.dataList[i] = (np.array(self.dataList[i]) + self.offsetData[i][0][0]).tolist()
        newData += self.offsetData
        
        if len(self.dataList) == 0:
            self.dataList = np.array(newData).tolist()
        else:                
            ## self.dataList = np.swapaxes(self.dataList, 0,1)
            # dim x sample x length
            for i in xrange(self.nEmissionDim):
                self.dataList[i][0] = self.dataList[i][0] + [newData[i][0][0]]
            ## self.dataList = np.swapaxes(self.dataList, 0,1)
                       
        self.lock.release()
        self.t2 = datetime.datetime.now()
        print "time: ", self.t2 - self.t1
        self.t1 = self.t1


    def statusCallback(self, msg):
        '''
        Subscribe current task 
        '''
        self.cur_task = msg.data.lower()


    def sensitivityCallback(self, msg):
        '''
        Requested value's range is 0~1.
        Update the classifier only using current training data!!
        '''
        
        sensitivity_req = msg.data
        if sensitivity_req > 1.0: sensitivity_req = 1.0
        if sensitivity_req < 0.0: sensitivity_req = 0.0
        print "Requested sensitivity is [0~1]: ", sensitivity_req

        if 'svm' in self.classifier_method or 'sgd' in self.classifier_method:
            if self.exp_sensitivity:
                sensitivity_des = np.exp(sensitivity_req*(self.w_max-self.w_min)+self.w_min)
            else:
                sensitivity_des = sensitivity_req*(self.w_max-self.w_min)+self.w_min                
            self.classifier.set_params(class_weight=sensitivity_des)

            if 'svm' in self.classifier_method:
                self.classifier.fit(self.X_scaled, self.Y_train_org, self.idx_train_org)
            elif 'sgd' in self.classifier_method:
                self.classifier.fit(self.X_scaled, self.Y_train_org, self.idx_train_org)
            print "Classifier is updated!"

        else:
            print "not supported method"
            sys.exit()
            
            ## elif self.classifier_method == 'progress_time_cluster':                    
            ##     self.classifier.set_params(ths_mult=sensitivity_req*(self.w_max-self.w_min)+self.w_min)
            ## elif self.classifier_method == 'progress_time_cluster':                    
            ##     sensitivity = (self.classifier.ths_mult-self.w_min)/(self.w_max-self.w_min)
            ##     print "Current sensitivity is ", sensitivity, self.classifier.ths_mult

        self.pubSensitivity()

        
    def userfbCallback(self, msg):
        user_feedback = msg.data
        print user_feedback

        print "Logger feedback received"
        if (user_feedback == "SUCCESS" or user_feedback == "FAIL") and self.auto_update:
            if self.used_file_list == []: return

            rospy.sleep(2.0)
            
            # check unused data
            (success_list, failure_list) = \
              util.getSubjectFileList(self.raw_data_path, self.subject_names, self.task_name)
            unused_fileList = [filename for filename in success_list if filename not in self.used_file_list]
            unused_fileList += [filename for filename in failure_list if filename not in self.used_file_list]

            ## print "Used file list ------------------------"
            ## for f in self.used_file_list:
            ##     print f
            
            print "Unused file list ------------------------"
            for f in unused_fileList:
                print os.path.split(f)[1]
            print "-----------------------------------------"
            
            if len(unused_fileList) == 0: return

            Y_test_org = []

            # check if both success and failure data exists
            s_flag = 0
            f_flag = 0
            for f in unused_fileList:
                if f.find("success")>=0:
                    s_flag = 1
                    Y_test_org.append(-1)
                elif f.find("failure")>=0:
                    f_flag = 1
                    Y_test_org.append(1)
            if s_flag*f_flag == 0: return

            print "Start to collect #success=",s_flag, " #failure=", f_flag
            trainData = dm.getDataList(unused_fileList, self.rf_center, self.rf_radius,\
                                       self.handFeatureParams,\
                                       downSampleSize = self.downSampleSize, \
                                       cut_data       = self.cut_data,\
                                       handFeatures   = self.handFeatures)

            # update
            ## HMM
            ll_logp, ll_post = self.ml.loglikelihoods(trainData, bPosterior=True)
            X, Y = learning_hmm.getHMMinducedFeatures(ll_logp, ll_post, Y_test_org)
            print "Features: ", np.shape(X), np.shape(Y)
            print "cur method", self.classifier_method
            
            ## Remove unseparable region and scaling it
            if self.classifier_method.find('svm')>=0:
                X_train_org, Y_train_org, _ = dm.flattenSample(X, Y, remove_fp=True)
                print "Before: ", np.shape(self.X_scaled), np.shape(self.Y_train_org)                
                self.X_scaled    = np.vstack([ self.X_scaled, self.scaler.transform(X_train_org) ])
                self.Y_train_org = np.hstack([ self.Y_train_org, Y_train_org])
                print "After : ", np.shape(self.X_scaled), np.shape(self.Y_train_org)                
                self.classifier.fit(self.X_scaled, self.Y_train_org)
            elif self.classifier_method.find('sgd')>=0:
                print "Before: ", np.shape(self.X_scaled), np.shape(self.Y_train_org)                
                p_train_X = []
                for i in xrange(len(X)):
                    p_train_X.append( self.scaler.transform(X[i]) )
                p_train_X, p_train_Y, p_train_W = getProcessSGDdata(p_train_X, Y, \
                                                                    weight=self.classifier.class_weight )
                print np.shape(p_train_X), np.shape(p_train_Y)
                self.classifier.partial_fit(p_train_X, p_train_Y, classes=[-1,1])

                self.X_scaled    = np.vstack([ self.X_scaled, p_train_X ])
                self.Y_train_org = np.hstack([ self.Y_train_org, p_train_Y ])
                print "After : ", np.shape(self.X_scaled), np.shape(self.Y_train_org)
            else:
                ## self.X_scaled = np.vstack([ self.X_scaled, X ])
                ## self.classifier.fit(self.X_scaled, self.Y_test_org)
                print "Not available update method"
                            
            print "Update completed!!!"

            

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
        X, Y = learning_hmm.getHMMinducedFeatures(ll_logp, ll_post, Y_test_org, c=1.0, add_delta_logp=self.add_logp_d)
        print "Features: ", np.shape(X), np.shape(Y)

        ## Remove unseparable region and scaling it
        X_train_org, Y_train_org, _ = dm.flattenSample(X, Y, remove_fp=True)
        if 'svm' in self.classifier_method:
            self.X_scaled = np.vstack([ self.X_scaled, self.scaler.transform(X_train_org) ])
        elif 'sgd' in self.classifier_method:
            self.X_scaled = self.scaler.transform(X_train_org)
        else:
            print "Not available method"
            sys.exit()

        # Run SGD? or SVM?
        if self.classifier_method.find('svm') >= 0:
            self.classifier.fit(self.X_scaled, self.Y_test_org)
        elif self.classifier_method.find('svm') >= 0:
            print "Not available"
            return StringArray_NoneResponse()
            self.classifier.partial_fit(self.X_scaled, self.Y_test_org, classes=[-1,1])            
        else:
            print "Not available update method"
            
        return StringArray_NoneResponse()



    #-------------------------- General fuctions --------------------------

    def pubSensitivity(self):
        if 'svm' in self.classifier_method or 'sgd' in self.classifier_method:        
            if self.exp_sensitivity:
                sensitivity = (np.log(self.classifier.class_weight)-self.w_min)/(self.w_max-self.w_min)
            else:
                sensitivity = (self.classifier.class_weight-self.w_min)/(self.w_max-self.w_min)
        else:
            print self.classifier_method, " is not supported method"
            sys.exit()

        ## elif self.classifier_method == 'progress_time_cluster':                    
        ##     sensitivity = (self.classifier.ths_mult-self.w_min)/(self.w_max-self.w_min)
        self.sensitivity_pub.publish(sensitivity)                                   
        print "Current sensitivity is [0~1]: ", sensitivity, \
          ', internal weight is ', self.classifier.class_weight

    ## def calSensitivity(self, val):
    ##     '''
    ##     Return scaled sensitivity (for visualization)
    ##     '''
    ##     if self.exp_sensitivity:
            
                                      
    def extractHandFeature(self):
        '''
        Run it on every time step
        '''

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

        # Crossmodal feature - relative Velocity --------------------------
        if 'crossmodal_targetEEVel' in self.handFeatures:
            print "Not available"            

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

        # Crossmodal feature - vision relative dist with sub vision target----
        if 'crossmodal_subArtagEEDist' in self.handFeatures:
            print "Not available"            

        # Crossmodal feature - vision relative angle --------------------------
        if 'crossmodal_subArtagEEAng' in self.handFeatures:                
            print "Not available"            

        # Crossmodal feature - vision relative dist with main(first) vision target----
        if 'crossmodal_landmarkEEDist' in self.handFeatures:
            data_dict['kinEEPosList']     = [np.array([self.kinematics_ee_pos]).T]
            data_dict['visionLandmarkPosList'] = [np.array([self.vision_landmark_pos]).T]

        # Crossmodal feature - vision relative angle --------------------------
        if 'crossmodal_landmarkEEAng' in self.handFeatures:
            data_dict['kinEEQuatList'] = [np.array([self.kinematics_ee_quat]).T]
            data_dict['visionLandmarkPosList'] = [np.array([self.vision_landmark_pos]).T]
            data_dict['visionLandmarkQuatList'] = [np.array([self.vision_landmark_quat]).T]

        data, _ = dm.extractHandFeature(data_dict, self.handFeatures, scale=1.0, \
                                        param_dict = self.handFeatureParams)
                                        
        return data

    '''
    Reset parameters
    '''
    def reset(self):
        self.dataList = []
        self.enable_detector = False

    '''
    Run detector
    '''
    def run(self, freq=20):
        rospy.loginfo("Start to run anomaly detection: " + self.task_name)
        rate = rospy.Rate(freq) # 25Hz, nominally.
        while not rospy.is_shutdown():

            if len(self.dataList) >0 and self.viz:
                self.visualization()
            
            if self.enable_detector is False: 
                self.dataList = []
                self.last_logp = None
                self.last_post = None
                self.figure_flag = False
                continue
            
            if len(self.dataList) == 0 or len(self.dataList[0][0]) < self.startCheckIdx: continue

            #-----------------------------------------------------------------------
            self.lock.acquire()
            cur_length     = len(self.dataList[0][0])
            logp, post = self.ml.loglikelihood(self.dataList, bPosterior=True)
            self.lock.release()

            if logp is None: 
                print "logp is None => anomaly"
                self.action_interruption_pub.publish(self.task_name+'_anomaly')
                self.task_interruption_pub.publish(self.task_name+'_anomaly')
                self.soundHandle.play(2)
                self.enable_detector = False
                self.reset()
                continue

            post = post[cur_length-1]
            print "logp: ", logp, "  state: ", np.argmax(post), \
              " cutoff: ", self.param_dict['HMM']['nState']*0.9
            if np.argmax(post)==0 and logp < 0.0: continue
            if np.argmax(post)>self.param_dict['HMM']['nState']*0.9: continue

            if self.last_logp is None or self.last_post is None:
                self.last_logp = logp
                self.last_post = post
                continue
            else:                
                d_logp = logp - self.last_logp
                ## print np.shape(self.last_post), np.shape(post)
                d_post = hmm_util.symmetric_entropy(self.last_post, post)
                ll_classifier_test_X = [logp] + [d_logp/(d_post+1.0)] + post.tolist()
                self.last_logp = logp
                self.last_post = post
                

            if 'svm' in self.classifier_method or 'sgd' in self.classifier_method:
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

    def visualization(self):
        if self.figure_flag is False:
            fig = plt.figure()
            for i in xrange(self.nEmissionDim):
                self.ax = fig.add_subplot(100*self.nEmissionDim+10+i+1)
            plt.ion()
            plt.show()
            self.figure_flag = True

        del self.ax.collections[:]

        ## normalTrainData = self.scaler.inverse_transform( self.normalTrainData )
        ## if len(self.dataList) > 0:
        ##     dataList        = np.squeeze(self.dataList).swapaxes(0,1)
        ##     dataList        = self.scaler.inverse_transform( )
        for i in xrange(self.nEmissionDim):
            self.ax = plt.subplot(self.nEmissionDim,1,i+1)
            if len(self.dataList) > 0:
                self.ax.plot(self.dataList[i][0], '-r')

            # training data
            for j in xrange(len(self.normalTrainData[i])):
                self.ax.plot(self.normalTrainData[i][j],'-b')
            
            ## ax.set_xlim([0.3, 1.4])
            self.ax.set_ylim([-1.0, 2.0])
        plt.draw()
        
        if self.debug:
            rate = rospy.Rate(5) # 25Hz, nominally.            
            while not rospy.is_shutdown():
                continue
        
        

if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    p.add_option('--task', action='store', dest='task', type='string', default='scooping',
                 help='type the desired task name')
    p.add_option('--method', '--m', action='store', dest='method', type='string', default='svm',
                 help='type the method name')
    p.add_option('--dim', action='store', dest='dim', type=int, default=4,
                 help='type the desired dimension')
    p.add_option('--auto_update', '--au', action='store_true', dest='bAutoUpdate',
                 default=False, help='Enable auto update.')
    p.add_option('--debug', '--d', action='store_true', dest='bDebug',
                 default=False, help='Enable debugging mode.')
    

    p.add_option('--dataRenew', '--dr', action='store_true', dest='bDataRenew',
                 default=False, help='Renew pickle files.')
    p.add_option('--hmmRenew', '--hr', action='store_true', dest='bHMMRenew',
                 default=False, help='Renew HMM parameters.')
    p.add_option('--viz', action='store_true', dest='bViz',
                 default=False, help='Visualize data.')
    
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
            print "Not supported task"
            sys.exit()
    else:
        from hrl_anomaly_detection.ICRA2017_params import *
        subject = 'park'
        
        if opt.task == 'scooping':
            ## subject_names = ['test'] 
            subject_names = ['Zack'] 
            raw_data_path, save_data_path, param_dict = getScooping(opt.task, False, \
                                                                    False, False,\
                                                                    rf_center, local_range, dim=opt.dim)
            check_method      = opt.method
            save_data_path    = os.path.expanduser('~')+'/hrl_file_server/dpark_data/anomaly/ICRA2017/'+\
              subject_names[0]+'_'+opt.task+'_data/demo'
            param_dict['SVM'] = {'renew': False, 'w_negative': 4.0, 'gamma': 0.04, 'cost': 4.6, \
                                 'class_weight': 1.5e-2, 'logp_offset': 0, 'ths_mult': -2.0,\
                                 'sgd_gamma':0.32, 'sgd_w_negative':2.5,}

            param_dict['data_param']['nNormalFold']   = 1
            param_dict['data_param']['nAbnormalFold'] = 1

        elif opt.task == 'feeding':
            ## subject_names = ['test'] 
            subject_names = [subject] 
            raw_data_path, save_data_path, param_dict = getFeeding(opt.task, False, \
                                                                    False, False,\
                                                                    rf_center, local_range, dim=opt.dim)
            check_method      = opt.method
            save_data_path    = os.path.expanduser('~')+'/hrl_file_server/dpark_data/anomaly/ICRA2017/'+\
              subject_names[0]+'_'+opt.task+'_data/demo'
            param_dict['SVM'] = {'renew': False, 'w_negative': 4.0, 'gamma': 0.04, 'cost': 4.6, \
                                 'class_weight': 1.5e-2, 'logp_offset': 0, 'ths_mult': -2.0,\
                                 'sgd_gamma':0.32, 'sgd_w_negative':2.5}

            param_dict['data_param']['nNormalFold']   = 1
            param_dict['data_param']['nAbnormalFold'] = 1


    ad = anomaly_detector(subject_names, opt.task, check_method, raw_data_path, save_data_path, \
                          param_dict, data_renew=opt.bDataRenew, hmm_renew=opt.bHMMRenew, \
                          viz=opt.bViz, auto_update=opt.bAutoUpdate,\
                          debug=opt.bDebug)
    ad.run()

