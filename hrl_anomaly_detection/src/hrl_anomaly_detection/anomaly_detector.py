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
## import hrl_lib.circular_buffer as cb
from collections import deque
import pickle

# data
import hrl_manipulation_task.record_data as rd


# learning
from hrl_anomaly_detection.hmm import learning_hmm
from hrl_anomaly_detection.hmm import learning_util as hmm_util
from sklearn import preprocessing

# Classifier
from hrl_anomaly_detection.classifiers import classifier as clf
from hrl_anomaly_detection.classifiers.classifier_util import *

# msg
from hrl_anomaly_detection.msg import MultiModality
from std_msgs.msg import String, Float64
from hrl_srvs.srv import Bool_None, Bool_NoneResponse, StringArray_None
from hrl_msgs.msg import FloatArray, StringArray

#
from matplotlib import pyplot as plt


QUEUE_SIZE = 10

class anomaly_detector:
    def __init__(self, subject_names, task_name, check_method, raw_data_path, save_data_path,\
                 param_dict, data_renew=False, hmm_renew=False, clf_renew=False, viz=False, \
                 auto_update=False, \
                 debug=False, sim=False):
        rospy.loginfo('Initializing anomaly detector')

        self.subject_names   = subject_names
        self.task_name       = task_name.lower()
        self.raw_data_path   = raw_data_path
        self.save_data_path  = save_data_path
        self.debug           = debug

        self.enable_detector = False
        self.soundHandle     = SoundClient()
        self.dataList        = []
        self.auto_update     = auto_update        

        # auto update related params
        self.nMinUpdateFiles = 1
        self.used_file_list  = []
        self.anomaly_flag    = False
        self.figure_flag     = False
        self.update_count    = 0.0
        ## self.fileList_buf = cb.CircularBuffer(self.nMinUpdateFiles, (1,))       
        
        # Params
        self.param_dict = param_dict        
        self.classifier_method = check_method
        self.startOffsetSize = 4
        self.startCheckIdx   = 20
        self.nUpdateFreq = 3
        self.sgd_n_iter = 100
        
        self.nEmissionDim = None
        self.ml = None
        self.classifier = None
        self.bSim       = sim
        if self.bSim: self.cur_task = self.task_name
        else:         self.cur_task = None
        ## self.t1 = datetime.datetime.now()



        # evaluation test
        self.nRecentTests = 2
        self.ll_recent_test_X = deque([],self.nRecentTests)
        self.ll_recent_test_Y = deque([],self.nRecentTests)
        self.nTests           = 20
        self.ll_test_X        = deque([],self.nTests)
        self.ll_test_Y        = deque([],self.nTests)        

        # evaluation reference data
        self.eval_fileList = None
        self.eval_test_X = None
        self.eval_test_Y = None
        self.ref_acc_list  = []
        self.cum_acc_list  = []
        self.update_list   = []
        self.acc_part = 100.0
        self.acc_all  = 100.0
        self.acc_ref  = 100.0

        # Comms
        self.lock = threading.Lock()        

        self.initParams()
        self.initComms()
        self.initDetector(data_renew=data_renew, hmm_renew=hmm_renew, clf_renew=clf_renew)

        self.viz = viz
        if viz:
            rospy.loginfo( "Visualization enabled!!!")
            self.figure_flag = False
        
        self.reset()
        rospy.loginfo( "==========================================================")
        rospy.loginfo( "Initialization completed!! : %s", self.task_name)
        rospy.loginfo( "==========================================================")

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
            self.w_positive = rospy.get_param(self.classifier_method+'_w_positive')
            self.w_max = self.param_dict['ROC'][self.classifier_method+'_param_range'][-1]
            self.w_min = self.param_dict['ROC'][self.classifier_method+'_param_range'][0]
            self.exp_sensitivity = True
        elif self.classifier_method == 'progress_time_cluster':                    
            self.w_positive = rospy.get_param('progress_ths_mult')
            self.w_max = self.param_dict['ROC']['progress_param_range'][-1]
            self.w_min = self.param_dict['ROC']['progress_param_range'][0]
            self.exp_sensitivity = False
        else:
            rospy.loginfo( "sensitivity info is not available")
            sys.exit()

        if self.w_min > self.w_max:
            temp = self.w_min
            self.w_min = self.w_max
            self.w_max = temp

        if self.w_positive > self.w_max:
            self.w_positive = self.w_max
            if 'svm' in self.classifier_method or 'sgd' in self.classifier_method:
                rospy.set_param(self.classifier_method+'_w_positive', float(self.w_positive))
            elif self.classifier_method == 'progress_time_cluster':                    
                rospy.set_param('progress_ths_mult', float(self.w_positive))                
        elif self.w_positive < self.w_min:
            self.w_positive = self.w_min
            if 'svm' in self.classifier_method or 'sgd' in self.classifier_method:
                rospy.set_param(self.classifier_method+'_w_positive', float(self.w_positive))
            elif self.classifier_method == 'progress_time_cluster':                    
                rospy.set_param('progress_ths_mult', float(self.w_positive))

        # we use logarlism for the sensitivity
        if self.exp_sensitivity:
            self.w_max = np.log10(self.w_max)
            self.w_min = np.log10(self.w_min)


        if self.bSim:
            rospy.loginfo( "get subject files for simulation" )
            ## sensitivity_des = self.sensitivity_GUI_to_clf(0.5)
            ## self.w_positive = sensitivity_des                
            ## self.classifier.set_params(class_weight=self.w_positive)
            ## rospy.set_param(self.classifier_method+'_w_positive', float(sensitivity_des))            
            test_fileList = util.getSubjectFileList(self.raw_data_path, \
                                                    subject_names, \
                                                    self.task_name, \
                                                    time_sort=True,\
                                                    no_split=True)

            idx_list = range(len(test_fileList))
            random.shuffle(idx_list)
            self.eval_run_fileList = test_fileList[:len(idx_list)/2]
            self.eval_ref_fileList = test_fileList[len(idx_list)/2:]



    def initComms(self):
        # Publisher
        self.action_interruption_pub = rospy.Publisher('/hrl_manipulation_task/InterruptAction', String,
                                                       queue_size=QUEUE_SIZE)
        self.task_interruption_pub   = rospy.Publisher("/manipulation_task/emergency", String,
                                                       queue_size=QUEUE_SIZE)
        self.sensitivity_pub         = rospy.Publisher("manipulation_task/ad_sensitivity_state", \
                                                       Float64, queue_size=QUEUE_SIZE, latch=True)
        self.accuracy_pub = rospy.Publisher("manipulation_task/eval_status",\
                                            FloatArray, queue_size=QUEUE_SIZE, latch=True)


        # Subscriber # TODO: topic should include task name prefix?
        rospy.Subscriber('/hrl_manipulation_task/raw_data', MultiModality, self.rawDataCallback)
        rospy.Subscriber('/manipulation_task/status', String, self.statusCallback)
        rospy.Subscriber('/manipulation_task/user_feedback', StringArray, self.userfbCallback)
        rospy.Subscriber('manipulation_task/ad_sensitivity_request', Float64, self.sensitivityCallback)

        # Service
        self.detection_service = rospy.Service('anomaly_detector_enable', Bool_None, self.enablerCallback)
        ## self.update_service    = rospy.Service('anomaly_detector_update', StringArray_None, self.updateCallback)
        # NOTE: when and how update?

    def initDetector(self, data_renew=False, hmm_renew=False, clf_renew=False):
        rospy.loginfo( "Initializing a detector with %s of %s", self.classifier_method, self.task_name)
        
        self.hmm_model_pkl = os.path.join(self.save_data_path, 'hmm_'+self.task_name + '.pkl')
        self.scaler_model_file = os.path.join(self.save_data_path, 'scaler_'+self.task_name+'.pkl' )
        self.classifier_model_file = os.path.join(self.save_data_path, 'classifier_'+self.task_name+\
                                                  '_'+self.classifier_method+'.pkl' )
        
        startIdx  = 4
        (success_list, failure_list) = \
          util.getSubjectFileList(self.raw_data_path, self.subject_names, self.task_name, time_sort=True)
        self.used_file_list = success_list+failure_list

        if os.path.isfile(self.hmm_model_pkl) and hmm_renew is False:
            rospy.loginfo( "Start to load an hmm model of %s", self.task_name)
            d = ut.load_pickle(self.hmm_model_pkl)
            # HMM
            self.nEmissionDim = d['nEmissionDim']
            self.A            = d['A']
            self.B            = d['B']
            self.pi           = d['pi']
            self.ml = learning_hmm.learning_hmm(self.nState, self.nEmissionDim, verbose=False)
            self.ml.set_hmm_object(self.A, self.B, self.pi)
            
            ll_classifier_train_X = d['ll_classifier_train_X']
            ll_classifier_train_Y = d['ll_classifier_train_Y']
            X_train_org   = d['X_train_org'] 
            Y_train_org   = d['Y_train_org']
            idx_train_org = d['idx_train_org']
            self.nLength       = d['nLength']            
            self.handFeatureParams = d['param_dict']
            self.normalTrainData   = d.get('normalTrainData', None)

            if self.debug:
                self.visualization()
                sys.exit()

        else:
            rospy.loginfo( "Start to train an hmm model of %s", self.task_name)
            rospy.loginfo( "Started get data set")
            dd = dm.getDataSet(self.subject_names, self.task_name, self.raw_data_path, \
                               self.save_data_path, self.rf_center, \
                               self.rf_radius,\
                               downSampleSize=self.downSampleSize, \
                               scale=1.0,\
                               ae_data=False,\
                               handFeatures=self.handFeatures, \
                               cut_data=self.cut_data,\
                               data_renew=data_renew, \
                               time_sort=True)

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
            normalTrainData   = dd['successData'][:, normalTrainIdx, :]   * self.scale
            abnormalTrainData = dd['failureData'][:, abnormalTrainIdx, :] * self.scale 

            if self.debug:
                self.normalTrainData = normalTrainData
                self.nEmissionDim   = len(normalTrainData)
                self.visualization()
                sys.exit()


            # training hmm
            self.nEmissionDim   = len(normalTrainData)
            #detection_param_pkl = os.path.join(self.save_data_path, 'hmm_'+self.task_name+'_demo.pkl')
            self.ml = learning_hmm.learning_hmm(self.nState, self.nEmissionDim, verbose=False)
            if self.param_dict['data_param']['handFeatures_noise']:
                ret = self.ml.fit(normalTrainData+
                                  np.random.normal(0.0, 0.03, np.shape(normalTrainData) )*self.scale, \
                                  cov_mult=[self.cov]*(self.nEmissionDim**2))
                                  ## ml_pkl=detection_param_pkl, use_pkl=(not hmm_renew))
            else:
                ret = self.ml.fit(normalTrainData, cov_mult=[self.cov]*(self.nEmissionDim**2))
                                  ## ml_pkl=detection_param_pkl, use_pkl=(not hmm_renew))

            if ret == 'Failure':
                rospy.loginfo( "-------------------------")
                rospy.loginfo( "HMM returned failure!!   ")
                rospy.loginfo( "-------------------------")
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

            d                  = {}
            d['A']             = self.ml.A
            d['B']             = self.ml.B
            d['pi']            = self.ml.pi
            d['nEmissionDim']  = self.nEmissionDim
            d['ll_classifier_train_X'] = ll_classifier_train_X
            d['ll_classifier_train_Y'] = ll_classifier_train_Y
            d['X_train_org']   = X_train_org
            d['Y_train_org']   = Y_train_org
            d['idx_train_org'] = idx_train_org
            d['nLength']       = self.nLength = len(normalTrainData[0][0])
            d['param_dict']    = self.handFeatureParams
            d['normalTrainData'] = self.normalTrainData = normalTrainData
            ut.save_pickle(d, self.hmm_model_pkl)

        self.nTrainData = len(self.normalTrainData[0])

        # Train a scaler and data preparation
        if 'svm' in self.classifier_method or 'sgd' in self.classifier_method:
            rospy.loginfo( "Start to load/train a scaler model")
            if os.path.isfile(self.scaler_model_file):
                rospy.loginfo("Start to load a scaler model")
                with open(self.scaler_model_file, 'rb') as f:
                    self.scaler = pickle.load(f)                
                self.X_train_org = self.scaler.transform(X_train_org)                
            else:
                self.scaler      = preprocessing.StandardScaler()
                self.X_train_org = self.scaler.fit_transform(X_train_org)
            self.Y_train_org   = Y_train_org
            self.idx_train_org = idx_train_org

            self.X_partial_train   = self.X_train_org[len(self.X_train_org)/4]
            self.Y_partial_train   = self.Y_train_org[len(self.Y_train_org)/4]
            self.idx_partial_train = self.idx_train_org[len(self.idx_train_org)/4]            
        else:
            self.X_train_org = X_train_org
            self.Y_train_org   = Y_train_org
            self.idx_train_org = idx_train_org

        rospy.loginfo( self.classifier_method+" : Before classification : "+ \
          str(np.shape(self.X_train_org))+' '+str( np.shape(self.Y_train_org)))

                               
        ## if self.bSim:
        ##     # temp
        ##     self.w_positive = self.sensitivity_GUI_to_clf(0.5)                
    
          
        # Decareing Classifier
        self.classifier = clf.classifier(method=self.classifier_method, nPosteriors=self.nState, \
                                        nLength=self.nLength - startIdx)
        self.classifier.set_params(**self.SVM_dict)
        if 'sgd' in self.classifier_method or 'svm' in self.classifier_method:
            self.classifier.set_params( class_weight=self.w_positive )
            self.classifier.set_params( sgd_n_iter=self.sgd_n_iter )
        elif self.classifier_method == 'progress_time_cluster':
            ## ths_mult = np.ones(self.nState)*self.w_positive
            ths_mult = self.w_positive
            self.classifier.set_params( ths_mult=ths_mult )

        # Load / Fit the classifier
        if os.path.isfile(self.classifier_model_file) and clf_renew is False:
            rospy.loginfo( "Start to load a classifier model")
            self.classifier.load_model(self.classifier_model_file)
        else:
            rospy.loginfo( "Start to train a classifier model")
            self.classifier.fit(self.X_train_org, self.Y_train_org, self.idx_train_org)
            if self.bSim: self.classifier.save_model(self.classifier_model_file)
            rospy.loginfo( "Finished to train "+self.classifier_method)


        # scaling training data
        idx_list = range(len(ll_classifier_train_X))
        random.shuffle(idx_list)
        s_flag = True
        f_flag = True
        for i, count in enumerate(idx_list):
            train_X = []
            for j in xrange(len(ll_classifier_train_X[i])):
                if 'sgd' in self.classifier_method or 'svm' in self.classifier_method:
                    train_X.append( self.scaler.transform(ll_classifier_train_X[i][j]) )
                else:
                    train_X.append( ll_classifier_train_X[i][j] )

            ## if (s_flag is True and ll_classifier_train_Y[i][0] < 0) or True:
            ##     s_flag = False                
            ##     self.ll_test_X.append( train_X )
            ##     self.ll_test_Y.append( ll_classifier_train_Y[i] )
            ## elif (f_flag is True and ll_classifier_train_Y[i][0] > 0) or True:
            ##     f_flag = False                
            ##     self.ll_test_X.append( train_X )
            ##     self.ll_test_Y.append( ll_classifier_train_Y[i] )

        # recent data
        ## for i in xrange(self.nRecentTests):
        ##     self.ll_recent_test_X.append(self.ll_test_X[-self.nRecentTests+i])
        ##     self.ll_recent_test_Y.append(self.ll_test_Y[-self.nRecentTests+i])
            
        # info for GUI
        self.pubSensitivity()
        ## self.acc_part, _, _ = evaluation(list(self.ll_test_X), list(self.ll_test_Y), self.classifier)
        ## if self.bSim: acc, _, _ = self.evaluation_ref()
            
        ## msg = FloatArray()
        ## msg.data = [self.acc_part, self.acc_all]
        ## self.accuracy_pub.publish(msg)
        ## vizDecisionBoundary(self.X_train_org, self.Y_train_org, self.classifier, self.classifier.rbf_feature)

        

    #-------------------------- Communication fuctions --------------------------
    def enablerCallback(self, msg):

        if msg.data is True:
            rospy.loginfo("%s anomaly detector enabled", self.task_name)
            self.enable_detector = True
            self.anomaly_flag    = False            
            # visualize sensitivity
            self.pubSensitivity()                    
        else:
            rospy.loginfo("%s anomaly detector disabled", self.task_name)
            # Reset detector
            self.enable_detector = False
            self.reset() #TODO: may be it should be removed

        return Bool_NoneResponse()


    def rawDataCallback(self, msg):
        '''
        Subscribe raw data
        '''
        if self.cur_task is None: return
        if self.cur_task.find(self.task_name) < 0: return

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
            # dim x sample x length
            for i in xrange(self.nEmissionDim):
                self.dataList[i][0] = self.dataList[i][0] + [newData[i][0][0]]
                       
        self.lock.release()
        ## self.t2 = datetime.datetime.now()
        ## rospy.loginfo( "time: ", self.t2 - self.t1
        ## self.t1 = self.t1


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
        if self.classifier is None: return
        sensitivity_des = self.sensitivity_GUI_to_clf(msg.data)

        if 'svm' in self.classifier_method or 'sgd' in self.classifier_method:
            self.w_positive = sensitivity_des
            self.classifier.set_params(class_weight=self.w_positive)
            rospy.set_param(self.classifier_method+'_w_positive', float(sensitivity_des))
            self.classifier.fit(self.X_train_org, self.Y_train_org, self.idx_train_org, warm_start=True)
        elif self.classifier_method == 'progress_time_cluster':
            self.w_positive = sensitivity_des
            ## ths_mult = np.ones(self.nState)*self.w_positive
            ths_mult = self.w_positive
            self.classifier.set_params( ths_mult=ths_mult )
            rospy.set_param('progress_ths_mult', float(sensitivity_des))            
        else:
            rospy.loginfo( "not supported method")
            sys.exit()

        rospy.loginfo( "Classifier is updated!")

        self.acc_all, _, _ = evaluation(list(self.ll_test_X), list(self.ll_test_Y), self.classifier)
        if self.bSim:
            self.acc_ref, _, _ = self.evaluation_ref()
            print "acc ref: ", self.acc_ref

        msg = FloatArray()
        msg.data = [self.acc_part, self.acc_all]            
        self.accuracy_pub.publish(msg)                                   
            
        self.pubSensitivity()

        
    def userfbCallback(self, msg):
        
        if self.cur_task is None and self.bSim is False: return        
        if self.cur_task.find(self.task_name) < 0  and self.bSim is False: return
       
        user_feedback = rd.feedback_to_label( msg.data )        
        rospy.loginfo( "Logger feedback received: %s", user_feedback)

        if (user_feedback == "success" or user_feedback.find("fail" )>=0 ) and self.auto_update:
            if self.used_file_list == []: return

            print "If does not wake, check use_sim_time. If you are not running GAZEBO, it should be false."
            ## Need to wait until the last file saved!!
            rospy.sleep(2.0)

            # 4 cases
            update_flag  = False          
            if user_feedback == "success":
                if self.anomaly_flag is False:
                    rospy.loginfo( "Detection Status: True Negative - no update!!")
                else:
                    rospy.loginfo( "Detection Status: False positive - update!!!!!!!!!!!!!!!!!!!!!!! ")
                    update_flag = True
            else:
                if self.anomaly_flag is False:
                    rospy.loginfo( "Detection Status: False Negative - update!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    update_flag = True
                else:
                    rospy.loginfo( "Detection Status: True Positive - No update!!")

                                    
            # check unused data
            if self.bSim is False:
                unused_fileList = util.getSubjectFileList(self.raw_data_path, \
                                                          self.subject_names, \
                                                          self.task_name, \
                                                          time_sort=True,\
                                                          no_split=True)
                unused_fileList = [filename for filename in unused_fileList \
                                   if filename not in self.used_file_list]
            else:
                unused_fileList = self.new_run_file
                ## unused_fileList = self.unused_fileList
                
            # Remove no update data
            ## if update_flag is False:
            ##     self.used_file_list += unused_fileList
            ##     self.unused_fileList = []
            ##     ## return
            ## else:
            self.unused_fileList = unused_fileList


            rospy.loginfo( "Unused file list ------------------------")
            for f in self.unused_fileList:
                rospy.loginfo( os.path.split(f)[1])
            rospy.loginfo( "-----------------------------------------")
            
            if len(self.unused_fileList) == 0:
                rospy.logwarn("No saved file exists!")
                return
            elif len(self.unused_fileList) < self.nMinUpdateFiles:
                rospy.logwarn("Need %s more data", str(self.nMinUpdateFiles-len(self.unused_fileList)))
                return

            # need to update if 
            # check if both success and failure data exists
            # if no file exists, force to use the recent success/failure files.
            Y_test_org = []
            s_flag = 0
            f_flag = 0
            for f in self.unused_fileList:
                if f.find("success")>=0:
                    s_flag += 1
                    Y_test_org.append(-1)
                elif f.find("failure")>=0:
                    f_flag += 1
                    Y_test_org.append(1)

            rospy.loginfo( "Start to load #success= %i #failure= %i", s_flag, f_flag)

            
            nFakeData = 0
            if 'svm' in self.classifier_method or 'sgd' in self.classifier_method:
                if s_flag == 0:
                    for i in range(len(self.used_file_list)-1,-1,-1):
                        if self.used_file_list[i].find("success")>=0:
                            self.unused_fileList.append(self.used_file_list[i])
                            Y_test_org.append(-1)
                            nFakeData += 1
                            break
                if f_flag == 0:
                    for i in range(len(self.used_file_list)-1,-1,-1):
                        if self.used_file_list[i].find("failure")>=0:
                            self.unused_fileList.append(self.used_file_list[i])
                            Y_test_org.append(1)
                            nFakeData += 1
                            break

                ## if s_flag < f_flag:
                ##     max_count = f_flag-s_flag
                ##     for i in range(len(self.used_file_list)-1,-1,-1):
                ##         if self.used_file_list[i].find("success")>=0:
                ##             self.unused_fileList.append(self.used_file_list[i])
                ##             Y_test_org.append(-1)
                ##             nFakeData += 1
                ##             if nFakeData == max_count:
                ##                 break
                ## if s_flag > f_flag:
                ##     max_count = s_flag-f_flag
                ##     for i in range(len(self.used_file_list)-1,-1,-1):
                ##         if self.used_file_list[i].find("failure")>=0:
                ##             self.unused_fileList.append(self.used_file_list[i])
                ##             Y_test_org.append(1)
                ##             nFakeData += 1
                ##             if nFakeData == max_count:
                ##                 break

                        
            trainData = dm.getDataList(self.unused_fileList, self.rf_center, self.rf_radius,\
                                       self.handFeatureParams,\
                                       downSampleSize = self.downSampleSize, \
                                       cut_data       = self.cut_data,\
                                       handFeatures   = self.handFeatures)

            # scaling and applying offset            
            trainData = np.array(trainData)*self.scale
            trainData = self.applying_offset(trainData)

            # update
            ## HMM
            ll_logp, ll_post = self.ml.loglikelihoods(trainData, bPosterior=True)
            X, Y = learning_hmm.getHMMinducedFeatures(ll_logp, ll_post, Y_test_org)
            ## rospy.loginfo( "Features: "+ str(np.shape(X)) +" "+ str( np.shape(Y) ))
            ## rospy.loginfo( "Currrent method: " + self.classifier_method)           

            test_X = [] #copy.copy(self.ll_test_X) #need?
            test_Y = [] #copy.copy(self.ll_test_Y)
            for i in xrange(len(X)):

                if 'svm' in self.classifier_method or 'sgd' in self.classifier_method:
                    X_scaled = self.scaler.transform(X[i])
                else:
                    X_scaled = X[i]
                
                if i > len(X)-nFakeData-1: break
                test_X.append(X_scaled)
                test_Y.append(Y[i])
                self.ll_test_X.append(X_scaled)
                self.ll_test_Y.append(Y[i])
                self.ll_recent_test_X.append(X_scaled)
                self.ll_recent_test_Y.append(Y[i])

            ## test_X = list(test_X)
            ## test_Y = list(test_Y)
            ## test_X = list(self.ll_test_X)
            ## test_Y = list(self.ll_test_Y)

            
            ## Remove unseparable region and scaling it
            if self.classifier_method.find('svm')>=0:
                p_train_X, p_train_Y, _ = dm.flattenSample(X, Y, remove_fp=True)
                self.classifier.fit(self.X_train_org, self.Y_train_org)                
            elif self.classifier_method.find('sgd')>=0:
                #remove fp and flattening     
                p_train_X, p_train_Y, _ = getProcessSGDdata(test_X, test_Y)
                self.X_partial_train = np.vstack([ self.X_partial_train, p_train_X ])
                self.Y_partial_train = np.hstack([ self.Y_partial_train, p_train_Y ])
                ## self.X_train_org = np.vstack([ self.X_train_org, p_train_X ])
                ## self.Y_train_org = np.hstack([ self.Y_train_org, p_train_Y ])

                if update_flag or True:
                    nLength = len(p_train_X)/self.nTests
                    self.X_partial_train = np.delete(self.X_partial_train, np.s_[:nLength], 0)
                    self.Y_partial_train = np.delete(self.Y_partial_train, np.s_[:nLength], 0)
                    ## self.X_train_org = np.delete(self.X_train_org, np.s_[:nLength], 0)
                    ## self.Y_train_org = np.delete(self.Y_train_org, np.s_[:nLength], 0)
                    
                    ## sample_weights    = 1.0-np.exp( -0.0001* np.arange(0., len(self.X_partial_train), 1.0 ) )
                    sample_weights    = np.ones(len(self.X_partial_train))
                    ## sample_weights    = 1.0-np.exp( -0.0001* np.arange(0., len(self.X_train_org), 1.0 ) )
                    ## sample_weights    = 1.0-np.exp( -0.00001* np.arange(0., len(self.X_train_org), 1.0 ) )
                    ## sample_weights    = np.linspace(0.1, 1.0, len(self.X_train_org))
                    ## sample_weights    = np.ones(len(self.X_train_org))
                    ## if s_flag == 0 or f_flag == 0:
                    ##     sample_weights[-1] *= 1.0
                    ##     sample_weights[-2] *= 1.0
                    ## else:
                    ##     sample_weights[-1] *= 1.0
                    ## sample_weights[-2] = 20.0
                    p_train_W = sample_weights
                                    
                    rospy.loginfo("Start to Update!!! with %s data", str(len(test_X)) )
                    ## self.classifier.set_params( class_weight=1.0 )
                    alpha    = 1.0 #np.exp(-0.16*self.update_count)*0.8 + 0.2
                    nMaxIter = 1 #int(5.0*alpha)
                    ## alpha = np.exp(-0.16*self.update_count)*0.8 + 0.2
                    ## nMaxIter = int(5.0*alpha)
                    self.classifier = partial_fit(self.X_partial_train, self.Y_partial_train, p_train_W, \
                                                  self.classifier, \
                                                  test_X, test_Y, nMaxIter=nMaxIter, shuffle=True, alpha=alpha)
                    ## self.classifier = partial_fit(self.X_train_org, self.Y_train_org, p_train_W, \
                    ##                               self.classifier, \
                    ##                               test_X, test_Y, nMaxIter=nMaxIter, shuffle=True, alpha=alpha)
                    ## self.classifier.set_params( class_weight=self.w_positive )
            elif self.classifier_method.find('progress')>=0:

                max_rate      = 0.0 #0.1
                alpha         = np.exp(-0.16*self.update_count)*0.5 + 0.5
                update_weight = np.exp(-0.16*self.update_count)*float(self.nTrainData)/2.0 + 1.0
                ## update_weight = np.exp(-0.32*self.update_count)*0.7 + 0.3

                if user_feedback == "success":

                    l_mu   = list(self.classifier.ll_mu)
                    l_std  = list(self.classifier.ll_std)

                    ll_idx = []
                    for i in xrange(len(ll_logp)):
                        ll_idx.append( range(self.nLength-len(ll_logp[0]), self.nLength) )

                    # If true negative, update mean and var with new incoming data
                    # If false positive, update mean and var with new incoming data, lower ths mult
                    for i in xrange(len(l_mu)):
                        _, l_mu[i], l_std[i] = clf.update_time_cluster(i, ll_idx, ll_logp, ll_post,\
                                                                       self.classifier.g_mu_list[i],\
                                                                       self.classifier.g_sig, \
                                                                       l_mu[i], l_std[i],\
                                                                       self.nState,\
                                                                       self.nTrainData+len(self.update_list)-1,\
                                                                       update_weight=update_weight)
                    # update
                    self.classifier.ll_mu = l_mu
                    self.classifier.ll_std = l_std

                    print "Upppppppppppppppppppppppppp dateeeeeeeeeeeeeeeee!!!!!!!!!!!!!!!",\
                      update_weight, alpha, self.w_positive                    

                    if update_flag:
                        sensitivity = self.sensitivity_clf_to_GUI()
                        sensitivity -= max_rate*alpha
                        sensitivity = self.sensitivity_GUI_to_clf(sensitivity)
                        self.w_positive = sensitivity 
                        ## self.w_positive -= max_rate*alpha
                        self.classifier.set_params( ths_mult=self.w_positive )
                        rospy.set_param('progress_ths_mult', float(self.w_positive) )            
                        self.pubSensitivity()
                    
                else:                    
                    # If true positive, no update
                    if self.anomaly_flag: return
                
                    # If false negative, raise ths mult
                    if update_flag is False:
                        sensitivity = self.sensitivity_clf_to_GUI()
                        sensitivity += max_rate*alpha
                        sensitivity = self.sensitivity_GUI_to_clf(sensitivity)
                        self.w_positive = sensitivity 
                        ## self.w_positive += max_rate*alpha
                        self.classifier.set_params( ths_mult=self.w_positive )
                        rospy.set_param('progress_ths_mult', float(self.w_positive) )            
                        self.pubSensitivity()
                
                print "ths_mult: ", self.classifier.ths_mult, " internal weight: ", self.sensitivity_clf_to_GUI()
            else:
                rospy.loginfo( "Not available update method")

            # TODO: remove fake data
            ## nLength = len(p_train_X)/self.nTests
            ## self.X_train_org = np.delete(self.X_train_org, np.s_[:nLength], 0)
            ## self.Y_train_org = np.delete(self.Y_train_org, np.s_[:nLength], 0)
            ## self.X_train_org = np.vstack([ self.X_train_org, p_train_X[-nLength:] ])
            ## self.Y_train_org = np.hstack([ self.Y_train_org, p_train_Y[-nLength:] ])

            # ------------------------------------------------------------------------------------------
            ## print "################ Only recent data ####################"
            ## self.acc_part, _, _ = evaluation(list(self.ll_recent_test_X), list(self.ll_recent_test_Y), \
            ##                             self.classifier)
            ## self.acc_part, _, _ = evaluation(list(test_X)[:3], list(test_Y)[:3], \
            ##                        self.classifier)
            if self.bSim is False:
                self.acc_all, _, _ = evaluation(list(self.ll_test_X), list(self.ll_test_Y), self.classifier)
                self.cum_acc_list.append(self.acc_all)
                   
            # pub accuracy
            msg = FloatArray()
            msg.data = [self.acc_part, self.acc_all]
            self.accuracy_pub.publish(msg)                                   

            # update file list
            self.used_file_list += self.unused_fileList
            self.unused_fileList = []
            self.update_count += 1.0
            rospy.loginfo( "Update completed!!!")

    #-------------------------- General fuctions --------------------------

    def pubSensitivity(self):
        sensitivity = self.sensitivity_clf_to_GUI()
        if 'svm' in self.classifier_method or 'sgd' in self.classifier_method:        
            rospy.loginfo( "Current sensitivity is [0~1]: "+ str(sensitivity)+ \
                           ', internal weight is '+ str(self.classifier.class_weight) )
        else:
            rospy.loginfo( "Current sensitivity is [0~1]: "+ str(sensitivity)+ \
                           ', internal multiplier is '+ str(self.classifier.ths_mult) )

        self.sensitivity_pub.publish(sensitivity)                                   
            
                                      
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
            rospy.loginfo( 'unimodal_kinVel not implemented')

        # Unimodal feature - Force -------------------------------------------
        if 'unimodal_ftForce' in self.handFeatures or 'unimodal_ftForceZ' in self.handFeatures:
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
            rospy.loginfo( 'unimodal_visionChange may not be implemented properly')

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
            rospy.loginfo( "Not available")

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
            rospy.loginfo( "Not available" )

        # Crossmodal feature - vision relative angle --------------------------
        if 'crossmodal_subArtagEEAng' in self.handFeatures:                
            rospy.loginfo( "Not available" )

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
                                        init_param_dict = self.handFeatureParams)
                                        
        return data

    '''
    Reset parameters
    '''
    def reset(self):
        self.dataList = []
        self.enable_detector = False

    def run(self, freq=20):
        '''
        Run detector
        '''
        rospy.loginfo("Start to run anomaly detection: " + self.task_name)
        rate = rospy.Rate(freq) # 25Hz, nominally.
        while not rospy.is_shutdown():

            if len(self.dataList) >0 and self.viz:
                self.visualization()

            ## if self.cur_task is None: continue
            ## if not(self.cur_task == self.task_name): continue

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
                rospy.loginfo( "logp is None => anomaly" )
                self.action_interruption_pub.publish(self.task_name+'_anomaly')
                self.task_interruption_pub.publish(self.task_name+'_anomaly')
                self.soundHandle.play(2)
                self.enable_detector = False
                self.reset()
                continue

            post = post[cur_length-1]
            print "logp: ", logp, "  state: ", np.argmax(post)
            if np.argmax(post)==0 and logp < 0.0: continue
            if np.argmax(post)>self.param_dict['HMM']['nState']*0.9: continue

            if self.last_logp is None or self.last_post is None:
                self.last_logp = logp
                self.last_post = post
                continue
            else:                
                ## d_logp = logp - self.last_logp
                ## rospy.loginfo( np.shape(self.last_post), np.shape(post)
                ## d_post = hmm_util.symmetric_entropy(self.last_post, post)
                ## ll_classifier_test_X = [logp] + [d_logp/(d_post+1.0)] + post.tolist()
                ll_classifier_test_X = [logp] + self.last_post.tolist() + post.tolist()
                self.last_logp = logp
                self.last_post = post
                

            if 'svm' in self.classifier_method or 'sgd' in self.classifier_method:
                X = self.scaler.transform([ll_classifier_test_X])
            elif self.classifier_method == 'progress_time_cluster' or \
              self.classifier_method == 'fixed':
                X = ll_classifier_test_X
            else:
                rospy.loginfo( 'Invalid classifier method. Exiting.')
                exit()

            est_y = self.classifier.predict(X)
            if type(est_y) == list:
                est_y = est_y[-1]

            if est_y > 0.0:
                rospy.loginfo( '-'*15 +  'Anomaly has occured!' + '-'*15 )
                self.action_interruption_pub.publish(self.task_name+'_anomaly')
                self.task_interruption_pub.publish(self.task_name+'_anomaly')
                self.soundHandle.play(2)
                self.anomaly_flag    = True                
                self.enable_detector = False
                self.reset()

            rate.sleep()

        # save model and param
        self.save()
        rospy.loginfo( "Saved current parameters")

    def runSim(self, auto=True, subject_names=['ari']):
        '''
        Run detector with offline data
        '''

        checked_fileList = []
        self.unused_fileList = []

        ## fb = ut.get_keystroke('Hit a key to load a new file')
        ## sys.exit()


        print "############## CUMULATIVE / REF EVAL ###################"
        self.acc_all, _, _ = evaluation(list(self.ll_test_X), list(self.ll_test_Y), self.classifier)
        self.acc_ref, _, _ = self.evaluation_ref()
        self.update_list.append(0)
        self.cum_acc_list.append(self.acc_all)
        self.ref_acc_list.append(self.acc_ref)
        print "######################################################"


        for i in xrange(100):

            if rospy.is_shutdown(): break

            if auto:
                if i < len(self.eval_run_fileList):
                    ## import shutil
                    ## tgt_dir = os.path.join(self.raw_data_path, 'new_'+self.task_name)
                    ## shutil.copy2(self.eval_run_fileList[i], tgt_dir)
                    self.new_run_file = self.eval_run_fileList[i:i+1]
                    unused_fileList = self.new_run_file
                else:
                    print "no more file"
                    break
            else:            
                # load new file            
                fb = ut.get_keystroke('Hit a key to load a new file')
                if fb == 'z' or fb == 's': break
                                                          
                unused_fileList = util.getSubjectFileList(self.raw_data_path, \
                                                          self.subject_names, \
                                                          self.task_name, \
                                                          time_sort=True,\
                                                          no_split=True)                
                unused_fileList = [filename for filename in unused_fileList \
                                   if filename not in self.used_file_list]
                unused_fileList = [filename for filename in unused_fileList if filename not in checked_fileList]


            rospy.loginfo( "New file list ------------------------")
            for f in unused_fileList:
                rospy.loginfo( os.path.split(f)[1] )
            rospy.loginfo( "-----------------------------------------")

            if len(unused_fileList)>1:
                print "Unexpected addition of files"
                break

            for j in xrange(len(unused_fileList)):
                self.anomaly_flag = False
                if unused_fileList[j] in checked_fileList: continue
                if unused_fileList[j].find('success')>=0: label = -1
                else: label = 1
                    
                trainData = dm.getDataList([unused_fileList[j]], self.rf_center, self.rf_radius,\
                                           self.handFeatureParams,\
                                           downSampleSize = self.downSampleSize, \
                                           cut_data       = self.cut_data,\
                                           handFeatures   = self.handFeatures)
                                           
                # scaling and subtracting offset
                trainData = np.array(trainData)*self.scale
                trainData = self.applying_offset(trainData)
                
                
                ll_logp, ll_post = self.ml.loglikelihoods(trainData, bPosterior=True)
                X, Y = learning_hmm.getHMMinducedFeatures(ll_logp, ll_post, [label])
                X_test, Y_train_org, _ = dm.flattenSample(X, Y)
                
                if 'svm' in self.classifier_method or 'sgd' in self.classifier_method:
                    X_scaled = self.scaler.transform(X_test)
                else:
                    X_scaled = X_test
                y_est    = self.classifier.predict(X_scaled)

                for ii in xrange(len(y_est[self.startCheckIdx:])):
                    if y_est[ii] > 0.0:
                        rospy.loginfo('Anomaly has occured! idx=%s', str(ii) )
                        self.anomaly_flag    = True
                        break

                self.unused_fileList.append( unused_fileList[j] )
                # Quick feedback
                msg = StringArray()
                if label == 1:
                    msg.data = ['FALSE', 'TRUE', 'TRUE']
                    self.userfbCallback(msg)
                else:
                    msg.data = ['TRUE', 'FALSE', 'FALSE']
                    self.userfbCallback(msg)

                    
                true_label = rd.feedback_to_label(msg.data)
                update_flag  = False          
                if true_label == "success":
                    if self.anomaly_flag is True:
                        update_flag = True
                else:
                    if self.anomaly_flag is False:
                        update_flag = True

                print "############## CUMULATIVE / REF EVAL ###################"
                self.acc_all, _, _ = evaluation(list(self.ll_test_X), list(self.ll_test_Y), self.classifier)
                print "######################################################"
                if update_flag or i==0:
                    self.acc_ref, _, _ = self.evaluation_ref()
                    self.update_list.append(1)
                else:
                    self.update_list.append(0)
                self.cum_acc_list.append(self.acc_all)
                self.ref_acc_list.append(self.acc_ref)
                print self.cum_acc_list
                print self.ref_acc_list
                print self.update_list
                print self.w_positive, self.classifier.ths_mult
                print "######################################################"
                ## sys.exit()

                
                
                ## if (label ==1 and self.anomaly_flag is False) or \
                ##   (label ==-1 and self.anomaly_flag is True):
                ##     print "Before######################################33"
                ##     print y_est
                ##     print "Before######################################33"

                ##     print "Confirm######################################33"
                ##     y_est    = self.classifier.predict(X_scaled)
                ##     print y_est
                ##     print "Confirm######################################33"

                if auto is False:
                    fb =  ut.get_keystroke('Hit a key after providing user fb')
                    if fb == 'z' or fb == 's': break

            checked_fileList = [filename for filename in self.unused_fileList if filename not in self.used_file_list]
            print "===================================================================="
            # check anomaly
            # send feedback

        # save model and param
        if fb == 's':
            self.save()
            rospy.loginfo( "Saved current parameters")


    '''
    Save detector
    '''
    def save(self):
        pkg_path    = os.path.expanduser('~')+'/catkin_ws/src/hrl-assistive/hrl_anomaly_detection/params/'
        yaml_file   = os.path.join(pkg_path, 'anomaly_detection_'+self.task_name+'.yaml')
        param_namespace = '/'+self.task_name 
        os.system('rosparam dump '+yaml_file+' '+param_namespace)

        # Save scaler
        if 'svm' in self.classifier_method or 'sgd' in self.classifier_method:
            with open(self.scaler_model_file, 'wb') as f:
                pickle.dump(self.scaler, f)
                
        # Save classifier
        if self.bSim is False:
            print "save model"
            self.classifier.save_model(self.classifier_model_file)
        

    def applying_offset(self, data):

        # get offset
        refData = np.reshape( np.mean(self.normalTrainData[:,:,:self.startOffsetSize], axis=(1,2)), \
                              (self.nEmissionDim,1,1) ) # 4,1,1

        curData = np.reshape( np.mean(data[:,:,:self.startOffsetSize], axis=(1,2)), \
                              (self.nEmissionDim,1,1) ) # 4,1,1
        offsetData = refData - curData
                                  
        for i in xrange(self.nEmissionDim):
            data[i] = (np.array(data[i]) + offsetData[i][0][0]).tolist()

        return data


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

    def evaluation_ref(self):

        if self.eval_ref_fileList is None:
            self.eval_ref_fileList = util.getSubjectFileList(self.raw_data_path, \
                                                         self.param_dict['AD']['eval_target'], \
                                                         self.task_name, \
                                                         no_split=True)

        if self.eval_test_X is None:
            trainData = dm.getDataList(self.eval_ref_fileList, self.rf_center, self.rf_radius,\
                                       self.handFeatureParams,\
                                       downSampleSize = self.downSampleSize, \
                                       cut_data       = self.cut_data,\
                                       handFeatures   = self.handFeatures)

            # scaling and applying offset            
            trainData = np.array(trainData)*self.scale
            trainData = self.applying_offset(trainData)

            Y_test_org = []
            for f in self.eval_ref_fileList:
                if f.find("success")>=0:
                    Y_test_org.append(-1)
                elif f.find("failure")>=0:
                    Y_test_org.append(1)

            # update
            ## HMM
            ll_logp, ll_post = self.ml.loglikelihoods(trainData, bPosterior=True)
            X, Y = learning_hmm.getHMMinducedFeatures(ll_logp, ll_post, Y_test_org)

            self.eval_test_X = [] #copy.copy(self.ll_recent_test_X) #need?
            self.eval_test_Y = [] #copy.copy(self.ll_recent_test_Y)
            for i in xrange(len(X)):
                if 'svm' in self.classifier_method or 'sgd' in self.classifier_method:
                    X_scaled = self.scaler.transform(X[i])
                else:
                    X_scaled = X[i]
                    
                self.eval_test_X.append(X_scaled)
                self.eval_test_Y.append(Y[i])

        ## acc, nFP, nFN = evaluation(list(self.ll_test_X), list(self.ll_test_Y), self.classifier)        
        acc, nFP, nFN = evaluation(self.eval_test_X, self.eval_test_Y, self.classifier)
        return acc, nFP, nFN


    def sensitivity_clf_to_GUI(self):
        if self.exp_sensitivity:
            if 'svm' in self.classifier_method or 'sgd' in self.classifier_method:
                sensitivity = (np.log10(self.classifier.class_weight)-self.w_min)/(self.w_max-self.w_min)
            else:
                sensitivity = (np.log10(self.classifier.ths_mult)-self.w_min)/(self.w_max-self.w_min)
        else:
            if 'svm' in self.classifier_method or 'sgd' in self.classifier_method:
                sensitivity = (self.classifier.class_weight-self.w_min)/(self.w_max-self.w_min)
            else:
                sensitivity = (self.classifier.ths_mult-self.w_min)/(self.w_max-self.w_min)

        return sensitivity

    def sensitivity_GUI_to_clf(self, sensitivity_req):

        if sensitivity_req > 1.0: sensitivity_req = 1.0
        if sensitivity_req < 0.0: sensitivity_req = 0.0
        rospy.loginfo( "Requested sensitivity is [0~1]: %s", sensitivity_req)

        if self.exp_sensitivity:
            sensitivity_des = np.power(10, sensitivity_req*(self.w_max-self.w_min)+self.w_min)
        else:
            sensitivity_des = sensitivity_req*(self.w_max-self.w_min)+self.w_min                

        return sensitivity_des

    
###############################################################################

## def optFunc(x, clf, scaler, X, Y, verbose=False):

##     clf.dt.intercept_ = np.array([x])
##     ## acc, _, _ = evaluation(X, Y, clf, verbose=False)
##     ## return (100. - acc)/100.0
##     return evaluation_cost(X, Y, clf, verbose)
    

def evaluation(X, Y, clf, verbose=False):

    if X is None: return 0, 0, 0
    if len(X) == 0: return 0, 0, 0
    if len(X) is not len(Y):
        if len(np.shape(X)) == 2: X=[X]    
        if len(np.shape(Y)) == 1: Y=[Y]
    if len(Y) != len(X):
        print "wrong dim: ", np.shape(X), np.shape(Y)
        sys.exit()


    tp_l = []
    fp_l = []
    fn_l = []
    tn_l = []

    for i in xrange(len(X)):
   
        anomaly = False
        est_y   = clf.predict(X[i])
        for j in xrange(len(est_y)):

            if j < 4: continue
            if est_y[j] > 0:
                anomaly = True
                break

        if anomaly is True and  Y[i][0] > 0: tp_l += [1]
        if anomaly is True and  Y[i][0] < 0: fp_l += [1]
        if anomaly is False and  Y[i][0] > 0: fn_l += [1]
        if anomaly is False and  Y[i][0] < 0: tn_l += [1]

    try:
        tpr = float(np.sum(tp_l)) / float(np.sum(tp_l)+np.sum(fn_l)) * 100.0
        fpr = float(np.sum(fp_l)) / float(np.sum(fp_l)+np.sum(tn_l)) * 100.0
    except:
        print "tp, fp, tn, fn: ", tp_l, fp_l, tn_l, fn_l

    if np.sum(tp_l+fn_l+fp_l+tn_l) == 0: return
    acc = float(np.sum(tp_l+tn_l)) / float(np.sum(tp_l+fn_l+fp_l+tn_l)) * 100.0
    print "tp=",np.sum(tp_l), " fn=",np.sum(fn_l), " fp=",np.sum(fp_l), " tn=",np.sum(tn_l), " ACC: ",  acc
    return acc, np.sum(fp_l), np.sum(fn_l)


def evaluation_cost(X, Y, clf, verbose=False):

    ## if X is None: return 0, 0, 0
    ## if len(X) is not len(Y):
    ##     if len(np.shape(X)) == 2: X=[X]    
    ##     if len(np.shape(Y)) == 1: Y=[Y]
    ## if len(Y) != len(X):
    ##     print "wrong dim: ", np.shape(X), np.shape(Y)
    ##     sys.exit()

    cost = []
    if clf.method.find('svm')>=0 or clf.method.find('sgd')>=0:
        for i in xrange(len(X)):
            est_y   = clf.predict(X[i])
            est_p   = clf.decision_function(X[i])

            anomaly = False
            for j in xrange(len(est_y)):

                if j < 4: continue
                if est_y[j] > 0:
                    if Y[i][0]<0: #fp
                        cost.append(abs(est_p[j]))
                        ## if verbose: print "fp: ", cost, i,j, ", intercept: ", clf.dt.intercept_
                    anomaly = True
                    break

            if anomaly is False and Y[i][0] > 0:
                cost.append(abs(np.mean(est_p)))
                ## if verbose: print "fn: ", cost, i,j, ", intercept: ", clf.dt.intercept_
                
    else:
        print "Not available method"
        sys.exit()

    if verbose: print "cost: ", np.sum(cost) #,  ", intercept: ", clf.dt.intercept_
    if len(cost) == 0: return 0.0
    return np.sum(cost)


def partial_fit(X, Y, W, clf, XX, YY, nMaxIter=100, shuffle=True, alpha=1.0 ):

    last_Coef  = copy.deepcopy(clf.dt.coef_)
    last_dCoef = 0.0
    last_p     = 0.0
    last_cost  = 100.0
    if nMaxIter == 0: nMaxIter = 1

    for i in xrange(nMaxIter):

        clf.partial_fit(X,Y, classes=[-1,1],n_iter=int(20.*alpha), sample_weight=W, shuffle=shuffle)
        cost = evaluation_cost(XX, YY, clf)
        print "cost: ", cost, "dCost: ", cost-last_cost
        if cost < 0.005: break
        if abs(cost-last_cost) < 0.001: break
        if cost-last_cost > 0.005: break
        last_cost = cost
        
        ## dCoef = np.linalg.norm(last_Coef-copy.deepcopy(clf.dt.coef_))        
        ## est_p = np.mean(abs( clf.decision_function(X)) )

        ## if clf.predict()
        
        ## print "dCoef: ", dCoef, " dp", est_p-last_p
        ## ## if est_p-last_p < 0.0:
        ## ##     break
        ## if dCoef < 0.1 and dCoef - last_dCoef < 0 and est_p-last_p < 0.0005:
        ##     break
        
        ## last_dCoef = dCoef
        ## last_Coef  = copy.deepcopy(clf.dt.coef_)
        ## last_p     = est_p

    ## new_X = scaler.transform(XX[0])
    ## print clf.predict(new_X)
    return clf

## def minimize(f, x0, clf, scaler, X, Y, nMaxIter=1000):
##     x     = x0
##     x_pre = x0
##     r_pre = 1.0
##     alpha = 1.0
##     dx = 1.0
##     dr = 1.0
##     for i in xrange(nMaxIter):

##         if r_pre == 0.0:
##             return x            
##         elif dr == 0.0:
##             if r_pre == 1.0:
##                 x = np.random.normal(x,0.3,1)[0]
##             else:
                
##         else:
##             learning_rate = 1.0/(alpha*( float(i)+1.0 ) )
##             x = x - learning_rate * dr/dx
            
##         r = optFunc(x, clf, scaler, X, Y)
##         dr = r-r_pre
##         dx = x-x_pre
##         x_pre = x
##         r_pre = r
        

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
    p.add_option('--simulation', '--sim', action='store_true', dest='bSim',
                 default=False, help='Enable a simulation mode.')
    

    p.add_option('--dataRenew', '--dr', action='store_true', dest='bDataRenew',
                 default=False, help='Renew pickle files.')
    p.add_option('--hmmRenew', '--hr', action='store_true', dest='bHMMRenew',
                 default=False, help='Renew HMM parameters.')
    p.add_option('--clfRenew', '--cr', action='store_true', dest='bCLFRenew',
                 default=False, help='Renew classifier.')
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
            rospy.loginfo( "Not supported task")
            sys.exit()
    else:
        from hrl_anomaly_detection.ICRA2017_params import *

        raw_data_path, save_data_path, param_dict = getParams(opt.task, False, \
                                                              False, False, opt.dim,\
                                                              rf_center, local_range, \
                                                              nPoints=10)
        
        if opt.task == 'scooping':
            ## subject_names = ['test'] 
            ## subject_names = ['Zack'] 
            subject_names = ['park', 'new'] 
            test_subject  = ['park'] # sim only
            
            check_method      = opt.method
            save_data_path    = os.path.expanduser('~')+'/hrl_file_server/dpark_data/anomaly/ICRA2017/'+\
              opt.task+'_demo_data'
            param_dict['SVM'] = {'renew': False, 'w_negative': 4.0, 'gamma': 0.04, 'cost': 4.6, \
                                 'class_weight': 1.5e-2, 'logp_offset': 0, 'ths_mult': -2.0,\
                                 'sgd_gamma':0.32, 'sgd_w_negative':2.5,}

            param_dict['data_param']['nNormalFold']   = 1
            param_dict['data_param']['nAbnormalFold'] = 1
            param_dict['AD']['eval_target'] = ['ref']

        elif opt.task == 'feeding':
            ## subject_names = ['test'] 
            subject_names = ['zack', 'hkim', 'ari'] #, 'zack'
            test_subject  = ['jina'] # sim only
            
            check_method      = opt.method
            save_data_path    = os.path.expanduser('~')+'/hrl_file_server/dpark_data/anomaly/ICRA2017/'+\
              opt.task+'_demo_data'
            param_dict['SVM'] = {'renew': False, 'w_negative': 4.0, 'gamma': 0.04, 'cost': 4.6, \
                                 'class_weight': 1.5e-2, 'logp_offset': 30, 'ths_mult': -1.0,\
                                 'sgd_gamma':0.32, 'sgd_w_negative':2.5}

            param_dict['data_param']['nNormalFold']   = 1
            param_dict['data_param']['nAbnormalFold'] = 1
            param_dict['AD']['eval_target'] = ['ref']
        else:
            rospy.loginfo( "Not supported task")
            sys.exit()


    ad = anomaly_detector(subject_names, opt.task, check_method, raw_data_path, save_data_path, \
                          param_dict, data_renew=opt.bDataRenew, hmm_renew=opt.bHMMRenew, \
                          clf_renew=opt.bCLFRenew, \
                          viz=opt.bViz, auto_update=opt.bAutoUpdate,\
                          debug=opt.bDebug, sim=opt.bSim )
    if opt.bSim is False:
        ad.run()
    else:
        ad.runSim(subject_names=test_subject)












    ## def updateCallback(self, msg):
    ##     fileNames = msg.data

    ##     if len(fileNames) == 0 or os.path.isfile(fileName) is False:
    ##         rospy.loginfo( "Warning>> there is no recorded file"
    ##         return StringArray_NoneResponse()
              
    ##     rospy.loginfo( "Start to update detector using ", fileName

    ##     # Get label
    ##     for f in fileNames:
    ##         if 'success' in f: self.Y_test_org.append(0)
    ##         else: Y_test_org.append(1)

    ##     # Preprocessing
    ##     trainData,_ = dm.getDataList(fileNames, self.rf_center, self.local_range,\
    ##                                  self.handFeatureParams,\
    ##                                  downSampleSize = self.downSampleSize, \
    ##                                  cut_data       = self.cut_data,\
    ##                                  handFeatures   = self.handFeatures)
    ##     rospy.loginfo( "Preprocessing: ", np.shape(trainData), np.shape(Y_test_org)

    ##     ## HMM
    ##     ll_logp, ll_post = self.ml.loglikelihoods(trainData, bPosterior=True)
    ##     X, Y = learning_hmm.getHMMinducedFeatures(ll_logp, ll_post, Y_test_org, c=1.0, add_delta_logp=self.add_logp_d)
    ##     rospy.loginfo( "Features: ", np.shape(X), np.shape(Y)

    ##     ## Remove unseparable region and scaling it
    ##     X_train_org, Y_train_org, _ = dm.flattenSample(X, Y, remove_fp=True)
    ##     if 'svm' in self.classifier_method:
    ##         self.X_train_org = np.vstack([ self.X_train_org, self.scaler.transform(X_train_org) ])
    ##     elif 'sgd' in self.classifier_method:
    ##         self.X_train_org = self.scaler.transform(X_train_org)
    ##     else:
    ##         rospy.loginfo( "Not available method"
    ##         sys.exit()

    ##     # Run SGD? or SVM?
    ##     if self.classifier_method.find('svm') >= 0:
    ##         self.classifier.fit(self.X_train_org, self.Y_test_org)
    ##     elif self.classifier_method.find('svm') >= 0:
    ##         rospy.loginfo( "Not available"
    ##         return StringArray_NoneResponse()
    ##         self.classifier.partial_fit(self.X_train_org, self.Y_test_org, classes=[-1,1])            
    ##     else:
    ##         rospy.loginfo( "Not available update method"
            
    ##     return StringArray_NoneResponse()



                
            ## # adjust the sensitivity until classify the new data correctly.
            ## if (self.classifier_method.find('sgd')>=0 or self.classifier_method.find('svm')>=0) and False:

            ##     import scipy
            ##     if self.classifier_method.find('sgd')>=0:
            ##         print "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            ##         print "Previous intercept: ", self.classifier.dt.intercept_ 
            ##         res = scipy.optimize.fmin(optFunc, x0=float(self.classifier.dt.intercept_), \
            ##                                   args=(self.classifier, self.scaler,\
            ##                                         list(self.ll_recent_test_X), \
            ##                                         list(self.ll_recent_test_Y),\
            ##                                         True),\
            ##                                         ## epsilon=0.1,\
            ##                                         xtol=0.05,\
            ##                                         ftol=0.0005,\
            ##                                         maxiter=10 )
            ##         sensitivity_des = float(res)
            ##         ## res = scipy.optimize.minimize(optFunc, x0=float(self.classifier.dt.intercept_), \
            ##         ##                               args=(self.classifier, self.scaler,\
            ##         ##                                     ## list(self.ll_test_X),\
            ##         ##                                     ## list(self.ll_test_Y),\
            ##         ##                                     list(self.ll_recent_test_X), \
            ##         ##                                     list(self.ll_recent_test_Y),\
            ##         ##                                     True),\
            ##         ##                                     jac=False, \
            ##         ##                                     tol=0.001,\
            ##         ##                                     options={'maxiter': 20, } )
            ##         ## sensitivity_des = float(res.x)
            ##         print res
            ##         if self.w_max < sensitivity_des: self.w_max = sensitivity_des
            ##         elif self.w_min < sensitivity_des: self.w_min = sensitivity_des
            ##         self.classifier.dt.intercept_ = np.array([sensitivity_des])
            ##         print "Current intercept: ", self.classifier.dt.intercept_ 
            ##         rospy.set_param(self.classifier_method+'_intercept', float(sensitivity_des))
            ##         print "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            ##     else:
            ##         sys.exit()
                
