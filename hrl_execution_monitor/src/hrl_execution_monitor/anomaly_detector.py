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
import random, os, sys, threading, copy
from joblib import Parallel, delayed
import datetime
import numpy as np
import pickle

# util
from sound_play.libsoundplay import SoundClient

# Utility
import hrl_lib.util as ut
from hrl_execution_monitor import anomaly_detector_util as adu
from hrl_execution_monitor import util as autil

# msg
from hrl_anomaly_detection.msg import MultiModality
from std_msgs.msg import String, Float64, Bool
from hrl_srvs.srv import Bool_None, Bool_NoneResponse, StringArray_None
from hrl_msgs.msg import FloatArray, StringArray

from matplotlib import pyplot as plt


QUEUE_SIZE = 10

class anomaly_detector:
    def __init__(self, task_name, method, detector_id, save_data_path,\
                 param_dict, debug=False, sim=False, sim_subject_names=None,
                 viz=False):
        rospy.loginfo('Initializing anomaly detector')

        self.task_name       = task_name.lower()
        self.method          = method
        self.id              = detector_id
        self.save_data_path  = save_data_path
        self.debug           = debug
        self.viz             = viz

        # Important containers
        self.enable_detector = False
        self.dataList        = []
        self.logpDataList    = []
        ## self.anomaly_flag    = False
        self.init_msg        = None
        self.last_msg        = None
        self.last_data       = None
        self.refData         = None

        # Params
        self.param_dict      = param_dict        
        self.f_param_dict    = None
        self.startOffsetSize = 4
        self.startCheckIdx   = 10
        self.yaml_file       = '/home/dpark/catkin_ws/src/hrl-assistive/hrl_execution_monitor/params/anomaly_detection_'+self.task_name+'.yaml'

        # HMM, Classifier
        self.nEmissionDim = len(param_dict['data_param']['handFeatures'][detector_id])
        self.ml           = None
        self.classifier   = None
        self.soundHandle  = SoundClient()

        # sim  ---------------------------------------------
        self.bSim         = sim
        if self.bSim: self.cur_task = self.task_name
        else:         self.cur_task = None
        self.sim_subject_names = sim_subject_names
        self.t1 = datetime.datetime.now()
        # -------------------------------------------------

        # Comms
        self.lock = threading.Lock()        
        self.initParams()
        self.initDetector()
        self.initComms()

        # -------------------------------------------------
        self.viz = viz
        self.count = 0
        if viz:
            rospy.loginfo( "Visualization enabled!!!")
            self.fig = plt.figure()
            plt.ion()
            plt.show()            
            self.figure_flag = False
        # -------------------------------------------------
        
        self.reset()
        rospy.loginfo( "==========================================================")
        rospy.loginfo( "Detector initialized!! : %s", self.task_name)
        rospy.loginfo( "==========================================================")

    '''
    Load feature list
    '''
    def initParams(self):

        # Features and parameters
        self.handFeatures = self.param_dict['data_param']['handFeatures'][self.id]
        self.nState = self.param_dict['HMM']['nState']
        self.scale  = self.param_dict['HMM']['scale'][self.id]

        self.nDetector = rospy.get_param('nDetector')
        self.w_positive = rospy.get_param(self.method+str(self.id)+'_ths_mult')
        self.w_max = self.param_dict['ROC'][self.method+str(self.id)+'_param_range'][-1]
        self.w_min = self.param_dict['ROC'][self.method+str(self.id)+'_param_range'][0]
        self.exp_sensitivity = False

        # Weight ordering
        if self.w_min > self.w_max:
            self.w_max, self.w_min = self.w_min, self.w_max

        if self.w_positive > self.w_max: self.w_positive = self.w_max
        elif self.w_positive < self.w_min: self.w_positive = self.w_min
        rospy.set_param(self.method+str(self.id)+'_ths_mult', float(self.w_positive))

        # we use logarlism for the sensitivity
        if self.exp_sensitivity:
            self.w_max = np.log10(self.w_max)
            self.w_min = np.log10(self.w_min)


    def initComms(self):
        # Publisher
        self.action_interruption_pub = rospy.Publisher('/manipulation_task/InterruptAction', String,
                                                       queue_size=QUEUE_SIZE)
        self.task_interruption_pub   = rospy.Publisher("/manipulation_task/emergency", String,
                                                       queue_size=QUEUE_SIZE)
        self.sensitivity_pub         = rospy.Publisher("manipulation_task/ad_sensitivity_state", \
                                                       Float64, queue_size=QUEUE_SIZE, latch=True)

        # temp
        self.isolation_info_pub = rospy.Publisher("/manipulation_task/anomaly_type", String,
                                                  queue_size=QUEUE_SIZE)
        self.hmm_input_pub           = rospy.Publisher("manipulation_task/hmm_input"+str(self.id),
                                                       FloatArray, queue_size=QUEUE_SIZE)


        # Subscriber # TODO: topic should include task name prefix?
        rospy.Subscriber('/hrl_manipulation_task/raw_data', MultiModality, self.rawDataCallback)
        rospy.Subscriber('/manipulation_task/status', String, self.statusCallback)
        rospy.Subscriber('manipulation_task/ad_sensitivity_request', Float64, self.sensitivityCallback)

        ## rospy.Subscriber('/manipulation_task/proceed', String, self.debugCallback)

        # Service
        self.detection_service = rospy.Service('anomaly_detector'+str(self.id)+'_enable',
                                               Bool_None, self.enablerCallback)
        # info for GUI
        self.pubSensitivity()


    def initDetector(self):
        ''' init detector ''' 
        rospy.loginfo( "Initializing a detector (%s) for %s", self.method, self.task_name)
        if self.nDetector == 0: return
        
        self.f_param_dict, self.ml, self.classifier = adu.get_detector_modules(self.save_data_path,
                                                                               self.task_name,
                                                                               self.param_dict,
                                                                               self.id)
        normalTrainData = self.f_param_dict['successData'] * self.scale
        self.refData = np.reshape( np.mean(normalTrainData[:,:,:self.startOffsetSize], axis=(1,2)), \
                                  (self.nEmissionDim,1,1) ) # 4,1,1
        self.classifier.set_params( ths_mult=self.w_positive )
        self.classifier.set_params( logp_offset=self.param_dict['SVM']['logp_offset'] )
        self.classifier.set_params( hmmgp_logp_offset=self.param_dict['SVM']['logp_offset'] )
        
        if self.viz or True:
            self.mean_train = np.mean(normalTrainData/self.scale, axis=1)
            self.std_train = np.std(normalTrainData/self.scale, axis=1)
            ## self.max_train = np.amax(normalTrainData, axis=(1,2))
            ## self.min_train = np.amin(normalTrainData, axis=(1,2))
            ## print self.max_train
            ## print self.min_train

            

    #-------------------------- Communication fuctions --------------------------
    def enablerCallback(self, msg):

        if msg.data is True:
            rospy.loginfo("%s anomaly detector %s enabled", self.task_name, str(self.id))
            self.enable_detector = True
            ## self.anomaly_flag    = False            
            self.pubSensitivity()
            if self.viz:
                self.fig.clf()
                self.figure_flag=False

        else:
            rospy.loginfo("%s anomaly detector %s disabled", self.task_name, str(self.id))
            self.enable_detector = False
            self.reset() #TODO: may be it should be removed

        return Bool_NoneResponse()


    def rawDataCallback(self, msg):
        '''
        Subscribe raw data
        '''
        if msg is None: return
        if self.f_param_dict is None or self.refData is None: return
        if self.cur_task is None: return
        if self.cur_task.find(self.task_name) < 0: return

        # If detector is disbled, detector does not fill out the dataList.
        if self.enable_detector is False or self.init_msg is None:
            self.init_msg = copy.copy(msg)
            self.last_msg = copy.copy(msg)
            return

        self.lock.acquire()
        ########################################################################
        # Run your custom feature extraction function
        dataSample, scaled_dataSample = autil.extract_feature(msg,
                                                              self.init_msg,
                                                              self.last_msg,
                                                              self.last_data,
                                                              self.handFeatures,
                                                              self.f_param_dict['feature_params'],
                                                              count=len(self.dataList[0][0])
                                                              if len(self.dataList)>0 else 0)
        scaled_dataSample = np.array(scaled_dataSample)*self.scale
        self.last_msg  = copy.copy(msg)
        self.last_data = copy.copy(dataSample)
        ########################################################################

        # Subtract white noise by measuring offset using scaled data
        if self.dataList == [] or len(self.dataList[0][0]) < self.startOffsetSize:
            self.offsetData = np.zeros(np.shape(scaled_dataSample))            
        elif len(self.dataList[0][0]) == self.startOffsetSize:
            curData = np.reshape( np.mean(self.dataList, axis=(1,2)), (self.nEmissionDim,1,1) ) # 4,1,1
            self.offsetData = self.refData - curData
            for i in xrange(self.nEmissionDim):
                self.dataList[i] = (np.array(self.dataList[i]) + self.offsetData[i][0][0]).tolist()
            self.offsetData = self.offsetData[:,0,0]
            
        scaled_dataSample += self.offsetData

        
        if len(self.dataList) == 0:
            self.dataList = np.reshape(scaled_dataSample, (self.nEmissionDim,1,1) ).tolist() # = newData.tolist()
        else:                
            # dim x sample x length
            for i in xrange(self.nEmissionDim):
                self.dataList[i][0] = self.dataList[i][0] + [scaled_dataSample[i]]

        
        self.lock.release()

        ## self.t2 = datetime.datetime.now()
        ## print "time: ", self.t2 - self.t1
        ## self.t1 = self.t2


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
        rospy.loginfo( "Started to update the classifier!")

        self.w_positive = sensitivity_des
        self.classifier.set_params( ths_mult=self.w_positive )
        rospy.set_param(self.method+str(self.id)+'_ths_mult', float(sensitivity_des))            

        rospy.loginfo( "Classifier is updated!")
        self.pubSensitivity()


    def debugCallback(self, msg):
        if msg.data.find("Set: Feeding 3, Feeding 4, retrieving")>=0:            
            rospy.loginfo("%s anomaly detector %s enabled", self.task_name, str(self.id))
            self.enable_detector = True
            ## self.anomaly_flag    = False            
            self.pubSensitivity()                    
        else:
            rospy.loginfo("%s anomaly detector %s disabled", self.task_name, str(self.id))
            self.enable_detector = False
            self.reset() #TODO: may be it should be removed

    #-------------------------- General fuctions --------------------------
    def pubSensitivity(self):
        if self.classifier is None: return
        sensitivity = self.sensitivity_clf_to_GUI()
        rospy.loginfo( "Current sensitivity is [0~1]: "+ str(sensitivity)+ \
                       ', internal multiplier is '+ str(self.classifier.ths_mult) )
        self.sensitivity_pub.publish(sensitivity)                                   


    def set_anomaly_alarm(self, dataList):
        rospy.loginfo( '-'*15 +  'Anomaly has occured!' + '-'*15 )
        self.action_interruption_pub.publish(self.task_name+'_anomaly')
        self.task_interruption_pub.publish(self.task_name+'_anomaly')
        ## self.hmm_input_pub.publish(dataList)
        self.soundHandle.play(1)
        ## self.anomaly_flag    = True                
        self.enable_detector = False

        rospy.sleep(10.0)
        self.isolation_info_pub.publish(" XXXXXXXXXXX ")
        
        self.reset()
        

        
    def reset(self):
        ''' Reset parameters '''
        self.lock.acquire()
        self.dataList = []
        self.logpDataList = []
        self.enable_detector = False
        self.lock.release()


    def run(self, freq=20):
        ''' Run detector '''
        rospy.loginfo("Start to run anomaly detection: " + self.task_name)
        rate = rospy.Rate(freq) # 20Hz, nominally.
        while not rospy.is_shutdown():

            if self.enable_detector is False or self.classifier is None: 
                self.dataList = []
                self.logpDataList = []
                rate.sleep()                
                continue

            #-----------------------------------------------------------------------
            if len(self.dataList) == 0 or len(self.dataList[0][0]) <= self.startCheckIdx:
                rate.sleep()                
                continue
            self.lock.acquire()
            dataList = copy.deepcopy(self.dataList)
            self.lock.release()
            cur_length = len(dataList[0][0])

            ## # moving avg filter
            ## for i in xrange(self.nEmissionDim):
            ##     x = autil.running_mean(dataList[i][0], 4)
            ##     dataList[i][0] = x.tolist() #[x[0]]*4 + x.tolist()

            logp, post = self.ml.loglikelihood(dataList, bPosterior=True)
                
            #-----------------------------------------------------------------------
            if logp is None:
                rospy.loginfo( "logp is None => anomaly" )
                self.set_anomaly_alarm(dataList)
                continue

            post = post[cur_length-1]

            if np.argmax(post)==0 and logp < 0.0: continue
            ## if np.argmax(post)>self.param_dict['HMM']['nState']*0.9: continue

            ll_classifier_test_X = [logp] + post.tolist()                
            if 'svm' in self.method or 'sgd' in self.method:
                X = self.scaler.transform([ll_classifier_test_X])
            else:
                X = ll_classifier_test_X

            # anomal classification
            err, y_pred, sigma = self.classifier.predict(X, debug=True)
            if self.viz:
                self.logpDataList.append([len(dataList[0][0]), logp, y_pred, y_pred-sigma ])
                ## self.viz_raw_input(self.dataList)
                self.viz_decision_boundary(np.array(dataList)/self.scale, self.logpDataList)
            
            print len(dataList[0][0]), " : logp: ", logp, "  state: ", np.argmax(post), " y_pred: ", y_pred, sigma, self.id #err[-1]+logp, self.id            
        
            if type(err) == list: err = err[-1]
            if err > 0.0: self.set_anomaly_alarm(dataList)
            rate.sleep()

        # save model and params
        self.save_config()
        rospy.loginfo( "Saved current parameters")


    def save_config(self):
        ''' Save detector '''
        # name matches with detector parameter file name.
        param_namespace = '/'+self.task_name 
        os.system('rosparam dump '+self.yaml_file+' '+param_namespace)
        

    def sensitivity_clf_to_GUI(self):
        if self.exp_sensitivity:
            sensitivity = (np.log10(self.classifier.ths_mult)-self.w_min)/(self.w_max-self.w_min)
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

    
    #-------------------------- ETC --------------------------

    def viz_raw_input(self, x):

        if self.figure_flag is False:
            self.fig = plt.figure()
            plt.ion()
            plt.show()
            print 
        ## else:            
        ##     del self.ax.collections[:]

        for i in xrange(self.nEmissionDim):
            self.ax = self.fig.add_subplot(100*self.nEmissionDim+10+i+1)
            self.ax.plot(np.array(x)[i,0], 'r-')
            self.ax.plot(self.mean_train[i], 'b-', lw=3.0)

        plt.axis('tight')
        plt.draw()
        self.figure_flag = True
        ## ut.get_keystroke('Hit a key')

    def viz_decision_boundary(self, x, logpDataList):

        if self.figure_flag is False:
            ## self.fig = plt.figure()
            ## plt.ion()
            ## plt.show()
            print 
        else:            
            del self.ax.collections[:]

        idx_lst   = np.array(logpDataList)[:,0]
        logp      = np.array(logpDataList)[:,1]
        logp_pred = np.array(logpDataList)[:,2]
        logp_sig  = np.array(logpDataList)[:,3]

        for i in xrange(self.nEmissionDim+1):
            self.ax = self.fig.add_subplot(100*(self.nEmissionDim+1)+10+i+1)

            if i < self.nEmissionDim:
                self.ax.plot(np.array(x)[i,0], 'r-')
                self.ax.plot(self.mean_train[i], 'b-', lw=3.0)
                self.ax.plot(self.mean_train[i]+self.std_train[i], 'b--', lw=1.0)
                self.ax.plot(self.mean_train[i]-self.std_train[i], 'b--', lw=1.0)
            else:
                self.ax.plot(idx_lst, logp, 'r-')
                self.ax.plot(idx_lst, logp_pred, 'b-', lw=3.0)
                self.ax.plot(idx_lst, logp_sig, 'b--', lw=1.0)
                self.ax.set_xlim([0,140])

        plt.axis('tight')
        plt.draw()
        self.figure_flag = True
        ## ut.get_keystroke('Hit a key')



if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    p.add_option('--task', action='store', dest='task', type='string', default='feeding',
                 help='type the desired task name')
    p.add_option('--method', '--m', action='store', dest='method', type='string', default='progress',
                 help='type the method name')
    p.add_option('--id', action='store', dest='id', type=int, default=0,
                 help='type the detector id')
    p.add_option('--debug', '--d', action='store_true', dest='bDebug',
                 default=False, help='Enable debugging mode.')
    
    p.add_option('--viz', action='store_true', dest='bViz',
                 default=False, help='Visualize data.')
    
    opt, args = p.parse_args()
    rospy.init_node(opt.task+'_detector')

    if True:
        from hrl_execution_monitor.params.IROS2017_params import *
        # IROS2017
        subject_names = ['s2', 's3','s4','s5', 's6','s7','s8', 's9']
        raw_data_path, save_data_path, param_dict = getParams(opt.task)
        ## save_data_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/IROS2017/'+opt.task+'_demo'
        save_data_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/IROS2017/'+opt.task+'_demo1'
        
    else:
        rospy.loginfo( "Not supported task")
        sys.exit()


    ad = anomaly_detector(opt.task, opt.method, opt.id, save_data_path, \
                          param_dict, debug=opt.bDebug, viz=False)
    ad.run()








