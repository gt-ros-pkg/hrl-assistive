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
from std_msgs.msg import String, Float64
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
        self.anomaly_flag    = False
        self.init_msg        = None
        self.last_msg        = None
        self.last_data       = None
        self.refData         = None

        # Params
        self.param_dict      = param_dict        
        self.f_param_dict    = None
        self.startOffsetSize = 4
        self.startCheckIdx   = 10

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
        self.initComms()
        self.initDetector()

        # -------------------------------------------------
        self.viz = viz
        if viz:
            rospy.loginfo( "Visualization enabled!!!")
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

        # Subscriber # TODO: topic should include task name prefix?
        rospy.Subscriber('/hrl_manipulation_task/raw_data', MultiModality, self.rawDataCallback)
        rospy.Subscriber('/manipulation_task/status', String, self.statusCallback)
        rospy.Subscriber('manipulation_task/ad_sensitivity_request', Float64, self.sensitivityCallback)

        # Service
        self.detection_service = rospy.Service('anomaly_detector'+str(self.id)+'_enable',
                                               Bool_None, self.enablerCallback)


    def initDetector(self):
        ''' init detector ''' 
        rospy.loginfo( "Initializing a detector with %s of %s", self.method, self.task_name)
        
        self.f_param_dict, self.ml, self.classifier = adu.get_detector_modules(self.save_data_path,
                                                                               self.task_name,
                                                                               self.param_dict,
                                                                               self.id)
        normalTrainData = self.f_param_dict['successData'] * self.scale
        self.refData = np.reshape( np.mean(normalTrainData[:,:,:self.startOffsetSize], axis=(1,2)), \
                                  (self.nEmissionDim,1,1) ) # 4,1,1

        # info for GUI
        self.pubSensitivity()

        if self.viz:
            self.mean_train = np.mean(normalTrainData, axis=1)
        

    #-------------------------- Communication fuctions --------------------------
    def enablerCallback(self, msg):

        if msg.data is True:
            rospy.loginfo("%s anomaly detector %s enabled", self.task_name, str(self.id))
            self.enable_detector = True
            self.anomaly_flag    = False            
            self.pubSensitivity()                    
        else:
            rospy.loginfo("%s anomaly detector %s disabled", self.task_name, str(self.id))
            self.enable_detector = False
            self.reset() #TODO: may be it should be removed

        return Bool_NoneResponse()


    def rawDataCallback(self, msg):
        '''
        Subscribe raw data
        '''
        if self.f_param_dict is None or self.refData is None: return
        if self.cur_task is None: return
        if self.cur_task.find(self.task_name) < 0: return

        # If detector is disbled, detector does not fill out the dataList.
        if self.enable_detector is False:
            self.init_msg = copy.copy(msg)
            return

        self.lock.acquire()
        ########################################################################
        # Run your custom feature extraction function
        newData = np.array( autil.extract_feature(msg,
                                                  self.init_msg,
                                                  self.last_msg,
                                                  self.last_data,
                                                  self.handFeatures,
                                                  self.f_param_dict['feature_params'])
                                                  )*self.scale
        self.last_msg  = copy.copy(msg)
        self.last_data = copy.copy(newData)
        ########################################################################

        # Subtract white noise by measuring offset
        if self.dataList == [] or len(self.dataList[0][0]) < self.startOffsetSize:
            self.offsetData = np.zeros(np.shape(newData))            
        elif len(self.dataList[0][0]) == self.startOffsetSize:
            curData = np.reshape( np.mean(self.dataList, axis=(1,2)), (self.nEmissionDim,1,1) ) # 4,1,1
            self.offsetData = self.refData - curData
            for i in xrange(self.nEmissionDim):
                self.dataList[i] = (np.array(self.dataList[i]) + self.offsetData[i][0][0]).tolist()

        newData = newData+ self.offsetData
        
        if len(self.dataList) == 0:
            self.dataList = newData.tolist()
        else:                
            # dim x sample x length
            for i in xrange(self.nEmissionDim):
                self.dataList[i][0] = self.dataList[i][0] + [newData[i][0][0]]
                       
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
        rospy.set_param(self.method+'_ths_mult', float(sensitivity_des))            

        rospy.loginfo( "Classifier is updated!")
        self.pubSensitivity()
            

    #-------------------------- General fuctions --------------------------
    def pubSensitivity(self):
        sensitivity = self.sensitivity_clf_to_GUI()
        rospy.loginfo( "Current sensitivity is [0~1]: "+ str(sensitivity)+ \
                       ', internal multiplier is '+ str(self.classifier.ths_mult) )
        self.sensitivity_pub.publish(sensitivity)                                   

        
    def reset(self):
        ''' Reset parameters '''
        self.lock.acquire()
        self.dataList = []
        self.enable_detector = False
        self.lock.release()

    def run(self, freq=20):
        ''' Run detector '''
        rospy.loginfo("Start to run anomaly detection: " + self.task_name)
        rate = rospy.Rate(freq) # 25Hz, nominally.
        while not rospy.is_shutdown():

            if self.enable_detector is False: 
                self.dataList = []
                rate.sleep()                
                continue

            #-----------------------------------------------------------------------
            self.lock.acquire()
            if len(self.dataList) == 0 or len(self.dataList[0][0]) < self.startCheckIdx:
                rate.sleep()                
                self.lock.release()
                continue
            if self.viz: self.viz_raw_input(self.dataList)

            cur_length = len(self.dataList[0][0])
            logp, post = self.ml.loglikelihood(self.dataList, bPosterior=True)
            self.lock.release()
            #-----------------------------------------------------------------------

            print logp
            continue

            
            if logp is None: 
                rospy.loginfo( "logp is None => anomaly" )
                self.action_interruption_pub.publish(self.task_name+'_anomaly')
                self.task_interruption_pub.publish(self.task_name+'_anomaly')
                self.soundHandle.play(2)
                self.enable_detector = False
                self.reset()
                continue

            if np.argmax(post)==0 and logp < 0.0: continue
            if np.argmax(post)>self.param_dict['HMM']['nState']*0.9: continue

            post = post[cur_length-1]
            ll_classifier_test_X = [logp] + post.tolist()                
            if 'svm' in self.method or 'sgd' in self.method:
                X = self.scaler.transform([ll_classifier_test_X])
            else:
                X = ll_classifier_test_X

            # anomal classification
            y_pred = self.classifier.predict(X)
            print "logp: ", logp, "  state: ", np.argmax(post), " y_pred: ", y_pred
            if type(y_pred) == list: y_pred = y_pred[-1]

            if y_pred > 0.0:
                rospy.loginfo( '-'*15 +  'Anomaly has occured!' + '-'*15 )
                self.action_interruption_pub.publish(self.task_name+'_anomaly')
                self.task_interruption_pub.publish(self.task_name+'_anomaly')
                self.soundHandle.play(2)
                self.anomaly_flag    = True                
                self.enable_detector = False
                self.reset()

            rate.sleep()


        # save model and params
        self.save_config()
        rospy.loginfo( "Saved current parameters")

    def runSim(self, auto=True, subject_names=['ari'], raw_data_path=None):
        '''
        Run detector with offline data
        '''
        from hrl_anomaly_detection import data_manager as dm
        from hrl_anomaly_detection.hmm import learning_hmm
        from hrl_anomaly_detection import util
        import hrl_manipulation_task.record_data as rd

        self.rf_radius = self.param_dict['data_param']['local_range'] #temp
        self.rf_center = self.param_dict['data_param']['rf_center']   #temp
        checked_fileList = []
        self.unused_fileList = []

        ## fb = ut.get_keystroke('Hit a key to load a new file')
        ## sys.exit()


        print "############## CUMULATIVE / REF EVAL ###################"
        self.acc_all, _, _ = evaluation(list(self.ll_test_X), list(self.ll_test_Y), self.classifier)
        ## self.acc_ref, _, _ = self.evaluation_ref()
        self.update_list.append(0)
        self.cum_acc_list.append(self.acc_all)
        self.ref_acc_list.append(self.acc_ref)
        print "######################################################"


        for i in xrange(100):

            if rospy.is_shutdown(): break

            if auto:
                if i < len(self.eval_run_fileList):
                    ## import shutil
                    ## tgt_dir = os.path.join(raw_data_path, 'new_'+self.task_name)
                    ## shutil.copy2(self.eval_run_fileList[i], tgt_dir)
                    self.new_run_file = self.eval_run_fileList[i:i+1]
                    unused_fileList = self.new_run_file
                else:
                    print "no more file"
                    fb = ut.get_keystroke('Hit a key to exit')
                    sys.exit()                    
            else:            
                # load new file            
                fb = ut.get_keystroke('Hit a key to load a new file')
                if fb == 'z' or fb == 's': break
                                                          
                unused_fileList = util.getSubjectFileList(raw_data_path, \
                                                          subject_names, \
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
                    
                trainData,_ = dm.getDataList([unused_fileList[j]], self.rf_center, self.rf_radius,\
                                           self.handFeatureParams,\
                                           downSampleSize = self.param_dict['data_param']['downSampleSize'], \
                                           handFeatures   = self.handFeatures)
                                           
                # scaling and subtracting offset
                trainData = np.array(trainData)*self.scale
                trainData = self.applying_offset(trainData)
                                
                ll_logp, ll_post = self.ml.loglikelihoods(trainData, bPosterior=True)
                X, Y = learning_hmm.getHMMinducedFeatures(ll_logp, ll_post, [label])
                X_test, Y_train_org, _ = dm.flattenSample(X, Y)

                if 'svm' in self.method or 'sgd' in self.method:
                    X_scaled = self.scaler.transform(X_test)
                    y_est    = self.classifier.predict(X_scaled)
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
                ## self.acc_ref, _, _ = self.evaluation_ref()
                if update_flag:
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


    def save_config(self):
        ''' Save detector '''
        # name matches with detector parameter file name.
        yaml_file = os.path.join(self.save_data_path, 'anomaly_detection_'+self.task_name+'.yaml')
        param_namespace = '/'+self.task_name 
        os.system('rosparam dump '+yaml_file+' '+param_namespace)

        # Save scaler
        if 'svm' in self.method or 'sgd' in self.method:
            with open(self.scaler_model_file, 'wb') as f:
                pickle.dump(self.scaler, f)
                
        # Save classifier
        if self.bSim is False:
            print "save model"
            self.classifier.save_model(self.classifier_model_file)
        

    def applying_offset(self, data):

        # get offset
        curData = np.reshape( np.mean(data[:,:,:self.startOffsetSize], axis=(1,2)), \
                              (self.nEmissionDim,1,1) ) # 4,1,1
        offsetData = self.refData - curData
                                  
        for i in xrange(self.nEmissionDim):
            data[i] = (np.array(data[i]) + offsetData[i][0][0]).tolist()

        return data


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

    

    def viz_raw_input(self, x):

        if self.figure_flag is False:
            self.fig = plt.figure()
            plt.ion()
            plt.show()
        else:            
            del self.ax.collections[:]

        for i in xrange(len(x)):
            self.ax = self.fig.add_subplot(100*self.nEmissionDim+10+i+1)
            self.ax.plot(np.array(x)[i,0], 'r-')
            self.ax.plot(self.mean_train[i], 'b-', lw=3.0)

        ## plt.axis('tight')
        plt.draw()
        self.figure_flag = True
        ## ut.get_keystroke('Hit a key')



if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    p.add_option('--task', action='store', dest='task', type='string', default='feeding',
                 help='type the desired task name')
    p.add_option('--method', '--m', action='store', dest='method', type='string', default='hmmgp',
                 help='type the method name')
    p.add_option('--id', action='store', dest='id', type=int, default=0,
                 help='type the detector id')
    p.add_option('--debug', '--d', action='store_true', dest='bDebug',
                 default=False, help='Enable debugging mode.')
    p.add_option('--simulation', '--sim', action='store_true', dest='bSim',
                 default=False, help='Enable a simulation mode.')
    
    p.add_option('--viz', action='store_true', dest='bViz',
                 default=False, help='Visualize data.')
    
    opt, args = p.parse_args()
    rospy.init_node(opt.task)

    if True:
        from hrl_execution_monitor.params.IROS2017_params import *
        # IROS2017
        subject_names = ['s2', 's3','s4','s5', 's6','s7','s8', 's9']
        raw_data_path, save_data_path, param_dict = getParams(opt.task)
        save_data_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/IROS2017/'+opt.task+'_demo'
        
    else:
        rospy.loginfo( "Not supported task")
        sys.exit()


    ad = anomaly_detector(opt.task, opt.method, opt.id, save_data_path, \
                          param_dict, debug=opt.bDebug, viz=True)
                          
    if opt.bSim is False: ad.run()
    else:                 ad.runSim(subject_names=test_subject, raw_data_path=raw_data_path)








