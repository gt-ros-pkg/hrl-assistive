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

# system
import rospy, os, sys, threading, datetime, copy
import random, numpy as np

## from hrl_lib import circular_buffer as cb
from sklearn import preprocessing

# Utility
import hrl_lib.util as ut
from hrl_execution_monitor import anomaly_isolator_util as aiu
from hrl_execution_monitor import util as autil

from cv_bridge import CvBridge, CvBridgeError
import cv2

#msg
from hrl_anomaly_detection.msg import MultiModality
from std_msgs.msg import String
from sensor_msgs.msg import CameraInfo, Image
from hrl_srvs.srv import Bool_None, Bool_NoneResponse, StringArray_None
from hrl_msgs.msg import FloatArray, StringArray, FloatMatrix

QUEUE_SIZE = 10
img_size  = (256,256)
crop_size = (224,224)

class anomaly_isolator:
    def __init__(self, task_name, save_data_path, param_dict, debug=False, verbose=False):
        rospy.loginfo('Initializing anomaly isolator')

        self.task_name      = task_name.lower()
        self.save_data_path = save_data_path        
        self.cur_task       = None
        self.verbose        = verbose
        self.debug           = debug
        self.bridge = CvBridge()

        # Important containers
        self.enable_isolator = False
        self.refData         = None
        self.dyn_data1       = []
        self.dyn_data2       = []
        self.stc_data        = []
        self.img_data        = None
        self.init_msg        = None
        
        # Params
        self.param_dict      = param_dict        
        self.f_param_dict    = None
        self.startOffsetSize = 4
        self.startCheckIdx   = 10
        self.max_length      = 10

        # HMM, Classifier
        self.hmm_list        = None

        # Comms
        self.lock = threading.Lock()        
        self.img_lock = threading.Lock()        
        self.initParams()
        self.initComms()
        self.initIsolator()

        if self.verbose:
            rospy.loginfo( "==========================================================")
            rospy.loginfo( "Isolator initialized!! : %s", self.task_name)
            rospy.loginfo( "==========================================================")


    def initParams(self):

        # Features and parameters
        self.dynFeatures = self.param_dict['data_param']['handFeatures']
        self.stcFeatures = self.param_dict['data_param']['staticFeatures']
        self.nState = self.param_dict['HMM']['nState']
        self.scale  = self.param_dict['HMM']['scale']
        self.nDetector = rospy.get_param('nDetector')
        self.nStcFeatures = len(self.stcFeatures)

        self.n_labels = 12
        self.classes = ['Object collision', 'Noisy environment', 'Spoon miss by a user', 'Spoon collision by a user', 'Robot-body collision by a user', 'Aggressive eating', 'Anomalous sound from a user', 'Unreachable mouth pose', 'Face occlusion by a user', 'Spoon miss by system fault', 'Spoon collision by system fault', 'Freeze by system fault']


    def initComms(self):
        # Publisher
        self.isolation_info_pub = rospy.Publisher("/manipulation_task/anomaly_type", String,
                                                  queue_size=QUEUE_SIZE)
        
        # Subscriber # TODO: topic should include task name prefix?
        rospy.Subscriber('/hrl_manipulation_task/raw_data', MultiModality, self.staticDataCallback)
        rospy.Subscriber('manipulation_task/hmm_input0', FloatMatrix, self.dtc1DataCallback)
        rospy.Subscriber('manipulation_task/hmm_input1', FloatMatrix, self.dtc2DataCallback)
        rospy.Subscriber('/SR300/rgb/image_raw_rotated', Image, self.imgDataCallback)

        rospy.Subscriber('/manipulation_task/status', String, self.statusCallback)

        # Service
        self.isolation_srv = rospy.Service('anomaly_isolator_enable',
                                           Bool_None, self.enablerCallback)


    def initIsolator(self):
        ''' init detector ''' 
        rospy.loginfo( "Initializing a detector for %s", self.task_name)

        self.f_param_dict, self.hmm_list, self.scr, _\
        = aiu.get_isolator_modules(self.save_data_path, self.task_name, self.param_dict,
                                   nDetector=self.nDetector)
        normalTrainData = self.f_param_dict['successData'] 
        self.refData = np.reshape( np.mean(normalTrainData[:,:,:self.startOffsetSize], axis=(1,2)), \
                                  (len(self.stcFeatures),1) ) # 4,1,1

        input_shape1 = (3,224,224)
        input_shape2 = (31,)
        self.keras_model = get_isolator_keras_model(input_shape1, input_shape2, self.n_labels,
                                                    self.save_data_path, prefix='vgg')

                                  

    #-------------------------- Communication fuctions --------------------------
    def enablerCallback(self, msg):

        if msg.data is True:
            rospy.loginfo("%s anomaly isolator enabled", self.task_name)
            self.enable_isolator = True
        else:
            rospy.loginfo("%s anomaly isolator disabled", self.task_name)
            self.enable_detector = False

        return Bool_NoneResponse()

    
    def imgDataCallback(self, msg):
        '''
        capture image
        '''
        with self.img_lock:
            self.img_data = self.bridge.imgmsg_to_cv2(msg)
            self.img_data = self.img_data[:,:,[2,1,0]] # rgb to bgr

        ## cv2.imshow('image',self.img_data)
        ## cv2.waitKey(0)
        ## cv2.destroyAllWindows()
        ## print np.shape(self.img_data), msg.height, msg.width
        ## h, w = msg.height, msg.width


    def dtc1DataCallback(self, msg):
        '''
        Subscribe HMM0 data
        '''
        print "HMM0 data came in ", np.shape(msg.data)
        self.dyn_data1 = np.array(msg.data).reshape((msg.size, 1, len(msg.data)/msg.size))

    def dtc2DataCallback(self, msg):
        '''
        Subscribe HMM1 data
        '''
        print "HMM1 data came in ", np.shape(msg.data)
        self.dyn_data2 = np.array(msg.data).reshape((msg.size, 1, len(msg.data)/msg.size))

    def staticDataCallback(self, msg):
        '''
        Subscribe static data
        '''
        if msg is None: return
        if self.f_param_dict is None or self.refData is None: return
        if self.cur_task is None: return
        if self.cur_task.find(self.task_name) < 0: return

        # If detector is disbled, detector does not fill out the stc_data.
        if self.enable_isolator is False or self.init_msg is None:
            self.init_msg = copy.copy(msg)
            return

        self.lock.acquire()
        ########################################################################
        # Run your custom feature extraction function
        dataSample, scaled_dataSample = autil.extract_feature(msg,
                                                              self.init_msg,
                                                              self.init_msg,
                                                              None,
                                                              self.stcFeatures,
                                                              self.f_param_dict['feature_params'],
                                                              count=len(self.stc_data[0])
                                                              if len(self.stc_data)>0 else 0)
        ########################################################################

        # Subtract white noise by measuring offset using scaled data
        if len(self.stc_data)==0 or len(self.stc_data[0]) < self.startOffsetSize:
            self.offsetData = np.zeros(np.shape(scaled_dataSample))            
        elif len(self.stc_data[0]) == self.startOffsetSize:
            curData = np.reshape( np.mean(self.stc_data, axis=1), (self.nStcFeatures,1) ) # 4,1
            self.offsetData = self.refData - curData
            for i in xrange(self.nStcFeatures):
                self.stc_data[i] = (np.array(self.stc_data[i]) +
                                            self.offsetData[i][0]).tolist()
            self.offsetData = self.offsetData[:,0]
            
        scaled_dataSample = np.array(scaled_dataSample) + self.offsetData
        scaled_dataSample = scaled_dataSample.tolist()
        
        if len(self.stc_data) == 0:
            self.stc_data = np.reshape(scaled_dataSample,
                                               (self.nStcFeatures,1) ).tolist() # = newData.tolist()
        else:                
            # dim x length
            self.stc_data = np.hstack([self.stc_data, np.reshape(scaled_dataSample,
                                                                 (self.nStcFeatures,1) ) ])
            ## for i in xrange(self.nStcFeatures):
            ##     self.stc_data[i] += scaled_dataSample[i]

        self.lock.release()


    def statusCallback(self, msg):
        '''
        Subscribe current task 
        '''
        self.cur_task = msg.data.lower()
        

    #-------------------------- General fuctions --------------------------
    def reset(self):
        ''' Reset parameters '''
        self.lock.acquire()
        self.dyn_data1 = []
        self.dyn_data2 = []
        self.stc_data  = []
        self.enable_isolator = False
        self.cur_task = None
        self.lock.release()


    def run(self, freq=5):
        ''' Run detector '''
        rospy.loginfo("Start to run anomaly isolation: " + self.task_name)
        rate = rospy.Rate(freq) # 20Hz, nominally.
        while not rospy.is_shutdown():

            if len(self.dyn_data1)>0 and len(self.dyn_data2)>0 and len(self.stc_data)>0 and\
                len(self.img_data)>0 and self.enable_isolator:
                print "Start to isolate an anomaly!!!!!!!!!!!!!!!!!!!!!!!!!!"
                # run isolator
                x1 = autil.temporal_features(np.array(self.dyn_data1)[:,0], self.max_length, self.hmm_list[0],
                                             self.scale[0])
                x2 = autil.temporal_features(np.array(self.dyn_data2)[:,0], self.max_length, self.hmm_list[1],
                                             self.scale[1])
                vs = np.hstack([x1, x2])

                # temporal feature
                x = np.amin(vs[:1], axis=0)
                x = np.vstack([ x, np.amin(vs[:4], axis=0) ])
                x = np.vstack([ x, np.amin(vs[:8], axis=0) ])
                x = x.flatten().tolist()

                # dim x length?
                self.lock.acquire()
                stc_data = copy.deepcopy(self.stc_data)
                self.lock.release()
                
                max_vals = np.amax(stc_data, axis=1)
                min_vals = np.amin(stc_data, axis=1)
                vals = [mx if abs(mx) > abs(mi) else mi for (mx, mi) in zip(max_vals, min_vals) ]
                x += vals
                x = self.scr.transform(x)

                ## Image -------------------------------------------------------------
                print np.shape(self.img_data)

                with self.img_lock:
                    img = cv2.resize(self.img_data, img_size).astype(np.float32)
                img = img[(img_size[0]-crop_size[0])//2:(img_size[0]+crop_size[0])//2
                          ,(img_size[1]-crop_size[1])//2:(img_size[1]+crop_size[1])//2,:]

                # for vgg 
                img[:,:,0] -= 103.939
                img[:,:,1] -= 116.779
                img[:,:,2] -= 123.68
                img = np.expand_dims(img.transpose((2,0,1)), axis=0)
                x   = np.expand_dims(x,axis=0)

                # ---------------------------------------------------------------------
                y_pred = self.keras_model.predict([img/255., x])
                y_pred = np.argmax(y_pred, axis=1)
                ## y_pred = self.cf.predict(x)
                anomaly_type = self.classes[y_pred[0]]
                print "Detected anomaly is ", anomaly_type
                self.isolation_info_pub.publish(anomaly_type)

                # reset                
                self.reset()
            
            rate.sleep()




def get_isolator_keras_model(input_shape1, input_shape2, n_labels, save_data_path, prefix='vgg',
                             fold_idx=0):

    if prefix is 'vgg': prefix = 'vgg_'

    ## weights_path = os.path.join(save_data_path,'keras',prefix+'all_weights_'+str(fold_idx)+'_demo.h5')
    weights_path = os.path.join(save_data_path,'keras',prefix+'all_weights_'+str(fold_idx)+'.h5')

    from hrl_execution_monitor.keras_util import keras_model as km    
    model = km.vgg16_net(input_shape1, n_labels, with_multi_top=True,
                         input_shape2=input_shape2, weights_path=weights_path,
                         fine_tune=True)

    return model



if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    p.add_option('--task', action='store', dest='task', type='string', default='feeding',
                 help='type the desired task name')
    p.add_option('--debug', '--d', action='store_true', dest='bDebug',
                 default=False, help='Enable debugging mode.')
    
    p.add_option('--viz', action='store_true', dest='bViz',
                 default=False, help='Visualize data.')
    
    opt, args = p.parse_args()
    rospy.init_node(opt.task+'_isolator')

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

    ai = anomaly_isolator(opt.task, save_data_path, param_dict)
    ai.run()


    


