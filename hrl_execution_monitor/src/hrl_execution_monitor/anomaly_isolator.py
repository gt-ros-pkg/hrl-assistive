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
import rospy, os, sys, threading, datetime
import random, numpy as np

from sklearn import preprocessing

# Utility
import hrl_lib.util as ut
from hrl_execution_monitor import anomaly_isolator_util as aiu
from hrl_execution_monitor import util as autil

#msg
from hrl_anomaly_detection.msg import MultiModality
from std_msgs.msg import String, Float64
from sensor_msgs.msg import CameraInfo, Image

QUEUE_SIZE = 10

class anomaly_isolator:
    def __init__(self, task_name, save_data_path, param_dict, verbose=False):
        rospy.loginfo('Initializing anomaly isolator')

        self.task_name      = task_name.lower()
        self.save_data_path = save_data_path        
        self.verbose        = verbose
        self.debug           = debug
        self.viz             = viz

        # Important containers
        self.enable_isolator = False
        self.data_list       = []
        self.refData         = None
        
        # Params
        self.param_dict      = param_dict        

        # HMM, Classifier
        self.hmm_list        = None

        # Comms
        self.lock = threading.Lock()        
        self.initParams()
        self.initComms()
        self.initIsolator()

        if self.verbose:
            rospy.loginfo( "==========================================================")
            rospy.loginfo( "Isolator initialized!! : %s", self.task_name)
            rospy.loginfo( "==========================================================")


    def initParams(self):

        # Features and parameters
        self.staticFeatures = self.param_dict['data_param']['staticFeatures']
        self.nState = self.param_dict['HMM']['nState']
        self.scale  = self.param_dict['HMM']['scale']
        
        self.nStaticDim = len(staticFeatures)
        self.nDetector = rospy.get_param('nDetector')


    def initComms(self):
        # Publisher
        self.isolation_info_pub = rospy.Publisher("/manipulation_task/anomaly_type", String,
                                                  queue_size=QUEUE_SIZE)
        
        # Subscriber # TODO: topic should include task name prefix?
        rospy.Subscriber('/hrl_manipulation_task/raw_data', MultiModality, self.rawDataCallback)
        rospy.Subscriber('/manipulation_task/dtc1_data', FloatArray, self.dtc1DataCallback)
        rospy.Subscriber('/manipulation_task/dtc2_data', FloatArray, self.dtc2DataCallback)
        rospy.Subscriber('/SR300/rgb/image_raw_rotated', Image, self.imgDataCallback)

        rospy.Subscriber('/manipulation_task/status', String, self.statusCallback)



    def initIsolator(self):
        ''' init detector ''' 
        rospy.loginfo( "Initializing a detector for %s", self.task_name)

        self.hmm_list = adu.get_isolator_modules(self.save_data_path,
                                                 self.task_name,
                                                 self.param_dict)

        

    #-------------------------- Communication fuctions --------------------------
    def imgDataCallback(self, msg):
        '''
        capture image
        '''
        msg.data
        
    def dtc1DataCallback(self, msg):
        '''
        Subscribe raw data
        '''
        self.data_list[0] = msg.data

    def dtc2DataCallback(self, msg):
        '''
        Subscribe raw data
        '''
        self.data_list[1] = msg.data

    def staticDataCallback(self, msg):
        '''
        Subscribe raw data
        '''
        self.data_list[2] = msg.data

    #-------------------------- General fuctions --------------------------
    def reset(self):
        ''' Reset parameters '''
        self.lock.acquire()
        self.dataList = []
        self.enable_isolator = False
        self.lock.release()


    def run(self, freq=5):
        ''' Run detector '''
        rospy.loginfo("Start to run anomaly isolation: " + self.task_name)
        rate = rospy.Rate(freq) # 20Hz, nominally.
        while not rospy.is_shutdown():
            rate.sleep()






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


    save_data_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/IROS2017/'+opt.task+'_demo1'

    ai = anomaly_isolator(opt.task, save_data_path, param_dict, debug=opt.bDebug, viz=False)
    ai.run()


    
