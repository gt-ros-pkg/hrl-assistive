#!/usr/bin/env python  

# System
import numpy as np, math
import time
import threading
import os, sys

# ROS
import roslib;roslib.load_manifest('hrl_anomaly_detection')
import rospy
import tf

# HRL
from hrl_srvs.srv import Bool_None, Bool_NoneResponse
from std_msgs.msg import Bool
from hrl_msgs.msg import FloatArray

# Private
import hrl_anomaly_detection.door_opening.door_open_common as doc
from hrl_anomaly_detection.HMM.anomaly_checker import anomaly_checker
import hrl_anomaly_detection.door_opening.mechanism_analyse_daehyung as mad
from hrl_anomaly_detection.HMM.learning_hmm import learning_hmm


class anomaly_checker_door(anomaly_checker):

    ## Init variables
    ## data_path = os.environ['HRLBASEPATH']+'_data/usr/advait/ram_www/data_from_robot_trials/'
    data_path = os.environ['HRLBASEPATH']+'/src/projects/modeling_forces/handheld_hook/'
    root_path = os.environ['HRLBASEPATH']+'/'
    ## nState    = 19
    nMaxStep  = 36 # total step of data. It should be automatically assigned...
    nFutureStep = 8
    ## data_column_idx = 1
    fObsrvResol = 0.1
    nCurrentStep = 4  #14

    nClass = 2
    mech_class = doc.class_list[nClass]
    
    def __init__(self, nDim=1, fXInterval=1.0, fXMax=90.0, sig_mult=1.0, sig_off=0.3, bManual=False):

        self.bManual = bManual 
        self.nDim    = nDim
        self.fXInterval = fXInterval
        self.fXMax   = fXMax
        self.sig_mult = sig_mult
        self.sig_off  = sig_off
        self.bManual  = bManual

        self.robot_path = '/pr2'
        self.anomaly_detector_en = False
        
        # Load parameters
        pkl_file  = "mech_class_"+doc.class_dir_list[self.nClass]+".pkl"      
        data_vecs, _, _ = mad.get_data(pkl_file, mech_class=self.mech_class, renew=False) # human data       
        A, B, pi, nState = doc.get_hmm_init_param(mech_class=self.mech_class)        

        # Training 
        self.lh = learning_hmm(data_path=os.getcwd(), aXData=data_vecs[0], nState=nState, 
                          nMaxStep=self.nMaxStep, nFutureStep=self.nFutureStep, 
                          fObsrvResol=self.fObsrvResol, nCurrentStep=self.nCurrentStep)    
        self.lh.fit(self.lh.aXData, A=A, B=B, verbose=False)    

        # define lock
        self.mech_data_lock = threading.RLock() ## mechanism data lock
        
        #
        rospy.init_node("detector_online")        
        self.init_vars()
        self.getParams()        
        self.initComms()

        
    def init_vars(self):
        # Checker
        anomaly_checker.__init__(self, self.lh, self.nDim, self.fXInterval, self.fXMax, self.sig_mult, self.sig_off)        
        self.f_list  = []
        self.a_list  = []

        
    def getParams(self):

        self.torso_frame = rospy.get_param('haptic_mpc'+self.robot_path+'/torso_frame' )
        self.ee_frame = rospy.get_param('haptic_mpc'+self.robot_path+'/end_effector_frame' )

        
    def initComms(self):
        
        # service
        rospy.Service('/door_opening/anomaly_detector_enable', Bool_None, self.detector_en_cb)

        # Publisher
        self.anomaly_pub = rospy.Publisher("door_opening/anomaly", Bool)        
        
        # Subscriber
        rospy.Subscriber('door_opening/mech_data', FloatArray, self.mech_data_cb)                    
        
        
    def detector_en_cb(self, req):

        # Run manipulation tasks
        if req.data is True:
            self.anomaly_detector_en = True            
        else:
            self.anomaly_detector_en = False
                        
        return Bool_NoneResponse()
        
    
    def mech_data_cb(self, msg):
        with self.mech_data_lock:        
            [f_tan_mag, ang] = msg.data # must be series data           

            if self.anomaly_detector_en or self.bManual:
                # anomaly detection
                self.f_list.append(f_tan_mag)
                self.a_list.append(ang)                

                mu_list, var_list, idx = self.update_buffer(self.f_list,self.a_list)            

                ## # check anomaly score
                bFlag, fScore, _ = self.check_anomaly(y[-1])
                
                if fScore>=self.fAnomaly and ang>0.0:
                    self.anomaly_pub.publish(True)
                else:
                    self.anomaly_pub.publish(False)
            else:
                self.init_vars()
                ## self.anomaly_pub.publish(False)


    ## def start(self, bManual=False):
    ##     rospy.loginfo("Mech Analyse Online start")
        
    ##     rate = rospy.Rate(2) # 25Hz, nominally.
    ##     rospy.loginfo("Beginning publishing waypoints")
    ##     while not rospy.is_shutdown():         
    ##         self.generateMechData()
    ##         #print rospy.Time()
    ##         rate.sleep()        
                

if __name__ == '__main__':

    import optparse
    p = optparse.OptionParser()
    opt, args = p.parse_args()
    
    acd = anomaly_checker_door(sig_mult=1.0)
    rospy.spin()
    
    ## rate = rospy.Rate(25) # 25Hz, nominally.
    ## rospy.loginfo("Beginning publishing waypoints")
    ## while not rospy.is_shutdown():         
    ##     ## self.generateWaypoint()
    ##     #print rospy.Time()
    ##     rate.sleep()        
    
