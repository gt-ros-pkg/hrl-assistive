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

#  \author You Keun Kim  (Healthcare Robotics Lab, Georgia Tech.)
#  \author Daehyung Park (Healthcare Robotics Lab, Georgia Tech.)

# system library
import time
import datetime
import multiprocessing

# ROS library
import rospy, roslib

from std_msgs.msg import String

# HRL library
from hrl_srvs.srv import String_String, String_StringRequest
import hrl_lib.util as ut

from hrl_manipulation_task.record_data import logger


class armReacherGUI:

    def __init__(self, detection_flag=False, log=None):
        '''Initialize GUI'''

        #variables
        self.emergencyStatus = False
        self.inputStatus = False
        self.actionStatus = 'Both'
        self.inputMsg = None
        self.feedbackMsg = None
        self.emergencyMsg = None
        self.ScoopNumber = 0
        self.FeedNumber = 0
        self.detection_flag = detection_flag
        self.log = log
        self.left_mtx = False
        self.right_mtx = False
        self.recordStatus = False
        ##manipulation_task/user_input (user_feedback)(emergency)(status)

        self.initComms()
        self.run()


    def initComms(self):
        #Publisher:
        self.emergencyPub = rospy.Publisher("/hrl_manipulation_task/InterruptAction", String)        
        self.falselogPub = rospy.Publisher("/manipulation_task/feedbackRequest", String)
                
        #subscriber:
        self.inputSubscriber = rospy.Subscriber("/manipulation_task/user_input", String, self.inputCallback)
        self.emergencySubscriber = rospy.Subscriber("/manipulation_task/emergency", String, self.emergencyCallback)
        self.feedbackSubscriber = rospy.Subscriber("/manipulation_task/user_feedback", String, self.feedbackCallback)
        self.statusSubscriber = rospy.Subscriber("/manipulation_task/status", String, self.statusCallback)
        
        rospy.wait_for_service("/arm_reach_enable")
        self.armReachActionLeft  = rospy.ServiceProxy("/arm_reach_enable", String_String)
        self.armReachActionRight = rospy.ServiceProxy("/right/arm_reach_enable", String_String)
        

    def inputCallback(self, data):
        #Callback function for input. It communicate with both start and continue button.
        rospy.wait_for_service("/arm_reach_enable")
        self.inputMSG = data.data
        self.inputStatus = True
        #Maybe had to add if statement.
        self.emergencyStatus = False
        print "Input received"
        if self.inputMSG == 'Start':
            self.ScoopNumber = 0
            self.FeedNumber = 0


    def emergencyCallback(self, data):
        #Emergency status button.
        self.emergencyStatus = True
        self.inputStatus = False
        self.emergencyMsg = data.data
        if self.emergencyMsg == 'STOP':
            self.emergencyPub.publish("STOP")
        print "Emergency received"
        if self.recordStatus:    
            self.log.close_log_file_GUI()
            self.recordStatus = False
        rospy.sleep(3.0)

        rospy.wait_for_service("/arm_reach_enable")
        
        print "Aborting Sequence"
        self.testing(self.armReachActionLeft, self.armReachActionRight, self.log, self.detection_flag)



    def feedbackCallback(self, data):
        #record_data.py take cares of logging. This is here, just incase implementation to this program is needed.
        self.feedbackMsg = data.data

    def statusCallback(self, data):
        #Change the status, depending on the button pressed.
        self.actionStatus = data.data
        rospy.loginfo("status received")
        if self.log != None:
            if self.actionStatus == "Scooping":
                self.log.setTask('scooping')
            elif self.actionStatus == "Feeding":
                self.log.setTask('feeding')

            print "" + self.log.task


    # --------------------------------------------------------------------------
    def run(self):
        print "Continous run function called"
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.inputStatus and self.actionStatus == 'Scooping':
                self.inputStatus = False
                rospy.loginfo("Scooping Starting...")
                self.scooping(self.armReachActionLeft, self.armReachActionRight, self.log, self.detection_flag)
            elif self.inputStatus and self.actionStatus == 'Feeding':
                self.inputStatus = False
                rospy.loginfo("Feeding Starting....")
                self.feeding(self.armReachActionLeft, self.armReachActionRight, self.log, self.detection_flag)
            elif self.inputStatus and self.actionStatus == 'Both':
                self.inputStatus = False
                rospy.loginfo("Scoop and Feed starting...")
                self.scooping(self.armReachActionLeft, self.armReachActionRight, self.log, self.detection_flag)
                self.feeding(self.armReachActionLeft, self.armReachActionRight, self.log, self.detection_flag)
            rate.sleep()
 

    def scooping(self, armReachActionLeft, armReachActionRight, log, detection_flag, \
                 train=False, abnormal=False):

        while not self.emergencyStatus and not rospy.is_shutdown():

            if self.log != None:
                self.log.setTask('scooping')
                self.log.initParams()
            self.FeedNumber = 0
            ## Scooping -----------------------------------    
            if self.ScoopNumber < 1:
                rospy.loginfo("Initializing arms for scooping")
                leftProc = multiprocessing.Process(target=self.ServiceCallLeft, args=("initScooping1",))
                rightProc = multiprocessing.Process(target=self.ServiceCallRight, args=("initScooping2",))
                leftProc.start(); rightProc.start()
                leftProc.join(); rightProc.join()
                if self.emergencyStatus: break
                self.ScoopNumber = 1
        
            if self.ScoopNumber < 2:        
                self.ServiceCallLeft("getBowlPos")            
                self.ServiceCallLeft('lookAtBowl')
                if self.emergencyStatus: break
                self.ServiceCallLeft("initScooping2")
                if self.emergencyStatus: break
                self.ScoopNumber = 2        
    
    
            rospy.loginfo("Start to log!")
            if self.log != None:
                self.log.log_start()
                self.recordStatus = True
            if detection_flag: self.log.enableDetector(True)
        
            rospy.loginfo("Running scooping!")
            self.ServiceCallLeft("runScooping")
            if self.emergencyStatus: break

            if self.log == None:
                self.falselogPub.publish("Requesting Feedback!")
    
            if detection_flag: self.log.enableDetector(False)
            rospy.loginfo("Finish to log!")
            if self.log != None and self.recordStatus:
                self.log.close_log_file_GUI()
                self.recordStatus = False 
            self.ScoopNumber = 0
            break

    
    def feeding(self, armReachActionLeft, armReachActionRight, log, detection_flag):

        while not self.emergencyStatus and not rospy.is_shutdown():
            if self.log != None:
                self.log.setTask('feeding' )
                self.log.initParams()

            if self.FeedNumber < 1:
                ## Feeding -----------------------------------
                rospy.loginfo("Initializing left arm for feeding")
                self.ServiceCallLeft("lookToRight")
                if self.emergencyStatus: break
                self.ServiceCallLeft("getHeadPos")
                self.ServiceCallLeft("initFeeding1")
                if self.emergencyStatus: break
                self.ServiceCallRight("getHeadPos")
                print armReachActionRight("initFeeding")
                if self.emergencyStatus: break                
                self.FeedNumber = 1
    
            if self.FeedNumber < 2:
                rospy.loginfo("Detect a mouth")
                self.ServiceCallLeft("getHeadPos")
                self.ServiceCallLeft("getHeadPos")
                self.FeedNumber = 2
    
            if self.FeedNumber < 3:
                rospy.loginfo("Running init feeding2")
                self.ServiceCallLeft("initFeeding2")
                if self.emergencyStatus: break
                self.FeedNumber = 3
    
            rospy.loginfo("Start to log!")
            if self.log != None:
                self.log.log_start()
                self.recordStatus = True
            if detection_flag: self.log.enableDetector(True)
        
            rospy.loginfo("Running feeding")
            self.ServiceCallLeft("runFeeding")
            if self.emergencyStatus: break
            
            if self.log == None:
                self.falselogPub.publish("Requesting Feedback!")
    
            if detection_flag: self.log.enableDetector(False)
            rospy.loginfo("Finish to log!")
            if self.log != None and self.recordStatus:
                self.log.close_log_file_GUI()
                self.recordStatus = False
            self.ServiceCallLeft("initScooping1")
            if self.emergencyStatus: break
            self.FeedNumber = 0
            break

        
    def ServiceCallLeft(self, cmd):
        if self.left_mtx is not True:
            self.left_mtx = True
            self.armReachActionLeft(cmd)            
            self.left_mtx = False
        else:
            print "Ignore last command...."

    def ServiceCallRight(self, cmd):
        if self.right_mtx is not True:
            self.right_mtx = True
            self.armReachActionRight(cmd)            
            self.right_mtx = False
        else:
            print "Ignore last command...."
        
            
    def testing(self, armReachActionLeft, armReachActionRight, log, detection_flag):
        if self.FeedNumber<1:
            self.ScoopNumber = 0         
            self.ServiceCallLeft("initScooping1")
            self.ServiceCallRight("initScooping1")
        elif self.FeedNumber<2:
            self.ServiceCallLeft("initFeeding")
        else: 
            #print self.armReachActionLeft("runFeeding1")
            self.ServiceCallLeft("initFeeding")


 
if __name__ == '__main__':
    
    import optparse
    p = optparse.OptionParser()
    p.add_option('--data_pub', '--dp', action='store_true', dest='bDataPub',
                 default=False, help='Continuously publish data.')
    p.add_option('--en_anomaly_detector', '--ad', action='store_true', dest='bAD',
                 default=False, help='Enable anomaly detector.')
    p.add_option('--en_logger', '--l', action='store_true', dest='bLog',
                 default=False, help='Enable logger.')
    p.add_option('--data_path', action='store', dest='sRecordDataPath',
                 default='/home/dpark/hrl_file_server/dpark_data/anomaly/ICRA2017', \
                 help='Enter a record data path')
 
    opt, args = p.parse_args()
    rospy.init_node('arm_reach_client')

    ## rospy.sleep(2.0)    
    ## #print armReachActionLeft('lookAtMouth')
    ## print armReachActionLeft('lookAtBowl')
    
    if opt.bLog or opt.bDataPub:
        log = logger(ft=True, audio=False, audio_wrist=True, kinematics=True, vision_artag=False, \
                     vision_landmark=False, vision_change=False, pps=True, skin=False, \
                     subject="test", task='scooping', data_pub=opt.bDataPub, detector=opt.bAD, \
                     record_root_path=opt.sRecordDataPath, verbose=False)
    else:
        log = None

    last_trial  = '4'
    last_detect = '2'

    gui = armReacherGUI(detection_flag=opt.bAD, log=log)
    rospy.spin()


