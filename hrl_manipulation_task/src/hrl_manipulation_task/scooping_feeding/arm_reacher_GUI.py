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
import multiprocessing, threading

# ROS library
import rospy, roslib

from std_msgs.msg import String, Bool
from hrl_msgs.msg import StringArray

# HRL library
from hrl_srvs.srv import String_String, String_StringRequest
import hrl_lib.util as ut

from hrl_manipulation_task.record_data import logger
QUEUE_SIZE = 10


class armReacherGUI:

    def __init__(self, detection_flag=False, log=None, quick_feeding=False):
        '''Initialize GUI'''

        #variables
        self.emergencyStatus = False
        self.inputStatus = False
        self.actionStatus = 'Init'
        self.inputMsg = None
        self.feedbackMsg = None
        self.ScoopNumber = 0
        self.FeedNumber = 0
        self.detection_flag = detection_flag
        self.log = log
        self.left_mtx = False
        self.right_mtx = False
        self.status_lock = threading.RLock()
        self.encountered_emergency = 0
        self.expected_emergency = 0
        self.guiStatusReady = False
        self.gui_status = None
        self.feedback_received = True
        self.quick_feeding = quick_feeding
        self.quick_feeding_ready=False
        ##manipulation_task/user_input (user_feedback)(emergency)(status)

        self.initComms()
        self.run()


    def initComms(self):
        #Publisher:
        self.emergencyPub = rospy.Publisher("/manipulation_task/InterruptAction", String, queue_size=QUEUE_SIZE)        
        self.logRequestPub  = rospy.Publisher("/manipulation_task/feedbackRequest", String, queue_size=QUEUE_SIZE)
        self.availablePub = rospy.Publisher("/manipulation_task/available", String, queue_size=QUEUE_SIZE)
        self.proceedPub   = rospy.Publisher("/manipulation_task/proceed", String, queue_size=10, latch=True) 
        self.guiStatusPub = rospy.Publisher("/manipulation_task/gui_status", String, queue_size=1, latch=True)
        self.debugPub1    = rospy.Publisher("/manipulation_task/debug/input_bool", Bool, queue_size=QUEUE_SIZE)
        self.debugPub2    = rospy.Publisher("/manipulation_task/debug/emergency_bool", Bool, queue_size=QUEUE_SIZE)

        #subscriber:
        self.inputSubscriber = rospy.Subscriber("/manipulation_task/user_input", String, self.inputCallback)
        self.emergencySubscriber = rospy.Subscriber("/manipulation_task/emergency", String, self.emergencyCallback, queue_size=10)
        self.feedbackSubscriber = rospy.Subscriber("/manipulation_task/user_feedback", StringArray, self.feedbackCallback)
        self.statusSubscriber = rospy.Subscriber("/manipulation_task/status", String, self.statusCallback)
        self.guiStatusSub = rospy.Subscriber("/manipulation_task/gui_status", String, self.guiCallback, queue_size=1)
        
        rospy.wait_for_service("/arm_reach_enable")
        rospy.wait_for_service("/right/arm_reach_enable")
        self.armReachActionLeft  = rospy.ServiceProxy("/arm_reach_enable", String_String)
        self.armReachActionRight = rospy.ServiceProxy("/right/arm_reach_enable", String_String)

    def inputCallback(self, msg):
        #Callback function for input. It communicate with both start and continue button.
        if not self.guiStatusReady:
            return
        rospy.wait_for_service("/arm_reach_enable")
        rospy.wait_for_service("/right/arm_reach_enable")
        with self.status_lock:
            if not self.emergencyStatus:
                self.inputMSG = msg.data
                # initialize current motion only when Start button is pushed
                if self.inputMSG == 'Start':
                    self.ScoopNumber = 0
                    self.FeedNumber = 0
                if self.gui_status == 'wait start' or self.gui_status == 'stopped':#self.inputMSG == 'Start' or self.inputMSG == 'Continue':
                    self.inputStatus = True
                    #Maybe had to add if statement.
                    self.emergencyStatus = False
                    print "Input received"
                    self.feedback_received = False
                    self.guiStatusPub.publish("in motion")

    def emergencyCallback(self, msg):
        #Emergency status button.
        print "in emergency callback", msg
        self.emergencyMsg = msg.data
        if self.emergencyMsg == 'STOP':
            #if self.guiStatusReady:
            self.emergencyPub.publish("STOP")
            #else:
            #    return
        self.emergencyStatus = True
        self.inputStatus = False
        self.quick_feeding_ready = False

        temp_gui_status = self.gui_status
        if self.gui_status == "in motion":
            self.guiStatusPub.publish("stopping")
        print "Emergency received"
        if self.log != None:
            if self.log.getLogStatus(): self.log.log_stop()
        
        print "Wait arm reach service"
        rospy.wait_for_service("/arm_reach_enable")
        rospy.wait_for_service("/right/arm_reach_enable")

        emergency_wait_rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            #print "Waiting aborting Sequence"
            if self.left_mtx is False and self.right_mtx is False: break
            emergency_wait_rate.sleep()
        self.safetyMotion(self.armReachActionLeft, self.armReachActionRight)
        if temp_gui_status != "request feedback":
            temp_gui_status = self.gui_status
        self.availablePub.publish("true")
        if self.gui_status == "stopping":
            self.guiStatusPub.publish("stopped")
        else:
            print self.gui_status
        self.emergencyStatus = False            
        #if self.log != None: self.log.close_log_file_GUI()
        rospy.sleep(2.0)

    def feedbackCallback(self, msg):
        #record_data.py take cares of logging. This is here, just incase implementation to this program is needed.
        if not self.guiStatusReady:
            return
        self.feedbackMsg = msg.data
        self.feedback_received = True
        self.guiStatusPub.publish("select task")

    def statusCallback(self, msg):
        #Change the status, depending on the button pressed.
        if not self.guiStatusReady:
            return
        with self.status_lock:
            if not self.emergencyStatus:
                self.inputStatus = False
                self.actionStatus = msg.data
                rospy.loginfo("status received")
                self.availablePub.publish("true")
                self.guiStatusPub.publish("wait start")
                self.FeedNumber = 0
                self.ScoopNumber = 0
                if self.log != None:
                    if self.actionStatus == "Scooping":
                        self.log.setTask('scooping')
                    elif self.actionStatus == "Feeding":
                        self.log.setTask('feeding')

                    print "" + self.log.task

    def guiCallback(self, msg):
        self.guiStatusReady = True
        self.gui_status = msg.data

    # --------------------------------------------------------------------------
    def run(self):
        rospy.loginfo("Continous run function called")
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            if not self.guiStatusReady:
                self.guiStatusPub.publish("select task")
                #self.availablePub.publish("true")
                print "stuck?"
                rate.sleep()
                continue
            self.debugPub1.publish(self.inputStatus)
            self.debugPub2.publish(self.emergencyStatus)
            if self.inputStatus and self.actionStatus == 'Scooping':
                self.inputStatus = False
                self.quick_feeding_ready = False
                rospy.loginfo("Scooping Starting...")
                self.scooping(self.armReachActionLeft, self.armReachActionRight, self.log, self.detection_flag)
            elif self.inputStatus and self.actionStatus == 'Feeding':
                self.inputStatus = False
                rospy.loginfo("Feeding Starting....")
                self.feeding(self.armReachActionLeft, self.armReachActionRight, self.log, self.detection_flag)
            elif self.inputStatus and self.actionStatus == 'Clean':
                self.inputStatus = False
                self.quick_feeding_ready = False
                rospy.loginfo("Clean motion...")
                # TODO: cleaning motion
                self.cleanMotion(self.armReachActionLeft, self.armReachActionRight)
                #self.logRequestPub.publish("Requesting Feedback!")
                self.guiStatusPub.publish("select task")
            rate.sleep()


    def initMotion(self, armReachActionLeft, armReachActionRight):
        rospy.loginfo("Initializing arms")
        leftProc = multiprocessing.Process(target=self.ServiceCallLeft, args=("initScooping1",))
        rightProc = multiprocessing.Process(target=self.ServiceCallRight, args=("initScooping1",))
        leftProc.start(); rightProc.start()
        leftProc.join(); rightProc.join()
        self.ScoopNumber = 1
        self.proceedPub.publish("Set: Scooping 1, Scooping 2, Scooping 3")
        #self.proceedPub.publish("Start: Scooping 1, Scooping 2")
        #self.proceedPub.publish("Next: Scooping 3")

    def scooping(self, armReachActionLeft, armReachActionRight, log, detection_flag, \
                 train=False, abnormal=False):
        while not self.emergencyStatus and not rospy.is_shutdown():

            if self.log != None:
                self.log.setTask('scooping')
                self.log.initParams()
            self.FeedNumber = 0
            
            ## Scooping -----------------------------------    
            if self.ScoopNumber < 1:
                #self.proceedPub.publish("Start: Scooping 1, Scooping 2")
                self.proceedPub.publish("Set: , Scooping 1, Scooping 2")
                rospy.loginfo("Initializing arms for scooping")
                self.initMotion(armReachActionLeft, armReachActionRight)
                if self.emergencyStatus: break
                self.ScoopNumber = 1
                self.proceedPub.publish("Set: Scooping 1, Scooping 2, Scooping 3")
                #self.proceedPub.publish("Next: Scooping 3")
        
            if self.ScoopNumber < 2:        
                self.ServiceCallLeft("getBowlPos")            
                self.ServiceCallLeft('lookAtBowl')
                if self.emergencyStatus: break
                self.ScoopNumber = 2            
                #self.proceedPub.publish("Next: Scooping 4")
                self.proceedPub.publish("Set: Scooping 2, Scooping 3, Scooping 4")
            if self.ScoopNumber < 3:        
                self.ServiceCallLeft("initScooping2")
                if self.emergencyStatus: break
                self.ScoopNumber = 3            
                #self.proceedPub.publish("Next: Done")
                self.proceedPub.publish("Set: Scooping 3, Scooping 4, Done")
    
            if self.log is not None:
                self.log.log_start()
                if detection_flag: self.log.enableDetector(True)
        
            rospy.loginfo("Running scooping!")
            self.ServiceCallLeft("runScooping")
            self.proceedPub.publish("Done")
            self.guiStatusPub.publish("request feedback")
            if self.emergencyStatus:
                if detection_flag: self.log.enableDetector(False)                
                break
            if self.log is not None:
                self.logRequestPub.publish("Requesting Feedback!")    
                if detection_flag: self.log.enableDetector(False)
                self.log.close_log_file_GUI()
            else:
                self.logRequestPub.publish("No feedback requested")
            self.ScoopNumber = 0
            break

    
    def feeding(self, armReachActionLeft, armReachActionRight, log, detection_flag):

        while not self.emergencyStatus and not rospy.is_shutdown():
            if self.log != None:
                self.log.setTask('feeding' )
                self.log.initParams()

            if self.FeedNumber < 1 and self.quick_feeding_ready is False:
                ## Feeding -----------------------------------
                rospy.loginfo("Initializing left arm for feeding")
                self.proceedPub.publish("Set: , Feeding 1, Feeding 2")
                ## self.ServiceCallLeft("lookToRight")
                ## if self.emergencyStatus: break
                self.ServiceCallLeft("initFeeding1")
                if self.emergencyStatus: break
                self.ServiceCallRight("getHeadPos")
                self.ServiceCallRight("initFeeding")
                if self.emergencyStatus: break                
                self.FeedNumber = 1
                #self.proceedPub.publish("Next: Feeding 3")
                self.proceedPub.publish("Set: Feeding 1, Feeding 2, Feeding 3")
    
            if self.FeedNumber < 2:
                rospy.loginfo("Detect a mouth")
                self.ServiceCallLeft("getHeadPos")
                self.ServiceCallLeft("initFeeding2")
                self.FeedNumber = 2
                #self.proceedPub.publish("Next: Feeding 4")
                self.proceedPub.publish("Set: Feeding 2, Feeding 3, Feeding 4")
    
            if self.FeedNumber < 3:
                rospy.loginfo("Running init feeding2")
                self.ServiceCallLeft("initFeeding3")
                if self.emergencyStatus: break
                self.FeedNumber = 3
                #self.proceedPub.publish("Next: retrieving")
                self.proceedPub.publish("Set: Feeding 3, Feeding 4, retrieving")
    
            if self.FeedNumber < 4:
                if self.log is not None:
                    self.log.log_start()
                    if detection_flag: self.log.enableDetector(True)

                rospy.loginfo("Running feeding")
                self.ServiceCallLeft("runFeeding")
                self.proceedPub.publish("Done")
                #self.proceedPub.publish("Next: Done")
                self.guiStatusPub.publish("request feedback")
                emergencyStatus = self.emergencyStatus
                if self.log is not None:
                    self.logRequestPub.publish("Requesting Feedback!")
                    print "before logging"
                    if detection_flag: self.log.enableDetector(False)
                    print "after logging"
                    self.log.close_log_file_GUI()
                    print "after log close log file"
                else:
                    self.logRequestPub.publish("No feedback requested")
                if emergencyStatus or self.emergencyStatus: break
                self.FeedNumber = 4
                    
            if self.FeedNumber < 5:
                # Returning motion
                self.ServiceCallLeft("initFeeding2")
                if self.emergencyStatus: break
                self.quick_feeding_ready = True

            self.FeedNumber = 0            
            break


    def cleanMotion(self, armReachActionLeft, armReachActionRight):
        rospy.loginfo("Initializing arms")
        self.proceedPub.publish("Set: , cleaning, ")
        self.ServiceCallLeft("getBowlPos")            
        leftProc = multiprocessing.Process(target=self.ServiceCallLeft, args=("cleanSpoon1",))
        rightProc = multiprocessing.Process(target=self.ServiceCallRight, args=("cleanSpoon1",))
        leftProc.start(); rightProc.start()
        leftProc.join(); rightProc.join()
        #self.ScoopNumber = 1
        #self.proceedPub.publish("Start: Scooping 1, Scooping 2") #TODO need to fix?
        #self.proceedPub.publish("Next: Scooping 3")
        #self.proceedPub.publish("Set: Scooping 1, Scooping 2, Scooping 3")

            
    def ServiceCallLeft(self, cmd):
        if self.left_mtx is not True:
            self.left_mtx = True
            print self.armReachActionLeft(cmd)            
            self.left_mtx = False
            return True
        else:
            print "Ignore last command...."
            return False

    def ServiceCallRight(self, cmd):
        if self.right_mtx is not True:
            self.right_mtx = True
            print self.armReachActionRight(cmd)            
            self.right_mtx = False
            return True
        else:
            print "Ignore last command...."
            return False
        
            
    def safetyMotion(self, armReachActionLeft, armReachActionRight):

        if self.actionStatus == 'Scooping':
            if self.ScoopNumber<3:
                self.initMotion(armReachActionLeft, armReachActionRight)
            else:
                self.ServiceCallLeft("initScooping2")

        elif self.actionStatus == 'Feeding':
            if self.FeedNumber<1:
                self.ServiceCallRight("initScooping1")
            elif self.FeedNumber<2:
                self.ServiceCallLeft("initFeeding1")
            elif self.FeedNumber<3:
                self.ServiceCallLeft("initFeeding2")
            elif self.FeedNumber<4:
                self.ServiceCallLeft("initFeeding3")
                self.FeedNumber = 2
                self.ServiceCallLeft("initFeeding2")
            else: 
                #print self.armReachActionLeft("runFeeding1")
                self.ServiceCallLeft("initFeeding2")
                ## self.initMotion(armReachActionLeft, armReachActionRight)

        else:
            rospy.loginfo("There is no safe motion for Init movement")


 
if __name__ == '__main__':
    
    import optparse
    p = optparse.OptionParser()
    p.add_option('--data_pub', '--dp', action='store_true', dest='bDataPub',
                 default=False, help='Continuously publish data.')
    p.add_option('--en_anomaly_detector', '--ad', action='store_true', dest='en_ad',
                 default=False, help='Enable anomaly detector.')
    p.add_option('--en_logger', '--l', action='store_true', dest='bLog',
                 default=False, help='Enable logger.')
    p.add_option('--en_quick_feeding', '--eqf', action='store_true', dest='bQuickFeeding',
                 default=False, help='Enable a quick feeding mode.')
    p.add_option('--data_path', action='store', dest='sRecordDataPath',
                 default='/home/dpark/hrl_file_server/dpark_data/anomaly/IROS2017', \
                 help='Enter a record data path')
 
    opt, args = p.parse_args()
    rospy.init_node('arm_reach_client')

    if opt.bLog or opt.bDataPub:
        # for adaptation, please add 'new' as the subject.         
        log = logger(ft=True, audio=False, audio_wrist=True, kinematics=True, vision_artag=False, \
                     vision_landmark=True, vision_change=False, pps=False, skin=False, \
                     subject="test", task='feeding', data_pub=opt.bDataPub, en_ad=opt.en_ad, \
                     record_root_path=opt.sRecordDataPath)
    else:
        log = None

    gui = armReacherGUI(detection_flag=opt.en_ad, log=log, quick_feeding=opt.bQuickFeeding)
    rospy.spin()


