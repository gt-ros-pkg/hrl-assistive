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

#  \author You Keun Kim (Healthcare Robotics Lab, Georgia Tech.)

# system library
import time
import datetime

# ROS library
import rospy, roslib

from std_msgs.msg import String

# HRL library
from hrl_srvs.srv import String_String
import hrl_lib.util as ut

from hrl_manipulation_task.record_data import logger


class armReacherGUI:

    def __init__(self, detection_flag=False, log=None):
        '''Initialize GUI'''
        rospy.wait_for_service("/arm_reach_enable")
        self.armReachActionLeft  = rospy.ServiceProxy("/arm_reach_enable", String_String)
        self.armReachActionRight = rospy.ServiceProxy("/right/arm_reach_enable", String_String)
        
        #subscriber:
        self.inputSubscriber = rospy.Subscriber("/manipulation_task/user_input", String, self.inputCallback)
        self.emergencySubscriber = rospy.Subscriber("/manipulation_task/emergency", String, self.emergencyCallback)
        self.feedbackSubscriber = rospy.Subscriber("/manipulation_task/user_feedback", String, self.feedbackCallback)
        self.statusSubscriber = rospy.Subscriber("/manipulation_task/status", String, self.statusCallback)
        
        #Publisher:
        self.emergencyPub = rospy.Publisher("/hrl_manipulation_task/InterruptAction", String)        
        self.falselogPub = rospy.Publisher("/manipulation_task/feedbackRequest", String)

        #variables
        self.emergencyStatus = False
        self.inputStatus = False
        self.actionStatus = 'Both'
        self.inputMsg = None
        self.feedbackMsg = None
        self.ScoopNumber = 0
        self.FeedNumber = 0
        self.detection_flag = detection_flag
        self.log = log
##manipulation_task/user_input (user_feedback)(emergency)(status)

        self.Continuous()

    def inputCallback(self, data):
    #Check status, change true/false While not will be useful then. Right? emergecny stop, then initiate it.


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
    #Change the true/false.
        self.emergencyStatus = True
        self.inputStatus = False
        #Publish to interruptAction in server?
        self.emergencyPub.publish("STOP")
        print "Emergency received"
        rospy.sleep(3.0)

        rospy.wait_for_service("/arm_reach_enable")
        
        print "Aborting Sequence"
        self.testing(self.armReachActionLeft, self.armReachActionRight, self.log, self.detection_flag)
        #if self.FeedNumber<1:
        #    self.ScoopNumber = 0         
        #    print self.armReachActionLeft("initScooping1")
        #    print self.armReachActionRight("initScooping1")
        #elif self.FeedNumber<2:
        #    print self.armReachActionLeft("initFeeding")
        #else: 
        #    print self.armReachActionLeft("runFeeding1")
        #    print self.armReachActionLeft("initFeeding")

    def feedbackCallback(self, data):
    #Just...log? idk where this one will go. I assume it is integrated with log....
        self.feedbackMsg = data.data
        #print "feedback received"
        if self.feedbackMsg == "SUCCESS":
            print '1'
        elif self.feedbackMsg == "FAIL":
            print '2'
        else:
            print '3'

    def statusCallback(self, data):
    #Change the status, depending on message.
        self.actionStatus = data.data
        print "status received"
        if self.log != None:
            if self.actionStatus == "Scooping":
                self.log.setTask('scooping')
            elif self.actionStatus == "Feeding":
                self.log.setTask('feeding')

            print "" + self.log.task

    def Continuous(self):
        print "Continous function called"
        while not rospy.is_shutdown():
            if self.inputStatus and self.actionStatus == 'Scooping':
                self.inputStatus = False
                print "Scooping Starting..."
               # self.testing(self.armReachActionLeft, self.armReachActionRight, log, self.detection_flag)
                self.scooping(self.armReachActionLeft, self.armReachActionRight, self.log, self.detection_flag)
            elif self.inputStatus and self.actionStatus == 'Feeding':
                self.inputStatus = False
                print "Feeding Starting...."
                self.feeding(self.armReachActionLeft, self.armReachActionRight, self.log, self.detection_flag)
            elif self.inputStatus and self.actionStatus == 'Both':
                self.inputStatus = False
                print "Scoop and Feed starting..."
                self.scooping(self.armReachActionLeft, self.armReachActionRight, self.log, self.detection_flag)
                self.feeding(self.armReachActionLeft, self.armReachActionRight, self.log, self.detection_flag)    
        


    def scooping(self, armReachActionLeft, armReachActionRight, log, detection_flag, \
                 train=False, abnormal=False):

        while not self.emergencyStatus and not rospy.is_shutdown():
            #log.task = 'scooping'
            #log.initParams()
            if self.log != None:
                self.log.setTask('scooping')
                self.log.initParams()
            self.FeedNumber = 0
            ## Scooping -----------------------------------    
            if self.ScoopNumber < 1:
                print "Initializing left arm for scooping"
                try:
                    armReachActionLeft("initScooping1")
                except rospy.ServiceException, e:
                    print "================ Service call failed: %s ==============="%e     
                if self.emergencyStatus: break
                print armReachActionRight("initScooping1")
                if self.emergencyStatus: break
                self.ScoopNumber = 1
    
    
            if self.ScoopNumber < 2:        
            #ut.get_keystroke('Hit a key to proceed next')
                if train: 
                    print armReachActionRight("initScooping2Random")
                    if self.emergencyStatus: break
                    if abnormal:
                        print armReachActionLeft("getBowlPosRandom")
                    else:
                        print armReachActionLeft("getBowlPos")            
                else: 
                    print armReachActionRight("initScooping2")
                    if self.emergencyStatus: break
                    print armReachActionLeft("getBowlPos")            
                print armReachActionLeft('lookAtBowl')
                if self.emergencyStatus: break
                print armReachActionLeft("initScooping2")
                if self.emergencyStatus: break
                self.ScoopNumber = 2        
    
    
            print "Start to log!"    
            if self.log != None:
                self.log.log_start()
            if detection_flag: self.log.enableDetector(True)
        
            print "Running scooping!"
            print armReachActionLeft("runScooping")
            if self.emergencyStatus: break
            print self.log
            print self.log==None
            if self.log == None:
                self.falselogPub.publish("Requesting Feedback!")
    
            if detection_flag: self.log.enableDetector(False)
            print "Finish to log!"    
            if self.log != None:
                self.log.close_log_file_GUI()
    
            self.ScoopNumber = 0
            break

    
    def feeding(self, armReachActionLeft, armReachActionRight, log, detection_flag):

        while not self.emergencyStatus and not rospy.is_shutdown():
            if self.log != None:
                self.log.setTask('feeding' )
                self.log.initParams()

            if self.FeedNumber < 1:
                #self.FeedNumber = 0.5
                ## Feeding -----------------------------------
                print "Initializing left arm for feeding"
                try:
                    armReachActionLeft("initFeeding")
                except:
                    print "service call error !!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                if self.emergencyStatus: break
                ##print armReachActionRight("initFeeding") 
                self.FeedNumber = 1
    
            if self.FeedNumber < 2:
                print "Detect ar tag on the head"
                print armReachActionLeft('lookAtMouth')
                if self.emergencyStatus: break
                rospy.sleep(2)
                print armReachActionLeft("getHeadPos")
                print armReachActionLeft("getHeadPos")
                #ut.get_keystroke('Hit a key to proceed next')        
                self.FeedNumber = 2
    
            if self.FeedNumber < 3:
                print "Running feeding1"    
                print armReachActionLeft("runFeeding1")
                if self.emergencyStatus: break
                self.FeedNumber = 3
    
            print "Start to log!"    
            if self.log != None:
                self.log.log_start()
            if detection_flag: self.log.enableDetector(True)
        
            print "Running feeding2"    
            print armReachActionLeft("runFeeding2")
            if self.emergencyStatus: break
            
            if self.log == None:
                self.falselogPub.publish("Requesting Feedback!")

    
            if detection_flag: self.log.enableDetector(False)
            print "Finish to log!"    
            if self.log != None:
                self.log.close_log_file_GUI()
    
            print armReachActionLeft("initScooping1")
            if self.emergencyStatus: break
            self.FeedNumber = 0
            break

        
    ## def ServiceCallLeft(self, cmd):
    ##     if self.left_mtx is not True:
    ##         self.left_mtx = True
    ##         self.armReachActionLeft(cmd)            
    ##         self.left_mtx = False
    ##     else:
    ##         print "Ignore last command...."
        
            
    def testing(self, armReachActionLeft, armReachActionRight, log, detection_flag):
        if self.FeedNumber<1:
            self.ScoopNumber = 0         
            self.armReachActionLeft("initScooping1")
            self.armReachActionRight("initScooping1")
        elif self.FeedNumber<2:
            self.armReachActionLeft("initFeeding")
        else: 
            #print self.armReachActionLeft("runFeeding1")
            self.armReachActionLeft("initFeeding")



        #log.task = 'testing'
        #log.initParams()

        ## Testing --------------------------------
        #print "Testing motion"
        #print armReachActionLeft("testingMotion")
        #print armReachActionRight("testingMotion")
        #print "Testing motion Done"

 
if __name__ == '__main__':
    
    import optparse
    p = optparse.OptionParser()
    p.add_option('--data_pub', '--dp', action='store_true', dest='bDataPub',
                 default=False, help='Continuously publish data.')
    p.add_option('--en_anomaly_detector', '--ad', action='store_true', dest='bAD',
                 default=False, help='Enable anomaly detector.')
    p.add_option('--en_logger', '--l', action='store_true', dest='bLog',
                 default=False, help='Enable logger.')
 
    opt, args = p.parse_args()
    rospy.init_node('arm_reach_client')

    ## rospy.sleep(2.0)    
    ## #print armReachActionLeft('lookAtMouth')
    ## print armReachActionLeft('lookAtBowl')
    
    if opt.bLog:
        log = logger(ft=True, audio=False, audio_wrist=True, kinematics=True, vision_artag=True, \
                     vision_change=False, pps=False, skin=False, \
                     subject="GUI_Testing", task='scooping', data_pub=opt.bDataPub, detector=opt.bAD, \
                     verbose=False)
    else:
        log = None

    last_trial  = '4'
    last_detect = '2'

    gui = armReacherGUI(detection_flag=opt.bAD, log=log)
    rospy.spin()


