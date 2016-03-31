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

    def __init__(self):
        '''Initialize GUI'''
        #subscriber:
        self.inputSubscriber = rospy.Subscriber("/manipulation_task/user_input", String, self.inputCallback)
        self.emergencySubscriber = rospy.Subscriber("/manipulation_task/emergency", String, self.emergencyCallback)
        self.feedbackSubscriber = rospy.Subscriber("/manipulation_task/user_feedback", String, self.feedbackCallback)
        self.statusSubscriber = rospy.Subscriber("/manipulation_task/status", String, self.statusCallback)
        
        #Publisher:
        self.emergencyPub = rospy.Publisher("/hrl_manipulation_task/InterruptAction", String)
        

        #variables
        self.emergencyStatus = False
        self.inputStatus = False
        self.actionStatus = 'Both'
        self.inputMsg = None
        self.feedbackMsg = None
        self.ScoopNumber = 0
        self.FeedNumber = 0

##manipulation_task/user_input (user_feedback)(emergency)(status)

        self.Continuous()

    def inputCallback(self, data):
    #Check status, change true/false While not will be useful then. Right? emergecny stop, then initiate it.
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
        #Publish to interruptAction in server?
        self.emergencyPub.publish("STOP")
        print "Emergency received"

    def feedbackCallback(self, data):
    #Just...log? idk where this one will go. I assume it is integrated with log....
        self.feedbackMsg = data.data
        print "feedback received"

    def statusCallback(self, data):
    #Change the status, depending on message.
        self.actionStatus = data.data
        print "status received"

    def Continuous(self):
        print "Continous function called"
        while not rospy.is_shutdown():
            if self.inputStatus and self.actionStatus == 'Scooping':
                self.inputStatus = False
                print "Scooping Starting..."
               # self.testing(armReachActionLeft, armReachActionRight, log, detection_flag)
                self.scooping(armReachActionLeft, armReachActionRight, log, detection_flag)
            elif self.inputStatus and self.actionStatus == 'Feeding':
                self.inputStatus = False
                print "Feeding Starting...."
                self.feeding(armReachActionLeft, armReachActionRight, log, detection_flag)
            elif self.inputStatus and self.actionStatus == 'Both':
                self.inputStatus = False
                print "Scoop and Feed starting..."
                self.scooping(armReachActionLeft, armReachActionRight, log, detection_flag)
                self.feeding(armReachActionLeft, armReachActionRight, log, detection_flag)    
        


    def scooping(self, armReachActionLeft, armReachActionRight, log, detection_flag, train=False, abnormal=False):

        while not self.emergencyStatus and not rospy.is_shutdown():
            #log.task = 'scooping'
            #log.initParams()
        
            ## Scooping -----------------------------------    
            if self.ScoopNumber < 1:
                print "Initializing left arm for scooping"
                print armReachActionLeft("initScooping1")
                print armReachActionRight("initScooping1")
                self.ScoopNumber = 1
    
    
            if self.ScoopNumber < 2:        
            #ut.get_keystroke('Hit a key to proceed next')
                if train: 
                    print armReachActionRight("initScooping2Random")
                    if abnormal:
                        print armReachActionLeft("getBowlPosRandom")
                    else:
                        print armReachActionLeft("getBowlPos")            
                else: 
                    print armReachActionRight("initScooping2")
                    print armReachActionLeft("getBowlPos")            
                print armReachActionLeft('lookAtBowl')
                print armReachActionLeft("initScooping2")
                self.ScoopNumber = 2        
    
    
            print "Start to log!"    
            #log.log_start()
            if detection_flag: log.enableDetector(True)
        
            print "Running scooping!"
            print armReachActionLeft("runScooping")
    
            if detection_flag: log.enableDetector(False)
            print "Finish to log!"    
            #log.close_log_file()
    
            self.ScoopNumber = 0
            break

    
    def feeding(self, armReachActionLeft, armReachActionRight, log, detection_flag):

        while not self.emergencyStatus and not rospy.is_shutdown():
            #log.task = 'feeding'
            #log.initParams()
        
            if self.FeedNumber < 1:
                ## Feeding -----------------------------------
                print "Initializing left arm for feeding"
                print armReachActionLeft("initFeeding")
                ##print armReachActionRight("initFeeding") 
                self.FeedNumber = 1
    
            if self.FeedNumber < 2:
                print "Detect ar tag on the head"
                print armReachActionLeft('lookAtMouth')
                rospy.sleep(3)
                print armReachActionLeft("getHeadPos")
                #ut.get_keystroke('Hit a key to proceed next')        
                self.FeedNumber = 2
    
            if self.FeedNumber < 3:
                print "Running feeding1"    
                print armReachActionLeft("runFeeding1")
                self.FeedNumber = 3
    
            print "Start to log!"    
            #log.log_start()
            if detection_flag: log.enableDetector(True)
        
            print "Running feeding2"    
            print armReachActionLeft("runFeeding2")
    
            if detection_flag: log.enableDetector(False)
            print "Finish to log!"    
            #log.close_log_file()
    
            print armReachActionLeft("initScooping1")
            self.FeedNumber = 0
            break
         
    def testing(self, armReachActionLeft, armReachActionRight, log, detection_flag):
   

        #log.task = 'testing'
        #log.initParams()

        ## Testing --------------------------------
        print "Testing motion"
        print armReachActionLeft("testingMotion")
        print armReachActionRight("testingMotion")
        print "Testing motion Done"

 
if __name__ == '__main__':
    
    import optparse
    p = optparse.OptionParser()
    p.add_option('--data_pub', '--dp', action='store_true', dest='bDataPub',
                 default=False, help='Continuously publish data.')
    opt, args = p.parse_args()

    rospy.init_node('arm_reach_client')

    
    rospy.wait_for_service("/arm_reach_enable")
    armReachActionLeft  = rospy.ServiceProxy("/arm_reach_enable", String_String)
    armReachActionRight = rospy.ServiceProxy("/right/arm_reach_enable", String_String)



    ## rospy.sleep(2.0)    
    ## #print armReachActionLeft('lookAtMouth')
    ## print armReachActionLeft('lookAtBowl')
    
#    log = logger(ft=False, audio=False, kinematics=False, vision_artag=False, vision_change=False, \
#                 pps=False, skin=False, \
#                 subject="gatsbii", task='test', data_pub=opt.bDataPub, verbose=False)

    log = None

    last_trial  = '4'
    last_detect = '2'

    detection_flag = False

    gui = armReacherGUI()

    rospy.spin()

                 
##    while not rospy.is_shutdown():

##        detection_flag = False
        #GUI = armReacherGUI()
##        trial  = raw_input('Enter trial\'s status (e.g. 1:scooping, 2:feeding, 3: both, 4:scoopingNormalTrain, 5:scoopingAbnormalTrain, else: exit): ')
  ##      if trial=='': trial=last_trial
            
    ##    if trial is '1' or trial is '2' or trial is '3' or trial is '4' or trial is '5':
      ##      detect = raw_input('Enable anomaly detection? (e.g. 1:enable else: disable): ')
        ##    if detect == '': detect=last_detect
          ##  if detect == '1': detection_flag = True
            
            ##if trial == '1':
#                scooping(armReachActionLeft, armReachActionRight, log, detection_flag)
 #           elif trial == '4':
  #              scooping(armReachActionLeft, armReachActionRight, log, detection_flag, train=True)
   #         elif trial == '5':
    #            scooping(armReachActionLeft, armReachActionRight, log, detection_flag, train=True, abnormal=True)
     #       elif trial == '2':
      #          feeding(armReachActionLeft, armReachActionRight, log, detection_flag)
       #     else:
        #        scooping(armReachActionLeft, armReachActionRight, log, detection_flag)
         #       feeding(armReachActionLeft, armReachActionRight, log, detection_flag)
#        else:
 #           break
#
  #      last_trial  = trial
   #     last_detect = detect
    
    ## t1 = datetime.datetime.now()
    ## t2 = datetime.datetime.now()
    ## t  = t2-t1
    ## print "time delay: ", t.seconds
    

