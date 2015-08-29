#!/usr/bin/env python

import sys
import time
import rospy
from record_data import ADL_log
import onlineRecordData
import roslib
roslib.load_manifest('hrl_multimodal_anomaly_detection')
import hrl_lib.util as ut
from hrl_multimodal_anomaly_detection.srv import String_String

class dataRecord:
    AUDIO = True
    FT = True
    KINEMATICS = True

    def __init__(self):
        rospy.wait_for_service("/arm_reach_enable")
        self.armReachAction = rospy.ServiceProxy("/arm_reach_enable", String_String)
        rospy.loginfo("arm reach server connected!!")
        #This service takes requests in the form of string commands
        #Refer to arm_reacher_feeding.py for list of commands

        subject = raw_input("Enter subject name: ")
        self.task = raw_input("Run scooping, feeding, both, or exit? [s/f/b/x] ")
        while not any(self.task == s for s in ['s', 'f', 'b', 'x']):
            print "Please enter 's', 'f', 'b', or 'x' ! "
            self.task = raw_input("Run scooping, feeding, both, or exit? [s/f/b/x] ")
        if self.task == 'x':
            print "Exiting program!"
            sys.exit()

        if self.task != 'b':
            isOnline = raw_input("Perform online anomaly detection? [y] ('n' for data recording): ") == 'y'
        if self.task == 'b' or isOnline:
            self.log = onlineRecordData.ADL_log(subject=subject, task=self.task)
        else:
            self.log = ADL_log(ft=self.FT, audio=self.AUDIO, kinematics=self.KINEMATICS, subject=subject, task=self.task)

    def run(self):
        if self.task == 's':
            print "Running scooping! "
            self.scooping()
        elif self.task == 'f':
            print "Running feeding! "
            self.feeding()
        elif self.task == 'b':
            print "Scooping and feeding! "
            self.scoopAndFeed()

    def scoopAndFeed(self):
        runScooping = True
        while runScooping:
            ## Scooping -----------------------------------
            print "Initializing left arm for scooping"
            print self.armReachAction("leftArmInitScooping")

            print self.armReachAction("chooseManualBowlPos")
            ut.get_keystroke('Hit a key to proceed next')

            print 'Initializing scooping'
            print self.armReachAction('initArmScooping')

            time.sleep(2)
            self.log.log_start()

            print "Running scooping!"
            print self.armReachAction("runScooping")
            # time.sleep(1)
            self.log.close_log_file()


            ## Feeding -----------------------------------
            print "Initializing left arm for feeding"
            print self.armReachAction("leftArmInitFeeding")

            time.sleep(2.0)
            print self.armReachAction("chooseManualHeadPos")

            print 'Initializing feeding'
            print self.armReachAction('initArmFeeding')

            time.sleep(2.0)
            self.log.log_start(secondDetector=True)
            ## ut.get_keystroke('Hit a key to proceed next')

            print "Running feeding!"
            print self.armReachAction("runFeeding")

            # time.sleep(1)
            self.log.close_log_file(secondDetector=True)


            runScoopingAns = raw_input("Run process again? [y/n] ")
            while runScoopingAns != 'y' and runScoopingAns != 'n':
                print "Please enter 'y' or 'n' ! "
                runScoopingAns = raw_input("Run process again? [y/n] ")
            runScooping = runScoopingAns == 'y'

        print "Finished scooping and feeding trials!"
        print "Exiting program!"

        sys.exit()


    def scooping(self):
        runScooping = True
        while runScooping:
            print "Initializing left arm for scooping"
            print self.armReachAction("leftArmInitScooping")

            #print "Initializing right arm for scooping"
            #print self.armReachAction("rightArmInitScooping")

            # time.sleep(1)
            print self.armReachAction("chooseManualBowlPos")

            print 'Initializing scooping'
            print self.armReachAction('initArmScooping')

            time.sleep(2)
            self.log.log_start()

            print "Running scooping!"
            print self.armReachAction("runScooping")

            time.sleep(1)

            self.log.close_log_file()

            runScoopingAns = raw_input("Run scooping again? [y/n] ")
            while runScoopingAns != 'y' and runScoopingAns != 'n':
                print "Please enter 'y' or 'n' ! "
                runScoopingAns = raw_input("Run scooping again? [y/n] ")
            runScooping = runScoopingAns == 'y'

        print "Finished scooping trials!"
        print "Exiting program!"

        sys.exit()


    def feeding(self):
        # *CHOOSE HEAD POSITION!!
        #headPosType = self.armReachAction("getHeadPosType")

        #print "Initializing right arm for feeding"
        #print self.armReachAction("rightArmInitFeeding")

        runFeeding = True
        while runFeeding:
            print "Initializing left arm for feeding"
            print self.armReachAction("leftArmInitFeeding")

            print self.armReachAction("chooseManualHeadPos")

            print 'Initializing feeding'
            print self.armReachAction('initArmFeeding')

            time.sleep(2)
            self.log.log_start()

            print "Running feeding!"
            print self.armReachAction("runFeeding")

            time.sleep(1)

            self.log.close_log_file()

            runFeedingAns = raw_input("Run feeding again? [y/n] ")
            while runFeedingAns != 'y' and runFeedingAns != 'n':
                print "Please enter 'y' or 'n' ! "
                runFeedingAns = raw_input("Run feeding again? [y/n] ")
            runFeeding = runFeedingAns == 'y'

        print "Finished feeding trials!"
        print "Exiting program!"
        
        sys.exit()

if __name__ == '__main__':
    rospy.init_node('local_data_record')

    recorder = dataRecord()
    recorder.run()
