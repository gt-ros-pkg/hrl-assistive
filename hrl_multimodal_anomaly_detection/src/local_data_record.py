#!/usr/bin/env python

from record_data import *
import roslib
roslib.load_manifest('hrl_multimodal_anomaly_detection')
from hrl_multimodal_anomaly_detection.srv import String_String

class dataRecord:

    AUDIO = True
    AUDIORECORD = True
    FT = True
    VISION = True
    KINEMATICS = True
    TEST_MODE = False


    def __init__(self):

        rospy.wait_for_service("/arm_reach_enable")
        self.armReachAction = rospy.ServiceProxy("/arm_reach_enable", String_String)
        rospy.loginfo("arm reach server connected!!")

        subject = raw_input("Enter subject name: ")
        self.task = raw_input("Run scooping, feeding, or exit? [s/f/x] ")
        while self.task != 's' and self.task != 'f' and self.task != 'x':
            print "Please enter 's' or 'f' or 'x' ! "
            self.task = raw_input("Run scooping, feeding, or exit? [s/f/x] ")
        if self.task == 'x':
            print "Exiting program!"
            sys.exit()

        self.log = ADL_log(ft=self.FT, audio=self.AUDIO, vision=self.VISION, kinematics=self.KINEMATICS, subject=subject, task=self.task)

        #This should only run when MANIP = False, since log file isn't closed by ADL_log itself...
        # repeatAns = raw_input("Change trial name before starting? [y/n]")
        # while repeatAns != 'n':
        #     self.trial_name = raw_input("Enter trial name: ")
        #     repeatAns = raw_input("Change trial name before starting? [y/n]")

    def run(self):

        if self.task == 's':
            print "Running scooping! "
            self.scooping()
        elif self.task == 'f':
            print "Running feeding! "
            self.feeding()

    def scooping(self):

        # *CHOOSE BOWL POSITION!!
        #bowlPosType = self.armReachAction("getBowlPosType")

        print self.armReachAction("chooseManualBowlPos")

        runScooping = True
        while runScooping:
            print "Initializing left arm for scooping"
            print self.armReachAction("leftArmInitScooping")

            #print "Initializing right arm for scooping"
            #print self.armReachAction("rightArmInitScooping")

            time.sleep(1)

            self.log.log_start()

            print "Running scooping!"
            print self.armReachAction("runScooping")
            #print "Finished running scooping!"

            self.log.close_log_file()

            runScoopingAns = raw_input("Run scooping again? [y/n] ")
            while runScoopingAns != 'y' and runScoopingAns != 'n':
                print "Please enter 'y' or 'n' ! "
                runScoopingAns = raw_input("Run scooping again? [y/n] ")
            if runScoopingAns == 'y':
                runScooping = True
            elif runScoopingAns == 'n':
                runScooping = False

        print "Finished scooping trials!"


    def feeding(self):

        # *CHOOSE HEAD POSITION!!
        #headPosType = self.armReachAction("getHeadPosType")

        print self.armReachAction("chooseManualHeadPos")

        runFeeding = True
        while runFeeding:
            print "Initializing left arm for feeding"
            print self.armReachAction("leftArmInitFeeding")

            #print "Initializing right arm for feeding"
            #print self.armReachAction("rightArmInitFeeding")

            time.sleep(1)

            self.log.log_start()

            print "Running feeding!"
            print self.armReachAction("runFeeding")
            #print "Finished running feeding!"

            self.log.close_log_file()

            runFeedingAns = raw_input("Run feeding again? [y/n] ")
            while runFeedingAns != 'y' and runFeedingAns != 'n':
                print "Please enter 'y' or 'n' ! "
                runFeedingAns = raw_input("Run feeding again? [y/n] ")
            if runFeedingAns == 'y':
                runFeeding = True
            elif runFeedingAns == 'n':
                runFeeding = False

        print "Finished feeding trials!"

if __name__ == '__main__':
    rospy.init_node('local_data_record')

    recorder = dataRecord()
    recorder.run()
