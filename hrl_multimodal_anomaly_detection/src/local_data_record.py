#!/usr/bin/env python

from record_data import *
import roslib
roslib.load_manifest('hrl_multimodal_anomaly_detection')
#* Added by Hyder
from hrl_srvs.srv import None_Bool, None_BoolResponse, Int_Int
from hrl_python_servicer.srv import String_String

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
        task = raw_input("Enter task name: ")
        actor = raw_input("Enter actor name: ")
        self.trial_name = raw_input("Enter trial name: ")

        self.log = ADL_log(audio=self.AUDIO, audioRecord=self.AUDIORECORD, vision=self.VISION, ft=self.FT, kinematics=self.KINEMATICS,
                                manip=self.MANIP, test_mode=self.TEST_MODE, subject=subject, task=task, actor=actor)

        #This should only run when MANIP = False, since log file isn't closed by ADL_log itself...
        repeatAns = raw_input("Change trial name before starting? [y/n]")
        while repeatAns != 'n':
            self.trial_name = raw_input("Enter trial name: ")
            repeatAns = raw_input("Change trial name before starting? [y/n]")

        self.log.log_start(self.trial_name)

    def run(self):

        whichTask = raw_input("Run scooping, feeding, or exit? [s/f/x] ")
        while whichTask != 's' and whichTask != 'f' and whichTask != 'x':
            print "Please enter 's' or 'f' or 'x' ! "
            whichTask = raw_input("Run scooping, feeding, or exit? [s/f/x] ")
        if whichTask == 's':
            print "Running scooping! "
            self.scooping()
        elif whichTask == 'f':
            print "Running feeding! "
            self.feeding()
        elif whichTask == 'x':
            print "Exiting program! "
            sys.exit()

    def scooping(self):

        # *CHOOSE BOWL POSITION!!
        #bowlPosType = self.armReachAction("getBowlPosType")

        print self.armReachAction("chooseManualBowlPos")

        runScooping = True
        while runScooping:
            print "Initializing left arm for scooping"
            print self.armReachAction("leftArmInitScooping")

            print "Initializing right arm for scooping"
            print self.armReachAction("rightArmInitScooping")

            time.sleep(1)

            # * Open log file with new name

            print "Running scooping!"
            print self.armReachAction("runScooping")
            #print "Finished running scooping!"

            # * Close log file with same new name

            runScoopingAns = raw_input("Run scooping again? [y/n] ")
            while runScoopingAns != 'y' and runScoopingAns != 'n':
                print "Please enter 'y' or 'n' ! "
                runScoopingAns = raw_input("Run scooping again? [y/n] ")
            if runScoopingAns == 'y':
                runScooping = True
            elif runScoopingAns == 'n':
                runScooping = False

        print "Finished scooping trials!"
        #sys.exit()


    def feeding(self):

        # *CHOOSE HEAD POSITION!!
        #headPosType = self.armReachAction("getHeadPosType")

        print self.armReachAction("chooseManualHeadPos")

        runFeeding = True
        while runFeeding:
            print "Initializing left arm for feeding"
            print self.armReachAction("leftArmInitFeeding")

            print "Initializing right arm for feeding"
            print self.armReachAction("rightArmInitFeeding")

            time.sleep(1)

            # * Open log file with new name

            print "Running feeding!"
            print self.armReachAction("runFeeding")
            #print "Finished running feeding!"

            # * Close log file with same new name

            runFeedingAns = raw_input("Run feeding again? [y/n] ")
            while runFeedingAns != 'y' and runFeedingAns != 'n':
                print "Please enter 'y' or 'n' ! "
                runFeedingAns = raw_input("Run feeding again? [y/n] ")
            if runFeedingAns == 'y':
                runFeeding = True
            elif runFeedingAns == 'n':
                runFeeding = False

        print "Finished feeding trials!"
        #sys.exit()

if __name__ == '__main__':
    rospy.init_node('local_data_record')

    recorder = dataRecord()
    recorder.run()

    #rospy.spin()
