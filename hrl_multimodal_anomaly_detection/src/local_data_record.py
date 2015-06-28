#!/usr/bin/env python

import roslib; roslib.load_manifest('sandbox_dpark_darpa_m3')
roslib.load_manifest('hrl_multimodal_anomaly_detection')
import rospy

from record_data import *
from load_data import *

class dataRecord():

    AUDIO = True
    AUDIORECORD = True
    FT = True
    VISION = True
    KINEMATICS = True
    MANIP = True
    TEST_MODE = False


    def __init__(self):

        self.subject = raw_input("Enter subject name: ")
        self.task = raw_input("Enter task name: ")
        self.actor = raw_input("Enter actor name: ")
        self.trial_name = raw_input("Enter trial name: ")

        self.log = ADL_log(audio=self.AUDIO, audioRecord=self.AUDIORECORD, vision=self.VISION,  ft=self.FT, kinematics=self.KINEMATICS,  manip=self.MANIP, test_mode=self.TEST_MODE)

        self.log.init_log_file(self.subject, self.task, self.actor)

        #This should only run when MANIP = False, since log file isn't closed by ADL_log itself...
        repeatAns = raw_input("Change trial name before starting? [y/n]")
        while repeatAns != 'n':
            self.trial_name = raw_input("Enter trial name: ")
            repeatAns = raw_input("Change trial name before starting? [y/n]")

        self.log.log_start(self.trial_name)


    def run(self):
        ans = raw_input("Press 'x' to stop logging: ")
        while ans != 'x':
            ans = raw_input("Press 'x' to stop logging: ")
        self.log.close_log_file(self.trial_name)
        print "Closing log file"
                # break

        sys.exit()

if __name__ == '__main__':
    rospy.init_node('local_data_record')

    recorder = dataRecord()
    recorder.run()

    rospy.spin()
