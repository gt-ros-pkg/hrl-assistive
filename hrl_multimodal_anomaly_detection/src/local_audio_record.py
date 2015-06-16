#!/usr/bin/env python

import roslib; roslib.load_manifest('sandbox_dpark_darpa_m3')
roslib.load_manifest('hrl_multimodal_anomaly_detection')
import rospy
import numpy as np, math
import time
import tf

from record_data import *
from load_data import *


class audioRecord():
    def __init__(self):

        self.log = ADL_log(audio=True, audioRecord=True, ft=False, kinematics=False,  manip=False, test_mode=False)

        self.subject = raw_input("Enter subject name: ")
        self.task = raw_input("Enter task name: ")
        self.actor = raw_input("Enter actor name: ")
        self.trial_name = raw_input("Enter trial name: ")

        self.log.init_log_file(self.subject, self.task, self.actor)

        repeatAns = raw_input("Change trial name before starting? [y/n]")
        while repeatAns != 'n':
            self.trial_name = raw_input("Enter trial name: ")
            repeatAns = raw_input("Change trial name before starting? [y/n]")

        self.log.log_start(self.trial_name)


    def run(self):
        ans = raw_input("Press 'x' to stop logging: ")
        while ans != 'x':
            ans = raw_input("Press 'x' to stop logging: ")
        print "Closing audio log file"
        self.log.close_log_file(self.trial_name)
                # break

        sys.exit()

if __name__ == '__main__':
    rospy.init_node('local_audio_record')

    recorder = audioRecord()
    recorder.run()

    rospy.spin()
