#!/usr/bin/env python

import roslib; roslib.load_manifest('sandbox_dpark_darpa_m3')
roslib.load_manifest('hrl_summer_2015')
import rospy
import numpy as np, math
import time
import tf

from record_data import *


class audioRecord():
    def __init__(self):
        subject = 'gatsbii'
        task = '10'
        actor = '2'

        self.log = ADL_log(audio=True, ft=False, kinematics=False,  manip=False, test_mode=False)
        self.log.init_log_file(subject, task, actor)

        self.trial_name = raw_input("Enter trial name: ")

        print self.trial_name

        repeatAns = raw_input("Change trial name before starting? [y/n]")
        while repeatAns != 'n':
            self.trial_name = raw_input("Enter trial name: ")
            repeatAns = raw_input("Change trial name before starting? [y/n]")

        self.log.log_start(self.trial_name)

    def run(self):
        print "Test 1"
        ans = raw_input("Press 'x' to stop logging")
        if ans == 'x':
            self.log.close_log_file(self.trial_name)



if __name__ == '__main__':
    rospy.init_node('local_audio_record')

    recorder = audioRecord()
    recorder.run()

    rospy.spin()
