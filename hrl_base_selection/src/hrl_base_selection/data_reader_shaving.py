#!/usr/bin/env python

import roslib
roslib.load_manifest('hrl_base_selection')
roslib.load_manifest('hrl_haptic_mpc')
roslib.load_manifest('hrl_lib')
import numpy as np
import math as m
import time
import roslib
import rospy
from helper_functions import createBMatrix, Bmat_to_pos_quat
from data_reader import DataReader


class DataReader_Shaving(object):
    def __init__(self, visualize=False, subject=0, task='shaving', model='chair'):
        self.visualize = visualize
        self.model = model
        self.task = task
        self.subject = subject
        self.goals = []
        self.reset_goals()

    def reset_goals(self):
        liftlink_B_head = createBMatrix([1.249848, -0.013344, 0.1121597], [0.044735, -0.010481, 0.998626, -0.025188])

        liftlink_B_goal = [[1.107086, -0.019988, 0.014680, 0.011758, 0.014403, 0.031744, 0.999323],
                           [1.089931, -0.023529, 0.115044, 0.008146, 0.125716, 0.032856, 0.991489],
                           [1.123504, -0.124174, 0.060517, 0.065528, -0.078776, 0.322874, 0.940879],
                           [1.192543, -0.178014, 0.093005, -0.032482, 0.012642, 0.569130, 0.821509],
                           [1.194537, -0.180350, 0.024144, 0.055151, -0.113447, 0.567382, 0.813736],
                           [1.103003, 0.066879, 0.047237, 0.023224, -0.083593, -0.247144, 0.965087],
                           [1.180539, 0.155222, 0.061160, -0.048171, -0.076155, -0.513218, 0.853515],
                           [1.181696, 0.153536, 0.118200, 0.022272, 0.045203, -0.551630, 0.832565]]
        self.goals = []
        for i in xrange(len(liftlink_B_goal)):
            self.goals.append(liftlink_B_head.I*createBMatrix(liftlink_B_goal[i][0:3], liftlink_B_goal[i][3:]))  # all in reference to head

        self.goals = np.array(self.goals)
        return self.goals

    def generate_score(self):

        # for item in data:
        #     print Bmat_to_pos_quat(item)

        num = np.ones([len(self.goals), 1])
        reference_options = ['head']
        reference = np.zeros([len(self.goals), 1])

        print 'Starting to convert data!'
        run_data = DataReader(subject=self.subject, model=self.model, task=self.task)
        run_data.receive_input_data(self.goals, num, reference_options, reference)
        run_data.generate_output_goals()
        run_data.generate_score(viz_rviz=True, visualize=self.visualize, plot=False)

if __name__ == "__main__":
    model = 'autobed'  # options are: 'chair', 'bed', 'autobed'
    task = 'shaving'
    subject = 'any_subject'
    rospy.init_node(''.join(['data_reader_', subject, '_', model, '_', task]))
    start_time = time.time()
    shaving_data_reader = DataReader_Shaving(model=model, task=task, subject=None)
    shaving_data_reader.generate_score()
    print 'Done! Time to generate all scores: %fs' % (time.time() - start_time)
