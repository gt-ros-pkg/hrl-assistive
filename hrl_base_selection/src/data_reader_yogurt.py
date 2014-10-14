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


class DataReader_Yogurt(object):
    def __init__(self, visualize=False, subject='any_subject', task='yogurt', model='chair', tf_listener=None):

        self.model = model
        self.task = task
        self.subject = subject

        baselink_B_liftlink = createBMatrix([-0.05, 0.0, 0.8897], [0, 0, 0, 1])

        goals = [[0.301033944729, 0.461276517595, 0.196885866571,
                  0.553557277528, 0.336724229346, -0.075691681684, 0.757932650828],
                 [0.377839595079, 0.11569018662, 0.0419789999723,
                  0.66106069088, 0.337429642677, -0.519856214523, 0.422953367233],
                 [0.2741387011303321, 0.005522571699560719, -0.011919598309888757,
                 -0.023580897114171894, 0.7483633417869068, 0.662774596931439, 0.011228696415565394],
                 [0.13608632401364894, 0.003540318703608347, 0.00607600258150498,
                  -0.015224467044577382, 0.7345761465214938, 0.6783020152473445, -0.008513323454022942]]
        liftlink_B_goal = createBMatrix([0.5309877259429142, 0.4976163448816489, 0.16719537682372823],
                                        [0.7765742993649133, -0.37100605554316285, -0.27784851903166524,
                                         0.42671660945891])
        data = np.array([baselink_B_liftlink*createBMatrix(goals[0][0:3], goals[0][3:]),  # In reference to base link
                         baselink_B_liftlink*createBMatrix(goals[1][0:3], goals[1][3:]),  # In reference to base link
                         createBMatrix(goals[2][0:3], goals[2][3:]),
                         createBMatrix(goals[3][0:3], goals[3][3:])])  # This one is in reference to the head

        for item in data:
            print Bmat_to_pos_quat(item)

        # For my information, these are the [xyz] and quaternion [x,y,z,w] for the PoseStamped messages for the goal
        # positions. The first two have parent tf /base_link. The third has parent link /head
        # (array([ 0.48098773,  0.49761634,  0.91837238]), array([ 0.7765743 , -0.37100606, -0.27784852,  0.42671661]))
        # (array([ 0.4598544 ,  0.8806009 ,  0.65371782]), array([ 0.45253993,  0.53399713, -0.17283745,  0.69295158]))
        # (array([ 0.2741387 ,  0.05522572, -0.0119196 ]), array([-0.0235809 ,  0.74836334,  0.6627746 ,  0.0112287 ]))

        num = np.ones([len(data), 1])
        reference_options = ['head', 'base_link']
        reference = np.array([[1], [1], [0], [0]])



        print 'Starting to convert data!'
        runData = DataReader(subject=self.subject, model=self.model, task=self.task)

        runData.receive_input_data(data, num, reference_options, reference)
        runData.generate_output_goals()
        runData.generate_score(viz_rviz=True, visualize=False, plot=False)

if __name__ == "__main__":
    model = 'chair'  # options are: 'chair', 'bed', 'autobed'
    task = 'yogurt'
    subject = 'any_subject'
    rospy.init_node(''.join(['data_reader_', subject, '_', model, '_quick_', task]))
    start_time = time.time()
    yogurt_data_reader = DataReader_Yogurt(model=model, task=task, subject=subject)
    print 'Done! Time to generate all scores: %fs' % (time.time() - start_time)
    rospy.spin()


















