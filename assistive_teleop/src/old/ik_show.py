#!/usr/bin/python

import roslib; roslib.load_manifest('web_teleop_trunk')
import rospy
import pickle
import pprint

from geometry_msgs.msg import PoseStamped

class ik_results_display():
    def __init__(self):
        rospy.init_node('ik_results_viewer')
        pickle_file = open('ik_pickle.pkl', 'rb')
        #pickle_file = open('w1_data.pkl', 'rb')
        data_in = pickle.load(pickle_file)
        pprint.pprint(data_in)
        #print data_in[0][:]



if __name__=='__main__':
    ird = ik_results_display()

