#!/usr/bin/env python

import numpy as np
import math as m
import copy

import time
import roslib
roslib.load_manifest('hrl_base_selection')
roslib.load_manifest('hrl_haptic_mpc')
import rospy, rospkg
from helper_functions import createBMatrix, Bmat_to_pos_quat
import random

import pickle as pkl
roslib.load_manifest('hrl_lib')
from hrl_lib.util import save_pickle
from random import gauss
import hrl_lib.util as ut


class SkelParser(object):
    def __init__(self):
        print 'Skeleton Parser is ready!'

    def parse_skel(self, file_path):
        print file_path

        spheres = []
        links = []
        print 'File has been parsed'
        return spheres, links

if __name__ == "__main__":
    rospy.init_node('score_generator')
    sp = SkelParser()
    folder_path = '/home/ari/git/catkin_ws/src/hrl-assistive/hrl_base_selection/models/'
    file_name = 'fullbody_alex_capsule.skel'
    sp.parse_skel(folder_path+file_name)




