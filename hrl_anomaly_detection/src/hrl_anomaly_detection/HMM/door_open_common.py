#!/usr/local/bin/python

import sys, os, copy
import numpy as np, math
import glob

import roslib; roslib.load_manifest('hrl_anomaly_detection')
import rospy

class_list = ['Freezer','Fridge','Kitchen Cabinet','Office Cabinet']
class_dir_list = ['Freezer','Fridge','Kitchen_Cabinet','Office_Cabinet']
