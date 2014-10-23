#!/usr/local/bin/python

import sys, os, glob
import numpy as np, math
import roslib; roslib.load_manifest('hrl_anomaly_detection')
import rospy
import inspect
import warnings
import random

# Util
import hrl_lib.util as ut

if __name__ == '__main__':


    data_path = '/home/dpark/hrl_file_server/dpark_data/anomaly/RSS2015/door_tune'

    file_list = glob.glob(data_path+'/*.pkl')
    for pkl_file in file_list:
        try:
            data = ut.load_pickle(pkl_file)
        except:
            print "check file: "+pkl_file
            print "failed to load pickle, corrupted? ..."
            failure_file = os.path.join(data_path,'failure_'+pkl_file+'.txt')
            touch(failure_file)
            ## os.system('rm *') 
            sys.exit()
        if data == None:
            print "failed to load pickle, none there? ..."
            failure_file = os.path.join(data_path,'failure_'+pkl_file+'.txt')
            touch(failure_file)            
            sys.exit()


        print data
