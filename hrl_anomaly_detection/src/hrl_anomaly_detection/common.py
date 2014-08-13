#!/usr/local/bin/python                                                                                          

# System library
import sys
import os
import time
import numpy as np

# ROS library
import roslib; roslib.load_manifest('hrl_anomaly_detection') 
import hrl_lib.util as ut


def getAllFiles(dirName):

    lFile = []
    for root, dirs, files in os.walk(dirName):

        if root.find('.svn') >= 0: continue
                
        for sub_dir in dirs:
            if sub_dir.find('.svn') >= 0: continue
            lFile = lFile + getAllFiles(sub_dir)

        for sub_file in files:
            lFile.append(os.path.join(root,sub_file))

    return lFile
