#!/usr/local/bin/python                                                                                          

# System library
import sys
import os
import time
import math
import numpy as np
import glob

# ROS library
import roslib; roslib.load_manifest('hrl_anomaly_detection') 
import hrl_lib.util as ut
import matplotlib.pyplot as plt

#
import hrl_lib.circular_buffer as cb
import sandbox_dpark_darpa_m3.lib.hrl_dh_lib as hdl


class anomaly_checker():

    def __init__(self, nMaxBuf, nDim=1, fXInterval=1.0, fXMax=90.0):

        # Variables
        self.nMaxBuf    = nMaxBuf
        self.nDim       = nDim        
        self.fXInterval = fXInterval
        self.fXMax      = fXMax
        self.aXRange    = np.arange(0.0,fXMax,self.fXInterval)
        self.fXTOL      = 10e-1
        
        # N-buffers
        self.buf_dict = {}
        for i in xrange(self.nMaxBuf):
            self.buf_dict['buf_'+str(i)] = cb.CircularBuffer(self.nMaxBuf, (nDim,))       

        # x buffer
        self.x_buf = cb.CircularBuffer(self.nMaxBuf, (1,))        
        self.x_buf.append(-1.0)
        
        pass

        
    def update_buffer(self, x, y)

        x_c = hdl.find_nearest(self.aXRange, x, sup=True)

        if x - x_c < self.fXTOL:
        
        x_buf = self.x_buf.get_array()



        
        # Skip: assumption that x is an increasing variable
        if x-self.x_buf[-1] < self.fXInterval:
            return
        elif x-self.x_buf[-1]
                
        return

        
    def find_sup(self, x):
