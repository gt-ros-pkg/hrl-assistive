#!/usr/bin/env python

# System
import numpy as np
import time, sys, threading
import cPickle as pkl


# ROS
import roslib
roslib.load_manifest('hrl_anomaly_detection')
roslib.load_manifest('geometry_msgs')
roslib.load_manifest('hrl_lib')
import rospy, optparse, math, time
import tf
from geometry_msgs.msg import Wrench
from geometry_msgs.msg import TransformStamped, WrenchStamped
from std_msgs.msg import Bool

# HRL
import hrl_lib.util as ut
import matplotlib.pyplot as pp
import matplotlib as mpl


if __name__ == '__main__':

    pkl_file = './test_cup_human_t1.pkl'
    pkl_file = './test.pkl'
    d = ut.load_pickle(pkl_file)

    print d.keys()
    ft = False
    audio = True
    
    if ft:
        ftime = d.get('ftime',None)
        force = d.get('force_raw',None)

        aForce = np.squeeze(force).T
        print aForce.shape

        pp.figure()
        pp.plot(ftime, aForce[0])
        pp.show()

    if audio:
        audio_data = d['audio_data']
        audio_amp = d['audio_amp']
        audio_freq = d['audio_freq']
        audio_chunk = d['audio_chunk']

        pp.figure()        
        pp.subplot(211)
        pp.plot(audio_data,'b-')
        
        pp.subplot(212)
        pp.plot(audio_freq[:audio_chunk/10],np.abs(audio_amp[:audio_chunk/10]),'b')
        ## pp.stem(noise_freq_l, values, 'k-*', bottom=0)        
        pp.show()
        
        
