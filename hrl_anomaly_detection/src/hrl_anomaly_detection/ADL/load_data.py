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
        pp.plot(audio_freq[:audio_chunk/10],np.log(np.abs(audio_amp[:audio_chunk/10])),'b')
        ## pp.stem(noise_freq_l, values, 'k-*', bottom=0)        
        pp.show()

        import pyaudio        
        MAX_INT = 32768.0
        CHUNK   = 1024 #frame per buffer
        RATE    = 44100 #sampling rate
        UNIT_SAMPLE_TIME = 1.0 / float(RATE)
        CHANNEL=1 #number of channels
        FORMAT=pyaudio.paInt16
        DTYPE = np.int16

        p=pyaudio.PyAudio()
        stream=p.open(format=FORMAT, channels=CHANNEL, rate=RATE, \
                                input=True, frames_per_buffer=CHUNK)
        
        string_audio_data = np.array(audio_data, dtype=DTYPE).tostring() 
        import wave
        WAVE_OUTPUT_FILENAME = "/home/dpark/git/pyaudio/test/output.wav"
        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNEL)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(string_audio_data))
        wf.close()
        
