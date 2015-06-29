#!/usr/bin/env python

# System
import numpy as np
import time, sys, os
import cPickle as pkl
import pandas as pd
import getpass


# ROS
import roslib
import random
roslib.load_manifest('hrl_multimodal_anomaly_detection')
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
from mpl_toolkits.mplot3d import Axes3D

from scipy.fftpack import fft

# External lib
## import yaafelib as yaafe
## from yaafelib import AudioFeature, check_dataflow_params, dataflow_safe_append, DataFlow
## import yaafefeatures as yf

def butter_bandpass(lowcut, highcut, fs, order=5):
    '''
    fs: sampling frequency
    '''
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')

    ## print low, high
    ## wp = [low, high]
    ## ws = [0., high+0.05]
    ## b, a = signal.iirdesign(wp, ws, 10, 1, ftype='butter')

    return b, a


if __name__ == '__main__':

    current_user = getpass.getuser()
    folder_name = '/home/' + current_user + '/git/hrl-assistive/hrl_multimodal_anomaly_detection/recordings/'
    change_folder = raw_input("Current folder is: %s, press [y] to change " % folder_name)
    if change_folder == 'y':
        folder_name = raw_input("Enter new folder name and press [Enter] ")
    # Current PICKLE file format method    
    pkl_file_name = raw_input("Enter exact name of pkl file, ex: [test.pkl] ")
    pkl_file_path = folder_name+pkl_file_name
    print os.path.isfile(pkl_file_path)
    d = ut.load_pickle(pkl_file_path)

    # Alternate PANDAS file format method
    # csv_file_name = raw_input("Enter exact name of csv file, ex: [test.csv] ")
    # csv_file_path = folder_name + csv_file_name
    # csv_file_path = csv_file_path.replace("\\", "\\\\")
    # df = pd.read_csv(csv_file_path)
    # print "Read CSV file as :"
    # print df

    print d.keys()

    ft_kinematics = True
    audio = False
    vision = False


    if ft_kinematics:

        ft_time = np.array(d.get('ft_time',None))
        ft_force = np.array(d.get('ft_force_raw',None))
        ft_torque = np.array(d.get('ft_torque_raw', None))

        tf_l_ee_pos = np.array(d.get('l_end_effector_pos', None))
        tf_l_ee_quat = np.array(d.get('l_end_effector_quat', None))
        tf_r_ee_pos = np.array(d.get('r_end_effector_pos', None))
        tf_r_ee_quat = np.array(d.get('r_end_effector_quat', None))

        # Graphing X Force/Pos Data
        fig1, (forceAx1, tfAx1) = pp.subplots(2, 1, sharex=True)

        forceAx1.plot(ft_time, ft_force[:,0], 'b-')
        forceAx1.set_xlabel('time (s)')
        forceAx1.set_ylabel('X Force')
        tfAx1.plot(ft_time, tf_l_ee_pos[:,0], 'r-')
        tfAx1.set_ylabel('X Pos')

        fig1.subplots_adjust(hspace=0)
        pp.setp([a.get_xticklabels() for a in fig1.axes[:-1]], visible=False)

        # Graphing Y Force/Pos Data
        fig2, (forceAx2, tfAx2) = pp.subplots(2, 1, sharex=True)

        forceAx2.plot(ft_time, ft_force[:,1], 'b-')
        forceAx2.set_xlabel('time (s)')
        forceAx2.set_ylabel('Y Force')
        tfAx2.plot(ft_time, tf_l_ee_pos[:,1], 'r-')
        tfAx2.set_ylabel('Y Pos')

        fig2.subplots_adjust(hspace=0)
        pp.setp([a.get_xticklabels() for a in fig2.axes[:-1]], visible=False)

        # Graphing Z Force/Pos Data
        fig3, (forceAx3, tfAx3) = pp.subplots(2, 1, sharex=True)

        forceAx3.plot(ft_time, ft_force[:,2], 'b-')
        forceAx3.set_xlabel('time (s)')
        forceAx3.set_ylabel('Z Force')
        tfAx3.plot(ft_time, tf_l_ee_pos[:,2], 'r-')
        tfAx3.set_ylabel('Z Pos')

        fig3.subplots_adjust(hspace=0)
        pp.setp([a.get_xticklabels() for a in fig3.axes[:-1]], visible=False)

        #pp.show()

    if audio:
        audio_data = d['audio_data']
        audio_amp = d['audio_amp']
        audio_freq = d['audio_freq']
        audio_chunk = d['audio_chunk']

        audio_data = np.array(audio_data).flatten()

        import scipy.signal as signal
        RATE    = 44100 #sampling rate
        CHUNK   = 1024 #frame per buffer

        ## b, a = butter_bandpass(1,600, RATE, order=3)
        ## audio_data = signal.lfilter(b, a, audio_data)
        ## audio_data = signal.filtfilt(b,a,audio_data,padlen=2000)

        ## for chunk in audio_data
        ##     audio_amp = np.fft.fft(audio_data / float(self.MAX_INT))  #normalization & FFT

        print np.array(audio_data).shape

        ## win = np.hamming(CHUNK)
        ## rms = []
        ## for d in audio_data:
        ##     d = np.array(d)/32768.0
        ##     windowedSignal = d*win;
        ##     fft = np.fft.fft(windowedSignal)
        ##     rms.append(np.log(np.abs(fft)))


        pp.figure()
        pp.subplot(411)
        pp.plot(audio_data,'b.')
        pp.xlabel('Time')
        pp.ylabel('Audio Data... Amplitude?')
        pp.title("Audio Data")

        pp.subplot(412)
        xs = audio_freq[:audio_chunk/16]
        ys = np.abs(audio_amp[:][:audio_chunk/16])
        ys = np.multiply(20,np.log10(ys))
        pp.plot(xs,ys,'y')
        pp.xlabel('Audio freqs')
        pp.ylabel('Amplitudes')
        pp.title("Audio Frequency and Amplitude")

        pp.subplot(413)
        pp.plot(audio_freq, 'b')
        pp.title('Raw audio_freq data')

        pp.subplot(414)
        pp.plot(audio_amp, 'y')
        pp.title('Raw audio_amp data')

        #pp.show()

    pp.show()

    if vision:
        data = d['visual_points']
        points = dict()
        for dataSet in data:
            for key, value in dataSet.iteritems():
                if key in points:
                    points[key].append(value)
                else:
                    points[key] = [value]

        fig = pp.figure()
        ax = fig.add_subplot(111, projection='3d')

        point = random.choice(points.keys())
        point = np.array(points[point])
        xs, ys ,zs = point[:, 0], point[:, 1], point[:, 2]
        ax.scatter(xs, ys, zs, c='r')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        pp.show()


	#pp.subplot(616)
	#p = 20*np.log10(np.abs(np.fft.rfft(audio_data_raw[:2048, 0])))
	#f = np.linspace(0, rate/2.0, len(p))
	#pp.plot(f, p)
	#pp.xlabel("Frequency(Hz)")
	#pp.ylabel("Power(dB)")
	#pp.title("Raw audio_data (same as wav data)")
        
	## pp.plot(rms)
        ## pp.stem(noise_freq_l, values, 'k-*', bottom=0)

        ## import pyaudio
        ## MAX_INT = 32768.0
        ## CHUNK   = 1024 #frame per buffer
        ## RATE    = 44100 #sampling rate
        ## UNIT_SAMPLE_TIME = 1.0 / float(RATE)
        ## CHANNEL=1 #number of channels
        ## FORMAT= pyaudio.paInt16
        ## DTYPE = np.int16

        ## p=pyaudio.PyAudio()
        ## stream=p.open(format=FORMAT, channels=CHANNEL, rate=RATE, \
        ##                         input=True, frames_per_buffer=CHUNK)

        ## string_audio_data = np.array(audio_data, dtype=DTYPE).tostring()
        ## import wave
        ## WAVE_OUTPUT_FILENAME = "/home/dpark/git/pyaudio/test/output.wav"
        ## wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        ## wf.setnchannels(CHANNEL)
        ## wf.setsampwidth(p.get_sample_size(FORMAT))
        ## wf.setframerate(RATE)
        ## wf.writeframes(b''.join(string_audio_data))
        ## wf.close()
