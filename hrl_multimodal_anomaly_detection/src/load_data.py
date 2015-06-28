#!/usr/bin/env python

# System
import numpy as np
import time, sys, os
import cPickle as pkl
import pandas as pd
import getpass


# ROS
import roslib
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

        ft_time = d.get('ft_time',None)
        ft_force = d.get('ft_force_raw',None)
        ft_torque = d.get('ft_torque_raw', None)

        tf_l_ee_pos = d.get('l_end_effector_pos', None)
        tf_l_ee_quat = d.get('l_end_effector_quat', None)
        tf_r_ee_pos = d.get('r_end_effector_pos', None)
        tf_r_ee_quat = d.get('r_end_effector_quat', None)


        ft_time_array = np.array(ft_time)
        ft_force_array = np.array(ft_force)
        ft_torque_array = np.array(ft_torque)

        tf_l_ee_pos_array = np.array(tf_l_ee_pos)
        tf_l_ee_quat_array = np.array(tf_l_ee_quat)
        tf_r_ee_pos_array = np.array(tf_r_ee_pos)
        tf_r_ee_quat_array = np.array(tf_r_ee_quat)

        # aForce = np.squeeze(force).T
        # print aForce.shape

        # force_array = np.array(force)
        # torque_array = np.array(torque)

        pp.figure()
        pp.subplot(311)
        pp.plot(ftime, force_array[:,0])
        pp.title('Force X')

        pp.subplot(312)
        pp.plot(ftime, force_array[:,1])
        pp.title('Force Y')

        pp.subplot(313)
        pp.plot(ftime, force_array[:,2])
        pp.title('Force Z')



        fig, ax1 = pp.subplots()

        # t = np.arange(0.01, 10.0, 0.01)
        # s1 = np.exp(t)

        ax1.plot(ft_time, ft_force_array[:,0], 'b-')
        ax1.set_xlabel('time (s)')
        # Make the y-axis label and tick labels match the line color.
        ax1.set_ylabel('Force X', color='b')
        for tl in ax1.get_yticklabels():
            tl.set_color('b')

        ax2 = ax1.twinx()
        s2 = np.sin(2*np.pi*t)
        ax2.plot(t, s2, 'r.')
        ax2.set_ylabel('sin', color='r')
        for tl in ax2.get_yticklabels():
            tl.set_color('r')
        pp.show()








    if ft:
        

        pp.show()

    if kinematics:
        kinematics_time = d.get('kinematics_time',None)
        l_end_effector_pos = d.get('l_end_effector_pos')
        l_end_effector_quat = d.get('l_end_effector_quat')
        r_end_effector_pos = d.get('r_end_effector_pos')
        r_end_effector_quat = d.get('r_end_effector_quat')

        l_end_effector_pos_array = np.array(l_end_effector_pos)
        l_end_effector_quat_array = np.array(l_end_effector_quat) 
        r_end_effector_pos_array = np.array(r_end_effector_pos)
        r_end_effector_quat_array = np.array(r_end_effector_quat) 

        pp.figure()
        pp.subplot(321)
        pp.plot(kinematics_time, r_end_effector_pos_array[:,0], 'r')
        pp.plot(ftime, force_array[:,0], 'y')
        pp.title('R EE X - Red, F X - Yellow')

        pp.subplot(322)
        pp.plot(kinematics_time, r_end_effector_pos_array[:,1], 'r')
        pp.plot(ftime, force_array[:,1], 'y')
        pp.title('R EE Y - Red, F Y - Yellow')

        pp.subplot(323)
        pp.plot(kinematics_time, r_end_effector_pos_array[:,2], 'r')
        pp.plot(ftime, force_array[:,2], 'y')
        pp.title('R EE Z - Red, F Z - Yellow')

        pp.show()

    if audio:
        audio_data = d['audio_data']
        audio_amp = d['audio_amp']
        audio_freq = d['audio_freq']
        audio_chunk = d['audio_chunk']
	#audio_data_raw = d['audio_data_raw']


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
