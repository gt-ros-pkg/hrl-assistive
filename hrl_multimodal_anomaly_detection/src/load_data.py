#!/usr/bin/env python

# System
import os
import random
import getpass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from scipy.fftpack import fft
# import yaafelib as yaafe
# from yaafelib import AudioFeature, check_dataflow_params, dataflow_safe_append, DataFlow
# import yaafefeatures as yf

# ROS
import roslib
roslib.load_manifest('hrl_multimodal_anomaly_detection')

# HRL
import hrl_lib.util as ut

class graphing():
    def __init__(self):

        self.DTYPE = np.int16

        self.FT_KINEMATICS = True
        self.AUDIO = False
        self.VISION = False

        current_user = getpass.getuser()
        folder_name_main = '/home/' + current_user + '/git/hrl-assistive/hrl_multimodal_anomaly_detection/recordings/'
        change_folder = raw_input("Current recordings folder is: %s, press [y] to change " % folder_name_main)
        if change_folder == 'y':
            folder_name_main = raw_input("Enter new folder name and press [Enter] ")
        trial_name = raw_input("Enter trial name and press [Enter] ")

        folder_name_trial = folder_name_main + trial_name + "/"
        print "Current trial folder is %s " % folder_name_trial
       
        pkl_file_name = raw_input("Enter exact name of pkl file, ex: [test.pkl] ")
        

        pkl_file_path = folder_name_trial + pkl_file_name
        # pkl_file_path = raw_input("Enter full path of pickle file to load: ")
        print os.path.isfile(pkl_file_path)
        self.d = ut.load_pickle(pkl_file_path)

        print self.d.keys()

    def run(self):

        self.gs = gridspec.GridSpec(2,3)
        #self.gs.update(left = 0.4, right = 0.5, bottom = 0.05, hspace=0)
        self.gs2 = gridspec.GridSpec(3,1)

        if self.FT_KINEMATICS:
            self.ft_kinematics()
        if self.AUDIO:
            self.audio()
        if self.VISION:
            self.vision()
        else: 
            print "No graphing selected, not doing anything"

        plt.show()

    def ft_kinematics(self):

        ft_time = np.array(self.d.get('ft_time',None))
        kinematics_time = np.array(self.d.get('kinematics_time', None))

        ft_force = np.array(self.d.get('ft_force_raw',None))
        ft_torque = np.array(self.d.get('ft_torque_raw', None))

        tf_l_ee_pos = np.array(self.d.get('l_end_effector_pos', None))
        tf_l_ee_quat = np.array(self.d.get('l_end_effector_quat', None))
        tf_r_ee_pos = np.array(self.d.get('r_end_effector_pos', None))
        tf_r_ee_quat = np.array(self.d.get('r_end_effector_quat', None))

        forceAx1 = plt.subplot(self.gs[1,0])
        tfAx1 = plt.subplot(self.gs[0,0])

        forceAx2 = plt.subplot(self.gs[1,1])
        tfAx2 = plt.subplot(self.gs[0,1])

        forceAx3 = plt.subplot(self.gs[1,2])
        tfAx3 = plt.subplot(self.gs[0,2])

        forceAx1.plot(ft_time, ft_force[:,0], 'b-')
        forceAx1.set_xlabel('time (s)')
        forceAx1.set_title('X Force')
        tfAx1.plot(kinematics_time, tf_l_ee_pos[:,0], 'r-')
        tfAx1.set_title('X Pos')

        forceAx2.plot(ft_time, ft_force[:,1], 'b-')
        forceAx2.set_xlabel('time (s)')
        forceAx2.set_title('Y Force')
        tfAx2.plot(kinematics_time, tf_l_ee_pos[:,1], 'r-')
        tfAx2.set_title('Y Pos')

        forceAx3.plot(ft_time, ft_force[:,2], 'b-')
        forceAx3.set_xlabel('time (s)')
        forceAx3.set_title('Z Force')
        tfAx3.plot(kinematics_time, tf_l_ee_pos[:,2], 'r-')
        tfAx3.set_title('Z Pos')

        return True
        # plt.show()


def butter_bandpass(lowcut, highcut, fs, order=5):
    """
    fs: sampling frequency
    """
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

    #rospy.init_node("data_visualization")
    grapher = graphing()
    grapher.run()

    # ft_kinematics = True
    # audio = False
    # vision = False

    # if ft_kinematics:

        

    # if audio:



    #     audio_data_raw = d.get('audio_data_raw')
    #     audio_time = d.get('audio_time')
    #     audio_sample_time = d.get('audio_sample_time')
    #     audio_chunk = d.get('audio_chunk')

    #     audio_data = np.


    #     audio_data = np.array(audio_data).flatten()

    #     import scipy.signal as signal
    #     RATE    = 44100 #sampling rate
    #     CHUNK   = 1024 #frame per buffer

    #     ## b, a = butter_bandpass(1,600, RATE, order=3)
    #     ## audio_data = signal.lfilter(b, a, audio_data)
    #     ## audio_data = signal.filtfilt(b,a,audio_data,padlen=2000)

    #     ## for chunk in audio_data
    #     ##     audio_amp = np.fft.fft(audio_data / float(self.MAX_INT))  #normalization & FFT

    #     print np.array(audio_data).shape

    #     ## win = np.hamming(CHUNK)
    #     ## rms = []
    #     ## for d in audio_data:
    #     ##     d = np.array(d)/32768.0
    #     ##     windowedSignal = d*win;
    #     ##     fft = np.fft.fft(windowedSignal)
    #     ##     rms.append(np.log(np.abs(fft)))


    #     plt.figure()
    #     plt.subplot(411)
    #     plt.plot(audio_data,'b.')
    #     plt.xlabel('Time')
    #     plt.ylabel('Audio Data... Amplitude?')
    #     plt.title("Audio Data")

    #     plt.subplot(412)
    #     xs = audio_freq[:audio_chunk/16]
    #     ys = np.abs(audio_amp[:][:audio_chunk/16])
    #     ys = np.multiply(20,np.log10(ys))
    #     plt.plot(xs,ys,'y')
    #     plt.xlabel('Audio freqs')
    #     plt.ylabel('Amplitudes')
    #     plt.title("Audio Frequency and Amplitude")

    #     plt.subplot(413)
    #     plt.plot(audio_freq, 'b')
    #     plt.title('Raw audio_freq data')

    #     plt.subplot(414)
    #     plt.plot(audio_amp, 'y')
    #     plt.title('Raw audio_amp data')

    #     #plt.show()

    # plt.show()

    # if vision:
    #     data = d['visual_points']
    #     points = dict()
    #     for dataSet in data:
    #         for key, value in dataSet.iteritems():
    #             if key in points:
    #                 points[key].append(value)
    #             else:
    #                 points[key] = [value]

    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')

    #     point = random.choice(points.keys())
    #     point = np.array(points[point])
    #     xs, ys ,zs = point[:, 0], point[:, 1], point[:, 2]
    #     ax.scatter(xs, ys, zs, c='r')
    #     ax.set_xlabel('X')
    #     ax.set_ylabel('Y')
    #     ax.set_zlabel('Z')

    #     plt.show()


    # #plt.subplot(616)
    # #p = 20*np.log10(np.abs(np.fft.rfft(audio_data_raw[:2048, 0])))
    # #f = np.linspace(0, rate/2.0, len(p))
    # #plt.plot(f, p)
    # #plt.xlabel("Frequency(Hz)")
    # #plt.ylabel("Power(dB)")
    # #plt.title("Raw audio_data (same as wav data)")
        
    # ## plt.plot(rms)
    #     ## plt.stem(noise_freq_l, values, 'k-*', bottom=0)

    #     ## import pyaudio
    #     ## MAX_INT = 32768.0
    #     ## CHUNK   = 1024 #frame per buffer
    #     ## RATE    = 44100 #sampling rate
    #     ## UNIT_SAMPLE_TIME = 1.0 / float(RATE)
    #     ## CHANNEL=1 #number of channels
    #     ## FORMAT= pyaudio.paInt16
    #     ## DTYPE = np.int16

    #     ## p=pyaudio.PyAudio()
    #     ## stream=p.open(format=FORMAT, channels=CHANNEL, rate=RATE, \
    #     ##                         input=True, frames_per_buffer=CHUNK)

    #     ## string_audio_data = np.array(audio_data, dtype=DTYPE).tostring()
    #     ## import wave
    #     ## WAVE_OUTPUT_FILENAME = "/home/dpark/git/pyaudio/test/output.wav"
    #     ## wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    #     ## wf.setnchannels(CHANNEL)
    #     ## wf.setsampwidth(p.get_sample_size(FORMAT))
    #     ## wf.setframerate(RATE)
    #     ## wf.writeframes(b''.join(string_audio_data))
    #     ## wf.close()
