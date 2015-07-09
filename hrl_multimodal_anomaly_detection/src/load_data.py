#!/usr/bin/env python

# System
import os
import fnmatch
#import random
import getpass
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from scipy.fftpack import fft
import sys
#import glob 
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
        self.AUDIO = True
        self.VISION = False

        self.numPkls = 0
        self.pklsList = []

        self.gs = gridspec.GridSpec(3,3)
        #self.gs.update(left = 0.4, right = 0.5, bottom = 0.05, hspace=0)
        self.gs2 = gridspec.GridSpec(3,1)

        current_user = getpass.getuser()
        folder_name_recordings = '/home/' + current_user + '/git/hrl-assistive/hrl_multimodal_anomaly_detection/recordings/'
        change_folder = raw_input("Current recordings folder is: %s, press [y] to change " % folder_name_recordings)
        if change_folder == 'y':
            folder_name_recordings = raw_input("Enter new recordings folder name and press [Enter] ")
        self.trial_name = raw_input("Enter trial name and press [Enter] ")

        folder_name_trial = folder_name_recordings + self.trial_name + "/"
        print "Current trial folder is %s " % folder_name_trial

        whichOpen = raw_input("Load all pickle files, only successful, or only failed? [a/s/f] ")
        while whichOpen != 'a' and whichOpen != 's' and whichOpen != 'f':
            print "Please enter 'a' or 's' or 'f' ! "
            whichOpen = raw_input("Load all pickle files, only successful, or only failed? [a/s/f] ")

        if whichOpen == 'a':
            pkl_file_pattern = 'iteration_*_*.pkl'
            self.whichOpenString = "successful and failed"
            print "Loading all pickle files \n"
        elif whichOpen == 's':
            pkl_file_pattern = 'iteration_*_success.pkl'
            self.whichOpenString = "only successful"
            print "Only loading successful pickle files \n"
        elif whichOpen == 'f':
            pkl_file_pattern = 'iteration_*_failure.pkl'
            self.whichOpenString = "only failed"
            print "Only loading successful pickle files \n"

        for file in os.listdir(folder_name_trial):
            if fnmatch.fnmatch(file, pkl_file_pattern):
                self.numPkls += 1
                pkl_file_path = folder_name_trial + file
                d = ut.load_pickle(pkl_file_path)
                self.pklsList.append(d)

        print "Number of pickle files loaded: %d \n" % self.numPkls

        if self.numPkls == 0:
            print "NO PICKLE FILES LOADED! CHECK YOU HAVE APPROPRIATE FILES!"
            sys.exit()

        for iterations in self.pklsList:
            print "Contents: "
            print iterations.keys()
            print "\n"

    def run(self):

        if self.FT_KINEMATICS:
            self.ft_kinematics()
        if self.AUDIO:
            self.audio()
        if self.VISION:
            self.vision()
        # Uncomment next line if you want to display all graphs at once...
        # ... otherwise ft/kinematics and sound graphs will display separately
        #plt.show()

    def ft_kinematics(self):

        # self.gs.title('test')
        forceAx1 = plt.subplot(self.gs[1,0])
        tfAx1 = plt.subplot(self.gs[0,0])
        forceAx1.set_xlabel('time (s)')
        forceAx1.set_title('X Force')
        tfAx1.set_title('X Pos')


        forceAx2 = plt.subplot(self.gs[1,1])
        tfAx2 = plt.subplot(self.gs[0,1])
        forceAx2.set_xlabel('time (s)')
        forceAx2.set_title('Y Force')
        tfAx2.set_title('Y Pos')

        forceAx3 = plt.subplot(self.gs[1,2])
        tfAx3 = plt.subplot(self.gs[0,2])
        forceAx3.set_xlabel('time (s)')
        forceAx3.set_title('Z Force')
        tfAx3.set_title('Z Pos')

        forceMagAx4 = plt.subplot(self.gs[2,:])
        forceMagAx4.set_xlabel('time (s) \n \n Trial Name: ' + self.trial_name + '\n Loaded ' + self.whichOpenString + ' iterations')
        forceMagAx4.set_title('Force Normalization (Magnitude)')

        for iterations in self.pklsList:
            ft_time = np.array(iterations.get('ft_time',None))
            kinematics_time = np.array(iterations.get('kinematics_time', None))

            ft_force = np.array(iterations.get('ft_force_raw',None))
            ft_torque = np.array(iterations.get('ft_torque_raw', None))

            tf_l_ee_pos = np.array(iterations.get('l_end_effector_pos', None))
            tf_l_ee_quat = np.array(iterations.get('l_end_effector_quat', None))
            tf_r_ee_pos = np.array(iterations.get('r_end_effector_pos', None))
            tf_r_ee_quat = np.array(iterations.get('r_end_effector_quat', None))

            ft_force_mag = np.linalg.norm(ft_force, axis=1)

            scooping_steps_times = np.array(self.pklsList[0].get('scooping_steps_times'), None)

            forceAx1.plot(ft_time, ft_force[:,0], 'b-')
            tfAx1.plot(kinematics_time, tf_l_ee_pos[:,0], 'r-')

            forceAx2.plot(ft_time, ft_force[:,1], 'b-')
            tfAx2.plot(kinematics_time, tf_l_ee_pos[:,1], 'r-')

            forceAx3.plot(ft_time, ft_force[:,2], 'b-')
            tfAx3.plot(kinematics_time, tf_l_ee_pos[:,2], 'r-')

            forceMagAx4.plot(ft_time, ft_force_mag, 'y-')

            for i in range(0, len(scooping_steps_times)):
                forceAx1.axvline(scooping_steps_times[i], color='k', linestyle='dotted')
                tfAx1.axvline(scooping_steps_times[i], color='k', linestyle='dotted' )
                forceAx2.axvline(scooping_steps_times[i], color='k', linestyle='dotted' )
                tfAx2.axvline(scooping_steps_times[i], color='k', linestyle='dotted' )
                forceAx3.axvline(scooping_steps_times[i], color='k', linestyle='dotted' )
                tfAx3.axvline(scooping_steps_times[i], color='k', linestyle='dotted' )
                forceMagAx4.axvline(scooping_steps_times[i], color='k', linestyle='dotted')


        plt.show()

        return True
        
    def audio(self):

        audioDataAx = plt.subplot(self.gs2[0,0])
        audioAmpAx = plt.subplot(self.gs2[1,0])
        audioFreqAx = plt.subplot(self.gs2[2,0])

        audioDataAx.set_title('Audio Data')
        audioAmpAx.set_title('Audio Amplitude')
        audioFreqAx.set_title('Audio Frequency')

        for iterations in self.pklsList:
            audio_data_raw = iterations.get('audio_data_raw')
            audio_time = iterations.get('audio_time')
            audio_sample_time = iterations.get('audio_sample_time')
            audio_chunk = iterations.get('audio_chunk')

            audio_data = np.fromstring(audio_data_raw, self.DTYPE)
            audio_data = np.array(audio_data).flatten()

            audioDataAx.plot(audio_time, audio_data, 'y-')

            plt.show()

        return True


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
