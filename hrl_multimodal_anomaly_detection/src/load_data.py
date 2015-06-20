#!/usr/bin/env python

# System
import numpy as np
import time, sys, os
import cPickle as pkl


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

    pkl_file = raw_input("Enter exact name of pkl file, ex: './test.pkl: '")
    #pkl_file = './noise.pkl'
    ## pkl_file = '/home/dpark/svn/robot1/src/projects/anomaly/test_data/s_cup_human_b1.pkl'
    ## pkl_file = '/home/dpark/svn/robot1/src/projects/anomaly/test_data/drawer_cup_human_b3.pkl'
    ## pkl_file = '/home/dpark/svn/robot1/src/projects/anomaly/test_data/cup_cup_human_b1.pkl'
    print os.path.isfile(pkl_file)
    d = ut.load_pickle(pkl_file)

    print d.keys()
    ft = True
    audio = True

    if ft:
        ftime = d.get('ft_time',None)
        force = d.get('ft_force_raw',None)

        aForce = np.squeeze(force).T
        print aForce.shape

        pp.figure()
        pp.plot(ftime, aForce[2])
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
        pp.subplot(511)
        pp.plot(audio_data,'b.')
	pp.xlabel('Time')
	pp.ylabel('Audio Data... Amplitude?')
        pp.title("Audio Data")

        pp.subplot(512)
        xs = audio_freq[:audio_chunk/16]
        ys = np.abs(audio_amp[:][:audio_chunk/16])
        ys = np.multiply(20,np.log10(ys))
        pp.plot(xs,ys,'y')
	pp.xlabel('Audio freqs')
	pp.ylabel('Amplitudes')
        pp.title("Audio Frequency and Amplitude")
        
	pp.subplot(513)
	pp.plot(audio_freq, 'b')
	pp.title('Raw audio_freq data')

	pp.subplot(514)
	pp.plot(audio_amp, 'y')
	pp.title('Raw audio_amp data')

	pp.subplot(515)
	pp.plot(audio_data, 'b')
	pp.title('Raw audio_data data')

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
