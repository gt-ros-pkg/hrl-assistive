#! /usr/bin/env python
#
# This node will record background noise and continue recording while detecting
# amplitude anomalies based on a statistical model of previous trials. If an
# amplitude anomaly is detected, it will publish a ROS message to the 'emergency' topic.
# Otherwise, it will take a DFT spectrogram of the audio data, find the average
# power for each frequency and publish a 1D numpy array ([Average Power, frequency])
# of this data to the 'audio_analysis' topic.

from __future__ import division
import numpy as np
import rospy
import pylab
from collections import deque
from std_msgs.msg import String
from matplotlib import pyplot as plt
from matplotlib import mlab
from scipy import stats
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from std_msgs.msg import Float32
import math

import pyaudio
import wave


class audio_core():
    def __init__(self):
        self.chunk=1024 #frame size
        self.form=pyaudio.paFloat32
        self.channel=1 #number of channels
        self.rate1=44100 #sampling rate
        self.secs=5 #number of seconds to establish background noise baseline
        self.stop=True
        self.mu=0	#mean as calculated from previous trials
        self.sigma=170.
        self.stddevs=1.1	#number of standard deviations above mean to allow as a threshhold
        self.pub1=rospy.Publisher('emergency', String)
        self.pub2=rospy.Publisher('audio_analysis', numpy_msg(Floats))
        self.raw_audio_pub=rospy.Publisher('feeding/raw_audio', Float32)        
        rospy.init_node('audio_talker', anonymous=False)
        self.p=pyaudio.PyAudio()
        self.stream=self.p.open(format=self.form, channels=self.channel, rate=self.rate1, input=True, frames_per_buffer=self.chunk)

    #def background(self):
        #print "legacy code for background removal."
        #print ("*recording background noise")
        #frames_back=deque([])
        #amp_back=deque([])
        #for i in range(0, int(rate1/chunk*secs)): #Record a couple (5) seconds of background noise
        # data=stream.read(chunk)
        # decoded=np.fromstring(data, 'Float32')
        # decoded2=np.fromstring(data, 'Int16')
        # frames_back.append(decoded)
        # amp_back=deque([])

        #print("*done recording background noise")
        #Model of Background Noise
        #mu_back=np.mean(amp_back)

    def audio_analyzer(self, z):#, data):
        r=rospy.Rate(10) #in hz
        #Check if amplitude is above threshold
        #print z, self.stddevs*self.sigma

        a=abs(z)>=self.stddevs*self.sigma
        b=a.any()
        if b:
            #Continue checking amplitude and publish stop message if anomaly occurs
            self.pub1.publish('STOP')
            print "Audio anomaly"
        #else:
            #If no anomaly continue publishing power and frequency data
            #self.pub2.publish(data.astype(float))
        while not rospy.is_shutdown() and b:
            a=abs(z)>=self.stddevs*self.sigma
            b=a.any()

            if b:
                #Continue checking amplitude and publish stop message if anomaly occurs
                self.pub1.publish('STOP')
		print "the end is upon us"
                print "Audio anomaly"
            #else:
                #If no anomaly continue publishing power and frequency data
                #self.pub2.publish(data.astype(float))

    def compute(self):
        frames=deque([], 3000)
        amp_frames=deque([], 3000)
        try:
            while not rospy.is_shutdown() and self.stop:
                data=self.stream.read(self.chunk)
                decoded=np.fromstring(data, 'Float32')
                decoded2=np.fromstring(data, 'Int16')

                # Raw data publisher
                self.raw_audio_pub.publish(decoded[0])
                
                amp_frames.append(decoded2)
                frames.append(decoded)
                l=len(frames)
                if l*self.chunk>=3000:
                    index=range(0, l-2)
                    frames_arr=np.array(frames.popleft())
                    amp_frames_arr=np.array(amp_frames.popleft())
                    for i in index:
                        e=frames.popleft()
                        frames_arr=np.concatenate((frames_arr, e), axis=0)
                    for i in index:
                        s=amp_frames.popleft()
                        amp_frames_arr=np.concatenate((amp_frames_arr, s), axis=0)
                    frames_arr.reshape(-1)
                    #DFT Spectrogram
                    #Pxx, freqs, t=mlab.specgram(frames_arr, NFFT=256, Fs=rate1)
                    #Find average power
                    #Pxx_ave=np.mean(Pxx, axis=1)
                    #ind=freqs<=3000
                    #Look at frequencies that are in expected range
                    #c=Pxx_ave[ind]
                    #d=freqs[ind]
                    #Amalgamate into one 1D array, so it can be sent as a ROS message
                    #data1=np.concatenate((c, d))
                    #Make histogram of frequency vs power spectral density
                    #H, xedges, yedges=np.histogram2d(Pxx_ave, freqs, bins=16, normed=True)
                    #Find Z score with which to check amplitude
                    z=((amp_frames_arr-self.mu)/self.sigma)
                    
                    a=abs(z)>=self.stddevs*self.sigma
                    
                    #Publish ROS messages
                    self.audio_analyzer(z)#,data1)
        except:
            a = 1

def main():
    core = audio_core()
    core.compute()
    rospy.spin()


if __name__=='__main__':
    main()
