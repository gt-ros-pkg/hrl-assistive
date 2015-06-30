#!/usr/bin/env python

import pyaudio
import pylab
import numpy as np
import time

import roslib
roslib.load_manifest('hrl_multimodal_anomaly_detection')

### RECORD AUDIO FROM MICROPHONE ###
class realtimeFFT:
	CHUNK = 1024
	FORMAT = pyaudio.paInt16
	CHANNELS = 2
	RATE = 44100

	def __init__(self):
		p=pyaudio.PyAudio()
		self.stream = p.open(format=self.FORMAT,
		                channels=self.CHANNELS,
		                rate=self.RATE,
		                input=True,
		                frames_per_buffer=self.CHUNK)
	
	def run(self):
		pylab.ion()
		pylab.plot()

		streamData = self.captureStream()
		streamDataShaped = self.shapeTriangle(streamData)
		self.fftComputeAndGraph(streamDataShaped)

		print "Graphed initial stuff... now trying to realtime update!"

		time.sleep(3)

		#while True:
		#	streamData = self.captureStream()
		#	streamDataShaped = self.shapeTriangle(streamData)
		#	self.fftComputeAndGraph(streamDataShaped)

	@staticmethod
	def shapeTriangle(data):
		triangle = np.array(range(len(data)/2)+range(len(data)/2)[::-1])+1
		return data

	def captureStream(self):
		self.stream.read(self.CHUNK) #prime the sound card this way
		pcm=np.fromstring(self.stream.read(self.CHUNK), dtype=np.int16)
		return pcm

	def fftComputeAndGraph(self, data):
		fft = np.fft.fft(data)
		fftr = 10*np.log10(abs(fft.real))[:len(data)/2]
		ffti = 10*np.log10(abs(fft.imag))[:len(data)/2]
		fftb = 10*np.log10(np.sqrt(fft.imag**2+fft.real**2))[:len(data)/2]
		freq = np.fft.fftfreq(np.arange(len(data)).shape[-1])[:len(data)/2]
		freq = freq*self.RATE/1000 #make the frequency scale
		pylab.subplot(411)
		pylab.title("Original Data")
		pylab.grid()    
		pylab.plot(np.arange(len(data))/float(self.RATE)*1000,data,'r-',alpha=1)
		pylab.xlabel("Time (milliseconds)")
		pylab.ylabel("Amplitude")
		pylab.subplot(412)
		pylab.title("Real FFT")
		pylab.xlabel("Frequency (kHz)")
		pylab.ylabel("Power")
		pylab.grid()    
		pylab.plot(freq,fftr,'b-',alpha=1)
		pylab.subplot(413)
		pylab.title("Imaginary FFT")
		pylab.xlabel("Frequency (kHz)")
		pylab.ylabel("Power")
		pylab.grid()    
		pylab.plot(freq,ffti,'g-',alpha=1)
		pylab.subplot(414)
		pylab.title("Real+Imaginary FFT")
		pylab.xlabel("Frequency (kHz)")
		pylab.ylabel("Power")
		pylab.grid()    
		pylab.plot(freq,fftb,'k-',alpha=1)
		pylab.draw()
		pylab.pause(0.0001)
		pylab.clf()


if __name__=='__main__':
	fft = realtimeFFT()
	fft.run()