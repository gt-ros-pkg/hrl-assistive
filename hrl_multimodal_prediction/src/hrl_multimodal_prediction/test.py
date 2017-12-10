# from predictor import predictor
import config
import librosa
import pyaudio
import wave
import sys
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import host_subplot
import matplotlib.animation as animation
import librosa
import scipy.io.wavfile as wav
import scipy as sp
import scipy.interpolate

import rospy
from std_msgs.msg import String
import config as cf

# def talker():
#     pub = rospy.Publisher('chatter', String, queue_size=10)
#     rospy.init_node('talker', anonymous=True)
#     rate = rospy.Rate(10) # 10hz
#     while not rospy.is_shutdown():
#         hello_str = "hello world %s" % rospy.get_time()
#         rospy.loginfo(hello_str)
#         pub.publish(hello_str)
#         rate.sleep()

# if __name__ == '__main__':
#     try:
#         talker()
#     except rospy.ROSInterruptException:
#         pass


# Sent for figure
# font = {'size'   : 9}
# # Setup figure and subplots
# f0 = figure(num = 0, figsize = (12, 8))#, dpi = 100)
# f0.suptitle("ARtag & Audio combined Prediction", fontsize=12)
# ax01 = subplot2grid((2, 2), (0, 0))


# fig, ax = plt.subplots()
# x = np.arange(0, 2*np.pi, 0.01)
# line, = ax.plot(x, np.sin(x))

# def animate(i):
# 	print i
# 	line.set_ydata(np.sin(x + i/10.0))  # update the data
# 	# line.set_ydata()
# 	return line,

# def main():
# 	# pred = predictor()
# 	# pred.test()
# 	# print pred.TESTA

# 	ani = animation.FuncAnimation(fig, animate, np.arange(1, 20), interval=25, repeat=False)
# 	plt.show()

from numpy import sin, linspace, pi
from pylab import plot, show, title, xlabel, ylabel, subplot
from scipy import fft, arange, signal, fftpack

def normalize(y, min_y, max_y):
	# normalize to range (-1,1)
	#NewValue = (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin
	y = (((y - min_y) * (1 + 1)) / (max_y - min_y)) -1

	# this normalizes to (0,1)
	# y = (y - min_y) / (max_y - min_y)
	return y

def scale_back(seq, min_y, max_y):
	#scale back from -1 1 range
	seq = (((seq + 1)*(max_y - min_y)) / (1 + 1)) + min_y

	# scale back 
	# seq = seq * (max_y - min_y) + min_y
	return seq

def reconstruct_mfcc(mfccs, y):
	print mfccs
	# plt.plot(mfccs[0])
	# plt.plot(mfccs[1])
	# plt.plot(mfccs[2])
	# plt.show()
	# mi = np.min(mfccs)
	# ma = np.max(mfccs)
	# mfccs = normalize(mfccs, np.min(mfccs), np.max(mfccs))
	# mfccs = mfccs**np.exp(1)
	# mfccs = scale_back(mfccs, mi, ma)
	# print mfccs
	# plt.plot(mfccs[0])
	# plt.plot(mfccs[1])
	# plt.plot(mfccs[2])
	# plt.show()

	#build reconstruction mappings
	n_mfcc = mfccs.shape[0]
	n_mel = cf.N_MEL
	dctm = librosa.filters.dct(n_mfcc, n_mel)
	n_fft = cf.N_FFT
	mel_basis = librosa.filters.mel(cf.RATE, n_fft, n_mels=n_mel)

	#Empirical scaling of channels to get ~flat amplitude mapping.
	bin_scaling = 1.0/np.maximum(0.0005, np.sum(np.dot(mel_basis.T, mel_basis), axis=0))
	#Reconstruct the approximate STFT squared-magnitude from the MFCCs.
	recon_stft = bin_scaling[:, np.newaxis] * np.dot(mel_basis.T, invlogamplitude(np.dot(dctm.T, mfccs)))
	#Impose reconstructed magnitude on white noise STFT.
	excitation = np.random.randn(33)
	E = librosa.stft(excitation, n_fft=n_fft)
	recon = librosa.istft(E/np.abs(E)*np.sqrt(recon_stft))
	
	# How to manipulate reconstructed sound?
	# recon = recon*100
	# print recon
	# recon = recon - noise
	# n = 40  # the larger n is, the smoother curve will be
	# b = [1.0 / n] * n
	# a = 1
	# recon = lfilter(b,a,recon)
	# print recon
	#print recon.shape
	
	# recon -= 0.5

	print recon
	plt.plot(recon, color='green')
	plt.show()
	# recon = np.array(recon, dtype=object)
	# recon = normalize(recon, np.min(recon), np.max(recon))
	
	# n = 2  # the larger n is, the smoother curve will be
	# b = [1.0 / n] * n
	# a = 1
	# recon = signal.lfilter(b,a,recon)


	recon = normalize(recon, np.min(recon), np.max(recon))
	for i in range(recon.shape[0]):
		if recon[i] < 0:
			recon[i] = -np.abs(recon[i])**(np.exp(1.3))
		else:
			recon[i] = recon[i]**(np.exp(1.3))
	
	# recon = np.array(recon)
	# recon = signal.hilbert(signal)
	# recon = normalize(recon, np.min(recon), np.max(recon))
	# recon = recon*2
	# n = 1  # the larger n is, the smoother curve will be
	# b = [1.0 / n] * n
	# a = 1
	# recon = signal.lfilter(b,a,recon)
	# recon = recon*2

	plt.plot(recon, color='green')
	plt.show()
	wav.write('dtmfcc.wav', cf.RATE/2, recon)
	return recon

def invlogamplitude(S):
#"""librosa.logamplitude is actually 10_log10, so invert that."""
	return 10.0**(S/10.0)


def main():
	y, sr = librosa.load('data1.wav', mono=True)	

	# y = y[0:15700]
	# y = np.hstack((y,y[15500:15700]))
	# y = np.hstack((y,y[0:5000]))
	
	# 2)
	y1 = y[0:5000]
	y1 = np.hstack((y1,y[0:5000]))
	y1 = np.hstack((y1,y[15800:y.shape[0]]))
	print y1.shape
	plt.plot(y1)
	plt.show()
	# wav.write('data1-2.wav', cf.RATE/2, y)

	mfccs = librosa.feature.mfcc(y=y, sr=cf.RATE, hop_length=cf.HOP_LENGTH, n_fft=cf.N_FFT, n_mfcc=3)# default hop_length=512, hop_length=int(0.01*sr))
	recon = reconstruct_mfcc(mfccs, y)


	yf = fftpack.fft(y)
	print yf # y is in frequency domain
	T = 1.0 / 44100.0
	N = y.shape[0]
	xf = np.linspace(0.0, 2.0/(T), N//2)
	print xf.shape, yf.shape
	plt.plot(xf[1:xf.shape[0]], 1.0/2*N * np.abs(yf[1:N//2]))
	plt.grid()
	# plt.show()
	y2 = fftpack.ifft(yf).real
	print y2

	yf2 = fftpack.fft(recon)
	print yf2 # y is in frequency domain
	T = 1.0 / 44100.0
	N = recon.shape[0]
	xf2 = np.linspace(0.0, 2.0/(T), N//2)
	print xf2.shape, yf2.shape
	plt.plot(xf2[1:xf2.shape[0]], 1.0/2*N * np.abs(yf2[1:N//2]), color='red')
	plt.grid()
	plt.show()
	y2 = fftpack.ifft(yf2).real
	print y2

	# #shift freq up
	# # yf -= 50000
	# # T = 1.0 / 44100.0
	# # N = y.shape[0]
	# # xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
	# # plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
	# # plt.grid()
	# # plt.show()
	# # y2 = fftpack.ifft(yf).real
	# # print y2

	# librosa.output.write_wav('dt1.wav', y2, sr)
	# plt.plot(y2)
	# plt.show()


	# (2) Online timestep play
	# p = pyaudio.PyAudio()
	# stream = p.open(format=pyaudio.paFloat32, channels=1, rate=44100/2, output=True,
	# 				 input_device_index=0)#,frames_per_buffer=1)
	
	# print y.shape[0], y[0:4096].shape[0]
	# for i in range(0,y.shape[0], 4096):
	# 	tmp = y[i:i+4096]
	# 	print tmp.shape
	# 	mfccs = librosa.feature.mfcc(y=tmp, sr=cf.RATE, hop_length=cf.HOP_LENGTH, n_fft=cf.N_FFT, n_mfcc=cf.N_MFCC)# default hop_length=512, hop_length=int(0.01*sr))
	# 	print mfccs.shape
	# 	recon = reconstruct_mfcc(mfccs, y)
	# 	data = recon.astype(np.float32).tostring()
	# 	stream.write(data)

	# stream.close()
	# p.terminate()
	# print("* Preview completed!")



if __name__ == '__main__':
	main()    
