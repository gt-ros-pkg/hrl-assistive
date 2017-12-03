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

def reconstruct_mfcc(mfccs, y):
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
	excitation = np.random.randn(cf.HOP_LENGTH*cf.P_MFCC_TIMESTEP-1)
	E = librosa.stft(excitation, n_fft=n_fft)
	recon = librosa.istft(E/np.abs(E)*np.sqrt(recon_stft))
	#print recon
	#print recon.shape
	wav.write('dtmfcc.wav', cf.RATE/2, recon)
	return recon

def invlogamplitude(S):
#"""librosa.logamplitude is actually 10_log10, so invert that."""
	return 10.0**(S/10.0)

def main():
	y, sr = librosa.load('./bagfiles/unpacked/data1.wav', mono=True)	
	librosa.output.write_wav('dt1.wav', y, sr)
	p = pyaudio.PyAudio()
	stream = p.open(format=pyaudio.paFloat32, channels=1, rate=44100/2, output=True,
					 input_device_index=0)#,frames_per_buffer=1)
	
	print y.shape[0], y[0:4096].shape[0]
	for i in range(0,y.shape[0], 4096):
		tmp = y[i:i+4096]
		print tmp.shape
		mfccs = librosa.feature.mfcc(y=tmp, sr=cf.RATE, hop_length=cf.HOP_LENGTH, n_fft=cf.N_FFT, n_mfcc=cf.N_MFCC)# default hop_length=512, hop_length=int(0.01*sr))
		print mfccs.shape
		recon = reconstruct_mfcc(mfccs, y)
		data = recon.astype(np.float32).tostring()
		stream.write(data)

	stream.close()
	p.terminate()
	print("* Preview completed!")



if __name__ == '__main__':
	main()    
