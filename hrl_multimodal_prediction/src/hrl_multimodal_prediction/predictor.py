# Reads realtime data from a buffer in subscriber node
# predicts the future 10-20 timesteps
# Using Animation plot both the original position and predicted position
# Using Animation plot both the original sound and predicted sound
# Using Soundplay or PyAudio play predicted sound in real time
import rosbag
import rospy
import numpy as np
import matplotlib.pyplot as plt
import librosa
import os, copy, sys
import tensorflow as tf
import numpy.matlib
import scipy as sp
import scipy.interpolate

from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout, RepeatVector, TimeDistributed, Input
from keras.models import Sequential, Model
from keras import optimizers

import config as cf
import pyaudio
from std_msgs.msg import String, Float64

from predict_subscriber import predict_subscriber
# Predictor and visualizer go hand in hand
# Predict 10 time steps and save in a variable
# unlock mutex_viz
# variable read from visualizer, plot and lock mutex_viz -- wait for predicted value
# repeat
class predictor():
	g_pred_mfcc = None
	g_pred_relpos = None
	g_actual_mfcc = None
	g_actual_relpos = None

	def define_network(self, batch_size, time_in, time_out, input_dim, n_neurons):
		model = Sequential()
		model.add(LSTM(input_dim*time_out, batch_input_shape=(batch_size, time_in, input_dim),
						stateful=True, return_sequences=True, activation='tanh'))
		model.add(LSTM(input_dim*time_out, stateful=True, return_sequences=True, activation='tanh'))
		model.add(TimeDistributed(Dense(input_dim*time_out, activation='linear')))

		model.load_weights(WEIGHT_FILE)
		model.compile(loss='mse', optimizer='RMSprop')
		print model.summary()
		print "Inputs: {}".format(model.input_shape)
		print "Outputs: {}".format(model.output_shape)
		return model

	def rescale(self, dataset):
		# rescale values to -1, 1 for tanh
		scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
		dim1 = dataset.shape[0]
		dim2 = dataset.shape[1]
		dim3 = dataset.shape[2]
		dataset = dataset.reshape(dim1*dim2, dim3)
		dataset[:,0:3] = scaler.fit_transform(dataset[:,0:3])
		dataset = dataset.reshape(dim1, dim2, dim3)
		return dataset, scaler

	def scale_back(self, pred, mfcc_scaler, relpos_scaler):
		g_pred_mfcc = mfcc_scaler.inverse_transform(pred[0:3])
		g_pred_relpos = relpos_scaler.inverse_transform(pred[3:6])
		print 'saving values scaled back'

	def preprocess(self, mfcc, relpos):
		#interpolate so that mfcc and relpos have the same length?
		mfcc, mfcc_scaler = rescale(mfcc)
		relpos, relpos_scaler = rescale(relpos)
		comb_data = np.concatenate((mfccs, relpos), axis=2)
		return comb_data, mfcc_scaler, relpos_scaler

	def predict(self, model, comb_data):
		#some reshaping before prediction?
		for i in range(0, comb_data.shape[0]):
			p_tmp = model.predict_on_batch(comb_data[i]) 
			rst.append(p_tmp)
		rst = np.array(rst)
		return rst
	
	def reconstruct_mfcc(self, mfccs):
		#build reconstruction mappings
		n_mfcc = mfccs.shape[0]
		n_mel = cf.N_MEL
		dctm = librosa.filters.dct(n_mfcc, n_mel)
		n_fft = cf.N_FFT
		mel_basis = librosa.filters.mel(self.RATE, n_fft, n_mels=n_mel)

		#Empirical scaling of channels to get ~flat amplitude mapping.
		bin_scaling = 1.0/np.maximum(0.0005, np.sum(np.dot(mel_basis.T, mel_basis), axis=0))
		#Reconstruct the approximate STFT squared-magnitude from the MFCCs.
		recon_stft = bin_scaling[:, np.newaxis] * np.dot(mel_basis.T, self.invlogamplitude(np.dot(dctm.T, mfccs)))
		#Impose reconstructed magnitude on white noise STFT.
		excitation = np.random.randn(y.shape[0]) # this will be constant--based on one msgsize
		E = librosa.stft(excitation, n_fft=cf.N_FFT)
		recon = librosa.istft(E/np.abs(E)*np.sqrt(recon_stft))
		#print recon
		#print recon.shape
		wav.write('reconsturct.wav', cf.RATE, recon)
		return recon

	def invlogamplitude(self, S):
	#"""librosa.logamplitude is actually 10_log10, so invert that."""
		return 10.0**(S/10.0)

	def play_sound_realtime(self, mfcc):
		recon = reconstruct_mfcc(mfcc)
		pya = pyaudio.PyAudio()
		stream = pya.open(format=pyaudio.paFloat32, channels=1, rate=cf.RATE, output=True)
		stream.write(recon)
		stream.stop_stream()
		stream.close()
		pya.terminate()

	def m_callback(self, data):
		print data

	def a_callback(self, data):
		print data

	def run(self):
		p = predict_subscriber()
		while True:
			print p.Audio_Buffer
			print p.RelPos_Buffer

		# rospy.init_node('predictor', anonymous=True)
		# rospy.Subscriber('processed_audio', Float64, self.a_callback)
		# rospy.Subscriber('processed_relpos', Float64, self.m_callback)
		# rospy.spin()

		# r = rospy.Rate(8) # 8hz		
		# while not rospy.is_shutdown():
		# 	mfcc = psub.Audio_buffer().pop() # takes the last element
		# 	relpos = psub.RelPos_buffer().pop()
		# 	g_actual_mfcc = mfcc_scaler
		# 	g_actual_relpos = relpos
		# 	comb_data, mfcc_scaler, relpos_scaler = preprocess(mfcc, relpos)
		# 	model = define_network(cf.PRED_BATCH_SIZE, cf.TIMESTEP_IN, cf.TIMESTEP_OUT, cf.INPUT_DIM, cf.N_NEURONS)
		# 	pred = predict(model, comb_data)
		# 	scale_back(pred, mfcc_scaler, relpos_scaler)
		# 	r.sleep()

def main():
	pred = predictor()
	pred.run()

if __name__ == '__main__':
	main()    
