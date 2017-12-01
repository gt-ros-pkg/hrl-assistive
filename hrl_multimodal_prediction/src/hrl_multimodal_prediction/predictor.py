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
from sklearn import preprocessing

from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout, RepeatVector, TimeDistributed, Input
from keras.models import Sequential, Model
from keras import optimizers
from keras import backend as K

import config as cf
import pyaudio
from std_msgs.msg import String, Float64, Float64MultiArray, MultiArrayLayout
from hrl_multimodal_prediction.msg import audio, pub_relpos, pub_mfcc, plot_pub
from visualization_msgs.msg import Marker

# Predictor and visualizer go hand in hand
# Predict 10 time steps and save in a variable
# unlock mutex_viz
# variable read from visualizer, plot and lock mutex_viz -- wait for predicted value
# repeat
class predictor():
	posUpdate = False
	mfccUpdate = False
	relpos = None
	mfcc = None
	
	model = None #LSTM Model
	graph = None
	stream = None
	plot_pub = None

	def __init__(self):
		# pya = pyaudio.PyAudio()
		# self.stream = pya.open(format=pyaudio.paFloat32, channels=1, rate=cf.RATE, output=True)		
		
		# must publish original and predicted for visualizer
		# self.plot_pub = rospy.Publisher('plot_pub', plot_pub, queue_size=10)

	def define_network(self, batch_size, time_in, time_out, input_dim, n_neurons):
		model = Sequential()
		model.add(LSTM(input_dim*time_out, batch_input_shape=(batch_size, time_in, input_dim),
						stateful=True, return_sequences=True, activation='tanh'))
		model.add(LSTM(input_dim*time_out, stateful=True, return_sequences=True, activation='tanh'))
		model.add(TimeDistributed(Dense(input_dim*time_out, activation='linear')))

		model.load_weights(cf.WEIGHT_FILE)
		model.compile(loss='mse', optimizer='RMSprop')
		print model.summary()
		print "Inputs: {}".format(model.input_shape)
		print "Outputs: {}".format(model.output_shape)
		return model

	def reconstruct_mfcc(self, mfccs):
		#build reconstruction mappings
		n_mfcc = mfccs.shape[0]
		n_mel = cf.N_MEL
		dctm = librosa.filters.dct(n_mfcc, n_mel)
		n_fft = cf.N_FFT
		mel_basis = librosa.filters.mel(cf.RATE, n_fft, n_mels=n_mel)

		#Empirical scaling of channels to get ~flat amplitude mapping.
		bin_scaling = 1.0/np.maximum(0.0005, np.sum(np.dot(mel_basis.T, mel_basis), axis=0))
		#Reconstruct the approximate STFT squared-magnitude from the MFCCs.
		recon_stft = bin_scaling[:, np.newaxis] * np.dot(mel_basis.T, self.invlogamplitude(np.dot(dctm.T, mfccs)))
		#Impose reconstructed magnitude on white noise STFT.
		excitation = np.random.randn(cf.HOP_LENGTH*cf.TIMESTEP_OUT-1) # this will be constant--based on one msgsize
		E = librosa.stft(excitation, n_fft=cf.N_FFT)
		recon = librosa.istft(E/np.abs(E)*np.sqrt(recon_stft))
		#print recon
		#print recon.shape
		# wav.write('reconsturct.wav', cf.RATE, recon)
		return recon

	def invlogamplitude(self, S):
	#"""librosa.logamplitude is actually 10_log10, so invert that."""
		return 10.0**(S/10.0)

	def play_sound_realtime(self, mfcc):
		mfcc = mfcc.reshape(cf.MFCC_DIM, cf.TIMESTEP_OUT)
		recon = self.reconstruct_mfcc(mfcc)
		self.stream.write(recon)
		# stream.stop_stream()
		# stream.close()
		# pya.terminate()

	def rescale(self, dataset):
		# rescale values to -1, 1 for tanh
		scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
		dim1 = dataset.shape[0]
		dim2 = dataset.shape[1]
		dim3 = dataset.shape[2]
		dataset = dataset.reshape(dim1*dim2, dim3)
		dataset[:,:] = scaler.fit_transform(dataset[:,:])
		dataset = dataset.reshape(dim1, dim2, dim3)
		return dataset, scaler

	def scale_back(self, pred, mfcc_scaler, relpos_scaler):		
		dim1 = pred.shape[0]
		dim2 = pred.shape[1]
		dim3 = pred.shape[2]
		pred = pred.reshape(dim1*dim2, dim3)
		mfcc = mfcc_scaler.inverse_transform(pred[:,0:3])
		mfcc = mfcc.reshape(1, cf.TIMESTEP_OUT, cf.MFCC_DIM)
		relpos = relpos_scaler.inverse_transform(pred[:,3:6])
		relpos = relpos.reshape(1, cf.TIMESTEP_OUT, cf.IMAGE_DIM)
		return mfcc, relpos
	
	def callback(self, data):
		if data._type == 'hrl_multimodal_prediction/pub_mfcc':
			self.mfcc = data.mfcc
			self.mfccUpdate = True
		elif data._type == 'hrl_multimodal_prediction/pub_relpos':
			self.relpos = data.relpos
			self.posUpdate = True

		if self.mfccUpdate and self.posUpdate:
			self.posUpdate = False
			self.mfccUpdate = False

			# Convert shape to fit LSTM
			self.mfcc = np.array(self.mfcc).reshape(1, cf.P_MFCC_TIMESTEP, cf.N_MFCC) # shape=(t, n_mfcc)
			self.relpos = np.array(self.relpos).reshape(1, 1, cf.IMAGE_DIM) 
			# print self.mfcc.shape, self.relpos.shape
			tmp = self.relpos
			for i in range(cf.P_MFCC_TIMESTEP-1): #make position data match mfcc timestep
				self.relpos = np.concatenate((self.relpos,tmp), axis=1)
			# print self.mfcc.shape, self.relpos.shape

			# Rescale perFeature, save Scaler
			norm_mfcc, mfcc_scaler = self.rescale(self.mfcc)
			norm_relpos, relpos_scaler = self.rescale(self.relpos)

			# Combine two modes 
			comb_data = np.concatenate((norm_mfcc, norm_relpos), axis=2)
			# print comb_data.shape
			comb_data = np.array(comb_data)

			# Predic -- graph default must to fix multithread bug in keras-tensorflow
			with self.graph.as_default():
				for i in range(0, cf.P_MFCC_TIMESTEP):
					pred = self.model.predict_on_batch(comb_data[:,i:i+1,:])
			#Now pred has the last value
			pred = pred.reshape(1, cf.TIMESTEP_OUT, cf.INPUT_DIM)
			# print pred.shape

			# # Scaleback --only the last one to play nd plot in realtime
			#**********************************************************************
			# SCALE - MUST BE SAVED FROM THE TRAINING DATA, SCALED BACK -- SCALE of TOTAL DATA
			#**********************************************************************
			sb_mfcc, sb_relpos = self.scale_back(pred, mfcc_scaler, relpos_scaler) 
			# print sb_mfcc.shape, sb_relpos.shape

			# Play -- Not detecting sound device, Use a Latop for this
			# self.play_sound_realtime(sb_mfcc)

			# Publish for plot
			# processed data time stamp has delay for processing time
			# when plotting time for prediction will be added to the processing time as plotting latency
			print self.mfcc.shape, self.relpos.shape, sb_mfcc.shape, sb_relpos.shape
			# msg = plot_pub()
			# msg.orig_mfcc =
			# msg.pred_mfcc
			# msg.orig_relpos=
			# msg.pred_relpos = 


	def run(self):
		rospy.init_node('predictor', anonymous=True)
		# while not rospy.is_shutdown():
		rospy.Subscriber('preprocessed_audio', pub_mfcc, self.callback)
		rospy.Subscriber('preprocessed_relpos', pub_relpos, self.callback)
		rospy.spin()
		
def main():
	p = predictor()
	p.graph = tf.get_default_graph()
	p.model = p.define_network(cf.PRED_BATCH_SIZE, cf.TIMESTEP_IN, cf.TIMESTEP_OUT, cf.INPUT_DIM, cf.N_NEURONS)
	p.run()


if __name__ == '__main__':
	main()    
