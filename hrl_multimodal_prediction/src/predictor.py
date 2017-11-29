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

from predict_subscriber import predict_subscriber #filename
import config as cf
from threading import Thread, Lock
import predmutex as pm

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
	
	def run():
		psub = predict_subscriber()
		r = rospy.Rate(8) # 8hz
		while not rospy.is_shutdown():
			pm.mutex_viz.acquire()
			if not psub.Audio_buffer() and not psub.RelPos_buffer():
				mfcc = psub.Audio_buffer().pop() # takes the last element
				relpos = psub.RelPos_buffer().pop()
				g_actual_mfcc = mfcc_scaler
				g_actual_relpos = relpos
				comb_data, mfcc_scaler, relpos_scaler = preprocess(mfcc, relpos)
				model = define_network(cf.PRED_BATCH_SIZE, cf.TIMESTEP_IN, cf.TIMESTEP_OUT, cf.INPUT_DIM, cf.N_NEURONS)
				pred = predict(model, comb_data)
				scale_back(pred, mfcc_scaler, relpos_scaler)
			else:
				print 'circular buffer empty'
			pm.mutex_viz.release()
			r.sleep()

def main():
	pred = predictor()
	t = Thread(target = pred.run())
	t.start()

if __name__ == '__main__':
	main()    
