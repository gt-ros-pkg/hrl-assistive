#Test script for offline predicting
import numpy as np
import config as cf
import matplotlib.pyplot as plt
import os, copy, sys
import tensorflow as tf
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout, RepeatVector, TimeDistributed, Input
from keras.models import Sequential, Model
from keras import optimizers
from keras import backend as K

def define_network(batch_size, time_in, time_out, input_dim, n_neurons):
	model = Sequential()
	model.add(LSTM(input_dim*time_out, batch_input_shape=(batch_size, time_in, input_dim),
					stateful=True, return_sequences=True, activation='tanh'))
	model.add(LSTM(input_dim*time_out, stateful=True, return_sequences=True, activation='tanh'))
	model.add(TimeDistributed(Dense(input_dim*time_out, activation='linear')))

	model.load_weights('./weights/0.00311671_0.0031128real_data.h5')
	model.compile(loss='mse', optimizer='RMSprop')
	print model.summary()
	print "Inputs: {}".format(model.input_shape)
	print "Outputs: {}".format(model.output_shape)
	return model

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

def main():	
	# --------------------------------------------------------------------------
	# (1) Read Data and Combine
	# This Assumes we are receving data from predict_subscriber.py except no data loss so 
	# test data exactly matches train data
	mfcc = np.load(cf.ROSBAG_UNPACK_PATH + 'data1_mfccs_.npy')
	relpos = np.load(cf.ROSBAG_UNPACK_PATH + 'data1_relpos_intp_.npy')

	mfcc = np.swapaxes(mfcc, 0, 1)
	print mfcc.shape, relpos.shape
	combined = np.concatenate((mfcc, relpos), axis=1)
	print combined.shape #(173, 6)

	# --------------------------------------------------------------------------
	# # (2) Convert to MFCC, Normalize, PreProcess for Prediction
	X, y = [], []
	for i in range(combined.shape[0] - cf.TIMESTEP_IN - cf.TIMESTEP_OUT):
		X.append(combined[i:i+cf.TIMESTEP_IN])
		y.append(combined[i+cf.TIMESTEP_IN:i+cf.TIMESTEP_IN+cf.TIMESTEP_OUT])
	X = np.array(X)
	y = np.array(y)
	print X.shape, y.shape
	X = X.reshape(X.shape[0], cf.PRED_BATCH_SIZE, cf.TIMESTEP_IN, cf.INPUT_DIM)
	print X.shape, y.shape

	mm = np.load(cf.PROCESSED_DATA_PATH + 'combined_train_minmax.npy')		
	a_min, a_max, i_min, i_max = mm[0], mm[1], mm[2], mm[3]
	
	# Rescale perFeature, save Scaler
	norm_mfcc = normalize(X[:,:,:,0:3], a_min, a_max)
	norm_relpos = normalize(X[:,:,:,3:6], i_min, i_max)
	# Combine two modes 
	comb_data = np.concatenate((norm_mfcc, norm_relpos), axis=3)
	# print comb_data.shape
	X_norm = np.array(comb_data)

	# --------------------------------------------------------------------------
	# # (3) Predict and scale back
	new_model = define_network(cf.PRED_BATCH_SIZE, cf.TIMESTEP_IN, cf.TIMESTEP_OUT, cf.INPUT_DIM, cf.N_NEURONS)
	rst = []
	for i in range(0, X.shape[0],4):
		p_tmp = new_model.predict_on_batch(X_norm[i]) #, batch_size=PRED_BATCH_SIZE)
		rst.append(p_tmp)
	rst = np.array(rst)
	print 'results shape'
	print rst.shape

	# convertin back predicted y
	if cf.INPUT_DIM > 1:
		rst = rst.reshape(rst.shape[0], rst.shape[1], cf.TIMESTEP_OUT, cf.INPUT_DIM)
		print rst.shape

	# Rescale perFeature, save Scaler
	sb_mfcc = scale_back(X_norm[:,:,:,0:3], a_min, a_max)
	sb_relpos = scale_back(X_norm[:,:,:,3:6], i_min, i_max)
	# Combine two modes 
	sb_comb_data = np.concatenate((sb_mfcc, sb_relpos), axis=3)
	# print comb_data.shape
	X_sb = np.array(sb_comb_data)

	# Rescale perFeature, save Scaler
	sb_mfcc = scale_back(rst[:,:,:,0:3], a_min, a_max)
	sb_relpos = scale_back(rst[:,:,:,3:6], i_min, i_max)
	# Combine two modes 
	sb_comb_data = np.concatenate((sb_mfcc, sb_relpos), axis=3)
	# print comb_data.shape
	rst_sb = np.array(sb_comb_data)

	# --------------------------------------------------------------------------
	# # (4) Plot and Check
	# for i in range(cf.BATCH_SIZE):
	#     pyplot.plot(x[i,:,:,0])
	#     pyplot.plot(x[i,:,:,1])
	#     pyplot.plot(x[i,:,:,2])
	# pyplot.show()
	# for i in range(cf.BATCH_SIZE):
	#     pyplot.plot(x[i,:,:,3])
	#     pyplot.plot(x[i,:,:,4])
	#     pyplot.plot(x[i,:,:,5])
	# pyplot.show()

	# Batchsize = 1
	plt.plot(X_sb[:,0,:,0], color='blue')	#(num_timestep, batchsize, timestep, feature)
	plt.plot(X_sb[:,0,:,1], color='blue')
	plt.plot(X_sb[:,0,:,2], color='blue')
	for i in range(0, rst.shape[0]):
		xaxis2 = [w for w in range(i+cf.TIMESTEP_IN, i+cf.TIMESTEP_IN+cf.TIMESTEP_OUT)]
		plt.plot(xaxis2 ,rst_sb[i,0,:,0:3], color='red')
	plt.show()
	plt.plot(X_sb[:,0,:,3], color='blue')	#(num_timestep, batchsize, timestep, feature)
	plt.plot(X_sb[:,0,:,4], color='blue')
	plt.plot(X_sb[:,0,:,5], color='blue')
	for i in range(0, rst.shape[0]):
		xaxis2 = [w for w in range(i+cf.TIMESTEP_IN, i+cf.TIMESTEP_IN+cf.TIMESTEP_OUT)]
		plt.plot(xaxis2 ,rst_sb[i,0,:,3:6], color='red')
	plt.show()

	# plt.plot(X_sb[:,0,0,0:3], color='blue')
	# for i in range(0, rst.shape[0]):
	# 	xaxis2 = [w for w in range(i+cf.TIMESTEP_IN, i+cf.TIMESTEP_IN+cf.TIMESTEP_OUT)]
	# 	plt.plot(xaxis2 ,rst_sb[i,0,:,0:3], color='red')
	# plt.show()
	# plt.plot(X_sb[:,0,0,3:6], color='blue')
	# for i in range(0, rst.shape[0]):
	# 	xaxis2 = [w for w in range(i+cf.TIMESTEP_IN, i+cf.TIMESTEP_IN+cf.TIMESTEP_OUT)]
	# 	plt.plot(xaxis2 ,rst_sb[i,0,:,3:6], color='red')
	# plt.show()

if __name__ == '__main__':
	main()
