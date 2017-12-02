#!/usr/bin/python
#
# Copyright (c) 2017, Georgia Tech Research Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the Georgia Tech Research Corporation nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY GEORGIA TECH RESEARCH CORPORATION ''AS IS'' AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL GEORGIA TECH BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#  \author Michael Park (Healthcare Robotics Lab, Georgia Tech.)

import rosbag
import rospy
import numpy as np
from matplotlib import pyplot 
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from numpy import array
import math
from matplotlib.pylab import *
from mpl_toolkits.axes_grid1 import host_subplot
import matplotlib.animation as animation
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import gc
import librosa
import os, copy, sys
import tensorflow as tf
import numpy.matlib
import scipy.io.wavfile as wav
import scipy as sp
import scipy.interpolate

from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout, RepeatVector, TimeDistributed, Input
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from keras import optimizers
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from keras import backend as K

import config as cf

def define_network(batch_size, time_in, time_out, input_dim, n_neurons):
	model = Sequential()
	model.add(LSTM(input_dim*time_out, batch_input_shape=(batch_size, time_in, input_dim),
					stateful=True, return_sequences=True, activation='tanh'))
	model.add(LSTM(input_dim*time_out, stateful=True, return_sequences=True, activation='tanh'))
	model.add(TimeDistributed(Dense(input_dim*time_out, activation='linear')))
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

def add_noise(X):
	# print 'add_noise to X'
	# print X.shape
	for i in range(X.shape[0]):
		X[i,:,:,:] = X[i,:,:,:] + np.random.normal(0.0, 0.0001, (X.shape[1], X.shape[2], X.shape[3]) ) 

	# print 'addnoise plot'
	# for i in range(43):
	#     pyplot.plot(X[i,:,:,0])
	#     pyplot.plot(X[i,:,:,1])
	#     pyplot.plot(X[i,:,:,2])
	# pyplot.show()
	# for i in range(43):
	#     pyplot.plot(X[i,:,:,3])
	#     pyplot.plot(X[i,:,:,4])
	#     pyplot.plot(X[i,:,:,5])
	# pyplot.show()

	return X

def fit_lstm(model, x_train, x_test, y_train, y_test):
	wait         = 0
	plateau_wait = 0
	min_loss = 1e+15
	patience = 5
	plot_tr_loss = []
	plot_te_loss = []

	print 'orig plot'
	for i in range(43):
		pyplot.plot(x_train[i,:,:,0])
		pyplot.plot(x_train[i,:,:,1])
		pyplot.plot(x_train[i,:,:,2])
	pyplot.show()
	for i in range(43):
		pyplot.plot(x_train[i,:,:,3])
		pyplot.plot(x_train[i,:,:,4])
		pyplot.plot(x_train[i,:,:,5])
	pyplot.show()

	mm = np.load(cf.PROCESSED_DATA_PATH + 'combined_train_minmax.npy')        
	a_min, a_max, i_min, i_max = mm[0], mm[1], mm[2], mm[3]

	for epoch in range(cf.NB_EPOCH):
		#train
		mean_tr_loss = []
		print 'x_train.shape[0]*cf.NUM_BATCH*2'
		print x_train.shape[0]*cf.NUM_BATCH*2
		for i in range(0,x_train.shape[0]*cf.NUM_BATCH*2, cf.BATCH_SIZE): #x_train.shape=BATCH_SIZE
			#per window
			seq_tr_loss = []
			# x = x_train[i:i+BATCH_SIZE]
			# y = y_train[i:i+BATCH_SIZE]
			x, y = x_train, y_train
			x = add_noise(x)
			
			# print x.shape, y.shape
			#normalize
			a_data = x[:,:,:,0:3]
			i_data = x[:,:,:,3:6]
			a_data = normalize(a_data, a_min, a_max)
			i_data = normalize(i_data, i_min, i_max)
			x = np.concatenate((a_data, i_data), axis=3)
			y = y.reshape(y.shape[0], y.shape[1], cf.TIMESTEP_OUT, cf.INPUT_DIM)
			a_data = y[:,:,:,0:3]
			i_data = y[:,:,:,3:6]
			a_data = normalize(a_data, a_min, a_max)
			i_data = normalize(i_data, i_min, i_max)
			y = np.concatenate((a_data, i_data), axis=3)
			y = y.reshape(y.shape[0], y.shape[1], 1, cf.TIMESTEP_OUT*cf.INPUT_DIM)
			# print x.shape, y.shape
			
			# plot after normalize
			# print 'after scale'
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
			# for i in range(cf.BATCH_SIZE):
			#     pyplot.plot(y[i,:,:,0])
			#     pyplot.plot(y[i,:,:,1])
			#     pyplot.plot(y[i,:,:,2])
			# pyplot.show()
			# for i in range(cf.BATCH_SIZE):
			#     pyplot.plot(y[i,:,:,3])
			#     pyplot.plot(y[i,:,:,4])
			#     pyplot.plot(y[i,:,:,5])
			# pyplot.show()
			
			x = np.swapaxes(x, 0, 1)
			y = np.swapaxes(y, 0, 1)

			# This loop is for number of windows - swap above necessary
			for j in range(x.shape[0]):
				tr_loss = model.train_on_batch(x[j], y[j])
				seq_tr_loss.append(tr_loss)
			mean_tr_loss.append( np.mean(seq_tr_loss) )
			model.reset_states()
		tr_loss = np.mean(mean_tr_loss)
		sys.stdout.write('Epoch {} / {} : loss training = {} , loss validating = {}\n'.format(epoch, cf.NB_EPOCH, tr_loss, 0))
		sys.stdout.flush()
		plot_tr_loss.append(tr_loss)

		#test(Validation)
		# This loop is for taking a batch from a large test data
		# Currently just using same data
		if epoch%2: #validate for every other training
			mean_te_loss = []
			for i in xrange(0, x_test.shape[0]*cf.NUM_BATCH, cf.BATCH_SIZE):
				seq_te_loss = []
				# x = x_test[i:i+BATCH_SIZE]
				# y = y_test[i:i+BATCH_SIZE]
				x, y = x_test, y_test

				#normalize
				a_data = x[:,:,:,0:3]
				i_data = x[:,:,:,3:6]
				a_data = normalize(a_data, a_min, a_max)
				i_data = normalize(i_data, i_min, i_max)
				x = np.concatenate((a_data, i_data), axis=3)
				y = y.reshape(y.shape[0], y.shape[1], cf.TIMESTEP_OUT, cf.INPUT_DIM)
				a_data = y[:,:,:,0:3]
				i_data = y[:,:,:,3:6]
				a_data = normalize(a_data, a_min, a_max)
				i_data = normalize(i_data, i_min, i_max)
				y = np.concatenate((a_data, i_data), axis=3)
				y = y.reshape(y.shape[0], y.shape[1], 1, cf.TIMESTEP_OUT*cf.INPUT_DIM)
				# print x.shape

				x = np.swapaxes(x, 0, 1)
				y = np.swapaxes(y, 0, 1)

				# This loop is for number of windows - swap above necessary
				for j in xrange(x.shape[0]):
					te_loss = model.test_on_batch(x[j], y[j])
					seq_te_loss.append(te_loss)
				mean_te_loss.append( np.mean(seq_te_loss) )
				model.reset_states()
			val_loss = np.mean(mean_te_loss)
			sys.stdout.write('Epoch {} / {} : loss training = {} , loss validating = {}\n'.format(epoch, cf.NB_EPOCH, tr_loss, val_loss))
			sys.stdout.flush()   
			plot_te_loss.append(val_loss)

			# Early Stopping
			if val_loss <= min_loss:
				min_loss = val_loss
				wait         = 0
				plateau_wait = 0
				print 'saving model'
				model.save_weights(cf.WEIGHT_FILE+str(tr_loss)+'_'+str(val_loss)+'real_data.h5') 
			else:
				if wait > patience:
					print "Over patience!"
					break
				else:
					wait += 1
					plateau_wait += 1

			#ReduceLROnPlateau
			if plateau_wait > 2:
				old_lr = float(K.get_value(model.optimizer.lr)) #K is a backend
				new_lr = old_lr * 0.2
				K.set_value(model.optimizer.lr, new_lr)
				plateau_wait = 0
				print 'Reduced learning rate {} to {}'.format(old_lr, new_lr)

		gc.collect()    

	# ---------------------------------------------------------------------------------
	# visualize outputs
	print "Training history"
	fig = pyplot.figure(figsize=(10,4))
	ax1 = fig.add_subplot(1, 2, 1)
	pyplot.plot(plot_tr_loss)
	ax1.set_title('loss')
	ax2 = fig.add_subplot(1, 2, 2)
	pyplot.plot(plot_te_loss)
	ax2.set_title('validation loss')
	pyplot.show()

	return model


def main():
	'''
	dataset.shape:: (num_window, batch x N, window_size, dim)
	'''
	print ('Loading training data')
	X_train = np.load(cf.PROCESSED_DATA_PATH + 'X_train.npy')
	y_train = np.load(cf.PROCESSED_DATA_PATH + 'y_train.npy')
	X_test = np.load(cf.PROCESSED_DATA_PATH + 'X_test.npy')
	y_test = np.load(cf.PROCESSED_DATA_PATH + 'y_test.npy')
	print ('Finished loading training data')
	print(X_train.shape)
	print(y_train.shape)
	print(X_test.shape)
	print(y_test.shape)   
	
	np.random.seed(3334)
	print('creating model...')
	lstm_model = define_network(cf.BATCH_SIZE, cf.TIMESTEP_IN, cf.TIMESTEP_OUT, cf.INPUT_DIM, cf.N_NEURONS)
	print('training model...')
	lstm_model = fit_lstm(lstm_model, X_train, X_test, y_train, y_test)

if __name__ == '__main__':
	main()
