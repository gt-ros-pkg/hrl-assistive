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

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from matplotlib import pyplot 
from numpy import array
import numpy as np
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout, RepeatVector, TimeDistributed, Input
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras import optimizers
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from keras import backend as K
import math
from matplotlib.pylab import *
from mpl_toolkits.axes_grid1 import host_subplot
import matplotlib.animation as animation
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.utils import shuffle
import gc
import config as cf


def normalize(y, min_y, max_y):
    y = (y - min_y) / (max_y - min_y)
    return y

def data_generator(dataset):
	dataset = augment_data(dataset)
	# print dataset.shape

	#normalize
	mm = np.load(cf.PROCESSED_DATA_PATH + 'combined_train_minmax.npy')		
	a_min, a_max, i_min, i_max = mm[0], mm[1], mm[2], mm[3]
	a_data = dataset[:,:,0:3]
	i_data = dataset[:,:,3:6]
	a_data = normalize(a_data, a_min, a_max)
	i_data = normalize(i_data, i_min, i_max)
	dataset = np.concatenate((a_data, i_data), axis=2)
	# print dataset.shape

	X, y = format_data(dataset) # X.shape=(Batch=256,10,2), y.shape=(256,10,2)
	return X,y #(num_window, batch, window_size, dim)

def augment_data(dataset):
	# for i in range(0, dataset.shape[0]):
	# 	pyplot.plot(dataset[i,:,0])
	# 	pyplot.plot(dataset[i,:,1])
	# 	pyplot.plot(dataset[i,:,2])
	# 	pyplot.show()
	# 	pyplot.plot(dataset[i,:,3])
	# 	pyplot.plot(dataset[i,:,4])
	# 	pyplot.plot(dataset[i,:,5])
	# 	pyplot.show()

	# Randomness present within the collected data already
	# comback and augment later if necessary
	# Original - has memory issue
	# more_data = dataset
	# while len(more_data) < BATCH_SIZE*NUM_BATCH - dataset.shape[0]:
	# 	more_data = np.concatenate((more_data, dataset), axis=0)
	# print more_data.shape
	# diff = BATCH_SIZE*NUM_BATCH - more_data.shape[0]
	# more_data = np.concatenate((more_data, dataset[0:diff,:,:]), axis=0)
	# print more_data.shape

	more_data = dataset
	while more_data.shape[0] < cf.BATCH_SIZE - dataset.shape[0]:
		more_data = np.concatenate((more_data, dataset), axis=0)
	print more_data.shape

	diff = cf.BATCH_SIZE - more_data.shape[0]
	more_data = np.concatenate((more_data, dataset[0:diff,:,:]), axis=0)
	print more_data.shape

	return more_data


def format_data(dataset): #dataset.shape=(batchsize=256, datapoints=100, dim=2)
	X, y = [], []
	for i in range(dataset.shape[1] - cf.TIMESTEP_IN - cf.TIMESTEP_OUT + 1):
		x_f = dataset[:, i:i+cf.TIMESTEP_IN, :] 
		y_f = dataset[:, i+cf.TIMESTEP_IN : i+cf.TIMESTEP_IN+cf.TIMESTEP_OUT, :]
		X.append(x_f)
		y.append(y_f)
	X = np.array(X)
	y = np.array(y)
	print 'windowed data'
	print X.shape, y.shape
	return X, y

def main():
	'''
	dataset.shape:: (num_window, batch x N, window_size, dim)
	'''
	# Read and plot and check for all 43 data
	inputFile = cf.PROCESSED_DATA_PATH + 'combined_train.npy'
	#Load up the training data
	print ('Loading raw data read from rosbag')
	dataset = np.load(inputFile)
	print dataset.shape

	# augment data
	# criteria:: padding for timing, slight updown shift?, slight amplify?
	# These randomnesses are all included in the data
	# If the data overfits and doesn't work than comeback and either collect more data or augemnt well
	# Skip for now
	X, y = data_generator(dataset)
	X = X.reshape(X.shape[0],X.shape[1],X.shape[2],cf.INPUT_DIM)
	y = y.reshape(y.shape[0],y.shape[1],y.shape[2],cf.INPUT_DIM) 
	X = np.swapaxes(X, 0, 1)
	y = np.swapaxes(y, 0, 1)
	print 'in main'
	print X.shape, y.shape
	gc.collect()

	# train, validation split, noise-denoise
	# X = add_noise(X)

	# flatten data to shape into lstm
	if cf.DENSE:
		y = y.reshape(y.shape[0], y.shape[1], 1, y.shape[2]*y.shape[3])
	else:
		y = y.reshape(y.shape[0], y.shape[1], y.shape[2]*y.shape[3])
	print X.shape, y.shape

	# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
	# print X_train.shape, X_test.shape, y_train.shape, y_test.shape
	# Manual way but roughly 1/3 for validation set
	# X_train = np.concatenate((X, X), axis=0)
	# y_train = np.concatenate((y, y), axis=0)
	# X_train = shuffle(X_train, random_state=42)
	# X_train = shuffle(X_train, random_state=42)

	# X_train = shuffle(X, random_state=42)
	# y_train = shuffle(y, random_state=42)	
	# X_test = shuffle(X, random_state=42)
	# y_test = shuffle(y, random_state=42)

	X_train, X_test, y_train, y_test = X, X, y, y
	print X_train.shape, X_test.shape, y_train.shape, y_test.shape


	pyplot.plot(X_test[0,:,:,0])
	pyplot.plot(X_test[0,:,:,1])
	pyplot.plot(X_test[0,:,:,2])
	pyplot.show()
	pyplot.plot(X_test[0,:,:,3])
	pyplot.plot(X_test[0,:,:,4])
	pyplot.plot(X_test[0,:,:,5])
	pyplot.show()

	np.save(cf.PROCESSED_DATA_PATH + 'X_train', X_train)
	np.save(cf.PROCESSED_DATA_PATH + 'X_test', X_test)
	np.save(cf.PROCESSED_DATA_PATH + 'y_train', y_train)
	np.save(cf.PROCESSED_DATA_PATH + 'y_test', y_test)

if __name__ == "__main__":
	main()

