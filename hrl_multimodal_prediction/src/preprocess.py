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
import gc

################################################
# Trick from 1D to 2D is flattening!!!!!
# First Check the base case 2-10 which worked well before
# Then Convert it into flattened version
################################################

#Use these config only in main
IMAGE_DIM = 3 #image dimension
MFCC_DIM = 3 #audio dimension
INPUT_DIM = MFCC_DIM + IMAGE_DIM #total dimension in LSTM
TIMESTEP_IN = 1
TIMESTEP_OUT = 10
N_NEURONS = TIMESTEP_OUT

BATCH_SIZE = 256
NUM_BATCH = 200 #Total #samples = Num_batch x Batch_size
NB_EPOCH = 500
PRED_BATCH_SIZE = 1

WEIGHT_FILE = './weights/stateful-tanh-denoise-2stack-Tdense-2D-lab.h5'
PLOT = True
DENSE = True #True if TimeDistributedDense layer is used

PROCESSED_DATA_PATH = './processed_data/'

def add_noise(X):
	print 'add_noise to X'
	print X.shape
	batch_size = BATCH_SIZE #32,64,128,256,512,1024,2048
	n = batch_size/16 
	for j in range(NUM_BATCH): 
		for i in xrange(batch_size):
			X[i,:,:,:] = X[i,:,:,:] + np.random.normal(0.0, 0.0025, (X.shape[1], X.shape[2], X.shape[3]) ) 

	# for j in range(NUM_BATCH): 
	# 	for i in range(batch_size):
	# 	  pyplot.plot(X[i,:,:,0])
	# pyplot.show()
	return X

def data_generator(dataset):
	dataset = augment_data(dataset) 
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
	more_data = dataset
	while len(more_data) < BATCH_SIZE*NUM_BATCH - dataset.shape[0]:
		more_data = np.concatenate((more_data, dataset), axis=0)
	print more_data.shape
	diff = BATCH_SIZE*NUM_BATCH - more_data.shape[0]
	more_data = np.concatenate((more_data, dataset[0:diff,:,:]), axis=0)
	print more_data.shape

	return more_data

def format_data(dataset): #dataset.shape=(batchsize=256, datapoints=100, dim=2)
	X, y = [], []
	for i in range(dataset.shape[1] - TIMESTEP_IN - TIMESTEP_OUT + 1):
		x_f = dataset[:, i:i+TIMESTEP_IN, :] 
		y_f = dataset[:, i+TIMESTEP_IN:i+TIMESTEP_IN+TIMESTEP_OUT, :]
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
	inputFile = PROCESSED_DATA_PATH + 'combined_train.npy'
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
	X = X.reshape(X.shape[0],X.shape[1],X.shape[2],INPUT_DIM)
	y = y.reshape(y.shape[0],y.shape[1],y.shape[2],INPUT_DIM) 
	X = np.swapaxes(X, 0, 1)
	y = np.swapaxes(y, 0, 1)
	print 'in main'
	print X.shape, y.shape

	# train, validation split, noise-denoise
	# X = add_noise(X)

	# rescale values to -1, 1 for tanh


	# flatten data to shape into lstm
	if DENSE:
		y = y.reshape(y.shape[0], y.shape[1], 1, y.shape[2]*y.shape[3])
		print y.shape
	else:
		y = y.reshape(y.shape[0], y.shape[1], y.shape[2]*y.shape[3])
		print y.shape

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
	print X_train.shape, X_test.shape, y_train.shape, y_test.shape

	pyplot.plot(X_test[0,:,:,0])
	pyplot.plot(X_test[0,:,:,1])
	pyplot.plot(X_test[0,:,:,2])
	pyplot.show()
	pyplot.plot(y_test[0,:,0,:])
	pyplot.plot(y_test[0,:,0,:])
	pyplot.plot(y_test[0,:,0,:])
	pyplot.show()

	np.save(PROCESSED_DATA_PATH + 'X_train', X_train)
	np.save(PROCESSED_DATA_PATH + 'X_test', X_test)
	np.save(PROCESSED_DATA_PATH + 'y_train', y_train)
	np.save(PROCESSED_DATA_PATH + 'y_test', y_test)

if __name__ == "__main__":
	main()

