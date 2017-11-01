import librosa
import librosa.display
import os, copy, sys
import numpy as np
import numpy.matlib
import scipy.io.wavfile as wav
import scipy as sp
import scipy.interpolate
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

SOUND_FOLDER = './sounds/cropped/'
CSV_FOLDER = './csv/'
COMBINED_FILENAME = {'data1.txt':'data1crop4.wav', 'data2.txt':'data2crop4.wav', 'data5.txt':'data5crop4.wav', 'data13.txt':'data13crop4.wav'}
N_MFCC =5
SINGLE_FILENAME_IMAGE = 'combined_predict_testdata13.txt'
SINGLE_FILENAME_AUDIO = 'data13crop4.wav'

def plot_image_only():
	image = np.loadtxt(CSV_FOLDER + 'predicted/' + SINGLE_FILENAME_IMAGE)
	# image = np.loadtxt('./csv/data1.txt')
	print image.shape
	image = np.rollaxis(image, 1, 0)
	x=image[0]
	y=image[1]
	z=image[2]
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.plot(x,y,z, label='parametric curve')
	ax.legend()
	plt.show()

#https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html
def plot_all_data():
	combined_filename = COMBINED_FILENAME
	for image_filename in combined_filename:
		audio_filename = combined_filename[image_filename]
		image = np.loadtxt(CSV_FOLDER + image_filename)
		audio, sr = librosa.load(SOUND_FOLDER + audio_filename, mono=True)
		mfccs = librosa.feature.mfcc(audio, n_mfcc=N_MFCC) #default hop_length=512
		# mfccs = np.rollaxis(mfccs, 1, 0)

		print image.shape
		image = np.rollaxis(image, 1, 0)
		x=image[0]
		y=image[1]
		z=image[2]
		fig = plt.figure()
		ax = fig.gca(projection='3d')
		ax.plot(x,y,z, label='parametric curve')
		ax.legend()

		plt.figure(figsize=(10, 4))
		librosa.display.specshow(mfccs, x_axis='time')
		plt.colorbar()
		plt.title('MFCC')
		plt.tight_layout()
		plt.show()

def main():
	plot_image_only()
	# plot_all_data()
	return 1

if __name__ == "__main__":
    sys.exit(main())
