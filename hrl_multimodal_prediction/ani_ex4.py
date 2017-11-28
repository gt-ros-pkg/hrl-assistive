import numpy
from matplotlib.pylab import *
from mpl_toolkits.axes_grid1 import host_subplot
import matplotlib.animation as animation
import librosa
import scipy.io.wavfile as wav
import scipy as sp
import scipy.interpolate

# Sent for figure
font = {'size'   : 9}
matplotlib.rc('font', **font)

# Setup figure and subplots
f0 = figure(num = 0, figsize = (12, 8))#, dpi = 100)
f0.suptitle("ARtag & Audio combined Prediction", fontsize=12)
ax01 = subplot2grid((4, 2), (0, 0))
ax02 = subplot2grid((4, 2), (1, 0))
ax03 = subplot2grid((4, 2), (2, 0))
ax04 = subplot2grid((4, 2), (3, 0))

ax05 = subplot2grid((4, 2), (0, 1))
ax06 = subplot2grid((4, 2), (1, 1))
ax07 = subplot2grid((4, 2), (2, 1))
ax08 = subplot2grid((4, 2), (3, 1))

def updateData(self):
	image = np.loadtxt('./predicted/testdata_XYZ.txt')	
	# image = np.rollaxis(image, 1, 0)
	x=image[0]
	y=image[1]
	z=image[2]
	l = image.shape[1]
	t = np.linspace(0,4,l)

	#orig xyz
	xs = []
	ys = []
	for i in range(l):
		xs.append(t[i])
		ys.append(x[i])
	ax01.clear()
	ax01.set_title('orig pos x')
	ax01.grid(True)
	ax01.set_xlabel("t")
	ax01.set_ylabel("x")
	ax01.plot(xs, ys)

	xs = []
	ys = []
	for i in range(l):
		xs.append(t[i])
		ys.append(y[i])
	ax02.clear()
	ax02.set_title('orig pos y')
	ax02.grid(True)
	ax02.set_xlabel("t")
	ax02.set_ylabel("y")
	ax02.plot(xs, ys)

	xs = []
	ys = []
	for i in range(l):
		xs.append(t[i])
		ys.append(z[i])
	ax03.clear()
	ax03.set_title('orig pos z')
	ax03.grid(True)
	ax03.set_xlabel("t")
	ax03.set_ylabel("z")
	ax03.plot(xs, ys)

	#audio
	audio, sr = librosa.load('./predicted/testdata_MFCC.wav', mono=True)
	l_a = audio.shape[0]
	t_a = np.linspace(0,4,l_a)

	mfccs = librosa.feature.mfcc(audio, n_mfcc=3) #default hop_length=512
	l_a = mfccs.shape[1]
	t_a = np.linspace(0,4,l_a)
	mfccs = mfccs[0]
	xs = []
	ys = []
	for i in range(l_a):
		xs.append(t_a[i])
		ys.append(mfccs[i])
	
	# xs = []
	# ys = []
	# for i in range(l_a):
	# 	xs.append(t_a[i])
	# 	ys.append(audio[i])

	ax04.clear()
	ax04.set_title('orig audio')
	ax04.grid(True)
	ax04.set_xlabel("t")
	ax04.set_ylabel("audio (db)")
	ax04.plot(xs, ys)

	####################################
	#pred
	# pred_image = np.loadtxt('./csv/predicted/combined_predict_testdata20.txt')#predicted/combined_predict_testdata13.txt')	
	# # pred_image = pred_image[:,26:]
	# # pred_image = np.rollaxis(pred_image, 1, 0)
	# # print pred_image.shape
	# pred_x=pred_image[0]
	# pred_y=pred_image[1]
	# pred_z=pred_image[2]
	# pred_l = pred_image.shape[1]
	# pred_t = np.linspace(0,2,pred_l)

	# xs = []
	# ys = []
	# for i in range(pred_l):
	# 	xs.append(pred_t[i])
	# 	ys.append(pred_x[i])
	# ax05.clear()
	# ax05.set_title('predicted pos x')
	# ax05.grid(True)
	# ax05.set_xlabel("t")
	# ax05.set_ylabel("x")
	# ax05.plot(xs, ys)

	# xs = []
	# ys = []
	# for i in range(pred_l):
	# 	xs.append(pred_t[i])
	# 	ys.append(pred_y[i])
	# ax06.clear()
	# ax06.set_title('predicted pos y')
	# ax06.grid(True)
	# ax06.set_xlabel("t")
	# ax06.set_ylabel("y")
	# ax06.plot(xs, ys)

	# xs = []
	# ys = []
	# for i in range(pred_l):
	# 	xs.append(pred_t[i])
	# 	ys.append(pred_z[i])
	# ax07.clear()
	# ax07.set_title('predicted pos z')
	# ax07.grid(True)
	# ax07.set_xlabel("t")
	# ax07.set_ylabel("z")
	# ax07.plot(xs, ys)

	# #pred audio
	# pred_audio, pred_sr = librosa.load('./sounds/predicted/combined_predict_testdata20FromMFCC3.wav', mono=True)
	# pred_mfccs = librosa.feature.mfcc(pred_audio, n_mfcc=3) #default hop_length=512
	# pred_l = 86
	# pred_t = np.linspace(0,2,pred_l)
	# pred_mfccs = pred_mfccs[0]
	
	# xs = []
	# ys = []
	# for i in range(pred_l):
	# 	xs.append(pred_t[i])
	# 	ys.append(pred_mfccs[i])
	# ax08.clear()

	# ax08.set_title('predicted audio')
	# ax08.grid(True)
	# ax08.set_xlabel("t")
	# ax08.set_ylabel("audio (db)")
	# ax08.plot(xs, ys)

# interval: draw new frame every 'interval' ms
# frames: number of frames to draw
simulation = animation.FuncAnimation(f0, updateData, blit=False, frames=200, interval=20, repeat=False)
plt.show()

