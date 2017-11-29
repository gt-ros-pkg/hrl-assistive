# One Core per thread
# publisher nodes, predict_subscriber, predictor, visualizer all have its own core

from predictor import predictor
import config as cf
from threading import Thread, Lock
import predmutex as pm
from matplotlib.pylab import *
from mpl_toolkits.axes_grid1 import host_subplot
import matplotlib.animation as animation

class visualizer():
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
	
	# def plot_realtime(self, p_mfcc, p_relpos, a_mfcc, a_relpos): #p for predict, a for actual


	def run(self):
		p = predictor()
		while True:
			pm.mutex_viz.acquire()
			if p.g_mfcc is not None and p.g_relpos is not None:
				play_sound_realtime(p.g_pred_mfcc)
				# Sent for figure
				# font = {'size'   : 9}
				# matplotlib.rc('font', **font)
				# # Setup figure and subplots
				# f0 = figure(num = 0, figsize = (12, 8))#, dpi = 100)
				# f0.suptitle("ARtag & Audio combined Prediction", fontsize=12)
				# ax01 = subplot2grid((2, 2), (0, 0))
				# ax02 = subplot2grid((2, 2), (1, 0))
				# ax03 = subplot2grid((2, 2), (0, 1))
				# ax04 = subplot2grid((2, 2), (1, 1))
				# simulation = animation.FuncAnimation(f0, updateData, blit=False, frames=200, interval=20, repeat=False)
				# plt.show()
			pm.mutex_viz.release()

def main():
	viz = visualizer()
	t = Thread(target = viz.run())
	t.start()

if __name__ == '__main__':
	main()    

