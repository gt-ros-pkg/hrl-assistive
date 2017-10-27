import os, copy, sys

# util
import numpy as np
import math
import struct
import array
try:
    from features import mfcc
except:
    from python_speech_features import mfcc
from scipy import signal, fftpack, conj, stats

import scipy.io.wavfile as wav
import librosa
import librosa.display

def invlogamplitude(S):
#"""librosa.logamplitude is actually 10_log10, so invert that."""
    return 10.0**(S/10.0)

def audio_creator(): 
    CHUNK = 1024
    CHANNELS = 1
    RATE = 44100
    RECORD_SECONDS = 2
    WAVE_OUTPUT_FILENAME = "data13crop4.wav" #n_mfcc=24
    #WAVE_OUTPUT_FILENAME = "AudioBookCrop.wav" #n_mfcc=12
    #WAVE_OUTPUT_FILENAME = "data5crop4.wav"  #n_mfcc=2
    
    y, sr = librosa.load('./sounds/cropped/' + WAVE_OUTPUT_FILENAME)
    print len(y)

    #calculate mfccs
    #Y = librosa.stft(y)
    #print Y.shape

    #####################################
    # Original #
    mfccs = librosa.feature.mfcc(y, n_mfcc=2)# default hop_length=512, hop_length=int(0.01*sr))
    print mfccs
    print mfccs.shape
    ############################

    #build reconstruction mappings
    n_mfcc = mfccs.shape[0]
    n_mel = 128
    dctm = librosa.filters.dct(n_mfcc, n_mel)
    n_fft = 2048
    mel_basis = librosa.filters.mel(sr, n_fft)

    #Empirical scaling of channels to get ~flat amplitude mapping.
    bin_scaling = 1.0/np.maximum(0.0005, np.sum(np.dot(mel_basis.T, mel_basis), axis=0))
    #Reconstruct the approximate STFT squared-magnitude from the MFCCs.
    recon_stft = bin_scaling[:, np.newaxis] * np.dot(mel_basis.T, invlogamplitude(np.dot(dctm.T, mfccs)))
    #Impose reconstructed magnitude on white noise STFT.
    excitation = np.random.randn(y.shape[0])
    E = librosa.stft(excitation)
    recon = librosa.istft(E/np.abs(E)*np.sqrt(recon_stft))
    #print recon
    #print recon.shape

    wav.write('./sounds/cropped/' + WAVE_OUTPUT_FILENAME +'FromMFCC', sr, recon)

def main():
    audio_creator()

if __name__ == '__main__':
    main()


