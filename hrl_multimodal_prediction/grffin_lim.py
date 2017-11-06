import os, copy, sys

# util
import numpy as np
import math
import pyaudio
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

def griffinlim(spectrogram, n_iter = 100, window = 'hann', n_fft = 2048, hop_length = -1, verbose = False):
    if hop_length == -1:
        hop_length = n_fft / 4

    angles = np.exp(2j * np.pi * np.random.rand(*spectrogram.shape))

    t = tqdm(range(n_iter), ncols=100, mininterval=2.0, disable=not verbose)
    
    for i in t:
        print spectrogram.shape
        print angles.shape
        full = np.abs(spectrogram).astype(np.complex) * angles
        inverse = librosa.istft(full, hop_length = hop_length, window = window)
        rebuilt = librosa.stft(inverse, n_fft = n_fft, hop_length = hop_length, window = window)
        angles = np.exp(1j * np.angle(rebuilt))

        if verbose:
            diff = np.abs(spectrogram) - np.abs(rebuilt)
            t.set_postfix(loss=np.linalg.norm(diff, 'fro'))

    full = np.abs(spectrogram).astype(np.complex) * angles
    inverse = librosa.istft(full, hop_length = hop_length, window = window)

    return inverse

def audio_creator(): 
    RATE = 44100
    RECORD_SECONDS = 4
    WAVE_OUTPUT_FILENAME = "AudioBookCrop.wav" 

    # Generate mfccs from a time series
    y, sr = librosa.load(WAVE_OUTPUT_FILENAME)
    # print y
    # print y.shape #46080
    
    mfccs = librosa.feature.mfcc(y)
    #mfccs = librosa.feature.melspectrogram(y)
    back = griffinlim(mfccs)
    
    wav.write(WAVE_OUTPUT_FILENAME+'FromMFCC', sr, back)

def main():
    audio_creator()

if __name__ == '__main__':
    main()


