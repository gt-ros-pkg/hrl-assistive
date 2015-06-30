#!/usr/bin/env python

import math
import glob
import wave
import rospy
import struct
import pyaudio
import numpy as np
from threading import Thread
import scipy.signal as signal

class tool_audio(Thread):
    MAX_INT = 32768.0
    CHUNK   = 1024 #frame per buffer
    RATE    = 44100 #sampling rate
    UNIT_SAMPLE_TIME = 1.0 / float(RATE)
    CHANNEL = 2 #number of channels
    FORMAT  = pyaudio.paInt16
    DTYPE   = np.int16

    def __init__(self):
        super(tool_audio, self).__init__()
        self.daemon = True
        self.cancelled = False

        self.init_time = 0.
        self.noise_freq_l = None
        self.noise_band = 150.0
        self.noise_amp_num = 0 #20 #10
        self.noise_amp_thres = 0.0
        self.noise_amp_mult = 2.0
        self.noise_bias = 0.0

        self.audio_freq = np.fft.fftfreq(self.CHUNK, self.UNIT_SAMPLE_TIME)
        self.audio_data = []
        self.audio_amp  = []

        self.audio_data_raw = []

        self.time_data = []

        self.b, self.a = self.butter_bandpass(1, 400, self.RATE, order=6)


        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.FORMAT, channels=self.CHANNEL, rate=self.RATE, input=True, frames_per_buffer=self.CHUNK)
        rospy.logout('Done subscribing audio')
        print 'Done subscribing audio'

    def run(self):
        """Overloaded Thread.run, runs the update
        method once per every xx milliseconds."""

        self.stream.start_stream()

        # rate = rospy.Rate(1000) # 25Hz, nominally.
        while not self.cancelled:
            self.log()
            # rate.sleep()

    def log(self):

        data = self.stream.read(self.CHUNK)
        self.time_data.append(rospy.get_time() - self.init_time)
        self.audio_data_raw.append(data)

        audio_data = np.fromstring(data, self.DTYPE)
        ## audio_data = signal.lfilter(self.b, self.a, audio_data)

        # Exclude low rms data
        ## amp = self.get_rms(data)
        ## if amp < self.noise_amp_thres*2.0:
        ##     audio_data = audio_data*np.exp( - self.noise_amp_mult*(self.noise_amp_thres - amp))

        ## audio_data -= self.noise_bias
        new_F = np.fft.fft(audio_data / float(self.MAX_INT))  #normalization & FFT

        # Remove noise
        ## for noise_freq in self.noise_freq_l:
        ##     new_F = np.array([self.filter_rule(x,self.audio_freq[j], noise_freq, self.noise_band) for j, x in enumerate(new_F)])

        ## audio_data = np.fft.ifft(new_F)*float(self.MAX_INT)

        self.audio_amp.append(new_F)
        # TODO This can be removed to save space
        self.audio_data.append(audio_data)

    def cancel(self):
        """End this timer thread"""
        self.cancelled = True
        rospy.sleep(1.0)

        self.stream.stop_stream()
        self.stream.close()

    def stopWav(self, file_name):
        wav_list = glob.glob('*.wav')
        max_num = 0
        for wavs in wav_list:
            if wavs.find(file_name) >= 0:
                num = int(wavs.split('_')[-1].split('.')[0])
                if max_num < num:
                    max_num = num
        max_num = int(max_num) + 1
        wav_name = file_name + '_' + str(max_num) + '.wav'

        print "Wav file name: ", wav_name

        wf = wave.open(wav_name, 'wb')
        wf.setnchannels(self.CHANNEL)
        wf.setsampwidth(self.p.get_sample_size(tool_audio.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(self.audio_wav_frames))
        wf.close()

        print 'Closing... audio wav recording has been saved...'



    def skip(self, seconds):
        samples = int(seconds * self.RATE)
        count = 0
        while count < samples:
            self.stream.read(self.CHUNK)
            count += self.CHUNK
            rospy.sleep(0.01)

    ## def save_audio(self):

    ##     ## RECORD_SECONDS = 9.0

    ##     string_audio_data = np.array(self.audio_data, dtype=self.DTYPE).tostring()
    ##     import wave
    ##     WAVE_OUTPUT_FILENAME = "/home/dpark/git/pyaudio/test/output.wav"
    ##     wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    ##     wf.setnchannels(self.CHANNEL)
    ##     wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
    ##     wf.setframerate(self.RATE)
    ##     wf.writeframes(b''.join(string_audio_data))
    ##     wf.close()


        ## pp.figure()
        ## pp.subplot(211)
        ## pp.plot(new_frames,'b-')
        ## pp.plot(new_filt_frames,'r-')

        ## pp.subplot(212)
        ## pp.plot(f[:n/10],np.abs(F[:n/10]),'b')
        ## if new_F is not None:
        ##     pp.plot(f[:n/10],np.abs(new_F[:n/10]),'r')
        ## pp.stem(noise_freq_l, values, 'k-*', bottom=0)
        ## pp.show()


    def reset(self):

        self.skip(1.0)
        rospy.sleep(1.0)

        # Get noise frequency
        # frames=None

        ## for i in range(0, int(self.RATE/self.CHUNK * RECORD_SECONDS)):
        data=self.stream.read(self.CHUNK)
        audio_data=np.fromstring(data, self.DTYPE)

        # if frames is None: frames = audio_data
        # else: frames = np.hstack([frames, audio_data])
        self.noise_amp_thres = self.get_rms(data)

        ## self.noise_bias = np.mean(audio_data)
        ## audio_data -= self.noise_bias

        F = np.fft.fft(audio_data / float(self.MAX_INT))  #normalization & FFT
        f  = np.fft.fftfreq(len(F), self.UNIT_SAMPLE_TIME)
        n=len(f)

        import heapq
        values = heapq.nlargest(self.noise_amp_num, F[:n/2]) #amplitude

        self.noise_freq_l = []
        for value in values:
            self.noise_freq_l.append([f[j] for j, k in enumerate(F[:n/2]) if k.real == value.real])
        self.noise_freq_l = np.array(self.noise_freq_l)

        print "Amplitude threshold: ", self.noise_amp_thres
        print "Noise bias: ", self.noise_bias

        ## self.skip(1.0)
        self.stream.stop_stream()

        ## #temp
        ## ## F1 = F[:n/2]
        ## for noise_freq in self.noise_freq_l:
        ##     F = np.array([self.filter_rule(x,f[j], noise_freq, self.noise_band) for j, x in enumerate(F)])
        ## ## new_F = np.hstack([F1, F1[::-1]])
        ## new_F = F

        ## temp_audio_data = np.fft.ifft(new_F) * float(self.MAX_INT)
        ## print len(temp_audio_data), self.noise_freq_l

        ## pp.figure()
        ## pp.subplot(211)
        ## pp.plot(audio_data,'r-')
        ## pp.plot(temp_audio_data,'b-')

        ## pp.subplot(212)
        ## pp.plot(f[:n/4],np.abs(F[:n/4]),'b')
        ## pp.stem(self.noise_freq_l, values, 'r-*', bottom=0)
        ## pp.show()
        ## ## raw_input("Enter anything to start: ")

    @staticmethod
    def filter_rule(x, freq, noise_freq, noise_band):
        if np.abs(freq) > noise_freq+noise_band or np.abs(freq) < noise_freq-noise_band:
            return x
        else:
            return 0

    @staticmethod
    def butter_bandpass(lowcut, highcut, fs, order=5):
        """
        fs: sampling frequency
        """
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        return b, a


    def get_rms(self, block):
        # Copy from http://stackoverflow.com/questions/4160175/detect-tap-with-pyaudio-from-live-mic

        # RMS amplitude is defined as the square root of the
        # mean over time of the square of the amplitude.
        # so we need to convert this string of bytes into
        # a string of 16-bit samples...

        # we will get one short out for each
        # two chars in the string.
        count = len(block)/2
        format = "%dh" % count
        shorts = struct.unpack( format, block )

        # iterate over the block.
        sum_squares = 0.0
        for sample in shorts:
        # sample is a signed short in +/- 32768.
        # normalize it to 1.0
            n = sample / self.MAX_INT
            sum_squares += n*n

        return math.sqrt( sum_squares / count )

