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

import wave
import random
import struct
import datetime
import numpy as np
import math
import pyaudio
import struct
import sys
import scipy.io.wavfile as wav

# def find_input_device():
#         device_index = None
#         for i in range(self.p.get_device_count()):
#             devinfo = self.p.get_device_info_by_index(i)
#             print('Device %d: %s'%(i, devinfo['name']))

#             for keyword in ['mic', 'input', 'icicle', 'creative']:
#                 if keyword in devinfo['name'].lower():
#                     print('Found an input: device %d - %s'%(i, devinfo['name']))
#                     device_index = i
#                     return device_index

#         if device_index is None:
#             print('No preferred input found; using default input device.')

#         return device_index

def record_audio():
	CHUNK = 1024
	FORMAT = pyaudio.paInt16
	CHANNELS = 2
	RATE = 44100
	RECORD_SECONDS = 25
	WAVE_OUTPUT_FILENAME = "changmoFull.wav"

	p = pyaudio.PyAudio()

	# deviceIndex = find_input_device()
	device_index = None
	for i in range(p.get_device_count()):
		devinfo = p.get_device_info_by_index(i)
		print('Device %d: %s'%(i, devinfo['name']))

		for keyword in ['mic', 'input', 'icicle', 'creative']:
			if keyword in devinfo['name'].lower():
				print('Found an input: device %d - %s'%(i, devinfo['name']))
				device_index = i

	if device_index is None:
		print('No preferred input found; using default input device.')
    ###############################################################
	
	devInfo = p.get_device_info_by_index(device_index)
	print 'Audio device:', device_index
	print 'Sample rate:', devInfo['defaultSampleRate']
	print 'Max find_input_deviceut channels:',  devInfo['maxInputChannels']

	stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

	print "* Recording audio..."

	frames = []
	data = []
	for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
		print "working??-1"
		data.append(stream.read(CHUNK))
		print "working??-2"

	for s in data:
		frames.append(np.fromstring(s, np.int16))		

	print frames
	#numpydata = np.reshape(frames, (CHUNK, CHANNELS))
	numpydata = np.hstack(frames)
	numpydata = np.reshape(numpydata, (len(numpydata)/2 ,2))

	print numpydata
	print "* done\n" 
	# print "datatype: " + str(type(data))
	# print "frametype: " + str(type(frames))
	# print "frametype2: " + str(type(frames[0]))

	stream.stop_stream()
	stream.close()
	p.terminate()

	wav.write(WAVE_OUTPUT_FILENAME, RATE, numpydata)
	# wf = wave.open(WAVE_OUTPUT_FILENAME, 'w')
	# wf.setnchannels(CHANNELS)
	# wf.setsampwidth(p.get_sample_size(FORMAT))
	# wf.setframerate(RATE)
	# wf.writeframes(b''.join(frames))
	# wf.close()

def main():
	record_audio()

if __name__ == '__main__':
    main()