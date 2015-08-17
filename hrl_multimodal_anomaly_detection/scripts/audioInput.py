#!/usr/bin/env python

import math
import time
import struct
import pyaudio
import numpy as np
import matplotlib.pyplot as plt

CHUNK   = 1024 # frame per buffer
RATE    = 44100 # sampling rate
UNIT_SAMPLE_TIME = 1.0 / float(RATE)
CHANNEL = 2 # number of channels
FORMAT  = pyaudio.paInt16
MAX_INT = 32768.0

p = pyaudio.PyAudio()

def find_input_device():
    device_index = None
    for i in range(p.get_device_count()):
        devinfo = p.get_device_info_by_index(i)
        print 'Device %d: %s'%(i, devinfo['name'])

        for keyword in ['mic', 'input']:
            if keyword in devinfo['name'].lower():
                print 'Found an input: device %d - %s' % (i, devinfo['name'])
                device_index = i
                return device_index

    if device_index is None:
        print 'No preferred input found; using default input device.'

    return device_index

def get_rms(block):
    # RMS amplitude is defined as the square root of the
    # mean over time of the square of the amplitude.
    # so we need to convert this string of bytes into
    # a string of 16-bit samples...

    # we will get one short out for each
    # two chars in the string.
    count = len(block)/2
    structFormat = '%dh' % count
    shorts = struct.unpack(structFormat, block)

    # iterate over the block.
    sum_squares = 0.0
    for sample in shorts:
        # sample is a signed short in +/- 32768.
        # normalize it to 1.0
        n = sample / MAX_INT
        sum_squares += n*n

    return math.sqrt(sum_squares / count)


deviceIndex = find_input_device()
print 'Audio device:', deviceIndex
print 'Sample rate:', p.get_device_info_by_index(0)['defaultSampleRate']

stream = p.open(format=FORMAT, channels=CHANNEL, rate=RATE, input=True, frames_per_buffer=CHUNK, input_device_index=deviceIndex)
# stream.start_stream()
start = time.time()

times = []
audioAmp = []
def log():
    data = stream.read(CHUNK)
    times.append(time.time() - start)
    amplitude = get_rms(data)
    audioAmp.append(amplitude)

while time.time() - start < 5:
    log()

print 'Start time:', start, 'end time:', time.time(), 'duration:', time.time() - start
print np.shape(times), np.shape(audioAmp)
print audioAmp[:5]

plt.plot(times, audioAmp)
plt.show()
