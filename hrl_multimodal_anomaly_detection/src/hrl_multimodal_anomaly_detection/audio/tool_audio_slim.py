#!/usr/bin/env python

import rospy
import pyaudio
from threading import Thread

class tool_audio_slim(Thread):
    MAX_INT = 32768.0
    CHUNK   = 4096 #frame per buffer
    RATE    = 44100 #sampling rate
    UNIT_SAMPLE_TIME = 1.0 / float(RATE)
    CHANNEL = 2 #number of channels
    FORMAT  = pyaudio.paInt16

    def __init__(self):
        super(tool_audio_slim, self).__init__()
        self.daemon = True
        self.cancelled = False

        self.init_time = 0.0

        self.audio_data_raw = []
        self.time_data = []

        self.p = pyaudio.PyAudio()
        deviceIndex = self.find_input_device()
        print 'Audio device:', deviceIndex
        print 'Sample rate:', self.p.get_device_info_by_index(0)['defaultSampleRate']

        self.stream = self.p.open(format=self.FORMAT, channels=self.CHANNEL, rate=self.RATE, input=True, frames_per_buffer=self.CHUNK, input_device_index=deviceIndex)
        # rospy.logout('Done subscribing audio')
        # print 'Done subscribing audio'

    def find_input_device(self):
        device_index = None
        for i in range(self.p.get_device_count()):
            devinfo = self.p.get_device_info_by_index(i)
            print('Device %d: %s'%(i, devinfo['name']))

            for keyword in ['mic', 'input']:
                if keyword in devinfo['name'].lower():
                    print('Found an input: device %d - %s'%(i, devinfo['name']))
                    device_index = i
                    return device_index

        if device_index is None:
            print('No preferred input found; using default input device.')

        return device_index

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

    def cancel(self):
        """End this timer thread"""
        self.cancelled = True
        rospy.sleep(1.0)

        self.stream.stop_stream()
        self.stream.close()

    def reset(self):
        pass
