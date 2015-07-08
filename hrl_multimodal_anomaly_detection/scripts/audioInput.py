#!/usr/bin/env python

import pyaudio

p = pyaudio.PyAudio()

def find_input_device():
    device_index = None
    for i in range(p.get_device_count()):
        devinfo = p.get_device_info_by_index(i)
        print('Device %d: %s'%(i, devinfo['name']))

        for keyword in ['mic', 'input']:
            if keyword in devinfo['name'].lower():
                print('Found an input: device %d - %s'%(i, devinfo['name']))
                device_index = i
                return device_index

    if device_index is None:
        print('No preferred input found; using default input device.')

    return device_index

print 'Audio device:', find_input_device()
