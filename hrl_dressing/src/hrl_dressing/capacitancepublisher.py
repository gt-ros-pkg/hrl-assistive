#!/usr/bin/env python

import serial, time, gc
import numpy as np

import rospy
from std_msgs.msg import Float64MultiArray

'''
Run on machine with Teensy connected and master set to PR2
'''

def setupSerial(devName, baudrate):
    serialDev = serial.Serial(port=devName, baudrate=baudrate, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE)
    if serialDev is None:
        raise RuntimeError('[%s]: Serial port %s not found!\n' % (rospy.get_name(), devDame))

    serialDev.write_timeout = .05
    serialDev.timeout = None
    serialDev.flushOutput()
    serialDev.flushInput()
    return serialDev

def getData(serialDev, numOutputs):
    numAttempts = 4
    for i in xrange(numAttempts):
        try:
            line = serialDev.readline()
            try:
                values = map(float, line.split(','))
                if len(values) == numOutputs:
                    return values
                else:
                    print 'Read values did not matched expected outputs:', line
            except ValueError:
                print 'Received suspect data: %s from socket.' % line
        except:
            print 'Unable to read line. Recommended to setup serial again.'
        serialDev.flush()
    return []

if __name__ == '__main__':
    # Setup serial port and ROS publisher
    baudrate = 115200
    # TODO: This address might be wrong, especially if you are using Windows
    serialTeensy = setupSerial('/dev/ttyACM0', baudrate)
    pub = rospy.Publisher('/capDressing/capacitance', Float64MultiArray, queue_size=10000)
    rospy.init_node('capacitancepublisher')

    # Read a few lines to get things rolling
    for _ in range(25):
        serialTeensy.readline()

    print 'Started publishing data'
    data = []
    times = []

    # t = rospy.get_time()
    t = time.time()
    i = 0
    while not rospy.is_shutdown():
        fa = Float64MultiArray()
        fa.data = getData(serialTeensy, 1)
        pub.publish(fa)
        # rospy.sleep(0.0001)

        # Measure frequency in Hz
        # i += 1
        # if time.time() - t > 5:
        #     print '%.2f Hz' % (i / 5.0)
        #     exit()

    serialTeensy.close()
