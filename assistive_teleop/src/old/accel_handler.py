#!/usr/bin/python

import roslib; roslib.load_manifest('web_teleop_trunk')
import rospy
import math
from collections import deque
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
from geometry_msgs.msg import Vector3
from pr2_msgs.msg import AccelerometerState
from std_msgs.msg import Float32


class AccelerometerListener():
    def __init__(self):
        rospy.init_node('accelerometer_listener')
        rospy.Subscriber('accelerometer/r_gripper_motor', AccelerometerState, self.state_filter)
        #rospy.Subscriber('accelerometer/r_gripper_motor', AccelerometerState, self.mag_filter)
        self.acc_out = rospy.Publisher('accelerometer_filtered', Vector3)
        #self.mag_out = rospy.Publisher('acc_mag', Float32)


        self.deq_max = 50
        self.x = deque([], self.deq_max)
        self.y = deque([], self.deq_max)
        self.z = deque([], self.deq_max)
        self.sm_x = deque([], self.deq_max)
        self.sm_y = deque([], self.deq_max)
        self.sm_z = deque([], self.deq_max)
        self.count = 0
        #self.mag = deque([], self.deq_max)

    #def mag_filter(self, acc_state):
        #x = []; y = []; z = [];
        #for i in range(len(acc_state.samples)):
            #x.append(acc_state.samples[i].x)
            #y.append(acc_state.samples[i].y)
            #z.append(acc_state.samples[i].z)
#
        #self.x.append(np.average(x))
        #self.y.append(np.average(y))
        #self.z.append(np.average(z))
        #self.mag.append(self.x[0] + self.y[0] + self.z[0]+9.80665)
        #mag = Float32()
        #mag.data = self.mag[0]
        #self.mag_out.publish(mag)
        #print self.mag[0]

    def state_filter(self, acc_state):
        x = []; y = []; z = [];
        for i in range(len(acc_state.samples)):
            x.append(acc_state.samples[i].x)
            y.append(acc_state.samples[i].y)
            z.append(acc_state.samples[i].z)

        self.x.append(np.average(x))
        self.y.append(np.average(y))
        self.z.append(np.average(z))

        self.sm_x.append(np.average(self.x))
        self.sm_y.append(np.average(self.y))
        self.sm_z.append(np.average(self.z))

        state = Vector3()
        #state.x = self.sm_x[0]
        #state.y = self.sm_y[0]
        #state.z = self.sm_z[0]
        #self.acc_out.publish(state)
        if len(self.x) == self.deq_max:
            FTx = np.fft.rfft(self.x)
            #print "FFT: %s" %FTx
            FTx = FTx[1:]
           # print "Trunc: %s" %FTx
            state.x = np.fft.irfft(FTx)[0]
            #print "Inv: %s" %IFFTx
            FTy = np.fft.rfft(self.y)
            FTy = FTy[1:]
            state.y = np.fft.irfft(FTy)[0]
            FTz = np.fft.rfft(self.z)
            FTz = FTz[1:]
            state.z = np.fft.irfft(FTz)[0]
            self.acc_out.publish(state)
            
            if abs(state.x) > 3 or abs(state.y) > 3 or abs(state.z) > 3:
                self.count += 1
                print "COLLISION %s DETECTED!" %self.count



if __name__ == '__main__':
    al = AccelerometerListener()

    while not rospy.is_shutdown():
        rospy.spin()
