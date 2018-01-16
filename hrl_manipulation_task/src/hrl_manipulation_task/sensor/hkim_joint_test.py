#!/usr/bin/env python
#test client for joint_states_listener

import numpy as np
import matplotlib.pyplot as plt
import roslib
import rospy
import tf
import tf.transformations as tft
import threading
import time
import sys
import hrl_lib.circular_buffer as cb
from geometry_msgs.msg import Point32, Quaternion, PoseStamped, WrenchStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import String

FT_INI_THRESH = -6.0
FT_MAX_THRESH = -1.0

class JointStateListener(object):
    def __init__(self, plot_en=False):
        self.plot_en = plot_en
        self.ft_lock = threading.RLock()
        self.force = None
        self.torque = None

        self.f = {}
        if self.plot_en:
            self.f['x'] = []
            self.f['y'] = []
            self.f['z'] = []        
            self.f['tx'] = []
            self.f['ty'] = []
            self.f['tz'] = []
        self.f_buf = cb.CircularBuffer(10, (1,))
        self.wiping = False
        self.detect_stop = False
        self.wipe_finished = False
        self.steady_detector = SteadyStateDetector(30, (2,), 5, mode='std monitor', overlap=-1)

        #Subscriber
        self.ft_sub     = rospy.Subscriber('/ft/l_gripper_motor', WrenchStamped, self.ft_callback)
        self.status_sub = rospy.Subscriber('/manipulation_task/proceed', String, self.status_callback) 

        #Publisher
        self.stop_motion_pub = rospy.Publisher('/manipulation_task/InterruptAction', String, queue_size = 10)
        self.emergency_pub = rospy.Publisher('/manipulation_task/emergency', String, queue_size = 10)
        
        self.run()

    def ft_callback(self, data):
        with self.ft_lock:
            self.force = data.wrench.force
            self.torque = data.wrench.torque

    def status_callback(self, data):
        if data.data == "Set: Wiping 2, Wiping 3, Wipe":
            self.detect_stop = True
            plt.close()
        else:
            self.detect_stop = False
        if data.data == "Set: Wiping 3, Wipe, Retract":
            self.wiping = True
            plt.close()
        else:
            if self.wiping and self.plot_en:
                self.wipe_finished = True
            self.wiping = False


    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            with self.ft_lock:
                force = self.force
                torque = self.torque
            if force is None:
                rate.sleep()
                continue
            if force.y >= FT_INI_THRESH and self.detect_stop:
                self.stop_motion_pub.publish("found good thresh")
            if force.y >= FT_MAX_THRESH and self.wiping:
                self.emergency_pub.publish("STOP")
                self.wiping = False
                self.wipe_finished = True
            elif self.wiping or self.detect_stop:
                if self.plot_en:
                    self.f['x'].append(force.x)
                    self.f['y'].append(force.y)
                    self.f['z'].append(force.z)
                    self.f['tx'].append(torque.x)
                    self.f['ty'].append(torque.y)
                    self.f['tz'].append(torque.z)
                if self.wiping:
                    self.f_buf.append(force.y)
                if len(self.f_buf) >= 10:
                    par = np.polyfit(xrange(10), self.f_buf, 1)
                    if par[0][0] >= .5:
                        if self.wiping:
                            self.f_buf = cb.CircularBuffer(10, (1,))
                            print "force spike detected for stop"
                            self.emergency_pub.publish("STOP")
                            self.wiping = False
                            self.wipe_finished = True
            elif self.wipe_finished:
                self.wiping = False
                self.detect_stop = False
                self.wipe_finished = False
                self.f_buf = cb.CircularBuffer(10, (1,))
                if self.plot_en:
                    plt.subplot(321)
                    plt.plot(self.f['x'], 'r')
                    plt.subplot(322)
                    plt.plot(self.f['y'], 'b')
                    plt.subplot(323)
                    plt.plot(self.f['z'], 'y')
                    plt.subplot(324)
                    plt.plot(self.f['tx'], 'r--')
                    plt.subplot(325)
                    plt.plot(self.f['ty'], 'b--')
                    plt.subplot(326)
                    plt.plot(self.f['tz'], 'y--')
                    plt.show()
                    self.f['x'] = []
                    self.f['y'] = []
                    self.f['z'] = []
                    self.f['tx'] = []
                    self.f['ty'] = []
                    self.f['tz'] = []

            rate.sleep()

        

            

if __name__ == "__main__":
    rospy.init_node('force_listener')
    ForceListener(plot_en=True)
        
            
