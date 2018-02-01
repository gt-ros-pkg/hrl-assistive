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
from sound_play.libsoundplay import SoundClient
from sensor_msgs.msg import JointState
from std_msgs.msg import String

FT_INI_THRESH = -7.5
FT_MAX_THRESH = -2.0

class ForceListener(object):
    def __init__(self, plot_en=False):
        self.sound_handle = SoundClient()
        
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
        self.cb_len = 20
        self.f_buf = cb.CircularBuffer(self.cb_len, (1,))
        self.wiping = False
        self.detect_stop = False
        self.wipe_finished = False
        self.coll_f = FT_MAX_THRESH
        self.coll_f_add = 5.5

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
            self.coll_f = FT_MAX_THRESH

    def run(self):
        rate = rospy.Rate(60)
        while not rospy.is_shutdown():
            with self.ft_lock:
                force = self.force
                torque = self.torque
            if force is None:
                rate.sleep()
                continue
            if force.y >= FT_INI_THRESH and self.detect_stop:
                print "first collision ", force.y
                self.coll_f = force.y + self.coll_f_add
                self.stop_motion_pub.publish("found good thresh")
                self.f_buf = cb.CircularBuffer(self.cb_len, (1,))
            if (force.y >= FT_MAX_THRESH or force.y >= self.coll_f) and self.wiping:
                print "too much force ", force.y  
                self.sound_handle.say('Force detect ed')
                self.emergency_pub.publish("STOP")
                self.wiping = False
                self.wipe_finished = True
                self.coll_f = FT_MAX_THRESH
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
                    par = np.polyfit(xrange(len(self.f_buf)), self.f_buf, 1)
                    print force.y, par[0][0]
                if len(self.f_buf) >= self.cb_len:
                    par = np.polyfit(xrange(self.cb_len), self.f_buf, 1)
                    if par[0][0] >= .2:
                        if self.wiping or self.detect_stop:
                            self.f_buf = cb.CircularBuffer(self.cb_len, (1,))
                            self.sound_handle.say('Force detect ed')
                            if self.detect_stop:
                                print "force spike detected for pause"
                                self.coll_f = force.y + self.coll_f_add
                                self.stop_motion_pub.publish("found good thresh")
                            else:
                                print "force spike detected for stop"
                                self.emergency_pub.publish("STOP")
                                self.coll_f = FT_MAX_THRESH
                                self.wiping = False
                                self.wipe_finished = True
            elif self.wipe_finished:
                self.wiping = False
                self.detect_stop = False
                self.wipe_finished = False
                self.f_buf = cb.CircularBuffer(self.cb_len, (1,))
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
    ForceListener(plot_en=False)
        
            
