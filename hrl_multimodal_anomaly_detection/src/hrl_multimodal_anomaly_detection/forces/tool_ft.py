#!/usr/bin/env python

import rospy
import numpy as np
from threading import Thread
from geometry_msgs.msg import WrenchStamped

class tool_ft(Thread):
    def __init__(self,ft_sensor_topic_name):
        super(tool_ft, self).__init__()
        self.daemon = True
        self.cancelled = False

        self.init_time = 0.
        self.counter = 0
        self.counter_prev = 0
        self.force = np.matrix([0.,0.,0.]).T
        self.force_raw = np.matrix([0.,0.,0.]).T
        self.torque = np.matrix([0.,0.,0.]).T
        self.torque_raw = np.matrix([0.,0.,0.]).T
        self.torque_bias = np.matrix([0.,0.,0.]).T

        self.time_data = []
        self.force_data = []
        self.force_raw_data = []
        self.torque_data = []
        self.torque_raw_data = []

        # capture the force on the tool tip
        # self.force_sub = rospy.Subscriber(ft_sensor_topic_name, WrenchStamped, self.force_cb)
        # raw ft values from the NetFT
        self.force_raw_sub = rospy.Subscriber(ft_sensor_topic_name, WrenchStamped, self.force_raw_cb)
        # self.force_zero = rospy.Publisher('/tool_netft_zeroer/rezero_wrench', Bool)
        rospy.logout('Done subscribing to '+ft_sensor_topic_name+' topic')


    def force_cb(self, msg):
        self.force = np.matrix([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z]).T
        self.torque = np.matrix([msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z]).T


    def force_raw_cb(self, msg):
        # self.time = msg.header.stamp.to_time()
        self.force_raw = np.matrix([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z]).T
        self.torque_raw = np.matrix([msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z]).T
        self.counter += 1


    def reset(self):
        ## self.force_zero.publish(Bool(True))
        pass

    def run(self):
        """Overloaded Thread.run, runs the update
        method once per every xx milliseconds."""

        rate = rospy.Rate(1000) # 25Hz, nominally.
        while not self.cancelled:
            self.log()
            rate.sleep()


    def log(self):
        if self.counter > self.counter_prev:
            self.counter_prev = self.counter

            ## self.force_data.append(self.force)
            self.force_raw_data.append(self.force_raw)
            ## self.torque_data.append(self.torque)
            self.torque_raw_data.append(self.torque_raw)
            self.time_data.append(rospy.get_time()-self.init_time)



    def cancel(self):
        """End this timer thread"""
        self.cancelled = True
        self.force_raw_sub.unregister()
        rospy.sleep(0.25)


    ## def static_bias(self):
    ##     print '!!!!!!!!!!!!!!!!!!!!'
    ##     print 'BIASING FT'
    ##     print '!!!!!!!!!!!!!!!!!!!!'
    ##     f_list = []
    ##     t_list = []
    ##     for i in range(20):
    ##         f_list.append(self.force)
    ##         t_list.append(self.torque)
    ##         rospy.sleep(2/100.)
    ##     if f_list[0] != None and t_list[0] !=None:
    ##         self.force_bias = np.mean(np.column_stack(f_list),1)
    ##         self.torque_bias = np.mean(np.column_stack(t_list),1)
    ##         print self.gravity
    ##         print '!!!!!!!!!!!!!!!!!!!!'
    ##         print 'DONE Biasing ft'
    ##         print '!!!!!!!!!!!!!!!!!!!!'
    ##     else:
    ##         print 'Biasing Failed!'

